//! FP16 vector pages and top-k search.

pub mod f16;
mod search;
pub use search::{SearchProgress, SearchStats, TopK};
#[doc(hidden)]
pub mod storage;

use arc_swap::{ArcSwap, ArcSwapOption};
use crossbeam_queue::SegQueue;
use half::f16 as F16;
use rayon::prelude::*;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    error::Error,
    fmt,
    hash::Hash,
    io,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

pub const SEGMENT_DIMENSION: usize = 64;
pub const DEFAULT_VECTOR_DIMENSION: usize = 384;
pub const DEFAULT_SEGMENT_NUM: usize = DEFAULT_VECTOR_DIMENSION / SEGMENT_DIMENSION;
pub const PAGE_SIZE: usize = u64::BITS as usize;
pub const DEFAULT_FILE_CACHE_LIMIT_BYTES: usize = 256 * 1024 * 1024;

pub type SegmentVector<P> = [P; SEGMENT_DIMENSION];
pub type NormsRatioVector<const N: usize = DEFAULT_SEGMENT_NUM> = [f32; N];
pub type SegmentVectorPage<P> = [SegmentVector<P>; PAGE_SIZE];
pub type NormsRatioVectorPage<const N: usize = DEFAULT_SEGMENT_NUM> =
    [NormsRatioVector<N>; PAGE_SIZE];

#[repr(align(64))]
#[derive(Clone)]
pub struct AlignedSegmentPage<P>(pub SegmentVectorPage<P>);

pub enum PageSegments<P, const N: usize = DEFAULT_SEGMENT_NUM> {
    OnMemory([Box<AlignedSegmentPage<P>>; N]),
    FileBacked {
        file_offset: u64,
        cache: [Arc<ArcSwapOption<AlignedSegmentPage<P>>>; N],
    },
}

pub struct VectorPage<K, P, const N: usize = DEFAULT_SEGMENT_NUM> {
    segments: PageSegments<P, N>,
    /// Each original segment norm divided by the original whole-vector norm.
    segment_norm_ratios: NormsRatioVectorPage<N>,
    valid_bitmap: u64,
    key_table: [Option<K>; PAGE_SIZE],
    generation: u64,
    checksums: [u32; N],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VecIndex(pub u64);

impl VecIndex {
    pub fn page_index(&self) -> usize {
        (self.0 >> 6) as usize
    }
    pub fn offset(&self) -> usize {
        (self.0 & 0x3f) as usize
    }
    pub fn new(page_index: usize, offset: usize) -> Self {
        assert!((page_index as u64) < (1_u64 << 58));
        assert!(offset < PAGE_SIZE);
        Self(((page_index as u64) << 6) | offset as u64)
    }
}

/// An error from vector construction, storage, or recovery.
#[derive(Debug)]
pub enum IndexError {
    Io(io::Error),
    InvalidInput(String),
    InvalidFormat(String),
    /// The commit record may have reached disk; reopen to determine its outcome.
    CommitOutcomeUnknown(io::Error),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::InvalidFormat(message) => write!(f, "invalid index format: {message}"),
            Self::CommitOutcomeUnknown(e) => write!(f, "commit outcome unknown; reopen index: {e}"),
        }
    }
}
impl Error for IndexError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) | Self::CommitOutcomeUnknown(e) => Some(e),
            _ => None,
        }
    }
}
impl From<io::Error> for IndexError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

struct KeyEntry {
    index: Mutex<Option<VecIndex>>,
}

type KeyRemoval<'a, K> = (&'a K, Arc<KeyEntry>, VecIndex);

struct PageSlot<K, P, const N: usize> {
    snapshot: ArcSwap<VectorPage<K, P, N>>,
}

pub struct VectorIndexCore<K, P, const N: usize = DEFAULT_SEGMENT_NUM> {
    pages: boxcar::Vec<PageSlot<K, P, N>>,
    free_queue: SegQueue<VecIndex>,
    mutation: Mutex<()>,
    keys: Mutex<HashMap<K, Arc<KeyEntry>>>,
    len: AtomicUsize,
}

struct PendingPage<K, const N: usize> {
    page_index: usize,
    page: VectorPage<K, F16, N>,
    file_segments: Option<[Box<AlignedSegmentPage<F16>>; N]>,
}

impl<K, P, const N: usize> PageSlot<K, P, N> {
    fn new(page: VectorPage<K, P, N>) -> Self {
        Self {
            snapshot: ArcSwap::from_pointee(page),
        }
    }
}

/// Current FileBacked segment cache occupancy and configured limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub limit_bytes: usize,
    pub resident_bytes: usize,
    pub resident_segments: usize,
}

struct CacheState<P> {
    limit_bytes: usize,
    resident_bytes: usize,
    resident_segments: usize,
    // Keep slots alive until eviction so dropping old pages cannot desync accounting.
    slots: VecDeque<Arc<ArcSwapOption<AlignedSegmentPage<P>>>>,
}

#[doc(hidden)]
pub struct SegmentCache<P>(Mutex<CacheState<P>>);

impl<P> SegmentCache<P> {
    fn new(limit_bytes: usize) -> Self {
        Self(Mutex::new(CacheState {
            limit_bytes,
            resident_bytes: 0,
            resident_segments: 0,
            slots: VecDeque::new(),
        }))
    }

    fn publish(
        &self,
        slot: &Arc<ArcSwapOption<AlignedSegmentPage<P>>>,
        data: Arc<AlignedSegmentPage<P>>,
    ) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if slot.load_full().is_some() {
            return;
        }
        slot.store(Some(data));
        state.resident_bytes = state
            .resident_bytes
            .saturating_add(std::mem::size_of::<AlignedSegmentPage<P>>());
        state.resident_segments += 1;
        state.slots.push_back(Arc::clone(slot));
        Self::evict(&mut state);
    }

    fn set_limit(&self, limit_bytes: usize) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.limit_bytes = limit_bytes;
        Self::evict(&mut state);
    }

    fn clear(&self) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        while let Some(slot) = state.slots.pop_front() {
            if slot.swap(None).is_some() {
                state.resident_bytes = state
                    .resident_bytes
                    .saturating_sub(std::mem::size_of::<AlignedSegmentPage<P>>());
                state.resident_segments = state.resident_segments.saturating_sub(1);
            }
        }
    }

    fn stats(&self) -> CacheStats {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        CacheStats {
            limit_bytes: state.limit_bytes,
            resident_bytes: state.resident_bytes,
            resident_segments: state.resident_segments,
        }
    }

    fn evict(state: &mut CacheState<P>) {
        let segment_bytes = std::mem::size_of::<AlignedSegmentPage<P>>();
        while state.resident_bytes > state.limit_bytes {
            let Some(slot) = state.slots.pop_front() else {
                break;
            };
            if slot.swap(None).is_some() {
                state.resident_bytes = state.resident_bytes.saturating_sub(segment_bytes);
                state.resident_segments = state.resident_segments.saturating_sub(1);
            }
        }
    }
}

impl<K, P, const N: usize> VectorIndexCore<K, P, N> {
    fn new() -> Self {
        Self {
            pages: boxcar::Vec::new(),
            free_queue: SegQueue::new(),
            mutation: Mutex::new(()),
            keys: Mutex::new(HashMap::new()),
            len: AtomicUsize::new(0),
        }
    }

    fn key_entry(&self, key: K) -> Arc<KeyEntry>
    where
        K: Eq + Hash,
    {
        let mut keys = self.keys.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(keys.entry(key).or_insert_with(|| {
            Arc::new(KeyEntry {
                index: Mutex::new(None),
            })
        }))
    }

    fn remove_idle_key_entry(&self, key: &K, entry: &Arc<KeyEntry>)
    where
        K: Eq + Hash,
    {
        let mut keys = self.keys.lock().unwrap_or_else(|e| e.into_inner());
        if Arc::strong_count(entry) == 2
            && entry
                .index
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_none()
            && keys
                .get(key)
                .is_some_and(|current| Arc::ptr_eq(current, entry))
        {
            keys.remove(key);
        }
    }
}

impl<K, const N: usize> VectorIndexCore<K, F16, N> {
    fn allocate_slot(&self, file_backed: bool) -> VecIndex {
        if let Some(index) = self.free_queue.pop() {
            return index;
        }
        let page_index = self.pages.push(PageSlot::new(empty_page(file_backed)));
        for slot in 1..PAGE_SIZE {
            self.free_queue.push(VecIndex::new(page_index, slot));
        }
        VecIndex::new(page_index, 0)
    }
}

pub enum VectorIndex<K, P, const N: usize = DEFAULT_SEGMENT_NUM> {
    OnMemory(VectorIndexCore<K, P, N>),
    FileBacked {
        core: VectorIndexCore<K, P, N>,
        storage: storage::Storage<K, N>,
        cache: SegmentCache<F16>,
    },
}

pub struct NormalizedSegmentedDocumentVector<K, P, const N: usize = DEFAULT_SEGMENT_NUM> {
    key: K,
    segments: [SegmentVector<P>; N],
    segment_norm_ratios: NormsRatioVector<N>,
}

/// FP16 index with 384 dimensions by default.
pub type F16Index<K, const N: usize = DEFAULT_SEGMENT_NUM> = VectorIndex<K, F16, N>;
/// A segment-normalized, FP16-quantized document vector.
pub type F16Vector<K, const N: usize = DEFAULT_SEGMENT_NUM> =
    NormalizedSegmentedDocumentVector<K, F16, N>;

impl<K, const N: usize> NormalizedSegmentedDocumentVector<K, F16, N> {
    /// Normalize 64-dimensional segments, quantize them to FP16, and retain norm ratios.
    pub fn from_f32(key: K, input: &[f32]) -> Result<Self, IndexError> {
        let expected = N
            .checked_mul(SEGMENT_DIMENSION)
            .filter(|_| N != 0)
            .ok_or_else(|| IndexError::InvalidInput("invalid segment count".into()))?;
        if input.len() != expected {
            return Err(IndexError::InvalidInput(format!(
                "expected {expected} coordinates"
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(IndexError::InvalidInput(
                "coordinates must be finite".into(),
            ));
        }
        // f64 prevents overflow or underflow when calculating norms of finite f32 inputs.
        let lengths: [f64; N] = std::array::from_fn(|i| {
            input[i * SEGMENT_DIMENSION..(i + 1) * SEGMENT_DIMENSION]
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt()
        });
        let whole = lengths.iter().map(|x| x * x).sum::<f64>().sqrt();
        let segment_norm_ratios = std::array::from_fn(|i| {
            if whole == 0.0 {
                0.0
            } else {
                (lengths[i] / whole) as f32
            }
        });
        let segments: [SegmentVector<F16>; N] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if lengths[i] == 0.0 {
                    F16::ZERO
                } else {
                    F16::from_f32((f64::from(input[i * SEGMENT_DIMENSION + j]) / lengths[i]) as f32)
                }
            })
        });
        Ok(Self {
            key,
            segments,
            segment_norm_ratios,
        })
    }
}

pub struct Hit<K> {
    pub key: K,
    pub score: f32,
}

pub trait NormRatios<const N: usize = DEFAULT_SEGMENT_NUM> {
    fn norm_ratios(&self) -> &NormsRatioVector<N>;
}
pub trait SegmentedVector<P, const N: usize = DEFAULT_SEGMENT_NUM> {
    fn segment(&self, index: usize) -> &SegmentVector<P>;
}
impl<K, P, const N: usize> NormRatios<N> for NormalizedSegmentedDocumentVector<K, P, N> {
    fn norm_ratios(&self) -> &NormsRatioVector<N> {
        &self.segment_norm_ratios
    }
}
impl<K, P, const N: usize> SegmentedVector<P, N> for NormalizedSegmentedDocumentVector<K, P, N> {
    fn segment(&self, index: usize) -> &SegmentVector<P> {
        &self.segments[index]
    }
}

pub trait SegmentDotKernel<P> {
    fn dot(&self, a: &SegmentVector<P>, b: &SegmentVector<P>) -> f32;
    fn dot_page(
        &self,
        query: &SegmentVector<P>,
        page: &AlignedSegmentPage<P>,
        candidates: u64,
        scores: &mut [f32; PAGE_SIZE],
    );
}

pub trait VectorIndexOps<K, P, const N: usize = DEFAULT_SEGMENT_NUM> {
    fn insert(&self, vector: NormalizedSegmentedDocumentVector<K, P, N>) -> Result<(), IndexError>;
    fn remove(&self, key: &K) -> Result<(), IndexError>;
    fn search(
        &self,
        query: &NormalizedSegmentedDocumentVector<K, P, N>,
        top_k: usize,
    ) -> Result<Vec<(K, f32)>, IndexError>;
}

fn empty_segments<const N: usize>() -> [Box<AlignedSegmentPage<F16>>; N] {
    std::array::from_fn(|_| {
        Box::new(AlignedSegmentPage(
            [[F16::ZERO; SEGMENT_DIMENSION]; PAGE_SIZE],
        ))
    })
}

fn empty_page<K, const N: usize>(file_backed: bool) -> VectorPage<K, F16, N> {
    VectorPage {
        segments: if file_backed {
            PageSegments::FileBacked {
                file_offset: 0,
                cache: std::array::from_fn(|_| Arc::new(ArcSwapOption::empty())),
            }
        } else {
            PageSegments::OnMemory(empty_segments())
        },
        segment_norm_ratios: [[0.0; N]; PAGE_SIZE],
        valid_bitmap: 0,
        key_table: std::array::from_fn(|_| None),
        generation: 0,
        checksums: [0; N],
    }
}

impl<K, const N: usize> VectorIndex<K, F16, N> {
    pub fn on_memory() -> Self {
        Self::OnMemory(VectorIndexCore::new())
    }

    /// Set the maximum bytes retained by the FileBacked segment cache.
    ///
    /// This has no effect on OnMemory indexes. It counts cached segment payloads. Active searches can temporarily keep
    /// evicted payloads alive through their `Arc` snapshots. A zero limit disables
    /// retention.
    pub fn set_cache_limit_bytes(&self, limit_bytes: usize) {
        if let Self::FileBacked { cache, .. } = self {
            cache.set_limit(limit_bytes);
        }
    }

    /// Return cached payload occupancy and limit; OnMemory indexes report zero values.
    pub fn cache_stats(&self) -> CacheStats {
        match self {
            Self::FileBacked { cache, .. } => cache.stats(),
            Self::OnMemory(_) => CacheStats {
                limit_bytes: 0,
                resident_bytes: 0,
                resident_segments: 0,
            },
        }
    }

    fn core(&self) -> &VectorIndexCore<K, F16, N> {
        match self {
            Self::OnMemory(core) | Self::FileBacked { core, .. } => core,
        }
    }

    /// Return keys observed as live during this call.
    pub fn keys(&self) -> Vec<K>
    where
        K: Clone,
    {
        let entries: Vec<_> = self
            .core()
            .keys
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .map(|(key, entry)| (key.clone(), Arc::clone(entry)))
            .collect();
        entries
            .into_iter()
            .filter_map(|(key, entry)| {
                entry
                    .index
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .is_some()
                    .then_some(key)
            })
            .collect()
    }

    /// Return a stored vector by key, loading its FileBacked segments as needed.
    pub fn vector_by_key(&self, key: &K) -> Result<F16Vector<K, N>, IndexError>
    where
        K: Clone + Eq + Hash,
    {
        let core = self.core();
        let entry = core
            .keys
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(key)
            .cloned()
            .ok_or_else(|| IndexError::InvalidInput("vector key not found".into()))?;
        let entry_index = entry
            .index
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let index = entry_index
            .as_ref()
            .copied()
            .ok_or_else(|| IndexError::InvalidInput("vector key not found".into()))?;
        let slot = index.offset();
        let page = core
            .pages
            .get(index.page_index())
            .ok_or_else(|| IndexError::InvalidFormat("key references a missing page".into()))?
            .snapshot
            .load_full();
        if page.valid_bitmap & (1_u64 << slot) == 0 || page.key_table[slot].as_ref() != Some(key) {
            return Err(IndexError::InvalidInput("vector key not found".into()));
        }

        let mut segments = Vec::with_capacity(N);
        for segment in 0..N {
            segments.push(self.with_segment(&page, segment, |data| data.0[slot])?);
        }
        let segments: [SegmentVector<F16>; N] = segments
            .try_into()
            .map_err(|_| IndexError::InvalidFormat("segment count mismatch".into()))?;
        Ok(NormalizedSegmentedDocumentVector {
            key: key.clone(),
            segments,
            segment_norm_ratios: page.segment_norm_ratios[slot],
        })
    }

    fn with_segment<R>(
        &self,
        page: &VectorPage<K, F16, N>,
        segment: usize,
        f: impl FnOnce(&AlignedSegmentPage<F16>) -> R,
    ) -> Result<R, IndexError> {
        match &page.segments {
            PageSegments::OnMemory(segments) => Ok(f(&segments[segment])),
            PageSegments::FileBacked { file_offset, cache } => {
                let data = match cache[segment].load_full() {
                    Some(data) => data,
                    None => {
                        let Self::FileBacked {
                            storage,
                            cache: manager,
                            ..
                        } = self
                        else {
                            return Err(IndexError::InvalidFormat(
                                "file page in memory index".into(),
                            ));
                        };
                        let data = Arc::new(storage.read_segment(
                            *file_offset,
                            segment,
                            page.checksums[segment],
                        )?);
                        manager.publish(&cache[segment], Arc::clone(&data));
                        data
                    }
                };
                Ok(f(&data))
            }
        }
    }
}

impl<K: Clone + Eq + Hash + Serialize + DeserializeOwned, const N: usize> VectorIndex<K, F16, N> {
    /// Create a new FileBacked index with a 256 MiB segment cache. Existing files are not overwritten.
    pub fn create_file_backed(path: impl AsRef<Path>) -> Result<Self, IndexError> {
        Self::create_file_backed_with_cache_limit(path, DEFAULT_FILE_CACHE_LIMIT_BYTES)
    }

    /// Create a FileBacked index with a segment cache byte limit.
    pub fn create_file_backed_with_cache_limit(
        path: impl AsRef<Path>,
        limit_bytes: usize,
    ) -> Result<Self, IndexError> {
        if N == 0 || N.checked_mul(SEGMENT_DIMENSION).is_none() {
            return Err(IndexError::InvalidInput("invalid segment count".into()));
        }
        Ok(Self::FileBacked {
            core: VectorIndexCore::new(),
            storage: storage::Storage::create(path)?,
            cache: SegmentCache::new(limit_bytes),
        })
    }

    /// Open an index with a 256 MiB segment cache, restoring metadata without loading payloads.
    pub fn open_file_backed(path: impl AsRef<Path>) -> Result<Self, IndexError> {
        Self::open_file_backed_with_cache_limit(path, DEFAULT_FILE_CACHE_LIMIT_BYTES)
    }

    /// Open an index with a segment cache byte limit.
    pub fn open_file_backed_with_cache_limit(
        path: impl AsRef<Path>,
        limit_bytes: usize,
    ) -> Result<Self, IndexError> {
        let (storage, mut pages) = storage::Storage::<K, N>::open(path)?;
        pages.sort_by_key(|page| page.page_index);
        let core = VectorIndexCore::new();
        {
            let mut keys = core.keys.lock().unwrap_or_else(|e| e.into_inner());
            for stored in pages {
                let index = usize::try_from(stored.page_index)
                    .map_err(|_| IndexError::InvalidFormat("page index too large".into()))?;
                if index != core.pages.count() {
                    return Err(IndexError::InvalidFormat(
                        "non-contiguous page indices".into(),
                    ));
                }
                let page = Self::from_stored(stored);
                for slot in 0..PAGE_SIZE {
                    if page.valid_bitmap & (1_u64 << slot) != 0 {
                        let key = page.key_table[slot]
                            .as_ref()
                            .ok_or_else(|| IndexError::InvalidFormat("missing key".into()))?;
                        if keys.contains_key(key) {
                            return Err(IndexError::InvalidFormat("duplicate key".into()));
                        }
                        keys.insert(
                            key.clone(),
                            Arc::new(KeyEntry {
                                index: Mutex::new(Some(VecIndex::new(index, slot))),
                            }),
                        );
                        core.len.fetch_add(1, Ordering::Relaxed);
                    } else {
                        core.free_queue.push(VecIndex::new(index, slot));
                    }
                }
                core.pages.push(PageSlot::new(page));
            }
        }
        Ok(Self::FileBacked {
            core,
            storage,
            cache: SegmentCache::new(limit_bytes),
        })
    }

    /// Rewrite the live pages into a compact file. Requires exclusive access.
    pub fn compact(&mut self) -> Result<(), IndexError> {
        match self {
            Self::OnMemory(_) => Ok(()),
            Self::FileBacked {
                core,
                storage,
                cache,
            } => {
                let pages: Vec<_> = core
                    .pages
                    .iter()
                    .map(|(index, cell)| Self::metadata(index, &cell.snapshot.load_full()))
                    .collect();
                let rewritten = storage.compact(&pages)?;
                for stored in rewritten {
                    let index = stored.page_index as usize;
                    core.pages
                        .get(index)
                        .expect("compacted page exists")
                        .snapshot
                        .store(Arc::new(Self::from_stored(stored)));
                }
                cache.clear();
                Ok(())
            }
        }
    }

    fn from_stored(stored: storage::StoredPage<K, N>) -> VectorPage<K, F16, N> {
        VectorPage {
            segments: PageSegments::FileBacked {
                file_offset: stored.segment_offset,
                cache: std::array::from_fn(|_| Arc::new(ArcSwapOption::empty())),
            },
            segment_norm_ratios: stored.segment_norm_ratios,
            valid_bitmap: stored.valid_bitmap,
            key_table: stored.keys,
            generation: stored.generation,
            checksums: stored.segment_checksums,
        }
    }

    fn metadata(page_index: usize, page: &VectorPage<K, F16, N>) -> storage::StoredPage<K, N> {
        let file_offset = match &page.segments {
            PageSegments::FileBacked { file_offset, .. } => *file_offset,
            PageSegments::OnMemory(_) => 0,
        };
        storage::StoredPage {
            page_index: page_index as u64,
            generation: page.generation,
            valid_bitmap: page.valid_bitmap,
            keys: page.key_table.clone(),
            segment_norm_ratios: page.segment_norm_ratios,
            segment_offset: file_offset,
            segment_checksums: page.checksums,
        }
    }

    fn copy_segments(
        &self,
        page: &VectorPage<K, F16, N>,
    ) -> Result<[Box<AlignedSegmentPage<F16>>; N], IndexError> {
        if page.valid_bitmap == 0 {
            return Ok(empty_segments());
        }
        let mut segments = Vec::with_capacity(N);
        for i in 0..N {
            segments.push(self.with_segment(page, i, |segment| Box::new(segment.clone()))?);
        }
        segments
            .try_into()
            .map_err(|_| IndexError::InvalidFormat("segment count mismatch".into()))
    }

    fn clone_page(&self, old: &VectorPage<K, F16, N>) -> Result<VectorPage<K, F16, N>, IndexError> {
        let segments = match &old.segments {
            PageSegments::OnMemory(_) => PageSegments::OnMemory(self.copy_segments(old)?),
            PageSegments::FileBacked { file_offset, .. } => PageSegments::FileBacked {
                file_offset: *file_offset,
                cache: std::array::from_fn(|_| Arc::new(ArcSwapOption::empty())),
            },
        };
        Ok(VectorPage {
            segments,
            segment_norm_ratios: old.segment_norm_ratios,
            valid_bitmap: old.valid_bitmap,
            key_table: old.key_table.clone(),
            generation: old
                .generation
                .checked_add(1)
                .ok_or_else(|| IndexError::InvalidInput("page generation overflow".into()))?,
            checksums: old.checksums,
        })
    }

    fn publish_pages(&self, mut pending_pages: Vec<PendingPage<K, N>>) -> Result<(), IndexError> {
        if let Self::FileBacked { storage, cache, .. } = self {
            let mut writes = pending_pages
                .iter_mut()
                .map(|pending| storage::PageWrite {
                    page: Self::metadata(pending.page_index, &pending.page),
                    segments: pending
                        .file_segments
                        .take()
                        .expect("FileBacked segment copy"),
                })
                .collect::<Vec<_>>();
            storage.commit_pages(&mut writes)?;
            for (pending, write) in pending_pages.iter_mut().zip(writes) {
                let storage::PageWrite { page, segments } = write;
                pending.page = Self::from_stored(page);
                let PageSegments::FileBacked { cache: slots, .. } = &pending.page.segments else {
                    unreachable!("FileBacked page created from stored metadata")
                };
                for (slot, segment) in slots.iter().zip(segments) {
                    cache.publish(slot, Arc::from(segment));
                }
            }
        }

        let core = self.core();
        for pending in pending_pages {
            core.pages
                .get(pending.page_index)
                .expect("page still exists")
                .snapshot
                .store(Arc::new(pending.page));
        }
        Ok(())
    }
}

impl<K: Clone + Eq + Hash + Serialize + DeserializeOwned, const N: usize> VectorIndex<K, F16, N> {
    /// Insert a vector, replacing an existing vector with the same key.
    pub fn insert(&self, vector: F16Vector<K, N>) -> Result<(), IndexError>
    where
        K: Send + Sync,
    {
        self.insert_batch(std::iter::once(vector))
    }

    /// Insert at most 64 inputs per group and write each touched page once per group.
    /// Earlier groups remain committed if a later group returns an error.
    pub fn insert_batch(
        &self,
        vectors: impl IntoIterator<Item = F16Vector<K, N>>,
    ) -> Result<(), IndexError>
    where
        K: Send + Sync,
    {
        let mut batch = Vec::with_capacity(PAGE_SIZE);
        for vector in vectors {
            batch.push(vector);
            if batch.len() == PAGE_SIZE {
                self.insert_batch_page(std::mem::take(&mut batch))?;
                batch = Vec::with_capacity(PAGE_SIZE);
            }
        }
        if !batch.is_empty() {
            self.insert_batch_page(batch)?;
        }
        Ok(())
    }

    fn insert_batch_page(&self, vectors: Vec<F16Vector<K, N>>) -> Result<(), IndexError>
    where
        K: Send + Sync,
    {
        let core = self.core();
        let _mutation = core
            .mutation
            .lock()
            .unwrap_or_else(|error| error.into_inner());

        let mut positions = HashMap::with_capacity(vectors.len());
        let mut unique = Vec::<F16Vector<K, N>>::with_capacity(vectors.len());
        for vector in vectors {
            if let Some(&position) = positions.get(&vector.key) {
                unique[position] = vector;
            } else {
                positions.insert(vector.key.clone(), unique.len());
                unique.push(vector);
            }
        }

        let mut changes = Vec::with_capacity(unique.len());
        for vector in unique {
            let entry = core.key_entry(vector.key.clone());
            let existing = *entry
                .index
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let is_new = existing.is_none();
            let index = existing
                .unwrap_or_else(|| core.allocate_slot(matches!(self, Self::FileBacked { .. })));
            changes.push((vector, entry, index, is_new));
        }

        let result = (|| {
            let mut groups = BTreeMap::<usize, Vec<usize>>::new();
            for (position, (_, _, index, _)) in changes.iter().enumerate() {
                groups.entry(index.page_index()).or_default().push(position);
            }

            let groups: Vec<_> = groups.into_iter().collect();
            let pending_pages = if groups.len() > 1 {
                groups
                    .into_par_iter()
                    .map(|(page_index, positions)| {
                        self.prepare_insert_page(page_index, &positions, &changes)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                groups
                    .into_iter()
                    .map(|(page_index, positions)| {
                        self.prepare_insert_page(page_index, &positions, &changes)
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };

            self.publish_pages(pending_pages)?;
            for (_, entry, index, is_new) in &changes {
                *entry
                    .index
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(*index);
                if *is_new {
                    core.len.fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        })();

        if result.is_err() {
            for (vector, entry, index, is_new) in &changes {
                if *is_new {
                    core.free_queue.push(*index);
                }
                core.remove_idle_key_entry(&vector.key, entry);
            }
        }
        result
    }

    fn prepare_insert_page(
        &self,
        page_index: usize,
        positions: &[usize],
        changes: &[(F16Vector<K, N>, Arc<KeyEntry>, VecIndex, bool)],
    ) -> Result<PendingPage<K, N>, IndexError> {
        let cell = self.core().pages.get(page_index).expect("allocated page");
        let old = cell.snapshot.load_full();
        let mut page = self.clone_page(&old)?;
        let mut file_segments = if matches!(self, Self::FileBacked { .. }) {
            Some(self.copy_segments(&old)?)
        } else {
            None
        };

        for &position in positions {
            let (vector, _, index, _) = &changes[position];
            let slot = index.offset();
            page.segment_norm_ratios[slot] = vector.segment_norm_ratios;
            page.key_table[slot] = Some(vector.key.clone());
            page.valid_bitmap |= 1_u64 << slot;
            match &mut page.segments {
                PageSegments::OnMemory(segments) => {
                    for (segment_index, segment) in segments.iter_mut().enumerate() {
                        segment.0[slot] = vector.segments[segment_index];
                    }
                }
                PageSegments::FileBacked { .. } => {
                    for (segment_index, segment) in file_segments
                        .as_mut()
                        .expect("FileBacked segment copy")
                        .iter_mut()
                        .enumerate()
                    {
                        segment.0[slot] = vector.segments[segment_index];
                    }
                }
            }
        }
        Ok(PendingPage {
            page_index,
            page,
            file_segments,
        })
    }

    /// Normalize and insert FP32 coordinates.
    pub fn insert_f32(&self, key: K, values: &[f32]) -> Result<(), IndexError>
    where
        K: Send + Sync,
    {
        self.insert(F16Vector::from_f32(key, values)?)
    }

    /// Remove a key if present.
    pub fn remove(&self, key: &K) -> Result<(), IndexError>
    where
        K: Send + Sync,
    {
        self.remove_batch(std::iter::once(key))
    }

    /// Remove keys in groups of at most 64 pages, committing each page once.
    pub fn remove_batch<'a>(&self, keys: impl IntoIterator<Item = &'a K>) -> Result<(), IndexError>
    where
        K: 'a + Send + Sync,
    {
        let core = self.core();
        let _mutation = core
            .mutation
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut seen = HashSet::new();
        let mut groups = BTreeMap::<usize, Vec<KeyRemoval<'_, K>>>::new();
        let mut idle_entries = Vec::new();
        for key in keys {
            if !seen.insert(key) {
                continue;
            }
            let entry = core
                .keys
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .get(key)
                .cloned();
            let Some(entry) = entry else { continue };
            let index = *entry
                .index
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if let Some(index) = index {
                groups
                    .entry(index.page_index())
                    .or_default()
                    .push((key, entry, index));
            } else {
                idle_entries.push((key, entry));
            }
        }

        for (key, entry) in idle_entries {
            core.remove_idle_key_entry(key, &entry);
        }
        let groups: Vec<_> = groups.into_iter().collect();
        for page_batch in groups.chunks(PAGE_SIZE) {
            self.remove_page_batch(page_batch)?;
        }
        Ok(())
    }

    fn remove_page_batch(
        &self,
        groups: &[(usize, Vec<KeyRemoval<'_, K>>)],
    ) -> Result<(), IndexError>
    where
        K: Send + Sync,
    {
        let core = self.core();
        let prepare_page = |(page_index, removals): &(usize, Vec<KeyRemoval<'_, K>>)| {
            let cell = core.pages.get(*page_index).expect("indexed page");
            let old = cell.snapshot.load_full();
            let mut page = self.clone_page(&old)?;
            let mut file_segments = if matches!(self, Self::FileBacked { .. }) {
                Some(self.copy_segments(&old)?)
            } else {
                None
            };
            for (_, _, index) in removals {
                let slot = index.offset();
                page.valid_bitmap &= !(1_u64 << slot);
                page.key_table[slot] = None;
                page.segment_norm_ratios[slot] = [0.0; N];
                if let Some(segments) = &mut file_segments {
                    for segment in segments {
                        segment.0[slot] = [F16::ZERO; SEGMENT_DIMENSION];
                    }
                }
            }
            Ok::<_, IndexError>(PendingPage {
                page_index: *page_index,
                page,
                file_segments,
            })
        };
        let pending_pages = if groups.len() > 1 {
            groups
                .par_iter()
                .map(&prepare_page)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            groups
                .iter()
                .map(&prepare_page)
                .collect::<Result<Vec<_>, _>>()?
        };

        self.publish_pages(pending_pages)?;
        for (_, removals) in groups {
            for (key, entry, index) in removals {
                *entry
                    .index
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = None;
                core.len.fetch_sub(1, Ordering::Relaxed);
                core.free_queue.push(*index);
                core.remove_idle_key_entry(key, entry);
            }
        }
        Ok(())
    }

    /// Return top-k results in descending score order using unit-norm segment bounds.
    /// Pruning ignores FP16 norm drift and may differ from a full evaluation.
    pub fn search<Q>(
        &self,
        query: &F16Vector<Q, N>,
        top_k: usize,
    ) -> Result<Vec<(K, f32)>, IndexError>
    where
        K: Send + Sync,
    {
        search::search(self, query, top_k)
    }

    /// 検索し、枝刈りとSegment演算の統計を返す。
    pub fn search_with_stats<Q>(
        &self,
        query: &F16Vector<Q, N>,
        top_k: usize,
    ) -> Result<(Vec<(K, f32)>, SearchStats<N>), IndexError>
    where
        K: Send + Sync,
    {
        search::search_with_stats(self, query, top_k)
    }

    /// 検索途中の累積統計を通知し、検索結果と最終統計を返す。
    /// 大規模な検索では通知はRayonワーカースレッド上で逐次実行される。
    /// Pruning ignores FP16 norm drift and may differ from a full evaluation.
    pub fn search_with_progress<Q, F: FnMut(SearchProgress<N>) + Send>(
        &self,
        query: &F16Vector<Q, N>,
        top_k: usize,
        progress: F,
    ) -> Result<(Vec<(K, f32)>, SearchStats<N>), IndexError>
    where
        K: Send + Sync,
    {
        search::search_with_progress(self, query, top_k, progress)
    }

    /// Evaluate every live vector without Segment-bound pruning.
    pub fn search_full_scan<Q>(
        &self,
        query: &F16Vector<Q, N>,
        top_k: usize,
    ) -> Result<Vec<(K, f32)>, IndexError>
    where
        K: Send + Sync,
    {
        search::search_full_scan(self, query, top_k)
    }

    /// Evaluate every live vector without pruning and return search statistics.
    pub fn search_full_scan_with_stats<Q>(
        &self,
        query: &F16Vector<Q, N>,
        top_k: usize,
    ) -> Result<(Vec<(K, f32)>, SearchStats<N>), IndexError>
    where
        K: Send + Sync,
    {
        search::search_full_scan_with_stats(self, query, top_k)
    }

    /// Evaluate every live vector without pruning and report progress.
    /// Large searches invoke the callback serially on Rayon worker threads.
    pub fn search_full_scan_with_progress<Q, F: FnMut(SearchProgress<N>) + Send>(
        &self,
        query: &F16Vector<Q, N>,
        top_k: usize,
        progress: F,
    ) -> Result<(Vec<(K, f32)>, SearchStats<N>), IndexError>
    where
        K: Send + Sync,
    {
        search::search_full_scan_with_progress(self, query, top_k, progress)
    }

    /// Normalize FP32 query coordinates and return top-k results.
    /// Pruning ignores FP16 norm drift and may differ from a full evaluation.
    pub fn search_f32(&self, values: &[f32], top_k: usize) -> Result<Vec<(K, f32)>, IndexError>
    where
        K: Send + Sync,
    {
        self.search(&F16Vector::from_f32((), values)?, top_k)
    }

    /// Normalize FP32 query coordinates and include pruning statistics.
    pub fn search_f32_with_stats(
        &self,
        values: &[f32],
        top_k: usize,
    ) -> Result<(Vec<(K, f32)>, SearchStats<N>), IndexError>
    where
        K: Send + Sync,
    {
        self.search_with_stats(&F16Vector::from_f32((), values)?, top_k)
    }

    /// Number of vectors in a quiescent index; concurrent updates may be in flight.
    pub fn len(&self) -> usize {
        self.core().len.load(Ordering::Relaxed)
    }

    /// Whether there are currently no live vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K: Clone + Eq + Hash + Serialize + DeserializeOwned + Send + Sync, const N: usize>
    VectorIndexOps<K, F16, N> for VectorIndex<K, F16, N>
{
    fn insert(
        &self,
        vector: NormalizedSegmentedDocumentVector<K, F16, N>,
    ) -> Result<(), IndexError> {
        self.insert_batch(std::iter::once(vector))
    }

    fn remove(&self, key: &K) -> Result<(), IndexError> {
        self.remove_batch(std::iter::once(key))
    }

    fn search(
        &self,
        query: &NormalizedSegmentedDocumentVector<K, F16, N>,
        top_k: usize,
    ) -> Result<Vec<(K, f32)>, IndexError> {
        search::search(self, query, top_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn file_cache_is_lazy_reused_and_skipped_for_pruned_pages() {
        const N: usize = 2;
        let path = std::env::temp_dir().join(format!(
            "vise-cache-{}-{}.idx",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let index = VectorIndex::<String, F16, N>::create_file_backed(&path).unwrap();
        let mut first = [0.0; N * SEGMENT_DIMENSION];
        first[0] = 1.0;
        let mut second = [0.0; N * SEGMENT_DIMENSION];
        second[SEGMENT_DIMENSION] = 1.0;
        for i in 0..=PAGE_SIZE {
            let values = if i == PAGE_SIZE { &second } else { &first };
            index
                .insert(NormalizedSegmentedDocumentVector::from_f32(i.to_string(), values).unwrap())
                .unwrap();
        }
        drop(index);

        let segment_bytes = std::mem::size_of::<AlignedSegmentPage<F16>>();
        let index = VectorIndex::<String, F16, N>::open_file_backed_with_cache_limit(
            &path,
            segment_bytes * 4,
        )
        .unwrap();
        let first_page = index.core().pages.get(0).unwrap().snapshot.load_full();
        let last_page = index.core().pages.get(1).unwrap().snapshot.load_full();
        let PageSegments::FileBacked {
            cache: first_cache, ..
        } = &first_page.segments
        else {
            panic!("file-backed page")
        };
        let PageSegments::FileBacked {
            cache: last_cache, ..
        } = &last_page.segments
        else {
            panic!("file-backed page")
        };
        assert!(first_cache.iter().all(|entry| entry.load_full().is_none()));
        assert!(last_cache.iter().all(|entry| entry.load_full().is_none()));
        let query =
            NormalizedSegmentedDocumentVector::from_f32("query".to_owned(), &first).unwrap();
        assert_eq!(index.search(&query, 1).unwrap()[0].1, 1.0);
        assert!(index.cache_stats().resident_bytes <= segment_bytes * 4);
        let loaded = first_cache[0].load_full().unwrap();
        assert!(last_cache.iter().all(|entry| entry.load_full().is_none()));
        index.search(&query, 1).unwrap();
        assert!(Arc::ptr_eq(&loaded, &first_cache[0].load_full().unwrap()));
        index.set_cache_limit_bytes(segment_bytes);
        assert!(index.cache_stats().resident_bytes <= segment_bytes);
        index.set_cache_limit_bytes(0);
        assert_eq!(index.cache_stats().resident_bytes, 0);
        index.search(&query, 1).unwrap();
        assert_eq!(index.cache_stats().resident_bytes, 0);
        drop(index);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn file_backed_batches_reuse_pages_and_compact() {
        const N: usize = 1;
        let path = std::env::temp_dir().join(format!(
            "vise-batch-{}-{}.idx",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut index = VectorIndex::<String, F16, N>::create_file_backed(&path).unwrap();
        let vectors = (0..PAGE_SIZE * 2 + 1)
            .map(|i| {
                let mut values = [0.0; SEGMENT_DIMENSION];
                values[0] = 1.0;
                F16Vector::from_f32(i.to_string(), &values)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        index.insert_batch(vectors).unwrap();
        assert_eq!(index.core().pages.count(), 3);

        let mut replacement = [0.0; SEGMENT_DIMENSION];
        replacement[1] = 1.0;
        index
            .insert_batch([F16Vector::from_f32("0".to_owned(), &replacement).unwrap()])
            .unwrap();
        let removed: Vec<_> = (1..PAGE_SIZE).map(|i| i.to_string()).collect();
        index.remove_batch(&removed).unwrap();
        assert_eq!(index.len(), PAGE_SIZE + 2);

        let added = (0..PAGE_SIZE - 1)
            .map(|i| {
                let mut values = [0.0; SEGMENT_DIMENSION];
                values[0] = 1.0;
                F16Vector::from_f32(format!("new-{i}"), &values).unwrap()
            })
            .collect::<Vec<_>>();
        index.insert_batch(added).unwrap();
        assert_eq!(index.core().pages.count(), 3);
        assert_eq!(index.len(), PAGE_SIZE * 2 + 1);

        let before_compact = std::fs::metadata(&path).unwrap().len();
        index.compact().unwrap();
        let after_compact = std::fs::metadata(&path).unwrap().len();
        assert!(after_compact < before_compact);
        drop(index);

        let index = VectorIndex::<String, F16, N>::open_file_backed(&path).unwrap();
        assert_eq!(index.len(), PAGE_SIZE * 2 + 1);
        let mut query = [0.0; SEGMENT_DIMENSION];
        query[1] = 1.0;
        assert_eq!(index.search_f32(&query, 1).unwrap()[0].0, "0");
        drop(index);
        std::fs::remove_file(path).unwrap();
    }
}
