use super::{
    DEFAULT_SEGMENT_NUM, SegmentDotKernel, VecIndex, VectorIndex,
    f16::{self, Kernel},
};
use half::f16 as F16;
use rayon::prelude::*;
use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    ops::Range,
    sync::{
        Mutex,
        atomic::{AtomicU32, Ordering as AtomicOrdering},
    },
};

struct Entry<K> {
    key: K,
    score: f32,
    index: VecIndex,
}

/// 検索対象数、枝刈り数、演算数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchStats<const N: usize = DEFAULT_SEGMENT_NUM> {
    /// 検索対象の有効ベクトル数。
    pub candidates: usize,
    /// 最初のSegment内積を計算する前に除外した候補数。
    pub initially_pruned: usize,
    /// Segment i の評価後に除外した候補数。
    /// 最後の要素は全Segment評価後の不採用数を含む。
    pub rejected_by_stage: [usize; N],
    /// Top-Kに渡した候補数。
    pub scored_candidates: usize,
    /// `dot_page()`の呼び出し回数。
    pub page_segment_calls: usize,
    /// 実際に計算した文書×Segmentの内積数。
    pub vector_segment_dots: usize,
}

impl<const N: usize> Default for SearchStats<N> {
    fn default() -> Self {
        Self {
            candidates: 0,
            initially_pruned: 0,
            rejected_by_stage: [0; N],
            scored_candidates: 0,
            page_segment_calls: 0,
            vector_segment_dots: 0,
        }
    }
}

impl<const N: usize> SearchStats<N> {
    /// 枝刈りで除外した候補数。
    pub fn pruned_candidates(self) -> usize {
        self.initially_pruned + self.rejected_by_stage.iter().sum::<usize>()
    }

    /// 枝刈りした候補の割合。候補がなければ`0.0`。
    pub fn pruning_ratio(self) -> f32 {
        if self.candidates == 0 {
            0.0
        } else {
            self.pruned_candidates() as f32 / self.candidates as f32
        }
    }

    fn merge(&mut self, other: Self) {
        self.candidates += other.candidates;
        self.initially_pruned += other.initially_pruned;
        for (total, count) in self
            .rejected_by_stage
            .iter_mut()
            .zip(other.rejected_by_stage)
        {
            *total += count;
        }
        self.scored_candidates += other.scored_candidates;
        self.page_segment_calls += other.page_segment_calls;
        self.vector_segment_dots += other.vector_segment_dots;
    }
}

/// 検索中に累積したページ処理状況。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchProgress<const N: usize = DEFAULT_SEGMENT_NUM> {
    /// 処理済みページ数。
    pub pages_completed: usize,
    /// 検索開始時のページ数。
    pub pages_total: usize,
    /// 候補数と枝刈り数の累計。
    pub stats: SearchStats<N>,
}

/// A bounded heap whose ties are resolved by the stable vector index.
pub struct TopK<K> {
    k: usize,
    heap: BinaryHeap<Reverse<Entry<K>>>,
}

impl<K> TopK<K> {
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::new(),
        }
    }

    pub(crate) fn threshold(&self) -> Option<f32> {
        (self.k != 0 && self.heap.len() == self.k)
            .then(|| self.heap.peek().expect("nonzero full heap").0.score)
    }

    pub(crate) fn push_with(&mut self, score: f32, index: VecIndex, key: impl FnOnce() -> K) {
        if self.k == 0 {
            return;
        }
        let insert = if self.heap.len() < self.k {
            true
        } else {
            self.heap.peek().is_some_and(|worst| {
                score
                    .total_cmp(&worst.0.score)
                    .then_with(|| worst.0.index.cmp(&index))
                    == Ordering::Greater
            })
        };
        if insert {
            if self.heap.len() == self.k {
                self.heap.pop();
            }
            self.heap.push(Reverse(Entry {
                key: key(),
                score,
                index,
            }));
        }
    }

    fn merge(&mut self, other: Self) {
        for Reverse(entry) in other.heap {
            self.push_with(entry.score, entry.index, || entry.key);
        }
    }

    pub(crate) fn into_sorted(self) -> Vec<(K, f32)> {
        let mut entries: Vec<_> = self.heap.into_iter().map(|Reverse(entry)| entry).collect();
        entries.sort_by(|a, b| b.cmp(a));
        entries
            .into_iter()
            .map(|entry| (entry.key, entry.score))
            .collect()
    }
}

impl<K> PartialEq for Entry<K> {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.index == other.index
    }
}
impl<K> Eq for Entry<K> {}
impl<K> PartialOrd for Entry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K> Ord for Entry<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            // For equal scores, the lower VecIndex ranks higher.
            .then_with(|| other.index.cmp(&self.index))
    }
}

/// Search live vectors with a unit-norm segment bound, ignoring FP16 norm drift.
pub(super) fn search<K: Clone + Send + Sync, Q, const N: usize>(
    index: &VectorIndex<K, F16, N>,
    query: &super::NormalizedSegmentedDocumentVector<Q, F16, N>,
    top_k: usize,
) -> Result<Vec<(K, f32)>, super::IndexError> {
    search_mode(index, query, top_k, true, false, |_| {}).map(|x| x.0)
}

pub(super) fn search_with_stats<K: Clone + Send + Sync, Q, const N: usize>(
    index: &VectorIndex<K, F16, N>,
    query: &super::NormalizedSegmentedDocumentVector<Q, F16, N>,
    top_k: usize,
) -> Result<(Vec<(K, f32)>, SearchStats<N>), super::IndexError> {
    search_mode(index, query, top_k, true, true, |_| {})
}

pub(super) fn search_with_progress<
    K: Clone + Send + Sync,
    Q,
    const N: usize,
    F: FnMut(SearchProgress<N>) + Send,
>(
    index: &VectorIndex<K, F16, N>,
    query: &super::NormalizedSegmentedDocumentVector<Q, F16, N>,
    top_k: usize,
    progress: F,
) -> Result<(Vec<(K, f32)>, SearchStats<N>), super::IndexError> {
    search_mode(index, query, top_k, true, true, progress)
}

pub(super) fn search_full_scan<K: Clone + Send + Sync, Q, const N: usize>(
    index: &VectorIndex<K, F16, N>,
    query: &super::NormalizedSegmentedDocumentVector<Q, F16, N>,
    top_k: usize,
) -> Result<Vec<(K, f32)>, super::IndexError> {
    search_mode(index, query, top_k, false, false, |_| {}).map(|x| x.0)
}

pub(super) fn search_full_scan_with_stats<K: Clone + Send + Sync, Q, const N: usize>(
    index: &VectorIndex<K, F16, N>,
    query: &super::NormalizedSegmentedDocumentVector<Q, F16, N>,
    top_k: usize,
) -> Result<(Vec<(K, f32)>, SearchStats<N>), super::IndexError> {
    search_mode(index, query, top_k, false, true, |_| {})
}

pub(super) fn search_full_scan_with_progress<
    K: Clone + Send + Sync,
    Q,
    const N: usize,
    F: FnMut(SearchProgress<N>) + Send,
>(
    index: &VectorIndex<K, F16, N>,
    query: &super::NormalizedSegmentedDocumentVector<Q, F16, N>,
    top_k: usize,
    progress: F,
) -> Result<(Vec<(K, f32)>, SearchStats<N>), super::IndexError> {
    search_mode(index, query, top_k, false, true, progress)
}

fn search_mode<K: Clone + Send + Sync, Q, const N: usize, F: FnMut(SearchProgress<N>) + Send>(
    index: &VectorIndex<K, F16, N>,
    query: &super::NormalizedSegmentedDocumentVector<Q, F16, N>,
    top_k: usize,
    pruning: bool,
    collect_stats: bool,
    progress: F,
) -> Result<(Vec<(K, f32)>, SearchStats<N>), super::IndexError> {
    if top_k == 0 {
        return Ok((Vec::new(), SearchStats::<N>::default()));
    }
    let kernel = f16::select();
    match kernel {
        Kernel::Scalar(k) => {
            search_kernel_mode(index, query, top_k, &k, pruning, collect_stats, progress)
        }
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2(k) => {
            search_kernel_mode(index, query, top_k, &k, pruning, collect_stats, progress)
        }
    }
}

fn search_kernel_mode<
    K: Clone + Send + Sync,
    Q,
    const N: usize,
    D: SegmentDotKernel<F16> + Sync,
    F: FnMut(SearchProgress<N>) + Send,
>(
    index: &VectorIndex<K, F16, N>,
    query: &super::NormalizedSegmentedDocumentVector<Q, F16, N>,
    top_k: usize,
    kernel: &D,
    pruning: bool,
    collect_stats: bool,
    progress: F,
) -> Result<(Vec<(K, f32)>, SearchStats<N>), super::IndexError> {
    let core = index.core();
    let pages_total = core.pages.count();
    let query = SearchQuery {
        segments: query.segments,
        segment_norm_ratios: query.segment_norm_ratios,
    };
    let threads = rayon::current_num_threads();
    if threads <= 1 || pages_total < threads.saturating_mul(2) {
        return search_sequential(
            index,
            &query,
            pages_total,
            top_k,
            kernel,
            pruning,
            collect_stats,
            progress,
        );
    }

    search_parallel(
        index,
        &query,
        pages_total,
        top_k,
        kernel,
        pruning,
        collect_stats,
        threads,
        progress,
    )
}

#[derive(Clone, Copy)]
struct SearchQuery<const N: usize> {
    segments: [super::SegmentVector<F16>; N],
    segment_norm_ratios: [f32; N],
}

struct SearchAccumulator<K, const N: usize, F> {
    top: TopK<K>,
    stats: SearchStats<N>,
    pages_completed: usize,
    progress: F,
}

fn search_sequential<K: Clone, const N: usize, D, F>(
    index: &VectorIndex<K, F16, N>,
    query: &SearchQuery<N>,
    pages_total: usize,
    top_k: usize,
    kernel: &D,
    pruning: bool,
    collect_stats: bool,
    mut progress: F,
) -> Result<(Vec<(K, f32)>, SearchStats<N>), super::IndexError>
where
    D: SegmentDotKernel<F16>,
    F: FnMut(SearchProgress<N>),
{
    let mut top = TopK::new(top_k);
    let mut stats = SearchStats::<N>::default();
    let mut pages_completed = 0;
    for (page_index, slot) in index.core().pages.iter() {
        let page = slot.snapshot.load_full();
        search_page(
            index,
            page_index,
            &page,
            query,
            &mut top,
            &mut stats,
            kernel,
            pruning,
            collect_stats,
            None,
        )?;
        if collect_stats {
            pages_completed += 1;
            progress(SearchProgress {
                pages_completed,
                pages_total,
                stats,
            });
        }
    }
    Ok((top.into_sorted(), stats))
}

fn search_parallel<K, const N: usize, D, F>(
    index: &VectorIndex<K, F16, N>,
    query: &SearchQuery<N>,
    pages_total: usize,
    top_k: usize,
    kernel: &D,
    pruning: bool,
    collect_stats: bool,
    threads: usize,
    progress: F,
) -> Result<(Vec<(K, f32)>, SearchStats<N>), super::IndexError>
where
    K: Clone + Send + Sync,
    D: SegmentDotKernel<F16> + Sync,
    F: FnMut(SearchProgress<N>) + Send,
{
    let chunk_size = pages_total.div_ceil(threads.saturating_mul(8)).max(1);
    let chunks: Vec<Range<usize>> = (0..pages_total)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(pages_total))
        .collect();
    let accumulator = Mutex::new(SearchAccumulator {
        top: TopK::new(top_k),
        stats: SearchStats::default(),
        pages_completed: 0,
        progress,
    });
    let global_threshold = AtomicU32::new(f32::NEG_INFINITY.to_bits());

    chunks.into_par_iter().try_for_each(|range| {
        let mut local_top = TopK::new(top_k);
        let mut local_stats = SearchStats::default();
        for page_index in range.clone() {
            let page = index
                .core()
                .pages
                .get(page_index)
                .expect("page count is stable for the search range")
                .snapshot
                .load_full();
            search_page(
                index,
                page_index,
                &page,
                query,
                &mut local_top,
                &mut local_stats,
                kernel,
                pruning,
                collect_stats,
                Some(&global_threshold),
            )?;
        }

        let mut state = accumulator
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.stats.merge(local_stats);
        state.top.merge(local_top);
        state.pages_completed += range.len();
        if pruning && let Some(threshold) = state.top.threshold() {
            update_threshold(&global_threshold, threshold);
        }
        if collect_stats {
            let report = SearchProgress {
                pages_completed: state.pages_completed,
                pages_total,
                stats: state.stats,
            };
            (state.progress)(report);
        }
        Ok::<(), super::IndexError>(())
    })?;

    let state = accumulator
        .into_inner()
        .unwrap_or_else(|error| error.into_inner());
    Ok((state.top.into_sorted(), state.stats))
}

fn update_threshold(threshold: &AtomicU32, candidate: f32) {
    let mut current = threshold.load(AtomicOrdering::Relaxed);
    loop {
        if f32::from_bits(current).total_cmp(&candidate) != Ordering::Less {
            return;
        }
        match threshold.compare_exchange_weak(
            current,
            candidate.to_bits(),
            AtomicOrdering::Release,
            AtomicOrdering::Relaxed,
        ) {
            Ok(_) => return,
            Err(actual) => current = actual,
        }
    }
}

fn search_page<K: Clone, const N: usize, D: SegmentDotKernel<F16>>(
    index: &VectorIndex<K, F16, N>,
    page_index: usize,
    page: &super::VectorPage<K, F16, N>,
    query: &SearchQuery<N>,
    top: &mut TopK<K>,
    stats: &mut SearchStats<N>,
    kernel: &D,
    pruning: bool,
    collect_stats: bool,
    global_threshold: Option<&AtomicU32>,
) -> Result<(), super::IndexError> {
    let mut candidates = page.valid_bitmap;
    if collect_stats {
        stats.candidates += candidates.count_ones() as usize;
    }
    if candidates == 0 {
        return Ok(());
    }
    let threshold = if pruning {
        let global_threshold = global_threshold
            .map(|threshold| f32::from_bits(threshold.load(AtomicOrdering::Acquire)))
            .filter(|threshold| threshold.is_finite());
        match (top.threshold(), global_threshold) {
            (Some(local), Some(shared)) => Some(if local.total_cmp(&shared).is_lt() {
                shared
            } else {
                local
            }),
            (local, shared) => local.or(shared),
        }
    } else {
        None
    };
    let mut suffix_bounds = None;
    let mut boundable = 0;
    if let Some(threshold) = threshold {
        let mut bounds = [[0.0_f32; N]; super::PAGE_SIZE];
        boundable = build_remaining_upper_bounds(query, page, candidates, &mut bounds);
        let mut remaining = candidates;
        while remaining != 0 {
            let offset = remaining.trailing_zeros() as usize;
            if boundable & (1_u64 << offset) != 0
                && remaining_upper(0.0, 0, &bounds[offset]).is_some_and(|upper| upper < threshold)
            {
                candidates &= !(1_u64 << offset);
                if collect_stats {
                    stats.initially_pruned += 1;
                }
            }
            remaining &= remaining - 1;
        }
        suffix_bounds = Some(bounds);
    }

    let mut scores = [0.0_f32; super::PAGE_SIZE];
    for segment_index in 0..N {
        if candidates == 0 {
            break;
        }
        let mut dots = [0.0_f32; super::PAGE_SIZE];
        if collect_stats {
            stats.page_segment_calls += 1;
            stats.vector_segment_dots += candidates.count_ones() as usize;
        }
        index.with_segment(page, segment_index, |segment| {
            kernel.dot_page(
                &query.segments[segment_index],
                segment,
                candidates,
                &mut dots,
            );
        })?;

        let mut remaining = candidates;
        while remaining != 0 {
            let offset = remaining.trailing_zeros() as usize;
            scores[offset] += query.segment_norm_ratios[segment_index]
                * page.segment_norm_ratios[offset][segment_index]
                * dots[offset];
            if boundable & (1_u64 << offset) != 0
                && let Some(threshold) = threshold
                && let Some(bounds) = &suffix_bounds
                && remaining_upper(scores[offset], segment_index + 1, &bounds[offset])
                    .is_some_and(|upper| upper < threshold)
            {
                candidates &= !(1_u64 << offset);
                if collect_stats {
                    stats.rejected_by_stage[segment_index] += 1;
                }
            }
            remaining &= remaining - 1;
        }
    }

    while candidates != 0 {
        let offset = candidates.trailing_zeros() as usize;
        top.push_with(scores[offset], VecIndex::new(page_index, offset), || {
            page.key_table[offset]
                .as_ref()
                .expect("valid slot has a key")
                .clone()
        });
        if collect_stats {
            stats.scored_candidates += 1;
        }
        candidates &= candidates - 1;
    }
    Ok(())
}

fn build_remaining_upper_bounds<const N: usize, K>(
    query: &SearchQuery<N>,
    page: &super::VectorPage<K, F16, N>,
    candidates: u64,
    bounds: &mut [[f32; N]; super::PAGE_SIZE],
) -> u64 {
    let dot_bound = f64::from((1.00001_f64 as f32).next_up());
    let mut boundable = candidates;
    let mut remaining = candidates;
    while remaining != 0 {
        let offset = remaining.trailing_zeros() as usize;
        let bit = 1_u64 << offset;
        let mut suffix = 0.0_f64;
        for segment in (0..N).rev() {
            let weight =
                query.segment_norm_ratios[segment] * page.segment_norm_ratios[offset][segment];
            if !weight.is_finite() || weight < 0.0 {
                boundable &= !bit;
                break;
            }
            let term = f64::from(weight) * dot_bound;
            if term != 0.0 {
                suffix = (suffix + term).next_up();
            }
            let Some(upper) = round_up_f32(suffix) else {
                boundable &= !bit;
                break;
            };
            bounds[offset][segment] = upper;
        }
        remaining &= remaining - 1;
    }
    boundable
}

fn remaining_upper<const N: usize>(
    partial: f32,
    next_segment: usize,
    suffix_bounds: &[f32; N],
) -> Option<f32> {
    if !partial.is_finite() || next_segment > N {
        return None;
    }
    if next_segment == N {
        return Some(partial);
    }
    round_up_f32(f64::from(partial) + f64::from(suffix_bounds[next_segment]))
}

fn round_up_f32(value: f64) -> Option<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return None;
    }
    let rounded = value as f32;
    let upper = if f64::from(rounded) < value {
        rounded.next_up()
    } else {
        rounded
    };
    upper.is_finite().then_some(upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::NormalizedSegmentedDocumentVector;

    #[test]
    fn top_k_ties_use_vec_index_order() {
        let mut top = TopK::new(2);
        top.push_with(1.0, VecIndex(7), || "later");
        top.push_with(1.0, VecIndex(3), || "earlier");
        top.push_with(1.0, VecIndex(5), || "middle");
        assert_eq!(top.into_sorted(), vec![("earlier", 1.0), ("middle", 1.0)]);
    }

    #[test]
    fn pruning_preserves_full_top_k_order() {
        const N: usize = 2;
        let query_input = vec![1.0_f32; N * 64];
        let query = NormalizedSegmentedDocumentVector::<String, F16, N>::from_f32(
            "query".into(),
            &query_input,
        )
        .unwrap();

        let winner = query_input.clone();
        let mut loser = query_input.clone();
        for value in loser.iter_mut().take(64) {
            *value = -1.0;
        }
        let inputs = [("winner", winner), ("loser", loser)];
        let index = VectorIndex::<String, F16, N>::on_memory();
        let mut expected = Vec::new();
        for (key, input) in inputs {
            let vector = NormalizedSegmentedDocumentVector::<String, F16, N>::from_f32(
                key.to_owned(),
                &input,
            )
            .unwrap();
            let mut score = 0.0_f32;
            for i in 0..N {
                score += query.segment_norm_ratios[i]
                    * vector.segment_norm_ratios[i]
                    * f16::select().dot(&query.segments[i], &vector.segments[i]);
            }
            expected.push((key.to_owned(), score));
            index.insert(vector).unwrap();
        }
        expected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let actual = index.search(&query, 1).unwrap();
        expected.truncate(1);
        assert_eq!(
            actual.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.0.as_str()).collect::<Vec<_>>()
        );
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual.1 - expected.1).abs() < 1e-5);
        }
    }

    #[test]
    fn randomized_pruning_matches_full_evaluation() {
        const N: usize = 3;
        let mut state = 0x6d2b_79f5_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as i32 as f32) / (i32::MAX as f32)
        };
        let query_input: Vec<_> = (0..N * 64).map(|_| next()).collect();
        let query =
            NormalizedSegmentedDocumentVector::<(), F16, N>::from_f32((), &query_input).unwrap();
        let index = VectorIndex::<String, F16, N>::on_memory();
        for doc in 0..80 {
            let input: Vec<_> = (0..N * 64).map(|_| next()).collect();
            index
                .insert(
                    NormalizedSegmentedDocumentVector::from_f32(format!("doc-{doc}"), &input)
                        .unwrap(),
                )
                .unwrap();
        }
        let kernel = f16::select();
        let (pruned, pruned_stats) =
            search_kernel_mode(&index, &query, 9, &kernel, true, true, |_| {}).unwrap();
        let (full, full_stats) =
            search_kernel_mode(&index, &query, 9, &kernel, false, true, |_| {}).unwrap();
        assert!(
            pruned_stats.pruned_candidates() > 0,
            "test data should exercise a pruning branch"
        );
        assert_eq!(full_stats.pruned_candidates(), 0);
        assert_eq!(pruned_stats.candidates, full_stats.candidates);
        assert_eq!(
            pruned_stats.scored_candidates + pruned_stats.pruned_candidates(),
            pruned_stats.candidates
        );
        assert_eq!(pruned, full);
        let (public_hits, stats) = search_with_stats(&index, &query, 9).unwrap();
        assert_eq!(public_hits, pruned);
        assert_eq!(stats, pruned_stats);
        assert!(stats.page_segment_calls > 0);
        assert!(stats.vector_segment_dots >= stats.page_segment_calls);
        assert!((0.0..=1.0).contains(&stats.pruning_ratio()));
    }

    #[test]
    fn zero_k_and_empty_index_return_no_hits() {
        const N: usize = 1;
        let query = NormalizedSegmentedDocumentVector::<String, F16, N>::from_f32(
            "query".into(),
            &[0.0; 64],
        )
        .unwrap();
        let index = VectorIndex::<String, F16, N>::on_memory();
        assert!(index.search(&query, 0).unwrap().is_empty());
        assert!(index.search(&query, 3).unwrap().is_empty());
    }
}
