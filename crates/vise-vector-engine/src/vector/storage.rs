use super::{AlignedSegmentPage, IndexError, PAGE_SIZE, SEGMENT_DIMENSION};
use half::f16;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{File, OpenOptions},
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Mutex,
};

const MAGIC: &[u8; 8] = b"VISEFP16";
const VERSION: u32 = 4;
const HEADER_LEN: u64 = 32;
const DATA: &[u8; 4] = b"DATA";
const META: &[u8; 4] = b"META";
const COMMIT: &[u8; 4] = b"CMIT";
const DATA_HEADER: u64 = 32;

/// Metadata for one committed page generation.
pub(crate) struct StoredPage<K, const N: usize> {
    pub page_index: u64,
    pub generation: u64,
    pub valid_bitmap: u64,
    pub keys: [Option<K>; PAGE_SIZE],
    pub segment_norm_ratios: [[f32; N]; PAGE_SIZE],
    pub segment_offset: u64,
    pub segment_checksums: [u32; N],
}

pub(crate) struct PageWrite<K, const N: usize> {
    pub page: StoredPage<K, N>,
    pub segments: [Box<AlignedSegmentPage<f16>>; N],
}

/// Immutable page generations and an append-only metadata commit journal.
pub struct Storage<K, const N: usize> {
    file: File,
    path: PathBuf,
    writer: Mutex<WriterState>,
    _key: PhantomData<fn() -> K>,
}

struct WriterState {
    poisoned: bool,
    append_end: u64,
}

impl<K: Serialize + DeserializeOwned, const N: usize> Storage<K, N> {
    pub(crate) fn create(path: impl AsRef<Path>) -> Result<Self, IndexError> {
        validate_dimensions::<N>()?;
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        write_all_at(&file, &make_header::<N>(), 0)?;
        file.sync_all()?;
        sync_parent(&path)?;
        Ok(Self {
            file,
            path,
            writer: Mutex::new(WriterState {
                poisoned: false,
                append_end: HEADER_LEN,
            }),
            _key: PhantomData,
        })
    }

    pub(crate) fn open(
        path: impl AsRef<Path>,
    ) -> Result<(Self, Vec<StoredPage<K, N>>), IndexError> {
        validate_dimensions::<N>()?;
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut header = [0u8; HEADER_LEN as usize];
        read_exact_at(&file, &mut header, 0).map_err(|e| format_err(e, "truncated header"))?;
        if &header[..8] != MAGIC
            || u32_at(&header, 8) != VERSION
            || u32_at(&header, 12) != SEGMENT_DIMENSION as u32
            || u32_at(&header, 16) != N as u32
            || u32_at(&header, 20) != (N * SEGMENT_DIMENSION) as u32
            || u32_at(&header, 24) != 1
            || u32_at(&header, 28) != 1
        {
            return Err(IndexError::InvalidFormat(
                "unsupported or mismatched vise header".into(),
            ));
        }

        let len = file.metadata()?.len();
        let mut pos = HEADER_LEN;
        let mut records = std::collections::HashMap::<u64, (u64, u64, Vec<u8>, u32)>::new();
        let mut commits = Vec::<(u64, u64, u64, u32, u64)>::new();
        while pos + 4 <= len {
            let mut tag = [0; 4];
            if read_exact_at(&file, &mut tag, pos).is_err() {
                break;
            }
            if &tag == DATA {
                let mut h = [0u8; DATA_HEADER as usize];
                if read_exact_at(&file, &mut h, pos).is_err() {
                    break;
                }
                let page = u64_at(&h, 4);
                let generation = u64_at(&h, 12);
                let payload_len = u64_at(&h, 20);
                if payload_len != N as u64 * PAGE_SIZE as u64 * SEGMENT_DIMENSION as u64 * 2 {
                    break;
                }
                let end = pos
                    .checked_add(DATA_HEADER)
                    .and_then(|v| v.checked_add(payload_len));
                let Some(end) = end.filter(|v| *v <= len) else {
                    break;
                };
                let _ = (page, generation);
                pos = end;
            } else if &tag == META {
                let mut h = [0u8; 28];
                if read_exact_at(&file, &mut h, pos).is_err() {
                    break;
                }
                let page = u64_at(&h, 4);
                let generation = u64_at(&h, 12);
                let size = u32_at(&h, 20) as u64;
                let end = pos.checked_add(28).and_then(|v| v.checked_add(size));
                let Some(end) = end.filter(|v| *v <= len) else {
                    break;
                };
                let mut payload = vec![0; size as usize];
                if read_exact_at(&file, &mut payload, pos + 28).is_err() {
                    break;
                }
                let check = crc32(&payload);
                if read_u32_at(&file, pos + 24)? != check {
                    break;
                }
                records.insert(pos, (page, generation, payload, check));
                pos = end;
            } else if &tag == COMMIT {
                let mut b = [0u8; 36];
                if read_exact_at(&file, &mut b, pos).is_err() {
                    break;
                }
                if u32_at(&b, 32) != crc32(&b[..32]) {
                    break;
                }
                commits.push((
                    u64_at(&b, 4),
                    u64_at(&b, 12),
                    u64_at(&b, 20),
                    u32_at(&b, 28),
                    pos + 36,
                ));
                pos += 36;
            } else {
                break;
            }
        }
        let mut latest = std::collections::HashMap::<u64, StoredPage<K, N>>::new();
        let mut committed_end = HEADER_LEN;
        for (page, generation, meta_pos, sum, commit_end) in commits {
            if let Some((p, meta_generation, payload, check)) = records.remove(&meta_pos)
                && p == page
                && meta_generation == generation
                && check == sum
            {
                let stored = decode_page::<K, N>(&payload, page, generation)?;
                validate_stored(&stored)?;
                let mut dh = [0u8; DATA_HEADER as usize];
                let data_offset = stored.segment_offset.saturating_sub(DATA_HEADER);
                let data_end = data_offset.checked_add(
                    DATA_HEADER + N as u64 * PAGE_SIZE as u64 * SEGMENT_DIMENSION as u64 * 2,
                );
                if data_end
                    .is_some_and(|end| data_offset >= HEADER_LEN && end <= meta_pos && end <= len)
                    && read_exact_at(&file, &mut dh, data_offset).is_ok()
                    && &dh[..4] == DATA
                    && u64_at(&dh, 4) == page
                    && u64_at(&dh, 12) == generation
                    && u64_at(&dh, 20) == N as u64 * PAGE_SIZE as u64 * SEGMENT_DIMENSION as u64 * 2
                {
                    committed_end = committed_end.max(commit_end);
                    latest.insert(page, stored);
                }
            }
        }
        let mut pages: Vec<_> = latest.into_values().collect();
        pages.sort_by_key(|p| p.page_index);
        let storage = Self {
            file,
            path,
            writer: Mutex::new(WriterState {
                poisoned: false,
                append_end: committed_end,
            }),
            _key: PhantomData,
        };
        Ok((storage, pages))
    }

    pub(crate) fn commit_pages(&self, pages: &mut [PageWrite<K, N>]) -> Result<(), IndexError> {
        if pages.is_empty() {
            return Ok(());
        }
        let page_len = N * PAGE_SIZE * SEGMENT_DIMENSION * 2;
        let data_len = page_len as u64;
        let mut data_positions = Vec::with_capacity(pages.len());
        {
            let mut state = self.writer.lock().unwrap_or_else(|e| e.into_inner());
            if state.poisoned {
                return Err(IndexError::InvalidInput(
                    "storage requires reopen after uncertain commit".into(),
                ));
            }
            let mut pos = state.append_end;
            let reservation = (|| -> io::Result<()> {
                self.file.set_len(pos)?;
                for page in pages.iter() {
                    data_positions.push(pos);
                    let mut header = Vec::with_capacity(DATA_HEADER as usize);
                    header.extend_from_slice(DATA);
                    header.extend_from_slice(&page.page.page_index.to_le_bytes());
                    header.extend_from_slice(&page.page.generation.to_le_bytes());
                    header.extend_from_slice(&data_len.to_le_bytes());
                    header.extend_from_slice(&0u32.to_le_bytes());
                    write_all_at(&self.file, &header, pos)?;
                    pos = pos
                        .checked_add(DATA_HEADER + data_len)
                        .ok_or_else(|| io::Error::other("index file offset overflow"))?;
                }
                self.file.set_len(pos)?;
                self.file.sync_all()
            })();
            if let Err(error) = reservation {
                return Err(error.into());
            }
            state.append_end = pos;
        }

        let mut payload = vec![0; page_len];
        for (page, data_pos) in pages.iter_mut().zip(data_positions) {
            page.page.segment_offset = data_pos + DATA_HEADER;
            for (segment_index, segment) in page.segments.iter().enumerate() {
                let start = segment_index * PAGE_SIZE * SEGMENT_DIMENSION * 2;
                let bytes = &mut payload[start..start + PAGE_SIZE * SEGMENT_DIMENSION * 2];
                let mut cursor = 0;
                for row in &segment.0 {
                    for value in row {
                        bytes[cursor..cursor + 2].copy_from_slice(&value.to_bits().to_le_bytes());
                        cursor += 2;
                    }
                }
                page.page.segment_checksums[segment_index] = crc32(bytes);
            }
            write_all_at(&self.file, &payload, page.page.segment_offset)?;
        }
        self.file.sync_all()?;

        let mut state = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        if state.poisoned {
            return Err(IndexError::InvalidInput(
                "storage requires reopen after uncertain commit".into(),
            ));
        }
        let mut pos = state.append_end;
        let mut commit_records = Vec::with_capacity(pages.len());
        self.file.set_len(pos)?;
        for page in pages.iter() {
            let payload = encode_page(&page.page)?;
            let checksum = crc32(&payload);
            let meta = encode_meta_record(&page.page, &payload)?;
            let meta_pos = pos;
            write_all_at(&self.file, &meta, pos)?;
            pos = pos
                .checked_add(meta.len() as u64)
                .ok_or_else(|| IndexError::InvalidInput("index file offset overflow".into()))?;
            commit_records.push((
                page.page.page_index,
                page.page.generation,
                meta_pos,
                checksum,
            ));
        }
        let mut marker_pos = pos;
        for (page, generation, meta_pos, checksum) in commit_records {
            let marker = encode_commit_marker(page, generation, meta_pos, checksum);
            if let Err(error) = write_all_at(&self.file, &marker, marker_pos) {
                state.poisoned = true;
                return Err(IndexError::CommitOutcomeUnknown(error));
            }
            marker_pos += marker.len() as u64;
        }
        if let Err(error) = self.file.sync_all() {
            state.poisoned = true;
            return Err(IndexError::CommitOutcomeUnknown(error));
        }
        state.append_end = marker_pos;
        Ok(())
    }

    pub(crate) fn compact(
        &mut self,
        pages: &[StoredPage<K, N>],
    ) -> Result<Vec<StoredPage<K, N>>, IndexError>
    where
        K: Clone,
    {
        let mut state = self.writer.lock().unwrap();
        if state.poisoned {
            return Err(IndexError::InvalidInput(
                "storage requires reopen after uncertain commit".into(),
            ));
        }
        let mut sequence = 0_u64;
        let (temp, target) = loop {
            let mut temp = self.path.clone();
            temp.set_extension(format!("compact-{}-{sequence}.tmp", std::process::id()));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&temp)
            {
                Ok(file) => break (temp, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    sequence = sequence.checked_add(1).ok_or_else(|| {
                        IndexError::InvalidInput("no available compact temporary file".into())
                    })?;
                }
                Err(error) => return Err(error.into()),
            }
        };
        let result = (|| -> Result<(Vec<StoredPage<K, N>>, u64), IndexError> {
            let header = make_header::<N>();
            write_all_at(&target, &header, 0)?;
            let mut out_pos = HEADER_LEN;
            let mut rewritten_pages = Vec::with_capacity(pages.len());
            let mut commit_records = Vec::with_capacity(pages.len());
            let size = N * PAGE_SIZE * SEGMENT_DIMENSION * 2;
            for page in pages {
                validate_stored(page)?;
                let mut bytes = vec![0; size];
                let mut checksums = [0; N];
                if page.segment_offset != 0 && page.valid_bitmap != 0 {
                    read_exact_at(&self.file, &mut bytes, page.segment_offset)?;
                    let segment_size = PAGE_SIZE * SEGMENT_DIMENSION * 2;
                    for (segment, checksum) in checksums.iter_mut().enumerate() {
                        let start = segment * segment_size;
                        let data = &mut bytes[start..start + segment_size];
                        if crc32(data) != page.segment_checksums[segment] {
                            return Err(IndexError::InvalidFormat(
                                "segment checksum mismatch during compact".into(),
                            ));
                        }
                        for slot in 0..PAGE_SIZE {
                            if page.valid_bitmap & (1_u64 << slot) == 0 {
                                let row_start = slot * SEGMENT_DIMENSION * 2;
                                data[row_start..row_start + SEGMENT_DIMENSION * 2].fill(0);
                            }
                        }
                        *checksum = crc32(data);
                    }
                } else {
                    if page.valid_bitmap != 0 {
                        return Err(IndexError::InvalidFormat(
                            "page with vectors has no segment data".into(),
                        ));
                    }
                    for (segment, checksum) in checksums.iter_mut().enumerate() {
                        let start = segment * PAGE_SIZE * SEGMENT_DIMENSION * 2;
                        *checksum = crc32(&bytes[start..start + PAGE_SIZE * SEGMENT_DIMENSION * 2]);
                    }
                }
                let data_pos = out_pos;
                let data_len = size;
                let mut header = Vec::with_capacity(DATA_HEADER as usize);
                header.extend_from_slice(DATA);
                header.extend_from_slice(&page.page_index.to_le_bytes());
                header.extend_from_slice(&page.generation.to_le_bytes());
                header.extend_from_slice(&(data_len as u64).to_le_bytes());
                header.extend_from_slice(&0u32.to_le_bytes());
                write_all_at(&target, &header, out_pos)?;
                write_all_at(&target, &bytes, out_pos + DATA_HEADER)?;
                out_pos += DATA_HEADER + data_len as u64;
                let rewritten = StoredPage {
                    page_index: page.page_index,
                    generation: page.generation,
                    valid_bitmap: page.valid_bitmap,
                    keys: page.keys.clone(),
                    segment_norm_ratios: page.segment_norm_ratios,
                    segment_offset: data_pos + DATA_HEADER,
                    segment_checksums: checksums,
                };
                let payload = encode_page(&rewritten)?;
                let meta_pos = out_pos;
                let meta_record = encode_meta_record(&rewritten, &payload)?;
                write_all_at(&target, &meta_record, out_pos)?;
                out_pos += meta_record.len() as u64;
                commit_records.push((page.page_index, page.generation, meta_pos, crc32(&payload)));
                rewritten_pages.push(rewritten);
            }
            target.sync_all()?;
            for (page, generation, meta_pos, checksum) in commit_records {
                let marker = encode_commit_marker(page, generation, meta_pos, checksum);
                write_all_at(&target, &marker, out_pos)?;
                out_pos += marker.len() as u64;
            }
            target.sync_all()?;
            Ok((rewritten_pages, out_pos))
        })();
        let (rewritten_pages, out_pos) = match result {
            Ok(v) => v,
            Err(error) => {
                drop(target);
                let _ = std::fs::remove_file(&temp);
                return Err(error);
            }
        };
        drop(target);
        // Windows refuses to replace an open destination unless its handle was
        // opened with FILE_SHARE_DELETE. Close it explicitly during replacement.
        let placeholder = match dummy_file() {
            Ok(file) => file,
            Err(error) => {
                let _ = std::fs::remove_file(&temp);
                return Err(error.into());
            }
        };
        let old_file = std::mem::replace(&mut self.file, placeholder);
        drop(old_file);
        if let Err(error) = replace_file(&temp, &self.path) {
            self.file = match OpenOptions::new().read(true).write(true).open(&self.path) {
                Ok(file) => file,
                Err(reopen_error) => {
                    state.poisoned = true;
                    return Err(IndexError::CommitOutcomeUnknown(reopen_error));
                }
            };
            let _ = std::fs::remove_file(&temp);
            return Err(error.into());
        }
        if let Err(error) = sync_parent(&self.path) {
            state.poisoned = true;
            return Err(IndexError::CommitOutcomeUnknown(error));
        }
        self.file = match OpenOptions::new().read(true).write(true).open(&self.path) {
            Ok(file) => file,
            Err(error) => {
                state.poisoned = true;
                return Err(IndexError::CommitOutcomeUnknown(error));
            }
        };
        state.append_end = out_pos;
        Ok(rewritten_pages)
    }
}

impl<K, const N: usize> Storage<K, N> {
    pub(crate) fn read_segment(
        &self,
        offset: u64,
        segment: usize,
        checksum: u32,
    ) -> Result<AlignedSegmentPage<f16>, IndexError> {
        if segment >= N {
            return Err(IndexError::InvalidInput(
                "segment index out of range".into(),
            ));
        }
        let size = PAGE_SIZE * SEGMENT_DIMENSION * 2;
        let mut bytes = [0u8; PAGE_SIZE * SEGMENT_DIMENSION * 2];
        let segment_offset = offset
            .checked_add(segment as u64 * size as u64)
            .ok_or_else(|| IndexError::InvalidFormat("segment offset overflow".into()))?;
        read_exact_at(&self.file, &mut bytes, segment_offset)?;
        if crc32(&bytes) != checksum {
            return Err(IndexError::InvalidFormat(
                "segment checksum mismatch".into(),
            ));
        }
        let mut page = AlignedSegmentPage([[f16::ZERO; SEGMENT_DIMENSION]; PAGE_SIZE]);
        let mut i = 0;
        for row in &mut page.0 {
            for value in row {
                *value = f16::from_bits(u16::from_le_bytes([bytes[i], bytes[i + 1]]));
                i += 2;
            }
        }
        Ok(page)
    }
}

fn encode_page<K: Serialize, const N: usize>(p: &StoredPage<K, N>) -> Result<Vec<u8>, IndexError> {
    let keys = postcard::to_allocvec(p.keys.as_slice())
        .map_err(|e| IndexError::InvalidInput(e.to_string()))?;
    let mut b = Vec::new();
    b.extend_from_slice(&p.valid_bitmap.to_le_bytes());
    b.extend_from_slice(&p.segment_offset.to_le_bytes());
    for x in p.segment_checksums {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    b.extend_from_slice(&keys);
    for row in &p.segment_norm_ratios {
        for x in row {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    Ok(b)
}

fn encode_meta_record<K, const N: usize>(
    page: &StoredPage<K, N>,
    payload: &[u8],
) -> Result<Vec<u8>, IndexError> {
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| IndexError::InvalidInput("page metadata exceeds format limit".into()))?;
    let mut record = Vec::with_capacity(28 + payload.len());
    record.extend_from_slice(META);
    record.extend_from_slice(&page.page_index.to_le_bytes());
    record.extend_from_slice(&page.generation.to_le_bytes());
    record.extend_from_slice(&payload_len.to_le_bytes());
    record.extend_from_slice(&crc32(payload).to_le_bytes());
    record.extend_from_slice(payload);
    Ok(record)
}

fn encode_commit_marker(page: u64, generation: u64, meta_pos: u64, checksum: u32) -> Vec<u8> {
    let mut marker = Vec::with_capacity(36);
    marker.extend_from_slice(COMMIT);
    marker.extend_from_slice(&page.to_le_bytes());
    marker.extend_from_slice(&generation.to_le_bytes());
    marker.extend_from_slice(&meta_pos.to_le_bytes());
    marker.extend_from_slice(&checksum.to_le_bytes());
    let check = crc32(&marker);
    marker.extend_from_slice(&check.to_le_bytes());
    marker
}

fn decode_page<K: DeserializeOwned, const N: usize>(
    b: &[u8],
    page_index: u64,
    generation: u64,
) -> Result<StoredPage<K, N>, IndexError> {
    let mut p = 0;
    let valid_bitmap = take_u64(b, &mut p)?;
    let segment_offset = take_u64(b, &mut p)?;
    let mut sums = [0; N];
    for x in &mut sums {
        *x = take_u32(b, &mut p)?;
    }
    let key_len = take_u32(b, &mut p)? as usize;
    let end = p
        .checked_add(key_len)
        .filter(|x| *x <= b.len())
        .ok_or_else(|| IndexError::InvalidFormat("truncated key table".into()))?;
    let keys_vec: Vec<Option<K>> =
        postcard::from_bytes(&b[p..end]).map_err(|e| IndexError::InvalidFormat(e.to_string()))?;
    let keys: [Option<K>; PAGE_SIZE] = keys_vec
        .try_into()
        .map_err(|_| IndexError::InvalidFormat("invalid key table length".into()))?;
    p = end;
    let mut segment_norm_ratios = [[0.0; N]; PAGE_SIZE];
    for row in &mut segment_norm_ratios {
        for x in row {
            *x = f32::from_le_bytes(take_arr4(b, &mut p)?);
        }
    }
    if p != b.len() {
        return Err(IndexError::InvalidFormat(
            "unexpected page metadata bytes".into(),
        ));
    }
    Ok(StoredPage {
        page_index,
        generation,
        valid_bitmap,
        keys,
        segment_norm_ratios,
        segment_offset,
        segment_checksums: sums,
    })
}

fn take_arr4(b: &[u8], p: &mut usize) -> Result<[u8; 4], IndexError> {
    let e = p
        .checked_add(4)
        .filter(|e| *e <= b.len())
        .ok_or_else(|| IndexError::InvalidFormat("truncated metadata".into()))?;
    let a = b[*p..e].try_into().unwrap();
    *p = e;
    Ok(a)
}
fn take_u32(b: &[u8], p: &mut usize) -> Result<u32, IndexError> {
    Ok(u32::from_le_bytes(take_arr4(b, p)?))
}
fn take_u64(b: &[u8], p: &mut usize) -> Result<u64, IndexError> {
    let e = p
        .checked_add(8)
        .filter(|e| *e <= b.len())
        .ok_or_else(|| IndexError::InvalidFormat("truncated metadata".into()))?;
    let a = u64::from_le_bytes(b[*p..e].try_into().unwrap());
    *p = e;
    Ok(a)
}
fn u32_at(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
}
fn crc32(b: &[u8]) -> u32 {
    crc32fast::hash(b)
}

fn validate_dimensions<const N: usize>() -> Result<(), IndexError> {
    if N == 0
        || N.checked_mul(SEGMENT_DIMENSION)
            .is_none_or(|n| n > u32::MAX as usize)
    {
        return Err(IndexError::InvalidInput("unsupported segment count".into()));
    }
    Ok(())
}

fn validate_stored<K, const N: usize>(p: &StoredPage<K, N>) -> Result<(), IndexError> {
    for i in 0..PAGE_SIZE {
        if (p.valid_bitmap & (1u64 << i) != 0) != p.keys[i].is_some() {
            return Err(IndexError::InvalidFormat(
                "valid bitmap and key table disagree".into(),
            ));
        }
        for j in 0..N {
            if !p.segment_norm_ratios[i][j].is_finite() || p.segment_norm_ratios[i][j] < 0.0 {
                return Err(IndexError::InvalidFormat("invalid norm metadata".into()));
            }
        }
    }
    Ok(())
}
fn format_err(e: io::Error, s: &str) -> IndexError {
    IndexError::InvalidFormat(format!("{s}: {e}"))
}
fn read_u32_at(f: &File, o: u64) -> io::Result<u32> {
    let mut b = [0; 4];
    read_exact_at(f, &mut b, o)?;
    Ok(u32::from_le_bytes(b))
}

fn write_all_at(f: &File, mut b: &[u8], mut o: u64) -> io::Result<()> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    while !b.is_empty() {
        #[cfg(unix)]
        let n = f.write_at(b, o)?;
        #[cfg(windows)]
        let n = f.seek_write(b, o)?;
        if n == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        o += n as u64;
        b = &b[n..];
    }
    Ok(())
}

fn make_header<const N: usize>() -> Vec<u8> {
    let mut b = Vec::with_capacity(HEADER_LEN as usize);
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&VERSION.to_le_bytes());
    b.extend_from_slice(&(SEGMENT_DIMENSION as u32).to_le_bytes());
    b.extend_from_slice(&(N as u32).to_le_bytes());
    b.extend_from_slice(&((N * SEGMENT_DIMENSION) as u32).to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes());
    b
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    File::open(parent)?.sync_all()
}

#[cfg(windows)]
fn sync_parent(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(windows)]
fn dummy_file() -> io::Result<File> {
    File::open("NUL")
}
#[cfg(unix)]
fn dummy_file() -> io::Result<File> {
    File::open("/dev/null")
}

#[cfg(unix)]
fn replace_file(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::rename(from, to)
}
#[cfg(windows)]
fn replace_file(from: &Path, to: &Path) -> io::Result<()> {
    use std::{ffi::OsStr, os::windows::ffi::OsStrExt};
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }
    let from: Vec<u16> = OsStr::new(from).encode_wide().chain(Some(0)).collect();
    let to: Vec<u16> = OsStr::new(to).encode_wide().chain(Some(0)).collect();
    // MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 0x1 | 0x8) } == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn read_exact_at(f: &File, mut b: &mut [u8], mut o: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !b.is_empty() {
        let n = f.read_at(b, o)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        o += n as u64;
        b = &mut b[n..]
    }
    Ok(())
}
#[cfg(windows)]
fn read_exact_at(f: &File, mut b: &mut [u8], mut o: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !b.is_empty() {
        let n = f.seek_read(b, o)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        o += n as u64;
        b = &mut b[n..]
    }
    Ok(())
}
