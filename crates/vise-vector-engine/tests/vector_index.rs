use std::{
    fs,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use half::f16;
use vise_vector_engine::vector::{NormalizedSegmentedDocumentVector, VectorIndex};

const N: usize = 1;
type Document = NormalizedSegmentedDocumentVector<String, f16, N>;
type Index = VectorIndex<String, f16, N>;

fn vector(key: impl Into<String>, values: &[f32]) -> Document {
    Document::from_f32(key.into(), values).expect("valid 64-dimensional vector")
}

fn basis(index: usize) -> Vec<f32> {
    let mut values = vec![0.0; 64];
    values[index] = 1.0;
    values
}

fn temp_file(label: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "vise-{label}-{}-{}.idx",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ))
}

#[test]
fn validates_inputs_and_handles_empty_and_zero_vectors() {
    assert!(Document::from_f32("bad".to_owned(), &[0.0; 63]).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = vec![0.0; 64];
        values[17] = bad;
        assert!(Document::from_f32("bad".to_owned(), &values).is_err());
    }

    let index = Index::on_memory();
    let query = vector("query", &basis(0));
    assert!(index.search(&query, 10).unwrap().is_empty());
    assert!(index.search(&query, 0).unwrap().is_empty());

    index.insert(vector("zero", &[0.0; 64])).unwrap();
    let hits = index.search(&query, 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, "zero");
    assert_eq!(hits[0].1, 0.0);
}

#[test]
fn ties_are_repeatable_and_search_is_exact_for_simple_vectors() {
    let index = Index::on_memory();
    let query = vector("q", &basis(0));
    index.insert(vector("first", &basis(0))).unwrap();
    index.insert(vector("second", &basis(0))).unwrap();
    index.insert(vector("orthogonal", &basis(1))).unwrap();

    let all = index.search(&query, 99).unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].1, 1.0);
    assert_eq!(all[1].1, 1.0);
    assert_eq!(all[2].1, 0.0);
    assert_eq!(index.search(&query, 99).unwrap(), all);
    assert_eq!(index.search(&query, 1).unwrap(), vec![all[0].clone()]);
}

#[test]
fn insert_replaces_and_remove_releases_key_and_slot() {
    let index = Index::on_memory();
    let query = vector("query", &basis(0));
    index.insert(vector("same", &basis(0))).unwrap();
    index.insert(vector("same", &basis(1))).unwrap();
    assert_eq!(index.keys(), vec!["same".to_owned()]);
    let hits = index.search(&query, 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0], ("same".to_owned(), 0.0));

    index.remove(&"missing".to_owned()).unwrap();
    index.remove(&"same".to_owned()).unwrap();
    assert!(index.keys().is_empty());
    assert!(index.search(&query, 10).unwrap().is_empty());
    index.insert(vector("same", &basis(0))).unwrap();
    assert_eq!(index.search(&query, 10).unwrap()[0].1, 1.0);
}

#[test]
fn limited_file_cache_handles_concurrent_updates_and_searches() {
    let path = temp_file("bounded-cache");
    let limit = 64 * 64 * 2;
    let index = Arc::new(Index::create_file_backed_with_cache_limit(&path, limit).unwrap());
    for i in 0..70 {
        index
            .insert_f32(format!("doc-{i}"), &basis(i % 64))
            .unwrap();
    }
    let mut workers = Vec::new();
    for _ in 0..3 {
        let index = Arc::clone(&index);
        workers.push(thread::spawn(move || {
            for _ in 0..20 {
                let (hits, stats) = index.search_f32_with_stats(&basis(0), 3).unwrap();
                assert!(hits.iter().all(|(_, score)| score.is_finite()));
                assert_eq!(
                    stats.candidates,
                    stats.pruned_candidates() + stats.scored_candidates
                );
                assert!(index.cache_stats().resident_bytes <= limit);
            }
        }));
    }
    for i in 0..12 {
        index.insert_f32("changing".to_owned(), &basis(i)).unwrap();
    }
    for worker in workers {
        worker.join().unwrap();
    }
    let (hits, _) = index.search_f32_with_stats(&basis(0), 10).unwrap();
    assert_eq!(hits, index.search_f32(&basis(0), 10).unwrap());
    index.set_cache_limit_bytes(0);
    assert_eq!(index.cache_stats().resident_bytes, 0);
    assert_eq!(hits, index.search_f32(&basis(0), 10).unwrap());
    assert_eq!(index.cache_stats().resident_bytes, 0);
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.windows(4).filter(|tag| *tag == b"DATA").count(), 82);
    let mut index = match Arc::try_unwrap(index) {
        Ok(index) => index,
        Err(_) => panic!("all index handles should be released"),
    };
    index.compact().unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.windows(4).filter(|tag| *tag == b"DATA").count(), 2);
    drop(index);
    fs::remove_file(path).unwrap();
}

#[test]
fn concurrent_updates_and_searches_keep_valid_snapshots() {
    let index = Arc::new(Index::on_memory());
    let query = Arc::new(vector("query", &basis(0)));
    let mut workers = Vec::new();
    for worker in 0..4 {
        let index = Arc::clone(&index);
        let query = Arc::clone(&query);
        workers.push(thread::spawn(move || {
            for iteration in 0..40 {
                let key = format!("{worker}-{iteration}");
                index
                    .insert(vector(key.clone(), &basis(iteration % 64)))
                    .unwrap();
                let hits = index.search(&query, 8).unwrap();
                assert!(hits.iter().all(|(_, score)| score.is_finite()));
                index.remove(&key).unwrap();
            }
            let key = format!("worker-{worker}");
            index.insert(vector(key, &basis(worker))).unwrap();
        }));
    }
    for worker in workers {
        worker.join().expect("worker thread");
    }
    assert_eq!(index.search(&query, 99).unwrap().len(), 4);
}

fn check_reused_slot_snapshots(index: Arc<Index>, iterations: usize) {
    let query = Arc::new(vector("query", &basis(0)));
    let index_writer = Arc::clone(&index);
    let writer = thread::spawn(move || {
        index_writer.insert(vector("axis", &basis(0))).unwrap();
        for _ in 0..iterations {
            index_writer.remove(&"axis".to_owned()).unwrap();
            index_writer.insert(vector("other", &basis(1))).unwrap();
            index_writer.remove(&"other".to_owned()).unwrap();
            index_writer.insert(vector("axis", &basis(0))).unwrap();
        }
    });

    let mut readers = Vec::new();
    for _ in 0..2 {
        let index = Arc::clone(&index);
        let query = Arc::clone(&query);
        readers.push(thread::spawn(move || {
            for _ in 0..iterations * 4 {
                for (key, score) in index.search(&query, 8).unwrap() {
                    match key.as_str() {
                        "axis" => assert_eq!(score, 1.0),
                        "other" => assert_eq!(score, 0.0),
                        unexpected => panic!("unexpected key in snapshot: {unexpected}"),
                    }
                }
            }
        }));
    }
    writer.join().unwrap();
    for reader in readers {
        reader.join().unwrap();
    }
}

#[test]
fn reused_slots_never_pair_a_key_with_another_documents_vector() {
    check_reused_slot_snapshots(Arc::new(Index::on_memory()), 100);
}

#[test]
fn file_backed_reused_slots_keep_keys_and_vectors_together() {
    let path = temp_file("reuse");
    let index = Arc::new(Index::create_file_backed(&path).unwrap());
    check_reused_slot_snapshots(index.clone(), 12);
    drop(index);
    let _ = fs::remove_file(path);
}

#[test]
fn file_backed_concurrent_inserts_fill_pages_and_reopen() {
    let path = temp_file("concurrent-pages");
    let index = Arc::new(Index::create_file_backed(&path).unwrap());
    let mut workers = Vec::new();
    for worker in 0..4 {
        let index = Arc::clone(&index);
        workers.push(thread::spawn(move || {
            for item in 0..20 {
                let key = format!("{worker}-{item}");
                index
                    .insert(vector(key, &basis((worker * 20 + item) % 64)))
                    .unwrap();
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(index.len(), 80);
    let mut keys = index.keys();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), 80);
    let hits = index.search(&vector("query", &basis(0)), 80).unwrap();
    assert_eq!(hits.len(), 80);
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.windows(4).filter(|tag| *tag == b"DATA").count(), 80);
    drop(index);

    let reopened = Index::open_file_backed(&path).unwrap();
    assert_eq!(reopened.len(), 80);
    assert_eq!(
        reopened
            .search(&vector("query", &basis(0)), 80)
            .unwrap()
            .len(),
        80
    );
    drop(reopened);
    let _ = fs::remove_file(path);
}

#[test]
fn file_backed_concurrent_updates_of_one_key_keep_one_page_slot() {
    let path = temp_file("concurrent-key");
    let index = Arc::new(Index::create_file_backed(&path).unwrap());
    let mut workers = Vec::new();
    for worker in 0..4 {
        let index = Arc::clone(&index);
        workers.push(thread::spawn(move || {
            for item in 0..5 {
                index
                    .insert(vector("shared", &basis((worker * 5 + item) % 64)))
                    .unwrap();
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(index.len(), 1);
    assert_eq!(index.keys(), vec!["shared"]);
    assert_eq!(
        index.search(&vector("query", &basis(0)), 10).unwrap().len(),
        1
    );

    let mut removers = Vec::new();
    for _ in 0..4 {
        let index = Arc::clone(&index);
        removers.push(thread::spawn(move || {
            index.remove(&"shared".to_owned()).unwrap()
        }));
    }
    for remover in removers {
        remover.join().unwrap();
    }
    assert!(index.is_empty());
    drop(index);
    let reopened = Index::open_file_backed(&path).unwrap();
    assert!(reopened.is_empty());
    drop(reopened);
    let _ = fs::remove_file(path);
}

#[test]
fn file_backed_reopen_compact_and_memory_results_match() {
    let path = temp_file("persist");
    let query = vector("query", &basis(0));
    let memory = Index::on_memory();
    let disk = Index::create_file_backed(&path).unwrap();
    for (key, values) in [("a", basis(0)), ("b", basis(1)), ("c", vec![0.5; 64])] {
        let document = vector(key, &values);
        memory.insert(vector(key, &values)).unwrap();
        disk.insert(document).unwrap();
    }
    let expected = memory.search(&query, 10).unwrap();
    assert_eq!(disk.search(&query, 10).unwrap(), expected);
    disk.remove(&"b".to_owned()).unwrap();
    memory.remove(&"b".to_owned()).unwrap();
    let expected = memory.search(&query, 10).unwrap();
    assert_eq!(disk.search(&query, 10).unwrap(), expected);
    drop(disk);

    let mut reopened = Index::open_file_backed(&path).unwrap();
    assert_eq!(reopened.search(&query, 10).unwrap(), expected);
    let before_compact = fs::metadata(&path).unwrap().len();
    reopened.compact().unwrap();
    assert_eq!(reopened.search(&query, 10).unwrap(), expected);
    assert!(fs::metadata(&path).unwrap().len() < before_compact);
    drop(reopened);
    let after_compact = Index::open_file_backed(&path).unwrap();
    assert_eq!(after_compact.search(&query, 10).unwrap(), expected);

    drop(after_compact);
    let _ = fs::remove_file(path);
}

#[test]
fn file_backed_detects_segment_corruption_only_when_loaded() {
    use std::io::{Seek, SeekFrom};

    let path = temp_file("corrupt-segment");
    let index = Index::create_file_backed(&path).unwrap();
    index.insert(vector("doc", &basis(0))).unwrap();
    drop(index);
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    // File header + DATA header.
    file.seek(SeekFrom::Start(64)).unwrap();
    file.write_all(&[0xff]).unwrap();
    drop(file);

    let reopened = Index::open_file_backed(&path).unwrap();
    let query = vector("query", &basis(0));
    assert!(matches!(
        reopened.search(&query, 1),
        Err(vise_vector_engine::vector::IndexError::InvalidFormat(_))
    ));
    drop(reopened);
    let _ = fs::remove_file(path);
}

#[test]
fn compact_does_not_read_deleted_page_data() {
    use std::io::{Seek, SeekFrom};

    let path = temp_file("compact-deleted");
    let index = Index::create_file_backed(&path).unwrap();
    index.insert(vector("doc", &basis(0))).unwrap();
    index.remove(&"doc".to_owned()).unwrap();
    drop(index);
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(64)).unwrap();
    file.write_all(&[0xff]).unwrap();
    drop(file);

    let mut reopened = Index::open_file_backed(&path).unwrap();
    reopened.compact().unwrap();
    assert!(
        reopened
            .search(&vector("query", &basis(0)), 1)
            .unwrap()
            .is_empty()
    );
    drop(reopened);
    let reopened = Index::open_file_backed(&path).unwrap();
    assert!(
        reopened
            .search(&vector("query", &basis(0)), 1)
            .unwrap()
            .is_empty()
    );
    drop(reopened);
    let _ = fs::remove_file(path);
}

#[test]
fn file_backed_ignores_an_uncommitted_trailing_fragment() {
    let path = temp_file("truncated");
    let query = vector("query", &basis(0));
    let index = Index::create_file_backed(&path).unwrap();
    index.insert(vector("committed", &basis(0))).unwrap();
    drop(index);

    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"partial-record")
        .unwrap();

    let reopened = Index::open_file_backed(&path).unwrap();
    assert_eq!(
        reopened.search(&query, 10).unwrap(),
        vec![("committed".to_owned(), 1.0)]
    );
    reopened.insert(vector("next", &basis(1))).unwrap();
    drop(reopened);

    let reopened = Index::open_file_backed(&path).unwrap();
    assert_eq!(reopened.search(&query, 10).unwrap().len(), 2);
    drop(reopened);
    let _ = fs::remove_file(path);
}

#[test]
fn file_backed_recovers_previous_commit_at_each_transaction_boundary() {
    let source = temp_file("recovery-source");
    let query = vector("query", &basis(0));
    let index = Index::create_file_backed(&source).unwrap();
    index.insert(vector("old", &basis(0))).unwrap();
    let previous_end = fs::metadata(&source).unwrap().len();
    index.insert(vector("new", &basis(1))).unwrap();
    drop(index);

    let bytes = fs::read(&source).unwrap();
    let find_tag = |tag: &[u8; 4], from: usize| {
        bytes[from..]
            .windows(4)
            .position(|window| window == tag)
            .map(|offset| from + offset)
            .expect("record tag")
    };
    assert_eq!(bytes.windows(4).filter(|tag| *tag == b"DATA").count(), 2);
    let meta_start = find_tag(b"META", previous_end as usize);
    let meta_size =
        u32::from_le_bytes(bytes[meta_start + 20..meta_start + 24].try_into().unwrap()) as usize;
    let commit_start = meta_start + 28 + meta_size;
    let full_end = bytes.len() as u64;
    let cuts = [
        previous_end + 1,
        (meta_start + 1) as u64,
        (meta_start + 28 + meta_size / 2) as u64,
        (commit_start + 1) as u64,
        (commit_start + 35) as u64,
    ];

    for (case, cut) in cuts.into_iter().enumerate() {
        let truncated = temp_file(&format!("recovery-{case}"));
        fs::copy(&source, &truncated).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&truncated)
            .unwrap()
            .set_len(cut)
            .unwrap();
        let recovered = Index::open_file_backed(&truncated).unwrap();
        assert_eq!(
            recovered.search(&query, 10).unwrap(),
            vec![("old".to_owned(), 1.0)],
            "partial transaction at byte {cut} should restore prior commit"
        );
        drop(recovered);
        let _ = fs::remove_file(truncated);
    }

    let complete = Index::open_file_backed(&source).unwrap();
    assert_eq!(
        complete.search(&query, 10).unwrap(),
        vec![("old".to_owned(), 1.0), ("new".to_owned(), 0.0)]
    );
    assert_eq!(full_end, (commit_start + 36) as u64);
    drop(complete);
    let _ = fs::remove_file(source);
}
