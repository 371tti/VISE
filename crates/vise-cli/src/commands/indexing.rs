use super::{
    documents::{
        collect_text_files, is_supported_text_file, key_in_scope, read_text_file, relative_path,
        text_chunks,
    },
    embedding::{default_embed_batch_size, device_name, load_embedder},
    terminal::{
        document_progress, finish_progress, format_duration, format_finish_time, start_spinner,
    },
};
use crate::{args, interrupt, progress_estimate};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashSet,
    error::Error,
    fs,
    io::{self, IsTerminal},
    path::Path,
    time::{Duration, Instant, SystemTime},
};
use vise_vector_engine::vector::{
    F16Index, NormalizedSegmentedDocumentVector, PAGE_SIZE, VectorIndex,
};

pub(super) fn index_directory(
    root: &Path,
    index_path: &Path,
    cache_limit: Option<usize>,
    requested_batch_size: Option<usize>,
    recursive: bool,
    dry_run: bool,
    device_kind: args::DeviceKind,
) -> Result<(), Box<dyn Error>> {
    interrupt::check()?;
    if !root.is_dir() {
        return Err(format!("documents directory does not exist: {}", root.display()).into());
    }
    let scanning = start_spinner(format!("Scanning {}", root.display()));
    let mut files = Vec::new();
    collect_text_files(root, recursive, &mut files)?;
    // An index may itself have a text extension and live inside the source tree.
    if index_path.is_file() && is_supported_text_file(index_path) {
        let index_canonical = fs::canonicalize(index_path)?;
        let mut source_files = Vec::with_capacity(files.len());
        for path in files {
            interrupt::check()?;
            if fs::canonicalize(&path)? != index_canonical {
                source_files.push(path);
            }
        }
        files = source_files;
    }
    files.sort_unstable();
    interrupt::check()?;
    finish_progress(&scanning, format!("Found {} supported files", files.len()));
    let existing_index = index_path.exists();
    if files.is_empty() && !dry_run {
        return Err(format!(
            "no supported text files found in {}{}",
            root.display(),
            if recursive {
                ""
            } else {
                " (use -r to include subdirectories)"
            }
        )
        .into());
    }
    let batch_setting = requested_batch_size.map_or_else(
        || "auto (CPU 32 / CUDA 16 / Vulkan 16)".to_owned(),
        |size| size.to_string(),
    );
    println!(
        "Embedding index: Bekko A8M | Scan: {} | {} files | batch size: {batch_setting}",
        if recursive { "recursive" } else { "top level" },
        files.len()
    );
    if !recursive {
        eprintln!("Subdirectory entries already in the index are retained; use -r to rescan them.");
    }
    if dry_run {
        for path in &files {
            interrupt::check()?;
            println!("  {}", relative_path(root, path)?);
        }
        println!("Dry run: no model loaded and no index changes made.");
        return Ok(());
    }
    let mut sizes = Vec::with_capacity(files.len());
    let mut total_bytes = 0u64;
    for path in &files {
        interrupt::check()?;
        // Empty input still has tokenizer special tokens and a nonzero processing cost.
        let bytes = fs::metadata(path)?.len().max(3);
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or("total input size is too large")?;
        sizes.push(bytes);
    }
    let estimated_total_tokens =
        progress_estimate::estimate_progress(total_bytes, 0, Duration::ZERO, SystemTime::now())
            .total_tokens;
    println!(
        "Estimated input: ~{estimated_total_tokens} tokens (file bytes / 3; excludes overlap and special tokens)."
    );
    let loading = start_spinner("Selecting device and loading Bekko A8M".to_owned());
    let (embedder, selected_device) = load_embedder(device_kind)?;
    let batch_size =
        requested_batch_size.unwrap_or_else(|| default_embed_batch_size(selected_device));
    interrupt::check()?;
    finish_progress(
        &loading,
        format!(
            "Bekko A8M ready on {}; inference batch size {batch_size}",
            device_name(selected_device)
        ),
    );
    if let Some(parent) = index_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }

    let loading = start_spinner(if existing_index {
        "Loading index metadata".to_owned()
    } else {
        "Creating index".to_owned()
    });
    let mut index = if existing_index {
        VectorIndex::<String, _>::open_file_backed(index_path)?
    } else {
        VectorIndex::<String, _>::create_file_backed(index_path)?
    };
    interrupt::check()?;
    if let Some(limit) = cache_limit {
        index.set_cache_limit_bytes(limit);
    }
    finish_progress(
        &loading,
        if existing_index {
            "Index metadata loaded".to_owned()
        } else {
            "Index created".to_owned()
        },
    );

    let old_keys: HashSet<String> = index
        .keys()
        .into_iter()
        .filter(|key| key_in_scope(key, recursive))
        .collect();
    let mut current_keys = HashSet::new();
    let mut completed_docs = 0usize;
    let mut completed_doc_time = Duration::ZERO;
    let mut indexed_chunks = 0usize;
    if !files.is_empty() {
        let progress = document_progress(files.len(), total_bytes)?;
        let started = Instant::now();
        let mut last_log = Instant::now();
        let mut completed_bytes = 0u64;
        let mut pending_vectors = Vec::with_capacity(PAGE_SIZE);
        for (path, &file_bytes) in files.iter().zip(&sizes) {
            if interrupt::is_requested() && !pending_vectors.is_empty() {
                index.insert_batch(std::mem::take(&mut pending_vectors))?;
                pending_vectors = Vec::with_capacity(PAGE_SIZE);
            }
            interrupt::check()?;
            let doc_started = Instant::now();
            let text = read_text_file(path)?;
            if interrupt::is_requested() && !pending_vectors.is_empty() {
                index.insert_batch(std::mem::take(&mut pending_vectors))?;
                pending_vectors = Vec::with_capacity(PAGE_SIZE);
            }
            interrupt::check()?;
            let relative = relative_path(root, path)?;
            let chunks: Vec<_> = text_chunks(&text).collect();
            let total_chunk_bytes: u128 =
                chunks.iter().map(|chunk| chunk.len().max(1) as u128).sum();
            let mut processed_chunk_bytes = 0u128;
            progress.set_message(format!(
                "{completed_docs}/{} docs | processing {} ({} chunks)\n  estimating current document time",
                files.len(),
                path.file_name().unwrap_or_default().to_string_lossy(),
                chunks.len()
            ));
            if interrupt::is_requested() && !pending_vectors.is_empty() {
                index.insert_batch(std::mem::take(&mut pending_vectors))?;
                pending_vectors = Vec::with_capacity(PAGE_SIZE);
            }
            interrupt::check()?;
            for (batch_number, batch) in chunks.chunks(batch_size).enumerate() {
                if interrupt::is_requested() && !pending_vectors.is_empty() {
                    index.insert_batch(std::mem::take(&mut pending_vectors))?;
                    pending_vectors = Vec::with_capacity(PAGE_SIZE);
                }
                interrupt::check()?;
                let embeddings = embedder.encode_batch(batch)?;
                for (batch_offset, (chunk, embedding)) in batch.iter().zip(embeddings).enumerate() {
                    let chunk_number = batch_number * batch_size + batch_offset;
                    let key = format!("{relative}\0{chunk_number}");
                    let vector = NormalizedSegmentedDocumentVector::<String, _>::from_f32(
                        key.clone(),
                        &embedding,
                    )?;
                    pending_vectors.push(vector);
                    current_keys.insert(key);
                    indexed_chunks += 1;
                    if pending_vectors.len() == PAGE_SIZE {
                        index.insert_batch(std::mem::take(&mut pending_vectors))?;
                        pending_vectors = Vec::with_capacity(PAGE_SIZE);
                    }
                    processed_chunk_bytes += chunk.len().max(1) as u128;
                }
                let chunk_number = batch_number * batch_size + batch.len() - 1;
                let completed_in_file =
                    (u128::from(file_bytes) * processed_chunk_bytes / total_chunk_bytes) as u64;
                let completed_work = completed_bytes.saturating_add(completed_in_file);
                let estimate = progress_estimate::estimate_progress(
                    total_bytes,
                    completed_work,
                    started.elapsed(),
                    SystemTime::now(),
                );
                let doc_fraction = (chunk_number + 1) as f64 / chunks.len() as f64;
                let docs_done = completed_docs as f64 + doc_fraction;
                let elapsed = started.elapsed().as_secs_f64();
                let docs_per_second = docs_done / elapsed.max(f64::EPSILON);
                let eta = estimate
                    .eta
                    .map(format_duration)
                    .unwrap_or_else(|| "estimating".to_owned());
                let finish = format_finish_time(estimate.finish_at)
                    .unwrap_or_else(|| "estimating".to_owned());
                let tokens_per_second = estimate.tokens_per_second.unwrap_or(0.0);
                let doc_finished = chunk_number + 1 == chunks.len();
                let displayed_completed_docs = completed_docs + usize::from(doc_finished);
                let displayed_doc_time = completed_doc_time
                    + if doc_finished {
                        doc_started.elapsed()
                    } else {
                        Duration::ZERO
                    };
                let average = if displayed_completed_docs == 0 {
                    "estimating".to_owned()
                } else {
                    format!(
                        "{:.2}s/doc",
                        displayed_doc_time.as_secs_f64() / displayed_completed_docs as f64
                    )
                };
                progress.set_position(completed_work);
                let message = format!(
                    "{}/{} docs | chunk {}/{} | {}\n  ~{}/{} tokens | ~{:.0} tokens/s | ~{:.2} docs/s\n  ~ETA {} | ~finish {} | avg {} | current {:.2}s",
                    displayed_completed_docs,
                    files.len(),
                    chunk_number + 1,
                    chunks.len(),
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    estimate.completed_tokens,
                    estimate.total_tokens,
                    tokens_per_second,
                    docs_per_second,
                    eta,
                    finish,
                    average,
                    doc_started.elapsed().as_secs_f64(),
                );
                progress.set_message(message.clone());
                if !io::stderr().is_terminal() && last_log.elapsed() >= Duration::from_secs(5) {
                    eprintln!("{}", message.replace('\n', " |"));
                    last_log = Instant::now();
                }
            }
            completed_doc_time += doc_started.elapsed();
            completed_docs += 1;
            completed_bytes = completed_bytes.saturating_add(file_bytes);
        }

        if !pending_vectors.is_empty() {
            index.insert_batch(pending_vectors)?;
        }

        finish_progress(
            &progress,
            format!(
                "{completed_docs} docs | {indexed_chunks} chunks | avg {:.2}s/doc",
                completed_doc_time.as_secs_f64() / completed_docs as f64
            ),
        );
    }

    let mut stale_keys: Vec<_> = old_keys.difference(&current_keys).cloned().collect();
    interrupt::check()?;
    if !stale_keys.is_empty() {
        stale_keys.sort_unstable();
        if !io::stderr().is_terminal() {
            eprintln!("Removing {} stale chunks...", stale_keys.len());
        }
        let progress = ProgressBar::new(stale_keys.len() as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "Removing stale chunks {wide_bar:.yellow/blue} {pos}/{len}",
            )?
            .progress_chars("=>-"),
        );
        for keys in stale_keys.chunks(PAGE_SIZE * PAGE_SIZE) {
            interrupt::check()?;
            index.remove_batch(keys)?;
            progress.inc(keys.len() as u64);
        }
        interrupt::check()?;
        finish_progress(
            &progress,
            format!("Removed {} stale chunks", stale_keys.len()),
        );
    }

    interrupt::check()?;
    let compacting =
        start_spinner("Packing vector pages and reclaiming metadata history".to_owned());
    let compact_started = Instant::now();
    index.compact()?;
    interrupt::check()?;
    finish_progress(
        &compacting,
        format!(
            "Index storage compacted in {:.2}s",
            compact_started.elapsed().as_secs_f64()
        ),
    );
    println!(
        "Synchronized {} files and {} chunks (removed {} stale chunks).\nIndex: {}",
        completed_docs,
        indexed_chunks,
        stale_keys.len(),
        index_path.display()
    );
    Ok(())
}

pub(super) fn compact_index(index_path: &Path) -> Result<(), Box<dyn Error>> {
    interrupt::check()?;
    let size_before = fs::metadata(index_path)?.len();
    let mut index = F16Index::<String>::open_file_backed(index_path)?;
    let compacting = start_spinner("Rewriting index and reclaiming unused space".to_owned());
    let started = Instant::now();
    index.compact()?;
    interrupt::check()?;
    let size_after = fs::metadata(index_path)?.len();
    let reclaimed = size_before.saturating_sub(size_after);
    finish_progress(
        &compacting,
        format!(
            "Compacted in {:.2}s | {:.1} MiB → {:.1} MiB | reclaimed {:.1} MiB",
            started.elapsed().as_secs_f64(),
            size_before as f64 / (1024.0 * 1024.0),
            size_after as f64 / (1024.0 * 1024.0),
            reclaimed as f64 / (1024.0 * 1024.0),
        ),
    );
    println!("Index: {}", index_path.display());
    Ok(())
}
