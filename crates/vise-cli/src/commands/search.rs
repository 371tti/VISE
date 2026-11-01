use super::{
    documents::{decode_key, read_preview},
    embedding::{device_name, load_embedder},
    terminal::{finish_progress, paint, start_spinner},
};
use crate::{args, interrupt};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    error::Error,
    io::{self, IsTerminal},
    path::Path,
    time::{Duration, Instant},
};
use vise_embed_bekko::BekkoEmbedder;
use vise_vector_engine::vector::{
    F16Index, F16Vector, NormalizedSegmentedDocumentVector, SearchProgress, VectorIndex,
};

pub(super) fn search_directory(
    index_path: &Path,
    examples_dir: Option<&Path>,
    query_text: Option<&str>,
    top_k: usize,
    cache_limit: Option<usize>,
    device_kind: args::DeviceKind,
    interactive: bool,
    exhaustive: bool,
) -> Result<(), Box<dyn Error>> {
    interrupt::check()?;
    if let Some(directory) = examples_dir
        && !directory.is_dir()
    {
        return Err(format!(
            "documents directory does not exist: {}",
            directory.display()
        )
        .into());
    }
    let loading = start_spinner("Loading index metadata".to_owned());
    let index = VectorIndex::<String, _>::open_file_backed(index_path)?;
    interrupt::check()?;
    if let Some(limit) = cache_limit {
        index.set_cache_limit_bytes(limit);
    }
    finish_progress(&loading, "Index metadata loaded".to_owned());
    if top_k == 0 {
        println!("top_k is 0; no query or vector scan was needed.");
        return Ok(());
    }

    let mut embedder = None;
    let mut next_query = query_text.map(str::to_owned);
    let interactive = interactive || next_query.is_none();
    loop {
        let query = match next_query.take() {
            Some(query) => query,
            None if interactive => {
                print!("\nquery> ");
                io::Write::flush(&mut io::stdout())?;
                let Some(query) = read_query_line()? else {
                    break;
                };
                let query = query.trim().to_owned();
                if query.is_empty() {
                    continue;
                }
                if matches!(query.as_str(), ":q" | "quit" | "exit") {
                    break;
                }
                query
            }
            None => break,
        };
        search_one_query(
            &index,
            &mut embedder,
            device_kind,
            examples_dir,
            &query,
            top_k,
            exhaustive,
        )?;
        if !interactive {
            break;
        }
    }
    Ok(())
}

fn read_query_line() -> io::Result<Option<String>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = io::stdin()
            .read_line(&mut line)
            .map(|length| (length, line));
        let _ = sender.send(result);
    });
    loop {
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok((0, _))) => return Ok(None),
            Ok(Ok((_, line))) => return Ok(Some(line)),
            Ok(Err(error)) => return Err(error),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) if interrupt::is_requested() => {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(io::Error::other("query input thread stopped unexpectedly"));
            }
        }
    }
}

fn search_one_query(
    index: &F16Index<String>,
    embedder: &mut Option<BekkoEmbedder>,
    device_kind: args::DeviceKind,
    examples_dir: Option<&Path>,
    query_text: &str,
    top_k: usize,
    exhaustive: bool,
) -> Result<(), Box<dyn Error>> {
    interrupt::check()?;
    if let Some(key) = query_text.strip_prefix(">>") {
        let literal_query = format!(">{key}");
        return search_text_query(
            index,
            embedder,
            device_kind,
            examples_dir,
            &literal_query,
            top_k,
            exhaustive,
        );
    }
    if let Some(displayed_key) = query_text.strip_prefix('>') {
        let (relative_path, chunk_index) = parse_document_key(displayed_key)?;
        let internal_key = format!("{relative_path}\0{chunk_index}");
        println!(
            "\n{} {displayed_key}",
            paint("Similar to:", "1;36", io::stdout().is_terminal())
        );
        let loading = start_spinner(format!("Loading vector for {displayed_key}"));
        let query = index.vector_by_key(&internal_key)?;
        interrupt::check()?;
        finish_progress(&loading, "Source vector loaded".to_owned());
        return search_vector(
            index,
            &query,
            top_k,
            examples_dir,
            exhaustive,
            Some(&internal_key),
        );
    }
    search_text_query(
        index,
        embedder,
        device_kind,
        examples_dir,
        query_text,
        top_k,
        exhaustive,
    )
}

fn search_text_query(
    index: &F16Index<String>,
    embedder: &mut Option<BekkoEmbedder>,
    device_kind: args::DeviceKind,
    examples_dir: Option<&Path>,
    query_text: &str,
    top_k: usize,
    exhaustive: bool,
) -> Result<(), Box<dyn Error>> {
    println!(
        "\n{} {query_text}",
        paint("Query:", "1;36", io::stdout().is_terminal())
    );
    if embedder.is_none() {
        let loading = start_spinner("Selecting device and loading Bekko A8M".to_owned());
        let (loaded, selected_device) = load_embedder(device_kind)?;
        interrupt::check()?;
        finish_progress(
            &loading,
            format!("Bekko A8M ready on {}", device_name(selected_device)),
        );
        *embedder = Some(loaded);
    }
    let embedding_progress = start_spinner("Embedding query".to_owned());
    let embedding_started = Instant::now();
    let embedding = embedder
        .as_ref()
        .expect("embedder initialized above")
        .encode(query_text)?;
    interrupt::check()?;
    finish_progress(
        &embedding_progress,
        format!(
            "Query embedding ready in {:.3}s",
            embedding_started.elapsed().as_secs_f64()
        ),
    );
    let query =
        NormalizedSegmentedDocumentVector::<String, _>::from_f32(String::new(), &embedding)?;
    search_vector(index, &query, top_k, examples_dir, exhaustive, None)
}

fn parse_document_key(key: &str) -> Result<(&str, usize), Box<dyn Error>> {
    let (path, chunk) = key
        .rsplit_once('#')
        .ok_or("document key must have the form <relative-path>#<chunk-index>")?;
    let chunk = chunk
        .parse::<usize>()
        .map_err(|_| "document key chunk index must be a non-negative integer")?;
    if path.is_empty() || path.contains('\0') {
        return Err("document key path is invalid".into());
    }
    Ok((path, chunk))
}

fn search_vector(
    index: &F16Index<String>,
    query: &F16Vector<String>,
    top_k: usize,
    examples_dir: Option<&Path>,
    exhaustive: bool,
    exclude_key: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if exhaustive {
        println!(
            "{}",
            paint(
                "Full scan enabled; pruning is disabled.",
                "33",
                io::stdout().is_terminal()
            )
        );
    }

    let search_started = Instant::now();
    let searching = ProgressBar::new(0);
    searching.set_style(
        ProgressStyle::with_template(
            "[{wide_bar:.cyan/blue}] {percent:>3}% | elapsed {elapsed_precise}\n  {msg}",
        )?
        .progress_chars("=>-"),
    );
    searching.set_message(if exhaustive {
        "Starting exhaustive top-k search".to_owned()
    } else {
        "Starting top-k search".to_owned()
    });
    let mut pages_total = 0;
    let search_k = top_k.saturating_add(usize::from(exclude_key.is_some()));
    let on_progress = |state: SearchProgress| {
        pages_total = state.pages_total;
        searching.set_length(state.pages_total as u64);
        searching.set_position(state.pages_completed as u64);
        let pruned = state.stats.pruned_candidates();
        let pruning_percent = if state.stats.candidates == 0 {
            0.0
        } else {
            state.stats.pruning_ratio() * 100.0
        };
        let stage_rejections = format_stage_rejections(&state.stats.rejected_by_stage);
        searching.set_message(format!(
            "{}/{} pages | {} candidates\n  Initially pruned: {} | rejected by stage: {} | total: {} ({pruning_percent:.1}%)\n  Scored: {} | dot_page calls: {} | vector-segment dots: {}",
            state.pages_completed,
            state.pages_total,
            state.stats.candidates,
            state.stats.initially_pruned,
            stage_rejections,
            pruned,
            state.stats.scored_candidates,
            state.stats.page_segment_calls,
            state.stats.vector_segment_dots,
        ));
    };
    let (mut hits, stats) = if exhaustive {
        index.search_full_scan_with_progress(query, search_k, on_progress)?
    } else {
        index.search_with_progress(query, search_k, on_progress)?
    };
    if let Some(excluded) = exclude_key {
        hits.retain(|(key, _)| key != excluded);
        hits.truncate(top_k);
    }
    interrupt::check()?;
    let elapsed = search_started.elapsed();
    finish_progress(
        &searching,
        format!(
            "Search completed | {pages_total} pages | {:.2}s",
            elapsed.as_secs_f64()
        ),
    );
    println!(
        "{}",
        paint("Search statistics", "1;36", io::stdout().is_terminal())
    );
    println!("  Pages processed:      {pages_total}");
    println!("  Candidates:           {}", stats.candidates);
    println!("  Initially pruned:     {}", stats.initially_pruned);
    let stage_rejections = format_stage_rejections(&stats.rejected_by_stage);
    println!("  Rejected by stage:    {stage_rejections} (last stage: after all segments)");
    println!(
        "  Total pruned:         {} ({:.1}%)",
        stats.pruned_candidates(),
        stats.pruning_ratio() * 100.0
    );
    println!("  Scored candidates:    {}", stats.scored_candidates);
    println!("  dot_page calls:       {}", stats.page_segment_calls);
    println!("  Vector-segment dots:  {}", stats.vector_segment_dots);
    println!("  Search elapsed:       {:.3}s", elapsed.as_secs_f64());

    if hits.is_empty() {
        println!(
            "  {}",
            paint("No matches found.", "33", io::stdout().is_terminal())
        );
        return Ok(());
    }
    println!(
        "{}",
        paint(
            &format!(
                "{} {} of {top_k}",
                if exclude_key.is_some() {
                    "Similar documents:"
                } else {
                    "Top results:"
                },
                hits.len()
            ),
            "1;36",
            io::stdout().is_terminal()
        )
    );
    for (rank, (key, score)) in hits.iter().enumerate() {
        interrupt::check()?;
        let (relative, chunk) = decode_key(key);
        let number = paint(
            &format!("{:>2}.", rank + 1),
            "1;36",
            io::stdout().is_terminal(),
        );
        let score = paint(&format!("{score:.4}"), "1;32", io::stdout().is_terminal());
        println!("  {number} {score}  {relative}#{chunk}");
    }
    if let Some(directory) = examples_dir {
        println!("{}", paint("Examples", "1;33", io::stdout().is_terminal()));
        for (rank, (key, _)) in hits.iter().enumerate() {
            interrupt::check()?;
            let (relative, chunk) = decode_key(key);
            let label = paint(&format!("[{}]", rank + 1), "36", io::stdout().is_terminal());
            match read_preview(directory, relative, chunk) {
                Some(preview) => println!("  {label} {preview}"),
                None => println!("  {label} [source excerpt unavailable]"),
            }
        }
    }
    Ok(())
}

fn format_stage_rejections(counts: &[usize]) -> String {
    use std::fmt::Write as _;

    let mut result = String::new();
    for (stage, count) in counts.iter().enumerate() {
        if stage != 0 {
            result.push_str(", ");
        }
        let _ = write!(&mut result, "S{}={count}", stage + 1);
    }
    result
}
