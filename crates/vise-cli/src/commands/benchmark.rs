use super::{
    documents::CHUNK_CHARS,
    embedding::{device_name, try_select_device},
    terminal::{finish_progress, start_spinner},
};
use crate::{args, interrupt};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    error::Error,
    time::{Duration, Instant},
};
use vise_embed_bekko::{BekkoEmbedder, BekkoModel};

const BENCH_BATCH_SIZES: [usize; 6] = [1, 2, 4, 8, 16, 32];
const BENCH_ROUNDS: usize = 3;

pub(super) fn benchmark_embeddings(
    requested: args::DeviceKind,
    sample_count: usize,
) -> Result<(), Box<dyn Error>> {
    let automatic = requested == args::DeviceKind::Auto;
    let devices = if automatic {
        vec![
            args::DeviceKind::Cuda,
            args::DeviceKind::Vulkan,
            args::DeviceKind::Cpu,
        ]
    } else {
        vec![requested]
    };
    let texts = dummy_benchmark_texts(sample_count);
    let text_refs: Vec<_> = texts.iter().map(String::as_str).collect();
    let mut measured_any = false;
    let mut failures = Vec::new();
    let mut overall_best: Option<(args::DeviceKind, usize, f64)> = None;

    println!(
        "Bekko A8M embedding benchmark | {} synthetic texts, up to {CHUNK_CHARS} characters each | batches: 1, 2, 4, 8, 16, 32",
        text_refs.len()
    );
    println!(
        "Each batch gets one warm-up and {BENCH_ROUNDS} timed passes; model loading and index writes are excluded."
    );

    for kind in devices {
        interrupt::check()?;
        let device = match try_select_device(kind) {
            Ok(device) => device,
            Err(error) if automatic => {
                println!("{} unavailable: {error}", device_name(kind));
                failures.push(format!("{}: {error}", device_name(kind)));
                continue;
            }
            Err(error) => return Err(error.into()),
        };

        let loading = start_spinner(format!("Loading Bekko A8M on {}", device_name(kind)));
        let load_started = Instant::now();
        let embedder = match BekkoEmbedder::new(BekkoModel::A8M, device) {
            Ok(embedder) => embedder,
            Err(error) if automatic => {
                finish_progress(
                    &loading,
                    format!("{} failed to load: {error}", device_name(kind)),
                );
                failures.push(format!("{} load: {error}", device_name(kind)));
                continue;
            }
            Err(error) => {
                finish_progress(
                    &loading,
                    format!("{} failed to load: {error}", device_name(kind)),
                );
                return Err(error.into());
            }
        };
        finish_progress(
            &loading,
            format!(
                "{} model ready in {:.2}s",
                device_name(kind),
                load_started.elapsed().as_secs_f64()
            ),
        );

        let mut best: Option<(usize, f64)> = None;
        let progress = ProgressBar::new(BENCH_BATCH_SIZES.len() as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "[{wide_bar:.cyan/blue}] {pos}/{len} batch sizes | elapsed {elapsed_precise}\n  {msg}",
            )?
            .progress_chars("=>-"),
        );
        progress.enable_steady_tick(Duration::from_millis(200));
        for batch_size in BENCH_BATCH_SIZES {
            interrupt::check()?;
            progress.set_message(format!("warming and measuring batch {batch_size}"));
            if batch_size > sample_count {
                println!("  batch {batch_size}: skipped (requires at least {batch_size} samples)");
                progress.inc(1);
                continue;
            }
            if let Err(error) = embedder.encode_batch(&text_refs[..batch_size]) {
                let reason = format!("{} batch {batch_size}: {error}", device_name(kind));
                eprintln!("  {reason}; skipping remaining batch sizes for this device");
                failures.push(reason);
                break;
            }

            let mut failure = None;
            let mut rates = Vec::with_capacity(BENCH_ROUNDS);
            for _ in 0..BENCH_ROUNDS {
                interrupt::check()?;
                let started = Instant::now();
                for batch in text_refs.chunks(batch_size) {
                    if let Err(error) = embedder.encode_batch(batch) {
                        failure = Some(error.to_string());
                        break;
                    }
                }
                if failure.is_some() {
                    break;
                }
                let seconds = started.elapsed().as_secs_f64().max(f64::EPSILON);
                rates.push(text_refs.len() as f64 / seconds);
            }
            if let Some(error) = failure {
                let reason = format!("{} batch {batch_size}: {error}", device_name(kind));
                eprintln!("  {reason}; skipping remaining batch sizes for this device");
                failures.push(reason);
                break;
            }

            rates.sort_by(|left, right| left.total_cmp(right));
            let texts_per_second = rates[rates.len() / 2];
            let milliseconds_per_text = 1_000.0 / texts_per_second;
            println!(
                "  batch {batch_size:>2}: {texts_per_second:>8.1} texts/s median ({:.1}–{:.1}) | {milliseconds_per_text:>7.2} ms/text",
                rates[0],
                rates[rates.len() - 1]
            );
            if best.is_none_or(|(_, fastest)| texts_per_second > fastest) {
                best = Some((batch_size, texts_per_second));
            }
            measured_any = true;
            progress.inc(1);
        }
        finish_progress(
            &progress,
            format!("{} benchmark pass complete", device_name(kind)),
        );

        if let Some((batch_size, rate)) = best {
            println!(
                "  Best measured for {}: batch {batch_size} ({rate:.1} texts/s); set it with `--device {} --batch-size {batch_size}`.",
                device_name(kind),
                device_name(kind).to_lowercase()
            );
            if overall_best.is_none_or(|(_, _, fastest)| rate > fastest) {
                overall_best = Some((kind, batch_size, rate));
            }
        }
    }

    if !measured_any {
        return Err(format!(
            "no embedding backend completed the benchmark ({})",
            failures.join("; ")
        )
        .into());
    }
    println!(
        "\nUse the measured batch size as a starting point; repeat the run when results are close."
    );
    if let Some((kind, batch_size, rate)) = overall_best {
        println!(
            "Fastest overall: {} batch {batch_size} ({rate:.1} texts/s); set it with `--device {} --batch-size {batch_size}`.",
            device_name(kind),
            device_name(kind).to_lowercase()
        );
    }
    Ok(())
}

fn dummy_benchmark_texts(count: usize) -> Vec<String> {
    const SAMPLE: &str = " Bekko embedding benchmark 用の合成テキストです。The quick brown fox jumps over the lazy dog. ";

    (0..count)
        .map(|sample_id| {
            let mut text = format!("Synthetic sample {sample_id}:");
            let mut chars = text.chars().count();
            while chars < CHUNK_CHARS {
                text.push_str(SAMPLE);
                chars += SAMPLE.chars().count();
            }
            text.chars().take(CHUNK_CHARS).collect()
        })
        .collect()
}
