use crate::progress_estimate;
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    env,
    error::Error,
    io::{self, IsTerminal},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub(super) fn format_finish_time(time: Option<SystemTime>) -> Option<String> {
    let seconds = time?.duration_since(UNIX_EPOCH).ok()?.as_secs();
    let utc = chrono::DateTime::from_timestamp(i64::try_from(seconds).ok()?, 0)?;
    Some(
        utc.with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M:%S %Z")
            .to_string(),
    )
}

pub(super) fn start_spinner(message: String) -> ProgressBar {
    if !io::stderr().is_terminal() {
        eprintln!("{message}...");
    }
    let progress = ProgressBar::new_spinner();
    progress.set_message(message);
    progress.enable_steady_tick(Duration::from_millis(100));
    progress
}

pub(super) fn finish_progress(progress: &ProgressBar, message: String) {
    let message = paint(&message, "32", io::stderr().is_terminal());
    if io::stderr().is_terminal() {
        progress.finish_with_message(message);
    } else {
        progress.finish_and_clear();
        eprintln!("{message}");
    }
}

pub(super) fn color_enabled(terminal: bool) -> bool {
    terminal && env::var_os("NO_COLOR").is_none()
}

pub(crate) fn paint(text: &str, style: &str, terminal: bool) -> String {
    if color_enabled(terminal) {
        format!("\x1b[{style}m{text}\x1b[0m")
    } else {
        text.to_owned()
    }
}

pub(super) fn print_help(help: &str) {
    let terminal = color_enabled(io::stdout().is_terminal());
    for line in help.lines() {
        let trimmed = line.trim_start();
        let indent = &line[..line.len() - trimmed.len()];
        if matches!(
            trimmed,
            "Commands:" | "Options:" | "Examples:" | "Defaults:"
        ) || trimmed.starts_with("Usage:")
        {
            println!("{}", paint(line, "1;36", terminal));
        } else if trimmed.starts_with("vise - ") {
            println!("{}", paint(line, "1;35", terminal));
        } else if let Some((option, description)) = trimmed
            .split_once("  ")
            .filter(|(option, _)| option.starts_with('-'))
        {
            println!(
                "{indent}{}  {description}",
                paint(option.trim_end(), "36", terminal)
            );
        } else if let Some((command, description)) =
            trimmed.split_once("    ").filter(|(command, _)| {
                matches!(*command, "index" | "search" | "compact" | "bench" | "help")
            })
        {
            println!(
                "{indent}{}    {description}",
                paint(command, "32", terminal)
            );
        } else {
            println!("{line}");
        }
    }
}

pub(super) fn document_progress(
    total_docs: usize,
    total_bytes: u64,
) -> Result<ProgressBar, Box<dyn Error>> {
    if !io::stderr().is_terminal() {
        eprintln!("Indexing {total_docs} documents...");
    }
    let progress = ProgressBar::new(total_bytes);
    progress.set_style(
        ProgressStyle::with_template(
            "[{wide_bar:.cyan/blue}] {percent:>3}% | elapsed {elapsed_precise}\n  {msg}",
        )?
        .progress_chars("=>-"),
    );
    let estimated_tokens =
        progress_estimate::estimate_progress(total_bytes, 0, Duration::ZERO, SystemTime::now())
            .total_tokens;
    progress.set_message(format!("0/{total_docs} docs | ~{estimated_tokens} tokens estimated from file sizes\n  estimating rate and remaining time"));
    progress.enable_steady_tick(Duration::from_millis(200));
    Ok(progress)
}

pub(super) fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 3600 {
        format!("{}h {:02}m", seconds / 3600, seconds % 3600 / 60)
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_display_scales_to_hours() {
        assert_eq!(format_duration(Duration::from_secs(9)), "9s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m 05s");
        assert_eq!(format_duration(Duration::from_secs(3660)), "1h 01m");
    }
}
