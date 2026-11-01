//! Command-line argument parsing for `vise`.
//!
//! This module deliberately uses only the standard library so the binary and
//! its parser tests do not need a separate argument-parsing dependency.

use std::{env, error::Error, fmt, path::PathBuf};

/// Parsed top-level command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliCommand {
    Index {
        documents_dir: PathBuf,
        index_file: PathBuf,
        recursive: bool,
        kind: IndexKind,
        device: DeviceKind,
        batch_size: Option<usize>,
        threads: Option<usize>,
        dry_run: bool,
        cache_limit_bytes: Option<usize>,
    },
    Search {
        examples_dir: Option<PathBuf>,
        index_file: PathBuf,
        query: Option<String>,
        top_k: usize,
        device: DeviceKind,
        interactive: bool,
        exhaustive: bool,
        threads: Option<usize>,
        cache_limit_bytes: Option<usize>,
    },
    Compact {
        index_file: PathBuf,
    },
    Bench {
        device: DeviceKind,
        samples: usize,
        threads: Option<usize>,
    },
    Help(Option<HelpTarget>),
    Version,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexKind {
    Embedding,
}

/// Compute device requested for embedding and search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DeviceKind {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Vulkan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HelpTarget {
    Index,
    Search,
    Compact,
    Bench,
}

/// A user-facing command-line syntax or value error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ParseError {}

/// Parse arguments excluding the executable name.
pub fn parse_from<I, S>(args: I) -> Result<CliCommand, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    match args.first().map(String::as_str) {
        None | Some("--help" | "-h") => Ok(CliCommand::Help(None)),
        Some("help") => match args.get(1).map(String::as_str) {
            None => Ok(CliCommand::Help(None)),
            Some("index") if args.len() == 2 => Ok(CliCommand::Help(Some(HelpTarget::Index))),
            Some("search") if args.len() == 2 => Ok(CliCommand::Help(Some(HelpTarget::Search))),
            Some("compact") if args.len() == 2 => Ok(CliCommand::Help(Some(HelpTarget::Compact))),
            Some("bench") if args.len() == 2 => Ok(CliCommand::Help(Some(HelpTarget::Bench))),
            Some(target) => Err(ParseError(format!(
                "unknown help topic `{target}`; expected `index`, `search`, `compact`, or `bench`"
            ))),
        },
        Some("--version" | "-V") => Ok(CliCommand::Version),
        Some("index") => parse_index(&args[1..]),
        Some("search") => parse_search(&args[1..]),
        Some("compact") => parse_compact(&args[1..]),
        Some("bench") => parse_bench(&args[1..]),
        Some(command) => Err(ParseError(format!(
            "unknown command `{command}`\n\n{}",
            help_text()
        ))),
    }
}

pub fn help_text_for(target: HelpTarget) -> &'static str {
    match target {
        HelpTarget::Index => {
            r#"Usage: vise index <documents-dir> <index-file> [-r] [-e] [-d DEVICE] [-b N] [-t N] [-c MiB] [--dry-run]

Options:
  -e, --embed                 Create an embedding index (default: Bekko A8M)
  -r, --recursive             Include subdirectories (default: off)
  -d, --device DEVICE         auto, cpu, cuda, or vulkan (default: auto; CUDA > Vulkan > CPU)
  -b, --batch-size N          Texts per inference batch (default: CPU 32, CUDA 16, Vulkan 16; 1 encodes singly)
  -t, --threads N             Rayon workers for CPU embedding and vector search (default: Rayon setting / available CPUs)
  -c, --segment-cache-mib N   Retained FileBacked Segment cache (default: 256 MiB; 0 disables)
      --dry-run               Preview only; no model or index changes (default: off)
  -h, --help                  Show this help

The cache limit applies only to retained Segment data, not total process memory.
Existing entries are synchronized within the selected scan scope. Without -r,
previously indexed subdirectory entries are retained.

Example: vise index ./docs ./docs/index.vise -er"#
        }
        HelpTarget::Search => {
            r#"Usage: vise search <index-file> [<query>] [-k N] [-d DEVICE] [-i] [-x <documents-dir>] [-t N] [-c MiB]

Options:
  -k, --top-k N               Maximum results (default: 5)
  -d, --device DEVICE         auto, cpu, cuda, or vulkan (default: auto; CUDA > Vulkan > CPU)
  -i, --interactive           Keep prompting after the first query (default: off;
                              omitting <query> starts interactive search)
  -f, --exhaustive            Evaluate every vector without pruning (default: off)
  -x, --examples DIR          Show excerpts from DIR after the ranked result list
                              (default: off; DIR is the original documents directory)
  -t, --threads N             Rayon workers for vector search and CPU query embedding (default: Rayon setting / available CPUs)
  -c, --segment-cache-mib N   Retained FileBacked Segment cache (default: 256 MiB; 0 disables)
  -h, --help                  Show this help

The cache limit applies only to retained Segment data, not total process memory.
Search results print reusable keys as `<relative-path>#<chunk-index>` without `>`.
Prefix that exact key with `>` to search using its stored vector (for example,
`">notes.md#0"`). Prefix with `>>` to search literal text that starts with `>`.
The key's chunk index is zero-based. Quote `>` queries in a shell because it is
also a redirection operator.
Enter :q, quit, or exit to stop interactive search.

Examples:
  vise search ./docs/index.vise "vector search" -k 10
  vise search ./docs/index.vise -i -x ./docs
  vise search ./docs/index.vise "vector search" --exhaustive"#
        }
        HelpTarget::Compact => {
            r#"Usage: vise compact <index-file>

Rewrite a FileBacked index using only its current pages and reclaim space left by
previous page updates. Stop other processes using this index before compacting.

Example: vise compact ./docs/index.vise"#
        }
        HelpTarget::Bench => {
            r#"Usage: vise bench [-d DEVICE] [-n N] [-t N]

Measure Bekko A8M embedding throughput with generated 1,200-character texts.
Each available device is loaded once, warmed up, then tested at batch sizes
1, 2, 4, 8, 16, and 32. No files or index are created.

Options:
  -d, --device DEVICE   auto benchmarks every available backend; or select cpu, cuda, or vulkan (default: auto)
  -n, --samples N       Synthetic texts per batch size (default: 32)
  -t, --threads N       Rayon workers for CPU embedding (default: Rayon setting / available CPUs)
  -h, --help            Show this help

Results report texts/second and the fastest measured batch size for each device
and overall.
Each candidate is timed three times; the median is reported. Tokenization and
embedding are included, while model loading and index writes are excluded.

Example: vise bench --device cpu --samples 64"#
        }
    }
}

/// Parse the current process arguments, excluding the executable name.
pub fn parse_env() -> Result<CliCommand, ParseError> {
    let args = env::args_os()
        .skip(1)
        .map(|arg| {
            arg.into_string()
                .map_err(|_| ParseError("arguments must be valid Unicode".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    parse_from(args)
}

pub fn help_text() -> &'static str {
    r#"vise - local file embedding and vector search

Usage:
  vise index <documents-dir> <index-file> [options]
  vise search <index-file> [<query>] [options]
  vise compact <index-file>
  vise bench [options]
  vise help [index|search|compact|bench]
  vise --help | --version

Commands:
  index    Create or update an embedding index
  search   Search an existing index
  compact  Reclaim unused space in an index file
  bench    Measure embedding throughput and batch-size choices
  help     Show help for a command

Defaults: Bekko A8M, embedding batch size by device (CPU 32, CUDA 16, Vulkan 16), top 5 results, 256 MiB retained Segment cache, and automatic
device selection (CUDA > Vulkan > CPU). The default build includes all three devices.
Index scans only the named directory; add -r to include subdirectories.
Search without a query starts interactive mode. Add -x <documents-dir> for excerpts.
Use `>path#chunk` to search by an indexed vector and `--exhaustive` to disable pruning.
Use `vise help <command>` for command details.
Terminal colors are disabled when output is redirected or NO_COLOR is set."#
}

fn parse_index(args: &[String]) -> Result<CliCommand, ParseError> {
    if has_help(args) {
        return Ok(CliCommand::Help(Some(HelpTarget::Index)));
    }
    let (positionals, options) = split_options(args, 2, 0, "index")?;
    let mut recursive = false;
    let mut kind = IndexKind::Embedding;
    let mut saw_embed = false;
    let mut dry_run = false;
    let mut cache_limit_bytes = None;
    let mut device = DeviceKind::default();
    let mut saw_device = false;
    let mut batch_size = None;
    let mut saw_batch_size = false;
    let mut threads = None;
    let mut i = 0;
    while i < options.len() {
        match options[i] {
            "-r" | "--recursive" => {
                if recursive {
                    return Err(ParseError("duplicate option `--recursive`".into()));
                }
                recursive = true;
                i += 1;
            }
            "-e" | "--embed" => {
                if saw_embed {
                    return Err(ParseError("duplicate option `--embed`".into()));
                }
                saw_embed = true;
                kind = IndexKind::Embedding;
                i += 1;
            }
            "--dry-run" => {
                if dry_run {
                    return Err(ParseError("duplicate option `--dry-run`".into()));
                }
                dry_run = true;
                i += 1;
            }
            "-c" | "--segment-cache-mib" | "--cache-mib" => {
                let value = option_value(&options, i, "--segment-cache-mib")?;
                if cache_limit_bytes.is_some() {
                    return Err(ParseError("duplicate option `--segment-cache-mib`".into()));
                }
                cache_limit_bytes = Some(parse_cache_mib(value)?);
                i += 2;
            }
            "-d" | "--device" => {
                let value = option_value(&options, i, "--device")?;
                if saw_device {
                    return Err(ParseError("duplicate option `--device`".into()));
                }
                device = parse_device(value)?;
                saw_device = true;
                i += 2;
            }
            "-b" | "--batch-size" => {
                let value = option_value(&options, i, "--batch-size")?;
                if saw_batch_size {
                    return Err(ParseError("duplicate option `--batch-size`".into()));
                }
                batch_size = Some(parse_positive_usize(value, "--batch-size")?);
                saw_batch_size = true;
                i += 2;
            }
            "-t" | "--threads" => {
                let value = option_value(&options, i, "--threads")?;
                if threads.is_some() {
                    return Err(ParseError("duplicate option `--threads`".into()));
                }
                threads = Some(parse_positive_usize(value, "--threads")?);
                i += 2;
            }
            option => return Err(ParseError(format!("unexpected index option `{option}`"))),
        }
    }
    Ok(CliCommand::Index {
        documents_dir: PathBuf::from(&positionals[0]),
        index_file: PathBuf::from(&positionals[1]),
        recursive,
        kind,
        device,
        batch_size,
        threads,
        dry_run,
        cache_limit_bytes,
    })
}

fn parse_search(args: &[String]) -> Result<CliCommand, ParseError> {
    if has_help(args) {
        return Ok(CliCommand::Help(Some(HelpTarget::Search)));
    }
    let (positionals, options) = split_options(args, 1, 1, "search")?;
    let mut examples_dir = None;
    let mut top_k = 5;
    let mut cache_limit_bytes = None;
    let mut device = DeviceKind::default();
    let mut saw_device = false;
    let mut interactive = false;
    let mut exhaustive = false;
    let mut threads = None;
    let mut saw_top_k = false;
    let mut i = 0;
    while i < options.len() {
        match options[i] {
            "-k" | "--top-k" => {
                let value = option_value(&options, i, "--top-k")?;
                if saw_top_k {
                    return Err(ParseError("duplicate option `--top-k`".into()));
                }
                top_k = parse_usize(value, "--top-k")?;
                saw_top_k = true;
                i += 2;
            }
            "-c" | "--segment-cache-mib" | "--cache-mib" => {
                let value = option_value(&options, i, "--segment-cache-mib")?;
                if cache_limit_bytes.is_some() {
                    return Err(ParseError("duplicate option `--segment-cache-mib`".into()));
                }
                cache_limit_bytes = Some(parse_cache_mib(value)?);
                i += 2;
            }
            "-d" | "--device" => {
                let value = option_value(&options, i, "--device")?;
                if saw_device {
                    return Err(ParseError("duplicate option `--device`".into()));
                }
                device = parse_device(value)?;
                saw_device = true;
                i += 2;
            }
            "-t" | "--threads" => {
                let value = option_value(&options, i, "--threads")?;
                if threads.is_some() {
                    return Err(ParseError("duplicate option `--threads`".into()));
                }
                threads = Some(parse_positive_usize(value, "--threads")?);
                i += 2;
            }
            "-i" | "--interactive" => {
                if interactive {
                    return Err(ParseError("duplicate option `--interactive`".into()));
                }
                interactive = true;
                i += 1;
            }
            "-f" | "--exhaustive" => {
                if exhaustive {
                    return Err(ParseError("duplicate option `--exhaustive`".into()));
                }
                exhaustive = true;
                i += 1;
            }
            "-x" | "--examples" => {
                let value = option_value(&options, i, "--examples <documents-dir>")?;
                if examples_dir.is_some() {
                    return Err(ParseError("duplicate option `--examples`".into()));
                }
                examples_dir = Some(PathBuf::from(value));
                i += 2;
            }
            option => return Err(ParseError(format!("unexpected search option `{option}`"))),
        }
    }
    Ok(CliCommand::Search {
        examples_dir,
        index_file: PathBuf::from(&positionals[0]),
        query: positionals.get(1).map(|query| (*query).to_owned()),
        top_k,
        device,
        interactive,
        exhaustive,
        threads,
        cache_limit_bytes,
    })
}

fn parse_compact(args: &[String]) -> Result<CliCommand, ParseError> {
    if has_help(args) {
        return Ok(CliCommand::Help(Some(HelpTarget::Compact)));
    }
    let (positionals, options) = split_options(args, 1, 0, "compact")?;
    if let Some(option) = options.first() {
        return Err(ParseError(format!("unexpected compact option `{option}`")));
    }
    Ok(CliCommand::Compact {
        index_file: PathBuf::from(positionals[0]),
    })
}

fn parse_bench(args: &[String]) -> Result<CliCommand, ParseError> {
    if has_help(args) {
        return Ok(CliCommand::Help(Some(HelpTarget::Bench)));
    }
    let (positionals, options) = split_options(args, 0, 0, "bench")?;
    if let Some(positional) = positionals.first() {
        return Err(ParseError(format!(
            "unexpected bench argument `{positional}`; see `vise help bench`"
        )));
    }
    let mut device = DeviceKind::Auto;
    let mut saw_device = false;
    let mut samples = 32;
    let mut saw_samples = false;
    let mut threads = None;
    let mut i = 0;
    while i < options.len() {
        match options[i] {
            "-d" | "--device" => {
                let value = option_value(&options, i, "--device")?;
                if saw_device {
                    return Err(ParseError("duplicate option `--device`".into()));
                }
                device = parse_device(value)?;
                saw_device = true;
                i += 2;
            }
            "-n" | "--samples" => {
                let value = option_value(&options, i, "--samples")?;
                if saw_samples {
                    return Err(ParseError("duplicate option `--samples`".into()));
                }
                samples = parse_positive_usize(value, "--samples")?;
                saw_samples = true;
                i += 2;
            }
            "-t" | "--threads" => {
                let value = option_value(&options, i, "--threads")?;
                if threads.is_some() {
                    return Err(ParseError("duplicate option `--threads`".into()));
                }
                threads = Some(parse_positive_usize(value, "--threads")?);
                i += 2;
            }
            option => return Err(ParseError(format!("unexpected bench option `{option}`"))),
        }
    }
    Ok(CliCommand::Bench {
        device,
        samples,
        threads,
    })
}

fn has_help(args: &[String]) -> bool {
    for arg in args {
        if arg == "--" {
            return false;
        }
        if arg == "--help" || arg == "-h" {
            return true;
        }
    }
    false
}

fn split_options<'a>(
    args: &'a [String],
    required: usize,
    optional: usize,
    command: &str,
) -> Result<(Vec<&'a str>, Vec<&'a str>), ParseError> {
    let mut positionals = Vec::with_capacity(required + optional);
    let mut options = Vec::new();
    let mut i = 0;
    let mut positional_only = false;
    while i < args.len() {
        let arg = args[i].as_str();
        if positional_only {
            positionals.push(arg);
        } else if arg == "--" {
            positional_only = true;
        } else if let Some((name, value)) = arg.split_once('=').filter(|(name, _)| {
            matches!(
                *name,
                "--cache-mib"
                    | "--segment-cache-mib"
                    | "--top-k"
                    | "--device"
                    | "--threads"
                    | "--batch-size"
                    | "--examples"
                    | "--samples"
            )
        }) {
            options.push(name);
            options.push(value);
        } else if arg.starts_with('-') && !arg.starts_with("--") && arg.len() > 2 {
            for short in arg[1..].chars() {
                match short {
                    'r' => options.push("-r"),
                    'e' => options.push("-e"),
                    'i' => options.push("-i"),
                    'f' => options.push("-f"),
                    _ => return Err(ParseError(format!("unknown short option `-{short}`"))),
                }
            }
        } else if matches!(
            arg,
            "-r" | "--recursive"
                | "-e"
                | "--embed"
                | "--dry-run"
                | "-c"
                | "-d"
                | "-b"
                | "-t"
                | "--cache-mib"
                | "--segment-cache-mib"
                | "--top-k"
                | "--device"
                | "--threads"
                | "--examples"
                | "--samples"
                | "-i"
                | "--interactive"
                | "-f"
                | "--exhaustive"
                | "-x"
                | "-k"
                | "-n"
        ) {
            options.push(arg);
            if matches!(
                arg,
                "-c" | "-d"
                    | "-b"
                    | "-t"
                    | "-x"
                    | "--cache-mib"
                    | "--segment-cache-mib"
                    | "--top-k"
                    | "--device"
                    | "--threads"
                    | "--batch-size"
                    | "--examples"
                    | "--samples"
                    | "-k"
                    | "-n"
            ) {
                let value = args
                    .get(i + 1)
                    .filter(|value| !value.starts_with('-'))
                    .ok_or_else(|| ParseError(format!("option `{arg}` requires a value")))?;
                options.push(value);
                i += 1;
            }
        } else if arg.starts_with('-') {
            options.push(arg);
        } else {
            positionals.push(arg);
        }
        i += 1;
    }
    if positionals.len() < required || positionals.len() > required + optional {
        let expected = if optional == 0 {
            required.to_string()
        } else {
            format!("{required} or {}", required + optional)
        };
        return Err(ParseError(format!(
            "{command} expects {expected} positional arguments, got {}\n\n{}",
            positionals.len(),
            help_text()
        )));
    }
    Ok((positionals, options))
}

fn parse_device(value: &str) -> Result<DeviceKind, ParseError> {
    match value {
        "auto" => Ok(DeviceKind::Auto),
        "cpu" => Ok(DeviceKind::Cpu),
        "cuda" => Ok(DeviceKind::Cuda),
        "vulkan" => Ok(DeviceKind::Vulkan),
        _ => Err(ParseError(format!(
            "invalid value `{value}` for `--device`: expected auto, cpu, cuda, or vulkan"
        ))),
    }
}

fn option_value<'a>(options: &'a [&str], index: usize, name: &str) -> Result<&'a str, ParseError> {
    options
        .get(index + 1)
        .copied()
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| ParseError(format!("option `{name}` requires a value")))
}

fn parse_cache_mib(value: &str) -> Result<usize, ParseError> {
    parse_usize(value, "--segment-cache-mib")?
        .checked_mul(1024 * 1024)
        .ok_or_else(|| ParseError(format!("cache limit `{value}` MiB is too large")))
}

fn parse_usize(value: &str, option: &str) -> Result<usize, ParseError> {
    value.parse().map_err(|_| {
        ParseError(format!(
            "invalid value `{value}` for `{option}`: expected a non-negative integer"
        ))
    })
}

fn parse_positive_usize(value: &str, option: &str) -> Result<usize, ParseError> {
    let parsed = parse_usize(value, option)?;
    if parsed == 0 {
        return Err(ParseError(format!(
            "invalid value `0` for `{option}`: expected a positive integer"
        )));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn index_options_support_short_and_long_spellings() {
        assert_eq!(
            parse_from(strings(&[
                "index",
                "docs",
                "idx",
                "-r",
                "-e",
                "-b",
                "4",
                "--cache-mib",
                "2"
            ]))
            .unwrap(),
            CliCommand::Index {
                documents_dir: "docs".into(),
                index_file: "idx".into(),
                recursive: true,
                kind: IndexKind::Embedding,
                device: DeviceKind::Auto,
                batch_size: Some(4),
                threads: None,
                dry_run: false,
                cache_limit_bytes: Some(2 * 1024 * 1024),
            }
        );
        assert!(matches!(
            parse_from(strings(
                &["index", "docs", "idx", "--recursive", "--embed",]
            )),
            Ok(CliCommand::Index {
                recursive: true,
                kind: IndexKind::Embedding,
                ..
            })
        ));
        assert!(matches!(
            parse_from(strings(&["index", "docs", "idx"])),
            Ok(CliCommand::Index {
                batch_size: None,
                ..
            })
        ));
        assert!(matches!(
            parse_from(strings(&["index", "docs", "idx", "--batch-size=12"])),
            Ok(CliCommand::Index {
                batch_size: Some(12),
                ..
            })
        ));
        assert!(parse_from(strings(&["index", "docs", "idx", "-b", "0"])).is_err());
        assert!(matches!(
            parse_from(strings(&["index", "docs", "idx", "-er"])),
            Ok(CliCommand::Index {
                recursive: true,
                kind: IndexKind::Embedding,
                ..
            })
        ));
    }

    #[test]
    fn search_defaults_and_explicit_options_are_parsed() {
        assert!(matches!(
            parse_from(strings(&["search", "idx", "two words"])),
            Ok(CliCommand::Search { top_k: 5, device: DeviceKind::Auto, threads: None, cache_limit_bytes: None, query, .. }) if query.as_deref() == Some("two words")
        ));
        assert!(matches!(
            parse_from(strings(&[
                "search",
                "idx",
                "query",
                "--top-k",
                "7",
                "--cache-mib",
                "0"
            ])),
            Ok(CliCommand::Search {
                top_k: 7,
                cache_limit_bytes: Some(0),
                ..
            })
        ));
        assert!(matches!(
            parse_from(strings(&[
                "search",
                "-k",
                "3",
                "idx",
                "query",
                "--cache-mib=0"
            ])),
            Ok(CliCommand::Search {
                top_k: 3,
                cache_limit_bytes: Some(0),
                ..
            })
        ));
    }

    #[test]
    fn device_selection_is_parsed_for_each_subcommand() {
        assert!(matches!(
            parse_from(strings(&["index", "docs", "idx", "--device", "cuda"])),
            Ok(CliCommand::Index {
                device: DeviceKind::Cuda,
                ..
            })
        ));
        assert!(matches!(
            parse_from(strings(&["search", "idx", "q", "--device=vulkan"])),
            Ok(CliCommand::Search {
                device: DeviceKind::Vulkan,
                ..
            })
        ));
        assert!(
            parse_from(strings(&["search", "idx", "q", "--device", "metal"]))
                .unwrap_err()
                .to_string()
                .contains("expected auto, cpu, cuda, or vulkan")
        );
        assert!(
            parse_from(strings(&[
                "index", "docs", "idx", "--device", "cpu", "--device", "cuda"
            ]))
            .unwrap_err()
            .to_string()
            .contains("duplicate option `--device`")
        );
    }

    #[test]
    fn thread_count_is_optional_positive_and_shared_by_subcommands() {
        assert!(matches!(
            parse_from(strings(&["index", "docs", "idx", "--threads", "4"])),
            Ok(CliCommand::Index {
                threads: Some(4),
                ..
            })
        ));
        assert!(matches!(
            parse_from(strings(&["search", "idx", "q", "--threads=2"])),
            Ok(CliCommand::Search {
                threads: Some(2),
                ..
            })
        ));
        assert!(
            parse_from(strings(&["index", "docs", "idx", "--threads", "0"]))
                .unwrap_err()
                .to_string()
                .contains("expected a positive integer")
        );
        assert!(
            parse_from(strings(&["search", "idx", "q", "--threads", "many"]))
                .unwrap_err()
                .to_string()
                .contains("expected a non-negative integer")
        );
    }

    #[test]
    fn help_and_version_are_recognized() {
        assert_eq!(parse_from(strings(&[])).unwrap(), CliCommand::Help(None));
        assert_eq!(
            parse_from(strings(&["-h"])).unwrap(),
            CliCommand::Help(None)
        );
        assert_eq!(
            parse_from(strings(&["search", "--help"])).unwrap(),
            CliCommand::Help(Some(HelpTarget::Search))
        );
        assert_eq!(
            parse_from(strings(&["help", "index"])).unwrap(),
            CliCommand::Help(Some(HelpTarget::Index))
        );
        assert!(matches!(
            parse_from(strings(&["search", "idx", "--", "--help"])),
            Ok(CliCommand::Search { query, .. }) if query.as_deref() == Some("--help")
        ));
        assert_eq!(
            parse_from(strings(&["--version"])).unwrap(),
            CliCommand::Version
        );
    }

    #[test]
    fn errors_explain_missing_unknown_and_invalid_options() {
        assert!(
            parse_from(strings(&["index", "docs"]))
                .unwrap_err()
                .to_string()
                .contains("expects 2 positional")
        );
        assert!(
            parse_from(strings(&["search", "idx", "q", "--top-k"]))
                .unwrap_err()
                .to_string()
                .contains("requires a value")
        );
        assert!(
            parse_from(strings(&["index", "docs", "idx", "--bogus"]))
                .unwrap_err()
                .to_string()
                .contains("unexpected index option")
        );
        assert!(
            parse_from(strings(&["search", "idx", "q", "--other"]))
                .unwrap_err()
                .to_string()
                .contains("unexpected search option")
        );
    }
}
