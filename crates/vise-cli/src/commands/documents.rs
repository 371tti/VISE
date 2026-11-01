use crate::interrupt;
use std::{
    error::Error,
    fs, io,
    path::{Component, Path, PathBuf},
};

pub(super) const CHUNK_CHARS: usize = 1200;
const CHUNK_OVERLAP: usize = 100;
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "txt",
    "md",
    "markdown",
    "rst",
    "adoc",
    "html",
    "htm",
    "xml",
    "json",
    "jsonl",
    "yaml",
    "yml",
    "toml",
    "ini",
    "conf",
    "log",
    "csv",
    "tsv",
    "rs",
    "py",
    "js",
    "jsx",
    "ts",
    "tsx",
    "java",
    "kt",
    "go",
    "c",
    "h",
    "cc",
    "cpp",
    "hpp",
    "cs",
    "swift",
    "sh",
    "sql",
    "css",
    "scss",
    "vue",
    "svelte",
    "graphql",
    "proto",
    "ipynb",
    "properties",
    "lock",
];

pub(super) fn collect_text_files(
    directory: &Path,
    recursive: bool,
    output: &mut Vec<PathBuf>,
) -> io::Result<()> {
    if interrupt::is_requested() {
        return Err(io::ErrorKind::Interrupted.into());
    }
    for entry in fs::read_dir(directory)? {
        if interrupt::is_requested() {
            return Err(io::ErrorKind::Interrupted.into());
        }
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() && recursive {
            collect_text_files(&path, recursive, output)?;
        } else if file_type.is_file() && is_supported_text_file(&path) {
            output.push(path);
        }
    }
    Ok(())
}

pub(super) fn read_text_file(path: &Path) -> io::Result<String> {
    fs::read_to_string(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("cannot read UTF-8 text file {}: {error}", path.display()),
        )
    })
}

pub(super) fn relative_path(root: &Path, path: &Path) -> Result<String, Box<dyn Error>> {
    let relative = path.strip_prefix(root)?.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("path is not valid Unicode: {}", path.display()),
        )
    })?;
    #[cfg(windows)]
    return Ok(relative.replace('\\', "/"));
    #[cfg(not(windows))]
    Ok(relative.to_owned())
}

pub(super) fn is_supported_text_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("Dockerfile" | "Makefile" | ".env")
    ) || path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            SUPPORTED_EXTENSIONS
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported))
        })
}

pub(super) fn text_chunks(text: &str) -> impl Iterator<Item = &str> {
    let mut start = 0;
    let mut empty_pending = text.is_empty();
    std::iter::from_fn(move || {
        if empty_pending {
            empty_pending = false;
            return Some("");
        }
        if start >= text.len() {
            return None;
        }

        let remaining = &text[start..];
        let mut hard_end = start;
        for (offset, ch) in remaining.char_indices().take(CHUNK_CHARS) {
            hard_end = start + offset + ch.len_utf8();
        }
        let end = if hard_end == text.len() {
            hard_end
        } else {
            text[start..hard_end]
                .char_indices()
                .rfind(|(_, ch)| ch.is_whitespace() || "。.!?！？".contains(*ch))
                .map(|(offset, ch)| start + offset + ch.len_utf8())
                .filter(|boundary| *boundary > start + CHUNK_CHARS / 2)
                .unwrap_or(hard_end)
        };

        let chunk_start = start;
        let chunk = &text[chunk_start..end];
        start = if end == text.len() {
            end
        } else {
            chunk
                .char_indices()
                .rev()
                .take(CHUNK_OVERLAP)
                .last()
                .map(|(offset, _)| chunk_start + offset)
                .filter(|next| *next > chunk_start)
                .unwrap_or(end)
        };
        Some(chunk)
    })
}

pub(super) fn decode_key(key: &str) -> (&str, usize) {
    key.rsplit_once('\0')
        .and_then(|(path, chunk)| chunk.parse().ok().map(|chunk| (path, chunk)))
        .unwrap_or((key, 0))
}

pub(super) fn is_cli_key(key: &str) -> bool {
    key.rsplit_once('\0')
        .is_some_and(|(path, chunk)| !path.is_empty() && chunk.parse::<usize>().is_ok())
}

pub(super) fn key_in_scope(key: &str, recursive: bool) -> bool {
    is_cli_key(key) && (recursive || !decode_key(key).0.contains('/'))
}

pub(super) fn read_preview(root: &Path, relative: &str, chunk: usize) -> Option<String> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return None;
    }
    let text = fs::read_to_string(root.join(path)).ok()?;
    let preview = text_chunks(&text).nth(chunk)?;
    let compact = preview.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let snippet = chars.by_ref().take(200).collect::<String>();
    Some(if chars.next().is_some() {
        format!("{snippet}…")
    } else {
        snippet
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_keep_unicode_boundaries_and_overlap() {
        let text = "あ".repeat(CHUNK_CHARS + 300);
        let chunks: Vec<_> = text_chunks(&text).collect();
        assert!(chunks.len() >= 2);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.chars().count() <= CHUNK_CHARS)
        );
        assert!(
            chunks
                .windows(2)
                .all(|pair| { pair[0].chars().last() == pair[1].chars().nth(CHUNK_OVERLAP - 1) })
        );
        assert_eq!(chunks[0].chars().count(), CHUNK_CHARS);
        assert_eq!(chunks[1].chars().count(), 300 + CHUNK_OVERLAP);
    }

    #[test]
    fn empty_file_has_one_chunk() {
        assert_eq!(text_chunks("").collect::<Vec<_>>(), [""]);
    }

    #[test]
    fn recognizes_common_text_and_source_files_case_insensitively() {
        assert!(is_supported_text_file(Path::new("notes.MD")));
        assert!(is_supported_text_file(Path::new("module.rs")));
        assert!(is_supported_text_file(Path::new("data.jsonl")));
        assert!(!is_supported_text_file(Path::new("image.png")));
    }

    #[test]
    fn recognizes_only_cli_chunk_keys_for_directory_sync() {
        assert!(is_cli_key(&format!("notes.md\0{}", 4)));
        assert!(!is_cli_key("notes.md"));
        assert!(!is_cli_key(&format!("\0{}", 1)));
        let nested = format!("sub/notes.md\0{}", 1);
        assert!(!key_in_scope(&nested, false));
        assert!(key_in_scope(&nested, true));
        assert!(key_in_scope(&format!("notes.md\0{}", 0), false));
    }
}
