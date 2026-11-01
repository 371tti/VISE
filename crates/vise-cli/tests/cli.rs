use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

use vise_vector_engine::F16Index;

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("vise-cli-test-{}-{id}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let (Ok(root), Ok(path)) = (
            fs::canonicalize(std::env::temp_dir()),
            fs::canonicalize(&self.0),
        ) && path.parent() == Some(root.as_path())
            && path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("vise-cli-test-"))
        {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vise"))
        .args(args)
        .output()
        .unwrap()
}

fn keys(index: &Path) -> Vec<String> {
    let mut keys = F16Index::<String>::open_file_backed(index).unwrap().keys();
    keys.sort();
    keys
}

#[test]
fn recursive_index_and_nonrecursive_update_preserve_nested_documents() {
    let temp = TempDir::new();
    let docs = temp.0.join("docs");
    fs::create_dir_all(docs.join("nested")).unwrap();
    fs::write(docs.join("root.txt"), "A short root document.").unwrap();
    fs::write(docs.join("nested/child.md"), "A nested document.").unwrap();
    let index = temp.0.join("index.vise");
    let docs_arg = docs.to_str().unwrap();
    let index_arg = index.to_str().unwrap();

    let output = run(&["index", "-er", docs_arg, index_arg, "--dry-run"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!index.exists());
    assert!(String::from_utf8_lossy(&output.stdout).contains("nested/child.md"));

    let output = run(&["index", "-er", docs_arg, index_arg, "--cache-mib=0"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let original = keys(&index);
    assert_eq!(original.len(), 2);
    assert!(
        original
            .iter()
            .any(|key| key.starts_with("nested/child.md\0"))
    );

    fs::write(docs.join("root.txt"), "Revised root document.").unwrap();
    let output = run(&["index", "-e", docs_arg, index_arg]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(keys(&index), original);

    fs::remove_file(docs.join("root.txt")).unwrap();
    let output = run(&["index", "-e", docs_arg, index_arg]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("use -r"));
    assert_eq!(keys(&index), original);

    let output = run(&[
        "search",
        index_arg,
        "nested",
        "-k",
        "1",
        "--cache-mib=0",
        "--threads",
        "1",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Top 1 of 1 results"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("CPU workers (Rayon): 1"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn unavailable_device_does_not_create_index() {
    let temp = TempDir::new();
    let docs = temp.0.join("docs");
    fs::create_dir(&docs).unwrap();
    fs::write(docs.join("example.txt"), "A test document.").unwrap();
    let index = temp.0.join("index.vise");
    let output = run(&[
        "index",
        docs.to_str().unwrap(),
        index.to_str().unwrap(),
        "--device",
        "cuda",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--features cuda"));
    assert!(!index.exists());
}
