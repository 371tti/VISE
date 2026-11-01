# vise-embed-bekko

Japanese and multilingual sentence embeddings using the Bekko ONNX models converted to Burn. Each embedding is the attention-mask-aware mean of token states, then L2-normalized. Output dimension is 384; tokenizer input is capped at 8192 tokens.

## Embedded model

With the default features, the selected model and tokenizer are embedded into the binary:

```rust,no_run
use vise_embed_bekko::{BekkoEmbedder, BekkoModel, Device, Embedding};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedder = BekkoEmbedder::new(BekkoModel::A8M, Device::flex())?;
    let embedding: Embedding = embedder.encode("日本語の文章")?;
    println!("using {:?} on {:?}", embedder.model(), embedder.device());
    Ok(())
}
```

## Load local model files

The repository keeps model files outside this crate. `from_files` loads a generated Burnpack (`model.bpk`) and the matching `tokenizer.json`, so applications can choose where to store model data.

```rust,no_run
use vise_embed_bekko::{BekkoEmbedder, BekkoModel, Device};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = BekkoEmbedder::from_files(
        BekkoModel::A8M,
        Device::flex(),
        "model/bekko-embedding-v1-a8m/generated/model.bpk",
        "model/bekko-embedding-v1-a8m/tokenizer.json",
    )?;
    let vector = model.encode("日本語の文章を検索します")?;
    assert_eq!(vector.len(), 384);
    Ok(())
}
```

Choose `BekkoModel::A8M` or `BekkoModel::A25M` to select the smaller or larger model. `BekkoModel::is_enabled()` reports whether its Cargo feature is enabled. The embedder keeps its model, tokenizer, and device private; `model()` and `device()` expose read-only information, and `Debug` reports those same settings without exposing model internals. A successfully constructed embedder is ready to encode, so there is no separate mutable readiness state. `Embedding`, `EMBEDDING_DIM`, and `MAX_SEQUENCE_LENGTH` describe the output and input limits. `encode_batch(&[&str])` returns one vector per input and accepts an empty slice.

## Features

The default features enable the A8M model, its embedded weights, and the Flex CPU backend. Add `bekko-a25m`, `cuda`, or `vulkan` only when needed. CUDA and Vulkan enable Burn fusion automatically. For external model files, disable default features and enable only the model and backend needed, for example:

```toml
vise-embed-bekko = { path = "../vise-embed-bekko", default-features = false, features = ["bekko-a8m", "cpu"] }
```

Feature names are `bekko-a8m`, `bekko-a25m`, `embedded-assets`, `cpu`, `cuda`, and `vulkan`. `cpu` selects Burn Flex, its SIMD/Rayon CPU backend, through `Device::flex()`. CUDA requires a compatible CUDA runtime; Vulkan requires a compatible Vulkan device/driver.

Models excluded by features return `Error::UnsupportedModel`. If embedded assets are disabled, `new` returns `Error::MissingEmbeddedAssets` and `from_files` can be used instead. An unavailable requested device returns `Error::DeviceUnavailable`; Burn or generated model code can still panic for internal execution failures.

## Run tests

From the workspace root, tests use the existing local `model/` assets and also verify embedded loading when that feature is enabled:

```powershell
$env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\Temp\vise-bekko-target"
cargo test -p vise-embed-bekko
```

CPU inference tests cover both models, vector shape and normalization, batch behavior, Unicode, tokenizer truncation, and file-loading errors. GPU inference tests are ignored in the default run so CPU-only CI does not try to initialize GPU runtimes. Run one backend at a time on a host with that runtime configured:

```powershell
cargo test -p vise-embed-bekko --no-default-features --features bekko-a8m,bekko-a25m,cpu,vulkan --test embedding vulkan_ -- --ignored
cargo test -p vise-embed-bekko --no-default-features --features bekko-a8m,bekko-a25m,cpu,cuda --test embedding cuda_ -- --ignored
```

## Benchmark

The example reports model load time, warmed single-item latency, and 32-item batch latency on CPU. It excludes the first inference from timed results.

```powershell
cargo run --release -p vise-embed-bekko --example benchmark -- a8m cpu model/bekko-embedding-v1-a8m/generated/model.bpk model/bekko-embedding-v1-a8m/tokenizer.json 100 32
```

The positional arguments after the tokenizer are iteration count, batch size, and optional text. Select `cpu`, `cuda`, or `vulkan` based on the crate's enabled features. Benchmark numbers depend on device, build flags, and input length; compare runs on the same machine and configuration.
