use std::path::PathBuf;

#[cfg(any(feature = "cuda", feature = "vulkan"))]
use burn::tensor::DeviceType;
#[cfg(all(feature = "cpu", feature = "bekko-a8m"))]
use vise_embed_bekko::Error;
use vise_embed_bekko::{BekkoEmbedder, BekkoModel, Device};

fn model_files(kind: BekkoModel) -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model");
    let dir = match kind {
        BekkoModel::A8M => "bekko-embedding-v1-a8m",
        BekkoModel::A25M => "bekko-embedding-v1-a25m",
    };
    let dir = root.join(dir);
    (dir.join("generated/model.bpk"), dir.join("tokenizer.json"))
}

fn assert_embedding(embedding: &[f32]) {
    assert_eq!(embedding.len(), 384, "embedding dimension");
    assert!(
        embedding.iter().all(|v| v.is_finite()),
        "embedding contains NaN or infinity"
    );
    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "expected L2 norm 1, got {norm}");
}

fn check_model_on(kind: BekkoModel, device: Device) {
    let (weights, tokenizer) = model_files(kind);
    assert!(kind.is_enabled());
    let embedder = BekkoEmbedder::from_files(kind, device.clone(), weights, tokenizer).unwrap();
    assert_eq!(embedder.model(), kind);
    assert_eq!(embedder.device(), &device);
    let texts = [
        "",
        "A short sentence for embedding.",
        "日本語の文章と Unicode 🦀 café.",
        "A longer but still small input checks ordinary multi-token inference.",
    ];

    let batch = embedder.encode_batch(&texts).unwrap();
    assert_eq!(batch.len(), texts.len());
    for embedding in &batch {
        assert_embedding(embedding);
    }
    assert!(embedder.encode_batch(&[]).unwrap().is_empty());
    let single = embedder.encode(texts[1]).unwrap();
    assert_embedding(&single);
}

#[cfg(feature = "cpu")]
#[cfg(feature = "bekko-a8m")]
#[test]
fn a8m_cpu_batch_and_unicode() {
    check_model_on(BekkoModel::A8M, Device::flex());
}

#[cfg(feature = "cpu")]
#[cfg(feature = "bekko-a25m")]
#[test]
fn a25m_cpu_batch_and_unicode() {
    check_model_on(BekkoModel::A25M, Device::flex());
}

#[cfg(all(feature = "cpu", feature = "bekko-a8m", feature = "embedded-assets"))]
#[test]
fn embedded_a8m_loads_and_encodes() {
    let embedder = BekkoEmbedder::new(BekkoModel::A8M, Device::flex()).unwrap();
    assert_embedding(&embedder.encode("埋め込み").unwrap());
}

#[cfg(all(feature = "cpu", feature = "bekko-a25m", feature = "embedded-assets"))]
#[test]
fn embedded_a25m_loads_and_encodes() {
    let embedder = BekkoEmbedder::new(BekkoModel::A25M, Device::flex()).unwrap();
    assert_embedding(&embedder.encode("埋め込み").unwrap());
}

#[cfg(any(feature = "cuda", feature = "vulkan"))]
fn first_gpu(backend: DeviceType) -> Device {
    Device::enumerate(backend.clone())
        .first()
        .cloned()
        .unwrap_or_else(|| panic!("no {backend:?} device is available"))
}

#[cfg(all(feature = "cuda", feature = "bekko-a8m"))]
#[test]
#[ignore = "run explicitly on a host with a configured CUDA runtime"]
fn cuda_a8m_inference() {
    check_model_on(BekkoModel::A8M, first_gpu(DeviceType::Cuda));
}

#[cfg(all(feature = "cuda", feature = "bekko-a25m"))]
#[test]
#[ignore = "run explicitly on a host with a configured CUDA runtime"]
fn cuda_a25m_inference() {
    check_model_on(BekkoModel::A25M, first_gpu(DeviceType::Cuda));
}

#[cfg(all(feature = "vulkan", feature = "bekko-a8m"))]
#[test]
#[ignore = "run explicitly on a host with an initialized Vulkan runtime"]
fn vulkan_a8m_inference() {
    check_model_on(BekkoModel::A8M, first_gpu(DeviceType::Vulkan));
}

#[cfg(all(feature = "vulkan", feature = "bekko-a25m"))]
#[test]
#[ignore = "run explicitly on a host with an initialized Vulkan runtime"]
fn vulkan_a25m_inference() {
    check_model_on(BekkoModel::A25M, first_gpu(DeviceType::Vulkan));
}

#[cfg(all(feature = "cpu", feature = "bekko-a8m"))]
#[test]
fn reports_missing_model_and_tokenizer_files() {
    let missing = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/does-not-exist");
    let (weights, tokenizer) = model_files(BekkoModel::A8M);
    assert!(matches!(
        BekkoEmbedder::from_files(BekkoModel::A8M, Device::flex(), &missing, &missing),
        Err(Error::Tokenizer(_))
    ));
    let missing_weights = missing.with_extension("bpk");
    assert!(matches!(
        BekkoEmbedder::from_files(
            BekkoModel::A8M,
            Device::flex(),
            missing_weights,
            tokenizer.clone()
        ),
        Err(Error::Io(_))
    ));
    let wrong_weights = weights
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("model.onnx");
    assert!(matches!(
        BekkoEmbedder::from_files(
            BekkoModel::A8M,
            Device::flex(),
            wrong_weights,
            tokenizer.clone()
        ),
        Err(Error::Weights(_))
    ));
    assert!(weights.exists());
}
