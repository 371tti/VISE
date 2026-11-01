//! Bekko sentence embeddings using generated Burn models.

use std::{fmt, path::Path};

use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{BurnpackStore, ModuleSnapshot};
use tokenizers::{PaddingStrategy, Tokenizer, utils::truncation::TruncationParams};

pub use burn::tensor::{Device, DeviceType};

/// Dimension of every embedding returned by this crate.
pub const EMBEDDING_DIM: usize = 384;
/// Maximum number of tokens processed from each input.
pub const MAX_SEQUENCE_LENGTH: usize = 8192;
/// A normalized Bekko embedding.
pub type Embedding = [f32; EMBEDDING_DIM];

#[cfg(feature = "embedded-assets")]
const TOKENIZER: &[u8] = include_bytes!("../../../model/bekko-embedding-v1-a8m/tokenizer.json");

#[cfg(all(feature = "embedded-assets", feature = "bekko-a8m"))]
#[repr(align(256))]
struct AlignedA8mWeights(
    [u8; include_bytes!("../../../model/bekko-embedding-v1-a8m/generated/model.bpk").len()],
);
#[cfg(all(feature = "embedded-assets", feature = "bekko-a8m"))]
static A8M_WEIGHTS: AlignedA8mWeights = AlignedA8mWeights(*include_bytes!(
    "../../../model/bekko-embedding-v1-a8m/generated/model.bpk"
));

#[cfg(all(feature = "embedded-assets", feature = "bekko-a25m"))]
#[repr(align(256))]
struct AlignedA25mWeights(
    [u8; include_bytes!("../../../model/bekko-embedding-v1-a25m/generated/model.bpk").len()],
);
#[cfg(all(feature = "embedded-assets", feature = "bekko-a25m"))]
static A25M_WEIGHTS: AlignedA25mWeights = AlignedA25mWeights(*include_bytes!(
    "../../../model/bekko-embedding-v1-a25m/generated/model.bpk"
));

#[cfg(feature = "bekko-a8m")]
#[allow(dead_code, clippy::all)]
mod a8m {
    include!("../../../model/bekko-embedding-v1-a8m/generated/model.rs");
}
#[cfg(feature = "bekko-a25m")]
#[allow(dead_code, clippy::all)]
mod a25m {
    include!("../../../model/bekko-embedding-v1-a25m/generated/model.rs");
}

/// Available model selection. A variant whose feature is disabled returns `UnsupportedModel`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BekkoModel {
    A8M,
    A25M,
}

impl BekkoModel {
    /// Returns whether this model was included by the crate's Cargo features.
    pub const fn is_enabled(self) -> bool {
        match self {
            Self::A8M => cfg!(feature = "bekko-a8m"),
            Self::A25M => cfg!(feature = "bekko-a25m"),
        }
    }
}

/// Loading, tokenization, or tensor execution failed.
#[derive(Debug)]
pub enum Error {
    UnsupportedModel(BekkoModel),
    MissingEmbeddedAssets,
    Io(std::io::Error),
    Tokenizer(String),
    Weights(String),
    Tensor(String),
    DeviceUnavailable,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedModel(model) => {
                write!(f, "model {model:?} is disabled by crate features")
            }
            Self::MissingEmbeddedAssets => {
                f.write_str("embedded assets are disabled; use from_files")
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Tokenizer(e) => write!(f, "tokenizer error: {e}"),
            Self::Weights(e) => write!(f, "model weights error: {e}"),
            Self::Tensor(e) => write!(f, "tensor error: {e}"),
            Self::DeviceUnavailable => f.write_str(
                "requested Burn device is not available in this build or on this machine",
            ),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

enum Network {
    #[cfg(feature = "bekko-a8m")]
    A8m(Box<a8m::Model>),
    #[cfg(feature = "bekko-a25m")]
    A25m(Box<a25m::Model>),
}

/// Loaded tokenizer and weights. Input tokenization truncates at 8192 tokens.
pub struct BekkoEmbedder {
    network: Network,
    tokenizer: Tokenizer,
    device: Device,
}

impl fmt::Debug for BekkoEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BekkoEmbedder")
            .field("model", &self.model())
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl BekkoEmbedder {
    /// Load a model from crate-embedded weights and tokenizer JSON.
    #[cfg(feature = "embedded-assets")]
    pub fn new(model: BekkoModel, device: Device) -> Result<Self, Error> {
        match model {
            BekkoModel::A8M => {
                #[cfg(feature = "bekko-a8m")]
                {
                    Self::from_bytes(model, &A8M_WEIGHTS.0, TOKENIZER, device)
                }
                #[cfg(not(feature = "bekko-a8m"))]
                {
                    Err(Error::UnsupportedModel(model))
                }
            }
            BekkoModel::A25M => {
                #[cfg(feature = "bekko-a25m")]
                {
                    Self::from_bytes(model, &A25M_WEIGHTS.0, TOKENIZER, device)
                }
                #[cfg(not(feature = "bekko-a25m"))]
                {
                    Err(Error::UnsupportedModel(model))
                }
            }
        }
    }

    /// Embedded loading is unavailable in builds without `embedded-assets`.
    #[cfg(not(feature = "embedded-assets"))]
    pub fn new(_model: BekkoModel, _device: Device) -> Result<Self, Error> {
        Err(Error::MissingEmbeddedAssets)
    }

    /// Load a model from external Burnpack weights and tokenizer JSON files.
    pub fn from_files(
        model: BekkoModel,
        device: Device,
        weights: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
    ) -> Result<Self, Error> {
        let mut tokenizer = Tokenizer::from_file(tokenizer.as_ref())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        configure_tokenizer(&mut tokenizer)?;
        std::fs::File::open(weights.as_ref())?;
        let network = load_network(model, Weights::File(weights.as_ref()), &device)?;
        Ok(Self {
            network,
            tokenizer,
            device,
        })
    }

    #[cfg(feature = "embedded-assets")]
    fn from_bytes(
        model: BekkoModel,
        weights: &'static [u8],
        tokenizer_json: &'static [u8],
        device: Device,
    ) -> Result<Self, Error> {
        let mut tokenizer =
            Tokenizer::from_bytes(tokenizer_json).map_err(|e| Error::Tokenizer(e.to_string()))?;
        configure_tokenizer(&mut tokenizer)?;
        let network = load_network(model, Weights::Bytes(weights), &device)?;
        Ok(Self {
            network,
            tokenizer,
            device,
        })
    }

    /// Produce an attention-masked mean-pooled and L2-normalized 384-dimensional vector.
    pub fn encode(&self, text: &str) -> Result<Embedding, Error> {
        let mut outputs = self.encode_batch(&[text])?;
        Ok(outputs.remove(0))
    }

    /// The model loaded by this embedder.
    pub fn model(&self) -> BekkoModel {
        match &self.network {
            #[cfg(feature = "bekko-a8m")]
            Network::A8m(_) => BekkoModel::A8M,
            #[cfg(feature = "bekko-a25m")]
            Network::A25m(_) => BekkoModel::A25M,
        }
    }

    /// The Burn device selected when this embedder was initialized.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Encode a batch of texts. Accepts borrowed strings and owned string types.
    /// Empty input returns an empty batch.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = texts.to_vec();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1)
            .max(1);
        let batch = texts.len();
        let mut ids = vec![0i64; batch * seq_len];
        let mut mask = vec![0i64; batch * seq_len];
        for (row, encoding) in encodings.iter().enumerate() {
            if !encoding.get_attention_mask().contains(&1) {
                return Err(Error::Tokenizer("input produced no unmasked tokens".into()));
            }
            for (col, &id) in encoding.get_ids().iter().enumerate() {
                if id >= 256_000 {
                    return Err(Error::Tokenizer("token ID exceeds model vocabulary".into()));
                }
                ids[row * seq_len + col] = i64::from(id);
                mask[row * seq_len + col] = i64::from(encoding.get_attention_mask()[col]);
            }
        }
        let input =
            Tensor::<2, Int>::from_data(TensorData::new(ids, [batch, seq_len]), &self.device);
        let attention =
            Tensor::<2, Int>::from_data(TensorData::new(mask, [batch, seq_len]), &self.device);
        let hidden = match &self.network {
            #[cfg(feature = "bekko-a8m")]
            Network::A8m(model) => model.forward(input, attention.clone()),
            #[cfg(feature = "bekko-a25m")]
            Network::A25m(model) => model.forward(input, attention.clone()),
        };
        // Pool and normalize on the selected device; only the compact [batch, 384] result is read back.
        let mask_f = attention.float();
        let weighted = hidden * mask_f.clone().unsqueeze_dims::<3>(&[2]);
        let pooled = weighted.sum_dim(1).squeeze_dim::<2>(1) / mask_f.sum_dim(1);
        let norm = pooled
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);
        let data = (pooled / norm)
            .try_into_data_as::<f32>()
            .map_err(|e| Error::Tensor(e.to_string()))?;
        let output = data
            .try_to_vec::<f32>()
            .map_err(|e| Error::Tensor(e.to_string()))?;
        if output.len() != batch * EMBEDDING_DIM {
            return Err(Error::Tensor(format!(
                "expected {} pooled values, got {}",
                batch * EMBEDDING_DIM,
                output.len()
            )));
        }
        let (rows, remainder) = output.as_chunks::<EMBEDDING_DIM>();
        if !remainder.is_empty() {
            return Err(Error::Tensor(
                "pooled embedding has an unexpected dimension".into(),
            ));
        }
        Ok(rows.to_vec())
    }
}

enum Weights<'a> {
    File(&'a Path),
    #[cfg(feature = "embedded-assets")]
    Bytes(&'static [u8]),
}

fn load_network(
    model: BekkoModel,
    weights: Weights<'_>,
    device: &Device,
) -> Result<Network, Error> {
    validate_device(device)?;
    match model {
        BekkoModel::A8M => {
            #[cfg(feature = "bekko-a8m")]
            {
                Ok(Network::A8m(Box::new(load_model(
                    a8m::Model::new(device),
                    &weights,
                )?)))
            }
            #[cfg(not(feature = "bekko-a8m"))]
            {
                let _ = (weights, device);
                Err(Error::UnsupportedModel(model))
            }
        }
        BekkoModel::A25M => {
            #[cfg(feature = "bekko-a25m")]
            {
                Ok(Network::A25m(Box::new(load_model(
                    a25m::Model::new(device),
                    &weights,
                )?)))
            }
            #[cfg(not(feature = "bekko-a25m"))]
            {
                let _ = (weights, device);
                Err(Error::UnsupportedModel(model))
            }
        }
    }
}

fn validate_device(requested: &Device) -> Result<(), Error> {
    use burn::backend::DispatchDevice;
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    use burn::backend::RuntimeId;
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    use burn::tensor::DeviceType;
    let device = requested.clone().without_autodiff();
    #[allow(unreachable_patterns)]
    match device.as_dispatch() {
        #[cfg(feature = "cpu")]
        DispatchDevice::Flex(_) => Ok(()),
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        DispatchDevice::Cube(cube_device) => {
            let filter = match cube_device.runtime() {
                #[cfg(feature = "cuda")]
                RuntimeId::Cuda => DeviceType::Cuda,
                #[cfg(feature = "vulkan")]
                RuntimeId::Wgpu => DeviceType::Vulkan,
                _ => return Err(Error::DeviceUnavailable),
            };
            if Device::enumerate(filter)
                .iter()
                .any(|available| available == &device)
            {
                Ok(())
            } else {
                Err(Error::DeviceUnavailable)
            }
        }
        _ => Err(Error::DeviceUnavailable),
    }
}

fn configure_tokenizer(tokenizer: &mut Tokenizer) -> Result<(), Error> {
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MAX_SEQUENCE_LENGTH,
            ..Default::default()
        }))
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let mut padding = tokenizer.get_padding().cloned().unwrap_or_default();
    padding.strategy = PaddingStrategy::BatchLongest;
    tokenizer.with_padding(Some(padding));
    Ok(())
}

fn load_model<M: ModuleSnapshot>(mut module: M, weights: &Weights<'_>) -> Result<M, Error> {
    let mut store = match weights {
        Weights::File(path) => BurnpackStore::from_file(path).auto_extension(false),
        #[cfg(feature = "embedded-assets")]
        Weights::Bytes(bytes) => BurnpackStore::from_static(bytes),
    };
    let applied = module
        .load_from(&mut store)
        .map_err(|e| Error::Weights(e.to_string()))?;
    if !applied.missing.is_empty() || !applied.errors.is_empty() {
        return Err(Error::Weights(format!(
            "incomplete or invalid weights; missing: {:?}; errors: {:?}",
            applied.missing, applied.errors
        )));
    }
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_truncates_at_model_limit_and_keeps_special_tokens() {
        let mut tokenizer = Tokenizer::from_bytes(include_bytes!(
            "../../../model/bekko-embedding-v1-a8m/tokenizer.json"
        ))
        .expect("bundled tokenizer is valid");
        configure_tokenizer(&mut tokenizer).expect("truncation config is valid");
        let text = "日本語".repeat(10_000);
        let encoding = tokenizer.encode(text, true).expect("text tokenizes");
        assert_eq!(encoding.get_ids().len(), 8192);
        assert_eq!(encoding.get_ids().last(), Some(&1));
    }
}
