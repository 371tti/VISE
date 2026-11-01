use crate::args;
use std::error::Error;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
use vise_embed_bekko::DeviceType;
use vise_embed_bekko::{BekkoEmbedder, BekkoModel, Device};

pub(super) fn device_name(kind: args::DeviceKind) -> &'static str {
    match kind {
        args::DeviceKind::Auto => "Auto",
        args::DeviceKind::Cpu => "CPU",
        args::DeviceKind::Cuda => "CUDA",
        args::DeviceKind::Vulkan => "Vulkan",
    }
}

pub(super) fn default_embed_batch_size(kind: args::DeviceKind) -> usize {
    match kind {
        args::DeviceKind::Cpu => 32,
        args::DeviceKind::Cuda => 16,
        args::DeviceKind::Vulkan => 16,
        args::DeviceKind::Auto => unreachable!("auto is resolved before choosing the batch size"),
    }
}

pub(super) fn load_embedder(
    requested: args::DeviceKind,
) -> Result<(BekkoEmbedder, args::DeviceKind), Box<dyn Error>> {
    let automatic = requested == args::DeviceKind::Auto;
    let order = if automatic {
        [
            args::DeviceKind::Cuda,
            args::DeviceKind::Vulkan,
            args::DeviceKind::Cpu,
        ]
    } else {
        [requested; 3]
    };
    let mut failures = Vec::new();
    for kind in order {
        if !automatic && kind != requested {
            break;
        }
        let device = match try_select_device(kind) {
            Ok(device) => device,
            Err(error) if automatic => {
                failures.push(format!("{}: {error}", device_name(kind)));
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        match BekkoEmbedder::new(BekkoModel::A8M, device) {
            Ok(embedder) => return Ok((embedder, kind)),
            Err(error) if automatic => {
                failures.push(format!(
                    "{} initialization failed: {error}",
                    device_name(kind)
                ));
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err(format!(
        "no compute device could load Bekko A8M ({})",
        failures.join("; ")
    )
    .into())
}

pub(super) fn try_select_device(kind: args::DeviceKind) -> Result<Device, String> {
    match kind {
        args::DeviceKind::Auto => Err("automatic selection is resolved by the caller".into()),
        args::DeviceKind::Cpu => {
            #[cfg(feature = "cpu")]
            {
                Ok(Device::flex())
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err("CPU backend is not enabled; rebuild with --features cpu".into())
            }
        }
        args::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Device::enumerate(DeviceType::Cuda)
                    .into_iter()
                    .next()
                    .ok_or_else(|| "no CUDA device available".to_owned())
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA backend is not enabled; rebuild with --features cuda".into())
            }
        }
        args::DeviceKind::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                Device::enumerate(DeviceType::Vulkan)
                    .into_iter()
                    .next()
                    .ok_or_else(|| "no Vulkan device available".to_owned())
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err("Vulkan backend is not enabled; rebuild with --features vulkan".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_defaults_follow_the_selected_device() {
        assert_eq!(default_embed_batch_size(args::DeviceKind::Cpu), 32);
        assert_eq!(default_embed_batch_size(args::DeviceKind::Cuda), 16);
        assert_eq!(default_embed_batch_size(args::DeviceKind::Vulkan), 16);
    }
}
