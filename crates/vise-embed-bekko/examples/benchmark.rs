use std::{env, time::Instant};

#[cfg(any(feature = "cuda", feature = "vulkan"))]
use burn::tensor::DeviceType;
use vise_embed_bekko::{BekkoEmbedder, BekkoModel, Device};

fn select_device(name: &str) -> Result<Device, Box<dyn std::error::Error>> {
    match name {
        "cpu" => {
            #[cfg(feature = "cpu")]
            {
                Ok(Device::flex())
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err("crate was built without the cpu feature".into())
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::enumerate(DeviceType::Cuda)
                    .first()
                    .cloned()
                    .ok_or("no CUDA device found")?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("crate was built without the cuda feature".into())
            }
        }
        "vulkan" => {
            #[cfg(feature = "vulkan")]
            {
                Ok(Device::enumerate(DeviceType::Vulkan)
                    .first()
                    .cloned()
                    .ok_or("no Vulkan device found")?)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err("crate was built without the vulkan feature".into())
            }
        }
        _ => Err("device must be cpu, cuda, or vulkan".into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: cargo run --release -p vise-embed-bekko --example benchmark -- <a8m|a25m> <cpu|cuda|vulkan> <weights.bpk> <tokenizer.json> [iterations] [batch-size] [text]"
        );
        std::process::exit(2);
    }
    let kind = match args[1].as_str() {
        "a8m" => BekkoModel::A8M,
        "a25m" => BekkoModel::A25M,
        _ => return Err("model must be a8m or a25m".into()),
    };
    let iterations = args
        .get(5)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(10usize);
    let batch_size = args
        .get(6)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(32usize);
    if iterations == 0 || batch_size == 0 {
        return Err("iterations and batch-size must be greater than zero".into());
    }
    let text = args
        .get(7)
        .map(String::as_str)
        .unwrap_or("日本語と English の sentence embedding benchmark.");
    let device = select_device(&args[2])?;
    println!("loading model {kind:?} on device {}", args[2]);
    let started = Instant::now();
    let model = BekkoEmbedder::from_files(kind, device, &args[3], &args[4])?;
    let load_time = started.elapsed();
    println!("model loaded in {load_time:.2?}");

    println!("encoding a single item with {kind:?}");
    model.encode(text)?;
    let started = Instant::now();
    for _ in 0..iterations {
        model.encode(text)?;
    }
    let single_time = started.elapsed();
    println!("completed {iterations} single-item iterations");

    println!("encoding a batch of {batch_size} items with {kind:?}");
    let batch = vec![text; batch_size];
    model.encode_batch(&batch)?;
    println!("model: {kind:?} / device: {}", args[2]);
    let started = Instant::now();
    for _ in 0..iterations {
        model.encode_batch(&batch)?;
        println!("completed batch of {batch_size} items");
    }
    let batch_time = started.elapsed();
    println!("completed {iterations} batch iterations");

    println!("load: {load_time:.2?}");
    println!(
        "single: {:.3} ms/item ({iterations} iterations)",
        single_time.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!(
        "batch: {:.3} ms/item ({batch_size} x {iterations} items)",
        batch_time.as_secs_f64() * 1000.0 / (batch_size * iterations) as f64
    );
    Ok(())
}
