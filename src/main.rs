use std::error::Error;

use vise_embed_bekko::{BekkoEmbedder, BekkoModel, Device};
use vise_vector_engine::vector::VectorIndex;

const DOCUMENTS: [(&str, &str); 4] = [
    (
        "rust",
        "Rustは安全性と速度を重視したシステムプログラミング言語です。",
    ),
    (
        "python",
        "Pythonはデータ分析や機械学習によく使われるプログラミング言語です。",
    ),
    (
        "vector-search",
        "ベクトル検索は文章の意味が近い文書を探す方法です。",
    ),
    (
        "tokyo-weather",
        "東京は晴れていて、日中は気温が上がる予報です。",
    ),
];

fn main() -> Result<(), Box<dyn Error>> {
    let embedder = BekkoEmbedder::new(BekkoModel::A8M, Device::flex())?;
    let index = VectorIndex::<String, _>::on_memory();

    for (key, text) in DOCUMENTS {
        let embedding = embedder.encode(text)?;
        index.insert_f32(key.to_owned(), &embedding)?;
    }

    let query = "文章の意味が似ているものを検索する";
    let query_embedding = embedder.encode(query)?;

    println!("Query: {query}");
    for (rank, (key, score)) in index.search_f32(&query_embedding, 3)?.iter().enumerate() {
        println!("{}. {score:.4}  {key}", rank + 1);
    }

    Ok(())
}
