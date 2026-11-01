# vise-vector-engine

64次元単位に正規化したベクトルをFP16で保存し、FP32内積とSegment枝刈りでTop-K検索するRust crateです。Scalarと実行時選択のAVX2+F16Cに対応します。

```rust
use vise_vector_engine::{F16Index, IndexError};

fn main() -> Result<(), IndexError> {
    let index = F16Index::<String>::on_memory();
    let vector = vec![1.0_f32; 384];
    index.insert_f32("doc-1".into(), &vector)?;
    let (hits, stats) = index.search_f32_with_stats(&vector, 10)?;
    println!("{}件; 枝刈り{}件", hits.len(), stats.pruned_candidates());
    Ok(())
}
```

永続化には`create_file_backed`と`open_file_backed`を使います。Segmentキャッシュの既定上限は256 MiBです。API、検索、同時更新、ファイル形式の詳細は[仕様書](../../SPEC.md)を参照してください。
