//! 64次元ごとの区分を使ったFP16ベクトルインデックスと検索。

pub mod vector;

pub use vector::{CacheStats, F16Index, F16Vector, IndexError, SearchStats};
