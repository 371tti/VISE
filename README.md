# vise

viseは、vector searchのためのRustクレート群です

あとまだ作成中だし遅いので待っててね　遊べはするけど

## 構成

| Crate | 役割 |
| --- | --- |
| [`vise-embed-bekko`](crates/vise-embed-bekko/README.md) | Bekko A8M/A25Mによる384次元埋め込み |
| [`vise-vector-engine`](crates/vise-vector-engine/README.md) | メモリ内・FileBackedのベクトル検索 |
| [`vise-cli`](crates/vise-cli/README.md) | ディレクトリのインデックス作成、検索、compact、bench |
| `vise` | `vise::vector` APIと`docs/`を使う実行例 |

## CLI

```powershell
cargo build --release -p vise-cli
cargo run --release -p vise-cli -- index -e -r ./documents ./data/index.vise
cargo run --release -p vise-cli -- search ./data/index.vise "検索語" -k 5
```

既定のCLIビルドにはCPU・CUDA・Vulkanを含みます。コマンド、既定値、デバイス指定は[CLIガイド](crates/vise-cli/README.md)を参照してください。