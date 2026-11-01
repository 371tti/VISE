# vise-cli

テキストディレクトリからBekko A8MのEmbedding Indexを作成・更新し、検索するCLIです。入力はUTF-8テキストです。Markdown、JSON/YAML/TOML、CSV/TSV、HTML/XML、一般的なソースコードなどに対応します。文書は最大1200文字、100文字重複で分割します。

## 基本操作

```powershell
cargo build --release -p vise-cli
cargo run --release -p vise-cli -- index -e -r ./documents ./data/index.vise
cargo run --release -p vise-cli -- search ./data/index.vise "検索語" -k 5
cargo run --release -p vise-cli -- search ./data/index.vise -i -x ./documents
cargo run --release -p vise-cli -- compact ./data/index.vise
cargo run --release -p vise-cli -- bench
```

`index`は既存インデックスを更新し、正常終了時に古いページ履歴をcompactします。`-r`を省略すると直下のファイルだけを走査し、既存のサブディレクトリエントリは残します。`--dry-run`はモデルやインデックスを変更せず、対象ファイルを表示します。

## オプション

| 指定 | 用途・既定値 |
| --- | --- |
| `-e`, `--embed` | Embedding Indexを選択。現在の唯一の種類で、既定もEmbeddingです |
| `-r`, `--recursive` | サブディレクトリも走査。既定は無効 |
| `-d`, `--device` | `auto`, `cpu`, `cuda`, `vulkan`。既定の自動選択順はCUDA、Vulkan、CPU |
| `-b`, `--batch-size` | 推論バッチ幅。既定はCPU 32、CUDA/Vulkan 16 |
| `-t`, `--threads` | Rayon worker数。既定はRayon設定またはCPU数 |
| `-c`, `--segment-cache-mib` | 保持するFileBacked Segmentキャッシュ。既定256 MiB、`0`で無効 |
| `--dry-run` | 対象ファイルだけを表示 |
| `-k`, `--top-k` | 検索結果数。既定5 |
| `-i`, `--interactive` | 最初の検索語の後も入力を続ける。検索語を省略した場合は対話検索 |
| `-f`, `--exhaustive` | 枝刈りせず全ベクトルを評価 |
| `-x`, `--examples DIR` | 結果一覧の下に文書抜粋を表示 |
| `-n`, `--samples` | `bench`の合成文書数。既定32 |

Segmentキャッシュ上限はプロセス全体のメモリ上限ではありません。`-t`はCPU worker数の設定で、プロセス全体のCPU使用率を制限しません。

## 類似文書検索と結果表示

検索結果のキー`relative/path#chunk`をそのままコピーし、先頭に`>`を付けると、その保存ベクトルで類似文書を検索できます。PowerShellなどでは`>`を引用符で囲んでください。`>>`で始めると、`>`を含む通常の検索語として扱います。

```powershell
cargo run --release -p vise-cli -- search ./data/index.vise '>notes.md#0'
```

`-x <documents-dir>`を指定すると、結果番号に対応した抜粋を一覧の後に表示します。検索結果には候補・枝刈り・内積数などの統計と進捗が表示されます。インデックス作成ではdocuments/s、tokens/s、文書あたりの時間、ETAを更新表示します。Ctrl+Cは進行中の処理を待って停止します。

対話検索は`:q`、`quit`、`exit`で終了します。詳しいオプションは`vise help index`、`vise help search`、`vise help compact`、`vise help bench`で確認できます。

## バックエンド

通常のCLIビルドにはCPU・CUDA・Vulkan featureが含まれます。個別に絞る場合は次のようにします。

```powershell
cargo build --release -p vise-cli --no-default-features --features cpu
cargo build --release -p vise-cli --no-default-features --features cuda
cargo build --release -p vise-cli --no-default-features --features vulkan
```

CUDAには対応するCUDA Toolkitが必要です。`libdevice.10.bc`が見つからない場合は、`CUDA_PATH`を`nvvm/libdevice/libdevice.10.bc`を含むToolkitのルートに設定してください。x86-64-v3専用バイナリは次の設定でビルドできます。そのバイナリには対応CPUが必要です。

```powershell
$env:RUSTFLAGS = "-C target-cpu=x86-64-v3"
cargo build --release -p vise-cli
```
