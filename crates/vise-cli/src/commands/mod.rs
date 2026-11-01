mod benchmark;
mod documents;
mod embedding;
mod indexing;
mod search;
mod terminal;

use std::error::Error;

pub(super) use terminal::paint;

pub(super) fn run(command: crate::args::CliCommand) -> Result<(), Box<dyn Error>> {
    match command {
        crate::args::CliCommand::Index {
            documents_dir,
            index_file,
            recursive,
            kind: crate::args::IndexKind::Embedding,
            device,
            batch_size,
            threads: _,
            dry_run,
            cache_limit_bytes,
        } => indexing::index_directory(
            &documents_dir,
            &index_file,
            cache_limit_bytes,
            batch_size,
            recursive,
            dry_run,
            device,
        ),
        crate::args::CliCommand::Search {
            examples_dir,
            index_file,
            query,
            top_k,
            device,
            interactive,
            exhaustive,
            threads: _,
            cache_limit_bytes,
        } => search::search_directory(
            &index_file,
            examples_dir.as_deref(),
            query.as_deref(),
            top_k,
            cache_limit_bytes,
            device,
            interactive,
            exhaustive,
        ),
        crate::args::CliCommand::Compact { index_file } => indexing::compact_index(&index_file),
        crate::args::CliCommand::Bench {
            device,
            samples,
            threads: _,
        } => benchmark::benchmark_embeddings(device, samples),
        crate::args::CliCommand::Help(target) => {
            terminal::print_help(
                target
                    .map(crate::args::help_text_for)
                    .unwrap_or_else(crate::args::help_text),
            );
            Ok(())
        }
        crate::args::CliCommand::Version => {
            println!("vise {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
    }
}
