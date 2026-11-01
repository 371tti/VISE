mod args;
mod commands;
mod interrupt;
mod progress_estimate;

use std::{io, io::IsTerminal, process::ExitCode};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> ExitCode {
    let command = match args::parse_env() {
        Ok(command) => command,
        Err(error) => {
            eprintln!(
                "{}: {error}",
                commands::paint("error", "1;31", io::stderr().is_terminal())
            );
            return ExitCode::FAILURE;
        }
    };
    let index_operation = matches!(&command, args::CliCommand::Index { dry_run: false, .. });
    if matches!(
        &command,
        args::CliCommand::Index { .. }
            | args::CliCommand::Search { .. }
            | args::CliCommand::Bench { .. }
            | args::CliCommand::Compact { .. }
    ) && let Err(error) = interrupt::install()
    {
        eprintln!(
            "{} installing Ctrl+C handler: {error}",
            commands::paint("error", "1;31", io::stderr().is_terminal())
        );
        return ExitCode::FAILURE;
    }
    let pool_threads = match &command {
        args::CliCommand::Index {
            dry_run: false,
            threads,
            ..
        }
        | args::CliCommand::Search { threads, .. }
        | args::CliCommand::Bench { threads, .. } => Some(*threads),
        _ => None,
    };
    if let Some(threads) = pool_threads {
        if let Some(count) = threads
            && let Err(error) = rayon::ThreadPoolBuilder::new()
                .num_threads(count)
                .build_global()
        {
            eprintln!(
                "{} configuring CPU workers: {error}",
                commands::paint("error", "1;31", io::stderr().is_terminal())
            );
            return ExitCode::FAILURE;
        }
        eprintln!(
            "{} {}",
            commands::paint("CPU workers (Rayon):", "36", io::stderr().is_terminal()),
            rayon::current_num_threads()
        );
    }
    match commands::run(command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error)
            if error.is::<interrupt::Interrupted>()
                || (interrupt::is_requested()
                    && error
                        .downcast_ref::<io::Error>()
                        .is_some_and(|e| e.kind() == io::ErrorKind::Interrupted)) =>
        {
            eprintln!("Interrupted safely.");
            if index_operation {
                eprintln!("The index may contain partial updates; rerun index to synchronize it.");
            }
            ExitCode::from(130)
        }
        Err(error) => {
            eprintln!(
                "{}: {error}",
                commands::paint("error", "1;31", io::stderr().is_terminal())
            );
            ExitCode::FAILURE
        }
    }
}
