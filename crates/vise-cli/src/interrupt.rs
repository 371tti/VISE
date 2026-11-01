use std::{
    fmt,
    io::Write,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Default)]
struct InterruptFlag(AtomicBool);

impl InterruptFlag {
    fn request(&self) -> bool {
        !self.0.swap(true, Ordering::Relaxed)
    }

    fn is_requested(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

static INTERRUPTED: InterruptFlag = InterruptFlag(AtomicBool::new(false));

#[derive(Debug)]
pub struct Interrupted;

impl fmt::Display for Interrupted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("interrupted")
    }
}
impl std::error::Error for Interrupted {}

/// The ctrlc crate invokes this callback on a dedicated thread.
pub fn install() -> Result<(), ctrlc::Error> {
    ctrlc::set_handler(|| {
        if INTERRUPTED.request() {
            let _ = writeln!(
                std::io::stderr(),
                "\nCtrl+C received; stopping after the current operation completes..."
            );
        }
    })
}

/// Returns whether Ctrl+C has been requested.
pub fn is_requested() -> bool {
    INTERRUPTED.is_requested()
}

pub fn check() -> Result<(), Interrupted> {
    if is_requested() {
        Err(Interrupted)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::InterruptFlag;

    #[test]
    fn flag_can_be_set_and_read_across_threads() {
        let flag = std::sync::Arc::new(InterruptFlag::default());
        assert!(!flag.is_requested());

        let worker_flag = std::sync::Arc::clone(&flag);
        std::thread::spawn(move || worker_flag.request())
            .join()
            .unwrap();

        assert!(flag.is_requested());
    }
}
