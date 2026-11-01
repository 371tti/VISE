use std::time::{Duration, SystemTime};

/// UTF-8 file size is only a rough multilingual token-count proxy.
const APPROX_BYTES_PER_TOKEN: u64 = 3;

/// Estimates token volume and completion time from byte-weighted progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressEstimate {
    pub total_tokens: u64,
    pub completed_tokens: u64,
    pub tokens_per_second: Option<f64>,
    pub eta: Option<Duration>,
    pub finish_at: Option<SystemTime>,
}

/// Estimates progress using the total and completed input sizes in bytes.
///
/// Token volume uses a rough multilingual estimate of one token per three UTF-8
/// bytes. `completed_bytes` is capped at `total_bytes`; rates and ETA are
/// unavailable until nonzero work has completed and elapsed time is nonzero.
pub fn estimate_progress(
    total_bytes: u64,
    completed_bytes: u64,
    elapsed: Duration,
    now: SystemTime,
) -> ProgressEstimate {
    let completed_bytes = completed_bytes.min(total_bytes);
    let total_tokens = approximate_tokens(total_bytes);
    let completed_tokens = approximate_tokens(completed_bytes);

    let tokens_per_second = (elapsed.as_secs_f64() > 0.0 && completed_bytes > 0)
        .then(|| (completed_bytes as f64 / APPROX_BYTES_PER_TOKEN as f64) / elapsed.as_secs_f64())
        .filter(|rate| rate.is_finite() && *rate > 0.0);

    let eta = if completed_bytes == total_bytes {
        Some(Duration::ZERO)
    } else {
        tokens_per_second.and_then(|rate| {
            let remaining = (total_bytes - completed_bytes) as f64 / APPROX_BYTES_PER_TOKEN as f64;
            duration_from_secs(remaining / rate)
        })
    };
    let finish_at = eta.and_then(|duration| now.checked_add(duration));

    ProgressEstimate {
        total_tokens,
        completed_tokens,
        tokens_per_second,
        eta,
        finish_at,
    }
}

fn approximate_tokens(bytes: u64) -> u64 {
    bytes.div_ceil(APPROX_BYTES_PER_TOKEN)
}

fn duration_from_secs(seconds: f64) -> Option<Duration> {
    if !seconds.is_finite() || seconds < 0.0 || seconds >= u64::MAX as f64 {
        return None;
    }
    Some(Duration::from_secs_f64(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_tokens_and_time_by_completed_bytes() {
        let estimate = estimate_progress(900, 300, Duration::from_secs(10), SystemTime::UNIX_EPOCH);
        assert_eq!(estimate.total_tokens, 300);
        assert_eq!(estimate.completed_tokens, 100);
        assert_eq!(estimate.tokens_per_second, Some(10.0));
        assert_eq!(estimate.eta, Some(Duration::from_secs(20)));
        assert_eq!(
            estimate.finish_at,
            Some(SystemTime::UNIX_EPOCH + Duration::from_secs(20))
        );
    }

    #[test]
    fn zero_work_and_zero_elapsed_are_well_defined() {
        let empty = estimate_progress(0, 0, Duration::ZERO, SystemTime::UNIX_EPOCH);
        assert_eq!(empty.total_tokens, 0);
        assert_eq!(empty.tokens_per_second, None);
        assert_eq!(empty.eta, Some(Duration::ZERO));
        assert_eq!(empty.finish_at, Some(SystemTime::UNIX_EPOCH));

        let starting = estimate_progress(30, 0, Duration::ZERO, SystemTime::UNIX_EPOCH);
        assert_eq!(starting.eta, None);
        assert_eq!(starting.finish_at, None);
    }

    #[test]
    fn caps_oversized_progress_and_handles_integer_limits() {
        let estimate = estimate_progress(
            u64::MAX,
            u64::MAX,
            Duration::from_secs(1),
            SystemTime::UNIX_EPOCH,
        );
        assert_eq!(estimate.completed_tokens, estimate.total_tokens);
        assert_eq!(estimate.eta, Some(Duration::ZERO));

        let capped =
            estimate_progress(30, u64::MAX, Duration::from_secs(1), SystemTime::UNIX_EPOCH);
        assert_eq!(capped.completed_tokens, 10);
        assert_eq!(capped.eta, Some(Duration::ZERO));
    }

    #[test]
    fn clock_overflow_does_not_panic() {
        let estimate = estimate_progress(
            u64::MAX,
            3,
            Duration::from_secs(u64::MAX / 2),
            SystemTime::UNIX_EPOCH,
        );
        assert!(estimate.finish_at.is_none());
    }

    #[test]
    fn rounded_token_counts_do_not_make_eta_zero_early() {
        let estimate = estimate_progress(5, 4, Duration::from_secs(4), SystemTime::UNIX_EPOCH);
        assert_eq!(estimate.eta, Some(Duration::from_secs(1)));
    }
}
