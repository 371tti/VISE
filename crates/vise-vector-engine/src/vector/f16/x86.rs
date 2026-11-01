use crate::vector::{AlignedSegmentPage, PAGE_SIZE, SegmentDotKernel, SegmentVector};
use half::f16;
use std::arch::x86_64::*;

/// AVX2 + F16C kernel. Its constructor is private to runtime dispatch.
pub struct Avx2F16c {
    _private: (),
}

impl Avx2F16c {
    pub(super) fn new() -> Self {
        Self { _private: () }
    }
}

impl SegmentDotKernel<f16> for Avx2F16c {
    #[inline]
    fn dot(&self, a: &SegmentVector<f16>, b: &SegmentVector<f16>) -> f32 {
        // Construction is gated by detection of both required CPU features.
        unsafe { dot_avx2_f16c(a, b) }
    }

    #[inline]
    fn dot_page(
        &self,
        query: &SegmentVector<f16>,
        page: &AlignedSegmentPage<f16>,
        candidates: u64,
        scores: &mut [f32; PAGE_SIZE],
    ) {
        let mut remaining = candidates;
        while remaining != 0 {
            let slot = remaining.trailing_zeros() as usize;
            scores[slot] = self.dot(query, &page.0[slot]);
            remaining &= remaining - 1;
        }
    }
}

#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_avx2_f16c(a: &SegmentVector<f16>, b: &SegmentVector<f16>) -> f32 {
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    for i in (0..64).step_by(16) {
        // Two independent accumulators shorten the FP32 add dependency chain.
        let ap0 = unsafe { _mm_loadu_si128(a.as_ptr().add(i).cast()) };
        let bp0 = unsafe { _mm_loadu_si128(b.as_ptr().add(i).cast()) };
        let af0 = _mm256_cvtph_ps(ap0);
        let bf0 = _mm256_cvtph_ps(bp0);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(af0, bf0));

        let ap1 = unsafe { _mm_loadu_si128(a.as_ptr().add(i + 8).cast()) };
        let bp1 = unsafe { _mm_loadu_si128(b.as_ptr().add(i + 8).cast()) };
        let af1 = _mm256_cvtph_ps(ap1);
        let bf1 = _mm256_cvtph_ps(bp1);
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(af1, bf1));
    }
    let sum = _mm256_add_ps(sum0, sum1);
    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    lanes.into_iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::f16::Scalar;

    #[test]
    fn agrees_with_scalar_when_supported() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("f16c") {
            return;
        }
        let a = std::array::from_fn(|i| f16::from_f32((i as f32 - 27.0) / 19.0));
        let b = std::array::from_fn(|i| f16::from_f32((i as f32 % 13.0 - 6.0) / 7.0));
        let simd = Avx2F16c::new().dot(&a, &b);
        let scalar = Scalar.dot(&a, &b);
        assert!((simd - scalar).abs() < 2e-5, "{simd} vs {scalar}");
    }

    #[test]
    fn page_masks_slots() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("f16c") {
            return;
        }
        let query = [f16::from_f32(0.5); 64];
        let page = AlignedSegmentPage([[f16::from_f32(2.0); 64]; PAGE_SIZE]);
        let mut scores = [-3.0; PAGE_SIZE];
        Avx2F16c::new().dot_page(&query, &page, 1 << 5, &mut scores);
        assert!((scores[5] - 64.0).abs() < 1e-5);
        assert_eq!(scores[4], -3.0);
    }
}
