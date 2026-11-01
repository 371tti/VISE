use crate::vector::{AlignedSegmentPage, PAGE_SIZE, SegmentDotKernel, SegmentVector};
use half::f16;

/// Portable FP32-accumulating FP16 dot product.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scalar;

impl SegmentDotKernel<f16> for Scalar {
    #[inline]
    fn dot(&self, a: &SegmentVector<f16>, b: &SegmentVector<f16>) -> f32 {
        a.iter().zip(b).map(|(x, y)| x.to_f32() * y.to_f32()).sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_page_only_writes_candidate_slots() {
        let query = [f16::from_f32(0.5); 64];
        let page = AlignedSegmentPage([[f16::from_f32(2.0); 64]; PAGE_SIZE]);
        let mut scores = [-7.0; PAGE_SIZE];
        Scalar.dot_page(&query, &page, (1 << 2) | (1 << 63), &mut scores);
        assert_eq!(scores[2], 64.0);
        assert_eq!(scores[63], 64.0);
        assert_eq!(scores[0], -7.0);
        assert_eq!(scores[62], -7.0);
    }
}
