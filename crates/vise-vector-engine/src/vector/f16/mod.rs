//! FP16 segment dot-product kernels.

mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

use crate::vector::SegmentDotKernel;

pub use scalar::Scalar;
#[cfg(target_arch = "x86_64")]
pub use x86::Avx2F16c;

/// Selected kernel. Detection is performed once when this value is created.
pub enum Kernel {
    Scalar(Scalar),
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2F16c),
}

/// Select the best kernel supported by this CPU.
pub fn select() -> Kernel {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
        return Kernel::Avx2(Avx2F16c::new());
    }
    Kernel::Scalar(Scalar)
}

impl SegmentDotKernel<half::f16> for Kernel {
    #[inline]
    fn dot(&self, a: &[half::f16; 64], b: &[half::f16; 64]) -> f32 {
        match self {
            Self::Scalar(k) => k.dot(a, b),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.dot(a, b),
        }
    }

    #[inline]
    fn dot_page(
        &self,
        query: &[half::f16; 64],
        page: &crate::vector::AlignedSegmentPage<half::f16>,
        candidates: u64,
        scores: &mut [f32; crate::vector::PAGE_SIZE],
    ) {
        match self {
            Self::Scalar(k) => k.dot_page(query, page, candidates, scores),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.dot_page(query, page, candidates, scores),
        }
    }
}
