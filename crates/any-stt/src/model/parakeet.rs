//! NVIDIA Parakeet family variants (FastConformer + TDT).
//!
//! Upstream: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
//!
//! **Note**: Parakeet TDT 0.6B v3 covers 25 European languages but
//! **does not support Japanese**. Use `ReazonSpeechVariant` or
//! `WhisperVariant::KotobaV2` for Japanese workloads.

/// Parakeet model variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParakeetVariant {
    /// parakeet-tdt-0.6b-v3 — FastConformer encoder + TDT (Token-Duration
    /// Transducer) decoder. ~600M parameters. SentencePiece (8,192 tokens).
    /// 25 European languages; no Japanese, Chinese, Korean.
    Tdt0_6bV3,
}

impl ParakeetVariant {
    /// Rough f16 model size estimate in MB.
    pub const fn size_estimate_f16_mb(&self) -> u64 {
        match self {
            // 600M × 2 bytes ≈ 1.2 GB
            Self::Tdt0_6bV3 => 1200,
        }
    }
}
