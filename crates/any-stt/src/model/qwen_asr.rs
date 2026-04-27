//! Qwen3-ASR family variants (Alibaba Qwen3-Omni based multimodal LLM).
//!
//! Upstream:
//! - <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
//! - <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
//!
//! Weights are Apache-2.0, Safetensors only (no official GGUF).
//! Requires a GGUF conversion step for the audio encoder + Qwen3 decoder.

/// Qwen3-ASR model variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QwenAsrVariant {
    /// Qwen3-ASR-0.6B — lightweight variant (~0.6B params).
    B0_6,
    /// Qwen3-ASR-1.7B — standard variant (~1.7B params).
    /// Supports 30+ languages including Japanese, Chinese, Korean.
    B1_7,
}

impl QwenAsrVariant {
    /// Rough f16 model size estimate in MB.
    pub const fn size_estimate_f16_mb(&self) -> u64 {
        match self {
            // ~0.6B × 2 bytes
            Self::B0_6 => 1200,
            // ~1.7B × 2 bytes ≈ 3.4 GB
            Self::B1_7 => 3400,
        }
    }
}
