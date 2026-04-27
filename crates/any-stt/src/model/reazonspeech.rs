//! ReazonSpeech family variants (Japanese ASR from Reazon Human Interaction Lab).
//!
//! Upstream: <https://research.reazon.jp/projects/ReazonSpeech/index.html>
//!
//! All three official variants (NeMo, k2, ESPnet) are listed. The initial
//! backend implementation targets `NemoV2` (FastConformer + RNN-T).

/// ReazonSpeech v2 model variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReazonSpeechVariant {
    /// reazonspeech-nemo-v2 — FastConformer with Longformer attention + RNN-T decoder.
    /// ~619M parameters. SentencePiece unigram (3,000 tokens).
    NemoV2,
    /// reazonspeech-k2-v2 — Next-gen Kaldi transducer.
    K2V2,
    /// reazonspeech-espnet-v2 — Conformer based.
    EspnetV2,
}

impl ReazonSpeechVariant {
    /// Rough f16 model size estimate in MB.
    /// Derived from parameter count × 2 bytes + overhead.
    pub const fn size_estimate_f16_mb(&self) -> u64 {
        match self {
            // 619M × 2 bytes ≈ 1.24 GB
            Self::NemoV2 => 1250,
            // Parameter counts not officially published; conservative estimate.
            Self::K2V2 => 1200,
            Self::EspnetV2 => 1200,
        }
    }
}
