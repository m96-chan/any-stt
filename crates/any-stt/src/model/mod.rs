//! Model taxonomy: top-level `Model` enum plus per-family variant modules.
//!
//! The code is split so that each family's variants live in their own file.
//! This keeps Whisper-specific logic isolated and lets new families be added
//! without editing existing modules.
//!
//! ```text
//! ModelFamily ──┬── Whisper       → whisper.rs       (crate whisper-backend)
//!               ├── ReazonSpeech  → reazonspeech.rs  (crate reazonspeech-backend)
//!               ├── Parakeet      → parakeet.rs      (crate parakeet-backend)
//!               └── QwenAsr       → qwen_asr.rs      (crate qwen-asr-backend)
//! ```

pub mod family;
pub mod parakeet;
pub mod qwen_asr;
pub mod reazonspeech;
pub mod whisper;

pub use family::ModelFamily;
pub use parakeet::ParakeetVariant;
pub use qwen_asr::QwenAsrVariant;
pub use reazonspeech::ReazonSpeechVariant;
pub use whisper::WhisperVariant;

/// A specific ASR model.
///
/// Use `model.family()` to route to the correct backend crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Model {
    /// OpenAI Whisper / distilled / kotoba.
    Whisper(WhisperVariant),
    /// ReazonSpeech (Japanese).
    ReazonSpeech(ReazonSpeechVariant),
    /// NVIDIA Parakeet TDT.
    Parakeet(ParakeetVariant),
    /// Qwen3-ASR (multimodal LLM).
    QwenAsr(QwenAsrVariant),
    /// Custom model identified by a string (typically a HuggingFace repo ID).
    ///
    /// The family defaults to `Whisper`. Callers with a non-Whisper custom
    /// model should construct the variant explicitly (e.g.
    /// `Model::ReazonSpeech(ReazonSpeechVariant::NemoV2)`).
    Custom(String),
}

impl Model {
    /// Which family this model belongs to.
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Whisper(_) => ModelFamily::Whisper,
            Self::ReazonSpeech(_) => ModelFamily::ReazonSpeech,
            Self::Parakeet(_) => ModelFamily::Parakeet,
            Self::QwenAsr(_) => ModelFamily::QwenAsr,
            Self::Custom(_) => ModelFamily::Whisper,
        }
    }

    /// Rough f16 model size estimate in MB. Used by the selector to choose
    /// a quantization that fits in available memory.
    pub fn size_estimate_f16_mb(&self) -> u64 {
        match self {
            Self::Whisper(v) => v.size_estimate_f16_mb(),
            Self::ReazonSpeech(v) => v.size_estimate_f16_mb(),
            Self::Parakeet(v) => v.size_estimate_f16_mb(),
            Self::QwenAsr(v) => v.size_estimate_f16_mb(),
            // Conservative default — large enough to trip the quant selector
            // toward a smaller quant on modest hardware.
            Self::Custom(_) => 1500,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_dispatch_all_variants() {
        assert_eq!(
            Model::Whisper(WhisperVariant::Tiny).family(),
            ModelFamily::Whisper
        );
        assert_eq!(
            Model::ReazonSpeech(ReazonSpeechVariant::NemoV2).family(),
            ModelFamily::ReazonSpeech
        );
        assert_eq!(
            Model::Parakeet(ParakeetVariant::Tdt0_6bV3).family(),
            ModelFamily::Parakeet
        );
        assert_eq!(
            Model::QwenAsr(QwenAsrVariant::B1_7).family(),
            ModelFamily::QwenAsr
        );
        assert_eq!(
            Model::Custom("kotoba-tech/kotoba-whisper-v2.0".into()).family(),
            ModelFamily::Whisper
        );
    }

    #[test]
    fn size_estimate_is_positive_for_all_variants() {
        let variants = [
            Model::Whisper(WhisperVariant::Tiny),
            Model::Whisper(WhisperVariant::LargeV3),
            Model::Whisper(WhisperVariant::KotobaV2),
            Model::ReazonSpeech(ReazonSpeechVariant::NemoV2),
            Model::ReazonSpeech(ReazonSpeechVariant::K2V2),
            Model::ReazonSpeech(ReazonSpeechVariant::EspnetV2),
            Model::Parakeet(ParakeetVariant::Tdt0_6bV3),
            Model::QwenAsr(QwenAsrVariant::B0_6),
            Model::QwenAsr(QwenAsrVariant::B1_7),
            Model::Custom("foo/bar".into()),
        ];
        for v in variants {
            assert!(
                v.size_estimate_f16_mb() > 0,
                "size must be positive for {v:?}"
            );
        }
    }

    #[test]
    fn qwen_1_7b_is_larger_than_0_6b() {
        let small = Model::QwenAsr(QwenAsrVariant::B0_6).size_estimate_f16_mb();
        let big = Model::QwenAsr(QwenAsrVariant::B1_7).size_estimate_f16_mb();
        assert!(big > small);
    }

    #[test]
    fn model_eq() {
        assert_eq!(
            Model::Whisper(WhisperVariant::Small),
            Model::Whisper(WhisperVariant::Small)
        );
        assert_ne!(
            Model::Whisper(WhisperVariant::Small),
            Model::Whisper(WhisperVariant::Tiny)
        );
        let custom = Model::Custom("kotoba-tech/kotoba-whisper-v2.0".into());
        assert_eq!(custom, custom.clone());
        assert_ne!(custom, Model::Whisper(WhisperVariant::KotobaV2));
    }
}
