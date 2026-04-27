//! Model families supported by any-stt.
//!
//! A family groups models that share an architecture (and therefore an
//! inference backend crate). Each family lives in its own submodule:
//! `whisper.rs`, `reazonspeech.rs`, `parakeet.rs`, `qwen_asr.rs`.

/// Top-level ASR model family.
///
/// Used to dispatch `initialize()` to the correct backend crate without
/// `any-stt` taking a dependency on any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    /// Whisper / kotoba (encoder-decoder transformer, ggml via whisper.cpp).
    Whisper,
    /// ReazonSpeech (Japanese, FastConformer + RNN-T, NeMo origin).
    ReazonSpeech,
    /// NVIDIA Parakeet TDT (FastConformer + Token-Duration Transducer, NeMo origin).
    Parakeet,
    /// Qwen3-ASR (Qwen3-Omni based multimodal LLM).
    QwenAsr,
}

impl ModelFamily {
    /// Name of the backend crate that implements this family.
    /// Used in diagnostic messages from `any_stt::initialize`.
    pub const fn backend_crate(self) -> &'static str {
        match self {
            Self::Whisper => "whisper_backend",
            Self::ReazonSpeech => "reazonspeech_backend",
            Self::Parakeet => "parakeet_backend",
            Self::QwenAsr => "qwen_asr_backend",
        }
    }

    /// Short human label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Whisper => "Whisper",
            Self::ReazonSpeech => "ReazonSpeech",
            Self::Parakeet => "Parakeet",
            Self::QwenAsr => "Qwen3-ASR",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_crate_names_are_distinct() {
        let all = [
            ModelFamily::Whisper,
            ModelFamily::ReazonSpeech,
            ModelFamily::Parakeet,
            ModelFamily::QwenAsr,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.backend_crate(), b.backend_crate());
                assert_ne!(a.label(), b.label());
            }
        }
    }
}
