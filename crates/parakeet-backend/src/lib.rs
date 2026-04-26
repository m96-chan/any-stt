//! NVIDIA Parakeet TDT backend for any-stt.
//!
//! Target model: **parakeet-tdt-0.6b-v3** (FastConformer + TDT). 600M
//! parameters, SentencePiece (8,192 tokens), 25 European languages.
//!
//! ⚠️ **Japanese is NOT supported by this model.** For Japanese workloads
//! use [`reazonspeech-backend`] or kotoba-whisper via `whisper-backend`.
//!
//! Upstream: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
//!
//! # Architecture
//!
//! Shares the FastConformer encoder with [`reazonspeech-backend`] (same
//! `scripts/convert-nemo-to-gguf.py` produces GGUFs for both). The decoder
//! is a **TDT (Token-Duration Transducer)** — an RNN-T generalization that
//! emits a (token, duration) pair per step, skipping multiple encoder
//! frames at once. Faster than RNN-T for the same accuracy.
//!
//! # Status
//!
//! Skeleton. Decoder logic is independent from reazonspeech but the encoder
//! forward is expected to share code once both crates pass CPU reference
//! tests — at that point we extract `crates/fastconformer-core/`.
//!
//! # Decoder sketch (for when we come back to implement)
//!
//! ```text
//! for each encoder frame t starting at t=0:
//!     loop:
//!         logits = joint(enc[t], pred_state)
//!         token_logits = logits[:vocab]       // vocab_size + 1 for blank
//!         dur_logits   = logits[vocab:]       // len(tdt_durations)
//!         token = argmax(token_logits)
//!         duration = tdt_durations[argmax(dur_logits)]
//!         if token != blank:
//!             emit token
//!             pred_state = lstm_step(embedding(token), pred_state)
//!         t += max(duration, 1)
//! ```

use std::path::{Path, PathBuf};

use any_stt::{Backend, HardwareInfo, SttEngine, SttError, SttResult};

pub struct ParakeetEngine {
    model_path: PathBuf,
    hardware_info: HardwareInfo,
    backend: Backend,
    language: String,
}

impl ParakeetEngine {
    pub fn new(
        model_path: &Path,
        language: &str,
        backend: Backend,
        hardware_info: HardwareInfo,
    ) -> Result<Self, SttError> {
        if !model_path.exists() {
            return Err(SttError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }
        // Parakeet-TDT does not support Japanese / Korean / Chinese.
        // We log but do not error — the caller may be using a custom fork.
        if matches!(language, "ja" | "ko" | "zh") {
            eprintln!(
                "warning: parakeet-tdt-0.6b-v3 does not support language {language:?}; \
                 consider reazonspeech-backend for Japanese"
            );
        }
        Ok(Self {
            model_path: model_path.to_path_buf(),
            hardware_info,
            backend,
            language: language.to_string(),
        })
    }
}

impl SttEngine for ParakeetEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError> {
        if audio.is_empty() {
            return Err(SttError::InvalidAudio("empty audio buffer".into()));
        }
        // TODO(#N5): FastConformer encoder (shared w/ reazonspeech) → TDT decode.
        Err(SttError::NotImplemented(format!(
            "ParakeetEngine::transcribe not yet implemented — \
             loaded {}, backend={:?}, language={}",
            self.model_path.display(),
            self.backend,
            self.language,
        )))
    }

    fn is_ready(&self) -> bool {
        self.model_path.exists()
    }

    fn hardware_info(&self) -> &HardwareInfo {
        &self.hardware_info
    }

    fn active_backend(&self) -> Backend {
        self.backend
    }
}

pub fn initialize(config: &any_stt::SttConfig) -> Result<Box<dyn SttEngine>, SttError> {
    let model_path = config.model_path.as_ref().ok_or_else(|| {
        SttError::TranscriptionFailed("model_path is required".into())
    })?;
    let hw = any_stt::detect_hardware();
    let selection = any_stt::selector::select(config, &hw);
    let engine = ParakeetEngine::new(model_path, &config.language, selection.backend, hw)?;
    Ok(Box::new(engine))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_model_returns_error() {
        let hw = any_stt::detect_hardware();
        let result = ParakeetEngine::new(
            Path::new("/does/not/exist.gguf"),
            "en",
            Backend::Cpu,
            hw,
        );
        assert!(matches!(result, Err(SttError::ModelNotFound { .. })));
    }
}
