//! ReazonSpeech backend for any-stt.
//!
//! Target model: **reazonspeech-nemo-v2** (FastConformer encoder with
//! Longformer attention + RNN-T decoder). 619M parameters, SentencePiece
//! unigram tokenizer (3,000 tokens), Japanese only.
//!
//! Upstream:
//! - <https://huggingface.co/reazon-research/reazonspeech-nemo-v2>
//! - <https://research.reazon.jp/projects/ReazonSpeech/index.html>
//!
//! # Architecture (file layout matches the inference pipeline)
//!
//! ```text
//!   audio PCM (16 kHz mono)
//!     │
//!     ▼
//!   mel.rs        — log-mel filterbank (n_mels=80, win=400, hop=160)
//!     │
//!     ▼
//!   encoder.rs    — FastConformer, subsampling ×8, Longformer attn
//!     │
//!     ▼
//!   decoder.rs    — RNN-T prediction + joint network, greedy beam
//!     │
//!     ▼
//!   tokenizer.rs  — SentencePiece detokenize → Japanese text
//! ```
//!
//! # Status
//!
//! Skeleton. `ReazonSpeechEngine::transcribe` currently returns
//! `SttError::NotImplemented`. Each submodule documents the concrete
//! work remaining before a round-trip transcription is possible.
//!
//! # NPU strategy (from project constraints)
//!
//! - Encoder MatMul is the hot path → offloaded to `qnn-backend` /
//!   `nnapi-backend` on Android. CPU fallback via ggml FFI.
//! - LayerNorm / Softmax / positional bias stay on CPU (NPU-unfriendly).
//! - Decoder (RNN-T) runs on CPU — small ops, loop-heavy, not NPU-shaped.

pub mod config;
pub mod decoder;
pub mod encoder;
pub mod mel;
pub mod tokenizer;

use std::path::{Path, PathBuf};

use any_stt::{Backend, HardwareInfo, SttEngine, SttError, SttResult};

pub use config::ReazonSpeechConfig;

/// ReazonSpeech-NeMo-v2 engine.
pub struct ReazonSpeechEngine {
    model_path: PathBuf,
    config: ReazonSpeechConfig,
    hardware_info: HardwareInfo,
    backend: Backend,
    language: String,
}

impl ReazonSpeechEngine {
    /// Load a reazonspeech-nemo-v2 GGUF model.
    ///
    /// Expects `model_path.tokenizer.model` alongside the `.gguf` file.
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

        let gguf = gguf_loader::GgufFile::open(model_path).map_err(|e| {
            SttError::TranscriptionFailed(format!("gguf open failed: {e}"))
        })?;

        let config = ReazonSpeechConfig::from_gguf(&gguf).map_err(|e| {
            SttError::TranscriptionFailed(format!("invalid model: {e}"))
        })?;

        Ok(Self {
            model_path: model_path.to_path_buf(),
            config,
            hardware_info,
            backend,
            language: language.to_string(),
        })
    }

    pub fn config(&self) -> &ReazonSpeechConfig {
        &self.config
    }
}

impl SttEngine for ReazonSpeechEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError> {
        if audio.is_empty() {
            return Err(SttError::InvalidAudio("empty audio buffer".into()));
        }

        // TODO(#N4): implement inference.
        //   1. mel::log_mel_spectrogram(audio)
        //   2. encoder::forward(&mel) — FastConformer + Longformer attn
        //   3. decoder::rnnt_greedy_decode(&enc_out) — RNN-T greedy
        //   4. tokenizer::detokenize(&token_ids)
        Err(SttError::NotImplemented(format!(
            "ReazonSpeechEngine::transcribe not yet implemented — \
             model loaded from {}, backend={:?}, language={}, \
             see crates/reazonspeech-backend/src/{{mel,encoder,decoder,tokenizer}}.rs",
            self.model_path.display(),
            self.backend,
            self.language,
        )))
    }

    fn is_ready(&self) -> bool {
        // Ready means: model file loaded + config parsed. Actual inference
        // capability is signalled by the NotImplemented from transcribe().
        self.model_path.exists()
    }

    fn hardware_info(&self) -> &HardwareInfo {
        &self.hardware_info
    }

    fn active_backend(&self) -> Backend {
        self.backend
    }
}

/// Initialize a ReazonSpeech engine with auto-selected backend.
///
/// Equivalent to `whisper_backend::initialize` but for the ReazonSpeech
/// family. `any_stt::initialize` directs callers here when the config's
/// model family is `ModelFamily::ReazonSpeech`.
pub fn initialize(config: &any_stt::SttConfig) -> Result<Box<dyn SttEngine>, SttError> {
    let model_path = config.model_path.as_ref().ok_or_else(|| {
        SttError::TranscriptionFailed("model_path is required".into())
    })?;

    let hw = any_stt::detect_hardware();
    let selection = any_stt::selector::select(config, &hw);

    let engine = ReazonSpeechEngine::new(model_path, &config.language, selection.backend, hw)?;
    Ok(Box::new(engine))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_nonexistent_file_returns_model_not_found() {
        let hw = any_stt::detect_hardware();
        let result = ReazonSpeechEngine::new(
            Path::new("/does/not/exist.gguf"),
            "ja",
            Backend::Cpu,
            hw,
        );
        assert!(matches!(result, Err(SttError::ModelNotFound { .. })));
    }

    #[test]
    fn empty_audio_returns_invalid_audio() {
        // Construct an engine with a dummy path — the audio check runs
        // before any model access.
        let cfg = ReazonSpeechConfig::dummy();
        let hw = any_stt::detect_hardware();
        let engine = ReazonSpeechEngine {
            model_path: PathBuf::from("/dummy"),
            config: cfg,
            hardware_info: hw,
            backend: Backend::Cpu,
            language: "ja".into(),
        };
        let result = engine.transcribe(&[]);
        assert!(matches!(result, Err(SttError::InvalidAudio(_))));
    }

    #[test]
    fn transcribe_currently_returns_not_implemented() {
        let cfg = ReazonSpeechConfig::dummy();
        let hw = any_stt::detect_hardware();
        let engine = ReazonSpeechEngine {
            model_path: PathBuf::from("/dummy"),
            config: cfg,
            hardware_info: hw,
            backend: Backend::Cpu,
            language: "ja".into(),
        };
        let audio = vec![0.0f32; 16000];
        match engine.transcribe(&audio) {
            Err(SttError::NotImplemented(msg)) => {
                assert!(msg.contains("ReazonSpeechEngine"));
            }
            Err(e) => panic!("expected NotImplemented, got {e}"),
            Ok(_) => panic!("expected NotImplemented, got Ok"),
        }
    }
}
