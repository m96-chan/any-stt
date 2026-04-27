//! ReazonSpeech backend for any-stt.
//!
//! Target: **reazonspeech-nemo-v2** — FastConformer encoder with Longformer
//! attention + RNN-T decoder. 619M params, SentencePiece unigram (3,000
//! tokens), Japanese.
//!
//! Upstream:
//! - <https://huggingface.co/reazon-research/reazonspeech-nemo-v2>
//! - <https://research.reazon.jp/projects/ReazonSpeech/index.html>
//!
//! # File layout
//! ```text
//! lib.rs       — engine (this file)
//! encoder.rs   — FastConformer-Longformer wrapper
//! decoder.rs   — RNN-T greedy
//! ```
//!
//! Audio preprocessing (log-mel) and tokenization (SentencePiece) live
//! in `fastconformer-core`, shared with `parakeet-backend`.
//!
//! # Status
//!
//! Skeleton + decoder primitives. `transcribe()` returns
//! `SttError::NotImplemented` until the encoder forward pass lands.

pub mod decoder;
pub mod encoder;

use std::path::{Path, PathBuf};

use any_stt::{Backend, HardwareInfo, SttEngine, SttError, SttResult};

pub use fastconformer_core::Config as ReazonSpeechConfig;

/// ReazonSpeech-NeMo-v2 engine.
pub struct ReazonSpeechEngine {
    model_path: PathBuf,
    config: ReazonSpeechConfig,
    hardware_info: HardwareInfo,
    backend: Backend,
    language: String,
}

impl ReazonSpeechEngine {
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
        // TODO(#N4): mel → encoder → decoder → tokenizer pipeline.
        Err(SttError::NotImplemented(format!(
            "ReazonSpeechEngine::transcribe not yet implemented — \
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
        let cfg = fastconformer_core::Config::dummy_reazonspeech_nemo_v2();
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
        let cfg = fastconformer_core::Config::dummy_reazonspeech_nemo_v2();
        let hw = any_stt::detect_hardware();
        let engine = ReazonSpeechEngine {
            model_path: PathBuf::from("/dummy"),
            config: cfg,
            hardware_info: hw,
            backend: Backend::Cpu,
            language: "ja".into(),
        };
        let audio = vec![0.0_f32; 16000];
        match engine.transcribe(&audio) {
            Err(SttError::NotImplemented(msg)) => {
                assert!(msg.contains("ReazonSpeechEngine"));
            }
            Err(e) => panic!("expected NotImplemented, got {e}"),
            Ok(_) => panic!("expected NotImplemented, got Ok"),
        }
    }
}
