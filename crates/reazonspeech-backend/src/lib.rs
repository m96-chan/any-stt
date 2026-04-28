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
//! - mel + tokenizer + RNN-T greedy: implemented (#N4)
//! - encoder forward (Conformer block, Longformer attention): implemented
//!   (#N7) — RelPosLocalAttn dispatched through `AttentionMode::LocalGlobal`
//! - RNN-T decoder GGUF loader: stub (blocked by #N8 — needs real .nemo)
//!
//! `transcribe()` runs mel → encoder.forward end-to-end and surfaces
//! `NotImplemented` at the RNN-T decoder GGUF loader seam, the only
//! remaining stub.

pub mod decoder;
pub mod encoder;

use std::path::{Path, PathBuf};
use std::time::Instant;

use any_stt::{Backend, HardwareInfo, SttEngine, SttError, SttResult};
use fastconformer_core::encoder::FastConformerEncoder;
use fastconformer_core::{log_mel_spectrogram, SentencePieceTokenizer};

pub use fastconformer_core::Config as ReazonSpeechConfig;

/// ReazonSpeech-NeMo-v2 engine.
pub struct ReazonSpeechEngine {
    model_path: PathBuf,
    config: ReazonSpeechConfig,
    encoder: FastConformerEncoder,
    tokenizer: Option<SentencePieceTokenizer>,
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
            SttError::TranscriptionFailed(format!("gguf open: {e}"))
        })?;
        let config = ReazonSpeechConfig::from_gguf(&gguf).map_err(|e| {
            SttError::TranscriptionFailed(format!("config: {e}"))
        })?;

        let encoder = encoder::load(&gguf, config.clone())
            .map_err(|e| SttError::TranscriptionFailed(format!("encoder load: {e}")))?;

        let companion = model_path.with_extension("tokenizer.model");
        let tokenizer = if companion.exists() {
            Some(
                SentencePieceTokenizer::load(&companion)
                    .map_err(|e| SttError::TranscriptionFailed(format!("tokenizer: {e}")))?,
            )
        } else {
            eprintln!(
                "warning: tokenizer companion not found at {} — \
                 transcribe will return token IDs instead of text",
                companion.display()
            );
            None
        };

        Ok(Self {
            model_path: model_path.to_path_buf(),
            config,
            encoder,
            tokenizer,
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
        let start = Instant::now();

        let mel = log_mel_spectrogram(audio, &self.config);

        let enc_out = self
            .encoder
            .forward(&mel.data, mel.n_frames)
            .map_err(|e| SttError::TranscriptionFailed(format!("encoder forward: {e}")))?;

        // RNN-T decoder needs a real GGUF loader. When #N8 lands, replace
        // this with `decoder.greedy_decode(&enc_out)` then
        // `tokenizer.detokenize(&token_ids)`.
        let _ = enc_out;
        Err(SttError::NotImplemented(format!(
            "ReazonSpeechEngine: encoder forward succeeded ({} frames @ {} d_model, \
             attention={:?}, {:.0}ms), but RNN-T decoder GGUF loader is not yet \
             implemented (#N8)",
            mel.n_frames,
            self.config.d_model,
            self.config.attention_type,
            start.elapsed().as_secs_f64() * 1000.0
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
}
