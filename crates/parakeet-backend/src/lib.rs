//! NVIDIA Parakeet TDT backend for any-stt.
//!
//! Target: **parakeet-tdt-0.6b-v3** — FastConformer (rel-pos) + TDT,
//! 600M params, SentencePiece (8,192 tokens), 25 European languages.
//!
//! ⚠️ **Japanese is NOT supported by this model.** For Japanese workloads
//! use `reazonspeech-backend` (when Longformer lands) or kotoba-whisper
//! via `whisper-backend`.
//!
//! Upstream: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
//!
//! # Pipeline
//!
//! ```text
//!   audio (f32, 16 kHz mono)
//!     ↓ log_mel_spectrogram (fastconformer-core)
//!   mel [time, n_mels]
//!     ↓ FastConformerEncoder.forward
//!   enc [time/8, d_model]
//!     ↓ TdtDecoder.greedy_decode
//!   token_ids [N]
//!     ↓ SentencePieceTokenizer.detokenize
//!   text
//! ```
//!
//! # Status
//!
//! Encoder forward is wired (vanilla rel-pos in fastconformer-core).
//! TDT decoder loader (`from_gguf`) is still stubbed, so transcribe
//! returns NotImplemented from there. Tokenizer load is plumbed.

pub mod decoder;
pub mod encoder;

use std::path::{Path, PathBuf};
use std::time::Instant;

use any_stt::{Backend, HardwareInfo, SttEngine, SttError, SttResult};
use fastconformer_core::encoder::FastConformerEncoder;
use fastconformer_core::{log_mel_spectrogram, SentencePieceTokenizer};

pub use fastconformer_core::Config as ParakeetConfig;

pub struct ParakeetEngine {
    model_path: PathBuf,
    config: ParakeetConfig,
    encoder: FastConformerEncoder,
    tokenizer: Option<SentencePieceTokenizer>,
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
        if matches!(language, "ja" | "ko" | "zh") {
            eprintln!(
                "warning: parakeet-tdt-0.6b-v3 does not support language \
                 {language:?}; consider reazonspeech-backend for Japanese"
            );
        }

        let gguf = gguf_loader::GgufFile::open(model_path)
            .map_err(|e| SttError::TranscriptionFailed(format!("gguf open: {e}")))?;
        let config = ParakeetConfig::from_gguf(&gguf)
            .map_err(|e| SttError::TranscriptionFailed(format!("config: {e}")))?;

        let encoder = encoder::load(&gguf, config.clone())
            .map_err(|e| SttError::TranscriptionFailed(format!("encoder load: {e}")))?;

        // Tokenizer companion lives at "<model>.tokenizer.model" by
        // convention (see `scripts/convert-nemo-to-gguf.py`).
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

    pub fn config(&self) -> &ParakeetConfig {
        &self.config
    }
}

impl SttEngine for ParakeetEngine {
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

        // TDT decoder needs a real GGUF loader (still stub). When it's
        // wired up, replace this with `decoder.greedy_decode(&enc_out)`.
        let _ = enc_out;
        Err(SttError::NotImplemented(format!(
            "ParakeetEngine: encoder forward succeeded ({} frames @ {} d_model, {:.0}ms), \
             but TDT decoder GGUF loader is not yet implemented",
            mel.n_frames,
            self.config.d_model,
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
