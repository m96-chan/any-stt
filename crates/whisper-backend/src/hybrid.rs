//! Hybrid CPU+NPU whisper engine.
//!
//! Combines:
//! - CPU preprocessor (mel spectrogram + Conv1d + GELU + pos_embed)
//! - NPU encoder (QNN HTP transformer blocks)
//! - CPU decoder (whisper.cpp-based, trait-swappable)
//!
//! This is the main integration point for Phase 3.

use std::path::Path;
use std::time::Instant;

use any_stt::config::Backend;
use any_stt::error::SttError;
use any_stt::hardware::HardwareInfo;
use any_stt::{SttEngine, SttResult};

use gguf_loader::GgufFile;
use qnn_backend::{EncoderConfig, QnnContext, QnnLibrary, WhisperEncoderGraph};

use crate::decoder::{WhisperCppDecoder, WhisperDecoder};
use crate::preprocess::Preprocessor;

/// Hybrid CPU+NPU Whisper engine.
///
/// Audio → Preprocessor (CPU) → Encoder (NPU) → Decoder (CPU) → Text
pub struct WhisperQnnEngine {
    preprocessor: Preprocessor,
    encoder: WhisperEncoderGraph,
    decoder: Box<dyn WhisperDecoder>,
    hardware_info: HardwareInfo,
    language: String,
    n_state: usize,
    n_ctx: usize,
}

impl WhisperQnnEngine {
    /// Build the hybrid engine from a GGUF model file.
    ///
    /// `gguf_path`: path to the whisper GGUF model file (for encoder weights + preprocessor)
    /// `model_path`: path to the whisper.cpp model file (for decoder)
    /// `encoder_config`: encoder architecture configuration
    /// `cache_dir`: optional directory for compiled QNN context cache
    /// `language`: language code
    /// `hardware_info`: detected hardware capabilities
    pub fn new(
        gguf_path: &Path,
        model_path: &Path,
        encoder_config: EncoderConfig,
        cache_dir: Option<&Path>,
        language: &str,
        hardware_info: HardwareInfo,
    ) -> Result<Self, SttError> {
        let n_state = encoder_config.n_state as usize;
        let n_ctx = encoder_config.n_ctx as usize;

        // Load GGUF for weights
        let gguf = GgufFile::open(gguf_path)
            .map_err(|e| SttError::TranscriptionFailed(format!("GGUF open: {e}")))?;

        // Build preprocessor from GGUF weights
        let preprocessor = Preprocessor::from_gguf(&gguf)
            .map_err(|e| SttError::TranscriptionFailed(format!("preprocessor: {e}")))?;

        // Build or load cached QNN encoder
        let encoder = Self::build_encoder(&gguf, encoder_config.clone(), cache_dir)?;

        // Build whisper.cpp decoder
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let decoder = WhisperCppDecoder::new(model_path, n_threads)
            .map_err(|e| SttError::TranscriptionFailed(format!("decoder: {e}")))?;

        Ok(Self {
            preprocessor,
            encoder,
            decoder: Box::new(decoder),
            hardware_info,
            language: language.to_string(),
            n_state,
            n_ctx,
        })
    }

    /// Build the QNN encoder, using cache if available.
    fn build_encoder(
        gguf: &GgufFile,
        config: EncoderConfig,
        cache_dir: Option<&Path>,
    ) -> Result<WhisperEncoderGraph, SttError> {
        let lib = QnnLibrary::load_htp().map_err(|e| {
            SttError::BackendUnavailable {
                backend: Backend::Qnn,
                reason: e,
            }
        })?;

        // Try loading from cache
        if let Some(dir) = cache_dir {
            let cache_path =
                QnnContext::cache_path(dir, "whisper", config.n_layer, config.n_state);
            if cache_path.exists() {
                match QnnContext::from_binary(lib, &cache_path) {
                    Ok(ctx) => {
                        eprintln!("QNN: loaded cached context from {}", cache_path.display());
                        // TODO: need to reconstruct graph handles from cached context
                        // For now, fall through to build from scratch
                        let _ = ctx;
                    }
                    Err(e) => {
                        eprintln!("QNN: cache load failed (will rebuild): {e}");
                    }
                }
            }
        }

        // Build from scratch
        let lib = QnnLibrary::load_htp().map_err(|e| {
            SttError::BackendUnavailable {
                backend: Backend::Qnn,
                reason: e,
            }
        })?;
        let ctx = QnnContext::new(lib).map_err(|e| {
            SttError::TranscriptionFailed(format!("QNN context: {e}"))
        })?;

        let encoder_config = config.clone();
        let encoder = WhisperEncoderGraph::build(ctx, encoder_config, |name| {
            gguf.dequantize_f32(name)
        })
        .map_err(|e| SttError::TranscriptionFailed(format!("encoder build: {e}")))?;

        // Save cache
        if let Some(dir) = cache_dir {
            let cache_path =
                QnnContext::cache_path(dir, "whisper", config.n_layer, config.n_state);
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!("QNN: failed to create cache dir: {e}");
            } else {
                // TODO: save_binary needs graph context access
                eprintln!("QNN: context cache not yet implemented for save");
                let _ = cache_path;
            }
        }

        Ok(encoder)
    }
}

impl SttEngine for WhisperQnnEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError> {
        let start = Instant::now();

        // Step 1: Mel spectrogram (via whisper.cpp FFI)
        // For now, we compute mel + conv on CPU via the preprocessor.
        // The mel computation will be done separately once we add the FFI.
        //
        // TODO: call whisper_pcm_to_mel to get mel spectrogram, then pass to preprocessor.
        // For now, use a placeholder approach.
        let _ = audio;

        // Step 2: CPU preprocessor (conv1d + gelu + pos_embed)
        // preprocessor.process_mel(mel, n_frames)

        // Step 3: NPU encoder
        // let enc_output = self.encoder.execute(&enc_input)?;

        // Step 4: CPU decoder
        // self.decoder.decode(&enc_output, self.n_ctx, self.n_state, &self.language)

        // Placeholder: not yet wired up
        let _ = start;
        Err(SttError::NotImplemented(
            "WhisperQnnEngine::transcribe not yet fully wired — \
             mel spectrogram FFI integration pending"
                .into(),
        ))
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn hardware_info(&self) -> &HardwareInfo {
        &self.hardware_info
    }

    fn active_backend(&self) -> Backend {
        Backend::Qnn
    }
}
