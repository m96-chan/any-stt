//! Decoder trait and whisper.cpp-based implementation.
//!
//! The decoder takes encoder output and produces transcribed text.
//! Multiple implementations allow backend swapping (CPU via whisper.cpp,
//! future: pure Rust, NPU decoder).

use std::ffi::CString;
use std::path::Path;
use std::time::Instant;

use any_stt::config::Backend;
use any_stt::error::SttError;
use any_stt::SttResult;

use crate::ffi::*;
use crate::helpers;

/// Backend-swappable decoder trait.
///
/// Takes encoder output (f32 array of shape [n_ctx, n_state]) and produces
/// transcription results. The encoder may run on NPU while decoder runs on CPU.
pub trait WhisperDecoder: Send + Sync {
    /// Decode encoder output into text.
    ///
    /// `encoder_output`: flattened [n_ctx, n_state] f32 array.
    /// `n_ctx`: number of audio context frames (typically 1500).
    /// `n_state`: encoder state dimension (e.g., 384 for tiny, 1280 for large).
    /// `language`: language code (e.g., "en", "ja").
    fn decode(
        &self,
        encoder_output: &[f32],
        n_ctx: usize,
        n_state: usize,
        language: &str,
    ) -> Result<SttResult, SttError>;
}

/// Whisper.cpp-based decoder implementation.
///
/// Uses whisper_full with the encoder_begin_callback to skip encoding,
/// after injecting external encoder output into the whisper context.
pub struct WhisperCppDecoder {
    ctx: *mut WhisperContext,
    n_threads: i32,
    owns_ctx: bool,
}

// SAFETY: whisper_context is thread-safe for non-concurrent access
unsafe impl Send for WhisperCppDecoder {}
unsafe impl Sync for WhisperCppDecoder {}

impl WhisperCppDecoder {
    /// Create a decoder by loading a whisper model.
    ///
    /// The model must match the encoder (same architecture/size).
    pub fn new(model_path: &Path, n_threads: i32) -> Result<Self, SttError> {
        let path_cstr = helpers::model_path_to_cstring(model_path)?;

        let mut cparams = unsafe { whisper_context_default_params() };
        cparams.use_gpu = false; // decoder runs on CPU

        let ctx = unsafe { whisper_init_from_file_with_params(path_cstr.as_ptr(), cparams) };
        if ctx.is_null() {
            return Err(SttError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }

        Ok(Self {
            ctx,
            n_threads,
            owns_ctx: true,
        })
    }

    /// Wrap an existing whisper context (does NOT take ownership).
    ///
    /// # Safety
    /// The context must outlive this decoder and must not be used concurrently.
    pub unsafe fn from_raw(ctx: *mut WhisperContext, n_threads: i32) -> Self {
        Self {
            ctx,
            n_threads,
            owns_ctx: false,
        }
    }

    /// Run full transcription (encoder + decoder) using whisper.cpp.
    /// This is a convenience for CPU-only mode.
    pub fn transcribe_full(
        &self,
        audio: &[f32],
        language: &str,
    ) -> Result<SttResult, SttError> {
        let start = Instant::now();
        let lang = CString::new(language)
            .map_err(|_| SttError::TranscriptionFailed("invalid language".into()))?;

        let params = helpers::create_params(&lang, self.n_threads)?;
        helpers::run_whisper_full(self.ctx, params, audio)?;
        Ok(helpers::collect_result(self.ctx, start, language, Backend::Cpu))
    }
}

impl WhisperDecoder for WhisperCppDecoder {
    fn decode(
        &self,
        _encoder_output: &[f32],
        _n_ctx: usize,
        _n_state: usize,
        language: &str,
    ) -> Result<SttResult, SttError> {
        // TODO: Phase 3 full implementation — inject encoder_output into
        // ctx->state->embd_enc and run decoder-only pass.
        //
        // For now, this is a placeholder that requires the full whisper_full
        // pipeline. The hybrid engine will need to:
        // 1. Copy encoder_output to the whisper state's embd_enc tensor
        // 2. Run whisper_full with audio=silence (mel will be computed but
        //    encoder output is overwritten)
        //
        // This will be connected once we have the encoder graph working
        // and can verify the output format matches whisper.cpp's expectations.
        let _ = language;
        Err(SttError::NotImplemented(
            "decoder-only mode not yet implemented — use transcribe_full for CPU-only".into(),
        ))
    }
}

impl Drop for WhisperCppDecoder {
    fn drop(&mut self) {
        if self.owns_ctx && !self.ctx.is_null() {
            unsafe { whisper_free(self.ctx) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../third-party/whisper.cpp/models/ggml-tiny.en.bin")
    }

    #[test]
    fn decoder_load_model() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let decoder = WhisperCppDecoder::new(&path, 4);
        assert!(decoder.is_ok());
    }

    #[test]
    fn decoder_transcribe_full_silence() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let decoder = WhisperCppDecoder::new(&path, 4).unwrap();
        let silence = vec![0.0f32; 16000];
        let result = decoder.transcribe_full(&silence, "en").unwrap();
        assert!(
            result.text.trim().is_empty() || result.text.trim().starts_with('['),
            "expected blank output for silence, got: {:?}",
            result.text
        );
    }
}
