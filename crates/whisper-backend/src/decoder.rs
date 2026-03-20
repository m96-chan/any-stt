//! Decoder trait and whisper.cpp-based implementation.
//!
//! The decoder takes encoder output and produces transcribed text.
//! Multiple implementations allow backend swapping (CPU via whisper.cpp,
//! future: pure Rust, NPU decoder).

use std::ffi::{CStr, CString};
use std::path::Path;
use std::time::Instant;

use any_stt::config::Backend;
use any_stt::error::SttError;
use any_stt::SttResult;

use crate::ffi::*;

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
        let path_cstr = CString::new(
            model_path
                .to_str()
                .ok_or_else(|| SttError::ModelNotFound {
                    path: model_path.to_path_buf(),
                })?,
        )
        .map_err(|_| SttError::ModelNotFound {
            path: model_path.to_path_buf(),
        })?;

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

        let params = unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
        if params.is_null() {
            return Err(SttError::TranscriptionFailed("failed to alloc params".into()));
        }

        unsafe {
            shim_params_set_language(params, lang.as_ptr());
            shim_params_set_n_threads(params, self.n_threads);
            shim_params_set_translate(params, false);
            shim_params_set_print_special(params, false);
            shim_params_set_print_progress(params, false);
            shim_params_set_print_realtime(params, false);
            shim_params_set_print_timestamps(params, false);
            shim_params_set_single_segment(params, false);
            shim_params_set_suppress_nst(params, true);
        }

        let ret = unsafe {
            shim_whisper_full(self.ctx, params, audio.as_ptr(), audio.len() as i32)
        };
        unsafe { shim_free_params(params) };

        if ret != 0 {
            return Err(SttError::TranscriptionFailed(format!(
                "whisper_full error {ret}"
            )));
        }

        self.collect_result(start, language)
    }

    /// Collect transcription segments from the whisper context.
    fn collect_result(&self, start: Instant, fallback_lang: &str) -> Result<SttResult, SttError> {
        let n_segments = unsafe { whisper_full_n_segments(self.ctx) };
        let mut text = String::new();
        for i in 0..n_segments {
            let segment_text = unsafe { whisper_full_get_segment_text(self.ctx, i) };
            if !segment_text.is_null() {
                let s = unsafe { CStr::from_ptr(segment_text) };
                text.push_str(&s.to_string_lossy());
            }
        }

        let lang_id = unsafe { whisper_full_lang_id(self.ctx) };
        let lang_str = unsafe { whisper_lang_str(lang_id) };
        let language = if !lang_str.is_null() {
            unsafe { CStr::from_ptr(lang_str) }
                .to_string_lossy()
                .into_owned()
        } else {
            fallback_lang.to_string()
        };

        Ok(SttResult {
            text,
            language,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            backend_used: Backend::Cpu,
        })
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
