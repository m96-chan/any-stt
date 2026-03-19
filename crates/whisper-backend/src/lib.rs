pub mod ffi;

use std::ffi::{CStr, CString};
use std::path::Path;
use std::time::Instant;

use any_stt::config::Backend;
use any_stt::error::SttError;
use any_stt::hardware::HardwareInfo;
use any_stt::{SttEngine, SttResult};

use ffi::*;

/// Safe wrapper around a whisper.cpp context.
///
/// Implements [`SttEngine`] so it can be used via the `any-stt` API.
pub struct WhisperEngine {
    ctx: *mut WhisperContext,
    hardware_info: HardwareInfo,
    backend: Backend,
    language: CString,
    n_threads: i32,
}

// SAFETY: whisper_context is internally thread-safe as long as the same
// context is not used concurrently (which we enforce via &self / &mut self).
unsafe impl Send for WhisperEngine {}
unsafe impl Sync for WhisperEngine {}

impl WhisperEngine {
    /// Load a whisper model from a ggml file.
    ///
    /// `model_path` must point to a valid .bin / .gguf whisper model file.
    pub fn new(
        model_path: &Path,
        language: &str,
        backend: Backend,
        hardware_info: HardwareInfo,
    ) -> Result<Self, SttError> {
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
        cparams.use_gpu = matches!(backend, Backend::Cuda | Backend::Metal | Backend::Vulkan);

        let ctx = unsafe { whisper_init_from_file_with_params(path_cstr.as_ptr(), cparams) };

        if ctx.is_null() {
            return Err(SttError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }

        let lang = CString::new(language).map_err(|_| {
            SttError::TranscriptionFailed("invalid language string".into())
        })?;

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        Ok(Self {
            ctx,
            hardware_info,
            backend,
            language: lang,
            n_threads,
        })
    }
}

impl Drop for WhisperEngine {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { whisper_free(self.ctx) };
        }
    }
}

impl SttEngine for WhisperEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError> {
        let start = Instant::now();

        // Create params via shim.
        let params =
            unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
        if params.is_null() {
            return Err(SttError::TranscriptionFailed(
                "failed to allocate whisper params".into(),
            ));
        }

        // Configure params.
        unsafe {
            shim_params_set_language(params, self.language.as_ptr());
            shim_params_set_n_threads(params, self.n_threads);
            shim_params_set_translate(params, false);
            shim_params_set_print_special(params, false);
            shim_params_set_print_progress(params, false);
            shim_params_set_print_realtime(params, false);
            shim_params_set_print_timestamps(params, false);
            shim_params_set_single_segment(params, false);
            shim_params_set_suppress_nst(params, true);
        }

        // Run inference.
        let ret = unsafe {
            shim_whisper_full(self.ctx, params, audio.as_ptr(), audio.len() as i32)
        };

        // Free params immediately after use.
        unsafe { shim_free_params(params) };

        if ret != 0 {
            return Err(SttError::TranscriptionFailed(format!(
                "whisper_full returned error code {ret}"
            )));
        }

        // Collect segments.
        let n_segments = unsafe { whisper_full_n_segments(self.ctx) };
        let mut text = String::new();
        for i in 0..n_segments {
            let segment_text = unsafe { whisper_full_get_segment_text(self.ctx, i) };
            if !segment_text.is_null() {
                let s = unsafe { CStr::from_ptr(segment_text) };
                text.push_str(&s.to_string_lossy());
            }
        }

        // Get detected language.
        let lang_id = unsafe { whisper_full_lang_id(self.ctx) };
        let lang_str = unsafe { whisper_lang_str(lang_id) };
        let language = if !lang_str.is_null() {
            unsafe { CStr::from_ptr(lang_str) }
                .to_string_lossy()
                .into_owned()
        } else {
            "unknown".to_string()
        };

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(SttResult {
            text,
            language,
            duration_ms,
            backend_used: self.backend,
        })
    }

    fn is_ready(&self) -> bool {
        !self.ctx.is_null()
    }

    fn hardware_info(&self) -> &HardwareInfo {
        &self.hardware_info
    }

    fn active_backend(&self) -> Backend {
        self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_model_returns_error() {
        let hw = any_stt::detect_hardware();
        let result = WhisperEngine::new(
            Path::new("/nonexistent/model.bin"),
            "en",
            Backend::Cpu,
            hw,
        );
        assert!(result.is_err());
    }
}
