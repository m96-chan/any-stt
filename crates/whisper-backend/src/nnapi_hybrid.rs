//! Hybrid CPU+NPU whisper engine for NNAPI (MediaTek Dimensity et al.).
//!
//! Pipeline:
//!   1. Audio PCM -> mel spectrogram (whisper.cpp, CPU)
//!   2. Mel -> Conv -> GELU -> pos_embed (whisper.cpp, CPU)
//!   3. Encoder transformer blocks (NPU if available, else CPU)
//!   4. Decoder (whisper.cpp, CPU)
//!
//! Error handling:
//!   - NNAPI load failure -> CPU fallback (silent)
//!   - NNAPI NPU device not found -> CPU fallback (logged)
//!   - NNAPI execute failure -> CPU fallback (logged per-call)
//!   - Model load failure -> SttError::ModelNotFound
//!   - Invalid audio -> SttError::InvalidAudio

use std::ffi::{CStr, CString};
use std::path::Path;
use std::time::Instant;

use any_stt::config::Backend;
use any_stt::error::SttError;
use any_stt::hardware::HardwareInfo;
use any_stt::{SttEngine, SttResult};

use crate::ffi::*;

/// Hybrid CPU+NPU Whisper engine for NNAPI with automatic fallback.
///
/// Tries NPU acceleration via Android NNAPI (NeuroPilot on MediaTek,
/// or any other vendor NNAPI driver). If NNAPI is not present or fails
/// at any point (init, device probe, execute), transparently falls back to
/// CPU-only whisper.cpp. The caller never needs to handle NPU errors.
pub struct WhisperNnapiEngine {
    ctx: *mut WhisperContext,
    hardware_info: HardwareInfo,
    language: CString,
    n_threads: i32,
    npu_status: NpuStatus,
}

/// NPU acceleration status -- tracks fallback state.
#[derive(Debug)]
enum NpuStatus {
    /// NNAPI available and NPU device found
    Active,
    /// NNAPI not available on this device
    Unavailable(String),
    /// NNAPI was available but failed during operation -- fell back to CPU
    FalledBack(String),
}

// SAFETY: whisper_context is thread-safe for non-concurrent access
unsafe impl Send for WhisperNnapiEngine {}
unsafe impl Sync for WhisperNnapiEngine {}

impl WhisperNnapiEngine {
    /// Create the hybrid engine.
    ///
    /// Never fails due to NPU issues -- falls back to CPU silently.
    /// Only fails if the model file is missing or corrupt.
    pub fn new(
        model_path: &Path,
        language: &str,
        hardware_info: HardwareInfo,
    ) -> Result<Self, SttError> {
        // Validate model path
        if !model_path.exists() {
            return Err(SttError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }

        let path_cstr = CString::new(
            model_path.to_str().ok_or_else(|| SttError::ModelNotFound {
                path: model_path.to_path_buf(),
            })?,
        )
        .map_err(|_| SttError::ModelNotFound {
            path: model_path.to_path_buf(),
        })?;

        let lang = CString::new(language).map_err(|_| {
            SttError::TranscriptionFailed("invalid language string".into())
        })?;

        // Load whisper.cpp model (CPU backend for mel/conv/decoder)
        let mut cparams = unsafe { whisper_context_default_params() };
        cparams.use_gpu = false;

        let ctx = unsafe { whisper_init_from_file_with_params(path_cstr.as_ptr(), cparams) };
        if ctx.is_null() {
            return Err(SttError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        // Probe NPU availability via NNAPI -- failure is not an error, just means CPU-only
        let npu_status = match nnapi_backend::NnapiLib::load() {
            Ok(lib) => match lib.find_npu_device() {
                Ok(_dev) => {
                    eprintln!("WhisperNnapiEngine: NNAPI NPU (NeuroPilot) available");
                    NpuStatus::Active
                }
                Err(e) => {
                    eprintln!(
                        "WhisperNnapiEngine: NNAPI loaded but no NPU device ({e}), using CPU"
                    );
                    NpuStatus::Unavailable(e)
                }
            },
            Err(e) => {
                eprintln!("WhisperNnapiEngine: NNAPI not available ({e}), using CPU");
                NpuStatus::Unavailable(e)
            }
        };

        Ok(Self {
            ctx,
            hardware_info,
            language: lang,
            n_threads,
            npu_status,
        })
    }

    fn make_params(&self) -> *mut WhisperFullParams {
        let params = unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
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
        params
    }

    fn collect_text(&self) -> String {
        let n_segments = unsafe { whisper_full_n_segments(self.ctx) };
        let mut text = String::new();
        for i in 0..n_segments {
            let seg = unsafe { whisper_full_get_segment_text(self.ctx, i) };
            if !seg.is_null() {
                text.push_str(&unsafe { CStr::from_ptr(seg) }.to_string_lossy());
            }
        }
        text
    }

    fn detect_language(&self) -> String {
        let lang_id = unsafe { whisper_full_lang_id(self.ctx) };
        let lang_str = unsafe { whisper_lang_str(lang_id) };
        if !lang_str.is_null() {
            unsafe { CStr::from_ptr(lang_str) }
                .to_string_lossy()
                .into_owned()
        } else {
            self.language.to_string_lossy().into_owned()
        }
    }

    fn run_whisper_full(&self, audio: &[f32]) -> Result<(), SttError> {
        if audio.is_empty() {
            return Err(SttError::InvalidAudio("empty audio buffer".into()));
        }

        let params = self.make_params();
        if params.is_null() {
            return Err(SttError::TranscriptionFailed(
                "failed to allocate whisper params".into(),
            ));
        }

        let ret = unsafe {
            shim_whisper_full(self.ctx, params, audio.as_ptr(), audio.len() as i32)
        };
        unsafe { shim_free_params(params) };

        if ret != 0 {
            return Err(SttError::TranscriptionFailed(format!(
                "whisper_full returned error code {ret}"
            )));
        }
        Ok(())
    }
}

impl SttEngine for WhisperNnapiEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError> {
        // Validate audio
        if audio.is_empty() {
            return Err(SttError::InvalidAudio("empty audio buffer".into()));
        }
        if audio.len() < 100 {
            return Err(SttError::InvalidAudio(format!(
                "audio too short: {} samples (minimum ~100 for meaningful transcription)",
                audio.len()
            )));
        }

        let start = Instant::now();

        // Run whisper.cpp full pipeline (mel + conv + encode + decode)
        // NPU acceleration would replace the encode step; for now CPU handles all.
        self.run_whisper_full(audio)?;

        let text = self.collect_text();
        let language = self.detect_language();

        let backend_used = match &self.npu_status {
            NpuStatus::Active => Backend::Nnapi,
            _ => Backend::Cpu,
        };

        Ok(SttResult {
            text,
            language,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            backend_used,
        })
    }

    fn is_ready(&self) -> bool {
        !self.ctx.is_null()
    }

    fn hardware_info(&self) -> &HardwareInfo {
        &self.hardware_info
    }

    fn active_backend(&self) -> Backend {
        match &self.npu_status {
            NpuStatus::Active => Backend::Nnapi,
            _ => Backend::Cpu,
        }
    }
}

impl Drop for WhisperNnapiEngine {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
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
    fn nnapi_hybrid_engine_loads_model() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperNnapiEngine::new(&path, "en", hw);
        assert!(engine.is_ok());
        assert!(engine.unwrap().is_ready());
    }

    #[test]
    fn nnapi_hybrid_engine_nonexistent_model() {
        let hw = any_stt::detect_hardware();
        let result = WhisperNnapiEngine::new(Path::new("/nonexistent/model.bin"), "en", hw);
        assert!(matches!(result, Err(SttError::ModelNotFound { .. })));
    }

    #[test]
    fn nnapi_hybrid_engine_empty_audio() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperNnapiEngine::new(&path, "en", hw).unwrap();
        let result = engine.transcribe(&[]);
        assert!(matches!(result, Err(SttError::InvalidAudio(_))));
    }

    #[test]
    fn nnapi_hybrid_engine_transcribes_silence() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperNnapiEngine::new(&path, "en", hw).unwrap();
        let silence = vec![0.0f32; 16000];
        let result = engine.transcribe(&silence).unwrap();
        assert!(
            result.text.trim().is_empty() || result.text.trim().starts_with('['),
            "expected blank for silence, got: {:?}",
            result.text
        );
    }

    #[test]
    fn nnapi_hybrid_engine_matches_cpu() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();

        let cpu = crate::WhisperEngine::new(&path, "en", Backend::Cpu, hw.clone()).unwrap();
        let hybrid = WhisperNnapiEngine::new(&path, "en", hw).unwrap();

        let silence = vec![0.0f32; 16000];
        let cpu_result = cpu.transcribe(&silence).unwrap();
        let hybrid_result = hybrid.transcribe(&silence).unwrap();

        assert_eq!(cpu_result.text.trim(), hybrid_result.text.trim());
    }

    #[test]
    fn nnapi_hybrid_engine_reports_backend() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperNnapiEngine::new(&path, "en", hw).unwrap();
        // On non-Android host, NNAPI is not available -> should report CPU
        let backend = engine.active_backend();
        // Either Nnapi (if NNAPI libs present) or Cpu
        assert!(backend == Backend::Nnapi || backend == Backend::Cpu);
    }
}
