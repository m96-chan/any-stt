pub mod decoder;
pub mod ffi;
pub mod hybrid;
pub mod nnapi_hybrid;
pub mod preprocess;
pub mod qnn;

#[cfg(target_os = "android")]
pub mod dlopen_ffi;

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

    /// Get the raw whisper context pointer (for encoder output injection).
    pub fn raw_ctx(&self) -> *mut WhisperContext {
        self.ctx
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
        if audio.is_empty() {
            return Err(SttError::InvalidAudio("empty audio buffer".into()));
        }

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

/// Initialize the best available STT engine with automatic fallback.
///
/// Backend selection priority:
/// - iOS/macOS: CoreML (ANE) → Metal (GPU) → CPU
/// - Android (Snapdragon): QNN HTP (NPU) → CPU
/// - Linux: CUDA → CPU
///
/// Never fails due to accelerator unavailability — always falls back to CPU.
///
/// # Errors
/// - `SttError::ModelNotFound` if `config.model_path` doesn't exist
/// - `SttError::TranscriptionFailed` if whisper.cpp fails to load the model
pub fn initialize(config: &any_stt::SttConfig) -> Result<Box<dyn SttEngine>, SttError> {
    let model_path = config.model_path.as_ref().ok_or_else(|| {
        SttError::TranscriptionFailed("model_path is required".into())
    })?;

    if !model_path.exists() {
        return Err(SttError::ModelNotFound {
            path: model_path.clone(),
        });
    }

    let hw = any_stt::detect_hardware();
    let selection = any_stt::selector::select(config, &hw);

    eprintln!("initialize: selected backend={:?}, quantization={:?}",
        selection.backend, selection.quantization);

    match selection.backend {
        // Apple platforms: CoreML (ANE) or Metal (GPU)
        // whisper.cpp handles these internally when compiled with WHISPER_COREML / GGML_METAL
        Backend::CoreMl | Backend::Metal => {
            let engine = WhisperEngine::new(
                model_path, &config.language, selection.backend, hw.clone(),
            );
            match engine {
                Ok(e) => {
                    eprintln!("initialize: using {:?} backend", selection.backend);
                    return Ok(Box::new(e));
                }
                Err(e) => {
                    eprintln!("initialize: {:?} failed ({e}), falling back to CPU",
                        selection.backend);
                }
            }
        }

        // Snapdragon: QNN HTP (NPU) with CPU fallback
        Backend::Qnn => {
            match hybrid::WhisperQnnEngine::new(model_path, &config.language, hw.clone()) {
                Ok(engine) => {
                    eprintln!("initialize: using {} backend",
                        if engine.active_backend() == Backend::Qnn { "QNN NPU" } else { "CPU (QNN unavailable)" });
                    return Ok(Box::new(engine));
                }
                Err(e) => {
                    eprintln!("initialize: hybrid engine failed ({e}), falling back to CPU");
                }
            }
        }

        // CUDA / Vulkan: whisper.cpp handles via use_gpu flag
        Backend::Cuda | Backend::Vulkan => {
            let engine = WhisperEngine::new(
                model_path, &config.language, selection.backend, hw.clone(),
            );
            match engine {
                Ok(e) => {
                    eprintln!("initialize: using {:?} backend", selection.backend);
                    return Ok(Box::new(e));
                }
                Err(e) => {
                    eprintln!("initialize: {:?} failed ({e}), falling back to CPU",
                        selection.backend);
                }
            }
        }

        _ => {}
    }

    // Final fallback: CPU-only whisper.cpp
    let engine = WhisperEngine::new(model_path, &config.language, Backend::Cpu, hw)?;
    eprintln!("initialize: using CPU backend");
    Ok(Box::new(engine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap()
    }

    fn model_path() -> PathBuf {
        repo_root().join("third-party/whisper.cpp/models/ggml-tiny.en.bin")
    }

    /// Helper: skip test if model not downloaded.
    fn require_model() -> PathBuf {
        let p = model_path();
        if !p.exists() {
            eprintln!("SKIPPED: model not found at {}", p.display());
        }
        p
    }

    // ---- Constructor tests ----

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
        match result {
            Err(SttError::ModelNotFound { .. }) => {}
            Err(other) => panic!("expected ModelNotFound, got: {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn load_directory_as_model_returns_error() {
        let hw = any_stt::detect_hardware();
        let result = WhisperEngine::new(Path::new("/tmp"), "en", Backend::Cpu, hw);
        assert!(result.is_err());
    }

    #[test]
    fn load_valid_model_succeeds() {
        let model = require_model();
        if !model.exists() {
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperEngine::new(&model, "en", Backend::Cpu, hw);
        assert!(engine.is_ok());
    }

    // ---- SttEngine trait method tests ----

    #[test]
    fn is_ready_after_load() {
        let model = require_model();
        if !model.exists() {
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperEngine::new(&model, "en", Backend::Cpu, hw).unwrap();
        assert!(engine.is_ready());
    }

    #[test]
    fn active_backend_matches_constructor_arg() {
        let model = require_model();
        if !model.exists() {
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperEngine::new(&model, "en", Backend::Cpu, hw).unwrap();
        assert_eq!(engine.active_backend(), Backend::Cpu);
    }

    #[test]
    fn hardware_info_is_accessible() {
        let model = require_model();
        if !model.exists() {
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperEngine::new(&model, "en", Backend::Cpu, hw).unwrap();
        let info = engine.hardware_info();
        assert!(!info.cpu.arch.is_empty());
        assert!(info.cpu.cores > 0);
    }

    #[test]
    fn transcribe_empty_audio_returns_empty_text() {
        let model = require_model();
        if !model.exists() {
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperEngine::new(&model, "en", Backend::Cpu, hw).unwrap();
        // 1 second of silence at 16kHz.
        let silence = vec![0.0f32; 16000];
        let result = engine.transcribe(&silence).unwrap();
        // Silence should produce empty or whitespace-only text.
        assert!(
            result.text.trim().is_empty() || result.text.trim().starts_with('['),
            "expected empty/blank output for silence, got: {:?}",
            result.text
        );
        assert_eq!(result.backend_used, Backend::Cpu);
        assert!(result.duration_ms > 0.0);
    }

    #[test]
    fn transcribe_returns_valid_language() {
        let model = require_model();
        if !model.exists() {
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperEngine::new(&model, "en", Backend::Cpu, hw).unwrap();
        let silence = vec![0.0f32; 16000];
        let result = engine.transcribe(&silence).unwrap();
        assert_eq!(result.language, "en");
    }

    // ---- FFI shim tests ----

    #[test]
    fn shim_default_params_allocates_and_frees() {
        let params = unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
        assert!(!params.is_null());
        unsafe { shim_free_params(params) };
    }

    #[test]
    fn shim_default_params_beam_search() {
        let params = unsafe { shim_default_params(WhisperSamplingStrategy::BeamSearch) };
        assert!(!params.is_null());
        unsafe { shim_free_params(params) };
    }

    #[test]
    fn shim_param_setters_dont_crash() {
        let params = unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
        assert!(!params.is_null());
        let lang = CString::new("ja").unwrap();
        unsafe {
            shim_params_set_language(params, lang.as_ptr());
            shim_params_set_n_threads(params, 4);
            shim_params_set_translate(params, true);
            shim_params_set_no_timestamps(params, true);
            shim_params_set_single_segment(params, true);
            shim_params_set_print_special(params, false);
            shim_params_set_print_progress(params, false);
            shim_params_set_print_realtime(params, false);
            shim_params_set_print_timestamps(params, false);
            shim_params_set_suppress_nst(params, true);
            shim_free_params(params);
        }
    }

    #[test]
    fn whisper_context_default_params_returns_valid() {
        let params = unsafe { whisper_context_default_params() };
        // GPU should be enabled by default in whisper.cpp.
        assert!(params.use_gpu);
    }

    #[test]
    fn whisper_lang_id_roundtrip() {
        let lang = CString::new("en").unwrap();
        let id = unsafe { whisper_lang_id(lang.as_ptr()) };
        assert!(id >= 0, "expected valid lang id for 'en'");
        let str_ptr = unsafe { whisper_lang_str(id) };
        assert!(!str_ptr.is_null());
        let s = unsafe { CStr::from_ptr(str_ptr) }.to_str().unwrap();
        assert_eq!(s, "en");
    }

    #[test]
    fn whisper_lang_id_unknown_returns_negative() {
        let lang = CString::new("zzznotareal").unwrap();
        let id = unsafe { whisper_lang_id(lang.as_ptr()) };
        assert!(id < 0, "expected negative for unknown language");
    }

    #[test]
    fn whisper_print_system_info_returns_nonnull() {
        let info = unsafe { whisper_print_system_info() };
        assert!(!info.is_null());
        let s = unsafe { CStr::from_ptr(info) }.to_str().unwrap();
        assert!(!s.is_empty());
    }

    #[test]
    fn transcribe_empty_audio_returns_error() {
        let model = require_model();
        if !model.exists() { return; }
        let hw = any_stt::detect_hardware();
        let engine = WhisperEngine::new(&model, "en", Backend::Cpu, hw).unwrap();
        let result = engine.transcribe(&[]);
        assert!(matches!(result, Err(SttError::InvalidAudio(_))));
    }

    #[test]
    fn initialize_with_valid_model() {
        let model = require_model();
        if !model.exists() { return; }
        let config = any_stt::SttConfig {
            model_path: Some(model),
            ..Default::default()
        };
        let engine = initialize(&config);
        assert!(engine.is_ok());
        assert!(engine.unwrap().is_ready());
    }

    #[test]
    fn initialize_with_missing_model() {
        let config = any_stt::SttConfig {
            model_path: Some(PathBuf::from("/nonexistent/model.bin")),
            ..Default::default()
        };
        let result = initialize(&config);
        assert!(matches!(result, Err(SttError::ModelNotFound { .. })));
    }

    #[test]
    fn initialize_without_model_path() {
        let config = any_stt::SttConfig::default();
        let result = initialize(&config);
        assert!(result.is_err());
    }
}
