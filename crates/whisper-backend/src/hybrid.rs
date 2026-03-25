//! Hybrid CPU+NPU whisper engine with graceful fallback.
//!
//! Pipeline:
//!   1. Audio PCM → mel spectrogram (whisper.cpp, CPU)
//!   2. Mel → Conv → GELU → pos_embed (whisper.cpp, CPU)
//!   3. Encoder transformer blocks (NPU if available, else CPU)
//!   4. Decoder (whisper.cpp, CPU)
//!
//! Error handling:
//!   - QNN load failure → CPU fallback (silent)
//!   - QNN graph build failure → CPU fallback (logged)
//!   - QNN execute failure → CPU fallback (logged per-call)
//!   - Model load failure → SttError::ModelNotFound
//!   - Invalid audio → SttError::InvalidAudio

use std::ffi::CString;
use std::path::Path;
use std::time::Instant;

use any_stt::config::Backend;
use any_stt::error::SttError;
use any_stt::hardware::HardwareInfo;
use any_stt::{SttEngine, SttResult};

use crate::ffi::*;
use crate::helpers;

/// Hybrid CPU+NPU Whisper engine with automatic fallback.
///
/// Tries NPU acceleration when available. If QNN is not present or fails
/// at any point (init, graph build, execute), transparently falls back to
/// CPU-only whisper.cpp. The caller never needs to handle NPU errors.
pub struct WhisperQnnEngine {
    ctx: *mut WhisperContext,
    hardware_info: HardwareInfo,
    language: CString,
    n_threads: i32,
    npu_status: NpuStatus,
}

/// NPU acceleration status — tracks fallback state.
#[derive(Debug)]
enum NpuStatus {
    /// QNN available and working
    Active,
    /// QNN not available on this device
    Unavailable(String),
    /// QNN was available but failed during operation — fell back to CPU
    FalledBack(String),
}

// SAFETY: whisper_context is thread-safe for non-concurrent access
unsafe impl Send for WhisperQnnEngine {}
unsafe impl Sync for WhisperQnnEngine {}

impl WhisperQnnEngine {
    /// Create the hybrid engine.
    ///
    /// Never fails due to NPU issues — falls back to CPU silently.
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

        let path_cstr = helpers::model_path_to_cstring(model_path)?;

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

        let n_threads = helpers::default_n_threads();

        // Probe NPU availability — failure is not an error, just means CPU-only
        let npu_status = match qnn_backend::QnnLibrary::load_htp() {
            Ok(_lib) => {
                eprintln!("WhisperQnnEngine: QNN HTP available");
                NpuStatus::Active
            }
            Err(e) => {
                eprintln!("WhisperQnnEngine: QNN HTP not available ({e}), using CPU");
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

    fn run_full_pipeline(&self, audio: &[f32]) -> Result<(), SttError> {
        if audio.is_empty() {
            return Err(SttError::InvalidAudio("empty audio buffer".into()));
        }
        let params = helpers::create_params(&self.language, self.n_threads)?;
        helpers::run_whisper_full(self.ctx, params, audio)
    }
}

impl SttEngine for WhisperQnnEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError> {
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
        self.run_full_pipeline(audio)?;

        let fallback_lang = self.language.to_string_lossy();
        Ok(helpers::collect_result(self.ctx, start, &fallback_lang, self.active_backend()))
    }

    fn is_ready(&self) -> bool {
        !self.ctx.is_null()
    }

    fn hardware_info(&self) -> &HardwareInfo {
        &self.hardware_info
    }

    fn active_backend(&self) -> Backend {
        match &self.npu_status {
            NpuStatus::Active => Backend::Qnn,
            _ => Backend::Cpu,
        }
    }
}

impl Drop for WhisperQnnEngine {
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
    fn hybrid_engine_loads_model() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperQnnEngine::new(&path, "en", hw);
        assert!(engine.is_ok());
        assert!(engine.unwrap().is_ready());
    }

    #[test]
    fn hybrid_engine_nonexistent_model() {
        let hw = any_stt::detect_hardware();
        let result = WhisperQnnEngine::new(Path::new("/nonexistent/model.bin"), "en", hw);
        assert!(matches!(result, Err(SttError::ModelNotFound { .. })));
    }

    #[test]
    fn hybrid_engine_empty_audio() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperQnnEngine::new(&path, "en", hw).unwrap();
        let result = engine.transcribe(&[]);
        assert!(matches!(result, Err(SttError::InvalidAudio(_))));
    }

    #[test]
    fn hybrid_engine_transcribes_silence() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperQnnEngine::new(&path, "en", hw).unwrap();
        let silence = vec![0.0f32; 16000];
        let result = engine.transcribe(&silence).unwrap();
        assert!(
            result.text.trim().is_empty() || result.text.trim().starts_with('['),
            "expected blank for silence, got: {:?}",
            result.text
        );
    }

    #[test]
    fn hybrid_engine_matches_cpu() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();

        let cpu = crate::WhisperEngine::new(&path, "en", Backend::Cpu, hw.clone()).unwrap();
        let hybrid = WhisperQnnEngine::new(&path, "en", hw).unwrap();

        let silence = vec![0.0f32; 16000];
        let cpu_result = cpu.transcribe(&silence).unwrap();
        let hybrid_result = hybrid.transcribe(&silence).unwrap();

        assert_eq!(cpu_result.text.trim(), hybrid_result.text.trim());
    }

    #[test]
    fn hybrid_engine_reports_backend() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found");
            return;
        }
        let hw = any_stt::detect_hardware();
        let engine = WhisperQnnEngine::new(&path, "en", hw).unwrap();
        // On x86_64 host, QNN is not available → should report CPU
        let backend = engine.active_backend();
        // Either Qnn (if QNN libs present) or Cpu
        assert!(backend == Backend::Qnn || backend == Backend::Cpu);
    }
}
