pub mod config;
pub mod error;
pub mod hardware;

pub use config::{Backend, Model, Quantization, SttConfig};
pub use error::SttError;
pub use hardware::HardwareInfo;

/// Result of a transcription.
#[derive(Debug, Clone)]
pub struct SttResult {
    pub text: String,
    pub language: String,
    pub duration_ms: f64,
    pub backend_used: Backend,
}

/// Core trait that all backends implement.
pub trait SttEngine: Send + Sync {
    /// Transcribe audio samples (f32 PCM, mono, at the configured sample rate).
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError>;

    /// Whether the engine has loaded a model and is ready to transcribe.
    fn is_ready(&self) -> bool;

    /// Hardware information detected at initialization.
    fn hardware_info(&self) -> &HardwareInfo;

    /// The acceleration backend currently in use.
    fn active_backend(&self) -> Backend;
}

/// Detect hardware, select the best backend and quantization, and load a model.
///
/// Returns a boxed [`SttEngine`] ready to transcribe audio.
pub fn initialize(_config: SttConfig) -> Result<Box<dyn SttEngine>, SttError> {
    Err(SttError::NotImplemented(
        "no backends compiled — enable a feature flag (e.g. `cuda`, `metal`, `cpu`)".into(),
    ))
}

/// Detect hardware capabilities without loading a model.
pub fn detect_hardware() -> HardwareInfo {
    // Stub: returns empty/default info until real detection is implemented.
    HardwareInfo {
        cpu: hardware::CpuInfo {
            arch: std::env::consts::ARCH.into(),
            features: Vec::new(),
            cores: 0,
        },
        gpu: None,
        npu: None,
        os: hardware::OsInfo {
            platform: if cfg!(target_os = "macos") {
                hardware::Platform::MacOs
            } else if cfg!(target_os = "linux") {
                hardware::Platform::Linux
            } else if cfg!(target_os = "android") {
                hardware::Platform::Android
            } else if cfg!(target_os = "ios") {
                hardware::Platform::Ios
            } else {
                hardware::Platform::Linux // fallback
            },
            version: String::new(),
        },
        available_ram_mb: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = SttConfig::default();
        assert_eq!(config.language, "en");
        assert_eq!(config.model, Model::Small);
        assert_eq!(config.sample_rate, 16000);
        assert!(!config.allow_cold_vulkan);
        assert!(config.backend.is_none());
        assert!(config.quantization.is_none());
        assert!(config.model_path.is_none());
    }

    #[test]
    fn initialize_returns_not_implemented() {
        let result = initialize(SttConfig::default());
        match result {
            Err(SttError::NotImplemented(_)) => {}
            Err(other) => panic!("expected NotImplemented, got: {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn detect_hardware_returns_stub() {
        let hw = detect_hardware();
        assert!(!hw.cpu.arch.is_empty());
        assert!(hw.gpu.is_none());
        assert!(hw.npu.is_none());
    }

    #[test]
    fn backend_eq() {
        assert_eq!(Backend::Cuda, Backend::Cuda);
        assert_ne!(Backend::Cuda, Backend::Metal);
    }

    #[test]
    fn model_custom() {
        let m = Model::Custom("kotoba-tech/kotoba-whisper-v2.0".into());
        assert_eq!(m, Model::Custom("kotoba-tech/kotoba-whisper-v2.0".into()));
        assert_ne!(m, Model::KotobaV2);
    }
}
