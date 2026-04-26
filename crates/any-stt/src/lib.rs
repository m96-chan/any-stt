pub mod config;
pub mod detect;
pub mod error;
pub mod hardware;
pub mod model;
pub mod selector;

pub use config::{Backend, Quantization, SttConfig};
pub use error::SttError;
pub use hardware::HardwareInfo;
pub use model::{
    Model, ModelFamily, ParakeetVariant, QwenAsrVariant, ReazonSpeechVariant, WhisperVariant,
};
pub use selector::Selection;

/// Result of a transcription.
#[derive(Debug, Clone)]
pub struct SttResult {
    pub text: String,
    pub language: String,
    pub duration_ms: f64,
    pub backend_used: Backend,
}

/// Core trait that all backends implement.
///
/// Each family's backend crate (`whisper-backend`, `reazonspeech-backend`,
/// `parakeet-backend`, `qwen-asr-backend`) provides its own engine types that
/// implement this trait plus a concrete `initialize()` factory.
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

/// Detect hardware, select the best backend and quantization, and return a
/// diagnostic pointing at the family-specific backend crate.
///
/// `any-stt` deliberately does not depend on the backend crates (to keep
/// the dependency tree one-way: `*-backend` → `any-stt`). Callers should
/// invoke `initialize()` on the crate that matches `config.model.family()`:
///
/// - [`ModelFamily::Whisper`] → `whisper_backend::initialize`
/// - [`ModelFamily::ReazonSpeech`] → `reazonspeech_backend::initialize`
/// - [`ModelFamily::Parakeet`] → `parakeet_backend::initialize`
/// - [`ModelFamily::QwenAsr`] → `qwen_asr_backend::initialize`
///
/// This function returns [`SttError::NotImplemented`] with a message that
/// names the correct backend crate for programmatic use.
pub fn initialize(config: SttConfig) -> Result<Box<dyn SttEngine>, SttError> {
    let hw = detect::detect_hardware();
    let selection = selector::select(&config, &hw);
    let family = config.model.family();

    Err(SttError::NotImplemented(format!(
        "family={} backend={:?} quantization={:?} — call {}::initialize() directly",
        family.label(),
        selection.backend,
        selection.quantization,
        family.backend_crate(),
    )))
}

/// Detect hardware capabilities without loading a model.
pub fn detect_hardware() -> HardwareInfo {
    detect::detect_hardware()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = SttConfig::default();
        assert_eq!(config.language, "en");
        assert_eq!(config.model, Model::Whisper(WhisperVariant::Small));
        assert_eq!(config.sample_rate, 16000);
        assert!(!config.allow_cold_vulkan);
        assert!(config.backend.is_none());
        assert!(config.quantization.is_none());
        assert!(config.model_path.is_none());
    }

    #[test]
    fn initialize_returns_error_pointing_at_backend_crate() {
        let result = initialize(SttConfig::default());
        match result {
            Err(SttError::NotImplemented(msg)) => {
                assert!(
                    msg.contains("whisper_backend"),
                    "default config is Whisper family; message should name \
                     whisper_backend crate, got: {msg}"
                );
            }
            Err(e) => panic!("expected NotImplemented, got different error: {e}"),
            Ok(_) => panic!("expected NotImplemented, got Ok"),
        }
    }

    #[test]
    fn initialize_for_reazonspeech_names_its_backend_crate() {
        let config = SttConfig {
            model: Model::ReazonSpeech(ReazonSpeechVariant::NemoV2),
            ..Default::default()
        };
        match initialize(config) {
            Err(SttError::NotImplemented(msg)) => {
                assert!(msg.contains("reazonspeech_backend"), "got: {msg}");
            }
            Err(e) => panic!("expected NotImplemented, got different error: {e}"),
            Ok(_) => panic!("expected NotImplemented, got Ok"),
        }
    }

    #[test]
    fn initialize_for_parakeet_names_its_backend_crate() {
        let config = SttConfig {
            model: Model::Parakeet(ParakeetVariant::Tdt0_6bV3),
            ..Default::default()
        };
        match initialize(config) {
            Err(SttError::NotImplemented(msg)) => {
                assert!(msg.contains("parakeet_backend"), "got: {msg}");
            }
            Err(e) => panic!("expected NotImplemented, got different error: {e}"),
            Ok(_) => panic!("expected NotImplemented, got Ok"),
        }
    }

    #[test]
    fn initialize_for_qwen_asr_names_its_backend_crate() {
        let config = SttConfig {
            model: Model::QwenAsr(QwenAsrVariant::B1_7),
            ..Default::default()
        };
        match initialize(config) {
            Err(SttError::NotImplemented(msg)) => {
                assert!(msg.contains("qwen_asr_backend"), "got: {msg}");
            }
            Err(e) => panic!("expected NotImplemented, got different error: {e}"),
            Ok(_) => panic!("expected NotImplemented, got Ok"),
        }
    }

    #[test]
    fn detect_hardware_returns_valid_info() {
        let hw = detect_hardware();
        assert!(!hw.cpu.arch.is_empty());
        assert!(hw.cpu.cores > 0);
    }

    #[test]
    fn backend_eq() {
        assert_eq!(Backend::Cuda, Backend::Cuda);
        assert_ne!(Backend::Cuda, Backend::Metal);
    }

    #[test]
    fn model_custom_defaults_to_whisper_family() {
        let m = Model::Custom("kotoba-tech/kotoba-whisper-v2.0".into());
        assert_eq!(m.family(), ModelFamily::Whisper);
        assert_eq!(
            m,
            Model::Custom("kotoba-tech/kotoba-whisper-v2.0".into())
        );
        assert_ne!(m, Model::Whisper(WhisperVariant::KotobaV2));
    }
}
