use std::path::PathBuf;

/// Configuration for initializing an STT engine.
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// Language code (e.g. "ja", "en").
    pub language: String,
    /// Whisper model variant to use.
    pub model: Model,
    /// Override: custom model file path.
    pub model_path: Option<PathBuf>,
    /// Override: specific quantization. `None` = auto-select based on hardware.
    pub quantization: Option<Quantization>,
    /// Override: specific backend. `None` = auto-select best available.
    pub backend: Option<Backend>,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Allow Vulkan even when shader cache is cold (causes multi-second startup delay).
    pub allow_cold_vulkan: bool,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            language: "en".into(),
            model: Model::Small,
            model_path: None,
            quantization: None,
            backend: None,
            sample_rate: 16000,
            allow_cold_vulkan: false,
        }
    }
}

/// Whisper model variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    Tiny,
    TinyEn,
    Base,
    BaseEn,
    Small,
    SmallEn,
    Medium,
    MediumEn,
    LargeV1,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
    DistilLargeV2,
    DistilLargeV3,
    DistilMediumEn,
    DistilSmallEn,
    KotobaV1,
    KotobaV2,
    /// Custom model specified by HuggingFace model ID.
    Custom(String),
}

/// Quantization formats for ggml models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    F16,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0,
}

/// Acceleration backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
    CoreMl,
    Vulkan,
    Nnapi,
    Qnn,
    Cpu,
}
