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

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda => write!(f, "CUDA"),
            Self::Metal => write!(f, "Metal"),
            Self::CoreMl => write!(f, "CoreML"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Nnapi => write!(f, "NNAPI"),
            Self::Qnn => write!(f, "QNN"),
            Self::Cpu => write!(f, "CPU"),
        }
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiny => write!(f, "tiny"),
            Self::TinyEn => write!(f, "tiny.en"),
            Self::Base => write!(f, "base"),
            Self::BaseEn => write!(f, "base.en"),
            Self::Small => write!(f, "small"),
            Self::SmallEn => write!(f, "small.en"),
            Self::Medium => write!(f, "medium"),
            Self::MediumEn => write!(f, "medium.en"),
            Self::LargeV1 => write!(f, "large-v1"),
            Self::LargeV2 => write!(f, "large-v2"),
            Self::LargeV3 => write!(f, "large-v3"),
            Self::LargeV3Turbo => write!(f, "large-v3-turbo"),
            Self::DistilLargeV2 => write!(f, "distil-large-v2"),
            Self::DistilLargeV3 => write!(f, "distil-large-v3"),
            Self::DistilMediumEn => write!(f, "distil-medium.en"),
            Self::DistilSmallEn => write!(f, "distil-small.en"),
            Self::KotobaV1 => write!(f, "kotoba-v1"),
            Self::KotobaV2 => write!(f, "kotoba-v2"),
            Self::Custom(id) => write!(f, "custom({id})"),
        }
    }
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F16 => write!(f, "f16"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q5_1 => write!(f, "q5_1"),
            Self::Q5_0 => write!(f, "q5_0"),
            Self::Q4_1 => write!(f, "q4_1"),
            Self::Q4_0 => write!(f, "q4_0"),
        }
    }
}
