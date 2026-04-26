use std::path::PathBuf;

use crate::model::{Model, WhisperVariant};

/// Configuration for initializing an STT engine.
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// Language code (e.g. "ja", "en").
    pub language: String,
    /// Model to load. See [`Model`] for available families and variants.
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
            model: Model::Whisper(WhisperVariant::Small),
            model_path: None,
            quantization: None,
            backend: None,
            sample_rate: 16000,
            allow_cold_vulkan: false,
        }
    }
}

/// GGUF / ggml quantization formats.
///
/// These are shared across all families that load weights through ggml
/// (Whisper, FastConformer-based families, Qwen3 LLM). If a family introduces
/// its own quantization scheme (e.g. GPTQ/AWQ for LLM-only paths) it should
/// extend this enum rather than bypass it.
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
