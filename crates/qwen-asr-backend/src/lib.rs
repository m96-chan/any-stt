//! Qwen3-ASR backend for any-stt.
//!
//! Target models:
//! - **Qwen3-ASR-1.7B** (primary) — <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
//! - **Qwen3-ASR-0.6B** (light)   — <https://huggingface.co/Qwen/Qwen3-ASR-0.6B>
//!
//! Apache-2.0, Safetensors. Based on Qwen3-Omni. Supports 30+ languages
//! including **Japanese, Korean, Chinese** — the only one of the three new
//! families that covers Japanese alongside the Asian and European set.
//!
//! # Architecture
//!
//! Unlike the NeMo FastConformer families, Qwen3-ASR is a multimodal LLM:
//! audio encoder (CTC-style) → audio→text projector → Qwen3 decoder-only LLM.
//! Text generation is autoregressive through the LLM, conditioned on
//! audio-derived tokens.
//!
//! # Status
//!
//! **Skeleton only.** Going from skeleton to functional requires:
//! 1. Conversion script (`scripts/convert-qwen-asr-to-gguf.py` — TODO) that
//!    splits Safetensors → two GGUFs: audio encoder + Qwen3 LLM decoder.
//! 2. Audio encoder GGUF loader (likely shares mel preprocessing with
//!    the NeMo family but different projection / post-processing).
//! 3. Qwen3 LLM runtime. Options under review:
//!    - `llama-cpp-2` (binding to llama.cpp; most mature but big dep)
//!    - direct ggml FFI with custom Qwen3 graph (more control)
//!    - `mistral.rs` (Pure Rust, lighter but less proven)
//!
//! These decisions should be made before any non-trivial code lands here —
//! the LLM runtime choice dictates the crate's shape significantly.

use std::path::{Path, PathBuf};

use any_stt::{Backend, HardwareInfo, SttEngine, SttError, SttResult};

pub struct QwenAsrEngine {
    model_path: PathBuf,
    hardware_info: HardwareInfo,
    backend: Backend,
    language: String,
}

impl QwenAsrEngine {
    pub fn new(
        model_path: &Path,
        language: &str,
        backend: Backend,
        hardware_info: HardwareInfo,
    ) -> Result<Self, SttError> {
        if !model_path.exists() {
            return Err(SttError::ModelNotFound {
                path: model_path.to_path_buf(),
            });
        }
        Ok(Self {
            model_path: model_path.to_path_buf(),
            hardware_info,
            backend,
            language: language.to_string(),
        })
    }
}

impl SttEngine for QwenAsrEngine {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError> {
        if audio.is_empty() {
            return Err(SttError::InvalidAudio("empty audio buffer".into()));
        }
        // TODO(#N6): audio encoder → projector → Qwen3 LLM autoregressive decode.
        Err(SttError::NotImplemented(format!(
            "QwenAsrEngine::transcribe not yet implemented — \
             loaded {}, backend={:?}, language={}. \
             LLM runtime selection is pending (llama-cpp-2 vs direct ggml \
             vs mistral.rs).",
            self.model_path.display(),
            self.backend,
            self.language,
        )))
    }

    fn is_ready(&self) -> bool {
        self.model_path.exists()
    }

    fn hardware_info(&self) -> &HardwareInfo {
        &self.hardware_info
    }

    fn active_backend(&self) -> Backend {
        self.backend
    }
}

pub fn initialize(config: &any_stt::SttConfig) -> Result<Box<dyn SttEngine>, SttError> {
    let model_path = config.model_path.as_ref().ok_or_else(|| {
        SttError::TranscriptionFailed("model_path is required".into())
    })?;
    let hw = any_stt::detect_hardware();
    let selection = any_stt::selector::select(config, &hw);
    let engine = QwenAsrEngine::new(model_path, &config.language, selection.backend, hw)?;
    Ok(Box::new(engine))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_model_returns_error() {
        let hw = any_stt::detect_hardware();
        let result = QwenAsrEngine::new(
            Path::new("/does/not/exist.gguf"),
            "ja",
            Backend::Cpu,
            hw,
        );
        assert!(matches!(result, Err(SttError::ModelNotFound { .. })));
    }
}
