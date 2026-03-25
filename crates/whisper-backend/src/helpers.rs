//! Shared helpers for whisper.cpp FFI operations.
//!
//! Extracts common patterns used across WhisperEngine, WhisperCppDecoder,
//! and WhisperQnnEngine to reduce duplication.

use std::ffi::{CStr, CString};
use std::path::Path;

use any_stt::config::Backend;
use any_stt::error::SttError;
use any_stt::SttResult;

use crate::ffi::*;

/// Convert a model path to a CString, returning ModelNotFound on failure.
pub fn model_path_to_cstring(path: &Path) -> Result<CString, SttError> {
    let s = path.to_str().ok_or_else(|| SttError::ModelNotFound {
        path: path.to_path_buf(),
    })?;
    CString::new(s).map_err(|_| SttError::ModelNotFound {
        path: path.to_path_buf(),
    })
}

/// Create and configure whisper_full_params with common defaults.
///
/// Returns a heap-allocated params pointer that must be freed with `shim_free_params`.
pub fn create_params(language: &CStr, n_threads: i32) -> Result<*mut WhisperFullParams, SttError> {
    let params = unsafe { shim_default_params(WhisperSamplingStrategy::Greedy) };
    if params.is_null() {
        return Err(SttError::TranscriptionFailed(
            "failed to allocate whisper params".into(),
        ));
    }
    unsafe {
        shim_params_set_language(params, language.as_ptr());
        shim_params_set_n_threads(params, n_threads);
        shim_params_set_translate(params, false);
        shim_params_set_print_special(params, false);
        shim_params_set_print_progress(params, false);
        shim_params_set_print_realtime(params, false);
        shim_params_set_print_timestamps(params, false);
        shim_params_set_single_segment(params, false);
        shim_params_set_suppress_nst(params, true);
    }
    Ok(params)
}

/// Run whisper_full and free params, returning an error on failure.
pub fn run_whisper_full(
    ctx: *mut WhisperContext,
    params: *mut WhisperFullParams,
    audio: &[f32],
) -> Result<(), SttError> {
    let ret = unsafe {
        shim_whisper_full(ctx, params, audio.as_ptr(), audio.len() as i32)
    };
    unsafe { shim_free_params(params) };
    if ret != 0 {
        return Err(SttError::TranscriptionFailed(format!(
            "whisper_full returned error code {ret}"
        )));
    }
    Ok(())
}

/// Collect transcription text from all segments in the whisper context.
pub fn collect_text(ctx: *mut WhisperContext) -> String {
    let n_segments = unsafe { whisper_full_n_segments(ctx) };
    let mut text = String::new();
    for i in 0..n_segments {
        let seg = unsafe { whisper_full_get_segment_text(ctx, i) };
        if !seg.is_null() {
            text.push_str(&unsafe { CStr::from_ptr(seg) }.to_string_lossy());
        }
    }
    text
}

/// Detect the language from the whisper context after transcription.
pub fn detect_language(ctx: *mut WhisperContext, fallback: &str) -> String {
    let lang_id = unsafe { whisper_full_lang_id(ctx) };
    let lang_str = unsafe { whisper_lang_str(lang_id) };
    if !lang_str.is_null() {
        unsafe { CStr::from_ptr(lang_str) }
            .to_string_lossy()
            .into_owned()
    } else {
        fallback.to_string()
    }
}

/// Build an SttResult from the whisper context after a successful transcription.
pub fn collect_result(
    ctx: *mut WhisperContext,
    start: std::time::Instant,
    fallback_lang: &str,
    backend: Backend,
) -> SttResult {
    SttResult {
        text: collect_text(ctx),
        language: detect_language(ctx, fallback_lang),
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        backend_used: backend,
    }
}

/// Get the default number of threads for inference.
pub fn default_n_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}
