//! Raw C FFI bindings to whisper.h and our shim layer.
//!
//! The whisper.h API is used directly for context lifecycle and result access.
//! For `whisper_full_params` (a large, version-dependent struct), we use a C
//! shim that allocates params on the heap and exposes setter functions.

use std::os::raw::{c_char, c_float, c_int, c_void};

/// Opaque whisper context.
#[repr(C)]
pub struct WhisperContext {
    _opaque: [u8; 0],
}

/// Opaque whisper_full_params (managed via shim).
#[repr(C)]
pub struct WhisperFullParams {
    _opaque: [u8; 0],
}

/// Context parameters for `whisper_init_from_file_with_params`.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct WhisperContextParams {
    pub use_gpu: bool,
    pub flash_attn: bool,
    pub gpu_device: c_int,
    pub dtw_token_timestamps: bool,
    pub dtw_aheads_preset: c_int,
    pub dtw_n_top: c_int,
    pub dtw_aheads: WhisperAheads,
    pub dtw_mem_size: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WhisperAhead {
    pub n_text_layer: c_int,
    pub n_head: c_int,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct WhisperAheads {
    pub n_heads: usize,
    pub heads: *const WhisperAhead,
}

/// Sampling strategy for whisper_full.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperSamplingStrategy {
    Greedy = 0,
    BeamSearch = 1,
}

extern "C" {
    // --- whisper.h direct bindings ---

    pub fn whisper_init_from_file_with_params(
        path_model: *const c_char,
        params: WhisperContextParams,
    ) -> *mut WhisperContext;

    pub fn whisper_free(ctx: *mut WhisperContext);

    pub fn whisper_context_default_params() -> WhisperContextParams;

    pub fn whisper_full_n_segments(ctx: *mut WhisperContext) -> c_int;

    pub fn whisper_full_get_segment_text(
        ctx: *mut WhisperContext,
        i_segment: c_int,
    ) -> *const c_char;

    pub fn whisper_full_lang_id(ctx: *mut WhisperContext) -> c_int;

    pub fn whisper_lang_str(id: c_int) -> *const c_char;
    pub fn whisper_lang_id(lang: *const c_char) -> c_int;

    pub fn whisper_print_timings(ctx: *mut WhisperContext);
    pub fn whisper_reset_timings(ctx: *mut WhisperContext);

    pub fn whisper_print_system_info() -> *const c_char;

    pub fn whisper_log_set(
        log_callback: Option<
            unsafe extern "C" fn(level: c_int, text: *const c_char, user_data: *mut c_void),
        >,
        user_data: *mut c_void,
    );

    // --- Shim functions (csrc/shim.c) ---

    /// Allocate a heap-allocated whisper_full_params with defaults.
    pub fn shim_default_params(strategy: WhisperSamplingStrategy) -> *mut WhisperFullParams;

    /// Free params allocated by `shim_default_params`.
    pub fn shim_free_params(params: *mut WhisperFullParams);

    /// Set the language field.
    pub fn shim_params_set_language(params: *mut WhisperFullParams, lang: *const c_char);

    /// Set number of threads.
    pub fn shim_params_set_n_threads(params: *mut WhisperFullParams, n: c_int);

    /// Set translate mode (true = translate to English).
    pub fn shim_params_set_translate(params: *mut WhisperFullParams, translate: bool);

    /// Disable timestamp generation.
    pub fn shim_params_set_no_timestamps(params: *mut WhisperFullParams, val: bool);

    /// Force single-segment output.
    pub fn shim_params_set_single_segment(params: *mut WhisperFullParams, val: bool);

    /// Suppress printing special tokens.
    pub fn shim_params_set_print_special(params: *mut WhisperFullParams, val: bool);

    /// Suppress printing progress.
    pub fn shim_params_set_print_progress(params: *mut WhisperFullParams, val: bool);

    /// Suppress realtime output.
    pub fn shim_params_set_print_realtime(params: *mut WhisperFullParams, val: bool);

    /// Suppress printing timestamps.
    pub fn shim_params_set_print_timestamps(params: *mut WhisperFullParams, val: bool);

    /// Suppress non-speech tokens.
    pub fn shim_params_set_suppress_nst(params: *mut WhisperFullParams, val: bool);

    /// Run whisper_full via the shim (params passed by pointer, dereferenced in C).
    pub fn shim_whisper_full(
        ctx: *mut WhisperContext,
        params: *const WhisperFullParams,
        samples: *const c_float,
        n_samples: c_int,
    ) -> c_int;
}
