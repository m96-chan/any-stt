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

    // --- Segment detail accessors ---

    pub fn whisper_full_get_segment_t0(ctx: *mut WhisperContext, i_segment: c_int) -> i64;
    pub fn whisper_full_get_segment_t1(ctx: *mut WhisperContext, i_segment: c_int) -> i64;
    pub fn whisper_full_n_tokens(ctx: *mut WhisperContext, i_segment: c_int) -> c_int;
    pub fn whisper_full_get_token_id(
        ctx: *mut WhisperContext,
        i_segment: c_int,
        i_token: c_int,
    ) -> c_int;
    pub fn whisper_full_get_token_p(
        ctx: *mut WhisperContext,
        i_segment: c_int,
        i_token: c_int,
    ) -> c_float;

    // --- Model info ---

    pub fn whisper_model_n_vocab(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_audio_ctx(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_audio_state(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_audio_head(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_audio_layer(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_text_ctx(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_text_state(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_text_head(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_text_layer(ctx: *mut WhisperContext) -> c_int;
    pub fn whisper_model_n_mels(ctx: *mut WhisperContext) -> c_int;

    // --- Timings ---

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

    /// Set params to skip encoding (for externally-provided encoder output).
    pub fn shim_params_set_skip_encoder(params: *mut WhisperFullParams, skip: bool);

    /// Get number of audio context frames (1500 for standard whisper).
    pub fn shim_model_n_audio_ctx(ctx: *mut WhisperContext) -> c_int;

    /// Get encoder state dimension (n_state).
    pub fn shim_model_n_audio_state(ctx: *mut WhisperContext) -> c_int;

    // --- Encoder output injection (for hybrid NPU+CPU pipeline) ---

    /// Get conv output (encoder input after mel→conv→gelu→pos_embed).
    pub fn whisper_get_conv_output(
        ctx: *mut WhisperContext,
        n_ctx: *mut c_int,
        n_state: *mut c_int,
    ) -> *mut c_float;

    /// Enable/disable encoder skip mode for decode-only execution.
    /// When enabled, conv still runs but encoder blocks are skipped.
    pub fn whisper_set_skip_encode(ctx: *mut WhisperContext, skip: bool);

    /// Get pointer to internal encoder output tensor data.
    /// Returns null if encoder hasn't been run yet.
    pub fn whisper_get_encoder_output(
        ctx: *mut WhisperContext,
        n_ctx: *mut c_int,
        n_state: *mut c_int,
    ) -> *mut c_float;

    /// Set encoder output from external source (e.g., NPU).
    /// Copies n_ctx*n_state floats into the internal embd_enc tensor.
    /// Returns 0 on success, -1 on failure.
    pub fn whisper_set_encoder_output(
        ctx: *mut WhisperContext,
        data: *const c_float,
        n_ctx: c_int,
        n_state: c_int,
    ) -> c_int;

    // --- Low-level encode/decode API (for hybrid NPU+CPU pipeline) ---

    /// Run the encoder on the audio features starting at the given offset.
    pub fn whisper_encode(ctx: *mut WhisperContext, offset: c_int, n_threads: c_int) -> c_int;

    /// Run the decoder to obtain logits for the next token.
    pub fn whisper_decode(
        ctx: *mut WhisperContext,
        tokens: *const c_int,
        n_tokens: c_int,
        n_past: c_int,
        n_threads: c_int,
    ) -> c_int;

    // --- Model tensor access (for extracting weights to NPU) ---

    /// Get a model tensor dequantized to FP32.
    /// If out is null, returns the number of elements needed.
    pub fn whisper_get_model_tensor_f32(
        ctx: *mut WhisperContext,
        name: *const c_char,
        out: *mut c_float,
        max_elements: c_int,
    ) -> c_int;

    /// Get a model tensor's shape.
    pub fn whisper_get_model_tensor_dims(
        ctx: *mut WhisperContext,
        name: *const c_char,
        dims: *mut c_int,
        max_dims: c_int,
    ) -> c_int;
}
