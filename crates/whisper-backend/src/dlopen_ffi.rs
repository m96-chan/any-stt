//! dlopen-based FFI for whisper.cpp (Android).
//!
//! On Android, whisper.cpp is loaded as a shared library at runtime
//! to avoid NDK static libc initialization crashes.
//! On other platforms, this module is not used (direct static linking).

use std::os::raw::{c_char, c_float, c_int, c_void};
use std::path::Path;

/// All whisper + shim function pointers resolved from .so files.
pub struct WhisperFunctions {
    _shim_lib: libloading::Library,
    _whisper_lib: libloading::Library,

    // whisper.h functions
    pub whisper_init_from_file_with_params:
        unsafe extern "C" fn(*const c_char, ffi::WhisperContextParams) -> *mut c_void,
    pub whisper_free: unsafe extern "C" fn(*mut c_void),
    pub whisper_context_default_params: unsafe extern "C" fn() -> ffi::WhisperContextParams,
    pub whisper_full_n_segments: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_full_get_segment_text:
        unsafe extern "C" fn(*mut c_void, c_int) -> *const c_char,
    pub whisper_full_lang_id: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_lang_str: unsafe extern "C" fn(c_int) -> *const c_char,
    pub whisper_lang_id: unsafe extern "C" fn(*const c_char) -> c_int,
    pub whisper_model_n_vocab: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_model_n_audio_state: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_model_n_audio_head: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_model_n_audio_layer: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_model_n_text_state: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_model_n_text_head: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_model_n_text_layer: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub whisper_model_n_mels: unsafe extern "C" fn(*mut c_void) -> c_int,

    // shim functions
    pub shim_default_params: unsafe extern "C" fn(c_int) -> *mut c_void,
    pub shim_free_params: unsafe extern "C" fn(*mut c_void),
    pub shim_params_set_language: unsafe extern "C" fn(*mut c_void, *const c_char),
    pub shim_params_set_n_threads: unsafe extern "C" fn(*mut c_void, c_int),
    pub shim_params_set_translate: unsafe extern "C" fn(*mut c_void, bool),
    pub shim_params_set_print_special: unsafe extern "C" fn(*mut c_void, bool),
    pub shim_params_set_print_progress: unsafe extern "C" fn(*mut c_void, bool),
    pub shim_params_set_print_realtime: unsafe extern "C" fn(*mut c_void, bool),
    pub shim_params_set_print_timestamps: unsafe extern "C" fn(*mut c_void, bool),
    pub shim_params_set_single_segment: unsafe extern "C" fn(*mut c_void, bool),
    pub shim_params_set_suppress_nst: unsafe extern "C" fn(*mut c_void, bool),
    pub shim_whisper_full:
        unsafe extern "C" fn(*mut c_void, *const c_void, *const c_float, c_int) -> c_int,
}

use crate::ffi;

macro_rules! load_fn {
    ($lib:expr, $name:ident) => {{
        let sym: libloading::Symbol<_> = $lib
            .get(concat!(stringify!($name), "\0").as_bytes())
            .map_err(|e| format!("dlsym {}: {e}", stringify!($name)))?;
        *sym
    }};
}

impl WhisperFunctions {
    /// Load whisper functions from shared libraries.
    ///
    /// `lib_dir` should contain libwhisper.so, libggml*.so, and libwhisper_shim.so.
    pub fn load(lib_dir: &Path) -> Result<Self, String> {
        // Load order matters: ggml-base first, then ggml, ggml-cpu, whisper, shim.
        let _ggml_base = unsafe {
            libloading::Library::new(lib_dir.join("libggml-base.so"))
        }
        .map_err(|e| format!("dlopen libggml-base.so: {e}"))?;

        let _ggml = unsafe {
            libloading::Library::new(lib_dir.join("libggml.so"))
        }
        .map_err(|e| format!("dlopen libggml.so: {e}"))?;

        let _ggml_cpu = unsafe {
            libloading::Library::new(lib_dir.join("libggml-cpu.so"))
        }
        .map_err(|e| format!("dlopen libggml-cpu.so: {e}"))?;

        let whisper_lib = unsafe {
            libloading::Library::new(lib_dir.join("libwhisper.so"))
        }
        .map_err(|e| format!("dlopen libwhisper.so: {e}"))?;

        let shim_lib = unsafe {
            libloading::Library::new(lib_dir.join("libwhisper_shim.so"))
        }
        .map_err(|e| format!("dlopen libwhisper_shim.so: {e}"))?;

        unsafe {
            Ok(Self {
                whisper_init_from_file_with_params: load_fn!(whisper_lib, whisper_init_from_file_with_params),
                whisper_free: load_fn!(whisper_lib, whisper_free),
                whisper_context_default_params: load_fn!(whisper_lib, whisper_context_default_params),
                whisper_full_n_segments: load_fn!(whisper_lib, whisper_full_n_segments),
                whisper_full_get_segment_text: load_fn!(whisper_lib, whisper_full_get_segment_text),
                whisper_full_lang_id: load_fn!(whisper_lib, whisper_full_lang_id),
                whisper_lang_str: load_fn!(whisper_lib, whisper_lang_str),
                whisper_lang_id: load_fn!(whisper_lib, whisper_lang_id),
                whisper_model_n_vocab: load_fn!(whisper_lib, whisper_model_n_vocab),
                whisper_model_n_audio_state: load_fn!(whisper_lib, whisper_model_n_audio_state),
                whisper_model_n_audio_head: load_fn!(whisper_lib, whisper_model_n_audio_head),
                whisper_model_n_audio_layer: load_fn!(whisper_lib, whisper_model_n_audio_layer),
                whisper_model_n_text_state: load_fn!(whisper_lib, whisper_model_n_text_state),
                whisper_model_n_text_head: load_fn!(whisper_lib, whisper_model_n_text_head),
                whisper_model_n_text_layer: load_fn!(whisper_lib, whisper_model_n_text_layer),
                whisper_model_n_mels: load_fn!(whisper_lib, whisper_model_n_mels),

                shim_default_params: load_fn!(shim_lib, shim_default_params),
                shim_free_params: load_fn!(shim_lib, shim_free_params),
                shim_params_set_language: load_fn!(shim_lib, shim_params_set_language),
                shim_params_set_n_threads: load_fn!(shim_lib, shim_params_set_n_threads),
                shim_params_set_translate: load_fn!(shim_lib, shim_params_set_translate),
                shim_params_set_print_special: load_fn!(shim_lib, shim_params_set_print_special),
                shim_params_set_print_progress: load_fn!(shim_lib, shim_params_set_print_progress),
                shim_params_set_print_realtime: load_fn!(shim_lib, shim_params_set_print_realtime),
                shim_params_set_print_timestamps: load_fn!(shim_lib, shim_params_set_print_timestamps),
                shim_params_set_single_segment: load_fn!(shim_lib, shim_params_set_single_segment),
                shim_params_set_suppress_nst: load_fn!(shim_lib, shim_params_set_suppress_nst),
                shim_whisper_full: load_fn!(shim_lib, shim_whisper_full),

                // Keep libs alive
                _shim_lib: shim_lib,
                _whisper_lib: whisper_lib,
            })
        }
    }
}

// Keep ggml libs alive as leaked (they need to stay loaded for whisper to work).
// This is intentional — they're process-lifetime resources.
impl WhisperFunctions {
    fn _keep_ggml_alive() {
        // The ggml libs are loaded in load() but not stored.
        // They stay loaded because libloading doesn't unload on drop
        // when other libs depend on them.
    }
}
