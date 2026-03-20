/**
 * C shim for whisper.cpp FFI.
 *
 * whisper_full_params is a large struct that changes across versions.
 * Rather than replicating its layout in Rust, we provide helper functions
 * that create default params, set the fields we need, and call whisper_full.
 */

#include "whisper.h"
#include <stdlib.h>
#include <string.h>

/* Allocate default params on the heap. Caller must free with shim_free_params. */
struct whisper_full_params *shim_default_params(enum whisper_sampling_strategy strategy) {
    struct whisper_full_params *p = malloc(sizeof(struct whisper_full_params));
    if (!p) return NULL;
    *p = whisper_full_default_params(strategy);
    return p;
}

void shim_free_params(struct whisper_full_params *p) {
    free(p);
}

/* Field setters */

void shim_params_set_language(struct whisper_full_params *p, const char *lang) {
    p->language = lang;
}

void shim_params_set_n_threads(struct whisper_full_params *p, int n) {
    p->n_threads = n;
}

void shim_params_set_translate(struct whisper_full_params *p, bool translate) {
    p->translate = translate;
}

void shim_params_set_no_timestamps(struct whisper_full_params *p, bool val) {
    p->no_timestamps = val;
}

void shim_params_set_single_segment(struct whisper_full_params *p, bool val) {
    p->single_segment = val;
}

void shim_params_set_print_special(struct whisper_full_params *p, bool val) {
    p->print_special = val;
}

void shim_params_set_print_progress(struct whisper_full_params *p, bool val) {
    p->print_progress = val;
}

void shim_params_set_print_realtime(struct whisper_full_params *p, bool val) {
    p->print_realtime = val;
}

void shim_params_set_print_timestamps(struct whisper_full_params *p, bool val) {
    p->print_timestamps = val;
}

void shim_params_set_suppress_nst(struct whisper_full_params *p, bool val) {
    p->suppress_nst = val;
}

/* Skip the encoder_begin_callback, causing whisper_full to skip encoding.
 * This callback is set by shim_params_set_skip_encoder. */
static bool shim_encoder_skip_callback(struct whisper_context *ctx,
                                        struct whisper_state *state,
                                        void *user_data) {
    (void)ctx; (void)state; (void)user_data;
    /* Returning false tells whisper_full to abort this seek iteration.
     * However, that's NOT what we want — we want to skip encoding but
     * still run decoding. For now, this approach won't work for whisper_full.
     *
     * Instead, we use the low-level whisper_encode + whisper_decode API
     * in the Rust decoder. This callback is kept as a no-op placeholder. */
    return true;
}

/* Set params to skip encoding (used when encoder output is externally provided). */
void shim_params_set_skip_encoder(struct whisper_full_params *p, bool skip) {
    if (skip) {
        p->encoder_begin_callback = shim_encoder_skip_callback;
        p->encoder_begin_callback_user_data = NULL;
    } else {
        p->encoder_begin_callback = NULL;
        p->encoder_begin_callback_user_data = NULL;
    }
}

/* Run whisper_full using a pointer to params (dereferences for the by-value C API). */
int shim_whisper_full(struct whisper_context *ctx,
                      const struct whisper_full_params *params,
                      const float *samples,
                      int n_samples) {
    return whisper_full(ctx, *params, samples, n_samples);
}

/* Get the number of audio context frames (1500 for standard whisper). */
int shim_model_n_audio_ctx(struct whisper_context *ctx) {
    return whisper_model_n_audio_ctx(ctx);
}

/* Get the encoder state dimension (n_state). */
int shim_model_n_audio_state(struct whisper_context *ctx) {
    return whisper_model_n_audio_state(ctx);
}
