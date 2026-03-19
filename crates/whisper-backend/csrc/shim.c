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

/* Run whisper_full using a pointer to params (dereferences for the by-value C API). */
int shim_whisper_full(struct whisper_context *ctx,
                      const struct whisper_full_params *params,
                      const float *samples,
                      int n_samples) {
    return whisper_full(ctx, *params, samples, n_samples);
}
