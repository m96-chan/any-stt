/**
 * Full 4-layer CPU reference encoder — compare each layer with NNAPI dumps.
 * Reads conv_output.bin + layer{0-3}_output.bin (dumped from bench --backend e2e).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "whisper.h"

static void cpu_ln(const float* x, const float* w, const float* b, float* o, int M, int N) {
    for (int i = 0; i < M; i++) {
        float m = 0; for (int j = 0; j < N; j++) m += x[i*N+j]; m /= N;
        float v = 0; for (int j = 0; j < N; j++) { float d=x[i*N+j]-m; v+=d*d; } v /= N;
        float s = 1.0f/sqrtf(v+1e-5f);
        for (int j = 0; j < N; j++) o[i*N+j] = w[j]*(x[i*N+j]-m)*s + b[j];
    }
}
static void cpu_fc(const float* x, const float* w, const float* b, float* o, int M, int K, int N) {
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        float s = b ? b[j] : 0; for (int k = 0; k < K; k++) s += x[i*K+k]*w[j*K+k]; o[i*N+j] = s;
    }
}
static void cpu_gelu(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = 0.5f*x[i]*(1.0f+tanhf(0.7978846f*x[i]*(1.0f+0.044715f*x[i]*x[i])));
}
static void cpu_add(const float* a, const float* b, float* o, int n) { for (int i=0;i<n;i++) o[i]=a[i]+b[i]; }
static void cpu_mha(const float* q, const float* k, const float* v, float* o, int nc, int ns, int nh) {
    int hd = ns/nh;
    float scale = 1.0f/sqrtf((float)hd);
    memset(o, 0, nc*ns*sizeof(float));
    for (int h = 0; h < nh; h++) {
        int off = h*hd;
        float* qk = calloc(nc*nc, sizeof(float));
        for (int i=0;i<nc;i++) for (int j=0;j<nc;j++) {
            float s=0; for(int d=0;d<hd;d++) s+=q[i*ns+off+d]*k[j*ns+off+d]; qk[i*nc+j]=s*scale;
        }
        for (int i=0;i<nc;i++) {
            float mx=-1e30f; for(int j=0;j<nc;j++) if(qk[i*nc+j]>mx) mx=qk[i*nc+j];
            float sm=0; for(int j=0;j<nc;j++){qk[i*nc+j]=expf(qk[i*nc+j]-mx);sm+=qk[i*nc+j];}
            for(int j=0;j<nc;j++) qk[i*nc+j]/=sm;
        }
        for (int i=0;i<nc;i++) for(int d=0;d<hd;d++) {
            float s=0; for(int t=0;t<nc;t++) s+=qk[i*nc+t]*v[t*ns+off+d]; o[i*ns+off+d]=s;
        }
        free(qk);
    }
}

static float* get_t(struct whisper_context* ctx, const char* name) {
    int n = whisper_get_model_tensor_f32(ctx, name, NULL, 0);
    if (n <= 0) { fprintf(stderr, "MISSING: %s\n", name); return calloc(1, sizeof(float)); }
    float* buf = malloc(n*sizeof(float));
    whisper_get_model_tensor_f32(ctx, name, buf, n);
    return buf;
}

static float maxdiff(const float* a, const float* b, int n) {
    float m=0; for(int i=0;i<n;i++){float e=fabsf(a[i]-b[i]);if(e>m)m=e;} return m;
}

static float* read_bin(const char* path, int expected_floats) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    float* buf = malloc(expected_floats * sizeof(float));
    fread(buf, sizeof(float), expected_floats, f);
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "ggml-tiny.en.bin";
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false; cp.flash_attn = false;
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cp);
    int ns = whisper_model_n_audio_state(ctx);
    int nc = whisper_model_n_audio_ctx(ctx);
    int nh = whisper_model_n_audio_head(ctx);
    int nl = whisper_model_n_audio_layer(ctx);
    int nff = 4*ns;
    int S = nc*ns;
    fprintf(stderr, "ns=%d nc=%d nh=%d nl=%d\n", ns, nc, nh, nl);

    float* conv = read_bin("conv_output.bin", S);
    if (!conv) { fprintf(stderr, "conv_output.bin not found\n"); return 1; }
    fprintf(stderr, "conv[0..4]: %.4f %.4f %.4f %.4f\n", conv[0], conv[1], conv[2], conv[3]);

    float* current = malloc(S*sizeof(float));
    memcpy(current, conv, S*sizeof(float));

    for (int l = 0; l < nl; l++) {
        char name[256];
        #define T(fmt, ...) (snprintf(name,256,fmt,##__VA_ARGS__), get_t(ctx, name))

        float *ln1w = T("encoder.blocks.%d.attn_ln.weight", l);
        float *ln1b = T("encoder.blocks.%d.attn_ln.bias", l);
        float *qw = T("encoder.blocks.%d.attn.query.weight", l);
        float *qb = T("encoder.blocks.%d.attn.query.bias", l);
        float *kw = T("encoder.blocks.%d.attn.key.weight", l);
        float *vw = T("encoder.blocks.%d.attn.value.weight", l);
        float *vb = T("encoder.blocks.%d.attn.value.bias", l);
        float *ow = T("encoder.blocks.%d.attn.out.weight", l);
        float *ob = T("encoder.blocks.%d.attn.out.bias", l);
        float *ln2w = T("encoder.blocks.%d.mlp_ln.weight", l);
        float *ln2b = T("encoder.blocks.%d.mlp_ln.bias", l);
        float *f1w = T("encoder.blocks.%d.mlp.0.weight", l);
        float *f1b = T("encoder.blocks.%d.mlp.0.bias", l);
        float *f2w = T("encoder.blocks.%d.mlp.2.weight", l);
        float *f2b = T("encoder.blocks.%d.mlp.2.bias", l);

        float *ln1 = malloc(S*sizeof(float));
        cpu_ln(current, ln1w, ln1b, ln1, nc, ns);

        float *q = malloc(S*sizeof(float));
        float *k = malloc(S*sizeof(float));
        float *v = malloc(S*sizeof(float));
        cpu_fc(ln1, qw, qb, q, nc, ns, ns);
        cpu_fc(ln1, kw, NULL, k, nc, ns, ns);
        cpu_fc(ln1, vw, vb, v, nc, ns, ns);

        float *attn = calloc(S, sizeof(float));
        cpu_mha(q, k, v, attn, nc, ns, nh);

        float *proj = malloc(S*sizeof(float));
        cpu_fc(attn, ow, ob, proj, nc, ns, ns);

        float *res1 = malloc(S*sizeof(float));
        cpu_add(current, proj, res1, S);

        float *ln2 = malloc(S*sizeof(float));
        cpu_ln(res1, ln2w, ln2b, ln2, nc, ns);

        float *fc1 = malloc(nc*nff*sizeof(float));
        cpu_fc(ln2, f1w, f1b, fc1, nc, ns, nff);
        cpu_gelu(fc1, nc*nff);

        float *fc2 = malloc(S*sizeof(float));
        cpu_fc(fc1, f2w, f2b, fc2, nc, nff, ns);

        float *output = malloc(S*sizeof(float));
        cpu_add(res1, fc2, output, S);

        // Post-LayerNorm on last layer
        if (l == nl - 1) {
            float *lnpw = get_t(ctx, "encoder.ln_post.weight");
            float *lnpb = get_t(ctx, "encoder.ln_post.bias");
            float *ln_post = malloc(S*sizeof(float));
            cpu_ln(output, lnpw, lnpb, ln_post, nc, ns);
            memcpy(output, ln_post, S*sizeof(float));
            free(ln_post); free(lnpw); free(lnpb);
        }

        fprintf(stderr, "\n=== Layer %d ===\n", l);
        fprintf(stderr, "CPU[0..4]: %.4f %.4f %.4f %.4f\n", output[0], output[1], output[2], output[3]);

        char fn[64]; snprintf(fn, 64, "layer%d_output.bin", l);
        float* nnapi = read_bin(fn, S);
        if (nnapi) {
            float me = maxdiff(output, nnapi, S);
            fprintf(stderr, "NPU[0..4]: %.4f %.4f %.4f %.4f\n", nnapi[0], nnapi[1], nnapi[2], nnapi[3]);
            fprintf(stderr, "max_err: %.6f %s\n", me, me < 0.01 ? "OK" : me < 0.1 ? "WARN" : "FAIL");
            free(nnapi);
        } else {
            fprintf(stderr, "%s not found\n", fn);
        }

        memcpy(current, output, S*sizeof(float));
        free(ln1); free(q); free(k); free(v); free(attn); free(proj);
        free(res1); free(ln2); free(fc1); free(fc2); free(output);
        free(ln1w); free(ln1b); free(qw); free(qb); free(kw);
        free(vw); free(vb); free(ow); free(ob); free(ln2w); free(ln2b);
        free(f1w); free(f1b); free(f2w); free(f2b);
    }

    free(conv); free(current);
    whisper_free(ctx);
    return 0;
}
