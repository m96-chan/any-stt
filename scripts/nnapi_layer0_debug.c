/**
 * Full Layer 0 CPU reference vs NNAPI comparison.
 * Reads conv_output.bin (dumped from callback) and model weights,
 * computes full encoder layer 0 on CPU, and builds equivalent NNAPI
 * graphs to compare at each step.
 *
 * Usage: ./nnapi_layer0_debug ggml-tiny.en.bin
 *        (requires conv_output.bin in current directory)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "whisper.h"

// CPU helper functions
static void cpu_layernorm(const float* x, const float* w, const float* b, float* out, int M, int N) {
    for (int i = 0; i < M; i++) {
        float mean = 0; for (int j = 0; j < N; j++) mean += x[i*N+j]; mean /= N;
        float var = 0; for (int j = 0; j < N; j++) { float d = x[i*N+j]-mean; var += d*d; } var /= N;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < N; j++) out[i*N+j] = w[j] * (x[i*N+j] - mean) * inv + b[j];
    }
}

static void cpu_fc(const float* x, const float* w, const float* bias, float* out, int M, int K, int N) {
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        float s = bias[j]; for (int k = 0; k < K; k++) s += x[i*K+k] * w[j*K+k]; out[i*N+j] = s;
    }
}

static void cpu_gelu(const float* x, float* out, int n) {
    for (int i = 0; i < n; i++) {
        float x3 = x[i]*x[i]*x[i];
        out[i] = 0.5f*x[i]*(1.0f + tanhf(0.7978846f*x[i]*(1.0f + 0.044715f*x[i]*x[i])));
    }
}

static void cpu_add(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static void cpu_multihead_attn(const float* q, const float* k, const float* v, float* out,
                                int nc, int ns, int n_head) {
    int hd = ns / n_head;
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < n_head; h++) {
        int off = h * hd;
        // QK^T per head
        float* qk = calloc(nc*nc, sizeof(float));
        for (int i = 0; i < nc; i++) for (int j = 0; j < nc; j++) {
            float s = 0;
            for (int d = 0; d < hd; d++) s += q[i*ns+off+d] * k[j*ns+off+d];
            qk[i*nc+j] = s * scale;
        }
        // softmax
        for (int i = 0; i < nc; i++) {
            float max_v = -1e30f; for (int j = 0; j < nc; j++) if(qk[i*nc+j]>max_v) max_v=qk[i*nc+j];
            float sum = 0; for (int j = 0; j < nc; j++) { qk[i*nc+j]=expf(qk[i*nc+j]-max_v); sum+=qk[i*nc+j]; }
            for (int j = 0; j < nc; j++) qk[i*nc+j] /= sum;
        }
        // attn@V
        for (int i = 0; i < nc; i++) for (int d = 0; d < hd; d++) {
            float s = 0; for (int t = 0; t < nc; t++) s += qk[i*nc+t] * v[t*ns+off+d];
            out[i*ns+off+d] = s;
        }
        free(qk);
    }
}

static float max_diff(const float* a, const float* b, int n) {
    float m = 0; for (int i = 0; i < n; i++) { float e = fabsf(a[i]-b[i]); if(e>m)m=e; } return m;
}

static float* get_tensor(struct whisper_context* ctx, const char* name, int* n) {
    *n = whisper_get_model_tensor_f32(ctx, name, NULL, 0);
    if (*n <= 0) { fprintf(stderr, "tensor not found: %s\n", name); return NULL; }
    float* buf = malloc(*n * sizeof(float));
    whisper_get_model_tensor_f32(ctx, name, buf, *n);
    return buf;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "ggml-tiny.en.bin";

    // Load model
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false; cp.flash_attn = false;
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cp);
    int ns = whisper_model_n_audio_state(ctx);
    int nc = whisper_model_n_audio_ctx(ctx);
    int nh = whisper_model_n_audio_head(ctx);
    int hd = ns / nh;
    fprintf(stderr, "n_state=%d n_ctx=%d n_head=%d head_dim=%d\n", ns, nc, nh, hd);

    // Read conv output
    FILE* f = fopen("conv_output.bin", "rb");
    if (!f) { fprintf(stderr, "conv_output.bin not found — run bench --backend e2e first\n"); return 1; }
    float* conv = malloc(nc * ns * sizeof(float));
    fread(conv, sizeof(float), nc * ns, f); fclose(f);
    fprintf(stderr, "conv[0..4]: %.4f %.4f %.4f %.4f\n", conv[0], conv[1], conv[2], conv[3]);

    // Get all Layer 0 weights
    int n;
    float *ln1_w = get_tensor(ctx, "encoder.blocks.0.attn_ln.weight", &n);
    float *ln1_b = get_tensor(ctx, "encoder.blocks.0.attn_ln.bias", &n);
    float *q_w = get_tensor(ctx, "encoder.blocks.0.attn.query.weight", &n);
    float *q_b = get_tensor(ctx, "encoder.blocks.0.attn.query.bias", &n);
    float *k_w = get_tensor(ctx, "encoder.blocks.0.attn.key.weight", &n);
    float *v_w = get_tensor(ctx, "encoder.blocks.0.attn.value.weight", &n);
    float *v_b = get_tensor(ctx, "encoder.blocks.0.attn.value.bias", &n);
    float *out_w = get_tensor(ctx, "encoder.blocks.0.attn.out.weight", &n);
    float *out_b = get_tensor(ctx, "encoder.blocks.0.attn.out.bias", &n);
    float *ln2_w = get_tensor(ctx, "encoder.blocks.0.mlp_ln.weight", &n);
    float *ln2_b = get_tensor(ctx, "encoder.blocks.0.mlp_ln.bias", &n);
    float *fc1_w = get_tensor(ctx, "encoder.blocks.0.mlp.0.weight", &n);
    float *fc1_b = get_tensor(ctx, "encoder.blocks.0.mlp.0.bias", &n);
    float *fc2_w = get_tensor(ctx, "encoder.blocks.0.mlp.2.weight", &n);
    float *fc2_b = get_tensor(ctx, "encoder.blocks.0.mlp.2.bias", &n);

    int S = nc * ns;
    int nff = 4 * ns;

    // Step 1: LayerNorm
    float* ln1_out = malloc(S * sizeof(float));
    cpu_layernorm(conv, ln1_w, ln1_b, ln1_out, nc, ns);
    fprintf(stderr, "\n=== Step 1: LayerNorm ===\n");
    fprintf(stderr, "LN1[0..4]: %.4f %.4f %.4f %.4f\n", ln1_out[0], ln1_out[1], ln1_out[2], ln1_out[3]);

    // Step 2: Q, K, V
    float* q = malloc(S * sizeof(float));
    float* k = malloc(S * sizeof(float));
    float* v = malloc(S * sizeof(float));
    cpu_fc(ln1_out, q_w, q_b, q, nc, ns, ns);
    float zero_bias[4096]; memset(zero_bias, 0, sizeof(zero_bias));
    cpu_fc(ln1_out, k_w, zero_bias, k, nc, ns, ns);
    cpu_fc(ln1_out, v_w, v_b, v, nc, ns, ns);
    fprintf(stderr, "\n=== Step 2: Q, K, V ===\n");
    fprintf(stderr, "Q[0..4]: %.4f %.4f %.4f %.4f\n", q[0], q[1], q[2], q[3]);
    fprintf(stderr, "K[0..4]: %.4f %.4f %.4f %.4f\n", k[0], k[1], k[2], k[3]);
    fprintf(stderr, "V[0..4]: %.4f %.4f %.4f %.4f\n", v[0], v[1], v[2], v[3]);

    // Step 3: Multi-head attention
    float* attn = calloc(S, sizeof(float));
    cpu_multihead_attn(q, k, v, attn, nc, ns, nh);
    fprintf(stderr, "\n=== Step 3: Multi-head Attention ===\n");
    fprintf(stderr, "attn[0..4]: %.4f %.4f %.4f %.4f\n", attn[0], attn[1], attn[2], attn[3]);

    // Step 4: Output projection
    float* proj = malloc(S * sizeof(float));
    cpu_fc(attn, out_w, out_b, proj, nc, ns, ns);
    fprintf(stderr, "\n=== Step 4: Output Projection ===\n");
    fprintf(stderr, "proj[0..4]: %.4f %.4f %.4f %.4f\n", proj[0], proj[1], proj[2], proj[3]);

    // Step 5: Residual 1 = conv + proj
    float* res1 = malloc(S * sizeof(float));
    cpu_add(conv, proj, res1, S);
    fprintf(stderr, "\n=== Step 5: Residual 1 (input + proj) ===\n");
    fprintf(stderr, "res1[0..4]: %.4f %.4f %.4f %.4f\n", res1[0], res1[1], res1[2], res1[3]);

    // Step 6: LayerNorm 2
    float* ln2_out = malloc(S * sizeof(float));
    cpu_layernorm(res1, ln2_w, ln2_b, ln2_out, nc, ns);
    fprintf(stderr, "\n=== Step 6: LayerNorm 2 ===\n");
    fprintf(stderr, "LN2[0..4]: %.4f %.4f %.4f %.4f\n", ln2_out[0], ln2_out[1], ln2_out[2], ln2_out[3]);

    // Step 7: FC1 + GELU
    float* fc1_out = malloc(nc * nff * sizeof(float));
    cpu_fc(ln2_out, fc1_w, fc1_b, fc1_out, nc, ns, nff);
    float* gelu_out = malloc(nc * nff * sizeof(float));
    cpu_gelu(fc1_out, gelu_out, nc * nff);
    fprintf(stderr, "\n=== Step 7: FC1 + GELU ===\n");
    fprintf(stderr, "FC1[0..4]: %.4f %.4f %.4f %.4f\n", fc1_out[0], fc1_out[1], fc1_out[2], fc1_out[3]);
    fprintf(stderr, "GELU[0..4]: %.4f %.4f %.4f %.4f\n", gelu_out[0], gelu_out[1], gelu_out[2], gelu_out[3]);

    // Step 8: FC2
    float* fc2_out = malloc(S * sizeof(float));
    cpu_fc(gelu_out, fc2_w, fc2_b, fc2_out, nc, nff, ns);
    fprintf(stderr, "\n=== Step 8: FC2 ===\n");
    fprintf(stderr, "FC2[0..4]: %.4f %.4f %.4f %.4f\n", fc2_out[0], fc2_out[1], fc2_out[2], fc2_out[3]);

    // Step 9: Residual 2 = res1 + fc2
    float* output = malloc(S * sizeof(float));
    cpu_add(res1, fc2_out, output, S);
    fprintf(stderr, "\n=== Step 9: Layer 0 output ===\n");
    fprintf(stderr, "output[0..4]: %.4f %.4f %.4f %.4f\n", output[0], output[1], output[2], output[3]);

    // Write output for comparison
    FILE* fout = fopen("layer0_cpu_output.bin", "wb");
    if (fout) { fwrite(output, sizeof(float), S, fout); fclose(fout); }

    // Compare with NNAPI layer 0 output if available
    FILE* fl0 = fopen("layer0_output.bin", "rb");
    if (fl0) {
        float* nnapi_l0 = malloc(S * sizeof(float));
        fread(nnapi_l0, sizeof(float), S, fl0); fclose(fl0);
        float me = max_diff(output, nnapi_l0, S);
        fprintf(stderr, "\n=== NNAPI Layer 0 comparison ===\n");
        fprintf(stderr, "NNAPI L0[0..4]: %.4f %.4f %.4f %.4f\n", nnapi_l0[0], nnapi_l0[1], nnapi_l0[2], nnapi_l0[3]);
        fprintf(stderr, "CPU   L0[0..4]: %.4f %.4f %.4f %.4f\n", output[0], output[1], output[2], output[3]);
        fprintf(stderr, "L0 max_err: %.6f %s\n", me, me < 0.01 ? "OK" : "FAIL");

        // If L0 matches, compute Layer 1 on NNAPI L0 output and compare with NNAPI L1
        if (me < 0.1) {
            float *l1_ln1_w = get_tensor(ctx, "encoder.blocks.1.attn_ln.weight", &n);
            float *l1_ln1_b = get_tensor(ctx, "encoder.blocks.1.attn_ln.bias", &n);
            float *l1_q_w = get_tensor(ctx, "encoder.blocks.1.attn.query.weight", &n);
            float *l1_q_b = get_tensor(ctx, "encoder.blocks.1.attn.query.bias", &n);
            float *l1_k_w = get_tensor(ctx, "encoder.blocks.1.attn.key.weight", &n);
            float *l1_v_w = get_tensor(ctx, "encoder.blocks.1.attn.value.weight", &n);
            float *l1_v_b = get_tensor(ctx, "encoder.blocks.1.attn.value.bias", &n);
            float *l1_out_w = get_tensor(ctx, "encoder.blocks.1.attn.out.weight", &n);
            float *l1_out_b = get_tensor(ctx, "encoder.blocks.1.attn.out.bias", &n);
            float *l1_ln2_w = get_tensor(ctx, "encoder.blocks.1.mlp_ln.weight", &n);
            float *l1_ln2_b = get_tensor(ctx, "encoder.blocks.1.mlp_ln.bias", &n);
            float *l1_fc1_w = get_tensor(ctx, "encoder.blocks.1.mlp.0.weight", &n);
            float *l1_fc1_b = get_tensor(ctx, "encoder.blocks.1.mlp.0.bias", &n);
            float *l1_fc2_w = get_tensor(ctx, "encoder.blocks.1.mlp.2.weight", &n);
            float *l1_fc2_b = get_tensor(ctx, "encoder.blocks.1.mlp.2.bias", &n);

            // Use NNAPI L0 output as L1 input
            float* l1_input = nnapi_l0;
            float* l1_ln1 = malloc(S*sizeof(float));
            cpu_layernorm(l1_input, l1_ln1_w, l1_ln1_b, l1_ln1, nc, ns);
            float* l1_q = malloc(S*sizeof(float));
            float* l1_k = malloc(S*sizeof(float));
            float* l1_v = malloc(S*sizeof(float));
            cpu_fc(l1_ln1, l1_q_w, l1_q_b, l1_q, nc, ns, ns);
            cpu_fc(l1_ln1, l1_k_w, zero_bias, l1_k, nc, ns, ns);
            cpu_fc(l1_ln1, l1_v_w, l1_v_b, l1_v, nc, ns, ns);
            float* l1_attn = calloc(S, sizeof(float));
            cpu_multihead_attn(l1_q, l1_k, l1_v, l1_attn, nc, ns, nh);
            float* l1_proj = malloc(S*sizeof(float));
            cpu_fc(l1_attn, l1_out_w, l1_out_b, l1_proj, nc, ns, ns);
            float* l1_res1 = malloc(S*sizeof(float));
            cpu_add(l1_input, l1_proj, l1_res1, S);
            float* l1_ln2 = malloc(S*sizeof(float));
            cpu_layernorm(l1_res1, l1_ln2_w, l1_ln2_b, l1_ln2, nc, ns);
            float* l1_fc1 = malloc(nc*nff*sizeof(float));
            cpu_fc(l1_ln2, l1_fc1_w, l1_fc1_b, l1_fc1, nc, ns, nff);
            float* l1_gelu = malloc(nc*nff*sizeof(float));
            cpu_gelu(l1_fc1, l1_gelu, nc*nff);
            float* l1_fc2 = malloc(S*sizeof(float));
            cpu_fc(l1_gelu, l1_fc2_w, l1_fc2_b, l1_fc2, nc, nff, ns);
            float* l1_output = malloc(S*sizeof(float));
            cpu_add(l1_res1, l1_fc2, l1_output, S);

            fprintf(stderr, "\n=== CPU Layer 1 (from NNAPI L0 output) ===\n");
            fprintf(stderr, "L1 out[0..4]: %.4f %.4f %.4f %.4f\n", l1_output[0], l1_output[1], l1_output[2], l1_output[3]);

            FILE* fl1 = fopen("layer1_output.bin", "rb");
            if (fl1) {
                float* nnapi_l1 = malloc(S*sizeof(float));
                fread(nnapi_l1, sizeof(float), S, fl1); fclose(fl1);
                float me1 = max_diff(l1_output, nnapi_l1, S);
                fprintf(stderr, "NNAPI L1[0..4]: %.4f %.4f %.4f %.4f\n", nnapi_l1[0], nnapi_l1[1], nnapi_l1[2], nnapi_l1[3]);
                fprintf(stderr, "L1 max_err: %.6f %s\n", me1, me1 < 0.01 ? "OK" : "FAIL");
                free(nnapi_l1);
            }
            free(l1_ln1); free(l1_q); free(l1_k); free(l1_v); free(l1_attn);
            free(l1_proj); free(l1_res1); free(l1_ln2); free(l1_fc1);
            free(l1_gelu); free(l1_fc2); free(l1_output);
            free(l1_ln1_w); free(l1_ln1_b); free(l1_q_w); free(l1_q_b);
            free(l1_k_w); free(l1_v_w); free(l1_v_b); free(l1_out_w);
            free(l1_out_b); free(l1_ln2_w); free(l1_ln2_b); free(l1_fc1_w);
            free(l1_fc1_b); free(l1_fc2_w); free(l1_fc2_b);
        }
        free(nnapi_l0);
    }

    // Cleanup
    free(conv); free(ln1_out); free(q); free(k); free(v); free(attn);
    free(proj); free(res1); free(ln2_out); free(fc1_out); free(gelu_out);
    free(fc2_out); free(output);
    free(ln1_w); free(ln1_b); free(q_w); free(q_b); free(k_w); free(v_w);
    free(v_b); free(out_w); free(out_b); free(ln2_w); free(ln2_b);
    free(fc1_w); free(fc1_b); free(fc2_w); free(fc2_b);
    whisper_free(ctx);
    return 0;
}
