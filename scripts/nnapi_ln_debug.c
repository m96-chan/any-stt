/**
 * Debug: Test NNAPI LayerNorm decomposition accuracy.
 * LayerNorm(x, γ, β, ε) = γ * (x - mean) / sqrt(var + ε) + β
 * Decomposed: MEAN → SUB → MUL(sq) → MEAN → ADD(ε) → RSQRT → MUL → MUL(γ) → ADD(β)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

typedef void ANeuralNetworksModel;
typedef void ANeuralNetworksCompilation;
typedef void ANeuralNetworksExecution;
typedef void ANeuralNetworksDevice;

typedef struct {
    int32_t type; uint32_t dimensionCount; const uint32_t* dimensions;
    float scale; int32_t zeroPoint;
} ANeuralNetworksOperandType;

#define T_F32 0
#define T_I32 1
#define T_TENSOR_F32 3
#define T_TENSOR_I32 4
#define OP_ADD 0
#define OP_MUL 18
#define OP_SUB 36
#define OP_MEAN 31
#define OP_RSQRT 83
#define FUSED_NONE 0

typedef int (*FnModelCreate)(ANeuralNetworksModel**);
typedef void (*FnModelFree)(ANeuralNetworksModel*);
typedef int (*FnModelAddOperand)(ANeuralNetworksModel*, const ANeuralNetworksOperandType*);
typedef int (*FnModelSetOperandValue)(ANeuralNetworksModel*, int32_t, const void*, size_t);
typedef int (*FnModelAddOperation)(ANeuralNetworksModel*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
typedef int (*FnModelIdentifyIO)(ANeuralNetworksModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
typedef int (*FnModelFinish)(ANeuralNetworksModel*);
typedef int (*FnCompCreate)(const ANeuralNetworksModel*, const ANeuralNetworksDevice* const*, uint32_t, ANeuralNetworksCompilation**);
typedef int (*FnCompSetPref)(ANeuralNetworksCompilation*, int32_t);
typedef int (*FnCompFinish)(ANeuralNetworksCompilation*);
typedef void (*FnCompFree)(ANeuralNetworksCompilation*);
typedef int (*FnExecCreate)(const ANeuralNetworksCompilation*, ANeuralNetworksExecution**);
typedef int (*FnExecSetInput)(ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*, const void*, size_t);
typedef int (*FnExecSetOutput)(ANeuralNetworksExecution*, int32_t, const ANeuralNetworksOperandType*, void*, size_t);
typedef int (*FnExecCompute)(ANeuralNetworksExecution*);
typedef void (*FnExecFree)(ANeuralNetworksExecution*);
typedef int (*FnGetDeviceCount)(uint32_t*);
typedef int (*FnGetDevice)(uint32_t, ANeuralNetworksDevice**);
typedef int (*FnDeviceGetName)(const ANeuralNetworksDevice*, const char**);

static FnModelCreate model_create;
static FnModelFree model_free;
static FnModelAddOperand model_add_operand;
static FnModelSetOperandValue model_set_value;
static FnModelAddOperation model_add_op;
static FnModelIdentifyIO model_identify_io;
static FnModelFinish model_finish;
static FnCompCreate comp_create;
static FnCompSetPref comp_set_pref;
static FnCompFinish comp_finish;
static FnCompFree comp_free;
static FnExecCreate exec_create;
static FnExecSetInput exec_set_input;
static FnExecSetOutput exec_set_output;
static FnExecCompute exec_compute;
static FnExecFree exec_free;
static FnGetDeviceCount get_device_count;
static FnGetDevice get_device;
static FnDeviceGetName device_get_name;

static int nop;
static int add_tf32(ANeuralNetworksModel* m, int nd, const uint32_t* d) {
    ANeuralNetworksOperandType t = {T_TENSOR_F32, nd, d, 0, 0}; int i=nop++; model_add_operand(m,&t); return i;
}
static int add_ti32(ANeuralNetworksModel* m, int nd, const uint32_t* d) {
    ANeuralNetworksOperandType t = {T_TENSOR_I32, nd, d, 0, 0}; int i=nop++; model_add_operand(m,&t); return i;
}
static int add_si32(ANeuralNetworksModel* m, int32_t val) {
    ANeuralNetworksOperandType t = {T_I32, 0, NULL, 0, 0}; int i=nop++; model_add_operand(m,&t); model_set_value(m,i,&val,4); return i;
}
static int add_sf32(ANeuralNetworksModel* m, float val) {
    ANeuralNetworksOperandType t = {T_F32, 0, NULL, 0, 0}; int i=nop++; model_add_operand(m,&t); model_set_value(m,i,&val,4); return i;
}
static int add_const_tf32(ANeuralNetworksModel* m, int nd, const uint32_t* d, const float* data, size_t nbytes) {
    int i = add_tf32(m, nd, d); model_set_value(m, i, data, nbytes); return i;
}

// CPU reference LayerNorm
static void cpu_layernorm(const float* x, const float* gamma, const float* beta,
                           float* out, int M, int N, float eps) {
    for (int i = 0; i < M; i++) {
        float mean = 0;
        for (int j = 0; j < N; j++) mean += x[i*N+j];
        mean /= N;
        float var = 0;
        for (int j = 0; j < N; j++) { float d = x[i*N+j] - mean; var += d*d; }
        var /= N;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < N; j++) {
            out[i*N+j] = gamma[j] * (x[i*N+j] - mean) * inv_std + beta[j];
        }
    }
}

int main() {
    void* lib = dlopen("libneuralnetworks.so", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen failed\n"); return 1; }
    model_create = dlsym(lib, "ANeuralNetworksModel_create");
    model_free = dlsym(lib, "ANeuralNetworksModel_free");
    model_add_operand = dlsym(lib, "ANeuralNetworksModel_addOperand");
    model_set_value = dlsym(lib, "ANeuralNetworksModel_setOperandValue");
    model_add_op = dlsym(lib, "ANeuralNetworksModel_addOperation");
    model_identify_io = dlsym(lib, "ANeuralNetworksModel_identifyInputsAndOutputs");
    model_finish = dlsym(lib, "ANeuralNetworksModel_finish");
    comp_create = dlsym(lib, "ANeuralNetworksCompilation_createForDevices");
    comp_set_pref = dlsym(lib, "ANeuralNetworksCompilation_setPreference");
    comp_finish = dlsym(lib, "ANeuralNetworksCompilation_finish");
    comp_free = dlsym(lib, "ANeuralNetworksCompilation_free");
    exec_create = dlsym(lib, "ANeuralNetworksExecution_create");
    exec_set_input = dlsym(lib, "ANeuralNetworksExecution_setInput");
    exec_set_output = dlsym(lib, "ANeuralNetworksExecution_setOutput");
    exec_compute = dlsym(lib, "ANeuralNetworksExecution_compute");
    exec_free = dlsym(lib, "ANeuralNetworksExecution_free");
    get_device_count = dlsym(lib, "ANeuralNetworks_getDeviceCount");
    get_device = dlsym(lib, "ANeuralNetworks_getDevice");
    device_get_name = dlsym(lib, "ANeuralNetworksDevice_getName");

    uint32_t nd=0; get_device_count(&nd);
    ANeuralNetworksDevice* neuron=NULL;
    for (uint32_t i=0;i<nd;i++) {
        ANeuralNetworksDevice*d=NULL; get_device(i,&d);
        const char*nm=NULL; device_get_name(d,&nm);
        if (nm && strstr(nm,"neuron")) neuron=d;
    }
    if (!neuron) { fprintf(stderr, "neuron not found\n"); return 1; }

    // Test LayerNorm: [4, 8] with gamma=[8], beta=[8]
    const int M = 4, N = 8;
    float input[32], gamma[8], beta[8], cpu_out[32], nnapi_out[32];
    for (int i=0;i<32;i++) input[i] = 0.1f * (i - 16);
    for (int i=0;i<8;i++) { gamma[i] = 1.0f + 0.1f*i; beta[i] = 0.01f * i; }

    cpu_layernorm(input, gamma, beta, cpu_out, M, N, 1e-5f);
    fprintf(stderr, "CPU LN [4,8] first row: ");
    for (int j=0;j<N;j++) fprintf(stderr, "%.4f ", cpu_out[j]);
    fprintf(stderr, "\n");

    // Build NNAPI LayerNorm graph
    nop = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d_full[] = {M, N};
    uint32_t d_red[] = {M, 1};
    uint32_t d_n[] = {N};
    uint32_t d_ax[] = {1};

    int inp = add_tf32(m, 2, d_full);  // 0: input [M,N]

    // mean(x, axis=1, keepdims=1)
    int ax1 = add_ti32(m, 1, d_ax); int32_t ax_val=1; model_set_value(m, ax1, &ax_val, 4);
    int kd1 = add_si32(m, 1);
    int mu = add_tf32(m, 2, d_red);
    model_add_op(m, OP_MEAN, 3, (uint32_t[]){inp, ax1, kd1}, 1, (uint32_t[]){mu});

    // diff = x - mu
    int f1 = add_si32(m, FUSED_NONE);
    int diff = add_tf32(m, 2, d_full);
    model_add_op(m, OP_SUB, 3, (uint32_t[]){inp, mu, f1}, 1, (uint32_t[]){diff});

    // diff_sq = diff * diff
    int f2 = add_si32(m, FUSED_NONE);
    int dsq = add_tf32(m, 2, d_full);
    model_add_op(m, OP_MUL, 3, (uint32_t[]){diff, diff, f2}, 1, (uint32_t[]){dsq});

    // var = mean(diff_sq, axis=1, keepdims=1)
    int ax2 = add_ti32(m, 1, d_ax); model_set_value(m, ax2, &ax_val, 4);
    int kd2 = add_si32(m, 1);
    int var = add_tf32(m, 2, d_red);
    model_add_op(m, OP_MEAN, 3, (uint32_t[]){dsq, ax2, kd2}, 1, (uint32_t[]){var});

    // var_eps = var + eps
    float eps_val = 1e-5f;
    int eps = add_const_tf32(m, 1, (uint32_t[]){1}, &eps_val, 4);
    int f3 = add_si32(m, FUSED_NONE);
    int var_eps = add_tf32(m, 2, d_red);
    model_add_op(m, OP_ADD, 3, (uint32_t[]){var, eps, f3}, 1, (uint32_t[]){var_eps});

    // inv_std = rsqrt(var_eps)
    int inv_std = add_tf32(m, 2, d_red);
    model_add_op(m, OP_RSQRT, 1, (uint32_t[]){var_eps}, 1, (uint32_t[]){inv_std});

    // normalized = diff * inv_std
    int f4 = add_si32(m, FUSED_NONE);
    int norm = add_tf32(m, 2, d_full);
    model_add_op(m, OP_MUL, 3, (uint32_t[]){diff, inv_std, f4}, 1, (uint32_t[]){norm});

    // scaled = norm * gamma
    int g = add_const_tf32(m, 1, d_n, gamma, N*4);
    int f5 = add_si32(m, FUSED_NONE);
    int scaled = add_tf32(m, 2, d_full);
    model_add_op(m, OP_MUL, 3, (uint32_t[]){norm, g, f5}, 1, (uint32_t[]){scaled});

    // output = scaled + beta
    int b = add_const_tf32(m, 1, d_n, beta, N*4);
    int f6 = add_si32(m, FUSED_NONE);
    int out = add_tf32(m, 2, d_full);
    model_add_op(m, OP_ADD, 3, (uint32_t[]){scaled, b, f6}, 1, (uint32_t[]){out});

    model_identify_io(m, 1, (uint32_t[]){inp}, 1, (uint32_t[]){out});
    int err = model_finish(m);
    if (err) { fprintf(stderr, "finish: %d\n", err); return 1; }

    ANeuralNetworksCompilation* comp = NULL;
    comp_create(m, (const ANeuralNetworksDevice*[]){neuron}, 1, &comp);
    comp_set_pref(comp, 2);
    err = comp_finish(comp);
    if (err) { fprintf(stderr, "compile: %d\n", err); comp_free(comp); model_free(m); return 1; }

    ANeuralNetworksExecution* ex = NULL;
    exec_create(comp, &ex);
    exec_set_input(ex, 0, NULL, input, sizeof(input));
    memset(nnapi_out, 0, sizeof(nnapi_out));
    exec_set_output(ex, 0, NULL, nnapi_out, sizeof(nnapi_out));
    err = exec_compute(ex);
    exec_free(ex);

    if (err) {
        fprintf(stderr, "compute: %d\n", err);
    } else {
        fprintf(stderr, "NNAPI LN first row: ");
        for (int j=0;j<N;j++) fprintf(stderr, "%.4f ", nnapi_out[j]);
        fprintf(stderr, "\n");
        float max_err = 0;
        for (int i=0;i<M*N;i++) {
            float e = fabsf(nnapi_out[i]-cpu_out[i]);
            if (e>max_err) max_err=e;
        }
        fprintf(stderr, "max error: %.6f %s\n", max_err, max_err < 0.01 ? "OK" : "FAIL");
    }

    comp_free(comp); model_free(m);
    dlclose(lib);
    return 0;
}
