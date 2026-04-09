/**
 * Debug: Compare NNAPI FULLY_CONNECTED output vs CPU matmul.
 * Tests with small known matrices to verify NNAPI FC semantics.
 *
 * NNAPI FC: output[i,j] = sum_k(input[i,k] * weights[j,k]) + bias[j]
 * i.e., output = input @ weights^T + bias
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
#define OP_FC 9
#define FUSED_NONE 0

// dlopen functions
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
typedef int (*FnDeviceGetType)(const ANeuralNetworksDevice*, int32_t*);

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
static FnDeviceGetType device_get_type;

static int nop;

static int add_tensor(ANeuralNetworksModel* m, int ndim, const uint32_t* dims) {
    ANeuralNetworksOperandType t = {T_TENSOR_F32, ndim, dims, 0, 0};
    int idx = nop++; model_add_operand(m, &t); return idx;
}
static int add_scalar_i32(ANeuralNetworksModel* m, int32_t val) {
    ANeuralNetworksOperandType t = {T_I32, 0, NULL, 0, 0};
    int idx = nop++; model_add_operand(m, &t);
    model_set_value(m, idx, &val, sizeof(val)); return idx;
}

/// CPU reference: output = input @ weights^T + bias
/// input [M, K], weights [N, K], bias [N], output [M, N]
static void cpu_fc(const float* input, const float* weights, const float* bias,
                   float* output, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias[j];
            for (int k = 0; k < K; k++) {
                sum += input[i*K + k] * weights[j*K + k];
            }
            output[i*N + j] = sum;
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
    device_get_type = dlsym(lib, "ANeuralNetworksDevice_getType");

    // Find neuron device
    uint32_t ndev = 0; get_device_count(&ndev);
    ANeuralNetworksDevice* neuron_dev = NULL;
    ANeuralNetworksDevice* ref_dev = NULL;
    for (uint32_t i = 0; i < ndev; i++) {
        ANeuralNetworksDevice* d = NULL; get_device(i, &d);
        const char* nm = NULL; device_get_name(d, &nm);
        if (nm && strstr(nm, "neuron")) neuron_dev = d;
        if (nm && strstr(nm, "reference")) ref_dev = d;
    }

    // Test: FC with known values
    // input [3, 4], weights [2, 4], bias [2] → output [3, 2]
    const int M = 3, K = 4, N = 2;
    float input[12]  = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    float weights[8] = {1,0,1,0, 0,1,0,1}; // w[0]=[1,0,1,0], w[1]=[0,1,0,1]
    float bias[2]    = {0.5, -0.5};
    float cpu_out[6], nnapi_out[6];

    // CPU reference
    cpu_fc(input, weights, bias, cpu_out, M, K, N);
    fprintf(stderr, "CPU FC output:\n");
    for (int i = 0; i < M; i++) {
        fprintf(stderr, "  [%d]: %.2f %.2f\n", i, cpu_out[i*N], cpu_out[i*N+1]);
    }
    // Expected: row0: (1+3)+0.5=4.5, (2+4)-0.5=5.5
    //           row1: (5+7)+0.5=12.5, (6+8)-0.5=13.5
    //           row2: (9+11)+0.5=20.5, (10+12)-0.5=21.5

    ANeuralNetworksDevice* devs[] = {neuron_dev, ref_dev};
    const char* dev_names[] = {"neuron", "reference"};
    for (int d = 0; d < 2; d++) {
        if (!devs[d]) continue;
        fprintf(stderr, "\nNNAPI FC on %s:\n", dev_names[d]);

        nop = 0;
        ANeuralNetworksModel* m = NULL; model_create(&m);

        uint32_t d_in[] = {M, K};
        uint32_t d_w[]  = {N, K};
        uint32_t d_b[]  = {N};
        uint32_t d_out[] = {M, N};

        int inp = add_tensor(m, 2, d_in);       // 0: input [3,4]
        int w   = add_tensor(m, 2, d_w);         // 1: weights [2,4]
        model_set_value(m, w, weights, sizeof(weights));
        int b   = add_tensor(m, 1, d_b);         // 2: bias [2]
        model_set_value(m, b, bias, sizeof(bias));
        int fuse = add_scalar_i32(m, FUSED_NONE); // 3: fuse
        int out  = add_tensor(m, 2, d_out);       // 4: output [3,2]

        uint32_t fc_in[] = {inp, w, b, fuse};
        uint32_t fc_out[] = {out};
        int err = model_add_op(m, OP_FC, 4, fc_in, 1, fc_out);
        if (err) { fprintf(stderr, "  addOp: %d\n", err); model_free(m); continue; }

        model_identify_io(m, 1, (uint32_t[]){inp}, 1, fc_out);
        err = model_finish(m);
        if (err) { fprintf(stderr, "  finish: %d\n", err); model_free(m); continue; }

        ANeuralNetworksCompilation* comp = NULL;
        comp_create(m, (const ANeuralNetworksDevice*[]){devs[d]}, 1, &comp);
        comp_set_pref(comp, 2);
        err = comp_finish(comp);
        if (err) { fprintf(stderr, "  compile: %d\n", err); comp_free(comp); model_free(m); continue; }

        ANeuralNetworksExecution* ex = NULL;
        exec_create(comp, &ex);
        exec_set_input(ex, 0, NULL, input, sizeof(input));
        memset(nnapi_out, 0, sizeof(nnapi_out));
        exec_set_output(ex, 0, NULL, nnapi_out, sizeof(nnapi_out));
        err = exec_compute(ex);
        exec_free(ex);

        if (err) {
            fprintf(stderr, "  compute: %d\n", err);
        } else {
            fprintf(stderr, "  output:\n");
            float max_err = 0;
            for (int i = 0; i < M; i++) {
                fprintf(stderr, "  [%d]: %.4f %.4f (cpu: %.4f %.4f)\n",
                    i, nnapi_out[i*N], nnapi_out[i*N+1],
                    cpu_out[i*N], cpu_out[i*N+1]);
                for (int j = 0; j < N; j++) {
                    float e = fabsf(nnapi_out[i*N+j] - cpu_out[i*N+j]);
                    if (e > max_err) max_err = e;
                }
            }
            fprintf(stderr, "  max error: %.6f %s\n", max_err, max_err < 0.01 ? "OK" : "FAIL");
        }

        comp_free(comp); model_free(m);
    }

    // Test 2: Larger FC matching whisper tiny encoder dimensions
    // input [1500, 384], weights [384, 384], bias [384]
    fprintf(stderr, "\n--- Large FC test [1500, 384] @ [384, 384] ---\n");
    const int LM = 1500, LK = 384, LN = 384;
    float* l_input = calloc(LM * LK, sizeof(float));
    float* l_weights = calloc(LN * LK, sizeof(float));
    float* l_bias = calloc(LN, sizeof(float));
    float* l_cpu_out = calloc(LM * LN, sizeof(float));
    float* l_nnapi_out = calloc(LM * LN, sizeof(float));

    // Initialize with small values
    for (int i = 0; i < LM * LK; i++) l_input[i] = 0.01f * (i % 100 - 50);
    for (int i = 0; i < LN * LK; i++) l_weights[i] = 0.001f * (i % 200 - 100);
    for (int i = 0; i < LN; i++) l_bias[i] = 0.01f * (i % 50);

    cpu_fc(l_input, l_weights, l_bias, l_cpu_out, LM, LK, LN);
    fprintf(stderr, "CPU first 4: %.4f %.4f %.4f %.4f\n",
        l_cpu_out[0], l_cpu_out[1], l_cpu_out[2], l_cpu_out[3]);

    for (int d = 0; d < 2; d++) {
        if (!devs[d]) continue;
        fprintf(stderr, "\nNNAPI FC [%d,%d]@[%d,%d] on %s:\n", LM, LK, LN, LK, dev_names[d]);

        nop = 0;
        ANeuralNetworksModel* m = NULL; model_create(&m);
        uint32_t d_in[] = {LM, LK}, d_w[] = {LN, LK}, d_b[] = {LN}, d_out[] = {LM, LN};

        int inp = add_tensor(m, 2, d_in);
        int w = add_tensor(m, 2, d_w);
        model_set_value(m, w, l_weights, LN*LK*4);
        int b = add_tensor(m, 1, d_b);
        model_set_value(m, b, l_bias, LN*4);
        int fuse = add_scalar_i32(m, FUSED_NONE);
        int out = add_tensor(m, 2, d_out);

        model_add_op(m, OP_FC, 4, (uint32_t[]){inp,w,b,fuse}, 1, (uint32_t[]){out});
        model_identify_io(m, 1, (uint32_t[]){inp}, 1, (uint32_t[]){out});
        model_finish(m);

        ANeuralNetworksCompilation* comp = NULL;
        comp_create(m, (const ANeuralNetworksDevice*[]){devs[d]}, 1, &comp);
        comp_set_pref(comp, 2);
        int err = comp_finish(comp);
        if (err) { fprintf(stderr, "  compile: %d\n", err); comp_free(comp); model_free(m); continue; }

        ANeuralNetworksExecution* ex = NULL;
        exec_create(comp, &ex);
        exec_set_input(ex, 0, NULL, l_input, LM*LK*4);
        memset(l_nnapi_out, 0, LM*LN*4);
        exec_set_output(ex, 0, NULL, l_nnapi_out, LM*LN*4);
        err = exec_compute(ex);
        exec_free(ex);

        if (err) {
            fprintf(stderr, "  compute: %d\n", err);
        } else {
            fprintf(stderr, "  first 4: %.4f %.4f %.4f %.4f\n",
                l_nnapi_out[0], l_nnapi_out[1], l_nnapi_out[2], l_nnapi_out[3]);
            float max_err = 0, sum_err = 0;
            int n_err = 0;
            for (int i = 0; i < LM * LN; i++) {
                float e = fabsf(l_nnapi_out[i] - l_cpu_out[i]);
                if (e > max_err) max_err = e;
                sum_err += e;
                if (e > 0.01) n_err++;
            }
            fprintf(stderr, "  max_err=%.6f avg_err=%.6f n_bad=%d/%d %s\n",
                max_err, sum_err / (LM*LN), n_err, LM*LN,
                max_err < 0.1 ? "OK" : "FAIL");
        }
        comp_free(comp); model_free(m);
    }

    free(l_input); free(l_weights); free(l_bias);
    free(l_cpu_out); free(l_nnapi_out);
    dlclose(lib);
    return 0;
}
