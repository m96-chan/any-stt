#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>

typedef void ANeuralNetworksModel;
typedef void ANeuralNetworksCompilation;
typedef void ANeuralNetworksExecution;
typedef void ANeuralNetworksDevice;

typedef struct {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
} ANeuralNetworksOperandType;

// Operand types
#define T_F32       0   // scalar float32
#define T_I32       1   // scalar int32
#define T_TENSOR_F32 3  // tensor float32
#define T_TENSOR_I32 4  // tensor int32
#define T_BOOL      6   // scalar bool

// Op codes
#define OP_ADD                0
#define OP_FULLY_CONNECTED    9
#define OP_LOGISTIC          14
#define OP_MUL               18
#define OP_RESHAPE           22
#define OP_SOFTMAX           25
#define OP_TANH              28
#define OP_MEAN              31
#define OP_SUB               36
#define OP_TRANSPOSE         37
#define OP_RSQRT             83
#define OP_BATCH_MATMUL     102

// Fuse
#define FUSED_NONE 0

// Function pointer types
typedef int (*FnModelCreate)(ANeuralNetworksModel**);
typedef void (*FnModelFree)(ANeuralNetworksModel*);
typedef int (*FnModelAddOperand)(ANeuralNetworksModel*, const ANeuralNetworksOperandType*);
typedef int (*FnModelSetOperandValue)(ANeuralNetworksModel*, int32_t, const void*, size_t);
typedef int (*FnModelAddOperation)(ANeuralNetworksModel*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
typedef int (*FnModelIdentifyIO)(ANeuralNetworksModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
typedef int (*FnModelFinish)(ANeuralNetworksModel*);
typedef int (*FnCompCreateForDevices)(const ANeuralNetworksModel*, const ANeuralNetworksDevice* const*, uint32_t, ANeuralNetworksCompilation**);
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

// Globals
static FnModelCreate model_create;
static FnModelFree model_free;
static FnModelAddOperand model_add_operand;
static FnModelSetOperandValue model_set_value;
static FnModelAddOperation model_add_op;
static FnModelIdentifyIO model_identify_io;
static FnModelFinish model_finish;
static FnCompCreateForDevices comp_create;
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

static int next_op;

static int add_tensor(ANeuralNetworksModel* m, uint32_t ndim, const uint32_t* dims) {
    ANeuralNetworksOperandType t = {T_TENSOR_F32, ndim, dims, 0.0f, 0};
    int idx = next_op++;
    return model_add_operand(m, &t) == 0 ? idx : -1;
}

static int add_scalar_i32(ANeuralNetworksModel* m, int32_t val) {
    ANeuralNetworksOperandType t = {T_I32, 0, NULL, 0.0f, 0};
    int idx = next_op++;
    if (model_add_operand(m, &t) != 0) return -1;
    model_set_value(m, idx, &val, sizeof(val));
    return idx;
}

static int add_scalar_f32(ANeuralNetworksModel* m, float val) {
    ANeuralNetworksOperandType t = {T_F32, 0, NULL, 0.0f, 0};
    int idx = next_op++;
    if (model_add_operand(m, &t) != 0) return -1;
    model_set_value(m, idx, &val, sizeof(val));
    return idx;
}

static int add_scalar_bool(ANeuralNetworksModel* m, uint8_t val) {
    ANeuralNetworksOperandType t = {T_BOOL, 0, NULL, 0.0f, 0};
    int idx = next_op++;
    if (model_add_operand(m, &t) != 0) return -1;
    model_set_value(m, idx, &val, sizeof(val));
    return idx;
}

// Test: compile + execute a minimal model on a given device
// Returns: 0=OK, 1=compile_fail, 2=exec_fail
static int test_model(ANeuralNetworksModel* m, const ANeuralNetworksDevice* dev,
                      const uint32_t* inputs, uint32_t n_in,
                      const uint32_t* outputs, uint32_t n_out,
                      float* in_data, size_t in_bytes,
                      float* out_data, size_t out_bytes) {
    int err;
    err = model_identify_io(m, n_in, inputs, n_out, outputs);
    if (err) { fprintf(stderr, "    identifyIO: %d\n", err); return 1; }
    err = model_finish(m);
    if (err) { fprintf(stderr, "    model_finish: %d\n", err); return 1; }

    ANeuralNetworksCompilation* comp = NULL;
    const ANeuralNetworksDevice* devs[] = {dev};
    err = comp_create(m, devs, 1, &comp);
    if (err) { fprintf(stderr, "    comp_create: %d\n", err); return 1; }
    comp_set_pref(comp, 2); // SUSTAINED_SPEED
    err = comp_finish(comp);
    if (err) { fprintf(stderr, "    comp_finish: %d\n", err); comp_free(comp); return 1; }

    ANeuralNetworksExecution* exec = NULL;
    err = exec_create(comp, &exec);
    if (err) { comp_free(comp); return 2; }
    exec_set_input(exec, 0, NULL, in_data, in_bytes);
    exec_set_output(exec, 0, NULL, out_data, out_bytes);
    err = exec_compute(exec);
    exec_free(exec);
    comp_free(comp);
    if (err) { fprintf(stderr, "    exec_compute: %d\n", err); return 2; }
    return 0;
}

// ===== Individual op tests =====

static void test_binop(const ANeuralNetworksDevice* dev, const char* dname, int op, const char* opname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d[] = {1, 4, 3};
    int a = add_tensor(m, 3, d);
    int b = add_tensor(m, 3, d);
    int fuse = add_scalar_i32(m, FUSED_NONE);
    int out = add_tensor(m, 3, d);
    uint32_t ins[] = {a, b, fuse}, outs[] = {out};
    model_add_op(m, op, 3, ins, 1, outs);

    // Manual compile+exec with 2 inputs
    model_identify_io(m, 2, (uint32_t[]){a, b}, 1, outs);
    int err = model_finish(m);
    if (err) { fprintf(stderr, "  [%s] %s: model_finish failed (%d)\n", dname, opname, err); model_free(m); return; }

    ANeuralNetworksCompilation* comp = NULL;
    const ANeuralNetworksDevice* devs[] = {dev};
    comp_create(m, devs, 1, &comp);
    comp_set_pref(comp, 2);
    err = comp_finish(comp);
    if (err) { fprintf(stderr, "  [%s] %s: COMPILE_FAIL (%d)\n", dname, opname, err); comp_free(comp); model_free(m); return; }

    ANeuralNetworksExecution* exec = NULL;
    exec_create(comp, &exec);
    float in_a[12], in_b[12], out_data[12];
    for(int i=0;i<12;i++) { in_a[i]=2.0f; in_b[i]=3.0f; }
    memset(out_data, 0, sizeof(out_data));
    exec_set_input(exec, 0, NULL, in_a, 48);
    exec_set_input(exec, 1, NULL, in_b, 48);
    exec_set_output(exec, 0, NULL, out_data, 48);
    err = exec_compute(exec);
    exec_free(exec); comp_free(comp); model_free(m);
    fprintf(stderr, "  [%s] %s: %s (out[0]=%.1f)\n", dname, opname,
        err==0?"OK":"EXEC_FAIL", out_data[0]);
}

static void test_softmax(const ANeuralNetworksDevice* dev, const char* dname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d[] = {1, 4, 3};
    int a = add_tensor(m, 3, d);
    int beta = add_scalar_f32(m, 1.0f);
    int out = add_tensor(m, 3, d);
    uint32_t ins[] = {a, beta}, outs[] = {out};
    model_add_op(m, OP_SOFTMAX, 2, ins, 1, outs);
    float in_data[12]; for(int i=0;i<12;i++) in_data[i]=0.1f;
    float out_data[12] = {0};
    int r = test_model(m, dev, (uint32_t[]){a}, 1, outs, 1, in_data, 48, out_data, 48);
    fprintf(stderr, "  [%s] SOFTMAX: %s\n", dname, r==0?"OK":r==1?"COMPILE_FAIL":"EXEC_FAIL");
    model_free(m);
}

static void test_logistic(const ANeuralNetworksDevice* dev, const char* dname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d[] = {1, 4, 3};
    int a = add_tensor(m, 3, d);
    int out = add_tensor(m, 3, d);
    uint32_t ins[] = {a}, outs[] = {out};
    model_add_op(m, OP_LOGISTIC, 1, ins, 1, outs);
    float in_data[12] = {0}; float out_data[12] = {0};
    int r = test_model(m, dev, (uint32_t[]){a}, 1, outs, 1, in_data, 48, out_data, 48);
    fprintf(stderr, "  [%s] LOGISTIC(sigmoid): %s\n", dname, r==0?"OK":r==1?"COMPILE_FAIL":"EXEC_FAIL");
    model_free(m);
}

static void test_tanh_op(const ANeuralNetworksDevice* dev, const char* dname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d[] = {1, 4, 3};
    int a = add_tensor(m, 3, d);
    int out = add_tensor(m, 3, d);
    uint32_t ins[] = {a}, outs[] = {out};
    model_add_op(m, OP_TANH, 1, ins, 1, outs);
    float in_data[12] = {0}; float out_data[12] = {0};
    int r = test_model(m, dev, (uint32_t[]){a}, 1, outs, 1, in_data, 48, out_data, 48);
    fprintf(stderr, "  [%s] TANH: %s\n", dname, r==0?"OK":r==1?"COMPILE_FAIL":"EXEC_FAIL");
    model_free(m);
}

static void test_mean(const ANeuralNetworksDevice* dev, const char* dname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d_in[] = {1, 4, 3};
    uint32_t d_out[] = {1, 4, 1};
    uint32_t d_axis[] = {1};
    int a = add_tensor(m, 3, d_in);
    int axis = add_tensor(m, 1, d_axis); // TENSOR_INT32 for axis
    // override to INT32 tensor
    {
        ANeuralNetworksOperandType t = {T_TENSOR_I32, 1, d_axis, 0.0f, 0};
        // Can't easily override; let's re-create
    }
    // Actually, axis must be TENSOR_INT32. Let me redo.
    model_free(m); m = NULL; model_create(&m); next_op = 0;
    int in_t = add_tensor(m, 3, d_in);
    // axis tensor (INT32)
    ANeuralNetworksOperandType axis_type = {T_TENSOR_I32, 1, d_axis, 0.0f, 0};
    int axis_idx = next_op++;
    model_add_operand(m, &axis_type);
    int32_t axis_val = 2; // last dim
    model_set_value(m, axis_idx, &axis_val, sizeof(axis_val));
    int keepdims = add_scalar_i32(m, 1);
    int out = add_tensor(m, 3, d_out);
    uint32_t ins[] = {in_t, axis_idx, keepdims}, outs[] = {out};
    int err = model_add_op(m, OP_MEAN, 3, ins, 1, outs);
    if (err) { fprintf(stderr, "  [%s] MEAN: addOp failed (%d)\n", dname, err); model_free(m); return; }
    float in_data[12]; for(int i=0;i<12;i++) in_data[i]=1.0f;
    float out_data[4] = {0};
    int r = test_model(m, dev, (uint32_t[]){in_t}, 1, outs, 1, in_data, 48, out_data, 16);
    fprintf(stderr, "  [%s] MEAN: %s\n", dname, r==0?"OK":r==1?"COMPILE_FAIL":"EXEC_FAIL");
    model_free(m);
}

static void test_rsqrt(const ANeuralNetworksDevice* dev, const char* dname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d[] = {1, 4, 3};
    int a = add_tensor(m, 3, d);
    int out = add_tensor(m, 3, d);
    uint32_t ins[] = {a}, outs[] = {out};
    model_add_op(m, OP_RSQRT, 1, ins, 1, outs);
    float in_data[12]; for(int i=0;i<12;i++) in_data[i]=4.0f;
    float out_data[12] = {0};
    int r = test_model(m, dev, (uint32_t[]){a}, 1, outs, 1, in_data, 48, out_data, 48);
    fprintf(stderr, "  [%s] RSQRT: %s (out[0]=%.4f, expect 0.5)\n", dname,
        r==0?"OK":r==1?"COMPILE_FAIL":"EXEC_FAIL", out_data[0]);
    model_free(m);
}

static void test_batch_matmul(const ANeuralNetworksDevice* dev, const char* dname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d_a[] = {1, 4, 3};
    uint32_t d_b[] = {1, 3, 2};
    uint32_t d_out[] = {1, 4, 2};
    int a = add_tensor(m, 3, d_a);
    int b = add_tensor(m, 3, d_b);
    int adj_x = add_scalar_bool(m, 0);
    int adj_y = add_scalar_bool(m, 0);
    int out = add_tensor(m, 3, d_out);
    uint32_t ins[] = {a, b, adj_x, adj_y}, outs[] = {out};
    int err = model_add_op(m, OP_BATCH_MATMUL, 4, ins, 1, outs);
    if (err) { fprintf(stderr, "  [%s] BATCH_MATMUL: addOp failed (%d)\n", dname, err); model_free(m); return; }
    float in_a[12]; for(int i=0;i<12;i++) in_a[i]=1.0f;
    float in_b[6]; for(int i=0;i<6;i++) in_b[i]=1.0f;
    float out_data[8] = {0};
    // Need 2 inputs
    model_identify_io(m, 2, (uint32_t[]){a, b}, 1, outs);
    model_finish(m);
    ANeuralNetworksCompilation* comp = NULL;
    const ANeuralNetworksDevice* devs[] = {dev};
    comp_create(m, devs, 1, &comp);
    comp_set_pref(comp, 2);
    err = comp_finish(comp);
    if (err) { fprintf(stderr, "  [%s] BATCH_MATMUL: COMPILE_FAIL (%d)\n", dname, err); comp_free(comp); model_free(m); return; }
    ANeuralNetworksExecution* exec = NULL;
    exec_create(comp, &exec);
    exec_set_input(exec, 0, NULL, in_a, 48);
    exec_set_input(exec, 1, NULL, in_b, 24);
    exec_set_output(exec, 0, NULL, out_data, 32);
    err = exec_compute(exec);
    exec_free(exec); comp_free(comp); model_free(m);
    fprintf(stderr, "  [%s] BATCH_MATMUL: %s (out[0]=%.1f, expect 3.0)\n", dname,
        err==0?"OK":"EXEC_FAIL", out_data[0]);
}

static void test_fully_connected(const ANeuralNetworksDevice* dev, const char* dname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    // input [4, 3], weights [2, 3], bias [2] → output [4, 2]
    uint32_t d_in[] = {4, 3};
    uint32_t d_w[] = {2, 3};
    uint32_t d_b[] = {2};
    uint32_t d_out[] = {4, 2};
    int inp = add_tensor(m, 2, d_in);
    int w = add_tensor(m, 2, d_w);
    // Set weights as constant
    float w_data[6]; for(int i=0;i<6;i++) w_data[i]=1.0f;
    model_set_value(m, w, w_data, sizeof(w_data));
    int b = add_tensor(m, 1, d_b);
    float b_data[2] = {0.0f, 0.0f};
    model_set_value(m, b, b_data, sizeof(b_data));
    int fuse = add_scalar_i32(m, FUSED_NONE);
    int out = add_tensor(m, 2, d_out);
    uint32_t ins[] = {inp, w, b, fuse}, outs[] = {out};
    model_add_op(m, OP_FULLY_CONNECTED, 4, ins, 1, outs);
    float in_data[12]; for(int i=0;i<12;i++) in_data[i]=1.0f;
    float out_data[8] = {0};
    int r = test_model(m, dev, (uint32_t[]){inp}, 1, outs, 1, in_data, 48, out_data, 32);
    fprintf(stderr, "  [%s] FULLY_CONNECTED: %s (out[0]=%.1f, expect 3.0)\n", dname,
        r==0?"OK":r==1?"COMPILE_FAIL":"EXEC_FAIL", out_data[0]);
    model_free(m);
}

static void test_reshape(const ANeuralNetworksDevice* dev, const char* dname) {
    next_op = 0;
    ANeuralNetworksModel* m = NULL; model_create(&m);
    uint32_t d_in[] = {1, 4, 3};
    uint32_t d_shape[] = {2};
    uint32_t d_out[] = {4, 3};
    int inp = add_tensor(m, 3, d_in);
    // shape tensor
    ANeuralNetworksOperandType shape_type = {T_TENSOR_I32, 1, d_shape, 0.0f, 0};
    int shape = next_op++;
    model_add_operand(m, &shape_type);
    int32_t shape_data[] = {4, 3};
    model_set_value(m, shape, shape_data, sizeof(shape_data));
    int out = add_tensor(m, 2, d_out);
    uint32_t ins[] = {inp, shape}, outs[] = {out};
    model_add_op(m, OP_RESHAPE, 2, ins, 1, outs);
    float in_data[12]; for(int i=0;i<12;i++) in_data[i]=1.0f;
    float out_data[12] = {0};
    int r = test_model(m, dev, (uint32_t[]){inp}, 1, outs, 1, in_data, 48, out_data, 48);
    fprintf(stderr, "  [%s] RESHAPE: %s\n", dname, r==0?"OK":r==1?"COMPILE_FAIL":"EXEC_FAIL");
    model_free(m);
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

    uint32_t n = 0; get_device_count(&n);
    fprintf(stderr, "=== NNAPI Op Compatibility Probe ===\n");
    fprintf(stderr, "Devices: %u\n\n", n);

    for (uint32_t i = 0; i < n; i++) {
        ANeuralNetworksDevice* dev = NULL;
        get_device(i, &dev);
        const char* name = NULL;
        int32_t type = 0;
        device_get_name(dev, &name);
        device_get_type(dev, &type);
        if (!name) name = "?";

        const char* tstr = type==2?"CPU":type==3?"GPU":type==4?"NPU":"OTHER";
        fprintf(stderr, "--- Device: %s (%s) ---\n", name, tstr);

        test_binop(dev, name, OP_ADD, "ADD");
        test_binop(dev, name, OP_MUL, "MUL");
        test_binop(dev, name, OP_SUB, "SUB");
        test_softmax(dev, name);
        test_logistic(dev, name);
        test_tanh_op(dev, name);
        test_mean(dev, name);
        test_rsqrt(dev, name);
        test_reshape(dev, name);
        test_fully_connected(dev, name);
        test_batch_matmul(dev, name);
        fprintf(stderr, "\n");
    }

    dlclose(lib);
    return 0;
}
