#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

// NNAPI types (from NeuralNetworks.h)
typedef void ANeuralNetworksModel;
typedef void ANeuralNetworksCompilation;
typedef void ANeuralNetworksDevice;

typedef struct {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
} ANeuralNetworksOperandType;

#define ANEURALNETWORKS_TENSOR_FLOAT32 3
#define ANEURALNETWORKS_INT32 1
#define ANEURALNETWORKS_ADD 0
#define ANEURALNETWORKS_FUSED_NONE 0

int main() {
    // dlopen NNAPI
    void* lib = dlopen("libneuralnetworks.so", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    fprintf(stderr, "NNAPI loaded OK\n");

    // Resolve symbols
    typedef int (*FnModelCreate)(ANeuralNetworksModel**);
    typedef void (*FnModelFree)(ANeuralNetworksModel*);
    typedef int (*FnModelAddOperand)(ANeuralNetworksModel*, const ANeuralNetworksOperandType*);
    typedef int (*FnModelSetOperandValue)(ANeuralNetworksModel*, int32_t, const void*, size_t);
    typedef int (*FnModelAddOperation)(ANeuralNetworksModel*, int32_t, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
    typedef int (*FnModelIdentifyIO)(ANeuralNetworksModel*, uint32_t, const uint32_t*, uint32_t, const uint32_t*);
    typedef int (*FnModelFinish)(ANeuralNetworksModel*);
    typedef int (*FnGetDeviceCount)(uint32_t*);
    typedef int (*FnGetDevice)(uint32_t, ANeuralNetworksDevice**);
    typedef int (*FnDeviceGetName)(const ANeuralNetworksDevice*, const char**);
    typedef int (*FnDeviceGetType)(const ANeuralNetworksDevice*, int32_t*);

    FnModelCreate model_create = dlsym(lib, "ANeuralNetworksModel_create");
    FnModelFree model_free = dlsym(lib, "ANeuralNetworksModel_free");
    FnModelAddOperand model_add_operand = dlsym(lib, "ANeuralNetworksModel_addOperand");
    FnModelSetOperandValue model_set_value = dlsym(lib, "ANeuralNetworksModel_setOperandValue");
    FnModelAddOperation model_add_op = dlsym(lib, "ANeuralNetworksModel_addOperation");
    FnModelIdentifyIO model_identify_io = dlsym(lib, "ANeuralNetworksModel_identifyInputsAndOutputs");
    FnModelFinish model_finish = dlsym(lib, "ANeuralNetworksModel_finish");
    FnGetDeviceCount get_device_count = dlsym(lib, "ANeuralNetworks_getDeviceCount");
    FnGetDevice get_device = dlsym(lib, "ANeuralNetworks_getDevice");
    FnDeviceGetName device_get_name = dlsym(lib, "ANeuralNetworksDevice_getName");
    FnDeviceGetType device_get_type = dlsym(lib, "ANeuralNetworksDevice_getType");

    if (!model_create || !model_add_operand) {
        fprintf(stderr, "Failed to resolve symbols\n");
        return 1;
    }

    // Enumerate devices
    uint32_t num_devices = 0;
    get_device_count(&num_devices);
    fprintf(stderr, "NNAPI devices: %u\n", num_devices);
    for (uint32_t i = 0; i < num_devices; i++) {
        ANeuralNetworksDevice* dev = NULL;
        get_device(i, &dev);
        const char* name = NULL;
        int32_t type = 0;
        device_get_name(dev, &name);
        device_get_type(dev, &type);
        fprintf(stderr, "  [%u] %s (type=%d)\n", i, name ? name : "?", type);
    }

    // Create model
    ANeuralNetworksModel* model = NULL;
    int err = model_create(&model);
    fprintf(stderr, "model_create: %d (model=%p)\n", err, model);
    if (err != 0) return 1;

    // Test 1: scalar INT32
    ANeuralNetworksOperandType scalar_type = {
        .type = ANEURALNETWORKS_INT32,
        .dimensionCount = 0,
        .dimensions = NULL,
        .scale = 0.0f,
        .zeroPoint = 0
    };
    err = model_add_operand(model, &scalar_type);
    fprintf(stderr, "addOperand(scalar INT32): %d\n", err);

    // Test 2: 1D tensor FLOAT32
    uint32_t dims1[] = {4};
    ANeuralNetworksOperandType tensor1_type = {
        .type = ANEURALNETWORKS_TENSOR_FLOAT32,
        .dimensionCount = 1,
        .dimensions = dims1,
        .scale = 0.0f,
        .zeroPoint = 0
    };
    err = model_add_operand(model, &tensor1_type);
    fprintf(stderr, "addOperand(tensor f32 [4]): %d\n", err);

    // Test 3: 3D tensor FLOAT32
    uint32_t dims3[] = {1, 1500, 384};
    ANeuralNetworksOperandType tensor3_type = {
        .type = ANEURALNETWORKS_TENSOR_FLOAT32,
        .dimensionCount = 3,
        .dimensions = dims3,
        .scale = 0.0f,
        .zeroPoint = 0
    };
    err = model_add_operand(model, &tensor3_type);
    fprintf(stderr, "addOperand(tensor f32 [1,1500,384]): %d\n", err);

    // If tensor works, try a complete ADD model
    if (err == 0) {
        // operand 3: another tensor [1,1500,384]
        err = model_add_operand(model, &tensor3_type);
        fprintf(stderr, "addOperand(output tensor): %d\n", err);

        // operand 4: fuse code (scalar INT32)
        err = model_add_operand(model, &scalar_type);
        fprintf(stderr, "addOperand(fuse code): %d\n", err);

        int32_t fuse = ANEURALNETWORKS_FUSED_NONE;
        err = model_set_value(model, 4, &fuse, sizeof(fuse));
        fprintf(stderr, "setOperandValue(fuse): %d\n", err);

        // ADD op: inputs=[1,2,4], output=[3]
        uint32_t add_inputs[] = {1, 2, 4};
        uint32_t add_outputs[] = {3};
        err = model_add_op(model, ANEURALNETWORKS_ADD, 3, add_inputs, 1, add_outputs);
        fprintf(stderr, "addOperation(ADD): %d\n", err);

        uint32_t model_inputs[] = {1, 2};
        uint32_t model_outputs[] = {3};
        err = model_identify_io(model, 2, model_inputs, 1, model_outputs);
        fprintf(stderr, "identifyIO: %d\n", err);

        err = model_finish(model);
        fprintf(stderr, "model_finish: %d\n", err);
    }

    model_free(model);
    fprintf(stderr, "sizeof(ANeuralNetworksOperandType) = %zu\n", sizeof(ANeuralNetworksOperandType));
    dlclose(lib);
    return 0;
}
