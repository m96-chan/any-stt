/**
 * Debug: Compare Layer 0 Q projection — CPU vs NNAPI.
 *
 * Steps:
 * 1. Load whisper model
 * 2. Run whisper_full to get conv output (encoder input)
 * 3. Extract attn_ln.weight/bias and attn.query.weight/bias
 * 4. Compute LayerNorm + FC(query) on CPU
 * 5. Build same in NNAPI and compare
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include "whisper.h"

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
#define OP_FC 9
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

#define LOAD_SYM(name) name = dlsym(nnapi_lib, "ANeuralNetworks" #name)
static FnModelCreate Model_create;
static FnModelFree Model_free;
static FnModelAddOperand Model_addOperand;
static FnModelSetOperandValue Model_setOperandValue;
static FnModelAddOperation Model_addOperation;
static FnModelIdentifyIO Model_identifyInputsAndOutputs;
static FnModelFinish Model_finish;
static FnCompCreate Compilation_createForDevices;
static FnCompSetPref Compilation_setPreference;
static FnCompFinish Compilation_finish;
static FnCompFree Compilation_free;
static FnExecCreate Execution_create;
static FnExecSetInput Execution_setInput;
static FnExecSetOutput Execution_setOutput;
static FnExecCompute Execution_compute;
static FnExecFree Execution_free;
static FnGetDeviceCount s_getDeviceCount;
static FnGetDevice s_getDevice;
static FnDeviceGetName Device_getName;

static int nop;
static int atf32(ANeuralNetworksModel* m, int nd, const uint32_t* d) {
    ANeuralNetworksOperandType t={T_TENSOR_F32,nd,d,0,0}; int i=nop++; Model_addOperand(m,&t); return i;
}
static int ati32(ANeuralNetworksModel* m, int nd, const uint32_t* d) {
    ANeuralNetworksOperandType t={T_TENSOR_I32,nd,d,0,0}; int i=nop++; Model_addOperand(m,&t); return i;
}
static int asi32(ANeuralNetworksModel* m, int32_t v) {
    ANeuralNetworksOperandType t={T_I32,0,NULL,0,0}; int i=nop++; Model_addOperand(m,&t); Model_setOperandValue(m,i,&v,4); return i;
}
static int ctf32(ANeuralNetworksModel* m, int nd, const uint32_t* d, const float* data, size_t nb) {
    int i=atf32(m,nd,d); Model_setOperandValue(m,i,data,nb); return i;
}

static float* get_tensor(struct whisper_context* ctx, const char* name, int* n) {
    *n = whisper_get_model_tensor_f32(ctx, name, NULL, 0);
    if (*n <= 0) return NULL;
    float* buf = malloc(*n * sizeof(float));
    whisper_get_model_tensor_f32(ctx, name, buf, *n);
    return buf;
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "ggml-tiny.en.bin";

    // Load NNAPI
    void* nnapi_lib = dlopen("libneuralnetworks.so", RTLD_NOW);
    Model_create = dlsym(nnapi_lib, "ANeuralNetworksModel_create");
    Model_free = dlsym(nnapi_lib, "ANeuralNetworksModel_free");
    Model_addOperand = dlsym(nnapi_lib, "ANeuralNetworksModel_addOperand");
    Model_setOperandValue = dlsym(nnapi_lib, "ANeuralNetworksModel_setOperandValue");
    Model_addOperation = dlsym(nnapi_lib, "ANeuralNetworksModel_addOperation");
    Model_identifyInputsAndOutputs = dlsym(nnapi_lib, "ANeuralNetworksModel_identifyInputsAndOutputs");
    Model_finish = dlsym(nnapi_lib, "ANeuralNetworksModel_finish");
    Compilation_createForDevices = dlsym(nnapi_lib, "ANeuralNetworksCompilation_createForDevices");
    Compilation_setPreference = dlsym(nnapi_lib, "ANeuralNetworksCompilation_setPreference");
    Compilation_finish = dlsym(nnapi_lib, "ANeuralNetworksCompilation_finish");
    Compilation_free = dlsym(nnapi_lib, "ANeuralNetworksCompilation_free");
    Execution_create = dlsym(nnapi_lib, "ANeuralNetworksExecution_create");
    Execution_setInput = dlsym(nnapi_lib, "ANeuralNetworksExecution_setInput");
    Execution_setOutput = dlsym(nnapi_lib, "ANeuralNetworksExecution_setOutput");
    Execution_compute = dlsym(nnapi_lib, "ANeuralNetworksExecution_compute");
    Execution_free = dlsym(nnapi_lib, "ANeuralNetworksExecution_free");
    s_getDeviceCount = dlsym(nnapi_lib, "ANeuralNetworks_getDeviceCount");
    s_getDevice = dlsym(nnapi_lib, "ANeuralNetworks_getDevice");
    Device_getName = dlsym(nnapi_lib, "ANeuralNetworksDevice_getName");

    uint32_t nd=0; s_getDeviceCount(&nd);
    ANeuralNetworksDevice* neuron=NULL;
    for (uint32_t i=0;i<nd;i++) {
        ANeuralNetworksDevice*d=NULL; s_getDevice(i,&d);
        const char*nm=NULL; Device_getName(d,&nm);
        if (nm && strstr(nm,"neuron")) neuron=d;
    }
    if (!neuron) { fprintf(stderr, "neuron not found\n"); return 1; }

    // Load whisper model
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cp);
    int ns = whisper_model_n_audio_state(ctx);
    int nc = whisper_model_n_audio_ctx(ctx);
    fprintf(stderr, "n_state=%d n_ctx=%d\n", ns, nc);

    // Run full pipeline with jfk.wav to get conv output
    // Read WAV file (assume 16kHz mono 16-bit PCM)
    const char* wav_path = argc > 2 ? argv[2] : "jfk.wav";
    FILE* wf = fopen(wav_path, "rb");
    float* audio = NULL;
    int n_samples = 0;
    if (wf) {
        fseek(wf, 44, SEEK_SET); // skip WAV header
        fseek(wf, 0, SEEK_END); long fsize = ftell(wf) - 44; fseek(wf, 44, SEEK_SET);
        int16_t* raw = malloc(fsize);
        fread(raw, 1, fsize, wf); fclose(wf);
        n_samples = fsize / 2;
        audio = malloc(n_samples * sizeof(float));
        for (int i = 0; i < n_samples; i++) audio[i] = raw[i] / 32768.0f;
        free(raw);
        fprintf(stderr, "Loaded %s: %d samples (%.1fs)\n", wav_path, n_samples, n_samples/16000.0);
    } else {
        n_samples = 16000 * 11;
        audio = calloc(n_samples, sizeof(float));
        fprintf(stderr, "Using %ds silence\n", n_samples/16000);
    }

    struct whisper_full_params fp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    fp.language="en"; fp.n_threads=8; fp.print_progress=false; fp.print_realtime=false; fp.print_timestamps=false;
    whisper_full(ctx, fp, audio, n_samples);
    free(audio);

    // Get conv output [n_state, n_ctx] layout
    int conv_nc=0, conv_ns=0;
    float* conv_raw = whisper_get_conv_output(ctx, &conv_nc, &conv_ns);
    fprintf(stderr, "conv output: [%d, %d]\n", conv_nc, conv_ns);
    if (!conv_raw) { fprintf(stderr, "conv output null\n"); return 1; }

    // Transpose conv: [n_state rows × n_ctx cols] → [n_ctx rows × n_state cols]
    float* conv = malloc(nc * ns * sizeof(float));
    for (int r=0;r<nc;r++) for (int c=0;c<ns;c++) conv[r*ns+c] = conv_raw[c*nc+r];

    fprintf(stderr, "conv transposed[0..4]: %.4f %.4f %.4f %.4f\n", conv[0], conv[1], conv[2], conv[3]);

    // Get Layer 0 weights
    int n;
    float* ln_w = get_tensor(ctx, "encoder.blocks.0.attn_ln.weight", &n);
    float* ln_b = get_tensor(ctx, "encoder.blocks.0.attn_ln.bias", &n);
    float* q_w = get_tensor(ctx, "encoder.blocks.0.attn.query.weight", &n);
    float* q_b = get_tensor(ctx, "encoder.blocks.0.attn.query.bias", &n);

    fprintf(stderr, "ln_w[0..4]: %.4f %.4f %.4f %.4f\n", ln_w[0], ln_w[1], ln_w[2], ln_w[3]);
    fprintf(stderr, "q_w[0..4]: %.6f %.6f %.6f %.6f\n", q_w[0], q_w[1], q_w[2], q_w[3]);

    // === CPU reference: LayerNorm → FC(query) ===
    float* ln_out = malloc(nc * ns * sizeof(float));
    for (int i=0;i<nc;i++) {
        float mean=0;
        for (int j=0;j<ns;j++) mean += conv[i*ns+j];
        mean /= ns;
        float var=0;
        for (int j=0;j<ns;j++) { float d=conv[i*ns+j]-mean; var+=d*d; }
        var /= ns;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        for (int j=0;j<ns;j++) {
            ln_out[i*ns+j] = ln_w[j] * (conv[i*ns+j] - mean) * inv_std + ln_b[j];
        }
    }
    fprintf(stderr, "CPU LN[0..4]: %.4f %.4f %.4f %.4f\n", ln_out[0], ln_out[1], ln_out[2], ln_out[3]);

    // FC: Q = ln_out @ q_w^T + q_b
    float* cpu_q = malloc(nc * ns * sizeof(float));
    for (int i=0;i<nc;i++) {
        for (int j=0;j<ns;j++) {
            float s = q_b[j];
            for (int k=0;k<ns;k++) s += ln_out[i*ns+k] * q_w[j*ns+k];
            cpu_q[i*ns+j] = s;
        }
    }
    fprintf(stderr, "CPU Q[0..4]: %.4f %.4f %.4f %.4f\n", cpu_q[0], cpu_q[1], cpu_q[2], cpu_q[3]);

    // === NNAPI: LayerNorm only ===
    {
        nop = 0;
        ANeuralNetworksModel* m2 = NULL; Model_create(&m2);
        uint32_t d_full[]={nc,ns}, d_red[]={nc,1}, d_ns[]={ns}, d_ax[]={1};
        int inp2 = atf32(m2, 2, d_full);
        // mean(x, axis=1, keepdims=1)
        int ax1=ati32(m2,1,d_ax); int32_t ax_val=1; Model_setOperandValue(m2,ax1,&ax_val,4);
        int kd1=asi32(m2,1);
        int mu=atf32(m2,2,d_red);
        Model_addOperation(m2,OP_MEAN,3,(uint32_t[]){inp2,ax1,kd1},1,(uint32_t[]){mu});
        // diff = x - mu
        int f1=asi32(m2,FUSED_NONE); int diff=atf32(m2,2,d_full);
        Model_addOperation(m2,OP_SUB,3,(uint32_t[]){inp2,mu,f1},1,(uint32_t[]){diff});
        // diff_sq
        int f2=asi32(m2,FUSED_NONE); int dsq=atf32(m2,2,d_full);
        Model_addOperation(m2,OP_MUL,3,(uint32_t[]){diff,diff,f2},1,(uint32_t[]){dsq});
        // var
        int ax2=ati32(m2,1,d_ax); Model_setOperandValue(m2,ax2,&ax_val,4);
        int kd2=asi32(m2,1); int var2=atf32(m2,2,d_red);
        Model_addOperation(m2,OP_MEAN,3,(uint32_t[]){dsq,ax2,kd2},1,(uint32_t[]){var2});
        // var+eps
        float eps_val=1e-5f;
        int eps2=ctf32(m2,1,(uint32_t[]){1},&eps_val,4);
        int f3=asi32(m2,FUSED_NONE); int ve=atf32(m2,2,d_red);
        Model_addOperation(m2,OP_ADD,3,(uint32_t[]){var2,eps2,f3},1,(uint32_t[]){ve});
        // rsqrt
        int is2=atf32(m2,2,d_red);
        Model_addOperation(m2,OP_RSQRT,1,(uint32_t[]){ve},1,(uint32_t[]){is2});
        // norm = diff * inv_std
        int f4=asi32(m2,FUSED_NONE); int norm2=atf32(m2,2,d_full);
        Model_addOperation(m2,OP_MUL,3,(uint32_t[]){diff,is2,f4},1,(uint32_t[]){norm2});
        // scaled = norm * gamma
        int g2=ctf32(m2,1,d_ns,ln_w,ns*4);
        int f5=asi32(m2,FUSED_NONE); int sc2=atf32(m2,2,d_full);
        Model_addOperation(m2,OP_MUL,3,(uint32_t[]){norm2,g2,f5},1,(uint32_t[]){sc2});
        // output = scaled + beta
        int b2=ctf32(m2,1,d_ns,ln_b,ns*4);
        int f6=asi32(m2,FUSED_NONE); int out2=atf32(m2,2,d_full);
        Model_addOperation(m2,OP_ADD,3,(uint32_t[]){sc2,b2,f6},1,(uint32_t[]){out2});

        Model_identifyInputsAndOutputs(m2,1,(uint32_t[]){inp2},1,(uint32_t[]){out2});
        Model_finish(m2);
        ANeuralNetworksCompilation* c2=NULL;
        Compilation_createForDevices(m2,(const ANeuralNetworksDevice*[]){neuron},1,&c2);
        Compilation_setPreference(c2,2);
        int err2 = Compilation_finish(c2);
        if (err2) { fprintf(stderr, "LN compile: %d\n", err2); } else {
            ANeuralNetworksExecution* e2=NULL;
            Execution_create(c2,&e2);
            Execution_setInput(e2,0,NULL,conv,nc*ns*4);
            float* nnapi_ln = calloc(nc*ns,sizeof(float));
            Execution_setOutput(e2,0,NULL,nnapi_ln,nc*ns*4);
            err2 = Execution_compute(e2);
            Execution_free(e2);
            if (err2) { fprintf(stderr, "LN compute: %d\n", err2); } else {
                fprintf(stderr, "NNAPI LN[0..4]: %.4f %.4f %.4f %.4f\n", nnapi_ln[0], nnapi_ln[1], nnapi_ln[2], nnapi_ln[3]);
                float me=0;
                for (int i=0;i<nc*ns;i++) { float e2=fabsf(nnapi_ln[i]-ln_out[i]); if(e2>me)me=e2; }
                fprintf(stderr, "LN max_err: %.6f %s\n", me, me < 0.01 ? "OK" : "FAIL");
            }
            free(nnapi_ln);
        }
        Compilation_free(c2); Model_free(m2);
    }

    // === NNAPI: Combined LN+FC in one graph (same as Rust build_qkv_graph) ===
    {
        nop = 0;
        ANeuralNetworksModel* m3 = NULL; Model_create(&m3);
        uint32_t d_full3[]={nc,ns}, d_red3[]={nc,1}, d_ns3[]={ns}, d_ax3[]={1};
        int inp3 = atf32(m3, 2, d_full3);
        // LN decomposition
        int ax1b=ati32(m3,1,d_ax3); int32_t axv=1; Model_setOperandValue(m3,ax1b,&axv,4);
        int kd1b=asi32(m3,1);
        int mu2=atf32(m3,2,d_red3);
        Model_addOperation(m3,OP_MEAN,3,(uint32_t[]){inp3,ax1b,kd1b},1,(uint32_t[]){mu2});
        int ff1=asi32(m3,FUSED_NONE); int diff2=atf32(m3,2,d_full3);
        Model_addOperation(m3,OP_SUB,3,(uint32_t[]){inp3,mu2,ff1},1,(uint32_t[]){diff2});
        int ff2=asi32(m3,FUSED_NONE); int dsq2=atf32(m3,2,d_full3);
        Model_addOperation(m3,OP_MUL,3,(uint32_t[]){diff2,diff2,ff2},1,(uint32_t[]){dsq2});
        int ax2b=ati32(m3,1,d_ax3); Model_setOperandValue(m3,ax2b,&axv,4);
        int kd2b=asi32(m3,1); int var3=atf32(m3,2,d_red3);
        Model_addOperation(m3,OP_MEAN,3,(uint32_t[]){dsq2,ax2b,kd2b},1,(uint32_t[]){var3});
        float epsv=1e-5f;
        int eps3=ctf32(m3,1,(uint32_t[]){1},&epsv,4);
        int ff3=asi32(m3,FUSED_NONE); int ve3=atf32(m3,2,d_red3);
        Model_addOperation(m3,OP_ADD,3,(uint32_t[]){var3,eps3,ff3},1,(uint32_t[]){ve3});
        int is3=atf32(m3,2,d_red3);
        Model_addOperation(m3,OP_RSQRT,1,(uint32_t[]){ve3},1,(uint32_t[]){is3});
        int ff4=asi32(m3,FUSED_NONE); int nm3=atf32(m3,2,d_full3);
        Model_addOperation(m3,OP_MUL,3,(uint32_t[]){diff2,is3,ff4},1,(uint32_t[]){nm3});
        int gm3=ctf32(m3,1,d_ns3,ln_w,ns*4);
        int ff5=asi32(m3,FUSED_NONE); int sc3=atf32(m3,2,d_full3);
        Model_addOperation(m3,OP_MUL,3,(uint32_t[]){nm3,gm3,ff5},1,(uint32_t[]){sc3});
        int bt3=ctf32(m3,1,d_ns3,ln_b,ns*4);
        int ff6=asi32(m3,FUSED_NONE); int ln3=atf32(m3,2,d_full3);
        Model_addOperation(m3,OP_ADD,3,(uint32_t[]){sc3,bt3,ff6},1,(uint32_t[]){ln3});
        // FC(query)
        uint32_t d_w3[]={ns,ns}, d_b3[]={ns};
        int w3=ctf32(m3,2,d_w3,q_w,ns*ns*4);
        int b3=ctf32(m3,1,d_b3,q_b,ns*4);
        int fuse3=asi32(m3,FUSED_NONE);
        int out3=atf32(m3,2,d_full3);
        Model_addOperation(m3,OP_FC,4,(uint32_t[]){ln3,w3,b3,fuse3},1,(uint32_t[]){out3});

        Model_identifyInputsAndOutputs(m3,1,(uint32_t[]){inp3},1,(uint32_t[]){out3});
        Model_finish(m3);
        ANeuralNetworksCompilation* c3=NULL;
        Compilation_createForDevices(m3,(const ANeuralNetworksDevice*[]){neuron},1,&c3);
        Compilation_setPreference(c3,2);
        int err3 = Compilation_finish(c3);
        if (err3) { fprintf(stderr, "LN+FC compile: %d\n", err3); } else {
            ANeuralNetworksExecution* e3=NULL; Execution_create(c3,&e3);
            Execution_setInput(e3,0,NULL,conv,nc*ns*4);
            float* cq3 = calloc(nc*ns,sizeof(float));
            Execution_setOutput(e3,0,NULL,cq3,nc*ns*4);
            err3 = Execution_compute(e3); Execution_free(e3);
            if (err3) { fprintf(stderr, "LN+FC compute: %d\n", err3); } else {
                fprintf(stderr, "NNAPI LN+FC Q[0..4]: %.4f %.4f %.4f %.4f\n", cq3[0], cq3[1], cq3[2], cq3[3]);
                float me=0;
                for (int i=0;i<nc*ns;i++) { float e=fabsf(cq3[i]-cpu_q[i]); if(e>me)me=e; }
                fprintf(stderr, "LN+FC max_err: %.6f %s\n", me, me < 0.01 ? "OK" : "FAIL");
            }
            free(cq3); Compilation_free(c3); Model_free(m3);
        }
    }

    // === NNAPI: Just FC (no LayerNorm, to isolate) ===
    // Test: FC on ln_out (CPU computed) with same weights
    nop = 0;
    ANeuralNetworksModel* m = NULL; Model_create(&m);
    uint32_t d_in[]={nc,ns}, d_w[]={ns,ns}, d_b[]={ns}, d_out[]={nc,ns};
    int inp = atf32(m, 2, d_in);
    int w = ctf32(m, 2, d_w, q_w, ns*ns*4);
    int b = ctf32(m, 1, d_b, q_b, ns*4);
    int fuse = asi32(m, FUSED_NONE);
    int out = atf32(m, 2, d_out);
    Model_addOperation(m, OP_FC, 4, (uint32_t[]){inp,w,b,fuse}, 1, (uint32_t[]){out});
    Model_identifyInputsAndOutputs(m, 1, (uint32_t[]){inp}, 1, (uint32_t[]){out});
    Model_finish(m);

    ANeuralNetworksCompilation* comp=NULL;
    Compilation_createForDevices(m, (const ANeuralNetworksDevice*[]){neuron}, 1, &comp);
    Compilation_setPreference(comp, 2);
    int err = Compilation_finish(comp);
    if (err) { fprintf(stderr, "compile: %d\n", err); return 1; }

    ANeuralNetworksExecution* ex=NULL;
    Execution_create(comp, &ex);
    Execution_setInput(ex, 0, NULL, ln_out, nc*ns*4);
    float* nnapi_q = calloc(nc*ns, sizeof(float));
    Execution_setOutput(ex, 0, NULL, nnapi_q, nc*ns*4);
    err = Execution_compute(ex);
    Execution_free(ex);

    if (err) {
        fprintf(stderr, "NNAPI FC compute: %d\n", err);
    } else {
        fprintf(stderr, "NNAPI Q[0..4]: %.4f %.4f %.4f %.4f\n", nnapi_q[0], nnapi_q[1], nnapi_q[2], nnapi_q[3]);
        float max_err = 0;
        for (int i=0;i<nc*ns;i++) {
            float e = fabsf(nnapi_q[i] - cpu_q[i]);
            if (e > max_err) max_err = e;
        }
        fprintf(stderr, "FC max_err: %.6f %s\n", max_err, max_err < 0.01 ? "OK" : "FAIL");
    }

    Compilation_free(comp); Model_free(m);
    free(conv); free(ln_out); free(cpu_q); free(nnapi_q);
    free(ln_w); free(ln_b); free(q_w); free(q_b);
    whisper_free(ctx);
    dlclose(nnapi_lib);
    return 0;
}
