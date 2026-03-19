# any-stt

Cross-platform Speech-to-Text engine for Rust.
Detects hardware (CPU/GPU/NPU) and OS at runtime, then selects the optimal acceleration backend automatically.

## Target Platforms

| Platform       | Arch   | Acceleration                          |
|----------------|--------|---------------------------------------|
| Linux          | x86_64 | CUDA, Vulkan†, CPU (AVX2/AVX-512)    |
| macOS          | ARM64  | CoreML, Metal, CPU (NEON)             |
| Android        | ARM64  | NNAPI, CPU (NEON), Vulkan†(opt-in)    |
| iOS            | ARM64  | CoreML, Metal, CPU (NEON)             |

> †Vulkan: shader pre-compilation required (see [Vulkan Caveats](#vulkan-caveats))

## How It Works

```
Audio in ──► any-stt::initialize(config)
                │
                ├─ 1. Detect hardware
                │     CPU: flags (AVX-512, NEON, etc.)
                │     GPU: vendor, VRAM, driver
                │     NPU: CoreML / NNAPI availability
                │
                ├─ 2. Score acceleration backends
                │     Each backend reports: supported? + estimated throughput
                │     Rank: NPU > GPU > CPU (with model-size awareness)
                │
                ├─ 3. Select model quantization
                │     VRAM/RAM budget → best quantization that fits
                │     e.g. 32GB VRAM → f16, 8GB → q5_0, 4GB → q4_0
                │
                └─ 4. Build engine
                      Load model with chosen backend + quantization
                      Return Box<dyn SttEngine>

engine.transcribe(&audio) ──► SttResult { text, language, duration_ms }
```

## Architecture

```
any-stt/
  crates/
    any-stt/                  # Core: trait, config, hardware detection, backend selection
      src/
        lib.rs                # pub trait SttEngine, initialize()
        config.rs             # SttConfig
        detect.rs             # Hardware detection (CPU features, GPU, NPU)
        selector.rs           # Backend scoring and selection logic

    whisper-backend/          # whisper.cpp unified backend (C FFI, not whisper-rs)
      src/
        lib.rs                # WhisperEngine: dispatches to acceleration
        ffi.rs                # Raw C bindings to whisper.h (cc/cmake build)
        accel/
          cpu.rs              # ggml CPU (AVX2, AVX-512, NEON, SVE)
          cuda.rs             # NVIDIA CUDA (Linux)
          metal.rs            # Apple Metal (macOS, iOS)
          vulkan.rs           # Vulkan (Linux desktop, Android opt-in)
          coreml.rs           # Apple CoreML / Neural Engine (macOS, iOS)
          nnapi.rs            # Android NNAPI / Hexagon NPU
      build.rs                # Compiles whisper.cpp from source via cc crate

  third-party/
    whisper.cpp/              # git submodule: m96-chan/whisper.cpp (fork)
```

## Supported Models

All Whisper-architecture models in ggml format. Auto-downloads via Hugging Face Hub.

### OpenAI Whisper

| Model          | Params | En WER | Multilingual | Notes                    |
|----------------|--------|--------|--------------|--------------------------|
| tiny           | 39M    | ~7.7%  | yes          | Fastest, lowest accuracy |
| tiny.en        | 39M    | ~6.4%  | no           |                          |
| base           | 74M    | ~5.7%  | yes          |                          |
| base.en        | 74M    | ~4.9%  | no           |                          |
| small          | 244M   | ~4.3%  | yes          | Good balance             |
| small.en       | 244M   | ~3.8%  | no           |                          |
| medium         | 769M   | ~3.5%  | yes          |                          |
| medium.en      | 769M   | ~3.2%  | no           |                          |
| large-v1       | 1550M  | ~2.9%  | yes          |                          |
| large-v2       | 1550M  | ~2.7%  | yes          |                          |
| large-v3       | 1550M  | ~2.5%  | yes          | Best accuracy            |
| large-v3-turbo | 809M   | ~2.6%  | yes          | large-v3 speed, ~medium size |

### Distil-Whisper (Distilled)

| Model              | Base       | Speedup | Notes                          |
|--------------------|------------|---------|--------------------------------|
| distil-large-v2    | large-v2   | ~6x     | English, near-original quality |
| distil-large-v3    | large-v3   | ~6x     | English                        |
| distil-medium.en   | medium.en  | ~4x     | English, compact               |
| distil-small.en    | small.en   | ~3x     | English, smallest distilled    |

### Japanese-Optimized

| Model              | Base       | Ja CER | Notes                         |
|--------------------|------------|--------|-------------------------------|
| kotoba-whisper-v1.0| large-v2   | ~5%    | Distilled for Japanese        |
| kotoba-whisper-v2.0| large-v3   | ~3%    | Best Japanese accuracy        |

### Quantization Formats (ggml)

| Format | Bits | Size vs f16 | Quality Loss | Use Case                      |
|--------|------|-------------|--------------|-------------------------------|
| f16    | 16   | 1.0x        | none         | Max quality, needs most VRAM  |
| q8_0   | 8    | ~0.5x       | negligible   | Quality-first with less RAM   |
| q5_1   | 5    | ~0.35x      | very small   | Balanced                      |
| q5_0   | 5    | ~0.33x      | small        | Default recommendation        |
| q4_1   | 4    | ~0.28x      | moderate     | Low memory                    |
| q4_0   | 4    | ~0.25x      | moderate     | Minimum memory                |

## Acceleration Backends

### Selection Priority

```
┌──────────────────────────────────────────────────────────┐
│ Platform   │  1st choice    │ 2nd choice  │ Fallback     │
├──────────────────────────────────────────────────────────┤
│ Linux      │ CUDA (NVIDIA)  │ Vulkan†     │ CPU AVX-512  │
│ macOS      │ CoreML (ANE)   │ Metal       │ CPU NEON     │
│ Android    │ NNAPI (NPU)    │ CPU NEON    │ Vulkan‡      │
│ iOS        │ CoreML (ANE)   │ Metal       │ CPU NEON     │
└──────────────────────────────────────────────────────────┘
† Vulkan selected only when shader cache is warm (see below)
‡ Android Vulkan: opt-in only — never auto-selected (see below)
```

> **Why is GPU off by default on Android?**
> Mobile apps typically use the GPU for 3D rendering. Vulkan STT inference would
> compete for the GPU and stall the render pipeline. NNAPI delegates to a dedicated
> NPU with zero GPU contention, making it the right default.
>
> If your app has no 3D rendering or you can afford GPU contention, explicitly set
> `backend: Some(Backend::Vulkan)` to opt in.

### Backend Details

**CUDA** — NVIDIA GPUs (Linux)
- whisper.cpp `GGML_CUDA` — offloads matmul to GPU
- Requires: CUDA toolkit, NVIDIA driver
- RTX 5090 (32GB): large-v3 f16 real-time factor <0.1x

**Metal** — Apple GPU (macOS, iOS)
- whisper.cpp `GGML_METAL` — GPU compute shaders
- M-series: unified memory, no copy overhead

**CoreML** — Apple Neural Engine (macOS, iOS)
- whisper.cpp `WHISPER_COREML` — encoder offloaded to ANE
- Lowest power, highest efficiency on Apple Silicon
- Requires .mlmodelc companion files

**Vulkan** — GPU acceleration (Linux desktop, Android opt-in) {#vulkan-caveats}
- whisper.cpp `GGML_VULKAN` — portable GPU acceleration
- Linux: NVIDIA, AMD, Intel Arc
- Android: available but **never auto-selected** — must be explicitly requested via `backend: Some(Backend::Vulkan)`
  - Faster than CPU when opted in — worthwhile if GPU headroom exists
  - GPU is typically occupied by 3D rendering; Vulkan STT will compete for GPU and may stall rendering
  - Use when your app has no 3D workload or GPU is otherwise idle
- **Shader compilation caveat** — first run compiles SPIR-V → device-specific shaders,
  causing multi-second startup delay. Mitigations:
  - `VkPipelineCache`: persist compiled shaders to disk, warm on subsequent launches
  - Ship pre-compiled pipeline cache per GPU vendor where possible
  - Selector falls back to CPU if shader cache is cold and `config.allow_cold_vulkan == false` (default)

**NNAPI** — Android Neural Networks API
- Delegates to on-device NPU (Hexagon, Samsung NPU, etc.)
- Best battery efficiency on Android

**CPU** — Universal fallback
- x86: AVX2 (most), AVX-512 (Zen 5, Intel server)
- ARM: NEON (all ARM64), SVE (server ARM)
- Always available, multi-threaded via ggml thread pool

## API

```rust
pub struct SttConfig {
    pub language: String,                // "ja", "en", etc.
    pub model: Model,                    // Model::LargeV3Turbo, Model::KotobaV2, etc.
    pub model_path: Option<PathBuf>,     // Override: custom model path
    pub quantization: Option<Quantization>, // Override: None = auto-select
    pub backend: Option<Backend>,        // Override: None = auto-select
    pub sample_rate: u32,                // default: 16000
    pub allow_cold_vulkan: bool,         // default: false — skip Vulkan if shader cache missing
}

pub enum Model {
    Tiny, TinyEn, Base, BaseEn,
    Small, SmallEn, Medium, MediumEn,
    LargeV1, LargeV2, LargeV3, LargeV3Turbo,
    DistilLargeV2, DistilLargeV3, DistilMediumEn, DistilSmallEn,
    KotobaV1, KotobaV2,
    Custom(String),  // HuggingFace model ID
}

pub enum Quantization { F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0 }

pub enum Backend { Cuda, Metal, CoreMl, Vulkan, Nnapi, Cpu }

pub struct HardwareInfo {
    pub cpu: CpuInfo,       // arch, features (AVX-512, NEON, etc.), cores
    pub gpu: Option<GpuInfo>, // vendor, name, vram_mb, driver
    pub npu: Option<NpuInfo>, // type (CoreML, NNAPI), available
    pub os: OsInfo,         // platform, version
    pub available_ram_mb: u64,
}

pub struct SttResult {
    pub text: String,
    pub language: String,
    pub duration_ms: f64,
    pub backend_used: Backend,
}

pub trait SttEngine: Send + Sync {
    fn transcribe(&self, audio: &[f32]) -> Result<SttResult, SttError>;
    fn is_ready(&self) -> bool;
    fn hardware_info(&self) -> &HardwareInfo;
    fn active_backend(&self) -> Backend;
}

/// Detects hardware, selects best backend + quantization, loads model.
pub fn initialize(config: SttConfig) -> Result<Box<dyn SttEngine>, SttError>;

/// Returns hardware info without loading a model.
pub fn detect_hardware() -> HardwareInfo;
```

## Usage

```rust
use any_stt::{SttConfig, SttEngine, Model};

// Auto-detect everything: best backend, best quantization for available hardware
let engine = any_stt::initialize(SttConfig {
    language: "ja".into(),
    model: Model::KotobaV2,
    ..Default::default()
})?;

println!("Backend: {:?}", engine.active_backend());  // e.g. Cuda
println!("Hardware: {:?}", engine.hardware_info());

let result = engine.transcribe(&audio_samples)?;
println!("{}", result.text);
```

```toml
[dependencies]
any-stt = { git = "https://github.com/m96-chan/any-stt" }

# Enable specific backends (default: cpu only)
[features]
default = ["cpu"]
cuda = ["any-stt/cuda"]
metal = ["any-stt/metal"]
coreml = ["any-stt/coreml"]
vulkan = ["any-stt/vulkan"]
nnapi = ["any-stt/nnapi"]
all-backends = ["cuda", "metal", "coreml", "vulkan", "nnapi"]
```

## Related
- [any-miotts](https://github.com/m96-chan/any-miotts) — TTS engine (same pattern)
- [whisper.cpp](https://github.com/m96-chan/whisper.cpp) — Maintained fork, included as git submodule in `third-party/`
- [kotoba-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) — Japanese ASR model
