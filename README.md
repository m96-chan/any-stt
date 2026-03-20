# any-stt

Cross-platform Speech-to-Text engine for Rust.
Detects hardware (CPU/GPU/NPU) and OS at runtime, then selects the optimal acceleration backend automatically.

## Benchmark Results

Measured on real devices. All times are median of 5 runs.
whisper.cpp CLI (beam_size=5) shown as reference baseline.

### tiny.en (77 MB) — JFK inaugural 11.0s

| Platform | Backend | Median | RTF | vs whisper.cpp CLI |
|----------|---------|--------|-----|--------------------|
| whisper.cpp CLI | CPU 16T (reference) | 259ms | 0.024 | 1.0x |
| Linux (RTX 5090) | **CUDA** | **23ms** | 0.002 | **11x faster** |
| Linux (RTX 5090) | Vulkan | 23ms | 0.002 | 11x faster |
| Linux (Ryzen 9950X3D) | CPU AVX-512 | 169ms | 0.015 | 1.5x faster |
| Android (SD 8 Gen 3) | **NPU INT8 + CPU** | **162ms** | 0.015 | 1.6x faster |
| Android (SD 8 Gen 3) | CPU NEON 8T | 851ms | 0.077 | 3.3x slower |

### Large-v2 / Kotoba (3.1 GB) — Japanese 7.4s

| Platform | Backend | Median | RTF | vs whisper.cpp CLI |
|----------|---------|--------|-----|--------------------|
| whisper.cpp CLI | CPU 16T (reference) | 7517ms | 1.020 | 1.0x |
| Linux (RTX 5090) | **CUDA** | **99ms** | 0.013 | **76x faster** |
| Linux (RTX 5090) | Vulkan | 106ms | 0.014 | 71x faster |
| Linux (Ryzen 9950X3D) | CPU AVX-512 | 6836ms | 0.928 | 1.1x faster |
| Android (SD 8 Gen 3) | **NPU INT8 + CPU** | **5.4s** | **0.73** | **1.4x faster** |
| Android (SD 8 Gen 3) | CPU NEON 8T | 66.6s | 9.05 | 8.9x slower |

> whisper.cpp CLI: `whisper-cli -t 16` (beam_size=5, best_of=5). any-stt: greedy decoding.
> Output: "我輩は猫である。名前はまだない。どこで生まれたかとんと見当がつかぬ。"
> Output MATCH verified between CPU and hybrid (NPU+CPU) paths.

## Target Platforms

| Platform | Arch | Acceleration | Status |
|----------|------|-------------|--------|
| **Linux** | x86_64 | CUDA, Vulkan, CPU (AVX2/AVX-512) | ✅ Tested on RTX 5090 + Ryzen 9950X3D |
| **Android** | ARM64 | QNN NPU (Hexagon HTP), CPU (NEON) | ✅ Tested on REDMAGIC 9 Pro (SD 8 Gen 3) |
| **iOS** | ARM64 | CoreML (ANE), Metal, CPU (NEON) | ✅ Implemented, pending device test |
| **macOS** | ARM64 | CoreML (ANE), Metal, CPU (NEON) | ✅ Implemented, pending device test |

## Architecture

```
Audio PCM f32
    ↓
any-stt::initialize(config)
    ├── 1. Detect hardware (CPU/GPU/NPU/RAM)
    ├── 2. Select backend (NPU > GPU > CPU)
    ├── 3. Select quantization (fits available memory)
    └── 4. Build engine → Box<dyn SttEngine>
            ↓
engine.transcribe(&audio) → SttResult { text, language, duration_ms }
```

### Heterogeneous CPU+NPU Pipeline (Android/Snapdragon)

```
Audio PCM
    ↓
┌── Preprocessor (CPU) ────────────────┐
│  mel spectrogram → Conv1d → GELU     │  whisper.cpp handles mel+conv
│  → Conv1d → GELU → + pos_embed      │
└──────────────────────────────────────┘
    ↓
┌── Encoder (NPU via QNN HTP) ─────────┐
│  4-32 transformer blocks (MatMul)    │  INT8 on Hexagon HMX
│  5.7x speedup over FP32             │
└──────────────────────────────────────┘
    ↓
┌── Decoder (CPU) ─────────────────────┐
│  whisper.cpp autoregressive decoder  │  skip_encode mode
│  with injected encoder output        │
└──────────────────────────────────────┘
    ↓
SttResult { text, language, duration_ms }
```

### Crate Structure

```
any-stt/
  crates/
    any-stt/            # Core: SttEngine trait, config, hardware detection, backend selection
    whisper-backend/    # whisper.cpp FFI, WhisperEngine, WhisperQnnEngine (hybrid)
    qnn-backend/        # Qualcomm QNN HTP: dlopen loader, graph builder, encoder, ops
    gguf-loader/        # GGUF v3 parser (memmap2 zero-copy, F32/F16/Q8_0/Q4_0/Q5_0)
    bench/              # Cross-platform benchmark tool

  third-party/
    whisper.cpp/        # Fork with encoder output injection API (skip_encode, get/set_encoder_output)

  scripts/
    bench-device.sh     # Android adb deploy + bench
    build-ios.sh        # iOS cross-compilation
    convert-to-gguf.py  # OpenAI whisper → GGUF v3
    convert-kotoba-to-ggml.py  # Kotoba HuggingFace → ggml
    dump_encoder_weights.py    # Weight dump for NPU testing
```

## Backend Selection

```
┌──────────────────────────────────────────────────────────────┐
│ Platform   │ 1st choice       │ 2nd choice   │ Fallback     │
├──────────────────────────────────────────────────────────────┤
│ Linux      │ CUDA (NVIDIA)    │ Vulkan (AMD) │ CPU AVX-512  │
│ macOS      │ CoreML (ANE)     │ Metal        │ CPU NEON     │
│ Android    │ QNN HTP (NPU)    │ CPU NEON     │ —            │
│ iOS        │ CoreML (ANE)     │ Metal        │ CPU NEON     │
└──────────────────────────────────────────────────────────────┘
```

GPU auto-detection on Linux:
- **NVIDIA**: nvidia-smi → CUDA auto-selected
- **AMD**: sysfs `/sys/class/drm` vendor `0x1002` → Vulkan (with `allow_cold_vulkan: true`)
- **Intel Arc**: sysfs vendor `0x8086` → Vulkan

## Quick Start

### Linux (NVIDIA CUDA)

```bash
# Build with CUDA
CUDA_HOME=/usr/local/cuda cargo build --release -p bench --features cuda

# Benchmark
cargo run -p bench --release --features cuda -- \
  --model models/ggml-tiny.en.bin \
  --audio samples/jfk.wav \
  --backend gpu --runs 5
```

### Linux (CPU only)

```bash
cargo build --release -p bench
cargo run -p bench --release -- \
  --model models/ggml-tiny.en.bin \
  --audio samples/jfk.wav \
  --backend cpu --runs 5
```

### Android (Snapdragon + QNN NPU)

```bash
# Cross-compile and deploy via adb
./scripts/bench-device.sh -t 1 --backend all --runs 5

# Requires:
#   ANDROID_NDK_HOME set
#   QNN SDK libs on device (/data/local/tmp/qnn/)
#   adb connected
```

### iOS (Metal + CoreML)

```bash
# On macOS with Xcode
./scripts/build-ios.sh          # Metal only
./scripts/build-ios.sh --coreml # Metal + CoreML
```

### Japanese (Large-v2)

```bash
# Download large-v2 model
cd third-party/whisper.cpp && bash models/download-ggml-model.sh large-v2

# Benchmark with Japanese audio
cargo run -p bench --release --features cuda -- \
  --model models/ggml-large-v2.bin \
  --audio samples/japanese_test.wav \
  --lang ja --runs 3
```

## API

```rust
use whisper_backend::initialize;
use any_stt::{SttConfig, SttEngine, Model};

// Auto-detect: best backend + quantization for available hardware
let config = SttConfig {
    language: "ja".into(),
    model: Model::LargeV2,
    model_path: Some("models/ggml-large-v2.bin".into()),
    ..Default::default()
};

let engine = initialize(&config)?;
// → "initialize: using QNN NPU backend" (on Snapdragon)
// → "initialize: using CUDA backend" (on NVIDIA Linux)
// → "initialize: using CPU backend" (fallback)

let result = engine.transcribe(&audio_f32)?;
println!("{}", result.text);
// → "我輩は猫である。名前はまだない。どこで生まれたかとんと見当がつかぬ。"
```

### Error Handling

```rust
// NPU failure → transparent CPU fallback (no error)
// Model not found → SttError::ModelNotFound
// Empty audio → SttError::InvalidAudio
// All errors are non-panic, returned as Result
```

### Feature Flags

```toml
[dependencies]
whisper-backend = { path = "crates/whisper-backend" }

# Enable acceleration backends
# whisper-backend features: cuda, vulkan, metal, coreml
```

## Supported Models

All Whisper-architecture models in ggml/GGUF format.

| Model | Params | Size (F16) | Quality | Notes |
|-------|--------|-----------|---------|-------|
| tiny.en | 39M | 77 MB | Good (English) | Fastest |
| small | 244M | 500 MB | Better | Good balance |
| large-v2 | 1550M | 3.1 GB | Best | Kotoba base |
| large-v3-turbo | 809M | 1.6 GB | Near-best | Speed+quality |
| kotoba-v2.0 | 1550M | 1.4 GB (F16) | Best Japanese | Distilled decoder |

### GGUF Conversion

```bash
# OpenAI whisper → GGUF v3
python3 scripts/convert-to-gguf.py tiny.en output.gguf

# Kotoba (HuggingFace) → ggml
python3 scripts/convert-kotoba-to-ggml.py
```

## Tests

85 tests across all crates:

```
any-stt:         25 (detection, selection, iOS/macOS/Android/Linux)
whisper-backend: 29 (FFI, engine, hybrid, error handling, initialize)
qnn-backend:      6 (dlopen, MatMul, probe)
gguf-loader:      3 (parser, F16)
layer-reference: 20 (per-layer Python fixture comparison)
bench:            1 (doctest)
transcribe:       1 (E2E JFK)
```

```bash
cargo test --workspace
```

## Issues

| # | Platform | Status |
|---|----------|--------|
| [#4](https://github.com/m96-chan/any-stt/issues/4) | Android ARM64: CPU + QNN NPU | ✅ RTF 0.73 (Large-v2) |
| [#5](https://github.com/m96-chan/any-stt/issues/5) | iOS ARM64: Metal + CoreML + CPU | ✅ Implemented |
| [#6](https://github.com/m96-chan/any-stt/issues/6) | Linux x86_64: CUDA + Vulkan + CPU | ✅ RTF 0.013 (Large-v2) |
| [#7](https://github.com/m96-chan/any-stt/issues/7) | macOS ARM64: Metal + CoreML + CPU | ✅ Implemented |
| [#8](https://github.com/m96-chan/any-stt/issues/8) | Linux: Intel NPU + AMD XDNA | 📋 Planned |

## Related

- [any-miotts](https://github.com/m96-chan/any-miotts) — TTS engine (same pattern)
- [whisper.cpp](https://github.com/m96-chan/whisper.cpp) — Fork with encoder output injection API
- [kotoba-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) — Japanese ASR model
