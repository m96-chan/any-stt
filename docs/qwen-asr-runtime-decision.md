# Qwen3-ASR — Rust runtime decision (#N12)

**Date:** 2026-04-28
**Status:** Decision recorded. Implementation tracked in #N6.

## TL;DR

**Adopt `llama-cpp-2` (utilityai/llama-cpp-rs) with the `mtmd`, `vulkan`,
and `android-shared-stdcxx` features.** Upstream llama.cpp gained
Qwen3-ASR support on 2026-04-12 (PR #19441), official GGUFs are
published, and the Rust binding is healthy and recently released
(v0.1.145, 2026-04-22). Mobile (Android NDK + Vulkan) is documented
and matches our existing whisper.cpp integration pattern.

## Architecture summary

Qwen3-ASR-1.7B is a **dual-module** model:

- **Audio encoder ("AuT", ~300M params)**
  - Input: 16 kHz mono audio → 128-bin log-mel (Whisper-compatible mel
    basis: 400 FFT, 160 hop)
  - Conv2d stem (3 layers, stride 2 each) → 8× time downsample,
    12.5 Hz token rate
  - Sinusoidal positional embedding
  - 24-layer transformer encoder, hidden 1024, hybrid
    dense / segmented (windowed) attention
  - LayerNorm → GELU → MLP projector → Qwen3 token-embedding space
- **LLM decoder: Qwen3-1.7B**
  - RoPE, GQA, SwiGLU, RMSNorm (Qwen3 family standard)
  - Audio frames injected between `<|audio_start|>` / `<|audio_end|>`
  - Autoregressive decode until EOS
- **License:** Apache-2.0
- **Total:** ~2B params (1.7B LLM + 0.3B encoder + projector)

## Runtime candidates

### llama-cpp-2 (utilityai/llama-cpp-rs) — ✅ viable, RECOMMENDED

| Aspect | Status |
|--------|--------|
| Latest release | v0.1.145 (2026-04-22), multi-release-per-month |
| Maintainer health | utilityai org, multiple contributors — not abandoned |
| Qwen3-ASR upstream | **YES — merged 2026-04-12 (PR #19441)** |
| Official GGUFs | `ggml-org/Qwen3-ASR-{0.6B,1.7B}-GGUF` (Q8_0 = 2.17 GB, BF16 = 4.07 GB) |
| Rust audio mtmd | C API exposed via `llama-cpp-sys-2`; safe wrapper for audio bitmap may need an upstream PR (vision is demonstrated; audio path is the same shape) |
| Android NDK | First-class via `android-shared-stdcxx` / `android-static-stdcxx` features |
| Vulkan | First-class for Android Adreno + desktop |
| QNN/Hexagon NPU | **NOT yet merged in llama.cpp upstream** — community fork exists, but for our needs we route AuT MatMuls through our own `qnn-backend` and inject embeddings (same pattern as the existing NNAPI Whisper E2E path) |
| Known caveats | (1) PR #21847 long-audio empty-output bug for clips > ~2 min; (2) audio is "highly experimental" upstream; (3) 30-second flat chunking vs reference's windowed 1-8 s |

### mistral.rs — ⚠️ partial, NOT chosen

- Qwen3 LLM: full support (Qwen3, Qwen3-VL, Qwen3.5, Qwen3-Next).
- Qwen3-ASR audio encoder: **not supported**. ASR support is Voxtral-only.
- Android: **no documented target.** No Vulkan, no QNN. CUDA / Metal / CPU only.
- Adopting would require porting AuT into mistral.rs's model registry +
  adding aarch64-linux-android target + losing all NPU options.
- Maintainer concentration: heavily Eric Buehler-driven. Flag for the
  "single-maintainer risk" memo per `feedback_dep_policy`.

### Direct ggml FFI (no llama.cpp) — ❌ blocked

- All Rust ggml binding crates are dead-or-minimal:
  - `ggml` (lib.rs flag: "minimal maintenance")
  - `rustformers/llm` — unmaintained
  - `rusty-ggml` — pre-alpha
  - `ggml-rs` (michaelgiba) — WIP
- Per `feedback_dep_policy` (reject inactive maintainers), all available
  crates are out.
- We could roll our own ggml-sys (precedent: `whisper-backend` already
  does raw FFI through cmake/cc + libloading without an upstream crate).
  Then port: Qwen3 graph (RoPE, GQA, RMSNorm, SwiGLU, KV cache,
  sampling) + AuT encoder (Conv2d, attention with windowed mask,
  projector) + tokenizer + chat template + chunking + K-quants
  (Q4_K/Q5_K/Q6_K — currently unsupported in our `gguf-loader`).
- LoC estimate: ~5–8k Rust + sys bindings. Reinvents the upstream work
  that just landed. Bad ROI.

### antirez/qwen-asr (pure-C reference) — reference-only

- Pure C, MIT, ~6 commits, personal project by Salvatore Sanfilippo.
- Implements full AuT + Qwen3 decode in a few thousand LoC with BLAS.
- 13.38× RTF on M3 Max for 0.6B / 7.63× for 1.7B — proves the model
  is fast at FP32 with no llama.cpp tricks.
- Use as **layer-output ground-truth** for cross-validating any port,
  not as a runtime.

### Other (Burn, Candle, rust-bert, ONNX, MLX, CoreML)

- **Candle**: explicitly excluded by user direction.
- **Burn**: no Qwen3-ASR port today; mobile/NPU story unproven.
- **rust-bert**: not designed for ASR / multimodal LLM.
- **ONNX runtime**: excluded by `project_backend_decisions.md`.
- **MLX / CoreML / Swift**: Apple-only; not Rust on Android.

## Decision rationale

1. **Upstream support is current.** Qwen3-ASR landed in llama.cpp
   2026-04-12; official GGUFs exist; bug list is community-tracked
   and actively iterated. We don't reinvent encoder graphs or chunking.
2. **Rust binding is healthy.** `llama-cpp-2` v0.1.145 ships 2026-04-22.
   Multi-contributor org. Not on the rejection list.
3. **Mobile is real, not theoretical.** llama.cpp is the only candidate
   with documented Android NDK builds + Vulkan on Adreno + a path
   (community fork) to QNN/Hexagon. mistral.rs lacks all three.
4. **Matches existing project shape.** `whisper-backend` already uses
   raw cmake/cc + libloading FFI to vendored whisper.cpp. Replicating
   that for `qwen-asr-backend` over llama.cpp is a known idiom in this
   codebase.
5. **NPU offload is naturally separable.** The AuT encoder is a
   separable subgraph (Conv2d stem + transformer + projector, all
   fixed-shape FP16-friendly MatMuls — exactly what HTP eats well). We
   build a QNN graph offline via our `qnn-backend` (same pattern as the
   existing FastConformer / Whisper-encoder path), feed mel chunks
   through QNN, and **inject** the resulting embeddings into the
   llama.cpp LLM context. LLM stays on CPU / Vulkan via `llama-cpp-2`.
6. **"Reject inactive deps" policy clears it.** No abandoned
   transitive dependencies in this path.

## Trade-offs accepted

- **Upstream audio is officially "highly experimental"** — quality may
  trail the Python reference. Mitigation: keep `antirez/qwen-asr`
  checked out as a reference for layer-output cross-validation.
- **The mtmd Rust API for audio may need an upstream PR.** Vision is
  demonstrated end-to-end; audio bitmap helpers are exposed via sys but
  the safe Rust wrappers are vision-flavored. Budget 1–2 weeks for
  building (and possibly upstreaming) the audio path.
- **`llama-cpp-sys-2` build on Android NDK has rough edges** (OpenMP
  must be off, periodic Direct-IO issues). Same class of problems we
  already navigate with whisper.cpp. Solvable.
- **PR #21847 long-audio bug** is open as of 2026-04. Clips > 2 min
  may produce empty output until upstream fixes ship.

## Implementation plan for #N6 (rough)

| # | Subtask | Effort |
|---|---------|--------|
| 1 | Add `llama-cpp-2` dep + audit Android build (`mtmd`, `vulkan`, `android-shared-stdcxx`) | 0.5 day |
| 2 | Decide vendor strategy: pin `llama-cpp-sys-2` rev with Qwen3-ASR support, or vendor llama.cpp into `third-party/` mirroring `third-party/whisper.cpp` | 0.5 day |
| 3 | Tokenizer integration via `LlamaModel::chat_template`; verify `<|audio_start|>` / `<|audio_end|>` roundtrip | 0.5 day |
| 4 | Audio preprocessor (16 kHz resample, 128-bin log-mel, 30-s chunking) — share DSP path with `whisper-backend` where possible | 1 day |
| 5 | mtmd audio Rust wrapper (possibly an upstream PR to `utilityai/llama-cpp-rs`) | 2–4 days |
| 6 | Autoregressive decode loop via `llama-cpp-2`'s sampler stack; greedy first | 1–2 days |
| 7 | Backend selection wiring (CPU + Vulkan); QNN encoder offload deferred | 1 day |
| 8 | NPU offload of AuT encoder (separable subgraph through `qnn-backend`, embeddings injected into llama.cpp context) | future milestone, ~1–2 weeks |
| 9 | Tests + bench (cross-check against `antirez/qwen-asr` outputs; long-audio regression once upstream fixes #21847) | 1 day |

**Time to first working CPU/Vulkan path:** ~5–7 working days.
**Time to NPU-accelerated AuT encoder:** ~+2 weeks.

## Risk flags

- PR #21847 long-audio bug — patch or chunk client-side until upstream fixes.
- mtmd Rust API for audio is currently vision-shaped — likely first
  upstream PR contributor.
- `mistral.rs` single-maintainer concentration is logged in the dep
  policy memo as a future-watch even though we're not picking it.

## Sources

- [Qwen/Qwen3-ASR-1.7B model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR Technical Report (arxiv 2601.21337)](https://arxiv.org/html/2601.21337v1)
- [llama.cpp PR #19441 — mtmd qwen3 audio support](https://github.com/ggml-org/llama.cpp/pull/19441)
- [llama.cpp issue #21847 — long-audio empty output bug](https://github.com/ggml-org/llama.cpp/issues/21847)
- [llama.cpp multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [llama.cpp Android build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
- [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- [crates.io llama-cpp-2](https://crates.io/crates/llama-cpp-2)
- [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [mistral.rs Voxtral docs](https://ericlbuehler.github.io/mistral.rs/VOXTRAL.html)
- [antirez/qwen-asr — pure-C reference impl](https://github.com/antirez/qwen-asr)
- [andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx)
- [ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF)
- [llama.cpp Hexagon backend issue #18139](https://github.com/ggml-org/llama.cpp/issues/18139)
