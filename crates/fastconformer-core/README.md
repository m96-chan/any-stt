# fastconformer-core

Shared FastConformer ASR primitives used by both
`reazonspeech-backend` (Japanese, Longformer + RNN-T) and
`parakeet-backend` (English / European, rel-pos + TDT).

## What's in here

| Module | Purpose |
|--------|---------|
| `mel.rs`            | NeMo-compatible log-mel filterbank |
| `tokenizer.rs`      | SentencePiece detokenizer (decode-only) |
| `config.rs`         | GGUF metadata → typed `Config` |
| `encoder/ops.rs`    | Linear, LayerNorm, Swish, GLU, softmax, BN primitives |
| `encoder/subsample.rs` | Striding (3×Conv2d) + dw_striding (Conv2d + 2×DW+PW) |
| `encoder/attention.rs` | MHA with rel-pos + optional Longformer mode |
| `encoder/conv_module.rs` | Pointwise + depthwise Conv1d + GLU + BN |
| `encoder/ff.rs`     | Conformer macaron-half feed-forward |
| `encoder/block.rs`  | Full ConformerBlock |
| `encoder/mod.rs`    | `FastConformerEncoder` integration |

## Mel preprocessor — NeMo numerical equivalence

`log_mel_spectrogram` matches NeMo's `AudioToMelSpectrogramPreprocessor`
to within float32 precision. Validated stage-by-stage against
torchaudio (which NeMo wraps internally) using the
`scripts/validate-mel.py --dump-intermediates` reference dumps.

| Stage | `max_abs` vs torchaudio | Notes |
|-------|------------------------|-------|
| pre-emphasis (α=0.97)              | **0.0**       | `y[n] = x[n] - α·x[n-1]` |
| Hann window (periodic)             | **formula**   | `0.5·(1 − cos(2π·i/n))`, n=400 |
| frame extract (after reflect pad)  | **3.0e-8**    | `center=True`, `pad_mode="reflect"` |
| mel filterbank (slaney scale)      | **0.0**       | matches `melscale_fbanks(norm="slaney", mel_scale="slaney")` |
| power spectrum, frame 0            | **0.0**       | silent region, log(1e-5) |
| power spectrum, frame 100          | **9.5e-7**    | active region |
| log(mel + 1e-5) pre-normalize      | **2.9e-4**    | full sequence |
| post-normalize (per-feature)       | **8.9e-5**    | full sequence vs NeMo target |

Each layer has a dedicated `#[test]` in
`tests/layer_reference.rs` that auto-skips when the
`scripts/validate-mel.py --dump-intermediates` outputs aren't present
under `tests/fixtures/<model>/debug/` (the path is gitignored — fixtures
are developer-local).

### The bug that mattered most

Initial Rust mel had `max_abs = 3.16` vs NeMo. After fixing window-
inside-FFT placement (torch.stft pads `win_length`-long Hann to the
**center** of the `n_fft` buffer with leading + trailing zeros, not
to position 0 with trailing zeros), `max_abs` dropped to 8.9e-5 — bit-
equivalent for inference purposes.

Run the validation locally:

```sh
# one-time deps
python -m venv /tmp/anyvenv
/tmp/anyvenv/bin/pip install torch torchaudio soundfile numpy

# regenerate references
/tmp/anyvenv/bin/python scripts/validate-mel.py \
    --audio third-party/whisper.cpp/samples/japanese_test.wav \
    --out crates/fastconformer-core/tests/fixtures/reazonspeech-nemo-v2/mel_ja.npy \
    --dump-intermediates

# run all comparisons (the strict #[test]'s are NOT #[ignore]'d)
cargo test -p fastconformer-core --test layer_reference
# the post-normalize comparison runs only with --ignored:
cargo test -p fastconformer-core --test layer_reference \
    mel_matches_nemo_reference -- --ignored
```

## Encoder forward — known drift (not yet validated)

The encoder body (subsample → N × ConformerBlock → final LN) runs to
completion on the real `reazonspeech-nemo-v2` GGUF, but its outputs
do **not** numerically match NeMo. Symptom: greedy RNN-T decoding
produces gibberish (a few correct Japanese fragments mixed with long
runs of repeated tokens).

Status:
- mel input is now NeMo-equivalent (above)
- Conformer modules are structurally correct (zero-input passes through
  cleanly, lib unit tests cover identity / bias-only paths)
- Numerical drift must be in one or more of: dw_striding subsample,
  rel-pos attention math (Q+u/Q+v bias and rel-shift), conv module
  (BN + GLU axis), or the per-block residual scaling

This is the next #N9 milestone. Approach:
1. Build a minimal Python encoder that loads `model_weights.ckpt`
   directly (no nemo-toolkit, since it's incompatible with Python 3.14)
2. Dump per-layer outputs as `.npy`
3. Add stage-by-stage comparison tests in
   `tests/layer_reference.rs` (same pattern as the mel layer harness)
4. Locate and fix the first stage where `max_abs > 1e-3`

## Tokenizer

`SentencePieceTokenizer` parses the SentencePiece `.model` proto3 file
shipped alongside each `.nemo`. Decode-only (no encode); avoids the
1 MB C++ sentencepiece binding and the platform compatibility concerns
that come with it. Tested against synthetic models in
`tokenizer.rs::tests`.

## Encoder primitives

`encoder/ops.rs` provides scalar f32 implementations of the building
blocks that compose the Conformer block:

| Function | Purpose |
|----------|---------|
| `matvec_add`             | `out += W · x` (Linear without bias) |
| `linear`                 | `y = W · x + b` |
| `layer_norm` / `layer_norm_rows` | last-dim LN, optionally per-row |
| `swish_inplace`          | SiLU activation |
| `glu_channels`           | GLU split-and-gate |
| `softmax_inplace`        | numerically stable max-shifted softmax |
| `batch_norm_1d_inference` | inference-only BN with running stats |

Each has unit tests covering identity / zero / bias-only / overflow-
adjacent inputs. These are the building blocks for the full encoder
forward in `encoder/{block,conv_module,attention,subsample,ff}.rs`.
