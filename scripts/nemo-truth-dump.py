#!/usr/bin/env python3
"""Dump per-layer encoder intermediates from the REAL NeMo ConformerEncoder.

Run on Python 3.11 with `nemo-toolkit[asr]` installed (it doesn't yet
support 3.14). This is the ground truth for `dump-encoder-fixtures.py`'s
hand-rolled reference — if their outputs match, the hand-roll is correct
and we compare Rust against either. If they don't match, the hand-roll
is wrong and we fix it first.

Usage:
    /tmp/nemo_venv/bin/python scripts/nemo-truth-dump.py \\
        --nemo ~/asr-models/reazonspeech-nemo-v2/reazonspeech-nemo-v2.nemo \\
        --audio third-party/whisper.cpp/samples/japanese_test.wav \\
        --out-dir crates/fastconformer-core/tests/fixtures/reazonspeech-nemo-v2/encoder_truth
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Suppress NeMo telemetry chatter on stdout/stderr.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ONE_LOGGER_DISABLED", "true")

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nemo", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    print(f"loading {args.nemo} ...")
    model = EncDecRNNTBPEModel.restore_from(str(args.nemo), map_location="cpu")
    model.eval()
    # Disable dither so the dumps are deterministic — Rust's mel uses no
    # dither, so comparing against dither-on NeMo dumps adds spurious drift.
    model.preprocessor.featurizer.dither = 0.0
    print(f"  loaded {type(model).__name__}")

    # Run transcribe once BEFORE dumping. Empirically, the encoder
    # state-of-the-world after `model.transcribe()` produces different
    # (and correctly-decodable) outputs than a cold encoder forward —
    # this is the state Rust's greedy decoder should target. NeMo's
    # `EncDecRNNTBPEModel.transcribe()` runs its full encode+decode path,
    # which apparently primes the encoder. We don't have a smaller hook,
    # so we just call transcribe and discard its output.
    print(f"priming via transcribe({args.audio}) ...")
    hyps = model.transcribe([str(args.audio)], batch_size=1)
    print(f"  raw hyp: {hyps[0]!r}")
    if hasattr(hyps[0], "text"):
        print(f"  text: {hyps[0].text}")
    else:
        print(f"  text: {hyps[0]}")

    # Now extract the encoder and run forward with hooks to dump intermediates.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enc = model.encoder

    # Build mel input the same way NeMo does (via preprocessor).
    import soundfile as sf
    audio_np, sr = sf.read(str(args.audio), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"expected 16 kHz, got {sr}")
    print(f"  audio: {audio_np.shape}, sr={sr}")
    audio = torch.from_numpy(audio_np).unsqueeze(0)  # [1, N]
    audio_len = torch.tensor([audio.shape[1]])
    with torch.no_grad():
        proc_out, proc_len = model.preprocessor(input_signal=audio, length=audio_len)
    print(f"  preprocessor out: {tuple(proc_out.shape)}, len={proc_len.item()}")
    # proc_out: [B, n_mels, T]
    mel_arr = proc_out.squeeze(0).T.cpu().numpy().astype(np.float32)
    mel_arr = np.ascontiguousarray(mel_arr)
    np.save(args.out_dir / "mel.npy", mel_arr)

    # Hook every ConformerLayer's forward.
    dumps = {}
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            # ConformerLayer returns just a Tensor or (Tensor, pad_mask, ...) depending on version.
            if isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            dumps[name] = t.detach().clone()
        return hook

    # Pre-encoder (subsample) hook
    if hasattr(enc, "pre_encode"):
        handles.append(enc.pre_encode.register_forward_hook(make_hook("after_pre_encode")))
    # Per-layer hooks
    for i, layer in enumerate(enc.layers):
        handles.append(layer.register_forward_hook(make_hook(f"block_{i}_out")))

    print("running encoder ...")
    with torch.no_grad():
        enc_out, enc_len = enc(audio_signal=proc_out, length=proc_len)
    print(f"  encoder out: {tuple(enc_out.shape)}, len={enc_len.item()}")

    for h in handles:
        h.remove()

    # Save each dumped tensor.
    for name, tensor in dumps.items():
        # ConformerLayer returns [B, T, D] in NeMo's later versions, or [B, D, T] earlier.
        # We probe by shape: dim 1 == d_model (1024) is BTD; otherwise BDT.
        arr = tensor.cpu().numpy().astype(np.float32)
        # Drop batch dim
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        # If shape is (d_model, T) flip to (T, d_model)
        if arr.ndim == 2 and arr.shape[0] == 1024 and arr.shape[1] != 1024:
            arr = arr.T
        # If shape is (T, d_model, ...) just take it as-is
        arr = np.ascontiguousarray(arr)
        np.save(args.out_dir / f"{name}.npy", arr)
        print(f"  wrote {name}.npy shape={arr.shape}")

    # Encoder output: NeMo returns [B, D, T] — transpose for consistency with our convention.
    final = enc_out.squeeze(0).cpu().numpy().astype(np.float32)
    if final.shape[0] == 1024:
        final = final.T
    final = np.ascontiguousarray(final)
    np.save(args.out_dir / "encoder_output.npy", final)
    print(f"  wrote encoder_output.npy shape={final.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
