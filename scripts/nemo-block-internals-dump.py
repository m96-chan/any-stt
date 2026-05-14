#!/usr/bin/env python3
"""Dump per-sub-module intermediates from block 0 of NeMo's Conformer encoder.

Uses `register_forward_pre_hook` and `register_forward_hook` on sub-modules
inside ConformerLayer (norm_feed_forward1, feed_forward1, self_attn, etc.)
to capture inputs/outputs WITHOUT replacing the layer's forward (which
proved error-prone — the previous hand-rolled patched_forward diverged
from NeMo by ~14 max_abs).

Captures match the names of stages we care about for Rust validation:
  - after_ff1, after_attn, after_conv, after_ff2, block_out
  - pre_attn (= norm_self_att output), pre_conv (= norm_conv output)
  - attn_out, conv_out (the raw sub-module outputs, pre-residual)
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ONE_LOGGER_DISABLED", "true")

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from nemo.collections.asr.models import EncDecRNNTBPEModel


def to_TD(t: torch.Tensor) -> np.ndarray:
    a = t.detach().squeeze(0).cpu().numpy().astype(np.float32)
    if a.ndim == 2 and a.shape[0] == 1024 and a.shape[1] != 1024:
        a = a.T
    return np.ascontiguousarray(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layer-idx", type=int, default=0)
    args = ap.parse_args()

    model = EncDecRNNTBPEModel.restore_from(str(args.nemo), map_location="cpu")
    model.eval()
    model.preprocessor.featurizer.dither = 0.0

    audio_np, sr = sf.read(str(args.audio), dtype="float32")
    audio = torch.from_numpy(audio_np).unsqueeze(0)
    audio_len = torch.tensor([audio.shape[1]])
    with torch.no_grad():
        proc_out, proc_len = model.preprocessor(input_signal=audio, length=audio_len)

    enc = model.encoder
    layer = enc.layers[args.layer_idx]

    captures = {}
    handles = []

    # `pre_attn` = input to self_attn = output of norm_self_att.
    handles.append(layer.norm_self_att.register_forward_hook(
        lambda m, i, o: captures.__setitem__("pre_attn", o.detach().clone())
    ))
    # `attn_out` = output of self_attn (pre-residual).
    handles.append(layer.self_attn.register_forward_hook(
        lambda m, i, o: captures.__setitem__(
            "attn_out", (o[0] if isinstance(o, tuple) else o).detach().clone()
        )
    ))
    # `pre_conv` = input to conv = output of norm_conv.
    handles.append(layer.norm_conv.register_forward_hook(
        lambda m, i, o: captures.__setitem__("pre_conv", o.detach().clone())
    ))
    # `conv_out` = output of conv (pre-residual).
    handles.append(layer.conv.register_forward_hook(
        lambda m, i, o: captures.__setitem__(
            "conv_out", (o[0] if isinstance(o, tuple) else o).detach().clone()
        )
    ))
    # `after_ff1`, `after_attn`, `after_conv`, `after_ff2` need rolling
    # residual tracking. The cleanest way: hook each *norm* with a pre-hook
    # to capture the residual entering that stage. Then after_X = input_to_next_norm.
    handles.append(layer.norm_feed_forward1.register_forward_pre_hook(
        lambda m, i: captures.__setitem__("residual_before_ff1", i[0].detach().clone())
    ))
    # after_ff1 == input to norm_self_att (residual fed to attn branch)
    handles.append(layer.norm_self_att.register_forward_pre_hook(
        lambda m, i: captures.__setitem__("after_ff1", i[0].detach().clone())
    ))
    # after_attn == input to norm_conv
    handles.append(layer.norm_conv.register_forward_pre_hook(
        lambda m, i: captures.__setitem__("after_attn", i[0].detach().clone())
    ))
    # after_conv == input to norm_feed_forward2
    handles.append(layer.norm_feed_forward2.register_forward_pre_hook(
        lambda m, i: captures.__setitem__("after_conv", i[0].detach().clone())
    ))
    # after_ff2 == input to norm_out
    handles.append(layer.norm_out.register_forward_pre_hook(
        lambda m, i: captures.__setitem__("after_ff2", i[0].detach().clone())
    ))
    # block_out == output of norm_out (the whole layer's output)
    handles.append(layer.norm_out.register_forward_hook(
        lambda m, i, o: captures.__setitem__("block_out", o.detach().clone())
    ))

    with torch.no_grad():
        enc_out, _ = enc(audio_signal=proc_out, length=proc_len)

    for h in handles:
        h.remove()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in captures.items():
        a = to_TD(v)
        np.save(args.out_dir / f"block{args.layer_idx}_{k}.npy", a)
        print(f"wrote block{args.layer_idx}_{k}.npy shape={a.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
