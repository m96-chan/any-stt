#!/usr/bin/env python3
"""Cross-check `fastconformer_core::log_mel_spectrogram` against the
NeMo `AudioToMelSpectrogramPreprocessor` reference implementation.

Independent of nemo-toolkit (which is heavy and python-3.14-incompatible
as of 2026-04). Reimplements the NeMo preprocessor with stock
torchaudio + numpy primitives matching NeMo's defaults
(sample_rate=16000, window_size=0.025, window_stride=0.010,
features=80, n_fft=512, preemph=0.97, log_zero_guard=1e-5,
normalize=per_feature, mag_power=2.0).

Usage:
    python scripts/validate-mel.py \\
        --audio third-party/whisper.cpp/samples/japanese_test.wav \\
        --out /tmp/mel_ref.npy

Then compare in Rust:
    cargo test -p fastconformer-core --test layer_reference \\
        mel_matches_nemo_reference -- --ignored
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output .npy path (shape [n_frames, n_mels])")
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win", type=int, default=400)
    ap.add_argument("--hop", type=int, default=160)
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--preemph", type=float, default=0.97)
    ap.add_argument("--log-eps", type=float, default=1e-5)
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        import torchaudio
    except ImportError as e:
        sys.exit(f"missing dep: {e.name} — pip install torch torchaudio numpy")

    if not args.audio.exists():
        sys.exit(f"not found: {args.audio}")

    # Load audio (WAV) via soundfile to avoid torchaudio's heavy backend.
    import soundfile as sf
    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio.copy())
    if sr != args.sr:
        waveform = torchaudio.functional.resample(waveform, sr, args.sr)
    print(f"loaded {waveform.numel()} samples = {waveform.numel()/args.sr:.2f}s")

    # ---- NeMo-style preprocessing ----

    # 1. Pre-emphasis (matches NeMo when preemph != 0).
    pre = torch.cat([waveform[:1], waveform[1:] - args.preemph * waveform[:-1]])

    # 2. STFT. NeMo uses torch.stft with a Hann window of `win` samples,
    #    n_fft chosen as next pow2 of win (here 400 → 512), hop `hop`,
    #    center=True (default), pad_mode='reflect' (NeMo default).
    window = torch.hann_window(args.win, periodic=True)  # NeMo uses periodic
    spec = torch.stft(
        pre,
        n_fft=args.n_fft,
        hop_length=args.hop,
        win_length=args.win,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )  # [n_freq=257, n_frames]

    # 3. Power spectrum |X|².
    power = spec.abs() ** 2  # [n_freq, n_frames]

    # 4. Mel filterbank. NeMo uses slaney scale, slaney area normalization.
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=args.n_fft // 2 + 1,
        f_min=0.0,
        f_max=args.sr / 2,
        n_mels=args.n_mels,
        sample_rate=args.sr,
        norm="slaney",
        mel_scale="slaney",
    )  # [n_freq, n_mels]

    mel = mel_fb.T @ power  # [n_mels, n_frames]

    # 5. Log compression with NeMo's zero guard.
    log_mel = torch.log(mel + args.log_eps)

    # 6. Per-feature normalization across time (NeMo `normalize=per_feature`).
    mean = log_mel.mean(dim=1, keepdim=True)
    std = log_mel.std(dim=1, keepdim=True, unbiased=False) + args.log_eps
    log_mel = (log_mel - mean) / std

    # Transpose to [n_frames, n_mels] for the Rust convention.
    # `.contiguous()` so numpy stores in C-order (the npy reader on the
    # Rust side rejects fortran_order=True).
    out = log_mel.T.contiguous().numpy().astype(np.float32, copy=False)
    out = np.ascontiguousarray(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"wrote {args.out} shape={out.shape} dtype={out.dtype}")

    # Quick sanity print.
    print(f"  range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"  per-feature mean (first 5): {out.mean(axis=0)[:5]}")
    print(f"  per-feature std  (first 5): {out.std(axis=0)[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
