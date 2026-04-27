#!/usr/bin/env python3
"""Dump per-layer tensor fixtures from a NeMo FastConformer ASR model.

Used to build numerical reference data for the Rust unit tests in
`crates/fastconformer-core/tests/` and `crates/reazonspeech-backend/
tests/`. Run this once per model version; commit the resulting `.npy`
files (or stash them in the runner's local cache).

Outputs (under `<out_dir>/<model_id>/`):

    mel_<sample_id>.npy           - log-mel features [n_frames, n_mels]
    enc_subsample_<sample_id>.npy - after striding conv2d ×8
    enc_block_<L>_<sample_id>.npy - block L output (one per encoder layer)
    enc_post_ln_<sample_id>.npy   - final encoder output
    pred_state_<token>.npy        - LSTM hidden after embed(token) (one per
                                   each token in PRED_TEST_TOKENS)
    joint_<sample_id>.npy         - joint logits for first encoder frame
                                   with zero pred_state
    fixture_meta.json             - dimensions + sample IDs for the Rust
                                   side to discover what's available

Requirements:
    pip install nemo-toolkit==2.* numpy soundfile

Typical use:
    huggingface-cli download reazon-research/reazonspeech-nemo-v2 \\
        --local-dir ./reazonspeech-nemo-v2

    python scripts/dump-nemo-fixtures.py \\
        --model-path ./reazonspeech-nemo-v2/model.nemo \\
        --model-id reazonspeech-nemo-v2 \\
        --audio third-party/whisper.cpp/samples/japanese_test.wav \\
        --out-dir crates/fastconformer-core/tests/fixtures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Tokens to seed the prediction network with — small fixed set so the
# Rust LSTM tests can verify exact equality. Picks span low / mid / high
# vocab indices to flush out shape / order bugs.
PRED_TEST_TOKENS = [0, 1, 100, 1500, 2999]


def _require_deps() -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # noqa: WPS433
    except ImportError:
        sys.exit("missing dep: numpy — pip install numpy")
    try:
        import torch  # noqa: WPS433
    except ImportError:
        sys.exit("missing dep: torch — pip install torch")
    try:
        import soundfile as sf  # noqa: WPS433
    except ImportError:
        sys.exit("missing dep: soundfile — pip install soundfile")
    return np, torch, sf


def _require_nemo():
    try:
        import nemo.collections.asr as nemo_asr  # noqa: WPS433
        return nemo_asr
    except ImportError:
        sys.exit(
            "missing dep: nemo-toolkit — "
            "pip install nemo-toolkit==2.* "
            "(this is a heavy install — torch + cuda libs)"
        )


def _save_npy(path: Path, arr) -> None:
    """Save an ndarray as f32 to disk."""
    import numpy as np
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def dump_mel(np, sf, model: Any, audio_path: Path, sample_id: str, out_dir: Path) -> tuple:
    """Run NeMo's preprocessor on the audio and return (mel_array, sample_rate)."""
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        sys.exit(f"audio must be 16 kHz, got {sr}")

    import torch
    audio_t = torch.tensor(audio).unsqueeze(0)  # [1, T]
    audio_len = torch.tensor([audio.shape[0]])

    # NeMo's preprocessor lives at model.preprocessor; output is
    # [batch, mel, time].
    mel, mel_len = model.preprocessor(input_signal=audio_t, length=audio_len)
    # Reshape to [time, mel] for the Rust convention.
    mel_np = mel.squeeze(0).transpose(0, 1).cpu().numpy()

    out_path = out_dir / f"mel_{sample_id}.npy"
    _save_npy(out_path, mel_np)
    print(f"  mel:        {out_path} shape={mel_np.shape}")

    return mel_np, sr


def dump_encoder_layers(
    np,
    torch,
    model: Any,
    mel_np,
    sample_id: str,
    out_dir: Path,
) -> dict:
    """Hook every Conformer block + final norm and dump intermediate tensors."""
    encoder = model.encoder

    # NeMo Conformer encoder has `.layers` (ModuleList of ConformerLayer)
    # plus a pre-encoder (subsampling) and a final norm.
    layer_outputs: list = []
    hooks = []

    def _make_hook(name: str, store: list):
        def hook(_module, _inp, output):
            # Conformer layers return a tuple (out, mask) or just the
            # tensor depending on NeMo version.
            tensor = output[0] if isinstance(output, tuple) else output
            store.append((name, tensor.detach().cpu()))
        return hook

    for i, layer in enumerate(encoder.layers):
        h = layer.register_forward_hook(_make_hook(f"enc_block_{i}", layer_outputs))
        hooks.append(h)

    # Run encoder.
    mel_t = torch.tensor(mel_np).transpose(0, 1).unsqueeze(0)  # [1, mel, time]
    length = torch.tensor([mel_np.shape[0]])
    enc_out, enc_len = encoder(audio_signal=mel_t, length=length)

    for h in hooks:
        h.remove()

    written: dict = {}

    # Subsampling output is the input to the first block — capture from
    # the saved sequence by indexing zero before any block hook fired.
    # We use a separate hook on the pre_encode submodule instead.
    pre_outs: list = []
    pre_hook = encoder.pre_encode.register_forward_hook(_make_hook("enc_subsample", pre_outs))
    encoder(audio_signal=mel_t, length=length)
    pre_hook.remove()
    if pre_outs:
        _, pre_t = pre_outs[-1]
        pre_np = pre_t.squeeze(0).cpu().numpy()
        path = out_dir / f"enc_subsample_{sample_id}.npy"
        _save_npy(path, pre_np)
        written["enc_subsample"] = list(pre_np.shape)
        print(f"  subsample:  {path} shape={pre_np.shape}")

    for name, t in layer_outputs:
        arr = t.squeeze(0).cpu().numpy()
        path = out_dir / f"{name}_{sample_id}.npy"
        _save_npy(path, arr)
        written[name] = list(arr.shape)
        print(f"  {name:14}{path} shape={arr.shape}")

    enc_np = enc_out.squeeze(0).transpose(0, 1).cpu().numpy()
    path = out_dir / f"enc_post_ln_{sample_id}.npy"
    _save_npy(path, enc_np)
    written["enc_post_ln"] = list(enc_np.shape)
    print(f"  enc final:  {path} shape={enc_np.shape}")

    return written


def dump_predictor(np, torch, model: Any, out_dir: Path) -> dict:
    """Run a few embedding lookups + LSTM steps, dump the resulting state."""
    decoder = model.decoder
    written: dict = {}
    for tok in PRED_TEST_TOKENS:
        try:
            tok_t = torch.tensor([[tok]], dtype=torch.long)
            length = torch.tensor([1])
            h, _ = decoder.predict(tok_t, length=length)
            arr = h.squeeze(0).squeeze(0).cpu().numpy()  # [pred_hidden]
            path = out_dir / f"pred_state_{tok}.npy"
            _save_npy(path, arr)
            written[f"pred_state_{tok}"] = list(arr.shape)
            print(f"  pred[{tok:5}]: {path} shape={arr.shape}")
        except Exception as e:
            print(f"  WARNING: pred dump for token {tok} failed: {e}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", type=Path, required=True,
                    help="path to .nemo file")
    ap.add_argument("--model-id", type=str, required=True,
                    help="short identifier (subdir under out-dir)")
    ap.add_argument("--audio", type=Path, required=True,
                    help="16 kHz mono WAV file")
    ap.add_argument("--sample-id", type=str, default="ja",
                    help="short label embedded in fixture filenames")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="root fixtures directory")
    ap.add_argument("--skip-encoder", action="store_true",
                    help="dump only mel + predictor (faster smoke test)")
    args = ap.parse_args()

    np, torch, sf = _require_deps()
    nemo_asr = _require_nemo()

    if not args.model_path.exists():
        print(f"model not found: {args.model_path}", file=sys.stderr)
        return 1
    if not args.audio.exists():
        print(f"audio not found: {args.audio}", file=sys.stderr)
        return 1

    out_dir = args.out_dir / args.model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_path} ...")
    model = nemo_asr.models.ASRModel.restore_from(str(args.model_path))
    model.eval()

    print(f"dumping fixtures → {out_dir}")
    meta: dict = {
        "model_id": args.model_id,
        "sample_id": args.sample_id,
        "audio": str(args.audio),
        "tensors": {},
    }

    mel_np, _ = dump_mel(np, sf, model, args.audio, args.sample_id, out_dir)
    meta["tensors"][f"mel_{args.sample_id}"] = list(mel_np.shape)

    if not args.skip_encoder:
        enc_meta = dump_encoder_layers(np, torch, model, mel_np, args.sample_id, out_dir)
        meta["tensors"].update(enc_meta)

    pred_meta = dump_predictor(np, torch, model, out_dir)
    meta["tensors"].update(pred_meta)

    meta_path = out_dir / "fixture_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
