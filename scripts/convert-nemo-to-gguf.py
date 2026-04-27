#!/usr/bin/env python3
"""Convert a NeMo FastConformer ASR checkpoint to GGUF v3.

Supports:
  - reazonspeech-nemo-v2 (FastConformer + Longformer attention + RNN-T)
  - parakeet-tdt-0.6b-v3  (FastConformer + rel-pos attention + TDT)

Output artifacts:
  <output>.gguf            — weights + architecture metadata
  <output>.tokenizer.model — raw SentencePiece model (separate file)

Requirements:
  pip install torch numpy pyyaml gguf sentencepiece

Typical use:
  # download and unpack the HuggingFace model first
  huggingface-cli download reazon-research/reazonspeech-nemo-v2 \\
      --local-dir ./reazonspeech-nemo-v2

  python scripts/convert-nemo-to-gguf.py \\
      ./reazonspeech-nemo-v2/model.nemo \\
      ./models/reazonspeech-nemo-v2.gguf \\
      --dtype f16

This script does NOT require nemo-toolkit to run — it only reads the
`.nemo` tar archive and the PyTorch checkpoint inside. nemo-toolkit is
only needed when regenerating reference tensor fixtures for Rust unit
tests (see scripts/dump-nemo-fixtures.py — TODO).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Heavy deps are optional at module load so --self-test and --help run on
# stock Python 3. `_require_heavy_deps()` is called from main() before any
# code that needs them.
try:
    import numpy as np
    import torch
    import yaml
    _HEAVY_OK, _HEAVY_MISSING = True, None
except ImportError as e:
    np = torch = yaml = None  # type: ignore[assignment]
    _HEAVY_OK, _HEAVY_MISSING = False, e.name

try:
    import gguf
    _GGUF_OK = True
except ImportError:
    gguf = None  # type: ignore[assignment]
    _GGUF_OK = False


def _require_heavy_deps() -> None:
    if not _HEAVY_OK:
        sys.exit(f"missing dependency: {_HEAVY_MISSING} — pip install torch numpy pyyaml")
    if not _GGUF_OK:
        sys.exit("missing dependency: gguf — pip install gguf")


log = logging.getLogger("convert-nemo-to-gguf")


# ---------------------------------------------------------------------------
# GGUF architecture key — one shared family for both NeMo FastConformer
# variants. Per-variant differences live in metadata fields, not separate
# architecture strings, so one backend crate can load both.
# ---------------------------------------------------------------------------

ARCH = "fastconformer"


# ---------------------------------------------------------------------------
# NeMo → any-stt tensor name mapping.
#
# The mapping below targets the Conformer/FastConformer checkpoint layout
# shipped by NVIDIA NeMo 2.x (as of 2026-03). If NeMo renames a submodule
# upstream, adjust the patterns here — this is the only place NeMo names
# touch the codebase.
# ---------------------------------------------------------------------------

# (regex, replacement) — first match wins. Uses Python named groups.
TENSOR_RENAMES: list[tuple[re.Pattern[str], str]] = [
    # Subsampling (stride=8 striding_conv2d)
    (re.compile(r"^encoder\.pre_encode\.conv\.(\d+)\.(weight|bias)$"),
     r"enc.subsample.conv.\1.\2"),
    (re.compile(r"^encoder\.pre_encode\.out\.(weight|bias)$"),
     r"enc.subsample.out.\1"),

    # Positional encoding
    (re.compile(r"^encoder\.pos_enc\.pe$"), r"enc.pos.pe"),

    # Conformer blocks: FF module 1 (before self-attention)
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_feed_forward1\.(weight|bias)$"),
     r"enc.block.\1.ff1.ln.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.feed_forward1\.linear1\.(weight|bias)$"),
     r"enc.block.\1.ff1.fc1.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.feed_forward1\.linear2\.(weight|bias)$"),
     r"enc.block.\1.ff1.fc2.\2"),

    # Self-attention
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_self_att\.(weight|bias)$"),
     r"enc.block.\1.attn.ln.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_q\.(weight|bias)$"),
     r"enc.block.\1.attn.q.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_k\.(weight|bias)$"),
     r"enc.block.\1.attn.k.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_v\.(weight|bias)$"),
     r"enc.block.\1.attn.v.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_out\.(weight|bias)$"),
     r"enc.block.\1.attn.out.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_pos\.(weight|bias)$"),
     r"enc.block.\1.attn.pos.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.pos_bias_u$"),
     r"enc.block.\1.attn.pos_bias_u"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.pos_bias_v$"),
     r"enc.block.\1.attn.pos_bias_v"),

    # Convolution module
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_conv\.(weight|bias)$"),
     r"enc.block.\1.conv.ln.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.conv\.pointwise_conv1\.(weight|bias)$"),
     r"enc.block.\1.conv.pw1.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.conv\.depthwise_conv\.(weight|bias)$"),
     r"enc.block.\1.conv.dw.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.conv\.batch_norm\.(weight|bias|running_mean|running_var)$"),
     r"enc.block.\1.conv.bn.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.conv\.pointwise_conv2\.(weight|bias)$"),
     r"enc.block.\1.conv.pw2.\2"),

    # FF module 2 (after conv)
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_feed_forward2\.(weight|bias)$"),
     r"enc.block.\1.ff2.ln.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.feed_forward2\.linear1\.(weight|bias)$"),
     r"enc.block.\1.ff2.fc1.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.feed_forward2\.linear2\.(weight|bias)$"),
     r"enc.block.\1.ff2.fc2.\2"),

    # Post block LN
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_out\.(weight|bias)$"),
     r"enc.block.\1.ln_post.\2"),

    # Final encoder norm
    (re.compile(r"^encoder\.norm\.(weight|bias)$"),
     r"enc.ln_post.\1"),

    # Decoder (RNN-T / TDT prediction network)
    (re.compile(r"^decoder\.prediction\.embed\.weight$"),
     r"dec.embed.weight"),
    (re.compile(r"^decoder\.prediction\.dec_rnn\.lstm\.weight_ih_l(\d+)$"),
     r"dec.rnn.\1.weight_ih"),
    (re.compile(r"^decoder\.prediction\.dec_rnn\.lstm\.weight_hh_l(\d+)$"),
     r"dec.rnn.\1.weight_hh"),
    (re.compile(r"^decoder\.prediction\.dec_rnn\.lstm\.bias_ih_l(\d+)$"),
     r"dec.rnn.\1.bias_ih"),
    (re.compile(r"^decoder\.prediction\.dec_rnn\.lstm\.bias_hh_l(\d+)$"),
     r"dec.rnn.\1.bias_hh"),

    # Joint network (RNN-T joint / TDT joint)
    (re.compile(r"^joint\.enc\.(weight|bias)$"), r"joint.enc.\1"),
    (re.compile(r"^joint\.pred\.(weight|bias)$"), r"joint.pred.\1"),
    (re.compile(r"^joint\.joint_net\.0\.(weight|bias)$"),
     r"joint.fc1.\1"),
    (re.compile(r"^joint\.joint_net\.2\.(weight|bias)$"),
     r"joint.fc2.\1"),
]


def rename_tensor(name: str) -> str | None:
    for pat, repl in TENSOR_RENAMES:
        m = pat.match(name)
        if m:
            return pat.sub(repl, name)
    return None


# Sanity-check rename patterns against representative NeMo names.
# Run with: python scripts/convert-nemo-to-gguf.py --self-test
_SELF_TEST_CASES: list[tuple[str, str]] = [
    # (NeMo name, expected GGUF name)
    ("encoder.layers.0.self_attn.linear_q.weight", "enc.block.0.attn.q.weight"),
    ("encoder.layers.17.self_attn.pos_bias_u",    "enc.block.17.attn.pos_bias_u"),
    ("encoder.layers.3.norm_self_att.weight",     "enc.block.3.attn.ln.weight"),
    ("encoder.layers.5.conv.depthwise_conv.weight", "enc.block.5.conv.dw.weight"),
    ("encoder.layers.5.conv.batch_norm.running_mean", "enc.block.5.conv.bn.running_mean"),
    ("encoder.layers.9.feed_forward1.linear1.bias", "enc.block.9.ff1.fc1.bias"),
    ("encoder.layers.9.feed_forward2.linear2.weight", "enc.block.9.ff2.fc2.weight"),
    ("encoder.layers.11.norm_out.bias",            "enc.block.11.ln_post.bias"),
    ("encoder.norm.weight",                        "enc.ln_post.weight"),
    ("encoder.pre_encode.conv.0.weight",           "enc.subsample.conv.0.weight"),
    ("encoder.pre_encode.out.weight",              "enc.subsample.out.weight"),
    ("encoder.pos_enc.pe",                         "enc.pos.pe"),
    ("decoder.prediction.embed.weight",            "dec.embed.weight"),
    ("decoder.prediction.dec_rnn.lstm.weight_ih_l0", "dec.rnn.0.weight_ih"),
    ("joint.joint_net.0.weight",                   "joint.fc1.weight"),
    ("joint.joint_net.2.bias",                     "joint.fc2.bias"),
]


def run_self_test() -> int:
    fails = 0
    for input_name, expected in _SELF_TEST_CASES:
        got = rename_tensor(input_name)
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            fails += 1
            print(f"[{status}] {input_name!r}")
            print(f"       expected: {expected!r}")
            print(f"       got:      {got!r}")
        else:
            print(f"[{status}] {input_name} → {got}")
    return 0 if fails == 0 else 1


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

@dataclass
class FastConformerConfig:
    # Encoder
    n_layers: int
    d_model: int
    n_heads: int
    feat_in: int            # n_mels
    conv_kernel_size: int
    subsampling_factor: int
    # Attention
    attention_type: str     # "rel_pos" | "rel_pos_local_attn"
    local_window: int       # only meaningful for rel_pos_local_attn
    global_tokens: int      # Longformer global tokens, typically 0 or 1
    # Decoder
    decoder_type: str       # "rnnt" | "tdt"
    vocab_size: int
    pred_hidden: int
    joint_hidden: int
    blank_id: int
    tdt_durations: list[int] | None  # TDT only
    # Audio preprocessing
    sample_rate: int
    win_length: int
    hop_length: int

    @classmethod
    def from_nemo_cfg(cls, cfg: dict) -> "FastConformerConfig":
        model = cfg.get("model", cfg)  # support both flattened and nested
        enc = model["encoder"]
        dec = model.get("decoder", {})
        joint = model.get("joint", {})

        # Attention type detection
        sa_model = enc.get("self_attention_model", "rel_pos")
        att_ctx = enc.get("att_context_size", [-1, -1])
        local_window = 0
        if isinstance(att_ctx, (list, tuple)) and len(att_ctx) == 2:
            # NeMo stores [left, right]; "256" means local window 256 each side.
            if att_ctx[0] > 0:
                local_window = int(att_ctx[0])
        global_tokens = int(enc.get("global_tokens", 0) or 0)

        # Decoder type detection
        dec_target = str(dec.get("_target_", "")).lower()
        joint_target = str(joint.get("_target_", "")).lower()
        if "tdt" in dec_target or "tdt" in joint_target:
            decoder_type = "tdt"
        else:
            decoder_type = "rnnt"

        # TDT durations (default [0,1,2,3,4] per Parakeet TDT config)
        tdt_durations = None
        if decoder_type == "tdt":
            defaults = model.get("model_defaults", {})
            tdt_durations = list(defaults.get("tdt_durations", [0, 1, 2, 3, 4]))

        vocab_size = int(
            joint.get("num_classes")
            or dec.get("vocab_size")
            or joint.get("vocabulary", {}).get("num_classes")
            or 0
        )
        if vocab_size == 0:
            raise ValueError("could not determine vocab_size from NeMo config")

        prednet = dec.get("prednet", {})
        jointnet = joint.get("jointnet", {})

        preproc = model.get("preprocessor", {})

        return cls(
            n_layers=int(enc["n_layers"]),
            d_model=int(enc["d_model"]),
            n_heads=int(enc["n_heads"]),
            feat_in=int(enc.get("feat_in", 80)),
            conv_kernel_size=int(enc.get("conv_kernel_size", 9)),
            subsampling_factor=int(enc.get("subsampling_factor", 8)),
            attention_type=sa_model,
            local_window=local_window,
            global_tokens=global_tokens,
            decoder_type=decoder_type,
            vocab_size=vocab_size,
            pred_hidden=int(prednet.get("pred_hidden", 640)),
            joint_hidden=int(jointnet.get("joint_hidden", 640)),
            blank_id=int(dec.get("blank_as_pad", True) and (vocab_size - 1) or 0),
            tdt_durations=tdt_durations,
            sample_rate=int(preproc.get("sample_rate", 16000)),
            win_length=int(preproc.get("window_size", 0.025) * preproc.get("sample_rate", 16000)),
            hop_length=int(preproc.get("window_stride", 0.01) * preproc.get("sample_rate", 16000)),
        )


# ---------------------------------------------------------------------------
# .nemo archive handling
# ---------------------------------------------------------------------------

def extract_nemo(nemo_path: Path, dest: Path) -> tuple[Path, Path, Path | None]:
    """Unpack a .nemo file and return (config_yaml, weights_ckpt, tokenizer_model).

    NeMo .nemo is just a tar archive with:
      - model_config.yaml
      - model_weights.ckpt (PyTorch state_dict) OR .safetensors
      - *.model (SentencePiece tokenizer — name varies)
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path) as tar:
        tar.extractall(dest)

    files = list(dest.rglob("*"))
    cfg = next((f for f in files if f.name == "model_config.yaml"), None)
    ckpt = next((f for f in files if f.name == "model_weights.ckpt"), None)
    if cfg is None or ckpt is None:
        raise FileNotFoundError(
            f"expected model_config.yaml and model_weights.ckpt in {nemo_path}; "
            f"found: {[f.name for f in files if f.is_file()]}"
        )
    # SentencePiece model may have varied names; pick the largest *.model.
    sp_candidates = [f for f in files if f.suffix == ".model" and f.stat().st_size > 0]
    sp_model = max(sp_candidates, key=lambda f: f.stat().st_size, default=None)
    return cfg, ckpt, sp_model


# ---------------------------------------------------------------------------
# GGUF writer
# ---------------------------------------------------------------------------

def dtype_to_gguf(name: str) -> "gguf.GGMLQuantizationType":
    return {
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
    }[name]


def cast_tensor(t: torch.Tensor, dtype: str) -> np.ndarray:
    arr = t.detach().cpu().contiguous()
    if dtype == "f32":
        return arr.to(torch.float32).numpy()
    if dtype == "f16":
        return arr.to(torch.float16).numpy()
    raise ValueError(f"unsupported dtype: {dtype}")


def write_gguf(
    cfg: FastConformerConfig,
    state_dict: dict[str, torch.Tensor],
    output: Path,
    dtype: str,
    source_label: str,
) -> None:
    writer = gguf.GGUFWriter(str(output), ARCH)

    # --- General metadata ---
    writer.add_architecture()
    writer.add_name(source_label)
    writer.add_description(
        f"Converted from NeMo {source_label} — "
        f"FastConformer {cfg.decoder_type.upper()} {cfg.n_layers}L "
        f"d_model={cfg.d_model} vocab={cfg.vocab_size}"
    )

    # --- Encoder metadata ---
    writer.add_uint32(f"{ARCH}.encoder.n_layers", cfg.n_layers)
    writer.add_uint32(f"{ARCH}.encoder.d_model", cfg.d_model)
    writer.add_uint32(f"{ARCH}.encoder.n_heads", cfg.n_heads)
    writer.add_uint32(f"{ARCH}.encoder.feat_in", cfg.feat_in)
    writer.add_uint32(f"{ARCH}.encoder.conv_kernel_size", cfg.conv_kernel_size)
    writer.add_uint32(f"{ARCH}.encoder.subsampling_factor", cfg.subsampling_factor)
    writer.add_string(f"{ARCH}.encoder.attention_type", cfg.attention_type)
    writer.add_uint32(f"{ARCH}.encoder.local_window", cfg.local_window)
    writer.add_uint32(f"{ARCH}.encoder.global_tokens", cfg.global_tokens)

    # --- Decoder metadata ---
    writer.add_string(f"{ARCH}.decoder.type", cfg.decoder_type)
    writer.add_uint32(f"{ARCH}.decoder.vocab_size", cfg.vocab_size)
    writer.add_uint32(f"{ARCH}.decoder.pred_hidden", cfg.pred_hidden)
    writer.add_uint32(f"{ARCH}.decoder.joint_hidden", cfg.joint_hidden)
    writer.add_uint32(f"{ARCH}.decoder.blank_id", cfg.blank_id)
    if cfg.decoder_type == "tdt" and cfg.tdt_durations:
        writer.add_array(f"{ARCH}.decoder.tdt_durations",
                         [int(d) for d in cfg.tdt_durations])

    # --- Audio preprocessing ---
    writer.add_uint32(f"{ARCH}.audio.sample_rate", cfg.sample_rate)
    writer.add_uint32(f"{ARCH}.audio.win_length", cfg.win_length)
    writer.add_uint32(f"{ARCH}.audio.hop_length", cfg.hop_length)
    writer.add_uint32(f"{ARCH}.audio.n_mels", cfg.feat_in)

    # --- Tensors ---
    skipped: list[str] = []
    renamed_count = 0
    for orig_name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        new_name = rename_tensor(orig_name)
        if new_name is None:
            skipped.append(orig_name)
            continue
        arr = cast_tensor(tensor, dtype)
        writer.add_tensor(new_name, arr)
        renamed_count += 1

    if skipped:
        log.info("skipped %d tensors without a mapping (first 10):", len(skipped))
        for n in skipped[:10]:
            log.info("  - %s", n)

    log.info("wrote %d tensors to %s", renamed_count, output)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("nemo_path", type=Path, nargs="?",
                    help="path to .nemo archive (required unless --self-test)")
    ap.add_argument("output", type=Path, nargs="?",
                    help="output .gguf path (required unless --self-test)")
    ap.add_argument(
        "--dtype",
        choices=("f32", "f16"),
        default="f16",
        help="quantization for non-quant tensors (default: f16)",
    )
    ap.add_argument(
        "--source-label",
        default=None,
        help="human-readable model name stored in GGUF metadata. "
        "defaults to output file stem.",
    )
    ap.add_argument(
        "--dump-raw-names",
        action="store_true",
        help="log every tensor name in the checkpoint (debug)",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="only run rename-table self-tests, do not convert",
    )
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    if args.nemo_path is None or args.output is None:
        ap.error("nemo_path and output are required (or pass --self-test)")

    _require_heavy_deps()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.nemo_path.exists():
        log.error("not found: %s", args.nemo_path)
        return 1

    label = args.source_label or args.output.stem

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log.info("extracting %s ...", args.nemo_path)
        cfg_yaml, ckpt_path, sp_model = extract_nemo(args.nemo_path, tmp_path)

        log.info("parsing config ...")
        with open(cfg_yaml) as f:
            cfg_raw = yaml.safe_load(f)
        cfg = FastConformerConfig.from_nemo_cfg(cfg_raw)
        log.info(
            "detected: %s decoder, %d layers, d_model=%d, vocab=%d",
            cfg.decoder_type.upper(), cfg.n_layers, cfg.d_model, cfg.vocab_size,
        )

        log.info("loading weights %s ...", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if args.dump_raw_names:
            for name in sorted(state_dict.keys()):
                print(name)
            return 0

        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_gguf(cfg, state_dict, args.output, args.dtype, label)

        # Copy SentencePiece tokenizer alongside
        if sp_model is not None:
            companion = args.output.with_suffix(".tokenizer.model")
            companion.write_bytes(sp_model.read_bytes())
            log.info("wrote tokenizer → %s", companion)

    return 0


if __name__ == "__main__":
    sys.exit(main())
