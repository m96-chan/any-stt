#!/usr/bin/env python3
"""Dump whisper encoder weights as raw FP32 binary files for QNN graph."""

import sys
import struct
from pathlib import Path
import numpy as np
import torch
import whisper

def dump_tensor(t: torch.Tensor, path: Path):
    """Save tensor as raw FP32 little-endian binary."""
    data = t.detach().float().cpu().contiguous().numpy()
    data.tofile(str(path))
    print(f"  {path.name}: {list(t.shape)} -> {path.stat().st_size} bytes")

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "tiny.en"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else f"weights_{model_name.replace('.','_')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)
    enc = model.encoder

    # Conv stem
    print("Conv stem:")
    dump_tensor(enc.conv1.weight, out_dir / "conv1_weight.bin")  # [384, 80, 3]
    dump_tensor(enc.conv1.bias, out_dir / "conv1_bias.bin")
    dump_tensor(enc.conv2.weight, out_dir / "conv2_weight.bin")  # [384, 384, 3]
    dump_tensor(enc.conv2.bias, out_dir / "conv2_bias.bin")

    # Positional embedding
    print("Positional embedding:")
    dump_tensor(enc.positional_embedding, out_dir / "pos_embed.bin")  # [1500, 384]

    # Encoder blocks
    for i, block in enumerate(enc.blocks):
        print(f"Block {i}:")
        prefix = f"block{i}"

        # Attention LayerNorm
        dump_tensor(block.attn_ln.weight, out_dir / f"{prefix}_attn_ln_w.bin")
        dump_tensor(block.attn_ln.bias, out_dir / f"{prefix}_attn_ln_b.bin")

        # Attention Q/K/V/Out
        dump_tensor(block.attn.query.weight.T, out_dir / f"{prefix}_wq.bin")  # Transpose for MatMul
        dump_tensor(block.attn.query.bias, out_dir / f"{prefix}_bq.bin")
        dump_tensor(block.attn.key.weight.T, out_dir / f"{prefix}_wk.bin")
        # key has no bias
        dump_tensor(block.attn.value.weight.T, out_dir / f"{prefix}_wv.bin")
        dump_tensor(block.attn.value.bias, out_dir / f"{prefix}_bv.bin")
        dump_tensor(block.attn.out.weight.T, out_dir / f"{prefix}_wo.bin")
        dump_tensor(block.attn.out.bias, out_dir / f"{prefix}_bo.bin")

        # MLP LayerNorm
        dump_tensor(block.mlp_ln.weight, out_dir / f"{prefix}_mlp_ln_w.bin")
        dump_tensor(block.mlp_ln.bias, out_dir / f"{prefix}_mlp_ln_b.bin")

        # FFN
        dump_tensor(block.mlp[0].weight.T, out_dir / f"{prefix}_fc1_w.bin")  # [384, 1536]
        dump_tensor(block.mlp[0].bias, out_dir / f"{prefix}_fc1_b.bin")
        dump_tensor(block.mlp[2].weight.T, out_dir / f"{prefix}_fc2_w.bin")  # [1536, 384]
        dump_tensor(block.mlp[2].bias, out_dir / f"{prefix}_fc2_b.bin")

    # Final LayerNorm
    print("Final LN:")
    dump_tensor(enc.ln_post.weight, out_dir / "ln_post_w.bin")
    dump_tensor(enc.ln_post.bias, out_dir / "ln_post_b.bin")

    # Also dump a test mel spectrogram from JFK audio
    print("\nGenerating mel spectrogram for JFK audio...")
    audio = whisper.load_audio("third-party/whisper.cpp/samples/jfk.wav")
    mel = whisper.log_mel_spectrogram(audio, n_mels=80, padding=0)
    mel_padded = whisper.pad_or_trim(mel, 3000)  # [80, 3000]
    # QNN expects [1, 3000, 80] (NWC) but our encoder takes [1500, 384] after conv
    # Let's dump the conv output instead
    mel_input = mel_padded.unsqueeze(0).to(next(enc.parameters()).device)  # [1, 80, 3000]

    # Run conv stem + positional embedding on CPU to get encoder input
    with torch.no_grad():
        x = torch.nn.functional.gelu(enc.conv1(mel_input))
        x = torch.nn.functional.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)  # [1, 1500, 384]
        x = x + enc.positional_embedding
        encoder_input = x.squeeze(0)  # [1500, 384]

    dump_tensor(encoder_input, out_dir / "encoder_input.bin")
    print(f"  encoder_input: {list(encoder_input.shape)}")

    # Run full encoder to get reference output
    with torch.no_grad():
        encoder_output = enc(mel_input).squeeze(0)  # [1500, 384]
    dump_tensor(encoder_output, out_dir / "encoder_output_ref.bin")
    print(f"  encoder_output_ref: {list(encoder_output.shape)}")

    # Summary
    total_size = sum(f.stat().st_size for f in out_dir.iterdir())
    print(f"\nTotal: {total_size / 1024 / 1024:.1f} MB in {out_dir}")

if __name__ == "__main__":
    main()
