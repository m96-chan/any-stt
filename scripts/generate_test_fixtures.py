#!/usr/bin/env python3
"""
Generate reference test fixtures for whisper-backend layer-by-layer testing.

Runs OpenAI whisper (Python reference implementation) on test audio and captures
intermediate outputs at each layer. These fixtures are used by Rust integration
tests to verify that whisper.cpp produces matching results.

Usage:
    python scripts/generate_test_fixtures.py

Output:
    crates/whisper-backend/tests/fixtures/
        reference_tiny_en_jfk.json   — Full reference data
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import whisper
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim, N_FRAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "crates" / "whisper-backend" / "tests" / "fixtures"
AUDIO_PATH = REPO_ROOT / "third-party" / "whisper.cpp" / "samples" / "jfk.wav"

MODEL_NAME = "tiny.en"


def tensor_stats(t) -> dict:
    """Compute summary statistics for a tensor (or first element of a tuple)."""
    if isinstance(t, tuple):
        t = t[0]
    t_float = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(t_float.min()),
        "max": float(t_float.max()),
        "mean": float(t_float.mean()),
        "std": float(t_float.std()) if t.numel() > 1 else 0.0,
        "abs_mean": float(t_float.abs().mean()),
        # First few values for spot-checking
        "head": t_float.flatten()[:16].tolist(),
    }


def capture_mel_spectrogram(audio_path: Path) -> dict:
    """Capture mel spectrogram from audio file."""
    audio = load_audio(str(audio_path))
    mel = log_mel_spectrogram(audio, n_mels=80, padding=0)
    mel_padded = pad_or_trim(mel, N_FRAMES)

    return {
        "audio_samples": len(audio),
        "audio_duration_s": len(audio) / 16000,
        "mel_raw": tensor_stats(mel),
        "mel_padded": tensor_stats(mel_padded),
    }


def capture_encoder_layers(model, mel: torch.Tensor) -> dict:
    """Run encoder and capture per-layer outputs."""
    encoder = model.encoder
    device = next(encoder.parameters()).device
    mel = mel.unsqueeze(0).to(device)

    results = {}
    hooks = []

    # Conv layers
    conv1_output = {}
    conv2_output = {}

    def hook_conv1(module, input, output):
        conv1_output["input"] = tensor_stats(input[0])
        conv1_output["output"] = tensor_stats(output)

    def hook_conv2(module, input, output):
        conv2_output["input"] = tensor_stats(input[0])
        conv2_output["output"] = tensor_stats(output)

    hooks.append(encoder.conv1.register_forward_hook(hook_conv1))
    hooks.append(encoder.conv2.register_forward_hook(hook_conv2))

    # Encoder blocks
    block_outputs = {}
    for i, block in enumerate(encoder.blocks):
        layer_data = {}

        def make_hooks(idx, data):
            def hook_attn_ln(module, input, output):
                data["attn_ln_output"] = tensor_stats(output)

            def hook_attn(module, input, output):
                data["attn_output"] = tensor_stats(output)

            def hook_mlp_ln(module, input, output):
                data["mlp_ln_output"] = tensor_stats(output)

            def hook_mlp(module, input, output):
                data["mlp_output"] = tensor_stats(output)

            def hook_block(module, input, output):
                data["block_input"] = tensor_stats(input[0])
                data["block_output"] = tensor_stats(output)

            return hook_attn_ln, hook_attn, hook_mlp_ln, hook_mlp, hook_block

        h_attn_ln, h_attn, h_mlp_ln, h_mlp, h_block = make_hooks(i, layer_data)
        hooks.append(block.attn_ln.register_forward_hook(h_attn_ln))
        hooks.append(block.attn.register_forward_hook(h_attn))
        hooks.append(block.mlp_ln.register_forward_hook(h_mlp_ln))
        hooks.append(block.mlp.register_forward_hook(h_mlp))
        hooks.append(block.register_forward_hook(h_block))

        block_outputs[f"block_{i}"] = layer_data

    # ln_post
    ln_post_data = {}

    def hook_ln_post(module, input, output):
        ln_post_data["input"] = tensor_stats(input[0])
        ln_post_data["output"] = tensor_stats(output)

    hooks.append(encoder.ln_post.register_forward_hook(hook_ln_post))

    # Run encoder
    with torch.no_grad():
        encoder_output = encoder(mel)

    # Clean up hooks
    for h in hooks:
        h.remove()

    results["conv1"] = conv1_output
    results["conv2"] = conv2_output
    results["blocks"] = block_outputs
    results["ln_post"] = ln_post_data
    results["final_output"] = tensor_stats(encoder_output)

    return results, encoder_output


def capture_decoder_layers(model, encoder_output: torch.Tensor, tokens: torch.Tensor) -> dict:
    """Run decoder and capture per-layer outputs."""
    decoder = model.decoder
    device = next(decoder.parameters()).device
    tokens = tokens.to(device)

    results = {}
    hooks = []

    # Token + positional embedding
    embed_data = {}

    def hook_token_embed(module, input, output):
        embed_data["token_embedding"] = tensor_stats(output)

    hooks.append(decoder.token_embedding.register_forward_hook(hook_token_embed))

    # Decoder blocks
    block_outputs = {}
    for i, block in enumerate(decoder.blocks):
        layer_data = {}

        def make_hooks(idx, data):
            def hook_attn_ln(module, input, output):
                data["self_attn_ln_output"] = tensor_stats(output)

            def hook_attn(module, input, output):
                data["self_attn_output"] = tensor_stats(output)

            def hook_cross_attn_ln(module, input, output):
                data["cross_attn_ln_output"] = tensor_stats(output)

            def hook_cross_attn(module, input, output):
                data["cross_attn_output"] = tensor_stats(output)

            def hook_mlp_ln(module, input, output):
                data["mlp_ln_output"] = tensor_stats(output)

            def hook_mlp(module, input, output):
                data["mlp_output"] = tensor_stats(output)

            def hook_block(module, input, output):
                data["block_output"] = tensor_stats(output)

            return (hook_attn_ln, hook_attn, hook_cross_attn_ln,
                    hook_cross_attn, hook_mlp_ln, hook_mlp, hook_block)

        (h_attn_ln, h_attn, h_cross_attn_ln, h_cross_attn,
         h_mlp_ln, h_mlp, h_block) = make_hooks(i, layer_data)

        hooks.append(block.attn_ln.register_forward_hook(h_attn_ln))
        hooks.append(block.attn.register_forward_hook(h_attn))
        hooks.append(block.cross_attn_ln.register_forward_hook(h_cross_attn_ln))
        hooks.append(block.cross_attn.register_forward_hook(h_cross_attn))
        hooks.append(block.mlp_ln.register_forward_hook(h_mlp_ln))
        hooks.append(block.mlp.register_forward_hook(h_mlp))
        hooks.append(block.register_forward_hook(h_block))

        block_outputs[f"block_{i}"] = layer_data

    # Final layer norm
    ln_data = {}

    def hook_ln(module, input, output):
        ln_data["input"] = tensor_stats(input[0])
        ln_data["output"] = tensor_stats(output)

    hooks.append(decoder.ln.register_forward_hook(hook_ln))

    # Run decoder
    with torch.no_grad():
        logits = decoder(tokens, encoder_output)

    for h in hooks:
        h.remove()

    results["token_embedding"] = embed_data
    results["blocks"] = block_outputs
    results["ln"] = ln_data
    results["logits"] = tensor_stats(logits)

    return results


def capture_full_transcription(model, audio_path: Path) -> dict:
    """Run full transcription and capture results."""
    result = whisper.transcribe(model, str(audio_path), language="en", fp16=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "tokens": seg["tokens"],
        })

    return {
        "text": result["text"],
        "language": result["language"],
        "segments": segments,
    }


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    if not AUDIO_PATH.exists():
        print(f"ERROR: Audio file not found: {AUDIO_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {MODEL_NAME}")
    model = whisper.load_model(MODEL_NAME)
    device = next(model.parameters()).device
    print(f"Device: {device}")

    # 1. Mel spectrogram
    print("Capturing mel spectrogram...")
    mel_data = capture_mel_spectrogram(AUDIO_PATH)

    # Prepare mel for encoder
    audio = load_audio(str(AUDIO_PATH))
    mel = log_mel_spectrogram(audio, n_mels=80, padding=0)
    mel_padded = pad_or_trim(mel, N_FRAMES)

    # 2. Encoder layers
    print("Capturing encoder layers...")
    encoder_data, encoder_output = capture_encoder_layers(model, mel_padded)

    # 3. Decoder layers (using SOT tokens as prompt)
    print("Capturing decoder layers...")
    sot_token = model.decoder.token_embedding.num_embeddings - 1  # approximate
    # Use the standard Whisper SOT sequence for English transcription
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
    sot_sequence = [tokenizer.sot, tokenizer.transcribe, tokenizer.no_timestamps]
    tokens = torch.tensor([sot_sequence], dtype=torch.long)
    decoder_data = capture_decoder_layers(model, encoder_output, tokens)

    # 4. Full transcription
    print("Capturing full transcription...")
    transcription_data = capture_full_transcription(model, AUDIO_PATH)

    # Assemble fixture
    fixture = {
        "model": MODEL_NAME,
        "audio_file": "jfk.wav",
        "audio_path": str(AUDIO_PATH),
        "mel_spectrogram": mel_data,
        "encoder": encoder_data,
        "decoder": decoder_data,
        "transcription": transcription_data,
    }

    output_path = FIXTURES_DIR / "reference_tiny_en_jfk.json"
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)

    print(f"\nFixture written to: {output_path}")
    print(f"Transcription: {transcription_data['text']}")

    # Print summary
    print(f"\n--- Summary ---")
    print(f"Audio: {mel_data['audio_samples']} samples, {mel_data['audio_duration_s']:.2f}s")
    print(f"Mel: {mel_data['mel_padded']['shape']}")
    print(f"Encoder output: {encoder_data['final_output']['shape']}")
    print(f"Encoder blocks: {len(encoder_data['blocks'])}")
    print(f"Decoder blocks: {len(decoder_data['blocks'])}")
    print(f"Logits: {decoder_data['logits']['shape']}")
    print(f"Segments: {len(transcription_data['segments'])}")
    print(f"Language: {transcription_data['language']}")


if __name__ == "__main__":
    main()
