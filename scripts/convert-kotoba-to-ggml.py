#!/usr/bin/env python3
"""Convert kotoba-whisper-v2.0 (HuggingFace) to ggml format for whisper.cpp.

Writes the ggml binary format directly, handling:
- n_mels=128 (large-v3 spec)
- n_text_layer=2 (distilled)
- Correct vocab from HF tokenizer (51866 tokens)

Usage:
    python3 scripts/convert-kotoba-to-ggml.py [output_path]

Requires: pip install transformers openai-whisper torch numpy
"""

import sys
import struct
import numpy as np
from pathlib import Path


def main():
    output_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                       "third-party/whisper.cpp/models/ggml-kotoba-v2.bin")

    print("Loading kotoba-tech/kotoba-whisper-v2.0...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch

    model = WhisperForConditionalGeneration.from_pretrained(
        "kotoba-tech/kotoba-whisper-v2.0")
    processor = WhisperProcessor.from_pretrained(
        "kotoba-tech/kotoba-whisper-v2.0")
    config = model.config

    # Map HF keys → whisper.cpp ggml keys
    def map_key(k):
        if k.startswith("model."):
            k = k[6:]
        k = k.replace("layers.", "blocks.")
        k = k.replace("self_attn.q_proj", "attn.query")
        k = k.replace("self_attn.k_proj", "attn.key")
        k = k.replace("self_attn.v_proj", "attn.value")
        k = k.replace("self_attn.out_proj", "attn.out")
        k = k.replace("self_attn_layer_norm", "attn_ln")
        k = k.replace("encoder_attn.q_proj", "cross_attn.query")
        k = k.replace("encoder_attn.k_proj", "cross_attn.key")
        k = k.replace("encoder_attn.v_proj", "cross_attn.value")
        k = k.replace("encoder_attn.out_proj", "cross_attn.out")
        k = k.replace("encoder_attn_layer_norm", "cross_attn_ln")
        k = k.replace("fc1", "mlp.0")
        k = k.replace("fc2", "mlp.2")
        k = k.replace("final_layer_norm", "mlp_ln")
        k = k.replace("encoder.layer_norm", "encoder.ln_post")
        k = k.replace("decoder.layer_norm", "decoder.ln")
        k = k.replace("encoder.embed_positions.weight", "encoder.positional_embedding")
        k = k.replace("decoder.embed_positions.weight", "decoder.positional_embedding")
        k = k.replace("decoder.embed_tokens.weight", "decoder.token_embedding.weight")
        return k

    tensors = {}
    for hf_key, param in model.state_dict().items():
        ggml_key = map_key(hf_key)
        if ggml_key == "proj_out.weight":
            continue  # tied with token_embedding
        data = param.detach().cpu().float().numpy().squeeze()
        tensors[ggml_key] = data

    # Hyperparameters
    n_vocab = config.vocab_size
    n_audio_ctx = config.max_source_positions
    n_audio_state = config.d_model
    n_audio_head = config.encoder_attention_heads
    n_audio_layer = config.encoder_layers
    n_text_ctx = config.max_target_positions
    n_text_state = config.d_model
    n_text_head = config.decoder_attention_heads
    n_text_layer = config.decoder_layers
    n_mels = config.num_mel_bins
    ftype = 1  # mostly F16 (large tensors in F16, bias/embed in F32)

    print(f"  n_audio: ctx={n_audio_ctx} state={n_audio_state} head={n_audio_head} layer={n_audio_layer}")
    print(f"  n_text:  ctx={n_text_ctx} state={n_text_state} head={n_text_head} layer={n_text_layer}")
    print(f"  n_mels={n_mels} n_vocab={n_vocab} ftype={ftype}")
    print(f"  {len(tensors)} tensors")

    # Mel filters
    import whisper
    try:
        mel_filters = whisper.audio.mel_filters("cpu", n_mels).numpy()
    except TypeError:
        mel_filters = whisper.audio.mel_filters(
            whisper.audio.SAMPLE_RATE, whisper.audio.N_FFT, n_mels).numpy()
    n_fft_half = mel_filters.shape[1]
    print(f"  mel_filters: [{n_mels}, {n_fft_half}]")

    # Vocab from HF tokenizer
    tokenizer = processor.tokenizer
    vocab_dict = tokenizer.get_vocab()
    # Sort by token ID
    vocab_by_id = [""] * n_vocab
    for token, idx in vocab_dict.items():
        if idx < n_vocab:
            vocab_by_id[idx] = token
    # Fill any gaps
    for i in range(n_vocab):
        if not vocab_by_id[i]:
            vocab_by_id[i] = f"<extra_{i}>"
    print(f"  vocab: {n_vocab} tokens")

    # Write ggml binary
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("i", 0x67676d6c))  # magic: "ggml"
        for v in [n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
                  n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype]:
            f.write(struct.pack("i", v))

        # Mel filters
        f.write(struct.pack("i", n_mels))
        f.write(struct.pack("i", n_fft_half))
        mel_filters.astype(np.float32).tofile(f)

        # Vocab
        f.write(struct.pack("i", n_vocab))
        for token in vocab_by_id:
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("i", len(token_bytes)))
            f.write(token_bytes)

        # Tensors (same format as convert-pt-to-ggml.py)
        GGML_TYPE_F32 = 0
        GGML_TYPE_F16 = 1

        n_written = 0
        for name, data in tensors.items():
            n_dims = len(data.shape)

            # Small tensors stay F32, large tensors use F16
            use_f16 = ftype == 1 and n_dims >= 2 and \
                name != "encoder.conv1.bias" and \
                name != "encoder.conv2.bias" and \
                name != "encoder.positional_embedding" and \
                name != "decoder.positional_embedding"

            if use_f16:
                data_out = data.astype(np.float16)
                ttype = GGML_TYPE_F16
            else:
                data_out = data.astype(np.float32)
                ttype = GGML_TYPE_F32

            name_bytes = name.encode("utf-8")

            # Header: n_dims, name_len, ftype
            f.write(struct.pack("iii", n_dims, len(name_bytes), ttype))

            # Dimensions in reversed order (ggml convention, same as official converter)
            for i in range(n_dims):
                f.write(struct.pack("i", data.shape[n_dims - 1 - i]))

            # Name
            f.write(name_bytes)

            # Data (no alignment padding — whisper.cpp reads immediately)
            data_out.tofile(f)
            n_written += 1

        print(f"  {n_written} tensors written")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nWritten: {output_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
