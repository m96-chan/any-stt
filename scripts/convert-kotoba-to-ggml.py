#!/usr/bin/env python3
"""Convert kotoba-whisper-v2.0 (HuggingFace) to ggml format for whisper.cpp.

Usage:
    python3 scripts/convert-kotoba-to-ggml.py [output_dir]

Downloads kotoba-tech/kotoba-whisper-v2.0 and converts to ggml binary format
compatible with whisper.cpp's model loader.
"""

import sys
import struct
import numpy as np
from pathlib import Path

def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "third-party/whisper.cpp/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ggml-kotoba-v2.bin"

    print("Loading kotoba-tech/kotoba-whisper-v2.0 from HuggingFace...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained("kotoba-tech/kotoba-whisper-v2.0")
    processor = WhisperProcessor.from_pretrained("kotoba-tech/kotoba-whisper-v2.0")

    config = model.config
    print(f"  n_audio_state={config.d_model}, n_audio_layer={config.encoder_layers}")
    print(f"  n_text_state={config.d_model}, n_text_layer={config.decoder_layers}")
    print(f"  n_audio_head={config.encoder_attention_heads}, n_text_head={config.decoder_attention_heads}")
    print(f"  n_vocab={config.vocab_size}, n_mels={config.num_mel_bins}")
    print(f"  n_audio_ctx={config.max_source_positions}, n_text_ctx={config.max_target_positions}")

    state_dict = model.state_dict()

    # Map HuggingFace key names to whisper.cpp ggml names
    # HF format: model.encoder.layers.0.self_attn.q_proj.weight
    # ggml format: encoder.blocks.0.attn.query.weight
    def map_key(hf_key):
        k = hf_key
        # Remove "model." prefix
        if k.startswith("model."):
            k = k[6:]

        # Encoder mappings
        k = k.replace("encoder.layers.", "encoder.blocks.")
        k = k.replace("decoder.layers.", "decoder.blocks.")
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
        k = k.replace("decoder.embed_tokens", "decoder.token_embedding")
        k = k.replace("encoder.conv1", "encoder.conv1")
        k = k.replace("encoder.conv2", "encoder.conv2")
        k = k.replace("proj_out.weight", "decoder.proj.weight")
        return k

    # Collect and map tensors
    tensors = {}
    for hf_key, param in state_dict.items():
        ggml_key = map_key(hf_key)
        data = param.detach().cpu().float().numpy()
        tensors[ggml_key] = data

    print(f"  {len(tensors)} tensors mapped")

    # Write ggml binary format
    # Header: magic(u32) + hparams(11 x u32) + mel_filters + vocab + tensors
    GGML_FILE_MAGIC = 0x67676d6c  # "ggml"

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
    ftype = 1  # F16 for large tensors (but we write F32)

    # Actually write as ftype=0 (F32) since all tensors are F32
    ftype = 0

    with open(output_path, "wb") as f:
        # Magic
        f.write(struct.pack("<I", GGML_FILE_MAGIC))

        # Hyperparameters
        f.write(struct.pack("<I", n_vocab))
        f.write(struct.pack("<I", n_audio_ctx))
        f.write(struct.pack("<I", n_audio_state))
        f.write(struct.pack("<I", n_audio_head))
        f.write(struct.pack("<I", n_audio_layer))
        f.write(struct.pack("<I", n_text_ctx))
        f.write(struct.pack("<I", n_text_state))
        f.write(struct.pack("<I", n_text_head))
        f.write(struct.pack("<I", n_text_layer))
        f.write(struct.pack("<I", n_mels))
        f.write(struct.pack("<I", ftype))

        # Mel filters — load from whisper library
        import whisper
        # whisper.audio.mel_filters(device, n_mels) in newer versions
        try:
            mel_filters = whisper.audio.mel_filters("cpu", n_mels).numpy()
        except TypeError:
            mel_filters = whisper.audio.mel_filters(whisper.audio.SAMPLE_RATE, whisper.audio.N_FFT, n_mels).numpy()
        # mel_filters shape: [n_mels, n_fft/2+1]
        n_fft_half = mel_filters.shape[1]
        f.write(struct.pack("<I", n_mels))
        f.write(struct.pack("<I", n_fft_half))
        mel_filters.astype(np.float32).tofile(f)
        print(f"  mel_filters: [{n_mels}, {n_fft_half}]")

        # Vocab — use the tokenizer
        tokenizer = processor.tokenizer
        vocab = tokenizer.get_vocab()
        # Sort by ID
        vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
        n_vocab_write = len(vocab_sorted)
        f.write(struct.pack("<I", n_vocab_write))
        for token, idx in vocab_sorted:
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)
        print(f"  vocab: {n_vocab_write} tokens")

        # Tensors
        # ggml tensor format: name_len(i32) + n_dims(i32) + [dims...](i32) + type(i32) + data(aligned)
        GGML_TYPE_F32 = 0
        GGML_TYPE_F16 = 1

        n_written = 0
        for name, data in tensors.items():
            name_bytes = name.encode("utf-8")
            n_dims = len(data.shape)
            # ggml format: dims are ne[0], ne[1], ... (same as numpy shape for 1D/2D)
            # For tensors with shape [A, B, C], ggml ne = [C, B, A]
            # But the whisper.cpp loader reads ne[i] for i in 0..n_dims
            # and checks against the internal tensor ne which is also reversed.
            # So we write in the same order as the original convert-pt-to-ggml.py:
            # dims in file = reversed shape (ggml convention)
            dims = list(reversed(data.shape))

            # Determine type — use F32 for all (simplicity)
            dtype = GGML_TYPE_F32
            raw = data.astype(np.float32).tobytes()

            f.write(struct.pack("<I", n_dims))
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(struct.pack("<I", dtype))
            for d in dims:
                f.write(struct.pack("<I", d))
            f.write(name_bytes)

            # No alignment padding — whisper.cpp reads data immediately after name
            f.write(raw)
            n_written += 1

        print(f"  {n_written} tensors written")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nWritten: {output_path} ({size_mb:.0f} MB)")

if __name__ == "__main__":
    main()
