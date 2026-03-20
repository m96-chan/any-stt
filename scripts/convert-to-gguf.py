#!/usr/bin/env python3
"""Convert OpenAI whisper model to GGUF v3 format for any-stt.

Usage:
    python3 scripts/convert-to-gguf.py tiny.en [output.gguf]

Requires: pip install openai-whisper numpy
"""

import sys
import struct
import numpy as np
from pathlib import Path

# GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8

# ggml types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def write_string(f, s: str):
    """Write GGUF string: u64 length + UTF-8 bytes."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def write_kv_string(f, key: str, value: str):
    """Write a string key-value pair."""
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    write_string(f, value)


def write_kv_u32(f, key: str, value: int):
    """Write a uint32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", value))


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "tiny.en"
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"ggml-{model_name.replace('.', '-')}.gguf"

    print(f"Loading OpenAI whisper model: {model_name}")
    import whisper
    model = whisper.load_model(model_name)

    # Collect all tensors
    tensors = []
    for name, param in model.named_parameters():
        data = param.detach().cpu()
        # Store as F32 for simplicity
        np_data = data.float().numpy()
        tensors.append((name, np_data))

    for name, buf in model.named_buffers():
        t = buf.detach().cpu()
        if t.is_sparse:
            t = t.to_dense()
        np_data = t.float().numpy()
        tensors.append((name, np_data))

    print(f"  {len(tensors)} tensors")

    # Metadata
    dims = model.dims
    metadata = [
        ("general.architecture", "whisper"),
        ("general.name", f"whisper-{model_name}"),
        ("whisper.encoder.n_audio_ctx", dims.n_audio_ctx),
        ("whisper.encoder.n_audio_state", dims.n_audio_state),
        ("whisper.encoder.n_audio_head", dims.n_audio_head),
        ("whisper.encoder.n_audio_layer", dims.n_audio_layer),
        ("whisper.decoder.n_text_ctx", dims.n_text_ctx),
        ("whisper.decoder.n_text_state", dims.n_text_state),
        ("whisper.decoder.n_text_head", dims.n_text_head),
        ("whisper.decoder.n_text_layer", dims.n_text_layer),
        ("whisper.encoder.n_mels", dims.n_mels),
        ("whisper.encoder.n_vocab", dims.n_vocab),
    ]

    n_kv = len(metadata)
    n_tensors = len(tensors)

    # Compute tensor data offsets
    # First pass: compute header size, then align data
    # Header: magic(4) + version(4) + n_tensors(8) + n_kv(8)
    # + KV pairs + tensor infos
    # Then: aligned tensor data

    # Build header in memory
    header_buf = bytearray()
    header_buf += GGUF_MAGIC
    header_buf += struct.pack("<I", GGUF_VERSION)
    header_buf += struct.pack("<Q", n_tensors)
    header_buf += struct.pack("<Q", n_kv)

    # KV pairs
    for key, val in metadata:
        if isinstance(val, str):
            encoded_key = key.encode("utf-8")
            header_buf += struct.pack("<Q", len(encoded_key))
            header_buf += encoded_key
            header_buf += struct.pack("<I", GGUF_TYPE_STRING)
            encoded_val = val.encode("utf-8")
            header_buf += struct.pack("<Q", len(encoded_val))
            header_buf += encoded_val
        elif isinstance(val, int):
            encoded_key = key.encode("utf-8")
            header_buf += struct.pack("<Q", len(encoded_key))
            header_buf += encoded_key
            header_buf += struct.pack("<I", GGUF_TYPE_UINT32)
            header_buf += struct.pack("<I", val)

    # Tensor info entries
    tensor_data_list = []
    current_offset = 0
    for name, np_data in tensors:
        raw = np_data.astype(np.float32).tobytes()
        tensor_data_list.append(raw)

        # Tensor info: name + n_dims + dims[n_dims] + type + offset
        encoded_name = name.encode("utf-8")
        header_buf += struct.pack("<Q", len(encoded_name))
        header_buf += encoded_name

        n_dims = len(np_data.shape)
        header_buf += struct.pack("<I", n_dims)
        for dim in np_data.shape:
            header_buf += struct.pack("<Q", dim)

        header_buf += struct.pack("<I", GGML_TYPE_F32)  # always F32
        header_buf += struct.pack("<Q", current_offset)

        # Align offset
        size = len(raw)
        current_offset += size
        # Align to GGUF_DEFAULT_ALIGNMENT
        aligned = (current_offset + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
        current_offset = aligned

    # Align header to data section
    header_size = len(header_buf)
    data_start = (header_size + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
    padding = data_start - header_size

    # Write file
    output = Path(output_path)
    with open(output, "wb") as f:
        f.write(header_buf)
        f.write(b"\x00" * padding)

        # Write tensor data with alignment
        for i, raw in enumerate(tensor_data_list):
            f.write(raw)
            # Pad to alignment
            pos = f.tell() - data_start
            aligned_pos = (pos + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
            if aligned_pos > pos:
                f.write(b"\x00" * (aligned_pos - pos))

    total_size = output.stat().st_size
    print(f"Written: {output} ({total_size / 1024 / 1024:.1f} MB)")
    print(f"  {n_tensors} tensors, {n_kv} metadata entries, GGUF v{GGUF_VERSION}")


if __name__ == "__main__":
    main()
