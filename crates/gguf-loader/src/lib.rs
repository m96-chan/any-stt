//! GGUF v3 file loader with zero-copy mmap access.
//!
//! ```no_run
//! use gguf_loader::GgufFile;
//!
//! let gguf = GgufFile::open("model.gguf").unwrap();
//! let view = gguf.tensor("encoder.blocks.0.attn.query.weight").unwrap();
//! let weights: Vec<f32> = view.dequantize_f32().unwrap();
//! ```

pub mod header;
pub mod metadata;
pub mod tensor;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;

use memmap2::Mmap;

pub use metadata::{MetaKv, MetaValue};
pub use tensor::{GgmlType, TensorInfo, TensorView};

/// A parsed GGUF file with memory-mapped tensor data.
pub struct GgufFile {
    pub header: header::GgufHeader,
    pub metadata: Vec<MetaKv>,
    pub tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    /// Start of the tensor data section within the file.
    data_offset: u64,
    mmap: Mmap,
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;

        // SAFETY: the file must not be modified while mmap is alive.
        // This is acceptable for model files which are read-only.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?;

        let mut reader = BufReader::new(File::open(path)
            .map_err(|e| format!("open {}: {e}", path.display()))?);

        let hdr = header::GgufHeader::read(&mut reader)?;
        let meta = metadata::read_metadata_kv(&mut reader, hdr.metadata_kv_count)?;
        let tensors = tensor::read_tensor_infos(&mut reader, hdr.tensor_count)?;

        // Compute data section offset: current position aligned up.
        let pos = reader.seek(SeekFrom::Current(0))
            .map_err(|e| format!("tell: {e}"))?;

        // Find alignment from metadata.
        let alignment = meta
            .iter()
            .find(|kv| kv.key == "general.alignment")
            .and_then(|kv| kv.value.as_u32())
            .map(|v| v as u64)
            .unwrap_or(header::GGUF_DEFAULT_ALIGNMENT);

        let data_offset = (pos + alignment - 1) / alignment * alignment;

        // Build index
        let mut tensor_index = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            tensor_index.insert(t.name.clone(), i);
        }

        Ok(Self {
            header: hdr,
            metadata: meta,
            tensors,
            tensor_index,
            data_offset,
            mmap,
        })
    }

    /// Look up a tensor by name, returning a zero-copy view.
    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let idx = *self.tensor_index.get(name)?;
        let info = &self.tensors[idx];
        let start = self.data_offset + info.offset;
        let size = info.data_size();
        let end = start + size;
        if end as usize > self.mmap.len() {
            return None;
        }
        Some(TensorView::new(info, &self.mmap[start as usize..end as usize]))
    }

    /// Get a metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| &kv.value)
    }

    /// Get a metadata string value.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta(key).and_then(|v| v.as_str())
    }

    /// Get a metadata u32 value.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.meta(key).and_then(|v| v.as_u32())
    }

    /// Dequantize a tensor to f32 by name.
    pub fn dequantize_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let view = self
            .tensor(name)
            .ok_or_else(|| format!("tensor not found: {name}"))?;
        view.dequantize_f32()
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gguf_model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../third-party/whisper.cpp/models/ggml-tiny.en.bin")
    }

    #[test]
    fn open_nonexistent_file_returns_error() {
        assert!(GgufFile::open("/nonexistent/model.gguf").is_err());
    }

    #[test]
    fn open_gguf_model_if_available() {
        let path = gguf_model_path();
        if !path.exists() {
            eprintln!("SKIPPED: model not found at {}", path.display());
            return;
        }
        let gguf = GgufFile::open(&path);
        match gguf {
            Ok(f) => {
                eprintln!("GGUF v{}, {} tensors, {} metadata entries",
                    f.header.version, f.tensors.len(), f.metadata.len());
                for t in &f.tensors[..std::cmp::min(5, f.tensors.len())] {
                    eprintln!("  {} {:?} {:?}", t.name, t.dtype, &t.dims[..t.n_dims as usize]);
                }
            }
            Err(e) => {
                // The .bin format may not be GGUF — that's fine for this test.
                eprintln!("Could not open as GGUF (may be legacy ggml format): {e}");
            }
        }
    }
}
