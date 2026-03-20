//! GGUF v3 header parsing.

use std::io::{self, Read, Seek, SeekFrom};

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const GGUF_VERSION_MIN: u32 = 2;
pub const GGUF_VERSION_MAX: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Parsed GGUF file header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// Read the GGUF header from the start of a reader.
    pub fn read<R: Read + Seek>(r: &mut R) -> Result<Self, String> {
        r.seek(SeekFrom::Start(0)).map_err(|e| format!("seek: {e}"))?;

        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)
            .map_err(|e| format!("read magic: {e}"))?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "invalid magic: {:?}, expected {:?}",
                magic, GGUF_MAGIC
            ));
        }

        // Version
        let version = read_u32(r)?;
        if version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX {
            return Err(format!(
                "unsupported GGUF version {version} (supported: {GGUF_VERSION_MIN}..={GGUF_VERSION_MAX})"
            ));
        }

        // Tensor count and KV count
        let tensor_count = read_u64(r)?;
        let metadata_kv_count = read_u64(r)?;

        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

// Little-endian readers

pub(crate) fn read_u8<R: Read>(r: &mut R) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(|e| format!("read u8: {e}"))?;
    Ok(buf[0])
}

pub(crate) fn read_i8<R: Read>(r: &mut R) -> Result<i8, String> {
    Ok(read_u8(r)? as i8)
}

pub(crate) fn read_u16<R: Read>(r: &mut R) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(|e| format!("read u16: {e}"))?;
    Ok(u16::from_le_bytes(buf))
}

pub(crate) fn read_i16<R: Read>(r: &mut R) -> Result<i16, String> {
    Ok(read_u16(r)? as i16)
}

pub(crate) fn read_u32<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn read_i32<R: Read>(r: &mut R) -> Result<i32, String> {
    Ok(read_u32(r)? as i32)
}

pub(crate) fn read_u64<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| format!("read u64: {e}"))?;
    Ok(u64::from_le_bytes(buf))
}

pub(crate) fn read_i64<R: Read>(r: &mut R) -> Result<i64, String> {
    Ok(read_u64(r)? as i64)
}

pub(crate) fn read_f32<R: Read>(r: &mut R) -> Result<f32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| format!("read f32: {e}"))?;
    Ok(f32::from_le_bytes(buf))
}

pub(crate) fn read_f64<R: Read>(r: &mut R) -> Result<f64, String> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| format!("read f64: {e}"))?;
    Ok(f64::from_le_bytes(buf))
}

pub(crate) fn read_bool<R: Read>(r: &mut R) -> Result<bool, String> {
    Ok(read_i8(r)? != 0)
}

/// Read a GGUF string: u64 length prefix + UTF-8 bytes.
pub(crate) fn read_string<R: Read>(r: &mut R) -> Result<String, String> {
    let len = read_u64(r)? as usize;
    if len > 1024 * 1024 {
        return Err(format!("string too long: {len}"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .map_err(|e| format!("read string data ({len} bytes): {e}"))?;
    String::from_utf8(buf).map_err(|e| format!("invalid UTF-8: {e}"))
}

/// Skip n bytes.
#[allow(dead_code)]
pub(crate) fn skip<R: Read + Seek>(r: &mut R, n: u64) -> io::Result<()> {
    r.seek(SeekFrom::Current(n as i64))?;
    Ok(())
}
