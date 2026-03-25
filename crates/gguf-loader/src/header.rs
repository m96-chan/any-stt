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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_valid_header(version: u32, tensors: u64, kvs: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&tensors.to_le_bytes());
        buf.extend_from_slice(&kvs.to_le_bytes());
        buf
    }

    #[test]
    fn read_valid_header_v3() {
        let data = make_valid_header(3, 100, 5);
        let mut cursor = Cursor::new(data);
        let hdr = GgufHeader::read(&mut cursor).unwrap();
        assert_eq!(hdr.version, 3);
        assert_eq!(hdr.tensor_count, 100);
        assert_eq!(hdr.metadata_kv_count, 5);
    }

    #[test]
    fn read_valid_header_v2() {
        let data = make_valid_header(2, 50, 10);
        let mut cursor = Cursor::new(data);
        let hdr = GgufHeader::read(&mut cursor).unwrap();
        assert_eq!(hdr.version, 2);
    }

    #[test]
    fn reject_invalid_magic() {
        let mut data = make_valid_header(3, 0, 0);
        data[0] = b'X'; // corrupt magic
        let mut cursor = Cursor::new(data);
        let result = GgufHeader::read(&mut cursor);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid magic"));
    }

    #[test]
    fn reject_unsupported_version() {
        let data = make_valid_header(1, 0, 0); // version 1 is too old
        let mut cursor = Cursor::new(data);
        let result = GgufHeader::read(&mut cursor);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsupported"));
    }

    #[test]
    fn reject_future_version() {
        let data = make_valid_header(99, 0, 0);
        let mut cursor = Cursor::new(data);
        assert!(GgufHeader::read(&mut cursor).is_err());
    }

    #[test]
    fn reject_truncated_header() {
        let data = b"GGU".to_vec(); // too short
        let mut cursor = Cursor::new(data);
        assert!(GgufHeader::read(&mut cursor).is_err());
    }

    // --- Primitive readers ---

    #[test]
    fn read_primitives() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&42u8.to_le_bytes());
        buf.extend_from_slice(&(-1i8 as u8).to_le_bytes());
        buf.extend_from_slice(&1000u16.to_le_bytes());
        buf.extend_from_slice(&(-500i16 as u16).to_le_bytes());
        buf.extend_from_slice(&100000u32.to_le_bytes());
        buf.extend_from_slice(&(-999i32 as u32).to_le_bytes());
        buf.extend_from_slice(&99999999u64.to_le_bytes());
        buf.extend_from_slice(&(-1i64 as u64).to_le_bytes());
        buf.extend_from_slice(&3.14f32.to_le_bytes());
        buf.extend_from_slice(&2.718f64.to_le_bytes());

        let mut c = Cursor::new(buf);
        assert_eq!(read_u8(&mut c).unwrap(), 42);
        assert_eq!(read_i8(&mut c).unwrap(), -1);
        assert_eq!(read_u16(&mut c).unwrap(), 1000);
        assert_eq!(read_i16(&mut c).unwrap(), -500);
        assert_eq!(read_u32(&mut c).unwrap(), 100000);
        assert_eq!(read_i32(&mut c).unwrap(), -999);
        assert_eq!(read_u64(&mut c).unwrap(), 99999999);
        assert_eq!(read_i64(&mut c).unwrap(), -1);
        assert!((read_f32(&mut c).unwrap() - 3.14).abs() < 0.001);
        assert!((read_f64(&mut c).unwrap() - 2.718).abs() < 0.001);
    }

    #[test]
    fn read_bool_values() {
        let mut c = Cursor::new(vec![0u8, 1u8, 255u8]);
        assert!(!read_bool(&mut c).unwrap());
        assert!(read_bool(&mut c).unwrap());
        assert!(read_bool(&mut c).unwrap()); // non-zero is true
    }

    #[test]
    fn read_string_valid() {
        let s = "hello";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        let mut c = Cursor::new(buf);
        assert_eq!(read_string(&mut c).unwrap(), "hello");
    }

    #[test]
    fn read_string_empty() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u64.to_le_bytes());
        let mut c = Cursor::new(buf);
        assert_eq!(read_string(&mut c).unwrap(), "");
    }

    #[test]
    fn read_string_too_long() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(2_000_000u64).to_le_bytes());
        let mut c = Cursor::new(buf);
        assert!(read_string(&mut c).is_err());
    }
}
