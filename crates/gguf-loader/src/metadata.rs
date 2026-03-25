//! GGUF metadata key-value parsing.

use std::io::{Read, Seek};

use crate::header::*;

/// GGUF metadata value types.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufMetaType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufMetaType {
    pub fn from_u32(v: u32) -> Result<Self, String> {
        match v {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(format!("unknown GGUF metadata type: {v}")),
        }
    }
}

/// A parsed metadata value.
#[derive(Debug, Clone)]
pub enum MetaValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    ArrayUint8(Vec<u8>),
    ArrayInt8(Vec<i8>),
    ArrayUint16(Vec<u16>),
    ArrayInt16(Vec<i16>),
    ArrayUint32(Vec<u32>),
    ArrayInt32(Vec<i32>),
    ArrayFloat32(Vec<f32>),
    ArrayBool(Vec<bool>),
    ArrayString(Vec<String>),
    ArrayUint64(Vec<u64>),
    ArrayInt64(Vec<i64>),
    ArrayFloat64(Vec<f64>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::Uint32(v) => Some(*v),
            MetaValue::Uint8(v) => Some(*v as u32),
            MetaValue::Uint16(v) => Some(*v as u32),
            MetaValue::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::Uint64(v) => Some(*v),
            MetaValue::Uint32(v) => Some(*v as u64),
            _ => None,
        }
    }
}

/// A metadata key-value entry.
#[derive(Debug, Clone)]
pub struct MetaKv {
    pub key: String,
    pub value: MetaValue,
}

/// Read all metadata KV pairs from the reader (positioned right after the header).
pub fn read_metadata_kv<R: Read + Seek>(
    r: &mut R,
    count: u64,
) -> Result<Vec<MetaKv>, String> {
    let mut kvs = Vec::with_capacity(count as usize);
    for i in 0..count {
        let key = read_string(r).map_err(|e| format!("kv {i} key: {e}"))?;
        let type_id = read_u32(r).map_err(|e| format!("kv {i} type: {e}"))?;
        let meta_type = GgufMetaType::from_u32(type_id)?;
        let value = read_meta_value(r, meta_type)
            .map_err(|e| format!("kv {i} ({key}): {e}"))?;
        kvs.push(MetaKv { key, value });
    }
    Ok(kvs)
}

fn read_meta_value<R: Read + Seek>(
    r: &mut R,
    meta_type: GgufMetaType,
) -> Result<MetaValue, String> {
    match meta_type {
        GgufMetaType::Uint8 => Ok(MetaValue::Uint8(read_u8(r)?)),
        GgufMetaType::Int8 => Ok(MetaValue::Int8(read_i8(r)?)),
        GgufMetaType::Uint16 => Ok(MetaValue::Uint16(read_u16(r)?)),
        GgufMetaType::Int16 => Ok(MetaValue::Int16(read_i16(r)?)),
        GgufMetaType::Uint32 => Ok(MetaValue::Uint32(read_u32(r)?)),
        GgufMetaType::Int32 => Ok(MetaValue::Int32(read_i32(r)?)),
        GgufMetaType::Float32 => Ok(MetaValue::Float32(read_f32(r)?)),
        GgufMetaType::Bool => Ok(MetaValue::Bool(read_bool(r)?)),
        GgufMetaType::String => Ok(MetaValue::String(read_string(r)?)),
        GgufMetaType::Uint64 => Ok(MetaValue::Uint64(read_u64(r)?)),
        GgufMetaType::Int64 => Ok(MetaValue::Int64(read_i64(r)?)),
        GgufMetaType::Float64 => Ok(MetaValue::Float64(read_f64(r)?)),
        GgufMetaType::Array => read_array_value(r),
    }
}

/// Read `len` elements using `read_fn`, collecting into a Vec.
fn read_array_elems<R: Read + Seek, T>(
    r: &mut R,
    len: usize,
    read_fn: fn(&mut R) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(read_fn(r)?);
    }
    Ok(v)
}

fn read_array_value<R: Read + Seek>(r: &mut R) -> Result<MetaValue, String> {
    let elem_type_id = read_u32(r)?;
    let elem_type = GgufMetaType::from_u32(elem_type_id)?;
    let len = read_u64(r)? as usize;

    if len > 16 * 1024 * 1024 {
        return Err(format!("array too large: {len}"));
    }

    match elem_type {
        GgufMetaType::Uint8 => Ok(MetaValue::ArrayUint8(read_array_elems(r, len, read_u8)?)),
        GgufMetaType::Int8 => Ok(MetaValue::ArrayInt8(read_array_elems(r, len, read_i8)?)),
        GgufMetaType::Uint16 => Ok(MetaValue::ArrayUint16(read_array_elems(r, len, read_u16)?)),
        GgufMetaType::Int16 => Ok(MetaValue::ArrayInt16(read_array_elems(r, len, read_i16)?)),
        GgufMetaType::Uint32 => Ok(MetaValue::ArrayUint32(read_array_elems(r, len, read_u32)?)),
        GgufMetaType::Int32 => Ok(MetaValue::ArrayInt32(read_array_elems(r, len, read_i32)?)),
        GgufMetaType::Float32 => Ok(MetaValue::ArrayFloat32(read_array_elems(r, len, read_f32)?)),
        GgufMetaType::Bool => Ok(MetaValue::ArrayBool(read_array_elems(r, len, read_bool)?)),
        GgufMetaType::String => Ok(MetaValue::ArrayString(read_array_elems(r, len, read_string)?)),
        GgufMetaType::Uint64 => Ok(MetaValue::ArrayUint64(read_array_elems(r, len, read_u64)?)),
        GgufMetaType::Int64 => Ok(MetaValue::ArrayInt64(read_array_elems(r, len, read_i64)?)),
        GgufMetaType::Float64 => Ok(MetaValue::ArrayFloat64(read_array_elems(r, len, read_f64)?)),
        GgufMetaType::Array => Err("nested arrays not supported".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_type_from_u32_all_valid() {
        assert_eq!(GgufMetaType::from_u32(0).unwrap(), GgufMetaType::Uint8);
        assert_eq!(GgufMetaType::from_u32(1).unwrap(), GgufMetaType::Int8);
        assert_eq!(GgufMetaType::from_u32(6).unwrap(), GgufMetaType::Float32);
        assert_eq!(GgufMetaType::from_u32(8).unwrap(), GgufMetaType::String);
        assert_eq!(GgufMetaType::from_u32(9).unwrap(), GgufMetaType::Array);
        assert_eq!(GgufMetaType::from_u32(12).unwrap(), GgufMetaType::Float64);
    }

    #[test]
    fn meta_type_from_u32_invalid() {
        assert!(GgufMetaType::from_u32(13).is_err());
        assert!(GgufMetaType::from_u32(255).is_err());
    }

    #[test]
    fn meta_value_as_u32() {
        assert_eq!(MetaValue::Uint32(42).as_u32(), Some(42));
        assert_eq!(MetaValue::Uint8(5).as_u32(), Some(5));
        assert_eq!(MetaValue::Uint16(1000).as_u32(), Some(1000));
        assert_eq!(MetaValue::Int32(10).as_u32(), Some(10));
        assert_eq!(MetaValue::Float32(1.0).as_u32(), None);
        assert_eq!(MetaValue::String("test".into()).as_u32(), None);
    }

    #[test]
    fn meta_value_as_f32() {
        assert_eq!(MetaValue::Float32(3.14).as_f32(), Some(3.14));
        assert_eq!(MetaValue::Uint32(1).as_f32(), None);
    }

    #[test]
    fn meta_value_as_str() {
        let v = MetaValue::String("hello".into());
        assert_eq!(v.as_str(), Some("hello"));
        assert_eq!(MetaValue::Uint32(1).as_str(), None);
    }

    #[test]
    fn meta_value_as_u64() {
        assert_eq!(MetaValue::Uint64(999).as_u64(), Some(999));
        assert_eq!(MetaValue::Uint32(42).as_u64(), Some(42));
        assert_eq!(MetaValue::Float32(1.0).as_u64(), None);
    }
}
