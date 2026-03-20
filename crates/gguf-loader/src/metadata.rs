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

fn read_array_value<R: Read + Seek>(r: &mut R) -> Result<MetaValue, String> {
    let elem_type_id = read_u32(r)?;
    let elem_type = GgufMetaType::from_u32(elem_type_id)?;
    let len = read_u64(r)? as usize;

    if len > 16 * 1024 * 1024 {
        return Err(format!("array too large: {len}"));
    }

    match elem_type {
        GgufMetaType::Uint8 => {
            let mut v = vec![0u8; len];
            for item in v.iter_mut() {
                *item = read_u8(r)?;
            }
            Ok(MetaValue::ArrayUint8(v))
        }
        GgufMetaType::Int8 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_i8(r)?);
            }
            Ok(MetaValue::ArrayInt8(v))
        }
        GgufMetaType::Uint16 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_u16(r)?);
            }
            Ok(MetaValue::ArrayUint16(v))
        }
        GgufMetaType::Int16 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_i16(r)?);
            }
            Ok(MetaValue::ArrayInt16(v))
        }
        GgufMetaType::Uint32 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_u32(r)?);
            }
            Ok(MetaValue::ArrayUint32(v))
        }
        GgufMetaType::Int32 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_i32(r)?);
            }
            Ok(MetaValue::ArrayInt32(v))
        }
        GgufMetaType::Float32 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_f32(r)?);
            }
            Ok(MetaValue::ArrayFloat32(v))
        }
        GgufMetaType::Bool => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_bool(r)?);
            }
            Ok(MetaValue::ArrayBool(v))
        }
        GgufMetaType::String => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_string(r)?);
            }
            Ok(MetaValue::ArrayString(v))
        }
        GgufMetaType::Uint64 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_u64(r)?);
            }
            Ok(MetaValue::ArrayUint64(v))
        }
        GgufMetaType::Int64 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_i64(r)?);
            }
            Ok(MetaValue::ArrayInt64(v))
        }
        GgufMetaType::Float64 => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_f64(r)?);
            }
            Ok(MetaValue::ArrayFloat64(v))
        }
        GgufMetaType::Array => Err("nested arrays not supported".into()),
    }
}
