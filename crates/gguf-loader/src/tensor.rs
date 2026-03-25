//! GGUF tensor info and data access.

use std::io::{Read, Seek};

use crate::header::*;

/// ggml tensor data types (matches ggml_type enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self, String> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            16 => Ok(Self::Iq2Xxs),
            17 => Ok(Self::Iq2Xs),
            18 => Ok(Self::Iq3Xxs),
            19 => Ok(Self::Iq1S),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::Iq1M),
            30 => Ok(Self::Bf16),
            _ => Err(format!("unknown ggml_type: {v}")),
        }
    }

    /// Block size for quantized types (number of elements per block).
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 1,
            Self::F16 | Self::Bf16 | Self::I16 => 1,
            Self::I8 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
            _ => 32, // reasonable default for IQ types
        }
    }

    /// Byte size per block.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::F64 | Self::I64 => 8,
            Self::Q4_0 => 18,  // 32 * 4 bits / 8 + 2 (scale f16)
            Self::Q4_1 => 20,  // 32 * 4 bits / 8 + 2 (scale) + 2 (min)
            Self::Q5_0 => 22,  // 32 * 5 bits / 8 + 2 (scale)  (actually 2 + 4 + 16)
            Self::Q5_1 => 24,  // 32 * 5 bits / 8 + 2 (scale) + 2 (min)
            Self::Q8_0 => 34,  // 32 * 8 bits / 8 + 2 (scale f16)
            Self::Q8_1 => 40,  // 32 * 8 bits / 8 + 4 (scale f32) + 4 (min f32)
            _ => 0, // unsupported for now
        }
    }

    /// Total byte size for `n_elements` of this type.
    pub fn data_size(self, n_elements: u64) -> u64 {
        let bs = self.block_size() as u64;
        let ts = self.type_size() as u64;
        if bs == 0 || ts == 0 {
            return 0;
        }
        (n_elements + bs - 1) / bs * ts
    }
}

/// Parsed tensor info from the GGUF header (not the data itself).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub dtype: GgmlType,
    /// Offset from the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements.
    pub fn n_elements(&self) -> u64 {
        let mut n = 1u64;
        for i in 0..self.n_dims as usize {
            n *= self.dims[i];
        }
        n
    }

    /// Total byte size of the tensor data.
    pub fn data_size(&self) -> u64 {
        self.dtype.data_size(self.n_elements())
    }
}

/// Read tensor info entries from the reader.
pub fn read_tensor_infos<R: Read + Seek>(
    r: &mut R,
    count: u64,
) -> Result<Vec<TensorInfo>, String> {
    let mut infos = Vec::with_capacity(count as usize);
    for i in 0..count {
        let name = read_string(r).map_err(|e| format!("tensor {i} name: {e}"))?;
        let n_dims = read_u32(r).map_err(|e| format!("tensor {i} n_dims: {e}"))?;
        if n_dims > 4 {
            return Err(format!("tensor {i} ({name}): n_dims={n_dims} > 4"));
        }
        let mut dims = [1u64; 4];
        for d in dims.iter_mut().take(n_dims as usize) {
            *d = read_u64(r).map_err(|e| format!("tensor {i} dim: {e}"))?;
        }
        let dtype_u32 = read_u32(r).map_err(|e| format!("tensor {i} type: {e}"))?;
        let dtype = GgmlType::from_u32(dtype_u32)
            .map_err(|e| format!("tensor {i} ({name}): {e}"))?;
        let offset = read_u64(r).map_err(|e| format!("tensor {i} offset: {e}"))?;

        infos.push(TensorInfo {
            name,
            n_dims,
            dims,
            dtype,
            offset,
        });
    }
    Ok(infos)
}

/// A zero-copy view of tensor data in a memory-mapped file.
pub struct TensorView<'a> {
    pub info: &'a TensorInfo,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn new(info: &'a TensorInfo, data: &'a [u8]) -> Self {
        Self { info, data }
    }

    /// Raw bytes of the tensor.
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Interpret as f32 slice (only valid for F32 tensors).
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.info.dtype != GgmlType::F32 {
            return None;
        }
        let n = self.info.n_elements() as usize;
        if self.data.len() < n * 4 {
            return None;
        }
        // SAFETY: data is aligned via mmap, F32 type checked, length verified.
        // memmap2 returns page-aligned data, which satisfies f32 alignment.
        let ptr = self.data.as_ptr() as *const f32;
        Some(unsafe { std::slice::from_raw_parts(ptr, n) })
    }

    /// Dequantize to f32. Works for F32, F16, Q8_0.
    pub fn dequantize_f32(&self) -> Result<Vec<f32>, String> {
        let n = self.info.n_elements() as usize;
        match self.info.dtype {
            GgmlType::F32 => {
                self.as_f32()
                    .map(|s| s.to_vec())
                    .ok_or_else(|| "data too short for F32".into())
            }
            GgmlType::F16 => dequantize_f16(self.data, n),
            GgmlType::Q8_0 => dequantize_q8_0(self.data, n),
            GgmlType::Q4_0 => dequantize_q4_0(self.data, n),
            GgmlType::Q5_0 => dequantize_q5_0(self.data, n),
            other => Err(format!("dequantize not implemented for {other:?}")),
        }
    }
}

/// Dequantize f16 data to f32.
fn dequantize_f16(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    if data.len() < n * 2 {
        return Err("data too short for F16".into());
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        out.push(f16_to_f32(bits));
    }
    Ok(out)
}

/// Dequantize Q8_0: block of 32 int8 values with an f16 scale factor.
/// Block layout: f16 scale (2 bytes) + 32 x int8 (32 bytes) = 34 bytes per block.
fn dequantize_q8_0(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let block_size = 32usize;
    let bytes_per_block = 34usize;
    let n_blocks = (n + block_size - 1) / block_size;
    if data.len() < n_blocks * bytes_per_block {
        return Err(format!(
            "Q8_0: data too short: {} < {}",
            data.len(),
            n_blocks * bytes_per_block
        ));
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * bytes_per_block..];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = f16_to_f32(scale_bits);
        let remaining = std::cmp::min(block_size, n - b * block_size);
        for i in 0..remaining {
            let q = block[2 + i] as i8;
            out.push(q as f32 * scale);
        }
    }
    Ok(out)
}

/// Dequantize Q4_0: block of 32 values, each stored as 4 bits + f16 scale.
/// Block layout: f16 scale (2 bytes) + 16 bytes (32 x 4-bit) = 18 bytes per block.
fn dequantize_q4_0(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let block_size = 32usize;
    let bytes_per_block = 18usize;
    let n_blocks = (n + block_size - 1) / block_size;
    if data.len() < n_blocks * bytes_per_block {
        return Err("Q4_0: data too short".into());
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * bytes_per_block..];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = f16_to_f32(scale_bits);
        let remaining = std::cmp::min(block_size, n - b * block_size);
        for i in 0..remaining {
            let byte = block[2 + i / 2];
            let nibble = if i % 2 == 0 {
                (byte & 0x0F) as i8 - 8
            } else {
                ((byte >> 4) & 0x0F) as i8 - 8
            };
            out.push(nibble as f32 * scale);
        }
    }
    Ok(out)
}

/// Dequantize Q5_0: block of 32 values, 5 bits each + f16 scale.
/// Block layout: f16 scale (2 bytes) + 4 bytes high bits + 16 bytes low nibbles = 22 bytes.
fn dequantize_q5_0(data: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let block_size = 32usize;
    let bytes_per_block = 22usize;
    let n_blocks = (n + block_size - 1) / block_size;
    if data.len() < n_blocks * bytes_per_block {
        return Err("Q5_0: data too short".into());
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let block = &data[b * bytes_per_block..];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = f16_to_f32(scale_bits);
        // High bits: 4 bytes = 32 bits, one per element
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let remaining = std::cmp::min(block_size, n - b * block_size);
        for i in 0..remaining {
            let byte = block[6 + i / 2];
            let lo = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            let hi = ((qh >> i) & 1) as u8;
            let val = (lo | (hi << 4)) as i8 - 16;
            out.push(val as f32 * scale);
        }
    }
    Ok(out)
}

/// Convert IEEE 754 half-precision (binary16) to f32.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut val = frac as f32 / 1024.0;
            val *= 1.0 / 16384.0; // 2^-14
            if sign == 1 {
                -val
            } else {
                val
            }
        }
    } else if exp == 31 {
        if frac == 0 {
            // Infinity
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            // NaN
            f32::from_bits((sign << 31) | 0x7FC00000)
        }
    } else {
        // Normal
        let new_exp = (exp as i32 - 15 + 127) as u32;
        let new_frac = frac << 13;
        f32::from_bits((sign << 31) | (new_exp << 23) | new_frac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- f16_to_f32 ---

    #[test]
    fn f16_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // Negative zero
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn f16_normal_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert!((f16_to_f32(0x4200) - 3.0).abs() < 0.001);
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 0.001);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 0.001);
    }

    #[test]
    fn f16_infinity_and_nan() {
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00).is_sign_positive());
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00).is_sign_negative());
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn f16_subnormal() {
        // Smallest positive subnormal: 0x0001 = 2^-24 ≈ 5.96e-8
        let val = f16_to_f32(0x0001);
        assert!(val > 0.0);
        assert!(val < 1e-6);
    }

    // --- GgmlType ---

    #[test]
    fn ggml_type_from_u32_known() {
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2).unwrap(), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_u32(30).unwrap(), GgmlType::Bf16);
    }

    #[test]
    fn ggml_type_from_u32_unknown() {
        assert!(GgmlType::from_u32(4).is_err()); // gap in enum
        assert!(GgmlType::from_u32(5).is_err());
        assert!(GgmlType::from_u32(15).is_err());
        assert!(GgmlType::from_u32(999).is_err());
    }

    #[test]
    fn ggml_type_block_size() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F16.block_size(), 1);
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q6K.block_size(), 256);
    }

    #[test]
    fn ggml_type_size() {
        assert_eq!(GgmlType::F32.type_size(), 4);
        assert_eq!(GgmlType::F16.type_size(), 2);
        assert_eq!(GgmlType::Q4_0.type_size(), 18);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);
        assert_eq!(GgmlType::F64.type_size(), 8);
        assert_eq!(GgmlType::I8.type_size(), 1);
    }

    #[test]
    fn ggml_type_data_size() {
        // F32: 1 element per block, 4 bytes each
        assert_eq!(GgmlType::F32.data_size(100), 400);
        // F16: 1 element per block, 2 bytes each
        assert_eq!(GgmlType::F16.data_size(100), 200);
        // Q4_0: 32 elements per block, 18 bytes each
        assert_eq!(GgmlType::Q4_0.data_size(32), 18);
        assert_eq!(GgmlType::Q4_0.data_size(64), 36);
        // Q8_0: 32 elements per block, 34 bytes each
        assert_eq!(GgmlType::Q8_0.data_size(32), 34);
        // Partial blocks round up
        assert_eq!(GgmlType::Q4_0.data_size(33), 36); // 2 blocks
    }

    // --- TensorInfo ---

    #[test]
    fn tensor_info_n_elements() {
        let info = TensorInfo {
            name: "test".into(),
            n_dims: 3,
            dims: [10, 20, 30, 1],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(info.n_elements(), 6000);
    }

    #[test]
    fn tensor_info_n_elements_1d() {
        let info = TensorInfo {
            name: "bias".into(),
            n_dims: 1,
            dims: [384, 1, 1, 1],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(info.n_elements(), 384);
    }

    #[test]
    fn tensor_info_data_size() {
        let info = TensorInfo {
            name: "weight".into(),
            n_dims: 2,
            dims: [384, 80, 1, 1],
            dtype: GgmlType::F16,
            offset: 0,
        };
        // 384 * 80 = 30720 elements, 2 bytes each
        assert_eq!(info.data_size(), 61440);
    }

    // --- Dequantization ---

    #[test]
    fn dequantize_f16_simple() {
        // Two f16 values: 1.0 (0x3C00) and -1.0 (0xBC00)
        let data: Vec<u8> = vec![0x00, 0x3C, 0x00, 0xBC];
        let result = dequantize_f16(&data, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn dequantize_f16_too_short() {
        let data: Vec<u8> = vec![0x00, 0x3C];
        assert!(dequantize_f16(&data, 2).is_err());
    }

    #[test]
    fn dequantize_q8_0_one_block() {
        // One Q8_0 block: f16 scale (1.0 = 0x3C00) + 32 int8 values
        let mut data = vec![0x00u8, 0x3C]; // scale = 1.0
        for i in 0..32i8 {
            data.push(i as u8);
        }
        let result = dequantize_q8_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);
        // q[0] = 0 * 1.0 = 0.0
        assert!((result[0] - 0.0).abs() < 0.001);
        // q[1] = 1 * 1.0 = 1.0
        assert!((result[1] - 1.0).abs() < 0.001);
        // q[10] = 10 * 1.0 = 10.0
        assert!((result[10] - 10.0).abs() < 0.001);
    }

    #[test]
    fn dequantize_q8_0_too_short() {
        let data = vec![0u8; 10]; // too small for a block
        assert!(dequantize_q8_0(&data, 32).is_err());
    }

    #[test]
    fn dequantize_q4_0_one_block() {
        // One Q4_0 block: f16 scale (1.0 = 0x3C00) + 16 bytes of 4-bit nibbles
        let mut data = vec![0x00u8, 0x3C]; // scale = 1.0
        // All nibbles = 0x00 → low=0, high=0 → both produce (0 - 8) * 1.0 = -8.0
        data.extend_from_slice(&[0x00; 16]);
        let result = dequantize_q4_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);
        // Each nibble is 0, so value = (0 - 8) * 1.0 = -8.0
        for v in &result {
            assert!((*v - (-8.0)).abs() < 0.001);
        }
    }

    #[test]
    fn dequantize_q4_0_varied_nibbles() {
        let mut data = vec![0x00u8, 0x3C]; // scale = 1.0
        // First byte: low=8 (0x8), high=8 (0x8) → 0x88
        // value for nibble 8: (8 - 8) * 1.0 = 0.0
        data.push(0x88);
        data.extend_from_slice(&[0x00; 15]); // rest are 0
        let result = dequantize_q4_0(&data, 32).unwrap();
        // First two elements: nibble=8 → (8-8)*1.0 = 0.0
        assert!((result[0] - 0.0).abs() < 0.001);
        assert!((result[1] - 0.0).abs() < 0.001);
    }

    #[test]
    fn dequantize_q5_0_one_block() {
        // Q5_0 block: scale(2) + high_bits(4) + nibbles(16) = 22 bytes
        let mut data = vec![0x00u8, 0x3C]; // scale = 1.0
        data.extend_from_slice(&[0x00; 4]); // high bits all 0
        // All nibbles = 0 → lo=0, hi=0, val = (0|0<<4) - 16 = -16
        data.extend_from_slice(&[0x00; 16]);
        let result = dequantize_q5_0(&data, 32).unwrap();
        assert_eq!(result.len(), 32);
        for v in &result {
            assert!((*v - (-16.0)).abs() < 0.001);
        }
    }

    // --- TensorView ---

    #[test]
    fn tensor_view_as_f32() {
        let info = TensorInfo {
            name: "test".into(),
            n_dims: 1,
            dims: [3, 1, 1, 1],
            dtype: GgmlType::F32,
            offset: 0,
        };
        let values: [f32; 3] = [1.0, 2.0, 3.0];
        let data: &[u8] = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, 12)
        };
        let view = TensorView::new(&info, data);
        let f32s = view.as_f32().unwrap();
        assert_eq!(f32s, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn tensor_view_as_f32_wrong_type() {
        let info = TensorInfo {
            name: "test".into(),
            n_dims: 1,
            dims: [3, 1, 1, 1],
            dtype: GgmlType::F16,
            offset: 0,
        };
        let view = TensorView::new(&info, &[0u8; 6]);
        assert!(view.as_f32().is_none());
    }
}
