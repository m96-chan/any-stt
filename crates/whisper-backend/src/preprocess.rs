//! CPU-based whisper preprocessor.
//!
//! Pipeline: audio PCM → mel spectrogram → Conv1d → GELU → Conv1d → GELU → + pos_embed
//! Output: [n_ctx, n_state] f32 tensor ready for the encoder.
//!
//! Mel spectrogram is computed via whisper.cpp FFI (whisper_pcm_to_mel) for
//! numerical accuracy. Conv1d and GELU are implemented in pure Rust (~5ms).

use gguf_loader::GgufFile;

/// CPU preprocessor that transforms audio PCM into encoder input.
pub struct Preprocessor {
    /// Conv1d layer 1: weight [n_state, 80, 3], bias [n_state]
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    /// Conv1d layer 2: weight [n_state, n_state, 3], bias [n_state]
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
    /// Positional embedding: [n_ctx, n_state]
    positional_embedding: Vec<f32>,
    /// Model dimensions
    n_state: usize,
    n_ctx: usize,
    n_mels: usize,
}

impl Preprocessor {
    /// Create preprocessor with random dummy weights (for benchmarking only).
    pub fn with_dummy_weights(n_state: usize, n_ctx: usize, n_mels: usize) -> Self {
        Self {
            conv1_weight: vec![0.01; n_state * n_mels * 3],
            conv1_bias: vec![0.0; n_state],
            conv2_weight: vec![0.01; n_state * n_state * 3],
            conv2_bias: vec![0.0; n_state],
            positional_embedding: vec![0.0; n_ctx * n_state],
            n_state,
            n_ctx,
            n_mels,
        }
    }

    /// Load preprocessor weights from a GGUF model file.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let conv1_weight = gguf.dequantize_f32("encoder.conv1.weight")?;
        let conv1_bias = gguf.dequantize_f32("encoder.conv1.bias")?;
        let conv2_weight = gguf.dequantize_f32("encoder.conv2.weight")?;
        let conv2_bias = gguf.dequantize_f32("encoder.conv2.bias")?;
        let positional_embedding = gguf.dequantize_f32("encoder.positional_embedding")?;

        // Infer dimensions from weight shapes
        let conv1_info = gguf
            .tensor("encoder.conv1.weight")
            .ok_or("missing encoder.conv1.weight")?;
        let pos_info = gguf
            .tensor("encoder.positional_embedding")
            .ok_or("missing encoder.positional_embedding")?;

        // conv1.weight shape: [n_state, n_mels, 3]
        let n_state = conv1_info.info.dims[0] as usize;
        let n_mels = conv1_info.info.dims[1] as usize;
        // positional_embedding shape: [n_ctx, n_state]
        let n_ctx = pos_info.info.dims[0] as usize;

        Ok(Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            positional_embedding,
            n_state,
            n_ctx,
            n_mels,
        })
    }

    /// Process audio PCM samples into encoder input [n_ctx, n_state].
    ///
    /// `mel`: pre-computed mel spectrogram [n_mels, n_frames].
    /// The mel should be computed by whisper.cpp's `whisper_pcm_to_mel`.
    pub fn process_mel(&self, mel: &[f32], n_frames: usize) -> Vec<f32> {
        // Conv1d(80 -> n_state, kernel=3, padding=1)
        let conv1_out = self.conv1d(
            mel,
            n_frames,
            self.n_mels,
            &self.conv1_weight,
            &self.conv1_bias,
            self.n_state,
            3,
            1, // padding
            1, // stride
        );
        let conv1_out = gelu_vec(&conv1_out);

        // Conv1d(n_state -> n_state, kernel=3, padding=1, stride=2)
        let conv1_frames = n_frames; // same padding preserves length
        let conv2_out = self.conv1d(
            &conv1_out,
            conv1_frames,
            self.n_state,
            &self.conv2_weight,
            &self.conv2_bias,
            self.n_state,
            3,
            1,
            2, // stride=2 halves the sequence length
        );
        let conv2_out = gelu_vec(&conv2_out);

        let conv2_frames = (conv1_frames + 1) / 2; // stride=2

        // Truncate/pad to n_ctx
        let frames = std::cmp::min(conv2_frames, self.n_ctx);
        let mut output = vec![0.0f32; self.n_ctx * self.n_state];

        // conv2_out is [n_state, conv2_frames] — transpose to [frames, n_state]
        // and add positional embedding
        for t in 0..frames {
            for s in 0..self.n_state {
                let conv_val = conv2_out[s * conv2_frames + t];
                let pos_val = self.positional_embedding[t * self.n_state + s];
                output[t * self.n_state + s] = conv_val + pos_val;
            }
        }

        output
    }

    /// 1D convolution: input [in_channels, length] -> output [out_channels, out_length]
    /// weight: [out_channels, in_channels, kernel_size]
    #[allow(clippy::too_many_arguments)]
    fn conv1d(
        &self,
        input: &[f32],
        length: usize,
        in_channels: usize,
        weight: &[f32],
        bias: &[f32],
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
    ) -> Vec<f32> {
        let out_length = (length + 2 * padding - kernel_size) / stride + 1;
        let mut output = vec![0.0f32; out_channels * out_length];

        for oc in 0..out_channels {
            for t in 0..out_length {
                let mut sum = bias[oc];
                let t_start = t * stride;
                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        let pos = t_start + k;
                        if pos >= padding && pos < length + padding {
                            let input_idx = ic * length + (pos - padding);
                            let weight_idx =
                                oc * in_channels * kernel_size + ic * kernel_size + k;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                output[oc * out_length + t] = sum;
            }
        }

        output
    }

    /// Get the expected n_state dimension.
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Get the expected n_ctx dimension.
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Get the expected n_mels dimension.
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }
}

/// GELU activation function: x * Φ(x) where Φ is the CDF of the standard normal.
/// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[inline]
fn gelu(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_vec(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| gelu(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_basic() {
        assert!((gelu(0.0)).abs() < 1e-6);
        assert!((gelu(1.0) - 0.8412).abs() < 0.01);
        assert!((gelu(-1.0) - (-0.1588)).abs() < 0.01);
    }

    #[test]
    fn conv1d_identity_kernel() {
        let prep = Preprocessor {
            conv1_weight: vec![],
            conv1_bias: vec![],
            conv2_weight: vec![],
            conv2_bias: vec![],
            positional_embedding: vec![],
            n_state: 0,
            n_ctx: 0,
            n_mels: 0,
        };

        // 1 input channel, 1 output channel, kernel_size=1, no padding, stride=1
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![2.0]; // kernel [1, 1, 1]
        let bias = vec![0.5];
        let output = prep.conv1d(&input, 5, 1, &weight, &bias, 1, 1, 0, 1);
        assert_eq!(output.len(), 5);
        assert!((output[0] - 2.5).abs() < 1e-6); // 1*2 + 0.5
        assert!((output[1] - 4.5).abs() < 1e-6); // 2*2 + 0.5
    }
}
