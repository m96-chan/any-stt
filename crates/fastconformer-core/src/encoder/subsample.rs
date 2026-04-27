//! Striding Conv2d subsampling stack.
//!
//! NeMo Conformer's `subsampling="striding"` does three Conv2d stages
//! (kernel=3, stride=2, pad=1) interleaved with ReLU, then a final
//! Linear that projects the flattened [intermediate, mel/8] feature map
//! down to `d_model`.
//!
//! Input shape  : `[time, n_mels]` row-major.
//! Output shape : `[time/8, d_model]` row-major.
//!
//! For typical 16 kHz / 80-mel input, the mel axis goes 80 → 40 → 20 → 10.
//! The intermediate channel count `C` is configurable but standard NeMo
//! uses `C = d_model` so the final feature map is `[d_model, time/8, 10]`,
//! which is then flattened to `[time/8, d_model * 10]` and projected by
//! the Linear back to `[time/8, d_model]`.

/// Subsampling weights.
pub struct Subsample {
    pub n_mels: usize,
    pub d_model: usize,
    /// Output mel dim after 3 stages of stride-2 (≈ n_mels / 8, ceil semantics).
    pub n_mels_out: usize,

    /// `[C, 1, 3, 3]` flattened: out_channels × in_channels × kh × kw.
    pub conv0_weight: Vec<f32>,
    pub conv0_bias: Vec<f32>,
    /// `[C, C, 3, 3]`
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    /// `[C, C, 3, 3]`
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    /// Intermediate channel count (`C`).
    pub channels: usize,

    /// Final linear: `[d_model, channels * n_mels_out]`
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
}

impl Subsample {
    /// Apply subsampling to a `[time, n_mels]` mel spectrogram.
    /// Returns a `[time_out, d_model]` flat row-major buffer plus the
    /// time count.
    pub fn forward(&self, mel: &[f32], time: usize) -> (Vec<f32>, usize) {
        debug_assert_eq!(mel.len(), time * self.n_mels);

        // Stage 0: 1 → C, mel: n_mels → n_mels/2 (ceil), time → time/2 (ceil).
        let (mut buf, t1, h1) = conv2d_stride2(
            mel,
            1,
            time,
            self.n_mels,
            self.channels,
            &self.conv0_weight,
            &self.conv0_bias,
        );
        relu(&mut buf);

        // Stage 1: C → C, half again.
        let (mut buf, t2, h2) = conv2d_stride2(
            &buf,
            self.channels,
            t1,
            h1,
            self.channels,
            &self.conv1_weight,
            &self.conv1_bias,
        );
        relu(&mut buf);

        // Stage 2: C → C, half again.
        let (buf, t3, h3) = conv2d_stride2(
            &buf,
            self.channels,
            t2,
            h2,
            self.channels,
            &self.conv2_weight,
            &self.conv2_bias,
        );

        debug_assert_eq!(h3, self.n_mels_out, "mel-axis dim mismatch after subsample");

        // Flatten: [C, T_out, mel_out] → [T_out, C * mel_out] row-major.
        // Source layout from conv2d_stride2 is `[C, T_out, mel_out]`.
        let feat_per_step = self.channels * h3;
        let mut flat = vec![0.0_f32; t3 * feat_per_step];
        for c in 0..self.channels {
            for t in 0..t3 {
                for h in 0..h3 {
                    let src = c * t3 * h3 + t * h3 + h;
                    let dst = t * feat_per_step + c * h3 + h;
                    flat[dst] = buf[src];
                }
            }
        }

        // Final Linear: per-step [C * mel_out] → [d_model].
        let mut out = vec![0.0_f32; t3 * self.d_model];
        for t in 0..t3 {
            let src = &flat[t * feat_per_step..(t + 1) * feat_per_step];
            let dst = &mut out[t * self.d_model..(t + 1) * self.d_model];
            // dst = bias + W @ src
            dst.copy_from_slice(&self.out_bias);
            for r in 0..self.d_model {
                let row = &self.out_weight[r * feat_per_step..(r + 1) * feat_per_step];
                let mut s = 0.0_f32;
                for c in 0..feat_per_step {
                    s += row[c] * src[c];
                }
                dst[r] += s;
            }
        }

        (out, t3)
    }
}

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Conv2d with kernel=3, stride=2, pad=1.
///
/// Input layout : `[in_c, H, W]` row-major.
/// Weight layout: `[out_c, in_c, 3, 3]` row-major.
/// Output layout: `[out_c, H_out, W_out]` row-major.
///
/// Returns (output_buffer, H_out, W_out).
fn conv2d_stride2(
    input: &[f32],
    in_c: usize,
    h_in: usize,
    w_in: usize,
    out_c: usize,
    weight: &[f32],
    bias: &[f32],
) -> (Vec<f32>, usize, usize) {
    let kh = 3;
    let kw = 3;
    let stride = 2;
    let pad = 1;
    let h_out = (h_in + 2 * pad - kh) / stride + 1;
    let w_out = (w_in + 2 * pad - kw) / stride + 1;
    debug_assert_eq!(input.len(), in_c * h_in * w_in);
    debug_assert_eq!(weight.len(), out_c * in_c * kh * kw);
    debug_assert_eq!(bias.len(), out_c);

    let mut out = vec![0.0_f32; out_c * h_out * w_out];

    for oc in 0..out_c {
        for h_o in 0..h_out {
            for w_o in 0..w_out {
                let mut s = bias[oc];
                let h_origin = h_o as isize * stride as isize - pad as isize;
                let w_origin = w_o as isize * stride as isize - pad as isize;
                for ic in 0..in_c {
                    let w_base = ((oc * in_c + ic) * kh) * kw;
                    for ki in 0..kh {
                        let h_in_idx = h_origin + ki as isize;
                        if h_in_idx < 0 || h_in_idx >= h_in as isize {
                            continue;
                        }
                        for kj in 0..kw {
                            let w_in_idx = w_origin + kj as isize;
                            if w_in_idx < 0 || w_in_idx >= w_in as isize {
                                continue;
                            }
                            let in_off = (ic * h_in + h_in_idx as usize) * w_in
                                + w_in_idx as usize;
                            let w_off = w_base + ki * kw + kj;
                            s += input[in_off] * weight[w_off];
                        }
                    }
                }
                out[(oc * h_out + h_o) * w_out + w_o] = s;
            }
        }
    }
    (out, h_out, w_out)
}

/// Compute the output mel dimension after 3 stages of stride-2.
pub fn subsampled_mel_dim(n_mels: usize) -> usize {
    let pad = 1;
    let kh = 3;
    let stride = 2;
    let mut h = n_mels;
    for _ in 0..3 {
        h = (h + 2 * pad - kh) / stride + 1;
    }
    h
}

/// Compute the output time dimension after 3 stages of stride-2.
pub fn subsampled_time_dim(time: usize) -> usize {
    subsampled_mel_dim(time) // same arithmetic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampled_dims_match_factor_8_for_typical_inputs() {
        // 80 mel → 40 → 20 → 10.
        assert_eq!(subsampled_mel_dim(80), 10);
        // 1600 frames → 800 → 400 → 200 (close to time/8).
        assert_eq!(subsampled_time_dim(1600), 200);
    }

    #[test]
    fn conv2d_zero_weight_returns_bias_planes() {
        // 1 input channel, 1 output channel, all-zero weight, bias=2.
        // Output = bias plane (size H_out × W_out).
        // (2 + 2 - 3) / 2 + 1 = 1. So h_out=w_out=1.
        let input = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let weight = vec![0.0; 9]; // 1*1*3*3
        let bias = vec![2.0];
        let (out, h_out, w_out) = conv2d_stride2(&input, 1, 2, 2, 1, &weight, &bias);
        assert_eq!(h_out, 1);
        assert_eq!(w_out, 1);
        assert_eq!(out, vec![2.0]);
    }

    #[test]
    fn conv2d_unit_weight_at_center_is_pixel_pick() {
        // 1 in, 1 out, weight = [[0,0,0],[0,1,0],[0,0,0]] picks the
        // center pixel of each receptive field. With stride=2 pad=1 on a
        // 4×4 input the center of (h_o=0, w_o=0) lands at (h=0, w=0),
        // (h_o=1, w_o=0) → (h=2, w=0), etc.
        let input = vec![
            1.0_f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut weight = vec![0.0_f32; 9];
        weight[4] = 1.0; // center of the 3x3 kernel
        let bias = vec![0.0];
        let (out, h_out, w_out) = conv2d_stride2(&input, 1, 4, 4, 1, &weight, &bias);
        assert_eq!(h_out, 2);
        assert_eq!(w_out, 2);
        // Picks (0,0), (0,2), (2,0), (2,2) → 1, 3, 9, 11.
        assert_eq!(out, vec![1.0, 3.0, 9.0, 11.0]);
    }

    #[test]
    fn subsample_zero_weights_returns_bias_only() {
        let n_mels = 8;
        let d_model = 4;
        let n_mels_out = subsampled_mel_dim(n_mels); // (8→4→2→1) = 1
        let channels = 4;
        let feat_per_step = channels * n_mels_out;
        let s = Subsample {
            n_mels,
            d_model,
            n_mels_out,
            channels,
            conv0_weight: vec![0.0; channels * 1 * 3 * 3],
            conv0_bias: vec![0.0; channels],
            conv1_weight: vec![0.0; channels * channels * 3 * 3],
            conv1_bias: vec![0.0; channels],
            conv2_weight: vec![0.0; channels * channels * 3 * 3],
            conv2_bias: vec![0.0; channels],
            out_weight: vec![0.0; d_model * feat_per_step],
            out_bias: vec![1.0; d_model],
        };
        // 16 frames in → 16 → 8 → 4 → 2 (after 3 stride-2). t_out = 2.
        let mel = vec![0.0_f32; 16 * n_mels];
        let (out, t_out) = s.forward(&mel, 16);
        assert_eq!(t_out, 2);
        assert_eq!(out.len(), 2 * d_model);
        // All outputs equal out_bias.
        for v in &out {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }
}
