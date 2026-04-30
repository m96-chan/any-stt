//! Subsampling stack: mel spectrogram → encoder feature stream.
//!
//! NeMo's `ConvSubsampling` module supports several variants. This file
//! implements the two we have ASR models for:
//!
//! - **`Striding`** — three plain Conv2d stages (kernel=3, stride=2,
//!   pad=1) interleaved with ReLU, then a Linear projection. Used by
//!   parakeet-tdt's reference build.
//!
//! - **`DwStriding`** — one regular Conv2d followed by two
//!   `(depthwise + pointwise)` pairs, each pair acting as one stride-2
//!   stage. Used by reazonspeech-nemo-v2 and other modern NeMo
//!   FastConformers. Same overall ×8 factor as `Striding` but uses
//!   ~⅓ the parameters in the conv stack.
//!
//! Input shape  : `[time, n_mels]` row-major.
//! Output shape : `[time/8, d_model]` row-major.
//!
//! For typical 16 kHz / 80-mel input, the mel axis goes 80 → 40 → 20 → 10.

/// Plain striding (3 × Conv2d) subsample.
pub struct SubsampleStriding {
    pub n_mels: usize,
    pub d_model: usize,
    /// `n_mels / 8` (ceil semantics).
    pub n_mels_out: usize,
    /// Intermediate channel count.
    pub channels: usize,

    /// `[C, 1, 3, 3]` flattened.
    pub conv0_weight: Vec<f32>,
    pub conv0_bias: Vec<f32>,
    /// `[C, C, 3, 3]`
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    /// `[C, C, 3, 3]`
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,

    /// Final linear: `[d_model, C * n_mels_out]`
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
}

/// Depthwise-striding (regular + 2 × DW+PW) subsample.
pub struct SubsampleDwStriding {
    pub n_mels: usize,
    pub d_model: usize,
    pub n_mels_out: usize,
    pub channels: usize,

    /// Regular Conv2d 1→C, kernel=3, stride=2: `[C, 1, 3, 3]`.
    pub conv0_weight: Vec<f32>,
    pub conv0_bias: Vec<f32>,

    /// Depthwise Conv2d C→C (groups=C), kernel=3, stride=2: `[C, 1, 3, 3]`.
    pub dw1_weight: Vec<f32>,
    pub dw1_bias: Vec<f32>,
    /// Pointwise Conv2d C→C, kernel=1: `[C, C, 1, 1]`.
    pub pw1_weight: Vec<f32>,
    pub pw1_bias: Vec<f32>,

    /// Depthwise Conv2d C→C (groups=C), kernel=3, stride=2: `[C, 1, 3, 3]`.
    pub dw2_weight: Vec<f32>,
    pub dw2_bias: Vec<f32>,
    /// Pointwise Conv2d C→C, kernel=1: `[C, C, 1, 1]`.
    pub pw2_weight: Vec<f32>,
    pub pw2_bias: Vec<f32>,

    /// Final linear: `[d_model, C * n_mels_out]`.
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
}

/// Subsampling variant.
pub enum Subsample {
    Striding(SubsampleStriding),
    DwStriding(SubsampleDwStriding),
}

impl Subsample {
    /// Apply the configured subsampler.
    /// Returns `(features [t_out * d_model], t_out)`.
    pub fn forward(&self, mel: &[f32], time: usize) -> (Vec<f32>, usize) {
        match self {
            Self::Striding(s) => s.forward(mel, time),
            Self::DwStriding(s) => s.forward(mel, time),
        }
    }

    pub fn d_model(&self) -> usize {
        match self {
            Self::Striding(s) => s.d_model,
            Self::DwStriding(s) => s.d_model,
        }
    }

    pub fn n_mels(&self) -> usize {
        match self {
            Self::Striding(s) => s.n_mels,
            Self::DwStriding(s) => s.n_mels,
        }
    }
}

// ---------------------------------------------------------------------------
// SubsampleStriding (3 plain Conv2d).
// ---------------------------------------------------------------------------

impl SubsampleStriding {
    pub fn forward(&self, mel: &[f32], time: usize) -> (Vec<f32>, usize) {
        debug_assert_eq!(mel.len(), time * self.n_mels);

        let (mut buf, t1, h1) = conv2d_stride2(
            mel, 1, time, self.n_mels, self.channels,
            &self.conv0_weight, &self.conv0_bias,
        );
        relu(&mut buf);

        let (mut buf, t2, h2) = conv2d_stride2(
            &buf, self.channels, t1, h1, self.channels,
            &self.conv1_weight, &self.conv1_bias,
        );
        relu(&mut buf);

        let (buf, t3, h3) = conv2d_stride2(
            &buf, self.channels, t2, h2, self.channels,
            &self.conv2_weight, &self.conv2_bias,
        );

        debug_assert_eq!(h3, self.n_mels_out);
        flatten_and_project(
            &buf, self.channels, t3, h3,
            self.d_model, &self.out_weight, &self.out_bias,
        )
    }
}

// ---------------------------------------------------------------------------
// SubsampleDwStriding (regular + 2 × (depthwise + pointwise)).
// ---------------------------------------------------------------------------

impl SubsampleDwStriding {
    pub fn forward(&self, mel: &[f32], time: usize) -> (Vec<f32>, usize) {
        debug_assert_eq!(mel.len(), time * self.n_mels);

        // Stage 1: regular Conv2d 1 → C, stride 2 (mel & time both halved).
        let (mut buf, t1, h1) = conv2d_stride2(
            mel, 1, time, self.n_mels, self.channels,
            &self.conv0_weight, &self.conv0_bias,
        );
        relu(&mut buf);

        // Stage 2: depthwise Conv2d C → C (groups=C), stride 2.
        let (buf, t2, h2) = depthwise_conv2d_stride2(
            &buf, self.channels, t1, h1,
            &self.dw1_weight, &self.dw1_bias,
        );
        // Stage 3: pointwise Conv2d C → C, kernel 1 (no spatial change).
        let mut buf = pointwise_conv2d(
            &buf, self.channels, t2, h2, self.channels,
            &self.pw1_weight, &self.pw1_bias,
        );
        relu(&mut buf);

        // Stage 4: depthwise Conv2d C → C (groups=C), stride 2.
        let (buf, t3, h3) = depthwise_conv2d_stride2(
            &buf, self.channels, t2, h2,
            &self.dw2_weight, &self.dw2_bias,
        );
        // Stage 5: pointwise Conv2d C → C, kernel 1.
        let mut buf = pointwise_conv2d(
            &buf, self.channels, t3, h3, self.channels,
            &self.pw2_weight, &self.pw2_bias,
        );
        relu(&mut buf);

        debug_assert_eq!(h3, self.n_mels_out);
        flatten_and_project(
            &buf, self.channels, t3, h3,
            self.d_model, &self.out_weight, &self.out_bias,
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Flatten `[C, T, H]` to `[T, C * H]` and project with a Linear.
fn flatten_and_project(
    buf: &[f32],
    channels: usize,
    t: usize,
    h: usize,
    d_model: usize,
    out_weight: &[f32],
    out_bias: &[f32],
) -> (Vec<f32>, usize) {
    let feat_per_step = channels * h;
    let mut flat = vec![0.0_f32; t * feat_per_step];
    for c in 0..channels {
        for tt in 0..t {
            for hh in 0..h {
                let src = c * t * h + tt * h + hh;
                let dst = tt * feat_per_step + c * h + hh;
                flat[dst] = buf[src];
            }
        }
    }
    let mut out = vec![0.0_f32; t * d_model];
    for tt in 0..t {
        let src = &flat[tt * feat_per_step..(tt + 1) * feat_per_step];
        let dst = &mut out[tt * d_model..(tt + 1) * d_model];
        dst.copy_from_slice(out_bias);
        for r in 0..d_model {
            let row = &out_weight[r * feat_per_step..(r + 1) * feat_per_step];
            let mut s = 0.0_f32;
            for c in 0..feat_per_step {
                s += row[c] * src[c];
            }
            dst[r] += s;
        }
    }
    (out, t)
}

/// Plain 3×3 Conv2d, stride=2, pad=1.
///
/// Input layout : `[in_c, H, W]` row-major.
/// Weight layout: `[out_c, in_c, 3, 3]` row-major.
/// Output layout: `[out_c, H_out, W_out]` row-major.
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

/// Depthwise 3×3 Conv2d, stride=2, pad=1, `groups = C`.
///
/// Each output channel uses ONLY its own input channel. Weight layout:
/// `[C, 1, 3, 3]` (PyTorch convention for depthwise: out=C, in/groups=1,
/// H=3, W=3). bias `[C]`.
fn depthwise_conv2d_stride2(
    input: &[f32],
    channels: usize,
    h_in: usize,
    w_in: usize,
    weight: &[f32],
    bias: &[f32],
) -> (Vec<f32>, usize, usize) {
    let kh = 3;
    let kw = 3;
    let stride = 2;
    let pad = 1;
    let h_out = (h_in + 2 * pad - kh) / stride + 1;
    let w_out = (w_in + 2 * pad - kw) / stride + 1;
    debug_assert_eq!(input.len(), channels * h_in * w_in);
    debug_assert_eq!(weight.len(), channels * kh * kw);
    debug_assert_eq!(bias.len(), channels);

    let mut out = vec![0.0_f32; channels * h_out * w_out];
    for c in 0..channels {
        let kbase = c * kh * kw;
        for h_o in 0..h_out {
            for w_o in 0..w_out {
                let mut s = bias[c];
                let h_origin = h_o as isize * stride as isize - pad as isize;
                let w_origin = w_o as isize * stride as isize - pad as isize;
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
                        let in_off = (c * h_in + h_in_idx as usize) * w_in
                            + w_in_idx as usize;
                        s += input[in_off] * weight[kbase + ki * kw + kj];
                    }
                }
                out[(c * h_out + h_o) * w_out + w_o] = s;
            }
        }
    }
    (out, h_out, w_out)
}

/// Pointwise 1×1 Conv2d with no spatial reduction.
///
/// Weight layout: `[out_c, in_c, 1, 1]` flattened to `[out_c, in_c]`.
fn pointwise_conv2d(
    input: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    out_c: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(input.len(), in_c * h * w);
    debug_assert_eq!(weight.len(), out_c * in_c);
    debug_assert_eq!(bias.len(), out_c);

    let mut out = vec![0.0_f32; out_c * h * w];
    for oc in 0..out_c {
        for hh in 0..h {
            for ww in 0..w {
                let mut s = bias[oc];
                for ic in 0..in_c {
                    s += input[(ic * h + hh) * w + ww] * weight[oc * in_c + ic];
                }
                out[(oc * h + hh) * w + ww] = s;
            }
        }
    }
    out
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
    subsampled_mel_dim(time)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampled_dims_match_factor_8_for_typical_inputs() {
        assert_eq!(subsampled_mel_dim(80), 10);
        assert_eq!(subsampled_time_dim(1600), 200);
    }

    #[test]
    fn conv2d_zero_weight_returns_bias_planes() {
        // (2 + 2 - 3) / 2 + 1 = 1.
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![0.0; 9];
        let bias = vec![2.0];
        let (out, h_out, w_out) = conv2d_stride2(&input, 1, 2, 2, 1, &weight, &bias);
        assert_eq!(h_out, 1);
        assert_eq!(w_out, 1);
        assert_eq!(out, vec![2.0]);
    }

    #[test]
    fn conv2d_unit_weight_at_center_is_pixel_pick() {
        let input = vec![
            1.0_f32, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut weight = vec![0.0_f32; 9];
        weight[4] = 1.0;
        let bias = vec![0.0];
        let (out, h_out, w_out) = conv2d_stride2(&input, 1, 4, 4, 1, &weight, &bias);
        assert_eq!(h_out, 2);
        assert_eq!(w_out, 2);
        assert_eq!(out, vec![1.0, 3.0, 9.0, 11.0]);
    }

    #[test]
    fn striding_zero_weights_returns_bias_only() {
        let n_mels = 8;
        let d_model = 4;
        let n_mels_out = subsampled_mel_dim(n_mels);
        let channels = 4;
        let feat_per_step = channels * n_mels_out;
        let s = SubsampleStriding {
            n_mels,
            d_model,
            n_mels_out,
            channels,
            conv0_weight: vec![0.0; channels * 1 * 9],
            conv0_bias: vec![0.0; channels],
            conv1_weight: vec![0.0; channels * channels * 9],
            conv1_bias: vec![0.0; channels],
            conv2_weight: vec![0.0; channels * channels * 9],
            conv2_bias: vec![0.0; channels],
            out_weight: vec![0.0; d_model * feat_per_step],
            out_bias: vec![1.0; d_model],
        };
        let mel = vec![0.0_f32; 16 * n_mels];
        let (out, t_out) = s.forward(&mel, 16);
        assert_eq!(t_out, 2);
        assert_eq!(out.len(), 2 * d_model);
        for v in &out {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dw_striding_zero_weights_returns_bias_only() {
        let n_mels = 8;
        let d_model = 4;
        let n_mels_out = subsampled_mel_dim(n_mels);
        let channels = 4;
        let feat_per_step = channels * n_mels_out;
        let s = SubsampleDwStriding {
            n_mels,
            d_model,
            n_mels_out,
            channels,
            conv0_weight: vec![0.0; channels * 1 * 9],
            conv0_bias: vec![0.0; channels],
            dw1_weight: vec![0.0; channels * 9],
            dw1_bias: vec![0.0; channels],
            pw1_weight: vec![0.0; channels * channels],
            pw1_bias: vec![0.0; channels],
            dw2_weight: vec![0.0; channels * 9],
            dw2_bias: vec![0.0; channels],
            pw2_weight: vec![0.0; channels * channels],
            pw2_bias: vec![0.0; channels],
            out_weight: vec![0.0; d_model * feat_per_step],
            out_bias: vec![1.0; d_model],
        };
        let mel = vec![0.0_f32; 16 * n_mels];
        let (out, t_out) = s.forward(&mel, 16);
        assert_eq!(t_out, 2);
        assert_eq!(out.len(), 2 * d_model);
        for v in &out {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn depthwise_conv2d_unit_kernel_passes_each_channel_through() {
        // 2-channel input, identity-center kernels. Expect: each channel
        // independent, stride-2 picks the (0,0) and (0,2) and (2,0) and
        // (2,2) pixels.
        let input = vec![
            // channel 0
            1.0_f32, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            3.0, 0.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            // channel 1
            5.0, 0.0, 6.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            7.0, 0.0, 8.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let mut weight = vec![0.0_f32; 2 * 9];
        weight[4] = 1.0;        // center of channel 0's kernel
        weight[9 + 4] = 1.0;    // center of channel 1's kernel
        let bias = vec![0.0; 2];
        let (out, h_out, w_out) = depthwise_conv2d_stride2(&input, 2, 4, 4, &weight, &bias);
        assert_eq!(h_out, 2);
        assert_eq!(w_out, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn pointwise_conv2d_unit_weight_is_identity() {
        // 2-in, 2-out, identity weight: out_c == in_c → output = input.
        let input = vec![1.0_f32, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0]; // [2, 2, 2]
        let mut weight = vec![0.0_f32; 4];
        weight[0] = 1.0; // out=0 takes in=0
        weight[3] = 1.0; // out=1 takes in=1
        let bias = vec![0.0; 2];
        let out = pointwise_conv2d(&input, 2, 2, 2, 2, &weight, &bias);
        assert_eq!(out, input);
    }
}
