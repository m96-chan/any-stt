//! RNN-T (Recurrent Neural Network Transducer) decoder.
//!
//! Architecture:
//!   - Prediction network: token embedding → LSTM stack → output [pred_hidden]
//!   - Joint network:      tanh(enc_proj + pred_proj) → linear → vocab_size + 1 logits
//!     (the +1 is the blank token; in metadata it's `cfg.blank_id`)
//!
//! Greedy decode loop:
//! ```text
//!   t = 0
//!   pred_state = lstm_init
//!   pred_h = embedding(blank)
//!   while t < n_frames:
//!       loop:
//!           logits = joint(enc[t], pred_h)
//!           tok = argmax(logits)
//!           if tok == blank: break
//!           emit tok
//!           pred_h = lstm_step(embedding(tok), pred_state)
//!       t += 1
//! ```
//!
//! All ops are pure CPU f32. The decoder is small (~5 ms / step on a fast
//! core for d=640) and loop-heavy — NPU offload does not pay off.

use fastconformer_core::Config;
use fastconformer_core::encoder::EncoderOutput;
use gguf_loader::GgufFile;

/// One LSTM layer.
struct LstmLayer {
    input_size: usize,
    hidden_size: usize,
    /// `[4 * hidden_size, input_size]` (gate order: i, f, g, o per torch).
    w_ih: Vec<f32>,
    /// `[4 * hidden_size, hidden_size]`.
    w_hh: Vec<f32>,
    /// `[4 * hidden_size]`.
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

/// Prediction network: embedding + multi-layer LSTM.
pub struct PredictionNetwork {
    /// `[vocab_size, pred_hidden]`. NeMo uses pred_hidden as both
    /// embedding dim and hidden state size.
    embedding: Vec<f32>,
    pred_hidden: usize,
    vocab_size: usize,
    layers: Vec<LstmLayer>,
}

/// Joint network: tanh(W_enc·enc + W_pred·pred) → vocab+blank logits.
pub struct JointNetwork {
    /// `[joint_hidden, encoder_dim]`
    w_enc: Vec<f32>,
    /// `[joint_hidden]`
    b_enc: Vec<f32>,
    /// `[joint_hidden, pred_hidden]`
    w_pred: Vec<f32>,
    /// `[joint_hidden]`
    b_pred: Vec<f32>,
    /// `[vocab_size + 1, joint_hidden]`
    w_out: Vec<f32>,
    /// `[vocab_size + 1]`
    b_out: Vec<f32>,
    joint_hidden: usize,
    encoder_dim: usize,
    pred_hidden: usize,
    vocab_size_with_blank: usize,
}

/// Hidden state across an LSTM stack.
struct LstmState {
    /// One (h, c) pair per layer.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl LstmState {
    fn zeros(layers: &[LstmLayer]) -> Self {
        Self {
            layers: layers
                .iter()
                .map(|l| (vec![0.0; l.hidden_size], vec![0.0; l.hidden_size]))
                .collect(),
        }
    }
}

/// RNN-T decoder.
pub struct RnntDecoder {
    cfg: Config,
    pred: PredictionNetwork,
    joint: JointNetwork,
}

impl RnntDecoder {
    /// Build the decoder from a GGUF model. Not yet wired — needs the
    /// real `dec.*` and `joint.*` tensor names which only solidify once
    /// we run the converter on a real `.nemo` file.
    pub fn from_gguf(_gguf: &GgufFile, _cfg: Config) -> Result<Self, String> {
        Err("RnntDecoder::from_gguf not yet implemented — \
             needs `dec.rnn.*` and `joint.*` tensor wire-up"
            .into())
    }

    /// Build a decoder directly from in-memory weights (used by unit tests
    /// and for offline weight conversion).
    pub fn from_weights(
        cfg: Config,
        pred: PredictionNetwork,
        joint: JointNetwork,
    ) -> Result<Self, String> {
        if pred.pred_hidden != joint.pred_hidden {
            return Err(format!(
                "pred hidden mismatch: pred={} joint={}",
                pred.pred_hidden, joint.pred_hidden
            ));
        }
        if pred.vocab_size + 1 != joint.vocab_size_with_blank {
            return Err(format!(
                "vocab mismatch: pred={} joint(includes blank)={}",
                pred.vocab_size, joint.vocab_size_with_blank
            ));
        }
        Ok(Self { cfg, pred, joint })
    }

    /// Greedy RNN-T decode. Returns emitted token IDs (excluding blanks).
    pub fn greedy_decode(&self, enc: &EncoderOutput) -> Vec<u32> {
        if enc.d_model != self.joint.encoder_dim {
            return Vec::new();
        }
        let blank = self.cfg.blank_id;
        let mut emitted = Vec::new();
        let mut state = LstmState::zeros(&self.pred.layers);
        // Initial pred_h: zero vector (no token has been emitted yet).
        // NeMo's RNN-T does the same — the LSTM is conditioned only on
        // the sequence of *non-blank* emissions, and the joint network
        // starts from zero state.
        let mut pred_h = vec![0.0_f32; self.pred.pred_hidden];

        // Cap on emissions per frame to guard against degenerate models
        // that would otherwise loop forever picking non-blank.
        const MAX_EMIT_PER_FRAME: usize = 30;

        for t in 0..enc.n_frames {
            let enc_frame = &enc.data[t * enc.d_model..(t + 1) * enc.d_model];
            let mut emits = 0;
            loop {
                let logits = self.joint.forward(enc_frame, &pred_h);
                let tok = argmax(&logits) as u32;
                if tok == blank {
                    break;
                }
                emitted.push(tok);
                pred_h = self.pred.forward(tok, &mut state);
                emits += 1;
                if emits >= MAX_EMIT_PER_FRAME {
                    break;
                }
            }
        }

        emitted
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
}

impl PredictionNetwork {
    /// Constructor for hand-built test weights.
    #[doc(hidden)]
    pub fn from_test_weights(
        embedding: Vec<f32>,
        pred_hidden: usize,
        vocab_size: usize,
        layers: Vec<LstmLayerWeights>,
    ) -> Self {
        let layers = layers
            .into_iter()
            .map(|w| LstmLayer {
                input_size: w.input_size,
                hidden_size: w.hidden_size,
                w_ih: w.w_ih,
                w_hh: w.w_hh,
                b_ih: w.b_ih,
                b_hh: w.b_hh,
            })
            .collect();
        Self {
            embedding,
            pred_hidden,
            vocab_size,
            layers,
        }
    }

    /// Run one prediction step (embed + LSTM stack). Returns the top
    /// layer's hidden state.
    fn forward(&self, token: u32, state: &mut LstmState) -> Vec<f32> {
        debug_assert!((token as usize) < self.vocab_size);
        let s = (token as usize) * self.pred_hidden;
        let mut x = self.embedding[s..s + self.pred_hidden].to_vec();
        for (layer, (h, c)) in self.layers.iter().zip(state.layers.iter_mut()) {
            x = layer.step(&x, h, c);
        }
        x
    }
}

#[doc(hidden)]
pub struct LstmLayerWeights {
    pub input_size: usize,
    pub hidden_size: usize,
    pub w_ih: Vec<f32>,
    pub w_hh: Vec<f32>,
    pub b_ih: Vec<f32>,
    pub b_hh: Vec<f32>,
}

impl LstmLayer {
    fn step(&self, input: &[f32], h: &mut [f32], c: &mut [f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input_size);
        debug_assert_eq!(h.len(), self.hidden_size);
        let h4 = 4 * self.hidden_size;
        let mut gates = vec![0.0_f32; h4];
        matvec_add(&self.w_ih, input, &mut gates);
        for i in 0..h4 {
            gates[i] += self.b_ih[i];
        }
        matvec_add(&self.w_hh, h, &mut gates);
        for i in 0..h4 {
            gates[i] += self.b_hh[i];
        }
        // torch order: i, f, g, o
        let hs = self.hidden_size;
        let (gi, gf, gg, go) = (
            &gates[0..hs],
            &gates[hs..2 * hs],
            &gates[2 * hs..3 * hs],
            &gates[3 * hs..4 * hs],
        );
        for j in 0..hs {
            let i_t = sigmoid(gi[j]);
            let f_t = sigmoid(gf[j]);
            let g_t = gg[j].tanh();
            let o_t = sigmoid(go[j]);
            c[j] = f_t * c[j] + i_t * g_t;
            h[j] = o_t * c[j].tanh();
        }
        h.to_vec()
    }
}

impl JointNetwork {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_test_weights(
        encoder_dim: usize,
        pred_hidden: usize,
        joint_hidden: usize,
        vocab_size_with_blank: usize,
        w_enc: Vec<f32>,
        b_enc: Vec<f32>,
        w_pred: Vec<f32>,
        b_pred: Vec<f32>,
        w_out: Vec<f32>,
        b_out: Vec<f32>,
    ) -> Self {
        Self {
            w_enc,
            b_enc,
            w_pred,
            b_pred,
            w_out,
            b_out,
            joint_hidden,
            encoder_dim,
            pred_hidden,
            vocab_size_with_blank,
        }
    }

    /// `joint(enc, pred) = w_out · tanh(W_enc·enc + b_enc + W_pred·pred + b_pred) + b_out`
    fn forward(&self, enc: &[f32], pred: &[f32]) -> Vec<f32> {
        debug_assert_eq!(enc.len(), self.encoder_dim);
        debug_assert_eq!(pred.len(), self.pred_hidden);
        let mut hidden = vec![0.0_f32; self.joint_hidden];
        matvec_add(&self.w_enc, enc, &mut hidden);
        for j in 0..self.joint_hidden {
            hidden[j] += self.b_enc[j];
        }
        let mut pp = vec![0.0_f32; self.joint_hidden];
        matvec_add(&self.w_pred, pred, &mut pp);
        for j in 0..self.joint_hidden {
            hidden[j] = (hidden[j] + pp[j] + self.b_pred[j]).tanh();
        }
        let mut out = vec![0.0_f32; self.vocab_size_with_blank];
        matvec_add(&self.w_out, &hidden, &mut out);
        for j in 0..self.vocab_size_with_blank {
            out[j] += self.b_out[j];
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Math helpers (scalar f32; replaceable with ggml/SIMD later).
// ---------------------------------------------------------------------------

/// `out += W · x` where W is `[rows, cols]` row-major.
fn matvec_add(w: &[f32], x: &[f32], out: &mut [f32]) {
    let rows = out.len();
    let cols = x.len();
    debug_assert_eq!(w.len(), rows * cols);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut s = 0.0_f32;
        for c in 0..cols {
            s += row[c] * x[c];
        }
        out[r] += s;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decoder(force_blank: bool, vocab_size: usize) -> RnntDecoder {
        let enc_dim = 4;
        let pred_h = 3;
        let joint_h = 5;

        // Embedding: token i → vector of i*0.1 + k*0.01 (deterministic).
        let mut emb = vec![0.0_f32; vocab_size * pred_h];
        for tok in 0..vocab_size {
            for k in 0..pred_h {
                emb[tok * pred_h + k] = (tok as f32) * 0.1 + (k as f32) * 0.01;
            }
        }
        // Single LSTM layer with zero weights — pred state stays at 0.
        let layers = vec![LstmLayerWeights {
            input_size: pred_h,
            hidden_size: pred_h,
            w_ih: vec![0.0; 4 * pred_h * pred_h],
            w_hh: vec![0.0; 4 * pred_h * pred_h],
            b_ih: vec![0.0; 4 * pred_h],
            b_hh: vec![0.0; 4 * pred_h],
        }];
        let pred = PredictionNetwork::from_test_weights(emb, pred_h, vocab_size, layers);

        let blank = vocab_size; // last index is blank in joint output
        let vsb = vocab_size + 1;
        let w_enc = vec![0.0; joint_h * enc_dim];
        let b_enc = vec![0.0; joint_h];
        let w_pred = vec![0.0; joint_h * pred_h];
        let b_pred = vec![0.0; joint_h];
        let w_out = vec![0.0_f32; vsb * joint_h];
        let mut b_out = vec![0.0_f32; vsb];
        if force_blank {
            b_out[blank] = 100.0;
        } else {
            b_out[1] = 100.0;
        }
        let joint = JointNetwork::from_test_weights(
            enc_dim, pred_h, joint_h, vsb, w_enc, b_enc, w_pred, b_pred, w_out, b_out,
        );

        let mut cfg = Config::dummy_reazonspeech_nemo_v2();
        cfg.vocab_size = vocab_size as u32;
        cfg.blank_id = blank as u32;
        cfg.pred_hidden = pred_h as u32;
        cfg.joint_hidden = joint_h as u32;

        RnntDecoder::from_weights(cfg, pred, joint).unwrap()
    }

    fn dummy_enc(n_frames: usize, dim: usize) -> EncoderOutput {
        EncoderOutput {
            data: vec![0.0; n_frames * dim],
            n_frames,
            d_model: dim,
        }
    }

    #[test]
    fn forced_blank_emits_nothing() {
        let dec = make_decoder(true, 4);
        let enc = dummy_enc(5, 4);
        let out = dec.greedy_decode(&enc);
        assert!(out.is_empty());
    }

    #[test]
    fn forced_token_emits_per_frame_capped() {
        let dec = make_decoder(false, 4);
        let enc = dummy_enc(2, 4);
        let out = dec.greedy_decode(&enc);
        // Each frame triggers MAX_EMIT_PER_FRAME (=30) emits because the
        // joint always picks token 1 — never blank. 2 frames × 30 = 60.
        assert_eq!(out.len(), 60);
        assert!(out.iter().all(|&t| t == 1));
    }

    #[test]
    fn empty_encoder_emits_nothing() {
        let dec = make_decoder(false, 4);
        let enc = dummy_enc(0, 4);
        let out = dec.greedy_decode(&enc);
        assert!(out.is_empty());
    }

    #[test]
    fn encoder_dim_mismatch_returns_empty() {
        let dec = make_decoder(false, 4);
        let enc = dummy_enc(3, 99);
        let out = dec.greedy_decode(&enc);
        assert!(out.is_empty());
    }

    #[test]
    fn from_weights_rejects_pred_dim_mismatch() {
        let cfg = Config::dummy_reazonspeech_nemo_v2();
        let pred = PredictionNetwork::from_test_weights(
            vec![0.0; 4 * 3],
            3,
            4,
            vec![LstmLayerWeights {
                input_size: 3,
                hidden_size: 3,
                w_ih: vec![0.0; 4 * 3 * 3],
                w_hh: vec![0.0; 4 * 3 * 3],
                b_ih: vec![0.0; 12],
                b_hh: vec![0.0; 12],
            }],
        );
        let joint = JointNetwork::from_test_weights(
            4,
            7, // wrong: should be 3
            5,
            5,
            vec![0.0; 5 * 4],
            vec![0.0; 5],
            vec![0.0; 5 * 7],
            vec![0.0; 5],
            vec![0.0; 5 * 5],
            vec![0.0; 5],
        );
        assert!(RnntDecoder::from_weights(cfg, pred, joint).is_err());
    }

    #[test]
    fn lstm_step_zero_weights_preserves_zero_state() {
        let layer = LstmLayer {
            input_size: 3,
            hidden_size: 3,
            w_ih: vec![0.0; 36],
            w_hh: vec![0.0; 36],
            b_ih: vec![0.0; 12],
            b_hh: vec![0.0; 12],
        };
        let mut h = vec![0.0; 3];
        let mut c = vec![0.0; 3];
        let new_h = layer.step(&[0.1, 0.2, 0.3], &mut h, &mut c);
        // gates all zero → i=σ(0)=0.5, f=0.5, g=tanh(0)=0, o=0.5
        // c = 0.5*0 + 0.5*0 = 0
        // h = 0.5 * tanh(0) = 0
        assert!(new_h.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn matvec_add_basic() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0_f32; 2];
        matvec_add(&w, &x, &mut out);
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn argmax_returns_index_of_max() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0, 5.0, 2.0]), 1);
        assert_eq!(argmax(&[7.0]), 0);
    }
}
