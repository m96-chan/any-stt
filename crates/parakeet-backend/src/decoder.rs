//! TDT (Token-Duration Transducer) decoder.
//!
//! TDT generalizes RNN-T: instead of stepping one encoder frame at a time,
//! the joint network outputs both a token AND a duration token. The
//! duration token says how many encoder frames to skip before the next
//! decision. This roughly halves the number of frames the decoder visits
//! at the same accuracy.
//!
//! Joint output split:
//!   logits[0..vocab_size+1]              — token logits (incl. blank)
//!   logits[vocab_size+1..vocab_size+1+D] — duration logits (D = #durations)
//!
//! Greedy loop:
//! ```text
//!   t = 0
//!   pred_h = 0
//!   while t < n_frames:
//!       loop:
//!           logits = joint(enc[t], pred_h)
//!           tok = argmax(logits[0..vocab+1])
//!           dur = argmax(logits[vocab+1..])  // index into durations[]
//!           if tok != blank:
//!               emit tok
//!               pred_h = lstm_step(embedding(tok), state)
//!           if dur > 0 or tok == blank:
//!               t += max(durations[dur], 1)
//!               break  // advance frame
//!   // edge case: if dur == 0 and tok != blank, stay on same frame
//!   //            and emit again. Loop guarded by MAX_EMIT_PER_FRAME.
//! ```
//!
//! Reference: arxiv 2304.06795 "Token-and-Duration Transducer for ASR".

use fastconformer_core::Config;
use fastconformer_core::encoder::EncoderOutput;
use gguf_loader::GgufFile;

/// One LSTM layer.
struct LstmLayer {
    input_size: usize,
    hidden_size: usize,
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

pub struct PredictionNetwork {
    embedding: Vec<f32>,
    pred_hidden: usize,
    vocab_size: usize,
    layers: Vec<LstmLayer>,
}

/// TDT joint network: outputs both token and duration logits in one shot.
/// Output layout: `[vocab_size + 1 (incl. blank), n_durations]` flattened
/// as `[token_logits | duration_logits]`.
pub struct JointNetwork {
    w_enc: Vec<f32>,
    b_enc: Vec<f32>,
    w_pred: Vec<f32>,
    b_pred: Vec<f32>,
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    joint_hidden: usize,
    encoder_dim: usize,
    pred_hidden: usize,
    vocab_size_with_blank: usize,
    n_durations: usize,
}

impl JointNetwork {
    fn out_dim(&self) -> usize {
        self.vocab_size_with_blank + self.n_durations
    }

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
        let total = self.out_dim();
        let mut out = vec![0.0_f32; total];
        matvec_add(&self.w_out, &hidden, &mut out);
        for j in 0..total {
            out[j] += self.b_out[j];
        }
        out
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn from_test_weights(
        encoder_dim: usize,
        pred_hidden: usize,
        joint_hidden: usize,
        vocab_size_with_blank: usize,
        n_durations: usize,
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
            n_durations,
        }
    }
}

struct LstmState {
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

/// TDT decoder.
pub struct TdtDecoder {
    cfg: Config,
    pred: PredictionNetwork,
    joint: JointNetwork,
    /// Duration values, e.g. `[0, 1, 2, 3, 4]` for parakeet-tdt-0.6b-v3.
    durations: Vec<u32>,
}

impl TdtDecoder {
    /// Build the decoder from a GGUF model produced by
    /// `scripts/convert-nemo-to-gguf.py`. Tensor naming follows the same
    /// convention as the RNN-T family but the joint output is wider:
    /// `[vocab_with_blank + n_durations]`. The durations array is read
    /// from the `fastconformer.decoder.tdt_durations` metadata key.
    pub fn from_gguf(gguf: &GgufFile, cfg: Config) -> Result<Self, String> {
        let pred_hidden = cfg.pred_hidden as usize;
        let joint_hidden = cfg.joint_hidden as usize;
        let d_model = cfg.d_model as usize;

        // tdt_durations metadata array (uint32 typically; accept ints too).
        let durations: Vec<u32> = match gguf.meta("fastconformer.decoder.tdt_durations") {
            Some(gguf_loader::MetaValue::ArrayUint32(v)) => v.clone(),
            Some(gguf_loader::MetaValue::ArrayInt32(v)) => {
                v.iter().map(|&x| x as u32).collect()
            }
            Some(gguf_loader::MetaValue::ArrayInt64(v)) => {
                v.iter().map(|&x| x as u32).collect()
            }
            Some(other) => {
                return Err(format!(
                    "fastconformer.decoder.tdt_durations: \
                     unexpected metadata variant {other:?}"
                ));
            }
            None => {
                // Fall back to NeMo's default [0, 1, 2, 3, 4].
                vec![0, 1, 2, 3, 4]
            }
        };
        let n_durations = durations.len();

        let embedding = gguf.dequantize_f32("dec.embed.weight")?;
        if embedding.len() % pred_hidden != 0 {
            return Err(format!(
                "dec.embed.weight size {} not a multiple of pred_hidden={}",
                embedding.len(),
                pred_hidden
            ));
        }
        let vocab_with_blank = embedding.len() / pred_hidden;

        let mut layers = Vec::new();
        for l in 0_usize.. {
            let key = format!("dec.rnn.{l}.weight_ih");
            if !gguf.has_tensor(&key) {
                break;
            }
            let w_ih = gguf.dequantize_f32(&format!("dec.rnn.{l}.weight_ih"))?;
            let w_hh = gguf.dequantize_f32(&format!("dec.rnn.{l}.weight_hh"))?;
            let b_ih = gguf.dequantize_f32(&format!("dec.rnn.{l}.bias_ih"))?;
            let b_hh = gguf.dequantize_f32(&format!("dec.rnn.{l}.bias_hh"))?;
            layers.push(LstmLayerWeights {
                input_size: pred_hidden,
                hidden_size: pred_hidden,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
            });
        }
        if layers.is_empty() {
            return Err("no `dec.rnn.0.*` tensors found".into());
        }

        let pred = PredictionNetwork::from_test_weights(
            embedding,
            pred_hidden,
            vocab_with_blank,
            layers,
        );

        let joint = JointNetwork::from_test_weights(
            d_model,
            pred_hidden,
            joint_hidden,
            vocab_with_blank,
            n_durations,
            gguf.dequantize_f32("joint.enc.weight")?,
            gguf.dequantize_f32("joint.enc.bias")?,
            gguf.dequantize_f32("joint.pred.weight")?,
            gguf.dequantize_f32("joint.pred.bias")?,
            gguf.dequantize_f32("joint.fc2.weight")?,
            gguf.dequantize_f32("joint.fc2.bias")?,
        );

        Self::from_weights(cfg, pred, joint, durations)
    }

    /// Build from in-memory weights (used by tests).
    pub fn from_weights(
        cfg: Config,
        pred: PredictionNetwork,
        joint: JointNetwork,
        durations: Vec<u32>,
    ) -> Result<Self, String> {
        if pred.pred_hidden != joint.pred_hidden {
            return Err(format!(
                "pred hidden mismatch: pred={} joint={}",
                pred.pred_hidden, joint.pred_hidden
            ));
        }
        if pred.vocab_size != joint.vocab_size_with_blank {
            return Err(format!(
                "vocab mismatch: pred.vocab_size={} joint.vocab_with_blank={}",
                pred.vocab_size, joint.vocab_size_with_blank
            ));
        }
        if durations.len() != joint.n_durations {
            return Err(format!(
                "duration count mismatch: durations={} joint.n_durations={}",
                durations.len(),
                joint.n_durations
            ));
        }
        Ok(Self {
            cfg,
            pred,
            joint,
            durations,
        })
    }

    /// Greedy TDT decode.
    pub fn greedy_decode(&self, enc: &EncoderOutput) -> Vec<u32> {
        if enc.d_model != self.joint.encoder_dim {
            return Vec::new();
        }
        let blank = self.cfg.blank_id;
        let vsb = self.joint.vocab_size_with_blank;
        let mut emitted = Vec::new();
        let mut state = LstmState::zeros(&self.pred.layers);
        let mut pred_h = vec![0.0_f32; self.pred.pred_hidden];

        const MAX_EMIT_PER_FRAME: usize = 30;

        let mut t = 0;
        while t < enc.n_frames {
            let enc_frame = &enc.data[t * enc.d_model..(t + 1) * enc.d_model];
            let mut emits = 0;
            loop {
                let logits = self.joint.forward(enc_frame, &pred_h);
                let tok = argmax(&logits[..vsb]) as u32;
                let dur_idx = argmax(&logits[vsb..]);
                let dur = self.durations[dur_idx];

                if tok != blank {
                    emitted.push(tok);
                    pred_h = self.pred.forward(tok, &mut state);
                }
                emits += 1;

                // TDT advance rules:
                //   dur > 0  OR  tok == blank → advance by `dur` frames
                //   dur == 0 AND tok != blank → stay on same frame (re-emit)
                if dur > 0 || tok == blank {
                    t += dur.max(1) as usize;
                    break;
                }
                if emits >= MAX_EMIT_PER_FRAME {
                    t += 1; // forced advance to break runaway loops
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

    fn make_decoder(force_token: Option<u32>, force_dur_idx: usize) -> TdtDecoder {
        let enc_dim = 4;
        let pred_h = 3;
        let joint_h = 5;
        let vocab_size = 4;
        let durations: Vec<u32> = vec![0, 1, 2, 3, 4];
        let n_dur = durations.len();
        let blank = vocab_size as u32;
        let vsb = vocab_size + 1;

        // Embedding rows for tokens 0..vsb (NeMo includes blank slot).
        let mut emb = vec![0.0_f32; vsb * pred_h];
        for tok in 0..vsb {
            for k in 0..pred_h {
                emb[tok * pred_h + k] = (tok as f32) * 0.1 + (k as f32) * 0.01;
            }
        }
        let layers = vec![LstmLayerWeights {
            input_size: pred_h,
            hidden_size: pred_h,
            w_ih: vec![0.0; 4 * pred_h * pred_h],
            w_hh: vec![0.0; 4 * pred_h * pred_h],
            b_ih: vec![0.0; 4 * pred_h],
            b_hh: vec![0.0; 4 * pred_h],
        }];
        let pred = PredictionNetwork::from_test_weights(emb, pred_h, vsb, layers);

        let total_out = vsb + n_dur;
        let w_enc = vec![0.0; joint_h * enc_dim];
        let b_enc = vec![0.0; joint_h];
        let w_pred = vec![0.0; joint_h * pred_h];
        let b_pred = vec![0.0; joint_h];
        let w_out = vec![0.0; total_out * joint_h];
        let mut b_out = vec![0.0_f32; total_out];

        // Force the desired token (or blank if None).
        let tok = force_token.unwrap_or(blank);
        b_out[tok as usize] = 100.0;
        // Force the desired duration index.
        b_out[vsb + force_dur_idx] = 100.0;

        let joint = JointNetwork::from_test_weights(
            enc_dim, pred_h, joint_h, vsb, n_dur, w_enc, b_enc, w_pred, b_pred, w_out, b_out,
        );

        let mut cfg = Config::dummy_parakeet_tdt_v3();
        cfg.vocab_size = vocab_size as u32;
        cfg.blank_id = blank;
        cfg.pred_hidden = pred_h as u32;
        cfg.joint_hidden = joint_h as u32;

        TdtDecoder::from_weights(cfg, pred, joint, durations).unwrap()
    }

    fn dummy_enc(n_frames: usize, dim: usize) -> EncoderOutput {
        EncoderOutput {
            data: vec![0.0; n_frames * dim],
            n_frames,
            d_model: dim,
        }
    }

    #[test]
    fn blank_with_zero_dur_advances_one_frame() {
        // tok=blank, dur_idx=0 (=duration 0). Per TDT rule:
        // dur==0 but tok==blank → advance by max(dur, 1) = 1 frame. So
        // 5 frames → 5 iterations, each advancing 1. No emissions.
        let dec = make_decoder(None, 0);
        let enc = dummy_enc(5, 4);
        let out = dec.greedy_decode(&enc);
        assert!(out.is_empty());
    }

    #[test]
    fn token_with_duration_2_emits_then_skips() {
        // tok=1, dur_idx=2 (=duration 2). Per frame: emit tok=1 then
        // advance by 2 frames. n_frames=6 → t goes 0, 2, 4, 6 (stop).
        // So 3 emissions of token 1.
        let dec = make_decoder(Some(1), 2);
        let enc = dummy_enc(6, 4);
        let out = dec.greedy_decode(&enc);
        assert_eq!(out, vec![1, 1, 1]);
    }

    #[test]
    fn token_with_duration_4_skips_aggressively() {
        // tok=2, dur_idx=4 (=duration 4). t goes 0, 4 (stop, 8 > 5).
        // 2 emissions of token 2.
        let dec = make_decoder(Some(2), 4);
        let enc = dummy_enc(5, 4);
        let out = dec.greedy_decode(&enc);
        assert_eq!(out, vec![2, 2]);
    }

    #[test]
    fn token_with_duration_zero_re_emits_until_capped() {
        // tok=3, dur_idx=0 (=duration 0). dur==0 AND tok != blank →
        // re-emit on same frame. Capped at MAX_EMIT_PER_FRAME=30, then
        // forced advance by 1.
        let dec = make_decoder(Some(3), 0);
        let enc = dummy_enc(2, 4);
        let out = dec.greedy_decode(&enc);
        assert_eq!(out.len(), 60); // 2 frames × 30 cap each
        assert!(out.iter().all(|&t| t == 3));
    }

    #[test]
    fn empty_encoder_emits_nothing() {
        let dec = make_decoder(Some(1), 1);
        let enc = dummy_enc(0, 4);
        assert!(dec.greedy_decode(&enc).is_empty());
    }

    #[test]
    fn encoder_dim_mismatch_returns_empty() {
        let dec = make_decoder(Some(1), 1);
        let enc = dummy_enc(3, 99);
        assert!(dec.greedy_decode(&enc).is_empty());
    }

    #[test]
    fn from_weights_rejects_duration_count_mismatch() {
        let cfg = Config::dummy_parakeet_tdt_v3();
        // Embedding rows = vocab_with_blank = 5 to match joint output.
        let pred = PredictionNetwork::from_test_weights(
            vec![0.0; 5 * 3],
            3,
            5, // vocab_with_blank
            vec![LstmLayerWeights {
                input_size: 3,
                hidden_size: 3,
                w_ih: vec![0.0; 36],
                w_hh: vec![0.0; 36],
                b_ih: vec![0.0; 12],
                b_hh: vec![0.0; 12],
            }],
        );
        let joint = JointNetwork::from_test_weights(
            4,
            3,
            5,
            5, // vocab_with_blank
            5, // expects 5 durations
            vec![0.0; 5 * 4],
            vec![0.0; 5],
            vec![0.0; 5 * 3],
            vec![0.0; 5],
            vec![0.0; (5 + 5) * 5],
            vec![0.0; 5 + 5],
        );
        // Pass only 3 durations — should error on duration count.
        assert!(TdtDecoder::from_weights(cfg, pred, joint, vec![0, 1, 2]).is_err());
    }
}
