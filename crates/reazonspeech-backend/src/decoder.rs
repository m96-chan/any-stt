//! RNN-T (Recurrent Neural Network Transducer) decoder.
//!
//! Three components: prediction network (LSTM over emitted tokens),
//! encoder-side projection, and a joint network that combines them to
//! produce token logits. Greedy decoding is the default for RTF-sensitive
//! paths; beam search is a follow-up for accuracy-first use.
//!
//! ## Status
//! Stub. The greedy loop is sketched; the LSTM and joint net forward
//! passes are TODO. Decoder runs on CPU (small ops, loop-heavy — NPU
//! offload does not pay off).

use crate::config::ReazonSpeechConfig;
use crate::encoder::EncoderOutput;

/// RNN-T decoder handle.
pub struct RnntDecoder {
    cfg: ReazonSpeechConfig,
    // TODO(#N4): prediction net (embedding + LSTM) weights
    //            joint net (enc_proj, pred_proj, fc1, fc2) weights
}

impl RnntDecoder {
    pub fn from_gguf(
        _gguf: &gguf_loader::GgufFile,
        cfg: ReazonSpeechConfig,
    ) -> Result<Self, String> {
        // TODO(#N4): load
        //   - dec.embed.weight [vocab_size, pred_hidden]
        //   - dec.rnn.{L}.{weight_ih,weight_hh,bias_ih,bias_hh} for L in 0..n_rnn_layers
        //   - joint.enc.{weight,bias}
        //   - joint.pred.{weight,bias}
        //   - joint.fc1.{weight,bias}
        //   - joint.fc2.{weight,bias}
        Ok(Self { cfg })
    }

    /// Greedy RNN-T decode → sequence of token IDs (excluding blanks).
    pub fn greedy_decode(&self, _enc: &EncoderOutput) -> Result<Vec<u32>, String> {
        // TODO(#N4): implement greedy loop.
        //   Standard RNN-T greedy: for each encoder frame t,
        //     while best_token != blank:
        //       joint_logits = joint(enc[t], pred_state)
        //       token = argmax(joint_logits)
        //       if token != blank:
        //         emit token
        //         pred_state = lstm_step(embedding(token), pred_state)
        //     advance t
        Err("RnntDecoder::greedy_decode not yet implemented".into())
    }

    pub fn config(&self) -> &ReazonSpeechConfig {
        &self.cfg
    }
}
