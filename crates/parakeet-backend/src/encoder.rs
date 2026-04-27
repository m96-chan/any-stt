//! Parakeet encoder wrapper.
//!
//! Same loader logic as reazonspeech-backend — pulls
//! `enc.{subsample,block.L.*,ln_post}.*` tensors and constructs the
//! shared `fastconformer_core::FastConformerEncoder`. Difference vs
//! reazonspeech is the encoder config: vanilla rel-pos attention, no
//! Longformer, vocab=8192.

use fastconformer_core::encoder::{
    ConformerBlock, ConvModule, FastConformerEncoder, FeedForward, MultiHeadAttention,
    Subsample, subsampled_mel_dim,
};
pub use fastconformer_core::encoder::EncoderOutput;
use fastconformer_core::Config;
use gguf_loader::GgufFile;

pub fn load(gguf: &GgufFile, cfg: Config) -> Result<FastConformerEncoder, String> {
    let dm = cfg.d_model as usize;
    let n_heads = cfg.n_heads as usize;
    let conv_kernel = cfg.conv_kernel_size as usize;
    let n_mels = cfg.n_mels as usize;
    let n_mels_out = subsampled_mel_dim(n_mels);

    let conv0_weight = gguf.dequantize_f32("enc.subsample.conv.0.weight")?;
    let conv0_bias = gguf.dequantize_f32("enc.subsample.conv.0.bias")?;
    let conv1_weight = gguf.dequantize_f32("enc.subsample.conv.1.weight")?;
    let conv1_bias = gguf.dequantize_f32("enc.subsample.conv.1.bias")?;
    let conv2_weight = gguf.dequantize_f32("enc.subsample.conv.2.weight")?;
    let conv2_bias = gguf.dequantize_f32("enc.subsample.conv.2.bias")?;
    let out_weight = gguf.dequantize_f32("enc.subsample.out.weight")?;
    let out_bias = gguf.dequantize_f32("enc.subsample.out.bias")?;
    let channels = conv0_bias.len();

    let subsample = Subsample {
        n_mels,
        d_model: dm,
        n_mels_out,
        channels,
        conv0_weight,
        conv0_bias,
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        out_weight,
        out_bias,
    };

    let mut blocks = Vec::with_capacity(cfg.n_layers as usize);
    for l in 0..cfg.n_layers as usize {
        blocks.push(load_block(gguf, l, dm, n_heads, conv_kernel)?);
    }

    let ln_post_gamma = gguf.dequantize_f32("enc.ln_post.weight")?;
    let ln_post_beta = gguf.dequantize_f32("enc.ln_post.bias")?;

    Ok(FastConformerEncoder {
        cfg,
        subsample,
        blocks,
        ln_post_gamma,
        ln_post_beta,
    })
}

fn load_block(
    gguf: &GgufFile,
    l: usize,
    dm: usize,
    n_heads: usize,
    conv_kernel: usize,
) -> Result<ConformerBlock, String> {
    let pfx = format!("enc.block.{l}");
    let head_dim = dm / n_heads;

    let ff1 = FeedForward {
        ln_gamma: gguf.dequantize_f32(&format!("{pfx}.ff1.ln.weight"))?,
        ln_beta: gguf.dequantize_f32(&format!("{pfx}.ff1.ln.bias"))?,
        fc1_weight: gguf.dequantize_f32(&format!("{pfx}.ff1.fc1.weight"))?,
        fc1_bias: gguf.dequantize_f32(&format!("{pfx}.ff1.fc1.bias"))?,
        fc2_weight: gguf.dequantize_f32(&format!("{pfx}.ff1.fc2.weight"))?,
        fc2_bias: gguf.dequantize_f32(&format!("{pfx}.ff1.fc2.bias"))?,
        d_model: dm,
    };
    let attn = MultiHeadAttention {
        d_model: dm,
        n_heads,
        head_dim,
        ln_gamma: gguf.dequantize_f32(&format!("{pfx}.attn.ln.weight"))?,
        ln_beta: gguf.dequantize_f32(&format!("{pfx}.attn.ln.bias"))?,
        q_weight: gguf.dequantize_f32(&format!("{pfx}.attn.q.weight"))?,
        q_bias: gguf
            .dequantize_f32(&format!("{pfx}.attn.q.bias"))
            .unwrap_or_else(|_| vec![0.0; dm]),
        k_weight: gguf.dequantize_f32(&format!("{pfx}.attn.k.weight"))?,
        v_weight: gguf.dequantize_f32(&format!("{pfx}.attn.v.weight"))?,
        v_bias: gguf
            .dequantize_f32(&format!("{pfx}.attn.v.bias"))
            .unwrap_or_else(|_| vec![0.0; dm]),
        out_weight: gguf.dequantize_f32(&format!("{pfx}.attn.out.weight"))?,
        out_bias: gguf
            .dequantize_f32(&format!("{pfx}.attn.out.bias"))
            .unwrap_or_else(|_| vec![0.0; dm]),
        pos_weight: gguf.dequantize_f32(&format!("{pfx}.attn.pos.weight"))?,
        pos_bias_u: gguf.dequantize_f32(&format!("{pfx}.attn.pos_bias_u"))?,
        pos_bias_v: gguf.dequantize_f32(&format!("{pfx}.attn.pos_bias_v"))?,
    };
    let conv = ConvModule {
        d_model: dm,
        conv_kernel_size: conv_kernel,
        ln_gamma: gguf.dequantize_f32(&format!("{pfx}.conv.ln.weight"))?,
        ln_beta: gguf.dequantize_f32(&format!("{pfx}.conv.ln.bias"))?,
        pw1_weight: gguf.dequantize_f32(&format!("{pfx}.conv.pw1.weight"))?,
        pw1_bias: gguf
            .dequantize_f32(&format!("{pfx}.conv.pw1.bias"))
            .unwrap_or_else(|_| vec![0.0; 2 * dm]),
        dw_weight: gguf.dequantize_f32(&format!("{pfx}.conv.dw.weight"))?,
        dw_bias: gguf
            .dequantize_f32(&format!("{pfx}.conv.dw.bias"))
            .unwrap_or_else(|_| vec![0.0; dm]),
        bn_gamma: gguf.dequantize_f32(&format!("{pfx}.conv.bn.weight"))?,
        bn_beta: gguf.dequantize_f32(&format!("{pfx}.conv.bn.bias"))?,
        bn_running_mean: gguf.dequantize_f32(&format!("{pfx}.conv.bn.running_mean"))?,
        bn_running_var: gguf.dequantize_f32(&format!("{pfx}.conv.bn.running_var"))?,
        pw2_weight: gguf.dequantize_f32(&format!("{pfx}.conv.pw2.weight"))?,
        pw2_bias: gguf
            .dequantize_f32(&format!("{pfx}.conv.pw2.bias"))
            .unwrap_or_else(|_| vec![0.0; dm]),
    };
    let ff2 = FeedForward {
        ln_gamma: gguf.dequantize_f32(&format!("{pfx}.ff2.ln.weight"))?,
        ln_beta: gguf.dequantize_f32(&format!("{pfx}.ff2.ln.bias"))?,
        fc1_weight: gguf.dequantize_f32(&format!("{pfx}.ff2.fc1.weight"))?,
        fc1_bias: gguf.dequantize_f32(&format!("{pfx}.ff2.fc1.bias"))?,
        fc2_weight: gguf.dequantize_f32(&format!("{pfx}.ff2.fc2.weight"))?,
        fc2_bias: gguf.dequantize_f32(&format!("{pfx}.ff2.fc2.bias"))?,
        d_model: dm,
    };
    Ok(ConformerBlock {
        ff1,
        attn,
        conv,
        ff2,
        ln_post_gamma: gguf.dequantize_f32(&format!("{pfx}.ln_post.weight"))?,
        ln_post_beta: gguf.dequantize_f32(&format!("{pfx}.ln_post.bias"))?,
        d_model: dm,
    })
}
