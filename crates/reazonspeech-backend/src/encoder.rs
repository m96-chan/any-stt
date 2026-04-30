//! ReazonSpeech encoder wrapper.
//!
//! Re-exports `fastconformer_core::encoder::FastConformerEncoder` and adds
//! a thin `from_gguf` loader specific to the reazonspeech-nemo-v2 tensor
//! naming. The encoder forward itself is shared with parakeet-backend.

use fastconformer_core::encoder::{
    AttentionMode, ConformerBlock, ConvModule, FastConformerEncoder, FeedForward,
    MultiHeadAttention, Subsample, SubsampleDwStriding, SubsampleStriding,
    subsampled_mel_dim,
};
pub use fastconformer_core::encoder::EncoderOutput;
use fastconformer_core::config::AttentionType;
use fastconformer_core::Config;
use gguf_loader::GgufFile;

/// Build a FastConformer encoder from a reazonspeech-nemo-v2 GGUF file.
///
/// The subsample stack is auto-detected from the tensor list:
/// - `enc.subsample.conv.{0,1,2}` only → plain striding (parakeet-style)
/// - `enc.subsample.conv.{0,2,3,5,6}` → dw_striding (reazonspeech-style)
pub fn load(gguf: &GgufFile, cfg: Config) -> Result<FastConformerEncoder, String> {
    let dm = cfg.d_model as usize;
    let n_heads = cfg.n_heads as usize;
    let conv_kernel = cfg.conv_kernel_size as usize;
    let n_mels = cfg.n_mels as usize;
    let n_mels_out = subsampled_mel_dim(n_mels);

    let out_weight = gguf.dequantize_f32("enc.subsample.out.weight")?;
    let out_bias = gguf.dequantize_f32("enc.subsample.out.bias")?;

    let subsample = if gguf.has_tensor("enc.subsample.conv.3.weight") {
        // dw_striding: conv.{0, 2, 3, 5, 6}.
        let conv0_weight = gguf.dequantize_f32("enc.subsample.conv.0.weight")?;
        let conv0_bias = gguf.dequantize_f32("enc.subsample.conv.0.bias")?;
        let dw1_weight = gguf.dequantize_f32("enc.subsample.conv.2.weight")?;
        let dw1_bias = gguf.dequantize_f32("enc.subsample.conv.2.bias")?;
        let pw1_weight = gguf.dequantize_f32("enc.subsample.conv.3.weight")?;
        let pw1_bias = gguf.dequantize_f32("enc.subsample.conv.3.bias")?;
        let dw2_weight = gguf.dequantize_f32("enc.subsample.conv.5.weight")?;
        let dw2_bias = gguf.dequantize_f32("enc.subsample.conv.5.bias")?;
        let pw2_weight = gguf.dequantize_f32("enc.subsample.conv.6.weight")?;
        let pw2_bias = gguf.dequantize_f32("enc.subsample.conv.6.bias")?;
        let channels = conv0_bias.len();
        Subsample::DwStriding(SubsampleDwStriding {
            n_mels,
            d_model: dm,
            n_mels_out,
            channels,
            conv0_weight,
            conv0_bias,
            dw1_weight,
            dw1_bias,
            pw1_weight,
            pw1_bias,
            dw2_weight,
            dw2_bias,
            pw2_weight,
            pw2_bias,
            out_weight,
            out_bias,
        })
    } else {
        // Plain striding: conv.{0, 1, 2}.
        let conv0_weight = gguf.dequantize_f32("enc.subsample.conv.0.weight")?;
        let conv0_bias = gguf.dequantize_f32("enc.subsample.conv.0.bias")?;
        let conv1_weight = gguf.dequantize_f32("enc.subsample.conv.1.weight")?;
        let conv1_bias = gguf.dequantize_f32("enc.subsample.conv.1.bias")?;
        let conv2_weight = gguf.dequantize_f32("enc.subsample.conv.2.weight")?;
        let conv2_bias = gguf.dequantize_f32("enc.subsample.conv.2.bias")?;
        let channels = conv0_bias.len();
        Subsample::Striding(SubsampleStriding {
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
        })
    };

    // --- Conformer blocks ---
    let attn_mode = match cfg.attention_type {
        AttentionType::RelPos => AttentionMode::Full,
        AttentionType::RelPosLocalAttn => AttentionMode::LocalGlobal {
            local_window: cfg.local_window as usize,
            global_tokens: cfg.global_tokens as usize,
        },
    };
    let mut blocks = Vec::with_capacity(cfg.n_layers as usize);
    for l in 0..cfg.n_layers as usize {
        let block = load_block(gguf, l, dm, n_heads, conv_kernel, attn_mode)?;
        blocks.push(block);
    }

    // NeMo Conformer's encoder doesn't have a final post-LN; each block
    // owns its own. Default to identity (γ=1, β=0) when absent.
    let ln_post_gamma = gguf
        .dequantize_f32("enc.ln_post.weight")
        .unwrap_or_else(|_| vec![1.0; dm]);
    let ln_post_beta = gguf
        .dequantize_f32("enc.ln_post.bias")
        .unwrap_or_else(|_| vec![0.0; dm]);

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
    attn_mode: AttentionMode,
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
        attn_mode,
        conv,
        ff2,
        ln_post_gamma: gguf.dequantize_f32(&format!("{pfx}.ln_post.weight"))?,
        ln_post_beta: gguf.dequantize_f32(&format!("{pfx}.ln_post.bias"))?,
        d_model: dm,
    })
}
