//! Differential test: the materialised GQA forward (`backward::attention_gqa`)
//! vs the real Qwen3.5 attention block at a Full-attention layer.
//!
//! Proves forward-matches-real-model (max-diff < 1e-3), which the self-consistent
//! backward gradcheck cannot: that gradcheck validates backward-matches-forward
//! for whatever convention the materialised forward uses, so a forward that
//! diverges from the real model (wrong gate handling, plain-vs-shifted q_norm,
//! wrong rope pairing) passes the gradcheck yet trains against the wrong layer.
//! This runner is the discipline that closes that gap.
//!
//! Requires the on-disk model, so it is a manual gate (like `bench-compare`),
//! not a CI unit test.
//!
//! ```text
//! cargo run -p lattice-inference --release --features train-backward,f16 \
//!     --example diff_attn_layer23
//! ```

use std::path::PathBuf;

use lattice_inference::backward::attention_gqa::gqa_forward_with_cache;
use lattice_inference::model::qwen35::Qwen35Model;

fn main() {
    let model_dir = std::env::var("LATTICE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME unset");
            PathBuf::from(home).join(".lattice/models/qwen3.5-0.8b")
        });
    // Layer 23 is the top Full-attention (GQA) layer — the layer-23 LoRA
    // trainer's target, so it is the one whose forward must be verified.
    let layer = 23usize;

    let model = Qwen35Model::from_safetensors(&model_dir)
        .unwrap_or_else(|e| panic!("load model {}: {e}", model_dir.display()));
    let cfg = model.config();
    let hidden = cfg.hidden_size;
    let num_q_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let rope_dim = cfg.rope_dim();
    let eps = cfg.rms_norm_eps;

    // Arbitrary valid token ids — the diff test exercises the attention code
    // path, not the embedding table, so real text is unnecessary.
    let tokens: Vec<u32> = vec![1, 42, 7, 100, 2048, 9, 256, 3, 88, 17, 1024, 5];
    let seq_len = tokens.len();

    // Real model: capture layer 23's pre-input-layernorm residual h_in and its
    // gated o_proj output, per position.
    let (h_in, real_out) = model
        .capture_attn_io(&tokens, layer)
        .unwrap_or_else(|e| panic!("capture_attn_io: {e}"));

    let (w_q, w_k, w_v, w_o, q_norm, k_norm, pre_attn_norm, _post, _g, _u, _d) = model
        .gqa_layer_weights(layer)
        .expect("layer 23 is a Full+Dense GQA layer");

    // Recompute normed = shifted rms_norm(h_in, pre_attn_norm) — the
    // post-input-layernorm attention input the materialised forward expects.
    // The (1 + gamma) shift matches qwen35_rms_norm; this also gives the diff
    // test input-layernorm parity coverage.
    let mut normed = h_in.clone();
    for row in normed.chunks_mut(hidden) {
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for (v, &g) in row.iter_mut().zip(pre_attn_norm.iter()) {
            *v = *v * inv * (1.0 + g);
        }
    }

    let (cos, sin) = model.rope_cos_sin_tables(seq_len);

    let (mat_out, _cache) = gqa_forward_with_cache(
        &normed,
        w_q,
        w_k,
        w_v,
        w_o,
        q_norm,
        k_norm,
        None,
        None,
        None,
        None,
        0,
        0.0,
        seq_len,
        hidden,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rope_dim,
        &cos,
        &sin,
        eps,
    );

    assert_eq!(mat_out.len(), real_out.len(), "output length mismatch");
    let (mut max_diff, mut argmax) = (0.0f32, 0usize);
    for (i, (a, b)) in mat_out.iter().zip(real_out.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            argmax = i;
        }
    }
    let mean_abs: f32 = real_out.iter().map(|v| v.abs()).sum::<f32>() / real_out.len() as f32;

    println!("layer {layer}: seq_len={seq_len} hidden={hidden}");
    println!(
        "  max |materialised - real| = {max_diff:.3e}  at pos {}, dim {}",
        argmax / hidden,
        argmax % hidden
    );
    println!("  mean |real|               = {mean_abs:.3e}");
    println!(
        "  relative                  = {:.3e}",
        max_diff / mean_abs.max(1e-9)
    );

    if max_diff < 1e-3 {
        println!("PASS: materialised GQA forward matches real model (< 1e-3)");
    } else {
        println!("FAIL: divergence >= 1e-3 — materialised forward does NOT match real model");
        std::process::exit(1);
    }
}
