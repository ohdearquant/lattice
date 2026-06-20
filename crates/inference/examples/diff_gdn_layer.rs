//! Differential test: the materialised GatedDeltaNet forward
//! (`attention::gdn_backward::gdn_forward_save`) vs the real Qwen3.5 GDN mixer
//! at a linear-attention layer.
//!
//! Proves forward-matches-real-model (max-diff < 1e-3) for the GDN block, which
//! the self-consistent backward gradcheck cannot: that gradcheck only validates
//! that `gdn_backward` is the true VJP of `gdn_forward_save`, for whatever
//! convention `gdn_forward_save` uses. If `gdn_forward_save` diverges from the
//! real model's `gated_delta_net_step_fused` (wrong conv layout, wrong gate,
//! wrong recurrence), the gradcheck still passes yet the full-depth tape would
//! propagate the wrong dx through the 18 frozen GDN layers. This runner is the
//! discipline that closes that gap — the GDN analogue of `diff_attn_layer23`.
//!
//! Requires the on-disk model, so it is a manual gate (like `bench-compare`),
//! not a CI unit test.
//!
//! ```text
//! cargo run -p lattice-inference --release --features train-backward,f16 \
//!     --example diff_gdn_layer
//! ```

use std::path::PathBuf;

use lattice_inference::attention::gdn_backward::{GdnSaved, gdn_forward_save};
use lattice_inference::model::qwen35::Qwen35Model;

fn main() {
    let model_dir = std::env::var("LATTICE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME unset");
            PathBuf::from(home).join(".lattice/models/qwen3.5-0.8b")
        });
    // Layer 2 is a GatedDeltaNet (linear-attention) layer — the GQA/Full layers
    // are at [3,7,11,15,19,23], so 0,1,2 are GDN. The full-depth tape must
    // backprop dx through these frozen GDN layers, so their forward must match.
    let layer = std::env::var("LATTICE_GDN_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2usize);

    let model = Qwen35Model::from_safetensors(&model_dir)
        .unwrap_or_else(|e| panic!("load model {}: {e}", model_dir.display()));
    let cfg = model.config();
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    // Arbitrary valid token ids — the diff test exercises the GDN code path,
    // not the embedding table, so real text is unnecessary. GDN is recurrent,
    // so a longer sequence exercises more of the cross-time state flow.
    let tokens: Vec<u32> = vec![
        1, 42, 7, 100, 2048, 9, 256, 3, 88, 17, 1024, 5, 64, 200, 11, 99,
    ];
    let seq_len = tokens.len();

    // Real model: capture layer `layer`'s pre-input-layernorm residual h_in and
    // its GDN mixer output (before the residual add), per position. The capture
    // tap fires for any layer kind, so for a GDN layer attn_out is the mixer out.
    let (h_in, real_out) = model
        .capture_attn_io(&tokens, layer)
        .unwrap_or_else(|e| panic!("capture_attn_io: {e}"));

    let (gdn_w, input_layernorm) = model
        .gdn_layer_weights(layer)
        .unwrap_or_else(|| panic!("layer {layer} is not a GatedDeltaNet layer"));

    // Recompute normed = shifted rms_norm(h_in, input_layernorm) — the
    // post-input-layernorm mixer input the real model feeds to the GDN. The
    // (1 + gamma) shift matches qwen35_rms_norm.
    let mut normed = h_in.clone();
    for row in normed.chunks_mut(hidden) {
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
        for (v, &g) in row.iter_mut().zip(input_layernorm.iter()) {
            *v = *v * inv * (1.0 + g);
        }
    }

    let num_kh = cfg.linear_num_key_heads;
    let value_heads = cfg.linear_num_value_heads();
    let key_dim = cfg.linear_key_head_dim;
    let value_dim = cfg.linear_value_head_dim;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let kernel_size = cfg.linear_conv_kernel_dim;
    let scale = 1.0 / (key_dim as f32).sqrt();

    let mut saved = GdnSaved::new(
        seq_len,
        num_kh,
        value_heads,
        key_dim,
        value_dim,
        hidden,
        qkv_dim,
        output_dim,
        kernel_size,
        scale,
        eps,
    );
    let mut mat_out = vec![0.0f32; seq_len * hidden];
    gdn_forward_save(&normed, gdn_w, cfg, &mut saved, &mut mat_out);

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

    println!("GDN layer {layer}: seq_len={seq_len} hidden={hidden}");
    println!(
        "  dims: num_kh={num_kh} value_heads={value_heads} key_dim={key_dim} \
         value_dim={value_dim} qkv_dim={qkv_dim} output_dim={output_dim} kernel={kernel_size}"
    );
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
        println!("PASS: materialised GDN forward matches real model (< 1e-3)");
    } else {
        println!("FAIL: divergence >= 1e-3 — materialised GDN forward does NOT match real model");
        std::process::exit(1);
    }
}
