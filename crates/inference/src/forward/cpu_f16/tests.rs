use super::*;
use crate::decoder::standalone_cpu::parity;

#[test]
#[allow(clippy::type_complexity)]
fn test_f16_forward_compiles() {
    // Verify the function signatures are correct by constructing the types
    // and calling the functions with a trivial (1-layer, tiny) config.
    let cfg = Qwen35Config::qwen35_2b();

    // Verify forward_step_f16 signature (pub(crate))
    let _fn_ptr: fn(
        &F16ModelWeights,
        &Qwen35Config,
        &RopeTable,
        u32,
        usize,
        &mut [GatedDeltaNetState],
        &mut KvCache,
        &mut ForwardScratch,
        Option<&[f32]>,
        Option<(&[f32], &[f32])>,
    ) -> Result<(), crate::error::InferenceError> = forward_step_f16;

    // Verify gated_delta_net_step_fused_f16 signature
    let _gdn_fn_ptr: fn(
        &[f32],
        &mut GatedDeltaNetState,
        &F16GatedDeltaNetWeights,
        &Qwen35Config,
        &mut GatedDeltaNetFusedScratch,
        &mut [f32],
    ) = gated_delta_net_step_fused_f16;

    // Verify generate_f16 returns the right type
    let _gen_fn_ptr: fn(
        &F16ModelWeights,
        &Qwen35Config,
        &BpeTokenizer,
        &RopeTable,
        &str,
        &GenerateConfig,
    ) -> Result<GenerateOutput, crate::error::InferenceError> = generate_f16;

    // Verify the config helpers work
    assert!(cfg.num_full_attention_layers() > 0);
    assert!(cfg.num_linear_attention_layers() > 0);
    assert_eq!(
        cfg.num_full_attention_layers() + cfg.num_linear_attention_layers(),
        cfg.num_hidden_layers
    );
}

/// Regression test for #392: cpu F16 RoPE must use stride-half pairing (i, half+i), not
/// interleaved (2i, 2i+1).
///
/// Design: call `full_attention_step_f16` with an identity K-projection so k_buf equals
/// the input exactly (1.0 in f16 is exact, no rounding error on the diagonal).
/// Independently reproduce the same matmul + QK-norm + stride-half RoPE in the test body
/// and compare against the post-call KV-cache. The two paths agree to <1e-4 when the
/// production loops are correct; reverting either loop to 2*i interleaved produces
/// max_diff ~0.9 (observed during mutation verification).
#[test]
fn test_full_attn_step_f16_rope_stride_half_parity() {
    use crate::model::qwen35_config::LayerType;
    use crate::weights::f16_weights::f32_to_f16_slice;

    let head_dim: usize = 32;
    let num_q_heads: usize = 1;
    let num_kv_heads: usize = 1;
    let hidden: usize = 64;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let position: usize = 3;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 2,
        vocab_size: 128,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        num_attention_heads: num_q_heads,
        num_key_value_heads: num_kv_heads,
        head_dim,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5, // rope_dim = 16, half = 8
        rope_parameters: None,
        linear_num_key_heads: 2,
        linear_num_value_heads: Some(2),
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        layer_mask: vec![true; 2],
        eos_token_id: 127,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let rope_dim = cfg.rope_dim(); // = 16
    let half = rope_dim / 2; // = 8
    let rope = RopeTable::new(rope_dim, 512, cfg.rope_theta);

    // Helper: convert f32 slice to packed f16 Vec<u16>.
    let to_f16 = |src: &[f32]| -> Vec<u16> {
        let mut dst = vec![0u16; src.len()];
        f32_to_f16_slice(src, &mut dst);
        dst
    };

    // W_k = identity [kv_dim, hidden]: row j selects input[j] exactly (1.0 in f16 is exact).
    let mut k_proj_f32 = vec![0.0f32; kv_dim * hidden];
    for j in 0..kv_dim {
        k_proj_f32[j * hidden + j] = 1.0;
    }

    // W_q = identity for first q_dim rows (Q part), zeros for next q_dim rows (gate part).
    // Row j selects input[j] exactly so scratch.q_buf is non-trivial and Q-loop mutation
    // changes the assertion result.
    let mut q_proj_f32 = vec![0.0f32; 2 * q_dim * hidden];
    for j in 0..q_dim {
        q_proj_f32[j * hidden + j] = 1.0;
    }

    let weights = F16FullAttentionLayerWeights {
        q_proj: to_f16(&q_proj_f32),
        k_proj: to_f16(&k_proj_f32),
        v_proj: to_f16(&vec![0.0f32; kv_dim * hidden]),
        o_proj: to_f16(&vec![0.0f32; hidden * q_dim]),
        q_norm: vec![0.0f32; head_dim],
        k_norm: vec![0.0f32; head_dim],
    };

    // Distinct non-trivial input values (positions 0..64 scaled to small floats).
    let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.07).collect();

    let mut scratch = ForwardScratch::new();
    scratch.ensure_capacity(&cfg, 2);
    scratch.attn_out[..hidden].copy_from_slice(&input);

    let mut kv_cache = KvCache::new(1);
    full_attention_step_f16(
        &weights,
        0,
        position,
        &mut kv_cache,
        &mut scratch,
        &cfg,
        &rope,
        hidden,
        None,
    );

    // Reference: reproduce the same matmul + QK-norm + stride-half RoPE.
    // Using the same production matmul and norm keeps f16 rounding error identical on
    // both sides, so the only source of divergence under mutation is the RoPE pairing.

    // --- K reference ---
    let mut k_ref = vec![0.0f32; kv_dim];
    matmul_bt_f16(&input, &weights.k_proj, &mut k_ref, 1, hidden, kv_dim);
    qwen35_rms_norm(&mut k_ref, &weights.k_norm, head_dim, cfg.rms_norm_eps);

    // Stride-half reference (correct pairing).
    let base = position * half;
    for i in 0..half {
        let cos_val = rope.cos_at(base + i);
        let sin_val = rope.sin_at(base + i);
        let x0 = k_ref[i];
        let x1 = k_ref[half + i];
        k_ref[i] = x0 * cos_val - x1 * sin_val;
        k_ref[half + i] = x0 * sin_val + x1 * cos_val;
    }

    let k_cached = &kv_cache.k[0][..kv_dim];
    let max_k_diff = k_cached
        .iter()
        .zip(k_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_k_diff < 1e-4,
        "cpu F16 K-loop stride-half RoPE diverges from reference: max_k_diff = {max_k_diff:.6}. \
             With interleaved pairing the diff is O(0.1-1). Bug: #392."
    );

    // --- Q reference (guards the Q loop mutation) ---
    // The production scatter copies q_and_gate[0..q_dim] → scratch.q_buf[0..q_dim]
    // for head 0 (num_q_heads=1).
    let mut q_and_gate_ref = vec![0.0f32; 2 * q_dim];
    matmul_bt_f16(
        &input,
        &weights.q_proj,
        &mut q_and_gate_ref,
        1,
        hidden,
        2 * q_dim,
    );
    let mut q_ref = q_and_gate_ref[..q_dim].to_vec();
    qwen35_rms_norm(&mut q_ref, &weights.q_norm, head_dim, cfg.rms_norm_eps);

    for i in 0..half {
        let cos_val = rope.cos_at(base + i);
        let sin_val = rope.sin_at(base + i);
        let x0 = q_ref[i];
        let x1 = q_ref[half + i];
        q_ref[i] = x0 * cos_val - x1 * sin_val;
        q_ref[half + i] = x0 * sin_val + x1 * cos_val;
    }

    let max_q_diff = scratch.q_buf[..q_dim]
        .iter()
        .zip(q_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_q_diff < 1e-4,
        "cpu F16 Q-loop stride-half RoPE diverges from reference: max_q_diff = {max_q_diff:.6}. \
             With interleaved pairing the diff is O(0.1-1). Bug: #392."
    );
}

/// ADR-080 C1 (#780): `full_attention_step_f16`'s decode-attention row
/// finalizer must fail closed on a NaN score, not propagate the bare
/// `1.0 / sum_exp` into the context output. A prior cached position is
/// poisoned (NaN K); the current position's own K/V stay finite. RED
/// before the fix: the poisoned row's NaN leaked into every context lane.
#[test]
fn test_full_attn_step_f16_nan_cached_score_fails_closed() {
    use crate::model::qwen35_config::LayerType;
    use crate::weights::f16_weights::f32_to_f16_slice;

    let head_dim: usize = 32;
    let num_q_heads: usize = 1;
    let num_kv_heads: usize = 1;
    let hidden: usize = 64;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let position: usize = 1;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 2,
        vocab_size: 128,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        num_attention_heads: num_q_heads,
        num_key_value_heads: num_kv_heads,
        head_dim,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: 2,
        linear_num_value_heads: Some(2),
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        layer_mask: vec![true; 2],
        eos_token_id: 127,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let rope = RopeTable::new(cfg.rope_dim(), 512, cfg.rope_theta);

    let to_f16 = |src: &[f32]| -> Vec<u16> {
        let mut dst = vec![0u16; src.len()];
        f32_to_f16_slice(src, &mut dst);
        dst
    };

    // W_k = identity, W_q = identity for the Q half (gate half zero), W_v
    // = identity too (so the CURRENT token's V is a distinct, finite,
    // predictable vector rather than zero -- confirms the fail-closed
    // path isn't trivially passing because every V is zero anyway).
    let mut k_proj_f32 = vec![0.0f32; kv_dim * hidden];
    for j in 0..kv_dim {
        k_proj_f32[j * hidden + j] = 1.0;
    }
    let mut v_proj_f32 = vec![0.0f32; kv_dim * hidden];
    for j in 0..kv_dim {
        v_proj_f32[j * hidden + j] = 1.0;
    }
    let mut q_proj_f32 = vec![0.0f32; 2 * q_dim * hidden];
    for j in 0..q_dim {
        q_proj_f32[j * hidden + j] = 1.0;
    }

    let weights = F16FullAttentionLayerWeights {
        q_proj: to_f16(&q_proj_f32),
        k_proj: to_f16(&k_proj_f32),
        v_proj: to_f16(&v_proj_f32),
        o_proj: to_f16(&vec![0.0f32; hidden * q_dim]),
        q_norm: vec![0.0f32; head_dim],
        k_norm: vec![0.0f32; head_dim],
    };

    let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.07).collect();

    let mut scratch = ForwardScratch::new();
    scratch.ensure_capacity(&cfg, 2);
    scratch.attn_out[..hidden].copy_from_slice(&input);

    // Pre-load one poisoned cached position (position 0): NaN K, finite V.
    let mut kv_cache = KvCache::new(1);
    let mut poisoned_k = vec![0.0f32; kv_dim];
    poisoned_k[0] = f32::NAN;
    let finite_v = vec![5.0f32; kv_dim];
    kv_cache.append_kv(0, &poisoned_k, &finite_v);
    kv_cache.seq_len = 1;

    full_attention_step_f16(
        &weights,
        0,
        position,
        &mut kv_cache,
        &mut scratch,
        &cfg,
        &rope,
        hidden,
        None,
    );

    assert!(
        scratch.context[..head_dim].iter().all(|&v| v == 0.0),
        "expected exact-zero context for a NaN-poisoned cached score, \
             got {:?}",
        &scratch.context[..head_dim]
    );
}

#[test]
fn test_gdn_f16_step_with_zeros() {
    // Run the GDN f16 step with zero weights/inputs to verify it doesn't crash.
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let num_heads = cfg.linear_num_key_heads;
    let kernel_size = cfg.linear_conv_kernel_dim;

    let weights = F16GatedDeltaNetWeights {
        in_proj_qkv: vec![0u16; qkv_dim * hidden],
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: vec![0u16; output_dim * hidden],
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: vec![0u16; num_heads * hidden],
        in_proj_b_rows: num_heads,
        in_proj_b_cols: hidden,
        in_proj_a: vec![0u16; num_heads * hidden],
        in_proj_a_rows: num_heads,
        in_proj_a_cols: hidden,
        a_log: vec![0.0f32; num_heads],
        dt_bias: vec![0.0f32; num_heads],
        conv1d_weight: vec![0.0f32; qkv_dim * kernel_size],
        conv_dim: qkv_dim,
        kernel_size,
        norm_weight: vec![0.0f32; output_dim],
        out_proj: vec![0u16; hidden * output_dim],
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    };

    let mut state = GatedDeltaNetState::new(&cfg);
    let mut scratch = GatedDeltaNetFusedScratch::default();
    let input = vec![0.0f32; hidden];
    let mut output = vec![0.0f32; hidden];

    gated_delta_net_step_fused_f16(
        &input,
        &mut state,
        &weights,
        &cfg,
        &mut scratch,
        &mut output,
    );

    // With all-zero weights and input, output should be all zeros
    for &v in &output[..hidden] {
        assert_eq!(
            v, 0.0,
            "zero weights + zero input should produce zero output"
        );
    }
}

/// A NaN reaching the f16 MoE router (corrupt f16 router gate weight or an
/// upstream activation overflow) makes every router logit NaN. Before the
/// fail-closed guards `moe_ffn_step_f16` left NaN probabilities (the
/// `denom > 0.0` path was skipped), top-k selected nothing (`NaN > NEG_INF`
/// is false), and the routed-expert loop indexed expert weights at
/// `usize::MAX * stride` → overflow/OOB panic on the (bench-only public)
/// `generate_f16` path. This is the f16 sibling of the f32 fix in
/// qwen35/moe.rs (#410). The router must fail closed and no `usize::MAX`
/// sentinel may reach accumulation.
#[test]
fn test_moe_ffn_step_f16_nan_router_fails_closed_no_panic() {
    use crate::weights::f16_weights::{
        F16, F16MoeLayerWeights, F16MoeRouter, F16RoutedExperts, F16SharedExpert,
    };
    let num_experts = 4usize;
    let hidden = 4usize;
    let inter = 2usize;
    let shared_inter = 2usize;
    let top_k = 2usize;

    let nan16 = F16::from_f32(f32::NAN).0;
    let zeros = |n: usize| vec![F16::from_f32(0.0).0; n];

    // Every router gate weight is NaN → every router logit is NaN.
    let router = F16MoeRouter::new(
        vec![nan16; num_experts * hidden],
        num_experts,
        top_k,
        hidden,
    )
    .unwrap();
    let experts = F16RoutedExperts::new(
        zeros(num_experts * 2 * inter * hidden),
        zeros(num_experts * hidden * inter),
        num_experts,
        hidden,
        inter,
    )
    .unwrap();
    let shared = F16SharedExpert::new(
        zeros(shared_inter * hidden),
        zeros(shared_inter * hidden),
        zeros(hidden * shared_inter),
        zeros(hidden),
        hidden,
        shared_inter,
    )
    .unwrap();
    let moe = F16MoeLayerWeights {
        router,
        experts,
        shared_expert: shared,
    };

    let mut scratch = ForwardScratch::new();
    let buf = inter.max(shared_inter);
    scratch.ffn_out.resize(hidden, 1.0);
    scratch.input_tmp.resize(hidden, 0.0);
    scratch.expert_out.resize(hidden, 0.0);
    scratch.gate_buf.resize(buf, 0.0);
    scratch.up_buf.resize(buf, 0.0);
    scratch.down_input.resize(buf, 0.0);
    scratch.router_logits.resize(num_experts, 0.0);
    scratch.router_selected.resize(top_k, (usize::MAX, 0.0));

    // Must not panic (was: usize::MAX expert_id → OOB slice / overflow).
    moe_ffn_step_f16(&moe, &mut scratch, hidden);

    assert!(
        scratch.router_selected[..top_k]
            .iter()
            .all(|(id, _)| *id < num_experts),
        "degenerate f16 router must not leave a usize::MAX sentinel selected"
    );
}

/// The non-obvious case the denom-else (not a max-only guard) is meant to
/// catch: ONE router row is NaN while the rest are finite, so `max_logit`
/// stays finite (Rust `f32::max` ignores a single NaN) but the NaN still
/// lands in `denom`. A max-only guard (`if !max_logit.is_finite()`) would
/// pass this and leave un-normalized raw `exp` mass; the denom-else fills
/// the row with 0.0. f16 analogue of qwen35/moe.rs
/// `test_moe_router_finite_max_nan_tail_fails_closed`.
#[test]
fn test_moe_ffn_step_f16_finite_max_nan_tail_fails_closed() {
    use crate::weights::f16_weights::{
        F16, F16MoeLayerWeights, F16MoeRouter, F16RoutedExperts, F16SharedExpert,
    };
    let num_experts = 4usize;
    let hidden = 4usize;
    let inter = 2usize;
    let shared_inter = 2usize;
    let top_k = 2usize;

    let nan16 = F16::from_f32(f32::NAN).0;
    let zeros = |n: usize| vec![F16::from_f32(0.0).0; n];

    // Only expert 0's gate row is NaN → logit[0] = NaN, logit[1..] finite,
    // so `max_logit` is finite but `denom` is NaN.
    let mut gate = zeros(num_experts * hidden);
    for w in &mut gate[..hidden] {
        *w = nan16;
    }
    let router = F16MoeRouter::new(gate, num_experts, top_k, hidden).unwrap();
    let experts = F16RoutedExperts::new(
        zeros(num_experts * 2 * inter * hidden),
        zeros(num_experts * hidden * inter),
        num_experts,
        hidden,
        inter,
    )
    .unwrap();
    let shared = F16SharedExpert::new(
        zeros(shared_inter * hidden),
        zeros(shared_inter * hidden),
        zeros(hidden * shared_inter),
        zeros(hidden),
        hidden,
        shared_inter,
    )
    .unwrap();
    let moe = F16MoeLayerWeights {
        router,
        experts,
        shared_expert: shared,
    };

    let mut scratch = ForwardScratch::new();
    let buf = inter.max(shared_inter);
    scratch.ffn_out.resize(hidden, 1.0);
    scratch.input_tmp.resize(hidden, 0.0);
    scratch.expert_out.resize(hidden, 0.0);
    scratch.gate_buf.resize(buf, 0.0);
    scratch.up_buf.resize(buf, 0.0);
    scratch.down_input.resize(buf, 0.0);
    scratch.router_logits.resize(num_experts, 0.0);
    scratch.router_selected.resize(top_k, (usize::MAX, 0.0));

    moe_ffn_step_f16(&moe, &mut scratch, hidden);

    assert!(
        scratch.router_logits[..num_experts]
            .iter()
            .all(|p| *p == 0.0),
        "finite-max + NaN-tail router row must fail closed to all-zero probs \
             (a max-only guard would miss this)"
    );
}

// NOTE: the storage-bound guard (`expert_id >= moe.experts.num_experts`) is
// release-only defense-in-depth — in debug the `debug_assert_eq!(moe.experts
// .num_experts, num_experts)` at the top of `moe_ffn_step_f16` fires first on
// a router/expert count mismatch, so the guard cannot be exercised by a debug
// unit test. It mirrors the f32 sibling guard (qwen35/moe.rs) and costs one
// comparison; it hardens the bench-only public `generate_f16` path against a
// manually constructed f16 weight set whose router declares more experts than
// the routed-expert storage holds.

/// Build a zero-layer F16 model fixture for generate_f16 unit tests.
///
/// All-zero u16 (= f16 zero) embeddings → logits all 0 → greedy picks token 0.
/// eos_token_id = 5 so that greedy token 0 is NOT eos, making stop_token_ids=[0]
/// detectable as a distinct stop path.
fn zero_layer_f16_fixture() -> (Qwen35Config, F16ModelWeights, RopeTable, BpeTokenizer) {
    use std::collections::HashMap;

    let hidden = 4usize;
    let vocab = 8usize;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 0,
        vocab_size: vocab,
        intermediate_size: 4,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: 1,
        linear_num_value_heads: Some(1),
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![],
        layer_mask: vec![],
        // eos is 5 so that greedy token 0 is NOT eos — allows stop_token_ids=[0]
        // to be a distinct, detectable stop signal.
        eos_token_id: 5,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    // embed_tokens is [vocab * hidden] packed u16 (f16 zeros = 0u16).
    // All zeros → logits all 0 → greedy always picks token 0.
    let weights = F16ModelWeights {
        embed_tokens: vec![0u16; vocab * hidden],
        final_norm: vec![0.0f32; hidden],
        layers: vec![],
    };

    // rope_dim = head_dim * partial_rotary_factor = 4 * 0.5 = 2.
    let rope = RopeTable::new(2, 64, 10_000.0);

    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_map, merges).unwrap();

    (cfg, weights, rope, tokenizer)
}

// -----------------------------------------------------------------------
// generate_f16 output-stream goldens
// -----------------------------------------------------------------------
//
// A small (2-layer) synthetic Qwen3.5-shaped model with deterministic
// non-zero weights (an LCG, matching `cpu_q8::tests::make_nonzero_q8_cpu_test_model`
// and `neon_forward::tests::make_nonzero_q8_neon_test_model`), covering both
// hybrid layer kinds so the golden run actually exercises GatedDeltaNet
// (linear attention) and GQA (full attention) rather than a degenerate
// zero-layer stub. Unlike the zero-layer fixture above (used for preflight/
// guard tests, where the *only* interesting behaviour is what happens before
// any weight is touched), this fixture exists so the pinned token streams
// below characterize the actual decode loop.
//
// Regenerating the goldens: run
//   cargo test -p lattice-inference --features f16 --lib \
//     forward::cpu_f16::tests::print_f16_generate_goldens -- --ignored --nocapture
// and copy the printed `token_ids`/`stop_reason`/`stopped`/`text` lines into
// the four `#[test]` functions below. Re-run whenever the fixture, the
// shared sampler, or `forward_step_f16` changes on purpose.
fn make_nonzero_f16_cpu_test_model() -> (Qwen35Config, F16ModelWeights, RopeTable) {
    use crate::model::qwen35_config::LayerType;
    use crate::weights::f16_weights::f32_to_f16_slice;

    let hidden: usize = 64;
    let vocab: usize = 128;
    let inter: usize = 128;
    let num_attn_heads: usize = 2;
    let num_kv_heads: usize = 1;
    let head_dim: usize = 32;
    let q_dim = num_attn_heads * head_dim; // 64
    let kv_dim = num_kv_heads * head_dim; // 32
    let lin_key_heads: usize = 2;
    let lin_val_heads: usize = 2;
    let lin_key_dim: usize = 32;
    let lin_val_dim: usize = 32;
    let lin_qkv_dim = lin_key_heads * lin_key_dim * 2 + lin_val_heads * lin_val_dim; // 192
    let lin_output_dim = lin_val_heads * lin_val_dim; // 64
    let kernel_size: usize = 4;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 2,
        vocab_size: vocab,
        intermediate_size: inter,
        rms_norm_eps: 1e-6,
        num_attention_heads: num_attn_heads,
        num_key_value_heads: num_kv_heads,
        head_dim,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: lin_key_heads,
        linear_num_value_heads: Some(lin_val_heads),
        linear_key_head_dim: lin_key_dim,
        linear_value_head_dim: lin_val_dim,
        linear_conv_kernel_dim: kernel_size,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        layer_mask: vec![true; 2],
        // Out of the fixture's vocabulary range: EOS never fires "by luck" on
        // this LCG-generated model, so every golden case's stop behaviour
        // (or lack of it) is controlled entirely by `GenerateConfig`.
        eos_token_id: 999_999,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let rope_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize; // 16
    let rope = RopeTable::new(rope_dim, cfg.max_position_embeddings, cfg.rope_theta);

    // Deterministic weight generator: LCG producing small non-zero floats,
    // packed to f16. Same constants as the Q8/NEON nonzero fixtures so all
    // three wrappers' goldens are built by the same recipe.
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    // Shared LCG step, reused by every draw below (weights, norm gains,
    // conv taps, decay-gate biases) so the whole fixture stays one
    // deterministic stream from a single seed.
    fn lcg_step(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 33) as f32 / u32::MAX as f32
    }
    fn next_small_vec(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| lcg_step(seed) * scale - scale / 2.0)
            .collect()
    }
    // Norm gains, decay-gate params and conv taps: distinct, non-uniform
    // per-element draws (never a repeated literal) so these vectors carry
    // real per-channel structure instead of acting as either a no-op
    // (`1 + gamma` with every gamma identical) or a dead branch (`gamma`
    // uniformly `0.0` feeding a direct, non-shifted `x * gamma` norm).
    // Drawn BEFORE `next_weight` is even defined: that closure holds a
    // unique borrow of `seed` for its whole life once created, so any
    // `next_small_vec(&mut seed, ..)` call sitting inside that span
    // (even textually before the closure's first call) conflicts.
    let gdn_a_log = next_small_vec(&mut seed, lin_key_heads, 1.0);
    let gdn_dt_bias = next_small_vec(&mut seed, lin_key_heads, 1.0);
    let gdn_conv1d_weight = next_small_vec(&mut seed, lin_qkv_dim * kernel_size, 0.04);
    // The gated-attention output norm (`scalar_gated_rms_norm`,
    // attention/gdn_fused.rs) applies its weight directly
    // (`x * inv_rms * gamma`), unlike the shifted `1 + gamma`
    // convention `qwen35_rms_norm` uses everywhere else in this
    // fixture. A uniform `0.0` here zeroed this layer's entire output
    // for every input, which is what made the golden streams below
    // identical across every prompt regardless of length or content.
    // A non-uniform, non-zero draw keeps the layer's output both
    // nonzero and content-shaped (a single shared constant would
    // still be a pure, direction-preserving rescale).
    let gdn_norm_weight: Vec<f32> = next_small_vec(&mut seed, lin_val_dim, 1.0)
        .iter()
        .map(|g| g + 1.0)
        .collect();
    // q_norm/k_norm gains at scale 8.0 (Qwen3.5's shifted `1 + gamma`
    // convention, so this puts the per-channel Q/K rescale in roughly
    // [-4, 4] instead of [0.5, 1.5]), tried together with the v_proj/
    // o_proj scale below to make the softmax over `full_attention_step_f16`'s
    // attended keys sensitive to the RoPE phase shift a decode position
    // change produces. Measured (round 3, all-full-attention, this
    // fixture's own history in `crates/inference/src/forward/cpu_q8.rs`):
    // raw-logit deltas from a one-position shift stay ~4 orders of
    // magnitude below this fixture's candidate-logit spacing at this
    // scale, so the greedy argmax never moves. `full_attention_step_f16`
    // reads its attend-window size from `kv_cache.seq_len`, never from
    // the `position` argument the caller passes in, so `position` only
    // ever perturbs RoPE phase here; the fixture's own signal path
    // (GDN's linear recurrence, not full attention) dominates the
    // residual at this problem size regardless of this gain.
    let full_q_norm = next_small_vec(&mut seed, head_dim, 8.0);
    let full_k_norm = next_small_vec(&mut seed, head_dim, 8.0);

    let mut next_weight = |n: usize, k: usize, scale: f32| -> Vec<u16> {
        let floats: Vec<f32> = (0..n * k)
            .map(|_| lcg_step(&mut seed) * scale - scale / 2.0)
            .collect();
        let mut packed = vec![0u16; n * k];
        f32_to_f16_slice(&floats, &mut packed);
        packed
    };

    let gdn_w = F16GatedDeltaNetWeights {
        in_proj_qkv: next_weight(lin_qkv_dim, hidden, 0.04),
        in_proj_qkv_rows: lin_qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: next_weight(lin_output_dim, hidden, 0.04),
        in_proj_z_rows: lin_output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: next_weight(lin_key_heads, hidden, 0.04),
        in_proj_b_rows: lin_key_heads,
        in_proj_b_cols: hidden,
        in_proj_a: next_weight(lin_key_heads, hidden, 0.04),
        in_proj_a_rows: lin_key_heads,
        in_proj_a_cols: hidden,
        a_log: gdn_a_log,
        dt_bias: gdn_dt_bias,
        conv1d_weight: gdn_conv1d_weight,
        conv_dim: lin_qkv_dim,
        kernel_size,
        norm_weight: gdn_norm_weight,
        out_proj: next_weight(hidden, lin_output_dim, 0.04),
        out_proj_rows: hidden,
        out_proj_cols: lin_output_dim,
    };

    // v_proj/o_proj at a distinctly larger scale (4.0 vs the 0.04 every
    // other matrix uses), tried alongside the q_norm/k_norm gain above
    // so the full-attention layer's residual contribution would compete
    // with the FFN's attenuation. Measured insufficient at this scale
    // (see the comment on `full_q_norm`/`full_k_norm` above) — kept at
    // this larger value because it strictly widens the draw range used
    // for the content-sensitivity property this fixture guards, without
    // regressing it, not because it reaches position sensitivity.
    let full_w = F16FullAttentionLayerWeights {
        q_proj: next_weight(2 * q_dim, hidden, 0.04),
        k_proj: next_weight(kv_dim, hidden, 0.04),
        v_proj: next_weight(kv_dim, hidden, 4.0),
        o_proj: next_weight(hidden, q_dim, 4.0),
        q_norm: full_q_norm,
        k_norm: full_k_norm,
    };

    let make_common = |seed: &mut u64| -> F16CommonLayerWeights {
        let input_layernorm = next_small_vec(seed, hidden, 1.0);
        let post_attention_layernorm = next_small_vec(seed, hidden, 1.0);
        let mut nw = |n: usize, k: usize| -> Vec<u16> {
            let floats: Vec<f32> = (0..n * k).map(|_| lcg_step(seed) * 0.04 - 0.02).collect();
            let mut packed = vec![0u16; n * k];
            f32_to_f16_slice(&floats, &mut packed);
            packed
        };
        F16CommonLayerWeights {
            input_layernorm,
            post_attention_layernorm,
            ffn: F16FeedForwardWeights::Dense {
                gate_proj: nw(inter, hidden),
                up_proj: nw(inter, hidden),
                down_proj: nw(hidden, inter),
            },
        }
    };

    let layers = vec![
        (F16AttentionWeights::Linear(gdn_w), make_common(&mut seed)),
        (F16AttentionWeights::Full(full_w), make_common(&mut seed)),
    ];

    let embed_floats: Vec<f32> = (0..vocab * hidden)
        .map(|_| lcg_step(&mut seed) * 0.04 - 0.02)
        .collect();
    let mut embed_tokens = vec![0u16; vocab * hidden];
    f32_to_f16_slice(&embed_floats, &mut embed_tokens);

    let weights = F16ModelWeights {
        embed_tokens,
        final_norm: next_small_vec(&mut seed, hidden, 1.0),
        layers,
    };

    (cfg, weights, rope)
}

/// Tokenizer covering every id in the nonzero fixture's `vocab_size` (128).
/// Ids 0..=7 keep the single-character tokens the other tests in this file
/// already rely on; the golden prompt `"world"` encodes to five of them. Ids 8..128 get
/// distinct two-character printable tokens so every id the decode loop can
/// produce has real text instead of falling through to an empty string.
fn nonzero_fixture_tokenizer() -> BpeTokenizer {
    use std::collections::HashMap;
    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for id in 8u32..128 {
        let j = (id - 8) as usize;
        let a = ALPHABET[j / ALPHABET.len()] as char;
        let b = ALPHABET[j % ALPHABET.len()] as char;
        vocab_map.insert(format!("{a}{b}"), id);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    BpeTokenizer::from_vocab_and_merges(vocab_map, merges).unwrap()
}

/// Writer path for the four `generate_f16` goldens below: prints the exact
/// fields the `#[test]`s pin. Not run by default (`#[ignore]`); run
/// explicitly (see the module-level comment above) and copy its output into
/// the assertions when the fixture or the production decode path changes on
/// purpose.
#[test]
#[ignore = "golden writer: run explicitly to (re)print the pinned values"]
fn print_f16_generate_goldens() {
    let (cfg, weights, rope) = make_nonzero_f16_cpu_test_model();
    let tokenizer = nonzero_fixture_tokenizer();

    let greedy_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let greedy = generate_f16(&weights, &cfg, &tokenizer, &rope, "world", &greedy_cfg)
        .expect("greedy golden must succeed");
    println!("GREEDY token_ids={:?}", greedy.token_ids);
    println!(
        "GREEDY stopped={} stop_reason={:?}",
        greedy.stopped, greedy.stop_reason
    );
    println!("GREEDY text={:?}", greedy.text);

    let seeded_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.8,
        top_k: 5,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: Some(1234),
        ..Default::default()
    };
    let seeded = generate_f16(&weights, &cfg, &tokenizer, &rope, "world", &seeded_cfg)
        .expect("seeded golden must succeed");
    println!("SEEDED token_ids={:?}", seeded.token_ids);
    println!(
        "SEEDED stopped={} stop_reason={:?}",
        seeded.stopped, seeded.stop_reason
    );
    println!("SEEDED text={:?}", seeded.text);

    // Stop-token case: derive a real mid-stream token from the greedy run
    // above and make it a stop token, so the golden proves an ACTUAL
    // decode-loop stop (not a fixture that vacuously never generates).
    assert!(
        greedy.token_ids.len() >= 2,
        "need at least 2 greedy tokens to derive a mid-stream stop token"
    );
    let stop_tok = greedy.token_ids[1];
    let stop_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        stop_token_ids: vec![stop_tok],
        ..Default::default()
    };
    let stopped = generate_f16(&weights, &cfg, &tokenizer, &rope, "world", &stop_cfg)
        .expect("stop-token golden must succeed");
    println!("STOP stop_tok={stop_tok}");
    println!("STOP token_ids={:?}", stopped.token_ids);
    println!(
        "STOP stopped={} stop_reason={:?}",
        stopped.stopped, stopped.stop_reason
    );
    println!("STOP text={:?}", stopped.text);

    let one_tok_cfg = GenerateConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let one_tok = generate_f16(&weights, &cfg, &tokenizer, &rope, "world", &one_tok_cfg)
        .expect("max_new_tokens=1 golden must succeed");
    println!("ONE token_ids={:?}", one_tok.token_ids);
    println!(
        "ONE stopped={} stop_reason={:?}",
        one_tok.stopped, one_tok.stop_reason
    );
    println!("ONE text={:?}", one_tok.text);
}

/// `generate_f16` greedy decode (`temperature: 0.0`). Values captured from
/// `print_f16_generate_goldens` (see that test's doc comment for the exact
/// regeneration command); identical to `generate_q8`'s and
/// `generate_q8_neon`'s greedy goldens for this fixture (small,
/// well-separated logits at this problem size are not moved by f16/int8
/// quantization noise here).
///
/// Mutation check: changing `max_new_tokens: 6` to `5` must turn
/// `token_ids` red (truncated sequence). `seed` is not the mutated field
/// here: at `temperature: 0.0` the sampler takes the degenerate-temperature
/// argmax short-circuit before the RNG is ever consulted, so a seed change
/// on a greedy golden cannot move the output.
#[test]
fn generate_f16_greedy_golden() {
    let (cfg, weights, rope) = make_nonzero_f16_cpu_test_model();
    let tokenizer = nonzero_fixture_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "world", &gen_cfg)
        .expect("greedy golden must succeed");
    assert_eq!(
        out.prompt_tokens, 5,
        "the prompt must span several prefill positions"
    );
    assert_eq!(out.token_ids, vec![36, 80, 112, 62, 103, 86]);
    assert!(!out.stopped);
    assert_eq!(
        out.stop_reason,
        Some(crate::stop_reason::StopReason::Length)
    );
    assert_eq!(out.text, "aCbkbQa2bHbq");
}

/// `generate_f16` with temperature/top_k/top_p/repetition_penalty all set
/// (seeded sampling, not greedy).
///
/// Mutation check: changing `top_k: 5` to `top_k: 1` (equivalent to greedy
/// given the other settings) must turn `token_ids` red against this golden.
#[test]
fn generate_f16_seeded_sampling_golden() {
    let (cfg, weights, rope) = make_nonzero_f16_cpu_test_model();
    let tokenizer = nonzero_fixture_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.8,
        top_k: 5,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: Some(1234),
        ..Default::default()
    };
    let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "world", &gen_cfg)
        .expect("seeded golden must succeed");
    assert_eq!(out.token_ids, vec![36, 80, 103, 112, 87, 86]);
    assert!(!out.stopped);
    assert_eq!(
        out.stop_reason,
        Some(crate::stop_reason::StopReason::Length)
    );
    assert_eq!(out.text, "aCbkbHbQbrbq");
}

/// `generate_f16` stopping on a `stop_token_ids` match. `stop_tok = 80` is
/// the greedy golden's own second token (see `generate_f16_greedy_golden`),
/// so this proves an actual decode-loop stop, not a fixture that vacuously
/// never generates that token.
///
/// Mutation check: removing `stop_token_ids` (or pointing it at a token the
/// greedy prefix never produces) must turn `stopped`/`stop_reason`/
/// `token_ids.len()` red against this golden.
#[test]
fn generate_f16_stop_token_golden() {
    let (cfg, weights, rope) = make_nonzero_f16_cpu_test_model();
    let tokenizer = nonzero_fixture_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        stop_token_ids: vec![80],
        ..Default::default()
    };
    let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "world", &gen_cfg)
        .expect("stop-token golden must succeed");
    assert_eq!(out.token_ids, vec![36]);
    assert!(out.stopped);
    assert_eq!(out.stop_reason, Some(crate::stop_reason::StopReason::Eos));
    assert_eq!(out.text, "aC");
}

/// `generate_f16` with `max_new_tokens: 1`, the boundary case where the
/// decode loop must stop after exactly one token.
///
/// Mutation check: changing `max_new_tokens` to `2` must turn
/// `token_ids.len()` red against this golden (it would produce 2 tokens,
/// matching the greedy golden's first two: `[36, 80]`).
#[test]
fn generate_f16_max_new_tokens_one_golden() {
    let (cfg, weights, rope) = make_nonzero_f16_cpu_test_model();
    let tokenizer = nonzero_fixture_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "world", &gen_cfg)
        .expect("max_new_tokens=1 golden must succeed");
    assert_eq!(out.token_ids, vec![36]);
    assert!(!out.stopped);
    assert_eq!(
        out.stop_reason,
        Some(crate::stop_reason::StopReason::Length)
    );
    assert_eq!(out.text, "aC");
}

/// The fixture the goldens above use, bundled for the shared standalone-session checks.
struct SessionFixture {
    cfg: Qwen35Config,
    weights: F16ModelWeights,
    rope: RopeTable,
    tokenizer: BpeTokenizer,
}

impl SessionFixture {
    fn new() -> Self {
        let (cfg, weights, rope) = make_nonzero_f16_cpu_test_model();
        Self {
            cfg,
            weights,
            rope,
            tokenizer: nonzero_fixture_tokenizer(),
        }
    }

    fn standalone(&self) -> StandaloneWeights<'_> {
        StandaloneWeights::F16(&self.weights)
    }

    fn public(
        &self,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, crate::error::InferenceError> {
        generate_f16(
            &self.weights,
            &self.cfg,
            &self.tokenizer,
            &self.rope,
            "world",
            gen_cfg,
        )
    }

    fn forward(
        &self,
        token_id: u32,
        position: usize,
        gdn_states: &mut [GatedDeltaNetState],
        kv_cache: &mut KvCache,
        scratch: &mut ForwardScratch,
    ) -> Result<(), crate::error::InferenceError> {
        forward_step_f16(
            &self.weights,
            &self.cfg,
            &self.rope,
            token_id,
            position,
            gdn_states,
            kv_cache,
            scratch,
            None,
            None,
        )
    }
}

#[test]
fn standalone_session_matches_recorded_outputs_deterministic_cases() {
    let f = SessionFixture::new();
    for (name, gen_cfg, recorded) in parity::deterministic_cases() {
        parity::assert_matches_recorded(
            &|c| f.public(c),
            f.standalone(),
            &f.cfg,
            &f.tokenizer,
            &f.rope,
            "world",
            name,
            &gen_cfg,
            &recorded,
        );
    }
}

#[test]
fn standalone_session_matches_recorded_outputs_seeded() {
    let f = SessionFixture::new();
    parity::assert_matches_recorded(
        &|c| f.public(c),
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        "seeded",
        &parity::seeded_case(),
        &parity::seeded_recorded(),
    );
}

#[test]
fn standalone_session_logits_match_reference_replay() {
    let f = SessionFixture::new();
    parity::assert_logits_replay(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
        4,
        &|t, p, g, k, s| f.forward(t, p, g, k, s),
    );
}

#[test]
fn standalone_session_rejects_a_stale_prediction() {
    let f = SessionFixture::new();
    parity::assert_rejects_stale_prediction(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn standalone_session_cancellation_precedes_state_change() {
    let f = SessionFixture::new();
    parity::assert_cancellation_precedes_state_change(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn standalone_session_refuses_undeclared_controls() {
    let f = SessionFixture::new();
    parity::assert_refuses_undeclared_controls(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn standalone_session_select_refuses_config_controls() {
    let f = SessionFixture::new();
    parity::assert_select_refuses_config_controls(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::seeded_case(),
    );
}

#[test]
fn standalone_session_finish_invalidates_the_live_prediction() {
    let f = SessionFixture::new();
    parity::assert_finish_invalidates_the_live_prediction(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn standalone_session_refuses_a_second_prefill() {
    let f = SessionFixture::new();
    parity::assert_second_prefill_is_refused(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

/// `generate_f16` must reject a request whose prompt + max_new_tokens exceeds
/// the RoPE table capacity with a clean error, not an out-of-bounds RoPE index
/// (in a real model) or a runaway allocation. The preflight returns before any
/// forward pass, so the zero-layer fixture is sufficient. Mutation check:
/// removing the preflight lets the zero-layer model run the decode to
/// completion and return Ok (it has no RoPE-indexing attention layer), which
/// trips the `expect_err` below — changing the test from PASS to FAIL. Mirrors
/// `generate_q8`'s `test_generate_q8_rejects_context_overflow`.
#[test]
fn test_generate_f16_rejects_context_overflow() {
    let (cfg, weights, rope, tokenizer) = zero_layer_f16_fixture();
    let max_context = rope.max_positions(); // 64 from the fixture
    // "hello" is >= 1 token, so prompt_len + max_context > max_context.
    let gen_cfg = GenerateConfig {
        max_new_tokens: max_context,
        ..Default::default()
    };
    let err = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg)
        .expect_err("request beyond context window must error, not panic");
    let msg = format!("{err}");
    assert!(
        msg.contains("context window"),
        "error must name the context window; got: {msg}"
    );
}

/// `generate_f16` must stop on a token in `stop_token_ids` even when that
/// token differs from `eos_token_id`.
///
/// Setup: all-zero f16 weights → greedy sampling always picks token 0.
/// Config has eos_token_id=5 (not 0) and stop_token_ids=[0].
/// With the fix the first sampled token (0) hits the stop list and the
/// function returns 0 generated tokens.
///
/// Mutation check: reverting `should_stop_token` back to
/// `next_id == cfg.eos_token_id` in either check causes `0 == 5` to be false,
/// so token 0 is pushed to output and `generated_tokens` becomes ≥ 1.
#[test]
fn test_generate_f16_honors_stop_token_ids() {
    let (cfg, weights, rope, tokenizer) = zero_layer_f16_fixture();

    let gen_cfg = GenerateConfig {
        max_new_tokens: 4,
        stop_token_ids: vec![0], // token 0 is the stop signal, NOT eos (5)
        temperature: 0.0,        // greedy: all-zero logits always yield token 0
        ..Default::default()
    };

    let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "h", &gen_cfg)
        .expect("generate_f16 must succeed with valid stop_token_ids");

    assert_eq!(
        out.generated_tokens, 0,
        "generate_f16 must stop immediately when the first greedy token (0) \
             is in stop_token_ids — got {} generated tokens instead",
        out.generated_tokens
    );
}

/// `generate_f16` must also stop when the stop token first appears in the
/// **decode loop**, not only at the post-prefill check.
///
/// Fixture: a "bouncing" 0-layer f16 model.
///   embed[0] = [-1, 1, 0, 0]  (token 0, f16)
///   embed[1] = [ 1, 1, 0, 0]  (token 1, f16)
///   final_norm gamma = [-2, 0, 0, 0]
///
/// The negative gamma at dim-0 flips the sign of that component after RMSNorm,
/// creating a "bounce" between tokens 0 and 1:
///   from token 1: hidden = [-√2, +√2, 0, 0] → logit[0] = 2√2 wins → generates 0
///   from token 0: hidden = [+√2, +√2, 0, 0] → logit[1] = 2√2 wins → generates 1
///
/// Greedy sequence from prompt "e" (→ token 1, eos_token_id=5):
///   post-prefill  → token 0  (not stop=1)
///   decode step 1 → token 1  (stop) → decode-loop fires
///
/// Mutation proof: reverting ONLY the decode-loop `should_stop_token` check
/// (line 946 at time of writing) to `next_id == cfg.eos_token_id` leaves
/// token 1 uncaught (1 ≠ eos=5), the sequence continues, and generated_tokens
/// becomes ≥ 2 — failing the assertion below.
#[test]
fn test_generate_f16_honors_stop_token_ids_decode_loop() {
    use crate::weights::f16_weights::f32_to_f16_slice;
    use std::collections::HashMap;

    let hidden = 4usize;
    let vocab = 8usize;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 0,
        vocab_size: vocab,
        intermediate_size: 4,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: 1,
        linear_num_value_heads: Some(1),
        linear_key_head_dim: 4,
        linear_value_head_dim: 4,
        linear_conv_kernel_dim: 4,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![],
        layer_mask: vec![],
        // eos=5 so the stop at token 1 is detectable only via stop_token_ids.
        eos_token_id: 5,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    // The negative gamma at dim-0 flips the sign of that component after
    // RMSNorm, creating a deterministic "bounce" between tokens 0 and 1.
    // from embed[1]=[1,1,0,0]: hidden→[-√2,+√2,0,0] → dot(embed[0]=[-1,1,..]) = 2√2 > 0
    // from embed[0]=[-1,1,0,0]: hidden→[+√2,+√2,0,0] → dot(embed[1]=[1,1,..]) = 2√2 > 0
    let embed_f32: Vec<f32> = {
        let mut v = vec![0.0f32; vocab * hidden];
        v[0] = -1.0; // token 0, dim 0
        v[1] = 1.0; // token 0, dim 1
        v[hidden] = 1.0; // token 1, dim 0
        v[hidden + 1] = 1.0; // token 1, dim 1
        v
    };
    let mut embed_f16 = vec![0u16; vocab * hidden];
    f32_to_f16_slice(&embed_f32, &mut embed_f16);

    let weights = F16ModelWeights {
        embed_tokens: embed_f16,
        final_norm: vec![-2.0f32, 0.0, 0.0, 0.0],
        layers: vec![],
    };

    let rope = RopeTable::new(2, 64, 10_000.0);

    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_map, merges).unwrap();

    let gen_cfg = GenerateConfig {
        max_new_tokens: 10,
        stop_token_ids: vec![1], // stop on token 1 mid-decode-loop; eos_token_id=5≠1
        temperature: 0.0,        // greedy: deterministic bouncing sequence
        ..Default::default()
    };

    // Prompt "e" → token 1.
    // Post-prefill generates token 0 (not stop=1).
    // Decode step 1 generates token 1 → decode-loop stop fires.
    let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "e", &gen_cfg)
        .expect("generate_f16 must succeed");

    assert_eq!(
        out.generated_tokens, 1,
        "generate_f16 must stop at decode-loop step 1 when token 1 is in \
             stop_token_ids — got {} tokens; reverting only the decode-loop check \
             lets token 1 through and produces ≥ 2 tokens",
        out.generated_tokens
    );
    assert!(
        out.stopped,
        "generate_f16 must set stopped=true when the decode-loop stop fires"
    );
}

/// `generate_f16` must reject an empty prompt with a typed
/// `Err(Inference("empty prompt"))` before any weight dereference or
/// state allocation (#856): this is one of the three CPU forward paths
/// the shared `check_prompt_not_empty` preflight unifies with the four
/// Metal paths, which used to silently accept an empty prompt and
/// return an empty `Ok`. See docs/generation-entrypoint-matrix.md row 2.
///
/// The guard fires before any weight dereference, so empty weight vecs
/// are sufficient (mirrors `generate_f16_rejects_grammar_config_before_sampling`
/// below).
///
/// Mutation sensitivity: bypassing the shared preparation at this entry
/// point makes the function proceed past the guard with a
/// zero-length prompt, either panicking in the prefill/decode loop
/// (`all_ids.last()` on an empty vec) or producing a non-`Inference`
/// error — this assert fails either way.
#[test]
fn generate_f16_rejects_empty_prompt() {
    use crate::error::InferenceError;
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let weights = F16ModelWeights {
        embed_tokens: vec![],
        final_norm: vec![],
        layers: vec![],
    };
    let gen_cfg = GenerateConfig::default();

    let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
        "generate_f16 must reject an empty prompt with Err(Inference(\"empty \
             prompt\")) (#856); got {result:?}"
    );
}

/// `generate_f16` must reject a `GenerateConfig` that sets `grammar` with a
/// typed `InvalidInput` error before sampling any token (#397/#398).
///
/// Before the fix, grammar was silently ignored and unconstrained output was
/// produced. The guard fires before any weight dereference or state allocation,
/// so empty weight vecs are sufficient.
///
/// Mutation sensitivity: removing the `check_grammar_not_set` call makes the
/// function proceed past the guard and attempt to forward with empty weights,
/// producing a panic or a non-`InvalidInput` error — this assert fails either way.
#[test]
fn generate_f16_rejects_grammar_config_before_sampling() {
    use crate::error::InferenceError;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let weights = F16ModelWeights {
        embed_tokens: vec![],
        final_norm: vec![],
        layers: vec![],
    };

    let spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
    let grammar_vocab = vec![b"t".to_vec(), b"f".to_vec()];
    let engine = GrammarEngine::new(&spec, grammar_vocab).expect("trivial grammar must compile");

    let gen_cfg = GenerateConfig {
        grammar: Some(Arc::new(engine)),
        ..Default::default()
    };

    let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "generate_f16 must fail closed with InvalidInput when grammar is set (#397/#398); \
             got {result:?}"
    );
}

/// `generate_f16` must reject a `GenerateConfig` that sets `logprobs` with a
/// typed `InvalidInput` error before sampling any token (#585): this wrapper
/// has not wired per-step log-probability capture into its decode loop, so a
/// request asking for it must fail closed rather than silently return an
/// empty `token_logprobs`.
///
/// Sibling of `generate_f16_rejects_grammar_config_before_sampling` /
/// `..._stop_strings_..` / `..._reasoning_budget_..`, which already cover the
/// other three `StandaloneCpu`-contract guards; this one was the gap.
/// Mutation sensitivity: removing the `check_logprobs_not_set`
/// call in `GenerationEntryContract::StandaloneCpu`'s `validate_capabilities`
/// arm makes the function proceed past the guard and attempt to forward with
/// empty weights, producing a panic or a non-`InvalidInput` error — this
/// assert fails either way.
#[test]
fn generate_f16_rejects_logprobs_config_before_sampling() {
    use crate::error::InferenceError;
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let weights = F16ModelWeights {
        embed_tokens: vec![],
        final_norm: vec![],
        layers: vec![],
    };

    let gen_cfg = GenerateConfig {
        logprobs: Some(0),
        ..Default::default()
    };

    let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("per-token logprobs are not yet supported")
        ),
        "generate_f16 must fail closed with the exact logprobs-unsupported \
             InvalidInput (#585); got {result:?}"
    );
}

/// `generate_f16` must reject a `GenerateConfig` that sets `stop_strings` with a
/// typed `InvalidInput` error before sampling any token (ADR-080 C3, #783).
///
/// Mutation sensitivity: removing the `check_stop_strings_not_set` call makes the
/// function proceed past the guard and attempt to forward with empty weights,
/// producing a panic or a non-`InvalidInput` error — this assert fails either way.
#[test]
fn generate_f16_rejects_stop_strings_config_before_sampling() {
    use crate::error::InferenceError;
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let weights = F16ModelWeights {
        embed_tokens: vec![],
        final_norm: vec![],
        layers: vec![],
    };

    let gen_cfg = GenerateConfig {
        stop_strings: vec!["</s>".to_string()],
        ..Default::default()
    };

    let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "generate_f16 must fail closed with InvalidInput when stop_strings is set \
             (ADR-080 C3, #783); got {result:?}"
    );
}

/// `generate_f16` must reject a `GenerateConfig` that sets `reasoning_budget` with
/// a typed `InvalidInput` error before sampling any token (ADR-080 C3, #783).
///
/// Mutation sensitivity: removing the `check_reasoning_budget_not_set` call makes
/// the function proceed past the guard and attempt to forward with empty weights,
/// producing a panic or a non-`InvalidInput` error — this assert fails either way.
#[test]
fn generate_f16_rejects_reasoning_budget_config_before_sampling() {
    use crate::error::InferenceError;
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let weights = F16ModelWeights {
        embed_tokens: vec![],
        final_norm: vec![],
        layers: vec![],
    };

    let gen_cfg = GenerateConfig {
        reasoning_budget: Some(16),
        ..Default::default()
    };

    let result = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "generate_f16 must fail closed with InvalidInput when reasoning_budget is set \
             (ADR-080 C3, #783); got {result:?}"
    );
}

/// `generate_f16` with `max_new_tokens == 0` must return zero generated tokens
/// without running prefill or sampling anything (#612, 3rd recurrence of the
/// #226/#456 bug class).
///
/// The guard fires before any weight dereference or state allocation, so
/// empty weight vecs are sufficient — mirrors the grammar-guard test above.
///
/// Mutation sensitivity: removing the `max_new_tokens == 0` early return
/// causes the function to run prefill (against empty weight vecs, which
/// would panic) and sample one token, so `generated_tokens` becomes 1
/// instead of 0 and the assertion below fails.
#[test]
fn generate_f16_max_new_tokens_zero_returns_empty() {
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let weights = F16ModelWeights {
        embed_tokens: vec![],
        final_norm: vec![],
        layers: vec![],
    };

    let gen_cfg = GenerateConfig {
        max_new_tokens: 0,
        ..Default::default()
    };

    let out = generate_f16(&weights, &cfg, &tokenizer, &rope, "hello", &gen_cfg)
        .expect("max_new_tokens=0 must succeed, not error");

    assert_eq!(
        out.generated_tokens, 0,
        "max_new_tokens=0 must produce zero generated tokens"
    );
    assert!(
        out.token_ids.is_empty(),
        "max_new_tokens=0 must produce an empty token list"
    );
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Length),
        "max_new_tokens=0 must report stop_reason=Length"
    );
}

// -----------------------------------------------------------------
// ADR-069 Stage 5b: visual injection + M-RoPE splice (pure CPU, no
// checkpoint required).
// -----------------------------------------------------------------

use crate::model::qwen35_config::{LayerType, RopeParams};
use crate::weights::f16_weights::{
    F16CommonLayerWeights, F16FullAttentionLayerWeights, f32_to_f16_slice,
};

/// A minimal one-layer, full-attention-only (no GDN) config + f16 weight
/// set: small enough to hand-construct, but with non-trivial (identity)
/// Q/K/V/O projections so RoPE and embedding-source mutations actually
/// move the logits, unlike an all-zero model.
fn tiny_vision_splice_model() -> (Qwen35Config, F16ModelWeights) {
    let hidden = 8usize;
    let vocab = 4usize;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 1,
        vocab_size: vocab,
        intermediate_size: 4,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: hidden,
        rope_theta: 1.0e7,
        partial_rotary_factor: 1.0, // rope_dim=8, half=4 (matches production theta scale)
        rope_parameters: Some(RopeParams {
            rope_theta: 1.0e7,
            partial_rotary_factor: Some(1.0),
            mrope_section: Some(vec![2, 1, 1]),
            mrope_interleaved: Some(true),
        }),
        linear_num_key_heads: 2,
        linear_num_value_heads: Some(2),
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 1,
        layer_types: vec![LayerType::FullAttention],
        layer_mask: vec![true],
        eos_token_id: 999,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: Some(3),
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let to_f16 = |src: &[f32]| -> Vec<u16> {
        let mut dst = vec![0u16; src.len()];
        f32_to_f16_slice(src, &mut dst);
        dst
    };
    let identity = |rows: usize, cols: usize| -> Vec<f32> {
        let mut m = vec![0.0f32; rows * cols];
        for i in 0..rows.min(cols) {
            m[i * cols + i] = 1.0;
        }
        m
    };

    let embed_tokens_f32: Vec<f32> = (0..vocab * hidden)
        .map(|k| ((k % 11) as f32) * 0.05 - 0.2)
        .collect();

    let q_dim = hidden;
    let mut q_proj_f32 = vec![0.0f32; 2 * q_dim * hidden];
    q_proj_f32[..q_dim * hidden].copy_from_slice(&identity(q_dim, hidden));

    let full_weights = F16FullAttentionLayerWeights {
        q_proj: to_f16(&q_proj_f32),
        k_proj: to_f16(&identity(hidden, hidden)),
        v_proj: to_f16(&identity(hidden, hidden)),
        o_proj: to_f16(&identity(hidden, q_dim)),
        q_norm: vec![0.0f32; hidden],
        k_norm: vec![0.0f32; hidden],
    };

    let common = F16CommonLayerWeights {
        input_layernorm: vec![0.0f32; hidden],
        post_attention_layernorm: vec![0.0f32; hidden],
        ffn: F16FeedForwardWeights::Dense {
            gate_proj: to_f16(&vec![0.0f32; 4 * hidden]),
            up_proj: to_f16(&vec![0.0f32; 4 * hidden]),
            down_proj: to_f16(&vec![0.0f32; hidden * 4]),
        },
    };

    let weights = F16ModelWeights {
        embed_tokens: to_f16(&embed_tokens_f32),
        final_norm: vec![0.0f32; hidden],
        layers: vec![(F16AttentionWeights::Full(full_weights), common)],
    };

    (cfg, weights)
}

/// Injection reaches the decoder: a supplied embedding replaces the
/// token-id lookup (not adds to it), mutating the supplied row changes
/// the resulting logits, and mutating an unrelated vocab row leaves the
/// injected-slot output unchanged (proves the None-path embedding table
/// isn't consulted when `injected_embedding` is `Some`).
#[test]
fn injection_replaces_lookup_and_is_mutation_sensitive() {
    let (cfg, mut weights) = tiny_vision_splice_model();
    let hidden = cfg.hidden_size;

    // Compares the pre-lm_head hidden state (not `scratch.logits`): `embed_tokens` is
    // tied to the output projection too, so a logits comparison would show every vocab
    // row mutation regardless of whether the *input* lookup was ever consulted. The
    // decoder's final hidden state isolates exactly the quantity injection controls.
    let run = |weights: &F16ModelWeights, injected: Option<&[f32]>| -> Vec<f32> {
        let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
        let mut gdn_states: Vec<GatedDeltaNetState> = vec![];
        let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
        let mut scratch = ForwardScratch::new();
        forward_step_f16(
            weights,
            &cfg,
            &rope,
            0,
            0,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            injected,
            None,
        )
        .expect("forward step succeeds");
        scratch.hidden[..hidden].to_vec()
    };

    let baseline = run(&weights, None);

    let mut visual_row = vec![0.9f32, -0.4, 0.2, 0.6, -0.1, 0.3, 0.7, -0.8];
    let injected_hidden = run(&weights, Some(&visual_row));
    assert_ne!(
        injected_hidden, baseline,
        "an injected embedding must produce a different hidden state than the token-id lookup"
    );

    // Mutate the supplied visual scalar: the image-pad-slot output must change.
    visual_row[0] += 1.0;
    let mutated_visual_hidden = run(&weights, Some(&visual_row));
    assert_ne!(
        mutated_visual_hidden, injected_hidden,
        "mutating the supplied visual row must change the injected-slot hidden state"
    );
    visual_row[0] -= 1.0; // restore

    // Mutate a non-pad vocab row in the embedding table: the injected-slot output
    // (still using the ORIGINAL visual_row) must be unchanged.
    let embed_start = hidden; // token id 1's row
    let mutated_row_f32 = vec![1.0f32; hidden];
    f32_to_f16_slice(
        &mutated_row_f32,
        &mut weights.embed_tokens[embed_start..embed_start + hidden],
    );
    let after_table_mutation = run(&weights, Some(&visual_row));
    assert_eq!(
        after_table_mutation, injected_hidden,
        "mutating an unrelated embedding-table row must not affect the injected slot"
    );

    // Sanity: length mismatch and non-finite values must fail closed.
    let mut gdn_states: Vec<GatedDeltaNetState> = vec![];
    let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let short_row = vec![0.0f32; hidden - 1];
    assert!(
        forward_step_f16(
            &weights,
            &cfg,
            &rope,
            0,
            0,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            Some(&short_row),
            None,
        )
        .is_err(),
        "wrong-length injected_embedding must be rejected"
    );
    let mut nan_row = vec![0.0f32; hidden];
    nan_row[2] = f32::NAN;
    assert!(
        forward_step_f16(
            &weights,
            &cfg,
            &rope,
            0,
            0,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            Some(&nan_row),
            None,
        )
        .is_err(),
        "non-finite injected_embedding must be rejected"
    );
}

/// The cos/sin actually applied inside the wired GQA path
/// (`full_attention_step_f16` via `forward_step_f16`) at an image-pad
/// token matches `build_cos_sin`'s output at that same (t,h,w) position —
/// exercised through the wired forward, not just the S5a unit builder.
#[test]
fn mrope_cos_sin_applied_in_wired_forward_matches_builder() {
    use crate::vision::qwen35_mrope::{MRopePositions, build_cos_sin};

    let (cfg, weights) = tiny_vision_splice_model();
    let hidden = cfg.hidden_size;
    let rope_params = cfg.rope_parameters.as_ref().unwrap();
    let mrope_section = rope_params.mrope_section.as_ref().unwrap();

    let positions = MRopePositions {
        positions: vec![(2, 3, 5)],
        rope_delta: 0,
    };
    let tables = build_cos_sin(
        &positions,
        cfg.head_dim,
        rope_params.partial_rotary_factor.unwrap(),
        rope_params.rope_theta as f32,
        mrope_section,
    )
    .expect("builds tables");
    let (cos_row, sin_row) = (tables.cos[0].as_slice(), tables.sin[0].as_slice());

    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let mut gdn_states: Vec<GatedDeltaNetState> = vec![];
    let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();

    forward_step_f16(
        &weights,
        &cfg,
        &rope,
        1,
        0,
        &mut gdn_states,
        &mut kv_cache,
        &mut scratch,
        None,
        Some((cos_row, sin_row)),
    )
    .expect("forward step succeeds");

    // Independently reproduce the K rotation using the SAME cos/sin row (identity
    // k_proj means k_buf before rotation equals the RMSNorm'd embedding row).
    let mut k_ref = vec![0.0f32; hidden];
    f16_to_f32_slice(&weights.embed_tokens[hidden..2 * hidden], &mut k_ref);
    let zero_norm = vec![0.0f32; hidden];
    qwen35_rms_norm(&mut k_ref, &zero_norm, hidden, cfg.rms_norm_eps);
    let half = cfg.rope_dim() / 2;
    for i in 0..half {
        let x0 = k_ref[i];
        let x1 = k_ref[half + i];
        k_ref[i] = x0 * cos_row[i] - x1 * sin_row[i];
        k_ref[half + i] = x0 * sin_row[i] + x1 * cos_row[i];
    }

    let k_cached = &kv_cache.k[0][..hidden];
    let max_diff = k_cached
        .iter()
        .zip(k_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-3,
        "wired M-RoPE rotation diverges from build_cos_sin's own row: max_diff={max_diff}"
    );
}

/// A text-only `Qwen35VisionRequest` (no image runs) driven through
/// `generate_multimodal_f16` must be bit-identical to the same token
/// sequence driven through `forward_step_f16` directly with the plain
/// 1-D `RopeTable` (mirroring `generate_f16`'s own prefill+decode loop) —
/// proving `generate_multimodal_f16`'s text-only path never falls
/// through to the M-RoPE table when the request has no image.
#[test]
fn generate_multimodal_text_only_matches_plain_forward_bit_identical() {
    use crate::vision::multimodal::Qwen35VisionRequest;

    let (cfg, weights) = tiny_vision_splice_model();
    let input_ids: Vec<u32> = vec![0, 1, 2, 0];

    let request = Qwen35VisionRequest {
        input_ids: input_ids.clone(),
        image_grids: vec![],
        post_merger_rows: vec![],
        image_token_id: 3,
        spatial_merge_size: 2,
        decoder_hidden_size: cfg.hidden_size,
    };

    let gen_cfg = GenerateConfig {
        max_new_tokens: 2,
        temperature: 0.0,
        seed: Some(1),
        stop_token_ids: vec![],
        ..Default::default()
    };

    let multimodal_out = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
        .expect("text-only multimodal generate succeeds");

    // Reference: hand-drive forward_step_f16 exactly like generate_f16's own loop,
    // with the same seed and the plain 1-D RopeTable.
    let rope = RopeTable::new(cfg.rope_dim(), 512, cfg.rope_theta);
    let mut gdn_states: Vec<GatedDeltaNetState> = vec![];
    let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    for (pos, &token_id) in input_ids.iter().enumerate() {
        forward_step_f16(
            &weights,
            &cfg,
            &rope,
            token_id,
            pos,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            None,
            None,
        )
        .expect("reference forward step succeeds");
        if pos < input_ids.len() - 1 {
            kv_cache.seq_len += 1;
        }
    }
    kv_cache.seq_len = input_ids.len();

    let mut rng_state = 1u64;
    let mut all_ids = input_ids.clone();
    let mut ref_ids = Vec::new();

    let next_id = sample_token(
        &scratch.logits[..cfg.vocab_size],
        &gen_cfg,
        &all_ids,
        &mut rng_state,
    );
    ref_ids.push(next_id);
    all_ids.push(next_id);

    for _ in 1..gen_cfg.max_new_tokens {
        let pos = kv_cache.seq_len;
        let last_token = *all_ids.last().unwrap();
        forward_step_f16(
            &weights,
            &cfg,
            &rope,
            last_token,
            pos,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
            None,
            None,
        )
        .expect("reference decode step succeeds");
        kv_cache.seq_len += 1;
        let next_id = sample_token(
            &scratch.logits[..cfg.vocab_size],
            &gen_cfg,
            &all_ids,
            &mut rng_state,
        );
        ref_ids.push(next_id);
        all_ids.push(next_id);
    }

    assert_eq!(
        multimodal_out.token_ids, ref_ids,
        "text-only generate_multimodal_f16 token ids must match the plain forward_step_f16 \
             reference bit-for-bit"
    );
}

/// `generate_multimodal_f16` fails closed on an invalid request (mismatched
/// image-pad count vs grid) before any decoder work begins — it delegates to
/// `Qwen35VisionRequest::validate()` rather than re-deriving the checks.
#[test]
fn generate_multimodal_f16_rejects_invalid_request() {
    use crate::vision::multimodal::Qwen35VisionRequest;

    let (cfg, weights) = tiny_vision_splice_model();
    let request = Qwen35VisionRequest {
        input_ids: vec![0, 3, 3, 1], // 2 image-pad tokens
        image_grids: vec![crate::vision::qwen35_vit::GridThw { t: 1, h: 4, w: 4 }], // needs 4
        post_merger_rows: vec![0.0f32; 2 * cfg.hidden_size],
        image_token_id: 3,
        spatial_merge_size: 2,
        decoder_hidden_size: cfg.hidden_size,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    assert!(
        generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg).is_err(),
        "a request whose image-pad count does not match its grid must be rejected"
    );
}

/// `generate_multimodal_f16` fails closed when the prompt plus
/// `max_new_tokens` would exceed the model's context window, mirroring
/// `generate_f16`'s own preflight.
#[test]
fn generate_multimodal_f16_rejects_context_overflow() {
    use crate::vision::multimodal::Qwen35VisionRequest;

    let (mut cfg, weights) = tiny_vision_splice_model();
    cfg.max_position_embeddings = 3;
    let request = Qwen35VisionRequest {
        input_ids: vec![0, 1, 2],
        image_grids: vec![],
        post_merger_rows: vec![],
        image_token_id: 3,
        spatial_merge_size: 2,
        decoder_hidden_size: cfg.hidden_size,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: 5,
        ..Default::default()
    };
    assert!(
        generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg).is_err(),
        "prompt_len + max_new_tokens exceeding max_position_embeddings must be rejected"
    );
}

/// A caller-supplied `input_ids` entry at or past
/// `cfg.vocab_size` must be rejected with `InvalidInput` before any decoder
/// allocation/work, not panic in `forward_step_f16`'s embedding-table slice.
/// Mutation check: removing the guard lets the request reach the None-branch
/// embedding lookup, which indexes `embed_tokens[token_id * hidden..]` past
/// the table's end and panics, turning this `expect_err` into a test-binary
/// abort rather than a clean assertion failure — either way the test fails.
#[test]
fn generate_multimodal_f16_rejects_out_of_vocab_input_id() {
    use crate::vision::multimodal::Qwen35VisionRequest;

    let (cfg, weights) = tiny_vision_splice_model();
    let request = Qwen35VisionRequest {
        input_ids: vec![0, 1, cfg.vocab_size as u32], // last id == vocab_size (OOV)
        image_grids: vec![],
        post_merger_rows: vec![],
        image_token_id: 3,
        spatial_merge_size: 2,
        decoder_hidden_size: cfg.hidden_size,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
        .expect_err("an out-of-vocabulary input_id must be rejected, not panic");
    assert!(
        matches!(err, crate::error::InferenceError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// Sibling path of the input-id guard: `generate_f16` accepts `cfg` and
/// `tokenizer` as independent parameters, so a mismatched pair can still
/// tokenize a prompt into an id at or past `cfg.vocab_size`. Simulates that
/// mismatch directly (a tokenizer vocab entry the fixture's 8-row embedding
/// table cannot cover) rather than relying on tokenizer internals to produce
/// an OOV id by accident.
/// Mutation check: removing the guard lets `forward_step_f16`'s prefill call
/// index `embed_tokens[8 * hidden..]` on a `vocab=8` table — out of bounds —
/// so the panic (or, pre-guard, a wrong-but-silent read) replaces this clean
/// `expect_err`, failing the test either way.
#[test]
fn test_generate_f16_rejects_out_of_vocab_prompt_id() {
    use std::collections::HashMap;

    let (cfg, weights, rope, _tokenizer) = zero_layer_f16_fixture();

    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    // OOV against the fixture's cfg.vocab_size=8 embedding table -- a
    // mismatched cfg/tokenizer pair, exactly the scenario the guard covers.
    vocab_map.insert("z".to_string(), cfg.vocab_size as u32);
    let mismatched_tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_map, vec![])
        .expect("tokenizer with an OOV vocab entry still constructs");

    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    let err = generate_f16(&weights, &cfg, &mismatched_tokenizer, &rope, "z", &gen_cfg)
        .expect_err("an out-of-vocabulary prompt token id must be rejected, not panic");
    assert!(
        matches!(err, crate::error::InferenceError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// Build the [`tiny_vision_splice_model`] fixture plus a populated
/// `vision_config` (spatial_merge_size=2, out_hidden_size == decoder
/// hidden_size) so checkpoint-binding mismatches are
/// testable against a checkpoint that genuinely carries vision metadata.
fn tiny_vision_splice_model_with_vision_cfg() -> (Qwen35Config, F16ModelWeights) {
    use crate::model::qwen35_config::VisionModelConfig;

    let (mut cfg, weights) = tiny_vision_splice_model();
    cfg.vision_config = Some(VisionModelConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 1,
        patch_size: 1,
        spatial_merge_size: 2,
        out_hidden_size: cfg.hidden_size,
        temporal_patch_size: 1,
        num_position_embeddings: 1,
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: None,
    });
    (cfg, weights)
}

/// An internally-consistent multimodal request whose
/// `image_token_id` does not match the loaded checkpoint's must be rejected
/// up front, before image slots are selected -- otherwise it silently
/// injects/rotates at the wrong slots against a real checkpoint.
/// Mutation check: removing just this guard branch (leaving the other FIX-3
/// checks in place) lets the request reach the decoder unmodified; since
/// `generate_multimodal_f16`'s own injection loop keys off the request's
/// (not the checkpoint's) `image_token_id`, the run completes with `Ok`,
/// flipping this `expect_err` to a failing assertion.
#[test]
fn generate_multimodal_f16_rejects_mismatched_image_token_id() {
    use crate::vision::multimodal::Qwen35VisionRequest;
    use crate::vision::qwen35_vit::GridThw;

    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    assert_eq!(cfg.image_token_id, Some(3));

    // Internally consistent (validate() passes): pad id 2, matching grid/rows,
    // but 2 != the checkpoint's image_token_id (3).
    let request = Qwen35VisionRequest {
        input_ids: vec![0, 2, 2, 2, 2, 1],
        image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
        post_merger_rows: vec![0.1f32; 4 * cfg.hidden_size],
        image_token_id: 2,
        spatial_merge_size: 2,
        decoder_hidden_size: cfg.hidden_size,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
        .expect_err("mismatched image_token_id must be rejected, not silently run");
    let msg = format!("{err}");
    assert!(
        msg.contains("image_token_id"),
        "error must name image_token_id; got: {msg}"
    );
}

/// A request whose `spatial_merge_size` does not match the
/// checkpoint's `vision_config.spatial_merge_size` must be rejected up front.
/// Mutation check: removing just this guard branch lets an internally
/// consistent (but checkpoint-mismatched) request run to completion (`Ok`),
/// since nothing downstream re-derives merge size from `cfg.vision_config`.
#[test]
fn generate_multimodal_f16_rejects_mismatched_spatial_merge_size() {
    use crate::vision::multimodal::Qwen35VisionRequest;
    use crate::vision::qwen35_vit::GridThw;

    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    assert_eq!(cfg.vision_config.as_ref().unwrap().spatial_merge_size, 2);

    // merge_size=1 (checkpoint says 2): 1*4*4/1^2 = 16 post-merger rows,
    // internally consistent with the request's own spatial_merge_size.
    let mut input_ids = vec![0u32];
    input_ids.extend(std::iter::repeat_n(3u32, 16));
    input_ids.push(1);
    let request = Qwen35VisionRequest {
        input_ids,
        image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
        post_merger_rows: vec![0.1f32; 16 * cfg.hidden_size],
        image_token_id: 3,
        spatial_merge_size: 1,
        decoder_hidden_size: cfg.hidden_size,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
        .expect_err("mismatched spatial_merge_size must be rejected, not silently run");
    let msg = format!("{err}");
    assert!(
        msg.contains("spatial_merge_size"),
        "error must name spatial_merge_size; got: {msg}"
    );
}

/// A request whose `decoder_hidden_size` does not match
/// the checkpoint's `hidden_size` must be rejected up front by name, before
/// image slots are selected -- not only by the unrelated, later
/// `forward_step_f16` injected-row-length guard that happens to catch this
/// specific case too. Asserting the error text names `decoder_hidden_size`
/// keeps this test sensitive to *this* guard specifically.
/// Mutation check: removing just this guard branch still leaves the request
/// failing (via `forward_step_f16`'s unrelated length check deep in the
/// prefill loop), but the error text no longer mentions
/// `decoder_hidden_size` -- flipping the `contains` assertion to failing.
#[test]
fn generate_multimodal_f16_rejects_mismatched_decoder_hidden_size() {
    use crate::vision::multimodal::Qwen35VisionRequest;
    use crate::vision::qwen35_vit::GridThw;

    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    assert_eq!(cfg.hidden_size, 8);

    // decoder_hidden_size=4 (checkpoint hidden_size is 8): internally
    // consistent request (post_merger_rows sized to 4 rows * 4), but wrong
    // against this checkpoint.
    let request = Qwen35VisionRequest {
        input_ids: vec![0, 3, 3, 3, 3, 1],
        image_grids: vec![GridThw { t: 1, h: 4, w: 4 }],
        post_merger_rows: vec![0.1f32; 4 * 4],
        image_token_id: 3,
        spatial_merge_size: 2,
        decoder_hidden_size: 4,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg)
        .expect_err("mismatched decoder_hidden_size must be rejected, not silently run");
    let msg = format!("{err}");
    assert!(
        msg.contains("decoder_hidden_size"),
        "error must name decoder_hidden_size (not just the unrelated downstream \
             injected-row-length message); got: {msg}"
    );
}

/// The M-RoPE table builder resolves
/// `partial_rotary_factor` from `cfg.rope_parameters`, while the attention
/// loop derives its rotary half-width from the separately public
/// `cfg.rope_dim()` (`cfg.partial_rotary_factor`). A constructible config
/// where these diverge (`head_dim=256`, `cfg.partial_rotary_factor=0.5` ->
/// decoder half=64, but `rope_parameters.partial_rotary_factor=Some(0.25)`
/// with section `[11,11,10]` -> table half=32) must fail closed here, before
/// the first forward pass indexes `cos_row[32]`/`sin_row[32]` past the
/// table's actual 32-lane row and panics.
/// Mutation check: removing the guard lets `full_attention_step_f16` index
/// the 32-lane row at `half=64`, panicking mid-attention instead of
/// returning the clean `InvalidInput` this test expects.
#[test]
fn generate_multimodal_f16_rejects_mismatched_mrope_row_width() {
    use crate::model::qwen35_config::RopeParams;
    use crate::vision::multimodal::Qwen35VisionRequest;

    let (mut cfg, weights) = tiny_vision_splice_model();
    cfg.head_dim = 256;
    cfg.partial_rotary_factor = 0.5; // decoder rotary half = 64
    cfg.rope_parameters = Some(RopeParams {
        rope_theta: 1.0e7,
        partial_rotary_factor: Some(0.25), // table half = 32 (diverges from 64)
        mrope_section: Some(vec![11, 11, 10]),
        mrope_interleaved: Some(true),
    });

    // Text-only request: no image, so this exercises the guard on the
    // prefill table path alone (the decode-time sibling guard covers the
    // per-token `build_decode_cos_sin` path independently).
    let request = Qwen35VisionRequest {
        input_ids: vec![0, 1, 2],
        image_grids: vec![],
        post_merger_rows: vec![],
        image_token_id: 3,
        spatial_merge_size: 2,
        decoder_hidden_size: cfg.hidden_size,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    let err = generate_multimodal_f16(&weights, &cfg, &request, &gen_cfg).expect_err(
        "a config whose rope_parameters and rope_dim() disagree on rotary width must be \
             rejected, not panic at attention-lane indexing",
    );
    assert!(
        matches!(err, crate::error::InferenceError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

// -----------------------------------------------------------------
// Pooled embedding extraction (vision-embed-pooling)
// -----------------------------------------------------------------

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// A one-image request over [`tiny_vision_splice_model_with_vision_cfg`]:
/// grid (1,4,4), merge_size=2 -> 4 post-merger rows, embedded in a short
/// text scaffold `[0, <pad>x4, 1]`. `visual_rows` lets callers vary the
/// "image content" while keeping every shape/id fixed.
fn one_image_request(visual_rows: Vec<f32>) -> Qwen35VisionRequest {
    let mut input_ids = vec![0u32];
    input_ids.extend(std::iter::repeat_n(3u32, 4));
    input_ids.push(1);
    Qwen35VisionRequest {
        input_ids,
        image_grids: vec![crate::vision::qwen35_vit::GridThw { t: 1, h: 4, w: 4 }],
        post_merger_rows: visual_rows,
        image_token_id: 3,
        spatial_merge_size: 2,
        decoder_hidden_size: 8,
    }
}

#[test]
fn embed_image_f16_is_deterministic() {
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    let request = one_image_request(vec![0.3f32; 4 * cfg.hidden_size]);

    let v1 = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::MeanVisualTokens)
        .expect("embed_image_f16 succeeds");
    let v2 = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::MeanVisualTokens)
        .expect("embed_image_f16 succeeds");
    assert_eq!(v1, v2, "same input must produce an identical vector");

    let v3 = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::LastToken)
        .expect("embed_image_f16 succeeds");
    let v4 = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::LastToken)
        .expect("embed_image_f16 succeeds");
    assert_eq!(
        v3, v4,
        "same input must produce an identical vector (LastToken)"
    );
}

#[test]
fn embed_image_f16_is_finite_and_unit_norm() {
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    for pooling in [
        PoolingStrategy::MeanVisualTokens,
        PoolingStrategy::LastToken,
    ] {
        let request = one_image_request(vec![0.4f32; 4 * cfg.hidden_size]);
        let v =
            embed_image_f16(&weights, &cfg, &request, pooling).expect("embed_image_f16 succeeds");
        assert_eq!(v.len(), cfg.hidden_size);
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{pooling:?}: non-finite output"
        );
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "{pooling:?}: expected unit norm, got {norm}"
        );
    }
}

#[test]
fn embed_image_f16_discriminates_different_images_but_matches_itself() {
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();

    let request_a = one_image_request(
        (0..4 * cfg.hidden_size)
            .map(|i| (i as f32) * 0.05 - 0.8)
            .collect(),
    );
    let request_b = one_image_request(
        (0..4 * cfg.hidden_size)
            .map(|i| -(i as f32) * 0.03 + 0.5)
            .collect(),
    );

    let emb_a = embed_image_f16(
        &weights,
        &cfg,
        &request_a,
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("embed_image_f16 succeeds");
    let emb_a_again = embed_image_f16(
        &weights,
        &cfg,
        &request_a,
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("embed_image_f16 succeeds");
    let emb_b = embed_image_f16(
        &weights,
        &cfg,
        &request_b,
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("embed_image_f16 succeeds");

    let self_cos = cosine(&emb_a, &emb_a_again);
    assert!(
        (self_cos - 1.0).abs() < 1e-5,
        "an image embedded against itself must have cosine ~1.0, got {self_cos}"
    );

    let cross_cos = cosine(&emb_a, &emb_b);
    assert!(
        cross_cos < 0.999,
        "two different images must not collapse to near-identical embeddings, got cosine {cross_cos}"
    );
}

/// Direct unit test on the pooling primitive: pooling over the correct
/// image-pad window vs. an off-by-one-shifted window over the SAME
/// hidden-state matrix must produce different vectors. This is the
/// mutation-sensitivity gate for the position-selection logic
/// `embed_image_f16` relies on — an off-by-one bug in
/// `image_pad_positions` (or a copy-pasted sibling that reintroduces
/// one) changes the output instead of silently passing.
#[test]
fn pool_hidden_states_wrong_positions_change_the_output() {
    let hidden_size = 4;
    let seq_len = 6;
    // Row i is a constant-i vector, so shifting the pooled window by one
    // position is guaranteed to change the mean.
    let hidden_states: Vec<f32> = (0..seq_len)
        .flat_map(|i| std::iter::repeat_n(i as f32, hidden_size))
        .collect();

    let correct_positions = [1usize, 2, 3, 4];
    let off_by_one_positions = [2usize, 3, 4, 5];

    let correct = pool_hidden_states(
        &hidden_states,
        hidden_size,
        seq_len,
        &correct_positions,
        PoolingStrategy::MeanVisualTokens,
    );
    let wrong = pool_hidden_states(
        &hidden_states,
        hidden_size,
        seq_len,
        &off_by_one_positions,
        PoolingStrategy::MeanVisualTokens,
    );

    assert_ne!(
        correct, wrong,
        "pooling over an off-by-one-shifted position window must change the output"
    );
}

#[test]
fn embed_image_f16_wrong_pad_run_placement_changes_embedding() {
    // Same checkpoint, same post-merger rows, but the image-pad run sits
    // at a different physical offset in input_ids (shifted by one text
    // token) -- the practical shape of a real "wrong positions" bug
    // (e.g. an off-by-one in scaffold assembly). The resulting pooled
    // embedding must differ: different M-RoPE coordinates and different
    // neighboring context both feed into it.
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    let visual_rows = vec![0.25f32; 4 * cfg.hidden_size];

    let correct = one_image_request(visual_rows.clone());
    let mut shifted_ids = vec![0u32, 2]; // extra leading text token
    shifted_ids.extend(std::iter::repeat_n(3u32, 4));
    shifted_ids.push(1);
    let shifted = Qwen35VisionRequest {
        input_ids: shifted_ids,
        ..one_image_request(visual_rows)
    };

    let emb_correct = embed_image_f16(&weights, &cfg, &correct, PoolingStrategy::MeanVisualTokens)
        .expect("embed_image_f16 succeeds");
    let emb_shifted = embed_image_f16(&weights, &cfg, &shifted, PoolingStrategy::MeanVisualTokens)
        .expect("embed_image_f16 succeeds");

    assert_ne!(
        emb_correct, emb_shifted,
        "shifting the image-pad run's position in input_ids must change the pooled embedding"
    );
}

#[test]
fn embed_text_vlm_f16_is_deterministic_and_unit_norm() {
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    let mut vocab_map = std::collections::HashMap::new();
    for (i, c) in ["a", "b", "c"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    let tokenizer =
        BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs");

    for pooling in [
        PoolingStrategy::MeanVisualTokens,
        PoolingStrategy::LastToken,
    ] {
        let v1 = embed_text_vlm_f16(&weights, &cfg, &tokenizer, "abc", pooling)
            .expect("embed_text_vlm_f16 succeeds");
        let v2 = embed_text_vlm_f16(&weights, &cfg, &tokenizer, "abc", pooling)
            .expect("embed_text_vlm_f16 succeeds");
        assert_eq!(
            v1, v2,
            "{pooling:?}: same prompt must produce an identical vector"
        );
        assert_eq!(v1.len(), cfg.hidden_size);
        let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "{pooling:?}: expected unit norm, got {norm}"
        );
    }
}

#[test]
fn embed_text_vlm_f16_and_embed_image_f16_share_the_same_space() {
    // Not a quality claim (see PoolingStrategy's doc comment) -- just
    // proves both entry points route through the same decoder + pooling
    // call so their outputs are directly comparable vectors of the same
    // dimension, which is the structural property retrieval depends on.
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    let mut vocab_map = std::collections::HashMap::new();
    vocab_map.insert("a".to_string(), 0u32);
    let tokenizer =
        BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs");

    let text_emb = embed_text_vlm_f16(&weights, &cfg, &tokenizer, "a", PoolingStrategy::LastToken)
        .expect("embed_text_vlm_f16 succeeds");
    let image_emb = embed_image_f16(
        &weights,
        &cfg,
        &one_image_request(vec![0.1f32; 4 * cfg.hidden_size]),
        PoolingStrategy::LastToken,
    )
    .expect("embed_image_f16 succeeds");

    assert_eq!(text_emb.len(), image_emb.len());
    let cos = cosine(&text_emb, &image_emb);
    assert!(cos.is_finite());
}

#[test]
fn embed_text_vlm_f16_rejects_prompt_colliding_with_image_token_id() {
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    assert_eq!(cfg.image_token_id, Some(3));
    // Vocab entry "z" is deliberately assigned id 3, the checkpoint's
    // image_token_id -- a tokenized prompt must never silently contain it.
    let mut vocab_map = std::collections::HashMap::new();
    vocab_map.insert("z".to_string(), 3u32);
    let tokenizer =
        BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs");

    let err = embed_text_vlm_f16(&weights, &cfg, &tokenizer, "z", PoolingStrategy::LastToken)
        .expect_err("a prompt colliding with image_token_id must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
}

#[test]
fn embed_image_f16_rejects_invalid_request() {
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    let mut request = one_image_request(vec![0.1f32; 4 * cfg.hidden_size]);
    request.post_merger_rows.pop(); // now the wrong length
    assert!(embed_image_f16(&weights, &cfg, &request, PoolingStrategy::MeanVisualTokens).is_err());
}

/// The context-window limit must be evaluated BEFORE `build_mrope_tables`
/// materializes per-token position/cos/sin rows: an over-context request
/// must fail cheaply, not after unbounded allocation work. The request
/// here passes `Qwen35VisionRequest::validate` (run count, TOTAL pad
/// count, and row-buffer length all line up) but carries per-run lengths
/// [1, 3] against grids expecting [2, 2] — a mismatch only the M-RoPE
/// builder detects — so getting the context-window error (not the
/// builder's run-length error) proves the ordering.
#[test]
fn prefill_rejects_over_context_before_mrope_table_construction() {
    let (mut cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    let base = one_image_request(vec![0.1f32; 4 * cfg.hidden_size]);
    let request = Qwen35VisionRequest {
        // Two pad runs of lengths 1 and 3 (total 4, matching the grids'
        // total merged rows), while each grid below expects a run of 2.
        input_ids: vec![0u32, 3, 1, 3, 3, 3, 1],
        image_grids: vec![
            crate::vision::qwen35_vit::GridThw { t: 1, h: 2, w: 4 },
            crate::vision::qwen35_vit::GridThw { t: 1, h: 2, w: 4 },
        ],
        ..base
    };
    request
        .validate()
        .expect("request must pass validation so only the builder would catch it");
    cfg.max_position_embeddings = request.input_ids.len() - 1;

    let err = prefill_hidden_states_f16(&weights, &cfg, &request)
        .expect_err("over-context request must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("context window"),
        "must fail on the context-window check, before M-RoPE table \
             construction; got: {msg}"
    );
}

/// Golden-reference check for `embed_image_f16`'s own image-pad
/// position-selection wiring (not just the generic `pool_hidden_states`
/// primitive, which `pool_hidden_states_wrong_positions_change_the_output`
/// already covers in isolation): `one_image_request`'s fixed layout
/// `[0, <pad>x4, 1]` puts the pad run at physical positions `[1,2,3,4]`
/// by construction. This test independently derives the expected
/// pooled/normalized vector from `prefill_hidden_states_f16` using that
/// hand-known-correct window and asserts `embed_image_f16` matches it
/// exactly.
///
/// Mutation-sensitive: an off-by-one in `embed_image_f16`'s
/// `image_pad_positions` computation (e.g. `.map(|(i, _)| i + 1)`) still
/// passes `embed_image_f16_is_deterministic`,
/// `_discriminates_different_images_but_matches_itself`, and
/// `_wrong_pad_run_placement_changes_embedding` (all of them assert only
/// relative properties -- determinism, discrimination, "differs from a
/// differently-shaped request" -- that remain true under a consistently
/// applied shift), but fails this golden check because the reference
/// value is computed independently of that internal computation.
#[test]
fn embed_image_f16_matches_independently_computed_golden_pool() {
    let (cfg, weights) = tiny_vision_splice_model_with_vision_cfg();
    let request = one_image_request(vec![0.37f32; 4 * cfg.hidden_size]);

    let hidden_states = prefill_hidden_states_f16(&weights, &cfg, &request)
        .expect("prefill_hidden_states_f16 succeeds");
    let known_correct_pad_positions = [1usize, 2, 3, 4]; // by construction of one_image_request
    let golden = l2_normalize_owned(mean_pool_rows(
        &hidden_states,
        cfg.hidden_size,
        &known_correct_pad_positions,
    ));

    let got = embed_image_f16(&weights, &cfg, &request, PoolingStrategy::MeanVisualTokens)
        .expect("embed_image_f16 succeeds");

    assert_eq!(
        got, golden,
        "embed_image_f16 must pool over exactly the known-correct image-pad positions"
    );
}
