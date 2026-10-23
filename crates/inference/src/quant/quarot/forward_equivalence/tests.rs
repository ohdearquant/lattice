use super::*;
use crate::model::qwen35_config::{LayerType, Qwen35Config, compute_layer_types};
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::lm_head::{
    materialize_lm_head_for_qwen35, qwen35_final_norm_fusion_target,
};
use crate::quant::quarot::pipeline::{absorb_rotations, fuse_rmsnorms};
use crate::quant::quarot::plan::RotationPlan;
use crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan;

/// Tiny Qwen3.5 config tuned for tractable probe tests:
/// - `hidden_size = 8` (power of 2 for Hadamard)
/// - 2 layers, one GDN + one GQA so the probe exercises both branches
/// - small intermediate/vocab so matvecs stay sub-millisecond
fn tiny_test_cfg() -> Qwen35Config {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.hidden_size = 8;
    cfg.num_hidden_layers = 2;
    cfg.vocab_size = 4;
    cfg.intermediate_size = 16;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 4;
    cfg.linear_num_key_heads = 1;
    cfg.linear_key_head_dim = 2;
    cfg.linear_value_head_dim = 2;
    cfg.linear_num_value_heads = Some(1);
    cfg.full_attention_interval = 2;
    cfg.layer_types = compute_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
    cfg.layer_mask = vec![true; cfg.num_hidden_layers];
    cfg.tie_word_embeddings = false;
    cfg.rms_norm_eps = 1e-6;
    cfg
}

fn tied_tiny_test_cfg() -> Qwen35Config {
    let mut cfg = tiny_test_cfg();
    cfg.tie_word_embeddings = true;
    cfg
}

fn synthetic_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 11) as u32;
            (bits as f64 / u32::MAX as f64) - 0.5
        })
        .collect()
}

fn insert_tensor(
    tensors: &mut HashMap<String, TensorEntry>,
    name: &str,
    shape: Vec<usize>,
    data: Vec<f64>,
) {
    tensors.insert(
        name.to_string(),
        TensorEntry {
            name: name.to_string(),
            shape,
            data,
        },
    );
}

/// Build a working set with shapes consistent with `cfg`'s loader
/// expectations. Includes every tensor the fusion plan and rotation
/// plan reference, plus `embed_tokens` and `final_norm`. Does NOT
/// include `lm_head.weight` — the caller adds it separately (or via
/// `materialize_lm_head_for_qwen35` for tied configs).
fn build_working_set(cfg: &Qwen35Config, seed: u64) -> HashMap<String, TensorEntry> {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let intermediate = cfg.intermediate_size;
    let full_q_dim = cfg.full_q_dim();
    let full_kv_dim = cfg.full_kv_dim();
    let linear_qkv_dim = cfg.linear_qkv_dim();
    let linear_output_dim = cfg.linear_output_dim();
    let linear_num_heads = cfg.linear_num_key_heads;

    let mut tensors = HashMap::new();
    insert_tensor(
        &mut tensors,
        QWEN35_EMBED_TOKENS_NAME,
        vec![vocab, hidden],
        synthetic_f64(vocab * hidden, seed.wrapping_add(1)),
    );
    insert_tensor(
        &mut tensors,
        QWEN35_FINAL_NORM_NAME,
        vec![hidden],
        synthetic_f64(hidden, seed.wrapping_add(2)),
    );

    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{i}");
        let layer_seed = seed.wrapping_add(100 + i as u64);

        insert_tensor(
            &mut tensors,
            &format!("{prefix}.input_layernorm.weight"),
            vec![hidden],
            synthetic_f64(hidden, layer_seed.wrapping_add(1)),
        );
        insert_tensor(
            &mut tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
            synthetic_f64(hidden, layer_seed.wrapping_add(2)),
        );

        if cfg.is_full_attention(i) {
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.q_proj.weight"),
                vec![2 * full_q_dim, hidden],
                synthetic_f64(2 * full_q_dim * hidden, layer_seed.wrapping_add(10)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.k_proj.weight"),
                vec![full_kv_dim, hidden],
                synthetic_f64(full_kv_dim * hidden, layer_seed.wrapping_add(11)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.v_proj.weight"),
                vec![full_kv_dim, hidden],
                synthetic_f64(full_kv_dim * hidden, layer_seed.wrapping_add(12)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, full_q_dim],
                synthetic_f64(hidden * full_q_dim, layer_seed.wrapping_add(13)),
            );
        } else {
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                vec![linear_qkv_dim, hidden],
                synthetic_f64(linear_qkv_dim * hidden, layer_seed.wrapping_add(20)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.linear_attn.in_proj_z.weight"),
                vec![linear_output_dim, hidden],
                synthetic_f64(linear_output_dim * hidden, layer_seed.wrapping_add(21)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.linear_attn.in_proj_a.weight"),
                vec![linear_num_heads, hidden],
                synthetic_f64(linear_num_heads * hidden, layer_seed.wrapping_add(22)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.linear_attn.in_proj_b.weight"),
                vec![linear_num_heads, hidden],
                synthetic_f64(linear_num_heads * hidden, layer_seed.wrapping_add(23)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.linear_attn.out_proj.weight"),
                vec![hidden, linear_output_dim],
                synthetic_f64(hidden * linear_output_dim, layer_seed.wrapping_add(24)),
            );
        }

        insert_tensor(
            &mut tensors,
            &format!("{prefix}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
            synthetic_f64(intermediate * hidden, layer_seed.wrapping_add(30)),
        );
        insert_tensor(
            &mut tensors,
            &format!("{prefix}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
            synthetic_f64(intermediate * hidden, layer_seed.wrapping_add(31)),
        );
        insert_tensor(
            &mut tensors,
            &format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
            synthetic_f64(hidden * intermediate, layer_seed.wrapping_add(32)),
        );
    }

    tensors
}

fn full_pipeline_plans(
    cfg: &Qwen35Config,
) -> (
    Vec<crate::quant::quarot::rmsnorm_fusion::RmsNormFusionTarget>,
    RotationPlan,
) {
    let mut fusion = qwen35_per_layer_fusion_plan(cfg).unwrap();
    fusion.push(qwen35_final_norm_fusion_target());
    (fusion, RotationPlan::qwen35_residual_stream_linear_layers())
}

/// Sanity: the tiny test config really has one GDN + one GQA layer.
#[test]
fn tiny_cfg_has_one_full_and_one_linear_layer() {
    let cfg = tiny_test_cfg();
    assert_eq!(cfg.num_hidden_layers, 2);
    assert_eq!(cfg.layer_types[0], LayerType::LinearAttention);
    assert_eq!(cfg.layer_types[1], LayerType::FullAttention);
}

#[test]
#[allow(
    clippy::type_complexity,
    reason = "the explicit completion signature is the compile-time assertion"
)]
fn prepared_completion_cannot_accept_dense_to_moe_handoff() {
    let dense_cfg = tied_tiny_test_cfg();
    let original = build_working_set(&dense_cfg, 0xA11C_E5E5);
    let rotation = RandomizedHadamard::new(0xD3A5_EA5E, dense_cfg.hidden_size).unwrap();
    let forward_cfg = ForwardEquivalenceConfig {
        num_probe_tokens: 1,
        ..Default::default()
    };
    let snapshot =
        prepare_forward_equivalence_qwen35(&original, &dense_cfg, &rotation, &forward_cfg).unwrap();
    assert!(std::ptr::eq(snapshot.cfg, &dense_cfg));
    assert!(std::ptr::eq(snapshot.rotation, &rotation));

    let mut mismatched_moe_cfg = dense_cfg.clone();
    mismatched_moe_cfg.num_experts = Some(1);
    assert!(!dense_cfg.is_moe());
    assert!(mismatched_moe_cfg.is_moe());

    let completion: for<'snapshot, 'reader, 'rotated> fn(
        ForwardEquivalenceSnapshot<'snapshot>,
        &'reader QuarotTensorReader,
        &'rotated HashMap<String, TensorEntry>,
    ) -> Result<
        ForwardEquivalenceReport,
        InferenceError,
    > = assert_prepared_forward_equivalence_qwen35;

    let _ = (completion, snapshot, mismatched_moe_cfg);
}

#[test]
fn prepared_snapshot_enforces_configured_logit_budget_before_tensor_access() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let original = HashMap::new();
    let rotation = RandomizedHadamard::new(0x51A5_EEED, cfg.hidden_size).unwrap();
    let bytes_per_probe =
        cfg.vocab_size * size_of::<f64>() + size_of::<Vec<f64>>() + size_of::<u32>();
    let max_probe_tokens = MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES / bytes_per_probe;
    assert_eq!(max_probe_tokens, 33);
    assert!(max_probe_tokens * bytes_per_probe <= MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES);
    assert!(
        (max_probe_tokens + 1) * bytes_per_probe > MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES
    );

    let at_limit_cfg = ForwardEquivalenceConfig {
        num_probe_tokens: max_probe_tokens,
        ..Default::default()
    };
    let at_limit = prepare_forward_equivalence_qwen35(&original, &cfg, &rotation, &at_limit_cfg)
        .err()
        .expect("the empty fixture must fail at tensor access")
        .to_string();
    assert!(
        at_limit.contains(QWEN35_EMBED_TOKENS_NAME),
        "the largest in-budget probe count must reach tensor access: {at_limit}"
    );

    let over_limit_cfg = ForwardEquivalenceConfig {
        num_probe_tokens: max_probe_tokens + 1,
        ..Default::default()
    };
    let over_limit =
        prepare_forward_equivalence_qwen35(&original, &cfg, &rotation, &over_limit_cfg)
            .err()
            .expect("the over-budget request must fail admission")
            .to_string();
    assert!(
        over_limit.contains("retained chain-logit budget"),
        "one probe past the configured limit must be rejected before tensor access: \
             {over_limit}"
    );
    assert!(
        !over_limit.contains(QWEN35_EMBED_TOKENS_NAME),
        "budget rejection must precede tensor access: {over_limit}"
    );
}

#[test]
fn prepared_snapshot_rejects_over_budget_before_any_owned_allocation() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let original = HashMap::new();
    let rotation = RandomizedHadamard::new(0x51A5_EEED, cfg.hidden_size).unwrap();
    let bytes_per_probe =
        cfg.vocab_size * size_of::<f64>() + size_of::<Vec<f64>>() + size_of::<u32>();
    let max_probe_tokens = MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES / bytes_per_probe;
    let forward_cfg = ForwardEquivalenceConfig {
        num_probe_tokens: max_probe_tokens + 1,
        ..Default::default()
    };

    let tracking = pre_admission_allocation_tracking::start();
    let result = prepare_forward_equivalence_qwen35(&original, &cfg, &rotation, &forward_cfg);
    let observation = tracking.finish();

    assert!(
        observation.rejection_seen,
        "the measured call must reach budget rejection"
    );
    assert_eq!(
        observation.before_rejection_allocation_calls, 0,
        "over-budget preparation allocated before rejecting"
    );
    assert!(
        observation.after_rejection_allocation_calls > 0,
        "the diagnostic allocation after budget rejection must be observed"
    );
    let error = result
        .err()
        .expect("the over-budget request must fail admission")
        .to_string();
    assert!(
        error.contains("retained chain-logit budget"),
        "unexpected error: {error}"
    );
}

#[test]
fn prepared_snapshot_rejects_logit_budget_arithmetic_overflow() {
    let original = HashMap::new();
    for (num_probe_tokens, vocab_size) in [(2, usize::MAX / 2 + 1), (1, usize::MAX / 8 + 1)] {
        let mut cfg = tied_tiny_test_cfg();
        cfg.vocab_size = vocab_size;
        let rotation = RandomizedHadamard::new(0x51A5_EEED, cfg.hidden_size).unwrap();
        let forward_cfg = ForwardEquivalenceConfig {
            num_probe_tokens,
            ..Default::default()
        };
        let err = prepare_forward_equivalence_qwen35(&original, &cfg, &rotation, &forward_cfg)
            .err()
            .expect("overflowing budget arithmetic must fail admission")
            .to_string();
        assert!(
            err.contains("retained chain-logit size overflow"),
            "overflowing budget arithmetic must return an error: {err}"
        );
    }
}

/// Happy path (untied): build → materialize is a no-op (already untied
/// with lm_head present) → fuse → rotate → harness must pass.
#[test]
fn forward_equivalence_passes_on_full_untied_pipeline() {
    let cfg = tiny_test_cfg();
    let mut original = build_working_set(&cfg, 1);
    // Untied: lm_head must be present pre-pipeline.
    insert_tensor(
        &mut original,
        QWEN35_LM_HEAD_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        synthetic_f64(cfg.vocab_size * cfg.hidden_size, 999),
    );
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0xC0FFEE, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    let report = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap();
    assert!(report.max_abs_error < 1e-5, "unexpected delta: {report:?}");
    assert_eq!(report.probe_tokens.len(), 4);
    for &t in &report.probe_tokens {
        assert!((t as usize) < cfg.vocab_size);
    }
}

/// Happy path (tied): pre-pipeline lm_head is absent (probe falls
/// back to embed_tokens); rotated set materializes + fuses + rotates.
#[test]
fn forward_equivalence_passes_on_full_tied_pipeline() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 2);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0xFEED_FACE, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    let report = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap();
    assert!(report.max_abs_error < 1e-5, "unexpected delta: {report:?}");
}

/// Refuse: rotate without fusing the final_norm target. The mismatch
/// `lm_head_rot · rmsnorm(R·h)` vs `embed_tokens · diag(1+γ_final) · rmsnorm(h)`
/// produces a non-trivial delta and the gate must refuse.
#[test]
fn forward_equivalence_refuses_when_final_norm_fusion_skipped() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 4);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    // BUG: drop the final_norm fusion target — fuse only per-layer norms.
    let per_layer = qwen35_per_layer_fusion_plan(&cfg).unwrap();
    fuse_rmsnorms(&mut rotated, &per_layer).unwrap();
    let rotation = RandomizedHadamard::new(0xBADC0DE, cfg.hidden_size).unwrap();
    absorb_rotations(
        &mut rotated,
        &RotationPlan::qwen35_residual_stream_linear_layers(),
        &rotation,
    )
    .unwrap();

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("forward-equivalence refused"),
        "unexpected error: {msg}"
    );
    assert!(msg.contains("exceeds tolerance"), "unexpected error: {msg}");
}

/// Refuse: a single tensor in the rotated set is silently corrupted
/// after a correct pipeline run. Real-world analogue: a converter bug
/// that re-quantizes a tensor a second time, or scales the wrong
/// matrix. The harness must catch this even though the rotation
/// algebra itself was applied correctly to every other tensor.
#[test]
fn forward_equivalence_refuses_on_corrupted_rotated_tensor() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 5);
    let mut rotated = original.clone();

    // Correct pipeline.
    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0x55AA_55AA, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    // Corrupt one matrix after the pipeline.
    let lm = rotated
        .get_mut(QWEN35_LM_HEAD_NAME)
        .expect("lm_head should exist after materialize");
    for v in lm.data.iter_mut() {
        *v *= 1.25;
    }

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("forward-equivalence refused"),
        "unexpected error: {msg}"
    );
}

/// The poisoned rotated norm is consumed only by the chain probe, so
/// removing the chain's non-finite guard lets this conversion pass.
#[test]
fn forward_equivalence_refuses_non_finite_chain_probe_error() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 32);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0xCA11_0BAD, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    rotated.get_mut(QWEN35_FINAL_NORM_NAME).unwrap().data[0] = f64::NAN;

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("chain probe non-finite error"),
        "unexpected error: {msg}"
    );
    assert!(msg.contains("abs_error=NaN"), "unexpected error: {msg}");
}

/// `k_proj` is excluded from the chain probe, so removing the
/// per-tensor non-finite guard lets this conversion pass.
#[test]
fn forward_equivalence_refuses_non_finite_per_tensor_error() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 33);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0xFACE_FEED, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    let k_name = "model.language_model.layers.1.self_attn.k_proj.weight";
    rotated.get_mut(k_name).unwrap().data[0] = f64::NAN;

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("per-tensor non-finite error"),
        "unexpected error: {msg}"
    );
    assert!(msg.contains(k_name), "unexpected error: {msg}");
    assert!(msg.contains("abs_error=NaN"), "unexpected error: {msg}");
}

/// Refuse: rotate without fusing per-layer norms first. Rotation
/// absorbed AROUND a non-zero γ_in breaks the residual stream
/// because `diag(1+γ) · R^T ≠ R^T · diag(1+γ)`.
#[test]
fn forward_equivalence_refuses_when_per_layer_fusion_skipped() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 6);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    // BUG: only fuse final_norm, skip per-layer.
    let final_only = vec![qwen35_final_norm_fusion_target()];
    fuse_rmsnorms(&mut rotated, &final_only).unwrap();
    let rotation = RandomizedHadamard::new(0xABCDEF, cfg.hidden_size).unwrap();
    absorb_rotations(
        &mut rotated,
        &RotationPlan::qwen35_residual_stream_linear_layers(),
        &rotation,
    )
    .unwrap();

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("forward-equivalence refused"),
        "unexpected error: {msg}"
    );
}

/// Untied original missing lm_head → harness errors with a useful
/// message (vs. the silent fall-through that would happen for tied).
#[test]
fn forward_equivalence_errors_on_untied_original_missing_lm_head() {
    let cfg = tiny_test_cfg(); // untied
    let original = build_working_set(&cfg, 7); // no lm_head added
    let mut rotated = original.clone();
    insert_tensor(
        &mut rotated,
        QWEN35_LM_HEAD_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        synthetic_f64(cfg.vocab_size * cfg.hidden_size, 7777),
    );
    let rotation = RandomizedHadamard::new(1, cfg.hidden_size).unwrap();

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains(QWEN35_LM_HEAD_NAME), "unexpected error: {msg}");
    assert!(msg.contains("untied"), "unexpected error: {msg}");
}

#[test]
fn forward_equivalence_errors_on_missing_required_tensor() {
    let cfg = tiny_test_cfg();
    let mut original = build_working_set(&cfg, 8);
    insert_tensor(
        &mut original,
        QWEN35_LM_HEAD_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        synthetic_f64(cfg.vocab_size * cfg.hidden_size, 888),
    );
    let rotated = original.clone();

    // Remove a required tensor from `original` to simulate a partial load.
    original.remove(QWEN35_FINAL_NORM_NAME);
    let rotation = RandomizedHadamard::new(2, cfg.hidden_size).unwrap();

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(QWEN35_FINAL_NORM_NAME),
        "unexpected error: {msg}"
    );
    assert!(
        msg.contains("not in working set"),
        "unexpected error: {msg}"
    );
}

#[test]
fn forward_equivalence_errors_on_shape_mismatch() {
    let cfg = tiny_test_cfg();
    let mut original = build_working_set(&cfg, 9);
    // Inject a wrong-shape embed_tokens.
    original.insert(
        QWEN35_EMBED_TOKENS_NAME.to_string(),
        TensorEntry {
            name: QWEN35_EMBED_TOKENS_NAME.to_string(),
            shape: vec![cfg.vocab_size, cfg.hidden_size + 1],
            data: vec![0.0; cfg.vocab_size * (cfg.hidden_size + 1)],
        },
    );
    insert_tensor(
        &mut original,
        QWEN35_LM_HEAD_NAME,
        vec![cfg.vocab_size, cfg.hidden_size],
        synthetic_f64(cfg.vocab_size * cfg.hidden_size, 999),
    );
    let rotated = original.clone();
    let rotation = RandomizedHadamard::new(3, cfg.hidden_size).unwrap();

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains(QWEN35_EMBED_TOKENS_NAME),
        "unexpected error: {msg}"
    );
    assert!(msg.contains("shape"), "unexpected error: {msg}");
}

#[test]
fn forward_equivalence_rejects_moe_config() {
    let cfg = Qwen35Config::qwen36_35b_a3b();
    assert!(cfg.is_moe());
    let original = HashMap::new();
    let rotated = HashMap::new();
    // MoE check returns before rotation.dim() is consulted; any
    // power-of-2-dim rotation suffices for the parameter slot.
    let rotation = RandomizedHadamard::new(0, 8).unwrap();
    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("MoE"), "unexpected error: {msg}");
}

#[test]
fn forward_equivalence_rejects_zero_probe_tokens() {
    let cfg = tiny_test_cfg();
    let original = HashMap::new();
    let rotated = HashMap::new();
    let rotation = RandomizedHadamard::new(0, cfg.hidden_size).unwrap();
    let fc = ForwardEquivalenceConfig {
        num_probe_tokens: 0,
        ..Default::default()
    };
    let err =
        assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &rotation, &fc).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("num_probe_tokens must be > 0"),
        "unexpected error: {msg}"
    );
}

#[test]
fn forward_equivalence_rejects_non_positive_tolerance() {
    let cfg = tiny_test_cfg();
    let original = HashMap::new();
    let rotated = HashMap::new();
    let rotation = RandomizedHadamard::new(0, cfg.hidden_size).unwrap();
    for bad in [0.0_f64, -1e-5, f64::NAN] {
        let fc = ForwardEquivalenceConfig {
            tolerance: bad,
            ..Default::default()
        };
        let err = assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &rotation, &fc)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("tolerance must be a positive finite value"),
            "unexpected error for tolerance={bad}: {msg}"
        );
    }
}

/// Refuse: rotation.dim() != cfg.hidden_size — caller passed a
/// rotation built for the wrong space.
#[test]
fn forward_equivalence_rejects_rotation_dim_mismatch() {
    let cfg = tiny_test_cfg();
    let original = build_working_set(&cfg, 11);
    let rotated = original.clone();
    let rotation = RandomizedHadamard::new(0, cfg.hidden_size * 2).unwrap();
    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("rotation.dim()"), "unexpected error: {msg}");
    assert!(
        msg.contains(&format!("cfg.hidden_size={}", cfg.hidden_size)),
        "unexpected error: {msg}"
    );
}

/// Determinism: same seed → same probe tokens. Catches a clock-based or
/// uninitialized RNG regression.
#[test]
fn probe_tokens_are_deterministic_in_seed() {
    let a = deterministic_probe_tokens(0xDEAD_BEEF, 4, 100);
    let b = deterministic_probe_tokens(0xDEAD_BEEF, 4, 100);
    assert_eq!(a, b);
    let c = deterministic_probe_tokens(0xDEAD_BEEF_u64.wrapping_add(1), 4, 100);
    assert_ne!(
        a, c,
        "different seeds should produce different probe tokens"
    );
}

/// Refuse error message includes the observed delta and tolerance —
/// catches the regression where the error message would lose the
/// diagnostic numbers operators need to triage in the binary.
#[test]
fn refuse_error_message_includes_max_abs_and_tolerance() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 10);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    // Real divergence: fuse only per-layer norms (skip final_norm) then
    // rotate. `diag(1 + γ_final) · R^T ≠ R^T · diag(1 + γ_final)` makes
    // the rotated lm_head produce different logits than the original.
    let per_layer = qwen35_per_layer_fusion_plan(&cfg).unwrap();
    fuse_rmsnorms(&mut rotated, &per_layer).unwrap();
    let rotation = RandomizedHadamard::new(0x1234_5678, cfg.hidden_size).unwrap();
    absorb_rotations(
        &mut rotated,
        &RotationPlan::qwen35_residual_stream_linear_layers(),
        &rotation,
    )
    .unwrap();

    let fc = ForwardEquivalenceConfig {
        tolerance: 1e-12,
        ..Default::default()
    };
    let err =
        assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &rotation, &fc).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("max_abs_error="), "unexpected error: {msg}");
    assert!(
        msg.contains("exceeds tolerance="),
        "unexpected error: {msg}"
    );
    // Either chain-probe or per-tensor message format is acceptable —
    // both name the failing check explicitly.
    assert!(
        msg.contains("chain probe") || msg.contains("per-tensor"),
        "expected refuse message to name the failing check: {msg}"
    );
}

/// Per-tensor coverage helper: run the full pipeline correctly, then
/// scale `victim_name` by `factor` in the rotated working set and
/// assert the gate refuses. Covers the chain-probe-skipped planned
/// tensors.
fn assert_corrupting_planned_tensor_refuses(victim_name: &str, factor: f64, seed: u64) {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, seed);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation =
        RandomizedHadamard::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    // Verify the pipeline is correct BEFORE corruption (so a refuse
    // after corruption is attributable to the corruption alone).
    assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_or_else(|e| {
        panic!("pre-corruption pipeline must pass for victim `{victim_name}`: {e}")
    });

    let victim = rotated
        .get_mut(victim_name)
        .unwrap_or_else(|| panic!("victim tensor `{victim_name}` missing from working set"));
    for v in victim.data.iter_mut() {
        *v *= factor;
    }

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("forward-equivalence refused"),
        "expected refuse for corrupted `{victim_name}`: {msg}"
    );
    assert!(
        msg.contains("per-tensor"),
        "corruption of `{victim_name}` should be caught by the per-tensor check: {msg}"
    );
}

/// Per-tensor coverage: corrupting `k_proj` is invisible to the chain
/// probe (probe shortcuts through `q_proj` → `o_proj`) but must be
/// caught by the per-tensor algebraic check. This regression case is
/// the specific gap the chain probe alone leaves in PR #28.
#[test]
fn per_tensor_check_catches_corrupted_k_proj() {
    assert_corrupting_planned_tensor_refuses(
        "model.language_model.layers.1.self_attn.k_proj.weight",
        1000.0,
        21,
    );
}

/// Per-tensor coverage: `v_proj` is also chain-probe-skipped.
#[test]
fn per_tensor_check_catches_corrupted_v_proj() {
    assert_corrupting_planned_tensor_refuses(
        "model.language_model.layers.1.self_attn.v_proj.weight",
        -2.0,
        22,
    );
}

/// Per-tensor coverage: `q_proj` is shape `[2 * full_q_dim, hidden]`
/// because Qwen3.5 fuses Q + gate_z. The chain probe takes only the
/// first `full_q_dim` (the Q half). Corrupting the second half
/// (gate_z) must still be caught.
#[test]
fn per_tensor_check_catches_corrupted_q_proj_gate_z_half() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 23);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0xAB12_34CD, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    // Locate the layer index that is a full-attention layer in this
    // tiny config (the tied_tiny_test_cfg has one full and one linear).
    let full_layer = (0..cfg.num_hidden_layers)
        .find(|&i| cfg.is_full_attention(i))
        .expect("tied_tiny_test_cfg must have at least one full-attention layer");
    let q_name = format!("model.language_model.layers.{full_layer}.self_attn.q_proj.weight");
    let full_q_dim = cfg.full_q_dim();
    let hidden = cfg.hidden_size;
    let victim = rotated.get_mut(&q_name).unwrap();
    // Scale only the gate-z half (rows full_q_dim .. 2 * full_q_dim).
    for r in full_q_dim..(2 * full_q_dim) {
        for c in 0..hidden {
            victim.data[r * hidden + c] *= 3.0;
        }
    }

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("per-tensor"), "unexpected error: {msg}");
}

/// Per-tensor coverage: GDN `in_proj_qkv` is chain-probe-skipped (the
/// GDN branch probes `in_proj_z`). Corruption must still refuse.
#[test]
fn per_tensor_check_catches_corrupted_in_proj_qkv() {
    assert_corrupting_planned_tensor_refuses(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        10.0,
        24,
    );
}

/// Per-tensor coverage: GDN `in_proj_a` is chain-probe-skipped.
#[test]
fn per_tensor_check_catches_corrupted_in_proj_a() {
    assert_corrupting_planned_tensor_refuses(
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        0.5,
        25,
    );
}

/// Per-tensor coverage: GDN `in_proj_b` is chain-probe-skipped.
#[test]
fn per_tensor_check_catches_corrupted_in_proj_b() {
    assert_corrupting_planned_tensor_refuses(
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        -1.0,
        26,
    );
}

/// Matrix-equivalence vs single-vector: the
/// single-probe-vector implementation would miss a row
/// perturbation orthogonal to the probe direction. The current
/// matrix-equivalence implementation compares every stored element,
/// so any non-zero element-level perturbation is caught. This test
/// confirms by perturbing a single element of `k_proj` (chain probe
/// never matvecs k_proj, single-vector check could miss the row).
#[test]
fn per_tensor_check_catches_single_element_perturbation_orthogonal_to_probe_vector() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 30);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0x9876_5432, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    // Sanity: the correct pipeline passes.
    assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap();

    // Perturb a single element of k_proj. A single-probe-vector
    // check could mask this if the probe row was orthogonal; the
    // matrix-equivalence check sees the element directly.
    let k_name = "model.language_model.layers.1.self_attn.k_proj.weight";
    let victim = rotated.get_mut(k_name).unwrap();
    victim.data[0] += 0.5;

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("per-tensor"), "unexpected error: {msg}");
}

/// Refuse: planned tensor missing from BOTH `original` and
/// `rotated`. Without an enumerated expected-planned-tensor set,
/// an iteration-driven check would silently skip the absent tensor
/// and the gate could return `Ok` despite a partial working set.
/// The gate must enumerate the config-required set and refuse.
#[test]
fn per_tensor_check_errors_when_planned_tensor_missing_from_both_maps() {
    let cfg = tied_tiny_test_cfg();
    let mut original = build_working_set(&cfg, 31);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0xFEED_0BAD, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    // Drop k_proj from BOTH maps. Chain probe never uses k_proj, so
    // it would happily pass. Per-tensor check must enumerate the
    // expected set and notice.
    let k_name = "model.language_model.layers.1.self_attn.k_proj.weight".to_string();
    assert!(original.remove(&k_name).is_some());
    assert!(rotated.remove(&k_name).is_some());

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains(&k_name), "unexpected error: {msg}");
    assert!(msg.contains("missing"), "unexpected error: {msg}");
}

/// `embed_tokens` corruption: covered by BOTH the chain probe and
/// the per-tensor check. The chain probe runs first, so its message
/// is what we see — but either is a valid refuse.
#[test]
fn either_check_catches_corrupted_embed_tokens() {
    let cfg = tied_tiny_test_cfg();
    let original = build_working_set(&cfg, 27);
    let mut rotated = original.clone();

    materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
    let (fusion, rot_plan) = full_pipeline_plans(&cfg);
    fuse_rmsnorms(&mut rotated, &fusion).unwrap();
    let rotation = RandomizedHadamard::new(0xEDEDED, cfg.hidden_size).unwrap();
    absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

    let embed = rotated.get_mut(QWEN35_EMBED_TOKENS_NAME).unwrap();
    for v in embed.data.iter_mut() {
        *v *= 1.5;
    }

    let err = assert_forward_equivalence_qwen35(
        &original,
        &rotated,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig::default(),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("forward-equivalence refused"),
        "unexpected error: {msg}"
    );
    assert!(
        msg.contains("chain probe") || msg.contains("per-tensor"),
        "expected refuse message to name the failing check: {msg}"
    );
}
