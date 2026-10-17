use super::*;

#[test]
fn qwen35_plan_covers_residual_stream_tensors() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    assert_eq!(plan.rule_count(), 14);
    let cases = [
        // GQA full-attention
        (
            "model.layers.0.self_attn.q_proj.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.5.self_attn.k_proj.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.5.self_attn.v_proj.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.5.self_attn.o_proj.weight",
            AbsorptionSide::OutputSide,
        ),
        // GDN linear-attention
        (
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.0.linear_attn.in_proj_z.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.0.linear_attn.in_proj_b.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.0.linear_attn.in_proj_a.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.0.linear_attn.out_proj.weight",
            AbsorptionSide::OutputSide,
        ),
        // MLP
        (
            "model.layers.23.mlp.gate_proj.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.23.mlp.up_proj.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "model.layers.23.mlp.down_proj.weight",
            AbsorptionSide::OutputSide,
        ),
        // Embed + lm_head (both input-side)
        (
            "model.language_model.embed_tokens.weight",
            AbsorptionSide::InputSide,
        ),
        ("lm_head.weight", AbsorptionSide::InputSide),
    ];
    for (name, expected_side) in cases {
        let tr = plan
            .for_tensor(name)
            .unwrap_or_else(|| panic!("plan missed tensor {name}"));
        assert_eq!(tr.side, expected_side, "wrong side for {name}");
        assert_eq!(tr.rotation_id, RotationId::ResidualStream);
    }
}

#[test]
fn qwen35_plan_misses_rmsnorm_and_scalar_weights() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let skipped = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "model.layers.0.linear_attn.norm.weight",
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.dt_bias",
        "model.layers.0.linear_attn.conv1d.weight",
    ];
    for name in skipped {
        assert!(
            plan.for_tensor(name).is_none(),
            "non-linear-layer tensor {name} should not match plan"
        );
    }
}

/// Required-rule list that step 3b's converter MUST find in any Qwen3.5
/// hybrid SafeTensors file. Suffixes match the loader's expectations at
/// `crates/inference/src/model/qwen35/loading.rs:16, 33`.
fn qwen35_required_residual_tensor_suffixes() -> &'static [&'static str] {
    &[
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.out_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "embed_tokens.weight",
    ]
}

fn synthetic_qwen35_tensor_names(include_lm_head: bool) -> Vec<String> {
    let mut names: Vec<String> = qwen35_required_residual_tensor_suffixes()
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if s.contains("embed_tokens") {
                format!("model.language_model.{s}")
            } else {
                format!("model.layers.{i}.{s}")
            }
        })
        .collect();
    if include_lm_head {
        names.push("lm_head.weight".to_string());
    }
    names
}

#[test]
fn validate_coverage_tied_embeddings_is_complete_without_lm_head() {
    // Qwen3.5 default: tie_word_embeddings=true, so lm_head.weight is
    // ABSENT from the SafeTensors file. The plan's lm_head rule is
    // Optional — its absence must NOT fail is_complete().
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let names = synthetic_qwen35_tensor_names(false);
    let report = plan.validate_coverage(names.iter().map(String::as_str));
    assert!(
        report.is_complete(),
        "tied-embeddings tensor list should yield complete coverage, got: {report:?}"
    );
    assert_eq!(
        report.unmatched_required_rules.len(),
        0,
        "no required rule should be missing: {:?}",
        report.unmatched_required_rules
    );
    assert_eq!(
        report.unmatched_optional_rules,
        vec!["lm_head.weight".to_string()],
        "lm_head should appear in unmatched_optional_rules"
    );
    assert_eq!(report.matched_tensors.len(), 13);
    assert!(report.ambiguous_tensors.is_empty());
}

#[test]
fn validate_coverage_untied_embeddings_matches_lm_head() {
    // tie_word_embeddings=false case: lm_head.weight is loaded.
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let names = synthetic_qwen35_tensor_names(true);
    let report = plan.validate_coverage(names.iter().map(String::as_str));
    assert!(
        report.is_complete(),
        "untied case should be complete: {report:?}"
    );
    assert!(report.unmatched_optional_rules.is_empty());
    assert_eq!(report.matched_tensors.len(), 14);
}

#[test]
fn validate_coverage_flags_unmatched_required_rule() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    // Drop o_proj — a Required rule
    let names: Vec<String> = synthetic_qwen35_tensor_names(false)
        .into_iter()
        .filter(|n| !n.ends_with("self_attn.o_proj.weight"))
        .collect();
    let report = plan.validate_coverage(names.iter().map(String::as_str));
    assert!(
        !report.is_complete(),
        "missing required rule should fail completeness"
    );
    assert_eq!(
        report.unmatched_required_rules,
        vec!["self_attn.o_proj.weight".to_string()]
    );
}

#[test]
fn validate_coverage_lists_unplanned_tensors() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let mut names = synthetic_qwen35_tensor_names(false);
    names.push("model.layers.0.input_layernorm.weight".to_string());
    names.push("model.mtp.head.weight".to_string());
    let report = plan.validate_coverage(names.iter().map(String::as_str));
    // input_layernorm + mtp are unplanned, but the required rules still all match
    assert!(
        report.is_complete(),
        "unplanned tensors should not break required-rule completeness"
    );
    assert_eq!(report.unplanned_tensors.len(), 2);
    assert!(
        report
            .unplanned_tensors
            .iter()
            .any(|n| n.ends_with("input_layernorm.weight"))
    );
}

#[test]
fn qwen35_plan_misses_moe_expert_weights() {
    // Documents the v1 deferral — MoE expert tensors are NOT in the v0 plan.
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let moe_names = [
        "model.layers.0.mlp.experts.gate_up_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
    ];
    for name in moe_names {
        assert!(
            plan.for_tensor(name).is_none(),
            "MoE expert tensor {name} should not match v0 plan"
        );
    }
}

#[test]
fn apply_tensor_rotation_skips_unplanned() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let hidden = 64;
    let r = RandomizedHadamard::new(7, hidden).unwrap();
    let mut weight = vec![1.0_f32; hidden * hidden];
    let weight_copy = weight.clone();

    let rotated = apply_tensor_rotation(
        "model.layers.0.linear_attn.in_proj.weight",
        &mut weight,
        hidden,
        hidden,
        &plan,
        &r,
    )
    .unwrap();
    assert!(!rotated, "unplanned tensor should report not rotated");
    assert_eq!(weight, weight_copy, "unplanned tensor must not be mutated");
}

#[test]
fn apply_tensor_rotation_mlp_layer_pair_rotates_output() {
    // Dense MLP layer pair: gate_proj (input-side) + down_proj (output-side).
    // Rotation R is on the residual stream. After absorption:
    //   gate_proj' · (R · x) = gate_proj · x   (intermediate unchanged)
    //   down_proj' · intermediate = R · (down_proj · intermediate) = R · y_original
    // So the rotated pipeline output equals R · y_original, NOT y_original —
    // because down_proj writes BACK to the rotated residual stream.
    let hidden = 64;
    let intermediate = 128;
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let r = RandomizedHadamard::new(0xC0FFEE, hidden).unwrap();

    let mut state = 1_u64;
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
    };
    let gate_proj: Vec<f32> = (0..intermediate * hidden).map(|_| rand()).collect();
    let down_proj: Vec<f32> = (0..hidden * intermediate).map(|_| rand()).collect();
    let x: Vec<f32> = (0..hidden).map(|_| rand()).collect();

    let matvec = |w: &[f32], r: usize, c: usize, v: &[f32]| -> Vec<f32> {
        (0..r)
            .map(|i| (0..c).map(|j| w[i * c + j] * v[j]).sum())
            .collect()
    };
    let intermediate_out = matvec(&gate_proj, intermediate, hidden, &x);
    let y_original = matvec(&down_proj, hidden, intermediate, &intermediate_out);
    let mut y_expected = y_original.clone();
    r.apply(&mut y_expected).unwrap();

    let mut gate_proj_abs = gate_proj.clone();
    let mut down_proj_abs = down_proj.clone();
    assert!(
        apply_tensor_rotation(
            "model.layers.0.mlp.gate_proj.weight",
            &mut gate_proj_abs,
            intermediate,
            hidden,
            &plan,
            &r,
        )
        .unwrap()
    );
    assert!(
        apply_tensor_rotation(
            "model.layers.0.mlp.down_proj.weight",
            &mut down_proj_abs,
            hidden,
            intermediate,
            &plan,
            &r,
        )
        .unwrap()
    );

    let mut x_rotated = x.clone();
    r.apply(&mut x_rotated).unwrap();
    let intermediate_rot = matvec(&gate_proj_abs, intermediate, hidden, &x_rotated);
    let y_rotated = matvec(&down_proj_abs, hidden, intermediate, &intermediate_rot);

    let max_abs_diff = y_expected
        .iter()
        .zip(y_rotated.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-3,
        "MLP pair output should equal R · y_original: max_abs_diff={max_abs_diff}"
    );
}

#[test]
fn absorption_for_module_input_side() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    for module in [
        "q_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
    ] {
        assert_eq!(
            plan.absorption_for_module(module),
            Some(AbsorptionSide::InputSide),
            "{module} should be InputSide"
        );
    }
}

#[test]
fn absorption_for_module_output_side() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    for module in ["o_proj", "down_proj", "out_proj"] {
        assert_eq!(
            plan.absorption_for_module(module),
            Some(AbsorptionSide::OutputSide),
            "{module} should be OutputSide"
        );
    }
}

#[test]
fn absorption_for_module_unknown() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    assert_eq!(plan.absorption_for_module("conv1d"), None);
    assert_eq!(plan.absorption_for_module("norm"), None);
}

#[test]
fn qwen3_plan_covers_residual_stream_tensors() {
    let plan = RotationPlan::qwen3_residual_stream_linear_layers();
    // 8 required + 1 optional = 9 rules (no GDN)
    assert_eq!(plan.rule_count(), 9);
    let cases = [
        (
            "layers.0.self_attn.q_proj.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "layers.5.self_attn.k_proj.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "layers.5.self_attn.v_proj.weight",
            AbsorptionSide::InputSide,
        ),
        (
            "layers.5.self_attn.o_proj.weight",
            AbsorptionSide::OutputSide,
        ),
        ("layers.27.mlp.gate_proj.weight", AbsorptionSide::InputSide),
        ("layers.27.mlp.up_proj.weight", AbsorptionSide::InputSide),
        ("layers.27.mlp.down_proj.weight", AbsorptionSide::OutputSide),
        ("embed_tokens.weight", AbsorptionSide::InputSide),
        ("lm_head.weight", AbsorptionSide::InputSide),
    ];
    for (name, expected_side) in cases {
        let tr = plan
            .for_tensor(name)
            .unwrap_or_else(|| panic!("qwen3 plan missed tensor {name}"));
        assert_eq!(tr.side, expected_side, "wrong side for {name}");
        assert_eq!(tr.rotation_id, RotationId::ResidualStream);
    }
}

#[test]
fn qwen3_plan_does_not_cover_gdn_tensors() {
    let plan = RotationPlan::qwen3_residual_stream_linear_layers();
    let gdn_names = [
        "layers.0.linear_attn.in_proj_qkv.weight",
        "layers.0.linear_attn.in_proj_z.weight",
        "layers.0.linear_attn.in_proj_b.weight",
        "layers.0.linear_attn.in_proj_a.weight",
        "layers.0.linear_attn.out_proj.weight",
    ];
    for name in gdn_names {
        assert!(
            plan.for_tensor(name).is_none(),
            "GDN tensor {name} should not match qwen3 plan"
        );
    }
}

#[test]
fn qwen3_0_6b_required_tensor_names_count_and_spot_check() {
    let cfg = QwenConfig::qwen3_embedding_0_6b();
    let names = qwen3_required_tensor_names(&cfg);
    // 2 global + 28 layers × 11 per-layer = 310
    assert_eq!(
        names.len(),
        310,
        "expected 310 names for Qwen3-0.6B (28 layers × 11 + 2 global)"
    );
    assert!(names.contains(&"embed_tokens.weight".to_string()));
    assert!(names.contains(&"norm.weight".to_string()));
    assert!(names.contains(&"layers.0.self_attn.q_proj.weight".to_string()));
    assert!(names.contains(&"layers.27.mlp.down_proj.weight".to_string()));
    assert!(names.contains(&"layers.0.self_attn.q_norm.weight".to_string()));
    assert!(names.contains(&"layers.0.self_attn.k_norm.weight".to_string()));
    assert!(
        !names.iter().any(|n| n.contains("linear_attn")),
        "Qwen3 required names must not include GDN linear_attn tensors"
    );
}

#[test]
fn qwen3_0_6b_validate_coverage_complete_without_lm_head() {
    let plan = RotationPlan::qwen3_residual_stream_linear_layers();
    let cfg = QwenConfig::qwen3_embedding_0_6b();
    let names = qwen3_required_tensor_names(&cfg);
    let report = plan.validate_coverage(names.iter().map(String::as_str));
    assert!(
        report.is_complete(),
        "Qwen3-0.6B required names yield complete coverage: {report:?}"
    );
    assert_eq!(report.unmatched_required_rules.len(), 0);
    assert_eq!(
        report.unmatched_optional_rules,
        vec!["lm_head.weight".to_string()],
        "lm_head.weight should be the only unmatched-optional rule (tied embeddings)"
    );
}

#[test]
fn f64_apply_matches_f32_apply_pattern() {
    let plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let hidden = 64;
    let r = RandomizedHadamard::new(11, hidden).unwrap();
    let mut weight_f32: Vec<f32> = (0..hidden * hidden)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let mut weight_f64: Vec<f64> = weight_f32.iter().map(|&v| f64::from(v)).collect();

    let r1 = apply_tensor_rotation(
        "model.layers.0.self_attn.q_proj.weight",
        &mut weight_f32,
        hidden,
        hidden,
        &plan,
        &r,
    )
    .unwrap();
    let r2 = apply_tensor_rotation_f64(
        "model.layers.0.self_attn.q_proj.weight",
        &mut weight_f64,
        hidden,
        hidden,
        &plan,
        &r,
    )
    .unwrap();
    assert!(r1 && r2);

    for (i, (a, b)) in weight_f32.iter().zip(weight_f64.iter()).enumerate() {
        let delta = (f64::from(*a) - b).abs();
        assert!(
            delta < 1e-5,
            "tensor[{i}]: f32={a} vs f64={b}, delta={delta}"
        );
    }
}

// --- OnlineRotationSpec (R3/R4 contract, issue #703 PR1) ---

#[test]
fn r3_full_attention_scope_matches_qwen35_0_8b_layer_pattern() {
    let cfg = Qwen35Config::qwen35_0_8b();
    // block_size == num_attention_heads (8): one dense cross-head
    // Hadamard covering every head, matching the paper's H_num_heads.
    let spec = OnlineRotationSpec::r3_full_attention(&cfg, 7, 8).unwrap();
    assert_eq!(spec.id, RotationId::AttentionOutputR3);
    assert_eq!(spec.side, AbsorptionSide::InputSide);
    assert_eq!(spec.block_size, 8);
    // interval=4 over 24 layers -> layers 3,7,11,15,19,23 (0-indexed)
    assert_eq!(
        spec.layer_scope,
        Some(vec![3usize, 7, 11, 15, 19, 23]),
        "R3 scope must be exactly the config's full-attention layers"
    );
    assert_eq!(spec.layer_scope.as_ref().unwrap().len(), 6);
    for &idx in spec.layer_scope.as_ref().unwrap() {
        assert!(
            cfg.is_full_attention(idx),
            "layer {idx} must be full-attention"
        );
    }
    for idx in 0..cfg.num_hidden_layers {
        if !spec.layer_scope.as_ref().unwrap().contains(&idx) {
            assert!(
                !cfg.is_full_attention(idx),
                "layer {idx} is full-attention but missing from R3 scope"
            );
        }
    }
}

#[test]
fn r3_rejects_block_size_not_dividing_num_attention_heads() {
    let cfg = Qwen35Config::qwen35_0_8b();
    // num_attention_heads = 8; 16 does not divide it.
    assert!(OnlineRotationSpec::r3_full_attention(&cfg, 1, 16).is_err());
}

#[test]
fn r3_accepts_block_size_grouping_non_power_of_two_head_count() {
    // Qwen3.6-27B has num_attention_heads = 24, not itself a power of
    // two. block_size = 8 (a power-of-two divisor of 24) partitions the
    // 24 heads into 3 independently-rotated cross-head blocks — the
    // documented BlockHadamard fallback, not a randomized-orthogonal
    // matrix, for a non-power-of-two head count.
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen36_27b();
    assert_eq!(cfg.num_attention_heads, 24);
    let spec = OnlineRotationSpec::r3_full_attention(&cfg, 1, 8).unwrap();
    assert_eq!(spec.block_size, 8);
    // 24 itself is not a power of two, so the "one dense H_num_heads"
    // form from the paper is unavailable here.
    assert!(!cfg.num_attention_heads.is_power_of_two());
    assert!(OnlineRotationSpec::r3_full_attention(&cfg, 1, 24).is_err());
}

#[test]
fn r3_rejects_non_power_of_two_block_size() {
    let cfg = Qwen35Config::qwen35_0_8b();
    assert!(OnlineRotationSpec::r3_full_attention(&cfg, 1, 96).is_err());
}

#[test]
fn r3_rejects_zero_block_size() {
    let cfg = Qwen35Config::qwen35_0_8b();
    assert!(OnlineRotationSpec::r3_full_attention(&cfg, 1, 0).is_err());
}

#[test]
fn r4_dense_mlp_has_no_layer_scope_restriction() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let spec = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 256).unwrap();
    assert_eq!(spec.id, RotationId::MlpDownR4);
    assert_eq!(spec.side, AbsorptionSide::InputSide);
    assert_eq!(spec.layer_scope, None);
}

/// A hand-built spec with a `layer_scope` far larger than any real R3
/// recipe (which scopes to a config's full-attention layer count, tens
/// of entries) must be rejected before the O(n) per-layer membership
/// check runs, so an untrusted descriptor cannot force unbounded
/// validation work by inflating `layer_scope` alone.
#[test]
fn r3_rejects_oversized_layer_scope() {
    let oversized: Vec<usize> = (0..(OnlineRotationSpec::MAX_LAYER_SCOPE_ENTRIES + 1)).collect();
    let hand_built = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 1,
        block_size: 1,
        layer_scope: Some(oversized),
    };
    let err = hand_built.validate(None).unwrap_err();
    assert!(
        format!("{err}").contains(&OnlineRotationSpec::MAX_LAYER_SCOPE_ENTRIES.to_string()),
        "expected the layer_scope cap error naming the maximum, got: {err}"
    );
}

/// `qwen36_35b_a3b()` is MoE (loader requires `mlp.experts.down_proj`,
/// not the dense `mlp.down_proj.weight` this constructor targets), so
/// both the constructor and the independent `validate` invariant check
/// must refuse it rather than certify a fabricated dense per-layer name.
#[test]
fn r4_dense_mlp_rejects_moe_config() {
    let moe_cfg = Qwen35Config::qwen36_35b_a3b();
    assert!(moe_cfg.is_moe(), "sanity: qwen36_35b_a3b is MoE");
    let err = OnlineRotationSpec::r4_dense_mlp(&moe_cfg, 9, 256).unwrap_err();
    assert!(format!("{err}").contains("MoE"), "got: {err}");

    // A hand-built R4 spec (bypassing the constructor) must also be
    // rejected by `validate` when a MoE cfg is supplied — the
    // constructor and the invariant check are independent gates.
    let hand_built = OnlineRotationSpec {
        id: RotationId::MlpDownR4,
        side: AbsorptionSide::InputSide,
        seed: 9,
        block_size: 256,
        layer_scope: None,
    };
    let err = hand_built.validate(Some(&moe_cfg)).unwrap_err();
    assert!(format!("{err}").contains("MoE"), "got: {err}");

    // The dense 0.8B config is unaffected — still passes both paths.
    let dense_cfg = Qwen35Config::qwen35_0_8b();
    assert!(!dense_cfg.is_moe(), "sanity: qwen35_0_8b is dense");
    assert!(OnlineRotationSpec::r4_dense_mlp(&dense_cfg, 9, 256).is_ok());
    assert!(hand_built.validate(Some(&dense_cfg)).is_ok());
}

#[test]
fn r4_rejects_block_size_not_dividing_intermediate_size() {
    let cfg = Qwen35Config::qwen35_0_8b();
    // intermediate_size = 3584 = 2^9 * 7. 1024 is a power of two but
    // does not divide 3584 (3584 / 1024 = 3.5), so this exercises the
    // divisibility check on its own — a non-power-of-two block_size
    // like 100 would be rejected earlier by the power-of-two check
    // regardless of divisibility, leaving this test green even if the
    // divisibility check were removed.
    assert_eq!(
        cfg.intermediate_size % 1024,
        512,
        "sanity: 1024 does not divide 3584"
    );
    assert!(OnlineRotationSpec::r4_dense_mlp(&cfg, 1, 1024).is_err());
}

/// Qwen3.6-27B is dense (`intermediate_size = 17408`), and 4 is a
/// power-of-two divisor of it, so `block_size = 4` clears every
/// divisibility check — but `17408 / 4 = 4,352` blocks exceeds
/// `BlockHadamard`'s `MAX_BLOCK_HADAMARD_BLOCKS` cap of 4,096, so
/// `BlockHadamard::new` refuses to construct this recipe. Both the
/// constructor and the independent `validate` gate (for a hand-built
/// spec bypassing the constructor) must reject this at certification
/// time, not merely at construction time.
#[test]
fn r4_dense_mlp_rejects_num_blocks_above_block_hadamard_cap() {
    let cfg = Qwen35Config::qwen36_27b();
    assert!(!cfg.is_moe(), "sanity: qwen36_27b is dense");
    assert_eq!(cfg.intermediate_size, 17408);
    assert_eq!(
        cfg.intermediate_size / 4,
        4352,
        "sanity: 17408 / 4 = 4352 blocks"
    );

    let err = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 4).unwrap_err();
    assert!(
        format!("{err}").contains("4352") && format!("{err}").contains("4096"),
        "expected the num_blocks cap error naming both the block count \
             and the cap, got: {err}"
    );

    // A hand-built spec bypassing the constructor must be rejected by
    // `validate` too — the mutation-sensitive gate: removing this check
    // from `validate` while leaving the constructor's check in place
    // must make this half of the assertion fail.
    let hand_built = OnlineRotationSpec {
        id: RotationId::MlpDownR4,
        side: AbsorptionSide::InputSide,
        seed: 9,
        block_size: 4,
        layer_scope: None,
    };
    let err = hand_built.validate(Some(&cfg)).unwrap_err();
    assert!(
        format!("{err}").contains("4352") && format!("{err}").contains("4096"),
        "expected validate() to independently reject the hand-built \
             spec with the num_blocks cap error, got: {err}"
    );

    // BlockHadamard itself must actually refuse to construct this
    // geometry, confirming the cap check reflects a real limitation
    // rather than an overly conservative guess.
    assert!(super::super::hadamard::BlockHadamard::new(9, cfg.intermediate_size, 4).is_err());
}

/// The same 27B geometry with a realistic `block_size = 128` stays
/// comfortably under the cap (136 blocks) — both certification gates
/// accept it, and `BlockHadamard::new` actually constructs it.
#[test]
fn r4_dense_mlp_accepts_block_size_within_num_blocks_cap() {
    let cfg = Qwen35Config::qwen36_27b();
    assert_eq!(
        cfg.intermediate_size / 128,
        136,
        "sanity: 17408 / 128 = 136 blocks"
    );

    let spec = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 128).unwrap();
    assert!(spec.validate(Some(&cfg)).is_ok());

    let bh = super::super::hadamard::BlockHadamard::new(9, cfg.intermediate_size, 128)
        .expect("136 blocks must be constructible — well under the 4096 cap");
    assert_eq!(bh.num_blocks(), 136);
}

/// `intermediate_size =
/// 33_554_432, block_size = 8192` gives exactly 4,096 blocks — AT the
/// `MAX_BLOCK_HADAMARD_BLOCKS` cap, so the block-count check alone
/// passes it — but `intermediate_size` itself (2^25) exceeds
/// `MAX_BLOCK_HADAMARD_LEN` (2^24), a bound `BlockHadamard::new`
/// enforces independently of block count. Before this fix the plan
/// gate certified this recipe while `BlockHadamard::new` refused to
/// construct it — validation and runtime disagreeing about what is
/// materializable.
#[test]
fn r4_dense_mlp_rejects_dim_over_max_block_hadamard_len_even_within_block_cap() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.intermediate_size = 33_554_432;
    assert_eq!(
        cfg.intermediate_size / 8192,
        4096,
        "sanity: exactly at MAX_BLOCK_HADAMARD_BLOCKS, not over it"
    );

    let err = OnlineRotationSpec::r4_dense_mlp(&cfg, 9, 8192)
        .expect_err("dim over MAX_BLOCK_HADAMARD_LEN must be rejected by the plan gate");
    assert!(
        format!("{err}").contains("MAX_BLOCK_HADAMARD_LEN"),
        "expected the plan gate to reject on the length bound, not the \
             block-count bound (which this pair sits exactly at), got: {err}"
    );

    // Runtime agrees: BlockHadamard::new independently refuses this
    // dimension too, confirming the plan gate now matches it.
    assert!(super::super::hadamard::BlockHadamard::new(9, cfg.intermediate_size, 8192).is_err());

    // A legitimate dimension at the same block_size still passes both
    // gates.
    let small_cfg = Qwen35Config::qwen35_0_8b();
    assert!(OnlineRotationSpec::r4_dense_mlp(&small_cfg, 9, 128).is_ok());
}

#[test]
fn r4_accepts_all_named_design_doc_block_sizes() {
    let cfg = Qwen35Config::qwen35_0_8b();
    for &b in &[64usize, 128, 256] {
        assert_eq!(
            cfg.intermediate_size % b,
            0,
            "block size {b} must divide 3584"
        );
        assert!(OnlineRotationSpec::r4_dense_mlp(&cfg, 1, b).is_ok());
    }
}

#[test]
fn rotation_id_online_transform_site_pins_every_variant() {
    assert_eq!(RotationId::ResidualStream.online_transform_site(), None);
    assert_eq!(
        RotationId::AttentionOutputR3.online_transform_site(),
        Some(OnlineTransformSite::AttentionOutputPreOProj)
    );
    assert_eq!(
        RotationId::MlpDownR4.online_transform_site(),
        Some(OnlineTransformSite::MlpPreDownProj)
    );
    assert_eq!(
        OnlineTransformSite::AttentionOutputPreOProj.weight_tensor_suffix(),
        "self_attn.o_proj.weight"
    );
    assert_eq!(
        OnlineTransformSite::MlpPreDownProj.weight_tensor_suffix(),
        "mlp.down_proj.weight"
    );
}

#[test]
fn apply_tensor_rotation_refuses_r3_rotation_id() {
    // Mutation/refusal: a plan rule that (incorrectly, for this PR)
    // referenced RotationId::AttentionOutputR3 must be refused loudly
    // by apply_tensor_rotation rather than silently treated as a
    // residual-stream rotation or a no-op.
    let plan = RotationPlan {
        rules: vec![Rule {
            pattern: "self_attn.o_proj.weight".to_string(),
            rotation: TensorRotation {
                side: AbsorptionSide::InputSide,
                rotation_id: RotationId::AttentionOutputR3,
            },
            requirement: RuleRequirement::Required,
        }],
    };
    let hidden = 64;
    let r = RandomizedHadamard::new(1, hidden).unwrap();
    let mut weight = vec![1.0_f32; hidden * hidden];
    let err = apply_tensor_rotation(
        "model.layers.0.self_attn.o_proj.weight",
        &mut weight,
        hidden,
        hidden,
        &plan,
        &r,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("contract-only"),
        "expected a contract-only refusal, got: {msg}"
    );
}
