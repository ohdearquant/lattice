use super::*;
use crate::quarot::io::{ArtifactVersion, OnlineArtifactDescriptor};

#[test]
fn qwen35_target_preserves_dense_preset_layout() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let target = qwen35_rotation_target(&cfg);
    assert_eq!(target.num_hidden_layers(), 24);
    assert_eq!(target.num_attention_heads(), 8);
    assert_eq!(target.intermediate_size(), 3584);
    assert_eq!(target.full_attention_layers(), &[3, 7, 11, 15, 19, 23]);
    assert!(target.dense_mlp());
    for idx in 0..24 {
        assert_eq!(
            target.is_full_attention(idx),
            [3, 7, 11, 15, 19, 23].contains(&idx)
        );
        assert_eq!(
            target.layer_tensor_prefix(idx),
            format!("model.language_model.layers.{idx}")
        );
    }
}

#[test]
fn qwen35_target_preserves_expert_preset_layout() {
    let cfg = Qwen35Config::qwen36_35b_a3b();
    let target = qwen35_rotation_target(&cfg);
    assert_eq!(target.num_hidden_layers(), 40);
    assert_eq!(target.num_attention_heads(), 16);
    assert_eq!(target.intermediate_size(), 6144);
    assert_eq!(
        target.full_attention_layers(),
        &[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    );
    assert!(!target.dense_mlp());
    for idx in 0..40 {
        assert_eq!(
            target.is_full_attention(idx),
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39].contains(&idx)
        );
        assert_eq!(
            target.layer_tensor_prefix(idx),
            format!("model.language_model.layers.{idx}")
        );
    }
}

fn outcome<T>(result: Result<T, InferenceError>) -> Result<T, String> {
    result.map_err(|err| err.to_string())
}

#[allow(deprecated)]
fn assert_legacy_equivalence(cfg: &Qwen35Config) {
    let target = crate::quarot::plan::qwen35_rotation_target(cfg);
    for block_size in [0, 3, 4, 8, 256] {
        assert_eq!(
            outcome(OnlineRotationSpec::r3_full_attention(cfg, 17, block_size)),
            outcome(
                crate::quarot::plan::OnlineRotationSpec::attention_output_r3(
                    &target, 17, block_size
                )
            )
        );
        assert_eq!(
            outcome(OnlineRotationSpec::r4_dense_mlp(cfg, 29, block_size)),
            outcome(crate::quarot::plan::OnlineRotationSpec::mlp_down_r4(
                &target, 29, block_size
            ))
        );
    }
    for spec in [
        OnlineRotationSpec::attention_output_r3(&target, 17, 4).unwrap(),
        OnlineRotationSpec {
            id: RotationId::MlpDownR4,
            side: AbsorptionSide::InputSide,
            seed: 29,
            block_size: 256,
            layer_scope: None,
        },
        OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 17,
            block_size: 4,
            layer_scope: Some(vec![3]),
        },
    ] {
        assert_eq!(
            outcome(spec.validate(Some(cfg))),
            outcome(spec.validate_target(Some(&target)))
        );
        assert_eq!(
            outcome(spec.validate(None)),
            outcome(spec.validate_target(None))
        );
        let suffix = spec
            .id
            .online_transform_site()
            .unwrap()
            .weight_tensor_suffix();
        let layers = spec
            .layer_scope
            .clone()
            .unwrap_or_else(|| (0..cfg.num_hidden_layers).collect());
        let descriptor = OnlineArtifactDescriptor {
            version: ArtifactVersion::V1Online,
            online_rotations: vec![spec],
            asymmetric_tensor_names: layers
                .iter()
                .map(|idx| format!("model.language_model.layers.{idx}.{suffix}"))
                .collect(),
        };
        assert_eq!(
            outcome(descriptor.validate(Some(cfg))),
            outcome(descriptor.validate_target(Some(&target)))
        );
        assert_eq!(
            outcome(descriptor.validate(None)),
            outcome(descriptor.validate_target(None))
        );
    }
    let legacy = RotationPlan::qwen35_residual_stream_linear_layers();
    let replacement = crate::quarot::plan::qwen35_residual_stream_linear_layers();
    assert_eq!(legacy.rules.len(), replacement.rules.len());
    for (old, new) in legacy.rules.iter().zip(&replacement.rules) {
        assert_eq!(old.pattern, new.pattern);
        assert_eq!(old.rotation, new.rotation);
        assert_eq!(old.requirement, new.requirement);
    }
}

#[test]
fn legacy_rotation_methods_match_neutral_replacements_for_dense_preset() {
    assert_legacy_equivalence(&Qwen35Config::qwen35_0_8b());
}

#[test]
fn legacy_rotation_methods_match_neutral_replacements_for_expert_preset() {
    assert_legacy_equivalence(&Qwen35Config::qwen36_35b_a3b());
}

#[test]
#[allow(deprecated)]
fn legacy_rotation_methods_preserve_raw_invalid_config_errors() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 0;
    let target = qwen35_rotation_target(&cfg);
    assert_eq!(
        OnlineRotationSpec::r3_full_attention(&cfg, 17, 4)
            .unwrap_err()
            .to_string(),
        "Inference error: OnlineRotationSpec::r3_full_attention: config resolved zero full-attention layers — refusing an empty-scope R3 artifact"
    );
    assert_eq!(
        outcome(OnlineRotationSpec::r3_full_attention(&cfg, 17, 4)),
        outcome(OnlineRotationSpec::attention_output_r3(&target, 17, 4))
    );
    cfg.num_attention_heads = 0;
    assert_eq!(
        OnlineRotationSpec::r3_full_attention(&cfg, 17, 4)
            .unwrap_err()
            .to_string(),
        "Inference error: OnlineRotationSpec: num_attention_heads must be non-zero — BlockHadamard::new refuses a zero-length rotation axis"
    );
    cfg.intermediate_size = 0;
    assert_eq!(
        OnlineRotationSpec::r4_dense_mlp(&cfg, 29, 4)
            .unwrap_err()
            .to_string(),
        "Inference error: OnlineRotationSpec: intermediate_size must be non-zero — BlockHadamard::new refuses a zero-length rotation axis"
    );
}

fn without_full_target<T>(operation: impl FnOnce() -> T) -> T {
    let before = full_target_factory_calls();
    let result = operation();
    assert_eq!(full_target_factory_calls(), before);
    result
}

#[test]
#[allow(deprecated)]
fn legacy_r3_scalar_refusals_precede_full_target_construction() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 64;
    cfg.num_attention_heads = 12;
    for block_size in [0, 3] {
        let err =
            without_full_target(|| OnlineRotationSpec::r3_full_attention(&cfg, 17, block_size))
                .unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "Inference error: OnlineRotationSpec::r3_full_attention requires a power-of-two block_size, got {block_size}"
            )
        );
    }
    let err =
        without_full_target(|| OnlineRotationSpec::r3_full_attention(&cfg, 17, 8)).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::r3_full_attention: block_size 8 does not divide num_attention_heads 12 — R3's online factor is QuaRot Eq. 9's cross-head Hadamard (H_num_heads ⊗ I_head_dim), so block_size divides the head axis, not head_dim"
    );
    cfg.num_attention_heads = 0;
    let err =
        without_full_target(|| OnlineRotationSpec::r3_full_attention(&cfg, 17, 4)).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec: num_attention_heads must be non-zero — BlockHadamard::new refuses a zero-length rotation axis"
    );
}

#[test]
#[allow(deprecated)]
fn legacy_r4_builder_and_validator_never_construct_full_target() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.layer_types.clear();
    cfg.intermediate_size = 24;
    for num_hidden_layers in [0, 64] {
        cfg.num_hidden_layers = num_hidden_layers;
        let spec = without_full_target(|| OnlineRotationSpec::r4_dense_mlp(&cfg, 29, 8)).unwrap();
        assert_eq!(
            spec,
            OnlineRotationSpec {
                id: RotationId::MlpDownR4,
                side: AbsorptionSide::InputSide,
                seed: 29,
                block_size: 8,
                layer_scope: None
            }
        );
        assert!(without_full_target(|| spec.validate(Some(&cfg))).is_ok());
        for block_size in [0, 3] {
            let err =
                without_full_target(|| OnlineRotationSpec::r4_dense_mlp(&cfg, 29, block_size))
                    .unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "Inference error: OnlineRotationSpec::r4_dense_mlp requires a power-of-two block_size, got {block_size}"
                )
            );
        }
    }
    let moe = Qwen35Config::qwen36_35b_a3b();
    let err = without_full_target(|| OnlineRotationSpec::r4_dense_mlp(&moe, 29, 0)).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::r4_dense_mlp: this config is a MoE configuration (loader requires mlp.experts.down_proj, not the dense mlp.down_proj.weight this constructor targets) — MoE R4 targets are not yet modeled"
    );
}

fn raw_r3_spec() -> OnlineRotationSpec {
    OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 17,
        block_size: 4,
        layer_scope: Some(vec![3]),
    }
}

#[test]
#[allow(deprecated)]
fn legacy_spec_structural_refusals_precede_full_target_construction() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 64;
    let mut spec = raw_r3_spec();
    spec.side = AbsorptionSide::OutputSide;
    spec.block_size = 3;
    spec.layer_scope = None;
    let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::validate: AttentionOutputR3 requires side=InputSide (the only orientation proven correct by the R3/R4 reference test), got OutputSide"
    );
    spec.side = AbsorptionSide::InputSide;
    let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::validate: AttentionOutputR3 requires a power-of-two block_size, got 3"
    );
    spec.block_size = 4;
    let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::validate: AttentionOutputR3 (R3) requires an explicit non-empty layer_scope (full-attention layers only) — layer_scope=None is invalid for R3"
    );
    spec.layer_scope = Some(vec![]);
    let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::validate: AttentionOutputR3 (R3) layer_scope must not be empty"
    );
    spec.layer_scope = Some((0..=OnlineRotationSpec::MAX_LAYER_SCOPE_ENTRIES).collect());
    let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        format!(
            "Inference error: OnlineRotationSpec::validate: R3 layer_scope declares {} entries, exceeding the maximum of {} — a spec this large is rejected before the per-layer membership check to keep validation cost bounded regardless of input",
            OnlineRotationSpec::MAX_LAYER_SCOPE_ENTRIES + 1,
            OnlineRotationSpec::MAX_LAYER_SCOPE_ENTRIES
        )
    );
    for layers in [vec![3, 3], vec![7, 3]] {
        spec.layer_scope = Some(layers.clone());
        let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "Inference error: OnlineRotationSpec::validate: R3 layer_scope {layers:?} must be strictly sorted ascending with no duplicates — found {} at or after {}",
                layers[1], layers[0]
            )
        );
    }
}

#[test]
#[allow(deprecated)]
fn legacy_spec_dimension_and_membership_refusals_precede_full_target_construction() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 64;
    cfg.num_attention_heads = 12;
    let mut spec = raw_r3_spec();
    spec.block_size = 8;
    let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::validate: R3 block_size 8 does not divide cfg.num_attention_heads 12"
    );
    cfg.num_attention_heads = 0;
    spec.block_size = 4;
    let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec: num_attention_heads must be non-zero — BlockHadamard::new refuses a zero-length rotation axis"
    );
    cfg.num_attention_heads = 8;
    for idx in [0, 64] {
        spec.layer_scope = Some(vec![idx]);
        let err = without_full_target(|| spec.validate(Some(&cfg))).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "Inference error: OnlineRotationSpec::validate: R3 layer_scope includes layer {idx}, which is not a valid full-attention layer for this config"
            )
        );
    }
}

#[test]
#[allow(deprecated)]
fn legacy_r3_materializes_full_target_after_valid_membership() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let before = full_target_factory_calls();
    let spec = OnlineRotationSpec::r3_full_attention(&cfg, 17, 4).unwrap();
    assert_eq!(full_target_factory_calls(), before + 1);
    assert_eq!(spec.layer_scope, Some(vec![3, 7, 11, 15, 19, 23]));
    assert!(spec.validate(Some(&cfg)).is_ok());
    assert_eq!(full_target_factory_calls(), before + 2);
    let partial = raw_r3_spec();
    let err = partial.validate(Some(&cfg)).unwrap_err();
    assert_eq!(full_target_factory_calls(), before + 3);
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::validate: R3 layer_scope [3] does not cover all of this config's full-attention layers [3, 7, 11, 15, 19, 23] — missing layer(s) [7, 11, 15, 19, 23]"
    );
}

#[test]
#[allow(deprecated)]
fn legacy_residual_descriptor_ignores_configuration_without_full_target_construction() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 64;
    cfg.layer_types.clear();
    let mut descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V0Residual,
        online_rotations: vec![],
        asymmetric_tensor_names: vec![],
    };
    assert!(without_full_target(|| descriptor.validate(Some(&cfg))).is_ok());
    let mut spec = raw_r3_spec();
    spec.side = AbsorptionSide::OutputSide;
    descriptor.online_rotations.push(spec);
    let err = without_full_target(|| descriptor.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineArtifactDescriptor: a V0Residual artifact must not carry online rotation metadata — a v0 residual artifact is never interpreted as online-capable"
    );
    descriptor.online_rotations.clear();
    descriptor
        .asymmetric_tensor_names
        .push("decoder.block.0.mlp.down_proj.weight".into());
    let err = without_full_target(|| descriptor.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineArtifactDescriptor: a V0Residual artifact must not declare any asymmetric_tensor_names — the V0 contract is symmetric-only Q4 with no R3/R4 metadata"
    );
}

#[test]
#[allow(deprecated)]
fn legacy_online_descriptor_preserves_spec_order_and_scalar_only_layout() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 3;
    cfg.layer_types.clear();
    cfg.intermediate_size = 24;
    let mut descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![],
        asymmetric_tensor_names: vec![],
    };
    let err = without_full_target(|| descriptor.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineArtifactDescriptor: a V1Online artifact must declare at least one R3/R4 online rotation"
    );
    let mut bad_spec = raw_r3_spec();
    bad_spec.side = AbsorptionSide::OutputSide;
    descriptor.online_rotations.push(bad_spec);
    let err = without_full_target(|| descriptor.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::validate: AttentionOutputR3 requires side=InputSide (the only orientation proven correct by the R3/R4 reference test), got OutputSide"
    );
    descriptor.online_rotations = vec![OnlineRotationSpec {
        id: RotationId::MlpDownR4,
        side: AbsorptionSide::InputSide,
        seed: 29,
        block_size: 8,
        layer_scope: None,
    }];
    descriptor.asymmetric_tensor_names = vec![
        "model.language_model.layers.0.mlp.down_proj.weight".into(),
        "model.language_model.layers.1.mlp.down_proj.weight".into(),
        "model.language_model.layers.2.mlp.down_proj.weight".into(),
    ];
    assert!(without_full_target(|| descriptor.validate(Some(&cfg))).is_ok());
}

#[test]
#[allow(deprecated)]
fn legacy_descriptor_entry_caps_precede_spec_or_layout_work() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 64;
    let mut spec = raw_r3_spec();
    spec.side = AbsorptionSide::OutputSide;
    let mut descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![spec.clone(); 4097],
        asymmetric_tensor_names: vec![],
    };
    let err = without_full_target(|| descriptor.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineArtifactDescriptor: online_rotations declares 4097 entries, exceeding the maximum of 4096 — a descriptor this large is rejected before validation to keep work bounded over untrusted input"
    );
    descriptor.online_rotations = vec![spec];
    descriptor.asymmetric_tensor_names = vec![String::new(); 4097];
    let err = without_full_target(|| descriptor.validate(Some(&cfg))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineArtifactDescriptor: asymmetric_tensor_names declares 4097 entries, exceeding the maximum of 4096 — a descriptor this large is rejected before validation to keep work bounded over untrusted input"
    );
}
