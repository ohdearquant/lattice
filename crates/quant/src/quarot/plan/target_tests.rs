use super::*;

fn layer_prefix(idx: usize) -> String {
    format!("decoder.block.{idx}")
}

fn target(dense_mlp: bool, full_attention_layers: Vec<usize>) -> RotationTarget {
    RotationTarget::new(5, 8, 16, full_attention_layers, dense_mlp, layer_prefix).unwrap()
}

#[test]
fn rotation_target_rejects_zero_layers() {
    assert!(RotationTarget::new(0, 8, 16, vec![], true, layer_prefix).is_err());
}

#[test]
fn rotation_target_rejects_out_of_range_layer() {
    assert!(RotationTarget::new(5, 8, 16, vec![5], true, layer_prefix).is_err());
}

#[test]
fn rotation_target_rejects_unsorted_layers() {
    assert!(RotationTarget::new(5, 8, 16, vec![4, 1], true, layer_prefix).is_err());
}

#[test]
fn rotation_target_rejects_duplicate_layers() {
    assert!(RotationTarget::new(5, 8, 16, vec![1, 1], true, layer_prefix).is_err());
}

#[test]
fn rotation_target_accepts_recurrent_layout_and_preserves_dimensions() {
    let target = target(true, vec![]);
    assert_eq!(target.num_hidden_layers(), 5);
    assert_eq!(target.num_attention_heads(), 8);
    assert_eq!(target.intermediate_size(), 16);
    assert!(target.full_attention_layers().is_empty());
    assert!(target.dense_mlp());
    assert!(!target.is_full_attention(0));
    assert!(!target.is_full_attention(5));
    assert_eq!(target.layer_tensor_prefix(2), "decoder.block.2");
}

#[test]
fn attention_output_r3_uses_exact_target_full_attention_layers() {
    let target = target(true, vec![1, 4]);
    let spec = OnlineRotationSpec::attention_output_r3(&target, 31, 4).unwrap();
    assert_eq!(
        spec,
        OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 31,
            block_size: 4,
            layer_scope: Some(vec![1, 4]),
        }
    );
    assert!(target.is_full_attention(1));
    assert!(target.is_full_attention(4));
    assert!(!target.is_full_attention(2));
    assert!(spec.validate_target(Some(&target)).is_ok());
    assert!(spec.validate_target(None).is_ok());
}

#[test]
fn attention_output_r3_refuses_recurrent_target() {
    let target = target(true, vec![]);
    let err = OnlineRotationSpec::attention_output_r3(&target, 31, 4).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::r3_full_attention: config resolved zero full-attention layers — refusing an empty-scope R3 artifact"
    );
}

#[test]
fn attention_output_r3_refuses_invalid_blocks_on_numeric_target() {
    let target = RotationTarget::new(5, 12, 24, vec![1, 4], true, layer_prefix).unwrap();
    for block_size in [0, 3] {
        assert_eq!(
            OnlineRotationSpec::attention_output_r3(&target, 31, block_size)
                .unwrap_err()
                .to_string(),
            format!(
                "Inference error: OnlineRotationSpec::r3_full_attention requires a power-of-two block_size, got {block_size}"
            )
        );
    }
    assert_eq!(
        OnlineRotationSpec::attention_output_r3(&target, 31, 8)
            .unwrap_err()
            .to_string(),
        "Inference error: OnlineRotationSpec::r3_full_attention: block_size 8 does not divide num_attention_heads 12 — R3's online factor is QuaRot Eq. 9's cross-head Hadamard (H_num_heads ⊗ I_head_dim), so block_size divides the head axis, not head_dim"
    );
}

#[test]
fn mlp_down_r4_preserves_dense_recipe_and_rejects_invalid_blocks() {
    let target = target(true, vec![1, 4]);
    let spec = OnlineRotationSpec::mlp_down_r4(&target, 43, 8).unwrap();
    assert_eq!(
        spec,
        OnlineRotationSpec {
            id: RotationId::MlpDownR4,
            side: AbsorptionSide::InputSide,
            seed: 43,
            block_size: 8,
            layer_scope: None,
        }
    );
    assert!(spec.validate_target(Some(&target)).is_ok());
    assert!(spec.validate_target(None).is_ok());
    let invalid_target = RotationTarget::new(5, 12, 24, vec![1, 4], true, layer_prefix).unwrap();
    for block_size in [0, 3] {
        assert_eq!(
            OnlineRotationSpec::mlp_down_r4(&invalid_target, 43, block_size)
                .unwrap_err()
                .to_string(),
            format!(
                "Inference error: OnlineRotationSpec::r4_dense_mlp requires a power-of-two block_size, got {block_size}"
            )
        );
    }
    assert_eq!(
        OnlineRotationSpec::mlp_down_r4(&invalid_target, 43, 16)
            .unwrap_err()
            .to_string(),
        "Inference error: OnlineRotationSpec::r4_dense_mlp: block_size 16 does not divide intermediate_size 24"
    );
}

#[test]
fn mlp_down_r4_refuses_non_dense_target_in_builder_and_validator() {
    let target = target(false, vec![1, 4]);
    assert!(!target.dense_mlp());
    assert!(OnlineRotationSpec::mlp_down_r4(&target, 43, 8).is_err());
    let spec = OnlineRotationSpec {
        id: RotationId::MlpDownR4,
        side: AbsorptionSide::InputSide,
        seed: 43,
        block_size: 8,
        layer_scope: None,
    };
    assert!(spec.validate_target(None).is_ok());
    assert!(spec.validate_target(Some(&target)).is_err());
}

#[test]
fn validate_target_refuses_missing_full_attention_layer() {
    let target = target(true, vec![1, 4]);
    let spec = OnlineRotationSpec {
        id: RotationId::AttentionOutputR3,
        side: AbsorptionSide::InputSide,
        seed: 31,
        block_size: 4,
        layer_scope: Some(vec![1]),
    };
    assert!(spec.validate_target(None).is_ok());
    assert_eq!(
        spec.validate_target(Some(&target)).unwrap_err().to_string(),
        "Inference error: OnlineRotationSpec::validate: R3 layer_scope [1] does not cover all of this config's full-attention layers [1, 4] — missing layer(s) [4]"
    );
}
