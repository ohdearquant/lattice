use super::*;
use crate::quarot::plan::{AbsorptionSide, RotationId};

fn layer_prefix(idx: usize) -> String {
    format!("decoder.block.{idx}")
}

fn target() -> RotationTarget {
    RotationTarget::new(3, 8, 16, vec![0, 2], true, layer_prefix).unwrap()
}

fn descriptor() -> OnlineArtifactDescriptor {
    OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![
            OnlineRotationSpec::attention_output_r3(&target(), 31, 4).unwrap(),
            OnlineRotationSpec::mlp_down_r4(&target(), 43, 8).unwrap(),
        ],
        asymmetric_tensor_names: vec![
            "decoder.block.0.self_attn.o_proj.weight".into(),
            "decoder.block.2.self_attn.o_proj.weight".into(),
            "decoder.block.0.mlp.down_proj.weight".into(),
            "decoder.block.1.mlp.down_proj.weight".into(),
            "decoder.block.2.mlp.down_proj.weight".into(),
        ],
    }
}

#[test]
fn custom_prefix_names_validate_and_verify_actual_affected_tensors() {
    let descriptor = descriptor();
    let target = target();
    let expected = vec![
        (
            RotationId::AttentionOutputR3,
            0,
            "decoder.block.0.self_attn.o_proj.weight".into(),
        ),
        (
            RotationId::AttentionOutputR3,
            2,
            "decoder.block.2.self_attn.o_proj.weight".into(),
        ),
        (
            RotationId::MlpDownR4,
            0,
            "decoder.block.0.mlp.down_proj.weight".into(),
        ),
        (
            RotationId::MlpDownR4,
            1,
            "decoder.block.1.mlp.down_proj.weight".into(),
        ),
        (
            RotationId::MlpDownR4,
            2,
            "decoder.block.2.mlp.down_proj.weight".into(),
        ),
    ];
    assert_eq!(descriptor.derive_expected_tensor_names(&target), expected);
    assert!(descriptor.validate_target(Some(&target)).is_ok());
    let transformed: Vec<&str> = descriptor
        .asymmetric_tensor_names
        .iter()
        .map(String::as_str)
        .collect();
    assert!(descriptor.verify_affected(&target, &transformed).is_ok());
    assert!(
        descriptor
            .verify_affected(&target, &transformed[..4])
            .is_err()
    );
    let mut extra = transformed;
    extra.push("decoder.block.1.self_attn.o_proj.weight");
    assert!(descriptor.verify_affected(&target, &extra).is_err());
}

#[test]
fn custom_prefix_descriptor_and_affected_verifier_refuse_wrong_namespace() {
    let mut descriptor = descriptor();
    let target = target();
    descriptor.asymmetric_tensor_names = vec![
        "model.language_model.layers.0.self_attn.o_proj.weight".into(),
        "model.language_model.layers.2.self_attn.o_proj.weight".into(),
        "model.language_model.layers.0.mlp.down_proj.weight".into(),
        "model.language_model.layers.1.mlp.down_proj.weight".into(),
        "model.language_model.layers.2.mlp.down_proj.weight".into(),
    ];
    assert!(descriptor.validate_target(Some(&target)).is_err());
    let transformed: Vec<&str> = descriptor
        .asymmetric_tensor_names
        .iter()
        .map(String::as_str)
        .collect();
    assert!(descriptor.verify_affected(&target, &transformed).is_err());
}

#[test]
fn descriptor_validation_requires_complete_target_full_attention_set() {
    let descriptor = OnlineArtifactDescriptor {
        version: ArtifactVersion::V1Online,
        online_rotations: vec![OnlineRotationSpec {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed: 31,
            block_size: 4,
            layer_scope: Some(vec![0]),
        }],
        asymmetric_tensor_names: vec!["decoder.block.0.self_attn.o_proj.weight".into()],
    };
    let err = descriptor.validate_target(Some(&target())).unwrap_err();
    assert_eq!(
        err.to_string(),
        "Inference error: OnlineRotationSpec::validate: R3 layer_scope [0] does not cover all of this config's full-attention layers [0, 2] — missing layer(s) [2]"
    );
}

#[test]
fn descriptor_validation_retains_target_absence_rules() {
    let descriptor = descriptor();
    assert!(descriptor.validate_target(None).is_err());
    let residual = OnlineArtifactDescriptor {
        version: ArtifactVersion::V0Residual,
        online_rotations: vec![],
        asymmetric_tensor_names: vec![],
    };
    assert!(residual.validate_target(None).is_ok());
}
