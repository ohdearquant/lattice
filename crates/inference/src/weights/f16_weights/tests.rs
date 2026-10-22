use super::*;

#[test]
fn test_f16_roundtrip_exact_representable() {
    for v in [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 65504.0] {
        let h = F16::from_f32(v);
        assert_eq!(h.to_f32(), v, "exact roundtrip failed for {v}");
    }
}

#[test]
fn test_f16_special_values() {
    assert_eq!(F16::from_f32(f32::INFINITY).to_f32(), f32::INFINITY);
    assert_eq!(F16::from_f32(f32::NEG_INFINITY).to_f32(), f32::NEG_INFINITY);
    assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
    assert_eq!(F16::from_f32(100000.0).to_f32(), f32::INFINITY);
}

#[test]
fn test_f16_denormals() {
    let smallest_denorm = F16(0x0001);
    let v = smallest_denorm.to_f32();
    assert!((v - 5.960464e-8).abs() < 1e-12);
}

#[test]
fn test_f16_signed_zero() {
    let pz = F16::from_f32(0.0);
    let nz = F16::from_f32(-0.0);
    assert_eq!(pz.0, 0x0000);
    assert_eq!(nz.0, 0x8000);
    assert_eq!(pz.to_f32().to_bits(), 0.0f32.to_bits());
    assert_eq!(nz.to_f32().to_bits(), (-0.0f32).to_bits());
}

#[test]
fn test_f16_round_to_nearest_even() {
    let a = F16(0x3c00).to_f32();
    let b = F16(0x3c01).to_f32();
    let half = (a + b) * 0.5;
    assert_eq!(F16::from_f32(half).0, 0x3c00);

    let c = F16(0x3c01).to_f32();
    let d = F16(0x3c02).to_f32();
    let half2 = (c + d) * 0.5;
    assert_eq!(F16::from_f32(half2).0, 0x3c02);
}

#[test]
fn test_bulk_f16_to_f32_matches_scalar() {
    let src: Vec<u16> = (0..1024)
        .map(|i| F16::from_f32(i as f32 * 0.1 - 50.0).0)
        .collect();
    let mut dst_bulk = vec![0.0f32; 1024];
    let mut dst_scalar = vec![0.0f32; 1024];

    f16_to_f32_slice(&src, &mut dst_bulk);
    for (i, &v) in src.iter().enumerate() {
        dst_scalar[i] = F16(v).to_f32();
    }

    assert_eq!(
        dst_bulk, dst_scalar,
        "bulk and scalar conversion must match exactly"
    );
}

#[test]
fn test_bulk_f32_to_f16_matches_scalar() {
    let src: Vec<f32> = (0..1024)
        .map(|i| ((i * 17 + 29) % 4096) as f32 / 256.0 - 8.0)
        .collect();
    let mut dst_bulk = vec![0u16; 1024];
    let mut dst_scalar = vec![0u16; 1024];

    f32_to_f16_slice(&src, &mut dst_bulk);
    for (i, &v) in src.iter().enumerate() {
        dst_scalar[i] = F16::from_f32(v).0;
    }

    assert_eq!(dst_bulk, dst_scalar);
}

#[test]
fn test_f16_moe_router_shape_validation() {
    let router = F16MoeRouter::new(vec![F16::ONE.0; 6], 3, 2, 2).expect("valid router shape");
    assert_eq!(router.num_experts, 3);
    assert_eq!(router.num_experts_per_tok, 2);
    assert_eq!(router.hidden_size, 2);
    assert_eq!(router.gate.len(), 6);

    assert!(matches!(
        F16MoeRouter::new(vec![0; 2], 1, 2, 2),
        Err(InferenceError::UnsupportedModel(_))
    ));

    assert!(matches!(
        F16MoeRouter::new(vec![0; 5], 3, 1, 2),
        Err(InferenceError::ShapeMismatch { name, expected, actual })
            if name == "mlp.gate.weight" && expected == vec![3, 2] && actual == vec![5]
    ));
}

#[test]
fn test_f16_routed_experts_shape_validation() {
    let experts = F16RoutedExperts::new(vec![0; 48], vec![0; 24], 2, 3, 4)
        .expect("valid routed expert shapes");
    assert_eq!(experts.num_experts, 2);
    assert_eq!(experts.hidden_size, 3);
    assert_eq!(experts.intermediate_size, 4);
    assert_eq!(experts.gate_up_proj.len(), 48);
    assert_eq!(experts.down_proj.len(), 24);

    assert!(matches!(
        F16RoutedExperts::new(vec![0; 47], vec![0; 24], 2, 3, 4),
        Err(InferenceError::ShapeMismatch { name, expected, actual })
            if name == "mlp.experts.gate_up_proj"
                && expected == vec![2, 8, 3]
                && actual == vec![47]
    ));

    assert!(matches!(
        F16RoutedExperts::new(vec![0; 48], vec![0; 23], 2, 3, 4),
        Err(InferenceError::ShapeMismatch { name, expected, actual })
            if name == "mlp.experts.down_proj"
                && expected == vec![2, 3, 4]
                && actual == vec![23]
    ));
}

#[test]
fn test_f16_shared_expert_shape_validation() {
    let shared = F16SharedExpert::new(vec![0; 6], vec![0; 6], vec![0; 6], vec![0; 3], 3, 2)
        .expect("valid shared expert shapes");
    assert_eq!(shared.hidden_size, 3);
    assert_eq!(shared.intermediate_size, 2);
    assert_eq!(shared.gate_proj.len(), 6);
    assert_eq!(shared.up_proj.len(), 6);
    assert_eq!(shared.down_proj.len(), 6);
    assert_eq!(shared.shared_expert_gate.len(), 3);

    assert!(matches!(
        F16SharedExpert::new(vec![0; 5], vec![0; 6], vec![0; 6], vec![0; 3], 3, 2),
        Err(InferenceError::ShapeMismatch { name, expected, actual })
            if name == "mlp.shared_expert.gate_proj.weight"
                && expected == vec![2, 3]
                && actual == vec![5]
    ));

    assert!(matches!(
        F16SharedExpert::new(vec![0; 6], vec![0; 5], vec![0; 6], vec![0; 3], 3, 2),
        Err(InferenceError::ShapeMismatch { name, expected, actual })
            if name == "mlp.shared_expert.up_proj.weight"
                && expected == vec![2, 3]
                && actual == vec![5]
    ));

    assert!(matches!(
        F16SharedExpert::new(vec![0; 6], vec![0; 6], vec![0; 5], vec![0; 3], 3, 2),
        Err(InferenceError::ShapeMismatch { name, expected, actual })
            if name == "mlp.shared_expert.down_proj.weight"
                && expected == vec![3, 2]
                && actual == vec![5]
    ));

    assert!(matches!(
        F16SharedExpert::new(vec![0; 6], vec![0; 6], vec![0; 6], vec![0; 2], 3, 2),
        Err(InferenceError::ShapeMismatch { name, expected, actual })
            if name == "mlp.shared_expert_gate.weight"
                && expected == vec![1, 3]
                && actual == vec![2]
    ));
}

#[test]
fn test_f16_moe_tensor_names() {
    let prefix = "model.language_model.layers.17";
    let names = [
        format!("{prefix}.mlp.gate.weight"),
        format!("{prefix}.mlp.experts.gate_up_proj"),
        format!("{prefix}.mlp.experts.down_proj"),
        format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
        format!("{prefix}.mlp.shared_expert.up_proj.weight"),
        format!("{prefix}.mlp.shared_expert.down_proj.weight"),
        format!("{prefix}.mlp.shared_expert_gate.weight"),
    ];

    assert_eq!(
        names,
        [
            "model.language_model.layers.17.mlp.gate.weight".to_string(),
            "model.language_model.layers.17.mlp.experts.gate_up_proj".to_string(),
            "model.language_model.layers.17.mlp.experts.down_proj".to_string(),
            "model.language_model.layers.17.mlp.shared_expert.gate_proj.weight".to_string(),
            "model.language_model.layers.17.mlp.shared_expert.up_proj.weight".to_string(),
            "model.language_model.layers.17.mlp.shared_expert.down_proj.weight".to_string(),
            "model.language_model.layers.17.mlp.shared_expert_gate.weight".to_string(),
        ]
    );
    assert!(!names.contains(&format!("{prefix}.mlp.experts.gate_up_proj.weight")));
    assert!(!names.contains(&format!("{prefix}.mlp.experts.down_proj.weight")));
    assert!(!names.contains(&format!("{prefix}.mlp.gate_proj.weight")));
}
