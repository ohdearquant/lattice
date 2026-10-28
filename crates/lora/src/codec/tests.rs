use super::*;

#[test]
fn test_parse_peft_key_self_attn() {
    let key = "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight";
    let parsed = parse_peft_key(key).unwrap();
    assert_eq!(parsed.layer_idx, 5);
    assert_eq!(parsed.module, "q_proj");
    assert_eq!(parsed.matrix, LoraMatrix::A);
}

#[test]
fn test_parse_peft_key_mlp() {
    let key = "base_model.model.model.layers.12.mlp.gate_proj.lora_B.weight";
    let parsed = parse_peft_key(key).unwrap();
    assert_eq!(parsed.layer_idx, 12);
    assert_eq!(parsed.module, "gate_proj");
    assert_eq!(parsed.matrix, LoraMatrix::B);
}

#[test]
fn test_parse_peft_key_simple_prefix() {
    let key = "model.layers.0.self_attn.v_proj.lora_A.weight";
    let parsed = parse_peft_key(key).unwrap();
    assert_eq!(parsed.layer_idx, 0);
    assert_eq!(parsed.module, "v_proj");
    assert_eq!(parsed.matrix, LoraMatrix::A);
}

#[test]
fn test_parse_peft_key_no_weight_suffix() {
    let key = "base_model.model.model.layers.3.mlp.up_proj.lora_B";
    let parsed = parse_peft_key(key).unwrap();
    assert_eq!(parsed.layer_idx, 3);
    assert_eq!(parsed.module, "up_proj");
    assert_eq!(parsed.matrix, LoraMatrix::B);
}

#[test]
fn test_parse_peft_key_rejects_non_lora() {
    assert!(parse_peft_key("model.layers.0.self_attn.q_proj.weight").is_none());
    assert!(parse_peft_key("some_random_tensor").is_none());
    assert!(parse_peft_key("").is_none());
}

#[test]
fn test_parse_mlx_key_lowercase() {
    let key = "model.layers.3.self_attn.q_proj.lora_a";
    let parsed = parse_peft_key(key).unwrap();
    assert_eq!(parsed.layer_idx, 3);
    assert_eq!(parsed.module, "q_proj");
    assert_eq!(parsed.matrix, LoraMatrix::A);
    assert!(parsed.transposed);
}

#[test]
fn test_parse_mlx_key_mlp() {
    let key = "model.layers.7.mlp.down_proj.lora_b";
    let parsed = parse_peft_key(key).unwrap();
    assert_eq!(parsed.layer_idx, 7);
    assert_eq!(parsed.module, "down_proj");
    assert_eq!(parsed.matrix, LoraMatrix::B);
    assert!(parsed.transposed);
}

#[test]
fn test_peft_key_not_transposed() {
    let key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
    let parsed = parse_peft_key(key).unwrap();
    assert!(!parsed.transposed);
}

#[test]
fn test_parse_peft_key_all_target_modules() {
    let modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ];
    for (module, block) in modules.iter().zip(
        [
            "self_attn",
            "self_attn",
            "self_attn",
            "self_attn",
            "mlp",
            "mlp",
            "mlp",
        ]
        .iter(),
    ) {
        let key = format!("base_model.model.model.layers.7.{block}.{module}.lora_A.weight");
        let parsed = parse_peft_key(&key).unwrap();
        assert_eq!(parsed.layer_idx, 7);
        assert_eq!(parsed.module, *module);
        assert_eq!(parsed.matrix, LoraMatrix::A);
    }
}

#[test]
fn test_f16_to_f32_roundtrip() {
    // Test a few known values
    // 1.0 in f16 = 0x3C00
    assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
    // -1.0 in f16 = 0xBC00
    assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
    // 0.0 in f16 = 0x0000
    assert_eq!(f16_to_f32(0x0000), 0.0);
    // 0.5 in f16 = 0x3800
    assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
}

#[test]
fn test_bf16_to_f32() {
    // 1.0 in bf16 = 0x3F80 (same as upper 16 bits of f32 1.0)
    assert_eq!(bf16_to_f32(0x3F80), 1.0);
    // 0.0
    assert_eq!(bf16_to_f32(0x0000), 0.0);
}

#[test]
fn decode_errors_are_format_and_file_errors_are_io() {
    let err = decode_peft_safetensors(b"not a safetensors file").unwrap_err();
    assert!(matches!(err, LoraError::Format(_)), "got {err:?}");
    assert!(
        err.message().starts_with("failed to parse safetensors"),
        "got {err}"
    );

    let missing = std::env::temp_dir().join("lattice-lora-codec-missing-adapter.safetensors");
    let err = read_adapter_file_bounded(&missing, MAX_ADAPTER_FILE_BYTES).unwrap_err();
    assert!(matches!(err, LoraError::Io(_)), "got {err:?}");
    assert!(
        err.message().contains("failed to read metadata for"),
        "got {err}"
    );
}

#[test]
fn encode_then_decode_round_trips_factors_and_alpha() {
    let layer = LoraLayer {
        a: vec![0.5, -1.0, 2.0, 0.25],
        b: vec![1.5, -0.5, 3.0, 4.0, -2.0, 0.125],
        d_in: 2,
        d_out: 3,
        rank: 2,
    };
    let empty = LoraLayer {
        a: Vec::new(),
        b: Vec::new(),
        d_in: 2,
        d_out: 3,
        rank: 2,
    };
    let metadata = HashMap::from([("alpha".to_string(), "4".to_string())]);
    let bytes =
        encode_peft_safetensors([(1, "gate_proj", &layer), (0, "q_proj", &empty)], metadata)
            .unwrap();

    let decoded = decode_peft_safetensors(&bytes).unwrap();
    assert_eq!(decoded.rank, 2);
    assert_eq!(decoded.alpha, 4.0);
    assert_eq!(decoded.target_modules, vec!["gate_proj".to_string()]);
    assert_eq!(decoded.layers.len(), 1, "the empty pair is not written");
    let got = &decoded.layers[&(1, "gate_proj".to_string())];
    assert_eq!(got.a, layer.a);
    assert_eq!(got.b, layer.b);
    assert_eq!((got.d_in, got.d_out, got.rank), (2, 3, 2));
    assert_eq!(
        read_header_metadata_value(&bytes, "alpha").as_deref(),
        Some("4")
    );
    assert_eq!(read_header_metadata_value(&bytes, "absent"), None);
}

fn adapter_bytes(tensors: &[(&str, &[f32], &[usize])]) -> Vec<u8> {
    let raw: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
        .iter()
        .map(|(name, data, shape)| {
            let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            (name.to_string(), bytes, shape.to_vec())
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = raw
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    serialize(views, &None).unwrap()
}

const Q_PREFIX: &str = "base_model.model.model.layers.0.self_attn.q_proj";

#[test]
fn duplicate_factor_keys_are_refused_even_with_identical_bytes() {
    let a = [0.5f32, -1.0];
    let b = [2.0f32, 3.0];
    let with_suffix = format!("{Q_PREFIX}.lora_A.weight");
    let without_suffix = format!("{Q_PREFIX}.lora_A");
    let b_key = format!("{Q_PREFIX}.lora_B.weight");
    let bytes = adapter_bytes(&[
        (&with_suffix, &a, &[1, 2]),
        (&without_suffix, &a, &[1, 2]),
        (&b_key, &b, &[2, 1]),
    ]);
    let err = decode_peft_safetensors(&bytes).unwrap_err();
    assert!(matches!(err, LoraError::Format(_)), "got {err:?}");
    let msg = err.message();
    assert!(
        msg.contains("two lora_A tensors for layer 0 module 'q_proj'"),
        "got {msg}"
    );
    assert!(
        msg.contains(&with_suffix) && msg.contains(&without_suffix),
        "got {msg}"
    );
}

#[test]
fn a_pair_mixing_peft_and_mlx_key_forms_is_refused() {
    let a_key = format!("{Q_PREFIX}.lora_A.weight");
    let b_key = format!("{Q_PREFIX}.lora_b");
    let bytes = adapter_bytes(&[
        (&a_key, &[0.5, -1.0], &[1, 2]),
        (&b_key, &[2.0, 3.0], &[1, 2]),
    ]);
    let err = decode_peft_safetensors(&bytes).unwrap_err();
    assert!(matches!(err, LoraError::Format(_)), "got {err:?}");
    let msg = err.message();
    assert!(
        msg.contains("mixes PEFT and MLX key forms for layer 0 module 'q_proj'"),
        "got {msg}"
    );
    assert!(msg.contains(&a_key) && msg.contains(&b_key), "got {msg}");
}

#[test]
fn single_dialect_pairs_still_decode() {
    let peft = adapter_bytes(&[
        (&format!("{Q_PREFIX}.lora_A.weight"), &[0.5, -1.0], &[1, 2]),
        (&format!("{Q_PREFIX}.lora_B.weight"), &[2.0, 3.0], &[2, 1]),
    ]);
    let mlx = adapter_bytes(&[
        (&format!("{Q_PREFIX}.lora_a"), &[0.5, -1.0], &[2, 1]),
        (&format!("{Q_PREFIX}.lora_b"), &[2.0, 3.0], &[1, 2]),
    ]);
    for bytes in [peft, mlx] {
        let decoded = decode_peft_safetensors(&bytes).unwrap();
        let layer = &decoded.layers[&(0, "q_proj".to_string())];
        assert_eq!((layer.rank, layer.d_in, layer.d_out), (1, 2, 2));
        assert_eq!(layer.a, vec![0.5, -1.0]);
        assert_eq!(layer.b, vec![2.0, 3.0]);
    }
}
