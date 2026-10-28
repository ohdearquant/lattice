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

#[test]
fn explicit_text_namespaces_and_blocks_decode_peft_and_mlx() {
    // The language-model PEFT names are present in the supplied export headers;
    // other namespace and MLX forms exercise the supported table synthetically.
    let prefixes = [
        "base_model.model.model.layers",
        "base_model.model.model.language_model.layers",
        "model.layers",
        "model.language_model.layers",
    ];
    let blocks = [
        ("self_attn", "q_proj"),
        ("linear_attn", "in_proj_a"),
        ("mlp", "down_proj"),
    ];
    for prefix in prefixes {
        for (block, module) in blocks {
            for mlx in [false, true] {
                let stem = format!("{prefix}.0.{block}.{module}");
                let (a_key, b_key, a_shape, b_shape) = if mlx {
                    (
                        format!("{stem}.lora_a"),
                        format!("{stem}.lora_b"),
                        [2, 1],
                        [1, 2],
                    )
                } else {
                    (
                        format!("{stem}.lora_A.weight"),
                        format!("{stem}.lora_B.weight"),
                        [1, 2],
                        [2, 1],
                    )
                };
                let bytes = adapter_bytes(&[
                    (&a_key, &[0.5, -1.0], &a_shape),
                    (&b_key, &[2.0, 3.0], &b_shape),
                ]);
                let decoded = decode_peft_safetensors(&bytes).unwrap();
                let factor = &decoded.layers[&(0, module.to_string())];
                assert_eq!((factor.rank, factor.d_in, factor.d_out), (1, 2, 2));
                assert_eq!(factor.a, [0.5, -1.0]);
                assert_eq!(factor.b, [2.0, 3.0]);
                assert!(decoded.skipped.is_empty());
            }
        }
    }
}

#[test]
fn unknown_module_is_skipped_and_reported() {
    let a = "model.layers.0.mlp.unrecognized.lora_A.weight";
    let b = "model.layers.0.mlp.unrecognized.lora_B.weight";
    let bytes = adapter_bytes(&[
        (
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            &[1.0],
            &[1, 1],
        ),
        (
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            &[2.0],
            &[1, 1],
        ),
        (a, &[3.0], &[1, 1]),
        (b, &[4.0], &[1, 1]),
    ]);
    let decoded = decode_peft_safetensors(&bytes).unwrap();
    assert_eq!(decoded.layers.len(), 1);
    assert_eq!(decoded.skipped, vec![a.to_string(), b.to_string()]);
}

#[test]
fn outside_text_factor_is_refused() {
    for key in [
        "base_model.model.model.visual.layers.5.self_attn.k_proj.lora_A.weight",
        "model.audio_tower.layers.5.self_attn.k_proj.lora_A.weight",
        "arbitrary.model.layers.5.self_attn.k_proj.lora_A.weight",
        "layers.5.self_attn.k_proj.lora_A.weight",
    ] {
        let bytes = adapter_bytes(&[
            (
                "model.layers.0.self_attn.q_proj.lora_A.weight",
                &[1.0],
                &[1, 1],
            ),
            (
                "model.layers.0.self_attn.q_proj.lora_B.weight",
                &[2.0],
                &[1, 1],
            ),
            (key, &[3.0], &[1, 1]),
        ]);
        let error = decode_peft_safetensors(&bytes).unwrap_err();
        assert!(matches!(error, LoraError::Format(_)));
        assert!(error.message().contains("unsupported LoRA factor key"));
        assert!(error.message().contains(key));
    }
}

#[test]
fn unknown_block_is_refused() {
    let key = "model.layers.0.cross_attn.q_proj.lora_A.weight";
    let bytes = adapter_bytes(&[
        (
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            &[1.0],
            &[1, 1],
        ),
        (
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            &[2.0],
            &[1, 1],
        ),
        (key, &[3.0], &[1, 1]),
    ]);
    let error = decode_peft_safetensors(&bytes).unwrap_err();
    assert!(matches!(error, LoraError::Format(_)));
    assert!(error.message().contains("unsupported LoRA factor key"));
    assert!(error.message().contains(key));
}

#[test]
fn unknown_factor_suffix_is_refused() {
    for key in [
        "model.layers.0.self_attn.q_proj.lora_C",
        "model.layers.0.self_attn.q_proj.lora_A.bias",
        "model.layers.0.self_attn.q_proj.lora_a.weight",
    ] {
        let bytes = adapter_bytes(&[
            (
                "model.layers.0.self_attn.q_proj.lora_A.weight",
                &[1.0],
                &[1, 1],
            ),
            (
                "model.layers.0.self_attn.q_proj.lora_B.weight",
                &[2.0],
                &[1, 1],
            ),
            (key, &[3.0], &[1, 1]),
        ]);
        let error = decode_peft_safetensors(&bytes).unwrap_err();
        assert!(matches!(error, LoraError::Format(_)));
        assert!(error.message().contains(key));
    }
}

#[test]
fn unrelated_nonfactor_is_ignored() {
    let bytes = adapter_bytes(&[
        (
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            &[1.0],
            &[1, 1],
        ),
        (
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            &[2.0],
            &[1, 1],
        ),
        ("visual.layers.0.self_attn.q_proj.weight", &[3.0], &[1, 1]),
    ]);
    let decoded = decode_peft_safetensors(&bytes).unwrap();
    assert_eq!(decoded.layers.len(), 1);
    assert!(decoded.skipped.is_empty());
}

#[test]
fn real_export_text_key_names_decode_with_synthetic_factors() {
    for (stem, layer, module) in [
        (
            "base_model.model.model.language_model.layers.11.self_attn.k_proj",
            11,
            "k_proj",
        ),
        (
            "base_model.model.model.language_model.layers.0.linear_attn.in_proj_a",
            0,
            "in_proj_a",
        ),
        (
            "base_model.model.model.language_model.layers.0.mlp.down_proj",
            0,
            "down_proj",
        ),
    ] {
        let a_key = format!("{stem}.lora_A.weight");
        let b_key = format!("{stem}.lora_B.weight");
        let bytes = adapter_bytes(&[(&a_key, &[0.5], &[1, 1]), (&b_key, &[2.0], &[1, 1])]);
        let decoded = decode_peft_safetensors(&bytes).unwrap();
        assert!(decoded.layers.contains_key(&(layer, module.to_string())));
        assert!(decoded.skipped.is_empty());
    }
}

#[test]
fn unpaired_factors_and_rank_mismatch_preserve_error_precedence() {
    let a = "model.layers.0.self_attn.q_proj.lora_A.weight";
    let b = "model.layers.0.self_attn.q_proj.lora_B.weight";
    let orphan_b = "model.layers.5.self_attn.k_proj.lora_B.weight";
    let only_b = adapter_bytes(&[(orphan_b, &[1.0], &[1, 1])]);
    assert_eq!(
        decode_peft_safetensors(&only_b).unwrap_err().message(),
        "LoRA adapter has lora_B for layer 5 module 'k_proj' but no matching lora_A"
    );
    let missing_b = adapter_bytes(&[(a, &[1.0], &[1, 1]), (orphan_b, &[2.0], &[1, 1])]);
    assert_eq!(
        decode_peft_safetensors(&missing_b).unwrap_err().message(),
        "LoRA adapter has lora_A for layer 0 module 'q_proj' but no matching lora_B"
    );
    let rank_mismatch = adapter_bytes(&[
        (a, &[1.0, 2.0], &[1, 2]),
        (b, &[1.0, 2.0], &[1, 2]),
        (orphan_b, &[3.0], &[1, 1]),
    ]);
    assert_eq!(
        decode_peft_safetensors(&rank_mismatch)
            .unwrap_err()
            .message(),
        "LoRA rank mismatch for layer 0 module 'q_proj': A rank=1, B rank=2"
    );
}

#[test]
fn bounded_read_u64_max_round_trips_and_honors_finite_cap() {
    use std::io::Write;
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "lattice-lora-bounded-read-{}-{unique}",
        std::process::id()
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let bytes = b"small adapter fixture";
    file.write_all(bytes).unwrap();
    file.flush().unwrap();
    assert_eq!(read_adapter_file_bounded(&path, u64::MAX).unwrap(), bytes);
    assert_eq!(
        read_adapter_file_bounded(&path, bytes.len() as u64).unwrap(),
        bytes
    );
    let err = read_adapter_file_bounded(&path, bytes.len() as u64 - 1).unwrap_err();
    assert!(matches!(err, LoraError::Io(_)));
    assert!(err.message().contains("exceeds maximum"));
    drop(file);
    std::fs::remove_file(path).unwrap();
}
