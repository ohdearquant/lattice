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

#[test]
fn encode_peft_uses_checkpoint_block_for_every_supported_projection() {
    let layer = LoraLayer {
        a: vec![0.5, -1.0, 2.0, 0.25],
        b: vec![1.5, -0.5, 3.0, 4.0, -2.0, 0.125],
        d_in: 2,
        d_out: 3,
        rank: 2,
    };
    let modules = [
        ("self_attn", "q_proj"),
        ("self_attn", "k_proj"),
        ("self_attn", "v_proj"),
        ("self_attn", "o_proj"),
        ("linear_attn", "in_proj_qkv"),
        ("linear_attn", "in_proj_z"),
        ("linear_attn", "in_proj_b"),
        ("linear_attn", "in_proj_a"),
        ("linear_attn", "out_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    ];
    let bytes = encode_peft_safetensors(
        modules.iter().map(|(_, module)| (5, *module, &layer)),
        HashMap::new(),
    )
    .unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let decoded = decode_peft_safetensors(&bytes).unwrap();
    assert_eq!(decoded.layers.len(), modules.len());
    for (block, module) in modules {
        for factor in ["A", "B"] {
            let key =
                format!("base_model.model.model.layers.5.{block}.{module}.lora_{factor}.weight");
            assert!(tensors.tensor(&key).is_ok(), "missing {key}");
        }
        let actual = &decoded.layers[&(5, module.to_string())];
        assert_eq!(actual.a, layer.a, "{module}");
        assert_eq!(actual.b, layer.b, "{module}");
        assert_eq!((actual.d_in, actual.d_out, actual.rank), (2, 3, 2));
    }
}

#[test]
fn encode_peft_rejects_nonempty_unplaceable_modules() {
    let layer = LoraLayer {
        a: vec![1.0],
        b: vec![2.0],
        d_in: 1,
        d_out: 1,
        rank: 1,
    };
    for module in [
        "unknown",
        "query",
        "key",
        "value",
        "attn_output",
        "ffn_intermediate",
        "ffn_output",
    ] {
        let error = encode_peft_safetensors([(0, module, &layer)], HashMap::new()).unwrap_err();
        assert!(matches!(&error, LoraError::Format(_)));
        assert!(error.message().contains(module));
    }
}

#[test]
fn encode_peft_keeps_skipping_empty_unplaceable_placeholders() {
    let trained = LoraLayer {
        a: vec![1.0],
        b: vec![2.0],
        d_in: 1,
        d_out: 1,
        rank: 1,
    };
    let empty = LoraLayer {
        a: Vec::new(),
        b: Vec::new(),
        d_in: 1,
        d_out: 1,
        rank: 1,
    };
    let bytes = encode_peft_safetensors(
        [(0, "q_proj", &trained), (1, "unknown", &empty)],
        HashMap::new(),
    )
    .unwrap();
    let decoded = decode_peft_safetensors(&bytes).unwrap();
    assert_eq!(decoded.layers.len(), 1);
    assert!(decoded.layers.contains_key(&(0, "q_proj".to_string())));
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
fn a_file_mixing_peft_and_mlx_modules_is_refused() {
    let k_prefix = "base_model.model.model.layers.0.self_attn.k_proj";
    let peft_a = format!("{Q_PREFIX}.lora_A.weight");
    let mlx_a = format!("{k_prefix}.lora_a");
    let bytes = adapter_bytes(&[
        (&peft_a, &[0.5, -1.0], &[1, 2]),
        (&format!("{Q_PREFIX}.lora_B.weight"), &[2.0, 3.0], &[2, 1]),
        (&mlx_a, &[0.5, -1.0], &[2, 1]),
        (&format!("{k_prefix}.lora_b"), &[2.0, 3.0], &[1, 2]),
    ]);
    let err = decode_peft_safetensors(&bytes).unwrap_err();
    assert!(matches!(err, LoraError::Format(_)), "got {err:?}");
    let msg = err.message();
    assert!(
        msg.contains("mixes PEFT and MLX key forms across modules"),
        "got {msg}"
    );
    assert!(msg.contains(&peft_a) && msg.contains(&mlx_a), "got {msg}");
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

fn q_pair_bytes(metadata: Option<HashMap<String, String>>) -> Vec<u8> {
    let a = [0.5f32; 6];
    let b = [0.25f32; 8];
    let raw = [
        (format!("{Q_PREFIX}.lora_A.weight"), &a[..], vec![2usize, 3]),
        (format!("{Q_PREFIX}.lora_B.weight"), &b[..], vec![4usize, 2]),
    ];
    let bytes: Vec<(String, Vec<u8>, Vec<usize>)> = raw
        .iter()
        .map(|(name, data, shape)| {
            (
                name.clone(),
                data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                shape.clone(),
            )
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = bytes
        .iter()
        .map(|(name, data, shape)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    serialize(views, &metadata).unwrap()
}

fn decoded_q_pair(header_alpha: Option<&str>) -> DecodedAdapter {
    let metadata = header_alpha.map(|a| HashMap::from([("alpha".to_string(), a.to_string())]));
    decode_peft_safetensors(&q_pair_bytes(metadata)).unwrap()
}

fn sidecar(json: &str) -> PeftSidecar {
    parse_peft_sidecar(json.as_bytes()).unwrap()
}

const SZL_TARGET_MODULES: &str = r"(?:.*?(?:language|text).*?(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer).*?(?:qkv|proj|linear_fc1|linear_fc2|out_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj))|(?:\bmodel\.layers\.[\d]{1,}\.(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer)\.(?:(?:qkv|proj|linear_fc1|linear_fc2|out_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)))";

#[test]
fn real_export_sidecar_shapes_parse() {
    let list = sidecar(
        r#"{"peft_type":"LORA","r":16,"lora_alpha":32,"target_modules":["up_proj","down_proj","o_proj","gate_proj","k_proj","v_proj","q_proj"],"use_rslora":false,"use_dora":false,"rank_pattern":{},"alpha_pattern":{},"modules_to_save":null,"bias":"none","lora_bias":false,"exclude_modules":null,"layers_to_transform":null,"trainable_token_indices":null}"#,
    );
    assert_eq!(list.lora_alpha, 32.0);
    assert_eq!(list.r, 16);
    assert!(matches!(&list.target_modules, TargetModules::Names(n) if n.len() == 7));
    assert_eq!(list.exclude_modules, None);

    let json = serde_json::json!({
        "peft_type": "LORA", "r": 16, "lora_alpha": 32,
        "target_modules": SZL_TARGET_MODULES,
        "rank_pattern": {}, "alpha_pattern": {}, "bias": "none"
    });
    let pattern = parse_peft_sidecar(json.to_string().as_bytes()).unwrap();
    assert_eq!(
        pattern.target_modules,
        TargetModules::Pattern(SZL_TARGET_MODULES.to_string())
    );
    let matcher = pattern.target_modules.matcher().unwrap();
    for path in [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.3.linear_attn.in_proj_qkv",
        "model.language_model.layers.23.mlp.down_proj",
        "model.layers.4.mlp.up_proj",
    ] {
        assert!(matcher.matches(path), "{path}");
    }
    assert!(!matcher.matches("model.visual.blocks.0.attn.qkv"));
}

#[test]
fn sidecar_alpha_replaces_the_rank_default() {
    let mut adapter = decoded_q_pair(None);
    assert_eq!(adapter.alpha_source, AlphaSource::RankDefault);
    assert_eq!(adapter.alpha, 2.0);
    assert!(!adapter.has_mlx_factors);
    assert_eq!(
        adapter.peft_module_paths,
        vec!["model.layers.0.self_attn.q_proj".to_string()]
    );
    apply_peft_sidecar(
        &mut adapter,
        &sidecar(r#"{"r":2,"lora_alpha":8,"target_modules":["q_proj"]}"#),
    )
    .unwrap();
    assert_eq!(adapter.alpha, 8.0);
    assert_eq!(adapter.alpha_source, AlphaSource::PeftSidecar);
}

#[test]
fn header_alpha_must_agree_with_sidecar_alpha() {
    let mut adapter = decoded_q_pair(Some("4"));
    assert_eq!(adapter.alpha_source, AlphaSource::Header);
    let err = apply_peft_sidecar(
        &mut adapter,
        &sidecar(r#"{"r":2,"lora_alpha":8,"target_modules":["q_proj"]}"#),
    )
    .unwrap_err();
    assert!(matches!(err, LoraError::Validation(_)));
    let msg = err.message();
    assert!(
        msg.contains("header alpha 4") && msg.contains("lora_alpha 8"),
        "{msg}"
    );
    assert_eq!(adapter.alpha, 4.0);

    apply_peft_sidecar(
        &mut adapter,
        &sidecar(r#"{"r":2,"lora_alpha":4.0,"target_modules":["q_proj"]}"#),
    )
    .unwrap();
    assert_eq!(adapter.alpha_source, AlphaSource::PeftSidecar);
}

#[test]
fn sidecar_rank_must_equal_factor_rank() {
    let mut adapter = decoded_q_pair(None);
    let err = apply_peft_sidecar(
        &mut adapter,
        &sidecar(r#"{"r":16,"lora_alpha":32,"target_modules":["q_proj"]}"#),
    )
    .unwrap_err();
    assert!(matches!(err, LoraError::Validation(_)));
    assert!(err.message().contains("rank 2 but PEFT r is 16"));
    assert_eq!(adapter.alpha_source, AlphaSource::RankDefault);
}

#[test]
fn target_modules_must_cover_every_module_path() {
    for json in [
        r#"{"r":2,"lora_alpha":8,"target_modules":["v_proj"]}"#,
        // A name matches a whole path component, never a substring of one.
        r#"{"r":2,"lora_alpha":8,"target_modules":["proj"]}"#,
        // A pattern must match the whole path, as PEFT's fullmatch does.
        r#"{"r":2,"lora_alpha":8,"target_modules":"q_proj"}"#,
    ] {
        let mut adapter = decoded_q_pair(None);
        let err = apply_peft_sidecar(&mut adapter, &sidecar(json)).unwrap_err();
        assert!(matches!(err, LoraError::Validation(_)), "{json}");
        assert!(
            err.message()
                .contains("'model.layers.0.self_attn.q_proj' is not covered"),
            "{json}: {}",
            err.message()
        );
        assert_eq!(adapter.alpha, 2.0, "{json}");
    }
    for json in [
        r#"{"r":2,"lora_alpha":8,"target_modules":["self_attn.q_proj"]}"#,
        r#"{"r":2,"lora_alpha":8,"target_modules":["model.layers.0.self_attn.q_proj"]}"#,
        r#"{"r":2,"lora_alpha":8,"target_modules":".*\\.q_proj"}"#,
    ] {
        let mut adapter = decoded_q_pair(None);
        apply_peft_sidecar(&mut adapter, &sidecar(json)).unwrap();
        assert_eq!(adapter.alpha, 8.0, "{json}");
    }
}

#[test]
fn pattern_is_matched_whole_and_cannot_escape_the_wrapper() {
    for json in [
        r#"{"r":2,"lora_alpha":8,"target_modules":"q_proj)|(?:.*"}"#,
        r#"{"r":2,"lora_alpha":8,"target_modules":["q_proj"],"exclude_modules":"q_proj)|(?:.*"}"#,
    ] {
        let err = parse_peft_sidecar(json.as_bytes()).unwrap_err();
        assert!(matches!(err, LoraError::Unsupported(_)), "{json}: {err:?}");
    }
    let mut adapter = decoded_q_pair(None);
    let err = apply_peft_sidecar(
        &mut adapter,
        &sidecar(r#"{"r":2,"lora_alpha":8,"target_modules":"model|.*\\.k_proj"}"#),
    )
    .unwrap_err();
    assert!(
        err.message().contains("is not covered"),
        "{}",
        err.message()
    );
    apply_peft_sidecar(
        &mut adapter,
        &sidecar(r#"{"r":2,"lora_alpha":8,"target_modules":".*\\.k_proj|.*\\.q_proj"}"#),
    )
    .unwrap();
    assert_eq!(adapter.alpha, 8.0);
}

#[test]
fn exclude_modules_must_match_no_module_path() {
    let mut adapter = decoded_q_pair(None);
    let err = apply_peft_sidecar(
        &mut adapter,
        &sidecar(r#"{"r":2,"lora_alpha":8,"target_modules":["q_proj"],"exclude_modules":["self_attn.q_proj"]}"#),
    )
    .unwrap_err();
    assert!(matches!(err, LoraError::Validation(_)));
    assert!(err.message().contains("matches PEFT exclude_modules"));
    apply_peft_sidecar(
        &mut adapter,
        &sidecar(
            r#"{"r":2,"lora_alpha":8,"target_modules":["q_proj"],"exclude_modules":["k_proj"]}"#,
        ),
    )
    .unwrap();
}

#[test]
fn sidecar_does_not_apply_to_mlx_factors() {
    let a = [0.5f32; 6];
    let b = [0.25f32; 8];
    let mut adapter = decode_peft_safetensors(&adapter_bytes(&[
        ("model.layers.0.self_attn.q_proj.lora_a", &a, &[3, 2]),
        ("model.layers.0.self_attn.q_proj.lora_b", &b, &[2, 4]),
    ]))
    .unwrap();
    assert!(adapter.has_mlx_factors);
    assert!(adapter.peft_module_paths.is_empty());
    let err = apply_peft_sidecar(
        &mut adapter,
        &sidecar(r#"{"r":2,"lora_alpha":8,"target_modules":["q_proj"]}"#),
    )
    .unwrap_err();
    assert!(matches!(err, LoraError::Validation(_)));
}

#[test]
fn unsupported_peft_options_are_refused() {
    let base = r#""r":2,"lora_alpha":8,"target_modules":["q_proj"]"#;
    for extra in [
        r#""use_rslora":true"#,
        r#""use_dora":true"#,
        r#""rank_pattern":{"q_proj":4}"#,
        r#""alpha_pattern":{"q_proj":4}"#,
        r#""peft_type":"LOHA""#,
        r#""bias":"all""#,
        r#""lora_bias":true"#,
        r#""modules_to_save":["lm_head"]"#,
        r#""trainable_token_indices":[1,2]"#,
    ] {
        let json = format!("{{{base},{extra}}}");
        let err = parse_peft_sidecar(json.as_bytes()).unwrap_err();
        assert!(matches!(err, LoraError::Unsupported(_)), "{json}: {err:?}");
    }
    for target in [r#""all-linear""#, r#""(?=q)q_proj""#] {
        let json = format!(r#"{{"r":2,"lora_alpha":8,"target_modules":{target}}}"#);
        let err = parse_peft_sidecar(json.as_bytes()).unwrap_err();
        assert!(matches!(err, LoraError::Unsupported(_)), "{json}: {err:?}");
    }
    let ok = format!(
        r#"{{{base},"use_rslora":false,"use_dora":null,"rank_pattern":{{}},"alpha_pattern":null,"bias":"none","modules_to_save":[]}}"#
    );
    parse_peft_sidecar(ok.as_bytes()).unwrap();
}

#[test]
fn malformed_sidecars_are_format_errors() {
    for json in [
        "not json",
        "[1, 2]",
        r#"{"r":2,"target_modules":["q_proj"]}"#,
        r#"{"r":2,"lora_alpha":"8","target_modules":["q_proj"]}"#,
        r#"{"r":0,"lora_alpha":8,"target_modules":["q_proj"]}"#,
        r#"{"r":"2","lora_alpha":8,"target_modules":["q_proj"]}"#,
        r#"{"r":2.5,"lora_alpha":8,"target_modules":["q_proj"]}"#,
        r#"{"lora_alpha":8,"target_modules":["q_proj"]}"#,
        r#"{"r":2,"lora_alpha":8}"#,
        r#"{"r":2,"lora_alpha":8,"target_modules":null}"#,
        r#"{"r":2,"lora_alpha":8,"target_modules":[]}"#,
        r#"{"r":2,"lora_alpha":8,"target_modules":[1]}"#,
    ] {
        let err = parse_peft_sidecar(json.as_bytes()).unwrap_err();
        assert!(matches!(err, LoraError::Format(_)), "{json}: {err:?}");
    }
}

fn scratch_dir(label: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "lattice-lora-{label}-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    dir
}

#[test]
fn sidecar_read_distinguishes_missing_from_broken() {
    let dir = scratch_dir("sidecar");
    let adapter = dir.join("adapter_model.safetensors");
    assert_eq!(read_peft_sidecar(&adapter).unwrap(), None);

    let config = dir.join(PEFT_SIDECAR_FILE_NAME);
    std::fs::write(
        &config,
        r#"{"r":2,"lora_alpha":8,"target_modules":["q_proj"]}"#,
    )
    .unwrap();
    let read = read_peft_sidecar(&adapter).unwrap().unwrap();
    assert_eq!(read.lora_alpha, 8.0);

    std::fs::write(&config, r#"{"r":2}"#).unwrap();
    let err = read_peft_sidecar(&adapter).unwrap_err();
    assert!(matches!(err, LoraError::Format(_)));
    assert!(
        err.message().contains(PEFT_SIDECAR_FILE_NAME),
        "{}",
        err.message()
    );

    std::fs::remove_file(&config).unwrap();
    std::fs::create_dir(&config).unwrap();
    assert!(read_peft_sidecar(&adapter).is_err());
    std::fs::remove_dir(&config).unwrap();

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(dir.join("absent.json"), &config).unwrap();
        assert!(
            read_peft_sidecar(&adapter).is_err(),
            "a dangling config link is broken, not missing"
        );
        std::fs::remove_file(&config).unwrap();
    }
    std::fs::remove_dir(&dir).unwrap();
}
