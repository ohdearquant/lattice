use super::*;

fn write_constructor_source(input: &Path) -> Qwen35Config {
    let mut cfg = tiny_cfg_with_mtp(true);
    cfg.hidden_size = 256;
    cfg.head_dim = 256;
    cfg.num_attention_heads = 1;
    cfg.num_key_value_heads = 1;
    cfg.intermediate_size = 256;
    cfg.num_hidden_layers = 1;
    cfg.full_attention_interval = 1;
    cfg.layer_types = vec![LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    write_input_dir_with_mtp(&cfg, input, 70);

    let path = input.join("model.safetensors");
    let bytes = fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let payload = &bytes[8 + header_len..];
    let mut entries = Vec::new();
    for (name, tensor) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        assert_eq!(tensor["dtype"], "F32");
        let mut shape: Vec<usize> = tensor["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        let start = usize::try_from(tensor["data_offsets"][0].as_u64().unwrap()).unwrap();
        let end = usize::try_from(tensor["data_offsets"][1].as_u64().unwrap()).unwrap();
        let mut values: Vec<f64> = payload[start..end]
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
            .collect();
        if name == "model.language_model.norm.weight" {
            assert_eq!(shape, [256]);
            values = (0..256)
                .map(|i| [0.0, 1.0001, -0.5001, -1.0001][i % 4])
                .collect();
        }
        let mtp_shape = match name.as_str() {
            "mtp.layers.0.self_attn.q_proj.weight" => Some(vec![512, 256]),
            "mtp.layers.0.self_attn.k_proj.weight" | "mtp.layers.0.self_attn.v_proj.weight" => {
                Some(vec![256, 256])
            }
            "mtp.layers.0.self_attn.o_proj.weight" => Some(vec![256, 256]),
            _ => None,
        };
        if let Some(expected_shape) = mtp_shape {
            // The converter-only fixture overpads these projections; this one also reaches the full loader.
            values.truncate(expected_shape.iter().product());
            shape = expected_shape;
        }
        assert_eq!(values.len(), shape.iter().product::<usize>());
        entries.push((name.clone(), shape, values));
    }
    let borrowed: Vec<(&str, Vec<usize>, &[f64])> = entries
        .iter()
        .map(|(name, shape, values)| (name.as_str(), shape.clone(), values.as_slice()))
        .collect();
    write_test_safetensors(&path, &borrowed);
    cfg
}

#[test]
#[ignore = "requires a fresh absolute LATTICE_QUAROT_MTP_FULL_FIXTURE_ROOT for full-constructor compatibility"]
fn original_final_norm_export_full_constructor_fixture() {
    let root = PathBuf::from(
        std::env::var_os("LATTICE_QUAROT_MTP_FULL_FIXTURE_ROOT")
            .expect("LATTICE_QUAROT_MTP_FULL_FIXTURE_ROOT must name a fresh absolute directory"),
    );
    assert!(root.is_absolute());
    fs::create_dir(&root).expect("full-constructor fixture root must not already exist");
    let input = root.join("input");
    let output = root.join("output");
    let cfg = write_constructor_source(&input);
    convert_original_final_norm_fixture(&input, &output).unwrap();
    assert_original_final_norm_artifact(&output, cfg.hidden_size);
    let index: Value =
        serde_json::from_slice(&fs::read(output.join("quantize_index.json")).unwrap()).unwrap();
    let entry = index["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["file"] == ORIGINAL_FINAL_NORM_FILE)
        .unwrap();
    assert_eq!(entry["name"], "quarot.original_final_norm.weight");
    assert_eq!(
        fs::metadata(output.join(ORIGINAL_FINAL_NORM_FILE))
            .unwrap()
            .len(),
        540
    );
    fs::write(
        root.join("fixture.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "quarot-mtp-full-constructor-fixture-v1",
            "source": "input/model.safetensors",
            "artifact": "output",
            "source_tensor": "model.language_model.norm.weight",
            "index_name": "quarot.original_final_norm.weight",
            "file": "quarot_original_final_norm_weight.f16",
            "hidden_size": 256,
            "head_dim": 256,
            "attention_heads": 1,
            "kv_heads": 1,
            "layers": 1,
            "source_pattern": [0.0, 1.0001, -0.5001, -1.0001],
            "expected_half_bits": [0, 15360, 47104, 48128],
            "sidecar_bytes": 540,
            "doctor_resident_byte_delta": 1080
        }))
        .unwrap(),
    )
    .unwrap();
}
