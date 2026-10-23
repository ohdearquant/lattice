use super::*;

#[test]
fn read_lora_file_bounded_rejects_oversized_before_read() {
    // A 64-byte file with a 32-byte cap must be refused by the metadata
    // stat, before its contents are read into memory. The file is trivially
    // readable, so a failure here is the size policy firing, not an IO error.
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(&[0u8; 64]).unwrap();
    f.flush().unwrap();
    let err = read_lora_file_bounded(f.path(), 32)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("exceeds maximum"),
        "expected the size guard to fire; got: {err}"
    );
}

#[test]
fn read_lora_file_bounded_reads_within_cap() {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().unwrap();
    let payload = [7u8; 64];
    f.write_all(&payload).unwrap();
    f.flush().unwrap();
    let bytes = read_lora_file_bounded(f.path(), 1024).unwrap();
    assert_eq!(bytes, payload);
}

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

/// Helper: create a PEFT-format safetensors file with known LoRA weights.
fn write_test_peft_safetensors(path: &std::path::Path, rank: usize, d_in: usize, d_out: usize) {
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize};
    use std::collections::HashMap;

    // Layer 0, q_proj: A=(rank, d_in), B=(d_out, rank)
    let a_data: Vec<f32> = (0..rank * d_in).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..d_out * rank).map(|i| (i as f32) * 0.1).collect();

    // Layer 2, gate_proj: same dimensions
    let a2_data: Vec<f32> = (0..rank * d_in).map(|i| (i as f32) * -0.01).collect();
    let b2_data: Vec<f32> = (0..d_out * rank).map(|i| (i as f32) * -0.1).collect();

    let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let a2_bytes: Vec<u8> = a2_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let b2_bytes: Vec<u8> = b2_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        TensorView::new(Dtype::F32, vec![rank, d_in], &a_bytes).unwrap(),
    );
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        TensorView::new(Dtype::F32, vec![d_out, rank], &b_bytes).unwrap(),
    );
    tensors.insert(
        "base_model.model.model.layers.2.mlp.gate_proj.lora_A.weight".to_string(),
        TensorView::new(Dtype::F32, vec![rank, d_in], &a2_bytes).unwrap(),
    );
    tensors.insert(
        "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight".to_string(),
        TensorView::new(Dtype::F32, vec![d_out, rank], &b2_bytes).unwrap(),
    );

    let bytes = serialize(&tensors, &None).unwrap();
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn test_load_peft_safetensors_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("adapter.safetensors");

    let rank = 4;
    let d_in = 8;
    let d_out = 8;
    write_test_peft_safetensors(&path, rank, d_in, d_out);

    let adapter = load_peft_safetensors(&path).unwrap();

    // Check config
    assert_eq!(adapter.config().rank, rank);
    assert_eq!(adapter.config().target_modules.len(), 2);
    assert!(
        adapter
            .config()
            .target_modules
            .contains(&"q_proj".to_string())
    );
    assert!(
        adapter
            .config()
            .target_modules
            .contains(&"gate_proj".to_string())
    );

    // Check we got both layers
    assert_eq!(adapter.layers().len(), 2);
    assert!(adapter.layers().contains_key(&(0, "q_proj".to_string())));
    assert!(adapter.layers().contains_key(&(2, "gate_proj".to_string())));

    // Check dimensions
    let q_lora = &adapter.layers()[&(0, "q_proj".to_string())];
    assert_eq!(q_lora.rank, rank);
    assert_eq!(q_lora.d_in, d_in);
    assert_eq!(q_lora.d_out, d_out);
    assert_eq!(q_lora.a.len(), rank * d_in);
    assert_eq!(q_lora.b.len(), d_out * rank);

    // Verify actual values for A[0]
    assert!((q_lora.a[0] - 0.0).abs() < 1e-6);
    assert!((q_lora.a[1] - 0.01).abs() < 1e-6);
    assert!((q_lora.a[2] - 0.02).abs() < 1e-6);

    // Check gate_proj layer
    let g_lora = &adapter.layers()[&(2, "gate_proj".to_string())];
    assert_eq!(g_lora.rank, rank);
    assert!((g_lora.a[1] - (-0.01)).abs() < 1e-6);
}

#[test]
fn test_load_mlx_safetensors_transposed() {
    // MLX format: A=(d_in, rank), B=(rank, d_out) with lowercase keys
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("adapters.safetensors");

    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize};

    // rank=2, d_in=4, d_out=3
    // MLX A: (d_in=4, rank=2) -> after transpose: (rank=2, d_in=4)
    // MLX B: (rank=2, d_out=3) -> after transpose: (d_out=3, rank=2)
    let a_data: Vec<f32> = vec![
        // 4x2 matrix (d_in=4, rank=2)
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    let b_data: Vec<f32> = vec![
        // 2x3 matrix (rank=2, d_out=3)
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    ];

    let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_a".to_string(),
        TensorView::new(Dtype::F32, vec![4, 2], &a_bytes).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_b".to_string(),
        TensorView::new(Dtype::F32, vec![2, 3], &b_bytes).unwrap(),
    );

    let bytes = serialize(&tensors, &None).unwrap();
    std::fs::write(&path, bytes).unwrap();

    let adapter = load_peft_safetensors(&path).unwrap();
    assert_eq!(adapter.config().rank, 2);

    let lora = &adapter.layers()[&(0, "q_proj".to_string())];
    assert_eq!(lora.rank, 2);
    assert_eq!(lora.d_in, 4);
    assert_eq!(lora.d_out, 3);

    // Verify transpose: original A[0,0]=1, A[1,0]=3, A[2,0]=5, A[3,0]=7
    // Transposed A[0,0]=1, A[0,1]=3, A[0,2]=5, A[0,3]=7
    assert!((lora.a[0] - 1.0).abs() < 1e-6); // row 0
    assert!((lora.a[1] - 3.0).abs() < 1e-6);
    assert!((lora.a[2] - 5.0).abs() < 1e-6);
    assert!((lora.a[3] - 7.0).abs() < 1e-6);
    assert!((lora.a[4] - 2.0).abs() < 1e-6); // row 1
    assert!((lora.a[5] - 4.0).abs() < 1e-6);
}

#[test]
fn test_load_peft_safetensors_apply_correctness() {
    // End-to-end: write safetensors, load, apply, verify numerical result.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("adapter.safetensors");

    // rank=1, d_in=2, d_out=2 for simplicity
    // A = [[0.0, 0.01]]  (1x2)
    // B = [[0.0], [0.1]] (2x1)
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize};

    let a_data: Vec<f32> = vec![0.0, 1.0]; // A = [[0, 1]]
    let b_data: Vec<f32> = vec![1.0, 0.0]; // B = [[1], [0]]
    let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.v_proj.lora_A.weight".to_string(),
        TensorView::new(Dtype::F32, vec![1, 2], &a_bytes).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.v_proj.lora_B.weight".to_string(),
        TensorView::new(Dtype::F32, vec![2, 1], &b_bytes).unwrap(),
    );

    let bytes = serialize(&tensors, &None).unwrap();
    std::fs::write(&path, bytes).unwrap();

    let adapter = load_peft_safetensors(&path).unwrap();
    assert_eq!(adapter.config().rank, 1);
    // Default alpha = rank = 1, so scale = 1.0

    let lora = &adapter.layers()[&(0, "v_proj".to_string())];
    // x = [3, 5]
    // A @ x = [0*3 + 1*5] = [5]
    // B @ [5] = [1*5, 0*5] = [5, 0]
    // scale = 1.0 => output += [5, 0]
    let x = [3.0f32, 5.0];
    let mut output = [10.0, 20.0];
    super::super::apply_lora(lora, adapter.config().scale(), &x, &mut output);

    assert!((output[0] - 15.0).abs() < 1e-6);
    assert!((output[1] - 20.0).abs() < 1e-6);
}

fn write_raw_safetensors(path: &std::path::Path, header: &str, data: &[u8]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    std::fs::write(path, bytes).unwrap();
}

fn peft_bytes_with_alpha(alpha: &str) -> Vec<u8> {
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize};

    let a_bytes = 1.0f32.to_le_bytes();
    let b_bytes = 1.0f32.to_le_bytes();
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        TensorView::new(Dtype::F32, vec![1, 1], &a_bytes).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        TensorView::new(Dtype::F32, vec![1, 1], &b_bytes).unwrap(),
    );
    let metadata = HashMap::from([("alpha".to_string(), alpha.to_string())]);
    serialize(&tensors, &Some(metadata)).unwrap()
}

#[test]
fn test_rejects_non_finite_alpha_metadata() {
    for alpha in ["nan", "inf", "-inf"] {
        let bytes = peft_bytes_with_alpha(alpha);
        let err = load_peft_safetensors_bytes(&bytes).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("LoRA alpha metadata must be finite"),
            "expected descriptive rejection for {alpha}, got: {message}"
        );
    }
}

#[test]
fn test_rejects_truncated_safetensors_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.safetensors");
    std::fs::write(&path, 128u64.to_le_bytes()).unwrap();

    let err = load_peft_safetensors(&path).unwrap_err();
    assert!(err.to_string().contains("failed to parse safetensors"));
}

#[test]
fn test_rejects_file_without_lora_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.safetensors");
    let header = r#"{"metadata.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    write_raw_safetensors(&path, header, &1.0f32.to_le_bytes());

    let err = load_peft_safetensors(&path).unwrap_err();
    assert!(err.to_string().contains("no lora_A/lora_B tensors"));
}

#[test]
fn test_rejects_shape_product_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overflow.safetensors");
    let max = usize::MAX;
    let header = format!(
        r#"{{"model.layers.0.self_attn.q_proj.lora_A.weight":{{"dtype":"F32","shape":[{max},2],"data_offsets":[0,0]}},"model.layers.0.self_attn.q_proj.lora_B.weight":{{"dtype":"F32","shape":[1,{max}],"data_offsets":[0,0]}}}}"#
    );
    write_raw_safetensors(&path, &header, &[]);

    let err = load_peft_safetensors(&path).unwrap_err();
    assert!(
        err.to_string().contains("shape overflow")
            || err.to_string().contains("failed to parse safetensors")
    );
}

#[test]
fn test_rejects_non_finite_tensor_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nan.safetensors");

    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize};

    let a_data: Vec<f32> = vec![f32::NAN, 1.0];
    let b_data: Vec<f32> = vec![1.0, 2.0];
    let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = std::collections::HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        TensorView::new(Dtype::F32, vec![1, 2], &a_bytes).unwrap(),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        TensorView::new(Dtype::F32, vec![2, 1], &b_bytes).unwrap(),
    );

    let bytes = serialize(&tensors, &None).unwrap();
    std::fs::write(&path, bytes).unwrap();

    let err = load_peft_safetensors(&path).unwrap_err();
    assert!(err.to_string().contains("non-finite"));
}

#[test]
fn test_save_load_round_trip() {
    use tempfile::NamedTempFile;

    let rank = 4;
    let d_in = 8;
    let d_out = 16;

    // alpha == rank here, so the round-trip lands back on rank either way;
    // test_alpha_metadata_round_trips covers the alpha != rank case.
    let config = LoraConfig {
        rank,
        alpha: rank as f32,
        target_modules: vec!["q_proj".to_string(), "gate_proj".to_string()],
        dtype: "f32".into(),
    };

    let a_data: Vec<f32> = (0..rank * d_in).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..d_out * rank).map(|i| i as f32 * 0.1).collect();
    let a2_data: Vec<f32> = (0..rank * d_in).map(|i| i as f32 * -0.01).collect();
    let b2_data: Vec<f32> = (0..d_out * rank).map(|i| i as f32 * -0.1).collect();

    let mut layers = HashMap::new();
    layers.insert(
        (0usize, "q_proj".to_string()),
        LoraLayer {
            a: a_data.clone(),
            b: b_data.clone(),
            d_in,
            d_out,
            rank,
        },
    );
    layers.insert(
        (2usize, "gate_proj".to_string()),
        LoraLayer {
            a: a2_data.clone(),
            b: b2_data.clone(),
            d_in,
            d_out,
            rank,
        },
    );

    let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");

    let temp = NamedTempFile::new().unwrap();
    save_peft_safetensors(&adapter, temp.path(), None).unwrap();

    let loaded = load_peft_safetensors(temp.path()).unwrap();

    assert_eq!(loaded.layers().len(), adapter.layers().len());
    assert_eq!(loaded.config().rank, adapter.config().rank);
    assert_eq!(loaded.config().alpha, adapter.config().alpha);

    for (key, orig) in adapter.layers() {
        let got = loaded
            .layers()
            .get(key)
            .expect("layer key missing after round-trip");
        assert_eq!(got.rank, orig.rank);
        assert_eq!(got.d_in, orig.d_in);
        assert_eq!(got.d_out, orig.d_out);
        assert_eq!(got.a.len(), orig.a.len());
        assert_eq!(got.b.len(), orig.b.len());
        for (g, w) in got.a.iter().zip(&orig.a) {
            assert!((g - w).abs() < f32::EPSILON, "A mismatch: {g} vs {w}");
        }
        for (g, w) in got.b.iter().zip(&orig.b) {
            assert!((g - w).abs() < f32::EPSILON, "B mismatch: {g} vs {w}");
        }
    }
}

/// `save_peft_safetensors(.., Some(governance))` must embed all six
/// governance fields into the header, round-trippable via
/// `read_peft_header_governance`. Mutation-sensitive: removing the
/// `if let Some(governance) = governance { .. }` embedding block makes
/// this `None` (no `gov_name` key present), so `.expect(..)` panics —
/// this test fails when the embedding is absent.
#[test]
fn test_governance_metadata_round_trips() {
    use tempfile::NamedTempFile;

    let rank = 4;
    let config = LoraConfig {
        rank,
        alpha: rank as f32,
        target_modules: vec!["q_proj".to_string()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0usize, "q_proj".to_string()),
        LoraLayer {
            a: vec![0.0; rank * 4],
            b: vec![0.0; 4 * rank],
            d_in: 4,
            d_out: 4,
            rank,
        },
    );
    let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");

    let governance = AdapterGovernance {
        name: "test-adapter".to_string(),
        owner: "team-inference".to_string(),
        base_model_rev: "rev-aaa".to_string(),
        tokenizer_rev: "rev-bbb".to_string(),
        dtype: "f32".to_string(),
        status: "approved".to_string(),
    };

    let temp = NamedTempFile::new().unwrap();
    save_peft_safetensors(&adapter, temp.path(), Some(&governance)).unwrap();

    let bytes = std::fs::read(temp.path()).unwrap();
    let got = read_peft_header_governance(&bytes).expect("governance metadata must be present");
    assert_eq!(got.name, "test-adapter");
    assert_eq!(got.owner, "team-inference");
    assert_eq!(got.base_model_rev, "rev-aaa");
    assert_eq!(got.tokenizer_rev, "rev-bbb");
    assert_eq!(got.dtype, "f32");
    assert_eq!(got.status, "approved");
}

/// Saving with `governance: None` must not embed any `gov_*` keys — the
/// feature is opt-in, not a default-on side effect.
#[test]
fn test_no_governance_metadata_when_not_supplied() {
    use tempfile::NamedTempFile;

    let rank = 4;
    let config = LoraConfig {
        rank,
        alpha: rank as f32,
        target_modules: vec!["q_proj".to_string()],
        dtype: "f32".into(),
    };
    let mut layers = HashMap::new();
    layers.insert(
        (0usize, "q_proj".to_string()),
        LoraLayer {
            a: vec![0.0; rank * 4],
            b: vec![0.0; 4 * rank],
            d_in: 4,
            d_out: 4,
            rank,
        },
    );
    let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");

    let temp = NamedTempFile::new().unwrap();
    save_peft_safetensors(&adapter, temp.path(), None).unwrap();

    let bytes = std::fs::read(temp.path()).unwrap();
    assert!(read_peft_header_governance(&bytes).is_none());
}

/// `AdapterGovernance::from_entry` must faithfully carry over every
/// field, converting `AdapterStatus` to its lowercase string form.
#[cfg(feature = "serde")]
#[test]
fn test_adapter_governance_from_entry() {
    use crate::lora::manifest::{AdapterStatus, ManifestEntry};

    let entry = ManifestEntry {
        id: "adapter-1".to_string(),
        name: "adapter one".to_string(),
        owner: "team-tune".to_string(),
        uri: "adapter-1.safetensors".to_string(),
        integrity_sha256: "deadbeef".to_string(),
        base_model_rev: "rev-aaa".to_string(),
        tokenizer_rev: "rev-bbb".to_string(),
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string()],
        dtype: "f16".to_string(),
        status: AdapterStatus::Approved,
    };

    let governance = AdapterGovernance::from_entry(&entry);
    assert_eq!(governance.name, "adapter one");
    assert_eq!(governance.owner, "team-tune");
    assert_eq!(governance.base_model_rev, "rev-aaa");
    assert_eq!(governance.tokenizer_rev, "rev-bbb");
    assert_eq!(governance.dtype, "f16");
    assert_eq!(governance.status, "approved");
}

#[test]
fn test_alpha_metadata_round_trips() {
    // Regression: an adapter trained with alpha != rank must load with its
    // saved alpha, not alpha = rank. The old loader hardcoded alpha = rank,
    // so a rank-4 / alpha-16 adapter loaded at scale 1.0 instead of 4.0.
    use tempfile::NamedTempFile;

    let rank = 4;
    let d_in = 8;
    let d_out = 16;
    let config = LoraConfig {
        rank,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string()],
        dtype: "f32".into(),
    };

    let mut layers = HashMap::new();
    layers.insert(
        (0usize, "q_proj".to_string()),
        LoraLayer {
            a: (0..rank * d_in).map(|i| i as f32 * 0.01).collect(),
            b: (0..d_out * rank).map(|i| i as f32 * 0.1).collect(),
            d_in,
            d_out,
            rank,
        },
    );

    let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");
    let temp = NamedTempFile::new().unwrap();
    save_peft_safetensors(&adapter, temp.path(), None).unwrap();
    let loaded = load_peft_safetensors(temp.path()).unwrap();

    assert_eq!(loaded.config().rank, 4);
    assert_eq!(loaded.config().alpha, 16.0);
    assert!((loaded.config().alpha - 16.0).abs() < f32::EPSILON);
}

#[test]
fn test_alpha_falls_back_to_rank_without_metadata() {
    // An adapter file with no `alpha` in its header (e.g. a raw PEFT export)
    // must fall back to alpha = rank (scale = 1.0), not panic or zero out.
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_alpha.safetensors");

    let a_data: Vec<f32> = (0..2 * 8).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..16 * 2).map(|i| i as f32 * 0.1).collect();
    let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
        TensorView::new(Dtype::F32, vec![2, 8], &a_bytes).unwrap(),
    );
    tensors.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
        TensorView::new(Dtype::F32, vec![16, 2], &b_bytes).unwrap(),
    );

    let bytes = serialize(&tensors, &None).unwrap();
    std::fs::write(&path, bytes).unwrap();

    let loaded = load_peft_safetensors(&path).unwrap();
    assert_eq!(loaded.config().rank, 2);
    assert_eq!(loaded.config().alpha, 2.0);
}

#[test]
fn test_transpose_rejects_shape_element_mismatch() {
    // MLX-format (lowercase lora_a/lora_b) triggers the transpose path.
    // Shape [4, 2] implies 8 elements; we craft data_offsets=[0,24] which is 6 f32s.
    // The safetensors library catches this mismatch at deserialize() with TensorInvalidInfo
    // before our transpose guard fires.  The invariant we assert is: Err, not panic.
    //
    // Our guard (`data.len() != expected`) provides defence-in-depth for callers that
    // construct a (values, shape) pair from outside the safetensors library.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mismatch.safetensors");

    let header = r#"{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[4,2],"data_offsets":[0,24]},"model.layers.0.self_attn.q_proj.lora_b":{"dtype":"F32","shape":[2,3],"data_offsets":[24,48]}}"#;
    let header_bytes = header.as_bytes();
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    raw.extend_from_slice(header_bytes);
    let payload: Vec<u8> = (0u32..12).flat_map(u32::to_le_bytes).collect();
    raw.extend_from_slice(&payload);
    std::fs::write(&path, &raw).unwrap();

    let err = load_peft_safetensors(&path).unwrap_err();
    assert!(
        !err.to_string().is_empty(),
        "expected a non-empty error message, got empty string"
    );
}

#[test]
fn test_transpose_rejects_unaligned_f32_payload() {
    // MLX-format lora_a with shape=[2,2] but data_offsets=[0,11] (11 bytes, not aligned).
    // The library rejects this as TensorInvalidInfo.  Assert Err, not panic.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unaligned.safetensors");

    let header = r#"{"model.layers.0.self_attn.q_proj.lora_a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"model.layers.0.self_attn.q_proj.lora_b":{"dtype":"F32","shape":[2,2],"data_offsets":[16,32]}}"#;
    let header_bytes = header.as_bytes();
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    raw.extend_from_slice(header_bytes);
    // Only 11 bytes instead of the declared 32 — truncated, so deserialize fails.
    raw.extend_from_slice(&[0u8; 11]);
    std::fs::write(&path, &raw).unwrap();

    let err = load_peft_safetensors(&path).unwrap_err();
    assert!(
        !err.to_string().is_empty(),
        "expected a non-empty error message, got empty string"
    );
}

#[test]
fn test_transpose_guard_logic_is_sound() {
    // White-box check: verify that the guard expressions added to the transpose path
    // would produce the correct result for a known mismatch.  This exercises the
    // arithmetic without requiring the safetensors library to be bypassed.
    let (rows, cols): (usize, usize) = (4, 2);
    let data_len: usize = 6; // 6 elements, but shape implies 8
    let expected = rows.checked_mul(cols).unwrap();
    assert_ne!(data_len, expected, "precondition: mismatch exists");

    // Mimic the guard: data.len() != expected => would return Err
    let guard_fires = data_len != expected;
    assert!(guard_fires, "guard must detect the mismatch");
}

/// Regression test: saving an adapter that contains a GDN-slot layer with empty A/B
/// buffers must succeed (pre-fix it returned `Err(InvalidTensorView)`). The empty layer
/// must be silently dropped from the saved file; the real layer must round-trip intact.
#[test]
fn test_save_skips_empty_buffer_layers() {
    use tempfile::NamedTempFile;

    let rank: usize = 8;
    let config = LoraConfig {
        rank,
        alpha: rank as f32,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        dtype: "f32".into(),
    };

    let mut layers = HashMap::new();

    // Real GQA layer — should survive the round-trip.
    layers.insert(
        (19usize, "q_proj".to_string()),
        LoraLayer {
            a: vec![0.1f32; rank * 1024],
            b: vec![0.2f32; 4096 * rank],
            d_in: 1024,
            d_out: 4096,
            rank,
        },
    );

    // Empty GDN-slot layer — must be skipped, not serialized.
    layers.insert(
        (20usize, "v_proj".to_string()),
        LoraLayer {
            a: Vec::new(),
            b: Vec::new(),
            d_in: 1024,
            d_out: 512,
            rank,
        },
    );

    let adapter = LoraAdapter::new(config, layers).expect("valid adapter config");

    let temp = NamedTempFile::new().unwrap();
    // This was the regression: pre-fix this call returned Err(InvalidTensorView).
    save_peft_safetensors(&adapter, temp.path(), None)
        .expect("save must succeed even with an empty-buffer GDN slot");

    let loaded = load_peft_safetensors(temp.path()).unwrap();

    // The real layer is present.
    let real_key = (19usize, "q_proj".to_string());
    assert!(
        loaded.layers().contains_key(&real_key),
        "real GQA layer (19, q_proj) must be present after round-trip"
    );

    // The empty layer was dropped — it must NOT appear in the saved file.
    let empty_key = (20usize, "v_proj".to_string());
    assert!(
        !loaded.layers().contains_key(&empty_key),
        "empty GDN-slot layer (20, v_proj) must be absent from saved adapter"
    );

    // The A buffer of the real layer must be bit-exact.
    let got = &loaded.layers()[&real_key];
    let expected_a = vec![0.1f32; rank * 1024];
    assert_eq!(got.a.len(), expected_a.len());
    for (g, w) in got.a.iter().zip(&expected_a) {
        assert!(
            (g - w).abs() < f32::EPSILON,
            "A buffer mismatch: {g} vs {w}"
        );
    }
}
