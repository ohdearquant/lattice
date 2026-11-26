use super::*;

/// Every required tensor stored as BF16, except one matrix stored as F16.
fn bf16_fixture(cfg: &Qwen35Config) -> Vec<(String, String, u64, u64)> {
    let mut tensors = required_tensor_fixture(cfg);
    for tensor in &mut tensors {
        tensor.1 = "BF16".to_string();
    }
    let f16_matrix = tensors
        .iter()
        .position(|(name, ..)| bf16_store_holds_as_bits(name))
        .expect("the fixture holds matrices");
    tensors[f16_matrix].1 = "F16".to_string();
    tensors
}

fn write_fixture(dir: &Path, tensors: &[(String, String, u64, u64)]) {
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);
}

#[test]
fn bf16_store_counts_bf16_matrices_at_their_on_disk_size() {
    let dir = tempdir("st-bf16-store");
    let cfg = Qwen35Config::qwen35_0_8b();
    let tensors = bf16_fixture(&cfg);
    write_fixture(&dir, &tensors);
    let total = tensors.len() as u64;
    let held_as_bits = tensors
        .iter()
        .filter(|(name, dtype, ..)| dtype == "BF16" && bf16_store_holds_as_bits(name))
        .count() as u64;
    assert!(held_as_bits > 0 && held_as_bits < total - 1);

    // Every tensor is 64 bytes on disk and every dtype here is two bytes wide: the f32
    // store doubles all of them, the bf16 store doubles all but the BF16 matrices.
    let f32_store = inspect_safetensors_dir_in_store(&dir, &cfg, CpuWeightStore::F32).unwrap();
    assert_eq!(f32_store.total_bytes, total * 128);
    let bf16_store = inspect_safetensors_dir_in_store(&dir, &cfg, CpuWeightStore::Bf16).unwrap();
    assert_eq!(
        bf16_store.total_bytes,
        (total - held_as_bits) * 128 + held_as_bits * 64
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn bf16_store_report_names_the_store_and_lowers_the_weight_estimate() {
    let dir = tempdir("report-bf16-store");
    let cfg = Qwen35Config::qwen35_0_8b();
    write_fixture(&dir, &bf16_fixture(&cfg));
    write_config_json(&dir, &cfg);
    fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let memory = Some(1u64 << 40);

    let f32_store = build_report_in_store(&dir, None, None, memory, CpuWeightStore::F32).unwrap();
    let bf16_store = build_report_in_store(&dir, None, None, memory, CpuWeightStore::Bf16).unwrap();
    assert!(f32_store.is_ready(), "{:?}", f32_store.blocking_reasons);
    assert!(bf16_store.is_ready(), "{:?}", bf16_store.blocking_reasons);
    assert!(bf16_store.weight_bytes < f32_store.weight_bytes);
    assert!(bf16_store.max_context_len >= f32_store.max_context_len);
    assert_eq!(bf16_store.cpu_weight_store, CpuWeightStore::Bf16);
    let line = "CPU weight store: bf16";
    assert!(bf16_store.to_string().contains(line));
    assert!(!f32_store.to_string().contains(line));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn bf16_store_is_a_blocking_reason_for_a_mixture_of_experts_checkpoint() {
    let dir = tempdir("report-bf16-moe");
    let cfg = Qwen35Config::qwen35_0_8b();
    write_fixture(&dir, &required_tensor_fixture(&cfg));
    write_config_json(&dir, &cfg);
    let path = dir.join("config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    config["text_config"]["num_experts"] = serde_json::json!(4);
    config["text_config"]["num_experts_per_tok"] = serde_json::json!(2);
    fs::write(&path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let refusal = "not measured for mixture-of-experts";

    let bf16_store =
        build_report_in_store(&dir, None, None, Some(1u64 << 40), CpuWeightStore::Bf16).unwrap();
    assert!(
        bf16_store
            .blocking_reasons
            .iter()
            .any(|reason| reason.contains(refusal)),
        "{:?}",
        bf16_store.blocking_reasons
    );
    let f32_store =
        build_report_in_store(&dir, None, None, Some(1u64 << 40), CpuWeightStore::F32).unwrap();
    assert!(
        !f32_store
            .blocking_reasons
            .iter()
            .any(|reason| reason.contains(refusal)),
        "{:?}",
        f32_store.blocking_reasons
    );
    fs::remove_dir_all(&dir).ok();
}
