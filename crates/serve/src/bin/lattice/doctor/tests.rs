use super::*;
use lattice_arch::qwen35_tensors::{
    FLAT_MODEL_PREFIX, LANGUAGE_MODEL_PREFIX, qwen_required_tensor_names,
};
use std::fs;

fn tempdir(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "lattice-doctor-test-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&dir).expect("create tempdir");
    #[cfg(unix)]
    set_fixture_permissions(&dir, 0o700);
    dir
}

#[cfg(unix)]
fn set_fixture_permissions(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}

// ---- human_bytes ----------------------------------------------

#[test]
fn human_bytes_formats_units() {
    assert_eq!(human_bytes(0), "0 B");
    assert_eq!(human_bytes(512), "512 B");
    assert_eq!(human_bytes(1024), "1.00 KiB");
    assert_eq!(human_bytes(1024 * 1024), "1.00 MiB");
    assert_eq!(human_bytes(1024 * 1024 * 1024), "1.00 GiB");
    assert_eq!(human_bytes(1536 * 1024 * 1024), "1.50 GiB");
}

// ---- plan_memory (pure math) ------------------------------------

#[test]
fn plan_memory_hand_computed() {
    // weight=100 bytes, kv=10 bytes/token, available=1000 bytes.
    // usable = 1000 - 100 = 900. max_by_memory = 900 / 10 = 90.
    let plan = plan_memory(100, 10, 1000, 1_000_000, Some(50));
    assert_eq!(plan.max_context_by_memory, 90);
    assert_eq!(plan.max_context_len, 90); // min(90, 1_000_000)
    assert_eq!(plan.requested_fits, Some(true)); // 50 <= 90

    let plan2 = plan_memory(100, 10, 1000, 1_000_000, Some(95));
    assert_eq!(plan2.requested_fits, Some(false)); // 95 > 90
}

#[test]
fn plan_memory_capped_by_max_position_embeddings() {
    // Effectively unlimited memory; the model architecture caps
    // context instead.
    let plan = plan_memory(0, 1, 1_000_000_000_000, 4096, None);
    assert_eq!(plan.max_context_by_memory, 1_000_000_000_000);
    assert_eq!(plan.max_context_len, 4096);
}

// ---- effective_max_position_embeddings (Medium-2 fix) -----------

#[test]
fn effective_max_position_embeddings_caps_metal_placement_at_runtime_limit() {
    // Qwen3.6-27B's real max_position_embeddings (131072) far
    // exceeds MetalChatBackend's actual 4096-token runtime cap --
    // doctor must not report a feasible context the binary would
    // refuse to serve.
    assert_eq!(
        effective_max_position_embeddings(Placement::Metal, 131_072),
        METAL_RUNTIME_MAX_CACHE_LEN
    );
}

#[test]
fn effective_max_position_embeddings_leaves_smaller_model_ceiling_untouched() {
    // A model whose own ceiling is already below the runtime cap
    // must not be inflated up to it.
    assert_eq!(
        effective_max_position_embeddings(Placement::Metal, 2048),
        2048
    );
}

#[test]
fn effective_max_position_embeddings_does_not_cap_cpu_placement() {
    // The CPU/safetensors path has no equivalent runtime cache cap.
    assert_eq!(
        effective_max_position_embeddings(Placement::Cpu, 131_072),
        131_072
    );
}

#[test]
fn plan_memory_weight_bytes_exceeding_available_yields_zero_context() {
    // Weights alone don't fit -- no room for any KV cache at all.
    let plan = plan_memory(2_000, 10, 1_000, 1_000_000, Some(1));
    assert_eq!(plan.max_context_by_memory, 0);
    assert_eq!(plan.max_context_len, 0);
    assert_eq!(plan.requested_fits, Some(false));
}

#[test]
fn plan_memory_kv_bytes_per_token_matches_qwen35_0_8b_doc_identity() {
    // Cross-check against `Qwen35Config::kv_bytes_per_token`'s own
    // doc-comment worked example: 6 full-attention layers * 2 (K+V)
    // * 512 full_kv_dim * 2 bytes (f16) = 12_288 bytes/token.
    let cfg = Qwen35Config::qwen35_0_8b();
    assert_eq!(cfg.num_full_attention_layers(), 6);
    assert_eq!(cfg.full_kv_dim(), 512);
    assert_eq!(cfg.kv_bytes_per_token(2), 12_288);
    assert_eq!(cfg.kv_bytes_per_token(4), 24_576);
}

// ---- safetensors header parsing ---------------------------------

fn write_fake_safetensors(path: &Path, tensors: &[(&str, &str, u64, u64)]) {
    let mut header = serde_json::Map::new();
    for (name, dtype, start, end) in tensors {
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": [1],
                "data_offsets": [start, end],
            }),
        );
    }
    let header_json = serde_json::Value::Object(header).to_string();
    let header_bytes = header_json.as_bytes();
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(header_bytes);
    let payload_len = tensors.iter().map(|(_, _, _, e)| *e).max().unwrap_or(0);
    buf.resize(buf.len() + payload_len as usize, 0);
    fs::write(path, buf).expect("write fake safetensors file");
}

#[test]
fn read_safetensors_header_computes_exact_byte_lengths() {
    let dir = tempdir("st-header");
    let path = dir.join("model.safetensors");
    write_fake_safetensors(
        &path,
        &[("tensor.a", "F32", 0, 400), ("tensor.b", "BF16", 400, 600)],
    );
    let tensors = read_safetensors_header(&path).unwrap();
    assert_eq!(tensors.len(), 2);
    assert_eq!(tensors["tensor.a"].byte_len, 400);
    assert_eq!(tensors["tensor.a"].dtype, "F32");
    assert_eq!(tensors["tensor.b"].byte_len, 200);
    assert_eq!(tensors["tensor.b"].dtype, "BF16");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn read_safetensors_header_rejects_oversized_header_length() {
    let dir = tempdir("st-header-bad");
    let path = dir.join("model.safetensors");
    // A header-length prefix claiming far more bytes than the file
    // actually has -- must fail, never attempt the allocation.
    let mut buf = Vec::new();
    buf.extend_from_slice(&(1_000_000_000u64).to_le_bytes());
    buf.extend_from_slice(b"tiny");
    fs::write(&path, buf).unwrap();
    let err = read_safetensors_header(&path).unwrap_err();
    assert!(err.contains("exceeds file size"));
    fs::remove_dir_all(&dir).ok();
}

fn required_tensor_fixture(cfg: &Qwen35Config) -> Vec<(String, String, u64, u64)> {
    let mut offset = 0u64;
    qwen_required_tensor_names(cfg)
        .into_iter()
        .map(|name| {
            let start = offset;
            offset += 64;
            (name, "F32".to_string(), start, offset)
        })
        .collect()
}

#[test]
fn inspect_safetensors_dir_all_tensors_present_no_missing() {
    let dir = tempdir("st-complete");
    let cfg = Qwen35Config::qwen35_0_8b();
    let tensors = required_tensor_fixture(&cfg);
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);

    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    assert!(inv.missing_tensors.is_empty());
    assert!(inv.unsupported_dtypes.is_empty());
    assert_eq!(inv.total_bytes, tensors.len() as u64 * 64);
    assert_eq!(inv.quantization, "F32");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_safetensors_dir_flat_names_are_complete() {
    let dir = tempdir("st-flat-complete");
    let cfg = Qwen35Config::qwen35_0_8b();
    let tensors: Vec<_> = required_tensor_fixture(&cfg)
        .into_iter()
        .map(|(name, dtype, start, end)| {
            (
                name.replacen("model.language_model.", "model.", 1),
                dtype,
                start,
                end,
            )
        })
        .collect();
    let refs: Vec<_> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);

    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    assert!(inv.missing_tensors.is_empty(), "{:?}", inv.missing_tensors);
    assert!(inv.unsupported_dtypes.is_empty());
    assert_eq!(inv.tensor_count, tensors.len());
    assert_eq!(inv.total_bytes, tensors.len() as u64 * 64);
    assert_eq!(inv.quantization, "F32");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inspect_safetensors_dir_reports_missing_embedding_anchors() {
    let dir = tempdir("st-missing-anchors");
    let cfg = Qwen35Config::qwen35_0_8b();
    let tensors: Vec<_> = required_tensor_fixture(&cfg)
        .into_iter()
        .filter(|(name, _, _, _)| name != LANGUAGE_MODEL_EMBED)
        .collect();
    let refs: Vec<_> = tensors
        .iter()
        .map(|(name, dtype, start, end)| (name.as_str(), dtype.as_str(), *start, *end))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);
    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    assert_eq!(
        inv.missing_tensors,
        ["model.language_model.embed_tokens.weight"]
    );
    assert_eq!(inv.tensor_count, tensors.len());
    assert_eq!(inv.total_bytes, tensors.len() as u64 * 64);
    assert_eq!(inv.quantization, "F32");
    assert!(inv.unsupported_dtypes.is_empty());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inspect_safetensors_dir_flat_checkpoint_missing_only_embedding() {
    let dir = tempdir("st-flat-missing-only-embedding");
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    let names = [
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];
    let tensors: Vec<_> = names
        .into_iter()
        .enumerate()
        .map(|(i, name)| (name, "F32", i as u64 * 64, (i as u64 + 1) * 64))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &tensors);

    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.missing_tensors, ["model.embed_tokens.weight"]);
    assert_eq!(inv.tensor_count, 12);
    assert_eq!(inv.total_bytes, 12 * 64);
    assert_eq!(inv.quantization, "F32");
    assert!(inv.unsupported_dtypes.is_empty());
    assert!(inv.quantization_error.is_none());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inspect_q4_manifest_reports_missing_embedding_anchors() {
    let dir = tempdir("q4-missing-anchors");
    write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 1);
    let entries = serde_json::json!([
        {"name": "model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4"}
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&entries).unwrap(),
    )
    .unwrap();
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.missing_tensors.len(), 12);
    assert!(
        inv.missing_tensors
            .iter()
            .any(|name| name == "model.embed_tokens.weight")
    );
    assert!(
        inv.missing_tensors
            .iter()
            .any(|name| name == "model.norm.weight")
    );
    assert!(
        !inv.missing_tensors
            .iter()
            .any(|name| name == "model.layers.0.self_attn.q_proj.weight")
    );
    assert_eq!(inv.tensor_count, 1);
    assert_eq!(inv.total_bytes, 48);
    assert_eq!(inv.quantization, "Q4_0 (lattice native, v2 asymmetric)");
    assert!(inv.quantization_error.is_none());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn required_inventory_tensor_names_refuses_ambiguous_embedding_anchors() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let err = required_inventory_tensor_names(
        &cfg,
        [
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ],
    )
    .unwrap_err();
    assert!(err.contains("both"), "{err}");
    assert!(
        err.contains("model.language_model.embed_tokens.weight"),
        "{err}"
    );
    assert!(err.contains("model.embed_tokens.weight"), "{err}");
}

#[test]
fn inspect_safetensors_dir_preserves_diagnostics_without_embedding_anchors() {
    let dir = tempdir("st-missing-anchor-diagnostics");
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    let unsupported_name = "model.language_model.layers.0.input_layernorm.weight";
    write_fake_safetensors(
        &dir.join("model.safetensors"),
        &[
            (unsupported_name, "I64", 0, 24),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                "F16",
                24,
                40,
            ),
            ("unused.weight", "BF16", 40, 72),
        ],
    );

    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.tensor_count, 3);
    assert_eq!(inv.total_bytes, 24 + 2 * 16 + 2 * 32);
    assert_eq!(inv.quantization, "BF16, F16, I64");
    assert_eq!(inv.missing_tensors.len(), 11);
    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    ] {
        assert!(inv.missing_tensors.iter().any(|missing| missing == name));
    }
    assert_eq!(
        inv.unsupported_dtypes,
        [format!(
            "tensor '{unsupported_name}' has dtype I64, which is not supported (supported: F32, F16, BF16)"
        )]
    );
    assert!(inv.quantization_error.is_none());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inspect_q4_manifest_preserves_diagnostics_without_embedding_anchors() {
    let dir = tempdir("q4-missing-anchor-diagnostics");
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    write_fake_q4_file(&dir.join("q_proj.q4"), 1, &[32], 32, 1);
    let entries = serde_json::json!([
        {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4"},
        {"name": "model.language_model.norm.weight", "file": "missing.f16"},
        {"name": "model.language_model.layers.0.mlp.gate_proj.weight", "file": "missing_gate.q4"}
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&entries).unwrap(),
    )
    .unwrap();

    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.tensor_count, 3);
    assert_eq!(inv.total_bytes, 48);
    assert_eq!(inv.quantization, "unknown (see blocking reasons)");
    let error = inv.quantization_error.as_deref().unwrap();
    assert!(error.contains("unsupported quantization scheme"), "{error}");
    assert!(error.contains("q_proj.q4"), "{error}");
    assert_eq!(inv.missing_tensors.len(), 14);
    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.norm.weight (listed in quantize_index.json as 'missing.f16', file not found)",
        "model.language_model.layers.0.mlp.gate_proj.weight (listed in quantize_index.json as 'missing_gate.q4', file not found)",
    ] {
        assert!(inv.missing_tensors.iter().any(|missing| missing == name));
    }
    assert!(inv.unsupported_dtypes.is_empty());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inspect_q4_manifest_counts_inspected_anchor_metadata() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    for (anchor_name, prefix) in [
        (LANGUAGE_MODEL_EMBED, LANGUAGE_MODEL_PREFIX),
        (FLAT_MODEL_EMBED, FLAT_MODEL_PREFIX),
    ] {
        let dir = tempdir("q4-anchor-metadata-size");
        let anchor_file = format!("{}.q4", anchor_name.replace('.', "_"));
        write_fake_q4_file(&dir.join(&anchor_file), 2, &[32], 32, 1);
        write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[64], 64, 2);
        let ordinary_entry = serde_json::json!({
            "name": format!("{prefix}layers.0.self_attn.q_proj.weight"),
            "file": "q_proj.q4",
        });
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&serde_json::json!([ordinary_entry.clone()])).unwrap(),
        )
        .unwrap();
        let ordinary = inspect_q4_dir(&dir, &cfg).unwrap();
        assert_eq!(ordinary.total_bytes, 68);
        assert_eq!(ordinary.tensor_count, 1);

        let entries = serde_json::json!([
            {"name": anchor_name, "file": anchor_file},
            ordinary_entry,
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&entries).unwrap(),
        )
        .unwrap();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert_eq!(inv.total_bytes - ordinary.total_bytes, 48, "{prefix}");
        assert_eq!(inv.total_bytes, 48 + 68, "{prefix}");
        assert_eq!(inv.tensor_count, 2);
        assert_eq!(inv.missing_tensors.len(), 11);
        assert!(!inv.missing_tensors.iter().any(|name| name == anchor_name));
        assert!(inv.missing_tensors.iter().all(|name| {
            name.starts_with(prefix) && !name.contains("listed in quantize_index.json")
        }));
        assert_eq!(inv.quantization, "Q4_0 (lattice native, v2 asymmetric)");
        assert!(inv.quantization_error.is_none());
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn inspect_q4_manifest_warns_for_aliases_and_counts_native_anchors() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum NativeAnchorState {
        Missing,
        Directory,
        Regular,
    }

    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    cfg.full_attention_interval = 1;
    for (anchor_name, expected_file, prefix) in [
        (
            "model.language_model.embed_tokens.weight",
            "model_language_model_embed_tokens_weight.q4",
            "model.language_model.",
        ),
        (
            "model.embed_tokens.weight",
            "model_embed_tokens_weight.q4",
            "model.",
        ),
    ] {
        let dir = tempdir("q4-embedding-filename-alias");
        write_config_json(&dir, &cfg);
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        write_fake_q4_file(&dir.join("alias.q4"), 2, &[32], 32, 1);
        write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[64], 64, 2);
        let q_proj_name = format!("{prefix}layers.0.self_attn.q_proj.weight");
        let mut entries = serde_json::json!([
            {"name": anchor_name, "file": "alias.q4"},
            {"name": q_proj_name, "file": "q_proj.q4"},
        ]);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&entries).unwrap(),
        )
        .unwrap();
        assert!(!dir.join(expected_file).exists());

        let warning = format!(
            "{anchor_name}: manifest alias 'alias.q4' ignored; the loader reads '{expected_file}'"
        );
        for (native_kind, expected_bytes, missing_count) in [
            (NativeAnchorState::Missing, 68, 13),
            (NativeAnchorState::Directory, 68, 13),
            (NativeAnchorState::Regular, 156, 11),
        ] {
            match native_kind {
                NativeAnchorState::Directory => fs::create_dir(dir.join(expected_file)).unwrap(),
                NativeAnchorState::Regular => {
                    fs::remove_dir(dir.join(expected_file)).unwrap();
                    write_fake_q4_file(&dir.join(expected_file), 2, &[96], 96, 3);
                }
                NativeAnchorState::Missing => {}
            }
            let inv = inspect_q4_dir(&dir, &cfg).unwrap();
            assert_eq!(
                inv.total_bytes, expected_bytes,
                "{anchor_name}: {native_kind:?}"
            );
            assert_eq!(inv.tensor_count, 2);
            assert_eq!(inv.missing_tensors.len(), missing_count);
            assert_eq!(inv.warnings, std::slice::from_ref(&warning));
            assert_eq!(
                inv.missing_tensors.iter().any(|name| name == anchor_name),
                native_kind != NativeAnchorState::Regular
            );
            if native_kind == NativeAnchorState::Regular {
                assert!(
                    inv.missing_tensors
                        .iter()
                        .all(|name| !name.contains("listed in"))
                );
            } else {
                let reason = if native_kind == NativeAnchorState::Missing {
                    format!("native embedding file '{expected_file}' not found")
                } else {
                    format!(
                        "Q4 embedding anchor {} is not a regular file",
                        dir.join(expected_file).display()
                    )
                };
                assert!(inv.missing_tensors.contains(&format!(
                    "{anchor_name} (listed in quantize_index.json as 'alias.q4', {reason})"
                )));
            }
            assert!(
                inv.missing_tensors
                    .contains(&format!("{prefix}norm.weight"))
            );
            assert!(!inv.missing_tensors.contains(&q_proj_name));
            assert_eq!(inv.quantization, "Q4_0 (lattice native, v2 asymmetric)");
            assert!(inv.quantization_error.is_none());
            assert!(inv.unsupported_dtypes.is_empty());
            let report = build_report(&dir, None, None, Some(1u64 << 40)).unwrap();
            assert_eq!(report.weight_bytes, expected_bytes);
            assert_eq!(report.missing_tensors, inv.missing_tensors);
            assert_eq!(report.warnings, inv.warnings);
            assert!(format!("{report}").contains(&format!("Warning: {warning}")));
            assert!(!report.is_ready(), "other required tensors remain missing");
        }

        let alias_report = build_report(&dir, None, None, Some(1u64 << 40)).unwrap();
        entries[0]["file"] = serde_json::json!(expected_file);
        fs::write(
            dir.join("quantize_index.json"),
            serde_json::to_vec(&entries).unwrap(),
        )
        .unwrap();
        let inv = inspect_q4_dir(&dir, &cfg).unwrap();
        assert_eq!(inv.total_bytes, 88 + 68);
        assert_eq!(inv.tensor_count, 2);
        assert_eq!(inv.missing_tensors.len(), 11);
        assert!(inv.warnings.is_empty());
        assert!(!inv.missing_tensors.iter().any(|name| name == anchor_name));
        assert!(!inv.missing_tensors.contains(&q_proj_name));
        assert!(inv.missing_tensors.iter().all(|name| {
            name.starts_with(prefix) && !name.contains("listed in quantize_index.json")
        }));
        assert_eq!(inv.quantization, "Q4_0 (lattice native, v2 asymmetric)");
        assert!(inv.quantization_error.is_none());
        let report = build_report(&dir, None, None, Some(1u64 << 40)).unwrap();
        assert!(report.warnings.is_empty());
        assert_eq!(report.blocking_reasons, alias_report.blocking_reasons);
        assert_eq!(report.is_ready(), alias_report.is_ready());
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn inspect_q4_manifest_reports_directory_anchors_and_keeps_inventory() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    for (invalid_name, opposite_name, opposite_prefix) in [
        (LANGUAGE_MODEL_EMBED, FLAT_MODEL_EMBED, FLAT_MODEL_PREFIX),
        (
            FLAT_MODEL_EMBED,
            LANGUAGE_MODEL_EMBED,
            LANGUAGE_MODEL_PREFIX,
        ),
    ] {
        for keep_opposite in [false, true] {
            let dir = tempdir("q4-directory-anchor");
            let prefix = if keep_opposite {
                opposite_prefix
            } else {
                LANGUAGE_MODEL_PREFIX
            };
            let anchor_file = format!("{}.q4", invalid_name.replace('.', "_"));
            let anchor = dir.join(&anchor_file);
            write_fake_q4_file(&anchor, 2, &[32], 32, 1);
            write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 1);
            let mut entries = vec![
                serde_json::json!({"name": invalid_name, "file": anchor_file}),
                serde_json::json!({"name": format!("{prefix}layers.0.self_attn.q_proj.weight"), "file": "q_proj.q4"}),
                serde_json::json!({"name": format!("{prefix}norm.weight"), "file": "missing.f16"}),
            ];
            if keep_opposite {
                let opposite_file = format!("{}.q4", opposite_name.replace('.', "_"));
                write_fake_q4_file(&dir.join(&opposite_file), 2, &[32], 32, 1);
                entries.push(serde_json::json!({"name": opposite_name, "file": opposite_file}));
            }
            fs::write(
                dir.join("quantize_index.json"),
                serde_json::to_vec(&entries).unwrap(),
            )
            .unwrap();
            if keep_opposite {
                let error = inspect_q4_dir(&dir, &cfg)
                    .err()
                    .expect("two regular embedding anchors must remain ambiguous");
                assert!(error.contains("checkpoint contains both"), "{error}");
            } else {
                let regular = inspect_q4_dir(&dir, &cfg).unwrap();
                assert_eq!(regular.total_bytes, 96);
                assert_eq!(regular.quantization, "Q4_0 (lattice native, v2 asymmetric)");
                assert!(regular.quantization_error.is_none());
            }

            fs::remove_file(&anchor).unwrap();
            fs::create_dir(&anchor).unwrap();
            let inv = inspect_q4_dir(&dir, &cfg).unwrap();
            assert_eq!(inv.tensor_count, if keep_opposite { 4 } else { 3 });
            assert_eq!(inv.total_bytes, if keep_opposite { 96 } else { 48 });
            assert_eq!(inv.quantization, "Q4_0 (lattice native, v2 asymmetric)");
            assert!(inv.quantization_error.is_none());
            assert_eq!(
                inv.missing_tensors.len(),
                if keep_opposite { 13 } else { 14 }
            );
            for diagnostic in [
                format!(
                    "{invalid_name} (listed in quantize_index.json as '{anchor_file}', Q4 embedding anchor {} is not a regular file)",
                    anchor.display()
                ),
                format!(
                    "{prefix}norm.weight (listed in quantize_index.json as 'missing.f16', file not found)"
                ),
                format!("{prefix}norm.weight"),
            ] {
                assert!(inv.missing_tensors.contains(&diagnostic), "{diagnostic}");
            }
            assert_eq!(
                inv.missing_tensors
                    .contains(&format!("{prefix}embed_tokens.weight")),
                !keep_opposite
            );
            fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[test]
fn inspect_q4_manifest_reports_both_invalid_anchors_and_quantization_error() {
    let dir = tempdir("q4-both-directory-anchors");
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    fs::create_dir(dir.join("model_language_model_embed_tokens_weight.q4")).unwrap();
    fs::create_dir(dir.join("model_embed_tokens_weight.q4")).unwrap();
    write_fake_q4_file(&dir.join("q_proj.q4"), 1, &[32], 32, 1);
    let entries = serde_json::json!([
        {"name": LANGUAGE_MODEL_EMBED, "file": "model_language_model_embed_tokens_weight.q4"},
        {"name": FLAT_MODEL_EMBED, "file": "model_embed_tokens_weight.q4"},
        {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4"},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&entries).unwrap(),
    )
    .unwrap();

    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.tensor_count, 3);
    assert_eq!(inv.total_bytes, 48);
    assert_eq!(inv.missing_tensors.len(), 14);
    for (name, file) in [
        (
            LANGUAGE_MODEL_EMBED,
            "model_language_model_embed_tokens_weight.q4",
        ),
        (FLAT_MODEL_EMBED, "model_embed_tokens_weight.q4"),
    ] {
        assert!(inv.missing_tensors.contains(&format!(
            "{name} (listed in quantize_index.json as '{file}', Q4 embedding anchor {} is not a regular file)",
            dir.join(file).display()
        )));
    }
    assert!(
        inv.missing_tensors
            .contains(&LANGUAGE_MODEL_EMBED.to_string())
    );
    assert!(
        inv.missing_tensors
            .contains(&"model.language_model.norm.weight".to_string())
    );
    assert_eq!(inv.quantization, "unknown (see blocking reasons)");
    let error = inv.quantization_error.as_deref().unwrap();
    let directory_error = [
        "model_language_model_embed_tokens_weight.q4",
        "model_embed_tokens_weight.q4",
    ]
    .iter()
    .any(|name| {
        let path = dir.join(name);
        error.starts_with(&format!("failed to open {}: ", path.display()))
            || error.starts_with(&format!(
                "unsupported quantization scheme in {}: ",
                path.display()
            ))
    });
    assert!(
        directory_error
            || error.starts_with(&format!(
                "unsupported quantization scheme in {}: ",
                dir.join("q_proj.q4").display()
            )),
        "{error}"
    );

    // Leave one sampling candidate to pin the legacy-file diagnostic independently.
    fs::remove_dir(dir.join("model_language_model_embed_tokens_weight.q4")).unwrap();
    fs::remove_dir(dir.join("model_embed_tokens_weight.q4")).unwrap();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.quantization, "unknown (see blocking reasons)");
    let error = inv.quantization_error.as_deref().unwrap();
    assert!(error.contains("unsupported quantization scheme"), "{error}");
    assert!(error.contains("q_proj.q4"), "{error}");
    fs::remove_dir_all(dir).unwrap();
}

fn complete_q4_namespace_inventory(prefix: &str) {
    let dir = tempdir("q4-namespace-complete");
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    const VECTOR_SUFFIXES: [&str; 5] = [
        "norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
    ];
    let mut entries = Vec::new();
    for canonical in qwen_required_tensor_names(&cfg) {
        let suffix = canonical.strip_prefix(LANGUAGE_MODEL_PREFIX).unwrap();
        let name = format!("{prefix}{suffix}");
        let is_vector = VECTOR_SUFFIXES.contains(&suffix);
        let ext = if is_vector { "f16" } else { "q4" };
        let file = format!("{}.{ext}", name.replace('.', "_"));
        if is_vector {
            let mut bytes = b"KHF1".to_vec();
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&32u64.to_le_bytes());
            bytes.extend_from_slice(&32u64.to_le_bytes());
            for _ in 0..32 {
                bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
            }
            fs::write(dir.join(&file), bytes).unwrap();
        } else {
            write_fake_q4_file(&dir.join(&file), 2, &[32, 32], 1024, 32);
        }
        entries.push(serde_json::json!({"name": name, "file": file}));
    }
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&entries).unwrap(),
    )
    .unwrap();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert!(inv.missing_tensors.is_empty(), "{:?}", inv.missing_tensors);
    assert!(inv.quantization_error.is_none());
    assert!(inv.unsupported_dtypes.is_empty());
    let matrix_count = 1 + 4 + 3; // embedding, attention projections, MLP projections
    let vector_count = VECTOR_SUFFIXES.len();
    let tensor_count = matrix_count + vector_count;
    assert_eq!(tensor_count, 13);
    assert_eq!(inv.tensor_count, tensor_count);
    let header_prefix_bytes = 4 + 4 + 4;
    let dimension_bytes = 8;
    let element_count_bytes = 8;
    let matrix_header_bytes = header_prefix_bytes + 2 * dimension_bytes + element_count_bytes;
    let vector_header_bytes = header_prefix_bytes + dimension_bytes + element_count_bytes;
    let q4_block_bytes = 2 + 2 + 16;
    let q4_block_elements = 32;
    let matrix_elements = 32 * 32;
    let q4_file_bytes =
        matrix_header_bytes + (matrix_elements / q4_block_elements) * q4_block_bytes;
    let vector_elements = 32;
    let f16_bytes = 2;
    let f32_bytes = 4;
    // The resident estimator scales the whole F16 file, including its header.
    let f16_resident_bytes =
        (vector_header_bytes + vector_elements * f16_bytes) * (f32_bytes / f16_bytes);
    assert_eq!(q4_file_bytes, 676);
    assert_eq!(f16_resident_bytes, 184);
    assert_eq!(
        inv.total_bytes,
        matrix_count as u64 * q4_file_bytes + vector_count as u64 * f16_resident_bytes
    );
    assert!(!inv.has_mtp_tensors);

    let missing_name = format!("{prefix}norm.weight");
    let missing_file = format!("{}.f16", missing_name.replace('.', "_"));
    fs::remove_file(dir.join(&missing_file)).unwrap();
    let missing = inspect_q4_dir(&dir, &cfg).unwrap();
    assert_eq!(missing.missing_tensors.len(), 2);
    assert!(missing.missing_tensors.contains(&missing_name));
    assert!(
        missing
            .missing_tensors
            .iter()
            .any(|entry| entry.contains(&missing_file) && entry.contains("file not found"))
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inspect_q4_dir_flat_manifest_names_are_complete() {
    complete_q4_namespace_inventory(FLAT_MODEL_PREFIX);
}

#[test]
fn inspect_q4_dir_canonical_manifest_names_remain_complete() {
    complete_q4_namespace_inventory(LANGUAGE_MODEL_PREFIX);
}

#[test]
fn inspect_safetensors_dir_scales_f16_bf16_to_resident_f32_bytes() {
    // The CPU loader always materializes weights as owned f32,
    // converting F16/BF16 on load (see `cpu_resident_bytes`'s doc
    // comment) -- so a 2-byte-per-element on-disk tensor must count
    // double toward the RAM budget, while F32 tensors (already 4
    // bytes/elem) must not be scaled.
    let dir = tempdir("st-mixed-dtype");
    let cfg = Qwen35Config::qwen35_0_8b();
    let mut tensors = required_tensor_fixture(&cfg);
    assert!(
        tensors.len() >= 2,
        "fixture must have room to mutate two entries"
    );
    tensors[0].1 = "F16".to_string();
    tensors[1].1 = "BF16".to_string();
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);

    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    // Every tensor is 64 on-disk bytes; the two F16/BF16 entries
    // must count as 128 resident bytes each, the rest stay at 64.
    let expected = (tensors.len() as u64 - 2) * 64 + 2 * 128;
    assert_eq!(inv.total_bytes, expected);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_safetensors_dir_detects_missing_tensor() {
    let dir = tempdir("st-missing");
    let cfg = Qwen35Config::qwen35_0_8b();
    let mut tensors = required_tensor_fixture(&cfg);
    let (dropped_name, _, _, _) = tensors.pop().unwrap();
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);

    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.missing_tensors, vec![dropped_name]);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_safetensors_dir_flags_unsupported_dtype() {
    let dir = tempdir("st-baddtype");
    let cfg = Qwen35Config::qwen35_0_8b();
    let mut tensors = required_tensor_fixture(&cfg);
    // Corrupt one required tensor's dtype to something unsupported.
    tensors[0].1 = "I64".to_string();
    let bad_name = tensors[0].0.clone();
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);

    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.unsupported_dtypes.len(), 1);
    assert!(inv.unsupported_dtypes[0].contains(&bad_name));
    assert!(inv.unsupported_dtypes[0].contains("I64"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_safetensors_dir_prefers_single_file_over_index() {
    // Mirrors `Qwen35Model::from_safetensors`'s precedence: when
    // both `model.safetensors` and `model.safetensors.index.json`
    // exist, the single file wins and the index is never consulted.
    let dir = tempdir("st-precedence");
    let cfg = Qwen35Config::qwen35_0_8b();
    let tensors = required_tensor_fixture(&cfg);
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);
    // A bogus index.json that would error if it were ever read.
    fs::write(dir.join("model.safetensors.index.json"), b"not valid json").unwrap();

    let inv = inspect_safetensors_dir(&dir, &cfg).unwrap();
    assert!(inv.missing_tensors.is_empty());
    fs::remove_dir_all(&dir).ok();
}

// ---- Q4 directory inventory --------------------------------------

fn write_fake_q4_file(
    path: &Path,
    version: u32,
    shape: &[u64],
    original_len: u64,
    n_blocks: usize,
) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHQ4");
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for s in shape {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf.extend_from_slice(&original_len.to_le_bytes());
    buf.resize(buf.len() + n_blocks * 20, 0);
    fs::write(path, buf).expect("write fake q4 file");
    #[cfg(unix)]
    set_fixture_permissions(path, 0o600);
}

fn write_q4_embedding_fixture(dir: &Path) -> serde_json::Value {
    write_fake_q4_file(
        &dir.join("model_language_model_embed_tokens_weight.q4"),
        2,
        &[32],
        32,
        1,
    );
    serde_json::json!({
        "name": LANGUAGE_MODEL_EMBED,
        "file": "model_language_model_embed_tokens_weight.q4",
        "shape": [32],
        "numel": 32,
    })
}

#[test]
fn detect_q4_quantization_label_accepts_v2_file() {
    let dir = tempdir("q4-label-v2");
    write_fake_q4_file(&dir.join("sample.q4"), 2, &[32], 32, 1);
    let label = detect_q4_quantization_label(&dir).unwrap();
    assert!(label.contains("Q4_0"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_q4_quantization_label_rejects_legacy_v1_file() {
    let dir = tempdir("q4-label-v1");
    write_fake_q4_file(&dir.join("sample.q4"), 1, &[32], 32, 1);
    let err = detect_q4_quantization_label(&dir).unwrap_err();
    assert!(err.contains("legacy"));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn q4_quantization_label_reads_the_provided_handle() {
    let dir = tempdir("q4-provided-sample-handle");
    let valid = dir.join("valid.q4");
    let invalid = dir.join("invalid.q4");
    write_fake_q4_file(&valid, 2, &[32], 32, 1);
    fs::write(&invalid, b"ordinary invalid Q4 header").unwrap();
    #[cfg(unix)]
    set_fixture_permissions(&invalid, 0o600);
    let sample = Q4QuantizationSample {
        path: invalid.clone(),
        file: open_trusted_mmap_file(&valid).unwrap(),
    };
    assert_eq!(
        q4_quantization_label_from_sample(sample).unwrap(),
        "Q4_0 (lattice native, v2 asymmetric)"
    );

    let sample = Q4QuantizationSample {
        path: valid.clone(),
        file: open_trusted_mmap_file(&invalid).unwrap(),
    };
    let error = q4_quantization_label_from_sample(sample).unwrap_err();
    assert!(error.starts_with(&format!(
        "unsupported quantization scheme in {}:",
        valid.display()
    )));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn q4_quantization_sample_retains_directory_refusal_and_excludes_cache_files() {
    let dir = tempdir("q4-sample-file-kinds");
    let directory = dir.join("00-directory.q4");
    fs::create_dir(&directory).unwrap();
    write_fake_q4_file(&dir.join("merged_qkvz_00.q4"), 2, &[32], 32, 1);
    fs::write(dir.join("notes.txt"), b"not a Q4 candidate").unwrap();
    let expected = open_trusted_mmap_file(&directory).unwrap_err();
    let error = select_q4_quantization_sample(&dir).err().unwrap();
    assert_eq!(error, expected);

    let regular = dir.join("99-regular.q4");
    write_fake_q4_file(&regular, 2, &[32], 32, 1);
    assert_eq!(
        select_q4_quantization_sample(&dir)
            .unwrap()
            .map(|sample| sample.path),
        Some(regular)
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[test]
fn q4_quantization_sample_refuses_links_and_unix_special_entries() {
    use std::os::unix::fs::{FileTypeExt, symlink};

    let dir = tempdir("q4-sample-unix-file-kinds");
    let target = dir.join("regular-target.bin");
    write_fake_q4_file(&target, 2, &[32], 32, 1);
    let directory = dir.join("directory-target");
    fs::create_dir(&directory).unwrap();
    let fifo = dir.join("03-pipe.q4");
    let status = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .expect("mkfifo is required for the Unix sample-opening fixture");
    assert!(status.success(), "mkfifo failed: {status}");
    assert!(fs::symlink_metadata(&fifo).unwrap().file_type().is_fifo());
    let link = |link_name: &str, referent: &Path| {
        let link = dir.join(link_name);
        symlink(referent, &link).unwrap();
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        link
    };
    let directory_link = link("01-directory-link.q4", &directory);
    link("02-dangling-link.q4", &dir.join("missing-target"));
    link("04-pipe-link.q4", &fifo);
    let error = select_q4_quantization_sample(&dir).err().unwrap();
    assert!(error.contains(&directory_link.display().to_string()));
    assert!(error.contains("final path component is a symlink"));

    let regular_link = link("00-regular-link.q4", &target);
    let error = select_q4_quantization_sample(&dir).err().unwrap();
    assert!(error.contains(&regular_link.display().to_string()));
    assert!(error.contains("final path component is a symlink"));

    let regular = dir.join("99-regular.q4");
    write_fake_q4_file(&regular, 2, &[32], 32, 1);
    let sample = select_q4_quantization_sample(&dir).unwrap().unwrap();
    assert_eq!(sample.path, regular);
    assert_eq!(
        q4_quantization_label_from_sample(sample).unwrap(),
        "Q4_0 (lattice native, v2 asymmetric)"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[cfg(unix)]
#[test]
fn q4_quantization_sample_retains_regular_file_trust_refusal() {
    let dir = tempdir("q4-sample-file-trust");
    let untrusted = dir.join("00-untrusted.q4");
    write_fake_q4_file(&untrusted, 2, &[32], 32, 1);
    set_fixture_permissions(&untrusted, 0o620);
    let error = select_q4_quantization_sample(&dir).err().unwrap();
    assert!(error.contains(&untrusted.display().to_string()));
    assert!(error.contains("outside lattice's trust boundary"));

    let regular = dir.join("99-regular.q4");
    write_fake_q4_file(&regular, 2, &[32], 32, 1);
    let sample = select_q4_quantization_sample(&dir).unwrap().unwrap();
    assert_eq!(sample.path, regular);
    assert_eq!(
        q4_quantization_label_from_sample(sample).unwrap(),
        "Q4_0 (lattice native, v2 asymmetric)"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn detect_q4_quantization_label_uses_sorted_regular_sample() {
    let dir = tempdir("q4-sample-order");
    let earlier = dir.join("10-legacy.q4");
    let later = dir.join("20-current.q4");
    write_fake_q4_file(&later, 2, &[32], 32, 1);
    write_fake_q4_file(&earlier, 1, &[32], 32, 1);
    assert_eq!(
        select_q4_quantization_sample(&dir)
            .unwrap()
            .map(|sample| sample.path),
        Some(earlier.clone())
    );
    let error = detect_q4_quantization_label(&dir).unwrap_err();
    assert!(
        error.starts_with(&format!(
            "unsupported quantization scheme in {}:",
            earlier.display()
        )),
        "{error}"
    );
    assert!(error.contains("legacy"), "{error}");

    fs::remove_file(earlier).unwrap();
    assert_eq!(
        select_q4_quantization_sample(&dir)
            .unwrap()
            .map(|sample| sample.path),
        Some(later)
    );
    assert_eq!(
        detect_q4_quantization_label(&dir).unwrap(),
        "Q4_0 (lattice native, v2 asymmetric)"
    );
    fs::remove_dir_all(dir).unwrap();
}

fn write_q4_filename_policy_fixture(dir: &Path) {
    write_fake_q4_file(&dir.join("z.q4"), 2, &[32], 32, 1);
    write_fake_q4_file(&dir.join("merged_qkvz_0.q4"), 2, &[64], 64, 2);
    write_fake_q4_file(&dir.join("0.Q4"), 2, &[32], 32, 1);
    let mut half = b"KHF1".to_vec();
    half.extend_from_slice(&1u32.to_le_bytes());
    half.extend_from_slice(&1u32.to_le_bytes());
    half.extend_from_slice(&4u64.to_le_bytes());
    half.extend_from_slice(&4u64.to_le_bytes());
    half.extend_from_slice(&[0u8; 8]);
    for name in ["a_norm_weight.f16", "merged_qkvz_0.f16", "0.F16"] {
        fs::write(dir.join(name), &half).unwrap();
    }
}

#[test]
fn q4_sample_filename_policy_excludes_caches_and_f16() {
    let dir = tempdir("q4-sample-filename-policy");
    write_q4_filename_policy_fixture(&dir);
    assert_eq!(
        select_q4_quantization_sample(&dir)
            .unwrap()
            .map(|sample| sample.path),
        Some(dir.join("z.q4"))
    );
    assert_eq!(
        detect_q4_quantization_label(&dir).unwrap(),
        "Q4_0 (lattice native, v2 asymmetric)"
    );

    let mixed_case = dir.join("Merged_qkvz_0.q4");
    write_fake_q4_file(&mixed_case, 2, &[32], 32, 1);
    assert_eq!(
        select_q4_quantization_sample(&dir)
            .unwrap()
            .map(|sample| sample.path),
        Some(mixed_case)
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn q4_fallback_filename_policy_counts_f16_and_excludes_all_caches() {
    let dir = tempdir("q4-inventory-filename-policy");
    write_q4_filename_policy_fixture(&dir);
    assert!(!dir.join("quantize_index.json").exists());
    let cfg = Qwen35Config::qwen35_0_8b();
    let inventory = inspect_q4_dir(&dir, &cfg).unwrap();
    assert_eq!(inventory.tensor_count, 2);
    assert_eq!(inventory.total_bytes, 120); // Q4: 48; F16 norm: 36 × 2.
    assert_eq!(
        inventory.quantization,
        "Q4_0 (lattice native, v2 asymmetric)"
    );
    assert!(inventory.quantization_error.is_none());
    assert!(inventory.missing_tensors.is_empty());
    assert!(inventory.warnings.is_empty());

    write_fake_q4_file(&dir.join("Merged_qkvz_0.q4"), 2, &[32], 32, 1);
    let inventory = inspect_q4_dir(&dir, &cfg).unwrap();
    assert_eq!(inventory.tensor_count, 3);
    assert_eq!(inventory.total_bytes, 168);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn inspect_q4_dir_uses_index_manifest_and_ignores_merged_cache_files() {
    let dir = tempdir("q4-merge-safe");
    write_fake_q4_file(&dir.join("layer0_qkv.q4"), 2, &[32], 32, 1);
    write_fake_q4_file(&dir.join("layer0_z.q4"), 2, &[32], 32, 1);
    // A runtime-created merge cache duplicating the two tensors
    // above -- must NOT be double-counted *from its own file* (it
    // has no manifest entry), but its bytes ARE real resident
    // duplicates of in_proj_qkv/in_proj_z, so each of those two
    // entries counts twice (see `q4_resident_bytes`).
    write_fake_q4_file(&dir.join("merged_qkvz_0_100_50.q4"), 2, &[64], 64, 2);

    let qkv_len = fs::metadata(dir.join("layer0_qkv.q4")).unwrap().len();
    let z_len = fs::metadata(dir.join("layer0_z.q4")).unwrap().len();

    let embedding_entry = write_q4_embedding_fixture(&dir);
    let index = serde_json::json!([
        embedding_entry,
        {"name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "file": "layer0_qkv.q4", "quantized": true, "shape": [32], "numel": 32},
        {"name": "model.language_model.layers.0.linear_attn.in_proj_z.weight", "file": "layer0_z.q4", "quantized": true, "shape": [32], "numel": 32},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    let embed_len = fs::metadata(dir.join("model_language_model_embed_tokens_weight.q4"))
        .unwrap()
        .len();
    assert_eq!(inv.total_bytes, embed_len + 2 * (qkv_len + z_len));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_q4_dir_resident_bytes_exceed_raw_on_disk_sum_for_expanded_tensors() {
    // What this test guards: several Q4/Metal tensor
    // categories are dequantized or duplicated at load time, so a
    // flat `meta.len()` sum under-reports real resident bytes. This
    // must fail if `inspect_q4_dir` reverts to summing raw on-disk
    // sizes unmodified.
    let dir = tempdir("q4-resident-expansion");
    // embed_tokens: expect 1x (mapped in place; the tied GPU logits
    // path maps the same file again and shares its pages).
    write_fake_q4_file(
        &dir.join("model_language_model_embed_tokens_weight.q4"),
        2,
        &[32],
        32,
        10, // 10 blocks * 20 bytes = 200 on-disk bytes
    );
    // in_proj_a: expect 3.2x (dequantized to an f16 Metal buffer).
    write_fake_q4_file(&dir.join("proj_a.q4"), 2, &[32], 32, 10);
    // A plain mmap'd tensor (q_proj): expect exactly 1x, unchanged.
    write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 10);

    let embed_len = fs::metadata(dir.join("model_language_model_embed_tokens_weight.q4"))
        .unwrap()
        .len();
    let proj_a_len = fs::metadata(dir.join("proj_a.q4")).unwrap().len();
    let q_proj_len = fs::metadata(dir.join("q_proj.q4")).unwrap().len();

    let index = serde_json::json!([
        {"name": "model.language_model.embed_tokens.weight", "file": "model_language_model_embed_tokens_weight.q4", "quantized": true, "shape": [32], "numel": 32},
        {"name": "model.language_model.layers.0.linear_attn.in_proj_a.weight", "file": "proj_a.q4", "quantized": true, "shape": [32], "numel": 32},
        {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();

    let naive_sum = embed_len + proj_a_len + q_proj_len;
    let expected = embed_len + (proj_a_len as f64 * 3.2).round() as u64 + q_proj_len;
    assert!(
        inv.total_bytes > naive_sum,
        "resident estimate ({}) must exceed the naive on-disk sum ({naive_sum}) \
             once in_proj_a expansion is accounted for",
        inv.total_bytes
    );
    assert_eq!(inv.total_bytes, expected);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn q4_resident_bytes_classifies_every_known_category() {
    // Direct unit coverage for the classifier itself, independent of
    // the manifest/fallback-scan plumbing above -- also exercises
    // the sanitized-filename form (dots -> `_`, `.q4`/`.f16` suffix)
    // used by the no-manifest fallback path.
    assert_eq!(
        q4_resident_bytes("model.language_model.norm.weight", 100),
        200
    );
    assert_eq!(
        q4_resident_bytes("model_language_model_norm_weight.f16", 100),
        200
    );
    assert_eq!(
        q4_resident_bytes("model.language_model.layers.0.linear_attn.A_log", 100),
        200
    );
    assert_eq!(
        q4_resident_bytes("model.language_model.layers.0.linear_attn.dt_bias", 100),
        200
    );
    assert_eq!(
        q4_resident_bytes(
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            100
        ),
        200
    );
    assert_eq!(
        q4_resident_bytes(
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            100
        ),
        320
    );
    assert_eq!(
        q4_resident_bytes(
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            100
        ),
        320
    );
    assert_eq!(
        q4_resident_bytes(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            100
        ),
        200
    );
    assert_eq!(
        q4_resident_bytes(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            100
        ),
        200
    );
    assert_eq!(
        q4_resident_bytes("model.language_model.embed_tokens.weight", 100),
        100
    );
    assert_eq!(
        q4_resident_bytes("mtp.pre_fc_norm_embedding.weight", 100),
        200
    );
    assert_eq!(q4_resident_bytes("mtp.pre_fc_norm_hidden.weight", 100), 200);
    assert_eq!(
        q4_resident_bytes("mtp_pre_fc_norm_embedding_weight.f16", 100),
        200
    );
    assert_eq!(
        q4_resident_bytes("mtp_pre_fc_norm_hidden_weight.f16", 100),
        200
    );
    // mmap'd-or-fused categories all stay at exactly 1x.
    assert_eq!(
        q4_resident_bytes("model.language_model.layers.0.self_attn.q_proj.weight", 100),
        100
    );
    assert_eq!(
        q4_resident_bytes("model.language_model.layers.0.mlp.gate_proj.weight", 100),
        100
    );
    assert_eq!(
        q4_resident_bytes(
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            100
        ),
        100
    );
    assert_eq!(q4_resident_bytes("lm_head.weight", 100), 100);
}

#[test]
fn q4_resident_bytes_embed_tokens_counts_its_file_once() {
    // The CPU lookup maps `embed_tokens.weight.q4` in place, and a
    // tied checkpoint's GPU logits path maps that same file again.
    // Two mappings of one file share its physical pages, so the
    // embedding contributes its on-disk size once -- unlike
    // `in_proj_qkv`/`in_proj_z`, whose second copy is a separate
    // merged file. An untied checkpoint's GPU mapping is the
    // separate `lm_head.weight.q4`, counted on its own entry.
    let tensor = "model.language_model.embed_tokens.weight";
    assert_eq!(q4_resident_bytes(tensor, 1000), 1000);
    assert_eq!(q4_resident_bytes("lm_head.weight", 1000), 1000);
}

#[test]
fn inspect_q4_dir_manifest_flags_missing_file() {
    let dir = tempdir("q4-missing-file");
    let index = serde_json::json!([
        {"name": LANGUAGE_MODEL_EMBED, "file": "model_language_model_embed_tokens_weight.q4", "shape": [32], "numel": 32},
        {"name": "some.tensor.weight", "file": "does_not_exist.q4", "quantized": true, "shape": [32], "numel": 32},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    write_fake_q4_file(
        &dir.join("model_language_model_embed_tokens_weight.q4"),
        2,
        &[32],
        32,
        1,
    );

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert!(
        inv.missing_tensors
            .iter()
            .any(|m| m.contains("some.tensor.weight"))
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_q4_dir_falls_back_to_directory_scan_without_manifest() {
    let dir = tempdir("q4-no-manifest");
    write_fake_q4_file(&dir.join("a.q4"), 2, &[32], 32, 1);
    write_fake_q4_file(&dir.join("b.q4"), 2, &[32], 32, 1);
    write_fake_q4_file(&dir.join("merged_qkvz_x.q4"), 2, &[64], 64, 2);
    let a_len = fs::metadata(dir.join("a.q4")).unwrap().len();
    let b_len = fs::metadata(dir.join("b.q4")).unwrap().len();

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert_eq!(inv.total_bytes, a_len + b_len);
    assert_eq!(inv.tensor_count, 2);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_q4_dir_fallback_scan_applies_resident_bytes_by_sanitized_filename() {
    // The no-manifest fallback path only sees sanitized filenames
    // (q4_tensor_path: dots -> `_`), not the original dotted tensor
    // name -- this proves the classifier's suffix matching still
    // works against that form for an expanded category.
    let dir = tempdir("q4-no-manifest-expanded");
    write_fake_q4_file(
        &dir.join("model_language_model_layers_0_linear_attn_in_proj_a_weight.q4"),
        2,
        &[32],
        32,
        10,
    );
    write_fake_q4_file(
        &dir.join("model_language_model_layers_0_self_attn_q_proj_weight.q4"),
        2,
        &[32],
        32,
        10,
    );
    let proj_a_len =
        fs::metadata(dir.join("model_language_model_layers_0_linear_attn_in_proj_a_weight.q4"))
            .unwrap()
            .len();
    let q_proj_len =
        fs::metadata(dir.join("model_language_model_layers_0_self_attn_q_proj_weight.q4"))
            .unwrap()
            .len();

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    let expected = (proj_a_len as f64 * 3.2).round() as u64 + q_proj_len;
    assert_eq!(inv.total_bytes, expected);
    assert!(inv.total_bytes > proj_a_len + q_proj_len);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_q4_dir_detects_mtp_tensors_via_manifest() {
    let dir = tempdir("q4-mtp-manifest");
    write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 1);
    write_fake_q4_file(&dir.join("mtp_norm.f16"), 2, &[32], 32, 1);
    let embedding_entry = write_q4_embedding_fixture(&dir);
    let index = serde_json::json!([
        embedding_entry,
        {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
        {"name": "mtp.norm.weight", "file": "mtp_norm.f16", "quantized": false, "shape": [32], "numel": 32},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert!(inv.has_mtp_tensors);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_q4_dir_no_mtp_tensors_via_manifest_when_absent() {
    let dir = tempdir("q4-no-mtp-manifest");
    write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 1);
    let embedding_entry = write_q4_embedding_fixture(&dir);
    let index = serde_json::json!([
        embedding_entry,
        {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert!(!inv.has_mtp_tensors);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_q4_dir_accepts_both_bare_array_and_quarot_object_manifest_shapes() {
    // Regression test for #626: `quantize_q4` writes
    // `quantize_index.json` as a bare array, but `quantize_quarot`
    // (ADR-051) writes an object -- `{"quarot_seed": ..., "tensors":
    // [...]}` -- so a loader can recover the QuaRot rotation seed
    // without parsing `config.json`. `doctor` crashed on the latter
    // shape with "invalid type: map, expected a sequence" before this
    // fix; both shapes must now load and produce equivalent
    // inventories for an identical tensor list.
    let bare_dir = tempdir("q4-manifest-bare-array");
    let object_dir = tempdir("q4-manifest-quarot-object");
    let [bare_embedding_entry, object_embedding_entry] = [&bare_dir, &object_dir].map(|dir| {
        write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 10);
        write_q4_embedding_fixture(dir)
    });
    let q_proj_len = fs::metadata(bare_dir.join("q_proj.q4")).unwrap().len();
    let embed_len = fs::metadata(bare_dir.join("model_language_model_embed_tokens_weight.q4"))
        .unwrap()
        .len();

    let tensors = serde_json::json!([
        bare_embedding_entry,
        {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
    ]);
    fs::write(
        bare_dir.join("quantize_index.json"),
        serde_json::to_vec(&tensors).unwrap(),
    )
    .unwrap();

    let quarot_seed: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let object_manifest = serde_json::json!({
        "quarot_seed": quarot_seed,
        "tensors": [
            object_embedding_entry,
            {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
        ],
    });
    fs::write(
        object_dir.join("quantize_index.json"),
        serde_json::to_vec(&object_manifest).unwrap(),
    )
    .unwrap();

    let cfg = Qwen35Config::qwen35_0_8b();
    let bare_inv = inspect_q4_dir(&bare_dir, &cfg)
        .expect("bare-array quantize_index.json (quantize_q4's shape) must load");
    let object_inv = inspect_q4_dir(&object_dir, &cfg)
        .expect("object-form quantize_index.json (quantize_quarot's shape) must load");

    assert_eq!(object_inv.total_bytes, bare_inv.total_bytes);
    assert_eq!(object_inv.total_bytes, embed_len + q_proj_len);
    assert_eq!(object_inv.tensor_count, bare_inv.tensor_count);
    assert_eq!(object_inv.tensor_count, 2);
    assert_eq!(object_inv.missing_tensors, bare_inv.missing_tensors);
    assert_eq!(object_inv.has_mtp_tensors, bare_inv.has_mtp_tensors);

    fs::remove_dir_all(&bare_dir).ok();
    fs::remove_dir_all(&object_dir).ok();
}

// Manifest shape/malformation parsing itself (missing-field errors,
// non-array `tensors`, scalar roots, oversized/truncated files) is
// covered directly against `lattice_inference::quant::q4_manifest`
// (issue #655 centralization). The tests here instead exercise
// `inspect_q4_dir`'s use of that shared parser end-to-end, so a
// malformed manifest surfaces as a `doctor`-level failure too.
#[test]
fn inspect_q4_dir_malformed_manifest_surfaces_precise_schema_error() {
    let dir = tempdir("q4-manifest-malformed");
    fs::write(dir.join("quantize_index.json"), br#"[{"name": "x"}]"#).unwrap();
    let cfg = Qwen35Config::qwen35_0_8b();
    let err = inspect_q4_dir(&dir, &cfg)
        .err()
        .expect("bare-array entry missing `file` must fail");
    assert!(
        err.contains("file"),
        "error must name the missing `file` field; got: {err}"
    );
    assert!(
        !err.contains("did not match any variant"),
        "error must not be the generic untagged-enum fallthrough; got: {err}"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_q4_dir_detects_mtp_tensors_via_fallback_scan() {
    let dir = tempdir("q4-mtp-fallback");
    write_fake_q4_file(
        &dir.join("model_language_model_layers_0_self_attn_q_proj_weight.q4"),
        2,
        &[32],
        32,
        1,
    );
    write_fake_q4_file(
        &dir.join("mtp_pre_fc_norm_embedding_weight.f16"),
        2,
        &[32],
        32,
        1,
    );

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert!(inv.has_mtp_tensors);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn inspect_q4_dir_no_mtp_tensors_via_fallback_scan_when_absent() {
    let dir = tempdir("q4-no-mtp-fallback");
    write_fake_q4_file(
        &dir.join("model_language_model_layers_0_self_attn_q_proj_weight.q4"),
        2,
        &[32],
        32,
        1,
    );

    let cfg = Qwen35Config::qwen35_0_8b();
    let inv = inspect_q4_dir(&dir, &cfg).unwrap();
    assert!(!inv.has_mtp_tensors);
    fs::remove_dir_all(&dir).ok();
}

// ---- system memory detection --------------------------------------

#[test]
fn detect_total_memory_bytes_is_plausible_when_known() {
    if let Some(bytes) = detect_total_memory_bytes() {
        assert!(bytes > 0);
        assert!(bytes < (1u64 << 50)); // sanity upper bound: 1 PiB
    }
}

// ---- build_report end-to-end --------------------------------------

fn write_config_json(dir: &Path, cfg: &Qwen35Config) {
    let config_json = serde_json::json!({
        "text_config": {
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "vocab_size": cfg.vocab_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "rope_theta": cfg.rope_theta,
            "partial_rotary_factor": cfg.partial_rotary_factor,
            "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_num_value_heads": cfg.linear_num_value_heads,
            "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_value_head_dim": cfg.linear_value_head_dim,
            "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "max_position_embeddings": cfg.max_position_embeddings,
            "eos_token_id": cfg.eos_token_id,
            "full_attention_interval": cfg.full_attention_interval,
        }
    });
    fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config_json).unwrap(),
    )
    .unwrap();
}

fn write_complete_safetensors_fixture(dir: &Path, cfg: &Qwen35Config) {
    let tensors = required_tensor_fixture(cfg);
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);
    write_config_json(dir, cfg);
    fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
}

#[test]
fn build_report_happy_path_is_ready() {
    let dir = tempdir("report-happy");
    let cfg = Qwen35Config::qwen35_0_8b();
    write_complete_safetensors_fixture(&dir, &cfg);

    let report = build_report(&dir, None, Some(4096), Some(1u64 << 40)).unwrap();
    assert!(
        report.is_ready(),
        "blocking reasons: {:?}",
        report.blocking_reasons
    );
    assert_eq!(report.placement, Placement::Cpu);
    assert_eq!(report.requested_fits, Some(true));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_infeasible_context_via_memory_override_exits_not_ready() {
    let dir = tempdir("report-infeasible");
    let cfg = Qwen35Config::qwen35_0_8b();
    write_complete_safetensors_fixture(&dir, &cfg);

    // Force a tiny "machine": far less memory than the weights
    // alone need, so no context length fits.
    let report = build_report(&dir, None, Some(1), Some(1024)).unwrap();
    assert!(!report.is_ready());
    assert_eq!(report.requested_fits, Some(false));
    assert!(
        report
            .blocking_reasons
            .iter()
            .any(|r| r.contains("does not fit"))
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_context_exceeding_any_real_machine_is_not_ready() {
    // Belt-and-suspenders alternative to the override above: an
    // absurd requested context against REAL detected memory (no
    // override) must still be infeasible on any real machine.
    let dir = tempdir("report-absurd-context");
    let cfg = Qwen35Config::qwen35_0_8b();
    write_complete_safetensors_fixture(&dir, &cfg);

    let report = build_report(&dir, None, Some(usize::MAX / 2), None).unwrap();
    if report.available_memory_bytes.is_some() {
        assert_eq!(report.requested_fits, Some(false));
        assert!(!report.is_ready());
    }
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_weights_exceed_ram_fails_without_explicit_context() {
    // Issue #875: doctor's own numbers said weights (64.60 GiB)
    // exceeded system memory (32.00 GiB) yet the verdict was still
    // "OK" because no `--context` was passed. Reproduce with no
    // requested context at all -- the fail-closed check must not
    // depend on `requested_fits`.
    let dir = tempdir("report-weights-exceed-ram-no-ctx");
    let cfg = Qwen35Config::qwen35_0_8b();
    write_complete_safetensors_fixture(&dir, &cfg);
    let weight_bytes = required_tensor_fixture(&cfg).len() as u64 * 64;

    let report = build_report(&dir, None, None, Some(weight_bytes - 1)).unwrap();
    assert!(!report.is_ready());
    assert_eq!(report.requested_context, None);
    assert!(
        report
            .blocking_reasons
            .iter()
            .any(|r| r.contains("exceeds detected system memory")),
        "blocking reasons: {:?}",
        report.blocking_reasons
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_zero_feasible_context_fails_without_explicit_context() {
    // Weights alone fit, but leave less than one KV-cache token's
    // worth of headroom -- max feasible context is 0. Must fail
    // closed even though no `--context` was requested.
    let dir = tempdir("report-zero-ctx-no-request");
    let cfg = Qwen35Config::qwen35_0_8b();
    write_complete_safetensors_fixture(&dir, &cfg);
    let weight_bytes = required_tensor_fixture(&cfg).len() as u64 * 64;
    let kv_bytes_per_token = cfg.kv_bytes_per_token(kv_cache_dtype_bytes()) as u64;
    assert!(
        kv_bytes_per_token > 1,
        "fixture assumption: at least 2 bytes/token of KV cache"
    );

    let report = build_report(&dir, None, None, Some(weight_bytes + 1)).unwrap();
    assert_eq!(report.max_context_len, Some(0));
    assert!(!report.is_ready());
    assert_eq!(report.requested_context, None);
    assert!(
        report
            .blocking_reasons
            .iter()
            .any(|r| r.contains("max feasible context length is 0 tokens")),
        "blocking reasons: {:?}",
        report.blocking_reasons
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_fits_within_ram_is_ready_without_explicit_context() {
    // Existing passing configurations must still report OK when no
    // `--context` is requested (acceptance criterion: no
    // regression on the happy path).
    let dir = tempdir("report-fits-no-ctx");
    let cfg = Qwen35Config::qwen35_0_8b();
    write_complete_safetensors_fixture(&dir, &cfg);

    let report = build_report(&dir, None, None, Some(1u64 << 40)).unwrap();
    assert!(
        report.is_ready(),
        "blocking reasons: {:?}",
        report.blocking_reasons
    );
    assert!(report.max_context_len.unwrap_or(0) > 0);
    fs::remove_dir_all(&dir).ok();
}

/// Minimal Q4 fixture for the threshold-regression tests below: an
/// `embed_tokens` tensor, a still-expanding `in_proj_a`, and a plain
/// `q_proj` rather than the full `qwen_required_tensor_names` set.
/// `inspect_q4_dir` sums `total_bytes` only over manifest entries
/// actually present, so this keeps the expected total small enough to
/// state as a closed formula in the test -- computed independently of
/// `q4_resident_bytes`/`inspect_q4_dir` themselves, so a regression in
/// either is caught rather than silently absorbed into a
/// self-referential expectation. `missing_tensors` will list the rest
/// of `qwen_required_tensor_names(cfg)` as absent, which is irrelevant
/// here: it drives a separate, unrelated blocking reason, not the
/// "exceeds detected system memory" one under test.
fn write_expanded_q4_fixture(dir: &Path, cfg: &Qwen35Config) -> (u64, u64, u64) {
    write_fake_q4_file(
        &dir.join("model_language_model_embed_tokens_weight.q4"),
        2,
        &[32],
        32,
        5,
    );
    write_fake_q4_file(&dir.join("proj_a.q4"), 2, &[32], 32, 20);
    write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 5);
    let embed_bytes = fs::metadata(dir.join("model_language_model_embed_tokens_weight.q4"))
        .unwrap()
        .len();
    let proj_a_bytes = fs::metadata(dir.join("proj_a.q4")).unwrap().len();
    let q_proj_bytes = fs::metadata(dir.join("q_proj.q4")).unwrap().len();
    let index = serde_json::json!([
        {"name": "model.language_model.embed_tokens.weight", "file": "model_language_model_embed_tokens_weight.q4", "quantized": true, "shape": [32], "numel": 32},
        {"name": "model.language_model.layers.0.linear_attn.in_proj_a.weight", "file": "proj_a.q4", "quantized": true, "shape": [32], "numel": 32},
        {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    write_config_json(dir, cfg);
    fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    (embed_bytes, proj_a_bytes, q_proj_bytes)
}

#[test]
fn build_report_q4_resident_estimate_no_longer_overcounts_embed_tokens() {
    // The CPU embedding lookup now maps `embed_tokens.weight.q4` and
    // reads rows from it, so the embedding contributes its on-disk
    // size once. Pin the boundary with an INDEPENDENTLY hand-computed
    // total -- embed 1x + in_proj_a 3.2x (still dequantized to f16)
    // + q_proj 1x -- rather than by calling `q4_resident_bytes`, so a
    // regression back to a larger embedding charge pushes the
    // report's real `inventory.total_bytes` over this override and
    // is caught.
    let dir = tempdir("report-q4-resident-threshold");
    let cfg = Qwen35Config::qwen35_0_8b();
    let (embed_bytes, proj_a_bytes, q_proj_bytes) = write_expanded_q4_fixture(&dir, &cfg);

    let corrected_total = embed_bytes + (proj_a_bytes as f64 * 3.2).round() as u64 + q_proj_bytes;
    let threshold_override = corrected_total + 1;

    let report = build_report(&dir, None, None, Some(threshold_override)).unwrap();
    assert!(
        !report
            .blocking_reasons
            .iter()
            .any(|r| r.contains("exceeds detected system memory")),
        "the resident estimate must fit a budget one byte above the hand-computed \
             total; blocking reasons: {:?}",
        report.blocking_reasons
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_q4_true_over_budget_still_fails() {
    // Belt-and-suspenders alongside the threshold test above: a Q4
    // checkpoint that genuinely exceeds the resident estimate must
    // still fail closed -- the fix must not turn into a blanket pass
    // for Q4. Same independent hand-computed formula as the test
    // above.
    let dir = tempdir("report-q4-true-over-budget");
    let cfg = Qwen35Config::qwen35_0_8b();
    let (embed_bytes, proj_a_bytes, q_proj_bytes) = write_expanded_q4_fixture(&dir, &cfg);

    let corrected_total = embed_bytes + (proj_a_bytes as f64 * 3.2).round() as u64 + q_proj_bytes;
    let report = build_report(&dir, None, None, Some(corrected_total - 1)).unwrap();
    assert!(
        report
            .blocking_reasons
            .iter()
            .any(|r| r.contains("exceeds detected system memory")),
        "a checkpoint exceeding even the corrected estimate must still fail closed; \
             blocking reasons: {:?}",
        report.blocking_reasons
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_missing_tokenizer_is_not_ready() {
    let dir = tempdir("report-no-tokenizer");
    let cfg = Qwen35Config::qwen35_0_8b();
    let tensors = required_tensor_fixture(&cfg);
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);
    write_config_json(&dir, &cfg);
    // No tokenizer.json written.

    let report = build_report(&dir, None, None, Some(1u64 << 40)).unwrap();
    assert!(!report.is_ready());
    assert!(!report.tokenizer_present);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_unknown_format_is_err() {
    let dir = tempdir("report-unknown");
    let err = build_report(&dir, None, None, None).unwrap_err();
    assert!(err.contains("not a recognized model directory"));
    fs::remove_dir_all(&dir).ok();
}

// ---- #923: missing config.json is a hard, uniform error -----------
//
// Before the fix, this Safetensors branch silently substituted
// `Qwen35Config::qwen35_2b()` and the Q4 branch silently substituted
// `Qwen35Config::qwen36_27b()` -- two different guessed presets, and
// neither told the caller a config.json was even missing. Both
// branches now go through the same `Qwen35Config::from_model_dir`
// fail-closed helper as every other loader in this crate. Reverting
// either branch's call site back to the old exists-then-preset
// pattern makes its half of this test fail (it would return `Ok`
// instead of an error naming `config.json`).

#[test]
fn build_report_safetensors_missing_config_json_is_hard_error() {
    let dir = tempdir("report-safetensors-no-config");
    let cfg = Qwen35Config::qwen35_0_8b();
    let tensors = required_tensor_fixture(&cfg);
    let refs: Vec<(&str, &str, u64, u64)> = tensors
        .iter()
        .map(|(n, d, s, e)| (n.as_str(), d.as_str(), *s, *e))
        .collect();
    write_fake_safetensors(&dir.join("model.safetensors"), &refs);
    // Deliberately no config.json and no tokenizer.json.

    let err = build_report(&dir, None, None, Some(1u64 << 40)).unwrap_err();
    assert!(
        err.contains("config.json"),
        "error must name the missing config.json, not a guessed preset: {err}"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn build_report_q4_missing_config_json_is_hard_error() {
    let dir = tempdir("report-q4-no-config");
    write_fake_q4_file(
        &dir.join("model_language_model_embed_tokens_weight.q4"),
        2,
        &[32],
        32,
        5,
    );
    write_fake_q4_file(&dir.join("q_proj.q4"), 2, &[32], 32, 3);
    let index = serde_json::json!([
        {"name": "model.language_model.embed_tokens.weight", "file": "model_language_model_embed_tokens_weight.q4", "quantized": true, "shape": [32], "numel": 32},
        {"name": "model.language_model.layers.0.self_attn.q_proj.weight", "file": "q_proj.q4", "quantized": true, "shape": [32], "numel": 32},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    // Deliberately no config.json.

    let err = build_report(&dir, None, None, Some(1u64 << 40)).unwrap_err();
    assert!(
        err.contains("config.json"),
        "error must name the missing config.json, not a guessed preset: {err}"
    );
    fs::remove_dir_all(&dir).ok();
}

// ---- DoctorReport Display: MTP disclosure --------------------------

fn minimal_doctor_report(has_mtp_tensors: bool) -> DoctorReport {
    // Constructed directly rather than through `build_report` -- this
    // targets the `Display` impl's MTP-disclosure branch in
    // isolation, without needing a full valid Q4 checkpoint fixture
    // (manifest + every required tensor + tokenizer.json) that no
    // other test in this module builds either.
    DoctorReport {
        model_dir: PathBuf::from("/fake/model"),
        format: crate::backend::ModelFormat::Q4,
        placement: Placement::Metal,
        quantization: "Q4_0".to_string(),
        tensor_count: 1,
        weight_bytes: 100,
        kv_bytes_per_token: 100,
        max_position_embeddings: 4096,
        metal_runtime_cache_cap: Some(METAL_RUNTIME_MAX_CACHE_LEN),
        available_memory_bytes: Some(1u64 << 40),
        max_context_len: Some(4096),
        requested_context: None,
        requested_fits: None,
        tokenizer_path: PathBuf::from("/fake/model/tokenizer.json"),
        tokenizer_present: true,
        missing_tensors: Vec::new(),
        blocking_reasons: Vec::new(),
        warnings: Vec::new(),
        has_mtp_tensors,
    }
}

#[test]
fn doctor_report_warnings_render_without_blocking_readiness() {
    let mut report = minimal_doctor_report(false);
    assert!(!format!("{report}").contains("Warning:"));
    report
        .warnings
        .push("manifest alias ignored; using native file".to_string());
    assert!(report.is_ready());
    let text = format!("{report}");
    assert!(text.contains("Warning: manifest alias ignored; using native file"));
    assert!(text.contains("Result: OK"));
}

#[test]
fn doctor_report_display_includes_mtp_disclosure_when_mtp_tensors_present() {
    let report = minimal_doctor_report(true);
    let text = format!("{report}");
    assert!(
        text.contains("MTP"),
        "report must disclose the uncounted MTP K/V cache when MTP tensors are present:\n{text}"
    );
}

#[test]
fn doctor_report_display_omits_mtp_disclosure_when_no_mtp_tensors() {
    let report = minimal_doctor_report(false);
    let text = format!("{report}");
    assert!(
        !text.contains("MTP"),
        "report must not mention MTP for a directory with no MTP tensors:\n{text}"
    );
}

#[test]
fn doctor_report_escapes_control_characters_in_each_diagnostic_route() {
    let mut report = minimal_doctor_report(false);
    let diagnostic = "line\nbreak\u{1b}\tend\\n";
    report.missing_tensors.push(diagnostic.to_string());
    report.warnings.push(diagnostic.to_string());
    report.blocking_reasons.push(diagnostic.to_string());

    let text = format!("{report}");
    assert!(!report.is_ready());
    assert_eq!(report.warnings, [diagnostic]);
    assert!(!text.contains('\u{1b}'));
    assert!(!text.contains('\t'));
    assert_eq!(
        text.lines()
            .filter(|line| *line == r"  - line\nbreak\u{1b}\tend\n")
            .count(),
        2
    );
    assert_eq!(
        text.lines()
            .filter(|line| *line == r"Warning: line\nbreak\u{1b}\tend\n")
            .count(),
        1
    );
    assert!(!text.lines().any(|line| line.starts_with("break")));
}

#[test]
fn doctor_report_escapes_manifest_alias_and_missing_entry_strings() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_hidden_layers = 1;
    cfg.layer_types = vec![lattice_arch::qwen35_config::LayerType::FullAttention];
    cfg.layer_mask = vec![true];
    cfg.tie_word_embeddings = true;
    cfg.full_attention_interval = 1;
    let dir = tempdir("q4-report-escaped-manifest");
    write_config_json(&dir, &cfg);
    fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    write_fake_q4_file(&dir.join("sample.q4"), 2, &[32], 32, 1);
    let alias = "alias\n\u{1b}.q4";
    let missing_name = "missing\n\u{1b}.weight";
    let missing_file = "absent\n\u{1b}.q4";
    let entries = serde_json::json!([
        {"name": "model.language_model.embed_tokens.weight", "file": alias},
        {"name": missing_name, "file": missing_file},
    ]);
    fs::write(
        dir.join("quantize_index.json"),
        serde_json::to_vec(&entries).unwrap(),
    )
    .unwrap();

    let report = build_report(&dir, None, None, Some(1u64 << 40)).unwrap();
    assert!(!report.is_ready());
    assert!(report.warnings[0].contains(alias));
    assert!(
        report
            .missing_tensors
            .iter()
            .any(|name| name.contains(missing_name))
    );
    assert!(
        report
            .blocking_reasons
            .iter()
            .any(|reason| reason.contains(alias))
    );
    let text = format!("{report}");
    assert!(!text.contains('\u{1b}'));
    assert!(!text.contains(alias));
    assert_eq!(text.matches(r"alias\n\u{1b}.q4").count(), 3);
    assert!(text.contains(r"missing\n\u{1b}.weight"));
    assert!(text.contains(r"absent\n\u{1b}.q4"));
    assert_eq!(
        text.lines()
            .filter(|line| line.starts_with("Warning:"))
            .count(),
        1
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn q4_sample_continues_after_resolution_error_to_first_regular_handle() {
    let dir = tempdir("q4-sample-unresolved-then-regular");
    let unresolved = dir.join("a-unresolved.q4");
    let first_regular = dir.join("b-regular.q4");
    let later = dir.join("c-later.q4");
    fs::write(&unresolved, b"ordinary resolver fixture").unwrap();
    write_fake_q4_file(&first_regular, 2, &[32], 32, 1);
    write_fake_q4_file(&later, 1, &[32], 32, 1);
    let mut visited = Vec::new();
    let sample = select_q4_quantization_sample_with_opener(&dir, |path| {
        visited.push(path.to_path_buf());
        if path == unresolved.as_path() {
            Err(format!(
                "failed to open {}: modeled refusal",
                path.display()
            ))
        } else {
            open_trusted_mmap_file(path)
        }
    })
    .unwrap()
    .unwrap();
    assert_eq!(visited, [unresolved, first_regular.clone()]);
    assert_eq!(sample.path, first_regular);
    assert_eq!(
        q4_quantization_label_from_sample(sample).unwrap(),
        "Q4_0 (lattice native, v2 asymmetric)"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn q4_sample_retains_first_resolution_error_when_none_qualifies() {
    let dir = tempdir("q4-sample-all-unresolved");
    let first = dir.join("a.q4");
    let second = dir.join("b.q4");
    fs::write(&first, b"ordinary first fixture").unwrap();
    fs::write(&second, b"ordinary second fixture").unwrap();
    let mut visited = Vec::new();
    let result = select_q4_quantization_sample_with_opener(&dir, |path| {
        visited.push(path.to_path_buf());
        Err(format!(
            "failed to open {}: unresolved target",
            path.display()
        ))
    });
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("unresolved candidates must retain their opening error"),
    };
    assert_eq!(visited, [first.clone(), second]);
    assert_eq!(
        error,
        format!("failed to open {}: unresolved target", first.display())
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn q4_sample_distinguishes_refused_candidates_from_empty_inventory() {
    let dir = tempdir("q4-sample-refusals");
    fs::write(dir.join("notes.txt"), b"not eligible").unwrap();
    assert!(select_q4_quantization_sample(&dir).unwrap().is_none());
    let missing = dir.join("a.q4");
    let non_regular = dir.join("b.q4");
    fs::write(&missing, b"ordinary missing-result fixture").unwrap();
    fs::write(&non_regular, b"ordinary non-regular-result fixture").unwrap();
    let first_error = format!("failed to open {}: modeled missing file", missing.display());
    let mut visited = Vec::new();
    let result = select_q4_quantization_sample_with_opener(&dir, |path| {
        visited.push(path.to_path_buf());
        if path == missing.as_path() {
            Err(first_error.clone())
        } else {
            Err(format!(
                "refusing to load {}: not a regular file",
                path.display()
            ))
        }
    });
    assert_eq!(result.err().unwrap(), first_error);
    assert_eq!(visited, [missing, non_regular]);
    fs::remove_dir_all(&dir).unwrap();
    let error = match select_q4_quantization_sample(&dir) {
        Err(error) => error,
        Ok(_) => panic!("directory enumeration failure must remain an error"),
    };
    assert!(error.starts_with("failed to read directory "));
}

#[test]
fn q4_sample_retains_modeled_trust_refusal() {
    let dir = tempdir("q4-sample-modeled-trust");
    let ordinary = dir.join("ordinary.q4");
    write_fake_q4_file(&ordinary, 2, &[32], 32, 1);
    let mut visited = Vec::new();
    let result = select_q4_quantization_sample_with_opener(&dir, |path| {
        visited.push(path.to_path_buf());
        Err("modeled trust refusal".to_string())
    });
    assert_eq!(result.err().unwrap(), "modeled trust refusal");
    assert_eq!(visited, std::slice::from_ref(&ordinary));

    let sample = select_q4_quantization_sample(&dir).unwrap().unwrap();
    assert_eq!(sample.path, ordinary);
    assert_eq!(
        q4_quantization_label_from_sample(sample).unwrap(),
        "Q4_0 (lattice native, v2 asymmetric)"
    );
    fs::remove_dir_all(&dir).unwrap();
}
