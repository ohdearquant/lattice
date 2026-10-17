use super::*;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::measurement::gpu_test_lock;

#[test]
fn test_qwen_config_dimensions() {
    let cfg = QwenConfig::qwen3_embedding_0_6b();
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.q_dim(), 2048); // 16 * 128
    assert_eq!(cfg.kv_dim(), 1024); // 8 * 128
    assert_eq!(cfg.num_groups(), 2); // 16 / 8
}

#[test]
fn test_parse_config_zero_heads_is_error_not_panic() {
    let tmp = std::env::temp_dir().join("lattice_test_zero_heads");
    std::fs::create_dir_all(&tmp).unwrap();

    // num_attention_heads=0 WITH head_dim present: proves the eager
    // `unwrap_or(hidden_size / num_attention_heads)` divide-by-zero is
    // guarded even on the head_dim-explicit path.
    let cfg_zero_attn = tmp.join("zero_attn_heads.json");
    std::fs::write(
            &cfg_zero_attn,
            r#"{"vocab_size":151669,"hidden_size":1024,"num_hidden_layers":1,"num_attention_heads":0,"num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,"max_position_embeddings":32768}"#,
        )
        .unwrap();
    let r = parse_qwen_config(&cfg_zero_attn);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "num_attention_heads=0 must be InvalidInput, not panic; got {r:?}"
    );

    // num_key_value_heads=0: sibling divisor (num_groups / GQA grouping).
    let cfg_zero_kv = tmp.join("zero_kv_heads.json");
    std::fs::write(
            &cfg_zero_kv,
            r#"{"vocab_size":151669,"hidden_size":1024,"num_hidden_layers":1,"num_attention_heads":16,"num_key_value_heads":0,"head_dim":128,"intermediate_size":3072,"max_position_embeddings":32768}"#,
        )
        .unwrap();
    let r = parse_qwen_config(&cfg_zero_kv);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "num_key_value_heads=0 must be InvalidInput, not panic; got {r:?}"
    );

    // Sanity: a valid config still parses.
    let cfg_ok = tmp.join("valid.json");
    std::fs::write(
            &cfg_ok,
            r#"{"vocab_size":151669,"hidden_size":1024,"num_hidden_layers":1,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,"max_position_embeddings":32768}"#,
        )
        .unwrap();
    assert!(
        parse_qwen_config(&cfg_ok).is_ok(),
        "valid config must still parse"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_parse_config_non_divisible_gqa_heads_is_error_not_panic() {
    let tmp = std::env::temp_dir().join("lattice_test_nondiv_gqa_heads");
    std::fs::create_dir_all(&tmp).unwrap();

    // num_attention_heads=16 not divisible by num_key_value_heads=6.
    // `GqaConfig::groups()` only `debug_assert`s divisibility (stripped in
    // release), so without this load-time guard a release build would reach
    // the hard `assert_eq!` in `apply_gqa_attention` and panic deep in the
    // forward pass. Must be rejected as InvalidInput at parse time.
    let cfg_bad = tmp.join("non_divisible.json");
    std::fs::write(
            &cfg_bad,
            r#"{"vocab_size":151669,"hidden_size":1024,"num_hidden_layers":1,"num_attention_heads":16,"num_key_value_heads":6,"head_dim":128,"intermediate_size":3072,"max_position_embeddings":32768}"#,
        )
        .unwrap();
    let r = parse_qwen_config(&cfg_bad);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "non-divisible GQA heads must be InvalidInput, not panic; got {r:?}"
    );
    assert!(
        r.unwrap_err().to_string().contains("divisible"),
        "error message must name the divisibility violation"
    );

    // Sanity: an evenly-divisible config (16 % 8 == 0) still parses.
    let cfg_ok = tmp.join("divisible.json");
    std::fs::write(
            &cfg_ok,
            r#"{"vocab_size":151669,"hidden_size":1024,"num_hidden_layers":1,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,"max_position_embeddings":32768}"#,
        )
        .unwrap();
    assert!(
        parse_qwen_config(&cfg_ok).is_ok(),
        "divisible config must still parse"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_parse_config_rejects_invalid_rope_head_dim() {
    let tmp = std::env::temp_dir().join("lattice_test_invalid_qwen_head_dim");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();

    for (name, head_dim) in [("zero", 0), ("odd", 3)] {
        let path = tmp.join(format!("{name}.json"));
        std::fs::write(
                &path,
                format!(
                    r#"{{"vocab_size":8,"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":{head_dim},"intermediate_size":4,"max_position_embeddings":8}}"#
                ),
            )
            .unwrap();

        let err = parse_qwen_config(&path).unwrap_err();
        assert!(
            matches!(err, InferenceError::InvalidInput(_)),
            "head_dim={head_dim} must be rejected as InvalidInput; got {err:?}"
        );
        assert!(
            err.to_string().contains("head_dim"),
            "error must identify head_dim; got {err}"
        );
    }

    let valid = tmp.join("valid.json");
    std::fs::write(
            &valid,
            r#"{"vocab_size":8,"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"intermediate_size":4,"max_position_embeddings":8}"#,
        )
        .unwrap();
    assert!(parse_qwen_config(&valid).is_ok());

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_parse_config_json() {
    let json = r#"{
            "vocab_size": 151669,
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 3072,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0
        }"#;
    assert_eq!(extract_json_usize(json, "hidden_size"), Some(1024));
    assert_eq!(extract_json_usize(json, "num_key_value_heads"), Some(8));
    assert_eq!(extract_json_usize(json, "head_dim"), None); // not in this test JSON
    assert_eq!(extract_json_f64(json, "rope_theta"), Some(1_000_000.0));
}

#[test]
#[ignore = "Requires model files: set LATTICE_INFERENCE_MODEL_DIR"]
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn test_qwen_long_text_bench() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let _gpu_lock = gpu_test_lock();
    let model = QwenModel::from_directory(std::path::Path::new(&model_dir)).unwrap();
    // Warmup
    let _ = model.encode("warmup").unwrap();

    let long_text = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention and rotary position embeddings for efficient multilingual text representation. ".repeat(25);
    println!("Long text: {} chars", long_text.len());

    let t = std::time::Instant::now();
    let n = 3;
    for _ in 0..n {
        let _ = model.encode(&long_text).unwrap();
    }
    let ms = t.elapsed().as_millis() as f64 / n as f64;
    println!("Long encode: {ms:.0}ms avg ({n} runs)");
}

#[test]
#[ignore = "Requires model files: set LATTICE_INFERENCE_MODEL_DIR"]
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn test_qwen_multilingual() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let _gpu_lock = gpu_test_lock();
    let model = QwenModel::from_directory(std::path::Path::new(&model_dir)).unwrap();

    let pairs = [
        ("记得吃药", "remember to take medicine"),
        ("今天天气不错", "the weather is nice today"),
        ("量子物理", "quantum physics"),
        ("我想吃火锅", "I want to eat hotpot"),
    ];

    println!("\n--- Multilingual cosine similarity ---");
    for (zh, en) in &pairs {
        let e_zh = model.encode(zh).unwrap();
        let e_en = model.encode(en).unwrap();
        let cos: f32 = e_zh.iter().zip(e_en.iter()).map(|(a, b)| a * b).sum();
        println!("{zh:12} <-> {en:35} = {cos:.4}");
    }

    // Cross-pair: unrelated should be low.
    let pill_zh = model.encode("记得吃药").unwrap();
    let physics_en = model.encode("quantum physics").unwrap();
    let cross: f32 = pill_zh
        .iter()
        .zip(physics_en.iter())
        .map(|(a, b)| a * b)
        .sum();
    println!("记得吃药       <-> quantum physics                      = {cross:.4} (unrelated)");

    // Related pair should be higher than unrelated.
    let pill_en = model.encode("remember to take medicine").unwrap();
    let related: f32 = pill_zh.iter().zip(pill_en.iter()).map(|(a, b)| a * b).sum();
    assert!(
        related > cross + 0.1,
        "related pair ({related:.4}) should score higher than unrelated ({cross:.4})"
    );
}

#[test]
#[ignore = "Requires model files: set LATTICE_INFERENCE_MODEL_DIR"]
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn test_qwen_encode_real_model() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");

    let _gpu_lock = gpu_test_lock();
    let model = QwenModel::from_directory(std::path::Path::new(&model_dir)).unwrap();
    assert_eq!(model.config().hidden_size, 1024);
    assert_eq!(model.dimensions(), 1024);

    let embedding = model.encode("hello world").unwrap();
    assert_eq!(embedding.len(), 1024);

    // Verify L2-normalized.
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "embedding should be L2-normalized, got norm={norm}"
    );

    // Different texts should produce different embeddings.
    let emb2 = model.encode("quantum physics").unwrap();
    let dot: f32 = embedding.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    assert!(
        dot < 0.99,
        "different texts should have different embeddings, cosine={dot}"
    );

    println!(
        "embed dim={}, norm={norm:.4}, cosine_sim={dot:.4}",
        embedding.len()
    );
}

/// The sequence `tokenize_for_embedding` builds from one tokenizer result and its counts,
/// restated independently of `embedding_sequence` with one explicit branch per case.
fn reference_sequence(
    real: &[u32],
    pre_truncation_len: usize,
    max_len: usize,
    eos: u32,
) -> (Vec<u32>, EmbeddingTokenCounts) {
    let (ids, before_truncation) = if real.last() == Some(&eos) {
        (real.to_vec(), pre_truncation_len)
    } else if real.len() < max_len {
        let mut ids = real.to_vec();
        ids.push(eos);
        (ids, pre_truncation_len + 1)
    } else {
        let mut ids = real[..real.len().saturating_sub(1)].to_vec();
        ids.push(eos);
        (ids, pre_truncation_len + 1)
    };
    let embedded = ids.len();
    (
        ids,
        EmbeddingTokenCounts {
            before_truncation,
            embedded,
        },
    )
}

fn tokenized(ids: &[u32], pre_truncation_len: usize) -> TokenizedInput {
    TokenizedInput {
        input_ids: ids.to_vec(),
        attention_mask: vec![1; ids.len()],
        token_type_ids: vec![0; ids.len()],
        real_length: ids.len(),
        pre_truncation_len,
    }
}

#[test]
fn embedding_sequence_builds_the_ids_and_counts_for_both_end_token_sources() {
    const EOS: u32 = 99;
    let counts = |before_truncation, embedded| EmbeddingTokenCounts {
        before_truncation,
        embedded,
    };
    // The tokenizer ends the sequence itself: nothing is added, and a cut that kept the
    // end token reports the full pre-truncation length.
    assert_eq!(
        embedding_sequence(&tokenized(&[1, 2, 3, EOS], 4), 8, EOS),
        (vec![1, 2, 3, EOS], counts(4, 4))
    );
    assert_eq!(
        embedding_sequence(&tokenized(&[1, 2, 3, EOS], 9), 4, EOS),
        (vec![1, 2, 3, EOS], counts(9, 4))
    );
    // The model appends the end token when there is room.
    assert_eq!(
        embedding_sequence(&tokenized(&[1, 2, 3], 3), 8, EOS),
        (vec![1, 2, 3, EOS], counts(4, 4))
    );
    // At the limit the end token overwrites the last token, which counts as lost context.
    assert_eq!(
        embedding_sequence(&tokenized(&[1, 2, 3, 4], 4), 4, EOS),
        (vec![1, 2, 3, EOS], counts(5, 4))
    );
    assert_eq!(
        embedding_sequence(&tokenized(&[1, 2, 3, 4], 10), 4, EOS),
        (vec![1, 2, 3, EOS], counts(11, 4))
    );
}

#[test]
fn embedding_sequence_matches_an_independent_restatement_of_the_rule() {
    const EOS: u32 = 99;
    for max_len in [1usize, 2, 5, 16] {
        for real_len in 0..=max_len {
            for ends_with_eos in [false, true] {
                if ends_with_eos && real_len == 0 {
                    continue;
                }
                let mut real: Vec<u32> = (1..=real_len as u32).collect();
                if ends_with_eos && let Some(last) = real.last_mut() {
                    *last = EOS;
                }
                for extra_cut in [0usize, 3] {
                    let pre_truncation_len = real_len + extra_cut;
                    let (ids, counts) =
                        embedding_sequence(&tokenized(&real, pre_truncation_len), max_len, EOS);
                    assert_eq!(
                        (ids.clone(), counts),
                        reference_sequence(&real, pre_truncation_len, max_len, EOS),
                        "max_len={max_len} real_len={real_len} eos={ends_with_eos} cut={extra_cut}"
                    );
                    assert_eq!(counts.embedded, ids.len());
                    assert!(counts.embedded <= counts.before_truncation);
                }
            }
        }
    }
}

#[test]
fn test_default_inference_config_matches_06b() {
    let cfg = ModelInferenceConfig::default();
    assert_eq!(cfg.eos_token_id, 151_643);
    assert_eq!(cfg.rope_table_max_seq_len, 8192);
    assert_eq!(cfg.gpu_max_seq_len, 2048);
    assert_eq!(cfg.base_model_rev, "none");
    assert_eq!(cfg.tokenizer_rev, "none");
}

#[test]
fn test_load_inference_config_from_json() {
    let json = r#"{"eos_token_id": 151645, "rope_table_max_seq_len": 32768, "gpu_max_seq_len": 4096, "base_model_rev": "qwen3-0.6b-rev2", "tokenizer_rev": "tok-rev3"}"#;
    let cfg: ModelInferenceConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.eos_token_id, 151_645);
    assert_eq!(cfg.rope_table_max_seq_len, 32_768);
    assert_eq!(cfg.gpu_max_seq_len, 4096);
    assert_eq!(cfg.base_model_rev, "qwen3-0.6b-rev2");
    assert_eq!(cfg.tokenizer_rev, "tok-rev3");
}

#[test]
fn test_partial_config_uses_serde_defaults() {
    let json = r#"{"gpu_max_seq_len": 4096}"#;
    let cfg: ModelInferenceConfig = serde_json::from_str(json).unwrap();
    assert_eq!(
        cfg.eos_token_id, 151_643,
        "unset eos_token_id should use 0.6B default"
    );
    assert_eq!(
        cfg.rope_table_max_seq_len, 8192,
        "unset rope_table_max_seq_len should use 0.6B default"
    );
    assert_eq!(
        cfg.gpu_max_seq_len, 4096,
        "explicitly set field should override default"
    );

    let json2 = r#"{"eos_token_id": 151645}"#;
    let cfg2: ModelInferenceConfig = serde_json::from_str(json2).unwrap();
    assert_eq!(cfg2.eos_token_id, 151_645, "set field should override");
    assert_eq!(
        cfg2.rope_table_max_seq_len, 8192,
        "unset field should use default"
    );
    assert_eq!(cfg2.gpu_max_seq_len, 2048, "unset field should use default");

    let empty: ModelInferenceConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(empty.eos_token_id, 151_643);
    assert_eq!(empty.rope_table_max_seq_len, 8192);
    assert_eq!(empty.gpu_max_seq_len, 2048);
    assert_eq!(empty.base_model_rev, "none");
    assert_eq!(empty.tokenizer_rev, "none");
}

#[test]
fn test_missing_config_non_06b_errors() {
    let tmp = std::env::temp_dir().join("lattice_test_non06b_err");
    let model_dir = tmp.join("qwen3-embedding-4b");
    std::fs::create_dir_all(&model_dir).unwrap();

    assert!(
        !is_qwen3_embedding_0_6b_dir(&model_dir),
        "4b dir must NOT be detected as 0.6B"
    );

    let config_path = model_dir.join("config.json");
    let result = parse_qwen_config(&config_path);
    assert!(
        result.is_err(),
        "parse_qwen_config must error when config.json does not exist"
    );

    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("No such file")
            || err_msg.contains("not found")
            || err_msg.contains("os error"),
        "error should indicate missing file, got: {err_msg}"
    );

    let dir_06b = tmp.join("qwen3-embedding-0.6b");
    std::fs::create_dir_all(&dir_06b).unwrap();
    assert!(
        is_qwen3_embedding_0_6b_dir(&dir_06b),
        "0.6b dir must be detected as 0.6B"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_is_qwen3_embedding_06b_dir_detection() {
    use std::path::Path;
    assert!(is_qwen3_embedding_0_6b_dir(Path::new(
        "/models/qwen3-embedding-0.6b"
    )));
    assert!(is_qwen3_embedding_0_6b_dir(Path::new(
        "/models/qwen3_embedding_0_6b"
    )));
    assert!(
        !is_qwen3_embedding_0_6b_dir(Path::new("/models/Qwen3-Embedding-0.6B")),
        "case-sensitive: uppercase 0.6B does not match lowercase check"
    );
    assert!(!is_qwen3_embedding_0_6b_dir(Path::new(
        "/models/qwen3-embedding-4b"
    )));
    assert!(!is_qwen3_embedding_0_6b_dir(Path::new(
        "/models/some-other-model"
    )));
    assert!(!is_qwen3_embedding_0_6b_dir(Path::new("/")));
}

#[test]
fn test_eos_token_id_wired_correctly() {
    let tmp = std::env::temp_dir().join("lattice_test_eos_wired");
    std::fs::create_dir_all(&tmp).unwrap();

    let custom_eos: u32 = 151_645;
    let json = format!(r#"{{"eos_token_id": {custom_eos}}}"#);
    std::fs::write(tmp.join("inference_config.json"), &json).unwrap();

    let cfg = ModelInferenceConfig::load(&tmp);
    assert_eq!(
        cfg.eos_token_id, custom_eos,
        "load() must read eos_token_id from inference_config.json"
    );
    assert_eq!(
        cfg.rope_table_max_seq_len, 8192,
        "unset rope field must get 0.6B default"
    );
    assert_eq!(
        cfg.gpu_max_seq_len, 2048,
        "unset gpu field must get 0.6B default"
    );

    let full_json =
        r#"{"eos_token_id": 99999, "rope_table_max_seq_len": 16384, "gpu_max_seq_len": 1024}"#;
    std::fs::write(tmp.join("inference_config.json"), full_json).unwrap();
    let cfg2 = ModelInferenceConfig::load(&tmp);
    assert_eq!(cfg2.eos_token_id, 99_999);
    assert_eq!(cfg2.rope_table_max_seq_len, 16_384);
    assert_eq!(cfg2.gpu_max_seq_len, 1024);

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_model_inference_config_load_missing_file() {
    let dir = std::env::temp_dir().join("lattice_test_missing_cfg");
    std::fs::create_dir_all(&dir).ok();
    let cfg = ModelInferenceConfig::load(&dir);
    assert_eq!(cfg.eos_token_id, 151_643);
    assert_eq!(cfg.rope_table_max_seq_len, 8192);
    assert_eq!(cfg.gpu_max_seq_len, 2048);
    // Neither config.json nor tokenizer.json exist in `dir` either, so
    // derivation has no source bytes to hash: the fields must stay at
    // their "none" sentinel rather than gain a new construction failure.
    assert_eq!(cfg.base_model_rev, "none");
    assert_eq!(cfg.tokenizer_rev, "none");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_model_inference_config_load_derives_cache_compat_rev() {
    let tmp = std::env::temp_dir().join("lattice_test_derive_cache_compat_rev");
    std::fs::create_dir_all(&tmp).unwrap();
    let config_json_bytes = b"{\"hidden_size\":1024}";
    let tokenizer_json_bytes = b"{\"vocab\":{}}";
    std::fs::write(tmp.join("config.json"), config_json_bytes).unwrap();
    std::fs::write(tmp.join("tokenizer.json"), tokenizer_json_bytes).unwrap();

    // No inference_config.json at all: both fields start at the "none"
    // sentinel and must be derived from the sibling files' content hash.
    let cfg = ModelInferenceConfig::load(&tmp);
    let expected_base = format!(
        "sha256:{}",
        &embedding_cache_sha256_hex(config_json_bytes)[..16]
    );
    let expected_tok = format!(
        "sha256:{}",
        &embedding_cache_sha256_hex(tokenizer_json_bytes)[..16]
    );
    assert_eq!(cfg.base_model_rev, expected_base);
    assert_eq!(cfg.tokenizer_rev, expected_tok);
    assert!(cfg.base_model_rev.starts_with("sha256:"));
    assert_eq!(cfg.base_model_rev.len(), "sha256:".len() + 16);
    assert_ne!(
        cfg.base_model_rev, cfg.tokenizer_rev,
        "distinct source files must derive distinct revisions"
    );

    // An explicit value in inference_config.json always wins over
    // derivation, even though config.json/tokenizer.json are present.
    std::fs::write(
        tmp.join("inference_config.json"),
        r#"{"base_model_rev": "explicit-rev", "tokenizer_rev": "explicit-tok"}"#,
    )
    .unwrap();
    let cfg2 = ModelInferenceConfig::load(&tmp);
    assert_eq!(cfg2.base_model_rev, "explicit-rev");
    assert_eq!(cfg2.tokenizer_rev, "explicit-tok");

    std::fs::remove_dir_all(&tmp).ok();
}

/// Writes `len` bytes of a deterministic (non-zero, non-repeating-run)
/// pattern to `path`, for `derive_base_model_rev` weight-fingerprint
/// tests. No real model weights needed — the derivation only samples
/// filename/length/boundary bytes.
fn write_pattern_bytes(path: &Path, len: usize) {
    let bytes: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
    std::fs::write(path, bytes).unwrap();
}

/// Flips one byte (XOR 0xFF) at `offset` in the file at `path`.
fn flip_byte_at(path: &Path, offset: usize) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[offset] ^= 0xFF;
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn derive_base_model_rev_stable_across_calls() {
    let tmp = std::env::temp_dir().join("lattice_test_derive_base_rev_stable");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), b"{\"hidden_size\":1024}").unwrap();
    write_pattern_bytes(&tmp.join("model.safetensors"), 3 * 1024 * 1024);

    let rev1 = derive_base_model_rev(&tmp);
    let rev2 = derive_base_model_rev(&tmp);
    assert!(rev1.is_some());
    assert_eq!(rev1, rev2, "deriving twice on the same dir must be stable");

    std::fs::remove_dir_all(&tmp).ok();
}

// #1380: `derive_base_model_rev` folds weight-shard bytes into the
// revision fingerprint via `fold_weight_file_into_hasher`, which used to
// call plain `File::open` before any file-type check ran. A FIFO
// planted at the weight-shard path -- reachable the moment a checkpoint
// directory is pointed at by config, before any tensor is even loaded --
// blocked this call indefinitely, defeating the doc comment's own
// "degrades gracefully to the config-only derivation" promise for a read
// failure. Proves the fix the same way as the other loader entry
// points: the call must return (any return) inside a deadline; asserts
// separately that it took the documented graceful-degradation path
// (`Some` config-only rev, not a panic or a hang).
#[test]
fn derive_base_model_rev_rejects_a_fifo_without_blocking() {
    // The single-file `model.safetensors` path is deliberately NOT used
    // here: `resolve_weight_fingerprint_files` gates that branch on
    // `.is_file()`, which is already `false` for a FIFO, so a FIFO named
    // `model.safetensors` never reaches `fold_weight_file_into_hasher`
    // at all -- that branch was never part of this defect. The sharded
    // `weight_map` branch has no such filter (it forwards manifest
    // strings unchecked, by the same threat model as
    // `open_manifest_entry_once`), so a FIFO named by the index is what
    // actually exercises the guard this test is for.
    let tmp = std::env::temp_dir().join("lattice_test_derive_base_rev_fifo");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), b"{\"hidden_size\":1024}").unwrap();
    let index_json = r#"{"weight_map": {"a.weight": "shard0.safetensors"}}"#;
    std::fs::write(tmp.join("model.safetensors.index.json"), index_json).unwrap();
    let status = std::process::Command::new("mkfifo")
        .arg(tmp.join("shard0.safetensors"))
        .status()
        .expect("run mkfifo");
    assert!(status.success(), "mkfifo must succeed to set up this test");

    let (tx, rx) = std::sync::mpsc::channel();
    let probe_dir = tmp.clone();
    std::thread::spawn(move || {
        let result = derive_base_model_rev(&probe_dir);
        let _ = tx.send(result);
    });
    let result = rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
        "derive_base_model_rev did not return within 5s -- it blocked on the planted \
             FIFO, meaning the open-time regular-file guard regressed",
    );
    assert!(
        result.is_some(),
        "a FIFO weight file must degrade gracefully to the config-only revision, \
             not return None"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn derive_base_model_rev_changes_on_first_1mib_flip() {
    let tmp = std::env::temp_dir().join("lattice_test_derive_base_rev_head_flip");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), b"{\"hidden_size\":1024}").unwrap();
    let weights_path = tmp.join("model.safetensors");
    write_pattern_bytes(&weights_path, 3 * 1024 * 1024);

    let before = derive_base_model_rev(&tmp);
    flip_byte_at(&weights_path, 512 * 1024); // inside the first 1 MiB
    let after = derive_base_model_rev(&tmp);
    assert_ne!(
        before, after,
        "flipping a byte in the first 1 MiB of a >2 MiB weight file must change the rev"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn derive_base_model_rev_changes_on_last_1mib_flip() {
    let tmp = std::env::temp_dir().join("lattice_test_derive_base_rev_tail_flip");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), b"{\"hidden_size\":1024}").unwrap();
    let weights_path = tmp.join("model.safetensors");
    let len = 3 * 1024 * 1024;
    write_pattern_bytes(&weights_path, len);

    let before = derive_base_model_rev(&tmp);
    flip_byte_at(&weights_path, len - 512 * 1024); // inside the last 1 MiB
    let after = derive_base_model_rev(&tmp);
    assert_ne!(
        before, after,
        "flipping a byte in the last 1 MiB of a >2 MiB weight file must change the rev"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn derive_base_model_rev_changes_on_length_change() {
    let tmp = std::env::temp_dir().join("lattice_test_derive_base_rev_len_change");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), b"{\"hidden_size\":1024}").unwrap();
    let weights_path = tmp.join("model.safetensors");
    write_pattern_bytes(&weights_path, 4096);

    let before = derive_base_model_rev(&tmp);
    let mut bytes = std::fs::read(&weights_path).unwrap();
    bytes.push(0u8);
    std::fs::write(&weights_path, &bytes).unwrap();
    let after = derive_base_model_rev(&tmp);
    assert_ne!(
        before, after,
        "changing the weight file length (content otherwise unchanged) must change the rev"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn derive_base_model_rev_sharded_flip_second_shard() {
    let tmp = std::env::temp_dir().join("lattice_test_derive_base_rev_sharded");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), b"{\"hidden_size\":1024}").unwrap();
    let shard1 = tmp.join("model-00001-of-00002.safetensors");
    let shard2 = tmp.join("model-00002-of-00002.safetensors");
    write_pattern_bytes(&shard1, 4096);
    write_pattern_bytes(&shard2, 4096);
    let index_json = r#"{"weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
            "c.weight": "model-00002-of-00002.safetensors"
        }}"#;
    std::fs::write(tmp.join("model.safetensors.index.json"), index_json).unwrap();

    let before = derive_base_model_rev(&tmp);
    flip_byte_at(&shard2, 10);
    let after = derive_base_model_rev(&tmp);
    assert_ne!(
        before, after,
        "flipping a byte in the second (non-first-mentioned) shard must change the rev"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn derive_base_model_rev_mixed_layout_follows_loader_precedence() {
    let tmp = std::env::temp_dir().join("lattice_test_derive_base_rev_mixed_layout");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), b"{\"hidden_size\":1024}").unwrap();

    // Both a single-file checkpoint AND a sharded index are present.
    // `QwenModel::from_directory` loads `model.safetensors` first in
    // this layout, so the fingerprint must track that file, not the
    // index's shard.
    let single_path = tmp.join("model.safetensors");
    write_pattern_bytes(&single_path, 4096);

    let shard_path = tmp.join("model-00001-of-00001.safetensors");
    write_pattern_bytes(&shard_path, 4096);
    let index_json = r#"{"weight_map": {
            "a.weight": "model-00001-of-00001.safetensors"
        }}"#;
    std::fs::write(tmp.join("model.safetensors.index.json"), index_json).unwrap();

    let before = derive_base_model_rev(&tmp);

    flip_byte_at(&single_path, 10);
    let after_single_flip = derive_base_model_rev(&tmp);
    assert_ne!(
        before, after_single_flip,
        "flipping a byte in the actually-loaded model.safetensors must change the rev"
    );

    flip_byte_at(&single_path, 10); // restore
    flip_byte_at(&shard_path, 10);
    let after_shard_flip = derive_base_model_rev(&tmp);
    assert_eq!(
        before, after_shard_flip,
        "flipping a byte in the unused index shard must NOT change the rev — the \
             fingerprint follows the file the loader actually loads"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn derive_base_model_rev_no_weight_files_equals_config_only() {
    let tmp = std::env::temp_dir().join("lattice_test_derive_base_rev_no_weights");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    let config_bytes = b"{\"hidden_size\":1024}";
    std::fs::write(tmp.join("config.json"), config_bytes).unwrap();

    let rev = derive_base_model_rev(&tmp);
    let expected = derive_cache_compat_rev(&tmp.join("config.json"));
    assert!(rev.is_some());
    assert_eq!(
        rev, expected,
        "with no weight files present, derivation must equal the config-only derivation"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn test_model_inference_config_load_malformed_json() {
    let tmp = std::env::temp_dir().join("lattice_test_malformed_cfg");
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("inference_config.json"), "not valid json {{{").unwrap();
    let cfg = ModelInferenceConfig::load(&tmp);
    assert_eq!(
        cfg.eos_token_id, 151_643,
        "malformed JSON must fall back to defaults"
    );
    assert_eq!(cfg.rope_table_max_seq_len, 8192);
    assert_eq!(cfg.gpu_max_seq_len, 2048);
    std::fs::remove_dir_all(&tmp).ok();
}

/// Builds an intact, fully-wrapped embedding cache file for one record
/// with the given compatibility fields and floats.
fn build_test_embedding_cache_bytes(
    base_model_rev: &str,
    tokenizer_rev: &str,
    dim: usize,
    floats: Vec<f32>,
) -> Vec<u8> {
    let mut cache = HashMap::new();
    cache.insert(42u64, floats);
    let payload = serialize_embedding_cache_payload(&cache);
    wrap_embedding_cache_payload(payload, base_model_rev, tokenizer_rev, dim, 1).unwrap()
}

#[test]
fn qwen_embedding_cache_intact_manifest_loads() {
    let bytes = build_test_embedding_cache_bytes("model-a", "tok-a", 3, vec![1.0, 2.0, 3.0]);
    let entries = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, 42);
    assert_eq!(entries[0].1, vec![1.0, 2.0, 3.0]);
}

#[test]
fn qwen_embedding_cache_rejects_corrupt_payload_byte() {
    let mut bytes = build_test_embedding_cache_bytes("model-a", "tok-a", 3, vec![1.0, 2.0, 3.0]);
    let manifest_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let payload_start = 12 + manifest_len;
    bytes[payload_start] ^= 0xFF;
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("payload_sha256 mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_wrong_base_model_rev() {
    let bytes = build_test_embedding_cache_bytes("model-a", "tok-a", 3, vec![1.0, 2.0, 3.0]);
    let err = parse_embedding_cache_file(&bytes, 3, "model-b", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("base_model_rev mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_wrong_tokenizer_rev() {
    let bytes = build_test_embedding_cache_bytes("model-a", "tok-a", 3, vec![1.0, 2.0, 3.0]);
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-b").unwrap_err();
    assert!(
        format!("{err}").contains("tokenizer_rev mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_wrong_dim_even_with_valid_hash() {
    let bytes = build_test_embedding_cache_bytes("model-a", "tok-a", 3, vec![1.0, 2.0, 3.0]);
    // Payload SHA-256 and compat fields are untouched and would pass;
    // only the caller-expected dimension differs from the manifest.
    let err = parse_embedding_cache_file(&bytes, 4, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("embedding_dim mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_non_finite_payload() {
    let bytes = build_test_embedding_cache_bytes("model-a", "tok-a", 3, vec![1.0, f32::NAN, 3.0]);
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("non-finite float"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_trailing_payload_bytes() {
    let mut cache = HashMap::new();
    cache.insert(42u64, vec![1.0f32, 2.0, 3.0]);
    let mut payload = serialize_embedding_cache_payload(&cache);
    // Trailing bytes past the one declared record: the SHA-256 is
    // computed over the payload including this garbage, so the hash
    // check alone would pass — only exact-consumption validation
    // against the manifest-declared entry count catches it.
    payload.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
    let bytes = wrap_embedding_cache_payload(payload, "model-a", "tok-a", 3, 1).unwrap();
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("trailing bytes"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_missing_file_returns_ok_zero() {
    let path = std::env::temp_dir().join("lattice_test_embedding_cache_missing.bin");
    std::fs::remove_file(&path).ok();
    let data = read_embedding_cache_file(&path, 3);
    assert!(matches!(&data, Err(e) if e.kind() == std::io::ErrorKind::NotFound));
}

#[test]
fn qwen_embedding_cache_bounded_read_rejects_oversized_file() {
    // The read is bounded independently of the stat fast-path: a file
    // larger than the cap is rejected rather than fully allocated. This
    // closes the stat-to-read window (a file that grows after the stat).
    let path = std::env::temp_dir().join("lattice_test_embedding_cache_oversized.bin");
    std::fs::write(&path, vec![0u8; 64]).unwrap();
    let err =
        read_embedding_cache_file_bounded(&path, 16).expect_err("oversized file must be rejected");
    std::fs::remove_file(&path).ok();
    assert!(
        format!("{err}").contains("too large"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_bounded_read_accepts_within_cap() {
    let path = std::env::temp_dir().join("lattice_test_embedding_cache_within_cap.bin");
    std::fs::write(&path, vec![7u8; 16]).unwrap();
    let bytes = read_embedding_cache_file_bounded(&path, 32).expect("file within cap must be read");
    std::fs::remove_file(&path).ok();
    assert_eq!(bytes, vec![7u8; 16]);
}

#[test]
fn qwen_embedding_cache_rejects_wrong_magic() {
    let mut bytes = build_test_embedding_cache_bytes("model-a", "tok-a", 3, vec![1.0, 2.0, 3.0]);
    bytes[0] = b'X';
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("magic mismatch"),
        "unexpected error: {err}"
    );
}

/// Raw `[magic][manifest_len][manifest JSON][payload]` bytes built by hand
/// (bypassing `wrap_embedding_cache_payload`) so a test can set manifest
/// fields — like `version` — that the wrapper always fills in correctly.
fn build_raw_cache_bytes(manifest_json: &str, payload: &[u8]) -> Vec<u8> {
    let manifest_bytes = manifest_json.as_bytes();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(EMBEDDING_CACHE_MAGIC);
    bytes.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(manifest_bytes);
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn qwen_embedding_cache_rejects_version_mismatch() {
    // The version check runs before the payload hash check, so a bogus
    // payload_sha256 here is fine — it must never be reached.
    let manifest = r#"{"version":99,"payload_sha256":"deadbeef","base_model_rev":"model-a","tokenizer_rev":"tok-a","embedding_dim":3,"entry_count":0}"#;
    let bytes = build_raw_cache_bytes(manifest, &[]);
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("version mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_truncated_manifest() {
    // manifest_len declares 100 bytes but far fewer actually follow.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(EMBEDDING_CACHE_MAGIC);
    bytes.extend_from_slice(&100u32.to_le_bytes());
    bytes.extend_from_slice(b"{\"short\":true}");
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("manifest truncated"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_manifest_len_over_cap() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(EMBEDDING_CACHE_MAGIC);
    let over_cap = EMBEDDING_CACHE_MANIFEST_CAP + 1;
    bytes.extend_from_slice(&over_cap.to_le_bytes());
    // No manifest bytes needed: the cap check rejects before any read
    // past the 12-byte header.
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("manifest length") && msg.contains("exceeds cap"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_entry_count_over_cap() {
    let bytes =
        wrap_embedding_cache_payload(Vec::new(), "model-a", "tok-a", 3, EMBEDDING_CACHE_CAP + 1)
            .unwrap();
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("entry_count") && msg.contains("exceeds cap"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_record_dim_mismatch() {
    // The record's own dim field disagrees with the manifest-declared /
    // caller-expected dimension, even though the manifest-level
    // `embedding_dim` and the payload hash both match — only the
    // per-record dim field is wrong.
    let mut payload = Vec::new();
    payload.extend_from_slice(&42u64.to_le_bytes());
    payload.extend_from_slice(&4u32.to_le_bytes()); // record claims dim=4
    for f in [1.0f32, 2.0, 3.0] {
        payload.extend_from_slice(&f.to_le_bytes());
    }
    let bytes = wrap_embedding_cache_payload(payload, "model-a", "tok-a", 3, 1).unwrap();
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("record dimension mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn qwen_embedding_cache_rejects_truncated_record() {
    // entry_count declares 2 records but the payload holds only one
    // record's worth of bytes.
    let mut cache = HashMap::new();
    cache.insert(42u64, vec![1.0f32, 2.0, 3.0]);
    let payload = serialize_embedding_cache_payload(&cache);
    let bytes = wrap_embedding_cache_payload(payload, "model-a", "tok-a", 3, 2).unwrap();
    let err = parse_embedding_cache_file(&bytes, 3, "model-a", "tok-a").unwrap_err();
    assert!(
        format!("{err}").contains("truncated record"),
        "unexpected error: {err}"
    );
}

/// No-op tokenizer for the `QwenModel` cache-test fixture below.
/// `cache_load`/`cache_save` never touch the tokenizer, so every method
/// here is unreachable in the tests that use the fixture.
struct NullTokenizer;
impl Tokenizer for NullTokenizer {
    fn tokenize(&self, _text: &str) -> crate::tokenizer::common::TokenizedInput {
        unimplemented!("NullTokenizer is a cache-test fixture; tokenize is never called")
    }
    fn tokenize_batch(&self, _texts: &[&str]) -> Vec<crate::tokenizer::common::TokenizedInput> {
        unimplemented!("NullTokenizer is a cache-test fixture; tokenize_batch is never called")
    }
    fn vocab_size(&self) -> usize {
        0
    }
    fn max_seq_len(&self) -> usize {
        0
    }
}

struct TwoTokenTokenizer;
impl Tokenizer for TwoTokenTokenizer {
    fn tokenize(&self, _text: &str) -> crate::tokenizer::common::TokenizedInput {
        crate::tokenizer::common::TokenizedInput {
            input_ids: vec![0, 1],
            attention_mask: vec![1, 1],
            token_type_ids: vec![0, 0],
            real_length: 2,
            pre_truncation_len: 2,
        }
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Vec<crate::tokenizer::common::TokenizedInput> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }

    fn vocab_size(&self) -> usize {
        2
    }

    fn max_seq_len(&self) -> usize {
        2
    }
}

/// Build a `QwenModel` fixture with no real transformer weights — just
/// enough to exercise the public `cache_load`/`cache_save` contract,
/// which only touch `inference_config`, `cache`, and `dimensions()`
/// (derived from `config.hidden_size` / `output_dim`), never `weights`,
/// `tokenizer`, or `rope`. `from_directory` requires real safetensors
/// weight files that unit tests don't have, so this constructs the
/// struct directly (accessible since `mod tests` is a descendant of the
/// module that declares `QwenModel`'s private fields).
fn build_cache_test_model(inference_config: ModelInferenceConfig, dim: usize) -> QwenModel {
    let config = QwenConfig {
        vocab_size: 8,
        hidden_size: dim,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        intermediate_size: 1,
        max_position_embeddings: 8,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
    };
    let rope = RopeTable::new(2, 1, 10_000.0);
    let weights = QwenWeights {
        embed_tokens: crate::weights::Tensor2D {
            data: &[],
            rows: 0,
            cols: 0,
        },
        norm_weight: crate::weights::Tensor1D { data: &[], len: 0 },
        layers: Vec::new(),
    };
    let storage = ShardedQwenBacking {
        embed_tokens: Vec::new(),
        norm_weight: Vec::new(),
        q_proj: Vec::new(),
        k_proj: Vec::new(),
        v_proj: Vec::new(),
        o_proj: Vec::new(),
        q_norm: Vec::new(),
        k_norm: Vec::new(),
        input_ln: Vec::new(),
        gate_proj: Vec::new(),
        up_proj: Vec::new(),
        down_proj: Vec::new(),
        post_ln: Vec::new(),
    };
    QwenModel {
        config,
        inference_config,
        tokenizer: Box::new(NullTokenizer),
        rope,
        output_dim: None,
        buffers: Mutex::new(ForwardBuffers::new()),
        cache: Mutex::new(HashMap::new()),
        metal: None,
        weights: ManuallyDrop::new(weights),
        _storage: ManuallyDrop::new(SafetensorsStorage::Sharded(Box::new(storage))),
    }
}

fn build_rope_capacity_test_model() -> QwenModel {
    static EMBED_TOKENS: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    static NORM_WEIGHT: [f32; 2] = [1.0, 1.0];

    let mut model = build_cache_test_model(
        ModelInferenceConfig {
            eos_token_id: 1,
            rope_table_max_seq_len: 1,
            ..Default::default()
        },
        2,
    );
    model.tokenizer = Box::new(TwoTokenTokenizer);
    model.weights = ManuallyDrop::new(QwenWeights {
        embed_tokens: crate::weights::Tensor2D {
            data: &EMBED_TOKENS,
            rows: 2,
            cols: 2,
        },
        norm_weight: crate::weights::Tensor1D {
            data: &NORM_WEIGHT,
            len: 2,
        },
        layers: Vec::new(),
    });
    model
}

/// Base config for `effective_context_len` tests — cheap to construct, none of
/// these fields participate in the min() policy under test.
fn context_len_test_config(max_position_embeddings: usize) -> QwenConfig {
    QwenConfig {
        vocab_size: 8,
        hidden_size: 2,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        intermediate_size: 1,
        max_position_embeddings,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
    }
}

#[test]
fn effective_context_len_picks_max_position_embeddings_when_tightest() {
    let config = context_len_test_config(10);
    let inference_config = ModelInferenceConfig {
        rope_table_max_seq_len: 50,
        ..Default::default()
    };
    let result = effective_context_len(&config, &inference_config, 1000).unwrap();
    assert_eq!(
        result, 10,
        "must take the minimum, not max_position_embeddings alone"
    );
}

#[test]
fn effective_context_len_picks_rope_table_max_seq_len_when_tightest() {
    let config = context_len_test_config(100);
    let inference_config = ModelInferenceConfig {
        rope_table_max_seq_len: 20,
        ..Default::default()
    };
    let result = effective_context_len(&config, &inference_config, 1000).unwrap();
    assert_eq!(
        result, 20,
        "must take the minimum, not max_position_embeddings alone"
    );
}

#[test]
fn effective_context_len_picks_tokenizer_max_seq_len_when_tightest() {
    let config = context_len_test_config(100);
    let inference_config = ModelInferenceConfig {
        rope_table_max_seq_len: 50,
        ..Default::default()
    };
    let result = effective_context_len(&config, &inference_config, 5).unwrap();
    assert_eq!(result, 5, "must take the minimum, not the other two bounds");
}

#[test]
fn effective_context_len_rejects_zero_result() {
    // max_position_embeddings=0 forces the min() to 0 regardless of the other
    // two bounds — this is the constructor policy `RopeTable::new` depends on
    // (a zero-length RoPE table) and must fail closed, not build a table nobody
    // can ever index into.
    let config = context_len_test_config(0);
    let inference_config = ModelInferenceConfig::default();
    let err = effective_context_len(&config, &inference_config, 100)
        .expect_err("zero effective context must be rejected");
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn qwen_encode_rejects_input_beyond_rope_capacity() {
    let model = build_rope_capacity_test_model();
    let err = model
        .encode("two tokens")
        .expect_err("two tokens must exceed the one-position RoPE table");
    assert!(matches!(err, InferenceError::InvalidInput(_)));
    let message = err.to_string();
    assert!(
        message.contains("2") && message.contains("1"),
        "error must identify the admitted length and RoPE capacity; got {message}"
    );
}

#[test]
fn qwen_cache_load_missing_path_returns_ok_zero() {
    // Public contract of `cache_load`: a missing cache file is not an
    // error (no artifact was expected to be loaded), so it returns
    // `Ok(0)` rather than propagating the NotFound error.
    let model = build_cache_test_model(ModelInferenceConfig::default(), 3);
    let path = std::env::temp_dir().join("lattice_test_cache_load_missing_public.bin");
    std::fs::remove_file(&path).ok();
    let result = model.cache_load(&path);
    assert!(
        matches!(result, Ok(0)),
        "cache_load on a missing path must return Ok(0), got {result:?}"
    );
}

#[test]
fn qwen_cache_save_then_load_round_trip() {
    let tmp = std::env::temp_dir().join("lattice_test_cache_save_load_round_trip");
    // Clean up any leftovers from a prior interrupted/failed run before
    // creating, so this test's own assertions can't be polluted by them.
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("cache.bin");

    let cfg = ModelInferenceConfig {
        base_model_rev: "model-a".to_string(),
        tokenizer_rev: "tok-a".to_string(),
        ..Default::default()
    };
    let saver = build_cache_test_model(cfg.clone(), 3);
    saver.cache.lock().unwrap().insert(42, vec![1.0, 2.0, 3.0]);
    let saved = saver.cache_save(&path).unwrap();
    assert_eq!(saved, 1);

    let loader = build_cache_test_model(cfg, 3);
    let loaded = loader.cache_load(&path).unwrap();
    assert_eq!(loaded, 1);
    assert_eq!(
        loader.cache.lock().unwrap().get(&42),
        Some(&vec![1.0f32, 2.0, 3.0])
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn qwen_cache_save_leaves_no_tmp_file_behind() {
    // FIX 2 regression: `cache_save` writes to a temp file and renames
    // it over the target, so no `.tmp.<pid>.<seq>` file should remain in
    // the directory once `cache_save` returns `Ok`.
    let tmp = std::env::temp_dir().join("lattice_test_cache_save_no_tmp_leftover");
    // Clean up any leftovers from a prior interrupted/failed run before
    // creating, so this test's own assertions can't be polluted by them.
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("cache.bin");

    let cfg = ModelInferenceConfig {
        base_model_rev: "model-a".to_string(),
        tokenizer_rev: "tok-a".to_string(),
        ..Default::default()
    };
    let model = build_cache_test_model(cfg, 3);
    model.cache.lock().unwrap().insert(42, vec![1.0, 2.0, 3.0]);
    let n = model.cache_save(&path).unwrap();
    assert_eq!(n, 1);
    assert!(path.exists(), "final cache file must exist after save");

    let leftover: Vec<_> = std::fs::read_dir(&tmp)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(
        leftover.is_empty(),
        "temp file(s) left behind after cache_save: {leftover:?}"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn qwen_cache_load_rejects_stale_cache_after_weight_only_change() {
    // FIX-A end-to-end regression: two checkpoints sharing config.json
    // (and tokenizer.json, absent here) but differing only in weights
    // must NOT derive the same `base_model_rev` — otherwise a cache
    // built against the old weights would silently pass `cache_load`'s
    // compat check against the new ones.
    let tmp = std::env::temp_dir().join("lattice_test_cache_rejects_weight_drift");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), b"{\"hidden_size\":1024}").unwrap();
    let weights_path = tmp.join("model.safetensors");
    write_pattern_bytes(&weights_path, 4096);
    let cache_path = tmp.join("cache.bin");

    let cfg1 = ModelInferenceConfig::load(&tmp);
    let saver = build_cache_test_model(cfg1.clone(), 3);
    saver.cache.lock().unwrap().insert(42, vec![1.0, 2.0, 3.0]);
    saver.cache_save(&cache_path).unwrap();

    // Weight-only drift: config.json (and the absent tokenizer.json) are
    // untouched, only the weight file's bytes change.
    flip_byte_at(&weights_path, 10);

    let cfg2 = ModelInferenceConfig::load(&tmp);
    assert_ne!(
        cfg1.base_model_rev, cfg2.base_model_rev,
        "weight-only drift must change the derived base_model_rev"
    );

    let loader = build_cache_test_model(cfg2, 3);
    let err = loader.cache_load(&cache_path).unwrap_err();
    assert!(
        format!("{err}").contains("base_model_rev"),
        "error must name base_model_rev, got: {err}"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn qwen_cache_save_sequential_calls_leave_one_final_file() {
    // FIX-B regression: repeated `cache_save` calls in the same process
    // (now each minting a distinct `<name>.tmp.<pid>.<seq>` name) must
    // still converge on exactly one final file with no temp leftovers.
    let tmp = std::env::temp_dir().join("lattice_test_cache_save_sequential");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("cache.bin");

    let cfg = ModelInferenceConfig {
        base_model_rev: "model-a".to_string(),
        tokenizer_rev: "tok-a".to_string(),
        ..Default::default()
    };
    let model = build_cache_test_model(cfg, 3);
    for i in 0..5u64 {
        model
            .cache
            .lock()
            .unwrap()
            .insert(i, vec![i as f32, 0.0, 0.0]);
        let n = model.cache_save(&path).unwrap();
        assert_eq!(n, (i + 1) as usize);
    }

    let entries: Vec<_> = std::fs::read_dir(&tmp)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "exactly one final file expected, found: {entries:?}"
    );
    assert_eq!(
        entries[0].file_name(),
        std::ffi::OsString::from("cache.bin")
    );

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn qwen_cache_save_concurrent_saves_no_torn_output() {
    // FIX-B regression: distinct threads in one process sharing the same
    // final cache path must not collide on `cache_save`'s temp file name
    // and clobber/tear each other's write. Each thread saves a
    // DIFFERENT single-entry cache to the SAME final path; whichever
    // thread's rename lands last must win cleanly (exactly one intact
    // entry), never a torn or merged file.
    //
    // Mutation-sensitivity note (see task report): this exercises real
    // OS thread interleaving, so it is a probabilistic guard, not a
    // deterministic one — under the reverted PID-only naming the race
    // window is narrow (temp file open + write + rename), so failure
    // isn't guaranteed on every run even though the bug is real.
    let tmp = std::env::temp_dir().join("lattice_test_cache_save_concurrent");
    std::fs::remove_dir_all(&tmp).ok();
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("cache.bin");

    let cfg = ModelInferenceConfig {
        base_model_rev: "model-a".to_string(),
        tokenizer_rev: "tok-a".to_string(),
        ..Default::default()
    };

    const N: u64 = 8;
    let candidates: Vec<(u64, Vec<f32>)> = (0..N)
        .map(|i| (i, vec![i as f32, i as f32 + 1.0, i as f32 + 2.0]))
        .collect();

    let handles: Vec<_> = candidates
        .clone()
        .into_iter()
        .map(|(key, floats)| {
            let cfg = cfg.clone();
            let path = path.clone();
            std::thread::spawn(move || {
                let model = build_cache_test_model(cfg, 3);
                model.cache.lock().unwrap().insert(key, floats);
                model.cache_save(&path)
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("save thread must not panic")
            .expect("cache_save must not error under concurrent same-path saves");
    }

    let loader = build_cache_test_model(cfg, 3);
    let loaded = loader.cache_load(&path).unwrap();
    assert_eq!(
        loaded, 1,
        "final file must contain exactly one thread's single entry, no torn/merged output"
    );
    {
        let cache = loader.cache.lock().unwrap();
        assert_eq!(cache.len(), 1);
        let (&winning_key, winning_floats) = cache.iter().next().unwrap();
        let expected = candidates
            .iter()
            .find(|(k, _)| *k == winning_key)
            .expect("winning key must be exactly one of the 8 candidate entries");
        assert_eq!(winning_floats, &expected.1);
    }

    let leftover: Vec<_> = std::fs::read_dir(&tmp)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(
        leftover.is_empty(),
        "temp file(s) left behind after concurrent cache_save: {leftover:?}"
    );

    std::fs::remove_dir_all(&tmp).ok();
}

// ---------------------------------------------------------------
// ADR-080 C1: `causal_softmax_row` (shared by the `forward_all_layers` /
// `forward_profiled` secondary-prefill duplicates, #780) must fail
// closed on a NaN score, a `+inf` max, or a fully-masked row, not
// propagate garbage.
//
// Note on scope: an earlier version of this test built a full
// `QwenModel` fixture and poisoned a token's *embedding* to check that
// the NaN didn't leak into sibling positions' hidden states end to end.
// That test was infeasible and was removed: poisoning a token's
// embedding poisons Q, K, *and* V for that position (all three are
// linear projections of the same hidden row), and `0.0 * NaN == NaN` in
// IEEE 754 — so even a *correctly* fail-closed softmax weight of exactly
// `0.0` for the poisoned position still yields NaN once multiplied
// against that position's NaN V-vector in the context matmul
// (`scores @ V`). That is a property of plain matmul shared by the
// canonical `gqa.rs`/`decode.rs` implementations too (neither guards
// exact-zero-weight multiplication against a NaN operand); it is a
// separate, out-of-scope concern from the softmax row-finalizer contract
// this cluster fixes. `gqa.rs`'s own existing NaN tests
// (`fused_softmax_all_neg_inf_row_is_zero_not_nan`,
// `fused_softmax_pos_inf_row_is_zero_not_nan`) test at the same level as
// here: the softmax row function directly, not through a full
// model forward pass.
// ---------------------------------------------------------------

/// A NaN score anywhere in the visible (non-causally-masked) range must
/// zero the whole row. RED before the fix: `if sum > 0.0` with no `else`
/// left a NaN-poisoned row's NaN/garbage exponentials un-zeroed.
#[test]
fn causal_softmax_row_nan_score_fails_closed() {
    let mut row = vec![1.0f32, f32::NAN, 2.0];
    causal_softmax_row(&mut row, 2, 1.0);
    assert!(
        row.iter().all(|&v| v == 0.0),
        "expected all-zero row: {row:?}"
    );
}

/// A `+inf` score must zero the whole row (fail-closed on non-finite max).
#[test]
fn causal_softmax_row_pos_inf_score_fails_closed() {
    let mut row = vec![1.0f32, f32::INFINITY, 2.0];
    causal_softmax_row(&mut row, 2, 1.0);
    assert!(
        row.iter().all(|&v| v == 0.0),
        "expected all-zero row: {row:?}"
    );
}

/// The causal mask sets every `ki > qi` lane to `-inf` before the
/// fail-closed decision; a query that only sees itself (`qi == 0`) must
/// still normalize correctly (this is not a poisoned-row case).
#[test]
fn causal_softmax_row_masks_future_positions() {
    let mut row = vec![1.0f32, 2.0, 3.0];
    causal_softmax_row(&mut row, 0, 1.0);
    assert_eq!(
        row,
        vec![1.0, 0.0, 0.0],
        "query 0 must attend only to itself: {row:?}"
    );
}

/// A row that is fully masked to `-inf` (degenerate `qi` with no visible
/// keys) must zero out rather than compute `exp(NaN)` from a `-inf - -inf`
/// subtraction.
#[test]
fn causal_softmax_row_all_masked_zeros_row() {
    let mut row = vec![f32::NEG_INFINITY; 3];
    causal_softmax_row(&mut row, 0, 1.0);
    assert!(
        row.iter().all(|&v| v == 0.0),
        "expected all-zero row: {row:?}"
    );
}

/// A well-formed row still normalizes to a valid probability
/// distribution over the visible (non-masked) positions (guards against
/// the fixture trivially zeroing everything regardless of the fix).
#[test]
fn causal_softmax_row_normal_row_sums_to_one() {
    let mut row = vec![1.0f32, 2.0, 3.0, 4.0];
    causal_softmax_row(&mut row, 3, 1.0);
    assert!(row.iter().all(|v| v.is_finite() && *v >= 0.0));
    let sum: f32 = row.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "expected sum ~1.0, got {sum}");
}
