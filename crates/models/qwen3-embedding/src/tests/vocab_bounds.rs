use super::*;

fn qwen_vocab_fixture(vocab: usize, rows: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(
        root.join("config.json"),
        serde_json::json!({
            "vocab_size": vocab,
            "hidden_size": 2,
            "num_hidden_layers": 0,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "intermediate_size": 2,
            "max_position_embeddings": 8
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        root.join("vocab.txt"),
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\n",
    )
    .unwrap();
    std::fs::write(root.join("inference_config.json"), "{\"eos_token_id\":1}").unwrap();
    write_report_test_tensor(
        &root.join("a.safetensors"),
        "embed_tokens.weight",
        &[rows, 2],
        &vec![1.0; rows * 2],
    );
    write_report_test_tensor(&root.join("b.safetensors"), "norm.weight", &[2], &[1., 1.]);
    std::fs::write(
        root.join("model.safetensors.index.json"),
        r#"{"weight_map":{"embed_tokens.weight":"a.safetensors","norm.weight":"b.safetensors"}}"#,
    )
    .unwrap();
    dir
}

#[test]
fn qwen_embedding_rejects_mismatched_vocab_rows_at_load() {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let _guard = gpu_test_lock();

    let good = qwen_vocab_fixture(6, 6);
    let tokens = load_tokenizer(good.path()).unwrap().tokenize("a");
    assert!(tokens.input_ids[..tokens.real_length].contains(&5));
    let model = QwenModel::from_directory(good.path()).expect("matching checkpoint must load");
    assert_eq!(model.encode("a").expect("valid token must encode").len(), 2);

    let short = qwen_vocab_fixture(6, 5);
    match QwenModel::from_directory(short.path()) {
        Err(error) => {
            let message = error.to_string();
            assert!(message.contains("5") && message.contains("6"), "{message}");
        }
        Ok(_) => panic!("checkpoint row count must match config vocabulary"),
    }
}

#[test]
fn qwen_embedding_rejects_out_of_range_emitted_token() {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let _guard = gpu_test_lock();

    let dir = qwen_vocab_fixture(5, 5);
    let tokens = load_tokenizer(dir.path()).unwrap().tokenize("a");
    assert!(tokens.input_ids[..tokens.real_length].contains(&5));
    let model = QwenModel::from_directory(dir.path()).expect("row count matches config");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.encode("a")));
    let error = result
        .expect("out-of-range token must not panic")
        .expect_err("must refuse");
    let message = error.to_string();
    assert!(
        message.contains('5') && message.contains("token"),
        "{message}"
    );
}

#[test]
fn qwen_embedding_rejects_out_of_range_eos() {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let _guard = gpu_test_lock();

    let dir = qwen_vocab_fixture(5, 5);
    std::fs::write(
        dir.path().join("inference_config.json"),
        "{\"eos_token_id\":5}",
    )
    .unwrap();
    let tokens = load_tokenizer(dir.path()).unwrap().tokenize("unknown");
    assert!(
        tokens.input_ids[..tokens.real_length]
            .iter()
            .all(|&id| id < 5)
    );
    let model = QwenModel::from_directory(dir.path()).expect("row count matches config");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.encode("unknown")));
    let error = result
        .expect("out-of-range EOS must not panic")
        .expect_err("must refuse");
    let message = error.to_string();
    assert!(
        message.contains('5') && message.contains("token"),
        "{message}"
    );
}
