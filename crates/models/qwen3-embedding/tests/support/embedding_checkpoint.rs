use lattice_model_qwen3_embedding::QwenModel;
use serde_json::json;

pub const HIDDEN: usize = 32;
pub const LENGTHS: [usize; 3] = [17, 5, 11];
pub const TEXTS: [&str; 3] = [
    "a b c a b c a b c a b c a b c",
    "c a b",
    "b c a b c a b c a",
];

pub fn checkpoint() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&json!({
            "vocab_size": 8,
            "hidden_size": HIDDEN,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 48,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("inference_config.json"),
        br#"{"eos_token_id":3,"rope_table_max_seq_len":64,"gpu_max_seq_len":64}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("vocab.txt"),
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\n",
    )
    .unwrap();

    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut seed = 0usize;
    let mut add = |name: &str, shape: &[usize], norm: bool| {
        seed += 1;
        let start = payload.len();
        for index in 0..shape.iter().product::<usize>() {
            let centered = ((index * 17 + seed * 13) % 97) as f32 - 48.0;
            let value = if norm {
                1.0 + centered * 0.002
            } else {
                centered * 0.004
            };
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.into(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    };
    add("embed_tokens.weight", &[8, HIDDEN], false);
    add("norm.weight", &[HIDDEN], true);
    for layer in 0..2 {
        for (suffix, shape, norm) in [
            ("self_attn.q_proj.weight", vec![32, HIDDEN], false),
            ("self_attn.k_proj.weight", vec![16, HIDDEN], false),
            ("self_attn.v_proj.weight", vec![16, HIDDEN], false),
            ("self_attn.o_proj.weight", vec![HIDDEN, 32], false),
            ("self_attn.q_norm.weight", vec![16], true),
            ("self_attn.k_norm.weight", vec![16], true),
            ("input_layernorm.weight", vec![HIDDEN], true),
            ("mlp.gate_proj.weight", vec![48, HIDDEN], false),
            ("mlp.up_proj.weight", vec![48, HIDDEN], false),
            ("mlp.down_proj.weight", vec![HIDDEN, 48], false),
            ("post_attention_layernorm.weight", vec![HIDDEN], true),
        ] {
            add(&format!("layers.{layer}.{suffix}"), &shape, norm);
        }
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload);
    std::fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
    dir
}

pub fn assert_fixture(model: &QwenModel) {
    assert!(!model.has_gpu(), "fixture must execute the CPU path");
    assert_eq!(model.config().hidden_size, HIDDEN);
    assert_eq!(model.config().num_hidden_layers, 2);
    assert_eq!(model.config().num_attention_heads, 2);
    assert_eq!(model.config().num_key_value_heads, 1);
    assert_eq!(model.config().intermediate_size, 48);
    assert_eq!(model.tokenizer().max_seq_len(), 64);
    for (text, length) in TEXTS.iter().zip(LENGTHS) {
        let input = model.tokenizer().tokenize(text);
        assert_eq!(input.input_ids.len(), 64);
        assert_eq!(input.real_length, length);
        assert_eq!(input.input_ids[length - 1], 3);
        assert!(input.attention_mask[..length].iter().all(|&mask| mask == 1));
        assert!(input.attention_mask[length..].iter().all(|&mask| mask == 0));
        assert_eq!(model.embedding_token_counts(text).embedded, length);
    }
}
