use std::fs;

pub const MIXED_INPUTS: [&str; 17] = [
    "hello world foo bar hello world foo bar hello world foo bar hello world foo bar",
    "foo",
    "hello world",
    "bar hello world foo bar hello",
    "",
    "world bar foo",
    "hello hello world",
    "bar",
    "foo world hello bar foo world hello bar foo world hello bar foo world hello bar hello world",
    "hello",
    "foo bar",
    "world world foo bar hello",
    "bar foo bar foo bar",
    "hello bar world",
    "world",
    "foo hello world bar foo hello world bar",
    "bar hello",
];

pub fn tiny_bert_checkpoint(layers: usize, heads: usize) -> tempfile::TempDir {
    assert!(layers > 0);
    assert!(heads > 0 && 4_usize.is_multiple_of(heads));
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("vocab.txt"),
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nfoo\nbar\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "vocab_size": 9, "hidden_size": 4, "num_hidden_layers": layers,
            "num_attention_heads": heads, "intermediate_size": 4,
            "max_position_embeddings": 32, "type_vocab_size": 2, "layer_norm_eps": 1e-5
        })
        .to_string(),
    )
    .unwrap();
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut seed = 0u32;
    let mut add = |name: &str, shape: &[usize], values: &[f32]| {
        let start = payload.len();
        // Small deterministic non-zero weights so embeddings depend on the
        // tokens and positions; LayerNorm weights stay as given.
        let is_norm_weight = name.ends_with("LayerNorm.weight");
        payload.extend(values.iter().flat_map(|value| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let jitter = ((seed >> 16) % 17) as f32 * 0.02 - 0.16;
            let value = if is_norm_weight {
                *value
            } else {
                value + jitter
            };
            value.to_le_bytes()
        }));
        header.insert(
            name.to_owned(),
            serde_json::json!({
                "dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]
            }),
        );
    };
    add("embeddings.word_embeddings.weight", &[9, 4], &[0.0; 36]);
    add(
        "embeddings.position_embeddings.weight",
        &[32, 4],
        &[0.0; 128],
    );
    add(
        "embeddings.token_type_embeddings.weight",
        &[2, 4],
        &[0.0; 8],
    );
    add("embeddings.LayerNorm.weight", &[4], &[1.0; 4]);
    add("embeddings.LayerNorm.bias", &[4], &[0.0; 4]);
    for layer in 0..layers {
        for name in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ] {
            add(
                &format!("encoder.layer.{layer}.{name}.weight"),
                &[4, 4],
                &[0.0; 16],
            );
            add(
                &format!("encoder.layer.{layer}.{name}.bias"),
                &[4],
                &[0.0; 4],
            );
        }
        for name in ["attention.output.LayerNorm", "output.LayerNorm"] {
            add(
                &format!("encoder.layer.{layer}.{name}.weight"),
                &[4],
                &[1.0; 4],
            );
            add(
                &format!("encoder.layer.{layer}.{name}.bias"),
                &[4],
                &[0.0; 4],
            );
        }
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
    dir
}
