use super::BertModel;
use lattice_tokenizer::wordpiece::WordPieceTokenizer;
use lattice_weights::consumed::WeightDigestMode;
use lattice_weights::test_fixtures::bert_namespace::{
    HIDDEN, INTERMEDIATE, LAYERS, checkpoint, safetensors_bytes,
};
use std::path::Path;

const VOCAB: &str =
    "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nred\nblue\ngreen\ncat\ndog\nruns\nsleeps\n";
/// The same tokens with `red` and `blue` swapped, so the two files disagree.
const CONFLICTING_TOKENIZER_JSON: &str = r#"{"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"blue":5,"red":6,"green":7,"cat":8,"dog":9,"runs":10,"sleeps":11}}}"#;
const TEXT: &str = "red cat";

fn write_model(dir: &Path) {
    std::fs::create_dir(dir).unwrap();
    std::fs::write(dir.join("vocab.txt"), VOCAB).unwrap();
    std::fs::write(dir.join("tokenizer.json"), CONFLICTING_TOKENIZER_JSON).unwrap();
    let config = serde_json::json!({
        "vocab_size": 12,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 2,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": 32,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-5,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.join("model.safetensors"),
        safetensors_bytes(&checkpoint("", true)),
    )
    .unwrap();
}

fn vocab_ids() -> Vec<u32> {
    let input = WordPieceTokenizer::from_str(VOCAB).unwrap().tokenize(TEXT);
    input.input_ids[..input.real_length].to_vec()
}

fn ids(model: &BertModel) -> Vec<u32> {
    let input = model.tokenizer().tokenize(TEXT);
    input.input_ids[..input.real_length].to_vec()
}

#[test]
fn managed_vocab_directory_reads_the_pinned_vocab_not_tokenizer_json() {
    let root = tempfile::tempdir().unwrap();
    for name in ["bge-small-en-v1.5", "all-minilm-l6-v2"] {
        let dir = root.path().join(name);
        write_model(&dir);

        let direct = BertModel::from_directory(&dir).unwrap();
        assert_ne!(ids(&direct), vocab_ids(), "{name}: fixture must conflict");

        let managed = BertModel::from_managed_directory(&dir).unwrap();
        assert_eq!(ids(&managed), vocab_ids(), "{name}");
    }
}

#[test]
fn managed_vocab_directory_report_names_only_the_pinned_vocab() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("bge-small-en-v1.5");
    write_model(&dir);

    let direct =
        BertModel::from_directory_with_digest_mode(&dir, WeightDigestMode::Header, &mut |_| {})
            .unwrap();
    assert_eq!(
        direct.load_report().unwrap().tokenizer_format(),
        "wordpiece-json"
    );

    let managed = BertModel::from_managed_directory_with_digest_mode(
        &dir,
        WeightDigestMode::Header,
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(ids(&managed), vocab_ids());
    let report = managed.load_report().unwrap();
    assert_eq!(report.tokenizer_format(), "wordpiece-vocab-txt");
    let names: Vec<&str> = report
        .tokenizer()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(
        names.iter().any(|name| name.ends_with("/vocab.txt")),
        "{names:?}"
    );
    assert!(
        !names.iter().any(|name| name.ends_with("/tokenizer.json")),
        "{names:?}"
    );
}

#[test]
fn other_directories_keep_tokenizer_json_priority() {
    let root = tempfile::tempdir().unwrap();
    for name in ["local-bert", "multilingual-e5-small"] {
        let dir = root.path().join(name);
        write_model(&dir);
        let direct = BertModel::from_directory(&dir).unwrap();
        let managed = BertModel::from_managed_directory(&dir).unwrap();
        assert_eq!(ids(&managed), ids(&direct), "{name}");
        assert_ne!(ids(&managed), vocab_ids(), "{name}");
    }
}
