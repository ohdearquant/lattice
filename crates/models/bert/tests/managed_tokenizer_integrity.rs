#![cfg(unix)]

use lattice_model_bert::BertModel;
use lattice_tokenizer::wordpiece::WordPieceTokenizer;
use std::path::PathBuf;

#[test]
#[ignore = "requires LATTICE_2602_BGE_DIR with pinned bge-small model.safetensors and vocab.txt"]
fn managed_bert_uses_verified_vocab_when_tokenizer_json_conflicts() {
    const CHILD: &str = "LATTICE_2602_CHILD";
    const MODEL: &str = "bge-small-en-v1.5";
    if std::env::var_os(CHILD).is_some() {
        let cache = PathBuf::from(std::env::var_os("LATTICE_MODEL_CACHE").expect("child cache"));
        let model_dir = cache.join(MODEL);
        let vocab = WordPieceTokenizer::from_file(&model_dir.join("vocab.txt"))
            .expect("load pinned vocabulary")
            .with_max_seq_len(16);
        let expected = vocab.tokenize("hello").input_ids;

        let direct = BertModel::from_directory(&model_dir).expect("load direct model");
        let direct_ids = direct.tokenizer().tokenize("hello").input_ids;
        assert_ne!(direct_ids, expected, "fixture must select conflicting JSON");
        drop(direct);

        let managed = BertModel::from_pretrained(MODEL).expect("load verified managed model");
        assert_eq!(managed.tokenizer().max_seq_len(), 16);
        assert_eq!(managed.tokenizer().tokenize("hello").input_ids, expected);
        drop(managed);
        let reported = BertModel::from_pretrained_with_digest_mode(
            MODEL,
            lattice_weights::consumed::WeightDigestMode::Header,
        )
        .expect("load reported managed model");
        assert_eq!(reported.tokenizer().tokenize("hello").input_ids, expected);
        let report = reported
            .load_report()
            .expect("explicit config yields report");
        assert_eq!(report.tokenizer_format(), "wordpiece-vocab-txt");
        assert!(
            report
                .tokenizer()
                .iter()
                .any(|(name, _)| name.ends_with("/vocab.txt"))
        );
        assert!(
            !report
                .tokenizer()
                .iter()
                .any(|(name, _)| name.ends_with("/tokenizer.json"))
        );
        return;
    }

    let source =
        PathBuf::from(std::env::var_os("LATTICE_2602_BGE_DIR").expect("set LATTICE_2602_BGE_DIR"));
    let temp = tempfile::tempdir().expect("temporary cache");
    let model_dir = temp.path().join(MODEL);
    std::fs::create_dir(&model_dir).expect("create model directory");
    for name in ["model.safetensors", "vocab.txt"] {
        let path = std::fs::canonicalize(source.join(name)).expect("pinned source file");
        std::os::unix::fs::symlink(path, model_dir.join(name)).expect("link pinned source file");
    }
    std::fs::write(
        model_dir.join("config.json"),
        r#"{"vocab_size":30522,"hidden_size":384,"num_hidden_layers":12,"num_attention_heads":12,"intermediate_size":1536,"max_position_embeddings":512,"type_vocab_size":2,"layer_norm_eps":1e-12}"#,
    )
    .expect("write bge-small configuration");
    std::fs::write(
        model_dir.join("tokenizer_config.json"),
        r#"{"model_max_length":16}"#,
    )
    .expect("write tokenizer length configuration");
    std::fs::write(
        model_dir.join("tokenizer.json"),
        r#"{"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"hello":5}}}"#,
    )
    .expect("write conflicting tokenizer");

    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "managed_bert_uses_verified_vocab_when_tokenizer_json_conflicts",
            "--ignored",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("LATTICE_MODEL_CACHE", temp.path())
        .env("LATTICE_OFFLINE", "1")
        .status()
        .expect("run isolated child");
    assert!(status.success(), "managed tokenizer child failed: {status}");
}
