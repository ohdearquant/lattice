use super::*;

#[test]
fn test_default_model() {
    let model = EmbeddingModel::default();
    assert_eq!(model, EmbeddingModel::BgeSmallEnV15);
}

#[test]
fn test_model_provenance_new() {
    let provenance = ModelProvenance::new(
        EmbeddingModel::BgeSmallEnV15,
        "BAAI/bge-small-en-v1.5".into(),
    );

    assert_eq!(provenance.model, EmbeddingModel::BgeSmallEnV15);
    assert_eq!(provenance.model_id, "BAAI/bge-small-en-v1.5");
    assert!(!provenance.hash.is_empty());
    assert_eq!(provenance.hash.len(), 64); // sha256 hex is 64 chars
    assert!(!provenance.loaded_at_iso.is_empty());
}

/// Pins the hash to the documented formula and a named digest, so an
/// algorithm swap (e.g. to another 32-byte digest) fails this test even
/// though it would still produce a 64-char hex string.
#[test]
fn test_model_provenance_hash_matches_documented_sha256_formula() {
    let provenance = ModelProvenance::new(
        EmbeddingModel::BgeSmallEnV15,
        "BAAI/bge-small-en-v1.5".into(),
    );

    let expected_input = format!(
        "{}:{}:{:?}",
        provenance.model_id, provenance.loaded_at_iso, provenance.model
    );
    let expected_hash = format!("{:x}", sha2::Sha256::digest(expected_input.as_bytes()));

    assert_eq!(provenance.hash, expected_hash);
}

#[test]
fn test_model_provenance_unique_hash() {
    let p1 = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "model1".into());
    std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamp
    let p2 = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "model1".into());

    // Different timestamps should produce different hashes
    assert_ne!(p1.hash, p2.hash);
}

#[test]
fn test_model_provenance_dimensions() {
    let p1 = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "small".into());
    assert_eq!(p1.dimensions(), 384);

    let p2 = ModelProvenance::new(EmbeddingModel::BgeBaseEnV15, "base".into());
    assert_eq!(p2.dimensions(), 768);

    let p3 = ModelProvenance::new(EmbeddingModel::BgeLargeEnV15, "large".into());
    assert_eq!(p3.dimensions(), 1024);
}

#[test]
fn test_model_provenance_matches_model() {
    let provenance = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "test".into());

    assert!(provenance.matches_model(EmbeddingModel::BgeSmallEnV15));
    assert!(!provenance.matches_model(EmbeddingModel::BgeBaseEnV15));
    assert!(!provenance.matches_model(EmbeddingModel::BgeLargeEnV15));
}

#[test]
fn test_model_provenance_serialization() {
    let provenance = ModelProvenance::new(EmbeddingModel::BgeSmallEnV15, "test-model".into());

    let json = serde_json::to_string(&provenance).unwrap();
    // FP-037: EmbeddingModel has #[serde(rename_all = "snake_case")] so
    // BgeSmallEnV15 serializes as "bge_small_en_v15", not "BgeSmallEnV15".
    assert!(json.contains("bge_small_en_v15"), "json={json}");
    assert!(json.contains("test-model"));
    assert!(json.contains(&provenance.hash));

    let parsed: ModelProvenance = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model, provenance.model);
    assert_eq!(parsed.model_id, provenance.model_id);
    assert_eq!(parsed.hash, provenance.hash);
}

#[test]
fn test_dimensions() {
    assert_eq!(EmbeddingModel::BgeSmallEnV15.dimensions(), 384);
    assert_eq!(EmbeddingModel::BgeBaseEnV15.dimensions(), 768);
    assert_eq!(EmbeddingModel::BgeLargeEnV15.dimensions(), 1024);
    assert_eq!(EmbeddingModel::Qwen3Embedding4B.dimensions(), 2560);
}

#[test]
fn test_model_config_native_dims() {
    assert_eq!(
        ModelConfig::new(EmbeddingModel::Qwen3Embedding4B).dimensions(),
        2560
    );
    assert_eq!(
        ModelConfig::new(EmbeddingModel::Qwen3Embedding0_6B).dimensions(),
        1024
    );
    assert_eq!(
        ModelConfig::new(EmbeddingModel::BgeSmallEnV15).dimensions(),
        384
    );
}

#[test]
fn test_model_config_configured_dim() {
    let cfg = ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(1024)).unwrap();
    assert_eq!(cfg.dimensions(), 1024);

    let cfg = ModelConfig::try_new(EmbeddingModel::Qwen3Embedding0_6B, Some(512)).unwrap();
    assert_eq!(cfg.dimensions(), 512);
}

#[test]
fn test_model_config_validation_below_min() {
    assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(31)).is_err());
    assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(0)).is_err());
}

#[test]
fn test_model_config_validation_above_native() {
    assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(2561)).is_err());
    assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding0_6B, Some(1025)).is_err());
}

#[test]
fn test_model_config_validation_non_mrl_model() {
    assert!(ModelConfig::try_new(EmbeddingModel::BgeSmallEnV15, Some(128)).is_err());
    assert!(ModelConfig::try_new(EmbeddingModel::BgeBaseEnV15, Some(512)).is_err());
}

#[test]
fn test_model_config_none_output_dim_ok_for_any_model() {
    assert!(ModelConfig::try_new(EmbeddingModel::BgeSmallEnV15, None).is_ok());
    assert!(ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, None).is_ok());
}

#[test]
fn test_is_local() {
    assert!(EmbeddingModel::BgeSmallEnV15.is_local());
    assert!(EmbeddingModel::BgeBaseEnV15.is_local());
    assert!(EmbeddingModel::BgeLargeEnV15.is_local());
}

#[test]
fn test_display() {
    assert_eq!(
        EmbeddingModel::BgeSmallEnV15.to_string(),
        "bge-small-en-v1.5"
    );
    assert_eq!(EmbeddingModel::BgeBaseEnV15.to_string(), "bge-base-en-v1.5");
    assert_eq!(
        EmbeddingModel::BgeLargeEnV15.to_string(),
        "bge-large-en-v1.5"
    );
}

#[test]
fn test_serialization_roundtrip() {
    let model = EmbeddingModel::BgeSmallEnV15;
    let json = serde_json::to_string(&model).unwrap();
    let parsed: EmbeddingModel = serde_json::from_str(&json).unwrap();
    assert_eq!(model, parsed);
}

#[test]
fn test_max_input_tokens() {
    assert_eq!(EmbeddingModel::BgeSmallEnV15.max_input_tokens(), 512);
    assert_eq!(EmbeddingModel::BgeBaseEnV15.max_input_tokens(), 512);
    assert_eq!(EmbeddingModel::BgeLargeEnV15.max_input_tokens(), 512);
    assert_eq!(EmbeddingModel::Qwen3Embedding0_6B.max_input_tokens(), 2048);
    assert_eq!(EmbeddingModel::Qwen3Embedding4B.max_input_tokens(), 2048);
}

/// The table follows the limit the tokenizer applies when a Qwen3 Embedding directory
/// is loaded. The directory is synthetic, with context lengths well above the cap, so no
/// checkpoint is needed; it checks the table against the cap, not a real checkpoint's files.
#[cfg(feature = "native")]
#[test]
fn test_qwen3_max_input_tokens_matches_loaded_tokenizer_limit() {
    use lattice_inference::load_tokenizer;

    let dir = tempfile::tempdir().expect("tempdir");
    let write = |name: &str, body: &str| {
        std::fs::write(dir.path().join(name), body).expect("write fixture file");
    };
    write("vocab.json", r#"{"a": 0, "<|endoftext|>": 1}"#);
    write("merges.txt", "#version: 0.2\n");
    write("config.json", r#"{"max_position_embeddings": 32768}"#);
    write("tokenizer_config.json", r#"{"model_max_length": 131072}"#);

    let tokenizer = load_tokenizer(dir.path()).expect("tokenizer loads");
    for model in [
        EmbeddingModel::Qwen3Embedding0_6B,
        EmbeddingModel::Qwen3Embedding4B,
    ] {
        assert_eq!(
            tokenizer.max_seq_len(),
            model.max_input_tokens(),
            "{model:?}: max_input_tokens() must equal the tokenizer's sequence limit"
        );
    }
}

#[test]
fn test_from_str_display_names() {
    assert_eq!(
        "bge-small-en-v1.5".parse::<EmbeddingModel>().unwrap(),
        EmbeddingModel::BgeSmallEnV15
    );
    assert_eq!(
        "bge-base-en-v1.5".parse::<EmbeddingModel>().unwrap(),
        EmbeddingModel::BgeBaseEnV15
    );
    assert_eq!(
        "bge-large-en-v1.5".parse::<EmbeddingModel>().unwrap(),
        EmbeddingModel::BgeLargeEnV15
    );
}

#[test]
fn test_from_str_short_names() {
    assert_eq!(
        "small".parse::<EmbeddingModel>().unwrap(),
        EmbeddingModel::BgeSmallEnV15
    );
    assert_eq!(
        "bge-base".parse::<EmbeddingModel>().unwrap(),
        EmbeddingModel::BgeBaseEnV15
    );
    assert_eq!(
        "LARGE".parse::<EmbeddingModel>().unwrap(), // case insensitive
        EmbeddingModel::BgeLargeEnV15
    );
}

#[test]
fn test_from_str_huggingface_ids() {
    assert_eq!(
        "BAAI/bge-small-en-v1.5".parse::<EmbeddingModel>().unwrap(),
        EmbeddingModel::BgeSmallEnV15
    );
}

#[test]
fn test_from_str_invalid() {
    let result = "unknown-model".parse::<EmbeddingModel>();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unknown embedding model"));
}

// -------------------------------------------------------------------------
// bert_pooling() routing tests (P1-E3) — require `native` feature
// -------------------------------------------------------------------------

/// BGE small/base/large must use CLS pooling per their HF model cards.
#[cfg(feature = "native")]
#[test]
fn test_bge_models_use_cls_pooling() {
    use lattice_inference::BertPooling;

    assert_eq!(
        EmbeddingModel::BgeSmallEnV15.bert_pooling(),
        Some(BertPooling::CLS),
        "BgeSmallEnV15 must use CLS pooling"
    );
    assert_eq!(
        EmbeddingModel::BgeBaseEnV15.bert_pooling(),
        Some(BertPooling::CLS),
        "BgeBaseEnV15 must use CLS pooling"
    );
    assert_eq!(
        EmbeddingModel::BgeLargeEnV15.bert_pooling(),
        Some(BertPooling::CLS),
        "BgeLargeEnV15 must use CLS pooling"
    );
}

/// E5 models must use mean pooling per their HF model cards.
#[cfg(feature = "native")]
#[test]
fn test_e5_models_use_mean_pooling() {
    use lattice_inference::BertPooling;

    assert_eq!(
        EmbeddingModel::MultilingualE5Small.bert_pooling(),
        Some(BertPooling::Mean),
        "MultilingualE5Small must use mean pooling"
    );
    assert_eq!(
        EmbeddingModel::MultilingualE5Base.bert_pooling(),
        Some(BertPooling::Mean),
        "MultilingualE5Base must use mean pooling"
    );
}

/// MiniLM models must use mean pooling per sentence-transformers convention.
#[cfg(feature = "native")]
#[test]
fn test_minilm_models_use_mean_pooling() {
    use lattice_inference::BertPooling;

    assert_eq!(
        EmbeddingModel::AllMiniLmL6V2.bert_pooling(),
        Some(BertPooling::Mean),
        "AllMiniLmL6V2 must use mean pooling"
    );
    assert_eq!(
        EmbeddingModel::ParaphraseMultilingualMiniLmL12V2.bert_pooling(),
        Some(BertPooling::Mean),
        "ParaphraseMultilingualMiniLmL12V2 must use mean pooling"
    );
}

/// Qwen and remote models return None — they have separate pooling paths.
#[cfg(feature = "native")]
#[test]
fn test_non_bert_models_return_none_pooling() {
    assert_eq!(
        EmbeddingModel::Qwen3Embedding0_6B.bert_pooling(),
        None,
        "Qwen model must return None for bert_pooling()"
    );
    assert_eq!(
        EmbeddingModel::Qwen3Embedding4B.bert_pooling(),
        None,
        "Qwen model must return None for bert_pooling()"
    );
    assert_eq!(
        EmbeddingModel::TextEmbedding3Small.bert_pooling(),
        None,
        "Remote model must return None for bert_pooling()"
    );
}

/// BGE and E5 use DIFFERENT pooling strategies — this is the key correctness distinction.
#[cfg(feature = "native")]
#[test]
fn test_bge_and_e5_use_different_pooling() {
    assert_ne!(
        EmbeddingModel::BgeSmallEnV15.bert_pooling(),
        EmbeddingModel::MultilingualE5Small.bert_pooling(),
        "BGE and E5 must use different pooling strategies"
    );
}

/// Guards `lattice_inference::pool::default_bert_pooling_for_model_name`
/// (the served-directory-name lookup `lattice_serve` uses) against
/// drifting from `EmbeddingModel::bert_pooling` (this table). The
/// expectation comes from calling `bert_pooling()` itself, never from
/// retyping its rule, so a wrong mirror cannot agree with a wrong
/// expectation.
///
/// This lives in `lattice-embed`, not `lattice-inference`, so both sides
/// of the comparison are the same compiled unit: `lattice-embed` already
/// depends on `lattice-inference` (this crate's own `[dependencies]`,
/// gated by the `native` feature), so `EmbeddingModel::bert_pooling()`'s
/// return type and `default_bert_pooling_for_model_name`'s return type
/// are literally the same `lattice_inference::BertPooling`, comparable
/// with `==` directly -- no `format!("{:?}")` string workaround needed.
///
/// Enumeration: `EmbeddingModel` has no `EnumIter`/`VariantArray`/
/// `ALL_MODELS` constant (checked via `grep -n
/// "strum\|EnumIter\|VariantArray\|ALL_MODELS" crates/embed/src/model.rs`,
/// no hits), so `variants` below is still a hand-transcribed list of all
/// 11 current variants. But `#[non_exhaustive]` only forces a wildcard
/// arm in matches written *outside* the declaring crate -- this test
/// lives inside `lattice-embed` itself, where `EmbeddingModel` is
/// declared, so a wildcard-free match over it type-checks here. The
/// `assert_variants_exhaustively_covered` match below has no wildcard
/// arm; it is checked against the enum's full current variant set
/// regardless of which values are actually passed to it, so a variant
/// added to `EmbeddingModel` without a matching arm here is a compile
/// error ("non-exhaustive patterns"), not a silent gap in `variants`.
#[cfg(feature = "native")]
#[test]
fn default_bert_pooling_for_model_name_matches_embedding_model_bert_pooling_table() {
    use lattice_inference::pool::default_bert_pooling_for_model_name;

    fn assert_variants_exhaustively_covered(m: &EmbeddingModel) {
        match m {
            EmbeddingModel::BgeSmallEnV15
            | EmbeddingModel::BgeBaseEnV15
            | EmbeddingModel::BgeLargeEnV15
            | EmbeddingModel::MultilingualE5Small
            | EmbeddingModel::MultilingualE5Base
            | EmbeddingModel::AllMiniLmL6V2
            | EmbeddingModel::ParaphraseMultilingualMiniLmL12V2
            | EmbeddingModel::Qwen3Embedding0_6B
            | EmbeddingModel::Qwen3Embedding4B
            | EmbeddingModel::EmbeddingGemma2
            | EmbeddingModel::TextEmbedding3Small => {}
        }
    }

    let variants = [
        EmbeddingModel::BgeSmallEnV15,
        EmbeddingModel::BgeBaseEnV15,
        EmbeddingModel::BgeLargeEnV15,
        EmbeddingModel::MultilingualE5Small,
        EmbeddingModel::MultilingualE5Base,
        EmbeddingModel::AllMiniLmL6V2,
        EmbeddingModel::ParaphraseMultilingualMiniLmL12V2,
        EmbeddingModel::Qwen3Embedding0_6B,
        EmbeddingModel::Qwen3Embedding4B,
        EmbeddingModel::EmbeddingGemma2,
        EmbeddingModel::TextEmbedding3Small,
    ];

    let mut bert_family_checked = 0usize;
    for variant in variants {
        assert_variants_exhaustively_covered(&variant);

        let Some(expected) = variant.bert_pooling() else {
            continue;
        };
        bert_family_checked += 1;
        // Same shape of value production reaches
        // `default_bert_pooling_for_model_name` in production:
        // `lattice_serve` passes it the served directory's last path
        // component (`resolve_model_dir(...).file_name()`), not the
        // model's full HF id. Simulate that here by taking the last
        // path component of the model's own `model_id()`.
        let dir_name = std::path::Path::new(variant.model_id())
            .file_name()
            .and_then(|n| n.to_str())
            .expect("model_id() yields a valid path component");
        let actual = default_bert_pooling_for_model_name(dir_name);
        assert_eq!(
            actual, expected,
            "pooling mismatch for {variant:?} (derived dir name {dir_name:?})"
        );
    }
    // BGE small/base/large + E5 small/base + MiniLM + paraphrase-MiniLM:
    // the 7 variants whose `bert_pooling()` returns `Some(_)` as of this
    // writing. Qwen3 x2, EmbeddingGemma2 and TextEmbedding3Small return `None` and are
    // skipped above. This catches a variant's `bert_pooling()` gate
    // moving, but -- per the enumeration note above -- a wholly new
    // variant absent from `variants` only surfaces as the compile error
    // from `assert_variants_exhaustively_covered`, not from this count.
    assert_eq!(
        bert_family_checked, 7,
        "EmbeddingModel's BERT-family (Some(_) from bert_pooling()) variant count changed"
    );
}

#[test]
fn test_embeddinggemma2_registry_entry() {
    let m = EmbeddingModel::EmbeddingGemma2;
    assert_eq!(m.native_dimensions(), 768);
    assert!(m.is_local() && !m.is_remote());
    assert_eq!(m.max_input_tokens(), 8192);
    assert_eq!(m.query_instruction(), Some("task: search result | query: "));
    assert_eq!(m.document_instruction(), Some("title: none | text: "));
    assert_eq!(m.model_id(), "google/embeddinggemma-2");
    assert!(m.supports_output_dim());
    for name in [
        "embeddinggemma-2",
        "EmbeddingGemma2",
        "embedding_gemma2",
        "google/embeddinggemma-2",
    ] {
        assert_eq!(name.parse::<EmbeddingModel>(), Ok(m), "{name}");
    }
    assert_eq!(m.to_string().parse::<EmbeddingModel>(), Ok(m));
    let json = serde_json::to_string(&m).unwrap();
    assert_eq!(json, "\"embedding_gemma2\"");
    assert_eq!(serde_json::from_str::<EmbeddingModel>(&json).unwrap(), m);
    assert_eq!(
        serde_json::from_str::<EmbeddingModel>("\"EmbeddingGemma2\"").unwrap(),
        m
    );

    for width in [768, 512, 256, 128] {
        let config = ModelConfig::try_new(m, Some(width)).unwrap();
        assert_eq!(config.dimensions(), width);
    }
    assert!(ModelConfig::try_new(m, Some(769)).is_err());
    assert!(ModelConfig::try_new(m, Some(MIN_MRL_OUTPUT_DIM - 1)).is_err());
}
