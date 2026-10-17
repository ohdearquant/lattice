use super::*;

#[test]
fn four_supported_spellings_map_to_three_canonical_models() {
    use PreparedBertTokenizerJsonModel::{Bpe, Unigram, WordPiece};

    for (model_type, expected) in [
        ("WordPiece", WordPiece),
        ("BPE", Bpe),
        ("Unigram", Unigram),
        ("SentencePieceUnigram", Unigram),
    ] {
        assert_eq!(
            classify_prepared_bert_tokenizer_json_model_type(Some(model_type)),
            Ok(expected)
        );
    }
}

#[test]
fn missing_empty_unknown_and_wrong_case_fail_without_fallback() {
    use PreparedBertTokenizerJsonModelError::{Empty, Missing, Unsupported};

    assert_eq!(
        classify_prepared_bert_tokenizer_json_model_type(None),
        Err(Missing)
    );
    assert_eq!(
        classify_prepared_bert_tokenizer_json_model_type(Some("")),
        Err(Empty)
    );
    for unsupported in ["BertWordPiece", "wordpiece", "bpe", "SentencePiece"] {
        assert_eq!(
            classify_prepared_bert_tokenizer_json_model_type(Some(unsupported)),
            Err(Unsupported)
        );
    }
}
