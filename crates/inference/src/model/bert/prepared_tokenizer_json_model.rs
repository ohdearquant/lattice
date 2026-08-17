//! Dormant prepared-BERT `tokenizer.json` decoded model-type contract.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonModel {
    WordPiece,
    Bpe,
    Unigram,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonModelError {
    Missing,
    Empty,
    Unsupported,
}

pub(super) fn classify_prepared_bert_tokenizer_json_model_type(
    model_type: Option<&str>,
) -> Result<PreparedBertTokenizerJsonModel, PreparedBertTokenizerJsonModelError> {
    match model_type {
        None => Err(PreparedBertTokenizerJsonModelError::Missing),
        Some("") => Err(PreparedBertTokenizerJsonModelError::Empty),
        Some("WordPiece") => Ok(PreparedBertTokenizerJsonModel::WordPiece),
        Some("BPE") => Ok(PreparedBertTokenizerJsonModel::Bpe),
        Some("Unigram" | "SentencePieceUnigram") => Ok(PreparedBertTokenizerJsonModel::Unigram),
        Some(_) => Err(PreparedBertTokenizerJsonModelError::Unsupported),
    }
}

#[cfg(test)]
mod tests {
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
}
