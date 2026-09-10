//! Dormant prepared-BERT WordPiece vocabulary/cardinality agreement.
//!
//! The tensor matcher proves that the prepared geometry's vocabulary size is
//! both the strict `config.json` value and the first dimension of the required
//! word-embedding tensor. This module adds the selected WordPiece `vocab.txt`
//! dense-cardinality proof without accepting a caller-supplied row count.

use super::prepared_tensor_inventory::MatchedBertInventoryFacts;
use super::prepared_vocab_txt::PreparedBertWordPieceVocabTxtFacts;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertWordPieceVocabMatchError {
    TokenizerCardinalityPlatformUnrepresentable {
        cardinality: NonZeroU64,
    },
    VocabularyCardinalityMismatch {
        tokenizer_cardinality: NonZeroU64,
        validated_config_and_embedding_rows: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MatchedPreparedBertWordPieceVocabFacts {
    wordpiece_vocab: PreparedBertWordPieceVocabTxtFacts,
    model_inventory: MatchedBertInventoryFacts,
}

impl MatchedPreparedBertWordPieceVocabFacts {
    pub(super) fn wordpiece_vocab(self) -> PreparedBertWordPieceVocabTxtFacts {
        self.wordpiece_vocab
    }

    pub(super) fn model_inventory(self) -> MatchedBertInventoryFacts {
        self.model_inventory
    }
}

pub(super) fn match_prepared_bert_wordpiece_vocab(
    wordpiece_vocab: PreparedBertWordPieceVocabTxtFacts,
    model_inventory: MatchedBertInventoryFacts,
) -> Result<MatchedPreparedBertWordPieceVocabFacts, PreparedBertWordPieceVocabMatchError> {
    validate_cardinality(wordpiece_vocab, model_inventory.geometry().vocab_size())?;
    Ok(MatchedPreparedBertWordPieceVocabFacts {
        wordpiece_vocab,
        model_inventory,
    })
}

fn validate_cardinality(
    wordpiece_vocab: PreparedBertWordPieceVocabTxtFacts,
    validated_config_and_embedding_rows: usize,
) -> Result<(), PreparedBertWordPieceVocabMatchError> {
    let tokenizer_cardinality = wordpiece_vocab.vocab_txt().vocabulary_cardinality();
    validate_cardinality_with_platform_max(
        tokenizer_cardinality,
        validated_config_and_embedding_rows,
        usize::MAX as u64,
    )
}

fn validate_cardinality_with_platform_max(
    tokenizer_cardinality: NonZeroU64,
    validated_config_and_embedding_rows: usize,
    platform_max: u64,
) -> Result<(), PreparedBertWordPieceVocabMatchError> {
    if tokenizer_cardinality.get() > platform_max {
        return Err(
            PreparedBertWordPieceVocabMatchError::TokenizerCardinalityPlatformUnrepresentable {
                cardinality: tokenizer_cardinality,
            },
        );
    }
    let tokenizer_cardinality_usize =
        usize::try_from(tokenizer_cardinality.get()).map_err(|_| {
            PreparedBertWordPieceVocabMatchError::TokenizerCardinalityPlatformUnrepresentable {
                cardinality: tokenizer_cardinality,
            }
        })?;
    if tokenizer_cardinality_usize != validated_config_and_embedding_rows {
        return Err(
            PreparedBertWordPieceVocabMatchError::VocabularyCardinalityMismatch {
                tokenizer_cardinality,
                validated_config_and_embedding_rows,
            },
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::bert::prepared_vocab_txt::{
        PreparedBertVocabTxtLimits, parse_prepared_bert_vocab_txt,
        validate_prepared_bert_wordpiece_vocab_txt,
    };
    use std::num::NonZeroU64;

    fn nz(value: u64) -> NonZeroU64 {
        match NonZeroU64::new(value) {
            Some(value) => value,
            None => panic!("test limit must be nonzero"),
        }
    }

    fn wordpiece() -> PreparedBertWordPieceVocabTxtFacts {
        let limits =
            match PreparedBertVocabTxtLimits::try_new(nz(256), nz(16), nz(1024), nz(16_384)) {
                Ok(limits) => limits,
                Err(error) => panic!("valid limits: {error:?}"),
            };
        let generic = match parse_prepared_bert_vocab_txt(
            b"[CLS]\n[SEP]\n[PAD]\n[UNK]\n[MASK]\nordinary",
            &limits,
        ) {
            Ok(facts) => facts,
            Err(error) => panic!("valid vocab: {error:?}"),
        };
        match validate_prepared_bert_wordpiece_vocab_txt(generic) {
            Ok(facts) => facts,
            Err(error) => panic!("valid WordPiece vocab: {error:?}"),
        }
    }

    #[test]
    fn dense_cardinality_must_equal_validated_config_and_embedding_rows() {
        let wordpiece = wordpiece();
        assert_eq!(validate_cardinality(wordpiece, 6), Ok(()));
        for rows in [5, 7] {
            assert_eq!(
                validate_cardinality(wordpiece, rows),
                Err(
                    PreparedBertWordPieceVocabMatchError::VocabularyCardinalityMismatch {
                        tokenizer_cardinality: nz(6),
                        validated_config_and_embedding_rows: rows,
                    }
                )
            );
        }
    }

    #[test]
    fn matching_does_not_rewrite_wordpiece_ids_or_generic_facts() {
        let wordpiece = wordpiece();
        assert_eq!(wordpiece.cls_id(), 0);
        assert_eq!(wordpiece.sep_id(), 1);
        assert_eq!(wordpiece.pad_id(), 2);
        assert_eq!(wordpiece.unk_id(), 3);
        assert_eq!(wordpiece.mask_id(), 4);
        assert_eq!(wordpiece.vocab_txt().vocabulary_cardinality().get(), 6);
        assert_eq!(validate_cardinality(wordpiece, 6), Ok(()));
        assert_eq!(
            validate_cardinality_with_platform_max(nz(6), 6, 5),
            Err(
                PreparedBertWordPieceVocabMatchError::TokenizerCardinalityPlatformUnrepresentable {
                    cardinality: nz(6),
                }
            )
        );
    }

    #[test]
    fn entrypoint_requires_and_retains_the_matched_inventory_capability() {
        let _entrypoint: fn(
            PreparedBertWordPieceVocabTxtFacts,
            MatchedBertInventoryFacts,
        ) -> Result<
            MatchedPreparedBertWordPieceVocabFacts,
            PreparedBertWordPieceVocabMatchError,
        > = match_prepared_bert_wordpiece_vocab;
        let _wordpiece_accessor: fn(
            MatchedPreparedBertWordPieceVocabFacts,
        ) -> PreparedBertWordPieceVocabTxtFacts =
            MatchedPreparedBertWordPieceVocabFacts::wordpiece_vocab;
        let _inventory_accessor: fn(
            MatchedPreparedBertWordPieceVocabFacts,
        ) -> MatchedBertInventoryFacts = MatchedPreparedBertWordPieceVocabFacts::model_inventory;
    }
}
