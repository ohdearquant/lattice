//! Dormant prepared-BERT raw-BPE lexical vocabulary agreement.
//!
//! This module composes the selected dense `vocab.txt` facts, bounded lexical
//! `merges.txt` census, and the matched config/tensor inventory capability. It
//! proves only vocabulary-cardinality agreement for `VocabTxtMerges`; it does
//! not claim retained merge operands, an effective rank map, emitted-token
//! closure, tokenizer construction, or a live prepared path.

use super::prepared_merges_txt::PreparedBertBpeMergesTxtLexicalFacts;
use super::prepared_tensor_inventory::MatchedBertInventoryFacts;
use super::prepared_vocab_txt::PreparedBertBpeVocabTxtFacts;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertBpeVocabMatchError {
    TokenizerCardinalityPlatformUnrepresentable {
        cardinality: NonZeroU64,
    },
    VocabularyCardinalityMismatch {
        tokenizer_cardinality: NonZeroU64,
        validated_config_and_embedding_rows: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MatchedPreparedBertBpeVocabTxtMergesLexicalFacts {
    bpe_vocab: PreparedBertBpeVocabTxtFacts,
    merges_lexical: PreparedBertBpeMergesTxtLexicalFacts,
    model_inventory: MatchedBertInventoryFacts,
}

impl MatchedPreparedBertBpeVocabTxtMergesLexicalFacts {
    pub(super) fn bpe_vocab(self) -> PreparedBertBpeVocabTxtFacts {
        self.bpe_vocab
    }

    pub(super) fn merges_lexical(self) -> PreparedBertBpeMergesTxtLexicalFacts {
        self.merges_lexical
    }

    pub(super) fn model_inventory(self) -> MatchedBertInventoryFacts {
        self.model_inventory
    }
}

pub(super) fn match_prepared_bert_bpe_vocab_txt_merges_lexical(
    bpe_vocab: PreparedBertBpeVocabTxtFacts,
    merges_lexical: PreparedBertBpeMergesTxtLexicalFacts,
    model_inventory: MatchedBertInventoryFacts,
) -> Result<MatchedPreparedBertBpeVocabTxtMergesLexicalFacts, PreparedBertBpeVocabMatchError> {
    validate_cardinality(bpe_vocab, model_inventory.geometry().vocab_size())?;
    Ok(MatchedPreparedBertBpeVocabTxtMergesLexicalFacts {
        bpe_vocab,
        merges_lexical,
        model_inventory,
    })
}

fn validate_cardinality(
    bpe_vocab: PreparedBertBpeVocabTxtFacts,
    validated_config_and_embedding_rows: usize,
) -> Result<(), PreparedBertBpeVocabMatchError> {
    let tokenizer_cardinality = bpe_vocab.vocab_txt().vocabulary_cardinality();
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
) -> Result<(), PreparedBertBpeVocabMatchError> {
    if tokenizer_cardinality.get() > platform_max {
        return Err(
            PreparedBertBpeVocabMatchError::TokenizerCardinalityPlatformUnrepresentable {
                cardinality: tokenizer_cardinality,
            },
        );
    }
    let tokenizer_cardinality_usize =
        usize::try_from(tokenizer_cardinality.get()).map_err(|_| {
            PreparedBertBpeVocabMatchError::TokenizerCardinalityPlatformUnrepresentable {
                cardinality: tokenizer_cardinality,
            }
        })?;
    if tokenizer_cardinality_usize != validated_config_and_embedding_rows {
        return Err(
            PreparedBertBpeVocabMatchError::VocabularyCardinalityMismatch {
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
    use crate::model::bert::prepared_merges_txt::{
        PreparedBertBpeMergesTxtLimits, census_prepared_bert_bpe_merges_txt,
    };
    use crate::model::bert::prepared_vocab_txt::{
        PreparedBertVocabTxtLimits, parse_prepared_bert_vocab_txt,
        resolve_prepared_bert_bpe_vocab_txt,
    };
    use std::num::NonZeroU64;

    fn nz(value: u64) -> NonZeroU64 {
        match NonZeroU64::new(value) {
            Some(value) => value,
            None => panic!("test value must be nonzero"),
        }
    }

    fn bpe_vocab() -> PreparedBertBpeVocabTxtFacts {
        let limits = match PreparedBertVocabTxtLimits::try_new(nz(64), nz(8), nz(512), nz(4096)) {
            Ok(limits) => limits,
            Err(error) => panic!("valid limits: {error:?}"),
        };
        let vocab = match parse_prepared_bert_vocab_txt(b"a\nb\nc", &limits) {
            Ok(vocab) => vocab,
            Err(error) => panic!("valid vocab: {error:?}"),
        };
        resolve_prepared_bert_bpe_vocab_txt(vocab)
    }

    #[test]
    fn dense_bpe_cardinality_must_equal_validated_config_and_embedding_rows() {
        let bpe_vocab = bpe_vocab();
        assert_eq!(validate_cardinality(bpe_vocab, 3), Ok(()));
        for rows in [2, 4] {
            assert_eq!(
                validate_cardinality(bpe_vocab, rows),
                Err(
                    PreparedBertBpeVocabMatchError::VocabularyCardinalityMismatch {
                        tokenizer_cardinality: nz(3),
                        validated_config_and_embedding_rows: rows,
                    }
                )
            );
        }
    }

    #[test]
    fn platform_failure_precedes_cardinality_mismatch_and_empty_merges_are_valid() {
        let bpe_vocab = bpe_vocab();
        assert_eq!(
            validate_cardinality_with_platform_max(nz(3), 2, 2),
            Err(
                PreparedBertBpeVocabMatchError::TokenizerCardinalityPlatformUnrepresentable {
                    cardinality: nz(3),
                }
            )
        );

        let limits = match PreparedBertBpeMergesTxtLimits::try_new(nz(1), nz(1), nz(8)) {
            Ok(limits) => limits,
            Err(error) => panic!("valid limits: {error:?}"),
        };
        let merges = match census_prepared_bert_bpe_merges_txt(b"", &limits) {
            Ok(merges) => merges,
            Err(error) => panic!("empty merge list is valid: {error:?}"),
        };
        assert_eq!(merges.merge_entry_count(), 0);
        assert_eq!(validate_cardinality(bpe_vocab, 3), Ok(()));
    }

    #[test]
    fn entrypoint_requires_and_retains_all_three_evidence_capabilities() {
        let _entrypoint: fn(
            PreparedBertBpeVocabTxtFacts,
            PreparedBertBpeMergesTxtLexicalFacts,
            MatchedBertInventoryFacts,
        ) -> Result<
            MatchedPreparedBertBpeVocabTxtMergesLexicalFacts,
            PreparedBertBpeVocabMatchError,
        > = match_prepared_bert_bpe_vocab_txt_merges_lexical;
        let _vocab_accessor: fn(
            MatchedPreparedBertBpeVocabTxtMergesLexicalFacts,
        ) -> PreparedBertBpeVocabTxtFacts =
            MatchedPreparedBertBpeVocabTxtMergesLexicalFacts::bpe_vocab;
        let _merges_accessor: fn(
            MatchedPreparedBertBpeVocabTxtMergesLexicalFacts,
        ) -> PreparedBertBpeMergesTxtLexicalFacts =
            MatchedPreparedBertBpeVocabTxtMergesLexicalFacts::merges_lexical;
        let _inventory_accessor: fn(
            MatchedPreparedBertBpeVocabTxtMergesLexicalFacts,
        ) -> MatchedBertInventoryFacts =
            MatchedPreparedBertBpeVocabTxtMergesLexicalFacts::model_inventory;
    }
}
