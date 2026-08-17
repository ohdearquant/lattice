//! Dormant prepared-BERT tokenizer candidate-presence selection contract.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RawBertTokenizerCandidatePresence {
    tokenizer_json: bool,
    vocab_json: bool,
    merges_txt: bool,
    vocab_txt: bool,
    tokenizer_model: bool,
}

impl RawBertTokenizerCandidatePresence {
    pub(super) fn new(
        tokenizer_json: bool,
        vocab_json: bool,
        merges_txt: bool,
        vocab_txt: bool,
        tokenizer_model: bool,
    ) -> Self {
        Self {
            tokenizer_json,
            vocab_json,
            merges_txt,
            vocab_txt,
            tokenizer_model,
        }
    }

    pub(super) fn mask(self) -> u8 {
        u8::from(self.tokenizer_json) << 4
            | u8::from(self.vocab_json) << 3
            | u8::from(self.merges_txt) << 2
            | u8::from(self.vocab_txt) << 1
            | u8::from(self.tokenizer_model)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerLayout {
    TokenizerJson,
    VocabJsonMerges,
    VocabTxtMerges,
    VocabTxt,
    TokenizerModel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerSelectionError {
    MissingRepresentation,
    VocabJsonWithoutMerges,
    MergesWithoutVocab,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertTokenizerSelection {
    layout: PreparedBertTokenizerLayout,
    presence: RawBertTokenizerCandidatePresence,
}

impl PreparedBertTokenizerSelection {
    pub(super) fn layout(self) -> PreparedBertTokenizerLayout {
        self.layout
    }

    pub(super) fn presence(self) -> RawBertTokenizerCandidatePresence {
        self.presence
    }
}

pub(super) fn select_prepared_bert_tokenizer_layout(
    presence: RawBertTokenizerCandidatePresence,
) -> Result<PreparedBertTokenizerSelection, PreparedBertTokenizerSelectionError> {
    if presence.vocab_json && !presence.merges_txt {
        return Err(PreparedBertTokenizerSelectionError::VocabJsonWithoutMerges);
    }
    if presence.merges_txt && !presence.vocab_json && !presence.vocab_txt {
        return Err(PreparedBertTokenizerSelectionError::MergesWithoutVocab);
    }

    let layout = if presence.tokenizer_json {
        PreparedBertTokenizerLayout::TokenizerJson
    } else if presence.vocab_json && presence.merges_txt {
        PreparedBertTokenizerLayout::VocabJsonMerges
    } else if presence.vocab_txt && presence.merges_txt {
        PreparedBertTokenizerLayout::VocabTxtMerges
    } else if presence.vocab_txt {
        PreparedBertTokenizerLayout::VocabTxt
    } else if presence.tokenizer_model {
        PreparedBertTokenizerLayout::TokenizerModel
    } else {
        return Err(PreparedBertTokenizerSelectionError::MissingRepresentation);
    };

    Ok(PreparedBertTokenizerSelection { layout, presence })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MISSING: PreparedBertTokenizerSelectionError =
        PreparedBertTokenizerSelectionError::MissingRepresentation;
    const PARTIAL_VOCAB_JSON: PreparedBertTokenizerSelectionError =
        PreparedBertTokenizerSelectionError::VocabJsonWithoutMerges;
    const ORPHAN_MERGES: PreparedBertTokenizerSelectionError =
        PreparedBertTokenizerSelectionError::MergesWithoutVocab;

    const J: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::TokenizerJson;
    const VM: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::VocabJsonMerges;
    const TM: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::VocabTxtMerges;
    const T: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::VocabTxt;
    const S: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::TokenizerModel;

    fn presence(mask: u8) -> RawBertTokenizerCandidatePresence {
        RawBertTokenizerCandidatePresence::new(
            mask & 0b1_0000 != 0,
            mask & 0b0_1000 != 0,
            mask & 0b0_0100 != 0,
            mask & 0b0_0010 != 0,
            mask & 0b0_0001 != 0,
        )
    }

    #[test]
    fn all_32_presence_states_have_one_exact_result() {
        let expected = [
            Err(MISSING),
            Ok(S),
            Ok(T),
            Ok(T),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Ok(TM),
            Ok(TM),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Ok(J),
            Ok(J),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
        ];

        for (mask, expected_layout) in expected.into_iter().enumerate() {
            let raw = presence(u8::try_from(mask).unwrap());
            assert_eq!(
                select_prepared_bert_tokenizer_layout(raw)
                    .map(PreparedBertTokenizerSelection::layout),
                expected_layout,
                "JVMTS mask {mask:05b}"
            );
        }
    }

    #[test]
    fn global_partial_validation_precedes_even_tokenizer_json() {
        for mask in [0b1_0100, 0b1_0101] {
            assert_eq!(
                select_prepared_bert_tokenizer_layout(presence(mask)),
                Err(ORPHAN_MERGES),
                "orphan merges must fail under tokenizer.json for {mask:05b}"
            );
        }
        for mask in [0b1_1000, 0b1_1001, 0b1_1010, 0b1_1011] {
            assert_eq!(
                select_prepared_bert_tokenizer_layout(presence(mask)),
                Err(PARTIAL_VOCAB_JSON),
                "partial vocab.json must fail under tokenizer.json for {mask:05b}"
            );
        }
    }

    #[test]
    fn selected_layout_preserves_the_complete_raw_presence_fact() {
        for (mask, layout) in [
            (0b1_1111, J),
            (0b0_1111, VM),
            (0b0_0111, TM),
            (0b0_0011, T),
            (0b0_0001, S),
        ] {
            let raw = presence(mask);
            let selected = select_prepared_bert_tokenizer_layout(raw).unwrap();
            assert_eq!(selected.layout(), layout);
            assert_eq!(selected.presence(), raw);
            assert_eq!(selected.presence().mask(), mask);
        }
    }
}
