//! Dormant prepared-BERT tokenizer candidate-presence selection contract.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RawBertTokenizerCandidatePresence {
    tokenizer_json: bool,
    vocab_json: bool,
    merges_txt: bool,
    vocab_txt: bool,
    tokenizer_model: bool,
    sentencepiece_bpe_model: bool,
    spiece_model: bool,
}

impl RawBertTokenizerCandidatePresence {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        tokenizer_json: bool,
        vocab_json: bool,
        merges_txt: bool,
        vocab_txt: bool,
        tokenizer_model: bool,
        sentencepiece_bpe_model: bool,
        spiece_model: bool,
    ) -> Self {
        Self {
            tokenizer_json,
            vocab_json,
            merges_txt,
            vocab_txt,
            tokenizer_model,
            sentencepiece_bpe_model,
            spiece_model,
        }
    }

    /// Bit layout, MSB to LSB: `tokenizer_json`(6), `vocab_json`(5), `merges_txt`(4),
    /// `vocab_txt`(3), `tokenizer_model`(2), `sentencepiece_bpe_model`(1),
    /// `spiece_model`(0).
    pub(super) fn mask(self) -> u8 {
        u8::from(self.tokenizer_json) << 6
            | u8::from(self.vocab_json) << 5
            | u8::from(self.merges_txt) << 4
            | u8::from(self.vocab_txt) << 3
            | u8::from(self.tokenizer_model) << 2
            | u8::from(self.sentencepiece_bpe_model) << 1
            | u8::from(self.spiece_model)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerLayout {
    TokenizerJson,
    VocabJsonMerges,
    VocabTxtMerges,
    VocabTxt,
}

/// Which SentencePiece candidate file a rejected selection named. ADR-088 D4
/// requires prepared BERT mode to reject a selected SentencePiece layout rather
/// than fall through to it, since SentencePiece/Unigram is outside the supported
/// tokenizer closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SentencePieceCandidate {
    TokenizerDotModel,
    SentencePieceBpe,
    Spiece,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerSelectionError {
    MissingRepresentation,
    VocabJsonWithoutMerges,
    MergesWithoutVocab,
    SentencePieceSelected(SentencePieceCandidate),
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

/// ADR-088 D4 lists the three SentencePiece candidates (`tokenizer.model`,
/// `sentencepiece.bpe.model`, `spiece.model`) without ordering them against each
/// other; it only orders the SentencePiece tier as a whole below WordPiece/BPE.
/// This selector breaks the tie in the order D4 itself lists the three names,
/// both where it introduces the recognized-file set and where it states the
/// tokenizer precedence: `tokenizer.model`, then `sentencepiece.bpe.model`, then
/// `spiece.model`.
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
        return Err(PreparedBertTokenizerSelectionError::SentencePieceSelected(
            SentencePieceCandidate::TokenizerDotModel,
        ));
    } else if presence.sentencepiece_bpe_model {
        return Err(PreparedBertTokenizerSelectionError::SentencePieceSelected(
            SentencePieceCandidate::SentencePieceBpe,
        ));
    } else if presence.spiece_model {
        return Err(PreparedBertTokenizerSelectionError::SentencePieceSelected(
            SentencePieceCandidate::Spiece,
        ));
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
    const SP_TOKENIZER_MODEL: PreparedBertTokenizerSelectionError =
        PreparedBertTokenizerSelectionError::SentencePieceSelected(
            SentencePieceCandidate::TokenizerDotModel,
        );
    const SP_SENTENCEPIECE_BPE: PreparedBertTokenizerSelectionError =
        PreparedBertTokenizerSelectionError::SentencePieceSelected(
            SentencePieceCandidate::SentencePieceBpe,
        );
    const SP_SPIECE: PreparedBertTokenizerSelectionError =
        PreparedBertTokenizerSelectionError::SentencePieceSelected(SentencePieceCandidate::Spiece);

    const J: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::TokenizerJson;
    const VM: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::VocabJsonMerges;
    const TM: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::VocabTxtMerges;
    const T: PreparedBertTokenizerLayout = PreparedBertTokenizerLayout::VocabTxt;

    fn presence(mask: u8) -> RawBertTokenizerCandidatePresence {
        RawBertTokenizerCandidatePresence::new(
            mask & 0b100_0000 != 0,
            mask & 0b010_0000 != 0,
            mask & 0b001_0000 != 0,
            mask & 0b000_1000 != 0,
            mask & 0b000_0100 != 0,
            mask & 0b000_0010 != 0,
            mask & 0b000_0001 != 0,
        )
    }

    /// Every one of the 2^7 = 128 tokenizer-candidate presence states has exactly
    /// one specified result. Bit order matches `presence()`: tokenizer_json,
    /// vocab_json, merges_txt, vocab_txt, tokenizer_model, sentencepiece_bpe_model,
    /// spiece_model (MSB to LSB). Generated from the precedence rules in ADR-088
    /// D4, independent of the implementation under test.
    #[test]
    fn all_128_presence_states_have_one_exact_result() {
        let expected = [
            Err(MISSING),
            Err(SP_SPIECE),
            Err(SP_SENTENCEPIECE_BPE),
            Err(SP_SENTENCEPIECE_BPE),
            Err(SP_TOKENIZER_MODEL),
            Err(SP_TOKENIZER_MODEL),
            Err(SP_TOKENIZER_MODEL),
            Err(SP_TOKENIZER_MODEL),
            Ok(T),
            Ok(T),
            Ok(T),
            Ok(T),
            Ok(T),
            Ok(T),
            Ok(T),
            Ok(T),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Ok(TM),
            Ok(TM),
            Ok(TM),
            Ok(TM),
            Ok(TM),
            Ok(TM),
            Ok(TM),
            Ok(TM),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(VM),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Err(ORPHAN_MERGES),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Err(PARTIAL_VOCAB_JSON),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
            Ok(J),
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
                "JVMTSPS mask {mask:07b}"
            );
        }
    }

    #[test]
    fn global_partial_validation_precedes_even_tokenizer_json() {
        for mask in [0b101_0000, 0b101_0001] {
            assert_eq!(
                select_prepared_bert_tokenizer_layout(presence(mask)),
                Err(ORPHAN_MERGES),
                "orphan merges must fail under tokenizer.json for {mask:07b}"
            );
        }
        for mask in [0b110_0000, 0b110_0001, 0b110_0010, 0b110_0011] {
            assert_eq!(
                select_prepared_bert_tokenizer_layout(presence(mask)),
                Err(PARTIAL_VOCAB_JSON),
                "partial vocab.json must fail under tokenizer.json for {mask:07b}"
            );
        }
    }

    #[test]
    fn selected_layout_preserves_the_complete_raw_presence_fact() {
        for (mask, layout) in [
            (0b111_1111, J),
            (0b011_1111, VM),
            (0b001_1111, TM),
            (0b000_1000, T),
        ] {
            let raw = presence(mask);
            let selected = select_prepared_bert_tokenizer_layout(raw).unwrap();
            assert_eq!(selected.layout(), layout);
            assert_eq!(selected.presence(), raw);
            assert_eq!(selected.presence().mask(), mask);
        }
    }

    /// A selected SentencePiece candidate is rejected rather than accepted, and
    /// the error names which of the three files was selected, per the
    /// tokenizer.model > sentencepiece.bpe.model > spiece.model precedence this
    /// module documents.
    #[test]
    fn sentencepiece_precedence_among_the_three_candidates_when_selected() {
        for (mask, expected) in [
            (0b000_0001, SP_SPIECE),
            (0b000_0010, SP_SENTENCEPIECE_BPE),
            (0b000_0011, SP_SENTENCEPIECE_BPE),
            (0b000_0100, SP_TOKENIZER_MODEL),
            (0b000_0101, SP_TOKENIZER_MODEL),
            (0b000_0110, SP_TOKENIZER_MODEL),
            (0b000_0111, SP_TOKENIZER_MODEL),
        ] {
            assert_eq!(
                select_prepared_bert_tokenizer_layout(presence(mask)),
                Err(expected),
                "SentencePiece-only mask {mask:07b}"
            );
        }
    }

    #[test]
    fn a_higher_tier_candidate_shadows_a_present_sentencepiece_candidate() {
        // tokenizer.json plus every SentencePiece candidate still selects
        // TokenizerJson; the SentencePiece bits are shadowed, not consulted.
        let raw = presence(0b111_0111);
        assert_eq!(
            select_prepared_bert_tokenizer_layout(raw).map(PreparedBertTokenizerSelection::layout),
            Ok(J)
        );
    }
}
