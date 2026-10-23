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

/// ADR-088 D4's 2026-09-24 amendment fixes the intra-tier order of the three
/// SentencePiece candidates: `tokenizer.model`, then `sentencepiece.bpe.model`,
/// then `spiece.model`. When more than one is present, the typed error names the
/// first present candidate in that order; this selector implements that rule,
/// not a choice made independently of the ADR.
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
mod tests;
