//! Dormant prepared-BERT recognized-file presence contract.

use super::prepared_tokenizer_selection::{
    PreparedBertTokenizerLayout, PreparedBertTokenizerSelection,
    PreparedBertTokenizerSelectionError, RawBertTokenizerCandidatePresence,
    select_prepared_bert_tokenizer_layout,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RawPreparedBertFilePresence {
    model_safetensors: bool,
    config_json: bool,
    tokenizer_config_json: bool,
    tokenizer_json: bool,
    vocab_json: bool,
    merges_txt: bool,
    vocab_txt: bool,
    tokenizer_model: bool,
    sentencepiece_bpe_model: bool,
    spiece_model: bool,
}

impl RawPreparedBertFilePresence {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        model_safetensors: bool,
        config_json: bool,
        tokenizer_config_json: bool,
        tokenizer_json: bool,
        vocab_json: bool,
        merges_txt: bool,
        vocab_txt: bool,
        tokenizer_model: bool,
        sentencepiece_bpe_model: bool,
        spiece_model: bool,
    ) -> Self {
        Self {
            model_safetensors,
            config_json,
            tokenizer_config_json,
            tokenizer_json,
            vocab_json,
            merges_txt,
            vocab_txt,
            tokenizer_model,
            sentencepiece_bpe_model,
            spiece_model,
        }
    }

    /// Bit layout, MSB to LSB: `model_safetensors`(9), `config_json`(8),
    /// `tokenizer_config_json`(7), `tokenizer_json`(6), `vocab_json`(5),
    /// `merges_txt`(4), `vocab_txt`(3), `tokenizer_model`(2),
    /// `sentencepiece_bpe_model`(1), `spiece_model`(0). Ten recognized-file facts
    /// no longer fit a `u8`; this mask widened to `u16` to keep every candidate
    /// individually addressable.
    pub(super) fn mask(self) -> u16 {
        u16::from(self.model_safetensors) << 9
            | u16::from(self.config_json) << 8
            | u16::from(self.tokenizer_config_json) << 7
            | u16::from(self.tokenizer_json) << 6
            | u16::from(self.vocab_json) << 5
            | u16::from(self.merges_txt) << 4
            | u16::from(self.vocab_txt) << 3
            | u16::from(self.tokenizer_model) << 2
            | u16::from(self.sentencepiece_bpe_model) << 1
            | u16::from(self.spiece_model)
    }

    fn tokenizer_candidates(self) -> RawBertTokenizerCandidatePresence {
        RawBertTokenizerCandidatePresence::new(
            self.tokenizer_json,
            self.vocab_json,
            self.merges_txt,
            self.vocab_txt,
            self.tokenizer_model,
            self.sentencepiece_bpe_model,
            self.spiece_model,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertFilePresenceError {
    MissingModelSafetensors,
    MissingConfigJson,
    MissingTokenizerConfigJson,
    Tokenizer(PreparedBertTokenizerSelectionError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertFilePresence {
    presence: RawPreparedBertFilePresence,
    tokenizer: PreparedBertTokenizerSelection,
}

impl PreparedBertFilePresence {
    pub(super) fn presence(self) -> RawPreparedBertFilePresence {
        self.presence
    }

    pub(super) fn tokenizer_layout(self) -> PreparedBertTokenizerLayout {
        self.tokenizer.layout()
    }
}

pub(super) fn validate_prepared_bert_file_presence(
    presence: RawPreparedBertFilePresence,
) -> Result<PreparedBertFilePresence, PreparedBertFilePresenceError> {
    if !presence.model_safetensors {
        return Err(PreparedBertFilePresenceError::MissingModelSafetensors);
    }
    if !presence.config_json {
        return Err(PreparedBertFilePresenceError::MissingConfigJson);
    }
    if !presence.tokenizer_config_json {
        return Err(PreparedBertFilePresenceError::MissingTokenizerConfigJson);
    }

    let tokenizer = select_prepared_bert_tokenizer_layout(presence.tokenizer_candidates())
        .map_err(PreparedBertFilePresenceError::Tokenizer)?;

    Ok(PreparedBertFilePresence {
        presence,
        tokenizer,
    })
}

#[cfg(test)]
mod tests {
    use super::super::prepared_tokenizer_selection::{
        PreparedBertTokenizerLayout, PreparedBertTokenizerSelectionError, SentencePieceCandidate,
    };
    use super::*;

    /// Bit order matches `RawPreparedBertFilePresence::mask()`: model_safetensors,
    /// config_json, tokenizer_config_json, tokenizer_json, vocab_json, merges_txt,
    /// vocab_txt, tokenizer_model, sentencepiece_bpe_model, spiece_model (MSB to
    /// LSB, 10 bits).
    fn presence(mask: u16) -> RawPreparedBertFilePresence {
        RawPreparedBertFilePresence::new(
            mask & 0b10_0000_0000 != 0,
            mask & 0b01_0000_0000 != 0,
            mask & 0b00_1000_0000 != 0,
            mask & 0b00_0100_0000 != 0,
            mask & 0b00_0010_0000 != 0,
            mask & 0b00_0001_0000 != 0,
            mask & 0b00_0000_1000 != 0,
            mask & 0b00_0000_0100 != 0,
            mask & 0b00_0000_0010 != 0,
            mask & 0b00_0000_0001 != 0,
        )
    }

    #[test]
    fn required_files_fail_in_model_config_tokenizer_config_order() {
        for (mask, expected) in [
            (
                0b00_0100_0000,
                PreparedBertFilePresenceError::MissingModelSafetensors,
            ),
            (
                0b10_0100_0000,
                PreparedBertFilePresenceError::MissingConfigJson,
            ),
            (
                0b11_0100_0000,
                PreparedBertFilePresenceError::MissingTokenizerConfigJson,
            ),
        ] {
            assert_eq!(
                validate_prepared_bert_file_presence(presence(mask)),
                Err(expected)
            );
        }
    }

    #[test]
    fn tokenizer_validation_and_precedence_are_delegated_to_the_selector() {
        use PreparedBertTokenizerLayout::{
            TokenizerJson, VocabJsonMerges, VocabTxt, VocabTxtMerges,
        };

        for (mask, expected) in [
            (0b11_1111_1111, Ok(TokenizerJson)),
            (0b11_1011_0000, Ok(VocabJsonMerges)),
            (0b11_1001_1000, Ok(VocabTxtMerges)),
            (0b11_1000_1000, Ok(VocabTxt)),
            (
                0b11_1000_0100,
                Err(PreparedBertFilePresenceError::Tokenizer(
                    PreparedBertTokenizerSelectionError::SentencePieceSelected(
                        SentencePieceCandidate::TokenizerDotModel,
                    ),
                )),
            ),
            (
                0b11_1000_0010,
                Err(PreparedBertFilePresenceError::Tokenizer(
                    PreparedBertTokenizerSelectionError::SentencePieceSelected(
                        SentencePieceCandidate::SentencePieceBpe,
                    ),
                )),
            ),
            (
                0b11_1000_0001,
                Err(PreparedBertFilePresenceError::Tokenizer(
                    PreparedBertTokenizerSelectionError::SentencePieceSelected(
                        SentencePieceCandidate::Spiece,
                    ),
                )),
            ),
            (
                0b11_1001_0000,
                Err(PreparedBertFilePresenceError::Tokenizer(
                    PreparedBertTokenizerSelectionError::MergesWithoutVocab,
                )),
            ),
            (
                0b11_1010_0000,
                Err(PreparedBertFilePresenceError::Tokenizer(
                    PreparedBertTokenizerSelectionError::VocabJsonWithoutMerges,
                )),
            ),
        ] {
            let actual = validate_prepared_bert_file_presence(presence(mask))
                .map(PreparedBertFilePresence::tokenizer_layout);
            assert_eq!(actual, expected, "recognized-file mask {mask:010b}");
        }
    }

    #[test]
    fn validated_output_preserves_all_ten_raw_presence_facts() {
        for mask in [
            0b11_1111_1111u16,
            0b11_1011_0000,
            0b11_1001_1000,
            0b11_1000_1000,
        ] {
            let raw = presence(mask);
            let validated = validate_prepared_bert_file_presence(raw).unwrap();
            assert_eq!(validated.presence(), raw);
            assert_eq!(validated.presence().mask(), mask);
        }
    }

    /// Every present recognized SentencePiece candidate is still carried in the
    /// raw presence facts even though selecting one is rejected: the inventory
    /// evidence (this struct) and the selection outcome are independent.
    #[test]
    fn a_rejected_sentencepiece_selection_does_not_erase_the_original_presence_mask() {
        let mask = 0b11_1000_0111u16; // required files + all three SentencePiece candidates
        let raw = presence(mask);
        assert_eq!(raw.mask(), mask);
        assert_eq!(
            validate_prepared_bert_file_presence(raw),
            Err(PreparedBertFilePresenceError::Tokenizer(
                PreparedBertTokenizerSelectionError::SentencePieceSelected(
                    SentencePieceCandidate::TokenizerDotModel,
                ),
            ))
        );
        // The caller retains the original Copy value regardless of the error.
        assert_eq!(raw.mask(), mask);
    }
}
