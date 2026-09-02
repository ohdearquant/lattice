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
        }
    }

    pub(super) fn mask(self) -> u8 {
        u8::from(self.model_safetensors) << 7
            | u8::from(self.config_json) << 6
            | u8::from(self.tokenizer_config_json) << 5
            | u8::from(self.tokenizer_json) << 4
            | u8::from(self.vocab_json) << 3
            | u8::from(self.merges_txt) << 2
            | u8::from(self.vocab_txt) << 1
            | u8::from(self.tokenizer_model)
    }

    fn tokenizer_candidates(self) -> RawBertTokenizerCandidatePresence {
        RawBertTokenizerCandidatePresence::new(
            self.tokenizer_json,
            self.vocab_json,
            self.merges_txt,
            self.vocab_txt,
            self.tokenizer_model,
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
        PreparedBertTokenizerLayout, PreparedBertTokenizerSelectionError,
    };
    use super::*;

    fn presence(mask: u8) -> RawPreparedBertFilePresence {
        RawPreparedBertFilePresence::new(
            mask & 0b1000_0000 != 0,
            mask & 0b0100_0000 != 0,
            mask & 0b0010_0000 != 0,
            mask & 0b0001_0000 != 0,
            mask & 0b0000_1000 != 0,
            mask & 0b0000_0100 != 0,
            mask & 0b0000_0010 != 0,
            mask & 0b0000_0001 != 0,
        )
    }

    #[test]
    fn required_files_fail_in_model_config_tokenizer_config_order() {
        for (mask, expected) in [
            (
                0b0001_0000,
                PreparedBertFilePresenceError::MissingModelSafetensors,
            ),
            (
                0b1001_0000,
                PreparedBertFilePresenceError::MissingConfigJson,
            ),
            (
                0b1101_0000,
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
            TokenizerJson, TokenizerModel, VocabJsonMerges, VocabTxt, VocabTxtMerges,
        };

        for (mask, expected) in [
            (0b1111_1111, Ok(TokenizerJson)),
            (0b1110_1101, Ok(VocabJsonMerges)),
            (0b1110_0111, Ok(VocabTxtMerges)),
            (0b1110_0011, Ok(VocabTxt)),
            (0b1110_0001, Ok(TokenizerModel)),
            (
                0b1111_0100,
                Err(PreparedBertFilePresenceError::Tokenizer(
                    PreparedBertTokenizerSelectionError::MergesWithoutVocab,
                )),
            ),
            (
                0b1111_1000,
                Err(PreparedBertFilePresenceError::Tokenizer(
                    PreparedBertTokenizerSelectionError::VocabJsonWithoutMerges,
                )),
            ),
        ] {
            let actual = validate_prepared_bert_file_presence(presence(mask))
                .map(PreparedBertFilePresence::tokenizer_layout);
            assert_eq!(actual, expected, "recognized-file mask {mask:08b}");
        }
    }

    #[test]
    fn validated_output_preserves_all_eight_raw_presence_facts() {
        for mask in [
            0b1111_1111,
            0b1110_1101,
            0b1110_0111,
            0b1110_0011,
            0b1110_0001,
        ] {
            let raw = presence(mask);
            let validated = validate_prepared_bert_file_presence(raw).unwrap();
            assert_eq!(validated.presence(), raw);
            assert_eq!(validated.presence().mask(), mask);
        }
    }
}
