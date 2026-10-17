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
mod tests;
