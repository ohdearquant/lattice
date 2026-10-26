//! Dormant adapter from parsed prepared-BERT config facts to sequence-cap resolution.

use super::prepared_config::{ParsedBertConfigFacts, PreparedBertConfigSequenceCandidateError};
use super::prepared_sequence_cap::{
    PreparedBertSequenceCap, PreparedBertSequenceCapError, PreparedBertSequenceCapKey,
    RawPreparedBertSequenceCapCandidate, RawPreparedBertSequenceCapCandidates,
    resolve_prepared_bert_sequence_cap,
};
use super::prepared_tokenizer_config::{
    ParsedBertTokenizerConfigFacts, PreparedBertTokenizerConfigCandidateKey,
    PreparedBertTokenizerMaxLengthCandidate,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertParsedSequenceCapError {
    ConfigJsonCandidate(PreparedBertConfigSequenceCandidateError),
    Resolver(PreparedBertSequenceCapError),
}

pub(super) fn resolve_prepared_bert_sequence_cap_from_facts(
    tokenizer_config: ParsedBertTokenizerConfigFacts,
    config: ParsedBertConfigFacts,
) -> Result<Option<PreparedBertSequenceCap>, PreparedBertParsedSequenceCapError> {
    let tokenizer_config_json = tokenizer_config.candidate().map(map_tokenizer_candidate);
    let config_json = if tokenizer_config_json.is_some() {
        None
    } else {
        Some(
            config
                .sequence_cap_candidate()
                .map_err(PreparedBertParsedSequenceCapError::ConfigJsonCandidate)?,
        )
    };
    resolve_prepared_bert_sequence_cap(RawPreparedBertSequenceCapCandidates::new(
        tokenizer_config_json,
        config_json,
    ))
    .map_err(PreparedBertParsedSequenceCapError::Resolver)
}

fn map_tokenizer_candidate(
    candidate: PreparedBertTokenizerMaxLengthCandidate,
) -> RawPreparedBertSequenceCapCandidate {
    let key = match candidate.key() {
        PreparedBertTokenizerConfigCandidateKey::ModelMaxLength => {
            PreparedBertSequenceCapKey::ModelMaxLength
        }
        PreparedBertTokenizerConfigCandidateKey::MaxPositionEmbeddings => {
            PreparedBertSequenceCapKey::MaxPositionEmbeddings
        }
        PreparedBertTokenizerConfigCandidateKey::NPositions => {
            PreparedBertSequenceCapKey::NPositions
        }
        PreparedBertTokenizerConfigCandidateKey::MaxSeqLen => PreparedBertSequenceCapKey::MaxSeqLen,
        PreparedBertTokenizerConfigCandidateKey::TruncationMaxLength => {
            PreparedBertSequenceCapKey::TruncationMaxLength
        }
    };
    RawPreparedBertSequenceCapCandidate::new(key, candidate.raw_value())
}

#[cfg(test)]
mod tests;
