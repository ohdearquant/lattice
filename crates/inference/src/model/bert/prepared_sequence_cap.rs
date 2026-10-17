//! Dormant cross-file prepared-BERT sequence-cap resolution contract.

use std::num::NonZeroU64;

const PREPARED_BERT_SEQUENCE_CAP: u64 = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertSequenceCapSource {
    TokenizerConfigJson,
    ConfigJson,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertSequenceCapKey {
    ModelMaxLength,
    MaxPositionEmbeddings,
    NPositions,
    MaxSeqLen,
    TruncationMaxLength,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RawPreparedBertSequenceCapCandidate {
    key: PreparedBertSequenceCapKey,
    raw_value: u64,
}

impl RawPreparedBertSequenceCapCandidate {
    pub(super) fn new(key: PreparedBertSequenceCapKey, raw_value: u64) -> Self {
        Self { key, raw_value }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RawPreparedBertSequenceCapCandidates {
    tokenizer_config_json: Option<RawPreparedBertSequenceCapCandidate>,
    config_json: Option<RawPreparedBertSequenceCapCandidate>,
}

impl RawPreparedBertSequenceCapCandidates {
    pub(super) fn new(
        tokenizer_config_json: Option<RawPreparedBertSequenceCapCandidate>,
        config_json: Option<RawPreparedBertSequenceCapCandidate>,
    ) -> Self {
        Self {
            tokenizer_config_json,
            config_json,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertSequenceCapError {
    Zero {
        source: PreparedBertSequenceCapSource,
        key: PreparedBertSequenceCapKey,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertSequenceCap {
    source: PreparedBertSequenceCapSource,
    key: PreparedBertSequenceCapKey,
    raw_value: NonZeroU64,
    capped_value: NonZeroU64,
}

impl PreparedBertSequenceCap {
    pub(super) fn source(self) -> PreparedBertSequenceCapSource {
        self.source
    }

    pub(super) fn key(self) -> PreparedBertSequenceCapKey {
        self.key
    }

    pub(super) fn raw_value(self) -> NonZeroU64 {
        self.raw_value
    }

    pub(super) fn capped_value(self) -> NonZeroU64 {
        self.capped_value
    }
}

pub(super) fn resolve_prepared_bert_sequence_cap(
    candidates: RawPreparedBertSequenceCapCandidates,
) -> Result<Option<PreparedBertSequenceCap>, PreparedBertSequenceCapError> {
    let selected = candidates
        .tokenizer_config_json
        .map(|candidate| {
            (
                PreparedBertSequenceCapSource::TokenizerConfigJson,
                candidate,
            )
        })
        .or_else(|| {
            candidates
                .config_json
                .map(|candidate| (PreparedBertSequenceCapSource::ConfigJson, candidate))
        });
    let Some((source, candidate)) = selected else {
        return Ok(None);
    };
    let raw_value =
        NonZeroU64::new(candidate.raw_value).ok_or(PreparedBertSequenceCapError::Zero {
            source,
            key: candidate.key,
        })?;
    let capped_value = NonZeroU64::new(raw_value.get().min(PREPARED_BERT_SEQUENCE_CAP)).ok_or(
        PreparedBertSequenceCapError::Zero {
            source,
            key: candidate.key,
        },
    )?;
    Ok(Some(PreparedBertSequenceCap {
        source,
        key: candidate.key,
        raw_value,
        capped_value,
    }))
}

#[cfg(test)]
mod tests;
