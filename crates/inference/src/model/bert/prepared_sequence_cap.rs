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
mod tests {
    use super::*;

    fn candidate(
        key: PreparedBertSequenceCapKey,
        raw_value: u64,
    ) -> RawPreparedBertSequenceCapCandidate {
        RawPreparedBertSequenceCapCandidate::new(key, raw_value)
    }

    #[test]
    fn tokenizer_config_precedes_config_for_all_file_slot_masks() {
        use PreparedBertSequenceCapKey::{MaxPositionEmbeddings, ModelMaxLength};
        use PreparedBertSequenceCapSource::{ConfigJson, TokenizerConfigJson};

        for (raw, expected) in [
            (RawPreparedBertSequenceCapCandidates::new(None, None), None),
            (
                RawPreparedBertSequenceCapCandidates::new(
                    None,
                    Some(candidate(MaxPositionEmbeddings, 22)),
                ),
                Some((ConfigJson, MaxPositionEmbeddings, 22)),
            ),
            (
                RawPreparedBertSequenceCapCandidates::new(
                    Some(candidate(ModelMaxLength, 11)),
                    None,
                ),
                Some((TokenizerConfigJson, ModelMaxLength, 11)),
            ),
            (
                RawPreparedBertSequenceCapCandidates::new(
                    Some(candidate(ModelMaxLength, 11)),
                    Some(candidate(MaxPositionEmbeddings, 22)),
                ),
                Some((TokenizerConfigJson, ModelMaxLength, 11)),
            ),
        ] {
            let actual = resolve_prepared_bert_sequence_cap(raw)
                .unwrap()
                .map(|cap| (cap.source(), cap.key(), cap.raw_value().get()));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn selected_raw_value_is_preserved_and_capped_at_exactly_2048() {
        use PreparedBertSequenceCapKey::{
            MaxPositionEmbeddings, MaxSeqLen, ModelMaxLength, NPositions, TruncationMaxLength,
        };

        for key in [
            ModelMaxLength,
            MaxPositionEmbeddings,
            NPositions,
            MaxSeqLen,
            TruncationMaxLength,
        ] {
            for (raw_value, capped_value) in [(1, 1), (2048, 2048), (2049, 2048), (u64::MAX, 2048)]
            {
                let resolved =
                    resolve_prepared_bert_sequence_cap(RawPreparedBertSequenceCapCandidates::new(
                        Some(candidate(key, raw_value)),
                        None,
                    ))
                    .unwrap()
                    .unwrap();
                assert_eq!(resolved.key(), key);
                assert_eq!(resolved.raw_value().get(), raw_value);
                assert_eq!(resolved.capped_value().get(), capped_value);
            }
        }
    }

    #[test]
    fn selected_zero_fails_without_falling_back_to_the_other_file() {
        use PreparedBertSequenceCapKey::{MaxPositionEmbeddings, ModelMaxLength};
        use PreparedBertSequenceCapSource::{ConfigJson, TokenizerConfigJson};

        assert_eq!(
            resolve_prepared_bert_sequence_cap(RawPreparedBertSequenceCapCandidates::new(
                Some(candidate(ModelMaxLength, 0)),
                Some(candidate(MaxPositionEmbeddings, 512)),
            )),
            Err(PreparedBertSequenceCapError::Zero {
                source: TokenizerConfigJson,
                key: ModelMaxLength,
            })
        );
        assert_eq!(
            resolve_prepared_bert_sequence_cap(RawPreparedBertSequenceCapCandidates::new(
                None,
                Some(candidate(MaxPositionEmbeddings, 0)),
            )),
            Err(PreparedBertSequenceCapError::Zero {
                source: ConfigJson,
                key: MaxPositionEmbeddings,
            })
        );
    }
}
