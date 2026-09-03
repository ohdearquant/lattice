//! Dormant final validation of prepared-BERT sequence-cap facts against matched tensors.

use super::prepared_sequence_cap::{
    PreparedBertSequenceCap, PreparedBertSequenceCapKey, PreparedBertSequenceCapSource,
};
use super::prepared_tensor_inventory::MatchedBertInventoryFacts;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertSequenceCapFinalizationError {
    ExceedsValidatedPositionRows {
        capped_value: NonZeroU64,
        validated_position_rows: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RealizedPreparedBertSequenceCapFacts {
    source: PreparedBertSequenceCapSource,
    key: PreparedBertSequenceCapKey,
    raw_value: NonZeroU64,
    capped_value: NonZeroU64,
    validated_position_rows: usize,
}

impl RealizedPreparedBertSequenceCapFacts {
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

    pub(super) fn validated_position_rows(self) -> usize {
        self.validated_position_rows
    }
}

pub(super) fn finalize_prepared_bert_sequence_cap(
    sequence_cap: PreparedBertSequenceCap,
    matched_inventory: MatchedBertInventoryFacts,
) -> Result<RealizedPreparedBertSequenceCapFacts, PreparedBertSequenceCapFinalizationError> {
    finalize_prepared_bert_sequence_cap_with_position_rows(
        sequence_cap,
        matched_inventory.geometry().position_embeddings(),
    )
}

fn finalize_prepared_bert_sequence_cap_with_position_rows(
    sequence_cap: PreparedBertSequenceCap,
    validated_position_rows: usize,
) -> Result<RealizedPreparedBertSequenceCapFacts, PreparedBertSequenceCapFinalizationError> {
    let capped_value = sequence_cap.capped_value();
    if u64::try_from(validated_position_rows)
        .is_ok_and(|validated_position_rows| capped_value.get() > validated_position_rows)
    {
        return Err(
            PreparedBertSequenceCapFinalizationError::ExceedsValidatedPositionRows {
                capped_value,
                validated_position_rows,
            },
        );
    }
    Ok(RealizedPreparedBertSequenceCapFacts {
        source: sequence_cap.source(),
        key: sequence_cap.key(),
        raw_value: sequence_cap.raw_value(),
        capped_value,
        validated_position_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::bert::prepared_sequence_cap::{
        RawPreparedBertSequenceCapCandidate, RawPreparedBertSequenceCapCandidates,
        resolve_prepared_bert_sequence_cap,
    };

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn sequence_cap(
        source: PreparedBertSequenceCapSource,
        key: PreparedBertSequenceCapKey,
        raw_value: u64,
    ) -> PreparedBertSequenceCap {
        let candidate = RawPreparedBertSequenceCapCandidate::new(key, raw_value);
        let candidates = match source {
            PreparedBertSequenceCapSource::TokenizerConfigJson => {
                RawPreparedBertSequenceCapCandidates::new(Some(candidate), None)
            }
            PreparedBertSequenceCapSource::ConfigJson => {
                RawPreparedBertSequenceCapCandidates::new(None, Some(candidate))
            }
        };
        resolve_prepared_bert_sequence_cap(candidates)
            .unwrap()
            .unwrap()
    }

    #[test]
    fn exact_and_capped_values_fit_validated_position_rows_and_preserve_all_facts() {
        let _entrypoint: fn(
            PreparedBertSequenceCap,
            MatchedBertInventoryFacts,
        ) -> Result<
            RealizedPreparedBertSequenceCapFacts,
            PreparedBertSequenceCapFinalizationError,
        > = finalize_prepared_bert_sequence_cap;

        for (source, key, raw_value, capped_value) in [
            (
                PreparedBertSequenceCapSource::ConfigJson,
                PreparedBertSequenceCapKey::ModelMaxLength,
                2048,
                2048,
            ),
            (
                PreparedBertSequenceCapSource::TokenizerConfigJson,
                PreparedBertSequenceCapKey::NPositions,
                2049,
                2048,
            ),
        ] {
            let realized = finalize_prepared_bert_sequence_cap_with_position_rows(
                sequence_cap(source, key, raw_value),
                2048,
            )
            .unwrap();
            assert_eq!(realized.source(), source);
            assert_eq!(realized.key(), key);
            assert_eq!(realized.raw_value(), nz(raw_value));
            assert_eq!(realized.capped_value(), nz(capped_value));
            assert_eq!(realized.validated_position_rows(), 2048);
        }
    }

    #[test]
    fn capped_values_above_validated_position_rows_fail_at_exact_boundaries() {
        for (sequence_cap, validated_position_rows, capped_value) in [
            (
                sequence_cap(
                    PreparedBertSequenceCapSource::ConfigJson,
                    PreparedBertSequenceCapKey::MaxPositionEmbeddings,
                    513,
                ),
                512,
                513,
            ),
            (
                sequence_cap(
                    PreparedBertSequenceCapSource::TokenizerConfigJson,
                    PreparedBertSequenceCapKey::TruncationMaxLength,
                    4096,
                ),
                1024,
                2048,
            ),
        ] {
            assert_eq!(
                finalize_prepared_bert_sequence_cap_with_position_rows(
                    sequence_cap,
                    validated_position_rows,
                ),
                Err(
                    PreparedBertSequenceCapFinalizationError::ExceedsValidatedPositionRows {
                        capped_value: nz(capped_value),
                        validated_position_rows,
                    }
                )
            );
        }
    }
}
