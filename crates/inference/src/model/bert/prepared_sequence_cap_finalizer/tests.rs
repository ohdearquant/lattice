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
