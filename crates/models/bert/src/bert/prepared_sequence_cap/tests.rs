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
            RawPreparedBertSequenceCapCandidates::new(Some(candidate(ModelMaxLength, 11)), None),
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
        for (raw_value, capped_value) in [(1, 1), (2048, 2048), (2049, 2048), (u64::MAX, 2048)] {
            let resolved = resolve_prepared_bert_sequence_cap(
                RawPreparedBertSequenceCapCandidates::new(Some(candidate(key, raw_value)), None),
            )
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
