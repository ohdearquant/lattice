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
mod tests {
    use super::*;
    use crate::model::bert::prepared_config::{
        PreparedBertConfigLimits, PreparedBertConfigValueKind, parse_prepared_bert_config_json,
    };
    use crate::model::bert::prepared_sequence_cap::PreparedBertSequenceCapSource;
    use crate::model::bert::prepared_tokenizer_config::{
        PreparedBertTokenizerConfigLimits, parse_prepared_bert_tokenizer_config_json,
    };
    use std::num::NonZeroU64;

    const CONFIG: &str = concat!(
        "{\"vocab_size\":11,",
        "\"hidden_size\":12,",
        "\"num_hidden_layers\":3,",
        "\"num_attention_heads\":3,",
        "\"intermediate_size\":17,",
        "\"max_position_embeddings\":19,",
        "\"type_vocab_size\":5,",
        "\"layer_norm_eps\":1e-12}"
    );

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn parse_config(bytes: &[u8]) -> ParsedBertConfigFacts {
        let len = u64::try_from(bytes.len()).unwrap();
        let limits = PreparedBertConfigLimits::try_new(nz(len), nz(len * 2)).unwrap();
        parse_prepared_bert_config_json(bytes, &limits).unwrap()
    }

    fn parse_tokenizer_config(bytes: &[u8]) -> ParsedBertTokenizerConfigFacts {
        let len = u64::try_from(bytes.len()).unwrap();
        let limits = PreparedBertTokenizerConfigLimits::try_new(nz(len), nz(len * 2)).unwrap();
        parse_prepared_bert_tokenizer_config_json(bytes, &limits).unwrap()
    }

    fn config_with_model_max_length(value: &str) -> String {
        CONFIG.replacen(
            "\"type_vocab_size\"",
            &format!("\"model_max_length\":{value},\"type_vocab_size\""),
            1,
        )
    }

    #[test]
    fn every_tokenizer_key_maps_exhaustively_and_shadows_deferred_config_failure() {
        use PreparedBertSequenceCapKey::{
            MaxPositionEmbeddings, MaxSeqLen, ModelMaxLength, NPositions, TruncationMaxLength,
        };

        let invalid_config = parse_config(config_with_model_max_length("\"bad\"").as_bytes());
        for (json, expected_key) in [
            (br#"{"model_max_length":3000}"#.as_slice(), ModelMaxLength),
            (
                br#"{"max_position_embeddings":3000}"#.as_slice(),
                MaxPositionEmbeddings,
            ),
            (br#"{"n_positions":3000}"#.as_slice(), NPositions),
            (br#"{"max_seq_len":3000}"#.as_slice(), MaxSeqLen),
            (
                br#"{"truncation":{"max_length":3000}}"#.as_slice(),
                TruncationMaxLength,
            ),
        ] {
            let resolved = resolve_prepared_bert_sequence_cap_from_facts(
                parse_tokenizer_config(json),
                invalid_config,
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                (
                    resolved.source(),
                    resolved.key(),
                    resolved.raw_value().get(),
                    resolved.capped_value().get(),
                ),
                (
                    PreparedBertSequenceCapSource::TokenizerConfigJson,
                    expected_key,
                    3000,
                    2048,
                )
            );
        }
    }

    #[test]
    fn missing_tokenizer_candidate_uses_config_model_max_length_or_position_fallback() {
        let no_tokenizer_candidate = parse_tokenizer_config(br#"{}"#);
        for (config, expected_key, raw_value, capped_value) in [
            (
                parse_config(config_with_model_max_length("4096").as_bytes()),
                PreparedBertSequenceCapKey::ModelMaxLength,
                4096,
                2048,
            ),
            (
                parse_config(CONFIG.as_bytes()),
                PreparedBertSequenceCapKey::MaxPositionEmbeddings,
                19,
                19,
            ),
        ] {
            let resolved =
                resolve_prepared_bert_sequence_cap_from_facts(no_tokenizer_candidate, config)
                    .unwrap()
                    .unwrap();
            assert_eq!(resolved.source(), PreparedBertSequenceCapSource::ConfigJson);
            assert_eq!(resolved.key(), expected_key);
            assert_eq!(resolved.raw_value().get(), raw_value);
            assert_eq!(resolved.capped_value().get(), capped_value);
        }
    }

    #[test]
    fn missing_tokenizer_candidate_surfaces_deferred_config_and_resolver_errors() {
        let no_tokenizer_candidate = parse_tokenizer_config(br#"{}"#);
        let invalid_config = parse_config(config_with_model_max_length("\"bad\"").as_bytes());
        assert!(matches!(
            resolve_prepared_bert_sequence_cap_from_facts(no_tokenizer_candidate, invalid_config,),
            Err(PreparedBertParsedSequenceCapError::ConfigJsonCandidate(
                PreparedBertConfigSequenceCandidateError::InvalidType {
                    key: PreparedBertSequenceCapKey::ModelMaxLength,
                    actual: PreparedBertConfigValueKind::String,
                    ..
                }
            ))
        ));

        let zero_positions = CONFIG.replace(
            "\"max_position_embeddings\":19",
            "\"max_position_embeddings\":0",
        );
        assert_eq!(
            resolve_prepared_bert_sequence_cap_from_facts(
                no_tokenizer_candidate,
                parse_config(zero_positions.as_bytes()),
            ),
            Err(PreparedBertParsedSequenceCapError::Resolver(
                PreparedBertSequenceCapError::Zero {
                    source: PreparedBertSequenceCapSource::ConfigJson,
                    key: PreparedBertSequenceCapKey::MaxPositionEmbeddings,
                }
            ))
        );
    }
}
