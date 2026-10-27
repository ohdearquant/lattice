use super::*;
use std::num::NonZeroU64;

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn exact_limits(bytes: &[u8]) -> PreparedBertTokenizerConfigLimits {
    let len = u64::try_from(bytes.len()).unwrap();
    PreparedBertTokenizerConfigLimits::try_new(nz(len), nz(len.checked_mul(2).unwrap())).unwrap()
}

fn parse(bytes: &[u8]) -> Result<ParsedBertTokenizerConfigFacts, PreparedBertTokenizerConfigError> {
    parse_prepared_bert_tokenizer_config_json(bytes, &exact_limits(bytes))
}

#[test]
fn fixed_precedence_selects_the_first_present_candidate() {
    use PreparedBertTokenizerConfigCandidateKey::{
        MaxPositionEmbeddings, MaxSeqLen, ModelMaxLength, NPositions, TruncationMaxLength,
    };

    for (json, expected) in [
        (r#"{"model_max_length":4096}"#, Some((ModelMaxLength, 4096))),
        (
            r#"{"max_position_embeddings":1024}"#,
            Some((MaxPositionEmbeddings, 1024)),
        ),
        (r#"{"n_positions":768}"#, Some((NPositions, 768))),
        (r#"{"max_seq_len":512}"#, Some((MaxSeqLen, 512))),
        (
            r#"{"truncation":{"max_length":256}}"#,
            Some((TruncationMaxLength, 256)),
        ),
        (
            r#"{"truncation":{"max_length":256},"max_seq_len":512,"n_positions":768,"max_position_embeddings":1024,"model_max_length":4096}"#,
            Some((ModelMaxLength, 4096)),
        ),
        (r#"{"max_length":99,"unrelated":true}"#, None),
    ] {
        let actual = parse(json.as_bytes())
            .unwrap()
            .candidate()
            .map(|candidate| (candidate.key(), candidate.raw_value()));
        assert_eq!(actual, expected, "{json}");
    }
}

#[test]
fn selected_invalid_candidate_fails_without_fallback() {
    use PreparedBertTokenizerConfigCandidateKey::ModelMaxLength;
    use PreparedBertTokenizerConfigError::{InvalidCandidateType, InvalidPositiveInteger};
    use PreparedBertTokenizerConfigUnsignedFault::{
        Exponent, Fractional, Negative, Overflow, Zero,
    };

    for (value, expected) in [
        (
            r#""4096""#,
            InvalidCandidateType {
                key: ModelMaxLength,
                expected: PreparedBertTokenizerConfigExpectedType::PositiveInteger,
                actual: PreparedBertTokenizerConfigValueKind::String,
                at: 20,
            },
        ),
        (
            "0",
            InvalidPositiveInteger {
                key: ModelMaxLength,
                fault: Zero,
                at: 20,
            },
        ),
        (
            "-1",
            InvalidPositiveInteger {
                key: ModelMaxLength,
                fault: Negative,
                at: 20,
            },
        ),
        (
            "1.5",
            InvalidPositiveInteger {
                key: ModelMaxLength,
                fault: Fractional,
                at: 20,
            },
        ),
        (
            "1e3",
            InvalidPositiveInteger {
                key: ModelMaxLength,
                fault: Exponent,
                at: 20,
            },
        ),
        (
            "18446744073709551616",
            InvalidPositiveInteger {
                key: ModelMaxLength,
                fault: Overflow,
                at: 20,
            },
        ),
    ] {
        let json = format!(r#"{{"model_max_length":{value},"max_seq_len":512}}"#);
        assert_eq!(parse(json.as_bytes()), Err(expected), "{json}");
    }
}

#[test]
fn lower_semantic_faults_are_ignored_after_full_syntax_validation() {
    let json = br#"{"model_max_length":4096,"max_position_embeddings":false,"n_positions":-2,"max_seq_len":1.5,"truncation":{"max_length":"bad"}}"#;
    let parsed = parse(json).unwrap();
    let candidate = parsed.candidate().unwrap();
    assert_eq!(
        candidate.key(),
        PreparedBertTokenizerConfigCandidateKey::ModelMaxLength
    );
    assert_eq!(candidate.raw_value(), 4096);

    assert!(matches!(
        parse(br#"{"model_max_length":4096,"ignored":[1,]}"#),
        Err(PreparedBertTokenizerConfigError::MalformedJson { .. })
    ));
}

#[test]
fn duplicate_recognized_paths_fail_for_plain_and_escaped_keys() {
    for (json, key) in [
        (
            br#"{"model_max_length":1,"model_max_leng\u0074h":2}"#.as_slice(),
            PreparedBertTokenizerConfigCandidateKey::ModelMaxLength,
        ),
        (
            br#"{"truncation":{"max_length":1,"max_leng\u0074h":2}}"#.as_slice(),
            PreparedBertTokenizerConfigCandidateKey::TruncationMaxLength,
        ),
        (
            br#"{"truncation":{"max_length":1},"truncation":{"max_length":2}}"#.as_slice(),
            PreparedBertTokenizerConfigCandidateKey::TruncationMaxLength,
        ),
    ] {
        assert!(matches!(
            parse(json),
            Err(PreparedBertTokenizerConfigError::DuplicateCandidate {
                key: actual,
                ..
            }) if actual == key
        ));
    }
}

#[test]
fn strict_root_syntax_and_fixed_nesting_bound_are_enforced() {
    assert!(matches!(
        parse(br#"[]"#),
        Err(PreparedBertTokenizerConfigError::MalformedJson {
            fault: PreparedBertTokenizerConfigSyntaxFault::ExpectedObjectStart,
            ..
        })
    ));
    assert!(matches!(
        parse(br#"{} trailing"#),
        Err(PreparedBertTokenizerConfigError::MalformedJson {
            fault: PreparedBertTokenizerConfigSyntaxFault::TrailingNonWhitespace,
            ..
        })
    ));

    let mut nested = String::from(r#"{"ignored":"#);
    nested.extend(std::iter::repeat_n(
        '[',
        MAX_PREPARED_BERT_TOKENIZER_CONFIG_NESTING,
    ));
    nested.push('0');
    nested.extend(std::iter::repeat_n(
        ']',
        MAX_PREPARED_BERT_TOKENIZER_CONFIG_NESTING,
    ));
    nested.push('}');
    assert!(matches!(
        parse(nested.as_bytes()),
        Err(PreparedBertTokenizerConfigError::MalformedJson {
            fault: PreparedBertTokenizerConfigSyntaxFault::NestingLimitExceeded { .. },
            ..
        })
    ));
}

#[test]
fn independent_byte_and_parse_work_limits_are_reported_with_owned_facts() {
    let bytes = br#"{"model_max_length":2049}"#;
    let len = u64::try_from(bytes.len()).unwrap();
    let exact = exact_limits(bytes);
    let parsed = parse_prepared_bert_tokenizer_config_json(bytes, &exact).unwrap();
    assert_eq!(parsed.tokenizer_config_bytes(), len);
    assert_eq!(parsed.logical_tokenizer_config_parse_work_bytes(), len * 2);
    assert_eq!(parsed.candidate().unwrap().raw_value(), 2049);

    let short_bytes = PreparedBertTokenizerConfigLimits::try_new(nz(len - 1), nz(len * 2)).unwrap();
    assert!(matches!(
        parse_prepared_bert_tokenizer_config_json(bytes, &short_bytes),
        Err(PreparedBertTokenizerConfigError::Exceeded {
            axis: PreparedBertTokenizerConfigLimitAxis::TokenizerConfigBytes,
            ..
        })
    ));

    let short_work = PreparedBertTokenizerConfigLimits::try_new(nz(len), nz(len * 2 - 1)).unwrap();
    assert!(matches!(
        parse_prepared_bert_tokenizer_config_json(bytes, &short_work),
        Err(PreparedBertTokenizerConfigError::Exceeded {
            axis: PreparedBertTokenizerConfigLimitAxis::TokenizerConfigParseWorkBytes,
            ..
        })
    ));
    assert_eq!(
        checked_tokenizer_config_parse_work_bytes(u64::MAX),
        Err(PreparedBertTokenizerConfigError::ArithmeticOverflow(
            PreparedBertTokenizerConfigExpression::TokenizerConfigParseWorkBytes
        ))
    );
}
