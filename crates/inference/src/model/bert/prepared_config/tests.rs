use super::*;
use crate::model::bert::prepared_facts::{
    BertGeometryLimits, BertPoolerMembers, PreparedBertFactError, RawBertConfigFacts,
    analyze_prepared_geometry,
};
use crate::model::bert::prepared_sequence_cap::{
    PreparedBertSequenceCapKey, RawPreparedBertSequenceCapCandidate,
};
use std::num::NonZeroU64;

const CANONICAL: &str = concat!(
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

fn limits(config_bytes: u64, parse_work_bytes: u64) -> PreparedBertConfigLimits {
    PreparedBertConfigLimits::try_new(nz(config_bytes), nz(parse_work_bytes)).unwrap()
}

fn exact_limits(bytes: &[u8]) -> PreparedBertConfigLimits {
    let len = u64::try_from(bytes.len()).unwrap();
    limits(len, len.checked_mul(2).unwrap())
}

fn parse(bytes: &[u8]) -> Result<ParsedBertConfigFacts, PreparedBertConfigError> {
    parse_prepared_bert_config_json(bytes, &exact_limits(bytes))
}

fn expected_raw(epsilon: f64) -> RawBertConfigFacts {
    RawBertConfigFacts::new(11, 12, 3, 3, 17, 19, 5, epsilon)
}

fn field_members() -> [(&'static str, &'static str, PreparedBertConfigField); 8] {
    [
        ("vocab_size", "11", PreparedBertConfigField::VocabSize),
        ("hidden_size", "12", PreparedBertConfigField::HiddenSize),
        (
            "num_hidden_layers",
            "3",
            PreparedBertConfigField::NumHiddenLayers,
        ),
        (
            "num_attention_heads",
            "3",
            PreparedBertConfigField::NumAttentionHeads,
        ),
        (
            "intermediate_size",
            "17",
            PreparedBertConfigField::IntermediateSize,
        ),
        (
            "max_position_embeddings",
            "19",
            PreparedBertConfigField::MaxPositionEmbeddings,
        ),
        (
            "type_vocab_size",
            "5",
            PreparedBertConfigField::TypeVocabSize,
        ),
        (
            "layer_norm_eps",
            "1e-12",
            PreparedBertConfigField::LayerNormEps,
        ),
    ]
}

fn document_with(
    omitted: Option<PreparedBertConfigField>,
    replacement: Option<(PreparedBertConfigField, &str)>,
    suffix_members: &[(&str, &str)],
) -> String {
    let mut members = Vec::new();
    for (name, value, field) in field_members() {
        if omitted == Some(field) {
            continue;
        }
        let value = replacement
            .filter(|(replacement_field, _)| *replacement_field == field)
            .map_or(value, |(_, replacement_value)| replacement_value);
        members.push(format!("\"{name}\":{value}"));
    }
    members.extend(
        suffix_members
            .iter()
            .map(|(name, value)| format!("\"{name}\":{value}")),
    );
    format!("{{{}}}", members.join(","))
}

fn generous_geometry(raw: RawBertConfigFacts) -> Result<(), PreparedBertFactError> {
    let geometry = analyze_prepared_geometry(
        raw,
        &BertGeometryLimits::new(
            nz(1_000),
            nz(1_000),
            nz(1_000),
            nz(1_000),
            nz(1_000),
            nz(1_000),
            nz(1_000),
            nz(100_000),
            nz(100_000_000),
        ),
        BertPoolerMembers::new(false, false),
    )?;
    assert_eq!(geometry.vocab_size(), 11);
    assert_eq!(geometry.hidden_size(), 12);
    assert_eq!(geometry.hidden_layers(), 3);
    assert_eq!(geometry.attention_heads(), 3);
    assert_eq!(geometry.intermediate_size(), 17);
    assert_eq!(geometry.position_embeddings(), 19);
    assert_eq!(geometry.type_vocab_size(), 5);
    Ok(())
}

#[test]
fn canonical_permuted_and_whitespace_configs_preserve_exact_facts_and_work() {
    let parsed = parse(CANONICAL.as_bytes()).unwrap();
    assert_eq!(parsed.raw(), expected_raw(1e-12));
    assert_eq!(
        parsed.config_bytes(),
        u64::try_from(CANONICAL.len()).unwrap()
    );
    assert_eq!(
        parsed.logical_config_parse_work_bytes(),
        u64::try_from(CANONICAL.len()).unwrap() * 2
    );
    generous_geometry(parsed.raw()).unwrap();

    let permuted = concat!(
        " { \n\t\"layer_norm_eps\" : 0.000000000001,",
        "\"type_vocab_size\":5,",
        "\"max_position_embeddings\":19,",
        "\"intermediate_size\":17,",
        "\"num_attention_heads\":3,",
        "\"num_hidden_layers\":3,",
        "\"hidden_size\":12,",
        "\"vocab_size\":11 } \r\n"
    );
    let parsed = parse(permuted.as_bytes()).unwrap();
    assert_eq!(parsed.raw(), expected_raw(1e-12));
    assert_eq!(
        parsed.logical_config_parse_work_bytes(),
        u64::try_from(permuted.len()).unwrap() * 2
    );
}

#[test]
fn unknown_values_and_duplicate_unknown_keys_are_structurally_checked_but_ignored() {
    let document = document_with(
        None,
        None,
        &[
            ("unknown", "null"),
            ("unknown", "[true,{\"nested\":[1,2,3]}]"),
            ("nested", "{\"vocab_size\":999,\"layer_norm_eps\":0}"),
            ("text", "\"vocab_size layer_norm_eps\""),
        ],
    );
    assert_eq!(
        parse(document.as_bytes()).unwrap().raw(),
        expected_raw(1e-12)
    );

    let missing_root = document_with(
        Some(PreparedBertConfigField::VocabSize),
        None,
        &[("nested", "{\"vocab_size\":11}")],
    );
    assert_eq!(
        parse(missing_root.as_bytes()),
        Err(PreparedBertConfigError::MissingField {
            field: PreparedBertConfigField::VocabSize,
        })
    );
}

#[test]
fn escaped_required_keys_decode_and_collide_with_literal_spellings() {
    let escaped = CANONICAL.replacen("\"vocab_size\"", "\"vocab_\\u0073ize\"", 1);
    assert_eq!(
        parse(escaped.as_bytes()).unwrap().raw(),
        expected_raw(1e-12)
    );

    let duplicate = document_with(None, None, &[("vocab_\\u0073ize", "11")]);
    let first_at = duplicate.find("\"vocab_size\"").unwrap();
    let duplicate_at = duplicate.rfind("\"vocab_\\u0073ize\"").unwrap();
    assert_eq!(
        parse(duplicate.as_bytes()),
        Err(PreparedBertConfigError::DuplicateField {
            field: PreparedBertConfigField::VocabSize,
            first_at,
            duplicate_at,
        })
    );
}

#[test]
fn every_required_field_has_typed_missing_and_duplicate_failures() {
    for (name, value, field) in field_members() {
        let missing = document_with(Some(field), None, &[]);
        assert_eq!(
            parse(missing.as_bytes()),
            Err(PreparedBertConfigError::MissingField { field }),
            "missing {name}"
        );

        let duplicate = document_with(None, None, &[(name, value)]);
        let first_at = duplicate.find(&format!("\"{name}\"")).unwrap();
        let duplicate_at = duplicate.rfind(&format!("\"{name}\"")).unwrap();
        assert_eq!(
            parse(duplicate.as_bytes()),
            Err(PreparedBertConfigError::DuplicateField {
                field,
                first_at,
                duplicate_at,
            }),
            "duplicate {name}"
        );
    }
}

#[test]
fn required_value_kinds_are_exact_and_numeric_strings_are_rejected() {
    for (_, _, field) in field_members() {
        let document = document_with(None, Some((field, "\"11\"")), &[]);
        let expected = if field == PreparedBertConfigField::LayerNormEps {
            PreparedBertConfigExpectedType::Number
        } else {
            PreparedBertConfigExpectedType::UnsignedInteger
        };
        assert!(matches!(
            parse(document.as_bytes()),
            Err(PreparedBertConfigError::InvalidFieldType {
                field: actual_field,
                expected: actual_expected,
                actual: PreparedBertConfigValueKind::String,
                ..
            }) if actual_field == field && actual_expected == expected
        ));
    }

    for (value, kind) in [
        ("null", PreparedBertConfigValueKind::Null),
        ("true", PreparedBertConfigValueKind::Boolean),
        ("[]", PreparedBertConfigValueKind::Array),
        ("{}", PreparedBertConfigValueKind::Object),
    ] {
        let document = document_with(None, Some((PreparedBertConfigField::VocabSize, value)), &[]);
        assert!(matches!(
            parse(document.as_bytes()),
            Err(PreparedBertConfigError::InvalidFieldType {
                field: PreparedBertConfigField::VocabSize,
                expected: PreparedBertConfigExpectedType::UnsignedInteger,
                actual,
                ..
            }) if actual == kind
        ));
    }
}

#[test]
fn unsigned_integer_lexemes_pin_zero_max_and_each_rejected_class() {
    for value in ["0", "18446744073709551615"] {
        let document = document_with(None, Some((PreparedBertConfigField::VocabSize, value)), &[]);
        assert!(parse(document.as_bytes()).is_ok(), "u64 lexeme {value}");
    }

    for (value, fault) in [
        ("-0", PreparedBertConfigUnsignedFault::Negative),
        ("1.0", PreparedBertConfigUnsignedFault::Fractional),
        ("1e0", PreparedBertConfigUnsignedFault::Exponent),
        (
            "18446744073709551616",
            PreparedBertConfigUnsignedFault::Overflow,
        ),
    ] {
        let document = document_with(None, Some((PreparedBertConfigField::VocabSize, value)), &[]);
        assert!(matches!(
            parse(document.as_bytes()),
            Err(PreparedBertConfigError::InvalidUnsignedInteger {
                field: PreparedBertConfigField::VocabSize,
                fault: actual,
                ..
            }) if actual == fault
        ));
    }

    let leading_zero = document_with(None, Some((PreparedBertConfigField::VocabSize, "01")), &[]);
    assert!(matches!(
        parse(leading_zero.as_bytes()),
        Err(PreparedBertConfigError::MalformedJson {
            fault: PreparedBertConfigSyntaxFault::InvalidNumber,
            ..
        })
    ));
}

#[test]
fn epsilon_preserves_f64_facts_and_defers_realization_failures_to_geometry() {
    for (lexeme, expected) in [
        ("0.000000000001", 1e-12_f64),
        ("1e-12", 1e-12_f64),
        ("1.0000000000000002", 1.0000000000000002_f64),
    ] {
        let document = document_with(
            None,
            Some((PreparedBertConfigField::LayerNormEps, lexeme)),
            &[],
        );
        assert_eq!(
            parse(document.as_bytes()).unwrap().raw(),
            expected_raw(expected)
        );
    }

    for lexeme in ["0", "-1", "1e999", "1e-9999"] {
        let document = document_with(
            None,
            Some((PreparedBertConfigField::LayerNormEps, lexeme)),
            &[],
        );
        let raw = parse(document.as_bytes()).unwrap().raw();
        assert_eq!(
            generous_geometry(raw),
            Err(PreparedBertFactError::InvalidLayerNormEpsilon),
            "epsilon {lexeme}"
        );
    }
}

#[test]
fn malformed_json_classes_are_typed_before_semantics() {
    let cases: Vec<(Vec<u8>, PreparedBertConfigSyntaxFault)> = vec![
        (
            b"[]".to_vec(),
            PreparedBertConfigSyntaxFault::ExpectedObjectStart,
        ),
        (
            format!("{CANONICAL} trailing").into_bytes(),
            PreparedBertConfigSyntaxFault::TrailingNonWhitespace,
        ),
        (
            CANONICAL.replace("11,", "11,,").into_bytes(),
            PreparedBertConfigSyntaxFault::ExpectedString,
        ),
        (
            CANONICAL.replace("11,", "11 12,").into_bytes(),
            PreparedBertConfigSyntaxFault::ExpectedCommaOrEnd,
        ),
        (
            CANONICAL.replace("11,", "01,").into_bytes(),
            PreparedBertConfigSyntaxFault::InvalidNumber,
        ),
        (
            CANONICAL.replace("11,", "txue,").into_bytes(),
            PreparedBertConfigSyntaxFault::InvalidLiteral,
        ),
        (
            CANONICAL
                .replace("\"vocab_size\"", "\"vocab_\\qsize\"")
                .into_bytes(),
            PreparedBertConfigSyntaxFault::InvalidStringEscape,
        ),
        (
            CANONICAL
                .replace("\"vocab_size\"", "\"vocab_\\uD800size\"")
                .into_bytes(),
            PreparedBertConfigSyntaxFault::InvalidUnicodeEscape,
        ),
        (
            CANONICAL.trim_end_matches('}').as_bytes().to_vec(),
            PreparedBertConfigSyntaxFault::UnexpectedEof,
        ),
    ];
    for (bytes, fault) in cases {
        assert!(matches!(
            parse(&bytes),
            Err(PreparedBertConfigError::MalformedJson { fault: actual, .. })
                if actual == fault
        ));
    }

    let mut invalid_utf8 = CANONICAL.as_bytes().to_vec();
    invalid_utf8.splice(2..3, [0xff]);
    assert!(matches!(
        parse(&invalid_utf8),
        Err(PreparedBertConfigError::MalformedJson {
            fault: PreparedBertConfigSyntaxFault::InvalidUtf8,
            ..
        })
    ));

    let semantic_then_syntax = CANONICAL
        .replacen("\"vocab_size\":11", "\"vocab_size\":\"bad\"", 1)
        .trim_end_matches('}')
        .to_owned();
    assert!(matches!(
        parse(semantic_then_syntax.as_bytes()),
        Err(PreparedBertConfigError::MalformedJson {
            fault: PreparedBertConfigSyntaxFault::UnexpectedEof,
            ..
        })
    ));
}

#[test]
fn fixed_nesting_depth_accepts_64_and_rejects_65_without_recursion() {
    fn nested(depth: usize) -> String {
        let mut value = "0".to_owned();
        for _ in 0..depth {
            value = format!("[{value}]");
        }
        document_with(None, None, &[("unknown", &value)])
    }

    let at_limit = nested(MAX_PREPARED_BERT_CONFIG_NESTING - 1);
    assert!(parse(at_limit.as_bytes()).is_ok());

    let over_limit = nested(MAX_PREPARED_BERT_CONFIG_NESTING);
    assert!(matches!(
        parse(over_limit.as_bytes()),
        Err(PreparedBertConfigError::MalformedJson {
            fault: PreparedBertConfigSyntaxFault::NestingLimitExceeded { limit },
            ..
        }) if limit == MAX_PREPARED_BERT_CONFIG_NESTING
    ));
}

#[test]
fn config_and_parse_work_ceilings_are_independent_exact_gates() {
    let len = u64::try_from(CANONICAL.len()).unwrap();
    let work = len * 2;
    let exact = limits(len, work);
    assert!(parse_prepared_bert_config_json(CANONICAL.as_bytes(), &exact).is_ok());

    let short_config = limits(len - 1, work - 1);
    assert_eq!(
        parse_prepared_bert_config_json(CANONICAL.as_bytes(), &short_config),
        Err(PreparedBertConfigError::Exceeded {
            axis: PreparedBertConfigLimitAxis::ConfigBytes,
            actual: len,
            limit: len - 1,
        })
    );

    let short_work = limits(len, work - 1);
    assert_eq!(
        parse_prepared_bert_config_json(CANONICAL.as_bytes(), &short_work),
        Err(PreparedBertConfigError::Exceeded {
            axis: PreparedBertConfigLimitAxis::ConfigParseWorkBytes,
            actual: work,
            limit: work - 1,
        })
    );
}

#[test]
fn parse_work_and_platform_arithmetic_pin_exact_boundaries() {
    assert_eq!(
        checked_config_parse_work_bytes(u64::MAX / 2),
        Ok(u64::MAX - 1)
    );
    assert_eq!(
        checked_config_parse_work_bytes(u64::MAX / 2 + 1),
        Err(PreparedBertConfigError::ArithmeticOverflow(
            PreparedBertConfigExpression::ConfigParseWorkBytes,
        ))
    );

    let exact = PreparedBertConfigLimits::try_new_with_platform_max(nz(8), nz(16), 16);
    assert!(exact.is_ok());
    assert_eq!(
        PreparedBertConfigLimits::try_new_with_platform_max(nz(17), nz(16), 16),
        Err(PreparedBertConfigError::PlatformUnrepresentable {
            axis: PreparedBertConfigLimitAxis::ConfigBytes,
            value: 17,
        })
    );
    assert_eq!(
        PreparedBertConfigLimits::try_new_with_platform_max(nz(8), nz(17), 16),
        Err(PreparedBertConfigError::PlatformUnrepresentable {
            axis: PreparedBertConfigLimitAxis::ConfigParseWorkBytes,
            value: 17,
        })
    );
}

#[test]
fn semantic_precedence_is_duplicate_then_type_then_number_then_missing() {
    let duplicate_bad_type = document_with(None, None, &[("vocab_size", "\"bad\"")]);
    assert!(matches!(
        parse(duplicate_bad_type.as_bytes()),
        Err(PreparedBertConfigError::DuplicateField {
            field: PreparedBertConfigField::VocabSize,
            ..
        })
    ));

    let first_fault = document_with(
        None,
        Some((PreparedBertConfigField::VocabSize, "-1")),
        &[("hidden_size", "18446744073709551616")],
    );
    assert!(matches!(
        parse(first_fault.as_bytes()),
        Err(PreparedBertConfigError::InvalidUnsignedInteger {
            field: PreparedBertConfigField::VocabSize,
            fault: PreparedBertConfigUnsignedFault::Negative,
            ..
        })
    ));

    let first_missing = document_with(Some(PreparedBertConfigField::VocabSize), None, &[])
        .replace("\"hidden_size\":12,", "");
    assert_eq!(
        parse(first_missing.as_bytes()),
        Err(PreparedBertConfigError::MissingField {
            field: PreparedBertConfigField::VocabSize,
        })
    );
}

#[test]
fn parsed_facts_do_not_borrow_the_input_buffer() {
    let parsed = {
        let bytes = CANONICAL.as_bytes().to_vec();
        parse(&bytes).unwrap()
    };
    assert_eq!(parsed.raw(), expected_raw(1e-12));
}

#[test]
fn model_max_length_wins_and_absence_falls_back_to_position_embeddings() {
    assert_eq!(
        parse(CANONICAL.as_bytes())
            .unwrap()
            .sequence_cap_candidate(),
        Ok(RawPreparedBertSequenceCapCandidate::new(
            PreparedBertSequenceCapKey::MaxPositionEmbeddings,
            19,
        ))
    );

    let present = document_with(None, None, &[("model_max_length", "128")]);
    assert_eq!(
        parse(present.as_bytes()).unwrap().sequence_cap_candidate(),
        Ok(RawPreparedBertSequenceCapCandidate::new(
            PreparedBertSequenceCapKey::ModelMaxLength,
            128,
        ))
    );
}

#[test]
fn present_model_max_length_rejects_non_positive_or_inexact_u64_without_fallback() {
    let string = document_with(None, None, &[("model_max_length", "\"128\"")]);
    assert!(matches!(
        parse(string.as_bytes()).unwrap().sequence_cap_candidate(),
        Err(PreparedBertConfigSequenceCandidateError::InvalidType {
            key: PreparedBertSequenceCapKey::ModelMaxLength,
            actual: PreparedBertConfigValueKind::String,
            ..
        })
    ));

    for (value, fault) in [
        ("0", PreparedBertConfigUnsignedFault::Zero),
        ("-1", PreparedBertConfigUnsignedFault::Negative),
        ("1.0", PreparedBertConfigUnsignedFault::Fractional),
        ("1e0", PreparedBertConfigUnsignedFault::Exponent),
        (
            "18446744073709551616",
            PreparedBertConfigUnsignedFault::Overflow,
        ),
    ] {
        let document = document_with(None, None, &[("model_max_length", value)]);
        assert!(matches!(
            parse(document.as_bytes())
                .unwrap()
                .sequence_cap_candidate(),
            Err(PreparedBertConfigSequenceCandidateError::InvalidPositiveInteger {
                key: PreparedBertSequenceCapKey::ModelMaxLength,
                fault: actual,
                ..
            }) if actual == fault
        ));
    }
}

#[test]
fn decoded_model_max_length_duplicates_are_rejected() {
    let duplicate = document_with(
        None,
        None,
        &[
            ("model_max_length", "128"),
            ("model_max_\\u006cength", "64"),
        ],
    );
    let first_at = duplicate.find("\"model_max_length\"").unwrap();
    let duplicate_at = duplicate.rfind("\"model_max_\\u006cength\"").unwrap();
    assert_eq!(
        parse(duplicate.as_bytes()),
        Err(PreparedBertConfigError::DuplicateField {
            field: PreparedBertConfigField::ModelMaxLength,
            first_at,
            duplicate_at,
        })
    );
}
