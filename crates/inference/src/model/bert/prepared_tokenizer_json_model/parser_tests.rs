use super::*;
use std::num::NonZeroU64;

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn exact_limits(bytes: &[u8]) -> PreparedBertTokenizerJsonLimits {
    let len = u64::try_from(bytes.len()).unwrap();
    PreparedBertTokenizerJsonLimits::try_new(nz(len), nz(len * 2)).unwrap()
}

fn parse(
    bytes: &[u8],
) -> Result<ParsedBertTokenizerJsonModelFacts, PreparedBertTokenizerJsonParseError> {
    parse_prepared_bert_tokenizer_json_model(bytes, &exact_limits(bytes))
}

#[test]
fn four_supported_direct_model_type_spellings_parse_including_escapes() {
    use PreparedBertTokenizerJsonModel::{Bpe, Unigram, WordPiece};

    for (json, expected) in [
        (br#"{"model":{"type":"WordPiece"}}"#.as_slice(), WordPiece),
        (
            br#"{"m\u006fdel":{"t\u0079pe":"B\u0050E"}}"#.as_slice(),
            Bpe,
        ),
        (br#"{"model":{"type":"Unigram"}}"#.as_slice(), Unigram),
        (
            br#"{"model":{"type":"SentencePiece\u0055nigram"}}"#.as_slice(),
            Unigram,
        ),
    ] {
        let parsed = parse(json).unwrap();
        assert_eq!(parsed.model(), expected);
        assert_eq!(
            parsed.tokenizer_json_bytes(),
            u64::try_from(json.len()).unwrap()
        );
        assert_eq!(
            parsed.logical_tokenizer_json_parse_work_bytes(),
            u64::try_from(json.len()).unwrap() * 2
        );
    }
}

#[test]
fn missing_wrong_empty_unsupported_and_wrong_case_are_typed() {
    use PreparedBertTokenizerJsonField::{Model, ModelType};
    use PreparedBertTokenizerJsonModelError::{Empty, Missing, Unsupported};
    use PreparedBertTokenizerJsonValueKind::{Array, Number};

    assert_eq!(
        parse(br#"{}"#),
        Err(PreparedBertTokenizerJsonParseError::MissingField { field: Model })
    );
    assert!(matches!(
        parse(br#"{"model":[]}"#),
        Err(PreparedBertTokenizerJsonParseError::InvalidFieldType {
            field: Model,
            expected: PreparedBertTokenizerJsonExpectedType::Object,
            actual: Array,
            ..
        })
    ));
    assert_eq!(
        parse(br#"{"model":{}}"#),
        Err(PreparedBertTokenizerJsonParseError::ModelType(Missing))
    );
    assert!(matches!(
        parse(br#"{"model":{"type":1}}"#),
        Err(PreparedBertTokenizerJsonParseError::InvalidFieldType {
            field: ModelType,
            expected: PreparedBertTokenizerJsonExpectedType::String,
            actual: Number,
            ..
        })
    ));
    for (value, error) in [
        ("", Empty),
        ("BertWordPiece", Unsupported),
        ("wordpiece", Unsupported),
    ] {
        let json = format!(r#"{{"model":{{"type":"{value}"}}}}"#);
        assert_eq!(
            parse(json.as_bytes()),
            Err(PreparedBertTokenizerJsonParseError::ModelType(error))
        );
    }
}

#[test]
fn decoded_relevant_keys_collide_at_root_and_model_child() {
    for (json, field) in [
        (
            br#"{"model":{"type":"WordPiece"},"m\u006fdel":{"type":"BPE"}}"#.as_slice(),
            PreparedBertTokenizerJsonField::Model,
        ),
        (
            br#"{"model":{"type":"WordPiece","t\u0079pe":"BPE"}}"#.as_slice(),
            PreparedBertTokenizerJsonField::ModelType,
        ),
    ] {
        assert!(matches!(
            parse(json),
            Err(PreparedBertTokenizerJsonParseError::DuplicateField {
                field: actual,
                ..
            }) if actual == field
        ));
    }
}

#[test]
fn nested_decoys_are_ignored_but_unrelated_malformed_json_wins_first() {
    let decoys = br#"{"nested":{"model":{"type":"BPE"}},"model":{"nested":{"type":"Unigram"},"type":"WordPiece"}}"#;
    assert_eq!(
        parse(decoys).unwrap().model(),
        PreparedBertTokenizerJsonModel::WordPiece
    );

    assert!(matches!(
        parse(br#"{"model":{"type":"WordPiece"},"ignored":[1,]}"#),
        Err(PreparedBertTokenizerJsonParseError::MalformedJson {
            fault: PreparedBertTokenizerJsonSyntaxFault::TrailingComma,
            ..
        })
    ));
}

#[test]
fn byte_work_nesting_platform_and_overflow_bounds_are_exact() {
    let bytes = br#"{"model":{"type":"WordPiece"}}"#;
    let len = u64::try_from(bytes.len()).unwrap();
    let work = len * 2;
    assert!(parse_prepared_bert_tokenizer_json_model(bytes, &exact_limits(bytes)).is_ok());

    let short_bytes = PreparedBertTokenizerJsonLimits::try_new(nz(len - 1), nz(work)).unwrap();
    assert_eq!(
        parse_prepared_bert_tokenizer_json_model(bytes, &short_bytes),
        Err(PreparedBertTokenizerJsonParseError::Exceeded {
            axis: PreparedBertTokenizerJsonLimitAxis::TokenizerJsonBytes,
            actual: len,
            limit: len - 1,
        })
    );
    let short_work = PreparedBertTokenizerJsonLimits::try_new(nz(len), nz(work - 1)).unwrap();
    assert_eq!(
        parse_prepared_bert_tokenizer_json_model(bytes, &short_work),
        Err(PreparedBertTokenizerJsonParseError::Exceeded {
            axis: PreparedBertTokenizerJsonLimitAxis::TokenizerJsonParseWorkBytes,
            actual: work,
            limit: work - 1,
        })
    );

    assert_eq!(
        checked_tokenizer_json_parse_work_bytes(u64::MAX / 2),
        Ok(u64::MAX - 1)
    );
    assert_eq!(
        checked_tokenizer_json_parse_work_bytes(u64::MAX / 2 + 1),
        Err(PreparedBertTokenizerJsonParseError::ArithmeticOverflow(
            PreparedBertTokenizerJsonExpression::TokenizerJsonParseWorkBytes,
        ))
    );
    assert!(PreparedBertTokenizerJsonLimits::try_new_with_platform_max(nz(8), nz(16), 16).is_ok());
    assert_eq!(
        PreparedBertTokenizerJsonLimits::try_new_with_platform_max(nz(17), nz(16), 16),
        Err(
            PreparedBertTokenizerJsonParseError::PlatformUnrepresentable {
                axis: PreparedBertTokenizerJsonLimitAxis::TokenizerJsonBytes,
                value: 17,
            }
        )
    );
    assert_eq!(
        PreparedBertTokenizerJsonLimits::try_new_with_platform_max(nz(8), nz(17), 16),
        Err(
            PreparedBertTokenizerJsonParseError::PlatformUnrepresentable {
                axis: PreparedBertTokenizerJsonLimitAxis::TokenizerJsonParseWorkBytes,
                value: 17,
            }
        )
    );

    fn nested(depth: usize) -> String {
        let mut json = String::from(r#"{"model":{"type":"WordPiece"},"ignored":"#);
        json.extend(std::iter::repeat_n('[', depth));
        json.push('0');
        json.extend(std::iter::repeat_n(']', depth));
        json.push('}');
        json
    }

    let at_limit = nested(MAX_PREPARED_BERT_TOKENIZER_JSON_NESTING - 1);
    assert!(parse(at_limit.as_bytes()).is_ok());
    let over_limit = nested(MAX_PREPARED_BERT_TOKENIZER_JSON_NESTING);
    assert!(matches!(
        parse(over_limit.as_bytes()),
        Err(PreparedBertTokenizerJsonParseError::MalformedJson {
            fault: PreparedBertTokenizerJsonSyntaxFault::NestingLimitExceeded { limit },
            ..
        }) if limit == MAX_PREPARED_BERT_TOKENIZER_JSON_NESTING
    ));
}
