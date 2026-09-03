//! Dormant prepared-BERT `tokenizer.json` decoded model-type contract.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonModel {
    WordPiece,
    Bpe,
    Unigram,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonModelError {
    Missing,
    Empty,
    Unsupported,
}

pub(super) fn classify_prepared_bert_tokenizer_json_model_type(
    model_type: Option<&str>,
) -> Result<PreparedBertTokenizerJsonModel, PreparedBertTokenizerJsonModelError> {
    match model_type {
        None => Err(PreparedBertTokenizerJsonModelError::Missing),
        Some("") => Err(PreparedBertTokenizerJsonModelError::Empty),
        Some("WordPiece") => Ok(PreparedBertTokenizerJsonModel::WordPiece),
        Some("BPE") => Ok(PreparedBertTokenizerJsonModel::Bpe),
        Some("Unigram" | "SentencePieceUnigram") => Ok(PreparedBertTokenizerJsonModel::Unigram),
        Some(_) => Err(PreparedBertTokenizerJsonModelError::Unsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_supported_spellings_map_to_three_canonical_models() {
        use PreparedBertTokenizerJsonModel::{Bpe, Unigram, WordPiece};

        for (model_type, expected) in [
            ("WordPiece", WordPiece),
            ("BPE", Bpe),
            ("Unigram", Unigram),
            ("SentencePieceUnigram", Unigram),
        ] {
            assert_eq!(
                classify_prepared_bert_tokenizer_json_model_type(Some(model_type)),
                Ok(expected)
            );
        }
    }

    #[test]
    fn missing_empty_unknown_and_wrong_case_fail_without_fallback() {
        use PreparedBertTokenizerJsonModelError::{Empty, Missing, Unsupported};

        assert_eq!(
            classify_prepared_bert_tokenizer_json_model_type(None),
            Err(Missing)
        );
        assert_eq!(
            classify_prepared_bert_tokenizer_json_model_type(Some("")),
            Err(Empty)
        );
        for unsupported in ["BertWordPiece", "wordpiece", "bpe", "SentencePiece"] {
            assert_eq!(
                classify_prepared_bert_tokenizer_json_model_type(Some(unsupported)),
                Err(Unsupported)
            );
        }
    }
}

#[cfg(test)]
mod parser_tests {
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
        assert!(
            PreparedBertTokenizerJsonLimits::try_new_with_platform_max(nz(8), nz(16), 16).is_ok()
        );
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
}

use std::num::NonZeroU64;
use std::ops::Range;

pub(super) const MAX_PREPARED_BERT_TOKENIZER_JSON_NESTING: usize = 64;
const MAX_PREPARED_BERT_TOKENIZER_JSON_MODEL_TYPE_BYTES: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonLimitAxis {
    TokenizerJsonBytes,
    TokenizerJsonParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonExpression {
    TokenizerJsonParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonValueKind {
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonExpectedType {
    Object,
    String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonField {
    Model,
    ModelType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonSyntaxFault {
    ExpectedObjectStart,
    ExpectedString,
    ExpectedColon,
    ExpectedValue,
    ExpectedCommaOrEnd,
    TrailingComma,
    TrailingNonWhitespace,
    InvalidLiteral,
    InvalidNumber,
    InvalidStringEscape,
    InvalidUnicodeEscape,
    InvalidUtf8,
    UnescapedControlCharacter,
    UnexpectedEof,
    NestingLimitExceeded { limit: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerJsonParseError {
    InputLengthUnrepresentable,
    Exceeded {
        axis: PreparedBertTokenizerJsonLimitAxis,
        actual: u64,
        limit: u64,
    },
    PlatformUnrepresentable {
        axis: PreparedBertTokenizerJsonLimitAxis,
        value: u64,
    },
    ArithmeticOverflow(PreparedBertTokenizerJsonExpression),
    MalformedJson {
        at: usize,
        fault: PreparedBertTokenizerJsonSyntaxFault,
    },
    DuplicateField {
        field: PreparedBertTokenizerJsonField,
        first_at: usize,
        duplicate_at: usize,
    },
    MissingField {
        field: PreparedBertTokenizerJsonField,
    },
    InvalidFieldType {
        field: PreparedBertTokenizerJsonField,
        expected: PreparedBertTokenizerJsonExpectedType,
        actual: PreparedBertTokenizerJsonValueKind,
        at: usize,
    },
    ModelType(PreparedBertTokenizerJsonModelError),
}

type TokenizerJsonResult<T> = Result<T, PreparedBertTokenizerJsonParseError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertTokenizerJsonLimits {
    max_tokenizer_json_bytes: NonZeroU64,
    max_tokenizer_json_parse_work_bytes: NonZeroU64,
}

impl PreparedBertTokenizerJsonLimits {
    pub(super) fn try_new(
        max_tokenizer_json_bytes: NonZeroU64,
        max_tokenizer_json_parse_work_bytes: NonZeroU64,
    ) -> TokenizerJsonResult<Self> {
        let usize_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
        let isize_max = u64::try_from(isize::MAX).unwrap_or(u64::MAX);
        Self::try_new_with_platform_max(
            max_tokenizer_json_bytes,
            max_tokenizer_json_parse_work_bytes,
            usize_max.min(isize_max),
        )
    }

    fn try_new_with_platform_max(
        max_tokenizer_json_bytes: NonZeroU64,
        max_tokenizer_json_parse_work_bytes: NonZeroU64,
        platform_max: u64,
    ) -> TokenizerJsonResult<Self> {
        for (axis, value) in [
            (
                PreparedBertTokenizerJsonLimitAxis::TokenizerJsonBytes,
                max_tokenizer_json_bytes.get(),
            ),
            (
                PreparedBertTokenizerJsonLimitAxis::TokenizerJsonParseWorkBytes,
                max_tokenizer_json_parse_work_bytes.get(),
            ),
        ] {
            if value > platform_max {
                return Err(
                    PreparedBertTokenizerJsonParseError::PlatformUnrepresentable { axis, value },
                );
            }
        }
        Ok(Self {
            max_tokenizer_json_bytes,
            max_tokenizer_json_parse_work_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ParsedBertTokenizerJsonModelFacts {
    model: PreparedBertTokenizerJsonModel,
    tokenizer_json_bytes: u64,
    logical_tokenizer_json_parse_work_bytes: u64,
}

impl ParsedBertTokenizerJsonModelFacts {
    pub(super) fn model(self) -> PreparedBertTokenizerJsonModel {
        self.model
    }

    pub(super) fn tokenizer_json_bytes(self) -> u64 {
        self.tokenizer_json_bytes
    }

    pub(super) fn logical_tokenizer_json_parse_work_bytes(self) -> u64 {
        self.logical_tokenizer_json_parse_work_bytes
    }
}

fn checked_tokenizer_json_parse_work_bytes(tokenizer_json_bytes: u64) -> TokenizerJsonResult<u64> {
    tokenizer_json_bytes.checked_mul(2).ok_or(
        PreparedBertTokenizerJsonParseError::ArithmeticOverflow(
            PreparedBertTokenizerJsonExpression::TokenizerJsonParseWorkBytes,
        ),
    )
}

pub(super) fn parse_prepared_bert_tokenizer_json_model(
    bytes: &[u8],
    limits: &PreparedBertTokenizerJsonLimits,
) -> TokenizerJsonResult<ParsedBertTokenizerJsonModelFacts> {
    let tokenizer_json_bytes = u64::try_from(bytes.len())
        .map_err(|_| PreparedBertTokenizerJsonParseError::InputLengthUnrepresentable)?;
    if tokenizer_json_bytes > limits.max_tokenizer_json_bytes.get() {
        return Err(PreparedBertTokenizerJsonParseError::Exceeded {
            axis: PreparedBertTokenizerJsonLimitAxis::TokenizerJsonBytes,
            actual: tokenizer_json_bytes,
            limit: limits.max_tokenizer_json_bytes.get(),
        });
    }
    let logical_tokenizer_json_parse_work_bytes =
        checked_tokenizer_json_parse_work_bytes(tokenizer_json_bytes)?;
    if logical_tokenizer_json_parse_work_bytes > limits.max_tokenizer_json_parse_work_bytes.get() {
        return Err(PreparedBertTokenizerJsonParseError::Exceeded {
            axis: PreparedBertTokenizerJsonLimitAxis::TokenizerJsonParseWorkBytes,
            actual: logical_tokenizer_json_parse_work_bytes,
            limit: limits.max_tokenizer_json_parse_work_bytes.get(),
        });
    }

    TokenizerJsonCursor::new(bytes).validate_document()?;
    let model = TokenizerJsonSemanticParser::new(bytes).parse()?;
    Ok(ParsedBertTokenizerJsonModelFacts {
        model,
        tokenizer_json_bytes,
        logical_tokenizer_json_parse_work_bytes,
    })
}

struct TokenizerJsonCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> TokenizerJsonCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn malformed<T>(
        &self,
        at: usize,
        fault: PreparedBertTokenizerJsonSyntaxFault,
    ) -> TokenizerJsonResult<T> {
        Err(PreparedBertTokenizerJsonParseError::MalformedJson { at, fault })
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn validate_document(mut self) -> TokenizerJsonResult<()> {
        self.skip_whitespace();
        if self.bytes.get(self.pos) != Some(&b'{') {
            return self.malformed(
                self.pos,
                PreparedBertTokenizerJsonSyntaxFault::ExpectedObjectStart,
            );
        }
        self.skip_object(1)?;
        self.skip_whitespace();
        if self.pos != self.bytes.len() {
            return self.malformed(
                self.pos,
                PreparedBertTokenizerJsonSyntaxFault::TrailingNonWhitespace,
            );
        }
        Ok(())
    }

    fn skip_value(
        &mut self,
        enclosing_depth: usize,
    ) -> TokenizerJsonResult<PreparedBertTokenizerJsonValueKind> {
        self.skip_whitespace();
        let Some(byte) = self.bytes.get(self.pos).copied() else {
            return self.malformed(
                self.pos,
                PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof,
            );
        };
        match byte {
            b'"' => {
                self.scan_string()?;
                Ok(PreparedBertTokenizerJsonValueKind::String)
            }
            b'{' => {
                self.check_nested_depth(enclosing_depth)?;
                self.skip_object(enclosing_depth + 1)?;
                Ok(PreparedBertTokenizerJsonValueKind::Object)
            }
            b'[' => {
                self.check_nested_depth(enclosing_depth)?;
                self.skip_array(enclosing_depth + 1)?;
                Ok(PreparedBertTokenizerJsonValueKind::Array)
            }
            b't' => {
                self.scan_literal(b"true")?;
                Ok(PreparedBertTokenizerJsonValueKind::Boolean)
            }
            b'f' => {
                self.scan_literal(b"false")?;
                Ok(PreparedBertTokenizerJsonValueKind::Boolean)
            }
            b'n' => {
                self.scan_literal(b"null")?;
                Ok(PreparedBertTokenizerJsonValueKind::Null)
            }
            b'-' | b'0'..=b'9' => {
                self.scan_number()?;
                Ok(PreparedBertTokenizerJsonValueKind::Number)
            }
            b'+' | b'.' => self.malformed(
                self.pos,
                PreparedBertTokenizerJsonSyntaxFault::InvalidNumber,
            ),
            b'a'..=b'z' | b'A'..=b'Z' => self.malformed(
                self.pos,
                PreparedBertTokenizerJsonSyntaxFault::InvalidLiteral,
            ),
            _ => self.malformed(
                self.pos,
                PreparedBertTokenizerJsonSyntaxFault::ExpectedValue,
            ),
        }
    }

    fn check_nested_depth(&self, enclosing_depth: usize) -> TokenizerJsonResult<()> {
        if enclosing_depth == MAX_PREPARED_BERT_TOKENIZER_JSON_NESTING {
            return self.malformed(
                self.pos,
                PreparedBertTokenizerJsonSyntaxFault::NestingLimitExceeded {
                    limit: MAX_PREPARED_BERT_TOKENIZER_JSON_NESTING,
                },
            );
        }
        Ok(())
    }

    fn skip_object(&mut self, depth: usize) -> TokenizerJsonResult<()> {
        self.pos += 1;
        self.skip_whitespace();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.scan_string()?;
            self.skip_whitespace();
            if self.bytes.get(self.pos) != Some(&b':') {
                return self.malformed(
                    self.pos,
                    if self.pos == self.bytes.len() {
                        PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof
                    } else {
                        PreparedBertTokenizerJsonSyntaxFault::ExpectedColon
                    },
                );
            }
            self.pos += 1;
            self.skip_value(depth)?;
            self.skip_whitespace();
            match self.bytes.get(self.pos) {
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(());
                }
                Some(b',') => {
                    self.pos += 1;
                    self.skip_whitespace();
                    if self.bytes.get(self.pos) == Some(&b'}') {
                        return self.malformed(
                            self.pos,
                            PreparedBertTokenizerJsonSyntaxFault::TrailingComma,
                        );
                    }
                }
                None => {
                    return self.malformed(
                        self.pos,
                        PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof,
                    );
                }
                _ => {
                    return self.malformed(
                        self.pos,
                        PreparedBertTokenizerJsonSyntaxFault::ExpectedCommaOrEnd,
                    );
                }
            }
        }
    }

    fn skip_array(&mut self, depth: usize) -> TokenizerJsonResult<()> {
        self.pos += 1;
        self.skip_whitespace();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.skip_value(depth)?;
            self.skip_whitespace();
            match self.bytes.get(self.pos) {
                Some(b']') => {
                    self.pos += 1;
                    return Ok(());
                }
                Some(b',') => {
                    self.pos += 1;
                    self.skip_whitespace();
                    if self.bytes.get(self.pos) == Some(&b']') {
                        return self.malformed(
                            self.pos,
                            PreparedBertTokenizerJsonSyntaxFault::TrailingComma,
                        );
                    }
                }
                None => {
                    return self.malformed(
                        self.pos,
                        PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof,
                    );
                }
                _ => {
                    return self.malformed(
                        self.pos,
                        PreparedBertTokenizerJsonSyntaxFault::ExpectedCommaOrEnd,
                    );
                }
            }
        }
    }

    fn peek_value_kind(&self) -> TokenizerJsonResult<PreparedBertTokenizerJsonValueKind> {
        let mut probe = Self {
            bytes: self.bytes,
            pos: self.pos,
        };
        probe.skip_whitespace();
        let Some(byte) = probe.bytes.get(probe.pos) else {
            return probe.malformed(
                probe.pos,
                PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof,
            );
        };
        Ok(match byte {
            b'n' => PreparedBertTokenizerJsonValueKind::Null,
            b't' | b'f' => PreparedBertTokenizerJsonValueKind::Boolean,
            b'-' | b'0'..=b'9' => PreparedBertTokenizerJsonValueKind::Number,
            b'"' => PreparedBertTokenizerJsonValueKind::String,
            b'[' => PreparedBertTokenizerJsonValueKind::Array,
            b'{' => PreparedBertTokenizerJsonValueKind::Object,
            _ => {
                return probe.malformed(
                    probe.pos,
                    PreparedBertTokenizerJsonSyntaxFault::ExpectedValue,
                );
            }
        })
    }

    fn scan_literal(&mut self, literal: &[u8]) -> TokenizerJsonResult<()> {
        let at = self.pos;
        let end = self.pos.saturating_add(literal.len());
        if self.bytes.get(self.pos..end) != Some(literal) {
            return self.malformed(at, PreparedBertTokenizerJsonSyntaxFault::InvalidLiteral);
        }
        self.pos = end;
        Ok(())
    }

    fn scan_number(&mut self) -> TokenizerJsonResult<()> {
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        match self.bytes.get(self.pos) {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    return self.malformed(
                        self.pos,
                        PreparedBertTokenizerJsonSyntaxFault::InvalidNumber,
                    );
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            None => {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof,
                );
            }
            _ => {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerJsonSyntaxFault::InvalidNumber,
                );
            }
        }

        if self.bytes.get(self.pos) == Some(&b'.') {
            self.pos += 1;
            let digits_at = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == digits_at {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerJsonSyntaxFault::InvalidNumber,
                );
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let digits_at = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == digits_at {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerJsonSyntaxFault::InvalidNumber,
                );
            }
        }
        Ok(())
    }

    fn scan_string(&mut self) -> TokenizerJsonResult<Range<usize>> {
        if self.bytes.get(self.pos) != Some(&b'"') {
            return self.malformed(
                self.pos,
                if self.pos == self.bytes.len() {
                    PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof
                } else {
                    PreparedBertTokenizerJsonSyntaxFault::ExpectedString
                },
            );
        }
        self.pos += 1;
        let start = self.pos;
        loop {
            let Some(byte) = self.bytes.get(self.pos).copied() else {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof,
                );
            };
            match byte {
                b'"' => {
                    let end = self.pos;
                    self.pos += 1;
                    return Ok(start..end);
                }
                b'\\' => self.scan_escape()?,
                0x00..=0x1f => {
                    return self.malformed(
                        self.pos,
                        PreparedBertTokenizerJsonSyntaxFault::UnescapedControlCharacter,
                    );
                }
                0x20..=0x7f => self.pos += 1,
                _ => self.scan_utf8_scalar()?,
            }
        }
    }

    fn scan_escape(&mut self) -> TokenizerJsonResult<()> {
        let slash_at = self.pos;
        self.pos += 1;
        let Some(escape) = self.bytes.get(self.pos).copied() else {
            return self.malformed(
                self.pos,
                PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof,
            );
        };
        match escape {
            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                self.pos += 1;
                Ok(())
            }
            b'u' => {
                self.pos += 1;
                let first = self.scan_hex_quad(slash_at)?;
                if (0xd800..=0xdbff).contains(&first) {
                    if self.bytes.get(self.pos..self.pos.saturating_add(2)) != Some(b"\\u") {
                        return self.malformed(
                            slash_at,
                            PreparedBertTokenizerJsonSyntaxFault::InvalidUnicodeEscape,
                        );
                    }
                    self.pos += 2;
                    let second = self.scan_hex_quad(slash_at)?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return self.malformed(
                            slash_at,
                            PreparedBertTokenizerJsonSyntaxFault::InvalidUnicodeEscape,
                        );
                    }
                } else if (0xdc00..=0xdfff).contains(&first) {
                    return self.malformed(
                        slash_at,
                        PreparedBertTokenizerJsonSyntaxFault::InvalidUnicodeEscape,
                    );
                }
                Ok(())
            }
            _ => self.malformed(
                slash_at,
                PreparedBertTokenizerJsonSyntaxFault::InvalidStringEscape,
            ),
        }
    }

    fn scan_hex_quad(&mut self, at: usize) -> TokenizerJsonResult<u16> {
        let mut value = 0_u16;
        for _ in 0..4 {
            let Some(byte) = self.bytes.get(self.pos).copied() else {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerJsonSyntaxFault::UnexpectedEof,
                );
            };
            let digit = match byte {
                b'0'..=b'9' => u16::from(byte - b'0'),
                b'a'..=b'f' => u16::from(byte - b'a' + 10),
                b'A'..=b'F' => u16::from(byte - b'A' + 10),
                _ => {
                    return self.malformed(
                        at,
                        PreparedBertTokenizerJsonSyntaxFault::InvalidUnicodeEscape,
                    );
                }
            };
            value = value * 16 + digit;
            self.pos += 1;
        }
        Ok(value)
    }

    fn scan_utf8_scalar(&mut self) -> TokenizerJsonResult<()> {
        let at = self.pos;
        let first = self.bytes[self.pos];
        let width = match first {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => {
                return self.malformed(at, PreparedBertTokenizerJsonSyntaxFault::InvalidUtf8);
            }
        };
        let Some(sequence) = self.bytes.get(self.pos..self.pos.saturating_add(width)) else {
            return self.malformed(at, PreparedBertTokenizerJsonSyntaxFault::InvalidUtf8);
        };
        if sequence[1..]
            .iter()
            .any(|byte| !matches!(byte, 0x80..=0xbf))
            || (width == 3
                && ((first == 0xe0 && sequence[1] < 0xa0)
                    || (first == 0xed && sequence[1] >= 0xa0)))
            || (width == 4
                && ((first == 0xf0 && sequence[1] < 0x90)
                    || (first == 0xf4 && sequence[1] >= 0x90)))
        {
            return self.malformed(at, PreparedBertTokenizerJsonSyntaxFault::InvalidUtf8);
        }
        self.pos += width;
        Ok(())
    }
}

struct TokenizerJsonSemanticParser<'a> {
    cursor: TokenizerJsonCursor<'a>,
}

impl<'a> TokenizerJsonSemanticParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: TokenizerJsonCursor::new(bytes),
        }
    }

    fn parse(mut self) -> TokenizerJsonResult<PreparedBertTokenizerJsonModel> {
        self.cursor.skip_whitespace();
        self.cursor.pos += 1;
        self.cursor.skip_whitespace();
        let mut model_at = None;
        let mut model = None;
        if self.cursor.bytes.get(self.cursor.pos) != Some(&b'}') {
            loop {
                self.cursor.skip_whitespace();
                let key_at = self.cursor.pos;
                let key = self.cursor.scan_string()?;
                self.cursor.skip_whitespace();
                self.cursor.pos += 1;
                self.cursor.skip_whitespace();
                let value_at = self.cursor.pos;

                if decoded_tokenizer_json_key_equals(&self.cursor.bytes[key], b"model") {
                    if let Some(first_at) = model_at {
                        return Err(PreparedBertTokenizerJsonParseError::DuplicateField {
                            field: PreparedBertTokenizerJsonField::Model,
                            first_at,
                            duplicate_at: key_at,
                        });
                    }
                    model_at = Some(key_at);
                    let actual = self.cursor.peek_value_kind()?;
                    if actual != PreparedBertTokenizerJsonValueKind::Object {
                        return Err(PreparedBertTokenizerJsonParseError::InvalidFieldType {
                            field: PreparedBertTokenizerJsonField::Model,
                            expected: PreparedBertTokenizerJsonExpectedType::Object,
                            actual,
                            at: value_at,
                        });
                    }
                    model = Some(self.parse_model_object()?);
                } else {
                    self.cursor.skip_value(1)?;
                }

                self.cursor.skip_whitespace();
                match self.cursor.bytes.get(self.cursor.pos) {
                    Some(b',') => self.cursor.pos += 1,
                    Some(b'}') => {
                        self.cursor.pos += 1;
                        break;
                    }
                    _ => {
                        return self.cursor.malformed(
                            self.cursor.pos,
                            PreparedBertTokenizerJsonSyntaxFault::ExpectedCommaOrEnd,
                        );
                    }
                }
            }
        }
        model.ok_or(PreparedBertTokenizerJsonParseError::MissingField {
            field: PreparedBertTokenizerJsonField::Model,
        })
    }

    fn parse_model_object(&mut self) -> TokenizerJsonResult<PreparedBertTokenizerJsonModel> {
        self.cursor.pos += 1;
        self.cursor.skip_whitespace();
        let mut type_at = None;
        let mut model_type = None;
        if self.cursor.bytes.get(self.cursor.pos) != Some(&b'}') {
            loop {
                self.cursor.skip_whitespace();
                let key_at = self.cursor.pos;
                let key = self.cursor.scan_string()?;
                self.cursor.skip_whitespace();
                self.cursor.pos += 1;
                self.cursor.skip_whitespace();
                let value_at = self.cursor.pos;

                if decoded_tokenizer_json_key_equals(&self.cursor.bytes[key], b"type") {
                    if let Some(first_at) = type_at {
                        return Err(PreparedBertTokenizerJsonParseError::DuplicateField {
                            field: PreparedBertTokenizerJsonField::ModelType,
                            first_at,
                            duplicate_at: key_at,
                        });
                    }
                    type_at = Some(key_at);
                    let actual = self.cursor.peek_value_kind()?;
                    if actual != PreparedBertTokenizerJsonValueKind::String {
                        return Err(PreparedBertTokenizerJsonParseError::InvalidFieldType {
                            field: PreparedBertTokenizerJsonField::ModelType,
                            expected: PreparedBertTokenizerJsonExpectedType::String,
                            actual,
                            at: value_at,
                        });
                    }
                    model_type = Some(self.cursor.scan_string()?);
                } else {
                    self.cursor.skip_value(2)?;
                }

                self.cursor.skip_whitespace();
                match self.cursor.bytes.get(self.cursor.pos) {
                    Some(b',') => self.cursor.pos += 1,
                    Some(b'}') => {
                        self.cursor.pos += 1;
                        break;
                    }
                    _ => {
                        return self.cursor.malformed(
                            self.cursor.pos,
                            PreparedBertTokenizerJsonSyntaxFault::ExpectedCommaOrEnd,
                        );
                    }
                }
            }
        }
        match model_type {
            Some(range) => classify_encoded_tokenizer_json_model_type(&self.cursor.bytes[range]),
            None => classify_prepared_bert_tokenizer_json_model_type(None),
        }
        .map_err(PreparedBertTokenizerJsonParseError::ModelType)
    }
}

fn decoded_tokenizer_json_key_equals(encoded: &[u8], target: &[u8]) -> bool {
    let mut source = 0_usize;
    let mut expected = 0_usize;
    while source < encoded.len() {
        let decoded = if encoded[source] == b'\\' {
            source += 1;
            match encoded[source] {
                b'"' | b'\\' | b'/' => {
                    let value = encoded[source];
                    source += 1;
                    value
                }
                b'b' => {
                    source += 1;
                    0x08
                }
                b'f' => {
                    source += 1;
                    0x0c
                }
                b'n' => {
                    source += 1;
                    b'\n'
                }
                b'r' => {
                    source += 1;
                    b'\r'
                }
                b't' => {
                    source += 1;
                    b'\t'
                }
                b'u' => {
                    source += 1;
                    let value = decode_tokenizer_json_hex_quad(encoded, &mut source);
                    if value > u16::from(u8::MAX) || (0xd800..=0xdfff).contains(&value) {
                        return false;
                    }
                    u8::try_from(value).unwrap_or(u8::MAX)
                }
                _ => return false,
            }
        } else if encoded[source].is_ascii() {
            let value = encoded[source];
            source += 1;
            value
        } else {
            return false;
        };
        if target.get(expected) != Some(&decoded) {
            return false;
        }
        expected += 1;
    }
    expected == target.len()
}

fn classify_encoded_tokenizer_json_model_type(
    encoded: &[u8],
) -> Result<PreparedBertTokenizerJsonModel, PreparedBertTokenizerJsonModelError> {
    let mut decoded = [0_u8; MAX_PREPARED_BERT_TOKENIZER_JSON_MODEL_TYPE_BYTES];
    let mut decoded_len = 0_usize;
    let mut source = 0_usize;
    let mut unsupported = false;
    while source < encoded.len() {
        let value = if encoded[source] == b'\\' {
            source += 1;
            match encoded[source] {
                b'"' | b'\\' | b'/' => {
                    let value = Some(encoded[source]);
                    source += 1;
                    value
                }
                b'b' => {
                    source += 1;
                    Some(0x08)
                }
                b'f' => {
                    source += 1;
                    Some(0x0c)
                }
                b'n' => {
                    source += 1;
                    Some(b'\n')
                }
                b'r' => {
                    source += 1;
                    Some(b'\r')
                }
                b't' => {
                    source += 1;
                    Some(b'\t')
                }
                b'u' => {
                    source += 1;
                    let scalar = decode_tokenizer_json_hex_quad(encoded, &mut source);
                    if (0xd800..=0xdbff).contains(&scalar) {
                        source += 2;
                        decode_tokenizer_json_hex_quad(encoded, &mut source);
                        unsupported = true;
                        None
                    } else if scalar <= 0x7f {
                        Some(u8::try_from(scalar).unwrap_or(0x7f))
                    } else {
                        unsupported = true;
                        None
                    }
                }
                _ => {
                    unsupported = true;
                    source += 1;
                    None
                }
            }
        } else if encoded[source].is_ascii() {
            let value = Some(encoded[source]);
            source += 1;
            value
        } else {
            unsupported = true;
            source += 1;
            None
        };
        if let Some(value) = value {
            if decoded_len == decoded.len() {
                unsupported = true;
            } else {
                decoded[decoded_len] = value;
                decoded_len += 1;
            }
        }
    }

    if unsupported {
        return classify_prepared_bert_tokenizer_json_model_type(Some("\0"));
    }
    match std::str::from_utf8(&decoded[..decoded_len]) {
        Ok(model_type) => classify_prepared_bert_tokenizer_json_model_type(Some(model_type)),
        Err(_) => classify_prepared_bert_tokenizer_json_model_type(Some("\0")),
    }
}

fn decode_tokenizer_json_hex_quad(bytes: &[u8], position: &mut usize) -> u16 {
    let mut value = 0_u16;
    for _ in 0..4 {
        let byte = bytes[*position];
        let digit = match byte {
            b'0'..=b'9' => u16::from(byte - b'0'),
            b'a'..=b'f' => u16::from(byte - b'a' + 10),
            _ => u16::from(byte - b'A' + 10),
        };
        value = value * 16 + digit;
        *position += 1;
    }
    value
}
