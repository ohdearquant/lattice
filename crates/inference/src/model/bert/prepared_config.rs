//! Dormant checked parsing contract for prepared BERT `config.json` bytes.

use super::prepared_facts::RawBertConfigFacts;
use super::prepared_sequence_cap::{
    PreparedBertSequenceCapKey, RawPreparedBertSequenceCapCandidate,
};
use std::num::NonZeroU64;
use std::ops::Range;

pub(super) const MAX_PREPARED_BERT_CONFIG_NESTING: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertConfigField {
    VocabSize,
    HiddenSize,
    NumHiddenLayers,
    NumAttentionHeads,
    IntermediateSize,
    MaxPositionEmbeddings,
    TypeVocabSize,
    LayerNormEps,
    ModelMaxLength,
}

impl PreparedBertConfigField {
    const ALL: [Self; 9] = [
        Self::VocabSize,
        Self::HiddenSize,
        Self::NumHiddenLayers,
        Self::NumAttentionHeads,
        Self::IntermediateSize,
        Self::MaxPositionEmbeddings,
        Self::TypeVocabSize,
        Self::LayerNormEps,
        Self::ModelMaxLength,
    ];

    const REQUIRED: [Self; 8] = [
        Self::VocabSize,
        Self::HiddenSize,
        Self::NumHiddenLayers,
        Self::NumAttentionHeads,
        Self::IntermediateSize,
        Self::MaxPositionEmbeddings,
        Self::TypeVocabSize,
        Self::LayerNormEps,
    ];

    fn index(self) -> usize {
        match self {
            Self::VocabSize => 0,
            Self::HiddenSize => 1,
            Self::NumHiddenLayers => 2,
            Self::NumAttentionHeads => 3,
            Self::IntermediateSize => 4,
            Self::MaxPositionEmbeddings => 5,
            Self::TypeVocabSize => 6,
            Self::LayerNormEps => 7,
            Self::ModelMaxLength => 8,
        }
    }

    fn name(self) -> &'static [u8] {
        match self {
            Self::VocabSize => b"vocab_size",
            Self::HiddenSize => b"hidden_size",
            Self::NumHiddenLayers => b"num_hidden_layers",
            Self::NumAttentionHeads => b"num_attention_heads",
            Self::IntermediateSize => b"intermediate_size",
            Self::MaxPositionEmbeddings => b"max_position_embeddings",
            Self::TypeVocabSize => b"type_vocab_size",
            Self::LayerNormEps => b"layer_norm_eps",
            Self::ModelMaxLength => b"model_max_length",
        }
    }

    fn expected_type(self) -> PreparedBertConfigExpectedType {
        if self == Self::LayerNormEps {
            PreparedBertConfigExpectedType::Number
        } else {
            PreparedBertConfigExpectedType::UnsignedInteger
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertConfigLimitAxis {
    ConfigBytes,
    ConfigParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertConfigExpression {
    ConfigParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertConfigValueKind {
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertConfigExpectedType {
    UnsignedInteger,
    Number,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertConfigUnsignedFault {
    Zero,
    Negative,
    Fractional,
    Exponent,
    Overflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertConfigSequenceCandidateError {
    InvalidType {
        key: PreparedBertSequenceCapKey,
        actual: PreparedBertConfigValueKind,
        at: usize,
    },
    InvalidPositiveInteger {
        key: PreparedBertSequenceCapKey,
        fault: PreparedBertConfigUnsignedFault,
        at: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertConfigSyntaxFault {
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
pub(super) enum PreparedBertConfigError {
    InputLengthUnrepresentable,
    Exceeded {
        axis: PreparedBertConfigLimitAxis,
        actual: u64,
        limit: u64,
    },
    PlatformUnrepresentable {
        axis: PreparedBertConfigLimitAxis,
        value: u64,
    },
    ArithmeticOverflow(PreparedBertConfigExpression),
    MalformedJson {
        at: usize,
        fault: PreparedBertConfigSyntaxFault,
    },
    DuplicateField {
        field: PreparedBertConfigField,
        first_at: usize,
        duplicate_at: usize,
    },
    MissingField {
        field: PreparedBertConfigField,
    },
    InvalidFieldType {
        field: PreparedBertConfigField,
        expected: PreparedBertConfigExpectedType,
        actual: PreparedBertConfigValueKind,
        at: usize,
    },
    InvalidUnsignedInteger {
        field: PreparedBertConfigField,
        fault: PreparedBertConfigUnsignedFault,
        at: usize,
    },
    InvalidFloat {
        field: PreparedBertConfigField,
        at: usize,
    },
}

type ConfigResult<T> = Result<T, PreparedBertConfigError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertConfigLimits {
    max_config_bytes: NonZeroU64,
    max_config_parse_work_bytes: NonZeroU64,
}

impl PreparedBertConfigLimits {
    pub(super) fn try_new(
        max_config_bytes: NonZeroU64,
        max_config_parse_work_bytes: NonZeroU64,
    ) -> ConfigResult<Self> {
        let usize_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
        let isize_max = u64::try_from(isize::MAX).unwrap_or(u64::MAX);
        Self::try_new_with_platform_max(
            max_config_bytes,
            max_config_parse_work_bytes,
            usize_max.min(isize_max),
        )
    }

    fn try_new_with_platform_max(
        max_config_bytes: NonZeroU64,
        max_config_parse_work_bytes: NonZeroU64,
        platform_max: u64,
    ) -> ConfigResult<Self> {
        for (axis, value) in [
            (
                PreparedBertConfigLimitAxis::ConfigBytes,
                max_config_bytes.get(),
            ),
            (
                PreparedBertConfigLimitAxis::ConfigParseWorkBytes,
                max_config_parse_work_bytes.get(),
            ),
        ] {
            if value > platform_max {
                return Err(PreparedBertConfigError::PlatformUnrepresentable { axis, value });
            }
        }
        Ok(Self {
            max_config_bytes,
            max_config_parse_work_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ParsedBertConfigFacts {
    raw: RawBertConfigFacts,
    sequence_cap_candidate:
        Result<RawPreparedBertSequenceCapCandidate, PreparedBertConfigSequenceCandidateError>,
    config_bytes: u64,
    logical_config_parse_work_bytes: u64,
}

impl ParsedBertConfigFacts {
    pub(super) fn raw(self) -> RawBertConfigFacts {
        self.raw
    }

    pub(super) fn sequence_cap_candidate(
        self,
    ) -> Result<RawPreparedBertSequenceCapCandidate, PreparedBertConfigSequenceCandidateError> {
        self.sequence_cap_candidate
    }

    pub(super) fn config_bytes(self) -> u64 {
        self.config_bytes
    }

    pub(super) fn logical_config_parse_work_bytes(self) -> u64 {
        self.logical_config_parse_work_bytes
    }
}

fn checked_config_parse_work_bytes(config_bytes: u64) -> ConfigResult<u64> {
    config_bytes
        .checked_mul(2)
        .ok_or(PreparedBertConfigError::ArithmeticOverflow(
            PreparedBertConfigExpression::ConfigParseWorkBytes,
        ))
}

pub(super) fn parse_prepared_bert_config_json(
    bytes: &[u8],
    limits: &PreparedBertConfigLimits,
) -> ConfigResult<ParsedBertConfigFacts> {
    let config_bytes = u64::try_from(bytes.len())
        .map_err(|_| PreparedBertConfigError::InputLengthUnrepresentable)?;
    if config_bytes > limits.max_config_bytes.get() {
        return Err(PreparedBertConfigError::Exceeded {
            axis: PreparedBertConfigLimitAxis::ConfigBytes,
            actual: config_bytes,
            limit: limits.max_config_bytes.get(),
        });
    }
    let logical_config_parse_work_bytes = checked_config_parse_work_bytes(config_bytes)?;
    if logical_config_parse_work_bytes > limits.max_config_parse_work_bytes.get() {
        return Err(PreparedBertConfigError::Exceeded {
            axis: PreparedBertConfigLimitAxis::ConfigParseWorkBytes,
            actual: logical_config_parse_work_bytes,
            limit: limits.max_config_parse_work_bytes.get(),
        });
    }

    JsonCursor::new(bytes).validate_document()?;
    let (raw, sequence_cap_candidate) = SemanticParser::new(bytes).parse()?;
    Ok(ParsedBertConfigFacts {
        raw,
        sequence_cap_candidate,
        config_bytes,
        logical_config_parse_work_bytes,
    })
}

#[derive(Clone, Copy)]
enum ContainerState {
    ArrayFirstOrEnd,
    ArrayValue,
    ArrayCommaOrEnd,
    ObjectFirstKeyOrEnd,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectCommaOrEnd,
}

#[derive(Clone)]
struct NumberToken {
    range: Range<usize>,
    negative: bool,
    fractional: bool,
    exponent: bool,
}

struct JsonCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn malformed<T>(&self, at: usize, fault: PreparedBertConfigSyntaxFault) -> ConfigResult<T> {
        Err(PreparedBertConfigError::MalformedJson { at, fault })
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn validate_document(mut self) -> ConfigResult<()> {
        self.skip_whitespace();
        if self.bytes.get(self.pos) != Some(&b'{') {
            return self.malformed(self.pos, PreparedBertConfigSyntaxFault::ExpectedObjectStart);
        }
        self.skip_value()?;
        self.skip_whitespace();
        if self.pos != self.bytes.len() {
            return self.malformed(
                self.pos,
                PreparedBertConfigSyntaxFault::TrailingNonWhitespace,
            );
        }
        Ok(())
    }

    fn skip_value(&mut self) -> ConfigResult<PreparedBertConfigValueKind> {
        self.skip_whitespace();
        let kind = self.begin_value()?;
        let first_state = match kind {
            PreparedBertConfigValueKind::Array => Some(ContainerState::ArrayFirstOrEnd),
            PreparedBertConfigValueKind::Object => Some(ContainerState::ObjectFirstKeyOrEnd),
            _ => None,
        };
        let Some(first_state) = first_state else {
            return Ok(kind);
        };

        let mut stack = [ContainerState::ArrayFirstOrEnd; MAX_PREPARED_BERT_CONFIG_NESTING];
        stack[0] = first_state;
        let mut depth = 1_usize;
        loop {
            let state = stack[depth - 1];
            match state {
                ContainerState::ArrayFirstOrEnd => {
                    self.skip_whitespace();
                    if self.bytes.get(self.pos) == Some(&b']') {
                        self.pos += 1;
                        depth -= 1;
                    } else {
                        stack[depth - 1] = ContainerState::ArrayCommaOrEnd;
                        self.begin_nested_value(&mut stack, &mut depth)?;
                    }
                }
                ContainerState::ArrayValue => {
                    self.skip_whitespace();
                    if self.bytes.get(self.pos) == Some(&b']') {
                        return self
                            .malformed(self.pos, PreparedBertConfigSyntaxFault::TrailingComma);
                    }
                    stack[depth - 1] = ContainerState::ArrayCommaOrEnd;
                    self.begin_nested_value(&mut stack, &mut depth)?;
                }
                ContainerState::ArrayCommaOrEnd => {
                    self.skip_whitespace();
                    match self.bytes.get(self.pos) {
                        Some(b',') => {
                            self.pos += 1;
                            stack[depth - 1] = ContainerState::ArrayValue;
                        }
                        Some(b']') => {
                            self.pos += 1;
                            depth -= 1;
                        }
                        None => {
                            return self
                                .malformed(self.pos, PreparedBertConfigSyntaxFault::UnexpectedEof);
                        }
                        _ => {
                            return self.malformed(
                                self.pos,
                                PreparedBertConfigSyntaxFault::ExpectedCommaOrEnd,
                            );
                        }
                    }
                }
                ContainerState::ObjectFirstKeyOrEnd => {
                    self.skip_whitespace();
                    if self.bytes.get(self.pos) == Some(&b'}') {
                        self.pos += 1;
                        depth -= 1;
                    } else {
                        self.scan_string()?;
                        stack[depth - 1] = ContainerState::ObjectColon;
                    }
                }
                ContainerState::ObjectKey => {
                    self.skip_whitespace();
                    if self.bytes.get(self.pos) == Some(&b'}') {
                        return self
                            .malformed(self.pos, PreparedBertConfigSyntaxFault::TrailingComma);
                    }
                    self.scan_string()?;
                    stack[depth - 1] = ContainerState::ObjectColon;
                }
                ContainerState::ObjectColon => {
                    self.skip_whitespace();
                    if self.bytes.get(self.pos) != Some(&b':') {
                        return self.malformed(
                            self.pos,
                            if self.pos == self.bytes.len() {
                                PreparedBertConfigSyntaxFault::UnexpectedEof
                            } else {
                                PreparedBertConfigSyntaxFault::ExpectedColon
                            },
                        );
                    }
                    self.pos += 1;
                    stack[depth - 1] = ContainerState::ObjectValue;
                }
                ContainerState::ObjectValue => {
                    stack[depth - 1] = ContainerState::ObjectCommaOrEnd;
                    self.begin_nested_value(&mut stack, &mut depth)?;
                }
                ContainerState::ObjectCommaOrEnd => {
                    self.skip_whitespace();
                    match self.bytes.get(self.pos) {
                        Some(b',') => {
                            self.pos += 1;
                            stack[depth - 1] = ContainerState::ObjectKey;
                        }
                        Some(b'}') => {
                            self.pos += 1;
                            depth -= 1;
                        }
                        None => {
                            return self
                                .malformed(self.pos, PreparedBertConfigSyntaxFault::UnexpectedEof);
                        }
                        _ => {
                            return self.malformed(
                                self.pos,
                                PreparedBertConfigSyntaxFault::ExpectedCommaOrEnd,
                            );
                        }
                    }
                }
            }
            if depth == 0 {
                return Ok(kind);
            }
        }
    }

    fn begin_nested_value(
        &mut self,
        stack: &mut [ContainerState; MAX_PREPARED_BERT_CONFIG_NESTING],
        depth: &mut usize,
    ) -> ConfigResult<()> {
        self.skip_whitespace();
        let at = self.pos;
        let state = match self.begin_value()? {
            PreparedBertConfigValueKind::Array => Some(ContainerState::ArrayFirstOrEnd),
            PreparedBertConfigValueKind::Object => Some(ContainerState::ObjectFirstKeyOrEnd),
            _ => None,
        };
        if let Some(state) = state {
            if *depth == MAX_PREPARED_BERT_CONFIG_NESTING {
                return self.malformed(
                    at,
                    PreparedBertConfigSyntaxFault::NestingLimitExceeded {
                        limit: MAX_PREPARED_BERT_CONFIG_NESTING,
                    },
                );
            }
            stack[*depth] = state;
            *depth += 1;
        }
        Ok(())
    }

    fn begin_value(&mut self) -> ConfigResult<PreparedBertConfigValueKind> {
        self.skip_whitespace();
        let Some(byte) = self.bytes.get(self.pos).copied() else {
            return self.malformed(self.pos, PreparedBertConfigSyntaxFault::UnexpectedEof);
        };
        match byte {
            b'"' => {
                self.scan_string()?;
                Ok(PreparedBertConfigValueKind::String)
            }
            b'{' => {
                self.pos += 1;
                Ok(PreparedBertConfigValueKind::Object)
            }
            b'[' => {
                self.pos += 1;
                Ok(PreparedBertConfigValueKind::Array)
            }
            b't' => {
                self.scan_literal(b"true")?;
                Ok(PreparedBertConfigValueKind::Boolean)
            }
            b'f' => {
                self.scan_literal(b"false")?;
                Ok(PreparedBertConfigValueKind::Boolean)
            }
            b'n' => {
                self.scan_literal(b"null")?;
                Ok(PreparedBertConfigValueKind::Null)
            }
            b'-' | b'0'..=b'9' => {
                self.scan_number()?;
                Ok(PreparedBertConfigValueKind::Number)
            }
            b'+' | b'.' => self.malformed(self.pos, PreparedBertConfigSyntaxFault::InvalidNumber),
            b'a'..=b'z' | b'A'..=b'Z' => {
                self.malformed(self.pos, PreparedBertConfigSyntaxFault::InvalidLiteral)
            }
            _ => self.malformed(self.pos, PreparedBertConfigSyntaxFault::ExpectedValue),
        }
    }

    fn scan_literal(&mut self, literal: &[u8]) -> ConfigResult<()> {
        let at = self.pos;
        let end = self.pos.saturating_add(literal.len());
        if self.bytes.get(self.pos..end) != Some(literal) {
            return self.malformed(at, PreparedBertConfigSyntaxFault::InvalidLiteral);
        }
        self.pos = end;
        Ok(())
    }

    fn scan_number(&mut self) -> ConfigResult<NumberToken> {
        let start = self.pos;
        let negative = self.bytes.get(self.pos) == Some(&b'-');
        if negative {
            self.pos += 1;
        }
        match self.bytes.get(self.pos) {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    return self.malformed(self.pos, PreparedBertConfigSyntaxFault::InvalidNumber);
                }
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            None => {
                return self.malformed(self.pos, PreparedBertConfigSyntaxFault::UnexpectedEof);
            }
            _ => {
                return self.malformed(self.pos, PreparedBertConfigSyntaxFault::InvalidNumber);
            }
        }

        let mut fractional = false;
        if self.bytes.get(self.pos) == Some(&b'.') {
            fractional = true;
            self.pos += 1;
            let digits_at = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == digits_at {
                return self.malformed(self.pos, PreparedBertConfigSyntaxFault::InvalidNumber);
            }
        }

        let mut exponent = false;
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            exponent = true;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let digits_at = self.pos;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if self.pos == digits_at {
                return self.malformed(self.pos, PreparedBertConfigSyntaxFault::InvalidNumber);
            }
        }
        Ok(NumberToken {
            range: start..self.pos,
            negative,
            fractional,
            exponent,
        })
    }

    fn scan_string(&mut self) -> ConfigResult<Range<usize>> {
        if self.bytes.get(self.pos) != Some(&b'"') {
            return self.malformed(
                self.pos,
                if self.pos == self.bytes.len() {
                    PreparedBertConfigSyntaxFault::UnexpectedEof
                } else {
                    PreparedBertConfigSyntaxFault::ExpectedString
                },
            );
        }
        self.pos += 1;
        let start = self.pos;
        loop {
            let Some(byte) = self.bytes.get(self.pos).copied() else {
                return self.malformed(self.pos, PreparedBertConfigSyntaxFault::UnexpectedEof);
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
                        PreparedBertConfigSyntaxFault::UnescapedControlCharacter,
                    );
                }
                0x20..=0x7f => self.pos += 1,
                _ => self.scan_utf8_scalar()?,
            }
        }
    }

    fn scan_escape(&mut self) -> ConfigResult<()> {
        let slash_at = self.pos;
        self.pos += 1;
        let Some(escape) = self.bytes.get(self.pos).copied() else {
            return self.malformed(self.pos, PreparedBertConfigSyntaxFault::UnexpectedEof);
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
                            PreparedBertConfigSyntaxFault::InvalidUnicodeEscape,
                        );
                    }
                    self.pos += 2;
                    let second = self.scan_hex_quad(slash_at)?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return self.malformed(
                            slash_at,
                            PreparedBertConfigSyntaxFault::InvalidUnicodeEscape,
                        );
                    }
                } else if (0xdc00..=0xdfff).contains(&first) {
                    return self.malformed(
                        slash_at,
                        PreparedBertConfigSyntaxFault::InvalidUnicodeEscape,
                    );
                }
                Ok(())
            }
            _ => self.malformed(slash_at, PreparedBertConfigSyntaxFault::InvalidStringEscape),
        }
    }

    fn scan_hex_quad(&mut self, at: usize) -> ConfigResult<u16> {
        let mut value = 0_u16;
        for _ in 0..4 {
            let Some(byte) = self.bytes.get(self.pos).copied() else {
                return self.malformed(self.pos, PreparedBertConfigSyntaxFault::UnexpectedEof);
            };
            let digit = match byte {
                b'0'..=b'9' => u16::from(byte - b'0'),
                b'a'..=b'f' => u16::from(byte - b'a' + 10),
                b'A'..=b'F' => u16::from(byte - b'A' + 10),
                _ => {
                    return self.malformed(at, PreparedBertConfigSyntaxFault::InvalidUnicodeEscape);
                }
            };
            value = value * 16 + digit;
            self.pos += 1;
        }
        Ok(value)
    }

    fn scan_utf8_scalar(&mut self) -> ConfigResult<()> {
        let at = self.pos;
        let first = self.bytes[self.pos];
        let width = match first {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => {
                return self.malformed(at, PreparedBertConfigSyntaxFault::InvalidUtf8);
            }
        };
        let Some(sequence) = self.bytes.get(self.pos..self.pos.saturating_add(width)) else {
            return self.malformed(at, PreparedBertConfigSyntaxFault::InvalidUtf8);
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
            return self.malformed(at, PreparedBertConfigSyntaxFault::InvalidUtf8);
        }
        self.pos += width;
        Ok(())
    }
}

struct SemanticParser<'a> {
    cursor: JsonCursor<'a>,
    seen_at: [Option<usize>; 9],
    unsigned: [u64; 7],
    epsilon: f64,
    model_max_length_candidate: Option<
        Result<RawPreparedBertSequenceCapCandidate, PreparedBertConfigSequenceCandidateError>,
    >,
}

impl<'a> SemanticParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: JsonCursor::new(bytes),
            seen_at: [None; 9],
            unsigned: [0; 7],
            epsilon: 0.0,
            model_max_length_candidate: None,
        }
    }

    fn parse(
        mut self,
    ) -> ConfigResult<(
        RawBertConfigFacts,
        Result<RawPreparedBertSequenceCapCandidate, PreparedBertConfigSequenceCandidateError>,
    )> {
        self.cursor.skip_whitespace();
        self.cursor.pos += 1;
        self.cursor.skip_whitespace();
        if self.cursor.bytes.get(self.cursor.pos) == Some(&b'}') {
            self.cursor.pos += 1;
        } else {
            loop {
                self.cursor.skip_whitespace();
                let key_at = self.cursor.pos;
                let key = self.cursor.scan_string()?;
                let field = classify_field(self.cursor.bytes, key);
                self.cursor.skip_whitespace();
                self.cursor.pos += 1;
                self.cursor.skip_whitespace();
                let value_at = self.cursor.pos;

                if let Some(field) = field {
                    let index = field.index();
                    if let Some(first_at) = self.seen_at[index] {
                        return Err(PreparedBertConfigError::DuplicateField {
                            field,
                            first_at,
                            duplicate_at: key_at,
                        });
                    }
                    self.seen_at[index] = Some(key_at);
                    let actual = self.cursor.peek_value_kind()?;
                    if field == PreparedBertConfigField::ModelMaxLength {
                        if actual == PreparedBertConfigValueKind::Number {
                            let token = self.cursor.scan_number()?;
                            self.model_max_length_candidate =
                                Some(parse_model_max_length_candidate(
                                    self.cursor.bytes,
                                    &token,
                                    value_at,
                                ));
                        } else {
                            self.cursor.skip_value()?;
                            self.model_max_length_candidate =
                                Some(Err(PreparedBertConfigSequenceCandidateError::InvalidType {
                                    key: PreparedBertSequenceCapKey::ModelMaxLength,
                                    actual,
                                    at: value_at,
                                }));
                        }
                    } else {
                        if actual != PreparedBertConfigValueKind::Number {
                            return Err(PreparedBertConfigError::InvalidFieldType {
                                field,
                                expected: field.expected_type(),
                                actual,
                                at: value_at,
                            });
                        }
                        let token = self.cursor.scan_number()?;
                        if field == PreparedBertConfigField::LayerNormEps {
                            self.epsilon =
                                parse_float(self.cursor.bytes, token.range, field, value_at)?;
                        } else {
                            let value = parse_unsigned(self.cursor.bytes, &token, field, value_at)?;
                            self.unsigned[index] = value;
                        }
                    }
                } else {
                    self.cursor.skip_value()?;
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
                            PreparedBertConfigSyntaxFault::ExpectedCommaOrEnd,
                        );
                    }
                }
            }
        }

        for field in PreparedBertConfigField::REQUIRED {
            if self.seen_at[field.index()].is_none() {
                return Err(PreparedBertConfigError::MissingField { field });
            }
        }
        let sequence_cap_candidate = match self.model_max_length_candidate {
            Some(candidate) => candidate,
            None => Ok(RawPreparedBertSequenceCapCandidate::new(
                PreparedBertSequenceCapKey::MaxPositionEmbeddings,
                self.unsigned[PreparedBertConfigField::MaxPositionEmbeddings.index()],
            )),
        };
        Ok((
            RawBertConfigFacts::new(
                self.unsigned[0],
                self.unsigned[1],
                self.unsigned[2],
                self.unsigned[3],
                self.unsigned[4],
                self.unsigned[5],
                self.unsigned[6],
                self.epsilon,
            ),
            sequence_cap_candidate,
        ))
    }
}

impl JsonCursor<'_> {
    fn peek_value_kind(&self) -> ConfigResult<PreparedBertConfigValueKind> {
        let mut probe = JsonCursor {
            bytes: self.bytes,
            pos: self.pos,
        };
        probe.skip_whitespace();
        let Some(byte) = probe.bytes.get(probe.pos) else {
            return probe.malformed(probe.pos, PreparedBertConfigSyntaxFault::UnexpectedEof);
        };
        Ok(match byte {
            b'n' => PreparedBertConfigValueKind::Null,
            b't' | b'f' => PreparedBertConfigValueKind::Boolean,
            b'-' | b'0'..=b'9' => PreparedBertConfigValueKind::Number,
            b'"' => PreparedBertConfigValueKind::String,
            b'[' => PreparedBertConfigValueKind::Array,
            b'{' => PreparedBertConfigValueKind::Object,
            _ => {
                return probe.malformed(probe.pos, PreparedBertConfigSyntaxFault::ExpectedValue);
            }
        })
    }
}

fn parse_unsigned(
    bytes: &[u8],
    token: &NumberToken,
    field: PreparedBertConfigField,
    at: usize,
) -> ConfigResult<u64> {
    parse_unsigned_value(bytes, token)
        .map_err(|fault| PreparedBertConfigError::InvalidUnsignedInteger { field, fault, at })
}

fn parse_model_max_length_candidate(
    bytes: &[u8],
    token: &NumberToken,
    at: usize,
) -> Result<RawPreparedBertSequenceCapCandidate, PreparedBertConfigSequenceCandidateError> {
    let key = PreparedBertSequenceCapKey::ModelMaxLength;
    let value = parse_unsigned_value(bytes, token).map_err(|fault| {
        PreparedBertConfigSequenceCandidateError::InvalidPositiveInteger { key, fault, at }
    })?;
    if value == 0 {
        return Err(
            PreparedBertConfigSequenceCandidateError::InvalidPositiveInteger {
                key,
                fault: PreparedBertConfigUnsignedFault::Zero,
                at,
            },
        );
    }
    Ok(RawPreparedBertSequenceCapCandidate::new(key, value))
}

fn parse_unsigned_value(
    bytes: &[u8],
    token: &NumberToken,
) -> Result<u64, PreparedBertConfigUnsignedFault> {
    if token.negative {
        return Err(PreparedBertConfigUnsignedFault::Negative);
    }
    if token.fractional {
        return Err(PreparedBertConfigUnsignedFault::Fractional);
    }
    if token.exponent {
        return Err(PreparedBertConfigUnsignedFault::Exponent);
    }
    let mut value = 0_u64;
    for digit in &bytes[token.range.clone()] {
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add(u64::from(*digit - b'0')))
            .ok_or(PreparedBertConfigUnsignedFault::Overflow)?;
    }
    Ok(value)
}

fn parse_float(
    bytes: &[u8],
    range: Range<usize>,
    field: PreparedBertConfigField,
    at: usize,
) -> ConfigResult<f64> {
    let token = std::str::from_utf8(&bytes[range])
        .map_err(|_| PreparedBertConfigError::InvalidFloat { field, at })?;
    token
        .parse::<f64>()
        .map_err(|_| PreparedBertConfigError::InvalidFloat { field, at })
}

fn classify_field(bytes: &[u8], range: Range<usize>) -> Option<PreparedBertConfigField> {
    PreparedBertConfigField::ALL
        .into_iter()
        .find(|field| decoded_key_equals(&bytes[range.clone()], field.name()))
}

fn decoded_key_equals(encoded: &[u8], target: &[u8]) -> bool {
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
                    let value = decode_hex_quad(encoded, &mut source);
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

fn decode_hex_quad(bytes: &[u8], position: &mut usize) -> u16 {
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

#[cfg(test)]
mod tests {
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
            let document =
                document_with(None, Some((PreparedBertConfigField::VocabSize, value)), &[]);
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
            let document =
                document_with(None, Some((PreparedBertConfigField::VocabSize, value)), &[]);
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
            let document =
                document_with(None, Some((PreparedBertConfigField::VocabSize, value)), &[]);
            assert!(matches!(
                parse(document.as_bytes()),
                Err(PreparedBertConfigError::InvalidUnsignedInteger {
                    field: PreparedBertConfigField::VocabSize,
                    fault: actual,
                    ..
                }) if actual == fault
            ));
        }

        let leading_zero =
            document_with(None, Some((PreparedBertConfigField::VocabSize, "01")), &[]);
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
                CANONICAL.replace("11,", "tru,").into_bytes(),
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
}
