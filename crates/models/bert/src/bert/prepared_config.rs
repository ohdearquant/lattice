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
mod tests;
