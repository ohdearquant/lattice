//! Dormant checked parsing contract for prepared BERT `tokenizer_config.json` bytes.

use std::num::NonZeroU64;

pub(super) const MAX_PREPARED_BERT_TOKENIZER_CONFIG_NESTING: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerConfigCandidateKey {
    ModelMaxLength,
    MaxPositionEmbeddings,
    NPositions,
    MaxSeqLen,
    TruncationMaxLength,
}

impl PreparedBertTokenizerConfigCandidateKey {
    const ALL: [Self; 5] = [
        Self::ModelMaxLength,
        Self::MaxPositionEmbeddings,
        Self::NPositions,
        Self::MaxSeqLen,
        Self::TruncationMaxLength,
    ];

    fn index(self) -> usize {
        match self {
            Self::ModelMaxLength => 0,
            Self::MaxPositionEmbeddings => 1,
            Self::NPositions => 2,
            Self::MaxSeqLen => 3,
            Self::TruncationMaxLength => 4,
        }
    }

    fn root_name(self) -> Option<&'static [u8]> {
        match self {
            Self::ModelMaxLength => Some(b"model_max_length"),
            Self::MaxPositionEmbeddings => Some(b"max_position_embeddings"),
            Self::NPositions => Some(b"n_positions"),
            Self::MaxSeqLen => Some(b"max_seq_len"),
            Self::TruncationMaxLength => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerConfigLimitAxis {
    TokenizerConfigBytes,
    TokenizerConfigParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerConfigExpression {
    TokenizerConfigParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerConfigValueKind {
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerConfigExpectedType {
    PositiveInteger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerConfigUnsignedFault {
    Zero,
    Negative,
    Fractional,
    Exponent,
    Overflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTokenizerConfigSyntaxFault {
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
pub(super) enum PreparedBertTokenizerConfigError {
    InputLengthUnrepresentable,
    Exceeded {
        axis: PreparedBertTokenizerConfigLimitAxis,
        actual: u64,
        limit: u64,
    },
    PlatformUnrepresentable {
        axis: PreparedBertTokenizerConfigLimitAxis,
        value: u64,
    },
    ArithmeticOverflow(PreparedBertTokenizerConfigExpression),
    MalformedJson {
        at: usize,
        fault: PreparedBertTokenizerConfigSyntaxFault,
    },
    DuplicateCandidate {
        key: PreparedBertTokenizerConfigCandidateKey,
        first_at: usize,
        duplicate_at: usize,
    },
    InvalidCandidateType {
        key: PreparedBertTokenizerConfigCandidateKey,
        expected: PreparedBertTokenizerConfigExpectedType,
        actual: PreparedBertTokenizerConfigValueKind,
        at: usize,
    },
    InvalidPositiveInteger {
        key: PreparedBertTokenizerConfigCandidateKey,
        fault: PreparedBertTokenizerConfigUnsignedFault,
        at: usize,
    },
}

type ConfigResult<T> = Result<T, PreparedBertTokenizerConfigError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertTokenizerConfigLimits {
    max_tokenizer_config_bytes: NonZeroU64,
    max_tokenizer_config_parse_work_bytes: NonZeroU64,
}

impl PreparedBertTokenizerConfigLimits {
    pub(super) fn try_new(
        max_tokenizer_config_bytes: NonZeroU64,
        max_tokenizer_config_parse_work_bytes: NonZeroU64,
    ) -> ConfigResult<Self> {
        let usize_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
        let isize_max = u64::try_from(isize::MAX).unwrap_or(u64::MAX);
        Self::try_new_with_platform_max(
            max_tokenizer_config_bytes,
            max_tokenizer_config_parse_work_bytes,
            usize_max.min(isize_max),
        )
    }

    fn try_new_with_platform_max(
        max_tokenizer_config_bytes: NonZeroU64,
        max_tokenizer_config_parse_work_bytes: NonZeroU64,
        platform_max: u64,
    ) -> ConfigResult<Self> {
        for (axis, value) in [
            (
                PreparedBertTokenizerConfigLimitAxis::TokenizerConfigBytes,
                max_tokenizer_config_bytes.get(),
            ),
            (
                PreparedBertTokenizerConfigLimitAxis::TokenizerConfigParseWorkBytes,
                max_tokenizer_config_parse_work_bytes.get(),
            ),
        ] {
            if value > platform_max {
                return Err(PreparedBertTokenizerConfigError::PlatformUnrepresentable {
                    axis,
                    value,
                });
            }
        }
        Ok(Self {
            max_tokenizer_config_bytes,
            max_tokenizer_config_parse_work_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertTokenizerMaxLengthCandidate {
    key: PreparedBertTokenizerConfigCandidateKey,
    raw_value: u64,
}

impl PreparedBertTokenizerMaxLengthCandidate {
    pub(super) fn key(self) -> PreparedBertTokenizerConfigCandidateKey {
        self.key
    }

    pub(super) fn raw_value(self) -> u64 {
        self.raw_value
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ParsedBertTokenizerConfigFacts {
    candidate: Option<PreparedBertTokenizerMaxLengthCandidate>,
    tokenizer_config_bytes: u64,
    logical_tokenizer_config_parse_work_bytes: u64,
}

impl ParsedBertTokenizerConfigFacts {
    pub(super) fn candidate(self) -> Option<PreparedBertTokenizerMaxLengthCandidate> {
        self.candidate
    }

    pub(super) fn tokenizer_config_bytes(self) -> u64 {
        self.tokenizer_config_bytes
    }

    pub(super) fn logical_tokenizer_config_parse_work_bytes(self) -> u64 {
        self.logical_tokenizer_config_parse_work_bytes
    }
}

fn checked_tokenizer_config_parse_work_bytes(tokenizer_config_bytes: u64) -> ConfigResult<u64> {
    tokenizer_config_bytes.checked_mul(2).ok_or(
        PreparedBertTokenizerConfigError::ArithmeticOverflow(
            PreparedBertTokenizerConfigExpression::TokenizerConfigParseWorkBytes,
        ),
    )
}

pub(super) fn parse_prepared_bert_tokenizer_config_json(
    bytes: &[u8],
    limits: &PreparedBertTokenizerConfigLimits,
) -> ConfigResult<ParsedBertTokenizerConfigFacts> {
    let tokenizer_config_bytes = u64::try_from(bytes.len())
        .map_err(|_| PreparedBertTokenizerConfigError::InputLengthUnrepresentable)?;
    if tokenizer_config_bytes > limits.max_tokenizer_config_bytes.get() {
        return Err(PreparedBertTokenizerConfigError::Exceeded {
            axis: PreparedBertTokenizerConfigLimitAxis::TokenizerConfigBytes,
            actual: tokenizer_config_bytes,
            limit: limits.max_tokenizer_config_bytes.get(),
        });
    }
    let logical_tokenizer_config_parse_work_bytes =
        checked_tokenizer_config_parse_work_bytes(tokenizer_config_bytes)?;
    if logical_tokenizer_config_parse_work_bytes
        > limits.max_tokenizer_config_parse_work_bytes.get()
    {
        return Err(PreparedBertTokenizerConfigError::Exceeded {
            axis: PreparedBertTokenizerConfigLimitAxis::TokenizerConfigParseWorkBytes,
            actual: logical_tokenizer_config_parse_work_bytes,
            limit: limits.max_tokenizer_config_parse_work_bytes.get(),
        });
    }

    JsonCursor::new(bytes).validate_document()?;
    let candidate = SemanticParser::new(bytes).parse()?;
    Ok(ParsedBertTokenizerConfigFacts {
        candidate,
        tokenizer_config_bytes,
        logical_tokenizer_config_parse_work_bytes,
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

#[derive(Clone, Copy)]
struct NumberToken {
    start: usize,
    end: usize,
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

    fn malformed<T>(
        &self,
        at: usize,
        fault: PreparedBertTokenizerConfigSyntaxFault,
    ) -> ConfigResult<T> {
        Err(PreparedBertTokenizerConfigError::MalformedJson { at, fault })
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn validate_document(mut self) -> ConfigResult<()> {
        self.skip_whitespace();
        if self.bytes.get(self.pos) != Some(&b'{') {
            return self.malformed(
                self.pos,
                PreparedBertTokenizerConfigSyntaxFault::ExpectedObjectStart,
            );
        }
        self.skip_value()?;
        self.skip_whitespace();
        if self.pos != self.bytes.len() {
            return self.malformed(
                self.pos,
                PreparedBertTokenizerConfigSyntaxFault::TrailingNonWhitespace,
            );
        }
        Ok(())
    }

    fn skip_value(&mut self) -> ConfigResult<PreparedBertTokenizerConfigValueKind> {
        self.skip_whitespace();
        let kind = self.begin_value()?;
        let first_state = match kind {
            PreparedBertTokenizerConfigValueKind::Array => Some(ContainerState::ArrayFirstOrEnd),
            PreparedBertTokenizerConfigValueKind::Object => {
                Some(ContainerState::ObjectFirstKeyOrEnd)
            }
            _ => None,
        };
        let Some(first_state) = first_state else {
            return Ok(kind);
        };

        let mut stack =
            [ContainerState::ArrayFirstOrEnd; MAX_PREPARED_BERT_TOKENIZER_CONFIG_NESTING];
        stack[0] = first_state;
        let mut depth = 1_usize;
        loop {
            match stack[depth - 1] {
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
                        return self.malformed(
                            self.pos,
                            PreparedBertTokenizerConfigSyntaxFault::TrailingComma,
                        );
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
                            return self.malformed(
                                self.pos,
                                PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof,
                            );
                        }
                        _ => {
                            return self.malformed(
                                self.pos,
                                PreparedBertTokenizerConfigSyntaxFault::ExpectedCommaOrEnd,
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
                        return self.malformed(
                            self.pos,
                            PreparedBertTokenizerConfigSyntaxFault::TrailingComma,
                        );
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
                                PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof
                            } else {
                                PreparedBertTokenizerConfigSyntaxFault::ExpectedColon
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
                            return self.malformed(
                                self.pos,
                                PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof,
                            );
                        }
                        _ => {
                            return self.malformed(
                                self.pos,
                                PreparedBertTokenizerConfigSyntaxFault::ExpectedCommaOrEnd,
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
        stack: &mut [ContainerState; MAX_PREPARED_BERT_TOKENIZER_CONFIG_NESTING],
        depth: &mut usize,
    ) -> ConfigResult<()> {
        self.skip_whitespace();
        let at = self.pos;
        let state = match self.begin_value()? {
            PreparedBertTokenizerConfigValueKind::Array => Some(ContainerState::ArrayFirstOrEnd),
            PreparedBertTokenizerConfigValueKind::Object => {
                Some(ContainerState::ObjectFirstKeyOrEnd)
            }
            _ => None,
        };
        if let Some(state) = state {
            if *depth == MAX_PREPARED_BERT_TOKENIZER_CONFIG_NESTING {
                return self.malformed(
                    at,
                    PreparedBertTokenizerConfigSyntaxFault::NestingLimitExceeded {
                        limit: MAX_PREPARED_BERT_TOKENIZER_CONFIG_NESTING,
                    },
                );
            }
            stack[*depth] = state;
            *depth += 1;
        }
        Ok(())
    }

    fn begin_value(&mut self) -> ConfigResult<PreparedBertTokenizerConfigValueKind> {
        self.skip_whitespace();
        let Some(byte) = self.bytes.get(self.pos).copied() else {
            return self.malformed(
                self.pos,
                PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof,
            );
        };
        match byte {
            b'"' => {
                self.scan_string()?;
                Ok(PreparedBertTokenizerConfigValueKind::String)
            }
            b'{' => {
                self.pos += 1;
                Ok(PreparedBertTokenizerConfigValueKind::Object)
            }
            b'[' => {
                self.pos += 1;
                Ok(PreparedBertTokenizerConfigValueKind::Array)
            }
            b't' => {
                self.scan_literal(b"true")?;
                Ok(PreparedBertTokenizerConfigValueKind::Boolean)
            }
            b'f' => {
                self.scan_literal(b"false")?;
                Ok(PreparedBertTokenizerConfigValueKind::Boolean)
            }
            b'n' => {
                self.scan_literal(b"null")?;
                Ok(PreparedBertTokenizerConfigValueKind::Null)
            }
            b'-' | b'0'..=b'9' => {
                self.scan_number()?;
                Ok(PreparedBertTokenizerConfigValueKind::Number)
            }
            b'+' | b'.' => self.malformed(
                self.pos,
                PreparedBertTokenizerConfigSyntaxFault::InvalidNumber,
            ),
            b'a'..=b'z' | b'A'..=b'Z' => self.malformed(
                self.pos,
                PreparedBertTokenizerConfigSyntaxFault::InvalidLiteral,
            ),
            _ => self.malformed(
                self.pos,
                PreparedBertTokenizerConfigSyntaxFault::ExpectedValue,
            ),
        }
    }

    fn scan_literal(&mut self, literal: &[u8]) -> ConfigResult<()> {
        let at = self.pos;
        let end = self.pos.saturating_add(literal.len());
        if self.bytes.get(self.pos..end) != Some(literal) {
            return self.malformed(at, PreparedBertTokenizerConfigSyntaxFault::InvalidLiteral);
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
                    return self.malformed(
                        self.pos,
                        PreparedBertTokenizerConfigSyntaxFault::InvalidNumber,
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
                    PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof,
                );
            }
            _ => {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerConfigSyntaxFault::InvalidNumber,
                );
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
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerConfigSyntaxFault::InvalidNumber,
                );
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
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerConfigSyntaxFault::InvalidNumber,
                );
            }
        }
        Ok(NumberToken {
            start,
            end: self.pos,
            negative,
            fractional,
            exponent,
        })
    }

    fn scan_string(&mut self) -> ConfigResult<(usize, usize)> {
        if self.bytes.get(self.pos) != Some(&b'"') {
            return self.malformed(
                self.pos,
                if self.pos == self.bytes.len() {
                    PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof
                } else {
                    PreparedBertTokenizerConfigSyntaxFault::ExpectedString
                },
            );
        }
        self.pos += 1;
        let start = self.pos;
        loop {
            let Some(byte) = self.bytes.get(self.pos).copied() else {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof,
                );
            };
            match byte {
                b'"' => {
                    let end = self.pos;
                    self.pos += 1;
                    return Ok((start, end));
                }
                b'\\' => self.scan_escape()?,
                0x00..=0x1f => {
                    return self.malformed(
                        self.pos,
                        PreparedBertTokenizerConfigSyntaxFault::UnescapedControlCharacter,
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
            return self.malformed(
                self.pos,
                PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof,
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
                            PreparedBertTokenizerConfigSyntaxFault::InvalidUnicodeEscape,
                        );
                    }
                    self.pos += 2;
                    let second = self.scan_hex_quad(slash_at)?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return self.malformed(
                            slash_at,
                            PreparedBertTokenizerConfigSyntaxFault::InvalidUnicodeEscape,
                        );
                    }
                } else if (0xdc00..=0xdfff).contains(&first) {
                    return self.malformed(
                        slash_at,
                        PreparedBertTokenizerConfigSyntaxFault::InvalidUnicodeEscape,
                    );
                }
                Ok(())
            }
            _ => self.malformed(
                slash_at,
                PreparedBertTokenizerConfigSyntaxFault::InvalidStringEscape,
            ),
        }
    }

    fn scan_hex_quad(&mut self, at: usize) -> ConfigResult<u16> {
        let mut value = 0_u16;
        for _ in 0..4 {
            let Some(byte) = self.bytes.get(self.pos).copied() else {
                return self.malformed(
                    self.pos,
                    PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof,
                );
            };
            let digit = match byte {
                b'0'..=b'9' => u16::from(byte - b'0'),
                b'a'..=b'f' => u16::from(byte - b'a' + 10),
                b'A'..=b'F' => u16::from(byte - b'A' + 10),
                _ => {
                    return self.malformed(
                        at,
                        PreparedBertTokenizerConfigSyntaxFault::InvalidUnicodeEscape,
                    );
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
                return self.malformed(at, PreparedBertTokenizerConfigSyntaxFault::InvalidUtf8);
            }
        };
        let Some(sequence) = self.bytes.get(self.pos..self.pos.saturating_add(width)) else {
            return self.malformed(at, PreparedBertTokenizerConfigSyntaxFault::InvalidUtf8);
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
            return self.malformed(at, PreparedBertTokenizerConfigSyntaxFault::InvalidUtf8);
        }
        self.pos += width;
        Ok(())
    }

    fn peek_value_kind(&self) -> ConfigResult<PreparedBertTokenizerConfigValueKind> {
        let mut probe = Self {
            bytes: self.bytes,
            pos: self.pos,
        };
        probe.skip_whitespace();
        let Some(byte) = probe.bytes.get(probe.pos) else {
            return probe.malformed(
                probe.pos,
                PreparedBertTokenizerConfigSyntaxFault::UnexpectedEof,
            );
        };
        Ok(match byte {
            b'n' => PreparedBertTokenizerConfigValueKind::Null,
            b't' | b'f' => PreparedBertTokenizerConfigValueKind::Boolean,
            b'-' | b'0'..=b'9' => PreparedBertTokenizerConfigValueKind::Number,
            b'"' => PreparedBertTokenizerConfigValueKind::String,
            b'[' => PreparedBertTokenizerConfigValueKind::Array,
            b'{' => PreparedBertTokenizerConfigValueKind::Object,
            _ => {
                return probe.malformed(
                    probe.pos,
                    PreparedBertTokenizerConfigSyntaxFault::ExpectedValue,
                );
            }
        })
    }
}

#[derive(Clone, Copy)]
struct CandidateOccurrence {
    key_at: usize,
    value_at: usize,
    kind: PreparedBertTokenizerConfigValueKind,
    number: Option<NumberToken>,
}

struct SemanticParser<'a> {
    cursor: JsonCursor<'a>,
    candidates: [Option<CandidateOccurrence>; 5],
}

impl<'a> SemanticParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: JsonCursor::new(bytes),
            candidates: [None; 5],
        }
    }

    fn parse(mut self) -> ConfigResult<Option<PreparedBertTokenizerMaxLengthCandidate>> {
        self.cursor.skip_whitespace();
        self.cursor.pos += 1;
        self.parse_members(false)?;

        let Some(key) = PreparedBertTokenizerConfigCandidateKey::ALL
            .into_iter()
            .find(|key| self.candidates[key.index()].is_some())
        else {
            return Ok(None);
        };
        let occurrence = self.candidates[key.index()].ok_or(
            PreparedBertTokenizerConfigError::InvalidCandidateType {
                key,
                expected: PreparedBertTokenizerConfigExpectedType::PositiveInteger,
                actual: PreparedBertTokenizerConfigValueKind::Null,
                at: 0,
            },
        )?;
        if occurrence.kind != PreparedBertTokenizerConfigValueKind::Number {
            return Err(PreparedBertTokenizerConfigError::InvalidCandidateType {
                key,
                expected: PreparedBertTokenizerConfigExpectedType::PositiveInteger,
                actual: occurrence.kind,
                at: occurrence.value_at,
            });
        }
        let number =
            occurrence
                .number
                .ok_or(PreparedBertTokenizerConfigError::InvalidCandidateType {
                    key,
                    expected: PreparedBertTokenizerConfigExpectedType::PositiveInteger,
                    actual: occurrence.kind,
                    at: occurrence.value_at,
                })?;
        let raw_value =
            parse_positive_integer(self.cursor.bytes, number, key, occurrence.value_at)?;
        Ok(Some(PreparedBertTokenizerMaxLengthCandidate {
            key,
            raw_value,
        }))
    }

    fn parse_members(&mut self, truncation: bool) -> ConfigResult<()> {
        self.cursor.skip_whitespace();
        if self.cursor.bytes.get(self.cursor.pos) == Some(&b'}') {
            self.cursor.pos += 1;
            return Ok(());
        }
        loop {
            self.cursor.skip_whitespace();
            let key_at = self.cursor.pos;
            let (start, end) = self.cursor.scan_string()?;
            self.cursor.skip_whitespace();
            self.cursor.pos += 1;
            self.cursor.skip_whitespace();
            let value_at = self.cursor.pos;
            let key_bytes = &self.cursor.bytes[start..end];

            if truncation && decoded_key_equals(key_bytes, b"max_length") {
                self.capture_candidate(
                    PreparedBertTokenizerConfigCandidateKey::TruncationMaxLength,
                    key_at,
                    value_at,
                )?;
            } else if !truncation {
                if let Some(key) = classify_root_candidate(key_bytes) {
                    self.capture_candidate(key, key_at, value_at)?;
                } else if decoded_key_equals(key_bytes, b"truncation")
                    && self.cursor.peek_value_kind()?
                        == PreparedBertTokenizerConfigValueKind::Object
                {
                    self.cursor.pos += 1;
                    self.parse_members(true)?;
                } else {
                    self.cursor.skip_value()?;
                }
            } else {
                self.cursor.skip_value()?;
            }

            self.cursor.skip_whitespace();
            match self.cursor.bytes.get(self.cursor.pos) {
                Some(b',') => self.cursor.pos += 1,
                Some(b'}') => {
                    self.cursor.pos += 1;
                    return Ok(());
                }
                _ => {
                    return self.cursor.malformed(
                        self.cursor.pos,
                        PreparedBertTokenizerConfigSyntaxFault::ExpectedCommaOrEnd,
                    );
                }
            }
        }
    }

    fn capture_candidate(
        &mut self,
        key: PreparedBertTokenizerConfigCandidateKey,
        key_at: usize,
        value_at: usize,
    ) -> ConfigResult<()> {
        let index = key.index();
        if let Some(first) = self.candidates[index] {
            return Err(PreparedBertTokenizerConfigError::DuplicateCandidate {
                key,
                first_at: first.key_at,
                duplicate_at: key_at,
            });
        }
        let kind = self.cursor.peek_value_kind()?;
        let number = if kind == PreparedBertTokenizerConfigValueKind::Number {
            Some(self.cursor.scan_number()?)
        } else {
            self.cursor.skip_value()?;
            None
        };
        self.candidates[index] = Some(CandidateOccurrence {
            key_at,
            value_at,
            kind,
            number,
        });
        Ok(())
    }
}

fn classify_root_candidate(encoded: &[u8]) -> Option<PreparedBertTokenizerConfigCandidateKey> {
    PreparedBertTokenizerConfigCandidateKey::ALL
        .into_iter()
        .filter_map(|key| key.root_name().map(|name| (key, name)))
        .find_map(|(key, name)| decoded_key_equals(encoded, name).then_some(key))
}

fn parse_positive_integer(
    bytes: &[u8],
    token: NumberToken,
    key: PreparedBertTokenizerConfigCandidateKey,
    at: usize,
) -> ConfigResult<u64> {
    use PreparedBertTokenizerConfigUnsignedFault::{
        Exponent, Fractional, Negative, Overflow, Zero,
    };

    let fault = if token.negative {
        Some(Negative)
    } else if token.fractional {
        Some(Fractional)
    } else if token.exponent {
        Some(Exponent)
    } else {
        None
    };
    if let Some(fault) = fault {
        return Err(PreparedBertTokenizerConfigError::InvalidPositiveInteger { key, fault, at });
    }
    let mut value = 0_u64;
    for digit in &bytes[token.start..token.end] {
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add(u64::from(*digit - b'0')))
            .ok_or(PreparedBertTokenizerConfigError::InvalidPositiveInteger {
                key,
                fault: Overflow,
                at,
            })?;
    }
    if value == 0 {
        return Err(PreparedBertTokenizerConfigError::InvalidPositiveInteger {
            key,
            fault: Zero,
            at,
        });
    }
    Ok(value)
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
    use std::num::NonZeroU64;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn exact_limits(bytes: &[u8]) -> PreparedBertTokenizerConfigLimits {
        let len = u64::try_from(bytes.len()).unwrap();
        PreparedBertTokenizerConfigLimits::try_new(nz(len), nz(len.checked_mul(2).unwrap()))
            .unwrap()
    }

    fn parse(
        bytes: &[u8],
    ) -> Result<ParsedBertTokenizerConfigFacts, PreparedBertTokenizerConfigError> {
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

        let short_bytes =
            PreparedBertTokenizerConfigLimits::try_new(nz(len - 1), nz(len * 2)).unwrap();
        assert!(matches!(
            parse_prepared_bert_tokenizer_config_json(bytes, &short_bytes),
            Err(PreparedBertTokenizerConfigError::Exceeded {
                axis: PreparedBertTokenizerConfigLimitAxis::TokenizerConfigBytes,
                ..
            })
        ));

        let short_work =
            PreparedBertTokenizerConfigLimits::try_new(nz(len), nz(len * 2 - 1)).unwrap();
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
}
