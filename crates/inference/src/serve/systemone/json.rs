//! Python-compatible JSON for the system-one prompt layer.
//!
//! The reference reads a request with `json.loads` and renders the state and
//! every non-string description with `json.dumps(value, ensure_ascii=False)`
//! using Python's default separators `", "` and `": "`. `serde_json` is not a
//! drop-in for either direction: without `preserve_order` it discards object
//! key order (and enabling that feature would reorder maps for every other
//! `serde_json` user in the workspace through feature unification), its number
//! model collapses large integers, and its float rendering is Rust's, not
//! Python's `repr`. This module therefore carries a small insertion-ordered
//! value type, a strict RFC 8259 parser, and a renderer that reproduces
//! `json.loads`/`json.dumps` byte for byte on everything the prompt layer
//! accepts.

use indexmap::IndexMap;

use super::SystemOneError;
use super::text::is_zero;

/// A JSON value carrying the two Python behaviours the prompt layer depends on:
/// object keys in document order (a duplicate key keeps the first position and
/// takes the last value) and integer literals that stay integers.
#[derive(Debug, Clone, PartialEq)]
pub enum PyValue {
    /// JSON `null`.
    Null,
    /// JSON `true` / `false`.
    Bool(bool),
    /// An integer literal, kept exact for every magnitude `json.loads` would
    /// keep exact up to `i128`.
    Int(i128),
    /// A number literal that carried `.`, `e`, or `E`.
    Float(f64),
    /// A JSON string (already unescaped and valid UTF-8).
    Str(String),
    /// A JSON array.
    Array(Vec<PyValue>),
    /// A JSON object with keys in document order.
    Object(IndexMap<String, PyValue>),
}

impl PyValue {
    /// The string behind a [`PyValue::Str`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PyValue::Str(value) => Some(value),
            _ => None,
        }
    }

    /// The elements behind a [`PyValue::Array`].
    pub fn as_array(&self) -> Option<&[PyValue]> {
        match self {
            PyValue::Array(values) => Some(values),
            _ => None,
        }
    }

    /// The entries behind a [`PyValue::Object`].
    pub fn as_object(&self) -> Option<&IndexMap<String, PyValue>> {
        match self {
            PyValue::Object(entries) => Some(entries),
            _ => None,
        }
    }

    /// A key lookup that returns the stored value for a present key, even a
    /// `null` one, matching Python `dict.get` with a default.
    pub fn get(&self, key: &str) -> Option<&PyValue> {
        self.as_object()?.get(key)
    }

    /// Python truthiness for the values that reach a boolean context.
    pub fn is_truthy(&self) -> bool {
        match self {
            PyValue::Null => false,
            PyValue::Bool(value) => *value,
            PyValue::Int(value) => *value != 0,
            PyValue::Float(value) => !is_zero(*value),
            PyValue::Str(value) => !value.is_empty(),
            PyValue::Array(values) => !values.is_empty(),
            PyValue::Object(entries) => !entries.is_empty(),
        }
    }

    /// Whether this value is `None` or `""`, the membership test the reference
    /// uses for "no description" (`value in (None, "")`).
    pub fn is_none_or_empty_str(&self) -> bool {
        match self {
            PyValue::Null => true,
            PyValue::Str(value) => value.is_empty(),
            _ => false,
        }
    }

    /// Convert to `serde_json::Value` for consumers that already speak
    /// `serde_json`. Ordering and integer width are those `serde_json` can hold
    /// (an integer beyond `i64` falls back to `f64`).
    pub fn to_serde_json(&self) -> serde_json::Value {
        match self {
            PyValue::Null => serde_json::Value::Null,
            PyValue::Bool(value) => serde_json::Value::Bool(*value),
            PyValue::Int(value) => match i64::try_from(*value) {
                Ok(exact) => serde_json::Value::Number(serde_json::Number::from(exact)),
                Err(_) => serde_json::Number::from_f64(*value as f64)
                    .map_or(serde_json::Value::Null, serde_json::Value::Number),
            },
            PyValue::Float(value) => serde_json::Number::from_f64(*value)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            PyValue::Str(value) => serde_json::Value::String(value.clone()),
            PyValue::Array(values) => {
                serde_json::Value::Array(values.iter().map(PyValue::to_serde_json).collect())
            }
            PyValue::Object(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_serde_json()))
                    .collect(),
            ),
        }
    }
}

/// Parse `body` the way `json.loads` would, refusing the values this port cannot
/// render identically (non-finite literals, lone surrogates, integers past
/// `i128`).
pub fn parse(body: &str) -> Result<PyValue, SystemOneError> {
    let mut parser = Parser {
        chars: body.chars().collect(),
        pos: 0,
    };
    parser.skip_json_ws();
    let value = parser.parse_value()?;
    parser.skip_json_ws();
    if parser.pos != parser.chars.len() {
        return Err(refuse("JSON: unexpected trailing content"));
    }
    Ok(value)
}

/// Render a value the way `json.dumps(value, ensure_ascii=False)` would.
pub fn render(value: &PyValue) -> String {
    let mut out = String::new();
    render_into(value, &mut out);
    out
}

fn refuse(message: impl Into<String>) -> SystemOneError {
    SystemOneError::Refused(message.into())
}

fn render_into(value: &PyValue, out: &mut String) {
    match value {
        PyValue::Null => out.push_str("null"),
        PyValue::Bool(true) => out.push_str("true"),
        PyValue::Bool(false) => out.push_str("false"),
        PyValue::Int(number) => out.push_str(&number.to_string()),
        PyValue::Float(number) => out.push_str(&python_float_repr(*number)),
        PyValue::Str(text) => push_quoted(text, out),
        PyValue::Array(values) => {
            out.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                render_into(item, out);
            }
            out.push(']');
        }
        PyValue::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                push_quoted(key, out);
                out.push_str(": ");
                render_into(item, out);
            }
            out.push('}');
        }
    }
}

/// Python `json.dumps` string escaping with `ensure_ascii=False`: `"`, `\` and
/// U+0000..U+001F only, with the five C0 controls that have short escapes and
/// lowercase `\u00xx` for the rest. Every other code point, including U+007F
/// and every non-ASCII character, is emitted literally.
fn push_quoted(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                out.push_str("\\u00");
                let code = control as u32;
                out.push(hex_digit((code >> 4) & 0xf));
                out.push(hex_digit(code & 0xf));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

fn hex_digit(value: u32) -> char {
    char::from_digit(value, 16).unwrap_or('0')
}

/// Python `repr`/`json.dumps` float rendering: shortest round-trip digits, laid
/// out positionally when the decimal exponent is in `[-4, 16)` and in signed,
/// zero-padded scientific form otherwise, always with at least one fractional
/// digit in the positional form.
///
/// The shortest round-trip digit string is what Rust's `{:e}` already produces;
/// this function only re-lays those digits out the Python way.
pub fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if is_zero(value) {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }

    let scientific = format!("{value:e}");
    let Some((mantissa, exponent_text)) = scientific.split_once('e') else {
        return scientific;
    };
    let Ok(exponent) = exponent_text.parse::<i32>() else {
        return scientific;
    };

    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|ch| *ch != '.').collect();
    if digits.is_empty() {
        return scientific;
    }

    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if !(-4..16).contains(&exponent) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exponent < 0 { '-' } else { '+' };
        let magnitude = exponent.unsigned_abs();
        out.push_str(&format!("e{sign}{magnitude:02}"));
        return out;
    }

    let integer_digits = exponent + 1;
    if integer_digits <= 0 {
        out.push_str("0.");
        for _ in 0..-integer_digits {
            out.push('0');
        }
        out.push_str(&digits);
    } else if integer_digits as usize >= digits.len() {
        out.push_str(&digits);
        for _ in 0..(integer_digits as usize - digits.len()) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        let split = integer_digits as usize;
        out.push_str(&digits[..split]);
        out.push('.');
        out.push_str(&digits[split..]);
    }
    out
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn skip_json_ws(&mut self) {
        while let Some(&ch) = self.chars.get(self.pos) {
            if matches!(ch, ' ' | '\t' | '\n' | '\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<PyValue, SystemOneError> {
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(PyValue::Str(self.parse_string()?)),
            Some('t') => {
                self.expect_literal("true")?;
                Ok(PyValue::Bool(true))
            }
            Some('f') => {
                self.expect_literal("false")?;
                Ok(PyValue::Bool(false))
            }
            Some('n') => {
                self.expect_literal("null")?;
                Ok(PyValue::Null)
            }
            Some('N') => Err(refuse("JSON: the NaN literal is not supported")),
            Some('I') => Err(refuse("JSON: the Infinity literal is not supported")),
            Some('-') => self.parse_number(),
            Some(ch) if ch.is_ascii_digit() => self.parse_number(),
            other => Err(refuse(format!(
                "JSON: expected a value, found {}",
                match other {
                    Some(ch) => format!("{ch:?}"),
                    None => "end of input".to_string(),
                }
            ))),
        }
    }

    fn expect_literal(&mut self, literal: &str) -> Result<(), SystemOneError> {
        for expected in literal.chars() {
            if self.peek() != Some(expected) {
                return Err(refuse(format!("JSON: expected {literal:?}")));
            }
            self.pos += 1;
        }
        Ok(())
    }

    fn parse_object(&mut self) -> Result<PyValue, SystemOneError> {
        self.pos += 1;
        let mut entries: IndexMap<String, PyValue> = IndexMap::new();
        self.skip_json_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(PyValue::Object(entries));
        }
        loop {
            self.skip_json_ws();
            if self.peek() != Some('"') {
                return Err(refuse("JSON: expected a string key"));
            }
            let key = self.parse_string()?;
            self.skip_json_ws();
            if self.peek() != Some(':') {
                return Err(refuse("JSON: expected ':' after a key"));
            }
            self.pos += 1;
            self.skip_json_ws();
            let value = self.parse_value()?;
            // A duplicate key keeps the first position and takes the last
            // value, which is exactly `IndexMap::insert`.
            entries.insert(key, value);
            self.skip_json_ws();
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                }
                Some('}') => {
                    self.pos += 1;
                    return Ok(PyValue::Object(entries));
                }
                _ => return Err(refuse("JSON: expected ',' or '}'")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<PyValue, SystemOneError> {
        self.pos += 1;
        let mut values = Vec::new();
        self.skip_json_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(PyValue::Array(values));
        }
        loop {
            self.skip_json_ws();
            values.push(self.parse_value()?);
            self.skip_json_ws();
            match self.peek() {
                Some(',') => {
                    self.pos += 1;
                }
                Some(']') => {
                    self.pos += 1;
                    return Ok(PyValue::Array(values));
                }
                _ => return Err(refuse("JSON: expected ',' or ']'")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, SystemOneError> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let Some(ch) = self.peek() else {
                return Err(refuse("JSON: unterminated string"));
            };
            self.pos += 1;
            match ch {
                '"' => return Ok(out),
                '\\' => {
                    let Some(escape) = self.peek() else {
                        return Err(refuse("JSON: unterminated escape"));
                    };
                    self.pos += 1;
                    match escape {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{08}'),
                        'f' => out.push('\u{0c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let code = self.parse_hex4()?;
                            if (0xd800..=0xdbff).contains(&code) {
                                let low = self.parse_low_surrogate()?;
                                let combined = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                                let Some(decoded) = char::from_u32(combined) else {
                                    return Err(refuse("JSON: invalid surrogate pair escape"));
                                };
                                out.push(decoded);
                            } else if (0xdc00..=0xdfff).contains(&code) {
                                return Err(refuse("JSON: lone low surrogate escape"));
                            } else {
                                let Some(decoded) = char::from_u32(code) else {
                                    return Err(refuse("JSON: invalid \\u escape"));
                                };
                                out.push(decoded);
                            }
                        }
                        other => {
                            return Err(refuse(format!("JSON: invalid escape \\{other}")));
                        }
                    }
                }
                control if (control as u32) < 0x20 => {
                    return Err(refuse("JSON: raw control character in a string"));
                }
                other => out.push(other),
            }
        }
    }

    fn parse_low_surrogate(&mut self) -> Result<u32, SystemOneError> {
        if self.peek() != Some('\\') {
            return Err(refuse("JSON: lone high surrogate escape"));
        }
        self.pos += 1;
        if self.peek() != Some('u') {
            return Err(refuse("JSON: lone high surrogate escape"));
        }
        self.pos += 1;
        let low = self.parse_hex4()?;
        if !(0xdc00..=0xdfff).contains(&low) {
            return Err(refuse("JSON: lone high surrogate escape"));
        }
        Ok(low)
    }

    fn parse_hex4(&mut self) -> Result<u32, SystemOneError> {
        let mut code = 0u32;
        for _ in 0..4 {
            let Some(ch) = self.peek() else {
                return Err(refuse("JSON: truncated \\u escape"));
            };
            let Some(digit) = ch.to_digit(16) else {
                return Err(refuse("JSON: invalid hex digit in \\u escape"));
            };
            self.pos += 1;
            code = (code << 4) | digit;
        }
        Ok(code)
    }

    fn parse_number(&mut self) -> Result<PyValue, SystemOneError> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
            if matches!(self.peek(), Some('I')) {
                return Err(refuse("JSON: the -Infinity literal is not supported"));
            }
        }
        match self.peek() {
            Some('0') => {
                self.pos += 1;
            }
            Some(ch) if ch.is_ascii_digit() => {
                self.consume_digits();
            }
            _ => return Err(refuse("JSON: malformed number")),
        }
        let mut is_float = false;
        if self.peek() == Some('.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(ch) if ch.is_ascii_digit()) {
                return Err(refuse("JSON: malformed number"));
            }
            self.consume_digits();
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(ch) if ch.is_ascii_digit()) {
                return Err(refuse("JSON: malformed exponent"));
            }
            self.consume_digits();
        }
        let literal: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            let Ok(value) = literal.parse::<f64>() else {
                return Err(refuse(format!("JSON: cannot parse the number {literal}")));
            };
            if !value.is_finite() {
                return Err(refuse(format!("JSON: the number {literal} is not finite")));
            }
            Ok(PyValue::Float(value))
        } else {
            match literal.parse::<i128>() {
                Ok(value) => Ok(PyValue::Int(value)),
                Err(_) => Err(refuse(format!(
                    "JSON: the integer {literal} is beyond the supported range"
                ))),
            }
        }
    }

    fn consume_digits(&mut self) {
        while matches!(self.peek(), Some(ch) if ch.is_ascii_digit()) {
            self.pos += 1;
        }
    }
}

#[cfg(test)]
mod tests;
