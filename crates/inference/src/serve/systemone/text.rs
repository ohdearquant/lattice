//! The four Python behaviours of `str` and `re` this port has to reproduce.
//!
//! Rust's `char::is_whitespace` and `char::is_numeric`, `f64::round`, and the
//! `Debug` formatting of a string are each *close* to Python and each measurably
//! different, and the prompt layer reads all four. The reference builds a level
//! label with `re.sub(r"^\s*-?\d+\s*:\s*", "", text)` on a `str` pattern, rounds
//! reported probabilities with `round`, and embeds a question type in an error
//! message with `{t!r}`.

use super::SystemOneError;

/// Whether `value` is positive or negative zero. `f64`'s `== 0.0` is a float
/// comparison (denied by the workspace lints); the bit test is exact and
/// catches both zeros.
pub fn is_zero(value: f64) -> bool {
    value.to_bits() << 1 == 0
}

/// Python `str.isspace()` for one character: Rust's `char::is_whitespace` plus
/// U+001C..U+001F, the four file/group separators Python also counts as
/// whitespace (measured on Python 3.12.14).
pub fn python_is_space(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

/// The starts of the 68 Unicode `Nd` runs of ten consecutive code points, which
/// is exactly Python's `\d` for `str` patterns on Unicode 15.0.0 (Python 3.12).
const ND_RUN_STARTS: [u32; 68] = [
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
    0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
    0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10,
    0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650,
    0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0,
    0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950,
    0x1fbf0,
];

/// Python `\d` for one character: Unicode `Nd` only. Deliberately narrower than
/// `char::is_numeric`, which also accepts `No`/`Nl`.
pub fn python_is_digit(ch: char) -> bool {
    let code = ch as u32;
    ND_RUN_STARTS
        .iter()
        .any(|&start| code >= start && code < start + 10)
}

/// Python `re.sub(r"^\s*-?\d+\s*:\s*", "", text)` with `str` pattern semantics.
///
/// The pattern is a prefix of the text and every quantifier is greedy, so the
/// match is the longest prefix in that little language: an optional whitespace
/// run, an optional `-`, a nonempty digit run, a whitespace run, one `:`, then a
/// whitespace run, with the two `\s` definitions above.
pub fn strip_level_number(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut best = 0usize;
    for colon in 0..chars.len() {
        if chars[colon] != ':' {
            continue;
        }
        if !is_number_whitespace_prefix(&chars[..colon]) {
            continue;
        }
        let mut end = colon + 1;
        while end < chars.len() && python_is_space(chars[end]) {
            end += 1;
        }
        best = best.max(end);
    }
    chars[best..].iter().collect()
}

/// Whether `prefix` matches `\s*-?\d+\s*` in full.
fn is_number_whitespace_prefix(prefix: &[char]) -> bool {
    let mut index = 0;
    while index < prefix.len() && python_is_space(prefix[index]) {
        index += 1;
    }
    if index < prefix.len() && prefix[index] == '-' {
        index += 1;
    }
    let digits_start = index;
    while index < prefix.len() && python_is_digit(prefix[index]) {
        index += 1;
    }
    if index == digits_start {
        return false;
    }
    while index < prefix.len() && python_is_space(prefix[index]) {
        index += 1;
    }
    index == prefix.len()
}

/// Python `round(value, digits)`.
///
/// Python and Rust differ here: Python rounds the decimal string of the value
/// half to even (so `round(0.005, 2)` is `0.01`), while scaling by `10^digits`
/// and rounding the scaled `f64` sees `0.005 * 100` as the `f64` just below
/// `0.5` and rounds the wrong way. Rust's `{:.digits}` formatting is itself
/// correctly rounded to nearest, ties to even on the exact binary value, which
/// is what Python does, so the value is rendered at that precision and parsed
/// back. Digits is a small compile-time constant (2 or 4) in every call, so the
/// format string is used as a route, not built dynamically.
pub fn python_round(value: f64, digits: i32) -> f64 {
    if !value.is_finite() {
        return value;
    }
    match digits {
        2 => format!("{value:.2}").parse::<f64>().unwrap_or(value),
        4 => format!("{value:.4}").parse::<f64>().unwrap_or(value),
        other => {
            let factor = 10f64.powi(other);
            let scaled = value * factor;
            if !scaled.is_finite() {
                return value;
            }
            scaled.round_ties_even() / factor
        }
    }
}

/// Python `str()` for the values a choice's list criteria can hold, which this
/// port accepts only for strings and integers.
pub fn python_str_of_list_criterion(
    value: &super::json::PyValue,
) -> Result<String, SystemOneError> {
    use super::json::PyValue;
    match value {
        PyValue::Str(text) => Ok(text.clone()),
        PyValue::Int(number) => Ok(number.to_string()),
        other => Err(SystemOneError::Refused(format!(
            "choice list criteria: option {} is not a string or an integer",
            python_repr_of(other)
        ))),
    }
}

/// Python `repr` for a value, as far as this layer needs it in a message.
pub fn python_repr_of(value: &super::json::PyValue) -> String {
    use super::json::PyValue;
    match value {
        PyValue::Str(text) => python_repr_str(text),
        PyValue::Int(number) => number.to_string(),
        PyValue::Float(number) => super::json::python_float_repr(*number),
        PyValue::Bool(true) => "True".to_string(),
        PyValue::Bool(false) => "False".to_string(),
        PyValue::Null => "None".to_string(),
        PyValue::Array(_) => "[...]".to_string(),
        PyValue::Object(_) => "{...}".to_string(),
    }
}

/// Python `repr` for a string: single-quoted, with the backslash, the quote and
/// the C0 controls escaped and everything else literal.
pub fn python_repr_str(text: &str) -> String {
    let mut out = String::from("'");
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 || control == '\u{7f}' => {
                out.push_str(&format!("\\x{:02x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('\'');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_zero_catches_both_zeros() {
        assert!(is_zero(0.0));
        assert!(is_zero(-0.0));
        assert!(!is_zero(1e-300));
        assert!(!is_zero(-1.0));
    }

    #[test]
    fn whitespace_matches_str_isspace_plus_the_four_separators() {
        assert!(python_is_space(' '));
        assert!(python_is_space('\n'));
        assert!(python_is_space('\u{1c}'));
        assert!(python_is_space('\u{1f}'));
        // Rust already counts these; Python also counts them.
        assert!(python_is_space('\u{85}'));
        assert!(python_is_space('\u{a0}'));
        assert!(!python_is_space('a'));
        assert!(!python_is_space('\u{1b}'));
    }

    #[test]
    fn digit_is_nd_only() {
        assert!(python_is_digit('0'));
        assert!(python_is_digit('\u{660}'));
        assert!(python_is_digit('\u{663}'));
        assert!(python_is_digit('\u{ff10}'));
        // `char::is_numeric` accepts these; Nd does not.
        assert!(!python_is_digit('½'));
        assert!(!python_is_digit('Ⅻ'));
        assert!(!python_is_digit('a'));
    }

    #[test]
    fn strips_the_level_number_prefix() {
        assert_eq!(strip_level_number("2: somewhat"), "somewhat");
        assert_eq!(strip_level_number(" -1 : minus"), "minus");
        assert_eq!(
            strip_level_number("\u{663}: arabic three digit"),
            "arabic three digit"
        );
        assert_eq!(strip_level_number("1: 1: level 1"), "1: level 1");
        assert_eq!(strip_level_number("low"), "low");
        assert_eq!(strip_level_number(""), "");
        assert_eq!(strip_level_number("-3:neg"), "neg");
    }

    #[test]
    fn python_round_is_half_to_even() {
        assert_eq!(python_round(0.125, 2), 0.12);
        assert_eq!(python_round(0.375, 2), 0.38);
        assert_eq!(python_round(2.675, 2), 2.67);
        assert_eq!(python_round(1.0 / 3.0, 4), 0.3333);
        assert_eq!(python_round(0.99996, 4), 1.0);
    }

    #[test]
    fn repr_of_a_string_quotes_like_python() {
        assert_eq!(python_repr_str("rank"), "'rank'");
        assert_eq!(python_repr_str("a'b"), "'a\\'b'");
    }
}
