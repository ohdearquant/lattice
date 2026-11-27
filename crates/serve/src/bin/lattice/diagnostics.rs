/// Unicode 15.1.0 general category Cf (format characters), checked against the pinned
/// DerivedGeneralCategory extract in `tests/fixtures/unicode`; its README records provenance.
/// These characters can reorder or hide text in a rendered report line.
pub(super) const FORMAT_CHARACTER_RANGES: &[(char, char)] = &[
    ('\u{00AD}', '\u{00AD}'),
    ('\u{0600}', '\u{0605}'),
    ('\u{061C}', '\u{061C}'),
    ('\u{06DD}', '\u{06DD}'),
    ('\u{070F}', '\u{070F}'),
    ('\u{0890}', '\u{0891}'),
    ('\u{08E2}', '\u{08E2}'),
    ('\u{180E}', '\u{180E}'),
    ('\u{200B}', '\u{200F}'),
    ('\u{202A}', '\u{202E}'),
    ('\u{2060}', '\u{2064}'),
    ('\u{2066}', '\u{206F}'),
    ('\u{FEFF}', '\u{FEFF}'),
    ('\u{FFF9}', '\u{FFFB}'),
    ('\u{110BD}', '\u{110BD}'),
    ('\u{110CD}', '\u{110CD}'),
    ('\u{13430}', '\u{1343F}'),
    ('\u{1BCA0}', '\u{1BCA3}'),
    ('\u{1D173}', '\u{1D17A}'),
    ('\u{E0001}', '\u{E0001}'),
    ('\u{E0020}', '\u{E007F}'),
];

/// Whether a report diagnostic prints `character` escaped: control characters (Cc), format
/// characters (Cf), the line separator U+2028 (Zl) and the paragraph separator U+2029 (Zp).
/// Backslashes are escaped so literal text cannot be mistaken for an escape sequence.
/// Non-ASCII space separators and Hangul fillers are escaped so blank glyphs cannot hide
/// path or label contents. This bounded set is Unicode 15.1 Zs excluding U+0020, plus
/// U+115F (choseong filler), U+1160 (jungseong filler), U+3164 (Hangul filler) and
/// U+FFA0 (halfwidth Hangul filler); ordinary ASCII space remains printable.
pub(super) fn is_escaped_in_report(character: char) -> bool {
    character == '\\'
        || character.is_control()
        || matches!(character, '\u{2028}' | '\u{2029}')
        || matches!(
            character,
            '\u{00A0}' | '\u{1680}' | '\u{2000}'
                ..='\u{200A}'
                    | '\u{202F}'
                    | '\u{205F}'
                    | '\u{3000}'
                    | '\u{115F}'
                    | '\u{1160}'
                    | '\u{3164}'
                    | '\u{FFA0}'
        )
        || FORMAT_CHARACTER_RANGES
            .iter()
            .any(|&(first, last)| (first..=last).contains(&character))
}

pub(super) fn write_report_diagnostic(
    f: &mut std::fmt::Formatter<'_>,
    prefix: &str,
    text: &str,
) -> std::fmt::Result {
    f.write_str(prefix)?;
    for character in text.chars() {
        if is_escaped_in_report(character) {
            write!(f, "{}", character.escape_default())?;
        } else {
            write!(f, "{character}")?;
        }
    }
    writeln!(f)
}

pub(super) fn error_diagnostic(text: &str) -> impl std::fmt::Display + '_ {
    diagnostic("Error: ", text)
}

struct Diagnostic<'a> {
    prefix: &'a str,
    text: &'a str,
}

impl std::fmt::Display for Diagnostic<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_report_diagnostic(f, self.prefix, self.text)
    }
}

pub(super) fn diagnostic<'a>(prefix: &'a str, text: &'a str) -> impl std::fmt::Display + 'a {
    Diagnostic { prefix, text }
}
