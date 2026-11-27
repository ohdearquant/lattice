//! Pre-tokenization by the regex `Split` a `tokenizer.json` declares.

use lattice_inference_core::InferenceError;
use regex::Regex;

/// The one look-around the published byte-level BPE patterns use, as a whole
/// top-level alternative: a whitespace run not followed by non-whitespace.
const TRAILING_WS: &str = r"\s+(?!\S)";

/// [`TRAILING_WS`] without look-around. Greedy `\s+` followed by one more
/// whitespace character or the end of the text succeeds exactly where the
/// look-ahead form succeeds, and the named group spans the look-ahead form's
/// match. The extra character stays in the text for the next piece.
const TRAILING_WS_STANDIN: &str = r"(?P<lattice_trailing_ws>\s+)(?:\s|\z)";
const TRAILING_WS_GROUP: &str = "lattice_trailing_ws";

/// A declared `Split` pattern with `Isolated` behavior: every match is a
/// piece, and so is every run of text between matches.
#[derive(Debug)]
pub(super) struct DeclaredSplit {
    pattern: String,
    regex: Regex,
    trailing_ws_group: Option<usize>,
}

impl DeclaredSplit {
    /// Compile `pattern`, refusing one the `regex` crate cannot run once the
    /// trailing-whitespace alternative is replaced (any other look-around,
    /// back-references, or that alternative inside a group).
    pub(super) fn compile(pattern: &str) -> Result<Self, InferenceError> {
        let alternatives = top_level_alternatives(pattern);
        let trailing = alternatives
            .iter()
            .filter(|alt| **alt == TRAILING_WS)
            .count();
        if trailing > 1 {
            return Err(refusal(
                pattern,
                "the trailing-whitespace alternative appears more than once",
            ));
        }
        let rewritten = alternatives
            .iter()
            .map(|alt| {
                if *alt == TRAILING_WS {
                    TRAILING_WS_STANDIN
                } else {
                    alt
                }
            })
            .collect::<Vec<_>>()
            .join("|");
        let regex = Regex::new(&rewritten).map_err(|e| refusal(pattern, &e.to_string()))?;
        let trailing_ws_group = if trailing == 1 {
            regex
                .capture_names()
                .position(|name| name == Some(TRAILING_WS_GROUP))
        } else {
            None
        };
        Ok(Self {
            pattern: pattern.to_string(),
            regex,
            trailing_ws_group,
        })
    }

    pub(super) fn pattern(&self) -> &str {
        &self.pattern
    }

    pub(super) fn split(&self, text: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        let mut locations = self.regex.capture_locations();
        let mut kept = 0;
        let mut search = 0;
        while search < text.len() {
            let Some(whole) = self.regex.captures_read_at(&mut locations, text, search) else {
                break;
            };
            let (start, end) = self
                .trailing_ws_group
                .and_then(|group| locations.get(group))
                .unwrap_or((whole.start(), whole.end()));
            if start == end {
                // An empty match splits nothing; search again one character on.
                search = text[start..]
                    .chars()
                    .next()
                    .map_or(text.len(), |c| start + c.len_utf8());
                continue;
            }
            if start > kept {
                pieces.push(text[kept..start].to_string());
            }
            pieces.push(text[start..end].to_string());
            kept = end;
            search = end;
        }
        if kept < text.len() {
            pieces.push(text[kept..].to_string());
        }
        pieces
    }
}

/// Split `pattern` at every `|` outside groups and character classes.
fn top_level_alternatives(pattern: &str) -> Vec<&str> {
    let bytes = pattern.as_bytes();
    let mut alternatives = Vec::new();
    let mut start = 0;
    let mut group_depth = 0usize;
    let mut class_depth = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                i += 2;
                continue;
            }
            b'[' => {
                class_depth += 1;
                // A `]` straight after `[` or `[^` is a member, not the end.
                let mut j = i + 1;
                if bytes.get(j) == Some(&b'^') {
                    j += 1;
                }
                if bytes.get(j) == Some(&b']') {
                    j += 1;
                }
                i = j;
                continue;
            }
            b']' if class_depth > 0 => class_depth -= 1,
            b'(' if class_depth == 0 => group_depth += 1,
            b')' if class_depth == 0 => group_depth = group_depth.saturating_sub(1),
            b'|' if class_depth == 0 && group_depth == 0 => {
                alternatives.push(&pattern[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    alternatives.push(&pattern[start..]);
    alternatives
}

fn refusal(pattern: &str, reason: &str) -> InferenceError {
    InferenceError::Tokenizer(format!(
        "tokenizer.json declares a BPE Split pattern this tokenizer cannot run \
         ({reason}): {pattern:?}"
    ))
}
