use crate::common::JsonValue;
use base64::Engine as _;
use lattice_inference_core::InferenceError;
use regex::{NoExpand, Regex};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use unicode_segmentation::UnicodeSegmentation;

fn invalid(message: impl std::fmt::Display) -> InferenceError {
    InferenceError::Tokenizer(format!("SentencePiece precompiled normalizer: {message}"))
}

fn validate_regex_pattern(pattern: &str) -> Result<(), InferenceError> {
    let mut chars = pattern.chars();
    let mut incompatible = pattern.contains('^')
        || pattern.contains('$')
        || pattern.contains("(?")
        || pattern.contains("[[:");
    while let Some(ch) = chars.next() {
        if ch == '\\'
            && let Some(escaped) = chars.next()
            && matches!(
                escaped,
                'A' | 'z'
                    | 'Z'
                    | 'G'
                    | 'd'
                    | 'D'
                    | 'w'
                    | 'W'
                    | 's'
                    | 'S'
                    | 'h'
                    | 'H'
                    | 'b'
                    | 'B'
                    | 'R'
                    | 'X'
                    | 'K'
                    | 'p'
                    | 'P'
            )
        {
            incompatible = true;
        }
    }
    if incompatible {
        return Err(invalid(format!("unsupported Replace regex {pattern:?}")));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct CharMap {
    trie: Vec<u32>,
    pool: String,
    sha256: String,
}

impl CharMap {
    fn from_base64(encoded: &str) -> Result<Self, InferenceError> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|error| invalid(format!("invalid base64: {error}")))?;
        let size = bytes
            .get(..4)
            .ok_or_else(|| invalid("missing trie length"))?;
        let trie_bytes = u32::from_le_bytes([size[0], size[1], size[2], size[3]]) as usize;
        if !trie_bytes.is_multiple_of(4) {
            return Err(invalid("trie alignment must be a multiple of four bytes"));
        }
        let end = 4usize
            .checked_add(trie_bytes)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| invalid("trie length exceeds charsmap"))?;
        if trie_bytes == 0 {
            return Err(invalid("trie length must not be zero"));
        }
        let trie = bytes[4..end]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let pool = std::str::from_utf8(&bytes[end..])
            .map_err(|error| invalid(format!("pool UTF-8: {error}")))?
            .to_owned();
        let map = Self {
            trie,
            pool,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        };
        map.validate()?;
        Ok(map)
    }

    fn offset(unit: u32) -> usize {
        ((unit >> 10) << ((unit & (1 << 9)) >> 6)) as usize
    }

    fn validate(&self) -> Result<(), InferenceError> {
        let last_terminator = self.pool.as_bytes().iter().rposition(|&byte| byte == 0);
        let mut pending = vec![Self::offset(self.trie[0])];
        let mut visited = HashSet::new();
        while let Some(base) = pending.pop() {
            if !visited.insert(base) {
                continue;
            }
            if (base | 255) >= self.trie.len() {
                return Err(invalid("trie traversal out of bounds"));
            }
            for byte in 1..=255usize {
                let pos = base ^ byte;
                let unit = self.trie[pos];
                if (unit & (0x8000_0000 | 0xff)) as usize != byte {
                    continue;
                }
                let next = pos ^ Self::offset(unit);
                if unit & (1 << 8) != 0 {
                    let leaf = self
                        .trie
                        .get(next)
                        .ok_or_else(|| invalid("trie traversal leaf out of bounds"))?;
                    let offset = (leaf & 0x7fff_ffff) as usize;
                    self.pool
                        .get(offset..)
                        .ok_or_else(|| invalid("pool offset out of bounds or UTF-8 boundary"))?;
                    if !matches!(last_terminator, Some(last) if offset <= last) {
                        return Err(invalid("pool entry missing NUL terminator"));
                    }
                }
                pending.push(next);
            }
        }
        Ok(())
    }

    fn transform<'a>(&'a self, text: &str) -> Option<&'a str> {
        let mut base = Self::offset(*self.trie.first()?);
        for &byte in text.as_bytes() {
            if byte == 0 {
                break;
            }
            let pos = base ^ usize::from(byte);
            let unit = *self.trie.get(pos)?;
            if (unit & (0x8000_0000 | 0xff)) != u32::from(byte) {
                return None;
            }
            base = pos ^ Self::offset(unit);
            if unit & (1 << 8) != 0 {
                let offset = (*self.trie.get(base)? & 0x7fff_ffff) as usize;
                let tail = self.pool.get(offset..)?;
                let end = tail.as_bytes().iter().position(|&byte| byte == 0)?;
                return tail.get(..end);
            }
        }
        None
    }

    fn normalize(&self, input: &str) -> String {
        let mut output = String::with_capacity(input.len());
        for grapheme in input.graphemes(true) {
            if grapheme.len() < 6
                && let Some(replacement) = self.transform(grapheme)
            {
                output.push_str(replacement);
                continue;
            }
            for character in grapheme.chars() {
                let mut buffer = [0; 4];
                let scalar: &str = character.encode_utf8(&mut buffer);
                output.push_str(self.transform(scalar).unwrap_or(scalar));
            }
        }
        output
    }
}

#[derive(Debug, Clone)]
enum Step {
    Precompiled(CharMap),
    Literal {
        pattern: String,
        content: String,
    },
    Regex {
        pattern: String,
        compiled: Regex,
        content: String,
    },
}

#[derive(Debug, Clone)]
pub(super) struct Pipeline {
    steps: Vec<Step>,
}

fn has_precompiled(value: &JsonValue) -> bool {
    match value.get("type").and_then(JsonValue::as_str) {
        Some("Precompiled") => {
            matches!(value.get("precompiled_charsmap"), Some(value) if !matches!(value, JsonValue::Null))
        }
        Some("Sequence") => value
            .get("normalizers")
            .and_then(JsonValue::as_array)
            .is_some_and(|steps| steps.iter().any(has_precompiled)),
        _ => false,
    }
}

impl Pipeline {
    pub(super) fn from_json(root: &JsonValue) -> Result<Option<Self>, InferenceError> {
        let Some(normalizer) = root.get("normalizer") else {
            return Ok(None);
        };
        if !has_precompiled(normalizer) {
            return Ok(None);
        }
        let mut steps = Vec::new();
        Self::parse_steps(normalizer, &mut steps)?;
        Ok(Some(Self { steps }))
    }

    fn parse_steps(value: &JsonValue, steps: &mut Vec<Step>) -> Result<(), InferenceError> {
        let kind = value
            .get("type")
            .and_then(JsonValue::as_str)
            .unwrap_or("<missing>");
        match kind {
            "Sequence" => {
                let children = value
                    .get("normalizers")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| invalid("Sequence has no normalizers array"))?;
                for child in children {
                    Self::parse_steps(child, steps)?;
                }
            }
            "Precompiled" => {
                let encoded = value
                    .get("precompiled_charsmap")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| invalid("Precompiled charsmap must be a base64 string"))?;
                steps.push(Step::Precompiled(CharMap::from_base64(encoded)?));
            }
            "Replace" => {
                let pattern = value
                    .get("pattern")
                    .ok_or_else(|| invalid("Replace missing pattern"))?;
                let content = value
                    .get("content")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| invalid("Replace missing string content"))?
                    .to_owned();
                if let Some(pattern) = pattern.get("String").and_then(JsonValue::as_str) {
                    steps.push(Step::Literal {
                        pattern: pattern.to_owned(),
                        content,
                    });
                } else if let Some(pattern) = pattern.get("Regex").and_then(JsonValue::as_str) {
                    validate_regex_pattern(pattern)?;
                    let compiled = Regex::new(pattern)
                        .map_err(|error| invalid(format!("Replace regex {pattern:?}: {error}")))?;
                    steps.push(Step::Regex {
                        pattern: pattern.to_owned(),
                        compiled,
                        content,
                    });
                } else {
                    return Err(invalid("Replace requires a String or Regex pattern"));
                }
            }
            other => return Err(invalid(format!("unsupported normalizer step {other:?}"))),
        }
        Ok(())
    }

    pub(super) fn normalize(&self, input: &str) -> String {
        let mut value = input.to_owned();
        for step in &self.steps {
            // HF's pattern matching reports no matches for an empty input.
            if value.is_empty() {
                break;
            }
            value = match step {
                Step::Precompiled(map) => map.normalize(&value),
                Step::Literal { pattern, content } => value.replace(pattern, content),
                Step::Regex {
                    compiled, content, ..
                } => compiled.replace_all(&value, NoExpand(content)).into_owned(),
            };
        }
        value
    }

    pub(super) fn identity_fields(&self) -> Vec<(String, String)> {
        let mut fields = vec![("normalizer.kind".into(), "precompiled-sequence-v1".into())];
        for (index, step) in self.steps.iter().enumerate() {
            let prefix = format!("normalizer.{index}");
            match step {
                Step::Precompiled(map) => {
                    fields.push((format!("{prefix}.kind"), "precompiled".into()));
                    fields.push((format!("{prefix}.charsmap_sha256"), map.sha256.clone()));
                }
                Step::Literal { pattern, content } => {
                    fields.push((format!("{prefix}.kind"), "replace-string".into()));
                    fields.push((format!("{prefix}.pattern"), pattern.clone()));
                    fields.push((format!("{prefix}.content"), content.clone()));
                }
                Step::Regex {
                    pattern, content, ..
                } => {
                    fields.push((format!("{prefix}.kind"), "replace-regex".into()));
                    fields.push((format!("{prefix}.pattern"), pattern.clone()));
                    fields.push((format!("{prefix}.content"), content.clone()));
                }
            }
        }
        fields
    }
}
