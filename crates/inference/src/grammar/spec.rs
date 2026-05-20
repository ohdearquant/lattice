//! Grammar specification types for structured output.
//!
//! A `GrammarSpec` is the entry point for callers.  It accepts either a JSON
//! Schema (the primary format for agent tool-call constraints) or a GBNF string
//! (the llama.cpp-compatible escape hatch for custom grammars).  Both are
//! normalised into the same internal grammar representation at
//! `GrammarEngine::new` time.

/// **Unstable**: grammar specification for constrained decoding.
///
/// Pass to [`GrammarEngine::new`](super::engine::GrammarEngine::new) together
/// with the model vocabulary to obtain a `GrammarEngine` that can be embedded
/// in a `GenerateConfig`.
#[derive(Debug, Clone)]
pub enum GrammarSpec {
    /// JSON Schema (draft 2020-12 subset).
    ///
    /// Supported keywords: `type`, `properties`, `required`, `items`,
    /// `prefixItems`, `minItems`, `maxItems`, `enum`, `const`, `anyOf`,
    /// `oneOf`, `$ref`, `$defs`/`definitions` (local only).
    ///
    /// Unsupported (deferred): `pattern`, external `$ref`, `if`/`then`/`else`.
    JsonSchema(serde_json::Value),

    /// GBNF grammar string (llama.cpp-compatible subset).
    ///
    /// The grammar must start with a `root` rule.  Rules are separated by
    /// newlines.  Supported constructs: literals, character ranges `[a-z]`,
    /// concatenation (space-separated), alternation `|`, grouping `( ... )`,
    /// repetition `*`, `+`, `?`.
    Gbnf(String),
}

impl GrammarSpec {
    /// Convenience: build a JSON Schema spec from a JSON string.
    ///
    /// Returns an error if `json` is not valid JSON.
    pub fn json_schema_str(json: &str) -> Result<Self, serde_json::Error> {
        let v = serde_json::from_str(json)?;
        Ok(Self::JsonSchema(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_schema_str_roundtrip() {
        let spec = GrammarSpec::json_schema_str(r#"{"type":"object"}"#).unwrap();
        assert!(matches!(spec, GrammarSpec::JsonSchema(_)));
    }

    #[test]
    fn json_schema_str_invalid() {
        assert!(GrammarSpec::json_schema_str("not json").is_err());
    }

    #[test]
    fn gbnf_variant() {
        let spec = GrammarSpec::Gbnf("root ::= \"hello\"".to_string());
        assert!(matches!(spec, GrammarSpec::Gbnf(_)));
    }

    #[test]
    fn clone_and_debug() {
        let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "string"}));
        let cloned = spec.clone();
        assert!(format!("{cloned:?}").contains("JsonSchema"));
    }
}
