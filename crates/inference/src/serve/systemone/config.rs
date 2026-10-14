//! The checkpoint's `decider_config.json` allowlist.
//!
//! The checkpoint's own config file is the one source of truth for selecting
//! this port. A port names the `version` values it implements and the exact set
//! of fields that version carries, refuses any other version or field with a
//! message naming it, and checks the values this layout depends on instead of
//! assuming them.

use super::SystemOneError;

/// Every field the `0.8b-v1` checkpoint config carries. Any other field is
/// refused at load so a config this port does not understand cannot silently
/// select it.
const KNOWN_FIELDS_0_8B_V1: &[&str] = &[
    "temperature",
    "temperature_schema_first",
    "neutralize_none",
    "version",
    "base",
    "max_options",
    "max_state_tokens",
    "schema_first",
    "schema_first_trained",
    "isolated_levels",
    "recipe",
    "release_date",
];

/// The `version` values this port implements.
const SUPPORTED_VERSIONS: &[&str] = &["0.8b-v1"];

/// The layout facts this port depends on for `0.8b-v1`, validated at load.
const REQUIRED_MAX_OPTIONS_0_8B_V1: i128 = 255;
const REQUIRED_NEUTRALIZE_NONE_0_8B_V1: bool = false;
const REQUIRED_SCHEMA_FIRST_0_8B_V1: bool = false;

/// A validated `decider_config.json` with the fields this port reads.
#[derive(Debug, Clone, PartialEq)]
pub struct DeciderConfig {
    /// The checkpoint's `version`.
    pub version: String,
    /// The sampling temperature this checkpoint publishes, carried for the
    /// answer read-out (applied above this port).
    pub temperature: f64,
    /// The truncation length for the state prefix, in tokens.
    pub max_state_tokens: usize,
    /// The maximum option count for a `choice` question.
    pub max_options: usize,
    /// Whether a `score` question's levels are judged one per row.
    pub isolated_levels: bool,
}

/// Parse and validate the checkpoint config text.
pub fn parse_decider_config(text: &str) -> Result<DeciderConfig, SystemOneError> {
    let root: serde_json::Value = serde_json::from_str(text).map_err(|error| {
        SystemOneError::Refused(format!("decider config is not valid JSON: {error}"))
    })?;
    let object = root.as_object().ok_or_else(|| {
        SystemOneError::Refused("decider config must be a JSON object".to_string())
    })?;

    for key in object.keys() {
        if !KNOWN_FIELDS_0_8B_V1.contains(&key.as_str()) {
            return Err(SystemOneError::Refused(format!(
                "decider config field {key:?} is not part of 0.8b-v1"
            )));
        }
    }

    let version = object
        .get("version")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            SystemOneError::Refused("decider config is missing the string field \"version\"".into())
        })?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(SystemOneError::Refused(format!(
            "unsupported decider config version {version:?}"
        )));
    }

    if object
        .get("schema_first")
        .and_then(serde_json::Value::as_bool)
        != Some(REQUIRED_SCHEMA_FIRST_0_8B_V1)
    {
        return Err(SystemOneError::Refused(
            "decider config: this port implements the state-first layout only, so \
             \"schema_first\" must be false"
                .to_string(),
        ));
    }

    let max_options = object
        .get("max_options")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            SystemOneError::Refused(
                "decider config is missing the integer field \"max_options\"".into(),
            )
        })?;
    if i128::from(max_options) != REQUIRED_MAX_OPTIONS_0_8B_V1 {
        return Err(SystemOneError::Refused(format!(
            "decider config: max_options must be {REQUIRED_MAX_OPTIONS_0_8B_V1}, found {max_options}"
        )));
    }

    if object
        .get("neutralize_none")
        .and_then(serde_json::Value::as_bool)
        != Some(REQUIRED_NEUTRALIZE_NONE_0_8B_V1)
    {
        return Err(SystemOneError::Refused(
            "decider config: neutralization is not part of this port, so \
             \"neutralize_none\" must be false"
                .to_string(),
        ));
    }

    let max_state_tokens = object
        .get("max_state_tokens")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            SystemOneError::Refused(
                "decider config is missing the positive integer field \"max_state_tokens\"".into(),
            )
        })?;

    let temperature = object
        .get("temperature")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            SystemOneError::Refused(
                "decider config is missing the number field \"temperature\"".into(),
            )
        })?;

    let isolated_levels = object
        .get("isolated_levels")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            SystemOneError::Refused(
                "decider config is missing the boolean field \"isolated_levels\"".into(),
            )
        })?;

    Ok(DeciderConfig {
        version: version.to_string(),
        temperature,
        max_state_tokens,
        max_options: max_options as usize,
        isolated_levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{
        "temperature": 1.03,
        "temperature_schema_first": 1.03,
        "neutralize_none": false,
        "version": "0.8b-v1",
        "base": "Qwen/Qwen3.5-0.8B-Base",
        "max_options": 255,
        "max_state_tokens": 32768,
        "schema_first": false,
        "schema_first_trained": true,
        "isolated_levels": true,
        "recipe": "scripts/train.sh full (single run)",
        "release_date": "2026-09-19"
    }"#;

    #[test]
    fn accepts_the_checkpoint_config() {
        let config = parse_decider_config(GOOD).expect("parses");
        assert_eq!(config.version, "0.8b-v1");
        assert_eq!(config.max_state_tokens, 32768);
        assert_eq!(config.max_options, 255);
        assert!(config.isolated_levels);
        assert!((config.temperature - 1.03).abs() < 1e-12);
    }

    #[test]
    fn refuses_an_unknown_version() {
        let text = GOOD.replace("\"0.8b-v1\"", "\"2b-v3\"");
        let error = parse_decider_config(&text).expect_err("refused");
        assert!(error.to_string().contains("2b-v3"), "{error}");
    }

    #[test]
    fn refuses_an_unknown_field() {
        let text = GOOD.replace("\"release_date\"", "\"temperature_by_type\"");
        let error = parse_decider_config(&text).expect_err("refused");
        assert!(error.to_string().contains("temperature_by_type"), "{error}");
    }

    #[test]
    fn refuses_a_layout_value_it_does_not_implement() {
        let schema_first = GOOD.replace("\"schema_first\": false", "\"schema_first\": true");
        assert!(parse_decider_config(&schema_first).is_err());
        let neutralize = GOOD.replace("\"neutralize_none\": false", "\"neutralize_none\": true");
        assert!(parse_decider_config(&neutralize).is_err());
        let max_options = GOOD.replace("\"max_options\": 255", "\"max_options\": 128");
        assert!(parse_decider_config(&max_options).is_err());
    }

    #[test]
    fn refuses_a_missing_required_field() {
        let text = GOOD.replace("\"max_state_tokens\": 32768,", "");
        let error = parse_decider_config(&text).expect_err("refused");
        assert!(error.to_string().contains("max_state_tokens"), "{error}");
    }
}
