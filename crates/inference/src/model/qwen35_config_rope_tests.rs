use super::Qwen35Config;
use crate::error::InferenceError;

#[derive(Clone, Copy)]
enum Shape {
    Wrapped,
    Flat,
}

impl Shape {
    fn field_prefix(self) -> &'static str {
        match self {
            Self::Wrapped => "text_config.",
            Self::Flat => "",
        }
    }

    fn json(self, flat_theta: Option<&str>, nested: Option<&str>) -> String {
        let mut text = String::from(r#""hidden_size":2048,"num_hidden_layers":4"#);
        if let Some(theta) = flat_theta {
            text.push_str(&format!(r#", "rope_theta":{theta}"#));
        }
        if let Some(parameters) = nested {
            text.push_str(&format!(r#", "rope_parameters":{parameters}"#));
        }
        match self {
            Self::Wrapped => format!(r#"{{"text_config":{{{text}}}}}"#),
            Self::Flat => format!("{{{text}}}"),
        }
    }
}

fn parsed_configs(json: &str) -> [Qwen35Config; 2] {
    [
        Qwen35Config::from_config_json_str(json)
            .unwrap_or_else(|error| panic!("raw parser refused {json}: {error}")),
        Qwen35Config::from_config_json_str_validated(json)
            .unwrap_or_else(|error| panic!("validated parser refused {json}: {error}"))
            .into_inner(),
    ]
}

fn assert_resolved(json: &str, theta: f64, partial_rotary_factor: f32) {
    for config in parsed_configs(json) {
        assert_eq!(config.rope_theta, theta, "wrong resolved theta for {json}");
        assert_eq!(
            config.partial_rotary_factor, partial_rotary_factor,
            "wrong resolved partial factor for {json}"
        );
    }
}

fn assert_rejected(json: &str, expected: &str) {
    for (parser, result) in [
        ("raw", Qwen35Config::from_config_json_str(json).map(|_| ())),
        (
            "validated",
            Qwen35Config::from_config_json_str_validated(json).map(|_| ()),
        ),
    ] {
        let error = result.expect_err(&format!("{parser} parser accepted {json}"));
        let InferenceError::Inference(message) = error else {
            panic!("{parser} parser returned wrong error category for {json}: {error}");
        };
        assert_eq!(message, expected, "wrong {parser} refusal for {json}");
    }
}

fn invalid_theta(theta: f64) -> String {
    format!("invalid Qwen config.json: rope_theta ({theta}) must be finite and > 0.0")
}

fn conflict(shape: Shape, flat: f64, nested: f64) -> String {
    let prefix = shape.field_prefix();
    format!(
        "invalid Qwen config.json: {prefix}rope_theta ({flat}) conflicts with \
         {prefix}rope_parameters.rope_theta ({nested})"
    )
}

#[test]
fn nested_rope_theta_rejects_negative_values() {
    for shape in [Shape::Wrapped, Shape::Flat] {
        for theta in [-1.0, -0.5, -10_000_000.0] {
            let theta_text = theta.to_string();
            let nested = format!(r#"{{"rope_theta":{theta_text}}}"#);
            for flat in [None, Some("0.0"), Some("-0.0"), Some(theta_text.as_str())] {
                let json = shape.json(flat, Some(&nested));
                assert_rejected(&json, &invalid_theta(theta));
            }
            let json = shape.json(Some("12345.0"), Some(&nested));
            assert_rejected(&json, &conflict(shape, 12_345.0, theta));
        }
    }
}

#[test]
fn nested_rope_theta_preserves_zero_sentinels() {
    let nested_cases = [
        (None, 0.25),
        (Some("{}"), 0.25),
        (Some(r#"{"rope_theta":0.0}"#), 0.25),
        (Some(r#"{"rope_theta":-0.0}"#), 0.25),
        (Some(r#"{"partial_rotary_factor":0.5}"#), 0.5),
    ];
    for shape in [Shape::Wrapped, Shape::Flat] {
        for (nested, factor) in nested_cases {
            assert_resolved(&shape.json(None, nested), 10_000_000.0, factor);
            assert_resolved(&shape.json(Some("12345.0"), nested), 12_345.0, factor);
            for (flat, theta) in [("0.0", 0.0), ("-0.0", -0.0), ("-1.0", -1.0)] {
                assert_rejected(&shape.json(Some(flat), nested), &invalid_theta(theta));
            }
        }
    }
}

#[test]
fn nested_rope_theta_preserves_positive_precedence() {
    let nested = r#"{"rope_theta":12345.0}"#;
    for shape in [Shape::Wrapped, Shape::Flat] {
        for flat in [None, Some("0.0"), Some("-0.0"), Some("12345.0")] {
            assert_resolved(&shape.json(flat, Some(nested)), 12_345.0, 0.25);
        }
        for (flat, value) in [("23456.0", 23_456.0), ("-1.0", -1.0)] {
            assert_rejected(
                &shape.json(Some(flat), Some(nested)),
                &conflict(shape, value, 12_345.0),
            );
        }
    }
}

#[test]
fn nested_rope_theta_checkpoint_fixtures_stay_admitted() {
    let fixtures = [
        (
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/qwen35_0_8b_config.json"
            )),
            1024,
            24,
        ),
        (
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/qwen36_config.json"
            )),
            2048,
            40,
        ),
    ];
    for (wrapped, hidden_size, num_hidden_layers) in fixtures {
        let fixture: serde_json::Value = serde_json::from_str(wrapped).unwrap();
        let flat = serde_json::to_string(&fixture["text_config"]).unwrap();
        for json in [wrapped, flat.as_str()] {
            let [raw, validated] = parsed_configs(json);
            for config in [&raw, &validated] {
                assert_eq!(config.rope_theta, 10_000_000.0);
                assert_eq!(config.partial_rotary_factor, 0.25);
                assert_eq!(config.hidden_size, hidden_size);
                assert_eq!(config.num_hidden_layers, num_hidden_layers);
            }
            assert_eq!(raw.rope_theta, validated.rope_theta);
            assert_eq!(raw.partial_rotary_factor, validated.partial_rotary_factor);
            assert_eq!(raw.layer_types, validated.layer_types);
            assert_eq!(raw.rope_dim(), validated.rope_dim());
        }
    }
}
