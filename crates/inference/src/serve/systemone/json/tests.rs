use super::*;

#[test]
fn python_float_repr_matches_the_seven_reference_values() {
    assert_eq!(python_float_repr(1e16), "1e+16");
    assert_eq!(python_float_repr(1e15), "1000000000000000.0");
    assert_eq!(python_float_repr(1e-5), "1e-05");
    assert_eq!(python_float_repr(0.0001), "0.0001");
    assert_eq!(python_float_repr(-0.0), "-0.0");
    assert_eq!(python_float_repr(2.5e-7), "2.5e-07");
    assert_eq!(
        python_float_repr(1.2345678901234568e17),
        "1.2345678901234568e+17"
    );
}

#[test]
fn python_float_repr_covers_positional_and_scientific_boundaries() {
    assert_eq!(python_float_repr(0.0), "0.0");
    assert_eq!(python_float_repr(1.0), "1.0");
    assert_eq!(python_float_repr(-1.5), "-1.5");
    assert_eq!(python_float_repr(100.0), "100.0");
    assert_eq!(python_float_repr(123456789012345.0), "123456789012345.0");
    assert_eq!(python_float_repr(1e-4), "0.0001");
    assert_eq!(python_float_repr(1e100), "1e+100");
    assert_eq!(python_float_repr(5e-324), "5e-324");
}

#[test]
fn render_escapes_only_python_escapes() {
    let value = PyValue::Str("q\"b\\\n\r\t\u{1}\u{1f}\u{7f}é".to_string());
    assert_eq!(
        render(&value),
        "\"q\\\"b\\\\\\n\\r\\t\\u0001\\u001f\u{7f}é\""
    );
}

#[test]
fn render_uses_default_separators_and_key_order() {
    let mut entries = IndexMap::new();
    entries.insert("b".to_string(), PyValue::Int(1));
    entries.insert("a".to_string(), PyValue::Array(vec![PyValue::Null]));
    let rendered = render(&PyValue::Object(entries));
    assert_eq!(rendered, "{\"b\": 1, \"a\": [null]}");
}

#[test]
fn parse_keeps_first_key_position_and_last_value() {
    let value = parse(r#"{"a": 1, "b": 2, "a": 3}"#).expect("parses");
    let object = value.as_object().expect("object");
    let keys: Vec<&String> = object.keys().collect();
    assert_eq!(keys, vec!["a", "b"]);
    assert_eq!(object.get("a"), Some(&PyValue::Int(3)));
}

#[test]
fn parse_distinguishes_integers_from_floats() {
    assert_eq!(parse("1").expect("int"), PyValue::Int(1));
    assert_eq!(parse("1.0").expect("float"), PyValue::Float(1.0));
    assert_eq!(parse("1e2").expect("float"), PyValue::Float(100.0));
    assert_eq!(
        parse("12345678901234567890").expect("big int"),
        PyValue::Int(12345678901234567890_i128)
    );
}

#[test]
fn parse_refuses_what_cannot_be_rendered_identically() {
    for body in ["NaN", "Infinity", "-Infinity", "1e400", "1.0e309"] {
        let error = parse(body).expect_err(body);
        assert!(
            matches!(error, SystemOneError::Refused(_)),
            "{body}: {error:?}"
        );
    }
    assert!(parse("170141183460469231731687303715884105728").is_err());
    assert!(parse("\"\\ud800\"").is_err());
    assert!(parse("\"\\udc00\"").is_err());
}

#[test]
fn parse_accepts_surrogate_pairs_and_control_escapes() {
    assert_eq!(
        parse("\"\\ud83d\\ude00\"").expect("pair"),
        PyValue::Str("😀".to_string())
    );
    assert_eq!(
        parse("\"\\u0000\"").expect("nul"),
        PyValue::Str("\0".to_string())
    );
}

#[test]
fn render_round_trips_big_integers_without_a_fraction() {
    assert_eq!(
        render(&PyValue::Int(12345678901234567890)),
        "12345678901234567890"
    );
}
