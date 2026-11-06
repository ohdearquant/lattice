use lattice_grammar::json_schema::compile_json_schema;
use serde_json::{Value, json};

fn assert_overlap(left: Value, right: Value) {
    assert_ne!(left.to_string(), right.to_string());
    let mut branches = vec![json!({"const": left}), json!({"const": right})];
    for _ in 0..2 {
        let error = compile_json_schema(&json!({"oneOf": branches}))
            .expect_err("mathematically equal composite literals must overlap");
        assert!(
            error
                .0
                .contains("oneOf branch 1 overlaps an earlier branch"),
            "expected the overlap diagnostic, got: {error}"
        );
        branches.reverse();
    }
}

#[test]
fn oneof_array_signed_zero_literals_overlap() {
    let left = json!([-0.0]);
    let right = json!([0.0]);
    assert!(left[0].is_f64() && right[0].is_f64());
    assert!(left[0].as_f64().unwrap().is_sign_negative());
    assert!(right[0].as_f64().unwrap().is_sign_positive());
    assert_eq!(left, right);
    assert_overlap(left, right);
}

#[test]
fn oneof_object_signed_zero_literals_overlap() {
    let left = json!({"x": -0.0});
    let right = json!({"x": 0.0});
    assert!(left["x"].is_f64() && right["x"].is_f64());
    assert!(left["x"].as_f64().unwrap().is_sign_negative());
    assert!(right["x"].as_f64().unwrap().is_sign_positive());
    assert_eq!(left, right);
    assert_overlap(left, right);
}

#[test]
fn oneof_integer_float_composite_literals_overlap() {
    for (left, right) in [
        (json!([0]), json!([0.0])),
        (json!([1]), json!([1.0])),
        (json!({"x": 0}), json!({"x": 0.0})),
        (json!({"x": 1}), json!({"x": 1.0})),
    ] {
        assert_ne!(left, right, "serde_json storage equality is not the oracle");
        assert_overlap(left, right);
    }
}

#[test]
fn oneof_nested_numeric_literals_overlap() {
    assert_overlap(
        json!({"outer": [{"x": 1, "zero": -0.0}], "other": [2, 2.0]}),
        json!({"other": [2.0, 2], "outer": [{"zero": 0, "x": 1.0}]}),
    );
}

#[test]
fn oneof_distinct_composite_literals_still_compile() {
    for (left, right) in [
        (json!([false, null]), json!([null, false])),
        (json!({"x": -0.0}), json!({"y": 0.0})),
        (json!([1]), json!([2.0])),
        (json!([1]), json!([1.0, 1.0])),
        (json!([1, 2]), json!([2.0, 1.0])),
        (json!({"x": 1}), json!({"x": 1.0, "y": null})),
        (json!({"x": [1]}), json!({"x": [2.0]})),
        (json!({"x": 1}), json!({"x": "1"})),
    ] {
        assert_ne!(left, right);
        let mut branches = vec![json!({"const": left}), json!({"const": right})];
        for _ in 0..2 {
            compile_json_schema(&json!({"oneOf": branches}))
                .expect("distinct composite values must remain distinct");
            branches.reverse();
        }
    }
}
