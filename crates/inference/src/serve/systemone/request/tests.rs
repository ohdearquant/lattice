use super::super::json;
use super::*;

fn spec(text: &str) -> PyValue {
    json::parse(text).expect("spec parses")
}

#[test]
fn renders_a_choice_with_mixed_descriptions() {
    let question = render_question(&spec(
            r#"{"type": "choice", "instructions": "Pick.", "criteria": {"a": "", "b": null, "c": {"x": 1}}}"#,
        ))
        .expect("renders");
    assert_eq!(question.question, "Pick.");
    assert_eq!(question.options, vec!["a", "b", "c: {\"x\": 1}"]);
    assert_eq!(question.names, vec!["a", "b", "c"]);
}

#[test]
fn renders_a_list_criteria_choice() {
    let question = render_question(&spec(
        r#"{"type": "choice", "instructions": "Pick.", "criteria": ["red", "green", 3]}"#,
    ))
    .expect("renders");
    assert_eq!(question.options, vec!["red", "green", "3"]);
}

#[test]
fn refuses_a_list_criteria_element_that_is_not_a_string_or_integer() {
    let error = render_question(&spec(
        r#"{"type": "choice", "instructions": "Pick.", "criteria": ["a", 2.5]}"#,
    ))
    .expect_err("refused");
    assert!(
        error.to_string().contains("not a string or an integer"),
        "{error}"
    );
}

#[test]
fn renders_a_dict_score_legend_sorted_by_float() {
    let question = render_question(&spec(
            r#"{"type": "score", "instructions": "Rate.", "criteria": {"10": "ten", "2": "two", "0": "zero"}}"#,
        ))
        .expect("renders");
    assert_eq!(question.options, vec!["0: zero", "1: two", "2: ten"]);
    assert_eq!(
        question.legend,
        Some(vec![
            "zero".to_string(),
            "two".to_string(),
            "ten".to_string()
        ])
    );
}

#[test]
fn renders_a_noul_with_descriptions() {
    let question = render_question(&spec(
            r#"{"type": "noul", "instructions": "Is it?", "criteria": {"true": "it is", "false": {"why": "not"}}}"#,
        ))
        .expect("renders");
    assert_eq!(question.kind, QuestionKind::Noul);
    assert_eq!(
        question.options,
        vec!["no: {\"why\": \"not\"}", "yes: it is"]
    );
}

#[test]
fn bool_is_an_alias_for_noul() {
    let question =
        render_question(&spec(r#"{"type": "bool", "instructions": "Is it?"}"#)).expect("renders");
    assert_eq!(question.kind, QuestionKind::Noul);
    assert_eq!(question.options, vec!["no", "yes"]);
}

#[test]
fn question_and_options_are_aliases() {
    let question = render_question(&spec(
        r#"{"question": "Alias field.", "options": ["a", "b"]}"#,
    ))
    .expect("renders");
    assert_eq!(question.question, "Alias field.");
    assert_eq!(question.options, vec!["a", "b"]);
}

#[test]
fn instruction_json_renders_and_null_renders_as_the_word_null() {
    let json_instruction = render_question(&spec(
        r#"{"type": "choice", "instructions": {"task": "pick", "n": 2}, "criteria": ["a", "b"]}"#,
    ))
    .expect("renders");
    assert_eq!(json_instruction.question, "{\"task\": \"pick\", \"n\": 2}");

    let null_instruction = render_question(&spec(
        r#"{"type": "choice", "instructions": null, "criteria": ["a", "b"]}"#,
    ))
    .expect("renders");
    assert_eq!(null_instruction.question, "null");
}

#[test]
fn refuses_the_reference_messages() {
    let cases = [
        (
            r#"{"type": "choice", "instructions": "", "criteria": ["a", "b"]}"#,
            "question without instructions",
        ),
        (
            r#"{"type": "choice", "instructions": "Pick.", "criteria": ["a"]}"#,
            "choice criteria: a map of 2..255 options",
        ),
        (
            r#"{"type": "score", "instructions": "Rate.", "criteria": ["a","b","c","d","e","f","g","h","i","j","k"]}"#,
            "score criteria: an ordered list of 2..10 level descriptions",
        ),
        (
            r#"{"type": "rank", "instructions": "Rank.", "criteria": ["a", "b"]}"#,
            "unknown question type 'rank'",
        ),
    ];
    for (text, expected) in cases {
        let error = render_question(&spec(text)).expect_err(text);
        assert_eq!(error.to_string(), expected, "{text}");
    }
}

#[test]
fn annotates_long_arrays_and_keeps_the_first_index_position() {
    let state = json::parse("[1,2,3,4,5,6,7,8]").expect("state");
    assert_eq!(
        render_state(&state),
        "[{\"_index\": 0, \"value\": 1}, {\"_index\": 1, \"value\": 2}, {\"_index\": 2, \"value\": 3}, {\"_index\": 3, \"value\": 4}, {\"_index\": 4, \"value\": 5}, {\"_index\": 5, \"value\": 6}, {\"_index\": 6, \"value\": 7}, {\"_index\": 7, \"value\": 8}]"
    );

    let short = json::parse("[1,2,3,4,5,6,7]").expect("state");
    assert_eq!(render_state(&short), "[1, 2, 3, 4, 5, 6, 7]");

    let collision = json::parse(r#"[{"_index": "mine", "v": 0},2,3,4,5,6,7,8]"#).expect("state");
    assert!(
        render_state(&collision).starts_with("[{\"_index\": \"mine\", \"v\": 0}"),
        "{}",
        render_state(&collision)
    );
}

#[test]
fn a_string_state_passes_through() {
    assert_eq!(render_state(&PyValue::Str("raw".into())), "raw");
}

#[test]
fn plan_rows_isolates_score_levels() {
    let mut rendered = IndexMap::new();
    rendered.insert(
        "m".to_string(),
        render_question(&spec(
            r#"{"type": "score", "instructions": "M?", "criteria": ["lo", "hi"]}"#,
        ))
        .expect("renders"),
    );
    let (rows, index) = plan_rows(&rendered, true);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].options, vec!["no", "yes"]);
    assert!(
        rows[0]
            .question
            .starts_with("M?\nProposed answer: lo\nDoes the proposed answer fit?")
    );
    assert_eq!(index[0].kind, RowKind::Isolated);
    assert_eq!(index[0].count, 2);
}

#[test]
fn plan_rows_leaves_a_non_isolated_score_as_one_row() {
    let mut rendered = IndexMap::new();
    rendered.insert(
            "q".to_string(),
            render_question(&spec(
                r#"{"type": "score", "instructions": "Rate.", "criteria": ["low", "high"], "isolated": false}"#,
            ))
            .expect("renders"),
        );
    let (rows, index) = plan_rows(&rendered, true);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].options, vec!["0: low", "1: high"]);
    assert_eq!(index[0].kind, RowKind::List);
}

#[test]
fn parse_request_ignores_unknown_fields_and_refuses_non_independent() {
    let request = parse_request(
            r#"{"state": "s", "model": "m", "layout": "state_first", "questions": {"q": {"type": "noul", "instructions": "?"}}}"#,
        )
        .expect("parses");
    assert_eq!(request.questions.len(), 1);
    assert!(request.independent);

    let error = parse_request(r#"{"state": "s", "independent": false, "questions": {}}"#)
        .expect_err("refused");
    assert!(error.to_string().contains("not supported"), "{error}");
}
