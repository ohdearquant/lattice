//! Alternatives that start with the same byte must all stay reachable.
//!
//! The matcher keeps every parse consistent with the bytes seen so far instead
//! of committing to the first alternative that consumes a byte. Each row feeds
//! one input through `advance_byte` and checks the final verdict:
//!
//! * `Accept`: every byte is consumed and the state is complete;
//! * `Reject { at_byte }`: the byte at that index cannot continue any parse;
//! * `Incomplete`: every byte is consumed but the input is only a prefix.
//!
//! The first block lists inputs that a single-stack matcher over-rejects (the
//! grammar admits them, so every row must accept). The second block lists
//! inputs the grammar does not admit, which must keep rejecting at the byte
//! where no parse can continue, so that widening the matcher cannot turn into
//! over-acceptance.

use lattice_inference::grammar::gbnf::parse_gbnf;
use lattice_inference::grammar::json_schema::compile_json_schema;
use lattice_inference::grammar::pda::{CompiledGrammar, GrammarState, StepResult, advance_byte};

#[derive(Clone, Copy)]
enum Kind {
    Gbnf,
    Schema,
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Accept,
    Reject { at_byte: usize },
    Incomplete,
    StackLimitExceeded { at_byte: usize },
}

use Kind::{Gbnf, Schema};

fn compile(kind: Kind, spec: &str) -> CompiledGrammar {
    match kind {
        Kind::Gbnf => parse_gbnf(&spec.replace("\\n", "\n"))
            .unwrap_or_else(|e| panic!("GBNF fixture must parse: {e}: {spec}")),
        Kind::Schema => {
            let schema: serde_json::Value = serde_json::from_str(spec)
                .unwrap_or_else(|e| panic!("schema fixture must be JSON: {e}: {spec}"));
            compile_json_schema(&schema)
                .unwrap_or_else(|e| panic!("schema fixture must compile: {e}: {spec}"))
        }
    }
}

fn verdict(grammar: &CompiledGrammar, input: &str) -> Verdict {
    let mut state = GrammarState::initial();
    for (at_byte, &b) in input.as_bytes().iter().enumerate() {
        match advance_byte(&mut state, grammar, b) {
            StepResult::Accepted => {}
            StepResult::Rejected => return Verdict::Reject { at_byte },
            StepResult::StackLimitExceeded => return Verdict::StackLimitExceeded { at_byte },
        }
    }
    if state.is_complete() {
        Verdict::Accept
    } else {
        Verdict::Incomplete
    }
}

fn check(rows: &[(Kind, &str, &str, Verdict)]) {
    let mut failures = Vec::new();
    for (row, (kind, spec, input, expected)) in rows.iter().enumerate() {
        let got = verdict(&compile(*kind, spec), input);
        if &got != expected {
            failures.push(format!(
                "row {row}: spec {spec} input {input}: expected {expected:?}, got {got:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} rows diverged:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

const ALPHA_BETA: &str = r#"{"type":"object","properties":{"alpha":{"type":"integer"},"beta":{"type":"integer"}},"additionalProperties":false}"#;
const ALPHA_ALPS: &str = r#"{"type":"object","properties":{"alpha":{"type":"integer"},"alps":{"type":"integer"}},"additionalProperties":false}"#;
const ALPHA_ALPS_REQUIRED: &str = r#"{"type":"object","properties":{"alpha":{"type":"integer"},"alps":{"type":"integer"}},"required":["alpha","alps"],"additionalProperties":false}"#;
const MAX_RESULTS_TOKENS: &str = r#"{"type":"object","properties":{"max_results":{"type":"integer"},"max_tokens":{"type":"integer"}},"additionalProperties":false}"#;
const NAME_ENUM_OBJECT: &str = r#"{"type":"object","properties":{"name":{"enum":["get_weather","get_time"]}},"required":["name"],"additionalProperties":false}"#;
const TOOL_CALL_NAME_ONLY: &str = r#"{"anyOf":[{"type":"object","properties":{"name":{"const":"get_weather"}},"required":["name"],"additionalProperties":false},{"type":"object","properties":{"name":{"const":"get_time"}},"required":["name"],"additionalProperties":false}]}"#;
const TOOL_CALL_WITH_ARGUMENTS: &str = r#"{"anyOf":[{"type":"object","properties":{"name":{"const":"get_weather"},"arguments":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}},"required":["name","arguments"],"additionalProperties":false},{"type":"object","properties":{"name":{"const":"list_files"},"arguments":{"type":"object","properties":{"dir":{"type":"string"}},"required":["dir"],"additionalProperties":false}},"required":["name","arguments"],"additionalProperties":false}]}"#;
const ANY_OF_A_OR_B: &str = r#"{"anyOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false},{"type":"object","properties":{"b":{"type":"integer"}},"required":["b"],"additionalProperties":false}]}"#;
const INTEGER_OR_NUMBER: &str = r#"{"anyOf":[{"type":"integer"},{"type":"number"}]}"#;
const STRING_OR_OBJECT: &str = r#"{"anyOf":[{"type":"string"},{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}]}"#;
const ONE_OF_KIND: &str = r#"{"oneOf":[{"type":"object","properties":{"kind":{"const":"a"},"x":{"type":"integer"}},"required":["kind","x"],"additionalProperties":false},{"type":"object","properties":{"kind":{"const":"b"},"y":{"type":"integer"}},"required":["kind","y"],"additionalProperties":false}]}"#;
const ARRAY_OF_ENUM_ANY_OF: &str =
    r#"{"type":"array","items":{"anyOf":[{"const":"read"},{"const":"reset"}]}}"#;

#[test]
fn sibling_alternatives_sharing_a_first_byte_accept_every_branch() {
    check(&[
        // GBNF alternations.
        (Gbnf, r#"root ::= "ab" | "ac""#, "ab", Verdict::Accept),
        (Gbnf, r#"root ::= "ab" | "ac""#, "ac", Verdict::Accept),
        // After "a" the first alternative still needs "b" while the second is
        // already complete, so completeness must look past the first stack.
        (Gbnf, r#"root ::= "ab" | "a" "x"?"#, "a", Verdict::Accept),
        (Gbnf, r#"root ::= "a" ("b" | "c")"#, "ac", Verdict::Accept),
        (
            Gbnf,
            r#"root ::= "get_weather" | "get_time""#,
            "get_weather",
            Verdict::Accept,
        ),
        (
            Gbnf,
            r#"root ::= "get_weather" | "get_time""#,
            "get_time",
            Verdict::Accept,
        ),
        (
            Gbnf,
            r#"root ::= "get_" ("weather" | "time")"#,
            "get_time",
            Verdict::Accept,
        ),
        (
            Gbnf,
            r#"root ::= x "c" | x "d"\nx ::= "ab""#,
            "abd",
            Verdict::Accept,
        ),
        (Gbnf, r#"root ::= "foo" | "food""#, "foo", Verdict::Accept),
        (Gbnf, r#"root ::= "foo" | "food""#, "food", Verdict::Accept),
        (Gbnf, r#"root ::= "food" | "foo""#, "foo", Verdict::Accept),
        (Gbnf, r#"root ::= "food" | "foo""#, "food", Verdict::Accept),
        (Gbnf, r#"root ::= "a"? "b""#, "b", Verdict::Accept),
        (Gbnf, r#"root ::= "a"? "b""#, "ab", Verdict::Accept),
        (Gbnf, r#"root ::= ("a" | "b") "c""#, "bc", Verdict::Accept),
        // String enums and consts.
        (
            Schema,
            r#"{"enum":["foo","food"]}"#,
            r#""foo""#,
            Verdict::Accept,
        ),
        (
            Schema,
            r#"{"enum":["foo","food"]}"#,
            r#""food""#,
            Verdict::Accept,
        ),
        (
            Schema,
            r#"{"type":"string","enum":["get_weather","get_time"]}"#,
            r#""get_weather""#,
            Verdict::Accept,
        ),
        (
            Schema,
            r#"{"type":"string","enum":["get_weather","get_time"]}"#,
            r#""get_time""#,
            Verdict::Accept,
        ),
        (
            Schema,
            r#"{"anyOf":[{"const":"ab"},{"const":"ac"}]}"#,
            r#""ac""#,
            Verdict::Accept,
        ),
        (
            Schema,
            NAME_ENUM_OBJECT,
            r#"{"name":"get_time"}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            ARRAY_OF_ENUM_ANY_OF,
            r#"["reset"]"#,
            Verdict::Accept,
        ),
        // Objects whose optional properties share a leading byte.
        (Schema, ALPHA_BETA, r#"{"beta":2}"#, Verdict::Accept),
        (Schema, ALPHA_BETA, r#"{"alpha":1}"#, Verdict::Accept),
        (
            Schema,
            ALPHA_BETA,
            r#"{"alpha":1,"beta":2}"#,
            Verdict::Accept,
        ),
        (Schema, ALPHA_BETA, "{}", Verdict::Accept),
        (Schema, ALPHA_ALPS, r#"{"alpha":1}"#, Verdict::Accept),
        (Schema, ALPHA_ALPS, r#"{"alps":2}"#, Verdict::Accept),
        (
            Schema,
            ALPHA_ALPS,
            r#"{"alpha":1,"alps":2}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            ALPHA_ALPS_REQUIRED,
            r#"{"alpha":1,"alps":2}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            r#"{"type":"object","properties":{"alps":{"type":"integer"},"alpha":{"type":"integer"}},"additionalProperties":false}"#,
            r#"{"alpha":1}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            MAX_RESULTS_TOKENS,
            r#"{"max_tokens":5}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            MAX_RESULTS_TOKENS,
            r#"{"max_results":5,"max_tokens":6}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            r#"{"type":"object","properties":{"max_results":{"type":"integer"},"limit":{"type":"integer"}},"additionalProperties":false}"#,
            r#"{"limit":5}"#,
            Verdict::Accept,
        ),
        // anyOf / oneOf over object and numeric branches.
        (
            Schema,
            TOOL_CALL_NAME_ONLY,
            r#"{"name":"get_weather"}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            TOOL_CALL_NAME_ONLY,
            r#"{"name":"get_time"}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            TOOL_CALL_WITH_ARGUMENTS,
            r#"{"arguments":{"city":"x"},"name":"get_weather"}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            TOOL_CALL_WITH_ARGUMENTS,
            r#"{"arguments":{"dir":"x"},"name":"list_files"}"#,
            Verdict::Accept,
        ),
        (Schema, ANY_OF_A_OR_B, r#"{"a":1}"#, Verdict::Accept),
        (Schema, ANY_OF_A_OR_B, r#"{"b":1}"#, Verdict::Accept),
        (Schema, INTEGER_OR_NUMBER, "1", Verdict::Accept),
        (Schema, INTEGER_OR_NUMBER, "1.5", Verdict::Accept),
        (Schema, INTEGER_OR_NUMBER, "-12e3", Verdict::Accept),
        (Schema, STRING_OR_OBJECT, r#"{"a":1}"#, Verdict::Accept),
        (Schema, STRING_OR_OBJECT, r#""text""#, Verdict::Accept),
        (
            Schema,
            ONE_OF_KIND,
            r#"{"kind":"a","x":1}"#,
            Verdict::Accept,
        ),
        (
            Schema,
            ONE_OF_KIND,
            r#"{"kind":"b","y":1}"#,
            Verdict::Accept,
        ),
    ]);
}

#[test]
fn invalid_inputs_still_reject_where_no_parse_can_continue() {
    check(&[
        // A byte that neither alternative can take.
        (
            Gbnf,
            r#"root ::= "ab" | "ac""#,
            "ad",
            Verdict::Reject { at_byte: 1 },
        ),
        (Gbnf, r#"root ::= "ab" | "ac""#, "a", Verdict::Incomplete),
        (
            Gbnf,
            r#"root ::= "ab" | "ac""#,
            "abc",
            Verdict::Reject { at_byte: 2 },
        ),
        (
            Gbnf,
            r#"root ::= x "c" | x "d"\nx ::= "ab""#,
            "abe",
            Verdict::Reject { at_byte: 2 },
        ),
        (
            Gbnf,
            r#"root ::= "foo" | "food""#,
            "fooe",
            Verdict::Reject { at_byte: 3 },
        ),
        (
            Gbnf,
            r#"root ::= "foo" | "food""#,
            "fo",
            Verdict::Incomplete,
        ),
        (
            Gbnf,
            r#"root ::= "a"? "b""#,
            "aab",
            Verdict::Reject { at_byte: 1 },
        ),
        // Object keys: only declared keys, each at most once, in declared order.
        (
            Schema,
            ALPHA_BETA,
            r#"{"gamma":1}"#,
            Verdict::Reject { at_byte: 2 },
        ),
        (
            Schema,
            ALPHA_BETA,
            r#"{"beta":2,"alpha":1}"#,
            // `beta` is the last declared key, so the comma after its value
            // already has no continuation.
            Verdict::Reject { at_byte: 9 },
        ),
        // Duplicate key: after `alpha` only `beta` may follow.
        (
            Schema,
            ALPHA_BETA,
            r#"{"alpha":1,"alpha":2}"#,
            Verdict::Reject { at_byte: 12 },
        ),
        (
            Schema,
            ALPHA_BETA,
            r#"{"alpha":1,}"#,
            Verdict::Reject { at_byte: 11 },
        ),
        (
            Schema,
            ALPHA_ALPS,
            r#"{"alps":2,"alpha":1}"#,
            Verdict::Reject { at_byte: 9 },
        ),
        (
            Schema,
            ALPHA_ALPS,
            r#"{"alpha":1,"alps":2,}"#,
            Verdict::Reject { at_byte: 19 },
        ),
        (
            Schema,
            ALPHA_ALPS_REQUIRED,
            r#"{"alpha":1}"#,
            Verdict::Reject { at_byte: 10 },
        ),
        // Enums and tool-call branches: a value outside every branch.
        (
            Schema,
            r#"{"enum":["foo","food"]}"#,
            r#""fool""#,
            Verdict::Reject { at_byte: 4 },
        ),
        (
            Schema,
            NAME_ENUM_OBJECT,
            r#"{"name":"get_timee"}"#,
            Verdict::Reject { at_byte: 17 },
        ),
        (
            Schema,
            TOOL_CALL_NAME_ONLY,
            r#"{"name":"get_x"}"#,
            Verdict::Reject { at_byte: 13 },
        ),
        (
            Schema,
            TOOL_CALL_NAME_ONLY,
            r#"{"name":"get_weather","name":"get_time"}"#,
            Verdict::Reject { at_byte: 21 },
        ),
        // Lattice emits required properties in sorted order, so `arguments`
        // must precede `name`. The reordered object is a valid tool call under
        // JSON Schema but is deliberately not generated by this compiler.
        (
            Schema,
            TOOL_CALL_WITH_ARGUMENTS,
            r#"{"name":"list_files","arguments":{"dir":"x"}}"#,
            Verdict::Reject { at_byte: 2 },
        ),
        (
            Schema,
            ANY_OF_A_OR_B,
            r#"{"a":1,"b":1}"#,
            Verdict::Reject { at_byte: 6 },
        ),
        (
            Schema,
            ANY_OF_A_OR_B,
            r#"{"c":1}"#,
            Verdict::Reject { at_byte: 2 },
        ),
        // Numbers: a trailing byte after a complete number, and leading zeros.
        (
            Schema,
            INTEGER_OR_NUMBER,
            "1.5x",
            Verdict::Reject { at_byte: 3 },
        ),
        (
            Schema,
            INTEGER_OR_NUMBER,
            "01",
            Verdict::Reject { at_byte: 1 },
        ),
        (Schema, INTEGER_OR_NUMBER, "1.", Verdict::Incomplete),
        (
            Schema,
            STRING_OR_OBJECT,
            r#"{"b":1}"#,
            Verdict::Reject { at_byte: 2 },
        ),
        (
            Schema,
            ONE_OF_KIND,
            r#"{"kind":"b","x":1}"#,
            Verdict::Reject { at_byte: 13 },
        ),
        (
            Schema,
            ONE_OF_KIND,
            r#"{"kind":"c","y":1}"#,
            Verdict::Reject { at_byte: 9 },
        ),
        (
            Schema,
            ARRAY_OF_ENUM_ANY_OF,
            r#"["rea"]"#,
            Verdict::Reject { at_byte: 5 },
        ),
    ]);
}
