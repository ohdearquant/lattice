//! The reply split on hand-written id sequences, with the published marker ids.

use lattice_inference_core::InferenceError;
use lattice_tokenizer::BpeTokenizer;

use crate::reply::{ReplyField, ReplyMarkers, ReplySplit};

const MODEL: u32 = 200_001;
const XML: u32 = 200_024;
const MARKERS: ReplyMarkers = ReplyMarkers {
    end_message: 200_010,
    thinking: 200_008,
    text: 200_004,
    tool_error: 200_022,
    invoke_tool_text: 200_057,
    invoke_tool_json: 200_049,
};
const END: u32 = MARKERS.end_message;
const THINK: u32 = MARKERS.thinking;
const TEXT: u32 = MARKERS.text;
const JSON: u32 = MARKERS.invoke_tool_json;

fn fields(ids: &[u32]) -> Vec<Option<ReplyField>> {
    let mut split = ReplySplit::new(MARKERS);
    ids.iter().map(|&id| split.push(id)).collect()
}

fn of(field: ReplyField, ids: &[u32]) -> Vec<u32> {
    ids.iter()
        .zip(fields(ids))
        .filter(|(_, f)| *f == Some(field))
        .map(|(&id, _)| id)
        .collect()
}

#[test]
fn a_message_after_the_model_marker_and_a_name_is_content() {
    let ids = [MODEL, 100, 101, TEXT, 200, 201, END];
    assert_eq!(of(ReplyField::Content, &ids), [200, 201]);
    assert_eq!(fields(&ids)[..4], [None, None, None, None]);
}

#[test]
fn content_messages_join_with_nothing_between() {
    let ids = [
        MODEL,
        TEXT,
        200,
        END,
        MODEL,
        MARKERS.tool_error,
        201,
        END,
        MODEL,
        MARKERS.invoke_tool_text,
        202,
        203,
        END,
    ];
    assert_eq!(of(ReplyField::Content, &ids), [200, 201, 202, 203]);
}

#[test]
fn a_reply_that_ends_inside_reasoning_has_no_content() {
    let ids = [MODEL, THINK, 300, 301, 302];
    assert_eq!(of(ReplyField::Reasoning, &ids), [300, 301, 302]);
    assert!(of(ReplyField::Content, &ids).is_empty());
}

#[test]
fn a_tool_call_message_is_split_off_from_the_content() {
    let ids = [MODEL, JSON, 400, 401, END, MODEL, TEXT, 200, END];
    assert_eq!(of(ReplyField::ToolCall, &ids), [400, 401]);
    assert_eq!(of(ReplyField::Content, &ids), [200]);
}

#[test]
fn an_unknown_marker_and_tokens_outside_a_message_are_in_no_field() {
    let ids = [
        500, MODEL, XML, 501, END, 502, END, MODEL, TEXT, 200, END, 503,
    ];
    assert_eq!(of(ReplyField::Content, &ids), [200]);
    let kept = fields(&ids).iter().filter(|f| f.is_some()).count();
    assert_eq!(kept, 1);
}

#[test]
fn an_opener_inside_a_body_belongs_to_that_body() {
    let ids = [MODEL, THINK, TEXT, 300, END, MODEL, TEXT, THINK, 200, END];
    assert_eq!(of(ReplyField::Reasoning, &ids), [TEXT, 300]);
    assert_eq!(of(ReplyField::Content, &ids), [THINK, 200]);
}

const NAMES: [&str; 6] = [
    "<|end_message|>",
    "<|content_thinking|>",
    "<|content_text|>",
    "<|content_tool_error|>",
    "<|content_invoke_tool_text|>",
    "<|content_invoke_tool_json|>",
];

fn tokenizer_with(names: &[&str]) -> BpeTokenizer {
    let added: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            format!(
                r#"{{"id": {}, "content": "{name}", "special": true}}"#,
                i + 2
            )
        })
        .collect();
    let json = format!(
        r#"{{"model": {{"type": "BPE", "merges": [], "vocab": {{"a": 0, "b": 1}}}},
            "added_tokens": [{}]}}"#,
        added.join(", ")
    );
    BpeTokenizer::from_tokenizer_json_str(&json).expect("marker tokenizer")
}

#[test]
fn the_markers_resolve_from_the_tokenizer_and_a_missing_one_is_named() {
    let markers = ReplyMarkers::from_tokenizer(&tokenizer_with(&NAMES)).expect("markers");
    assert_eq!(
        markers,
        ReplyMarkers {
            end_message: 2,
            thinking: 3,
            text: 4,
            tool_error: 5,
            invoke_tool_text: 6,
            invoke_tool_json: 7,
        }
    );

    for missing in NAMES {
        let names: Vec<&str> = NAMES.iter().copied().filter(|n| *n != missing).collect();
        match ReplyMarkers::from_tokenizer(&tokenizer_with(&names)) {
            Err(InferenceError::InvalidInput(message)) => {
                assert!(message.contains(missing), "{missing}: {message}")
            }
            Err(other) => panic!("{missing}: expected InvalidInput, got {other}"),
            Ok(_) => panic!("{missing}: a tokenizer without it resolved"),
        }
    }
}
