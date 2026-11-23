use super::preparation::{
    Inkling2PromptAdapter, PreparedInkling2Request, prepare_inkling2_chat_request,
};
use super::*;
use crate::serve::ApiError;
use crate::serve::contract::{ChatRequest, NormalizedChatMessage, NormalizedChatRole};
use crate::serve::prompt_adapter::PromptAdapter as _;
use crate::serve::route::INKLING2_STANDALONE_UNSUPPORTED_CODE;
use crate::serving_provider::{SelectionError, inspect_checkpoint, select};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

const MODEL: &str = "inkling2-mini";
const EOS: u32 = 200_006;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../models/inkling2/tests/fixtures/chat_template")
}

fn adapter() -> Inkling2PromptAdapter {
    Inkling2PromptAdapter::new(EOS)
}

fn role(name: &str) -> NormalizedChatRole {
    match name {
        "system" => NormalizedChatRole::System,
        "user" => NormalizedChatRole::User,
        "assistant" => NormalizedChatRole::Assistant,
        other => panic!("unexpected fixture role {other}"),
    }
}

fn manifest_cases() -> Vec<Value> {
    let text = std::fs::read_to_string(fixtures().join("MANIFEST.json")).expect("read manifest");
    let manifest: Value = serde_json::from_str(&text).expect("manifest is JSON");
    manifest["cases"].as_array().expect("cases").clone()
}

fn prepare(body: Value) -> Result<PreparedInkling2Request, ApiError> {
    let req: ChatRequest = serde_json::from_value(body).expect("chat request body");
    prepare_inkling2_chat_request(&adapter(), &req, MODEL, 64, 4096, str::len, || 1 << 20)
}

fn with_messages(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
    });
    for (key, value) in extra.as_object().expect("object") {
        body[key] = value.clone();
    }
    body
}

fn refusal_message(result: Result<PreparedInkling2Request, ApiError>) -> String {
    match result {
        Err(ApiError::BadRequest { message, .. }) => message,
        Err(other) => panic!("expected a BadRequest refusal, got {other:?}"),
        Ok(prepared) => panic!("expected a refusal, got prompt {:?}", prepared.prompt),
    }
}

#[test]
fn the_prompt_matches_every_render_of_the_published_template() {
    let cases = manifest_cases();
    assert_eq!(cases.len(), 7, "the manifest lists seven renders");
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let expected =
            std::fs::read_to_string(fixtures().join(case["file"].as_str().expect("case file")))
                .expect("read rendered fixture");
        assert_eq!(
            expected.len() as u64,
            case["bytes"].as_u64().expect("byte count"),
            "{name}: fixture size differs from the manifest"
        );
        let messages: Vec<NormalizedChatMessage> = case["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|message| NormalizedChatMessage {
                role: role(message["role"].as_str().expect("role")),
                content: message["content"].as_str().expect("content").to_owned(),
                image: None,
            })
            .collect();
        let actual = match case["variables"].get("reasoning_effort") {
            None => adapter().render(&messages),
            Some(effort) => {
                assert_eq!(
                    effort.as_u64(),
                    Some(0),
                    "{name}: only effort 0 is rendered"
                );
                adapter().render_with_thinking(&messages, false)
            }
        };
        assert_eq!(actual, expected, "{name}: rendered prompt differs");
    }
}

#[test]
fn a_request_with_thinking_off_renders_the_effort_zero_template() {
    let messages = json!([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "Line one\nLine two é"},
    ]);
    let off = prepare(with_messages(json!({
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": false},
    })))
    .expect("thinking off is served");
    let expected_off =
        std::fs::read_to_string(fixtures().join("06-system-user-assistant-user-effort0.txt"))
            .expect("read fixture 06");
    assert_eq!(off.prompt, expected_off);

    let default_effort =
        std::fs::read_to_string(fixtures().join("05-system-user-assistant-user.txt"))
            .expect("read fixture 05");
    for extra in [
        json!({"messages": messages}),
        json!({"messages": messages, "chat_template_kwargs": {"enable_thinking": true}}),
    ] {
        let prepared = prepare(with_messages(extra)).expect("default effort is served");
        assert_eq!(prepared.prompt, default_effort);
    }
}

#[test]
fn an_assistant_reasoning_field_and_a_message_name_are_dropped_from_the_prompt() {
    let plain = json!([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "Line one\nLine two é"},
    ]);
    let with_fields = json!([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello", "name": "ana"},
        {"role": "assistant", "content": "Hi.", "reasoning_content": "Greet back."},
        {"role": "user", "content": "Line one\nLine two é"},
    ]);
    let expected = std::fs::read_to_string(fixtures().join("05-system-user-assistant-user.txt"))
        .expect("read fixture 05");
    assert!(!expected.contains("<|content_thinking|>"));
    for messages in [plain, with_fields] {
        let prepared = prepare(with_messages(json!({"messages": messages})))
            .expect("a request with the dropped fields is served");
        assert_eq!(prepared.prompt, expected);
    }
}

#[test]
fn the_generation_config_stops_on_the_end_of_sequence_id_alone() {
    let prepared = prepare(with_messages(json!({
        "max_tokens": 32,
        "temperature": 0.0,
    })))
    .expect("plain request is served");
    assert_eq!(prepared.gen_cfg.stop_token_ids, vec![EOS]);
    assert_eq!(prepared.gen_cfg.max_new_tokens, 32);
    assert!(!prepared.gen_cfg.enable_thinking);
    assert!(prepared.gen_cfg.grammar.is_none());
    assert!(prepared.gen_cfg.stop_strings.is_empty());
    assert_eq!(prepared.gen_cfg.reasoning_budget, None);
    assert_eq!(prepared.gen_cfg.logprobs, None);
}

#[test]
fn each_unsupported_request_field_is_refused_by_name() {
    let cases = [
        (
            "tools",
            json!({"tools": [{"type": "function", "function": {"name": "f"}}]}),
        ),
        ("tool_choice", json!({"tool_choice": "auto"})),
        (
            "role 'tool'",
            json!({"messages": [
                {"role": "user", "content": "a"},
                {"role": "tool", "content": "x", "tool_call_id": "call-1"},
                {"role": "user", "content": "b"},
            ]}),
        ),
        (
            "tool_calls",
            json!({"messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }]},
                {"role": "user", "content": "b"},
            ]}),
        ),
        (
            "typed content parts",
            json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ]}),
        ),
        ("logprobs", json!({"logprobs": true})),
        ("stop", json!({"stop": ["x"]})),
        ("reasoning_budget", json!({"reasoning_budget": 16})),
        (
            "response_format",
            json!({"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {"type": "object"}},
            }}),
        ),
        (
            "response_format",
            json!({"response_format": {"type": "json_object"}}),
        ),
    ];
    for (field, extra) in cases {
        let message = refusal_message(prepare(with_messages(extra.clone())));
        assert!(
            message.contains(field),
            "refusal of {extra} does not name {field:?}: {message}"
        );
    }
}

#[test]
fn the_stop_id_is_the_single_generation_config_end_of_sequence_id() {
    let dir = tempfile::tempdir().expect("temp checkpoint dir");
    let write = |file: &str, text: &str| {
        std::fs::write(dir.path().join(file), text).expect("write checkpoint config");
    };
    write("config.json", r#"{"eos_token_id": 200006}"#);
    write("generation_config.json", r#"{"eos_token_id": 200006}"#);
    assert_eq!(
        Inkling2PromptAdapter::checkpoint_stop_token_id(dir.path()).expect("single id"),
        200_006
    );
    for file in ["generation_config.json", "config.json"] {
        for refused in [
            r#"{"eos_token_id": [200006, 200007]}"#,
            r#"{"bos_token_id": 1}"#,
            r#"{"eos_token_id": -1}"#,
            r#"{"eos_token_id": 4294967296}"#,
        ] {
            write(file, refused);
            assert!(
                Inkling2PromptAdapter::checkpoint_stop_token_id(dir.path()).is_err(),
                "{file} {refused} was accepted"
            );
        }
        write(file, r#"{"eos_token_id": 200006}"#);
    }
    write("generation_config.json", r#"{"eos_token_id": 200001}"#);
    let error = Inkling2PromptAdapter::checkpoint_stop_token_id(dir.path())
        .expect_err("differing end-of-sequence ids are refused")
        .to_string();
    assert!(
        error.contains("200001") && error.contains("200006"),
        "{error}"
    );
}

fn write_minimal_safetensors(path: &Path) {
    let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0u8; 4]);
    std::fs::write(path, bytes).expect("write minimal safetensors");
}

fn inkling2_checkpoint(format: ModelFormat) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp checkpoint dir");
    std::fs::write(
        dir.path().join("config.json"),
        br#"{"model_type":"inkling2_loop"}"#,
    )
    .expect("write config.json");
    match format {
        ModelFormat::Safetensors => {
            write_minimal_safetensors(&dir.path().join("model.safetensors"))
        }
        ModelFormat::Q4 => {
            std::fs::write(dir.path().join("weights.q4"), b"q4").expect("write q4 sentinel");
        }
        other => panic!("no fixture for {other:?}"),
    }
    dir
}

#[test]
fn an_inkling2_safetensors_checkpoint_selects_the_cpu_route() {
    let dir = inkling2_checkpoint(ModelFormat::Safetensors);
    let evidence = inspect_checkpoint(dir.path());
    assert_eq!(evidence.format, ModelFormat::Safetensors);
    let selected = select(evidence, ServingEntry::Lattice).expect("inkling2 is admitted");
    assert_eq!(selected.provider.name(), "inkling2");
    assert_eq!(selected.legacy_route(), Some(ServedRoute::INKLING2_CPU));
    let marker = selected
        .lattice_selection_marker()
        .expect("admitted lattice route has a marker");
    assert_eq!(
        marker,
        ServedRoute::INKLING2_CPU.selection_marker(ModelFormat::Safetensors)
    );
    assert!(marker.contains("family=inkling2"), "{marker}");
    assert!(marker.contains("backend=cpu"), "{marker}");
    let embeddings = selected.implicit_embeddings("m");
    assert_eq!(embeddings.directory_to_try, None);
    assert!(
        embeddings
            .disabled_line
            .as_deref()
            .is_some_and(|line| line.contains("Inkling 2"))
    );
}

#[test]
fn the_standalone_server_refuses_an_inkling2_checkpoint_by_name() {
    let dir = inkling2_checkpoint(ModelFormat::Safetensors);
    let error = select(inspect_checkpoint(dir.path()), ServingEntry::Standalone)
        .err()
        .expect("the standalone server refuses inkling2");
    assert_eq!(
        error,
        SelectionError::Route {
            directory: dir.path().to_path_buf(),
            refusal: RouteRefusal::Inkling2StandaloneUnsupported,
        }
    );
    let message = error.message();
    assert!(
        message.starts_with(INKLING2_STANDALONE_UNSUPPORTED_CODE),
        "{message}"
    );
    assert!(message.contains("lattice serve"), "{message}");
}

#[test]
fn a_q4_inkling2_checkpoint_keeps_the_route_table_refusal() {
    let dir = inkling2_checkpoint(ModelFormat::Q4);
    let evidence = inspect_checkpoint(dir.path());
    assert_eq!(evidence.format, ModelFormat::Q4);
    let expected = select_route(ModelFormat::Q4, ModelFamily::Inkling2)
        .expect_err("the route table has no Q4 inkling2 route");
    assert_eq!(expected, RouteRefusal::UnrecognizedFormat);
    let error = select(evidence, ServingEntry::Lattice)
        .err()
        .expect("Q4 inkling2 is refused");
    assert_eq!(
        error,
        SelectionError::Route {
            directory: dir.path().to_path_buf(),
            refusal: expected,
        }
    );
}
