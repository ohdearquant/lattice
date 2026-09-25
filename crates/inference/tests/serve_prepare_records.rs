//! Observation records for chat-request preparation on both serving routes.
//!
//! Runs a matrix of chat request bodies through each route's handler
//! preparation and records, per body, the effective request values, the
//! `GenerateConfig` the handler builds and the rendered prompt (or the
//! refusal). `records_match_fixture` requires the current code to reproduce
//! the committed records byte for byte, so any change to a default, a
//! refusal, its precedence or the rendered prompt shows up as a diff.
//!
//! Regenerate the fixture only on purpose:
//! `cargo test -p lattice-inference --test serve_prepare_records -- --ignored write_records`.
//! No checkpoint is loaded: token counts come from the rendered prompt's byte
//! length and each configuration names its own context size.
#![cfg(feature = "serve")]

use lattice_inference::GenerateConfig;
use lattice_inference::forward::metal_qwen35::format_chat_template;
use lattice_inference::serve::ApiError;
use lattice_inference::serve::contract::{
    ChatRequest, GenerationDefaults, ServeProfile, normalize_request,
};
use lattice_inference::serve::into_engine_chat_messages;
use lattice_inference::serve::prepare::{build_cfg, lattice_gen_cfg, prepare_chat_request};

const FIXTURE: &str = "tests/fixtures/serve_prepare_records.txt";
const MODEL: &str = "served-model";

/// Chat request bodies. Every body names the served model unless the case is
/// about the model field itself.
const BODIES: &[(&str, &str)] = &[
    // Every option absent.
    (
        "all_absent",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}]}"#,
    ),
    (
        "model_absent",
        r#"{"messages":[{"role":"user","content":"Hi"}]}"#,
    ),
    (
        "options_null",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":null,"temperature":null,"top_p":null,"top_k":null,"repetition_penalty":null,"seed":null,"stream":null,"stop":null,"reasoning_budget":null,"logprobs":null}"#,
    ),
    // Each option explicitly set to its default value.
    (
        "max_tokens_256",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":256}"#,
    ),
    (
        "max_tokens_512",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":512}"#,
    ),
    (
        "max_tokens_100",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":100}"#,
    ),
    (
        "temperature_default",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"temperature":0.7}"#,
    ),
    (
        "top_p_default",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_p":0.9}"#,
    ),
    (
        "top_k_default",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_k":50}"#,
    ),
    (
        "repetition_penalty_default",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"repetition_penalty":1.1}"#,
    ),
    (
        "stream_false",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stream":false}"#,
    ),
    (
        "reasoning_budget_zero",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"reasoning_budget":0}"#,
    ),
    (
        "logprobs_false",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"logprobs":false}"#,
    ),
    (
        "top_logprobs_zero",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"logprobs":true,"top_logprobs":0}"#,
    ),
    // Each option set to a non-default value.
    (
        "max_tokens_32",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}"#,
    ),
    (
        "max_completion_tokens_48",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":48}"#,
    ),
    (
        "max_tokens_pair_equal",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":40,"max_completion_tokens":40}"#,
    ),
    (
        "max_tokens_5000",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":5000}"#,
    ),
    (
        "temperature_zero",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"temperature":0.0}"#,
    ),
    (
        "temperature_high",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"temperature":1.5}"#,
    ),
    (
        "temperature_two",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"temperature":2.0}"#,
    ),
    (
        "top_p_half",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_p":0.5}"#,
    ),
    (
        "top_p_one",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_p":1.0}"#,
    ),
    (
        "top_k_7",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_k":7}"#,
    ),
    (
        "repetition_penalty_125",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"repetition_penalty":1.25}"#,
    ),
    (
        "seed_42",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"seed":42}"#,
    ),
    (
        "stream_true",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stream":true}"#,
    ),
    (
        "logprobs_true",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"logprobs":true}"#,
    ),
    (
        "top_logprobs_5",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"logprobs":true,"top_logprobs":5}"#,
    ),
    (
        "all_non_default",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":64,"temperature":0.3,"top_p":0.8,"top_k":12,"repetition_penalty":1.05,"seed":7,"stream":true,"stop":["\n\n"],"reasoning_budget":16}"#,
    ),
    // The reasoning budget.
    (
        "reasoning_budget_128",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"reasoning_budget":128}"#,
    ),
    (
        "reasoning_budget_large",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":64,"reasoning_budget":100000}"#,
    ),
    (
        "reasoning_budget_negative",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"reasoning_budget":-1}"#,
    ),
    (
        "reasoning_budget_string",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"reasoning_budget":"many"}"#,
    ),
    // Stop strings.
    (
        "stop_string",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stop":"END"}"#,
    ),
    (
        "stop_array",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stop":["a","b"]}"#,
    ),
    (
        "stop_empty_array",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stop":[]}"#,
    ),
    (
        "stop_empty_string",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stop":""}"#,
    ),
    (
        "stop_number",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stop":5}"#,
    ),
    (
        "stop_too_many",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stop":["a","b","c","d","e"]}"#,
    ),
    // Conversation shapes and rendering.
    (
        "system_and_user",
        r#"{"model":"served-model","messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]}"#,
    ),
    (
        "multi_turn",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello."},{"role":"user","content":"Again"}]}"#,
    ),
    (
        "text_parts",
        r#"{"model":"served-model","messages":[{"role":"user","content":[{"type":"text","text":"one "},{"type":"text","text":"two"}]}]}"#,
    ),
    (
        "response_format_text",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"text"}}"#,
    ),
    (
        "response_format_json_schema",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}}"#,
    ),
    // Refusals.
    (
        "empty_messages",
        r#"{"model":"served-model","messages":[]}"#,
    ),
    (
        "wrong_model",
        r#"{"model":"other-model","messages":[{"role":"user","content":"Hi"}]}"#,
    ),
    (
        "tools",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"tools":[]}"#,
    ),
    (
        "tool_choice",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"tool_choice":"auto"}"#,
    ),
    (
        "n_two",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"n":2}"#,
    ),
    (
        "temperature_out_of_range",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"temperature":2.5}"#,
    ),
    (
        "temperature_negative",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"temperature":-0.1}"#,
    ),
    (
        "top_p_zero",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_p":0.0}"#,
    ),
    (
        "top_p_above_one",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_p":1.5}"#,
    ),
    (
        "max_tokens_zero",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":0}"#,
    ),
    (
        "max_tokens_conflict",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
    ),
    (
        "logprobs_with_stream",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"logprobs":true,"stream":true}"#,
    ),
    (
        "top_logprobs_without_logprobs",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_logprobs":3}"#,
    ),
    (
        "top_logprobs_21",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"logprobs":true,"top_logprobs":21}"#,
    ),
    (
        "last_message_assistant",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello."}]}"#,
    ),
    (
        "tool_role",
        r#"{"model":"served-model","messages":[{"role":"tool","content":"42"},{"role":"user","content":"Hi"}]}"#,
    ),
    (
        "unknown_role",
        r#"{"model":"served-model","messages":[{"role":"narrator","content":"Hi"}]}"#,
    ),
    (
        "top_k_string",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_k":"many"}"#,
    ),
    (
        "repetition_penalty_string",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"repetition_penalty":"high"}"#,
    ),
    (
        "image_part",
        r#"{"model":"served-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}"#,
    ),
    (
        "context_exceeded",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":4090}"#,
    ),
    (
        "context_exceeded_by_budget",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":2048,"reasoning_budget":2040}"#,
    ),
    // Precedence between refusals.
    (
        "context_before_stop",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":4090,"stop":5}"#,
    ),
    (
        "wrong_model_and_tools",
        r#"{"model":"other-model","messages":[{"role":"user","content":"Hi"}],"tools":[]}"#,
    ),
    (
        "wrong_model_and_max_tokens_conflict",
        r#"{"model":"other-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
    ),
    (
        "bad_temperature_and_top_p",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"temperature":9.0,"top_p":9.0}"#,
    ),
    (
        "bad_top_p_and_stop",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"top_p":9.0,"stop":5}"#,
    ),
    (
        "bad_stop_and_top_k",
        r#"{"model":"served-model","messages":[{"role":"user","content":"Hi"}],"stop":5,"top_k":"many"}"#,
    ),
];

/// One serving configuration: a route plus the defaults and limits it runs with.
#[derive(Clone, Copy)]
enum Config {
    /// `lattice serve`: `prepare_chat_request` then `lattice_gen_cfg`.
    Lattice {
        name: &'static str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        max_context: usize,
    },
    /// `lattice_serve`: `normalize_request`, `build_cfg`, engine messages,
    /// then the worker's prompt render.
    Daemon {
        name: &'static str,
        defaults: GenerationDefaults,
        model_max_context: usize,
    },
}

const CONFIGS: &[Config] = &[
    Config::Lattice {
        name: "lattice",
        default_max_tokens: 256,
        max_tokens_cap: 4096,
        max_context: 4096,
    },
    // An operator default above the cap refuses every request that omits
    // `max_tokens`, at the position the default is resolved.
    Config::Lattice {
        name: "lattice_default_above_cap",
        default_max_tokens: 8192,
        max_tokens_cap: 4096,
        max_context: 16384,
    },
    Config::Daemon {
        name: "lattice_serve",
        defaults: GenerationDefaults::standard(512),
        model_max_context: 4096,
    },
    // Operator-set defaults, with a small context so the reasoning budget
    // default is clamped.
    Config::Daemon {
        name: "lattice_serve_operator_defaults",
        defaults: GenerationDefaults {
            max_tokens: 100,
            temperature: 0.2,
            top_k: 20,
            top_p: 0.5,
            repetition_penalty: 1.3,
            reasoning_budget: Some(64),
        },
        model_max_context: 160,
    },
    // Out-of-range operator defaults refuse requests that omit the option.
    Config::Daemon {
        name: "lattice_serve_invalid_defaults",
        defaults: GenerationDefaults {
            max_tokens: 100,
            temperature: 3.0,
            top_k: 50,
            top_p: 1.5,
            repetition_penalty: 1.1,
            reasoning_budget: None,
        },
        model_max_context: 4096,
    },
    Config::Daemon {
        name: "lattice_serve_zero_budget_default",
        defaults: GenerationDefaults {
            max_tokens: 0,
            ..GenerationDefaults::standard(0)
        },
        model_max_context: 4096,
    },
];

fn config_name(config: Config) -> &'static str {
    match config {
        Config::Lattice { name, .. } | Config::Daemon { name, .. } => name,
    }
}

fn refusal(err: &ApiError) -> String {
    match err {
        ApiError::BadRequest { message, code } => {
            format!("refused code={code} message={message:?}")
        }
        other => format!("refused other={other:?}"),
    }
}

fn gen_cfg_line(cfg: &GenerateConfig) -> String {
    format!(
        "config: max_new_tokens={} temperature={:?} top_k={} top_p={:?} min_p={:?} \
         repetition_penalty={:?} seed={:?} stop_token_ids={:?} enable_thinking={} \
         enable_mtp={:?} grammar={} stop_strings={:?} reasoning_budget={:?} logprobs={:?}",
        cfg.max_new_tokens,
        cfg.temperature,
        cfg.top_k,
        cfg.top_p,
        cfg.min_p,
        cfg.repetition_penalty,
        cfg.seed,
        cfg.stop_token_ids,
        cfg.enable_thinking,
        cfg.enable_mtp,
        cfg.grammar.is_some(),
        cfg.stop_strings,
        cfg.reasoning_budget,
        cfg.logprobs,
    )
}

fn record(config: Config, case: &str, body: &str, out: &mut String) {
    out.push_str(&format!("== config={} case={case}\n", config_name(config)));
    out.push_str(&format!("body: {body}\n"));
    let req: ChatRequest = match serde_json::from_str(body) {
        Ok(req) => req,
        Err(e) => {
            out.push_str(&format!("parse_error: {e}\n"));
            return;
        }
    };
    match config {
        Config::Lattice {
            default_max_tokens,
            max_tokens_cap,
            max_context,
            ..
        } => match prepare_chat_request(
            &req,
            MODEL,
            default_max_tokens,
            max_tokens_cap,
            false,
            str::len,
            || max_context,
        ) {
            Err(err) => out.push_str(&format!("{}\n", refusal(&err))),
            Ok(prepared) => {
                out.push_str(&format!(
                    "request: messages={:?} max_tokens={} temperature={:?} top_p={:?} \
                     logprobs={:?} stop_strings={:?} reasoning_budget={:?} seed={:?} stream={}\n",
                    prepared.messages,
                    prepared.max_tokens,
                    prepared.temperature,
                    prepared.top_p,
                    prepared.logprobs,
                    prepared.stop_strings,
                    prepared.reasoning_budget,
                    prepared.seed,
                    prepared.stream,
                ));
                out.push_str(&format!("prompt: {:?}\n", prepared.prompt));
                let cfg = lattice_gen_cfg(
                    prepared.max_tokens,
                    prepared.temperature,
                    prepared.top_p,
                    prepared.seed,
                    prepared.stop_strings,
                    prepared.reasoning_budget,
                    prepared.logprobs,
                );
                out.push_str(&gen_cfg_line(&cfg));
                out.push('\n');
            }
        },
        Config::Daemon {
            defaults,
            model_max_context,
            ..
        } => match normalize_request(
            &req,
            defaults,
            ServeProfile::lattice_serve(MODEL, model_max_context).with_vision_support(false),
        ) {
            Err(err) => out.push_str(&format!("{}\n", refusal(&err))),
            Ok(validated) => {
                out.push_str(&format!(
                    "request: messages={:?} max_tokens={} temperature={:?} top_k={} top_p={:?} \
                     repetition_penalty={:?} seed={:?} stream={} stop_strings={:?} \
                     reasoning_budget={:?} logprobs={:?}\n",
                    validated.messages,
                    validated.max_tokens,
                    validated.temperature,
                    validated.top_k,
                    validated.top_p,
                    validated.repetition_penalty,
                    validated.seed,
                    validated.stream,
                    validated.stop_strings,
                    validated.reasoning_budget,
                    validated.logprobs,
                ));
                let cfg = build_cfg(&validated);
                match into_engine_chat_messages(validated.messages) {
                    Ok(messages) => {
                        out.push_str(&format!("prompt: {:?}\n", format_chat_template(&messages)))
                    }
                    Err(err) => out.push_str(&format!("engine_messages: {}\n", refusal(&err))),
                }
                out.push_str(&gen_cfg_line(&cfg));
                out.push('\n');
            }
        },
    }
}

fn all_records() -> String {
    let mut out = String::new();
    for &config in CONFIGS {
        for (case, body) in BODIES {
            record(config, case, body, &mut out);
        }
    }
    out
}

fn fixture_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE)
}

#[test]
#[ignore = "writes the fixture; run explicitly to regenerate it"]
fn write_records() {
    let path = fixture_path();
    std::fs::write(&path, all_records()).expect("fixture must be writable");
    eprintln!(
        "wrote {} records to {}",
        CONFIGS.len() * BODIES.len(),
        path.display()
    );
}

#[test]
fn records_match_fixture() {
    let expected = std::fs::read_to_string(fixture_path()).expect("fixture must exist");
    let actual = all_records();
    let expected_cases = expected.lines().filter(|l| l.starts_with("== ")).count();
    assert_eq!(
        expected_cases,
        CONFIGS.len() * BODIES.len(),
        "fixture case count differs from the matrix"
    );
    if actual != expected {
        let mut diffs = Vec::new();
        let mut case = "";
        for (index, (want, got)) in expected.lines().zip(actual.lines()).enumerate() {
            if want.starts_with("== ") {
                case = want;
            }
            if want != got && diffs.len() < 5 {
                diffs.push(format!(
                    "line {}, {case}\n  expected: {want}\n  actual:   {got}",
                    index + 1
                ));
            }
        }
        panic!(
            "preparation records differ from the fixture ({} vs {} lines):\n{}",
            expected.lines().count(),
            actual.lines().count(),
            diffs.join("\n")
        );
    }
}
