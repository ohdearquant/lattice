#![cfg(feature = "serve")]

mod common;

use base64::Engine as _;
use common::{Binary, Server, require_checkpoint, response_text};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead as _, Read as _, Write as _};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier, Mutex, PoisonError};
use std::time::Duration;

static CPU_CELL_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn lattice_qwen_cpu_concurrent_greedy_requests_are_isolated_and_stream_disconnect_recovers() {
    let _cell = CPU_CELL_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(model) =
        require_checkpoint(common::QWEN_DIR_ENV, "qwen3.5-0.8b", "cpu-isolation-qwen")
    else {
        return;
    };
    report_capture_identity(&model, "qwen35");
    let server = Server::spawn(Binary::Lattice, &model, &[]);
    observe_http_characterization(&server, "qwen35");
    assert_concurrent_greedy_results(
        &server,
        "Qwen CPU cell",
        16,
        &[
            "Reply with only the word ALPHA.",
            "Reply with only the word BRAVO.",
            "Reply with only the word CHARLIE.",
            "Reply with only the word DELTA.",
        ],
    );
    assert_stream_overlap_and_recovery(
        &server,
        "qwen35",
        "Reply with only the word STABLE.",
        "Reply with only the word NEXT.",
    );
}

#[test]
fn lattice_gemma_cpu_two_concurrent_greedy_requests_are_isolated() {
    let _cell = CPU_CELL_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(model) = require_checkpoint(
        common::GEMMA_DIR_ENV,
        "gemma-4-e2b-it",
        "cpu-isolation-gemma",
    ) else {
        return;
    };
    report_capture_identity(&model, "gemma4");
    let server = Server::spawn(Binary::Lattice, &model, &[]);
    observe_http_characterization(&server, "gemma4");
    assert_concurrent_greedy_results(
        &server,
        "Gemma CPU cell",
        12,
        &[
            "Reply with only the word KIWI.",
            "Reply with only the word TULIP.",
        ],
    );
    assert_stream_overlap_and_recovery(
        &server,
        "gemma4",
        "Reply with only the word ORCHID.",
        "Reply with only the word MAPLE.",
    );
}

fn assert_stream_overlap_and_recovery(
    server: &Server,
    family: &str,
    survivor_prompt: &str,
    next_prompt: &str,
) {
    let answer = |prompt: &str| {
        let body = server.chat_body(json!({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 12,
            "temperature": 0.0,
            "seed": 37,
            "stream": false,
        }));
        let response = request_json(server, &body);
        assert_eq!(
            response.status, 200,
            "{family} solo {prompt}: {}",
            response.body
        );
        response_text_from_wire(&response)
    };
    let solo_survivor = answer(survivor_prompt);
    let solo_next = answer(next_prompt);
    let expected = match family {
        "qwen35" => (
            "<think>\n\n</think>\n\nSTABLE",
            "<think>\n\n</think>\n\nNEXT",
        ),
        "gemma4" => ("ORCHID", "MAPLE"),
        _ => panic!("unknown CPU family {family}"),
    };
    assert_eq!(solo_survivor, expected.0, "{family} solo survivor golden");
    assert_eq!(solo_next, expected.1, "{family} solo next-request golden");

    let stream_body = server.chat_body(json!({
        "messages": [{
            "role": "user",
            "content": "Continue the sequence for as long as possible: 1, 2, 3, 4"
        }],
        "max_tokens": 256,
        "temperature": 0.0,
        "seed": 53,
        "stream": true,
    }));
    let (stream, status, received) = open_stream_response(server, &stream_body);
    assert_eq!(status, 200, "{family} stream status");
    assert!(received > 0, "{family} stream produced response bytes");

    let survivor_body = server.chat_body(json!({
        "messages": [{"role": "user", "content": survivor_prompt}],
        "max_tokens": 12,
        "temperature": 0.0,
        "seed": 37,
        "stream": false,
    }));
    let port = server.port();
    let survivor = std::thread::spawn(move || common::request_at_port(port, &survivor_body));
    std::thread::sleep(Duration::from_millis(100));
    stream
        .get_ref()
        .shutdown(std::net::Shutdown::Both)
        .expect("disconnect streaming request");
    let concurrent = survivor.join().expect("surviving non-stream request joins");
    assert_eq!(
        concurrent.status, 200,
        "{family} concurrent survivor: {}",
        concurrent.body
    );
    assert_eq!(
        response_text(&concurrent),
        solo_survivor,
        "{family} non-stream answer changed while a stream was cancelled"
    );

    let next_body = server.chat_body(json!({
        "messages": [{"role": "user", "content": next_prompt}],
        "max_tokens": 12,
        "temperature": 0.0,
        "seed": 37,
        "stream": false,
    }));
    let next = request_json(server, &next_body);
    assert_eq!(next.status, 200, "{family} next request: {}", next.body);
    assert_eq!(
        response_text_from_wire(&next),
        solo_next,
        "{family} answer after stream disconnect changed"
    );
    eprintln!(
        "CPU overlap family={family} disconnect_status={status} received_bytes={received} survivor={solo_survivor:?} next={solo_next:?}"
    );
}

fn open_stream_response(
    server: &Server,
    body: &Value,
) -> (std::io::BufReader<TcpStream>, u16, usize) {
    let mut stream = TcpStream::connect(("127.0.0.1", server.port())).expect("connect SSE route");
    stream
        .set_read_timeout(Some(Duration::from_secs(480)))
        .expect("set SSE read timeout");
    stream
        .set_write_timeout(Some(Duration::from_secs(480)))
        .expect("set SSE write timeout");
    let payload = serde_json::to_vec(body).expect("SSE request serializes");
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|()| stream.write_all(&payload))
        .expect("send SSE request");
    let mut reader = std::io::BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).expect("read SSE status line");
    let status = line
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u16>().ok())
        .expect("SSE status is numeric");
    loop {
        line.clear();
        reader
            .read_line(&mut line)
            .expect("read SSE response headers");
        if line == "\r\n" || line.is_empty() {
            break;
        }
    }
    let mut first_bytes = [0u8; 32];
    let received = reader
        .read(&mut first_bytes)
        .expect("read SSE response bytes");
    (reader, status, received)
}

fn response_text_from_wire(response: &WireResponse) -> String {
    serde_json::from_str::<Value>(&response.body).expect("completion response is JSON")["choices"]
        [0]["message"]["content"]
        .as_str()
        .expect("completion has message content")
        .to_owned()
}

fn observe_http_characterization(server: &Server, family: &str) {
    observe_transcripts(server, family);
    observe_refusal_matrix(server, family);
    observe_competing_errors(server, family);
    observe_raw_http_errors(server, family);
    observe_logprobs(server, family);
}

fn observe_transcripts(server: &Server, family: &str) {
    for streaming in [false, true] {
        let mode = if streaming { "stream" } else { "nonstream" };
        let body = server.chat_body(json!({
            "messages": [{"role": "user", "content": "Reply with exactly the word amber."}],
            "max_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 41,
            "stream": streaming,
        }));
        let first = request_json(server, &body);
        let second = request_json(server, &body);
        let first_transcript = transcript(&first, streaming);
        let second_transcript = transcript(&second, streaming);
        assert_eq!(
            first_transcript, second_transcript,
            "{family} {mode} repeat"
        );
        assert_eq!(
            first_transcript,
            expected_transcript(family, streaming),
            "{family} {mode} pre-migration HTTP transcript"
        );
        eprintln!("CPU HTTP transcript family={family} mode={mode} {first_transcript}");
    }
}

fn expected_transcript(family: &str, streaming: bool) -> Value {
    let (content, finish_reason, prompt_tokens, completion_tokens, stream_deltas) = match family {
        "qwen35" => ("<think>\n\n", "length", 15, 2, vec!["<think>", "\n\n"]),
        "gemma4" => ("amber", "stop", 16, 1, vec!["amber"]),
        _ => panic!("unknown CPU family {family}"),
    };
    let total_tokens = prompt_tokens + completion_tokens;
    if streaming {
        let mut events = vec![json!({
            "id": "<normalized-id>",
            "created": 0,
            "model": "characterization-model",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}]
        })];
        events.extend(stream_deltas.iter().map(|delta| {
            json!({
                "id": "<normalized-id>",
                "created": 0,
                "model": "characterization-model",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": delta}}]
            })
        }));
        events.push(json!({
            "id": "<normalized-id>",
            "created": 0,
            "model": "characterization-model",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
        }));
        events.push(json!("[DONE]"));
        json!({
            "status": 200,
            "content_type": "text/event-stream",
            "events": events,
            "content_deltas_joined": content,
        })
    } else {
        json!({
            "status": 200,
            "content_type": "application/json",
            "body": {
                "id": "<normalized-id>",
                "created": 0,
                "model": "characterization-model",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
        })
    }
}

fn observe_refusal_matrix(server: &Server, family: &str) {
    let mut observations = Map::new();
    for streaming in [false, true] {
        let mode = if streaming { "stream" } else { "nonstream" };
        let mut mode_results = Map::new();
        for (name, mut body) in refusal_probes(server) {
            body["stream"] = json!(streaming);
            let response = request_json(server, &body);
            let observed = if response.status == 200 {
                success_contract(&response)
            } else {
                wire_contract(&response)
            };
            assert_eq!(
                observed,
                expected_refusal_contract(family, name, streaming),
                "{family} {mode} refusal probe {name}"
            );
            if response.status != 200 {
                mode_results.insert(name.to_owned(), observed);
            } else {
                assert_success_contract(&response, streaming, family, mode, name);
                mode_results.insert(name.to_owned(), observed);
            }
        }
        observations.insert(mode.to_owned(), Value::Object(mode_results));
    }
    eprintln!(
        "CPU refusal matrix family={family} {}",
        Value::Object(observations)
    );
}

fn expected_refusal_contract(family: &str, name: &str, streaming: bool) -> Value {
    let stream_error = (
        "unsupported_feature",
        "logprobs is not supported together with stream: true",
    );
    let error = match name {
        "image_content_part" => Some((
            "vision_unsupported",
            "image input requires a vision-capable model",
        )),
        "json_object_before_zero_max_tokens" | "response_format_json_object" => Some((
            "unsupported_feature",
            "response_format.type 'json_object' is not supported; use 'text'",
        )),
        "response_format_json_schema" => Some((
            "unsupported_feature",
            "response_format.type 'json_schema' is not supported; use 'text'",
        )),
        "lora" => Some((
            "lora_unsupported_backend",
            "this server was built without Metal support; runtime LoRA adapters require a macOS Metal build",
        )),
        "max_tokens_over_cap" => Some((
            "max_tokens_exceeds_limit",
            "max_tokens 999999 exceeds server limit 4096",
        )),
        "n_before_zero_max_tokens" | "n_two" => {
            Some(("unsupported_feature", "n > 1 is not supported"))
        }
        "reasoning_budget" if family == "gemma4" => Some((
            "unsupported_feature",
            "reasoning_budget is not supported for this model",
        )),
        "stop" if family == "gemma4" => Some((
            "unsupported_feature",
            "stop is not supported for this model",
        )),
        "tools" | "tools_before_zero_max_tokens" => Some((
            "unsupported_feature",
            "tools and tool_choice are not supported by this server",
        )),
        "typed_text_content_parts" if family == "gemma4" => Some((
            "unsupported_feature",
            "typed content parts are not supported for this model; send message content as a string",
        )),
        "logprobs_top_logprobs" if streaming => Some(stream_error),
        "logprobs_top_logprobs" if family == "gemma4" => Some((
            "unsupported_feature",
            "logprobs are not supported for this model",
        )),
        _ => None,
    };
    error.map_or_else(
        || expected_success_contract(streaming),
        |(code, message)| expected_error_contract(400, code, message),
    )
}

fn expected_success_contract(streaming: bool) -> Value {
    success_contract_fields(
        200,
        if streaming {
            "text/event-stream"
        } else {
            "application/json"
        },
    )
}

fn success_contract(response: &WireResponse) -> Value {
    success_contract_fields(response.status, content_type(response))
}

fn success_contract_fields(status: u16, content_type: &str) -> Value {
    json!({"status": status, "content_type": content_type})
}

fn expected_error_contract(status: u16, code: &str, message: &str) -> Value {
    json!({
        "status": status,
        "content_type": "application/json",
        "body": {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
                "param": null
            }
        }
    })
}

fn assert_success_contract(
    response: &WireResponse,
    streaming: bool,
    family: &str,
    mode: &str,
    name: &str,
) {
    assert_eq!(response.status, 200, "{family} {mode} {name}");
    if streaming {
        let body = transcript(response, true);
        let events = body["events"].as_array().expect("SSE events are an array");
        assert!(
            events
                .first()
                .is_some_and(|event| { event["choices"][0]["delta"]["role"] == "assistant" }),
            "{family} {mode} {name} starts with the role event: {body}"
        );
        assert_eq!(events.last(), Some(&json!("[DONE]")), "{body}");
    } else {
        assert!(
            serde_json::from_str::<Value>(&response.body).is_ok(),
            "{family} {mode} {name} success body is JSON"
        );
    }
}

fn refusal_probes(server: &Server) -> Vec<(&'static str, Value)> {
    let png = small_png_base64();
    let json_schema = json!({
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": true,
                "schema": {"type": "string"}
            }
        }
    });
    let tools = json!({
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}}
        }]
    });
    let mut over_cap = json!({});
    over_cap["max_tokens"] = json!(999_999);
    let mut tools_before_zero = tools.clone();
    tools_before_zero["max_tokens"] = json!(0);
    let n_before_zero = json!({"n": 2, "max_tokens": 0});
    let json_object_before_zero = json!({
        "response_format": {"type": "json_object"},
        "max_tokens": 0
    });
    vec![
        ("plain", server.chat_body(json!({}))),
        ("stop", server.chat_body(json!({"stop": ["never-match"]}))),
        (
            "logprobs_top_logprobs",
            server.chat_body(json!({"logprobs": true, "top_logprobs": 2})),
        ),
        ("response_format_json_schema", server.chat_body(json_schema)),
        (
            "response_format_json_object",
            server.chat_body(json!({"response_format": {"type": "json_object"}})),
        ),
        (
            "lora",
            server.chat_body(json!({"lora": [{"id": 7, "scale": 1.0}]})),
        ),
        (
            "reasoning_budget",
            server.chat_body(json!({"reasoning_budget": 2})),
        ),
        (
            "image_content_part",
            server.chat_body(json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image_url", "image_url": {
                            "url": format!("data:image/png;base64,{png}")
                        }}
                    ]
                }]
            })),
        ),
        ("n_two", server.chat_body(json!({"n": 2}))),
        ("tools", server.chat_body(tools)),
        ("max_tokens_over_cap", server.chat_body(over_cap)),
        (
            "typed_text_content_parts",
            server.chat_body(json!({
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "Say ok."}]
                }]
            })),
        ),
        (
            "tools_before_zero_max_tokens",
            server.chat_body(tools_before_zero),
        ),
        ("n_before_zero_max_tokens", server.chat_body(n_before_zero)),
        (
            "json_object_before_zero_max_tokens",
            server.chat_body(json_object_before_zero),
        ),
    ]
}

fn small_png_base64() -> String {
    let image = image::RgbImage::from_pixel(32, 32, image::Rgb([12, 34, 56]));
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode small image fixture");
    base64::engine::general_purpose::STANDARD.encode(png.into_inner())
}

fn observe_competing_errors(server: &Server, family: &str) {
    let png = small_png_base64();
    let long_prompt = "x ".repeat(100_000);
    let mut wrong_model_and_format = server.chat_body(json!({
        "response_format": {"type": "json_object"}
    }));
    wrong_model_and_format["model"] = json!("wrong-model");
    let with_image = json!({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {
                    "url": format!("data:image/png;base64,{png}")
                }}
            ]
        }],
        "lora": [{"id": 7, "scale": 1.0}]
    });
    let context_and_stop = json!({
        "messages": [{"role": "user", "content": long_prompt}],
        "stop": [""],
    });
    let typed_and_logprobs = json!({
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "Say ok."}]
        }],
        "logprobs": true,
        "top_logprobs": 2
    });
    let logprobs_stop_budget = json!({
        "logprobs": true,
        "top_logprobs": 2,
        "stop": [""],
        "reasoning_budget": 1
    });
    let mut stream_logprobs_wrong_model = server.chat_body(json!({
        "stream": true,
        "logprobs": true,
        "top_logprobs": 2
    }));
    stream_logprobs_wrong_model["model"] = json!("wrong-model");
    let cases = [
        ("unsupported_feature_vs_wrong_model", wrong_model_and_format),
        ("lora_vs_image_content", server.chat_body(with_image)),
        (
            "context_vs_malformed_stop",
            server.chat_body(context_and_stop),
        ),
        (
            "gemma_typed_parts_vs_logprobs",
            server.chat_body(typed_and_logprobs),
        ),
        (
            "logprobs_vs_stop_vs_budget",
            server.chat_body(logprobs_stop_budget),
        ),
        (
            "stream_logprobs_vs_wrong_model",
            stream_logprobs_wrong_model,
        ),
    ];
    let mut observations = Map::new();
    for (name, original_body) in cases {
        let mut modes = Map::new();
        for streaming in [false, true] {
            let mode = if streaming { "stream" } else { "nonstream" };
            let mut body = original_body.clone();
            body["stream"] = json!(streaming);
            let response = request_json(server, &body);
            let observed = if response.status == 200 {
                success_contract(&response)
            } else {
                wire_contract(&response)
            };
            let repeated_response = request_json(server, &body);
            let repeated = if repeated_response.status == 200 {
                success_contract(&repeated_response)
            } else {
                wire_contract(&repeated_response)
            };
            assert_eq!(observed, repeated, "{family} {name} {mode} repeat");
            assert_eq!(
                observed,
                expected_competing_contract(family, name, streaming),
                "{family} {name} {mode} competing request checks"
            );
            modes.insert(mode.to_owned(), observed);
        }
        observations.insert(name.to_owned(), Value::Object(modes));
    }
    eprintln!(
        "CPU competing requests family={family} {}",
        Value::Object(observations)
    );
}

fn expected_competing_contract(family: &str, name: &str, streaming: bool) -> Value {
    let logprobs_stream_error = expected_error_contract(
        400,
        "unsupported_feature",
        "logprobs is not supported together with stream: true",
    );
    match name {
        "unsupported_feature_vs_wrong_model" => expected_error_contract(
            400,
            "unsupported_feature",
            "response_format.type 'json_object' is not supported; use 'text'",
        ),
        "lora_vs_image_content" => expected_error_contract(
            400,
            "lora_unsupported_backend",
            "this server was built without Metal support; runtime LoRA adapters require a macOS Metal build",
        ),
        "context_vs_malformed_stop" if family == "qwen35" => expected_error_contract(
            400,
            "context_length_exceeded",
            "prompt (100009 tokens) plus max_tokens (2) plus reasoning_budget (0) exceeds model context window (8192): 100012 tokens required",
        ),
        "context_vs_malformed_stop" => {
            expected_error_contract(400, "invalid_stop", "stop string must not be empty")
        }
        "gemma_typed_parts_vs_logprobs" if streaming => logprobs_stream_error,
        "gemma_typed_parts_vs_logprobs" if family == "qwen35" => expected_success_contract(false),
        "gemma_typed_parts_vs_logprobs" => expected_error_contract(
            400,
            "unsupported_feature",
            "typed content parts are not supported for this model; send message content as a string",
        ),
        "logprobs_vs_stop_vs_budget" if streaming => logprobs_stream_error,
        "logprobs_vs_stop_vs_budget" => {
            expected_error_contract(400, "invalid_stop", "stop string must not be empty")
        }
        "stream_logprobs_vs_wrong_model" if streaming => logprobs_stream_error,
        "stream_logprobs_vs_wrong_model" => expected_error_contract(
            400,
            "model_not_found",
            "model 'wrong-model' is not loaded; this server serves 'characterization-model'",
        ),
        _ => panic!("unknown competing request {name}"),
    }
}

fn observe_raw_http_errors(server: &Server, family: &str) {
    let mut observations = Map::new();
    for streaming in [false, true] {
        let mode = if streaming { "stream" } else { "nonstream" };
        let malformed = if streaming {
            br#"{"stream":true,"messages":["# as &[u8]
        } else {
            br#"{"stream":false,"messages":["# as &[u8]
        };
        let malformed_response = request_raw(server, "application/json", malformed);
        let media_body = serde_json::to_vec(&json!({"stream": streaming})).expect("JSON bytes");
        let media_response = request_raw(server, "text/plain", &media_body);
        let mut oversized = format!("{{\"stream\":{streaming},\"padding\":\"").into_bytes();
        oversized.resize(1_048_576, b'x');
        oversized.extend_from_slice(b"\"}");
        let oversized_response = request_raw(server, "application/json", &oversized);
        let malformed_contract = wire_contract(&malformed_response);
        let media_contract = wire_contract(&media_response);
        let oversized_contract = wire_contract(&oversized_response);
        assert_eq!(
            malformed_contract,
            expected_error_contract(400, "invalid_request_body", "invalid JSON request body",),
            "{family} {mode} malformed JSON envelope"
        );
        assert_eq!(
            media_contract,
            expected_error_contract(
                415,
                "unsupported_media_type",
                "Content-Type must be application/json",
            ),
            "{family} {mode} unsupported media envelope"
        );
        assert_eq!(
            oversized_contract,
            expected_error_contract(
                413,
                "request_body_too_large",
                "request body exceeds 1 MiB limit",
            ),
            "{family} {mode} oversize body envelope"
        );
        observations.insert(
            mode.to_owned(),
            json!({
                "malformed_json": malformed_contract,
                "unsupported_media_type": media_contract,
                "oversize_body": oversized_contract,
            }),
        );
    }
    eprintln!(
        "CPU raw HTTP errors family={family} {}",
        Value::Object(observations)
    );
}

fn observe_logprobs(server: &Server, family: &str) {
    let body = server.chat_body(json!({"logprobs": true, "top_logprobs": 2}));
    let response = request_json(server, &body);
    let observed = wire_contract(&response);
    let repeated = wire_contract(&request_json(server, &body));
    assert_eq!(observed, repeated, "{family} logprobs repeat");
    let expected = if family == "qwen35" {
        expected_qwen_logprobs()
    } else {
        expected_error_contract(
            400,
            "unsupported_feature",
            "logprobs are not supported for this model",
        )
    };
    assert_eq!(observed, expected, "{family} logprobs response contract");
    eprintln!("CPU logprobs family={family} {observed}");
}

fn expected_qwen_logprobs() -> Value {
    json!({
        "status": 200,
        "content_type": "application/json",
        "body": {
            "id": "<normalized-id>",
            "created": 0,
            "model": "characterization-model",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "<think>\n\n"},
                "finish_reason": "length",
                "logprobs": {"content": [
                    {
                        "token": "<think>",
                        "bytes": [60, 116, 104, 105, 110, 107, 62],
                        "logprob": -0.0003875935,
                        "top_logprobs": [
                            {
                                "token": "<think>",
                                "bytes": [60, 116, 104, 105, 110, 107, 62],
                                "logprob": -0.0003875935
                            },
                            {
                                "token": "Sure",
                                "bytes": [83, 117, 114, 101],
                                "logprob": -9.102976
                            }
                        ]
                    },
                    {
                        "token": "\n\n",
                        "bytes": [10, 10],
                        "logprob": -0.9119996,
                        "top_logprobs": [
                            {
                                "token": "\n",
                                "bytes": [10],
                                "logprob": -0.5146111
                            },
                            {
                                "token": "\n\n",
                                "bytes": [10, 10],
                                "logprob": -0.9119996
                            }
                        ]
                    }
                ]}
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
        }
    })
}

fn report_capture_identity(model_dir: &Path, family: &str) {
    let model_name = model_dir
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .expect("checkpoint directory has a UTF-8 model name");
    let config_hash = hash_file(&model_dir.join("config.json"));
    let (manifest_hash, file_count) = checkpoint_manifest(model_dir);
    let binary_hash = hash_file(Path::new(env!("CARGO_BIN_EXE_lattice")));
    eprintln!(
        "CPU checkpoint family={family} source=7c42306c34b1f2f642b14de33cd5326dd94f8c0d model={model_name} config_sha256={config_hash} file_manifest_sha256={manifest_hash} file_count={file_count} binary_sha256={binary_hash} features=std,serve"
    );
}

fn checkpoint_manifest(root: &Path) -> (String, usize) {
    fn collect(root: &Path, dir: &Path, files: &mut Vec<(PathBuf, u64)>) {
        for entry in fs::read_dir(dir).expect("read checkpoint directory") {
            let entry = entry.expect("read checkpoint entry");
            let path = entry.path();
            let kind = entry.file_type().expect("inspect checkpoint entry");
            if kind.is_dir() {
                collect(root, &path, files);
            } else if kind.is_file() {
                let relative = path
                    .strip_prefix(root)
                    .expect("relative checkpoint path")
                    .into();
                let size = entry.metadata().expect("checkpoint file metadata").len();
                files.push((relative, size));
            }
        }
    }

    let mut files = Vec::new();
    collect(root, root, &mut files);
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    let file_count = files.len();
    for (relative, size) in files {
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update(size.to_le_bytes());
    }
    let output = digest.finalize();
    (digest_hex(&output), file_count)
}

fn hash_file(path: &Path) -> String {
    let mut file = fs::File::open(path).expect("open lattice binary");
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).expect("read lattice binary");
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let output = digest.finalize();
    digest_hex(&output)
}

fn digest_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct WireResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

fn request_json(server: &Server, body: &Value) -> WireResponse {
    let payload = serde_json::to_vec(body).expect("request JSON serializes");
    request_raw(server, "application/json", &payload)
}

fn request_raw(server: &Server, content_type: &str, payload: &[u8]) -> WireResponse {
    let mut stream = TcpStream::connect(("127.0.0.1", server.port())).expect("connect HTTP route");
    stream
        .set_read_timeout(Some(Duration::from_secs(480)))
        .expect("set HTTP read timeout");
    stream
        .set_write_timeout(Some(Duration::from_secs(480)))
        .expect("set HTTP write timeout");
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|()| stream.write_all(payload))
        .expect("send HTTP request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read HTTP response");
    parse_wire_response(&raw)
}

fn parse_wire_response(raw: &[u8]) -> WireResponse {
    let split = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("HTTP response has header terminator");
    let head = String::from_utf8_lossy(&raw[..split]);
    let status = head
        .split_whitespace()
        .nth(1)
        .expect("HTTP status exists")
        .parse::<u16>()
        .expect("HTTP status is numeric");
    let headers = head
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect::<Vec<_>>();
    let body = &raw[split + 4..];
    let body = if headers
        .iter()
        .any(|(name, value)| name == "transfer-encoding" && value.eq_ignore_ascii_case("chunked"))
    {
        decode_chunked_body(body)
    } else {
        body.to_vec()
    };
    WireResponse {
        status,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    }
}

fn decode_chunked_body(mut raw: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    loop {
        let Some(end) = raw.windows(2).position(|pair| pair == b"\r\n") else {
            return body;
        };
        let Some(size) = std::str::from_utf8(&raw[..end])
            .ok()
            .and_then(|line| usize::from_str_radix(line.split(';').next()?.trim(), 16).ok())
        else {
            return body;
        };
        raw = &raw[end + 2..];
        if size == 0 || raw.len() < size + 2 {
            return body;
        }
        body.extend_from_slice(&raw[..size]);
        raw = &raw[size + 2..];
    }
}

fn content_type(response: &WireResponse) -> &str {
    response
        .headers
        .iter()
        .find(|(name, _)| name == "content-type")
        .map(|(_, value)| value.split(';').next().unwrap_or(value))
        .unwrap_or("<missing>")
}

fn wire_contract(response: &WireResponse) -> Value {
    let mut body = serde_json::from_str::<Value>(&response.body).expect("response body is JSON");
    normalize_volatile_fields(&mut body);
    json!({
        "status": response.status,
        "content_type": content_type(response),
        "body": body,
    })
}

fn transcript(response: &WireResponse, streaming: bool) -> Value {
    if streaming {
        let events = response
            .body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| {
                if data == "[DONE]" {
                    json!("[DONE]")
                } else {
                    let mut event = serde_json::from_str::<Value>(data).expect("SSE event JSON");
                    normalize_volatile_fields(&mut event);
                    event
                }
            })
            .collect::<Vec<_>>();
        let joined = events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .collect::<String>();
        json!({
            "status": response.status,
            "content_type": content_type(response),
            "events": events,
            "content_deltas_joined": joined,
        })
    } else {
        wire_contract(response)
    }
}

fn normalize_volatile_fields(value: &mut Value) {
    // Only opaque response IDs and server timestamps vary between identical requests.
    match value {
        Value::Object(fields) => {
            if fields.contains_key("id") {
                fields.insert("id".to_owned(), json!("<normalized-id>"));
            }
            if fields.contains_key("created") {
                fields.insert("created".to_owned(), json!(0));
            }
            for field in fields.values_mut() {
                normalize_volatile_fields(field);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_volatile_fields(item);
            }
        }
        _ => {}
    }
}

fn assert_concurrent_greedy_results(
    server: &Server,
    cell: &str,
    max_tokens: usize,
    prompts: &[&str],
) {
    let barrier = Arc::new(Barrier::new(prompts.len() + 1));
    let port = server.port();
    let requests = prompts
        .iter()
        .map(|prompt| {
            let body = server.chat_body(json!({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": false
            }));
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                common::request_at_port(port, &body)
            })
        })
        .collect::<Vec<_>>();

    barrier.wait();
    let concurrent = requests
        .into_iter()
        .map(|request| request.join().expect("concurrent request thread joins"))
        .collect::<Vec<_>>();
    let concurrent_text = concurrent
        .iter()
        .enumerate()
        .map(|(index, response)| {
            assert_eq!(
                response.status, 200,
                "concurrent request {index}: {}",
                response.body
            );
            let text = response_text(response);
            assert!(
                !text.starts_with("<missing completion text:"),
                "concurrent request {index}: {}",
                response.body
            );
            text
        })
        .collect::<Vec<_>>();
    let mut solo_texts = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        let body = server.chat_body(json!({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": false
        }));
        let response = server.request("POST", "/v1/chat/completions", Some(&body));
        assert_eq!(
            response.status, 200,
            "solo request {index}: {}",
            response.body
        );
        solo_texts.push(response_text(&response));
    }

    for left in 0..solo_texts.len() {
        for right in (left + 1)..solo_texts.len() {
            assert_ne!(
                solo_texts[left], solo_texts[right],
                "{cell} solo outputs must be pairwise distinct for prompts {:?} and {:?}",
                prompts[left], prompts[right]
            );
        }
    }
    eprintln!("{cell} pairwise-distinct solo outputs: {solo_texts:?}");

    for (index, prompt) in prompts.iter().enumerate() {
        assert_eq!(
            concurrent_text[index], solo_texts[index],
            "{cell} concurrent output differed from the same prompt sent alone: {prompt}"
        );
    }
}
