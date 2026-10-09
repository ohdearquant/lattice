#![cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "f16",
    feature = "serve"
))]

mod common;

use base64::Engine as _;
use common::{
    Binary, HttpResponse, Server, acquire_gpu_lock, assert_http_golden, error_code,
    require_checkpoint, stage_q4_with_tokenizer,
};
use serde_json::{Value, json};
use std::sync::{Mutex, PoisonError};

static MATRIX_CELL_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy)]
struct Expected {
    status: u16,
    code: Option<&'static str>,
}

const OK: Expected = Expected {
    status: 200,
    code: None,
};
const UNSUPPORTED: Expected = Expected {
    status: 400,
    code: Some("unsupported_feature"),
};
const VISION_UNSUPPORTED: Expected = Expected {
    status: 400,
    code: Some("vision_unsupported"),
};
const STRICT_SCHEMA_UNSUPPORTED: Expected = Expected {
    status: 400,
    code: Some("unsupported_strict_schema"),
};
const INTERNAL_ERROR: Expected = Expected {
    status: 500,
    code: Some("internal_error"),
};

const LATTICE_QWEN_CPU: [Expected; 15] = [
    OK,
    OK,
    OK,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("lora_unsupported_backend"),
    },
    OK,
    VISION_UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("max_tokens_exceeds_limit"),
    },
    OK,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
];

const LATTICE_GEMMA_CPU: [Expected; 15] = [
    OK,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("lora_unsupported_backend"),
    },
    UNSUPPORTED,
    VISION_UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("max_tokens_exceeds_limit"),
    },
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
];

const LATTICE_QWEN_Q4_METAL: [Expected; 15] = [
    OK,
    OK,
    INTERNAL_ERROR,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("lora_adapter_not_found"),
    },
    OK,
    OK,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("max_tokens_exceeds_limit"),
    },
    OK,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
];

const STANDALONE_QWEN_METAL: [Expected; 15] = [
    OK,
    OK,
    UNSUPPORTED,
    STRICT_SCHEMA_UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("lora_adapter_not_found"),
    },
    OK,
    OK,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("context_length_exceeded"),
    },
    OK,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
];

const STANDALONE_GEMMA_CPU: [Expected; 15] = [
    OK,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("lora_unsupported_backend"),
    },
    UNSUPPORTED,
    VISION_UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    Expected {
        status: 400,
        code: Some("context_length_exceeded"),
    },
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
    UNSUPPORTED,
];

#[test]
fn lattice_qwen_safetensors_cpu_request_refusal_matrix() {
    let _cell = MATRIX_CELL_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let Some(model) = require_checkpoint(common::QWEN_DIR_ENV, "qwen3.5-0.8b", "lattice-qwen-cpu")
    else {
        return;
    };
    let server = Server::spawn(Binary::Lattice, &model, &[]);
    assert_matrix(&server, "lattice-qwen-cpu", &LATTICE_QWEN_CPU);
    assert_http_golden(
        &server.embeddings(),
        200,
        None,
        "lattice-qwen-cpu embeddings",
    );
    assert_eq!(
        server.model_list_status(),
        200,
        "lattice-qwen-cpu GET /v1/models"
    );
}

#[test]
fn lattice_gemma_e2b_cpu_request_refusal_matrix() {
    let _cell = MATRIX_CELL_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let Some(model) =
        require_checkpoint(common::GEMMA_DIR_ENV, "gemma-4-e2b-it", "lattice-gemma-cpu")
    else {
        return;
    };
    let server = Server::spawn(Binary::Lattice, &model, &[]);
    assert_matrix(&server, "lattice-gemma-cpu", &LATTICE_GEMMA_CPU);
    assert_http_golden(
        &server.embeddings(),
        400,
        Some("vision_unsupported"),
        "lattice-gemma-cpu POST /v1/embeddings",
    );
    assert_eq!(
        server.model_list_status(),
        200,
        "lattice-gemma-cpu GET /v1/models"
    );
}

#[test]
fn lattice_qwen_q4_metal_request_refusal_matrix() {
    let _cell = MATRIX_CELL_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let Some(qwen) = require_checkpoint(
        common::QWEN_DIR_ENV,
        "qwen3.5-0.8b",
        "lattice-qwen-q4-tokenizer",
    ) else {
        return;
    };
    let Some(q4) = require_checkpoint(
        common::QWEN_Q4_DIR_ENV,
        "qwen3.5-0.8b-q4",
        "lattice-qwen-q4",
    ) else {
        return;
    };
    let _gpu = acquire_gpu_lock();
    let staged = stage_q4_with_tokenizer(&q4, &qwen);
    let server = Server::spawn(Binary::Lattice, staged.path(), &[]);
    assert_matrix(&server, "lattice-qwen-q4-metal", &LATTICE_QWEN_Q4_METAL);
    assert_http_golden(
        &server.embeddings(),
        400,
        Some("vision_unsupported"),
        "lattice-qwen-q4 POST /v1/embeddings",
    );
    assert_eq!(
        server.model_list_status(),
        200,
        "lattice-qwen-q4 GET /v1/models"
    );
}

#[test]
fn standalone_qwen_safetensors_metal_request_refusal_matrix() {
    let _cell = MATRIX_CELL_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let Some(model) = require_checkpoint(
        common::QWEN_DIR_ENV,
        "qwen3.5-0.8b",
        "standalone-qwen-metal",
    ) else {
        return;
    };
    let _gpu = acquire_gpu_lock();
    let server = Server::spawn(Binary::LatticeServe, &model, &[]);
    assert_matrix(&server, "standalone-qwen-metal", &STANDALONE_QWEN_METAL);
    assert_http_golden(
        &server.embeddings(),
        503,
        Some("embedding_model_not_loaded"),
        "standalone-qwen POST /v1/embeddings without --embedding-model",
    );
    assert_eq!(
        server.model_list_status(),
        200,
        "standalone-qwen GET /v1/models"
    );
}

#[test]
fn standalone_gemma_e2b_cpu_request_refusal_matrix() {
    let _cell = MATRIX_CELL_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let Some(model) = require_checkpoint(
        common::GEMMA_DIR_ENV,
        "gemma-4-e2b-it",
        "standalone-gemma-cpu",
    ) else {
        return;
    };
    let server = Server::spawn(Binary::LatticeServe, &model, &[]);
    assert_matrix(&server, "standalone-gemma-cpu", &STANDALONE_GEMMA_CPU);
    assert_http_golden(
        &server.embeddings(),
        503,
        Some("embedding_model_not_loaded"),
        "standalone-gemma POST /v1/embeddings without --embedding-model",
    );
    assert_eq!(
        server.model_list_status(),
        200,
        "standalone-gemma GET /v1/models"
    );
}

fn assert_matrix(server: &Server, cell: &str, expected: &[Expected; 15]) {
    let probes = probes(server);
    assert_eq!(
        probes.len(),
        expected.len(),
        "{cell} probe and expected table lengths differ"
    );
    for ((name, body), expected) in probes.into_iter().zip(expected) {
        let response = server.request("POST", "/v1/chat/completions", Some(&body));
        assert_http_golden(
            &response,
            expected.status,
            expected.code,
            &format!("{cell} {name}"),
        );
        if matches!(
            name,
            "tools_before_zero_max_tokens"
                | "n_before_zero_max_tokens"
                | "json_object_before_zero_max_tokens"
        ) {
            assert_combination_message(name, &response);
        }
    }
}

fn probes(server: &Server) -> Vec<(&'static str, Value)> {
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
                        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{png}")}}
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

fn assert_combination_message(name: &str, response: &HttpResponse) {
    let value: Value = serde_json::from_str(&response.body).expect("error response is JSON");
    let message = value["error"]["message"].as_str().unwrap_or_default();
    let expected = match name {
        "tools_before_zero_max_tokens" => "tools and tool_choice are not supported by this server",
        "n_before_zero_max_tokens" => "n > 1 is not supported",
        "json_object_before_zero_max_tokens" => {
            "response_format.type 'json_object' is not supported; use 'text'"
        }
        _ => unreachable!("the combination probe list is fixed"),
    };
    assert_eq!(message, expected, "{name}: {}", response.body);
    assert_eq!(error_code(response).as_deref(), Some("unsupported_feature"));
}

#[test]
fn q4_staging_replaces_an_existing_tokenizer_name_without_changing_sources() {
    let q4_dir = tempfile::tempdir().expect("create Q4 checkpoint fixture");
    let tokenizer_dir = tempfile::tempdir().expect("create tokenizer checkpoint fixture");
    let q4_tokenizer = b"pre-existing Q4 tokenizer";
    let replacement_tokenizer = b"replacement tokenizer from safetensors checkpoint";
    std::fs::write(q4_dir.path().join("tokenizer.json"), q4_tokenizer)
        .expect("write pre-existing Q4 tokenizer");
    std::fs::write(
        tokenizer_dir.path().join("tokenizer.json"),
        replacement_tokenizer,
    )
    .expect("write replacement tokenizer");

    let staged = stage_q4_with_tokenizer(q4_dir.path(), tokenizer_dir.path());

    assert_eq!(
        std::fs::read(staged.path().join("tokenizer.json")).expect("read staged tokenizer"),
        replacement_tokenizer
    );
    assert_eq!(
        std::fs::read(q4_dir.path().join("tokenizer.json")).expect("read Q4 source tokenizer"),
        q4_tokenizer
    );
    assert_eq!(
        std::fs::read(tokenizer_dir.path().join("tokenizer.json")).expect("read tokenizer source"),
        replacement_tokenizer
    );
}
