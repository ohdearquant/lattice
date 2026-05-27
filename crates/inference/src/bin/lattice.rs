//! `lattice` CLI — interactive chat and HTTP serve subcommands.
//!
//! # Usage
//!
//! ```text
//! lattice chat --model /path/to/model [--max-tokens 256] [--temperature 0.7]
//! lattice serve --model /path/to/model [--port 8080] [--max-tokens 256]
//! ```

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "lattice", about = "Pure-Rust transformer inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive chat with a model
    Chat {
        /// Path to model directory (SafeTensors + config.json)
        #[arg(long)]
        model: String,
        /// Maximum tokens to generate per response
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },
    /// Start HTTP server with OpenAI-compatible API
    Serve {
        /// Path to model directory
        #[arg(long)]
        model: String,
        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Maximum tokens to generate per request (default when request omits max_tokens)
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Model identifier echoed in responses (defaults to the model path basename)
        #[arg(long)]
        model_id: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// chat subcommand
// ---------------------------------------------------------------------------

fn run_chat(model_path: &str, max_tokens: usize, temperature: f32) {
    use std::io::{BufRead, Write};
    use std::path::Path;

    let path = Path::new(model_path);
    eprintln!("Loading model from {model_path}...");
    let model = match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: failed to load model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("Model loaded. Type 'exit' or 'quit' to stop.\n");

    let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        temperature,
        ..Default::default()
    };

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let prompt = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        };
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            break;
        }

        match model.generate(trimmed, &gen_cfg) {
            Ok(output) => {
                let _ = writeln!(stdout, "{}", output.text);
                let _ = writeln!(
                    stdout,
                    "[{} prompt tokens, {} generated]",
                    output.prompt_tokens, output.generated_tokens
                );
            }
            Err(e) => {
                eprintln!("Generation error: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// serve subcommand: OpenAI-compatible HTTP API
// ---------------------------------------------------------------------------

mod serve {
    use axum::{
        Json, Router,
        extract::State,
        http::StatusCode,
        response::{IntoResponse, Response},
        routing::{get, post},
    };
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    // -----------------------------------------------------------------------
    // Shared application state
    // -----------------------------------------------------------------------

    /// State shared across all request handlers via axum's `State` extractor.
    #[derive(Clone)]
    pub struct AppState {
        /// The loaded model, wrapped in Arc so it can be cheaply cloned into
        /// `spawn_blocking` closures without copying weights.
        pub model: Arc<lattice_inference::model::qwen35::Qwen35Model>,
        /// Default `max_tokens` value used when a request omits the field.
        /// Set from the `--max-tokens` CLI flag passed to `lattice serve`.
        pub default_max_tokens: usize,
        /// Canonical model identifier echoed in every response.
        /// Derived from the `--model-id` flag or the model path basename.
        pub model_id: String,
    }

    // -----------------------------------------------------------------------
    // Error type
    // -----------------------------------------------------------------------

    /// Structured HTTP error that serialises to the OpenAI error envelope so
    /// that clients can parse failure responses uniformly.
    #[derive(Debug)]
    pub enum ApiError {
        /// Caller mistake — HTTP 400.
        BadRequest { message: String, code: &'static str },
        /// Server-side failure — HTTP 500.
        Internal { message: String },
    }

    #[derive(Serialize)]
    struct ErrorBody {
        error: ErrorDetail,
    }

    #[derive(Serialize)]
    struct ErrorDetail {
        message: String,
        r#type: &'static str,
        code: String,
    }

    impl IntoResponse for ApiError {
        fn into_response(self) -> Response {
            match self {
                ApiError::BadRequest { message, code } => {
                    let body = Json(ErrorBody {
                        error: ErrorDetail {
                            message,
                            r#type: "invalid_request_error",
                            code: code.to_string(),
                        },
                    });
                    (StatusCode::BAD_REQUEST, body).into_response()
                }
                ApiError::Internal { message } => {
                    let body = Json(ErrorBody {
                        error: ErrorDetail {
                            message,
                            r#type: "server_error",
                            code: "internal_error".to_string(),
                        },
                    });
                    (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Request / response types
    // -----------------------------------------------------------------------

    #[derive(Deserialize)]
    pub struct ChatCompletionRequest {
        /// Required: must match the served model identifier.
        pub model: String,
        pub messages: Vec<Message>,
        pub max_tokens: Option<usize>,
        pub temperature: Option<f32>,
    }

    #[derive(Deserialize)]
    pub struct Message {
        pub role: String,
        pub content: String,
    }

    #[derive(Serialize)]
    pub struct ChatCompletionResponse {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<Choice>,
        pub usage: Usage,
    }

    #[derive(Serialize)]
    pub struct Choice {
        pub index: usize,
        pub message: ResponseMessage,
        pub finish_reason: String,
    }

    #[derive(Serialize)]
    pub struct ResponseMessage {
        pub role: String,
        pub content: String,
    }

    #[derive(Serialize)]
    pub struct Usage {
        pub prompt_tokens: usize,
        pub completion_tokens: usize,
        pub total_tokens: usize,
    }

    #[derive(Serialize)]
    pub struct HealthResponse {
        pub status: &'static str,
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a single prompt string from the full message list using a simple
    /// role-prefixed format compatible with instruction-tuned models.
    ///
    /// Format (one block per message, in order):
    /// ```text
    /// <|system|>
    /// {content}
    /// <|user|>
    /// {content}
    /// <|assistant|>
    /// {content}
    /// ```
    /// The caller is expected to validate that the last message is a user turn.
    fn render_prompt(messages: &[Message]) -> String {
        let mut buf = String::new();
        for msg in messages {
            buf.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
        }
        buf
    }

    // -----------------------------------------------------------------------
    // Handlers
    // -----------------------------------------------------------------------

    pub async fn health() -> Json<HealthResponse> {
        Json(HealthResponse { status: "ok" })
    }

    pub async fn chat_completions(
        State(state): State<AppState>,
        Json(req): Json<ChatCompletionRequest>,
    ) -> Result<Json<ChatCompletionResponse>, ApiError> {
        // Validate that the caller targets the served model.
        if req.model != state.model_id {
            return Err(ApiError::BadRequest {
                message: format!(
                    "model '{}' is not loaded; this server serves '{}'",
                    req.model, state.model_id
                ),
                code: "model_not_found",
            });
        }

        if req.messages.is_empty() {
            return Err(ApiError::BadRequest {
                message: "messages must not be empty".to_string(),
                code: "invalid_messages",
            });
        }

        // Require the conversation to end with a user turn.
        let last_role = req.messages.last().map(|m| m.role.as_str()).unwrap_or("");
        if last_role != "user" {
            return Err(ApiError::BadRequest {
                message: "the last message must have role 'user'".to_string(),
                code: "invalid_messages",
            });
        }

        // Concatenate all messages into a single prompt preserving full context.
        let prompt = render_prompt(&req.messages);

        let max_tokens = req.max_tokens.unwrap_or(state.default_max_tokens);
        let temperature = req.temperature.unwrap_or(0.7);

        let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
            max_new_tokens: max_tokens,
            temperature,
            ..Default::default()
        };

        let model = Arc::clone(&state.model);

        // `generate` is CPU-bound blocking work; run it on the blocking thread pool.
        let output = tokio::task::spawn_blocking(move || model.generate(&prompt, &gen_cfg))
            .await
            .map_err(|e| ApiError::Internal {
                message: format!("task join error: {e}"),
            })?
            .map_err(|e| ApiError::Internal {
                message: format!("generation error: {e}"),
            })?;

        // Distinguish "hit token cap" from "natural stop" (EOS / stop token).
        // `GenerateOutput` does not carry an explicit stop reason, so we infer
        // it: if the model generated exactly `max_new_tokens` tokens the cap
        // was reached.
        let finish_reason = if output.generated_tokens >= max_tokens {
            "length"
        } else {
            "stop"
        };

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{created}"),
            object: "chat.completion".to_string(),
            created,
            model: state.model_id.clone(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: output.text.clone(),
                },
                finish_reason: finish_reason.to_string(),
            }],
            usage: Usage {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.generated_tokens,
                total_tokens: output.prompt_tokens + output.generated_tokens,
            },
        };

        Ok(Json(response))
    }

    // -----------------------------------------------------------------------
    // Router
    // -----------------------------------------------------------------------

    pub fn router(state: AppState) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state)
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Chat {
            model,
            max_tokens,
            temperature,
        } => {
            run_chat(&model, max_tokens, temperature);
        }
        Command::Serve {
            model,
            port,
            max_tokens,
            model_id,
        } => {
            use std::path::Path;
            use std::sync::Arc;

            // Derive a model identifier from the path basename when --model-id
            // is not provided.
            let served_model_id = model_id.unwrap_or_else(|| {
                Path::new(&model)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("lattice")
                    .to_string()
            });

            eprintln!("Loading model from {model}...");
            let qwen_model = match lattice_inference::model::qwen35::Qwen35Model::from_safetensors(
                Path::new(&model),
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Error: failed to load model: {e}");
                    std::process::exit(1);
                }
            };
            eprintln!("Model loaded. Serving as '{served_model_id}'.");

            let state = serve::AppState {
                model: Arc::new(qwen_model),
                default_max_tokens: max_tokens,
                model_id: served_model_id.clone(),
            };

            let app = serve::router(state);

            let addr = format!("0.0.0.0:{port}");
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error: failed to bind to {addr}: {e}");
                    std::process::exit(1);
                }
            };
            eprintln!(
                "Listening on {addr}  (model: {served_model_id}, max_tokens default: {max_tokens})"
            );
            eprintln!("  POST /v1/chat/completions");
            eprintln!("  GET  /health");

            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
    }
}
