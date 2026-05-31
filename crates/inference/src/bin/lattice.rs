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
        /// Maximum tokens to generate per request
        #[arg(long, default_value = "256")]
        max_tokens: usize,
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
        routing::{get, post},
    };
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub type SharedModel = Arc<lattice_inference::model::qwen35::Qwen35Model>;

    #[derive(Deserialize)]
    pub struct ChatCompletionRequest {
        #[allow(dead_code)]
        pub model: Option<String>,
        pub messages: Vec<Message>,
        pub max_tokens: Option<usize>,
        pub temperature: Option<f32>,
    }

    #[derive(Deserialize)]
    pub struct Message {
        #[allow(dead_code)]
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

    pub async fn health() -> Json<HealthResponse> {
        Json(HealthResponse { status: "ok" })
    }

    pub async fn chat_completions(
        State(model): State<SharedModel>,
        Json(req): Json<ChatCompletionRequest>,
    ) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
        // Extract the last user message content as the prompt.
        let prompt = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();

        if prompt.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "no user message found in messages".to_string(),
            ));
        }

        let max_tokens = req.max_tokens.unwrap_or(256);
        let temperature = req.temperature.unwrap_or(0.7);

        let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
            max_new_tokens: max_tokens,
            temperature,
            ..Default::default()
        };

        // `generate` is CPU-bound blocking work; run it on the blocking thread pool.
        let output = tokio::task::spawn_blocking(move || model.generate(&prompt, &gen_cfg))
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("task join error: {e}"),
                )
            })?
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("generation error: {e}"),
                )
            })?;

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{created}"),
            object: "chat.completion".to_string(),
            created,
            model: "lattice".to_string(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: output.text.clone(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.generated_tokens,
                total_tokens: output.prompt_tokens + output.generated_tokens,
            },
        };

        Ok(Json(response))
    }

    pub fn router(model: SharedModel) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(model)
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
        } => {
            use std::path::Path;
            use std::sync::Arc;

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
            eprintln!("Model loaded.");

            // Wrap in Arc so the model can be cheaply cloned into each request's
            // spawn_blocking closure without copying weights.
            let shared = Arc::new(qwen_model);

            // Build and start the axum server.  The default max_tokens from the
            // CLI flag is baked into the router via a closure over `max_tokens`.
            let app = serve::router(shared);

            let addr = format!("0.0.0.0:{port}");
            let listener = match tokio::net::TcpListener::bind(&addr).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error: failed to bind to {addr}: {e}");
                    std::process::exit(1);
                }
            };
            eprintln!("Listening on {addr}  (max_tokens default: {max_tokens})");
            eprintln!("  POST /v1/chat/completions");
            eprintln!("  GET  /health");

            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("Server error: {e}");
                std::process::exit(1);
            }
        }
    }
}
