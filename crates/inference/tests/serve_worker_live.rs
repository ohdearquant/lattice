//! Live characterization of the Metal worker route behind both shipped servers on
//! real Qwen3.5 checkpoints: the standalone `lattice_serve` in both checkpoint
//! formats (safetensors, and Q4 staged with the safetensors tokenizer), and
//! `lattice serve` on Q4, the format it serves on the worker.
//!
//! Each test starts a shipped binary and observes it only through HTTP and
//! the worker's own stderr lines:
//!
//! - `[metal-worker] cross-turn cache: mode=.. reused=.. prefetched=.. prompt=..`
//!   is printed once per text request that reached generation, with the prefix
//!   reuse plan that request ran under.
//! - `[route] served family=qwen35 backend=metal mode=.. driver=.. opened=..
//!   consumed=..` is printed once per request that reached generation, text or
//!   image.
//! - `[metal-worker] route=vision dispatch=multimodal ..` is printed once per
//!   image request that reached the vision encoder.
//!
//! What is pinned: how a client disconnect ends a request (mid-decode, during
//! prefill, and while queued behind another request), how adapter residency and
//! application interact with the reused prefix, how an appended, diverged or
//! repeated conversation is planned, and how image requests reach the vision
//! route and leave the text prefix behind them.
//!
//! A checkpoint that is absent is a loud `LATTICE_SERVE_CHARACTERIZATION_SKIPPED`
//! line, never a pass; `LATTICE_SERVE_CHARACTERIZATION_GATE_ENFORCE=1` turns the
//! same condition into a failure. The target is outside `cargo test`'s default
//! set and runs when named:
//!
//! ```bash
//! LATTICE_SERVE_CHARACTERIZATION_GATE_ENFORCE=1 cargo test --release \
//!   -p lattice-inference --features f16,metal-gpu,serve \
//!   --test serve_worker_live -- --nocapture --test-threads=1
//! ```

#![cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "f16",
    feature = "serve"
))]

mod common;

use base64::Engine as _;
use common::{
    Binary, HttpResponse, Server, acquire_gpu_lock, require_checkpoint, response_text,
    stage_q4_with_tokenizer,
};
use lattice_inference::lora_hook::qwen35_projection_shape;
use lattice_inference::model::qwen35_config::Qwen35Config;
use serde_json::{Value, json};
use std::io::{BufRead as _, Write as _};
use std::net::{Shutdown, TcpStream};
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MARKER_DEADLINE: Duration = Duration::from_secs(120);
const STREAM_READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Short greedy turns keep each request cheap; the reuse plan does not depend
/// on the answer length.
const TURN_TOKENS: usize = 12;

/// A prompt that decodes for a long time under greedy decoding, so a request
/// that stopped early is distinguishable from one that ran to its budget.
const COUNT_PROMPT: &str = "Count from 1 to 300, separated by commas. Output only the numbers.";
const COUNT_TOKENS: usize = 320;

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

struct Checkpoint {
    format: &'static str,
    dir: PathBuf,
    _staged: Option<tempfile::TempDir>,
}

/// The safetensors checkpoint, then the Q4 checkpoint staged with the
/// safetensors tokenizer. Q4 needs the safetensors directory for its tokenizer,
/// so it is only attempted when that directory is present.
fn qwen_checkpoints(cell: &str) -> Vec<Checkpoint> {
    let mut checkpoints = Vec::new();
    let Some(safetensors) = require_checkpoint(
        common::QWEN_DIR_ENV,
        "qwen3.5-0.8b",
        &format!("{cell}-safetensors"),
    ) else {
        return checkpoints;
    };
    checkpoints.push(Checkpoint {
        format: "safetensors",
        dir: safetensors.clone(),
        _staged: None,
    });
    if let Some(q4) = require_checkpoint(
        common::QWEN_Q4_DIR_ENV,
        "qwen3.5-0.8b-q4",
        &format!("{cell}-q4"),
    ) {
        let staged = stage_q4_with_tokenizer(&q4, &safetensors);
        checkpoints.push(Checkpoint {
            format: "q4",
            dir: staged.path().to_path_buf(),
            _staged: Some(staged),
        });
    }
    checkpoints
}

/// Every Metal worker route: the standalone server on both formats, and the
/// `lattice serve` CLI on Q4 (it serves safetensors on the CPU instead).
fn targets(checkpoints: &[Checkpoint]) -> Vec<(Binary, &Checkpoint)> {
    let mut targets = Vec::new();
    for checkpoint in checkpoints {
        targets.push((Binary::LatticeServe, checkpoint));
        if checkpoint.format == "q4" {
            targets.push((Binary::Lattice, checkpoint));
        }
    }
    targets
}

fn label(binary: Binary, checkpoint: &Checkpoint) -> String {
    let server = match binary {
        Binary::Lattice => "cli",
        Binary::LatticeServe => "standalone",
    };
    format!("{server} {}", checkpoint.format)
}

fn spawn(binary: Binary, checkpoint: &Checkpoint, extra: &[String]) -> Server {
    let server = Server::spawn(binary, &checkpoint.dir, extra);
    let selected = format!(
        "[route] selected family=qwen35 backend=metal format={}",
        checkpoint.format
    );
    assert!(
        server.diagnostics().contains(&selected),
        "{}: {}",
        label(binary, checkpoint),
        server.diagnostics()
    );
    server
}

// ---------------------------------------------------------------------------
// Worker stderr lines
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheLine {
    mode: String,
    reused: usize,
    prefetched: usize,
    prompt: usize,
}

const CACHE_PREFIX: &str = "[metal-worker] cross-turn cache: mode=";

fn cache_lines(stderr: &str) -> Vec<CacheLine> {
    stderr
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix(CACHE_PREFIX)?;
            let (mode, counts) = rest.split_once(" reused=")?;
            let mut fields = counts.split_whitespace();
            let reused = fields.next()?.parse().ok()?;
            let prefetched = fields.next()?.strip_prefix("prefetched=")?.parse().ok()?;
            let prompt = fields.next()?.strip_prefix("prompt=")?.parse().ok()?;
            Some(CacheLine {
                mode: mode.to_owned(),
                reused,
                prefetched,
                prompt,
            })
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Served {
    stream: bool,
    driver: String,
    opened: usize,
    consumed: usize,
}

const SERVED_PREFIX: &str = "[route] served family=qwen35 backend=metal mode=";

fn served_lines(stderr: &str) -> Vec<Served> {
    stderr
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix(SERVED_PREFIX)?;
            let mut fields = rest.split_whitespace();
            let stream = match fields.next()? {
                "stream" => true,
                "nonstream" => false,
                _ => return None,
            };
            let driver = fields.next()?.strip_prefix("driver=")?.to_owned();
            let opened = fields.next()?.strip_prefix("opened=")?.parse().ok()?;
            let consumed = fields.next()?.strip_prefix("consumed=")?.parse().ok()?;
            Some(Served {
                stream,
                driver,
                opened,
                consumed,
            })
        })
        .collect()
}

const VISION_PREFIX: &str = "[metal-worker] route=vision dispatch=multimodal ";

/// `(metal_gemm_dispatches, metal_gemm_calls)` per vision request.
fn vision_lines(stderr: &str) -> Vec<(usize, usize)> {
    stderr
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix(VISION_PREFIX)?;
            let mut fields = rest.split_whitespace();
            let dispatches = fields
                .next()?
                .strip_prefix("metal_gemm_dispatches=")?
                .parse()
                .ok()?;
            let calls = fields
                .next()?
                .strip_prefix("metal_gemm_calls=")?
                .parse()
                .ok()?;
            Some((dispatches, calls))
        })
        .collect()
}

/// The server's stderr once `done` holds. The worker prints its lines before it
/// sends the terminal event, but the draining thread can trail the HTTP
/// response, so the wait is bounded rather than assumed.
fn wait_for_stderr(server: &Server, what: &str, done: impl Fn(&str) -> bool) -> String {
    let started = Instant::now();
    loop {
        let seen = server.diagnostics();
        if done(&seen) {
            return seen;
        }
        assert!(
            started.elapsed() < MARKER_DEADLINE,
            "timed out waiting for {what}; stderr:\n{seen}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn assistant(text: &str) -> Value {
    json!({"role": "assistant", "content": text})
}

/// The empty think block a thinking-off generation prompt ends with.
const EMPTY_THINK: &str = "<think>\n\n</think>\n\n";

/// An earlier answer as history, carrying the empty think block its generation
/// prompt ended with. The Qwen template renders history turns without that
/// block, so only this form reproduces the tokens the worker already holds.
fn echoed(answer: &str) -> Value {
    assistant(&format!("{EMPTY_THINK}{answer}"))
}

fn chat_body(server: &Server, messages: &[Value], max_tokens: usize, extra: Value) -> Value {
    let mut body = server.chat_body(json!({
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
    }));
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        body.extend(extra.clone());
    }
    body
}

fn chat(server: &Server, messages: &[Value], max_tokens: usize, extra: Value) -> HttpResponse {
    let body = chat_body(server, messages, max_tokens, extra);
    server.request("POST", "/v1/chat/completions", Some(&body))
}

fn completion_tokens(response: &HttpResponse) -> usize {
    serde_json::from_str::<Value>(&response.body)
        .ok()
        .and_then(|value| value["usage"]["completion_tokens"].as_u64())
        .and_then(|count| usize::try_from(count).ok())
        .unwrap_or_else(|| panic!("no usage.completion_tokens in {}", response.body))
}

/// One text request, its answer and the reuse plan it ran under.
fn text_turn(
    server: &Server,
    messages: &[Value],
    extra: Value,
    context: &str,
) -> (String, CacheLine) {
    let before = cache_lines(&server.diagnostics()).len();
    let response = chat(server, messages, TURN_TOKENS, extra);
    assert_eq!(
        response.status,
        200,
        "{context}: {}\n{}",
        response.body,
        server.diagnostics()
    );
    let text = response_text(&response);
    let seen = wait_for_stderr(server, context, |seen| cache_lines(seen).len() > before);
    let lines = cache_lines(&seen);
    assert_eq!(
        lines.len(),
        before + 1,
        "{context}: one cache line per text request:\n{seen}"
    );
    let line = lines[before].clone();
    eprintln!("LIVE-CACHE {context}: {line:?} text={text:?}");
    (text, line)
}

fn assert_refilled(line: &CacheLine, context: &str) {
    assert_eq!(
        (line.mode.as_str(), line.reused),
        ("FullRefill", 0),
        "{context}: {line:?}"
    );
    assert_eq!(line.prefetched, line.prompt, "{context}: {line:?}");
}

fn assert_appended(line: &CacheLine, context: &str) {
    assert_eq!(line.mode, "ExactAppend", "{context}: {line:?}");
    assert!(
        line.reused > 0 && line.reused < line.prompt,
        "{context}: {line:?}"
    );
    assert_eq!(
        line.reused + line.prefetched,
        line.prompt,
        "{context}: {line:?}"
    );
}

/// A raw streaming exchange that the test can abandon at any point.
struct OpenStream {
    reader: std::io::BufReader<TcpStream>,
}

fn open_stream(port: u16, body: &Value) -> OpenStream {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect stream");
    stream
        .set_read_timeout(Some(STREAM_READ_TIMEOUT))
        .expect("set stream read timeout");
    let payload = serde_json::to_vec(body).expect("stream body serializes");
    let head = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|()| stream.write_all(&payload))
        .expect("send streaming request");
    OpenStream {
        reader: std::io::BufReader::new(stream),
    }
}

impl OpenStream {
    /// The status code, with the headers consumed.
    fn status(&mut self) -> u16 {
        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .expect("read streaming status");
        let status = line
            .split_whitespace()
            .nth(1)
            .and_then(|status| status.parse().ok())
            .unwrap_or_else(|| panic!("no status in {line:?}"));
        loop {
            line.clear();
            self.reader
                .read_line(&mut line)
                .expect("read streaming headers");
            if line == "\r\n" || line.is_empty() {
                return status;
            }
        }
    }

    /// Content deltas until `limit` have arrived or the stream ends; the second
    /// value says whether `[DONE]` was read.
    fn content_deltas(&mut self, limit: Option<usize>) -> (Vec<String>, bool) {
        let mut deltas = Vec::new();
        let mut line = String::new();
        while limit.is_none_or(|limit| deltas.len() < limit) {
            line.clear();
            if self.reader.read_line(&mut line).expect("read stream") == 0 {
                return (deltas, false);
            }
            let Some(data) = line.trim_end().strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                return (deltas, true);
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data)
                && let Some(text) = chunk["choices"][0]["delta"]["content"].as_str()
                && !text.is_empty()
            {
                deltas.push(text.to_owned());
            }
        }
        (deltas, false)
    }

    fn disconnect(self) {
        let _ = self.reader.get_ref().shutdown(Shutdown::Both);
    }
}

// ---------------------------------------------------------------------------
// Synthetic adapters (the same construction as `serve_lora_e2e_test`)
// ---------------------------------------------------------------------------

const ADAPTER_RANK: usize = 4;
const ADAPTER_ALPHA: f32 = 8.0;
const ADAPTER_MODULES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
const ADAPTER_B_MAGNITUDE: f32 = 0.5;
/// Request scales for the conversation turns. At full scale the synthetic
/// adapters decode to text that does not re-tokenize to the generated ids, and
/// echoing such an answer cannot reproduce the tokens the worker holds.
const LIGHT: f32 = 0.05;
const LIGHTER: f32 = 0.03;
const FIRST_ADAPTER_SEED: u64 = 0x1790_0A11_CE5E_ED01;
const SECOND_ADAPTER_SEED: u64 = 0x1790_0B22_DF6F_FE02;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as f32;
        bits / 16_777_216.0 * 2.0 - 1.0
    }
}

fn adapter_bytes(cfg: &Qwen35Config, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64(seed);
    let mut header = format!("{{\"__metadata__\":{{\"alpha\":\"{ADAPTER_ALPHA}\"}}");
    let mut data: Vec<u8> = Vec::new();
    for layer in 0..cfg.num_hidden_layers {
        for module in ADAPTER_MODULES {
            let shape = qwen35_projection_shape(cfg, layer, module)
                .unwrap_or_else(|err| panic!("layer {layer} module {module}: {err}"));
            let a_scale = 1.0 / (shape.d_in as f32).sqrt();
            let prefix = format!("base_model.model.model.layers.{layer}.mlp.{module}");
            for (suffix, rows, cols, scale) in [
                ("lora_A", ADAPTER_RANK, shape.d_in, a_scale),
                ("lora_B", shape.d_out, ADAPTER_RANK, ADAPTER_B_MAGNITUDE),
            ] {
                let start = data.len();
                for _ in 0..rows * cols {
                    data.extend_from_slice(&(rng.next_unit() * scale).to_le_bytes());
                }
                header.push_str(&format!(
                    ",\"{prefix}.{suffix}.weight\":{{\"dtype\":\"F32\",\"shape\":[{rows},{cols}],\
                     \"data_offsets\":[{start},{}]}}",
                    data.len()
                ));
            }
        }
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut bytes = Vec::with_capacity(8 + header.len() + data.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    bytes
}

/// Owner-only, as the loader's trust guard requires of a mapped file.
fn write_adapter(path: &Path, cfg: &Qwen35Config, seed: u64) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .unwrap_or_else(|err| panic!("creating {}: {err}", path.display()));
    file.write_all(&adapter_bytes(cfg, seed))
        .unwrap_or_else(|err| panic!("writing {}: {err}", path.display()));
}

fn load_adapter(server: &Server, path: &Path, name: &str) -> u64 {
    let response = server.request(
        "POST",
        "/v1/lora/load",
        Some(&json!({"path": path.to_str().expect("utf-8 temp path"), "name": name})),
    );
    assert_eq!(response.status, 200, "load {name}: {}", response.body);
    let body: Value = serde_json::from_str(&response.body).expect("load body is JSON");
    body["id"].as_u64().expect("load body carries an id")
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

fn image_data_uri() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("vision")
        .join("golden_image.png");
    let image =
        std::fs::read(&path).unwrap_or_else(|err| panic!("reading {}: {err}", path.display()));
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(image)
    )
}

fn image_message(role: &str, text: &str, uri: &str) -> Value {
    json!({
        "role": role,
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": uri}}
        ]
    })
}

fn error_code(response: &HttpResponse) -> String {
    response
        .error_code()
        .unwrap_or_else(|| format!("<no error code in {}>", response.body))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn stderr_line_parsers_read_the_worker_formats() {
    let stderr = "[metal-worker] cross-turn cache: mode=ExactAppend reused=31 prefetched=9 prompt=40\n\
                  [metal-worker] cross-turn cache: mode=ReplayFromCheckpoint { checkpoint_len: 12 } reused=12 prefetched=8 prompt=20\n\
                  [route] served family=qwen35 backend=metal mode=stream driver=shared opened=5 consumed=4\n\
                  [route] served family=qwen35 backend=metal mode=nonstream driver=bypassed opened=0 consumed=0\n\
                  [route] served family=qwen35 backend=cpu mode=stream driver=shared opened=1 consumed=1\n\
                  [metal-worker] route=vision dispatch=multimodal metal_gemm_dispatches=337 metal_gemm_calls=337\n";
    assert_eq!(
        cache_lines(stderr),
        vec![
            CacheLine {
                mode: "ExactAppend".to_owned(),
                reused: 31,
                prefetched: 9,
                prompt: 40
            },
            CacheLine {
                mode: "ReplayFromCheckpoint { checkpoint_len: 12 }".to_owned(),
                reused: 12,
                prefetched: 8,
                prompt: 20
            },
        ]
    );
    assert_eq!(
        served_lines(stderr),
        vec![
            Served {
                stream: true,
                driver: "shared".to_owned(),
                opened: 5,
                consumed: 4
            },
            Served {
                stream: false,
                driver: "bypassed".to_owned(),
                opened: 0,
                consumed: 0
            },
        ]
    );
    assert_eq!(vision_lines(stderr), vec![(337, 337)]);
}

#[test]
fn the_worker_route_reuses_an_appended_conversation_and_refills_a_diverged_one() {
    let _gpu = acquire_gpu_lock();
    let checkpoints = qwen_checkpoints("live-prefix");
    for (binary, checkpoint) in targets(&checkpoints) {
        let server = spawn(binary, checkpoint, &[]);
        let format = &label(binary, checkpoint);
        let first = [user("Name one primary color.")];
        let (answer, line) = text_turn(&server, &first, json!({}), &format!("{format} cold"));
        assert_refilled(&line, &format!("{format}: a cold worker refills"));
        let cold_prompt = line.prompt;

        // The next turn as a client sends it, with the answer it received as
        // history: the template renders that turn without the think block the
        // generation prompt ended with, so the tokens part right after the
        // assistant header and nothing is reused.
        let plain = [
            user("Name one primary color."),
            assistant(&answer),
            user("Name another one."),
        ];
        let (_, line) = text_turn(
            &server,
            &plain,
            json!({}),
            &format!("{format} plain append"),
        );
        assert_refilled(
            &line,
            &format!("{format}: the next turn with the answer as plain history"),
        );

        let (again, line) = text_turn(&server, &first, json!({}), &format!("{format} cold again"));
        assert_refilled(
            &line,
            &format!("{format}: the first turn after the plain append"),
        );
        assert_eq!(again, answer, "{format}: the first turn's answer changed");

        // The same turn with the think block echoed into the history reproduces
        // the held tokens through the whole answer.
        let appended = [
            user("Name one primary color."),
            echoed(&answer),
            user("Name another one."),
        ];
        let (appended_answer, line) =
            text_turn(&server, &appended, json!({}), &format!("{format} append"));
        assert_appended(
            &line,
            &format!("{format}: the next turn with the think block echoed"),
        );
        assert!(line.reused > cold_prompt, "{format}: {line:?}");

        let (_, line) = text_turn(
            &server,
            &[user("Write one word about the sea.")],
            json!({}),
            &format!("{format} diverged"),
        );
        assert_refilled(&line, &format!("{format}: a different conversation"));

        // The same appended request on a worker whose prefix no longer matches
        // it: a full refill that must give the answer the reused prefix gave.
        let (refilled_answer, line) =
            text_turn(&server, &appended, json!({}), &format!("{format} refill"));
        assert_refilled(
            &line,
            &format!("{format}: the appended request after a diverged one"),
        );
        assert_eq!(
            refilled_answer, appended_answer,
            "{format}: reusing the prefix changed the answer"
        );

        // An exact repeat of the request just served: its prompt is a strict
        // prefix of what the worker now holds, so there is no suffix to append.
        let (repeated_answer, line) =
            text_turn(&server, &appended, json!({}), &format!("{format} repeat"));
        assert_ne!(line.mode, "ExactAppend", "{format}: {line:?}");
        assert_eq!(
            repeated_answer, appended_answer,
            "{format}: a repeated request changed its answer"
        );
    }
}

#[test]
fn a_disconnect_ends_the_worker_request_and_the_next_request_is_unchanged() {
    let _gpu = acquire_gpu_lock();
    let checkpoints = qwen_checkpoints("live-cancel");
    for (binary, checkpoint) in targets(&checkpoints) {
        let server = spawn(binary, checkpoint, &[]);
        let format = &label(binary, checkpoint);
        let counting = [user(COUNT_PROMPT)];
        let streaming = chat_body(&server, &counting, COUNT_TOKENS, json!({"stream": true}));

        let control = chat(&server, &counting, COUNT_TOKENS, json!({}));
        assert_eq!(control.status, 200, "{format}: {}", control.body);
        let control_tokens = completion_tokens(&control);
        let control_text = response_text(&control);
        assert!(
            control_tokens >= COUNT_TOKENS / 2,
            "{format}: the counting prompt must decode long enough to tell a stopped request \
             from a finished one, got {control_tokens} tokens"
        );

        // 1. Mid-decode: read a few deltas, then hang up.
        let streams_before = served_lines(&server.diagnostics())
            .iter()
            .filter(|served| served.stream)
            .count();
        let mut stream = open_stream(server.port(), &streaming);
        assert_eq!(stream.status(), 200, "{format}");
        let (deltas, done) = stream.content_deltas(Some(3));
        assert_eq!((deltas.len(), done), (3, false), "{format}");
        stream.disconnect();
        let seen = wait_for_stderr(&server, "the mid-decode stream's served line", |seen| {
            served_lines(seen)
                .iter()
                .filter(|served| served.stream)
                .count()
                > streams_before
        });
        let stopped = served_lines(&seen)
            .into_iter()
            .filter(|served| served.stream)
            .nth(streams_before)
            .expect("the stream's served line");
        eprintln!("LIVE-CANCEL {format} mid-decode: {stopped:?}");
        assert!(
            stopped.consumed >= 3 && stopped.consumed < control_tokens / 2,
            "{format}: a stream abandoned after 3 deltas decoded {} of {control_tokens} tokens",
            stopped.consumed
        );
        let after = chat(&server, &counting, COUNT_TOKENS, json!({}));
        assert_eq!(
            (
                after.status,
                completion_tokens(&after),
                response_text(&after)
            ),
            (200, control_tokens, control_text.clone()),
            "{format}: the request after a disconnect"
        );

        // 2. During prefill: a long prompt, abandoned as soon as it is sent.
        let long_prompt = format!(
            "{} Summarize the text above in one word.",
            "The quick brown fox jumps over the lazy dog. ".repeat(400)
        );
        let long_body = chat_body(
            &server,
            &[user(&long_prompt)],
            COUNT_TOKENS,
            json!({"stream": true}),
        );
        // The hang-up races the worker picking the request up: either the
        // request is dropped before it runs and leaves no line, or it runs and
        // stops early. The next request's own served line bounds the wait.
        let served_before = served_lines(&server.diagnostics()).len();
        open_stream(server.port(), &long_body).disconnect();
        let next = chat(&server, &counting, COUNT_TOKENS, json!({}));
        assert_eq!(
            (next.status, completion_tokens(&next), response_text(&next)),
            (200, control_tokens, control_text.clone()),
            "{format}: the request after a prefill disconnect"
        );
        let seen = wait_for_stderr(
            &server,
            "the request after the prefill disconnect",
            |seen| {
                served_lines(seen)[served_before..]
                    .iter()
                    .any(|served| !served.stream)
            },
        );
        let abandoned: Vec<_> = served_lines(&seen)[served_before..]
            .iter()
            .filter(|served| served.stream)
            .cloned()
            .collect();
        eprintln!("LIVE-CANCEL {format} prefill: {abandoned:?}");
        assert!(
            abandoned.len() <= 1
                && abandoned
                    .iter()
                    .all(|served| served.consumed < control_tokens / 2),
            "{format}: a stream abandoned during prefill: {abandoned:?}"
        );

        // 3. While queued: A decodes, B arrives behind it and hangs up, C follows.
        let served_before = served_lines(&server.diagnostics()).len();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let port = server.port();
        let first_body = streaming.clone();
        let first = std::thread::spawn(move || {
            let mut stream = open_stream(port, &first_body);
            let status = stream.status();
            let (head, _) = stream.content_deltas(Some(1));
            let _ = started_tx.send(());
            let (tail, done) = stream.content_deltas(None);
            (status, head.len() + tail.len(), done)
        });
        started_rx
            .recv_timeout(STREAM_READ_TIMEOUT)
            .expect("the first stream started decoding");
        open_stream(server.port(), &streaming).disconnect();
        let (status, deltas, done) = first.join().expect("first stream thread");
        assert_eq!((status, done), (200, true), "{format}");
        assert!(
            deltas > 3,
            "{format}: the first stream ran: {deltas} deltas"
        );
        let last = chat(&server, &counting, COUNT_TOKENS, json!({}));
        assert_eq!(
            (last.status, completion_tokens(&last), response_text(&last)),
            (200, control_tokens, control_text),
            "{format}: the request after a queued disconnect"
        );
        let seen = wait_for_stderr(&server, "the request after the queue", |seen| {
            served_lines(seen)[served_before..]
                .iter()
                .any(|served| !served.stream)
        });
        let after_queue: Vec<_> = served_lines(&seen)[served_before..].to_vec();
        eprintln!("LIVE-CANCEL {format} queued: {after_queue:?}");
        assert_eq!(
            after_queue.iter().filter(|served| served.stream).count(),
            1,
            "{format}: only the first stream reached the model; the queued one that hung up \
             never did: {after_queue:?}"
        );
    }
}

#[test]
fn adapter_changes_refill_the_prefix_and_an_unchanged_selection_keeps_it() {
    let _gpu = acquire_gpu_lock();
    let checkpoints = qwen_checkpoints("live-lora");
    for (binary, checkpoint) in targets(&checkpoints) {
        let format = &label(binary, checkpoint);
        let cfg = Qwen35Config::from_model_dir(&checkpoint.dir)
            .unwrap_or_else(|err| panic!("{format}: reading config.json: {err}"));
        let adapters = tempfile::tempdir().expect("adapter dir");
        let first_path = adapters.path().join("first.safetensors");
        let second_path = adapters.path().join("second.safetensors");
        write_adapter(&first_path, &cfg, FIRST_ADAPTER_SEED);
        write_adapter(&second_path, &cfg, SECOND_ADAPTER_SEED);

        let server = spawn(binary, checkpoint, &[]);
        let probe = [user("Write one word about the sea.")];
        let (base_probe, line) = text_turn(&server, &probe, json!({}), &format!("{format} probe"));
        assert_refilled(&line, &format!("{format}: cold probe"));

        let mut conversation = vec![user("Name one primary color.")];
        let mut turn = |server: &Server, extra: Value, context: &str| -> CacheLine {
            let (answer, line) = text_turn(server, &conversation, extra, context);
            conversation.push(echoed(&answer));
            conversation.push(user("Name another one."));
            line
        };

        let line = turn(&server, json!({}), &format!("{format} base"));
        assert_refilled(&line, &format!("{format}: the conversation's first turn"));

        // Loading makes an adapter resident and leaves the applied set alone.
        let first = load_adapter(&server, &first_path, "first");
        let line = turn(&server, json!({}), &format!("{format} after load"));
        assert_appended(
            &line,
            &format!("{format}: a load does not touch the prefix"),
        );

        let selected = |entries: &[(u64, f32)]| {
            json!({"lora": entries
                .iter()
                .map(|(id, scale)| json!({"id": id, "scale": scale}))
                .collect::<Vec<_>>()})
        };
        let line = turn(
            &server,
            selected(&[(first, LIGHT)]),
            &format!("{format} apply"),
        );
        assert_refilled(&line, &format!("{format}: applying an adapter"));
        let line = turn(
            &server,
            selected(&[(first, LIGHT)]),
            &format!("{format} same"),
        );
        assert_appended(&line, &format!("{format}: the same selection again"));
        let line = turn(
            &server,
            selected(&[(first, LIGHTER)]),
            &format!("{format} scale"),
        );
        assert_refilled(&line, &format!("{format}: a changed scale"));

        let second = load_adapter(&server, &second_path, "second");
        let line = turn(
            &server,
            selected(&[(first, LIGHT), (second, LIGHT)]),
            &format!("{format} pair"),
        );
        assert_refilled(&line, &format!("{format}: a two-adapter mix"));
        let line = turn(
            &server,
            selected(&[(second, LIGHT), (first, LIGHT)]),
            &format!("{format} swapped"),
        );
        assert_refilled(&line, &format!("{format}: the same mix in the other order"));
        let line = turn(
            &server,
            selected(&[(second, LIGHT), (first, LIGHT)]),
            &format!("{format} swapped again"),
        );
        assert_appended(&line, &format!("{format}: the swapped mix again"));

        let line = turn(&server, json!({}), &format!("{format} absent"));
        assert_refilled(&line, &format!("{format}: no selection after a mix"));
        let line = turn(&server, json!({}), &format!("{format} absent again"));
        assert_appended(&line, &format!("{format}: no selection again"));

        // Unloading an applied adapter takes it off the worker even though the
        // next request's empty selection matches the empty applied set.
        let line = turn(
            &server,
            selected(&[(first, LIGHT)]),
            &format!("{format} reapply"),
        );
        assert_refilled(&line, &format!("{format}: re-applying the first adapter"));
        let response = server.request("POST", "/v1/lora/unload", Some(&json!({"id": first})));
        assert_eq!(response.status, 200, "{format}: {}", response.body);
        let line = turn(&server, json!({}), &format!("{format} after unload"));
        assert_refilled(
            &line,
            &format!("{format}: base after unloading the applied adapter"),
        );

        let (second_probe, _) = text_turn(
            &server,
            &probe,
            selected(&[(second, 1.0)]),
            &format!("{format} second probe"),
        );
        assert_ne!(
            second_probe, base_probe,
            "{format}: the second adapter left the probe answer unchanged"
        );
        let (final_probe, line) =
            text_turn(&server, &probe, json!({}), &format!("{format} final probe"));
        assert_refilled(&line, &format!("{format}: base probe after the adapters"));
        assert_eq!(
            final_probe, base_probe,
            "{format}: the base answer did not come back after the adapters"
        );
    }
}

#[test]
fn image_requests_reach_the_vision_route_and_leave_no_text_prefix_behind() {
    let _gpu = acquire_gpu_lock();
    let uri = image_data_uri();
    let checkpoints = qwen_checkpoints("live-vision");
    for (binary, checkpoint) in targets(&checkpoints) {
        for preload in [false, true] {
            let format = label(binary, checkpoint);
            let loading = if preload { "preloaded" } else { "lazy" };
            let extra: Vec<String> = if preload {
                vec!["--preload-vision".to_owned()]
            } else {
                Vec::new()
            };
            let server = spawn(binary, checkpoint, &extra);
            let context = format!("{format} {loading}");

            let conversation = [user("Name one primary color.")];
            let (answer, line) = text_turn(&server, &conversation, json!({}), &context);
            assert_refilled(&line, &format!("{context}: cold"));

            let cache_before = cache_lines(&server.diagnostics()).len();
            let served_before = served_lines(&server.diagnostics()).len();
            let image = chat(
                &server,
                &[image_message("user", "Describe this image.", &uri)],
                8,
                json!({}),
            );
            assert_eq!(
                image.status,
                200,
                "{context}: {}\n{}",
                image.body,
                server.diagnostics()
            );
            assert!(
                !response_text(&image).trim().is_empty(),
                "{context}: {}",
                image.body
            );
            let seen = wait_for_stderr(&server, "the image request's served line", |seen| {
                served_lines(seen).len() > served_before
            });
            let vision = vision_lines(&seen);
            assert_eq!(vision.len(), 1, "{context}: {seen}");
            assert!(
                vision[0].0 > 0 && vision[0].0 == vision[0].1,
                "{context}: every vision GEMM dispatches to Metal: {vision:?}"
            );
            assert_eq!(
                served_lines(&seen)[served_before],
                Served {
                    stream: false,
                    driver: "bypassed".to_owned(),
                    opened: 0,
                    consumed: 0
                },
                "{context}: the image request is served outside the shared text driver"
            );
            assert_eq!(
                cache_lines(&seen).len(),
                cache_before,
                "{context}: the image request runs no text prefix plan"
            );

            // The image request replaced the worker's state: the next turn of
            // the earlier text conversation, which would otherwise append,
            // refills, and the one after reuses.
            let appended = [
                user("Name one primary color."),
                echoed(&answer),
                user("Name another one."),
            ];
            let (second_answer, line) = text_turn(&server, &appended, json!({}), &context);
            assert_refilled(&line, &format!("{context}: the text turn after an image"));
            let mut next = appended.to_vec();
            next.push(echoed(&second_answer));
            next.push(user("And one more."));
            let (_, line) = text_turn(&server, &next, json!({}), &context);
            assert_appended(&line, &format!("{context}: the turn after that"));

            for (messages, extra, code) in [
                (
                    vec![json!({"role": "user", "content": [
                        {"type": "text", "text": "Compare."},
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "image_url", "image_url": {"url": uri}}
                    ]})],
                    json!({}),
                    "multiple_images_unsupported",
                ),
                (
                    vec![
                        user("Hello."),
                        image_message("assistant", "Here is a picture.", &uri),
                        user("What was it?"),
                    ],
                    json!({}),
                    "invalid_image_role",
                ),
                (
                    vec![image_message(
                        "user",
                        "Describe this image.",
                        "data:image/png;base64,AAAA",
                    )],
                    json!({}),
                    "invalid_image",
                ),
                (
                    vec![image_message("user", "Describe this image.", &uri)],
                    json!({"stream": true}),
                    "unsupported_feature",
                ),
            ] {
                let response = chat(&server, &messages, 8, extra);
                assert_eq!(
                    (response.status, error_code(&response).as_str()),
                    (400, code),
                    "{context}: {}",
                    response.body
                );
            }
            let seen = server.diagnostics();
            assert_eq!(
                vision_lines(&seen).len(),
                1,
                "{context}: the refusals never reached the vision encoder: {seen}"
            );
        }
    }
}
