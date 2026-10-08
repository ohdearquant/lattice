//! Standalone `lattice_serve` family and backend routing, driven through the
//! shipped binary.
//!
//! The first group starts the real process on directories that hold a
//! `config.json` and a stub weight file (or nothing at all), so the process
//! reaches its routing decision and then its loader, or ends before either.
//! What is asserted is the route the process reports and the loader or refusal
//! it reaches. The stub weights are never loadable, so every one of these cases
//! ends in a startup error and none of them listens. They need no checkpoint
//! and no GPU.
//!
//! The second group serves a real checkpoint over HTTP: Gemma 4 E2B on the CPU
//! and Qwen3.5 on the Metal worker, each answering a non-streaming and a
//! streaming chat request. A checkpoint that is absent is a loud skip line
//! (`LATTICE_SERVE_STANDALONE_SKIPPED`), never a pass: a skipped test still
//! prints `ok` under libtest, so the line is what tells the two apart, and
//! `LATTICE_SERVE_STANDALONE_GATE_ENFORCE=1` turns the same condition into a
//! failure.
//!
//! ```bash
//! LATTICE_SERVE_STANDALONE_GATE_ENFORCE=1 cargo test --release \
//!   -p lattice-inference --features f16,metal-gpu,serve \
//!   --test serve_standalone_route -- --nocapture
//! ```

#![cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "f16",
    feature = "serve"
))]

use lattice_inference::measurement::gpu_test_lock;
use serde_json::{Value, json};
use std::io::{BufRead as _, Read as _, Write as _};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

const ENFORCE_ENV: &str = "LATTICE_SERVE_STANDALONE_GATE_ENFORCE";
const GEMMA_DIR_ENV: &str = "LATTICE_SERVE_STANDALONE_GEMMA_DIR";
const QWEN_DIR_ENV: &str = "LATTICE_SERVE_STANDALONE_QWEN_DIR";
const HEALTH_TIMEOUT_ENV: &str = "LATTICE_SERVE_STANDALONE_HEALTH_TIMEOUT_SECS";
const STARTUP_ERROR_DEADLINE: Duration = Duration::from_secs(120);

// ---------------------------------------------------------------------------
// Startup decisions on stub directories
// ---------------------------------------------------------------------------

struct Run {
    code: Option<i32>,
    stderr: String,
}

fn gemma_config() -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("gemma4")
            .join("e2b_config.json"),
    )
    .expect("committed Gemma 4 config fixture")
}

fn qwen_config() -> Vec<u8> {
    br#"{"model_type": "qwen3_5"}"#.to_vec()
}

fn checkpoint_dir(config: &[u8], weights_file: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp model dir");
    std::fs::write(dir.path().join("config.json"), config).expect("write config.json");
    std::fs::write(dir.path().join(weights_file), b"stub").expect("write stub weights");
    dir
}

/// Run `lattice_serve` on `model` with `extra` arguments and wait for it to
/// exit. A server that starts listening instead is killed at the deadline and
/// fails the test.
fn start(model: &Path, extra: &[&str]) -> Run {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lattice_serve"))
        .args(["--host", "127.0.0.1", "--port", "0", "--model"])
        .arg(model)
        .args(extra)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lattice_serve");
    let started = Instant::now();
    loop {
        if child.try_wait().expect("poll lattice_serve").is_some() {
            break;
        }
        if started.elapsed() > STARTUP_ERROR_DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            panic!("lattice_serve kept running on a directory that cannot load");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let output = child
        .wait_with_output()
        .expect("collect lattice_serve output");
    let run = Run {
        code: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    };
    // Shown under `--nocapture`, so the process's own words are on the record.
    eprintln!(
        "lattice_serve exit={:?} stderr:\n{}",
        run.code,
        run.stderr.trim_end()
    );
    run
}

#[test]
fn gemma_safetensors_directory_selects_the_gemma_cpu_route_and_its_loader() {
    let dir = checkpoint_dir(&gemma_config(), "model.safetensors");
    let run = start(dir.path(), &[]);
    assert!(
        run.stderr
            .contains("[route] selected family=gemma4 backend=cpu format=safetensors"),
        "{}",
        run.stderr
    );
    assert!(
        run.stderr
            .contains("lattice_serve: Gemma 4 model load failed:"),
        "the Gemma loader is the one that ran: {}",
        run.stderr
    );
    assert!(!run.stderr.contains("ready"), "{}", run.stderr);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn qwen_safetensors_directory_keeps_the_metal_worker_route() {
    let dir = checkpoint_dir(&qwen_config(), "model.safetensors");
    let run = start(dir.path(), &[]);
    assert!(
        run.stderr
            .contains("[route] selected family=qwen35 backend=metal format=safetensors"),
        "{}",
        run.stderr
    );
    assert!(!run.stderr.contains("Gemma"), "{}", run.stderr);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn gemma_in_the_metal_format_is_refused_before_any_loader_runs() {
    let dir = checkpoint_dir(&gemma_config(), "model_layers_0_weight.q4");
    let run = start(dir.path(), &[]);
    assert!(
        run.stderr
            .contains("lattice_serve: gemma_metal_unsupported: "),
        "{}",
        run.stderr
    );
    assert!(
        !run.stderr.contains("[route] selected"),
        "a refused checkpoint selects no route: {}",
        run.stderr
    );
    assert!(
        !run.stderr.contains("loading model") && !run.stderr.contains("load failed"),
        "no loader ran: {}",
        run.stderr
    );
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn qwen_in_the_metal_format_is_not_caught_by_the_gemma_refusal() {
    let dir = checkpoint_dir(&qwen_config(), "model_layers_0_weight.q4");
    let run = start(dir.path(), &[]);
    assert!(
        run.stderr
            .contains("[route] selected family=qwen35 backend=metal format=q4"),
        "{}",
        run.stderr
    );
    assert!(
        !run.stderr.contains("gemma_metal_unsupported"),
        "{}",
        run.stderr
    );
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn a_missing_model_directory_keeps_its_startup_error() {
    let parent = tempfile::tempdir().expect("temp dir");
    let missing = parent.path().join("absent");
    let run = start(&missing, &[]);
    assert_eq!(
        run.stderr.trim_end(),
        format!(
            "lattice_serve: model directory not found: {}",
            missing.display()
        )
    );
    assert_eq!(run.code, Some(1));
}

#[test]
fn a_directory_with_no_recognized_format_keeps_the_error_it_always_reported() {
    let dir = tempfile::tempdir().expect("temp model dir");
    let run = start(dir.path(), &[]);
    assert!(
        run.stderr
            .contains("lattice_serve: config.json load failed:"),
        "{}",
        run.stderr
    );
    assert!(
        !run.stderr.contains("[route]"),
        "no route was selected: {}",
        run.stderr
    );
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn gemma_refuses_startup_options_it_cannot_honor_before_loading_anything() {
    let dir = checkpoint_dir(&gemma_config(), "model.safetensors");
    for flag in ["--preload-vision", "--max-resident-adapters"] {
        let extra: Vec<&str> = if flag == "--preload-vision" {
            vec![flag]
        } else {
            vec![flag, "4"]
        };
        let run = start(dir.path(), &extra);
        assert!(
            run.stderr.contains(&format!(
                "lattice_serve: unsupported_feature: {flag} is not supported for Gemma 4 checkpoints"
            )),
            "{flag}: {}",
            run.stderr
        );
        assert!(
            !run.stderr.contains("load failed") && !run.stderr.contains("[route] selected"),
            "{flag}: nothing started: {}",
            run.stderr
        );
        assert_eq!(run.code, Some(1), "{flag}: {}", run.stderr);
    }
}

// ---------------------------------------------------------------------------
// Real checkpoints over HTTP
// ---------------------------------------------------------------------------

fn enforce() -> bool {
    std::env::var(ENFORCE_ENV).as_deref() == Ok("1")
}

/// The explicit variable is authoritative: when it is set and does not exist
/// the test skips (or fails under enforcement) instead of silently using the
/// default checkpoint.
fn require_checkpoint(env_name: &str, default_name: &str) -> Option<PathBuf> {
    let path = match std::env::var_os(env_name) {
        Some(value) => PathBuf::from(value),
        None => PathBuf::from(std::env::var_os("HOME")?)
            .join(".lattice")
            .join("models")
            .join(default_name),
    };
    if path.join("config.json").is_file() {
        return Some(path);
    }
    assert!(
        !enforce(),
        "no checkpoint at {} ({env_name}) while {ENFORCE_ENV}=1",
        path.display()
    );
    eprintln!(
        "LATTICE_SERVE_STANDALONE_SKIPPED reason=no_checkpoint path={} ({env_name})",
        path.display()
    );
    None
}

struct Server {
    child: Child,
    port: u16,
    stderr: Arc<Mutex<String>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_loopback_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("ephemeral port must bind")
        .local_addr()
        .expect("bound listener must have an address")
        .port()
}

fn health_wait_timeout() -> Duration {
    Duration::from_secs(
        std::env::var(HEALTH_TIMEOUT_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(300),
    )
}

/// Decode a `Transfer-Encoding: chunked` body.
fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(line_end) = raw.windows(2).position(|pair| pair == b"\r\n") else {
            return out;
        };
        let size = std::str::from_utf8(&raw[..line_end]).ok().and_then(|line| {
            usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16).ok()
        });
        let Some(size) = size else { return out };
        raw = &raw[line_end + 2..];
        if size == 0 || raw.len() < size {
            return out;
        }
        out.extend_from_slice(&raw[..size]);
        raw = raw.get(size + 2..).unwrap_or(&[]);
    }
}

impl Server {
    fn spawn(model_dir: &Path) -> Self {
        let port = free_loopback_port();
        let child = Command::new(env!("CARGO_BIN_EXE_lattice_serve"))
            .arg("--model")
            .arg(model_dir)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("lattice_serve must spawn");
        let mut server = Self {
            child,
            port,
            stderr: Arc::new(Mutex::new(String::new())),
        };
        let stderr_pipe = server
            .child
            .stderr
            .take()
            .expect("child stderr must be captured");
        let stderr = Arc::clone(&server.stderr);
        std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stderr_pipe);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => stderr
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push_str(&line),
                }
            }
        });
        let budget = health_wait_timeout();
        let started = Instant::now();
        loop {
            if matches!(server.try_request("GET", "/health", None), Some((200, _))) {
                break;
            }
            assert!(
                server
                    .child
                    .try_wait()
                    .expect("poll lattice_serve")
                    .is_none(),
                "lattice_serve exited before it listened; stderr:\n{}",
                server.diagnostics()
            );
            assert!(
                started.elapsed() < budget,
                "lattice_serve did not become healthy after {:?} (override with \
                 {HEALTH_TIMEOUT_ENV}); stderr:\n{}",
                started.elapsed(),
                server.diagnostics()
            );
            std::thread::sleep(Duration::from_millis(200));
        }
        eprintln!("lattice_serve healthy after {:?}", started.elapsed());
        server
    }

    fn diagnostics(&self) -> String {
        self.stderr
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// One HTTP/1.1 exchange on a fresh connection, read to the end of the
    /// response. `None` when the connection itself failed.
    fn try_request(&self, method: &str, path: &str, body: Option<&Value>) -> Option<(u16, String)> {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_secs(900)))
            .ok()?;
        let payload = body.map_or_else(Vec::new, |body| {
            serde_json::to_vec(body).expect("request body must serialize")
        });
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            payload.len()
        );
        stream.write_all(head.as_bytes()).ok()?;
        stream.write_all(&payload).ok()?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).ok()?;
        let split = raw.windows(4).position(|window| window == b"\r\n\r\n")?;
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let status = head.split_whitespace().nth(1)?.parse().ok()?;
        let body = &raw[split + 4..];
        let body = if head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            dechunk(body)
        } else {
            body.to_vec()
        };
        Some((status, String::from_utf8_lossy(&body).into_owned()))
    }

    fn request(&self, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
        self.try_request(method, path, body).unwrap_or_else(|| {
            panic!(
                "{method} {path} failed at the transport level; stderr:\n{}",
                self.diagnostics()
            )
        })
    }

    fn chat(&self, extra: Value) -> (u16, String) {
        let mut body = json!({
            "messages": [{"role": "user", "content": "Name three primary colors."}],
            "max_tokens": 8,
            "temperature": 0.0,
        });
        if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            body.extend(extra.clone());
        }
        self.request("POST", "/v1/chat/completions", Some(&body))
    }

    /// The server's stderr once it holds `needle`, or whatever it holds after
    /// a grace period for the draining thread.
    fn stderr_with(&self, needle: &str) -> String {
        let started = Instant::now();
        loop {
            let seen = self.diagnostics();
            if seen.contains(needle) || started.elapsed() > Duration::from_secs(10) {
                return seen;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

fn error_code(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| value["error"]["code"].as_str().map(str::to_owned))
        .unwrap_or_else(|| format!("<no error code in {body}>"))
}

fn assert_answers_chat(server: &Server) {
    let (status, body) = server.chat(json!({}));
    assert_eq!(
        status,
        200,
        "non-streaming chat: {body}\n{}",
        server.diagnostics()
    );
    let value: Value = serde_json::from_str(&body).expect("chat body is JSON");
    assert!(
        value["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|text| !text.is_empty()),
        "{body}"
    );
    assert!(
        value["usage"]["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "{body}"
    );

    let (status, body) = server.chat(json!({"stream": true}));
    assert_eq!(
        status,
        200,
        "streaming chat: {body}\n{}",
        server.diagnostics()
    );
    assert!(body.contains("data: [DONE]"), "{body}");
    assert!(
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .any(|chunk| chunk["choices"][0]["delta"]["content"]
                .as_str()
                .is_some_and(|text| !text.is_empty())),
        "a stream carries at least one content delta: {body}"
    );
}

fn assert_shared_route_markers(stderr: &str, family: &str, backend: &str) {
    for mode in ["nonstream", "stream"] {
        let prefix = format!(
            "[route] served family={family} backend={backend} mode={mode} driver=shared opened="
        );
        let marker = stderr
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("missing {mode} shared-route marker:\n{stderr}"));
        let fields = marker.split_whitespace().collect::<Vec<_>>();
        let opened = fields
            .iter()
            .find_map(|field| field.strip_prefix("opened="))
            .and_then(|value| value.parse::<usize>().ok());
        let consumed = fields
            .iter()
            .find_map(|field| field.strip_prefix("consumed="))
            .and_then(|value| value.parse::<usize>().ok());
        assert!(opened.is_some_and(|count| count > 0), "{marker}");
        assert!(consumed.is_some_and(|count| count > 0), "{marker}");
        eprintln!("R13-MARKER {marker}");
    }
}

#[test]
fn dechunk_joins_chunks_and_stops_at_the_terminator() {
    assert_eq!(
        dechunk(b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"),
        b"hello world"
    );
    assert_eq!(
        dechunk(b"a;ext=1\r\n0123456789\r\n0\r\n\r\n"),
        b"0123456789"
    );
    assert_eq!(dechunk(b""), b"");
}

#[test]
fn gemma_e2b_is_served_on_the_cpu_by_the_standalone_server() {
    let Some(dir) = require_checkpoint(GEMMA_DIR_ENV, "gemma-4-e2b-it") else {
        return;
    };
    let server = Server::spawn(&dir);
    assert!(
        server
            .diagnostics()
            .contains("[route] selected family=gemma4 backend=cpu format=safetensors"),
        "{}",
        server.diagnostics()
    );

    assert_answers_chat(&server);
    let seen = server.stderr_with("mode=stream driver=shared");
    assert_shared_route_markers(&seen, "gemma4", "cpu");

    for (extra, code) in [
        (json!({"stop": ["x"]}), "unsupported_feature"),
        (json!({"reasoning_budget": 8}), "unsupported_feature"),
        (
            json!({"lora": [{"id": 1, "scale": 1.0}]}),
            "lora_unsupported_backend",
        ),
        (
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}}),
            "unsupported_feature",
        ),
        (
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}),
            "unsupported_feature",
        ),
    ] {
        let (status, body) = server.chat(extra.clone());
        assert_eq!(status, 400, "{extra}: {body}");
        assert_eq!(error_code(&body), code, "{extra}: {body}");
    }
    let (status, body) = server.request("GET", "/v1/lora", None);
    assert_eq!(
        (status, error_code(&body).as_str()),
        (400, "lora_unsupported_backend")
    );
    let (status, body) = server.request(
        "POST",
        "/v1/lora/load",
        Some(&json!({"path": "/nonexistent", "name": "x"})),
    );
    assert_eq!(
        (status, error_code(&body).as_str()),
        (400, "lora_unsupported_backend")
    );
    let (status, body) = server.request("POST", "/v1/embeddings", Some(&json!({"input": "hi"})));
    assert_eq!(
        (status, error_code(&body).as_str()),
        (503, "embedding_model_not_loaded")
    );
}

#[test]
fn qwen_is_served_on_the_metal_worker_by_the_standalone_server() {
    let Some(dir) = require_checkpoint(QWEN_DIR_ENV, "qwen3.5-0.8b") else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    let server = Server::spawn(&dir);
    let format = if std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| entry.file_name().to_string_lossy().ends_with(".q4"))
        && !dir.join("model.safetensors").exists()
    {
        "q4"
    } else {
        "safetensors"
    };
    let selected = format!("[route] selected family=qwen35 backend=metal format={format}");
    assert!(
        server.diagnostics().contains(&selected),
        "{}",
        server.diagnostics()
    );

    assert_answers_chat(&server);
    let seen = server.stderr_with("mode=stream driver=shared");
    assert_shared_route_markers(&seen, "qwen35", "metal");
    // Qwen keeps the features Gemma refuses: a stop string is admitted.
    let (status, body) = server.chat(json!({"stop": ["\n\n"]}));
    assert_eq!(status, 200, "{body}");
    let (status, body) = server.request("GET", "/v1/lora", None);
    assert_eq!(status, 200, "{body}");
}
