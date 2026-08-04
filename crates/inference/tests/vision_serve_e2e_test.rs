//! ADR-069 S6 gate: post an inline image to the real `lattice_serve`
//! process and prove that the shared HTTP contract reaches the production
//! vision decode route.
//!
//! Model-gated: `LATTICE_VISION_S3_MODEL_DIR` wins, followed by
//! `~/.lattice/models/qwen3.5-0.8b`. A missing checkpoint emits a loud skip;
//! `LATTICE_VISION_S3_GATE_ENFORCE=1` makes the same condition fail closed.
//! The Mac mini gate should run:
//!
//! ```bash
//! LATTICE_VISION_S3_GATE_ENFORCE=1 cargo test --release \
//!   -p lattice-inference --features f16,metal-gpu \
//!   --test vision_serve_e2e_test -- --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "metal-gpu"))]

use base64::Engine as _;
use lattice_inference::measurement::gpu_test_lock;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const VISION_DISPATCH_MARKER: &str = "route=vision dispatch=multimodal";
const METAL_DISPATCH_FIELD: &str = "metal_gemm_dispatches=";
const GEMM_CALL_FIELD: &str = "metal_gemm_calls=";

fn enforce() -> bool {
    std::env::var("LATTICE_VISION_S3_GATE_ENFORCE").as_deref() == Ok("1")
}

fn expand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}/{rest}");
    }
    path.to_string()
}

fn default_model_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(
        PathBuf::from(home)
            .join(".lattice")
            .join("models")
            .join("qwen3.5-0.8b"),
    )
}

fn require_model_dir() -> Option<PathBuf> {
    const MODEL_DIR_ENV: &str = "LATTICE_VISION_S3_MODEL_DIR";
    if let Ok(value) = std::env::var(MODEL_DIR_ENV) {
        let path = PathBuf::from(expand_home(&value));
        if path.exists() {
            return Some(path);
        }
        if enforce() {
            panic!(
                "{MODEL_DIR_ENV}={} does not exist while \
                 LATTICE_VISION_S3_GATE_ENFORCE=1",
                path.display()
            );
        }
    }
    if let Some(path) = default_model_dir()
        && path.exists()
    {
        return Some(path);
    }
    if enforce() {
        panic!(
            "no vision checkpoint found via {MODEL_DIR_ENV} or \
             ~/.lattice/models/qwen3.5-0.8b while \
             LATTICE_VISION_S3_GATE_ENFORCE=1"
        );
    }
    eprintln!(
        "LATTICE_VISION_S6_SERVE_SKIPPED reason=no_checkpoint \
         tried={MODEL_DIR_ENV} and ~/.lattice/models/qwen3.5-0.8b"
    );
    None
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_loopback_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("ephemeral port must bind")
        .local_addr()
        .expect("bound listener must have an address")
        .port()
}

fn wait_for_health(port: u16, deadline: Instant) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    while Instant::now() < deadline {
        if let Ok(response) = ureq::get(&url).call()
            && response.status() == 200
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

fn post_chat_completion(port: u16, body: &serde_json::Value) -> serde_json::Value {
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let response = ureq::post(&url)
        .set("content-type", "application/json")
        .send_bytes(&serde_json::to_vec(body).expect("request body must serialize"));
    let response = match response {
        Ok(response) => response,
        Err(ureq::Error::Status(code, response)) => {
            let body = response.into_string().unwrap_or_default();
            panic!("chat completion returned HTTP {code}; body: {body}");
        }
        Err(err) => panic!("chat completion request failed: {err}"),
    };
    assert_eq!(response.status(), 200);
    serde_json::from_str(
        &response
            .into_string()
            .expect("response body must be readable UTF-8"),
    )
    .expect("response body must be JSON")
}

fn marker_count(line: &str, field: &str) -> Option<usize> {
    line.split_ascii_whitespace()
        .find_map(|part| part.strip_prefix(field)?.parse().ok())
}

#[test]
fn vision_dispatch_marker_counts_are_machine_readable() {
    let marker = "[metal-worker] route=vision dispatch=multimodal \
                  metal_gemm_dispatches=337 metal_gemm_calls=337";
    assert_eq!(marker_count(marker, METAL_DISPATCH_FIELD), Some(337));
    assert_eq!(marker_count(marker, GEMM_CALL_FIELD), Some(337));
    assert_eq!(marker_count(marker, "missing="), None);
}

#[test]
fn serve_chat_completions_reaches_vision_forward_path() {
    let Some(model_dir) = require_model_dir() else {
        return;
    };
    let image_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("vision")
        .join("golden_image.png");
    let image = std::fs::read(&image_path)
        .unwrap_or_else(|err| panic!("reading {}: {err}", image_path.display()));
    let image_data_uri = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(image)
    );
    let port = free_loopback_port();
    let _gpu_guard = gpu_test_lock();
    let mut child = ChildGuard(
        Command::new(env!("CARGO_BIN_EXE_lattice_serve"))
            .arg("--model")
            .arg(&model_dir)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("lattice_serve must spawn"),
    );
    let stderr_pipe = child
        .0
        .stderr
        .take()
        .expect("child stderr must be captured");
    let stderr = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let stderr_reader = {
        let stderr = std::sync::Arc::clone(&stderr);
        std::thread::spawn(move || {
            use std::io::BufRead as _;

            let mut reader = std::io::BufReader::new(stderr_pipe);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => stderr
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push_str(&line),
                }
            }
        })
    };

    if !wait_for_health(port, Instant::now() + Duration::from_secs(120)) {
        let output = stderr
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        panic!("lattice_serve did not become healthy; stderr:\n{output}");
    }

    let image_response = post_chat_completion(
        port,
        &serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
            }],
            "max_tokens": 8,
            "temperature": 0.0
        }),
    );
    assert!(
        image_response["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|answer| !answer.trim().is_empty()),
        "vision response must contain generated text"
    );
    let text_response = post_chat_completion(
        port,
        &serde_json::json!({
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 4,
            "temperature": 0.0
        }),
    );
    assert!(
        text_response["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|answer| !answer.trim().is_empty()),
        "text control response must contain generated text"
    );

    std::thread::sleep(Duration::from_millis(200));
    let output = stderr
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let vision_markers: Vec<_> = output
        .lines()
        .filter(|line| line.contains(VISION_DISPATCH_MARKER))
        .collect();
    assert_eq!(
        vision_markers.len(),
        1,
        "vision marker must appear only for the image request; stderr:\n{output}"
    );
    let marker = vision_markers[0];
    let metal_dispatches = marker_count(marker, METAL_DISPATCH_FIELD)
        .unwrap_or_else(|| panic!("vision marker omitted {METAL_DISPATCH_FIELD}: {marker}"));
    let gemm_calls = marker_count(marker, GEMM_CALL_FIELD)
        .unwrap_or_else(|| panic!("vision marker omitted {GEMM_CALL_FIELD}: {marker}"));
    assert!(
        metal_dispatches > 0,
        "vision encoder silently fell back to CPU for every GEMM; marker: {marker}"
    );
    assert_eq!(
        metal_dispatches, gemm_calls,
        "every production vision GEMM must dispatch to Metal; marker: {marker}"
    );

    drop(child);
    let _ = stderr_reader.join();
}
