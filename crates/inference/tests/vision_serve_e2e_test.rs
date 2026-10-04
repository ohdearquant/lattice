//! ADR-069 S6 gate: post an inline image to the real `lattice_serve`
//! process and prove that the shared HTTP contract reaches the production
//! vision decode route.
//!
//! Model-gated: `LATTICE_VISION_S3_MODEL_DIR` is authoritative when set; the
//! default `~/.lattice/models/qwen3.5-0.8b` is consulted only when it is unset.
//! A missing checkpoint emits a loud skip;
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

const MODEL_DIR_ENV: &str = "LATTICE_VISION_S3_MODEL_DIR";

#[derive(Debug, PartialEq)]
enum ModelDirResolution {
    Use(PathBuf),
    Skip(String),
}

/// The explicit variable is authoritative: when it is set and does not exist the
/// test skips (or panics under enforcement) instead of using the default checkpoint.
/// The default directory is consulted only when the variable is unset.
fn resolve_model_dir(
    var: Option<PathBuf>,
    var_exists: bool,
    default: Option<PathBuf>,
    default_exists: bool,
    enforce: bool,
) -> ModelDirResolution {
    if let Some(path) = var {
        if var_exists {
            return ModelDirResolution::Use(path);
        }
        if enforce {
            panic!(
                "{MODEL_DIR_ENV}={} does not exist while \
                 LATTICE_VISION_S3_GATE_ENFORCE=1",
                path.display()
            );
        }
        return ModelDirResolution::Skip(format!(
            "LATTICE_VISION_S6_SERVE_SKIPPED reason=no_checkpoint \
             tried={MODEL_DIR_ENV}={}",
            path.display()
        ));
    }
    if let Some(path) = default
        && default_exists
    {
        return ModelDirResolution::Use(path);
    }
    if enforce {
        panic!(
            "no vision checkpoint found via {MODEL_DIR_ENV} or \
             ~/.lattice/models/qwen3.5-0.8b while \
             LATTICE_VISION_S3_GATE_ENFORCE=1"
        );
    }
    ModelDirResolution::Skip(format!(
        "LATTICE_VISION_S6_SERVE_SKIPPED reason=no_checkpoint \
         tried={MODEL_DIR_ENV} and ~/.lattice/models/qwen3.5-0.8b"
    ))
}

fn require_model_dir() -> Option<PathBuf> {
    let var = std::env::var(MODEL_DIR_ENV)
        .ok()
        .map(|value| PathBuf::from(expand_home(&value)));
    let var_exists = var.as_ref().is_some_and(|path| path.exists());
    let default = default_model_dir();
    let default_exists = default.as_ref().is_some_and(|path| path.exists());
    match resolve_model_dir(var, var_exists, default, default_exists, enforce()) {
        ModelDirResolution::Use(path) => Some(path),
        ModelDirResolution::Skip(line) => {
            eprintln!("{line}");
            None
        }
    }
}

#[test]
fn model_dir_resolution_set_but_missing_skips_even_when_default_exists() {
    let resolved = resolve_model_dir(
        Some(PathBuf::from("/nonexistent/ckpt")),
        false,
        Some(PathBuf::from("/default/models/qwen3.5-0.8b")),
        true,
        false,
    );
    match resolved {
        ModelDirResolution::Skip(line) => {
            assert!(line.contains("LATTICE_VISION_S6_SERVE_SKIPPED"), "{line}");
            assert!(line.contains("/nonexistent/ckpt"), "{line}");
        }
        other => panic!("a set-but-missing variable must skip, got {other:?}"),
    }
}

#[test]
fn model_dir_resolution_set_and_present_uses_that_path() {
    let resolved = resolve_model_dir(
        Some(PathBuf::from("/data/ckpt")),
        true,
        Some(PathBuf::from("/default/models/qwen3.5-0.8b")),
        true,
        false,
    );
    assert_eq!(
        resolved,
        ModelDirResolution::Use(PathBuf::from("/data/ckpt"))
    );
}

#[test]
fn model_dir_resolution_unset_with_default_present_uses_default() {
    let default = PathBuf::from("/default/models/qwen3.5-0.8b");
    let resolved = resolve_model_dir(None, false, Some(default.clone()), true, false);
    assert_eq!(resolved, ModelDirResolution::Use(default));
}

#[test]
fn model_dir_resolution_unset_with_default_missing_skips() {
    let resolved = resolve_model_dir(
        None,
        false,
        Some(PathBuf::from("/default/models/qwen3.5-0.8b")),
        false,
        false,
    );
    match resolved {
        ModelDirResolution::Skip(line) => {
            assert!(line.contains("LATTICE_VISION_S6_SERVE_SKIPPED"), "{line}");
        }
        other => panic!("an unset variable with no default must skip, got {other:?}"),
    }
}

#[test]
#[should_panic(expected = "does not exist while LATTICE_VISION_S3_GATE_ENFORCE=1")]
fn model_dir_resolution_set_but_missing_panics_under_enforce() {
    let _ = resolve_model_dir(
        Some(PathBuf::from("/nonexistent/ckpt")),
        false,
        Some(PathBuf::from("/default/models/qwen3.5-0.8b")),
        true,
        true,
    );
}

#[test]
#[should_panic(expected = "no vision checkpoint found")]
fn model_dir_resolution_unset_with_default_missing_panics_under_enforce() {
    let _ = resolve_model_dir(
        None,
        false,
        Some(PathBuf::from("/default/models/qwen3.5-0.8b")),
        false,
        true,
    );
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

/// How long to wait for `lattice_serve` to report healthy before treating the
/// process as stuck. Defaults to 120s (loading a multimodal checkpoint on a
/// local Apple-silicon machine fits comfortably inside that); overridable via
/// `LATTICE_VISION_SERVE_HEALTH_TIMEOUT_SECS` because a hosted CI runner can be
/// considerably slower for the same work, and a fixed deadline with no escape
/// hatch turns a slow-but-healthy start into an indistinguishable panic.
fn health_wait_timeout() -> Duration {
    Duration::from_secs(
        std::env::var("LATTICE_VISION_SERVE_HEALTH_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(120),
    )
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

    let health_timeout = health_wait_timeout();
    let health_wait_started = Instant::now();
    if !wait_for_health(port, health_wait_started + health_timeout) {
        let output = stderr
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        panic!(
            "lattice_serve did not become healthy after {:?} \
             (override with LATTICE_VISION_SERVE_HEALTH_TIMEOUT_SECS); stderr:\n{output}",
            health_wait_started.elapsed()
        );
    }
    eprintln!(
        "lattice_serve healthy after {:?} (budget {:?})",
        health_wait_started.elapsed(),
        health_timeout
    );

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
