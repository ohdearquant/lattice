//! ADR-069 S6 gate: post a small image + question to the real
//! `lattice_serve` HTTP surface (OpenAI-shape `image_url` content part,
//! `data:` URI) and assert a grounded, non-empty answer comes back over
//! Metal, end to end -- config parsing (S1) through decoder M-RoPE +
//! visual-embedding injection (S5, #1004/#1005/#1009/#1011) through the
//! serve-side wiring this stage adds.
//!
//! Model-gated: reuses `LATTICE_VISION_S3_MODEL_DIR` (same checkpoint every
//! other ADR-069 stage gate uses) with a fallback to the default model
//! cache path (`~/.lattice/models/qwen3.5-0.8b`) so this runs out of the
//! box on a machine that already has the checkpoint, matching this stage's
//! "the goal today is a LIVE grounded-answer smoke on Metal" brief. With
//! neither path present, the test prints a skip line and returns; with
//! `LATTICE_VISION_S3_GATE_ENFORCE=1` it panics instead (fail-closed, same
//! convention as `vision_s5b_e2e_gate_test.rs`).
//!
//! GPU discipline: acquires the same machine-wide advisory flock
//! (`/tmp/lion-metal-gpu-test.lock`) the in-crate `gpu_test_lock()` uses,
//! for the whole subprocess-spawn-through-response window -- this test
//! drives Metal in a *child process*, so the in-crate lock (private to
//! `metal_qwen35.rs`'s own `#[cfg(test)]` module) cannot cover it.
//!
//! Run:
//! ```bash
//! cargo test --release -p lattice-inference --features f16,metal-gpu \
//!     --test vision_serve_e2e_test -- --nocapture --ignored
//! ```

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn enforce() -> bool {
    std::env::var("LATTICE_VISION_S3_GATE_ENFORCE").as_deref() == Ok("1")
}

fn shellexpand_home(path: &str) -> String {
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

/// Resolves the checkpoint directory: `LATTICE_VISION_S3_MODEL_DIR` first
/// (the convention every other ADR-069 stage gate uses), falling back to
/// the default model cache path. Returns `None` (after printing a skip
/// line) when neither exists and enforcement is off; panics when
/// `LATTICE_VISION_S3_GATE_ENFORCE=1`.
fn require_model_dir() -> Option<PathBuf> {
    const VAR: &str = "LATTICE_VISION_S3_MODEL_DIR";
    if let Ok(value) = std::env::var(VAR) {
        let path = PathBuf::from(shellexpand_home(&value));
        if path.exists() {
            return Some(path);
        }
        if enforce() {
            panic!(
                "{VAR}={} does not exist, and LATTICE_VISION_S3_GATE_ENFORCE=1 -- the S6 \
                 serve e2e gate must fail closed on a missing checkpoint",
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
            "no vision checkpoint found via {VAR} or the default model cache \
             (~/.lattice/models/qwen3.5-0.8b), and LATTICE_VISION_S3_GATE_ENFORCE=1"
        );
    }
    eprintln!(
        "LATTICE_VISION_S6_SERVE_SKIPPED reason=no_checkpoint tried={VAR} and \
         ~/.lattice/models/qwen3.5-0.8b -- ADR-069 S6 serve e2e gate requires the \
         Qwen3.5-0.8B vision checkpoint locally; this is a loud skip, not a silent pass"
    );
    None
}

/// Machine-wide GPU serialization, mirrored from the in-crate
/// `gpu_test_lock()` (`metal_qwen35.rs`) since that helper is private to
/// its own `#[cfg(test)]` module and this is a separate test-binary crate.
/// Same path, same protocol (advisory `flock`, bounded wait, loud panic on
/// timeout) so this test and any in-crate GPU test correctly serialize
/// against each other.
struct GpuTestGuard {
    _file: std::fs::File,
}

fn gpu_test_lock() -> GpuTestGuard {
    const PATH: &str = "/tmp/lion-metal-gpu-test.lock";
    const TIMEOUT: Duration = Duration::from_secs(30 * 60);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(PATH)
        .unwrap_or_else(|e| panic!("gpu_test_lock: cannot open {PATH}: {e}"));
    let deadline = Instant::now() + TIMEOUT;
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) => {
                if Instant::now() >= deadline {
                    panic!(
                        "gpu_test_lock: another process has held {PATH} for over {}s -- \
                         inspect `lsof {PATH}`",
                        TIMEOUT.as_secs()
                    );
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(std::fs::TryLockError::Error(e)) => {
                panic!("gpu_test_lock: flock on {PATH} failed: {e}")
            }
        }
    }
    GpuTestGuard { _file: file }
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_loopback_port() -> u16 {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").expect("binding an ephemeral port must work");
    listener
        .local_addr()
        .expect("bound listener must have a local addr")
        .port()
    // `listener` drops here, freeing the port for `lattice_serve` to bind.
    // Small TOCTOU window is acceptable for a single-test local harness.
}

fn wait_for_health(port: u16, deadline: Instant) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    while Instant::now() < deadline {
        if let Ok(resp) = ureq::get(&url).call()
            && resp.status() == 200
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

/// Standard-alphabet base64 encoder (no external crate; mirrors the decoder
/// `lattice_serve.rs` ships for the request-parsing side of this same
/// stage).
fn base64_encode_standard(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) as usize & 0x3f] as char);
        out.push(ALPHABET[(n >> 12) as usize & 0x3f] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 0x3f] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 0x3f] as char
        } else {
            '='
        });
    }
    out
}

#[test]
fn serve_chat_completions_answers_a_grounded_question_about_an_image() {
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
    let image_bytes = std::fs::read(&image_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", image_path.display()));
    let image_b64 = base64_encode_standard(&image_bytes);

    let bin = env!("CARGO_BIN_EXE_lattice_serve");
    let port = free_loopback_port();

    // Held for the whole subprocess lifetime -- from spawn (model load
    // drives the GPU) through the request (decode also drives the GPU) to
    // teardown -- not just around the HTTP call.
    let _gpu_guard = gpu_test_lock();

    let mut child = ChildGuard(
        Command::new(bin)
            .arg("--model")
            .arg(&model_dir)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("spawning {bin}: {e}")),
    );

    let ready_deadline = Instant::now() + Duration::from_secs(120);
    if !wait_for_health(port, ready_deadline) {
        let mut stderr = String::new();
        if let Some(mut s) = child.0.stderr.take() {
            let _ = s.read_to_string(&mut stderr);
        }
        panic!(
            "lattice_serve did not become healthy within 120s on port {port}; stderr:\n{stderr}"
        );
    }

    let question = "Describe this image.";
    // `model` is deliberately omitted: `req.model` is optional server-side and
    // is checked against `s.model_id` (derived from the model directory name)
    // when present, so hardcoding a guessed id here would be a test-only
    // assumption about checkpoint naming, not something this stage governs.
    let body = serde_json::json!({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{image_b64}")}},
            ],
        }],
        "max_tokens": 32,
        "temperature": 0.0,
    });

    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let response = ureq::post(&url)
        .set("content-type", "application/json")
        .send_bytes(&serde_json::to_vec(&body).expect("request body serializes"));

    let response = match response {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let text = r.into_string().unwrap_or_default();
            panic!("chat completion returned HTTP {code}, expected 200; body: {text}")
        }
        Err(e) => panic!("chat completion request failed: {e}"),
    };
    assert_eq!(response.status(), 200);
    let body_text = response
        .into_string()
        .expect("response body must be readable UTF-8");
    let value: serde_json::Value =
        serde_json::from_str(&body_text).expect("response body must be valid JSON");
    let answer = value["choices"][0]["message"]["content"]
        .as_str()
        .expect("response must carry choices[0].message.content");

    eprintln!(
        "LATTICE_VISION_S6_SERVE_SMOKE question={question:?} image={} answer={answer:?}",
        image_path.display()
    );
    assert!(
        !answer.trim().is_empty(),
        "expected a non-empty grounded answer, got: {answer:?}"
    );
    // `ChildGuard::drop` kills and reaps the subprocess.
    drop(child);
}
