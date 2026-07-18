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
//! (`/tmp/lion-metal-gpu-test.lock`) the in-crate `gpu_test_lock()` uses, via the shared
//! `test-utils` seam (`lattice_inference::serve::gpu_test_lock`, PR #1021 review round 6
//! issue 6) -- this test drives Metal in a *child process*, so the in-crate lock (private to
//! `metal_qwen35.rs`'s own `#[cfg(test)]` module) cannot cover it, and previously reimplemented
//! the same lock path/timeout/polling protocol independently instead of sharing it.
//!
//! Dispatch proof (PR #1021 review round 6, issue 3): a passing HTTP 200 with non-empty text
//! does not by itself prove the request took the vision path -- a regression that silently
//! drops the image and falls back to a plausible-looking text-only answer would still pass
//! that check. This test instead captures the server's stderr and asserts the vision-dispatch
//! marker line (`route=vision dispatch=multimodal`, emitted by `serve::metal_worker` at the
//! moment a request routes to the vision-aware decode path) appears for the image request and
//! does NOT appear for a text-only control request sent to the same running server.
//!
//! Run:
//! ```bash
//! cargo test --release -p lattice-inference --features f16,metal-gpu,test-utils \
//!     --test vision_serve_e2e_test -- --nocapture
//! ```

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

/// The structured, single-line marker `serve::metal_worker` emits at the moment a request
/// routes to the vision-aware decode path (PR #1021 review round 6, issue 3). Kept as one
/// named constant so the test and the production emit site can't drift on wording without a
/// compile-time-visible diff -- this test asserts on the fixed prefix, not the whole line.
const VISION_DISPATCH_MARKER: &str = "route=vision dispatch=multimodal";

fn post_chat_completion(port: u16, body: &serde_json::Value) -> serde_json::Value {
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let response = ureq::post(&url)
        .set("content-type", "application/json")
        .send_bytes(&serde_json::to_vec(body).expect("request body serializes"));
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
    serde_json::from_str(&body_text).expect("response body must be valid JSON")
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
    let image_b64 = lattice_inference::serve::base64_codec::encode_standard(&image_bytes);

    let bin = env!("CARGO_BIN_EXE_lattice_serve");
    let port = free_loopback_port();

    // Held for the whole subprocess lifetime -- from spawn (model load
    // drives the GPU) through the request (decode also drives the GPU) to
    // teardown -- not just around the HTTP call.
    let _gpu_guard = lattice_inference::serve::gpu_test_lock::gpu_test_lock();

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

    // Captured continuously from spawn onward (not just on a health-check failure) so the
    // vision-dispatch marker assertions below see the server's full stderr across BOTH
    // requests sent later in this test.
    let stderr_pipe = child
        .0
        .stderr
        .take()
        .expect("child spawned with Stdio::piped() stderr");
    let stderr_buf = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let stderr_reader = {
        let buf = stderr_buf.clone();
        std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stderr_pipe);
            let mut line = String::new();
            loop {
                use std::io::BufRead;
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => buf
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push_str(&line),
                }
            }
        })
    };

    let ready_deadline = Instant::now() + Duration::from_secs(120);
    if !wait_for_health(port, ready_deadline) {
        let stderr = stderr_buf
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        panic!(
            "lattice_serve did not become healthy within 120s on port {port}; stderr:\n{stderr}"
        );
    }

    let question = "Describe this image.";
    // `model` is deliberately omitted: `req.model` is optional server-side and
    // is checked against `s.model_id` (derived from the model directory name)
    // when present, so hardcoding a guessed id here would be a test-only
    // assumption about checkpoint naming, not something this stage governs.
    let image_body = serde_json::json!({
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
    let value = post_chat_completion(port, &image_body);
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

    // Text-only control request, same running server: proves the marker is specific to
    // image-bearing requests, not emitted on every request regardless of content.
    let text_only_body = serde_json::json!({
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 8,
        "temperature": 0.0,
    });
    let text_only_value = post_chat_completion(port, &text_only_body);
    assert!(
        text_only_value["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty()),
        "text-only control request must also get a non-empty answer"
    );

    // Give the server a moment to flush its stderr for both requests before inspecting it.
    std::thread::sleep(Duration::from_millis(200));
    let stderr_snapshot = stderr_buf
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let marker_count = stderr_snapshot.matches(VISION_DISPATCH_MARKER).count();
    assert_eq!(
        marker_count, 1,
        "expected the vision-dispatch marker exactly once (the image request only, not the \
         text-only control request); stderr:\n{stderr_snapshot}"
    );

    // `ChildGuard::drop` kills and reaps the subprocess; join the reader thread after so it
    // observes EOF instead of leaking (best-effort -- the process is already being torn down).
    drop(child);
    let _ = stderr_reader.join();
}
