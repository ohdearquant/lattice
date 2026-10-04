//! Checkpoint-gated test for LoRA adapter application through the real
//! `lattice_serve` process: `/v1/lora/load`, per-request adapter selection on
//! `/v1/chat/completions`, and `/v1/lora/unload`, all reaching the Metal serving
//! runtime (`QwenMetalRuntime::generate` and `::control`) on a real model.
//!
//! The adapter is synthesized into a temp dir from a fixed seed, so no adapter
//! file has to exist on the machine. Model-gated: `LATTICE_SERVE_LORA_MODEL_DIR`
//! is authoritative when set (a missing path skips, it never falls back), and
//! `~/.lattice/models/qwen3.5-0.8b` is used only when the variable is unset. A
//! missing checkpoint emits a loud skip line; `LATTICE_SERVE_LORA_GATE_ENFORCE=1`
//! turns the same condition into a failure.
//!
//! ```bash
//! LATTICE_SERVE_LORA_GATE_ENFORCE=1 cargo test --release \
//!   -p lattice-inference --features download,f16,metal-gpu,serve \
//!   --test serve_lora_e2e_test -- --nocapture
//! ```
//!
//! Observable surface. `lattice_serve` answers chat completions with text,
//! `finish_reason` and token counts; it does not return token ids or logprobs
//! (`logprobs` is rejected on this server). Comparisons below are therefore on
//! the full assistant message, the finish reason and the completion token count,
//! requested with greedy decoding (`temperature: 0`). Greedy decoding is not taken
//! to be deterministic on the backend: repeatability is observed for the exact
//! request sequence below, not assumed. The base request is compared with a repeat
//! of itself first (step 1); the adapted and reloaded requests are repeated after
//! they have been compared with the base output.
//!
//! Resident adapters are never applied implicitly: this server has no router, so
//! a request that omits `lora` selects the base model whether or not an adapter
//! is resident. The unload step therefore cannot be observed through base-request
//! output alone: a base request clears an applied adapter by itself whenever the
//! applied selection differs from the empty one (the registry does nothing when the
//! requested selection already equals the applied one). The unload is asserted
//! through the published residency snapshot (`GET /v1/lora`) and through the
//! refusal of a request that names the removed id.

#![cfg(all(target_os = "macos", feature = "metal-gpu"))]

use lattice_inference::lora_file::load_lora_safetensors;
use lattice_inference::lora_hook::qwen35_projection_shape;
use lattice_inference::measurement::gpu_test_lock;
use lattice_inference::model::qwen35_config::Qwen35Config;
use serde_json::{Value, json};
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

const MODEL_DIR_ENV: &str = "LATTICE_SERVE_LORA_MODEL_DIR";
const ENFORCE_ENV: &str = "LATTICE_SERVE_LORA_GATE_ENFORCE";
const HEALTH_TIMEOUT_ENV: &str = "LATTICE_SERVE_LORA_HEALTH_TIMEOUT_SECS";

/// Fixed prompt and budget: long enough for a perturbed model to diverge from the
/// base, short enough that every request is cheap on a 0.8B checkpoint.
const PROMPT: &str = "List three primary colors and one fact about each.";
const MAX_TOKENS: usize = 24;

/// Adapter shape. Rank 4 with `alpha = 8` gives an effective scale of `alpha / rank = 2`.
/// `gate_proj`/`up_proj`/`down_proj` exist on every layer of both layer kinds
/// (linear-attention and full-attention), so one module set is valid for all layers
/// without per-layer-kind branching, and the Metal path can apply all three.
const ADAPTER_RANK: usize = 4;
const ADAPTER_ALPHA: f32 = 8.0;
const ADAPTER_MODULES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
/// `A` entries are uniform in `[-1, 1) / sqrt(d_in)`, which keeps `A x` at the same
/// order as the activation's RMS. `B` entries are uniform in `[-0.5, 0.5)`.
/// With the effective scale of 2 the added term is of the same order as the base
/// projection output in every layer, which is large enough to move greedy decoding
/// off the base trajectory within 24 tokens while staying finite (no entry is large
/// enough to push an f16 activation toward overflow).
const ADAPTER_B_MAGNITUDE: f32 = 0.5;
const ADAPTER_SEED: u64 = 0x1790_0A11_CE5E_ED01;
/// A second adapter with the same shapes and a different weight stream, loaded after
/// the first is unloaded so that reusing the first adapter's weights is observable.
const RELOAD_ADAPTER_SEED: u64 = 0x1790_0B22_DF6F_FE02;
const ADAPTER_NAME: &str = "synthetic-mlp-adapter";

fn enforce() -> bool {
    std::env::var(ENFORCE_ENV).as_deref() == Ok("1")
}

/// A leading `~/` is expanded when `home` is given. The prefix is recognised on the raw
/// bytes, so a non-UTF-8 suffix is still expanded; any other value is used as given.
fn expand_home(value: &OsStr, home: Option<&OsStr>) -> PathBuf {
    if let Some(rest) = value.as_bytes().strip_prefix(b"~/")
        && let Some(home) = home
    {
        let mut expanded = home.to_os_string();
        expanded.push("/");
        expanded.push(OsStr::from_bytes(rest));
        return PathBuf::from(expanded);
    }
    PathBuf::from(value)
}

fn default_model_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".lattice")
            .join("models")
            .join("qwen3.5-0.8b"),
    )
}

/// The explicit variable is authoritative: when it is set and does not exist the
/// test skips (or fails under enforcement) instead of silently using the default
/// checkpoint, so pointing it at a missing path is a reliable way to disable the run.
fn require_model_dir() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os(MODEL_DIR_ENV) {
        let path = expand_home(&value, std::env::var_os("HOME").as_deref());
        if path.exists() {
            return Some(path);
        }
        if enforce() {
            panic!(
                "{MODEL_DIR_ENV}={} does not exist while {ENFORCE_ENV}=1",
                path.display()
            );
        }
        eprintln!(
            "LATTICE_SERVE_LORA_SKIPPED reason=no_checkpoint tried={MODEL_DIR_ENV}={}",
            path.display()
        );
        return None;
    }
    if let Some(path) = default_model_dir()
        && path.exists()
    {
        return Some(path);
    }
    if enforce() {
        panic!(
            "no checkpoint found via {MODEL_DIR_ENV} or ~/.lattice/models/qwen3.5-0.8b \
             while {ENFORCE_ENV}=1"
        );
    }
    eprintln!(
        "LATTICE_SERVE_LORA_SKIPPED reason=no_checkpoint \
         tried={MODEL_DIR_ENV} and ~/.lattice/models/qwen3.5-0.8b"
    );
    None
}

// ---------------------------------------------------------------------------
// Deterministic adapter synthesis
// ---------------------------------------------------------------------------

/// SplitMix64: a fixed, dependency-free generator whose stream is fully determined
/// by the seed.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`, built from 24 bits so the f32 conversion is exact.
    fn next_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as f32;
        bits / 16_777_216.0 * 2.0 - 1.0
    }
}

struct SynthTensor {
    name: String,
    shape: [usize; 2],
    values: Vec<f32>,
}

fn synth_tensors(cfg: &Qwen35Config, seed: u64) -> Vec<SynthTensor> {
    let mut rng = SplitMix64(seed);
    let mut tensors = Vec::new();
    for layer in 0..cfg.num_hidden_layers {
        for module in ADAPTER_MODULES {
            let shape = qwen35_projection_shape(cfg, layer, module)
                .unwrap_or_else(|err| panic!("layer {layer} module {module}: {err}"));
            let a_scale = 1.0 / (shape.d_in as f32).sqrt();
            let a = (0..ADAPTER_RANK * shape.d_in)
                .map(|_| rng.next_unit() * a_scale)
                .collect();
            let b = (0..shape.d_out * ADAPTER_RANK)
                .map(|_| rng.next_unit() * ADAPTER_B_MAGNITUDE)
                .collect();
            let prefix = format!("base_model.model.model.layers.{layer}.mlp.{module}");
            tensors.push(SynthTensor {
                name: format!("{prefix}.lora_A.weight"),
                shape: [ADAPTER_RANK, shape.d_in],
                values: a,
            });
            tensors.push(SynthTensor {
                name: format!("{prefix}.lora_B.weight"),
                shape: [shape.d_out, ADAPTER_RANK],
                values: b,
            });
        }
    }
    tensors
}

/// Serialize a PEFT-layout safetensors file (`lora_A` is `[rank, d_in]`, `lora_B` is
/// `[d_out, rank]`, F32) with `alpha` in `__metadata__`.
fn adapter_bytes(cfg: &Qwen35Config, seed: u64) -> Vec<u8> {
    let tensors = synth_tensors(cfg, seed);
    let mut header = format!("{{\"__metadata__\":{{\"alpha\":\"{ADAPTER_ALPHA}\"}}");
    let mut data: Vec<u8> = Vec::new();
    for tensor in &tensors {
        let start = data.len();
        for value in &tensor.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.push_str(&format!(
            ",\"{}\":{{\"dtype\":\"F32\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
            tensor.name,
            tensor.shape[0],
            tensor.shape[1],
            start,
            data.len()
        ));
    }
    header.push('}');
    // Keep the tensor payload 8-byte aligned, as safetensors writers do.
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
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .unwrap_or_else(|err| panic!("creating {}: {err}", path.display()));
    file.write_all(&adapter_bytes(cfg, seed))
        .unwrap_or_else(|err| panic!("writing {}: {err}", path.display()));
}

// ---------------------------------------------------------------------------
// Server process and HTTP helpers
// ---------------------------------------------------------------------------

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

/// A hosted runner can load the same checkpoint far more slowly than a local
/// machine; the budget is overridable so a slow-but-healthy start is not a panic.
fn health_wait_timeout() -> Duration {
    Duration::from_secs(
        std::env::var(HEALTH_TIMEOUT_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(120),
    )
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
        // The child belongs to `Server` from the moment it exists, so a panic anywhere
        // below (stderr capture, the drainer thread, the health wait) kills and waits it.
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
        {
            // Draining stderr is also what keeps the child from blocking on a full pipe.
            let stderr = Arc::clone(&server.stderr);
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
                            .unwrap_or_else(PoisonError::into_inner)
                            .push_str(&line),
                    }
                }
            });
        }
        let budget = health_wait_timeout();
        let started = Instant::now();
        let url = format!("http://127.0.0.1:{port}/health");
        loop {
            if let Ok(response) = ureq::get(&url).call()
                && response.status() == 200
            {
                break;
            }
            assert!(
                started.elapsed() < budget,
                "lattice_serve did not become healthy after {:?} (override with \
                 {HEALTH_TIMEOUT_ENV}); stderr:\n{}",
                started.elapsed(),
                server.diagnostics()
            );
            std::thread::sleep(Duration::from_millis(200));
        }
        eprintln!(
            "lattice_serve healthy after {:?} (budget {budget:?})",
            started.elapsed()
        );
        server
    }

    fn diagnostics(&self) -> String {
        self.stderr
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    fn request(&self, method: &str, path: &str, body: Option<&Value>) -> (u16, String) {
        let request = ureq::request(method, &self.url(path));
        let outcome = match body {
            Some(body) => request
                .set("content-type", "application/json")
                .send_bytes(&serde_json::to_vec(body).expect("request body must serialize")),
            None => request.call(),
        };
        match outcome {
            Ok(response) => {
                let status = response.status();
                (status, response.into_string().unwrap_or_default())
            }
            Err(ureq::Error::Status(status, response)) => {
                (status, response.into_string().unwrap_or_default())
            }
            Err(err) => panic!(
                "{method} {path} failed at the transport level: {err}; stderr:\n{}",
                self.diagnostics()
            ),
        }
    }

    /// A 200 JSON answer; anything else fails with the body and the server's stderr.
    fn ok(&self, method: &str, path: &str, body: Option<&Value>) -> Value {
        let (status, text) = self.request(method, path, body);
        assert_eq!(
            status,
            200,
            "{method} {path} returned HTTP {status}: {text}; stderr:\n{}",
            self.diagnostics()
        );
        serde_json::from_str(&text).unwrap_or_else(|err| panic!("{path} body is not JSON: {err}"))
    }

    fn residency(&self) -> Value {
        self.ok("GET", "/v1/lora", None)
    }

    fn load_adapter(&self, path: &Path) -> (u32, Value) {
        let body = self.ok(
            "POST",
            "/v1/lora/load",
            Some(&json!({"path": path.to_str().expect("utf-8 temp path"), "name": ADAPTER_NAME})),
        );
        assert_eq!(body["status"], "loaded", "load body: {body}");
        let id = body["id"].as_u64().expect("load body carries an id");
        (u32::try_from(id).expect("adapter id fits u32"), body)
    }

    fn unload_adapter(&self, id: u32) -> Value {
        let body = self.ok("POST", "/v1/lora/unload", Some(&json!({"id": id})));
        assert_eq!(body["status"], "unloaded", "unload body: {body}");
        assert_eq!(body["id"], id, "unload body: {body}");
        body
    }

    /// Greedy chat completion, optionally naming one resident adapter and the request
    /// scale applied on top of the adapter's own `alpha / rank` scale.
    fn chat_raw(&self, adapter: Option<(u32, f32)>) -> (u16, String) {
        let mut body = json!({
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0
        });
        if let Some((id, scale)) = adapter {
            body["lora"] = json!([{"id": id, "scale": scale}]);
        }
        self.request("POST", "/v1/chat/completions", Some(&body))
    }

    fn chat(&self, adapter: Option<(u32, f32)>) -> Output {
        let (status, text) = self.chat_raw(adapter);
        assert_eq!(
            status,
            200,
            "chat completion (adapter {adapter:?}) returned HTTP {status}: {text}; stderr:\n{}",
            self.diagnostics()
        );
        let body: Value = serde_json::from_str(&text).expect("chat body is JSON");
        Output {
            message: body["choices"][0]["message"].to_string(),
            finish_reason: body["choices"][0]["finish_reason"].to_string(),
            completion_tokens: body["usage"]["completion_tokens"]
                .as_u64()
                .expect("usage.completion_tokens must be present"),
        }
    }
}

/// What the chat response exposes and the test compares: the message, the finish
/// reason and the completion token count.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Output {
    message: String,
    finish_reason: String,
    completion_tokens: u64,
}

fn assert_residency(server: &Server, resident: &[u32], applied: &[u32], context: &str) {
    let body = server.residency();
    let resident_ids: Vec<u64> = body["adapters"]
        .as_array()
        .unwrap_or_else(|| panic!("{context}: `adapters` must be an array: {body}"))
        .iter()
        .map(|adapter| adapter["id"].as_u64().expect("adapter id"))
        .collect();
    let applied_ids: Vec<u64> = body["applied"]
        .as_array()
        .unwrap_or_else(|| panic!("{context}: `applied` must be an array: {body}"))
        .iter()
        .map(|entry| entry["id"].as_u64().expect("applied id"))
        .collect();
    let want_resident: Vec<u64> = resident.iter().map(|&id| u64::from(id)).collect();
    let want_applied: Vec<u64> = applied.iter().map(|&id| u64::from(id)).collect();
    assert_eq!(
        resident_ids, want_resident,
        "{context}: resident ids; {body}"
    );
    assert_eq!(applied_ids, want_applied, "{context}: applied ids; {body}");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Needs no checkpoint and no GPU: proves the synthesized file is byte-for-byte
/// reproducible and is accepted by the same parser `/v1/lora/load` uses, so a red
/// end-to-end run is about the serving path and not the fixture.
#[test]
fn synthesized_adapter_is_deterministic_and_accepted_by_the_load_parser() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let first = adapter_bytes(&cfg, ADAPTER_SEED);
    assert_eq!(
        first,
        adapter_bytes(&cfg, ADAPTER_SEED),
        "same seed, same bytes"
    );
    assert_ne!(
        first,
        adapter_bytes(&cfg, ADAPTER_SEED ^ 1),
        "a different seed must change the bytes"
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("adapter.safetensors");
    write_adapter(&path, &cfg, ADAPTER_SEED);
    assert_eq!(std::fs::read(&path).expect("read back"), first);

    let (layers, descriptor) = load_lora_safetensors(&path).expect("loader must accept the file");
    assert_eq!(layers.len(), cfg.num_hidden_layers * ADAPTER_MODULES.len());
    assert_eq!(descriptor.rank, ADAPTER_RANK);
    assert_eq!(descriptor.alpha, ADAPTER_ALPHA);
    let mut modules = ADAPTER_MODULES.map(String::from).to_vec();
    modules.sort();
    assert_eq!(descriptor.target_modules, modules);
    for layer in &layers {
        let shape = qwen35_projection_shape(&cfg, layer.layer_idx, &layer.module)
            .expect("module is valid for its layer");
        assert_eq!(layer.rank, ADAPTER_RANK);
        assert_eq!((layer.d_in, layer.d_out), (shape.d_in, shape.d_out));
        assert_eq!(layer.a.len(), ADAPTER_RANK * shape.d_in);
        assert_eq!(layer.b.len(), shape.d_out * ADAPTER_RANK);
        assert!(layer.a.iter().chain(&layer.b).all(|v| v.is_finite()));
    }
    let b_peak = layers
        .iter()
        .flat_map(|layer| layer.b.iter())
        .fold(0.0f32, |peak, v| peak.max(v.abs()));
    assert!(
        b_peak > 0.9 * ADAPTER_B_MAGNITUDE && b_peak <= ADAPTER_B_MAGNITUDE,
        "B magnitudes must span the configured range, peak {b_peak}"
    );
}

#[test]
fn expand_home_joins_home_with_a_tilde_prefixed_value() {
    let home = OsStr::new("/fixture/h");
    assert_eq!(
        expand_home(OsStr::new("~/models/x"), Some(home)),
        PathBuf::from("/fixture/h/models/x")
    );
}

#[test]
fn expand_home_keeps_a_non_utf8_suffix() {
    let home = OsStr::new("/fixture/h");
    let value = OsStr::from_bytes(b"~/m\xff");
    let expected = OsStr::from_bytes(b"/fixture/h/m\xff");
    assert_eq!(expand_home(value, Some(home)), PathBuf::from(expected));
}

#[test]
fn expand_home_leaves_a_tilde_value_unchanged_without_home() {
    assert_eq!(
        expand_home(OsStr::new("~/models/x"), None),
        PathBuf::from("~/models/x")
    );
}

#[test]
fn expand_home_leaves_a_value_without_the_prefix_unchanged() {
    let home = OsStr::new("/fixture/h");
    for value in ["models/x", "/fixture/abs", "~x/y", "~"] {
        assert_eq!(
            expand_home(OsStr::new(value), Some(home)),
            PathBuf::from(value)
        );
    }
}

#[test]
fn serve_applies_and_removes_a_lora_adapter_on_the_metal_worker() {
    let Some(model_dir) = require_model_dir() else {
        return;
    };
    let cfg = Qwen35Config::from_model_dir(&model_dir)
        .unwrap_or_else(|err| panic!("reading config.json in {}: {err}", model_dir.display()));
    let adapter_dir = tempfile::tempdir().expect("tempdir");
    let adapter_path = adapter_dir.path().join("synthetic_mlp_adapter.safetensors");
    write_adapter(&adapter_path, &cfg, ADAPTER_SEED);
    let expected_layers = cfg.num_hidden_layers * ADAPTER_MODULES.len();

    let _gpu_guard = gpu_test_lock();
    let server = Server::spawn(&model_dir);

    // Nothing resident, nothing applied.
    assert_residency(&server, &[], &[], "fresh server");

    // 1. Base output, and a repeat of it. The repeat shows the base trajectory is
    //    stable across identical requests here; it does not by itself show that a
    //    later difference comes from the adapter. The scale-0 / scale-1 pair below
    //    adds that: the same adapter stays resident and selected, only the requested
    //    scale moves, and the output moves from equal-to-base to different-from-base.
    let base = server.chat(None);
    assert!(
        base.message.contains("\"content\":\"") && base.completion_tokens > 0,
        "the base model must generate text: {base:?}"
    );
    assert_eq!(
        server.chat(None),
        base,
        "base decoding must repeat for identical requests"
    );

    // 2. Load. Residency changes; nothing is applied by loading alone.
    let (id, loaded) = server.load_adapter(&adapter_path);
    assert_eq!(loaded["rank"], ADAPTER_RANK, "load body: {loaded}");
    assert_eq!(loaded["layers"], expected_layers, "load body: {loaded}");
    assert_eq!(loaded["name"], ADAPTER_NAME, "load body: {loaded}");
    assert_residency(&server, &[id], &[], "after load");

    // 3a. A resident adapter is not applied to a request that does not name it.
    assert_eq!(
        server.chat(None),
        base,
        "a resident but unnamed adapter must not change base output"
    );

    // 3b. The request scale multiplies the adapter's own alpha / rank scale. Naming the
    //     adapter at request scale 0 selects it (it is published as applied) with an
    //     effective scale of 0, so the output must be exactly the base output.
    assert_eq!(
        server.chat(Some((id, 0.0))),
        base,
        "naming the adapter at request scale 0 must reproduce base output"
    );
    assert_residency(&server, &[id], &[id], "after a scale-0 request");

    // 4. The same adapter at request scale 1 (effective scale alpha / rank = 2) must
    //    change the output. This is the assertion that fails when
    //    `QwenMetalRuntime::generate` stops applying the selection, and, paired with
    //    3b, when the registry ignores the requested scale.
    let adapted = server.chat(Some((id, 1.0)));
    assert_ne!(
        adapted, base,
        "naming the adapter at scale 1 left the output identical to base; the adapter was \
         not applied (or the synthetic magnitude is too small for this checkpoint)"
    );
    assert_residency(&server, &[id], &[id], "after a scale-1 request");
    assert_eq!(
        server.chat(Some((id, 1.0))),
        adapted,
        "decoding with the adapter must repeat for identical requests"
    );

    // 3c. The same request without naming the adapter returns to base: the runtime
    //     applies the empty selection, which unloads the slot because an adapter was
    //     applied.
    assert_eq!(
        server.chat(None),
        base,
        "an unnamed request after an adapted one must return to base output"
    );
    assert_residency(&server, &[id], &[], "after reverting to the base selection");

    // Re-applying after the revert reproduces the first adapted output exactly.
    assert_eq!(
        server.chat(Some((id, 1.0))),
        adapted,
        "re-applying the adapter must reproduce the first adapted output"
    );
    assert_residency(&server, &[id], &[id], "after re-applying");

    // 5. Unload while applied. The residency snapshot must show the adapter gone and
    //    nothing applied immediately, before any later request could mask a stale slot.
    server.unload_adapter(id);
    assert_residency(&server, &[], &[], "after unloading the applied adapter");
    let (status, body) = server.chat_raw(Some((id, 1.0)));
    assert_eq!(
        status, 400,
        "a request naming an unloaded adapter must be refused: {body}"
    );
    assert!(
        body.contains("lora_adapter_not_found"),
        "refusal must carry lora_adapter_not_found: {body}"
    );
    assert_eq!(
        server.chat(None),
        base,
        "after unload the base request must reproduce the original base output exactly"
    );
    let (status, body) = server.request("POST", "/v1/lora/unload", Some(&json!({"id": id})));
    assert_eq!(
        status, 400,
        "unloading an unknown id must be refused: {body}"
    );
    assert!(
        body.contains("lora_adapter_not_found"),
        "refusal must carry lora_adapter_not_found: {body}"
    );

    // 6. Load a different adapter (same shapes, different weights) after the unload. It
    //    gets a fresh id and its own output: different from base and from the first
    //    adapter's output, so reusing the first adapter's blend for the new id cannot
    //    pass, and it repeats exactly. Then it unloads cleanly.
    let second_path = adapter_dir
        .path()
        .join("synthetic_mlp_adapter_reload.safetensors");
    write_adapter(&second_path, &cfg, RELOAD_ADAPTER_SEED);
    let (second_id, _) = server.load_adapter(&second_path);
    assert_ne!(second_id, id, "an unloaded id must never be reused");
    assert_residency(&server, &[second_id], &[], "after the second load");
    let reloaded = server.chat(Some((second_id, 1.0)));
    assert_ne!(
        reloaded, base,
        "the second adapter left the output identical to base"
    );
    assert_ne!(
        reloaded, adapted,
        "the second adapter reproduced the first adapter's output; the new id was served \
         with the previous adapter's weights"
    );
    assert_residency(
        &server,
        &[second_id],
        &[second_id],
        "after the second adapter",
    );
    assert_eq!(
        server.chat(Some((second_id, 1.0))),
        reloaded,
        "decoding with the second adapter must repeat for identical requests"
    );
    server.unload_adapter(second_id);
    assert_residency(&server, &[], &[], "after the second unload");
    assert_eq!(
        server.chat(None),
        base,
        "after the second unload the base request must reproduce base output"
    );
}
