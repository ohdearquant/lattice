//! HTTP serving cost against a real `lattice serve` process.
//!
//! Spawns the `lattice` binary as `lattice serve` on an ephemeral loopback
//! port, waits for its `Listening on` line, then drives
//! `POST /v1/chat/completions` with a short and a long prompt, streaming and
//! non-streaming. Nothing here links the serving code: the process under
//! measurement is the shipped binary, so the same source measures both sides of
//! a change to the request path, including the handler composition that the
//! in-process `bench_serve_prepare` example never reaches.
//!
//! The binary is not built by this example. Build it first
//! (`cargo build --release -p lattice-inference --bin lattice`), then run this
//! example from the same profile; `BENCH_BIN` names a binary explicitly. The
//! sha256 of the binary that served the run is printed, so two arms that ran
//! the same executable are visible as such.
//!
//! Certification. A run is refused (nonzero exit) rather than reported when
//! the model directory holds no checkpoint, when the server exits before it
//! listens (whatever its status), when a request fails, when a non-streaming
//! response reports zero `usage.completion_tokens`, and when a stream carries an
//! error event, no content delta, no finish reason or no `[DONE]` terminator.
//! Streaming chunks carry no `usage` object, so a stream is certified by its
//! content deltas and reports `completion_tokens=na`.
//!
//! Env:
//!   LATTICE_MODEL_DIR          checkpoint directory (default
//!                              ~/.lattice/models/qwen3.5-0.8b)
//!   BENCH_BIN                  `lattice` binary (default: next to this
//!                              example's profile directory)
//!   BENCH_RUNS                 measured runs per case and mode, after one
//!                              untimed warmup (default 5)
//!   BENCH_MAX_TOKENS           `max_tokens` per request, 1..=4096 (default 32)
//!   BENCH_LONG_REPEATS         repetitions of the long-prompt paragraph
//!                              (default 100); `prompt_tokens` reports what
//!                              that came to
//!   BENCH_STARTUP_TIMEOUT_SECS wait for `Listening on` (default 900)
//!   BENCH_REQUEST_TIMEOUT_SECS wall clock per request (default 900)
//!   BENCH_STDERR_MARKER        substring to count in the server's stderr
//!
//! Output:
//!   ROUTE binary=<path> binary_bytes=<n> binary_sha256=<hex> model_dir=<dir>
//!     format=<safetensors|q4>
//!   STARTUP startup_ms=<f>
//!   RESULT case=<short|long> mode=<nonstream|stream> run=<n>
//!     prompt_tokens=<n|na> completion_tokens=<n|na> deltas=<n|na>
//!     first_event_ms=<f|na> first_delta_ms=<f|na> total_ms=<f> finish=<reason>
//!   MARKER pattern=<p> count=<n> requests=<n>   (only with BENCH_STDERR_MARKER)
//!   SERVER_STDERR <line>
//!   SKIP reason=<...>      (exit status 2: nothing was measured)
//!   REFUSED reason=<...>   (exit status 1)
//!
//! `first_event_ms` is the first SSE event (the role chunk); `first_delta_ms`
//! is the first event carrying text. Neither this example nor the `lattice
//! serve` process it spawns calls `gpu_test_lock()`, so on a Metal host the
//! machine-wide GPU lock is the caller's to hold for the whole window:
//! `scripts/bench-command.sh --durable` takes it.

use lattice_inference::model_format::{ModelFormat, detect_format};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MODEL_ID: &str = "bench-model";
const LISTENING_PREFIX: &str = "Listening on ";
const SERVER_MAX_TOKENS_CAP: usize = 4096;
const STDERR_TAIL_LINES: usize = 20;
const POLL: Duration = Duration::from_millis(25);
const SHORT_PROMPT: &str = "Name three primary colors and say which one you like best.";
const LONG_PARAGRAPH: &str = "The quick brown fox jumps over the lazy dog while the river keeps running past the old stone bridge.";
const LONG_QUESTION: &str = "Summarize the passage above in one sentence.";

#[derive(Debug)]
enum Refusal {
    BadConfig(String),
    BinaryAbsent(String),
    ModelAbsent(String),
    Spawn(String),
    ServerExited {
        status: String,
        stderr_tail: String,
    },
    NotListening {
        waited_secs: u64,
        stderr_tail: String,
    },
    Transport(String),
    HttpStatus {
        status: u16,
        body: String,
    },
    Timeout {
        waited_secs: u64,
    },
    MalformedBody(String),
    ServerError(String),
    ZeroCompletionTokens,
    StreamError(String),
    NoContentDelta,
    StreamUnfinished,
}

impl Refusal {
    fn exit_code(&self) -> i32 {
        match self {
            Self::BinaryAbsent(_) | Self::ModelAbsent(_) => 2,
            _ => 1,
        }
    }

    fn verdict(&self) -> &'static str {
        if self.exit_code() == 2 {
            "SKIP"
        } else {
            "REFUSED"
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadConfig(why) => write!(f, "bad_config {why}"),
            Self::BinaryAbsent(path) => write!(f, "binary_absent path={path}"),
            Self::ModelAbsent(why) => write!(f, "checkpoint_absent {why}"),
            Self::Spawn(why) => write!(f, "spawn_failed {why}"),
            Self::ServerExited {
                status,
                stderr_tail,
            } => write!(
                f,
                "server_exited_before_listening status={status} stderr_tail={stderr_tail:?}"
            ),
            Self::NotListening {
                waited_secs,
                stderr_tail,
            } => write!(
                f,
                "server_not_listening waited_secs={waited_secs} stderr_tail={stderr_tail:?}"
            ),
            Self::Transport(why) => write!(f, "transport {why}"),
            Self::HttpStatus { status, body } => write!(f, "http_status={status} body={body:?}"),
            Self::Timeout { waited_secs } => write!(f, "request_timeout waited_secs={waited_secs}"),
            Self::MalformedBody(why) => write!(f, "malformed_response {why}"),
            Self::ServerError(error) => write!(f, "server_error {error}"),
            Self::ZeroCompletionTokens => write!(f, "zero_completion_tokens"),
            Self::StreamError(error) => write!(f, "stream_error {error}"),
            Self::NoContentDelta => write!(f, "stream_without_content_delta"),
            Self::StreamUnfinished => write!(f, "stream_without_finish_reason_or_done"),
        }
    }
}

struct Config {
    bin: PathBuf,
    model_dir: PathBuf,
    runs: usize,
    max_tokens: usize,
    long_repeats: usize,
    startup_timeout: Duration,
    request_timeout: Duration,
    marker: Option<String>,
}

fn positive(
    lookup: &dyn Fn(&str) -> Option<String>,
    name: &str,
    default: usize,
) -> Result<usize, Refusal> {
    match lookup(name) {
        None => Ok(default),
        Some(raw) => match raw.trim().parse::<usize>() {
            Ok(value) if value > 0 => Ok(value),
            _ => Err(Refusal::BadConfig(format!(
                "{name}={raw:?} is not a positive integer"
            ))),
        },
    }
}

impl Config {
    fn from_env() -> Result<Self, Refusal> {
        Self::from_lookup(&|name| std::env::var(name).ok())
    }

    fn from_lookup(lookup: &dyn Fn(&str) -> Option<String>) -> Result<Self, Refusal> {
        let model_dir = match lookup("LATTICE_MODEL_DIR") {
            Some(dir) => PathBuf::from(dir),
            None => {
                let home = lookup("HOME").ok_or_else(|| {
                    Refusal::BadConfig("set LATTICE_MODEL_DIR or HOME".to_string())
                })?;
                PathBuf::from(home).join(".lattice/models/qwen3.5-0.8b")
            }
        };
        let bin = match lookup("BENCH_BIN") {
            Some(path) => PathBuf::from(path),
            None => default_binary()?,
        };
        let max_tokens = positive(lookup, "BENCH_MAX_TOKENS", 32)?;
        if max_tokens > SERVER_MAX_TOKENS_CAP {
            return Err(Refusal::BadConfig(format!(
                "BENCH_MAX_TOKENS={max_tokens} exceeds the server cap {SERVER_MAX_TOKENS_CAP}"
            )));
        }
        let marker = match lookup("BENCH_STDERR_MARKER") {
            None => None,
            Some(pattern) if pattern.is_empty() => {
                return Err(Refusal::BadConfig(
                    "BENCH_STDERR_MARKER must not be empty".to_string(),
                ));
            }
            Some(pattern) => Some(pattern),
        };
        Ok(Self {
            bin,
            model_dir,
            runs: positive(lookup, "BENCH_RUNS", 5)?,
            max_tokens,
            long_repeats: positive(lookup, "BENCH_LONG_REPEATS", 100)?,
            startup_timeout: Duration::from_secs(positive(
                lookup,
                "BENCH_STARTUP_TIMEOUT_SECS",
                900,
            )? as u64),
            request_timeout: Duration::from_secs(positive(
                lookup,
                "BENCH_REQUEST_TIMEOUT_SECS",
                900,
            )? as u64),
            marker,
        })
    }
}

fn default_binary() -> Result<PathBuf, Refusal> {
    let exe = std::env::current_exe().map_err(|e| Refusal::BinaryAbsent(e.to_string()))?;
    let profile_dir = exe
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| Refusal::BinaryAbsent(exe.display().to_string()))?;
    Ok(profile_dir.join(format!("lattice{}", std::env::consts::EXE_SUFFIX)))
}

fn binary_identity(path: &Path) -> Result<(usize, String), Refusal> {
    let bytes = std::fs::read(path)
        .map_err(|e| Refusal::BinaryAbsent(format!("{}: {e}", path.display())))?;
    Ok((bytes.len(), format!("{:x}", Sha256::digest(&bytes))))
}

fn format_name(format: ModelFormat) -> &'static str {
    match format {
        ModelFormat::Safetensors => "safetensors",
        ModelFormat::Q4 => "q4",
        _ => "unknown",
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn tail(lines: &[String]) -> String {
    let from = lines.len().saturating_sub(STDERR_TAIL_LINES);
    lines[from..].join("\n")
}

struct Server {
    child: Child,
    stderr: Arc<Mutex<Vec<String>>>,
    reader: JoinHandle<()>,
}

impl Server {
    fn stderr_lines(&self) -> Vec<String> {
        self.stderr
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_stderr(stderr: impl Read, lines: &Mutex<Vec<String>>, listening: &mpsc::Sender<Instant>) {
    let mut reader = BufReader::new(stderr);
    let mut raw = Vec::new();
    let mut announced = false;
    loop {
        raw.clear();
        match reader.read_until(b'\n', &mut raw) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&raw)
            .trim_end_matches(['\r', '\n'])
            .to_string();
        let seen_at = Instant::now();
        let is_listening = !announced && line.starts_with(LISTENING_PREFIX);
        lines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(line);
        if is_listening {
            announced = true;
            let _ = listening.send(seen_at);
        }
    }
}

fn start_server(mut command: Command, timeout: Duration) -> Result<(Server, f64), Refusal> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let started = Instant::now();
    let mut child = command.spawn().map_err(|e| Refusal::Spawn(e.to_string()))?;
    let Some(stderr) = child.stderr.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(Refusal::Spawn("child stderr was not piped".to_string()));
    };
    let lines = Arc::new(Mutex::new(Vec::new()));
    let (listening_tx, listening_rx) = mpsc::channel::<Instant>();
    let reader = std::thread::spawn({
        let lines = Arc::clone(&lines);
        move || read_stderr(stderr, &lines, &listening_tx)
    });
    let mut server = Server {
        child,
        stderr: lines,
        reader,
    };
    loop {
        match listening_rx.recv_timeout(POLL) {
            Ok(at) => return Ok((server, at.duration_since(started).as_secs_f64() * 1000.0)),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => std::thread::sleep(POLL),
        }
        let exited = server
            .child
            .try_wait()
            .map_err(|e| Refusal::Spawn(e.to_string()))?;
        if let Some(status) = exited {
            let grace = Instant::now();
            while !server.reader.is_finished() && grace.elapsed() < Duration::from_secs(1) {
                std::thread::sleep(Duration::from_millis(10));
            }
            return Err(Refusal::ServerExited {
                status: status.to_string(),
                stderr_tail: tail(&server.stderr_lines()),
            });
        }
        if started.elapsed() > timeout {
            return Err(Refusal::NotListening {
                waited_secs: timeout.as_secs(),
                stderr_tail: tail(&server.stderr_lines()),
            });
        }
    }
}

fn free_loopback_port() -> Result<u16, Refusal> {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| Refusal::Spawn(e.to_string()))?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|e| Refusal::Spawn(e.to_string()))
}

fn serve_command(cfg: &Config, port: u16) -> Command {
    let mut command = Command::new(&cfg.bin);
    command
        .arg("serve")
        .arg("--model")
        .arg(&cfg.model_dir)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--model-id")
        .arg(MODEL_ID);
    command
}

#[derive(Debug)]
struct NonStreamStats {
    prompt_tokens: u64,
    completion_tokens: u64,
    finish_reason: String,
}

fn certify_nonstream(body: &str) -> Result<NonStreamStats, Refusal> {
    let value: Value =
        serde_json::from_str(body).map_err(|e| Refusal::MalformedBody(e.to_string()))?;
    if let Some(error) = value.get("error") {
        return Err(Refusal::ServerError(error.to_string()));
    }
    let completion_tokens = value["usage"]["completion_tokens"]
        .as_u64()
        .ok_or_else(|| {
            Refusal::MalformedBody(
                "usage.completion_tokens is missing or not an integer".to_string(),
            )
        })?;
    let prompt_tokens = value["usage"]["prompt_tokens"].as_u64().ok_or_else(|| {
        Refusal::MalformedBody("usage.prompt_tokens is missing or not an integer".to_string())
    })?;
    if completion_tokens == 0 {
        return Err(Refusal::ZeroCompletionTokens);
    }
    let finish_reason = value["choices"][0]["finish_reason"]
        .as_str()
        .ok_or_else(|| Refusal::MalformedBody("choices[0].finish_reason is missing".to_string()))?
        .to_string();
    Ok(NonStreamStats {
        prompt_tokens,
        completion_tokens,
        finish_reason,
    })
}

fn sse_data(line: &str) -> Option<&str> {
    let line = line.trim_end_matches(['\r', '\n']);
    let data = line.strip_prefix("data:")?;
    Some(data.strip_prefix(' ').unwrap_or(data))
}

#[derive(Debug, Default)]
struct StreamTally {
    first_event_ms: Option<f64>,
    first_delta_ms: Option<f64>,
    content_deltas: usize,
    finish_reason: Option<String>,
    error: Option<String>,
    saw_done: bool,
}

impl StreamTally {
    fn feed_data(&mut self, data: &str, elapsed_ms: f64) -> Result<(), Refusal> {
        if data == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }
        let value: Value =
            serde_json::from_str(data).map_err(|e| Refusal::MalformedBody(e.to_string()))?;
        if let Some(error) = value.get("error") {
            self.error = Some(error.to_string());
            return Ok(());
        }
        self.first_event_ms.get_or_insert(elapsed_ms);
        let choice = &value["choices"][0];
        if let Some(text) = choice["delta"]["content"].as_str()
            && !text.is_empty()
        {
            self.content_deltas += 1;
            self.first_delta_ms.get_or_insert(elapsed_ms);
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish_reason = Some(reason.to_string());
        }
        Ok(())
    }
}

#[derive(Debug)]
struct StreamStats {
    deltas: usize,
    first_event_ms: f64,
    first_delta_ms: f64,
    finish_reason: String,
}

fn certify_stream(tally: &StreamTally) -> Result<StreamStats, Refusal> {
    if let Some(error) = &tally.error {
        return Err(Refusal::StreamError(error.clone()));
    }
    let (Some(first_event_ms), Some(first_delta_ms)) = (tally.first_event_ms, tally.first_delta_ms)
    else {
        return Err(Refusal::NoContentDelta);
    };
    let Some(finish_reason) = tally.finish_reason.clone() else {
        return Err(Refusal::StreamUnfinished);
    };
    if !tally.saw_done {
        return Err(Refusal::StreamUnfinished);
    }
    Ok(StreamStats {
        deltas: tally.content_deltas,
        first_event_ms,
        first_delta_ms,
        finish_reason,
    })
}

struct Client {
    agent: ureq::Agent,
    base: String,
    request_timeout: Duration,
}

impl Client {
    fn new(base: String, request_timeout: Duration) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .timeout_read(request_timeout)
                .build(),
            base,
            request_timeout,
        }
    }

    fn check_health(&self) -> Result<(), Refusal> {
        match self.agent.get(&format!("{}/health", self.base)).call() {
            Ok(response) if response.status() == 200 => Ok(()),
            Ok(response) => Err(Refusal::HttpStatus {
                status: response.status(),
                body: String::new(),
            }),
            Err(ureq::Error::Status(status, response)) => Err(Refusal::HttpStatus {
                status,
                body: response.into_string().unwrap_or_default(),
            }),
            Err(e) => Err(Refusal::Transport(e.to_string())),
        }
    }

    fn post_chat(&self, body: &Value) -> Result<ureq::Response, Refusal> {
        let bytes = serde_json::to_vec(body).map_err(|e| Refusal::MalformedBody(e.to_string()))?;
        match self
            .agent
            .post(&format!("{}/v1/chat/completions", self.base))
            .set("content-type", "application/json")
            .send_bytes(&bytes)
        {
            Ok(response) => Ok(response),
            Err(ureq::Error::Status(status, response)) => Err(Refusal::HttpStatus {
                status,
                body: response.into_string().unwrap_or_default(),
            }),
            Err(e) => Err(Refusal::Transport(e.to_string())),
        }
    }

    fn run_nonstream(&self, body: &Value) -> Result<(NonStreamStats, f64), Refusal> {
        let start = Instant::now();
        let response = self.post_chat(body)?;
        let text = response
            .into_string()
            .map_err(|e| Refusal::Transport(e.to_string()))?;
        let total_ms = ms(start);
        Ok((certify_nonstream(&text)?, total_ms))
    }

    fn run_stream(&self, body: &Value) -> Result<(StreamStats, f64), Refusal> {
        let start = Instant::now();
        let response = self.post_chat(body)?;
        let mut reader = BufReader::new(response.into_reader());
        let mut tally = StreamTally::default();
        let mut line = String::new();
        loop {
            line.clear();
            let read = reader
                .read_line(&mut line)
                .map_err(|e| Refusal::Transport(e.to_string()))?;
            if read == 0 {
                break;
            }
            if let Some(data) = sse_data(&line) {
                tally.feed_data(data, ms(start))?;
                if tally.saw_done {
                    break;
                }
            }
            if start.elapsed() > self.request_timeout {
                return Err(Refusal::Timeout {
                    waited_secs: self.request_timeout.as_secs(),
                });
            }
        }
        let total_ms = ms(start);
        Ok((certify_stream(&tally)?, total_ms))
    }
}

#[derive(Clone, Copy)]
enum Case {
    Short,
    Long,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Long => "long",
        }
    }

    fn prompt(self, cfg: &Config) -> String {
        match self {
            Self::Short => SHORT_PROMPT.to_string(),
            Self::Long => {
                let mut prompt = String::new();
                for paragraph in 0..cfg.long_repeats {
                    prompt.push_str(&format!("Paragraph {paragraph}: {LONG_PARAGRAPH} "));
                }
                prompt.push_str(LONG_QUESTION);
                prompt
            }
        }
    }
}

fn request_body(case: Case, run: usize, stream: bool, cfg: &Config) -> Value {
    json!({
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": format!("Request {run}. {}", case.prompt(cfg)),
        }],
        "max_tokens": cfg.max_tokens,
        "temperature": 0.0,
        "stream": stream,
    })
}

#[derive(Default)]
struct Progress {
    requests: usize,
}

fn measure(client: &Client, cfg: &Config, progress: &mut Progress) -> Result<(), Refusal> {
    client.check_health()?;
    for case in [Case::Short, Case::Long] {
        for stream in [false, true] {
            let mode = if stream { "stream" } else { "nonstream" };
            for run in 0..=cfg.runs {
                let body = request_body(case, run, stream, cfg);
                progress.requests += 1;
                let row = if stream {
                    let (stats, total_ms) = client.run_stream(&body)?;
                    format!(
                        "prompt_tokens=na completion_tokens=na deltas={} first_event_ms={:.1} \
                         first_delta_ms={:.1} total_ms={total_ms:.1} finish={}",
                        stats.deltas,
                        stats.first_event_ms,
                        stats.first_delta_ms,
                        stats.finish_reason
                    )
                } else {
                    let (stats, total_ms) = client.run_nonstream(&body)?;
                    format!(
                        "prompt_tokens={} completion_tokens={} deltas=na first_event_ms=na \
                         first_delta_ms=na total_ms={total_ms:.1} finish={}",
                        stats.prompt_tokens, stats.completion_tokens, stats.finish_reason
                    )
                };
                if run > 0 {
                    println!("RESULT case={} mode={mode} run={run} {row}", case.name());
                }
            }
        }
    }
    Ok(())
}

fn count_marker_lines(lines: &[String], pattern: &str) -> usize {
    lines.iter().filter(|line| line.contains(pattern)).count()
}

fn report_stderr(cfg: &Config, lines: &[String], requests: usize) {
    if let Some(pattern) = &cfg.marker {
        println!(
            "MARKER pattern={pattern:?} count={} requests={requests}",
            count_marker_lines(lines, pattern)
        );
    }
    for line in lines {
        println!("SERVER_STDERR {line}");
    }
}

fn run(cfg: &Config) -> Result<(), Refusal> {
    let format = detect_format(&cfg.model_dir);
    if !cfg.model_dir.is_dir() || format == ModelFormat::Unknown {
        return Err(Refusal::ModelAbsent(format!(
            "model_dir={} format={}",
            cfg.model_dir.display(),
            format_name(format)
        )));
    }
    let (binary_bytes, binary_sha256) = binary_identity(&cfg.bin)?;
    println!(
        "ROUTE binary={} binary_bytes={binary_bytes} binary_sha256={binary_sha256} \
         model_dir={} format={}",
        cfg.bin.display(),
        cfg.model_dir.display(),
        format_name(format)
    );
    let port = free_loopback_port()?;
    let (server, startup_ms) = start_server(serve_command(cfg, port), cfg.startup_timeout)?;
    println!("STARTUP startup_ms={startup_ms:.1}");
    let client = Client::new(format!("http://127.0.0.1:{port}"), cfg.request_timeout);
    let mut progress = Progress::default();
    let outcome = measure(&client, cfg, &mut progress);
    report_stderr(cfg, &server.stderr_lines(), progress.requests);
    outcome
}

fn finish(refusal: &Refusal) -> ! {
    println!("{} reason={refusal}", refusal.verdict());
    std::process::exit(refusal.exit_code());
}

fn main() {
    let cfg = match Config::from_env() {
        Ok(cfg) => cfg,
        Err(refusal) => finish(&refusal),
    };
    if let Err(refusal) = run(&cfg) {
        finish(&refusal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};

    fn lookup_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let owned: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |name| {
            owned
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        }
    }

    fn test_config(bin: PathBuf, model_dir: PathBuf) -> Config {
        Config {
            bin,
            model_dir,
            runs: 1,
            max_tokens: 4,
            long_repeats: 2,
            startup_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(10),
            marker: None,
        }
    }

    fn model_dir_with_weights() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"").unwrap();
        dir
    }

    fn fake_binary(dir: &Path, name: &str, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn data_chunk(text: &str, finish: Option<&str>) -> String {
        let delta = if text.is_empty() {
            json!({})
        } else {
            json!({"content": text})
        };
        json!({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).to_string()
    }

    fn role_chunk() -> String {
        json!({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]})
            .to_string()
    }

    fn feed(events: &[String]) -> Result<StreamTally, Refusal> {
        let mut tally = StreamTally::default();
        for (at, event) in events.iter().enumerate() {
            tally.feed_data(event, at as f64)?;
        }
        Ok(tally)
    }

    #[test]
    fn missing_model_dir_is_a_skip_with_a_nonzero_status() {
        let scratch = tempfile::tempdir().unwrap();
        let bin = fake_binary(scratch.path(), "lattice", "exit 0");
        let cfg = test_config(bin, scratch.path().join("no-such-checkpoint"));
        let refusal = run(&cfg).unwrap_err();
        assert!(matches!(refusal, Refusal::ModelAbsent(_)), "{refusal}");
        assert_eq!(refusal.exit_code(), 2);
        assert_eq!(refusal.verdict(), "SKIP");
    }

    #[test]
    fn model_dir_without_weights_is_a_skip() {
        let scratch = tempfile::tempdir().unwrap();
        let bin = fake_binary(scratch.path(), "lattice", "exit 0");
        let refusal = run(&test_config(bin, scratch.path().to_path_buf())).unwrap_err();
        assert!(matches!(refusal, Refusal::ModelAbsent(_)), "{refusal}");
    }

    #[test]
    fn missing_binary_is_a_skip_and_never_a_success() {
        let model = model_dir_with_weights();
        let cfg = test_config(
            model.path().join("no-such-lattice"),
            model.path().to_path_buf(),
        );
        let refusal = run(&cfg).unwrap_err();
        assert!(matches!(refusal, Refusal::BinaryAbsent(_)), "{refusal}");
        assert_eq!(refusal.exit_code(), 2);
    }

    #[test]
    fn server_that_exits_before_listening_is_refused_whatever_its_status() {
        for (name, status_line) in [("fails", "exit 1"), ("clean", "exit 0")] {
            let scratch = tempfile::tempdir().unwrap();
            let model = model_dir_with_weights();
            let bin = fake_binary(
                scratch.path(),
                name,
                &format!("echo 'Error: failed to load model: boom' >&2\n{status_line}"),
            );
            let cfg = test_config(bin, model.path().to_path_buf());
            let refusal = run(&cfg).unwrap_err();
            match &refusal {
                Refusal::ServerExited { stderr_tail, .. } => {
                    assert!(stderr_tail.contains("boom"), "{name}: {stderr_tail:?}");
                }
                other => panic!("{name}: expected ServerExited, got {other}"),
            }
            assert_eq!(refusal.exit_code(), 1, "{name}");
        }
    }

    #[test]
    fn server_that_announces_listening_is_admitted_and_timed() {
        let scratch = tempfile::tempdir().unwrap();
        let bin = fake_binary(
            scratch.path(),
            "lattice",
            "echo 'Loading model...' >&2\necho 'Listening on 127.0.0.1:1  (model: m)' >&2\nexec sleep 37",
        );
        let (server, startup_ms) = start_server(Command::new(bin), Duration::from_secs(10))
            .unwrap_or_else(|refusal| panic!("{refusal}"));
        assert!(startup_ms >= 0.0);
        let lines = server.stderr_lines();
        assert!(
            lines.iter().any(|line| line.starts_with(LISTENING_PREFIX)),
            "{lines:?}"
        );
    }

    #[test]
    fn server_that_never_listens_times_out_as_a_refusal() {
        let scratch = tempfile::tempdir().unwrap();
        let bin = fake_binary(
            scratch.path(),
            "lattice",
            "echo 'Loading model...' >&2\nexec sleep 37",
        );
        let refusal = start_server(Command::new(bin), Duration::from_millis(300))
            .err()
            .expect("a server that never listens must be refused");
        assert!(matches!(refusal, Refusal::NotListening { .. }), "{refusal}");
    }

    #[test]
    fn zero_completion_tokens_refuse_certification_whatever_the_text() {
        let body = json!({
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7},
        })
        .to_string();
        assert!(matches!(
            certify_nonstream(&body),
            Err(Refusal::ZeroCompletionTokens)
        ));
    }

    #[test]
    fn positive_completion_tokens_are_certified_and_reported() {
        let body = json!({
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        })
        .to_string();
        let stats = certify_nonstream(&body).unwrap();
        assert_eq!(stats.prompt_tokens, 7);
        assert_eq!(stats.completion_tokens, 3);
        assert_eq!(stats.finish_reason, "length");
    }

    #[test]
    fn nonstream_body_without_usage_or_with_an_error_is_refused() {
        let no_usage = json!({"choices": [{"finish_reason": "stop"}]}).to_string();
        assert!(matches!(
            certify_nonstream(&no_usage),
            Err(Refusal::MalformedBody(_))
        ));
        let error = json!({"error": {"message": "inference failed"}}).to_string();
        assert!(matches!(
            certify_nonstream(&error),
            Err(Refusal::ServerError(_))
        ));
        assert!(matches!(
            certify_nonstream("not json"),
            Err(Refusal::MalformedBody(_))
        ));
    }

    #[test]
    fn sse_lines_yield_data_and_ignore_comments_and_blanks() {
        assert_eq!(sse_data("data: {\"a\":1}\n"), Some("{\"a\":1}"));
        assert_eq!(sse_data("data:{\"a\":1}\r\n"), Some("{\"a\":1}"));
        assert_eq!(sse_data("data: [DONE]\n"), Some("[DONE]"));
        assert_eq!(sse_data(": keep-alive\n"), None);
        assert_eq!(sse_data("\n"), None);
        assert_eq!(sse_data("event: ping\n"), None);
    }

    #[test]
    fn complete_stream_is_certified_with_the_first_delta_after_the_role_chunk() {
        let mut events = vec![
            role_chunk(),
            data_chunk("Hel", None),
            data_chunk("lo", None),
            data_chunk("", Some("length")),
        ];
        events.push("[DONE]".to_string());
        let stats = certify_stream(&feed(&events).unwrap()).unwrap();
        assert_eq!(stats.deltas, 2);
        assert_eq!(stats.first_event_ms, 0.0);
        assert_eq!(stats.first_delta_ms, 1.0);
        assert_eq!(stats.finish_reason, "length");
    }

    #[test]
    fn stream_without_a_content_delta_is_refused() {
        let events = vec![
            role_chunk(),
            data_chunk("", Some("length")),
            "[DONE]".to_string(),
        ];
        assert!(matches!(
            certify_stream(&feed(&events).unwrap()),
            Err(Refusal::NoContentDelta)
        ));
    }

    #[test]
    fn stream_error_event_is_refused_even_with_earlier_deltas() {
        let events = vec![
            role_chunk(),
            data_chunk("Hel", None),
            json!({"error": {"message": "inference failed", "code": "internal_error"}}).to_string(),
            "[DONE]".to_string(),
        ];
        assert!(matches!(
            certify_stream(&feed(&events).unwrap()),
            Err(Refusal::StreamError(_))
        ));
    }

    #[test]
    fn truncated_stream_is_refused_for_a_missing_finish_or_done() {
        let no_done = vec![
            role_chunk(),
            data_chunk("Hel", None),
            data_chunk("", Some("length")),
        ];
        assert!(matches!(
            certify_stream(&feed(&no_done).unwrap()),
            Err(Refusal::StreamUnfinished)
        ));
        let no_finish = vec![role_chunk(), data_chunk("Hel", None), "[DONE]".to_string()];
        assert!(matches!(
            certify_stream(&feed(&no_finish).unwrap()),
            Err(Refusal::StreamUnfinished)
        ));
    }

    #[test]
    fn marker_lines_are_counted_by_substring() {
        let lines = vec![
            "Loading model".to_string(),
            "[M] a=1".to_string(),
            "noise [M] a=2".to_string(),
        ];
        assert_eq!(count_marker_lines(&lines, "[M]"), 2);
        assert_eq!(count_marker_lines(&lines, "[X]"), 0);
    }

    #[test]
    fn config_rejects_unusable_values_and_applies_defaults() {
        let base = [("BENCH_BIN", "/x/lattice"), ("HOME", "/h")];
        let cfg = Config::from_lookup(&lookup_from(&base)).unwrap();
        assert_eq!(cfg.runs, 5);
        assert_eq!(cfg.max_tokens, 32);
        assert!(cfg.model_dir.ends_with(".lattice/models/qwen3.5-0.8b"));
        for (name, value) in [
            ("BENCH_RUNS", "0"),
            ("BENCH_MAX_TOKENS", "0"),
            ("BENCH_MAX_TOKENS", "4097"),
            ("BENCH_LONG_REPEATS", "x"),
            ("BENCH_STDERR_MARKER", ""),
        ] {
            let mut pairs = base.to_vec();
            pairs.push((name, value));
            assert!(
                matches!(
                    Config::from_lookup(&lookup_from(&pairs)),
                    Err(Refusal::BadConfig(_))
                ),
                "{name}={value:?} must be refused"
            );
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Fake {
        Healthy,
        ZeroTokens,
        NoDeltas,
    }

    fn read_request(stream: &mut TcpStream) -> Option<(String, Vec<u8>)> {
        let mut reader = BufReader::new(stream.try_clone().ok()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line).ok()?;
        let mut content_length = 0usize;
        loop {
            let mut header = String::new();
            reader.read_line(&mut header).ok()?;
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some((name, value)) = header.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                content_length = value.trim().parse().ok()?;
            }
        }
        let mut body = vec![0u8; content_length];
        reader.read_exact(&mut body).ok()?;
        Some((request_line, body))
    }

    fn respond(stream: &mut TcpStream, mode: Fake) -> Option<()> {
        let (request_line, body) = read_request(stream)?;
        if request_line.starts_with("GET /health") {
            let payload = b"{\"status\":\"ok\"}";
            write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                payload.len()
            )
            .ok()?;
            stream.write_all(payload).ok()?;
            return Some(());
        }
        let request: Value = serde_json::from_slice(&body).ok()?;
        if request["stream"].as_bool() == Some(true) {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n"
            )
            .ok()?;
            let mut events = vec![role_chunk()];
            if mode != Fake::NoDeltas {
                events.push(data_chunk("Hel", None));
                events.push(data_chunk("lo", None));
            }
            events.push(data_chunk("", Some("length")));
            events.push("[DONE]".to_string());
            for event in events {
                let frame = format!("data: {event}\n\n");
                write!(stream, "{:x}\r\n{frame}\r\n", frame.len()).ok()?;
                stream.flush().ok()?;
                std::thread::sleep(Duration::from_millis(5));
            }
            stream.write_all(b"0\r\n\r\n").ok()?;
            return Some(());
        }
        let completion = if mode == Fake::ZeroTokens { 0 } else { 3 };
        let payload = json!({
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": completion, "total_tokens": 7 + completion},
        })
        .to_string();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
            payload.len()
        )
        .ok()?;
        Some(())
    }

    fn spawn_fake_server(mode: Fake) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let _ = respond(&mut stream, mode);
            }
        });
        port
    }

    fn measure_against(mode: Fake) -> (Result<(), Refusal>, usize) {
        let port = spawn_fake_server(mode);
        let cfg = test_config(PathBuf::new(), PathBuf::new());
        let client = Client::new(format!("http://127.0.0.1:{port}"), cfg.request_timeout);
        let mut progress = Progress::default();
        let outcome = measure(&client, &cfg, &mut progress);
        (outcome, progress.requests)
    }

    #[test]
    fn healthy_http_server_is_certified_in_both_modes_for_both_prompts() {
        let (outcome, requests) = measure_against(Fake::Healthy);
        outcome.unwrap_or_else(|refusal| panic!("{refusal}"));
        assert_eq!(
            requests,
            2 * 2 * (1 + 1),
            "warmup plus one measured run per case and mode"
        );
    }

    #[test]
    fn http_response_reporting_zero_completion_tokens_is_refused() {
        let (outcome, _) = measure_against(Fake::ZeroTokens);
        assert!(
            matches!(outcome, Err(Refusal::ZeroCompletionTokens)),
            "{outcome:?}"
        );
    }

    #[test]
    fn http_stream_without_content_deltas_is_refused() {
        let (outcome, requests) = measure_against(Fake::NoDeltas);
        assert!(
            matches!(outcome, Err(Refusal::NoContentDelta)),
            "{outcome:?}"
        );
        assert_eq!(
            requests, 3,
            "two short non-stream requests succeed, the third (the first stream) is refused"
        );
    }
}
