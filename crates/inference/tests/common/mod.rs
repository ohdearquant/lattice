#![allow(dead_code)]

use serde_json::{Value, json};
use std::io::{BufRead as _, Read as _, Write as _};
use std::net::{Shutdown, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime};

pub const GATE_ENFORCE_ENV: &str = "LATTICE_SERVE_CHARACTERIZATION_GATE_ENFORCE";
pub const QWEN_DIR_ENV: &str = "LATTICE_SERVE_STANDALONE_QWEN_DIR";
pub const GEMMA_DIR_ENV: &str = "LATTICE_SERVE_STANDALONE_GEMMA_DIR";
pub const QWEN_Q4_DIR_ENV: &str = "LATTICE_SERVE_CHARACTERIZATION_QWEN_Q4_DIR";
pub const SKIP_MARKER: &str = "LATTICE_SERVE_CHARACTERIZATION_SKIPPED";

const STARTUP_DEADLINE: Duration = Duration::from_secs(120);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(480);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(480);
const ADDRESS_ANNOUNCEMENT_TIMEOUT: Duration = Duration::from_secs(2);

const SERVER_STARTUP_ENV: &[&str] = &[
    "HOME",
    "LATTICE_MODEL_CACHE",
    "LATTICE_SERVE_MODEL",
    "LATTICE_SERVE_PORT",
    "LATTICE_SERVE_EMBEDDING_MODEL",
    "LATTICE_SERVE_EMBEDDING_TEST_MODEL_DIR",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Lattice,
    #[cfg(all(feature = "metal-gpu", feature = "f16"))]
    LatticeServe,
}

impl Binary {
    fn command(self) -> Command {
        match self {
            Self::Lattice => Command::new(env!("CARGO_BIN_EXE_lattice")),
            #[cfg(all(feature = "metal-gpu", feature = "f16"))]
            Self::LatticeServe => Command::new(env!("CARGO_BIN_EXE_lattice_serve")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Lattice => "lattice serve",
            #[cfg(all(feature = "metal-gpu", feature = "f16"))]
            Self::LatticeServe => "lattice_serve",
        }
    }
}

pub struct StartupRun {
    pub code: Option<i32>,
    pub stderr: String,
}

pub fn gemma_config() -> Vec<u8> {
    include_bytes!("../fixtures/gemma4/e2b_config.json").to_vec()
}

pub fn qwen_config() -> Vec<u8> {
    br#"{"model_type":"qwen3_5"}"#.to_vec()
}

pub fn write_model_dir(dir: &Path, config: &[u8], weights_file: Option<&str>) {
    std::fs::create_dir_all(dir).expect("create model directory");
    std::fs::write(dir.join("config.json"), config).expect("write config.json");
    if let Some(weights_file) = weights_file {
        std::fs::write(dir.join(weights_file), b"stub").expect("write stub weights");
    }
}

pub fn stub_model(config: &[u8], weights_file: Option<&str>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temporary model directory");
    write_model_dir(dir.path(), config, weights_file);
    dir
}

pub fn run_startup(binary: Binary, model: Option<&Path>, extra: &[String]) -> StartupRun {
    let mut command = binary.command();
    remove_server_startup_env(&mut command);
    match binary {
        Binary::Lattice => {
            command.args(["serve", "--host", "127.0.0.1", "--port", "0"]);
        }
        #[cfg(all(feature = "metal-gpu", feature = "f16"))]
        Binary::LatticeServe => {
            command.args(["--host", "127.0.0.1", "--port", "0"]);
        }
    }
    if let Some(model) = model {
        command.arg("--model").arg(model);
    }
    command.args(extra);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .unwrap_or_else(|error| panic!("spawn {}: {error}", binary.label()));
    let mut stderr_pipe = child.stderr.take().expect("child stderr pipe");
    let stderr_reader = std::thread::spawn(move || {
        let mut stderr = Vec::new();
        stderr_pipe
            .read_to_end(&mut stderr)
            .expect("read startup stderr");
        stderr
    });
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .unwrap_or_else(|error| panic!("poll {}: {error}", binary.label()))
        {
            break status;
        }
        if started.elapsed() >= STARTUP_DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            let stderr = String::from_utf8_lossy(
                &stderr_reader.join().expect("startup stderr reader joins"),
            )
            .into_owned();
            panic!(
                "{} did not exit on a startup-failure fixture; stderr:\n{stderr}",
                binary.label(),
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let stderr =
        String::from_utf8_lossy(&stderr_reader.join().expect("startup stderr reader joins"))
            .into_owned();
    StartupRun {
        code: status.code(),
        stderr,
    }
}

fn remove_server_startup_env(command: &mut Command) {
    for name in SERVER_STARTUP_ENV {
        command.env_remove(name);
    }
}

pub fn assert_startup_golden(
    run: &StartupRun,
    temp_roots: &[(&Path, &str)],
    expected_code: Option<i32>,
    expected_last_line: &str,
    expected_markers: &[&str],
) {
    let stderr = normalize_paths(&run.stderr, temp_roots);
    assert_eq!(run.code, expected_code, "stderr:\n{stderr}");
    let last_line = stderr.trim_end().lines().last().unwrap_or("");
    assert_eq!(last_line, expected_last_line, "stderr:\n{stderr}");

    let markers = startup_marker_lines(&stderr);
    assert_eq!(
        markers, expected_markers,
        "startup marker order; stderr:\n{stderr}"
    );
}

fn normalize_paths(text: &str, roots: &[(&Path, &str)]) -> String {
    let mut roots = roots
        .iter()
        .map(|(path, placeholder)| (path.display().to_string(), *placeholder))
        .collect::<Vec<_>>();
    roots.sort_by_key(|(path, _)| std::cmp::Reverse(path.len()));
    roots
        .into_iter()
        .fold(text.to_owned(), |text, (path, placeholder)| {
            text.replace(&path, placeholder)
        })
}

fn startup_marker_lines(stderr: &str) -> Vec<&str> {
    stderr
        .lines()
        .filter(|line| {
            line.starts_with("[route] selected ")
                || line.starts_with("[lattice_serve] loading model from ")
                || line.starts_with("[lattice_serve] model '")
                || line.starts_with("[lattice_serve] WARNING:")
                || line.starts_with("Loading model from ")
                || line.starts_with("Model loaded. Serving as '")
                || line.starts_with("Listening on ")
                || line.starts_with("Embeddings disabled (")
                || line.starts_with("Embeddings enabled:")
                || line.starts_with("Router gate loaded:")
                || line.starts_with("Warning: --preload-vision")
        })
        .collect()
}

pub fn require_checkpoint(env_name: &str, default_name: &str, cell: &str) -> Option<PathBuf> {
    let path = match std::env::var_os(env_name) {
        Some(path) => PathBuf::from(path),
        None => std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(".lattice")
            .join("models")
            .join(default_name),
    };
    if path.join("config.json").is_file() {
        return Some(path);
    }
    if std::env::var(GATE_ENFORCE_ENV).as_deref() == Ok("1") {
        panic!(
            "no checkpoint at {} ({env_name}) while {GATE_ENFORCE_ENV}=1; cell={cell}",
            path.display()
        );
    }
    eprintln!(
        "{SKIP_MARKER} reason=no_checkpoint cell={cell} path={} ({env_name})",
        path.display()
    );
    None
}

pub fn stage_q4_with_tokenizer(q4_dir: &Path, tokenizer_dir: &Path) -> tempfile::TempDir {
    let staged = tempfile::tempdir().expect("temporary staged Q4 directory");
    let q4_files_before = checkpoint_file_metadata(q4_dir);
    let tokenizer_files_before = checkpoint_file_metadata(tokenizer_dir);
    copy_tree_hard_link(q4_dir, staged.path());
    for entry in std::fs::read_dir(tokenizer_dir).expect("read tokenizer checkpoint") {
        let entry = entry.expect("read tokenizer checkpoint entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("tokenizer")
            || name.starts_with("vocab")
            || name.starts_with("merges")
            || name == "special_tokens_map.json"
        {
            let destination = staged.path().join(entry.file_name());
            if entry.file_type().expect("inspect tokenizer file").is_file() {
                hard_link_or_copy(&entry.path(), &destination);
            }
        }
    }
    assert!(
        staged.path().join("tokenizer.json").is_file(),
        "staged Q4 checkpoint has tokenizer.json"
    );
    assert_eq!(
        checkpoint_file_metadata(q4_dir),
        q4_files_before,
        "Q4 checkpoint files changed while staging"
    );
    assert_eq!(
        checkpoint_file_metadata(tokenizer_dir),
        tokenizer_files_before,
        "tokenizer checkpoint files changed while staging"
    );
    staged
}

fn checkpoint_file_metadata(root: &Path) -> Vec<(PathBuf, u64, SystemTime)> {
    fn collect_files(directory: &Path, files: &mut Vec<(PathBuf, u64, SystemTime)>) {
        for entry in std::fs::read_dir(directory).expect("read checkpoint directory") {
            let entry = entry.expect("read checkpoint entry");
            let path = entry.path();
            let file_type = entry.file_type().expect("inspect checkpoint entry");
            if file_type.is_dir() {
                collect_files(&path, files);
            } else if file_type.is_file() {
                let metadata = entry.metadata().expect("read checkpoint file metadata");
                files.push((
                    path,
                    metadata.len(),
                    metadata.modified().expect("read checkpoint file mtime"),
                ));
            }
        }
    }

    let mut files = Vec::new();
    collect_files(root, &mut files);
    files.sort_by(|left, right| left.0.cmp(&right.0));
    files
}

fn copy_tree_hard_link(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination).expect("create staged checkpoint directory");
    for entry in std::fs::read_dir(source).expect("read Q4 checkpoint") {
        let entry = entry.expect("read Q4 checkpoint entry");
        let target = destination.join(entry.file_name());
        if entry
            .file_type()
            .expect("inspect Q4 checkpoint entry")
            .is_dir()
        {
            copy_tree_hard_link(&entry.path(), &target);
        } else if entry
            .file_type()
            .expect("inspect Q4 checkpoint entry")
            .is_file()
        {
            hard_link_or_copy(&entry.path(), &target);
        }
    }
}

fn hard_link_or_copy(source: &Path, destination: &Path) {
    match std::fs::symlink_metadata(destination) {
        Ok(metadata) if metadata.file_type().is_dir() => {
            std::fs::remove_dir_all(destination).expect("remove staged destination directory");
        }
        Ok(_) => std::fs::remove_file(destination).expect("remove staged destination file"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("inspect staged destination: {error}"),
    }
    if std::fs::hard_link(source, destination).is_err() {
        let mut input = std::fs::File::open(source).expect("open checkpoint source file");
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .expect("create staged checkpoint file");
        std::io::copy(&mut input, &mut output).expect("copy staged checkpoint file");
        std::fs::set_permissions(
            destination,
            std::fs::metadata(source)
                .expect("read checkpoint source metadata")
                .permissions(),
        )
        .expect("copy staged checkpoint permissions");
    }
}

pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

impl HttpResponse {
    pub fn error_code(&self) -> Option<String> {
        serde_json::from_str::<Value>(&self.body)
            .ok()?
            .get("error")?
            .get("code")?
            .as_str()
            .map(str::to_owned)
    }
}

pub struct Server {
    child: Child,
    port: u16,
    stderr: Arc<Mutex<String>>,
    binary: Binary,
    model_id: Option<String>,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    pub fn spawn(binary: Binary, model_dir: &Path, extra: &[String]) -> Self {
        let port = free_loopback_port();
        let model_id = (binary == Binary::Lattice).then(|| "characterization-model".to_owned());
        let mut command = binary.command();
        remove_server_startup_env(&mut command);
        match binary {
            Binary::Lattice => {
                command.args([
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    &port.to_string(),
                    "--model",
                ]);
            }
            #[cfg(all(feature = "metal-gpu", feature = "f16"))]
            Binary::LatticeServe => {
                command.args([
                    "--host",
                    "127.0.0.1",
                    "--port",
                    &port.to_string(),
                    "--model",
                ]);
            }
        }
        command.arg(model_dir).args(["--max-tokens", "2"]);
        if let Some(model_id) = &model_id {
            command.args(["--model-id", model_id]);
        }
        command.args(extra);
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|error| panic!("spawn {}: {error}", binary.label()));
        let mut server = Self {
            child,
            port,
            stderr: Arc::new(Mutex::new(String::new())),
            binary,
            model_id,
        };
        let stderr_pipe = server.child.stderr.take().expect("server stderr pipe");
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
        let started = Instant::now();
        loop {
            if matches!(server.try_request("GET", "/health", None), Ok(response) if response.status == 200)
            {
                break;
            }
            assert!(
                server
                    .child
                    .try_wait()
                    .unwrap_or_else(|error| panic!("poll {}: {error}", binary.label()))
                    .is_none(),
                "{} exited before health; stderr:\n{}",
                binary.label(),
                server.diagnostics()
            );
            assert!(
                started.elapsed() < HEALTH_TIMEOUT,
                "{} did not become healthy; stderr:\n{}",
                binary.label(),
                server.diagnostics()
            );
            std::thread::sleep(Duration::from_millis(200));
        }
        server.assert_address_announcement();
        server
    }

    fn assert_address_announcement(&mut self) {
        let expected = format!("127.0.0.1:{}", self.port);
        let started = Instant::now();
        loop {
            let diagnostics = self.diagnostics();
            if let Some(announced) = diagnostics
                .lines()
                .find_map(|line| announced_address(self.binary, line))
            {
                assert_eq!(
                    announced,
                    expected,
                    "{} announced a different address; stderr:\n{diagnostics}",
                    self.binary.label()
                );
                return;
            }
            assert!(
                self.child
                    .try_wait()
                    .unwrap_or_else(|error| panic!("poll {}: {error}", self.binary.label()))
                    .is_none(),
                "{} exited after health returned 200 without announcing its address; stderr:\n{diagnostics}",
                self.binary.label()
            );
            assert!(
                started.elapsed() < ADDRESS_ANNOUNCEMENT_TIMEOUT,
                "{} returned health 200 without announcing {expected}; stderr:\n{diagnostics}",
                self.binary.label()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn diagnostics(&self) -> String {
        self.stderr
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn chat_body(&self, extra: Value) -> Value {
        let mut body = json!({
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 2,
            "temperature": 0.0
        });
        if let Some(model_id) = &self.model_id {
            body["model"] = json!(model_id);
        }
        if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            body.extend(extra.clone());
        }
        body
    }

    pub fn request(&self, method: &str, path: &str, body: Option<&Value>) -> HttpResponse {
        self.try_request(method, path, body)
            .unwrap_or_else(|error| {
                panic!(
                    "{method} {path} transport failure against {}; {error}; stderr:\n{}",
                    self.binary.label(),
                    self.diagnostics()
                )
            })
    }

    fn try_request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> Result<HttpResponse, String> {
        exchange(self.port, method, path, body)
    }

    pub fn disconnect_stream_early(&self, body: &Value) -> u16 {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).expect("connect stream");
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .expect("set stream timeout");
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
        let mut reader = std::io::BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).expect("read streaming status");
        let status = line
            .split_whitespace()
            .nth(1)
            .and_then(|status| status.parse::<u16>().ok())
            .expect("stream response status");
        loop {
            line.clear();
            reader.read_line(&mut line).expect("read streaming headers");
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        let mut first_body_bytes = [0u8; 32];
        let _ = reader.read(&mut first_body_bytes);
        let _ = reader.get_ref().shutdown(Shutdown::Both);
        status
    }

    pub fn model_list_status(&self) -> u16 {
        self.request("GET", "/v1/models", None).status
    }

    pub fn embeddings(&self) -> HttpResponse {
        self.request("POST", "/v1/embeddings", Some(&json!({"input": "hello"})))
    }
}

fn announced_address(binary: Binary, line: &str) -> Option<String> {
    match binary {
        Binary::Lattice => line
            .strip_prefix("Listening on ")
            .map(|rest| rest.split_whitespace().next().unwrap_or("").to_owned()),
        #[cfg(all(feature = "metal-gpu", feature = "f16"))]
        Binary::LatticeServe => line
            .strip_prefix("[lattice_serve] OpenAI-compatible API on http://")
            .and_then(|rest| rest.strip_suffix("/v1"))
            .map(str::to_owned),
    }
}

pub fn request_at_port(port: u16, body: &Value) -> HttpResponse {
    exchange(port, "POST", "/v1/chat/completions", Some(body))
        .unwrap_or_else(|error| panic!("concurrent HTTP request failed: {error}"))
}

fn free_loopback_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("read ephemeral port")
        .port()
}

fn parse_http_response(raw: &[u8]) -> Result<HttpResponse, String> {
    let split = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "response has no header terminator".to_owned())?;
    let head = String::from_utf8_lossy(&raw[..split]);
    let status = head
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| "response has no status".to_owned())?
        .parse::<u16>()
        .map_err(|error| error.to_string())?;
    let raw_body = &raw[split + 4..];
    let body = if head
        .lines()
        .any(|line| line.eq_ignore_ascii_case("transfer-encoding: chunked"))
    {
        dechunk(raw_body)
    } else {
        raw_body.to_vec()
    };
    Ok(HttpResponse {
        status,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

fn exchange(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> Result<HttpResponse, String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(|error| error.to_string())?;
    stream
        .set_read_timeout(Some(REQUEST_TIMEOUT))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(REQUEST_TIMEOUT))
        .map_err(|error| error.to_string())?;
    let payload = body.map_or_else(Vec::new, |body| {
        serde_json::to_vec(body).expect("request body serializes")
    });
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|()| stream.write_all(&payload))
        .map_err(|error| error.to_string())?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|error| error.to_string())?;
    parse_http_response(&raw)
}

fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let Some(line_end) = raw.windows(2).position(|pair| pair == b"\r\n") else {
            return output;
        };
        let Ok(line) = std::str::from_utf8(&raw[..line_end]) else {
            return output;
        };
        let Some(size) = line
            .split(';')
            .next()
            .and_then(|size| usize::from_str_radix(size.trim(), 16).ok())
        else {
            return output;
        };
        raw = &raw[line_end + 2..];
        if size == 0 || raw.len() < size + 2 {
            return output;
        }
        output.extend_from_slice(&raw[..size]);
        raw = &raw[size + 2..];
    }
}

pub fn error_code(response: &HttpResponse) -> Option<String> {
    response.error_code()
}

pub fn assert_http_golden(
    response: &HttpResponse,
    expected_status: u16,
    expected_code: Option<&str>,
    context: &str,
) {
    assert_eq!(
        (response.status, response.error_code()),
        (expected_status, expected_code.map(str::to_owned)),
        "{context}: {}",
        response.body.chars().take(320).collect::<String>()
    );
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn acquire_gpu_lock() -> impl Sized {
    lattice_inference::measurement::gpu_test_lock()
}

pub fn response_text(response: &HttpResponse) -> String {
    serde_json::from_str::<Value>(&response.body)
        .ok()
        .and_then(|value| {
            value["choices"][0]["message"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| format!("<missing completion text: {}>", response.body))
}
