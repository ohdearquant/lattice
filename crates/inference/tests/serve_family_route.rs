//! `lattice serve` family and backend routing, driven through the shipped
//! binary with no checkpoint.
//!
//! Each case starts the real `lattice serve` process on a directory that holds
//! a `config.json` and a stub weight file, so the process reaches its routing
//! decision and then its loader. What is asserted is the route the process
//! reports and the loader it reaches. The stub weights are never loadable, so
//! every case ends in a startup error and none of them listens.
//!
//! A real checkpoint pair (Gemma 4 E2B and Qwen3.5 over HTTP, streaming and
//! not) is measured by the `bench_serve_http` example, which reports `SKIP`
//! when the checkpoint directory is absent.

#![cfg(feature = "serve")]

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const STARTUP_ERROR_DEADLINE: Duration = Duration::from_secs(120);

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

/// Run `lattice serve` on `dir` and wait for it to exit. A server that starts
/// listening instead is killed at the deadline and fails the test.
fn serve(dir: &Path) -> Run {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lattice"))
        .args(["serve", "--host", "127.0.0.1", "--port", "0", "--model"])
        .arg(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lattice serve");
    let started = Instant::now();
    loop {
        if child.try_wait().expect("poll lattice serve").is_some() {
            break;
        }
        if started.elapsed() > STARTUP_ERROR_DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            panic!("lattice serve kept running on a directory that cannot load");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let output = child
        .wait_with_output()
        .expect("collect lattice serve output");
    let run = Run {
        code: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    };
    // Shown under `--nocapture`, so the process's own words are on the record.
    eprintln!(
        "lattice serve exit={:?} stderr:\n{}",
        run.code,
        run.stderr.trim_end()
    );
    run
}

#[test]
fn gemma_safetensors_directory_selects_the_gemma_cpu_route() {
    let dir = checkpoint_dir(&gemma_config(), "model.safetensors");
    let run = serve(dir.path());
    assert!(
        run.stderr
            .contains("[route] selected family=gemma4 backend=cpu format=safetensors"),
        "{}",
        run.stderr
    );
    assert!(
        run.stderr.contains("Error: failed to load Gemma 4 model:"),
        "the Gemma loader is the one that ran: {}",
        run.stderr
    );
    assert!(!run.stderr.contains("Listening on"), "{}", run.stderr);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn qwen_safetensors_directory_keeps_the_qwen_cpu_route() {
    let dir = checkpoint_dir(&qwen_config(), "model.safetensors");
    let run = serve(dir.path());
    assert!(
        run.stderr
            .contains("[route] selected family=qwen35 backend=cpu format=safetensors"),
        "{}",
        run.stderr
    );
    assert!(
        run.stderr.contains("Error: failed to load model:"),
        "the Qwen loader is the one that ran: {}",
        run.stderr
    );
    assert!(!run.stderr.contains("Gemma"), "{}", run.stderr);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn a_directory_with_no_config_is_still_routed_to_qwen() {
    let dir = tempfile::tempdir().expect("temp model dir");
    std::fs::write(dir.path().join("model.safetensors"), b"stub").expect("write stub weights");
    let run = serve(dir.path());
    assert!(
        run.stderr
            .contains("[route] selected family=qwen35 backend=cpu format=safetensors"),
        "{}",
        run.stderr
    );
    assert!(
        run.stderr.contains("Error: failed to load model:"),
        "{}",
        run.stderr
    );
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn gemma_in_the_metal_format_is_refused_before_any_loader_runs() {
    let dir = checkpoint_dir(&gemma_config(), "model_layers_0_weight.q4");
    let run = serve(dir.path());
    assert!(
        run.stderr.contains("Error: gemma_metal_unsupported: "),
        "{}",
        run.stderr
    );
    assert!(
        !run.stderr.contains("[route] selected"),
        "a refused checkpoint selects no route: {}",
        run.stderr
    );
    assert!(
        !run.stderr.contains("failed to load"),
        "no loader ran: {}",
        run.stderr
    );
    assert!(!run.stderr.contains("Listening on"), "{}", run.stderr);
    assert_eq!(run.code, Some(1), "{}", run.stderr);
}

#[test]
fn qwen_in_the_metal_format_is_not_caught_by_the_gemma_refusal() {
    let dir = checkpoint_dir(&qwen_config(), "model_layers_0_weight.q4");
    let run = serve(dir.path());
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

#[cfg(all(target_os = "macos", feature = "metal-gpu", feature = "f16"))]
mod live_routes {
    use serde_json::{Value, json};
    use std::io::{Read as _, Write as _};
    use std::net::{TcpStream, ToSocketAddrs};
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::{Arc, Mutex, PoisonError};
    use std::time::{Duration, Instant};

    const ENFORCE_ENV: &str = "LATTICE_SERVE_FAMILY_ROUTE_ENFORCE";
    const HEALTH_TIMEOUT: Duration = Duration::from_secs(300);

    fn checkpoint(env_name: &str, default_name: &str) -> Option<PathBuf> {
        let dir = match std::env::var_os(env_name) {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from(std::env::var_os("HOME").expect("HOME is set"))
                .join(".lattice")
                .join("models")
                .join(default_name),
        };
        if dir.join("config.json").is_file() {
            Some(dir)
        } else {
            assert_ne!(
                std::env::var(ENFORCE_ENV).as_deref(),
                Ok("1"),
                "no checkpoint at {} ({env_name}) while {ENFORCE_ENV}=1",
                dir.display()
            );
            eprintln!(
                "LATTICE_SERVE_FAMILY_ROUTE_SKIPPED reason=no_checkpoint path={} ({env_name})",
                dir.display()
            );
            None
        }
    }

    fn q4_checkpoint_with_tokenizer(q4_dir: &Path, tokenizer_dir: &Path) -> tempfile::TempDir {
        let tokenizer = tokenizer_dir.join("tokenizer.json");
        assert!(
            tokenizer.is_file(),
            "missing tokenizer at {}",
            tokenizer.display()
        );
        let staged = tempfile::tempdir().expect("temporary Q4 checkpoint directory");
        for entry in std::fs::read_dir(q4_dir).expect("read Q4 checkpoint directory") {
            let entry = entry.expect("read Q4 checkpoint entry");
            let source = entry.path();
            let destination = staged.path().join(entry.file_name());
            assert!(entry.file_type().expect("Q4 entry type").is_file());
            if std::fs::hard_link(&source, &destination).is_err() {
                std::fs::copy(&source, &destination).expect("copy Q4 checkpoint entry");
            }
        }
        std::os::unix::fs::symlink(tokenizer, staged.path().join("tokenizer.json"))
            .expect("link the matching Qwen tokenizer into the Q4 checkpoint");
        staged
    }

    fn free_loopback_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("ephemeral port binds")
            .local_addr()
            .expect("listener has an address")
            .port()
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

    impl Server {
        fn spawn(model: &Path) -> Self {
            let port = free_loopback_port();
            let mut child = Command::new(env!("CARGO_BIN_EXE_lattice"))
                .args(["serve", "--host", "127.0.0.1", "--port"])
                .arg(port.to_string())
                .arg("--model")
                .arg(model)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .expect("lattice serve starts");
            let stderr = Arc::new(Mutex::new(String::new()));
            let stderr_pipe = child.stderr.take().expect("stderr is piped");
            let captured = Arc::clone(&stderr);
            std::thread::spawn(move || {
                let mut reader = std::io::BufReader::new(stderr_pipe);
                let mut line = String::new();
                loop {
                    line.clear();
                    match std::io::BufRead::read_line(&mut reader, &mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => captured
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .push_str(&line),
                    }
                }
            });
            let mut server = Self {
                child,
                port,
                stderr,
            };
            let started = Instant::now();
            loop {
                if matches!(server.try_request("GET", "/health", None), Some((200, _))) {
                    break;
                }
                assert!(
                    server
                        .child
                        .try_wait()
                        .expect("poll lattice serve")
                        .is_none(),
                    "lattice serve exited before becoming healthy:\n{}",
                    server.diagnostics()
                );
                assert!(
                    started.elapsed() < HEALTH_TIMEOUT,
                    "lattice serve did not become healthy:\n{}",
                    server.diagnostics()
                );
                std::thread::sleep(Duration::from_millis(200));
            }
            server
        }

        fn diagnostics(&self) -> String {
            self.stderr
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }

        fn try_request(
            &self,
            method: &str,
            path: &str,
            body: Option<&Value>,
        ) -> Option<(u16, String)> {
            let address = ("127.0.0.1", self.port).to_socket_addrs().ok()?.next()?;
            let mut stream = TcpStream::connect(address).ok()?;
            stream
                .set_read_timeout(Some(Duration::from_secs(900)))
                .ok()?;
            let payload = body.map_or_else(Vec::new, |value| {
                serde_json::to_vec(value).expect("request body serializes")
            });
            let header = format!(
                "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                payload.len()
            );
            stream.write_all(header.as_bytes()).ok()?;
            stream.write_all(&payload).ok()?;
            let mut raw = Vec::new();
            stream.read_to_end(&mut raw).ok()?;
            let split = raw.windows(4).position(|window| window == b"\r\n\r\n")?;
            let response_header = String::from_utf8_lossy(&raw[..split]);
            let status = response_header
                .split_whitespace()
                .nth(1)?
                .parse::<u16>()
                .ok()?;
            let body = &raw[split + 4..];
            let body = if response_header
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
            self.try_request(method, path, body)
                .unwrap_or_else(|| panic!("{method} {path} failed:\n{}", self.diagnostics()))
        }

        fn marker(&self, family: &str, backend: &str, mode: &str) -> String {
            let prefix = format!(
                "[route] served family={family} backend={backend} mode={mode} driver=shared opened="
            );
            let started = Instant::now();
            loop {
                let stderr = self.diagnostics();
                if let Some(line) = stderr.lines().find(|line| line.starts_with(&prefix)) {
                    let fields = line.split_whitespace().collect::<Vec<_>>();
                    let opened = fields
                        .iter()
                        .find_map(|field| field.strip_prefix("opened="))
                        .and_then(|value| value.parse::<usize>().ok());
                    let consumed = fields
                        .iter()
                        .find_map(|field| field.strip_prefix("consumed="))
                        .and_then(|value| value.parse::<usize>().ok());
                    assert!(opened.is_some_and(|count| count > 0), "{line}");
                    assert!(consumed.is_some_and(|count| count > 0), "{line}");
                    eprintln!("R13-MARKER {line}");
                    return line.to_string();
                }
                assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "missing {mode} shared-route marker:\n{stderr}"
                );
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    fn dechunk(mut raw: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        loop {
            let Some(end) = raw.windows(2).position(|pair| pair == b"\r\n") else {
                return body;
            };
            let size = std::str::from_utf8(&raw[..end])
                .ok()
                .and_then(|value| usize::from_str_radix(value.split(';').next()?.trim(), 16).ok());
            let Some(size) = size else { return body };
            raw = &raw[end + 2..];
            if size == 0 || raw.len() < size + 2 {
                return body;
            }
            body.extend_from_slice(&raw[..size]);
            raw = &raw[size + 2..];
        }
    }

    fn assert_chat_routes(server: &Server, family: &str, backend: &str, model_name: &str) {
        for (streaming, mode) in [(false, "nonstream"), (true, "stream")] {
            let body = json!({
                "model": model_name,
                "messages": [{"role": "user", "content": "Name three primary colors."}],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": streaming,
            });
            let (status, response) = server.request("POST", "/v1/chat/completions", Some(&body));
            assert_eq!(status, 200, "{mode}: {response}\n{}", server.diagnostics());
            if streaming {
                assert!(response.contains("data: [DONE]"), "{response}");
                assert!(
                    response
                        .lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                        .any(|chunk| chunk["choices"][0]["delta"]["content"]
                            .as_str()
                            .is_some_and(|text| !text.is_empty())),
                    "stream has a nonempty answer delta: {response}"
                );
            } else {
                let value: Value = serde_json::from_str(&response).expect("response is JSON");
                assert!(
                    value["choices"][0]["message"]["content"]
                        .as_str()
                        .is_some_and(|text| !text.is_empty()),
                    "{response}"
                );
            }
            server.marker(family, backend, mode);
        }
    }

    fn serve_checkpoint(
        env_name: &str,
        default_name: &str,
        family: &str,
        backend: &str,
        tokenizer_fallback: Option<(&str, &str)>,
    ) {
        let Some(dir) = checkpoint(env_name, default_name) else {
            return;
        };
        let staged = if dir.join("tokenizer.json").is_file() {
            None
        } else if let Some((tokenizer_env, tokenizer_default)) = tokenizer_fallback {
            let Some(tokenizer_dir) = checkpoint(tokenizer_env, tokenizer_default) else {
                return;
            };
            Some(q4_checkpoint_with_tokenizer(&dir, &tokenizer_dir))
        } else {
            None
        };
        let model_dir = staged.as_ref().map_or(dir.as_path(), |temp| temp.path());
        let server = Server::spawn(model_dir);
        assert!(
            server.diagnostics().contains(&format!(
                "[route] selected family={family} backend={backend}"
            )),
            "{}",
            server.diagnostics()
        );
        let model_name = model_dir
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .expect("checkpoint directory has a model name");
        assert_chat_routes(&server, family, backend, model_name);
    }

    #[test]
    fn lattice_serve_qwen_cpu_uses_shared_route() {
        serve_checkpoint(
            "LATTICE_SERVE_FAMILY_ROUTE_QWEN_CPU_DIR",
            "qwen3.5-0.8b",
            "qwen35",
            "cpu",
            None,
        );
    }

    #[test]
    fn lattice_serve_qwen_metal_uses_shared_route() {
        let _gpu = lattice_inference::measurement::gpu_test_lock();
        serve_checkpoint(
            "LATTICE_SERVE_FAMILY_ROUTE_QWEN_METAL_DIR",
            "qwen3.5-0.8b-q4",
            "qwen35",
            "metal",
            Some(("LATTICE_SERVE_FAMILY_ROUTE_QWEN_CPU_DIR", "qwen3.5-0.8b")),
        );
    }

    #[test]
    fn lattice_serve_gemma_cpu_uses_shared_route() {
        serve_checkpoint(
            "LATTICE_SERVE_FAMILY_ROUTE_GEMMA_CPU_DIR",
            "gemma-4-e2b-it",
            "gemma4",
            "cpu",
            None,
        );
    }
}
