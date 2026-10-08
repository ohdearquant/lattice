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
