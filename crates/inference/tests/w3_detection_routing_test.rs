//! Integration tests for W3-artifact detection/routing in `bench_decode_ab`
//! and `chat_metal` (round 2, issue #420/#531/MAJ-2).
//!
//! Both binaries classify a model directory by which weight-file extension
//! is present (no `model.safetensors` + a `.w3` file => W3; no
//! `model.safetensors` + a `.q4` file => Q4; else safetensors) and dispatch
//! accordingly. A W3 MLP directory also ships `.q4`/`.f16` files for the
//! non-MLP tensors, so the W3 check MUST run before the Q4 check or a real
//! W3 directory would be misclassified as Q4. These tests drive the actual
//! built binaries (`CARGO_BIN_EXE_*`) against synthetic fixture directories
//! — no multi-hundred-MB real model weights are required, since detection
//! is a filesystem check that happens before any tensor is read, and the
//! subsequent (deliberately incomplete) load failure is itself part of what
//! we assert: detection must route to the right loader, and that loader
//! must fail closed rather than silently falling back.
//!
//! Both binaries have `required-features = ["f16", "metal-gpu"]` in
//! Cargo.toml, so this whole file is gated the same way — it simply won't
//! compile/run without those features, matching how these binaries are
//! normally tested.
//!
//! Run: cargo test --features f16,metal-gpu --test w3_detection_routing_test

#![cfg(all(target_os = "macos", feature = "f16", feature = "metal-gpu"))]

use std::path::PathBuf;
use std::process::{Command, Output};

fn tokenizer_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizers/qwen3-embedding-0.6b")
}

fn make_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("w3_detection_routing_test_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn assert_no_panic(out: &Output, label: &str) {
    let err = stderr(out);
    assert!(
        !err.contains("panicked at"),
        "{label} must fail closed via a clean error, not a panic; stderr:\n{err}"
    );
}

// ---------------------------------------------------------------------------
// bench_decode_ab
// ---------------------------------------------------------------------------

fn run_bench_decode_ab(model_dir: &std::path::Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bench_decode_ab"))
        .env("LATTICE_MODEL_DIR", model_dir)
        .env("LATTICE_TOKENIZER_DIR", tokenizer_dir())
        .env("BENCH_N", "8")
        .env("BENCH_RUNS", "1")
        .output()
        .expect("failed to spawn bench_decode_ab")
}

#[test]
fn bench_decode_ab_detects_w3_dir_and_routes_to_w3_loader() {
    let dir = make_dir("bench_w3_only");
    std::fs::write(dir.join("some_mlp_layer.w3"), b"not a real w3 tensor").unwrap();

    let out = run_bench_decode_ab(&dir);
    let err = stderr(&out);
    assert!(
        err.contains("(W3)"),
        "must log W3 detection before attempting to load; stderr:\n{err}"
    );
    assert!(
        !err.contains("(Q4)") && !err.contains("(safetensors)"),
        "must not also claim another format; stderr:\n{err}"
    );
    assert!(
        !out.status.success(),
        "an incomplete/fake W3 dir must fail closed, not silently succeed"
    );
    assert_no_panic(&out, "bench_decode_ab on W3-only dir");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bench_decode_ab_detects_q4_dir_when_no_w3_files_present() {
    // Regression: adding W3 detection must not break the existing Q4 path.
    let dir = make_dir("bench_q4_only");
    std::fs::write(dir.join("some_layer.q4"), b"not a real q4 tensor").unwrap();

    let out = run_bench_decode_ab(&dir);
    let err = stderr(&out);
    assert!(
        err.contains("(Q4)"),
        "must log Q4 detection for a .q4-only dir; stderr:\n{err}"
    );
    assert!(
        !err.contains("(W3)"),
        "must not misclassify a Q4-only dir as W3; stderr:\n{err}"
    );
    assert!(!out.status.success());
    assert_no_panic(&out, "bench_decode_ab on Q4-only dir");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bench_decode_ab_prefers_w3_over_q4_when_both_file_types_present() {
    // The real scenario the round-1 critic flagged: a W3 MLP directory also
    // ships .q4/.f16 files for the non-MLP tensors (attention, GDN, embed,
    // lm_head, norms). W3 detection must run first or this dir would be
    // misclassified as plain Q4 and silently lose the W3 MLP tensors.
    let dir = make_dir("bench_w3_and_q4_superset");
    std::fs::write(dir.join("mlp_gate.w3"), b"fake w3 tensor").unwrap();
    std::fs::write(dir.join("attn_qkv.q4"), b"fake q4 tensor").unwrap();

    let out = run_bench_decode_ab(&dir);
    let err = stderr(&out);
    assert!(
        err.contains("(W3)"),
        "a dir containing both .w3 and .q4 files must be detected as W3; stderr:\n{err}"
    );
    assert!(!out.status.success());
    assert_no_panic(&out, "bench_decode_ab on W3+Q4 superset dir");

    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// chat_metal
// ---------------------------------------------------------------------------

fn run_chat_metal(model_dir: &std::path::Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_chat_metal"))
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--tokenizer-dir")
        .arg(tokenizer_dir())
        .arg("--max-tokens")
        .arg("1")
        .output()
        .expect("failed to spawn chat_metal")
}

#[test]
fn chat_metal_detects_w3_dir_and_routes_to_w3_loader() {
    let dir = make_dir("chat_w3_only");
    std::fs::write(dir.join("some_mlp_layer.w3"), b"not a real w3 tensor").unwrap();

    let out = run_chat_metal(&dir);
    let err = stderr(&out);
    assert!(
        err.contains("Detected W3 MLP model directory"),
        "must log W3 detection before attempting to load; stderr:\n{err}"
    );
    assert!(
        !err.contains("Detected Q4 model directory"),
        "must not also claim Q4 detection; stderr:\n{err}"
    );
    assert!(
        !out.status.success(),
        "an incomplete/fake W3 dir must fail closed, not silently succeed"
    );
    assert_no_panic(&out, "chat_metal on W3-only dir");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn chat_metal_detects_q4_dir_when_no_w3_files_present() {
    let dir = make_dir("chat_q4_only");
    std::fs::write(dir.join("some_layer.q4"), b"not a real q4 tensor").unwrap();

    let out = run_chat_metal(&dir);
    let err = stderr(&out);
    assert!(
        err.contains("Detected Q4 model directory"),
        "must log Q4 detection for a .q4-only dir; stderr:\n{err}"
    );
    assert!(
        !err.contains("Detected W3 MLP model directory"),
        "must not misclassify a Q4-only dir as W3; stderr:\n{err}"
    );
    assert!(!out.status.success());
    assert_no_panic(&out, "chat_metal on Q4-only dir");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn chat_metal_prefers_w3_over_q4_when_both_file_types_present() {
    let dir = make_dir("chat_w3_and_q4_superset");
    std::fs::write(dir.join("mlp_gate.w3"), b"fake w3 tensor").unwrap();
    std::fs::write(dir.join("attn_qkv.q4"), b"fake q4 tensor").unwrap();

    let out = run_chat_metal(&dir);
    let err = stderr(&out);
    assert!(
        err.contains("Detected W3 MLP model directory"),
        "a dir containing both .w3 and .q4 files must be detected as W3; stderr:\n{err}"
    );
    assert!(!out.status.success());
    assert_no_panic(&out, "chat_metal on W3+Q4 superset dir");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn chat_metal_no_model_found_error_mentions_w3_files() {
    // The "no model found" error message was updated to mention .w3 files
    // alongside .q4/model.safetensors — verify a completely empty dir
    // (no safetensors, no .q4, no .w3) still produces that specific message.
    let dir = make_dir("chat_empty_no_format");

    let out = run_chat_metal(&dir);
    let err = stderr(&out);
    assert!(!out.status.success());
    assert!(
        err.contains("expected model.safetensors, .q4, or .w3 files"),
        "no-model-found error must mention .w3 as a recognized format; stderr:\n{err}"
    );
    assert_no_panic(&out, "chat_metal on empty dir");

    std::fs::remove_dir_all(&dir).ok();
}
