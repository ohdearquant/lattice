//! CLI-surface integration tests for `eval_perplexity --w3-mlp-dir` (round 2,
//! issue #420/#531/MAJ-1).
//!
//! These drive the actual built binary via `CARGO_BIN_EXE_eval_perplexity` so
//! they exercise the real `std::env::args()` parsing and process exit code,
//! not just an inlined copy of the argument logic. They deliberately avoid
//! `--features metal-gpu` fixtures for real W3 weight loading (that requires
//! a multi-hundred-MB local model cache under `~/.lattice/models`, which is
//! not reproducible on CI) — the mutual-exclusion and missing/corrupt-dir
//! fail-closed checks below all happen before any Metal device is touched,
//! so they run deterministically on any machine, with or without the
//! `metal-gpu` feature.
//!
//! Run: cargo test --test eval_perplexity_w3_cli_test

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_eval_perplexity"))
}

fn corpus_file() -> PathBuf {
    // Content is irrelevant to the tests below since every case is expected
    // to fail before tokenization/scoring. Each call gets its own uniquely
    // named file (via `tempfile`) since tests in this binary run
    // concurrently in the same process — a PID-based shared filename would
    // race across tests.
    let file = tempfile::Builder::new()
        .prefix("eval_perplexity_w3_cli_test_corpus_")
        .suffix(".txt")
        .tempfile()
        .expect("create corpus fixture");
    let path = file.path().to_path_buf();
    std::fs::write(&path, "hello world, this is a tiny corpus.\n").expect("write corpus fixture");
    file.keep().expect("persist corpus fixture").1
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("failed to spawn eval_perplexity")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Directory containing a real, valid, checked-in `tokenizer.json` fixture
/// (content/vocab mismatch with the model is irrelevant — `eval_perplexity`
/// loads the tokenizer before it ever inspects the model-mode dir, so any
/// well-formed tokenizer.json lets these tests reach the check they target).
fn valid_tokenizer_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizers/qwen3-embedding-0.6b")
}

fn assert_no_panic(out: &Output) {
    let err = stderr(out);
    assert!(
        !err.contains("panicked at"),
        "process must fail closed via a clean error, not a panic; stderr:\n{err}"
    );
}

// ---------------------------------------------------------------------------
// Mutual exclusivity: --w3-mlp-dir vs the other three mode flags.
// ---------------------------------------------------------------------------

#[test]
fn w3_mlp_dir_rejects_combination_with_q4_dir() {
    let corpus = corpus_file();
    let out = run(&[
        "--w3-mlp-dir",
        "/tmp/does-not-need-to-exist-w3",
        "--q4-dir",
        "/tmp/does-not-need-to-exist-q4",
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "must exit non-zero");
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        err.contains("--w3-mlp-dir is mutually exclusive with --q4-dir / --quarot-q4-dir"),
        "stderr must name the conflicting flags; got:\n{err}"
    );
    assert_no_panic(&out);
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn w3_mlp_dir_rejects_combination_with_quarot_q4_dir() {
    let corpus = corpus_file();
    let out = run(&[
        "--w3-mlp-dir",
        "/tmp/does-not-need-to-exist-w3",
        "--quarot-q4-dir",
        "/tmp/does-not-need-to-exist-quarot",
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        err.contains("--w3-mlp-dir is mutually exclusive with --q4-dir / --quarot-q4-dir"),
        "stderr must name the conflicting flags; got:\n{err}"
    );
    assert_no_panic(&out);
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn w3_mlp_dir_rejects_combination_with_both_q4_and_quarot_dirs_at_once() {
    let corpus = corpus_file();
    let out = run(&[
        "--w3-mlp-dir",
        "/tmp/does-not-need-to-exist-w3",
        "--q4-dir",
        "/tmp/does-not-need-to-exist-q4",
        "--quarot-q4-dir",
        "/tmp/does-not-need-to-exist-quarot",
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    assert_no_panic(&out);
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn w3_mlp_dir_rejects_combination_with_model_dir() {
    let corpus = corpus_file();
    let out = run(&[
        "--w3-mlp-dir",
        "/tmp/does-not-need-to-exist-w3",
        "--model-dir",
        "/tmp/does-not-need-to-exist-model",
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        err.contains(
            "--model-dir is mutually exclusive with --q4-dir / --quarot-q4-dir / --w3-mlp-dir"
        ),
        "stderr must name the conflicting flags; got:\n{err}"
    );
    assert_no_panic(&out);
    let _ = std::fs::remove_file(&corpus);
}

// ---------------------------------------------------------------------------
// --w3-mlp-dir alone must be *accepted* by the mode-exclusivity gate (i.e. no
// mutual-exclusion error) and must require --tokenizer-dir like the other
// two Metal modes.
// ---------------------------------------------------------------------------

#[test]
fn w3_mlp_dir_alone_requires_tokenizer_dir() {
    let corpus = corpus_file();
    let out = run(&[
        "--w3-mlp-dir",
        "/tmp/does-not-need-to-exist-w3",
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        !err.contains("mutually exclusive"),
        "single --w3-mlp-dir must not trip the mutual-exclusivity check; got:\n{err}"
    );
    assert!(
        err.contains(
            "--tokenizer-dir is required when using --q4-dir / --quarot-q4-dir / --w3-mlp-dir"
        ),
        "stderr must demand --tokenizer-dir; got:\n{err}"
    );
    assert_no_panic(&out);
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn w3_mlp_dir_without_any_mode_flag_is_not_required_alone() {
    // Sanity check on the updated "one of ..." usage message wording — it
    // must now mention --w3-mlp-dir as a valid third option.
    let corpus = corpus_file();
    let out = run(&["--corpus-file", corpus.to_str().unwrap()]);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(
        err.contains("one of --model-dir, --q4-dir, --quarot-q4-dir, or --w3-mlp-dir is required"),
        "usage error must list all four flags including --w3-mlp-dir; got:\n{err}"
    );
    let _ = std::fs::remove_file(&corpus);
}

// ---------------------------------------------------------------------------
// Fail-closed on missing/corrupt W3 artifacts.
// ---------------------------------------------------------------------------

#[test]
fn w3_mlp_dir_fails_closed_on_nonexistent_directory() {
    let corpus = corpus_file();
    let out = run(&[
        "--w3-mlp-dir",
        "/tmp/eval-perplexity-w3-cli-test-nonexistent-dir-xyz",
        "--tokenizer-dir",
        valid_tokenizer_dir().to_str().unwrap(),
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "a nonexistent --w3-mlp-dir must be a hard failure, never a silent report"
    );
    assert_eq!(out.status.code(), Some(1));
    let out_text = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out_text.contains("=== Perplexity Report"),
        "must never print a report for a nonexistent artifact dir"
    );
    assert_no_panic(&out);
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn decode_loop_ppl_w3_mlp_dir_fails_closed_on_nonexistent_directory() {
    // Round-3 (issue #420/#530/#531): --decode-loop-ppl must thread through
    // exactly like the default scorer for artifact-loading failures — it
    // only changes how a *loaded* session is scored, not the fail-closed
    // load-failure path.
    let corpus = corpus_file();
    let out = run(&[
        "--decode-loop-ppl",
        "--w3-mlp-dir",
        "/tmp/eval-perplexity-w3-cli-test-nonexistent-dir-decode-loop",
        "--tokenizer-dir",
        valid_tokenizer_dir().to_str().unwrap(),
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(
        !out.status.success(),
        "a nonexistent --w3-mlp-dir must be a hard failure under --decode-loop-ppl too"
    );
    assert_eq!(out.status.code(), Some(1));
    let out_text = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out_text.contains("=== Perplexity Report"),
        "must never print a report for a nonexistent artifact dir"
    );
    assert_no_panic(&out);
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn w3_mlp_dir_fails_closed_on_directory_missing_config_json() {
    // A directory that exists but is missing config.json is what a
    // corrupted/incomplete `quantize_w3_mlp` output looks like: present on
    // disk, unusable. Must fail closed, not panic, not fall back silently.
    let corpus = corpus_file();
    let empty_w3_dir = std::env::temp_dir().join(format!(
        "eval_perplexity_w3_cli_test_empty_dir_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&empty_w3_dir).unwrap();

    let out = run(&[
        "--w3-mlp-dir",
        empty_w3_dir.to_str().unwrap(),
        "--tokenizer-dir",
        valid_tokenizer_dir().to_str().unwrap(),
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        err.contains("config.json"),
        "error must name the missing artifact (config.json); got:\n{err}"
    );
    let out_text = String::from_utf8_lossy(&out.stdout);
    assert!(!out_text.contains("=== Perplexity Report"));
    assert_no_panic(&out);

    std::fs::remove_dir_all(&empty_w3_dir).ok();
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn decode_loop_ppl_w3_mlp_dir_fails_closed_on_directory_missing_config_json() {
    // Same corrupted-artifact shape as `w3_mlp_dir_fails_closed_on_directory_missing_config_json`
    // (a directory that exists but is missing config.json), with `--decode-loop-ppl` added.
    // Artifact loading happens before the scoring-mode branch, so this must fail closed
    // identically to the default scorer — not panic, not silently skip validation because
    // decode-loop mode was requested.
    let corpus = corpus_file();
    let empty_w3_dir = std::env::temp_dir().join(format!(
        "eval_perplexity_w3_cli_test_empty_dir_decode_loop_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&empty_w3_dir).unwrap();

    let out = run(&[
        "--decode-loop-ppl",
        "--w3-mlp-dir",
        empty_w3_dir.to_str().unwrap(),
        "--tokenizer-dir",
        valid_tokenizer_dir().to_str().unwrap(),
        "--corpus-file",
        corpus.to_str().unwrap(),
    ]);
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        err.contains("config.json"),
        "error must name the missing artifact (config.json); got:\n{err}"
    );
    let out_text = String::from_utf8_lossy(&out.stdout);
    assert!(!out_text.contains("=== Perplexity Report"));
    assert_no_panic(&out);

    std::fs::remove_dir_all(&empty_w3_dir).ok();
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn w3_mlp_dir_fails_closed_on_corrupt_w3_artifact_if_real_model_cache_present() {
    // Best-effort test against the real converter output the round-1/round-2
    // work order references: `~/.lattice/models/qwen3.5-0.8b-w3-mlp-420`
    // (W3 dir) + `~/.lattice/models/qwen3.5-0.8b` (tokenizer). Skips cleanly
    // if the local model cache isn't present (e.g. CI, a fresh checkout) —
    // mirrors the existing `Device::system_default()`-absent skip idiom used
    // elsewhere in this crate for hardware/fixture-dependent tests, so this
    // never produces a false failure on a machine without the cache.
    let Ok(home) = std::env::var("HOME") else {
        return;
    };
    let real_w3_dir = PathBuf::from(&home).join(".lattice/models/qwen3.5-0.8b-w3-mlp-420");
    let real_tokenizer_dir = PathBuf::from(&home).join(".lattice/models/qwen3.5-0.8b");
    if !real_w3_dir.is_dir() || !real_tokenizer_dir.join("tokenizer.json").is_file() {
        eprintln!(
            "skipping: local W3 model cache not present at {}",
            real_w3_dir.display()
        );
        return;
    }

    // Build a lightweight fixture: symlink every file from the real W3 dir
    // (avoids copying ~600MB), except corrupt exactly one dense-MLP .w3
    // tensor by truncating it to a handful of garbage bytes.
    let fixture_dir = std::env::temp_dir().join(format!(
        "eval_perplexity_w3_cli_test_corrupt_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&fixture_dir).unwrap();
    let mut corrupted_one = false;
    for entry in std::fs::read_dir(&real_w3_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let dest = fixture_dir.join(&name);
        if !corrupted_one && name_str.ends_with(".w3") && name_str.contains("mlp") {
            // Truncated to 4 bytes: shorter than the safetensors-style header
            // length prefix — must be rejected during loading, not panic.
            std::fs::write(&dest, [0xDEu8, 0xAD, 0xBE, 0xEF]).unwrap();
            corrupted_one = true;
        } else {
            std::os::unix::fs::symlink(entry.path(), &dest).unwrap();
        }
    }
    assert!(
        corrupted_one,
        "test setup bug: no dense-MLP .w3 file found to corrupt in {}",
        real_w3_dir.display()
    );

    let corpus = corpus_file();
    let out = run(&[
        "--w3-mlp-dir",
        fixture_dir.to_str().unwrap(),
        "--tokenizer-dir",
        real_tokenizer_dir.to_str().unwrap(),
        "--corpus-file",
        corpus.to_str().unwrap(),
        "--max-tokens",
        "8",
    ]);

    assert!(
        !out.status.success(),
        "a corrupted W3 tensor must be a hard failure, never a silent/degraded report"
    );
    let out_text = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out_text.contains("=== Perplexity Report"),
        "must never print a report when a required W3 artifact is corrupt"
    );
    assert_no_panic(&out);

    std::fs::remove_dir_all(&fixture_dir).ok();
    let _ = std::fs::remove_file(&corpus);
}

#[test]
fn decode_loop_ppl_fails_closed_on_corrupt_w3_artifact_if_real_model_cache_present() {
    // Same corrupted-tensor fixture as
    // `w3_mlp_dir_fails_closed_on_corrupt_w3_artifact_if_real_model_cache_present`,
    // with `--decode-loop-ppl` added. Round-2 known risk (issue #420/#531): the
    // default scorer panics mid-scoring on a W3 session; `--decode-loop-ppl`
    // must not silently paper over a corrupt artifact either — loading still
    // happens before the scoring-mode branch is reached, so this must fail
    // closed identically. Skips cleanly if the local model cache isn't present.
    let Ok(home) = std::env::var("HOME") else {
        return;
    };
    let real_w3_dir = PathBuf::from(&home).join(".lattice/models/qwen3.5-0.8b-w3-mlp-420");
    let real_tokenizer_dir = PathBuf::from(&home).join(".lattice/models/qwen3.5-0.8b");
    if !real_w3_dir.is_dir() || !real_tokenizer_dir.join("tokenizer.json").is_file() {
        eprintln!(
            "skipping: local W3 model cache not present at {}",
            real_w3_dir.display()
        );
        return;
    }

    let fixture_dir = std::env::temp_dir().join(format!(
        "eval_perplexity_w3_cli_test_corrupt_fixture_decode_loop_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&fixture_dir).unwrap();
    let mut corrupted_one = false;
    for entry in std::fs::read_dir(&real_w3_dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let dest = fixture_dir.join(&name);
        if !corrupted_one && name_str.ends_with(".w3") && name_str.contains("mlp") {
            std::fs::write(&dest, [0xDEu8, 0xAD, 0xBE, 0xEF]).unwrap();
            corrupted_one = true;
        } else {
            std::os::unix::fs::symlink(entry.path(), &dest).unwrap();
        }
    }
    assert!(
        corrupted_one,
        "test setup bug: no dense-MLP .w3 file found to corrupt in {}",
        real_w3_dir.display()
    );

    let corpus = corpus_file();
    let out = run(&[
        "--decode-loop-ppl",
        "--w3-mlp-dir",
        fixture_dir.to_str().unwrap(),
        "--tokenizer-dir",
        real_tokenizer_dir.to_str().unwrap(),
        "--corpus-file",
        corpus.to_str().unwrap(),
        "--max-tokens",
        "8",
    ]);

    assert!(
        !out.status.success(),
        "a corrupted W3 tensor must be a hard failure under --decode-loop-ppl too"
    );
    let out_text = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out_text.contains("=== Perplexity Report"),
        "must never print a report when a required W3 artifact is corrupt"
    );
    assert_no_panic(&out);

    std::fs::remove_dir_all(&fixture_dir).ok();
    let _ = std::fs::remove_file(&corpus);
}
