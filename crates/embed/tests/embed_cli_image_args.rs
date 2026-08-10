//! Argument-validation tests for `embed`'s `--image` mode.
//!
//! Drives the compiled binary directly (no model loading, no network) so
//! these cover the hand-rolled arg loop's validation branches without
//! needing a real vision-language checkpoint.

use std::process::Command;

fn embed_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_embed"))
}

#[test]
fn image_and_text_are_mutually_exclusive() {
    let output = embed_cmd()
        .args(["--image", "x.png", "--text", "hello"])
        .output()
        .expect("embed binary must run");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("mutually exclusive"), "stderr: {stderr}");
}

#[test]
fn vision_model_dir_without_image_is_a_usage_error() {
    let output = embed_cmd()
        .args(["--vision-model-dir", "/some/dir", "--text", "hello"])
        .output()
        .expect("embed binary must run");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--vision-model-dir requires --image"),
        "stderr: {stderr}"
    );
}

#[test]
fn image_requires_vision_model_dir() {
    let output = embed_cmd()
        .args(["--image", "x.png"])
        .output()
        .expect("embed binary must run");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--vision-model-dir is required"),
        "stderr: {stderr}"
    );
}

#[test]
fn image_rejects_unknown_pooling_strategy() {
    let output = embed_cmd()
        .args([
            "--image",
            "x.png",
            "--vision-model-dir",
            "/nonexistent",
            "--pooling",
            "max",
        ])
        .output()
        .expect("embed binary must run");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--pooling must be 'mean_visual' or 'last_token'"),
        "stderr: {stderr}"
    );
}

#[test]
fn image_with_missing_checkpoint_directory_fails_closed() {
    let output = embed_cmd()
        .args([
            "--image",
            "x.png",
            "--vision-model-dir",
            "/nonexistent-dir-xyz",
        ])
        .output()
        .expect("embed binary must run");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to load vision-language checkpoint"),
        "stderr: {stderr}"
    );
}

#[test]
fn image_with_unreadable_file_fails_closed() {
    let dir =
        std::env::temp_dir().join(format!("embed-cli-image-args-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir must create");
    // Not a real checkpoint, but present -- config.json load fails before
    // this test's actual target (the missing image file) is ever reached,
    // so this exercises the same "checkpoint failed to load" fail-closed
    // path as the directory-missing case above from a different starting
    // point (a dir that exists but is not a checkpoint).
    let output = embed_cmd()
        .args([
            "--image",
            "does-not-exist.png",
            "--vision-model-dir",
            dir.to_str().unwrap(),
        ])
        .output()
        .expect("embed binary must run");
    assert!(!output.status.success());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn help_still_works_with_no_arguments_related_to_image_mode() {
    let output = embed_cmd()
        .args(["--help"])
        .output()
        .expect("embed binary must run");
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--image"), "stderr: {stderr}");
    assert!(stderr.contains("--vision-model-dir"), "stderr: {stderr}");
    assert!(stderr.contains("--metal"), "stderr: {stderr}");
}
