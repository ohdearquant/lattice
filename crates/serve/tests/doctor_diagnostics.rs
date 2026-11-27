use std::path::Path;
use std::process::{Command, Stdio};

fn model_command_error(command: &str, model: &Path) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_lattice"))
        .args([command, "--model"])
        .arg(model)
        .env("RUST_LOG", "off")
        .stdin(Stdio::null())
        .output()
        .expect("run the real lattice command");
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    String::from_utf8(output.stderr).unwrap()
}

#[test]
fn doctor_cli_escapes_failed_report_diagnostics() {
    let root = tempfile::tempdir().unwrap();
    let model = root
        .path()
        .join("missing\nname\u{1b}\u{202e}\u{2028}\u{2029}é中");
    assert!(!model.exists());
    let stderr = model_command_error("doctor", &model);
    assert!(stderr.contains(r"missing\nname\u{1b}\u{202e}\u{2028}\u{2029}é中"));
    assert_eq!(stderr.lines().count(), 1, "{stderr:?}");
    for character in ['\u{1b}', '\u{202e}', '\u{2028}', '\u{2029}'] {
        assert!(!stderr.contains(character), "{stderr:?}");
    }
}

#[test]
fn doctor_cli_preserves_plain_error_text() {
    let root = tempfile::tempdir().unwrap();
    let stderr = model_command_error("doctor", &root.path().join("ordinary-é中"));
    assert!(stderr.starts_with("Error: '"), "{stderr:?}");
    assert!(
        stderr.contains("ordinary-é中' is not a recognized model directory"),
        "{stderr:?}"
    );
    assert_eq!(stderr.lines().count(), 1);
    assert!(stderr.ends_with('\n'));
}

#[test]
fn doctor_cli_distinguishes_literal_escapes_from_control_characters() {
    let root = tempfile::tempdir().unwrap();
    let literal = model_command_error("doctor", &root.path().join(r"missing\u{1b}end"));
    let control = model_command_error("doctor", &root.path().join("missing\u{1b}end"));
    assert_ne!(literal, control);
    assert!(literal.contains(r"missing\\u{1b}end"), "{literal:?}");
    assert!(control.contains(r"missing\u{1b}end"), "{control:?}");
    assert!(!control.contains('\u{1b}'));
}

#[test]
fn chat_and_serve_escape_model_path_errors() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("missing\nname\u{1b}\u{202e}é中");
    for command in ["chat", "serve"] {
        let stderr = model_command_error(command, &model);
        let expected = r"missing\nname\u{1b}\u{202e}é中";
        let lines: Vec<_> = stderr.lines().collect();
        assert_eq!(lines.len(), 2, "{command}: {stderr:?}");
        assert!(lines[1].starts_with("Error: "), "{command}: {stderr:?}");
        assert!(
            lines.iter().all(|line| line.contains(expected)),
            "{command}: {stderr:?}"
        );
        assert!(
            !stderr.contains(['\u{1b}', '\u{202e}']),
            "{command}: {stderr:?}"
        );
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main_prune_score_error(calibration: &Path) -> String {
    let root = calibration.parent().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_lattice"))
        .args(["prune-score", "--q4-dir"])
        .arg(root.join("unused-q4"))
        .arg("--tokenizer-dir")
        .arg(root.join("unused-tokenizer"))
        .arg("--calibration-corpus")
        .arg(calibration)
        .arg("--validation-corpus")
        .arg(root.join("unused-validation.txt"))
        .args(["--prune-layers", "1"])
        .env("RUST_LOG", "off")
        .stdin(Stdio::null())
        .output()
        .expect("run the real lattice prune-score command");
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.starts_with("Error: failed to resolve calibration corpus "),
        "wrong main refusal: {stderr:?}"
    );
    stderr
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main_error_cli_escapes_prune_score_path() {
    let root = tempfile::tempdir().unwrap();
    let calibration = root
        .path()
        .join("calibration\nname\u{1b}\u{202e}\u{2028}\u{2029}é中\\literal");
    assert!(!calibration.exists());
    let stderr = main_prune_score_error(&calibration);
    assert!(
        stderr.contains(r"calibration\nname\u{1b}\u{202e}\u{2028}\u{2029}é中\\literal"),
        "main prune-score error did not escape the calibration path: {stderr:?}"
    );
    assert_eq!(stderr.lines().count(), 1, "{stderr:?}");
    assert!(stderr.ends_with('\n'), "{stderr:?}");
    for character in ['\r', '\u{1b}', '\u{202e}', '\u{2028}', '\u{2029}'] {
        assert!(!stderr.contains(character), "{stderr:?}");
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main_error_cli_preserves_plain_prune_score_error() {
    let root = tempfile::tempdir().unwrap();
    let calibration = root.path().join("ordinary-é中.txt");
    assert!(!calibration.exists());
    let error = std::fs::canonicalize(&calibration).unwrap_err();
    let expected = format!(
        "Error: failed to resolve calibration corpus {}: {error}\n",
        calibration.display()
    );
    let stderr = main_prune_score_error(&calibration);
    assert_eq!(stderr, expected);
    assert_eq!(stderr.lines().count(), 1);
    assert!(stderr.ends_with('\n'));
}
