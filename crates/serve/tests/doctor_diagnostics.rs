use std::path::Path;
use std::process::{Command, Stdio};

fn doctor_error(model: &Path) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_lattice"))
        .args(["doctor", "--model"])
        .arg(model)
        .env("RUST_LOG", "off")
        .stdin(Stdio::null())
        .output()
        .expect("run the real lattice doctor command");
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.starts_with("Error: '"), "{stderr:?}");
    assert!(stderr.contains("is not a recognized model directory"));
    stderr
}

#[test]
fn doctor_cli_escapes_failed_report_diagnostics() {
    let root = tempfile::tempdir().unwrap();
    let model = root
        .path()
        .join("missing\nname\u{1b}\u{202e}\u{2028}\u{2029}é中");
    assert!(!model.exists());
    let stderr = doctor_error(&model);
    assert!(stderr.contains(r"missing\nname\u{1b}\u{202e}\u{2028}\u{2029}é中"));
    assert_eq!(stderr.lines().count(), 1, "{stderr:?}");
    for character in ['\u{1b}', '\u{202e}', '\u{2028}', '\u{2029}'] {
        assert!(!stderr.contains(character), "{stderr:?}");
    }
}

#[test]
fn doctor_cli_preserves_plain_error_text() {
    let root = tempfile::tempdir().unwrap();
    let stderr = doctor_error(&root.path().join("ordinary-é中"));
    assert!(stderr.contains("ordinary-é中' is not a recognized model directory"));
    assert_eq!(stderr.lines().count(), 1);
    assert!(stderr.ends_with('\n'));
}

#[test]
fn doctor_cli_distinguishes_literal_escapes_from_control_characters() {
    let root = tempfile::tempdir().unwrap();
    let literal = doctor_error(&root.path().join(r"missing\u{1b}"));
    let control = doctor_error(&root.path().join("missing\u{1b}"));
    assert_ne!(literal, control);
    assert!(literal.contains(r"missing\\u{1b}'"), "{literal:?}");
    assert!(control.contains(r"missing\u{1b}'"), "{control:?}");
    assert!(!control.contains('\u{1b}'));
}
