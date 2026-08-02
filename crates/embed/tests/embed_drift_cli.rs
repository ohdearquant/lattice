#![cfg(feature = "native")]

use std::process::Command;

fn drift_command(model_root: &std::path::Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_embed-drift"));
    command.env("LATTICE_MODEL_CACHE", model_root).arg("--json");
    command
}

fn json_event(stdout: &[u8]) -> serde_json::Value {
    let stdout = String::from_utf8_lossy(stdout);
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("@@lattice "))
        .expect("missing @@lattice event");
    serde_json::from_str(line).unwrap()
}

#[test]
fn absent_weights_are_visible_without_enforcement() {
    let model_root = tempfile::tempdir().unwrap();
    let output = drift_command(model_root.path()).output().unwrap();

    assert_eq!(output.status.code(), Some(0));
    let event = json_event(&output.stdout);
    assert_eq!(event["checked"], 0);
    assert!(event["skipped"].as_u64().unwrap() > 0);
    assert!(
        event["results"].as_array().unwrap().iter().all(|result| {
            result["status"] == "weights_absent" && result["verdict"] == "skipped"
        })
    );
}

#[test]
fn absent_weights_fail_with_enforcement() {
    let model_root = tempfile::tempdir().unwrap();
    let output = drift_command(model_root.path())
        .arg("--enforce")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let event = json_event(&output.stdout);
    assert_eq!(event["checked"], 0);
    assert!(event["skipped"].as_u64().unwrap() > 0);
}

#[test]
fn requested_model_without_fixture_is_distinct() {
    let model_root = tempfile::tempdir().unwrap();
    let output = drift_command(model_root.path())
        .args(["--model", "bge-base-en-v1.5"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(3));
    let event = json_event(&output.stdout);
    assert_eq!(event["checked"], 0);
    assert_eq!(event["skipped"], 0);
    assert_eq!(event["no_baseline"], 1);
    assert_eq!(event["results"][0]["status"], "no_baseline");
}

#[test]
fn baseline_update_refuses_environment_enforcement() {
    let model_root = tempfile::tempdir().unwrap();
    let output = drift_command(model_root.path())
        .env("LATTICE_DRIFT_GATE_ENFORCE", "1")
        .arg("--update-baseline")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("--update-baseline cannot be combined with --enforce")
    );
}
