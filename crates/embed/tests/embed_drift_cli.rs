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
fn deleting_a_quartet_fixture_is_a_no_baseline_failure_not_a_silent_shrink() {
    // A missing fixture for one of the four production models must not
    // silently shrink the default (no --model) checked set down to the
    // fixtures that remain on disk; it must surface as NoBaseline for the
    // still-required model and exit 3. This guards F1: `requested_models()`
    // binds the expected quartet independently of directory discovery.
    let checked_in_fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("embed_drift_baseline_v1");

    let scratch_baselines = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(&checked_in_fixtures).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        // Omit bge_small_en_v15.json to simulate a deleted fixture.
        if path.file_name().unwrap() == "bge_small_en_v15.json" {
            continue;
        }
        std::fs::copy(
            &path,
            scratch_baselines.path().join(path.file_name().unwrap()),
        )
        .unwrap();
    }

    let model_root = tempfile::tempdir().unwrap();
    let output = drift_command(model_root.path())
        .env("LATTICE_DRIFT_BASELINE_DIR", scratch_baselines.path())
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(3));
    let event = json_event(&output.stdout);
    assert_eq!(event["no_baseline"], 1);
    assert_eq!(event["checked"], 0);
    let results = event["results"].as_array().unwrap();
    let bge_result = results
        .iter()
        .find(|result| result["model"] == "bge-small-en-v1.5")
        .expect("bge-small-en-v1.5 must still be in the requested (default) set");
    assert_eq!(bge_result["status"], "no_baseline");
    // The other three quartet members are still enforced via WeightsAbsent
    // (no weights provisioned in this scratch model root), not dropped.
    assert!(
        results
            .iter()
            .filter(|result| result["status"] == "weights_absent")
            .count()
            == 3
    );
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
