//! Real-checkpoint gate for continuing a training run across invocations.
//!
//! One run of four steps and a run split into two invocations of two steps,
//! the second continuing from the first's adapter and saved state, must end on
//! the same adapter bit for bit. A continuation that loads the adapter without
//! the state restarts the optimizer and the step cursor, so it must end on a
//! different adapter; that arm shows the comparison can tell them apart. The
//! remaining arms check that a state is refused beside the wrong adapter, the
//! wrong data or different settings.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use lattice_tune::lora::LoraAdapter;

fn write_data(dir: &Path) {
    let rows = [
        ("The capital of France is", " Paris."),
        ("Two plus three equals", " five."),
        ("Water freezes at zero degrees", " Celsius."),
    ];
    let mut text = String::new();
    for (prompt, completion) in rows {
        text.push_str(&format!(
            "{{\"prompt\": \"{prompt}\", \"completion\": \"{completion}\"}}"
        ));
        text.push('\n');
    }
    std::fs::write(dir.join("train.jsonl"), text).unwrap();
}

/// Run the trainer with fixed defaults; a flag in `extra` replaces its default
/// rather than following it, because the parser reads a flag's first value.
fn train(model_dir: &Path, data_dir: &Path, extra: &[&str]) -> Output {
    let defaults = [
        ("--first-layer", "21"),
        ("--rank", "2"),
        ("--alpha", "4"),
        ("--lr", "1e-3"),
        ("--seq-len", "48"),
        ("--max-train", "3"),
        ("--max-valid", "0"),
        ("--log-every", "1"),
    ];
    let mut args = vec![
        "--model-dir".to_string(),
        model_dir.display().to_string(),
        "--data-dir".to_string(),
        data_dir.display().to_string(),
    ];
    for (flag, value) in defaults {
        if !extra.contains(&flag) {
            args.push(flag.to_string());
            args.push(value.to_string());
        }
    }
    args.extend(extra.iter().map(ToString::to_string));
    Command::new(env!("CARGO_BIN_EXE_train_grad_full"))
        .args(&args)
        .output()
        .unwrap()
}

fn succeeded(label: &str, output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{label} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

fn refused(label: &str, output: &Output, needle: &str) {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!output.status.success(), "{label} was accepted:\n{text}");
    assert!(
        text.contains(needle),
        "{label}: expected `{needle}` in:\n{text}"
    );
}

/// Every adapter tensor as raw bits, keyed and sorted, so equality is exact.
fn bits(path: &Path) -> Vec<(String, Vec<u32>, Vec<u32>)> {
    let adapter = LoraAdapter::from_safetensors(path).unwrap();
    let mut out: Vec<_> = adapter
        .layers()
        .iter()
        .map(|((layer, module), l)| {
            (
                format!("{layer}.{module}"),
                l.a.iter().map(|v| v.to_bits()).collect(),
                l.b.iter().map(|v| v.to_bits()).collect(),
            )
        })
        .collect();
    out.sort();
    out
}

#[test]
fn resumed_run_matches_an_uninterrupted_run() {
    if std::env::var_os("LATTICE_RESUME_GATE_ENFORCE").is_none() {
        eprintln!(
            "resumable-training gate skipped; set LATTICE_RESUME_GATE_ENFORCE=1 with LATTICE_MODEL_DIR"
        );
        return;
    }
    let model_dir = PathBuf::from(
        std::env::var_os("LATTICE_MODEL_DIR")
            .expect("LATTICE_MODEL_DIR is required when the resumable-training gate is enforced"),
    );
    assert!(
        model_dir.join("config.json").is_file(),
        "enforced gate requires a provisioned model at {}",
        model_dir.display()
    );

    let temp = tempfile::tempdir().unwrap();
    let data = temp.path().join("data");
    std::fs::create_dir(&data).unwrap();
    write_data(&data);
    let p = |name: &str| temp.path().join(name).display().to_string();

    // Uninterrupted: four steps visiting samples 0, 1, 2, 0, with train scoring
    // on, so the split runs below also show that scoring off trains identically.
    let whole = train(
        &model_dir,
        &data,
        &["--steps", "4", "--save", &p("whole.st")],
    );
    succeeded("uninterrupted run", &whole);

    let first = train(
        &model_dir,
        &data,
        &[
            "--steps",
            "2",
            "--no-train-score",
            "--save",
            &p("first.st"),
            "--save-state",
            &p("first.state"),
        ],
    );
    let first_out = succeeded("first half", &first);
    assert!(
        first_out.contains("saved training state (next step 3)"),
        "{first_out}"
    );

    let second = train(
        &model_dir,
        &data,
        &[
            "--steps",
            "2",
            "--no-train-score",
            "--load",
            &p("first.st"),
            "--load-state",
            &p("first.state"),
            "--save",
            &p("second.st"),
            "--save-state",
            &p("second.state"),
        ],
    );
    let second_out = succeeded("second half", &second);
    assert!(second_out.contains("resuming at step 3"), "{second_out}");
    assert!(
        second_out.contains("step    3  sample 2 NLL"),
        "{second_out}"
    );
    assert!(
        second_out.contains("step    4  sample 0 NLL"),
        "{second_out}"
    );
    assert!(
        second_out.contains("saved training state (next step 5)"),
        "{second_out}"
    );

    let whole_bits = bits(Path::new(&p("whole.st")));
    assert!(!whole_bits.is_empty());
    assert_eq!(
        whole_bits,
        bits(Path::new(&p("second.st"))),
        "the continued run must end on the uninterrupted run's adapter"
    );

    // Control: the same continuation without the state restarts Adam and the
    // cursor (it trains samples 0 and 1 again), so its adapter must differ.
    let restart = train(
        &model_dir,
        &data,
        &[
            "--steps",
            "2",
            "--no-train-score",
            "--load",
            &p("first.st"),
            "--save",
            &p("restart.st"),
        ],
    );
    succeeded("restart without state", &restart);
    assert_ne!(
        whole_bits,
        bits(Path::new(&p("restart.st"))),
        "a continuation without its state must not match; the comparison is blind otherwise"
    );

    let resume_with = |extra: &[&str]| {
        let mut args = vec!["--steps", "2", "--no-train-score", "--load-state"];
        let state = p("first.state");
        args.push(&state);
        args.extend_from_slice(extra);
        train(&model_dir, &data, &args)
    };
    refused(
        "state beside the wrong adapter",
        &resume_with(&["--load", &p("whole.st")]),
        "not the adapter this state was saved beside",
    );
    refused(
        "state with different data",
        &resume_with(&["--load", &p("first.st"), "--max-train", "2"]),
        "was saved for 3 training samples",
    );
    refused(
        "state with a different learning rate",
        &resume_with(&["--load", &p("first.st"), "--lr", "5e-4"]),
        "different settings",
    );
}
