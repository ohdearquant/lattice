#![cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "f16",
    feature = "serve"
))]

mod common;

use common::{Binary, assert_startup_golden, gemma_config, qwen_config, run_startup, stub_model};
use std::path::Path;

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn run(binary: Binary, model: Option<&Path>, extra: &[&str]) -> common::StartupRun {
    run_startup(binary, model, &args(extra))
}

fn assert_startup(
    result: &common::StartupRun,
    root: Option<&Path>,
    code: Option<i32>,
    last: &str,
    markers: &[&str],
) {
    let roots = root.map(|root| vec![(root, "<TEMP>")]).unwrap_or_default();
    assert_startup_golden(result, &roots, code, last, markers);
}

#[test]
fn standalone_missing_model_has_one_exact_error_and_no_startup_markers() {
    let result = run(Binary::LatticeServe, None, &[]);
    assert_startup(
        &result,
        None,
        Some(1),
        "lattice_serve: missing --model <name-or-path> (e.g. --model qwen3.5-0.8b)",
        &[],
    );
}

#[test]
fn standalone_nonexistent_directory_fails_before_route_selection() {
    let root = tempfile::tempdir().expect("temporary parent");
    let model = root.path().join("absent");
    let result = run(Binary::LatticeServe, Some(&model), &[]);
    assert_startup(
        &result,
        Some(root.path()),
        Some(1),
        "lattice_serve: model directory not found: <TEMP>/absent",
        &[],
    );
}

#[test]
fn standalone_empty_directory_keeps_the_qwen_config_error_after_loading_marker() {
    let root = tempfile::tempdir().expect("temporary model directory");
    let result = run(Binary::LatticeServe, Some(root.path()), &[]);
    assert_startup(
        &result,
        Some(root.path()),
        Some(1),
        "lattice_serve: config.json load failed: Model not found: missing config.json in <TEMP> -- every supported Qwen checkpoint ships one; no architecture preset is inferred from a config-less directory",
        &["[lattice_serve] loading model from <TEMP> (unknown) ..."],
    );
}

#[test]
fn standalone_gemma_q4_is_refused_before_selection_or_loading() {
    let dir = stub_model(&gemma_config(), Some("model_layers_0_weight.q4"));
    let result = run(Binary::LatticeServe, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: gemma_metal_unsupported: '<TEMP>' is a Gemma 4 checkpoint in the native Q4 format, which only the Metal backend reads, and Gemma 4 is served on the CPU backend only. Point --model at the Gemma 4 safetensors directory instead.",
        &[],
    );
}

#[test]
fn standalone_gemma_refuses_preload_vision_before_route_selection() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--preload-vision"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: unsupported_feature: --preload-vision is not supported for Gemma 4 checkpoints",
        &[],
    );
}

#[test]
fn standalone_gemma_refuses_tokenizer_dir_before_route_selection() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let alternate = dir.path().join("alternate");
    std::fs::create_dir_all(&alternate).expect("create alternate tokenizer directory");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gemma4/tokenizer/tokenizer.json"),
        alternate.join("tokenizer.json"),
    )
    .expect("copy Gemma tokenizer fixture");
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &[
            "--tokenizer-dir",
            alternate.to_str().expect("UTF-8 temp path"),
        ],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: unsupported_feature: --tokenizer-dir is not supported for Gemma 4 checkpoints",
        &[],
    );
}

#[test]
fn standalone_gemma_refuses_max_resident_adapters_before_route_selection() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--max-resident-adapters", "4"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: unsupported_feature: --max-resident-adapters is not supported for Gemma 4 checkpoints",
        &[],
    );
}

#[test]
fn standalone_gemma_refuses_max_resident_adapter_bytes_before_route_selection() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--max-resident-adapter-bytes", "1048576"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: unsupported_feature: --max-resident-adapter-bytes is not supported for Gemma 4 checkpoints",
        &[],
    );
}

#[test]
fn standalone_gemma_refuses_positive_reasoning_budget_before_route_selection() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--reasoning-budget", "1"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: unsupported_feature: --reasoning-budget is not supported for Gemma 4 checkpoints",
        &[],
    );
}

#[test]
fn standalone_zero_reasoning_budget_reaches_the_gemma_loader() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--reasoning-budget", "0"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: Gemma 4 model load failed: Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
        &[
            "[route] selected family=gemma4 backend=cpu format=safetensors",
            "[lattice_serve] loading model from <TEMP> (safetensors) ...",
        ],
    );
}

#[test]
fn standalone_invalid_max_pending_precedes_gemma_flag_refusal() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--preload-vision", "--max-pending", "invalid"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: --max-pending: invalid value \"invalid\" (expected a positive integer)",
        &[],
    );
}

#[test]
fn standalone_invalid_residency_limit_precedes_gemma_flag_refusal() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--preload-vision", "--max-resident-adapter-bytes", "0"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: --max-resident-adapter-bytes: expected a positive integer",
        &[],
    );
}

#[test]
fn standalone_qwen_stub_without_tokenizer_reaches_the_metal_loader() {
    let dir = stub_model(&qwen_config(), Some("model.safetensors"));
    let result = run(Binary::LatticeServe, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: tokenizer load failed (<TEMP>/tokenizer.json): Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP> (bf16) ...",
        ],
    );
}

#[test]
fn standalone_malformed_qwen_config_fails_after_route_selection() {
    let dir = stub_model(b"{", Some("model.safetensors"));
    let result = run(Binary::LatticeServe, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: EOF while parsing an object at line 1 column 1",
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP> (bf16) ...",
        ],
    );
}

#[test]
fn standalone_malformed_gemma_config_fails_after_route_selection() {
    let dir = stub_model(br#"{"model_type":"gemma4"}"#, Some("model.safetensors"));
    let result = run(Binary::LatticeServe, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: Gemma 4 model load failed: Inference error: invalid Gemma 4 config.json: missing field `text_config` at line 1 column 23",
        &[
            "[route] selected family=gemma4 backend=cpu format=safetensors",
            "[lattice_serve] loading model from <TEMP> (safetensors) ...",
        ],
    );
}

#[test]
fn standalone_gemma_valid_config_without_tokenizer_reaches_tokenizer_load() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(Binary::LatticeServe, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: Gemma 4 model load failed: Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
        &[
            "[route] selected family=gemma4 backend=cpu format=safetensors",
            "[lattice_serve] loading model from <TEMP> (safetensors) ...",
        ],
    );
}

#[test]
fn lattice_unrecognized_format_fails_after_the_initial_loading_marker() {
    let dir = tempfile::tempdir().expect("temporary model directory");
    let result = run(Binary::Lattice, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "Error: '<TEMP>' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
        &["Loading model from <TEMP>..."],
    );
}

#[test]
fn lattice_nonexistent_model_is_rejected_by_format_detection_after_loading_marker() {
    let root = tempfile::tempdir().expect("temporary parent");
    let model = root.path().join("absent");
    let result = run(Binary::Lattice, Some(&model), &[]);
    assert_startup(
        &result,
        Some(root.path()),
        Some(1),
        "Error: '<TEMP>/absent' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
        &["Loading model from <TEMP>/absent..."],
    );
}

#[test]
fn lattice_gemma_q4_is_refused_after_loading_marker_and_before_route_marker() {
    let dir = stub_model(&gemma_config(), Some("model_layers_0_weight.q4"));
    let result = run(Binary::Lattice, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "Error: gemma_metal_unsupported: '<TEMP>' is a Gemma 4 checkpoint in the native Q4 format, which only the Metal backend reads, and Gemma 4 is served on the CPU backend only. Point --model at the Gemma 4 safetensors directory instead.",
        &["Loading model from <TEMP>..."],
    );
}

#[test]
fn lattice_qwen_safetensors_stub_reaches_the_cpu_loader() {
    let dir = stub_model(&qwen_config(), Some("model.safetensors"));
    let result = run(Binary::Lattice, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "Error: failed to load model: Invalid safetensors file: file too small to contain safetensors header",
        &[
            "Loading model from <TEMP>...",
            "[route] selected family=qwen35 backend=cpu format=safetensors",
        ],
    );
}

#[test]
fn lattice_gemma_stub_without_tokenizer_fails_in_the_cpu_loader() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(Binary::Lattice, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "Error: failed to load Gemma 4 model: Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
        &[
            "Loading model from <TEMP>...",
            "[route] selected family=gemma4 backend=cpu format=safetensors",
        ],
    );
}

#[test]
fn lattice_q4_without_tokenizer_fails_before_worker_startup() {
    let dir = stub_model(&qwen_config(), Some("model_layers_0_weight.q4"));
    let result = run(Binary::Lattice, Some(dir.path()), &[]);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "Error: failed to load Q4 model: tokenizer load failed (<TEMP>/tokenizer.json): Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
        &[
            "Loading model from <TEMP>...",
            "[route] selected family=qwen35 backend=metal format=q4",
        ],
    );
}

#[test]
fn lattice_gemma_ignores_preload_vision_until_its_loader_fails() {
    lattice_gemma_flag_is_ignored(&["--preload-vision"]);
}

#[test]
fn lattice_gemma_ignores_tokenizer_dir_until_its_loader_fails() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let alternate = dir.path().join("alternate");
    std::fs::create_dir_all(&alternate).expect("create alternate tokenizer directory");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gemma4/tokenizer/tokenizer.json"),
        alternate.join("tokenizer.json"),
    )
    .expect("copy Gemma tokenizer fixture");
    let result = run(
        Binary::Lattice,
        Some(dir.path()),
        &[
            "--tokenizer-dir",
            alternate.to_str().expect("UTF-8 temp path"),
        ],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "Error: failed to load Gemma 4 model: Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
        &[
            "Loading model from <TEMP>...",
            "[route] selected family=gemma4 backend=cpu format=safetensors",
        ],
    );
}

#[test]
fn lattice_gemma_ignores_max_resident_adapters_until_its_loader_fails() {
    lattice_gemma_flag_is_ignored(&["--max-resident-adapters", "4"]);
}

#[test]
fn lattice_gemma_ignores_max_resident_adapter_bytes_until_its_loader_fails() {
    lattice_gemma_flag_is_ignored(&["--max-resident-adapter-bytes", "1048576"]);
}

#[test]
fn lattice_gemma_ignores_max_pending_until_its_loader_fails() {
    lattice_gemma_flag_is_ignored(&["--max-pending", "4"]);
}

fn lattice_gemma_flag_is_ignored(extra: &[&str]) {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(Binary::Lattice, Some(dir.path()), extra);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "Error: failed to load Gemma 4 model: Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
        &[
            "Loading model from <TEMP>...",
            "[route] selected family=gemma4 backend=cpu format=safetensors",
        ],
    );
}

#[test]
fn lattice_reasoning_budget_flag_is_rejected_by_clap_before_startup() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::Lattice,
        Some(dir.path()),
        &["--reasoning-budget", "1"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(2),
        "For more information, try '--help'.",
        &[],
    );
    assert!(
        result
            .stderr
            .lines()
            .any(|line| line == "error: unexpected argument '--reasoning-budget' found"),
        "clap usage error line was not pinned; stderr:\n{}",
        result.stderr
    );
}

#[test]
fn lattice_router_state_does_not_refuse_before_qwen_loads() {
    lattice_router_or_embedding_flag_loses_to_qwen_load(&["--router-state", "router-state"]);
}

#[test]
fn lattice_router_pin_does_not_refuse_before_qwen_loads() {
    lattice_router_or_embedding_flag_loses_to_qwen_load(&["--router-pin", "1"]);
}

#[test]
fn lattice_embedding_model_does_not_refuse_before_qwen_loads() {
    lattice_router_or_embedding_flag_loses_to_qwen_load(&["--embedding-model", "embedder"]);
}

#[test]
fn lattice_embedding_model_id_does_not_refuse_before_qwen_loads() {
    lattice_router_or_embedding_flag_loses_to_qwen_load(&["--embedding-model-id", "embedder"]);
}

fn lattice_router_or_embedding_flag_loses_to_qwen_load(extra: &[&str]) {
    let dir = stub_model(&qwen_config(), Some("model.safetensors"));
    let result = run(Binary::Lattice, Some(dir.path()), extra);
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "Error: failed to load model: Invalid safetensors file: file too small to contain safetensors header",
        &[
            "Loading model from <TEMP>...",
            "[route] selected family=qwen35 backend=cpu format=safetensors",
        ],
    );
}
