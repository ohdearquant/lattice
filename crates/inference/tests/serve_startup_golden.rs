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

#[derive(Clone, Copy)]
enum UnknownConfig {
    Valid,
    Malformed,
    Missing,
}

#[derive(Clone, Copy)]
enum UnknownTokenizer {
    Valid,
    Unparseable,
    Missing,
}

fn valid_unknown_config(identity: &str) -> Vec<u8> {
    match identity {
        "gemma4" => gemma_config(),
        "qwen3_5" => include_bytes!("fixtures/qwen35_0_8b_config.json").to_vec(),
        "absent" => {
            let mut config: serde_json::Value =
                serde_json::from_slice(include_bytes!("fixtures/qwen35_0_8b_config.json"))
                    .expect("valid Qwen config fixture");
            config
                .as_object_mut()
                .expect("Qwen config object")
                .remove("model_type");
            serde_json::to_vec(&config).expect("serialize config without model_type")
        }
        _ => unreachable!("known fixture identity"),
    }
}

fn unknown_model_dir(
    identity: &str,
    config_kind: UnknownConfig,
    tokenizer_kind: UnknownTokenizer,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temporary unknown-format model directory");
    match config_kind {
        UnknownConfig::Valid => {
            std::fs::write(
                dir.path().join("config.json"),
                valid_unknown_config(identity),
            )
            .expect("write valid config.json");
        }
        UnknownConfig::Malformed if identity == "absent" => {
            std::fs::write(dir.path().join("config.json"), b"{")
                .expect("write malformed config.json");
        }
        UnknownConfig::Malformed => {
            std::fs::write(
                dir.path().join("config.json"),
                format!(r#"{{"model_type":"{identity}""#),
            )
            .expect("write malformed config.json");
        }
        UnknownConfig::Missing => {}
    }
    match tokenizer_kind {
        UnknownTokenizer::Valid => {
            std::fs::copy(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/tokenizers/qwen3-embedding-0.6b/tokenizer.json"),
                dir.path().join("tokenizer.json"),
            )
            .expect("copy valid Qwen tokenizer fixture");
        }
        UnknownTokenizer::Unparseable => {
            std::fs::write(dir.path().join("tokenizer.json"), b"{")
                .expect("write unparseable tokenizer.json");
        }
        UnknownTokenizer::Missing => {}
    }
    dir
}

fn assert_unknown_case(
    binary: Binary,
    identity: &str,
    config: UnknownConfig,
    tokenizer: UnknownTokenizer,
    extra: &[&str],
    expected_error: &str,
) {
    let dir = unknown_model_dir(identity, config, tokenizer);
    let result = run(binary, Some(dir.path()), extra);
    let markers: &[&str] = match binary {
        Binary::LatticeServe => &["[lattice_serve] loading model from <TEMP> (unknown) ..."],
        Binary::Lattice => &["Loading model from <TEMP>..."],
    };
    assert_startup(&result, Some(dir.path()), Some(1), expected_error, markers);
    assert!(
        !result.stderr.contains(" ready (context=")
            && !result.stderr.contains("Model loaded. Serving as '")
            && !result.stderr.contains("Listening on "),
        "unknown-format startup printed a ready marker; stderr:\n{}",
        result.stderr
    );
}

#[test]
fn standalone_gemma_q4_precedes_preload_vision_refusal() {
    let dir = stub_model(&gemma_config(), Some("model_layers_0_weight.q4"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--preload-vision"],
    );
    assert_startup(
        &result,
        Some(dir.path()),
        Some(1),
        "lattice_serve: gemma_metal_unsupported: '<TEMP>' is a Gemma 4 checkpoint in the native Q4 format, which only the Metal backend reads, and Gemma 4 is served on the CPU backend only. Point --model at the Gemma 4 safetensors directory instead.",
        &[],
    );
}

#[test]
fn standalone_invalid_max_pending_precedes_bad_resident_bytes() {
    let dir = stub_model(&qwen_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &[
            "--max-pending",
            "invalid",
            "--max-resident-adapter-bytes",
            "0",
        ],
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
fn standalone_resident_count_precedes_resident_bytes_gemma_refusal() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &[
            "--max-resident-adapters",
            "4",
            "--max-resident-adapter-bytes",
            "1048576",
        ],
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
fn standalone_preload_vision_precedes_tokenizer_dir_gemma_refusal() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--preload-vision", "--tokenizer-dir", "alternate"],
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
fn standalone_malformed_qwen_config_precedes_zero_max_pending() {
    let dir = stub_model(b"{", Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--max-pending", "0"],
    );
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
fn standalone_gemma_refusal_precedes_zero_max_pending() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--max-pending", "0", "--preload-vision"],
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
fn standalone_malformed_reasoning_budget_is_ignored() {
    let dir = stub_model(&gemma_config(), Some("model.safetensors"));
    let result = run(
        Binary::LatticeServe,
        Some(dir.path()),
        &["--reasoning-budget", "not-a-number"],
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
fn standalone_unknown_gemma4_identity_valid_config() {
    assert_unknown_case(
        Binary::LatticeServe,
        "gemma4",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &[],
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: unknown variant \x60sliding_attention\x60, expected \x60linear_attention\x60 or \x60full_attention\x60 at line 83 column 25",
    );
}

#[test]
fn standalone_unknown_gemma4_identity_malformed_config() {
    assert_unknown_case(
        Binary::LatticeServe,
        "gemma4",
        UnknownConfig::Malformed,
        UnknownTokenizer::Valid,
        &[],
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: EOF while parsing an object at line 1 column 22",
    );
}

#[test]
fn standalone_unknown_qwen3_5_identity_valid_config() {
    assert_unknown_case(
        Binary::LatticeServe,
        "qwen3_5",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &[],
        "lattice_serve: '<TEMP>' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
    );
}

#[test]
fn standalone_unknown_qwen3_5_identity_malformed_config() {
    assert_unknown_case(
        Binary::LatticeServe,
        "qwen3_5",
        UnknownConfig::Malformed,
        UnknownTokenizer::Valid,
        &[],
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: EOF while parsing an object at line 1 column 23",
    );
}

#[test]
fn standalone_unknown_absent_identity_valid_config() {
    assert_unknown_case(
        Binary::LatticeServe,
        "absent",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &[],
        "lattice_serve: '<TEMP>' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
    );
}

#[test]
fn standalone_unknown_absent_identity_malformed_config() {
    assert_unknown_case(
        Binary::LatticeServe,
        "absent",
        UnknownConfig::Malformed,
        UnknownTokenizer::Valid,
        &[],
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: EOF while parsing an object at line 1 column 1",
    );
}

#[test]
fn standalone_unknown_missing_config_reaches_the_qwen_config_loader() {
    assert_unknown_case(
        Binary::LatticeServe,
        "absent",
        UnknownConfig::Missing,
        UnknownTokenizer::Valid,
        &[],
        "lattice_serve: config.json load failed: Model not found: missing config.json in <TEMP> -- every supported Qwen checkpoint ships one; no architecture preset is inferred from a config-less directory",
    );
}

#[test]
fn standalone_unknown_gemma4_valid_config_missing_tokenizer() {
    assert_unknown_case(
        Binary::LatticeServe,
        "gemma4",
        UnknownConfig::Valid,
        UnknownTokenizer::Missing,
        &[],
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: unknown variant \x60sliding_attention\x60, expected \x60linear_attention\x60 or \x60full_attention\x60 at line 83 column 25",
    );
}

#[test]
fn standalone_unknown_gemma4_valid_config_unparseable_tokenizer() {
    assert_unknown_case(
        Binary::LatticeServe,
        "gemma4",
        UnknownConfig::Valid,
        UnknownTokenizer::Unparseable,
        &[],
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: unknown variant \x60sliding_attention\x60, expected \x60linear_attention\x60 or \x60full_attention\x60 at line 83 column 25",
    );
}

#[test]
fn standalone_unknown_qwen3_5_valid_config_missing_tokenizer() {
    assert_unknown_case(
        Binary::LatticeServe,
        "qwen3_5",
        UnknownConfig::Valid,
        UnknownTokenizer::Missing,
        &[],
        "lattice_serve: tokenizer load failed (<TEMP>/tokenizer.json): Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
    );
}

#[test]
fn standalone_unknown_qwen3_5_valid_config_unparseable_tokenizer() {
    assert_unknown_case(
        Binary::LatticeServe,
        "qwen3_5",
        UnknownConfig::Valid,
        UnknownTokenizer::Unparseable,
        &[],
        "lattice_serve: tokenizer load failed (<TEMP>/tokenizer.json): Tokenizer error: expected JSON string at offset 1",
    );
}

#[test]
fn standalone_unknown_absent_valid_config_missing_tokenizer() {
    assert_unknown_case(
        Binary::LatticeServe,
        "absent",
        UnknownConfig::Valid,
        UnknownTokenizer::Missing,
        &[],
        "lattice_serve: tokenizer load failed (<TEMP>/tokenizer.json): Tokenizer error: failed to read <TEMP>/tokenizer.json: No such file or directory (os error 2)",
    );
}

#[test]
fn standalone_unknown_absent_valid_config_unparseable_tokenizer() {
    assert_unknown_case(
        Binary::LatticeServe,
        "absent",
        UnknownConfig::Valid,
        UnknownTokenizer::Unparseable,
        &[],
        "lattice_serve: tokenizer load failed (<TEMP>/tokenizer.json): Tokenizer error: expected JSON string at offset 1",
    );
}

#[test]
fn standalone_unknown_gemma4_valid_config_valid_tokenizer_zero_max_pending() {
    assert_unknown_case(
        Binary::LatticeServe,
        "gemma4",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &["--max-pending", "0"],
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: unknown variant \x60sliding_attention\x60, expected \x60linear_attention\x60 or \x60full_attention\x60 at line 83 column 25",
    );
}

#[test]
fn standalone_unknown_gemma4_valid_config_valid_tokenizer_preload_vision() {
    assert_unknown_case(
        Binary::LatticeServe,
        "gemma4",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &["--preload-vision"],
        "lattice_serve: config.json load failed: Inference error: invalid Qwen config.json: unknown variant \x60sliding_attention\x60, expected \x60linear_attention\x60 or \x60full_attention\x60 at line 83 column 25",
    );
}

#[test]
fn standalone_unknown_qwen3_5_valid_config_valid_tokenizer_zero_max_pending() {
    assert_unknown_case(
        Binary::LatticeServe,
        "qwen3_5",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &["--max-pending", "0"],
        "lattice_serve: --max-pending must be between 1 and 2305843009213693951 (got 0)",
    );
}

#[test]
fn standalone_unknown_qwen3_5_valid_config_valid_tokenizer_preload_vision() {
    assert_unknown_case(
        Binary::LatticeServe,
        "qwen3_5",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &["--preload-vision"],
        "lattice_serve: '<TEMP>' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
    );
}

#[test]
fn standalone_unknown_absent_valid_config_valid_tokenizer_zero_max_pending() {
    assert_unknown_case(
        Binary::LatticeServe,
        "absent",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &["--max-pending", "0"],
        "lattice_serve: --max-pending must be between 1 and 2305843009213693951 (got 0)",
    );
}

#[test]
fn standalone_unknown_absent_valid_config_valid_tokenizer_preload_vision() {
    assert_unknown_case(
        Binary::LatticeServe,
        "absent",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &["--preload-vision"],
        "lattice_serve: '<TEMP>' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
    );
}

#[test]
fn lattice_unknown_gemma4_valid_config_valid_tokenizer() {
    assert_unknown_case(
        Binary::Lattice,
        "gemma4",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &[],
        "Error: '<TEMP>' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
    );
}

#[test]
fn lattice_unknown_qwen3_5_valid_config_valid_tokenizer() {
    assert_unknown_case(
        Binary::Lattice,
        "qwen3_5",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &[],
        "Error: '<TEMP>' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
    );
}

#[test]
fn lattice_unknown_absent_valid_config_valid_tokenizer() {
    assert_unknown_case(
        Binary::Lattice,
        "absent",
        UnknownConfig::Valid,
        UnknownTokenizer::Valid,
        &[],
        "Error: '<TEMP>' is not a recognized model directory: no model.safetensors, model.safetensors.index.json, or *.q4 tensor files were found",
    );
}
