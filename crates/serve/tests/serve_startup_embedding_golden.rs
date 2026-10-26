#![cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "f16",
    feature = "test-utils"
))]

mod common;

use common::{Binary, Server, acquire_gpu_lock, assert_startup_golden, run_startup};
use lattice_inference::{bert_pooling_fixture, qwen35_checkpoint_fixture};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn generation_checkpoint(root: &Path) -> PathBuf {
    let path = root.join("generation");
    qwen35_checkpoint_fixture::write_checkpoint(&path)
        .unwrap_or_else(|error| panic!("write Qwen3.5 fixture: {error}"));
    path
}

fn bert_checkpoint(root: &Path) -> PathBuf {
    let path = root.join("bert-fixture");
    bert_pooling_fixture::write_checkpoint(&path)
        .unwrap_or_else(|error| panic!("write BERT fixture: {error}"));
    path
}

fn assert_failure(
    run: &common::StartupRun,
    root: &Path,
    code: i32,
    last_line: &str,
    markers: &[&str],
) {
    assert_startup_golden(run, &[(root, "<TEMP>")], Some(code), last_line, markers);
}

fn startup_markers(stderr: &str, root: &Path, port: u16) -> Vec<String> {
    let normalized = stderr
        .replace(&root.display().to_string(), "<TEMP>")
        .replace(&format!("127.0.0.1:{port}"), "127.0.0.1:<PORT>");
    normalized
        .lines()
        .filter(|line| {
            line.starts_with("Loading model from ")
                || line.starts_with("[route] selected ")
                || line.starts_with("Model loaded. Serving as '")
                || line.starts_with("Embeddings disabled (")
                || line.starts_with("Listening on ")
                || line.starts_with("[lattice_serve] loading model from ")
                || line.starts_with("[lattice_serve] model '")
                || line.starts_with("[lattice_serve] loading embedding model from ")
                || line.starts_with("[lattice_serve] embedding model '")
                || line.starts_with("[lattice_serve] OpenAI-compatible API on ")
        })
        .map(str::to_owned)
        .collect()
}

fn assert_server_markers(stderr: &str, root: &Path, port: u16, expected: &[&str]) {
    let expected = expected
        .iter()
        .map(|line| (*line).to_owned())
        .collect::<Vec<_>>();
    assert_eq!(startup_markers(stderr, root, port), expected);
}

#[test]
fn lattice_without_embedding_flag_logs_implicit_text_only_refusal() {
    let _gpu = acquire_gpu_lock();
    let root = tempfile::tempdir().expect("temporary fixture root");
    let generation = generation_checkpoint(root.path());
    let server = Server::spawn(Binary::Lattice, &generation, &[]);

    assert_server_markers(
        &server.diagnostics(),
        root.path(),
        server.port(),
        &[
            "Loading model from <TEMP>/generation...",
            "[route] selected family=qwen35 backend=cpu format=safetensors",
            "Model loaded. Serving as 'characterization-model'.",
            "Embeddings disabled (<TEMP>/generation): <TEMP>/generation has no vision_config; not a vision-language checkpoint",
            "Listening on 127.0.0.1:<PORT>  (model: characterization-model, max_tokens default: 2)",
        ],
    );
}

#[test]
fn lattice_missing_embedding_directory_fails_after_generation_load() {
    let _gpu = acquire_gpu_lock();
    let root = tempfile::tempdir().expect("temporary fixture root");
    let generation = generation_checkpoint(root.path());
    let missing = root.path().join("missing-embedding");
    let run = run_startup(
        Binary::Lattice,
        Some(&generation),
        &args(&["--embedding-model", missing.to_str().expect("UTF-8 path")]),
    );

    assert_failure(
        &run,
        root.path(),
        1,
        "Error: --embedding-model <TEMP>/missing-embedding cannot serve as an embedding model: config.json: Model not found: missing config.json in <TEMP>/missing-embedding -- every supported Qwen checkpoint ships one; no architecture preset is inferred from a config-less directory",
        &[
            "Loading model from <TEMP>/generation...",
            "[route] selected family=qwen35 backend=cpu format=safetensors",
            "Model loaded. Serving as 'generation'.",
        ],
    );
}

#[test]
fn lattice_text_checkpoint_is_refused_as_explicit_embedding_model() {
    let _gpu = acquire_gpu_lock();
    let root = tempfile::tempdir().expect("temporary fixture root");
    let generation = generation_checkpoint(root.path());
    let run = run_startup(
        Binary::Lattice,
        Some(&generation),
        &args(&[
            "--embedding-model",
            generation.to_str().expect("UTF-8 path"),
        ]),
    );

    assert_failure(
        &run,
        root.path(),
        1,
        "Error: --embedding-model <TEMP>/generation cannot serve as an embedding model: <TEMP>/generation has no vision_config; not a vision-language checkpoint",
        &[
            "Loading model from <TEMP>/generation...",
            "[route] selected family=qwen35 backend=cpu format=safetensors",
            "Model loaded. Serving as 'generation'.",
        ],
    );
}

#[test]
fn lattice_serve_without_embedding_flag_returns_full_no_model_refusal() {
    let _gpu = acquire_gpu_lock();
    let root = tempfile::tempdir().expect("temporary fixture root");
    let generation = generation_checkpoint(root.path());
    let server = Server::spawn(Binary::LatticeServe, &generation, &[]);

    assert_server_markers(
        &server.diagnostics(),
        root.path(),
        server.port(),
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP>/generation (bf16) ...",
            "[lattice_serve] model 'generation' (bf16) ready (context=1024, from model max_position_embeddings)",
            "[lattice_serve] OpenAI-compatible API on http://127.0.0.1:<PORT>/v1",
        ],
    );
    let response = server.embeddings();
    assert_eq!(response.status, 503);
    assert_eq!(
        response.body,
        r#"{"error":{"message":"no embedding model is loaded; start lattice_serve with --embedding-model <dir>","type":"server_error","code":"embedding_model_not_loaded","param":null}}"#
    );
}

#[test]
fn lattice_serve_loads_bert_embedding_and_returns_finite_vectors() {
    let _gpu = acquire_gpu_lock();
    let root = tempfile::tempdir().expect("temporary fixture root");
    let generation = generation_checkpoint(root.path());
    let embedding = bert_checkpoint(root.path());
    let server = Server::spawn(
        Binary::LatticeServe,
        &generation,
        &args(&["--embedding-model", embedding.to_str().expect("UTF-8 path")]),
    );

    assert_server_markers(
        &server.diagnostics(),
        root.path(),
        server.port(),
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP>/generation (bf16) ...",
            "[lattice_serve] model 'generation' (bf16) ready (context=1024, from model max_position_embeddings)",
            "[lattice_serve] loading embedding model from <TEMP>/bert-fixture ...",
            "[lattice_serve] embedding model 'bert-fixture' ready (dimensions=384)",
            "[lattice_serve] OpenAI-compatible API on http://127.0.0.1:<PORT>/v1",
        ],
    );
    let response = server.embeddings();
    assert_eq!(response.status, 200, "{}", response.body);
    let body: Value = serde_json::from_str(&response.body).expect("embedding response JSON");
    let vectors = body["data"]
        .as_array()
        .expect("embedding response data array");
    let vector = vectors[0]["embedding"]
        .as_array()
        .expect("embedding vector array");
    assert_eq!(vector.len(), bert_pooling_fixture::DIMENSIONS);
    assert!(
        vector
            .iter()
            .all(|value| value.as_f64().is_some_and(f64::is_finite))
    );
}

#[test]
fn lattice_serve_missing_embedding_directory_fails_after_generation_load() {
    let _gpu = acquire_gpu_lock();
    let root = tempfile::tempdir().expect("temporary fixture root");
    let generation = generation_checkpoint(root.path());
    let missing = root.path().join("missing-embedding");
    let run = run_startup(
        Binary::LatticeServe,
        Some(&generation),
        &args(&["--embedding-model", missing.to_str().expect("UTF-8 path")]),
    );

    assert_failure(
        &run,
        root.path(),
        1,
        "lattice_serve: embedding model directory not found: <TEMP>/missing-embedding",
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP>/generation (bf16) ...",
            "[lattice_serve] model 'generation' (bf16) ready (context=1024, from model max_position_embeddings)",
        ],
    );
}

#[test]
fn lattice_serve_pooling_validation_runs_after_directory_and_model_load() {
    let _gpu = acquire_gpu_lock();
    let root = tempfile::tempdir().expect("temporary fixture root");
    let generation = generation_checkpoint(root.path());
    let embedding = bert_checkpoint(root.path());
    let invalid_pool = run_startup(
        Binary::LatticeServe,
        Some(&generation),
        &args(&[
            "--embedding-model",
            embedding.to_str().expect("UTF-8 path"),
            "--embedding-pooling",
            "max",
        ]),
    );
    assert_failure(
        &invalid_pool,
        root.path(),
        1,
        "lattice_serve: invalid --embedding-pooling 'max'; expected 'mean' or 'cls'",
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP>/generation (bf16) ...",
            "[lattice_serve] model 'generation' (bf16) ready (context=1024, from model max_position_embeddings)",
        ],
    );
    assert_server_markers(
        &invalid_pool.stderr,
        root.path(),
        0,
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP>/generation (bf16) ...",
            "[lattice_serve] model 'generation' (bf16) ready (context=1024, from model max_position_embeddings)",
            "[lattice_serve] loading embedding model from <TEMP>/bert-fixture ...",
        ],
    );
    assert!(
        !invalid_pool
            .stderr
            .contains("embedding model 'bert-fixture' ready")
    );

    let missing = root.path().join("missing-bert");
    let missing_with_invalid_pool = run_startup(
        Binary::LatticeServe,
        Some(&generation),
        &args(&[
            "--embedding-model",
            missing.to_str().expect("UTF-8 path"),
            "--embedding-pooling",
            "max",
        ]),
    );
    assert_failure(
        &missing_with_invalid_pool,
        root.path(),
        1,
        "lattice_serve: embedding model directory not found: <TEMP>/missing-bert",
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP>/generation (bf16) ...",
            "[lattice_serve] model 'generation' (bf16) ready (context=1024, from model max_position_embeddings)",
        ],
    );
    assert_server_markers(
        &missing_with_invalid_pool.stderr,
        root.path(),
        0,
        &[
            "[route] selected family=qwen35 backend=metal format=safetensors",
            "[lattice_serve] loading model from <TEMP>/generation (bf16) ...",
            "[lattice_serve] model 'generation' (bf16) ready (context=1024, from model max_position_embeddings)",
        ],
    );
    assert!(
        !missing_with_invalid_pool
            .stderr
            .contains("[lattice_serve] loading embedding model from ")
    );
}

#[test]
fn lattice_router_startup_errors_follow_successful_generation_load() {
    let _gpu = acquire_gpu_lock();
    let root = tempfile::tempdir().expect("temporary fixture root");
    let generation = generation_checkpoint(root.path());
    let missing_router = root.path().join("missing-router");
    let missing_state = run_startup(
        Binary::Lattice,
        Some(&generation),
        &args(&[
            "--router-state",
            missing_router.to_str().expect("UTF-8 path"),
        ]),
    );
    assert_failure(
        &missing_state,
        root.path(),
        1,
        "Error: --router-state <TEMP>/missing-router holds no router artifact. Omit the flag to serve without a router, or point it at a directory holding one",
        &[
            "Loading model from <TEMP>/generation...",
            "[route] selected family=qwen35 backend=cpu format=safetensors",
            "Model loaded. Serving as 'generation'.",
            "Embeddings disabled (<TEMP>/generation): <TEMP>/generation has no vision_config; not a vision-language checkpoint",
        ],
    );

    let pin_without_state = run_startup(
        Binary::Lattice,
        Some(&generation),
        &args(&["--router-pin", "1"]),
    );
    assert_failure(
        &pin_without_state,
        root.path(),
        1,
        "Error: --router-pin 1 was given without --router-state; a pinned version names an artifact in a directory, and no directory was configured",
        &[
            "Loading model from <TEMP>/generation...",
            "[route] selected family=qwen35 backend=cpu format=safetensors",
            "Model loaded. Serving as 'generation'.",
            "Embeddings disabled (<TEMP>/generation): <TEMP>/generation has no vision_config; not a vision-language checkpoint",
        ],
    );
}
