//! ADR-082 stage 5 e2e gate: real `google/gemma-4-E2B-it` checkpoint,
//! CPU-only, greedy-decode a fixed text prompt and compare against the
//! committed HF differential golden (`scripts/gemma4_stage5_e2e_golden.py`).
//!
//! **Fail-closed contract** (mirrors `vision_s5b_e2e_gate_test.rs`):
//! `LATTICE_GEMMA4_MODEL_DIR` (default `~/.lattice/models/gemma-4-e2b-it`)
//! points at the downloaded checkpoint. With the checkpoint missing, this
//! test prints a skip line and returns; with `LATTICE_GEMMA4_GATE_ENFORCE=1`
//! a missing checkpoint panics instead of skipping. Requires the `f16`
//! cargo feature (the checkpoint's decoder weights are BF16 in the
//! safetensors file).
//!
//! Run:
//! ```bash
//! cargo test --release -p lattice-inference --features f16 \
//!     --test gemma4_e2e_forward_test -- --nocapture
//! ```

use lattice_inference::Tokenizer as _;
use lattice_inference::model::gemma4_model::Gemma4Model;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct GoldenFixture {
    bos_token_id: u32,
    prompt_token_ids: Vec<u32>,
    input_ids: Vec<u32>,
    greedy_tokens: Vec<u32>,
    probe_layers: Vec<usize>,
    hidden_state_last_pos: std::collections::HashMap<String, Vec<f32>>,
    final_logits_last_pos_top8: Top8,
}

#[derive(Deserialize)]
struct Top8 {
    indices: Vec<u32>,
    values: Vec<f32>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
        .join("stage5")
        .join("e2e_golden.json")
}

fn load_golden() -> GoldenFixture {
    let bytes = std::fs::read(fixture_path()).expect("reading committed stage5 golden fixture");
    serde_json::from_slice(&bytes).expect("parsing stage5 golden fixture")
}

/// Fixture-only check: parses and has the expected shape. Runs
/// unconditionally, no checkpoint required.
#[test]
fn stage5_golden_fixture_is_valid() {
    let golden = load_golden();
    assert_eq!(golden.probe_layers, vec![0, 4, 15, 34]);
    assert_eq!(golden.greedy_tokens.len(), 3);
    assert_eq!(golden.input_ids.first().copied(), Some(golden.bos_token_id));
    assert_eq!(
        golden.input_ids[1..],
        golden.prompt_token_ids[..],
        "input_ids must be exactly [bos] + prompt_token_ids"
    );
    for layer in &golden.probe_layers {
        let hs = golden
            .hidden_state_last_pos
            .get(&layer.to_string())
            .unwrap_or_else(|| panic!("missing hidden_state_last_pos for layer {layer}"));
        assert_eq!(
            hs.len(),
            1536,
            "layer {layer} hidden state must be hidden_size-wide"
        );
        assert!(hs.iter().all(|v| v.is_finite()));
    }
    assert_eq!(golden.final_logits_last_pos_top8.indices.len(), 8);
    assert_eq!(golden.final_logits_last_pos_top8.values.len(), 8);
}

fn enforce() -> bool {
    std::env::var("LATTICE_GEMMA4_GATE_ENFORCE").as_deref() == Ok("1")
}

fn shellexpand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}/{rest}");
    }
    path.to_string()
}

fn resolve_model_dir() -> Option<PathBuf> {
    const VAR: &str = "LATTICE_GEMMA4_MODEL_DIR";
    let raw = std::env::var(VAR).unwrap_or_else(|_| "~/.lattice/models/gemma-4-e2b-it".to_string());
    let path = PathBuf::from(shellexpand_home(&raw));
    if path.join("model.safetensors").exists() {
        Some(path)
    } else if enforce() {
        panic!(
            "{VAR}={} has no model.safetensors, and LATTICE_GEMMA4_GATE_ENFORCE=1 -- the \
             stage-5 e2e gate must fail closed on a missing checkpoint",
            path.display()
        );
    } else {
        eprintln!(
            "LATTICE_GEMMA4_E2E_SKIPPED reason=missing_checkpoint path={}",
            path.display()
        );
        None
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[cfg(feature = "f16")]
fn run_gate(model_dir: &Path) {
    let golden = load_golden();
    let model =
        Gemma4Model::from_safetensors(model_dir).expect("loading real gemma-4-e2b-it checkpoint");

    // Independently tokenize the prompt through the lattice Gemma tokenizer
    // and prepend BOS, asserting it matches the golden's input_ids exactly
    // -- a free stage-1 tokenizer parity cross-check alongside the stage-5
    // forward gate.
    let metadata_prompt = "The capital of France is";
    let tok = model.tokenizer().tokenize(metadata_prompt);
    let mut input_ids = vec![golden.bos_token_id];
    input_ids.extend_from_slice(&tok.input_ids[..tok.real_length]);
    assert_eq!(
        input_ids, golden.input_ids,
        "lattice GemmaBpeTokenizer must tokenize the fixture's prompt identically to HF"
    );

    let (greedy, _final_logits, probe) = model
        .generate_greedy_with_probe(&golden.input_ids, 3, 64, &golden.probe_layers)
        .expect("gemma4 greedy generate with probe");

    // Gate 3: e2e greedy token parity, exact (CPU-vs-CPU f32).
    assert_eq!(
        greedy, golden.greedy_tokens,
        "greedy first-3 tokens must match the HF golden exactly"
    );

    // Gate 4: per-layer hidden-state trace at the last prompt position.
    assert_eq!(probe.len(), golden.probe_layers.len());
    for (layer, hidden) in &probe {
        let expected = &golden.hidden_state_last_pos[&layer.to_string()];
        let diff = max_abs_diff(hidden, expected);
        // Bounded f32 parity, not bit-for-bit (BF16-origin weights widened
        // to f32, plus lattice's own matmul/RMSNorm accumulation order vs
        // HF's torch -- same class of bound as gemma4_ops's rms_norm_wide
        // golden). Measured on the reference machine (release build): layer
        // 0 ~2e-6, layer 4 ~5e-6, layer 15 ~1.3e-5, layer 34 ~9e-6 (all four
        // captured via a forward hook directly on the decoder layer, not
        // `output_hidden_states`'s tuple -- see
        // scripts/gemma4_stage5_e2e_golden.py's module docs for why the
        // tuple's own indexing is unsafe to rely on at the last layer).
        // 1e-3 gives roughly two orders of magnitude of headroom over the
        // measured value at every probed layer without being loose enough
        // to mask a real per-layer divergence.
        let tol = 1e-3;
        assert!(
            diff <= tol,
            "layer {layer} hidden-state max-abs-diff {diff} exceeds tolerance {tol}"
        );
    }
}

#[cfg(not(feature = "f16"))]
fn run_gate(_model_dir: &Path) {
    if enforce() {
        panic!(
            "LATTICE_GEMMA4_GATE_ENFORCE=1 but the `f16` feature is not enabled -- the real \
             BF16 checkpoint cannot be loaded without it"
        );
    }
    eprintln!("LATTICE_GEMMA4_E2E_SKIPPED reason=f16_feature_disabled");
}

#[test]
fn stage5_e2e_greedy_and_per_layer_probe_match_hf_golden() {
    let Some(model_dir) = resolve_model_dir() else {
        return;
    };
    run_gate(&model_dir);
}
