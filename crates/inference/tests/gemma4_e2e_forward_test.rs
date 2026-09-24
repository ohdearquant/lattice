//! ADR-082 stage 5 e2e gate: real `google/gemma-4-E2B-it` checkpoint,
//! CPU-only, greedy-decode a fixed text prompt and compare against the
//! committed HF differential golden (`scripts/gemma4_stage5_e2e_golden.py`).
//!
//! **Fail-closed contract** (inverted from `vision_s5b_e2e_gate_test.rs`'s
//! default-skip/opt-in-enforce shape on purpose -- a prior round of this gate
//! defaulted to skip-as-green on a missing checkpoint, observed passing in
//! 0.00s against a bogus `LATTICE_GEMMA4_MODEL_DIR`). `LATTICE_GEMMA4_MODEL_DIR`
//! (default `~/.lattice/models/gemma-4-e2b-it`) points at the downloaded
//! checkpoint. A missing checkpoint or missing `f16` feature **panics** by
//! default; only `LATTICE_GEMMA4_GATE_SKIP=1` downgrades that to a skip, and
//! even then an unmistakable `LATTICE_GEMMA4_E2E_SKIPPED` marker is printed.
//! General-purpose CI/dev test runs (`cargo test --workspace`, without this
//! gate as their target) must set `LATTICE_GEMMA4_GATE_SKIP=1` explicitly
//! (see `scripts/ci.sh`) -- an unset environment is read as "this run means
//! to exercise the real gate."
//!
//! Run:
//! ```bash
//! cargo test --release -p lattice-inference --features f16 \
//!     --test gemma4_e2e_forward_test -- --nocapture
//! ```

#[cfg(feature = "f16")]
use lattice_inference::GenerateConfig;
#[cfg(feature = "f16")]
use lattice_inference::Tokenizer as _;
#[cfg(feature = "f16")]
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

/// Sliding-window boundary coverage: each case is a
/// synthetic prompt of exactly `length` tokens, built by cycling the main
/// golden's own (tokenizer-verified) prompt ids -- see
/// `scripts/gemma4_stage5_e2e_golden.py`'s `BOUNDARY_LENGTHS` docs for why
/// 511/512/513 exercise "not yet at the window", "exactly window-sized",
/// and "must evict token 0" respectively.
#[derive(Deserialize)]
struct BoundaryFixture {
    cases: Vec<BoundaryCase>,
}

#[derive(Deserialize)]
struct BoundaryCase {
    length: usize,
    input_ids: Vec<u32>,
    final_logits_last_pos_top8: Top8,
    greedy_tokens: Vec<u32>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
        .join("stage5")
        .join("e2e_golden.json")
}

fn boundary_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
        .join("stage5")
        .join("boundary_golden.json")
}

fn load_golden() -> GoldenFixture {
    let bytes = std::fs::read(fixture_path()).expect("reading committed stage5 golden fixture");
    serde_json::from_slice(&bytes).expect("parsing stage5 golden fixture")
}

fn load_boundary_golden() -> BoundaryFixture {
    let bytes =
        std::fs::read(boundary_fixture_path()).expect("reading committed boundary golden fixture");
    serde_json::from_slice(&bytes).expect("parsing boundary golden fixture")
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

/// Explicit opt-out: the only way to turn a missing checkpoint (or missing
/// `f16` feature) into a skip instead of a panic. `LATTICE_GEMMA4_GATE_ENFORCE=1`
/// is still accepted for the documented command above but no longer changes
/// behavior -- enforcement is the default now, not an opt-in.
fn skip_allowed() -> bool {
    std::env::var("LATTICE_GEMMA4_GATE_SKIP").as_deref() == Ok("1")
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
    } else if skip_allowed() {
        eprintln!(
            "LATTICE_GEMMA4_E2E_SKIPPED reason=missing_checkpoint path={}",
            path.display()
        );
        None
    } else {
        panic!(
            "{VAR}={} has no model.safetensors -- the stage-5 e2e gate fails closed by \
             default on a missing checkpoint. Set LATTICE_GEMMA4_GATE_SKIP=1 to explicitly \
             skip (only for general-purpose test runs that are not targeting this gate).",
            path.display()
        );
    }
}

#[cfg(feature = "f16")]
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

    let (greedy, final_logits, probe) = model
        .generate_greedy_with_probe(&golden.input_ids, 3, 64, &golden.probe_layers)
        .expect("gemma4 greedy generate with probe");

    // Gate 3: e2e greedy token parity, exact (CPU-vs-CPU f32).
    assert_eq!(
        greedy, golden.greedy_tokens,
        "greedy first-3 tokens must match the HF golden exactly"
    );

    // Gate 3b: final post-softcap logits top-8 parity at the prompt's last
    // position. `generate_greedy_with_probe` returns the last computed
    // logits vector, which after the prompt-prefill loop is exactly the
    // prompt's-last-position logits used to pick the golden's first greedy
    // token (mirrors the golden script's `final_logits_last_pos_top8`,
    // computed before any generated token is produced). Compares indices
    // exactly and values within a tolerance, so a regression in the final
    // RMSNorm/tied-output-projection/soft-cap that happens to preserve the
    // wide-margin argmax token cannot hide behind the pre-final-norm probes
    // above.
    let top8_tol = 1e-2;
    for (rank, &expected_idx) in golden.final_logits_last_pos_top8.indices.iter().enumerate() {
        let expected_val = golden.final_logits_last_pos_top8.values[rank];
        let actual_val = final_logits[expected_idx as usize];
        assert!(
            (actual_val - expected_val).abs() <= top8_tol,
            "final logits top-8 rank {rank} (vocab index {expected_idx}): lattice value \
             {actual_val} vs HF golden {expected_val}, diff exceeds tolerance {top8_tol}"
        );
    }
    let mut ranked: Vec<(usize, f32)> = final_logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite logits"));
    let top8_indices: Vec<u32> = ranked[..8].iter().map(|&(i, _)| i as u32).collect();
    assert_eq!(
        top8_indices, golden.final_logits_last_pos_top8.indices,
        "lattice's own top-8 ranking of final logits must match the HF golden's top-8 indices"
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

#[cfg(feature = "f16")]
fn run_boundary_gate(model_dir: &Path) {
    let boundary = load_boundary_golden();
    let model =
        Gemma4Model::from_safetensors(model_dir).expect("loading real gemma-4-e2b-it checkpoint");

    for case in &boundary.cases {
        assert_eq!(case.input_ids.len(), case.length);
        let (greedy, final_logits, _probe) = model
            .generate_greedy_with_probe(&case.input_ids, 2, case.length + 4, &[])
            .expect("gemma4 greedy generate at sliding-window boundary length");

        assert_eq!(
            greedy, case.greedy_tokens,
            "length {}: greedy tokens after the boundary must match the HF golden exactly",
            case.length
        );

        let top8_tol = 1e-2;
        for (rank, &expected_idx) in case.final_logits_last_pos_top8.indices.iter().enumerate() {
            let expected_val = case.final_logits_last_pos_top8.values[rank];
            let actual_val = final_logits[expected_idx as usize];
            assert!(
                (actual_val - expected_val).abs() <= top8_tol,
                "length {} top-8 rank {rank} (vocab index {expected_idx}): lattice value \
                 {actual_val} vs HF golden {expected_val}, diff exceeds tolerance {top8_tol}",
                case.length
            );
        }
        let mut ranked: Vec<(usize, f32)> = final_logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite logits"));
        let top8_indices: Vec<u32> = ranked[..8].iter().map(|&(i, _)| i as u32).collect();
        assert_eq!(
            top8_indices, case.final_logits_last_pos_top8.indices,
            "length {}: lattice's own top-8 ranking must match the HF golden's top-8 indices",
            case.length
        );
    }
}

#[cfg(not(feature = "f16"))]
fn run_boundary_gate(_model_dir: &Path) {
    if skip_allowed() {
        eprintln!("LATTICE_GEMMA4_BOUNDARY_SKIPPED reason=f16_feature_disabled");
        return;
    }
    panic!(
        "a gemma-4-e2b-it checkpoint is present but the `f16` feature is not enabled -- this \
         gate's pinned invocation contract requires --features f16. Set \
         LATTICE_GEMMA4_GATE_SKIP=1 to explicitly skip (only for general-purpose test runs that \
         are not targeting this gate)."
    );
}

#[cfg(not(feature = "f16"))]
fn run_gate(_model_dir: &Path) {
    if skip_allowed() {
        eprintln!("LATTICE_GEMMA4_E2E_SKIPPED reason=f16_feature_disabled");
        return;
    }
    panic!(
        "a gemma-4-e2b-it checkpoint is present but the `f16` feature is not enabled -- this \
         gate's pinned invocation contract requires --features f16. Set \
         LATTICE_GEMMA4_GATE_SKIP=1 to explicitly skip (only for general-purpose test runs that \
         are not targeting this gate)."
    );
}

#[test]
fn stage5_e2e_greedy_and_per_layer_probe_match_hf_golden() {
    let Some(model_dir) = resolve_model_dir() else {
        return;
    };
    run_gate(&model_dir);
}

#[cfg(feature = "f16")]
fn run_shared_driver_gate(model_dir: &Path) {
    let golden = load_golden();
    let model =
        Gemma4Model::from_safetensors(model_dir).expect("loading real gemma-4-e2b-it checkpoint");

    // `stop_token_ids: vec![]` overrides `GenerateConfig::default()`'s
    // `QWEN_CHAT_IM_END_TOKEN_ID` (248_046): Gemma's vocab_size (262_144) is
    // large enough that this Qwen-specific id is a valid, unrelated Gemma
    // token id, so leaving the default in place risks an early, semantically
    // meaningless stop unrelated to this gate's own greedy-token-parity
    // question (see `Gemma4Model::generate`'s own doc comment on this
    // landmine).
    // `GenerateConfig` is `#[non_exhaustive]` (`crates/inference/src/generation.rs:31`):
    // this file compiles as a separate integration-test crate, so a struct-literal
    // construction (including `..Default::default()` functional-update syntax, which
    // is still struct-literal syntax) is rejected at that boundary (E0639). Starting
    // from `GenerateConfig::default()` and assigning the fields this gate needs is the
    // only construction this crate boundary allows.
    let mut gen_cfg = GenerateConfig::default();
    gen_cfg.max_new_tokens = 3;
    gen_cfg.temperature = 0.0;
    gen_cfg.repetition_penalty = 1.0;
    gen_cfg.stop_token_ids = vec![];
    let output = model
        .generate(&golden.input_ids, &gen_cfg)
        .expect("gemma4 shared-driver generate");

    // Gate 3, replayed through the shared decoder driver's public entry
    // point: identical assertion to `run_gate`'s own greedy-token check
    // above, over the SAME golden and the SAME real checkpoint, but through
    // `Gemma4Model::generate` (ADR-090 row R04) instead of the fixed-count
    // `generate_greedy_with_probe` diagnostic. `output.stopped` is asserted
    // `false` first: if the run stopped early (e.g. one of the golden's 3
    // greedy tokens happens to equal this checkpoint's real
    // `eos_token_id` -- a real, checkpoint-dependent possibility `generate`'s
    // EOS-aware contract introduces that the diagnostic path above never had
    // to consider), `token_ids` would be shorter than 3 and the length
    // assertion below would fail with a confusing message; asserting
    // `!stopped` first names the actual condition directly.
    assert!(
        !output.stopped,
        "the shared-driver path must not stop before 3 tokens for this golden prompt (stop \
         reason: {:?}) -- if this fires, one of the golden's 3 greedy tokens likely equals \
         this checkpoint's real eos_token_id, a real EOS-aware-vs-fixed-count behavioral \
         difference from generate_greedy_with_probe that needs its own investigation, not a \
         silently loosened assertion here",
        output.stop_reason
    );
    assert_eq!(
        output.token_ids, golden.greedy_tokens,
        "shared-driver greedy tokens must match the HF golden exactly"
    );
}

#[cfg(not(feature = "f16"))]
fn run_shared_driver_gate(_model_dir: &Path) {
    if skip_allowed() {
        eprintln!("LATTICE_GEMMA4_E2E_SKIPPED reason=f16_feature_disabled");
        return;
    }
    panic!(
        "a gemma-4-e2b-it checkpoint is present but the `f16` feature is not enabled -- this \
         gate's pinned invocation contract requires --features f16. Set \
         LATTICE_GEMMA4_GATE_SKIP=1 to explicitly skip (only for general-purpose test runs that \
         are not targeting this gate)."
    );
}

/// ADR-090 row R04 (#1597): the same stage-5 golden, replayed through the shared decoder
/// driver's public entry point (`Gemma4Model::generate`) instead of the fixed-count
/// `generate_greedy_with_probe` diagnostic path
/// [`stage5_e2e_greedy_and_per_layer_probe_match_hf_golden`] above exercises. Deliberately not a
/// new file/module: both arms share one checkpoint resolution and skip/enforce convention. Does
/// not touch, weaken, or duplicate that test's per-layer-probe or final-logits-top-8 assertions.
///
/// The `DriverTrace` opened/consumed assertion lives in `model::gemma4_model`'s own
/// `#[cfg(test)]` unit tests, not here: `generate_with_trace` and `DriverTrace` are `pub(crate)`
/// by design, and this file compiles as a separate integration-test crate that can only see
/// `lattice_inference`'s `pub` surface. `model::qwen35`'s analogous driver-trace test hits the
/// same crate-boundary constraint and resolves it the same way.
#[test]
fn stage5_shared_driver_greedy_matches_hf_golden() {
    let Some(model_dir) = resolve_model_dir() else {
        return;
    };
    run_shared_driver_gate(&model_dir);
}

/// Fixture-only check: the boundary golden parses and every case's
/// `input_ids` length matches its declared `length`. Runs unconditionally,
/// no checkpoint required -- mirrors `stage5_golden_fixture_is_valid`.
#[test]
fn boundary_golden_fixture_is_valid() {
    let boundary = load_boundary_golden();
    assert_eq!(
        boundary.cases.iter().map(|c| c.length).collect::<Vec<_>>(),
        vec![511, 512, 513]
    );
    for case in &boundary.cases {
        assert_eq!(case.input_ids.len(), case.length);
        assert_eq!(case.final_logits_last_pos_top8.indices.len(), 8);
        assert_eq!(case.final_logits_last_pos_top8.values.len(), 8);
        assert!(!case.greedy_tokens.is_empty());
    }
}

/// ADR-082 stage 5: real-checkpoint sliding-window
/// boundary coverage at prompt lengths 511/512/513, plus a decode step past
/// the boundary (`greedy_tokens` in the fixture covers 2 new tokens).
/// Same fail-closed contract as
/// [`stage5_e2e_greedy_and_per_layer_probe_match_hf_golden`].
#[test]
fn stage5_sliding_window_boundary_matches_hf_golden() {
    let Some(model_dir) = resolve_model_dir() else {
        return;
    };
    run_boundary_gate(&model_dir);
}
