//! QuaRot+Q4 composed-path golden regression gate (issue #320).
//!
//! **Purpose**: `crates/inference/src/quant/quarot/forward_equivalence.rs`
//! gates the converter itself (pre-quant f64 vs post-rotation-fusion,
//! `1e-5` tolerance) but nothing previously exercised the actual runtime
//! consumer of a QuaRot+Q4 artifact —
//! [`lattice_inference::forward::metal_qwen35::MetalQwen35State::from_q4_dir`]
//! followed by greedy decode. This test closes that gap by loading a real
//! rotated Q4 artifact through the production loader and comparing greedy
//! token output against a frozen lattice-self golden.
//!
//! **Tolerance**: exact first-N greedy token ID agreement, NOT a logit
//! tolerance. The golden is generated FROM the Q4 artifact itself, so Q4
//! quantization error and the QuaRot rotation are already baked into the
//! expected tokens — a `1e-5`-style float tolerance would be meaningless
//! here (Q4 per-weight error is on the order of half a quantization step
//! plus f16 scale/bias rounding, several orders above `1e-5`). Greedy
//! decoding is deterministic and quant-robust, so exact token-ID equality
//! is the honest gate.
//!
//! **Fail-closed contract**: locally, with `LATTICE_QUAROT_Q4_DIR` /
//! `LATTICE_MODEL_DIR` unset or pointing at missing paths, this test prints
//! a skip line and returns — most dev machines do not have an ephemeral
//! QuaRot Q4 artifact lying around. When `LATTICE_Q4_COMPOSED_GATE_ENFORCE=1`
//! is set (the dedicated CI job sets this), a missing model/Q4 dir panics
//! instead of skipping, so a provisioning failure cannot masquerade as a
//! passing gate. Mirrors `LATTICE_DRIFT_GATE_ENFORCE` in
//! `crates/embed/tests/embed_drift_baseline.rs`.
//!
//! **How to regenerate the golden** (deliberate, reviewable, never done by CI):
//!   ```bash
//!   python3 scripts/gen_quarot_q4_composed_golden.py \
//!       --model-dir ~/.lattice/models/qwen3.5-0.8b \
//!       --q4-dir target/quarot-q4-golden/qwen3.5-0.8b-quarot-q4 \
//!       --seed 0xCAFE_BABE_DEAD_BEEF \
//!       --update-golden
//!   ```
//!
//! **Run this test** (`LATTICE_QUAROT_Q4_DIR` must be absolute — `cargo test`
//! runs test binaries with the crate directory as CWD, not the workspace root):
//!   ```bash
//!   LATTICE_MODEL_DIR=~/.lattice/models/qwen3.5-0.8b \
//!   LATTICE_QUAROT_Q4_DIR="$(pwd)/target/quarot-q4-golden/qwen3.5-0.8b-quarot-q4" \
//!   cargo test --release -p lattice-inference --test quarot_q4_composed_golden \
//!       --features "f16,metal-gpu" -- --nocapture
//!   ```

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Single generation case in the fixture: a prompt plus its frozen expected
/// greedy token IDs.
#[derive(Deserialize)]
struct GoldenCase {
    name: String,
    #[allow(dead_code)]
    prompt: String,
    expected_generated_ids: Vec<u32>,
}

/// Fixture format written by `dump_quarot_q4_golden` / `gen_quarot_q4_composed_golden.py`.
#[derive(Deserialize)]
struct Golden {
    schema_version: u32,
    #[allow(dead_code)]
    model_id: String,
    #[allow(dead_code)]
    quarot_seed: String,
    max_new_tokens: usize,
    cases: Vec<GoldenCase>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("quarot_q4_composed_v1")
        .join("qwen35_0_8b_quarot_q4_greedy_tokens.json")
}

fn load_golden() -> Golden {
    let path = fixture_path();
    assert!(
        path.exists(),
        "committed golden not found at {} — run scripts/gen_quarot_q4_composed_golden.py \
         and commit the output",
        path.display()
    );
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let golden: Golden = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("bad JSON in {}: {e}", path.display()));
    assert_eq!(
        golden.schema_version, 1,
        "unsupported quarot_q4_composed_v1 schema_version {}",
        golden.schema_version
    );
    assert!(
        !golden.cases.is_empty(),
        "golden fixture at {} has no cases",
        path.display()
    );
    golden
}

/// Whether the CI gate must fail closed instead of skipping on missing artifacts.
fn enforce() -> bool {
    std::env::var("LATTICE_Q4_COMPOSED_GATE_ENFORCE").is_ok()
}

/// Reads an env var naming a directory. Returns `None` (and prints a skip
/// line) when unset or the path is missing, UNLESS `enforce()` is true, in
/// which case it panics — a missing artifact under the enforced CI gate
/// means provisioning failed, and a silent skip there would verify nothing.
fn require_env_dir(name: &str) -> Option<PathBuf> {
    match std::env::var(name) {
        Ok(value) => {
            let path = PathBuf::from(shellexpand_home(&value));
            if path.exists() {
                Some(path)
            } else if enforce() {
                panic!(
                    "{name}={} does not exist, and LATTICE_Q4_COMPOSED_GATE_ENFORCE=1 — \
                     the composed-path gate must fail closed on missing artifacts",
                    path.display()
                );
            } else {
                eprintln!(
                    "LATTICE_Q4_COMPOSED_SKIPPED test=quarot_q4_composed_greedy_tokens_match_lattice_self_golden \
                     reason=missing_path {name}={}",
                    path.display()
                );
                None
            }
        }
        Err(_) if enforce() => {
            panic!("{name} must be set when LATTICE_Q4_COMPOSED_GATE_ENFORCE=1")
        }
        Err(_) => {
            eprintln!(
                "LATTICE_Q4_COMPOSED_SKIPPED test=quarot_q4_composed_greedy_tokens_match_lattice_self_golden \
                 reason=unset_env {name}"
            );
            None
        }
    }
}

/// Minimal `~` expansion so `LATTICE_MODEL_DIR=~/.lattice/models/...` works
/// the same way it does when a user types it at a shell prompt.
fn shellexpand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}/{rest}");
    }
    path.to_string()
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn run_composed_gate(model_dir: &Path, q4_dir: &Path, golden: &Golden) {
    use lattice_inference::forward::metal_qwen35::MetalQwen35State;
    use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
    use lattice_inference::tokenizer::bpe::BpeTokenizer;

    let cfg = Qwen35Config::from_config_json(&q4_dir.join("config.json"))
        .unwrap_or_else(|e| panic!("failed to parse {}/config.json: {e}", q4_dir.display()));
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path).unwrap_or_else(|e| {
        panic!(
            "failed to load tokenizer from {}: {e}",
            tokenizer_path.display()
        )
    });

    let mut metal = MetalQwen35State::from_q4_dir(q4_dir, &tokenizer_path, &cfg, 4096)
        .unwrap_or_else(|e| panic!("from_q4_dir({}): {e}", q4_dir.display()));

    let gen_cfg = GenerateConfig {
        max_new_tokens: golden.max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let mut mismatches = 0;
    for case in &golden.cases {
        let out = metal.generate(&case.prompt, &tokenizer, &gen_cfg);
        if out.token_ids != case.expected_generated_ids {
            mismatches += 1;
            eprintln!(
                "QUAROT_Q4_COMPOSED_MISMATCH case={} prompt={:?}\n  expected={:?}\n  actual=  {:?}",
                case.name, case.prompt, case.expected_generated_ids, out.token_ids
            );
        }
        assert_eq!(
            out.token_ids,
            case.expected_generated_ids,
            "QuaRot Q4 composed-path greedy token mismatch for case '{}' — the rotated+Q4 \
             forward path diverged from the frozen lattice-self golden at {}",
            case.name,
            fixture_path().display()
        );
    }
    assert_eq!(
        mismatches,
        0,
        "composed golden mismatches across {} cases",
        golden.cases.len()
    );
}

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn run_composed_gate(_model_dir: &Path, _q4_dir: &Path, _golden: &Golden) {
    if enforce() {
        panic!(
            "LATTICE_Q4_COMPOSED_GATE_ENFORCE=1 but this build lacks macOS + metal-gpu — \
             the composed path is Metal-only, so the enforced CI job must run on macOS \
             with --features metal-gpu"
        );
    }
    eprintln!(
        "LATTICE_Q4_COMPOSED_SKIPPED test=quarot_q4_composed_greedy_tokens_match_lattice_self_golden \
         reason=metal_gpu_unavailable"
    );
}

/// Fixture-only check: the committed golden parses, has the schema this
/// test expects, and every case carries at least one expected token. Runs
/// unconditionally (no model/Metal required) so a malformed fixture is
/// caught even on non-macOS CI.
#[test]
fn quarot_q4_composed_golden_fixture_is_valid() {
    let golden = load_golden();
    for case in &golden.cases {
        assert!(
            !case.expected_generated_ids.is_empty(),
            "case '{}' has no expected_generated_ids",
            case.name
        );
        assert!(
            case.expected_generated_ids.len() <= golden.max_new_tokens,
            "case '{}' has {} expected token IDs but max_new_tokens is {}",
            case.name,
            case.expected_generated_ids.len(),
            golden.max_new_tokens
        );
    }
}

/// The composed-path gate: real QuaRot+Q4 artifact -> `from_q4_dir` ->
/// greedy `generate` -> exact token-ID match against the frozen golden.
#[test]
fn quarot_q4_composed_greedy_tokens_match_lattice_self_golden() {
    let golden = load_golden();

    let Some(model_dir) = require_env_dir("LATTICE_MODEL_DIR") else {
        return;
    };
    let Some(q4_dir) = require_env_dir("LATTICE_QUAROT_Q4_DIR") else {
        return;
    };

    run_composed_gate(&model_dir, &q4_dir, &golden);
}
