//! Pre-migration CPU greedy golden gate (ADR-090 rollout row R03).
//!
//! **Why this exists**: R03 moves the canonical Qwen CPU decode onto the shared
//! private driver. The only way to show the move did not change behaviour is to
//! freeze what the *pre-migration* route produced, from the unchanged entry
//! point and checkpoint, before the driver exists. A golden taken after the
//! move would be the new route agreeing with itself.
//!
//! **What is frozen**: exact greedy token IDs for two fixed prompts (one short,
//! one long) from [`lattice_inference::model::qwen35::Qwen35Model::generate`],
//! loaded by `from_safetensors` from unquantized bf16 weights under the `f16`
//! feature. Exact ID equality, not a float tolerance: greedy decode is
//! deterministic, and the capture was verified reproducible across repeat runs.
//!
//! **Each prompt is frozen twice, and the second capture is the one that took
//! work to justify.** The original pair pinned `reasoning_budget: null`, and
//! `effective_reasoning_budget` returns `None` whenever the budget is unset, so
//! `DecodePolicy::apply_override` never executed in either case: a migration
//! that got the candidate-to-final override wrong would have reproduced those
//! ids exactly. The `_reasoning_budget_4` cases close that hole by replaying the
//! same prompts with a budget, which forces `</think>` at the budget index and
//! puts the separate reasoning/answer accounting on the measured path. A
//! control asserts each budgeted case agrees with its unbudgeted pair up to the
//! budget and disagrees at it, because a budgeted case that matched its pair
//! would look like coverage while proving nothing.
//!
//! **Not interchangeable with the QuaRot Q4 golden.** `quarot_q4_composed_golden`
//! freezes tokens generated *from a rotated Q4 artifact* through
//! `MetalQwen35State::from_q4_dir`, so quantization error and the rotation are
//! baked into its expected IDs. Different loader, different numerical path.
//! (The two do agree on the `short_factual` prefix, which is corroboration that
//! the capture is real, not a reason to treat either as a substitute.)
//!
//! **Fail-closed contract, and it is the INVERSE of the other gates in this
//! crate.** `quarot_q4_composed_golden` and `embed_drift_baseline` default to a
//! printed skip line and enforce only when an opt-in env var is set. That shape
//! fails toward looking safe: a machine without the checkpoint prints "skipping"
//! and the suite reports green. This gate enforces BY DEFAULT — an absent
//! checkpoint, an absent `f16` feature, or an attempted bypass is a panic, never
//! a skip. There is no env var that turns the gate off; see
//! `refuse_bypass_or_panic`.
//!
//! **How CI stays green without a skip line**: this target is declared
//! `test = false` in `crates/inference/Cargo.toml`, so `cargo test --workspace`
//! does not run it. Exclusion is a visible manifest declaration a reader can
//! grep, not a runtime branch that renders as a pass. Invoking the target at all
//! means enforcement.
//!
//! **Run it**:
//!   ```bash
//!   LATTICE_CPU_GREEDY_MODEL_DIR=/abs/path/to/qwen3.5-0.8b \
//!   cargo test --release -p lattice-inference \
//!       --test cpu_pre_migration_greedy_golden --features f16 -- --nocapture
//!   ```
//!   The path must be absolute: `cargo test` runs test binaries with the crate
//!   directory as CWD, not the workspace root.
//!
//! **How to regenerate** (deliberate, reviewable, never done by CI):
//!   ```bash
//!   python3 scripts/gen_cpu_pre_migration_greedy_golden.py \
//!       --model-dir /abs/path/to/qwen3.5-0.8b --update-golden
//!   ```
#![allow(clippy::field_reassign_with_default)]

use serde::Deserialize;
use std::path::PathBuf;

const FIXTURE: &str =
    include_str!("fixtures/cpu_pre_migration_greedy_v1/qwen35_0_8b_cpu_greedy_tokens.json");

/// Env vars a reader might reach for to make this gate go quiet. None of them
/// work; naming them explicitly is what makes the refusal a refusal rather than
/// an unimplemented feature someone later adds.
const REFUSED_BYPASS_VARS: &[&str] = &[
    "LATTICE_CPU_GREEDY_GOLDEN_SKIP",
    "LATTICE_CPU_GREEDY_GOLDEN_ENFORCE",
    "LATTICE_SKIP_GOLDEN_GATES",
];

#[derive(Deserialize)]
struct GoldenCase {
    name: String,
    prompt: String,
    prompt_tokens: usize,
    expected_generated_ids: Vec<u32>,
    /// Per-case override of `generation.reasoning_budget`. Absent means the
    /// shared unbudgeted config, which is what the two original cases use.
    #[serde(default)]
    reasoning_budget: Option<usize>,
    /// The unbudgeted case this one was captured against, so the divergence
    /// control can compare two real captures instead of a capture against a
    /// hand-written expectation.
    #[serde(default)]
    paired_with: Option<String>,
}

#[derive(Deserialize)]
struct GoldenGeneration {
    temperature: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
    /// Load-bearing for the budgeted cases: `force_close_think` requires it, so
    /// the gate sets it explicitly rather than inheriting whatever
    /// `GenerateConfig::default()` happens to carry.
    enable_thinking: bool,
    top_k: usize,
    top_p: f32,
    stop_token_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Golden {
    max_new_tokens: usize,
    generation: GoldenGeneration,
    cases: Vec<GoldenCase>,
}

/// Panics if any bypass variable is set to anything at all, including `0`.
///
/// Taking the *presence* of the variable as the trigger rather than its value is
/// deliberate: a gate that honours `ENFORCE=0` has an off switch, and the row
/// this gate implements requires that it not have one. Someone who sets one of
/// these has stated an intent the gate must contradict loudly.
fn refuse_bypass_or_panic(lookup: impl Fn(&str) -> Option<String>) {
    for var in REFUSED_BYPASS_VARS {
        if let Some(value) = lookup(var) {
            panic!(
                "{var} is set (to {value:?}) but this gate does not have a bypass. \
                 It enforces by default by design (ADR-090 R03). If the checkpoint \
                 is genuinely unavailable, do not run the target."
            );
        }
    }
}

/// Resolves the checkpoint directory, or returns the panic message explaining
/// why it could not. Returned rather than panicked so a control can drive it.
fn resolve_model_dir(
    lookup: impl Fn(&str) -> Option<String>,
    exists: impl Fn(&PathBuf) -> bool,
) -> Result<PathBuf, String> {
    let raw = lookup("LATTICE_CPU_GREEDY_MODEL_DIR")
        .or_else(|| lookup("LATTICE_MODEL_DIR"))
        .ok_or_else(|| {
            "neither LATTICE_CPU_GREEDY_MODEL_DIR nor LATTICE_MODEL_DIR is set. \
             This gate enforces by default: an unset checkpoint is a failure, \
             not a skip."
                .to_string()
        })?;
    let path = PathBuf::from(&raw);
    if !path.is_absolute() {
        return Err(format!(
            "checkpoint path {raw:?} is relative; cargo test runs test binaries \
             with the crate directory as CWD, so it would resolve somewhere the \
             caller did not mean. Pass an absolute path."
        ));
    }
    if !exists(&path) {
        return Err(format!(
            "checkpoint {raw:?} does not exist. This gate enforces by default: \
             a missing artifact is a failure, not a skip."
        ));
    }
    Ok(path)
}

#[cfg(feature = "f16")]
fn env_lookup(var: &str) -> Option<String> {
    std::env::var(var).ok()
}

#[cfg(not(feature = "f16"))]
#[test]
fn cpu_pre_migration_greedy_golden() {
    // An absent feature enforces exactly like an absent checkpoint. The
    // checkpoint is bf16 safetensors, which `from_safetensors` refuses without
    // `f16`; compiling this target into a silent no-op is the skip-shaped
    // failure the row forbids.
    panic!(
        "this target was built without the `f16` feature, but the checkpoint is \
         bf16 and cannot be loaded without it. Re-run with --features f16."
    );
}

#[cfg(feature = "f16")]
#[test]
fn cpu_pre_migration_greedy_golden() {
    use lattice_inference::GenerateConfig;
    use lattice_inference::model::qwen35::Qwen35Model;

    refuse_bypass_or_panic(env_lookup);
    let model_dir = match resolve_model_dir(env_lookup, |p| p.exists()) {
        Ok(dir) => dir,
        Err(message) => panic!("{message}"),
    };

    let golden: Golden = serde_json::from_str(FIXTURE).expect("golden fixture parses");
    let model = Qwen35Model::from_safetensors(&model_dir)
        .unwrap_or_else(|e| panic!("loading {model_dir:?} failed: {e}"));

    let mut failures = Vec::new();
    for case in &golden.cases {
        let mut cfg = GenerateConfig::default();
        cfg.max_new_tokens = golden.max_new_tokens;
        cfg.temperature = golden.generation.temperature;
        cfg.repetition_penalty = golden.generation.repetition_penalty;
        cfg.seed = golden.generation.seed;
        cfg.enable_thinking = golden.generation.enable_thinking;
        cfg.reasoning_budget = case.reasoning_budget;

        let output = model
            .generate(&case.prompt, &cfg)
            .unwrap_or_else(|e| panic!("case {}: generation failed: {e}", case.name));

        if output.prompt_tokens != case.prompt_tokens {
            failures.push(format!(
                "case {}: prompt tokenized to {} tokens, golden recorded {}",
                case.name, output.prompt_tokens, case.prompt_tokens
            ));
        }
        if output.token_ids != case.expected_generated_ids {
            failures.push(format!(
                "case {}: greedy token IDs diverged\n  expected: {:?}\n  actual:   {:?}",
                case.name, case.expected_generated_ids, output.token_ids
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "pre-migration CPU greedy golden diverged:\n{}",
        failures.join("\n")
    );
}

/// Controls. These run in the same target and need no checkpoint: they drive the
/// refusal paths directly, so a reader can see that the gate refuses rather than
/// taking the doc comment's word for it.
mod controls {
    use super::*;

    fn from_pairs(
        pairs: &'static [(&'static str, &'static str)],
    ) -> impl Fn(&str) -> Option<String> {
        move |var| {
            pairs
                .iter()
                .find(|(k, _)| *k == var)
                .map(|(_, v)| (*v).to_string())
        }
    }

    /// `</think>`. Duplicated as a literal on purpose: the production constant
    /// lives in a `pub` module, but writing the number here means a rename or a
    /// re-point of that constant cannot silently carry this control with it.
    /// The capture that produced the budgeted cases emitted exactly this id.
    const THINK_CLOSE_TOKEN_ID: u32 = 248_069;

    #[test]
    fn fixture_parses_and_declares_all_four_cases() {
        let golden: Golden = serde_json::from_str(FIXTURE).expect("golden fixture parses");
        let names: Vec<&str> = golden.cases.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "short_factual",
                "long_prose",
                "short_factual_reasoning_budget_4",
                "long_prose_reasoning_budget_4",
            ]
        );
        for case in &golden.cases {
            // `decode_cap`'s documented contract: a reasoning budget gives the
            // reasoning tokens their OWN budget on top of the answer budget,
            // plus one for the forced `</think>` delimiter. An unbudgeted case
            // is capped at `max_new_tokens` alone. Both captures ran to their
            // cap, so equality is the right assertion, and it fails loudly if
            // that accounting ever changes.
            let expected_len = match case.reasoning_budget {
                Some(budget) if budget > 0 => budget + golden.max_new_tokens + 1,
                _ => golden.max_new_tokens,
            };
            assert_eq!(
                case.expected_generated_ids.len(),
                expected_len,
                "case {} froze {} ids but its cap is {}",
                case.name,
                case.expected_generated_ids.len(),
                expected_len
            );
        }
        // The config the gate replays must be the greedy one, not GenerateConfig's
        // serving default: repetition_penalty 1.1 is applied BEFORE the degenerate
        // temperature check routes to argmax, so temperature alone does not pin
        // greedy. See crates/inference/src/model/qwen35/sampling.rs.
        assert_eq!(golden.generation.temperature, 0.0);
        assert_eq!(golden.generation.repetition_penalty, 1.0);
        assert_eq!(golden.generation.seed, None);
        // Without this the budgeted cases would be capturing the disabled path:
        // `effective_reasoning_budget` returns None whenever thinking is off, so
        // a fixture recording a budget alongside `enable_thinking: false` would
        // freeze ordinary unbudgeted ids under a budgeted name.
        assert!(golden.generation.enable_thinking);
    }

    #[test]
    fn fields_the_gate_does_not_set_match_the_config_default() {
        // `top_k`, `top_p` and `stop_token_ids` are recorded in the fixture but
        // never written onto the replayed config, which is only safe while they
        // equal what `GenerateConfig::default()` already carries. Co-located
        // values that nothing reads are how a fixture comes to describe a run
        // that never happened; this is the read that keeps them honest.
        let golden: Golden = serde_json::from_str(FIXTURE).expect("golden fixture parses");
        let default = lattice_inference::GenerateConfig::default();
        assert_eq!(golden.generation.top_k, default.top_k);
        assert_eq!(golden.generation.top_p, default.top_p);
        assert_eq!(golden.generation.stop_token_ids, default.stop_token_ids);
    }

    #[test]
    fn every_budgeted_case_diverges_from_its_pair_at_the_budget() {
        // The point of the budgeted cases. `apply_override` replaces the sampled
        // id with `</think>` once `generated_so_far` reaches the budget, so a
        // budgeted capture must agree with its unbudgeted pair on every earlier
        // token and disagree at exactly that index. A budgeted case whose ids
        // matched its pair would prove nothing at all, which is the failure this
        // control exists to make impossible to miss.
        let golden: Golden = serde_json::from_str(FIXTURE).expect("golden fixture parses");
        let mut checked = 0usize;
        for case in &golden.cases {
            let (Some(budget), Some(pair_name)) =
                (case.reasoning_budget, case.paired_with.as_ref())
            else {
                continue;
            };
            let pair = golden
                .cases
                .iter()
                .find(|c| &c.name == pair_name)
                .unwrap_or_else(|| panic!("case {} names a pair that is absent", case.name));
            assert_eq!(
                pair.prompt, case.prompt,
                "case {} and its pair must share a prompt or the comparison is between two runs",
                case.name
            );
            assert!(pair.reasoning_budget.is_none(), "a pair must be unbudgeted");
            assert!(budget < pair.expected_generated_ids.len());
            assert_eq!(
                case.expected_generated_ids[..budget],
                pair.expected_generated_ids[..budget],
                "case {} diverges from its pair BEFORE the budget",
                case.name
            );
            assert_eq!(
                case.expected_generated_ids[budget], THINK_CLOSE_TOKEN_ID,
                "case {} does not carry the forced close token at its budget",
                case.name
            );
            assert_ne!(
                case.expected_generated_ids[budget], pair.expected_generated_ids[budget],
                "case {} agrees with its unbudgeted pair at the budget index, so the \
                 override path was never exercised",
                case.name
            );
            checked += 1;
        }
        assert_eq!(
            checked, 2,
            "expected two budgeted cases to check; a fixture edit that drops them \
             would otherwise leave this control vacuously green"
        );
    }

    #[test]
    fn the_pair_is_actually_short_and_long() {
        // R03's PAIR clause asks for an ordinary short and long prompt. Asserting
        // it here keeps the fixture honest if someone edits the prompts, and it
        // is what reads `prompt` and `prompt_tokens` outside the f16 arm, so
        // neither field needs an allow(dead_code) to describe itself.
        let golden: Golden = serde_json::from_str(FIXTURE).expect("golden fixture parses");
        let short = &golden.cases[0];
        let long = &golden.cases[1];
        assert!(!short.prompt.is_empty() && !long.prompt.is_empty());
        assert!(short.prompt_tokens > 0);
        assert!(
            long.prompt_tokens >= 8 * short.prompt_tokens,
            "long case is {} prompt tokens against the short case's {}, which is \
             not a short/long pair",
            long.prompt_tokens,
            short.prompt_tokens
        );
    }

    #[test]
    #[should_panic(expected = "does not have a bypass")]
    fn bypass_attempt_refuses() {
        refuse_bypass_or_panic(from_pairs(&[("LATTICE_CPU_GREEDY_GOLDEN_SKIP", "1")]));
    }

    #[test]
    #[should_panic(expected = "does not have a bypass")]
    fn enforce_zero_also_refuses() {
        // The shape that would quietly disable the sibling gates in this crate.
        refuse_bypass_or_panic(from_pairs(&[("LATTICE_CPU_GREEDY_GOLDEN_ENFORCE", "0")]));
    }

    #[test]
    fn no_bypass_var_set_is_accepted() {
        // Must-pass arm: without it the two should_panic controls above would
        // still pass if `refuse_bypass_or_panic` panicked unconditionally.
        refuse_bypass_or_panic(from_pairs(&[("PATH", "/usr/bin")]));
    }

    #[test]
    fn unset_checkpoint_refuses() {
        let err = resolve_model_dir(from_pairs(&[]), |_| true).unwrap_err();
        assert!(err.contains("not a skip"), "unexpected message: {err}");
    }

    #[test]
    fn missing_checkpoint_refuses() {
        let err = resolve_model_dir(
            from_pairs(&[("LATTICE_CPU_GREEDY_MODEL_DIR", "/nonexistent/checkpoint")]),
            |_| false,
        )
        .unwrap_err();
        assert!(err.contains("does not exist"), "unexpected message: {err}");
    }

    #[test]
    fn relative_checkpoint_refuses() {
        let err = resolve_model_dir(
            from_pairs(&[("LATTICE_CPU_GREEDY_MODEL_DIR", "models/qwen3.5-0.8b")]),
            |_| true,
        )
        .unwrap_err();
        assert!(err.contains("relative"), "unexpected message: {err}");
    }

    #[test]
    fn present_absolute_checkpoint_resolves() {
        // Must-pass arm for the resolver: the three refusal controls above prove
        // nothing unless the accepting path is shown to exist.
        let dir = resolve_model_dir(
            from_pairs(&[("LATTICE_CPU_GREEDY_MODEL_DIR", "/abs/checkpoint")]),
            |_| true,
        )
        .expect("absolute existing path resolves");
        assert_eq!(dir, PathBuf::from("/abs/checkpoint"));
    }

    #[test]
    fn dedicated_var_wins_over_the_shared_one() {
        let dir = resolve_model_dir(
            from_pairs(&[
                ("LATTICE_CPU_GREEDY_MODEL_DIR", "/abs/dedicated"),
                ("LATTICE_MODEL_DIR", "/abs/shared"),
            ]),
            |_| true,
        )
        .expect("resolves");
        assert_eq!(dir, PathBuf::from("/abs/dedicated"));
    }
}
