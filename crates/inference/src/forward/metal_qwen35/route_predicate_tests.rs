use super::{mtp_route_active, self_spec_route_active, use_batch_gemm_verifier};
use crate::generation::GenerateConfig;

fn greedy_gen_cfg(stop_strings: Vec<String>) -> GenerateConfig {
    GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(true),
        grammar: None,
        stop_strings,
        reasoning_budget: None,
        logprobs: None,
    }
}

/// Baseline: with every other precondition satisfied and NO stop strings,
/// the MTP route is active. Establishes the contrast the next test
/// depends on -- without this, `mtp_route_active` returning `false` when
/// `stop_strings` is non-empty would be trivially true for the wrong
/// reason (e.g. a typo`'d always-false condition).
#[test]
fn mtp_route_active_when_no_stop_strings() {
    let gen_cfg = greedy_gen_cfg(vec![]);
    assert!(mtp_route_active(true, true, &gen_cfg, false));
}

/// Non-empty `stop_strings` must route around MTP even
/// though every other MTP precondition (present, enabled, greedy,
/// non-compact, no grammar) is satisfied.
#[test]
fn mtp_route_blocked_by_nonempty_stop_strings() {
    let gen_cfg = greedy_gen_cfg(vec!["never_matches_xyz".to_string()]);
    assert!(
        !mtp_route_active(true, true, &gen_cfg, false),
        "MTP must not activate when stop_strings is non-empty -- its \
         draft/verify loop has no string-stop awareness"
    );
}

/// Baseline for self-spec, mirroring the MTP baseline above.
#[test]
fn self_spec_route_active_when_no_stop_strings() {
    let gen_cfg = greedy_gen_cfg(vec![]);
    assert!(self_spec_route_active(true, true, &gen_cfg, false, 1));
}

/// Non-empty `stop_strings` must route around
/// self-speculative decode even though every other precondition
/// (checkpoints present, env set, greedy, non-compact, no grammar, has
/// linear-attention layers) is satisfied.
#[test]
fn self_spec_route_blocked_by_nonempty_stop_strings() {
    let gen_cfg = greedy_gen_cfg(vec!["never_matches_xyz".to_string()]);
    assert!(
        !self_spec_route_active(true, true, &gen_cfg, false, 1),
        "self-spec must not activate when stop_strings is non-empty -- \
         same string-stop-unaware draft/verify loop as MTP"
    );
}

/// ADR-080 C3 (#783): a set `reasoning_budget` must route around MTP even
/// though every other MTP precondition is satisfied -- MTP's draft/verify
/// loop has no budget-forcing (`decode_cap` / `force_close_think`)
/// awareness, mirroring the `stop_strings` route-around above.
///
/// Mutation sensitivity: removing the `reasoning_budget.is_none()` clause
/// from `mtp_route_active` makes this assertion fail (the route stays
/// active), which would silently drop the caller's reasoning budget once
/// MTP takes over decoding.
#[test]
fn mtp_route_blocked_by_set_reasoning_budget() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(64),
        ..greedy_gen_cfg(vec![])
    };
    assert!(
        !mtp_route_active(true, true, &gen_cfg, false),
        "MTP must not activate when reasoning_budget is set -- its \
         draft/verify loop has no budget-forcing awareness"
    );
}

/// Sibling to `mtp_route_blocked_by_set_reasoning_budget` for the
/// self-speculative route (ADR-080 C3, #783).
#[test]
fn self_spec_route_blocked_by_set_reasoning_budget() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        reasoning_budget: Some(64),
        ..greedy_gen_cfg(vec![])
    };
    assert!(
        !self_spec_route_active(true, true, &gen_cfg, false, 1),
        "self-spec must not activate when reasoning_budget is set -- same \
         budget-forcing-unaware draft/verify loop as MTP"
    );
}

/// PR #787: a non-identity `repetition_penalty`
/// must route around MTP even though every other MTP precondition
/// (present, enabled, greedy, non-compact, no grammar, no stop strings,
/// no reasoning budget) is satisfied. MTP's draft/verify loop selects via
/// raw `argmax_f32_first_wins`, never `sample_token`, so a caller-supplied
/// penalty would otherwise be silently ignored.
///
/// Mutation sensitivity: removing the `repetition_penalty == 1.0` clause
/// from `mtp_route_active` makes this assertion fail (the route stays
/// active), reproducing the reported failure mode exactly.
#[test]
fn mtp_route_blocked_by_nonidentity_repetition_penalty() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        repetition_penalty: 1.1,
        ..greedy_gen_cfg(vec![])
    };
    assert!(
        !mtp_route_active(true, true, &gen_cfg, false),
        "MTP must not activate when repetition_penalty is not 1.0 -- its \
         draft/verify loop selects via raw argmax and has no \
         repetition-penalty awareness"
    );
}

/// Sibling to `mtp_route_blocked_by_nonidentity_repetition_penalty` for
/// the self-speculative route (PR #787).
#[test]
fn self_spec_route_blocked_by_nonidentity_repetition_penalty() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        repetition_penalty: 1.1,
        ..greedy_gen_cfg(vec![])
    };
    assert!(
        !self_spec_route_active(true, true, &gen_cfg, false, 1),
        "self-spec must not activate when repetition_penalty is not 1.0 \
         -- same raw-argmax-selecting, penalty-unaware draft/verify loop \
         as MTP"
    );
}

/// Output-level discriminating case (PR #787):
/// proves the route-around above is not merely cosmetic. With
/// `repetition_penalty = 1.0` the raw dense argmax MTP/self-spec use
/// (`argmax_f32_first_wins`) agrees with the canonical
/// repetition-penalty-aware `sample_token` path. Once a previously-seen
/// token depresses the raw argmax below the runner-up (same mechanism as
/// `test_repetition_penalty_can_flip_greedy_argmax_when_seen_in_history`
/// in `model::qwen35::sampling`), the two paths disagree -- exactly the
/// silent-divergence scenario the new predicate clause closes.
#[test]
fn nonidentity_repetition_penalty_flips_winner_vs_raw_argmax() {
    use crate::sampling::{argmax_f32_first_wins, sample_full_logits as sample_token};

    // Token 0 has the highest raw logit; token 1 is a close second, chosen
    // so 5.0 / 1.1 falls below token 1's unpenalized 4.6.
    let logits = [5.0_f32, 4.6, 0.1];
    let previous_ids = [0_u32, 0, 0];

    // Raw dense argmax (what MTP/self-spec actually call): always token 0,
    // with no notion of `previous_ids` or `repetition_penalty` at all.
    assert_eq!(
        argmax_f32_first_wins(&logits),
        0,
        "raw argmax is penalty-blind by construction"
    );

    // Canonical path with the caller's non-identity penalty and the
    // argmax token already in history: the penalized pick flips to the
    // runner-up.
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        repetition_penalty: 1.1,
        ..greedy_gen_cfg(vec![])
    };
    let mut rng = 7u64;
    let canonical = sample_token(&logits, &gen_cfg, &previous_ids, &mut rng, 0.0, 0.0);
    assert_eq!(
        canonical, 1,
        "sample_token must penalize the previously-seen argmax enough to \
         flip greedy selection to the runner-up"
    );

    assert_ne!(
        argmax_f32_first_wins(&logits),
        canonical,
        "raw argmax and the canonical repetition-aware pick must disagree \
         here -- this is exactly the output-correctness gap that routing \
         MTP/self-spec around non-identity repetition_penalty closes"
    );
}

/// PR #787: a set `logprobs` must route
/// around MTP even though every other MTP precondition (present,
/// enabled, greedy, non-compact, no grammar, no stop strings, no
/// reasoning budget, identity repetition penalty) is satisfied.
/// `generate_greedy_mtp` unconditionally returns `token_logprobs: vec![]`,
/// so a caller requesting logprobs would otherwise get `Ok` with the
/// requested output silently absent.
///
/// Falsification recipe: add `logprobs: Some(0)` to the
/// otherwise-active predicate fixture and assert the route is inactive.
///
/// Mutation sensitivity: removing the `logprobs.is_none()` clause from
/// `mtp_route_active` makes this assertion fail (the route stays
/// active), reproducing the reported failure mode exactly.
#[test]
fn mtp_route_blocked_by_set_logprobs() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        logprobs: Some(0),
        ..greedy_gen_cfg(vec![])
    };
    assert!(
        !mtp_route_active(true, true, &gen_cfg, false),
        "MTP must not activate when logprobs is set -- its output path \
         unconditionally returns an empty token_logprobs vector"
    );
}

/// Sibling to `mtp_route_blocked_by_set_logprobs` for the
/// self-speculative route (PR #787):
/// `generate_greedy_self_spec` has the same unconditional empty-vector
/// return.
#[test]
fn self_spec_route_blocked_by_set_logprobs() {
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        logprobs: Some(0),
        ..greedy_gen_cfg(vec![])
    };
    assert!(
        !self_spec_route_active(true, true, &gen_cfg, false, 1),
        "self-spec must not activate when logprobs is set -- same \
         unconditional empty-token_logprobs output path as MTP"
    );
}

/// Baseline: with the batch switch on and no adapter loaded, the
/// batch-GEMM verifier is selected. Establishes the contrast the next
/// test depends on, so an always-false predicate cannot pass it.
#[test]
fn batch_gemm_verifier_selected_when_switch_on_and_no_adapter() {
    assert!(use_batch_gemm_verifier(true, false));
}

/// A loaded adapter must keep the per-token verifier even with the batch
/// switch on: the batch-GEMM dispatch applies base projections only.
///
/// Mutation sensitivity: making the predicate ignore `adapter_loaded`
/// makes this assertion fail.
#[test]
fn batch_gemm_verifier_not_selected_when_adapter_loaded() {
    assert!(
        !use_batch_gemm_verifier(true, true),
        "the batch-GEMM verifier applies no LoRA adapter, so it must not \
         be selected while one is loaded"
    );
}

#[test]
fn batch_gemm_verifier_not_selected_when_switch_off() {
    assert!(!use_batch_gemm_verifier(false, false));
    assert!(!use_batch_gemm_verifier(false, true));
}
