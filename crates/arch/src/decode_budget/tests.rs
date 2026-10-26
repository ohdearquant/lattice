use super::*;
use crate::generation::GenerateConfig;
use crate::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;

#[test]
fn test_generate_config_defaults() {
    let gen_cfg = GenerateConfig::default();
    assert_eq!(gen_cfg.max_new_tokens, 256);
    assert!((gen_cfg.temperature - 0.7).abs() < 1e-6);
    assert_eq!(gen_cfg.top_k, 50);
    assert!((gen_cfg.top_p - 0.9).abs() < 1e-6);
    assert!((gen_cfg.repetition_penalty - 1.1).abs() < 1e-6);
    assert!(
        gen_cfg.stop_token_ids.contains(&QWEN_CHAT_IM_END_TOKEN_ID),
        "default stop tokens must include im_end"
    );
}

#[test]
fn test_generate_config_enable_thinking_default_and_toggle() {
    let default_cfg = GenerateConfig::default();
    assert!(
        default_cfg.enable_thinking,
        "default must have thinking enabled"
    );

    let no_think = GenerateConfig {
        min_p: 0.0,
        enable_thinking: false,
        ..GenerateConfig::default()
    };
    assert!(!no_think.enable_thinking);
    // Other fields unaffected
    assert_eq!(no_think.max_new_tokens, 256);
    assert!((no_think.temperature - 0.7).abs() < 1e-6);
}

// ── decode_cap unit tests ────────────────────────────────────────────────

#[test]
fn decode_cap_none_budget_returns_max() {
    // Disabled path must be byte-identical to the pre-budget behaviour.
    assert_eq!(decode_cap(None, 512), 512);
    assert_eq!(decode_cap(None, 0), 0);
}

#[test]
fn decode_cap_zero_budget_returns_max() {
    // Some(0) is treated as disabled.
    assert_eq!(decode_cap(Some(0), 512), 512);
    assert_eq!(decode_cap(Some(0), 1), 1);
}

#[test]
fn decode_cap_nonzero_budget_adds_budgets() {
    // Worst case = rb + max_new_tokens + 1 (the +1 is the forced </think> delimiter).
    // Mutation-sensitive: revert to rb+max and these assertions fail.
    assert_eq!(decode_cap(Some(2048), 512), 2561);
    assert_eq!(decode_cap(Some(1), 1), 3);
    assert_eq!(decode_cap(Some(100), 200), 301);
}

#[test]
fn decode_cap_saturates_on_overflow() {
    // saturating_add must not wrap on usize::MAX inputs.
    assert_eq!(decode_cap(Some(usize::MAX), 1), usize::MAX);
    assert_eq!(decode_cap(Some(1), usize::MAX), usize::MAX);
}

// ── force_close_think unit tests ────────────────────────────────────────

#[test]
fn force_close_think_disabled_when_budget_none() {
    // budget=None → always returns None regardless of other args.
    assert_eq!(
        force_close_think(None, true, false, 100, Some(99), true),
        None,
        "None budget must disable forcing"
    );
}

#[test]
fn force_close_think_disabled_when_budget_zero() {
    // budget=Some(0) → budget > 0 guard fails → None.
    assert_eq!(
        force_close_think(Some(0), true, false, 100, Some(99), true),
        None,
        "budget=0 must disable forcing"
    );
}

#[test]
fn force_close_think_disabled_when_enable_thinking_false() {
    // enable_thinking=false → no reasoning block → forcing is a no-op.
    assert_eq!(
        force_close_think(Some(10), false, false, 20, Some(99), true),
        None,
        "enable_thinking=false must disable forcing"
    );
}

#[test]
fn force_close_think_disabled_when_already_closed() {
    // thinking_closed=true → block already closed → should not force again.
    assert_eq!(
        force_close_think(Some(10), true, true, 20, Some(99), true),
        None,
        "already-closed thinking block must not force again"
    );
}

#[test]
fn force_close_think_disabled_when_close_id_none() {
    // close_id=None → model has no </think> token → forcing is a no-op.
    assert_eq!(
        force_close_think(Some(10), true, false, 20, None, true),
        None,
        "close_id=None must disable forcing"
    );
}

#[test]
fn force_close_think_fires_at_budget_boundary() {
    let close_id = 248_069_u32;
    // generated_so_far == budget → should force (mutation: >= not >).
    assert_eq!(
        force_close_think(Some(10), true, false, 10, Some(close_id), true),
        Some(close_id),
        "must force when generated_so_far equals budget"
    );
    // generated_so_far > budget → also forces.
    assert_eq!(
        force_close_think(Some(10), true, false, 11, Some(close_id), true),
        Some(close_id),
        "must force when generated_so_far exceeds budget"
    );
}

#[test]
fn force_close_think_waits_for_a_boundary_within_the_grace_window() {
    let close_id = 248_069_u32;
    assert_eq!(
        force_close_think(Some(10), true, false, 10, Some(close_id), false),
        None,
        "the budget alone must not close before a boundary or the fallback"
    );
    assert_eq!(
        force_close_think(Some(10), true, false, 20, Some(close_id), true),
        Some(close_id),
        "a boundary after the budget must close"
    );
    assert_eq!(
        force_close_think(Some(10), true, false, 41, Some(close_id), false),
        None,
        "the fallback is exactly B + 32, not earlier"
    );
    assert_eq!(
        force_close_think(Some(10), true, false, 42, Some(close_id), false),
        Some(close_id),
        "the fallback must close at B + 32"
    );
}

#[test]
fn force_close_think_accepts_a_boundary_at_the_budget() {
    let close_id = 248_069_u32;
    assert_eq!(
        force_close_think(Some(10), true, false, 10, Some(close_id), true),
        Some(close_id),
        "a boundary at the budget must close without delay"
    );
}

#[test]
fn force_close_think_does_not_fire_before_budget() {
    let close_id = 248_069_u32;
    // generated_so_far < budget → must NOT force (mutation-sensitive boundary).
    assert_eq!(
        force_close_think(Some(10), true, false, 9, Some(close_id), false),
        None,
        "must not force when generated_so_far is one below budget"
    );
    assert_eq!(
        force_close_think(Some(10), true, false, 0, Some(close_id), false),
        None,
        "must not force when zero tokens generated"
    );
}
