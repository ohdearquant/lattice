//! GDN-first self-speculative decoding.

use super::*;
// -----------------------------------------------------------------------
// GDN-first self-speculative decoding tests
// -----------------------------------------------------------------------

const OPTION_PROMPT: [u32; 5] = [4, 9, 13, 20, 11];
const OPTION_LETTERS: [u32; 4] = [3, 8, 14, 30];
const OPTION_CACHE_LEN: usize = 16;

/// A state on `option_scoring_fixture`, or `None` when this host has no Metal device.
/// Callers hold `gpu_test_lock()` before calling.
fn option_scoring_state(context: &str) -> Option<MetalQwen35State> {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!("[METAL_TEST_SKIP] context={context} reason=no_metal_device");
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present ({context})"
        );
        return None;
    };
    let (cfg, weights) = option_scoring_fixture();
    Some(
        MetalQwen35State::new(&weights, &cfg, OPTION_CACHE_LEN)
            .expect("option scoring fixture constructs"),
    )
}

/// Leave the state mid-session: one KV row written and a non-initial GDN state.
fn dirty_option_scoring_state(state: &mut MetalQwen35State) {
    use crate::speculative::MtpTargetVerifier as _;

    state.reset_state();
    state
        .try_forward_step(2, 0)
        .expect("step at the live cursor");
    let mut seeded = state.snapshot_gdn_states();
    seeded[0].0[0] = 1.0;
    seeded[0].1[0] = -1.0;
    state.restore_gdn_states(&seeded);
    assert_eq!(state.session.kv_cache.seq_len, 1);
    assert!(!state.gdn_state_is_initial());
}

#[test]
fn score_option_letters_logits_equal_try_forward_prefill_at_the_letters_bitwise() {
    let _gpu = gpu_test_lock();
    let Some(mut state) = option_scoring_state(
        "score_option_letters_logits_equal_try_forward_prefill_at_the_letters_bitwise",
    ) else {
        return;
    };

    state.reset_state();
    let full = state
        .try_forward_prefill(&OPTION_PROMPT)
        .expect("reference prefill");
    let expected = f32_bits(
        &OPTION_LETTERS
            .iter()
            .map(|&id| full[id as usize])
            .collect::<Vec<_>>(),
    );

    // The reference prefill left the session advanced, so the scorer has to reset it.
    let scores = state
        .score_option_letters(&OPTION_PROMPT, &OPTION_LETTERS)
        .expect("scores");
    assert_eq!(f32_bits(&scores.logits), expected);
    assert_eq!(
        scores,
        crate::option_scoring::option_scores_from_logits(&full, &OPTION_LETTERS)
            .expect("pure scoring of the reference row"),
        "probs and label_mass derive from the same row"
    );

    // The fixture can tell the letters apart and can tell the last position from the first.
    let mut sorted = expected.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        OPTION_LETTERS.len(),
        "letter logits must not tie"
    );
    state.reset_state();
    let first = state
        .try_forward_prefill(&OPTION_PROMPT[..1])
        .expect("one-token prefill");
    let at_first = f32_bits(
        &OPTION_LETTERS
            .iter()
            .map(|&id| first[id as usize])
            .collect::<Vec<_>>(),
    );
    assert_ne!(at_first, expected, "first-position read must differ");
}

#[test]
fn score_option_letters_logits_depend_on_earlier_prompt_tokens() {
    let _gpu = gpu_test_lock();
    let Some(mut state) =
        option_scoring_state("score_option_letters_logits_depend_on_earlier_prompt_tokens")
    else {
        return;
    };

    let mut other_prompt = OPTION_PROMPT;
    other_prompt[2] = 14;
    assert_eq!(other_prompt.last(), OPTION_PROMPT.last());
    let base = state
        .score_option_letters(&OPTION_PROMPT, &OPTION_LETTERS)
        .expect("scores");
    let changed = state
        .score_option_letters(&other_prompt, &OPTION_LETTERS)
        .expect("scores with a different earlier token");
    assert_ne!(
        f32_bits(&base.logits),
        f32_bits(&changed.logits),
        "letter logits must see the earlier prompt tokens"
    );
}

#[test]
fn score_option_letters_repeat_calls_are_bit_identical_from_any_session_state() {
    let _gpu = gpu_test_lock();
    let Some(mut state) = option_scoring_state(
        "score_option_letters_repeat_calls_are_bit_identical_from_any_session_state",
    ) else {
        return;
    };

    let fresh = state
        .score_option_letters(&OPTION_PROMPT, &OPTION_LETTERS)
        .expect("scores on a fresh session");
    let repeat = state
        .score_option_letters(&OPTION_PROMPT, &OPTION_LETTERS)
        .expect("scores on the session the first call left advanced");
    assert_eq!(f32_bits(&repeat.logits), f32_bits(&fresh.logits));
    assert_eq!(f32_bits(&repeat.probs), f32_bits(&fresh.probs));
    assert_eq!(repeat.label_mass.to_bits(), fresh.label_mass.to_bits());

    dirty_option_scoring_state(&mut state);
    let after_dirty = state
        .score_option_letters(&OPTION_PROMPT, &OPTION_LETTERS)
        .expect("scores on a mid-session state");
    assert_eq!(f32_bits(&after_dirty.logits), f32_bits(&fresh.logits));
    assert_eq!(f32_bits(&after_dirty.probs), f32_bits(&fresh.probs));
    assert_eq!(after_dirty.label_mass.to_bits(), fresh.label_mass.to_bits());

    let sum: f32 = fresh.probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "probs sum to {sum}");
    assert!((0.0..=1.0).contains(&fresh.label_mass));
}

#[test]
fn score_option_letters_refuses_before_resetting_or_dispatching() {
    let _gpu = gpu_test_lock();
    let Some(mut state) =
        option_scoring_state("score_option_letters_refuses_before_resetting_or_dispatching")
    else {
        return;
    };
    dirty_option_scoring_state(&mut state);
    let seq_len_before = state.session.kv_cache.seq_len;
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = snapshot_gdn_bytes(&state);
    let vocab = state.engine.config.vocab_size as u32;
    let too_long = vec![1_u32; OPTION_CACHE_LEN + 1];
    let oov_prompt = [1_u32, vocab];

    let cases: [(&str, &[u32], &[u32]); 7] = [
        ("empty prompt", &[], &OPTION_LETTERS),
        ("no letters", &OPTION_PROMPT, &[]),
        ("letter id at vocab", &OPTION_PROMPT, &[3, vocab]),
        ("duplicate letter ids", &OPTION_PROMPT, &[3, 8, 3]),
        ("prompt token at vocab", &oov_prompt, &OPTION_LETTERS),
        ("prompt beyond capacity", &too_long, &OPTION_LETTERS),
        ("letter id far out of range", &OPTION_PROMPT, &[u32::MAX]),
    ];
    for (name, prompt, letters) in cases {
        match state.score_option_letters(prompt, letters) {
            Err(crate::error::InferenceError::InvalidInput(_)) => {}
            other => panic!("{name}: expected InvalidInput, got {other:?}"),
        }
        assert_eq!(state.session.kv_cache.seq_len, seq_len_before, "{name}");
        assert_eq!(snapshot_kv_bytes(&state), kv_before, "{name}");
        assert_eq!(snapshot_gdn_bytes(&state), gdn_before, "{name}");
    }

    // The capacity boundary itself is admitted.
    let at_capacity = vec![1_u32; OPTION_CACHE_LEN];
    state
        .score_option_letters(&at_capacity, &OPTION_LETTERS)
        .expect("a prompt of exactly the session capacity is valid");
}

/// Regression test: `generate_with_speculation` must REPORT an unsupported
/// (MoE) multi-token prefill as an error, not abort the process. It is a
/// request-facing entry point, so reaching the batched path's MoE rejection
/// through the panicking `forward_prefill` wrapper turns an ordinary valid
/// prompt into a host crash. Two tokens is the smallest prompt that takes
/// the batched (M>1) path.
#[test]
fn generate_with_speculation_reports_moe_prefill_rejection_without_panicking() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_moe_prefill_fixture();
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 16).expect("moe prefill fixture must construct");

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.generate_with_speculation(&[1u32, 3, 5], 4, 0, 2, 2)
    }));

    let returned =
        outcome.expect("unsupported-MoE prefill must be returned as Err, never unwound as a panic");
    let error = returned.expect_err("unsupported-MoE prefill must not produce tokens");
    let message = match &error {
        crate::error::InferenceError::UnsupportedModel(message) => message.clone(),
        other => panic!(
            "the documented contract for an unsupported-MoE prefill is \
                     InferenceError::UnsupportedModel; got {other:?}. Any other variant \
                     means a different rejection fired and this test would pass without \
                     the guard under test having run at all"
        ),
    };
    assert!(
        message.contains("MoE"),
        "the rejection must name the MoE batched-prefill limitation so a caller \
                 can act on it; got {message:?}"
    );
}

/// Regression test: batched prefill on an unsupported (MoE) config must
/// reject before any state mutation, so a retry after the failure observes
/// exactly the same clean session state — not a partially-encoded one.
/// `assert_batched_prefill_dense_only` runs before the embedding copy, the
/// command buffer, and any recurrent/KV encode, so `session.position` and
/// `activations.hidden` (non-zero embedding rows would prove the copy ran)
/// must stay untouched across repeated failed attempts.
#[test]
fn forward_prefill_batched_chunk_rejects_moe_before_any_state_mutation() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_moe_prefill_fixture();
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 16).expect("moe prefill fixture must construct");

    let initial_position = state.session.position();
    let tokens = [1u32, 3, 5];

    for attempt in 0..2 {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.forward_prefill_batched_chunk(&tokens, 0, true, true, false)
        }));
        assert!(
            result.is_err(),
            "attempt {attempt}: unsupported-MoE batched prefill must fail cleanly, \
                     not silently produce logits"
        );
        assert_eq!(
            state.session.position(),
            initial_position,
            "attempt {attempt}: session position must be unchanged by a rejected prefill"
        );
        // SAFETY: StorageModeShared, read-only; the guard panics before a
        // command buffer is ever created, so there is no GPU work in flight.
        let hidden_after =
            unsafe { read_buffer(&state.session.activations.hidden, cfg.hidden_size) };
        assert!(
            hidden_after.iter().all(|&v| v == 0.0),
            "attempt {attempt}: embedding copy must not run before the MoE guard \
                     fires (activations.hidden should stay zero-initialized)"
        );
    }
}

/// Issue #171 smoke test: the block-top-k Stage-1 (fused GEMV + block-local
/// argmax) / Stage-2 (existing `argmax_merge`) dispatch must select the
/// same logit value as the untouched full-logit exact path — it is a
/// reduction over identical dot products, not an approximation. Compares
/// logit *values* (with a full-logits lookup for the returned token_id)
/// rather than requiring an exact token_id match, since the tiny fixture's
/// zeroed projection weights make several vocab rows tie on value; a
/// tie-break difference between the CPU iteration order and the GPU
/// bitonic sort is not a bug. This is a compile/dispatch smoke test for
/// the tester's deeper 100%-agreement suite, not a replacement for it.
#[test]
fn lm_head_block_argmax_matches_full_logits() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();

    let mut baseline = MetalQwen35State::new(&weights, &cfg, 8).expect("baseline state");
    let full_logits = baseline.forward_step(3, 0);
    let expected_max = full_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut blocked = MetalQwen35State::new(&weights, &cfg, 8).expect("blocked state");
    blocked.session.compact_route = GpuTopkRoute::BlockArgmax;
    blocked.session.compact_topk = 1;
    let empty_logits = blocked.forward_step(3, 0);
    assert!(
        empty_logits.is_empty(),
        "compact route must return empty logits and populate compact_result instead"
    );
    assert_eq!(
        blocked.session.compact_result.len(),
        1,
        "BlockArgmax must produce exactly one candidate"
    );
    let got = blocked.session.compact_result[0];
    assert!(
        (got.logit - expected_max).abs() < 1e-3,
        "block-topk argmax logit {} disagreed with full-logit max {expected_max}",
        got.logit
    );
    assert!(
        (full_logits[got.token_id as usize] - expected_max).abs() < 1e-3,
        "block-topk returned token_id {} whose full-logit value {} is not the max \
                 {expected_max} — Stage-1/Stage-2 row/token_id bookkeeping bug",
        got.token_id,
        full_logits[got.token_id as usize]
    );
}

/// Issue #171 smoke test: Stage-1 block-local top-k (LOCAL_K=8) merged via
/// the existing generic `topk_merge_pass` kernel must return the same
/// candidate *values* as the true top-8 of the full-logit exact path (see
/// the argmax test above for why value/row comparison is used instead of
/// token_id set equality).
#[test]
fn lm_head_block_topk8_matches_full_logits() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();

    let mut baseline = MetalQwen35State::new(&weights, &cfg, 8).expect("baseline state");
    let full_logits = baseline.forward_step(3, 0);
    let mut sorted_desc = full_logits.clone();
    sorted_desc.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let eighth_largest = sorted_desc[7];

    let mut blocked = MetalQwen35State::new(&weights, &cfg, 8).expect("blocked state");
    blocked.session.compact_route = GpuTopkRoute::BlockTopK { local_k: 8 };
    blocked.session.compact_topk = 8;
    let empty_logits = blocked.forward_step(3, 0);
    assert!(empty_logits.is_empty());
    assert_eq!(
        blocked.session.compact_result.len(),
        8,
        "BlockTopK{{local_k:8}} must produce exactly 8 candidates"
    );

    let mut seen_ids = std::collections::HashSet::new();
    for c in &blocked.session.compact_result {
        assert!(
            seen_ids.insert(c.token_id),
            "duplicate token_id {} in block-topk result",
            c.token_id
        );
        assert!(
            (full_logits[c.token_id as usize] - c.logit).abs() < 1e-3,
            "candidate token_id {} logit {} does not match full-logit value {} \
                     — Stage-1/Stage-2 row/token_id bookkeeping bug",
            c.token_id,
            c.logit,
            full_logits[c.token_id as usize]
        );
        assert!(
            c.logit >= eighth_largest - 1e-3,
            "candidate logit {} is outside the true top-8 (8th largest = {eighth_largest})",
            c.logit
        );
    }
}
