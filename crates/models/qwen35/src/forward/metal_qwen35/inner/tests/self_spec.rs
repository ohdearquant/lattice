//! GDN-first self-speculative decoding.

use super::*;
// -----------------------------------------------------------------------
// GDN-first self-speculative decoding tests
// -----------------------------------------------------------------------

#[test]
fn self_spec_round_respects_room_and_returns_next_at_committed_boundary() {
    let _gpu = gpu_test_lock();
    Device::system_default().expect("Metal device required for self-spec room regression");
    with_self_spec_env(|| {
        let (cfg, mut weights) = tiny_hybrid_fixture();
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        weights.embed_tokens.fill(0.0);
        for token in 0..5 {
            weights.embed_tokens[token * hidden + token] = 1.0;
        }
        let crate::model::qwen35::FeedForwardWeights::Dense(ffn) = &mut weights.layers[0].1.ffn
        else {
            panic!("fixture requires a dense FFN");
        };
        for token in 0..5 {
            ffn.gate_proj[token * hidden + token] = 1.0;
            ffn.up_proj[token * hidden + token] = 1.0;
            ffn.down_proj[token * intermediate + token] = -0.01;
            ffn.down_proj[((token + 1) % 5) * intermediate + token] = 0.01;
        }

        for room in [1, 2] {
            let mut state = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("self-spec fixture must construct");
            let mut reference = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("reference fixture must construct");
            assert!(state.session.gdn_checkpoints.is_some());
            let pos = state.session.kv_cache.seq_len;
            let mut reference_tokens = Vec::new();
            let mut reference_next = Vec::new();
            let mut token = 0;
            for position in 0..=SELF_SPEC_MAX_DRAFT {
                reference_tokens.push(token);
                let logits = reference
                    .try_forward_step(token, position)
                    .expect("reference forward");
                token = crate::sampling::argmax_f32_first_wins(&logits);
                reference_next.push(token);
            }
            assert_eq!(reference_tokens, [0, 1, 2, 3, 4]);
            assert_ne!(
                reference_next[room - 1],
                reference_next[SELF_SPEC_MAX_DRAFT]
            );

            let mut metrics = SpeculativeMetrics::self_spec();
            let round = state.speculative_round(&mut metrics, 0, room, &|_| false);
            let SpeculativeMetrics::SelfSpec(metrics) = metrics else {
                panic!("self-spec route required");
            };
            assert_eq!(metrics.verify_calls, 1, "real verification must run");
            assert_eq!(metrics.draft_forwards, SELF_SPEC_MAX_DRAFT);
            assert_eq!(metrics.fallback_tokens, 0);
            assert_eq!(metrics.accepted_extra_tokens, room - 1);
            assert_eq!(round.committed, reference_tokens[..room]);
            assert_eq!(state.session.kv_cache.seq_len, pos + round.committed.len());
            assert_eq!(round.next, Some(reference_next[room - 1]));
            assert!(!round.cache_full);
            eprintln!("[SELF_SPEC_ROOM_VERIFIED] room={room}");
        }
    });
}

#[test]
fn self_spec_accepted_stop_restores_committed_kv_and_gdn_state() {
    use crate::speculative::MtpTargetVerifier as _;

    let _gpu = gpu_test_lock();
    Device::system_default().expect("Metal device required for accepted-stop witness");
    with_self_spec_env(|| {
        let (cfg, mut weights) = tiny_hybrid_fixture();
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        weights.embed_tokens.fill(0.0);
        for token in 0..5 {
            weights.embed_tokens[token * hidden + token] = 1.0;
        }
        let crate::model::qwen35::FeedForwardWeights::Dense(ffn) = &mut weights.layers[0].1.ffn
        else {
            panic!("fixture requires a dense FFN");
        };
        for token in 0..5 {
            ffn.gate_proj[token * hidden + token] = 1.0;
            ffn.up_proj[token * hidden + token] = 1.0;
            ffn.down_proj[token * intermediate + token] = -0.01;
            ffn.down_proj[((token + 1) % 5) * intermediate + token] = 0.01;
        }
        let crate::model::qwen35::AttentionWeights::Full(full) = &mut weights.layers[3].0 else {
            panic!("fixture requires a full attention layer");
        };
        full.k_proj[0] = 1.0;
        full.v_proj[0] = 1.0;

        let mut state =
            MetalQwen35State::new(&weights, &cfg, 16).expect("self-spec fixture must construct");
        let mut reference =
            MetalQwen35State::new(&weights, &cfg, 16).expect("reference fixture must construct");
        assert!(state.session.gdn_checkpoints.is_some());
        state.try_forward_step(4, 0).expect("state prefill");
        reference.try_forward_step(4, 0).expect("reference prefill");
        let pos = state.session.kv_cache.seq_len;
        assert_eq!(pos, reference.session.kv_cache.seq_len);

        let mut seed = state.snapshot_gdn_states();
        assert!(!seed.is_empty());
        for (s, conv) in &mut seed {
            s.fill(0.5);
            for (i, value) in conv.iter_mut().enumerate() {
                *value = ((i % 7) + 1) as f32 * 0.125;
            }
        }
        state.restore_gdn_states(&seed);
        reference.restore_gdn_states(&seed);
        assert_eq!(state.snapshot_gdn_states(), seed);
        assert_eq!(reference.snapshot_gdn_states(), seed);

        let predict = |logits: &[f32]| crate::sampling::argmax_f32_first_wins(logits);
        let after_pending = reference
            .try_forward_step(0, pos)
            .expect("reference pending");
        assert_eq!(predict(&after_pending), 1);
        let after_accepted = reference
            .try_forward_step(1, pos + 1)
            .expect("reference accepted draft");
        assert_eq!(predict(&after_accepted), 2);
        let committed_gdn = reference.snapshot_gdn_states();
        let committed_kv = snapshot_kv_bytes(&reference);
        reference
            .try_forward_step(2, pos + 2)
            .expect("reference stop token");
        let overrun_gdn = reference.snapshot_gdn_states();
        assert_ne!(
            committed_gdn[0].0, overrun_gdn[0].0,
            "S must distinguish boundaries"
        );
        assert_ne!(
            committed_gdn[0].1, overrun_gdn[0].1,
            "conv must distinguish boundaries"
        );

        let mut metrics = SpeculativeMetrics::self_spec();
        let round = state.speculative_round(&mut metrics, 0, 3, &|id| id == 2);
        let SpeculativeMetrics::SelfSpec(metrics) = metrics else {
            panic!("self-spec route required");
        };
        assert_eq!(metrics.verify_calls, 1, "real verification must run");
        assert_eq!(metrics.draft_forwards, 2);
        assert_eq!(metrics.fallback_tokens, 0);
        assert_eq!(metrics.accepted_extra_tokens, 1);
        assert_eq!(metrics.rounds, 0, "accepted stop is not a continuing round");
        assert_eq!(round.committed, [0, 1]);
        assert_eq!(round.next, Some(2));
        assert!(!round.cache_full);
        let actual_gdn = state.snapshot_gdn_states();
        eprintln!(
            "[SELF_SPEC_STOP_WITNESS] verified accepted draft then EOS; S_prefix={} conv_prefix={} S_overrun={} conv_overrun={}",
            actual_gdn[0].0 == committed_gdn[0].0,
            actual_gdn[0].1 == committed_gdn[0].1,
            actual_gdn[0].0 == overrun_gdn[0].0,
            actual_gdn[0].1 == overrun_gdn[0].1,
        );

        assert_eq!(state.session.kv_cache.seq_len, pos + round.committed.len());
        let actual_kv = snapshot_kv_bytes(&state);
        assert_eq!(actual_kv.len(), committed_kv.len());
        assert!(!actual_kv.is_empty());
        for (actual, expected) in actual_kv.iter().zip(&committed_kv) {
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.len() % state.session.kv_cache.max_cache_len, 0);
            let row_bytes = actual.len() / state.session.kv_cache.max_cache_len;
            let used_bytes = row_bytes * (pos + round.committed.len());
            assert_eq!(&actual[..used_bytes], &expected[..used_bytes]);
            assert!(expected[..used_bytes].iter().any(|&byte| byte != 0));
        }
        assert_eq!(actual_gdn, committed_gdn);
        eprintln!("[SELF_SPEC_STOP_STATE_FIXED] committed boundary restored");
    });
}

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
        lattice_compute::option_scoring::option_scores_from_logits(&full, &OPTION_LETTERS)
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
            Err(lattice_inference_core::InferenceError::InvalidInput(_)) => {}
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

#[test]
fn generate_with_speculation_accepts_multi_token_moe_prompt_without_panicking() {
    let _guard = gpu_test_lock();
    let has_device = metal::Device::system_default().is_some();
    assert!(has_device, "Metal device required for MoE prefill test");
    let (cfg, weights) = tiny_moe_prefill_fixture();
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 16).expect("moe prefill fixture must construct");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.generate_with_speculation(&[1u32, 3, 5], 4, 0, 2, 2)
    }));
    let generated = outcome
        .expect("per-token MoE speculation must not panic")
        .expect("multi-token MoE prompt must support per-token speculation");
    assert!(!generated.is_empty());
    assert!(generated.len() <= 4);
    assert!(generated.iter().all(|&id| (id as usize) < cfg.vocab_size));
    assert!(state.session.position() >= 3);
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

fn moe_prefill_parity_fixture() -> (Qwen35Config, ModelWeights) {
    let (mut cfg, mut weights) = super::prefix_cache_route::patterned_model();
    let (moe_cfg, _) = tiny_moe_prefill_fixture();
    assert_eq!(cfg.hidden_size, moe_cfg.hidden_size);
    cfg.num_experts = moe_cfg.num_experts;
    cfg.num_experts_per_tok = moe_cfg.num_experts_per_tok;
    cfg.moe_intermediate_size = moe_cfg.moe_intermediate_size;
    cfg.shared_expert_intermediate_size = moe_cfg.shared_expert_intermediate_size;
    for (_, common) in &mut weights.layers {
        let (_, mut donor) = tiny_moe_prefill_fixture();
        let (_, donor_common) = donor.layers.pop().expect("one MoE donor layer");
        assert!(matches!(&donor_common.ffn, FeedForwardWeights::Moe(_)));
        common.ffn = donor_common.ffn;
    }
    (cfg, weights)
}

fn require_moe_prefill_parity_device() {
    let has_device = Device::system_default().is_some();
    assert!(has_device, "Metal device required for MoE prefill parity");
}

fn assert_moe_prefill_state_eq(
    actual: &MetalQwen35State,
    expected: &MetalQwen35State,
    rows: usize,
) {
    use crate::speculative::MtpTargetVerifier as _;

    assert_eq!(actual.session.position(), rows);
    assert_eq!(expected.session.position(), rows);
    assert_eq!(actual.use_kv_f16, expected.use_kv_f16);
    assert_eq!(actual.session.kv_cache.seq_len, rows);
    assert_eq!(expected.session.kv_cache.seq_len, rows);
    assert_eq!(
        actual.session.kv_cache.max_cache_len,
        expected.session.kv_cache.max_cache_len
    );
    // Only the common continuation fills capacity; avoid reading unwritten KV rows.
    if rows == expected.session.kv_cache.max_cache_len {
        for state in [actual, expected] {
            let cache = &state.session.kv_cache;
            assert_eq!(cache.seq_len, cache.max_cache_len);
            let bytes = rows * cache.kv_dim * if state.use_kv_f16 { 2 } else { 4 };
            for buffer in cache.k_bufs.iter().chain(&cache.v_bufs) {
                assert_eq!(buffer.length() as usize, bytes);
            }
        }
        let expected_kv = snapshot_kv_bytes(expected);
        assert!(!expected_kv.is_empty());
        assert!(
            expected_kv
                .iter()
                .all(|buffer| buffer.iter().any(|&b| b != 0))
        );
        let actual_kv = snapshot_kv_bytes(actual);
        assert_eq!(actual_kv, expected_kv);
    }
    let expected_gdn = expected.snapshot_gdn_states();
    let actual_gdn = actual.snapshot_gdn_states();
    assert_eq!(expected_gdn.len(), 3);
    assert_eq!(actual_gdn.len(), expected_gdn.len());
    for ((actual_s, actual_conv), (expected_s, expected_conv)) in
        actual_gdn.iter().zip(&expected_gdn)
    {
        for values in [expected_s, expected_conv] {
            assert!(!values.is_empty());
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(values.iter().any(|&value| value != 0.0));
        }
        assert_eq!(f32_bits(actual_s), f32_bits(expected_s));
        assert_eq!(f32_bits(actual_conv), f32_bits(expected_conv));
    }
}

fn moe_prefill_manual_rows(state: &mut MetalQwen35State, tokens: &[u32]) -> Vec<Vec<f32>> {
    tokens
        .iter()
        .enumerate()
        .map(|(position, &token)| {
            let row = state.forward_step(token, position);
            assert_eq!(row.len(), state.engine.config.vocab_size);
            assert!(row.iter().all(|value| value.is_finite()));
            row
        })
        .collect()
}

#[test]
fn moe_prefill_last_all_positions_and_option_scores_match_manual_steps_bitwise() {
    let _gpu = gpu_test_lock();
    require_moe_prefill_parity_device();
    let (cfg, weights) = moe_prefill_parity_fixture();
    let tokens = [1u32, 4, 2, 7, 3];
    for arm in ["last", "all", "all_compact", "scores", "hidden"] {
        let mut reference =
            MetalQwen35State::new(&weights, &cfg, tokens.len() + 1).expect("reference fixture");
        assert_eq!(reference.session.kv_cache.max_cache_len, tokens.len() + 1);
        let rows = moe_prefill_manual_rows(&mut reference, &tokens);
        assert!(
            rows.windows(2)
                .all(|pair| f32_bits(&pair[0]) != f32_bits(&pair[1]))
        );
        assert_ne!(f32_bits(&rows[0]), f32_bits(rows.last().unwrap()));
        let last = rows.last().expect("nonempty prompt");
        let mut actual =
            MetalQwen35State::new(&weights, &cfg, tokens.len() + 1).expect("MoE fixture");
        assert_eq!(actual.session.kv_cache.max_cache_len, tokens.len() + 1);
        assert!(actual.engine.has_moe_layer);
        assert!(actual.lora.is_none());
        if arm == "all_compact" {
            let mut seed = MetalQwen35State::new(&weights, &cfg, tokens.len() + 1)
                .expect("stale candidate seed");
            seed.session.compact_route = GpuTopkRoute::BlockArgmax;
            seed.session.compact_topk = 1;
            let seed_logits = seed.forward_step(tokens[0], 0);
            assert!(seed_logits.is_empty());
            assert!(!seed.session.compact_result.is_empty());
            std::mem::swap(
                &mut actual.session.compact_result,
                &mut seed.session.compact_result,
            );
            actual.session.compact_route = GpuTopkRoute::BlockArgmax;
            actual.session.compact_topk = 1;
            let refused = actual.try_forward_step_with_dense_logits(cfg.vocab_size as u32, 0);
            assert!(matches!(
                refused,
                Err(lattice_inference_core::InferenceError::InvalidInput(_))
            ));
            assert_eq!(actual.session.compact_route, GpuTopkRoute::BlockArgmax);
            assert_eq!(actual.session.compact_topk, 1);
            assert!(!actual.session.compact_result.is_empty());
            assert_eq!(actual.session.position(), 0);
        }
        match arm {
            "last" => {
                let logits = actual.try_forward_prefill(&tokens).expect("MoE prefill");
                assert_eq!(logits.len(), cfg.vocab_size);
                assert!(logits.iter().all(|value| value.is_finite()));
                assert_eq!(f32_bits(&logits), f32_bits(last));
            }
            "all" | "all_compact" => {
                let logits = actual
                    .forward_prefill_all_logits(&tokens)
                    .expect("MoE all positions");
                assert_eq!(logits.len(), tokens.len() * cfg.vocab_size);
                assert!(logits.iter().all(|value| value.is_finite()));
                let expected: Vec<f32> = rows.iter().flatten().copied().collect();
                assert_eq!(f32_bits(&logits), f32_bits(&expected));
            }
            "hidden" => {
                let mut hidden_reference = MetalQwen35State::new(&weights, &cfg, tokens.len() + 1)
                    .expect("hidden reference fixture");
                for (position, &token) in tokens[..tokens.len() - 1].iter().enumerate() {
                    let _ = hidden_reference.forward_step(token, position);
                }
                let (expected_logits, expected_hidden) = hidden_reference
                    .forward_step_with_hidden(tokens[tokens.len() - 1], tokens.len() - 1)
                    .expect("manual hidden step");
                let (logits, hidden) = actual
                    .forward_prefill_with_hidden(&tokens)
                    .expect("MoE hidden prefill");
                assert_eq!(logits.len(), cfg.vocab_size);
                assert_eq!(hidden.len(), cfg.hidden_size);
                assert_eq!(f32_bits(&logits), f32_bits(&expected_logits));
                assert_eq!(f32_bits(&hidden), f32_bits(&expected_hidden));
            }
            "scores" => {
                let _ = actual.forward_step(11, 0);
                let letters = [2u32, 5, 9, 13];
                let scores = actual
                    .score_option_letters(&tokens, &letters)
                    .expect("MoE option scores");
                let selected: Vec<f32> = letters.iter().map(|&id| last[id as usize]).collect();
                assert_eq!(f32_bits(&scores.logits), f32_bits(&selected));
                assert_eq!(
                    scores,
                    lattice_compute::option_scoring::option_scores_from_logits(last, &letters)
                        .unwrap()
                );
            }
            _ => unreachable!(),
        }
        if arm == "all_compact" {
            assert_eq!(actual.session.compact_route, GpuTopkRoute::BlockArgmax);
            assert_eq!(actual.session.compact_topk, 1);
            assert!(actual.session.compact_result.is_empty());
            actual.session.compact_route = GpuTopkRoute::CpuFallback;
            actual.session.compact_topk = 0;
        }
        assert_moe_prefill_state_eq(&actual, &reference, tokens.len());
        let expected_next = reference.forward_step(9, tokens.len());
        let actual_next = actual.forward_step(9, tokens.len());
        assert_eq!(actual_next.len(), cfg.vocab_size);
        assert!(actual_next.iter().all(|value| value.is_finite()));
        assert_eq!(f32_bits(&actual_next), f32_bits(&expected_next));
        assert_moe_prefill_state_eq(&actual, &reference, tokens.len() + 1);
    }
}

#[test]
fn moe_prefill_reused_prefix_last_and_all_positions_match_manual_steps_bitwise() {
    let _gpu = gpu_test_lock();
    require_moe_prefill_parity_device();
    let (cfg, weights) = moe_prefill_parity_fixture();
    let tokens = [1u32, 4, 2, 7, 3];
    let prefix_len = 2;
    for (all_positions, compact) in [(false, false), (true, false), (true, true)] {
        let mut reference =
            MetalQwen35State::new(&weights, &cfg, tokens.len() + 1).expect("reference fixture");
        assert_eq!(reference.session.kv_cache.max_cache_len, tokens.len() + 1);
        let rows = moe_prefill_manual_rows(&mut reference, &tokens);
        assert!(
            rows.windows(2)
                .all(|pair| f32_bits(&pair[0]) != f32_bits(&pair[1]))
        );
        assert_ne!(f32_bits(&rows[0]), f32_bits(rows.last().unwrap()));
        let mut actual =
            MetalQwen35State::new(&weights, &cfg, tokens.len() + 1).expect("MoE fixture");
        assert_eq!(actual.session.kv_cache.max_cache_len, tokens.len() + 1);
        assert!(actual.engine.has_moe_layer);
        assert!(actual.lora.is_none());
        let prefix_rows = moe_prefill_manual_rows(&mut actual, &tokens[..prefix_len]);
        for (actual_row, expected_row) in prefix_rows.iter().zip(&rows) {
            assert_eq!(f32_bits(actual_row), f32_bits(expected_row));
        }
        let mut prefix_reference =
            MetalQwen35State::new(&weights, &cfg, tokens.len() + 1).expect("prefix reference");
        let _ = moe_prefill_manual_rows(&mut prefix_reference, &tokens[..prefix_len]);
        assert_moe_prefill_state_eq(&actual, &prefix_reference, prefix_len);
        let slot = lattice_compute::kv_cache::CrossTurnSlotId::DEFAULT;
        let tokenizer = single_char_vocab_tokenizer();
        let metadata = actual.cross_turn_metadata(&tokenizer);
        actual
            .save_cross_turn_prefix(slot, metadata, tokens[..prefix_len].to_vec(), None)
            .expect("save nonempty MoE prefix");
        let metadata = actual.cross_turn_metadata(&tokenizer);
        let plan = actual.plan_cross_turn_reuse(slot, &metadata, &tokens);
        assert_eq!(
            plan.mode,
            lattice_compute::kv_cache::PrefixReuseMode::ExactAppend
        );
        assert_eq!(plan.reusable_len, prefix_len);
        assert_eq!(plan.suffix_start, prefix_len);
        actual
            .plan_prefix_request(&tokens, &plan)
            .expect("valid suffix plan");
        let restored = actual
            .restore_cross_turn_prefix(slot, &plan)
            .expect("restore prefix");
        assert!(restored.is_some());
        let saved_entry_removed = actual.cross_turn_prefix_cache.get(slot).is_none();
        assert!(saved_entry_removed);
        assert_moe_prefill_state_eq(&actual, &prefix_reference, prefix_len);
        if compact {
            actual.session.compact_route = GpuTopkRoute::BlockArgmax;
            actual.session.compact_topk = 1;
        }
        let logits = actual
            .forward_prefill_from(&tokens[prefix_len..], prefix_len, all_positions)
            .expect("multi-token MoE suffix after a reused prefix");
        let expected: Vec<f32> = if all_positions {
            rows[prefix_len..].iter().flatten().copied().collect()
        } else {
            rows.last().unwrap().clone()
        };
        assert_eq!(
            logits.len(),
            if all_positions {
                (tokens.len() - prefix_len) * cfg.vocab_size
            } else {
                cfg.vocab_size
            }
        );
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(f32_bits(&logits), f32_bits(&expected));
        if compact {
            assert_eq!(actual.session.compact_route, GpuTopkRoute::BlockArgmax);
            assert_eq!(actual.session.compact_topk, 1);
            assert!(actual.session.compact_result.is_empty());
            actual.session.compact_route = GpuTopkRoute::CpuFallback;
            actual.session.compact_topk = 0;
        }
        assert_moe_prefill_state_eq(&actual, &reference, tokens.len());
        let expected_next = reference.forward_step(9, tokens.len());
        let actual_next = actual.forward_step(9, tokens.len());
        assert_eq!(actual_next.len(), cfg.vocab_size);
        assert!(actual_next.iter().all(|value| value.is_finite()));
        assert_eq!(f32_bits(&actual_next), f32_bits(&expected_next));
        assert_moe_prefill_state_eq(&actual, &reference, tokens.len() + 1);
    }
}

#[test]
fn moe_all_position_overflow_refuses_before_advancing_state() {
    let _gpu = gpu_test_lock();
    require_moe_prefill_parity_device();
    let (cfg, weights) = moe_prefill_parity_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 8).expect("MoE fixture");
    state.engine.config.vocab_size = usize::MAX;
    let before_position = state.session.position();
    let before_kv_len = state.session.kv_cache.seq_len;
    let outcome = state.forward_prefill_all_logits(&[1, 2]);
    assert!(matches!(
        outcome,
        Err(lattice_inference_core::InferenceError::InvalidInput(ref message))
            if message.contains("logit buffer size exceeds the addressable limit")
    ));
    assert_eq!(state.session.position(), before_position);
    assert_eq!(state.session.kv_cache.seq_len, before_kv_len);
}

#[test]
fn moe_compact_prefill_matches_manual_final_candidates() {
    let _gpu = gpu_test_lock();
    require_moe_prefill_parity_device();
    let (cfg, weights) = moe_prefill_parity_fixture();
    let tokens = [1u32, 4, 2, 7, 3];
    let mut expected = MetalQwen35State::new(&weights, &cfg, 8).expect("manual fixture");
    expected.session.compact_route = GpuTopkRoute::BlockArgmax;
    expected.session.compact_topk = 1;
    for (position, &token) in tokens.iter().enumerate() {
        let logits = expected.forward_step(token, position);
        assert!(logits.is_empty());
    }
    let expected_candidates: Vec<_> = expected
        .session
        .compact_result
        .iter()
        .map(|candidate| (candidate.token_id, candidate.logit.to_bits()))
        .collect();
    assert_eq!(expected_candidates.len(), 1);
    let mut actual = MetalQwen35State::new(&weights, &cfg, 8).expect("prefill fixture");
    actual.session.compact_route = GpuTopkRoute::BlockArgmax;
    actual.session.compact_topk = 1;
    let logits = actual
        .try_forward_prefill(&tokens)
        .expect("compact MoE prefill");
    assert!(logits.is_empty());
    let actual_candidates: Vec<_> = actual
        .session
        .compact_result
        .iter()
        .map(|candidate| (candidate.token_id, candidate.logit.to_bits()))
        .collect();
    assert_eq!(actual_candidates, expected_candidates);
    assert_eq!(actual.session.position(), tokens.len());
}

#[test]
fn moe_single_token_all_positions_is_dense_under_compact_route() {
    let _gpu = gpu_test_lock();
    require_moe_prefill_parity_device();
    let (cfg, weights) = moe_prefill_parity_fixture();
    let mut expected = MetalQwen35State::new(&weights, &cfg, 8).expect("reference fixture");
    let expected_logits = expected.forward_step(1, 0);
    let mut actual = MetalQwen35State::new(&weights, &cfg, 8).expect("prefill fixture");
    actual.session.compact_route = GpuTopkRoute::BlockArgmax;
    actual.session.compact_topk = 1;
    let logits = actual
        .forward_prefill_all_logits(&[1])
        .expect("all-position prefill");
    assert_eq!(f32_bits(&logits), f32_bits(&expected_logits));
    assert_eq!(actual.session.compact_route, GpuTopkRoute::BlockArgmax);
    assert_eq!(actual.session.compact_topk, 1);
    assert!(actual.session.compact_result.is_empty());
    let mut suffix = MetalQwen35State::new(&weights, &cfg, 8).expect("suffix fixture");
    suffix.session.compact_route = GpuTopkRoute::BlockArgmax;
    suffix.session.compact_topk = 1;
    let suffix_logits = suffix
        .forward_prefill_from(&[1], 0, true)
        .expect("all-position suffix");
    assert_eq!(f32_bits(&suffix_logits), f32_bits(&expected_logits));
    assert_eq!(suffix.session.compact_route, GpuTopkRoute::BlockArgmax);
    assert_eq!(suffix.session.compact_topk, 1);
    assert!(suffix.session.compact_result.is_empty());
}

#[test]
fn moe_compact_driver_prefill_samples_once_from_the_final_prompt_state() {
    use crate::decoder::qwen_metal::{MetalEntryProfile, QwenMetalSession, ReadbackMode};
    use crate::decoder::{DecoderSession, SelectOutcome, SelectionRequest};
    use crate::model::qwen35::GenerationPlan;

    let _gpu = gpu_test_lock();
    require_moe_prefill_parity_device();
    let (cfg, weights) = moe_prefill_parity_fixture();
    let tokens = [1u32, 4, 2, 7, 3];
    let mut generation = GenerateConfig::default();
    generation.temperature = 0.8;
    generation.top_k = 4;
    generation.top_p = 1.0;
    generation.repetition_penalty = 1.0;
    generation.enable_mtp = Some(false);
    generation.enable_thinking = false;
    let environment = SamplingRouteEnvironment {
        compact: true,
        selection: false,
        approximate_top_p: false,
    };
    let mut sampled = Vec::new();
    for seed in [
        1u64,
        0x1000_0000,
        0x1000_0000_0000,
        0x1234_5678_9abc_def0,
        0xdead_beef,
        0x8765_4321_1234_5678,
        0x5555_5555_5555_5555,
        0xaaaa_aaaa_aaaa_aaaa,
    ] {
        let plan = || GenerationPlan {
            prompt_len: tokens.len(),
            required_capacity: tokens.len() + 1,
            prompt_ids: tokens.to_vec(),
            rng_state: seed,
        };
        let mut actual_state = MetalQwen35State::new(&weights, &cfg, 8).expect("actual fixture");
        let mut actual = QwenMetalSession::with_route_environment(
            &mut actual_state,
            plan(),
            &generation,
            MetalEntryProfile::PrefixCacheStreaming,
            environment,
        )
        .expect("compact session");
        assert!(matches!(actual.mode(), ReadbackMode::Compact { .. }));
        let _ = actual.prefill(&|| false).expect("multi-token prefill");
        let selected = actual
            .select(&SelectionRequest::new(&generation, &tokens))
            .expect("sample");
        let actual_id = match selected {
            SelectOutcome::Candidate(candidate) => candidate.candidate_id,
            _ => panic!("expected a sampled candidate"),
        };
        drop(actual);

        let mut reference_state =
            MetalQwen35State::new(&weights, &cfg, 8).expect("reference fixture");
        let _ = moe_prefill_manual_rows(&mut reference_state, &tokens[..tokens.len() - 1]);
        let mut reference = QwenMetalSession::over_restored_state(
            &mut reference_state,
            plan(),
            tokens.len() - 1,
            &generation,
            environment,
        )
        .expect("reference compact session");
        assert!(matches!(reference.mode(), ReadbackMode::Compact { .. }));
        let _ = reference.prefill(&|| false).expect("last-token prefill");
        let selected = reference
            .select(&SelectionRequest::new(&generation, &tokens))
            .expect("sample");
        let reference_id = match selected {
            SelectOutcome::Candidate(candidate) => candidate.candidate_id,
            _ => panic!("expected a sampled candidate"),
        };
        assert_eq!(actual_id, reference_id, "seed {seed}");
        sampled.push(actual_id);
    }
    assert!(
        sampled.iter().any(|&id| id != sampled[0]),
        "fixture must exercise a non-degenerate sampling distribution"
    );
}
