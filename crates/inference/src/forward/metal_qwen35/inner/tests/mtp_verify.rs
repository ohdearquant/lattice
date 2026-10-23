//! MTP draft and verify: hidden-row verification and the generic loop on a real checkpoint.

use super::*;
// ---------------------------------------------------------------------
// Post-final-norm hidden rows: `MtpTargetVerifier::verify_tokens_with_hidden`
// ---------------------------------------------------------------------

/// Agreement tolerance between the CPU final norm and an independent f64 reference.
const POST_FINAL_TOL: f32 = 1e-4;

/// `option_scoring_fixture` with patterned embeddings, a patterned, nonzero
/// final-norm scale and patterned, nonzero GDN projections: hidden states are
/// non-trivial in many channels, differ across tokens and positions, the final norm
/// visibly changes them, and the recurrent state is nonzero after a forward, so a
/// GDN snapshot comparison can tell two histories apart.
fn post_final_fixture() -> (Qwen35Config, ModelWeights) {
    let (cfg, mut weights) = option_scoring_fixture();
    let hidden = cfg.hidden_size;
    let patterned = |n: usize, seed: usize, amp: f32| -> Vec<f32> {
        (0..n)
            .map(|i| amp * ((i + seed) as f32 * 0.37).sin())
            .collect()
    };
    for (layer, (attention, _)) in weights.layers.iter_mut().enumerate() {
        if let AttentionWeights::Linear(gdn) = attention {
            let seed = 1000 * (layer + 1);
            gdn.in_proj_qkv = patterned(gdn.in_proj_qkv.len(), seed + 1, 0.05);
            gdn.in_proj_z = patterned(gdn.in_proj_z.len(), seed + 2, 0.05);
            gdn.in_proj_b = patterned(gdn.in_proj_b.len(), seed + 3, 0.05);
            gdn.in_proj_a = patterned(gdn.in_proj_a.len(), seed + 4, 0.05);
            gdn.conv1d_weight = patterned(gdn.conv1d_weight.len(), seed + 5, 0.5);
            gdn.out_proj = patterned(gdn.out_proj.len(), seed + 6, 0.05);
        }
    }
    for token in 0..cfg.vocab_size {
        let row = &mut weights.embed_tokens[token * hidden..(token + 1) * hidden];
        for (c, v) in row.iter_mut().enumerate().take(24).skip(2) {
            *v = 0.8 * ((token * 24 + c) as f32 * 0.37).sin();
        }
    }
    weights.final_norm = (0..hidden)
        .map(|i| 0.4 * (i as f32 * 0.29 + 0.5).sin())
        .collect();
    (cfg, weights)
}

/// A state with the speculative checkpoint pool, so `verify_tokens*` can run.
fn post_final_state(cfg: &Qwen35Config, weights: &ModelWeights) -> MetalQwen35State {
    with_self_spec_env(|| {
        MetalQwen35State::new(weights, cfg, 32).expect("post-final fixture constructs")
    })
}

/// `false` (after printing the skip marker) when this host has no Metal device.
/// Callers hold `gpu_test_lock()`.
fn metal_device_or_skip(context: &str) -> bool {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    if Device::system_default().is_some() {
        return true;
    }
    eprintln!("[METAL_TEST_SKIP] context={context} reason=no_metal_device");
    assert!(
        !enforce,
        "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present ({context})"
    );
    false
}

/// The shifted final RMSNorm `x / rms(x) * (1 + gamma)` in f64, from the fixture's own
/// scale vector (not the engine's buffer).
fn reference_final_norm(pre: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let mean_square = pre
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / pre.len() as f64;
    let inv_rms = 1.0 / (mean_square + f64::from(eps)).sqrt();
    pre.iter()
        .zip(gamma)
        .map(|(&v, &g)| (f64::from(v) * inv_rms * (1.0 + f64::from(g))) as f32)
        .collect()
}

const POST_FINAL_TOKENS: [u32; 5] = [3, 8, 14, 5, 21];

/// A GDN snapshot comparison is vacuous when every state is zero (#1838): require
/// recurrent state in every GDN layer, in both the S matrix and the conv buffer.
fn assert_gdn_state_is_nonzero(snapshot: &crate::attention::gdn::GdnSnapshot, what: &str) {
    assert!(!snapshot.is_empty(), "{what}: no GDN layers");
    for (layer, (s, conv)) in snapshot.iter().enumerate() {
        assert!(
            s.iter().any(|&v| v != 0.0),
            "{what}: layer {layer} S matrix is all zero, so equality proves nothing"
        );
        assert!(
            conv.iter().any(|&v| v != 0.0),
            "{what}: layer {layer} conv buffer is all zero, so equality proves nothing"
        );
    }
}

#[test]
fn verify_tokens_with_hidden_rows_equal_independent_post_final_hidden() {
    use crate::speculative::MtpTargetVerifier as _;

    let _gpu = gpu_test_lock();
    if !metal_device_or_skip("verify_tokens_with_hidden_rows_equal_independent_post_final_hidden") {
        return;
    }
    let (cfg, weights) = post_final_fixture();
    let hidden = cfg.hidden_size;
    let tokens = POST_FINAL_TOKENS;

    let mut verified = post_final_state(&cfg, &weights);
    assert_eq!(
        verified.mtp_hidden_tap(),
        None,
        "the fixture carries no MTP weights, so the tap cannot decide the rows"
    );
    assert!(!verified.checkpoint_is_rotated());
    let (logits, rows) = verified
        .verify_tokens_with_hidden(&tokens, 0)
        .expect("verify with hidden rows");
    let rows = rows.expect("an unrotated state without MTP weights supplies every row");
    assert_eq!(rows.len(), tokens.len());
    assert_eq!(logits.len(), tokens.len());
    assert!(rows.iter().all(|r| r.len() == hidden));

    // Independent route: the same tokens one at a time, then the final norm in f64.
    let mut sequential = post_final_state(&cfg, &weights);
    for (i, &token) in tokens.iter().enumerate() {
        let (_, pre) = sequential
            .forward_step_with_hidden(token, i)
            .expect("sequential reference step");
        let expected = reference_final_norm(&pre, &weights.final_norm, cfg.rms_norm_eps);
        assert!(
            expected.iter().any(|&v| v != 0.0) && pre.iter().any(|&v| v != 0.0),
            "row {i}: the reference hidden must carry data"
        );
        let diff = max_abs_diff(&rows[i], &expected);
        assert!(
            diff <= POST_FINAL_TOL,
            "row {i}: returned row differs from the independent post-final hidden by {diff}"
        );

        // The row is the vector the output head reads: recompute the logits from it
        // with the unquantized tied embedding. The GPU head uses Q8 weights, so the
        // bound is a fraction of the logit scale, far tighter than a wrong norm gives.
        let reference_logits: Vec<f32> = (0..cfg.vocab_size)
            .map(|v| {
                weights.embed_tokens[v * hidden..(v + 1) * hidden]
                    .iter()
                    .zip(&rows[i])
                    .map(|(&e, &h)| f64::from(e) * f64::from(h))
                    .sum::<f64>() as f32
            })
            .collect();
        let scale = reference_logits.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let head_diff = max_abs_diff(&logits[i], &reference_logits);
        assert!(
            scale > 1.0,
            "row {i}: logit scale {scale} is too small to discriminate"
        );
        assert!(
            head_diff <= 0.05 * scale,
            "row {i}: logits differ from head(row) by {head_diff} on scale {scale}"
        );
    }

    // Same forward as `verify_tokens`: identical logits and identical resulting state.
    let mut plain = post_final_state(&cfg, &weights);
    let plain_logits = plain.verify_tokens(&tokens, 0).expect("plain verify");
    let bits = |l: &[Vec<f32>]| -> Vec<Vec<u32>> { l.iter().map(|r| f32_bits(r)).collect() };
    assert_eq!(bits(&logits), bits(&plain_logits));
    assert_eq!(
        verified.session.kv_cache.seq_len,
        plain.session.kv_cache.seq_len
    );
    assert_eq!(verified.snapshot_gdn_states(), plain.snapshot_gdn_states());
    assert_gdn_state_is_nonzero(&verified.snapshot_gdn_states(), "verified");
    assert_gdn_state_is_nonzero(&plain.snapshot_gdn_states(), "plain");
}

#[test]
fn post_final_rows_differ_from_pre_final_rows_on_the_fixture() {
    use crate::speculative::MtpTargetVerifier as _;

    let _gpu = gpu_test_lock();
    if !metal_device_or_skip("post_final_rows_differ_from_pre_final_rows_on_the_fixture") {
        return;
    }
    let (cfg, weights) = post_final_fixture();
    let tokens = POST_FINAL_TOKENS;
    let mut verified = post_final_state(&cfg, &weights);
    let (_, rows) = verified
        .verify_tokens_with_hidden(&tokens, 0)
        .expect("verify with hidden rows");
    let rows = rows.expect("rows");

    let mut sequential = post_final_state(&cfg, &weights);
    for (i, &token) in tokens.iter().enumerate() {
        let (_, pre) = sequential
            .forward_step_with_hidden(token, i)
            .expect("sequential reference step");
        assert!(
            pre.iter().any(|&v| v != 0.0),
            "row {i}: pre-final hidden is empty"
        );
        let diff = max_abs_diff(&rows[i], &pre);
        assert!(
            diff > 10.0 * POST_FINAL_TOL,
            "row {i}: post-final row is within {diff} of the pre-final hidden; the \
                     fixture cannot tell the two apart"
        );
    }
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            assert!(
                max_abs_diff(&rows[i], &rows[j]) > 10.0 * POST_FINAL_TOL,
                "rows {i} and {j} must differ, or a shifted set would pass"
            );
        }
    }
}

#[test]
fn verify_tokens_with_hidden_on_rotated_checkpoint_matches_plain_verify_and_returns_no_rows() {
    use crate::speculative::MtpTargetVerifier as _;

    let _gpu = gpu_test_lock();
    if !metal_device_or_skip(
        "verify_tokens_with_hidden_on_rotated_checkpoint_matches_plain_verify_and_returns_no_rows",
    ) {
        return;
    }
    let (cfg, weights) = post_final_fixture();
    let hidden = cfg.hidden_size;
    let tokens = POST_FINAL_TOKENS;

    let mut plain = post_final_state(&cfg, &weights);
    let plain_logits = plain.verify_tokens(&tokens, 0).expect("plain verify");
    let bits = |l: &[Vec<f32>]| -> Vec<Vec<u32>> { l.iter().map(|r| f32_bits(r)).collect() };

    // Both ways a state can carry rotation, neither of which loads MTP weights.
    for arm in ["seed", "rotation"] {
        let mut rotated = post_final_state(&cfg, &weights);
        match arm {
            "seed" => rotated.engine.base_quarot_seed = Some(42),
            _ => {
                rotated.engine.quarot_rotation = Some(
                    crate::quant::quarot::hadamard::RandomizedHadamard::new(0x51, hidden)
                        .expect("rotation"),
                )
            }
        }
        assert!(rotated.checkpoint_is_rotated(), "{arm}");
        assert_eq!(
            rotated.mtp_hidden_tap(),
            None,
            "{arm}: without MTP weights the tap cannot say the checkpoint is rotated"
        );
        assert_eq!(rotated.session.kv_cache.seq_len, 0, "{arm}");
        let gdn_before = rotated.snapshot_gdn_states();
        assert_eq!(
            gdn_before,
            post_final_state(&cfg, &weights).snapshot_gdn_states()
        );

        let (logits, rows) = rotated
            .verify_tokens_with_hidden(&tokens, 0)
            .expect("verify on a rotated state");
        assert!(
            rows.is_none(),
            "{arm}: a rotated state must not return rows"
        );
        assert!(rotated.last_hidden().is_none(), "{arm}");

        // The call is exactly a plain `verify_tokens`: same logits, cursor and GDN
        // state, so a caller that falls back to the logits alone sees nothing odd.
        assert_eq!(bits(&logits), bits(&plain_logits), "{arm}");
        assert_eq!(
            rotated.session.kv_cache.seq_len, plain.session.kv_cache.seq_len,
            "{arm}"
        );
        assert_eq!(
            rotated.snapshot_gdn_states(),
            plain.snapshot_gdn_states(),
            "{arm}"
        );
        assert_gdn_state_is_nonzero(&rotated.snapshot_gdn_states(), arm);
        assert_eq!(
            rotated.session.kv_cache.seq_len,
            tokens.len(),
            "{arm}: the forward must have run, or the equalities above prove nothing"
        );
        assert!(
            logits.iter().flatten().any(|&v| v != 0.0),
            "{arm}: logits must carry data"
        );
    }
}

/// The independent post-final hidden of the token at the end of `tokens`: the same
/// tokens one step at a time on a fresh state, then the final norm in f64.
fn independent_post_final_hidden(
    cfg: &Qwen35Config,
    weights: &ModelWeights,
    tokens: &[u32],
) -> Vec<f32> {
    let mut sequential = post_final_state(cfg, weights);
    let mut pre = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        pre = sequential
            .forward_step_with_hidden(token, i)
            .expect("independent step")
            .1;
    }
    assert!(
        pre.iter().any(|&v| v != 0.0),
        "the reference must carry data"
    );
    reference_final_norm(&pre, &weights.final_norm, cfg.rms_norm_eps)
}

/// A batched prefill and a per-token run take different GEMM paths, so their hidden
/// states agree to ~1e-3, not 1e-4; a hidden state of the neighbouring token is far
/// outside this bound (checked where it is used).
const PREFILL_VS_STEP_TOL: f32 = 5e-3;

fn assert_last_hidden_is(state: &MetalQwen35State, expected: &[f32], context: &str) -> Vec<f32> {
    assert_last_hidden_within(state, expected, POST_FINAL_TOL, context)
}

fn assert_last_hidden_within(
    state: &MetalQwen35State,
    expected: &[f32],
    tol: f32,
    context: &str,
) -> Vec<f32> {
    use crate::speculative::MtpTargetVerifier as _;
    let got = state
        .last_hidden()
        .unwrap_or_else(|| panic!("{context}: last_hidden must be Some"));
    let diff = max_abs_diff(&got, expected);
    assert!(
        diff <= tol,
        "{context}: last_hidden differs from the independent post-final hidden by {diff}"
    );
    got
}

#[test]
fn last_hidden_after_a_capturing_prefill_equals_the_independent_post_final_hidden() {
    use crate::speculative::MtpTargetVerifier as _;

    let _gpu = gpu_test_lock();
    if !metal_device_or_skip(
        "last_hidden_after_a_capturing_prefill_equals_the_independent_post_final_hidden",
    ) {
        return;
    }
    let (cfg, weights) = post_final_fixture();
    let tokens = POST_FINAL_TOKENS;
    let expected = independent_post_final_hidden(&cfg, &weights, &tokens);
    let earlier = independent_post_final_hidden(&cfg, &weights, &tokens[..tokens.len() - 1]);
    assert!(
        max_abs_diff(&expected, &earlier) > 4.0 * PREFILL_VS_STEP_TOL,
        "the fixture must tell the last token's hidden from the one before it"
    );

    let fresh = post_final_state(&cfg, &weights);
    assert!(fresh.last_hidden().is_none(), "a fresh state has no hidden");

    let mut state = post_final_state(&cfg, &weights);
    state
        .forward_prefill_with_hidden(&tokens)
        .expect("capturing prefill");
    assert_eq!(state.session.kv_cache.seq_len, tokens.len());
    assert_last_hidden_within(
        &state,
        &expected,
        PREFILL_VS_STEP_TOL,
        "after a capturing prefill",
    );

    // The plain prefill does not capture on a state without MTP weights: no value.
    let mut plain = post_final_state(&cfg, &weights);
    plain.forward_prefill(&tokens);
    assert_eq!(plain.session.kv_cache.seq_len, tokens.len());
    assert!(
        plain.last_hidden().is_none(),
        "a plain prefill leaves no provable hidden"
    );

    // Sequential capturing steps and a verify both end on the last token too.
    let mut verified = post_final_state(&cfg, &weights);
    verified
        .verify_tokens(&tokens, 0)
        .expect("sequential verify");
    assert_last_hidden_is(&verified, &expected, "after verify_tokens");
}

#[test]
fn last_hidden_is_none_whenever_the_buffer_is_not_the_live_cursors_token() {
    use crate::speculative::MtpTargetVerifier as _;

    let _gpu = gpu_test_lock();
    if !metal_device_or_skip(
        "last_hidden_is_none_whenever_the_buffer_is_not_the_live_cursors_token",
    ) {
        return;
    }
    let (cfg, weights) = post_final_fixture();
    let tokens = POST_FINAL_TOKENS;
    let n = tokens.len();
    let expected = independent_post_final_hidden(&cfg, &weights, &tokens);

    // A plain, non-capturing step after a capture: the cursor moved, the buffer did not.
    let mut state = post_final_state(&cfg, &weights);
    state.forward_prefill_with_hidden(&tokens).expect("prefill");
    assert_last_hidden_within(
        &state,
        &expected,
        PREFILL_VS_STEP_TOL,
        "plain step: positive control",
    );
    state.try_forward_step(9, n).expect("plain step");
    assert_eq!(state.session.kv_cache.seq_len, n + 1);
    assert!(state.last_hidden().is_none(), "after a plain step");

    // A rollback below the marker, then a replay of different tokens without capture
    // that brings the cursor back to the marker's value: the buffer is the OLD
    // history's hidden even though the cursor equals the marker again.
    let mut state = post_final_state(&cfg, &weights);
    state.verify_tokens(&tokens, 0).expect("verify");
    assert_last_hidden_is(&state, &expected, "rollback: positive control");
    state.rollback_cache_to(2).expect("rollback below marker");
    for (i, &token) in [10u32, 12, 16].iter().enumerate() {
        state.try_forward_step(token, 2 + i).expect("replay step");
    }
    assert_eq!(state.session.kv_cache.seq_len, n);
    assert_eq!(
        state.session.last_hidden_cursor, None,
        "the rollback below the marker must clear it"
    );
    assert!(
        state.last_hidden().is_none(),
        "after a rollback below the marker and a replay to the same cursor"
    );

    // reset_state, then a replay to the same cursor.
    let mut state = post_final_state(&cfg, &weights);
    state.forward_prefill_with_hidden(&tokens).expect("prefill");
    assert_last_hidden_within(
        &state,
        &expected,
        PREFILL_VS_STEP_TOL,
        "reset: positive control",
    );
    state.reset_state();
    assert!(state.last_hidden().is_none(), "right after reset_state");
    for (i, &token) in tokens.iter().enumerate() {
        state.try_forward_step(token, i).expect("replay step");
    }
    assert_eq!(state.session.kv_cache.seq_len, n);
    assert!(
        state.last_hidden().is_none(),
        "after reset_state and a replay to the same cursor"
    );

    // The cross-turn restore: the entry saved at the cursor `n` comes back at cursor
    // `n`, so only clearing at the restore keeps the stale buffer out.
    let mut state = post_final_state(&cfg, &weights);
    state.forward_prefill_with_hidden(&tokens).expect("prefill");
    assert_last_hidden_within(
        &state,
        &expected,
        PREFILL_VS_STEP_TOL,
        "cross-turn: positive control",
    );
    let metadata = crate::kv_cache::CrossTurnPrefixMetadata {
        model_fingerprint: 1,
        tokenizer_fingerprint: 2,
        adapter_id: crate::kv_cache::AdapterId::BASE,
        vocab_size: cfg.vocab_size,
        max_cache_len: state.session.kv_cache.max_cache_len,
        kv_f16: state.use_kv_f16,
        rope_theta_bits: cfg.rope_theta.to_bits(),
        partial_rotary_factor_bits: Some(cfg.partial_rotary_factor.to_bits()),
        layer_pattern_hash: 3,
        chat_template_version: 1,
    };
    let slot = crate::kv_cache::CrossTurnSlotId(1);
    state
        .save_cross_turn_prefix(slot, metadata.clone(), tokens.to_vec(), None)
        .expect("save the turn's prefix");
    let mut next_prompt = tokens.to_vec();
    next_prompt.push(9);
    let plan = state.plan_cross_turn_reuse(slot, &metadata, &next_prompt);
    assert_eq!(plan.mode, crate::kv_cache::PrefixReuseMode::ExactAppend);
    assert_eq!(plan.reusable_len, n);
    state
        .restore_cross_turn_prefix(slot, &plan)
        .expect("restore")
        .expect("an exact append returns the consumed entry");
    assert_eq!(state.session.kv_cache.seq_len, n);
    assert!(
        state.last_hidden().is_none(),
        "after the cross-turn restore"
    );
}

#[test]
fn last_hidden_survives_a_rollback_to_exactly_the_marker() {
    use crate::speculative::MtpTargetVerifier as _;

    let _gpu = gpu_test_lock();
    if !metal_device_or_skip("last_hidden_survives_a_rollback_to_exactly_the_marker") {
        return;
    }
    let (cfg, weights) = post_final_fixture();
    let tokens = POST_FINAL_TOKENS;
    let n = tokens.len();
    let expected = independent_post_final_hidden(&cfg, &weights, &tokens);

    // Rollback to the cursor itself (a full accept).
    let mut state = post_final_state(&cfg, &weights);
    state.verify_tokens(&tokens, 0).expect("verify");
    let before = assert_last_hidden_is(&state, &expected, "before the rollback");
    state.rollback_cache_to(n).expect("rollback to the marker");
    assert_eq!(state.last_hidden().as_deref(), Some(before.as_slice()));

    // Rollback to the marker from a later cursor: the first `n` tokens are intact.
    let mut state = post_final_state(&cfg, &weights);
    state.verify_tokens(&tokens, 0).expect("verify");
    let before = assert_last_hidden_is(&state, &expected, "before the later steps");
    state.try_forward_step(9, n).expect("later step");
    state.try_forward_step(10, n + 1).expect("later step");
    assert!(state.last_hidden().is_none(), "left behind by the steps");
    state.rollback_cache_to(n).expect("rollback to the marker");
    assert_eq!(state.session.kv_cache.seq_len, n);
    assert_eq!(state.last_hidden().as_deref(), Some(before.as_slice()));
}

/// The real Metal target, with a record of the rows `verify_tokens_with_hidden`
/// returned. Every method, `last_hidden` included, is the real implementation's.
struct MetalTargetRecordingRows<'a> {
    inner: &'a mut MetalQwen35State,
    rows: Option<Vec<Vec<f32>>>,
}

impl crate::speculative::MtpTargetVerifier for MetalTargetRecordingRows<'_> {
    fn cache_position(&self) -> usize {
        self.inner.cache_position()
    }
    fn rollback_cache_to(&mut self, seq_len: usize) -> Result<(), crate::error::InferenceError> {
        self.inner.rollback_cache_to(seq_len)
    }
    fn verify_tokens(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
        self.inner.verify_tokens(tokens, start_pos)
    }
    fn last_hidden(&self) -> Option<Vec<f32>> {
        self.inner.last_hidden()
    }
    #[allow(clippy::type_complexity)]
    fn verify_tokens_with_hidden(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>), crate::error::InferenceError> {
        let out = self.inner.verify_tokens_with_hidden(tokens, start_pos)?;
        self.rows = out.1.clone();
        Ok(out)
    }
    fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
        self.inner.snapshot_gdn_states()
    }
    fn restore_gdn_states(&mut self, snapshot: &crate::attention::gdn::GdnSnapshot) {
        self.inner.restore_gdn_states(snapshot)
    }
}

/// A CPU MTP head sized to `post_final_fixture`, with patterned weights.
struct CpuHeadFixture {
    cfg: crate::speculative::MtpConfig,
    weights: crate::speculative::MtpWeights,
    embed: Vec<f32>,
    lm_head: Vec<f32>,
}

fn cpu_head_fixture(hidden: usize, vocab: usize, draft_length: usize) -> CpuHeadFixture {
    use crate::speculative::{
        MtpAttentionWeights, MtpConfig, MtpLayerWeights, MtpMoeWeights, MtpWeights,
    };
    let cfg = MtpConfig {
        draft_length,
        num_hidden_layers: 1,
        hidden_size: hidden,
        vocab_size: vocab,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        num_experts: 1,
        num_experts_per_tok: 1,
        moe_intermediate_size: 2,
        shared_expert_intermediate_size: 2,
        use_dedicated_embeddings: false,
    };
    let fill = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|i| scale * ((i as f32 + 1.0) * 0.01).sin())
            .collect()
    };
    let ones = |n: usize| vec![1.0f32; n];
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let layer = MtpLayerWeights {
        input_layernorm: ones(hidden),
        post_attention_layernorm: ones(hidden),
        self_attn: MtpAttentionWeights {
            q_proj: fill(2 * q_dim * hidden, 0.1),
            k_proj: fill(kv_dim * hidden, 0.1),
            v_proj: fill(kv_dim * hidden, 0.1),
            o_proj: fill(hidden * q_dim, 0.1),
            q_norm: ones(cfg.head_dim),
            k_norm: ones(cfg.head_dim),
        },
        mlp: MtpMoeWeights {
            router_gate: fill(hidden, 0.1),
            experts_gate_up_proj: fill(2 * 2 * hidden, 0.05),
            experts_down_proj: fill(hidden * 2, 0.05),
            shared_gate_proj: fill(2 * hidden, 0.05),
            shared_up_proj: fill(2 * hidden, 0.05),
            shared_down_proj: fill(hidden * 2, 0.05),
            shared_expert_gate: fill(hidden, 0.1),
        },
    };
    let weights = MtpWeights {
        fc_weight: fill(hidden * 2 * hidden, 0.1),
        layers: vec![layer],
        norm_weight: ones(hidden),
        pre_fc_norm_embedding_weight: ones(hidden),
        pre_fc_norm_hidden_weight: ones(hidden),
    };
    let embed = (0..vocab * hidden)
        .map(|i| ((i as f32 + 1.0) * 0.01).sin())
        .collect();
    let lm_head = (0..vocab * hidden)
        .map(|i| ((i as f32 + 2.0) * 0.01).cos())
        .collect();
    CpuHeadFixture {
        cfg,
        weights,
        embed,
        lm_head,
    }
}

fn mtp_kv_row(
    v: &crate::speculative::MtpVerifier<'_>,
    row: usize,
) -> (Vec<half::f16>, Vec<half::f16>) {
    let kv_dim = v.config.num_key_value_heads * v.config.head_dim;
    (
        v.cache.k_buffer(0)[row * kv_dim..(row + 1) * kv_dim].to_vec(),
        v.cache.v_buffer(0)[row * kv_dim..(row + 1) * kv_dim].to_vec(),
    )
}

/// The K/V row a full `forward_one` step writes for (`token`, `hidden`) at `position`
/// on a fresh CPU head.
fn mtp_reference_row(
    head: &CpuHeadFixture,
    token: u32,
    position: usize,
    hidden: &[f32],
) -> (Vec<half::f16>, Vec<half::f16>) {
    let mut v = crate::speculative::MtpVerifier::new(
        head.cfg.clone(),
        &head.weights,
        &head.embed,
        &head.lm_head,
        32,
    )
    .expect("cpu head");
    v.forward_one(token, position, hidden)
        .expect("reference step");
    mtp_kv_row(&v, 0)
}

/// One generic round: previous token at 0, pending token at 1 on a fresh Metal state,
/// then `mtp_verify_draft_with_seed` with a CPU head. With `with_last` the pending
/// token goes through a capturing step, so the state supplies `last_hidden`; without
/// it the pending token goes through a plain step, which leaves the buffer stale, and
/// the state supplies none.
struct MetalRound<'h> {
    result: crate::speculative::MtpVerifyResult,
    verifier: crate::speculative::MtpVerifier<'h>,
    target_cursor: usize,
    hidden_predicting: Vec<f32>,
    last: Vec<f32>,
    rows: Option<Vec<Vec<f32>>>,
    last_hidden_after: Option<Vec<f32>>,
}

const ROUND_PREVIOUS_TOKEN: u32 = 6;
const ROUND_PENDING_TOKEN: u32 = 17;
const ROUND_TEMPERATURE: f32 = 50.0;

fn metal_round<'h>(
    head: &'h CpuHeadFixture,
    cfg: &Qwen35Config,
    weights: &ModelWeights,
    seed: u64,
    with_last: bool,
) -> MetalRound<'h> {
    use crate::speculative::{MtpTargetVerifier as _, MtpVerifier, mtp_verify_draft_with_seed};

    let mut state = post_final_state(cfg, weights);
    let (_, pre_previous) = state
        .forward_step_with_hidden(ROUND_PREVIOUS_TOKEN, 0)
        .expect("previous token");
    let hidden_predicting =
        reference_final_norm(&pre_previous, &weights.final_norm, cfg.rms_norm_eps);
    let (initial_logits, last) = if with_last {
        let (logits, pre_pending) = state
            .forward_step_with_hidden(ROUND_PENDING_TOKEN, 1)
            .expect("pending token");
        let last = reference_final_norm(&pre_pending, &weights.final_norm, cfg.rms_norm_eps);
        let supplied = state
            .last_hidden()
            .expect("a capturing step leaves a fresh last_hidden");
        let diff = max_abs_diff(&supplied, &last);
        assert!(
            diff <= POST_FINAL_TOL,
            "last_hidden differs from the independent post-final hidden by {diff}"
        );
        (logits, last)
    } else {
        let logits = state
            .try_forward_step(ROUND_PENDING_TOKEN, 1)
            .expect("pending token");
        assert!(
            state.last_hidden().is_none(),
            "a plain step must leave no last_hidden"
        );
        (logits, Vec::new())
    };
    let mut verifier = MtpVerifier::new(
        head.cfg.clone(),
        &head.weights,
        &head.embed,
        &head.lm_head,
        32,
    )
    .expect("cpu head");
    let (result, target_cursor, rows, last_hidden_after) = if with_last {
        let mut target = MetalTargetRecordingRows {
            inner: &mut state,
            rows: None,
        };
        let result = mtp_verify_draft_with_seed(
            &mut verifier,
            ROUND_PENDING_TOKEN,
            1,
            &hidden_predicting,
            &initial_logits,
            None,
            &mut target,
            ROUND_TEMPERATURE,
            Some(seed),
        )
        .expect("generic round with last_hidden");
        (
            result,
            target.inner.cache_position(),
            target.rows.take(),
            target.inner.last_hidden(),
        )
    } else {
        let result = mtp_verify_draft_with_seed(
            &mut verifier,
            ROUND_PENDING_TOKEN,
            1,
            &hidden_predicting,
            &initial_logits,
            None,
            &mut state,
            ROUND_TEMPERATURE,
            Some(seed),
        )
        .expect("generic round on the plain Metal target");
        (result, state.cache_position(), None, state.last_hidden())
    };
    MetalRound {
        result,
        verifier,
        target_cursor,
        hidden_predicting,
        last,
        rows,
        last_hidden_after,
    }
}

#[test]
fn mtp_verify_draft_through_the_metal_target_keeps_the_mtp_cache_aligned() {
    let _gpu = gpu_test_lock();
    if !metal_device_or_skip(
        "mtp_verify_draft_through_the_metal_target_keeps_the_mtp_cache_aligned",
    ) {
        return;
    }
    let (cfg, weights) = post_final_fixture();
    const DRAFT_LENGTH: usize = 3;
    let head = cpu_head_fixture(cfg.hidden_size, cfg.vocab_size, DRAFT_LENGTH);

    // The draft is fixed, the rejection draw depends on the seed: scan for one seed that
    // gives a full accept and one that gives a partial accept with at least one token.
    let mut full_seed = None;
    let mut partial_seed = None;
    for seed in 0u64..256 {
        let a = metal_round(&head, &cfg, &weights, seed, false)
            .result
            .accepted_count;
        if a == DRAFT_LENGTH && full_seed.is_none() {
            full_seed = Some(seed);
        }
        if (1..DRAFT_LENGTH).contains(&a) && partial_seed.is_none() {
            partial_seed = Some(seed);
        }
        if full_seed.is_some() && partial_seed.is_some() {
            break;
        }
    }
    let full_seed = full_seed.expect("a seed with a full accept within 256 seeds");
    let partial_seed = partial_seed.expect("a seed with a partial accept within 256 seeds");

    for (label, seed) in [("full", full_seed), ("partial", partial_seed)] {
        // Without a `last_hidden` the Metal target behaves as before: the MTP cursor
        // ends at the drafted rows of the accepted prefix.
        let plain = metal_round(&head, &cfg, &weights, seed, false);
        let a = plain.result.accepted_count;
        assert_eq!(
            plain.verifier.cache.seq_len(),
            a,
            "{label}: fallback cursor"
        );
        assert_eq!(plain.target_cursor, 2 + a, "{label}: target cursor");

        // With `last_hidden` supplied, the Metal rows drive the rewrite.
        let round = metal_round(&head, &cfg, &weights, seed, true);
        assert_eq!(
            round.result, plain.result,
            "{label}: outputs do not depend on rows"
        );
        assert_eq!(round.target_cursor, 2 + a, "{label}: target cursor");
        assert_eq!(
            round.verifier.cache.seq_len(),
            1 + a,
            "{label}: cursor is the pending token's row plus one row per accepted token"
        );
        let rows = round
            .rows
            .as_ref()
            .expect("the Metal target returned its rows through the generic loop");
        assert_eq!(rows.len(), DRAFT_LENGTH, "{label}");
        assert_eq!(round.result.draft_tokens.len(), DRAFT_LENGTH, "{label}");
        // The recorded rows are the target's post-final hiddens for the drafted tokens,
        // checked against an independent sequential run, so the pairing checks below
        // cannot be satisfied by rows that are wrong in the same way as the expectation.
        let mut independent = post_final_state(&cfg, &weights);
        for (pos, token) in [(0, ROUND_PREVIOUS_TOKEN), (1, ROUND_PENDING_TOKEN)] {
            independent
                .forward_step_with_hidden(token, pos)
                .expect("independent prefix");
        }
        for (i, &token) in round.result.draft_tokens.iter().enumerate() {
            let (_, pre) = independent
                .forward_step_with_hidden(token, 2 + i)
                .expect("independent draft step");
            let expected = reference_final_norm(&pre, &weights.final_norm, cfg.rms_norm_eps);
            let diff = max_abs_diff(&rows[i], &expected);
            assert!(
                diff <= POST_FINAL_TOL,
                "{label}: row {i} differs from the independent post-final hidden by {diff}"
            );
        }
        // What the generic loop sees afterwards: a full accept leaves the last drafted
        // token as the most recent one in the target, so its post-final hidden is
        // still provably current; a partial accept rewound below the marker.
        match (label, round.last_hidden_after.as_deref()) {
            ("full", Some(after)) => {
                let diff = max_abs_diff(after, &rows[DRAFT_LENGTH - 1]);
                assert!(
                    diff <= POST_FINAL_TOL,
                    "full: last_hidden after the round differs from the last row by {diff}"
                );
            }
            ("partial", None) => {}
            (label, after) => panic!(
                "{label}: unexpected last_hidden after the round (is_some={})",
                after.is_some()
            ),
        }
        assert_eq!(
            mtp_kv_row(&round.verifier, 0),
            mtp_reference_row(&head, ROUND_PENDING_TOKEN, 1, &round.hidden_predicting),
            "{label}: the pending token's row keeps the caller's hidden state"
        );
        for k in 1..=a {
            let predictor = if k == 1 { &round.last } else { &rows[k - 2] };
            let token = round.result.draft_tokens[k - 1];
            let expected = mtp_reference_row(&head, token, 1 + k, predictor);
            assert_eq!(
                mtp_kv_row(&round.verifier, k),
                expected,
                "{label}: row for accepted draft token {k}"
            );
            let wrong = mtp_reference_row(&head, token, 1 + k, &round.hidden_predicting);
            assert_ne!(
                expected, wrong,
                "{label}: the fixture must make a wrong pairing observable"
            );
        }
    }
}

// ---------------------------------------------------------------------
// The generic MTP draft/verify loop on a real checkpoint.
//
// Target = `MetalQwen35State` on the checkpoint, drafter = the CPU `MtpVerifier` built
// from the same checkpoint's MTP tensors, loop = `mtp_verify_draft_with_seed`. The loop
// is driven for 64 new tokens per prompt, once with the rows the target supplies and
// once with a target that supplies none, and both token streams are compared with plain
// greedy decoding of the same prompt.
//
//   LATTICE_MTP_REAL_LOOP_ENFORCE=1 LATTICE_MODEL_DIR=~/.lattice/models/qwen3.5-0.8b \
//   cargo test --release -p lattice-inference --features "f16,metal-gpu" \
//       mtp_generic_loop_on_real_checkpoint -- --nocapture --test-threads=1
// ---------------------------------------------------------------------

const REAL_LOOP_DRAFT_LENGTH: usize = 3;
const REAL_LOOP_NEW_TOKENS: usize = 64;
const REAL_LOOP_MAX_CACHE: usize = 1024;
/// The generic loop has no greedy switch: it always accepts with probability
/// `min(1, p/q)` and corrects from the residual. At a temperature this small `p` and
/// `q` are one-hot on the argmax of the target and the draft, so a draft token is
/// accepted exactly when it is the target's argmax and the correction is that argmax:
/// the loop then has to reproduce plain greedy decoding token for token.
const REAL_LOOP_TEMPERATURE: f32 = 1e-6;

/// Fixed prompts of different lengths, tokenized by the checkpoint's own tokenizer.
const REAL_LOOP_PROMPTS: [&str; 8] = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Photosynthesis is the process by which green plants convert light energy into \
             chemical energy.",
    "Once upon a time, in a small village at the edge of a great forest, there lived a \
             woodcutter and his three daughters.",
    "1, 2, 3, 4, 5, 6, 7, 8,",
    "In 1969, Apollo 11 landed on the Moon. The mission was commanded by Neil \
             Armstrong, who was accompanied by Buzz Aldrin and Michael Collins. Armstrong was \
             the first person to walk on the lunar surface, and he described the moment as",
    "Q: What is the difference between a process and a thread in an operating \
             system?\nA:",
    "The transformer architecture replaced recurrent networks in most sequence \
             modelling tasks because attention lets every position look at every other \
             position directly. Its main cost is that the key-value cache grows linearly with \
             the length of the context, so",
];

fn real_loop_enforced() -> bool {
    matches!(
        std::env::var("LATTICE_MTP_REAL_LOOP_ENFORCE").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// The checkpoint's MTP head through the public loader and config constructor. The 0.8B
/// checkpoint carries a dense MLP in its MTP layer, which the loader stores as one
/// expert with top-1 routing and a zero-width shared expert.
fn real_loop_head(
    dir: &std::path::Path,
    cfg: &Qwen35Config,
) -> (
    crate::speculative::MtpConfig,
    crate::speculative::MtpWeights,
) {
    use crate::speculative::{MtpConfig, MtpWeights};
    use crate::weights::{SafetensorsFile, ShardedSafetensors};

    let head_cfg = MtpConfig::from_qwen35(cfg, REAL_LOOP_DRAFT_LENGTH)
        .expect("build the MTP config from the model config");
    let single = dir.join("model.safetensors");
    let weights = if single.exists() {
        let mut source = SafetensorsFile::open(&single).expect("open model.safetensors");
        MtpWeights::load_from_source(&mut source, &head_cfg)
    } else {
        let mut source = ShardedSafetensors::open_index(&dir.join("model.safetensors.index.json"))
            .expect("open sharded index");
        MtpWeights::load_from_source(&mut source, &head_cfg)
    }
    .expect("load the MTP head");
    let mlp = &weights.layers[0].mlp;
    assert!(
        mlp.experts_gate_up_proj.iter().any(|&v| v != 0.0)
            && mlp.experts_down_proj.iter().any(|&v| v != 0.0),
        "the MLP tensors are all zero, so the head would not read the checkpoint"
    );
    (head_cfg, weights)
}

struct RealLoopHead<'a> {
    cfg: crate::speculative::MtpConfig,
    weights: crate::speculative::MtpWeights,
    /// Tied embedding and output projection.
    embed: &'a [f32],
}

/// The target with `last_hidden` and `verify_tokens_with_hidden` left at the trait
/// defaults, as a target that cannot supply hidden states.
struct NoRowsTarget<'a>(&'a mut MetalQwen35State);

impl crate::speculative::MtpTargetVerifier for NoRowsTarget<'_> {
    fn cache_position(&self) -> usize {
        crate::speculative::MtpTargetVerifier::cache_position(&*self.0)
    }
    fn rollback_cache_to(&mut self, seq_len: usize) -> Result<(), crate::error::InferenceError> {
        crate::speculative::MtpTargetVerifier::rollback_cache_to(&mut *self.0, seq_len)
    }
    fn verify_tokens(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
        crate::speculative::MtpTargetVerifier::verify_tokens(&mut *self.0, tokens, start_pos)
    }
    fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
        crate::speculative::MtpTargetVerifier::snapshot_gdn_states(&*self.0)
    }
    fn restore_gdn_states(&mut self, snapshot: &crate::attention::gdn::GdnSnapshot) {
        crate::speculative::MtpTargetVerifier::restore_gdn_states(&mut *self.0, snapshot)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RealLoopPath {
    /// The generic loop on the real target as shipped.
    Rows,
    /// The same loop on `NoRowsTarget`.
    NoRows,
}

impl RealLoopPath {
    fn label(self) -> &'static str {
        match self {
            Self::Rows => "rows",
            Self::NoRows => "fallback",
        }
    }
}

#[derive(Default, Clone)]
struct RealLoopStats {
    verify_steps: usize,
    drafted: usize,
    accepted: usize,
    accepted_hist: [usize; REAL_LOOP_DRAFT_LENGTH + 1],
    /// Rows of the draft cache checked against the independent construction.
    rows_checked: usize,
    /// The largest `d_right / d_wrong` over the rows checked, see
    /// `real_loop_check_rows`.
    worst_row_ratio: f32,
}

impl RealLoopStats {
    fn add(&mut self, other: &Self) {
        self.verify_steps += other.verify_steps;
        self.drafted += other.drafted;
        self.accepted += other.accepted;
        self.rows_checked += other.rows_checked;
        self.worst_row_ratio = self.worst_row_ratio.max(other.worst_row_ratio);
        for (a, b) in self.accepted_hist.iter_mut().zip(other.accepted_hist) {
            *a += b;
        }
    }
}

/// The post-final-norm hidden state of every position of `tokens`, from sequential
/// capturing steps on `state` and an f64 final norm, so the draft head's input does
/// not depend on the path under test. Leaves `state` reset.
fn real_loop_hidden_table(
    state: &mut MetalQwen35State,
    tokens: &[u32],
    final_norm: &[f32],
    eps: f32,
) -> Vec<Vec<f32>> {
    state.reset_state();
    let table = tokens
        .iter()
        .enumerate()
        .map(|(position, &token)| {
            let (_, pre) = state
                .forward_step_with_hidden(token, position)
                .expect("hidden table step");
            reference_final_norm(&pre, final_norm, eps)
        })
        .collect();
    state.reset_state();
    table
}

fn real_loop_row(v: &crate::speculative::MtpVerifier<'_>, row: usize) -> Vec<f32> {
    let (k, v) = mtp_kv_row(v, row);
    k.iter().chain(v.iter()).map(|x| x.to_f32()).collect()
}

/// The K/V row the draft head builds for `token` at `position` from `hidden`, on an
/// otherwise empty cache.
fn real_loop_reference_row(
    scratch: &mut crate::speculative::MtpVerifier<'_>,
    token: u32,
    position: usize,
    hidden: &[f32],
) -> Vec<f32> {
    scratch
        .rollback_cache_to(0)
        .expect("reset the scratch head");
    scratch
        .append_kv_row(token, position, hidden)
        .expect("reference row");
    real_loop_row(scratch, 0)
}

/// After a round on a target that supplies rows, every row the round left in the draft
/// cache (the pending token's and one per accepted token) must be the row built from
/// the hidden state that predicted its token. The pending token's row takes the
/// caller's hidden state, so it matches bit for bit. The others take the target's
/// batched hidden states, which differ from the sequential ones in the table by float
/// noise, so each is compared with the row built from the right hidden state and with
/// the row built from the previous token's hidden state (the pairing a shifted set of
/// rows would give): it must be at least four times closer to the right one.
#[allow(clippy::too_many_arguments)]
fn real_loop_check_rows(
    verifier: &crate::speculative::MtpVerifier<'_>,
    scratch: &mut crate::speculative::MtpVerifier<'_>,
    hidden_table: &[Vec<f32>],
    head_before: usize,
    position: usize,
    tokens: &[u32],
    stats: &mut RealLoopStats,
) {
    for (k, &token) in tokens.iter().enumerate() {
        let at = position + k;
        let (Some(right), Some(wrong)) = (hidden_table.get(at - 1), hidden_table.get(at - 2))
        else {
            continue;
        };
        let actual = real_loop_row(verifier, head_before + k);
        let right = real_loop_reference_row(scratch, token, at, right);
        let wrong = real_loop_reference_row(scratch, token, at, wrong);
        let d_wrong = max_abs_diff(&actual, &wrong);
        assert!(
            d_wrong > 0.0,
            "position {at}: the two pairings give the same row, so the check cannot tell them apart"
        );
        if k == 0 {
            assert_eq!(
                actual, right,
                "position {position}: the pending token's row must be bit-equal to the \
                         row built from the caller's hidden state"
            );
        }
        let d_right = max_abs_diff(&actual, &right);
        assert!(
            d_right * 4.0 < d_wrong,
            "position {at} (round row {k}): the draft cache row is not the row for the \
                     hidden state that predicted its token: distance to it {d_right}, to the \
                     previous token's pairing {d_wrong}"
        );
        stats.rows_checked += 1;
        stats.worst_row_ratio = stats.worst_row_ratio.max(d_right / d_wrong);
    }
}

/// Greedy decode of `REAL_LOOP_NEW_TOKENS` tokens through `mtp_verify_draft_with_seed`.
///
/// Prefill, then per round: the pending token goes through a capturing step, the loop
/// drafts, verifies and rolls back, and its accepted tokens plus its correction or
/// bonus token are emitted; the correction becomes the next pending token. The draft
/// head's cache is first filled with the prompt, one row per position from 1, each
/// pairing the hidden state that predicted the token with the token's embedding.
/// `supplies_rows` says whether `Rows` is expected to find hidden states on this
/// target; it decides the cache cursor the loop must leave behind.
fn real_loop_decode(
    state: &mut MetalQwen35State,
    head: &RealLoopHead<'_>,
    prompt: &[u32],
    hidden_table: &[Vec<f32>],
    path: RealLoopPath,
    supplies_rows: bool,
) -> (Vec<u32>, RealLoopStats) {
    use crate::speculative::{MtpTargetVerifier as _, MtpVerifier, mtp_verify_draft_with_seed};

    state.reset_state();
    let mut verifier = MtpVerifier::new(
        head.cfg.clone(),
        &head.weights,
        head.embed,
        head.embed,
        REAL_LOOP_MAX_CACHE,
    )
    .expect("cpu draft head");
    for position in 1..prompt.len() {
        verifier
            .append_kv_row(prompt[position], position, &hidden_table[position - 1])
            .expect("prompt row for the draft head");
    }
    let rows_expected = path == RealLoopPath::Rows && supplies_rows;
    let mut scratch = MtpVerifier::new(
        head.cfg.clone(),
        &head.weights,
        head.embed,
        head.embed,
        REAL_LOOP_MAX_CACHE,
    )
    .expect("scratch draft head");

    let prefill_logits = state.forward_prefill(prompt);
    let mut pending = crate::sampling::argmax_f32_first_wins(&prefill_logits);
    let mut ids = vec![pending];
    let mut stats = RealLoopStats::default();
    while ids.len() < REAL_LOOP_NEW_TOKENS {
        let position = state.cache_position();
        let Some(predicting) = hidden_table.get(position - 1) else {
            break;
        };
        let (logits, _) = state
            .forward_step_with_hidden(pending, position)
            .expect("capturing step for the pending token");
        assert_eq!(
            state.last_hidden().is_some(),
            supplies_rows,
            "freshness of last_hidden after a capturing step at position {position}"
        );
        let target_before = state.cache_position();
        let head_before = verifier.cache.seq_len();
        let seed = 0x1396_0000 + stats.verify_steps as u64;
        let result = match path {
            RealLoopPath::Rows => mtp_verify_draft_with_seed(
                &mut verifier,
                pending,
                position,
                predicting,
                &logits,
                None,
                &mut *state,
                REAL_LOOP_TEMPERATURE,
                Some(seed),
            ),
            RealLoopPath::NoRows => mtp_verify_draft_with_seed(
                &mut verifier,
                pending,
                position,
                predicting,
                &logits,
                None,
                &mut NoRowsTarget(&mut *state),
                REAL_LOOP_TEMPERATURE,
                Some(seed),
            ),
        }
        .expect("generic round on the real target");
        let accepted = result.accepted_count;
        assert_eq!(
            state.cache_position(),
            target_before + accepted,
            "target cursor after the round"
        );
        // The pending token's row plus one row per accepted token with the target's
        // rows; the rows of the accepted prefix of the draft without them.
        let expected_head = head_before + accepted + usize::from(rows_expected);
        assert_eq!(
            verifier.cache.seq_len(),
            expected_head,
            "draft cache cursor after a round that accepted {accepted} ({})",
            path.label()
        );
        if rows_expected {
            assert_eq!(
                state.cache_position() - verifier.cache.seq_len(),
                1,
                "with the target's rows the draft cache ends at the target's position"
            );
        }
        if rows_expected {
            let mut round_tokens = vec![pending];
            round_tokens.extend_from_slice(&result.accepted_tokens);
            real_loop_check_rows(
                &verifier,
                &mut scratch,
                hidden_table,
                head_before,
                position,
                &round_tokens,
                &mut stats,
            );
        }
        stats.verify_steps += 1;
        stats.drafted += result.draft_tokens.len();
        stats.accepted += accepted;
        stats.accepted_hist[accepted] += 1;
        ids.extend_from_slice(&result.accepted_tokens);
        pending = result
            .fallback_token
            .expect("a correction or bonus token without an eos");
        ids.push(pending);
    }
    ids.truncate(REAL_LOOP_NEW_TOKENS);
    (ids, stats)
}

fn real_loop_truncate_at_stop(mut ids: Vec<u32>, stop: u32) -> Vec<u32> {
    if let Some(i) = ids.iter().position(|&t| t == stop) {
        ids.truncate(i);
    }
    ids
}

fn real_loop_gen_cfg() -> crate::generation::GenerateConfig {
    crate::generation::GenerateConfig {
        min_p: 0.0,
        max_new_tokens: REAL_LOOP_NEW_TOKENS,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    }
}

/// Compare the generic loop with plain greedy decoding for every prompt on one
/// target family. `make_reference` builds a fresh state, `spec` is the reused loop
/// state (reset per run), `table_state` is the original-basis state that supplies the
/// draft head's hidden inputs.
#[allow(clippy::too_many_arguments)]
fn real_loop_compare(
    label: &str,
    tokenizer: &crate::tokenizer::bpe::BpeTokenizer,
    eos: u32,
    head: &RealLoopHead<'_>,
    final_norm: &[f32],
    eps: f32,
    make_reference: &dyn Fn() -> MetalQwen35State,
    spec: &mut MetalQwen35State,
    table_state: &mut MetalQwen35State,
    supplies_rows: bool,
) -> Vec<String> {
    let gen_cfg = real_loop_gen_cfg();
    let mut totals = [RealLoopStats::default(), RealLoopStats::default()];
    let mut mismatches = Vec::new();
    for (index, prompt) in REAL_LOOP_PROMPTS.iter().enumerate() {
        let input = tokenizer.tokenize(prompt);
        let prompt_ids = input.input_ids[..input.real_length].to_vec();

        let mut reference_state = make_reference();
        let reference = reference_state
            .generate(prompt, tokenizer, &gen_cfg)
            .expect("plain greedy reference")
            .token_ids;
        drop(reference_state);
        assert!(!reference.is_empty(), "prompt {index}: empty reference");

        let mut sequence = prompt_ids.clone();
        sequence.extend_from_slice(&reference);
        let table = real_loop_hidden_table(table_state, &sequence, final_norm, eps);

        for (slot, path) in [RealLoopPath::Rows, RealLoopPath::NoRows]
            .into_iter()
            .enumerate()
        {
            let (ids, stats) =
                real_loop_decode(spec, head, &prompt_ids, &table, path, supplies_rows);
            println!(
                "[real-loop] {label} path={} prompt={index} prompt_len={} \
                         verify_steps={} drafted={} accepted={} accepted_per_step={:?}",
                path.label(),
                prompt_ids.len(),
                stats.verify_steps,
                stats.drafted,
                stats.accepted,
                stats.accepted_hist,
            );
            totals[slot].add(&stats);
            let ids = real_loop_truncate_at_stop(ids, eos);
            if ids != reference {
                let first = ids
                    .iter()
                    .zip(&reference)
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| ids.len().min(reference.len()));
                mismatches.push(format!(
                    "{label} prompt {index} path {}: ids differ from plain greedy at \
                             index {first} (loop {} ids, reference {} ids)\n  loop:      \
                             {ids:?}\n  reference: {reference:?}",
                    path.label(),
                    ids.len(),
                    reference.len()
                ));
            }
        }
    }
    for (slot, path) in [RealLoopPath::Rows, RealLoopPath::NoRows]
        .into_iter()
        .enumerate()
    {
        let t = &totals[slot];
        println!(
            "[real-loop] {label} path={} TOTAL verify_steps={} drafted={} accepted={} \
                     accepted_per_step={:?} draft_rows_checked={} worst_row_distance_ratio={:.4}",
            path.label(),
            t.verify_steps,
            t.drafted,
            t.accepted,
            t.accepted_hist,
            t.rows_checked,
            t.worst_row_ratio,
        );
        assert!(
            t.drafted > 0 && t.accepted > 0,
            "{label} path {}: no draft token was accepted, so the cache rewrite was \
                     never exercised",
            path.label()
        );
        if path == RealLoopPath::Rows && supplies_rows {
            assert!(
                t.rows_checked > t.verify_steps,
                "{label}: too few draft cache rows were checked ({}) to cover accepted \
                         tokens",
                t.rows_checked
            );
        }
    }
    mismatches
}

#[test]
fn mtp_generic_loop_on_real_checkpoint_matches_plain_greedy() {
    let _gpu = gpu_test_lock();
    let enforce = real_loop_enforced();
    if Device::system_default().is_none() {
        println!("SKIP mtp generic loop on a real checkpoint: no Metal device");
        assert!(
            !enforce,
            "LATTICE_MTP_REAL_LOOP_ENFORCE=1 but no Metal device present"
        );
        return;
    }
    let Some(dir) = hidden_parity_model_dir()
        .filter(|d| qwen35_checkpoint_weights_present(d) && d.join("tokenizer.json").is_file())
    else {
        println!(
            "SKIP mtp generic loop on a real checkpoint: no Qwen3.5 checkpoint with a \
                     tokenizer (set LATTICE_MODEL_DIR, or LATTICE_MTP_REAL_LOOP_ENFORCE=1 to \
                     make this a failure)"
        );
        assert!(
            !enforce,
            "LATTICE_MTP_REAL_LOOP_ENFORCE=1 but no checkpoint found: set \
                     LATTICE_MODEL_DIR to a Qwen3.5 safetensors checkout"
        );
        return;
    };

    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&dir).expect("load the checkpoint");
    let cfg = model.config().clone();
    assert!(
        cfg.mtp_num_hidden_layers == 1 && cfg.tie_word_embeddings,
        "the draft head here reads a tied head and one MTP layer"
    );
    let weights = model.weights();
    let max_gamma = weights
        .final_norm
        .iter()
        .fold(0.0f32, |m, g| m.max(g.abs()));
    assert!(
        max_gamma > 1e-3,
        "final_norm is an identity on this checkpoint (max |gamma| = {max_gamma}), so a \
                 pre-norm hidden state could not be told from a post-norm one"
    );
    let (head_cfg, head_weights) = real_loop_head(&dir, &cfg);
    let head = RealLoopHead {
        cfg: head_cfg,
        weights: head_weights,
        embed: weights.logits_weight(),
    };
    let tokenizer = model.tokenizer();
    let new_state = || {
        MetalQwen35State::new(weights, &cfg, REAL_LOOP_MAX_CACHE)
            .expect("build the Metal state from the checkpoint")
    };
    let mut spec = with_self_spec_env(new_state);
    let mut table_state = new_state();
    assert!(!spec.checkpoint_is_rotated());
    assert!(
        !spec.has_mtp(),
        "a state built from safetensors carries no GPU head"
    );
    println!(
        "[real-loop] checkpoint={} rotated=false draft_length={REAL_LOOP_DRAFT_LENGTH} \
                 new_tokens={REAL_LOOP_NEW_TOKENS} temperature={REAL_LOOP_TEMPERATURE:e} \
                 prompts=text, tokenized by the checkpoint tokenizer; final_norm max|gamma|=\
                 {max_gamma:.4}",
        dir.display()
    );
    let mismatches = real_loop_compare(
        "unrotated",
        tokenizer,
        cfg.eos_token_id,
        &head,
        &weights.final_norm,
        cfg.rms_norm_eps,
        &new_state,
        &mut spec,
        &mut table_state,
        true,
    );
    assert!(
        mismatches.is_empty(),
        "the generic loop differs from plain greedy decoding:\n{}",
        mismatches.join("\n")
    );

    // The rotated checkpoint, when one with MTP weights is present beside it.
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let models = std::path::PathBuf::from(&home).join(".lattice/models");
    let rotated_dir = models.join("qwen3.5-0.8b-q4-quarot");
    let mtp_present = std::fs::read_dir(&rotated_dir)
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.file_name().to_string_lossy().starts_with("mtp_"))
        })
        .unwrap_or(false);
    if !(rotated_dir.join("config.json").is_file() && mtp_present) {
        let listing: Vec<String> = std::fs::read_dir(&models)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        println!(
            "[real-loop] no rotated checkpoint with MTP weights under {}: {listing:?}",
            models.display()
        );
        return;
    }
    let rotated_cfg = Qwen35Config::from_config_json(&rotated_dir.join("config.json"))
        .expect("parse the rotated config");
    let tokenizer_path = dir.join("tokenizer.json");
    let new_rotated = || {
        MetalQwen35State::from_q4_dir(
            &rotated_dir,
            &tokenizer_path,
            &rotated_cfg,
            REAL_LOOP_MAX_CACHE,
        )
        .expect("build the Metal state from the rotated checkpoint")
    };
    let mut rotated = new_rotated();
    assert!(rotated.checkpoint_is_rotated());
    assert!(rotated.has_mtp() && rotated.mtp_hidden_tap().is_some());
    println!(
        "[real-loop] checkpoint={} rotated=true (draft head and its hidden inputs are \
                 the unrotated checkpoint's; the rotated target cannot supply rows)",
        rotated_dir.display()
    );
    let mismatches = real_loop_compare(
        "rotated",
        tokenizer,
        rotated_cfg.eos_token_id,
        &head,
        &weights.final_norm,
        cfg.rms_norm_eps,
        &new_rotated,
        &mut rotated,
        &mut table_state,
        false,
    );

    // Directly: a rotated target supplies neither the hidden state of the last token
    // nor rows, whatever it has just processed.
    {
        use crate::speculative::MtpTargetVerifier as _;
        let input = tokenizer.tokenize(REAL_LOOP_PROMPTS[0]);
        let prompt = input.input_ids[..input.real_length].to_vec();
        rotated.reset_state();
        rotated.forward_prefill(&prompt);
        assert!(rotated.last_hidden().is_none());
        let tokens = [prompt[0], prompt[1]];
        let (logits, rows) = rotated
            .verify_tokens_with_hidden(&tokens, prompt.len())
            .expect("verify on the rotated target");
        assert_eq!(logits.len(), tokens.len());
        assert!(rows.is_none(), "a rotated target must return no rows");
        assert!(rotated.last_hidden().is_none());
    }
    assert!(
        mismatches.is_empty(),
        "the generic loop differs from plain greedy decoding on the rotated \
                 checkpoint:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn forward_prefill_layer_traces_rejects_token_sequence_past_cache_capacity() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (cfg, weights) = tiny_metal_qwen35_fixture();
    // `forward_prefill_layer_traces_last_token` resets `kv_cache.seq_len` to 0
    // and then drives `position` from `0..token_ids.len()` — reachable from the
    // public `score_layer_importance` via caller-controlled `calibration_prompts`.
    // A 2-token sequence against a 1-slot cache must be rejected before any pass
    // dispatches.
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 1).expect("tiny MetalQwen35State fixture constructs");

    let result = state.forward_prefill_layer_traces_last_token(&[1, 2]);
    match result {
        Ok(_) => panic!("token sequence longer than max_cache_len must be rejected"),
        Err(crate::error::InferenceError::InvalidInput(_)) => {}
        Err(other) => panic!("expected InvalidInput, got {other}"),
    }
    assert_eq!(state.session.kv_cache.seq_len, 0);
}

#[test]
fn test_metal_qwen35_engine_session_isolation() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();

    // Build engine, then create two independent sessions from it.
    let engine = MetalQwen35Engine::new(&weights, &cfg)
        .expect("MetalQwen35Engine constructs from tiny fixture");
    let session_a = engine.new_session(16).expect("session_a constructs");
    let session_b = engine.new_session(16).expect("session_b constructs");

    // Both sessions start at position 0.
    assert_eq!(session_a.position(), 0, "session_a starts at position 0");
    assert_eq!(session_b.position(), 0, "session_b starts at position 0");
    assert_eq!(
        session_a.kv_cache.seq_len, 0,
        "session_a kv_cache starts empty"
    );
    assert_eq!(
        session_b.kv_cache.seq_len, 0,
        "session_b kv_cache starts empty"
    );

    // Wrap engine + session_a into a full state and run forward steps.
    let mut state_a = MetalQwen35State {
        engine,
        session: session_a,
        lora: None,
        use_gdn_chunked: true,
        use_kv_f16: false,
        cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
        path_proof_enabled: false,
        path_proof: PathProofCounters::default(),
    };
    let _logits = state_a.forward_step(42, 0);
    let _logits = state_a.forward_step(7, 1);

    // session_a (inside state_a) has advanced.
    assert_eq!(
        state_a.session.kv_cache.seq_len, 2,
        "state_a advanced 2 steps"
    );

    // Create a fresh session from the same engine — must start at 0.
    let fresh_session = state_a
        .engine
        .new_session(16)
        .expect("fresh session constructs");
    assert_eq!(
        fresh_session.position(),
        0,
        "fresh session position independent of state_a"
    );
    assert_eq!(
        fresh_session.kv_cache.seq_len, 0,
        "fresh session kv_cache independent"
    );

    // session_b was never touched; its state should match a fresh session.
    assert_eq!(
        session_b.position(),
        0,
        "session_b untouched by state_a steps"
    );
    assert_eq!(
        session_b.kv_cache.seq_len, 0,
        "session_b kv_cache untouched"
    );
}
