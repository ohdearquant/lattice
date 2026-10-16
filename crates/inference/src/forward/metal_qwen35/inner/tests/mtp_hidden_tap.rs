//! The MTP head's hidden-state tap and the post-final-norm rows it reads.

use super::*;
// ---- MTP hidden-state tap ----------------------------------------------
//
// The final-norm scale and `pre_fc_norm_hidden` are both non-uniform, with
// `1 + gamma` changing sign across the final-norm scale, so the head's
// hidden-state input differs by far more than float noise depending on
// whether the target's final norm was applied (RMSNorm is scale-invariant,
// so a uniform scale could not tell the two apart).

fn mtp_tap_final_norm_gamma(hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| 1.5 * ((i as f32) * 0.37).sin())
        .collect()
}

fn mtp_tap_pre_fc_hidden_gamma(hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| -0.4 + 0.3 * ((i as f32) * 0.07).cos())
        .collect()
}

fn mtp_tap_target_hidden(hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25 + 0.05)
        .collect()
}

/// Shifted RMSNorm in f64, independent of the production helper.
fn shifted_rmsnorm_reference(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
    let inv_rms = 1.0 / (mean_sq + f64::from(eps)).sqrt();
    x.iter()
        .zip(gamma)
        .map(|(&v, &g)| (f64::from(v) * inv_rms * (1.0 + f64::from(g))) as f32)
        .collect()
}

fn metal_state_for_mtp_tap_test(
    weights: &ModelWeights,
    cfg: &Qwen35Config,
    rotation: Option<crate::quant::quarot::hadamard::RandomizedHadamard>,
) -> MetalQwen35State {
    let hidden = cfg.hidden_size;
    let mut state = metal_state_with_synthetic_mtp_for_test(weights, cfg, rotation);
    state.engine.final_norm = make_buffer(
        &state.engine.device,
        &mtp_tap_final_norm_gamma(hidden),
        "test.final_norm.nonuniform",
    );
    let pre_fc_norm_hidden = make_buffer(
        &state.engine.device,
        &mtp_tap_pre_fc_hidden_gamma(hidden),
        "test.mtp.pre_fc_norm_hidden.nonuniform",
    );
    state
        .engine
        .mtp_weights
        .as_mut()
        .expect("synthetic MTP weights present")
        .pre_fc_norm_hidden = pre_fc_norm_hidden;
    state
}

/// The hidden half of the fused `[embedding || hidden]` buffer the head's
/// `fc` consumes, as left by the most recent MTP entry-point call.
fn mtp_fused_hidden_half(state: &MetalQwen35State) -> Vec<f32> {
    let hidden = state.engine.config.hidden_size;
    let fused = &state.session.mtp.as_ref().unwrap().activations.fused;
    // SAFETY: `fused` is a StorageModeShared f32 buffer of `2 * hidden`
    // values; both MTP entry points wait for their command buffer before
    // returning, so no GPU write is in flight here.
    unsafe {
        std::slice::from_raw_parts((fused.contents() as *const f32).add(hidden), hidden).to_vec()
    }
}

/// Decode path, unrotated checkpoint: the head's hidden input is
/// `pre_fc_norm_hidden(final_norm(h))`, not `pre_fc_norm_hidden(h)`.
#[test]
fn mtp_forward_one_feeds_head_the_post_final_norm_hidden() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let hidden = cfg.hidden_size;
    let mut state = metal_state_for_mtp_tap_test(&weights, &cfg, None);
    assert_eq!(state.mtp_hidden_tap(), Some(MtpHiddenTap::PostFinalNorm));

    let h = mtp_tap_target_hidden(hidden);
    let gamma_final = mtp_tap_final_norm_gamma(hidden);
    let gamma_pre = mtp_tap_pre_fc_hidden_gamma(hidden);
    let eps = cfg.rms_norm_eps;
    let expected_post = shifted_rmsnorm_reference(
        &shifted_rmsnorm_reference(&h, &gamma_final, eps),
        &gamma_pre,
        eps,
    );
    let without_final_norm = shifted_rmsnorm_reference(&h, &gamma_pre, eps);
    let separation = max_abs_diff(&expected_post, &without_final_norm);
    assert!(
        separation > 0.2,
        "fixture must separate the two candidate inputs; max diff {separation}"
    );

    state.session.last_pre_final_hidden = h;
    let _ = state.mtp_forward_one(2u32, 0);
    let got = mtp_fused_hidden_half(&state);

    let to_post = max_abs_diff(&got, &expected_post);
    let to_pre = max_abs_diff(&got, &without_final_norm);
    assert!(
        to_post < 1e-4,
        "hidden half must equal pre_fc_norm_hidden(final_norm(h)); max diff {to_post} \
                 (distance to the no-final-norm candidate: {to_pre})"
    );
    assert!(
        to_pre > 0.2,
        "hidden half must not equal pre_fc_norm_hidden(h); max diff {to_pre}"
    );
}

/// Prompt/accepted-row path, unrotated checkpoint: same input as the decode path.
#[test]
fn mtp_prefill_append_feeds_head_the_post_final_norm_hidden() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let hidden = cfg.hidden_size;
    let mut state = metal_state_for_mtp_tap_test(&weights, &cfg, None);
    assert_eq!(state.mtp_hidden_tap(), Some(MtpHiddenTap::PostFinalNorm));

    let h = mtp_tap_target_hidden(hidden);
    let gamma_final = mtp_tap_final_norm_gamma(hidden);
    let gamma_pre = mtp_tap_pre_fc_hidden_gamma(hidden);
    let eps = cfg.rms_norm_eps;
    let expected_post = shifted_rmsnorm_reference(
        &shifted_rmsnorm_reference(&h, &gamma_final, eps),
        &gamma_pre,
        eps,
    );
    let without_final_norm = shifted_rmsnorm_reference(&h, &gamma_pre, eps);
    let separation = max_abs_diff(&expected_post, &without_final_norm);
    assert!(
        separation > 0.2,
        "fixture must separate the two candidate inputs; max diff {separation}"
    );

    state.mtp_prefill_append(2u32, &h, 3);
    let got = mtp_fused_hidden_half(&state);

    let to_post = max_abs_diff(&got, &expected_post);
    let to_pre = max_abs_diff(&got, &without_final_norm);
    assert!(
        to_post < 1e-4,
        "hidden half must equal pre_fc_norm_hidden(final_norm(h)); max diff {to_post} \
                 (distance to the no-final-norm candidate: {to_pre})"
    );
    assert!(
        to_pre > 0.2,
        "hidden half must not equal pre_fc_norm_hidden(h); max diff {to_pre}"
    );
}

/// Rotated checkpoint: the tap stays pre-final-norm and both entry points
/// compute exactly the counter-rotate-then-`pre_fc_norm_hidden` input, with
/// no final norm even though the engine holds a non-uniform final-norm scale.
#[test]
fn mtp_rotated_checkpoint_keeps_the_pre_final_norm_hidden() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let hidden = cfg.hidden_size;
    let rotation =
        crate::quant::quarot::hadamard::RandomizedHadamard::new(0x51_51_51_u64, hidden).unwrap();

    let h = mtp_tap_target_hidden(hidden);
    let mut h_rotated = h.clone();
    rotation.apply(&mut h_rotated).unwrap();
    let gamma_pre = mtp_tap_pre_fc_hidden_gamma(hidden);

    // The input as computed before the tap existed: counter-rotation, then
    // the pre-fc norm.
    let mut unchanged = h_rotated.clone();
    rotation.apply_inverse(&mut unchanged).unwrap();
    let mut with_final_norm = unchanged.clone();
    MetalQwen35State::mtp_pre_fc_rmsnorm(
        &mut with_final_norm,
        &mtp_tap_final_norm_gamma(hidden),
        cfg.rms_norm_eps,
    );
    MetalQwen35State::mtp_pre_fc_rmsnorm(&mut unchanged, &gamma_pre, cfg.rms_norm_eps);
    MetalQwen35State::mtp_pre_fc_rmsnorm(&mut with_final_norm, &gamma_pre, cfg.rms_norm_eps);
    let separation = max_abs_diff(&unchanged, &with_final_norm);
    assert!(
        separation > 0.2,
        "fixture must separate the two candidate inputs; max diff {separation}"
    );

    let mut decode_state = metal_state_for_mtp_tap_test(&weights, &cfg, Some(rotation.clone()));
    assert_eq!(
        decode_state.mtp_hidden_tap(),
        Some(MtpHiddenTap::PreFinalNorm)
    );
    decode_state.session.last_pre_final_hidden = h_rotated.clone();
    let _ = decode_state.mtp_forward_one(2u32, 0);
    assert_eq!(
        mtp_fused_hidden_half(&decode_state),
        unchanged,
        "decode path on a rotated checkpoint must feed the head exactly the \
                 counter-rotated, pre_fc-normed hidden with no final norm"
    );

    let mut prefill_state = metal_state_for_mtp_tap_test(&weights, &cfg, Some(rotation));
    assert_eq!(
        prefill_state.mtp_hidden_tap(),
        Some(MtpHiddenTap::PreFinalNorm)
    );
    prefill_state.mtp_prefill_append(2u32, &h_rotated, 3);
    assert_eq!(
        mtp_fused_hidden_half(&prefill_state),
        unchanged,
        "prompt/accepted-row path on a rotated checkpoint must feed the head \
                 exactly the counter-rotated, pre_fc-normed hidden with no final norm"
    );
}

#[test]
fn mtp_hidden_tap_is_none_without_mtp_weights() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let engine = MetalQwen35Engine::new(&weights, &cfg)
        .expect("tiny MetalQwen35Engine without MTP weights constructs");
    assert!(engine.mtp_weights.is_none());
    let session = engine.new_session(16).expect("tiny session constructs");
    let state = MetalQwen35State {
        engine,
        session,
        lora: None,
        use_gdn_chunked: true,
        use_kv_f16: false,
        cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
        path_proof_enabled: false,
        path_proof: PathProofCounters::default(),
    };
    assert!(!state.has_mtp());
    assert_eq!(state.mtp_hidden_tap(), None);
}

#[test]
fn mtp_hidden_tap_follows_checkpoint_rotation_on_constructed_states() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let plain = metal_state_with_synthetic_mtp_for_test(&weights, &cfg, None);
    assert!(plain.engine.quarot_rotation.is_none());
    assert_eq!(plain.mtp_hidden_tap(), Some(MtpHiddenTap::PostFinalNorm));

    let rotation =
        crate::quant::quarot::hadamard::RandomizedHadamard::new(0x51_51_51_u64, cfg.hidden_size)
            .unwrap();
    let rotated = metal_state_with_synthetic_mtp_for_test(&weights, &cfg, Some(rotation));
    assert!(rotated.engine.quarot_rotation.is_some());
    assert_eq!(rotated.mtp_hidden_tap(), Some(MtpHiddenTap::PreFinalNorm));
}

/// Regression test on the PRODUCTION `mtp_forward_one` path (not a test-only
/// duplicate): a synthetic MTP fixture whose `pre_fc_norm_embedding` weight is
/// uniformly -0.5 and whose `fc` connects the embedding half of the concat as
/// an identity (`synthetic_mtp_weights_for_test`), with attention/MLP weights
/// zero so `layer_output == fc_output == normed_embed` exactly. Feeding token 2
/// (`tiny_metal_qwen35_fixture`'s one-hot embedding: component 0 is the only
/// nonzero value, +1.0) leaves `normed_embed` nonzero only at index 0, so its
/// sign after the final norm (a GPU kernel, always shifted-convention, always
/// scales by a positive factor) is exactly `pre_fc_norm_embedding`'s effective
/// scale: `1.0 + (-0.5) = +0.5` under the fix, `-0.5` under the bug. The tied
/// lm_head's row for token 2 has `embed_tokens[2][0] = +1.0`
/// (`tiny_metal_qwen35_fixture`'s `token % 3 == 2` convention), so
/// `output.logits[2]` inherits that sign directly: positive under the fix,
/// negative under the bug.
#[test]
fn mtp_forward_one_uses_shifted_pre_fc_norm_convention() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let mut engine = MetalQwen35Engine::new(&weights, &cfg)
        .expect("tiny MetalQwen35Engine with MTP fixture constructs");
    let mut mtp_weights = synthetic_mtp_weights_for_test(&engine.device, &cfg);
    mtp_weights.pre_fc_norm_embedding = make_buffer(
        &engine.device,
        &vec![-0.5f32; cfg.hidden_size],
        "test.mtp.pre_fc_norm_embedding.negative",
    );
    engine.mtp_weights = Some(mtp_weights);
    let session = engine.new_session(16).expect("tiny MTP session constructs");
    let mut state = MetalQwen35State {
        engine,
        session,
        lora: None,
        use_gdn_chunked: true,
        use_kv_f16: false,
        cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
        path_proof_enabled: false,
        path_proof: PathProofCounters::default(),
    };
    assert!(
        state.session.mtp.is_some(),
        "MTP fixture must populate session.mtp"
    );

    // token 2's embedding is one-hot: component 0 is +1.0, everything else 0
    // (`tiny_metal_qwen35_fixture`'s convention, also relied on by the #1341
    // test immediately below). `last_pre_final_hidden` is left at its default
    // (zero-length / zeroed) so the hidden half of the concat contributes
    // nothing measurable here -- this test isolates the embedding-half norm.
    state.session.last_pre_final_hidden = vec![0.0f32; cfg.hidden_size];
    let out = state.mtp_forward_one(2u32, 0);

    assert_eq!(out.logits.len(), cfg.vocab_size);
    assert!(
        out.logits[2] > 0.0,
        "logits[2] = {} must be POSITIVE under the shifted (1.0 + gamma) \
                 convention (effective scale 1.0 + (-0.5) = +0.5 applied to the \
                 embedding's +1.0 component, propagated through the identity `fc` and \
                 zero attention/MLP to the final norm and the tied lm_head row for \
                 token 2, embed_tokens[2][0] = +1.0); a negative value here means \
                 pre_fc_norm_embedding regressed to plain `gamma` (effective scale \
                 -0.5), inverting the sign of every channel it touches",
        out.logits[2]
    );
}

/// The prompt-row path (`mtp_prefill_append_dispatch`) and the decode draft
/// path (`mtp_forward_one_dispatch`) must write the same MTP K/V row for the
/// same token, hidden vector and position. Both pre-fc weights are
/// non-uniform and centre negative, so plain `gamma` and `1.0 + gamma`
/// differ in sign on most channels of both halves of the concat, and the
/// non-uniform values keep the difference visible after the
/// scale-invariant `input_layernorm` that follows `fc`. The rows are read
/// back from the MTP cache buffers.
#[test]
fn mtp_prefill_append_row_matches_decode_draft_row_with_negative_pre_fc_weights() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let hidden = cfg.hidden_size;
    let kv_dim = cfg.full_kv_dim();

    let gamma_embedding: Vec<f32> = (0..hidden)
        .map(|i| -0.5 + 0.1 * ((i as f32) * 0.11).sin())
        .collect();
    let gamma_hidden: Vec<f32> = (0..hidden)
        .map(|i| -0.6 + 0.2 * ((i as f32) * 0.07).cos())
        .collect();
    let hidden_in: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.013).sin() * 0.25 + 0.05)
        .collect();

    let build_state = || {
        let mut engine = MetalQwen35Engine::new(&weights, &cfg)
            .expect("tiny MetalQwen35Engine with nonzero-KV MTP fixture constructs");
        let mut mtp_weights = synthetic_mtp_weights_with_nonzero_kv_for_test(&engine.device, &cfg);
        mtp_weights.pre_fc_norm_embedding = make_buffer(
            &engine.device,
            &gamma_embedding,
            "test.mtp.pre_fc_norm_embedding.negative_nonuniform",
        );
        mtp_weights.pre_fc_norm_hidden = make_buffer(
            &engine.device,
            &gamma_hidden,
            "test.mtp.pre_fc_norm_hidden.negative_nonuniform",
        );
        engine.mtp_weights = Some(mtp_weights);
        let session = engine.new_session(16).expect("tiny MTP session constructs");
        MetalQwen35State {
            engine,
            session,
            lora: None,
            use_gdn_chunked: true,
            use_kv_f16: false,
            cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
            path_proof_enabled: false,
            path_proof: PathProofCounters::default(),
        }
    };

    fn read_row(buf: &Buffer, row: usize, row_len: usize) -> Vec<f32> {
        // SAFETY: `MetalMtpCache`'s `k_buf`/`v_buf` are StorageModeShared
        // f32 buffers; both dispatches wait for their command buffer
        // before returning, so no GPU write is in flight here.
        unsafe {
            let ptr = (buf.contents() as *const f32).add(row * row_len);
            std::slice::from_raw_parts(ptr, row_len).to_vec()
        }
    }

    // Token 2's embedding is one-hot at component 0 (+1.0).
    let token = 2u32;
    let position = 3usize;

    let mut decode_state = build_state();
    decode_state.session.last_pre_final_hidden = hidden_in.clone();
    let _ = decode_state.mtp_forward_one(token, position);
    assert_eq!(
        decode_state.session.mtp.as_ref().unwrap().cache.seq_len,
        1,
        "the decode draft must append exactly one MTP row"
    );
    let decode_k = read_row(
        &decode_state.session.mtp.as_ref().unwrap().cache.k_buf,
        0,
        kv_dim,
    );
    let decode_v = read_row(
        &decode_state.session.mtp.as_ref().unwrap().cache.v_buf,
        0,
        kv_dim,
    );

    let mut prefill_state = build_state();
    prefill_state.mtp_prefill_append(token, &hidden_in, position);
    assert_eq!(
        prefill_state.session.mtp.as_ref().unwrap().cache.seq_len,
        1,
        "the prompt-row path must append exactly one MTP row"
    );
    let prefill_k = read_row(
        &prefill_state.session.mtp.as_ref().unwrap().cache.k_buf,
        0,
        kv_dim,
    );
    let prefill_v = read_row(
        &prefill_state.session.mtp.as_ref().unwrap().cache.v_buf,
        0,
        kv_dim,
    );

    assert!(
        decode_k.iter().any(|&v| v != 0.0) && decode_v.iter().any(|&v| v != 0.0),
        "fixture sanity: the decode row must be non-zero for the comparison to \
                 discriminate; got K={decode_k:?} V={decode_v:?}"
    );
    assert_eq!(
        prefill_k, decode_k,
        "the prompt-row K row must equal the decode draft's K row: a mismatch \
                 means one pre-fc norm in `mtp_prefill_append_dispatch` applies plain \
                 `gamma` instead of the shifted `1.0 + gamma` convention"
    );
    assert_eq!(
        prefill_v, decode_v,
        "the prompt-row V row must equal the decode draft's V row: a mismatch \
                 means one pre-fc norm in `mtp_prefill_append_dispatch` applies plain \
                 `gamma` instead of the shifted `1.0 + gamma` convention"
    );
}

// Sibling of `synthetic_mtp_weights_for_test` with a NON-ZERO `fc` hidden
// half and NON-ZERO K/V projections (lattice#1396).
// `synthetic_mtp_weights_for_test`'s all-zero K/V
// projections -- and `constant_zero_draft_mtp_weights_for_test`'s
// additionally-zeroed `fc`, which collapses the fused hidden to zero
// regardless of input -- make every MTP cache row identically zero no
// matter which hidden state it is paired with, so neither existing
// fixture can tell a correct pairing from a wrong one. This variant
// keeps that same property for the OUTPUT logits (so the draft's own
// predicted next token stays deterministic), while making the raw
// cache row content depend on the paired hidden state.
//
// `fc` is identity on BOTH halves (`fc_out[i] = normed_embed[i] +
// normed_hidden[i]`, instead of `synthetic_mtp_weights_for_test`'s
// embedding-half-only identity), so the fused hidden actually carries
// the paired hidden state's contribution forward into K/V. `k_proj`/
// `v_proj` are identity on their first `full_kv_dim()` rows, scaled
// differently (`k=1x`, `v=2x`) so a K/V transposition would also be
// caught. `q_proj` and `o_proj` stay zero: a zero `o_proj` nullifies
// the attention branch's contribution to the OUTPUT hidden regardless
// of what `k_proj`/`v_proj` write into the cache (the attention
// residual becomes `residual + 0` either way), so this change is
// invisible to the draft's own argmax and the deterministic
// accept/reject construction other MTP tests in this module rely on
// still holds unchanged.
fn synthetic_mtp_weights_with_nonzero_kv_for_test(
    device: &Device,
    cfg: &Qwen35Config,
) -> MetalMtpWeights {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();

    // fc shape: [hidden, 2*hidden]. The embedding half is identity
    // (`fc_out[i] = normed_embed[i]`). The hidden half is identity
    // shifted by ONE output index (`fc_out[(j + 1) % hidden] +=
    // normed_hidden[j]`) -- deliberately NOT same-index identity.
    //
    // `tiny_metal_qwen35_fixture`'s embeddings are
    // one-hot at dim 0 for every token, and this fixture's target
    // model has zero attn/FFN weights, so a pending token's own
    // pre-final hidden (what the accept arm pairs the appended row
    // with) is numerically the SAME one-hot vector as that token's
    // own embedding. With same-index identity on both halves,
    // `fc_out` became a positive scalar multiple of one shared
    // direction (`normed_embed` alone for the wrong pairing vs
    // `normed_embed + normed_hidden`, both pointing along `e_0`, for
    // the correct one) -- and the `input_layernorm` RMSNorm this
    // function's GPU phase dispatches immediately afterward
    // (`dispatch_copy_and_rms_norm`) is invariant to a uniform
    // positive rescale of its whole input. So the two pairings
    // collapsed to an IDENTICAL normalized hidden, and every row
    // derived from it (K via `k_proj`, V via `v_proj`) came out
    // identical too: `correct_ref_k == wrong_ref_k`, failing the
    // test's own fixture-sanity assert before it could discriminate
    // anything. Landing the hidden half on a DIFFERENT output index
    // makes the two pairings differ in DIRECTION, not merely
    // magnitude, which RMSNorm cannot erase.
    let mut fc = vec![0.0f32; hidden * 2 * hidden];
    for i in 0..hidden {
        fc[i * (2 * hidden) + i] = 1.0; // embedding half -> output i
    }
    for j in 0..hidden {
        let out_idx = (j + 1) % hidden;
        fc[out_idx * (2 * hidden) + hidden + j] = 1.0; // hidden half -> output (j+1) mod hidden
    }

    // k_proj/v_proj shape: [kv_dim, hidden]; identity (scaled
    // differently for K vs V) on the first `kv_dim` input dims.
    let mut k_proj = vec![0.0f32; kv_dim * hidden];
    let mut v_proj = vec![0.0f32; kv_dim * hidden];
    for i in 0..kv_dim {
        k_proj[i * hidden + i] = 1.0;
        v_proj[i * hidden + i] = 2.0;
    }

    MetalMtpWeights {
        fc: make_buffer_f16(device, &fc, "test.mtp.fc.nonzero_kv"),
        pre_fc_norm_embedding: make_buffer(
            device,
            &vec![1.0; hidden],
            "test.mtp.pre_fc_norm_embedding",
        ),
        pre_fc_norm_hidden: make_buffer(device, &vec![1.0; hidden], "test.mtp.pre_fc_norm_hidden"),
        layers: vec![MetalMtpLayerWeights {
            input_layernorm: make_buffer(device, &vec![1.0; hidden], "test.mtp.input_ln"),
            post_attention_layernorm: make_buffer(
                device,
                &vec![1.0; hidden],
                "test.mtp.post_attn_ln",
            ),
            q_proj: make_buffer_f16(device, &vec![0.0; 2 * q_dim * hidden], "test.mtp.q_proj"),
            k_proj: make_buffer_f16(device, &k_proj, "test.mtp.k_proj.nonzero"),
            v_proj: make_buffer_f16(device, &v_proj, "test.mtp.v_proj.nonzero"),
            o_proj: make_buffer_f16(device, &vec![0.0; hidden * q_dim], "test.mtp.o_proj"),
            q_norm: make_buffer(device, &vec![1.0; cfg.head_dim], "test.mtp.q_norm"),
            k_norm: make_buffer(device, &vec![1.0; cfg.head_dim], "test.mtp.k_norm"),
            mlp: MetalMtpDenseMlpWeights {
                gate_proj: make_buffer_f16(
                    device,
                    &vec![0.0; inter * hidden],
                    "test.mtp.gate_proj",
                ),
                up_proj: make_buffer_f16(device, &vec![0.0; inter * hidden], "test.mtp.up_proj"),
                down_proj: make_buffer_f16(
                    device,
                    &vec![0.0; hidden * inter],
                    "test.mtp.down_proj",
                ),
            },
        }],
        norm: make_buffer(device, &vec![1.0; hidden], "test.mtp.norm"),
        hidden_tap: MtpHiddenTap::PostFinalNorm,
    }
}

fn metal_state_with_nonzero_kv_mtp_for_test(
    weights: &ModelWeights,
    cfg: &Qwen35Config,
) -> MetalQwen35State {
    let mut engine = MetalQwen35Engine::new(weights, cfg)
        .expect("tiny MetalQwen35Engine with nonzero-KV MTP fixture constructs");
    engine.mtp_weights = Some(synthetic_mtp_weights_with_nonzero_kv_for_test(
        &engine.device,
        cfg,
    ));
    let session = engine.new_session(16).expect("tiny MTP session constructs");
    MetalQwen35State {
        engine,
        session,
        lora: None,
        use_gdn_chunked: true,
        use_kv_f16: false,
        cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
        path_proof_enabled: false,
        path_proof: PathProofCounters::default(),
    }
}

#[test]
fn rollback_speculative_state_to_preserves_mtp_cursor_after_k1_reject() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let mut state = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
    assert!(
        state.session.mtp.is_some(),
        "constant-draft MTP fixture must populate session.mtp"
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(true),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    // Force `pending_first == 2` directly rather than relying on a real
    // prefill: index 2 is the token whose real-model next-token prediction
    // (token 2 itself, see above) is guaranteed to mismatch the constant
    // draft (token 0).
    let mut prefill_logits = vec![-1.0f32; cfg.vocab_size];
    prefill_logits[2] = 100.0;

    let pos_before = state.session.kv_cache.seq_len;
    assert_eq!(pos_before, 0);
    let out = state
        .generate_greedy_mtp(&prefill_logits, 0, &tokenizer, &gen_cfg)
        .expect("mtp route");
    assert!(
        !out.stopped,
        "test assumes round 1 does not hit EOS; got {out:?}"
    );

    // K=1 reject keeps only `pending_token` (2), advancing the KV cache by 1;
    // a K=1 accept would keep `pending_token` + the draft, advancing by 2.
    let kv_seq_len = state.session.kv_cache.seq_len;
    assert_eq!(
        kv_seq_len, 1,
        "test setup assumption violated: expected round 1 to reject the \
                 constant draft (token 0) in favor of the target's own prediction \
                 (token 2), advancing the KV cache by 1; got {kv_seq_len}, meaning \
                 the draft was accepted instead"
    );

    let mtp_seq_len = state.session.mtp.as_ref().unwrap().cache.seq_len;
    assert_eq!(
        mtp_seq_len, 1,
        "K=1 rejection must leave the MTP cursor at the single row \
                 `mtp_forward_one` wrote this round; got {mtp_seq_len}, which means the \
                 next `mtp_forward_one` call will skip a slot and the attention window \
                 will read a never-written row"
    );
}

// `verify_tokens_batched`'s sequential-verifier path leaves
// `last_pre_final_hidden` set to the hidden state captured while processing
// the rejected draft token, because `forward_step_inner` writes that field
// for every token it processes, with no distinction between "provisional,
// pending verification" and "committed". A K=1 rollback undoes the GDN/KV
// mutations but not this field, so the next `mtp_forward_one` call is fed
// the wrong token's hidden state. This test isolates the field: after a
// forced K=1 reject, `last_pre_final_hidden` must equal the hidden state
// from processing ONLY the real `pending_token`, not the rejected draft.
#[test]
fn generate_greedy_mtp_restores_correct_hidden_after_k1_reject() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;

    // Reference: the hidden state that processing ONLY the real
    // `pending_token` (2) at position 0 produces -- no draft token involved.
    // This is what `last_pre_final_hidden` must equal after the K=1 reject
    // round below, per `mtp_forward_one`'s documented pairing contract
    // ("pairs `pending_token`'s embedding with `last_pre_final_hidden` --
    // the hidden state that predicted it").
    let mut reference_state = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
    let _ =
        reference_state.forward_step_inner(2, 0, true, crate::forward::signpost::Scope::NotDecode);
    let expected_hidden = reference_state.session.last_pre_final_hidden.clone();
    assert!(
        !expected_hidden.is_empty(),
        "reference forward_step_inner call must capture a hidden state"
    );

    let mut state = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(true),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };
    let mut prefill_logits = vec![-1.0f32; cfg.vocab_size];
    prefill_logits[2] = 100.0;
    let out = state
        .generate_greedy_mtp(&prefill_logits, 0, &tokenizer, &gen_cfg)
        .expect("mtp route");
    assert!(
        !out.stopped,
        "test assumes round 1 does not hit EOS; got {out:?}"
    );
    assert_eq!(
        state.session.kv_cache.seq_len, 1,
        "test setup assumption violated: expected round 1 to reject the \
                 constant draft (token 0) in favor of the target's own prediction \
                 (token 2)"
    );

    let actual_hidden = &state.session.last_pre_final_hidden;
    assert_eq!(
        actual_hidden.len(),
        expected_hidden.len(),
        "last_pre_final_hidden length changed unexpectedly"
    );
    let max_abs_diff = actual_hidden
        .iter()
        .zip(expected_hidden.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-5,
        "last_pre_final_hidden after a K=1 reject must equal the hidden state \
                 that predicted the real continuation (from processing pending_token \
                 alone), not the rejected draft's hidden state; \
                 max_abs_diff={max_abs_diff} actual={actual_hidden:?} \
                 expected={expected_hidden:?}"
    );
}

// lattice#1396: a K=1 full accept must give the MTP cache a row for the
// accepted draft token, not just for `pending_token`. `mtp_forward_one`
// writes exactly one row (for `pending_token`) and advances
// `mtp.cache.seq_len` by 1 before verification runs; on a full accept the
// draft token commits as a real, generated token at `pos + 1` without ever
// getting its own MTP-cache row, so the cursor stalls at `c0 + 1` instead of
// `c0 + 2` and the next round's `mtp_forward_one` call lands one physical
// slot behind its own RoPE position -- a positional hole per accepted
// transition (the issue's exact defect).
//
// Reuses `rollback_speculative_state_to_preserves_mtp_cursor_after_k1_reject`'s
// constant-zero-draft fixture (the draft head always predicts token 0), but
// forces `pending_first == 0` instead of `2`: token 0's real-model
// next-token prediction is token 0 itself (`0 % 3 == 0` gives it the
// negative-embedding residue class the target's zero-attention/zero-FFN
// stack turns into the logit maximum, and 0 is the first index in that
// class), which matches the constant draft (token 0) exactly -- a
// deterministic full accept, driven through the real `generate_greedy_mtp`
// path.
#[test]
fn full_accept_appends_mtp_cache_row_for_accepted_draft_token() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let mut state = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
    assert!(
        state.session.mtp.is_some(),
        "constant-draft MTP fixture must populate session.mtp"
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(true),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    // Force `pending_first == 0` directly rather than relying on a real
    // prefill, mirroring the sibling reject test's technique.
    let mut prefill_logits = vec![-1.0f32; cfg.vocab_size];
    prefill_logits[0] = 100.0;

    let pos_before = state.session.kv_cache.seq_len;
    assert_eq!(pos_before, 0);
    let out = state
        .generate_greedy_mtp(&prefill_logits, 0, &tokenizer, &gen_cfg)
        .expect("mtp route");
    assert!(
        !out.stopped,
        "test assumes round 1 does not hit EOS; got {out:?}"
    );

    // K=1 full accept keeps `pending_token` (0) AND the draft (0), advancing
    // the target KV cache by 2; a K=1 reject would advance it by only 1.
    let kv_seq_len = state.session.kv_cache.seq_len;
    assert_eq!(
        kv_seq_len, 2,
        "test setup assumption violated: expected round 1 to accept the \
                 constant draft (token 0), matching the target's own prediction for \
                 pending token 0, advancing the target KV cache by 2; got \
                 {kv_seq_len}, meaning the draft was rejected instead"
    );

    // The defect under test: after a full accept, the MTP cache cursor must
    // reach `c0 + 2` (one row for `pending_token`, one for the accepted
    // draft token), holding a row at every logical position through the
    // accepted token -- not `c0 + 1` with the accepted draft's own position
    // left as a hole.
    let mtp_seq_len = state.session.mtp.as_ref().unwrap().cache.seq_len;
    assert_eq!(
        mtp_seq_len, 2,
        "K=1 full accept must leave the MTP cursor two rows past where it \
                 started: one row `mtp_forward_one` wrote for `pending_token`, one \
                 row the accept arm must append for the accepted draft token; got \
                 {mtp_seq_len}, meaning the accepted draft's own row was never \
                 appended and the next round's `mtp_forward_one` will land one \
                 physical slot behind its RoPE position"
    );
}

// lattice#1396: `full_accept_appends_mtp_cache_row_for_accepted_draft_token`
// (#1731) proves only that the appended row EXISTS (the cursor reaches
// `c0 + 2`) -- its fixture has all-zero MTP K/V projection weights, so
// the appended row's raw content is zero regardless of which hidden
// state was paired with the accepted token, and a wrong pairing would
// be silently invisible to that test. This test uses
// `synthetic_mtp_weights_with_nonzero_kv_for_test`, whose non-zero `fc`
// hidden half and non-zero K/V projections make the cache row's raw
// content actually depend on the paired hidden state, and reads that
// content back to compare against two independently-built reference
// rows built through the SAME `mtp_prefill_append` primitive the
// accept arm calls:
//   - the INTENDED pairing: the accepted token with the target's own
//     pre-final hidden state that predicted it (`pending_token`'s own
//     pre-final hidden, i.e. `verify_out.first_pre_final_hidden`),
//     derived independently via a fresh `forward_step_inner` call
//     (the same technique
//     `generate_greedy_mtp_restores_correct_hidden_after_k1_reject`
//     uses for its reference hidden);
//   - a PLAUSIBLE WRONG pairing: the accepted token with
//     `self.session.last_pre_final_hidden`'s value as it actually
//     reads right after `verify_tokens_batched` returns -- this is
//     NOT a hardcoded/arbitrary value. `verify_tokens_batched`'s
//     per-token loop (this file, `fn verify_tokens_batched`) calls
//     `forward_step_inner(_, _, true, _)` once per verify token, and
//     that function (both branches, ~6664 and ~6862 in this file)
//     unconditionally writes `self.session.last_pre_final_hidden =
//     h.clone()` on every call -- so after the loop's *second*
//     iteration (the draft token), `last_pre_final_hidden` holds the
//     draft token's own pre-final hidden, overwriting the value the
//     first iteration (`pending_token`) produced. A wrong-pairing
//     change that substitutes `self.session.last_pre_final_hidden`
//     for `verify_out.first_pre_final_hidden` at the accept-arm's
//     `mtp_prefill_append` call therefore pairs the accepted token
//     with the draft token's *own* re-verified hidden state instead
//     of the pending token's.
//
//     An earlier version of this test forced `pending_token == 0`,
//     which (via this fixture's residue-class argmax construction,
//     see `full_accept_appends_mtp_cache_row_for_accepted_draft_token`)
//     makes the accepted draft token ALSO equal 0 -- the same token
//     id twice. Since this fixture's target attention/FFN are all
//     zero (`tiny_metal_qwen35_fixture`), a token's pre-final hidden
//     equals its own raw embedding independent of position, so
//     verifying token 0 at position 0 and token 0 again at position 1
//     produce BIT-IDENTICAL hidden states: `first_pre_final_hidden`
//     and the post-loop `last_pre_final_hidden` were provably equal
//     on that construction, making the wrong-pairing mutation an
//     unobservable no-op purely as an artifact of pending_token ==
//     draft.token_id, not because the bug class is unreal. Forcing
//     `pending_token = 1` instead breaks that coincidence: token 1
//     is in residue class 1 (`tiny_metal_qwen35_fixture`'s
//     `token % 3 == 1` branch), whose embedding is the all-zero
//     vector, so the target's tied-lm-head logits tie at zero across
//     the whole vocabulary and first-wins argmax resolves to token 0
//     -- the same token the MTP draft head resolves to via the
//     identical zero-input/first-wins mechanism (`fc`'s embedding
//     half is an identity copy per `synthetic_mtp_weights_for_test`'s
//     doc comment, and this fixture's own draft-side attention/FFN
//     are likewise all zero) -- still a deterministic K=1 full
//     accept, but now `pending_token` (1, embedding all-zero) and
//     `draft.token_id` (0, embedding one-hot at dim 0) are genuinely
//     different tokens with genuinely different embeddings, so their
//     pre-final hidden states differ and the wrong pairing is
//     observable.
#[test]
fn full_accept_mtp_cache_row_pairs_accepted_token_with_target_hidden() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let kv_dim = cfg.full_kv_dim();

    // Independently derive the target's real pre-final hidden after
    // processing `pending_token` (1) alone -- the value
    // `verify_out.first_pre_final_hidden` must equal per #1731's
    // pairing contract -- independent of the production round below.
    let mut h0_state = metal_state_with_nonzero_kv_mtp_for_test(&weights, &cfg);
    let _ = h0_state.forward_step_inner(1, 0, true, crate::forward::signpost::Scope::NotDecode);
    let h0 = h0_state.session.last_pre_final_hidden.clone();
    assert!(
        !h0.is_empty(),
        "reference forward_step_inner must capture a hidden state"
    );

    // Independently derive the SAME value
    // `self.session.last_pre_final_hidden` actually holds right after
    // `verify_tokens_batched` returns in the production round below:
    // the accepted draft token's (0) own pre-final hidden, from
    // `verify_tokens_batched`'s loop's second (and last) iteration --
    // see the doc comment above for why this, not an arbitrary or
    // hardcoded value, is the plausible wrong pairing a real mutation
    // produces.
    let mut wrong_state = metal_state_with_nonzero_kv_mtp_for_test(&weights, &cfg);
    let _ = wrong_state.forward_step_inner(0, 0, true, crate::forward::signpost::Scope::NotDecode);
    let wrong_hidden = wrong_state.session.last_pre_final_hidden.clone();
    assert_ne!(
        h0, wrong_hidden,
        "test setup requires the correct and wrong hidden states to \
                 differ, otherwise this test cannot distinguish a correct \
                 pairing from a wrong one"
    );

    fn read_row(buf: &Buffer, row: usize, row_len: usize) -> Vec<f32> {
        // SAFETY: `MetalMtpCache`'s `k_buf`/`v_buf` are
        // StorageModeShared f32 buffers (`make_zero_buffer`), written
        // by command buffers that have already completed by the time
        // the dispatching call (`mtp_prefill_append`/
        // `generate_greedy_mtp`) returns.
        unsafe {
            let ptr = (buf.contents() as *const f32).add(row * row_len);
            std::slice::from_raw_parts(ptr, row_len).to_vec()
        }
    }

    // Reference row: append the INTENDED (accepted-token,
    // correct-hidden) pair directly via the same primitive the accept
    // arm calls, on a fresh state with identical weights. Position 1
    // matches `pos + 1` for a round-1 accept (`pos == 0`); a fresh
    // cache's first append lands at row index 0 regardless of the
    // RoPE `position` argument (`mtp_prefill`'s own convention: the
    // row index tracks the cache cursor, the `position` argument only
    // feeds RoPE), which is why this reads row 0 here but row 1 from
    // the production state below.
    let mut correct_ref_state = metal_state_with_nonzero_kv_mtp_for_test(&weights, &cfg);
    correct_ref_state.mtp_prefill_append(0, &h0, 1);
    let correct_ref_k = read_row(
        &correct_ref_state.session.mtp.as_ref().unwrap().cache.k_buf,
        0,
        kv_dim,
    );
    let correct_ref_v = read_row(
        &correct_ref_state.session.mtp.as_ref().unwrap().cache.v_buf,
        0,
        kv_dim,
    );

    // Reference row: the same accepted token paired with the WRONG
    // hidden.
    let mut wrong_ref_state = metal_state_with_nonzero_kv_mtp_for_test(&weights, &cfg);
    wrong_ref_state.mtp_prefill_append(0, &wrong_hidden, 1);
    let wrong_ref_k = read_row(
        &wrong_ref_state.session.mtp.as_ref().unwrap().cache.k_buf,
        0,
        kv_dim,
    );
    let wrong_ref_v = read_row(
        &wrong_ref_state.session.mtp.as_ref().unwrap().cache.v_buf,
        0,
        kv_dim,
    );
    assert!(
        correct_ref_k != wrong_ref_k || correct_ref_v != wrong_ref_v,
        "test setup requires the two hidden states to actually produce \
                 different cache content through this fixture's non-zero K/V \
                 weights; got identical K={correct_ref_k:?} V={correct_ref_v:?} \
                 for both pairings"
    );

    // Now drive the real production round.
    let mut state = metal_state_with_nonzero_kv_mtp_for_test(&weights, &cfg);
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(true),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };
    let mut prefill_logits = vec![-1.0f32; cfg.vocab_size];
    prefill_logits[1] = 100.0;

    let out = state
        .generate_greedy_mtp(&prefill_logits, 0, &tokenizer, &gen_cfg)
        .expect("mtp route");
    assert!(
        !out.stopped,
        "test assumes round 1 does not hit EOS; got {out:?}"
    );
    assert_eq!(
        state.session.kv_cache.seq_len, 2,
        "test setup assumption violated: expected round 1 to accept the \
                 MTP draft (token 0), matching the target's own prediction for \
                 pending token 1 -- pending token 1's embedding is the \
                 all-zero vector (residue class 1), so the target's tied-head \
                 logits tie at zero across the whole vocabulary and \
                 first-wins argmax resolves to token 0, the same token the \
                 zero-input MTP draft head resolves to; got {}",
        state.session.kv_cache.seq_len
    );
    let mtp_seq_len = state.session.mtp.as_ref().unwrap().cache.seq_len;
    assert_eq!(
        mtp_seq_len, 2,
        "full accept must append a second MTP cache row; got {mtp_seq_len}"
    );

    let actual_k = read_row(&state.session.mtp.as_ref().unwrap().cache.k_buf, 1, kv_dim);
    let actual_v = read_row(&state.session.mtp.as_ref().unwrap().cache.v_buf, 1, kv_dim);

    assert_eq!(
        actual_k, correct_ref_k,
        "appended row's K content must match the intended \
                 (accepted-token, preceding-target-hidden) pairing; a wrong \
                 hidden state was paired instead"
    );
    assert_eq!(
        actual_v, correct_ref_v,
        "appended row's V content must match the intended \
                 (accepted-token, preceding-target-hidden) pairing; a wrong \
                 hidden state was paired instead"
    );
    assert_ne!(
        actual_k, wrong_ref_k,
        "appended row's K content coincidentally matches the plausible \
                 WRONG pairing (accepted token 0 + its own re-verified hidden \
                 state, the value `self.session.last_pre_final_hidden` holds \
                 after `verify_tokens_batched`'s second loop iteration) -- \
                 this pairing test cannot distinguish correct from wrong \
                 content for this fixture"
    );
}

// #1340: without `mtp_prefill`, `reset_state` zeroes the MTP cache and the
// only prefill on the Metal path fills the *target* model's cache, so the
// first `mtp_forward_one` call of a generation dispatches attention with
// `cache_len = 1` -- the draft attends to exactly the row it just wrote,
// independent of the prompt. This asserts the observable attention-set size
// (`cache.seq_len`, and the `cache_len` the first live dispatch sees), not
// generated text -- a target-verified MTP draft can never change output, so
// an output-only test cannot see this defect either way.
#[test]
fn mtp_prefill_builds_kv_cache_from_prompt_prefix() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let mut state = metal_state_with_synthetic_mtp_for_test(&weights, &cfg, None);
    assert!(
        state.session.mtp.is_some(),
        "synthetic MTP fixture must populate session.mtp"
    );

    let prompt_ids: Vec<u32> = vec![3, 7, 11, 19];
    let prompt_len = prompt_ids.len();

    let _ = state.forward_prefill(&prompt_ids);
    state.mtp_prefill(&prompt_ids);

    // One cache entry per prompt position beyond the first (position 0 has
    // no predecessor hidden state to pair with), so the cache scales
    // directly with the prompt instead of staying at 0.
    assert_eq!(
        state.session.mtp.as_ref().unwrap().cache.seq_len,
        prompt_len - 1,
        "mtp cache seq_len after prefill must reflect the prompt, not stay at 0"
    );

    // The first live draft call (mirroring `generate_greedy_mtp`'s first
    // round) appends one more entry for the just-sampled token, so its
    // attention dispatch sees cache_len == prompt_len, not 1.
    let before = state.session.mtp.as_ref().unwrap().cache.seq_len;
    let pending_token = 5u32;
    let _ = state.mtp_forward_one(pending_token, prompt_len);
    let after = state.session.mtp.as_ref().unwrap().cache.seq_len;
    assert_eq!(after, before + 1);
    assert_eq!(
        after, prompt_len,
        "the first draft's attention dispatch must see cache_len == prompt_len, not 1"
    );
}

#[test]
fn test_metal_qwen35_golden_logit_snapshot_forward_step_token_42_pos_0() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");

    let logits = state.forward_step(42, 0);
    assert_eq!(logits.len(), cfg.vocab_size);
    let actual = &logits[..10];
    // Math: token 42's embedding is one-hot: x[0]=1.0, x[1..]=0.0.
    // All attention and FFN weights are zero → residual stream equals the
    // raw embedding at every stage. final_norm then applies the shifted
    // RMSNorm (qwen35_rms_norm convention: output = x * (1 + gamma) / rms(x)).
    // With final_norm=[1.0] the scale is (1+1.0)=2; identity is gamma=0.
    //   rms(x) = sqrt(1/512),  output[0] = 1.0 * (1+1.0) * sqrt(512) = 2*sqrt(512) ≈ 45.254.
    // Tied lm_head col-0 pattern is [-1,0,1,-1,0,1,...], so logits ≈ ±45.25 / 0.
    // (Issue #31: the original golden ±22.62 assumed plain-gamma; ±45.24 is correct.)
    let expected = [
        -45.243256_f32,
        0.0,
        45.243256,
        -45.243256,
        0.0,
        45.243256,
        -45.243256,
        0.0,
        45.243256,
        -45.243256,
    ];
    let max_abs_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-4,
        "golden first-10 logits changed: actual={actual:?} expected={expected:?} max_abs_diff={max_abs_diff}"
    );
}

#[test]
fn forward_step_with_hidden_matches_logits_state_and_proves_explicit_readback() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=forward_step_with_hidden_matches_logits_state_and_proves_explicit_readback \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (forward_step_with_hidden_matches_logits_state_and_proves_explicit_readback)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut ordinary = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("ordinary tiny MetalQwen35State fixture constructs");
    let mut with_hidden = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("hidden tiny MetalQwen35State fixture constructs");
    ordinary.path_proof_enabled = true;
    with_hidden.path_proof_enabled = true;
    ordinary.reset_path_proof_counters();
    with_hidden.reset_path_proof_counters();

    let error = with_hidden
        .forward_step_with_hidden(cfg.vocab_size as u32, 0)
        .expect_err("out-of-vocabulary token must fail before dispatch");
    assert!(matches!(
        error,
        crate::error::InferenceError::InvalidInput(_)
    ));
    assert_eq!(with_hidden.session.kv_cache.seq_len, 0);
    assert_eq!(
        with_hidden.hidden_readback_path_proof_snapshot().decode,
        0,
        "invalid input must not read hidden state"
    );

    let ordinary_logits = ordinary.forward_step(42, 0);
    let (hidden_logits, hidden) = with_hidden
        .forward_step_with_hidden(42, 0)
        .expect("hidden-returning step succeeds");
    assert_forward_rows_close("step logits", &ordinary_logits, &hidden_logits);
    assert_eq!(ordinary.session.kv_cache.seq_len, 1);
    assert_eq!(with_hidden.session.kv_cache.seq_len, 1);
    assert_eq!(hidden.len(), cfg.hidden_size);
    assert!(hidden.iter().all(|value| value.is_finite()));
    assert!(hidden.iter().any(|&value| value != 0.0));
    assert_forward_rows_close(
        "returned and session hidden",
        &hidden,
        &with_hidden.session.last_pre_final_hidden,
    );
    assert!(
        ordinary
            .session
            .last_pre_final_hidden
            .iter()
            .all(|&value| value == 0.0),
        "ordinary step must not add an unconditional hidden readback"
    );

    let ordinary_hidden_proof = ordinary.hidden_readback_path_proof_snapshot();
    let explicit_hidden_proof = with_hidden.hidden_readback_path_proof_snapshot();
    assert_eq!(ordinary_hidden_proof.decode, 0);
    assert_eq!(explicit_hidden_proof.decode, 1);

    let ordinary_path = ordinary.path_proof_snapshot();
    let explicit_path = with_hidden.path_proof_snapshot();
    assert_eq!(ordinary_path.decode_kv_copy, explicit_path.decode_kv_copy);
    assert!(explicit_path.decode_kv_copy > 0);
    assert_eq!(
        ordinary_path.decode_attn_direct,
        explicit_path.decode_attn_direct
    );
    assert!(explicit_path.decode_attn_direct > 0);

    let ordinary_followup = ordinary.forward_step(5, 1);
    let hidden_followup = with_hidden.forward_step(5, 1);
    assert_forward_rows_close(
        "step follow-up logits",
        &ordinary_followup,
        &hidden_followup,
    );
    assert_eq!(ordinary.session.kv_cache.seq_len, 2);
    assert_eq!(with_hidden.session.kv_cache.seq_len, 2);
}

#[test]
fn forward_prefill_with_hidden_matches_logits_state_and_proves_explicit_readback() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=forward_prefill_with_hidden_matches_logits_state_and_proves_explicit_readback \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (forward_prefill_with_hidden_matches_logits_state_and_proves_explicit_readback)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut ordinary = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("ordinary tiny MetalQwen35State fixture constructs");
    let mut with_hidden = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("hidden tiny MetalQwen35State fixture constructs");
    ordinary.path_proof_enabled = true;
    with_hidden.path_proof_enabled = true;
    ordinary.reset_path_proof_counters();
    with_hidden.reset_path_proof_counters();

    let empty_error = with_hidden
        .forward_prefill_with_hidden(&[])
        .expect_err("empty prompt must fail before dispatch");
    assert!(matches!(
        empty_error,
        crate::error::InferenceError::InvalidInput(_)
    ));
    let oov_error = with_hidden
        .forward_prefill_with_hidden(&[cfg.vocab_size as u32])
        .expect_err("out-of-vocabulary token must fail before dispatch");
    assert!(matches!(
        oov_error,
        crate::error::InferenceError::InvalidInput(_)
    ));
    let oversized = vec![1_u32; 17];
    let capacity_error = with_hidden
        .forward_prefill_with_hidden(&oversized)
        .expect_err("prompt beyond session capacity must fail before dispatch");
    assert!(matches!(
        capacity_error,
        crate::error::InferenceError::InvalidInput(_)
    ));
    assert_eq!(with_hidden.session.kv_cache.seq_len, 0);
    assert_eq!(
        with_hidden.hidden_readback_path_proof_snapshot().prefill,
        0,
        "invalid inputs must not read hidden state"
    );

    let tokens = [42_u32, 2, 5];
    let ordinary_logits = ordinary.forward_prefill(&tokens);
    let (hidden_logits, hidden) = with_hidden
        .forward_prefill_with_hidden(&tokens)
        .expect("hidden-returning prefill succeeds");
    assert_forward_rows_close("prefill logits", &ordinary_logits, &hidden_logits);
    assert_eq!(ordinary.session.kv_cache.seq_len, tokens.len());
    assert_eq!(with_hidden.session.kv_cache.seq_len, tokens.len());
    assert_eq!(ordinary.session.position(), tokens.len());
    assert_eq!(with_hidden.session.position(), tokens.len());
    assert_eq!(hidden.len(), cfg.hidden_size);
    assert!(hidden.iter().all(|value| value.is_finite()));
    assert!(hidden.iter().any(|&value| value != 0.0));
    assert_forward_rows_close(
        "returned and session hidden",
        &hidden,
        &with_hidden.session.last_pre_final_hidden,
    );
    assert!(
        ordinary
            .session
            .last_pre_final_hidden
            .iter()
            .all(|&value| value == 0.0),
        "ordinary prefill must not add an unconditional hidden readback"
    );

    let ordinary_hidden_proof = ordinary.hidden_readback_path_proof_snapshot();
    let explicit_hidden_proof = with_hidden.hidden_readback_path_proof_snapshot();
    assert_eq!(ordinary_hidden_proof.prefill, 0);
    assert_eq!(explicit_hidden_proof.prefill, 1);

    let ordinary_path = ordinary.path_proof_snapshot();
    let explicit_path = with_hidden.path_proof_snapshot();
    assert_eq!(
        ordinary_path.prefill_kv_batch,
        explicit_path.prefill_kv_batch
    );
    assert!(explicit_path.prefill_kv_batch > 0);
    assert_eq!(
        ordinary_path.prefill_attn_batched,
        explicit_path.prefill_attn_batched
    );
    assert!(explicit_path.prefill_attn_batched > 0);

    let ordinary_followup = ordinary.forward_step(7, tokens.len());
    let hidden_followup = with_hidden.forward_step(7, tokens.len());
    assert_forward_rows_close(
        "prefill follow-up logits",
        &ordinary_followup,
        &hidden_followup,
    );
    assert_eq!(ordinary.session.kv_cache.seq_len, tokens.len() + 1);
    assert_eq!(with_hidden.session.kv_cache.seq_len, tokens.len() + 1);
}

/// #1708: `forward_step_decode` is the per-token step behind
/// `generate`/`generate_streaming*` — the main autoregressive decode
/// path. It only ever advances `kv_cache.seq_len`; with `position()`
/// derived from that same field rather than tracked separately, this
/// path can no longer leave the decode cursor stale.
#[test]
fn forward_step_decode_keeps_position_synced_with_seq_len() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (forward_step_decode_keeps_position_synced_with_seq_len)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");

    let tokens = [42_u32, 2, 5];
    let _ = state.forward_prefill(&tokens);
    assert_eq!(state.session.position(), tokens.len());
    assert_eq!(state.session.kv_cache.seq_len, tokens.len());

    let pos = state.session.kv_cache.seq_len;
    let _ = state.forward_step_decode(7, pos);

    assert_eq!(
        state.session.kv_cache.seq_len,
        tokens.len() + 1,
        "forward_step_decode must advance the live cache cursor"
    );
    assert_eq!(
        state.session.position(),
        state.session.kv_cache.seq_len,
        "forward_step_decode is the main decode path (#1708): position must \
                 stay in sync with kv_cache.seq_len rather than go stale"
    );
}

/// `tiny_metal_qwen35_fixture` (used by the tests above) has zero
/// GDN/linear-attention layers — `has_gdn_layers()` is false for it — so
/// neither proves explicit hidden readback works on a hybrid GDN+full
/// session. This test mirrors their structure on `tiny_hybrid_fixture`
/// (3 GDN layers + 1 full-attention layer) to close that gap.
#[test]
fn forward_step_with_hidden_matches_logits_on_hybrid_gdn_state_and_proves_explicit_readback() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=forward_step_with_hidden_matches_logits_on_hybrid_gdn_state_and_proves_explicit_readback \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (forward_step_with_hidden_matches_logits_on_hybrid_gdn_state_and_proves_explicit_readback)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut ordinary = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("ordinary tiny hybrid MetalQwen35State fixture constructs");
    let mut with_hidden = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("hidden tiny hybrid MetalQwen35State fixture constructs");
    assert!(
        with_hidden.has_gdn_layers(),
        "forward_step_with_hidden_matches_logits_on_hybrid_gdn_state_and_proves_explicit_readback's \
                 tiny_hybrid_fixture must retain its GDN layers for this test to exercise the \
                 GDN path"
    );
    ordinary.path_proof_enabled = true;
    with_hidden.path_proof_enabled = true;
    ordinary.reset_path_proof_counters();
    with_hidden.reset_path_proof_counters();

    let ordinary_logits = ordinary.forward_step(3, 0);
    let (hidden_logits, hidden) = with_hidden
        .forward_step_with_hidden(3, 0)
        .expect("hidden-returning step on a fresh hybrid/GDN session succeeds");
    assert_forward_rows_close("hybrid step logits", &ordinary_logits, &hidden_logits);
    assert_eq!(ordinary.session.kv_cache.seq_len, 1);
    assert_eq!(with_hidden.session.kv_cache.seq_len, 1);
    assert_eq!(hidden.len(), cfg.hidden_size);
    assert!(hidden.iter().all(|value| value.is_finite()));
    assert!(hidden.iter().any(|&value| value != 0.0));
    assert_forward_rows_close(
        "hybrid returned and session hidden",
        &hidden,
        &with_hidden.session.last_pre_final_hidden,
    );
    assert!(
        ordinary
            .session
            .last_pre_final_hidden
            .iter()
            .all(|&value| value == 0.0),
        "ordinary hybrid step must not add an unconditional hidden readback"
    );

    let ordinary_hidden_proof = ordinary.hidden_readback_path_proof_snapshot();
    let explicit_hidden_proof = with_hidden.hidden_readback_path_proof_snapshot();
    assert_eq!(ordinary_hidden_proof.decode, 0);
    assert_eq!(explicit_hidden_proof.decode, 1);

    let ordinary_path = ordinary.path_proof_snapshot();
    let explicit_path = with_hidden.path_proof_snapshot();
    assert_eq!(ordinary_path.decode_kv_copy, explicit_path.decode_kv_copy);
    assert!(explicit_path.decode_kv_copy > 0);
    assert_eq!(
        ordinary_path.decode_attn_direct,
        explicit_path.decode_attn_direct
    );
    assert!(explicit_path.decode_attn_direct > 0);
}

type CrossTurnCacheSnapshot = (
    crate::kv_cache::CrossTurnPrefixEntry,
    crate::attention::gdn::GdnSnapshot,
    Vec<(usize, crate::attention::gdn::GdnSnapshot)>,
);

fn snapshot_cross_turn_cache(state: &MetalQwen35State) -> Option<CrossTurnCacheSnapshot> {
    state.cross_turn_prefix_cache.entry.as_ref().map(|entry| {
        (
            entry.generic.clone(),
            entry.gdn_snapshot.clone(),
            entry
                .checkpoints
                .iter()
                .map(|checkpoint| (checkpoint.len, checkpoint.snapshot.clone()))
                .collect(),
        )
    })
}

fn mtp_loaded_live_hybrid_state_for_guard_test() -> MetalQwen35State {
    use crate::speculative::MtpTargetVerifier as _;

    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let mut state = metal_state_with_synthetic_mtp_for_test(&weights, &cfg, None);
    assert!(state.has_mtp(), "fixture must load MTP state");
    state.path_proof_enabled = true;

    let mut seeded_gdn = state.snapshot_gdn_states();
    assert_eq!(seeded_gdn.len(), 3, "fixture must contain three GDN layers");
    seeded_gdn[0].0[0] = 1.0;
    seeded_gdn[0].1[0] = -1.0;
    state.restore_gdn_states(&seeded_gdn);

    state
        .try_forward_step(1, 0)
        .expect("first raw step at the live cursor must succeed");
    assert_eq!(state.session.kv_cache.seq_len, 1);
    assert!(
        state.has_gdn_layers(),
        "mtp_loaded_live_hybrid_state_for_guard_test's tiny_hybrid_fixture \
                 must retain its GDN layers for gdn_state_is_initial() to be meaningful"
    );
    assert!(!state.gdn_state_is_initial());
    assert!(
        !state.session.last_pre_final_hidden.is_empty(),
        "MTP-loaded raw step must populate hidden readback state"
    );
    state.reset_path_proof_counters();
    state
}

#[test]
fn raw_step_with_mtp_rejects_stale_live_cursor_without_mutation() {
    use crate::speculative::MtpTargetVerifier as _;

    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=raw_step_with_mtp_rejects_stale_live_cursor_without_mutation \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (raw_step_with_mtp_rejects_stale_live_cursor_without_mutation)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32)
        .expect("tiny hybrid MetalQwen35State fixture constructs");
    state.path_proof_enabled = true;
    state
        .generate_streaming_with_prefix_cache(
            crate::kv_cache::CrossTurnSlotId::DEFAULT,
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(9, 2),
            |_, _| true,
        )
        .expect("cache-aware warm-up must establish live state");
    assert!(state.session.kv_cache.seq_len > 0);
    assert!(state.cross_turn_prefix_cache.entry.is_some());
    // tiny_hybrid_fixture's in_proj_qkv and conv1d_weight are both zero, so the
    // conv1d output (crates/inference/src/attention/gdn.rs: apply_causal_conv1d,
    // `sum += conv_buffer[..] * conv_weight[..]` / `sum += new_input[ch] *
    // conv_weight[..]`) is a product of a zero factor at every step — either
    // factor alone zeroing it, both would have to go nonzero together to move it
    // off zero — so q/k/v are provably zero at every step regardless of the other
    // fixture values. With v == kv_mem == 0 (kv_mem = 0 because the recurrent
    // state S starts at zero and only ever receives outer(k, delta) with
    // delta == 0), `gated_delta_net_step`'s update (gdn.rs: decay `s *= g` at
    // ~L291-294, then `S += outer(k, delta)` at ~L313-318) is homogeneous: g
    // scales an already-zero S, and the injection term is zero too, so S cannot
    // leave zero however many tokens are generated. This fixture does set
    // a_log = -1.0, dt_bias = 0.0, and norm_weight = 1.0 (nonzero,
    // zero-in-this-fixture, and nonzero respectively) — all three sit on the path
    // (compute_decay_gate uses a_log and dt_bias for g; norm_weight scales the
    // final gated-RMSNorm output) but the zero-state argument above never
    // depends on their values, only on v and kv_mem being zero. The Metal shader
    // (forward/shaders/qwen35.metal: gdn_recurrence_fused, ~L1099-1121) computes
    // the same recurrence in reordered form (`kv_mem` from pre-decay S, then
    // `delta = (v - g * kv_mem) * beta`, then `sr[i] = fma(k, delta, g * sr[i])`)
    // and agrees algebraically with the CPU path above. Falsifier: this reasoning
    // breaks the moment `make_gdn` sets *both* in_proj_qkv and conv1d_weight
    // nonzero at once — that is the only way ordinary decoding can move q/k/v
    // (and hence the recurrent state) off zero. Until then, seed live GDN state
    // directly instead, the same way `mtp_loaded_live_hybrid_state_for_guard_test`
    // does, so the immutability assertions below exercise genuinely non-initial
    // state rather than a vacuously-true reading of `gdn_state_is_initial()`.
    assert!(
        state.has_gdn_layers(),
        "fixture must contain GDN layers for this predicate to be meaningful"
    );
    let mut seeded_gdn = state.snapshot_gdn_states();
    seeded_gdn[0].0[0] = 1.0;
    state.restore_gdn_states(&seeded_gdn);
    assert!(!state.gdn_state_is_initial());
    state.reset_path_proof_counters();

    let seq_len_before = state.session.kv_cache.seq_len;
    let position_before = state.session.position();
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = snapshot_gdn_bytes(&state);
    let hidden_before = state.session.last_pre_final_hidden.clone();
    let hidden_proof_before = state.hidden_readback_path_proof_snapshot();
    let cross_turn_before = snapshot_cross_turn_cache(&state);

    let error = state
        .try_forward_step(2, 0)
        .expect_err("an in-range stale raw step must be rejected");
    let crate::error::InferenceError::InvalidInput(message) = error else {
        panic!("expected InvalidInput for a stale raw step, got {error}");
    };
    assert!(message.contains("try_forward_step"), "{message}");
    assert_eq!(state.session.kv_cache.seq_len, seq_len_before);
    assert_eq!(state.session.position(), position_before);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);
    assert_eq!(state.session.last_pre_final_hidden, hidden_before);
    assert_eq!(snapshot_cross_turn_cache(&state), cross_turn_before);
    assert_eq!(
        state.hidden_readback_path_proof_snapshot(),
        hidden_proof_before
    );

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.forward_step(2, 0)));
    assert!(
        panic.is_err(),
        "infallible raw step wrapper must reject the same stale cursor"
    );
    assert_eq!(state.session.kv_cache.seq_len, seq_len_before);
    assert_eq!(state.session.position(), position_before);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);
    assert_eq!(state.session.last_pre_final_hidden, hidden_before);
    assert_eq!(
        state.hidden_readback_path_proof_snapshot(),
        hidden_proof_before
    );
}

#[test]
fn raw_prefills_with_mtp_reject_live_session_without_mutation() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=raw_prefills_with_mtp_reject_live_session_without_mutation \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (raw_prefills_with_mtp_reject_live_session_without_mutation)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let mut state = mtp_loaded_live_hybrid_state_for_guard_test();

    let seq_len_before = state.session.kv_cache.seq_len;
    let position_before = state.session.position();
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = snapshot_gdn_bytes(&state);
    let hidden_before = state.session.last_pre_final_hidden.clone();
    let hidden_proof_before = state.hidden_readback_path_proof_snapshot();

    let error = state
        .try_forward_prefill(&[2, 3])
        .expect_err("fallible raw prefill must reject a live session");
    let crate::error::InferenceError::InvalidInput(message) = error else {
        panic!("expected InvalidInput for a live-session raw prefill, got {error}");
    };
    assert!(message.contains("try_forward_prefill"), "{message}");
    assert_eq!(state.session.kv_cache.seq_len, seq_len_before);
    assert_eq!(state.session.position(), position_before);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);
    assert_eq!(state.session.last_pre_final_hidden, hidden_before);
    assert_eq!(
        state.hidden_readback_path_proof_snapshot(),
        hidden_proof_before
    );

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.forward_prefill(&[2, 3])
    }));
    assert!(panic.is_err(), "raw prefill must reject a live session");
    assert_eq!(state.session.kv_cache.seq_len, seq_len_before);
    assert_eq!(state.session.position(), position_before);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);
    assert_eq!(state.session.last_pre_final_hidden, hidden_before);
    assert_eq!(
        state.hidden_readback_path_proof_snapshot(),
        hidden_proof_before
    );

    let error = state
        .forward_prefill_all_logits(&[2, 3])
        .expect_err("all-logits raw prefill must reject a live session");
    let crate::error::InferenceError::InvalidInput(message) = error else {
        panic!("expected InvalidInput for a live-session raw prefill, got {error}");
    };
    assert!(message.contains("forward_prefill_all_logits"), "{message}");
    assert_eq!(state.session.kv_cache.seq_len, seq_len_before);
    assert_eq!(state.session.position(), position_before);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);
    assert_eq!(state.session.last_pre_final_hidden, hidden_before);
    assert_eq!(
        state.hidden_readback_path_proof_snapshot(),
        hidden_proof_before
    );
}

#[cfg(feature = "bench-internals")]
#[test]
fn bench_prefill_helpers_reject_stale_cursor_without_mutation() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=bench_prefill_helpers_reject_stale_cursor_without_mutation \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (bench_prefill_helpers_reject_stale_cursor_without_mutation)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let mut state = mtp_loaded_live_hybrid_state_for_guard_test();

    let seq_len_before = state.session.kv_cache.seq_len;
    let position_before = state.session.position();
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = snapshot_gdn_bytes(&state);
    let hidden_before = state.session.last_pre_final_hidden.clone();
    let hidden_proof_before = state.hidden_readback_path_proof_snapshot();

    let isolated_error = bench_support::forward_prefill_gdn_isolated_chunk(&mut state, &[2, 3], 0)
        .expect_err("isolated bench prefill must reject a stale cursor");
    assert!(isolated_error.to_string().contains("isolated"));

    let capture_error =
        bench_support::forward_prefill_gdn_isolated_chunk_capture(&mut state, &[2, 3], 0, &[])
            .expect_err("capture bench prefill must reject a stale cursor");
    assert!(capture_error.to_string().contains("capture"));

    let production_error = bench_support::forward_prefill_production_chunk(&mut state, &[2, 3], 0)
        .expect_err("production bench prefill must reject a stale cursor");
    assert!(production_error.to_string().contains("production"));

    assert_eq!(state.session.kv_cache.seq_len, seq_len_before);
    assert_eq!(state.session.position(), position_before);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);
    assert_eq!(state.session.last_pre_final_hidden, hidden_before);
    assert_eq!(
        state.hidden_readback_path_proof_snapshot(),
        hidden_proof_before
    );
}

#[test]
fn forward_step_with_hidden_rejects_position_mismatched_with_cache_cursor() {
    use crate::speculative::MtpTargetVerifier as _;

    let Some(_) = Device::system_default() else {
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");
    state.path_proof_enabled = true;
    state.reset_path_proof_counters();

    // Defect-1 repro: on a fresh state (cursor at 0), a caller asks for
    // position 5. The old guard only checked `position < max_cache_len`
    // and would have RoPE-rotated token_a for position 5 while writing
    // its KV row at row 0 (derived from kv_cache.seq_len). It must now
    // be rejected before any dispatch.
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = state.snapshot_gdn_states();
    let err = state
        .forward_step_with_hidden(42, 5)
        .expect_err("position ahead of the live cache cursor must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(
        state.session.kv_cache.seq_len, 0,
        "rejection must not advance the KV cache cursor"
    );
    assert_eq!(
        state.session.position(),
        0,
        "rejection must not advance the decode cursor"
    );
    assert_eq!(
        state.hidden_readback_path_proof_snapshot().decode,
        0,
        "rejection must not read hidden state"
    );
    assert_eq!(
        snapshot_kv_bytes(&state),
        kv_before,
        "rejection must not write any KV row"
    );
    assert_eq!(
        state.snapshot_gdn_states(),
        gdn_before,
        "rejection must not mutate GDN recurrent state"
    );

    // The correctly-ordered sequence (position == cursor each call)
    // still succeeds and advances both cursors together.
    let (_, hidden_a) = state
        .forward_step_with_hidden(42, 0)
        .expect("position matching the live cache cursor succeeds");
    assert_eq!(state.session.kv_cache.seq_len, 1);
    assert_eq!(state.session.position(), 1);
    assert!(hidden_a.iter().any(|&value| value != 0.0));

    // Defect-1 repro, second call: token_b at position 1 is the correct
    // next position now, but a stale/out-of-order position must still
    // be rejected rather than silently accepted.
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = state.snapshot_gdn_states();
    let err = state
        .forward_step_with_hidden(2, 0)
        .expect_err("a stale position behind the live cache cursor must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(state.session.kv_cache.seq_len, 1);
    assert_eq!(state.session.position(), 1);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(state.snapshot_gdn_states(), gdn_before);

    let (_, hidden_b) = state
        .forward_step_with_hidden(2, 1)
        .expect("position matching the live cache cursor succeeds");
    assert_eq!(state.session.kv_cache.seq_len, 2);
    assert_eq!(state.session.position(), 2);
    assert!(hidden_b.iter().any(|&value| value != 0.0));
}

#[test]
fn try_forward_step_keeps_session_position_in_sync_with_kv_cache_cursor() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");

    assert_eq!(state.session.position(), 0);
    assert_eq!(state.session.kv_cache.seq_len, 0);

    // Drive several raw steps and assert the documented decode cursor
    // (`session.position()`) agrees with the live cache cursor
    // (`kv_cache.seq_len`) after every one of them, matching the
    // contract `forward_step_with_hidden` already keeps.
    for (position, token_id) in (0usize..3).zip([1u32, 2, 3]) {
        state
            .try_forward_step(token_id, position)
            .expect("position matching the live cache cursor succeeds");
        assert_eq!(
            state.session.kv_cache.seq_len,
            position + 1,
            "kv_cache.seq_len must advance by one token per step"
        );
        assert_eq!(
            state.session.position(),
            state.session.kv_cache.seq_len,
            "session.position() must track the live cache cursor after try_forward_step"
        );
    }
}

#[test]
fn try_forward_step_rejects_out_of_vocab_token_id() {
    use crate::speculative::MtpTargetVerifier as _;

    let Some(_) = Device::system_default() else {
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");

    // forward_step's doc promises try_forward_step returns a typed
    // input error instead of panicking on an out-of-vocab token_id;
    // this must be rejected before any dispatch, same as the sibling
    // forward_step_with_hidden guard above.
    let seq_len_before = state.session.kv_cache.seq_len;
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = state.snapshot_gdn_states();
    let err = state
        .try_forward_step(cfg.vocab_size as u32, 0)
        .expect_err("out-of-vocab token_id must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(
        state.session.kv_cache.seq_len, seq_len_before,
        "rejection must not advance the KV cache cursor"
    );
    assert_eq!(
        snapshot_kv_bytes(&state),
        kv_before,
        "rejection must not write any KV row"
    );
    assert_eq!(
        state.snapshot_gdn_states(),
        gdn_before,
        "rejection must not mutate GDN recurrent state"
    );
}

#[test]
fn mtp_verifier_boundaries_reject_mismatched_live_cursor() {
    use crate::speculative::MtpTargetVerifier as _;

    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=mtp_verifier_boundaries_reject_mismatched_live_cursor \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (mtp_verifier_boundaries_reject_mismatched_live_cursor)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny hybrid MetalQwen35State fixture constructs");
    state.path_proof_enabled = true;
    state.reset_path_proof_counters();
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = snapshot_gdn_bytes(&state);

    let error = state
        .verify_tokens(&[1], 1)
        .expect_err("public verifier must reject start_pos beyond the live cursor");
    let crate::error::InferenceError::InvalidInput(message) = error else {
        panic!("expected InvalidInput for mismatched public verifier cursor, got {error}");
    };
    assert!(
        message.contains("MtpTargetVerifier::verify_tokens"),
        "{message}"
    );
    assert_eq!(state.session.kv_cache.seq_len, 0);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);

    let error = state
        .verify_tokens_batched(&[1], 1)
        .err()
        .expect("sequential verifier must independently reject a cursor mismatch");
    let crate::error::InferenceError::InvalidInput(message) = error else {
        panic!("expected InvalidInput for mismatched sequential verifier cursor, got {error}");
    };
    assert!(message.contains("verify_tokens_batched"), "{message}");
    assert_eq!(state.session.kv_cache.seq_len, 0);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);

    let error = state
        .verify_tokens_batch_gemm(&[1], 1)
        .err()
        .expect("batch-GEMM verifier must independently reject a cursor mismatch");
    let crate::error::InferenceError::InvalidInput(message) = error else {
        panic!("expected InvalidInput for mismatched batch-GEMM verifier cursor, got {error}");
    };
    assert!(message.contains("verify_tokens_batch_gemm"), "{message}");
    assert_eq!(state.session.kv_cache.seq_len, 0);
    assert_eq!(snapshot_kv_bytes(&state), kv_before);
    assert_eq!(snapshot_gdn_bytes(&state), gdn_before);
    assert_eq!(
        state.hidden_readback_path_proof_snapshot(),
        HiddenReadbackPathProofSnapshot::default(),
        "cursor rejection must occur before any hidden readback"
    );
}

#[test]
fn forward_prefill_with_hidden_rejects_live_session_without_reset() {
    use crate::speculative::MtpTargetVerifier as _;

    let Some(_) = Device::system_default() else {
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");
    state.path_proof_enabled = true;

    // Establish live state: one token advances the KV cache cursor off
    // zero. (`tiny_metal_qwen35_fixture` has no linear-attention layers, so
    // `has_gdn_layers()` is false and the fresh-session gate's GDN conjunct
    // is vacuously satisfied regardless of activity here; this test
    // exercises the `kv_cache.seq_len` half of the gate, which alone is
    // already sufficient to reject this repro.)
    state
        .forward_step_with_hidden(42, 0)
        .expect("first token at position 0 succeeds");
    assert_eq!(state.session.kv_cache.seq_len, 1);

    state.reset_path_proof_counters();

    // Defect-2 repro: forward_prefill_with_hidden always dispatches from
    // position 0. Without a reset, the old code would overwrite KV row
    // 0 (and row 1) while leaving GDN recurrent state still conditioned
    // on the token it just overwrote — an unreconstructible mix. It
    // must now be rejected before any dispatch.
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = state.snapshot_gdn_states();
    let err = state
        .forward_prefill_with_hidden(&[2, 5])
        .expect_err("prefill against a live session must be rejected without reset_state()");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(
        state.session.kv_cache.seq_len, 1,
        "rejection must not overwrite the live KV prefix"
    );
    assert_eq!(
        state.hidden_readback_path_proof_snapshot().prefill,
        0,
        "rejection must not read hidden state"
    );
    assert_eq!(
        snapshot_kv_bytes(&state),
        kv_before,
        "rejection must not write any KV row"
    );
    assert_eq!(
        state.snapshot_gdn_states(),
        gdn_before,
        "rejection must not mutate GDN recurrent state"
    );

    // reset_state() clears both gates; the same call now succeeds.
    state.reset_state();
    assert_eq!(state.session.kv_cache.seq_len, 0);
    // tiny_metal_qwen35_fixture has no linear-attention layers, so
    // gdn_state_is_initial() has no population to check and panics if
    // called (see its doc comment) — assert the population fact this test
    // actually relies on instead of the vacuous predicate.
    assert!(!state.has_gdn_layers());
    let (_, hidden) = state
        .forward_prefill_with_hidden(&[2, 5])
        .expect("prefill against a freshly reset session succeeds");
    assert_eq!(state.session.kv_cache.seq_len, 2);
    assert!(hidden.iter().any(|&value| value != 0.0));
}

#[test]
fn forward_prefill_with_hidden_rejects_noninitial_gdn_state_with_empty_kv() {
    use crate::speculative::MtpTargetVerifier as _;

    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=forward_prefill_with_hidden_rejects_noninitial_gdn_state_with_empty_kv \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (forward_prefill_with_hidden_rejects_noninitial_gdn_state_with_empty_kv)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny hybrid MetalQwen35State fixture constructs");
    state.path_proof_enabled = true;

    assert_eq!(state.session.kv_cache.seq_len, 0);
    assert!(
        state.has_gdn_layers(),
        "forward_prefill_with_hidden_rejects_noninitial_gdn_state_with_empty_kv's \
                 tiny_hybrid_fixture must retain its GDN layers for gdn_state_is_initial() \
                 to be meaningful"
    );
    assert!(state.gdn_state_is_initial());
    let mut seeded_gdn = state.snapshot_gdn_states();
    assert_eq!(seeded_gdn.len(), 3, "fixture must contain three GDN layers");
    seeded_gdn[0].0[0] = 1.0;
    state.restore_gdn_states(&seeded_gdn);
    let _ = state.forward_step_gdn_only(1, 0);
    assert_eq!(
        state.session.kv_cache.seq_len, 0,
        "GDN-only forward must leave the first fresh-session disjunct false"
    );
    assert!(
        state.has_gdn_layers(),
        "forward_prefill_with_hidden_rejects_noninitial_gdn_state_with_empty_kv's \
                 tiny_hybrid_fixture must retain its GDN layers for gdn_state_is_initial() \
                 to be meaningful"
    );
    assert!(
        !state.gdn_state_is_initial(),
        "GDN-only forward must leave live recurrent state non-initial"
    );

    state.reset_path_proof_counters();
    let kv_before = snapshot_kv_bytes(&state);
    let gdn_before = state.snapshot_gdn_states();
    let error = state
        .forward_prefill_with_hidden(&[2])
        .expect_err("non-initial GDN state with empty KV cache must be rejected");
    let crate::error::InferenceError::InvalidInput(message) = error else {
        panic!("expected InvalidInput for non-initial GDN state, got {error}");
    };
    assert!(
        message.contains("GDN recurrent state at its initial condition"),
        "error must name the rejected GDN condition: {message}"
    );
    assert!(
        message.contains("kv_cache.seq_len=0"),
        "error must prove the KV half of the guard was false: {message}"
    );
    assert_eq!(state.session.kv_cache.seq_len, 0);
    assert_eq!(state.session.position(), 0);
    assert_eq!(
        state.hidden_readback_path_proof_snapshot().prefill,
        0,
        "rejection must not read hidden state"
    );
    assert_eq!(
        snapshot_kv_bytes(&state),
        kv_before,
        "rejection must not write any KV row"
    );
    assert_eq!(
        state.snapshot_gdn_states(),
        gdn_before,
        "rejection must not mutate GDN recurrent state"
    );
}

/// Mutation control for the `has_gdn_layers()` / `gdn_state_is_initial()` split:
/// calling `gdn_state_is_initial()` on a session with zero GDN layers must panic
/// rather than silently returning `true` (the pre-fix behavior, which conflated
/// "no buffers to check" with "buffers checked and clean"). Removing the
/// `num_layers > 0` assertion in `gdn_state_is_initial()` (letting it return `true`
/// for the empty case instead) makes this test fail on the missing panic; restoring
/// the assertion makes it pass again.
#[test]
fn gdn_state_is_initial_panics_on_empty_population() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=gdn_state_is_initial_panics_on_empty_population \
                     reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                     (gdn_state_is_initial_panics_on_empty_population)"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");
    assert!(
        !state.has_gdn_layers(),
        "fixture must have zero GDN layers for this control"
    );
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.gdn_state_is_initial()
    }))
    .is_err();
    assert!(
        panicked,
        "gdn_state_is_initial() must panic on an empty GDN population instead of \
                 silently returning true"
    );
}

#[test]
fn forward_step_rejects_position_and_cache_capacity_before_dispatch() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 1).expect("tiny MetalQwen35State fixture constructs");

    let err = state
        .try_forward_step(1, 1)
        .expect_err("position at max_cache_len must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(state.session.kv_cache.seq_len, 0);

    let err = state
        .forward_prefill_from(&[1], 1, false)
        .expect_err("one-token suffix at max_cache_len must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(state.session.kv_cache.seq_len, 0);

    state.session.kv_cache.seq_len = 1;
    let err = state
        .try_forward_step(1, 0)
        .expect_err("a full KV cache must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(state.session.kv_cache.seq_len, 1);
}

#[test]
fn verify_tokens_rejects_out_of_range_start_pos_before_dispatch() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    use crate::speculative::MtpTargetVerifier as _;

    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 1).expect("tiny MetalQwen35State fixture constructs");

    // `verify_tokens` is the public `MtpTargetVerifier` entry point (reachable by
    // any consumer holding `&mut MetalQwen35State`). A `start_pos` at
    // `max_cache_len` must be rejected before any GDN checkpoint or Metal
    // dispatch — including before `verify_tokens_batched`'s own "no checkpoint
    // pool" check, which this fixture never allocates, so a passing result here
    // cannot be explained by that unrelated early return.
    let err = state
        .verify_tokens(&[1], 1)
        .expect_err("start_pos at max_cache_len must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(
        state.session.kv_cache.seq_len, 0,
        "no dispatch must have advanced the KV cache"
    );

    // A batch whose last token would land at/after max_cache_len must also be
    // rejected even when start_pos itself is in range.
    let err = state
        .verify_tokens(&[1, 2], 0)
        .expect_err("token batch spilling past max_cache_len must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(state.session.kv_cache.seq_len, 0);
}

/// Pure test of the centralized capacity validator — no `Device`, no session,
/// runs on CPU. Guards the blocking gap: `verify_tokens_batched` checkpoints one
/// GDN pool slot per token (slot 0 = base, slot k = after k tokens), so a batch
/// larger than the pool's `max_tokens` must be rejected even when the KV/RoPE
/// cache has ample room — otherwise `checkpoint_gdn_to_slot` indexes past the
/// pool's allocated slots mid-batch.
#[test]
fn validate_dispatch_capacity_rejects_gdn_pool_overflow_independent_of_kv_capacity() {
    // In-bounds: 5 tokens exactly fill a 5-slot GDN pool, KV cache has room.
    assert!(
        MetalQwen35State::validate_dispatch_capacity(0, 5, 10, Some(5)).is_ok(),
        "5 tokens against a 5-token GDN pool and 10-slot KV cache must fit"
    );

    // KV/RoPE overflow still rejects (unchanged behavior from the prior round).
    let err = MetalQwen35State::validate_dispatch_capacity(8, 3, 10, Some(5))
        .expect_err("start_pos 8 + 3 tokens reaches position 10, at kv_capacity 10");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));

    // GDN pool overflow: 6 tokens against a 5-token pool must be rejected even
    // though the KV cache (100) has plenty of room — this is the blocking gap.
    let err = MetalQwen35State::validate_dispatch_capacity(0, 6, 100, Some(5))
        .expect_err("6 tokens against a 5-token GDN pool must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));

    // Callers that never checkpoint per-token (batch-GEMM, prefill) pass
    // `gdn_pool_capacity = None` and must not be bounded by the pool at all.
    assert!(
        MetalQwen35State::validate_dispatch_capacity(0, 6, 100, None).is_ok(),
        "a caller that never checkpoints per-token must not be pool-bounded"
    );
}

#[test]
fn validate_live_cursor_rejects_mismatched_position() {
    assert!(MetalQwen35State::validate_live_cursor("test", 3, 3).is_ok());
    let error = MetalQwen35State::validate_live_cursor("test", 4, 3)
        .expect_err("position beyond the live cursor must be rejected");
    let crate::error::InferenceError::InvalidInput(message) = error else {
        panic!("expected InvalidInput for mismatched cursor, got {error}");
    };
    assert!(message.contains("test"), "{message}");
    assert!(message.contains("supplied position 4"), "{message}");
    assert!(message.contains("live cache cursor 3"), "{message}");
}

#[test]
fn validate_hidden_prefill_fresh_session_checks_kv_and_gdn_state() {
    assert!(MetalQwen35State::validate_hidden_prefill_fresh_session(0, true).is_ok());
    assert!(
        MetalQwen35State::validate_hidden_prefill_fresh_session(1, true).is_err(),
        "a live KV cursor must be rejected"
    );
    assert!(
        MetalQwen35State::validate_hidden_prefill_fresh_session(0, false).is_err(),
        "non-initial GDN state must be rejected even when the KV cursor is zero"
    );
}

/// Deferred-run regression: full round-trip through the public
/// `MtpTargetVerifier::verify_tokens` entry point on a GDN-backed
/// `MetalQwen35State`, asserting a batch larger than the GDN checkpoint pool is
/// rejected before any dispatch or checkpoint mutation. Compiled under
/// `--features metal-gpu` (this crate's GDN checkpoint pool only exists in that
/// configuration) but NOT executed this session — GPU test runs are out of
/// scope tonight. Run on the next GPU pass; the CPU-only
/// `validate_dispatch_capacity_rejects_gdn_pool_overflow_independent_of_kv_capacity`
/// test above already proves the same guard logic without a device.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn verify_tokens_rejects_batch_exceeding_gdn_pool_capacity_before_dispatch() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    use crate::speculative::MtpTargetVerifier as _;

    let mut state = with_self_spec_env(|| {
        let (cfg, weights) = tiny_hybrid_fixture();
        MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture")
    });

    let pool_capacity = state
        .session
        .gdn_checkpoints
        .as_ref()
        .expect("self-spec pool must be allocated")
        .max_tokens;
    // One more token than the pool holds: `checkpoint_gdn_to_slot` would index
    // past the pool's allocated slots partway through this batch if dispatched.
    let too_many: Vec<u32> = (0..(pool_capacity as u32 + 1)).map(|i| i % 2).collect();

    let err = state
        .verify_tokens(&too_many, 0)
        .expect_err("a batch larger than the GDN checkpoint pool must be rejected");
    assert!(matches!(err, crate::error::InferenceError::InvalidInput(_)));
    assert_eq!(
        state.session.kv_cache.seq_len, 0,
        "rejected preflight must leave state unmutated — no dispatch, no checkpoint \
                 mutation"
    );
}
