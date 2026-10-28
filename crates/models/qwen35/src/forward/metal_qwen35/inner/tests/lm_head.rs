//! lm_head block top-k, argmax and real-checkpoint agreement.

use super::*;
// ===================================================================
// Mutation-sensitive correctness tests for
// the lm_head two-stage block-top-k path (issue #171).
// ===================================================================

/// Same construction as `tiny_metal_qwen35_fixture` but with a
/// caller-chosen vocab size and an all-zero embedding table (only
/// component 0 of a token's embedding is ever set below — the same
/// tied-lm_head one-hot trick `tiny_metal_qwen35_fixture` and the
/// golden-logit-snapshot test rely on), so individual vocab rows'
/// logits can be hand-crafted for adversarial distributions.
fn zeroed_block_topk_fixture(vocab: usize) -> (Qwen35Config, ModelWeights) {
    let hidden = 512usize;
    let intermediate = 64usize;
    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 1,
        vocab_size: vocab,
        intermediate_size: intermediate,
        rms_norm_eps: 1e-6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_parameters: None,
        linear_num_key_heads: 1,
        linear_num_value_heads: Some(1),
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        full_attention_interval: 1,
        layer_types: vec![LayerType::FullAttention],
        layer_mask: vec![true],
        eos_token_id: (vocab - 1) as u32,
        max_position_embeddings: 128,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let embed_tokens = vec![0.0f32; vocab * hidden];
    let common = CommonLayerWeights {
        input_layernorm: vec![1.0; hidden],
        post_attention_layernorm: vec![1.0; hidden],
        ffn: FeedForwardWeights::Dense(DenseFfnWeights {
            gate_proj: vec![0.0; intermediate * hidden],
            up_proj: vec![0.0; intermediate * hidden],
            down_proj: vec![0.0; hidden * intermediate],
        }),
    };
    let full = FullAttentionLayerWeights {
        q_proj: vec![0.0; 2 * cfg.full_q_dim() * hidden],
        k_proj: vec![0.0; cfg.full_kv_dim() * hidden],
        v_proj: vec![0.0; cfg.full_kv_dim() * hidden],
        o_proj: vec![0.0; hidden * cfg.full_q_dim()],
        q_norm: vec![1.0; cfg.head_dim],
        k_norm: vec![1.0; cfg.head_dim],
    };
    let weights = ModelWeights {
        embed_tokens,
        lm_head: None,
        final_norm: vec![1.0; hidden],
        layers: vec![(AttentionWeights::Full(full), common)],
    };
    (cfg, weights)
}

/// Adversarial distribution: every
/// global top-K token clusters inside ONE Stage-1 tile. Proves Stage 1
/// emits a true per-tile top-`local_k`, not merely a per-tile argmax
/// or an approximate/truncated local ranking — a distribution where
/// all global winners collide in one tile silently drops items under
/// either bug, per the exactness invariant documented at the Stage-1
/// top-k kernel.
///
/// MUTATION-SENSITIVE: if Stage 1 emitted only a block-local argmax
/// (ignoring `local_k`) or a truncated/approximate local top-k, this
/// test's GPU output would be missing entries the CPU oracle has, and
/// the set-equality assertion below would fail.
#[test]
fn lm_head_block_topk_clustered_single_tile_set_agreement() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    const ROWS_PER_TG: usize = 256;
    let vocab = ROWS_PER_TG * 5; // 5 exact tiles, no tail tile
    let target_tile = 3usize;
    let base = target_tile * ROWS_PER_TG;

    let (cfg, mut weights) = zeroed_block_topk_fixture(vocab);
    let hidden = cfg.hidden_size;

    // Input token 0 drives x[0] = -1.0 through the (all-zero-weight)
    // single layer, so the post-final-norm hidden is one-hot at
    // component 0 with a fixed, deterministically-negative scalar
    // c = -1.0 * (1+gamma) / rms(x) (gamma=1.0, rms>0 => c<0).
    weights.embed_tokens[0] = -1.0; // token 0, component 0

    // All 256 rows of the target tile get strictly monotonic,
    // large-magnitude NEGATIVE embedding values (-300.0 up to -44.0)
    // so that, multiplied by the negative c above, they become the
    // largest POSITIVE logits in the whole vocabulary. Every other
    // row stays at logit exactly 0.0 (zero embedding).
    for i in 0..ROWS_PER_TG {
        weights.embed_tokens[(base + i) * hidden] = -(300.0 - i as f32);
    }

    let mut baseline = MetalQwen35State::new(&weights, &cfg, 8).expect("fixture state");
    let full_logits = baseline.forward_step(0, 0);
    assert_eq!(full_logits.len(), vocab);

    for &k in &[8usize, 16, 40, 64] {
        let expected: std::collections::HashSet<u32> =
            cpu_topk_token_ids(&full_logits, k).into_iter().collect();
        // Sanity: the adversarial construction must actually cluster
        // the true top-K inside the target tile, or this test proves
        // nothing about the exactness invariant.
        assert!(
            expected
                .iter()
                .all(|&t| (t as usize) >= base && (t as usize) < base + ROWS_PER_TG),
            "test construction bug: CPU oracle top-{k} is not fully clustered \
                     in tile {target_tile}"
        );

        let mut blocked = MetalQwen35State::new(&weights, &cfg, 8).expect("fixture state");
        blocked.session.compact_route = GpuTopkRoute::BlockTopK { local_k: k as u32 };
        blocked.session.compact_topk = k;
        let empty = blocked.forward_step(0, 0);
        assert!(
            empty.is_empty(),
            "compact route must return empty logits (k={k})"
        );
        assert_eq!(
            blocked.session.compact_result.len(),
            k,
            "BlockTopK{{local_k:{k}}} must produce exactly {k} candidates"
        );
        let got: std::collections::HashSet<u32> = blocked
            .session
            .compact_result
            .iter()
            .map(|c| c.token_id)
            .collect();
        assert_eq!(
            got, expected,
            "clustered-tile top-{k} SET mismatch: GPU dropped or added tokens vs \
                     CPU oracle (tile={target_tile}, base={base}) — Stage-1 local top-k \
                     is not exact"
        );
    }
}

/// Design-spec dispatch-geometry requirement: "the known bug class is
/// under-dispatch leaving rows unwritten." Uses a vocab size NOT
/// divisible by `LM_HEAD_ROWS_PER_TG` (256) so the last Stage-1 tile
/// is partial (100 of 256 possible rows valid), and spikes the LAST
/// valid vocab row so it can only be found if the tail tile's
/// bounds-checked GEMV actually ran.
///
/// MUTATION-SENSITIVE: reverting `div_ceil(ROWS_PER_TG)` to plain
/// integer division, or removing the kernel's `row >= vocab_size`
/// guard, either leaves the tail tile's valid rows undispatched (the
/// winning row is never computed) or corrupts the tile — either way
/// this test's GPU argmax stops matching the CPU oracle's last-row
/// argmax.
#[test]
fn lm_head_block_argmax_tail_tile_boundary() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();

    const ROWS_PER_TG: usize = 256;
    let vocab = ROWS_PER_TG * 5 + 100; // 6 tiles; last tile has 100/256 valid rows
    let last_row = vocab - 1;

    let (cfg, mut weights) = zeroed_block_topk_fixture(vocab);
    let hidden = cfg.hidden_size;
    weights.embed_tokens[0] = -1.0; // input token 0, component 0 => c < 0
    weights.embed_tokens[last_row * hidden] = -400.0; // dominates every other row

    let mut baseline = MetalQwen35State::new(&weights, &cfg, 8).expect("fixture state");
    let full_logits = baseline.forward_step(0, 0);
    let expected_argmax = cpu_topk_token_ids(&full_logits, 1)[0];
    assert_eq!(
        expected_argmax as usize, last_row,
        "test construction bug: spiked last vocab row is not the CPU-oracle argmax"
    );

    let mut blocked = MetalQwen35State::new(&weights, &cfg, 8).expect("fixture state");
    blocked.session.compact_route = GpuTopkRoute::BlockArgmax;
    blocked.session.compact_topk = 1;
    let empty = blocked.forward_step(0, 0);
    assert!(empty.is_empty());
    assert_eq!(blocked.session.compact_result.len(), 1);
    assert_eq!(
        blocked.session.compact_result[0].token_id as usize,
        last_row,
        "tail-tile boundary row {last_row} (of vocab_size {vocab}, {} tiles, last \
                 tile partial) was not found by Stage-1/Stage-2 — dispatch geometry / \
                 tail guard bug",
        vocab.div_ceil(ROWS_PER_TG)
    );
}

/// Bypass-sensitive production-path test: poisons the persistent
/// logits buffer with a sentinel bit
/// pattern, runs the compact greedy route, and asserts the sentinel
/// survives untouched — proving the full `[vocab_size]` GEMV never
/// wrote to it, i.e. Stage 1/Stage 2 actually ran instead of the
/// full-logit path silently computing everything and discarding the
/// extra values.
///
/// MUTATION-SENSITIVE: disabling the Stage-1/2 early return in
/// `encode_final_head` (forcing every route through the full-logit
/// GEMV) flips both the sentinel-buffer assertion below AND the
/// `out.is_empty()` assertion.
#[test]
fn lm_head_stage1_bypass_is_detected_via_logits_buffer() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let vocab = cfg.vocab_size;

    let mut state = MetalQwen35State::new(&weights, &cfg, 8).expect("fixture state");

    let sentinel_bits: u32 = 0xDEAD_BEEF;
    // SAFETY: activations.logits is StorageModeShared f32[vocab_size];
    // no GPU work is in flight on this fresh session.
    unsafe {
        let ptr = state.session.activations.logits.contents() as *mut u32;
        for i in 0..vocab {
            std::ptr::write(ptr.add(i), sentinel_bits);
        }
    }

    state.session.compact_route = GpuTopkRoute::BlockArgmax;
    state.session.compact_topk = 1;
    let out = state.forward_step(3, 0);

    assert!(
        out.is_empty(),
        "compact greedy route must return empty logits, not the full vector — a \
                 non-empty result here means the full-logit path ran (stage-1 bypass)"
    );
    assert_eq!(state.session.compact_result.len(), 1);

    // SAFETY: same buffer; forward_step waits for GPU completion
    // synchronously before returning.
    unsafe {
        let ptr = state.session.activations.logits.contents() as *const u32;
        for i in 0..vocab {
            assert_eq!(
                std::ptr::read(ptr.add(i)),
                sentinel_bits,
                "logits buffer at vocab row {i} was overwritten — the full-logit \
                         GEMV ran despite compact_route=BlockArgmax (stage-1 bypass not \
                         caught by output alone)"
            );
        }
    }
}

/// Regression (issue #171): a compact top-k
/// request from one generation must not survive `reset_state()` into
/// the next generation on the same `MetalQwen35State`. Production
/// callers rely on this exact mechanism — every `generate_streaming`/
/// `generate`/cache-aware entry point calls `self.reset_state()` at
/// the top of every new generation, and a `use_compact == false`
/// decision (e.g. because the caller set `repetition_penalty != 1.0`,
/// which requires the full-logit exact path — see the route-decision
/// block's `else` arm a few hundred lines up) then simply never
/// re-engages the compact route for that call. If `reset_state`
/// failed to clear `compact_route`/`compact_topk`/`compact_result`,
/// that second generation would still see `compact_route ==
/// BlockArgmax` left over from the first, so `encode_final_head` /
/// `dispatch_lm_head_block_stage1_enc` would take the compact branch
/// again and return empty logits plus a stale one-candidate
/// `compact_result` instead of the full `[vocab_size]` logits a
/// non-compact (e.g. repetition-penalty) consumer requires —
/// sampling from that would read a stale/empty logits buffer.
///
/// Engages the compact route the same way
/// `lm_head_block_argmax_matches_full_logits` above does (direct
/// `session.compact_route`/`compact_topk` assignment + `forward_step`,
/// no env vars — the production route-decision block would set these
/// same two fields from `LATTICE_COMPACT_TOPK` + `top_k<=1`, but
/// setting them directly here isolates this test to the
/// `reset_state`/route-builder clearing bug rather than also
/// depending on route *selection*, which already has its own
/// coverage in `sampling_route_block_helper_gate`).
///
/// MUTATION-SENSITIVE: commenting out the `reset_state` compact-state
/// clear (the `self.session.compact_topk = 0; self.session.compact_route
/// = GpuTopkRoute::CpuFallback; self.session.compact_result.clear();`
/// block added at the end of `reset_state`) makes call 2 below return
/// empty logits instead of the full-vocab exact-path logits, failing
/// the `assert_eq!(logits.len(), vocab)` line. See the PR report for
/// the reverse-apply fail/pass proof.
#[test]
fn compact_state_does_not_survive_reset_state_into_next_generation() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let vocab = cfg.vocab_size;

    // Call 1: compact route engaged directly on a fresh state. This is
    // exactly the session state `encode_final_head` would leave behind
    // after a `generate_streaming` call with `LATTICE_COMPACT_TOPK` set
    // and `top_k<=1` ran the block-top-k path.
    let mut state = MetalQwen35State::new(&weights, &cfg, 8).expect("fixture state");
    state.session.compact_route = GpuTopkRoute::BlockArgmax;
    state.session.compact_topk = 1;
    let empty_logits = state.forward_step(3, 0);
    assert!(
        empty_logits.is_empty(),
        "test construction bug: call 1 did not take the compact route"
    );
    assert_eq!(state.session.compact_result.len(), 1);

    // Call 2: same state, `reset_state()` — exactly what every
    // `generate_*` entry point calls at the top of every new
    // generation — followed by a forward pass that never re-engages
    // the compact route (the production analogue: a caller-set
    // `repetition_penalty != 1.0` makes `use_compact == false`, so the
    // route-decision block's `else` arm runs instead of the `if
    // use_compact` arm and never sets `compact_route` back to a Block
    // route). `compact_route`/`compact_topk` are intentionally left
    // untouched by this test after `reset_state()` so the assertions
    // below observe exactly what `reset_state` itself did or didn't
    // clear.
    state.reset_state();
    let logits = state.forward_step(3, 0);
    assert_eq!(
        logits.len(),
        vocab,
        "call 2 must return full [vocab_size] logits via the exact path \
                 — a shorter/empty result means stale compact_route/compact_topk \
                 from call 1 survived reset_state and hijacked the exact-path \
                 forward pass"
    );
    assert_eq!(
        state.session.compact_route,
        GpuTopkRoute::CpuFallback,
        "reset_state must clear compact_route back to CpuFallback"
    );
    assert_eq!(
        state.session.compact_topk, 0,
        "reset_state must clear compact_topk"
    );
    assert!(
        state.session.compact_result.is_empty(),
        "reset_state must clear compact_result"
    );

    // Observable: a completely fresh state's exact-path forward_step
    // for the identical (token, position) must produce the same
    // logit values and the same greedy token — proving call 2 ran
    // the real full-logit GEMV, not a corrupted/stale variant. Uses
    // the same value-tolerance idiom as the argmax/topk8 smoke tests
    // above (bitonic-sort tie-break / dispatch-order noise is not a
    // bug) rather than requiring bit-for-bit equality.
    let mut fresh_state = MetalQwen35State::new(&weights, &cfg, 8).expect("fresh fixture state");
    let fresh_logits = fresh_state.forward_step(3, 0);
    assert_eq!(fresh_logits.len(), vocab);
    for (i, (&reused, &fresh)) in logits.iter().zip(fresh_logits.iter()).enumerate() {
        assert!(
            (reused - fresh).abs() < 1e-3,
            "logit mismatch at vocab row {i}: reused-state {reused} vs \
                     fresh-state {fresh} — call 1's compact state perturbed call \
                     2's full-logit computation"
        );
    }
    assert_eq!(
        cpu_topk_token_ids(&logits, 1)[0],
        cpu_topk_token_ids(&fresh_logits, 1)[0],
        "reused-state and fresh-state exact-path greedy token must match"
    );
}

/// Deterministic xorshift64* PRNG so the synthetic decode token
/// stream (and hence the test) is reproducible run-to-run (T.4)
/// without depending on real-tokenizer/corpus availability for a
/// 10K+-position sweep.
fn xorshift64_next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Acceptance test: greedy token agreement 100% vs. the full-logit path on
/// \>=10,000 positions using the real Qwen3.5-0.8B checkpoint.
/// Overridable via `LATTICE_TEST_GREEDY_POSITIONS` for fast local
/// iteration; defaults to 10,000 to satisfy the acceptance gate. KV
/// cache is reset every 64 positions on BOTH sessions identically to
/// keep per-step attention cost bounded (this test exercises lm_head
/// numerical correctness across many diverse real hidden states, not
/// long-context decode).
#[test]
fn lm_head_real_checkpoint_greedy_agreement_10k_positions() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let Some(model) = load_real_qwen35_0_8b_or_skip() else {
        return;
    };
    let _guard = gpu_test_lock();

    let num_positions: usize = std::env::var("LATTICE_TEST_GREEDY_POSITIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let reset_every = 64usize;
    let vocab = model.config().vocab_size;

    let mut baseline = MetalQwen35State::new(model.weights(), model.config(), 128)
        .expect("baseline real-checkpoint state");
    let mut blocked = MetalQwen35State::new(model.weights(), model.config(), 128)
        .expect("blocked real-checkpoint state");
    blocked.session.compact_route = GpuTopkRoute::BlockArgmax;
    blocked.session.compact_topk = 1;

    let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
    let mut accepted_ties = 0usize;
    let mut pos_in_window = 0usize;
    for step in 0..num_positions {
        if pos_in_window >= reset_every {
            baseline.reset_state();
            blocked.reset_state();
            // `reset_state` clears `compact_route`/`compact_topk` (issue #171
            // — a prior generation's compact request
            // must not leak into the next one). This test drives `forward_step`
            // directly rather than through `generate_streaming`'s route-decision
            // block, so it must re-apply the compact-route configuration itself
            // after every reset, exactly as a real caller starting a fresh
            // `generate_*` call after a KV-cache-clearing reset would via that
            // route-decision block.
            blocked.session.compact_route = GpuTopkRoute::BlockArgmax;
            blocked.session.compact_topk = 1;
            pos_in_window = 0;
        }
        let token = (xorshift64_next(&mut rng) % vocab as u64) as u32;

        let full_logits = baseline.forward_step(token, pos_in_window);
        let expected = cpu_topk_token_ids(&full_logits, 1)[0];

        let empty = blocked.forward_step(token, pos_in_window);
        assert!(empty.is_empty(), "compact route must return empty logits");
        assert_eq!(blocked.session.compact_result.len(), 1);
        let got = blocked.session.compact_result[0].token_id;

        // Greedy (k=1) is the SET agreement comparator specialized to a
        // singleton set: the full-logit GEMV (`baseline`, this test's CPU-side
        // oracle) and the compact Stage-1 block-argmax kernel (`blocked`)
        // accumulate the same ~1K f32 products in different reduction orders,
        // so a top-1/top-2 logit gap under LM_HEAD_TOPK_TIE_EPSILON is
        // legitimate boundary noise, not a correctness bug (issue #623).
        match topk_set_agreement_or_boundary_tie(
            &full_logits,
            &std::collections::HashSet::from([expected]),
            &std::collections::HashSet::from([got]),
            1,
            LM_HEAD_TOPK_TIE_EPSILON,
        ) {
            Ok(TopkSetAgreement::ExactSet) => {}
            Ok(TopkSetAgreement::AcceptedBoundaryTie {
                margin,
                epsilon,
                boundary_logit,
                differing_tokens,
            }) => {
                accepted_ties += 1;
                eprintln!(
                    "accepted greedy boundary tie at step {step} (pos {pos_in_window}): \
                             margin={margin:.6e}, epsilon={epsilon:.6e}, \
                             boundary_logit={boundary_logit:.6e}, \
                             differing_tokens={differing_tokens:?}"
                );
            }
            Err(mismatch) => {
                panic!("greedy mismatch at step {step} (pos {pos_in_window}): {mismatch:?}");
            }
        }
        pos_in_window += 1;
        if step % 2000 == 0 {
            eprintln!("greedy agreement progress: {step}/{num_positions}");
        }
    }

    eprintln!(
        "greedy agreement: {}/{num_positions} exact, {accepted_ties} accepted boundary ties",
        num_positions - accepted_ties
    );
}

/// Acceptance test: top-k SET agreement 100% for K in {8,16,40,64} vs. the
/// full-logit CPU oracle, on real checkpoint positions (the
/// clustered-tile adversarial case is covered separately by
/// `lm_head_block_topk_clustered_single_tile_set_agreement`).
/// Overridable via `LATTICE_TEST_TOPK_POSITIONS`.
#[test]
fn lm_head_real_checkpoint_topk_set_agreement() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let Some(model) = load_real_qwen35_0_8b_or_skip() else {
        return;
    };
    let _guard = gpu_test_lock();

    let num_positions: usize = std::env::var("LATTICE_TEST_TOPK_POSITIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let reset_every = 64usize;
    let vocab = model.config().vocab_size;
    let ks = [8usize, 16, 40, 64];

    let mut baseline = MetalQwen35State::new(model.weights(), model.config(), 128)
        .expect("baseline real-checkpoint state");
    let mut blocked_states: Vec<MetalQwen35State> = ks
        .iter()
        .map(|_| {
            MetalQwen35State::new(model.weights(), model.config(), 128)
                .expect("blocked real-checkpoint state")
        })
        .collect();
    for (i, &k) in ks.iter().enumerate() {
        blocked_states[i].session.compact_route = GpuTopkRoute::BlockTopK { local_k: k as u32 };
        blocked_states[i].session.compact_topk = k;
    }

    let mut rng: u64 = 0xFEED_FACE_C0FF_EE00;
    let mut accepted_ties = [0usize; 4];
    let mut pos_in_window = 0usize;
    for step in 0..num_positions {
        if pos_in_window >= reset_every {
            baseline.reset_state();
            // `reset_state` clears `compact_route`/`compact_topk` (issue #171).
            // This test drives `forward_step`
            // directly rather than through `generate_streaming`'s
            // route-decision block, so it must re-apply each state's
            // compact-route configuration after every reset, exactly as a
            // real caller starting a fresh `generate_*` call would via that
            // route-decision block.
            for (i, &k) in ks.iter().enumerate() {
                blocked_states[i].reset_state();
                blocked_states[i].session.compact_route =
                    GpuTopkRoute::BlockTopK { local_k: k as u32 };
                blocked_states[i].session.compact_topk = k;
            }
            pos_in_window = 0;
        }
        let token = (xorshift64_next(&mut rng) % vocab as u64) as u32;
        let full_logits = baseline.forward_step(token, pos_in_window);

        for (i, &k) in ks.iter().enumerate() {
            let expected: std::collections::HashSet<u32> =
                cpu_topk_token_ids(&full_logits, k).into_iter().collect();
            let empty = blocked_states[i].forward_step(token, pos_in_window);
            assert!(empty.is_empty());
            assert_eq!(blocked_states[i].session.compact_result.len(), k);
            let got: std::collections::HashSet<u32> = blocked_states[i]
                .session
                .compact_result
                .iter()
                .map(|c| c.token_id)
                .collect();
            match topk_set_agreement_or_boundary_tie(
                &full_logits,
                &expected,
                &got,
                k,
                LM_HEAD_TOPK_TIE_EPSILON,
            ) {
                Ok(TopkSetAgreement::ExactSet) => {}
                Ok(TopkSetAgreement::AcceptedBoundaryTie {
                    margin,
                    epsilon,
                    boundary_logit,
                    differing_tokens,
                }) => {
                    accepted_ties[i] += 1;
                    eprintln!(
                        "accepted top-{k} boundary tie at step {step} \
                                 (pos {pos_in_window}): margin={margin:.6e}, \
                                 epsilon={epsilon:.6e}, boundary_logit={boundary_logit:.6e}, \
                                 differing_tokens={differing_tokens:?}"
                    );
                }
                Err(mismatch) => {
                    panic!(
                        "top-{k} SET mismatch at step {step} \
                                 (pos {pos_in_window}): {mismatch:?}"
                    );
                }
            }
        }
        pos_in_window += 1;
    }

    for (i, &k) in ks.iter().enumerate() {
        eprintln!(
            "top-{k} set agreement: {}/{num_positions} exact, {} accepted boundary ties",
            num_positions - accepted_ties[i],
            accepted_ties[i]
        );
    }
}

/// Resolve the on-disk Q4 checkpoint dir + parsed config for the
/// real Qwen3.5-0.8B Q4 checkpoint, or `None` (with an `eprintln!`
/// explanation) if the checkpoint isn't present locally. Mirrors
/// `load_real_qwen35_0_8b_or_skip` but for the `from_q4_dir` loading
/// path (issue #171 Task 3: extend real-checkpoint agreement
/// coverage from Q8-equivalent f16 to Q4).
fn load_real_qwen35_0_8b_q4_dir_or_skip() -> Option<(std::path::PathBuf, Qwen35Config)> {
    let q4_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b-q4");
    let config_path = q4_dir.join("config.json");
    let embed_path = q4_dir.join("model_language_model_embed_tokens_weight.q4");
    if !config_path.exists() || !embed_path.exists() {
        eprintln!(
            "skipping real-checkpoint Q4 test: model files missing at {}",
            q4_dir.display()
        );
        return None;
    }
    let cfg =
        Qwen35Config::from_config_json(&config_path).expect("parse qwen3.5-0.8b-q4 config.json");
    Some((q4_dir, cfg))
}

/// Test-only: recompile `MSL_SOURCE` and build the pipeline for a single
/// named kernel, bypassing the group_size-based variant selection that
/// `MetalQwen35Engine::new`/`from_q4_dir` perform at load time. Used to
/// force the pre-specialization-equivalent g8 (`MAX_GRP=8`) prefill
/// attention variant onto an already-constructed state's pipeline table,
/// so a live in-session reference is available without a second model
/// checkpoint (see `prefill_g4_variant_matches_g8_reference_argmax_topk`).
fn force_prefill_pipeline_variant(device: &Device, kernel_name: &str) -> ComputePipelineState {
    let opts = CompileOptions::new();
    let library = device
        .new_library_with_source(MSL_SOURCE, &opts)
        .expect("recompile MSL_SOURCE for forced prefill variant");
    let func = library
        .get_function(kernel_name, None)
        .expect("find forced prefill kernel variant");
    device
        .new_compute_pipeline_state_with_function(&func)
        .expect("build forced prefill pipeline")
}

/// Ship-safety guard for the group_size -> MAX_GRP variant mapping
/// (rung-0 prefill-attention occupancy change). The kernel's own
/// `group_size > MAX_GRP` guard makes an under-sized variant pick
/// *silent* (zeroed/unwritten attention rows for the excess query
/// heads), not a compile error or panic — so this mapping is the only
/// thing standing between "occupancy win" and "silent correctness bug"
/// for models whose group_size falls between the specialized sizes.
/// MUTATION-SENSITIVE: a mapping that rounded DOWN (e.g. gs=6 -> g4)
/// would pass a naive "returns a valid suffix" check but fail this one.
#[test]
fn prefill_maxgrp_suffix_rounds_up_never_down() {
    assert_eq!(prefill_maxgrp_suffix(1).unwrap(), "g4");
    assert_eq!(prefill_maxgrp_suffix(2).unwrap(), "g4");
    assert_eq!(prefill_maxgrp_suffix(4).unwrap(), "g4");
    assert_eq!(
        prefill_maxgrp_suffix(5).unwrap(),
        "g6",
        "group_size=5 must round UP to g6 (a g4 pipeline would silently \
                 drop 1 of every 5 query heads' attention output)"
    );
    assert_eq!(prefill_maxgrp_suffix(6).unwrap(), "g6");
    assert_eq!(
        prefill_maxgrp_suffix(7).unwrap(),
        "g8",
        "group_size=7 must round UP to g8 (a g6 pipeline would silently \
                 drop 1 of every 7 query heads' attention output)"
    );
    assert_eq!(prefill_maxgrp_suffix(8).unwrap(), "g8");
    // Out-of-range fails CLOSED (Err), never silently maps to g8: there
    // is no MAX_GRP > 8 variant, so a group_size >= 9 mapped to g8 would
    // hit the kernel's silent no-attention guard. Rounding down/up-to-a-
    // too-small variant is the exact corruption this guards.
    assert!(
        prefill_maxgrp_suffix(0).is_err(),
        "group_size=0 must be an error, not a suffix"
    );
    assert!(
        prefill_maxgrp_suffix(9).is_err(),
        "group_size=9 must be an error, not silently rounded to g8"
    );
}

/// Rung-0 occupancy change (group_size-specialized `MAX_GRP` prefill
/// attention kernel, measured -17.4% TTFT@4096 on the 0.8b model at
/// `MAX_GRP=4`): a PURE occupancy change — the `MAX_GRP` rows above a
/// model's `group_size` were always idle in the old fleet-max-8 kernel,
/// so specializing `MAX_GRP` to `group_size` must not change any
/// computed value, only threadgroup-memory occupancy.
///
/// qwen3.5-0.8b-q4 has group_size=4 (8 attention heads / 2 KV heads),
/// so production loading now selects the g4 variant. This test builds
/// a second, independently-loaded state and forces its prefill
/// pipeline fields to the g8 variant (byte-for-byte equivalent to the
/// pre-change `MAX_GRP=8` kernel) as a live in-session reference, then
/// asserts the two variants agree exactly on a fixed synthetic prompt.
/// Any divergence is a wiring bug (wrong variant name, broken
/// selection) — NOT expected numeric drift, per the task's bit-identical
/// claim: the real per-row compute (`group_size`-bounded loops) indexes
/// threadgroup arrays by `HEAD_DIM`/`TILE_TOKENS` strides, not `MAX_GRP`,
/// so shrinking `MAX_GRP` only shrinks array allocation, never the
/// values or operation order for the rows that are actually used.
#[test]
fn prefill_g4_variant_matches_g8_reference_argmax_topk() {
    let Some(_) = Device::system_default() else {
        eprintln!("skipping prefill g4-vs-g8 parity test: no Metal device");
        return;
    };
    let Some((q4_dir, cfg)) = load_real_qwen35_0_8b_q4_dir_or_skip() else {
        return;
    };
    let tokenizer_path =
        std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
            .join(".lattice/models/qwen3.5-0.8b/tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!(
            "skipping prefill g4-vs-g8 parity test: tokenizer missing at {}",
            tokenizer_path.display()
        );
        return;
    }
    assert_eq!(
        cfg.num_attention_heads / cfg.num_key_value_heads,
        4,
        "this test's g4-vs-g8 comparison assumes qwen3.5-0.8b-q4 group_size=4 \
                 (8 attention heads / 2 KV heads); re-check the on-disk config if this fails"
    );

    let _guard = gpu_test_lock();
    let max_cache_len = 1024usize;

    // Path A: production selection — group_size=4 picks the g4 variant.
    let mut state_g4 =
        MetalQwen35State::from_q4_dir(&q4_dir, tokenizer_path.as_path(), &cfg, max_cache_len)
            .expect("load qwen3.5-0.8b-q4 (g4 production selection)");

    // Path B: independently-loaded state with its prefill pipelines
    // forced to the g8 (pre-change-equivalent) variant.
    let mut state_g8 =
        MetalQwen35State::from_q4_dir(&q4_dir, tokenizer_path.as_path(), &cfg, max_cache_len)
            .expect("load qwen3.5-0.8b-q4 (forced g8 reference)");
    state_g8.engine.pipelines.prefill_attention_batched = force_prefill_pipeline_variant(
        &state_g8.engine.device,
        "prefill_attention_batched_causal_g8",
    );
    state_g8.engine.pipelines.prefill_attention_batched_f16 = force_prefill_pipeline_variant(
        &state_g8.engine.device,
        "prefill_attention_batched_causal_g8_f16",
    );

    let seq_len = 512usize;
    let tokens: Vec<u32> = (0..seq_len as u32).map(|i| 100 + i % 1000).collect();
    assert!(
        tokens.iter().all(|&t| (t as usize) < cfg.vocab_size),
        "synthetic prompt token ids must be within vocab_size"
    );

    let logits_g4 = state_g4.forward_prefill(&tokens);
    let logits_g8 = state_g8.forward_prefill(&tokens);

    assert_eq!(logits_g4.len(), cfg.vocab_size);
    assert_eq!(logits_g8.len(), cfg.vocab_size);

    let top4_g4 = cpu_topk_token_ids(&logits_g4, 4);
    let top4_g8 = cpu_topk_token_ids(&logits_g8, 4);

    let max_abs_diff = logits_g4
        .iter()
        .zip(logits_g8.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "prefill g4-vs-g8 parity: argmax_g4={}, argmax_g8={}, top4_g4={top4_g4:?}, \
                 top4_g8={top4_g8:?}, max_abs_diff={max_abs_diff:.8}",
        top4_g4[0], top4_g8[0],
    );

    assert_eq!(
        top4_g4[0], top4_g8[0],
        "argmax diverged between g4 and g8 prefill attention variants — this is a bug \
                 in variant selection or macro parameterization, not expected numeric drift"
    );
    assert_eq!(
        top4_g4, top4_g8,
        "top-4 token ids diverged between g4 and g8 prefill attention variants"
    );
    assert_eq!(
        max_abs_diff, 0.0,
        "g4 and g8 prefill attention variants must be BIT-IDENTICAL (pure occupancy \
                 change): max_abs_diff={max_abs_diff}"
    );
}

/// Q4 counterpart of `lm_head_real_checkpoint_greedy_agreement_10k_positions`
/// (issue #171 Task 3). `from_q4_dir` loads+dequantizes the full weight
/// set from disk per call (no shared-weights constructor like the f16
/// `Qwen35Model` path), so this leg uses a reduced default position
/// count to keep runtime sane; override via
/// `LATTICE_TEST_Q4_GREEDY_POSITIONS`. Same reset/re-apply-compact-route
/// discipline as the f16 leg (issue #171).
#[test]
fn lm_head_q4_real_checkpoint_greedy_agreement() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let Some((q4_dir, cfg)) = load_real_qwen35_0_8b_q4_dir_or_skip() else {
        return;
    };
    let _guard = gpu_test_lock();

    let num_positions: usize = std::env::var("LATTICE_TEST_Q4_GREEDY_POSITIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_000);
    let reset_every = 64usize;
    let vocab = cfg.vocab_size;
    let max_cache_len = 128usize;
    let tokenizer_path = std::path::Path::new("/dev/null");

    let mut baseline = MetalQwen35State::from_q4_dir(&q4_dir, tokenizer_path, &cfg, max_cache_len)
        .expect("baseline real-checkpoint Q4 state");
    let mut blocked = MetalQwen35State::from_q4_dir(&q4_dir, tokenizer_path, &cfg, max_cache_len)
        .expect("blocked real-checkpoint Q4 state");
    blocked.session.compact_route = GpuTopkRoute::BlockArgmax;
    blocked.session.compact_topk = 1;

    let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
    let mut accepted_ties = 0usize;
    let mut pos_in_window = 0usize;
    for step in 0..num_positions {
        if pos_in_window >= reset_every {
            baseline.reset_state();
            blocked.reset_state();
            // `reset_state` clears `compact_route`/`compact_topk` (issue #171);
            // re-apply after every reset exactly as
            // a real caller starting a fresh `generate_*` call would via the
            // route-decision block.
            blocked.session.compact_route = GpuTopkRoute::BlockArgmax;
            blocked.session.compact_topk = 1;
            pos_in_window = 0;
        }
        let token = (xorshift64_next(&mut rng) % vocab as u64) as u32;

        let full_logits = baseline.forward_step(token, pos_in_window);
        let expected = cpu_topk_token_ids(&full_logits, 1)[0];

        let empty = blocked.forward_step(token, pos_in_window);
        assert!(empty.is_empty(), "compact route must return empty logits");
        assert_eq!(blocked.session.compact_result.len(), 1);
        let got = blocked.session.compact_result[0].token_id;

        // Q4's coarser (4-bit vs. 16-bit) quantization noise floor makes
        // near-tied top-1/top-2 logit gaps flip more often between the
        // full-logit GEMV (CPU oracle here) and the compact Stage-1
        // block-argmax kernel than the F16 path does, so this uses its own,
        // larger LM_HEAD_TOPK_TIE_EPSILON_Q4 (see its doc comment for the
        // empirical derivation) rather than the F16 epsilon (issue #623).
        match topk_set_agreement_or_boundary_tie(
            &full_logits,
            &std::collections::HashSet::from([expected]),
            &std::collections::HashSet::from([got]),
            1,
            LM_HEAD_TOPK_TIE_EPSILON_Q4,
        ) {
            Ok(TopkSetAgreement::ExactSet) => {}
            Ok(TopkSetAgreement::AcceptedBoundaryTie {
                margin,
                epsilon,
                boundary_logit,
                differing_tokens,
            }) => {
                accepted_ties += 1;
                eprintln!(
                    "accepted Q4 greedy boundary tie at step {step} \
                             (pos {pos_in_window}): margin={margin:.6e}, \
                             epsilon={epsilon:.6e}, boundary_logit={boundary_logit:.6e}, \
                             differing_tokens={differing_tokens:?}"
                );
            }
            Err(mismatch) => {
                panic!("Q4 greedy mismatch at step {step} (pos {pos_in_window}): {mismatch:?}");
            }
        }
        pos_in_window += 1;
        if step % 500 == 0 {
            eprintln!("Q4 greedy agreement progress: {step}/{num_positions}");
        }
    }

    eprintln!(
        "Q4 greedy agreement: {}/{num_positions} exact, {accepted_ties} accepted boundary ties",
        num_positions - accepted_ties
    );
}

/// Q4 counterpart of `lm_head_real_checkpoint_topk_set_agreement`
/// (issue #171 Task 3). Reduced default position count (100 vs. 500)
/// to bound the cost of the five separate `from_q4_dir` loads
/// (baseline + one blocked state per K); override via
/// `LATTICE_TEST_Q4_TOPK_POSITIONS`.
#[test]
fn lm_head_q4_real_checkpoint_topk_set_agreement() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let Some((q4_dir, cfg)) = load_real_qwen35_0_8b_q4_dir_or_skip() else {
        return;
    };
    let _guard = gpu_test_lock();

    let num_positions: usize = std::env::var("LATTICE_TEST_Q4_TOPK_POSITIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let reset_every = 64usize;
    let vocab = cfg.vocab_size;
    let max_cache_len = 128usize;
    let tokenizer_path = std::path::Path::new("/dev/null");
    let ks = [8usize, 16, 40, 64];

    let mut baseline = MetalQwen35State::from_q4_dir(&q4_dir, tokenizer_path, &cfg, max_cache_len)
        .expect("baseline real-checkpoint Q4 state");
    let mut blocked_states: Vec<MetalQwen35State> = ks
        .iter()
        .map(|_| {
            MetalQwen35State::from_q4_dir(&q4_dir, tokenizer_path, &cfg, max_cache_len)
                .expect("blocked real-checkpoint Q4 state")
        })
        .collect();
    for (i, &k) in ks.iter().enumerate() {
        blocked_states[i].session.compact_route = GpuTopkRoute::BlockTopK { local_k: k as u32 };
        blocked_states[i].session.compact_topk = k;
    }

    let mut rng: u64 = 0xFEED_FACE_C0FF_EE00;
    let mut accepted_ties = [0usize; 4];
    let mut pos_in_window = 0usize;
    for step in 0..num_positions {
        if pos_in_window >= reset_every {
            baseline.reset_state();
            // `reset_state` clears `compact_route`/`compact_topk` (issue #171);
            // re-apply each state's compact-route
            // configuration after every reset, exactly as a real caller
            // starting a fresh `generate_*` call would via the route-decision
            // block.
            for (i, &k) in ks.iter().enumerate() {
                blocked_states[i].reset_state();
                blocked_states[i].session.compact_route =
                    GpuTopkRoute::BlockTopK { local_k: k as u32 };
                blocked_states[i].session.compact_topk = k;
            }
            pos_in_window = 0;
        }
        let token = (xorshift64_next(&mut rng) % vocab as u64) as u32;
        let full_logits = baseline.forward_step(token, pos_in_window);

        for (i, &k) in ks.iter().enumerate() {
            let expected: std::collections::HashSet<u32> =
                cpu_topk_token_ids(&full_logits, k).into_iter().collect();
            let empty = blocked_states[i].forward_step(token, pos_in_window);
            assert!(empty.is_empty());
            assert_eq!(blocked_states[i].session.compact_result.len(), k);
            let got: std::collections::HashSet<u32> = blocked_states[i]
                .session
                .compact_result
                .iter()
                .map(|c| c.token_id)
                .collect();
            // Q4's coarser (4-bit vs. 16-bit) quantization noise floor makes
            // near-tied rank-k/rank-(k+1) logit gaps flip more often between
            // the full-logit GEMV (CPU oracle here) and the compact Stage-1
            // block-top-k kernel than the F16 path does, so this uses its own,
            // larger LM_HEAD_TOPK_TIE_EPSILON_Q4 (see its doc comment for the
            // empirical derivation) rather than the F16 epsilon (issue #623).
            match topk_set_agreement_or_boundary_tie(
                &full_logits,
                &expected,
                &got,
                k,
                LM_HEAD_TOPK_TIE_EPSILON_Q4,
            ) {
                Ok(TopkSetAgreement::ExactSet) => {}
                Ok(TopkSetAgreement::AcceptedBoundaryTie {
                    margin,
                    epsilon,
                    boundary_logit,
                    differing_tokens,
                }) => {
                    accepted_ties[i] += 1;
                    eprintln!(
                        "accepted Q4 top-{k} boundary tie at step {step} \
                                 (pos {pos_in_window}): margin={margin:.6e}, \
                                 epsilon={epsilon:.6e}, boundary_logit={boundary_logit:.6e}, \
                                 differing_tokens={differing_tokens:?}"
                    );
                }
                Err(mismatch) => {
                    panic!(
                        "Q4 top-{k} SET mismatch at step {step} \
                                 (pos {pos_in_window}): {mismatch:?}"
                    );
                }
            }
        }
        pos_in_window += 1;
    }

    for (i, &k) in ks.iter().enumerate() {
        eprintln!(
            "Q4 top-{k} set agreement: {}/{num_positions} exact, {} accepted boundary ties",
            num_positions - accepted_ties[i],
            accepted_ties[i]
        );
    }
}
