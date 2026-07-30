#[test]
fn forward_step_gdn_only_does_not_advance_kv_cache() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let seq_len_before = state.session.kv_cache.seq_len;
    let logits = state.forward_step_gdn_only(0, 0);

    assert_eq!(
        state.session.kv_cache.seq_len, seq_len_before,
        "forward_step_gdn_only must not advance KV cache"
    );
    assert_eq!(
        logits.len(),
        cfg.vocab_size,
        "forward_step_gdn_only must return vocab_size logits"
    );
}

#[test]
fn forward_step_gdn_only_returns_finite_logits() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let logits = state.forward_step_gdn_only(1, 0);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "all GDN-only logits must be finite"
    );
}

/// Mutation-sensitive guard for the GDN decode decay-gate clamp.
///
/// When `a_log > ~88`, the kernel's `exp(a_log)` overflows to `+inf`; when
/// `alpha + dt_bias` drives softplus to exactly `0.0`, the *unclamped*
/// product is `inf * 0 = NaN`, which poisons the recurrent state and every
/// subsequent logit (coherent-early / garbage-late on long generations).
/// The fix mirrors the CPU `compute_decay_gate` clamp (`exp(a_log).min(MAX)`).
/// This test FAILS (non-finite logits) if the decode clamp is reverted and
/// PASSES once it is present.
#[test]
fn forward_step_gdn_only_decay_gate_clamps_overflow() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, mut weights) = tiny_hybrid_fixture();

    // Drive every GDN head into the overflow corner: a_log huge (exp -> +inf)
    // and dt_bias very negative (softplus -> 0). Unclamped: inf * 0 = NaN.
    for (attn, _) in weights.layers.iter_mut() {
        if let AttentionWeights::Linear(gdn) = attn {
            for a in gdn.a_log.iter_mut() {
                *a = 100.0;
            }
            for b in gdn.dt_bias.iter_mut() {
                *b = -200.0;
            }
        }
    }

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let logits = state.forward_step_gdn_only(1, 0);

    assert!(
        logits.iter().all(|v| v.is_finite()),
        "GDN decode decay-gate must clamp exp(a_log) to finite; got non-finite \
             logits (inf * 0 = NaN poison) — the decode clamp was reverted"
    );
}
