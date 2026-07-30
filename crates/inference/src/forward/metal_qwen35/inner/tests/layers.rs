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

#[test]
fn test_metal_qwen35_golden_logit_snapshot_forward_step_token_42_pos_0() {
    let Some(_) = Device::system_default() else {
        return;
    };
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
fn test_metal_qwen35_kv_cache_determinism_replay_5_tokens() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let tokens = [42_u32, 7, 13, 8, 42];
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");

    // First pass
    let mut first_logits: Vec<Vec<f32>> = Vec::new();
    for &token in &tokens {
        let pos = state.session.kv_cache.seq_len;
        let logits = state.forward_step(token, pos);
        assert_eq!(
            state.session.kv_cache.seq_len,
            pos + 1,
            "forward_step must advance seq_len exactly once"
        );
        first_logits.push(logits);
    }
    let first_seq_len = state.session.kv_cache.seq_len;

    state.reset_state();

    // Second pass (replay)
    let mut second_logits: Vec<Vec<f32>> = Vec::new();
    for &token in &tokens {
        let pos = state.session.kv_cache.seq_len;
        let logits = state.forward_step(token, pos);
        assert_eq!(
            state.session.kv_cache.seq_len,
            pos + 1,
            "forward_step must advance seq_len exactly once"
        );
        second_logits.push(logits);
    }
    let second_seq_len = state.session.kv_cache.seq_len;

    assert_eq!(first_seq_len, tokens.len(), "first pass seq_len");
    assert_eq!(second_seq_len, tokens.len(), "second pass seq_len");
    for (step, (first, second)) in first_logits.iter().zip(second_logits.iter()).enumerate()
    {
        let max_abs_diff = first
            .iter()
            .zip(second.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "replay logits diverged at step {step}: max_abs_diff={max_abs_diff}"
        );
    }
}
