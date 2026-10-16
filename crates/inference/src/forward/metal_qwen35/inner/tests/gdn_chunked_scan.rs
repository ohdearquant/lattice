//! GDN chunked-scan algebra and its Metal kernels.

use super::*;
// -----------------------------------------------------------------------
// GDN chunked-scan unit tests (CPU algebra, no GPU required)
// -----------------------------------------------------------------------

/// Verify make_gdn_chunk_params returns None for unsupported shapes.
#[test]
fn gdn_chunk_params_unsupported_shape_returns_none() {
    let mut cfg = crate::model::qwen35_config::Qwen35Config::qwen35_0_8b();
    cfg.hidden_size = 2048; // not 0.8B shape
    assert!(
        MetalQwen35State::make_gdn_chunk_params(&cfg, 32).is_none(),
        "should return None for unsupported shape"
    );
}

/// Verify make_gdn_chunk_params returns correct values for 0.8B.
#[test]
fn gdn_chunk_params_08b_correct() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen35_0_8b();
    let p =
        MetalQwen35State::make_gdn_chunk_params(&cfg, 64).expect("0.8B shape should be supported");
    assert_eq!(p.key_dim, 128);
    assert_eq!(p.value_dim, 128);
    assert_eq!(p.num_key_heads, 16);
    assert_eq!(p.num_value_heads, 16);
    assert_eq!(p.n_tokens, 64);
    assert_eq!(p.num_chunks, 2); // 64 / 32 = 2
    assert_eq!(p.chunk_size, 32);
    assert_eq!(p.q_total, 16 * 128);
    assert_eq!(p.v_offset, 16 * 128 * 2);
}

/// Verify make_gdn_chunk_params handles n_tokens=0.
#[test]
fn gdn_chunk_params_zero_tokens_returns_none() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen35_0_8b();
    assert!(
        MetalQwen35State::make_gdn_chunk_params(&cfg, 0).is_none(),
        "n_tokens=0 should return None"
    );
}

#[cfg(feature = "bench-internals")]
fn run_gdn_prefill_scheduler_pair(
    cfg: &Qwen35Config,
    weights: &ModelWeights,
    tokens: &[u32],
    chunked: bool,
) -> (
    PrefillSchedulerArtifacts,
    PrefillSchedulerArtifacts,
    bench_support::GdnIsolatedChunkTiming,
    Vec<bench_support::GdnLayerCapture>,
) {
    let mut production =
        MetalQwen35State::new(weights, cfg, 16).expect("production scheduler fixture");
    production.use_gdn_chunked = chunked;
    let _ = production.forward_prefill_batched_chunk(tokens, 0, true, false, false);
    let production_artifacts = prefill_scheduler_artifacts(&mut production, tokens, 9);

    let mut isolated = MetalQwen35State::new(weights, cfg, 16).expect("isolated scheduler fixture");
    isolated.use_gdn_chunked = chunked;
    let capture_requests = if chunked {
        vec![bench_support::GdnCaptureRequest {
            linear_idx: 0,
            value_head: 3,
        }]
    } else {
        vec![]
    };
    let (timing, captures) =
        isolated.forward_prefill_chunk_gdn_isolated(tokens, 0, &capture_requests);
    let isolated_artifacts = prefill_scheduler_artifacts(&mut isolated, tokens, 9);
    (production_artifacts, isolated_artifacts, timing, captures)
}

#[cfg(feature = "bench-internals")]
#[test]
fn gdn_prefill_production_and_isolated_scheduler_state_matches() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_chunked_prefill_fixture();
    let tokens = [1, 3, 5, 7];
    for chunked in [false, true] {
        let (production, isolated, _, _) =
            run_gdn_prefill_scheduler_pair(&cfg, &weights, &tokens, chunked);
        assert_eq!(production.position, isolated.position, "chunked={chunked}");
        assert_eq!(production.kv_rows, isolated.kv_rows, "chunked={chunked}");
        assert_eq!(
            production.gdn_snapshot, isolated.gdn_snapshot,
            "chunked={chunked}"
        );
    }
}

#[cfg(feature = "bench-internals")]
#[test]
fn gdn_prefill_scheduler_followup_and_capture_match() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_chunked_prefill_fixture();
    let tokens = [1, 3, 5, 7];
    for chunked in [false, true] {
        let (production, isolated, timing, captures) =
            run_gdn_prefill_scheduler_pair(&cfg, &weights, &tokens, chunked);
        assert_eq!(
            production.next_token, isolated.next_token,
            "chunked={chunked}"
        );
        assert_eq!(timing.num_gdn_layers, 1, "chunked={chunked}");
        if chunked {
            assert_eq!(captures.len(), 1);
            let capture = &captures[0];
            assert_eq!(capture.linear_idx, 0);
            assert_eq!(capture.value_head, 3);
            assert_eq!(capture.length, tokens.len());
            assert_eq!(capture.dk, 128);
            assert_eq!(capture.dv, 128);
            assert_eq!(capture.q.len(), tokens.len() * capture.dk);
            assert_eq!(capture.k.len(), tokens.len() * capture.dk);
            assert_eq!(capture.v.len(), tokens.len() * capture.dv);
            assert_eq!(capture.beta.len(), tokens.len());
            assert_eq!(capture.log_alpha.len(), tokens.len());
        } else {
            assert!(captures.is_empty());
        }
    }
}

#[cfg(feature = "bench-internals")]
#[test]
fn gdn_prefill_schedulers_preserve_unsupported_chunked_policy() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, weights) = tiny_hybrid_fixture();
    assert!(!MetalQwen35State::supports_gdn_chunked_prefill(&cfg));

    let mut production =
        MetalQwen35State::new(&weights, &cfg, 8).expect("production fallback fixture");
    production.use_gdn_chunked = true;
    let _ = production.forward_prefill_batched_chunk(&[1, 2], 0, true, false, false);
    assert_eq!(production.session.position(), 2);
    assert_eq!(production.session.kv_cache.seq_len, 2);

    use crate::speculative::MtpTargetVerifier as _;
    let mut isolated =
        MetalQwen35State::new(&weights, &cfg, 8).expect("isolated preflight fixture");
    isolated.use_gdn_chunked = true;
    let before = isolated.snapshot_gdn_states();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        isolated.forward_prefill_chunk_gdn_isolated(&[1, 2], 0, &[])
    }));
    assert!(
        panic.is_err(),
        "isolated scheduler must reject unsupported chunked GDN"
    );
    assert_eq!(isolated.session.position(), 0);
    assert_eq!(isolated.session.kv_cache.seq_len, 0);
    assert_eq!(isolated.snapshot_gdn_states(), before);
}

/// Verify make_gdn_chunk_params with tail chunk (n_tokens not multiple of 32).
#[test]
fn gdn_chunk_params_tail_chunk() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen35_0_8b();
    let p =
        MetalQwen35State::make_gdn_chunk_params(&cfg, 33).expect("0.8B shape should be supported");
    assert_eq!(p.n_tokens, 33);
    assert_eq!(p.num_chunks, 2); // ceil(33/32) = 2
}

/// Regression for the chunked log-decay NaN edge (PR #235).
///
/// `gdn_chunk_materialize_c32` stores each token's log-decay `-a*sp` and
/// `gdn_chunk_solve_c32` forms decay ratios as `exp(cumsum[j] - cumsum[k])`.
/// When a softplus overflows, the unclamped `-a*sp` is `-inf`; once it enters
/// the cumulative sum, every later prefix is `-inf` and a same-token difference
/// is `-inf - -inf = NaN`, poisoning all downstream GDN state. The serial
/// recurrence maps the same token to `g = exp(-inf) = 0` (a finite full reset).
/// The kernel floors `-a*sp` at -88 (`fmax`) so the chunked log-difference
/// representation reproduces the serial gate-product while staying finite.
///
/// This mirrors that algebra on the CPU: the clamped chunked decay ratios must
/// be finite and equal the serial product of per-token gates, and the unclamped
/// form must still produce NaN (proving the floor is load-bearing).
#[test]
fn gdn_chunked_log_decay_floor_matches_serial_under_saturation() {
    // Must match the `fmax(-a*sp, -88.0f)` floor in MSL_SOURCE.
    const GDN_LOG_ALPHA_FLOOR: f32 = -88.0;

    let c = 8usize;
    let sat_idx = 3usize;

    // Per-token raw log-decay `-a*sp`. The saturated token overflows softplus
    // (s + dt_bias huge) so its raw value is -inf; the rest are ordinary.
    let raw_log_alpha: Vec<f32> = (0..c)
        .map(|i| {
            if i == sat_idx {
                let a = (0.5f32).exp();
                let sp = (1.0f32 + (200.0f32).exp()).ln(); // overflow -> inf
                -a * sp
            } else {
                let a = (0.3f32).exp();
                let sp = (1.0f32 + (-0.5f32 + 0.1f32 * i as f32).exp()).ln();
                -a * sp
            }
        })
        .collect();
    assert!(
        raw_log_alpha[sat_idx].is_infinite() && raw_log_alpha[sat_idx] < 0.0,
        "setup: saturated token must give -inf, got {}",
        raw_log_alpha[sat_idx]
    );

    // Serial reference gate g[m] = exp(log_alpha[m]); the -inf token -> exactly 0.
    let g_serial: Vec<f32> = raw_log_alpha.iter().map(|&x| x.exp()).collect();
    assert_eq!(
        g_serial[sat_idx], 0.0,
        "serial gate at the saturated token must be exactly 0"
    );

    // Chunked path: floor, cumulative-sum, decay ratios via log differences.
    let clamped: Vec<f32> = raw_log_alpha
        .iter()
        .map(|&x| x.max(GDN_LOG_ALPHA_FLOOR))
        .collect();
    let mut cumsum = vec![0.0f32; c];
    let mut acc = 0.0f32;
    for j in 0..c {
        acc += clamped[j];
        cumsum[j] = acc;
    }

    // Every causal ratio L[j][k] (j >= k) must be finite and equal the serial
    // product prod_{m=k+1..j} g[m].
    let mut max_abs = 0.0f32;
    for j in 0..c {
        for k in 0..=j {
            let l_chunked = (cumsum[j] - cumsum[k]).exp();
            assert!(
                l_chunked.is_finite(),
                "chunked decay ratio L[{j}][{k}] is non-finite ({l_chunked}) after floor"
            );
            let mut p_serial = 1.0f32;
            for m in (k + 1)..=j {
                p_serial *= g_serial[m];
            }
            max_abs = max_abs.max((l_chunked - p_serial).abs());
        }
    }
    assert!(
        max_abs < 1e-5,
        "floored chunked decay ratios diverge from serial gate products: max_abs={max_abs:.3e}"
    );

    // Negative control: without the floor, the log-difference is NaN — the exact
    // failure the floor exists to prevent.
    let mut cumsum_raw = vec![0.0f32; c];
    let mut acc_raw = 0.0f32;
    for j in 0..c {
        acc_raw += raw_log_alpha[j];
        cumsum_raw[j] = acc_raw;
    }
    let unclamped_ratio = (cumsum_raw[sat_idx + 1] - cumsum_raw[sat_idx]).exp();
    assert!(
        unclamped_ratio.is_nan(),
        "regression target: unclamped log-difference must be NaN, got {unclamped_ratio}"
    );
}

/// Verify supports_gdn_chunked_prefill accepts 0.8B.
#[test]
fn supports_gdn_chunked_prefill_08b() {
    let cfg = crate::model::qwen35_config::Qwen35Config::qwen35_0_8b();
    assert!(MetalQwen35State::supports_gdn_chunked_prefill(&cfg));
}

/// Verify supports_gdn_chunked_prefill rejects other shapes.
#[test]
fn supports_gdn_chunked_prefill_rejects_non_08b() {
    let mut cfg = crate::model::qwen35_config::Qwen35Config::qwen35_0_8b();
    cfg.hidden_size = 4096;
    assert!(!MetalQwen35State::supports_gdn_chunked_prefill(&cfg));
}

/// Gate 2: B-vs-B self-consistency.  Running forward_prefill (chunked GDN path, default ON)
/// 8 times on the same prompt must produce logits that agree within max_abs_diff < 1e-3 between
/// consecutive runs.  Tests that the chunked kernels are deterministic under back-to-back calls.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn gdn_chunked_b_vs_b_self_consistency() {
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir) {
        eprintln!("skipping: model missing");
        return;
    }
    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let Some(_device) = Device::system_default() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let _gpu = gpu_test_lock();
    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal state");
    let tokens = long_real_text_tokens(model.tokenizer());

    let mut runs: Vec<Vec<f32>> = Vec::new();
    for i in 0..8usize {
        state.reset_state();
        let logits = state.forward_prefill(&tokens);
        eprintln!("B-vs-B run {}: argmax={}", i, argmax_f32(&logits));
        runs.push(logits);
    }

    let mut max_overall = 0.0f32;
    for (pair_idx, pair) in runs.windows(2).enumerate() {
        let diff = pair[0]
            .iter()
            .zip(pair[1].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "B-vs-B runs {pair_idx}..{}: max_abs_diff={diff:.6}",
            pair_idx + 1
        );
        max_overall = max_overall.max(diff);
        assert!(
            diff < 1e-3,
            "B-vs-B self-consistency FAIL: runs {pair_idx} and {} differ by {diff:.6} (> 1e-3)",
            pair_idx + 1
        );
    }
    eprintln!("B-vs-B gate PASS: max_abs_diff across all pairs = {max_overall:.6}");
}

/// Per-layer and element-wise difference between two GDN state snapshots, with
/// `reference` on the denominator side.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
struct GdnSnapshotDiff {
    /// Largest absolute element difference over all layers.
    max_abs: f32,
    /// Largest relative difference over elements with |reference| > 1e-2.
    max_rel_sig: f32,
    /// Number of those elements whose relative difference is at least 5e-2.
    sig_over: usize,
    /// `||S - S_ref||_F / ||S_ref||_F` per layer, accumulated in f64.
    layer_rel_l2: Vec<f64>,
}

/// Compare the S matrices of two GDN snapshots. A raw relative metric explodes on
/// near-zero entries, so the element-wise relative error only covers elements whose
/// reference magnitude exceeds 1e-2. A non-finite difference is reported as infinite
/// everywhere, because `f32::max` would otherwise drop a NaN and let it pass.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn gdn_snapshot_diff(
    snap: &crate::attention::gdn::GdnSnapshot,
    reference: &crate::attention::gdn::GdnSnapshot,
) -> GdnSnapshotDiff {
    assert_eq!(snap.len(), reference.len(), "snapshot layer count mismatch");
    assert!(!snap.is_empty(), "snapshot holds no GDN layers");
    let (mut max_abs, mut max_rel_sig, mut sig_over) = (0.0f32, 0.0f32, 0usize);
    let mut all_finite = true;
    let mut layer_rel_l2 = Vec::with_capacity(snap.len());
    for ((s, _), (s_ref, _)) in snap.iter().zip(reference.iter()) {
        assert_eq!(s.len(), s_ref.len(), "S matrix element count mismatch");
        let (mut diff_sq, mut ref_sq) = (0.0f64, 0.0f64);
        let mut layer_finite = true;
        for (a, r) in s.iter().zip(s_ref.iter()) {
            let abs_diff = (a - r).abs();
            layer_finite &= abs_diff.is_finite();
            max_abs = max_abs.max(abs_diff);
            if r.abs() > 1e-2f32 {
                let rel = abs_diff / r.abs();
                max_rel_sig = max_rel_sig.max(rel);
                if rel >= 5e-2f32 {
                    sig_over += 1;
                }
            }
            diff_sq += f64::from(abs_diff) * f64::from(abs_diff);
            ref_sq += f64::from(*r) * f64::from(*r);
        }
        all_finite &= layer_finite;
        layer_rel_l2.push(if !layer_finite {
            f64::INFINITY
        } else if ref_sq > 0.0 {
            (diff_sq / ref_sq).sqrt()
        } else if diff_sq == 0.0 {
            0.0
        } else {
            f64::INFINITY
        });
    }
    if !all_finite {
        max_abs = f32::INFINITY;
        max_rel_sig = f32::INFINITY;
    }
    GdnSnapshotDiff {
        max_abs,
        max_rel_sig,
        sig_over,
        layer_rel_l2,
    }
}

/// Reset `state`, run batched `forward_prefill` over `tokens` with the chunked GDN scan
/// on or off, and return the resulting GDN state.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn gdn_prefill_snapshot(
    state: &mut MetalQwen35State,
    tokens: &[u32],
    chunked: bool,
) -> crate::attention::gdn::GdnSnapshot {
    use crate::speculative::MtpTargetVerifier as _;
    state.use_gdn_chunked = chunked;
    state.reset_state();
    let _ = state.forward_prefill(tokens);
    state.snapshot_gdn_states()
}

/// Gate A: the chunked GDN scan against the serial GDN scan.
///
/// Both sides run through batched `forward_prefill` and share every projection, so the
/// only difference is the recurrence itself: chunk-GEMMs in the chunked scan against the
/// token-by-token update in the serial scan. The final S matrices must agree within
/// max_abs < 5e-3 and, over elements with |S| > 1e-2, max relative difference < 5e-2.
/// The check runs at the long prompt and at its first 101 tokens.
///
/// The chunked snapshot, which is the path under test, is computed once per prompt and
/// held fixed. Only the serial reference is re-sampled (at most 3 attempts, best kept),
/// so a wrong chunked snapshot diverges on every attempt and cannot be masked by
/// best-of-N.
///
/// Prefill against decode is a separate comparison, bounded by
/// `gdn_prefill_state_vs_decode_state_diff`.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn gdn_chunked_state_vs_serial_state_diff() {
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir) {
        eprintln!("skipping: model missing");
        return;
    }
    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let Some(_device) = Device::system_default() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let _gpu = gpu_test_lock();
    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal state");
    let long_tokens = long_real_text_tokens(model.tokenizer());
    assert!(
        long_tokens.len() > 101,
        "long prompt must exceed 101 tokens"
    );

    const MAX_ABS_THRESHOLD: f32 = 5e-3;
    const MAX_REL_SIG_THRESHOLD: f32 = 5e-2;

    let mut failures = Vec::new();
    for (label, tokens) in [
        ("long", &long_tokens[..]),
        ("short101", &long_tokens[..101]),
    ] {
        // Chunked path, computed once and held fixed.
        let chunked_snap = gdn_prefill_snapshot(&mut state, tokens, true);

        let mut best: Option<(f32, f32, usize)> = None;
        for attempt in 0..3 {
            // Serial-scan prefill reference.
            let serial_snap = gdn_prefill_snapshot(&mut state, tokens, false);
            let d = gdn_snapshot_diff(&chunked_snap, &serial_snap);
            eprintln!(
                "State diff chunked vs serial prompt={label} attempt {attempt}: \
                         max_abs_diff={:.6}, max_rel_diff(|S|>1e-2)={:.6}, \
                         sig_elems_over_5e-2={}",
                d.max_abs, d.max_rel_sig, d.sig_over
            );
            // A passing attempt is kept as the result; otherwise the closest one is.
            if d.max_abs < MAX_ABS_THRESHOLD && d.max_rel_sig < MAX_REL_SIG_THRESHOLD {
                best = Some((d.max_abs, d.max_rel_sig, d.sig_over));
                break;
            }
            let better = best.map(|(a, _, _)| d.max_abs < a).unwrap_or(true);
            if better {
                best = Some((d.max_abs, d.max_rel_sig, d.sig_over));
            }
        }
        let (max_abs_diff, max_rel_sig, over) = best.expect("at least one attempt");
        eprintln!(
            "State diff chunked vs serial prompt={label} tokens={} (best of 3): \
                     max_abs_diff={max_abs_diff:.6}, max_rel_diff(|S|>1e-2)={max_rel_sig:.6}, \
                     sig_elems_over_5e-2={over}",
            tokens.len()
        );
        if !(max_abs_diff < MAX_ABS_THRESHOLD && max_rel_sig < MAX_REL_SIG_THRESHOLD) {
            failures.push(format!(
                "prompt={label} tokens={}: max_abs_diff={max_abs_diff:.6} \
                         (threshold {MAX_ABS_THRESHOLD:.0e}), max_rel_diff(|S|>1e-2)=\
                         {max_rel_sig:.6} (threshold {MAX_REL_SIG_THRESHOLD:.0e})",
                tokens.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "GDN state S diverges (chunked vs serial scan): {}",
        failures.join("; ")
    );
}

/// Gate B: batched prefill (serial GDN scan) against token-by-token `forward_step`
/// decode.
///
/// The two paths do not share projections. The Q4 prefill GEMM stages its operands in
/// half precision and the decode GEMV is f32, so the final S matrices differ for that
/// reason alone, independent of the scan. Measured on the real model, the per-layer
/// relative L2 difference `||S_prefill - S_decode||_F / ||S_decode||_F` peaks at 8.1e-4,
/// while single elements can differ by up to 0.112 relative and 2.7e-3 absolute. An
/// element-wise relative metric on a 1e-2 floor flips on those isolated elements, so this
/// gate bounds the per-layer relative L2 instead: every GDN layer must stay under 3e-3,
/// three times the largest measured value, and the overall max_abs under 5e-3. Layers are
/// compared individually so one drifting layer is not averaged away by the others.
///
/// The bound is deliberately loose: this gate cannot see a defect in a single layer that
/// moves that layer's state by less than about 3e-3 relative L2. Recurrence defects are
/// the job of `gdn_chunked_state_vs_serial_state_diff`, which holds the projections fixed.
///
/// The check runs at the long prompt and at its first 101 tokens. The prefill snapshot,
/// the path under test, is computed once per prompt and held fixed; only the decode
/// reference is re-sampled (at most 3 attempts, best kept).
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn gdn_prefill_state_vs_decode_state_diff() {
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir) {
        eprintln!("skipping: model missing");
        return;
    }
    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let Some(_device) = Device::system_default() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let _gpu = gpu_test_lock();
    use crate::speculative::MtpTargetVerifier as _;
    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal state");
    let long_tokens = long_real_text_tokens(model.tokenizer());
    assert!(
        long_tokens.len() > 101,
        "long prompt must exceed 101 tokens"
    );

    const MAX_LAYER_REL_L2_THRESHOLD: f64 = 3e-3;
    const MAX_ABS_THRESHOLD: f32 = 5e-3;

    let mut failures = Vec::new();
    for (label, tokens) in [
        ("long", &long_tokens[..]),
        ("short101", &long_tokens[..101]),
    ] {
        // Serial-scan prefill, computed once and held fixed.
        let prefill_snap = gdn_prefill_snapshot(&mut state, tokens, false);

        // (worst layer, its rel_L2, layers over the threshold, overall max_abs)
        let mut best: Option<(usize, f64, usize, f32)> = None;
        let mut best_layer_rel_l2: Vec<f64> = Vec::new();
        for attempt in 0..3 {
            // Token-by-token decode reference.
            state.use_gdn_chunked = false;
            state.reset_state();
            for (pos, &tok) in tokens.iter().enumerate() {
                let _ = state.forward_step(tok, pos);
            }
            let decode_snap = state.snapshot_gdn_states();
            state.use_gdn_chunked = true;

            let d = gdn_snapshot_diff(&prefill_snap, &decode_snap);
            let mut worst_layer = 0usize;
            let mut worst_rel = 0.0f64;
            for (layer, &rel) in d.layer_rel_l2.iter().enumerate() {
                if rel > worst_rel {
                    worst_layer = layer;
                    worst_rel = rel;
                }
            }
            let layers_over = d
                .layer_rel_l2
                .iter()
                .filter(|rel| **rel >= MAX_LAYER_REL_L2_THRESHOLD)
                .count();
            eprintln!(
                "State diff prefill vs decode prompt={label} attempt {attempt}: \
                         max_layer_rel_l2={worst_rel:.3e} (layer {worst_layer}), \
                         max_abs={:.6}",
                d.max_abs
            );
            // A passing attempt is kept as the result; otherwise the closest one is.
            if layers_over == 0 && d.max_abs < MAX_ABS_THRESHOLD {
                best = Some((worst_layer, worst_rel, layers_over, d.max_abs));
                best_layer_rel_l2 = d.layer_rel_l2;
                break;
            }
            let better = best.map(|(_, w, _, _)| worst_rel < w).unwrap_or(true);
            if better {
                best = Some((worst_layer, worst_rel, layers_over, d.max_abs));
                best_layer_rel_l2 = d.layer_rel_l2;
            }
        }
        let (worst_layer, worst_rel, layers_over, max_abs) = best.expect("at least one attempt");
        let per_layer: Vec<String> = best_layer_rel_l2
            .iter()
            .map(|rel| format!("{rel:.3e}"))
            .collect();
        eprintln!(
            "State diff prefill vs decode prompt={label} tokens={} (best of 3): \
                     max_layer_rel_l2={worst_rel:.3e} (layer {worst_layer}), max_abs={max_abs:.6}, \
                     per_layer_rel_l2=[{}]",
            tokens.len(),
            per_layer.join(", ")
        );
        if layers_over > 0 || max_abs >= MAX_ABS_THRESHOLD {
            failures.push(format!(
                "prompt={label} tokens={}: worst layer {worst_layer} rel_L2={worst_rel:.3e} \
                         (threshold {MAX_LAYER_REL_L2_THRESHOLD:.0e}, {layers_over} layer(s) over), \
                         max_abs={max_abs:.6} (threshold {MAX_ABS_THRESHOLD:.0e})",
                tokens.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "GDN state S diverges (prefill vs decode): {}",
        failures.join("; ")
    );
}

/// Boundary-sweep parity gate: chunked-GDN `forward_prefill` vs serial `forward_prefill`
/// must produce matching argmax at EVERY position for prompts that straddle the C=32 chunk
/// boundary and the 512 max_prefill boundary.
///
/// Both paths are selected via `state.use_gdn_chunked` (no process-env mutation), so there
/// are no setenv races. GPU access is additionally serialized via `gpu_test_lock()` so the
/// pre-existing batched/decode attention race (root cause unconfirmed; see
/// `gpu_test_lock`'s doc comment) is not amplified by concurrent device contention,
/// and the deterministic chunked logits are computed once per length while only the serial
/// reference is retried — a flaky chunked attempt cannot be masked by best-of-N.
///
/// This is a LOCAL/manual gate: `cargo test -p lattice-inference gdn_chunked
/// --features "f16,metal-gpu" -- --nocapture` with the model present.  The production
/// cross-framework gate is `e2e-parity.yml`, which runs HF-vs-lattice greedy-token parity
/// on macOS and includes an 816-token oversize prompt that forces the >512 chunked path.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn gdn_chunked_prefill_vs_serial_prefill_logit_parity() {
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir) {
        eprintln!("skipping gdn_chunked_prefill_vs_serial_prefill_logit_parity: model missing");
        return;
    }
    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let Some(_device) = Device::system_default() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let _gpu = gpu_test_lock();

    let max_ctx = 1024usize;
    let mut state = MetalQwen35State::new(model.weights(), model.config(), max_ctx)
        .expect("construct Metal state");
    let vocab = model.config().vocab_size;

    // Token source: a long real-text prompt; truncate to each sweep length below.
    let base_tokens = long_real_text_tokens(model.tokenizer());

    // Sweep lengths chosen to straddle C=32 chunk boundaries and the 512 max_prefill
    // boundary.  Skip lengths that exceed the model's max context.
    let sweep_lengths: &[usize] = &[1, 31, 32, 33, 64, 511, 512, 513, 1009];

    // Evidence table: (len, all-position max_abs_diff, argmax flip count, attempts used)
    let mut evidence: Vec<(usize, f32, usize, usize)> = Vec::new();
    let mut any_flip = false;

    // The chunked scan is deterministic (gdn_chunked_b_vs_b_self_consistency), but
    // chunked-vs-serial all-position value drift grows with prompt length even after
    // best-of-N discards the batched/decode attention race's serial-reference outliers
    // (root cause unconfirmed; see `gpu_test_lock`'s doc comment): best-of-5 max_abs_diff
    // reaches ~4.8e-2 at n=1009 and crosses the old 1e-2 bound at n=511/512/513
    // (#534). The value bound below is a drift sentinel, not the correctness gate —
    // the argmax-flip assertion after the sweep is the hard safety check. Tighten
    // this bound if a future kernel change brings chunked-vs-serial drift back down.
    const MAX_ABS_BOUND: f32 = 6.5e-2;
    const ATTEMPTS: usize = 5;

    for &n in sweep_lengths {
        if n > max_ctx {
            eprintln!("  len={n}: skipped (exceeds max_ctx={max_ctx})");
            continue;
        }
        // Pad base_tokens with repeated token 1 if needed.
        let tokens: Vec<u32> = if n <= base_tokens.len() {
            base_tokens[..n].to_vec()
        } else {
            let mut v = base_tokens.clone();
            v.resize(n, 1u32);
            v
        };

        // Chunked path (chunked ON) — the path under test.  Deterministic
        // (gdn_chunked_b_vs_b_self_consistency), so compute it exactly ONCE and hold it
        // fixed; the best-of-N below re-samples only the serial reference.  This is what
        // makes the gate sound: a flaky chunked result would diverge on every attempt and
        // fail the gate, never get masked by a lucky retry.
        state.use_gdn_chunked = true;
        state.reset_state();
        let chunked_all = state
            .forward_prefill_all_logits(&tokens)
            .expect("LoRA-inactive chunked all-position prefill succeeds");
        assert_eq!(
            chunked_all.len(),
            n * vocab,
            "chunked all_logits len mismatch at n={n}"
        );

        // Best-of-N: the serial reference is perturbed by the batched/decode attention
        // race (root cause unconfirmed; see `gpu_test_lock`'s doc comment) on a minority
        // of runs.  Keep the attempt with the smallest max_abs (and its flip count); a
        // single clean comparison proves the chunked algorithm correct, so stop early once
        // one attempt is within bound and flip-free.  Mirrors
        // gdn_chunked_state_vs_serial_state_diff.
        let mut best_max_abs = f32::MAX;
        let mut best_flips = usize::MAX;
        let mut attempts_used = 0usize;
        for attempt in 0..ATTEMPTS {
            // Serial path (chunked OFF): per-position logits via all_logits.
            state.use_gdn_chunked = false;
            state.reset_state();
            let serial_all = state
                .forward_prefill_all_logits(&tokens)
                .expect("LoRA-inactive serial all-position prefill succeeds");
            assert_eq!(
                serial_all.len(),
                n * vocab,
                "serial all_logits len mismatch at n={n}"
            );

            // Compare at every position.
            let mut max_abs = 0.0f32;
            let mut flips = 0usize;
            for pos in 0..n {
                let s = &serial_all[pos * vocab..(pos + 1) * vocab];
                let c = &chunked_all[pos * vocab..(pos + 1) * vocab];
                let pos_max = s
                    .iter()
                    .zip(c.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                max_abs = max_abs.max(pos_max);
                if argmax_f32(s) != argmax_f32(c) {
                    flips += 1;
                }
            }
            attempts_used = attempt + 1;
            eprintln!(
                "  len={n:4} attempt {attempt}: max_abs_diff={max_abs:.2e}  argmax_flips={flips}"
            );

            if max_abs < best_max_abs {
                best_max_abs = max_abs;
                best_flips = flips;
            }
            if best_max_abs < MAX_ABS_BOUND && best_flips == 0 {
                break;
            }
        }

        eprintln!(
            "  len={n:4}: best-of-{attempts_used} all-pos max_abs_diff={best_max_abs:.2e}  argmax_flips={best_flips}"
        );
        evidence.push((n, best_max_abs, best_flips, attempts_used));
        if best_flips > 0 {
            any_flip = true;
        }
    }

    // Restore default after sweep.
    state.use_gdn_chunked = true;

    eprintln!("Evidence table (boundary sweep, per-instance flag, no env mutation):");
    eprintln!("  len | all-pos max_abs_diff | argmax_flips | attempts");
    for (n, d, f, a) in &evidence {
        eprintln!("  {n:4} | {d:.2e}             | {f:<12} | {a}");
    }

    // Assert no argmax flips across all lengths and positions.
    assert!(
        !any_flip,
        "argmax flips detected across boundary sweep — default-on is unsafe. \
                 Evidence: {evidence:?}"
    );

    // Assert all-position max_abs_diff stays within the evidence-based drift sentinel.
    for (n, max_abs, _, _) in &evidence {
        assert!(
            *max_abs < MAX_ABS_BOUND,
            "len={n}: all-position max_abs_diff={max_abs:.2e} exceeds #534 drift sentinel {MAX_ABS_BOUND:.2e}"
        );
    }
}

/// gdn175 S1b: revalidate the chunkwise WY/DPLR GDN oracle
/// (`gdn_chunk_ref::{sequential_gdn, chunkwise_gdn}`, validated to <=1e-5 on
/// synthetic fixtures in S1a / PR #715) against REAL Qwen3.5-0.8B GDN-layer
/// recurrence inputs captured from a live prefill. Synthetic fixtures draw
/// i.i.d. random keys; real activations have correlated keys, which
/// conditions the chunkwise triangular solve differently — this is the last
/// S1 gate before any kernel work (S2).
///
/// KILL criterion: any real-input max_abs > 1e-5 means the math has a
/// conditioning problem the synthetic fixtures missed. This test does NOT
/// loosen the tolerance to compensate — a failure here must be reported,
/// not patched around.
///
/// `(max_abs, nan_count, inf_count)` over `|a[i]-b[i]|`. Deliberately NOT a
/// plain `.fold(0.0, f32::max)`: `f32::max` implements IEEE-754 maxNum,
/// which SILENTLY returns the non-NaN operand when one side is NaN — a
/// diff array containing NaNs would report a possibly-tiny max_abs while
/// hiding the NaNs entirely. `nan_count`/`inf_count` make that failure mode
/// visible instead of swallowed.
#[cfg(feature = "bench-internals")]
fn diff_stats(a: &[f32], b: &[f32]) -> (f32, usize, usize) {
    let mut max_abs = 0.0f32;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d.is_nan() {
            nan_count += 1;
        } else if d.is_infinite() {
            inf_count += 1;
            max_abs = f32::INFINITY;
        } else if d > max_abs {
            max_abs = d;
        }
    }
    (max_abs, nan_count, inf_count)
}

#[cfg(feature = "bench-internals")]
#[test]
fn gdn175_s1b_chunkwise_oracle_real_activations() {
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir) {
        eprintln!("skipping gdn175_s1b_chunkwise_oracle_real_activations: model missing");
        return;
    }
    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let Some(_device) = Device::system_default() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let _gpu = gpu_test_lock();

    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal state");
    assert!(
        bench_support::gdn_chunked_prefill_supported(&state),
        "qwen3.5-0.8b checkpoint at {} does not support the chunked GDN prefill path",
        model_dir.display()
    );

    // ~300 real tokens: ceil(300/32)=10 chunks incl. a partial 12-token tail
    // (S1b spec: ~200-400 tokens so there are 6-12 chunks incl. a partial tail).
    let tokens = {
        let mut t = long_real_text_tokens(model.tokenizer());
        assert!(
            t.len() >= 300,
            "long_real_text_tokens produced only {} tokens, need >=300",
            t.len()
        );
        t.truncate(300);
        t
    };
    let length = tokens.len();

    // Early/mid/late GDN ("linear_attention") layer indices — `linear_idx` is
    // the 0-based index among GDN layers only, computed from the checkpoint's
    // own layer_types rather than hardcoded, so this stays correct if a future
    // checkpoint changes the hybrid layer pattern.
    let num_linear = model
        .config()
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    assert!(
        num_linear >= 3,
        "need >=3 GDN layers for early/mid/late targets, found {num_linear}"
    );
    let layer_targets = [0usize, num_linear / 2, num_linear - 1];
    let head_targets = [0usize, 8usize];
    let num_vh = model.config().linear_num_value_heads();
    for &h in &head_targets {
        assert!(
            h < num_vh,
            "head target {h} out of range (num_value_heads={num_vh})"
        );
    }

    let capture_requests: Vec<bench_support::GdnCaptureRequest> = layer_targets
        .iter()
        .flat_map(|&linear_idx| {
            head_targets
                .iter()
                .map(move |&value_head| bench_support::GdnCaptureRequest {
                    linear_idx,
                    value_head,
                })
        })
        .collect();

    state.use_gdn_chunked = true;
    state.reset_state();
    let (_timing, captures) = bench_support::forward_prefill_gdn_isolated_chunk_capture(
        &mut state,
        &tokens,
        0,
        &capture_requests,
    )
    .expect("fresh benchmark state and in-range capture request");
    assert_eq!(
        captures.len(),
        capture_requests.len(),
        "capture count mismatch — a requested (layer, head) target never matched a \
                 processed GDN layer (layer_targets={layer_targets:?})"
    );

    const TOL: f32 = 1.0e-5;
    const D4_WATCH: f32 = 0.5 * TOL; // near-but-under-tolerance watch threshold

    let mut round_trip_report: Vec<(usize, usize, usize, f32, f32)> = Vec::new();
    // (layer, head, C, max_abs_out, max_abs_h, nan_out, inf_out, nan_h, inf_h)
    #[allow(clippy::type_complexity)]
    let mut table: Vec<(usize, usize, usize, f32, f32, usize, usize, usize, usize)> = Vec::new();
    // (layer, head, log_alpha_min, log_alpha_max, alpha_min, alpha_max,
    // gamma_at_c32_min) — diagnostic evidence for whether a real-input
    // divergence is decay-underflow conditioning (KILL-relevant) vs a
    // capture/transcription bug.
    let mut alpha_stats: Vec<(usize, usize, f32, f32, f32, f32, f32)> = Vec::new();
    let mut any_d4_watch = false;
    let mut failures: Vec<String> = Vec::new();

    for cap in &captures {
        assert_eq!(cap.length, length);
        let dk = cap.dk;
        let dv = cap.dv;

        // Round-trip sanity (S1b spec): ln(alpha[t]) must equal the captured
        // log_alpha[t] within 1e-6, sampled at a few tokens across the sequence.
        let sample_idx = [0usize, length / 2, length - 1];
        for &t in &sample_idx {
            let la = cap.log_alpha[t];
            let alpha = la.exp();
            let round_trip = alpha.ln();
            let diff = (round_trip - la).abs();
            round_trip_report.push((cap.linear_idx, cap.value_head, t, la, diff));
            if diff.is_nan() || diff > 1e-6 {
                failures.push(format!(
                    "ROUND-TRIP: layer={} head={} t={}: ln(exp(log_alpha))={} vs \
                             log_alpha={} diff={} exceeds 1e-6 round-trip bound",
                    cap.linear_idx, cap.value_head, t, round_trip, la, diff
                ));
            }
        }

        // The oracle feed transform (S1b spec): per-token alpha[t] =
        // exp(log_alpha[t]) — NEVER the gamma buffer or a cumulative
        // log-alpha; chunkwise_gdn recomputes gamma = cumprod(alpha) itself.
        let alpha: Vec<f32> = cap.log_alpha.iter().map(|x| x.exp()).collect();
        let h0 = vec![0.0f32; dv * dk];
        let (seq_out, seq_h) = crate::forward::gdn_chunk_ref::sequential_gdn(
            &cap.q, &cap.k, &cap.v, &cap.beta, &alpha, &h0, length, dk, dv, true,
        );

        // Diagnostic: min/max of log_alpha and alpha, plus the smallest
        // per-chunk cumulative product gamma would reach at C=32 (the
        // kernel's own chunk size) — a near-zero gamma means the LINEAR-
        // space cumprod the CPU oracle computes (`chunkwise_gdn`'s
        // `running *= a_c[i]`) is underflowing, which is a fundamentally
        // different failure mode than a solve-conditioning error: the
        // kernel's log-alpha floor (-88, see qwen35.metal
        // gdn_chunk_materialize_c32) only keeps ITS OWN log-space
        // cumulative scan finite; it says nothing about a linear-space
        // cumprod reference computing 1.0/gamma downstream.
        let log_alpha_min = cap.log_alpha.iter().copied().fold(f32::INFINITY, f32::min);
        let log_alpha_max = cap
            .log_alpha
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let alpha_min = alpha.iter().copied().fold(f32::INFINITY, f32::min);
        let alpha_max = alpha.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut gamma32_min = f32::INFINITY;
        for chunk_start in (0..length).step_by(32) {
            let chunk_end = (chunk_start + 32).min(length);
            let mut running = 1.0f32;
            for &a in &alpha[chunk_start..chunk_end] {
                running *= a;
            }
            gamma32_min = gamma32_min.min(running);
        }
        alpha_stats.push((
            cap.linear_idx,
            cap.value_head,
            log_alpha_min,
            log_alpha_max,
            alpha_min,
            alpha_max,
            gamma32_min,
        ));

        for &chunk_size in &[32usize, 64, 128] {
            let (chk_out, chk_h) = crate::forward::gdn_chunk_ref::chunkwise_gdn(
                &cap.q, &cap.k, &cap.v, &cap.beta, &alpha, &h0, length, dk, dv, chunk_size, true,
            );
            let (max_abs_out, nan_out, inf_out) = diff_stats(&chk_out, &seq_out);
            let (max_abs_h, nan_h, inf_h) = diff_stats(&chk_h, &seq_h);

            table.push((
                cap.linear_idx,
                cap.value_head,
                chunk_size,
                max_abs_out,
                max_abs_h,
                nan_out,
                inf_out,
                nan_h,
                inf_h,
            ));

            let worst = max_abs_out.max(max_abs_h);
            if worst.is_finite() && worst > D4_WATCH && worst <= TOL && nan_out == 0 && nan_h == 0 {
                any_d4_watch = true;
                eprintln!(
                    "D4-WATCH: layer={} head={} C={} max_abs_out={:.3e} \
                             max_abs_h={:.3e} is near (>{:.1e}) but under the {:.1e} KILL \
                             tolerance — candidate evidence for the D4 f32-accumulator \
                             mandate.",
                    cap.linear_idx,
                    cap.value_head,
                    chunk_size,
                    max_abs_out,
                    max_abs_h,
                    D4_WATCH,
                    TOL
                );
            }

            if max_abs_out > TOL || nan_out > 0 {
                failures.push(format!(
                    "KILL: layer={} head={} C={} chunkwise_gdn out diverged from \
                             sequential_gdn on REAL activations: max_abs={:.3e} > {:.1e} \
                             (nan_count={nan_out}, inf_count={inf_out})",
                    cap.linear_idx, cap.value_head, chunk_size, max_abs_out, TOL
                ));
            }
            if max_abs_h > TOL || nan_h > 0 {
                failures.push(format!(
                    "KILL: layer={} head={} C={} chunkwise_gdn h diverged from \
                             sequential_gdn on REAL activations: max_abs={:.3e} > {:.1e} \
                             (nan_count={nan_h}, inf_count={inf_h})",
                    cap.linear_idx, cap.value_head, chunk_size, max_abs_h, TOL
                ));
            }
        }
    }

    eprintln!(
        "gdn175 S1b: real-activation chunkwise-vs-sequential oracle parity \
                 (length={length}, {} layers x {} heads x 3 chunk sizes):",
        layer_targets.len(),
        head_targets.len()
    );
    eprintln!(
        "  layer | head | C   | max_abs_out | max_abs_h  | nan_out | inf_out | nan_h | inf_h"
    );
    for (layer, head, c, out, h, nan_out, inf_out, nan_h, inf_h) in &table {
        eprintln!(
            "  {layer:5} | {head:4} | {c:3} | {out:.3e}    | {h:.3e}   | \
                     {nan_out:7} | {inf_out:7} | {nan_h:5} | {inf_h:5}"
        );
    }
    eprintln!(
        "gdn175 S1b: per-(layer,head) decay diagnostics (log_alpha/alpha range, \
                 min cumulative gamma over any C=32 window):"
    );
    eprintln!(
        "  layer | head | log_alpha_min | log_alpha_max | alpha_min  | alpha_max | gamma32_min"
    );
    for (layer, head, lamin, lamax, amin, amax, g32) in &alpha_stats {
        eprintln!(
            "  {layer:5} | {head:4} | {lamin:.6e}   | {lamax:.6e}   | {amin:.3e} | {amax:.3e} | {g32:.3e}"
        );
    }
    eprintln!("gdn175 S1b: log_alpha round-trip sanity (ln(exp(log_alpha)) vs log_alpha):");
    eprintln!("  layer | head | t   | log_alpha    | diff");
    for (layer, head, t, la, diff) in &round_trip_report {
        eprintln!("  {layer:5} | {head:4} | {t:3} | {la:.6e} | {diff:.3e}");
    }
    if any_d4_watch {
        eprintln!(
            "gdn175 S1b: one or more cells landed near (but under) the 1e-5 KILL \
                     tolerance — see D4-WATCH lines above; this indicates a D4 f32-accumulator \
                     issue."
        );
    } else {
        eprintln!("gdn175 S1b: no cells near the 1e-5 tolerance boundary.");
    }

    assert!(
        failures.is_empty(),
        "gdn175 S1b: {} failing cell(s) — conditioning problem the synthetic \
                 fixtures missed, do NOT loosen tolerance:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Localization probe: run the SERIAL token-by-token path twice and diff the final GDN
/// state.  If the serial reference is itself nondeterministic, the chunked-vs-serial state
/// diff cannot isolate the chunked kernels — the race lives in shared input producers
/// (gdn_qkv / hidden / in_proj), not the chunked scan.  Also runs chunked twice for
/// comparison.  Diagnostic only — does not assert.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn gdn_chunked_race_localization_probe() {
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir) {
        eprintln!("skipping: model missing");
        return;
    }
    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let Some(_device) = Device::system_default() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let _gpu = gpu_test_lock();
    use crate::speculative::MtpTargetVerifier as _;
    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal state");
    let tokens = long_real_text_tokens(model.tokenizer());

    let snap_state = |st: &MetalQwen35State| -> Vec<f32> {
        st.snapshot_gdn_states()
            .into_iter()
            .flat_map(|(s, _conv)| s)
            .collect()
    };
    let max_diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    // Serial path, twice (chunked OFF via per-instance flag).
    state.use_gdn_chunked = false;
    state.reset_state();
    for (pos, &tok) in tokens.iter().enumerate() {
        let _ = state.forward_step(tok, pos);
    }
    let serial_a = snap_state(&state);
    state.reset_state();
    for (pos, &tok) in tokens.iter().enumerate() {
        let _ = state.forward_step(tok, pos);
    }
    let serial_b = snap_state(&state);

    // Batched prefill with chunked GDN OFF (batched producers + serial GDN consumer), twice.
    state.use_gdn_chunked = false;
    state.reset_state();
    let _ = state.forward_prefill(&tokens);
    let prefill_nochunk_a = snap_state(&state);
    state.reset_state();
    let _ = state.forward_prefill(&tokens);
    let prefill_nochunk_b = snap_state(&state);

    // Chunked path, twice (chunked ON via per-instance flag).
    state.use_gdn_chunked = true;
    state.reset_state();
    let _ = state.forward_prefill(&tokens);
    let chunked_a = snap_state(&state);
    state.reset_state();
    let _ = state.forward_prefill(&tokens);
    let chunked_b = snap_state(&state);

    eprintln!(
        "RACE-LOCALIZATION: serial-vs-serial (forward_step) state max_abs_diff = {:.6}",
        max_diff(&serial_a, &serial_b)
    );
    eprintln!(
        "RACE-LOCALIZATION: prefill-NOCHUNK-vs-NOCHUNK (batched producer + serial GDN) max_abs_diff = {:.6}",
        max_diff(&prefill_nochunk_a, &prefill_nochunk_b)
    );
    eprintln!(
        "RACE-LOCALIZATION: chunked-vs-chunked state max_abs_diff = {:.6}",
        max_diff(&chunked_a, &chunked_b)
    );
    eprintln!(
        "RACE-LOCALIZATION: chunked-vs-serial state max_abs_diff = {:.6}",
        max_diff(&chunked_a, &serial_a)
    );
}

// =====================================================================
// #850: GDN Q/K L2-normalization fail-closed — direct Metal kernel
// dispatch regression tests. No model checkpoint dependency; the only
// permitted skip is "no Metal device" (per #850's gate text: "no
// silent skip on a missing device or optional Qwen3.6 pipeline" —
// these tests build their own pipelines directly from MSL_SOURCE, so
// there is no optional-pipeline skip path at all).
//
// Methodology: for each of the 8 sites (4 kernels x Q/K), dispatch the
// SAME kernel twice with otherwise-identical inputs: once with one NaN
// lane injected into the Q (or K) input, once with that entire Q (or
// K) sub-vector explicitly zeroed instead. Per ADR-080 C1 + the #850
// contract (an invalid Q/K vector is assigned literal 0.0, matching
// ordinary zero-weight arithmetic), a correct kernel produces
// BIT-IDENTICAL output between the two runs. Since #1583 the two runs
// reach that identical output by DIFFERENT routes, which is worth
// stating because the earlier "both take the same branch" reasoning is
// no longer what makes this test valid: the poisoned run has a
// non-finite sum_sq and takes the fail-closed `assign 0.0` arm, while
// the explicitly-zeroed run has sum_sq == 0.0, which is finite, so it
// takes the ORDINARY arm and computes `0.0 * rsqrt(0.0 + 1e-6)` — an
// exact zero for every lane, because the input vector is all zeros.
// The assertion is unchanged and still discriminating; only its
// justification moved. This is a Metal-vs-Metal (poisoned-vs-explicit-zero-reference)
// differential, not a cross-language CPU-vs-Metal one; see the PR body
// for why that scope was chosen. Mutation-sensitive: the pre-fix
// `value * guarded_zero_reciprocal` code lets the poisoned lane itself
// stay NaN while every OTHER lane goes to `finite * 0.0 = 0.0`, so the
// poisoned run diverges from the all-zero reference at exactly that
// lane -- the bit-identical assertion below fails under the reverted
// code (see the reverse-mutation proof in the PR body).
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod gdn_850_failclosed {
    use super::*;

    /// Matches the MSL `GdnRecurParams` struct used by
    /// `gdn_recurrence_fused`, `gdn_recurrence_fused_q36`, and
    /// `gdn_precompute_keys` (field-for-field, see qwen35.metal:996-1005).
    #[repr(C)]
    struct TestGdnRecurParams {
        key_dim: u32,
        value_dim: u32,
        num_key_heads: u32,
        num_value_heads: u32,
        hidden_size: u32,
        q_total: u32,
        v_offset: u32,
        scale: f32,
        eps: f32,
    }

    fn compile_msl(device: &Device) -> Library {
        let opts = CompileOptions::new();
        device
            .new_library_with_source(MSL_SOURCE, &opts)
            .expect("MSL_SOURCE must compile")
    }

    fn pipeline_for(device: &Device, lib: &Library, name: &str) -> ComputePipelineState {
        let func = lib
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("kernel {name} not found in MSL_SOURCE: {e}"));
        device
            .new_compute_pipeline_state_with_function(&func)
            .unwrap_or_else(|e| panic!("failed to build pipeline for {name}: {e}"))
    }

    fn f32_buf(device: &Device, data: &[f32], label: &str) -> Buffer {
        let buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        buf
    }

    fn read_f32(buf: &Buffer, len: usize) -> Vec<f32> {
        // SAFETY: buffer is StorageModeShared, previously written by a
        // completed command buffer, and sized for `len` f32 elements.
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
    }

    fn lcg_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((state >> 32) as u32) as f32 / u32::MAX as f32 - 0.5) * 0.2
            })
            .collect()
    }

    fn assert_no_nonfinite(label: &str, buf_name: &str, data: &[f32]) {
        let bad = data.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            bad,
            0,
            "{label}: {buf_name} must fail closed (finite), got {bad} non-finite \
                     element(s) out of {}: {data:?}",
            data.len()
        );
    }

    // --------------------------------------------------------------
    // Kernel 1: gdn_recurrence_fused. Small synthetic dims (kd=vd=8,
    // hd=16, 1 key head, 1 value head) -- this kernel takes kd/vd/hd
    // as runtime params (bounds-checked via `tid < kd`/`tid < vd`), so
    // shrinking them is safe and keeps buffers tiny.
    // --------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn run_gdn_recurrence_fused(
        device: &Device,
        queue: &CommandQueue,
        pipe: &ComputePipelineState,
        kd: u32,
        vd: u32,
        hd: u32,
        conv_out: &[f32],
        s_init: &[f32],
        hidden_in: &[f32],
        in_proj_b: &[f32],
        in_proj_a: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let q_total = kd;
        let v_offset = q_total * 2;
        let output_dim = vd;

        let s_buf = f32_buf(device, s_init, "S_all");
        let conv_buf = f32_buf(device, conv_out, "conv_out");
        // z_proj was all zeros here, which made `silu(z)` zero and therefore
        // `output` identically zero for every input, whatever the kernel did.
        // Any assertion on `output` was vacuous, and since the state update
        // reads only k_tg/delta/g and never q_tg, the Q path had NO observable
        // at all. The offset keeps every lane's gate comfortably away from 0.
        let z_vals: Vec<f32> = lcg_vec(6, output_dim as usize)
            .iter()
            .map(|v| v + 0.5)
            .collect();
        let z_buf = f32_buf(device, &z_vals, "z_proj");
        let hidden_buf = f32_buf(device, hidden_in, "hidden_in");
        let inb_buf = make_buffer_f16(device, in_proj_b, "in_proj_b");
        let ina_buf = make_buffer_f16(device, in_proj_a, "in_proj_a");
        let alog_buf = f32_buf(device, &[-1.0f32], "a_log");
        let dtb_buf = f32_buf(device, &[0.0f32], "dt_bias");
        let normw_buf = f32_buf(device, &vec![1.0f32; vd as usize], "norm_w");
        let out_buf = make_zero_buffer(device, output_dim as usize, "output");

        let params = TestGdnRecurParams {
            key_dim: kd,
            value_dim: vd,
            num_key_heads: 1,
            num_value_heads: 1,
            hidden_size: hd,
            q_total,
            v_offset,
            scale: 1.0 / (kd as f32).sqrt(),
            eps: 1e-6,
        };

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&s_buf), 0);
        enc.set_buffer(1, Some(&conv_buf), 0);
        enc.set_buffer(2, Some(&z_buf), 0);
        enc.set_buffer(3, Some(&hidden_buf), 0);
        enc.set_buffer(4, Some(&inb_buf), 0);
        enc.set_buffer(5, Some(&ina_buf), 0);
        enc.set_buffer(6, Some(&alog_buf), 0);
        enc.set_buffer(7, Some(&dtb_buf), 0);
        enc.set_buffer(8, Some(&normw_buf), 0);
        enc.set_buffer(9, Some(&out_buf), 0);
        enc.set_bytes(
            10,
            std::mem::size_of::<TestGdnRecurParams>() as u64,
            &params as *const TestGdnRecurParams as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 4, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        (
            read_f32(&out_buf, output_dim as usize),
            read_f32(&s_buf, (kd * vd) as usize),
        )
    }

    #[test]
    fn gdn_recurrence_fused_fails_closed_on_nan_q_and_k_lane() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping gdn_recurrence_fused_fails_closed: no Metal device");
            return;
        };
        let _gpu = gpu_test_lock();
        let lib = compile_msl(&device);
        let pipe = pipeline_for(&device, &lib, "gdn_recurrence_fused");
        let queue = device.new_command_queue();

        let kd = 8u32;
        let vd = 8u32;
        let hd = 16u32;
        let hidden_in = lcg_vec(1, hd as usize);
        let in_proj_b = lcg_vec(2, hd as usize);
        let in_proj_a = lcg_vec(3, hd as usize);
        let s_init = lcg_vec(4, (kd * vd) as usize);

        for label in ["Q", "K"] {
            let base = lcg_vec(5, (3 * kd) as usize); // [Q(kd) | K(kd) | V(kd)]
            let mut poisoned = base.clone();
            let mut reference = base.clone();
            let (lo, hi) = if label == "Q" {
                (0usize, kd as usize)
            } else {
                (kd as usize, 2 * kd as usize)
            };
            poisoned[lo] = f32::NAN;
            for v in reference[lo..hi].iter_mut() {
                *v = 0.0;
            }

            // Stress/repeat coverage for the #862 WAR-hazard
            // fix (Q->K threadgroup_barrier after every sg_buf[0] consumer):
            // a passing single dispatch does not prove the race is closed
            // (simdgroups are not required to advance in lockstep, so the
            // hazard window may or may not be hit on a given run), so this
            // repeats the dispatch to raise the chance of exposing it if the
            // barrier were ever removed again. Not a substitute for the
            // access-order argument the barrier itself is required by.
            for _ in 0..20 {
                let (out_p, s_p) = run_gdn_recurrence_fused(
                    &device, &queue, &pipe, kd, vd, hd, &poisoned, &s_init, &hidden_in, &in_proj_b,
                    &in_proj_a,
                );
                let (out_r, s_r) = run_gdn_recurrence_fused(
                    &device, &queue, &pipe, kd, vd, hd, &reference, &s_init, &hidden_in,
                    &in_proj_b, &in_proj_a,
                );

                assert_no_nonfinite(&format!("gdn_recurrence_fused[{label}]"), "output", &out_p);
                assert_no_nonfinite(&format!("gdn_recurrence_fused[{label}]"), "S_all", &s_p);
                assert_eq!(
                    out_p, out_r,
                    "gdn_recurrence_fused[{label}-poisoned]: output must be bit-identical \
                             to the never-poisoned (explicit-zero-{label}) reference"
                );
                assert_eq!(
                    s_p, s_r,
                    "gdn_recurrence_fused[{label}-poisoned]: updated S_all must be \
                             bit-identical to the never-poisoned (explicit-zero-{label}) reference"
                );
            }
        }
    }

    /// #1583, divergence 1: the Metal Q/K L2 normalisation must carry the same
    /// epsilon the CPU reference uses, which is observable as a LOSS of scale
    /// invariance.
    ///
    /// This is the discriminating property, and it needs no CPU reimplementation of
    /// the recurrence -- which matters, because reimplementing it would make the
    /// test agree with my own port rather than with the kernel. `q / sqrt(sum_sq)`
    /// is homogeneous by construction: scaling the Q sub-vector by any positive `c`
    /// leaves the normalised vector, and therefore the whole kernel output,
    /// unchanged up to rounding. `q / sqrt(sum_sq + eps)` is NOT homogeneous,
    /// because the epsilon has a fixed scale. So scaling only the Q slice and
    /// requiring the output to MOVE is exactly an assertion that the epsilon is
    /// there.
    ///
    /// Mutation sensitivity is structural rather than incidental: reverting the
    /// eight sites to `rsqrt(sg_buf[0])` restores exact homogeneity, and the two
    /// runs collapse to within f32 rounding of each other on every lane at once.
    #[test]
    fn gdn_qk_l2_norm_is_not_scale_invariant_1583() {
        // Fail closed under LATTICE_METAL_TEST_ENFORCE=1, matching the other
        // Metal tests CI names explicitly: without this the CI step added for
        // this test would pass on a runner with no Metal device, which is the
        // one condition that makes the step worth having.
        let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
        let Some(device) = Device::system_default() else {
            assert!(
                !enforce,
                "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present (gdn_qk_l2_norm_is_not_scale_invariant_1583)"
            );
            return;
        };
        let _gpu = gpu_test_lock();
        let lib = compile_msl(&device);
        let pipe = pipeline_for(&device, &lib, "gdn_recurrence_fused");
        let queue = device.new_command_queue();

        let kd = 8u32;
        let vd = 8u32;
        let hd = 16u32;
        let hidden_in = lcg_vec(1, hd as usize);
        let in_proj_b = lcg_vec(2, hd as usize);
        let in_proj_a = lcg_vec(3, hd as usize);
        let s_init = lcg_vec(4, (kd * vd) as usize);

        // Chosen so the two runs sit on OPPOSITE sides of the epsilon's influence:
        // the unscaled sum_sq is far above 1e-6, the scaled one far below it. That
        // makes the predicted gap order-1 instead of marginal, so the assertion
        // does not depend on a finely-tuned threshold.
        const C: f32 = 1e-4;

        for label in ["Q", "K"] {
            let base = lcg_vec(5, (3 * kd) as usize); // [Q(kd) | K(kd) | V(kd)]
            let (lo, hi) = if label == "Q" {
                (0usize, kd as usize)
            } else {
                (kd as usize, 2 * kd as usize)
            };

            // The fixture's own preconditions, asserted rather than assumed: a
            // fixture that drifted to either side of the epsilon would make this
            // test pass or fail for a reason that has nothing to do with the code.
            let sum_sq: f32 = base[lo..hi].iter().map(|v| v * v).sum();
            assert!(
                sum_sq > 1e-3,
                "{label}: fixture sum_sq {sum_sq} is not comfortably above the 1e-6 epsilon"
            );
            assert!(
                sum_sq * C * C < 1e-7,
                "{label}: scaled sum_sq {} is not comfortably below the 1e-6 epsilon",
                sum_sq * C * C
            );

            let mut scaled = base.clone();
            for v in scaled[lo..hi].iter_mut() {
                *v *= C;
            }

            let (out_base, _) = run_gdn_recurrence_fused(
                &device, &queue, &pipe, kd, vd, hd, &base, &s_init, &hidden_in, &in_proj_b,
                &in_proj_a,
            );
            let (out_scaled, _) = run_gdn_recurrence_fused(
                &device, &queue, &pipe, kd, vd, hd, &scaled, &s_init, &hidden_in, &in_proj_b,
                &in_proj_a,
            );
            assert_no_nonfinite(label, "output(base)", &out_base);
            assert_no_nonfinite(label, "output(scaled)", &out_scaled);

            let max_abs = out_base.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let max_delta = out_base
                .iter()
                .zip(out_scaled.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs > 1e-6,
                "{label}: baseline output is ~zero (max_abs {max_abs}); this fixture \
                         cannot express a difference and the comparison below would be vacuous"
            );
            let rel = max_delta / max_abs;

            // Rounding noise between the two arms is order 1e-6 relative; the
            // epsilon's effect here is order 1. 1e-2 separates them with several
            // orders of margin on both sides.
            assert!(
                rel > 1e-2,
                "{label}: scaling the {label} sub-vector by {C} moved the output by only \
                         {rel} relative (max_delta {max_delta}, max_abs {max_abs}). The Q/K \
                         normalisation is still scale-invariant, which means the #1583 epsilon \
                         is absent from rsqrt(sg_buf[0])."
            );
        }
    }

    /// #1583, divergence 2: a tiny but FINITE Q/K vector must no longer be treated
    /// as if it were the zero vector.
    ///
    /// The Metal guard used to read `isfinite(sum_sq) && sum_sq > 1e-12f`, zeroing
    /// the whole vector below that threshold, while the CPU reference
    /// (`l2_normalize_rows`, gdn_chunk_ref.rs:38-56) fails closed on NON-FINITE
    /// input only and otherwise divides by `sqrt(sum_sq + 1e-6)`. So for
    /// `0 < sum_sq <= 1e-12` the two backends did categorically different things:
    /// Metal returned an exact zero vector, the CPU returned the input scaled by
    /// about 1e3. Adding the epsilon collapses both rules to the CPU's, because
    /// `sum_sq + 1e-6` is never near zero, so the threshold arm became unreachable
    /// and was removed.
    ///
    /// The assertion is a ratio rather than a threshold, which is what lets it be
    /// mutation-sensitive to the GUARD specifically and not merely to the epsilon:
    /// a run just below the old cliff must be CLOSER to its neighbour just above
    /// the cliff than it is to a run with Q identically zero. Restoring the
    /// `> 1e-12f` arm makes the below-cliff run bit-identical to the zero-Q run
    /// (distance 0) and wildly far from its neighbour, inverting the inequality.
    #[test]
    fn gdn_qk_l2_norm_has_no_cliff_at_the_old_guard_1583() {
        // Fail closed under LATTICE_METAL_TEST_ENFORCE=1, matching the other
        // Metal tests CI names explicitly: without this the CI step added for
        // this test would pass on a runner with no Metal device, which is the
        // one condition that makes the step worth having.
        let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
        let Some(device) = Device::system_default() else {
            assert!(
                !enforce,
                "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present (gdn_qk_l2_norm_has_no_cliff_at_the_old_guard_1583)"
            );
            return;
        };
        let _gpu = gpu_test_lock();
        let lib = compile_msl(&device);
        let pipe = pipeline_for(&device, &lib, "gdn_recurrence_fused");
        let queue = device.new_command_queue();

        let kd = 8u32;
        let vd = 8u32;
        let hd = 16u32;
        let hidden_in = lcg_vec(1, hd as usize);
        let in_proj_b = lcg_vec(2, hd as usize);
        let in_proj_a = lcg_vec(3, hd as usize);
        let s_init = lcg_vec(4, (kd * vd) as usize);

        let base = lcg_vec(5, (3 * kd) as usize);
        let sum_sq: f32 = base[..kd as usize].iter().map(|v| v * v).sum();
        assert!(
            sum_sq > 1e-3,
            "fixture sum_sq {sum_sq} too small to rescale from"
        );

        // Two Q vectors straddling the REMOVED 1e-12 threshold, plus the zero
        // vector the old guard mapped the lower one onto.
        let scale_to = |target: f32| {
            let c = (target / sum_sq).sqrt();
            let mut v = base.clone();
            for x in v[..kd as usize].iter_mut() {
                *x *= c;
            }
            v
        };
        let below = scale_to(9e-13);
        let above = scale_to(1.1e-12);
        let mut zeroed = base.clone();
        for x in zeroed[..kd as usize].iter_mut() {
            *x = 0.0;
        }

        let run = |conv: &[f32]| {
            run_gdn_recurrence_fused(
                &device, &queue, &pipe, kd, vd, hd, conv, &s_init, &hidden_in, &in_proj_b,
                &in_proj_a,
            )
            .0
        };
        let out_below = run(&below);
        let out_above = run(&above);
        let out_zero = run(&zeroed);
        assert_no_nonfinite("Q", "output(below)", &out_below);
        assert_no_nonfinite("Q", "output(above)", &out_above);
        assert_no_nonfinite("Q", "output(zero)", &out_zero);

        let dist = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let d_neighbour = dist(&out_below, &out_above);
        let d_zero = dist(&out_below, &out_zero);

        assert!(
            d_zero > d_neighbour,
            "a Q vector with sum_sq 9e-13 is closer to an identically-zero Q \
                     (distance {d_zero}) than to a Q with sum_sq 1.1e-12 (distance \
                     {d_neighbour}). That is the signature of the removed `sum_sq > 1e-12f` \
                     guard: the normalisation still has a cliff there and still zeroes \
                     tiny-but-finite vectors the CPU reference normalises."
        );
    }

    // --------------------------------------------------------------
    // Kernel 2: gdn_recurrence_fused_q36. kd=vd=128, hd=5120 are
    // Metal `constexpr` in this kernel (not runtime params), so they
    // are NOT shrinkable; num_value_heads=1 keeps the rest of the
    // dispatch small (k_head = h/3u still resolves to 0 for h=0).
    // --------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn run_gdn_recurrence_fused_q36(
        device: &Device,
        queue: &CommandQueue,
        pipe: &ComputePipelineState,
        conv_out: &[f32],
        s_init: &[f32],
        hidden_in: &[f32],
        in_proj_b: &[f32],
        in_proj_a: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        const KD: u32 = 128;
        const VD: u32 = 128;
        const HD: u32 = 5120;
        let q_total = KD;
        let v_offset = q_total * 2;

        let s_buf = f32_buf(device, s_init, "S_all");
        let conv_buf = f32_buf(device, conv_out, "conv_out");
        let z_buf = f32_buf(device, &vec![0.0f32; VD as usize], "z_proj");
        let hidden_buf = f32_buf(device, hidden_in, "hidden_in");
        let inb_buf = make_buffer_f16(device, in_proj_b, "in_proj_b");
        let ina_buf = make_buffer_f16(device, in_proj_a, "in_proj_a");
        let alog_buf = f32_buf(device, &[-1.0f32], "a_log");
        let dtb_buf = f32_buf(device, &[0.0f32], "dt_bias");
        let normw_buf = f32_buf(device, &vec![1.0f32; VD as usize], "norm_w");
        let out_buf = make_zero_buffer(device, VD as usize, "output");

        let params = TestGdnRecurParams {
            key_dim: KD,
            value_dim: VD,
            num_key_heads: 1,
            num_value_heads: 1,
            hidden_size: HD,
            q_total,
            v_offset,
            scale: 1.0 / (KD as f32).sqrt(),
            eps: 1e-6,
        };

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&s_buf), 0);
        enc.set_buffer(1, Some(&conv_buf), 0);
        enc.set_buffer(2, Some(&z_buf), 0);
        enc.set_buffer(3, Some(&hidden_buf), 0);
        enc.set_buffer(4, Some(&inb_buf), 0);
        enc.set_buffer(5, Some(&ina_buf), 0);
        enc.set_buffer(6, Some(&alog_buf), 0);
        enc.set_buffer(7, Some(&dtb_buf), 0);
        enc.set_buffer(8, Some(&normw_buf), 0);
        enc.set_buffer(9, Some(&out_buf), 0);
        enc.set_bytes(
            10,
            std::mem::size_of::<TestGdnRecurParams>() as u64,
            &params as *const TestGdnRecurParams as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 4, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        (
            read_f32(&out_buf, VD as usize),
            read_f32(&s_buf, (KD * VD) as usize),
        )
    }

    #[test]
    fn gdn_recurrence_fused_q36_fails_closed_on_nan_q_and_k_lane() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping gdn_recurrence_fused_q36_fails_closed: no Metal device");
            return;
        };
        let _gpu = gpu_test_lock();
        let lib = compile_msl(&device);
        let pipe = pipeline_for(&device, &lib, "gdn_recurrence_fused_q36");
        let queue = device.new_command_queue();

        const KD: u32 = 128;
        const HD: u32 = 5120;
        let hidden_in = lcg_vec(11, HD as usize);
        let in_proj_b = lcg_vec(12, HD as usize);
        let in_proj_a = lcg_vec(13, HD as usize);
        let s_init = lcg_vec(14, (KD * KD) as usize);

        for label in ["Q", "K"] {
            let base = lcg_vec(15, (3 * KD) as usize); // [Q(128) | K(128) | V(128)]
            let mut poisoned = base.clone();
            let mut reference = base.clone();
            let (lo, hi) = if label == "Q" {
                (0usize, KD as usize)
            } else {
                (KD as usize, 2 * KD as usize)
            };
            poisoned[lo] = f32::NAN;
            for v in reference[lo..hi].iter_mut() {
                *v = 0.0;
            }

            // Stress/repeat coverage for the #862 WAR-hazard
            // fix; see the identical comment in
            // gdn_recurrence_fused_fails_closed_on_nan_q_and_k_lane above.
            for _ in 0..20 {
                let (out_p, s_p) = run_gdn_recurrence_fused_q36(
                    &device, &queue, &pipe, &poisoned, &s_init, &hidden_in, &in_proj_b, &in_proj_a,
                );
                let (out_r, s_r) = run_gdn_recurrence_fused_q36(
                    &device, &queue, &pipe, &reference, &s_init, &hidden_in, &in_proj_b, &in_proj_a,
                );

                assert_no_nonfinite(
                    &format!("gdn_recurrence_fused_q36[{label}]"),
                    "output",
                    &out_p,
                );
                assert_no_nonfinite(&format!("gdn_recurrence_fused_q36[{label}]"), "S_all", &s_p);
                assert_eq!(
                    out_p, out_r,
                    "gdn_recurrence_fused_q36[{label}-poisoned]: output must be \
                             bit-identical to the never-poisoned (explicit-zero-{label}) reference"
                );
                assert_eq!(
                    s_p, s_r,
                    "gdn_recurrence_fused_q36[{label}-poisoned]: updated S_all must be \
                             bit-identical to the never-poisoned (explicit-zero-{label}) reference"
                );
            }
        }
    }

    // --------------------------------------------------------------
    // Kernel 3: gdn_precompute_keys. kd=128, hd=5120 are Metal
    // `constexpr` (not shrinkable). Writes into a split key_scratch
    // buffer: [q_norm(128) | k_norm(128) | k_dot_q(1)] per key head,
    // then [beta | g] per value head.
    // --------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn run_gdn_precompute_keys(
        device: &Device,
        queue: &CommandQueue,
        pipe: &ComputePipelineState,
        conv_out: &[f32],
        hidden_in: &[f32],
        in_proj_b: &[f32],
        in_proj_a: &[f32],
    ) -> Vec<f32> {
        const KD: u32 = 128;
        const HD: u32 = 5120;
        let q_total = KD;
        let key_stride = 2 * KD + 1;
        let scratch_len = key_stride + 2; // 1 key head + 1 value head

        let conv_buf = f32_buf(device, conv_out, "conv_out");
        let hidden_buf = f32_buf(device, hidden_in, "hidden_in");
        let inb_buf = make_buffer_f16(device, in_proj_b, "in_proj_b");
        let ina_buf = make_buffer_f16(device, in_proj_a, "in_proj_a");
        let alog_buf = f32_buf(device, &[-1.0f32], "a_log");
        let dtb_buf = f32_buf(device, &[0.0f32], "dt_bias");
        let scratch_buf = make_zero_buffer(device, scratch_len as usize, "key_scratch");

        let params = TestGdnRecurParams {
            key_dim: KD,
            value_dim: KD,
            num_key_heads: 1,
            num_value_heads: 1,
            hidden_size: HD,
            q_total,
            v_offset: q_total * 2,
            scale: 1.0 / (KD as f32).sqrt(),
            eps: 1e-6,
        };

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&conv_buf), 0);
        enc.set_buffer(1, Some(&hidden_buf), 0);
        enc.set_buffer(2, Some(&inb_buf), 0);
        enc.set_buffer(3, Some(&ina_buf), 0);
        enc.set_buffer(4, Some(&alog_buf), 0);
        enc.set_buffer(5, Some(&dtb_buf), 0);
        enc.set_buffer(6, Some(&scratch_buf), 0);
        enc.set_bytes(
            7,
            std::mem::size_of::<TestGdnRecurParams>() as u64,
            &params as *const TestGdnRecurParams as *const _,
        );
        // Production dispatch shape for gdn_precompute_keys (matches
        // encode_gdn_layer's H3 dispatch); thread_index_in_threadgroup is
        // linear regardless of the 3D shape, so this is a fidelity match,
        // not a correctness requirement.
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        read_f32(&scratch_buf, scratch_len as usize)
    }

    #[test]
    fn gdn_precompute_keys_fails_closed_on_nan_q_and_k_lane() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping gdn_precompute_keys_fails_closed: no Metal device");
            return;
        };
        let _gpu = gpu_test_lock();
        let lib = compile_msl(&device);
        // `gdn_precompute_keys` is defined unconditionally in qwen35.metal (no
        // #ifdef gate) -- it is not actually an optional/conditionally-compiled
        // kernel, so a missing lookup here means MSL_SOURCE was compiled but the
        // named kernel silently failed to link, which is itself a bug this test
        // must catch, not skip past (#862: only
        // device-absence may remain a platform-level skip).
        let pipe_func = lib
            .get_function("gdn_precompute_keys", None)
            .expect("gdn_precompute_keys must be present in MSL_SOURCE once compiled");
        let pipe = device
            .new_compute_pipeline_state_with_function(&pipe_func)
            .expect("build gdn_precompute_keys pipeline");
        let queue = device.new_command_queue();

        const KD: u32 = 128;
        const HD: u32 = 5120;
        let hidden_in = lcg_vec(21, HD as usize);
        let in_proj_b = lcg_vec(22, HD as usize);
        let in_proj_a = lcg_vec(23, HD as usize);

        for label in ["Q", "K"] {
            let base = lcg_vec(24, (2 * KD) as usize); // [Q(128) | K(128)]
            let mut poisoned = base.clone();
            let mut reference = base.clone();
            let (lo, hi) = if label == "Q" {
                (0usize, KD as usize)
            } else {
                (KD as usize, 2 * KD as usize)
            };
            poisoned[lo] = f32::NAN;
            for v in reference[lo..hi].iter_mut() {
                *v = 0.0;
            }

            // Stress/repeat coverage for the #862 WAR-hazard
            // fix (this kernel has TWO fixed handoffs: Q->K and K->k_dot_q);
            // see the identical comment in
            // gdn_recurrence_fused_fails_closed_on_nan_q_and_k_lane above.
            for _ in 0..20 {
                let scratch_p = run_gdn_precompute_keys(
                    &device, &queue, &pipe, &poisoned, &hidden_in, &in_proj_b, &in_proj_a,
                );
                let scratch_r = run_gdn_precompute_keys(
                    &device, &queue, &pipe, &reference, &hidden_in, &in_proj_b, &in_proj_a,
                );

                // scratch layout: [q_norm(128) | k_norm(128) | k_dot_q(1) | beta | g]
                assert_no_nonfinite(
                    &format!("gdn_precompute_keys[{label}]"),
                    "key_scratch",
                    &scratch_p,
                );
                assert_eq!(
                    scratch_p, scratch_r,
                    "gdn_precompute_keys[{label}-poisoned]: key_scratch must be \
                             bit-identical to the never-poisoned (explicit-zero-{label}) reference"
                );
            }
        }
    }

    // --------------------------------------------------------------
    // Kernel 4: gdn_chunk_materialize_c32. Implicitly requires
    // kd=vd=128 (the kernel indexes `kh*kd+tid` across all 128
    // dispatched threads with no `tid < kd` bound, unlike kernel 1).
    // Single token, single chunk (n_tokens=1, chunk_size=1,
    // num_chunks=1) to keep the dispatch minimal; conv1d kernel_size
    // (`ks`) is a Metal `constexpr 4u` in this kernel, independent of
    // GdnChunkParams.
    // --------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn run_gdn_chunk_materialize_c32(
        device: &Device,
        queue: &CommandQueue,
        pipe: &ComputePipelineState,
        gdn_qkv: &[f32],
        hidden_in: &[f32],
        in_proj_b: &[f32],
        in_proj_a: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        const KD: u32 = 128;
        const VD: u32 = 128;
        const HD: u32 = 16;
        const BUF_LEN: u32 = 3; // ks - 1
        let q_total = KD;
        let v_offset = q_total * 2;
        let qkv_dim = v_offset + VD;

        let conv_buf_state = make_zero_buffer(device, (qkv_dim * BUF_LEN) as usize, "conv_buf");
        let gdn_qkv_buf = f32_buf(device, gdn_qkv, "gdn_qkv");
        let conv_weight_buf = f32_buf(
            device,
            &vec![0.25f32; (qkv_dim * 4) as usize],
            "conv_weight",
        );
        let hidden_buf = f32_buf(device, hidden_in, "hidden_in");
        let inb_buf = make_buffer_f16(device, in_proj_b, "in_proj_b");
        let ina_buf = make_buffer_f16(device, in_proj_a, "in_proj_a");
        let alog_buf = f32_buf(device, &[-1.0f32], "a_log");
        let dtb_buf = f32_buf(device, &[0.0f32], "dt_bias");
        let out_q_buf = make_zero_buffer(device, KD as usize, "out_q");
        let out_k_buf = make_zero_buffer(device, KD as usize, "out_k");
        let out_v_buf = make_zero_buffer(device, VD as usize, "out_v");
        let out_bla_buf = make_zero_buffer(device, 2, "out_bla");

        let params = GdnChunkParams {
            key_dim: KD,
            value_dim: VD,
            num_key_heads: 1,
            num_value_heads: 1,
            hidden_size: HD,
            q_total,
            v_offset,
            qkv_dim,
            output_dim: VD,
            chunk_size: 1,
            n_tokens: 1,
            num_chunks: 1,
            active_chunk: 0,
            scale: 1.0 / (KD as f32).sqrt(),
            eps: 1e-6,
        };

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&conv_buf_state), 0);
        enc.set_buffer(1, Some(&gdn_qkv_buf), 0);
        enc.set_buffer(2, Some(&conv_weight_buf), 0);
        enc.set_buffer(3, Some(&hidden_buf), 0);
        enc.set_buffer(4, Some(&inb_buf), 0);
        enc.set_buffer(5, Some(&ina_buf), 0);
        enc.set_buffer(6, Some(&alog_buf), 0);
        enc.set_buffer(7, Some(&dtb_buf), 0);
        enc.set_buffer(8, Some(&out_q_buf), 0);
        enc.set_buffer(9, Some(&out_k_buf), 0);
        enc.set_buffer(10, Some(&out_v_buf), 0);
        enc.set_buffer(11, Some(&out_bla_buf), 0);
        enc.set_bytes(
            12,
            std::mem::size_of::<GdnChunkParams>() as u64,
            &params as *const GdnChunkParams as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 4, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        (
            read_f32(&out_q_buf, KD as usize),
            read_f32(&out_k_buf, KD as usize),
        )
    }

    #[test]
    fn gdn_chunk_materialize_c32_fails_closed_on_nan_q_and_k_lane() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping gdn_chunk_materialize_c32_fails_closed: no Metal device");
            return;
        };
        let _gpu = gpu_test_lock();
        let lib = compile_msl(&device);
        let pipe = pipeline_for(&device, &lib, "gdn_chunk_materialize_c32");
        let queue = device.new_command_queue();

        const KD: u32 = 128;
        const HD: u32 = 16;
        let hidden_in = lcg_vec(31, HD as usize);
        let in_proj_b = lcg_vec(32, HD as usize);
        let in_proj_a = lcg_vec(33, HD as usize);

        for label in ["Q", "K"] {
            let base = lcg_vec(34, (3 * KD) as usize); // [Q(128) | K(128) | V(128)]
            let mut poisoned = base.clone();
            let mut reference = base.clone();
            let (lo, hi) = if label == "Q" {
                (0usize, KD as usize)
            } else {
                (KD as usize, 2 * KD as usize)
            };
            poisoned[lo] = f32::NAN;
            for v in reference[lo..hi].iter_mut() {
                *v = 0.0;
            }

            let (outq_p, outk_p) = run_gdn_chunk_materialize_c32(
                &device, &queue, &pipe, &poisoned, &hidden_in, &in_proj_b, &in_proj_a,
            );
            let (outq_r, outk_r) = run_gdn_chunk_materialize_c32(
                &device, &queue, &pipe, &reference, &hidden_in, &in_proj_b, &in_proj_a,
            );

            assert_no_nonfinite(
                &format!("gdn_chunk_materialize_c32[{label}]"),
                "out_q",
                &outq_p,
            );
            assert_no_nonfinite(
                &format!("gdn_chunk_materialize_c32[{label}]"),
                "out_k",
                &outk_p,
            );
            assert_eq!(
                outq_p, outq_r,
                "gdn_chunk_materialize_c32[{label}-poisoned]: out_q must be \
                         bit-identical to the never-poisoned (explicit-zero-{label}) reference"
            );
            assert_eq!(
                outk_p, outk_r,
                "gdn_chunk_materialize_c32[{label}-poisoned]: out_k must be \
                         bit-identical to the never-poisoned (explicit-zero-{label}) reference"
            );
        }
    }

    // --------------------------------------------------------------
    // #862 follow-up: multi-token gdn_chunk_materialize_c32
    // dispatch (the CHUNKED PREFILL path #850's integration ask names) exercising
    // REAL within-chunk conv-window carryover, not a single n_tokens=1 dispatch.
    //
    // `gdn_chunk_materialize_c32`'s per-token loop (`for (uint j=0u;j<ci;j++)`)
    // reads each tap directly from the flat `gdn_qkv` input array by absolute
    // token index (`src = global_row+tap-buf_len`; `src<0` reads pre-chunk
    // `conv_buf` history, `src>=0` reads `gdn_qkv[src*qkv_dim+ch]`), so a NaN raw
    // value written at token 0's channel contaminates every token whose tap
    // window includes token 0 -- tokens 0,1,2,3 for `ks=4` (this kernel's
    // `constexpr ks=4u`, i.e. `buf_len=3`) -- and is naturally absent (no register
    // carryover needed, it is pure array indexing) for token 4 onward. This is the
    // dispatch-time analog of the decode-path `apply_causal_conv1d` rolling buffer
    // proven in `gdn_fused::tests::gated_delta_net_step_fused_k_poison_window_state_isolation_production_kernel`.
    //
    // Reference construction (same reasoning as the decode-path test above): BOTH
    // poisoned and reference runs hold the target head's whole channel-span
    // exactly zero for tokens 0..ks (the full contamination window), differing
    // only by the single NaN scalar at [token 0, poisoned_channel]. Tokens ks..N-1
    // ("at least 4 subsequent clean tokens") use ordinary ORDINARY per-token
    // random data identically in both runs. Whole-vector zero-by-assignment on a
    // non-finite reduced norm is bit-identical to true-zero-weight arithmetic for
    // as long as the corruption sits in the window, and the two runs are governed
    // by the SAME per-token raw array for every clean token, so they are proven
    // bit-identical, and finite, at every token, with no further guard dependency
    // once the window closes.
    // --------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn run_gdn_chunk_materialize_c32_multi_token(
        device: &Device,
        queue: &CommandQueue,
        pipe: &ComputePipelineState,
        gdn_qkv: &[f32], // [n_tokens, qkv_dim] flat, qkv_dim = 3*KD
        hidden_in: &[f32],
        in_proj_b: &[f32],
        in_proj_a: &[f32],
        n_tokens: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        const KD: u32 = 128;
        const VD: u32 = 128;
        const HD: u32 = 16;
        const BUF_LEN: u32 = 3; // ks - 1
        let q_total = KD;
        let v_offset = q_total * 2;
        let qkv_dim = v_offset + VD;

        let conv_buf_state = make_zero_buffer(device, (qkv_dim * BUF_LEN) as usize, "conv_buf");
        let gdn_qkv_buf = f32_buf(device, gdn_qkv, "gdn_qkv");
        let conv_weight_buf = f32_buf(
            device,
            &vec![0.25f32; (qkv_dim * 4) as usize],
            "conv_weight",
        );
        let hidden_buf = f32_buf(device, hidden_in, "hidden_in");
        let inb_buf = make_buffer_f16(device, in_proj_b, "in_proj_b");
        let ina_buf = make_buffer_f16(device, in_proj_a, "in_proj_a");
        let alog_buf = f32_buf(device, &[-1.0f32], "a_log");
        let dtb_buf = f32_buf(device, &[0.0f32], "dt_bias");
        let out_q_buf = make_zero_buffer(device, (KD * n_tokens) as usize, "out_q");
        let out_k_buf = make_zero_buffer(device, (KD * n_tokens) as usize, "out_k");
        let out_v_buf = make_zero_buffer(device, (VD * n_tokens) as usize, "out_v");
        let out_bla_buf = make_zero_buffer(device, (2 * n_tokens) as usize, "out_bla");

        let params = GdnChunkParams {
            key_dim: KD,
            value_dim: VD,
            num_key_heads: 1,
            num_value_heads: 1,
            hidden_size: HD,
            q_total,
            v_offset,
            qkv_dim,
            output_dim: VD,
            chunk_size: n_tokens,
            n_tokens,
            num_chunks: 1,
            active_chunk: 0,
            scale: 1.0 / (KD as f32).sqrt(),
            eps: 1e-6,
        };

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&conv_buf_state), 0);
        enc.set_buffer(1, Some(&gdn_qkv_buf), 0);
        enc.set_buffer(2, Some(&conv_weight_buf), 0);
        enc.set_buffer(3, Some(&hidden_buf), 0);
        enc.set_buffer(4, Some(&inb_buf), 0);
        enc.set_buffer(5, Some(&ina_buf), 0);
        enc.set_buffer(6, Some(&alog_buf), 0);
        enc.set_buffer(7, Some(&dtb_buf), 0);
        enc.set_buffer(8, Some(&out_q_buf), 0);
        enc.set_buffer(9, Some(&out_k_buf), 0);
        enc.set_buffer(10, Some(&out_v_buf), 0);
        enc.set_buffer(11, Some(&out_bla_buf), 0);
        enc.set_bytes(
            12,
            std::mem::size_of::<GdnChunkParams>() as u64,
            &params as *const GdnChunkParams as *const _,
        );
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 4, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        (
            read_f32(&out_q_buf, (KD * n_tokens) as usize),
            read_f32(&out_k_buf, (KD * n_tokens) as usize),
        )
    }

    #[test]
    fn gdn_chunk_materialize_c32_multi_token_k_poison_window_state_isolation() {
        let Some(device) = Device::system_default() else {
            eprintln!(
                "skipping gdn_chunk_materialize_c32_multi_token_k_poison_window_state_isolation: no Metal device"
            );
            return;
        };
        let _gpu = gpu_test_lock();
        let lib = compile_msl(&device);
        let pipe = pipeline_for(&device, &lib, "gdn_chunk_materialize_c32");
        let queue = device.new_command_queue();

        const KD: u32 = 128;
        const HD: u32 = 16;
        const WINDOW: u32 = 4; // ks (kernel_size), matches the kernel's constexpr ks=4u
        const CLEAN_TAIL: u32 = 4; // #862: "at least 4 subsequent clean tokens"
        const N_TOKENS: u32 = WINDOW + CLEAN_TAIL;
        let qkv_dim = 3 * KD; // [Q(128) | K(128) | V(128)] per token
        let hidden_in = lcg_vec(41, HD as usize);
        let in_proj_b = lcg_vec(42, HD as usize);
        let in_proj_a = lcg_vec(43, HD as usize);

        // K channel span (label fixed to K: the shipping-path production gap
        // #862 found; the sibling single-token test above already
        // covers both Q and K for a single dispatch).
        let (lo, hi) = (KD as usize, 2 * KD as usize);

        let base = lcg_vec(44, (qkv_dim * N_TOKENS) as usize);
        let mut poisoned = base.clone();
        let mut reference = base.clone();
        // Zero the K channel span for every token in the window (0..WINDOW) in
        // BOTH runs -- the only reference construction for which "bit-identical
        // to the poisoned run" is achievable (see module doc comment above).
        for t in 0..WINDOW {
            let row = (t * qkv_dim) as usize;
            for v in poisoned[row + lo..row + hi].iter_mut() {
                *v = 0.0;
            }
            for v in reference[row + lo..row + hi].iter_mut() {
                *v = 0.0;
            }
        }
        // Poison exactly one lane, only at token 0.
        poisoned[lo] = f32::NAN;

        let (outq_p, outk_p) = run_gdn_chunk_materialize_c32_multi_token(
            &device, &queue, &pipe, &poisoned, &hidden_in, &in_proj_b, &in_proj_a, N_TOKENS,
        );
        let (outq_r, outk_r) = run_gdn_chunk_materialize_c32_multi_token(
            &device, &queue, &pipe, &reference, &hidden_in, &in_proj_b, &in_proj_a, N_TOKENS,
        );

        assert_no_nonfinite(
            "gdn_chunk_materialize_c32[multi-token-K]",
            "out_q (all tokens)",
            &outq_p,
        );
        assert_no_nonfinite(
            "gdn_chunk_materialize_c32[multi-token-K]",
            "out_k (all tokens)",
            &outk_p,
        );
        assert_eq!(
            outq_p, outq_r,
            "gdn_chunk_materialize_c32[multi-token-K-poisoned]: out_q must be \
                     bit-identical to the never-poisoned (explicit-zero-window) reference \
                     at every token (Q was never touched by the K poison; this also \
                     confirms the poison doesn't cross-contaminate the other head channel)"
        );
        for t in 0..N_TOKENS as usize {
            let row = t * KD as usize;
            assert_eq!(
                outk_p[row..row + KD as usize],
                outk_r[row..row + KD as usize],
                "gdn_chunk_materialize_c32[multi-token-K-poisoned]: token {t}'s \
                         out_k must be bit-identical to the never-poisoned \
                         (explicit-zero-window) reference -- token {t} is {} the ks={WINDOW} \
                         contamination window",
                if (t as u32) < WINDOW {
                    "inside"
                } else {
                    "past (fully recovered from)"
                }
            );
        }
    }

    // --------------------------------------------------------------
    // decode_reference.metal oracle hardening: this kernel is a
    // frozen test-only correctness oracle (used by the flash-decode
    // parity suite above), not a live production kernel, but per
    // #850's checklist it must not itself reproduce the
    // multiply-through-zero bug class it exists to catch.
    // --------------------------------------------------------------
    #[test]
    fn decode_attention_reference_fails_closed_on_nan_q() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping decode_attention_reference_fails_closed: no Metal device");
            return;
        };
        let _gpu = gpu_test_lock();
        let opts = CompileOptions::new();
        let lib = device
            .new_library_with_source(MSL_DECODE_REFERENCE, &opts)
            .expect("MSL_DECODE_REFERENCE must compile");
        let pipe = pipeline_for(&device, &lib, "decode_attention_reference");
        let queue = device.new_command_queue();

        const HEAD_DIM: u32 = 8;
        const NUM_Q_HEADS: u32 = 1;
        const NUM_KV_HEADS: u32 = 1;
        const CACHE_LEN: u32 = 4;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let mut q = lcg_vec(41, HEAD_DIM as usize);
        q[0] = f32::NAN; // poison one Q lane -> every dot product for this head is NaN
        let k = lcg_vec(42, (CACHE_LEN * HEAD_DIM) as usize);
        let v = lcg_vec(43, (CACHE_LEN * HEAD_DIM) as usize);

        let q_buf = f32_buf(&device, &q, "q");
        let k_buf = f32_buf(&device, &k, "k_cache");
        let v_buf = f32_buf(&device, &v, "v_cache");
        let out_buf = make_zero_buffer(&device, HEAD_DIM as usize, "out");

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&k_buf), 0);
        enc.set_buffer(2, Some(&v_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        enc.set_bytes(4, 4, &CACHE_LEN as *const u32 as *const _);
        enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
        enc.set_bytes(6, 4, &NUM_Q_HEADS as *const u32 as *const _);
        enc.set_bytes(7, 4, &NUM_KV_HEADS as *const u32 as *const _);
        enc.set_bytes(8, 4, &HEAD_DIM as *const u32 as *const _);
        enc.set_bytes(9, 4, &HEAD_DIM as *const u32 as *const _);
        enc.set_bytes(10, 4, &scale as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(NUM_Q_HEADS as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out = read_f32(&out_buf, HEAD_DIM as usize);
        let nan_count = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            nan_count,
            0,
            "decode_attention_reference must fail closed (zero output) on a \
                     NaN-poisoned Q lane instead of propagating NaN through the \
                     `acc * inv_sum` finalize multiply (ADR-080 C1, #850); got \
                     {nan_count}/{} non-finite elements: {out:?}",
            out.len()
        );
        assert!(
            out.iter().all(|v| *v == 0.0),
            "decode_attention_reference must assign the literal 0.0 row on an \
                     invalid (non-finite) sum_exp, got {out:?}"
        );
    }
}
