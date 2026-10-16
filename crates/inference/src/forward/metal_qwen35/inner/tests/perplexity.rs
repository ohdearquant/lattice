//! Perplexity harness.

use super::*;
// -------------------------------------------------------------------
// Perplexity harness (ADR-044 step 4b)
// -------------------------------------------------------------------

#[test]
fn metal_compute_token_nlls_matches_manual_forward_loop() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];

    // Reference: manual reset + forward loop + log_softmax matches what
    // compute_token_nlls advertises.
    state.reset_state();
    let mut expected = Vec::with_capacity(tokens.len() - 1);
    for (pos, &tok) in tokens.iter().enumerate() {
        let logits = state.forward_step(tok, pos);
        if pos + 1 < tokens.len() {
            expected.push(crate::model::qwen35::log_softmax_nll(
                &logits[..cfg.vocab_size],
                tokens[pos + 1] as usize,
            ));
        }
    }

    // Subject under test — resets internally, returns identical NLLs.
    let got = state
        .compute_token_nlls(&tokens)
        .expect("compute_token_nlls ok on valid tokens");
    assert_eq!(got.len(), tokens.len() - 1);
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "metal compute_token_nlls[{i}] diverged from manual loop: got={g} expected={e}"
        );
    }
}

#[test]
fn metal_compute_token_nlls_is_repeatable_under_self_reset() {
    let _gpu_guard = gpu_test_lock();
    // The Metal harness resets recurrent state at the start of each
    // call. Two back-to-back calls on the same input must therefore
    // produce identical NLLs even though `forward_step` mutates the
    // session's KV cache and GDN state in place.
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");
    let tokens: Vec<u32> = vec![1, 7, 13, 8, 42];

    let first = state.compute_token_nlls(&tokens).expect("first call ok");
    let second = state.compute_token_nlls(&tokens).expect("second call ok");
    assert_eq!(first.len(), second.len());
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "back-to-back NLLs diverged at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn metal_compute_token_nlls_rejects_oversized_input() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    // KV cap = 4. compute_token_nlls(tokens) with tokens.len() > 4 must
    // refuse — would otherwise overflow the cache mid-window.
    let mut state = MetalQwen35State::new(&weights, &cfg, 4)
        .expect("tiny MetalQwen35State fixture constructs with max_cache_len=4");
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let err = state
        .compute_token_nlls(&tokens)
        .expect_err("tokens.len() > max_cache_len must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("KV-cache") && msg.contains('4'),
        "error must name the KV-cache cap; got: {msg}"
    );
}

#[test]
fn metal_compute_token_nlls_rejects_out_of_vocab_token() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");
    let vocab = cfg.vocab_size as u32;
    let err = state
        .compute_token_nlls(&[1, vocab, 3])
        .expect_err("out-of-vocab token must error");
    let msg = format!("{err}");
    assert!(
        msg.contains(&format!("{vocab}")),
        "error must name the bad token id; got: {msg}"
    );
}

#[test]
fn metal_compute_perplexity_equals_exp_mean_nll() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");
    let tokens: Vec<u32> = (1u32..=12).collect();
    let report = state
        .compute_perplexity(
            &tokens,
            &crate::model::qwen35::PerplexityConfig {
                window: 6,
                stride: 3,
            },
        )
        .expect("ppl ok");
    let expected_ppl = report.mean_nll.exp();
    assert!(
        (report.ppl - expected_ppl).abs() < 1e-9,
        "ppl must equal exp(mean_nll); got ppl={}, exp(mean_nll)={}",
        report.ppl,
        expected_ppl
    );
    assert!(
        (report.mean_nll - report.total_nll / report.num_tokens_scored as f64).abs() < 1e-9,
        "mean_nll must equal total_nll / num_tokens_scored"
    );
    assert!(report.ppl.is_finite() && report.ppl > 0.0);
}

#[test]
fn metal_compute_perplexity_matches_compute_token_nlls_on_single_window() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 16)
        .expect("tiny MetalQwen35State fixture constructs");
    let tokens: Vec<u32> = (1u32..=8).collect();
    // window covers the whole corpus → strided harness collapses to a
    // single window and the aggregated NLL must equal the direct sum.
    let direct_nlls = state.compute_token_nlls(&tokens).expect("direct ok");
    let direct_total: f64 = direct_nlls.iter().map(|&x| x as f64).sum();

    let report = state
        .compute_perplexity(
            &tokens,
            &crate::model::qwen35::PerplexityConfig {
                window: tokens.len(),
                stride: tokens.len() - 1,
            },
        )
        .expect("ppl ok");
    assert_eq!(report.num_windows, 1);
    assert_eq!(report.num_tokens_scored, tokens.len() - 1);
    assert!(
        (report.total_nll - direct_total).abs() < 1e-5,
        "single-window strided PPL must agree with compute_token_nlls sum: got {} vs {}",
        report.total_nll,
        direct_total
    );
}

#[test]
fn metal_compute_perplexity_rejects_window_above_max_cache_len() {
    let _gpu_guard = gpu_test_lock();
    // Counterpart of the CPU test:
    // `compute_perplexity_rejects_window_above_rope_capacity`. The
    // Metal-side cap is the KV-cache size, not the RoPE table.
    let Some(_) = Device::system_default() else {
        return;
    };
    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let max_cache_len = 8usize;
    let mut state = MetalQwen35State::new(&weights, &cfg, max_cache_len)
        .expect("tiny MetalQwen35State fixture constructs");
    // Long enough corpus that the effective window > max_cache_len.
    let tokens: Vec<u32> = (1..=8).cycle().take(max_cache_len + 4).collect();
    let err = state
        .compute_perplexity(
            &tokens,
            &crate::model::qwen35::PerplexityConfig {
                window: max_cache_len + 1,
                stride: 4,
            },
        )
        .expect_err("window > max_cache_len must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("context capacity") && msg.contains(&format!("{max_cache_len}")),
        "error must name backend context capacity + cap; got: {msg}"
    );
}

// Regression test for issue #540: the Metal no-copy mmap loader must
// fail closed on a truncated `.q4` block payload before the mmap is
// handed to Metal dispatch, mirroring the CPU sibling's
// `read_exact`-driven bounds discipline. Mutation check: reverting
// the `validate_q4_header_payload_bounds` call in `mmap_q4_weight`
// makes this test fail (return becomes `Ok`).
#[test]
fn mmap_q4_weight_rejects_truncated_block_payload_before_dispatch() {
    let Some(device) = Device::system_default() else {
        return;
    };

    let tmp = tempfile::tempdir().expect("tempdir create");
    let path = tmp.path().join("truncated_payload.q4");
    let mut file = std::fs::File::create(&path).expect("create truncated q4");
    use std::io::Write;
    file.write_all(b"KHQ4").expect("write magic");
    file.write_all(&2_u32.to_le_bytes()).expect("write version");
    file.write_all(&1_u32.to_le_bytes()).expect("write ndim");
    file.write_all(&64_u64.to_le_bytes()).expect("write shape");
    file.write_all(&64_u64.to_le_bytes())
        .expect("write original_len");
    file.flush().expect("flush q4 header");

    let result = mmap_q4_weight(&device, &path, "truncated-payload-proof", &[64]);

    assert!(
        result.is_err(),
        "mmap_q4_weight must reject a KHQ4 header whose original_len requires \
                 40 payload bytes but whose file ends at payload_offset"
    );
}

#[test]
fn mmap_q4_weight_rejects_payload_one_byte_short() {
    let Some(device) = Device::system_default() else {
        return;
    };

    let tmp = tempfile::tempdir().expect("tempdir create");
    let path = tmp.path().join("one_byte_short.q4");
    let mut file = std::fs::File::create(&path).expect("create truncated q4");
    use std::io::Write;
    file.write_all(b"KHQ4").expect("write magic");
    file.write_all(&2_u32.to_le_bytes()).expect("write version");
    file.write_all(&1_u32.to_le_bytes()).expect("write ndim");
    file.write_all(&32_u64.to_le_bytes()).expect("write shape");
    file.write_all(&32_u64.to_le_bytes())
        .expect("write original_len");
    // original_len=32 → exactly one block → 20 required payload bytes.
    // Write only 19 to prove the off-by-one boundary is enforced.
    file.write_all(&[0u8; 19]).expect("write short payload");
    file.flush().expect("flush truncated payload");

    let result = mmap_q4_weight(&device, &path, "one-byte-short-proof", &[32]);

    assert!(
        result.is_err(),
        "mmap_q4_weight must reject a payload one byte short of the required \
                 block length"
    );
}

#[test]
fn mmap_q4_weight_accepts_fully_populated_payload() {
    let Some(device) = Device::system_default() else {
        return;
    };

    let tmp = tempfile::tempdir().expect("tempdir create");
    let path = tmp.path().join("full_payload.q4");
    let tensor = crate::weights::q4_weights::Q4Tensor {
        blocks: vec![
            crate::weights::q4_weights::Q4Block {
                scale: 0x3C00,
                bias: 0,
                packed: [0u8; 16],
            };
            2
        ],
        shape: vec![64],
        original_len: 64,
    };
    crate::weights::q4_weights::save_q4_file(&path, &tensor).expect("save well-formed q4 file");

    let result = mmap_q4_weight(&device, &path, "full-payload-accept", &[64]);

    assert!(
        result.is_ok(),
        "mmap_q4_weight must accept a fully populated payload: {:?}",
        result.err()
    );
}
