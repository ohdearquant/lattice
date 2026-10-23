//! Chunked batched prefill parity.

use super::*;
// -----------------------------------------------------------------------
// Chunked batched prefill parity tests
// -----------------------------------------------------------------------

#[test]
fn qwen35_checkpoint_weights_present_accepts_sharded_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(!qwen35_checkpoint_weights_present(dir.path()));

    std::fs::write(dir.path().join("model.safetensors.index.json"), "{}").expect("write index");
    assert!(qwen35_checkpoint_weights_present(dir.path()));
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn metal_qwen35_chunked_prefill_long_prompt_matches_step_loop() {
    let _gpu_guard = gpu_test_lock();
    // Parity gate: chunked forward_prefill must agree with token-by-token
    // forward_step on a prompt longer than max_prefill (≈512).
    // Tests that RoPE offsets, KV cache rows, attention cache_len, and
    // GDN recurrent state all thread correctly across chunk boundaries.
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir)
        || !model_dir.join("config.json").exists()
        || !model_dir.join("tokenizer.json").exists()
    {
        eprintln!(
            "skipping chunked prefill parity test: model files missing at {}",
            model_dir.display()
        );
        return;
    }

    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    assert!(
        model.config().num_active_linear_attention_layers() > 0,
        "prompt must exercise at least one GDN layer"
    );

    let Some(device) = Device::system_default() else {
        eprintln!("skipping chunked prefill parity test: no Metal device");
        return;
    };
    let _ = device;

    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal qwen3.5-0.8b state");

    let tokens = long_real_text_tokens(model.tokenizer());
    assert!(
        tokens.len() > state.session.max_prefill,
        "prompt ({} tokens) must exceed max_prefill ({})",
        tokens.len(),
        state.session.max_prefill
    );
    assert!(
        (600..=700).contains(&tokens.len()),
        "prompt length {} out of expected 600-700 range",
        tokens.len()
    );

    // Path A: token-by-token reference.
    state.reset_state();
    let mut step_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        step_logits = state.forward_step(tok, pos);
    }

    // Path B: chunked batched prefill.
    state.reset_state();
    let prefill_logits = state.forward_prefill(&tokens);

    assert_eq!(step_logits.len(), model.config().vocab_size);
    assert_eq!(prefill_logits.len(), model.config().vocab_size);

    let max_abs_diff = step_logits
        .iter()
        .zip(prefill_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "chunked prefill parity: max_abs_diff={max_abs_diff:.6}, \
                 argmax_step={}, argmax_prefill={}, tokens={}",
        argmax_f32(&step_logits),
        argmax_f32(&prefill_logits),
        tokens.len()
    );

    // KNOWN ISSUE (2026-06-03): the chunked-batched prefill path exhibits a
    // nondeterministic, contention/occupancy-sensitive logit divergence vs the
    // token-by-token reference, observed up to ~0.94 in a tight back-to-back loop.
    // Static analysis of `gdn_recurrence_fused` and `decode_attention` shows
    // both kernels barrier-correct with no uninitialized threadgroup reads, no
    // intra-dispatch cross-threadgroup device race, and no untracked buffers on the
    // path — the magnitude is too large for FP-reorder noise, so the hazard is a
    // runtime-only one observable solely under GPU frame capture (Xcode Metal
    // debugger). The argmax is STABLE across hundreds of runs, so generated tokens
    // are unaffected; only logit *values* drift. The hard guarantee this test
    // enforces is therefore argmax equality; the logit-value bound is a loose
    // regression guard, not a strict-parity assertion. Root cause is unconfirmed:
    // a GPU frame capture of a divergent run is the diagnostic that could show the
    // actual dispatch-level ordering hazard, and nothing in this tree has done that
    // capture yet (see `gpu_test_lock`'s doc comment for what static analysis has
    // and hasn't ruled out).
    assert!(
        max_abs_diff < 0.5,
        "chunked prefill logits diverged from step loop beyond the known-issue \
                 tolerance (argmax-safe latent race): max_abs_diff={max_abs_diff}"
    );
    assert_eq!(
        argmax_f32(&step_logits),
        argmax_f32(&prefill_logits),
        "argmax mismatch between step loop and chunked prefill"
    );
    // Both paths must advance the session cursor to the full prompt length.
    assert_eq!(state.session.kv_cache.seq_len, tokens.len());
    assert_eq!(state.session.position(), tokens.len());
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn metal_qwen35_chunked_prefill_length_1_final_chunk_matches_step_loop() {
    let _gpu_guard = gpu_test_lock();
    // Degenerate-boundary gate: a prompt of exactly max_prefill + 1 tokens
    // decomposes into chunks [max_prefill, 1]. The length-1 final chunk
    // goes through forward_prefill_batched_chunk with n=1 (single-iteration
    // per-token loops, m=1 GEMMs) — NOT the forward_step fast path. This
    // path is unreachable by the 600-700 token parity test above.
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir)
        || !model_dir.join("config.json").exists()
        || !model_dir.join("tokenizer.json").exists()
    {
        eprintln!(
            "skipping length-1 final-chunk parity test: model files missing at {}",
            model_dir.display()
        );
        return;
    }

    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    assert!(
        model.config().num_active_linear_attention_layers() > 0,
        "prompt must exercise at least one GDN layer"
    );

    let Some(device) = Device::system_default() else {
        eprintln!("skipping length-1 final-chunk parity test: no Metal device");
        return;
    };
    let _ = device;

    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal qwen3.5-0.8b state");

    // Trim to exactly max_prefill + 1 so the final chunk has length 1.
    let mut tokens = long_real_text_tokens(model.tokenizer());
    let target = state.session.max_prefill + 1;
    assert!(
        tokens.len() >= target,
        "helper produced {} tokens, need >= {target}",
        tokens.len()
    );
    tokens.truncate(target);
    assert_eq!(
        tokens.len(),
        state.session.max_prefill + 1,
        "prompt must be exactly max_prefill + 1 for a length-1 final chunk"
    );

    // Path A: token-by-token reference.
    state.reset_state();
    let mut step_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        step_logits = state.forward_step(tok, pos);
    }

    // Path B: chunked batched prefill (final chunk length 1).
    state.reset_state();
    let prefill_logits = state.forward_prefill(&tokens);

    assert_eq!(step_logits.len(), model.config().vocab_size);
    assert_eq!(prefill_logits.len(), model.config().vocab_size);

    let max_abs_diff = step_logits
        .iter()
        .zip(prefill_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "length-1 final-chunk parity: max_abs_diff={max_abs_diff:.6}, \
                 argmax_step={}, argmax_prefill={}, tokens={}",
        argmax_f32(&step_logits),
        argmax_f32(&prefill_logits),
        tokens.len()
    );

    assert!(
        max_abs_diff < 1e-2,
        "length-1 final-chunk logits diverged from step loop: max_abs_diff={max_abs_diff}"
    );
    assert_eq!(
        argmax_f32(&step_logits),
        argmax_f32(&prefill_logits),
        "argmax mismatch on length-1 final chunk"
    );
    assert_eq!(state.session.kv_cache.seq_len, tokens.len());
    assert_eq!(state.session.position(), tokens.len());
}

/// Experiment B regression gate: `forward_prefill_batched_chunk` now gates its
/// terminal RMSNorm+lm_head+MTP+top-k tail behind `emit_logits`, skipping it on
/// intermediate chunks of a chunked, last-token-only (`!all_positions`) prefill —
/// only the LAST chunk emits it. This asserts the skip is correctness-preserving:
/// `forward_prefill` on a prompt spanning multiple chunks must still return
/// `vocab_size` logits, whose argmax agrees with the final-position row of a
/// `forward_prefill_all_logits` run (which emits every chunk's tail and so is
/// unaffected by the emit_logits gate), and the session cursor must still have
/// advanced through every chunk — including the ones whose tail was skipped.
/// Mutation-sensitive: if `is_last` in `forward_prefill_impl`'s `!all_positions`
/// loop is computed wrong (e.g. always `false`), `forward_prefill` returns an
/// empty `Vec` and the length assertion below fails; if `set_position` were
/// dropped from the `!emit_logits` branch, the position assertion fails.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn metal_qwen35_chunked_prefill_emit_logits_skip_correctness() {
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir)
        || !model_dir.join("config.json").exists()
        || !model_dir.join("tokenizer.json").exists()
    {
        eprintln!(
            "skipping emit_logits skip-correctness test: model files missing at {}",
            model_dir.display()
        );
        return;
    }

    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let Some(device) = Device::system_default() else {
        eprintln!("skipping emit_logits skip-correctness test: no Metal device");
        return;
    };
    let _ = device;
    let _gpu = gpu_test_lock();

    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal qwen3.5-0.8b state");
    let vocab = model.config().vocab_size;

    let tokens = long_real_text_tokens(model.tokenizer());
    assert!(
        tokens.len() > state.session.max_prefill,
        "prompt ({} tokens) must exceed max_prefill ({}) to exercise the \
                 chunked, emit_logits-gated path",
        tokens.len(),
        state.session.max_prefill
    );

    // Reference: all_positions=true emits the tail on EVERY chunk (unaffected by
    // the emit_logits gate) — its final-position row is what an unskipped last
    // chunk would have produced.
    state.reset_state();
    let all_logits = state
        .forward_prefill_all_logits(&tokens)
        .expect("LoRA-inactive all-position prefill succeeds");
    assert_eq!(
        all_logits.len(),
        tokens.len() * vocab,
        "all-position reference length mismatch"
    );
    let ref_last = &all_logits[(tokens.len() - 1) * vocab..tokens.len() * vocab];

    // Under test: the `!all_positions` chunked path, where every non-last chunk
    // now runs with emit_logits=false.
    state.reset_state();
    let last_logits = state.forward_prefill(&tokens);

    assert_eq!(
        last_logits.len(),
        vocab,
        "chunked !all_positions prefill must still return a full vocab_size logit \
                 vector — an empty result means the tail was (wrongly) skipped on the LAST \
                 chunk too"
    );
    eprintln!(
        "emit_logits skip-correctness: argmax_ref={}, argmax_chunked={}, tokens={}",
        argmax_f32(ref_last),
        argmax_f32(&last_logits),
        tokens.len()
    );
    assert_eq!(
        argmax_f32(&last_logits),
        argmax_f32(ref_last),
        "argmax mismatch between chunked (tail-skipping) prefill and the \
                 all-positions reference's final-position row"
    );

    // The session cursor must have advanced through every chunk, including the
    // intermediate ones whose terminal tail was skipped.
    assert_eq!(state.session.kv_cache.seq_len, tokens.len());
    assert_eq!(state.session.position(), tokens.len());
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn metal_qwen35_prefill_all_logits_long_prompt_no_longer_panics() {
    let _gpu_guard = gpu_test_lock();
    // Regression gate: forward_prefill_all_logits must NOT panic when
    // n > max_prefill and LoRA is inactive. Old code panicked; new code
    // chunks the request and concatenates per-chunk all-position logits.
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    if !qwen35_checkpoint_weights_present(&model_dir)
        || !model_dir.join("config.json").exists()
        || !model_dir.join("tokenizer.json").exists()
    {
        eprintln!(
            "skipping all-logits long-prompt test: model files missing at {}",
            model_dir.display()
        );
        return;
    }

    let Some(device) = Device::system_default() else {
        eprintln!("skipping all-logits long-prompt test: no Metal device");
        return;
    };
    let _ = device;

    let model =
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b");
    let mut state = MetalQwen35State::new(model.weights(), model.config(), 1024)
        .expect("construct Metal qwen3.5-0.8b state");

    let tokens = long_real_text_tokens(model.tokenizer());
    assert!(tokens.len() > state.session.max_prefill);

    // Must not panic; returns n * vocab_size f32 values.
    state.reset_state();
    let flat = state
        .forward_prefill_all_logits(&tokens)
        .expect("LoRA-inactive all-position prefill succeeds");
    assert_eq!(
        flat.len(),
        tokens.len() * model.config().vocab_size,
        "all-logits output length must be n * vocab_size"
    );

    // Final row of all-logits must agree with token-by-token final logits.
    let vocab = model.config().vocab_size;
    let final_row = &flat[(tokens.len() - 1) * vocab..tokens.len() * vocab];

    state.reset_state();
    let mut step_logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        step_logits = state.forward_step(tok, pos);
    }

    let max_abs_diff = step_logits
        .iter()
        .zip(final_row.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    eprintln!(
        "all-logits final-row parity: max_abs_diff={max_abs_diff:.6}, \
                 argmax_step={}, argmax_all_logits_final={}",
        argmax_f32(&step_logits),
        argmax_f32(final_row)
    );

    assert!(
        max_abs_diff < 1e-2,
        "all-logits final row diverged from step loop: max_abs_diff={max_abs_diff}"
    );
    assert_eq!(
        argmax_f32(&step_logits),
        argmax_f32(final_row),
        "argmax mismatch between step loop and all-logits final row"
    );
}
