//! Multimodal vision generation: cancellation, injected embeddings and M-RoPE.

use super::*;
/// `generate_multimodal` with `max_new_tokens == 0` must return zero
/// generated tokens without running prefill or sampling (#612 sibling-
/// invocation-path's 4th guard site; #621 tracks its
/// missing mutation-sensitive coverage).
///
/// Unlike the CPU-side zero-budget tests (which use empty weight vecs
/// to make mutation trivially observable via a panic), Metal GPU state
/// needs real buffers, so this loads the real Qwen3.5-0.8B checkpoint.
/// Mutation sensitivity: if the `max_new_tokens == 0` guard were
/// removed, execution would fall through to `reset_state` plus real
/// prefill/sampling against the loaded weights and produce at least
/// one generated token, failing the `generated_tokens == 0` assertion
/// below.
#[test]
fn generate_multimodal_max_new_tokens_zero_returns_empty() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let Some(model) = load_real_qwen35_0_8b_or_skip() else {
        return;
    };
    let _guard = gpu_test_lock();

    let mut state =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");

    let visual_tokens = 4usize;
    let d_model = model.config().hidden_size;
    let text_tokens = vec![1u32, 2, 3];
    let input = crate::vision::MultimodalInput {
        patch_embeddings: vec![0.0f32; visual_tokens * d_model],
        raw_patches: 16,
        visual_tokens,
        d_model,
        text_tokens: text_tokens.clone(),
    };

    let tokenizer = minimal_bpe_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 0,
        ..Default::default()
    };

    let out = state
        .generate_multimodal(input, &tokenizer, &gen_cfg)
        .expect("max_new_tokens=0 must succeed, not error");

    assert!(
        out.token_ids.is_empty(),
        "max_new_tokens=0 must produce no token IDs, got {:?}",
        out.token_ids
    );
    assert_eq!(
        out.generated_tokens, 0,
        "max_new_tokens=0 must produce zero generated tokens"
    );
    assert_eq!(
        out.prompt_tokens,
        visual_tokens + text_tokens.len(),
        "prompt_tokens must equal visual_tokens + text_tokens.len()"
    );
    assert!(
        !out.stopped,
        "max_new_tokens=0 exit is budget exhaustion, not a stop condition"
    );
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Length),
        "max_new_tokens=0 must report StopReason::Length"
    );
}

/// Qwen3.5 vision (ADR-069 Metal S5): shared test fixture for the MP2/MP3
/// `generate_multimodal_vision` gates below. One image run, grid
/// `(1, 2*merge, 2*merge)` so it always expands to exactly 4 post-merger
/// rows regardless of the checkpoint's `spatial_merge_size` (mirrors
/// `multimodal.rs`'s `base_request()` CPU unit fixture shape). Real
/// checkpoint config (`image_token_id`, `vision_config`, `hidden_size`) so
/// `generate_multimodal_vision`'s checkpoint-binding guards pass.
#[cfg(test)]
fn vision_gate_fixture(
    cfg: &Qwen35Config,
    seed: f32,
) -> crate::vision::multimodal::Qwen35VisionRequest {
    let vision_cfg = cfg
        .vision_config
        .as_ref()
        .expect("real 0.8b checkpoint carries vision_config");
    let image_token_id = cfg
        .image_token_id
        .expect("real 0.8b checkpoint carries image_token_id");
    let merge = vision_cfg.spatial_merge_size;
    let grid = crate::vision::qwen35_vit::GridThw {
        t: 1,
        h: 2 * merge,
        w: 2 * merge,
    };
    let num_rows = (grid.t * grid.h * grid.w) / (merge * merge);

    let mut input_ids = vec![10u32, 11];
    input_ids.extend(std::iter::repeat_n(image_token_id, num_rows));
    input_ids.push(12);

    // Non-constant per-lane content: a per-visual-row constant vector is
    // invariant under Qwen3.5's `(1+gamma) * x / rms(x)` RMSNorm (a
    // positive constant vector normalizes to the same direction
    // regardless of its magnitude), which would make an additive
    // mutation invisible one layer in. Vary by lane index instead.
    let mut post_merger_rows = vec![0.0f32; num_rows * cfg.hidden_size];
    for (i, v) in post_merger_rows.iter_mut().enumerate() {
        *v = seed + 0.01 * ((i % 97) as f32 - 48.0);
    }

    Qwen35VisionRequest {
        input_ids,
        image_grids: vec![grid],
        post_merger_rows,
        image_token_id,
        spatial_merge_size: merge,
        decoder_hidden_size: cfg.hidden_size,
    }
}

fn assert_vision_state_reusable_after_cancel(
    state: &mut MetalQwen35State,
    weights: &ModelWeights,
    cfg: &Qwen35Config,
    request: &Qwen35VisionRequest,
    tokenizer: &BpeTokenizer,
    gen_cfg: &GenerateConfig,
) {
    let resumed = state
        .generate_multimodal_vision(request, tokenizer, gen_cfg)
        .expect("same state must remain usable after cancellation");
    let mut fresh = MetalQwen35State::new(weights, cfg, 32).expect("fresh tiny vision state");
    let expected = fresh
        .generate_multimodal_vision(request, tokenizer, gen_cfg)
        .expect("fresh reference generation succeeds");
    assert_eq!(
        resumed.token_ids, expected.token_ids,
        "generation after cancellation must match a fresh state"
    );
    assert_eq!(
        resumed.stop_reason, expected.stop_reason,
        "generation after cancellation must preserve the fresh stop reason"
    );
}

#[test]
fn generate_multimodal_vision_cancel_immediate_preserves_and_reuses_state() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_vision_fixture();
    let request = vision_gate_fixture(&cfg, 0.01);
    let tokenizer = minimal_bpe_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        ..Default::default()
    };
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny vision state");

    let _ = state.forward_step(3, 0);
    let retained_seq_len = state.session.kv_cache.seq_len;
    assert!(retained_seq_len > 0, "precondition: state carries live KV");

    let mut polls = 0usize;
    let out = state
        .generate_multimodal_vision_with_cancel(&request, &tokenizer, &gen_cfg, || {
            polls += 1;
            true
        })
        .expect("immediate cancellation returns an output");
    assert_eq!(
        polls, 1,
        "immediate cancellation must stop at the first poll"
    );
    assert_eq!(out.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(out.generated_tokens, 0);
    assert!(
        !out.stopped,
        "client cancellation is not an OpenAI stop condition"
    );
    assert_eq!(
        state.session.kv_cache.seq_len, retained_seq_len,
        "immediate cancellation must happen before reset or prefill mutates state"
    );

    assert_vision_state_reusable_after_cancel(
        &mut state, &weights, &cfg, &request, &tokenizer, &gen_cfg,
    );
}

#[test]
fn generate_multimodal_vision_cancel_during_prefill_stops_before_next_token() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_vision_fixture();
    let request = vision_gate_fixture(&cfg, 0.01);
    let tokenizer = minimal_bpe_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        ..Default::default()
    };
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny vision state");

    let mut polls = 0usize;
    let out = state
        .generate_multimodal_vision_with_cancel(&request, &tokenizer, &gen_cfg, || {
            polls += 1;
            polls == 3
        })
        .expect("prefill cancellation returns an output");
    assert_eq!(polls, 3);
    assert_eq!(out.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(out.generated_tokens, 0);
    assert_eq!(
        state.session.kv_cache.seq_len, 1,
        "only the first prompt token may reach the decoder"
    );

    assert_vision_state_reusable_after_cancel(
        &mut state, &weights, &cfg, &request, &tokenizer, &gen_cfg,
    );
}

#[test]
fn generate_multimodal_vision_cancel_mid_decode_is_bounded_and_reuses_state() {
    let Some(_) = Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_metal_qwen35_vision_fixture();
    let request = vision_gate_fixture(&cfg, 0.01);
    let prompt_len = request.input_ids.len();
    let tokenizer = minimal_bpe_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        ..Default::default()
    };
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny vision state");

    let cancel_poll = prompt_len + 4;
    let mut polls = 0usize;
    let out = state
        .generate_multimodal_vision_with_cancel(&request, &tokenizer, &gen_cfg, || {
            polls += 1;
            polls == cancel_poll
        })
        .expect("mid-decode cancellation returns an output");
    assert_eq!(polls, cancel_poll);
    assert_eq!(out.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(
        out.generated_tokens, 2,
        "one sampled token and one decoded token must precede cancellation"
    );
    assert!(out.generated_tokens < gen_cfg.max_new_tokens);
    assert_eq!(
        state.session.kv_cache.seq_len,
        prompt_len + 1,
        "only one autoregressive decode step may reach the decoder"
    );

    assert_vision_state_reusable_after_cancel(
        &mut state, &weights, &cfg, &request, &tokenizer, &gen_cfg,
    );
}

/// Qwen3.5 vision (ADR-069 Metal S5, MP2 gate): isolates `forward_step_injected`
/// from the KV-cache history / decode-loop / sampling machinery entirely by
/// calling it once at position 0 on a freshly reset state, so any
/// insensitivity to row content can only come from the injection write
/// itself or the single-step layer pass — not from attention weighting
/// across later positions (which
/// `generate_multimodal_vision_injection_is_mutation_sensitive` below is
/// exposed to, with a synthetic 7-token prompt and only 2 greedy decode
/// steps).
#[test]
fn forward_step_injected_logits_differ_by_row_content() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "forward_step_injected_logits_differ_by_row_content",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = model.config().clone();
    let hidden = cfg.hidden_size;

    let mut row_a = vec![0.0f32; hidden];
    for (i, v) in row_a.iter_mut().enumerate() {
        *v = 0.01 + 0.01 * ((i % 97) as f32 - 48.0);
    }
    let row_b: Vec<f32> = row_a.iter().map(|v| -v).collect();

    let mut state =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    state.reset_state();
    let logits_a = state.forward_step_injected(&row_a, 0);
    state.reset_state();
    let logits_b = state.forward_step_injected(&row_b, 0);

    let diff: f32 = logits_a
        .iter()
        .zip(&logits_b)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "forward_step_injected output insensitive to row content: diff={diff}"
    );
}

/// Qwen3.5 vision (ADR-069 Metal S5, MP2 gate): mutation-sensitive test for
/// the residual-stream REPLACE injection at `<|image_pad|>` slots, mirroring
/// the CPU injection test's shape (cpu_f16.rs's `forward_step_f16`
/// `injected_embedding` coverage).
///
/// Mutating the supplied visual rows changes Metal's greedy output — the
/// rows are genuinely consumed, not discarded like the pre-MP2 stub
/// (`forward_step(0, vt)`, which fed every visual position the same fixed
/// embedding regardless of `patch_embeddings`). All four image-pad rows
/// are negated together (rather than just the first) — with only one
/// 7-token synthetic prompt and 2 greedy decode steps, a single mutated
/// row's effect on the final argmax token can be diluted by attention over
/// the later, unmutated image-pad positions;
/// `forward_step_injected_logits_differ_by_row_content` above already
/// isolates and proves per-row sensitivity at the logit level without that
/// dilution.
///
/// REPLACE semantics (the injected row replaces rather than falls back to
/// a token-id embedding lookup) is proved separately, structurally, by
/// `injected_embedding_never_falls_back_to_token_lookup` below — this
/// checkpoint ties `embed_tokens` to the lm_head weight matrix
/// (`tie_word_embeddings=true`), so *any* row mutation shifts that row's
/// logit component through the tied GEMV regardless of whether it was
/// ever used for input embedding lookup; a behavioral "mutate an unrelated
/// row, expect no output change" test is unsound on a tied-embeddings
/// checkpoint; (confirmed directly: mutating `embed_tokens[0]` collapsed
/// greedy output to token 0 on every step purely through the lm_head GEMV
/// component, with no injection-path involvement at all).
///
/// Mutation-sensitivity verified locally per repo convention: reverting
/// `forward_step_inner_impl`'s injection branch (falling through to the
/// plain `embed_tokens[token_id]` lookup unconditionally, matching the old
/// stub's behavior) was confirmed to make this assertion fail before the
/// fix was restored.
#[test]
fn generate_multimodal_vision_injection_is_mutation_sensitive() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "generate_multimodal_vision_injection_is_mutation_sensitive",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = model.config().clone();
    let tokenizer = minimal_bpe_tokenizer();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 2,
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        ..Default::default()
    };

    let mut state =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");

    let request_a = vision_gate_fixture(&cfg, 0.01);
    let out_a = state
        .generate_multimodal_vision(&request_a, &tokenizer, &gen_cfg)
        .expect("baseline multimodal generate succeeds");

    // Negate every supplied visual row (a sign flip that survives RMSNorm,
    // unlike a uniform additive shift on an otherwise-constant row; see
    // `vision_gate_fixture`'s doc comment). All four image-pad rows are
    // mutated together so the perturbation isn't diluted by attention over
    // the unmutated later image-pad positions.
    let mut request_b = vision_gate_fixture(&cfg, 0.01);
    for v in &mut request_b.post_merger_rows {
        *v = -*v;
    }
    let out_b = state
        .generate_multimodal_vision(&request_b, &tokenizer, &gen_cfg)
        .expect("mutated-visual-row multimodal generate succeeds");
    assert_ne!(
        out_a.token_ids, out_b.token_ids,
        "mutating the supplied visual rows must change Metal's greedy output \
                 (visual rows are not being consumed)"
    );
}

/// Qwen3.5 vision (ADR-069 Metal S5, MP2 gate): REPLACE-semantics
/// structural check, complementing the behavioral mutation-sensitivity
/// test above. `forward_step_inner_impl`'s `Some(row) => { ... }`
/// embedding-lookup arm (taken whenever `injected_embedding` is set at an
/// `<|image_pad|>` slot) must never reference `token_id` or read from
/// `embed_tokens` — proving by construction that an injected row can never
/// fall back to the token-id embedding lookup the pre-MP2 stub used, no
/// matter what runtime token id happens to occupy the slot. A behavioral
/// equivalent (mutate `embed_tokens` at some row, expect no output change)
/// is unsound on this checkpoint: `tie_word_embeddings=true` means
/// `embed_tokens` also serves as the lm_head weight matrix
/// (`encode_final_head`'s `QuantFormat::Q8_0` GEMV binds it directly), so
/// mutating any row shifts that row's logit component regardless of
/// whether the row was ever read for input embedding.
///
/// Mutation-sensitivity verified locally: temporarily adding a fallback
/// read of `embed_tokens[token_id]` inside the `Some(row) =>` arm (mirroring
/// the pre-MP2 stub) made this test fail immediately; removing it restored
/// the pass.
#[test]
fn injected_embedding_never_falls_back_to_token_lookup() {
    const RUST_SOURCE: &str = include_str!("../../../metal_qwen35.rs");
    let match_start = RUST_SOURCE
        .find("match injected_embedding {")
        .expect("forward_step_inner_impl's injected_embedding match must exist");
    let some_start = match_start
        + RUST_SOURCE[match_start..]
            .find("Some(row) => {")
            .expect("Some(row) arm must exist");
    let none_start = some_start
        + RUST_SOURCE[some_start..]
            .find("None => {")
            .expect("None arm must follow Some(row) arm");
    let some_arm_body = &RUST_SOURCE[some_start..none_start];
    assert!(
        !some_arm_body.contains("embed_tokens") && !some_arm_body.contains("token_id"),
        "the injected_embedding Some(row) arm must never reference embed_tokens \
                 or token_id — it must REPLACE the embedding lookup, not fall back to it"
    );
}

/// Qwen3.5 vision (ADR-069 Metal S5, MP2 gate): behavioral REPLACE
/// isolation test, complementing the structural spelling check above.
/// A source-scan test can be defeated by helper indirection (e.g. a
/// token-id-derived alias computed before the match and consumed by
/// a helper the `Some(row)` arm calls); this test instead proves the
/// REPLACE contract by calling `forward_step_inner_impl` directly
/// with the SAME injected row but DIFFERENT `token_id` values at the
/// same position and asserting byte-identical logits. Under REPLACE
/// semantics the token id occupying an injected slot can never
/// influence the step, regardless of buffer layout or aliasing — so
/// this is sound even on this tie_word_embeddings checkpoint where
/// the `embed_tokens`-mutation design (mentioned in the structural
/// test's doc comment above) is not: mutating `embed_tokens` here
/// would also perturb the lm_head GEMV, which reads the *same*
/// buffer for `QuantFormat::Q8_0`, and would falsely look like a
/// REPLACE violation. Varying only `token_id` (never touching
/// `embed_tokens`) sidesteps that confound entirely.
///
/// Mutation-sensitivity verified locally: temporarily adding a
/// fallback read of `embed_tokens[token_id]` inside the `Some(row) =>`
/// arm (mirroring the pre-MP2 stub) made this test fail immediately
/// (different token ids produced different logits); removing it
/// restored the pass.
#[test]
fn injected_embedding_logits_are_independent_of_token_id() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "injected_embedding_logits_are_independent_of_token_id",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = model.config().clone();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    let mut row = vec![0.0f32; hidden];
    for (i, v) in row.iter_mut().enumerate() {
        *v = 0.02 + 0.01 * ((i % 89) as f32 - 44.0);
    }

    let mut state =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");

    state.reset_state();
    let logits_id0 = state
        .forward_step_inner_impl(
            0,
            0,
            false,
            false,
            true,
            GdnStateTrafficScope::Decode,
            Some(&row),
            None,
            crate::forward::signpost::Scope::NotDecode,
        )
        .logits;

    state.reset_state();
    let other_token_id = vocab.saturating_sub(1) as u32;
    let logits_id_other = state
        .forward_step_inner_impl(
            other_token_id,
            0,
            false,
            false,
            true,
            GdnStateTrafficScope::Decode,
            Some(&row),
            None,
            crate::forward::signpost::Scope::NotDecode,
        )
        .logits;

    assert_eq!(logits_id0.len(), logits_id_other.len());
    let max_abs_diff = logits_id0
        .iter()
        .zip(&logits_id_other)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff < 1e-6,
        "an injected row's logits must be independent of the token_id occupying the \
                 slot — the injected_embedding arm must REPLACE the embedding lookup, never \
                 blend or fall back to it (max_abs_diff={max_abs_diff})"
    );
}

/// Qwen3.5 vision (ADR-069 Metal MP3 gate, test 1/4 — position
/// sensitivity): the SAME injected embedding row at two different
/// M-RoPE `(t, h, w)` positions must yield DIFFERENT logits.
///
/// `injected_embedding_logits_are_independent_of_token_id` above proves
/// the injected row is consumed regardless of which token id occupies
/// the slot, but it always ran at physical position 0 with no M-RoPE
/// table — it cannot catch a regression where MP3's wiring silently
/// ignores the supplied per-token cos/sin row (e.g. always dispatching
/// `pos_offset=0` into the override buffer, or falling back to
/// `engine.rope_cos`/`rope_sin`). This test drives two DIFFERENT
/// `(t, h, w)` triples (built the same way `build_mrope_tables` would
/// for an image spanning two rows) through `forward_step_injected_mrope`
/// at the SAME physical position, so only the M-RoPE row differs
/// between the two calls; the two logit vectors must differ.
///
/// Mutation-sensitivity verified locally: forcing `dispatch_partial_rope`
/// to ignore `mrope_override` and always take the `None` branch (the
/// pre-MP3 1-D fallback) collapsed both calls to identical logits
/// (`max_abs_diff` fell under `1e-6`), failing this assertion; restoring
/// the override branch made it pass again.
///
/// Both calls run after two identical priming tokens (not at a bare
/// reset). With an empty KV cache the injected token is the only
/// cached key, and softmax over a single score is always exactly
/// `1.0` regardless of the Q/K dot product — so a single-token
/// attention step is invariant to ANY Q/K rotation and cannot
/// discriminate this bug. Priming the cache to length 2 first makes
/// softmax weigh 3 distinct scores, where a rotation genuinely shifts
/// the attention distribution.
#[test]
fn forward_step_injected_mrope_logits_differ_by_position() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "forward_step_injected_mrope_logits_differ_by_position",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = model.config().clone();
    let hidden = cfg.hidden_size;

    let mut row = vec![0.0f32; hidden];
    for (i, v) in row.iter_mut().enumerate() {
        *v = 0.02 + 0.01 * ((i % 89) as f32 - 44.0);
    }

    let mrope_section = cfg
        .rope_parameters
        .as_ref()
        .and_then(|p| p.mrope_section.clone())
        .expect("real 0.8b checkpoint carries mrope_section");
    let theta = cfg
        .rope_parameters
        .as_ref()
        .map(|p| p.rope_theta as f32)
        .expect("real 0.8b checkpoint carries rope_theta");
    let partial_rotary_factor = cfg
        .rope_parameters
        .as_ref()
        .and_then(|p| p.partial_rotary_factor)
        .unwrap_or(0.25);

    let row_for = |t: u32, h: u32, w: u32| -> (Vec<f32>, Vec<f32>) {
        let positions = crate::vision::qwen35_mrope::MRopePositions {
            positions: vec![(t, h, w)],
            rope_delta: 0,
        };
        let tables = crate::vision::qwen35_mrope::build_cos_sin(
            &positions,
            cfg.head_dim,
            partial_rotary_factor,
            theta,
            &mrope_section,
        )
        .expect("valid checkpoint M-RoPE config builds a table");
        (tables.cos[0].clone(), tables.sin[0].clone())
    };
    let (cos_a, sin_a) = row_for(4, 4, 4);
    let (cos_b, sin_b) = row_for(4, 11, 11);

    let mut state =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");

    let prime = |s: &mut MetalQwen35State| {
        s.reset_state();
        s.forward_step(20, 0);
        s.forward_step(21, 1);
    };

    prime(&mut state);
    let logits_a = state.forward_step_injected_mrope(
        &row,
        2,
        Some((&cos_a, &sin_a)),
        true,
        crate::forward::signpost::Scope::NotDecode,
    );
    prime(&mut state);
    let logits_b = state.forward_step_injected_mrope(
        &row,
        2,
        Some((&cos_b, &sin_b)),
        true,
        crate::forward::signpost::Scope::NotDecode,
    );

    let max_abs_diff = logits_a
        .iter()
        .zip(&logits_b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff > 1e-3,
        "the same injected row at two different M-RoPE (t,h,w) positions must \
                 yield different logits — M-RoPE is not being applied (max_abs_diff={max_abs_diff})"
    );
}

/// Qwen3.5 vision (ADR-069 Metal MP3 gate, test 2/4 — routing): proves
/// image-token positions actually go through the 3-axis M-RoPE table
/// built by `build_mrope_tables`, not the 1-D `engine.rope_cos`/
/// `rope_sin` fallback keyed by physical position.
///
/// The synthetic fixture's image spans physical positions where the
/// M-RoPE `(t, h, w)` triple diverges from the physical index (H/W
/// sweep within the merged grid), so the correct per-position cos/sin
/// row differs from what the 1-D table would produce at that same
/// physical position. Compares the FINAL PREFILL STEP's raw logits
/// (not a downstream sampled/greedy token after further decode steps —
/// argmax over a real 24-layer/151936-vocab model can be robust to a
/// small per-position logit shift, which would make a token-level
/// comparison an insensitive gate) between a real run (using
/// `tables.cos[pos]`/`tables.sin[pos]`, the exact per-position rows
/// `generate_multimodal_vision` builds) and a forced-1-D run (`None`
/// throughout, i.e. `engine.rope_cos`/`rope_sin` keyed by physical
/// position — what `generate_multimodal_vision` would silently do if
/// it discarded `tables` and never wired the override through).
#[test]
fn generate_multimodal_vision_routes_image_tokens_through_mrope_table() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "generate_multimodal_vision_routes_image_tokens_through_mrope_table",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = model.config().clone();
    let request = vision_gate_fixture(&cfg, 0.01);
    let (_positions, tables) = request
        .build_mrope_tables(&cfg)
        .expect("fixture request builds M-RoPE tables");

    // Real path: every position rotates with its genuine `tables.cos[pos]`/
    // `tables.sin[pos]` row, exactly as `generate_multimodal_vision` does.
    let mut state_real =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    state_real.reset_state();
    let mut visual_row = 0usize;
    let mut last_logits_real = Vec::new();
    for (pos, &token_id) in request.input_ids.iter().enumerate() {
        let cos_sin = Some((tables.cos[pos].as_slice(), tables.sin[pos].as_slice()));
        if token_id == request.image_token_id {
            let start = visual_row * request.decoder_hidden_size;
            let end = start + request.decoder_hidden_size;
            let row = &request.post_merger_rows[start..end];
            visual_row += 1;
            last_logits_real = state_real.forward_step_injected_mrope(
                row,
                pos,
                cos_sin,
                true,
                crate::forward::signpost::Scope::NotDecode,
            );
        } else {
            last_logits_real = state_real.forward_step_mrope(
                token_id,
                pos,
                cos_sin,
                true,
                crate::forward::signpost::Scope::NotDecode,
            );
        }
    }

    // Forced-1-D path: identical injection sequence, but every call passes
    // `None` for `mrope_cos_sin`.
    let mut state_forced =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    state_forced.reset_state();
    let mut visual_row = 0usize;
    let mut last_logits_forced = Vec::new();
    for (pos, &token_id) in request.input_ids.iter().enumerate() {
        if token_id == request.image_token_id {
            let start = visual_row * request.decoder_hidden_size;
            let end = start + request.decoder_hidden_size;
            let row = &request.post_merger_rows[start..end];
            visual_row += 1;
            last_logits_forced = state_forced.forward_step_injected(row, pos);
        } else {
            last_logits_forced = state_forced.forward_step(token_id, pos);
        }
    }

    let max_abs_diff = last_logits_real
        .iter()
        .zip(&last_logits_forced)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff > 1e-3,
        "generate_multimodal_vision's prefill must route image-token positions \
                 through the 3-axis M-RoPE table, not the 1-D engine.rope_cos/rope_sin \
                 fallback — forcing the 1-D path produced indistinguishable final-prefill \
                 logits (max_abs_diff={max_abs_diff})"
    );
}

/// Issue #1336: mechanism-level proof that `emit_head=false` skips the
/// actual terminal RMSNorm + lm_head GPU dispatch, not merely its CPU
/// readback. Poisons the shared logits buffer with a sentinel pattern
/// before an `emit_head=false` step and asserts it is UNCHANGED
/// afterward — the only way the sentinel survives is if
/// `encode_final_head` (which writes this buffer) never ran.
///
/// This is the must-fail-arm control the equivalence test below
/// cannot provide on its own: both of that test's arms pass
/// `emit_head=true` at the position they compare, so an
/// implementation that ignores the flag and always dispatches the
/// head would still pass it trivially. Here, `emit_head=true` is the
/// positive control (must overwrite the sentinel) and `emit_head=
/// false` is the subject (must not).
///
/// Mutation-predicted: reverting `run_head`'s gating of
/// `self.encode_final_head(...)` in `forward_step_inner_impl` back to
/// an unconditional call reddens the `emit_head=false` assertion
/// below (the sentinel gets overwritten by the real lm_head GEMV
/// every time, regardless of the flag).
#[test]
fn forward_step_mrope_emit_head_false_skips_terminal_head_dispatch() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "forward_step_mrope_emit_head_false_skips_terminal_head_dispatch",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = model.config().clone();
    let probe_len = cfg.vocab_size.min(64);
    let sentinel: f32 = -12_345.679;
    let mut state =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");

    // SAFETY (both closures): `activations.logits` is StorageModeShared;
    // no GPU work is in flight between test steps (each
    // `forward_step_mrope` call waits for its own command buffer
    // before returning), and `probe_len <= vocab_size`.
    let poison = |state: &MetalQwen35State| unsafe {
        let ptr = state.session.activations.logits.contents() as *mut f32;
        for i in 0..probe_len {
            *ptr.add(i) = sentinel;
        }
    };
    let read_probe = |state: &MetalQwen35State| -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(
                state.session.activations.logits.contents() as *const f32,
                probe_len,
            )
            .to_vec()
        }
    };

    // Control: emit_head=true must overwrite the poison.
    state.reset_state();
    poison(&state);
    let control_logits =
        state.forward_step_mrope(7, 0, None, true, crate::forward::signpost::Scope::NotDecode);
    assert!(
        !control_logits.is_empty(),
        "emit_head=true must return full vocab logits"
    );
    let after_true = read_probe(&state);
    assert!(
        after_true.iter().any(|&v| v != sentinel),
        "control failed: emit_head=true must overwrite the poisoned logits buffer via \
                 the real lm_head dispatch — if this fails, the poison/readback mechanism \
                 itself is broken and the subject assertion below is meaningless"
    );

    // Subject: emit_head=false must leave the poison untouched.
    state.reset_state();
    poison(&state);
    let subject_logits = state.forward_step_mrope(
        7,
        0,
        None,
        false,
        crate::forward::signpost::Scope::NotDecode,
    );
    assert!(
        subject_logits.is_empty(),
        "emit_head=false must return no logits: nothing was computed this step"
    );
    let after_false = read_probe(&state);
    assert_eq!(
        after_false,
        vec![sentinel; probe_len],
        "emit_head=false must skip the terminal RMSNorm + lm_head dispatch entirely — \
                 the logits buffer must remain untouched, not merely unread"
    );
}

/// Issue #1336 round 2 (F1): `run_head` must key off THIS request's
/// own MTP activation (`session.mtp_active`), not off whether the
/// loaded checkpoint merely carries MTP weights
/// (`session.mtp.is_some()`). A checkpoint can ship MTP weights
/// while a specific request runs with MTP disabled
/// (`enable_mtp: Some(false)`, or simply a caller — like
/// `generate_multimodal_vision_impl`'s prefill loop — that never
/// takes `generate()`'s `use_mtp` branch at all); that combination
/// must still honor `emit_head=false`.
///
/// Builds a session with MTP weights resident via the same
/// `metal_state_with_constant_zero_draft_mtp_for_test` fixture the
/// MTP-rollback tests above use (so `session.mtp.is_some() ==
/// true`), but drives `forward_step_mrope` directly — the same call
/// `generate_multimodal_vision_impl`'s prefill loop makes — without
/// ever calling `generate()`, so `session.mtp_active` stays at its
/// post-`reset_state()` default of `false`. This is exactly the
/// "MTP weights present, this request never activated MTP" case the
/// old `self.session.mtp.is_some()` gate could not distinguish from
/// "MTP weights present and this request is using them".
///
/// Reuses `forward_step_mrope_emit_head_false_skips_terminal_head_dispatch`'s
/// poison/read_probe mechanism above: `emit_head=false` already
/// returns empty logits by contract whether or not the head actually
/// ran, so the only way to observe a stray dispatch is to poison the
/// shared logits buffer first and assert it survives untouched.
///
/// Expected-redden check: before this fix, `run_head` computed
/// `emit_head || capture_hidden || self.session.mtp.is_some()`.
/// Reverting to that expression makes `run_head` evaluate to `true`
/// here purely from `session.mtp.is_some() == true` (both `emit_head`
/// and `capture_hidden` are `false`), so `encode_final_head` runs and
/// overwrites the sentinel — reddening only the final `assert_eq!`
/// below (the two `mtp_active` setup assertions above it stay green
/// either way, since they inspect the flag directly rather than
/// `run_head`'s output).
#[test]
fn forward_step_mrope_emit_head_false_skips_terminal_head_dispatch_with_mtp_weights_present_but_request_inactive()
 {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = Device::system_default() else {
        return;
    };

    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let mut state = metal_state_with_constant_zero_draft_mtp_for_test(&weights, &cfg);
    assert!(
        state.session.mtp.is_some(),
        "fixture setup: this test requires MTP weights to be resident — otherwise it \
                 cannot distinguish the fixed gate from the pre-fix one"
    );
    assert!(
        !state.session.mtp_active,
        "fixture setup: a freshly-constructed session must not report MTP as active \
                 for the request merely because the checkpoint carries MTP weights"
    );

    let probe_len = cfg.vocab_size.min(64);
    let sentinel: f32 = -12_345.679;

    // SAFETY: same as `forward_step_mrope_emit_head_false_skips_terminal_head_dispatch`
    // above — `activations.logits` is StorageModeShared, no GPU work is in
    // flight between test steps, and probe_len <= vocab_size.
    let poison = |state: &MetalQwen35State| unsafe {
        let ptr = state.session.activations.logits.contents() as *mut f32;
        for i in 0..probe_len {
            *ptr.add(i) = sentinel;
        }
    };
    let read_probe = |state: &MetalQwen35State| -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(
                state.session.activations.logits.contents() as *const f32,
                probe_len,
            )
            .to_vec()
        }
    };

    state.reset_state();
    assert!(
        state.session.mtp.is_some(),
        "reset_state must not clear MTP weight residency"
    );
    assert!(
        !state.session.mtp_active,
        "reset_state must clear any stale per-request MTP activation from a prior call"
    );
    poison(&state);

    let subject_logits = state.forward_step_mrope(
        7,
        0,
        None,
        false,
        crate::forward::signpost::Scope::NotDecode,
    );
    assert!(
        subject_logits.is_empty(),
        "emit_head=false must return no logits regardless of whether MTP weights are \
                 resident on the checkpoint"
    );
    let after_false = read_probe(&state);
    assert_eq!(
        after_false,
        vec![sentinel; probe_len],
        "emit_head=false must skip the terminal RMSNorm + lm_head dispatch even when \
                 the checkpoint carries MTP weights and this request never activated MTP — \
                 gating on `self.session.mtp.is_some()` instead of the request's own \
                 `mtp_active` flag would dispatch the head here and overwrite the sentinel, \
                 which is exactly the #1336-round-2 defeat this test guards against"
    );
}

/// Issue #1336: `emit_head=false` on a non-final multimodal prefill
/// position must not change what that position contributes to the
/// residual stream, KV cache, or GDN recurrent state — only the
/// terminal RMSNorm/lm_head/vocab readback it skips. Runs the same
/// `vision_gate_fixture` prefill twice: once with `emit_head=true` at
/// every position (the "always-emitting reference path" the issue
/// calls out), once with `emit_head=false` at every position except
/// the last (mirrors `generate_multimodal_vision_impl`'s production
/// loop). Compares:
///  - final-position logits: bit-identical
///  - GDN recurrent state (every linear layer's conv + S-matrix
///    buffers, via `MtpTargetVerifier::snapshot_gdn_states`):
///    bit-identical
///  - KV cache (every full-attention layer's written prefix):
///    byte-identical
///  - a subsequent decode step off each final state: bit-identical
///    logits — the exact consumer a real state divergence would
///    reach.
///
/// This test alone cannot catch an `emit_head` no-op that always
/// dispatches the head (both arms compare identically in that case,
/// trivially) — that failure mode is covered by
/// `forward_step_mrope_emit_head_false_skips_terminal_head_dispatch`
/// above. Together they prove both halves the issue asks for: the
/// skip is real (mechanism test) and it is safe (this test).
#[test]
fn generate_multimodal_vision_emit_head_skip_matches_always_emit_reference() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "generate_multimodal_vision_emit_head_skip_matches_always_emit_reference",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();
    use crate::speculative::MtpTargetVerifier as _;

    let cfg = model.config().clone();
    let request = vision_gate_fixture(&cfg, 0.02);
    let (_positions, tables) = request
        .build_mrope_tables(&cfg)
        .expect("fixture request builds M-RoPE tables");
    let prompt_len = request.input_ids.len();

    let run_prefill = |always_emit: bool| -> (Vec<f32>, MetalQwen35State) {
        let mut state = MetalQwen35State::new(model.weights(), model.config(), 128)
            .expect("real-checkpoint state");
        state.reset_state();
        let mut visual_row = 0usize;
        let mut last_logits = Vec::new();
        for (pos, &token_id) in request.input_ids.iter().enumerate() {
            let cos_sin = Some((tables.cos[pos].as_slice(), tables.sin[pos].as_slice()));
            let emit_head = always_emit || pos + 1 == prompt_len;
            if token_id == request.image_token_id {
                let start = visual_row * request.decoder_hidden_size;
                let end = start + request.decoder_hidden_size;
                let row = &request.post_merger_rows[start..end];
                visual_row += 1;
                last_logits = state.forward_step_injected_mrope(
                    row,
                    pos,
                    cos_sin,
                    emit_head,
                    crate::forward::signpost::Scope::NotDecode,
                );
            } else {
                last_logits = state.forward_step_mrope(
                    token_id,
                    pos,
                    cos_sin,
                    emit_head,
                    crate::forward::signpost::Scope::NotDecode,
                );
            }
        }
        (last_logits, state)
    };

    let (logits_reference, mut state_reference) = run_prefill(true);
    let (logits_optimized, mut state_optimized) = run_prefill(false);

    // Final-logit equivalence: bit-identical.
    assert!(
        !logits_reference.is_empty(),
        "fixture's final position must produce real vocab logits"
    );
    assert_eq!(logits_reference.len(), logits_optimized.len());
    for (i, (a, b)) in logits_reference.iter().zip(&logits_optimized).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "final-position logit {i} diverged between the always-emit reference and \
                     the emit_head-skip optimization ({a} vs {b})"
        );
    }

    // GDN recurrent-state equivalence.
    let gdn_reference = state_reference.snapshot_gdn_states();
    let gdn_optimized = state_optimized.snapshot_gdn_states();
    assert_eq!(
        gdn_reference, gdn_optimized,
        "GDN recurrent state (conv + S-matrix buffers) diverged between the \
                 always-emit reference and the emit_head-skip optimization"
    );

    // KV-cache equivalence: every full-attention layer's written prefix.
    let kv_row_bytes = if state_reference.use_kv_f16 {
        2usize
    } else {
        4usize
    };
    let used_bytes = prompt_len * cfg.full_kv_dim() * kv_row_bytes;
    let num_full_layers = state_reference.session.kv_cache.k_bufs.len();
    for i in 0..num_full_layers {
        // SAFETY: both buffers are StorageModeShared; every command
        // buffer in `run_prefill`'s loop above was waited on before
        // the next step began (see `forward_step_inner_impl`).
        unsafe {
            let k_ref = std::slice::from_raw_parts(
                state_reference.session.kv_cache.k_bufs[i].contents() as *const u8,
                used_bytes,
            );
            let k_opt = std::slice::from_raw_parts(
                state_optimized.session.kv_cache.k_bufs[i].contents() as *const u8,
                used_bytes,
            );
            assert_eq!(k_ref, k_opt, "K cache layer {i} diverged");
            let v_ref = std::slice::from_raw_parts(
                state_reference.session.kv_cache.v_bufs[i].contents() as *const u8,
                used_bytes,
            );
            let v_opt = std::slice::from_raw_parts(
                state_optimized.session.kv_cache.v_bufs[i].contents() as *const u8,
                used_bytes,
            );
            assert_eq!(v_ref, v_opt, "V cache layer {i} diverged");
        }
    }

    // Decode continuation: the exact consumer "state equivalence"
    // exists for — a subsequent decode step must sample identically
    // off either state.
    let physical_pos_ref = state_reference.session.kv_cache.seq_len;
    let physical_pos_opt = state_optimized.session.kv_cache.seq_len;
    assert_eq!(
        physical_pos_ref, physical_pos_opt,
        "KV cursor position diverged between the two prefill arms"
    );
    let decode_token = 5u32;
    let decode_logits_ref = state_reference.forward_step_mrope(
        decode_token,
        physical_pos_ref,
        None,
        true,
        crate::forward::signpost::Scope::Decode,
    );
    let decode_logits_opt = state_optimized.forward_step_mrope(
        decode_token,
        physical_pos_opt,
        None,
        true,
        crate::forward::signpost::Scope::Decode,
    );
    assert_eq!(decode_logits_ref.len(), decode_logits_opt.len());
    for (i, (a, b)) in decode_logits_ref.iter().zip(&decode_logits_opt).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "post-prefill decode-step logit {i} diverged — KV/GDN state after the \
                     emit_head-skip optimization is not equivalent to the always-emit \
                     reference ({a} vs {b})"
        );
    }
}

/// Qwen3.5 vision (ADR-069 Metal MP3 gate, test 3/4 — text-only
/// bit-identity): a text-only request through `generate_multimodal_vision`
/// (no image runs) must produce bit-identical logits/tokens to the plain
/// `generate_streaming`/`forward_step` path, following the pattern of
/// `generate_multimodal_text_only_matches_plain_forward_bit_identical`
/// in `cpu_f16.rs`. `has_image=false` passes `None` for `mrope_cos_sin`
/// throughout, which keeps every `dispatch_partial_rope` call on the
/// unchanged `engine.rope_cos`/`rope_sin` 1-D table — this proves that
/// structural claim behaviorally.
#[test]
fn generate_multimodal_vision_text_only_matches_plain_forward_bit_identical() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "generate_multimodal_vision_text_only_matches_plain_forward_bit_identical",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = model.config().clone();
    let tokenizer = minimal_bpe_tokenizer();
    let input_ids: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 16];

    let request = crate::vision::multimodal::Qwen35VisionRequest {
        input_ids: input_ids.clone(),
        image_grids: vec![],
        post_merger_rows: vec![],
        image_token_id: cfg
            .image_token_id
            .expect("real 0.8b checkpoint carries image_token_id"),
        spatial_merge_size: cfg
            .vision_config
            .as_ref()
            .expect("real 0.8b checkpoint carries vision_config")
            .spatial_merge_size,
        decoder_hidden_size: cfg.hidden_size,
    };
    request.validate().expect("text-only request is valid");

    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        ..Default::default()
    };

    let mut state_vision =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    let out_vision = state_vision
        .generate_multimodal_vision(&request, &tokenizer, &gen_cfg)
        .expect("text-only multimodal generate succeeds");

    let mut state_plain =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    state_plain.reset_state();
    let mut all_ids = input_ids.clone();
    let mut last_logits = Vec::new();
    for (pos, &token_id) in input_ids.iter().enumerate() {
        last_logits = state_plain.forward_step(token_id, pos);
    }
    let mut rng_state = 1u64;
    let first_id = sample_token(&last_logits, &gen_cfg, &all_ids, &mut rng_state);
    all_ids.push(first_id);
    let mut plain_tokens = vec![first_id];
    for _ in 1..gen_cfg.max_new_tokens {
        let physical_pos = state_plain.session.kv_cache.seq_len;
        let last_token = *all_ids.last().expect("prompt or sample populated all_ids");
        let step_logits = state_plain.forward_step(last_token, physical_pos);
        let next_id = sample_token(&step_logits, &gen_cfg, &all_ids, &mut rng_state);
        plain_tokens.push(next_id);
        all_ids.push(next_id);
    }

    assert_eq!(
        out_vision.token_ids, plain_tokens,
        "text-only generate_multimodal_vision must be bit-identical to the plain \
                 forward_step decode loop"
    );

    // Equal greedy tokens can survive a non-bit-identical logit
    // regression, so token equality alone does not substantiate the
    // bit-identity contract. Drive the two entry points step-by-step
    // on fresh states and assert exact bit equality of every logit at
    // every prefill and decode step: `forward_step_mrope(.., None)` is
    // the exact call `generate_multimodal_vision` makes for text-only
    // requests, `forward_step` is the plain path.
    let mut state_mrope_none =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    let mut state_plain_bits =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    state_mrope_none.reset_state();
    state_plain_bits.reset_state();

    let assert_bits_equal = |a: &[f32], b: &[f32], step: usize| {
        assert_eq!(a.len(), b.len(), "logit width diverged at step {step}");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "step {step} logit {i}: text-only mrope-None path must be \
                         bit-identical to the plain path ({x} vs {y})"
            );
        }
    };

    let mut bits_last = Vec::new();
    for (pos, &token_id) in input_ids.iter().enumerate() {
        let a = state_mrope_none.forward_step_mrope(
            token_id,
            pos,
            None,
            true,
            crate::forward::signpost::Scope::NotDecode,
        );
        let b = state_plain_bits.forward_step(token_id, pos);
        assert_bits_equal(&a, &b, pos);
        bits_last = a;
    }
    let mut bits_ids = input_ids.clone();
    let mut bits_rng = 1u64;
    for step in 0..gen_cfg.max_new_tokens {
        let next_id = sample_token(&bits_last, &gen_cfg, &bits_ids, &mut bits_rng);
        bits_ids.push(next_id);
        let physical_pos = state_mrope_none.session.kv_cache.seq_len;
        assert_eq!(
            physical_pos, state_plain_bits.session.kv_cache.seq_len,
            "physical KV positions diverged before decode step {step}"
        );
        let a = state_mrope_none.forward_step_mrope(
            next_id,
            physical_pos,
            None,
            true,
            crate::forward::signpost::Scope::NotDecode,
        );
        let b = state_plain_bits.forward_step(next_id, physical_pos);
        assert_bits_equal(&a, &b, input_ids.len() + step);
        bits_last = a;
    }
}

/// Qwen3.5 vision (ADR-069 Metal MP3 gate, decode-coordinate
/// discrimination): every prefill-oriented gate above still passes if
/// only the DECODE-time M-RoPE wiring regresses (dropping
/// `decode_cos_sin` and rotating decode steps by raw physical
/// position) — verified empirically: mutating
/// `generate_multimodal_vision`'s decode call to pass `None` left the
/// oracle-parity test green. Greedy TOKENS are not a usable
/// observable for this wiring at all: 18 of 24 layers are GDN (no
/// RoPE), and both a synthetic merged 8x8 grid (`rope_delta` -56,
/// 4-step horizon) and the real #989 fixture (32-step horizon)
/// produced token-identical trajectories on the correct and
/// `None`-decode arms. So this test pins the function at LOGIT level
/// through the `generate_multimodal_vision_impl` decode probe:
///
/// - A manual replication of the function's exact prefill+first
///   decode step is run twice: once with the correct decode
///   coordinate (`decode_position(physical_pos, rope_delta)` +
///   `build_decode_cos_sin`) and once simulating the regression
///   (`None`, 1-D table keyed by physical position). The vacuity
///   guard requires their decode-step logits to differ by a
///   margin-safe amount — proving the fixture discriminates the two
///   behaviors at the observable this test actually asserts on.
/// - The function's own probed decode-step logits must be
///   bit-identical to the correct arm (Metal execution is
///   deterministic across freshly built states; the text-only
///   bit-identity test above establishes that). Mutation-verified:
///   forcing the function's decode call to `None` makes exactly this
///   assertion fail; restoring it passes.
#[test]
fn generate_multimodal_vision_decode_step_uses_rope_delta_coordinate() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "generate_multimodal_vision_decode_step_uses_rope_delta_coordinate",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let cfg = model.config().clone();
    let tokenizer = minimal_bpe_tokenizer();

    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("vision");
    if !fixture_dir.join("manifest.json").exists() {
        eprintln!(
            "[METAL_TEST_SKIP] context=generate_multimodal_vision_decode_step_uses_rope_delta_coordinate \
                     reason=no_vision_fixture path={}",
            fixture_dir.display()
        );
        let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but the #989 vision fixture is missing at {}",
            fixture_dir.display()
        );
        return;
    }

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture_dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    let input_ids: Vec<u32> = serde_json::from_slice::<Vec<i64>>(
        &std::fs::read(fixture_dir.join("input_ids.json")).expect("read input_ids.json"),
    )
    .expect("parse input_ids.json")
    .into_iter()
    .map(|v| v as u32)
    .collect();
    let post_merger_bytes = std::fs::read(fixture_dir.join("vit_post_merger_f32.bin"))
        .expect("read vit_post_merger_f32.bin");
    let post_merger: Vec<f32> = post_merger_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let grid_thw = manifest["image_grid_thw"]
        .as_array()
        .expect("image_grid_thw array")
        .first()
        .expect("at least one image grid")
        .as_array()
        .expect("grid triple");
    let grid = crate::vision::qwen35_vit::GridThw {
        t: grid_thw[0].as_u64().unwrap() as usize,
        h: grid_thw[1].as_u64().unwrap() as usize,
        w: grid_thw[2].as_u64().unwrap() as usize,
    };
    let vision_cfg = cfg
        .vision_config
        .as_ref()
        .expect("real 0.8b checkpoint carries vision_config");
    let image_token_id = cfg
        .image_token_id
        .expect("real 0.8b checkpoint carries image_token_id");
    let request = Qwen35VisionRequest {
        input_ids,
        image_grids: vec![grid],
        post_merger_rows: post_merger,
        image_token_id,
        spatial_merge_size: vision_cfg.spatial_merge_size,
        decoder_hidden_size: cfg.hidden_size,
    };
    request
        .validate()
        .expect("fixture-derived request is valid");

    let (positions, tables) = request
        .build_mrope_tables(&cfg)
        .expect("fixture request builds M-RoPE tables");
    assert_ne!(
        positions.rope_delta, 0,
        "fixture must have a nonzero rope_delta or this test cannot \
                 discriminate the decode coordinate from the physical position"
    );

    let gen_cfg = GenerateConfig {
        max_new_tokens: 2,
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        ..Default::default()
    };

    // Manual replication of `generate_multimodal_vision`'s exact
    // prefill + first greedy decode step, with the decode-time M-RoPE
    // coordinate either wired correctly or dropped to `None`. Returns
    // that decode step's full logits row.
    let manual_decode_logits = |use_decode_coordinate: bool| -> Vec<f32> {
        let mut state = MetalQwen35State::new(model.weights(), model.config(), 128)
            .expect("real-checkpoint state");
        state.reset_state();
        let mut visual_row = 0usize;
        let mut last_logits = Vec::new();
        for (pos, &token_id) in request.input_ids.iter().enumerate() {
            let cos_sin = Some((tables.cos[pos].as_slice(), tables.sin[pos].as_slice()));
            if token_id == request.image_token_id {
                let start = visual_row * request.decoder_hidden_size;
                let end = start + request.decoder_hidden_size;
                let row = &request.post_merger_rows[start..end];
                visual_row += 1;
                last_logits = state.forward_step_injected_mrope(
                    row,
                    pos,
                    cos_sin,
                    true,
                    crate::forward::signpost::Scope::NotDecode,
                );
            } else {
                last_logits = state.forward_step_mrope(
                    token_id,
                    pos,
                    cos_sin,
                    true,
                    crate::forward::signpost::Scope::NotDecode,
                );
            }
        }
        let mut all_ids = request.input_ids.clone();
        let mut rng_state = 1u64;
        let first_id = sample_token(&last_logits, &gen_cfg, &all_ids, &mut rng_state);
        assert_ne!(
            first_id, cfg.eos_token_id,
            "fixture's first greedy token must not be EOS or no decode step runs"
        );
        all_ids.push(first_id);
        let physical_pos = state.session.kv_cache.seq_len;
        let decode_row_owned;
        let cos_sin = if use_decode_coordinate {
            let decode_axis =
                crate::vision::qwen35_mrope::decode_position(physical_pos, positions.rope_delta)
                    .expect("fixture decode coordinate is valid");
            decode_row_owned = request
                .build_decode_cos_sin(&cfg, decode_axis)
                .expect("fixture decode row builds");
            Some((decode_row_owned.0.as_slice(), decode_row_owned.1.as_slice()))
        } else {
            None
        };
        state.forward_step_mrope(
            first_id,
            physical_pos,
            cos_sin,
            true,
            crate::forward::signpost::Scope::NotDecode,
        )
    };

    let logits_correct = manual_decode_logits(true);
    let logits_regressed = manual_decode_logits(false);
    let max_abs_diff = logits_correct
        .iter()
        .zip(&logits_regressed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff > 1e-3,
        "vacuity guard: the fixture's first decode-step logits must \
                 discriminate the rope_delta-shifted coordinate from the raw \
                 physical position by a margin-safe amount (max_abs_diff = \
                 {max_abs_diff:e}) — without this, the bit-equality assert \
                 below could pass vacuously"
    );

    let mut state_fn =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    let mut probed: Vec<Vec<f32>> = Vec::new();
    let out = state_fn
        .generate_multimodal_vision_impl(&request, &tokenizer, &gen_cfg, Some(&mut probed), None)
        .expect("fixture multimodal generate succeeds");
    assert_eq!(
        out.generated_tokens, 2,
        "expected prefill-sampled token + one decode step"
    );
    assert_eq!(probed.len(), 1, "one decode step probes one logits row");
    assert_eq!(probed[0].len(), logits_correct.len());
    for (lane, (&got, &want)) in probed[0].iter().zip(&logits_correct).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "generate_multimodal_vision's decode step must use the \
                     rope_delta-shifted M-RoPE coordinate: probed logits diverge \
                     from the correct arm at lane {lane} ({got} vs {want})"
        );
    }
}

/// Qwen3.5 vision (ADR-069 Metal MP3 gate, test 4/4 — CPU-vs-Metal
/// parity vs the oracle): loads the committed #989 differential
/// fixture (`tests/fixtures/vision/`, the same one
/// `vision_s5b_e2e_gate_test.rs` uses for the CPU-only S5b gate) and
/// runs the identical `Qwen35VisionRequest` through both
/// `cpu_f16::generate_multimodal_f16` (the numeric oracle, ADR-069
/// Amendment 1) and `generate_multimodal_vision` (this port), on the
/// real 82-token image+text prompt. Asserts the first greedy token
/// (the highest-margin, decode-independent position — see
/// `vision_s5b_e2e_gate_test.rs`'s `check_greedy_parity` doc comment)
/// matches the committed HF golden on BOTH paths, and that the two
/// paths agree with each other on all `max_new_tokens` tokens.
#[test]
fn generate_multimodal_vision_matches_cpu_oracle_on_real_fixture() {
    let Some(model) = require_metal_and_real_checkpoint_or_skip(
        "generate_multimodal_vision_matches_cpu_oracle_on_real_fixture",
    ) else {
        return;
    };
    let _guard = gpu_test_lock();

    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("vision");
    if !fixture_dir.join("manifest.json").exists() {
        eprintln!(
            "[METAL_TEST_SKIP] context=generate_multimodal_vision_matches_cpu_oracle_on_real_fixture \
                     reason=no_vision_fixture path={}",
            fixture_dir.display()
        );
        let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but the #989 vision fixture is missing at {}",
            fixture_dir.display()
        );
        return;
    }

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture_dir.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");

    let input_ids: Vec<u32> = serde_json::from_slice::<Vec<i64>>(
        &std::fs::read(fixture_dir.join("input_ids.json")).expect("read input_ids.json"),
    )
    .expect("parse input_ids.json")
    .into_iter()
    .map(|v| v as u32)
    .collect();
    assert_eq!(
        input_ids.len(),
        82,
        "fixture prompt drifted from the reviewed anchor"
    );

    let post_merger_bytes = std::fs::read(fixture_dir.join("vit_post_merger_f32.bin"))
        .expect("read vit_post_merger_f32.bin");
    let post_merger: Vec<f32> = post_merger_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let grid_thw = manifest["image_grid_thw"]
        .as_array()
        .expect("image_grid_thw array")
        .first()
        .expect("at least one image grid")
        .as_array()
        .expect("grid triple");
    let grid = crate::vision::qwen35_vit::GridThw {
        t: grid_thw[0].as_u64().unwrap() as usize,
        h: grid_thw[1].as_u64().unwrap() as usize,
        w: grid_thw[2].as_u64().unwrap() as usize,
    };

    let greedy: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture_dir.join("greedy_tokens.json")).expect("read greedy_tokens.json"),
    )
    .expect("parse greedy_tokens.json");
    let golden_first_token = greedy["token_ids"]
        .as_array()
        .expect("greedy_tokens.token_ids")
        .first()
        .and_then(serde_json::Value::as_i64)
        .expect("first golden token") as u32;

    let cfg = model.config().clone();
    let vision_cfg = cfg
        .vision_config
        .as_ref()
        .expect("0.8b checkpoint carries vision_config");
    let image_token_id = cfg
        .image_token_id
        .expect("0.8b checkpoint carries image_token_id");

    let request = crate::vision::multimodal::Qwen35VisionRequest {
        input_ids,
        image_grids: vec![grid],
        post_merger_rows: post_merger,
        image_token_id,
        spatial_merge_size: vision_cfg.spatial_merge_size,
        decoder_hidden_size: cfg.hidden_size,
    };
    request
        .validate()
        .expect("fixture-derived request is valid");

    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        ..Default::default()
    };

    // The CPU oracle (`generate_multimodal_f16`) needs `F16ModelWeights`,
    // a different weight representation than `Qwen35Model::weights()`
    // (used by the Metal path below) — load it directly from the same
    // checkpoint directory `require_metal_and_real_checkpoint_or_skip`
    // resolved, mirroring `vision_s5b_e2e_gate_test.rs`'s loader.
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    let safetensors_path = model_dir.join("model.safetensors");
    let sf = crate::weights::f32_weights::SafetensorsFile::open(&safetensors_path)
        .expect("opening the checkpoint's safetensors file");
    let f16_weights = crate::weights::f16_weights::load_f16_weights(&sf, &cfg)
        .expect("real f16 decoder weights must load");

    let cpu_out =
        crate::forward::cpu_f16::generate_multimodal_f16(&f16_weights, &cfg, &request, &gen_cfg)
            .expect("CPU oracle multimodal generate succeeds");

    let tokenizer = minimal_bpe_tokenizer();
    let mut metal_state =
        MetalQwen35State::new(model.weights(), model.config(), 128).expect("real-checkpoint state");
    let metal_out = metal_state
        .generate_multimodal_vision(&request, &tokenizer, &gen_cfg)
        .expect("Metal multimodal generate succeeds");

    eprintln!(
        "LATTICE_MP3_PARITY cpu={:?} metal={:?} golden_first={golden_first_token}",
        cpu_out.token_ids, metal_out.token_ids
    );

    assert_eq!(
        cpu_out.token_ids.first().copied(),
        Some(golden_first_token),
        "CPU oracle's first greedy token drifted from the committed HF golden"
    );
    assert_eq!(
        metal_out.token_ids.first().copied(),
        Some(golden_first_token),
        "Metal's first greedy token drifted from the committed HF golden"
    );
    assert_eq!(
        cpu_out.token_ids, metal_out.token_ids,
        "Metal generate_multimodal_vision must match the CPU oracle \
                 (generate_multimodal_f16) on the real vision fixture"
    );
}
