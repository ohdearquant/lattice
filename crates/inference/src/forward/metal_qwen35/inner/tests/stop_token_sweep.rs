//! Cross-path stop-token contract sweep.

use super::*;
// -------------------------------------------------------------
// #613 cross-path stop-token contract sweep (Metal family).
//
// See `crate::stop_token_contract` for the full entry-point
// manifest, the chosen EXCLUDE contract, and the CPU-family half
// of this sweep (these fixtures are GPU-only, so their half lives
// here rather than in that file). Covers: plain generate,
// MTP-enabled generate, self-spec-enabled generate,
// generate_streaming, and generate_streaming_with_prefix_cache.
//
// Each test sets `stop_token_ids` to the fixture's ENTIRE vocab
// range, so the very first greedy token — whatever it turns out
// to be for these non-degenerate (real GDN+GQA layer) fixtures —
// is guaranteed to already be a configured stop. That sidesteps
// hand-deriving the exact greedy output of `tiny_hybrid_fixture`.
// -------------------------------------------------------------

/// The #613 contract: a stop token is EXCLUDED from `token_ids`/`text`.
/// `stopped` is true and `stop_reason` is `Eos` because budget was
/// available when the stop fired on the very first generated token.
fn assert_excludes_stop_token_metal(out: &GenerateOutput, entry_point: &str) {
    assert_eq!(
        out.generated_tokens, 0,
        "{entry_point}: stop-token contract violated — generated_tokens \
                 should be 0 (token excluded), got {}",
        out.generated_tokens
    );
    assert!(
        out.token_ids.is_empty(),
        "{entry_point}: stop-token contract violated — token_ids should \
                 be empty, got {:?}",
        out.token_ids
    );
    assert!(
        out.text.is_empty(),
        "{entry_point}: stop-token contract violated — text should be \
                 empty, got {:?}",
        out.text
    );
    assert!(
        out.stopped,
        "{entry_point}: stopped must be true when a stop token is hit \
                 with budget available"
    );
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Eos),
        "{entry_point}: stop_reason must be Eos"
    );
}

#[test]
fn metal_generate_plain_excludes_stop_token() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: (0..cfg.vocab_size as u32).collect(),
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let out = state
        .generate("a", &tokenizer, &gen_cfg)
        .expect("generate (plain)");
    assert_excludes_stop_token_metal(&out, "MetalQwen35State::generate (plain)");
}

#[test]
fn metal_generate_mtp_excludes_stop_token() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.mtp_num_hidden_layers = 1;
    let mut state = metal_state_with_synthetic_mtp_for_test(&weights, &cfg, None);
    assert!(
        state.session.mtp.is_some(),
        "synthetic MTP fixture must populate session.mtp so generate() \
                 actually takes the MTP branch, not silently fall back"
    );
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: (0..cfg.vocab_size as u32).collect(),
        enable_thinking: false,
        enable_mtp: Some(true),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let out = state
        .generate("a", &tokenizer, &gen_cfg)
        .expect("generate (LATTICE_MTP)");
    assert_excludes_stop_token_metal(&out, "MetalQwen35State::generate (LATTICE_MTP)");
}

#[test]
fn metal_generate_self_spec_excludes_stop_token() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: (0..32u32).collect(),
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let out = with_self_spec_env(|| {
        let (cfg, weights) = tiny_hybrid_fixture();
        let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
        assert!(
            state.session.gdn_checkpoints.is_some(),
            "LATTICE_SELF_SPEC must be observed at construction time so \
                     generate() actually takes the self-spec branch"
        );
        state.generate("a", &tokenizer, &gen_cfg)
    });
    let out = out.expect("generate (LATTICE_SELF_SPEC)");
    assert_excludes_stop_token_metal(&out, "MetalQwen35State::generate (LATTICE_SELF_SPEC)");
}

#[test]
fn metal_generate_streaming_excludes_stop_token() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: (0..cfg.vocab_size as u32).collect(),
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let mut deltas: Vec<String> = vec![];
    let out = state
        .generate_streaming("a", &tokenizer, &gen_cfg, |delta, _id| {
            deltas.push(delta.to_string());
            true
        })
        .expect("generate_streaming");
    assert_excludes_stop_token_metal(&out, "MetalQwen35State::generate_streaming");
    assert!(
        deltas.is_empty(),
        "generate_streaming must not emit an on_token callback for an \
                 excluded stop token, got {deltas:?}"
    );
}

#[test]
fn metal_generate_streaming_with_prefix_cache_excludes_stop_token() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: (0..cfg.vocab_size as u32).collect(),
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let slot = crate::kv_cache::CrossTurnSlotId::new(1);
    let result = state
        .generate_streaming_with_prefix_cache(slot, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("generate_streaming_with_prefix_cache must succeed");
    assert_excludes_stop_token_metal(
        &result.output,
        "MetalQwen35State::generate_streaming_with_prefix_cache",
    );
}

#[test]
fn metal_generate_multimodal_excludes_stop_token() {
    // #632: `generate_multimodal` is a public
    // entry point with its own sampling loop, missing from both the
    // `stop_token_contract` manifest and this test module's #613
    // sweep. It already implements the EXCLUDE contract (checks
    // `is_stop` before pushing at the first-token sample and inside
    // the autoregressive decode loop); this test locks that in.
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = minimal_bpe_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let visual_tokens = 2usize;
    let d_model = cfg.hidden_size;
    let input = crate::vision::MultimodalInput {
        patch_embeddings: vec![0.0f32; visual_tokens * d_model],
        raw_patches: 4,
        visual_tokens,
        d_model,
        text_tokens: vec![1u32],
    };

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: (0..cfg.vocab_size as u32).collect(),
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let out = state
        .generate_multimodal(input, &tokenizer, &gen_cfg)
        .expect("generate_multimodal must succeed");
    assert_excludes_stop_token_metal(&out, "MetalQwen35State::generate_multimodal");
}

#[test]
fn self_spec_checkpoint_pool_allocated_when_env_set() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    let state = with_self_spec_env(|| {
        let (cfg, weights) = tiny_hybrid_fixture();
        MetalQwen35State::new(&weights, &cfg, 32)
            .expect("tiny hybrid fixture with LATTICE_SELF_SPEC")
    });

    assert!(
        state.session.gdn_checkpoints.is_some(),
        "gdn_checkpoints must be allocated when LATTICE_SELF_SPEC is set"
    );
}

#[test]
fn generate_greedy_self_spec_output_token_count_within_budget() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    let tokenizer = minimal_bpe_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 8,
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
    };

    let out = with_self_spec_env(|| {
        let (cfg, weights) = tiny_hybrid_fixture();
        let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

        let prompt_ids = vec![0u32];
        state.reset_state();
        let prefill_logits = state.forward_prefill(&prompt_ids);

        state
            .generate_greedy_self_spec(&prefill_logits, prompt_ids.len(), &tokenizer, &gen_cfg)
            .expect("self-spec route")
    });

    assert!(
        out.generated_tokens <= gen_cfg.max_new_tokens,
        "self-spec must not exceed max_new_tokens: got {} > {}",
        out.generated_tokens,
        gen_cfg.max_new_tokens
    );
    assert_eq!(
        out.token_ids.len(),
        out.generated_tokens,
        "token_ids.len() must match generated_tokens"
    );
}

/// Shared assertion for the #1448 entry-point sweep. Every public prefill
/// route that can reach the batched schedulers must REPORT an unsupported
/// (MoE) multi-token prefill as `UnsupportedModel`, never abort the process.
///
/// Each caller does its own `catch_unwind` around its own entry point and
/// hands the outcome here, because a panic is the exact failure under guard:
/// it has to be observed as a caught unwind, not allowed to propagate. The
/// variant is asserted rather than just "an error", and "did not panic" is
/// deliberately not the assertion — a test that only checks for the absence
/// of a crash passes against a route that rejects for some unrelated reason,
/// and so would pass without the guard under test ever running.
fn assert_moe_prefill_rejected<T: std::fmt::Debug>(
    outcome: std::thread::Result<Result<T, crate::error::InferenceError>>,
    entry_point: &str,
) {
    let Ok(returned) = outcome else {
        panic!(
            "{entry_point}: an unsupported-MoE prefill must come back as Err; it \
                     unwound as a panic instead, which in a release build is the process \
                     abort this guard exists to remove"
        )
    };
    let error = match returned {
        Err(error) => error,
        Ok(value) => {
            panic!("{entry_point}: an unsupported-MoE prefill must not succeed; got {value:?}")
        }
    };
    let message = match &error {
        crate::error::InferenceError::UnsupportedModel(message) => message.clone(),
        other => panic!(
            "{entry_point}: the documented contract is \
                     InferenceError::UnsupportedModel; got {other:?}. Any other variant \
                     means a different rejection fired first and this test would pass \
                     without the guard under test having run at all"
        ),
    };
    assert!(
        message.contains("MoE"),
        "{entry_point}: the rejection must name the MoE batched-prefill \
                 limitation so a caller can act on it; got {message:?}"
    );
}

/// Regression (#1448): `forward_prefill_with_hidden` takes the batched
/// schedulers for a multi-token prompt with no active LoRA adapter, and those
/// have no MoE schedule. Until the check was wired in here it aborted on that
/// input. Three tokens, one more than the two-token minimum, so the assertion
/// is not resting on the smallest possible batch.
#[test]
fn forward_prefill_with_hidden_reports_moe_prefill_rejection() {
    with_moe_prefill_state(16, |_cfg, state| {
        let position_before = state.session.position();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.forward_prefill_with_hidden(&[1u32, 3, 5])
        }));
        assert_moe_prefill_rejected(outcome, "forward_prefill_with_hidden");
        assert_eq!(
            state.session.position(),
            position_before,
            "the rejection runs before any dispatch, so session position must \
                     be untouched"
        );
    });
}

/// Regression (#1448): `forward_prefill_all_logits` is batched by
/// construction — it has no per-token fallback at all — so an MoE checkpoint
/// reached it and aborted. Its production caller is perplexity evaluation,
/// which now gets an error it can report.
#[test]
fn forward_prefill_all_logits_reports_moe_prefill_rejection() {
    with_moe_prefill_state(16, |_cfg, state| {
        let position_before = state.session.position();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.forward_prefill_all_logits(&[1u32, 3, 5])
        }));
        assert_moe_prefill_rejected(outcome, "forward_prefill_all_logits");
        assert_eq!(
            state.session.position(),
            position_before,
            "the rejection runs before any dispatch, so session position must \
                     be untouched"
        );
    });
}

/// Regression (#1448): `generate` prefilled through the panicking
/// `forward_prefill` wrapper, so the typed rejection that already existed one
/// level down was turned straight back into an abort on the request-facing
/// path. This is the one that matters for a server: an ordinary valid prompt
/// against an MoE checkpoint took the host down.
#[test]
fn generate_reports_moe_prefill_rejection_without_panicking() {
    with_moe_prefill_state(32, |_cfg, state| {
        use crate::tokenizer::common::Tokenizer;

        let tokenizer = single_char_vocab_tokenizer();
        let prompt = "abc";
        let tokenized = tokenizer.tokenize(prompt);
        assert!(
            tokenized.real_length > 1,
            "test construction: the prompt must tokenize to more than one \
                     token (got {}), or prefill never reaches the batched path under \
                     guard",
            tokenized.real_length
        );
        let gen_cfg = moe_rejection_gen_cfg();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.generate(prompt, &tokenizer, &gen_cfg)
        }));
        assert_moe_prefill_rejected(outcome, "generate");
    });
}

/// Regression (#1448): `generate_streaming` shares
/// `generate_streaming_with_cancel`'s one implementation, which prefilled
/// through the same panicking wrapper `generate` did. Covered separately
/// rather than assumed from `generate`, because the two build their prefill
/// call independently — the sibling-invocation-path case this fix's own
/// review has to answer for.
#[test]
fn generate_streaming_reports_moe_prefill_rejection_without_panicking() {
    with_moe_prefill_state(32, |_cfg, state| {
        use crate::tokenizer::common::Tokenizer;

        let tokenizer = single_char_vocab_tokenizer();
        let prompt = "abc";
        let tokenized = tokenizer.tokenize(prompt);
        assert!(
            tokenized.real_length > 1,
            "test construction: the prompt must tokenize to more than one \
                     token (got {}), or prefill never reaches the batched path under \
                     guard",
            tokenized.real_length
        );
        let gen_cfg = moe_rejection_gen_cfg();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.generate_streaming(prompt, &tokenizer, &gen_cfg, |_, _| true)
        }));
        assert_moe_prefill_rejected(outcome, "generate_streaming");
    });
}

/// Regression (#1448 follow-up): the prefix-cache generation path reaches the
/// batched schedulers through `forward_prefill_from`, not through
/// `try_forward_prefill`, so the first fix left it able to abort — the
/// exact sibling-invocation-path miss the original test comment warned about,
/// found by the review rather than by the grep. The serve worker calls this
/// entry point directly, so this abort was remotely reachable.
#[test]
fn generate_streaming_with_prefix_cache_reports_moe_prefill_rejection() {
    with_moe_prefill_state(32, |_cfg, state| {
        use crate::tokenizer::common::Tokenizer;

        let tokenizer = single_char_vocab_tokenizer();
        let prompt = "abc";
        let tokenized = tokenizer.tokenize(prompt);
        assert!(
            tokenized.real_length > 1,
            "test construction: the prompt must tokenize to more than one \
                     token (got {}), or the suffix prefill never spans multiple \
                     tokens",
            tokenized.real_length
        );
        let gen_cfg = moe_rejection_gen_cfg();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state
                .generate_streaming_with_prefix_cache(
                    crate::kv_cache::CrossTurnSlotId::DEFAULT,
                    prompt,
                    &tokenizer,
                    &gen_cfg,
                    |_, _| true,
                )
                .map(|cached| cached.output)
        }));
        assert_moe_prefill_rejected(outcome, "generate_streaming_with_prefix_cache");
    });
}

/// Rule-separating pair for `forward_prefill_from` on an MoE model (#1448
/// follow-up): a single-token suffix takes the per-token `try_forward_step`
/// fast path, which supports MoE, and must SUCCEED with full-vocabulary
/// logits; a multi-token suffix has only batched branches left and must be
/// rejected as `UnsupportedModel`. Asserting both on one fixture is what
/// pins the rejection to the batched routing rather than to the model kind —
/// a rejection keyed on "is MoE" alone would fail the first arm.
#[test]
fn forward_prefill_from_separates_single_token_success_from_batched_rejection() {
    with_moe_prefill_state(16, |cfg, state| {
        let logits = state
            .forward_prefill_from(&[2u32], 0, false)
            .expect("a single-token suffix takes the per-token path, which supports MoE");
        assert_eq!(
            logits.len(),
            cfg.vocab_size,
            "the single-token fast path must return full-vocabulary logits"
        );

        state.reset_state();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.forward_prefill_from(&[1u32, 3, 5], 0, false)
        }));
        assert_moe_prefill_rejected(outcome, "forward_prefill_from");
    });
}

/// Regression (#1448 follow-up): a prefill that fails after the decoder
/// session engaged the compact route must not leave that
/// route on the session — a raw `forward_step` afterward would take the
/// compact branch and return empty logits. Engages the route by direct field
/// assignment (the #171 pattern: route *selection* is env-keyed and
/// process-global, so an end-to-end engagement through `generate` is not
/// testable under parallel test execution), calls the cleanup the error arms
/// call, and asserts the outcome the contract names: `forward_step` returns
/// full-vocabulary logits. The wiring of the cleanup into all three
/// generation entry points' error arms is pinned by the source-consistency
/// rows in `public_scheduling_entry_points_route_through_their_preflights`.
#[test]
fn failed_prefill_compact_cleanup_restores_full_vocab_forward_step() {
    with_moe_prefill_state(16, |cfg, state| {
        state.session.compact_route = GpuTopkRoute::BlockArgmax;
        state.session.compact_topk = 1;

        state.disengage_compact_route();

        assert_eq!(
            state.session.compact_route,
            GpuTopkRoute::CpuFallback,
            "the cleanup must disengage the compact route"
        );
        assert_eq!(
            state.session.compact_topk, 0,
            "the cleanup must clear compact_topk"
        );
        let logits = state.forward_step(3, 0);
        assert_eq!(
            logits.len(),
            cfg.vocab_size,
            "after cleanup, a raw forward_step must return full [vocab_size] \
                     logits via the exact path — a shorter/empty result means the \
                     compact route survived the failed prefill"
        );
    });
}

/// Regression (#1448): `embed_tokens` is the third caller of the panicking
/// wrapper and is not named in the issue — found by grepping the wrapper's
/// call sites rather than by working the reported list. It is request-facing
/// (it backs Metal embedding), so it had the same abort. It also sets
/// `capture_final_hidden` around the prefill, so the rejection has to restore
/// that flag on the way out; the second assertion is what makes the early
/// return distinguishable from one that leaks session state.
#[test]
fn embed_tokens_reports_moe_prefill_rejection_without_panicking() {
    with_moe_prefill_state(16, |_cfg, state| {
        let capture_before = state.session.capture_final_hidden;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.embed_tokens(
                &[1u32, 3, 5],
                crate::model::qwen35::HiddenPooling::LastToken,
            )
        }));
        assert_moe_prefill_rejected(outcome, "embed_tokens");
        assert_eq!(
            state.session.capture_final_hidden, capture_before,
            "a rejected embed must restore capture_final_hidden; leaving it \
                     set would make the next unrelated forward pass capture a hidden \
                     row it was never asked for"
        );
    });
}

/// Minimal decode config for the #1448 rejection tests. Every field is set
/// so that nothing but the MoE prefill can reject: no grammar, no logprobs,
/// no MTP, no stop tokens, and a decode budget the tiny fixture's context
/// comfortably admits. If one of those guards fired first,
/// `assert_moe_prefill_rejected` would see the wrong error variant and fail
/// rather than pass on the wrong evidence.
fn moe_rejection_gen_cfg() -> crate::generation::GenerateConfig {
    crate::generation::GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
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

/// A vocab with two multibyte (CJK) single-token entries plus the same
/// single-char ASCII fallback as [`single_char_vocab_tokenizer`], so a
/// stop string spanning a token boundary can be built purely out of
/// multibyte UTF-8 codepoints (each 3 bytes) — exercising the UTF-8
/// char-boundary clamp that `single_char_vocab_tokenizer` (all 1-byte
/// tokens) cannot exercise on its own.
///
/// PR #657: the previous version of this
/// fixture inserted the literal chars `"世"`/`"界"` as BPE vocab
/// *strings*. Qwen's real decode path reverses GPT-2 byte-level token
/// strings through `byte_decoder` (`detokenize.rs::append_token_bytes`);
/// characters outside that decoder's placeholder set (any non-ASCII,
/// non-Latin-1-printable char — CJK included) are silently dropped, so
/// those literal entries never decoded to CJK bytes at all and any test
/// sampling ids 30/31 exercised nothing. Entries here instead use
/// `bytes_to_unicode()`'s placeholder chars over the raw UTF-8 bytes of
/// `好` (`0xE5 0xA5 0xBD`), exactly as a real byte-level BPE vocab
/// would encode it, so ids 30/31 decode to genuine CJK bytes if sampled.
fn multibyte_vocab_tokenizer() -> crate::tokenizer::bpe::BpeTokenizer {
    use crate::tokenizer::detokenize::bytes_to_unicode;
    use std::collections::HashMap;
    let byte_encoder = bytes_to_unicode();
    let byte_level_token =
        |bytes: &[u8]| -> String { bytes.iter().map(|&b| byte_encoder[b as usize]).collect() };
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0u32..30 {
        let byte = if i < 26 {
            b'a' + i as u8
        } else {
            b'A' + (i - 26) as u8
        };
        vocab.insert((byte as char).to_string(), i);
    }
    // 好 = 0xE5 0xA5 0xBD, split across two token ids so a stop string
    // built from the completed codepoint spans a real token boundary.
    vocab.insert(byte_level_token(&[0xE5, 0xA5]), 30);
    vocab.insert(byte_level_token(&[0xBD]), 31);
    crate::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("multibyte vocab tokenizer build")
}

/// #643: `MetalQwen35State::generate` (plain path) must honor
/// `GenerateConfig::stop_strings` with CPU-identical semantics — halt as
/// soon as a stop string completes and exclude the stop text from the
/// returned text.
///
/// This drives the same seeded greedy config twice on fresh state: once
/// with no stop strings to observe the natural output, once with a stop
/// string built from a later character in that output. If the fix in
/// `generate` is reverted, `gen_cfg.stop_strings` is never consulted and
/// this assertion fails because the stopped run produces the full,
/// untruncated text instead.
#[test]
fn metal_generate_plain_honors_stop_string_mid_generation() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let base_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 6,
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
    };

    let (cfg, weights) = tiny_hybrid_fixture();
    let baseline = {
        let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
        state
            .generate("a", &tokenizer, &base_cfg)
            .expect("baseline generate (no stop_strings)")
    };
    assert!(
        baseline.text.len() >= 3,
        "need at least 3 generated chars to test a mid-generation stop, got {:?}",
        baseline.text
    );

    // Stop on the character generated last. The expected truncation point
    // is the *earliest* occurrence of that character in the baseline text
    // (CPU-identical "earliest stop match" semantics) — for a baseline
    // with a repeated character that is index 0, giving an empty prefix;
    // either way this must match `earliest_stop_match`'s own answer.
    let stop_char = baseline.text.chars().last().expect("non-empty baseline");
    let hit = baseline
        .text
        .find(stop_char)
        .expect("stop_char was taken from baseline.text");
    let expected_prefix = baseline.text[..hit].to_string();

    let mut stopped_cfg = base_cfg.clone();
    stopped_cfg.stop_strings = vec![stop_char.to_string()];

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let out = state
        .generate("a", &tokenizer, &stopped_cfg)
        .expect("generate with stop_strings");

    assert_eq!(
        out.text, expected_prefix,
        "stop string must truncate text to the prefix before the earliest match \
                 (stop text excluded from returned output)"
    );
    assert!(out.stopped, "stop-string match must set stopped = true");
    assert_eq!(out.stop_reason, Some(StopReason::Eos));
    assert!(
        !out.text.contains(stop_char),
        "returned text must not contain the stop string"
    );
}

/// #643: `MetalQwen35State::generate_streaming_with_cancel` must honor
/// `GenerateConfig::stop_strings`: streamed chunks must concatenate to
/// exactly the truncated `GenerateOutput::text`, and the stop text itself
/// must never reach `on_token`.
#[test]
fn metal_generate_streaming_honors_stop_string_and_excludes_stop_text() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let base_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 6,
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
    };

    let (cfg, weights) = tiny_hybrid_fixture();
    let baseline = {
        let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
        state
            .generate("a", &tokenizer, &base_cfg)
            .expect("baseline generate (no stop_strings)")
    };
    assert!(
        baseline.text.len() >= 3,
        "need at least 3 generated chars to test a mid-generation stop, got {:?}",
        baseline.text
    );

    let stop_char = baseline.text.chars().last().expect("non-empty baseline");
    let hit = baseline
        .text
        .find(stop_char)
        .expect("stop_char was taken from baseline.text");
    let expected_prefix = baseline.text[..hit].to_string();

    let mut stopped_cfg = base_cfg.clone();
    stopped_cfg.stop_strings = vec![stop_char.to_string()];

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let mut streamed = String::new();
    let out = state
        .generate_streaming_with_cancel(
            "a",
            &tokenizer,
            &stopped_cfg,
            |delta, _id| {
                streamed.push_str(delta);
                true
            },
            || false,
        )
        .expect("generate_streaming_with_cancel with stop_strings");

    assert_eq!(out.text, expected_prefix);
    assert_eq!(
        streamed, expected_prefix,
        "streamed chunks must concatenate to exactly the truncated text"
    );
    assert!(!streamed.contains(stop_char));
    assert!(out.stopped);
    assert_eq!(out.stop_reason, Some(StopReason::Eos));
}

/// #643: a stop string built entirely out of multibyte (CJK, 3-byte)
/// codepoints must never split a codepoint across the emitted/held-back
/// boundary, and must still be excluded from the returned text once it
/// matches — exercising the UTF-8 char-boundary clamp in
/// `StopStringMatcher` from behind the Metal plain-generate path.
#[test]
fn metal_generate_plain_honors_multibyte_stop_string() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = multibyte_vocab_tokenizer();
    let base_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 6,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(7),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let (cfg, weights) = tiny_hybrid_fixture();
    let baseline = {
        let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
        state
            .generate("a", &tokenizer, &base_cfg)
            .expect("baseline generate (no stop_strings)")
    };
    assert!(
        baseline.text.chars().count() >= 2,
        "need at least 2 generated chars, got {:?}",
        baseline.text
    );

    // Stop on the last generated codepoint (which may itself be a 3-byte
    // CJK character if the tiny model happens to sample token 30/31, or
    // an ASCII fallback char otherwise) — either way this exercises a
    // matcher configured with `max_stop` sized for a multibyte codepoint.
    let stop_char = baseline.text.chars().last().expect("non-empty baseline");
    let mut stop_string = String::new();
    stop_string.push(stop_char);
    let hit = baseline
        .text
        .find(stop_string.as_str())
        .expect("stop_char was taken from baseline.text");
    let expected_prefix = baseline.text[..hit].to_string();

    let mut stopped_cfg = base_cfg.clone();
    stopped_cfg.stop_strings = vec![stop_string.clone()];

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let out = state
        .generate("a", &tokenizer, &stopped_cfg)
        .expect("generate with multibyte stop_strings");

    // Never split a codepoint: the returned text must itself be valid
    // UTF-8 (guaranteed by the `String` type) and must equal the exact
    // pre-stop prefix, byte for byte.
    assert_eq!(out.text, expected_prefix);
    assert!(!out.text.contains(stop_string.as_str()));
    assert!(out.stopped);
    assert_eq!(out.stop_reason, Some(StopReason::Eos));
}

/// #643: a stop string that spans a token boundary (built from two
/// consecutive generated characters, each produced by a *separate*
/// token push in `single_char_vocab_tokenizer`) must still be caught by
/// `generate_streaming_with_cancel` — the matcher must hold back the
/// ambiguous partial suffix across pushes rather than only checking
/// each token's delta in isolation. If the held-back-suffix logic were
/// dropped (checking only the newest delta against the stop strings),
/// this two-character stop could never match and the run would go to
/// `max_new_tokens` instead.
#[test]
fn metal_generate_streaming_honors_stop_string_spanning_token_boundary() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let base_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 8,
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
    };

    let (cfg, weights) = tiny_hybrid_fixture();
    let baseline = {
        let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
        state
            .generate("a", &tokenizer, &base_cfg)
            .expect("baseline generate (no stop_strings)")
    };
    assert!(
        baseline.text.chars().count() >= 4,
        "need at least 4 generated chars to build a 2-char boundary-spanning stop, got {:?}",
        baseline.text
    );

    // Every token in `single_char_vocab_tokenizer` decodes to exactly one
    // char, so any 2-char stop string necessarily spans the boundary
    // between two separate token pushes — there is no way to satisfy it
    // from a single token's delta alone.
    let chars: Vec<char> = baseline.text.chars().collect();
    let stop_string: String = chars[chars.len() - 2..].iter().collect();
    let hit = baseline
        .text
        .find(stop_string.as_str())
        .expect("stop_string was taken from baseline.text");
    let expected_prefix = baseline.text[..hit].to_string();

    let mut stopped_cfg = base_cfg.clone();
    stopped_cfg.stop_strings = vec![stop_string.clone()];

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let mut streamed = String::new();
    let out = state
        .generate_streaming_with_cancel(
            "a",
            &tokenizer,
            &stopped_cfg,
            |delta, _id| {
                streamed.push_str(delta);
                true
            },
            || false,
        )
        .expect("generate_streaming_with_cancel with boundary-spanning stop_strings");

    assert_eq!(
        out.text, expected_prefix,
        "boundary-spanning stop string must truncate at its earliest match"
    );
    assert_eq!(
        streamed, expected_prefix,
        "streamed chunks must concatenate to exactly the truncated text"
    );
    assert!(!streamed.contains(stop_string.as_str()));
    assert!(
        out.stopped,
        "boundary-spanning stop-string match must set stopped = true"
    );
    assert_eq!(out.stop_reason, Some(StopReason::Eos));
}

/// #643: when `stop_strings` is non-empty but never matches, generation
/// must run to `max_new_tokens` exactly as it would with no stop strings
/// at all — the matcher must not truncate, alter, or otherwise affect
/// output when it never sees a match. This guards against a mutation
/// that always treats a non-empty `stop_strings` config as "stop
/// immediately" or that corrupts the accumulated text when finishing
/// without a match.
#[test]
fn metal_generate_plain_runs_to_max_tokens_when_stop_string_absent() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let base_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 6,
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
    };

    let (cfg, weights) = tiny_hybrid_fixture();
    let baseline = {
        let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
        state
            .generate("a", &tokenizer, &base_cfg)
            .expect("baseline generate (no stop_strings)")
    };
    assert_eq!(
        baseline.stop_reason,
        Some(StopReason::Length),
        "baseline with no stop condition must exhaust max_new_tokens"
    );

    // "0" can never be produced by `single_char_vocab_tokenizer` (its
    // vocab is only a-z/A-Z), so this stop string can never match.
    let mut stopped_cfg = base_cfg.clone();
    stopped_cfg.stop_strings = vec!["0".to_string()];

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let out = state
        .generate("a", &tokenizer, &stopped_cfg)
        .expect("generate with unmatchable stop_strings");

    assert_eq!(
        out.text, baseline.text,
        "an absent stop string must not alter generated text"
    );
    assert!(
        !out.stopped,
        "an absent stop string must not report an early stop"
    );
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Length),
        "an absent stop string must still exhaust max_new_tokens with Length"
    );
}

/// #643: `generate_streaming_with_prefix_cache_inner` must honor
/// `GenerateConfig::stop_strings` with the same truncation/exclusion
/// semantics as the other Metal loops, AND must drop (not save) the
/// cross-turn prefix cache entry for the slot on a stop-string match —
/// CPU semantics can retain a generated token whose text was truncated
/// by the match, so caching the hidden state behind that token would
/// let a later turn resume from state representing text the caller
/// never received.
///
/// Turn 1 (no `stop_strings`) first proves the slot's cache entry is
/// normally populated by a successful turn — establishing the
/// contrast the turn-2 assertion depends on (a cache that starts
/// empty would make an `is_none()` check trivially true regardless of
/// whether the stop-string removal code ran). Turn 2 uses a
/// different prompt (`"b"` vs turn 1's `"a"`) so the reuse planner
/// picks `PrefixReuseMode::FullRefill` — the one mode where
/// `restore_cross_turn_prefix` does NOT `take()` (and thus does not
/// already empty) the slot's map entry, so turn 1's stale entry is
/// still sitting in `cross_turn_prefix_cache` when turn 2's stop
/// check runs. Turn 2's `stop_strings: [""]` — because
/// `str::find("")` always matches at byte 0 — deterministically
/// triggers `generate_streaming_with_prefix_cache_inner`'s
/// first-token stop-check branch regardless of the tiny fixture's
/// sampled output. This isolates the exact removal call the design
/// added: without it, turn 1's stale entry would survive turn 2's
/// string-stop return and misrepresent turn 2's (truncated, FullRefill)
/// state to a later caller.
#[test]
fn metal_generate_streaming_with_prefix_cache_honors_stop_string_and_clears_cache_on_match() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;
    use crate::kv_cache::PrefixReuseMode;

    let tokenizer = single_char_vocab_tokenizer();
    let base_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 6,
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
    };

    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot = crate::kv_cache::CrossTurnSlotId::new(1);

    // Turn 1: no stop_strings, must complete normally and populate the
    // slot's cross-turn cache entry.
    let turn1 = state
        .generate_streaming_with_prefix_cache(slot, "a", &tokenizer, &base_cfg, |_, _| true)
        .expect("turn 1 generate_streaming_with_prefix_cache (no stop_strings)");
    assert!(
        !turn1.output.stopped,
        "turn 1 baseline must run to max_new_tokens, not stop early"
    );
    assert!(
        state.cross_turn_prefix_cache.get(slot).is_some(),
        "a normal (non-string-stopped) turn must populate the slot's cross-turn cache entry \
                 — this is the baseline the mutation check below contrasts against"
    );

    // Turn 2: same state/slot, a DIFFERENT prompt so the planner picks
    // FullRefill (leaving turn 1's stale entry untouched by the
    // restore step), and `stop_strings: [""]` matches at byte 0
    // unconditionally, so this must stop on the very first generated
    // token and clear the (now stale) cache entry left by turn 1.
    let mut stopped_cfg = base_cfg.clone();
    stopped_cfg.stop_strings = vec![String::new()];
    let mut streamed = String::new();
    let turn2 = state
        .generate_streaming_with_prefix_cache(slot, "b", &tokenizer, &stopped_cfg, |d, _id| {
            streamed.push_str(d);
            true
        })
        .expect("turn 2 generate_streaming_with_prefix_cache with stop_strings: [\"\"]");

    assert_eq!(
        turn2.cache.mode,
        PrefixReuseMode::FullRefill,
        "turn 2's mismatched prompt must force FullRefill, so the stale entry from turn 1 \
                 is NOT already emptied by restore_cross_turn_prefix's take() before the stop check"
    );
    assert_eq!(
        turn2.output.text, "",
        "an empty-string stop must match at byte 0, excluding all generated text"
    );
    assert_eq!(
        streamed, "",
        "no text may be streamed before the byte-0 match"
    );
    assert!(turn2.output.stopped);
    assert_eq!(turn2.output.stop_reason, Some(StopReason::Eos));
    assert!(
        state.cross_turn_prefix_cache.get(slot).is_none(),
        "a stop-string match must drop the (now stale) cross-turn cache entry for the \
                 slot, never leave turn 1's entry surviving a string-stopped turn 2"
    );
}

/// #643: `generate_multimodal` must honor `GenerateConfig::stop_strings`
/// with the same semantics as the text-only Metal loops.
#[test]
fn metal_generate_multimodal_honors_stop_string() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = minimal_bpe_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();

    let visual_tokens = 2usize;
    let d_model = cfg.hidden_size;
    let make_input = || crate::vision::MultimodalInput {
        patch_embeddings: vec![0.0f32; visual_tokens * d_model],
        raw_patches: 4,
        visual_tokens,
        d_model,
        text_tokens: vec![1u32],
    };

    let base_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 6,
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
    };

    let baseline = {
        let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
        state
            .generate_multimodal(make_input(), &tokenizer, &base_cfg)
            .expect("baseline generate_multimodal (no stop_strings)")
    };
    assert!(
        baseline.text.len() >= 3,
        "need at least 3 generated chars to test a mid-generation stop, got {:?}",
        baseline.text
    );

    let stop_char = baseline.text.chars().last().expect("non-empty baseline");
    let hit = baseline
        .text
        .find(stop_char)
        .expect("stop_char was taken from baseline.text");
    let expected_prefix = baseline.text[..hit].to_string();

    let mut stopped_cfg = base_cfg.clone();
    stopped_cfg.stop_strings = vec![stop_char.to_string()];

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let out = state
        .generate_multimodal(make_input(), &tokenizer, &stopped_cfg)
        .expect("generate_multimodal with stop_strings");

    assert_eq!(
        out.text, expected_prefix,
        "stop string must truncate text to the prefix before the earliest match \
                 (stop text excluded from returned output)"
    );
    assert!(out.stopped, "stop-string match must set stopped = true");
    assert_eq!(out.stop_reason, Some(StopReason::Eos));
    assert!(
        !out.text.contains(stop_char),
        "returned text must not contain the stop string"
    );
}

#[test]
fn metal_generation_entrypoints_enforce_context_admission() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let gen_cfg = cross_turn_test_gen_cfg(1, 1);
    let exact_prompt = "a".repeat(31);
    let over_budget_prompt = "a".repeat(32);
    let over_window_prompt = "a".repeat(33);

    let mut direct = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    direct
        .generate(&exact_prompt, &tokenizer, &gen_cfg)
        .expect("31 prompt tokens plus one decode token must fit a 32-token window");
    let direct_seq_len = direct.session.kv_cache.seq_len;
    let direct_over = direct.generate(&over_budget_prompt, &tokenizer, &gen_cfg);
    assert_context_budget_error(&direct_over, 32, 1, 32);
    assert_eq!(
        direct.session.kv_cache.seq_len, direct_seq_len,
        "direct admission rejection must precede reset/prefill/decode"
    );
    let direct_prompt_over = direct.generate(&over_window_prompt, &tokenizer, &gen_cfg);
    assert_context_budget_error(&direct_prompt_over, 33, 1, 32);

    let mut streaming = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    streaming
        .generate_streaming(&exact_prompt, &tokenizer, &gen_cfg, |_, _| true)
        .expect("streaming exact-fit request must succeed");
    let streaming_seq_len = streaming.session.kv_cache.seq_len;
    let mut streaming_callbacks = 0usize;
    let streaming_over =
        streaming.generate_streaming(&over_budget_prompt, &tokenizer, &gen_cfg, |_, _| {
            streaming_callbacks += 1;
            true
        });
    assert_context_budget_error(&streaming_over, 32, 1, 32);
    assert_eq!(streaming_callbacks, 0, "rejection must precede callbacks");
    assert_eq!(
        streaming.session.kv_cache.seq_len, streaming_seq_len,
        "streaming admission rejection must precede reset/prefill/decode"
    );
    let streaming_prompt_over =
        streaming.generate_streaming(&over_window_prompt, &tokenizer, &gen_cfg, |_, _| true);
    assert_context_budget_error(&streaming_prompt_over, 33, 1, 32);

    let mut cached = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    cached
        .generate_streaming_with_prefix_cache(
            slot_id,
            &exact_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("prefix-cache exact-fit request must succeed");
    let cached_tokens = cached
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .expect("exact-fit request must retain a prefix-cache entry")
        .generic
        .token_ids
        .clone();
    let mut cached_callbacks = 0usize;
    let cached_over = cached.generate_streaming_with_prefix_cache(
        slot_id,
        &over_budget_prompt,
        &tokenizer,
        &gen_cfg,
        |_, _| {
            cached_callbacks += 1;
            true
        },
    );
    assert_context_budget_error(&cached_over, 32, 1, 32);
    assert_eq!(cached_callbacks, 0, "rejection must precede callbacks");
    assert_eq!(
        cached
            .cross_turn_prefix_cache
            .entry
            .as_ref()
            .map(|entry| entry.generic.token_ids.clone()),
        Some(cached_tokens.clone()),
        "prefix-cache admission rejection must not evict the warm entry"
    );
    let cached_prompt_over = cached.generate_streaming_with_prefix_cache(
        slot_id,
        &over_window_prompt,
        &tokenizer,
        &gen_cfg,
        |_, _| true,
    );
    assert_context_budget_error(&cached_prompt_over, 33, 1, 32);
    assert_eq!(
        cached
            .cross_turn_prefix_cache
            .entry
            .as_ref()
            .map(|entry| entry.generic.token_ids.clone()),
        Some(cached_tokens),
        "prompt-over-window rejection must not evict the warm entry"
    );

    let make_multimodal = |total_len: usize| crate::vision::MultimodalInput {
        patch_embeddings: vec![0.0; cfg.hidden_size],
        raw_patches: 1,
        visual_tokens: 1,
        d_model: cfg.hidden_size,
        text_tokens: vec![0; total_len - 1],
    };
    let mut multimodal = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    multimodal
        .generate_multimodal(make_multimodal(31), &tokenizer, &gen_cfg)
        .expect("multimodal exact-fit request must succeed");
    let multimodal_seq_len = multimodal.session.kv_cache.seq_len;
    let multimodal_over = multimodal.generate_multimodal(make_multimodal(32), &tokenizer, &gen_cfg);
    assert_context_budget_error(&multimodal_over, 32, 1, 32);
    assert_eq!(
        multimodal.session.kv_cache.seq_len, multimodal_seq_len,
        "multimodal admission rejection must precede reset/prefill/decode"
    );
    let multimodal_prompt_over =
        multimodal.generate_multimodal(make_multimodal(33), &tokenizer, &gen_cfg);
    assert_context_budget_error(&multimodal_prompt_over, 33, 1, 32);
}

#[test]
fn metal_speculative_generation_owns_prefill_and_context_admission() {
    use crate::speculative::MtpTargetVerifier as _;

    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state =
        MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture must construct");

    state.reset_state();
    let _ = state.forward_prefill(&[1]);
    assert_eq!(state.session.kv_cache.seq_len, 1);

    let mut warm_gdn = state.snapshot_gdn_states();
    for (layer, (s_matrix, conv)) in warm_gdn.iter_mut().enumerate() {
        for (index, value) in s_matrix.iter_mut().enumerate() {
            *value = 0.5 + layer as f32 * 0.01 + index as f32 * 1e-4;
        }
        for (index, value) in conv.iter_mut().enumerate() {
            *value = -0.7 - layer as f32 * 0.01 - index as f32 * 1e-4;
        }
    }
    state.restore_gdn_states(&warm_gdn);
    assert!(
        warm_gdn
            .iter()
            .flat_map(|(s, conv)| s.iter().chain(conv))
            .any(|&value| value != 0.0),
        "the warmed state must be observably non-zero before admission"
    );

    let over_budget_prompt = vec![0u32; 32];
    let rejected = state.generate_with_speculation(&over_budget_prompt, 1, u32::MAX, 5, 4);
    assert_context_budget_error(&rejected, 32, 1, 32);
    assert_eq!(
        state.session.kv_cache.seq_len, 1,
        "context rejection must precede reset, prefill, and decode"
    );
    assert_eq!(
        state.snapshot_gdn_states(),
        warm_gdn,
        "context rejection must preserve recurrent state before reset"
    );

    let mut exact_prompt = vec![0u32; 31];
    exact_prompt[30] = 1;
    let generated = state
        .generate_with_speculation(&exact_prompt, 1, u32::MAX, 5, 4)
        .expect("31 prompt tokens plus one generated token exactly fit");
    assert_eq!(
        generated,
        vec![1],
        "the first token must come from the completed prompt prefill"
    );
    assert_eq!(
        state.session.kv_cache.seq_len,
        exact_prompt.len() + generated.len(),
        "the final budget token must be forwarded into live state"
    );
    assert!(
        state
            .snapshot_gdn_states()
            .iter()
            .flat_map(|(s, conv)| s.iter().chain(conv))
            .all(|&value| value == 0.0),
        "state-owned generation must reset the planted recurrent state before prefill"
    );

    let invalid_prompt =
        state.generate_with_speculation(&[cfg.vocab_size as u32], 1, u32::MAX, 5, 4);
    assert!(matches!(
        invalid_prompt,
        Err(crate::error::InferenceError::InvalidInput(ref message))
            if message
                == "prompt contains out-of-vocabulary token id 32 (vocab_size=32)"
    ));
    assert_eq!(
        state.session.kv_cache.seq_len, 32,
        "token-id rejection must precede reset and GPU work"
    );

    let zero_budget = state
        .generate_with_speculation(&[0], 0, u32::MAX, 5, 4)
        .expect("zero budget must be a successful no-op");
    assert!(zero_budget.is_empty());
    assert_eq!(
        state.session.kv_cache.seq_len, 32,
        "zero budget must not reset an existing session"
    );
}

#[test]
fn lora_mixture_context_rejection_preserves_adapter_and_cache() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let mut loaded_layer = make_valid_layer(cfg.hidden_size, 2);
    loaded_layer.layer_idx = 3;
    state
        .load_lora_adapter(vec![loaded_layer], 1.0, None)
        .expect("precondition adapter must load");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let warm_cfg = cross_turn_test_gen_cfg(7, 1);
    state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &warm_cfg, |_, _| true)
        .expect("precondition cache warm-up must succeed");
    let cached_tokens = state
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .expect("precondition cache entry must exist")
        .generic
        .token_ids
        .clone();
    let seq_len = state.session.kv_cache.seq_len;

    let mut mixture_layer = make_valid_layer(cfg.hidden_size, 2);
    mixture_layer.layer_idx = 3;
    let mixture_layers = vec![mixture_layer];
    let rejected_cfg = cross_turn_test_gen_cfg(9, 1);
    let over_budget_prompt = "a".repeat(32);
    let result = state.generate_with_lora_mixture(
        &[(&mixture_layers, 1.0)],
        &over_budget_prompt,
        &tokenizer,
        &rejected_cfg,
    );

    assert_context_budget_error(&result, 32, 1, 32);
    assert!(
        state.has_lora_adapter(),
        "rejection must precede unloading the pre-existing adapter"
    );
    assert_eq!(
        state
            .cross_turn_prefix_cache
            .entry
            .as_ref()
            .map(|entry| entry.generic.token_ids.clone()),
        Some(cached_tokens),
        "rejection must precede adapter-driven prefix-cache invalidation"
    );
    assert_eq!(
        state.session.kv_cache.seq_len, seq_len,
        "rejection must precede generation-state reset or GPU decode"
    );
}

/// `generate_streaming` must report `StopReason::Interrupt` (not the default
/// `Length`) when the caller's callback returns `false` on the first
/// non-empty delta, and must not flush a trailing detokenizer tail after the
/// caller has stopped consuming the stream.
#[test]
fn stop_reason_interrupt_on_stream_callback_false() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 8,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let (mut cfg, weights) = tiny_hybrid_fixture();
    // Push eos_token_id out of the model's reachable output range (vocab_size
    // 32) so the prefill-sampled token can never be mistaken for an
    // immediate-EOS exit; this test only cares about the callback-interrupt
    // path, not which token greedy sampling happens to pick.
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let mut calls = 0u32;
    let out = state
        .generate_streaming("a", &tokenizer, &gen_cfg, |delta, _id| {
            calls += 1;
            assert!(
                !delta.is_empty(),
                "on_token must only be invoked with a non-empty delta"
            );
            false
        })
        .expect("no grammar configured; must not fail closed");

    assert_eq!(
        calls, 1,
        "callback must fire exactly once (the interrupting call); a second \
                 call would mean a trailing tail flush ran after interruption"
    );
    assert!(
        !out.stopped,
        "caller-interrupted generation is not an OpenAI stop condition"
    );
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Interrupt),
        "callback returning false must report Interrupt, got {:?}",
        out.stop_reason
    );
}

/// PR #606: `on_token`-only cancellation cannot
/// observe a disconnect that happens before or during prefill, since
/// prefill has no callback point at all. `generate_streaming_with_cancel`
/// must check `should_cancel` before paying for prefill, so a client
/// that is already gone never even starts it and `on_token` is never
/// called.
#[test]
fn stop_reason_interrupt_when_cancelled_before_prefill_starts() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 8,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        logprobs: None,
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
    };

    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let mut on_token_calls = 0u32;
    let out = state
        .generate_streaming_with_cancel(
            "a",
            &tokenizer,
            &gen_cfg,
            |_delta, _id| {
                on_token_calls += 1;
                true
            },
            || true, // already cancelled before generation ever starts
        )
        .expect("no grammar configured; must not fail closed");

    assert_eq!(
        on_token_calls, 0,
        "should_cancel already true before prefill must return before \
                 on_token is ever called, got {on_token_calls} calls"
    );
    // `on_token_calls == 0` alone doesn't prove *which* checkpoint fired --
    // the after-prefill check would produce the same on_token count if the
    // before-prefill one were missing, since prefill's own output never
    // reaches on_token either way. Prefill is the
    // only thing that advances kv_cache.seq_len (metal_qwen35_chunked_prefill*
    // tests assert seq_len == tokens.len() once it has run), so a still-zero
    // seq_len is direct evidence prefill itself never started.
    assert_eq!(
        state.session.kv_cache.seq_len, 0,
        "should_cancel already true before prefill must return before prefill \
                 itself runs, but kv_cache.seq_len={} shows it advanced",
        state.session.kv_cache.seq_len
    );
    assert_eq!(
        out.generated_tokens, 0,
        "cancelled-before-prefill must produce zero tokens"
    );
    assert!(!out.stopped, "cancellation is not an OpenAI stop condition");
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Interrupt),
        "cancelled-before-prefill must report Interrupt, got {:?}",
        out.stop_reason
    );
}

/// PR #606: the decode loop only called
/// `on_token` for non-empty deltas, so a client disconnect could be
/// missed for however many iterations produced an empty delta (an
/// incomplete UTF-8 tail mid-codepoint). This proves `should_cancel`
/// stops generation at the top of a decode iteration on its own, even
/// when `on_token` itself always answers "keep going" -- the decode
/// loop's own independent check, not the callback, is what halts it.
#[test]
fn stop_reason_interrupt_when_should_cancel_alone_stops_mid_decode() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 8,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        logprobs: None,
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
    };

    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let mut on_token_calls = 0u32;
    let mut should_cancel_calls = 0u32;
    let out = state
        .generate_streaming_with_cancel(
            "a",
            &tokenizer,
            &gen_cfg,
            |_delta, _id| {
                on_token_calls += 1;
                true // on_token itself never asks to stop.
            },
            move || {
                should_cancel_calls += 1;
                // False for the before-prefill and after-prefill checks
                // (calls 1-2), letting exactly the prefill-sampled token
                // through; true starting at the decode loop's very first
                // check (call 3), before that iteration samples anything.
                should_cancel_calls > 2
            },
        )
        .expect("no grammar configured; must not fail closed");

    assert_eq!(
        on_token_calls, 1,
        "exactly the prefill-sampled token should reach on_token before \
                 should_cancel stops the decode loop on its first iteration, got \
                 {on_token_calls} calls"
    );
    assert_eq!(
        out.generated_tokens, 1,
        "generation must stop after exactly 1 token (the prefill sample) \
                 once should_cancel goes true at the top of the first decode \
                 iteration, got {}",
        out.generated_tokens
    );
    assert!(!out.stopped, "cancellation is not an OpenAI stop condition");
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Interrupt),
        "should_cancel alone (on_token always returns true) must still \
                 report Interrupt, got {:?}",
        out.stop_reason
    );
}

/// `generate`'s `max_new_tokens == 0` guard must return before the prefill
/// token is ever pushed into the output, matching the CPU `generate()`
/// contract (model::qwen35::generation): zero tokens, `StopReason::Length`.
///
/// Mutation sensitivity: treating the shared preparation's
/// `GenerationPreparation::Complete` as ready would let `generate` sample
/// the prefill token and push it into `generated_ids` before the empty
/// decode loop runs, so `generated_tokens` would be 1 instead of 0.
/// `eos_token_id` is pushed out of the reachable vocab range so this holds
/// regardless of which token greedy sampling picks at prefill.
#[test]
fn metal_generate_zero_budget_reports_length() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 0,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let out = state
        .generate("a", &tokenizer, &gen_cfg)
        .expect("no grammar configured; must not fail closed");

    assert_eq!(
        out.stop_reason,
        Some(StopReason::Length),
        "max_new_tokens == 0 must report Length, got {:?}",
        out.stop_reason
    );
    assert_eq!(
        out.generated_tokens, 0,
        "max_new_tokens == 0 must produce no tokens, got {}",
        out.generated_tokens
    );
}

/// `MetalQwen35State::generate` (the plain/direct entry point) must
/// reject an empty prompt with a typed `Err(Inference("empty
/// prompt"))` (#856), unifying it with all three CPU forward paths
/// instead of the empty `Ok(GenerateOutput { stopped: false,
/// stop_reason: None, .. })` it used to return. See
/// docs/generation-entrypoint-matrix.md row 2.
///
/// This is the production-seam test for `generate`'s shared preparation
/// call; the generation-setup tests cover the pure contract without a GPU.
///
/// Mutation sensitivity: bypassing `prepare_direct_generation` in
/// `generate` lets this request proceed through prefill and sampling
/// instead of erroring, so `result.is_err()` fails.
///
/// Honors
/// `LATTICE_METAL_TEST_ENFORCE` like the prefix-cache empty-prompt
/// test, so a required Metal CI leg fails loud on a missing device
/// instead of silently passing without ever exercising the guard.
#[test]
fn metal_generate_plain_rejects_empty_prompt() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: true,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let result = state.generate("", &tokenizer, &gen_cfg);
    assert!(
        matches!(result, Err(crate::error::InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
        "MetalQwen35State::generate must reject an empty prompt with \
                 Err(Inference(\"empty prompt\")) (#856); got {result:?}"
    );
}

/// `MetalQwen35State::generate` (the plain/direct entry point) must
/// reject a `GenerateConfig` with `reasoning_budget` set, with a typed
/// `InvalidInput` error, rather than silently ignoring the budget and
/// generating unbudgeted output (ADR-080 C3, #783).
///
/// This is the production-seam test for `generate`'s
/// `GenerationEntryContract::MetalDirect`, proving the caller selects
/// the fail-closed contract.
///
/// Mutation sensitivity: selecting `MetalStreaming` instead lets this
/// request proceed through prefill and sampling instead of erroring, so
/// `result.is_err()` fails.
#[test]
fn metal_generate_plain_rejects_reasoning_budget_config() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: true,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: Some(2),
        logprobs: None,
    };

    let result = state.generate("a", &tokenizer, &gen_cfg);
    assert!(
        matches!(result, Err(crate::error::InferenceError::InvalidInput(_))),
        "MetalQwen35State::generate must fail closed with InvalidInput when \
                 reasoning_budget is set (ADR-080 C3, #783); got {result:?}"
    );
}

/// Sibling to `metal_generate_plain_rejects_reasoning_budget_config`
/// (PR #787): `MetalQwen35State::generate`
/// (the plain/direct entry point) unconditionally returns
/// `token_logprobs: vec![]` on every return path, including the MTP
/// and self-spec fast-path delegations, and had no
/// `check_logprobs_not_set` preflight at all -- routing a `logprobs`
/// request around the MTP/self-spec predicates alone would not have
/// closed this, since the plain fallback path is exactly as silent.
///
/// Mutation sensitivity: selecting `MetalStreaming` instead lets this
/// request proceed through prefill and sampling instead of erroring, so
/// `result.is_err()` fails.
#[test]
fn metal_generate_plain_rejects_logprobs_config() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
        stop_token_ids: vec![],
        enable_thinking: true,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: Some(0),
    };

    let result = state.generate("a", &tokenizer, &gen_cfg);
    assert!(
        matches!(result, Err(crate::error::InferenceError::InvalidInput(_))),
        "MetalQwen35State::generate must fail closed with InvalidInput when \
                 logprobs is set (PR #787); got {result:?}"
    );
}

/// `generate_streaming` must reject an empty prompt with a typed
/// `Err(Inference("empty prompt"))` (#856), unifying it with all
/// three CPU forward paths instead of the empty `Ok` (`on_token`
/// never invoked, `stopped: false`, `stop_reason: None`) it used to
/// return. `generate_streaming` is a thin `should_cancel = || false`
/// wrapper over `generate_streaming_with_cancel` (the actual guard
/// call site fixed by #856), so this one test covers both. See
/// docs/generation-entrypoint-matrix.md row 2.
///
/// Mutation sensitivity: bypassing the shared `prepare_generation`
/// call in `generate_streaming_with_cancel` lets this request proceed
/// through prefill and sampling instead of erroring.
///
/// Honors
/// `LATTICE_METAL_TEST_ENFORCE` like the prefix-cache empty-prompt
/// test, so a required Metal CI leg fails loud on a missing device
/// instead of silently passing without ever exercising the guard.
#[test]
fn metal_generate_streaming_rejects_empty_prompt() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _gpu_guard = gpu_test_lock();
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let mut calls = 0u32;
    let result = state.generate_streaming("", &tokenizer, &gen_cfg, |_delta, _id| {
        calls += 1;
        true
    });

    assert!(
        matches!(result, Err(crate::error::InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
        "generate_streaming must reject an empty prompt with \
                 Err(Inference(\"empty prompt\")) (#856); got {result:?}"
    );
    assert_eq!(
        calls, 0,
        "an empty-prompt rejection must never invoke on_token, got {calls} call(s)"
    );
}

/// `generate_streaming`'s `max_new_tokens == 0` guard must return before the
/// prefill token is sampled, pushed, or streamed, matching the CPU
/// `generate_streaming()` contract (model::qwen35::generation): zero tokens,
/// `StopReason::Length`, and `on_token` never invoked.
///
/// Mutation sensitivity: treating the shared preparation's
/// `GenerationPreparation::Complete` as ready would make streaming sample
/// and emit the prefill token before the empty decode loop runs, so
/// `generated_tokens` would be 1 and the callback would fire.
#[test]
fn metal_generate_streaming_zero_budget_reports_length() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 0,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let mut calls = 0u32;
    let out = state
        .generate_streaming("a", &tokenizer, &gen_cfg, |_delta, _id| {
            calls += 1;
            true
        })
        .expect("no grammar configured; must not fail closed");

    assert_eq!(
        calls, 0,
        "max_new_tokens == 0 must never invoke on_token, got {calls} call(s)"
    );
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Length),
        "max_new_tokens == 0 must report Length, got {:?}",
        out.stop_reason
    );
    assert_eq!(
        out.generated_tokens, 0,
        "max_new_tokens == 0 must produce no tokens, got {}",
        out.generated_tokens
    );
}

/// Regression guard for the streaming-retention decode loop: the non-prefix
/// `generate_streaming` decode loop must append every non-empty delta to the
/// returned `GenerateOutput.text`, not just hand it to `on_token`. A prior
/// version of the loop body called `on_token(&delta, next_id)` without first
/// doing `text.push_str(&delta)`, so `out.text` silently dropped every decode
/// token's contribution (only the prefill token and the final incomplete-UTF-8
/// tail were ever appended). `single_char_vocab_tokenizer` guarantees every
/// token decodes to exactly one non-empty-delta character (no UTF-8 boundary
/// holdback), so this test exercises the loop body on every one of its
/// iterations rather than being satisfied by the prefill token alone.
#[test]
fn streaming_text_accumulates_every_loop_delta_not_only_prefill() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 5,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let (mut cfg, weights) = tiny_hybrid_fixture();
    // Push eos_token_id out of the model's reachable output range so greedy
    // sampling cannot end the stream after the prefill token — this test
    // needs several decode-loop iterations (not just the prefill push) to
    // exercise the bug.
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let mut accumulated = String::new();
    let out = state
        .generate_streaming("a", &tokenizer, &gen_cfg, |delta, _id| {
            accumulated.push_str(delta);
            true
        })
        .expect("no grammar configured; must not fail closed");

    assert!(
        out.generated_tokens >= 3,
        "test needs at least 3 generated tokens to exercise the decode loop, got {}",
        out.generated_tokens
    );
    assert_eq!(
        out.text, accumulated,
        "GenerateOutput.text must equal the concatenation of every on_token \
                 delta (plus the finish() tail, already included in `accumulated` via \
                 the callback) — a mismatch means the non-prefix decode loop dropped \
                 deltas instead of appending them to the caller-owned `text` accumulator"
    );
}

/// KV capacity boundary for the streaming entry: a request whose prompt
/// plus `max_new_tokens` equals the cache length exactly is admitted and
/// runs to its token cap without reaching a full cache, because the
/// last emitted token is never forwarded. The cache is left one slot
/// short of full. Control: one more token of budget is refused at
/// admission, before any state is touched.
#[test]
fn metal_generate_streaming_at_exact_kv_capacity_runs_to_the_token_cap() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    const CAPACITY: usize = 32;
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, CAPACITY).expect("tiny hybrid fixture");
    assert_eq!(state.max_context(), CAPACITY);
    let gen_cfg = |max_new_tokens| GenerateConfig {
        min_p: 0.0,
        max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    // The one-character prompt is one token.
    let out = state
        .generate_streaming("a", &tokenizer, &gen_cfg(CAPACITY - 1), |_, _| true)
        .expect("prompt plus max_new_tokens equal to the capacity is admitted");
    assert_eq!(out.prompt_tokens, 1);
    assert_eq!(out.stop_reason, Some(StopReason::Length));
    assert!(!out.stopped);
    assert_eq!(out.generated_tokens, CAPACITY - 1);
    assert_eq!(
        state.session.position(),
        CAPACITY - 1,
        "the prompt and every emitted token but the last were forwarded"
    );

    let refused = state.generate_streaming("a", &tokenizer, &gen_cfg(CAPACITY), |_, _| {
        panic!("a refused request must not stream")
    });
    assert!(
        matches!(&refused, Err(crate::error::InferenceError::Inference(message))
            if message.contains("exceeds model context window")),
        "one token over the capacity must be refused at admission, got {refused:?}"
    );
}

/// The same KV capacity boundary for the direct entry, which decodes
/// through the same session and driver: a request whose prompt plus
/// `max_new_tokens` equals the cache length runs to its token cap with the
/// cache one slot short of full. Control: one more token of budget is
/// refused at admission.
#[test]
fn metal_generate_at_exact_kv_capacity_runs_to_the_token_cap() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    const CAPACITY: usize = 32;
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, CAPACITY).expect("tiny hybrid fixture");
    let gen_cfg = |max_new_tokens, temperature, top_k| GenerateConfig {
        min_p: 0.0,
        max_new_tokens,
        temperature,
        top_k,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    // Greedy (argmax readback) and sampled (dense readback) requests.
    for (temperature, top_k) in [(0.0, 1), (0.8, 8)] {
        let out = state
            .generate("a", &tokenizer, &gen_cfg(CAPACITY - 1, temperature, top_k))
            .expect("prompt plus max_new_tokens equal to the capacity is admitted");
        assert_eq!(out.prompt_tokens, 1);
        assert_eq!(out.stop_reason, Some(StopReason::Length));
        assert!(!out.stopped);
        assert_eq!(out.generated_tokens, CAPACITY - 1);
        assert_eq!(
            state.session.position(),
            CAPACITY - 1,
            "the prompt and every emitted token but the last were forwarded"
        );
    }

    let refused = state.generate("a", &tokenizer, &gen_cfg(CAPACITY, 0.0, 1));
    assert!(
        matches!(&refused, Err(crate::error::InferenceError::Inference(message))
            if message.contains("exceeds model context window")),
        "one token over the capacity must be refused at admission, got {refused:?}"
    );
}

/// A greedy direct request with no compact route reads back one argmax id
/// per decode step, never a vocabulary-sized row: the only full-vocabulary
/// readback is the one the prefill-derived token is sampled from. Control:
/// a repetition penalty needs the dense row, and reads one back per token.
#[test]
fn metal_generate_greedy_reads_back_one_argmax_id_per_decode_step() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;

    const MAX_NEW_TOKENS: usize = 9;
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    state.path_proof_enabled = true;
    let gen_cfg = |repetition_penalty| GenerateConfig {
        min_p: 0.0,
        max_new_tokens: MAX_NEW_TOKENS,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let full_rows = |state: &mut MetalQwen35State, repetition_penalty| {
        state.reset_path_proof_counters();
        let out = with_compact_topk_env("0", || {
            state.generate("a", &tokenizer, &gen_cfg(repetition_penalty))
        })
        .expect("an ordinary greedy request generates");
        assert_eq!(out.generated_tokens, MAX_NEW_TOKENS);
        let readback = state.logit_readback_path_proof_snapshot();
        assert_eq!(readback.decode_compact_candidate, 0);
        assert_eq!(readback.prefill_compact_candidate, 0);
        readback.decode_full_vocab + readback.prefill_full_vocab
    };

    assert_eq!(full_rows(&mut state, 1.0), 1);
    assert_eq!(full_rows(&mut state, 1.3), MAX_NEW_TOKENS as u64);
}

#[test]
fn self_spec_pool_holds_one_slot_per_verify_token_plus_base() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    let state = with_self_spec_env(|| {
        let (cfg, weights) = tiny_hybrid_fixture();
        MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture")
    });

    let pool = state
        .session
        .gdn_checkpoints
        .as_ref()
        .expect("self-spec pool must be allocated");
    // verify_tokens_batched writes slot 0 (pre-verify) plus one slot per processed
    // token. Self-spec sends `1 + SELF_SPEC_MAX_DRAFT` tokens, so the pool must
    // hold `2 + SELF_SPEC_MAX_DRAFT` slots total.
    assert!(
        pool.conv_slots.len() >= 2 + SELF_SPEC_MAX_DRAFT,
        "pool conv_slots = {} must be >= {} (1 base + 1 + SELF_SPEC_MAX_DRAFT verified tokens)",
        pool.conv_slots.len(),
        2 + SELF_SPEC_MAX_DRAFT,
    );
    assert!(
        pool.s_slots.len() >= 2 + SELF_SPEC_MAX_DRAFT,
        "pool s_slots = {} must be >= {}",
        pool.s_slots.len(),
        2 + SELF_SPEC_MAX_DRAFT,
    );
}

/// Regression: self-spec slot 0 must hold the *pre-draft* GDN state, not the
/// state after GDN-only draft forwards. The tiny fixture has all-zero weights
/// which makes GDN-only forwards numerically uninteresting on their own, so we
/// seed the live GDN buffers with a recognizable non-zero pattern before the
/// round, snapshot that as the pre-draft baseline, then assert slot 0 still
/// matches it after a self-spec round. If slot 0 were captured AFTER GDN-only
/// draft forwards, the conv buffer's rolling shift (which moves the planted
/// pattern even when conv weights are zero) would shift slot 0's contents and
/// the check would fail.
#[test]
fn self_spec_slot0_holds_pre_draft_state_after_round() {
    let _gpu_guard = gpu_test_lock();
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    use crate::generation::GenerateConfig;
    use crate::speculative::MtpTargetVerifier;

    let tokenizer = minimal_bpe_tokenizer();
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(7),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let mut state = with_self_spec_env(|| {
        let (cfg, weights) = tiny_hybrid_fixture();
        MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture")
    });

    state.reset_state();
    let real_prefill = state.forward_prefill(&[0u32]);

    // Plant a recognizable non-zero pattern into the live GDN GPU buffers so the
    // pre-draft / post-draft slot contents are observably distinct. The exact
    // values don't matter — only that conv rolling and S-matrix decay would
    // perturb them if GDN-only draft forwards touched the captured slot.
    for (li, layer) in state.session.gdn_gpu_state.layers().enumerate() {
        let buf = layer.s_matrix();
        let n = (buf.length() / 4) as usize;
        // SAFETY: StorageModeShared; no command buffer in flight in the test.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            for k in 0..n {
                *ptr.add(k) = 0.5 + (li as f32) * 0.01 + (k as f32) * 1e-4;
            }
        }
    }
    for (li, layer) in state.session.gdn_gpu_state.layers().enumerate() {
        let buf = layer.conv_buffer();
        let n = (buf.length() / 4) as usize;
        // SAFETY: see above.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            for k in 0..n {
                *ptr.add(k) = -0.7 - (li as f32) * 0.013 - (k as f32) * 2e-4;
            }
        }
    }

    let snap_pre_round = state.snapshot_gdn_states();
    assert!(
        !snap_pre_round.is_empty(),
        "tiny hybrid fixture must have GDN layers"
    );

    // Precondition for the regression: GDN-only draft forwards must actually
    // mutate the live GDN buffers so we are checking that slot 0 is restored
    // to pre-draft state. We don't want to leave the live state perturbed when
    // generate_greedy_self_spec runs, so snapshot first, run a draft-only
    // forward to confirm it perturbs GDN, then restore.
    let observed_drift = {
        let probe_token = 1u32;
        let probe_pos = state.session.kv_cache.seq_len;
        let _ = state.forward_step_gdn_only(probe_token, probe_pos);
        let post = state.snapshot_gdn_states();
        let mut max_abs = 0.0f32;
        for (li, layer) in post.iter().enumerate() {
            for (k, &v) in layer.0.iter().enumerate() {
                max_abs = max_abs.max((v - snap_pre_round[li].0[k]).abs());
            }
            for (k, &v) in layer.1.iter().enumerate() {
                max_abs = max_abs.max((v - snap_pre_round[li].1[k]).abs());
            }
        }
        // Restore pre-round baseline so the self-spec round sees the same state
        // the test sampled into `real_prefill`.
        state.restore_gdn_states(&snap_pre_round);
        max_abs
    };
    assert!(
        observed_drift > 0.0,
        "forward_step_gdn_only must actually perturb live GDN buffers; \
                 otherwise this test cannot detect slot-0 contamination"
    );

    // Force a non-stop pending_token via a one-hot logits vector at a non-EOS
    // id so generate_greedy_self_spec actually enters the round loop. Token 3
    // is a regular "t0" vocabulary entry in the minimal tokenizer.
    let mut forced_logits = vec![0.0f32; real_prefill.len()];
    let target_id = 3usize.min(real_prefill.len() - 1);
    forced_logits[target_id] = 100.0;

    let out = with_self_spec_env(|| {
        state
            .generate_greedy_self_spec(&forced_logits, 1, &tokenizer, &gen_cfg)
            .expect("self-spec route")
    });
    assert!(
        out.generated_tokens > 0,
        "self-spec round must actually execute; got {} generated tokens",
        out.generated_tokens
    );

    // After running a self-spec round, slot 0 must equal the pre-draft baseline
    // (which we captured AFTER prefill). If slot 0 had been written by the
    // GDN-only draft forwards or by the verify-loop checkpoint at the *wrong*
    // time, the conv buffer's rolling shift would have moved our planted
    // pattern and this check would fail.
    let pool = state
        .session
        .gdn_checkpoints
        .as_ref()
        .expect("pool must be allocated");
    for (li, snap_layer) in snap_pre_round.iter().enumerate() {
        let slot0_s = &pool.s_slots[0][li];
        let slot0_conv = &pool.conv_slots[0][li];
        let n_s = (slot0_s.length() / 4) as usize;
        let n_conv = (slot0_conv.length() / 4) as usize;
        // SAFETY: shared-storage buffers; the round has fully completed.
        let live_s = unsafe {
            let ptr = slot0_s.contents() as *const f32;
            std::slice::from_raw_parts(ptr, n_s)
        };
        let live_conv = unsafe {
            let ptr = slot0_conv.contents() as *const f32;
            std::slice::from_raw_parts(ptr, n_conv)
        };
        assert_eq!(live_s.len(), snap_layer.0.len());
        assert_eq!(live_conv.len(), snap_layer.1.len());
        for k in 0..live_s.len() {
            let delta = (live_s[k] - snap_layer.0[k]).abs();
            assert!(
                delta < 1e-5,
                "slot 0 s[{li}][{k}] = {} drifted from pre-draft = {} (delta {})",
                live_s[k],
                snap_layer.0[k],
                delta
            );
        }
        for k in 0..live_conv.len() {
            let delta = (live_conv[k] - snap_layer.1[k]).abs();
            assert!(
                delta < 1e-5,
                "slot 0 conv[{li}][{k}] = {} drifted from pre-draft = {} (delta {})",
                live_conv[k],
                snap_layer.1[k],
                delta
            );
        }
        // Sanity check: the planted pattern should still be non-zero so this
        // test would detect a bug; if the conv buffer were all zeros the
        // rolling shift would be invisible.
        assert!(
            snap_layer.1.iter().any(|&v| v.abs() > 0.0),
            "test fixture must plant non-zero conv pattern on layer {li}"
        );
    }
}

#[test]
fn snapshot_gdn_states_roundtrips_through_metal_buffers() {
    let _gpu_guard = gpu_test_lock();
    use crate::speculative::MtpTargetVerifier;
    let Some(_) = metal::Device::system_default() else {
        return;
    };

    let mut state = with_self_spec_env(|| {
        let (cfg, weights) = tiny_hybrid_fixture();
        MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture")
    });

    // Write a recognizable pattern into the live GPU GDN buffers.
    for (li, layer) in state.session.gdn_gpu_state.layers().enumerate() {
        let buf = layer.s_matrix();
        let n = (buf.length() / 4) as usize;
        // SAFETY: StorageModeShared buffer; no command buffer in flight in the test.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            for k in 0..n {
                *ptr.add(k) = (li as f32) + (k as f32) * 1e-4;
            }
        }
    }
    for (li, layer) in state.session.gdn_gpu_state.layers().enumerate() {
        let buf = layer.conv_buffer();
        let n = (buf.length() / 4) as usize;
        // SAFETY: see above.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            for k in 0..n {
                *ptr.add(k) = -(li as f32) - (k as f32) * 1e-4;
            }
        }
    }

    let snap = state.snapshot_gdn_states();
    assert_eq!(snap.len(), state.session.gdn_gpu_state.len());

    // Clobber the live buffers.
    for layer in state.session.gdn_gpu_state.layers() {
        let buf = layer.s_matrix();
        let n = (buf.length() / 4) as usize;
        // SAFETY: see above.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            for k in 0..n {
                *ptr.add(k) = 0.0;
            }
        }
    }
    for layer in state.session.gdn_gpu_state.layers() {
        let buf = layer.conv_buffer();
        let n = (buf.length() / 4) as usize;
        // SAFETY: see above.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            for k in 0..n {
                *ptr.add(k) = 0.0;
            }
        }
    }

    state.restore_gdn_states(&snap);

    for (li, layer) in state.session.gdn_gpu_state.layers().enumerate() {
        let buf = layer.s_matrix();
        let n = (buf.length() / 4) as usize;
        // SAFETY: see above.
        let live = unsafe {
            let ptr = buf.contents() as *const f32;
            std::slice::from_raw_parts(ptr, n)
        };
        for (k, &v) in live.iter().enumerate() {
            let expected = (li as f32) + (k as f32) * 1e-4;
            assert!(
                (v - expected).abs() < 1e-6,
                "s_matrices[{li}][{k}] = {v} after restore, expected {expected}"
            );
        }
    }
}

#[test]
fn restore_gdn_states_rejects_short_snapshot_without_writes() {
    use crate::speculative::MtpTargetVerifier;
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let mut state = with_self_spec_env(|| {
        let (cfg, weights) = tiny_hybrid_fixture();
        MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture")
    });
    let mut malformed = state.snapshot_gdn_states();
    assert!(
        malformed.len() > 1,
        "fixture must expose a partial-restore boundary"
    );
    malformed.pop();

    for (li, layer) in state.session.gdn_gpu_state.layers().enumerate() {
        let buf = layer.s_matrix();
        let n = (buf.length() / 4) as usize;
        // SAFETY: StorageModeShared buffer; no command buffer is in flight.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            for k in 0..n {
                *ptr.add(k) = 10.0 + li as f32 + k as f32 * 1e-4;
            }
        }
    }
    for (li, layer) in state.session.gdn_gpu_state.layers().enumerate() {
        let buf = layer.conv_buffer();
        let n = (buf.length() / 4) as usize;
        // SAFETY: see above.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            for k in 0..n {
                *ptr.add(k) = -10.0 - li as f32 - k as f32 * 1e-4;
            }
        }
    }

    let before = state.snapshot_gdn_states();
    state.restore_gdn_states(&malformed);
    let after = state.snapshot_gdn_states();
    assert!(
        after == before,
        "a malformed snapshot must not partially overwrite live GDN buffers"
    );
}
