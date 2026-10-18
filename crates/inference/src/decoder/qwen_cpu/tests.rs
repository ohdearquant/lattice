use super::*;
use crate::generation::GenerateConfig;
use crate::model::qwen35::test_support::tiny_zero_model;
use crate::tokenizer::common::Tokenizer;

fn cancel_false() -> impl Fn() -> bool {
    || false
}

fn cancel_true() -> impl Fn() -> bool {
    || true
}

#[test]
fn finish_invalidates_the_prediction_for_every_disposition() {
    let model = tiny_zero_model();
    let prompt_ids = tokenize(&model, "abc");
    let config = GenerateConfig {
        temperature: 0.0,
        max_new_tokens: 2,
        ..Default::default()
    };
    for disposition in [FinishDisposition::Reusable, FinishDisposition::Poisoned] {
        let mut session = QwenCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(7));
        session.prefill(&cancel_false()).expect("prefill");
        let outcome = session
            .select(&SelectionRequest {
                config: &config,
                history: &prompt_ids,
                grammar_mask: None,
            })
            .expect("selection");
        let SelectOutcome::Candidate(candidate) = outcome else {
            panic!("unconstrained selection must yield a candidate")
        };
        assert!(session.ledger.is_live(candidate.prediction));
        session.finish(disposition).expect("finish");
        assert!(
            !session.ledger.is_live(candidate.prediction),
            "{disposition:?}"
        );
    }
}

fn tokenize(model: &Qwen35Model, text: &str) -> Vec<u32> {
    let input = model.tokenizer.tokenize(text);
    input.input_ids[..input.real_length].to_vec()
}

#[test]
fn request_buffers_keep_kv_capacity() {
    let model = tiny_zero_model();
    let prompt_ids = tokenize(&model, "abc");
    let max_new_tokens = 16;
    let required = prompt_ids.len() + max_new_tokens;
    let mut session = QwenCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(7));
    session.reserve_generation(required);
    let kv_needed = required * model.config.full_kv_dim();
    for store in session.kv_cache.k.iter().chain(&session.kv_cache.v) {
        assert!(store.capacity() >= kv_needed);
        assert!(store.is_empty());
    }
    let score_capacity = session.scratch.scores.capacity();
    assert!(session.scratch.scores.len() >= model.config.num_attention_heads * (required + 1));
    let pointers: Vec<_> = session
        .kv_cache
        .k
        .iter()
        .chain(&session.kv_cache.v)
        .map(|store| (store.as_ptr(), store.capacity()))
        .collect();
    let cancel = cancel_false();
    session.prefill(&cancel).unwrap();
    let config = GenerateConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        max_new_tokens,
        ..Default::default()
    };
    let mut history = prompt_ids;
    for _ in 0..max_new_tokens {
        let selected = session
            .select(&SelectionRequest {
                config: &config,
                history: &history,
                grammar_mask: None,
            })
            .unwrap();
        let SelectOutcome::Candidate(candidate) = selected else {
            panic!("unconstrained fixture must select");
        };
        session
            .decode(
                &AcceptedToken {
                    final_id: candidate.candidate_id,
                    prediction: candidate.prediction,
                },
                &cancel,
            )
            .unwrap();
        history.push(candidate.candidate_id);
        for (store, &(pointer, capacity)) in session
            .kv_cache
            .k
            .iter()
            .chain(&session.kv_cache.v)
            .zip(&pointers)
        {
            assert_eq!(store.as_ptr(), pointer);
            assert_eq!(store.capacity(), capacity);
        }
        assert_eq!(session.scratch.scores.capacity(), score_capacity);
    }
    assert_eq!(session.kv_cache.seq_len, required);
}

// -----------------------------------------------------------------
// Acceptance 1 (checkpoint-free half): one full step,
// prefill -> select -> metadata -> decode, over a tiny zero-weight
// dense model. All-zero weights make every logit zero, so greedy
// (temperature 0.0) sampling is deterministic without needing a real
// checkpoint for this shape of assertion.
// -----------------------------------------------------------------

#[test]
fn full_step_lifecycle_over_tiny_dense_model() {
    let model = tiny_zero_model();
    let prompt_ids = tokenize(&model, "abc");
    assert!(
        !prompt_ids.is_empty(),
        "test tokenizer must produce >=1 id for \"abc\""
    );
    let prompt_len = prompt_ids.len();

    let mut session = QwenCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(7));

    let cancel = cancel_false();
    let stamp = session
        .prefill(&cancel)
        .expect("prefill over a dense tiny model must succeed");
    assert_eq!(stamp.evaluated_len, prompt_len);
    assert_eq!(
        stamp.prediction, None,
        "prefill must not open a prediction; select does"
    );

    let history = prompt_ids.clone();
    let gen_cfg = GenerateConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let request = SelectionRequest {
        config: &gen_cfg,
        history: &history,
        grammar_mask: None,
    };
    let candidate = match session
        .select(&request)
        .expect("select over freshly prefilled logits must succeed")
    {
        SelectOutcome::Candidate(c) => c,
        SelectOutcome::GrammarExhausted => panic!("no grammar set on this session"),
    };

    // Ledger lifecycle, open half: live immediately after select.
    assert!(session.ledger.is_live(candidate.prediction));

    let meta_request = MetadataRequest {
        top_logprobs: Some(2),
    };
    let metadata = session
        .metadata(candidate.prediction, candidate.candidate_id, &meta_request)
        .expect("metadata over a live prediction must succeed");
    assert_eq!(metadata.final_token_id, candidate.candidate_id);
    assert_eq!(metadata.prediction, candidate.prediction);
    // All-zero logits -> uniform distribution over `vocab_size` -> every token's
    // reporting log-probability is exactly -ln(vocab_size).
    let vocab_size = model.config.vocab_size;
    let expected_logprob = -(vocab_size as f32).ln();
    assert!(
        (metadata.final_logprob - expected_logprob).abs() < 1e-4,
        "final_logprob={} expected={}",
        metadata.final_logprob,
        expected_logprob
    );
    // metadata does not consume: still live after scoring.
    assert!(session.ledger.is_live(candidate.prediction));

    let accepted = AcceptedToken {
        final_id: metadata.final_token_id,
        prediction: candidate.prediction,
    };
    let stamp2 = session
        .decode(&accepted, &cancel)
        .expect("decode of the freshly opened prediction must succeed");
    assert_eq!(stamp2.evaluated_len, prompt_len + 1);
    assert_eq!(stamp2.prediction, None);

    // Ledger lifecycle, consumed half: no longer live after decode.
    assert!(!session.ledger.is_live(candidate.prediction));

    session
        .finish(FinishDisposition::Reusable)
        .expect("finish(Reusable) must always succeed");
}

// -----------------------------------------------------------------
// Acceptance 2: MoE fallback precedes mutation, at the session level
// (both `gdn_states` and `kv_cache`, not just `kv_cache` the way
// `test_batch_prefill_rejects_moe_without_panic` does). Checkpoint-free:
// `is_moe()` is a pure config predicate, so flagging a tiny dense model
// MoE-shaped via config alone is enough to reach the refusal branch
// without needing real MoE weights.
// -----------------------------------------------------------------

#[test]
fn moe_fallback_precedes_mutation_of_gdn_states_and_kv_cache() {
    let mut model = tiny_zero_model();
    model.config.num_experts = Some(1);
    model.config.num_experts_per_tok = Some(1);
    assert!(model.config.is_moe());

    let prompt_ids = tokenize(&model, "abc");
    assert!(!prompt_ids.is_empty());

    let mut session = QwenCpuSession::new(&model, prompt_ids, 0.0, Some(7));

    let pre_gdn: Vec<(Vec<f32>, Vec<f32>)> = session
        .gdn_states
        .iter()
        .map(|s| (s.s_matrices.clone(), s.conv_buffer.clone()))
        .collect();
    let pre_k = session.kv_cache.k.clone();
    let pre_v = session.kv_cache.v.clone();
    let pre_seq_len = session.kv_cache.seq_len;

    let err = session
        .model
        .prefill_tokens_batched_for_generate(
            &session.prompt_ids,
            &mut session.gdn_states,
            &mut session.kv_cache,
        )
        .expect_err("MoE-flagged config must refuse the batched path");
    assert!(matches!(err, InferenceError::UnsupportedModel(_)));

    let post_gdn: Vec<(Vec<f32>, Vec<f32>)> = session
        .gdn_states
        .iter()
        .map(|s| (s.s_matrices.clone(), s.conv_buffer.clone()))
        .collect();
    assert_eq!(
        pre_gdn, post_gdn,
        "gdn_states must be untouched by a refused batched prefill attempt \
             -- the doc comment on prefill_tokens_batched_for_generate names \
             gdn_states in the same breath as kv_cache, but the existing \
             function-level test never checks it"
    );
    assert_eq!(pre_k, session.kv_cache.k);
    assert_eq!(pre_v, session.kv_cache.v);
    assert_eq!(pre_seq_len, session.kv_cache.seq_len);
}

// -----------------------------------------------------------------
// The escape hatch is a peer of the other two branches, not a modifier:
// this session's own `prefill` must still take the serial path when
// `force_serial_prefill()` is forced true, and produce the same
// evaluated length / logits a batched-first prefill would (all-zero
// weights make the two paths trivially equal here; parity under real
// weights is what generation.rs's own delegation-parity test covers).
// -----------------------------------------------------------------

#[test]
fn escape_hatch_forces_the_serial_branch() {
    use crate::model::qwen35::{FORCE_SERIAL_PREFILL, SERIAL_PREFILL_TEST_LOCK};
    let _guard = SERIAL_PREFILL_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let model = tiny_zero_model();
    let prompt_ids = tokenize(&model, "abc");
    let prompt_len = prompt_ids.len();
    let mut session = QwenCpuSession::new(&model, prompt_ids, 0.0, Some(7));

    FORCE_SERIAL_PREFILL.store(true, std::sync::atomic::Ordering::SeqCst);
    let result = session.prefill(&cancel_false());
    FORCE_SERIAL_PREFILL.store(false, std::sync::atomic::Ordering::SeqCst);

    let stamp = result.expect("forced-serial prefill over a dense tiny model must succeed");
    assert_eq!(stamp.evaluated_len, prompt_len);
    assert_eq!(session.kv_cache.seq_len, prompt_len);
}

// -----------------------------------------------------------------
// Ledger lifecycle mutation target: a cancelled decode must consume
// nothing, and finish(Poisoned) must invalidate a still-live
// prediction. Both are asserted with a before/after control, per the
// fleet rule that a before/after comparison catches what a
// post-state-only check does not.
// -----------------------------------------------------------------

#[test]
fn cancelled_decode_consumes_nothing_and_finish_poisoned_invalidates() {
    let model = tiny_zero_model();
    let prompt_ids = tokenize(&model, "abc");
    let mut session = QwenCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(7));
    session
        .prefill(&cancel_false())
        .expect("prefill must succeed");

    let gen_cfg = GenerateConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let request = SelectionRequest {
        config: &gen_cfg,
        history: &prompt_ids,
        grammar_mask: None,
    };
    let candidate = match session.select(&request).expect("select must succeed") {
        SelectOutcome::Candidate(c) => c,
        SelectOutcome::GrammarExhausted => panic!("no grammar set on this session"),
    };
    assert!(session.ledger.is_live(candidate.prediction)); // control: live before

    let accepted = AcceptedToken {
        final_id: candidate.candidate_id,
        prediction: candidate.prediction,
    };
    let cancelled = session.decode(&accepted, &cancel_true());
    assert!(cancelled.is_err(), "a cancelled decode must not succeed");
    assert!(
        session.ledger.is_live(candidate.prediction),
        "a cancelled decode must not consume the live prediction"
    );

    session
        .finish(FinishDisposition::Poisoned)
        .expect("finish must always return Ok");
    assert!(
        !session.ledger.is_live(candidate.prediction),
        "finish(Poisoned) must invalidate whatever prediction was live"
    );
}

// -----------------------------------------------------------------
// Acceptance 1 (checkpoint-gated half): the same full-step lifecycle,
// over the real dense Qwen3.5-0.8B checkpoint. Requires
// LATTICE_INFERENCE_MODEL_DIR (bf16 safetensors; needs the `f16`
// feature to load) and panics rather than silently skipping when unset
// (`crate::test_support::require_checkpoint_dir`).
// -----------------------------------------------------------------

#[test]
#[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR"]
fn full_step_lifecycle_over_real_checkpoint() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
        .expect("dense Qwen3.5 checkpoint should load successfully");

    let prompt_ids = tokenize(&model, "The quick brown fox jumps over the lazy dog.");
    assert!(!prompt_ids.is_empty());
    let prompt_len = prompt_ids.len();

    let mut session = QwenCpuSession::new(&model, prompt_ids.clone(), 0.0, Some(1234));

    let cancel = cancel_false();
    let stamp = session
        .prefill(&cancel)
        .expect("prefill over the real checkpoint must succeed");
    assert_eq!(stamp.evaluated_len, prompt_len);
    assert_eq!(stamp.prediction, None);

    let gen_cfg = GenerateConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let request = SelectionRequest {
        config: &gen_cfg,
        history: &prompt_ids,
        grammar_mask: None,
    };
    let candidate = match session.select(&request).expect("select must succeed") {
        SelectOutcome::Candidate(c) => c,
        SelectOutcome::GrammarExhausted => panic!("no grammar set on this session"),
    };
    assert!(session.ledger.is_live(candidate.prediction)); // open at select

    let meta_request = MetadataRequest {
        top_logprobs: Some(3),
    };
    let metadata = session
        .metadata(candidate.prediction, candidate.candidate_id, &meta_request)
        .expect("metadata must succeed");
    assert_eq!(metadata.final_token_id, candidate.candidate_id);
    assert!(metadata.final_logprob.is_finite());
    assert_eq!(metadata.top.len(), 3);

    let accepted = AcceptedToken {
        final_id: metadata.final_token_id,
        prediction: candidate.prediction,
    };
    let stamp2 = session
        .decode(&accepted, &cancel)
        .expect("decode must succeed");
    assert_eq!(stamp2.evaluated_len, prompt_len + 1);
    assert!(
        !session.ledger.is_live(candidate.prediction),
        "consumed at decode"
    );

    session
        .finish(FinishDisposition::Reusable)
        .expect("finish must succeed");
}
