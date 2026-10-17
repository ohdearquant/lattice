use super::*;
use crate::stop_reason::StopReason;

/// The wrapper's own single-token forward step, standing in for the session's dispatch.
pub(crate) type ReferenceForward<'a> = dyn Fn(
        u32,
        usize,
        &mut [GatedDeltaNetState],
        &mut KvCache,
        &mut ForwardScratch,
    ) -> Result<(), InferenceError>
    + 'a;

/// The wrapper's public generate function, bound to its fixture.
pub(crate) type PublicGenerate<'a> =
    dyn Fn(&GenerateConfig) -> Result<GenerateOutput, InferenceError> + 'a;

fn plan_for(
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> GenerationPlan {
    match prepare_generation(
        tokenizer,
        prompt,
        gen_cfg,
        cfg.vocab_size,
        rope.max_positions(),
        GenerationEntryContract::StandaloneCpu,
    ) {
        Ok(GenerationPreparation::Ready(plan)) => plan,
        Ok(GenerationPreparation::Complete(_)) => {
            panic!("the request completes during preparation, so no session would run")
        }
        Err(error) => panic!("the request was refused during preparation: {error:?}"),
    }
}

/// The four configs the standalone wrappers pin generation goldens for, plus the step-0 stop
/// and zero-budget cases that reach the driver's early returns.
pub(crate) fn greedy_case() -> GenerateConfig {
    GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    }
}

/// The output a wrapper is pinned to for one config over the shared synthetic fixture: the
/// values the generation goldens record, plus the two early-return cases they do not.
pub(crate) struct Recorded {
    pub(crate) token_ids: &'static [u32],
    pub(crate) text: &'static str,
    pub(crate) stopped: bool,
    pub(crate) stop_reason: StopReason,
}

/// The prompt every fixture generates from; it spans several prefill positions.
pub(crate) const PROMPT_TOKENS: usize = 5;

pub(crate) fn deterministic_cases() -> Vec<(&'static str, GenerateConfig, Recorded)> {
    let base = greedy_case();
    vec![
        (
            "greedy",
            base.clone(),
            Recorded {
                token_ids: &[36, 80, 112, 62, 103, 86],
                text: "aCbkbQa2bHbq",
                stopped: false,
                stop_reason: StopReason::Length,
            },
        ),
        (
            "stop_token",
            GenerateConfig {
                stop_token_ids: vec![80],
                ..base.clone()
            },
            Recorded {
                token_ids: &[36],
                text: "aC",
                stopped: true,
                stop_reason: StopReason::Eos,
            },
        ),
        (
            "stops_at_step_zero",
            GenerateConfig {
                stop_token_ids: vec![36],
                ..base.clone()
            },
            Recorded {
                token_ids: &[],
                text: "",
                stopped: true,
                stop_reason: StopReason::Eos,
            },
        ),
        (
            "one_token",
            GenerateConfig {
                max_new_tokens: 1,
                ..base.clone()
            },
            Recorded {
                token_ids: &[36],
                text: "aC",
                stopped: false,
                stop_reason: StopReason::Length,
            },
        ),
        (
            "zero_tokens",
            GenerateConfig {
                max_new_tokens: 0,
                ..base
            },
            Recorded {
                token_ids: &[],
                text: "",
                stopped: false,
                stop_reason: StopReason::Length,
            },
        ),
    ]
}

pub(crate) fn seeded_case() -> GenerateConfig {
    GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.8,
        top_k: 5,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: Some(1234),
        ..Default::default()
    }
}

/// The seeded config's recorded output over the shared fixture.
pub(crate) fn seeded_recorded() -> Recorded {
    Recorded {
        token_ids: &[36, 80, 103, 112, 87, 86],
        text: "aCbkbHbQbrbq",
        stopped: false,
        stop_reason: StopReason::Length,
    }
}

fn assert_output_is_recorded(
    actual: &GenerateOutput,
    recorded: &Recorded,
    name: &str,
    route: &str,
) {
    assert_eq!(
        actual.token_ids, recorded.token_ids,
        "{name} via {route}: token_ids"
    );
    assert_eq!(actual.text, recorded.text, "{name} via {route}: text");
    assert_eq!(
        actual.prompt_tokens, PROMPT_TOKENS,
        "{name} via {route}: prompt_tokens"
    );
    assert_eq!(
        actual.generated_tokens,
        recorded.token_ids.len(),
        "{name} via {route}: generated_tokens"
    );
    assert_eq!(
        actual.stopped, recorded.stopped,
        "{name} via {route}: stopped"
    );
    assert_eq!(
        actual.stop_reason,
        Some(recorded.stop_reason),
        "{name} via {route}: stop_reason"
    );
    assert!(
        actual.token_logprobs.is_empty(),
        "{name} via {route}: token_logprobs"
    );
}

/// Both the wrapper's public function and the session run directly through the driver must
/// reproduce the recorded output field for field, and the ledger counters must show the
/// driver issued every step.
pub(crate) fn assert_matches_recorded(
    public: &PublicGenerate<'_>,
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    name: &str,
    gen_cfg: &GenerateConfig,
    recorded: &Recorded,
) {
    let wrapper = public(gen_cfg).unwrap_or_else(|e| panic!("{name}: wrapper: {e:?}"));
    assert_output_is_recorded(&wrapper, recorded, name, "the wrapper");
    let (actual, trace) = generate_with_trace(weights, cfg, tokenizer, rope, prompt, gen_cfg)
        .unwrap_or_else(|e| panic!("{name}: session: {e:?}"));
    assert_output_is_recorded(&actual, recorded, name, "the session");

    if gen_cfg.max_new_tokens > 0 {
        assert_eq!(
            trace.opened,
            actual.token_ids.len() + usize::from(actual.stopped),
            "{name}: one prediction is opened per sampled token, the stopping one included"
        );
        assert_eq!(
            trace.consumed + 1,
            trace.opened,
            "{name}: every prediction but the last is consumed by a decode step"
        );
    } else {
        assert_eq!(trace, DriverTrace::default(), "{name}: no step ran");
    }
}

/// Replays the session step by step against a hand-driven copy of the wrapper's own forward
/// step and requires bit-identical logits for the final prefill position and for each decode
/// step's output. Comparing logits rather than tokens matters because a wrong position or
/// window can change the logits without changing the sampled token on a given prompt.
pub(crate) fn assert_logits_replay(
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
    steps: usize,
    forward: &ReferenceForward<'_>,
) {
    let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
    let prompt_ids = plan.prompt_ids.clone();
    let prompt_len = plan.prompt_len;
    let vocab = cfg.vocab_size;
    let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

    let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
        .map(|_| GatedDeltaNetState::new(cfg))
        .collect();
    let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();

    for (pos, &token_id) in prompt_ids.iter().enumerate() {
        forward(token_id, pos, &mut gdn_states, &mut kv_cache, &mut scratch)
            .expect("reference prefill step");
        if pos + 1 < prompt_len {
            kv_cache.seq_len += 1;
        }
    }
    kv_cache.seq_len = prompt_len;

    let never = || false;
    let stamp = session.prefill(&never).expect("session prefill");
    assert_eq!(stamp.evaluated_len, kv_cache.seq_len, "prefill length");
    assert_logits_bits_equal(
        &session.scratch.logits[..vocab],
        &scratch.logits[..vocab],
        "prefill",
    );

    let mut history = prompt_ids;
    let mut previous: Vec<u32> = scratch.logits[..vocab]
        .iter()
        .map(|logit| logit.to_bits())
        .collect();
    for step in 0..steps {
        let request = SelectionRequest {
            config: gen_cfg,
            history: &history,
            grammar_mask: None,
        };
        let candidate = match session.select(&request).expect("session select") {
            SelectOutcome::Candidate(candidate) => candidate,
            SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
        };
        history.push(candidate.candidate_id);

        let stamp = session
            .decode(
                &AcceptedToken {
                    final_id: candidate.candidate_id,
                    prediction: candidate.prediction,
                },
                &never,
            )
            .expect("session decode");

        forward(
            candidate.candidate_id,
            kv_cache.seq_len,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        )
        .expect("reference decode step");
        kv_cache.seq_len += 1;

        assert_eq!(
            stamp.evaluated_len, kv_cache.seq_len,
            "decode step {step}: evaluated length"
        );
        assert_logits_bits_equal(
            &session.scratch.logits[..vocab],
            &scratch.logits[..vocab],
            &format!("decode step {step}"),
        );

        let current: Vec<u32> = scratch.logits[..vocab]
            .iter()
            .map(|logit| logit.to_bits())
            .collect();
        assert_ne!(
            current, previous,
            "decode step {step}: the logits did not move, so the comparison above proves nothing"
        );
        previous = current;
    }
}

fn assert_logits_bits_equal(session: &[f32], reference: &[f32], what: &str) {
    assert_eq!(session.len(), reference.len(), "{what}: logits length");
    let differing = session
        .iter()
        .zip(reference)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "{what}: {differing} of {} logits differ in their bits",
        session.len()
    );
}

/// A prediction can be consumed once: the second decode with the same accepted token must be
/// refused and must leave the cache position and logits as the first decode left them.
pub(crate) fn assert_rejects_stale_prediction(
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) {
    let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
    let prompt_ids = plan.prompt_ids.clone();
    let vocab = cfg.vocab_size;
    let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

    let never = || false;
    session.prefill(&never).expect("session prefill");
    let request = SelectionRequest {
        config: gen_cfg,
        history: &prompt_ids,
        grammar_mask: None,
    };
    let candidate = match session.select(&request).expect("session select") {
        SelectOutcome::Candidate(candidate) => candidate,
        SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
    };
    let accepted = AcceptedToken {
        final_id: candidate.candidate_id,
        prediction: candidate.prediction,
    };

    session
        .decode(&accepted, &never)
        .expect("the first decode consumes the live prediction");
    let seq_len = session.kv_cache.seq_len;
    let logits: Vec<u32> = session.scratch.logits[..vocab]
        .iter()
        .map(|logit| logit.to_bits())
        .collect();

    match session.decode(&accepted, &never) {
        Err(InferenceError::Inference(message)) => assert!(
            message.contains("stale prediction id"),
            "unexpected refusal text: {message}"
        ),
        Err(other) => panic!("a stale prediction was refused with the wrong error: {other:?}"),
        Ok(_) => panic!("a consumed prediction was accepted a second time"),
    }
    assert_eq!(
        session.kv_cache.seq_len, seq_len,
        "the refused decode advanced the cache"
    );
    let after: Vec<u32> = session.scratch.logits[..vocab]
        .iter()
        .map(|logit| logit.to_bits())
        .collect();
    assert_eq!(after, logits, "the refused decode ran a forward pass");
}

/// A cancelled step fails before it touches the session: prefill leaves the cache empty, and
/// a cancelled decode neither consumes the prediction nor moves the cache, so the same
/// token still decodes afterwards.
pub(crate) fn assert_cancellation_precedes_state_change(
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) {
    let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
    let prompt_ids = plan.prompt_ids.clone();
    let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

    let cancelled = || true;
    let never = || false;
    // The Q8 NEON session sizes its scratch at construction, so an empty logits buffer is
    // not the untouched state; compare against the buffer as constructed instead.
    let logits_before: Vec<u32> = session
        .scratch
        .logits
        .iter()
        .map(|logit| logit.to_bits())
        .collect();
    assert!(session.prefill(&cancelled).is_err());
    assert_eq!(session.kv_cache.seq_len, 0, "cancelled prefill ran a step");
    let logits_after: Vec<u32> = session
        .scratch
        .logits
        .iter()
        .map(|logit| logit.to_bits())
        .collect();
    assert_eq!(
        logits_after, logits_before,
        "cancelled prefill ran a forward pass"
    );

    session.prefill(&never).expect("session prefill");
    let request = SelectionRequest {
        config: gen_cfg,
        history: &prompt_ids,
        grammar_mask: None,
    };
    let candidate = match session.select(&request).expect("session select") {
        SelectOutcome::Candidate(candidate) => candidate,
        SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
    };
    let accepted = AcceptedToken {
        final_id: candidate.candidate_id,
        prediction: candidate.prediction,
    };
    let seq_len = session.kv_cache.seq_len;

    assert!(session.decode(&accepted, &cancelled).is_err());
    assert_eq!(
        session.kv_cache.seq_len, seq_len,
        "cancelled decode advanced the cache"
    );
    session
        .decode(&accepted, &never)
        .expect("the cancelled decode left the prediction live");
}

/// `select` and `metadata` refuse controls the session does not declare, and `select`
/// still works for the same session when none is supplied.
pub(crate) fn assert_refuses_undeclared_controls(
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) {
    let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
    let prompt_ids = plan.prompt_ids.clone();
    let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);
    let never = || false;
    session.prefill(&never).expect("session prefill");

    let mask = |_logits: &mut [f32]| -> Result<(), InferenceError> { Ok(()) };
    let masked = SelectionRequest {
        config: gen_cfg,
        history: &prompt_ids,
        grammar_mask: Some(&mask),
    };
    assert!(matches!(
        session.select(&masked),
        Err(InferenceError::InvalidInput(_))
    ));

    let plain = SelectionRequest {
        config: gen_cfg,
        history: &prompt_ids,
        grammar_mask: None,
    };
    let candidate = match session.select(&plain).expect("select without a mask") {
        SelectOutcome::Candidate(candidate) => candidate,
        SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
    };
    let metadata = session.metadata(
        candidate.prediction,
        candidate.candidate_id,
        &MetadataRequest {
            top_logprobs: Some(0),
        },
    );
    assert!(matches!(metadata, Err(InferenceError::InvalidInput(_))));
}

/// `select` refuses logprobs, stop strings and a reasoning budget in the request's config
/// with `InvalidInput`, each with its own message, and a refusal changes nothing: the RNG
/// state is untouched, no prediction is opened, and the next ordinary `select` returns the
/// candidate an untouched twin session returns.
pub(crate) fn assert_select_refuses_config_controls(
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) {
    let never = || false;
    let mut session = StandaloneCpuSession::new(
        weights,
        cfg,
        rope,
        plan_for(cfg, tokenizer, rope, prompt, gen_cfg),
    );
    let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
    let prompt_ids = plan.prompt_ids.clone();
    let mut twin = StandaloneCpuSession::new(weights, cfg, rope, plan);
    session.prefill(&never).expect("session prefill");
    twin.prefill(&never).expect("twin prefill");
    assert!(
        gen_cfg.temperature > 0.0,
        "a greedy config never draws from the RNG, so a refusal that drew would go unseen"
    );

    let controls: Vec<(&str, GenerateConfig, &str)> = vec![
        (
            "logprobs",
            GenerateConfig {
                logprobs: Some(0),
                ..gen_cfg.clone()
            },
            "session does not declare logprobs support but the selection config sets logprobs",
        ),
        (
            "stop_strings",
            GenerateConfig {
                stop_strings: vec!["</s>".to_string()],
                ..gen_cfg.clone()
            },
            "session does not declare stop_strings support but the selection config sets \
                 stop_strings",
        ),
        (
            "reasoning_budget",
            GenerateConfig {
                reasoning_budget: Some(16),
                ..gen_cfg.clone()
            },
            "session does not declare reasoning_budget support but the selection config sets \
                 reasoning_budget",
        ),
    ];
    let rng_before = session.rng_state;
    for (control, refused_cfg, message) in &controls {
        let request = SelectionRequest {
            config: refused_cfg,
            history: &prompt_ids,
            grammar_mask: None,
        };
        match session.select(&request) {
            Err(InferenceError::InvalidInput(actual)) => {
                assert_eq!(&actual, message, "{control}: refusal text")
            }
            Err(other) => panic!("{control}: wrong error {other:?}"),
            Ok(_) => panic!("{control}: select accepted the control"),
        }
        assert_eq!(
            session.rng_state, rng_before,
            "{control}: the refused select drew from the RNG"
        );
        assert_eq!(
            session.ledger.next_seq, 0,
            "{control}: the refused select opened a prediction"
        );
        assert!(
            session.ledger.live.is_none(),
            "{control}: the refused select left a live prediction"
        );
    }

    let plain = SelectionRequest {
        config: gen_cfg,
        history: &prompt_ids,
        grammar_mask: None,
    };
    let candidate = match session.select(&plain).expect("select after the refusals") {
        SelectOutcome::Candidate(candidate) => candidate,
        SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
    };
    let twin_candidate = match twin.select(&plain).expect("twin select") {
        SelectOutcome::Candidate(candidate) => candidate,
        SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
    };
    assert_eq!(
        candidate.candidate_id, twin_candidate.candidate_id,
        "the refusals moved the sampling state"
    );
    assert_eq!(session.rng_state, twin.rng_state, "RNG state after select");
    assert_eq!(session.ledger.next_seq, 1, "exactly one prediction opened");
}

/// `finish` ends the live prediction whatever the disposition: after
/// `finish(Reusable)` the last selected prediction cannot be decoded, and the refused
/// decode leaves the cache position and logits alone.
pub(crate) fn assert_finish_invalidates_the_live_prediction(
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) {
    let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
    let prompt_ids = plan.prompt_ids.clone();
    let vocab = cfg.vocab_size;
    let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

    let never = || false;
    session.prefill(&never).expect("session prefill");
    let request = SelectionRequest {
        config: gen_cfg,
        history: &prompt_ids,
        grammar_mask: None,
    };
    let candidate = match session.select(&request).expect("session select") {
        SelectOutcome::Candidate(candidate) => candidate,
        SelectOutcome::GrammarExhausted => panic!("no grammar is set, so none can exhaust"),
    };
    let accepted = AcceptedToken {
        final_id: candidate.candidate_id,
        prediction: candidate.prediction,
    };
    assert!(
        session.ledger.is_live(accepted.prediction),
        "the selected prediction must be live before finish"
    );

    session
        .finish(FinishDisposition::Reusable)
        .expect("finish(Reusable)");
    assert!(
        !session.ledger.is_live(accepted.prediction),
        "finish(Reusable) left the prediction live"
    );

    let seq_len = session.kv_cache.seq_len;
    let logits: Vec<u32> = session.scratch.logits[..vocab]
        .iter()
        .map(|logit| logit.to_bits())
        .collect();
    match session.decode(&accepted, &never) {
        Err(InferenceError::Inference(message)) => assert!(
            message.contains("stale prediction id"),
            "unexpected refusal text: {message}"
        ),
        Err(other) => {
            panic!("a finished prediction was refused with the wrong error: {other:?}")
        }
        Ok(_) => panic!("a prediction decoded after finish(Reusable)"),
    }
    assert_eq!(
        session.kv_cache.seq_len, seq_len,
        "the refused decode advanced the cache"
    );
    let after: Vec<u32> = session.scratch.logits[..vocab]
        .iter()
        .map(|logit| logit.to_bits())
        .collect();
    assert_eq!(after, logits, "the refused decode ran a forward pass");
}

/// The session runs one generation: a second `prefill` is refused with `InvalidInput`
/// and leaves the cache position and the logits as the first prefill left them.
pub(crate) fn assert_second_prefill_is_refused(
    weights: StandaloneWeights<'_>,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    rope: &RopeTable,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) {
    let plan = plan_for(cfg, tokenizer, rope, prompt, gen_cfg);
    let prompt_len = plan.prompt_len;
    let vocab = cfg.vocab_size;
    let mut session = StandaloneCpuSession::new(weights, cfg, rope, plan);

    let never = || false;
    session.prefill(&never).expect("first prefill");
    assert_eq!(session.kv_cache.seq_len, prompt_len, "first prefill length");
    let logits: Vec<u32> = session.scratch.logits[..vocab]
        .iter()
        .map(|logit| logit.to_bits())
        .collect();

    match session.prefill(&never) {
        Err(InferenceError::InvalidInput(message)) => assert_eq!(
            message, "this session runs one generation; build a new session",
            "refusal text"
        ),
        Err(other) => panic!("a second prefill was refused with the wrong error: {other:?}"),
        Ok(_) => panic!("a second prefill was accepted"),
    }
    assert_eq!(
        session.kv_cache.seq_len, prompt_len,
        "the refused prefill moved the cache"
    );
    let after: Vec<u32> = session.scratch.logits[..vocab]
        .iter()
        .map(|logit| logit.to_bits())
        .collect();
    assert_eq!(after, logits, "the refused prefill ran a forward pass");
}
