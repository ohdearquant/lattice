use super::*;
use crate::grammar::{GrammarEngine, GrammarSpec};
use std::collections::HashMap;
use std::sync::Arc;

fn hollow_f16() -> F16ModelWeights {
    F16ModelWeights {
        embed_tokens: vec![],
        final_norm: vec![],
        layers: vec![],
    }
}

fn hollow_q8() -> Q8ModelWeights {
    Q8ModelWeights {
        embed_tokens: vec![],
        final_norm: vec![],
        layers: vec![],
    }
}

fn hollow_neon() -> Q8NeonModel {
    Q8NeonModel {
        embed_tokens: vec![],
        final_norm: vec![],
        lm_head_packed: vec![],
        lm_head_rows: 0,
        lm_head_cols: 0,
        layers: vec![],
    }
}

fn tokenizer() -> BpeTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    BpeTokenizer::from_vocab_and_merges(vocab, merges).expect("test tokenizer builds")
}

fn plan(cfg: &Qwen35Config, rope: &RopeTable, max_new_tokens: usize) -> GenerationPlan {
    let gen_cfg = GenerateConfig {
        max_new_tokens,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    match prepare_generation(
        &tokenizer(),
        "hello",
        &gen_cfg,
        cfg.vocab_size,
        rope.max_positions(),
        GenerationEntryContract::StandaloneCpu,
    ) {
        Ok(GenerationPreparation::Ready(plan)) => plan,
        other => panic!("the request must prepare a session: {other:?}"),
    }
}

fn with_each_hollow_session(check: impl Fn(&str, StandaloneCpuSession<'_>)) {
    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 64, cfg.rope_theta);
    let f16 = hollow_f16();
    let q8 = hollow_q8();
    let neon = hollow_neon();
    for (name, weights) in [
        ("f16", StandaloneWeights::F16(&f16)),
        ("q8", StandaloneWeights::Q8(&q8)),
        ("q8_neon", StandaloneWeights::Q8Neon(&neon)),
    ] {
        let session = StandaloneCpuSession::new(weights, &cfg, &rope, plan(&cfg, &rope, 8));
        check(name, session);
    }
}

#[test]
fn session_declares_none_of_the_optional_controls() {
    with_each_hollow_session(|name, session| {
        let caps = session.capabilities();
        assert_eq!(*caps, ExecutionCapabilities::default(), "{name}");
        assert!(!caps.grammar, "{name}: grammar");
        assert!(!caps.logprobs, "{name}: logprobs");
        assert!(!caps.stop_strings, "{name}: stop_strings");
        assert!(!caps.reasoning_budget, "{name}: reasoning_budget");
    });
}

/// Each control is refused by the driver's capability check, which runs before the first
/// session call, so a session over empty weights is enough. The message text identifies the
/// capability check as the refusing mechanism: the wrappers' own refusal wording differs.
#[test]
fn driver_refuses_every_unsupported_control_on_the_standalone_profile() {
    let grammar_spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
    let engine = GrammarEngine::new(&grammar_spec, vec![b"t".to_vec(), b"f".to_vec()])
        .expect("trivial grammar compiles");

    let controls: Vec<(&str, GenerateConfig, &str)> = vec![
        (
            "grammar",
            GenerateConfig {
                grammar: Some(Arc::new(engine)),
                ..GenerateConfig::default()
            },
            "session does not declare grammar support but gen_cfg.grammar is set",
        ),
        (
            "logprobs",
            GenerateConfig {
                logprobs: Some(0),
                ..GenerateConfig::default()
            },
            "session does not declare logprobs support but gen_cfg.logprobs is set",
        ),
        (
            "stop_strings",
            GenerateConfig {
                stop_strings: vec!["</s>".to_string()],
                ..GenerateConfig::default()
            },
            "session does not declare stop_strings support but gen_cfg.stop_strings is set",
        ),
        (
            "reasoning_budget",
            GenerateConfig {
                reasoning_budget: Some(16),
                ..GenerateConfig::default()
            },
            "session does not declare reasoning_budget support but \
                 gen_cfg.reasoning_budget is set",
        ),
    ];

    with_each_hollow_session(|wrapper, mut session| {
        for (control, gen_cfg, message) in &controls {
            let never_cancel = || false;
            let mut text = String::new();
            let mut offsets: Vec<usize> = Vec::new();
            let result = driver::run(
                &mut session,
                gen_cfg,
                None,
                &[0, 1, 2],
                999_999,
                false,
                &never_cancel,
                |_| {},
                |_| String::new(),
                &mut text,
                &mut offsets,
                |_, _| true,
                || {},
                String::new,
            );
            match result {
                Err(InferenceError::InvalidInput(actual)) => {
                    assert_eq!(&actual, message, "{wrapper}: {control}")
                }
                Err(other) => panic!("{wrapper}: {control}: wrong error {other:?}"),
                Ok(_) => panic!("{wrapper}: {control}: the driver accepted the control"),
            }
        }
    });
}

#[test]
fn neon_session_reserves_kv_for_prompt_plus_budget() {
    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 64, cfg.rope_theta);
    let neon = hollow_neon();
    let plan = plan(&cfg, &rope, 8);
    let required = plan.prompt_len + 8 + 1;
    assert_eq!(plan.required_capacity, required, "plan capacity");
    assert!(
        cfg.num_full_attention_layers() > 0,
        "the fixture has no full-attention layer, so there is no cache to reserve"
    );

    let session = StandaloneCpuSession::new(StandaloneWeights::Q8Neon(&neon), &cfg, &rope, plan);

    let wanted = required * cfg.full_kv_dim();
    for (layer, (k, v)) in session
        .kv_cache
        .k
        .iter()
        .zip(&session.kv_cache.v)
        .enumerate()
    {
        assert!(k.capacity() >= wanted, "layer {layer}: k capacity");
        assert!(v.capacity() >= wanted, "layer {layer}: v capacity");
    }
    assert!(
        session.scratch.scores.len() >= cfg.num_attention_heads * (required + 1),
        "scratch was not sized for the planned capacity"
    );
    assert_eq!(
        session.q8_scratch.storage_snapshot(),
        Q8MatmulScratch::default().storage_snapshot()
    );
    assert_eq!(session.q8_scratch.call_count(), 0);
}

#[test]
fn q8_session_does_not_prereserve_kv() {
    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 64, cfg.rope_theta);
    let q8 = hollow_q8();
    assert!(cfg.num_full_attention_layers() > 0);

    let session = StandaloneCpuSession::new(
        StandaloneWeights::Q8(&q8),
        &cfg,
        &rope,
        plan(&cfg, &rope, 8),
    );

    assert!(session.kv_cache.k.iter().all(|k| k.capacity() == 0));
    assert!(session.kv_cache.v.iter().all(|v| v.capacity() == 0));
    assert!(session.scratch.scores.is_empty());
    assert_eq!(
        session.q8_scratch.storage_snapshot(),
        Q8MatmulScratch::default().storage_snapshot()
    );
    assert_eq!(session.q8_scratch.call_count(), 0);
}

#[test]
fn f16_session_does_not_prereserve_kv() {
    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 64, cfg.rope_theta);
    let f16 = hollow_f16();
    assert!(cfg.num_full_attention_layers() > 0);

    let session = StandaloneCpuSession::new(
        StandaloneWeights::F16(&f16),
        &cfg,
        &rope,
        plan(&cfg, &rope, 8),
    );

    assert!(session.kv_cache.k.iter().all(|k| k.capacity() == 0));
    assert!(session.kv_cache.v.iter().all(|v| v.capacity() == 0));
    assert!(session.scratch.scores.is_empty());
    assert_eq!(
        session.q8_scratch.storage_snapshot(),
        Q8MatmulScratch::default().storage_snapshot()
    );
    assert_eq!(session.q8_scratch.call_count(), 0);
}
