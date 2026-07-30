use super::generation::{
    check_grammar_not_set, check_logprobs_not_set, check_prompt_ids_in_vocab,
    check_prompt_not_empty, check_reasoning_budget_not_set, check_stop_strings_not_set,
};
use crate::error::InferenceError;
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput};
use crate::stop_reason::StopReason;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenerationEntryContract {
    StandaloneCpu,
}

impl GenerationEntryContract {
    fn validate(self, gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
        match self {
            Self::StandaloneCpu => {
                check_grammar_not_set(gen_cfg)?;
                check_logprobs_not_set(gen_cfg)?;
                check_stop_strings_not_set(gen_cfg)?;
                check_reasoning_budget_not_set(gen_cfg)
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct GenerationPlan {
    pub(crate) rng_state: u64,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) prompt_len: usize,
    pub(crate) required_capacity: usize,
}

#[derive(Debug)]
pub(crate) enum GenerationPreparation {
    Ready(GenerationPlan),
    Complete(GenerateOutput),
}

pub(crate) fn prepare_generation(
    tokenizer: &BpeTokenizer,
    prompt: &str,
    gen_cfg: &GenerateConfig,
    vocab_size: usize,
    max_context: usize,
    contract: GenerationEntryContract,
) -> Result<GenerationPreparation, InferenceError> {
    let rng_state = normalize_seed(gen_cfg.seed, system_seed);

    let input = tokenizer.tokenize(prompt);
    let prompt_ids = input.input_ids[..input.real_length].to_vec();
    let prompt_len = prompt_ids.len();

    check_prompt_not_empty(prompt_len)?;
    check_prompt_ids_in_vocab(&prompt_ids, vocab_size)?;

    if gen_cfg.max_new_tokens == 0 {
        return Ok(GenerationPreparation::Complete(GenerateOutput {
            text: String::new(),
            token_ids: Vec::new(),
            prompt_tokens: prompt_len,
            generated_tokens: 0,
            stopped: false,
            stop_reason: Some(StopReason::Length),
            token_logprobs: Vec::new(),
        }));
    }

    contract.validate(gen_cfg)?;

    if prompt_len.saturating_add(gen_cfg.max_new_tokens) > max_context {
        return Err(InferenceError::Inference(format!(
            "prompt ({prompt_len} tokens) plus max_new_tokens ({}) exceeds \
             model context window ({max_context})",
            gen_cfg.max_new_tokens
        )));
    }

    Ok(GenerationPreparation::Ready(GenerationPlan {
        rng_state,
        prompt_ids,
        prompt_len,
        required_capacity: prompt_len
            .saturating_add(gen_cfg.max_new_tokens)
            .saturating_add(1),
    }))
}

fn normalize_seed(seed: Option<u64>, fallback: impl FnOnce() -> u64) -> u64 {
    let seed = seed.unwrap_or_else(fallback);
    if seed == 0 { 1 } else { seed }
}

fn system_seed() -> u64 {
    use std::time::SystemTime;

    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x12345678_9abcdef0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn tokenizer(entries: &[(&str, u32)]) -> BpeTokenizer {
        let vocab = entries
            .iter()
            .map(|(token, id)| ((*token).to_string(), *id))
            .collect::<HashMap<_, _>>();
        BpeTokenizer::from_vocab_and_merges(vocab, Vec::new()).expect("test tokenizer constructs")
    }

    fn prepare(
        tokenizer: &BpeTokenizer,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        vocab_size: usize,
        max_context: usize,
    ) -> Result<GenerationPreparation, InferenceError> {
        prepare_generation(
            tokenizer,
            prompt,
            gen_cfg,
            vocab_size,
            max_context,
            GenerationEntryContract::StandaloneCpu,
        )
    }

    #[test]
    fn seed_normalization_preserves_nonzero_and_repairs_zero() {
        assert_eq!(
            normalize_seed(Some(7), || panic!("fallback must not run")),
            7
        );
        assert_eq!(
            normalize_seed(Some(0), || panic!("fallback must not run")),
            1
        );
        assert_eq!(normalize_seed(None, || 0), 1);
        assert_eq!(normalize_seed(None, || 9), 9);
    }

    #[test]
    fn preparation_applies_seed_normalization_to_the_returned_plan() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            seed: Some(0),
            ..Default::default()
        };
        let result = prepare(&tokenizer, "a", &gen_cfg, 1, 2).expect("preparation succeeds");
        let GenerationPreparation::Ready(plan) = result else {
            panic!("nonzero budget must return a ready plan");
        };

        assert_eq!(plan.rng_state, 1);
    }

    #[test]
    fn ready_plan_owns_prompt_seed_and_capacity_and_preserves_mtp_contract() {
        let tokenizer = tokenizer(&[("a", 0), ("b", 1)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 3,
            seed: Some(17),
            enable_mtp: Some(true),
            ..Default::default()
        };
        let result = prepare(&tokenizer, "ab", &gen_cfg, 2, 5).expect("preparation succeeds");
        let GenerationPreparation::Ready(plan) = result else {
            panic!("nonzero budget must return a ready plan");
        };

        assert_eq!(plan.rng_state, 17);
        assert_eq!(plan.prompt_ids, vec![0, 1]);
        assert_eq!(plan.prompt_len, 2);
        assert_eq!(plan.required_capacity, 6);
    }

    #[test]
    fn empty_prompt_precedes_zero_budget_and_capability_checks() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 0,
            stop_strings: vec!["stop".to_string()],
            ..Default::default()
        };
        let err = prepare(&tokenizer, "", &gen_cfg, 1, 0)
            .expect_err("empty prompt must reject before every later branch");

        assert!(matches!(
            err,
            InferenceError::Inference(ref message) if message == "empty prompt"
        ));
    }

    #[test]
    fn prompt_id_admission_precedes_zero_budget() {
        let tokenizer = tokenizer(&[("z", 2)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 0,
            ..Default::default()
        };
        let err = prepare(&tokenizer, "z", &gen_cfg, 2, usize::MAX)
            .expect_err("out-of-vocabulary prompt must reject before zero-budget completion");

        assert!(matches!(err, InferenceError::InvalidInput(_)));
    }

    #[test]
    fn zero_budget_precedes_capabilities_and_context() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 0,
            stop_strings: vec!["stop".to_string()],
            ..Default::default()
        };
        let result = prepare(&tokenizer, "a", &gen_cfg, 1, 0)
            .expect("zero budget must complete before unsupported features and context");
        let GenerationPreparation::Complete(output) = result else {
            panic!("zero budget must return an early completion");
        };

        assert!(output.text.is_empty());
        assert!(output.token_ids.is_empty());
        assert_eq!(output.prompt_tokens, 1);
        assert_eq!(output.generated_tokens, 0);
        assert!(!output.stopped);
        assert_eq!(output.stop_reason, Some(StopReason::Length));
        assert!(output.token_logprobs.is_empty());
    }

    #[test]
    fn standalone_cpu_contract_rejects_every_unwired_feature() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let grammar = GrammarEngine::new(
            &GrammarSpec::Gbnf("root ::= \"a\"\n".to_string()),
            vec![b"a".to_vec()],
        )
        .expect("test grammar compiles");
        let configs = [
            GenerateConfig {
                grammar: Some(Arc::new(grammar)),
                ..Default::default()
            },
            GenerateConfig {
                logprobs: Some(0),
                ..Default::default()
            },
            GenerateConfig {
                stop_strings: vec!["stop".to_string()],
                ..Default::default()
            },
            GenerateConfig {
                reasoning_budget: Some(1),
                ..Default::default()
            },
        ];

        for gen_cfg in &configs {
            let err = prepare(&tokenizer, "a", gen_cfg, 1, 0)
                .expect_err("standalone CPU contract must reject unwired features");
            assert!(matches!(err, InferenceError::InvalidInput(_)));
        }
    }

    #[test]
    fn context_preflight_preserves_the_standalone_cpu_error() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 2,
            ..Default::default()
        };
        let err = prepare(&tokenizer, "a", &gen_cfg, 1, 2)
            .expect_err("prompt plus decode budget must fit the context");

        assert!(matches!(
            err,
            InferenceError::Inference(ref message)
                if message
                    == "prompt (1 tokens) plus max_new_tokens (2) exceeds model context window (2)"
        ));
    }
}
