#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use super::generation::check_context_budget;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use super::generation::check_mtp_not_requested;
use super::generation::{
    check_grammar_not_set, check_logprobs_not_set, check_prompt_ids_in_vocab,
    check_prompt_not_empty, check_reasoning_budget_not_set, check_stop_strings_not_set,
};
use crate::error::InferenceError;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use crate::model::qwen35_config::decode_cap;
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput};
use crate::stop_reason::StopReason;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenerationEntryContract {
    StandaloneCpu,
    #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
    MetalDirect,
    #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
    MetalStreaming,
    /// Cross-turn prefix-cache streaming (#1354). Unlike every other
    /// variant, its capability guards (`logprobs`, `enable_mtp`) must run
    /// *before* tokenization: a request that violates one of these and is
    /// also empty must still surface the capability error, matching the
    /// order the public wrapper ran these checks in prior to this contract
    /// existing. See [`Self::validate_before_tokenization`].
    #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
    MetalPrefixCacheStreaming,
}

impl GenerationEntryContract {
    /// Guards that must be evaluated before the prompt is tokenized, so
    /// their errors take precedence over `check_prompt_not_empty` and every
    /// later step for a request that violates more than one guard at once.
    /// Every variant except [`Self::MetalPrefixCacheStreaming`] has no
    /// pre-tokenization guards and returns `Ok(())` unconditionally, leaving
    /// its existing step order (tokenize first, capabilities checked in
    /// [`Self::validate_capabilities`] afterward) exactly as it was.
    fn validate_before_tokenization(self, gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
        // Only `MetalPrefixCacheStreaming` (test/metal-gpu-gated) reads
        // `gen_cfg`; every other variant's arm is unconditionally `Ok(())`.
        let _ = gen_cfg;
        match self {
            Self::StandaloneCpu => Ok(()),
            #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
            Self::MetalDirect | Self::MetalStreaming => Ok(()),
            #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
            Self::MetalPrefixCacheStreaming => {
                check_logprobs_not_set(gen_cfg)?;
                check_mtp_not_requested(gen_cfg)
            }
        }
    }

    fn validate_prompt_ids(
        self,
        prompt_ids: &[u32],
        vocab_size: usize,
    ) -> Result<(), InferenceError> {
        match self {
            Self::StandaloneCpu => check_prompt_ids_in_vocab(prompt_ids, vocab_size),
            #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
            Self::MetalDirect | Self::MetalStreaming => Ok(()),
            #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
            Self::MetalPrefixCacheStreaming => Ok(()),
        }
    }

    fn validate_capabilities(self, gen_cfg: &GenerateConfig) -> Result<(), InferenceError> {
        match self {
            Self::StandaloneCpu => {
                check_grammar_not_set(gen_cfg)?;
                check_logprobs_not_set(gen_cfg)?;
                check_stop_strings_not_set(gen_cfg)?;
                check_reasoning_budget_not_set(gen_cfg)
            }
            #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
            Self::MetalDirect => {
                check_reasoning_budget_not_set(gen_cfg)?;
                check_logprobs_not_set(gen_cfg)
            }
            #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
            Self::MetalStreaming => Ok(()),
            // Both capability guards this variant carries already ran in
            // `validate_before_tokenization`, ahead of tokenization.
            #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
            Self::MetalPrefixCacheStreaming => Ok(()),
        }
    }

    fn validate_context(
        self,
        prompt_len: usize,
        gen_cfg: &GenerateConfig,
        max_context: usize,
    ) -> Result<usize, InferenceError> {
        match self {
            Self::StandaloneCpu => {
                if prompt_len.saturating_add(gen_cfg.max_new_tokens) > max_context {
                    return Err(InferenceError::Inference(format!(
                        "prompt ({prompt_len} tokens) plus max_new_tokens ({}) exceeds \
                         model context window ({max_context})",
                        gen_cfg.max_new_tokens
                    )));
                }
                Ok(gen_cfg.max_new_tokens)
            }
            #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
            Self::MetalDirect | Self::MetalStreaming | Self::MetalPrefixCacheStreaming => {
                check_context_budget(
                    prompt_len,
                    gen_cfg.reasoning_budget,
                    gen_cfg.max_new_tokens,
                    max_context,
                )?;
                Ok(decode_cap(gen_cfg.reasoning_budget, gen_cfg.max_new_tokens))
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

    contract.validate_before_tokenization(gen_cfg)?;

    let input = tokenizer.tokenize(prompt);
    let prompt_ids = input.input_ids[..input.real_length].to_vec();
    let prompt_len = prompt_ids.len();

    check_prompt_not_empty(prompt_len)?;
    contract.validate_prompt_ids(&prompt_ids, vocab_size)?;

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

    contract.validate_capabilities(gen_cfg)?;
    let effective_capacity = contract.validate_context(prompt_len, gen_cfg, max_context)?;

    Ok(GenerationPreparation::Ready(GenerationPlan {
        rng_state,
        prompt_ids,
        prompt_len,
        required_capacity: prompt_len
            .saturating_add(effective_capacity)
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

    fn prepare_with_contract(
        tokenizer: &BpeTokenizer,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        vocab_size: usize,
        max_context: usize,
        contract: GenerationEntryContract,
    ) -> Result<GenerationPreparation, InferenceError> {
        prepare_generation(
            tokenizer,
            prompt,
            gen_cfg,
            vocab_size,
            max_context,
            contract,
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
    fn metal_direct_contract_preserves_its_capability_set_and_guard_order() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let grammar = GrammarEngine::new(
            &GrammarSpec::Gbnf("root ::= \"a\"\n".to_string()),
            vec![b"a".to_vec()],
        )
        .expect("test grammar compiles");
        let supported = GenerateConfig {
            max_new_tokens: 1,
            grammar: Some(Arc::new(grammar)),
            stop_strings: vec!["stop".to_string()],
            enable_mtp: Some(true),
            ..Default::default()
        };
        let result = prepare_with_contract(
            &tokenizer,
            "a",
            &supported,
            1,
            2,
            GenerationEntryContract::MetalDirect,
        )
        .expect("Metal direct supports grammar, stop strings, and MTP routing");
        assert!(matches!(result, GenerationPreparation::Ready(_)));

        let rejected = GenerateConfig {
            max_new_tokens: 1,
            reasoning_budget: Some(1),
            logprobs: Some(0),
            ..Default::default()
        };
        let err = prepare_with_contract(
            &tokenizer,
            "a",
            &rejected,
            1,
            usize::MAX,
            GenerationEntryContract::MetalDirect,
        )
        .expect_err("Metal direct must reject reasoning before logprobs");
        assert!(matches!(
            err,
            InferenceError::InvalidInput(ref message)
                if message.starts_with("reasoning_budget is not yet supported")
        ));
    }

    #[test]
    fn metal_streaming_contract_accepts_its_wired_features() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let grammar = GrammarEngine::new(
            &GrammarSpec::Gbnf("root ::= \"a\"\n".to_string()),
            vec![b"a".to_vec()],
        )
        .expect("test grammar compiles");
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            grammar: Some(Arc::new(grammar)),
            logprobs: Some(0),
            stop_strings: vec!["stop".to_string()],
            reasoning_budget: Some(1),
            enable_mtp: Some(true),
            ..Default::default()
        };
        let result = prepare_with_contract(
            &tokenizer,
            "a",
            &gen_cfg,
            1,
            4,
            GenerationEntryContract::MetalStreaming,
        )
        .expect("Metal streaming supports the wired feature set and leaves MTP inert");
        let GenerationPreparation::Ready(plan) = result else {
            panic!("nonzero budget must return a ready plan");
        };

        assert_eq!(plan.required_capacity, 5);
    }

    #[test]
    fn metal_contracts_preserve_no_prompt_id_admission() {
        let tokenizer = tokenizer(&[("z", 2)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };

        for contract in [
            GenerationEntryContract::MetalDirect,
            GenerationEntryContract::MetalStreaming,
            GenerationEntryContract::MetalPrefixCacheStreaming,
        ] {
            let result = prepare_with_contract(&tokenizer, "z", &gen_cfg, 2, 2, contract)
                .expect("Metal preparation must preserve its existing admission ordering");
            assert!(matches!(result, GenerationPreparation::Ready(_)));
        }
    }

    #[test]
    fn metal_prefix_cache_streaming_contract_accepts_unset_capabilities() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            ..Default::default()
        };
        let result = prepare_with_contract(
            &tokenizer,
            "a",
            &gen_cfg,
            1,
            2,
            GenerationEntryContract::MetalPrefixCacheStreaming,
        )
        .expect("prefix-cache streaming must accept a request with no capabilities set");
        assert!(matches!(result, GenerationPreparation::Ready(_)));
    }

    /// #1354: `logprobs`/`enable_mtp` must be rejected before tokenization
    /// discovers the prompt is empty, so a request violating both a
    /// capability guard and the empty-prompt guard still returns the
    /// capability error -- exactly the order the public wrapper ran these
    /// checks in before this contract variant existed.
    ///
    /// Mutation sensitivity: moving `validate_before_tokenization`'s call
    /// site in `prepare_generation` to after tokenization (or folding these
    /// checks into `validate_capabilities` instead) flips both assertions
    /// below to `Err(Inference("empty prompt"))`.
    #[test]
    fn metal_prefix_cache_streaming_contract_rejects_capabilities_before_empty_prompt() {
        let tokenizer = tokenizer(&[("a", 0)]);

        let logprobs_and_empty = GenerateConfig {
            max_new_tokens: 1,
            logprobs: Some(0),
            ..Default::default()
        };
        let err = prepare_with_contract(
            &tokenizer,
            "",
            &logprobs_and_empty,
            1,
            2,
            GenerationEntryContract::MetalPrefixCacheStreaming,
        )
        .expect_err("logprobs set on an empty prompt must still reject");
        assert!(
            matches!(
                err,
                InferenceError::InvalidInput(ref message) if message.contains("logprobs")
            ),
            "expected the logprobs error ahead of the empty-prompt error; got {err:?}"
        );

        let mtp_and_empty = GenerateConfig {
            max_new_tokens: 1,
            enable_mtp: Some(true),
            ..Default::default()
        };
        let err = prepare_with_contract(
            &tokenizer,
            "",
            &mtp_and_empty,
            1,
            2,
            GenerationEntryContract::MetalPrefixCacheStreaming,
        )
        .expect_err("enable_mtp set on an empty prompt must still reject");
        assert!(
            matches!(
                err,
                InferenceError::InvalidInput(ref message) if message.contains("enable_mtp")
            ),
            "expected the MTP error ahead of the empty-prompt error; got {err:?}"
        );
    }

    /// #1354: when both capability guards are violated at once, `logprobs`
    /// must win because `validate_before_tokenization` checks it first --
    /// same order the public wrapper's two explicit preflight calls ran in.
    #[test]
    fn metal_prefix_cache_streaming_contract_rejects_logprobs_before_mtp() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            logprobs: Some(0),
            enable_mtp: Some(true),
            ..Default::default()
        };
        let err = prepare_with_contract(
            &tokenizer,
            "a",
            &gen_cfg,
            1,
            2,
            GenerationEntryContract::MetalPrefixCacheStreaming,
        )
        .expect_err("logprobs and enable_mtp set together must still reject");
        assert!(
            matches!(
                err,
                InferenceError::InvalidInput(ref message) if message.contains("logprobs")
            ),
            "expected the logprobs error (checked before MTP); got {err:?}"
        );
    }

    /// #1354: `logprobs`/`enable_mtp` must be rejected before the
    /// zero-budget short-circuit too, not only before the empty-prompt
    /// guard -- `validate_before_tokenization` runs unconditionally ahead
    /// of every later step in `prepare_generation`, zero-budget included.
    /// A `max_new_tokens: 0` request with `logprobs` set must therefore
    /// still surface the logprobs error, not an early `Complete` with an
    /// empty output.
    #[test]
    fn metal_prefix_cache_streaming_contract_rejects_capabilities_before_zero_budget() {
        let tokenizer = tokenizer(&[("a", 0)]);

        let logprobs_and_zero_budget = GenerateConfig {
            max_new_tokens: 0,
            logprobs: Some(0),
            ..Default::default()
        };
        let err = prepare_with_contract(
            &tokenizer,
            "a",
            &logprobs_and_zero_budget,
            1,
            2,
            GenerationEntryContract::MetalPrefixCacheStreaming,
        )
        .expect_err("logprobs set on a zero-budget request must still reject");
        assert!(
            matches!(
                err,
                InferenceError::InvalidInput(ref message) if message.contains("logprobs")
            ),
            "expected the logprobs error ahead of the zero-budget completion; got {err:?}"
        );

        let mtp_and_zero_budget = GenerateConfig {
            max_new_tokens: 0,
            enable_mtp: Some(true),
            ..Default::default()
        };
        let err = prepare_with_contract(
            &tokenizer,
            "a",
            &mtp_and_zero_budget,
            1,
            2,
            GenerationEntryContract::MetalPrefixCacheStreaming,
        )
        .expect_err("enable_mtp set on a zero-budget request must still reject");
        assert!(
            matches!(
                err,
                InferenceError::InvalidInput(ref message) if message.contains("enable_mtp")
            ),
            "expected the MTP error ahead of the zero-budget completion; got {err:?}"
        );
    }

    #[test]
    fn metal_streaming_context_uses_reasoning_aware_decode_cap() {
        let tokenizer = tokenizer(&[("a", 0)]);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 1,
            reasoning_budget: Some(1),
            ..Default::default()
        };
        let err = prepare_with_contract(
            &tokenizer,
            "a",
            &gen_cfg,
            1,
            3,
            GenerationEntryContract::MetalStreaming,
        )
        .expect_err("prompt plus reasoning-aware decode cap must fit the context");

        assert!(matches!(
            err,
            InferenceError::Inference(ref message)
                if message
                    == "prompt (1 tokens) plus effective decode cap (3 tokens; \
                        max_new_tokens=1, reasoning_budget=1) exceeds model context window (3)"
        ));
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
