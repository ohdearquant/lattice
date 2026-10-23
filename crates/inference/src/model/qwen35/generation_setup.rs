#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use super::generation::check_context_budget;
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use super::generation::check_mtp_not_requested;
use super::generation::{
    check_grammar_not_set, check_logprobs_not_set, check_prompt_ids_in_vocab,
    check_prompt_not_empty, check_reasoning_budget_not_set, check_stop_strings_not_set,
};
use crate::error::InferenceError;
use crate::generation::{GenerateConfig, GenerateOutput};
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use crate::model::qwen35_config::decode_cap;
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
                    gen_cfg.effective_reasoning_budget(),
                    gen_cfg.max_new_tokens,
                    max_context,
                )?;
                Ok(decode_cap(
                    gen_cfg.effective_reasoning_budget(),
                    gen_cfg.max_new_tokens,
                ))
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
mod tests;
