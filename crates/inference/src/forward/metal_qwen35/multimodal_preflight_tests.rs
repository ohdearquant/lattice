use super::multimodal_generate_preflight;
use crate::error::InferenceError;
use crate::generation::GenerateConfig;
use crate::grammar::{GrammarEngine, GrammarSpec};
use std::sync::Arc;

/// Grammar-constrained decoding is not wired into the multimodal path; the
/// preflight must reject a config with `grammar` set.
///
/// This test drives `multimodal_generate_preflight` — the exact function that
/// `generate_multimodal` calls as its first action — so mutating the guard
/// logic in that function directly fails this test.
///
/// Mutation sensitivity: change `multimodal_generate_preflight` to always
/// return `Ok(())` → `result.is_err()` below fails, catching the regression.
#[test]
fn generate_multimodal_grammar_guard_rejects_grammar_config() {
    let spec = GrammarSpec::Gbnf("root ::= \"a\" | \"b\"\n".to_string());
    let vocab = vec![b"a".to_vec(), b"b".to_vec()];
    let engine = GrammarEngine::new(&spec, vocab).expect("trivial grammar must compile");

    let cfg_with_grammar = GenerateConfig {
        grammar: Some(Arc::new(engine)),
        ..Default::default()
    };

    let result = multimodal_generate_preflight(&cfg_with_grammar);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "multimodal preflight must return InvalidInput when grammar is set; got {result:?}"
    );
}

/// A config with `grammar` unset must pass through the preflight without error.
#[test]
fn generate_multimodal_grammar_guard_allows_no_grammar() {
    assert!(
        multimodal_generate_preflight(&GenerateConfig::default()).is_ok(),
        "grammar = None must not trigger the preflight guard"
    );
}

/// Per-token logprob capture (#585) is not wired into the multimodal path;
/// the preflight must reject a config with `logprobs` set rather than
/// silently returning an output with an empty `token_logprobs`.
///
/// Mutation sensitivity: change `multimodal_generate_preflight` to always
/// return `Ok(())` → `result.is_err()` below fails, catching the regression.
#[test]
fn generate_multimodal_logprobs_guard_rejects_logprobs_config() {
    let cfg_with_logprobs = GenerateConfig {
        logprobs: Some(5),
        ..Default::default()
    };

    let result = multimodal_generate_preflight(&cfg_with_logprobs);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "multimodal preflight must return InvalidInput when logprobs is set; got {result:?}"
    );
}

/// A config with `logprobs` unset must pass through the preflight without error.
#[test]
fn generate_multimodal_logprobs_guard_allows_no_logprobs() {
    assert!(
        multimodal_generate_preflight(&GenerateConfig::default()).is_ok(),
        "logprobs = None must not trigger the preflight guard"
    );
}

/// Reasoning-budget forcing (ADR-080 C3, #783) is not wired into the
/// multimodal path; the preflight must reject a config with
/// `reasoning_budget` set rather than silently returning an unbudgeted
/// output.
///
/// Mutation sensitivity: change `multimodal_generate_preflight` to always
/// return `Ok(())` (or drop the new `check_reasoning_budget_not_set` call)
/// → `result.is_err()` below fails, catching the regression.
#[test]
fn generate_multimodal_reasoning_budget_guard_rejects_budget_config() {
    let cfg_with_budget = GenerateConfig {
        reasoning_budget: Some(32),
        ..Default::default()
    };

    let result = multimodal_generate_preflight(&cfg_with_budget);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "multimodal preflight must return InvalidInput when reasoning_budget is set; \
         got {result:?}"
    );
}

/// A config with `reasoning_budget` unset must pass through the preflight
/// without error.
#[test]
fn generate_multimodal_reasoning_budget_guard_allows_no_budget() {
    assert!(
        multimodal_generate_preflight(&GenerateConfig::default()).is_ok(),
        "reasoning_budget = None must not trigger the preflight guard"
    );
}
