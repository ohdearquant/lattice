//! Registered serving providers.

pub mod bert;
pub mod gemma;
pub mod qwen;

use super::RegisteredProvider;

pub(super) static QWEN_PROVIDER: qwen::QwenProvider = qwen::QwenProvider;
pub(super) static GEMMA_PROVIDER: gemma::GemmaProvider = gemma::GemmaProvider;
pub(super) static BERT_PROVIDER: bert::BertProvider = bert::BertProvider;

pub(super) static PROVIDERS: [&dyn RegisteredProvider; 3] =
    [&QWEN_PROVIDER, &GEMMA_PROVIDER, &BERT_PROVIDER];
