//! Registered serving providers.

pub mod bert;
pub mod gemma;
pub mod inkling2;
pub mod qwen;

use super::RegisteredProvider;

pub(super) static QWEN_PROVIDER: qwen::QwenProvider = qwen::QwenProvider;
pub(super) static GEMMA_PROVIDER: gemma::GemmaProvider = gemma::GemmaProvider;
pub(super) static INKLING2_PROVIDER: inkling2::Inkling2Provider = inkling2::Inkling2Provider;
pub(super) static BERT_PROVIDER: bert::BertProvider = bert::BertProvider;

pub(super) static PROVIDERS: [&dyn RegisteredProvider; 4] = [
    &QWEN_PROVIDER,
    &GEMMA_PROVIDER,
    &INKLING2_PROVIDER,
    &BERT_PROVIDER,
];
