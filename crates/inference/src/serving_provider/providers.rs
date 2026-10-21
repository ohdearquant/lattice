//! Registered serving providers.

pub mod gemma;
pub mod qwen;

use super::RegisteredProvider;

pub(super) static QWEN_PROVIDER: qwen::QwenProvider = qwen::QwenProvider;
pub(super) static GEMMA_PROVIDER: gemma::GemmaProvider = gemma::GemmaProvider;

pub(super) static PROVIDERS: [&dyn RegisteredProvider; 2] = [&QWEN_PROVIDER, &GEMMA_PROVIDER];
