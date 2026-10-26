//! Decoder sessions (ADR-090 D1/D2).
//!
//! The session contract (the `DecoderSession` trait, the prediction ledger, the per-step request
//! and result types) and the shared driver live in `lattice_arch::decoder`, which every model
//! family can reach; the speculative extension stays behind `lattice_arch::__internal`. This
//! module re-exports them at their previous paths and holds the concrete sessions, which move
//! to their family crates later. `model::qwen35::generation`'s
//! `generate()`/`generate_with_trace()` are the driver's production callers.

/// First concrete [`DecoderSession`] implementation (ADR-090 row B): the CPU dense
/// Qwen3.5 entry point wrapped over this module's typed vocabulary.
pub(crate) mod qwen_cpu;

/// The ordinary Qwen3.5 Metal generation path as a concrete [`DecoderSession`] (ADR-090
/// row R06).
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) mod qwen_metal;

/// Measurement-only seams for the per-token allocation instrument (ADR-090 D7, row R01).
#[cfg(feature = "bench-internals")]
pub mod bench_support;

/// One [`DecoderSession`] for the three standalone CPU generation wrappers
/// (`generate_f16`, `generate_q8`, `generate_q8_neon`), run by the shared driver.
pub(crate) mod standalone_cpu;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use lattice_arch::__internal::decoder::{
    SpeculativeSession, SpeculativeTrace, VerifiedRound,
};
pub(crate) use lattice_arch::decoder::{
    AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
    MetadataRequest, PredictionError, PredictionId, PredictionLedger, SelectOutcome,
    SelectionCandidate, SelectionRequest, StepStamp, TokenMetadata,
};

/// The autoregressive driver (ADR-090 row C), defined in `lattice-arch`.
pub(crate) mod driver {
    #[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
    pub(crate) use counted::{run, run_into, test_driver_run_count};
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub(crate) use lattice_arch::__internal::decoder::run_speculative;
    #[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
    pub(crate) use lattice_arch::decoder::driver::DriverResult;
    pub(crate) use lattice_arch::decoder::driver::DriverTrace;
    #[cfg(not(all(test, target_os = "macos", feature = "metal-gpu")))]
    pub(crate) use lattice_arch::decoder::driver::{run, run_into};

    /// Test builds with Metal count driver runs, so the speculative-route tests can assert that
    /// a request reached the driver. Each wrapper counts once and delegates unchanged.
    #[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
    mod counted {
        use super::DriverResult;
        use crate::decoder::{Cancellation, DecoderSession};
        use crate::error::InferenceError;
        use crate::generation::GenerateConfig;
        use lattice_arch::decoder::driver as arch_driver;
        use std::cell::Cell;
        use std::sync::Arc;

        thread_local! {
            static TEST_DRIVER_RUN_COUNT: Cell<usize> = const { Cell::new(0) };
        }

        pub(crate) fn test_driver_run_count() -> usize {
            TEST_DRIVER_RUN_COUNT.with(Cell::get)
        }

        pub(crate) fn run(
            session: &mut dyn DecoderSession,
            gen_cfg: &GenerateConfig,
            think_close_id: Option<u32>,
            reasoning_newline_boundary: Option<Arc<[bool]>>,
            prompt_ids: &[u32],
            eos_token_id: u32,
            streaming: bool,
            cancel: &dyn Cancellation,
            on_push: impl FnMut(usize),
            decode_delta: impl FnMut(u32) -> String,
            text: &mut String,
            token_logprob_end_offsets: &mut Vec<usize>,
            emit_confirmed: impl FnMut(&str, u32) -> bool,
            on_prefill_end: impl FnMut(),
            finish_tail: impl FnOnce() -> String,
        ) -> Result<DriverResult, InferenceError> {
            TEST_DRIVER_RUN_COUNT.with(|count| count.set(count.get() + 1));
            arch_driver::run(
                session,
                gen_cfg,
                think_close_id,
                reasoning_newline_boundary,
                prompt_ids,
                eos_token_id,
                streaming,
                cancel,
                on_push,
                decode_delta,
                text,
                token_logprob_end_offsets,
                emit_confirmed,
                on_prefill_end,
                finish_tail,
            )
        }

        pub(crate) fn run_into(
            session: &mut dyn DecoderSession,
            gen_cfg: &GenerateConfig,
            think_close_id: Option<u32>,
            reasoning_newline_boundary: Option<Arc<[bool]>>,
            prompt_ids: &[u32],
            eos_token_id: u32,
            streaming: bool,
            cancel: &dyn Cancellation,
            on_push: impl FnMut(usize),
            decode_delta: impl FnMut(u32, &mut String),
            text: &mut String,
            token_logprob_end_offsets: &mut Vec<usize>,
            emit_confirmed: impl FnMut(&str, u32) -> bool,
            on_prefill_end: impl FnMut(),
            finish_tail: impl FnOnce() -> String,
        ) -> Result<DriverResult, InferenceError> {
            TEST_DRIVER_RUN_COUNT.with(|count| count.set(count.get() + 1));
            arch_driver::run_into(
                session,
                gen_cfg,
                think_close_id,
                reasoning_newline_boundary,
                prompt_ids,
                eos_token_id,
                streaming,
                cancel,
                on_push,
                decode_delta,
                text,
                token_logprob_end_offsets,
                emit_confirmed,
                on_prefill_end,
                finish_tail,
            )
        }
    }
}
