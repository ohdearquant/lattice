//! Worker-local execution for model serving.

mod gemma_cpu;
mod qwen_cpu;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod qwen_metal;

use crate::serving_host::ServedOutput;
use crate::serving_runtime_contract::{RuntimeInput, WorkerMetadata};
pub(crate) use crate::serving_runtime_contract::{WorkerFailure, cancelled_output};
use crate::serving_worker::{ScoreError, ScoreOutcome, ScoreReply, ScoreRow};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use gemma_cpu::GemmaCpuRuntime;
pub(crate) use gemma_cpu::GemmaPreparedCpuRuntime;
pub(crate) use qwen_cpu::QwenCpuRuntime;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use qwen_metal::QwenMetalRuntime;
use std::time::{Duration, Instant};

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::lora::{AdapterControlError, AdapterControlResult};
use crate::serve::lora::{AdapterIndex, LoraSelection, ResidencyLimits};
use crate::serve::prepare::PreparationHandle;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_worker::AdapterCommand;
use lattice_inference::forward::metal_qwen35::ChatMessage;
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

/// The execution state stays on the thread that built it.
pub(crate) trait ServingRuntime {
    /// `stream` is whether the HTTP response is streamed. It does not change
    /// what is generated; a runtime that reports its route per request
    /// (`serve::route`) names it there.
    // The Metal worker invokes this method after its factory builds the runtime.
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        cfg: &GenerateConfig,
        lora: &[LoraSelection],
        stream: bool,
        explicitly_thinking: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        self.execute(
            RuntimeInput::ChatMessages {
                messages,
                lora,
                explicitly_thinking,
            },
            cfg,
            stream,
            on_token,
            should_cancel,
        )
    }

    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure>;

    /// [`Self::execute`] with a second sink for the text of the reply's reasoning field. The CPU
    /// host calls this for every request. A runtime whose replies have no reasoning field keeps
    /// this default, which runs `execute` and reports no reasoning; a runtime that overrides it
    /// passes each reasoning piece to `on_reasoning`, in generated order, from the thread that
    /// passes content to `on_token`, and stops when either sink returns false.
    fn execute_with_reasoning(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        _on_reasoning: &mut dyn FnMut(&str) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<ServedOutput, WorkerFailure> {
        self.execute(input, cfg, http_stream, on_token, should_cancel)
            .map(ServedOutput::without_reasoning)
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError>;

    // The Metal worker reads this flag after its factory builds the runtime.
    #[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
    fn vision_supported(&self) -> Arc<AtomicBool>;

    /// Called once with every row of a scoring request before
    /// [`Self::score_option_rows`] scores them, so a runtime can prepare state
    /// the rows share. The rows may then be scored in several calls. The
    /// default prepares nothing. A runtime may ask `hung_up` while it
    /// prepares; `None` means the caller hung up and the preparation stopped.
    fn begin_option_rows(
        &mut self,
        rows: &[ScoreRow],
        hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<Result<(), String>> {
        let _ = (rows, hung_up);
        Some(Ok(()))
    }

    /// Score option rows: for each row, return the raw logits at the last
    /// position of its ids for its label ids, in order, as if the row were
    /// prefilled from a fresh state. A runtime may ask `hung_up` while it
    /// scores; `None` means the caller hung up and the rows were not all
    /// scored. A runtime without an option scorer refuses every request.
    fn score_option_rows(
        &mut self,
        rows: &[ScoreRow],
        hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<ScoreOutcome> {
        let _ = (rows, hung_up);
        Some(Err("this model does not score option rows".into()))
    }
}

/// How a scoring run ended.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ScoreEnd {
    /// Every row was scored, or the scorer failed.
    Scored(ScoreOutcome),
    /// The caller hung up, so nobody is waiting for a reply.
    HungUp,
    /// Scoring reached this time limit and was stopped.
    TimeLimit(Duration),
}

impl ScoreEnd {
    /// The reply to send, or `None` when the caller hung up.
    pub(crate) fn into_reply(self) -> Option<ScoreReply> {
        match self {
            Self::Scored(outcome) => Some(outcome.map_err(ScoreError::Failed)),
            Self::HungUp => None,
            Self::TimeLimit(limit) => Some(Err(ScoreError::TimeLimit(limit))),
        }
    }
}

/// Score `rows` on `runtime` the way every serving host does: prepare the
/// state the rows share once, then score one row at a time. The run stops when
/// the caller hangs up or once `time_limit`, counted from this call, has
/// passed. Both are checked before the preparation and before each row, and
/// the check is handed to the runtime, which may ask it inside either. A run
/// that stops is reported on standard error with the rows it scored.
pub(crate) fn score_rows_within(
    runtime: &mut dyn ServingRuntime,
    rows: &[ScoreRow],
    time_limit: Option<Duration>,
    hung_up: &mut dyn FnMut() -> bool,
) -> ScoreEnd {
    let started = Instant::now();
    let mut timed_out = false;
    let mut scored = ScoredRows::default();
    let outcome = score_rows_until_stopped(
        runtime,
        rows,
        &mut || {
            if hung_up() {
                return true;
            }
            if time_limit.is_some_and(|limit| started.elapsed() >= limit) {
                timed_out = true;
            }
            timed_out
        },
        &mut scored,
    );
    if let Some(outcome) = outcome {
        return ScoreEnd::Scored(outcome);
    }
    let elapsed = started.elapsed().as_secs_f64();
    let (rows_scored, total, tokens) = (scored.rows, rows.len(), scored.tokens);
    match time_limit.filter(|_| timed_out) {
        Some(limit) => {
            eprintln!(
                "[systemone] scoring stopped at its {}s time limit after {elapsed:.1}s: \
                 {rows_scored} of {total} rows scored, {tokens} prompt tokens in those rows",
                limit.as_secs()
            );
            ScoreEnd::TimeLimit(limit)
        }
        None => {
            eprintln!(
                "[systemone] the caller hung up after {elapsed:.1}s of scoring: \
                 {rows_scored} of {total} rows scored, {tokens} prompt tokens in those rows"
            );
            ScoreEnd::HungUp
        }
    }
}

/// The rows a run scored in full and the prompt tokens they hold.
#[derive(Debug, Default)]
struct ScoredRows {
    rows: usize,
    tokens: usize,
}

fn score_rows_until_stopped(
    runtime: &mut dyn ServingRuntime,
    rows: &[ScoreRow],
    stop: &mut dyn FnMut() -> bool,
    scored: &mut ScoredRows,
) -> Option<ScoreOutcome> {
    if stop() {
        return None;
    }
    if let Err(error) = runtime.begin_option_rows(rows, stop)? {
        return Some(Err(error));
    }
    let mut logits = Vec::with_capacity(rows.len());
    for row in rows {
        if stop() {
            return None;
        }
        match runtime.score_option_rows(std::slice::from_ref(row), stop)? {
            Ok(mut row_logits) => logits.append(&mut row_logits),
            Err(error) => return Some(Err(error)),
        }
        scored.rows += 1;
        scored.tokens += row.ids.len();
    }
    Some(Ok(logits))
}

/// A one-shot builder moved into the worker before any Metal state exists.
// The Metal worker calls this factory to create its runtime.
#[cfg_attr(not(all(target_os = "macos", feature = "metal-gpu")), allow(dead_code))]
pub(crate) trait RuntimeFactory: Send + 'static {
    fn build(
        self: Box<Self>,
        index: Arc<RwLock<AdapterIndex>>,
        limits: ResidencyLimits,
    ) -> Result<(Box<dyn ServingRuntime>, WorkerMetadata, PreparationHandle), String>;
}

trait AmbiguousIfSend<A> {
    fn some_item() {}
}
impl<T: ?Sized> AmbiguousIfSend<()> for T {}
#[allow(dead_code)]
struct IsSend;
impl<T: ?Sized + Send> AmbiguousIfSend<IsSend> for T {}
const _: fn() = || {
    let _ = <dyn ServingRuntime as AmbiguousIfSend<_>>::some_item;
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores each row as its label ids, waiting `per_row` on each.
    #[derive(Default)]
    struct SteppedRuntime {
        per_row: Duration,
        began: usize,
        scored: usize,
    }

    impl ServingRuntime for SteppedRuntime {
        fn execute(
            &mut self,
            _input: RuntimeInput<'_>,
            _cfg: &GenerateConfig,
            _http_stream: bool,
            _on_token: &mut dyn FnMut(&str, u32) -> bool,
            _should_cancel: &mut dyn FnMut() -> bool,
        ) -> Result<GenerateOutput, WorkerFailure> {
            Err(WorkerFailure::Failed(
                "no generation in this test".to_owned(),
            ))
        }

        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        fn control(
            &mut self,
            _command: AdapterCommand,
        ) -> Result<AdapterControlResult, AdapterControlError> {
            Err(AdapterControlError::InvalidAdapter(
                "no adapters in this test".to_owned(),
            ))
        }

        fn vision_supported(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }

        fn begin_option_rows(
            &mut self,
            _rows: &[ScoreRow],
            _hung_up: &mut dyn FnMut() -> bool,
        ) -> Option<Result<(), String>> {
            self.began += 1;
            Some(Ok(()))
        }

        fn score_option_rows(
            &mut self,
            rows: &[ScoreRow],
            _hung_up: &mut dyn FnMut() -> bool,
        ) -> Option<ScoreOutcome> {
            std::thread::sleep(self.per_row);
            self.scored += rows.len();
            Some(Ok(rows
                .iter()
                .map(|row| row.labels.iter().map(|&label| label as f32).collect())
                .collect()))
        }
    }

    fn rows(count: u32) -> Vec<ScoreRow> {
        (0..count)
            .map(|index| ScoreRow {
                ids: vec![1, 2, 3],
                labels: vec![index],
            })
            .collect()
    }

    #[test]
    fn without_a_limit_every_row_is_scored() {
        let mut runtime = SteppedRuntime::default();
        assert_eq!(
            score_rows_within(&mut runtime, &rows(3), None, &mut || false),
            ScoreEnd::Scored(Ok(vec![vec![0.0], vec![1.0], vec![2.0]]))
        );
        assert_eq!((runtime.began, runtime.scored), (1, 3));
    }

    #[test]
    fn a_spent_limit_stops_before_the_preparation() {
        let mut runtime = SteppedRuntime::default();
        assert_eq!(
            score_rows_within(&mut runtime, &rows(3), Some(Duration::ZERO), &mut || false),
            ScoreEnd::TimeLimit(Duration::ZERO)
        );
        assert_eq!((runtime.began, runtime.scored), (0, 0));
    }

    #[test]
    fn a_hang_up_is_reported_as_a_hang_up_even_past_the_limit() {
        let mut runtime = SteppedRuntime::default();
        assert_eq!(
            score_rows_within(&mut runtime, &rows(3), Some(Duration::ZERO), &mut || true),
            ScoreEnd::HungUp
        );
        assert_eq!((runtime.began, runtime.scored), (0, 0));
    }

    #[test]
    fn a_limit_reached_between_rows_stops_the_run_there() {
        let mut runtime = SteppedRuntime {
            per_row: Duration::from_millis(40),
            ..SteppedRuntime::default()
        };
        let limit = Duration::from_millis(60);
        assert_eq!(
            score_rows_within(&mut runtime, &rows(5), Some(limit), &mut || false),
            ScoreEnd::TimeLimit(limit)
        );
        assert_eq!(runtime.began, 1);
        assert!(
            (1..5).contains(&runtime.scored),
            "scored {} of 5 rows",
            runtime.scored
        );
    }

    #[test]
    fn each_end_becomes_its_reply() {
        let limit = Duration::from_secs(7);
        assert_eq!(
            ScoreEnd::Scored(Ok(vec![vec![1.0]])).into_reply(),
            Some(Ok(vec![vec![1.0]]))
        );
        assert_eq!(
            ScoreEnd::Scored(Err("lost".to_owned())).into_reply(),
            Some(Err(ScoreError::Failed("lost".to_owned())))
        );
        assert_eq!(ScoreEnd::HungUp.into_reply(), None);
        assert_eq!(
            ScoreEnd::TimeLimit(limit).into_reply(),
            Some(Err(ScoreError::TimeLimit(limit)))
        );
    }
}
