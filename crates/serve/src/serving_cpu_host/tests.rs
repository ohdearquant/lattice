use super::{CpuRuntimeSource, SharedCpuHandle};
use crate::serve::prepare::PreparationHandle;
use crate::serving_runtime::{ServingRuntime, WorkerFailure};
use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Observation {
    entry: TextGenerationEntry,
    cancelled: bool,
}

#[derive(Default)]
struct Records {
    created_on: Mutex<Vec<ThreadId>>,
    observations: Mutex<Vec<Observation>>,
    dropped_on: Mutex<Vec<ThreadId>>,
}

struct RecordingSource {
    records: Arc<Records>,
}

impl CpuRuntimeSource for RecordingSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        let thread_id = std::thread::current().id();
        self.records
            .created_on
            .lock()
            .expect("record lock")
            .push(thread_id);
        Box::new(RecordingRuntime {
            records: Arc::clone(&self.records),
        })
    }
}

struct RecordingRuntime {
    records: Arc<Records>,
}

impl ServingRuntime for RecordingRuntime {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        _cfg: &GenerateConfig,
        _http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        let RuntimeInput::PreparedText { entry, .. } = input else {
            return Err(WorkerFailure::Failed("unexpected test input".to_owned()));
        };
        let cancelled = should_cancel();
        self.records
            .observations
            .lock()
            .expect("record lock")
            .push(Observation { entry, cancelled });
        let _ = on_token("delta", 0);
        Ok(GenerateOutput {
            text: "delta".to_owned(),
            token_ids: Vec::new(),
            prompt_tokens: 0,
            generated_tokens: 1,
            stopped: false,
            stop_reason: None,
            token_logprobs: Vec::new(),
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: crate::serve::metal_worker::AdapterCommand,
    ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
    {
        Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
            "test runtime does not accept adapters".to_owned(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }
}

impl Drop for RecordingRuntime {
    fn drop(&mut self) {
        self.records
            .dropped_on
            .lock()
            .expect("record lock")
            .push(std::thread::current().id());
    }
}

struct OverlapGate {
    barrier: Barrier,
    state: Mutex<(bool, usize)>,
    active: AtomicUsize,
    max_active: AtomicUsize,
}

impl OverlapGate {
    fn new() -> Self {
        Self {
            barrier: Barrier::new(2),
            state: Mutex::new((false, 0)),
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        }
    }

    fn enter(&self) -> bool {
        let mut state = self.state.lock().expect("overlap state lock");
        if state.0 {
            return false;
        }
        state.1 += 1;
        true
    }

    fn release_after_timeout(&self) -> usize {
        let mut state = self.state.lock().expect("overlap state lock");
        state.0 = true;
        state.1
    }
}

struct OverlapSource {
    gate: Arc<OverlapGate>,
}

impl CpuRuntimeSource for OverlapSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(OverlapRuntime {
            gate: Arc::clone(&self.gate),
            vision: Arc::new(AtomicBool::new(false)),
        })
    }
}

struct OverlapRuntime {
    gate: Arc<OverlapGate>,
    vision: Arc<AtomicBool>,
}

impl ServingRuntime for OverlapRuntime {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        _cfg: &GenerateConfig,
        _http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        _should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        let RuntimeInput::PreparedText { .. } = input else {
            return Err(WorkerFailure::Failed("unexpected test input".to_owned()));
        };
        let active = self.gate.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.gate.max_active.fetch_max(active, Ordering::SeqCst);
        if self.gate.enter() {
            let _ = self.gate.barrier.wait();
        }
        self.gate.active.fetch_sub(1, Ordering::SeqCst);
        let _ = on_token("delta", 0);
        Ok(GenerateOutput {
            text: "delta".to_owned(),
            token_ids: Vec::new(),
            prompt_tokens: 0,
            generated_tokens: 1,
            stopped: false,
            stop_reason: None,
            token_logprobs: Vec::new(),
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: crate::serve::metal_worker::AdapterCommand,
    ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
    {
        Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
            "test runtime does not accept adapters".to_owned(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.vision)
    }
}

struct ScoringSource {
    created_on: Mutex<Vec<ThreadId>>,
    calls: Mutex<Vec<String>>,
    refuse_begin: bool,
}

impl CpuRuntimeSource for ScoringSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        self.created_on
            .lock()
            .expect("record lock")
            .push(std::thread::current().id());
        Box::new(ScoringRuntime { source: self })
    }
}

/// The methods of a runtime that only scores: generation fails, adapter
/// control refuses, and vision is unsupported.
macro_rules! scoring_only_runtime {
    () => {
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
            _command: crate::serving_worker::AdapterCommand,
        ) -> Result<
            crate::serve::lora::AdapterControlResult,
            crate::serve::lora::AdapterControlError,
        > {
            Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
                "no adapters in this test".to_owned(),
            ))
        }

        fn vision_supported(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }
    };
}

/// Answers each row with its label ids as logits, recording each call.
struct ScoringRuntime<'a> {
    source: &'a ScoringSource,
}

impl ServingRuntime for ScoringRuntime<'_> {
    scoring_only_runtime!();

    fn begin_option_rows(
        &mut self,
        rows: &[crate::serving_worker::ScoreRow],
        _hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<Result<(), String>> {
        self.source
            .calls
            .lock()
            .expect("calls lock")
            .push(format!("begin {}", rows.len()));
        if self.source.refuse_begin {
            return Some(Err("the shared state could not be prepared".to_owned()));
        }
        Some(Ok(()))
    }

    fn score_option_rows(
        &mut self,
        rows: &[crate::serving_worker::ScoreRow],
        _hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<crate::serving_worker::ScoreOutcome> {
        self.source
            .calls
            .lock()
            .expect("calls lock")
            .push(format!("score {}", rows.len()));
        Some(Ok(rows
            .iter()
            .map(|row| row.labels.iter().map(|&label| label as f32).collect())
            .collect()))
    }
}

fn scoring_host() -> (SharedCpuHandle, Arc<ScoringSource>) {
    scoring_host_refusing_begin(false)
}

fn scoring_host_refusing_begin(refuse_begin: bool) -> (SharedCpuHandle, Arc<ScoringSource>) {
    let source = Arc::new(ScoringSource {
        created_on: Mutex::new(Vec::new()),
        calls: Mutex::new(Vec::new()),
        refuse_begin,
    });
    let host = SharedCpuHandle::new(
        Arc::clone(&source) as Arc<dyn CpuRuntimeSource>,
        preparation(),
    );
    (host, source)
}

fn score_rows() -> Vec<crate::serving_worker::ScoreRow> {
    vec![
        crate::serving_worker::ScoreRow {
            ids: vec![1, 2],
            labels: vec![10, 11, 12],
        },
        crate::serving_worker::ScoreRow {
            ids: vec![3],
            labels: vec![20, 21],
        },
    ]
}

#[tokio::test]
async fn cpu_scoring_runs_on_a_blocking_thread_and_returns_every_row() {
    let (host, source) = scoring_host();
    let admission = Arc::new(tokio::sync::Semaphore::new(1));
    let serving = crate::serving_host::ServingHandle::shared_cpu(host, Arc::clone(&admission));
    let receiver = serving.submit_score(score_rows(), None).expect("admitted");
    let logits = receiver.await.expect("replied").expect("scored");
    assert_eq!(logits, vec![vec![10.0, 11.0, 12.0], vec![20.0, 21.0]]);
    let created_on = source.created_on.lock().expect("record lock");
    assert_eq!(created_on.len(), 1);
    assert_ne!(created_on[0], std::thread::current().id());
    drop(created_on);
    assert_eq!(admission.available_permits(), 1, "the permit is returned");
}

#[tokio::test]
async fn cpu_scoring_beyond_the_generation_cap_is_refused_as_unavailable() {
    let (host, source) = scoring_host();
    let serving = crate::serving_host::ServingHandle::shared_cpu(
        host,
        Arc::new(tokio::sync::Semaphore::new(0)),
    );
    let refused = serving
        .submit_score(score_rows(), None)
        .expect_err("cap is full");
    assert!(
        matches!(refused, crate::serve::ApiError::ServiceUnavailable { .. }),
        "{refused:?}"
    );
    assert!(source.created_on.lock().expect("record lock").is_empty());
}

/// Scores one row at a time, holding the first row until the test releases it.
struct HangupSource {
    scored: Arc<std::sync::atomic::AtomicUsize>,
    started: Mutex<std::sync::mpsc::Sender<()>>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl CpuRuntimeSource for HangupSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(HangupRuntime { source: self })
    }
}

struct HangupRuntime<'a> {
    source: &'a HangupSource,
}

impl ServingRuntime for HangupRuntime<'_> {
    scoring_only_runtime!();

    fn score_option_rows(
        &mut self,
        rows: &[crate::serving_worker::ScoreRow],
        _hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<crate::serving_worker::ScoreOutcome> {
        let first = self
            .source
            .scored
            .fetch_add(rows.len(), std::sync::atomic::Ordering::SeqCst)
            == 0;
        if first {
            let _ = self.source.started.lock().expect("started lock").send(());
            let _ = self.source.release.lock().expect("release lock").recv();
        }
        Some(Ok(rows
            .iter()
            .map(|row| vec![0.0; row.labels.len()])
            .collect()))
    }
}

#[tokio::test]
async fn cpu_scoring_stops_between_rows_once_the_caller_hangs_up() {
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let source = Arc::new(HangupSource {
        scored: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        started: Mutex::new(started_tx),
        release: Mutex::new(release_rx),
    });
    let scored = Arc::clone(&source.scored);
    let host = SharedCpuHandle::new(source as Arc<dyn CpuRuntimeSource>, preparation());
    let admission = Arc::new(tokio::sync::Semaphore::new(1));
    let serving = crate::serving_host::ServingHandle::shared_cpu(host, Arc::clone(&admission));
    let receiver = serving.submit_score(score_rows(), None).expect("admitted");
    tokio::task::spawn_blocking(move || started_rx.recv().expect("first row started"))
        .await
        .expect("joined");
    drop(receiver);
    release_tx.send(()).expect("release");
    let permit = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        Arc::clone(&admission).acquire_owned(),
    )
    .await
    .expect("the scoring task finishes")
    .expect("semaphore open");
    drop(permit);
    assert_eq!(
        scored.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the second row is not scored after the caller hung up"
    );
}

fn preparation() -> PreparationHandle {
    let tokenizer = BpeTokenizer::from_vocab_and_merges(
        std::collections::HashMap::from([("a".to_owned(), 0)]),
        Vec::new(),
    )
    .expect("tiny tokenizer constructs");
    PreparationHandle::qwen(Arc::new(tokenizer), 128)
}

/// Records the thread name and rayon pool width each request runs with.
#[derive(Default)]
struct PoolProbeSource {
    seen: Mutex<Vec<(Option<String>, usize)>>,
}

impl CpuRuntimeSource for PoolProbeSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(PoolProbeRuntime { source: self })
    }
}

struct PoolProbeRuntime<'a> {
    source: &'a PoolProbeSource,
}

impl PoolProbeRuntime<'_> {
    fn record(&self) {
        self.source.seen.lock().expect("seen lock").push((
            std::thread::current().name().map(str::to_owned),
            rayon::current_num_threads(),
        ));
    }
}

impl ServingRuntime for PoolProbeRuntime<'_> {
    fn execute(
        &mut self,
        _input: RuntimeInput<'_>,
        _cfg: &GenerateConfig,
        _http_stream: bool,
        _on_token: &mut dyn FnMut(&str, u32) -> bool,
        _should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        self.record();
        Err(WorkerFailure::Failed(
            "no generation in this test".to_owned(),
        ))
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: crate::serving_worker::AdapterCommand,
    ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
    {
        Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
            "no adapters in this test".to_owned(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    fn score_option_rows(
        &mut self,
        rows: &[crate::serving_worker::ScoreRow],
        _hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<crate::serving_worker::ScoreOutcome> {
        self.record();
        Some(Ok(rows
            .iter()
            .map(|row| vec![0.0; row.labels.len()])
            .collect()))
    }
}

/// Run one completion, one streaming request and one scoring request through
/// `serving` and its CPU host, returning what each recorded.
async fn probe_every_entry(
    serving: crate::serving_host::ServingHandle,
    host: SharedCpuHandle,
    source: &PoolProbeSource,
) -> Vec<(Option<String>, usize)> {
    let _ = host
        .spawn_completion(String::new(), GenerateConfig::default())
        .await
        .expect("completion task joins");
    let (_cancel, cancel_rx) = tokio::sync::watch::channel(false);
    let (finish, finished) = tokio::sync::oneshot::channel();
    host.spawn_streaming(
        String::new(),
        GenerateConfig::default(),
        cancel_rx,
        |_| true,
        |_| true,
        move |result| {
            let _ = finish.send(result);
        },
    );
    let _ = finished.await.expect("streaming completion callback runs");
    serving
        .submit_score(score_rows(), None)
        .expect("admitted")
        .await
        .expect("replied")
        .expect("scored");
    source.seen.lock().expect("seen lock").clone()
}

#[tokio::test]
async fn cpu_threads_runs_generation_and_scoring_on_the_dedicated_pool() {
    let source = Arc::new(PoolProbeSource::default());
    let host = SharedCpuHandle::new(
        Arc::clone(&source) as Arc<dyn CpuRuntimeSource>,
        preparation(),
    );
    let serving = crate::serving_host::ServingHandle::shared_cpu(
        host,
        Arc::new(tokio::sync::Semaphore::new(1)),
    )
    .with_cpu_threads(std::num::NonZeroUsize::new(3).expect("non-zero"))
    .expect("the pool starts");
    let host = serving.shared_cpu_for_test().expect("a CPU host").clone();
    let seen = probe_every_entry(serving, host, &source).await;
    assert!(seen.len() >= 3, "{seen:?}");
    for (name, width) in &seen {
        assert!(
            name.as_deref()
                .is_some_and(|name| name.starts_with("lattice-cpu-")),
            "{seen:?}"
        );
        assert_eq!(*width, 3, "{seen:?}");
    }
}

#[tokio::test]
async fn without_cpu_threads_requests_stay_off_any_dedicated_pool() {
    let source = Arc::new(PoolProbeSource::default());
    let host = SharedCpuHandle::new(
        Arc::clone(&source) as Arc<dyn CpuRuntimeSource>,
        preparation(),
    );
    let serving = crate::serving_host::ServingHandle::shared_cpu(
        host.clone(),
        Arc::new(tokio::sync::Semaphore::new(1)),
    );
    let seen = probe_every_entry(serving, host, &source).await;
    assert!(seen.len() >= 3, "{seen:?}");
    for (name, width) in &seen {
        assert!(
            !name
                .as_deref()
                .is_some_and(|name| name.starts_with("lattice-cpu-")),
            "{seen:?}"
        );
        assert_eq!(*width, rayon::current_num_threads(), "{seen:?}");
    }
}

#[tokio::test]
async fn completion_and_streaming_keep_distinct_entries_on_blocking_threads() {
    let caller_thread = std::thread::current().id();
    let records = Arc::new(Records::default());
    let host = SharedCpuHandle::new(
        Arc::new(RecordingSource {
            records: Arc::clone(&records),
        }),
        preparation(),
    );

    host.spawn_completion(String::new(), GenerateConfig::default())
        .await
        .expect("completion task joins")
        .expect("completion succeeds");

    let (cancel, cancel_rx) = tokio::sync::watch::channel(true);
    let (finish, finished) = tokio::sync::oneshot::channel();
    host.spawn_streaming(
        String::new(),
        GenerateConfig::default(),
        cancel_rx,
        |_| true,
        |_| true,
        move |result| {
            let _ = finish.send(result);
        },
    );
    finished
        .await
        .expect("streaming completion callback runs")
        .expect("streaming succeeds");
    drop(cancel);

    let created_on = records.created_on.lock().expect("record lock").clone();
    let observations = records.observations.lock().expect("record lock").clone();
    let dropped_on = records.dropped_on.lock().expect("record lock").clone();
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].entry, TextGenerationEntry::Complete);
    assert!(!observations[0].cancelled);
    assert_eq!(
        observations[1].entry,
        TextGenerationEntry::StreamingWithCancel
    );
    assert!(observations[1].cancelled);
    assert_eq!(created_on.len(), 2);
    assert_eq!(dropped_on, created_on);
    assert!(
        created_on
            .iter()
            .all(|thread_id| *thread_id != caller_thread)
    );
}

#[tokio::test]
async fn completion_and_streaming_execute_at_the_same_time() {
    let gate = Arc::new(OverlapGate::new());
    let host = SharedCpuHandle::new(
        Arc::new(OverlapSource {
            gate: Arc::clone(&gate),
        }),
        preparation(),
    );
    let completion = host.spawn_completion(String::new(), GenerateConfig::default());
    let (finish, finished) = tokio::sync::oneshot::channel();
    host.spawn_streaming(
        String::new(),
        GenerateConfig::default(),
        tokio::sync::watch::channel(false).1,
        |_| true,
        |_| true,
        move |result| {
            let _ = finish.send(result);
        },
    );
    let requests = async {
        let completion = completion
            .await
            .expect("completion task joins")
            .expect("completion succeeds");
        let streaming = finished
            .await
            .expect("streaming callback runs")
            .expect("streaming succeeds");
        (completion, streaming)
    };
    let mut requests = Box::pin(requests);

    let outcome = tokio::time::timeout(Duration::from_secs(2), requests.as_mut()).await;
    let Ok((completion, streaming)) = outcome else {
        if gate.release_after_timeout() == 1 {
            let _ = gate.barrier.wait();
        }
        let _ = tokio::time::timeout(Duration::from_secs(2), requests.as_mut())
            .await
            .expect("requests exit after the bounded overlap probe releases the barrier");
        panic!(
            "completion and streaming did not enter execute concurrently; maximum active runtimes: {}",
            gate.max_active.load(Ordering::SeqCst)
        );
    };

    assert_eq!(completion.output.text, "delta");
    assert_eq!(streaming.text, "delta");
    assert_eq!(gate.max_active.load(Ordering::SeqCst), 2);
}

mod streaming_terminal {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Outcome {
        CreatePanic,
        ExecutePanic,
        DeltaPanic,
        DropPanic,
        Success,
        WorkerFailure,
    }

    #[derive(Default)]
    struct TerminalRecords {
        events: Mutex<Vec<(&'static str, ThreadId)>>,
        finished: AtomicUsize,
    }

    impl TerminalRecords {
        fn record(&self, event: &'static str) {
            self.events
                .lock()
                .expect("event lock")
                .push((event, std::thread::current().id()));
        }

        fn assert_events(&self, caller: ThreadId, expected: &[&str]) {
            let events = self.events.lock().expect("event lock");
            assert_eq!(
                events.iter().map(|(event, _)| *event).collect::<Vec<_>>(),
                expected
            );
            let blocking_thread = events[0].1;
            assert_ne!(blocking_thread, caller);
            assert!(events.iter().all(|(_, thread)| *thread == blocking_thread));
            assert_eq!(self.finished.load(Ordering::SeqCst), 1);
        }
    }

    struct TerminalSource {
        outcome: Outcome,
        records: Arc<TerminalRecords>,
    }

    impl CpuRuntimeSource for TerminalSource {
        fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
            self.records.record("create");
            if self.outcome == Outcome::CreatePanic {
                panic!("private creation panic payload");
            }
            Box::new(TerminalRuntime {
                outcome: self.outcome,
                records: Arc::clone(&self.records),
            })
        }
    }

    struct TerminalRuntime {
        outcome: Outcome,
        records: Arc<TerminalRecords>,
    }

    impl ServingRuntime for TerminalRuntime {
        fn execute(
            &mut self,
            input: RuntimeInput<'_>,
            config: &GenerateConfig,
            http_stream: bool,
            on_token: &mut dyn FnMut(&str, u32) -> bool,
            should_cancel: &mut dyn FnMut() -> bool,
        ) -> Result<GenerateOutput, WorkerFailure> {
            let RuntimeInput::PreparedText { prompt, entry } = input else {
                panic!("unexpected runtime input");
            };
            assert_eq!(prompt, "literal prompt");
            assert_eq!(entry, TextGenerationEntry::StreamingWithCancel);
            assert_eq!(config.max_new_tokens, 7);
            assert!(http_stream);
            assert!(!should_cancel());
            self.records.record("execute");
            assert!(on_token("partial", 17));
            match self.outcome {
                Outcome::ExecutePanic => panic!("private execution panic payload"),
                Outcome::WorkerFailure => {
                    Err(WorkerFailure::Failed("returned worker failure".to_owned()))
                }
                _ => Ok(GenerateOutput {
                    text: "confirmed output".to_owned(),
                    token_ids: vec![17, 19],
                    prompt_tokens: 3,
                    generated_tokens: 2,
                    stopped: false,
                    stop_reason: None,
                    token_logprobs: Vec::new(),
                }),
            }
        }

        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        fn control(
            &mut self,
            _command: crate::serve::metal_worker::AdapterCommand,
        ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
        {
            panic!("CPU terminal probe does not accept adapters");
        }

        fn vision_supported(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }
    }

    impl Drop for TerminalRuntime {
        fn drop(&mut self) {
            self.records.record("drop");
            if self.outcome == Outcome::DropPanic {
                panic!("private cleanup panic payload");
            }
        }
    }

    fn host(outcome: Outcome, records: &Arc<TerminalRecords>) -> SharedCpuHandle {
        SharedCpuHandle::new(
            Arc::new(TerminalSource {
                outcome,
                records: Arc::clone(records),
            }),
            preparation(),
        )
    }

    #[allow(clippy::field_reassign_with_default)]
    fn config() -> GenerateConfig {
        let mut config = GenerateConfig::default();
        config.max_new_tokens = 7;
        config
    }

    async fn observe(outcome: Outcome) -> Result<GenerateOutput, String> {
        let caller = std::thread::current().id();
        let records = Arc::new(TerminalRecords::default());
        let host = host(outcome, &records);
        let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
        let (finish, finished) = tokio::sync::oneshot::channel();
        let delta_records = Arc::clone(&records);
        let finish_records = Arc::clone(&records);
        host.spawn_streaming(
            "literal prompt".to_owned(),
            config(),
            cancel_rx,
            move |delta| {
                assert_eq!(delta, "partial");
                delta_records.record("delta");
                assert!(
                    outcome != Outcome::DeltaPanic,
                    "private delta panic payload"
                );
                true
            },
            |_| true,
            move |result| {
                finish_records.finished.fetch_add(1, Ordering::SeqCst);
                finish_records.record("finish");
                let _ = finish.send(result);
            },
        );
        let result = tokio::time::timeout(Duration::from_secs(5), finished)
            .await
            .expect("terminal callback must meet the bounded deadline")
            .expect("terminal callback must survive a runtime panic");
        drop(cancel);
        let expected: &[&str] = if outcome == Outcome::CreatePanic {
            &["create", "finish"]
        } else {
            &["create", "execute", "delta", "drop", "finish"]
        };
        records.assert_events(caller, expected);
        result
    }

    fn assert_error(result: Result<GenerateOutput, String>, expected: &str) {
        let Err(message) = result else {
            panic!("a failed runtime must not report successful completion");
        };
        assert_eq!(message, expected);
    }

    #[tokio::test]
    async fn streaming_create_runtime_panic_reports_terminal_error_once() {
        assert_error(observe(Outcome::CreatePanic).await, "inference failed");
    }

    #[tokio::test]
    async fn streaming_execute_panic_reports_terminal_error_after_cleanup_once() {
        assert_error(observe(Outcome::ExecutePanic).await, "inference failed");
    }

    #[tokio::test]
    async fn streaming_delta_callback_panic_reports_terminal_error_after_cleanup_once() {
        assert_error(observe(Outcome::DeltaPanic).await, "inference failed");
    }

    #[tokio::test]
    async fn streaming_runtime_drop_panic_reports_terminal_error_once() {
        assert_error(observe(Outcome::DropPanic).await, "inference failed");
    }

    #[tokio::test]
    async fn streaming_success_finishes_once_after_runtime_drop() {
        let output = observe(Outcome::Success).await.expect("runtime succeeds");
        assert_eq!(output.text, "confirmed output");
        assert_eq!(output.token_ids, [17, 19]);
        assert_eq!(output.prompt_tokens, 3);
        assert_eq!(output.generated_tokens, 2);
        assert!(!output.stopped);
        assert!(output.stop_reason.is_none());
        assert!(output.token_logprobs.is_empty());
    }

    #[tokio::test]
    async fn streaming_worker_failure_finishes_once_after_runtime_drop() {
        assert_error(
            observe(Outcome::WorkerFailure).await,
            "returned worker failure",
        );
    }

    struct FinishCapture {
        records: Arc<TerminalRecords>,
        released: Option<tokio::sync::oneshot::Sender<()>>,
    }

    impl Drop for FinishCapture {
        fn drop(&mut self) {
            self.records.record("finish_capture_drop");
            if let Some(released) = self.released.take() {
                let _ = released.send(());
            }
        }
    }

    #[tokio::test]
    async fn streaming_finish_callback_panic_is_not_retried_and_releases_capture() {
        let caller = std::thread::current().id();
        let records = Arc::new(TerminalRecords::default());
        let host = host(Outcome::Success, &records);
        let (released, release_rx) = tokio::sync::oneshot::channel();
        let capture = FinishCapture {
            records: Arc::clone(&records),
            released: Some(released),
        };
        let delta_records = Arc::clone(&records);
        host.spawn_streaming(
            "literal prompt".to_owned(),
            config(),
            tokio::sync::watch::channel(false).1,
            move |delta| {
                assert_eq!(delta, "partial");
                delta_records.record("delta");
                true
            },
            |_| true,
            move |result| {
                let completion_guard = capture;
                assert_eq!(result.expect("runtime succeeds").text, "confirmed output");
                completion_guard
                    .records
                    .finished
                    .fetch_add(1, Ordering::SeqCst);
                completion_guard.records.record("finish");
                panic!("private finish panic payload");
            },
        );
        tokio::time::timeout(Duration::from_secs(5), release_rx)
            .await
            .expect("a panicking callback must release its captures")
            .expect("the callback capture must signal its destruction");
        records.assert_events(
            caller,
            &[
                "create",
                "execute",
                "delta",
                "drop",
                "finish",
                "finish_capture_drop",
            ],
        );
    }
}

#[tokio::test]
async fn cpu_scoring_begins_once_with_every_row_before_scoring_any() {
    let (host, source) = scoring_host();
    let admission = Arc::new(tokio::sync::Semaphore::new(1));
    let serving = crate::serving_host::ServingHandle::shared_cpu(host, admission);
    let receiver = serving.submit_score(score_rows(), None).expect("admitted");
    receiver.await.expect("replied").expect("scored");
    assert_eq!(
        *source.calls.lock().expect("calls lock"),
        vec!["begin 2", "score 1", "score 1"]
    );
}

#[tokio::test]
async fn cpu_scoring_fails_the_request_when_the_runtime_cannot_begin() {
    let (host, source) = scoring_host_refusing_begin(true);
    let admission = Arc::new(tokio::sync::Semaphore::new(1));
    let serving = crate::serving_host::ServingHandle::shared_cpu(host, Arc::clone(&admission));
    let receiver = serving.submit_score(score_rows(), None).expect("admitted");
    let error = receiver.await.expect("replied").expect_err("refused");
    assert_eq!(
        error,
        crate::serving_worker::ScoreError::Failed("the shared state could not be prepared".into())
    );
    assert_eq!(*source.calls.lock().expect("calls lock"), vec!["begin 2"]);
    assert_eq!(admission.available_permits(), 1, "the permit is returned");
}

#[tokio::test]
async fn cpu_scoring_past_its_time_limit_is_told_so_and_the_next_request_is_scored_in_full() {
    let (host, source) = scoring_host();
    let admission = Arc::new(tokio::sync::Semaphore::new(1));
    let serving = crate::serving_host::ServingHandle::shared_cpu(host, Arc::clone(&admission));
    let stopped = serving
        .submit_score(score_rows(), Some(Duration::ZERO))
        .expect("admitted");
    assert_eq!(
        stopped.await.expect("replied"),
        Err(crate::serving_worker::ScoreError::TimeLimit(Duration::ZERO))
    );
    assert!(
        source.calls.lock().expect("calls lock").is_empty(),
        "a spent limit stops the run before the preparation"
    );
    assert_eq!(admission.available_permits(), 1, "the permit is returned");
    let next = serving.submit_score(score_rows(), None).expect("admitted");
    assert_eq!(
        next.await.expect("replied"),
        Ok(vec![vec![10.0, 11.0, 12.0], vec![20.0, 21.0]])
    );
    assert_eq!(
        *source.calls.lock().expect("calls lock"),
        vec!["begin 2", "score 1", "score 1"]
    );
}

const STACK_WAIT: Duration = Duration::from_secs(3);

fn wait_until(flag: &AtomicBool) {
    let started = std::time::Instant::now();
    while !flag.load(Ordering::SeqCst) && started.elapsed() < STACK_WAIT {
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Request `a` waits inside its own parallel work until request `b` has been
/// submitted; request `b` waits until `a` has finished.
#[derive(Default)]
struct StackSource {
    a_parallel_half_started: AtomicBool,
    b_submitted: AtomicBool,
    a_finished: AtomicBool,
}

impl CpuRuntimeSource for StackSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(StackRuntime { source: self })
    }
}

struct StackRuntime<'a> {
    source: &'a StackSource,
}

impl ServingRuntime for StackRuntime<'_> {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        _cfg: &GenerateConfig,
        _http_stream: bool,
        _on_token: &mut dyn FnMut(&str, u32) -> bool,
        _should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        let RuntimeInput::PreparedText { prompt, .. } = input else {
            return Err(WorkerFailure::Failed("no chat messages here".to_owned()));
        };
        let source = self.source;
        if prompt == "a" {
            // The second half is taken by the pool's other thread, so the
            // thread running `a` waits for it inside the join.
            rayon::join(
                || wait_until(&source.a_parallel_half_started),
                || {
                    source.a_parallel_half_started.store(true, Ordering::SeqCst);
                    wait_until(&source.b_submitted);
                    std::thread::sleep(Duration::from_millis(300));
                },
            );
            source.a_finished.store(true, Ordering::SeqCst);
            return Err(WorkerFailure::Failed("a finished".to_owned()));
        }
        wait_until(&source.a_finished);
        if source.a_finished.load(Ordering::SeqCst) {
            Err(WorkerFailure::Failed("b finished after a".to_owned()))
        } else {
            Err(WorkerFailure::Failed(
                "b ran while a could not finish".to_owned(),
            ))
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: crate::serving_worker::AdapterCommand,
    ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
    {
        Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
            "no adapters in this test".to_owned(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    fn score_option_rows(
        &mut self,
        _rows: &[crate::serving_worker::ScoreRow],
        _hung_up: &mut dyn FnMut() -> bool,
    ) -> Option<crate::serving_worker::ScoreOutcome> {
        None
    }
}

/// Two requests under `with_cpu_threads` never share a pool: a pool thread
/// waiting inside one request's parallel work cannot pick up the other request
/// and run it to its end on top of the first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cpu_threads_never_runs_one_request_on_top_of_another() {
    let source = Arc::new(StackSource::default());
    let host = SharedCpuHandle::new(
        Arc::clone(&source) as Arc<dyn CpuRuntimeSource>,
        preparation(),
    )
    .with_cpu_threads(std::num::NonZeroUsize::new(2).expect("non-zero"))
    .expect("the pool starts");
    let a = host.spawn_completion("a".to_owned(), GenerateConfig::default());
    let started = std::time::Instant::now();
    while !source.a_parallel_half_started.load(Ordering::SeqCst) {
        assert!(started.elapsed() < STACK_WAIT, "request a never split");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let b = host.spawn_completion("b".to_owned(), GenerateConfig::default());
    // Give `b` time to reach its pool while `a` is still waiting in its join.
    tokio::time::sleep(Duration::from_millis(100)).await;
    source.b_submitted.store(true, Ordering::SeqCst);
    let b = b.await.expect("b joins");
    let a = a.await.expect("a joins");
    assert_eq!(a.err().as_deref(), Some("a finished"));
    assert_eq!(b.err().as_deref(), Some("b finished after a"));
}
