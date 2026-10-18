use super::*;
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::ThreadId;

const WAIT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeCall {
    sequence: usize,
    kind: &'static str,
    thread_id: ThreadId,
}

#[derive(Default)]
struct RuntimeTrace {
    constructed_on: Option<ThreadId>,
    calls: Vec<RuntimeCall>,
    dropped_on: Option<ThreadId>,
}

struct FakeRuntime {
    trace: Arc<Mutex<RuntimeTrace>>,
    generate_started: SyncSender<()>,
    first_generate_release: Receiver<()>,
    second_generate_started: SyncSender<()>,
    control_started: SyncSender<()>,
    control_release: Receiver<()>,
    generate_calls: usize,
}

impl FakeRuntime {
    fn new(
        trace: Arc<Mutex<RuntimeTrace>>,
        generate_started: SyncSender<()>,
        first_generate_release: Receiver<()>,
        second_generate_started: SyncSender<()>,
        control_started: SyncSender<()>,
        control_release: Receiver<()>,
    ) -> Self {
        lock_unpoisoned(&trace).constructed_on = Some(std::thread::current().id());
        Self {
            trace,
            generate_started,
            first_generate_release,
            second_generate_started,
            control_started,
            control_release,
            generate_calls: 0,
        }
    }

    fn record_call(&self, kind: &'static str) {
        let mut trace = lock_unpoisoned(&self.trace);
        let sequence = trace.calls.len() + 1;
        trace.calls.push(RuntimeCall {
            sequence,
            kind,
            thread_id: std::thread::current().id(),
        });
    }

    fn generate(
        &mut self,
        _messages: &[ChatMessage],
        _cfg: &GenerateConfig,
        _explicitly_thinking: bool,
        _on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        self.record_call("generate");
        self.generate_calls += 1;
        match self.generate_calls {
            1 => {
                let _ = self.generate_started.send(());
                if self.first_generate_release.recv_timeout(WAIT).is_err() {
                    return Err(WorkerFailure::Failed(
                        "the first generation was not released".to_string(),
                    ));
                }
            }
            2 => {
                let _ = self.second_generate_started.send(());
            }
            _ => {}
        }
        let _ = should_cancel();
        Ok(empty_output())
    }

    fn control(
        &mut self,
        command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError> {
        self.record_call("control");
        let _ = self.control_started.send(());
        if self.control_release.recv_timeout(WAIT).is_err() {
            return Err("the control call was not released".into());
        }
        match command {
            AdapterCommand::Unload { id } => Ok(AdapterControlResult::Unloaded(id)),
            AdapterCommand::Load { .. } => Err("unexpected load command".into()),
        }
    }
}

impl Drop for FakeRuntime {
    fn drop(&mut self) {
        lock_unpoisoned(&self.trace).dropped_on = Some(std::thread::current().id());
    }
}

fn empty_output() -> GenerateOutput {
    GenerateOutput {
        text: String::new(),
        token_ids: Vec::new(),
        prompt_tokens: 1,
        generated_tokens: 0,
        stopped: false,
        stop_reason: None,
        token_logprobs: Vec::new(),
    }
}

fn collect_events(
    mut events: mpsc::UnboundedReceiver<WorkerEvent>,
    result: SyncSender<Vec<WorkerEvent>>,
) {
    let mut collected = Vec::new();
    while let Some(event) = events.blocking_recv() {
        collected.push(event);
    }
    let _ = result.send(collected);
}

fn is_one_complete(events: &[WorkerEvent]) -> bool {
    events.len() == 1 && matches!(events.first(), Some(WorkerEvent::Complete(_)))
}

#[test]
fn control_remains_fifo_and_ignores_a_dropped_reply_receiver() {
    let test_thread = std::thread::current().id();
    let trace = Arc::new(Mutex::new(RuntimeTrace::default()));
    let (constructed_tx, constructed_rx) = std::sync::mpsc::sync_channel(1);
    let (first_started_tx, first_started_rx) = std::sync::mpsc::sync_channel(1);
    let (release_first_tx, release_first_rx) = std::sync::mpsc::sync_channel(1);
    let (second_started_tx, second_started_rx) = std::sync::mpsc::sync_channel(1);
    let (control_started_tx, control_started_rx) = std::sync::mpsc::sync_channel(1);
    let (release_control_tx, release_control_rx) = std::sync::mpsc::sync_channel(1);
    let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let worker_trace = Arc::clone(&trace);

    let worker = std::thread::spawn(move || {
        let runtime = RefCell::new(FakeRuntime::new(
            worker_trace,
            first_started_tx,
            release_first_rx,
            second_started_tx,
            control_started_tx,
            release_control_rx,
        ));
        let _ = constructed_tx.send(std::thread::current().id());
        run_worker_loop_with_control(
            job_rx,
            |messages, cfg, thinking, on_token, should_cancel| {
                runtime
                    .borrow_mut()
                    .generate(messages, cfg, thinking, on_token, should_cancel)
            },
            |command| runtime.borrow_mut().control(command),
        );
        drop(runtime);
        let _ = done_tx.send(std::thread::current().id());
    });
    let owner = MetalWorkerOwner::from_handle(worker);
    let client = MetalWorkerClient::with_owner(
        job_tx,
        Arc::new(Semaphore::new(2)),
        Arc::new(AtomicBool::new(false)),
        owner.clone(),
    );

    let constructed_on = constructed_rx.recv_timeout(WAIT);
    let (first_guard, first_cancel) = crate::serve::cancel_pair();
    let first_job = client.submit(
        vec![ChatMessage::user("first")],
        GenerateConfig::default(),
        first_cancel,
    );
    let first_submitted = first_job.is_ok();
    let (first_events_tx, first_events_rx) = std::sync::mpsc::sync_channel(1);
    let first_collector = match first_job {
        Ok(events) => Some(std::thread::spawn(move || {
            collect_events(events, first_events_tx)
        })),
        Err(_) => None,
    };

    let first_started = first_started_rx.recv_timeout(WAIT);
    let control_reply = client.submit_adapter_command(AdapterCommand::Unload { id: 0 });
    let control_enqueued = control_reply.is_ok();
    drop(control_reply);
    let (second_guard, second_cancel) = crate::serve::cancel_pair();
    let second_job = client.submit(
        vec![ChatMessage::user("second")],
        GenerateConfig::default(),
        second_cancel,
    );
    let second_submitted = second_job.is_ok();
    let (second_events_tx, second_events_rx) = std::sync::mpsc::sync_channel(1);
    let second_collector = match second_job {
        Ok(events) => Some(std::thread::spawn(move || {
            collect_events(events, second_events_tx)
        })),
        Err(_) => None,
    };
    let _ = release_first_tx.send(());

    let first_result = first_events_rx.recv_timeout(WAIT);
    let control_started = control_started_rx.recv_timeout(WAIT);
    let second_stayed_queued = matches!(
        second_started_rx.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    );
    let _ = release_control_tx.send(());
    let second_result = second_events_rx.recv_timeout(WAIT);

    drop(client);
    let worker_done = done_rx.recv_timeout(WAIT);
    let worker_shutdown = owner._inner.wait_for_exit(WAIT);
    drop(owner);
    if let Some(collector) = first_collector {
        let _ = collector.join();
    }
    if let Some(collector) = second_collector {
        let _ = collector.join();
    }
    drop(first_guard);
    drop(second_guard);

    assert!(first_submitted, "the first job must be accepted");
    assert!(first_started.is_ok(), "the first generation must start");
    assert!(control_enqueued, "the control command must be enqueued");
    assert!(control_started.is_ok(), "the control command must run");
    assert!(
        second_stayed_queued,
        "the next generation must wait until the control call returns"
    );
    assert!(second_submitted, "the next job must be accepted");
    assert!(matches!(constructed_on, Ok(thread_id) if thread_id != test_thread));
    assert!(matches!(first_result, Ok(ref events) if is_one_complete(events)));
    assert!(matches!(second_result, Ok(ref events) if is_one_complete(events)));
    assert_eq!(worker_shutdown, WorkerShutdown::Joined);

    let trace = lock_unpoisoned(&trace);
    let worker_thread = trace
        .constructed_on
        .expect("the fake runtime was constructed");
    assert_ne!(worker_thread, test_thread);
    assert!(
        matches!(worker_done, Ok(thread_id) if thread_id == worker_thread),
        "dropping the last client must stop the worker on its runtime thread"
    );
    assert_eq!(
        trace
            .calls
            .iter()
            .map(|call| (call.sequence, call.kind))
            .collect::<Vec<_>>(),
        vec![(1, "generate"), (2, "control"), (3, "generate")]
    );
    assert!(
        trace
            .calls
            .iter()
            .all(|call| call.thread_id == worker_thread)
    );
    assert_eq!(trace.dropped_on, Some(worker_thread));
}

#[test]
fn control_and_generation_capacities_have_independent_refusals() {
    const GENERATION_CAPACITY: usize = DEFAULT_MAX_PENDING_JOBS;
    assert_eq!(DEFAULT_MAX_PENDING_JOBS, 32);
    let (client, _queued_jobs) = test_client_and_jobs_with_cap(GENERATION_CAPACITY);
    let mut guards = Vec::with_capacity(GENERATION_CAPACITY);
    for index in 0..GENERATION_CAPACITY {
        let (guard, cancel) = crate::serve::cancel_pair();
        client
            .submit(
                vec![ChatMessage::user(format!("job {index}"))],
                GenerateConfig::default(),
                cancel,
            )
            .expect("every generation slot below the default capacity is available");
        guards.push(guard);
    }
    let (_extra_guard, extra_cancel) = crate::serve::cancel_pair();
    match client
        .submit(
            vec![ChatMessage::user("over capacity")],
            GenerateConfig::default(),
            extra_cancel,
        )
        .expect_err("the next job exceeds the default generation capacity")
    {
        ApiError::ServiceUnavailable { message } => assert_eq!(
            message,
            "too many outstanding requests; the inference worker's pending-job queue is full, retry shortly"
        ),
        other => panic!("expected generation-capacity refusal, got {other:?}"),
    }

    let first_control = client
        .submit_adapter_command(AdapterCommand::Unload { id: 0 })
        .expect("the control slot is independent of the full generation capacity");
    match client.submit_adapter_command(AdapterCommand::Unload { id: 1 }) {
        Err(ApiError::ServiceUnavailable { message }) => assert_eq!(
            message,
            "an adapter command is already in flight on the inference worker; wait for its result before sending another"
        ),
        other => panic!("expected control-capacity refusal, got {other:?}"),
    }
    drop(first_control);
    drop(guards);
}

#[test]
fn running_cancellation_sends_one_terminal_event_and_keeps_worker_running() {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let (first_job, first_rx, first_guard) = make_job();
    let (second_job, second_rx, _second_guard) = make_job();
    job_tx.send(first_job).expect("first job enters the queue");
    job_tx
        .send(second_job)
        .expect("second job enters the queue");
    drop(job_tx);

    let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let (first_events_tx, first_events_rx) = std::sync::mpsc::sync_channel(1);
    let first_collector = std::thread::spawn(move || collect_events(first_rx, first_events_tx));
    let (second_events_tx, second_events_rx) = std::sync::mpsc::sync_channel(1);
    let second_collector = std::thread::spawn(move || collect_events(second_rx, second_events_tx));
    let calls = Arc::new(AtomicUsize::new(0));
    let cancel_observed = Arc::new(AtomicBool::new(false));
    let worker_calls = Arc::clone(&calls);
    let worker_cancel_observed = Arc::clone(&cancel_observed);
    let worker = std::thread::spawn(move || {
        run_worker_loop(
            job_rx,
            move |_messages, _cfg, _thinking, _on_token, should_cancel| {
                if worker_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    let _ = entered_tx.send(());
                    let _ = release_rx.recv_timeout(WAIT);
                    worker_cancel_observed.store(should_cancel(), Ordering::SeqCst);
                } else {
                    let _ = should_cancel();
                }
                Ok(empty_output())
            },
        );
    });

    let entered = entered_rx.recv_timeout(WAIT);
    drop(first_guard);
    let _ = release_tx.send(());
    let first_events = first_events_rx.recv_timeout(WAIT);
    let second_events = second_events_rx.recv_timeout(WAIT);
    let worker_joined = worker.join();
    let _ = first_collector.join();
    let _ = second_collector.join();

    assert!(entered.is_ok(), "the first fake generation must be running");
    assert!(cancel_observed.load(Ordering::SeqCst));
    assert!(matches!(first_events, Ok(ref events) if is_one_complete(events)));
    assert!(matches!(second_events, Ok(ref events) if is_one_complete(events)));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        worker_joined.is_ok(),
        "the worker must not panic after cancellation"
    );
}

#[test]
fn loader_panic_before_readiness_returns_thread_exited() {
    let result = MetalWorker::spawn(
        || -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String> {
            panic!("injected constructor panic");
        },
        DEFAULT_MAX_PENDING_JOBS,
        ResidencyLimits::default(),
    );
    let Err(error) = result else {
        panic!("a constructor panic must close the readiness channel")
    };

    assert!(matches!(&error, StartupError::ThreadExited));
    assert_eq!(
        error.to_string(),
        "worker thread exited before loading finished"
    );
}
