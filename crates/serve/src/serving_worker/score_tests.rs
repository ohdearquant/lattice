use super::*;
use std::sync::atomic::AtomicUsize;

fn row(ids: &[u32], labels: &[u32]) -> ScoreRow {
    ScoreRow {
        ids: ids.to_vec(),
        labels: labels.to_vec(),
    }
}

fn spawn_loop(
    mut score: impl FnMut(&[ScoreRow]) -> ScoreOutcome + Send + 'static,
) -> (
    mpsc::UnboundedSender<WorkerMessage>,
    std::thread::JoinHandle<()>,
) {
    spawn_loop_watching_hang_up(move |rows, _hung_up| Some(score(rows)))
}

fn spawn_loop_watching_hang_up(
    mut score: impl FnMut(&[ScoreRow], &mut dyn FnMut() -> bool) -> Option<ScoreOutcome>
    + Send
    + 'static,
) -> (
    mpsc::UnboundedSender<WorkerMessage>,
    std::thread::JoinHandle<()>,
) {
    spawn_loop_with_limit(move |rows, _time_limit, hung_up| {
        score(rows, hung_up).map_or(ScoreEnd::HungUp, ScoreEnd::Scored)
    })
}

fn spawn_loop_with_limit(
    score: impl FnMut(&[ScoreRow], Option<Duration>, &mut dyn FnMut() -> bool) -> ScoreEnd
    + Send
    + 'static,
) -> (
    mpsc::UnboundedSender<WorkerMessage>,
    std::thread::JoinHandle<()>,
) {
    let (tx, rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let handle = std::thread::spawn(move || {
        run_worker_loop_with_lora(
            rx,
            |_, _, _, _, _, _, _| -> Result<GenerateOutput, WorkerFailure> {
                panic!("no generation is submitted in these tests")
            },
            control_unsupported,
            score,
        );
    });
    (tx, handle)
}

fn request(rows: Vec<ScoreRow>) -> (WorkerMessage, oneshot::Receiver<ScoreReply>) {
    request_within(rows, None)
}

fn request_within(
    rows: Vec<ScoreRow>,
    time_limit: Option<Duration>,
) -> (WorkerMessage, oneshot::Receiver<ScoreReply>) {
    let (reply, rx) = oneshot::channel();
    let permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("fresh semaphore");
    (
        WorkerMessage::Score(ScoreRequest {
            rows,
            time_limit,
            reply,
            _admission_permit: permit,
        }),
        rx,
    )
}

#[test]
fn a_score_request_gets_the_scorer_output_for_its_rows() {
    let (tx, handle) = spawn_loop(|rows| {
        Ok(rows
            .iter()
            .map(|row| {
                row.labels
                    .iter()
                    .map(|&label| label as f32 + row.ids.len() as f32)
                    .collect()
            })
            .collect())
    });
    let (message, rx) = request(vec![row(&[1, 2, 3], &[10, 11]), row(&[4], &[20, 21, 22])]);
    tx.send(message).expect("worker running");
    let outcome = rx.blocking_recv().expect("reply");
    assert_eq!(outcome, Ok(vec![vec![13.0, 14.0], vec![21.0, 22.0, 23.0]]));
    drop(tx);
    handle.join().expect("worker exits");
}

#[test]
fn a_request_whose_caller_hung_up_is_not_scored() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let (tx, handle) = spawn_loop(move |rows| {
        seen.fetch_add(1, Ordering::SeqCst);
        Ok(rows.iter().map(|_| vec![0.0]).collect())
    });
    let (abandoned, abandoned_rx) = request(vec![row(&[1], &[2])]);
    drop(abandoned_rx);
    tx.send(abandoned).expect("worker running");
    let (live, live_rx) = request(vec![row(&[1], &[2])]);
    tx.send(live).expect("worker running");
    assert!(live_rx.blocking_recv().expect("reply").is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(tx);
    handle.join().expect("worker exits");
}

#[test]
fn a_caller_that_hangs_up_while_its_rows_are_scored_is_seen_by_the_scorer() {
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let (seen_tx, seen_rx) = std::sync::mpsc::channel();
    let (tx, handle) = spawn_loop_watching_hang_up(move |rows, hung_up| {
        let before = hung_up();
        started_tx.send(()).expect("test waiting");
        release_rx.recv().expect("released");
        let after = hung_up();
        seen_tx.send((before, after)).expect("test waiting");
        (!after).then(|| Ok(rows.iter().map(|_| vec![0.0]).collect()))
    });
    let (message, reply) = request(vec![row(&[1], &[2]), row(&[3], &[4])]);
    tx.send(message).expect("worker running");
    started_rx.recv().expect("scoring started");
    drop(reply);
    release_tx.send(()).expect("scorer waiting");
    assert_eq!(seen_rx.recv().expect("scorer reported"), (false, true));
    drop(tx);
    handle.join().expect("worker exits");
}

#[test]
fn a_worker_without_a_scorer_refuses() {
    let (tx, rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let handle = std::thread::spawn(move || {
        run_worker_loop(
            rx,
            |_, _, _, _, _| -> Result<GenerateOutput, WorkerFailure> {
                panic!("no generation is submitted in this test")
            },
        );
    });
    let (message, reply) = request(vec![row(&[1], &[2])]);
    tx.send(message).expect("worker running");
    let outcome = reply.blocking_recv().expect("reply");
    assert!(
        matches!(&outcome, Err(ScoreError::Failed(message)) if message.contains("does not score")),
        "{outcome:?}"
    );
    drop(tx);
    handle.join().expect("worker exits");
}

#[test]
fn submit_score_takes_a_generation_admission_slot() {
    let (tx, mut rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let client = WorkerClient::unattached_for_test(tx, Arc::new(Semaphore::new(1)));
    let _first = client
        .submit_score(vec![row(&[1], &[2])], None)
        .expect("admitted");
    let refused = client.submit_score(vec![row(&[1], &[2])], None);
    assert!(
        matches!(refused, Err(ApiError::ServiceUnavailable { .. })),
        "a full queue must refuse"
    );
    let queued = rx.try_recv().expect("the admitted request is queued");
    assert!(matches!(queued, WorkerMessage::Score(_)));
    drop(queued);
    assert!(
        client.submit_score(vec![row(&[1], &[2])], None).is_ok(),
        "the slot frees once the worker drops the request"
    );
}

#[test]
fn a_request_stopped_at_its_time_limit_is_told_so_and_the_next_one_is_scored() {
    let limits = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&limits);
    let (tx, handle) = spawn_loop_with_limit(move |rows, time_limit, _hung_up| {
        seen.lock().expect("limits lock").push(time_limit);
        match time_limit {
            Some(limit) => ScoreEnd::TimeLimit(limit),
            None => ScoreEnd::Scored(Ok(rows.iter().map(|_| vec![1.0]).collect())),
        }
    });
    let limit = Duration::from_secs(7);
    let (stopped, stopped_rx) = request_within(vec![row(&[1], &[2])], Some(limit));
    tx.send(stopped).expect("worker running");
    assert_eq!(
        stopped_rx.blocking_recv().expect("reply"),
        Err(ScoreError::TimeLimit(limit))
    );
    let (next, next_rx) = request(vec![row(&[1], &[2])]);
    tx.send(next).expect("worker running");
    assert_eq!(next_rx.blocking_recv().expect("reply"), Ok(vec![vec![1.0]]));
    assert_eq!(
        *limits.lock().expect("limits lock"),
        vec![Some(limit), None],
        "each request's own limit reaches the scorer"
    );
    drop(tx);
    handle.join().expect("worker exits");
}
