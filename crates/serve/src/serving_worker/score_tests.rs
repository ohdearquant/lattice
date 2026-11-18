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
    score: impl FnMut(&[ScoreRow], &mut dyn FnMut() -> bool) -> Option<ScoreOutcome> + Send + 'static,
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

fn request(rows: Vec<ScoreRow>) -> (WorkerMessage, oneshot::Receiver<ScoreOutcome>) {
    let (reply, rx) = oneshot::channel();
    let permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("fresh semaphore");
    (
        WorkerMessage::Score(ScoreRequest {
            rows,
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
        matches!(&outcome, Err(message) if message.contains("does not score")),
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
        .submit_score(vec![row(&[1], &[2])])
        .expect("admitted");
    let refused = client.submit_score(vec![row(&[1], &[2])]);
    assert!(
        matches!(refused, Err(ApiError::ServiceUnavailable { .. })),
        "a full queue must refuse"
    );
    let queued = rx.try_recv().expect("the admitted request is queued");
    assert!(matches!(queued, WorkerMessage::Score(_)));
    drop(queued);
    assert!(
        client.submit_score(vec![row(&[1], &[2])]).is_ok(),
        "the slot frees once the worker drops the request"
    );
}
