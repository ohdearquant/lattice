use super::*;
use http_body_util::BodyExt;
use std::sync::Mutex;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

/// One real delta, then a bounded should_cancel-only poll loop.
const MAX_POLLS: usize = 4000;
const POLL_INTERVAL: Duration = Duration::from_millis(5);
const PROBE_POLL_BUDGET: Duration = POLL_INTERVAL.saturating_mul(MAX_POLLS as u32);
const PROBE_COMPLETION_HEADROOM: Duration = Duration::from_secs(20);
const PROBE_COMPLETION_TIMEOUT: Duration =
    PROBE_POLL_BUDGET.saturating_add(PROBE_COMPLETION_HEADROOM);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

fn await_checkpoint(rx: &std::sync::mpsc::Receiver<()>, checkpoint: &str, timeout: Duration) {
    match rx.recv_timeout(timeout) {
        Ok(()) => {}
        Err(RecvTimeoutError::Timeout) => {
            panic!(
                "fake generator timed out waiting for the {checkpoint} \
                             checkpoint after {timeout:?}"
            );
        }
        Err(RecvTimeoutError::Disconnected) => {
            panic!(
                "fake generator's {checkpoint} checkpoint sender \
                             disconnected before signaling"
            );
        }
    }
}

fn probe_completion_timeout_message() -> String {
    format!(
        "observed no ProbeOutcome within {PROBE_COMPLETION_TIMEOUT:?} \
                     after the client disconnect; possible causes include a \
                     generator scheduling delay or stall, or cancellation \
                     propagation failing to complete within the \
                     {PROBE_POLL_BUDGET:?} probe poll budget"
    )
}

#[test]
fn completion_timeout_exceeds_probe_poll_budget() {
    assert_eq!(PROBE_POLL_BUDGET, Duration::from_secs(20));
    assert!(
        PROBE_COMPLETION_TIMEOUT > PROBE_POLL_BUDGET,
        "completion timeout {PROBE_COMPLETION_TIMEOUT:?} must exceed \
                     the complete probe poll budget {PROBE_POLL_BUDGET:?}"
    );
}

#[test]
#[should_panic(expected = "post-handler-return checkpoint")]
fn post_handler_return_checkpoint_timeout_fails_loudly() {
    let (tx, rx) = std::sync::mpsc::channel();
    await_checkpoint(&rx, "post-handler-return", Duration::ZERO);
    drop(tx);
}

#[test]
#[should_panic(expected = "post-body-drop checkpoint")]
fn post_body_drop_checkpoint_timeout_fails_loudly() {
    let (tx, rx) = std::sync::mpsc::channel();
    await_checkpoint(&rx, "post-body-drop", Duration::ZERO);
    drop(tx);
}

#[test]
fn completion_timeout_message_reports_observation_and_candidate_causes() {
    let message = probe_completion_timeout_message();
    assert!(message.contains("observed no ProbeOutcome"));
    assert!(message.contains("generator scheduling delay or stall"));
    assert!(message.contains("cancellation propagation"));
}

/// Same rationale as `disconnect_cancellation`'s constant of the
/// same value: the tiny test model's context window is a fixed
/// 1024 tokens, so this stays comfortably under it while still
/// being "effectively unbounded" relative to this test's own
/// timeouts.
const NEAR_MAX_CONTEXT_TOKENS: usize = 900;

/// `should_cancel`'s reading taken after the test's `checkpoint1`
/// signal (sent the instant `chat_completions(..).await`
/// returns, before the test reads any SSE frame or drops
/// anything). The generator acknowledges that reading over a
/// oneshot channel, and the test waits for the acknowledgement
/// before it can drop the body or send `go`. Must be `false` in
/// correctly-wired code:
/// `cancel_guard` is still alive at this point (held by
/// `body_stream`'s `flat_map` closure, itself alive because the
/// test hasn't dropped the body/receiver yet), so `cancel_rx`
/// cannot have flipped. A `true` reading here is the direct,
/// timing-independent signature of the guard-capture-removed
/// mutation: `cancel_guard` is then just an unused
/// local that drops the instant `chat_completions` returns --
/// gating this read on `checkpoint1` (rather than reading it
/// immediately after `on_token`, which raced against
/// `chat_completions`'s own return on an early version of this
/// test and non-deterministically read `false` even under the
/// mutation) guarantees the read happens causally after that
/// return, so an unused `cancel_guard` has unconditionally
/// already dropped by the time this reads it.
#[derive(Debug, PartialEq)]
struct ProbeOutcome {
    pre_drop_cancelled: bool,
    post_drop: PostDropOutcome,
}

#[derive(Debug, PartialEq)]
enum PostDropOutcome {
    /// `should_cancel` returned `true` after this many polls
    /// following the test's `go` signal (sent only after
    /// `drop(body)`) -- the correctly-wired behavior.
    CancelledAfterPolls(usize),
    /// The poll budget was exhausted without `should_cancel`
    /// ever returning `true` -- the `move || false` mutation's
    /// signature (an unthreaded/replaced predicate that can
    /// never observe the drop).
    ExhaustedWithoutCancel,
}

#[tokio::test]
async fn chat_completions_streaming_failure_emits_error_event() {
    let state = tiny_state_with_test_generator(64, |_prompt, _cfg, on_token, _should_cancel| {
        let _ = on_token("partial");
        Err(lattice_inference::error::InferenceError::InvalidInput(
            "blocked by grammar".to_string(),
        ))
    });
    let mut req = bare_req();
    req.model = Some("test-model".to_string());
    req.messages = vec![user_msg("hi")];
    req.max_tokens = Some(64);
    req.stream = Some(true);

    let response = chat_completions_with_request(State(state), req)
        .await
        .expect("streaming request must be accepted");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("SSE response body must be readable")
        .to_bytes();
    let text = String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8");
    assert!(
        text.contains("\"content\":\"partial\""),
        "partial output must precede the error event; got: {text}"
    );
    let error_payload = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .find(|payload| payload.contains("\"error\""))
        .expect("a failed generation must emit an SSE error payload");
    let error: serde_json::Value =
        serde_json::from_str(error_payload).expect("SSE error payload must be valid JSON");
    assert_eq!(error["error"]["type"], "server_error");
    assert_eq!(error["error"]["code"], "internal_error");
    assert!(
        !text.contains("\"finish_reason\":\"stop\""),
        "generation failure must not masquerade as a clean stop; got: {text}"
    );
}

/// Mutation-sensitive to BOTH known regressions
/// independently, via a two-way test<->generator handshake that
/// removes the timing race a plain poll-and-time-it design would have
/// (an early, timing-dependent version of this test observed
/// `CancelledAfterPolls(1)` even under mutation (a), because
/// `cancel_guard` -- unused once its capture is removed --
/// drops at `chat_completions`'s own return, which can race
/// ahead of or behind the generator's first poll depending on
/// blocking-thread-pool scheduling):
///
/// (a) removing `body_stream`'s
///     `let _cancel_guard_tied_to_stream_lifetime = &cancel_guard;`
///     capture flips `cancel_rx` the instant `chat_completions`
///     returns -- before the test even reads its first SSE
///     frame, let alone drops the body. `pre_drop_cancelled`
///     (read BEFORE the generator waits on the `go` signal,
///     which the test only sends after `drop(body)`) captures
///     exactly this: `true` here is impossible under correct
///     wiring regardless of scheduling, since `cancel_guard` is
///     provably still alive at that point.
/// (b) replacing the CPU `should_cancel` predicate
///     (`move || *cancel_rx.borrow()`) with `move || false`
///     means the post-`go` poll loop never observes `true`, so
///     it exhausts `MAX_POLLS` and reports
///     `PostDropOutcome::ExhaustedWithoutCancel`.
#[tokio::test]
async fn chat_completions_streaming_disconnect_cancellation_reaches_generator_post_drop() {
    let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel::<ProbeOutcome>();
    let outcome_tx = Mutex::new(Some(outcome_tx));
    let (checkpoint1_tx, checkpoint1_rx) = std::sync::mpsc::channel::<()>();
    let checkpoint1_rx = Mutex::new(Some(checkpoint1_rx));
    let (pre_drop_observed_tx, pre_drop_observed_rx) = tokio::sync::oneshot::channel::<bool>();
    let pre_drop_observed_tx = Mutex::new(Some(pre_drop_observed_tx));
    let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
    let go_rx = Mutex::new(Some(go_rx));

    let state = tiny_state_with_test_generator(
        NEAR_MAX_CONTEXT_TOKENS,
        move |_prompt, _cfg, on_token, should_cancel| {
            // One real delta -- exactly what the disconnect test
            // above reads as its second frame -- so
            // `chat_completions`'s SSE framing has genuine
            // content before this probe phase begins. Queued
            // into the reply channel immediately; the test can
            // read it as frame 2 regardless of whether this
            // thread is later blocked waiting on `checkpoint1`.
            on_token("probe");

            // Block until the test confirms `chat_completions`
            // itself has already returned (see `pre_drop_cancelled`'s
            // doc comment on why this ordering, not an
            // immediate post-`on_token` read, is what makes the
            // next line's reading timing-independent).
            if let Some(rx) = checkpoint1_rx.lock().unwrap().take() {
                await_checkpoint(&rx, "post-handler-return", HANDSHAKE_TIMEOUT);
            }
            let pre_drop_cancelled = should_cancel();
            if let Some(tx) = pre_drop_observed_tx.lock().unwrap().take() {
                let _ = tx.send(pre_drop_cancelled);
            }

            // Block until the test signals it has dropped the
            // body (or give up after 10s so a broken handshake
            // fails this test instead of hanging the process).
            if let Some(rx) = go_rx.lock().unwrap().take() {
                await_checkpoint(&rx, "post-body-drop", HANDSHAKE_TIMEOUT);
            }

            let mut polls = 0usize;
            let post_drop = loop {
                if should_cancel() {
                    break PostDropOutcome::CancelledAfterPolls(polls);
                }
                if polls >= MAX_POLLS {
                    break PostDropOutcome::ExhaustedWithoutCancel;
                }
                polls += 1;
                std::thread::sleep(POLL_INTERVAL);
            };
            if let Some(tx) = outcome_tx.lock().unwrap().take() {
                let _ = tx.send(ProbeOutcome {
                    pre_drop_cancelled,
                    post_drop,
                });
            }
            Ok(GenerateOutput {
                text: "probe".to_string(),
                token_ids: vec![],
                prompt_tokens: 1,
                generated_tokens: 1,
                stopped: false,
                stop_reason: Some(lattice_inference::stop_reason::StopReason::Interrupt),
                token_logprobs: vec![],
            })
        },
    );

    let mut req = bare_req();
    req.model = Some("test-model".to_string());
    req.messages = vec![user_msg("hi")];
    req.max_tokens = Some(NEAR_MAX_CONTEXT_TOKENS);
    req.stream = Some(true);
    let response = chat_completions_with_request(State(state), req)
        .await
        .expect("streaming request must be accepted");
    // `chat_completions` has now unconditionally returned, so an
    // unused `cancel_guard` (mutation (a)) has already dropped;
    // signal the generator it may take its `pre_drop_cancelled`
    // reading.
    checkpoint1_tx
        .send(())
        .expect("generator must still be waiting for the post-return checkpoint");
    let observed_pre_drop_cancelled = tokio::time::timeout(HANDSHAKE_TIMEOUT, pre_drop_observed_rx)
        .await
        .expect(
            "generator must acknowledge its pre-drop cancellation \
                             reading before the handshake timeout",
        )
        .expect(
            "generator must not drop the pre-drop observation \
                             sender without acknowledging its reading",
        );
    assert!(
        !observed_pre_drop_cancelled,
        "should_cancel must read false while the response body is \
                     still alive"
    );
    let mut body = response.into_body();

    // Frame 1: role chunk.
    tokio::time::timeout(Duration::from_secs(10), body.frame())
        .await
        .expect("role chunk frame must arrive quickly")
        .expect("role chunk frame must exist")
        .expect("role chunk frame must be Ok");

    // Frame 2: the fake generator's one real delta.
    tokio::time::timeout(Duration::from_secs(10), body.frame())
        .await
        .expect("delta frame must arrive quickly")
        .expect("delta frame must exist")
        .expect("delta frame must be Ok");

    // Client disconnect: drop the body. The fake generator is
    // waiting on `go_rx`, having already called `on_token` and
    // taken its `pre_drop_cancelled` reading.
    drop(body);
    go_tx
        .send(())
        .expect("generator must still be waiting for the post-drop checkpoint");

    let outcome = tokio::time::timeout(PROBE_COMPLETION_TIMEOUT, outcome_rx)
        .await
        .unwrap_or_else(|_| panic!("{}", probe_completion_timeout_message()))
        .expect("probe outcome sender must not be dropped without sending");

    assert!(
        !outcome.pre_drop_cancelled,
        "should_cancel must read false before the test has dropped the \
                     response body -- a true reading here means cancel_guard had \
                     already dropped (e.g. because it is no longer tied to the \
                     stream's lifetime) well before any disconnect happened: \
                     {outcome:?}"
    );
    assert!(
        matches!(outcome.post_drop, PostDropOutcome::CancelledAfterPolls(_)),
        "should_cancel must return true within the poll budget after \
                     the test drops the response body -- exhausting the budget \
                     means the disconnect signal never reached the generator at \
                     all: {outcome:?}"
    );
}
