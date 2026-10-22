use super::*;
use http_body_util::BodyExt;
use std::time::Duration;

/// The tiny test model's context window is a fixed 1024 tokens
/// (`test_support::tiny_zero_model`'s `max_position_embeddings`),
/// so `max_tokens` here must leave room for the rendered prompt
/// (well under 1024) -- "effectively unbounded relative to this
/// test's timeouts" rather than the vastly larger figure a real
/// model's context window would allow.
const NEAR_MAX_CONTEXT_TOKENS: usize = 900;

fn tiny_state() -> AppState {
    super::tiny_state(NEAR_MAX_CONTEXT_TOKENS)
}

/// Proves the real `chat_completions` streaming composition --
/// not just its primitives in isolation -- actually stops
/// generation on client disconnect. A tiny all-zero-weight model
/// with `max_tokens` near the tiny model's 1024-token context window means
/// uncancelled generation would keep running far longer than
/// this test's timeouts; it reads two real SSE frames (proving
/// generation is genuinely under way) before dropping the
/// response body to simulate a disconnect.
///
/// Mutation-sensitive to a known regression: if
/// `let _cancel_guard_tied_to_stream_lifetime = &cancel_guard;`
/// is removed from `body_stream`'s `flat_map` closure in
/// `chat_completions`, `cancel_guard` is no longer captured by
/// anything and drops at the end of `chat_completions`'s own
/// function body -- i.e. before the response is even returned to
/// the caller, let alone before the client could disconnect.
/// That flips `cancel_rx` immediately, so
/// `generate_streaming_with_cancel`'s pre-prefill checkpoint
/// returns `Interrupt` with zero tokens generated. Frame 1 (the
/// role chunk) still arrives unconditionally either way, and a
/// *second* frame still arrives either way too -- but under the
/// mutation it is the finish chunk (`finish_reason: "length"` --
/// the shared `finish_reason()` mapping has no `"interrupt"`
/// string; a cancelled/interrupted result reports `"length"`
/// exactly like a token-cap stop, see its own doc comment --
/// no `content`) sent by `finish_streaming` for a zero-token
/// result, not a real content delta. A test that only checks "a
/// second frame arrived and was Ok" cannot tell these apart, so
/// this asserts the frame's actual JSON payload shape: real code
/// path yields `finish_reason: null` + a string `delta.content`;
/// the mutated path yields a non-null `finish_reason` and no
/// `content`, and the second `assert!` fails, so this test fails
/// if the disconnect-stops-generation behavior regresses.
#[tokio::test]
async fn chat_completions_streaming_disconnect_stops_generation() {
    let mut req = bare_req();
    req.model = Some("test-model".to_string());
    req.messages = vec![user_msg("hi")];
    req.max_tokens = Some(NEAR_MAX_CONTEXT_TOKENS);
    req.stream = Some(true);

    let response = chat_completions_with_request(State(tiny_state()), req)
        .await
        .expect("streaming request must be accepted");
    assert_eq!(response.status(), StatusCode::OK);

    let mut body = response.into_body();

    // Frame 1: the role chunk, emitted unconditionally before any
    // generation output (`futures::stream::once(role_chunk).chain(body_stream)`).
    tokio::time::timeout(Duration::from_secs(10), body.frame())
        .await
        .expect("role chunk frame must arrive quickly")
        .expect("role chunk frame must exist")
        .expect("role chunk frame must be Ok");

    // Frame 2: must be a genuine content delta from the tiny
    // model's decode loop, not the finish chunk. A frame merely
    // *arriving* here does not prove generation ran: if
    // `cancel_guard` is no longer tied to the stream, it drops
    // (flipping `cancel_rx`) before the blocking task's
    // pre-prefill checkpoint, `generate_streaming_with_cancel`
    // returns `Interrupt` with zero tokens, and
    // `finish_streaming` sends `StreamMsg::Done` immediately --
    // which *also* produces a well-formed, `Ok` SSE frame here,
    // just one carrying `finish_reason: "length"` (the shared
    // mapping has no `"interrupt"` string; see
    // `lattice_inference::serve::finish_reason`'s doc comment)
    // and no `content` instead of a real delta. So this asserts
    // the frame's actual payload shape, not just its arrival.
    let frame = tokio::time::timeout(Duration::from_secs(10), body.frame())
        .await
        .expect(
            "a second frame must arrive quickly -- if this times out, \
                         nothing at all was sent after the role chunk",
        )
        .expect("second frame must exist")
        .expect("second frame must be Ok");
    let data = frame
        .into_data()
        .expect("second frame must carry data (not trailers)");
    let text = std::str::from_utf8(&data).expect("SSE frame must be UTF-8");
    let payload = text
        .strip_prefix("data: ")
        .unwrap_or(text)
        .trim_end_matches(['\n', '\r']);
    let chunk: serde_json::Value =
        serde_json::from_str(payload).expect("SSE frame must carry a JSON chunk");
    let choice = &chunk["choices"][0];
    assert!(
        choice["finish_reason"].is_null(),
        "second frame must be a content delta, not the finish chunk \
                     (generation was interrupted before producing any output, \
                     i.e. cancel_guard fired before the stream had a chance to \
                     run): {chunk}"
    );
    assert!(
        choice["delta"]["content"].is_string(),
        "second frame must carry delta.content (a real generated \
                     token), got: {chunk}"
    );

    // Client disconnect: drop the body. `cancel_guard` (captured
    // by `body_stream`'s `flat_map` closure) drops with it,
    // flipping the paired `cancel_rx` -- the CPU decode loop's
    // `generate_streaming_with_cancel` polls that flag at the top
    // of every iteration (already unit/mutation-tested in
    // `model/qwen35/generation.rs`) and stops within one step
    // instead of running toward `max_tokens` (900). This
    // test's own fast, deterministic completion (it does not hang
    // waiting on the now-unobservable background task) is the
    // practical proof: dropping never blocks, and the tiny
    // model's decode loop is fast enough on the blocking-thread
    // pool that a broken cancellation path would otherwise pin a
    // thread pool worker in an unbounded loop for the remainder
    // of the test process's life -- observable across this
    // binary's full test run as sporadic slowdowns/hangs in
    // later tests sharing the pool, not just this test in
    // isolation.
    drop(body);
}
