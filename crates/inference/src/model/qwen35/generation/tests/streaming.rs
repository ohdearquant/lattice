//! Streaming generation with cancellation and observers, and token parity of the public prefill delegation.

use super::*;

// -----------------------------------------------------------------------
// generate_streaming_with_cancel mutation-sensitive tests (ADR-080 C2, #744)
// -----------------------------------------------------------------------
//
// Zero-weight model: greedy sampling always picks token 0, which decodes to
// the literal text "<unk>" -- a non-empty delta on every step, so both the
// pre-loop first-token emission and every decode-loop iteration produce a
// delta for `on_token` to observe. This gives fully deterministic
// cancellation-checkpoint counting without relying on random-weight output.

/// `should_cancel` returning `true` before the prefill pass starts (the
/// very first checkpoint) must short-circuit generation entirely: no
/// prefill, no sampling, `generated_tokens == 0`.
///
/// Mutation sensitivity: removing this checkpoint (or its early return)
/// makes generation fall through to prefill + sampling, producing
/// `generated_tokens > 0` and failing the assertion below.
#[test]
fn generate_streaming_with_cancel_true_before_prefill_returns_interrupt() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };
    // Cancel on the very first `should_cancel` call only (the pre-prefill
    // checkpoint), so this test pins THAT checkpoint specifically rather
    // than any-of-the-four checkpoints: if the pre-prefill check were
    // removed, the post-prefill checkpoint (call 2) would see `false` and
    // generation would run to completion instead of stopping at
    // `generated_tokens == 0`.
    let calls = std::cell::Cell::new(0usize);
    let result = model
        .generate_streaming_with_cancel(
            "a",
            &gen_cfg,
            |_delta| true,
            || {
                let n = calls.get() + 1;
                calls.set(n);
                n == 1
            },
        )
        .expect("cancelled-before-prefill generate must succeed");
    assert!(
        !result.stopped,
        "a caller cancellation is not an OpenAI stop condition"
    );
    assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(result.generated_tokens, 0);
    assert!(result.text.is_empty());
}

/// `should_cancel` returning `true` on its SECOND call only -- i.e. the
/// pre-prefill checkpoint (call 1) sees `false` and lets prefill run,
/// then the post-prefill checkpoint (call 2, immediately after prefill,
/// before sampling) sees `true` and must stop before ANY token is
/// sampled or emitted. This isolates the post-prefill checkpoint
/// specifically (ADR-080 C2):
/// the pre-prefill test above (`n == 1`) cannot tell the two checkpoints
/// apart, since removing the post-prefill one entirely still leaves that
/// test green (its cancellation already fires at checkpoint 1).
///
/// Mutation sensitivity: removing the post-prefill `if should_cancel()`
/// guard (the one right after the prefill-logits copy, before grammar
/// masking/sampling) lets generation fall through to sampling the first
/// token, producing `generated_tokens > 0`, non-empty `text`, and at
/// least one `on_token` callback -- failing all three assertions below.
#[test]
fn generate_streaming_with_cancel_true_after_prefill_returns_interrupt() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 5,
        temperature: 0.0,
        ..Default::default()
    };
    let calls = std::cell::Cell::new(0usize);
    let on_token_calls = std::cell::Cell::new(0usize);
    let result = model
        .generate_streaming_with_cancel(
            "a",
            &gen_cfg,
            |_delta| {
                on_token_calls.set(on_token_calls.get() + 1);
                true
            },
            || {
                let n = calls.get() + 1;
                calls.set(n);
                n == 2
            },
        )
        .expect("cancelled-after-prefill generate must succeed");
    assert!(
        !result.stopped,
        "a caller cancellation is not an OpenAI stop condition"
    );
    assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(
        result.generated_tokens, 0,
        "post-prefill cancellation must stop before any token is sampled"
    );
    assert!(result.text.is_empty());
    assert_eq!(
        on_token_calls.get(),
        0,
        "post-prefill cancellation must stop before on_token is ever called"
    );
}

/// `should_cancel` flipping to `true` at the top of a later decode-loop
/// iteration (fast path, no `stop_strings`) must stop generation before
/// the `max_new_tokens` cap is reached, keeping the tokens already emitted
/// before the flip.
///
/// Call count: checkpoint 1 (pre-prefill) = call 1, checkpoint 2
/// (post-prefill) = call 2, first decode-loop top = call 3. Flipping true
/// at call 3 means the loop never runs its body, so only the one
/// pre-loop token (emitted right after prefill, before the decode loop
/// starts) is generated.
///
/// Mutation sensitivity: removing the decode-loop's `should_cancel` check
/// lets generation run to `max_new_tokens` (10), failing
/// `generated_tokens == 1`.
#[test]
fn generate_streaming_with_cancel_mid_decode_stops_early_fast_path() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };
    let calls = std::cell::Cell::new(0usize);
    let result = model
        .generate_streaming_with_cancel(
            "a",
            &gen_cfg,
            |_delta| true,
            || {
                let n = calls.get() + 1;
                calls.set(n);
                n >= 3
            },
        )
        .expect("mid-decode cancel generate must succeed");
    assert!(!result.stopped);
    assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(
        result.generated_tokens, 1,
        "should_cancel flipping true at the first decode-loop checkpoint must stop \
             after exactly the one pre-loop token; got {}",
        result.generated_tokens
    );
}

/// `on_token` returning `false` on the very first (pre-loop) delta must
/// stop generation immediately, in the fast path (no `stop_strings`).
///
/// Mutation sensitivity: dropping the `if !on_token(&delta)` early return
/// after the pre-loop delta lets generation continue into the decode
/// loop, failing `generated_tokens == 1`.
#[test]
fn generate_streaming_with_cancel_on_token_false_stops_generation_fast_path() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };
    let result = model
        .generate_streaming_with_cancel("a", &gen_cfg, |_delta| false, || false)
        .expect("on_token-false generate must succeed");
    assert!(!result.stopped);
    assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(
        result.generated_tokens, 1,
        "on_token returning false on the very first delta must stop after exactly \
             one token; got {}",
        result.generated_tokens
    );
}

/// Same `on_token`-returns-`false` cancellation, but in the `stop_strings`
/// path (`StopStringMatcher`'s sink has no return value, so cancellation
/// is threaded through a captured `caller_interrupted` flag instead --
/// this test pins that alternate code path independently of the fast
/// path above).
///
/// Mutation sensitivity: dropping the interrupted check after the
/// pre-loop `check_initial_stop` call lets generation continue into the
/// decode loop, failing `generated_tokens == 1`.
#[test]
fn generate_streaming_with_cancel_on_token_false_stops_generation_stop_string_path() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 10,
        temperature: 0.0,
        stop_strings: vec!["ZZZZ".to_string()],
        ..Default::default()
    };
    let result = model
        .generate_streaming_with_cancel("a", &gen_cfg, |_delta| false, || false)
        .expect("on_token-false generate (stop-string path) must succeed");
    assert!(!result.stopped);
    assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(
        result.generated_tokens, 1,
        "on_token returning false on the very first delta must stop after exactly \
             one token in the stop-string path too; got {}",
        result.generated_tokens
    );
}

/// Same mid-decode `should_cancel` cancellation as the fast-path test
/// above, but in the `stop_strings` path -- pins that the decode loop's
/// `should_cancel` checkpoint is present in both branches, not just the
/// fast path.
///
/// Mutation sensitivity: removing the decode-loop's `should_cancel` check
/// in the `stop_strings` branch lets generation run to `max_new_tokens`
/// (10), failing `generated_tokens == 1`.
#[test]
fn generate_streaming_with_cancel_mid_decode_stops_early_stop_string_path() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 10,
        temperature: 0.0,
        stop_strings: vec!["ZZZZ".to_string()],
        ..Default::default()
    };
    let calls = std::cell::Cell::new(0usize);
    let result = model
        .generate_streaming_with_cancel(
            "a",
            &gen_cfg,
            |_delta| true,
            || {
                let n = calls.get() + 1;
                calls.set(n);
                n >= 3
            },
        )
        .expect("mid-decode cancel generate (stop-string path) must succeed");
    assert!(!result.stopped);
    assert_eq!(result.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(
        result.generated_tokens, 1,
        "should_cancel flipping true at the first decode-loop checkpoint must stop \
             after exactly the one pre-loop token (stop-string path); got {}",
        result.generated_tokens
    );
}

// -----------------------------------------------------------------------
// generate_streaming_with_observer / RawGenEvent mutation-sensitive tests
// -----------------------------------------------------------------------

/// `RawGenEvent::PrefillEnd` must fire exactly once, before the first
/// `RawGenEvent::RawToken`, and every subsequent `RawToken` index must be
/// monotonically increasing 1..=generated_tokens -- the raw prefill/decode
/// boundary the CPU flagship smoke harness measures TTFT and decode
/// throughput off, independent of the text-delta callback.
///
/// Mutation sensitivity: moving the `on_raw_event(RawGenEvent::PrefillEnd)`
/// call to after the first token is pushed (the pre-fix bug this test
/// guards -- marking prefill_end off the first confirmed generated token
/// instead of the true prefill/decode boundary) makes `events.first()`
/// a `RawToken` instead of `PrefillEnd`, failing the first assertion.
/// Removing any of the three `on_raw_event(RawGenEvent::RawToken { .. })`
/// call sites (pre-loop first token, fast-path decode loop, stop-string
/// decode loop) drops an index from the collected sequence, failing the
/// monotonic-range assertion.
#[test]
fn raw_observer_prefill_end_precedes_first_raw_token_monotonic_index() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let events = std::cell::RefCell::new(Vec::<RawGenEvent>::new());
    let result = model
        .generate_streaming_with_observer(
            "a",
            &gen_cfg,
            |_delta| true,
            || false,
            |evt| events.borrow_mut().push(evt),
        )
        .expect("generation must succeed");

    let events = events.into_inner();
    assert_eq!(
        events.first(),
        Some(&RawGenEvent::PrefillEnd),
        "the very first raw event must be PrefillEnd -- prefill completed and logits \
             are ready before any token is sampled; got {events:?}"
    );
    let token_indices: Vec<usize> = events
        .iter()
        .skip(1)
        .map(|e| match e {
            RawGenEvent::RawToken { index } => *index,
            RawGenEvent::PrefillEnd => {
                panic!("PrefillEnd must fire exactly once, at the very start: {events:?}")
            }
        })
        .collect();
    assert_eq!(result.generated_tokens, 3);
    assert_eq!(
        token_indices,
        vec![1, 2, 3],
        "RawToken events must be one per generated token, in generation order, with a \
             monotonically increasing 1-based index equal to generated_tokens so far"
    );
}

/// `RawGenEvent::PrefillEnd` must fire before `sample_token` is entered
/// for the first time, not merely before the `RawToken` *callback*: the
/// test above cannot distinguish "PrefillEnd fired before sampling" from
/// "PrefillEnd fired after sampling but before the RawToken push",
/// because both orderings produce the same `[PrefillEnd, RawToken,
/// RawToken, RawToken]` event sequence. This test uses the
/// `test_record_first_sample_entry` seam, planted at the exact point
/// `sample_token` is called for the prefill-derived first token, to
/// check that ordering directly.
///
/// Mutation sensitivity: moving `on_raw_event(RawGenEvent::PrefillEnd)`
/// to immediately after `generated_ids.push(next_id)` (still before the
/// `RawToken` callback) leaves the event-order test above green, but
/// makes this test's `on_raw_event` closure observe `PrefillEnd` only
/// *after* `test_record_first_sample_entry` already ran, so
/// `test_take_first_sample_saw_prefill_end()` returns `Some(false)`
/// instead of `Some(true)` and the assertion below fails.
#[test]
fn raw_observer_prefill_end_precedes_first_sample_not_just_first_raw_token() {
    test_reset_sample_seam();
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let result = model
        .generate_streaming_with_observer(
            "a",
            &gen_cfg,
            |_delta| true,
            || false,
            |evt| {
                if evt == RawGenEvent::PrefillEnd {
                    test_mark_prefill_end_seen();
                }
            },
        )
        .expect("generation must succeed");

    assert_eq!(result.generated_tokens, 3);
    assert_eq!(
        test_take_first_sample_saw_prefill_end(),
        Some(true),
        "PrefillEnd must have already fired by the moment sample_token is entered for \
             the prefill-derived first token -- not merely before the RawToken callback, \
             which a PrefillEnd emitted after sampling (but before the RawToken push) would \
             also satisfy while silently including sampling time in the reported prefill \
             interval (#882)"
    );
}

/// One raw-token event fires per generated token even when the
/// text-delta stream buffers an incomplete multi-byte UTF-8 sequence and
/// therefore does NOT call `on_token` for that step at all: measuring
/// prefill/decode boundaries off text deltas is not equivalent to
/// measuring off raw sampled tokens, and can *lag* the true boundary by
/// one or more tokens.
///
/// Token id 0 is defined (via a custom tiny tokenizer vocab) to decode to
/// the single raw byte `0xC2` -- the lead byte of a 2-byte UTF-8 sequence
/// (`U+0080..=U+07FF`), which alone is an incomplete codepoint that
/// `IncrementalDetokenizer` must buffer rather than emit (verified
/// directly: pushing `0xC2` once in isolation yields an empty delta).
/// The all-zero-weight tiny model always greedily samples token 0 (tied
/// logits, first-wins), so the very *first* generated token -- the
/// prefill-derived one, pushed into a fresh, empty detokenizer buffer --
/// is guaranteed to buffer rather than emit: the fast decode path's
/// `StopMode::Disabled` branch of `stop_check` skips calling `on_token`
/// entirely whenever the resulting delta is empty (`generation.rs`:
/// `if delta.is_empty() { return StopCheckOutcome::Continue; }`).
///
/// This test snapshots `on_token`'s cumulative call count at the instant
/// each `RawToken` event fires, rather than comparing final totals: a
/// trailing `detok.finish()` flush (emitted once after the decode loop
/// ends, for whatever incomplete bytes never got a chance to complete)
/// also calls `on_token`, so the *final* on_token-call total is not a
/// reliable signal here -- it can coincidentally equal
/// `generated_tokens` even though a mid-generation step was buffered.
/// The per-event snapshot sidesteps that confound entirely: at the
/// moment `RawToken { index: 1 }` fires (in the observer, mid
/// generation, before the decode loop or any flush has run), `on_token`
/// must not yet have been called even once.
///
/// Mutation sensitivity: reverting to the pre-fix design (emitting
/// `token_available` from inside `on_token` instead of from
/// `on_raw_event`) would silently miss this buffered first token
/// entirely (its phase event would fire late, on whatever later step
/// finally produces a non-empty delta, or not at all if generation ends
/// first) -- this test's first-event snapshot fails immediately on that
/// regression, and the raw-index-sequence assertion fails independently
/// if any `on_raw_event(RawGenEvent::RawToken { .. })` call site is
/// removed.
#[test]
fn raw_observer_fires_before_on_token_for_buffered_incomplete_utf8_first_token() {
    const UTF8_LEAD_BYTE_TOK_JSON: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"Â":0,"<unk>":1,"a":2},"merges":[]}
}"#;
    let model = build_tiny_zero_model_tok(UTF8_LEAD_BYTE_TOK_JSON);

    // Independent confirmation that token id 0 alone is genuinely
    // incomplete UTF-8, not just asserted by comment: pushing it once
    // into a fresh detokenizer must yield an empty delta.
    let mut probe = IncrementalDetokenizer::new();
    assert_eq!(
        probe.push(model.tokenizer(), 0),
        "",
        "token id 0 (raw byte 0xC2) must be an incomplete UTF-8 lead byte on its own -- \
             this test's premise depends on it"
    );

    let gen_cfg = GenerateConfig {
        max_new_tokens: 4,
        temperature: 0.0,
        ..Default::default()
    };
    let raw_indices = std::cell::RefCell::new(Vec::<usize>::new());
    let on_token_calls = std::cell::Cell::new(0usize);
    // Snapshot of `on_token_calls` at the instant each RawToken event
    // fires -- index i (0-based) corresponds to RawToken{index: i+1}.
    let snapshots_at_raw_event = std::cell::RefCell::new(Vec::<usize>::new());
    let result = model
        .generate_streaming_with_observer(
            "a",
            &gen_cfg,
            |_delta: &str| {
                on_token_calls.set(on_token_calls.get() + 1);
                true
            },
            || false,
            |evt| {
                if let RawGenEvent::RawToken { index } = evt {
                    raw_indices.borrow_mut().push(index);
                    snapshots_at_raw_event
                        .borrow_mut()
                        .push(on_token_calls.get());
                }
            },
        )
        .expect("generation over an incomplete-lead-byte vocab must still succeed");

    assert_eq!(
        result.generated_tokens, 4,
        "token id 0 (eos_token_id is 96 on this tiny model, never sampled) never \
             satisfies should_stop_token, so all 4 requested tokens must be generated"
    );
    assert_eq!(
        raw_indices.into_inner(),
        vec![1, 2, 3, 4],
        "one monotonically indexed RawToken event per generated token, regardless of \
             detokenizer buffering"
    );
    assert_eq!(
        snapshots_at_raw_event.borrow()[0],
        0,
        "on_token must not have been called yet at the moment the FIRST RawToken event \
             fires -- the prefill-derived first token's delta is buffered (empty) by the \
             incomplete-UTF-8 lead byte, so a phase-event trace measured off on_token would \
             have missed or mis-timed this token entirely; got {} prior on_token calls",
        snapshots_at_raw_event.borrow()[0]
    );
}

/// Id 0 decodes to a newline and then the lone lead byte 0xC2, which the
/// detokenizer holds; the zero model samples it first, and because it ends a
/// line a reasoning budget of 1 forces `</think>` (a stop token) on the next
/// step. The held byte must reach both the returned and the streamed text.
#[test]
fn eos_flushes_incomplete_utf8_tail_in_streaming_and_nonstreaming() {
    const TOK_JSON: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,
  "added_tokens":[{"id":7,"content":"</think>","special":false}],
  "normalizer":null,
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
  "post_processor":null,
  "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
  "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,
    "end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
    "vocab":{"ĊÂ":0,"<unk>":1,"a":2},"merges":[]}
}"#;
    let model = build_tiny_zero_model_tok(TOK_JSON);
    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        enable_thinking: true,
        reasoning_budget: Some(1),
        stop_token_ids: vec![7],
        stop_strings: vec!["never".to_string()],
        ..Default::default()
    };

    let nonstreaming = model.generate("a", &gen_cfg).unwrap();
    let mut streamed_text = String::new();
    let streaming = model
        .generate_streaming("a", &gen_cfg, |delta| streamed_text.push_str(delta))
        .unwrap();

    assert_eq!(nonstreaming.stop_reason, Some(StopReason::Eos));
    assert_eq!(streaming.stop_reason, Some(StopReason::Eos));
    assert_eq!(nonstreaming.token_ids, vec![0]);
    assert_eq!(nonstreaming.text, "\n\u{fffd}");
    assert_eq!(nonstreaming.text, streaming.text);
    assert_eq!(streaming.text, streamed_text);
}

/// ADR-092 arm 1 regression: the streaming driver route's natural-end tail
/// flush must run even when the incremental detokenizer holds nothing back.
///
/// `StopStringMatcher::push` holds back `max_stop - 1` bytes on every call so a
/// stop string spanning a delta boundary is never streamed prematurely, and
/// `finish` is what releases them. Those bytes are held by the MATCHER, not by
/// the detokenizer, so gating the natural-end flush on a non-empty
/// `detok.finish()` drops them from both `text` and the caller's stream on every
/// generation that ends on a clean UTF-8 boundary with `stop_strings` set --
/// which is the ordinary case, not an edge case.
///
/// Mutation sensitivity: wrapping `policy.finish_stop` in `decoder::driver::run`
/// in an `if !tail.is_empty()` guard truncates `streaming.text` by up to
/// `max_stop - 1` trailing bytes, failing both assertions below.
#[test]
fn streaming_flushes_held_back_stop_prefix_bytes_when_the_detokenizer_tail_is_empty() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        stop_strings: vec!["never".to_string()],
        ..Default::default()
    };

    let nonstreaming = model
        .generate("a", &gen_cfg)
        .expect("non-streaming generate must succeed");
    let mut streamed_text = String::new();
    let streaming = model
        .generate_streaming("a", &gen_cfg, |delta| streamed_text.push_str(delta))
        .expect("streaming generate must succeed");

    assert_eq!(
        streaming.text, nonstreaming.text,
        "a stop string that never matches must not truncate the streamed output"
    );
    assert_eq!(
        streaming.text, streamed_text,
        "the returned text must equal the concatenation of every delta"
    );
}

/// Stop-string truncation must drop `token_logprobs` entries whose decoded
/// text didn't fully survive the truncation, not leave them describing
/// bytes the caller never sees in `text` (#620:
/// `build_choice_logprobs` in lattice.rs builds `logprobs.content` directly
/// off `token_logprobs`, so a stale entry there silently corrupts the
/// OpenAI-compatible response).
///
/// Zero-weight model → every sampled token is id 0, which decodes to the
/// literal text "<unk>" (all 5 chars sit in the byte-level decoder's
/// printable range, so `append_token_bytes` skips none of them as
/// "special"). Two tokens concatenate to "<unk><unk>" (10 bytes); the stop
/// string "k><unk" (6 bytes) first matches at byte 3 — BEFORE the token
/// boundary at byte 5, so the match also clips into the FIRST token's own
/// trailing bytes, not just the second token's. Both entries' text is
/// therefore only partially retained, and both must be dropped — this is
/// the multi-token-span case `truncate_token_logprobs_to_retained_text`
/// exists to handle, not just "drop the most recent entry".
///
/// Mutation sensitivity: without the truncation call, `token_logprobs`
/// keeps both entries (len 2) even though `text` is only "<un" (3 bytes) —
/// neither entry's full text is representable in the truncated output.
/// With the fix, `token_logprobs` is empty.
#[test]
fn stop_string_truncation_drops_stale_token_logprobs() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 10,
        temperature: 0.0,
        logprobs: Some(0),
        stop_strings: vec!["k><unk".to_string()],
        ..Default::default()
    };
    let result = model
        .generate("a", &gen_cfg)
        .expect("stop-string generate must succeed");

    assert_eq!(
        result.text, "<un",
        "stop string must truncate at the first match; got {:?}",
        result.text
    );
    assert_eq!(
        result.generated_tokens, 2,
        "both tokens were sampled before the match completed (can't un-generate); \
             got {}",
        result.generated_tokens
    );
    assert!(
        result.token_logprobs.is_empty(),
        "both tokens' text was only partially retained after truncation, so both \
             logprobs entries must be dropped; got {} entries",
        result.token_logprobs.len()
    );
}

/// PR #787: with `gen_cfg.logprobs`
/// left at its default (`None`), `DecodePolicy`'s logprobs gate (the
/// `if let Some(top_n) = self.logprobs` guard `init_with_metadata` and
/// `transition_inner` both apply in production, driven via
/// `decoder::driver::run`) must be a true no-op -- `token_logprobs` stays
/// empty for the
/// whole generation, not just for the truncated-text case the test above
/// covers. Replaces `sampling.rs`'s now-removed
/// `test_record_logprob_noop_when_not_requested`: that free function no
/// longer exists (its mutation into `crate::sampling` was the whole
/// point of the fix), so its behavioral contract is asserted here, at
/// the only place that can still exercise it.
///
/// Mutation sensitivity: removing the `let Some(top_n) = self.logprobs
/// else { return; };` guard inside `DecodePolicy::record_logprob` makes
/// every token here get an (incorrect) `TokenLogprob` entry, failing
/// `token_logprobs.is_empty()`.
#[test]
fn decode_policy_record_logprob_noop_when_not_requested() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        logprobs: None,
        ..Default::default()
    };
    let result = model
        .generate("a", &gen_cfg)
        .expect("plain generate must succeed");
    assert!(
        result.token_logprobs.is_empty(),
        "logprobs: None must record nothing across the whole generation \
             (prefill token via init_with_metadata, decode tokens via \
             transition_with_metadata); got {} entries",
        result.token_logprobs.len()
    );
}

/// PR #787: isolates
/// `DecodePolicy::init_with_metadata`'s first-step logprob ownership from
/// `transition_with_metadata`'s per-step logprob ownership, which
/// `transition_records_one_logprob_per_generated_token` below already
/// covers but does not itself distinguish. With `max_new_tokens: 1`,
/// `policy.cap()` is 1, so `decoder::driver::run`'s `for _ in 1..cap`
/// decode loop body never executes and `transition_with_metadata` is
/// never called at all -- the entire generation consists of the one
/// prefill-derived token `init_with_metadata` records. If that one
/// `TokenLogprob` entry exists, it can only have come from
/// `init_with_metadata`.
///
/// Mutation sensitivity: removing the
/// `record_metadata(first_emitted_id, top_n)` call inside
/// `DecodePolicy::init_with_metadata` makes `token_logprobs` come back
/// empty while `token_ids` still has 1 entry -- this test fails with a
/// length mismatch (`0 != 1`) instead of passing.
#[test]
fn init_records_the_prefill_tokens_logprob_before_any_transition_call() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        logprobs: Some(0),
        ..Default::default()
    };
    let result = model
        .generate("a", &gen_cfg)
        .expect("single-token generate must succeed");

    assert_eq!(
        result.generated_tokens, 1,
        "max_new_tokens: 1 must generate exactly the prefill-derived \
             first token and never run the decode loop; got {}",
        result.generated_tokens
    );
    assert_eq!(
        result.token_logprobs.len(),
        1,
        "the sole generated token's logprob must be recorded by \
             DecodePolicy::init_with_metadata alone (transition_with_metadata \
             is never called when max_new_tokens == 1); got {} entries",
        result.token_logprobs.len()
    );
    assert_eq!(
        result.token_logprobs[0].token_id, result.token_ids[0],
        "the recorded logprob entry must describe the actual first token"
    );
}

/// PR #787: `DecodePolicy::transition` owns
/// `record_logprob` internally now (the constituent method is private,
/// only reachable through `transition`). This asserts the positive case
/// the truncation test above does not: with no stop string to truncate
/// anything, every generated token gets exactly one `TokenLogprob` entry,
/// and its `token_id` matches the corresponding `token_ids` entry.
///
/// Mutation sensitivity: commenting out the `self.record_logprob(...)`
/// call inside `DecodePolicy::transition` makes `token_logprobs` come
/// back empty while `token_ids` still has 3 entries -- this test fails
/// with a length mismatch (`0 != 3`) instead of passing.
#[test]
fn transition_records_one_logprob_per_generated_token() {
    let model = build_tiny_zero_model();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        logprobs: Some(0),
        ..Default::default()
    };
    let result = model
        .generate("a", &gen_cfg)
        .expect("plain generate must succeed");

    assert_eq!(
        result.token_logprobs.len(),
        result.token_ids.len(),
        "every generated token must get exactly one TokenLogprob entry \
             when logprobs is requested and nothing truncates the output; \
             got {} logprobs for {} tokens",
        result.token_logprobs.len(),
        result.token_ids.len()
    );
    for (i, (logprob, &token_id)) in result
        .token_logprobs
        .iter()
        .zip(result.token_ids.iter())
        .enumerate()
    {
        assert_eq!(
            logprob.token_id, token_id,
            "token_logprobs[{i}] must describe the token actually emitted \
                 at that position"
        );
    }
}

// -------------------------------------------------------------------
// Public-prefill-delegation token parity (perf_hunt public-prefill
// experiment). Requires a real dense Qwen3.5 checkpoint; set
// LATTICE_INFERENCE_MODEL_DIR to a safetensors directory (e.g.
// ~/.lattice/models/qwen3.5-0.8b). Ignored by default so CI
// and plain `cargo test` runs never depend on local model files.
// -------------------------------------------------------------------

/// Runs greedy generation for `prompt` with both the pre-delegation
/// serial prefill path and the batched-prefill delegation path on the
/// same loaded model, asserting the generated token ids are identical.
/// Serialized on `SERIAL_PREFILL_TEST_LOCK` because `FORCE_SERIAL_PREFILL`
/// is process-global.
fn assert_batched_prefill_matches_serial(model: &Qwen35Model, prompt: &str, max_new_tokens: usize) {
    let _guard = SERIAL_PREFILL_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let gen_cfg = crate::generation::GenerateConfig {
        max_new_tokens,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    FORCE_SERIAL_PREFILL.store(true, std::sync::atomic::Ordering::SeqCst);
    let serial = model.generate(prompt, &gen_cfg);
    FORCE_SERIAL_PREFILL.store(false, std::sync::atomic::Ordering::SeqCst);
    let batched = model.generate(prompt, &gen_cfg);

    let serial = serial.expect("serial-prefill generate must succeed");
    let batched = batched.expect("batched-prefill generate must succeed");

    assert_eq!(
        serial.token_ids, batched.token_ids,
        "batched-prefill delegation changed generated token ids for prompt {prompt:?}: \
             serial={:?} batched={:?}",
        serial.token_ids, batched.token_ids
    );
    assert_eq!(
        serial.text, batched.text,
        "batched-prefill delegation changed decoded text for prompt {prompt:?}"
    );
    assert_eq!(serial.stop_reason, batched.stop_reason);
    assert_eq!(serial.stopped, batched.stopped);
}

#[test]
#[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR"]
fn generate_batched_prefill_matches_serial_for_seeded_dense_prompt() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
        .expect("dense Qwen3.5 model should load successfully");

    // 20 greedy tokens from a fixed prompt, per the perf_hunt experiment ask.
    assert_batched_prefill_matches_serial(
        &model,
        "The quick brown fox jumps over the lazy dog. In a distant future,",
        20,
    );
}

#[test]
#[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR"]
fn generate_streaming_batched_prefill_matches_nonstreaming_text() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
        .expect("dense Qwen3.5 model should load successfully");

    let gen_cfg = crate::generation::GenerateConfig {
        max_new_tokens: 20,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let prompt = "The quick brown fox jumps over the lazy dog. In a distant future,";

    let non_streaming = model
        .generate(prompt, &gen_cfg)
        .expect("non-streaming generate must succeed");

    let mut streamed_text = String::new();
    let streaming = model
        .generate_streaming(prompt, &gen_cfg, |delta| streamed_text.push_str(delta))
        .expect("streaming generate must succeed");

    assert_eq!(
        non_streaming.token_ids, streaming.token_ids,
        "streaming batched-prefill delegation diverged from non-streaming"
    );
    assert_eq!(non_streaming.text, streaming.text);
    assert_eq!(non_streaming.text, streamed_text);
}

#[test]
#[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR"]
fn generate_batched_prefill_matches_serial_across_prompt_lengths() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
        .expect("dense Qwen3.5 model should load successfully");

    for words in [8usize, 64, 256] {
        let prompt = "hello ".repeat(words);
        assert_batched_prefill_matches_serial(&model, prompt.trim_end(), 5);
    }
}

/// A/B time-to-first-token sweep: serial (pre-delegation) prefill vs.
/// batched-prefill delegation, same loaded model, back-to-back, for a
/// range of prompt lengths. `max_new_tokens: 1` isolates prefill + first
/// sample. Prints `ms` per path so the perf_hunt experiment report can
/// quote the raw numbers; not a pass/fail gate (that's the parity tests
/// above) — run with `--release --features f16 -- --ignored --nocapture`.
#[test]
#[ignore = "requires local Qwen3.5 checkpoint: set LATTICE_INFERENCE_MODEL_DIR; run --release"]
fn public_prefill_ttft_ab_sweep() {
    let model_dir = crate::test_support::require_checkpoint_dir("LATTICE_INFERENCE_MODEL_DIR");
    let model = Qwen35Model::from_safetensors(std::path::Path::new(&model_dir))
        .expect("dense Qwen3.5 model should load successfully");

    let _guard = SERIAL_PREFILL_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let gen_cfg = crate::generation::GenerateConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    println!("words\tprompt_tokens\tserial_ms\tbatched_ms\tspeedup");
    for words in [64usize, 512, 2000] {
        let prompt = "hello ".repeat(words);
        let prompt = prompt.trim_end();

        // Warm the model/tokenizer once outside the timed region.
        FORCE_SERIAL_PREFILL.store(false, std::sync::atomic::Ordering::SeqCst);
        let _ = model.generate(prompt, &gen_cfg).unwrap();

        FORCE_SERIAL_PREFILL.store(true, std::sync::atomic::Ordering::SeqCst);
        let t0 = std::time::Instant::now();
        let serial = model.generate(prompt, &gen_cfg).expect("serial generate");
        let serial_ms = t0.elapsed().as_secs_f64() * 1000.0;

        FORCE_SERIAL_PREFILL.store(false, std::sync::atomic::Ordering::SeqCst);
        let t0 = std::time::Instant::now();
        let batched = model.generate(prompt, &gen_cfg).expect("batched generate");
        let batched_ms = t0.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(
            serial.token_ids, batched.token_ids,
            "TTFT sweep: token mismatch at words={words}"
        );

        println!(
            "{words}\t{}\t{serial_ms:.1}\t{batched_ms:.1}\t{:.3}",
            serial.prompt_tokens,
            serial_ms / batched_ms.max(1e-6),
        );
    }
}
