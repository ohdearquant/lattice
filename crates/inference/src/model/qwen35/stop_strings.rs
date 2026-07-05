//! Shared string-level stop-sequence matcher for Qwen3.5 generation.
//!
//! Both the CPU path (`generation.rs`) and the Metal path (`forward::metal_qwen35`)
//! call the same [`StopStringMatcher`] and [`earliest_stop_match`] so that stop-string
//! enforcement has CPU-identical semantics everywhere: incremental scan over decoded
//! text, a held-back partial suffix for spans that could still complete a stop string
//! across a token boundary, UTF-8-safe emission (never split a codepoint), and the
//! matched stop text excluded from returned/streamed output.

/// Returns the smallest byte index in `haystack` at which any stop string begins,
/// or `None` if no stop string is present.
pub(crate) fn earliest_stop_match(haystack: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter_map(|s| haystack.find(s.as_str())).min()
}

/// Same as [`earliest_stop_match`] but only scans `haystack[start..]`, returning
/// an index still relative to the start of `haystack`.
///
/// `start` must land on a UTF-8 char boundary (callers derive it via
/// [`floor_char_boundary`]); `str::find` is not called with `start` itself but
/// with an already-sliced `&str`, so an off-boundary `start` panics rather than
/// silently corrupting the search.
///
/// Callers rely on a key correctness property to make bounding the scan safe:
/// any stop-string occurrence that lies *entirely* before `start` in the
/// growing `haystack` must already have been found by an earlier call at a
/// smaller `start` (because `haystack` is only ever appended to, never
/// mutated in its already-scanned prefix). Bounding the scan therefore
/// changes which bytes get re-examined on each call, not which occurrences
/// are ever found across the sequence of calls.
pub(crate) fn earliest_stop_match_from(
    haystack: &str,
    stops: &[String],
    start: usize,
) -> Option<usize> {
    let start = start.min(haystack.len());
    stops
        .iter()
        .filter_map(|s| haystack[start..].find(s.as_str()).map(|p| p + start))
        .min()
}

/// Rounds `idx` down to the nearest UTF-8 char boundary in `s` (never past `s.len()`).
///
/// `str::floor_char_boundary` is unstable; this is the stable equivalent used
/// wherever a byte offset computed from `max_stop_len` must be turned into a
/// safe slicing point.
pub(crate) fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Stateful hold-back streamer for string-level stops (used by streaming generation
/// on both the CPU and Metal paths).
///
/// Maintains a `full` string of all decoded text so far and an `emitted` cursor. After each
/// push it:
/// 1. Checks `full` for the earliest stop-string match.
///    - If found: emits `full[emitted..hit]` via `sink`, truncates `full` to `hit`, sets
///      `emitted = full.len()`, marks `stopped = true`, returns `true`.
/// 2. If no match: emits the "safe" prefix `full[emitted..safe]` where
///    `safe = full.len() - (max_stop - 1)`, clamped to a UTF-8 char boundary, returns `false`.
///
/// `finish` is called once after the decode loop ends (natural stop / EOS / token cap).
/// If `stopped` is already true it is a no-op; otherwise it appends the `detok.finish()` tail,
/// checks for a stop in the result, and emits whatever is safe.
pub(crate) struct StopStringMatcher<'a> {
    full: String,
    emitted: usize,
    max_stop: usize,
    stops: &'a [String],
    stopped: bool,
}

impl<'a> StopStringMatcher<'a> {
    pub(crate) fn new(stops: &'a [String]) -> Self {
        let max_stop = stops.iter().map(String::len).max().unwrap_or(1);
        Self {
            full: String::new(),
            emitted: 0,
            max_stop,
            stops,
            stopped: false,
        }
    }

    /// Append `delta` to the accumulated text, emit newly-safe text via `sink`.
    ///
    /// Returns `true` when a stop string was found (caller must break the decode loop).
    ///
    /// `delta` may be empty — `IncrementalDetokenizer::push` returns `""` while
    /// buffering a split UTF-8 scalar, and `append_token_bytes` silently drops
    /// characters outside the GPT-2 byte decoder (e.g. special tokens), so an
    /// empty delta is a normal occurrence, not just a boundary case. The match
    /// check below must still run on an empty delta: an empty stop string
    /// (`stops == [""]`) matches `full` at byte 0 unconditionally via
    /// `str::find("")`, and that match must fire on the very first generated
    /// token even when that token's delta happens to be empty. Skipping the
    /// scan here (as the previous `if delta.is_empty() { return false; }`
    /// early-return did) silently swallowed that first-token stop.
    pub(crate) fn push(&mut self, delta: &str, sink: &mut impl FnMut(&str)) -> bool {
        // Any match entirely within `full[..prev_len]` would already have been
        // found by a prior call (see `earliest_stop_match_from`'s doc comment),
        // so only the suffix that could contain a NEW match — the previous
        // held-back window plus the freshly appended delta — needs rescanning.
        // Bounds per-push work to O(max_stop + delta.len()) instead of
        // O(full.len()), which is what made `push` quadratic in total
        // generated bytes across a long absent-stop generation.
        let prev_len = self.full.len();
        if !delta.is_empty() {
            self.full.push_str(delta);
        }
        let search_start = floor_char_boundary(
            &self.full,
            prev_len.saturating_sub(self.max_stop.saturating_sub(1)),
        );

        if let Some(hit) = earliest_stop_match_from(&self.full, self.stops, search_start) {
            let slice = &self.full[self.emitted..hit];
            if !slice.is_empty() {
                sink(slice);
            }
            self.full.truncate(hit);
            // Advance emitted so the post-loop flush is a no-op.
            self.emitted = self.full.len();
            self.stopped = true;
            return true;
        }

        // Emit safe prefix: hold back (max_stop - 1) bytes so a stop string that
        // starts before the end of the current delta is never streamed prematurely.
        let mut safe = self
            .full
            .len()
            .saturating_sub(self.max_stop.saturating_sub(1));
        safe = safe.max(self.emitted);
        while safe > self.emitted && !self.full.is_char_boundary(safe) {
            safe -= 1;
        }
        if safe > self.emitted {
            sink(&self.full[self.emitted..safe]);
            self.emitted = safe;
        }
        false
    }

    /// Natural-end flush. `tail` is `detok.finish()`.
    ///
    /// No-op when already stopped (a stop string was found in the decode loop).
    /// Otherwise appends `tail`, checks for a late stop, and emits whatever remains.
    pub(crate) fn finish(&mut self, tail: &str, sink: &mut impl FnMut(&str)) {
        if self.stopped {
            return;
        }
        if !tail.is_empty() {
            self.full.push_str(tail);
        }
        // A stop string could complete inside the tail bytes.
        if let Some(hit) = earliest_stop_match(&self.full, self.stops) {
            let slice = &self.full[self.emitted..hit];
            if !slice.is_empty() {
                sink(slice);
            }
            self.full.truncate(hit);
            self.emitted = self.full.len();
            self.stopped = true;
            return;
        }
        // No stop found: emit everything remaining.
        if self.emitted < self.full.len() {
            sink(&self.full[self.emitted..]);
            self.emitted = self.full.len();
        }
    }

    /// Whether a stop string has matched so far.
    pub(crate) fn stopped(&self) -> bool {
        self.stopped
    }

    /// Consume the streamer and return the final (possibly truncated) text.
    pub(crate) fn into_text(self) -> String {
        self.full
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // earliest_stop_match
    // -----------------------------------------------------------------------

    #[test]
    fn earliest_stop_match_single_present() {
        assert_eq!(
            earliest_stop_match("hello world", &["world".to_string()]),
            Some(6)
        );
    }

    #[test]
    fn earliest_stop_match_no_match() {
        assert_eq!(
            earliest_stop_match("hello world", &["foo".to_string()]),
            None
        );
    }

    #[test]
    fn earliest_stop_match_multiple_returns_earliest() {
        // "world" at index 6, "lo" at index 3 — should return 3
        assert_eq!(
            earliest_stop_match("hello world", &["world".to_string(), "lo".to_string()]),
            Some(3)
        );
    }

    #[test]
    fn earliest_stop_match_at_index_zero() {
        assert_eq!(
            earliest_stop_match("stopword rest", &["stop".to_string()]),
            Some(0)
        );
    }

    #[test]
    fn earliest_stop_match_multibyte_utf8() {
        // "界" is a 3-byte UTF-8 codepoint; it starts at byte 3 in "世界hello".
        assert_eq!(
            earliest_stop_match("世界hello", &["界".to_string()]),
            Some(3)
        );
    }

    #[test]
    fn earliest_stop_match_empty_stops() {
        // No stops configured → None, even for non-empty haystack.
        assert_eq!(earliest_stop_match("hello", &[]), None);
    }

    // -----------------------------------------------------------------------
    // StopStringMatcher — regression tests for the two streaming bugs
    // -----------------------------------------------------------------------

    /// BUG 1 regression: stop split across deltas must NOT double-emit the pre-stop tail.
    ///
    /// deltas: ["hel", "lo W", "orld!"], stop="World"
    #[test]
    fn stop_streamer_stop_split_across_deltas_no_double_emit() {
        let stops = vec!["World".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut all_emitted: Vec<String> = Vec::new();

        // delta 1: "hel" — no stop, safe prefix held back
        let stopped1 = streamer.push("hel", &mut |s| all_emitted.push(s.to_string()));
        assert!(!stopped1);

        // delta 2: "lo W" — "hello W" so far, "World" not yet complete, some safe bytes emitted
        let stopped2 = streamer.push("lo W", &mut |s| all_emitted.push(s.to_string()));
        assert!(!stopped2);

        // delta 3: "orld!" — "hello World!" now, stop "World" at byte 6
        let stopped3 = streamer.push("orld!", &mut |s| all_emitted.push(s.to_string()));
        assert!(stopped3, "stop should be detected on third delta");

        let concatenated = all_emitted.join("");
        // The text before "World" is "hello " (6 bytes). Must appear EXACTLY ONCE.
        assert_eq!(
            concatenated, "hello ",
            "emitted concat must equal pre-stop text exactly once (BUG 1 regression)"
        );
        assert_eq!(streamer.into_text(), "hello ");
    }

    /// Stop in the very first delta: nothing before it, into_text() is empty.
    #[test]
    fn stop_streamer_stop_at_first_delta() {
        let stops = vec!["Stop".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("Stop now", &mut |s| emitted.push(s.to_string()));
        assert!(stopped);
        assert_eq!(emitted.join(""), "");
        assert_eq!(streamer.into_text(), "");
    }

    /// No stop hit, natural end: finish() emits the held-back tail, into_text() is full.
    #[test]
    fn stop_streamer_no_stop_natural_end() {
        let stops = vec!["zzz".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        streamer.push("abc", &mut |s| emitted.push(s.to_string()));
        streamer.push("def", &mut |s| emitted.push(s.to_string()));
        streamer.finish("", &mut |s| emitted.push(s.to_string()));
        assert_eq!(emitted.join(""), "abcdef");
        assert_eq!(streamer.into_text(), "abcdef");
    }

    /// Multibyte hold-back: no panic, emitted concat == into_text, boundaries respected.
    #[test]
    fn stop_streamer_multibyte_no_panic() {
        let stops = vec!["STOP".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        // "世" = 3 bytes, "界" = 3 bytes, "x" = 1 byte
        streamer.push("世", &mut |s| emitted.push(s.to_string()));
        streamer.push("界x", &mut |s| emitted.push(s.to_string()));
        streamer.finish("", &mut |s| emitted.push(s.to_string()));
        let concat = emitted.join("");
        assert_eq!(concat, streamer.into_text());
        // All emitted slices must be valid UTF-8 (would panic on invalid slice otherwise).
    }

    /// Stop at a delta boundary: prior delta is emitted cleanly, stop delta emits nothing.
    #[test]
    fn stop_streamer_stop_at_delta_boundary() {
        let stops = vec!["STOP".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped1 = streamer.push("abc", &mut |s| emitted.push(s.to_string()));
        assert!(!stopped1);
        let stopped2 = streamer.push("STOP", &mut |s| emitted.push(s.to_string()));
        assert!(stopped2);
        assert_eq!(emitted.join(""), "abc");
        assert_eq!(streamer.into_text(), "abc");
    }

    /// Hold-back correctness: single delta "abcde", stop=["xyz"] (max_stop=3).
    /// With max_stop=3, hold back 2 bytes (max_stop - 1). Safe prefix = "abc" (5-2=3).
    /// finish("", sink) emits "de". into_text() = "abcde".
    #[test]
    fn stop_streamer_hold_back_emits_safe_prefix_then_finish() {
        let stops = vec!["xyz".to_string()]; // max_stop = 3
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("abcde", &mut |s| emitted.push(s.to_string()));
        assert!(!stopped);
        // After push: "abcde" (5 bytes), hold back 2 → safe = 3, emits "abc".
        assert_eq!(emitted.join(""), "abc");
        streamer.finish("", &mut |s| emitted.push(s.to_string()));
        assert_eq!(emitted.join(""), "abcde");
        assert_eq!(streamer.into_text(), "abcde");
    }

    /// BUG 2 regression (non-streaming): finish() on StopStringMatcher is a no-op after stop hit.
    #[test]
    fn stop_streamer_finish_noop_after_stop() {
        let stops = vec!["END".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("helloENDextra", &mut |s| emitted.push(s.to_string()));
        assert!(stopped);
        // Simulate tail bytes that would corrupt output if finish were not a no-op.
        streamer.finish("should_not_appear", &mut |s| emitted.push(s.to_string()));
        assert_eq!(emitted.join(""), "hello");
        assert_eq!(streamer.into_text(), "hello");
    }

    /// Non-streaming blocking-path usage: `push` with a sink that appends to a local
    /// `String` buffer, matching the pattern Metal's blocking `generate` paths use.
    #[test]
    fn stop_streamer_blocking_style_sink_into_buffer() {
        let stops = vec!["STOP".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut buf = String::new();
        let stopped = streamer.push("hello STOP world", &mut |s| buf.push_str(s));
        assert!(stopped);
        assert_eq!(buf, "hello ");
        assert_eq!(streamer.into_text(), "hello ");
    }

    // -----------------------------------------------------------------------
    // Empty-stop / empty-delta regression (codex review round 1, PR #657)
    // -----------------------------------------------------------------------

    /// Codex's exact counterexample: `StopStringMatcher::new(&[""]).push("", ...)`
    /// must report a match at byte 0. `IncrementalDetokenizer::push` can return an
    /// empty delta for the very first generated token (buffering a split UTF-8
    /// scalar, or a token whose chars all fall outside the byte decoder), and an
    /// empty stop string is documented (see the Metal prefix-cache test) to match
    /// unconditionally at byte 0 via `str::find("")`. Before the fix, `push`
    /// returned `false` immediately on an empty delta without ever scanning,
    /// silently swallowing this first-token stop.
    #[test]
    fn stop_streamer_empty_stop_matches_on_empty_delta() {
        let stops = vec![String::new()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("", &mut |s| emitted.push(s.to_string()));
        assert!(
            stopped,
            "an empty stop string must match at byte 0 even when the delta is empty \
             (empty-delta early-return regression)"
        );
        assert_eq!(emitted.join(""), "");
        assert_eq!(streamer.into_text(), "");
    }

    /// Same as above but the empty delta arrives *after* one or more prior empty
    /// deltas (e.g. several skipped special-token pushes before the first real
    /// byte). The match must still fire on whichever push first observes it,
    /// not be deferred until a non-empty delta shows up.
    #[test]
    fn stop_streamer_empty_stop_matches_after_leading_empty_deltas() {
        let stops = vec![String::new()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("", &mut |s| emitted.push(s.to_string()));
        assert!(
            stopped,
            "empty stop must match on the first push regardless of prior empty pushes"
        );
    }

    /// A *non-empty* stop string must NOT falsely trigger on a leading empty
    /// delta (mutation guard: proves the fix didn't just remove the early return
    /// and start matching everything unconditionally). The matcher must keep
    /// scanning correctly once real bytes arrive.
    #[test]
    fn stop_streamer_empty_delta_no_false_positive_for_nonempty_stop() {
        let stops = vec!["STOP".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();

        // Simulate a leading empty delta (buffered incomplete UTF-8 / skipped token).
        let stopped1 = streamer.push("", &mut |s| emitted.push(s.to_string()));
        assert!(
            !stopped1,
            "a non-empty stop string must not match an empty delta"
        );

        // Now real bytes arrive and eventually contain the stop.
        let stopped2 = streamer.push("hello STOP world", &mut |s| emitted.push(s.to_string()));
        assert!(stopped2);
        assert_eq!(emitted.join(""), "hello ");
        assert_eq!(streamer.into_text(), "hello ");
    }

    /// An empty delta arriving in the *middle* of a normal (non-empty-stop) run
    /// must be a pure no-op: no spurious emit, no change to accumulated text,
    /// and the eventual stop match is unaffected.
    #[test]
    fn stop_streamer_empty_delta_mid_run_is_noop() {
        let stops = vec!["World".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();

        let stopped1 = streamer.push("hello ", &mut |s| emitted.push(s.to_string()));
        assert!(!stopped1);
        // Empty delta mid-stream (e.g. a skipped special token): no-op.
        let stopped_mid = streamer.push("", &mut |s| emitted.push(s.to_string()));
        assert!(!stopped_mid);
        let stopped2 = streamer.push("World!", &mut |s| emitted.push(s.to_string()));
        assert!(stopped2);

        assert_eq!(emitted.join(""), "hello ");
        assert_eq!(streamer.into_text(), "hello ");
    }

    // -----------------------------------------------------------------------
    // Token-level byte-split CJK stop string (codex round-1 medium #2, PR #657)
    //
    // The Metal fixture `multibyte_vocab_tokenizer` inserted literal CJK chars
    // ("世", "界") as BPE vocab *strings*. Qwen's real decode path reverses
    // GPT-2 byte-level token strings through `byte_decoder`
    // (`detokenize.rs::append_token_bytes`); characters not present in that
    // decoder (like a literal "世") are silently skipped, so those fixture
    // tokens never decoded to CJK bytes at all and the test exercised nothing
    // but ASCII fallback. These tests build vocab entries the way a real
    // byte-level BPE tokenizer does — via `bytes_to_unicode()`'s placeholder
    // chars over the raw UTF-8 bytes of a CJK codepoint, split across two
    // token ids — and drive `IncrementalDetokenizer::push` + `StopStringMatcher`
    // together end to end, proving no mojibake and no dropped tail.
    // -----------------------------------------------------------------------

    /// Build a byte-level BPE vocab entry: the GPT-2 placeholder-char string
    /// for a run of raw bytes, exactly how `append_token_bytes` expects to
    /// reverse it via `byte_decoder`.
    fn byte_level_token(byte_encoder: &[char], bytes: &[u8]) -> String {
        bytes.iter().map(|&b| byte_encoder[b as usize]).collect()
    }

    /// `好` = 0xE5 0xA5 0xBD, split across two token ids (2 bytes then 1).
    /// Token 0's delta must be empty (buffering an incomplete codepoint);
    /// token 1's delta must be the complete "好", not mojibake or U+FFFD.
    #[test]
    fn token_level_split_cjk_codepoint_decodes_without_mojibake() {
        use crate::model::qwen35::detokenize::{IncrementalDetokenizer, bytes_to_unicode};
        use crate::tokenizer::bpe::BpeTokenizer;
        use std::collections::HashMap;

        let byte_encoder = bytes_to_unicode();
        let mut vocab: HashMap<String, u32> = HashMap::new();
        vocab.insert(byte_level_token(&byte_encoder, &[0xE5, 0xA5]), 0);
        vocab.insert(byte_level_token(&byte_encoder, &[0xBD]), 1);
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
            .expect("byte-level split-CJK vocab builds");

        let mut detok = IncrementalDetokenizer::new();
        let delta0 = detok.push(&tokenizer, 0);
        assert_eq!(
            delta0, "",
            "first 2 of 3 bytes of a CJK codepoint must buffer, not emit a replacement char"
        );
        let delta1 = detok.push(&tokenizer, 1);
        assert_eq!(
            delta1, "好",
            "the completing byte must reveal the exact codepoint, not mojibake"
        );
    }

    /// Same split-CJK token stream, but driven through `StopStringMatcher` with
    /// `stop_strings: ["好"]` — proves the matcher correctly holds back through
    /// the mid-codepoint empty delta (finding #1's fix) and matches only once
    /// the codepoint actually completes, excluding it from `into_text()` and
    /// never emitting a dropped/garbled tail for the preceding ASCII text.
    #[test]
    fn stop_streamer_matches_split_cjk_stop_string_no_dropped_tail() {
        use crate::model::qwen35::detokenize::{IncrementalDetokenizer, bytes_to_unicode};
        use crate::tokenizer::bpe::BpeTokenizer;
        use std::collections::HashMap;

        let byte_encoder = bytes_to_unicode();
        let mut vocab: HashMap<String, u32> = HashMap::new();
        // id 2: ASCII "hi" prefix (single token, one BPE entry).
        vocab.insert(byte_level_token(&byte_encoder, b"hi"), 2);
        vocab.insert(byte_level_token(&byte_encoder, &[0xE5, 0xA5]), 0);
        vocab.insert(byte_level_token(&byte_encoder, &[0xBD]), 1);
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
            .expect("byte-level split-CJK vocab builds");

        let stops = vec!["好".to_string()];
        let mut detok = IncrementalDetokenizer::new();
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();

        // Token stream: "hi", then the two split bytes of "好".
        for id in [2u32, 0, 1] {
            let delta = detok.push(&tokenizer, id);
            let stopped = streamer.push(&delta, &mut |s| emitted.push(s.to_string()));
            if stopped {
                break;
            }
        }

        assert_eq!(
            emitted.join(""),
            "hi",
            "must emit exactly the pre-stop ASCII prefix, no dropped tail and no \
             CJK bytes leaking through before the codepoint completed"
        );
        assert_eq!(streamer.into_text(), "hi");
    }

    // -----------------------------------------------------------------------
    // O(bytes^2) fix: bounded suffix scan (stop_suffix_scan / stop_string_scan)
    // -----------------------------------------------------------------------

    /// Direct, deterministic proof that the search window handed to
    /// `earliest_stop_match_from` is bounded by `max_stop - 1` plus the new
    /// delta, not by the total accumulated text — the load-bearing invariant
    /// the perf fix depends on. Computed the same way `StopStringMatcher::push`
    /// computes it, so a regression that goes back to scanning from `0` (or
    /// any unbounded value) fails this assertion immediately regardless of
    /// timing noise.
    #[test]
    fn push_search_start_stays_within_max_stop_bound_of_prev_len() {
        let max_stop = 6usize; // len("needle")
        let mut full = String::new();
        // Simulate 5,000 non-matching one-byte pushes; after each, the window
        // start must never fall more than (max_stop - 1) bytes behind the
        // pre-push length, i.e. the scan window size is capped, independent
        // of how large `full` has grown.
        for i in 0..5_000usize {
            let prev_len = full.len();
            full.push('a');
            let search_start = floor_char_boundary(&full, prev_len.saturating_sub(max_stop - 1));
            let window = full.len() - search_start;
            assert!(
                window <= max_stop,
                "iteration {i}: scan window {window} exceeds max_stop {max_stop} \
                 (search_start={search_start}, full.len()={})",
                full.len()
            );
        }
    }

    /// Scaling proxy for the O(n^2) -> O(n) fix: push N and 4N non-matching
    /// tokens through `StopStringMatcher` and assert the 4x-input run takes
    /// well under 4x (let alone the ~16x an O(n^2) scan would produce) of the
    /// N-input run's time. Uses a long single-character-repeated haystack
    /// (worst case for `str::find` since every position is a partial match of
    /// a longer stop) with an ABSENT stop string so the fast path never
    /// short-circuits via an early match.
    #[test]
    fn push_scaling_is_linear_not_quadratic() {
        use std::time::Instant;

        fn run(tokens: usize) -> std::time::Duration {
            let stops = vec!["ZZZZZZZZZZ_NEVER_MATCHES".to_string()]; // len 24, max_stop=24
            let mut streamer = StopStringMatcher::new(&stops);
            let delta = "aaaaaaaaaa"; // 10 'a' bytes per push, all non-matching
            let start = Instant::now();
            for _ in 0..tokens {
                streamer.push(delta, &mut |_s| {});
            }
            start.elapsed()
        }

        // Warm up (page faults, allocator, etc.) before measuring either leg.
        run(1_000);

        let n = 20_000usize;
        let t_n = run(n);
        let t_4n = run(4 * n);

        // O(n) work => t_4n ~= 4 * t_n. O(n^2) work => t_4n ~= 16 * t_n.
        // Use a generous 8x ceiling so this holds on loaded/noisy CI runners
        // while still failing hard against quadratic behavior.
        let ratio = t_4n.as_secs_f64() / t_n.as_secs_f64().max(1e-9);
        assert!(
            ratio < 8.0,
            "4x input took {ratio:.2}x as long as 1x input (t_n={t_n:?}, t_4n={t_4n:?}); \
             expected close to linear (~4x), not quadratic (~16x)"
        );
    }

    // -----------------------------------------------------------------------
    // Boundary cases (S4 gate 4)
    // -----------------------------------------------------------------------

    /// Stop string longer than any single push: must accumulate across many
    /// small pushes before matching, with no premature stop and no missed one.
    #[test]
    fn stop_longer_than_any_single_push() {
        let stop = "a-very-long-stop-string-marker";
        let stops = vec![stop.to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();

        let prefix = "prefix text before ";
        let mut stopped_at = None;
        // Push the stop string one byte at a time (worst case for "spans many
        // small pushes"), preceded by an unrelated prefix pushed the same way.
        for (i, ch) in prefix.chars().chain(stop.chars()).enumerate() {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            let stopped = streamer.push(s, &mut |s| emitted.push(s.to_string()));
            if stopped {
                stopped_at = Some(i);
                break;
            }
        }
        assert!(stopped_at.is_some(), "stop string must eventually be found");
        assert_eq!(emitted.join(""), prefix);
        assert_eq!(streamer.into_text(), prefix);
    }

    /// Stop string arriving split across exactly 3 pushes, none of which
    /// individually contains a complete match.
    #[test]
    fn stop_split_across_three_pushes() {
        let stops = vec!["ABCDEF".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();

        let s1 = streamer.push("xy AB", &mut |s| emitted.push(s.to_string()));
        assert!(!s1);
        let s2 = streamer.push("CD", &mut |s| emitted.push(s.to_string()));
        assert!(!s2);
        let s3 = streamer.push("EF trailing", &mut |s| emitted.push(s.to_string()));
        assert!(s3, "stop must be detected once the third push completes it");

        assert_eq!(emitted.join(""), "xy ");
        assert_eq!(streamer.into_text(), "xy ");
    }

    /// Multibyte UTF-8 exactly at the truncation boundary: a push that ends
    /// mid-multi-byte-sequence relative to the safe-prefix cut point must
    /// never panic and must never emit a split codepoint. Uses a 4-byte
    /// codepoint (emoji) landing right where `max_stop - 1` would otherwise
    /// slice.
    #[test]
    fn multibyte_utf8_at_truncation_boundary_no_panic() {
        let stops = vec!["STOP".to_string()]; // max_stop = 4, hold back 3 bytes
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();

        // "😀" is 4 bytes (0xF0 0x9F 0x98 0x80). Push ASCII then the emoji
        // then more ASCII so the hold-back window repeatedly straddles it.
        for delta in ["ab", "😀", "cd", "😀", "ef"] {
            let stopped = streamer.push(delta, &mut |s| emitted.push(s.to_string()));
            assert!(!stopped);
        }
        streamer.finish("", &mut |s| emitted.push(s.to_string()));

        let concat = emitted.join("");
        assert_eq!(concat, "ab😀cd😀ef");
        assert_eq!(streamer.into_text(), "ab😀cd😀ef");
        // Every emitted slice must have been valid UTF-8 (str type already
        // guarantees this, but the loop completing without panicking on
        // `is_char_boundary`/slicing is itself the boundary-safety proof).
    }

    /// Stop at the very first token (delta on the first `push` call already
    /// contains the entire stop string) still goes through the bounded-scan
    /// path (`prev_len == 0`) without special-casing.
    #[test]
    fn stop_at_very_first_token() {
        let stops = vec!["GO".to_string()];
        let mut streamer = StopStringMatcher::new(&stops);
        let mut emitted: Vec<String> = Vec::new();
        let stopped = streamer.push("GO!", &mut |s| emitted.push(s.to_string()));
        assert!(stopped);
        assert_eq!(emitted.join(""), "");
        assert_eq!(streamer.into_text(), "");
    }
}
