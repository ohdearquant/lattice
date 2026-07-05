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
    pub(crate) fn push(&mut self, delta: &str, sink: &mut impl FnMut(&str)) -> bool {
        if delta.is_empty() {
            return false;
        }
        self.full.push_str(delta);

        if let Some(hit) = earliest_stop_match(&self.full, self.stops) {
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
}
