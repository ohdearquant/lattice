//! Shared string-level stop-sequence matcher for Qwen3.5 generation.
//!
//! Both the CPU path (`generation.rs`) and the Metal path (`forward::metal_qwen35`)
//! call the same [`StopStringMatcher`] and [`earliest_stop_match`] so that stop-string
//! enforcement has CPU-identical semantics everywhere: incremental scan over decoded
//! text, a held-back partial suffix for spans that could still complete a stop string
//! across a token boundary, UTF-8-safe emission (never split a codepoint), and the
//! matched stop text excluded from returned/streamed output.

// Test-only instrumentation: counts bytes actually handed to `str::find` across
// `earliest_stop_match` / `earliest_stop_match_from` calls, regardless of which
// of the two a caller uses. This lets a deterministic test observe the *real*
// scan cost incurred by the production call site in `StopStringMatcher::push`
// (crates/inference/src/model/qwen35/stop_strings.rs) and
// `DecodePolicy::stop_check`'s `FullScan` branch
// (crates/inference/src/generation.rs):
// if either regresses back to the full-rescan `earliest_stop_match`, the
// counter grows with `haystack.len()` on every call instead of with the
// bounded window, and a linear-bound assertion over many pushes fails.
#[cfg(test)]
static SCAN_BYTES_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn reset_scan_bytes_counter() {
    SCAN_BYTES_COUNTER.store(0, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn scan_bytes_counter() -> usize {
    SCAN_BYTES_COUNTER.load(std::sync::atomic::Ordering::SeqCst)
}

/// Returns the smallest byte index in `haystack` at which any stop string begins,
/// or `None` if no stop string is present.
pub(crate) fn earliest_stop_match(haystack: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter_map(|s| {
            #[cfg(test)]
            SCAN_BYTES_COUNTER.fetch_add(haystack.len(), std::sync::atomic::Ordering::SeqCst);
            haystack.find(s.as_str())
        })
        .min()
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
        .filter_map(|s| {
            #[cfg(test)]
            SCAN_BYTES_COUNTER
                .fetch_add(haystack.len() - start, std::sync::atomic::Ordering::SeqCst);
            haystack[start..].find(s.as_str()).map(|p| p + start)
        })
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

/// Shared by `StopStringMatcher::push` and
/// `crate::generation::DecodePolicy::stop_check`'s `FullScan` branch: the byte
/// offset the next scan should start from, given `prev_len` (haystack length
/// *before* this push's delta landed) and `max_stop` (the longest configured
/// stop string). Both call
/// sites must derive `search_start` through this one function — see
/// `earliest_stop_match_from`'s doc comment for why bounding the scan to
/// `[search_start..]` never misses a match.
pub(crate) fn stop_scan_search_start(haystack: &str, prev_len: usize, max_stop: usize) -> usize {
    floor_char_boundary(
        haystack,
        prev_len.saturating_sub(max_stop.saturating_sub(1)),
    )
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
pub(crate) struct StopStringMatcher {
    full: String,
    emitted: usize,
    max_stop: usize,
    stops: Vec<String>,
    stopped: bool,
}

impl StopStringMatcher {
    /// `stops` is cloned into an owned `Vec<String>` (PR #787): a
    /// `DecodePolicy`-owned `StopMode::Streaming(StopStringMatcher)`
    /// variant cannot borrow `gen_cfg.stop_strings`'s lifetime without
    /// infecting `DecodePolicy` itself with a lifetime parameter threaded
    /// through every call site; the stop-string list is tiny and constructed
    /// once per generation, so cloning it is free relative to the decode loop.
    pub(crate) fn new(stops: &[String]) -> Self {
        let max_stop = stops.iter().map(String::len).max().unwrap_or(1);
        Self {
            full: String::new(),
            emitted: 0,
            max_stop,
            stops: stops.to_vec(),
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
        let search_start = stop_scan_search_start(&self.full, prev_len, self.max_stop);

        if let Some(hit) = earliest_stop_match_from(&self.full, &self.stops, search_start) {
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
        if let Some(hit) = earliest_stop_match(&self.full, &self.stops) {
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
    ///
    /// Only reachable from Metal's non-streaming `generate()` fast/self-spec
    /// paths (which construct and drive their own local `StopStringMatcher`
    /// directly, outside `DecodePolicy` -- out of ADR-080 C3's scope) and
    /// this module's own tests; every `DecodePolicy`-driven call site
    /// (CPU and Metal streaming) accumulates into a caller-owned
    /// `text: &mut String` via `stop_check`/`finish_stop` instead (PR #787).
    #[cfg(any(test, feature = "metal-gpu"))]
    pub(crate) fn into_text(self) -> String {
        self.full
    }
}

#[cfg(test)]
mod tests;
