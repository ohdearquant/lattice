import Foundation

/// Pure split/finalization of a chat generation stream around the `<think>…</think>`
/// reasoning boundary. Extracted from `ChatScreen` so it can be unit-tested without a View.
///
/// Two surfaces share this logic and MUST agree, otherwise reasoning shown live in the
/// thinking area can land in the answer bubble at the end (or vice versa):
/// - the live `.onChange` streaming router (runs on every token delta), and
/// - `resolveTurn` finalization (runs once generation completes).
enum ChatFinalization {

    /// Split a cumulative generation string into (thinking, response) on the
    /// `<think>…</think>` boundary. Tag-only — it knows nothing about how the prompt was built.
    /// - `</think>` present: text before it (minus a leading `<think>`) is thinking; text after is response.
    /// - only `<think>` present: everything after it is thinking; response is empty.
    /// - neither tag present: all of it is the response; thinking is empty.
    /// Always parses the FULL cumulative string, so a tag split across token deltas resolves on the next delta.
    static func split(_ raw: String) -> (thinking: String, response: String) {
        let openTag = "<think>"
        let closeTag = "</think>"
        if let closeRange = raw.range(of: closeTag) {
            var thinking = String(raw[raw.startIndex..<closeRange.lowerBound])
            if let openRange = thinking.range(of: openTag) {
                thinking = String(thinking[openRange.upperBound...])
            }
            let response = String(raw[closeRange.upperBound...])
            return (thinking.trimmingCharacters(in: .whitespacesAndNewlines),
                    response.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        if let openRange = raw.range(of: openTag) {
            let thinking = String(raw[openRange.upperBound...])
            return (thinking.trimmingCharacters(in: .whitespacesAndNewlines), "")
        }
        return ("", raw.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Finalize a generation into (thinking, response), honoring whether the request
    /// PREFILLED an open `<think>` block into the prompt (the enable_thinking=true path in
    /// renderChatML). When thinking was prefilled the model's stream carries NO opening tag —
    /// it begins mid-reasoning and only emits `</think>` before the answer. So a stream with no
    /// `</think>` boundary is UNFINISHED REASONING, not the answer, and must not become the
    /// response: history gates on a non-empty responseText, so an unfinished turn stored as the
    /// answer would re-inject raw reasoning into the next prompt.
    ///
    /// - `</think>` present: standard split (thinking before, answer after; answer may be empty).
    /// - in-stream `<think>` but no `</think>`: open block never closed → all reasoning.
    /// - no tags but `prefilledOpenThink`: tagless stream is reasoning, response empty.
    /// - no tags and not prefilled (thinking off): the raw stream is the answer.
    static func finalize(_ raw: String, prefilledOpenThink: Bool) -> (thinking: String, response: String) {
        let parsed = split(raw)
        if raw.contains("</think>") {
            return parsed
        }
        if raw.contains("<think>") {
            return (parsed.thinking, "")
        }
        if prefilledOpenThink {
            return (raw.trimmingCharacters(in: .whitespacesAndNewlines), "")
        }
        return ("", raw.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
