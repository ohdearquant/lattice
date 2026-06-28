import XCTest
@testable import LatticeStudio

final class ChatFinalizationTests: XCTestCase {

    // MARK: - finalize: prefill-aware (the history-poisoning fix)

    /// Thinking-on prefills "<think>\n" into the PROMPT, so a truncated reasoning stream is
    /// tagless in genText. It must be treated as unfinished reasoning, NOT the answer — an empty
    /// response is the signal that renderChatML skips this turn from history (history gates on a
    /// non-empty responseText). This is the exact regression codex flagged on PR #428 round-2.
    func testThinkingOnTruncatedTaglessReasoningIsNotAnswer() {
        let (thinking, response) = ChatFinalization.finalize(
            "partial reasoning", prefilledOpenThink: true)
        XCTAssertEqual(thinking, "partial reasoning")
        XCTAssertEqual(response, "", "tagless reasoning must not become the answer (history poisoning)")
    }

    /// Thinking-on, complete stream: model emits reasoning then closes with </think> before the
    /// answer. No leading <think> in genText (it was prefilled into the prompt), so the splitter
    /// keys on </think> alone.
    func testThinkingOnCompleteSplitsOnCloseTag() {
        let (thinking, response) = ChatFinalization.finalize(
            "reasoning here</think>the answer", prefilledOpenThink: true)
        XCTAssertEqual(thinking, "reasoning here")
        XCTAssertEqual(response, "the answer")
    }

    /// Thinking-on, closed reasoning but no answer text after </think> (max tokens / stop right
    /// at the boundary). Reasoning is kept; response empty so the turn is skipped from history.
    func testThinkingOnClosedButEmptyAnswer() {
        let (thinking, response) = ChatFinalization.finalize(
            "reasoning here</think>", prefilledOpenThink: true)
        XCTAssertEqual(thinking, "reasoning here")
        XCTAssertEqual(response, "")
    }

    /// Thinking-off: prompt prefills a CLOSED empty block, model answers directly with no tags.
    /// The raw stream IS the answer (prefilledOpenThink is false on this path).
    func testThinkingOffTaglessIsAnswer() {
        let (thinking, response) = ChatFinalization.finalize(
            "the direct answer", prefilledOpenThink: false)
        XCTAssertEqual(thinking, "")
        XCTAssertEqual(response, "the direct answer")
    }

    /// In-stream open <think> that never closes (model re-emitted the tag and was truncated):
    /// everything after the open tag is reasoning, response empty — regardless of prefill flag.
    func testInStreamOpenTagNeverClosedIsReasoning() {
        let (thinking, response) = ChatFinalization.finalize(
            "<think>\nstill reasoning", prefilledOpenThink: false)
        XCTAssertEqual(thinking, "still reasoning")
        XCTAssertEqual(response, "")
    }

    // MARK: - split: pure tag splitter

    func testSplitClosedBlock() {
        let (thinking, response) = ChatFinalization.split("<think>reasons</think>answer")
        XCTAssertEqual(thinking, "reasons")
        XCTAssertEqual(response, "answer")
    }

    func testSplitNoTags() {
        let (thinking, response) = ChatFinalization.split("just text")
        XCTAssertEqual(thinking, "")
        XCTAssertEqual(response, "just text")
    }
}
