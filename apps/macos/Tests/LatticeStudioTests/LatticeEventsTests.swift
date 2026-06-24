import XCTest
@testable import LatticeStudio

final class LatticeEventsTests: XCTestCase {

    // MARK: - Regex compilation

    func testAllStaticPatternsCompile() {
        XCTAssertNotNil(HumanLineParser.reTrainStepFull,  "reTrainStepFull failed to compile")
        XCTAssertNotNil(HumanLineParser.reTrainStepNoVal, "reTrainStepNoVal failed to compile")
        XCTAssertNotNil(HumanLineParser.reDoneFull,       "reDoneFull failed to compile")
        XCTAssertNotNil(HumanLineParser.reDoneNoVal,      "reDoneNoVal failed to compile")
        XCTAssertNotNil(HumanLineParser.reQuantLayer,     "reQuantLayer failed to compile")
        XCTAssertNotNil(HumanLineParser.reInputSize,      "reInputSize failed to compile")
        XCTAssertNotNil(HumanLineParser.reOutputSize,     "reOutputSize failed to compile")
        XCTAssertNotNil(HumanLineParser.reRatio,          "reRatio failed to compile")
        XCTAssertNotNil(HumanLineParser.reForwardEquiv,   "reForwardEquiv failed to compile")
    }

    // MARK: - Train step

    func testParseTrainStepFull() {
        let line = "  step    5  train NLL: 3.9876  held-out NLL: 4.4321  (train d -0.1358)"
        guard case .trainStep(let s) = LatticeEventParser.parse(line: line) else {
            return XCTFail("expected trainStep")
        }
        XCTAssertEqual(s.step, 5)
        XCTAssertEqual(s.loss, 3.9876, accuracy: 1e-9)
        XCTAssertEqual(s.val_loss ?? 0, 4.4321, accuracy: 1e-9)
    }

    func testParseTrainStepNoVal() {
        let line = "  step    0  train NLL: 4.1234"
        guard case .trainStep(let s) = LatticeEventParser.parse(line: line) else {
            return XCTFail("expected trainStep")
        }
        XCTAssertEqual(s.step, 0)
        XCTAssertEqual(s.loss, 4.1234, accuracy: 1e-9)
        XCTAssertNil(s.val_loss)
    }

    // MARK: - Train done

    func testParseTrainDoneFull() {
        let line = "=== done: train 4.1234\u{2192}3.1234 (-1.0000)  |  held-out 4.5678\u{2192}4.1000 (-0.4678)  in 12.3s ==="
        guard case .trainDone(let d) = LatticeEventParser.parse(line: line) else {
            return XCTFail("expected trainDone")
        }
        XCTAssertEqual(d.base_nll ?? 0, 4.1234, accuracy: 1e-9)
        XCTAssertEqual(d.best_val ?? 0, 4.1000, accuracy: 1e-9)
    }

    func testParseTrainDoneNoVal() {
        let line = "=== done: base NLL 4.1234 → final NLL 3.1234 (-1.0000) in 12.3s ==="
        guard case .trainDone(let d) = LatticeEventParser.parse(line: line) else {
            return XCTFail("expected trainDone")
        }
        XCTAssertEqual(d.base_nll ?? 0, 4.1234, accuracy: 1e-9)
        XCTAssertNil(d.best_val)
    }

    // MARK: - Quant layer

    func testParseQuantLayer() {
        let line = "  [1/24] Q4_0  model.layers.0.self_attn.q_proj.weight  shape=[2048, 1024]  178.0MB\u{2192}22.2MB  0.45s"
        guard case .quantLayer(let q) = LatticeEventParser.parse(line: line) else {
            return XCTFail("expected quantLayer")
        }
        XCTAssertEqual(q.i, 1)
        XCTAssertEqual(q.n, 24)
        XCTAssertEqual(q.scheme, "Q4_0")
        XCTAssertEqual(q.before_mb ?? 0, 178.0, accuracy: 0.01)
    }
}
