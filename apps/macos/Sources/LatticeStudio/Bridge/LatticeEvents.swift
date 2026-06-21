import Foundation

// MARK: - The `@@lattice {json}` event protocol
//
// CONTRACT shared between the Swift app and the Rust binaries' `--json` mode.
// The Rust trainer/quantizer prints one line per event, prefixed with the sentinel
// so it is unambiguous even amid other stdout:
//
//   @@lattice {"ev":"train_step","step":420,"loss":0.6121,"lr":1.81e-4,"grad_norm":0.93,"tok_s":1820.0}
//
// The bridge strips the prefix and decodes by the `ev` tag. Unknown tags are ignored,
// so the protocol can grow without breaking older app builds.

let kLatticeEventPrefix = "@@lattice "

/// One decoded event from a lattice subprocess. Add cases as binaries learn to emit them.
enum LatticeEvent: Equatable {
    case trainStep(TrainStep)
    case trainEval(TrainEval)
    case trainDone(TrainDone)
    case quantLayer(QuantLayer)
    case quantDone(QuantDone)
    case genToken(GenToken)
    case perplexity(Perplexity)   // eval_perplexity --json: one row per measurement label
    case embedDone(EmbedDone)     // embed --json: cosine matrix + optional preview vectors
    case status(String)        // free-form human status line (non-JSON stdout/stderr)
    case unknown(String)       // a JSON event whose tag we don't recognize

    struct TrainStep: Codable, Equatable {
        var step: Int
        var loss: Double
        var lr: Double?
        var grad_norm: Double?
        var tok_s: Double?
        var val_loss: Double?
        var eta_s: Double?
    }
    struct TrainEval: Codable, Equatable {
        var step: Int
        var val_loss: Double
        var best_val: Double?
    }
    struct TrainDone: Codable, Equatable {
        var base_nll: Double?
        var final_nll: Double?
        var best_val: Double?
        var duration_s: Double?
        var saved: String?     // path to the saved adapter .safetensors, if any
    }
    struct QuantLayer: Codable, Equatable {
        var i: Int
        var n: Int
        var name: String
        var scheme: String     // "Q4_0" | "F16" | ...
        var before_mb: Double?
        var after_mb: Double?
    }
    struct QuantDone: Codable, Equatable {
        var before_mb: Double
        var after_mb: Double
        var ratio: Double
        var verdict: String?   // "PASS" | "WARN" | "FAIL"
        var max_abs: Double?   // QuaRot forward-equivalence
        var est_ppl_delta: Double?
    }
    struct GenToken: Codable, Equatable {
        var token: String
        var done: Bool?
        var tok_s: Double?
        var ttft_ms: Double?
    }

    // MARK: - Perplexity event (eval_perplexity --json)
    //
    // CONTRACT:
    //   @@lattice {"ev":"perplexity","label":"bf16","ppl":13.62,"nll":2.611,"tokens":4096,"windows":8,"ms":5234}
    //
    // `ppl` is the only required field beyond `ev`. A run may emit multiple
    // rows — one per measurement label (e.g. "bf16", "q4", "quarot", "adapter").
    struct Perplexity: Codable, Equatable {
        /// Human label for this measurement (e.g. "bf16", "q4", "quarot", "adapter").
        var label: String?
        /// Perplexity score (required).
        var ppl: Double
        /// Mean negative log-likelihood in nats (optional, derived from ppl but often emitted).
        var nll: Double?
        /// Total tokens scored in this run.
        var tokens: Int?
        /// Number of sliding windows evaluated.
        var windows: Int?
        /// Wall-clock milliseconds for this evaluation.
        var ms: Double?
    }

    // MARK: - EmbedDone event (embed --json)
    //
    // CONTRACT:
    //   @@lattice {"ev":"embed_done","model":"bge-small-en-v1.5","dims":384,"count":3,
    //              "cosine":[[1.0,0.83],[0.83,1.0]],"preview":[[..8 floats..]],"ms":140}
    //
    // `dims` and `count` are required. `cosine` and `preview` carry the payload
    // matrices; both may be absent for very large batches to keep the line bounded.
    struct EmbedDone: Codable, Equatable {
        /// Model identifier as reported by the binary (e.g. "bge-small-en-v1.5").
        var model: String?
        /// Output embedding dimensionality (required).
        var dims: Int
        /// Number of embeddings produced in this batch (required).
        var count: Int
        /// Pairwise cosine-similarity matrix [count × count]. Absent for large batches.
        var cosine: [[Double]]?
        /// Preview vectors — first 8 floats of each embedding. Absent for large batches.
        var preview: [[Double]]?
        /// Wall-clock milliseconds for this batch.
        var ms: Double?
    }
}

/// Tagged envelope used to peek the `ev` discriminator before decoding the full payload.
private struct EventTag: Codable { var ev: String }

enum LatticeEventParser {
    /// Parse a single raw stdout/stderr line into an event.
    ///
    /// Priority:
    ///   1. `@@lattice {json}` sentinel — structured JSON protocol (future --json mode).
    ///   2. Human-readable stdout patterns emitted by today's binaries (fallback).
    ///   3. Anything else → `.status(line)` so the UI can display it as a log line.
    static func parse(line rawLine: String) -> LatticeEvent? {
        let line = stripANSI(rawLine.trimmingCharacters(in: .whitespacesAndNewlines))
        guard !line.isEmpty else { return nil }

        // --- Path 1: structured JSON protocol ---
        if line.hasPrefix(kLatticeEventPrefix) {
            return parseJSON(String(line.dropFirst(kLatticeEventPrefix.count)))
        }

        // --- Path 2: human-readable fallback ---
        if let ev = HumanLineParser.parse(line) { return ev }

        // --- Path 3: raw status ---
        return .status(line)
    }

    // MARK: - JSON path

    private static func parseJSON(_ jsonText: String) -> LatticeEvent {
        guard let data = jsonText.data(using: .utf8) else { return .unknown(jsonText) }
        let decoder = JSONDecoder()
        guard let tag = try? decoder.decode(EventTag.self, from: data) else { return .unknown(jsonText) }

        switch tag.ev {
        case "train_step":  return (try? decoder.decode(LatticeEvent.TrainStep.self, from: data)).map(LatticeEvent.trainStep) ?? .unknown(jsonText)
        case "train_eval":  return (try? decoder.decode(LatticeEvent.TrainEval.self, from: data)).map(LatticeEvent.trainEval) ?? .unknown(jsonText)
        case "train_done":  return (try? decoder.decode(LatticeEvent.TrainDone.self, from: data)).map(LatticeEvent.trainDone) ?? .unknown(jsonText)
        case "quant_layer": return (try? decoder.decode(LatticeEvent.QuantLayer.self, from: data)).map(LatticeEvent.quantLayer) ?? .unknown(jsonText)
        case "quant_done":  return (try? decoder.decode(LatticeEvent.QuantDone.self, from: data)).map(LatticeEvent.quantDone) ?? .unknown(jsonText)
        case "gen_token":   return (try? decoder.decode(LatticeEvent.GenToken.self, from: data)).map(LatticeEvent.genToken) ?? .unknown(jsonText)
        case "perplexity":  return (try? decoder.decode(LatticeEvent.Perplexity.self, from: data)).map(LatticeEvent.perplexity) ?? .unknown(jsonText)
        case "embed_done":  return (try? decoder.decode(LatticeEvent.EmbedDone.self, from: data)).map(LatticeEvent.embedDone) ?? .unknown(jsonText)
        default:            return .unknown(jsonText)
        }
    }

    // MARK: - ANSI strip

    /// Remove ANSI escape sequences (e.g. colour codes) from a line before parsing.
    static func stripANSI(_ s: String) -> String {
        // Matches ESC [ ... m  and other common CSI sequences.
        guard s.contains("\u{1B}") else { return s }
        var result = ""
        result.reserveCapacity(s.count)
        var iter = s.unicodeScalars.makeIterator()
        while let c = iter.next() {
            if c == "\u{1B}" {
                // Consume until the terminating letter (A-Z, a-z, or for OSC: BEL/ST)
                while let n = iter.next() {
                    if n.value >= 0x40 && n.value <= 0x7E { break }  // final byte
                    if n == "\u{07}" { break }                         // BEL (OSC terminator)
                }
            } else {
                result.unicodeScalars.append(c)
            }
        }
        return result
    }
}

// MARK: - Human-readable line parser

/// Converts the current human-readable stdout from train_grad_full, quantize_q4, and
/// quantize_quarot into the same LatticeEvent cases used by the JSON protocol.
/// Returns nil for lines that don't match any known pattern (caller emits .status).
///
/// Accumulator for quantization summary fields (shared across the line stream for one job).
/// QuantDone is emitted when both size fields and ratio have been seen.
final class QuantAccumulator {
    var beforeMB: Double?
    var afterMB: Double?
    var ratio: Double?
    var maxAbs: Double?
    var verdict: String?

    /// Call with each parsed summary field. Returns a .quantDone event when the
    /// accumulator is complete (has at least before, after, and ratio).
    func update(beforeMB: Double? = nil, afterMB: Double? = nil,
                ratio: Double? = nil, maxAbs: Double? = nil,
                verdict: String? = nil) -> LatticeEvent? {
        if let v = beforeMB { self.beforeMB = v }
        if let v = afterMB  { self.afterMB  = v }
        if let v = ratio    { self.ratio    = v }
        if let v = maxAbs   { self.maxAbs   = v }
        if let v = verdict  { self.verdict  = v }
        guard let b = self.beforeMB, let a = self.afterMB, let r = self.ratio else { return nil }
        return .quantDone(LatticeEvent.QuantDone(
            before_mb: b, after_mb: a, ratio: r,
            verdict: self.verdict, max_abs: self.maxAbs, est_ppl_delta: nil
        ))
    }

    func reset() {
        beforeMB = nil; afterMB = nil; ratio = nil; maxAbs = nil; verdict = nil
    }
}

/// Shared per-process accumulator.  RunHandle is one-process-per-instance, so we
/// keep a single thread-local-style global here; the main-thread delivery in RunHandle
/// means there is no data race.
private var _quantAccumulator = QuantAccumulator()

/// Reset the shared quant accumulator at the start of every quantization run.
/// Called by the Store layer before launching a new quant subprocess.
func resetQuantAccumulator() { _quantAccumulator.reset() }

enum HumanLineParser {
    // MARK: Entry point

    static func parse(_ line: String) -> LatticeEvent? {
        if let ev = parseTrainStep(line)   { return ev }
        if let ev = parseTrainDone(line)   { return ev }
        if let ev = parseQuantLayer(line)  { return ev }
        if let ev = parseQuantSummary(line){ return ev }
        return nil
    }

    // MARK: - train_grad_full step lines

    // With validation:
    //   "  step    5  train NLL: 3.9876  held-out NLL: 4.4321  (train d -0.1358)"
    //   "  step    0  train NLL: 4.1234  held-out NLL: 4.5678"
    // Without validation:
    //   "  step    0  train NLL: 4.1234"
    //   "  step   10  train NLL: 3.80  (delta from base: -0.32)"
    private static let reTrainStepFull = try! NSRegularExpression(
        pattern: #"^\s*step\s+(\d+)\s+train NLL:\s*([0-9]+(?:\.[0-9]+)?)\s+held-out NLL:\s*([0-9]+(?:\.[0-9]+)?)"#
    )
    private static let reTrainStepNoVal = try! NSRegularExpression(
        pattern: #"^\s*step\s+(\d+)\s+train NLL:\s*([0-9]+(?:\.[0-9]+)?)"#
    )

    private static func parseTrainStep(_ line: String) -> LatticeEvent? {
        // Full variant first (with held-out NLL).
        if let m = reTrainStepFull.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) {
            guard let stepR  = Range(m.range(at: 1), in: line),
                  let lossR  = Range(m.range(at: 2), in: line),
                  let valR   = Range(m.range(at: 3), in: line),
                  let step   = Int(line[stepR]),
                  let loss   = Double(line[lossR]),
                  let valLoss = Double(line[valR]) else { return nil }
            return .trainStep(LatticeEvent.TrainStep(
                step: step, loss: loss, lr: nil, grad_norm: nil, tok_s: nil,
                val_loss: valLoss, eta_s: nil
            ))
        }
        // No-val variant.
        if let m = reTrainStepNoVal.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) {
            guard let stepR = Range(m.range(at: 1), in: line),
                  let lossR = Range(m.range(at: 2), in: line),
                  let step  = Int(line[stepR]),
                  let loss  = Double(line[lossR]) else { return nil }
            return .trainStep(LatticeEvent.TrainStep(
                step: step, loss: loss, lr: nil, grad_norm: nil, tok_s: nil,
                val_loss: nil, eta_s: nil
            ))
        }
        return nil
    }

    // MARK: - train_grad_full done lines

    // With validation:
    //   "=== done: train 4.1234→3.1234 (-1.0000)  |  held-out 4.5678→4.1000 (-0.4678)  in 12.3s ==="
    // Without validation:
    //   "=== done: base NLL 4.1234 → final NLL 3.1234 (-1.0000) in 12.3s ==="
    // Unicode right arrow (→ U+2192) or ASCII "->" both accepted.
    private static let reDoneFull = try! NSRegularExpression(
        pattern: #"^=== done: train ([0-9]+\.[0-9]+)[-→>]+([0-9]+\.[0-9]+)\s*\([^)]+\)\s*\|\s*held-out ([0-9]+\.[0-9]+)[-→>]+([0-9]+\.[0-9]+)\s*\([^)]+\)\s*in ([0-9]+(?:\.[0-9]+)?)s"#
    )
    private static let reDoneNoVal = try! NSRegularExpression(
        pattern: #"^=== done: base NLL ([0-9]+\.[0-9]+)[-→> ]+final NLL ([0-9]+\.[0-9]+)\s*\([^)]+\)\s*in ([0-9]+(?:\.[0-9]+)?)s"#
    )

    private static func parseTrainDone(_ line: String) -> LatticeEvent? {
        guard line.hasPrefix("===") && line.contains("done:") else { return nil }

        if let m = reDoneFull.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) {
            guard let baseR  = Range(m.range(at: 1), in: line),
                  let finR   = Range(m.range(at: 2), in: line),
                  let _      = Range(m.range(at: 3), in: line),  // held-out base (not stored)
                  let bestR  = Range(m.range(at: 4), in: line),
                  let durR   = Range(m.range(at: 5), in: line),
                  let baseNLL = Double(line[baseR]),
                  let finalNLL = Double(line[finR]),
                  let bestVal  = Double(line[bestR]),
                  let dur      = Double(line[durR]) else { return nil }
            return .trainDone(LatticeEvent.TrainDone(
                base_nll: baseNLL, final_nll: finalNLL,
                best_val: bestVal, duration_s: dur, saved: nil
            ))
        }
        if let m = reDoneNoVal.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) {
            guard let baseR  = Range(m.range(at: 1), in: line),
                  let finR   = Range(m.range(at: 2), in: line),
                  let durR   = Range(m.range(at: 3), in: line),
                  let baseNLL  = Double(line[baseR]),
                  let finalNLL = Double(line[finR]),
                  let dur      = Double(line[durR]) else { return nil }
            return .trainDone(LatticeEvent.TrainDone(
                base_nll: baseNLL, final_nll: finalNLL,
                best_val: nil, duration_s: dur, saved: nil
            ))
        }
        return nil
    }

    // MARK: - quantize per-tensor lines

    // "[1/24] Q4_0  model.layers.0...weight  shape=[2048, 1024]  178.0MB→22.2MB  0.45s"
    // "[2/24] F16   model.layers.0.input_layernorm.weight  shape=[1024]  0.0MB  ..."
    // The size part is optional (F16 lines may omit the → part).
    private static let reQuantLayer = try! NSRegularExpression(
        pattern: #"^\s*\[(\d+)/(\d+)\]\s+(\S+)\s+(\S+)\s+shape=\[[^\]]+\](?:\s+([0-9]+(?:\.[0-9]+)?)(MB|GB)[-→>]+([0-9]+(?:\.[0-9]+)?)(MB|GB))?"#
    )

    private static func parseQuantLayer(_ line: String) -> LatticeEvent? {
        guard line.contains("/") && line.contains("shape=") else { return nil }
        guard let m = reQuantLayer.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) else { return nil }
        guard let iR      = Range(m.range(at: 1), in: line),
              let nR      = Range(m.range(at: 2), in: line),
              let schemeR = Range(m.range(at: 3), in: line),
              let nameR   = Range(m.range(at: 4), in: line),
              let i       = Int(line[iR]),
              let n       = Int(line[nR]) else { return nil }

        let scheme = String(line[schemeR])
        let name   = String(line[nameR])

        var beforeMB: Double? = nil
        var afterMB: Double?  = nil

        // Groups 5-8: before value, before unit, after value, after unit (all optional)
        if m.range(at: 5).location != NSNotFound,
           let bValR = Range(m.range(at: 5), in: line),
           let bUnitR = Range(m.range(at: 6), in: line),
           let aValR  = Range(m.range(at: 7), in: line),
           let aUnitR = Range(m.range(at: 8), in: line),
           let bVal  = Double(line[bValR]),
           let aVal  = Double(line[aValR]) {
            let bUnit = String(line[bUnitR])
            let aUnit = String(line[aUnitR])
            beforeMB = bUnit == "GB" ? bVal * 1024.0 : bVal
            afterMB  = aUnit == "GB" ? aVal * 1024.0 : aVal
        }

        return .quantLayer(LatticeEvent.QuantLayer(
            i: i, n: n, name: name, scheme: scheme,
            before_mb: beforeMB, after_mb: afterMB
        ))
    }

    // MARK: - quantize summary lines (accumulate → emit quantDone)

    // q4 summary:
    //   "Input size:   27.00 GB"
    //   "Output size:   6.75 GB"
    //   "Ratio:        4.00x  (25.0%)"
    // quarot summary:
    //   "Input bytes:       27648.00 MB"
    //   "Output bytes:      6912.00 MB"
    //   "Compression:       4.00x (25.0%)"
    //   "Forward-equiv:     max_abs=1.234e-06, mean_abs=4.567e-07 (tol=1e-05, ...)"
    private static let reInputSize  = try! NSRegularExpression(pattern: #"Input (?:size|bytes):\s*([0-9]+(?:\.[0-9]+)?)\s*(MB|GB)"#)
    private static let reOutputSize = try! NSRegularExpression(pattern: #"Output (?:size|bytes):\s*([0-9]+(?:\.[0-9]+)?)\s*(MB|GB)"#)
    private static let reRatio      = try! NSRegularExpression(pattern: #"(?:Ratio|Compression):\s*([0-9]+(?:\.[0-9]+)?)x"#)
    private static let reForwardEquiv = try! NSRegularExpression(pattern: #"Forward-equiv:.*max_abs=([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)(?:.*tol=([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?))?"#)

    private static func parseQuantSummary(_ line: String) -> LatticeEvent? {
        let ns = NSRange(line.startIndex..., in: line)

        if let m = reInputSize.firstMatch(in: line, range: ns),
           let vR = Range(m.range(at: 1), in: line),
           let uR = Range(m.range(at: 2), in: line),
           let v  = Double(line[vR]) {
            let mb = String(line[uR]) == "GB" ? v * 1024.0 : v
            return _quantAccumulator.update(beforeMB: mb)
        }
        if let m = reOutputSize.firstMatch(in: line, range: ns),
           let vR = Range(m.range(at: 1), in: line),
           let uR = Range(m.range(at: 2), in: line),
           let v  = Double(line[vR]) {
            let mb = String(line[uR]) == "GB" ? v * 1024.0 : v
            return _quantAccumulator.update(afterMB: mb)
        }
        if let m = reRatio.firstMatch(in: line, range: ns),
           let vR = Range(m.range(at: 1), in: line),
           let v  = Double(line[vR]) {
            return _quantAccumulator.update(ratio: v)
        }
        if let m = reForwardEquiv.firstMatch(in: line, range: ns),
           let vR = Range(m.range(at: 1), in: line),
           let v  = Double(line[vR]) {
            var derived: String? = nil
            if m.range(at: 2).location != NSNotFound,
               let tolR = Range(m.range(at: 2), in: line),
               let tol  = Double(line[tolR]) {
                derived = v <= tol ? "PASS" : "FAIL"
            }
            return _quantAccumulator.update(maxAbs: v, verdict: derived)
        }
        return nil
    }
}

// MARK: - DEBUG self-test

#if DEBUG
/// Parses 4 representative lines and returns whether all decoded to expected cases.
/// Compile-time correct; call from unit tests or debug console.
func _latticeParserSelfTest() -> Bool {
    var ok = true

    let r1 = LatticeEventParser.parse(line: "  step    5  train NLL: 3.9876  held-out NLL: 4.4321  (train d -0.1358)")
    if case .trainStep(let s) = r1, s.step == 5, abs(s.loss - 3.9876) < 1e-9, s.val_loss == 4.4321 {} else { ok = false }

    let r2 = LatticeEventParser.parse(line: "  step    0  train NLL: 4.1234")
    if case .trainStep(let s) = r2, s.step == 0, abs(s.loss - 4.1234) < 1e-9, s.val_loss == nil {} else { ok = false }

    let r3 = LatticeEventParser.parse(line: "=== done: train 4.1234\u{2192}3.1234 (-1.0000)  |  held-out 4.5678\u{2192}4.1000 (-0.4678)  in 12.3s ===")
    if case .trainDone(let d) = r3, let b = d.base_nll, abs(b - 4.1234) < 1e-9,
       let bv = d.best_val, abs(bv - 4.1000) < 1e-9 {} else { ok = false }

    let r4 = LatticeEventParser.parse(line: "  [1/24] Q4_0  model.layers.0.self_attn.q_proj.weight  shape=[2048, 1024]  178.0MB\u{2192}22.2MB  0.45s")
    if case .quantLayer(let q) = r4, q.i == 1, q.n == 24, q.scheme == "Q4_0",
       let bm = q.before_mb, abs(bm - 178.0) < 0.01 {} else { ok = false }

    return ok
}
#endif
