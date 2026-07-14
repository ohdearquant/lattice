import Foundation
import Observation

// MARK: - Per-request log entry surfaced by ServeController.

/// One row in the live request log. Stamped with wall-clock time on receipt.
struct ServeLogEntry: Identifiable, Equatable {
    let id: UUID
    let clock: String   // "HH:mm:ss" local time, stamped when the event arrives
    let method: String
    let route: String
    let status: Int
    let tokens: Int?    // nil for non-generating routes (health, models, root)
    let durMs: Double
    let stream: Bool
}

// MARK: - OpenAI-compatible HTTP daemon lifecycle.
//
// Owns at most ONE `lattice_serve` subprocess. The daemon loads a single model and exposes
// `/v1/chat/completions` (+ `/v1/models`, `/health`) on 127.0.0.1, so standard OpenAI-format
// benchmark harnesses (and tools like the OpenAI SDK pointed at a custom base_url) can drive
// lattice without going through the app's chat UI. The tray menu starts/stops it.
//
// Single-instance by construction: `start` stops any existing daemon first. The engine
// serializes generation on one GPU, so running two would only contend for VRAM.
@MainActor
@Observable
final class ServeController {
    // Tracked (not @ObservationIgnored): `isRunning` reads `handle`, so reassigning it on
    // start/stop drives SwiftUI re-renders of the serve chip and the Serve surface.
    private var handle: RunHandle?
    let port: Int = 11435

    /// Fired on any state transition (started, stopped, or the child exited on its own) so the
    /// tray menu can refresh its title/enablement. Set by whoever owns the menu.
    var onChange: (() -> Void)?

    /// Last launch failure, surfaced in the tray. Cleared on a successful start.
    private(set) var lastError: String?
    /// The model name the running daemon was launched with (for the tray label).
    private(set) var servingModelName: String?
    /// The model path the running daemon was launched with (`--model`), for the CLI footer.
    private(set) var servingModelPath: URL?
    /// True after the daemon emits its `ready` event; false on stop or process exit.
    private(set) var isReady: Bool = false
    /// Live request log — newest entry last, capped at 200 rows.
    private(set) var requestLog: [ServeLogEntry] = []

    private static let clockFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    var isRunning: Bool { handle?.isRunning == true }
    var endpoint: String { "http://127.0.0.1:\(port)/v1" }

    /// Launch the daemon for `model`. Returns false (and sets `lastError`) if the binary is
    /// missing or the process fails to spawn. Idempotent: a second call replaces the daemon.
    @discardableResult
    func start(model: ModelInfo) -> Bool {
        stop()
        var args = ["--model", model.path.path, "--port", "\(port)"]
        if let tok = Self.resolveTokenizerDir(for: model) {
            args += ["--tokenizer-dir", tok.path]
        }
        guard let spec = LatticeBridge.launchSpec(.latticeServe, args: args) else {
            lastError = "lattice_serve binary not found (build it or package the app)"
            onChange?()
            return false
        }
        let h = RunHandle()
        h.onExit = { [weak self] _ in
            // RunHandle already hops to main for onExit; clear state and refresh the menu.
            guard let self else { return }
            if self.handle === h {
                self.handle = nil
                self.servingModelName = nil
                self.servingModelPath = nil
                self.isReady = false
                self.onChange?()
            }
        }
        h.onEvent = { [weak self] event in
            // RunHandle delivers events on the main actor already.
            guard let self else { return }
            switch event {
            case .ready:
                self.isReady = true
                self.onChange?()
            case .httpRequest(let r):
                let clock = ServeController.clockFormatter.string(from: Date())
                let entry = ServeLogEntry(
                    id: UUID(),
                    clock: clock,
                    method: r.method,
                    route: r.route,
                    status: r.status,
                    tokens: r.tokens,
                    durMs: r.dur_ms,
                    stream: r.stream
                )
                self.requestLog.append(entry)
                if self.requestLog.count > 200 {
                    self.requestLog.removeFirst(self.requestLog.count - 200)
                }
                self.onChange?()
            default:
                break
            }
        }
        do {
            try h.start(spec)
            handle = h
            servingModelName = model.name
            servingModelPath = model.path
            lastError = nil
            onChange?()
            return true
        } catch {
            lastError = error.localizedDescription
            onChange?()
            return false
        }
    }

    func stop() {
        handle?.stop()
        handle = nil
        servingModelName = nil
        servingModelPath = nil
        isReady = false
        onChange?()
    }

    /// Q4 models often ship without their own `tokenizer.json` (it lives in the bf16 sibling
    /// directory). Mirror ChatScreen.warmGenConfig: for a quantized model that lacks its own
    /// tokenizer, point the daemon at the de-suffixed sibling when that sibling has one.
    /// A quantized dir that DOES carry its own tokenizer (e.g. qwen3.6-27b-q4) needs no override
    /// — the daemon falls back to the model dir's `tokenizer.json`.
    static func resolveTokenizerDir(for model: ModelInfo) -> URL? {
        guard model.format.isQuantized, !model.hasTokenizer else { return nil }
        let baseName = model.name
            .replacingOccurrences(of: "-q4", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "-quarot", with: "", options: .caseInsensitive)
        let sibling = LatticeBridge.modelCacheDir.appendingPathComponent(baseName, isDirectory: true)
        let tok = sibling.appendingPathComponent("tokenizer.json")
        return FileManager.default.fileExists(atPath: tok.path) ? sibling : nil
    }
}
