import SwiftUI
import Observation

// MARK: - The single source of truth (Observation framework, @MainActor).
//
// One @Observable store held in @State at the app root, handed to views as `@Bindable`.
// No @EnvironmentObject, no Combine. CRITICAL: the live run is owned HERE, above the views,
// keyed by identity — a screen re-render must never reset a running job.
@MainActor
@Observable
final class AppStore {
    var selection: Screen = .models
    var models: [ModelInfo] = []
    var runs: [RunRecord] = []
    var liveRun: LiveRun?
    var inspectorCollapsed = false
    var commandBarOpen = false
    var binariesReady = false
    var rowComfortable = false

    var modelCachePath: String { LatticeBridge.modelCacheDir.path }
    var repoRootPath: String? { LatticeBridge.repoRoot?.path }

    private var handle: RunHandle?

    init() {
        runs = Self.loadRunArchive()
        // Reap trainer processes orphaned by a previous app crash or force-quit.
        // Done synchronously at init so no orphan can race with an immediately-launched run.
        let reaped = RunRegistry.reapOrphans()
        if reaped > 0 {
            print("[AppStore] reaped \(reaped) orphaned trainer process(es) from previous session")
        }
    }

    func onAppear() {
        refreshModels()
        binariesReady = LatticeBridge.prebuiltBinary(.lattice) != nil
    }

    // MARK: Run archive persistence

    /// The shared `<AppSupport>/LatticeStudio` base directory.
    ///
    /// Used by both the runs.json archive and RunRegistry's active-runs subdirectory so
    /// both always resolve to the same location.
    nonisolated static var appSupportDir: URL? {
        guard let appSupport = try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ) else { return nil }
        let dir = appSupport.appendingPathComponent("LatticeStudio", isDirectory: true)
        // Create the subdirectory on first access.
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// URL of the on-disk runs.json archive in the app's Application Support directory.
    private static var runsArchiveURL: URL? {
        appSupportDir?.appendingPathComponent("runs.json")
    }

    /// Decode the persisted run archive from disk. Returns an empty array on any failure
    /// (missing file, corrupt JSON, schema mismatch) — honest empty, never fabricated.
    private static func loadRunArchive() -> [RunRecord] {
        guard let url = runsArchiveURL,
              let data = try? Data(contentsOf: url) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([RunRecord].self, from: data)) ?? []
    }

    /// Atomically write the finished run archive to disk. Call only with completed records.
    private func persistRunArchive() {
        guard let url = Self.runsArchiveURL else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = .prettyPrinted
        guard let data = try? encoder.encode(runs) else { return }
        try? data.write(to: url, options: .atomic)
    }

    func refreshModels() {
        Task.detached {
            let found = LatticeBridge.discoverModels()
            await MainActor.run { self.models = found }
        }
    }

    func model(named name: String) -> ModelInfo? { models.first { $0.name == name } }

    // The model currently targeted across TRAIN / QUANTIZE / CHAT. Set from MODELS via `use(_:on:)`.
    var workingModel: ModelInfo?
    // Sensible default target: first non-embedding model, else first of any.
    var defaultModel: ModelInfo? { models.first { !$0.isEmbedding } ?? models.first }
    // The effective target a screen should drive: explicit working model, else the default.
    var targetModel: ModelInfo? { workingModel ?? defaultModel }

    /// Navigate to `screen` with `model` pre-selected as the working target.
    func use(_ model: ModelInfo, on screen: Screen) {
        workingModel = model
        selection = screen
    }

    // MARK: Launch / lifecycle (the generic primitive screens build typed configs on top of)

    @discardableResult
    func launch(_ bin: LatticeBinary, args: [String], kind: RunKind, model: String, totalSteps: Int? = nil) -> LiveRun {
        let run = LiveRun(kind: kind, modelName: model)
        run.totalSteps = totalSteps
        liveRun = run

        guard let spec = LatticeBridge.launchSpec(bin, args: args) else {
            run.status = .failed
            run.appendLog("error: could not resolve `\(bin.binName)` — no prebuilt binary and no cargo fallback. Run `make build` in the lattice repo, or set LATTICE_BIN_DIR.")
            return run
        }
        run.appendLog("$ \(spec.executable.lastPathComponent) \(args.joined(separator: " "))")

        if let prior = handle, prior.isRunning { prior.stop() }
        let h = RunHandle()
        h.onEvent = { [weak self] ev in self?.consume(ev, into: run) }
        do {
            try h.start(spec)
            // Register AFTER a successful start so we never record a pid that never launched.
            let pid = h.pid
            RunRegistry.register(
                pid: pid,
                binPath: spec.executable.path,
                kind: kind.rawValue,
                startedAt: run.startedAt
            )
            // Set onExit AFTER start so it captures `pid` (a value) rather than `h`. Capturing
            // `h` would retain-cycle through h.onExit and leak the RunHandle + its Process +
            // pipe file descriptors on every run. No fast-exit race: terminationHandler
            // dispatches onExit onto the main queue, and we are still on the main actor here,
            // so onExit is always assigned before it can fire.
            h.onExit = { [weak self] code in
                // Deregister before finish so the PID slot is freed even if finish throws.
                RunRegistry.deregister(pid: pid)
                self?.finish(run, code: code)
            }
        } catch {
            run.status = .failed
            run.appendLog("launch failed: \(error.localizedDescription)")
        }
        handle = h
        return run
    }

    private func consume(_ event: LatticeEvent, into run: LiveRun) {
        switch event {
        case .trainStep(let s):
            // Step 0 carries the pre-training NLL, which equals train_done.base_nll
            // (verified byte-identical against the binary). Capture it on the first event
            // so "Δ FROM BASE" reads live from step 0 instead of "—" until the run ends.
            // BEST VAL is deliberately NOT tracked here: the trainer computes its final
            // held-out NLL once at completion (eval_valid on the saved final weights, not a
            // best checkpoint), so any running minimum would diverge from the saved adapter
            // and jump at trainDone. The live per-step held-out NLL is already shown in the
            // HELD-OUT well; BEST VAL stays honest-nil until trainDone reports it.
            if s.step == 0 { run.baseNLL = s.loss }
            run.points.append(TrainPoint(step: s.step, loss: s.loss, valLoss: s.val_loss,
                                         gradNorm: s.grad_norm, lr: s.lr, tokS: s.tok_s))
        case .trainEval(let e):
            run.bestVal = e.best_val ?? min(run.bestVal ?? e.val_loss, e.val_loss)
        case .trainDone(let d):
            run.baseNLL = d.base_nll
            run.bestVal = d.best_val ?? run.bestVal
            run.savedAdapterPath = d.saved
        case .quantLayer(let q):
            run.quantLayerIndex = q.i
            run.quantLayerCount = q.n
            // Track the dominant quantized scheme (first non-passthrough scheme wins).
            // F16 layers are kept as-is; the quantized scheme is what gives the after-bits.
            if run.quantScheme == nil, q.scheme != "F16", q.scheme != "BF16" {
                run.quantScheme = q.scheme
            }
        case .quantDone(let q):
            run.quantBeforeMB = q.before_mb
            run.quantAfterMB = q.after_mb
            run.quantRatio = q.ratio
            run.verdict = q.verdict
            run.quantMaxAbs = q.max_abs
        case .genToken(let g):
            // Accumulate streamed text; do NOT write token deltas to the generic log
            // so the log stays clean for loader status lines and error messages.
            run.genText += g.token
            if g.done == true {
                run.genTokS = g.tok_s
                run.genDone = true
            }
        case .status(let line):
            run.appendLog(line)
        case .unknown(let j):
            run.appendLog(j)
        }
    }

    private func finish(_ run: LiveRun, code: Int32) {
        run.status = (code == 0) ? .done : .failed
        let rec = RunRecord(
            id: "\(run.kind.rawValue)-\(Int(run.startedAt.timeIntervalSince1970))",
            kind: run.kind, model: run.modelName, status: run.status, startedAt: run.startedAt,
            lastLoss: run.currentLoss, bestVal: run.bestVal,
            durationS: Date().timeIntervalSince(run.startedAt),
            configSummary: nil, adapterPath: run.savedAdapterPath
        )
        runs.insert(rec, at: 0)
        // Persist the archive immediately after appending the finished record.
        persistRunArchive()
        if run.savedAdapterPath != nil { refreshModels() }
    }

    func pauseRun() { handle?.pause(); liveRun?.status = .paused }
    func resumeRun() { handle?.resume(); liveRun?.status = .running }
    func stopRun() { handle?.stop() }

    func liveRun(matching kinds: Set<RunKind>) -> LiveRun? {
        guard let r = liveRun, kinds.contains(r.kind) else { return nil }
        return r
    }

    var memoryUsage: (usedGB: Double, totalGB: Double) {
        let total = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        let used = kerr == KERN_SUCCESS ? Double(info.resident_size) / 1_073_741_824.0 : 0
        return (used, total)
    }
}
