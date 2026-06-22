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
    var adapters: [AdapterInfo] = []
    var runs: [RunRecord] = []
    var liveRun: LiveRun?
    var inspectorPresented = false
    var commandBarOpen = false
    var binariesReady = false
    var rowComfortable = false

    var modelCachePath: String { LatticeBridge.modelCacheDir.path }
    var repoRootPath: String? { LatticeBridge.repoRoot?.path }

    // MARK: - Hoisted Chat state (survives NavigationSplitView teardown)

    // Single-mode conversation transcript.
    var chatTurns: [ChatTurn] = []
    // GPU/CPU inference mode. true = chat_metal (Metal GPU); false = generate_lora (CPU BF16).
    // Honest-label contract: this flag selects the binary AND anchors the label in each turn
    // bubble at send time. A GPU-flagged run NEVER appears with a CPU label, and vice versa.
    var chatUseGPU: Bool = false
    // Selections — survive navigation.
    var chatSelectedModelName: String = ""
    var chatSelectedAdapterName: String = "none"
    // Generation knob text fields.
    var chatTempText: String = "0.7"
    var chatMaxTokensText: String = "256"
    var chatSeedText: String = ""
    var chatTopKText: String = "50"
    var chatTopPText: String = "0.9"
    var chatRepPenaltyText: String = "1.1"
    // In-flight tracking — must be store-owned so a generation that finishes while
    // the user is on another screen still lands in chatTurns.
    var chatAwaitingTurnID: UUID? = nil
    var chatUserStoppedTurnID: UUID? = nil

    // MARK: - Hoisted Eval workspace state (Stage 1 — structural plumbing)

    // Active tab in EvalScreen ("PPL" | "Compare" | "Similar").
    var evalActiveTab: String = "PPL"
    // Model selection (multi-select; survives navigation).
    var evalSelectedModelNames: Set<String> = []
    // Per-model adapter selection: modelName → adapterName.
    var evalSelectedAdapterNames: [String: String] = [:]
    // Generation knobs — separate from chat context; eval and chat are independent contexts.
    var evalTempText: String = "0.7"
    var evalMaxTokensText: String = "256"
    var evalSeedText: String = ""
    // Whether Compare tab generation should use the GPU Metal path.
    // When true, each column dispatches to chat_metal (GPU Metal).
    // When false, each column dispatches to generate_lora (CPU BF16).
    // Honest-label contract: the label in EvalColumn.label is snapshotted at send time
    // from the actual GenConfig.useGPU value so it always matches the binary that ran.
    var evalUseGPU: Bool = false
    // PPL corpus path (nil = use built-in default ~200-token corpus).
    // Stored as a path string because URL is not directly @Observable-friendly.
    var evalCorpusPath: String? = nil

    // MARK: - Eval compare state (Stage 3 — N-way generation compare)

    // Accumulated compare experiment pairs for the COMPARE tab.
    // Each pair holds one EvalColumn per configured slot. Persists across navigation.
    var evalComparePairs: [EvalComparePair] = []
    // Index of the column currently being generated (0-based). -1 = idle.
    var evalComparePhase: Int = -1
    // ID of the pair currently being generated. nil = idle.
    var evalCompareAwaitingPairID: UUID? = nil

    // MARK: - Hoisted PPL state (survives ModelsScreen teardown)

    /// Last measured perplexity per model ID. Keyed by ModelInfo.id (path).
    /// Persisted across launches via ppl.json in the same AppSupport dir as runs.json.
    var measuredPPL: [String: MeasuredPPL] = [:]

    // MARK: - Get Models state

    // Canonical names of models currently being downloaded.
    var downloadingModels: Set<String> = []
    // Per-model download errors; cleared when a download is retried or succeeds.
    var downloadErrors: [String: String] = [:]
    // Name of model currently being imported from disk ("" when idle).
    var importingModel: String = ""
    // Last import error; cleared at the start of each import attempt.
    var importError: String? = nil

    private var handle: RunHandle?
    // Dedicated handles for concurrent downloads — keyed by ObjectIdentifier so each
    // download gets its own RunHandle and never evicts a training/eval job from `handle`.
    var _downloadHandles: [ObjectIdentifier: RunHandle] = [:]

    // Handles stopped to make way for a new run, kept alive here until their terminationHandler
    // delivers onExit. RunHandle.terminationHandler captures [weak self], so once `launch()`
    // reassigns `handle` the old handle would otherwise deallocate before its exit callback runs:
    // finish() (and the run's onComplete) would never fire and an awaiting chat turn would hang
    // .running forever. Each handle removes itself from this map inside its own onExit.
    private var retiringHandles: [ObjectIdentifier: RunHandle] = [:]

    init() {
        runs = Self.loadRunArchive()
        measuredPPL = Self.loadPPLArchive()
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

    // MARK: PPL archive persistence

    private static var pplArchiveURL: URL? {
        appSupportDir?.appendingPathComponent("ppl.json")
    }

    private static func loadPPLArchive() -> [String: MeasuredPPL] {
        guard let url = pplArchiveURL,
              let data = try? Data(contentsOf: url) else { return [:] }
        return (try? JSONDecoder().decode([String: MeasuredPPL].self, from: data)) ?? [:]
    }

    func persistPPLArchive() {
        guard let url = Self.pplArchiveURL else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        guard let data = try? encoder.encode(measuredPPL) else { return }
        try? data.write(to: url, options: .atomic)
    }

    func refreshModels() {
        Task.detached {
            let found = LatticeBridge.discoverModels()
            let foundAdapters: [AdapterInfo]
            if let root = LatticeBridge.repoRoot {
                foundAdapters = LatticeBridge.discoverAdapterPackages(in: root.appendingPathComponent("adapters", isDirectory: true))
            } else {
                foundAdapters = []
            }
            // Attach compatible adapters to each model so the Chat A/B picker
            // (selectedModel.adapters) can offer them; the global list still backs MODELS.
            let withAdapters = LatticeBridge.associateAdapters(foundAdapters, into: found)
            await MainActor.run {
                self.models = withAdapters
                self.adapters = foundAdapters
            }
        }
    }

    /// Move adapter directory to Trash and refresh the adapter list.
    func deleteAdapter(_ adapter: AdapterInfo) {
        var trashURL: NSURL?
        try? FileManager.default.trashItem(at: adapter.path, resultingItemURL: &trashURL)
        refreshModels()
    }

    func model(named name: String) -> ModelInfo? { models.first { $0.name == name } }

    // The model currently targeted across TRAIN / QUANTIZE / CHAT. Set from MODELS via `use(_:on:)`.
    var workingModel: ModelInfo?
    // The dataset currently selected in DATA. Consumed by TRAIN in a later task.
    var workingDataset: DatasetFileStat?
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

        if let prior = handle, prior.isRunning {
            // Retain the superseded handle until its onExit fires (see retiringHandles). The
            // reassignment of `handle` below would otherwise drop its last strong reference, and
            // its [weak self] terminationHandler could then no-op — stranding the prior run
            // unresolved (a chat turn stuck .running with chatAwaitingTurnID never cleared).
            retiringHandles[ObjectIdentifier(prior)] = prior
            prior.stop()
        }
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
            // ObjectIdentifier (a value) lets onExit drop a superseded handle from
            // retiringHandles without capturing `h` itself, which would re-introduce the
            // retain cycle the comment above avoids.
            let hid = ObjectIdentifier(h)
            h.onExit = { [weak self] code in
                // Deregister before finish so the PID slot is freed even if finish throws.
                RunRegistry.deregister(pid: pid)
                self?.finish(run, code: code)
                self?.retiringHandles.removeValue(forKey: hid)
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
        case .perplexity(let p):
            // Append — a run may emit several rows (bf16/q4/quarot/adapter).
            run.perplexities.append(p)
        case .embedDone(let e):
            // Replace — exactly one embed_done event per batch run.
            run.embed = e
        case .downloadDone:
            // download_done events are handled by downloadModel's dedicated RunHandle;
            // they should never reach a LiveRun's consume path.
            break
        case .status(let line):
            run.appendLog(line)
        case .unknown(let j):
            run.appendLog(j)
        }
    }

    private func finish(_ run: LiveRun, code: Int32) {
        run.status = (code == 0) ? .done : .failed
        // Capture the failure reason from the log when the process exits non-zero.
        // Look for the last non-empty log line that isn't a banner (===) or launch echo ($).
        // This surfaces the actual engine error (e.g. "Error: load model: Model not found…")
        // without fabricating anything — honest-nil when no such line exists.
        if code != 0 {
            run.failureReason = run.log.last(where: { line in
                !line.isEmpty && !line.hasPrefix("$") && !line.hasPrefix("===")
            })
        }
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
        // Fire the completion hook last — status, record, and disk state are all finalised.
        // Used by screens to chain A/B follow-up runs (e.g. base eval → adapter eval).
        run.onComplete?(run)
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
