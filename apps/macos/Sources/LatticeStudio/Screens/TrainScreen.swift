import SwiftUI
import AppKit

// MARK: - 02 TRAIN — LoRA Training Instrument
//
// The flagship screen. Two-pane layout:
//   LEFT  (~360pt, OpaquePanel): configuration form built from ParamRow* primitives.
//   RIGHT (fills):              live instrument — ReadoutWells, StripChart, HeroNumber.
//
// Scrub-to-freeze: when scrubStep != nil, the top ReadoutWells show that step's
// TrainPoint values instead of the latest. A live/frozen indicator sits below the chart.
//
// Store ownership: store.liveRun is read-only here. The store creates and owns the run;
// a re-render of this view must never reset it.

struct TrainScreen: View {
    @Bindable var store: AppStore

    // MARK: Config form state — all @State so they survive screen re-renders.

    // MODEL
    @State private var selectedModelName: String = ""

    // DATASET
    @State private var dataDirPath: String = ""

    // Numeric params (stored as Double for slider binding compatibility)
    @State private var firstLayer: Double = 19
    @State private var steps: Double = 25
    @State private var lr: Double = 1e-3
    @State private var rankStr: String = "8"          // picker: "4","8","16","32"
    @State private var alpha: Double = 16
    @State private var seqLen: Double = 64
    @State private var maxTrain: Double = 3
    @State private var maxValid: Double = 16
    @State private var logEvery: Double = 5

    // SAVE ADAPTER
    @State private var saveAdapterEnabled: Bool = false
    @State private var saveAdapterPath: String = ""

    // SCRUB
    @State private var scrubStep: Int? = nil

    // MARK: Derived helpers

    private var modelNames: [String] {
        store.models.filter { !$0.isEmbedding }.map(\.name)
    }

    private var resolvedModel: ModelInfo? {
        store.models.first { $0.name == selectedModelName } ?? store.targetModel
    }

    private var isRunning: Bool {
        store.liveRun(matching: [.train])?.status == .running
    }

    // MARK: Body

    var body: some View {
        ScreenScaffold(
            screen: .train,
            subtitle: liveSubtitle,
            trailing: {
                if let run = store.liveRun(matching: [.train]) {
                    liveStatusBadge(run: run)
                }
            }
        ) {
            HStack(alignment: .top, spacing: 0) {
                configColumn
                    .frame(width: 360)

                // 1px hairline separator between panes
                Theme.Palette.hairline
                    .frame(width: 1)
                    .frame(maxHeight: .infinity)

                liveInstrumentColumn
                    .frame(maxWidth: .infinity)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onAppear {
            applyDefaults()
        }
        .onChange(of: store.targetModel?.name) { _, newName in
            // Only update model selection if the user hasn't already picked one manually
            if selectedModelName.isEmpty, let name = newName {
                selectedModelName = name
                updateAdapterDefaultPath(modelName: name)
            }
        }
    }

    // MARK: Subtitle

    private var liveSubtitle: String {
        guard let run = store.liveRun(matching: [.train]) else {
            return resolvedModel.map { "\($0.name) · configure and press TRAIN" }
                ?? "select a model and press TRAIN"
        }
        switch run.status {
        case .running:
            let step = run.currentStep
            let total = run.totalSteps.map { "/\($0)" } ?? ""
            return "\(run.modelName) · step \(step)\(total)"
        case .paused:
            return "\(run.modelName) · paused"
        case .done:
            return "\(run.modelName) · done"
        case .failed:
            return "\(run.modelName) · failed"
        case .idle:
            return run.modelName
        }
    }

    // MARK: Trailing header badge

    @ViewBuilder
    private func liveStatusBadge(run: LiveRun) -> some View {
        HStack(spacing: Theme.Space.sm) {
            switch run.status {
            case .running:
                GatePill(.run, label: "RUN")
            case .paused:
                GatePill(.warn, label: "PAUSED")
            case .done:
                if let best = run.bestVal, let base = run.baseNLL {
                    GatePill(best < base ? .pass : .warn,
                             label: "DONE · val \(String(format: "%.4f", best))")
                } else {
                    GatePill(.pass, label: "DONE")
                }
            case .failed:
                GatePill(.fail, label: "FAILED")
            case .idle:
                EmptyView()
            }
        }
    }

    // MARK: - LEFT: Config Column

    private var configColumn: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 0) {

                // Section header: MODEL
                sectionLabel("MODEL")

                if modelNames.isEmpty {
                    ParamRow(label: "MODEL", value: "— none found —")
                } else {
                    ParamRowPicker(
                        label: "MODEL",
                        options: modelNames,
                        selection: Binding(
                            get: {
                                selectedModelName.isEmpty
                                    ? (modelNames.first ?? "")
                                    : selectedModelName
                            },
                            set: { newVal in
                                selectedModelName = newVal
                                updateAdapterDefaultPath(modelName: newVal)
                            }
                        )
                    )
                }

                // Section header: DATASET
                sectionLabel("DATASET")

                HStack(spacing: 0) {
                    ParamRowField(
                        label: "DATA DIR",
                        text: $dataDirPath,
                        placeholder: "path/to/data/"
                    )
                    // "Choose…" button inlined at the row edge
                    Button("Choose…") {
                        chooseDataDir()
                    }
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .buttonStyle(.plain)
                    .padding(.trailing, Theme.Space.lg)
                }

                // Section header: ARCHITECTURE
                sectionLabel("ARCHITECTURE")

                ParamRowSlider(
                    label: "FIRST LAYER",
                    value: $firstLayer,
                    range: 0...23,
                    step: 1,
                    format: "%.0f"
                )

                // Section header: TRAINING
                sectionLabel("TRAINING")

                ParamRowSlider(
                    label: "STEPS",
                    value: $steps,
                    range: 1...500,
                    step: 1,
                    format: "%.0f"
                )

                ParamRowSlider(
                    label: "LR",
                    value: $lr,
                    range: 1e-5...5e-3,
                    format: "%.1e"
                )

                // Section header: LORA
                sectionLabel("LORA")

                ParamRowPicker(
                    label: "RANK",
                    options: ["4", "8", "16", "32"],
                    selection: $rankStr
                )

                ParamRowSlider(
                    label: "ALPHA",
                    value: $alpha,
                    range: 1...64,
                    step: 1,
                    format: "%.0f"
                )

                // Section header: SEQUENCE
                sectionLabel("SEQUENCE")

                ParamRowSlider(
                    label: "SEQ LEN",
                    value: $seqLen,
                    range: 16...512,
                    step: 16,
                    format: "%.0f"
                )

                ParamRowSlider(
                    label: "MAX TRAIN",
                    value: $maxTrain,
                    range: 1...64,
                    step: 1,
                    format: "%.0f"
                )

                ParamRowSlider(
                    label: "MAX VALID",
                    value: $maxValid,
                    range: 0...64,
                    step: 1,
                    format: "%.0f"
                )

                ParamRowSlider(
                    label: "LOG EVERY",
                    value: $logEvery,
                    range: 1...50,
                    step: 1,
                    format: "%.0f"
                )

                // Section header: SAVE ADAPTER
                sectionLabel("SAVE ADAPTER")

                ParamRowToggle(label: "SAVE ADAPTER", isOn: $saveAdapterEnabled)

                if saveAdapterEnabled {
                    ParamRowField(
                        label: "OUTPUT PATH",
                        text: $saveAdapterPath,
                        placeholder: "adapters/<model>-r<rank>.safetensors"
                    )
                }

                // Spacer pushes CTA to bottom
                Spacer(minLength: Theme.Space.xl)

                // CTA — the ONE teal button on this screen
                ctaButton

            }
        }
        .instrumentPanel()
    }

    // MARK: Section label helper (11pt all-caps, hairline-ruled top)

    @ViewBuilder
    private func sectionLabel(_ title: String) -> some View {
        HStack {
            Text(title)
                .instrumentLabel()
            Spacer()
        }
        .frame(height: Theme.Space.rowHeight)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .top) {
            Theme.Palette.hairline.frame(height: 1)
        }
        .padding(.top, Theme.Space.sm)
    }

    // MARK: CTA button

    private var ctaButton: some View {
        HStack {
            Spacer()

            Button {
                launchTraining()
            } label: {
                HStack(spacing: Theme.Space.sm) {
                    Text("▶ TRAIN")
                        .font(Theme.Fonts.display(13, .semibold))
                    KeyCapChip("⌘↵")
                }
                .foregroundStyle(Theme.Palette.canvas)
                .padding(.horizontal, Theme.Space.lg)
                .padding(.vertical, Theme.Space.sm)
                .background(isRunning ? Theme.Palette.inkDim : Theme.Palette.signal)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
            }
            .buttonStyle(.plain)
            .disabled(isRunning)
            .keyboardShortcut(.return, modifiers: .command)

            Spacer()
        }
        .padding(.vertical, Theme.Space.lg)
        .padding(.horizontal, Theme.Space.lg)
    }

    // MARK: - RIGHT: Live Instrument Column

    @ViewBuilder
    private var liveInstrumentColumn: some View {
        if let run = store.liveRun(matching: [.train]) {
            activeRunView(run: run)
        } else {
            emptyState
        }
    }

    // MARK: Empty state

    private var emptyState: some View {
        VStack(spacing: Theme.Space.lg) {
            Spacer()
            VStack(spacing: Theme.Space.sm) {
                Text("02")
                    .font(Theme.Fonts.mono(34, .bold))
                    .foregroundStyle(Theme.Palette.hairline)
                    .monospacedDigit()
                Text("configure a run and press TRAIN")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.inkDim)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: Active run view

    @ViewBuilder
    private func activeRunView(run: LiveRun) -> some View {
        VStack(alignment: .leading, spacing: 0) {

            // TOP STRIP: ReadoutWells
            readoutStrip(run: run)
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.lg)
                .padding(.bottom, Theme.Space.sm)

            // 1px hairline rule under wells
            Theme.Palette.hairline
                .frame(height: 1)

            // CENTER: Chart region — hero + oscilloscope
            VStack(alignment: .leading, spacing: Theme.Space.md) {

                // HERO NUMBER — best (or current) val NLL
                heroSection(run: run)
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.md)

                // OSCILLOSCOPE — fills remaining space
                StripChart(
                    points: run.points,
                    series: .loss,
                    ghostBaseline: run.baseNLL,
                    ghostLabel: "base NLL",
                    scrubStep: $scrubStep
                )
                .frame(maxWidth: .infinity)
                .frame(height: 200)
                .padding(.horizontal, Theme.Space.lg)

                // Scrub indicator
                scrubIndicator(run: run)
                    .padding(.horizontal, Theme.Space.lg)

            }

            // 1px hairline rule above controls
            Theme.Palette.hairline
                .frame(height: 1)
                .padding(.top, Theme.Space.sm)

            // BOTTOM: Status + controls
            controlStrip(run: run)
                .padding(.horizontal, Theme.Space.lg)
                .padding(.vertical, Theme.Space.md)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // MARK: Readout strip

    @ViewBuilder
    private func readoutStrip(run: LiveRun) -> some View {
        // When scrubStep is set, freeze to that step's data; else use latest.
        let displayPoint: TrainPoint? = frozenPoint(run: run)
        let displayStep: Int = displayPoint?.step ?? run.currentStep
        let displayLoss: Double? = displayPoint?.loss ?? run.currentLoss
        let displayValLoss: Double? = displayPoint?.valLoss ?? run.points.last?.valLoss
        let displayTokS: Double? = displayPoint?.tokS ?? run.points.last?.tokS

        // Delta from base NLL
        let deltaBase: Double? = displayLoss.flatMap { l in run.baseNLL.map { b in l - b } }

        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: Theme.Space.sm) {

                // STEP
                ReadoutWell(
                    label: "STEP",
                    value: run.totalSteps.map {
                        "\(displayStep)/\($0)"
                    } ?? "\(displayStep)"
                )

                // TRAIN NLL
                ReadoutWell(
                    label: "TRAIN NLL",
                    value: displayLoss.map { String(format: "%.4f", $0) } ?? "—"
                )

                // HELD-OUT (val loss)
                ReadoutWell(
                    label: "HELD-OUT",
                    value: displayValLoss.map { String(format: "%.4f", $0) } ?? "—"
                )

                // DELTA FROM BASE
                ReadoutWell(
                    label: "Δ FROM BASE",
                    value: deltaBase.map { String(format: "%+.4f", $0) } ?? "—",
                    delta: deltaBase.map { d in
                        ReadoutWell.DeltaInfo(
                            String(format: "%.4f", abs(d)),
                            d < 0 ? .down : .up
                        )
                    }
                )

                // TOK/S
                ReadoutWell(
                    label: "TOK/S",
                    value: displayTokS.map { String(format: "%.0f", $0) } ?? "—",
                    unit: "t/s"
                )

                // BEST VAL
                ReadoutWell(
                    label: "BEST VAL",
                    value: run.bestVal.map { String(format: "%.4f", $0) } ?? "—"
                )
            }
        }
    }

    // MARK: Hero section

    private func heroSection(run: LiveRun) -> some View {
        // Headline: best val if available, else current loss, else placeholder
        let heroValue: String
        if let best = run.bestVal {
            heroValue = String(format: "%.4f", best)
        } else if let loss = frozenPoint(run: run)?.loss ?? run.currentLoss {
            heroValue = String(format: "%.4f", loss)
        } else {
            heroValue = "—"
        }

        // Unit subtitle: "BEST VAL" when best is set, else "TRAIN NLL"
        let heroUnit = run.bestVal != nil ? "BEST VAL" : "NLL"

        return HeroNumber(value: heroValue, unit: heroUnit, size: .hero)
    }

    // MARK: Scrub indicator

    @ViewBuilder
    private func scrubIndicator(run: LiveRun) -> some View {
        HStack(spacing: Theme.Space.sm) {
            if let frozen = scrubStep {
                Circle()
                    .fill(Theme.Palette.amber)
                    .frame(width: 5, height: 5)
                Text("FROZEN @ step \(frozen)")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.amber)
                    .monospacedDigit()
                Button("LIVE") {
                    scrubStep = nil
                }
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.signal)
                .buttonStyle(.plain)
            } else {
                Circle()
                    .fill(Theme.Palette.signal)
                    .frame(width: 5, height: 5)
                Text("● LIVE")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .monospacedDigit()
            }
            Spacer()
        }
        .padding(.bottom, Theme.Space.xs)
    }

    // MARK: Control strip (pause/resume/stop + adapter path when done)

    @ViewBuilder
    private func controlStrip(run: LiveRun) -> some View {
        HStack(spacing: Theme.Space.md) {

            // Gate pill status
            switch run.status {
            case .running:
                GatePill(.run, label: "TRAINING")
            case .paused:
                GatePill(.warn, label: "PAUSED")
            case .done:
                if let best = run.bestVal, let base = run.baseNLL {
                    GatePill(
                        best < base ? .pass : .warn,
                        label: best < base
                            ? "PASS · best val \(String(format: "%.4f", best))"
                            : "WARN · best val \(String(format: "%.4f", best))"
                    )
                } else {
                    GatePill(.pass, label: "DONE")
                }
            case .failed:
                GatePill(.fail, label: "FAILED")
            case .idle:
                EmptyView()
            }

            Spacer()

            // Pause / Resume toggle
            if run.status == .running || run.status == .paused {
                Button {
                    if run.status == .paused {
                        store.resumeRun()
                    } else {
                        store.pauseRun()
                    }
                } label: {
                    Text(run.status == .paused ? "▶ RESUME" : "❚❚ PAUSE")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.ink)
                        .monospacedDigit()
                        .padding(.horizontal, Theme.Space.sm)
                        .padding(.vertical, 4)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)

                // Stop button
                Button {
                    store.stopRun()
                } label: {
                    Text("■ STOP")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.crimson)
                        .monospacedDigit()
                        .padding(.horizontal, Theme.Space.sm)
                        .padding(.vertical, 4)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .strokeBorder(Theme.Palette.crimson.opacity(0.5), lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
            }
        }

        // Saved adapter path (when done and path is available)
        if run.status == .done, let adapterPath = run.savedAdapterPath {
            HStack(spacing: Theme.Space.sm) {
                Text("ADAPTER")
                    .instrumentLabel()
                Text(adapterPath)
                    .font(Theme.Fonts.readout)
                    .foregroundStyle(Theme.Palette.signal)
                    .monospacedDigit()
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .padding(.top, Theme.Space.xs)
        }
    }

    // MARK: - Helpers

    // Returns the frozen TrainPoint when scrubStep is set, else nil (caller falls back to latest).
    private func frozenPoint(run: LiveRun) -> TrainPoint? {
        guard let target = scrubStep else { return nil }
        // Find the closest point at or before the target step
        return run.points.last { $0.step <= target }
    }

    // Called once on appear to set sensible defaults from store context.
    private func applyDefaults() {
        // Model: prefer targetModel, else first non-embedding
        if selectedModelName.isEmpty {
            if let target = store.targetModel {
                selectedModelName = target.name
            } else if let first = modelNames.first {
                selectedModelName = first
            }
        }

        // Data dir: (repoRoot ?? "") + "/data"
        if dataDirPath.isEmpty {
            dataDirPath = (store.repoRootPath ?? "") + "/data"
        }

        // Adapter default path
        if saveAdapterPath.isEmpty {
            updateAdapterDefaultPath(modelName: selectedModelName)
        }
    }

    private func updateAdapterDefaultPath(modelName: String) {
        let rank = Int(rankStr) ?? 8
        let base = store.repoRootPath ?? ""
        saveAdapterPath = "\(base)/adapters/\(modelName)-r\(rank).safetensors"
    }

    private func chooseDataDir() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select the directory containing train.jsonl"
        panel.prompt = "Select"
        if panel.runModal() == .OK, let url = panel.url {
            dataDirPath = url.path
        }
    }

    private func launchTraining() {
        guard let modelInfo = resolvedModel else { return }

        // Build data dir URL — if empty fall back to current directory
        let dataURL: URL
        if dataDirPath.isEmpty {
            dataURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        } else {
            dataURL = URL(fileURLWithPath: dataDirPath)
        }

        // Resolve save path
        let saveURL: URL? = saveAdapterEnabled && !saveAdapterPath.isEmpty
            ? URL(fileURLWithPath: saveAdapterPath)
            : nil

        let config = TrainConfig(
            modelDir: modelInfo.path,
            dataDir: dataURL,
            firstLayer: Int(firstLayer),
            steps: Int(steps),
            lr: lr,
            rank: Int(rankStr) ?? 8,
            alpha: alpha,
            seqLen: Int(seqLen),
            maxTrain: Int(maxTrain),
            maxValid: Int(maxValid),
            logEvery: Int(logEvery),
            savePath: saveURL
        )

        // Reset scrub when a new run starts so we're not frozen on old data
        scrubStep = nil

        store.startTrain(config)
    }
}
