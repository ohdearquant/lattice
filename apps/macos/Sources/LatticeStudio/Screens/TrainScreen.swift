import SwiftUI
import AppKit

// MARK: - 03 TRAIN — LoRA Training Instrument
//
// Layout:
//   main canvas = configure card (idle) OR live instrument (when a run is active).
//   advanced knobs live in a toggleable right InspectorShell inspector
//   (default-CLOSED on first visit; ⌘\ or toolbar button reveals it).
//
// Configure card — two grouped OpaquePanel sections:
//
//   TARGET (model + dataset)
//     MODEL  — dropdown from store.workingModel ?? store.defaultModel
//     DATASET — inline folder picker; scans <repoRoot>/data on appear; "Browse…" for custom paths
//
//   LORA / TRAINING
//     RANK   — segmented picker 4 / 8 / 16 / 32
//     LR     — stepper
//     STEPS  — stepper
//
//   TRAIN ⌘↵  — LatticePrimaryButtonStyle CTA (full-width, below both panels)
//
// Advanced (inspector only):
//   ARCHITECTURE: first layer
//   LORA: alpha
//   SEQUENCE: seq len, max train, max valid, log every
//   SAVE ADAPTER: toggle + output path
//
// Scrub-to-freeze: when scrubStep != nil, the top ReadoutWells show that step's
// TrainPoint values instead of the latest. A live/frozen indicator sits below the chart.
//
// Store ownership: store.liveRun is read-only here. The store creates and owns the run;
// a re-render of this view must never reset it.

// MARK: - TrainingDataset
//
// The training binary takes --data-dir pointing at a FOLDER containing train.jsonl + valid.jsonl.
// The selection unit is therefore the folder, not the individual file. We group the raw
// DatasetFileStat results from DataScreen.scanDirectory by parent folder and require train.jsonl.

private struct TrainingDataset: Identifiable {
    let id: String              // folder path == the --data-dir argument
    let name: String            // folder's last path component, shown in the menu
    let folderURL: URL
    let trainFile: DatasetFileStat   // the train.jsonl stat; assigned to store.workingDataset
    let validCount: Int?        // valid.jsonl exampleCount when present; nil = honest-nil
}

struct TrainScreen: View {
    @Bindable var store: AppStore

    // MARK: Config form state — all @State so they survive screen re-renders.

    // MODEL
    @State private var selectedModelName: String = ""

    // Numeric params (stored as Double for stepper binding compatibility)
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

    // Inspector open-once guard — keeps it CLOSED on first visit
    @State private var didInitInspector = false

    // DATASET PICKER — inline scan within Train; no navigation required
    @State private var availableDatasets: [TrainingDataset] = []
    @State private var datasetFolder: String = ""     // folder being scanned; default <repoRoot>/data
    @State private var isScanningDatasets: Bool = false
    @State private var datasetScanError: String? = nil

    // MARK: Derived helpers

    private var modelNames: [String] {
        // LoRA training requires BF16 weights — Q4 models carry no model.safetensors
        // and train_grad_full errors immediately with "Model not found".
        store.models.filter { $0.format == .bf16 && !$0.isEmbedding }.map(\.name)
    }

    private var resolvedModel: ModelInfo? {
        // Only return a model that is bf16 — resolving to a Q4 target would pass a path
        // that train_grad_full cannot load (no model.safetensors).
        let candidate = store.models.first { $0.name == selectedModelName }
        if let c = candidate, c.format == .bf16 { return c }
        // Fall back to the working/default target only when it is also bf16.
        return store.targetModel.flatMap { $0.format == .bf16 ? $0 : nil }
    }

    private var isRunning: Bool {
        store.liveRun(matching: [.train])?.status == .running
    }

    private var firstLayerCaption: String {
        let first = Int(firstLayer)
        if first == 0 { return "adapts every layer" }
        return "adapts layer \(first) to last · layers 0–\(first - 1) frozen"
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
            liveInstrumentColumn
        }
        .inspector(isPresented: $store.inspectorPresented) {
            advancedInspector
                .inspectorColumnWidth(min: 300, ideal: 320, max: 380)
        }
        .onAppear {
            applyDefaults()
            closeInspectorOnce()
            if datasetFolder.isEmpty {
                datasetFolder = (store.repoRootPath ?? "") + "/data"
            }
            scanDatasets()
        }
        .onChange(of: store.targetModel?.name) { _, newName in
            // Only update model selection if the user hasn't already picked one manually
            if selectedModelName.isEmpty, let name = newName {
                selectedModelName = name
                updateAdapterDefaultPath(modelName: name)
            }
        }
    }

    // MARK: Close inspector on first visit (progressive disclosure default)

    private func closeInspectorOnce() {
        guard !didInitInspector else { return }
        didInitInspector = true
        store.inspectorPresented = false
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

    // MARK: - Advanced Inspector (right inspector — opened via ⌘\ or toolbar)

    private var advancedInspector: some View {
        InspectorShell(title: "Advanced") {
            VStack(spacing: 0) {
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(spacing: 0) {

                        // ARCHITECTURE
                        sectionLabel("ARCHITECTURE")

                        ParamRowStepper(
                            label: "FIRST LAYER",
                            value: $firstLayer,
                            range: 0...23,
                            step: 1,
                            format: "%.0f",
                            caption: firstLayerCaption
                        )

                        // LORA
                        sectionLabel("LORA")

                        ParamRowStepper(
                            label: "ALPHA",
                            value: $alpha,
                            range: 1...64,
                            step: 1,
                            format: "%.0f"
                        )

                        // SEQUENCE
                        sectionLabel("SEQUENCE")

                        ParamRowStepper(
                            label: "SEQ LEN",
                            value: $seqLen,
                            range: 16...512,
                            step: 16,
                            format: "%.0f"
                        )

                        ParamRowStepper(
                            label: "MAX TRAIN",
                            value: $maxTrain,
                            range: 1...64,
                            step: 1,
                            format: "%.0f"
                        )

                        ParamRowStepper(
                            label: "MAX VALID",
                            value: $maxValid,
                            range: 0...64,
                            step: 1,
                            format: "%.0f"
                        )

                        ParamRowStepper(
                            label: "LOG EVERY",
                            value: $logEvery,
                            range: 1...50,
                            step: 1,
                            format: "%.0f"
                        )

                        // SAVE ADAPTER
                        sectionLabel("SAVE ADAPTER")

                        ParamRowToggle(label: "SAVE ADAPTER", isOn: $saveAdapterEnabled)

                        if saveAdapterEnabled {
                            ParamRowField(
                                label: "OUTPUT PATH",
                                text: $saveAdapterPath,
                                placeholder: "adapters/<model>-r<rank>.safetensors"
                            )
                        }

                    }
                }
            }
        }
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

    // MARK: - Live Instrument Column (main canvas)

    @ViewBuilder
    private var liveInstrumentColumn: some View {
        if let run = store.liveRun(matching: [.train]) {
            activeRunView(run: run)
        } else {
            configureCard
        }
    }

    // MARK: Configure card (idle state)
    //
    // Two carded groups separated by a gap — makes the hierarchy visible without section banners:
    //
    //   [TARGET]  model + dataset
    //   [PARAMS]  rank + LR + steps
    //   [TRAIN ⌘↵]  full-width CTA
    //   [Advanced ⌘\]  footer hint

    private var configureCard: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(alignment: .leading, spacing: Theme.Space.md) {

                // ── TARGET — model + dataset ──────────────────────────────
                OpaquePanel {
                    VStack(spacing: 0) {
                        groupHeader("TARGET")

                        // MODEL
                        if modelNames.isEmpty {
                            ParamRow(label: "MODEL", value: "— no models found —")
                        } else {
                            ParamRowMenu(
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

                        // DATASET — inline status + actionable button
                        datasetRow
                    }
                }

                // ── PARAMS — rank / LR / steps ───────────────────────────
                OpaquePanel {
                    VStack(spacing: 0) {
                        groupHeader("LORA + TRAINING")

                        ParamRowPicker(
                            label: "RANK",
                            options: ["4", "8", "16", "32"],
                            selection: $rankStr
                        )

                        ParamRowStepper(
                            label: "LR",
                            value: $lr,
                            range: 1e-5...5e-3,
                            format: "%.1e"
                        )

                        ParamRowStepper(
                            label: "STEPS",
                            value: $steps,
                            range: 1...500,
                            step: 1,
                            format: "%.0f"
                        )
                    }
                }

                // ── TRAIN CTA ─────────────────────────────────────────────
                Button("Train") { launchTraining() }
                    .buttonStyle(LatticePrimaryButtonStyle(height: 34))
                    .frame(maxWidth: .infinity)
                    .disabled(isRunning || modelNames.isEmpty || store.workingDataset == nil)
                    .keyboardShortcut(.return, modifiers: .command)

                // Honest notice when no BF16 model is available (Q4-only setup)
                if modelNames.isEmpty {
                    HStack(spacing: Theme.Space.xs) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.amber)
                        Text("No BF16 model found — LoRA training requires a full-precision model. Quantized models cannot be trained.")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                    .padding(.horizontal, Theme.Space.xs)
                }

                // ── ADVANCED HINT ─────────────────────────────────────────
                HStack(spacing: Theme.Space.xs) {
                    Text("Advanced: alpha, sequence, layer range, save adapter")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                    Spacer()
                    KeyCapChip("⌘\\")
                }
                .padding(.horizontal, Theme.Space.xs)

                Spacer()
            }
            .padding(Theme.Space.lg)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // MARK: Group header (dim all-caps label, no hairline — card border is enough)

    @ViewBuilder
    private func groupHeader(_ title: String) -> some View {
        Text(title)
            .instrumentLabel()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, Theme.Space.lg)
            .padding(.top, Theme.Space.md)
            .padding(.bottom, Theme.Space.xs)
    }

    // MARK: Dataset row — inline folder picker (no navigation to Data tab required)

    @ViewBuilder
    private var datasetRow: some View {
        if isScanningDatasets {
            datasetRowScanning
        } else if !availableDatasets.isEmpty {
            datasetRowMenu
        } else {
            datasetRowEmpty
        }
    }

    // Sub-state: scanning in progress
    private var datasetRowScanning: some View {
        HStack(spacing: Theme.Space.sm) {
            Text("DATASET")
                .instrumentLabel()
            Spacer()
            ProgressView()
                .progressViewStyle(.circular)
                .controlSize(.mini)
            Text("scanning…")
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
        }
        .frame(minHeight: Theme.Space.rowHeightComfortable)
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.xs)
        .overlay(alignment: .top) { Theme.Palette.hairline.frame(height: 1) }
        .overlay(alignment: .bottom) { Theme.Palette.hairline.frame(height: 1) }
    }

    // Sub-state: datasets available — inline dropdown + count caption
    private var datasetRowMenu: some View {
        let selectedName = availableDatasets.first {
            $0.trainFile.id == store.workingDataset?.id
        }?.name ?? availableDatasets.first?.name ?? ""

        let menuBinding = Binding<String>(
            get: { selectedName },
            set: { chosen in
                if let match = availableDatasets.first(where: { $0.name == chosen }) {
                    store.workingDataset = match.trainFile
                }
            }
        )

        let activeDS = availableDatasets.first { $0.name == selectedName }

        return VStack(spacing: 0) {
            ParamRowMenu(
                label: "DATASET",
                options: availableDatasets.map(\.name),
                selection: menuBinding
            )
            if let ds = activeDS {
                datasetCountCaption(ds: ds)
            }
        }
    }

    // Caption row: train/valid counts + Browse + optional Data tab link
    private func datasetCountCaption(ds: TrainingDataset) -> some View {
        var countText = "\(ds.trainFile.exampleCount) train"
        if let v = ds.validCount { countText += " · \(v) valid" }
        countText += " examples"

        return HStack(spacing: Theme.Space.sm) {
            Spacer()
            Text(countText)
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
                .monospacedDigit()
            Button("Browse…") { browseDatasetFolder() }
                .buttonStyle(LatticeSecondaryButtonStyle())
                .controlSize(.small)
            Button("Inspect in Data") { store.selection = .data }
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
                .buttonStyle(.plain)
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.bottom, Theme.Space.xs)
    }

    // Sub-state: no datasets found or scan error
    private var datasetRowEmpty: some View {
        VStack(spacing: 0) {
            HStack(spacing: Theme.Space.sm) {
                Text("DATASET")
                    .instrumentLabel()
                Spacer()
                if let err = datasetScanError {
                    GatePill(.fail, label: "scan error")
                    Text(err)
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.amber)
                        .lineLimit(1)
                        .truncationMode(.tail)
                } else {
                    Text("— no datasets found in \(datasetFolder) —")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
                Button("Browse…") { browseDatasetFolder() }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                    .controlSize(.small)
            }
            .frame(minHeight: Theme.Space.rowHeightComfortable)
            .padding(.horizontal, Theme.Space.lg)
            .padding(.vertical, Theme.Space.xs)
            .overlay(alignment: .top) { Theme.Palette.hairline.frame(height: 1) }
            .overlay(alignment: .bottom) { Theme.Palette.hairline.frame(height: 1) }

            HStack(spacing: Theme.Space.sm) {
                Spacer()
                Text("build or inspect datasets in Data")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                Button("Open Data tab") { store.selection = .data }
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .buttonStyle(.plain)
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.bottom, Theme.Space.xs)
        }
    }

    // MARK: Dataset scan — groups DataScreen.scanDirectory results into training-ready folders

    private func scanDatasets() {
        let dir = datasetFolder.trimmingCharacters(in: .whitespaces)
        guard !dir.isEmpty else { return }

        isScanningDatasets = true
        datasetScanError = nil

        // Capture the current workingDataset before the async hop to preserve cross-tab selections.
        let existingDataset = store.workingDataset

        Task.detached(priority: .userInitiated) {
            let result = await DataScreen.scanDirectory(path: dir)
            await MainActor.run {
                isScanningDatasets = false
                switch result {
                case .success(let stats):
                    // Group by parent folder; only folders with train.jsonl qualify.
                    let byFolder = Dictionary(grouping: stats) {
                        $0.url.deletingLastPathComponent().path
                    }
                    var datasets: [TrainingDataset] = []
                    for (folderPath, files) in byFolder {
                        guard let trainFile = files.first(where: { $0.name == "train.jsonl" }) else {
                            continue
                        }
                        let validFile = files.first(where: { $0.name == "valid.jsonl" })
                        let folderURL = URL(fileURLWithPath: folderPath, isDirectory: true)
                        datasets.append(TrainingDataset(
                            id: folderPath,
                            name: folderURL.lastPathComponent,
                            folderURL: folderURL,
                            trainFile: trainFile,
                            validCount: validFile?.exampleCount
                        ))
                    }
                    datasets.sort { $0.name < $1.name }

                    // Preserve a cross-tab selection even when its folder isn't in the scanned set.
                    if let existing = existingDataset {
                        let existingFolder = existing.url.deletingLastPathComponent().path
                        if !datasets.contains(where: { $0.id == existingFolder }) {
                            let syntheticFolderURL = URL(fileURLWithPath: existingFolder, isDirectory: true)
                            datasets.append(TrainingDataset(
                                id: existingFolder,
                                name: syntheticFolderURL.lastPathComponent,
                                folderURL: syntheticFolderURL,
                                trainFile: existing,
                                validCount: nil
                            ))
                        }
                    }

                    availableDatasets = datasets

                    // Auto-select first dataset when nothing is chosen yet.
                    if store.workingDataset == nil, let first = datasets.first {
                        store.workingDataset = first.trainFile
                    }

                case .failure(let err):
                    datasetScanError = err.message
                    availableDatasets = []
                }
            }
        }
    }

    // MARK: Browse — NSOpenPanel for custom dataset folder

    private func browseDatasetFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Choose a folder containing train.jsonl"
        panel.prompt = "Select"
        if !datasetFolder.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: datasetFolder, isDirectory: true)
        }
        if panel.runModal() == .OK, let url = panel.url {
            datasetFolder = url.path
            scanDatasets()
        }
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
                Text("LIVE")
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
                VStack(alignment: .leading, spacing: Theme.Space.xs) {
                    GatePill(.fail, label: "FAILED")
                    if let reason = run.failureReason {
                        Text(reason)
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.crimson)
                            .lineLimit(3)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
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
                    HStack(spacing: 4) {
                        if run.status == .paused {
                            Image(systemName: "play.fill")
                                .font(.system(size: 10, weight: .medium))
                            Text("Resume")
                        } else {
                            Image(systemName: "pause.fill")
                                .font(.system(size: 10, weight: .medium))
                            Text("Pause")
                        }
                    }
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
                    HStack(spacing: 4) {
                        Image(systemName: "stop.fill")
                            .font(.system(size: 10, weight: .medium))
                        Text("Stop")
                    }
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

    private func launchTraining() {
        guard let modelInfo = resolvedModel else { return }

        // Build data dir URL from the working dataset selected in the Data tab,
        // falling back to the current directory if no dataset has been chosen yet.
        let dataURL: URL
        if let dataset = store.workingDataset {
            // Use the parent directory of the chosen dataset file
            dataURL = dataset.url.deletingLastPathComponent()
        } else {
            dataURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
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
