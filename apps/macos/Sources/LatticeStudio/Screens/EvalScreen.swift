import SwiftUI
import AppKit

// MARK: - 05 EVAL
//
// Unified evaluation workspace: three tabs for measuring and comparing models.
//
// Layout: ScreenScaffold (no right inspector — hasInspector returns false for .eval).
// The screen is split into a fixed-width left config rail and a flex right content area.
//
//   PPL tab     — Perplexity measurement (Stage 2: left rail + result pane)
//   COMPARE tab — N-way generation compare (Stage 3 implementation)
//   SIMILAR tab — Semantic similarity probe / Embeddings (Stage 4 implementation)
//
// Stage 2: PPL tab is fully wired to eval_perplexity via the existing EvalConfig /
// runEval() / RunHandle pattern. Compare and Similar remain placeholders.
//
// Device-label contract (Issue #7):
//   eval_perplexity has THREE modes; the hardware backend is FORMAT-DRIVEN, not toggleable:
//     BF16  model → --model-dir        → CPU                (no Metal BF16 PPL path)
//     Q4    model → --q4-dir           → GPU Metal          (MetalQwen35State)
//     QuaRot model → --quarot-q4-dir  → GPU Metal          (MetalQwen35State)
//   The UI renders an honest READ-ONLY label derived from ModelFormat. No dead toggle exists.
//   evalUseGPU on the store is for the Compare tab generation backend (Stage 3), NOT PPL.
//
// Screen enum: .eval (⌘5), replacing the old .embed destination.
// hasInspector = false — the left rail is the config surface (no right inspector panel).

// MARK: - EvalTab

private enum EvalTab: String, CaseIterable {
    case ppl     = "PPL"
    case compare = "COMPARE"
    case similar = "SIMILAR"
}

// MARK: - PPL phase tracker

private enum PPLPhase { case idle, base, quant }

// MARK: - Compare slot configuration
//
// A slot is one column's configuration in the COMPARE tab.
// Up to 4 slots (design max); minimum 2 for a meaningful comparison.
// Slots are local @State — they are per-session UI state, not shared across screens.

private struct CompareSlot: Identifiable {
    let id = UUID()
    var modelName: String = ""      // selected model name ("" = unset)
    var adapterName: String = "none" // "none" or an adapter name

    // Device selection is per-slot: GPU Metal (chat_metal) or CPU (generate_lora).
    // This is a REAL binary choice — both binaries exist and generate:
    //   false → generate_lora (CPU BF16): args include --model-dir, --lora optional
    //   true  → chat_metal (GPU Metal):   args include --model-dir, no --lora support
    // A BF16 model on CPU can load adapters; GPU Metal cannot. When the slot has an
    // adapter set, useGPU is forced false (CPU only) and the picker is disabled.
    var useGPU: Bool = false
}

// MARK: - EvalScreen

struct EvalScreen: View {
    @Bindable var store: AppStore

    // MARK: Local state

    // Active tab — synced with store so it survives navigation.
    @State private var evalTab: EvalTab = .ppl

    // PPL section state — kept local (@State) because PPL results are per-session,
    // not cross-screen shared state.
    @State private var pplPhase: PPLPhase = .idle
    @State private var pplBase: LatticeEvent.Perplexity?      // bf16 result
    @State private var pplQuant: LatticeEvent.Perplexity?     // q4/quarot result
    @State private var pplCorpusURL: URL?                     // nil = use embedded default
    @State private var pplError: String?
    @State private var pplMeasuredModelID: String?            // guards stale result delivery

    // MARK: Similar state (Stage 4)
    //
    // All Similar state is local (@State) — the similarity probe is a per-session
    // interactive tool; there is no need to hoist it into the store.
    // This mirrors the prior EmbeddingsScreen pattern exactly.

    /// Name of the embedding model currently selected in the SIMILAR tab picker.
    @State private var similarSelectedModelName: String = ""
    /// Texts the user wants to embed. Two rows pre-filled with illustrative examples.
    @State private var similarTexts: [String] = [
        "A cat sits quietly on the warm windowsill.",
        "A feline rests in the sunshine by the window.",
        "The stock market rallied sharply on Tuesday afternoon."
    ]
    /// Result from the most recent successful embed run. Nil until the first run completes.
    @State private var similarEmbedResult: LatticeEvent.EmbedDone? = nil
    /// Snapshot of similarTexts at the moment the last embed run was launched.
    /// Used to label matrix rows/columns without re-rendering on every text edit.
    @State private var similarResultTexts: [String] = []
    /// True while an embed run is in-flight.
    @State private var similarIsEmbedding: Bool = false
    /// Error message from the most recent failed embed run. Nil when no error.
    @State private var similarEmbedError: String? = nil

    // MARK: Compare state (Stage 3)
    //
    // Slots are the column configurators (model + adapter + device per column).
    // Start with 2 slots (the common A/B case); user can add up to 4 or remove down to 2.
    @State private var compareSlots: [CompareSlot] = [CompareSlot(), CompareSlot()]
    // Composer text for the shared prompt across all compare columns.
    @State private var comparePromptText: String = ""

    // MARK: Helpers

    // Embedding-format models discovered in the model cache.
    // Used by the SIMILAR tab picker to restrict the selection to embed-capable models.
    private var embedModels: [ModelInfo] {
        store.models.filter { $0.isEmbedding }.sorted { $0.name < $1.name }
    }

    // Known-working embed model short names, shown when no embedding model has been
    // discovered in the local cache (they will download on first use).
    private let staticCachedEmbedModels: [String] = [
        "bge-small-en-v1.5",
        "all-minilm-l6-v2",
        "multilingual-e5-small",
        "paraphrase-multilingual-minilm-l12-v2"
    ]

    // Picker options for the SIMILAR tab: real discovered embed models when available,
    // otherwise the static fallback list.
    // This exactly mirrors EmbeddingsScreen.pickerOptions.
    private var similarPickerOptions: [String] {
        if !embedModels.isEmpty { return embedModels.map(\.name) }
        return staticCachedEmbedModels
    }

    // True when the picker is showing the static fallback (no embed model discovered yet).
    private var similarUsingStaticFallback: Bool { embedModels.isEmpty }

    // Preconditions for the "Embed Texts" CTA in the SIMILAR tab.
    // At least 2 non-empty texts are required; a model must be selected; no run in-flight.
    private var similarCanEmbed: Bool {
        guard !similarIsEmbedding, !similarSelectedModelName.isEmpty else { return false }
        let nonEmpty = similarTexts.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        return nonEmpty.count >= 2
    }

    // Non-embedding models sorted by name; these are the selectable items for PPL and Compare.
    private var evalModels: [ModelInfo] {
        store.models.filter { !$0.isEmbedding }.sorted { $0.name < $1.name }
    }

    // BF16-only models available for generation (generate_lora is bf16-only on CPU;
    // chat_metal accepts bf16 too — both binaries require BF16 safetensors source).
    private var compareModels: [ModelInfo] {
        store.models.filter { $0.format == .bf16 }.sorted { $0.name < $1.name }
    }

    // Whether a compare generation is currently in-flight.
    private var compareIsRunning: Bool {
        store.evalComparePhase >= 0
    }

    // True when ALL configured slots have a model selected (minimum precondition to run).
    private var compareCanRun: Bool {
        guard !compareSlots.isEmpty, !compareIsRunning else { return false }
        guard !comparePromptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        return compareSlots.allSatisfy { !$0.modelName.isEmpty }
    }

    // Resolved ModelInfo for a slot (nil if not set or not found).
    private func resolvedModel(for slot: CompareSlot) -> ModelInfo? {
        guard !slot.modelName.isEmpty else { return nil }
        return store.models.first { $0.name == slot.modelName }
    }

    // Resolved AdapterInfo for a slot (nil if "none" or not found on the model).
    private func resolvedAdapter(for slot: CompareSlot) -> AdapterInfo? {
        guard slot.adapterName != "none" else { return nil }
        return resolvedModel(for: slot)?.adapters.first { $0.name == slot.adapterName }
    }

    // Adapter options for a given model — "none" + usable adapters (weightFile != nil).
    // Mirrors ChatScreen.adapterOptions: only adapters with a resolved weight file are shown.
    private func adapterOptions(for model: ModelInfo?) -> [String] {
        let adapters = model?.adapters.filter { $0.weightFile != nil }.map(\.name) ?? []
        return ["none"] + adapters
    }

    // The device label for a slot, honest and format-driven:
    //   - Slots with an adapter are forced CPU (generate_lora is the only loader with --lora).
    //   - Otherwise the slot's useGPU controls the binary choice.
    private func deviceLabel(for slot: CompareSlot) -> String {
        let hasAdapter = slot.adapterName != "none"
        if hasAdapter { return "CPU (adapter)" }
        return slot.useGPU ? "GPU Metal" : "CPU"
    }

    // The column header label snapshotted for a slot at send time.
    // Format: "model-name · adapter · device"
    private func columnLabel(for slot: CompareSlot) -> String {
        let modelPart = slot.modelName.isEmpty ? "—" : slot.modelName
        let adapterPart = slot.adapterName == "none" ? "base" : slot.adapterName
        let devicePart = deviceLabel(for: slot)
        return "\(modelPart) · \(adapterPart) · \(devicePart)"
    }

    // The currently selected PPL model (single-select; stored by name in the shared set).
    private var selectedPPLModel: ModelInfo? {
        guard let name = store.evalSelectedModelNames.first else { return nil }
        return store.models.first { $0.name == name }
    }

    // Derive the honest hardware label from ModelFormat.
    // This is the ONLY source of truth for the CPU vs GPU Metal label.
    // There is no toggle — the backend is determined by the model's format.
    private func hardwareLabel(for format: ModelFormat) -> String {
        switch format {
        case .bf16:      return "CPU"
        case .q4:        return "GPU Metal"
        case .quarot:    return "GPU Metal"
        case .embedding: return "CPU"
        case .unknown:   return "—"
        }
    }

    // Short format badge: "BF16", "Q4", "QuaRot"
    private func formatBadge(for format: ModelFormat) -> String {
        switch format {
        case .bf16:      return "BF16"
        case .q4:        return "Q4"
        case .quarot:    return "QuaRot"
        case .embedding: return "Embed"
        case .unknown:   return "—"
        }
    }

    // MARK: Subtitle

    private var subtitle: String {
        if store.evalSelectedModelNames.isEmpty {
            return "select a model to begin"
        }
        let names = store.evalSelectedModelNames.sorted().joined(separator: ", ")
        return "\(store.evalSelectedModelNames.count) model(s) · \(names)"
    }

    // MARK: Body

    var body: some View {
        ScreenScaffold(screen: .eval, subtitle: subtitle) {
            HStack(spacing: 0) {
                // LEFT RAIL — config panel (320pt fixed)
                evalLeftRail
                    .frame(width: 320)

                Theme.Palette.hairline
                    .frame(width: 1)

                // RIGHT CONTENT — three-tab area
                VStack(spacing: 0) {
                    // Tab segmented control
                    HStack {
                        Picker("", selection: $evalTab) {
                            ForEach(EvalTab.allCases, id: \.self) { tab in
                                Text(tab.rawValue).tag(tab)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 320)

                        Spacer()
                    }
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.md)
                    .padding(.bottom, Theme.Space.sm)

                    // Tab content
                    switch evalTab {
                    case .ppl:
                        pplResultPane
                    case .compare:
                        comparePane
                    case .similar:
                        similarPane
                    }
                }
            }
        }
        // Sync local tab state with store on appear (so navigation to .eval preserves tab).
        .onAppear {
            if let restored = EvalTab(rawValue: store.evalActiveTab) {
                evalTab = restored
            }
            // Apply the default embed model selection on first appear.
            applySimilarDefaults()
        }
        // Persist tab changes to store.
        .onChange(of: evalTab) { _, newTab in
            store.evalActiveTab = newTab.rawValue
        }
        // Clear stale PPL results when the selected model changes.
        .onChange(of: store.evalSelectedModelNames) { _, newNames in
            let firstNew = newNames.first
            if firstNew != pplMeasuredModelID {
                pplPhase = .idle
                pplBase = nil
                pplQuant = nil
                pplError = nil
                pplMeasuredModelID = nil
            }
        }
        // Re-apply embed model defaults when the model list changes (e.g. after discovery).
        .onChange(of: store.models) { _, _ in
            applySimilarDefaults()
        }
        // Route streamed gen_token events into the active compare column.
        // This handler runs on every genText change — it guards on evalComparePhase >= 0
        // and evalCompareAwaitingPairID to be inert during non-compare runs.
        .onChange(of: store.liveRun?.genText) { _, newText in
            guard store.liveRun?.kind == .chat else { return }
            let phase = store.evalComparePhase
            guard phase >= 0,
                  let pairID = store.evalCompareAwaitingPairID,
                  let pairIdx = store.evalComparePairs.firstIndex(where: { $0.id == pairID }),
                  phase < store.evalComparePairs[pairIdx].columns.count
            else { return }
            store.evalComparePairs[pairIdx].columns[phase].text = newText ?? ""
        }
    }

    // MARK: - Left Rail

    private var evalLeftRail: some View {
        VStack(alignment: .leading, spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.lg) {
                    if evalTab == .ppl {
                        // PPL: model selector (single) + corpus picker
                        modelsRailSection
                        corpusRailSection
                    } else if evalTab == .compare {
                        // COMPARE: slot configurators + generation knobs
                        compareSlotsRailSection
                        compareGenerationRailSection
                    } else {
                        // SIMILAR: embed model picker (embedding-format models only)
                        similarEmbedModelRailSection
                    }
                }
                .padding(Theme.Space.md)
            }

            Spacer()

            // RUN EVAL button (primary CTA, always visible at rail bottom)
            Divider()
            runEvalButton
                .padding(Theme.Space.md)
        }
    }

    // MARK: Models rail section

    private var modelsRailSection: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("MODEL")
                .instrumentLabel()
                .padding(.bottom, 2)

            Theme.Palette.hairline.frame(height: 1)

            if evalModels.isEmpty {
                Text("No models found. Run `make build` and check the model cache.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                // Single-select for PPL tab — shows all non-embedding models with
                // honest format + hardware label per row.
                VStack(spacing: 0) {
                    ForEach(evalModels) { model in
                        modelRailRow(model)
                    }
                }
                .background(Theme.Palette.panel)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Theme.Palette.hairline, lineWidth: 1)
                )
            }

            // Honest annotation: hardware is format-driven, not a toggle.
            Text("BF16=CPU · Q4/QuaRot=GPU Metal (format-driven, not a toggle)")
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.inkDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // Individual model row in the rail.
    @ViewBuilder
    private func modelRailRow(_ model: ModelInfo) -> some View {
        let isSelected = store.evalSelectedModelNames.contains(model.name)
        Button {
            // Single-select: replace the set with just this model.
            store.evalSelectedModelNames = [model.name]
        } label: {
            HStack(spacing: Theme.Space.sm) {
                // Selection indicator
                Image(systemName: isSelected ? "circle.fill" : "circle")
                    .font(.caption)
                    .foregroundStyle(isSelected ? Theme.Palette.signal : Theme.Palette.inkDim)
                    .frame(width: 14)

                // Model name
                Text(model.name)
                    .font(Theme.Fonts.mono(12))
                    .foregroundStyle(Theme.Palette.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)

                Spacer()

                // Format badge
                Text(formatBadge(for: model.format))
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .frame(minWidth: 36, alignment: .trailing)

                // Hardware label (read-only, derived from format)
                Text(hardwareLabel(for: model.format))
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(model.format.isQuantized ? Theme.Palette.signal : Theme.Palette.inkDim)
                    .frame(minWidth: 64, alignment: .trailing)
            }
            .padding(.horizontal, Theme.Space.sm)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(isSelected ? Theme.Palette.signal.opacity(0.08) : Color.clear)
    }

    // MARK: Corpus rail section

    private var corpusRailSection: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("CORPUS")
                .instrumentLabel()
                .padding(.bottom, 2)

            Theme.Palette.hairline.frame(height: 1)

            HStack(spacing: Theme.Space.sm) {
                if let customURL = pplCorpusURL {
                    Text(customURL.lastPathComponent)
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.ink)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button("Use default") {
                        pplCorpusURL = nil
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                    .font(Theme.Fonts.caption)
                } else {
                    Text("Default (200 tok)")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
                Spacer()
                Button("Choose…") {
                    let panel = NSOpenPanel()
                    panel.canChooseFiles = true
                    panel.canChooseDirectories = false
                    panel.allowsMultipleSelection = false
                    panel.allowedContentTypes = [.plainText]
                    panel.title = "Select corpus text file"
                    if panel.runModal() == .OK, let url = panel.url {
                        pplCorpusURL = url
                    }
                }
                .buttonStyle(LatticeSecondaryButtonStyle())
                .font(Theme.Fonts.caption)
            }
        }
    }

    // MARK: Run Eval button

    private var runEvalButton: some View {
        let canRun: Bool = {
            switch evalTab {
            case .ppl:
                guard let model = selectedPPLModel else { return false }
                if pplPhase != .idle { return false }
                if model.format.isQuantized {
                    let hasSibling = pplSibling(for: model) != nil
                    let hasTokenizer = model.hasTokenizer
                    return hasSibling || hasTokenizer
                }
                return true
            case .compare:
                return compareCanRun
            case .similar:
                return similarCanEmbed
            }
        }()

        let label: String = {
            switch evalTab {
            case .ppl:     return "Measure PPL"
            case .compare: return compareIsRunning ? "Running…" : "Run Compare"
            case .similar: return similarIsEmbedding ? "Embedding…" : "Embed Texts"
            }
        }()

        return Button(label) {
            switch evalTab {
            case .ppl:
                if let model = selectedPPLModel { measurePerplexity(model: model) }
            case .compare:
                sendCompare()
            case .similar:
                runSimilar()
            }
        }
        .buttonStyle(LatticePrimaryButtonStyle())
        .disabled(!canRun)
        .frame(maxWidth: .infinity)
    }

    // MARK: - PPL Result Pane

    private var pplResultPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                if let model = selectedPPLModel {
                    pplResultContent(model: model)
                } else {
                    pplEmptyState
                }
            }
            .padding(Theme.Space.lg)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var pplEmptyState: some View {
        VStack(spacing: Theme.Space.md) {
            Spacer().frame(height: Theme.Space.xxl)
            Text("PERPLEXITY")
                .instrumentLabel()
            Text("Select a model in the left panel, then press \"Measure PPL\".")
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.inkDim)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private func pplResultContent(model: ModelInfo) -> some View {
        let sibling = model.format.isQuantized ? pplSibling(for: model) : nil
        let canMeasure: Bool = {
            if model.format == .bf16 { return true }
            if sibling != nil { return true }
            if model.hasTokenizer { return true }
            return false
        }()

        // Model info header
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            Text("Selected model: \(model.name)")
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.ink)

            // Honest read-only hardware label
            HStack(spacing: Theme.Space.sm) {
                Text("Hardware:")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                Text(hardwareLabel(for: model.format))
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(model.format.isQuantized ? Theme.Palette.signal : Theme.Palette.ink)

                Text("·")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)

                Text(formatBadge(for: model.format))
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
            }

            // Tokenizer / sibling info
            if model.format.isQuantized {
                if let sib = sibling {
                    Text("Tokenizer: \(sib.name)/")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                } else if model.hasTokenizer {
                    Text("Tokenizer: embedded")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
            }
        }

        Theme.Palette.hairline.frame(height: 1)

        if !canMeasure {
            // Honest disabled explanation — no fabricated "measuring" state
            Text("Perplexity needs the BF16 source (for its tokenizer). Keep the source model in the list to compare.")
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.inkDim)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            // In-progress indicator (shown during measurement)
            if pplPhase != .idle {
                let methodLabel = model.format == .quarot ? "QUAROT" : "Q4"
                VStack(alignment: .leading, spacing: Theme.Space.xs) {
                    HStack(spacing: Theme.Space.sm) {
                        ProgressView()
                            .controlSize(.small)
                        let phaseLabel: String = {
                            if pplPhase == .base {
                                return "MEASURING BF16\u{2026} (CPU)"
                            } else {
                                return "MEASURING \(methodLabel)\u{2026} (\(hardwareLabel(for: model.format)))"
                            }
                        }()
                        GatePill(.run, label: phaseLabel)
                    }
                    if pplPhase == .base {
                        Text("CPU BF16 eval takes ~15s on Qwen3.5-0.8B")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                }
            }

            // Error display
            if let err = pplError {
                GatePill(.fail, label: err)
            }

            // Results — only when they belong to the currently selected model
            if pplMeasuredModelID == selectedPPLModel?.id,
               pplBase != nil || pplQuant != nil {
                pplResultWells(model: model)
            }
        }
    }

    // Build the PPL well list outside the @ViewBuilder so Swift's result-builder DSL
    // is not confused by imperative var + append before the first view expression.
    private func pplWells(model: ModelInfo) -> [PPLWellSpec] {
        let methodLabel = model.format == .quarot ? "QUAROT" : "Q4"
        let baseHW  = "CPU"
        let quantHW = hardwareLabel(for: model.format)
        let bf16Value  = pplBase.map { String(format: "%.3f", $0.ppl) } ?? "—"
        let quantValue = pplQuant.map { String(format: "%.3f", $0.ppl) } ?? "—"
        let deltaValue: String = {
            if let base = pplBase, let quant = pplQuant {
                let d = quant.ppl - base.ppl
                return d >= 0
                    ? String(format: "+%.3f", d)
                    : String(format: "%.3f", d)
            }
            return "—"
        }()

        var ws: [PPLWellSpec] = []
        ws.append(PPLWellSpec("BF16 PPL", bf16Value, subtitle: baseHW))
        if model.format != .bf16 {
            ws.append(PPLWellSpec("\(methodLabel) PPL", quantValue, subtitle: quantHW))
            ws.append(PPLWellSpec("\u{0394}PPL", deltaValue, subtitle: ""))
        }
        return ws
    }

    @ViewBuilder
    private func pplResultWells(model: ModelInfo) -> some View {
        let wells = pplWells(model: model)

        LazyVGrid(
            columns: [
                GridItem(.flexible(), spacing: Theme.Space.md),
                GridItem(.flexible(), spacing: Theme.Space.md)
            ],
            spacing: Theme.Space.md
        ) {
            ForEach(wells, id: \.label) { well in
                VStack(alignment: .leading, spacing: 2) {
                    ReadoutWell(label: well.label, value: well.value, unit: well.unit, minHeight: 56)
                    if !well.subtitle.isEmpty {
                        Text(well.subtitle)
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .padding(.leading, 4)
                    }
                }
            }
        }

        // Quality verdict — only when both values are present (honest: no verdict with one value)
        if let base = pplBase, let quant = pplQuant {
            let delta = quant.ppl - base.ppl
            if delta < 0.5 {
                GatePill(.pass, label: "MINIMAL LOSS (\u{0394}<0.5)")
            } else if delta < 1.5 {
                GatePill(.warn, label: "MODERATE LOSS")
            } else {
                GatePill(.fail, label: "SIGNIFICANT LOSS")
            }
        } else if model.format == .bf16, pplBase != nil {
            // Honest baseline label — only BF16 was measured; no quant to compare to
            GatePill(.pass, label: "BASELINE")
        }

        // Timing info if available
        if let base = pplBase, let ms = base.ms {
            let secs = ms / 1000.0
            let tokInfo = base.tokens.map { " · \($0) tokens" } ?? ""
            Text("BF16 eval: \(String(format: "%.1f", secs))s\(tokInfo)")
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.inkDim)
        }
        if let quant = pplQuant, let ms = quant.ms {
            let secs = ms / 1000.0
            let methodLabel2 = model.format == .quarot ? "QuaRot" : "Q4"
            Text("\(methodLabel2) eval: \(String(format: "%.1f", secs))s")
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.inkDim)
        }
    }

    // MARK: - Compare rail sections

    // Slot configurators: one card per slot, + add/remove buttons.
    // Minimum 2 slots; maximum 4 slots (design contract).
    private var compareSlotsRailSection: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            HStack {
                Text("SLOTS")
                    .instrumentLabel()
                Spacer()
                // Remove slot button (disabled at 2 slots)
                if compareSlots.count > 2 {
                    Button {
                        compareSlots.removeLast()
                    } label: {
                        Image(systemName: "minus.circle")
                            .font(.caption)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                    .buttonStyle(.plain)
                    .disabled(compareIsRunning)
                    .help("Remove last column")
                }
                // Add slot button (disabled at 4 slots)
                if compareSlots.count < 4 {
                    Button {
                        compareSlots.append(CompareSlot())
                    } label: {
                        Image(systemName: "plus.circle")
                            .font(.caption)
                            .foregroundStyle(Theme.Palette.signal)
                    }
                    .buttonStyle(.plain)
                    .disabled(compareIsRunning)
                    .help("Add column (max 4)")
                }
            }
            .padding(.bottom, 2)

            Theme.Palette.hairline.frame(height: 1)

            ForEach($compareSlots) { $slot in
                compareSlotCard(slot: $slot)
            }

            Text("Each slot runs independently. Columns with no model are skipped.")
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.inkDim)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func compareSlotCard(slot: Binding<CompareSlot>) -> some View {
        let slotVal = slot.wrappedValue
        let model = resolvedModel(for: slotVal)
        let adapterOptions = adapterOptions(for: model)
        // Adapter presence forces CPU: generate_lora is the only binary with --lora.
        let hasAdapter = slotVal.adapterName != "none"
        let deviceForced = hasAdapter // when true, useGPU is forced false and picker disabled

        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            // Model picker
            HStack(spacing: Theme.Space.xs) {
                Text("Model")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .frame(width: 44, alignment: .leading)
                Picker("", selection: slot.modelName) {
                    Text("—").tag("")
                    ForEach(compareModels, id: \.name) { m in
                        Text(m.name).tag(m.name)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .font(Theme.Fonts.caption)
                .disabled(compareIsRunning)
                // Reset adapter when model changes (avoid stale adapter from a different model).
                .onChange(of: slotVal.modelName) { _, _ in
                    slot.wrappedValue.adapterName = "none"
                }
            }

            // Adapter picker (shown only when the model has usable adapters)
            if let m = model, !m.adapters.filter({ $0.weightFile != nil }).isEmpty {
                HStack(spacing: Theme.Space.xs) {
                    Text("Adapter")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                        .frame(width: 44, alignment: .leading)
                    Picker("", selection: slot.adapterName) {
                        ForEach(adapterOptions, id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .font(Theme.Fonts.caption)
                    .disabled(compareIsRunning)
                }
            }

            // Device picker (GPU/CPU).
            // When an adapter is active, forced to CPU (GPU Metal has no --lora support).
            HStack(spacing: Theme.Space.xs) {
                Text("Device")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .frame(width: 44, alignment: .leading)
                if deviceForced {
                    // Honest read-only label — adapter forces CPU binary.
                    Text("CPU (adapter)")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                } else {
                    Picker("", selection: slot.useGPU) {
                        Text("CPU").tag(false)
                        Text("GPU Metal").tag(true)
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .font(Theme.Fonts.caption)
                    .disabled(compareIsRunning)
                }
            }
        }
        .padding(Theme.Space.sm)
        .background(Theme.Palette.panel)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Theme.Palette.hairline, lineWidth: 1)
        )
    }

    // Generation knobs for the COMPARE tab — mirroring the Chat inspector fields.
    // These bind to the store's evalTempText / evalMaxTokensText / evalSeedText, which are
    // separate from chatTempText / chatMaxTokensText / chatSeedText (eval ≠ chat context).
    private var compareGenerationRailSection: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("GENERATION")
                .instrumentLabel()
                .padding(.bottom, 2)

            Theme.Palette.hairline.frame(height: 1)

            HStack(spacing: Theme.Space.sm) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Max tokens")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                    TextField("256", text: $store.evalMaxTokensText)
                        .textFieldStyle(.roundedBorder)
                        .font(Theme.Fonts.caption)
                        .frame(maxWidth: .infinity)
                        .disabled(compareIsRunning)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Temp")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                    TextField("0.7", text: $store.evalTempText)
                        .textFieldStyle(.roundedBorder)
                        .font(Theme.Fonts.caption)
                        .frame(maxWidth: .infinity)
                        .disabled(compareIsRunning)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Seed (optional)")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                TextField("random", text: $store.evalSeedText)
                    .textFieldStyle(.roundedBorder)
                    .font(Theme.Fonts.caption)
                    .disabled(compareIsRunning)
            }
        }
    }

    // MARK: - Compare result pane

    private var comparePane: some View {
        VStack(spacing: 0) {
            // Prompt bar at the bottom (above the result columns).
            // Column area fills available space; prompt bar is fixed-height at bottom.

            // Column area
            if store.evalComparePairs.isEmpty && !compareIsRunning {
                compareEmptyState
            } else {
                compareColumnArea
            }

            // Prompt bar (always visible when COMPARE tab is active)
            comparePromptBar
        }
    }

    private var compareEmptyState: some View {
        VStack {
            Spacer().frame(minHeight: 0).frame(maxHeight: .infinity).layoutPriority(-1)
            VStack(spacing: Theme.Space.md) {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.system(size: 28))
                    .foregroundStyle(Theme.Palette.inkDim)
                Text("N-WAY COMPARE")
                    .instrumentLabel()
                Text("Configure columns in the left panel, type a prompt, and press Run Compare.")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)
            }
            .frame(maxWidth: .infinity)
            Spacer().frame(minHeight: 0).frame(maxHeight: .infinity)
        }
    }

    // N-column horizontal split showing all accumulated pairs.
    private var compareColumnArea: some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(Array(compareSlots.indices), id: \.self) { colIdx in
                if colIdx > 0 {
                    Theme.Palette.hairline.frame(width: 1).frame(maxHeight: .infinity)
                }
                compareResultColumn(colIdx: colIdx)
            }
        }
    }

    // One result column: header + scrollable pair list.
    @ViewBuilder
    private func compareResultColumn(colIdx: Int) -> some View {
        let slot = colIdx < compareSlots.count ? compareSlots[colIdx] : nil
        let headerLabel: String = slot.map { columnLabel(for: $0) } ?? "Column \(colIdx + 1)"

        VStack(alignment: .leading, spacing: 0) {
            // Column header bar
            VStack(alignment: .leading, spacing: 2) {
                Text(headerLabel)
                    .font(Theme.Fonts.mono(11))
                    .foregroundStyle(Theme.Palette.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .padding(.horizontal, Theme.Space.sm)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)

            Theme.Palette.hairline.frame(height: 1)

            // Scrollable pair list
            ScrollView(.vertical) {
                LazyVStack(alignment: .leading, spacing: Theme.Space.lg) {
                    ForEach(store.evalComparePairs) { pair in
                        if colIdx < pair.columns.count {
                            compareResultCell(pair: pair, colIdx: colIdx)
                                .id(pair.id)
                        }
                    }
                }
                .padding(Theme.Space.sm)
            }
        }
        .frame(maxWidth: .infinity)
    }

    // One cell: prompt caption + streamed response + tok/s.
    @ViewBuilder
    private func compareResultCell(pair: EvalComparePair, colIdx: Int) -> some View {
        let col = pair.columns[colIdx]

        // Is THIS specific column currently streaming?
        let isStreaming: Bool = {
            guard let pid = store.evalCompareAwaitingPairID,
                  pid == pair.id,
                  store.evalComparePhase == colIdx
            else { return false }
            return !col.done
        }()

        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            // Prompt caption (shared across all columns in this pair)
            Text(pair.prompt)
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.inkDim)
                .lineLimit(2)

            // Response area
            if isStreaming && col.text.isEmpty {
                // Typing indicator — waiting for first token
                HStack(spacing: Theme.Space.xs) {
                    ProgressView().controlSize(.mini)
                    Text("Generating\u{2026}")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(compareResponseBackground)
            } else if !col.text.isEmpty || col.done {
                if col.failed && col.text.isEmpty {
                    // Honest failure display — engine error or launch failure
                    let reason = col.failureReason ?? "Run failed."
                    Text(reason)
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.crimson)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(compareResponseBackground)
                } else {
                    Text(col.text.isEmpty ? "(no output)" : col.text)
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Palette.ink)
                        .multilineTextAlignment(.leading)
                        .textSelection(.enabled)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(compareResponseBackground)

                    // tok/s caption
                    if let tps = col.tokS {
                        Text(String(format: "%.1f tok/s", tps))
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .monospacedDigit()
                    }

                    // Δ tok/s chip between col 0 (reference) and this col — only when col > 0
                    // and both this col and col 0 of the same pair are done with a tok/s reading.
                    if colIdx > 0, col.done, pair.columns[0].done,
                       let refTokS = pair.columns[0].tokS,
                       let thisTokS = col.tokS {
                        let delta = thisTokS - refTokS
                        let positive = delta >= 0
                        let deltaLabel = positive
                            ? String(format: "+%.1f tok/s", delta)
                            : String(format: "%.1f tok/s", delta)
                        Text(deltaLabel)
                            .font(Theme.Fonts.caption)
                            .monospacedDigit()
                            .foregroundStyle(positive ? Theme.Palette.signal : Theme.Palette.crimson)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                RoundedRectangle(cornerRadius: 4, style: .continuous)
                                    .fill(positive
                                          ? Theme.Palette.signal.opacity(0.12)
                                          : Theme.Palette.crimson.opacity(0.12))
                            )
                    }
                }
            } else {
                // Column exists in the pair but hasn't started yet (waiting for prior columns)
                Text("Waiting\u{2026}")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(compareResponseBackground)
            }
        }
    }

    // Shared background for response bubbles — mirrors ChatScreen assistantBubbleBackground.
    private var compareResponseBackground: some View {
        RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
            .fill(Theme.Palette.wellSink)
    }

    // Prompt bar at the bottom of the compare pane: text field + Run Compare button.
    // The Run Compare button also appears in the left rail. Both are wired to sendCompare().
    private var comparePromptBar: some View {
        VStack(spacing: 0) {
            Theme.Palette.hairline.frame(height: 1)
            HStack(spacing: Theme.Space.sm) {
                ZStack(alignment: .topLeading) {
                    if comparePromptText.isEmpty {
                        Text("Prompt for all columns\u{2026}")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .padding(.horizontal, 10)
                            .padding(.top, 9)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: $comparePromptText)
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Palette.ink)
                        .scrollContentBackground(.hidden)
                        .background(.clear)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 4)
                }
                .frame(minHeight: 44, maxHeight: 120)
                .background(Theme.Palette.surfaceRaised)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                        .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
                )
                .disabled(compareIsRunning)

                // Send button
                Button {
                    sendCompare()
                } label: {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(compareCanRun ? Theme.Palette.onAccent : Theme.Palette.inkDim)
                        .frame(width: 30, height: 30)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .fill(compareCanRun ? Theme.Palette.signal : Theme.Palette.wellSink)
                        )
                }
                .buttonStyle(.plain)
                .disabled(!compareCanRun)
                .keyboardShortcut(.return, modifiers: .command)
                .help("Run compare  \u{2318}\u{23CE}")
            }
            .padding(Theme.Space.md)
            .background(Theme.Palette.canvas)
        }
    }

    // MARK: - Compare launch action
    //
    // Builds one EvalComparePair from the current slot configuration and prompt, then
    // runs each column sequentially (column[0].onComplete → launch column[1] → …).
    //
    // Sequential ordering is a deliberate design choice matching the prior A/B sendCompare():
    //   - The single AppStore.liveRun slot means only one subprocess can be active at a time.
    //   - Sequential ordering ensures clean run-to-run isolation and accurate tok/s readings.
    //   - Parallel runs would interfere on shared Metal memory; this is NOT a limitation
    //     but an accurate reflection of the hardware's single-process GPU constraint.
    //
    // GenConfig.useGPU mapping (REAL, both binaries exist):
    //   false → generate_lora  binary: CPU BF16, supports --lora adapter
    //   true  → chat_metal     binary: GPU Metal, no --lora support
    //
    // LaunchSpec examples (printed in the log as "$ binary args…"):
    //   CPU, no adapter:  $ generate_lora --model-dir /path/qwen3.5-0.8b --prompt "…" --max-tokens 256 --temperature 0.7 --json
    //   CPU, adapter:     $ generate_lora --model-dir /path/qwen3.5-0.8b --lora /path/adapter.safetensors --prompt "…" --max-tokens 256 --temperature 0.7 --json
    //   GPU Metal:        $ chat_metal --model-dir /path/qwen3.5-0.8b --prompt "…" --max-tokens 256 --temperature 0.7 --json
    private func sendCompare() {
        let rawPrompt = comparePromptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawPrompt.isEmpty, !compareIsRunning else { return }

        // Build only the slots that have a model selected.
        // Slots without a model are silently skipped (honest: they appear as "—" columns
        // in the pair if included, but we exclude them to keep the result clean).
        let activeSlots = compareSlots.filter { !$0.modelName.isEmpty }
        guard !activeSlots.isEmpty else { return }

        // Parse generation settings — safe fallbacks match binary defaults.
        let temperature = Double(store.evalTempText) ?? 0.7
        let maxTokens = Int(store.evalMaxTokensText) ?? 256
        let seed: UInt64? = store.evalSeedText.isEmpty ? nil : UInt64(store.evalSeedText)

        // Snapshot labels now — prevents relabeling if pickers change during generation.
        let columns = activeSlots.map { slot in
            EvalColumn(label: columnLabel(for: slot))
        }

        let pair = EvalComparePair(prompt: rawPrompt, columns: columns)
        let pairID = pair.id
        store.evalComparePairs.append(pair)
        // Clear the prompt field immediately (matches ChatScreen.send() UX).
        comparePromptText = ""

        // Start the sequential chain: column[0] → onComplete → column[1] → …
        store.evalComparePhase = 0
        store.evalCompareAwaitingPairID = pairID

        func launchColumn(_ colIdx: Int) {
            guard store.evalComparePairs.contains(where: { $0.id == pairID }) else {
                store.evalComparePhase = -1
                store.evalCompareAwaitingPairID = nil
                return
            }
            guard colIdx < activeSlots.count else {
                // All columns done — reset phase tracker.
                store.evalComparePhase = -1
                store.evalCompareAwaitingPairID = nil
                return
            }

            let slot = activeSlots[colIdx]
            let model = resolvedModel(for: slot)
            let adapter = resolvedAdapter(for: slot)
            // Adapter presence forces CPU regardless of slot.useGPU (binary contract).
            let useGPU = adapter == nil && slot.useGPU

            // Build GenConfig — mirrors ChatScreen.send() config construction.
            let cfg = GenConfig(
                modelDir: model?.path,
                model: model == nil ? (slot.modelName.isEmpty ? nil : slot.modelName) : nil,
                adapterPath: adapter?.weightFile,
                prompt: rawPrompt,
                maxTokens: maxTokens,
                seed: seed,
                temperature: temperature,
                useGPU: useGPU
            )

            store.evalComparePhase = colIdx
            let run = store.runGenerate(cfg)

            run.onComplete = { [pairID] finished in
                guard let pairIdx2 = store.evalComparePairs.firstIndex(where: { $0.id == pairID }),
                      colIdx < store.evalComparePairs[pairIdx2].columns.count
                else {
                    store.evalComparePhase = -1
                    store.evalCompareAwaitingPairID = nil
                    return
                }

                // Capture the final text and tok/s from the completed run.
                store.evalComparePairs[pairIdx2].columns[colIdx].text =
                    finished.genText.trimmingCharacters(in: .whitespacesAndNewlines)
                store.evalComparePairs[pairIdx2].columns[colIdx].tokS = finished.genTokS
                store.evalComparePairs[pairIdx2].columns[colIdx].done = true

                // Honest failure: mark failed only when the run failed AND produced no text.
                // Partial text on a user-stop is kept as-is (not marked failed).
                if finished.status == .failed && finished.genText.isEmpty {
                    store.evalComparePairs[pairIdx2].columns[colIdx].failed = true
                    store.evalComparePairs[pairIdx2].columns[colIdx].failureReason =
                        finished.failureReason ?? "Run failed."
                }

                // Sequence the next column (or finish if this was the last).
                launchColumn(colIdx + 1)
            }

            // Handle synchronous launch failure (e.g. binary not found).
            if run.status == .failed {
                if let pairIdx2 = store.evalComparePairs.firstIndex(where: { $0.id == pairID }),
                   colIdx < store.evalComparePairs[pairIdx2].columns.count {
                    store.evalComparePairs[pairIdx2].columns[colIdx].done = true
                    store.evalComparePairs[pairIdx2].columns[colIdx].failed = true
                    store.evalComparePairs[pairIdx2].columns[colIdx].failureReason =
                        run.failureReason ?? "Launch failed."
                }
                // Still continue to the next column — a single launch failure should not
                // block subsequent columns from running.
                launchColumn(colIdx + 1)
            }
        }

        launchColumn(0)
    }

    // MARK: - Similar tab — left rail section

    // Embed model picker for the SIMILAR tab.
    // Filtering to embedding-format models is enforced here: the picker only offers
    // embedModels (isEmbedding == true). When none are discovered, a static known-good
    // fallback list is shown with an honest note that they download on first use.
    //
    // Honest-nil: when no embedding model is available (empty picker + no static fallback
    // names apply), the section shows a guidance note instead of a dead control.
    private var similarEmbedModelRailSection: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("EMBED MODEL")
                .instrumentLabel()
                .padding(.bottom, 2)

            Theme.Palette.hairline.frame(height: 1)

            if similarPickerOptions.isEmpty {
                // Honest-nil: no embedding model present and no static fallback.
                Text("No embedding model found. Import or quantize an embedding model to use this tab.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Picker("", selection: $similarSelectedModelName) {
                    ForEach(similarPickerOptions, id: \.self) { name in
                        Text(name).tag(name)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .font(Theme.Fonts.caption)
                .disabled(similarIsEmbedding)

                if similarUsingStaticFallback {
                    Text("models download on first use if not cached locally")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
            }
        }
    }

    // MARK: - Similar tab — result pane

    // The SIMILAR pane renders three states:
    //   1. No embedding model present → honest-nil guidance.
    //   2. Ready or in-progress → text editor + (optional) result after a run.
    //   3. Error → GatePill(.fail) with the engine's error message verbatim.
    //
    // The text editor (similarTextEditor) is always shown so the user can type before running.
    // The cosine result panel appears only AFTER a successful run (embed_done received).
    // Each cosine score shown traces to a real embed_done vector — no synthesized numbers.
    private var similarPane: some View {
        Group {
            if similarPickerOptions.isEmpty {
                similarNoModelState
            } else {
                similarContentArea
            }
        }
    }

    // Honest-nil: no embedding model available at all.
    private var similarNoModelState: some View {
        VStack(spacing: Theme.Space.md) {
            Spacer()
            VStack(spacing: Theme.Space.sm) {
                Text("NO EMBEDDING MODEL")
                    .instrumentLabel()
                Text("Import or quantize an embedding model to test semantic similarity.")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 400)
            }
            .frame(maxWidth: .infinity)
            Spacer()
        }
    }

    // Main SIMILAR tab layout: purpose header + HSplitView (text editor left, results right).
    private var similarContentArea: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Self-explaining header — answers Ocean's "what is it?" question
            // (Issue #11: "the embedding page is useless, like what is it?").
            VStack(alignment: .leading, spacing: Theme.Space.xs) {
                Text("SEMANTIC SIMILARITY")
                    .instrumentLabel()
                Text("Does this embedding model place related texts close and unrelated texts far?")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                Text("Related sentences should score above 0.8. Unrelated sentences should score below 0.4.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.top, Theme.Space.md)
            .padding(.bottom, Theme.Space.sm)

            Theme.Palette.hairline.frame(height: 1)

            HSplitView {
                // Left: editable text list
                similarTextEditorPanel
                    .frame(minWidth: 260, idealWidth: 300, maxWidth: 360)

                // Right: result area (in-progress / error / cosine matrix)
                similarResultPanel
                    .frame(minWidth: 360)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // Left sub-panel: editable text list + add/remove controls.
    // Verbatim from EmbeddingsScreen.configPanel (PI_AEP: reuse, don't reinvent).
    private var similarTextEditorPanel: some View {
        OpaquePanel {
            VStack(alignment: .leading, spacing: 0) {
                Text("TEXTS")
                    .instrumentLabel()
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.md)
                    .padding(.bottom, Theme.Space.xs)

                VStack(spacing: 0) {
                    ForEach(similarTexts.indices, id: \.self) { i in
                        similarTextRow(index: i)
                    }
                }

                // Add text button — visible only when fewer than 8 rows are present
                // (no hard cap in the spec; 8 keeps the matrix legible).
                if similarTexts.count < 8 {
                    Button {
                        similarTexts.append("")
                    } label: {
                        HStack(spacing: Theme.Space.xs) {
                            Image(systemName: "plus.circle")
                                .font(.system(size: 12, weight: .medium))
                            Text("Add text")
                                .font(Theme.Fonts.cell)
                        }
                        .foregroundStyle(Theme.Palette.inkDim)
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.vertical, Theme.Space.sm)
                    .disabled(similarIsEmbedding)
                }

                Spacer()
            }
        }
    }

    // One editable text row with an optional remove button.
    // Verbatim from EmbeddingsScreen.textRow (PI_AEP: reuse).
    private func similarTextRow(index i: Int) -> some View {
        HStack(alignment: .top, spacing: Theme.Space.xs) {
            Text("T\(i + 1)")
                .font(Theme.Fonts.mono(11))
                .foregroundStyle(Theme.Palette.inkDim)
                .frame(width: 20, alignment: .leading)
                .padding(.top, 6)

            TextField("Text \(i + 1)", text: Binding(
                get: { i < similarTexts.count ? similarTexts[i] : "" },
                set: { if i < similarTexts.count { similarTexts[i] = $0 } }
            ), axis: .vertical)
            .font(Theme.Fonts.body)
            .foregroundStyle(Theme.Palette.ink)
            .lineLimit(2...4)
            .textFieldStyle(.plain)
            .padding(.vertical, Theme.Space.xs)
            .disabled(similarIsEmbedding)

            // Remove button — only when more than 2 texts exist (minimum required is 2).
            if similarTexts.count > 2 {
                Button {
                    if i < similarTexts.count { similarTexts.remove(at: i) }
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Theme.Palette.inkDim)
                }
                .buttonStyle(.plain)
                .padding(.top, 6)
                .disabled(similarIsEmbedding)
            }
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.xs)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    // Right sub-panel: shows in-progress / error / cosine result.
    @ViewBuilder
    private var similarResultPanel: some View {
        if similarIsEmbedding {
            // In-progress state
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                HStack(spacing: Theme.Space.sm) {
                    ProgressView().controlSize(.small)
                    GatePill(.run, label: "EMBEDDING\u{2026}")
                }
                Text("First run of an uncached model downloads ~130 MB.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
            }
            .padding(Theme.Space.xl)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

        } else if let errMsg = similarEmbedError {
            // Honest error display — engine failure reason verbatim (or a default if absent).
            VStack(alignment: .leading, spacing: Theme.Space.sm) {
                GatePill(.fail, label: errMsg)
            }
            .padding(Theme.Space.xl)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

        } else if let r = similarEmbedResult {
            // Cosine result — only when embed_done was received (every score is real).
            similarResultContent(r)

        } else {
            // Pre-run empty state — honest-nil, not a dead surface.
            VStack(alignment: .leading, spacing: Theme.Space.sm) {
                Text("NO RESULT YET")
                    .instrumentLabel()
                Text("Enter at least two texts and press \u{201c}Embed Texts\u{201d} in the left rail.")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(Theme.Space.xl)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }

    // Full result view — cosine matrix + optional preview vectors.
    // Every cosine shown here traces directly to r.cosine from a real embed_done event.
    // Verbatim structure from EmbeddingsScreen.resultContent (PI_AEP: reuse).
    private func similarResultContent(_ r: LatticeEvent.EmbedDone) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.xl) {

                // Model / dims / count / ms info wells
                OpaquePanel {
                    LazyVGrid(
                        columns: [GridItem(.flexible()), GridItem(.flexible())],
                        spacing: Theme.Space.sm
                    ) {
                        ReadoutWell(label: "MODEL",  value: r.model ?? similarSelectedModelName)
                        ReadoutWell(label: "DIMS",   value: "\(r.dims)")
                        ReadoutWell(label: "TEXTS",  value: "\(r.count)")
                        ReadoutWell(label: "MS",
                                    value: r.ms.map { String(format: "%.0f", $0) } ?? "\u{2014}",
                                    unit:  r.ms != nil ? "ms" : "")
                    }
                    .padding(Theme.Space.lg)
                }

                // Cosine similarity matrix — ranked interpretation label included.
                if let cosine = r.cosine,
                   cosine.count == r.count,
                   cosine.allSatisfy({ $0.count == r.count }) {
                    OpaquePanel {
                        VStack(alignment: .leading, spacing: Theme.Space.sm) {
                            Text("COSINE SIMILARITY")
                                .instrumentLabel()

                            similarCosineMatrix(cosine: cosine, count: r.count)

                            // Ranked interpretation: helps the user see whether the model
                            // understands meaning (Ocean Issue #11 — make purpose clear).
                            Text("1.000 = identical text \u{00B7} higher = more semantically similar")
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.inkDim)
                        }
                        .padding(Theme.Space.lg)
                    }
                } else if r.cosine == nil {
                    // Honest-nil: binary ran but cosine was absent (large batch suppression).
                    OpaquePanel {
                        Text("Cosine matrix not returned for this batch size.")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .padding(Theme.Space.lg)
                    }
                }

                // Preview vectors (first 8 dims of each embedding).
                // Only shown when the binary returned them; absent = section hidden (honest-nil).
                if let preview = r.preview, !preview.isEmpty {
                    OpaquePanel {
                        VStack(alignment: .leading, spacing: Theme.Space.xs) {
                            Text("EMBEDDING PREVIEW (first 8 dims)")
                                .instrumentLabel()

                            ForEach(preview.indices, id: \.self) { i in
                                let label = "T\(i + 1)"
                                let vals = preview[i].prefix(8)
                                    .map { String(format: "%.3f", $0) }
                                    .joined(separator: ",  ")
                                HStack(alignment: .top, spacing: Theme.Space.xs) {
                                    Text(label)
                                        .font(Theme.Fonts.mono(11))
                                        .foregroundStyle(Theme.Palette.inkDim)
                                        .frame(width: 24, alignment: .leading)
                                    Text("[\(vals)]")
                                        .font(Theme.Fonts.mono(11))
                                        .foregroundStyle(Theme.Palette.ink)
                                        .monospacedDigit()
                                }
                                .help(i < similarResultTexts.count ? similarResultTexts[i] : "")
                            }
                        }
                        .padding(Theme.Space.lg)
                    }
                }
            }
            .padding(Theme.Space.xl)
        }
    }

    // N×N cosine similarity grid — color-graded cells, row/column labels.
    // Verbatim from EmbeddingsScreen.cosineMatrix (PI_AEP: reuse).
    // All values come from r.cosine delivered by a real embed_done event.
    private func similarCosineMatrix(cosine: [[Double]], count: Int) -> some View {
        let cellSize: CGFloat = 60

        return VStack(alignment: .leading, spacing: 2) {
            // Column header row
            HStack(spacing: 2) {
                Color.clear.frame(width: cellSize, height: 20)
                ForEach(0..<count, id: \.self) { j in
                    Text("T\(j + 1)")
                        .font(Theme.Fonts.mono(11))
                        .foregroundStyle(Theme.Palette.inkDim)
                        .frame(width: cellSize, height: 20)
                        .multilineTextAlignment(.center)
                        .help(j < similarResultTexts.count ? similarResultTexts[j] : "")
                }
            }

            // Data rows
            ForEach(0..<count, id: \.self) { i in
                HStack(spacing: 2) {
                    Text("T\(i + 1)")
                        .font(Theme.Fonts.mono(11))
                        .foregroundStyle(Theme.Palette.inkDim)
                        .frame(width: cellSize, height: cellSize)
                        .multilineTextAlignment(.center)
                        .help(i < similarResultTexts.count ? similarResultTexts[i] : "")

                    ForEach(0..<count, id: \.self) { j in
                        let value = i < cosine.count && j < cosine[i].count
                            ? cosine[i][j] : 0.0
                        let isDiag = (i == j)
                        let clamped = max(0.0, min(1.0, value))

                        ZStack {
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .fill(isDiag
                                      ? Theme.Palette.signal.opacity(0.60)
                                      : Theme.Palette.signal.opacity(clamped * 0.50))

                            Text(String(format: "%.3f", value))
                                .font(.system(size: 11,
                                              weight: isDiag ? .semibold : .regular,
                                              design: .monospaced))
                                .monospacedDigit()
                                .foregroundStyle(isDiag ? Theme.Palette.canvas : Theme.Palette.ink)
                        }
                        .frame(width: cellSize, height: cellSize)
                    }
                }
            }
        }
    }

    // MARK: - Similar helpers

    // Apply the default embed model selection when the picker options are available
    // and no selection has been made (or the current selection is no longer valid).
    // Verbatim from EmbeddingsScreen.applyDefaults() (PI_AEP: reuse).
    private func applySimilarDefaults() {
        guard similarSelectedModelName.isEmpty || !similarPickerOptions.contains(similarSelectedModelName) else { return }
        if similarPickerOptions.contains("bge-small-en-v1.5") {
            similarSelectedModelName = "bge-small-en-v1.5"
        } else {
            similarSelectedModelName = similarPickerOptions.first ?? ""
        }
    }

    // MARK: - Similar embed action

    // Launches an embed run for the current text list.
    // All state updates happen on the main actor (this is a @MainActor view).
    // The cosine result is stored in similarEmbedResult ONLY when embed_done is received —
    // never synthesized.  A failed run sets similarEmbedError to the engine's message.
    private func runSimilar() {
        let payload = similarTexts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard payload.count >= 2, !similarIsEmbedding else { return }

        similarEmbedError = nil
        similarEmbedResult = nil
        similarIsEmbedding = true
        similarResultTexts = payload

        let config = EmbedConfig(model: similarSelectedModelName, texts: payload)
        let run = store.runEmbed(config)

        run.onComplete = { finished in
            if finished.status == .failed {
                self.similarEmbedError = finished.failureReason ?? "Embedding failed. Check the log."
            } else if finished.embed == nil {
                self.similarEmbedError = "No embedding result returned."
            }
            self.similarEmbedResult = finished.embed
            self.similarIsEmbedding = false
        }

        // Handle synchronous launch failure (binary not found, invalid spec).
        if run.status == .failed {
            similarEmbedError = run.failureReason ?? "Embed failed to launch."
            similarIsEmbedding = false
        }
    }

    // MARK: - Private helpers

    // WellSpec — local mirror of the private struct in ModelsScreen (PI_AEP: each screen
    // owns its own private helpers; extracting to shared would require moving PPLWellSpec
    // to a module-level file which is out of Stage 2 scope).
    private struct PPLWellSpec {
        let label: String
        let value: String
        let unit: String
        let subtitle: String
        init(_ label: String, _ value: String, unit: String = "", subtitle: String = "") {
            self.label = label; self.value = value; self.unit = unit; self.subtitle = subtitle
        }
    }

    // Default corpus used when no custom file is selected.
    // Verbatim from the prior ModelsScreen.defaultCorpus.
    private static let defaultCorpus = """
    The transformer architecture has become the dominant approach for natural language processing tasks. \
    Attention mechanisms allow models to weigh the importance of different tokens in a sequence. \
    Large language models are trained on vast corpora of text drawn from books, articles, and web pages. \
    Quantization reduces the memory footprint of these models by representing weights with fewer bits. \
    A four-bit quantized model can run on consumer hardware that would otherwise be unable to hold the full-precision weights. \
    Perplexity is a standard metric for evaluating how well a language model predicts a held-out corpus. \
    Lower perplexity indicates that the model assigns higher probability to the observed tokens. \
    Rotation-based methods such as QuaRot redistribute the magnitude of activations before quantization to reduce error. \
    The goal is to preserve as much of the original model quality as possible while shrinking the storage and compute requirements. \
    Inference engines must balance throughput, latency, and memory to serve these models efficiently on edge devices and in the cloud.
    """

    // Returns the corpus URL: custom if set, else writes the default to a temp file.
    // Verbatim from the prior ModelsScreen.corpusURL(), adapted to use pplCorpusURL.
    private func corpusURL() -> URL? {
        if let u = pplCorpusURL { return u }
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("lattice_eval_corpus.txt")
        do {
            try Self.defaultCorpus.write(to: tmp, atomically: true, encoding: .utf8)
            return tmp
        } catch {
            return nil
        }
    }

    // Try to find the BF16 sibling of a quantized model.
    // Matching rule: strip "-q4", "-quarot", or "-quarot-q4" suffix (case-insensitive),
    // then look for a BF16 model whose name equals the stripped base.
    // Fallback: BF16 model whose name is the quant name's prefix at a "-" boundary.
    // Verbatim from the prior ModelsScreen.pplSibling(for:).
    private func pplSibling(for quantModel: ModelInfo) -> ModelInfo? {
        guard quantModel.format.isQuantized else { return nil }

        let quantName = quantModel.name.lowercased()
        let suffixes = ["-quarot-q4", "-quarot", "-q4"]
        var baseName: String? = nil
        for suffix in suffixes {
            if quantName.hasSuffix(suffix) {
                baseName = String(quantName.dropLast(suffix.count))
                break
            }
        }

        let bf16 = store.models.filter { $0.format == .bf16 }
        // Primary: exact match after suffix removal.
        if let base = baseName,
           let exact = bf16.first(where: { $0.name.lowercased() == base }) {
            return exact
        }
        // Fallback: longest BF16 model whose name is a prefix of the quant name at a "-" boundary.
        return bf16
            .filter { quantName.hasPrefix($0.name.lowercased() + "-") }
            .max(by: { $0.name.count < $1.name.count })
    }

    // Launch the perplexity measurement for the given model.
    //
    // BF16 model:             single pass, CPU path only (--model-dir).
    // Quantized with sibling: sequential bf16 (CPU) → quant (GPU Metal) chain via onComplete.
    // Quantized, no sibling:  quant-only pass using the model's own tokenizer.
    //
    // Hardware note: this method NEVER passes a GPU flag to BF16 eval; there is no such path.
    // BF16 PPL is always CPU. Q4/QuaRot PPL is always GPU Metal. The labels are set accordingly.
    //
    // Migrated verbatim from ModelsScreen.measurePerplexity(model:) and adapted for EvalScreen.
    private func measurePerplexity(model: ModelInfo) {
        guard let corpus = corpusURL() else {
            pplError = "Could not write corpus file to temp directory."
            return
        }

        pplError = nil
        pplBase = nil
        pplQuant = nil
        pplMeasuredModelID = model.id

        if model.format == .bf16 {
            // BF16 baseline-only path — CPU, --model-dir
            pplPhase = .base
            let cfg = EvalConfig(modelDir: model.path, corpusFile: corpus, label: "bf16")
            let run = store.runEval(cfg)

            run.onComplete = { [modelID = model.id] finished in
                guard self.pplMeasuredModelID == modelID else { return }
                if finished.status == .failed || finished.perplexities.first == nil {
                    self.pplError = finished.status == .failed
                        ? "BF16 eval failed. Check the log."
                        : "No perplexity result returned."
                }
                self.pplBase = finished.perplexities.first
                self.pplPhase = .idle
            }

            if run.status == .failed {
                pplError = "BF16 eval failed to launch."
                pplPhase = .idle
                pplMeasuredModelID = nil
            }
            return
        }

        // Quantized model path
        if let sibling = pplSibling(for: model) {
            // Sequential chain: bf16 sibling (CPU) → then quant (GPU Metal)
            pplPhase = .base
            let baseCfg = EvalConfig(modelDir: sibling.path, corpusFile: corpus, label: "bf16")
            let baseRun = store.runEval(baseCfg)

            baseRun.onComplete = { [modelID = model.id, modelPath = model.path,
                                    siblingPath = sibling.path, modelFormat = model.format] finished in
                guard self.pplMeasuredModelID == modelID else { return }

                if finished.status == .failed || finished.perplexities.first == nil {
                    self.pplError = finished.status == .failed
                        ? "BF16 eval failed. Check the log."
                        : "No perplexity result returned from BF16 eval."
                    self.pplPhase = .idle
                    return
                }
                self.pplBase = finished.perplexities.first

                // Phase 2: quant eval — GPU Metal path (--q4-dir or --quarot-q4-dir)
                self.pplPhase = .quant

                guard let corpus2 = self.corpusURL() else {
                    self.pplPhase = .idle
                    return
                }

                let quantCfg: EvalConfig
                if modelFormat == .quarot {
                    // QuaRot: --quarot-q4-dir + --tokenizer-dir (GPU Metal)
                    quantCfg = EvalConfig(quarotDir: modelPath, tokenizerDir: siblingPath,
                                          corpusFile: corpus2, label: "quarot")
                } else {
                    // Q4: --q4-dir + --tokenizer-dir (GPU Metal)
                    quantCfg = EvalConfig(q4Dir: modelPath, tokenizerDir: siblingPath,
                                          corpusFile: corpus2, label: "q4")
                }
                let quantRun = self.store.runEval(quantCfg)

                quantRun.onComplete = { [modelID] done2 in
                    guard self.pplMeasuredModelID == modelID else { return }
                    if done2.status == .failed || done2.perplexities.first == nil {
                        self.pplError = done2.status == .failed
                            ? "Quant eval failed. Check the log."
                            : "No perplexity result returned from quant eval."
                    }
                    self.pplQuant = done2.perplexities.first
                    self.pplPhase = .idle
                }

                if quantRun.status == .failed {
                    self.pplError = "Quant eval failed to launch."
                    self.pplPhase = .idle
                }
            }

            if baseRun.status == .failed {
                pplError = "BF16 eval failed to launch."
                pplPhase = .idle
                pplMeasuredModelID = nil
            }

        } else if model.hasTokenizer {
            // No sibling but has its own tokenizer: quant-only, GPU Metal path
            pplPhase = .quant
            let quantCfg: EvalConfig
            if model.format == .quarot {
                // QuaRot: --quarot-q4-dir + --tokenizer-dir pointing at itself
                quantCfg = EvalConfig(quarotDir: model.path, tokenizerDir: model.path,
                                      corpusFile: corpus, label: "quarot")
            } else {
                // Q4: --q4-dir + --tokenizer-dir pointing at itself
                quantCfg = EvalConfig(q4Dir: model.path, tokenizerDir: model.path,
                                      corpusFile: corpus, label: "q4")
            }
            let run = store.runEval(quantCfg)

            run.onComplete = { [modelID = model.id] finished in
                guard self.pplMeasuredModelID == modelID else { return }
                if finished.status == .failed || finished.perplexities.first == nil {
                    self.pplError = finished.status == .failed
                        ? "Quant eval failed. Check the log."
                        : "No perplexity result returned."
                }
                self.pplQuant = finished.perplexities.first
                self.pplPhase = .idle
            }

            if run.status == .failed {
                pplError = "Quant eval failed to launch."
                pplPhase = .idle
                pplMeasuredModelID = nil
            }
        }
        // else: canMeasure = false, button was disabled — this branch is unreachable when
        // the UI guard is working, but the guard condition above (canRun) enforces it.
    }
}
