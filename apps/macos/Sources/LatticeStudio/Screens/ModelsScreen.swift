import SwiftUI
import AppKit

// MARK: - 01 MODELS
//
// Two-tab layout via [Models | Adapters] segmented Picker at the top:
//
//   MODELS tab  (unchanged):
//     • Trailing action row: Refresh + Reveal-in-Finder + Train/Quantize/Chat CTAs
//     • Main body: DataTable of store.models
//     • Right inspector: ReadoutWells for the selected model + adapter sub-list
//
//   ADAPTERS tab  (new):
//     • Main body: DataTable of store.adapters (scanned from <repoRoot>/adapters)
//     • Right inspector: ReadoutWells for the selected adapter + Delete button
//
// Empty state: if binariesReady is false hint "run `make build`" and show modelCachePath.
// One teal primary CTA per screen: "Train →" (filled); others are outline.

// MARK: - ModelsTab

private enum ModelsTab: String, CaseIterable {
    case models   = "Models"
    case adapters = "Adapters"
}

private let byteFormatter: ByteCountFormatter = {
    let f = ByteCountFormatter()
    f.allowedUnits = [.useBytes, .useKB, .useMB, .useGB]
    f.countStyle = .file
    return f
}()

// MARK: - PPL phase tracker (mirrors ABPhase in ChatScreen)

private enum PPLPhase { case idle, base, quant }

struct ModelsScreen: View {
    @Bindable var store: AppStore

    // Tab selection — Models or Adapters
    @State private var modelsTab: ModelsTab = .models
    // Local selection state — deselects when models list changes.
    @State private var selectedModelID: String?
    @State private var selectedAdapterID: String?
    // Quantize sheet: presented from the model inspector/action row (Phase A re-parenting).
    @State private var showQuantizeSheet: Bool = false
    @State private var didInitInspector = false

    // QUALITY (perplexity) section state
    @State private var pplPhase: PPLPhase = .idle
    @State private var pplBase: LatticeEvent.Perplexity?       // bf16 result
    @State private var pplQuant: LatticeEvent.Perplexity?      // q4/quarot result
    @State private var pplCorpusURL: URL?                      // nil = use embedded default
    @State private var pplError: String?
    @State private var pplMeasuredModelID: String?             // results belong to this model only

    private var selectedModel: ModelInfo? {
        store.models.first { $0.id == selectedModelID }
    }

    private var selectedAdapter: AdapterInfo? {
        store.adapters.first { $0.id == selectedAdapterID }
    }

    private var subtitle: String {
        switch modelsTab {
        case .models:
            let count = store.models.count
            if count == 0 { return "no models found" }
            return "\(count) model\(count == 1 ? "" : "s") · \(store.modelCachePath)"
        case .adapters:
            let count = store.adapters.count
            if count == 0 { return "no adapters found" }
            return "\(count) adapter\(count == 1 ? "" : "s")"
        }
    }

    var body: some View {
        ScreenScaffold(
            screen: .models,
            subtitle: subtitle,
            trailing: { actionRow }
        ) {
            VStack(spacing: 0) {
                // Segmented tab picker — mirrors ChatScreen pattern
                HStack {
                    Picker("", selection: $modelsTab) {
                        ForEach(ModelsTab.allCases, id: \.self) { tab in
                            Text(tab.rawValue).tag(tab)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 240)
                    Spacer()
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.md)
                .padding(.bottom, Theme.Space.sm)

                // Tab content
                switch modelsTab {
                case .models:
                    if store.models.isEmpty {
                        emptyState
                    } else {
                        modelTable
                    }
                case .adapters:
                    adapterTable
                }
            }
        }
        .inspector(isPresented: $store.inspectorPresented) {
            switch modelsTab {
            case .models:
                inspectorPanel
                    .inspectorColumnWidth(min: 260, ideal: 300, max: 320)
            case .adapters:
                adapterInspectorPanel
                    .inspectorColumnWidth(min: 260, ideal: 300, max: 320)
            }
        }
        .sheet(isPresented: $showQuantizeSheet) {
            QuantizeScreen(store: store)
                .frame(minWidth: 760, idealWidth: 900, maxWidth: .infinity,
                       minHeight: 540, idealHeight: 640, maxHeight: .infinity)
        }
        .onAppear { openInspectorOnce() }
        // Reset perplexity results when the selected model changes so results never
        // bleed across model selections.
        .onChange(of: selectedModelID) { _, _ in
            pplPhase = .idle
            pplBase = nil
            pplQuant = nil
            pplError = nil
            pplMeasuredModelID = nil
        }
    }

    // Open the detail inspector once on first appear (replaces the previously always-visible column).
    private func openInspectorOnce() {
        guard !didInitInspector else { return }
        didInitInspector = true
        store.inspectorPresented = true
    }

    // MARK: Action row (scaffold trailing)

    private var actionRow: some View {
        HStack(spacing: Theme.Space.sm) {
            // Refresh — secondary
            Button {
                store.refreshModels()
            } label: {
                Text("Refresh")
                    .font(Theme.Fonts.body)
            }
            .buttonStyle(LatticeSecondaryButtonStyle())
            .keyboardShortcut("r", modifiers: .command)
            .help("Refresh model list (⌘R)")

            // Tab-specific actions
            switch modelsTab {
            case .models:
                // Reveal in Finder — only when a model is selected
                if let model = selectedModel {
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([model.path])
                    } label: {
                        Text("Reveal")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                }

                Divider()
                    .frame(height: 16)

                // Navigation CTAs — Train is the single teal primary
                if let model = selectedModel {
                    Button {
                        store.use(model, on: .train)
                    } label: {
                        Text("Train")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticePrimaryButtonStyle())
                    .help("Send selected model to Train (⌘3)")

                    Button {
                        store.workingModel = model
                        showQuantizeSheet = true
                    } label: {
                        Text("Quantize…")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                    .help("Quantize selected model (Q4 / QuaRot)")

                    Button {
                        store.use(model, on: .chat)
                    } label: {
                        Text("Chat")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                    .help("Send selected model to Chat (⌘2)")
                }

            case .adapters:
                // Reveal in Finder — only when an adapter is selected
                if let adapter = selectedAdapter {
                    Button {
                        NSWorkspace.shared.activateFileViewerSelecting([adapter.path])
                    } label: {
                        Text("Reveal")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                }
            }
        }
    }

    // MARK: Master table

    private var modelTable: some View {
        ScrollView {
            DataTable(
                rows: store.models,
                columns: modelColumns,
                selectedID: $selectedModelID,
                comfortable: store.rowComfortable
            )
        }
        .instrumentPanel()
    }

    private var modelColumns: [ColumnDef<ModelInfo>] {
        [
            ColumnDef(
                id: "name",
                header: "NAME",
                alignment: .leading,
                minWidth: 180,
                isNumeric: false
            ) { $0.name },

            ColumnDef(
                id: "format",
                header: "FORMAT",
                alignment: .leading,
                minWidth: 80,
                isNumeric: false
            ) { $0.format.badge },

            ColumnDef(
                id: "params",
                header: "PARAMS",
                alignment: .trailing,
                minWidth: 64,
                isNumeric: false
            ) { $0.params ?? "—" },

            ColumnDef(
                id: "layers",
                header: "LAYERS",
                alignment: .leading,
                minWidth: 120,
                isNumeric: false
            ) { $0.layerSummary ?? "—" },

            ColumnDef(
                id: "size",
                header: "SIZE",
                alignment: .trailing,
                minWidth: 80,
                isNumeric: true
            ) { byteFormatter.string(fromByteCount: $0.sizeBytes) },

            ColumnDef(
                id: "files",
                header: "FILES",
                alignment: .trailing,
                minWidth: 48,
                isNumeric: true
            ) { "\($0.fileCount)" },

            ColumnDef(
                id: "tokenizer",
                header: "TOK",
                alignment: .center,
                minWidth: 40,
                isNumeric: false
            ) { $0.hasTokenizer ? "✓" : "—" },

            ColumnDef(
                id: "adapters",
                header: "#ADAPTERS",
                alignment: .trailing,
                minWidth: 72,
                isNumeric: true
            ) { "\($0.adapters.count)" },
        ]
    }

    // MARK: Adapter table

    private var adapterTable: some View {
        ScrollView {
            DataTable(
                rows: store.adapters,
                columns: adapterColumns,
                selectedID: $selectedAdapterID,
                comfortable: store.rowComfortable
            )
        }
        .instrumentPanel()
    }

    private var adapterColumns: [ColumnDef<AdapterInfo>] {
        [
            ColumnDef(
                id: "name",
                header: "NAME",
                alignment: .leading,
                minWidth: 180,
                isNumeric: false
            ) { $0.name },

            ColumnDef(
                id: "baseModel",
                header: "BASE MODEL",
                alignment: .leading,
                minWidth: 160,
                isNumeric: false
            ) { $0.baseModel ?? "—" },

            ColumnDef(
                id: "rank",
                header: "RANK",
                alignment: .trailing,
                minWidth: 56,
                isNumeric: true
            ) { $0.rank.map(String.init) ?? "—" },

            ColumnDef(
                id: "scale",
                header: "SCALE",
                alignment: .trailing,
                minWidth: 64,
                isNumeric: true
            ) {
                if let s = $0.scale { return String(format: "%.0f", s) }
                if let a = $0.alpha { return String(format: "%.0f", a) }
                return "—"
            },

            ColumnDef(
                id: "size",
                header: "SIZE",
                alignment: .trailing,
                minWidth: 80,
                isNumeric: true
            ) { byteFormatter.string(fromByteCount: $0.sizeBytes) },
        ]
    }

    // MARK: Adapter inspector panel

    @ViewBuilder
    private var adapterInspectorPanel: some View {
        if let adapter = selectedAdapter {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.xl) {
                    adapterInspector(adapter)
                }
                .padding(Theme.Space.lg)
            }
            .instrumentPanel()
        } else {
            VStack {
                Spacer()
                Text("SELECT AN ADAPTER")
                    .instrumentLabel()
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        }
    }

    private func adapterInspector(_ adapter: AdapterInfo) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.md) {
            // Adapter name header
            Text(adapter.name)
                .font(Theme.Fonts.title)
                .foregroundStyle(Theme.Palette.ink)
                .lineLimit(1)

            // Readout wells — 2-column grid
            let wells = adapterWells(for: adapter)
            LazyVGrid(
                columns: [GridItem(.flexible(), spacing: Theme.Space.md), GridItem(.flexible(), spacing: Theme.Space.md)],
                spacing: Theme.Space.md
            ) {
                ForEach(wells, id: \.label) { well in
                    ReadoutWell(label: well.label, value: well.value, unit: well.unit, minHeight: 56)
                }
            }

            // Delete button
            Button {
                store.deleteAdapter(adapter)
                selectedAdapterID = nil
            } label: {
                Text("Delete")
                    .font(Theme.Fonts.body)
            }
            .buttonStyle(LatticeSecondaryButtonStyle())
            .padding(.top, Theme.Space.sm)
        }
    }

    private func adapterWells(for adapter: AdapterInfo) -> [WellSpec] {
        var ws: [WellSpec] = []
        if let rank = adapter.rank {
            ws.append(WellSpec("RANK", "\(rank)"))
        }
        // Show scale (MLX) or alpha (PEFT), whichever is present
        if let scale = adapter.scale {
            ws.append(WellSpec("SCALE", String(format: "%.0f", scale)))
        } else if let alpha = adapter.alpha {
            ws.append(WellSpec("ALPHA", String(format: "%.0f", alpha)))
        }
        if let baseModel = adapter.baseModel {
            ws.append(WellSpec("BASE MODEL", baseModel))
        }
        if let numLayers = adapter.numLayers {
            ws.append(WellSpec("LAYERS", "\(numLayers)"))
        }
        if adapter.checkpointCount > 0 {
            ws.append(WellSpec("CHECKPOINTS", "\(adapter.checkpointCount)"))
        }
        ws.append(WellSpec("SIZE", byteFormatter.string(fromByteCount: adapter.sizeBytes)))
        if let modules = adapter.targetModules {
            ws.append(WellSpec("MODULES", modules))
        }
        return ws
    }

    // MARK: Inspector panel

    @ViewBuilder
    private var inspectorPanel: some View {
        if let model = selectedModel {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.xl) {
                    modelInspector(model)
                    qualitySection(model)
                    if !model.adapters.isEmpty {
                        adapterList(model.adapters)
                    }
                }
                .padding(Theme.Space.lg)
            }
            .instrumentPanel()
        } else {
            VStack {
                Spacer()
                Text("SELECT A MODEL")
                    .instrumentLabel()
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        }
    }

    private func modelInspector(_ model: ModelInfo) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.md) {
            // Model name header
            HStack(spacing: Theme.Space.sm) {
                Text(model.name)
                    .font(Theme.Fonts.title)
                    .foregroundStyle(Theme.Palette.ink)
                    .lineLimit(1)
                if model.format.isQuantized {
                    GatePill(.pass, label: model.format.badge)
                }
            }

            // Readout wells — 2-column grid
            let wells = modelWells(for: model)
            LazyVGrid(
                columns: [GridItem(.flexible(), spacing: Theme.Space.md), GridItem(.flexible(), spacing: Theme.Space.md)],
                spacing: Theme.Space.md
            ) {
                ForEach(wells, id: \.label) { well in
                    ReadoutWell(label: well.label, value: well.value, unit: well.unit, minHeight: 56)
                }
            }
        }
    }

    private struct WellSpec {
        let label: String
        let value: String
        let unit: String
        init(_ label: String, _ value: String, _ unit: String = "") {
            self.label = label; self.value = value; self.unit = unit
        }
    }

    private func modelWells(for model: ModelInfo) -> [WellSpec] {
        var ws: [WellSpec] = []
        ws.append(WellSpec("FORMAT", model.format.rawValue))
        ws.append(WellSpec("DTYPE", model.dtype))
        if let params = model.params {
            ws.append(WellSpec("PARAMS", params))
        }
        if let hidden = model.hidden {
            ws.append(WellSpec("HIDDEN", "\(hidden)"))
        }
        // intermediate_size — FFN/MLP inner width (3584 for qwen3.5, ~3.5× hidden).
        // Honest-nil: omitted when the model config has no intermediate_size.
        if let ffn = model.intermediateSize {
            ws.append(WellSpec("FFN", "\(ffn)"))
        }
        if let vocab = model.vocab {
            ws.append(WellSpec("VOCAB", "\(vocab)"))
        }
        if let ctx = model.contextLength {
            ws.append(WellSpec("CTX", "\(ctx)"))
        }
        if let h = model.attnHeads { ws.append(WellSpec("HEADS", "\(h)")) }
        if let kv = model.kvHeads { ws.append(WellSpec("KV HEADS", "\(kv)")) }
        if let hd = model.headDim { ws.append(WellSpec("HEAD DIM", "\(hd)")) }
        // GatedDeltaNet linear-attention heads; show K/V, honest-nil each, omit when both absent.
        if model.gdnKeyHeads != nil || model.gdnValueHeads != nil {
            let k = model.gdnKeyHeads.map(String.init) ?? "—"
            let v = model.gdnValueHeads.map(String.init) ?? "—"
            ws.append(WellSpec("GDN HEADS", "\(k)/\(v)"))
        }
        if let layers = model.layerSummary {
            ws.append(WellSpec("LAYERS", layers))
        }
        ws.append(WellSpec("SIZE", byteFormatter.string(fromByteCount: model.sizeBytes)))
        ws.append(WellSpec("FILES", "\(model.fileCount)"))
        ws.append(WellSpec("TOKENIZER", model.hasTokenizer ? "yes" : "—"))
        return ws
    }

    // MARK: QUALITY (perplexity) section

    // Default corpus — ~200 tokens of diverse English text for quick PPL measurement.
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

    /// Returns the corpus URL to use: custom if set, else writes the default to a temp file.
    private func corpusURL() -> URL? {
        if let u = pplCorpusURL { return u }
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("lattice_ppl_corpus.txt")
        do {
            try Self.defaultCorpus.write(to: tmp, atomically: true, encoding: .utf8)
            return tmp
        } catch {
            pplError = "Could not write corpus: \(error.localizedDescription)"
            return nil
        }
    }

    /// Try to find the bf16 sibling of a quantized model.
    /// Matching rule: strip a trailing "-q4", "-quarot", or "-quarot-q4" suffix (case-insensitive)
    /// and look for a bf16 model whose name equals the stripped base name.
    /// Fallback: bf16 model whose name is the quant name's prefix at a "-" boundary.
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
        // Primary: exact match after suffix removal — authoritative, and never lost to a
        // shorter substring. store.models is name-sorted, so a single-predicate
        // `.first { exact || contains }` would let a bare-substring bf16 model that sorts
        // earlier win the race over the true exact sibling (codex/critic B1).
        if let base = baseName,
           let exact = bf16.first(where: { $0.name.lowercased() == base }) {
            return exact
        }
        // Fallback: a bf16 model whose name is the quant's prefix at a "-" boundary
        // (e.g. "qwen3.5-0.8b" for "qwen3.5-0.8b-q4"); prefer the longest such match.
        return bf16
            .filter { quantName.hasPrefix($0.name.lowercased() + "-") }
            .max(by: { $0.name.count < $1.name.count })
    }

    /// Launch the perplexity measurement for the given model.
    /// For quantized models with a bf16 sibling: sequential bf16 → quant chain (sendCompare pattern).
    /// For quantized models without a sibling but with tokenizer: quant-only.
    /// For bf16 models: bf16-only baseline.
    private func measurePerplexity(model: ModelInfo) {
        guard let corpus = corpusURL() else { return }

        pplError = nil
        pplBase = nil
        pplQuant = nil
        pplMeasuredModelID = model.id

        if model.format == .bf16 {
            // BF16 baseline-only path
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
            // Sequential chain: bf16 sibling → then quant
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

                // Phase 2: quant eval
                self.pplPhase = .quant

                guard let corpus2 = self.corpusURL() else {
                    self.pplPhase = .idle
                    return
                }

                let quantCfg: EvalConfig
                if modelFormat == .quarot {
                    quantCfg = EvalConfig(quarotDir: modelPath, tokenizerDir: siblingPath,
                                          corpusFile: corpus2, label: "quarot")
                } else {
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
            // No sibling but has its own tokenizer: quant-only
            pplPhase = .quant
            let quantCfg: EvalConfig
            if model.format == .quarot {
                quantCfg = EvalConfig(quarotDir: model.path, tokenizerDir: model.path,
                                      corpusFile: corpus, label: "quarot")
            } else {
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
        // else: no button rendered — handled in qualitySection view
    }

    @ViewBuilder
    private func qualitySection(_ model: ModelInfo) -> some View {
        let sibling = model.format.isQuantized ? pplSibling(for: model) : nil
        let canMeasure: Bool = {
            if model.format == .bf16 { return true }
            if sibling != nil { return true }
            if model.hasTokenizer { return true }
            return false
        }()
        let methodLabel = model.format == .quarot ? "QUAROT" : "Q4"

        VStack(alignment: .leading, spacing: Theme.Space.md) {
            // Section header + hairline
            Text("QUALITY (PERPLEXITY)")
                .instrumentLabel()
                .padding(.bottom, 2)
            Theme.Palette.hairline.frame(height: 1)

            if !canMeasure {
                // Honest disabled state: no button, explain why
                Text("Perplexity needs the BF16 source (for its tokenizer). Keep the source model in the list to compare.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                // Corpus selector row
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
                        Text("Default corpus")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                    Spacer()
                    Button("Choose corpus…") {
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

                // Action button
                let buttonLabel: String = {
                    if model.format == .bf16 { return "Measure perplexity" }
                    if sibling != nil { return "Measure perplexity vs BF16" }
                    return "Measure perplexity"
                }()

                Button(buttonLabel) {
                    measurePerplexity(model: model)
                }
                .buttonStyle(LatticeSecondaryButtonStyle())
                .disabled(pplPhase != .idle)

                // In-progress indicator
                if pplPhase != .idle {
                    VStack(alignment: .leading, spacing: Theme.Space.xs) {
                        HStack(spacing: Theme.Space.sm) {
                            ProgressView()
                                .controlSize(.small)
                            GatePill(.run, label: pplPhase == .base ? "MEASURING BF16…" : "MEASURING \(methodLabel)…")
                        }
                        Text("CPU BF16 eval takes ~15s")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                }

                // Error display
                if let err = pplError {
                    GatePill(.fail, label: err)
                }

                // Results — only when they belong to the currently selected model
                if pplMeasuredModelID == selectedModelID,
                   pplBase != nil || pplQuant != nil {

                    let bf16Value = pplBase.map { String(format: "%.3f", $0.ppl) } ?? "—"
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

                    // 2-column grid of ReadoutWells
                    let showDelta = model.format != .bf16
                    let pplWells: [WellSpec] = {
                        var ws: [WellSpec] = []
                        ws.append(WellSpec("BF16 PPL", bf16Value))
                        if showDelta {
                            ws.append(WellSpec("\(methodLabel) PPL", quantValue))
                            ws.append(WellSpec("ΔPPL", deltaValue))
                        }
                        return ws
                    }()

                    LazyVGrid(
                        columns: [
                            GridItem(.flexible(), spacing: Theme.Space.md),
                            GridItem(.flexible(), spacing: Theme.Space.md)
                        ],
                        spacing: Theme.Space.md
                    ) {
                        ForEach(pplWells, id: \.label) { well in
                            ReadoutWell(label: well.label, value: well.value,
                                        unit: well.unit, minHeight: 56)
                        }
                    }

                    // Quality verdict — only when both PPLs present (honest: no verdict with one value)
                    if let base = pplBase, let quant = pplQuant {
                        let delta = quant.ppl - base.ppl
                        if delta < 0.5 {
                            GatePill(.pass, label: "MINIMAL LOSS (Δ<0.5)")
                        } else if delta < 1.5 {
                            GatePill(.warn, label: "MODERATE LOSS")
                        } else {
                            GatePill(.fail, label: "SIGNIFICANT LOSS")
                        }
                    } else if model.format == .bf16, pplBase != nil {
                        // Honest baseline label
                        GatePill(.pass, label: "BASELINE")
                    }
                }
            }
        }
    }

    private func adapterList(_ adapters: [AdapterInfo]) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            // Section label
            Text("ADAPTERS")
                .instrumentLabel()
                .padding(.bottom, 2)

            // Hairline divider
            Theme.Palette.hairline.frame(height: 1)

            ForEach(adapters) { adapter in
                adapterRow(adapter)
                Theme.Palette.hairline.frame(height: 1)
            }
        }
    }

    private func adapterRow(_ adapter: AdapterInfo) -> some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 2) {
                Text(adapter.name)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                    .lineLimit(1)

                HStack(spacing: Theme.Space.sm) {
                    if let rank = adapter.rank {
                        inlineStat("rank", "r\(rank)")
                    }
                    if let alpha = adapter.alpha {
                        inlineStat("α", String(format: "%.0f", alpha))
                    }
                    if let modules = adapter.targetModules {
                        inlineStat("modules", modules)
                    }
                }
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text(byteFormatter.string(fromByteCount: adapter.sizeBytes))
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .monospacedDigit()

                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([adapter.path])
                } label: {
                    Text("Reveal")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.signal)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.vertical, Theme.Space.xs)
    }

    private func inlineStat(_ label: String, _ value: String) -> some View {
        HStack(spacing: 2) {
            Text(label)
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
                .textCase(.uppercase)
            Text(value)
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
        }
    }

    // MARK: Empty state

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: Theme.Space.xl) {
            OpaquePanel {
                VStack(alignment: .leading, spacing: Theme.Space.lg) {
                    Text("NO MODELS FOUND")
                        .font(Theme.Fonts.title)
                        .foregroundStyle(Theme.Palette.ink)

                    VStack(alignment: .leading, spacing: Theme.Space.sm) {
                        Text("Model cache directory:")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)

                        Text(store.modelCachePath)
                            .font(Theme.Fonts.readout)
                            .foregroundStyle(Theme.Palette.ink)
                            .monospacedDigit()
                            .lineLimit(2)
                            .truncationMode(.middle)
                    }

                    if !store.binariesReady {
                        GatePill(.warn, label: "lattice binary not found")

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Build the lattice engine first:")
                                .font(Theme.Fonts.body)
                                .foregroundStyle(Theme.Palette.inkDim)
                            Text("make build")
                                .font(Theme.Fonts.readout)
                                .foregroundStyle(Theme.Palette.signal)
                                .monospacedDigit()
                        }
                    }

                    Button {
                        store.refreshModels()
                    } label: {
                        Text("Refresh")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                }
                .padding(Theme.Space.xl)
            }

            Spacer()
        }
    }
}

