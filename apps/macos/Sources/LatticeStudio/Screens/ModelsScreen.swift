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

struct ModelsScreen: View {
    @Bindable var store: AppStore

    // Tab selection — Models or Adapters
    @State private var modelsTab: ModelsTab = .models
    // Local selection state — deselects when models list changes.
    @State private var selectedModelID: String?
    @State private var selectedAdapterID: String?
    // Quantize sheet: presented from the model inspector/action row (Phase A re-parenting).
    @State private var showQuantizeSheet: Bool = false
    @State private var showGetModelsSheet: Bool = false
    @State private var didInitInspector = false

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
        .sheet(isPresented: $showGetModelsSheet) {
            GetModelsSheet(store: store)
                .frame(minWidth: 680, idealWidth: 760, maxWidth: .infinity,
                       minHeight: 500, idealHeight: 620, maxHeight: .infinity)
        }
        .onAppear { openInspectorOnce() }
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
            // Get Models — primary (the discovery CTA; no model selection required)
            Button {
                showGetModelsSheet = true
            } label: {
                Text("Get Models")
                    .font(Theme.Fonts.body)
            }
            .buttonStyle(LatticePrimaryButtonStyle())
            .help("Browse and download models or import from disk")

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
                    .help("Send selected model to Chat (⌘4)")

                    Button {
                        store.use(model, on: .eval)
                    } label: {
                        Text("Eval →")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                    .help("Send selected model to Eval workspace (⌘5)")
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

