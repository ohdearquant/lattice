import SwiftUI
import AppKit

// MARK: - 01 MODELS
//
// Model management screen:
//   • Trailing action row: Get Models + Refresh + Reveal-in-Finder + Chat CTA
//   • Main body: DataTable of store.models
//   • Right inspector: ReadoutWells for the selected model
//
// Empty state: if binariesReady is false hint "run `make build`" and show modelCachePath.

private let byteFormatter: ByteCountFormatter = {
    let f = ByteCountFormatter()
    f.allowedUnits = [.useBytes, .useKB, .useMB, .useGB]
    f.countStyle = .file
    return f
}()

struct ModelsScreen: View {
    @Bindable var store: AppStore

    // Local selection state — deselects when models list changes.
    @State private var selectedModelID: String?
    @State private var showGetModelsSheet: Bool = false
    @State private var didInitInspector = false

    private var selectedModel: ModelInfo? {
        store.models.first { $0.id == selectedModelID }
    }

    private var subtitle: String {
        let count = store.models.count
        if count == 0 { return "no models found" }
        return "\(count) model\(count == 1 ? "" : "s") · \(store.modelCachePath)"
    }

    var body: some View {
        ScreenScaffold(
            screen: .models,
            subtitle: subtitle,
            trailing: { actionRow }
        ) {
            if store.models.isEmpty {
                emptyState
            } else {
                modelTable
            }
        }
        .inspector(isPresented: $store.inspectorPresented) {
            inspectorPanel
                .inspectorColumnWidth(min: 260, ideal: 300, max: 320)
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

            // Reveal in Finder — only when a model is selected
            if let model = selectedModel {
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([model.path])
                } label: {
                    Text("Reveal")
                        .font(Theme.Fonts.body)
                }
                .buttonStyle(LatticeSecondaryButtonStyle())

                Divider()
                    .frame(height: 16)

                // Chat CTA — the single navigation action
                Button {
                    store.use(model, on: .chat)
                } label: {
                    Text("Chat")
                        .font(Theme.Fonts.body)
                }
                .buttonStyle(LatticeSecondaryButtonStyle())
                .help("Send selected model to Chat (⌘2)")
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
        ]
    }

    // MARK: Inspector panel

    @ViewBuilder
    private var inspectorPanel: some View {
        if let model = selectedModel {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.xl) {
                    modelInspector(model)
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

