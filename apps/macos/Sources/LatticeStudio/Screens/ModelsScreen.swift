import SwiftUI
import AppKit

// MARK: - 01 MODELS
//
// Three-section layout inside ScreenScaffold:
//   • Trailing action row: Refresh + Reveal-in-Finder + three navigation CTAs
//   • Main body: DataTable of store.models (full width) with 2px teal left-border on selection
//   • Right inspector: OpaquePanel with ReadoutWells for the selected model + adapter sub-list
//
// Empty state: if binariesReady is false hint "run `make build`" and show modelCachePath.
// One teal primary CTA per screen: "Train →" (filled); others are outline.

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
    // Quantize sheet: presented from the model inspector/action row (Phase A re-parenting).
    @State private var showQuantizeSheet: Bool = false

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
                HSplitView {
                    // Center: master table
                    modelTable
                        .frame(minWidth: 480)

                    // Right inspector (300px preferred)
                    inspectorPanel
                        .frame(minWidth: 260, idealWidth: 300, maxWidth: 320)
                }
            }
        }
        .sheet(isPresented: $showQuantizeSheet) {
            QuantizeScreen(store: store)
                .frame(minWidth: 760, idealWidth: 900, maxWidth: .infinity,
                       minHeight: 540, idealHeight: 640, maxHeight: .infinity)
        }
    }

    // MARK: Action row (scaffold trailing)

    private var actionRow: some View {
        HStack(spacing: Theme.Space.sm) {
            // Refresh — outline
            Button {
                store.refreshModels()
            } label: {
                HStack(spacing: 4) {
                    Text("Refresh")
                        .font(Theme.Fonts.body)
                    KeyCapChip("⌘R")
                }
            }
            .buttonStyle(OutlineButtonStyle())
            .keyboardShortcut("r", modifiers: .command)

            // Reveal in Finder — only when a model is selected
            if let model = selectedModel {
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([model.path])
                } label: {
                    Text("Reveal")
                        .font(Theme.Fonts.body)
                }
                .buttonStyle(OutlineButtonStyle())
            }

            Divider()
                .frame(height: 16)

            // Navigation CTAs — Train is the single teal primary
            if let model = selectedModel {
                Button {
                    store.use(model, on: .train)
                } label: {
                    HStack(spacing: 4) {
                        Text("Train →")
                            .font(Theme.Fonts.body)
                        KeyCapChip("⌘3")
                    }
                }
                .buttonStyle(PrimaryButtonStyle())

                Button {
                    store.workingModel = model
                    showQuantizeSheet = true
                } label: {
                    HStack(spacing: 4) {
                        Text("Quantize…")
                            .font(Theme.Fonts.body)
                    }
                }
                .buttonStyle(OutlineButtonStyle())

                Button {
                    store.use(model, on: .chat)
                } label: {
                    HStack(spacing: 4) {
                        Text("Chat →")
                            .font(Theme.Fonts.body)
                        KeyCapChip("⌘2")
                    }
                }
                .buttonStyle(OutlineButtonStyle())
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
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Theme.Space.sm) {
                ForEach(wells, id: \.label) { well in
                    ReadoutWell(label: well.label, value: well.value, unit: well.unit)
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
                    .buttonStyle(OutlineButtonStyle())
                }
                .padding(Theme.Space.xl)
            }

            Spacer()
        }
    }
}

// MARK: - Button styles (screen-local; conform to instrument design language)

private struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Theme.Palette.canvas)
            .padding(.horizontal, Theme.Space.md)
            .padding(.vertical, Theme.Space.xs + 2)
            .background(Theme.Palette.signal.opacity(configuration.isPressed ? 0.7 : 1.0))
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
            .animation(.easeOut(duration: Theme.Motion.tick), value: configuration.isPressed)
    }
}

private struct OutlineButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(
                configuration.isPressed
                    ? Theme.Palette.signal
                    : Theme.Palette.ink
            )
            .padding(.horizontal, Theme.Space.md)
            .padding(.vertical, Theme.Space.xs + 2)
            .background(.clear)
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
            )
            .animation(.easeOut(duration: Theme.Motion.tick), value: configuration.isPressed)
    }
}
