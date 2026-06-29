import SwiftUI
import AppKit

// MARK: - Inspect verb tab
//
// The architecture readout for the working model: format, dtype, params, and the full
// transformer geometry (hidden, FFN, heads, KV heads, GDN heads, context, vocab) as a grid of
// ReadoutWells. Every well is honest-nil — a field absent from the model config is omitted, never
// fabricated. Extracted verbatim from the retired ModelsScreen inspector so the geometry readout
// survives the redesign.

private let inspectByteFormatter: ByteCountFormatter = {
    let f = ByteCountFormatter()
    f.allowedUnits = [.useBytes, .useKB, .useMB, .useGB]
    f.countStyle = .file
    return f
}()

struct InspectTab: View {
    @Bindable var store: AppStore

    var body: some View {
        if let model = store.targetModel {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.lg) {
                    header(model)
                    wellGrid(model)
                    revealRow(model)
                }
                .padding(Theme.Space.xl)
                .frame(maxWidth: Theme.Space.dataMaxWidth, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            emptyState
        }
    }

    private func header(_ model: ModelInfo) -> some View {
        HStack(spacing: Theme.Space.sm) {
            Text(model.isEmbedding ? "Embedding model" : "Architecture")
                .font(Theme.Fonts.sectionLabel)
                .textCase(.uppercase)
                .tracking(Theme.Space.labelTracking)
                .foregroundStyle(Theme.Palette.textTertiary)
            Spacer()
            if model.format.isQuantized {
                GatePill(.pass, label: model.format.badge)
            }
        }
    }

    private func wellGrid(_ model: ModelInfo) -> some View {
        let wells = modelWells(for: model)
        return LazyVGrid(
            columns: Array(
                repeating: GridItem(.flexible(), spacing: Theme.Space.md, alignment: .top),
                count: 3),
            spacing: Theme.Space.md
        ) {
            ForEach(wells, id: \.label) { well in
                ReadoutWell(label: well.label, value: well.value, unit: well.unit, minHeight: 56)
            }
        }
    }

    @ViewBuilder
    private func revealRow(_ model: ModelInfo) -> some View {
        HStack(spacing: Theme.Space.sm) {
            Text(model.path.path)
                .font(Theme.Fonts.logFont)
                .foregroundStyle(Theme.Palette.textTertiary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: Theme.Space.md)
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([model.path])
            } label: {
                Text("Reveal in Finder").font(Theme.Fonts.controlText)
            }
            .buttonStyle(LatticeSecondaryButtonStyle())
        }
        .padding(.top, Theme.Space.xs)
    }

    private var emptyState: some View {
        VStack(spacing: Theme.Space.md) {
            Spacer()
            Text("NO MODEL SELECTED")
                .font(Theme.Fonts.sectionLabel)
                .textCase(.uppercase)
                .tracking(Theme.Space.labelTracking)
                .foregroundStyle(Theme.Palette.textTertiary)
            Text("Pick a model in the sidebar to inspect its architecture.")
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textTertiary)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: Well specs (honest-nil — a field absent from the config is omitted)

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
        if let params = model.params { ws.append(WellSpec("PARAMS", params)) }
        if let hidden = model.hidden { ws.append(WellSpec("HIDDEN", "\(hidden)")) }
        if let ffn = model.intermediateSize { ws.append(WellSpec("FFN", "\(ffn)")) }
        if let vocab = model.vocab { ws.append(WellSpec("VOCAB", "\(vocab)")) }
        if let ctx = model.contextLength { ws.append(WellSpec("CTX", "\(ctx)")) }
        if let h = model.attnHeads { ws.append(WellSpec("HEADS", "\(h)")) }
        if let kv = model.kvHeads { ws.append(WellSpec("KV HEADS", "\(kv)")) }
        if let hd = model.headDim { ws.append(WellSpec("HEAD DIM", "\(hd)")) }
        if model.gdnKeyHeads != nil || model.gdnValueHeads != nil {
            let k = model.gdnKeyHeads.map(String.init) ?? "—"
            let v = model.gdnValueHeads.map(String.init) ?? "—"
            ws.append(WellSpec("GDN HEADS", "\(k)/\(v)"))
        }
        if let layers = model.layerSummary { ws.append(WellSpec("LAYERS", layers)) }
        ws.append(WellSpec("SIZE", inspectByteFormatter.string(fromByteCount: model.sizeBytes)))
        ws.append(WellSpec("FILES", "\(model.fileCount)"))
        ws.append(WellSpec("TOKENIZER", model.hasTokenizer ? "yes" : "—"))
        return ws
    }
}
