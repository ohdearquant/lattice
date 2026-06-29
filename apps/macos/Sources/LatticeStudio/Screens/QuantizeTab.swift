import SwiftUI

// MARK: - Quantize verb tab
//
// Runs `quantize_q4` (plain Q4_0) or `quantize_quarot` (Hadamard-rotated Q4_0) over a BF16 model,
// writing a sibling `<name>-q4` / `<name>-q4-quarot` directory. Live progress (per-layer index,
// before/after size, compression ratio, equivalence verdict) streams from the run's LiveRun. Only
// BF16 models can be quantized; an already-quantized target shows an honest note instead of buttons.

struct QuantizeTab: View {
    @Bindable var store: AppStore

    private var model: ModelInfo? { store.targetModel }

    /// The currently-streaming quantization run, if it belongs to this surface.
    private var liveQuant: LiveRun? {
        guard let run = store.liveRun, run.kind == .quantizeQ4 || run.kind == .quantizeQuaRot
        else { return nil }
        return run
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                if let model {
                    if model.isEmbedding {
                        note("Embedding models are served at full precision — quantization is not wired for them.")
                    } else if model.format.isQuantized {
                        note("\(model.name) is already quantized (\(model.format.badge)). Pick a BF16 model to quantize.")
                    } else {
                        actionPanel(model)
                    }
                    if let run = liveQuant { progressPanel(run) }
                } else {
                    note("Pick a BF16 model in the sidebar to quantize it to Q4.")
                }
            }
            .padding(Theme.Space.xl)
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Actions

    private func actionPanel(_ model: ModelInfo) -> some View {
        OpaquePanel {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                Text("Quantize \(model.name)")
                    .font(Theme.Fonts.bodyStrong)
                    .foregroundStyle(Theme.Palette.textPrimary)
                Text("Writes a new sibling directory; the original BF16 weights are left untouched.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)

                HStack(spacing: Theme.Space.sm) {
                    Button {
                        store.startQuantize(quantConfig(model, method: .q4, suffix: "-q4"))
                    } label: {
                        Text("Quantize to Q4").font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticePrimaryButtonStyle())
                    .disabled(liveQuant != nil)

                    Button {
                        store.startQuantize(quantConfig(model, method: .quarot, suffix: "-q4-quarot"))
                    } label: {
                        Text("Quantize to QuaRot Q4").font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                    .disabled(liveQuant != nil)
                }
                Text("QuaRot applies a Hadamard rotation before Q4 to cut outlier error — slower, higher fidelity.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            .padding(Theme.Space.lg)
        }
    }

    private func quantConfig(_ model: ModelInfo, method: QuantMethod, suffix: String) -> QuantConfig {
        let outputDir = model.path
            .deletingLastPathComponent()
            .appendingPathComponent(model.name + suffix, isDirectory: true)
        return QuantConfig(modelDir: model.path, outputDir: outputDir, method: method)
    }

    // MARK: Live progress

    private func progressPanel(_ run: LiveRun) -> some View {
        let frac = run.quantLayerCount > 0
            ? Double(run.quantLayerIndex) / Double(run.quantLayerCount) : 0
        return OpaquePanel {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                HStack {
                    Text(run.status == .running ? "Quantizing…" : (run.status == .done ? "Done" : "Stopped"))
                        .font(Theme.Fonts.bodyStrong)
                        .foregroundStyle(Theme.Palette.textPrimary)
                    Spacer()
                    if run.quantLayerCount > 0 {
                        Text("\(run.quantLayerIndex)/\(run.quantLayerCount) layers")
                            .font(Theme.Fonts.metric)
                            .foregroundStyle(Theme.Palette.textSecondary)
                    }
                }

                ProgressView(value: frac)
                    .tint(run.status == .done ? Theme.Palette.success : Theme.Palette.signal)

                let wells = quantWells(run)
                if !wells.isEmpty {
                    LazyVGrid(
                        columns: Array(
                            repeating: GridItem(.flexible(), spacing: Theme.Space.md, alignment: .top),
                            count: 3),
                        spacing: Theme.Space.md
                    ) {
                        ForEach(wells, id: \.0) { label, value in
                            ReadoutWell(label: label, value: value, minHeight: 52)
                        }
                    }
                }

                if let verdict = run.verdict {
                    GatePill(run.status == .failed ? .fail : .pass, label: verdict)
                }
            }
            .padding(Theme.Space.lg)
        }
    }

    private func quantWells(_ run: LiveRun) -> [(String, String)] {
        var ws: [(String, String)] = []
        if let before = run.quantBeforeMB {
            ws.append(("BEFORE", String(format: "%.0f MB", before)))
        }
        if let after = run.quantAfterMB {
            ws.append(("AFTER", String(format: "%.0f MB", after)))
        }
        if let ratio = run.quantRatio {
            ws.append(("RATIO", String(format: "%.2f×", ratio)))
        }
        if let maxAbs = run.quantMaxAbs {
            ws.append(("MAX ABS ERR", String(format: "%.2e", maxAbs)))
        }
        if let scheme = run.quantScheme {
            ws.append(("SCHEME", scheme))
        }
        return ws
    }

    // MARK: Honest note

    private func note(_ text: String) -> some View {
        HStack(alignment: .top, spacing: Theme.Space.sm) {
            Image(systemName: "info.circle")
                .font(.system(size: 11))
                .foregroundStyle(Theme.Palette.textTertiary)
            Text(text)
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
