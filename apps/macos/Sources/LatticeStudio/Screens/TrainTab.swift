import SwiftUI
import AppKit

// MARK: - Train verb tab
//
// Runs `train_grad_full` — an exact-gradient multi-layer LoRA trainer (CPU backward) — over a BF16
// model and a folder containing `train.jsonl`. Live NLL streams from the run's TrainPoint series.
// Training targets a BF16 checkpoint; a quantized or embedding target shows an honest note. No
// fabricated metrics: the loss shown is the last point the trainer actually emitted.

struct TrainTab: View {
    @Bindable var store: AppStore
    @State private var dataDir: URL?
    @State private var dataError: String?

    private var model: ModelInfo? { store.targetModel }

    private var liveTrain: LiveRun? {
        guard let run = store.liveRun, run.kind == .train else { return nil }
        return run
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                if let model {
                    if model.isEmbedding {
                        note("Embedding models are not trainable here — pick a BF16 generative model.")
                    } else if model.format.isQuantized {
                        note("LoRA training needs full-precision weights. Pick the BF16 \(baseName(model)) instead of \(model.name).")
                    } else {
                        setupPanel(model)
                    }
                    if let run = liveTrain { progressPanel(run) }
                } else {
                    note("Pick a BF16 model in the sidebar to fine-tune it with LoRA.")
                }
            }
            .padding(Theme.Space.xl)
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Setup

    private func setupPanel(_ model: ModelInfo) -> some View {
        OpaquePanel {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                Text("Fine-tune \(model.name)")
                    .font(Theme.Fonts.bodyStrong)
                    .foregroundStyle(Theme.Palette.textPrimary)
                Text("Trains a LoRA adapter over the top transformer layers (default: last 5). The base weights are never modified.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: Theme.Space.sm) {
                    Button { chooseDataFolder() } label: {
                        Text("Choose data folder…").font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())

                    if let dataDir {
                        Text(dataDir.lastPathComponent)
                            .font(Theme.Fonts.codeFont)
                            .foregroundStyle(Theme.Palette.textSecondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    } else {
                        Text("must contain train.jsonl")
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(Theme.Palette.textTertiary)
                    }
                }

                if let dataError {
                    Text(dataError)
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.crimson)
                }

                Button {
                    if let dataDir {
                        store.startTrain(TrainConfig(modelDir: model.path, dataDir: dataDir))
                    }
                } label: {
                    Text("Start LoRA Training").font(Theme.Fonts.body)
                }
                .buttonStyle(LatticePrimaryButtonStyle())
                .disabled(dataDir == nil || liveTrain != nil)
            }
            .padding(Theme.Space.lg)
        }
    }

    private func chooseDataFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Select a folder containing train.jsonl"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let trainJSONL = url.appendingPathComponent("train.jsonl")
        if FileManager.default.fileExists(atPath: trainJSONL.path) {
            dataDir = url
            dataError = nil
        } else {
            dataDir = nil
            dataError = "No train.jsonl found in \(url.lastPathComponent)."
        }
    }

    // MARK: Live progress

    private func progressPanel(_ run: LiveRun) -> some View {
        let last = run.points.last
        let frac: Double = {
            guard let total = run.totalSteps, total > 0, let step = last?.step else { return 0 }
            return Double(step) / Double(total)
        }()
        return OpaquePanel {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                HStack {
                    Text(run.status == .running ? "Training…" : (run.status == .done ? "Done" : "Stopped"))
                        .font(Theme.Fonts.bodyStrong)
                        .foregroundStyle(Theme.Palette.textPrimary)
                    Spacer()
                    if let step = last?.step, let total = run.totalSteps {
                        Text("step \(step)/\(total)")
                            .font(Theme.Fonts.metric)
                            .foregroundStyle(Theme.Palette.textSecondary)
                    }
                }

                if run.totalSteps != nil {
                    ProgressView(value: frac)
                        .tint(run.status == .done ? Theme.Palette.success : Theme.Palette.signal)
                }

                LazyVGrid(
                    columns: Array(
                        repeating: GridItem(.flexible(), spacing: Theme.Space.md, alignment: .top),
                        count: 3),
                    spacing: Theme.Space.md
                ) {
                    if let nll = last?.loss {
                        ReadoutWell(label: "NLL", value: String(format: "%.4f", nll), minHeight: 52)
                    }
                    if let val = run.bestVal {
                        ReadoutWell(label: "BEST VAL", value: String(format: "%.4f", val), minHeight: 52)
                    }
                    if let base = run.baseNLL {
                        ReadoutWell(label: "BASE NLL", value: String(format: "%.4f", base), minHeight: 52)
                    }
                }

                if let saved = run.savedAdapterPath {
                    Text("Saved adapter → \(saved)")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.success)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .padding(Theme.Space.lg)
        }
    }

    // MARK: Helpers

    private func baseName(_ model: ModelInfo) -> String {
        model.name
            .replacingOccurrences(of: "-q4", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "-quarot", with: "", options: .caseInsensitive)
    }

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
