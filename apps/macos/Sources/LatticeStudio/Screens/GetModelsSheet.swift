import SwiftUI
import AppKit

// MARK: - Get Models sheet
//
// Three honest sections:
//   1. DOWNLOAD — 7 embedding models with verified --download-only support.
//      Shows INSTALLED when already in store.models; otherwise a real Download button.
//      In-flight downloads show a spinner + size hint. Errors are surfaced verbatim.
//
//   2. GENERATIVE — import-only models (no engine downloader exists).
//      Provides a "Copy HF URL" button per model. A clear caption explains the workflow.
//      Shows INSTALLED when the model appears in store.models.
//      NO Download button — that would be a dead control (honest-nil discipline).
//
//   3. IMPORT FROM DISK — NSOpenPanel for any model folder.
//      Validates config.json + .safetensors before copying to the model cache.
//      Shows import progress + errors verbatim.

struct GetModelsSheet: View {
    @Bindable var store: AppStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            sheetHeader
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.xl) {
                    downloadSection
                    generativeSection
                    importSection
                }
                .padding(Theme.Space.xl)
            }
        }
        .background(Theme.Palette.canvas)
    }

    // MARK: Header

    private var sheetHeader: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text("GET MODELS")
                    .font(Theme.Fonts.title)
                    .foregroundStyle(Theme.Palette.ink)
                Text("Download embedding models or import any model folder.")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
            }
            Spacer()
            Button("Done") { dismiss() }
                .buttonStyle(LatticeSecondaryButtonStyle())
        }
        .padding(Theme.Space.lg)
    }

    // MARK: Section 1 — Download (embedding models)

    private var downloadSection: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                sectionHeader("DOWNLOAD — EMBEDDING MODELS",
                              note: "checksums verified by the engine")

                ForEach(curatedCatalog.filter { $0.kind == .downloadable }) { model in
                    embeddingRow(model)
                    Theme.Palette.hairline.frame(height: 1)
                }
            }
        }
    }

    @ViewBuilder
    private func embeddingRow(_ model: CuratedModel) -> some View {
        let isInstalled = store.models.contains { $0.name.lowercased() == model.id.lowercased() }
        let isDownloading = store.downloadingModels.contains(model.id)
        let errorMsg = store.downloadErrors[model.id]

        HStack(spacing: Theme.Space.md) {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.name)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                HStack(spacing: Theme.Space.sm) {
                    Text(model.detail)
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                    if let sz = model.approxSize {
                        Text("·")
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                        Text(sz)
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                }
                if let err = errorMsg {
                    HStack(spacing: 4) {
                        GatePill(.fail, label: "FAILED")
                        Text(err)
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .lineLimit(2)
                    }
                    .padding(.top, 2)
                }
            }

            Spacer()

            // Status / action — exactly one of: INSTALLED pill, spinner, or Download button.
            if isInstalled {
                GatePill(.pass, label: "INSTALLED")
            } else if isDownloading {
                HStack(spacing: Theme.Space.sm) {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .controlSize(.small)
                    if let sz = model.approxSize {
                        Text("Downloading… (\(sz))")
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                    } else {
                        Text("Downloading…")
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                }
            } else {
                Button {
                    store.downloadModel(canonicalName: model.id)
                } label: {
                    Text("Download")
                        .font(Theme.Fonts.body)
                }
                .buttonStyle(LatticePrimaryButtonStyle())
            }
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.sm)
    }

    // MARK: Section 2 — Generative (import-only)

    private var generativeSection: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                sectionHeader("IMPORT-ONLY MODELS",
                              note: "download from HuggingFace, then use Import below")

                // Honest caption — these models have no in-app downloader (generative models
                // and the Qwen embedding model, whose loader only does a local-dir lookup).
                HStack {
                    Text("These models have no in-app downloader. Copy the HuggingFace URL, fetch the repo with `git clone` or `huggingface-cli download`, then import the folder below.")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.vertical, Theme.Space.sm)

                Theme.Palette.hairline.frame(height: 1)

                ForEach(curatedCatalog.filter { $0.kind == .importOnly }) { model in
                    generativeRow(model)
                    Theme.Palette.hairline.frame(height: 1)
                }
            }
        }
    }

    @ViewBuilder
    private func generativeRow(_ model: CuratedModel) -> some View {
        let isInstalled = store.models.contains { $0.name.lowercased() == model.id.lowercased() }

        HStack(spacing: Theme.Space.md) {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.name)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                Text(model.detail)
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                if let url = model.hfURL {
                    Text(url)
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.signal)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }

            Spacer()

            if isInstalled {
                GatePill(.pass, label: "INSTALLED")
            } else if let url = model.hfURL {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(url, forType: .string)
                } label: {
                    Text("Copy HF URL")
                        .font(Theme.Fonts.body)
                }
                .buttonStyle(LatticeSecondaryButtonStyle())
                .help("Copy HuggingFace URL to clipboard")
            }
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.sm)
    }

    // MARK: Section 3 — Import from disk

    private var importSection: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                sectionHeader("IMPORT FROM DISK",
                              note: "works for any model: BF16, Q4, embedding")

                VStack(alignment: .leading, spacing: Theme.Space.md) {
                    Text("Choose a model folder. It must contain config.json and at least one .safetensors file. The folder is copied into the model cache (~/.lattice/models).")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                        .fixedSize(horizontal: false, vertical: true)

                    // Progress and error feedback — always honest, never silent.
                    if !store.importingModel.isEmpty {
                        HStack(spacing: Theme.Space.sm) {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .controlSize(.small)
                            Text("Importing \(store.importingModel)…")
                                .font(Theme.Fonts.body)
                                .foregroundStyle(Theme.Palette.inkDim)
                        }
                    }

                    if let err = store.importError {
                        HStack(spacing: 4) {
                            GatePill(.fail, label: "IMPORT FAILED")
                            Text(err)
                                .font(Theme.Fonts.cell)
                                .foregroundStyle(Theme.Palette.inkDim)
                                .lineLimit(3)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    Button {
                        importFromDisk()
                    } label: {
                        Text("Import Model…")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(LatticePrimaryButtonStyle())
                    .disabled(!store.importingModel.isEmpty)
                }
                .padding(Theme.Space.lg)
            }
        }
    }

    // MARK: Helpers

    private func sectionHeader(_ title: String, note: String) -> some View {
        HStack {
            Text(title)
                .instrumentLabel()
            Spacer()
            Text(note)
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
        }
        .frame(height: Theme.Space.rowHeight)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    private func importFromDisk() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Choose a model directory (must contain config.json + .safetensors)"
        panel.prompt = "Import"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        store.importModel(from: url)
    }
}
