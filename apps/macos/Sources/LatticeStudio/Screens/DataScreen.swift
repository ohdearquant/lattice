import SwiftUI
import AppKit

// MARK: - 02 DATA
//
// Dataset browser and inspector for the lattice training pipeline.
//
// Workflow (top → bottom):
//   STEP 1 — POINT AT FOLDER: path field + Choose/Scan buttons. Defaults to <repoRoot>/data.
//   STEP 2 — INSPECT: FILES TABLE (DataTable, one row per .jsonl) + PREVIEW HSplit.
//             Selecting a file loads a preview and details in the right inspector.
//   STEP 3 — SELECT FOR TRAINING: "Use for Training" button on the selected file sets
//             store.workingDataset, which the TRAIN screen (⌘3) picks up.
//             A persistent status bar shows the currently-selected training dataset.
//   BUILDER SCRIPTS — two helper script commands (copy-only, v1 scope).
//
// File I/O always runs off the main thread (Task.detached); state published on @MainActor.
// Cap: first 5 000 lines per file for counting; preview shows the first 5 examples.

// MARK: - Local data models

struct DataLoadError: Error { let message: String }

struct DatasetFileStat: Identifiable {
    let id: String         // the file path
    let url: URL
    let name: String       // last path component
    let exampleCount: Int  // real line count, capped at 5 000; 0 only if unreadable/empty
    let isCapped: Bool     // true when the file had > 5 000 lines
    let approxTokens: Int  // rough estimate: total chars / 4
    let avgLen: Int        // approxTokens / exampleCount (or 0)
    let sizeBytes: Int64
    let schema: [String]?  // top-level JSON keys of the first row; nil if unreadable/non-object
}

/// One decoded example from a JSONL line.
private struct DatasetExample: Decodable {
    let prompt: String?
    let completion: String?
}

/// One preview pair to display.
private struct PreviewPair: Identifiable {
    let id: Int
    let prompt: String
    let completion: String
    let isRaw: Bool  // true when the line didn't parse as prompt/completion JSON
}

// MARK: - Byte formatter (module-private, single instance)

private let byteFormatter: ByteCountFormatter = {
    let f = ByteCountFormatter()
    f.allowedUnits = [.useBytes, .useKB, .useMB, .useGB]
    f.countStyle = .file
    return f
}()

// MARK: - Button styles (local to this file; mirrors ModelsScreen)

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
                configuration.isPressed ? Theme.Palette.signal : Theme.Palette.ink
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

// MARK: - Builder script definitions

struct BuilderScript {
    let name: String
    let command: String
    let description: String
}

let builderScripts: [BuilderScript] = [
    BuilderScript(
        name: "build_claude_lora_dataset",
        command: "uv run scripts/build_claude_lora_dataset.py",
        description: "Build LoRA dataset from Claude conversation logs"
    ),
    BuilderScript(
        name: "budget_lora_dataset",
        command: "uv run scripts/budget_lora_dataset.py",
        description: "Budget-trim a LoRA dataset to a token count target"
    ),
]

// MARK: - DataScreen

struct DataScreen: View {
    @Bindable var store: AppStore

    // MARK: Local state

    @State private var dataDir: String = ""
    @State private var isScanning: Bool = false
    @State private var scanError: String? = nil
    @State private var hasScanned: Bool = false  // true once scan() has been called

    @State private var files: [DatasetFileStat] = []
    @State private var selectedFileID: String? = nil

    @State private var previewPairs: [PreviewPair] = []
    @State private var previewLoading: Bool = false
    @State private var previewError: String? = nil

    // Transient confirmation flash after "Use for Training" is tapped
    @State private var didConfirmTraining: Bool = false

    // MARK: Derived

    private var selectedFile: DatasetFileStat? {
        files.first { $0.id == selectedFileID }
    }

    private var totalExamples: Int { files.reduce(0) { $0 + $1.exampleCount } }
    private var totalTokens: Int  { files.reduce(0) { $0 + $1.approxTokens } }
    private var avgTokensPerExample: Int {
        totalExamples > 0 ? totalTokens / totalExamples : 0
    }

    private var trainCount: Int { files.filter { $0.name.hasPrefix("train") }.reduce(0) { $0 + $1.exampleCount } }
    private var validCount: Int { files.filter { $0.name.hasPrefix("valid") }.reduce(0) { $0 + $1.exampleCount } }

    private var hasCappedFiles: Bool { files.contains { $0.isCapped } }

    private var subtitle: String {
        if isScanning { return "scanning…" }
        if !hasScanned { return "point at a folder of .jsonl files — then inspect and select for training" }
        if files.isEmpty { return "no .jsonl files found — try a different folder" }
        let cap = hasCappedFiles ? " · some files capped at 5 000 lines" : ""
        return "\(files.count) file\(files.count == 1 ? "" : "s") · \(totalExamples) examples\(cap)"
    }

    // MARK: Body

    var body: some View {
        ScreenScaffold(
            screen: .data,
            subtitle: subtitle
        ) {
            VStack(alignment: .leading, spacing: Theme.Space.xl) {

                // STEP 1 — POINT AT FOLDER
                sourcePanel

                // Empty state: before first scan
                if !hasScanned && !isScanning {
                    firstScanEmptyState
                }

                // Empty state: scanned but nothing found
                if hasScanned && !isScanning && files.isEmpty {
                    noFilesEmptyState
                }

                // STEP 2 — INSPECT files and preview
                if !files.isEmpty {
                    summaryStrip
                    filesAndPreview
                }

                // STEP 3 — TRAINING DATASET STATUS
                if !files.isEmpty {
                    trainingDatasetStatus
                }

                // BUILDER SCRIPTS
                if !files.isEmpty {
                    builderPanel
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .inspector(isPresented: $store.inspectorPresented) {
            dataInspectorPanel
                .inspectorColumnWidth(min: 260, ideal: 300, max: 320)
        }
        .onAppear {
            if dataDir.isEmpty {
                dataDir = (store.repoRootPath ?? "") + "/data"
            }
        }
    }

    // MARK: - Empty states

    private var firstScanEmptyState: some View {
        VStack {
            Spacer(minLength: 0)
            HStack {
                Spacer()
                EmptyStateView(
                    systemImage: "tablecells",
                    title: "Point at a folder of .jsonl files",
                    message: "Enter a directory path above and press Scan. Each .jsonl file becomes one row — inspect its schema, row count, and first examples, then select it for LoRA training.",
                    actionLabel: "Choose Directory"
                ) {
                    chooseDirectory()
                }
                Spacer()
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, minHeight: 220)
    }

    private var noFilesEmptyState: some View {
        VStack {
            Spacer(minLength: 0)
            HStack {
                Spacer()
                EmptyStateView(
                    systemImage: "folder.badge.questionmark",
                    title: "No .jsonl files found",
                    message: "The folder has no .jsonl files at the top level or one level deep. Try choosing a different directory, or use the builder scripts below to create a dataset.",
                    actionLabel: "Choose Different Directory"
                ) {
                    chooseDirectory()
                }
                Spacer()
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, minHeight: 220)
    }

    // MARK: - Inspector panel

    @ViewBuilder
    private var dataInspectorPanel: some View {
        if let file = selectedFile {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.xl) {
                    VStack(alignment: .leading, spacing: Theme.Space.lg) {
                        Text(file.name)
                            .font(Theme.Fonts.title)
                            .foregroundStyle(Theme.Palette.ink)
                            .lineLimit(2)

                        LazyVGrid(
                            columns: [
                                GridItem(.flexible(), spacing: Theme.Space.md),
                                GridItem(.flexible(), spacing: Theme.Space.md)
                            ],
                            spacing: Theme.Space.md
                        ) {
                            ReadoutWell(
                                label: "EXAMPLES",
                                value: file.isCapped ? "5 000+" : "\(file.exampleCount)"
                            )
                            ReadoutWell(
                                label: "≈ TOKENS",
                                value: formatLargeNumber(file.approxTokens)
                            )
                            ReadoutWell(
                                label: "AVG LEN",
                                value: "\(file.avgLen)",
                                unit: "tok"
                            )
                            ReadoutWell(
                                label: "SIZE",
                                value: byteFormatter.string(fromByteCount: file.sizeBytes)
                            )
                        }

                        if let schema = file.schema, !schema.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("SCHEMA")
                                    .instrumentLabel()
                                Text(schema.joined(separator: ", "))
                                    .font(Theme.Fonts.cell)
                                    .foregroundStyle(Theme.Palette.inkDim)
                                    .lineLimit(3)
                            }
                            .padding(Theme.Space.sm)
                            .readoutWellSurface()
                        }

                        // "Use for Training" CTA in the inspector
                        useForTrainingButton(file: file)
                            .padding(.top, Theme.Space.sm)
                    }
                }
                .padding(Theme.Space.lg)
            }
            .instrumentPanel()
        } else {
            VStack(spacing: Theme.Space.sm) {
                Spacer()
                Image(systemName: "doc.text")
                    .font(.system(size: 24))
                    .foregroundStyle(Theme.Palette.textTertiary)
                Text("SELECT A FILE")
                    .instrumentLabel()
                Text("Select a .jsonl file from the table to inspect its schema, row count, and example pairs.")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, Theme.Space.lg)
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        }
    }

    // MARK: - SOURCE panel (Step 1)

    private var sourcePanel: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                // Section label
                HStack {
                    Text("STEP 1 — DATASET FOLDER")
                        .instrumentLabel()
                    Spacer()
                }
                .frame(height: Theme.Space.rowHeight)
                .padding(.horizontal, Theme.Space.lg)
                .overlay(alignment: .bottom) {
                    Theme.Palette.hairline.frame(height: 1)
                }

                ParamRowField(
                    label: "FOLDER",
                    text: $dataDir,
                    placeholder: "/path/to/data"
                )

                HStack(spacing: Theme.Space.sm) {
                    Spacer()

                    Button {
                        chooseDirectory()
                    } label: {
                        Text("Choose…")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(OutlineButtonStyle())

                    Button {
                        scan()
                    } label: {
                        HStack(spacing: 4) {
                            if isScanning {
                                ProgressView()
                                    .progressViewStyle(.circular)
                                    .controlSize(.mini)
                            }
                            Text(isScanning ? "Scanning…" : "Scan")
                                .font(Theme.Fonts.body)
                        }
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    .disabled(isScanning || dataDir.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.vertical, Theme.Space.sm)

                if let err = scanError {
                    HStack(spacing: Theme.Space.sm) {
                        GatePill(.fail, label: "scan error")
                        Text(err)
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .lineLimit(1)
                            .truncationMode(.tail)
                        Spacer()
                    }
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.bottom, Theme.Space.sm)
                }
            }
        }
    }

    // MARK: - SUMMARY strip (Step 2 header)

    private var summaryStrip: some View {
        HStack(alignment: .top, spacing: Theme.Space.md) {
            VStack(alignment: .leading, spacing: 0) {
                HeroNumber(
                    value: totalExamples >= 5_000 && hasCappedFiles
                        ? "\(totalExamples)+"
                        : "\(totalExamples)",
                    unit: "EXAMPLES",
                    size: .hero,
                    unitPosition: .below
                )
            }
            .padding(Theme.Space.lg)
            .instrumentPanel()

            VStack(alignment: .leading, spacing: Theme.Space.sm) {
                HStack(spacing: Theme.Space.sm) {
                    ReadoutWell(label: "FILES", value: "\(files.count)")
                    ReadoutWell(label: "≈ TOKENS", value: formatLargeNumber(totalTokens))
                    ReadoutWell(label: "AVG LEN", value: "\(avgTokensPerExample)", unit: "tok")
                }

                if trainCount > 0 || validCount > 0 {
                    HStack(spacing: Theme.Space.sm) {
                        ReadoutWell(label: "TRAIN", value: "\(trainCount)")
                        if validCount > 0 {
                            ReadoutWell(label: "VALID", value: "\(validCount)")
                        }
                        if hasCappedFiles {
                            VStack(alignment: .leading, spacing: 3) {
                                Text("CAP")
                                    .instrumentLabel()
                                GatePill(.warn, label: "5 000+ lines")
                            }
                            .padding(.horizontal, Theme.Space.md)
                            .padding(.vertical, Theme.Space.sm)
                            .readoutWellSurface()
                        }
                    }
                }
            }

            Spacer()
        }
    }

    // MARK: - FILES TABLE + PREVIEW (Step 2 body)

    private var filesAndPreview: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            // Section label
            HStack(spacing: Theme.Space.sm) {
                Text("STEP 2 — SELECT A FILE TO INSPECT")
                    .instrumentLabel()
                if selectedFile == nil {
                    Text("· click a row to preview it and see details in the inspector (⌘\\)")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
                Spacer()
            }

            HSplitView {
                filesTable
                    .frame(minWidth: 400)

                previewPanel
                    .frame(minWidth: 300, idealWidth: 400, maxWidth: 480)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: - FILES TABLE

    private var filesTable: some View {
        ScrollView {
            DataTable(
                rows: files,
                columns: fileColumns,
                selectedID: $selectedFileID,
                comfortable: store.rowComfortable
            )
        }
        .instrumentPanel()
        .onChange(of: selectedFileID) { _, newID in
            if let id = newID, let file = files.first(where: { $0.id == id }) {
                loadPreview(for: file)
                // Open inspector automatically when a file is selected so the
                // user can see the details without needing to know ⌘\.
                if !store.inspectorPresented {
                    store.inspectorPresented = true
                }
            } else {
                previewPairs = []
                previewError = nil
            }
        }
    }

    private var fileColumns: [ColumnDef<DatasetFileStat>] {
        [
            ColumnDef(
                id: "name",
                header: "FILE",
                alignment: .leading,
                minWidth: 160,
                isNumeric: false
            ) { row in
                // Mark the currently-selected training dataset with a dot
                let isTraining = store.workingDataset?.id == row.id
                return (isTraining ? "● " : "") + (row.isCapped ? "\(row.name) *" : row.name)
            },
            ColumnDef(
                id: "examples",
                header: "EXAMPLES",
                alignment: .trailing,
                minWidth: 80,
                isNumeric: true
            ) { row in
                row.isCapped ? "5 000+" : "\(row.exampleCount)"
            },
            ColumnDef(
                id: "tokens",
                header: "≈ TOKENS",
                alignment: .trailing,
                minWidth: 90,
                isNumeric: true
            ) { row in
                row.isCapped
                    ? formatLargeNumber(row.approxTokens) + "+"
                    : formatLargeNumber(row.approxTokens)
            },
            ColumnDef(
                id: "avg",
                header: "AVG LEN",
                alignment: .trailing,
                minWidth: 72,
                isNumeric: true
            ) { "\($0.avgLen)" },
            ColumnDef(
                id: "size",
                header: "SIZE",
                alignment: .trailing,
                minWidth: 80,
                isNumeric: true
            ) { byteFormatter.string(fromByteCount: $0.sizeBytes) },
            ColumnDef(
                id: "schema",
                header: "SCHEMA",
                alignment: .leading,
                minWidth: 130,
                isNumeric: false
            ) { $0.schema?.joined(separator: ", ") ?? "—" },
        ]
    }

    // MARK: - PREVIEW panel

    @ViewBuilder
    private var previewPanel: some View {
        if previewLoading {
            VStack {
                Spacer()
                ProgressView()
                    .progressViewStyle(.circular)
                    .controlSize(.regular)
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        } else if let err = previewError {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                GatePill(.fail, label: "preview error")
                Text(err)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .lineLimit(4)
                Spacer()
            }
            .padding(Theme.Space.lg)
            .instrumentPanel()
        } else if selectedFileID == nil {
            VStack(spacing: Theme.Space.sm) {
                Spacer()
                Image(systemName: "text.alignleft")
                    .font(.system(size: 24))
                    .foregroundStyle(Theme.Palette.textTertiary)
                Text("SELECT A FILE")
                    .instrumentLabel()
                Text("Click any row in the table to preview its first examples here.")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, Theme.Space.lg)
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        } else if previewPairs.isEmpty {
            VStack(spacing: Theme.Space.sm) {
                Spacer()
                Text("NO EXAMPLES")
                    .instrumentLabel()
                Text("The file is empty or could not be parsed.")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.md) {
                    HStack {
                        Text("PREVIEW")
                            .instrumentLabel()
                        if let file = selectedFile {
                            Text("· \(file.name)")
                                .font(Theme.Fonts.cell)
                                .foregroundStyle(Theme.Palette.inkDim)
                                .monospacedDigit()
                        }
                        Spacer()
                        Text("first \(previewPairs.count) of \(selectedFile.map { $0.isCapped ? "5 000+" : "\($0.exampleCount)" } ?? "?")")
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                    .padding(.horizontal, Theme.Space.sm)

                    ForEach(previewPairs) { pair in
                        exampleCard(pair)
                    }
                }
                .padding(Theme.Space.lg)
            }
            .instrumentPanel()
        }
    }

    private func exampleCard(_ pair: PreviewPair) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            HStack {
                Text("EXAMPLE \(pair.id + 1)")
                    .instrumentLabel()
                if pair.isRaw {
                    GatePill(.warn, label: "RAW LINE")
                }
                Spacer()
            }

            if pair.isRaw {
                ScrollView(.horizontal, showsIndicators: false) {
                    Text(pair.prompt)
                        .font(Theme.Fonts.readout)
                        .foregroundStyle(Theme.Palette.ink)
                        .monospacedDigit()
                        .lineLimit(4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .readoutWellSurface()
                .padding(.vertical, 2)
            } else {
                VStack(alignment: .leading, spacing: 3) {
                    Text("PROMPT")
                        .instrumentLabel()
                    Text(truncated(pair.prompt, limit: 600))
                        .font(Theme.Fonts.readout)
                        .foregroundStyle(Theme.Palette.inkDim)
                        .monospacedDigit()
                        .lineLimit(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(Theme.Space.sm)
                .readoutWellSurface()

                VStack(alignment: .leading, spacing: 3) {
                    Text("COMPLETION")
                        .font(Theme.Fonts.label)
                        .textCase(.uppercase)
                        .tracking(Theme.Space.labelTracking)
                        .foregroundStyle(Theme.Palette.signal)
                    Text(truncated(pair.completion, limit: 600))
                        .font(Theme.Fonts.readout)
                        .foregroundStyle(Theme.Palette.ink)
                        .monospacedDigit()
                        .lineLimit(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(Theme.Space.sm)
                .readoutWellSurface()
            }
        }
        .padding(Theme.Space.sm)
        .background(Theme.Palette.panel)
        .overlay(
            Rectangle()
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
        )
    }

    // MARK: - STEP 3: TRAINING DATASET STATUS

    /// Persistent bar that shows which dataset is queued for training and
    /// exposes the "Use for Training" action for the currently selected file.
    private var trainingDatasetStatus: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                // Section label
                HStack {
                    Text("STEP 3 — TRAINING DATASET")
                        .instrumentLabel()
                    Spacer()
                    Text("consumed by TRAIN ⌘3")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
                .frame(height: Theme.Space.rowHeight)
                .padding(.horizontal, Theme.Space.lg)
                .overlay(alignment: .bottom) {
                    Theme.Palette.hairline.frame(height: 1)
                }

                // Current selection row
                HStack(spacing: Theme.Space.sm) {
                    if let ds = store.workingDataset {
                        // Show dot + name to match the table indicator
                        GatePill(.pass, label: "SELECTED")
                        Text(ds.name)
                            .font(Theme.Fonts.readout)
                            .foregroundStyle(Theme.Palette.ink)
                            .monospacedDigit()
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text("·")
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                        Text(ds.isCapped ? "5 000+ examples" : "\(ds.exampleCount) examples")
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                        Spacer()
                        // Navigate to TRAIN
                        Button {
                            store.selection = .train
                        } label: {
                            Text("Go to Train ⌘3")
                                .font(Theme.Fonts.body)
                        }
                        .buttonStyle(PrimaryButtonStyle())
                    } else {
                        Image(systemName: "circle.dashed")
                            .foregroundStyle(Theme.Palette.textTertiary)
                            .font(.system(size: 12))
                        Text("No dataset selected for training")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)
                        Spacer()
                        // Use selected file as training dataset
                        if let file = selectedFile {
                            useForTrainingButton(file: file)
                        } else {
                            Text("Select a file above")
                                .font(Theme.Fonts.cell)
                                .foregroundStyle(Theme.Palette.textTertiary)
                        }
                    }
                }
                .frame(height: Theme.Space.rowHeight)
                .padding(.horizontal, Theme.Space.lg)

                // If a file is selected and it's NOT the current training dataset,
                // offer to switch to it.
                if let file = selectedFile, store.workingDataset?.id != file.id {
                    HStack(spacing: Theme.Space.sm) {
                        Text("Selected: \(file.name)")
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        useForTrainingButton(file: file)
                    }
                    .frame(height: Theme.Space.rowHeight)
                    .padding(.horizontal, Theme.Space.lg)
                    .overlay(alignment: .top) {
                        Theme.Palette.hairline.frame(height: 1)
                    }
                }
            }
        }
    }

    /// The teal "Use for Training" button — sets store.workingDataset and shows a brief confirmation.
    @ViewBuilder
    private func useForTrainingButton(file: DatasetFileStat) -> some View {
        if didConfirmTraining && store.workingDataset?.id == file.id {
            // Brief confirmation label replaces the button
            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(Theme.Palette.signal)
                    .font(.system(size: 11))
                Text("Set for training")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.signal)
            }
        } else {
            Button {
                store.workingDataset = file
                didConfirmTraining = true
                // Clear the confirmation flash after 2 seconds
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(2))
                    didConfirmTraining = false
                }
            } label: {
                Text("Use for Training")
                    .font(Theme.Fonts.body)
            }
            .buttonStyle(PrimaryButtonStyle())
        }
    }

    // MARK: - BUILDER panel

    private var builderPanel: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                HStack {
                    Text("BUILDER SCRIPTS")
                        .instrumentLabel()
                    Spacer()
                    Text("building/running out of scope for v1 — copy command and run in terminal")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                        .lineLimit(1)
                }
                .frame(height: Theme.Space.rowHeight)
                .padding(.horizontal, Theme.Space.lg)
                .overlay(alignment: .bottom) {
                    Theme.Palette.hairline.frame(height: 1)
                }

                ForEach(builderScripts, id: \.name) { script in
                    scriptRow(script)
                }
            }
        }
    }

    private func scriptRow(_ script: BuilderScript) -> some View {
        HStack(spacing: Theme.Space.sm) {
            Text(script.command)
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer()

            Text(script.description)
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: 280, alignment: .trailing)

            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(script.command, forType: .string)
            } label: {
                Text("Copy")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.ink)
            }
            .buttonStyle(OutlineButtonStyle())
        }
        .frame(height: Theme.Space.rowHeight)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    // MARK: - Actions

    private func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Choose the dataset directory"
        panel.prompt = "Select"
        if !dataDir.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: dataDir, isDirectory: true)
        }
        if panel.runModal() == .OK, let url = panel.url {
            dataDir = url.path
        }
    }

    private func scan() {
        let dir = dataDir.trimmingCharacters(in: .whitespaces)
        guard !dir.isEmpty else { return }

        isScanning = true
        scanError = nil
        files = []
        selectedFileID = nil
        previewPairs = []
        previewError = nil
        didConfirmTraining = false

        Task.detached(priority: .userInitiated) {
            let result = await Self.scanDirectory(path: dir)
            await MainActor.run {
                isScanning = false
                hasScanned = true
                switch result {
                case .success(let stats):
                    files = stats
                case .failure(let err):
                    scanError = err.message
                }
            }
        }
    }

    private func loadPreview(for file: DatasetFileStat) {
        previewLoading = true
        previewPairs = []
        previewError = nil

        Task.detached(priority: .userInitiated) {
            let result = await Self.loadExamples(from: file.url, limit: 5)
            await MainActor.run {
                previewLoading = false
                switch result {
                case .success(let pairs):
                    previewPairs = pairs
                case .failure(let err):
                    previewError = err.message
                }
            }
        }
    }

    // MARK: - File I/O (off main thread)

    static func scanDirectory(path: String) async -> Result<[DatasetFileStat], DataLoadError> {
        let dirURL = URL(fileURLWithPath: path, isDirectory: true)
        let fm = FileManager.default

        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue else {
            return .failure(DataLoadError(message: "not a directory: \(path)"))
        }

        var candidates: [URL] = []

        do {
            let topContents = try fm.contentsOfDirectory(
                at: dirURL,
                includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
                options: [.skipsHiddenFiles]
            )
            for item in topContents {
                let vals = try? item.resourceValues(forKeys: [.isDirectoryKey])
                let itemIsDir = vals?.isDirectory ?? false
                if itemIsDir {
                    if let sub = try? fm.contentsOfDirectory(
                        at: item,
                        includingPropertiesForKeys: [.isDirectoryKey],
                        options: [.skipsHiddenFiles]
                    ) {
                        for f in sub where f.pathExtension == "jsonl" {
                            candidates.append(f)
                        }
                    }
                } else if item.pathExtension == "jsonl" {
                    candidates.append(item)
                }
            }
        } catch {
            return .failure(DataLoadError(message: "could not read directory: \(error.localizedDescription)"))
        }

        guard !candidates.isEmpty else {
            return .success([])
        }

        var stats: [DatasetFileStat] = []
        for url in candidates.sorted(by: { $0.path < $1.path }) {
            let stat = parseStat(for: url)
            stats.append(stat)
        }
        return .success(stats)
    }

    static func parseStat(for url: URL) -> DatasetFileStat {
        let fm = FileManager.default

        let sizeBytes: Int64
        if let attrs = try? fm.attributesOfItem(atPath: url.path),
           let sz = attrs[.size] as? Int64 {
            sizeBytes = sz
        } else {
            sizeBytes = 0
        }

        let cap = 5_000
        guard let content = try? String(contentsOf: url, encoding: .utf8) else {
            return DatasetFileStat(
                id: url.path, url: url, name: url.lastPathComponent,
                exampleCount: 0, isCapped: false,
                approxTokens: 0, avgLen: 0, sizeBytes: sizeBytes, schema: nil
            )
        }

        var lines = content.components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let isCapped = lines.count > cap
        if isCapped { lines = Array(lines.prefix(cap)) }

        let exampleCount = lines.count
        let totalChars = lines.reduce(0) { $0 + $1.count }
        let approxTokens = totalChars / 4
        let avgLen = exampleCount > 0 ? approxTokens / exampleCount : 0

        var schema: [String]? = nil
        if let first = lines.first,
           let d = first.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            schema = obj.keys.sorted()
        }

        return DatasetFileStat(
            id: url.path, url: url, name: url.lastPathComponent,
            exampleCount: exampleCount, isCapped: isCapped,
            approxTokens: approxTokens, avgLen: avgLen, sizeBytes: sizeBytes, schema: schema
        )
    }

    private static func loadExamples(from url: URL, limit: Int) async -> Result<[PreviewPair], DataLoadError> {
        guard let content = try? String(contentsOf: url, encoding: .utf8) else {
            return .failure(DataLoadError(message: "could not read file"))
        }

        let lines = content
            .components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }

        var pairs: [PreviewPair] = []
        var index = 0

        for line in lines.prefix(limit) {
            guard let data = line.data(using: .utf8) else { continue }

            if let example = try? JSONDecoder().decode(DatasetExample.self, from: data),
               let prompt = example.prompt,
               let completion = example.completion {
                pairs.append(PreviewPair(
                    id: index,
                    prompt: prompt,
                    completion: completion,
                    isRaw: false
                ))
            } else {
                pairs.append(PreviewPair(
                    id: index,
                    prompt: line,
                    completion: "",
                    isRaw: true
                ))
            }
            index += 1
        }

        return .success(pairs)
    }

    // MARK: - Helpers

    private func truncated(_ s: String, limit: Int) -> String {
        guard s.count > limit else { return s }
        return String(s.prefix(limit)) + "…"
    }

    private func formatLargeNumber(_ n: Int) -> String {
        switch n {
        case ..<10_000:
            return "\(n)"
        case ..<1_000_000:
            return String(format: "%.1fk", Double(n) / 1_000)
        default:
            return String(format: "%.1fM", Double(n) / 1_000_000)
        }
    }
}
