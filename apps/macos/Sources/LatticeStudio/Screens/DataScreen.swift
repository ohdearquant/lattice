import SwiftUI
import AppKit

// MARK: - 05 DATA
//
// Dataset builder / inspector for the lattice training pipeline.
//
// Layout:
//   SOURCE (OpaquePanel)    — data directory path field + Choose + Scan button
//   SUMMARY (readout strip)  — HeroNumber (total examples) + ReadoutWells for files/tokens/avg/splits
//   FILES TABLE (DataTable)  — one row per .jsonl file; selecting loads preview
//   PREVIEW (OpaquePanel)    — first 5 prompt/completion pairs from the selected file
//   BUILDER (OpaquePanel)    — two builder script commands with Copy buttons
//
// File I/O is always off the main thread (Task.detached); state published on MainActor.
// Cap: first 5 000 lines per file for counting; preview is the first 5 examples.

// MARK: - Local data models

/// One scanned JSONL file.
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

// MARK: - Button styles (mirrors ModelsScreen; local to this file)

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
//
// Top-level nav destination (Screen.data, ⌘2).
// Dataset builder / inspector for the lattice training pipeline.
// The static `scanDirectory(path:)` helper backs this screen's own dataset scan.

struct DataScreen: View {
    @Bindable var store: AppStore

    // MARK: Local state

    @State private var didInitInspector = false
    @State private var dataDir: String = ""
    @State private var isScanning: Bool = false
    @State private var scanError: String? = nil

    @State private var files: [DatasetFileStat] = []
    @State private var selectedFileID: String? = nil

    @State private var previewPairs: [PreviewPair] = []
    @State private var previewLoading: Bool = false
    @State private var previewError: String? = nil

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
        if files.isEmpty { return "no files scanned yet — enter a directory and press Scan" }
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
                // 1. SOURCE — path field + buttons
                sourcePanel

                // 2. SUMMARY — hero + readout wells
                if !files.isEmpty {
                    summaryStrip
                }

                // 3. FILES TABLE + PREVIEW (HSplit)
                if !files.isEmpty {
                    HSplitView {
                        // Files table (left/center)
                        filesTable
                            .frame(minWidth: 400)

                        // Preview panel (right)
                        previewPanel
                            .frame(minWidth: 300, idealWidth: 400, maxWidth: 480)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }

                // 4. BUILDER scripts
                builderPanel
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .inspector(isPresented: $store.inspectorPresented) {
            dataInspectorPanel
                .inspectorColumnWidth(min: 260, ideal: 300, max: 320)
        }
        .onAppear {
            openInspectorOnce()
            // Default to repoRootPath/data, or fall back to empty
            if dataDir.isEmpty {
                dataDir = (store.repoRootPath ?? "") + "/data"
            }
        }
    }

    private func openInspectorOnce() {
        guard !didInitInspector else { return }
        didInitInspector = true
        store.inspectorPresented = true
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
                            columns: [GridItem(.flexible(), spacing: Theme.Space.md), GridItem(.flexible(), spacing: Theme.Space.md)],
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
                    }
                }
                .padding(Theme.Space.lg)
            }
            .instrumentPanel()
        } else {
            VStack {
                Spacer()
                Text("SELECT A FILE")
                    .instrumentLabel()
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        }
    }

    // MARK: - SOURCE panel

    private var sourcePanel: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                ParamRowField(
                    label: "DATA DIR",
                    text: $dataDir,
                    placeholder: "/path/to/data"
                )

                HStack(spacing: Theme.Space.sm) {
                    Spacer()

                    // Choose directory
                    Button {
                        chooseDirectory()
                    } label: {
                        Text("Choose…")
                            .font(Theme.Fonts.body)
                    }
                    .buttonStyle(OutlineButtonStyle())

                    // Scan — the one teal primary on this screen
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

                // Error banner
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

    // MARK: - SUMMARY strip

    private var summaryStrip: some View {
        HStack(alignment: .top, spacing: Theme.Space.md) {
            // Hero: total examples
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

            // Readout wells
            VStack(alignment: .leading, spacing: Theme.Space.sm) {
                HStack(spacing: Theme.Space.sm) {
                    ReadoutWell(
                        label: "FILES",
                        value: "\(files.count)"
                    )
                    ReadoutWell(
                        label: "≈ TOKENS",
                        value: formatLargeNumber(totalTokens)
                    )
                    ReadoutWell(
                        label: "AVG LEN",
                        value: "\(avgTokensPerExample)",
                        unit: "tok"
                    )
                }

                if trainCount > 0 || validCount > 0 {
                    HStack(spacing: Theme.Space.sm) {
                        ReadoutWell(
                            label: "TRAIN",
                            value: "\(trainCount)"
                        )
                        if validCount > 0 {
                            ReadoutWell(
                                label: "VALID",
                                value: "\(validCount)"
                            )
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
                store.workingDataset = file
            } else {
                previewPairs = []
                previewError = nil
                store.workingDataset = nil
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
                row.isCapped ? "\(row.name) *" : row.name
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
                // Capped files sum tokens over only the first 5 000 lines, so the
                // total is a lower bound — mark it with "+" like the EXAMPLES column.
                row.isCapped ? formatLargeNumber(row.approxTokens) + "+" : formatLargeNumber(row.approxTokens)
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
        } else if previewPairs.isEmpty && selectedFileID == nil {
            VStack {
                Spacer()
                Text("SELECT A FILE")
                    .instrumentLabel()
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        } else if previewPairs.isEmpty {
            VStack {
                Spacer()
                Text("NO EXAMPLES")
                    .instrumentLabel()
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.md) {
                    // Preview header
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

                    // Example pairs
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
            // Header: example index
            HStack {
                Text("EXAMPLE \(pair.id + 1)")
                    .instrumentLabel()
                if pair.isRaw {
                    GatePill(.warn, label: "RAW LINE")
                }
                Spacer()
            }

            if pair.isRaw {
                // Raw unparsed line — show as-is
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
                // PROMPT
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

                // COMPLETION
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

    // MARK: - BUILDER panel

    private var builderPanel: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                // Section label row
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
            // Command (mono)
            Text(script.command)
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer()

            // Description (dim)
            Text(script.description)
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: 280, alignment: .trailing)

            // Copy button
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

        Task.detached(priority: .userInitiated) {
            let result = await Self.scanDirectory(path: dir)
            await MainActor.run {
                isScanning = false
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

    // MARK: - File IO (off main thread)

    static func scanDirectory(path: String) async -> Result<[DatasetFileStat], DataLoadError> {
        let dirURL = URL(fileURLWithPath: path, isDirectory: true)
        let fm = FileManager.default

        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue else {
            return .failure(DataLoadError(message: "not a directory: \(path)"))
        }

        // Collect .jsonl files: immediate children + one level of subdirectories
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
                    // One level deep
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

        // File size
        let sizeBytes: Int64
        if let attrs = try? fm.attributesOfItem(atPath: url.path),
           let sz = attrs[.size] as? Int64 {
            sizeBytes = sz
        } else {
            sizeBytes = 0
        }

        // Read content (capped)
        let cap = 5_000
        guard let content = try? String(contentsOf: url, encoding: .utf8) else {
            return DatasetFileStat(
                id: url.path, url: url, name: url.lastPathComponent,
                exampleCount: 0, isCapped: false,
                approxTokens: 0, avgLen: 0, sizeBytes: sizeBytes, schema: nil
            )
        }

        var lines = content.components(separatedBy: "\n").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let isCapped = lines.count > cap
        if isCapped { lines = Array(lines.prefix(cap)) }

        let exampleCount = lines.count
        let totalChars = lines.reduce(0) { $0 + $1.count }
        let approxTokens = totalChars / 4
        let avgLen = exampleCount > 0 ? approxTokens / exampleCount : 0

        // Real first-row schema: top-level JSON keys of the first example.
        // Honest-nil when the line is not a JSON object (raw text, malformed, etc.).
        var schema: [String]? = nil
        if let first = lines.first,
           let d = first.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            schema = obj.keys.sorted()
        }

        return DatasetFileStat(
            id: url.path, url: url, name: url.lastPathComponent,
            exampleCount: exampleCount, isCapped: isCapped,
            approxTokens: approxTokens, avgLen: avgLen, sizeBytes: sizeBytes,
            schema: schema
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
                // Could not parse as prompt/completion — show raw line
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

    /// Truncate a string to `limit` characters with an ellipsis if longer.
    private func truncated(_ s: String, limit: Int) -> String {
        guard s.count > limit else { return s }
        return String(s.prefix(limit)) + "…"
    }

    /// Format large numbers compactly: 1 234 → "1 234", 12 345 → "12.3k", 1 234 567 → "1.2M".
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
