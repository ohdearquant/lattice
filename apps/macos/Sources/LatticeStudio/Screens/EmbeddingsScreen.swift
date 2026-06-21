import SwiftUI

// MARK: - 05 EMBEDDINGS
//
// A single-pane HSplitView tool for testing embedding models.
// Left panel (CONFIG, ~300-340pt): model picker + editable text list + CTA.
// Right panel (RESULTS): readout wells + N×N cosine similarity matrix.
//
// Layout: ScreenScaffold (no .inspector — hasInspector returns false for .embed).
//
// Engine contract (verified 2026-06-21):
//   embed --model <NAME> --text <T1> --text <T2> ... --json
//   Emits: @@lattice {"ev":"embed_done","model":"...","dims":384,"count":3,
//                     "cosine":[[...]],"preview":[[...]],"ms":140}
//
// Honesty contract:
//   - MS shown as "—" when absent (r.ms == nil).
//   - Cosine matrix only rendered when r.cosine is present and square (count×count).
//   - Preview section only shown when r.preview is present and non-empty.
//   - embedModels falls back to a static cached-model list when store.models has no
//     embedding entries, with a caption explaining models may download on first use.

struct EmbeddingsScreen: View {
    @Bindable var store: AppStore

    // MARK: Config state

    @State private var selectedModelName: String = ""
    @State private var texts: [String] = [
        "A cat sits quietly on the warm windowsill.",
        "A feline rests in the sunshine by the window.",
        "The stock market rallied sharply on Tuesday afternoon."
    ]

    // MARK: Result state

    @State private var embedResult: LatticeEvent.EmbedDone?
    @State private var resultTexts: [String] = []   // snapshot of texts at embed time (for matrix labels)
    @State private var isEmbedding: Bool = false
    @State private var embedError: String?

    // MARK: Derived

    /// Cached embedding model names: always available even when the binary hasn't been run.
    private let staticCachedModels: [String] = [
        "bge-small-en-v1.5",
        "all-minilm-l6-v2",
        "multilingual-e5-small",
        "paraphrase-multilingual-minilm-l12-v2"
    ]

    private var embedModels: [ModelInfo] {
        store.models.filter { $0.isEmbedding }
    }

    /// Options shown in the model picker: real discovered models when available,
    /// otherwise the known-cached fallback list.
    private var pickerOptions: [String] {
        if !embedModels.isEmpty {
            return embedModels.map(\.name)
        }
        return staticCachedModels
    }

    private var usingStaticFallback: Bool { embedModels.isEmpty }

    /// At least 2 non-empty texts are required to enable the CTA.
    private var canEmbed: Bool {
        !isEmbedding &&
        !selectedModelName.isEmpty &&
        texts.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.count >= 2
    }

    // MARK: Subtitle

    private var subtitle: String {
        if let r = embedResult {
            let modelLabel = r.model ?? selectedModelName
            return "\(modelLabel) · \(r.dims) dims · \(r.count) texts"
        }
        if isEmbedding { return "embedding…" }
        return "\(embedModels.count) embed model(s) · cosine similarity"
    }

    // MARK: Body

    var body: some View {
        ScreenScaffold(screen: .embed, subtitle: subtitle) {
            HSplitView {
                configPanel
                    .frame(minWidth: 280, idealWidth: 300, maxWidth: 340)
                resultsPanel
                    .frame(minWidth: 400)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onAppear { applyDefaults() }
        .onChange(of: store.models) { _, _ in applyDefaults() }
    }

    // MARK: CONFIG panel (left)

    private var configPanel: some View {
        OpaquePanel {
            VStack(alignment: .leading, spacing: 0) {

                Text("CONFIG")
                    .instrumentLabel()
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.md)
                    .padding(.bottom, Theme.Space.xs)

                // MODEL picker
                ParamRowMenu(
                    label: "MODEL",
                    options: pickerOptions,
                    selection: $selectedModelName
                )

                if usingStaticFallback {
                    Text("models download on first use if not cached")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.inkDim)
                        .padding(.horizontal, Theme.Space.lg)
                        .padding(.bottom, Theme.Space.xs)
                }

                // TEXTS section header
                Text("TEXTS")
                    .instrumentLabel()
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.sm)
                    .padding(.bottom, Theme.Space.xs)

                // Text input rows
                VStack(spacing: 0) {
                    ForEach(texts.indices, id: \.self) { i in
                        textRow(index: i)
                    }
                }

                // Add text button
                Button {
                    texts.append("")
                } label: {
                    HStack(spacing: Theme.Space.xs) {
                        Image(systemName: "plus.circle")
                            .font(.system(size: 12, weight: .medium))
                        Text("Add text")
                            .font(Theme.Fonts.cell)
                    }
                    .foregroundStyle(Theme.Palette.inkDim)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, Theme.Space.lg)
                .padding(.vertical, Theme.Space.sm)

                Spacer()

                // Primary CTA
                Button {
                    runEmbed()
                } label: {
                    Text("▶ EMBED")
                        .font(Theme.Fonts.readout)
                        .foregroundStyle(Theme.Palette.canvas)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, Theme.Space.sm)
                        .background(
                            canEmbed
                                ? Theme.Palette.signal
                                : Theme.Palette.signal.opacity(0.4)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(!canEmbed)
                .padding(Theme.Space.lg)
            }
        }
    }

    // A single editable text row with an optional remove button.
    private func textRow(index i: Int) -> some View {
        HStack(alignment: .top, spacing: Theme.Space.xs) {
            // Label "T<n>"
            Text("T\(i + 1)")
                .font(Theme.Fonts.mono(11))
                .foregroundStyle(Theme.Palette.inkDim)
                .frame(width: 20, alignment: .leading)
                .padding(.top, 6)

            TextField("Text \(i + 1)", text: Binding(
                get: { i < texts.count ? texts[i] : "" },
                set: { if i < texts.count { texts[i] = $0 } }
            ), axis: .vertical)
            .font(Theme.Fonts.body)
            .foregroundStyle(Theme.Palette.ink)
            .lineLimit(2...4)
            .textFieldStyle(.plain)
            .padding(.vertical, Theme.Space.xs)

            // Remove button (only when > 2 texts exist)
            if texts.count > 2 {
                Button {
                    if i < texts.count { texts.remove(at: i) }
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(Theme.Palette.inkDim)
                }
                .buttonStyle(.plain)
                .padding(.top, 6)
            }
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.xs)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    // MARK: RESULTS panel (right)

    @ViewBuilder
    private var resultsPanel: some View {
        if isEmbedding {
            embeddingState
        } else if let errMsg = embedError {
            errorState(errMsg)
        } else if let r = embedResult {
            resultContent(r)
        } else {
            emptyLiveState
        }
    }

    // While the embed binary is running
    private var embeddingState: some View {
        VStack(alignment: .leading, spacing: Theme.Space.lg) {
            HStack(spacing: Theme.Space.sm) {
                GatePill(.run, label: "EMBEDDING…")
            }
            ProgressView()
                .progressViewStyle(.circular)
                .scaleEffect(0.7)
            Text("first run of an uncached model downloads ~130 MB")
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.inkDim)
        }
        .padding(Theme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // Error state
    private func errorState(_ msg: String) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            GatePill(.fail, label: msg)
        }
        .padding(Theme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // Empty state before first run
    private var emptyLiveState: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("NO RESULT YET")
                .instrumentLabel()
            Text("Enter at least two texts and press ▶ EMBED.")
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.inkDim)
        }
        .padding(Theme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // Full result display after a successful embed run
    private func resultContent(_ r: LatticeEvent.EmbedDone) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.xl) {

                // Readout wells: MODEL / DIMS / COUNT / MS
                OpaquePanel {
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Theme.Space.sm) {
                        ReadoutWell(label: "MODEL", value: r.model ?? selectedModelName)
                        ReadoutWell(label: "DIMS",  value: "\(r.dims)")
                        ReadoutWell(label: "COUNT", value: "\(r.count)")
                        ReadoutWell(label: "MS",    value: r.ms.map { String(format: "%.0f", $0) } ?? "—", unit: r.ms != nil ? "ms" : "")
                    }
                    .padding(Theme.Space.lg)
                }

                // Cosine similarity matrix
                if let cosine = r.cosine, cosine.count == r.count,
                   cosine.allSatisfy({ $0.count == r.count }) {
                    OpaquePanel {
                        VStack(alignment: .leading, spacing: Theme.Space.sm) {
                            Text("COSINE SIMILARITY MATRIX")
                                .instrumentLabel()

                            cosineMatrix(cosine: cosine, count: r.count)

                            Text("1.000 = identical · higher = more similar")
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.inkDim)
                        }
                        .padding(Theme.Space.lg)
                    }
                } else if r.cosine == nil {
                    OpaquePanel {
                        Text("cosine matrix not available")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .padding(Theme.Space.lg)
                    }
                }

                // Preview vectors (first 8 dims of each embedding)
                if let preview = r.preview, !preview.isEmpty {
                    OpaquePanel {
                        VStack(alignment: .leading, spacing: Theme.Space.xs) {
                            Text("PREVIEW (first 8 dims)")
                                .instrumentLabel()

                            ForEach(preview.indices, id: \.self) { i in
                                let label = "T\(i + 1)"
                                let vals = preview[i].prefix(8)
                                    .map { String(format: "%.3f", $0) }
                                    .joined(separator: ",  ")
                                HStack(alignment: .top, spacing: Theme.Space.xs) {
                                    Text(label)
                                        .font(Theme.Fonts.mono(11))
                                        .foregroundStyle(Theme.Palette.inkDim)
                                        .frame(width: 24, alignment: .leading)
                                    Text("[\(vals)]")
                                        .font(Theme.Fonts.mono(11))
                                        .foregroundStyle(Theme.Palette.ink)
                                        .monospacedDigit()
                                }
                                .help(i < resultTexts.count ? resultTexts[i] : "")
                            }
                        }
                        .padding(Theme.Space.lg)
                    }
                }
            }
            .padding(Theme.Space.xl)
        }
    }

    // N×N cosine similarity grid
    private func cosineMatrix(cosine: [[Double]], count: Int) -> some View {
        let cellSize: CGFloat = 60

        return VStack(alignment: .leading, spacing: 2) {
            // Column header row (T1, T2, …)
            HStack(spacing: 2) {
                // Blank corner cell aligned with row labels
                Color.clear
                    .frame(width: cellSize, height: 20)

                ForEach(0..<count, id: \.self) { j in
                    Text("T\(j + 1)")
                        .font(Theme.Fonts.mono(11))
                        .foregroundStyle(Theme.Palette.inkDim)
                        .frame(width: cellSize, height: 20)
                        .multilineTextAlignment(.center)
                        .help(j < resultTexts.count ? resultTexts[j] : "")
                }
            }

            // Data rows
            ForEach(0..<count, id: \.self) { i in
                HStack(spacing: 2) {
                    // Row label
                    Text("T\(i + 1)")
                        .font(Theme.Fonts.mono(11))
                        .foregroundStyle(Theme.Palette.inkDim)
                        .frame(width: cellSize, height: cellSize)
                        .multilineTextAlignment(.center)
                        .help(i < resultTexts.count ? resultTexts[i] : "")

                    ForEach(0..<count, id: \.self) { j in
                        let value = i < cosine.count && j < cosine[i].count
                            ? cosine[i][j]
                            : 0.0
                        let isDiag = (i == j)
                        let clampedValue = max(0.0, min(1.0, value))

                        ZStack {
                            // Color-graded background: teal intensity proportional to similarity
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .fill(isDiag
                                    ? Theme.Palette.signal.opacity(0.60)
                                    : Theme.Palette.signal.opacity(clampedValue * 0.50))

                            Text(String(format: "%.3f", value))
                                .font(.system(size: 11, weight: isDiag ? .semibold : .regular, design: .monospaced))
                                .monospacedDigit()
                                .foregroundStyle(isDiag ? Theme.Palette.canvas : Theme.Palette.ink)
                        }
                        .frame(width: cellSize, height: cellSize)
                    }
                }
            }
        }
    }

    // MARK: Embed action

    private func runEmbed() {
        let payload = texts
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard payload.count >= 2 else { return }

        embedError = nil
        embedResult = nil
        isEmbedding = true
        resultTexts = payload

        let config = EmbedConfig(model: selectedModelName, texts: payload)
        let run = store.runEmbed(config)

        run.onComplete = { finished in
            if finished.status == .failed || finished.embed == nil {
                if finished.status == .failed {
                    self.embedError = "Embedding failed. Check the log."
                } else {
                    self.embedError = "No embedding result returned."
                }
            }
            self.embedResult = finished.embed
            self.isEmbedding = false
        }

        // Handle synchronous launch failure
        if run.status == .failed {
            embedError = "Embed failed to launch."
            isEmbedding = false
        }
    }

    // MARK: Defaults

    private func applyDefaults() {
        guard selectedModelName.isEmpty || !pickerOptions.contains(selectedModelName) else { return }
        if pickerOptions.contains("bge-small-en-v1.5") {
            selectedModelName = "bge-small-en-v1.5"
        } else {
            selectedModelName = pickerOptions.first ?? ""
        }
    }
}
