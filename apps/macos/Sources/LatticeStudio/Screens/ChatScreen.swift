import SwiftUI

// MARK: - 02 CHAT
//
// Redesigned as a focused, minimal chat interface with progressive disclosure.
//
// Layout (two zones inside ScreenScaffold):
//   • TAB PICKER   — segmented control "Chat | History" (max 240pt, left-aligned).
//                    In Chat tab: A/B mode picker on the right (max 160pt).
//   • TRANSCRIPT   — scrollable column of chat bubbles (Chat tab), max-width 920pt, centered.
//                    Single mode: one column of ChatTurn bubbles.
//                    A/B compare mode: two equal columns (Base | Adapter), side by side.
//                    OR run-history DataTable + live banner (History tab).
//   • COMPOSER     — rounded TextEditor with Send/Stop as a 30×30 in-composer button
//                    (bottom-trailing overlay). Chat tab only; hidden in History tab.
//
// ScreenScaffold supplies the indexed header "02 · CHAT" + a subtitle line showing
// the active model and adapter. All knobs (model, adapter, temperature, max tokens,
// seed) live in a toggleable right-side .inspector sidebar using InspectorShell.
// Inspector content switches on chatTab: Chat → settings; History → selected-run detail.
//
// Generation model: unchanged from prior implementation.
//   generate_lora prints output ATOMICALLY. A "message" lifecycle is:
//   launch via store.runGenerate(GenConfig) → status == .running →
//   collect store.liveRun.genText (cumulative) → status == .done → copy into pending turn.
//
// Completion detection:
//   .onChange(of: store.liveRun?.status) fires when the subprocess exits.
//   awaitingTurnID: UUID? tracks which turn receives the result.
//
// A/B comparison:
//   AppStore holds a SINGLE handle + liveRun; each runGenerate STOPS any prior run.
//   A/B is therefore SEQUENTIAL: base completes (onComplete fires on MainActor) →
//   adapter run starts. onComplete is the sequencing primitive; status.onChange is
//   single-mode only (it guards on awaitingTurnID which compare mode never sets).

// MARK: - ChatTab

private enum ChatTab: String, CaseIterable {
    case chat    = "Chat"
    case history = "History"
}

// MARK: - ChatMode

private enum ChatMode: String, CaseIterable {
    case single  = "Single"
    case compare = "A/B"
}

// MARK: - Local domain model (single mode)

private struct ChatTurn: Identifiable {
    enum TurnStatus { case running, done, failed }

    let id = UUID()
    let prompt: String
    var responseText: String = ""
    var status: TurnStatus = .running
    var tokensPerSecond: Double? = nil
}

// MARK: - Compare domain model

private struct ComparePair: Identifiable {
    enum Side { case base, adapter }

    let id = UUID()
    let prompt: String           // raw user text (for display)
    let baseLabel: String        // base model name, snapshotted at creation so the
    let adapterLabel: String     // attribution survives later picker changes (codex B2)
    var baseText: String = ""
    var baseTokS: Double? = nil
    var baseDone: Bool = false
    var adapterText: String = ""
    var adapterTokS: Double? = nil
    var adapterDone: Bool = false
    var failed: Bool = false
}

// MARK: - A/B phase tracker

private enum ABPhase { case idle, base, adapter }

// MARK: - Typing indicator dots

private struct TypingDots: View {
    @State private var phase: Int = 0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(Theme.Palette.textTertiary)
                    .frame(width: 6, height: 6)
                    .scaleEffect(phase == i ? 1.25 : 0.85)
                    .animation(
                        .easeInOut(duration: 0.4)
                            .repeatForever(autoreverses: true)
                            .delay(Double(i) * 0.15),
                        value: phase
                    )
            }
        }
        .onAppear { phase = 0 }
        .task {
            // nudge phase so each dot has a unique animation offset
            try? await Task.sleep(nanoseconds: 50_000_000)
            phase = 1
        }
    }
}

// MARK: - Settings inspector content

private struct ChatInspector: View {
    let modelOptions: [ModelInfo]
    @Binding var selectedModelName: String
    let adapterOptions: [String]
    @Binding var selectedAdapterName: String
    @Binding var tempText: String
    @Binding var maxTokensText: String
    @Binding var seedText: String

    var body: some View {
        InspectorShell(title: "Settings") {
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                // Model
                settingsRow(label: "Model") {
                    Picker("", selection: $selectedModelName) {
                        ForEach(modelOptions, id: \.name) { model in
                            Text(model.name).tag(model.name)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .font(Theme.Fonts.body)
                    .disabled(modelOptions.count <= 1)
                }

                // Adapter
                settingsRow(label: "Adapter") {
                    Picker("", selection: $selectedAdapterName) {
                        ForEach(adapterOptions, id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .font(Theme.Fonts.body)
                    .disabled(adapterOptions.count <= 1)
                }

                Divider()

                // Temperature
                settingsRow(label: "Temperature") {
                    LatticeNumericField(prompt: "0.7", text: $tempText, width: 64)
                }

                // Max tokens
                settingsRow(label: "Max tokens") {
                    LatticeNumericField(prompt: "256", text: $maxTokensText, width: 72)
                }

                // Seed
                settingsRow(label: "Seed") {
                    LatticeNumericField(prompt: "random", text: $seedText, width: 88)
                }
            }
        }
    }

    @ViewBuilder
    private func settingsRow<Control: View>(label: String, @ViewBuilder control: () -> Control) -> some View {
        HStack {
            Text(label)
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
            control()
        }
    }
}

// MARK: - ChatScreen

struct ChatScreen: View {
    @Bindable var store: AppStore

    // MARK: Local state

    // Tab selection: Chat playground vs History (run notebook)
    @State private var chatTab: ChatTab = .chat
    // History tab: selected run (owned here so selection survives tab switches)
    @State private var selectedRunID: String?

    // Chat mode: single or A/B compare
    @State private var chatMode: ChatMode = .single

    // Single-mode state
    @State private var turns: [ChatTurn] = []
    @State private var composerText: String = ""
    @State private var awaitingTurnID: UUID?

    // Compare-mode state
    @State private var comparePairs: [ComparePair] = []
    @State private var awaitingPairID: UUID? = nil
    @State private var abPhase: ABPhase = .idle

    // Model selection
    @State private var selectedModelName: String = ""

    // Settings (advanced, in inspector sidebar)
    @State private var selectedAdapterName: String = "none"
    @State private var tempText: String = "0.7"
    @State private var maxTokensText: String = "256"
    @State private var seedText: String = ""

    // User-initiated stop: tracks which turn was running when Stop was pressed.
    // A turn whose id matches this value is treated as a clean stop (partial text
    // kept, not marked .failed) rather than an error.
    @State private var userStoppedTurnID: UUID? = nil

    // MARK: Derived helpers

    // Only bf16 models can be chatted — q4/quarot require safetensors and
    // generate_lora loads from_safetensors; embedding models are not generative.
    private var chatModels: [ModelInfo] {
        store.models.filter { $0.format == .bf16 }
    }

    private var selectedModel: ModelInfo? {
        // If the current selection is still in the filtered list, keep it.
        // Otherwise fall back to the first bf16 model (or nil if none exist).
        if let m = chatModels.first(where: { $0.name == selectedModelName }) { return m }
        return chatModels.first
    }

    private var adapterOptions: [String] {
        let adapters = selectedModel?.adapters.map(\.name) ?? []
        return ["none"] + adapters
    }

    private var selectedAdapter: AdapterInfo? {
        guard selectedAdapterName != "none" else { return nil }
        return selectedModel?.adapters.first { $0.name == selectedAdapterName }
    }

    // isRunning is true while a single-mode run is in flight OR while either phase
    // of A/B is active. This keeps the Stop button live and Send disabled in both modes.
    private var isRunning: Bool {
        store.liveRun(matching: [.chat])?.status == .running
            || abPhase != .idle
    }

    private var canSend: Bool {
        !composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !isRunning
            && !(chatMode == .compare && selectedAdapter == nil)
    }

    // MARK: Subtitle

    private var subtitle: String {
        guard let model = selectedModel else { return "no models found" }
        let adapterLabel = selectedAdapterName == "none" ? "no adapter" : selectedAdapterName
        return "\(model.name) · \(adapterLabel)"
    }

    // MARK: Body

    var body: some View {
        ScreenScaffold(screen: .chat, subtitle: subtitle) {
            VStack(spacing: 0) {
                // Tab / mode picker row
                HStack {
                    // Chat | History tab picker — max 240pt, left-aligned
                    Picker("", selection: $chatTab) {
                        ForEach(ChatTab.allCases, id: \.self) { tab in
                            Text(tab.rawValue).tag(tab)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(maxWidth: 240)

                    Spacer()

                    // Single | A/B mode picker — only visible in Chat tab
                    if chatTab == .chat {
                        Picker("", selection: $chatMode) {
                            ForEach(ChatMode.allCases, id: \.self) { mode in
                                Text(mode.rawValue).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 160)
                        // Disable A/B when no adapter is available for the current model
                        .disabled(chatMode == .single && adapterOptions.count <= 1)
                    }
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.md)
                .padding(.bottom, Theme.Space.sm)

                // Tab content
                switch chatTab {
                case .chat:
                    switch chatMode {
                    case .single:
                        transcriptArea
                    case .compare:
                        compareTranscriptArea
                    }
                    composerBar
                case .history:
                    RunsContent(store: store, selectedRunID: $selectedRunID)
                }
            }
        }
        .inspector(isPresented: $store.inspectorPresented) {
            switch chatTab {
            case .chat:
                ChatInspector(
                    modelOptions: chatModels,
                    selectedModelName: $selectedModelName,
                    adapterOptions: adapterOptions,
                    selectedAdapterName: $selectedAdapterName,
                    tempText: $tempText,
                    maxTokensText: $maxTokensText,
                    seedText: $seedText
                )
                .inspectorColumnWidth(min: 280, ideal: 320, max: 380)
            case .history:
                RunsContent(store: store, selectedRunID: $selectedRunID)
                    .inspectorPanel
                    .inspectorColumnWidth(min: 260, ideal: 300, max: 320)
            }
        }
        .onAppear { applyDefaults() }
        .onChange(of: store.models) { _, _ in applyDefaults() }
        .onChange(of: selectedModelName) { _, _ in
            // Reset adapter when model changes.
            selectedAdapterName = "none"
        }
        .onChange(of: store.liveRun?.status) { _, newStatus in
            // This handler is SINGLE-MODE ONLY. It guards on awaitingTurnID,
            // which compare mode never sets, so it is inert during A/B runs.
            handleRunStatusChange(newStatus)
        }
        .onChange(of: store.liveRun?.genText) { _, newText in
            // Route streaming tokens to whichever mode is currently active.
            guard store.liveRun?.kind == .chat else { return }

            if chatMode == .single {
                // Single mode: update the awaiting turn directly.
                guard let turnID = awaitingTurnID,
                      let idx = turns.firstIndex(where: { $0.id == turnID })
                else { return }
                turns[idx].responseText = newText ?? ""

            } else {
                // Compare mode: route to the correct column based on the current phase.
                guard let pid = awaitingPairID,
                      let idx = comparePairs.firstIndex(where: { $0.id == pid })
                else { return }
                switch abPhase {
                case .base:
                    comparePairs[idx].baseText = newText ?? ""
                case .adapter:
                    comparePairs[idx].adapterText = newText ?? ""
                case .idle:
                    break
                }
            }
        }
        .onChange(of: store.liveRun?.log.count) { _, _ in
            // Belt-and-suspenders: log.count changes are noted; status onChange is the handler.
        }
    }

    // MARK: - Single-mode transcript

    @ViewBuilder
    private var transcriptArea: some View {
        if turns.isEmpty {
            emptyState
        } else {
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    LazyVStack(alignment: .leading, spacing: Theme.Space.lg) {
                        ForEach(turns) { turn in
                            turnView(turn)
                                .id(turn.id)
                        }
                    }
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.vertical, Theme.Space.xl)
                    .frame(maxWidth: Theme.Space.chatMaxWidth)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: turns.count) { _, _ in
                    if let last = turns.last {
                        withAnimation(.easeOut(duration: Theme.Motion.focus)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: turns.last?.responseText) { _, _ in
                    if let last = turns.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack {
            Spacer()
                .frame(minHeight: 0)
                .frame(maxHeight: .infinity)
                .layoutPriority(-1)
            EmptyStateView(
                systemImage: "text.bubble",
                title: "Try it out",
                message: "Send a message to run this model locally on your Mac."
            )
            .frame(maxWidth: .infinity)
            Spacer()
                .frame(minHeight: 0)
                .frame(maxHeight: .infinity)
        }
    }

    // MARK: - Turn bubbles (single mode)

    @ViewBuilder
    private func turnView(_ turn: ChatTurn) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            // User bubble — right-aligned, neutral panel surface (no teal)
            HStack {
                Spacer(minLength: Theme.Space.xxl)
                Text(turn.prompt)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                    .multilineTextAlignment(.trailing)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                            .fill(Theme.Palette.panel)
                            .overlay(
                                RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
                            )
                    )
            }

            // Assistant bubble — left-aligned, surfaceRaised fill
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    if turn.status == .running && turn.responseText.isEmpty {
                        // Typing indicator before any tokens arrive
                        HStack(spacing: Theme.Space.xs) {
                            TypingDots()
                            Text("Thinking…")
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.textSecondary)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(assistantBubbleBackground)
                    } else {
                        // Response text — body font (prose-friendly, not mono)
                        Text(turn.responseText.isEmpty ? "(no output)" : turn.responseText)
                            .font(Theme.Fonts.body)
                            .foregroundStyle(
                                turn.status == .failed
                                    ? Theme.Palette.error
                                    : Theme.Palette.ink
                            )
                            .multilineTextAlignment(.leading)
                            .textSelection(.enabled)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .background(assistantBubbleBackground)

                        // Optional tok/s — muted, small
                        if let tps = turn.tokensPerSecond {
                            Text(String(format: "%.1f tok/s", tps))
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.textTertiary)
                                .monospacedDigit()
                                .padding(.leading, 4)
                        }
                    }
                }
                Spacer(minLength: Theme.Space.xxl)
            }
        }
    }

    private var assistantBubbleBackground: some View {
        RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
            .fill(Theme.Palette.surfaceRaised)
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                    .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
            )
    }

    // MARK: - Compare transcript (A/B mode)

    @ViewBuilder
    private var compareTranscriptArea: some View {
        if selectedAdapter == nil {
            // No adapter selected — show inline hint, disable send via canSend guard
            VStack {
                Spacer()
                    .frame(minHeight: 0)
                    .frame(maxHeight: .infinity)
                    .layoutPriority(-1)
                Text("Select an adapter in Settings to compare against the base model.")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(Theme.Space.lg)
                    .frame(maxWidth: .infinity)
                Spacer()
                    .frame(minHeight: 0)
                    .frame(maxHeight: .infinity)
            }
        } else if comparePairs.isEmpty {
            VStack {
                Spacer()
                    .frame(minHeight: 0)
                    .frame(maxHeight: .infinity)
                    .layoutPriority(-1)
                EmptyStateView(
                    systemImage: "arrow.left.arrow.right",
                    title: "A/B Compare",
                    message: "Send a message to compare the base model against the adapter side by side."
                )
                .frame(maxWidth: .infinity)
                Spacer()
                    .frame(minHeight: 0)
                    .frame(maxHeight: .infinity)
            }
        } else {
            // Two-column split view
            HStack(alignment: .top, spacing: 0) {
                // Base column
                compareColumn(
                    header: "Base",
                    pairs: comparePairs,
                    side: .base
                )

                // Hairline divider between columns
                Rectangle()
                    .fill(Theme.Palette.hairline)
                    .frame(width: 1)
                    .frame(maxHeight: .infinity)

                // Adapter column
                compareColumn(
                    header: "Adapter",
                    pairs: comparePairs,
                    side: .adapter
                )
            }
        }
    }

    // A single side of the two-column compare layout
    @ViewBuilder
    private func compareColumn(
        header: String,
        pairs: [ComparePair],
        side: ComparePair.Side
    ) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            // Column header
            Text(header)
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textSecondary)
                .padding(.horizontal, Theme.Space.md)
                .padding(.vertical, Theme.Space.sm)
                .frame(maxWidth: .infinity, alignment: .leading)

            Rectangle()
                .fill(Theme.Palette.hairline)
                .frame(height: 1)

            // Scrollable pair list
            ScrollView(.vertical) {
                LazyVStack(alignment: .leading, spacing: Theme.Space.lg) {
                    ForEach(pairs) { pair in
                        comparePairCell(pair: pair, side: side)
                            .id(pair.id)
                    }
                }
                .padding(Theme.Space.md)
            }
        }
        .frame(maxWidth: .infinity)
    }

    // One cell in a compare column: prompt label + response + tok/s + Δ chip
    @ViewBuilder
    private func comparePairCell(pair: ComparePair, side: ComparePair.Side) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            // Prompt label
            Text(pair.prompt)
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textSecondary)
                .lineLimit(2)

            // Per-pair attribution — which model/adapter produced this side.
            // Read from the pair's snapshot, so a later picker change cannot relabel
            // a completed result with the wrong model (codex B2).
            Text(side == .base ? pair.baseLabel : pair.adapterLabel)
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textTertiary)
                .lineLimit(1)

            // Response area
            let isThisSideStreaming: Bool = {
                guard let pid = awaitingPairID, pid == pair.id else { return false }
                switch side {
                case .base:    return abPhase == .base    && !pair.baseDone
                case .adapter: return abPhase == .adapter && !pair.adapterDone
                }
            }()

            let responseText: String = side == .base ? pair.baseText : pair.adapterText
            let isDone: Bool         = side == .base ? pair.baseDone : pair.adapterDone

            if isThisSideStreaming && responseText.isEmpty {
                // Typing indicator before any tokens
                HStack(spacing: Theme.Space.xs) {
                    TypingDots()
                    Text("Thinking…")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textSecondary)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(assistantBubbleBackground)
            } else if !responseText.isEmpty || isDone {
                Text(responseText.isEmpty ? "(no output)" : responseText)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(pair.failed ? Theme.Palette.error : Theme.Palette.ink)
                    .multilineTextAlignment(.leading)
                    .textSelection(.enabled)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(assistantBubbleBackground)

                // tok/s caption
                let tokS: Double? = side == .base ? pair.baseTokS : pair.adapterTokS
                if let tps = tokS {
                    Text(String(format: "%.1f tok/s", tps))
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .monospacedDigit()
                }

                // Δ tok/s chip — only when both sides are done
                if side == .adapter, pair.baseDone, pair.adapterDone,
                   let baseS = pair.baseTokS, let adpS = pair.adapterTokS {
                    let delta = adpS - baseS
                    let positive = delta >= 0
                    let label = positive
                        ? String(format: "+%.1f tok/s", delta)
                        : String(format: "%.1f tok/s", delta)
                    Text(label)
                        .font(Theme.Fonts.caption)
                        .monospacedDigit()
                        .foregroundStyle(positive ? Theme.Palette.signal : Theme.Palette.crimson)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 4, style: .continuous)
                                .fill(positive
                                      ? Theme.Palette.signal.opacity(0.12)
                                      : Theme.Palette.crimson.opacity(0.12))
                        )
                }
            } else {
                // Waiting for this side to start (e.g. adapter not yet begun)
                Text("Waiting…")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
            }
        }
    }

    // MARK: - Composer

    private var composerBar: some View {
        VStack(spacing: 0) {
            Rectangle()
                .fill(Theme.Palette.hairline)
                .frame(height: 1)

            // Centered 920-column composer, matching transcript column
            HStack {
                composerField
                    .frame(maxWidth: Theme.Space.chatMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            .padding(Theme.Space.lg)
            .background(Theme.Palette.canvas)
        }
    }

    // The text field with an in-compositor Send/Stop button at the bottom-trailing edge.
    private var composerField: some View {
        ZStack(alignment: .topLeading) {
            // Placeholder
            if composerText.isEmpty {
                Text("Message…")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, 10)
                    .padding(.top, 9)
                    .allowsHitTesting(false)
            }
            // TextEditor — trailing padding reserves space for the 30×30 button
            TextEditor(text: $composerText)
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.ink)
                .scrollContentBackground(.hidden)
                .background(.clear)
                .padding(.horizontal, 6)
                .padding(.vertical, 4)
                .padding(.trailing, 40)
        }
        .frame(minHeight: 44, maxHeight: 160)
        .background(Theme.Palette.surfaceRaised)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
        )
        // In-composer Send / Stop button at bottom-trailing
        .overlay(alignment: .bottomTrailing) {
            actionButton
                .padding(7)
        }
    }

    @ViewBuilder
    private var actionButton: some View {
        if isRunning {
            // Stop button — works for both single and compare modes
            Button {
                if chatMode == .single {
                    userStoppedTurnID = awaitingTurnID
                } else {
                    // In A/B mode, stop current run and mark the pair as done with partial text
                    if let pid = awaitingPairID,
                       let idx = comparePairs.firstIndex(where: { $0.id == pid }) {
                        switch abPhase {
                        case .base:
                            comparePairs[idx].baseDone = true
                        case .adapter:
                            comparePairs[idx].adapterDone = true
                        case .idle:
                            break
                        }
                        comparePairs[idx].failed = true
                    }
                    abPhase = .idle
                    awaitingPairID = nil
                }
                store.stopRun()
            } label: {
                Image(systemName: "stop.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.Palette.crimson)
                    .frame(width: 30, height: 30)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                            .fill(Theme.Palette.crimson.opacity(0.12))
                    )
            }
            .buttonStyle(.plain)
            .help("Stop generation")
        } else {
            // Send button — dispatches to the appropriate mode handler
            Button(action: {
                switch chatMode {
                case .single:
                    send()
                case .compare:
                    sendCompare()
                }
            }) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(canSend ? Theme.Palette.onAccent : Theme.Palette.textTertiary)
                    .frame(width: 30, height: 30)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                            .fill(canSend ? Theme.Palette.signal : Theme.Palette.wellSink)
                    )
            }
            .buttonStyle(.plain)
            .disabled(!canSend)
            .keyboardShortcut(.return, modifiers: .command)
            .help("Send  ⌘↵")
        }
    }

    // MARK: - Actions

    private func applyDefaults() {
        // Pick the first bf16 model if the current selection is missing or not bf16-generative.
        if selectedModelName.isEmpty || chatModels.first(where: { $0.name == selectedModelName }) == nil {
            // Prefer the store's target model if it is bf16, otherwise take the first bf16 model.
            let target = store.targetModel
            if let t = target, t.format == .bf16 {
                selectedModelName = t.name
            } else {
                selectedModelName = chatModels.first?.name ?? ""
            }
        }
    }

    /// Render completed turns + newUserText as Qwen ChatML, mirroring the
    /// authoritative `render_prompt` in crates/inference/src/bin/lattice.rs
    /// (the /v1/chat/completions serve path). The tokenizer carries
    /// <|im_start|>/<|im_end|> as special tokens and GenerateConfig's default
    /// stop_token_ids includes <|im_end|>, so the assistant turn terminates
    /// correctly. No trailing <|im_end|> is added after the final assistant tag.
    /// No system prompt is included (matches the serve path where system is optional).
    private func renderChatML(newUserText: String) -> String {
        var buf = ""
        for turn in turns {
            // Include only completed turns that have a successful, non-empty reply.
            // Skip: currently-awaiting turn (status == .running), failed turns,
            // and turns with empty responseText (nothing useful to provide as context).
            guard turn.status == .done,
                  !turn.responseText.isEmpty
            else { continue }
            buf += "<|im_start|>user\n\(turn.prompt)<|im_end|>\n"
            buf += "<|im_start|>assistant\n\(turn.responseText)<|im_end|>\n"
        }
        buf += "<|im_start|>user\n\(newUserText)<|im_end|>\n"
        buf += "<|im_start|>assistant\n"
        return buf
    }

    // MARK: Single-mode send

    private func send() {
        let rawUserText = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawUserText.isEmpty, !isRunning else { return }

        // The user's visible bubble shows the RAW text they typed.
        // The prompt sent to the binary is a full ChatML conversation string.
        let turn = ChatTurn(prompt: rawUserText)
        let turnID = turn.id
        turns.append(turn)
        awaitingTurnID = turnID

        composerText = ""

        // Parse settings — safe fallbacks, never crash, never force-unwrap.
        let temperature = Double(tempText) ?? 0.7
        let maxTokens = Int(maxTokensText) ?? 256
        let seed: UInt64? = seedText.isEmpty ? nil : UInt64(seedText)

        // Build the ChatML prompt AFTER the new turn is in the array so it is
        // excluded by the guard (status == .running) in renderChatML.
        let chatMLPrompt = renderChatML(newUserText: rawUserText)

        let cfg = GenConfig(
            modelDir: selectedModel?.path,
            model: selectedModel == nil ? (selectedModelName.isEmpty ? nil : selectedModelName) : nil,
            adapterPath: selectedAdapter?.path,
            prompt: chatMLPrompt,
            maxTokens: maxTokens,
            seed: seed,
            temperature: temperature
        )

        let run = store.runGenerate(cfg)

        // If the launch already failed synchronously, resolve immediately.
        // onChange(of: status) fires only on a transition; a run that is already
        // .failed when assigned produces no observable change, so we handle it here.
        if run.status == .failed {
            resolveTurn(id: turnID, from: run, status: .failed)
        }
    }

    // Resolve the awaiting turn with output from `run`.
    // Shared between the inline synchronous-failure path and handleRunStatusChange.
    private func resolveTurn(id turnID: UUID, from run: LiveRun, status: RunStatus) {
        guard let idx = turns.firstIndex(where: { $0.id == turnID }) else { return }

        // Was this turn stopped cleanly by the user (Stop button)?
        let wasUserStopped = (userStoppedTurnID == turnID)

        // Clear both awaiting and userStoppedTurnID before mutating to avoid re-entry.
        awaitingTurnID = nil
        userStoppedTurnID = nil

        let responseText: String
        if !run.genText.isEmpty {
            // Streaming path: genText holds the complete concatenated token stream.
            responseText = run.genText.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            // Non-streaming fallback (older binary or failure before any token):
            // drop lines beginning with "$ " (command echo).
            let cleaned = run.log
                .filter { !$0.hasPrefix("$ ") }
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if cleaned.isEmpty && status == .failed {
                responseText = wasUserStopped ? "Stopped." : (run.log.last ?? "Generation failed to start.")
            } else {
                responseText = cleaned
            }
        }

        turns[idx].responseText = responseText

        if wasUserStopped {
            // Clean stop: keep whatever partial text arrived, mark done so it is
            // not styled as an error. (If partial text is non-empty it reads
            // naturally as a truncated reply; if empty it reads "Stopped.".)
            turns[idx].status = .done
        } else {
            turns[idx].status = (status == .done) ? .done : .failed
        }

        if status == .done || wasUserStopped {
            turns[idx].tokensPerSecond = run.genTokS
        }
    }

    private func handleRunStatusChange(_ newStatus: RunStatus?) {
        // SINGLE-MODE ONLY. awaitingTurnID is never set in compare mode,
        // so this guard makes the handler inert during A/B runs.
        guard let status = newStatus,
              status == .done || status == .failed,
              let turnID = awaitingTurnID,
              let run = store.liveRun(matching: [.chat])
        else { return }

        resolveTurn(id: turnID, from: run, status: status)
    }

    // MARK: - A/B compare send (sequential: base → adapter via onComplete)

    private func sendCompare() {
        let rawUserText = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawUserText.isEmpty, !isRunning, let adapter = selectedAdapter else { return }

        // Build ChatML using the single-mode turns history for context.
        // Compare pairs are independent A/B experiments and are not fed back
        // into the conversation context (they are evaluation runs, not multi-turn chat).
        let chatMLPrompt = renderChatML(newUserText: rawUserText)

        // Create the pair and set state before launching the run.
        let pair = ComparePair(
            prompt: rawUserText,
            baseLabel: selectedModel?.name ?? (selectedModelName.isEmpty ? "base model" : selectedModelName),
            adapterLabel: adapter.name
        )
        let pairID = pair.id
        comparePairs.append(pair)
        awaitingPairID = pairID
        abPhase = .base

        composerText = ""

        // Parse settings — safe fallbacks.
        let temperature = Double(tempText) ?? 0.7
        let maxTokens = Int(maxTokensText) ?? 256
        let seed: UInt64? = seedText.isEmpty ? nil : UInt64(seedText)

        // Snapshot the adapter path now; the selection could change before the
        // adapter phase starts if the user interacts with the inspector.
        let adapterPath = adapter.path
        let modelDir = selectedModel?.path
        let modelName: String? = selectedModel == nil
            ? (selectedModelName.isEmpty ? nil : selectedModelName)
            : nil

        // Phase 1: base run (adapterPath = nil)
        let baseCfg = GenConfig(
            modelDir: modelDir,
            model: modelName,
            adapterPath: nil,
            prompt: chatMLPrompt,
            maxTokens: maxTokens,
            seed: seed,
            temperature: temperature
        )
        let baseRun = store.runGenerate(baseCfg)

        // onComplete fires on MainActor after AppStore.finish() is called.
        // It sequences the adapter run as Phase 2.
        baseRun.onComplete = { [adapterPath, modelDir, modelName, maxTokens, seed, temperature, chatMLPrompt, pairID] completed in
            // Update base side result.
            guard let idx = self.comparePairs.firstIndex(where: { $0.id == pairID }) else {
                self.abPhase = .idle
                self.awaitingPairID = nil
                return
            }
            self.comparePairs[idx].baseText = completed.genText
                .trimmingCharacters(in: .whitespacesAndNewlines)
            self.comparePairs[idx].baseTokS = completed.genTokS
            self.comparePairs[idx].baseDone = true

            // Only sequence the adapter phase if the base run completed cleanly.
            // A user Stop terminates the process with a non-zero code → status .failed
            // (RunStatus has no distinct .stopped case), so guarding on .done is what
            // prevents a stopped base with partial output from still launching phase 2
            // (codex B1 — the old `failed && genText.isEmpty` check let it through).
            guard completed.status == .done else {
                self.comparePairs[idx].failed = true
                self.comparePairs[idx].adapterDone = true
                self.abPhase = .idle
                self.awaitingPairID = nil
                return
            }

            // Phase 2: adapter run
            self.abPhase = .adapter

            let adpCfg = GenConfig(
                modelDir: modelDir,
                model: modelName,
                adapterPath: adapterPath,
                prompt: chatMLPrompt,
                maxTokens: maxTokens,
                seed: seed,
                temperature: temperature
            )
            let adpRun = self.store.runGenerate(adpCfg)

            adpRun.onComplete = { [pairID] done2 in
                guard let adpIdx = self.comparePairs.firstIndex(where: { $0.id == pairID }) else {
                    self.abPhase = .idle
                    self.awaitingPairID = nil
                    return
                }
                self.comparePairs[adpIdx].adapterText = done2.genText
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                self.comparePairs[adpIdx].adapterTokS = done2.genTokS
                self.comparePairs[adpIdx].adapterDone = true
                if done2.status == .failed && done2.genText.isEmpty {
                    self.comparePairs[adpIdx].failed = true
                }
                self.abPhase = .idle
                self.awaitingPairID = nil
            }

            // Handle synchronous failure of the adapter run launch.
            if adpRun.status == .failed {
                guard let adpIdx = self.comparePairs.firstIndex(where: { $0.id == pairID }) else {
                    self.abPhase = .idle
                    self.awaitingPairID = nil
                    return
                }
                self.comparePairs[adpIdx].adapterDone = true
                self.comparePairs[adpIdx].failed = true
                self.abPhase = .idle
                self.awaitingPairID = nil
            }
        }

        // Handle synchronous failure of the base run launch.
        if baseRun.status == .failed {
            if let idx = comparePairs.firstIndex(where: { $0.id == pairID }) {
                comparePairs[idx].baseDone = true
                comparePairs[idx].adapterDone = true
                comparePairs[idx].failed = true
            }
            abPhase = .idle
            awaitingPairID = nil
        }
    }
}
