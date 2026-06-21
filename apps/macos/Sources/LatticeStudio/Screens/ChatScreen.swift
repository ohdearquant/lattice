import SwiftUI

// MARK: - 04 CHAT
//
// Redesigned as a friendly, minimal chat interface with progressive disclosure.
//
// Layout (three zones):
//   • TOP BAR      — slim 44pt strip: model Menu (left).
//   • TRANSCRIPT   — scrollable column of chat bubbles, max-width 920pt, centered.
//   • COMPOSER     — rounded TextEditor + Send CTA (bottom).
//
// All knobs (adapter, temperature, max tokens, seed) are hidden in a toggleable
// right-side .inspector sidebar, shown via the window-toolbar toggle button or ⌘\.
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
// A/B comparison: dropped for this pass. Adapter is now a single advanced setting
// in the settings inspector sidebar. One active adapter, no fader/variant label.

// MARK: - Local domain model

private struct ChatTurn: Identifiable {
    enum TurnStatus { case running, done, failed }

    let id = UUID()
    let prompt: String
    var responseText: String = ""
    var status: TurnStatus = .running
    var tokensPerSecond: Double? = nil
}

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
    let adapterOptions: [String]
    @Binding var selectedAdapterName: String
    @Binding var tempText: String
    @Binding var maxTokensText: String
    @Binding var seedText: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                Text("Settings")
                    .font(Theme.Fonts.sectionLabel)
                    .tracking(0.5)
                    .textCase(.uppercase)
                    .foregroundStyle(Theme.Palette.textSecondary)

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
            Spacer(minLength: 0)
        }
        .padding(Theme.Space.lg)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Theme.Palette.panel)
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

    @State private var turns: [ChatTurn] = []
    @State private var composerText: String = ""
    @State private var awaitingTurnID: UUID?

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

    private var isRunning: Bool {
        store.liveRun(matching: [.chat])?.status == .running
    }

    private var canSend: Bool {
        !composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !isRunning
    }

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            topBar
            Rectangle()
                .fill(Theme.Palette.hairline)
                .frame(height: 1)
            transcriptArea
            composerBar
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.Palette.canvas)
        .inspector(isPresented: $store.inspectorPresented) {
            ChatInspector(
                adapterOptions: adapterOptions,
                selectedAdapterName: $selectedAdapterName,
                tempText: $tempText,
                maxTokensText: $maxTokensText,
                seedText: $seedText
            )
            .inspectorColumnWidth(min: 260, ideal: 300, max: 360)
        }
        .onAppear { applyDefaults() }
        .onChange(of: store.models) { _, _ in applyDefaults() }
        .onChange(of: selectedModelName) { _, _ in
            // Reset adapter when model changes.
            selectedAdapterName = "none"
        }
        .onChange(of: store.liveRun?.status) { _, newStatus in
            handleRunStatusChange(newStatus)
        }
        .onChange(of: store.liveRun?.genText) { _, newText in
            // Live streaming: genText is cumulative — assign, not append.
            guard let turnID = awaitingTurnID,
                  store.liveRun?.kind == .chat,
                  let idx = turns.firstIndex(where: { $0.id == turnID })
            else { return }
            turns[idx].responseText = newText ?? ""
        }
        .onChange(of: store.liveRun?.log.count) { _, _ in
            // Belt-and-suspenders: log.count changes are noted; status onChange is the handler.
        }
    }

    // MARK: Top bar

    private var topBar: some View {
        HStack(spacing: Theme.Space.sm) {
            // Model menu — left side, no caps label
            if chatModels.isEmpty {
                Text("No models")
                    .font(Theme.Fonts.bodyStrong)
                    .foregroundStyle(Theme.Palette.textSecondary)
            } else {
                Menu {
                    ForEach(chatModels, id: \.name) { model in
                        Button(model.name) {
                            selectedModelName = model.name
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(selectedModelName.isEmpty ? "Select model" : selectedModelName)
                            .font(Theme.Fonts.bodyStrong)
                            .foregroundStyle(Theme.Palette.ink)
                        Image(systemName: "chevron.down")
                            .font(.system(size: 10, weight: .medium))
                            .foregroundStyle(Theme.Palette.textSecondary)
                    }
                }
                .buttonStyle(.plain)
            }

            Spacer()
        }
        .frame(height: 44)
        .padding(.horizontal, Theme.Space.lg)
        .background(Theme.Palette.canvas)
    }

    // MARK: Transcript

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

    // MARK: Turn bubbles

    @ViewBuilder
    private func turnView(_ turn: ChatTurn) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            // User bubble — right-aligned
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
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Theme.Palette.selectionFill)
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .strokeBorder(Theme.Palette.selectionBorder, lineWidth: 1)
                            )
                    )
            }

            // Assistant bubble — left-aligned
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
        RoundedRectangle(cornerRadius: 14, style: .continuous)
            .fill(Theme.Palette.surfaceRaised)
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
            )
    }

    // MARK: Composer

    private var composerBar: some View {
        VStack(spacing: 0) {
            Rectangle()
                .fill(Theme.Palette.hairline)
                .frame(height: 1)

            HStack(alignment: .bottom, spacing: Theme.Space.sm) {
                // Multiline input
                ZStack(alignment: .topLeading) {
                    if composerText.isEmpty {
                        Text("Message…")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.textTertiary)
                            .padding(.horizontal, 10)
                            .padding(.top, 9)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: $composerText)
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Palette.ink)
                        .scrollContentBackground(.hidden)
                        .background(.clear)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 4)
                }
                .frame(minHeight: 52, maxHeight: 140)
                .background(Theme.Palette.surfaceRaised)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                        .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
                )

                // Send / Stop button stack
                VStack(alignment: .trailing, spacing: Theme.Space.xs) {
                    if isRunning {
                        // Stop button: visible while generation is in flight.
                        Button {
                            userStoppedTurnID = awaitingTurnID
                            store.stopRun()
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "stop.fill")
                                    .font(.system(size: 11, weight: .medium))
                                Text("Stop")
                                    .font(Theme.Fonts.bodyStrong)
                            }
                            .foregroundStyle(Theme.Palette.crimson)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .overlay(
                                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                    .strokeBorder(Theme.Palette.crimson.opacity(0.5), lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                    } else {
                        Button(action: send) {
                            Text("Send")
                        }
                        .buttonStyle(LatticePrimaryButtonStyle())
                        .disabled(!canSend)
                        .keyboardShortcut(.return, modifiers: .command)

                        HStack(spacing: 4) {
                            KeyCapChip("⌘↵")
                            Text("send")
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.textTertiary)
                        }
                    }
                }
            }
            .padding(Theme.Space.lg)
            .background(Theme.Palette.canvas)
        }
    }

    // MARK: Actions

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
        guard let status = newStatus,
              status == .done || status == .failed,
              let turnID = awaitingTurnID,
              let run = store.liveRun(matching: [.chat])
        else { return }

        resolveTurn(id: turnID, from: run, status: status)
    }
}
