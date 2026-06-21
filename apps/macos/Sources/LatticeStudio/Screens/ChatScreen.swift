import SwiftUI

// MARK: - 04 CHAT
//
// Layout (three zones):
//   • CONFIG STRIP  — top OpaquePanel: MODEL picker, ADAPTER picker, FaderToggle (BASE/+ADAPTER),
//                     max-tokens slider, temperature slider, seed field.
//   • TRANSCRIPT    — scrollable VStack of ChatTurn bubbles. Each turn shows the user prompt
//                     (right-aligned) then the model response on an opaque surface. While a run
//                     is in-flight the response slot shows an animated "▍ generating…" placeholder
//                     with a GatePill(.run). On completion the collected log text fills it in.
//   • COMPOSER      — multiline TextField + teal Send button + ⌘↵ KeyCapChip hint.
//
// Generation model:
//   generate_lora prints output ATOMICALLY. Output arrives as .status lines into store.liveRun.log.
//   A "message" lifecycle is: launch via store.runGenerate(GenConfig) → status == .running →
//   collect store.liveRun.log → status == .done → copy cleaned output into the pending turn.
//
//   Cleaning rule: drop lines that start with "$ " (the command echo). The remaining lines are
//   either loader status ("Adapter: …", "LoRA active …") or the generated text itself. We keep
//   all non-echo lines and present them mono, separated by newlines, so the user sees any
//   informational loader messages as subtle context before the generated text.
//
// Completion detection:
//   .onChange(of: store.liveRun?.status) fires when the subprocess exits and the store sets
//   status to .done or .failed. We track `awaitingTurnID: UUID?` in @State; only the turn
//   whose id matches gets updated, preventing stale binds across multiple generations.
//
// A/B usage pattern:
//   The user sends a prompt, reads the BASE response, flips the fader to +ADAPTER, re-sends
//   the SAME prompt. Two adjacent turns in the transcript provide the visual diff. We do NOT
//   auto-run both variants — manual flip+resend is the v1 A/B story. This matches the reality
//   that generate_lora runs synchronously and there is no lockstep-streaming path.

// MARK: - Local domain model

private struct ChatTurn: Identifiable {
    enum TurnStatus { case running, done, failed }

    let id = UUID()
    let prompt: String
    let variantLabel: String   // "BASE" or "+ADAPTER <adapter-name>"
    var responseText: String = ""
    var status: TurnStatus = .running
    var tokensPerSecond: Double? = nil
}

// MARK: - ChatScreen

struct ChatScreen: View {
    @Bindable var store: AppStore

    // MARK: Local state

    @State private var turns: [ChatTurn] = []
    @State private var composerText: String = ""
    @State private var awaitingTurnID: UUID?

    // Config pickers
    @State private var selectedModelName: String = ""
    @State private var selectedAdapterName: String = "none"
    @State private var useAdapter: Bool = false

    // Sampling params
    @State private var maxTokens: Double = 64
    @State private var temperature: Double = 0.7
    @State private var seedText: String = ""

    // MARK: Derived helpers

    private var chatModels: [ModelInfo] {
        store.models.filter { !$0.isEmbedding }
    }

    private var chatModelNames: [String] {
        chatModels.map(\.name)
    }

    private var selectedModel: ModelInfo? {
        chatModels.first { $0.name == selectedModelName }
            ?? store.targetModel
    }

    private var adapterOptions: [String] {
        let adapters = selectedModel?.adapters.map(\.name) ?? []
        return ["none"] + adapters
    }

    private var selectedAdapter: AdapterInfo? {
        guard useAdapter, selectedAdapterName != "none" else { return nil }
        return selectedModel?.adapters.first { $0.name == selectedAdapterName }
    }

    private var hasAdapters: Bool {
        guard let m = selectedModel else { return false }
        return !m.adapters.isEmpty
    }

    private var isRunning: Bool {
        store.liveRun(matching: [.chat])?.status == .running
    }

    private var canSend: Bool {
        !composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !isRunning
    }

    private var variantLabel: String {
        if let adapter = selectedAdapter {
            return "+ADAPTER \(adapter.name)"
        }
        return "BASE"
    }

    // MARK: Body

    var body: some View {
        ScreenScaffold(
            screen: .chat,
            subtitle: subtitleText,
            trailing: { liveStatusBadge }
        ) {
            VStack(spacing: 0) {
                configStrip
                Divider().background(Theme.Palette.hairline)
                transcriptArea
                Divider().background(Theme.Palette.hairline)
                composerBar
            }
        }
        .onAppear { applyDefaults() }
        .onChange(of: store.models) { _, _ in applyDefaults() }
        .onChange(of: selectedModelName) { _, _ in
            // Reset adapter selection when model changes.
            selectedAdapterName = "none"
            useAdapter = false
        }
        .onChange(of: store.liveRun?.status) { _, newStatus in
            handleRunStatusChange(newStatus)
        }
        .onChange(of: store.liveRun?.genText) { _, newText in
            // Live streaming: push incremental token text into the awaiting turn
            // so the response renders token-by-token as the engine streams.
            // genText is already cumulative, so assign (not append).
            guard let turnID = awaitingTurnID,
                  store.liveRun?.kind == .chat,
                  let idx = turns.firstIndex(where: { $0.id == turnID })
            else { return }
            turns[idx].responseText = newText ?? ""
        }
        .onChange(of: store.liveRun?.log.count) { _, _ in
            // Belt-and-suspenders: if status fires before log is flushed,
            // a log.count change while awaiting provides a secondary wake.
            // No action here; status onChange is the actual handler.
        }
    }

    // MARK: Subtitle + live badge

    private var subtitleText: String {
        if let m = selectedModel {
            return "\(m.name)\(hasAdapters ? " · \(m.adapters.count) adapter\(m.adapters.count == 1 ? "" : "s")" : " · no adapters")"
        }
        return "sample-test base vs +adapter"
    }

    @ViewBuilder
    private var liveStatusBadge: some View {
        if isRunning {
            GatePill(.run, label: "GENERATING")
        }
    }

    // MARK: Config strip

    private var configStrip: some View {
        OpaquePanel {
            VStack(spacing: 0) {
                modelAdapterRow
                faderRow
                Divider().background(Theme.Palette.hairline)
                samplingRows
            }
        }
    }

    private var modelAdapterRow: some View {
        HStack(spacing: 0) {
            // MODEL picker — occupies left half
            HStack {
                Text("MODEL")
                    .instrumentLabel()
                Spacer()
                if chatModelNames.isEmpty {
                    Text("no models")
                        .font(Theme.Fonts.readout)
                        .foregroundStyle(Theme.Palette.inkDim)
                } else {
                    Picker("", selection: $selectedModelName) {
                        ForEach(chatModelNames, id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .font(Theme.Fonts.readout)
                }
            }
            .frame(height: Theme.Space.rowHeightComfortable)
            .padding(.horizontal, Theme.Space.lg)
            .overlay(alignment: .bottom) {
                Theme.Palette.hairline.frame(height: 1)
            }
            .overlay(alignment: .trailing) {
                Theme.Palette.hairline.frame(width: 1)
            }

            // ADAPTER picker — occupies right half
            HStack {
                Text("ADAPTER")
                    .instrumentLabel()
                Spacer()
                Picker("", selection: $selectedAdapterName) {
                    ForEach(adapterOptions, id: \.self) { name in
                        Text(name).tag(name)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .font(Theme.Fonts.readout)
                .disabled(!hasAdapters)
            }
            .frame(height: Theme.Space.rowHeightComfortable)
            .padding(.horizontal, Theme.Space.lg)
            .overlay(alignment: .bottom) {
                Theme.Palette.hairline.frame(height: 1)
            }
        }
    }

    private var faderRow: some View {
        VStack(spacing: 0) {
            FaderToggle(
                labelA: "BASE",
                labelB: hasAdapters && selectedAdapterName != "none"
                    ? "+ADAPTER \(selectedAdapterName)"
                    : "+ADAPTER",
                isOnB: $useAdapter
            )
            .disabled(!hasAdapters)
            .opacity(hasAdapters ? 1 : 0.4)

            if !hasAdapters {
                Text("load a model with adapters to enable A/B comparison")
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.bottom, Theme.Space.sm)
            }
        }
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    private var samplingRows: some View {
        HStack(spacing: 0) {
            ParamRowSlider(
                label: "MAX TOKENS",
                value: $maxTokens,
                range: 1...512,
                step: 1,
                format: "%.0f"
            )
            .overlay(alignment: .trailing) {
                Theme.Palette.hairline.frame(width: 1)
            }

            ParamRowSlider(
                label: "TEMP",
                value: $temperature,
                range: 0...1.5,
                format: "%.2f"
            )
            .overlay(alignment: .trailing) {
                Theme.Palette.hairline.frame(width: 1)
            }

            // SEED field — optional
            HStack {
                Text("SEED")
                    .instrumentLabel()
                Spacer()
                TextField("random", text: $seedText)
                    .font(Theme.Fonts.readout)
                    .foregroundStyle(Theme.Palette.ink)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 100)
                    .textFieldStyle(.plain)
                    .monospacedDigit()
            }
            .frame(height: Theme.Space.rowHeightComfortable)
            .padding(.horizontal, Theme.Space.lg)
        }
    }

    // MARK: Transcript

    @ViewBuilder
    private var transcriptArea: some View {
        if turns.isEmpty {
            emptyTranscript
        } else {
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    LazyVStack(alignment: .leading, spacing: Theme.Space.md) {
                        ForEach(turns) { turn in
                            turnView(turn)
                                .id(turn.id)
                        }
                    }
                    .padding(Theme.Space.lg)
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

    private var emptyTranscript: some View {
        VStack(spacing: Theme.Space.sm) {
            Text("ask the model something")
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.inkDim)
            Text("flip the fader to A/B the adapter")
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim.opacity(0.7))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func turnView(_ turn: ChatTurn) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            // User prompt — right-aligned with "YOU" label
            HStack(alignment: .top, spacing: Theme.Space.sm) {
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("YOU")
                        .instrumentLabel()
                    Text(turn.prompt)
                        .font(Theme.Fonts.mono(12))
                        .foregroundStyle(Theme.Palette.ink)
                        .monospacedDigit()
                        .multilineTextAlignment(.trailing)
                        .textSelection(.enabled)
                }
            }

            // Model response — left-aligned, labeled with variant, on opaque surface
            VStack(alignment: .leading, spacing: Theme.Space.xs) {
                // Header: variant label + gate pill
                HStack(spacing: Theme.Space.sm) {
                    Text(turn.variantLabel)
                        .instrumentLabel()
                    switch turn.status {
                    case .running:
                        GatePill(.run, label: "generating")
                    case .done:
                        GatePill(.pass, label: "done")
                    case .failed:
                        GatePill(.fail, label: "failed")
                    }
                    Spacer()
                }
                .padding(.horizontal, Theme.Space.md)
                .padding(.top, Theme.Space.sm)

                Divider()
                    .background(Theme.Palette.hairline)

                // Response text — mono, opaque surface (numbers never touch glass law)
                if turn.status == .running && turn.responseText.isEmpty {
                    generatingPlaceholder
                        .padding(.horizontal, Theme.Space.md)
                        .padding(.bottom, Theme.Space.sm)
                } else {
                    Text(turn.responseText.isEmpty ? "(no output)" : turn.responseText)
                        .font(Theme.Fonts.mono(12))
                        .foregroundStyle(
                            turn.status == .failed
                                ? Theme.Palette.crimson
                                : Theme.Palette.ink
                        )
                        .monospacedDigit()
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, Theme.Space.md)
                        .padding(.bottom, turn.tokensPerSecond != nil ? Theme.Space.xs : Theme.Space.sm)

                    if let tps = turn.tokensPerSecond {
                        Text(String(format: "%.1f tok/s", tps))
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.signal)
                            .monospacedDigit()
                            .frame(maxWidth: .infinity, alignment: .trailing)
                            .padding(.horizontal, Theme.Space.md)
                            .padding(.bottom, Theme.Space.sm)
                    }
                }
            }
            .instrumentPanel()
        }
    }

    private var generatingPlaceholder: some View {
        HStack(spacing: Theme.Space.xs) {
            Text("▍")
                .font(Theme.Fonts.mono(12))
                .foregroundStyle(Theme.Palette.signal)
            Text("generating…")
                .font(Theme.Fonts.mono(12))
                .foregroundStyle(Theme.Palette.inkDim)
        }
        .padding(.vertical, Theme.Space.xs)
    }

    // MARK: Composer

    private var composerBar: some View {
        HStack(alignment: .bottom, spacing: Theme.Space.sm) {
            TextEditor(text: $composerText)
                .font(Theme.Fonts.mono(13))
                .foregroundStyle(Theme.Palette.ink)
                .scrollContentBackground(.hidden)
                .background(Theme.Palette.wellSink)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                        .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
                )
                .frame(minHeight: 56, maxHeight: 120)

            VStack(alignment: .trailing, spacing: Theme.Space.xs) {
                // Send button — the one teal CTA on this screen
                Button(action: send) {
                    Text("Send")
                        .font(Theme.Fonts.display(13, .semibold))
                        .foregroundStyle(Theme.Palette.canvas)
                        .padding(.horizontal, Theme.Space.lg)
                        .padding(.vertical, Theme.Space.sm)
                        .background(
                            canSend
                                ? Theme.Palette.signal
                                : Theme.Palette.signal.opacity(0.35)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(!canSend)
                .keyboardShortcut(.return, modifiers: .command)

                // Keyboard hint
                HStack(spacing: 4) {
                    KeyCapChip("⌘↵")
                    Text("to send")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
            }
        }
        .padding(Theme.Space.md)
        .instrumentPanel()
    }

    // MARK: Actions

    private func applyDefaults() {
        // Set selectedModelName to the current target model if not already set or stale.
        if selectedModelName.isEmpty || chatModels.first(where: { $0.name == selectedModelName }) == nil {
            selectedModelName = store.targetModel?.name ?? chatModelNames.first ?? ""
        }
    }

    private func send() {
        let prompt = composerText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !isRunning else { return }

        // Build the turn before launch so it appears immediately.
        let turn = ChatTurn(
            prompt: prompt,
            variantLabel: variantLabel
        )
        let turnID = turn.id
        turns.append(turn)
        awaitingTurnID = turnID

        composerText = ""

        // Build GenConfig.
        // Use modelDir (URL) so the binary gets the exact path rather than a name lookup.
        let seed: UInt64? = seedText.isEmpty ? nil : UInt64(seedText)
        let cfg = GenConfig(
            modelDir: selectedModel?.path,
            model: selectedModel == nil ? (selectedModelName.isEmpty ? nil : selectedModelName) : nil,
            adapterPath: selectedAdapter?.path,
            prompt: prompt,
            maxTokens: Int(maxTokens),
            seed: seed,
            temperature: temperature
        )

        let run = store.runGenerate(cfg)

        // If the launch already failed synchronously, resolve the turn immediately.
        // onChange(of: status) fires only on a transition; a run that is already
        // .failed when assigned produces no observable change, so we must handle
        // it here before returning.
        if run.status == .failed {
            resolveTurn(id: turnID, from: run, status: .failed)
        }
    }

    // Resolve the awaiting turn with the output from `run`.
    // Shared between the inline synchronous-failure path and handleRunStatusChange.
    private func resolveTurn(id turnID: UUID, from run: LiveRun, status: RunStatus) {
        guard let idx = turns.firstIndex(where: { $0.id == turnID }) else { return }

        // Clear awaiting before mutating so we do not re-enter on re-render.
        awaitingTurnID = nil

        let responseText: String
        if !run.genText.isEmpty {
            // Streaming path: genText holds the complete concatenated token stream.
            // Trim trailing whitespace/newlines that the engine may emit at end-of-sequence.
            responseText = run.genText.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            // Non-streaming fallback (older binary without --json, or failure before any token):
            // extract the generated text by dropping lines beginning with "$ " (command echo).
            let cleaned = run.log
                .filter { !$0.hasPrefix("$ ") }
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if cleaned.isEmpty && status == .failed {
                // Use the last log line as an error hint; fall back to a generic message.
                responseText = run.log.last ?? "generation failed to start"
            } else {
                responseText = cleaned
            }
        }

        turns[idx].responseText = responseText
        turns[idx].status = (status == .done) ? .done : .failed
        if status == .done {
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
