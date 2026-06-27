import SwiftUI

// MARK: - 02 CHAT
//
// Single-mode chat interface.
//
// Layout (inside ScreenScaffold):
//   • TRANSCRIPT — scrollable column of ChatTurn bubbles, max-width 920pt.
//   • COMPOSER   — rounded TextEditor with Send/Stop as a 30×30 in-composer button.
//
// State ownership:
//   All transcript state lives in AppStore so it survives NavigationSplitView teardown.
//   ChatScreen reads/writes via @Bindable var store.
//
// Generation model:
//   generate_lora prints output ATOMICALLY. A "message" lifecycle is:
//   launch via store.runGenerate(GenConfig) → status == .running →
//   collect store.liveRun.genText (cumulative) → status == .done → copy into pending turn.
//
// Completion detection:
//   Each run carries an onComplete hook (set in send()/retryTurn()) that
//   AppStore.finish() fires when the subprocess exits — independent of ChatScreen being
//   mounted, so a turn still resolves if Ocean navigates away mid-generation.

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
            try? await Task.sleep(nanoseconds: 50_000_000)
            phase = 1
        }
    }
}

// MARK: - ChatScreen

struct ChatScreen: View {
    @Bindable var store: AppStore

    // MARK: Local state (ephemeral; does NOT need to survive navigation)

    // Composer text is ephemeral — clearing on navigation is acceptable
    @State private var composerText: String = ""

    // MARK: Store-backed derived helpers

    // CPU mode (generate_lora): bf16 only — generate_lora loads from_safetensors.
    // GPU mode (chat_metal): bf16 + q4 — Metal path supports both formats.
    // Embedding models are never generative; excluded in both modes.
    private var chatModels: [ModelInfo] {
        if store.chatUseGPU {
            store.models.filter { $0.format == .bf16 || $0.format == .q4 }
        } else {
            store.models.filter { $0.format == .bf16 }
        }
    }

    private var selectedModel: ModelInfo? {
        if let m = chatModels.first(where: { $0.name == store.chatSelectedModelName }) { return m }
        return chatModels.first
    }

    // isRunning: true while a single-mode run is in-flight.
    private var isRunning: Bool {
        store.liveRun(matching: [.chat])?.status == .running
    }

    private var canSend: Bool {
        !composerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !isRunning
    }

    // MARK: Subtitle (model + residency status; honest)

    private var subtitle: String {
        guard let model = selectedModel else { return "no models found" }
        // "warm"  — GPU serve session is resident and the model is loaded in GPU memory.
        // "ready" — model directory exists on disk but no warm session (CPU mode, or first GPU send).
        // "not found" — model path is missing from disk.
        let diskStatus: String
        if FileManager.default.fileExists(atPath: model.path.path) {
            diskStatus = (store.chatUseGPU && store.isChatSessionWarm) ? "warm" : "ready"
        } else {
            diskStatus = "not found"
        }
        return "\(model.name) · \(diskStatus)"
    }

    // MARK: Body

    var body: some View {
        ScreenScaffold(screen: .chat, subtitle: subtitle) {
            VStack(spacing: 0) {
                transcriptArea
                composerBar
            }
        }
        .inspector(isPresented: $store.inspectorPresented) {
            settingsInspector
                // Fixed width, not a resizable range. The inspector holds greedy full-width
                // content (status pills with Spacers, the Load button); auto-sizing to that
                // content within a min/ideal/max range has no single stable width, so the
                // column oscillated every time layout re-ran (the 1 Hz memory tick re-triggered
                // it). A fixed column cannot resize itself.
                .inspectorColumnWidth(320)
        }
        .onAppear { applyDefaults() }
        .onChange(of: store.models) { _, _ in applyDefaults() }
        .onChange(of: store.chatSelectedModelName) { _, _ in
            // Picking a different model resets the sampling knobs to that model's recommended
            // defaults (from its generation_config.json). Manual edits hold until the next switch.
            if let model = selectedModel { applySamplingDefaults(for: model) }
        }
        .onChange(of: store.chatUseGPU) { _, _ in
            // Backend toggle changes the eligible model set (CPU drops Q4) — re-validate
            // the selection so the picker never shows a model that isn't in chatModels.
            applyDefaults()
        }
        .onChange(of: store.liveRun?.genText) { _, newText in
            // Route streaming tokens to the awaiting turn.
            guard store.liveRun?.kind == .chat else { return }
            guard let turnID = store.chatAwaitingTurnID,
                  let idx = store.chatTurns.firstIndex(where: { $0.id == turnID })
            else { return }
            store.chatTurns[idx].responseText = newText ?? ""
        }
    }

    // MARK: - Settings inspector (inline; binds directly to store)

    @ViewBuilder
    private var settingsInspector: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 0) {

                // ── TARGET ──────────────────────────────────────────────────────────

                Text("Target")
                    .instrumentLabel()
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.lg)
                    .padding(.bottom, Theme.Space.sm)
                    .frame(maxWidth: .infinity, alignment: .leading)

                VStack(spacing: 0) {
                    // Model picker — menu (may have many options; segmented would overflow)
                    ParamRowMenu(
                        label: "Model",
                        options: chatModels.map(\.name),
                        selection: Binding(
                            get: { store.chatSelectedModelName },
                            set: { store.chatSelectedModelName = $0 }
                        )
                    )
                    .disabled(chatModels.count <= 1)

                    // Backend — segmented CPU/GPU (two options; segmented is correct)
                    ParamRowPicker(
                        label: "Backend",
                        options: ["CPU bf16", "GPU Metal"],
                        selection: Binding(
                            get: { store.chatUseGPU ? "GPU Metal" : "CPU bf16" },
                            set: { store.chatUseGPU = ($0 == "GPU Metal") }
                        )
                    )

                    // Disk + load state — honest. States exactly which model is resident in
                    // memory, distinct from which is merely selected on disk.
                    if let model = selectedModel {
                        let exists = FileManager.default.fileExists(atPath: model.path.path)
                        let warmName = store.chatWarmModelName
                        // `warmName` is set the instant the serve process spawns, which is BEFORE
                        // the weights finish loading. Gate the resident state on the loading flag so
                        // "LOADED" never shows while the model is still streaming off disk.
                        let modelLoading = store.isChatModelLoading
                        let selectedLoaded = (warmName == model.name) && !modelLoading
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 6) {
                                GatePill(exists ? .pass : .fail, label: exists ? "ON DISK" : "NOT FOUND")
                                if modelLoading {
                                    GatePill(.run, label: "LOADING MODEL")
                                } else if selectedLoaded {
                                    GatePill(.pass, label: "LOADED")
                                } else if warmName != nil {
                                    GatePill(.warn, label: "WILL RELOAD")
                                } else {
                                    Text("not loaded")
                                        .font(Theme.Fonts.cell)
                                        .foregroundStyle(Theme.Palette.inkDim)
                                }
                                if let liveRun = store.liveRun(matching: [.chat]),
                                   liveRun.status == .running {
                                    // Empty genText = prefill (TTFT) phase, not model load — label it
                                    // honestly so a slow prefill doesn't read as a stuck model load.
                                    GatePill(.run, label: liveRun.genText.isEmpty ? "PREFILL" : "GEN")
                                }
                                Spacer()
                            }
                            // Explicit "what is in GPU memory right now" line — suppressed mid-load
                            // since the model is not actually resident until `ready` arrives.
                            if let warmName, !modelLoading {
                                HStack(spacing: 4) {
                                    Text("IN MEMORY")
                                        .font(Theme.Fonts.cell)
                                        .foregroundStyle(Theme.Palette.inkDim)
                                    Text(warmName)
                                        .font(Theme.Fonts.readout)
                                        .foregroundStyle(selectedLoaded ? Theme.Palette.signal : Theme.Palette.ink)
                                    Spacer()
                                }
                            }
                        }
                        .frame(minHeight: Theme.Space.rowHeight)
                        .padding(.vertical, Theme.Space.xs)
                        .padding(.horizontal, Theme.Space.lg)
                        .overlay(alignment: .bottom) {
                            Theme.Palette.hairline.frame(height: 1)
                        }

                        // Explicit preload control — warms the serve session so the first message
                        // doesn't pay the multi-second cold model load. GPU mode only; the CPU path
                        // is a one-shot subprocess with no persistent session to warm.
                        if store.chatUseGPU {
                            let canLoad = !modelLoading && !selectedLoaded && !isRunning
                            Button {
                                if let cfg = warmGenConfig() { store.warmChatSession(cfg) }
                            } label: {
                                HStack(spacing: 6) {
                                    if modelLoading {
                                        ProgressView().controlSize(.small)
                                        Text("Loading model…")
                                    } else if selectedLoaded {
                                        Image(systemName: "checkmark.circle.fill")
                                        Text("Model loaded")
                                    } else {
                                        Image(systemName: "arrow.down.circle")
                                        Text("Load model")
                                    }
                                    Spacer()
                                }
                                .font(Theme.Fonts.readout)
                                .foregroundStyle(canLoad ? Theme.Palette.signal : Theme.Palette.inkDim)
                                .padding(.vertical, Theme.Space.sm)
                                .padding(.horizontal, Theme.Space.md)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .contentShape(Rectangle())
                                .background(
                                    RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                        .fill(canLoad ? Theme.Palette.signalGlow : Theme.Palette.wellSink)
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                        .strokeBorder(canLoad ? Theme.Palette.signal : Theme.Palette.hairline, lineWidth: 1)
                                )
                            }
                            .buttonStyle(.plain)
                            .disabled(!canLoad)
                            .padding(.vertical, Theme.Space.sm)
                            .padding(.horizontal, Theme.Space.lg)
                            .help(selectedLoaded ? "Model is resident in GPU memory"
                                : "Preload the model into GPU memory before sending")
                        }
                    }
                }
                .instrumentPanel()
                .padding(.horizontal, Theme.Space.lg)

                // ── SAMPLING ────────────────────────────────────────────────────────

                Text("Sampling")
                    .instrumentLabel()
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.lg)
                    .padding(.bottom, Theme.Space.sm)
                    .frame(maxWidth: .infinity, alignment: .leading)

                VStack(spacing: 0) {
                    // Temperature — real flag: --temperature (default 0.7)
                    ParamRowField(
                        label: "Temperature",
                        text: Binding(
                            get: { store.chatTempText },
                            set: { store.chatTempText = $0 }
                        ),
                        placeholder: "0.7"
                    )

                    // Top-k — real flag: --top-k (default 50)
                    ParamRowField(
                        label: "Top-k",
                        text: Binding(
                            get: { store.chatTopKText },
                            set: { store.chatTopKText = $0 }
                        ),
                        placeholder: "50"
                    )

                    // Top-p — real flag: --top-p (default 0.9)
                    ParamRowField(
                        label: "Top-p",
                        text: Binding(
                            get: { store.chatTopPText },
                            set: { store.chatTopPText = $0 }
                        ),
                        placeholder: "0.9"
                    )

                    // Repetition penalty — real flag: --repetition-penalty (default 1.1)
                    ParamRowField(
                        label: "Rep. penalty",
                        text: Binding(
                            get: { store.chatRepPenaltyText },
                            set: { store.chatRepPenaltyText = $0 }
                        ),
                        placeholder: "1.1"
                    )
                }
                .instrumentPanel()
                .padding(.horizontal, Theme.Space.lg)

                // ── GENERATION ──────────────────────────────────────────────────────

                Text("Generation")
                    .instrumentLabel()
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.lg)
                    .padding(.bottom, Theme.Space.sm)
                    .frame(maxWidth: .infinity, alignment: .leading)

                VStack(spacing: 0) {
                    // Max tokens — real flag: --max-tokens (default 256)
                    ParamRowField(
                        label: "Max tokens",
                        text: Binding(
                            get: { store.chatMaxTokensText },
                            set: { store.chatMaxTokensText = $0 }
                        ),
                        placeholder: "256"
                    )

                    // Seed — real flag: --seed (nil = non-deterministic)
                    ParamRowField(
                        label: "Seed",
                        text: Binding(
                            get: { store.chatSeedText },
                            set: { store.chatSeedText = $0 }
                        ),
                        placeholder: "random"
                    )
                }
                .instrumentPanel()
                .padding(.horizontal, Theme.Space.lg)

                // ── ACTIONS ─────────────────────────────────────────────────────────

                Button("New conversation") {
                    newConversation()
                }
                .buttonStyle(LatticeSecondaryButtonStyle())
                .disabled(store.chatTurns.isEmpty)
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.lg)
                .padding(.bottom, Theme.Space.lg)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(Theme.Palette.panel)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // MARK: - Single-mode transcript

    @ViewBuilder
    private var transcriptArea: some View {
        if store.chatTurns.isEmpty {
            emptyState
        } else {
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    LazyVStack(alignment: .leading, spacing: Theme.Space.lg) {
                        ForEach(store.chatTurns) { turn in
                            turnView(turn)
                                .id(turn.id)
                        }
                    }
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.vertical, Theme.Space.xl)
                    .frame(maxWidth: Theme.Space.chatMaxWidth)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: store.chatTurns.count) { _, _ in
                    if let last = store.chatTurns.last {
                        withAnimation(.easeOut(duration: Theme.Motion.focus)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: store.chatTurns.last?.responseText) { _, _ in
                    if let last = store.chatTurns.last {
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
                    .multilineTextAlignment(.leading)
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
                        // Typing indicator: distinguish "loading model" from "generating"
                        HStack(spacing: Theme.Space.xs) {
                            TypingDots()
                            let liveText = store.liveRun(matching: [.chat])?.genText ?? ""
                            Text(liveText.isEmpty ? "Loading model…" : "Thinking…")
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.textSecondary)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(assistantBubbleBackground)

                    } else if turn.status == .failed && turn.responseText.isEmpty {
                        // Failed with NO output — show the engine reason, never silent "(no output)"
                        let reason = turn.errorMessage ?? "Generation failed."
                        Text(reason)
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.error)
                            .multilineTextAlignment(.leading)
                            .textSelection(.enabled)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .background(assistantBubbleBackground)

                        // Retry — re-launches the identical GenConfig
                        if let cfg = turn.retryConfig {
                            Button {
                                retryTurn(turn: turn, config: cfg)
                            } label: {
                                Label("Retry", systemImage: "arrow.clockwise")
                                    .font(Theme.Fonts.caption)
                            }
                            .buttonStyle(LatticeSecondaryButtonStyle())
                            .disabled(isRunning)
                            .padding(.leading, 4)
                        }

                    } else {
                        // Normal response — rendered as Markdown (headings, lists, code, **bold**).
                        // Red is for errors only; MarkdownText uses ink throughout.
                        MarkdownText(text: turn.responseText)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .background(assistantBubbleBackground)

                        // If the run failed but we got partial text, show the reason beneath
                        if turn.status == .failed, let reason = turn.errorMessage {
                            Text(reason)
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.error)
                                .padding(.leading, 4)
                        }

                        // tok/s + honest hardware label — muted, small
                        // The label was snapshotted at send time: "GPU Metal bf16", "CPU bf16", etc.
                        // It is never updated retroactively so what launched is always what shows.
                        if let tps = turn.tokensPerSecond {
                            let labelStr = turn.inferenceLabel.map { " · \($0)" } ?? ""
                            Text(String(format: "%.1f tok/s\(labelStr)", tps))
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.textTertiary)
                                .monospacedDigit()
                                .padding(.leading, 4)
                        } else if let label = turn.inferenceLabel, turn.status == .running {
                            // Show label while generating (before tok/s is known)
                            Text(label)
                                .font(Theme.Fonts.caption)
                                .foregroundStyle(Theme.Palette.textTertiary)
                                .padding(.leading, 4)
                        }

                        // Retry button for failed turns that do have partial text
                        if turn.status == .failed, let cfg = turn.retryConfig {
                            Button {
                                retryTurn(turn: turn, config: cfg)
                            } label: {
                                Label("Retry", systemImage: "arrow.clockwise")
                                    .font(Theme.Fonts.caption)
                            }
                            .buttonStyle(LatticeSecondaryButtonStyle())
                            .disabled(isRunning)
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

    // MARK: - Composer

    private var composerBar: some View {
        VStack(spacing: 0) {
            Rectangle()
                .fill(Theme.Palette.hairline)
                .frame(height: 1)

            HStack {
                composerField
                    .frame(maxWidth: Theme.Space.chatMaxWidth)
                    .frame(maxWidth: .infinity)
            }
            .padding(Theme.Space.lg)
            .background(Theme.Palette.canvas)
        }
    }

    private var composerField: some View {
        ZStack(alignment: .topLeading) {
            // Placeholder — shown only when composer is empty
            if composerText.isEmpty {
                Text("Message…")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, 10)
                    .padding(.top, 9)
                    .allowsHitTesting(false)
            }
            // TextEditor on macOS wraps NSTextView inside NSScrollView which shows
            // always-on scrollers by default. `.scrollIndicators(.never)` suppresses
            // them so the field looks like a native input area rather than a document.
            TextEditor(text: $composerText)
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.ink)
                .scrollContentBackground(.hidden)
                .scrollIndicators(.never)
                .background(.clear)
                .padding(.horizontal, 6)
                .padding(.vertical, 4)
                .padding(.trailing, 40)
        }
        // Auto-grow: starts at 44pt (single line), expands up to 160pt when content fills.
        // The outer frame clamps max height — the inner TextEditor fills it naturally.
        .frame(minHeight: 44, maxHeight: 160)
        .fixedSize(horizontal: false, vertical: true)
        .background(Theme.Palette.surfaceRaised)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
        )
        .overlay(alignment: .bottomTrailing) {
            actionButton
                .padding(7)
        }
    }

    @ViewBuilder
    private var actionButton: some View {
        if isRunning {
            Button {
                store.chatUserStoppedTurnID = store.chatAwaitingTurnID
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
            Button(action: {
                send()
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
        if store.chatSelectedModelName.isEmpty ||
           chatModels.first(where: { $0.name == store.chatSelectedModelName }) == nil {
            let target = store.targetModel
            if let t = target, t.format == .bf16 {
                store.chatSelectedModelName = t.name
            } else {
                store.chatSelectedModelName = chatModels.first?.name ?? ""
            }
        }
    }

    /// Load a model's recommended sampling defaults from its `generation_config.json` whenever
    /// the selected model changes. Qwen3.6 ships temperature 1.0 / top-k 20 / top-p 0.95; its
    /// config omits repetition_penalty, so 1.0 (off) is the correct default — not the 0.0 a
    /// hand-typed value might suggest. Manual edits persist until the model is switched again.
    private func applySamplingDefaults(for model: ModelInfo) {
        // Always-reset knobs the config never carries.
        store.chatRepPenaltyText = "1.0"

        // Resolve generation_config.json: the model dir first, then the bf16 sibling for Q4
        // models that don't ship the file alongside the weights.
        let base = model.name
            .replacingOccurrences(of: "-q4", with: "", options: .caseInsensitive)
            .replacingOccurrences(of: "-quarot", with: "", options: .caseInsensitive)
        var dirs = [model.path]
        if base != model.name {
            dirs.append(LatticeBridge.modelCacheDir.appendingPathComponent(base, isDirectory: true))
        }
        let configURL = dirs
            .map { $0.appendingPathComponent("generation_config.json") }
            .first { FileManager.default.fileExists(atPath: $0.path) }

        guard let url = configURL,
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }

        if let t = json["temperature"] as? Double { store.chatTempText = trimNumber(t) }
        if let k = json["top_k"] as? Int { store.chatTopKText = String(k) }
        if let p = json["top_p"] as? Double { store.chatTopPText = trimNumber(p) }
    }

    /// Build a prompt-less GenConfig for the selected GPU model, used to warm the serve session
    /// from the Load button. Mirrors `send()`'s model + tokenizer resolution (Q4 tokenizer lives
    /// in the bf16 sibling). Returns nil when GPU mode is off or no model is selected.
    private func warmGenConfig() -> GenConfig? {
        guard store.chatUseGPU, let model = selectedModel else { return nil }
        let tokenizerDirURL: URL? = {
            guard model.format == .q4 else { return nil }
            let baseName = model.name
                .replacingOccurrences(of: "-q4", with: "", options: .caseInsensitive)
                .replacingOccurrences(of: "-quarot", with: "", options: .caseInsensitive)
            let siblingURL = LatticeBridge.modelCacheDir.appendingPathComponent(baseName, isDirectory: true)
            let tokenizerJSON = siblingURL.appendingPathComponent("tokenizer.json")
            return FileManager.default.fileExists(atPath: tokenizerJSON.path) ? siblingURL : nil
        }()
        return GenConfig(
            modelDir: model.path,
            model: nil,
            tokenizerDir: tokenizerDirURL,
            adapterPath: nil,
            prompt: "",
            maxTokens: 1,
            seed: nil,
            temperature: 0.7,
            topK: 50,
            topP: 0.9,
            repetitionPenalty: 1.0,
            useGPU: true
        )
    }

    /// Format a sampling value without float noise: 1.0 -> "1.0", 0.95 -> "0.95", 0.7 -> "0.7".
    private func trimNumber(_ v: Double) -> String {
        var s = String(format: "%.4f", v)
        while s.hasSuffix("0") && !s.hasSuffix(".0") { s.removeLast() }
        return s
    }

    /// Clear conversation transcript while preserving model/adapter/settings selections.
    private func newConversation() {
        store.chatTurns = []
        store.chatAwaitingTurnID = nil
        store.chatUserStoppedTurnID = nil
        if isRunning { store.stopRun() }
    }

    /// Build ChatML from completed turns (single-mode history as context).
    private func renderChatML(newUserText: String) -> String {
        var buf = ""
        for turn in store.chatTurns {
            guard turn.status == .done, !turn.responseText.isEmpty else { continue }
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

        let temperature = Double(store.chatTempText) ?? 0.7
        let maxTokens = Int(store.chatMaxTokensText) ?? 256
        let seed: UInt64? = store.chatSeedText.isEmpty ? nil : UInt64(store.chatSeedText)
        let topK = Int(store.chatTopKText) ?? 50
        let topP = Double(store.chatTopPText) ?? 0.9
        let repetitionPenalty = Double(store.chatRepPenaltyText) ?? 1.1
        let chatMLPrompt = renderChatML(newUserText: rawUserText)
        let useGPU = store.chatUseGPU

        // For Q4 models on the GPU path: the tokenizer lives in the bf16 sibling directory.
        // The GPU binary (chat_metal) requires --tokenizer-dir when the model dir has no
        // tokenizer.json. CPU path (generate_lora) ignores this flag.
        let tokenizerDirURL: URL? = {
            guard useGPU, let model = selectedModel, model.format == .q4 else { return nil }
            // The bf16 sibling is the same name without a "-q4" / "-quarot" suffix.
            let modelName = model.name
            let baseName = modelName
                .replacingOccurrences(of: "-q4", with: "", options: .caseInsensitive)
                .replacingOccurrences(of: "-quarot", with: "", options: .caseInsensitive)
            let siblingURL = LatticeBridge.modelCacheDir.appendingPathComponent(baseName, isDirectory: true)
            let tokenizerJSON = siblingURL.appendingPathComponent("tokenizer.json")
            return FileManager.default.fileExists(atPath: tokenizerJSON.path) ? siblingURL : nil
        }()

        // Honest hardware label — snapshotted at send time, never changes after dispatch.
        // "GPU Metal q4" / "GPU Metal bf16" / "CPU bf16"
        let inferenceLabel: String = {
            if useGPU {
                let fmtTag = (selectedModel?.format == .q4) ? "q4" : "bf16"
                return "GPU Metal \(fmtTag)"
            } else {
                return "CPU bf16"
            }
        }()

        // Snapshot GenConfig for Retry (plain types; no URL reference needed)
        let retryCfg = ChatGenConfig(
            modelDirPath: selectedModel?.path.path,
            model: selectedModel == nil ? (store.chatSelectedModelName.isEmpty ? nil : store.chatSelectedModelName) : nil,
            tokenizerDirPath: tokenizerDirURL?.path,
            adapterFilePath: nil,
            prompt: chatMLPrompt,
            maxTokens: maxTokens,
            seed: seed,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            useGPU: useGPU
        )

        var turn = ChatTurn(prompt: rawUserText)
        turn.retryConfig = retryCfg
        turn.inferenceLabel = inferenceLabel
        let turnID = turn.id
        store.chatTurns.append(turn)
        store.chatAwaitingTurnID = turnID

        composerText = ""

        let cfg = GenConfig(
            modelDir: selectedModel?.path,
            model: selectedModel == nil ? (store.chatSelectedModelName.isEmpty ? nil : store.chatSelectedModelName) : nil,
            tokenizerDir: tokenizerDirURL,
            adapterPath: nil,
            prompt: chatMLPrompt,
            maxTokens: maxTokens,
            seed: seed,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            useGPU: useGPU
        )

        // GPU Metal path: use the persistent serve session (model stays warm between turns).
        // CPU path: keep using the one-shot generate_lora subprocess (unchanged).
        let run = useGPU ? store.runChatGPU(cfg) : store.runGenerate(cfg)

        // Resolve via the run's own completion hook so the turn lands even if Ocean navigates
        // away from Chat before generation finishes (the .onChange handlers only fire while
        // ChatScreen is mounted). It also fires if another screen's launch() supersedes this
        // run — finish() runs onComplete with a failed status, so the turn shows a real reason
        // instead of a permanent spinner.
        run.onComplete = { completed in
            self.resolveTurn(id: turnID, from: completed, status: completed.status)
        }

        // Synchronous launch failure: no process started, so onComplete never fires.
        if run.status == .failed {
            resolveTurn(id: turnID, from: run, status: .failed)
        }
    }

    /// Retry a failed/empty turn with its original GenConfig.
    private func retryTurn(turn: ChatTurn, config: ChatGenConfig) {
        guard !isRunning else { return }
        guard let idx = store.chatTurns.firstIndex(where: { $0.id == turn.id }) else { return }

        // Reset in-place to running state
        store.chatTurns[idx].responseText = ""
        store.chatTurns[idx].status = .running
        store.chatTurns[idx].errorMessage = nil
        store.chatTurns[idx].tokensPerSecond = nil
        store.chatAwaitingTurnID = turn.id

        let modelDir: URL? = config.modelDirPath.flatMap { URL(fileURLWithPath: $0) }
        let tokenizerDir: URL? = config.tokenizerDirPath.flatMap { URL(fileURLWithPath: $0) }
        let adapterPath: URL? = config.adapterFilePath.flatMap { URL(fileURLWithPath: $0) }
        let cfg = GenConfig(
            modelDir: modelDir,
            model: config.model,
            tokenizerDir: tokenizerDir,
            adapterPath: adapterPath,
            prompt: config.prompt,
            maxTokens: config.maxTokens,
            seed: config.seed,
            temperature: config.temperature,
            topK: config.topK,
            topP: config.topP,
            repetitionPenalty: config.repetitionPenalty,
            useGPU: config.useGPU
        )

        // GPU Metal path: use the persistent serve session (model stays warm between turns).
        // CPU path: keep using the one-shot generate_lora subprocess (unchanged).
        let run = config.useGPU ? store.runChatGPU(cfg) : store.runGenerate(cfg)
        run.onComplete = { [turnID = turn.id] completed in
            self.resolveTurn(id: turnID, from: completed, status: completed.status)
        }
        if run.status == .failed {
            resolveTurn(id: turn.id, from: run, status: .failed)
        }
    }

    private func resolveTurn(id turnID: UUID, from run: LiveRun, status: RunStatus) {
        guard let idx = store.chatTurns.firstIndex(where: { $0.id == turnID }) else { return }
        // Idempotent: once a turn is resolved to .done/.failed, ignore repeat calls. The run's
        // onComplete hook is authoritative; a synchronous-failure check can also land here.
        guard store.chatTurns[idx].status == .running else { return }

        let wasUserStopped = (store.chatUserStoppedTurnID == turnID)
        store.chatAwaitingTurnID = nil
        store.chatUserStoppedTurnID = nil

        let responseText: String
        if !run.genText.isEmpty {
            responseText = run.genText.trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            let cleaned = run.log
                .filter { !$0.hasPrefix("$ ") }
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if cleaned.isEmpty && status == .failed {
                responseText = wasUserStopped ? "Stopped." : ""
            } else {
                responseText = cleaned
            }
        }

        store.chatTurns[idx].responseText = responseText

        if wasUserStopped {
            store.chatTurns[idx].status = .done
        } else {
            store.chatTurns[idx].status = (status == .done) ? .done : .failed
        }

        // Surface engine error for failed turns with no output.
        if store.chatTurns[idx].status == .failed && responseText.isEmpty {
            let errorLine = run.log
                .filter { !$0.hasPrefix("$ ") }
                .last { line in
                    let lo = line.lowercased()
                    return lo.contains("error") || lo.contains("nan") ||
                           lo.contains("unrecognized") || lo.contains("failed") ||
                           lo.contains("panic")
                }
                ?? run.log.filter { !$0.hasPrefix("$ ") }.last
                ?? "Generation failed to start."
            store.chatTurns[idx].errorMessage = errorLine
        }

        if status == .done || wasUserStopped {
            store.chatTurns[idx].tokensPerSecond = run.genTokS
        }
    }

}
