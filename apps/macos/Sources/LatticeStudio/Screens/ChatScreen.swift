import AppKit
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
//   mounted, so a turn still resolves if the user navigates away mid-generation.

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

// File-scoped byte formatter for the model-bar meta + hero subline (GB-scale model sizes).
private let chatByteFormatter: ByteCountFormatter = {
    let f = ByteCountFormatter()
    f.allowedUnits = [.useMB, .useGB]
    f.countStyle = .file
    return f
}()

// MARK: - ChatScreen

struct ChatScreen: View {
    @Bindable var store: AppStore

    /// When embedded in the MainColumn (the redesigned shell), the column already supplies the
    /// model header + residency + new-chat/settings affordances, so the in-screen `chatModelBar`
    /// is suppressed to avoid a duplicate model strip. The settings inspector stays reachable from
    /// the column header's gear. Defaults to the standalone layout (own model bar).
    var embedded: Bool = false

    // MARK: Local state (ephemeral; does NOT need to survive navigation)

    // Composer text is ephemeral — clearing on navigation is acceptable
    @State private var composerText: String = ""
    // Explicit user override for the reasoning disclosure, per turn. nil = follow the default
    // (auto-open while streaming, collapsed after). A non-nil value is the user's own toggle and
    // always wins — including mid-stream, so reasoning is foldable while it is still generating.
    @State private var thinkingExpandedOverride: [UUID: Bool] = [:]
    // Turn whose response was just copied — drives the transient "Copied" affordance.
    @State private var copiedTurnID: UUID? = nil

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

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            if !embedded {
                chatModelBar
                Rectangle()
                    .fill(Theme.Palette.hairline)
                    .frame(height: 1)
            }
            if embedded {
                VStack(alignment: .leading, spacing: 2) {
                    Text("CHAT STREAM")
                        .font(Theme.Fonts.sectionLabel)
                        .textCase(.uppercase)
                        .tracking(Theme.Space.labelTracking)
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Text("Resident session")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.vertical, Theme.Space.sm)
                .frame(maxWidth: .infinity, alignment: .leading)
                Theme.Palette.hairline.frame(height: 1)
            }
            transcriptArea
            composerBar
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Theme.Palette.canvas)
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
            // Same prefill-aware split as finalization: while reasoning (thinking-on, no
            // </think> yet) the stream stays in the thinking area instead of flashing in the
            // answer bubble and then vanishing when resolveTurn corrects it.
            let parsed = ChatFinalization.finalize(newText ?? "",
                                                   prefilledOpenThink: store.chatTurns[idx].prefilledOpenThink)
            store.chatTurns[idx].thinkingText = parsed.thinking
            store.chatTurns[idx].responseText = parsed.response
        }
    }

    // MARK: - Model bar  (always-visible "what am I talking to, and is it fast")

    private var chatModelBar: some View {
        HStack(spacing: Theme.Space.md) {
            // Residency dot: generating (accent) · loaded in GPU memory (green) · ready (idle)
            Circle()
                .fill(statusColor)
                .frame(width: 7, height: 7)

            // Model picker — borderless menu so it reads as a title, not a boxed control
            Menu {
                ForEach(chatModels) { m in
                    Button {
                        store.chatSelectedModelName = m.name
                    } label: {
                        if m.name == selectedModel?.name {
                            Label(m.name, systemImage: "checkmark")
                        } else {
                            Text(m.name)
                        }
                    }
                }
            } label: {
                HStack(spacing: 5) {
                    Text(selectedModel?.name ?? "No model")
                        .font(Theme.Fonts.bodyStrong)
                        .foregroundStyle(Theme.Palette.ink)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textTertiary)
                }
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .disabled(chatModels.count <= 1)

            // Quiet meta: params · size · residency — the fit + local story, inline
            if let m = selectedModel {
                Text(metaLine(m))
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)
            }

            Spacer(minLength: Theme.Space.md)

            // Backend — compact CPU/GPU. Full sampling/generation knobs live in the inspector.
            Picker("", selection: Binding(
                get: { store.chatUseGPU ? "GPU" : "CPU" },
                set: { store.chatUseGPU = ($0 == "GPU") }
            )) {
                Text("CPU").tag("CPU")
                Text("GPU").tag("GPU")
            }
            .pickerStyle(.segmented)
            .controlSize(.small)
            .fixedSize()
            .help("CPU bf16 or GPU Metal")

            // New conversation
            Button { newConversation() } label: {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.Palette.textSecondary)
            }
            .buttonStyle(.plain)
            .disabled(store.chatTurns.isEmpty)
            .help("New conversation")
        }
        .padding(.horizontal, Theme.Space.lg)
        .frame(height: 48)
        .background(Theme.Palette.canvas)
    }

    // Residency color for the model-bar dot.
    private var statusColor: Color {
        if isRunning { return Theme.Palette.signal }
        if store.chatUseGPU && store.isChatSessionWarm { return Theme.Palette.success }
        return Theme.Palette.idle
    }

    // params · size · residency — honest, compact. "loaded" only when this exact model is warm.
    private func metaLine(_ model: ModelInfo) -> String {
        var parts: [String] = []
        if let p = model.params { parts.append(p) }
        parts.append(chatByteFormatter.string(fromByteCount: model.sizeBytes))
        if !FileManager.default.fileExists(atPath: model.path.path) {
            parts.append("not found")
        } else if store.chatUseGPU && store.isChatSessionWarm && store.chatWarmModelName == model.name {
            parts.append("loaded")
        }
        return parts.joined(separator: " · ")
    }

    // MARK: - Settings inspector (inline; binds directly to store)

    @ViewBuilder
    private var settingsInspector: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 0) {

                // ── HEADER ──────────────────────────────────────────────────────────

                VStack(alignment: .leading, spacing: 4) {
                    Text("RUN CONTROLS")
                        .font(Theme.Fonts.sectionLabel)
                        .tracking(Theme.Space.labelTracking)
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Text("Decode")
                        .font(Theme.Fonts.inspectorTitle)
                        .foregroundStyle(Theme.Palette.textPrimary)
                }
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.lg)
                .padding(.bottom, Theme.Space.md)
                .frame(maxWidth: .infinity, alignment: .leading)

                // ── TARGET ──────────────────────────────────────────────────────────

                Text("TARGET")
                    .font(Theme.Fonts.sectionLabel)
                    .tracking(Theme.Space.labelTracking)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, Theme.Space.lg)
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

                // ── THINK ──────────────────────────────────────────────────────────

                Text("THINK")
                    .font(Theme.Fonts.sectionLabel)
                    .tracking(Theme.Space.labelTracking)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.lg)
                    .padding(.bottom, Theme.Space.sm)
                    .frame(maxWidth: .infinity, alignment: .leading)

                VStack(spacing: 0) {
                    // Think on/off — when off, a closed <think></think> is injected so the model
                    // answers directly. Same boolean the composer bar "Reasoning" pill drives.
                    HStack {
                        Text("Think")
                            .font(Theme.Fonts.bodyStrong)
                            .foregroundStyle(Theme.Palette.textPrimary)
                        Spacer()
                        Toggle("", isOn: Binding(
                            get: { store.chatEnableThinking },
                            set: { store.chatEnableThinking = $0 }
                        ))
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(Theme.Palette.signal)
                    }
                    .frame(height: Theme.Space.rowHeightComfortable)
                    .padding(.horizontal, Theme.Space.lg)
                    .overlay(alignment: .bottom) {
                        Theme.Palette.hairline.frame(height: 1)
                    }

                    // Thinking budget — cap on reasoning tokens before engine force-injects </think>.
                    // 0 = no cap. Only meaningful when Reasoning is On. Step ±128, range 0–8192.
                    ParamRowStepper(
                        label: "Token budget",
                        value: Binding(
                            get: { Double(store.chatReasoningBudgetText) ?? 0 },
                            set: { store.chatReasoningBudgetText = String(Int($0)) }
                        ),
                        range: 0...8192,
                        step: 128,
                        format: "%.0f"
                    )
                }
                .instrumentPanel()
                .padding(.horizontal, Theme.Space.lg)

                // ── DECODE PARAMS ──────────────────────────────────────────────────

                Text("DECODE")
                    .font(Theme.Fonts.sectionLabel)
                    .tracking(Theme.Space.labelTracking)
                    .foregroundStyle(Theme.Palette.textTertiary)
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
                        label: "Rep penalty",
                        text: Binding(
                            get: { store.chatRepPenaltyText },
                            set: { store.chatRepPenaltyText = $0 }
                        ),
                        placeholder: "1.1"
                    )

                    // Max tokens — real flag: --max-tokens (default 2048 answer budget)
                    ParamRowField(
                        label: "Max tokens",
                        text: Binding(
                            get: { store.chatMaxTokensText },
                            set: { store.chatMaxTokensText = $0 }
                        ),
                        placeholder: "2048"
                    )

                    // Seed — real flag: --seed (empty = non-deterministic)
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

                // ── NOTE CARD ──────────────────────────────────────────────────────
                // True statement: the serve loop resets KV state per turn — there is no
                // cross-turn prefix cache. The GUI shows what the engine actually receives.

                VStack(alignment: .leading, spacing: 6) {
                    Text("NO CROSS-TURN PROMPT CACHE")
                        .font(Theme.Fonts.sectionLabel)
                        .textCase(.uppercase)
                        .tracking(Theme.Space.labelTracking)
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Text("Each turn is explicit. The GUI shows what the Rust engine actually receives.")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(Theme.Space.md)
                .frame(maxWidth: .infinity, alignment: .leading)
                .readoutWellSurface()
                .padding(.horizontal, Theme.Space.lg)
                .padding(.top, Theme.Space.lg)

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
        VStack(spacing: 0) {
            Spacer(minLength: 0)
                .frame(maxHeight: .infinity)
                .layoutPriority(-1)

            VStack(spacing: Theme.Space.lg) {
                Image(systemName: "circle.hexagongrid.fill")
                    .font(.system(size: 32, weight: .regular))
                    .foregroundStyle(Theme.Palette.signal)

                VStack(spacing: 6) {
                    Text(selectedModel?.name ?? "No model loaded")
                        .font(Theme.Fonts.display(22, .bold))
                        .foregroundStyle(Theme.Palette.ink)
                        .multilineTextAlignment(.center)
                    Text(heroSubline)
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Palette.textSecondary)
                        .multilineTextAlignment(.center)
                }

                if selectedModel != nil {
                    exampleChips
                } else {
                    Button("Get a model") { store.getModelsPresented = true }
                        .buttonStyle(LatticePrimaryButtonStyle())
                        .padding(.top, Theme.Space.xs)
                }
            }
            .frame(maxWidth: 480)
            .padding(.horizontal, Theme.Space.xl)

            Spacer(minLength: 0)
                .frame(maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity)
    }

    // Subline under the model name: params · size · "private and offline" — honest, no theater.
    private var heroSubline: String {
        guard let m = selectedModel else {
            return "Add a model to start chatting locally. Private and offline, no account."
        }
        var parts: [String] = []
        if let p = m.params { parts.append(p) }
        parts.append(chatByteFormatter.string(fromByteCount: m.sizeBytes))
        parts.append("private and offline")
        return parts.joined(separator: " · ")
    }

    // Starter prompts — tapping one fills the composer (does NOT auto-send). Lowers the
    // activation energy of an empty chat without committing the user to anything.
    private let examplePrompts: [String] = [
        "Explain how attention works, simply.",
        "Write a Python function to dedupe a list.",
        "Draft a polite follow-up email.",
        "Give me three ideas for a weekend project."
    ]

    private var exampleChips: some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(), spacing: Theme.Space.sm),
                GridItem(.flexible(), spacing: Theme.Space.sm)
            ],
            spacing: Theme.Space.sm
        ) {
            ForEach(examplePrompts, id: \.self) { prompt in
                Button { composerText = prompt } label: {
                    Text(prompt)
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textSecondary)
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .fill(Theme.Palette.surfaceRaised)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                                .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.top, Theme.Space.sm)
    }

    // MARK: - Turn bubbles (single mode)

    @ViewBuilder
    private func turnView(_ turn: ChatTurn) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.lg) {
            userBlock(turn)
            assistantBlock(turn)
        }
    }

    // MARK: User block — "YOU" label over an indigo-tinted prompt block that hugs its content

    private func userBlock(_ turn: ChatTurn) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("YOU")
                .font(Theme.Fonts.sectionLabel)
                .tracking(Theme.Space.labelTracking)
                .foregroundStyle(Theme.Palette.textTertiary)
            // The bubble hugs the prompt and leaves a right margin (mockup) rather than spanning
            // the full column; long prompts grow wide but never reach the edge.
            HStack(spacing: 0) {
                Text(turn.prompt)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                    .multilineTextAlignment(.leading)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                            .fill(Theme.Palette.signal.opacity(0.10))
                            .overlay(
                                RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                                    .strokeBorder(Theme.Palette.signal.opacity(0.22), lineWidth: 1)
                            )
                    )
                Spacer(minLength: 48)
            }
        }
    }

    // MARK: Assistant block — "▲ MODEL" label, reasoning fold, bubble-less answer, metrics + actions

    @ViewBuilder
    private func assistantBlock(_ turn: ChatTurn) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            HStack(spacing: 7) {
                LatticeMark()
                    .stroke(Theme.Palette.signal,
                            style: StrokeStyle(lineWidth: 1.3, lineCap: .round, lineJoin: .round))
                    .frame(width: 13, height: 13)
                Text((turn.modelName ?? selectedModel?.name ?? "assistant").uppercased())
                    .font(Theme.Fonts.sectionLabel)
                    .tracking(Theme.Space.labelTracking)
                    .foregroundStyle(Theme.Palette.textSecondary)
            }

            if !turn.thinkingText.isEmpty {
                reasoningDisclosure(turn)
            }

            assistantBody(turn)
        }
    }

    // Reasoning fold — chevron + "REASONING" + italic first-line preview when collapsed; full
    // trace when expanded. Auto-opens while reasoning streams, collapses once the answer lands;
    // an explicit user toggle always wins (foldable mid-stream).
    @ViewBuilder
    private func reasoningDisclosure(_ turn: ChatTurn) -> some View {
        let streamingOpen = (turn.status == .running && turn.responseText.isEmpty)
        let expanded = thinkingExpandedOverride[turn.id] ?? streamingOpen
        let preview = turn.thinkingText
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespaces)

        VStack(alignment: .leading, spacing: 8) {
            Button {
                thinkingExpandedOverride[turn.id] = !expanded
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: expanded ? "arrowtriangle.down.fill" : "arrowtriangle.right.fill")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Text("REASONING")
                        .font(Theme.Fonts.sectionLabel)
                        .tracking(Theme.Space.labelTracking)
                        .foregroundStyle(Theme.Palette.textTertiary)
                    if !expanded {
                        Text(preview)
                            .font(Theme.Fonts.caption)
                            .italic()
                            .foregroundStyle(Theme.Palette.textTertiary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if expanded {
                Text(turn.thinkingText)
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                .fill(Theme.Palette.panel)
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                        .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
                )
        )
    }

    // Answer body. Markdown sits directly on the canvas (no bubble) to match the mockup; the
    // streaming/failed states keep a light treatment so they read as transient, not the answer.
    @ViewBuilder
    private func assistantBody(_ turn: ChatTurn) -> some View {
        if turn.status == .running && turn.responseText.isEmpty {
            HStack(spacing: Theme.Space.xs) {
                TypingDots()
                let liveText = store.liveRun(matching: [.chat])?.genText ?? ""
                Text(liveText.isEmpty ? "Loading model…" : "Thinking…")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textSecondary)
            }
            .padding(.vertical, 2)

        } else if turn.status == .failed && turn.responseText.isEmpty {
            Text(turn.errorMessage ?? "Generation failed.")
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.error)
                .multilineTextAlignment(.leading)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                        .fill(Theme.Palette.error.opacity(0.08))
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                                .strokeBorder(Theme.Palette.error.opacity(0.25), lineWidth: 1)
                        )
                )
            if let cfg = turn.retryConfig { retryButton(turn, cfg) }

        } else {
            MarkdownText(text: turn.responseText)
                .frame(maxWidth: Theme.Space.chatReadingWidth, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .leading)

            if turn.status == .failed, let reason = turn.errorMessage {
                Text(reason)
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.error)
            }

            // Hardware label while generating, before tok/s is known.
            if turn.tokensPerSecond == nil, let label = turn.inferenceLabel, turn.status == .running {
                Text(label)
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)
            }

            if turn.tokensPerSecond != nil { metricsRow(turn) }
            if !turn.responseText.isEmpty { actionRow(turn) }
            if turn.status == .failed, let cfg = turn.retryConfig { retryButton(turn, cfg) }
        }
    }

    // Honest per-turn telemetry as pills: hardware tag (plain) · tok/s (accent) · prompt · reasoning
    // (only when > 0) · reply · total. No "cached" pill — there is no cross-turn prefix cache.
    @ViewBuilder
    private func metricsRow(_ turn: ChatTurn) -> some View {
        if let tps = turn.tokensPerSecond {
            HStack(spacing: 6) {
                if let label = turn.inferenceLabel {
                    metricPill(hardwareTag(label), highlighted: false)
                }
                metricPill(tpsText(tps), highlighted: true)
                if let pt = turn.promptTokens {
                    metricPill("\(pt.formatted()) prompt", highlighted: false)
                }
                if let rt = turn.reasoningTokens, rt > 0 {
                    metricPill("\(rt.formatted()) reasoning", highlighted: false)
                }
                if let rp = turn.responseTokens {
                    metricPill("\(rp.formatted()) reply", highlighted: false)
                }
                if let ms = turn.totalMs {
                    metricPill(String(format: "%.1f s", ms / 1000.0), highlighted: false)
                }
                Spacer(minLength: 0)
            }
            .padding(.top, 2)
        }
    }

    // Outlined capsules — a thin border on a near-flush fill, as the mockup. tok/s is the one
    // accented reading (indigo text + indigo-tinted border); every other metric stays neutral so
    // the throughput number is the single thing the eye lands on.
    private func metricPill(_ text: String, highlighted: Bool) -> some View {
        Text(text)
            .font(.system(size: 11, weight: highlighted ? .semibold : .regular, design: .monospaced))
            .foregroundStyle(highlighted ? Theme.Palette.signal : Theme.Palette.textSecondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(Capsule(style: .continuous).fill(Theme.Palette.hoverOverlay))
            .overlay(
                Capsule(style: .continuous).strokeBorder(
                    highlighted ? Theme.Palette.signal.opacity(0.35) : Theme.Palette.borderStandard,
                    lineWidth: 1)
            )
    }

    // Copy + regenerate, icon-only in subtle bordered squares (matches mockup).
    private func actionRow(_ turn: ChatTurn) -> some View {
        HStack(spacing: 6) {
            let copied = (copiedTurnID == turn.id)
            transcriptIconButton(copied ? "checkmark" : "doc.on.doc",
                                 active: copied,
                                 help: "Copy response") {
                copyResponse(turn.responseText, turnID: turn.id)
            }
            if let cfg = turn.retryConfig {
                transcriptIconButton("arrow.counterclockwise",
                                     active: false,
                                     help: "Regenerate") {
                    retryTurn(turn: turn, config: cfg)
                }
                .disabled(isRunning)
            }
            Spacer(minLength: 0)
        }
        .padding(.top, 2)
    }

    private func transcriptIconButton(_ symbol: String, active: Bool, help: String,
                                      action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(active ? Theme.Palette.signal : Theme.Palette.textTertiary)
                .frame(width: 28, height: 26)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                        .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }

    private func retryButton(_ turn: ChatTurn, _ cfg: ChatGenConfig) -> some View {
        Button { retryTurn(turn: turn, config: cfg) } label: {
            Label("Retry", systemImage: "arrow.clockwise")
                .font(Theme.Fonts.caption)
        }
        .buttonStyle(LatticeSecondaryButtonStyle())
        .disabled(isRunning)
    }

    private func hardwareTag(_ label: String) -> String {
        if label.hasPrefix("GPU") { return "GPU" }
        if label.hasPrefix("CPU") { return "CPU" }
        return label
    }

    private func tpsText(_ tps: Double) -> String {
        tps >= 10 ? String(format: "%.0f tok/s", tps) : String(format: "%.1f tok/s", tps)
    }

    // MARK: - Composer

    private var composerBar: some View {
        VStack(spacing: 8) {
            composerField
            composerFooter
        }
        .frame(maxWidth: Theme.Space.chatMaxWidth)
        .frame(maxWidth: .infinity)
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.md)
        .background(Theme.Palette.canvas)
    }

    // The field holds the text area plus an inline control cluster — reasoning toggle, CPU/GPU
    // backend, and Send — so the primary chat controls live where the message is typed (mockup).
    private var composerField: some View {
        HStack(alignment: .bottom, spacing: 10) {
            ZStack(alignment: .topLeading) {
                if composerText.isEmpty {
                    Text(composerPlaceholder)
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .padding(.horizontal, 5)
                        .padding(.top, 7)
                        .allowsHitTesting(false)
                        .lineLimit(1)
                }
                // TextEditor wraps NSTextView in an NSScrollView with always-on scrollers;
                // `.scrollIndicators(.never)` makes it read as an input area, not a document.
                TextEditor(text: $composerText)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                    .scrollContentBackground(.hidden)
                    .scrollIndicators(.never)
                    .background(.clear)
                    .frame(minHeight: 30, maxHeight: 132)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 1)
            }

            HStack(spacing: 8) {
                reasoningToggle
                backendToggle
                actionButton
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(Theme.Palette.surfaceRaised)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
        )
    }

    private var composerPlaceholder: String {
        if let name = selectedModel?.name { return "Message \(name)…" }
        return "Message…"
    }

    // Reasoning on/off — when off, a closed <think></think> is injected so the model answers
    // directly. Lives in the composer (mockup) instead of buried in the settings inspector.
    private var reasoningToggle: some View {
        let on = store.chatEnableThinking
        return Button { store.chatEnableThinking.toggle() } label: {
            HStack(spacing: 5) {
                Circle()
                    .fill(on ? Theme.Palette.signal : Theme.Palette.idle)
                    .frame(width: 6, height: 6)
                Text("Reasoning")
                    .font(Theme.Fonts.controlText)
                    .foregroundStyle(on ? Theme.Palette.signal : Theme.Palette.textTertiary)
            }
            .padding(.horizontal, 10)
            .frame(height: 30)
            .background(
                Capsule(style: .continuous)
                    .fill(on ? Theme.Palette.signal.opacity(0.12) : Theme.Palette.hoverOverlay)
            )
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(on ? Theme.Palette.signal.opacity(0.30) : Theme.Palette.borderStandard,
                                  lineWidth: 1)
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help(on ? "Reasoning on — the model thinks before answering"
                 : "Reasoning off — the model answers directly")
    }

    // CPU bf16 vs GPU Metal — custom segmented control matched to the composer's visual language.
    private var backendToggle: some View {
        HStack(spacing: 2) {
            backendSegment("CPU", isGPU: false)
            backendSegment("GPU", isGPU: true)
        }
        .padding(2)
        .frame(height: 30)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                .fill(Theme.Palette.wellSink)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
        )
    }

    private func backendSegment(_ title: String, isGPU: Bool) -> some View {
        let selected = (store.chatUseGPU == isGPU)
        return Button { store.chatUseGPU = isGPU } label: {
            Text(title)
                .font(Theme.Fonts.controlText)
                .foregroundStyle(selected ? Theme.Palette.textPrimary : Theme.Palette.textTertiary)
                .padding(.horizontal, 11)
                .frame(maxHeight: .infinity)
                .background(
                    RoundedRectangle(cornerRadius: Theme.Radius.badge, style: .continuous)
                        .fill(selected ? Theme.Palette.surfaceHover : .clear)
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(isGPU ? "GPU Metal (bf16 + Q4)" : "CPU bf16")
    }

    @ViewBuilder
    private var actionButton: some View {
        if isRunning {
            Button {
                store.chatUserStoppedTurnID = store.chatAwaitingTurnID
                store.stopRun()
            } label: {
                Image(systemName: "stop.fill")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.Palette.crimson)
                    .frame(width: 36, height: 36)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                            .fill(Theme.Palette.crimson.opacity(0.12))
                    )
            }
            .buttonStyle(.plain)
            .help("Stop generation")
        } else {
            Button(action: { send() }) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(canSend ? Theme.Palette.onAccent : Theme.Palette.onAccent.opacity(0.7))
                    .frame(width: 36, height: 36)
                    .background(
                        // Indigo identity even when disabled (mockup): a dimmed accent, not the
                        // near-black wellSink, so the send affordance always reads as the CTA.
                        RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                            .fill(canSend ? Theme.Palette.signal : Theme.Palette.signal.opacity(0.5))
                    )
            }
            .buttonStyle(.plain)
            .disabled(!canSend)
            .keyboardShortcut(.return, modifiers: .command)
            .help("Send  ⌘↵")
        }
    }

    // Privacy reassurance + the real keyboard affordance. The shortcut text states the actual
    // behavior (⌘↵ sends, plain ↵ inserts a newline) rather than a not-yet-wired Return-to-send.
    private var composerFooter: some View {
        HStack(spacing: 0) {
            HStack(spacing: 5) {
                Image(systemName: "lock.fill")
                    .font(.system(size: 10, weight: .regular))
                Text("Private · runs entirely on this Mac")
                    .font(Theme.Fonts.caption)
            }
            .foregroundStyle(Theme.Palette.textTertiary)
            Spacer()
            Text("⌘↵ send · ↵ newline")
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textTertiary)
        }
        .padding(.horizontal, 4)
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
        thinkingExpandedOverride.removeAll()
        copiedTurnID = nil
        if isRunning { store.stopRun() }
    }

    /// Copy the answer body to the clipboard and flash a transient "Copied" affordance.
    private func copyResponse(_ text: String, turnID: UUID) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copiedTurnID = turnID
        Task {
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            if copiedTurnID == turnID { copiedTurnID = nil }
        }
    }

    /// Compose the per-turn metrics readout. Returns nil until the turn is finalized (tok/s known),
    /// so a half-empty line never flashes mid-stream. "cached" is always 0: there is no cross-turn
    /// prefix cache — the serve loop resets KV state every request — and that is shown honestly.
    private func statsLine(_ turn: ChatTurn) -> String? {
        guard let tps = turn.tokensPerSecond else { return nil }
        var parts: [String] = [String(format: "%.1f tok/s", tps)]
        if let pt = turn.promptTokens { parts.append("\(pt) in") }
        parts.append("\(turn.cachedInputTokens) cached")
        if let rt = turn.reasoningTokens, rt > 0 { parts.append("\(rt) reasoning") }
        if let rp = turn.responseTokens { parts.append("\(rp) reply") }
        if let ms = turn.totalMs { parts.append(String(format: "%.1fs total", ms / 1000.0)) }
        if let label = turn.inferenceLabel { parts.append(label) }
        return parts.joined(separator: "  ·  ")
    }

    /// Build ChatML from completed turns (single-mode history as context).
    private func renderChatML(newUserText: String, enableThinking: Bool) -> String {
        var buf = ""
        for turn in store.chatTurns {
            guard turn.status == .done, !turn.responseText.isEmpty else { continue }
            buf += "<|im_start|>user\n\(turn.prompt)<|im_end|>\n"
            buf += "<|im_start|>assistant\n\(turn.responseText)<|im_end|>\n"
        }
        buf += "<|im_start|>user\n\(newUserText)<|im_end|>\n"
        if enableThinking {
            // Open think block = the official template's enable_thinking=true path
            // (chat_template.jinja:152). A thinking checkpoint continues inside the block
            // and closes with </think> before the answer; renderChatML's split keys on that
            // boundary. A non-thinking checkpoint ignores the prefix and answers directly.
            buf += "<|im_start|>assistant\n<think>\n"
        } else {
            // Closed empty think block = the official template's enable_thinking=false path;
            // the model emits the answer directly with no reasoning.
            buf += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        }
        return buf
    }

    // Reasoning split/finalization lives in ChatFinalization (pure, unit-tested).

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
        let reasoningBudget: Int? = {
            guard store.chatEnableThinking else { return nil }   // only caps when reasoning is on
            let n = Int(store.chatReasoningBudgetText) ?? 0
            return n > 0 ? n : nil
        }()
        let chatMLPrompt = renderChatML(newUserText: rawUserText, enableThinking: store.chatEnableThinking)
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
        turn.modelName = selectedModel?.name
        turn.retryConfig = retryCfg
        turn.inferenceLabel = inferenceLabel
        // Thinking-on prefills an open <think> into the prompt (renderChatML), so a truncated
        // reasoning stream is tagless. Snapshot the mode so finalization treats such a stream
        // as unfinished reasoning rather than the answer (prevents history poisoning).
        turn.prefilledOpenThink = store.chatEnableThinking
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
            reasoningBudget: reasoningBudget,
            useGPU: useGPU
        )

        // GPU Metal path: use the persistent serve session (model stays warm between turns).
        // CPU path: keep using the one-shot generate_lora subprocess (unchanged).
        let run = useGPU ? store.runChatGPU(cfg) : store.runGenerate(cfg)

        // Resolve via the run's own completion hook so the turn lands even if the user navigates
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
        let thinkingText: String
        if !run.genText.isEmpty {
            // Prefill-aware finalization. When thinking was prefilled (open <think> in the
            // prompt) and the stream has no </think>, the whole stream is unfinished reasoning,
            // not the answer — keep it as thinkingText and leave responseText empty so the turn
            // is skipped from history (renderChatML gates history on non-empty responseText).
            let parsed = ChatFinalization.finalize(run.genText,
                                                   prefilledOpenThink: store.chatTurns[idx].prefilledOpenThink)
            thinkingText = parsed.thinking
            responseText = parsed.response
        } else {
            thinkingText = ""
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
        store.chatTurns[idx].thinkingText = thinkingText

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
            store.chatTurns[idx].promptTokens = run.genPromptTokens
            store.chatTurns[idx].totalMs = run.genTotalMs
            // Reasoning/response split. Totals are anchored to the engine's exact gen_tokens; the
            // reasoning portion comes from the app-side count up to the </think> boundary.
            // When no </think> was generated, prefilledOpenThink disambiguates whether the whole
            // output was unfinished reasoning (thinking-on, truncated) or a direct answer (off).
            if run.sawThinkClose {
                let reasoning = run.genReasoningTokens
                store.chatTurns[idx].reasoningTokens = reasoning
                if let total = run.genTokensTotal {
                    store.chatTurns[idx].responseTokens = max(0, total - reasoning)
                }
            } else if store.chatTurns[idx].prefilledOpenThink {
                store.chatTurns[idx].reasoningTokens = run.genTokensTotal ?? run.genReasoningTokens
                store.chatTurns[idx].responseTokens = 0
            } else {
                store.chatTurns[idx].reasoningTokens = 0
                store.chatTurns[idx].responseTokens = run.genTokensTotal ?? run.genReasoningTokens
            }
        }
    }

}
