import SwiftUI

// MARK: - Main column
//
// The right two-thirds of the shell: a header naming the working model and its residency, a verb
// tab bar (Chat · Serve · Quantize · Train · Inspect), and the active verb's surface beneath. The
// working model is chosen in the sidebar; every tab here acts on that one model, so the header
// stays put as the user moves between verbs.

struct MainColumn: View {
    @Bindable var store: AppStore
    @State private var headerHovered = false

    var body: some View {
        VStack(spacing: 0) {
            header
            verbTabBar
            Rectangle().fill(Theme.Palette.hairline).frame(height: 1)
            tabBody
            if let cmd = cliCommand {
                CLIFooter(command: cmd, caption: cliCaption)
            }
        }
        .background(Theme.Palette.canvas)
    }

    // MARK: CLI footer helpers
    //
    // Derive the `lattice ...` command that mirrors the current GUI configuration so the footer
    // can show exactly what the engine would receive. Honest-nil for tabs with no CLI equivalent.

    /// The `lattice` CLI command string for the active tab, or nil when the tab has no equivalent.
    private var cliCommand: String? {
        switch store.selection {
        case .chat:
            let model = store.chatSelectedModelName.isEmpty
                ? (store.targetModel?.name ?? "model")
                : store.chatSelectedModelName
            let thinkFlag = store.chatEnableThinking ? "--think " : ""
            let budget = store.chatReasoningBudgetText
            let temp   = store.chatTempText
            let topk   = store.chatTopKText
            let topp   = store.chatTopPText
            // Build the command; coalesce any accidental doubled spaces from an empty thinkFlag.
            let raw = "lattice chat \(model) \(thinkFlag)--reasoning-budget \(budget) --temperature \(temp) --top-k \(topk) --top-p \(topp)"
            return raw.replacingOccurrences(of: "  ", with: " ")

        case .serve:
            // Mirrors ServeController.start(): it launches the standalone `lattice_serve`
            // binary with `--model <path> --port <port>` (no `--host` — the daemon always
            // binds 127.0.0.1), so the footer must match that shape, not the `lattice`
            // CLI's unrelated `serve` subcommand.
            let modelPath = store.serve?.servingModelPath?.path
                ?? store.targetModel?.path.path
                ?? "<model-path>"
            let port = store.serve?.port ?? 11435
            return "lattice_serve --model \(modelPath) --port \(port)"

        case .quantize, .train, .inspect:
            return nil
        }
    }

    /// Short caption shown trailing-right of the footer strip.
    private var cliCaption: String? {
        switch store.selection {
        case .chat:   return "streaming from local subprocess"
        case .serve:  return store.serve?.endpoint ?? "http://127.0.0.1:11435/v1"
        default:      return nil
        }
    }

    // MARK: Header — working model + residency

    private var header: some View {
        HStack(alignment: .center, spacing: Theme.Space.md) {
            // Title + spec on one baseline (mockup), not stacked: the muted metadata sits inline
            // to the right of the model name.
            HStack(alignment: .firstTextBaseline, spacing: Theme.Space.sm) {
                Text(store.targetModel?.name ?? "No model")
                    .font(Theme.Fonts.screenTitle)
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .lineLimit(1)
                if let meta = metaLine {
                    Text(meta)
                        .font(.system(size: 12, weight: .regular, design: .monospaced))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: Theme.Space.md)
            residency
            // New-chat + settings reveal on header hover so the resting corner matches the mockup's
            // clean "● Resident"; both stay reachable on hover (and via ⌘\ / the inspector).
            headerControls
                .opacity(headerHovered ? 1 : 0)
                .allowsHitTesting(headerHovered)
        }
        .padding(.horizontal, Theme.Space.xl)
        .padding(.top, Theme.Space.md)
        .padding(.bottom, Theme.Space.md)
        .onHover { headerHovered = $0 }
    }

    @ViewBuilder
    private var residency: some View {
        let resident = store.residentModel?.id == store.targetModel?.id && store.targetModel != nil
        let loading = store.isChatModelLoading
            && store.chatSelectedModelName == store.targetModel?.name
        HStack(spacing: 6) {
            Circle()
                .fill(resident ? Theme.Palette.success
                      : (loading ? Theme.Palette.signal : Theme.Palette.idle))
                .frame(width: 7, height: 7)
            Text(resident ? "Resident" : (loading ? "Loading…" : "Not loaded"))
                .font(Theme.Fonts.controlText)
                .foregroundStyle(resident ? Theme.Palette.success
                                 : (loading ? Theme.Palette.signal : Theme.Palette.textTertiary))
        }
    }

    // Quiet plain icons so the green residency pill stays the prominent element in the header,
    // matching the mockup. The primary controls (Reasoning, CPU/GPU) live in the composer now;
    // the gear only reaches the advanced sampling/generation knobs.
    @ViewBuilder
    private var headerControls: some View {
        if store.selection.hasInspector {
            Button { store.newChatConversation() } label: {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            .buttonStyle(.plain)
            .help("New conversation")

            Button { store.inspectorPresented.toggle() } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            .buttonStyle(.plain)
            .help("Chat settings")
        }
    }

    private var metaLine: String? {
        guard let model = store.targetModel else { return nil }
        var parts: [String] = []
        if let params = model.params { parts.append(params) }
        parts.append(formatLabel(model.format))
        if let ctx = model.contextLength { parts.append(ctxLabel(ctx)) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private func formatLabel(_ f: ModelFormat) -> String {
        switch f {
        case .bf16: return "BF16"
        case .q4: return "Q4"
        case .quarot: return "Q4 rotated"
        case .embedding: return "Embedding"
        case .unknown: return "—"
        }
    }

    private func ctxLabel(_ ctx: Int) -> String {
        if ctx >= 1024 && ctx % 1024 == 0 { return "\(ctx / 1024)k ctx" }
        if ctx >= 1000 { return "\(ctx / 1000)k ctx" }
        return "\(ctx) ctx"
    }

    // MARK: Verb tab bar

    private var verbTabBar: some View {
        HStack(spacing: Theme.Space.xl) {
            ForEach(Screen.allCases) { screen in
                VerbTab(screen: screen,
                        active: store.selection == screen) {
                    store.selection = screen
                }
            }
            Spacer()
        }
        .padding(.horizontal, Theme.Space.xl)
    }

    // MARK: Tab body

    @ViewBuilder
    private var tabBody: some View {
        switch store.selection {
        case .chat:     ChatScreen(store: store, embedded: true)
        case .serve:    ServeTab(store: store)
        case .quantize: QuantizeTab(store: store)
        case .train:    TrainTab(store: store)
        case .inspect:  InspectTab(store: store)
        }
    }
}

// MARK: - One verb tab (title + active indigo underline)

private struct VerbTab: View {
    let screen: Screen
    let active: Bool
    let action: () -> Void
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Text(screen.title)
                    .font(Theme.Fonts.controlText)
                    .foregroundStyle(active ? Theme.Palette.textPrimary
                                     : (hovered ? Theme.Palette.textSecondary : Theme.Palette.textTertiary))
                Rectangle()
                    .fill(active ? Theme.Palette.signal : .clear)
                    .frame(height: 2)
            }
            // Hug the title width so the underline spans only the label (mockup), and the
            // tabs stay tight-left-grouped instead of stretching edge-to-edge.
            .fixedSize()
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
        .animation(.easeOut(duration: Theme.Motion.hover), value: active)
    }
}
