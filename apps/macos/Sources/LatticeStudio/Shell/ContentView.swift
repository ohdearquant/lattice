import SwiftUI

struct ContentView: View {
    @Bindable var store: AppStore

    var body: some View {
        NavigationSplitView {
            LeftRail(store: store)
                // Sidebar: fixed ideal width with a collapse range so it never overlaps
                // the detail column on narrow windows.
                .navigationSplitViewColumnWidth(
                    min: Theme.Space.sidebarMin,
                    ideal: Theme.Space.railWidth,
                    max: Theme.Space.sidebarMax
                )
        } detail: {
            ZStack {
                Theme.Palette.canvas.ignoresSafeArea()
                detail
            }
            // Enforce a minimum detail width so the split view never collapses the
            // content area to zero on narrow resize — the window's own minWidth (1120)
            // is the outer bound; this clamps at the column level.
            .frame(minWidth: 640)
            .toolbar {
                // Run-status capsule: always reserve the slot; hide the capsule content
                // when there is no active run.  This prevents the toolbar from reflowing
                // when a run starts or stops, which previously caused the Embeddings
                // HSplitView to jump.
                ToolbarItem(placement: .primaryAction) {
                    if let run = store.liveRun, run.status == .running || run.status == .paused {
                        runStatusCapsule(run)
                    } else {
                        // Zero-size placeholder — keeps toolbar item count constant.
                        Color.clear.frame(width: 0, height: 0)
                    }
                }
                // Inspector toggle: only rendered for screens that have an inspector.
                if store.selection.hasInspector {
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            store.inspectorPresented.toggle()
                        } label: {
                            Image(systemName: "sidebar.right")
                        }
                        .help("Toggle settings (⌘\\)")
                        .foregroundStyle(store.inspectorPresented ? Theme.Palette.signal : Theme.Palette.textSecondary)
                    }
                }
            }
        }
        .navigationTitle("")
        // One neutral tint for every system control (Slider/Picker/Toggle/field caret).
        // Elements that should read teal set Theme.Palette.signal explicitly, so they are
        // unaffected — this keeps the accent reserved for live data + the single CTA.
        .tint(Theme.Palette.control)
        .background(shortcuts)
        .overlay {
            CommandBar(
                isPresented: $store.commandBarOpen,
                commands: CommandSpec.latticeDefaults,
                onRun: handleCommand
            )
        }
    }

    // Route a ⌘K command to a screen / action. A leading model-name argument
    // preselects the working model for that screen.
    private func handleCommand(_ cmd: String, _ args: [String]) {
        func retarget() {
            guard let arg = args.first(where: { !$0.isEmpty })?.lowercased() else { return }
            if let m = store.models.first(where: { $0.name.lowercased().contains(arg) }) {
                store.workingModel = m
            }
        }
        switch cmd {
        case "chat":   retarget(); store.selection = .chat
        case "models": store.selection = .models
        case "stop":   store.stopRun()
        default:       break
        }
    }

    // Compact toolbar capsule shown while a run is active.
    // Shows: status dot · model name · step counter — no invented fields.
    // Clicking navigates to the Runs screen.
    @ViewBuilder
    private func runStatusCapsule(_ run: LiveRun) -> some View {
        let activityLabel: String = {
            switch run.kind {
            case .train:          return "Training"
            case .quantizeQ4:     return "Quantizing"
            case .quantizeQuaRot: return "Quantizing"
            case .chat:           return "Generating"
            case .eval:           return "Evaluating"
            case .embed:          return "Embedding"
            }
        }()
        let stepLabel: String = {
            if run.kind == .train {
                let s = "step \(run.currentStep)"
                if let total = run.totalSteps { return s + "/\(total)" }
                return s
            }
            return run.status == .paused ? "paused" : "running"
        }()

        Button {
            // Jump to the screen that owns this run kind.
            switch run.kind {
            case .chat:
                store.selection = .chat
            default:
                store.selection = .models
            }
        } label: {
            HStack(spacing: 6) {
                Circle()
                    .fill(Theme.Palette.running)
                    .frame(width: 6, height: 6)
                    .opacity(run.status == .paused ? 0.5 : 1.0)
                Text(activityLabel)
                    .font(Theme.Fonts.controlText)
                    .foregroundStyle(Theme.Palette.textPrimary)
                Text(run.modelName)
                    .font(Theme.Fonts.controlText)
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .lineLimit(1)
                Text(stepLabel)
                    .font(.system(size: 11, weight: .regular, design: .monospaced))
                    .foregroundStyle(Theme.Palette.textSecondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(
                Theme.Palette.running.opacity(0.12)
                    .clipShape(Capsule())
            )
            .overlay(Capsule().strokeBorder(Theme.Palette.running.opacity(0.28), lineWidth: 1))
        }
        .buttonStyle(.plain)
        .help("View active run")
    }

    @ViewBuilder private var detail: some View {
        switch store.selection {
        case .models: ModelsScreen(store: store)
        case .chat:   ChatScreen(store: store)
        }
    }

    // Hidden buttons carry the global keyboard map (⌘1–6, ⌘K, ⌘\).
    private var shortcuts: some View {
        ZStack {
            ForEach(Screen.allCases) { s in
                Button("") { store.selection = s }
                    .keyboardShortcut(s.shortcut, modifiers: .command)
            }
            Button("") { store.commandBarOpen.toggle() }.keyboardShortcut("k", modifiers: .command)
            Button("") { store.inspectorPresented.toggle() }.keyboardShortcut("\\", modifiers: .command)
        }
        .opacity(0)
        .frame(width: 0, height: 0)
    }

}
