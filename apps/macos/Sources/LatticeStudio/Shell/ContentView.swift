import SwiftUI

struct ContentView: View {
    @Bindable var store: AppStore

    var body: some View {
        NavigationSplitView {
            LeftRail(store: store)
                .navigationSplitViewColumnWidth(Theme.Space.railWidth)
        } detail: {
            ZStack {
                Theme.Palette.canvas.ignoresSafeArea()
                detail
            }
            .toolbar {
                ToolbarItem(placement: .navigation) {
                    HStack(spacing: 8) {
                        Text(store.selection.index)
                            .font(Theme.Fonts.mono(12)).foregroundStyle(Theme.Palette.inkDim)
                        Text(store.selection.title)
                            .font(Theme.Fonts.display(13, .semibold)).foregroundStyle(Theme.Palette.ink)
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button { store.commandBarOpen.toggle() } label: {
                        Label("Command", systemImage: "command")
                    }
                    .help("Command palette (⌘K)")
                }
            }
        }
        .navigationTitle("")
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
    // (e.g. `train qwen3.5 r8`) preselects the working model for that screen.
    private func handleCommand(_ cmd: String, _ args: [String]) {
        func retarget() {
            guard let arg = args.first(where: { !$0.isEmpty })?.lowercased() else { return }
            if let m = store.models.first(where: { $0.name.lowercased().contains(arg) }) {
                store.workingModel = m
            }
        }
        switch cmd {
        case "train":    retarget(); store.selection = .train
        case "quantize": retarget(); store.selection = .quantize
        case "chat":     retarget(); store.selection = .chat
        case "models":   store.selection = .models
        case "data":     store.selection = .data
        case "runs":     store.selection = .runs
        case "stop":     store.stopRun()
        default:         break
        }
    }

    @ViewBuilder private var detail: some View {
        switch store.selection {
        case .models:   ModelsScreen(store: store)
        case .train:    TrainScreen(store: store)
        case .quantize: QuantizeScreen(store: store)
        case .chat:     ChatScreen(store: store)
        case .data:     DataScreen(store: store)
        case .runs:     RunsScreen(store: store)
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
            Button("") { store.inspectorCollapsed.toggle() }.keyboardShortcut("\\", modifiers: .command)
        }
        .opacity(0)
        .frame(width: 0, height: 0)
    }

}
