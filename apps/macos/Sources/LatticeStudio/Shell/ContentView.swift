import SwiftUI

// MARK: - Content view — the redesigned shell
//
// A full-width runtime bar spans the top (traffic lights + brand lockup + live telemetry), and
// beneath it a fixed model sidebar sits beside the verb-tab main column. The app is model-centric:
// pick a model on the left, act on it through a verb tab on the right. The system title bar is
// hidden (`.windowStyle(.hiddenTitleBar)`) so the runtime bar owns the whole top edge.

struct ContentView: View {
    @Bindable var store: AppStore

    var body: some View {
        VStack(spacing: 0) {
            SpanningRuntimeBar(store: store)
            HStack(spacing: 0) {
                ModelSidebar(store: store)
                MainColumn(store: store)
                    .frame(minWidth: 640, maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        // Pull the runtime bar up under the hidden titlebar so it owns the very top edge and the
        // floating traffic lights vertically center on the wordmark — without this, SwiftUI's
        // ~28pt top safe-area inset stacks the bar below the lights, giving a two-tier chrome.
        .ignoresSafeArea(.container, edges: .top)
        .background(Theme.Palette.window)
        // One neutral tint for every system control (Slider/Picker/Toggle/field caret). Elements
        // that should read indigo set Theme.Palette.signal explicitly, so they are unaffected —
        // this keeps the accent reserved for live data + the single CTA.
        .tint(Theme.Palette.control)
        .background(shortcuts)
        .overlay {
            CommandBar(
                isPresented: $store.commandBarOpen,
                commands: CommandSpec.latticeDefaults,
                onRun: handleCommand
            )
        }
        .sheet(isPresented: $store.getModelsPresented) {
            GetModelsSheet(store: store)
                .frame(minWidth: 680, idealWidth: 760, maxWidth: .infinity,
                       minHeight: 500, idealHeight: 620, maxHeight: .infinity)
        }
    }

    // Route a ⌘K command to a verb tab / action. A leading model-name argument preselects the
    // working model first, so `chat qwen3` opens Chat already pointed at that model.
    private func handleCommand(_ cmd: String, _ args: [String]) {
        func retarget() {
            guard let arg = args.first(where: { !$0.isEmpty })?.lowercased() else { return }
            if let m = store.models.first(where: { $0.name.lowercased().contains(arg) }) {
                store.selectSidebarModel(m)
            }
        }
        switch cmd {
        case "chat":       retarget(); store.selection = .chat
        case "serve":      retarget(); store.selection = .serve
        case "quantize":   retarget(); store.selection = .quantize
        case "train":      retarget(); store.selection = .train
        case "inspect":    retarget(); store.selection = .inspect
        case "get models": store.getModelsPresented = true
        case "stop":       store.stopRun()
        default:           break
        }
    }

    // Hidden buttons carry the global keyboard map (⌘1–5, ⌘K, ⌘\).
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
