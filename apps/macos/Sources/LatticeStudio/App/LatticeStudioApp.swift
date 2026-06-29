import SwiftUI
import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    // Weak back-reference injected by LatticeStudioApp so the delegate can reach the store.
    // (NSApplicationDelegate is an ObjC protocol; we cannot pass the store in an init.)
    weak var store: AppStore?

    /// The single OpenAI-format HTTP daemon, controlled from the menu-bar tray.
    let serve = ServeController()

    private var statusItem: NSStatusItem?
    /// The SwiftUI WindowGroup's NSWindow, captured via WindowAccessor. Kept alive
    /// (isReleasedWhenClosed=false) so close-to-background can re-show the same window.
    private weak var mainWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        serve.onChange = { [weak self] in self?.refreshStatusTitle() }
        installStatusItem()
    }

    /// Best-effort clean-quit: stop the active run so its trainer process exits normally
    /// and its PID is deregistered before the app terminates.  The startup reaper handles
    /// the crash / force-quit case where this callback never fires.
    func applicationWillTerminate(_ notification: Notification) {
        serve.stop()
        store?.stopRun()
    }

    // Closing the last window must NOT quit: the app lives in the menu bar and the OpenAI
    // daemon keeps serving. Quit is only via the tray's Quit item or ⌘Q.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    // Clicking the Dock icon (or `open`-ing the app again) re-shows the hidden window.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showMainWindow()
        return true
    }

    // MARK: Window capture + close-to-background

    /// Called from WindowAccessor once SwiftUI has created the NSWindow. Retargets the red
    /// close button to hide (orderOut) instead of destroy, and keeps the window object alive.
    func attach(window: NSWindow) {
        guard mainWindow !== window else { return }
        mainWindow = window
        window.isReleasedWhenClosed = false
        // The system title bar is hidden, so the full-width runtime bar owns the top edge. Let the
        // whole bar drag the window the way a title bar would (the traffic lights still float over
        // its leading inset).
        window.isMovableByWindowBackground = true
        if let closeButton = window.standardWindowButton(.closeButton) {
            closeButton.target = self
            closeButton.action = #selector(hideMainWindow)
        }
    }

    @objc private func hideMainWindow() {
        mainWindow?.orderOut(nil)
    }

    @objc private func showMainWindow() {
        NSApp.setActivationPolicy(.regular)
        mainWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    // MARK: Menu-bar tray

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            let img = NSImage(systemSymbolName: "point.3.connected.trianglepath.dotted",
                              accessibilityDescription: "Lattice")
            img?.isTemplate = true
            button.image = img
            button.toolTip = "Lattice"
        }
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu
        statusItem = item
    }

    /// Update only the status-item button tooltip when the daemon flips state. The menu body
    /// is rebuilt lazily in `menuNeedsUpdate`, so nothing else is needed here.
    private func refreshStatusTitle() {
        statusItem?.button?.toolTip = serve.isRunning ? "Lattice · API on :\(serve.port)" : "Lattice"
    }

    /// The model the tray will serve: whatever Chat has selected, else the first discovered model.
    private func serveTargetModel() -> ModelInfo? {
        guard let store else { return nil }
        return store.models.first { $0.name == store.chatSelectedModelName } ?? store.models.first
    }

    // NSMenuDelegate: rebuild the menu each time it opens so server state is always current.
    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()

        let show = NSMenuItem(title: "Show Lattice Window", action: #selector(showMainWindow), keyEquivalent: "")
        show.target = self
        menu.addItem(show)

        menu.addItem(.separator())

        let statusLine: String
        if serve.isRunning {
            let name = serve.servingModelName.map { " · \($0)" } ?? ""
            statusLine = "API server: running on :\(serve.port)\(name)"
        } else if let err = serve.lastError {
            statusLine = "API server: \(err)"
        } else {
            statusLine = "API server: stopped"
        }
        let statusItemRow = NSMenuItem(title: statusLine, action: nil, keyEquivalent: "")
        statusItemRow.isEnabled = false
        menu.addItem(statusItemRow)

        if serve.isRunning {
            let stop = NSMenuItem(title: "Stop API Server", action: #selector(toggleServer), keyEquivalent: "")
            stop.target = self
            menu.addItem(stop)

            let copy = NSMenuItem(title: "Copy Endpoint URL", action: #selector(copyEndpoint), keyEquivalent: "")
            copy.target = self
            menu.addItem(copy)
        } else {
            let target = serveTargetModel()
            let title = target.map { "Start API Server — \($0.name)" } ?? "Start API Server (no model)"
            let start = NSMenuItem(title: title, action: #selector(toggleServer), keyEquivalent: "")
            start.target = self
            start.isEnabled = target != nil
            menu.addItem(start)
        }

        menu.addItem(.separator())

        let quit = NSMenuItem(title: "Quit Lattice", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quit)
    }

    @objc private func toggleServer() {
        if serve.isRunning {
            serve.stop()
        } else if let model = serveTargetModel() {
            serve.start(model: model)
        }
    }

    @objc private func copyEndpoint() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(serve.endpoint, forType: .string)
    }
}

/// Bridges the SwiftUI WindowGroup's underlying NSWindow back to the AppDelegate. SwiftUI
/// creates the NSView's window slightly after first layout, so resolve it on the next runloop.
private struct WindowAccessor: NSViewRepresentable {
    let onResolve: (NSWindow) -> Void
    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        DispatchQueue.main.async { [weak v] in
            if let window = v?.window { onResolve(window) }
        }
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}

@main
struct LatticeStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var store = AppStore()

    var body: some Scene {
        WindowGroup("Lattice", id: "main") {
            ContentView(store: store)
                .frame(minWidth: 1120, minHeight: 720)
                // The studio is a dark instrument panel by identity — lock the canvas to dark
                // rather than tracking system appearance, so the surface, accent, and telemetry
                // read as designed regardless of the host's light/dark setting.
                .preferredColorScheme(.dark)
                .background(WindowAccessor { window in appDelegate.attach(window: window) })
                .onAppear {
                    store.onAppear()
                    // Give the delegate a weak reference so applicationWillTerminate can
                    // stop the active run for a clean-quit.  The reaper is the crash backstop.
                    appDelegate.store = store
                    // Share the tray's serve daemon with the store so the runtime telemetry
                    // and Serve surface observe its live state instead of polling the tray.
                    store.serve = appDelegate.serve
                }
        }
        .windowStyle(.hiddenTitleBar)
    }
}
