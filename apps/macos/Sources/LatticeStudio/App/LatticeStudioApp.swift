import SwiftUI
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    // Weak back-reference injected by LatticeStudioApp so the delegate can reach the store.
    // (NSApplicationDelegate is an ObjC protocol; we cannot pass the store in an init.)
    weak var store: AppStore?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Best-effort clean-quit: stop the active run so its trainer process exits normally
    /// and its PID is deregistered before the app terminates.  The startup reaper handles
    /// the crash / force-quit case where this callback never fires.
    func applicationWillTerminate(_ notification: Notification) {
        store?.stopRun()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct LatticeStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var store = AppStore()

    var body: some Scene {
        WindowGroup("Lattice", id: "main") {
            ContentView(store: store)
                .frame(minWidth: 1120, minHeight: 720)
                .onAppear {
                    store.onAppear()
                    // Give the delegate a weak reference so applicationWillTerminate can
                    // stop the active run for a clean-quit.  The reaper is the crash backstop.
                    appDelegate.store = store
                }
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified(showsTitle: false))
    }
}
