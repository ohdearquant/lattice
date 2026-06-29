import SwiftUI
import AppKit

// MARK: - The runtime bar
//
// The top chrome of the redesign: the brand lockup lives in the sidebar header, the live
// telemetry cluster lives at the trailing edge of the unified toolbar. Everything here reads
// REAL store state — resident model, unified memory, serve daemon — never a fabricated value.

// MARK: Brand mark — the lattice truss tower, drawn as a Shape so the build stays hermetic
// (no asset catalog). Single indigo stroke; swap the geometry when the logo mark is finalized.
struct LatticeMark: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let s = min(rect.width, rect.height) / 64.0
        let ox = rect.minX + (rect.width  - 64 * s) / 2
        let oy = rect.minY + (rect.height - 64 * s) / 2
        func pt(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: ox + x * s, y: oy + y * s) }

        // Pylon legs + cap.
        p.move(to: pt(27, 12)); p.addLine(to: pt(16, 52))
        p.move(to: pt(37, 12)); p.addLine(to: pt(48, 52))
        p.move(to: pt(27, 12)); p.addLine(to: pt(37, 12))

        // Horizontal ties.
        let ties: [(CGFloat, CGFloat, CGFloat)] = [
            (22, 24.3, 39.7), (32, 21.5, 42.5), (42, 18.75, 45.25), (52, 16, 48),
        ]
        for (y, x0, x1) in ties { p.move(to: pt(x0, y)); p.addLine(to: pt(x1, y)) }

        // Zigzag web.
        let web: [(CGFloat, CGFloat)] = [
            (39.7, 22), (24.3, 22), (42.5, 32), (21.5, 32), (45.25, 42), (18.75, 42), (48, 52),
        ]
        p.move(to: pt(27, 12))
        for q in web { p.addLine(to: pt(q.0, q.1)) }
        return p
    }
}

// MARK: Resident-model pill — which model is held in GPU memory right now. Honest states only.
struct ResidentPill: View {
    @Bindable var store: AppStore
    var body: some View {
        let loading = store.isChatModelLoading
        let name = store.chatWarmModelName
        // Decimal GB (÷10^9) to match the sidebar, the chat hero, and the mockup's figures.
        let sizeGB = store.residentModel.map { Double($0.sizeBytes) / 1_000_000_000.0 }
        let resident = name != nil
        let dot = resident ? Theme.Palette.success : (loading ? Theme.Palette.signal : Theme.Palette.idle)

        HStack(spacing: 6) {
            Circle().fill(dot).frame(width: 6, height: 6)
            if let name {
                Text(name)
                    .font(Theme.Fonts.controlText)
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .lineLimit(1)
                if let sizeGB {
                    Text("\(sizeGB, format: .number.precision(.fractionLength(2))) GB")
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundStyle(Theme.Palette.textSecondary)
                        .padding(.leading, 2)
                }
            } else if loading {
                Text("loading…").font(Theme.Fonts.controlText).foregroundStyle(Theme.Palette.textSecondary)
            } else {
                Text("no model resident").font(Theme.Fonts.controlText).foregroundStyle(Theme.Palette.textTertiary)
            }
        }
        .padding(.horizontal, 9).padding(.vertical, 4)
        .background(
            Capsule().fill(resident ? Theme.Palette.success.opacity(0.10) : Theme.Palette.hoverOverlay)
        )
        .overlay(
            Capsule().strokeBorder(
                resident ? Theme.Palette.success.opacity(0.28) : Theme.Palette.borderStandard,
                lineWidth: 1)
        )
        .help(resident ? "Model resident in GPU memory"
              : (loading ? "Loading model into GPU memory…" : "No model loaded — send a message or press Load"))
    }
}

// MARK: Unified-memory meter — the live 1 Hz readout, kept neutral grey so it stays informational.
struct RAMMeter: View {
    @Bindable var store: AppStore
    var body: some View {
        let m = store.memoryUsage
        let frac = m.totalGB > 0 ? min(m.usedGB / m.totalGB, 1.0) : 0
        // Tint the meter amber as unified memory approaches saturation — a model that no
        // longer fits is the failure this readout exists to warn about.
        let barColor = frac >= 0.9 ? Theme.Palette.amber : Theme.Palette.textSecondary.opacity(0.7)
        HStack(spacing: 7) {
            Text("RAM")
                .font(.system(size: 10, weight: .semibold))
                .tracking(0.5)
                .foregroundStyle(Theme.Palette.textTertiary)
            Capsule().fill(Theme.Palette.wellSink)
                .frame(width: 46, height: 4)
                .overlay(alignment: .leading) {
                    Capsule().fill(barColor)
                        .frame(width: CGFloat(frac) * 46, height: 4)
                }
            Text("\(m.usedGB, format: .number.precision(.fractionLength(1))) / \(m.totalGB, format: .number.precision(.fractionLength(0))) GB")
                .font(.system(size: 11, weight: .regular, design: .monospaced))
                .foregroundStyle(Theme.Palette.textSecondary)
                .fixedSize()
        }
        .help("Unified memory in use")
    }
}

// MARK: Serve chip — OpenAI-compatible API daemon status. Green only while actually serving.
struct ServeChip: View {
    @Bindable var store: AppStore
    var body: some View {
        let running = store.serveDisplayRunning
        let port = store.serveDisplayPort
        HStack(spacing: 5) {
            Circle().fill(running ? Theme.Palette.success : Theme.Palette.idle).frame(width: 6, height: 6)
            // verbatim: plain-String interpolation, so the port never picks up a locale
            // thousands separator (":11,435").
            Text(verbatim: ":\(port)")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(running ? Theme.Palette.textPrimary : Theme.Palette.textTertiary)
                .fixedSize()
            if !running {
                Text("off").font(Theme.Fonts.caption).foregroundStyle(Theme.Palette.textTertiary)
            }
        }
        .padding(.horizontal, 9).padding(.vertical, 4)
        .background(
            Capsule().fill(running ? Theme.Palette.success.opacity(0.10) : Theme.Palette.hoverOverlay)
        )
        .overlay(
            Capsule().strokeBorder(
                running ? Theme.Palette.success.opacity(0.28) : Theme.Palette.borderStandard,
                lineWidth: 1)
        )
        .help(running ? "OpenAI-compatible API serving on 127.0.0.1:\(port)" : "API server stopped")
    }
}

// MARK: The trailing telemetry cluster — resident model · unified memory · API server.
struct RuntimeTelemetry: View {
    @Bindable var store: AppStore
    private var divider: some View {
        Rectangle().fill(Theme.Palette.hairline).frame(width: 1, height: 14)
    }
    var body: some View {
        HStack(spacing: 10) {
            ResidentPill(store: store)
            divider
            RAMMeter(store: store)
            divider
            ServeChip(store: store)
        }
        .fixedSize()
    }
}

// MARK: The full-width runtime bar — the top chrome that spans sidebar + main column.
//
// `.windowStyle(.hiddenTitleBar)` removes the system title bar, so this bar owns the whole top
// edge: the traffic lights float over its leading inset and the live telemetry cluster anchors the
// trailing edge. A compact height keeps the chrome quiet so the chat below is the focus.
struct SpanningRuntimeBar: View {
    @Bindable var store: AppStore

    /// Leading inset that clears the macOS traffic-light cluster (≈14pt each at ~20pt pitch from
    /// the window's left edge), so the telemetry never crowds the lights on a narrow window.
    private let trafficLightInset: CGFloat = 78

    var body: some View {
        HStack(spacing: 0) {
            Spacer(minLength: 16)
            RuntimeTelemetry(store: store)
        }
        .padding(.leading, trafficLightInset)
        .padding(.trailing, 16)
        .frame(height: 44)
        .frame(maxWidth: .infinity)
        // Transparent AppKit hit-view (in front of the fill) reproduces the title-bar gestures the
        // hidden title bar would provide — drag to move, double-click to zoom — since the bar is
        // drawn over that region. The panel fill shows through it.
        .background(TitlebarBehavior())
        // Panel tone (the sidebar's surface) so the chrome reads as one lighter band wrapping the
        // darker content well — the near-black `window` tone flattened that depth and read jet-black.
        .background(Theme.Palette.panel)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Theme.Palette.hairline).frame(height: 1)
        }
    }
}

// MARK: - Title-bar gestures for the runtime bar
//
// The runtime bar is drawn over the hidden title bar (ContentView ignores the top safe area), so
// it must reproduce the two title-bar gestures the system would otherwise provide: single-click
// drag to move the window, and double-click to run the user's System Settings title-bar action
// (zoom / minimize). A plain background view loses double-click-to-zoom; this restores it.
struct TitlebarBehavior: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView { TitlebarHitView() }
    func updateNSView(_ nsView: NSView, context: Context) {}
}

private final class TitlebarHitView: NSView {
    // Let a drag begin even when the window isn't yet key, matching real title-bar behavior.
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func mouseDown(with event: NSEvent) {
        guard event.clickCount == 2 else {
            window?.performDrag(with: event)
            return
        }
        switch UserDefaults.standard.string(forKey: "AppleActionOnDoubleClick") {
        case "Minimize": window?.miniaturize(nil)
        case "None":     break
        default:         window?.zoom(nil)   // "Maximize" — the system default
        }
    }
}
