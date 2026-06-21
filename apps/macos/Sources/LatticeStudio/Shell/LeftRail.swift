import SwiftUI

// MARK: - The left rail: wordmark + engine-state header, indexed nav, compact memory footer.
struct LeftRail: View {
    @Bindable var store: AppStore

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().overlay(Theme.Palette.hairline)
            nav
            Spacer(minLength: 0)
            Divider().overlay(Theme.Palette.hairline)
            memoryFooter
        }
        .background(Theme.Palette.panel)
    }

    // MARK: Header (~58pt): wordmark + engine-state row

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Wordmark
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text("Lattice")
                    .font(Theme.Fonts.bodyStrong)   // 13pt medium
                    .foregroundStyle(Theme.Palette.ink)
                Spacer()
            }
            // Engine-state row (~24pt)
            engineStateRow
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.top, Theme.Space.md)
        .padding(.bottom, Theme.Space.sm)
    }

    @ViewBuilder private var engineStateRow: some View {
        let isActive: Bool = {
            guard let run = store.liveRun else { return false }
            return run.status == .running || run.status == .paused
        }()
        let dotColor = isActive ? Theme.Palette.running : Theme.Palette.idle

        HStack(spacing: 6) {
            Circle()
                .fill(dotColor)
                .frame(width: 6, height: 6)
                .modifier(PulsingDot(active: isActive))
            Text(engineStateLabel)
                .font(Theme.Fonts.caption)   // 11pt regular
                .foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
        }
        .frame(height: 24)
    }

    private var engineStateLabel: String {
        guard let run = store.liveRun,
              run.status == .running || run.status == .paused else {
            return "Idle"
        }
        switch run.kind {
        case .train:          return "Training"
        case .quantizeQ4:     return "Quantizing"
        case .quantizeQuaRot: return "Quantizing"
        case .chat:           return "Generating"
        }
    }

    // MARK: Nav rows

    private var nav: some View {
        VStack(spacing: 2) {
            ForEach(Screen.allCases) { screen in
                navRow(screen)
            }
        }
        .padding(.horizontal, Theme.Space.xs)    // 4pt outer — rows add their own 10pt h-pad
        .padding(.vertical, Theme.Space.sm)
    }

    private func navRow(_ screen: Screen) -> some View {
        let selected = store.selection == screen
        return NavRowButton(screen: screen, selected: selected) {
            store.selection = screen
        }
    }

    // MARK: Memory footer (~56pt)

    private var memoryFooter: some View {
        let mem = store.memoryUsage
        let frac = mem.totalGB > 0 ? min(mem.usedGB / mem.totalGB, 1.0) : 0

        return VStack(alignment: .leading, spacing: 6) {
            // Label
            Text("UNIFIED MEMORY")
                .font(Theme.Fonts.sectionLabel)
                .tracking(0.85)
                .textCase(.uppercase)
                .foregroundStyle(Theme.Palette.textTertiary)

            // Progress track (3pt, wellSink bg + running at 0.6 fill)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Theme.Palette.wellSink)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Theme.Palette.running.opacity(0.6))
                        .frame(width: geo.size.width * frac)
                }
            }
            .frame(height: 3)

            // Value in SF Mono
            Text("\(mem.usedGB, format: .number.precision(.fractionLength(1))) / \(mem.totalGB, format: .number.precision(.fractionLength(0))) GB")
                .font(.system(size: 11, weight: .regular, design: .monospaced))
                .foregroundStyle(Theme.Palette.textSecondary)
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.md)
        .frame(height: 56)
    }
}

// MARK: - Nav Row Button
// 32pt tall, 10pt h-padding, 8pt symbol-to-label gap.
// Selected: selectionFill bg + selectionBorder overlay + 2pt leading accent marker.
// Hover: hoverOverlay.

private struct NavRowButton: View {
    let screen: Screen
    let selected: Bool
    let action: () -> Void

    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 0) {
                // 2pt leading accent marker (only when selected)
                Rectangle()
                    .fill(selected ? Theme.Palette.signal : .clear)
                    .frame(width: 2, height: 20)
                    .clipShape(RoundedRectangle(cornerRadius: 1))
                    .padding(.trailing, 8)

                // SF Symbol at 14pt medium in a 16×16 frame
                Image(systemName: screen.symbol)
                    .font(.system(size: 14, weight: .medium))
                    .frame(width: 16, height: 16)
                    .foregroundStyle(selected ? Theme.Palette.ink : Theme.Palette.textSecondary)

                // 8pt gap between symbol and label
                Spacer().frame(width: 8)

                // Screen label
                Text(screen.title.localizedCapitalized)
                    .font(Theme.Fonts.bodyStrong)
                    .foregroundStyle(selected ? Theme.Palette.ink : Theme.Palette.textSecondary)

                Spacer()

                // Trailing shortcut at 10pt SF Mono tertiary
                Text("⌘\(String(screen.shortcut.character).uppercased())")
                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                    .foregroundStyle(Theme.Palette.textTertiary)
            }
            .padding(.horizontal, 10)
            .frame(height: 32)
            .background(rowBackground)
            .overlay(rowBorder)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
    }

    @ViewBuilder private var rowBackground: some View {
        if selected {
            Theme.Palette.selectionFill
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control))
        } else if hovered {
            Theme.Palette.hoverOverlay
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control))
        } else {
            Color.clear
        }
    }

    @ViewBuilder private var rowBorder: some View {
        if selected {
            RoundedRectangle(cornerRadius: Theme.Radius.control)
                .strokeBorder(Theme.Palette.selectionBorder, lineWidth: 1)
        }
    }
}

// MARK: - Pulsing dot modifier (opacity 65%→100%, 1.2s, active run only)

private struct PulsingDot: ViewModifier {
    let active: Bool
    @State private var pulsing = false

    func body(content: Content) -> some View {
        if active {
            content
                .opacity(pulsing ? 1.0 : 0.65)
                .animation(
                    Animation.easeInOut(duration: 0.6).repeatForever(autoreverses: true),
                    value: pulsing
                )
                .onAppear { pulsing = true }
        } else {
            content
        }
    }
}

// MARK: - Screen symbol mapping

extension Screen {
    var symbol: String {
        switch self {
        case .models: "shippingbox"
        case .chat:   "text.bubble"
        case .train:  "chart.line.uptrend.xyaxis"
        case .runs:   "clock.arrow.circlepath"
        }
    }
}
