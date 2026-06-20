import SwiftUI

// MARK: - The left rail: wordmark, live RUN block, indexed nav, and a pinned SYSTEM STRIP.
struct LeftRail: View {
    @Bindable var store: AppStore

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            wordmark
            Divider().overlay(Theme.Palette.hairline)
            runBlock
            Divider().overlay(Theme.Palette.hairline)
            nav
            Spacer(minLength: 0)
            Divider().overlay(Theme.Palette.hairline)
            systemStrip
        }
        .background(Theme.Palette.panel)
    }

    private var wordmark: some View {
        HStack(spacing: 6) {
            Text("LATTICE")
                .font(Theme.Fonts.display(15, .bold))
                .tracking(2)
                .foregroundStyle(Theme.Palette.ink)
            Spacer()
            Text("·studio")
                .font(Theme.Fonts.mono(10))
                .foregroundStyle(Theme.Palette.inkDim)
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.md)
    }

    @ViewBuilder private var runBlock: some View {
        if let run = store.liveRun, run.status == .running || run.status == .paused {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Circle().fill(Theme.Palette.signal).frame(width: 7, height: 7)
                    Text(run.status == .paused ? "PAUSED" : "RUN").instrumentLabel()
                    Spacer()
                    Text(run.modelName).font(Theme.Fonts.mono(10)).foregroundStyle(Theme.Palette.inkDim)
                        .lineLimit(1).truncationMode(.middle)
                }
                if run.kind == .train {
                    Text("step \(run.currentStep)" + (run.totalSteps.map { "/\($0)" } ?? ""))
                        .font(Theme.Fonts.mono(11)).foregroundStyle(Theme.Palette.ink)
                    if let loss = run.currentLoss {
                        Text("loss \(loss, format: .number.precision(.fractionLength(4)))")
                            .font(Theme.Fonts.mono(11)).foregroundStyle(Theme.Palette.ink)
                    }
                }
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.vertical, Theme.Space.md)
        } else {
            HStack {
                Text("idle").font(Theme.Fonts.mono(11)).foregroundStyle(Theme.Palette.inkDim)
                Spacer()
            }
            .padding(.horizontal, Theme.Space.lg)
            .padding(.vertical, Theme.Space.md)
        }
    }

    private var nav: some View {
        VStack(spacing: 2) {
            ForEach(Screen.allCases) { screen in
                navRow(screen)
            }
        }
        .padding(.horizontal, Theme.Space.sm)
        .padding(.vertical, Theme.Space.sm)
    }

    private func navRow(_ screen: Screen) -> some View {
        let selected = store.selection == screen
        return Button {
            store.selection = screen
        } label: {
            HStack(spacing: 8) {
                Rectangle()
                    .fill(selected ? Theme.Palette.signal : .clear)
                    .frame(width: 2, height: 16)
                Text(screen.index).font(Theme.Fonts.mono(11)).foregroundStyle(Theme.Palette.inkDim)
                Text(screen.title).font(Theme.Fonts.display(12, selected ? .semibold : .regular))
                    .foregroundStyle(selected ? Theme.Palette.ink : Theme.Palette.inkDim)
                Spacer()
                Text("⌘\(String(screen.shortcut.character).uppercased())")
                    .font(Theme.Fonts.mono(9)).foregroundStyle(Theme.Palette.inkDim.opacity(0.7))
            }
            .padding(.vertical, 6)
            .padding(.trailing, 8)
            .background(selected ? Theme.Palette.canvas : .clear)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var systemStrip: some View {
        let mem = store.memoryUsage
        let frac = mem.totalGB > 0 ? mem.usedGB / mem.totalGB : 0
        return VStack(alignment: .leading, spacing: 6) {
            Text("SYSTEM").instrumentLabel()
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Rectangle().fill(Theme.Palette.wellSink)
                    Rectangle().fill(Theme.Palette.signal.opacity(0.6))
                        .frame(width: geo.size.width * frac)
                }
            }
            .frame(height: 4)
            .clipShape(RoundedRectangle(cornerRadius: 2))
            Text("\(mem.usedGB, format: .number.precision(.fractionLength(1)))/\(mem.totalGB, format: .number.precision(.fractionLength(0))) GB")
                .font(Theme.Fonts.mono(10)).foregroundStyle(Theme.Palette.inkDim)
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.md)
    }
}
