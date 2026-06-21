import SwiftUI

// MARK: - InspectorShell
//
// Graphite Signal Lab — reusable edge-attached configuration-inspector container.
// Generalizes ChatScreen's `ChatInspector` into a standalone shell.
//
// Design laws:
//   - Edge-attached (not a rounded card): fills its column, single 1px hairline on leading edge.
//   - Background: Theme.Palette.panel (opaque, no glass).
//   - Internal padding: Theme.Space.lg (16pt) on all sides.
//   - Optional title in Theme.Fonts.inspectorTitle, top-aligned before content.
//   - No external border radius; the leading hairline provides the depth line.

/// A reusable edge-attached configuration-inspector container.
///
/// Drop inside a screen's `.inspector { }` closure or any fixed-width column:
///
/// ```swift
/// InspectorShell(title: "Settings") {
///     settingsRows
/// }
/// ```
///
/// Without a title:
/// ```swift
/// InspectorShell {
///     contentRows
/// }
/// ```
struct InspectorShell<Content: View>: View {
    let title: String?
    private let content: Content

    init(title: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Optional title row
            if let title {
                Text(title)
                    .font(Theme.Fonts.inspectorTitle)
                    .foregroundStyle(Theme.Palette.ink)
                    .padding(.bottom, Theme.Space.lg)
            }

            content
        }
        .padding(Theme.Space.lg)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Theme.Palette.panel)
        .overlay(alignment: .leading) {
            // 1px hairline on leading edge — edge-attached depth line, not a rounded card border
            Theme.Palette.hairline
                .frame(width: 1)
        }
    }
}

// MARK: - Previews

#Preview("InspectorShell") {
    HStack(spacing: 0) {
        // Simulate a main content area
        Rectangle()
            .fill(Theme.Palette.canvas)
            .frame(maxWidth: .infinity)
            .overlay {
                Text("Main content area")
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.inkDim)
            }

        // Inspector with title
        InspectorShell(title: "Settings") {
            VStack(alignment: .leading, spacing: Theme.Space.md) {
                HStack {
                    Text("Adapter")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textSecondary)
                    Spacer()
                    Text("none")
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Palette.inkDim)
                }

                Divider()

                HStack {
                    Text("Temperature")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textSecondary)
                    Spacer()
                    Text("0.7")
                        .font(Theme.Fonts.tableNumeric)
                        .monospacedDigit()
                        .foregroundStyle(Theme.Palette.ink)
                }

                HStack {
                    Text("Max tokens")
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textSecondary)
                    Spacer()
                    Text("256")
                        .font(Theme.Fonts.tableNumeric)
                        .monospacedDigit()
                        .foregroundStyle(Theme.Palette.ink)
                }
            }
        }
        .frame(width: 280)
    }
    .frame(width: 600, height: 400)
    .background(Theme.Palette.canvas)
}

#Preview("InspectorShell — no title") {
    HStack(spacing: 0) {
        Rectangle()
            .fill(Theme.Palette.canvas)
            .frame(maxWidth: .infinity)

        InspectorShell {
            Text("Content without title header")
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.ink)
        }
        .frame(width: 280)
    }
    .frame(width: 600, height: 300)
    .background(Theme.Palette.canvas)
}
