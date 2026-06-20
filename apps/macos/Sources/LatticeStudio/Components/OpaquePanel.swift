import SwiftUI

// MARK: - Primitive 1: OpaquePanel
//
// Law: "Numbers never touch glass."
// Enforces opaque Theme.Palette.panel fill with 1px Theme.Palette.hairline border.
// Corner radius is Theme.Radius.panel = 0px (panels are RULED, not carded).
//
// Usage:
//   SomeView()
//       .instrumentPanel()
//
//   // or wrap content:
//   OpaquePanel { ... }

/// An opaque, hairline-bordered instrument panel container.
/// Corner radius = 0 (ruled, not carded). No glass permitted inside.
struct OpaquePanel<Content: View>: View {
    private let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .background(Theme.Palette.panel)
            .overlay(
                Rectangle()
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
            )
    }
}

// MARK: - Instrument Panel View Modifier

struct InstrumentPanelModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(Theme.Palette.panel)
            .overlay(
                Rectangle()
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
            )
    }
}

extension View {
    /// Applies the opaque instrument-panel surface: Theme.Palette.panel fill + 1px hairline border.
    /// Corner radius = 0 (ruled, not carded). No glass permitted on this surface.
    func instrumentPanel() -> some View {
        modifier(InstrumentPanelModifier())
    }
}

// MARK: - Readout Well Surface Style
//
// Recessed machined-in surface used inside readout wells.
// Fill: Theme.Palette.wellSink
// Border: 1px Theme.Palette.hairline
// Corner radius: Theme.Radius.well = 6px
// Inner top-shadow: 2px so it reads machined-in, not raised.

struct ReadoutWellSurface: ViewModifier {
    @Environment(\.colorScheme) private var colorScheme

    private var shadowColor: Color {
        colorScheme == .dark
            ? Color.black.opacity(0.5)
            : Color.black.opacity(0.06)
    }

    func body(content: Content) -> some View {
        content
            .background(Theme.Palette.wellSink)
            .overlay(
                // 2px inner top-shadow: offset y=2, no spread — reads machined-in.
                // Simulated with a top-edge gradient overlay since SwiftUI has no inner-shadow.
                VStack(spacing: 0) {
                    shadowColor
                        .frame(height: 2)
                    Spacer()
                }
            )
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.well, style: .continuous)
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
            )
    }
}

extension View {
    /// Applies the recessed readout-well surface: wellSink fill + 2px inner top-shadow + hairline border.
    func readoutWellSurface() -> some View {
        modifier(ReadoutWellSurface())
    }
}

// MARK: - Previews

#Preview("OpaquePanel") {
    VStack(spacing: 0) {
        OpaquePanel {
            Text("Loss: 0.6121")
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .padding(Theme.Space.lg)
        }

        HStack {
            Text("INSTRUMENT PANEL — opaque, hairline-bordered, 0px radius")
                .instrumentLabel()
                .padding(Theme.Space.lg)
        }
        .instrumentPanel()

        HStack {
            Text("WELL SINK — recessed surface")
                .instrumentLabel()
                .padding(Theme.Space.sm)
        }
        .readoutWellSurface()
        .padding(Theme.Space.lg)
        .instrumentPanel()
    }
    .background(Theme.Palette.canvas)
    .frame(width: 360)
}
