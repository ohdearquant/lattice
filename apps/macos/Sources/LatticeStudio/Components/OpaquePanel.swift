import SwiftUI

// MARK: - Primitive 1: OpaquePanel
//
// Graphite Signal Lab design laws:
//   - Dark graphite surfaces (#121820 panel, #080B0F wellSink).
//   - 10pt continuous-radius cards — panels are CARDED, not ruled (0px era is over).
//   - White-opacity hairlines: borderStandard = white 8%, no static drop shadows.
//   - One cyan accent (#48D8C4) for signal / CTA — never decorative.
//   - Numbers never touch glass — wells stay opaque.
//
// Usage:
//   SomeView()
//       .instrumentPanel()
//
//   // or wrap content:
//   OpaquePanel { ... }

/// An opaque, hairline-bordered instrument panel container.
/// Corner radius = Theme.Radius.panel (10pt continuous). No glass permitted inside.
struct OpaquePanel<Content: View>: View {
    private let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .background(Theme.Palette.panel)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                    .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
            )
    }
}

// MARK: - Instrument Panel View Modifier

struct InstrumentPanelModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(Theme.Palette.panel)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.panel, style: .continuous)
                    .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
            )
    }
}

extension View {
    /// Applies the opaque instrument-panel surface: Theme.Palette.panel fill + 1px white-opacity border.
    /// Corner radius = Theme.Radius.panel (10pt continuous). No glass permitted on this surface.
    func instrumentPanel() -> some View {
        modifier(InstrumentPanelModifier())
    }
}

// MARK: - Readout Well Surface Style
//
// Recessed machined-in surface used inside readout wells.
// Fill: Theme.Palette.wellSink (#080B0F dark / #E8EDF2 light)
// Border: 1px white-opacity hairline (borderStandard)
// Corner radius: Theme.Radius.well (8pt continuous)
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
                    .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
            )
    }
}

extension View {
    /// Applies the recessed readout-well surface: wellSink fill + 2px inner top-shadow + white-opacity border.
    func readoutWellSurface() -> some View {
        modifier(ReadoutWellSurface())
    }
}

// MARK: - Previews

#Preview("OpaquePanel") {
    VStack(spacing: Theme.Space.md) {
        OpaquePanel {
            Text("Loss: 0.6121")
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .padding(Theme.Space.lg)
        }

        HStack {
            Text("INSTRUMENT PANEL — opaque, 10pt card, white-opacity hairline")
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
    .padding(Theme.Space.lg)
}
