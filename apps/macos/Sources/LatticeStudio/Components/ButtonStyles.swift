import SwiftUI

// MARK: - ButtonStyles
//
// Graphite Signal Lab button vocabulary — Section D "Primary button" + "Secondary button".
//
// Design laws (spec §D):
//   - Height 30pt standard; use frame(height: 34) for rail-footer placement.
//   - Horizontal padding 12pt, symbol gap 6pt, radius 6pt (Theme.Radius.control).
//   - Text 12pt medium (Theme.Fonts.controlText).
//   - Symbol: 13pt medium (see Label use in previews).
//   - Native keyboard activation and focusability preserved via ButtonStyle (not a custom control).
//   - Hover approximated with .onHover + state; SwiftUI on macOS 14 has no hover modifier on ButtonStyle.
//
// Types are named Lattice* to avoid collision with any private screen-local button styles.

// MARK: - LatticePrimaryButtonStyle

/// Accent-filled primary button: accent fill, onAccent text.
///
/// ```swift
/// Button("Run") { /* … */ }
///     .buttonStyle(LatticePrimaryButtonStyle())
///
/// // Rail-footer variant (34pt height):
/// Button("Run") { /* … */ }
///     .buttonStyle(LatticePrimaryButtonStyle(height: 34))
/// ```
struct LatticePrimaryButtonStyle: ButtonStyle {
    var height: CGFloat = Theme.Space.controlHeight  // 30; pass 34 for rail footer

    func makeBody(configuration: Configuration) -> some View {
        LatticePrimaryButtonBody(configuration: configuration, height: height)
    }
}

private struct LatticePrimaryButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let height: CGFloat

    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovered = false

    private var fillColor: Color {
        if configuration.isPressed { return Theme.Palette.accentActive }
        if isHovered { return Theme.Palette.accentHover }
        return Theme.Palette.accent
    }

    var body: some View {
        configuration.label
            .font(Theme.Fonts.controlText)
            .foregroundStyle(
                isEnabled
                    ? Theme.Palette.onAccent
                    : Theme.Palette.textDisabled
            )
            .padding(.horizontal, 12)
            .frame(height: height)
            .background(
                isEnabled ? fillColor : Theme.Palette.surfaceDisabled,
                in: RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
            )
            .animation(.easeOut(duration: Theme.Motion.hover), value: isHovered)
            .animation(.easeOut(duration: Theme.Motion.press), value: configuration.isPressed)
            .onHover { isHovered = $0 }
    }
}

// MARK: - LatticeSecondaryButtonStyle

/// Raised-surface secondary button: surfaceRaised fill, standard border, ink text.
///
/// ```swift
/// Button("Reveal") { /* … */ }
///     .buttonStyle(LatticeSecondaryButtonStyle())
/// ```
struct LatticeSecondaryButtonStyle: ButtonStyle {
    var height: CGFloat = Theme.Space.controlHeight  // 30; pass 34 for rail footer

    func makeBody(configuration: Configuration) -> some View {
        LatticeSecondaryButtonBody(configuration: configuration, height: height)
    }
}

private struct LatticeSecondaryButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let height: CGFloat

    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovered = false

    private var fillColor: Color {
        if configuration.isPressed { return Theme.Palette.panel }
        if isHovered { return Theme.Palette.surfaceHover }
        return Theme.Palette.surfaceRaised
    }

    var body: some View {
        configuration.label
            .font(Theme.Fonts.controlText)
            .foregroundStyle(
                isEnabled
                    ? Theme.Palette.ink
                    : Theme.Palette.textDisabled
            )
            .padding(.horizontal, 12)
            .frame(height: height)
            .background(
                isEnabled ? fillColor : Theme.Palette.surfaceDisabled,
                in: RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
            )
            .animation(.easeOut(duration: Theme.Motion.hover), value: isHovered)
            .animation(.easeOut(duration: Theme.Motion.press), value: configuration.isPressed)
            .onHover { isHovered = $0 }
    }
}

// MARK: - Previews

#Preview("ButtonStyles") {
    VStack(alignment: .leading, spacing: Theme.Space.md) {
        Text("PRIMARY")
            .instrumentLabel()

        HStack(spacing: Theme.Space.sm) {
            Button("Run") {}
                .buttonStyle(LatticePrimaryButtonStyle())

            Button { } label: {
                Label("Run", systemImage: "play.fill")
            }
            .buttonStyle(LatticePrimaryButtonStyle())

            Button("Rail Footer") {}
                .buttonStyle(LatticePrimaryButtonStyle(height: 34))

            Button("Disabled") {}
                .buttonStyle(LatticePrimaryButtonStyle())
                .disabled(true)
        }

        Text("SECONDARY")
            .instrumentLabel()
            .padding(.top, Theme.Space.xs)

        HStack(spacing: Theme.Space.sm) {
            Button("Reveal") {}
                .buttonStyle(LatticeSecondaryButtonStyle())

            Button { } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(LatticeSecondaryButtonStyle())

            Button("Rail Footer") {}
                .buttonStyle(LatticeSecondaryButtonStyle(height: 34))

            Button("Disabled") {}
                .buttonStyle(LatticeSecondaryButtonStyle())
                .disabled(true)
        }
    }
    .padding(Theme.Space.lg)
    .background(Theme.Palette.canvas)
}
