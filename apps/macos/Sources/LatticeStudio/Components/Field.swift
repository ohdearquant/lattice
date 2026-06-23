import SwiftUI

// MARK: - Field
//
// Graphite Signal Lab text and numeric field vocabulary — Section D "Text fields and numeric fields".
//
// Design laws (spec §D):
//   - Height 28–30pt, horizontal inset 8pt, radius 6pt (Theme.Radius.control).
//   - Fill: Theme.Palette.wellSink (surfaceInset / recessed) or surfaceRaised.
//   - Border: 1px borderStandard (white 8%).
//   - Placeholder: textTertiary.
//   - Focus: border accent 70% + subtle 2pt outer ring via .focusRing palette token.
//   - Error: border error 70% + 11pt caption beneath.
//   - Numeric variant: SF Mono (Fonts.tableNumeric), right-aligned, default width ~80.
//
// Implementation note (spec): "A plain field plus @FocusState when exact custom styling is required."
// We wrap TextField and apply all styling externally; no custom text engine.

// MARK: - LatticeField (text)

/// A styled single-line text field.
///
/// ```swift
/// @State private var name = ""
/// LatticeField("Model name", text: $name)
///
/// // With error:
/// LatticeField("Path", text: $path, errorMessage: pathError)
/// ```
struct LatticeField: View {
    let prompt: String
    @Binding var text: String
    var errorMessage: String? = nil
    var height: CGFloat = 30

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            ZStack {
                // Fill surface
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .fill(Theme.Palette.wellSink)

                // Border: focus ring → accent 70%, error → error 70%, default → white 8%
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .strokeBorder(borderColor, lineWidth: isFocused ? 1.5 : 1)

                // Focus outer ring (2pt, accent 55%)
                if isFocused && errorMessage == nil {
                    RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
                        .strokeBorder(Theme.Palette.focusRing, lineWidth: 2)
                        .padding(-3)
                }

                TextField(prompt, text: $text)
                    .textFieldStyle(.plain)
                    .font(Theme.Fonts.body)
                    .foregroundStyle(Theme.Palette.ink)
                    .focused($isFocused)
                    .padding(.horizontal, 8)
            }
            .frame(height: height)
            .animation(.easeOut(duration: Theme.Motion.hover), value: isFocused)

            // Error caption
            if let msg = errorMessage {
                Text(msg)
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.error)
            }
        }
    }

    private var borderColor: Color {
        if let _ = errorMessage { return Theme.Palette.error.opacity(0.70) }
        if isFocused { return Theme.Palette.accent.opacity(0.70) }
        return Theme.Palette.borderStandard
    }
}

// MARK: - LatticeNumericField

/// A styled numeric text field: SF Mono, right-aligned, fixed width.
///
/// ```swift
/// @State private var steps = "500"
/// LatticeNumericField("", text: $steps)
///
/// // Wider (e.g., for seeds):
/// LatticeNumericField("Seed", text: $seed, width: 120)
/// ```
struct LatticeNumericField: View {
    let prompt: String
    @Binding var text: String
    var width: CGFloat = 80
    var errorMessage: String? = nil
    var height: CGFloat = 30

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .trailing, spacing: 3) {
            ZStack {
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .fill(Theme.Palette.wellSink)

                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .strokeBorder(borderColor, lineWidth: isFocused ? 1.5 : 1)

                if isFocused && errorMessage == nil {
                    RoundedRectangle(cornerRadius: Theme.Radius.control + 2, style: .continuous)
                        .strokeBorder(Theme.Palette.focusRing, lineWidth: 2)
                        .padding(-3)
                }

                TextField(prompt, text: $text)
                    .textFieldStyle(.plain)
                    .font(Theme.Fonts.tableNumeric)
                    .monospacedDigit()
                    .foregroundStyle(Theme.Palette.ink)
                    .multilineTextAlignment(.trailing)
                    .focused($isFocused)
                    .padding(.horizontal, 8)
            }
            .frame(width: width, height: height)
            .animation(.easeOut(duration: Theme.Motion.hover), value: isFocused)

            if let msg = errorMessage {
                Text(msg)
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.error)
            }
        }
    }

    private var borderColor: Color {
        if let _ = errorMessage { return Theme.Palette.error.opacity(0.70) }
        if isFocused { return Theme.Palette.accent.opacity(0.70) }
        return Theme.Palette.borderStandard
    }
}

// MARK: - Previews

#Preview("Field") {
    VStack(alignment: .leading, spacing: Theme.Space.md) {
        Text("TEXT FIELD")
            .instrumentLabel()

        LatticeField(prompt: "Model name or path", text: .constant(""))

        LatticeField(prompt: "Dataset path", text: .constant("/Users/lion/data/train.jsonl"))

        LatticeField(prompt: "With error", text: .constant("bad-path"), errorMessage: "File not found")

        Divider()
            .padding(.vertical, Theme.Space.xs)

        Text("NUMERIC FIELD")
            .instrumentLabel()

        HStack(alignment: .top, spacing: Theme.Space.sm) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Steps")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textSecondary)
                LatticeNumericField(prompt: "500", text: .constant("500"))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Rank")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textSecondary)
                LatticeNumericField(prompt: "8", text: .constant("8"), width: 60)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("LR")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textSecondary)
                LatticeNumericField(prompt: "5e-4", text: .constant("5e-4"), width: 88)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Seed (error)")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textSecondary)
                LatticeNumericField(prompt: "", text: .constant("abc"), width: 88, errorMessage: "Must be integer")
            }
        }
    }
    .padding(Theme.Space.lg)
    .frame(width: 480)
    .background(Theme.Palette.canvas)
}
