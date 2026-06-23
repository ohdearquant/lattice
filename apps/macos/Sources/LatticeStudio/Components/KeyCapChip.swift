import SwiftUI

// MARK: - Primitive 7: KeyCapChip
//
// A tiny 1px-outlined ⌘-cap on actionable elements so the keyboard map self-documents.
//
// Design:
//   - 1px hairline border (no fill — transparent background)
//   - Rounded rect corners: Theme.Radius.control = 6px
//   - Mono text: Theme.Fonts.cell (11pt)
//   - Foreground: Theme.Palette.inkDim (dimmed — decorative, not primary)
//   - Can show a modifier glyph (⌘, ⌥, ⇧, ^) + key, or just the key

/// A tiny outlined keyboard shortcut chip for self-documenting UI.
///
/// ```swift
/// KeyCapChip("⌘1")
/// KeyCapChip("⌘K")
/// KeyCapChip("⌘↵")
/// KeyCapChip("⌥A")
/// ```
struct KeyCapChip: View {
    let keyText: String

    init(_ keyText: String) {
        self.keyText = keyText
    }

    var body: some View {
        Text(keyText)
            .font(Theme.Fonts.cell)
            .foregroundStyle(Theme.Palette.inkDim)
            .monospacedDigit()
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(.clear)
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                    .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
            )
    }
}

// MARK: - Previews

#Preview("KeyCapChip") {
    VStack(alignment: .leading, spacing: Theme.Space.sm) {
        HStack(spacing: Theme.Space.sm) {
            Text("01 MODELS")
                .instrumentLabel()
            Spacer()
            KeyCapChip("⌘1")
        }
        HStack(spacing: Theme.Space.sm) {
            Text("02 TRAIN")
                .instrumentLabel()
            Spacer()
            KeyCapChip("⌘2")
        }
        HStack(spacing: Theme.Space.sm) {
            Text("Command bar")
                .instrumentLabel()
            Spacer()
            KeyCapChip("⌘K")
        }
        HStack(spacing: Theme.Space.sm) {
            Text("Start run")
                .instrumentLabel()
            Spacer()
            KeyCapChip("⌘↵")
        }
        HStack(spacing: Theme.Space.sm) {
            Text("Hot-swap adapter")
                .instrumentLabel()
            Spacer()
            KeyCapChip("⌥A")
        }
        HStack(spacing: Theme.Space.sm) {
            Text("Inspector")
                .instrumentLabel()
            Spacer()
            KeyCapChip("⌘\\")
        }
    }
    .padding(Theme.Space.lg)
    .instrumentPanel()
    .background(Theme.Palette.canvas)
    .frame(width: 260)
}
