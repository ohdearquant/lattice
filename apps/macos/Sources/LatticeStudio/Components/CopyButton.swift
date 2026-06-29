import SwiftUI
import AppKit

// MARK: - CopyButton
//
// A small, quiet copy-to-clipboard affordance. Label flips to "Copied" for 1.2 s then reverts.
// Styled as a compact secondary button (LatticeSecondaryButtonStyle at .small control size) so it
// sits unobtrusively beside code blocks and in the CLI footer.

struct CopyButton: View {
    let text: String
    var label: String

    init(_ text: String, label: String = "Copy") {
        self.text = text
        self.label = label
    }

    @State private var copied = false

    var body: some View {
        Button(copied ? "Copied" : label) {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
            guard !copied else { return }
            copied = true
            Task {
                try? await Task.sleep(nanoseconds: 1_200_000_000)
                copied = false
            }
        }
        .buttonStyle(LatticeSecondaryButtonStyle(height: Theme.Space.controlHeightCompact))
        .controlSize(.small)
        .animation(.easeOut(duration: Theme.Motion.tick), value: copied)
    }
}

// MARK: - CopyableCodeBlock
//
// A titled code block with a copy button in its header and monospaced code inside a recessed well.
// Intended for multi-line CLI commands, curl examples, and SDK snippets.
//
// Layout (top to bottom):
//   header HStack — section label (title) + spacer + CopyButton
//   code text on wellSink background (.readoutWellSurface())
//
// The code area grows vertically with content (fixedSize horizontal:false, vertical:true).

struct CopyableCodeBlock: View {
    let title: String
    let code: String

    init(title: String, code: String) {
        self.title = title
        self.code = code
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            // Header
            HStack(spacing: Theme.Space.sm) {
                Text(title)
                    .font(Theme.Fonts.micro)
                    .textCase(.uppercase)
                    .tracking(0.45)
                    .foregroundStyle(Theme.Palette.textTertiary)
                Spacer(minLength: Theme.Space.sm)
                CopyButton(code)
            }

            // Code well
            Text(code)
                .font(Theme.Fonts.codeFont)
                .foregroundStyle(Theme.Palette.textSecondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, Theme.Space.md)
                .padding(.vertical, Theme.Space.sm)
                .readoutWellSurface()
        }
    }
}

// MARK: - Previews

#Preview("CopyButton") {
    HStack(spacing: Theme.Space.md) {
        CopyButton("lattice chat qwen3.5-0.8b --temperature 0.7")
        CopyButton("lattice chat qwen3.5-0.8b --temperature 0.7", label: "Copy command")
    }
    .padding(Theme.Space.lg)
    .background(Theme.Palette.canvas)
}

#Preview("CopyableCodeBlock") {
    VStack(spacing: Theme.Space.md) {
        CopyableCodeBlock(
            title: "CLI",
            code: "lattice chat qwen3.5-0.8b --think --reasoning-budget 1024 --temperature 0.7 --top-k 50 --top-p 0.9"
        )
        CopyableCodeBlock(
            title: "curl",
            code: """
            curl http://127.0.0.1:11435/v1/chat/completions \\
              -H "Content-Type: application/json" \\
              -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"hi"}]}'
            """
        )
    }
    .padding(Theme.Space.lg)
    .background(Theme.Palette.canvas)
    .frame(width: 560)
}
