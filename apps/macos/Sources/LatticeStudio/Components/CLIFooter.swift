import SwiftUI

// MARK: - CLIFooter
//
// A full-width strip that sits at the very bottom of the main column showing the exact
// `lattice ...` CLI command equivalent to the current GUI configuration — an honesty feature
// ("the GUI shows what the Rust engine actually receives").
//
// Layout (left to right, inside a horizontal padding of Theme.Space.xl):
//   "CLI" chip · command text (grows, truncates) · CopyButton · optional caption
//
// A top hairline Rule separates it from the tab body above.
// Background: Theme.Palette.canvas so it blends with the column floor.

struct CLIFooter: View {
    let command: String
    var caption: String?

    init(command: String, caption: String? = nil) {
        self.command = command
        self.caption = caption
    }

    var body: some View {
        VStack(spacing: 0) {
            // Top hairline — separates from the tab body above.
            Rectangle()
                .fill(Theme.Palette.hairline)
                .frame(height: 1)

            HStack(spacing: Theme.Space.sm) {
                // "CLI" label chip — tiny bordered box, reads as metadata not action.
                Text("CLI")
                    .font(Theme.Fonts.micro)
                    .textCase(.uppercase)
                    .tracking(0.45)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .padding(.horizontal, Theme.Space.xs)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.Radius.badge, style: .continuous)
                            .fill(Theme.Palette.wellSink)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.badge, style: .continuous)
                            .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
                    )

                // Command text — truncates if the window is narrow; selectable for manual copy.
                Text(command)
                    .font(Theme.Fonts.codeFont)
                    .foregroundStyle(Theme.Palette.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .textSelection(.enabled)
                    .layoutPriority(1)

                // Copy button.
                CopyButton(command)

                // Caption — right-aligned metadata (endpoint URL, backend label).
                if let caption = caption {
                    Spacer(minLength: Theme.Space.md)
                    Text(caption)
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
            .padding(.horizontal, Theme.Space.xl)
            .padding(.vertical, Theme.Space.sm)
            .frame(minHeight: 40)
        }
        .background(Theme.Palette.canvas)
    }
}

// MARK: - Previews

#Preview("CLIFooter – chat") {
    VStack(spacing: 0) {
        Spacer()
        CLIFooter(
            command: "lattice chat qwen3.6-27b-q4 --think --reasoning-budget 1024 --temperature 0.7 --top-k 50 --top-p 0.9",
            caption: "streaming from local subprocess"
        )
    }
    .frame(width: 760, height: 160)
    .background(Theme.Palette.canvas)
}

#Preview("CLIFooter – serve") {
    VStack(spacing: 0) {
        Spacer()
        CLIFooter(
            command: "lattice serve qwen3.5-0.8b --host 127.0.0.1 --port 11435",
            caption: "http://127.0.0.1:11435/v1"
        )
    }
    .frame(width: 760, height: 160)
    .background(Theme.Palette.canvas)
}
