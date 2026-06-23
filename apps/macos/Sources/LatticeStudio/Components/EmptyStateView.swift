import SwiftUI

// MARK: - EmptyStateView
//
// Graphite Signal Lab empty state — Section D "Empty states".
//
// Design laws (spec §D):
//   - NO enclosing border around the whole empty state.
//   - Max width 360pt (Theme.Space.emptyStateMaxWidth).
//   - Symbol: 28pt, textTertiary.
//   - Title: 15pt semibold (Theme.Fonts.display(15,.semibold)), ink.
//   - Body: 12pt secondary (Theme.Fonts.caption / textSecondary), 18pt line target.
//   - Optional primary action button 16pt below body (LatticePrimaryButtonStyle).
//   - Caller is responsible for placing this view at ~38–42% usable height.
//   - `ContentUnavailableView` is usable but limited; custom VStack is preferred (spec note).

/// A screen-specific empty state: symbol, title, body, optional primary action.
///
/// ```swift
/// EmptyStateView(
///     systemImage: "clock.arrow.circlepath",
///     title: "No runs yet",
///     message: "Training, quantization, and model tests are recorded here."
/// )
///
/// // With action:
/// EmptyStateView(
///     systemImage: "tablecells",
///     title: "No dataset scanned",
///     message: "Choose a directory containing JSONL, JSON, or supported dataset files.",
///     actionLabel: "Choose Directory"
/// ) {
///     openPanel()
/// }
/// ```
struct EmptyStateView: View {
    let systemImage: String
    let title: String
    let message: String
    var actionLabel: String? = nil
    var action: (() -> Void)? = nil

    var body: some View {
        VStack(spacing: 0) {
            // Symbol — 28pt, textTertiary
            Image(systemName: systemImage)
                .font(.system(size: 28, weight: .regular))
                .foregroundStyle(Theme.Palette.textTertiary)

            // Title — 15pt semibold, ink
            Text(title)
                .font(Theme.Fonts.display(15, .semibold))
                .foregroundStyle(Theme.Palette.ink)
                .multilineTextAlignment(.center)
                .padding(.top, Theme.Space.space3)   // 12pt

            // Body — 12pt regular, secondary, ~18pt line target via lineSpacing
            Text(message)
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textSecondary)
                .multilineTextAlignment(.center)
                .lineSpacing(4)   // 11pt font + 4pt spacing ≈ 15pt leading; close to 18pt target
                .padding(.top, Theme.Space.space2)   // 8pt

            // Optional primary action — 16pt separation below body
            if let label = actionLabel, let handler = action {
                Button(label, action: handler)
                    .buttonStyle(LatticePrimaryButtonStyle())
                    .padding(.top, Theme.Space.space4)  // 16pt
            }
        }
        .frame(maxWidth: Theme.Space.emptyStateMaxWidth)
    }
}

// MARK: - Previews

#Preview("EmptyStateView") {
    VStack(spacing: Theme.Space.xl) {
        // No action
        EmptyStateView(
            systemImage: "clock.arrow.circlepath",
            title: "No runs yet",
            message: "Training, quantization, and model tests are recorded here."
        )

        Divider()

        // With primary action
        EmptyStateView(
            systemImage: "tablecells",
            title: "No dataset scanned",
            message: "Choose a directory containing JSONL, JSON, or supported dataset files.",
            actionLabel: "Choose Directory"
        ) {
            // action placeholder
        }

        Divider()

        // Chat empty state
        EmptyStateView(
            systemImage: "text.bubble",
            title: "Model ready",
            message: "Ask a question or run an A/B adapter comparison.",
            actionLabel: "Start Chat"
        ) {}
    }
    .padding(Theme.Space.xl)
    .frame(width: 600, height: 700)
    .background(Theme.Palette.canvas)
}
