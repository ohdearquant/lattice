import SwiftUI

// MARK: - Badges
//
// Graphite Signal Lab badge vocabulary — Section D "Badges and pills".
//
// Design laws (spec §D):
//   - FormatBadge: height 20, h-pad 7, radius 4 (Theme.Radius.badge), 11pt medium, neutral fill/border.
//     Avoid assigning a different bright color to every model format.
//   - StatusBadge: height 22, 6pt semantic dot, ink text (AA on the tinted fill),
//     fill=semantic@12%, border=semantic@28%.
//     Only Running gets motion: opacity pulse 65%→100% over 1.2s (spec §F).

// MARK: - FormatBadge

/// Neutral pill labelling a model format: BF16, Q4, EMBED, ADAPTER, etc.
///
/// ```swift
/// FormatBadge("BF16")
/// FormatBadge("Q4")
/// FormatBadge("EMBED")
/// ```
struct FormatBadge: View {
    let label: String

    init(_ label: String) {
        self.label = label
    }

    var body: some View {
        Text(label)
            .font(Theme.Fonts.micro)
            .tracking(0.2)
            .foregroundStyle(Theme.Palette.inkDim)
            .padding(.horizontal, 7)
            .frame(height: 20)
            .background(Theme.Palette.surfaceRaised)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.badge, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.badge, style: .continuous)
                    .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
            )
    }
}

// MARK: - StatusBadge

/// Semantic status indicator: colored dot + label, fill/border at semantic opacity.
///
/// ```swift
/// StatusBadge(.idle)
/// StatusBadge(.running)
/// StatusBadge(.success)
/// StatusBadge(.warning)
/// StatusBadge(.error)
/// ```
struct StatusBadge: View {
    enum Status {
        case idle
        case running
        case success
        case warning
        case error

        var label: String {
            switch self {
            case .idle:    "Idle"
            case .running: "Running"
            case .success: "Complete"
            case .warning: "Warning"
            case .error:   "Failed"
            }
        }

        var color: Color {
            switch self {
            case .idle:    Theme.Palette.idle
            case .running: Theme.Palette.running
            case .success: Theme.Palette.success
            case .warning: Theme.Palette.warning
            case .error:   Theme.Palette.error
            }
        }

        var isRunning: Bool { self == .running }
    }

    let status: Status

    @State private var dotOpacity: Double = 1.0

    init(_ status: Status) {
        self.status = status
    }

    var body: some View {
        HStack(spacing: 6) {
            // 6pt semantic dot — Running pulses 65%→100% over 1.2s
            Circle()
                .fill(status.color)
                .frame(width: 6, height: 6)
                .opacity(dotOpacity)
                .onAppear {
                    guard status.isRunning else { return }
                    withAnimation(
                        .easeInOut(duration: 1.2)
                            .repeatForever(autoreverses: true)
                    ) {
                        dotOpacity = 0.65
                    }
                }
                .onChange(of: status.isRunning) { _, nowRunning in
                    if nowRunning {
                        withAnimation(
                            .easeInOut(duration: 1.2)
                                .repeatForever(autoreverses: true)
                        ) {
                            dotOpacity = 0.65
                        }
                    } else {
                        withAnimation(.easeOut(duration: Theme.Motion.metric)) {
                            dotOpacity = 1.0
                        }
                    }
                }

            Text(status.label)
                .font(Theme.Fonts.controlText)
                .foregroundStyle(Theme.Palette.ink)
        }
        .padding(.horizontal, 8)
        .frame(height: 22)
        .background(status.color.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.badge, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.badge, style: .continuous)
                .strokeBorder(status.color.opacity(0.28), lineWidth: 1)
        )
    }
}

// MARK: - Previews

#Preview("Badges") {
    VStack(alignment: .leading, spacing: Theme.Space.md) {
        Text("FORMAT BADGES")
            .instrumentLabel()

        HStack(spacing: Theme.Space.sm) {
            FormatBadge("BF16")
            FormatBadge("Q4")
            FormatBadge("EMBED")
            FormatBadge("ADAPTER")
            FormatBadge("F32")
        }

        Divider()
            .padding(.vertical, Theme.Space.xs)

        Text("STATUS BADGES")
            .instrumentLabel()

        HStack(spacing: Theme.Space.sm) {
            StatusBadge(.idle)
            StatusBadge(.running)
            StatusBadge(.success)
            StatusBadge(.warning)
            StatusBadge(.error)
        }
    }
    .padding(Theme.Space.lg)
    .background(Theme.Palette.canvas)
}
