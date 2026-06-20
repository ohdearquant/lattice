import SwiftUI

// MARK: - Primitive 6: GatePill
//
// Status capsule encoding the verdict of any measurable action.
//
// States:
//   PASS   — teal fill
//   WARN   — amber fill
//   FAIL   — crimson fill
//   RUN    — animated teal pulse (running job)
//
// Design spec:
//   - Corner radius: Theme.Radius.pill = 6px
//   - Mono text (Theme.Fonts.cell, 11pt)
//   - Background fills use signal/amber/crimson
//   - RUN state: repeating opacity animation (pulse)
//   - Respects accessibilityReduceMotion

/// The verdict of a measurable gate.
enum GateStatus {
    case pass
    case warn
    case fail
    case run

    var label: String {
        switch self {
        case .pass: "PASS"
        case .warn: "WARN"
        case .fail: "FAIL"
        case .run: "RUN"
        }
    }

    var fillColor: Color {
        switch self {
        case .pass: Theme.Palette.signal
        case .warn: Theme.Palette.amber
        case .fail: Theme.Palette.crimson
        case .run: Theme.Palette.signal
        }
    }

    // High-contrast text: canvas (dark/light) on the status fill
    var textColor: Color { Theme.Palette.canvas }
}

/// A PASS / WARN / FAIL / RUN status capsule.
///
/// ```swift
/// GatePill(.pass)
/// GatePill(.warn, label: "ΔPPL +0.09")
/// GatePill(.run, label: "step 420/1000")
/// GatePill(.fail, label: "NaN loss")
/// ```
struct GatePill: View {
    let status: GateStatus
    let label: String

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulseOpacity: Double = 1.0

    init(_ status: GateStatus, label: String? = nil) {
        self.status = status
        self.label = label ?? status.label
    }

    var body: some View {
        HStack(spacing: 4) {
            // Pulse indicator dot for RUN state
            if status == .run {
                Circle()
                    .fill(status.textColor)
                    .frame(width: 5, height: 5)
                    .opacity(pulseOpacity)
                    .onAppear {
                        guard !reduceMotion else { return }
                        withAnimation(
                            .easeInOut(duration: 0.7).repeatForever(autoreverses: true)
                        ) {
                            pulseOpacity = 0.3
                        }
                    }
            }

            Text(label)
                .font(Theme.Fonts.cell)
                .foregroundStyle(status.textColor)
                .monospacedDigit()
                .textCase(.uppercase)
        }
        .padding(.horizontal, Theme.Space.sm)
        .padding(.vertical, 3)
        .background(
            status == .run
                ? status.fillColor.opacity(pulseOpacity * 0.7 + 0.3)
                : status.fillColor
        )
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.pill, style: .continuous))
    }
}

// MARK: - Previews

#Preview("GatePill") {
    VStack(spacing: Theme.Space.sm) {
        HStack(spacing: Theme.Space.sm) {
            GatePill(.pass)
            GatePill(.warn)
            GatePill(.fail)
            GatePill(.run)
        }

        HStack(spacing: Theme.Space.sm) {
            GatePill(.pass, label: "Q4 ok 405MB −74%")
            GatePill(.warn, label: "ΔPPL +0.09")
            GatePill(.fail, label: "NaN loss @ step 840")
            GatePill(.run, label: "step 420/1000")
        }

        HStack(spacing: Theme.Space.sm) {
            GatePill(.pass, label: "eval PASS · best val 1.94")
            GatePill(.warn, label: "grad spike · norm 4.7")
        }
    }
    .padding(Theme.Space.lg)
    .instrumentPanel()
    .background(Theme.Palette.canvas)
}
