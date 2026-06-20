import SwiftUI

// MARK: - Primitive 2: ReadoutWell
//
// The atomic readout unit:
//   - 11pt all-caps dim label (LOSS, TOK/S, GRAD-NORM…)
//   - Tabular-mono value at Theme.Fonts.wellValue (15pt medium)
//   - Unit string (dim, 11pt)
//   - Optional delta caret (▲/▼): teal = good (falling), amber = bad (rising)
//   - All numerals use .monospacedDigit()
//   - Recessed machined-in well surface

/// Direction of movement for the delta caret.
enum DeltaDirection {
    case down  // value falling — teal (good for loss/PPL)
    case up    // value rising — amber (bad for loss/PPL; good for tok/s)
    case none

    var glyph: String {
        switch self {
        case .down: "▼"
        case .up: "▲"
        case .none: ""
        }
    }

    var color: Color {
        switch self {
        case .down: Theme.Palette.signal
        case .up: Theme.Palette.amber
        case .none: .clear
        }
    }
}

/// A single readout well: label + tabular-mono value + unit + optional delta caret.
///
/// ```swift
/// ReadoutWell(label: "LOSS", value: "0.6121", unit: "", delta: .init("▼0.004", .down))
/// ReadoutWell(label: "TOK/S", value: "1820", unit: "t/s")
/// ReadoutWell(label: "LR-NOW", value: "1.81e-4")
/// ```
struct ReadoutWell: View {
    let label: String
    let value: String
    let unit: String
    let delta: DeltaInfo?

    struct DeltaInfo {
        let text: String
        let direction: DeltaDirection

        init(_ text: String, _ direction: DeltaDirection) {
            self.text = text
            self.direction = direction
        }
    }

    init(label: String, value: String, unit: String = "", delta: DeltaInfo? = nil) {
        self.label = label
        self.value = value
        self.unit = unit
        self.delta = delta
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            // 11pt all-caps dim label
            Text(label)
                .instrumentLabel()

            HStack(alignment: .firstTextBaseline, spacing: 4) {
                // Tabular-mono value — .monospacedDigit() ensures fixed-width digits
                Text(value)
                    .font(Theme.Fonts.wellValue)
                    .foregroundStyle(Theme.Palette.ink)
                    .monospacedDigit()
                    .contentTransition(.numericText())

                // Unit (dim, 11pt)
                if !unit.isEmpty {
                    Text(unit)
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                }

                // Delta caret
                if let delta = delta, delta.direction != .none {
                    Text("\(delta.direction.glyph)\(delta.text)")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(delta.direction.color)
                        .monospacedDigit()
                        .animation(.easeOut(duration: Theme.Motion.tick), value: delta.direction.glyph)
                }
            }
        }
        .padding(.horizontal, Theme.Space.md)
        .padding(.vertical, Theme.Space.sm)
        .readoutWellSurface()
    }
}

// MARK: - Previews

#Preview("ReadoutWell") {
    HStack(spacing: Theme.Space.sm) {
        ReadoutWell(
            label: "LOSS",
            value: "0.6121",
            unit: "",
            delta: ReadoutWell.DeltaInfo("0.004", .down)
        )
        ReadoutWell(
            label: "TOK/S",
            value: "1820",
            unit: "t/s",
            delta: ReadoutWell.DeltaInfo("12", .up)
        )
        ReadoutWell(
            label: "LR-NOW",
            value: "1.81e-4"
        )
        ReadoutWell(
            label: "GRAD",
            value: "0.93"
        )
        ReadoutWell(
            label: "ETA",
            value: "4m 12s"
        )
    }
    .padding(Theme.Space.lg)
    .instrumentPanel()
    .background(Theme.Palette.canvas)
}
