import SwiftUI

// MARK: - Primitive 3: HeroNumber
//
// The visual anchor of TRAIN and QUANTIZE screens.
//   - Big tabular-mono value (Theme.Fonts.hero = 56pt Bold by default)
//   - Size param: .hero (56pt) or .heroAlt (34pt) or .heroMinor (21pt)
//   - Small mono unit below/beside the value
//   - 1px teal under-rule beneath the number
//   - Per-digit tick via .contentTransition(.numericText())
//   - .monospacedDigit() always

/// Preset sizes for HeroNumber, matching the modular scale.
enum HeroSize {
    case hero      // 56pt Bold — the loss / compression / PPL headline
    case heroAlt   // 34pt Bold — secondary hero
    case heroMinor // 21pt Semibold — tertiary large numeral

    var font: Font {
        switch self {
        case .hero: Theme.Fonts.hero
        case .heroAlt: Theme.Fonts.heroAlt
        case .heroMinor: Theme.Fonts.heroMinor
        }
    }
}

/// A big tabular-mono numeral with a unit label and a 1px teal under-rule.
/// Ticks per-digit via `.contentTransition(.numericText())`.
///
/// ```swift
/// HeroNumber(value: "0.6121", unit: "LOSS")
/// HeroNumber(value: "3.97×", unit: "SMALLER", size: .heroAlt)
/// HeroNumber(value: "405", unit: "MB", size: .heroAlt, unitPosition: .trailing)
/// ```
struct HeroNumber: View {
    let value: String
    let unit: String
    var size: HeroSize = .hero
    var unitPosition: UnitPosition = .below

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    enum UnitPosition {
        case below     // unit appears below the numeral (default)
        case trailing  // unit appears to the right at baseline
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if unitPosition == .trailing {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    numeralText
                    unitText
                }
            } else {
                numeralText
                if !unit.isEmpty {
                    unitText
                        .padding(.top, 2)
                }
            }

            // 1px teal under-rule
            Theme.Palette.signal
                .frame(height: 1)
                .padding(.top, 4)
        }
    }

    private var numeralText: some View {
        Text(value)
            .font(size.font)
            .foregroundStyle(Theme.Palette.ink)
            .monospacedDigit()
            .contentTransition(
                reduceMotion ? .identity : .numericText()
            )
            .animation(
                reduceMotion
                    ? nil
                    : .easeOut(duration: Theme.Motion.tick),
                value: value
            )
    }

    private var unitText: some View {
        Text(unit)
            .font(Theme.Fonts.cell)
            .foregroundStyle(Theme.Palette.inkDim)
            .textCase(.uppercase)
            .tracking(Theme.Space.labelTracking)
    }
}

// MARK: - Previews

#Preview("HeroNumber") {
    VStack(alignment: .leading, spacing: Theme.Space.xl) {
        // 56pt hero — primary loss display
        HeroNumber(value: "0.6121", unit: "LOSS")

        // 34pt heroAlt — compression ratio
        HeroNumber(value: "3.97×", unit: "SMALLER", size: .heroAlt)

        // 21pt heroMinor — secondary stat
        HeroNumber(value: "405", unit: "MB", size: .heroMinor, unitPosition: .trailing)

        // Trailing unit
        HeroNumber(value: "15.95", unit: "PPL", size: .heroAlt, unitPosition: .trailing)
    }
    .padding(Theme.Space.xl)
    .instrumentPanel()
    .background(Theme.Palette.canvas)
    .frame(width: 320)
}
