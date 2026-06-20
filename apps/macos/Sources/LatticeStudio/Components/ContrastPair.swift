import SwiftUI

// MARK: - Primitive 9: ContrastPair
//
// Before↔after comparator for the QuaRot quantization story.
//
// Design spec:
//   - Two stacked ReadoutWell columns (size / bits / PPL) with a centered Δ chip
//   - On completion: columns slide apart + a single hairline "fold" wipe reveals after-state
//   - The Δ chip encodes direction: teal = improvement, amber = regression
//   - Animation: 180ms ease-out (mechanical), respects accessibilityReduceMotion

/// A single metric row within the ContrastPair.
struct ContrastMetric: Identifiable {
    let id: String
    let label: String
    let beforeValue: String
    let afterValue: String
    let beforeUnit: String
    let afterUnit: String
    let delta: String          // e.g. "−74%", "+0.09"
    let deltaGood: Bool        // teal if true, amber if false (good = smaller for size, bigger for PPL budget)

    init(
        label: String,
        beforeValue: String,
        afterValue: String,
        beforeUnit: String = "",
        afterUnit: String = "",
        delta: String,
        deltaGood: Bool = true
    ) {
        self.id = label
        self.label = label
        self.beforeValue = beforeValue
        self.afterValue = afterValue
        self.beforeUnit = beforeUnit
        self.afterUnit = afterUnit
        self.delta = delta
        self.deltaGood = deltaGood
    }
}

/// Before↔after comparator: two ReadoutWell columns + centered Δ chip.
/// On `isComplete = true` the after-column slides in with a "fold" wipe.
///
/// ```swift
/// ContrastPair(
///     metrics: [
///         ContrastMetric(label: "SIZE", beforeValue: "1.61", afterValue: "0.41",
///                        beforeUnit: "GB", afterUnit: "GB", delta: "−74%", deltaGood: true),
///         ContrastMetric(label: "BITS", beforeValue: "16", afterValue: "4",
///                        delta: "−75%", deltaGood: true),
///         ContrastMetric(label: "PPL", beforeValue: "15.86", afterValue: "15.95",
///                        delta: "+0.09", deltaGood: false),
///     ],
///     isComplete: $isComplete
/// )
/// ```
struct ContrastPair: View {
    let metrics: [ContrastMetric]
    @Binding var isComplete: Bool

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var revealProgress: Double = 0.0  // 0→1 drives the fold wipe

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            // BEFORE column
            beforeColumn
                .frame(maxWidth: .infinity)

            // Center fold divider + Δ chips
            deltaColumn
                .frame(width: 80)

            // AFTER column — revealed by fold wipe
            afterColumn
                .frame(maxWidth: .infinity)
                .opacity(isComplete ? 1.0 : 0.0)
                .offset(x: isComplete ? 0 : 20)
                .clipped()
        }
        .onChange(of: isComplete) { _, newValue in
            if newValue {
                if reduceMotion {
                    revealProgress = 1.0
                } else {
                    withAnimation(.easeOut(duration: Theme.Motion.focus)) {
                        revealProgress = 1.0
                    }
                }
            } else {
                revealProgress = 0.0
            }
        }
    }

    private var beforeColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("BEFORE")
                .instrumentLabel()
                .padding(.horizontal, Theme.Space.md)
                .padding(.top, Theme.Space.sm)
                .padding(.bottom, Theme.Space.xs)

            ForEach(metrics) { metric in
                VStack(alignment: .leading, spacing: 2) {
                    Text(metric.label)
                        .instrumentLabel()
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        Text(metric.beforeValue)
                            .font(Theme.Fonts.wellValue)
                            .foregroundStyle(Theme.Palette.ink)
                            .monospacedDigit()
                        if !metric.beforeUnit.isEmpty {
                            Text(metric.beforeUnit)
                                .font(Theme.Fonts.cell)
                                .foregroundStyle(Theme.Palette.inkDim)
                        }
                    }
                }
                .padding(.horizontal, Theme.Space.md)
                .padding(.vertical, Theme.Space.sm)
                .overlay(alignment: .bottom) {
                    Theme.Palette.hairline.frame(height: 1)
                }
            }
        }
        .readoutWellSurface()
    }

    private var afterColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("AFTER")
                .instrumentLabel()
                .padding(.horizontal, Theme.Space.md)
                .padding(.top, Theme.Space.sm)
                .padding(.bottom, Theme.Space.xs)

            ForEach(metrics) { metric in
                VStack(alignment: .leading, spacing: 2) {
                    Text(metric.label)
                        .instrumentLabel()
                    HStack(alignment: .firstTextBaseline, spacing: 4) {
                        Text(metric.afterValue)
                            .font(Theme.Fonts.wellValue)
                            .foregroundStyle(Theme.Palette.ink)
                            .monospacedDigit()
                            .contentTransition(reduceMotion ? .identity : .numericText())
                        if !metric.afterUnit.isEmpty {
                            Text(metric.afterUnit)
                                .font(Theme.Fonts.cell)
                                .foregroundStyle(Theme.Palette.inkDim)
                        }
                    }
                }
                .padding(.horizontal, Theme.Space.md)
                .padding(.vertical, Theme.Space.sm)
                .overlay(alignment: .bottom) {
                    Theme.Palette.hairline.frame(height: 1)
                }
            }
        }
        .readoutWellSurface()
    }

    private var deltaColumn: some View {
        VStack(alignment: .center, spacing: 0) {
            // Header spacer to align with before/after columns
            Text("──fold──▶")
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim.opacity(isComplete ? 0.0 : 1.0))
                .padding(.horizontal, Theme.Space.xs)
                .padding(.top, Theme.Space.sm)
                .padding(.bottom, Theme.Space.xs)
                .animation(.easeOut(duration: Theme.Motion.focus), value: isComplete)

            ForEach(metrics) { metric in
                // Δ chip
                let deltaColor = metric.deltaGood ? Theme.Palette.signal : Theme.Palette.amber
                Text(metric.delta)
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(deltaColor)
                    .monospacedDigit()
                    .padding(.horizontal, Theme.Space.xs)
                    .padding(.vertical, Theme.Space.sm)
                    .frame(height: 52)  // approximate height of a metric row
                    .overlay(alignment: .bottom) {
                        Theme.Palette.hairline.frame(height: 1)
                    }
            }
        }
    }
}

// MARK: - Previews

#Preview("ContrastPair") {
    @State var isComplete: Bool = false

    return VStack(spacing: Theme.Space.lg) {
        ContrastPair(
            metrics: [
                ContrastMetric(
                    label: "SIZE",
                    beforeValue: "1.61", afterValue: "0.41",
                    beforeUnit: "GB", afterUnit: "GB",
                    delta: "−74%", deltaGood: true
                ),
                ContrastMetric(
                    label: "BITS",
                    beforeValue: "16", afterValue: "4",
                    delta: "−75%", deltaGood: true
                ),
                ContrastMetric(
                    label: "PPL (est.)",
                    beforeValue: "15.86", afterValue: "15.95",
                    delta: "+0.09", deltaGood: false
                ),
            ],
            isComplete: Binding(get: { isComplete }, set: { isComplete = $0 })
        )
        .padding(Theme.Space.lg)
        .instrumentPanel()

        Button(isComplete ? "Reset" : "Complete (fold wipe)") {
            isComplete.toggle()
        }
        .font(Theme.Fonts.body)
        .foregroundStyle(Theme.Palette.signal)
    }
    .padding(Theme.Space.xl)
    .background(Theme.Palette.canvas)
    .frame(width: 480)
}
