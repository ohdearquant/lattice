import SwiftUI

// MARK: - Primitive 10: MassBars
//
// Two horizontal bars to TRUE scale (before = inkDim, after = teal).
// The compression ratio counts up into a HeroNumber while bars animate to length.
//
// Design spec:
//   - fp16 "before" bar: Theme.Palette.inkDim fill
//   - Q4 "after" bar: Theme.Palette.signal fill
//   - Both bars share the same MAX width (true scale: wider = larger size)
//   - Ratio animates as a HeroNumber while bars grow
//   - 180ms ease-out for bar extension; respect accessibilityReduceMotion

/// Two true-scale horizontal mass bars with an animated compression ratio hero.
///
/// ```swift
/// MassBars(
///     beforeLabel: "fp16",
///     beforeMB: 1648,
///     afterLabel: "Q4",
///     afterMB: 405,
///     ratioLabel: "SMALLER"
/// )
/// ```
struct MassBars: View {
    let beforeLabel: String
    let beforeMB: Double
    let afterLabel: String
    let afterMB: Double
    var ratioLabel: String = "SMALLER"
    var animated: Bool = true

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var animationProgress: Double = 0.0  // 0→1

    private var ratio: Double {
        beforeMB > 0 ? beforeMB / afterMB : 1.0
    }

    private var displayedRatio: Double {
        1.0 + (ratio - 1.0) * animationProgress
    }

    private var afterFraction: Double {
        beforeMB > 0 ? afterMB / beforeMB : 1.0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            // Hero ratio number (counts up as bars animate)
            HeroNumber(
                value: String(format: "%.2f×", displayedRatio),
                unit: ratioLabel,
                size: .heroAlt,
                unitPosition: .trailing
            )
            .padding(.bottom, Theme.Space.xs)

            // BEFORE bar (full width = reference)
            barRow(
                label: beforeLabel,
                size: beforeMB,
                fraction: 1.0,
                color: Theme.Palette.inkDim,
                progress: animationProgress
            )

            // AFTER bar (proportionally narrower)
            barRow(
                label: afterLabel,
                size: afterMB,
                fraction: afterFraction,
                color: Theme.Palette.signal,
                progress: animationProgress
            )
        }
        .onAppear {
            if animated {
                if reduceMotion {
                    animationProgress = 1.0
                } else {
                    withAnimation(.easeOut(duration: Theme.Motion.focus)) {
                        animationProgress = 1.0
                    }
                }
            } else {
                animationProgress = 1.0
            }
        }
        .onChange(of: beforeMB) { _, _ in resetAndAnimate() }
        .onChange(of: afterMB) { _, _ in resetAndAnimate() }
    }

    private func resetAndAnimate() {
        animationProgress = 0.0
        if reduceMotion {
            animationProgress = 1.0
        } else {
            withAnimation(.easeOut(duration: Theme.Motion.focus)) {
                animationProgress = 1.0
            }
        }
    }

    private func barRow(
        label: String,
        size: Double,
        fraction: Double,
        color: Color,
        progress: Double
    ) -> some View {
        HStack(spacing: Theme.Space.sm) {
            // Label (fixed width)
            Text(label)
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
                .frame(width: 40, alignment: .trailing)

            // Bar track
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    // Track background
                    Rectangle()
                        .fill(Theme.Palette.wellSink)
                        .frame(maxWidth: .infinity)

                    // Filled portion — true scale, animated
                    Rectangle()
                        .fill(color)
                        .frame(width: geo.size.width * fraction * progress)
                }
            }
            .frame(height: 12)
            .clipShape(RoundedRectangle(cornerRadius: 2, style: .continuous))

            // Size label
            Text(formattedMB(size))
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
                .monospacedDigit()
                .frame(width: 64, alignment: .trailing)
        }
    }

    private func formattedMB(_ mb: Double) -> String {
        if mb >= 1024 {
            String(format: "%.2f GB", mb / 1024.0)
        } else {
            String(format: "%.0f MB", mb)
        }
    }
}

// MARK: - Previews

#Preview("MassBars") {
    VStack(spacing: Theme.Space.xl) {
        Text("MASS BARS — TRUE SCALE")
            .instrumentLabel()

        MassBars(
            beforeLabel: "fp16",
            beforeMB: 1648,
            afterLabel: "Q4",
            afterMB: 415
        )

        MassBars(
            beforeLabel: "fp16",
            beforeMB: 4096,
            afterLabel: "Q4",
            afterMB: 1024,
            ratioLabel: "COMPRESSED"
        )
    }
    .padding(Theme.Space.xl)
    .instrumentPanel()
    .background(Theme.Palette.canvas)
    .frame(width: 440)
}
