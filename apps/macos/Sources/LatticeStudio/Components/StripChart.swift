import SwiftUI
import Charts

// MARK: - Primitive 4: StripChart (Oscilloscope)
//
// The live training oscilloscope.
//
// Visual spec:
//   - LineMark: 1.5px Theme.Palette.signal stroke
//   - AreaMark: Theme.Palette.signalGlow gradient beneath the line
//   - RuleMark: 1px teal now-cursor at the latest step (or scrub position)
//   - Ghost baseline series: dashed inkDim line (optional, for base PPL / comparison)
//   - Mono axis ticks, hairline grid, NO legend
//   - Draggable scrub line: reports hovered step via scrubStep binding
//   - Throttle: caller is responsible for throttling updates to ~20Hz (chartCommitHz)
//   - Respects accessibilityReduceMotion: instant updates (no chart animation)

/// Which value series to plot on the primary axis.
enum StripSeries {
    case loss
    case valLoss
    case gradNorm

    var label: String {
        switch self {
        case .loss: "LOSS"
        case .valLoss: "VAL LOSS"
        case .gradNorm: "GRAD NORM"
        }
    }

    func value(from point: TrainPoint) -> Double? {
        switch self {
        case .loss: point.loss
        case .valLoss: point.valLoss
        case .gradNorm: point.gradNorm
        }
    }
}

/// The oscilloscope strip chart for training metrics.
///
/// ```swift
/// StripChart(
///     points: store.liveRun?.points ?? [],
///     series: .loss,
///     scrubStep: $scrubStep
/// )
///
/// // With ghost baseline (quantize PPL ghost)
/// StripChart(
///     points: points,
///     series: .valLoss,
///     ghostBaseline: 15.86,
///     ghostLabel: "base PPL",
///     scrubStep: $scrubStep
/// )
/// ```
struct StripChart: View {
    let points: [TrainPoint]
    var series: StripSeries = .loss
    var ghostBaseline: Double? = nil
    var ghostLabel: String = "base"
    @Binding var scrubStep: Int?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isDragging = false

    var body: some View {
        Chart {
            // Ghost baseline (dashed inkDim) — e.g., base PPL on quantize screen
            if let baseline = ghostBaseline {
                RuleMark(y: .value("Ghost", baseline))
                    .foregroundStyle(Theme.Palette.inkDim.opacity(0.6))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                    .annotation(position: .trailing, alignment: .leading) {
                        Text(ghostLabel)
                            .font(Theme.Fonts.cell)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
            }

            // Area fill (signalGlow gradient beneath line)
            ForEach(validPoints) { point in
                if let yVal = series.value(from: point) {
                    AreaMark(
                        x: .value("Step", point.step),
                        y: .value(series.label, yVal)
                    )
                    .foregroundStyle(
                        LinearGradient(
                            colors: [Theme.Palette.signalGlow, .clear],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                }
            }

            // Primary line (1.5px signal teal)
            ForEach(validPoints) { point in
                if let yVal = series.value(from: point) {
                    LineMark(
                        x: .value("Step", point.step),
                        y: .value(series.label, yVal)
                    )
                    .foregroundStyle(Theme.Palette.signal)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                }
            }

            // Val loss overlay (dashed, inkDim) when primary is .loss
            if series == .loss {
                ForEach(validPoints.filter { $0.valLoss != nil }) { point in
                    if let val = point.valLoss {
                        LineMark(
                            x: .value("Step", point.step),
                            y: .value("VAL", val)
                        )
                        .foregroundStyle(Theme.Palette.inkDim.opacity(0.7))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    }
                }
            }

            // Now-cursor (1px teal RuleMark at current/scrub step)
            if let nowStep = nowCursorStep {
                RuleMark(x: .value("Now", nowStep))
                    .foregroundStyle(Theme.Palette.signal.opacity(0.8))
                    .lineStyle(StrokeStyle(lineWidth: 1))
            }
        }
        // Axis styling: mono ticks, hairline grid, no legend
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { _ in
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                    .foregroundStyle(Theme.Palette.hairline)
                AxisTick(stroke: StrokeStyle(lineWidth: 0.5))
                    .foregroundStyle(Theme.Palette.hairline)
                AxisValueLabel()
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
            }
        }
        .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                    .foregroundStyle(Theme.Palette.hairline)
                AxisTick(stroke: StrokeStyle(lineWidth: 0.5))
                    .foregroundStyle(Theme.Palette.hairline)
                AxisValueLabel()
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
            }
        }
        .chartLegend(.hidden)
        .chartBackground { _ in
            Theme.Palette.panel
        }
        // Scrub-to-freeze: drag gesture on the chart overlay
        .chartOverlay { proxy in
            GeometryReader { geo in
                Rectangle()
                    .fill(.clear)
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                isDragging = true
                                let xPos = value.location.x - geo.frame(in: .local).minX
                                if let step: Int = proxy.value(atX: xPos) {
                                    // Clamp to available range
                                    let minStep = validPoints.first?.step ?? 0
                                    let maxStep = validPoints.last?.step ?? 0
                                    scrubStep = max(minStep, min(maxStep, step))
                                }
                            }
                            .onEnded { _ in
                                isDragging = false
                                // Release scrub — caller can decide to keep or clear
                            }
                    )
            }
        }
        .animation(reduceMotion ? nil : .easeOut(duration: 1.0 / Theme.Motion.chartCommitHz), value: points.count)
    }

    private var validPoints: [TrainPoint] {
        points.filter { series.value(from: $0) != nil }
    }

    private var nowCursorStep: Int? {
        scrubStep ?? validPoints.last?.step
    }
}

// MARK: - Previews

#Preview("StripChart") {
    @Previewable @State var scrubStep: Int? = nil

    let samplePoints: [TrainPoint] = (0..<60).map { i in
        let t = Double(i) / 60.0
        return TrainPoint(
            step: i * 10,
            loss: 2.5 * exp(-t * 2.5) + 0.5 + Double.random(in: -0.02...0.02),
            valLoss: 2.4 * exp(-t * 2.3) + 0.55 + Double.random(in: -0.015...0.015),
            gradNorm: 1.2 * exp(-t) + 0.3,
            lr: 2e-4 * (1.0 - t * 0.5),
            tokS: 1800 + Double.random(in: -50...50)
        )
    }

    return VStack(alignment: .leading, spacing: 0) {
        Text("STRIP CHART — LOSS + VAL LOSS")
            .instrumentLabel()
            .padding(.horizontal, Theme.Space.lg)
            .padding(.top, Theme.Space.lg)

        StripChart(
            points: samplePoints,
            series: .loss,
            scrubStep: Binding(
                get: { scrubStep },
                set: { scrubStep = $0 }
            )
        )
        .frame(height: 180)
        .padding(Theme.Space.lg)

        if let s = scrubStep {
            Text("Scrub: step \(s)")
                .font(Theme.Fonts.cell)
                .foregroundStyle(Theme.Palette.inkDim)
                .padding(.horizontal, Theme.Space.lg)
                .padding(.bottom, Theme.Space.sm)
        }
    }
    .instrumentPanel()
    .background(Theme.Palette.canvas)
    .frame(width: 520)
}
