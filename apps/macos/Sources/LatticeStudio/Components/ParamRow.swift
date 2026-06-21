import SwiftUI

// MARK: - Primitive 8: ParamRow
//
// Label (left, dim) + control (right) on one hairline-ruled line.
// Forms are stacks of ParamRows — never boxed, never cardd.
//
// Variants:
//   - Text/label only (read-only)
//   - TextField (editable string)
//   - Slider (with live mono readout showing current value in the track)
//   - Picker / segmented (for method selection)
//   - Toggle

// MARK: Text ParamRow

/// A read-only label + value display on a hairline-ruled line.
///
/// ```swift
/// ParamRow(label: "MODEL", value: "qwen3.5-0.8b")
/// ```
struct ParamRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .instrumentLabel()
            Spacer()
            Text(value)
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
        }
        .frame(height: Theme.Space.rowHeight)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }
}

// MARK: TextField ParamRow

/// An editable label + TextField on a hairline-ruled line.
///
/// ```swift
/// ParamRowField(label: "DATASET", text: $datasetPath, placeholder: "path/to/data.jsonl")
/// ```
struct ParamRowField: View {
    let label: String
    @Binding var text: String
    var placeholder: String = ""

    var body: some View {
        HStack {
            Text(label)
                .instrumentLabel()
            Spacer()
            TextField(placeholder, text: $text)
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 240)
                .textFieldStyle(.plain)
        }
        .frame(height: Theme.Space.rowHeight)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }
}

// MARK: Slider ParamRow

/// A label + Slider row where the current value appears as a live mono readout beside the track.
///
/// ```swift
/// ParamRowSlider(label: "RANK", value: $rank, range: 1...64, step: 1, format: "%.0f")
/// ParamRowSlider(label: "LR", value: $lr, range: 1e-5...1e-2, format: "%.2e")
/// ```
struct ParamRowSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double = 0
    var format: String = "%.4g"
    var unit: String = ""

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var formattedValue: String {
        String(format: format, value) + (unit.isEmpty ? "" : " \(unit)")
    }

    var body: some View {
        HStack(spacing: Theme.Space.sm) {
            Text(label)
                .instrumentLabel()
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .frame(width: 80, alignment: .leading)

            // Slider with live mono readout
            HStack(spacing: Theme.Space.sm) {
                if step > 0 {
                    Slider(value: $value, in: range, step: step)
                } else {
                    Slider(value: $value, in: range)
                }

                // Live mono readout in the track
                Text(formattedValue)
                    .font(Theme.Fonts.readout)
                    .foregroundStyle(Theme.Palette.ink)
                    .monospacedDigit()
                    .frame(width: 72, alignment: .trailing)
                    .contentTransition(reduceMotion ? .identity : .numericText())
                    .animation(
                        reduceMotion ? nil : .easeOut(duration: Theme.Motion.tick),
                        value: formattedValue
                    )
            }
        }
        .frame(height: Theme.Space.rowHeightComfortable)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }
}

// MARK: Picker ParamRow

/// A label + segmented picker on a hairline-ruled line.
///
/// ```swift
/// ParamRowPicker(label: "METHOD", options: ["Q4", "QuaRot"], selection: $method)
/// ```
struct ParamRowPicker: View {
    let label: String
    let options: [String]
    @Binding var selection: String

    var body: some View {
        HStack {
            Text(label)
                .instrumentLabel()
            Spacer()
            Picker("", selection: $selection) {
                ForEach(options, id: \.self) { opt in
                    Text(opt).tag(opt)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 200)
            .labelsHidden()
        }
        .frame(height: Theme.Space.rowHeightComfortable)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }
}

// MARK: Menu ParamRow

/// A label + dropdown menu picker on a hairline-ruled line.
///
/// Use this (not `ParamRowPicker`) for selections with many or long options — e.g. MODEL —
/// where a segmented control would overflow the panel. The dropdown stays a fixed-width button
/// showing the current selection and never grows with the option count.
///
/// ```swift
/// ParamRowMenu(label: "MODEL", options: modelNames, selection: $selectedModelName)
/// ```
struct ParamRowMenu: View {
    let label: String
    let options: [String]
    @Binding var selection: String

    var body: some View {
        HStack {
            Text(label)
                .instrumentLabel()
            Spacer()
            Picker("", selection: $selection) {
                ForEach(options, id: \.self) { opt in
                    Text(opt).tag(opt)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .font(Theme.Fonts.readout)
            .fixedSize()
        }
        .frame(height: Theme.Space.rowHeightComfortable)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }
}

// MARK: Toggle ParamRow

/// A label + toggle on a hairline-ruled line.
///
/// ```swift
/// ParamRowToggle(label: "COMFORTABLE ROWS", isOn: $comfortable)
/// ```
struct ParamRowToggle: View {
    let label: String
    @Binding var isOn: Bool

    var body: some View {
        HStack {
            Text(label)
                .instrumentLabel()
            Spacer()
            Toggle("", isOn: $isOn)
                .toggleStyle(.switch)
                .labelsHidden()
        }
        .frame(height: Theme.Space.rowHeight)
        .padding(.horizontal, Theme.Space.lg)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }
}

// MARK: - Previews

#Preview("ParamRow") {
    @Previewable @State var rank: Double = 8
    @Previewable @State var lr: Double = 2e-4
    @Previewable @State var method: String = "Q4"
    @Previewable @State var dataset: String = "claude-lora.jsonl"
    @Previewable @State var comfortable: Bool = false

    return VStack(spacing: 0) {
        ParamRow(label: "MODEL", value: "qwen3.5-0.8b")
        ParamRow(label: "LAYERS", value: "18 GDN · 6 GQA")
        ParamRowSlider(label: "RANK", value: $rank, range: 1...64, step: 1, format: "%.0f")
        ParamRowSlider(label: "LR", value: $lr, range: 1e-5...1e-2, format: "%.2e")
        ParamRowPicker(label: "METHOD", options: ["Q4", "QuaRot"], selection: $method)
        ParamRowField(label: "DATASET", text: $dataset, placeholder: "path/to/data.jsonl")
        ParamRowToggle(label: "COMFORTABLE ROWS", isOn: $comfortable)
    }
    .instrumentPanel()
    .background(Theme.Palette.canvas)
    .frame(width: 400)
}
