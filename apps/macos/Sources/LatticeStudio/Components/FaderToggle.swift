import SwiftUI

// MARK: - Primitive 11: FaderToggle
//
// The A/B adapter selector styled as a console fader.
// This is the ONE element where Theme.Motion.faderSpring is allowed.
//
// Design spec:
//   - Console fader metaphor: a slider-like control, left=BASE, right=BASE+LoRA
//   - Spring physics: Animation.spring(response: 0.32, dampingFraction: 0.85)
//   - Hairline track, teal thumb/knob when on B side
//   - Labels: A label (left) and B label (right)
//   - Current selection text below
//   - Respects accessibilityReduceMotion (spring → instant snap)

/// The A/B adapter console fader. The one place spring physics is permitted.
///
/// ```swift
/// FaderToggle(
///     labelA: "BASE",
///     labelB: "BASE + LoRA r8",
///     isOnB: $isAdapterActive
/// )
/// ```
struct FaderToggle: View {
    let labelA: String
    let labelB: String
    @Binding var isOnB: Bool

    var onSwap: (() -> Void)? = nil

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isDragging: Bool = false
    @State private var dragOffset: CGFloat = 0

    // Thumb size
    private let thumbW: CGFloat = 28
    private let thumbH: CGFloat = 48
    private let trackH: CGFloat = 4

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            // Track + thumb
            GeometryReader { geo in
                let trackWidth = geo.size.width
                let travelRange = trackWidth - thumbW
                let targetX: CGFloat = isOnB ? travelRange : 0

                ZStack(alignment: .leading) {
                    // Track
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Theme.Palette.wellSink)
                        .frame(height: trackH)
                        .overlay(
                            RoundedRectangle(cornerRadius: 2)
                                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
                        )
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, thumbW / 2)

                    // Teal active fill (left portion for B side)
                    if isOnB {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Theme.Palette.signal.opacity(0.4))
                            .frame(width: targetX + thumbW / 2, height: trackH)
                            .padding(.leading, thumbW / 2)
                    }

                    // Fader thumb (knob)
                    ZStack {
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .fill(isOnB ? Theme.Palette.signal : Theme.Palette.panel)
                            .overlay(
                                RoundedRectangle(cornerRadius: 4, style: .continuous)
                                    .strokeBorder(
                                        isOnB ? Theme.Palette.signal : Theme.Palette.hairline,
                                        lineWidth: 1
                                    )
                            )
                            .shadow(color: .black.opacity(0.3), radius: 2, x: 0, y: 1)

                        // Grip lines on the thumb
                        VStack(spacing: 3) {
                            ForEach(0..<3, id: \.self) { _ in
                                Theme.Palette.hairline
                                    .frame(width: 14, height: 1)
                            }
                        }
                    }
                    .frame(width: thumbW, height: thumbH)
                    .offset(x: isDragging ? dragOffset : targetX)
                    .animation(
                        isDragging
                            ? nil
                            : (reduceMotion ? nil : Theme.Motion.faderSpring),
                        value: isOnB
                    )
                    .gesture(
                        DragGesture()
                            .onChanged { val in
                                isDragging = true
                                let clamped = max(0, min(travelRange, targetX + val.translation.width))
                                dragOffset = clamped
                            }
                            .onEnded { val in
                                isDragging = false
                                let finalX = max(0, min(travelRange, targetX + val.translation.width))
                                let shouldBeB = finalX > travelRange / 2
                                if shouldBeB != isOnB {
                                    isOnB = shouldBeB
                                    onSwap?()
                                }
                            }
                    )
                    .onTapGesture {
                        withAnimation(reduceMotion ? nil : Theme.Motion.faderSpring) {
                            isOnB.toggle()
                        }
                        onSwap?()
                    }
                }
                .frame(height: thumbH)
            }
            .frame(height: thumbH)

            // A/B labels + status line
            HStack {
                Text(labelA)
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(isOnB ? Theme.Palette.inkDim : Theme.Palette.ink)

                Spacer()

                // Selection stamp — shown when B side is active.
                // The fader sets adapterPath for the NEXT generation; there is no hot-swap.
                if isOnB {
                    Text("applies next send")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.signal)
                }

                Spacer()

                Text(labelB)
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(isOnB ? Theme.Palette.ink : Theme.Palette.inkDim)
            }
            .frame(height: Theme.Space.rowHeight)
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.md)
    }
}

// MARK: - Previews

#Preview("FaderToggle") {
    @Previewable @State var isOnB: Bool = false

    return VStack(spacing: Theme.Space.xl) {
        Text("HOT-SWAP FADER")
            .instrumentLabel()
            .padding(.horizontal, Theme.Space.lg)
            .padding(.top, Theme.Space.lg)

        FaderToggle(
            labelA: "BASE",
            labelB: "BASE + LoRA r8",
            isOnB: Binding(get: { isOnB }, set: { isOnB = $0 }),
            onSwap: {
                // adapter swap callback
            }
        )

        Text(isOnB ? "Active: BASE + LoRA r8 · adapter applies next send" : "Active: BASE")
            .font(Theme.Fonts.readout)
            .foregroundStyle(Theme.Palette.inkDim)
            .padding(.horizontal, Theme.Space.lg)
            .padding(.bottom, Theme.Space.lg)
    }
    .instrumentPanel()
    .background(Theme.Palette.canvas)
    .frame(width: 400)
}
