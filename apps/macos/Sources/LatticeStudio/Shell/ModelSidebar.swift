import SwiftUI

// MARK: - Model sidebar
//
// The model-centric spine of the redesign: every discovered model, grouped into generative models
// and embeddings. Selecting a row makes that model the working target for every verb tab (Chat,
// Serve, Quantize, Train, Inspect). The green dot marks the model currently resident in GPU memory;
// hovering a non-resident generative model reveals a Load pill that warms it. Footer counts are
// live — jobs running through the engine and discovered LoRA adapters.

struct ModelSidebar: View {
    @Bindable var store: AppStore

    private var generative: [ModelInfo] { store.models.filter { !$0.isEmbedding } }
    private var embeddings: [ModelInfo] { store.models.filter { $0.isEmbedding } }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.lg) {
                    if store.models.isEmpty {
                        emptyHint
                    } else {
                        if !generative.isEmpty {
                            section("Models", generative, showAdd: true)
                        }
                        if !embeddings.isEmpty {
                            section("Embeddings", embeddings)
                        }
                    }
                }
                .padding(.horizontal, Theme.Space.md)
                .padding(.top, Theme.Space.lg)
                .padding(.bottom, Theme.Space.md)
            }
            footer
        }
        .frame(width: 268)
        .background(Theme.Palette.sidebar)
        .overlay(alignment: .trailing) {
            Rectangle().fill(Theme.Palette.hairline).frame(width: 1)
        }
    }

    // MARK: Sections

    private func section(_ title: String, _ models: [ModelInfo], showAdd: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            SectionHeader(title: title, showAdd: showAdd) { store.getModelsPresented = true }
            ForEach(models) { model in
                SidebarModelRow(store: store, model: model)
            }
        }
    }

    // MARK: Footer

    private var footer: some View {
        VStack(spacing: 0) {
            VStack(spacing: 1) {
                footerRow(symbol: "arrow.triangle.2.circlepath",
                          label: "Jobs",
                          value: "\(store.jobsRunning) running",
                          active: store.jobsRunning > 0)
                footerRow(symbol: "square.stack.3d.up",
                          label: "Adapters",
                          value: "\(store.adapters.count)",
                          active: false)
            }
            .padding(.horizontal, Theme.Space.md)
            .padding(.vertical, Theme.Space.md)
        }
    }

    // Status row: icon + label on the left, live count right-aligned — matches the mockup footer.
    private func footerRow(symbol: String, label: String, value: String, active: Bool) -> some View {
        HStack(spacing: 8) {
            Image(systemName: symbol)
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(active ? Theme.Palette.signal : Theme.Palette.textTertiary)
                .frame(width: 15)
            Text(label)
                .font(Theme.Fonts.controlText)
                .foregroundStyle(Theme.Palette.textSecondary)
            Spacer()
            Text(value)
                .font(Theme.Fonts.caption)
                .foregroundStyle(active ? Theme.Palette.signal : Theme.Palette.textTertiary)
                .monospacedDigit()
        }
        .padding(.vertical, 5)
    }

    // MARK: Empty hint

    private var emptyHint: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("NO MODELS")
                .font(Theme.Fonts.sectionLabel)
                .textCase(.uppercase)
                .tracking(Theme.Space.labelTracking)
                .foregroundStyle(Theme.Palette.textTertiary)
            if !store.binariesReady {
                Text("Build the engine first:")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)
                Text("make build")
                    .font(Theme.Fonts.codeFont)
                    .foregroundStyle(Theme.Palette.signal)
            } else {
                Text("Use Get Models below to download or import a model.")
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, Theme.Space.xs)
    }
}

// MARK: - Section header (uppercase label + hover-revealed add button)

private struct SectionHeader: View {
    let title: String
    let showAdd: Bool
    let onAdd: () -> Void
    @State private var hovered = false

    var body: some View {
        HStack(spacing: 0) {
            Text(title)
                .font(Theme.Fonts.sectionLabel)
                .textCase(.uppercase)
                .tracking(Theme.Space.labelTracking)
                .foregroundStyle(Theme.Palette.textTertiary)
            Spacer()
            if showAdd {
                // Reveal on hover so the resting header is just the label (mockup); Get Models
                // also stays reachable via ⌘K and the empty state.
                Button(action: onAdd) {
                    Image(systemName: "plus")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Get models")
                .opacity(hovered ? 1 : 0)
                .allowsHitTesting(hovered)
            }
        }
        .padding(.horizontal, Theme.Space.xs)
        .padding(.bottom, 2)
        .onHover { hovered = $0 }
    }
}

// MARK: - One model row

private struct SidebarModelRow: View {
    @Bindable var store: AppStore
    let model: ModelInfo
    @State private var hovered = false

    private var selected: Bool { store.targetModel?.id == model.id }
    private var resident: Bool { store.residentModel?.id == model.id }
    private var loading: Bool {
        store.isChatModelLoading && store.chatSelectedModelName == model.name
    }

    var body: some View {
        Button { store.selectSidebarModel(model) } label: {
            HStack(alignment: .top, spacing: 7) {
                dot
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(model.name)
                            .font(Theme.Fonts.bodyStrong)
                            .foregroundStyle(Theme.Palette.textPrimary)
                            .lineLimit(1)
                        Spacer(minLength: 4)
                        trailing
                    }
                    Text(subtitle)
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .lineLimit(1)
                }
            }
            .padding(.vertical, 7)
            .padding(.horizontal, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: Theme.Radius.well).fill(cardFill))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.well)
                    .strokeBorder(selected ? Theme.Palette.selectionBorder : .clear, lineWidth: 1)
            )
            .overlay(alignment: .leading) {
                if selected {
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(Theme.Palette.signal)
                        .frame(width: 3)
                        .padding(.vertical, 7)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
        .animation(.easeOut(duration: Theme.Motion.hover), value: hovered)
    }

    private var cardFill: Color {
        // Indigo-tinted fill (accent @12%) so a selected model reads as the active card, not a
        // hover. The neutral surfaceRaised it used before was indistinguishable from the hover row.
        if selected { return Theme.Palette.selectionFill }
        if hovered { return Theme.Palette.surfaceHover }
        return .clear
    }

    // Fixed-width dot column so names align whether or not a row is resident/loading.
    private var dot: some View {
        ZStack {
            if resident {
                Circle().fill(Theme.Palette.success).frame(width: 6, height: 6)
            } else if loading {
                Circle().fill(Theme.Palette.signal).frame(width: 6, height: 6)
            }
        }
        .frame(width: 6)
        .padding(.top, 4)
    }

    // Trailing: a Load pill on hover (non-resident generative models), else the format badge.
    @ViewBuilder
    private var trailing: some View {
        if hovered && !resident && !loading && !model.isEmbedding {
            Button { store.loadModel(model) } label: {
                Text("Load")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Theme.Palette.signal)
                    .padding(.horizontal, 7).padding(.vertical, 2)
                    .overlay(
                        Capsule().strokeBorder(Theme.Palette.selectionBorder, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
        } else if let badge = badgeText {
            Text(badge)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.Palette.textTertiary)
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(RoundedRectangle(cornerRadius: Theme.Radius.badge).fill(Theme.Palette.wellSink))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.Radius.badge)
                        .strokeBorder(Theme.Palette.borderStandard, lineWidth: 1)
                )
        }
    }

    private var badgeText: String? {
        switch model.format {
        case .bf16: return "BF16"
        case .q4, .quarot: return "Q4"
        case .embedding, .unknown: return nil
        }
    }

    private var subtitle: String {
        var parts = [compactSize(model.sizeBytes)]
        if model.format == .quarot { parts.append("rotated") }
        if model.isEmbedding, let h = model.hidden { parts.append("\(h)-d") }
        return parts.joined(separator: " · ")
    }

    // Decimal GB/MB (÷10^9, ÷10^6) to match Finder, the chat hero's ByteCountFormatter, and the
    // mockup's figures (19.62 GB for the 27B). Keeping every size readout on one convention avoids
    // the same model reading 1.64 GB in one place and 1.76 GB in another.
    private func compactSize(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_000_000_000.0
        if gb >= 1 { return String(format: "%.2f GB", gb) }
        let mb = Double(bytes) / 1_000_000.0
        return String(format: "%.0f MB", mb)
    }
}
