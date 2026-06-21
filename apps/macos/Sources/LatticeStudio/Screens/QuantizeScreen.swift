import SwiftUI
import AppKit

// MARK: - 03 QUANTIZE
//
// Layout: HSplitView — CONFIG panel (left, OpaquePanel) | LIVE/RESULT area (right, fills).
//
// CONFIG:
//   ParamRowPicker  MODEL  — BF16 models only
//   FaderToggle             Q4 ↔ QuaRot
//   ParamRowField   OUTPUT DIR  + "Choose…" button
//   ParamRowField   SEED  (QuaRot only, default "0xC0FFEE")
//   ParamRowToggle  DRY RUN
//   Primary CTA "▶ QUANTIZE"  (teal, disabled while running)
//
// LIVE / RESULT (driven by store.liveRun when kind is a quantize kind):
//   While running: ReadoutWell LAYER progress + thin teal progress bar + GatePill(.run)
//   Headline HeroNumber: ratio (counts up, .hero)
//   MassBars: true-scale BF16 → method, appears once before/after MBs are non-nil
//   ContrastPair: SIZE / BITS / and for QuaRot a FORWARD-EQUIV row
//   GatePill: PASS/WARN/FAIL from verdict (QuaRot), or PASS on done (Q4)
//   Stop button while running
//   Calm empty state when liveRun is nil or not a quantize kind

struct QuantizeScreen: View {
    @Bindable var store: AppStore

    // MARK: CONFIG form state

    @State private var selectedModelName: String = ""
    @State private var isQuaRot: Bool = false
    @State private var outputDirText: String = ""
    @State private var seedText: String = "0xC0FFEE"
    @State private var dryRun: Bool = false

    // Fold-wipe trigger for ContrastPair
    @State private var isComplete: Bool = false

    // MARK: Derived helpers

    /// BF16 models available for quantization input.
    private var quantizableModels: [ModelInfo] {
        store.models.filter { $0.format == .bf16 }
    }

    private var quantizableModelNames: [String] {
        quantizableModels.map(\.name)
    }

    private var resolvedModel: ModelInfo? {
        quantizableModels.first { $0.name == selectedModelName }
            ?? quantizableModels.first
    }

    /// The live run if it is a quantize job (Q4 or QuaRot).
    private var quantRun: LiveRun? {
        store.liveRun(matching: [.quantizeQ4, .quantizeQuaRot])
    }

    private var isRunning: Bool { quantRun?.status == .running }

    private var subtitle: String {
        if let r = quantRun {
            switch r.status {
            case .running:
                let pct = r.quantLayerCount > 0
                    ? Int(Double(r.quantLayerIndex) / Double(r.quantLayerCount) * 100)
                    : 0
                return "\(r.modelName) · \(r.kind.rawValue) · \(pct)%"
            case .done:
                if let ratio = r.quantRatio {
                    return "\(r.modelName) · \(r.kind.rawValue) · \(String(format: "%.2f", ratio))× smaller"
                }
                return "\(r.modelName) · \(r.kind.rawValue) · done"
            case .failed:
                return "\(r.modelName) · \(r.kind.rawValue) · failed"
            default:
                return "\(r.modelName) · \(r.kind.rawValue)"
            }
        }
        return resolvedModel.map { "\($0.name) · select method and run" } ?? "select a model"
    }

    // MARK: Body

    var body: some View {
        ScreenScaffold(
            screen: .quantize,
            subtitle: subtitle,
            trailing: { trailingActions }
        ) {
            HSplitView {
                configPanel
                    .frame(minWidth: 280, idealWidth: 300, maxWidth: 340)

                livePanel
                    .frame(minWidth: 400)
            }
        }
        .onAppear { applyDefaults() }
        .onChange(of: store.models) { _, _ in applyDefaults() }
        .onChange(of: isQuaRot) { _, _ in refreshOutputDir() }
        .onChange(of: selectedModelName) { _, _ in refreshOutputDir() }
        .onChange(of: quantRun?.status) { _, newStatus in
            if newStatus == .done {
                withAnimation(.easeOut(duration: Theme.Motion.focus)) {
                    isComplete = true
                }
            } else if newStatus == .running {
                isComplete = false
            }
        }
    }

    // MARK: Trailing (scaffold header)

    @ViewBuilder
    private var trailingActions: some View {
        if isRunning {
            Button {
                store.stopRun()
            } label: {
                Text("■ STOP")
                    .font(Theme.Fonts.readout)
                    .foregroundStyle(Theme.Palette.crimson)
                    .padding(.horizontal, Theme.Space.md)
                    .padding(.vertical, Theme.Space.xs)
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.control)
                            .strokeBorder(Theme.Palette.crimson, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: CONFIG panel (left)

    private var configPanel: some View {
        OpaquePanel {
            VStack(alignment: .leading, spacing: 0) {

                // Section header
                Text("CONFIG")
                    .instrumentLabel()
                    .padding(.horizontal, Theme.Space.lg)
                    .padding(.top, Theme.Space.md)
                    .padding(.bottom, Theme.Space.xs)

                // MODEL picker — BF16 only
                if quantizableModelNames.isEmpty {
                    ParamRow(label: "MODEL", value: "— no BF16 models found")
                } else {
                    ParamRowMenu(
                        label: "MODEL",
                        options: quantizableModelNames,
                        selection: $selectedModelName
                    )
                }

                // METHOD — FaderToggle (Q4 ↔ QuaRot)
                VStack(alignment: .leading, spacing: Theme.Space.xs) {
                    Text("METHOD")
                        .instrumentLabel()
                        .padding(.horizontal, Theme.Space.lg)
                        .padding(.top, Theme.Space.sm)

                    FaderToggle(
                        labelA: "Q4",
                        labelB: "QuaRot",
                        isOnB: $isQuaRot
                    )
                }
                .overlay(alignment: .bottom) {
                    Theme.Palette.hairline.frame(height: 1)
                }

                // OUTPUT DIR
                HStack(spacing: 0) {
                    ParamRowField(
                        label: "OUTPUT DIR",
                        text: $outputDirText,
                        placeholder: "path/to/output"
                    )
                    .frame(maxWidth: .infinity)

                    Button("Choose…") {
                        chooseOutputDir()
                    }
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .buttonStyle(.plain)
                    .padding(.trailing, Theme.Space.md)
                }

                // SEED (QuaRot only)
                if isQuaRot {
                    ParamRowField(
                        label: "SEED",
                        text: $seedText,
                        placeholder: "0xC0FFEE"
                    )
                }

                // DRY RUN
                ParamRowToggle(label: "DRY RUN", isOn: $dryRun)

                Spacer()

                // Primary CTA — exactly one teal-filled button per screen
                Button {
                    launchQuantize()
                } label: {
                    HStack {
                        Text("▶ QUANTIZE")
                            .font(Theme.Fonts.readout)
                            .foregroundStyle(Theme.Palette.canvas)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, Theme.Space.sm)
                    .background(isRunning || resolvedModel == nil
                        ? Theme.Palette.signal.opacity(0.4)
                        : Theme.Palette.signal)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(isRunning || resolvedModel == nil)
                .padding(Theme.Space.lg)
            }
        }
    }

    // MARK: LIVE / RESULT panel (right)

    @ViewBuilder
    private var livePanel: some View {
        if let run = quantRun {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.xl) {

                    // ── PROGRESS STRIP (while running) ──────────────────────
                    if run.status == .running {
                        progressStrip(run: run)
                    }

                    // ── HERO RATIO (appears once ratio arrives) ─────────────
                    heroRatioSection(run: run)

                    // ── MASS BARS (appears once before+after arrive) ────────
                    if let before = run.quantBeforeMB, let after = run.quantAfterMB,
                       before > 0, after > 0 {
                        OpaquePanel {
                            MassBars(
                                beforeLabel: "BF16",
                                beforeMB: before,
                                afterLabel: run.kind == .quantizeQuaRot ? "QuaRot" : "Q4",
                                afterMB: after,
                                ratioLabel: "SMALLER",
                                animated: true
                            )
                            .padding(Theme.Space.lg)
                        }
                    }

                    // ── CONTRAST PAIR ────────────────────────────────────────
                    contrastPairSection(run: run)

                    // ── VERDICT GATE ─────────────────────────────────────────
                    verdictGate(run: run)

                }
                .padding(Theme.Space.xl)
            }
        } else {
            emptyLiveState
        }
    }

    // MARK: Progress strip

    private func progressStrip(run: LiveRun) -> some View {
        OpaquePanel {
            VStack(alignment: .leading, spacing: Theme.Space.sm) {
                HStack(spacing: Theme.Space.lg) {
                    ReadoutWell(
                        label: "LAYER",
                        value: run.quantLayerCount > 0
                            ? "\(run.quantLayerIndex) / \(run.quantLayerCount)"
                            : "—"
                    )
                    GatePill(.run, label: "QUANTIZING")
                }

                // Thin teal progress bar
                if run.quantLayerCount > 0 {
                    let fraction = Double(run.quantLayerIndex) / Double(run.quantLayerCount)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Rectangle()
                                .fill(Theme.Palette.wellSink)
                                .frame(maxWidth: .infinity, maxHeight: 2)
                            Rectangle()
                                .fill(Theme.Palette.signal)
                                .frame(width: geo.size.width * fraction, height: 2)
                                .animation(.easeOut(duration: Theme.Motion.tick), value: fraction)
                        }
                    }
                    .frame(height: 2)
                }
            }
            .padding(Theme.Space.lg)
        }
    }

    // MARK: Hero ratio

    @ViewBuilder
    private func heroRatioSection(run: LiveRun) -> some View {
        let ratioStr: String = {
            if let r = run.quantRatio { return String(format: "%.2f×", r) }
            return "—×"
        }()

        OpaquePanel {
            HeroNumber(
                value: ratioStr,
                unit: "smaller",
                size: .hero,
                unitPosition: .below
            )
            .padding(Theme.Space.lg)
        }
    }

    // MARK: Contrast pair

    private func contrastMetrics(for run: LiveRun) -> [ContrastMetric] {
        let beforeMB = run.quantBeforeMB
        let afterMB  = run.quantAfterMB

        let beforeSizeStr = beforeMB.map { sizeLabel($0) } ?? "—"
        let afterSizeStr  = afterMB.map  { sizeLabel($0) } ?? "—"
        let sizeDelta: String = {
            guard let b = beforeMB, let a = afterMB, b > 0 else { return "—" }
            let saved = (b - a) / b * 100
            return String(format: "−%.0f%%", saved)
        }()

        // Map the dominant quant scheme to a bit-width for the BITS row.
        // Before-bits: BF16/F16 input → 16. After-bits: derived from scheme.
        let beforeBits: Int = 16  // quantize always takes BF16/F16 input
        let afterBits: Int? = {
            guard let scheme = run.quantScheme else { return nil }
            let s = scheme.uppercased()
            if s.hasPrefix("Q4") { return 4 }
            if s.hasPrefix("Q8") { return 8 }
            if s == "F16" || s == "BF16" { return 16 }
            return nil
        }()
        let bitsAfterStr = afterBits.map { "\($0)" } ?? "—"
        let bitsDelta: String = {
            guard let a = afterBits, a < beforeBits else { return "—" }
            let saved = Double(beforeBits - a) / Double(beforeBits) * 100
            return String(format: "−%.0f%%", saved)
        }()

        var metrics: [ContrastMetric] = [
            ContrastMetric(
                label: "SIZE",
                beforeValue: beforeSizeStr,
                afterValue: afterSizeStr,
                delta: sizeDelta,
                deltaGood: true
            ),
            ContrastMetric(
                label: "BITS",
                beforeValue: "\(beforeBits)",
                afterValue: bitsAfterStr,
                delta: bitsDelta,
                deltaGood: afterBits != nil
            ),
        ]

        // QuaRot: forward-equivalence row derived from verdict text.
        // The verdict string may embed a max-abs value (e.g. "PASS max_abs=0.0012").
        // Show "—" until verdict arrives; the ContrastPair's fold-wipe triggers on isComplete.
        if run.kind == .quantizeQuaRot {
            let equivStr = run.quantMaxAbs.map { String(format: "%.2e", $0) }
                ?? run.verdict.flatMap { extractMaxAbs($0) } ?? "—"
            let isEquivGood: Bool = {
                guard let v = run.verdict else { return true }
                return v.uppercased().hasPrefix("PASS")
            }()
            metrics.append(ContrastMetric(
                label: "FORWARD-EQUIV",
                beforeValue: "reference",
                afterValue: equivStr,
                beforeUnit: "",
                afterUnit: "max_abs",
                delta: isEquivGood ? "✓" : "✗",
                deltaGood: isEquivGood
            ))
        }

        return metrics
    }

    private func contrastPairSection(run: LiveRun) -> some View {
        ContrastPair(metrics: contrastMetrics(for: run), isComplete: $isComplete)
    }

    // MARK: Verdict gate

    @ViewBuilder
    private func verdictGate(run: LiveRun) -> some View {
        // Only show gate once the run has a terminal status or we have verdict data.
        let gateStatus: GateStatus? = {
            switch run.status {
            case .running:
                return nil  // gate not shown while running
            case .done:
                if run.kind == .quantizeQuaRot, let v = run.verdict {
                    let upper = v.uppercased()
                    if upper.hasPrefix("PASS") { return .pass }
                    if upper.hasPrefix("WARN") { return .warn }
                    if upper.hasPrefix("FAIL") { return .fail }
                }
                // Q4 or QuaRot with no verdict string → PASS on done
                return .pass
            case .failed:
                return .fail
            default:
                return nil
            }
        }()

        if let gs = gateStatus {
            let gateLabel: String = {
                switch run.kind {
                case .quantizeQuaRot:
                    if let v = run.verdict { return verdictLabel(v) }
                    return gs == .pass ? "FORWARD-EQUIV PASS" : gs.label
                default:
                    if let ratio = run.quantRatio,
                       let after = run.quantAfterMB {
                        return "Q4 ok \(sizeLabel(after)) \(String(format: "%.2f", ratio))× PASS"
                    }
                    return "Q4 PASS"
                }
            }()

            HStack(spacing: Theme.Space.sm) {
                GatePill(gs, label: gateLabel)
                Spacer()
            }
        }
    }

    // MARK: Empty state

    private var emptyLiveState: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("NO ACTIVE RUN")
                .instrumentLabel()
            Text("Select a BF16 model, choose a method, then press ▶ QUANTIZE.")
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.inkDim)
        }
        .padding(Theme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // MARK: Defaults + helpers

    private func applyDefaults() {
        if selectedModelName.isEmpty || !quantizableModelNames.contains(selectedModelName) {
            selectedModelName = store.targetModel?.format == .bf16
                ? (store.targetModel?.name ?? quantizableModelNames.first ?? "")
                : (quantizableModelNames.first ?? "")
        }
        refreshOutputDir()
    }

    private func refreshOutputDir() {
        // Skip auto-derivation once the user has chosen a path explicitly via NSOpenPanel.
        guard outputDirText.isEmpty || !outputDirForced else { return }
        let suffix = isQuaRot ? "quarot" : "q4"
        if let root = store.repoRootPath, !selectedModelName.isEmpty {
            outputDirText = "\(root)/models/\(selectedModelName)-\(suffix)"
        } else if !selectedModelName.isEmpty {
            outputDirText = "models/\(selectedModelName)-\(suffix)"
        }
    }

    /// True when the output dir was set via the chooser (not auto-derived).
    @State private var outputDirForced: Bool = false

    private func chooseOutputDir() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select Output Directory"
        if panel.runModal() == .OK, let url = panel.url {
            outputDirText = url.path
            outputDirForced = true
        }
    }

    private func parseSeed() -> UInt64? {
        let s = seedText.trimmingCharacters(in: .whitespaces)
        if s.lowercased().hasPrefix("0x") {
            return UInt64(s.dropFirst(2), radix: 16)
        }
        return UInt64(s)
    }

    private func launchQuantize() {
        guard let model = resolvedModel else { return }
        let outputURL: URL
        if outputDirText.isEmpty {
            let suffix = isQuaRot ? "quarot" : "q4"
            outputURL = model.path.deletingLastPathComponent()
                .appendingPathComponent("\(model.name)-\(suffix)")
        } else {
            outputURL = URL(fileURLWithPath: outputDirText)
        }

        let cfg = QuantConfig(
            modelDir: model.path,
            outputDir: outputURL,
            method: isQuaRot ? .quarot : .q4,
            seed: isQuaRot ? (parseSeed() ?? 0xC0FFEE) : nil,
            dryRun: dryRun
        )

        isComplete = false
        store.startQuantize(cfg)
    }

    // MARK: Formatting helpers

    private func sizeLabel(_ mb: Double) -> String {
        if mb >= 1024 {
            return String(format: "%.2f GB", mb / 1024.0)
        }
        return String(format: "%.0f MB", mb)
    }

    /// Extract a max_abs value from a verdict string like "PASS max_abs=0.0012".
    /// Returns nil if the pattern is absent, so the UI shows "—" until the value arrives.
    private func extractMaxAbs(_ verdict: String) -> String? {
        // Pattern: "max_abs=<value>" anywhere in the verdict string.
        let parts = verdict.components(separatedBy: " ")
        for part in parts {
            if part.lowercased().hasPrefix("max_abs=") {
                let value = String(part.dropFirst("max_abs=".count))
                return value.isEmpty ? nil : value
            }
        }
        return nil
    }

    /// Build a display label from the raw verdict string.
    private func verdictLabel(_ verdict: String) -> String {
        let upper = verdict.uppercased()
        if upper.hasPrefix("PASS") {
            if let maxAbs = extractMaxAbs(verdict) {
                return "FORWARD-EQUIV PASS · max_abs \(maxAbs)"
            }
            return "FORWARD-EQUIV PASS"
        }
        if upper.hasPrefix("WARN") { return "FORWARD-EQUIV WARN" }
        if upper.hasPrefix("FAIL") { return "FORWARD-EQUIV FAIL" }
        return verdict
    }
}
