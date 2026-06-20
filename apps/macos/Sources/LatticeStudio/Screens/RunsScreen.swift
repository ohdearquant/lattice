import SwiftUI
import AppKit

// MARK: - 06 RUNS — the lab notebook
//
// Layout:
//   • Live banner (shown when store.liveRun != nil): teal GatePill(.run) + step/loss status
//   • DataTable of store.runs: KIND · MODEL · STATUS · METRIC · DURATION · WHEN
//   • Right inspector OpaquePanel: ReadoutWells of run summary + GatePill status +
//     adapter path with "Reveal" affordance (when adapterPath != nil)
//
// Empty state when store.runs.isEmpty.
// RunRecord does NOT store a point series — no chart replay in history rows.

struct RunsScreen: View {
    @Bindable var store: AppStore

    @State private var selectedRunID: String?

    private var selectedRun: RunRecord? {
        store.runs.first { $0.id == selectedRunID }
    }

    private var subtitle: String {
        let count = store.runs.count
        if count == 0 { return "no runs yet" }
        return "\(count) run\(count == 1 ? "" : "s") logged"
    }

    var body: some View {
        ScreenScaffold(
            screen: .runs,
            subtitle: subtitle
        ) {
            VStack(spacing: 0) {
                // Live banner — only shown while a run is actively in-flight or paused.
                // A completed or failed run drops out of the banner (it appears in the table).
                if let live = store.liveRun,
                   live.status == .running || live.status == .paused {
                    liveBanner(live)
                }

                if store.runs.isEmpty {
                    emptyState
                } else {
                    HSplitView {
                        // Center: master runs table
                        runsTable
                            .frame(minWidth: 480)

                        // Right inspector
                        inspectorPanel
                            .frame(minWidth: 260, idealWidth: 300, maxWidth: 320)
                    }
                }
            }
        }
    }

    // MARK: Live banner

    private func liveBanner(_ live: LiveRun) -> some View {
        HStack(spacing: Theme.Space.md) {
            GatePill(.run, label: live.kind.rawValue)

            Text(live.modelName)
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
                .lineLimit(1)

            Spacer()

            // Train: show step / current loss
            if live.kind == .train {
                if let loss = live.currentLoss {
                    HStack(spacing: 4) {
                        Text("step")
                            .instrumentLabel()
                        Text("\(live.currentStep)")
                            .font(Theme.Fonts.readout)
                            .foregroundStyle(Theme.Palette.ink)
                            .monospacedDigit()
                        Text("·")
                            .foregroundStyle(Theme.Palette.inkDim)
                        Text("loss")
                            .instrumentLabel()
                        Text(String(format: "%.4f", loss))
                            .font(Theme.Fonts.readout)
                            .foregroundStyle(Theme.Palette.signal)
                            .monospacedDigit()
                            .contentTransition(.numericText())
                            .animation(.easeOut(duration: Theme.Motion.tick), value: loss)
                    }
                }
            }

            // Quantize: show layer i/n progress
            if live.kind == .quantizeQ4 || live.kind == .quantizeQuaRot {
                if live.quantLayerCount > 0 {
                    HStack(spacing: 4) {
                        Text("layer")
                            .instrumentLabel()
                        Text("\(live.quantLayerIndex)/\(live.quantLayerCount)")
                            .font(Theme.Fonts.readout)
                            .foregroundStyle(Theme.Palette.ink)
                            .monospacedDigit()
                    }
                }
            }

            // Status pill reflects live state
            GatePill(liveGateStatus(live.status))
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.sm)
        .background(Theme.Palette.panel)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    private func liveGateStatus(_ status: RunStatus) -> GateStatus {
        switch status {
        case .running: .run
        case .paused: .warn
        case .done: .pass
        case .failed: .fail
        case .idle: .warn
        }
    }

    // MARK: Runs table

    private var runsTable: some View {
        ScrollView {
            DataTable(
                rows: store.runs,
                columns: runColumns,
                selectedID: $selectedRunID,
                comfortable: store.rowComfortable
            )
        }
        .instrumentPanel()
    }

    private var runColumns: [ColumnDef<RunRecord>] {
        [
            ColumnDef(
                id: "kind",
                header: "KIND",
                alignment: .leading,
                minWidth: 72,
                isNumeric: false
            ) { $0.kind.rawValue },

            ColumnDef(
                id: "model",
                header: "MODEL",
                alignment: .leading,
                minWidth: 140,
                isNumeric: false
            ) { $0.model },

            ColumnDef(
                id: "status",
                header: "STATUS",
                alignment: .leading,
                minWidth: 64,
                isNumeric: false
            ) { $0.status.rawValue.uppercased() },

            ColumnDef(
                id: "metric",
                header: "METRIC",
                alignment: .trailing,
                minWidth: 80,
                isNumeric: true
            ) { rec in
                switch rec.kind {
                case .train:
                    if let best = rec.bestVal {
                        return String(format: "%.4f", best)
                    } else if let last = rec.lastLoss {
                        return String(format: "%.4f", last)
                    }
                    return "—"
                case .quantizeQ4, .quantizeQuaRot:
                    // For quant runs show configSummary if available (ratio/verdict), else "—"
                    return rec.configSummary ?? "—"
                case .chat:
                    return "—"
                }
            },

            ColumnDef(
                id: "duration",
                header: "DURATION",
                alignment: .trailing,
                minWidth: 72,
                isNumeric: true
            ) { rec in
                guard let d = rec.durationS else { return "—" }
                if d < 60 { return String(format: "%.0fs", d) }
                let m = Int(d) / 60
                let s = Int(d) % 60
                return "\(m)m\(s)s"
            },

            ColumnDef(
                id: "when",
                header: "WHEN",
                alignment: .trailing,
                minWidth: 80,
                isNumeric: false
            ) { rec in
                let interval = Date().timeIntervalSince(rec.startedAt)
                if interval < 60 { return "just now" }
                if interval < 3600 {
                    let m = Int(interval / 60)
                    return "\(m)m ago"
                }
                let h = Int(interval / 3600)
                if h < 24 { return "\(h)h ago" }
                let d = Int(interval / 86400)
                return "\(d)d ago"
            },
        ]
    }

    // MARK: Inspector panel

    @ViewBuilder
    private var inspectorPanel: some View {
        if let run = selectedRun {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.xl) {
                    runInspector(run)
                }
                .padding(Theme.Space.lg)
            }
            .instrumentPanel()
        } else {
            VStack {
                Spacer()
                Text("SELECT A RUN")
                    .instrumentLabel()
                Spacer()
            }
            .frame(maxWidth: .infinity)
            .instrumentPanel()
        }
    }

    private func runInspector(_ run: RunRecord) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.lg) {
            // Header: kind + status pill
            HStack(spacing: Theme.Space.sm) {
                Text(run.kind.rawValue)
                    .font(Theme.Fonts.title)
                    .foregroundStyle(Theme.Palette.ink)

                GatePill(gatePillStatus(run.status), label: run.status.rawValue.uppercased())
            }

            // Model name
            Text(run.model)
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.inkDim)
                .lineLimit(1)

            // Summary readout wells
            let wells = runWells(for: run)
            if !wells.isEmpty {
                LazyVGrid(
                    columns: [GridItem(.flexible()), GridItem(.flexible())],
                    spacing: Theme.Space.sm
                ) {
                    ForEach(wells, id: \.label) { w in
                        ReadoutWell(label: w.label, value: w.value, unit: w.unit)
                    }
                }
            }

            // Config summary (if stored)
            if let cfg = run.configSummary {
                VStack(alignment: .leading, spacing: 4) {
                    Text("CONFIG")
                        .instrumentLabel()
                    Text(cfg)
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                        .lineLimit(3)
                        .monospacedDigit()
                }
            }

            // Adapter path — reveal affordance
            if let adapterPath = run.adapterPath {
                adapterRevealRow(adapterPath)
            }
        }
    }

    private struct WellSpec {
        let label: String
        let value: String
        let unit: String
        init(_ label: String, _ value: String, _ unit: String = "") {
            self.label = label; self.value = value; self.unit = unit
        }
    }

    private func runWells(for run: RunRecord) -> [WellSpec] {
        var ws: [WellSpec] = []

        // Timing
        let startStr = relativeTimeString(run.startedAt)
        ws.append(WellSpec("STARTED", startStr))

        if let dur = run.durationS {
            if dur < 60 {
                ws.append(WellSpec("DURATION", String(format: "%.0f", dur), "s"))
            } else {
                let m = Int(dur) / 60
                let s = Int(dur) % 60
                ws.append(WellSpec("DURATION", "\(m)m \(s)s"))
            }
        }

        // Metrics by kind
        switch run.kind {
        case .train:
            if let last = run.lastLoss {
                ws.append(WellSpec("LAST LOSS", String(format: "%.4f", last)))
            }
            if let best = run.bestVal {
                ws.append(WellSpec("BEST VAL", String(format: "%.4f", best)))
            }
        case .quantizeQ4, .quantizeQuaRot:
            if let cfg = run.configSummary {
                // configSummary carries the ratio/verdict text for quant runs
                ws.append(WellSpec("RESULT", cfg))
            }
        case .chat:
            break
        }

        return ws
    }

    private func adapterRevealRow(_ path: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("ADAPTER")
                .instrumentLabel()

            HStack(spacing: Theme.Space.sm) {
                Text(path)
                    .font(Theme.Fonts.cell)
                    .foregroundStyle(Theme.Palette.inkDim)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .monospacedDigit()

                Spacer(minLength: 0)

                Button {
                    let url = URL(fileURLWithPath: path)
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                } label: {
                    Text("Reveal")
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.signal)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(Theme.Space.sm)
        .readoutWellSurface()
    }

    // MARK: Empty state

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: Theme.Space.xl) {
            OpaquePanel {
                VStack(alignment: .leading, spacing: Theme.Space.lg) {
                    Text("NO RUNS YET")
                        .font(Theme.Fonts.title)
                        .foregroundStyle(Theme.Palette.ink)

                    Text("Runs you launch from TRAIN, QUANTIZE, or CHAT land here as lab notebook entries.")
                        .font(Theme.Fonts.body)
                        .foregroundStyle(Theme.Palette.inkDim)
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: Theme.Space.sm) {
                        KeyCapChip("⌘2")
                        Text("Train")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)

                        KeyCapChip("⌘3")
                        Text("Quantize")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)

                        KeyCapChip("⌘4")
                        Text("Chat")
                            .font(Theme.Fonts.body)
                            .foregroundStyle(Theme.Palette.inkDim)
                    }
                }
                .padding(Theme.Space.xl)
            }

            Spacer()
        }
    }

    // MARK: Helpers

    private func gatePillStatus(_ status: RunStatus) -> GateStatus {
        switch status {
        case .running: .run
        case .paused: .warn
        case .done: .pass
        case .failed: .fail
        case .idle: .warn
        }
    }

    private func relativeTimeString(_ date: Date) -> String {
        let interval = Date().timeIntervalSince(date)
        if interval < 60 { return "just now" }
        if interval < 3600 {
            let m = Int(interval / 60)
            return "\(m)m ago"
        }
        let h = Int(interval / 3600)
        if h < 24 { return "\(h)h ago" }
        let d = Int(interval / 86400)
        return "\(d)d ago"
    }
}
