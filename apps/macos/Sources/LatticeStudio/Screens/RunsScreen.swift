import SwiftUI
import AppKit

// MARK: - Run-history embeddable views
//
// RunsScreen no longer renders a top-level screen. Its content is embedded inside
// ChatScreen's "History" segment. The three building blocks are exposed as internal
// (non-private) view-returning helpers on `RunsContent` so ChatScreen can compose them
// without duplicating any logic.
//
// Layout (composed by ChatScreen .history tab):
//   • Live banner (shown while store.liveRun is running/paused)
//   • DataTable of store.runs: KIND · MODEL · STATUS · METRIC · DURATION · WHEN
//   • Empty state when store.runs.isEmpty
//
// Inspector content (composed by ChatScreen .history inspector branch):
//   • Selected-run detail: ReadoutWells + GatePill + adapter "Reveal" affordance

// MARK: - RunsContent

/// Stateful container for the run-history tab. Owns `selectedRunID` so selection
/// survives tab switches within ChatScreen.
struct RunsContent: View {
    @Bindable var store: AppStore
    @Binding var selectedRunID: String?

    var body: some View {
        VStack(spacing: 0) {
            if let live = store.liveRun,
               live.status == .running || live.status == .paused {
                liveBanner(live)
            }

            if store.runs.isEmpty {
                runsEmptyState
            } else {
                runsTable
            }
        }
    }

    // MARK: Live banner

    func liveBanner(_ live: LiveRun) -> some View {
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

            StatusBadge(runStatusBadge(live.status))
        }
        .padding(.horizontal, Theme.Space.lg)
        .padding(.vertical, Theme.Space.sm)
        .background(Theme.Palette.panel)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
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
                minWidth: 88,
                isNumeric: false,
                badge: { runStatusBadge($0.status) }
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
                    return rec.configSummary ?? "—"
                case .chat, .eval, .embed:
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

    // MARK: Empty state

    var runsEmptyState: some View {
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
                }
                .padding(Theme.Space.xl)
            }

            Spacer()
        }
    }

    // MARK: Inspector panel

    var inspectorPanel: some View {
        Group {
            if let run = store.runs.first(where: { $0.id == selectedRunID }) {
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
    }

    // MARK: Run inspector detail

    private func runInspector(_ run: RunRecord) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.lg) {
            HStack(spacing: Theme.Space.sm) {
                Text(run.kind.rawValue)
                    .font(Theme.Fonts.title)
                    .foregroundStyle(Theme.Palette.ink)

                StatusBadge(runStatusBadge(run.status))
            }

            Text(run.model)
                .font(Theme.Fonts.body)
                .foregroundStyle(Theme.Palette.inkDim)
                .lineLimit(1)

            let wells = runWells(for: run)
            if !wells.isEmpty {
                LazyVGrid(
                    columns: [GridItem(.flexible(), spacing: Theme.Space.md), GridItem(.flexible(), spacing: Theme.Space.md)],
                    spacing: Theme.Space.md
                ) {
                    ForEach(wells, id: \.label) { w in
                        ReadoutWell(label: w.label, value: w.value, unit: w.unit, minHeight: 56)
                    }
                }
            }

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

            if let adapterPath = run.adapterPath {
                adapterRevealRow(adapterPath)
            }
        }
    }

    // MARK: Wells

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
                ws.append(WellSpec("RESULT", cfg))
            }
        case .chat, .eval, .embed:
            break
        }

        return ws
    }

    // MARK: Adapter reveal

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

    // MARK: Helpers

    func runStatusBadge(_ status: RunStatus) -> StatusBadge.Status {
        switch status {
        case .running: .running
        case .paused:  .warning
        case .done:    .success
        case .failed:  .error
        case .idle:    .idle
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
