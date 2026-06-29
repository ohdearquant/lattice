import SwiftUI
import AppKit

// MARK: - Serve tab — two-column layout
//
// Controls the OpenAI-compatible HTTP daemon (`lattice_serve`).
// LEFT panel  : server header, big address, start/stop, endpoint table, live request log.
// RIGHT panel : copyable curl + Python client blocks.
//
// HONEST DATA ONLY — status and log rows read through store.serveDisplay* accessors.
// Empty states are shown verbatim; no placeholder rows or fabricated status ever reach the ship path.
//
// Note: each request is stateless — the serve loop resets KV state per call, so there is no
// cross-turn prefix cache. The whole prompt is re-prefilled every request.

struct ServeTab: View {
    @Bindable var store: AppStore

    // MARK: - Derived state

    private var running: Bool { store.serveDisplayRunning }
    private var ready: Bool   { store.serveDisplayReady }
    private var port: Int     { store.serveDisplayPort }

    // MARK: - Status indicator helpers

    private var statusDotColor: Color {
        if running && ready { return Theme.Palette.success }
        if running          { return Theme.Palette.amber }
        return Theme.Palette.idle
    }

    private var statusText: String {
        if running && ready { return "healthy" }
        if running          { return "starting\u{2026}" }
        return "off"
    }

    private var statusTextColor: Color {
        if running && ready { return Theme.Palette.success }
        if running          { return Theme.Palette.amber }
        return Theme.Palette.textTertiary
    }

    // MARK: - Code block strings

    private var curlText: String {
        let model = store.serveDisplayServingModel ?? store.targetModel?.name ?? "model"
        return """
        curl http://127.0.0.1:\(port)/v1/chat/completions \\
          -H "Content-Type: application/json" \\
          -d '{
            "model": "\(model)",
            "stream": true,
            "messages": [
              {"role": "user", "content": "Explain the hybrid layer stripe."}
            ]
          }'
        """
    }

    private var pythonText: String {
        let model = store.serveDisplayServingModel ?? store.targetModel?.name ?? "model"
        return """
        from openai import OpenAI

        client = OpenAI(
            base_url="http://127.0.0.1:\(port)/v1",
            api_key="lattice-local",
        )

        stream = client.chat.completions.create(
            model="\(model)",
            messages=[{"role": "user", "content": "What is a GDN layer?"}],
            stream=True,
        )

        for event in stream:
            print(event.choices[0].delta.content or "", end="")
        """
    }

    // MARK: - Root body

    var body: some View {
        ScrollView {
            HStack(alignment: .top, spacing: Theme.Space.lg) {
                leftPanel
                    .frame(minWidth: 420, maxWidth: .infinity)
                rightPanel
                    .frame(minWidth: 360, maxWidth: .infinity)
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .padding(Theme.Space.xl)
        }
    }

    // MARK: - LEFT panel

    private var leftPanel: some View {
        OpaquePanel {
            VStack(alignment: .leading, spacing: Theme.Space.md) {

                // 1. Header row: eyebrow label + status indicator
                HStack(spacing: Theme.Space.xs) {
                    Text("OPENAI-COMPATIBLE SERVER")
                        .font(Theme.Fonts.sectionLabel)
                        .textCase(.uppercase)
                        .tracking(Theme.Space.labelTracking)
                        .foregroundStyle(Theme.Palette.textTertiary)
                    Spacer()
                    HStack(spacing: 5) {
                        Circle()
                            .fill(statusDotColor)
                            .frame(width: 7, height: 7)
                        Text(statusText)
                            .font(Theme.Fonts.caption)
                            .foregroundStyle(statusTextColor)
                    }
                }

                // 2. Big address (verbatim: prevents locale digit-grouping on port)
                Text(verbatim: "127.0.0.1:\(port)")
                    .font(Theme.Fonts.largeMetric)
                    .foregroundStyle(Theme.Palette.textPrimary)
                    .textSelection(.enabled)

                // 3. Start/Stop control row
                controlRow

                // 4. Endpoint table — dimmed when server is not running
                endpointTable
                    .opacity(running ? 1.0 : 0.5)

                // 5. Section divider
                Divider().overlay(Theme.Palette.hairline)

                // 6. Log eyebrow
                Text("LIVE REQUEST LOG")
                    .font(Theme.Fonts.sectionLabel)
                    .textCase(.uppercase)
                    .tracking(Theme.Space.labelTracking)
                    .foregroundStyle(Theme.Palette.textTertiary)

                // 7. Log table
                logTable
            }
            .padding(Theme.Space.lg)
        }
    }

    // MARK: Control row

    @ViewBuilder
    private var controlRow: some View {
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            HStack(spacing: Theme.Space.sm) {
                if running {
                    Button {
                        store.serve?.stop()
                    } label: {
                        Text("Stop Server")
                    }
                    .buttonStyle(LatticeSecondaryButtonStyle())
                } else {
                    Button {
                        if let m = store.targetModel { _ = store.serve?.start(model: m) }
                    } label: {
                        Text("Start Server")
                    }
                    .buttonStyle(LatticePrimaryButtonStyle())
                    .disabled(store.targetModel == nil)
                }

                if let name = store.serveDisplayServingModel ?? store.targetModel?.name {
                    Text(name)
                        .font(Theme.Fonts.caption)
                        .foregroundStyle(Theme.Palette.textTertiary)
                        .lineLimit(1)
                }
            }

            if let err = store.serve?.lastError, !running {
                Text(err)
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.crimson)
                    .lineLimit(2)
            }
        }
    }

    // MARK: Endpoint table

    private var endpointTable: some View {
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            endpointRow("base_url", "http://127.0.0.1:\(port)/v1")
            endpointRow("chat",     "/v1/chat/completions \u{00B7} streaming")
            endpointRow("models",   "/v1/models")
            endpointRow("health",   "/health")
        }
    }

    private func endpointRow(_ label: String, _ value: String) -> some View {
        HStack(spacing: Theme.Space.xs) {
            Text(label)
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textTertiary)
                .frame(width: 70, alignment: .leading)
            Spacer()
            Text(value)
                .font(Theme.Fonts.codeFont)
                .foregroundStyle(Theme.Palette.textSecondary)
                .textSelection(.enabled)
                .lineLimit(1)
        }
    }

    // MARK: Live request log

    // Delegates into a helper to keep `let` bindings out of the @ViewBuilder scope.
    private var logTable: some View {
        let entries = store.serveDisplayLog
        let tail = Array(entries.suffix(14))
        return logTableContent(tail: tail)
    }

    @ViewBuilder
    private func logTableContent(tail: [ServeLogEntry]) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            logHeaderRow
            if tail.isEmpty {
                HStack {
                    Spacer()
                    Text(
                        running
                            ? "No requests yet \u{2014} try a client block \u{2192}"
                            : "Start the server to log requests."
                    )
                    .font(Theme.Fonts.caption)
                    .foregroundStyle(Theme.Palette.textTertiary)
                    Spacer()
                }
                .padding(.vertical, Theme.Space.md)
            } else {
                ScrollView(.vertical) {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(tail) { entry in
                            logRowView(entry)
                        }
                    }
                }
                .frame(maxHeight: 220)
            }
        }
    }

    private var logHeaderRow: some View {
        HStack(spacing: 0) {
            Text("time")
                .frame(width: 64, alignment: .leading)
            Text("method")
                .frame(width: 62, alignment: .leading)
            Text("route")
                .frame(minWidth: 150, maxWidth: .infinity, alignment: .leading)
            Text("status")
                .frame(width: 48, alignment: .trailing)
            Text("tokens")
                .frame(width: 52, alignment: .trailing)
            Text("dur")
                .frame(width: 52, alignment: .trailing)
        }
        .font(Theme.Fonts.micro)
        .textCase(.uppercase)
        .tracking(0.45)
        .foregroundStyle(Theme.Palette.textTertiary)
        .padding(.vertical, Theme.Space.xs)
    }

    private func logRowView(_ entry: ServeLogEntry) -> some View {
        let durStr: String = entry.durMs < 1_000
            ? "\(Int(entry.durMs))ms"
            : String(format: "%.1fs", entry.durMs / 1_000)
        let tokStr: String = entry.tokens.map { String($0) } ?? "\u{2014}"
        let statusColor: Color = (200..<300).contains(entry.status)
            ? Theme.Palette.success
            : Theme.Palette.crimson
        let methodColor: Color = entry.method == "GET"
            ? Theme.Palette.textSecondary
            : Theme.Palette.signal

        return HStack(spacing: 0) {
            Text(entry.clock)
                .frame(width: 64, alignment: .leading)
                .foregroundStyle(Theme.Palette.textTertiary)
            Text(entry.method)
                .frame(width: 62, alignment: .leading)
                .foregroundStyle(methodColor)
            Text(entry.route)
                .frame(minWidth: 150, maxWidth: .infinity, alignment: .leading)
                .foregroundStyle(Theme.Palette.textSecondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Text("\(entry.status)")
                .frame(width: 48, alignment: .trailing)
                .foregroundStyle(statusColor)
            Text(tokStr)
                .frame(width: 52, alignment: .trailing)
                .foregroundStyle(Theme.Palette.textTertiary)
            Text(durStr)
                .frame(width: 52, alignment: .trailing)
                .foregroundStyle(Theme.Palette.textTertiary)
        }
        .font(Theme.Fonts.logFont)
        .padding(.vertical, 2)
    }

    // MARK: - RIGHT panel

    private var rightPanel: some View {
        OpaquePanel {
            VStack(alignment: .leading, spacing: Theme.Space.md) {

                // 1. Eyebrow
                Text("COPYABLE CLIENT BLOCKS")
                    .font(Theme.Fonts.sectionLabel)
                    .textCase(.uppercase)
                    .tracking(Theme.Space.labelTracking)
                    .foregroundStyle(Theme.Palette.textTertiary)

                // 2. Section title
                Text("Use the same local port everywhere")
                    .font(Theme.Fonts.inspectorTitle)
                    .foregroundStyle(Theme.Palette.textPrimary)

                // 3. curl block
                CopyableCodeBlock(title: "curl", code: curlText)

                // 4. Python block
                CopyableCodeBlock(title: "Python \u{00B7} openai SDK", code: pythonText)

                // 5. Honest footnote — stateless / no cross-turn prefix cache
                Text(
                    "Each request is stateless \u{2014} the serve loop resets KV state per call, " +
                    "so there is no cross-turn prefix cache."
                )
                .font(Theme.Fonts.caption)
                .foregroundStyle(Theme.Palette.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
            }
            .padding(Theme.Space.lg)
        }
    }
}
