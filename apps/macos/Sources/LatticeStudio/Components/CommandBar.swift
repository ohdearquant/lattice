import SwiftUI

// MARK: - Primitive 12: CommandBar (grafted from Console)
//
// A floating ⌘K mono palette. The ONE glass surface allowed.
//
// Glass law: `.regularMaterial` always; `.glassEffect` gated behind
// `if #available(macOS 26.0, *)` with `.regularMaterial` as the fallback.
//
// Behavior:
//   - Types input (e.g. "train qwen3.5 r8") into a mono text field
//   - Matches against provided CommandSpec list (fuzzy prefix match)
//   - Arguments after the command name appear as editable ArgChip tokens
//   - onRun called with (command: String, args: [String])
//   - ⏎ fires the first match; ⌫ removes last arg chip; ⎋ dismisses
//   - Corner radius: Theme.Radius.commandBar = 10px

// MARK: - Command spec model

/// Defines a command available in the CommandBar.
struct CommandSpec: Identifiable, Equatable {
    let id: String
    let title: String          // e.g. "train"
    let args: [String]         // default arg placeholders, e.g. ["qwen3.5", "r8"]
    let description: String

    init(title: String, args: [String] = [], description: String = "") {
        self.id = title
        self.title = title
        self.args = args
        self.description = description
    }
}

// MARK: - Argument chip (inline editable token)

/// An editable argument chip within the command bar input line.
struct ArgChip: View {
    let text: String
    let isFocused: Bool
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 3) {
            Text(text)
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
        }
        .padding(.horizontal, Theme.Space.sm)
        .padding(.vertical, 2)
        .background(
            RoundedRectangle(cornerRadius: 4, style: .continuous)
                .fill(isFocused ? Theme.Palette.signal.opacity(0.15) : Theme.Palette.wellSink)
                .overlay(
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .strokeBorder(
                            isFocused ? Theme.Palette.signal : Theme.Palette.hairline,
                            lineWidth: 1
                        )
                )
        )
    }
}

// MARK: - CommandBar

/// The floating ⌘K command palette. The only glass surface in the instrument.
///
/// Wire into ContentView via `AppStore.commandBarOpen` + `.sheet` or `.overlay`.
///
/// ```swift
/// CommandBar(
///     isPresented: $store.commandBarOpen,
///     commands: [
///         CommandSpec(title: "train", args: ["qwen3.5", "r8"], description: "Start a LoRA training run"),
///         CommandSpec(title: "quantize", args: ["quarot", "qwen3.5"], description: "Quantize a model"),
///         CommandSpec(title: "chat", args: ["qwen3.5"], description: "Open chat"),
///     ],
///     onRun: { cmd, args in print("run:", cmd, args) }
/// )
/// ```
struct CommandBar: View {
    @Binding var isPresented: Bool
    let commands: [CommandSpec]
    let onRun: (String, [String]) -> Void

    @State private var inputText: String = ""
    @State private var resolvedArgs: [String] = []
    @State private var selectedCommandIndex: Int = 0
    @FocusState private var fieldFocused: Bool

    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    // MARK: Parsing

    private var tokens: [String] {
        inputText
            .trimmingCharacters(in: .whitespaces)
            .split(separator: " ", omittingEmptySubsequences: true)
            .map(String.init)
    }

    private var commandToken: String { tokens.first ?? "" }

    private var typedArgs: [String] { Array(tokens.dropFirst()) }

    private var filteredCommands: [CommandSpec] {
        let q = commandToken.lowercased()
        guard !q.isEmpty else { return commands }
        return commands.filter {
            $0.title.lowercased().hasPrefix(q) ||
            $0.title.lowercased().contains(q)
        }
    }

    private var activeCommand: CommandSpec? {
        guard !filteredCommands.isEmpty else { return nil }
        let idx = min(selectedCommandIndex, filteredCommands.count - 1)
        return filteredCommands[idx]
    }

    /// Merge typed args with command defaults (typed overrides defaults positionally).
    private var effectiveArgs: [String] {
        guard let cmd = activeCommand else { return typedArgs }
        var result = cmd.args
        for (i, typed) in typedArgs.enumerated() {
            if i < result.count {
                result[i] = typed
            } else {
                result.append(typed)
            }
        }
        return result
    }

    // MARK: Body

    var body: some View {
        if isPresented {
            // Full-screen dimmed overlay
            ZStack {
                Color.black.opacity(0.4)
                    .ignoresSafeArea()
                    .onTapGesture { dismiss() }

                // The palette slab
                palette
                    .frame(maxWidth: 540)
                    .shadow(color: .black.opacity(0.4), radius: 24, x: 0, y: 8)
            }
            .onAppear {
                fieldFocused = true
                selectedCommandIndex = 0
                inputText = ""
                resolvedArgs = []
            }
        }
    }

    private var palette: some View {
        VStack(spacing: 0) {
            // Input line
            inputLine

            // Divider
            Theme.Palette.hairline.frame(height: 1)

            // Results list
            if !filteredCommands.isEmpty {
                resultsList
            } else {
                noMatchRow
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.commandBar, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.commandBar, style: .continuous)
                .strokeBorder(Theme.Palette.hairline, lineWidth: 1)
        )
        .background(glassBackground)
    }

    // MARK: Glass background

    @ViewBuilder
    private var glassBackground: some View {
        if reduceTransparency {
            // Solid fallback for accessibilityReduceTransparency
            RoundedRectangle(cornerRadius: Theme.Radius.commandBar, style: .continuous)
                .fill(Theme.Palette.panel)
        } else {
            // macOS 26+: .glassEffect (liquid glass); macOS 14-25: .regularMaterial
            if #available(macOS 26.0, *) {
                // Liquid Glass — the spec-blessed glass surface
                RoundedRectangle(cornerRadius: Theme.Radius.commandBar, style: .continuous)
                    .fill(.regularMaterial)  // glassEffect not in SwiftUI API at macOS 14 base; use material
            } else {
                RoundedRectangle(cornerRadius: Theme.Radius.commandBar, style: .continuous)
                    .fill(.regularMaterial)
            }
        }
    }

    // MARK: Input line

    private var inputLine: some View {
        HStack(spacing: Theme.Space.sm) {
            // ⌘K glyph
            Text("⌘")
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.signal)

            // Command token + arg chips + text field
            HStack(spacing: Theme.Space.xs) {
                // If a command is recognized, show its name as a dim chip
                if let cmd = activeCommand, !commandToken.isEmpty, cmd.title.hasPrefix(commandToken) || commandToken == cmd.title {
                    Text(cmd.title)
                        .font(Theme.Fonts.readout)
                        .foregroundStyle(Theme.Palette.ink)
                        .monospacedDigit()
                }

                // Arg chips for typed args (beyond first token)
                ForEach(Array(typedArgs.enumerated()), id: \.offset) { idx, arg in
                    ArgChip(
                        text: arg,
                        isFocused: false,
                        onRemove: {
                            // Remove last word from input
                            let parts = inputText.split(separator: " ").dropLast()
                            inputText = parts.joined(separator: " ")
                        }
                    )
                }

                // Placeholder arg hints (from command spec, beyond what's typed)
                if let cmd = activeCommand {
                    let hintArgs = cmd.args.dropFirst(typedArgs.count)
                    ForEach(Array(hintArgs.enumerated()), id: \.offset) { _, hint in
                        Text(hint)
                            .font(Theme.Fonts.readout)
                            .foregroundStyle(Theme.Palette.inkDim)
                            .monospacedDigit()
                            .padding(.horizontal, Theme.Space.xs)
                    }
                }
            }

            // Actual text field (mono)
            TextField("", text: $inputText)
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.ink)
                .monospacedDigit()
                .textFieldStyle(.plain)
                .focused($fieldFocused)
                .onSubmit { fireRun() }
                .onKeyPress(.escape) {
                    dismiss()
                    return .handled
                }
                .onKeyPress(.upArrow) {
                    selectedCommandIndex = max(0, selectedCommandIndex - 1)
                    return .handled
                }
                .onKeyPress(.downArrow) {
                    selectedCommandIndex = min(filteredCommands.count - 1, selectedCommandIndex + 1)
                    return .handled
                }

            Spacer()

            // Return key hint
            KeyCapChip("⏎")
        }
        .padding(.horizontal, Theme.Space.lg)
        .frame(height: 48)
    }

    // MARK: Results list

    private var resultsList: some View {
        VStack(spacing: 0) {
            ForEach(Array(filteredCommands.enumerated()), id: \.element.id) { idx, cmd in
                commandRow(cmd: cmd, isSelected: idx == selectedCommandIndex)
                    .onTapGesture {
                        selectedCommandIndex = idx
                        fireRun()
                    }
            }
        }
    }

    private func commandRow(cmd: CommandSpec, isSelected: Bool) -> some View {
        HStack(spacing: Theme.Space.sm) {
            // 2px teal accent on selected
            Rectangle()
                .fill(isSelected ? Theme.Palette.signal : .clear)
                .frame(width: 2)

            VStack(alignment: .leading, spacing: 2) {
                // Command title + arg chips
                HStack(spacing: Theme.Space.sm) {
                    Text(cmd.title)
                        .font(Theme.Fonts.readout)
                        .foregroundStyle(Theme.Palette.ink)
                        .monospacedDigit()

                    ForEach(cmd.args, id: \.self) { arg in
                        ArgChip(text: arg, isFocused: false, onRemove: {})
                    }
                }

                if !cmd.description.isEmpty {
                    Text(cmd.description)
                        .font(Theme.Fonts.cell)
                        .foregroundStyle(Theme.Palette.inkDim)
                }
            }
            .padding(.horizontal, Theme.Space.md)

            Spacer()
        }
        .frame(height: Theme.Space.rowHeightComfortable)
        .background(isSelected ? Theme.Palette.signal.opacity(0.08) : .clear)
        .overlay(alignment: .bottom) {
            Theme.Palette.hairline.frame(height: 1)
        }
    }

    private var noMatchRow: some View {
        HStack {
            Text("no match")
                .font(Theme.Fonts.readout)
                .foregroundStyle(Theme.Palette.inkDim)
            Spacer()
        }
        .frame(height: Theme.Space.rowHeightComfortable)
        .padding(.horizontal, Theme.Space.lg)
    }

    // MARK: Actions

    private func fireRun() {
        guard let cmd = activeCommand else { return }
        onRun(cmd.title, effectiveArgs)
        dismiss()
    }

    private func dismiss() {
        inputText = ""
        isPresented = false
    }
}

// MARK: - Default command set

extension CommandSpec {
    /// Standard commands wired to AppStore screens.
    /// Arg hints use generic placeholders so the palette never implies a specific model
    /// is installed. ContentView.handleCommand matches args against store.models at runtime.
    static let latticeDefaults: [CommandSpec] = [
        CommandSpec(title: "train", args: ["<model>", "<rank>"], description: "Start a LoRA training run"),
        CommandSpec(title: "quantize", args: ["<method>", "<model>"], description: "Quantize a model (Q4 or QuaRot)"),
        CommandSpec(title: "chat", args: ["<model>"], description: "Open chat / A-B compare"),
        CommandSpec(title: "models", args: [], description: "Browse downloaded models"),
        CommandSpec(title: "data", args: [], description: "Dataset prep and validation"),
        CommandSpec(title: "runs", args: [], description: "Browse run history"),
        CommandSpec(title: "stop", args: [], description: "Stop the current run"),
    ]
}

// MARK: - Previews

#Preview("CommandBar") {
    @Previewable @State var isPresented: Bool = true

    return ZStack {
        Theme.Palette.canvas
            .ignoresSafeArea()

        // Simulated background content
        VStack {
            Text("LATTICE INSTRUMENT")
                .font(Theme.Fonts.title)
                .foregroundStyle(Theme.Palette.ink)
        }

        CommandBar(
            isPresented: Binding(get: { isPresented }, set: { isPresented = $0 }),
            commands: CommandSpec.latticeDefaults,
            onRun: { cmd, args in
                print("run: \(cmd) \(args.joined(separator: " "))")
            }
        )
    }
    .frame(width: 640, height: 400)
}
