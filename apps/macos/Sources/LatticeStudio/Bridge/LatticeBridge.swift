import Foundation

// MARK: - Which lattice binary, and how to build/run it.
//
// Every binary is reachable two ways: a prebuilt artifact in `target/release` (preferred,
// instant), or `cargo run --release` (fallback, compiles on first use). Compile-time
// `--features` only matter for the cargo fallback — a prebuilt binary already has them baked in.

enum LatticeBinary {
    case trainGradFull        // lattice-tune, features: train-backward
    case quantizeQ4           // lattice-inference
    case quantizeQuaRot       // lattice-inference
    case generateLora         // lattice-tune, features: safetensors,inference-hook
    case lattice              // lattice-inference — `chat` / `serve` subcommands
    case qwen35Generate       // lattice-inference

    var binName: String {
        switch self {
        case .trainGradFull: "train_grad_full"
        case .quantizeQ4: "quantize_q4"
        case .quantizeQuaRot: "quantize_quarot"
        case .generateLora: "generate_lora"
        case .lattice: "lattice"
        case .qwen35Generate: "qwen35_generate"
        }
    }
    var crate: String {
        switch self {
        case .trainGradFull, .generateLora: "lattice-tune"
        default: "lattice-inference"
        }
    }
    var features: [String] {
        switch self {
        case .trainGradFull: ["train-backward"]
        case .generateLora: ["safetensors", "inference-hook"]
        default: []
        }
    }
}

struct LaunchSpec {
    var executable: URL
    var arguments: [String]
    var cwd: URL?
}

enum LatticeBridge {

    // MARK: Path resolution

    static var repoRoot: URL? {
        if let env = ProcessInfo.processInfo.environment["LATTICE_REPO_ROOT"] {
            return URL(fileURLWithPath: env, isDirectory: true)
        }
        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        for _ in 0..<10 {
            let crates = dir.appendingPathComponent("crates", isDirectory: true)
            let cargo = dir.appendingPathComponent("Cargo.toml")
            if FileManager.default.fileExists(atPath: crates.path),
               FileManager.default.fileExists(atPath: cargo.path) {
                return dir
            }
            let parent = dir.deletingLastPathComponent()
            if parent == dir { break }
            dir = parent
        }
        return nil
    }

    static var modelCacheDir: URL {
        if let env = ProcessInfo.processInfo.environment["LATTICE_MODEL_CACHE"] {
            return URL(fileURLWithPath: env, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".lattice/models", isDirectory: true)
    }

    static func prebuiltBinary(_ bin: LatticeBinary) -> URL? {
        let fm = FileManager.default
        if let dir = ProcessInfo.processInfo.environment["LATTICE_BIN_DIR"] {
            let u = URL(fileURLWithPath: dir).appendingPathComponent(bin.binName)
            if fm.isExecutableFile(atPath: u.path) { return u }
        }
        if let root = repoRoot {
            let u = root.appendingPathComponent("target/release/\(bin.binName)")
            if fm.isExecutableFile(atPath: u.path) { return u }
        }
        return nil
    }

    private static var cargoURL: URL? {
        let candidates = [
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cargo/bin/cargo"),
            URL(fileURLWithPath: "/opt/homebrew/bin/cargo"),
            URL(fileURLWithPath: "/usr/local/bin/cargo")
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    /// Produce a launch spec, preferring a prebuilt artifact and falling back to `cargo run`.
    static func launchSpec(_ bin: LatticeBinary, args: [String]) -> LaunchSpec? {
        if let exe = prebuiltBinary(bin) {
            return LaunchSpec(executable: exe, arguments: args, cwd: repoRoot)
        }
        guard let cargo = cargoURL, let root = repoRoot else { return nil }
        var cargoArgs = ["run", "--release", "-p", bin.crate, "--bin", bin.binName]
        if !bin.features.isEmpty {
            cargoArgs += ["--features", bin.features.joined(separator: ",")]
        }
        cargoArgs += ["--"] + args
        return LaunchSpec(executable: cargo, arguments: cargoArgs, cwd: root)
    }

    // MARK: Model discovery — scan the cache into ModelInfo + indented adapters.

    static func discoverModels() -> [ModelInfo] {
        let fm = FileManager.default
        let root = modelCacheDir
        guard let entries = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else {
            return []
        }
        var models: [ModelInfo] = []
        for entry in entries {
            guard (try? entry.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else { continue }
            if let m = inspectModelDir(entry) { models.append(m) }
        }
        return models.sorted { $0.name < $1.name }
    }

    static func inspectModelDir(_ dir: URL) -> ModelInfo? {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else {
            return nil
        }
        let names = Set(files.map { $0.lastPathComponent })
        let hasSafetensors = names.contains("model.safetensors") || names.contains("model.safetensors.index.json")
        let hasQ4 = files.contains { $0.pathExtension == "q4" }
        guard hasSafetensors || hasQ4 else { return nil }

        let name = dir.lastPathComponent
        let lower = name.lowercased()
        var format: ModelFormat = hasQ4 ? .q4 : .bf16
        if hasQ4 && lower.contains("quarot") { format = .quarot }
        let isEmbedding = ["minilm", "bge", "embedding", "e5", "gte"].contains { lower.contains($0) }
        if isEmbedding { format = .embedding }

        var size: Int64 = 0
        for f in files {
            size += Int64((try? f.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }

        var hidden: Int? = nil
        var vocab: Int? = nil
        var layerSummary: String? = nil
        var dtype = format == .bf16 ? "BF16" : "Q4_0"
        if let cfg = readConfig(dir.appendingPathComponent("config.json")) {
            hidden = cfg["hidden_size"] as? Int
            vocab = cfg["vocab_size"] as? Int
            if let nl = cfg["num_hidden_layers"] as? Int {
                layerSummary = "\(nl) layers"
            }
            if let dt = cfg["torch_dtype"] as? String { dtype = dt.uppercased() }
        }
        // The Qwen3.5 flagship is a known hybrid; surface its layer split when recognizable.
        if lower.contains("qwen3.5-0.8b") || lower.contains("qwen3.5") {
            layerSummary = "18 GDN · 6 GQA"
        }

        let params = parseParamCount(from: name)
        // Scan for adapters: loose .safetensors in the model dir (excluding the base
        // model.safetensors itself), plus anything in an `adapters/` subdirectory.
        let looseAdapters = discoverAdapters(in: dir)
            .filter { $0.name != "model" }
        let adapterSubdir = dir.appendingPathComponent("adapters", isDirectory: true)
        let subdirAdapters = discoverAdapters(in: adapterSubdir)
        let adapters = (looseAdapters + subdirAdapters).sorted { $0.name < $1.name }
        return ModelInfo(
            name: name, path: dir, format: format, params: params, dtype: dtype,
            sizeBytes: size, fileCount: files.count, hasTokenizer: names.contains("tokenizer.json"),
            layerSummary: layerSummary, hidden: hidden, vocab: vocab, isEmbedding: isEmbedding,
            adapters: adapters
        )
    }

    static func readConfig(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj
    }

    /// "qwen3.5-0.8b" -> "0.8B". Catches B/M suffix with optional decimal.
    static func parseParamCount(from name: String) -> String? {
        let pattern = #"(\d+(?:\.\d+)?)\s*([bBmM])"#
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(name.startIndex..., in: name)
        guard let m = re.firstMatch(in: name, range: range),
              let numR = Range(m.range(at: 1), in: name),
              let unitR = Range(m.range(at: 2), in: name) else { return nil }
        return "\(name[numR])\(name[unitR].uppercased())"
    }

    /// Scan a directory of `.safetensors` adapter files into AdapterInfo.
    static func discoverAdapters(in dir: URL) -> [AdapterInfo] {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else {
            return []
        }
        return files.filter { $0.pathExtension == "safetensors" }.map { f in
            let size = Int64((try? f.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
            return AdapterInfo(name: f.deletingPathExtension().lastPathComponent, path: f,
                               rank: nil, alpha: nil, targetModules: nil, sizeBytes: size)
        }.sorted { $0.name < $1.name }
    }
}

// MARK: - A single running lattice subprocess: streaming, pausable, killable.

final class RunHandle {
    private let process = Process()
    private let outPipe = Pipe()
    private let inPipe = Pipe()
    private var buffer = Data()
    private(set) var isPaused = false

    var onEvent: ((LatticeEvent) -> Void)?
    var onExit: ((Int32) -> Void)?

    var isRunning: Bool { process.isRunning }

    func start(_ spec: LaunchSpec, env: [String: String]? = nil) throws {
        process.executableURL = spec.executable
        process.arguments = spec.arguments
        if let cwd = spec.cwd { process.currentDirectoryURL = cwd }

        var environment = ProcessInfo.processInfo.environment
        env?.forEach { environment[$0.key] = $0.value }
        process.environment = environment

        process.standardOutput = outPipe
        process.standardError = outPipe
        process.standardInput = inPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            self?.ingest(data)
        }
        process.terminationHandler = { [weak self] proc in
            self?.outPipe.fileHandleForReading.readabilityHandler = nil
            let code = proc.terminationStatus
            DispatchQueue.main.async { self?.onExit?(code) }
        }
        try process.run()
    }

    private func ingest(_ data: Data) {
        buffer.append(data)
        let newline: UInt8 = 0x0A
        while let nl = buffer.firstIndex(of: newline) {
            let lineData = buffer.subdata(in: buffer.startIndex..<nl)
            buffer.removeSubrange(buffer.startIndex...nl)
            guard let text = String(data: lineData, encoding: .utf8),
                  let event = LatticeEventParser.parse(line: text) else { continue }
            DispatchQueue.main.async { [weak self] in self?.onEvent?(event) }
        }
    }

    /// Write a line to the child's stdin (for `lattice chat` / `chat_metal`).
    func send(line text: String) {
        guard let data = (text + "\n").data(using: .utf8) else { return }
        inPipe.fileHandleForWriting.write(data)
    }

    func closeInput() {
        try? inPipe.fileHandleForWriting.close()
    }

    func pause() {
        guard process.isRunning else { return }
        kill(process.processIdentifier, SIGSTOP)
        isPaused = true
    }
    func resume() {
        guard process.isRunning else { return }
        kill(process.processIdentifier, SIGCONT)
        isPaused = false
    }
    func stop() {
        guard process.isRunning else { return }
        process.terminate()
    }
}
