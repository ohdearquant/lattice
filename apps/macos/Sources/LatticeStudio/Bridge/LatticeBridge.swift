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
        // (a) Bundled binary — preferred when running as a .app from /Applications.
        if let resourceURL = Bundle.main.resourceURL {
            let u = resourceURL.appendingPathComponent("bin/\(bin.binName)")
            if fm.isExecutableFile(atPath: u.path) { return u }
        }
        // (b) Explicit LATTICE_BIN_DIR override — dev convenience.
        if let dir = ProcessInfo.processInfo.environment["LATTICE_BIN_DIR"] {
            let u = URL(fileURLWithPath: dir).appendingPathComponent(bin.binName)
            if fm.isExecutableFile(atPath: u.path) { return u }
        }
        // (c) Repo-relative target/release — running from source checkout.
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
        var contextLen: Int? = nil
        var attnHeads: Int? = nil
        var kvHeads: Int? = nil
        var headDim: Int? = nil
        var gdnKeyHeads: Int? = nil
        var gdnValueHeads: Int? = nil
        var dtype = format == .bf16 ? "BF16" : "Q4_0"
        if let rawCfg = readConfig(dir.appendingPathComponent("config.json")) {
            // Some models (e.g. MLX VLM repacks) nest text fields under `text_config`.
            // Prefer the nested dict when present; fall back to top-level.
            let cfg: [String: Any]
            if let nested = rawCfg["text_config"] as? [String: Any] {
                cfg = nested
            } else {
                cfg = rawCfg
            }
            hidden = cfg["hidden_size"] as? Int
            vocab = cfg["vocab_size"] as? Int
            // Real max context (HF `max_position_embeddings`). `cfg` is the nested
            // text_config for qwen3.5 or the top-level dict for flat configs, so one
            // read covers both layouts. Stays nil (CTX well hidden) when no config.json.
            contextLen = cfg["max_position_embeddings"] as? Int
            // Attention head config — same nested-resolved cfg; honest-nil for flat/absent configs.
            attnHeads = cfg["num_attention_heads"] as? Int
            kvHeads = cfg["num_key_value_heads"] as? Int
            headDim = cfg["head_dim"] as? Int
            gdnKeyHeads = cfg["linear_num_key_heads"] as? Int
            gdnValueHeads = cfg["linear_num_value_heads"] as? Int
            // Derive layer summary from real `layer_types` array when available.
            if let layerTypes = cfg["layer_types"] as? [String] {
                // Count each type and surface all non-zero counts.
                var gdn = 0
                var gqa = 0
                var other: [String: Int] = [:]
                for t in layerTypes {
                    switch t {
                    case "linear_attention": gdn += 1
                    case "full_attention":   gqa += 1
                    default:
                        other[t, default: 0] += 1
                    }
                }
                var parts: [String] = []
                if gdn > 0 { parts.append("\(gdn) GDN") }
                if gqa > 0 { parts.append("\(gqa) GQA") }
                for (typeName, count) in other.sorted(by: { $0.key < $1.key }) {
                    parts.append("\(count) \(typeName)")
                }
                layerSummary = parts.joined(separator: " · ")
            } else if let nl = cfg["num_hidden_layers"] as? Int {
                // No layer_types available; surface the raw count honestly.
                layerSummary = "\(nl) layers"
            }
            // (If neither field is present, layerSummary stays nil and renders "—".)
            if let dt = rawCfg["torch_dtype"] as? String { dtype = dt.uppercased() }
        }

        let params = parseParamCount(from: name)
        // Scan for adapters: loose .safetensors in the model dir (excluding model weight
        // shards), plus anything in an `adapters/` subdirectory.
        let looseAdapters = discoverAdapters(in: dir)
        let adapterSubdir = dir.appendingPathComponent("adapters", isDirectory: true)
        let subdirAdapters = discoverAdapters(in: adapterSubdir)
        let adapters = (looseAdapters + subdirAdapters).sorted { $0.name < $1.name }
        return ModelInfo(
            name: name, path: dir, format: format, params: params, dtype: dtype,
            sizeBytes: size, fileCount: files.count, hasTokenizer: names.contains("tokenizer.json"),
            layerSummary: layerSummary, hidden: hidden, vocab: vocab, contextLength: contextLen,
            attnHeads: attnHeads, kvHeads: kvHeads, headDim: headDim,
            gdnKeyHeads: gdnKeyHeads, gdnValueHeads: gdnValueHeads,
            isEmbedding: isEmbedding, adapters: adapters
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

    /// Return true if a filename looks like a sharded model weight, not an adapter.
    /// Matches: model.safetensors, model.safetensors.index.json,
    ///          model-00001-of-00002.safetensors, model.safetensors-00001-of-00001.safetensors
    private static func isModelWeight(_ filename: String) -> Bool {
        let lower = filename.lowercased()
        // Exact base weight
        if lower == "model.safetensors" { return true }
        // Shard patterns: model-00001-of-00002.safetensors
        if lower.hasPrefix("model-") && lower.hasSuffix(".safetensors") { return true }
        // Double-extension shard: model.safetensors-00001-of-00001.safetensors
        if lower.hasPrefix("model.safetensors-") && lower.hasSuffix(".safetensors") { return true }
        return false
    }

    /// Read the JSON header from a `.safetensors` file and return the `__metadata__` map.
    ///
    /// safetensors binary layout: bytes [0..8) = little-endian UInt64 N, bytes [8..8+N) = UTF-8
    /// JSON. The JSON object may contain a `"__metadata__"` key whose value is [String: String].
    /// Only the 8-byte length prefix and the N header bytes are read — the tensor payload is
    /// never touched, so this is safe even for multi-hundred-MB adapter files.
    ///
    /// Returns nil on any failure (truncated file, bad JSON, missing key, N out of bounds).
    static func readSafetensorsMetadata(_ url: URL) -> [String: String]? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        // Read the 8-byte little-endian header length.
        guard let lenData = try? handle.read(upToCount: 8), lenData.count == 8 else { return nil }
        let n = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }

        // Sanity-bound: reject empty or suspiciously large headers.
        guard n > 0, n <= 100_000_000 else { return nil }

        guard let headerData = try? handle.read(upToCount: Int(n)), headerData.count == Int(n) else {
            return nil
        }
        guard let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any],
              let meta = obj["__metadata__"] as? [String: String]
        else { return nil }

        return meta
    }

    /// Read adapter metadata from a PEFT-format `adapter_config.json` sibling file.
    ///
    /// Handles the standard PEFT keys (`r`, `lora_alpha`, `target_modules`). Used as a
    /// fallback when the safetensors `__metadata__` block is absent or empty, for adapters
    /// that were produced outside of lattice (e.g. imported from Hugging Face).
    private static func readPeftAdapterConfig(sibling url: URL) -> (rank: Int?, alpha: Double?, targetModules: String?)? {
        let configURL = url.deletingLastPathComponent().appendingPathComponent("adapter_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        let rank = obj["r"] as? Int
        let alpha: Double?
        if let a = obj["lora_alpha"] as? Double {
            alpha = a
        } else if let a = obj["lora_alpha"] as? Int {
            alpha = Double(a)
        } else {
            alpha = nil
        }
        let targetModules: String?
        if let mods = obj["target_modules"] as? [String], !mods.isEmpty {
            targetModules = mods.joined(separator: ", ")
        } else {
            targetModules = nil
        }

        return (rank: rank, alpha: alpha, targetModules: targetModules)
    }

    /// Scan a directory of `.safetensors` adapter files into AdapterInfo.
    /// Excludes sharded model weights (model*.safetensors, model-*-of-*.safetensors,
    /// model.safetensors-*-of-*.safetensors) which are base model files, not adapters.
    ///
    /// Metadata resolution order for each adapter file:
    ///   1. `__metadata__` block in the safetensors header (lattice-native format).
    ///   2. Sibling `adapter_config.json` using PEFT keys `r`/`lora_alpha`/`target_modules`
    ///      (for externally-imported adapters only).
    ///   3. All three fields stay nil when neither source is present (honest result).
    static func discoverAdapters(in dir: URL) -> [AdapterInfo] {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else {
            return []
        }
        return files
            .filter { $0.pathExtension == "safetensors" && !isModelWeight($0.lastPathComponent) }
            .map { f in
                let size = Int64((try? f.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)

                // --- Metadata resolution: safetensors __metadata__ first ---
                var rank: Int? = nil
                var alpha: Double? = nil
                var targetModules: String? = nil

                if let meta = readSafetensorsMetadata(f), !meta.isEmpty {
                    rank = meta["rank"].flatMap { Int($0) }
                    alpha = meta["alpha"].flatMap { Double($0) }
                    // Normalize: split on commas, trim whitespace, rejoin.
                    if let raw = meta["target_modules"], !raw.isEmpty {
                        let parts = raw.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                        if !parts.isEmpty { targetModules = parts.joined(separator: ", ") }
                    }
                } else if let peft = readPeftAdapterConfig(sibling: f) {
                    // --- Fallback: PEFT adapter_config.json for imported adapters ---
                    rank = peft.rank
                    alpha = peft.alpha
                    targetModules = peft.targetModules
                }

                return AdapterInfo(name: f.deletingPathExtension().lastPathComponent, path: f,
                                   rank: rank, alpha: alpha, targetModules: targetModules, sizeBytes: size)
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
    /// The OS-assigned process identifier. Valid after `start(_:)` returns without throwing.
    var pid: Int32 { process.processIdentifier }

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
