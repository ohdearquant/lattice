import Foundation

// MARK: - Typed driver configs and AppStore convenience methods.
//
// Each config struct maps 1-to-1 with a Lattice CLI binary's argument surface.
// Flag names are taken verbatim from the CLI contract (2026-06-20).
//
// Do NOT import AppStore.swift here — the extension lives in the same module.
// Do NOT edit LatticeBridge.swift or AppStore.swift directly.

// MARK: - TrainConfig

/// Configuration for `train_grad_full` (multi-layer exact-gradient LoRA trainer).
///
/// All defaults match the binary's own defaults so omitting a field is equivalent
/// to not passing the flag. `--json` is always appended so the JSON event protocol
/// is active when the binary gains that flag; older binaries ignore unknown flags.
struct TrainConfig {
    /// Full path to the BF16 model directory (must contain `model.safetensors` / index).
    var modelDir: URL
    /// Directory containing `train.jsonl` and optionally `valid.jsonl`.
    var dataDir: URL
    /// First materialised (trained) layer index. Default 19 (last 5 of 24 for Qwen3.5-0.8B).
    var firstLayer: Int = 19
    /// Adam optimizer steps. Default 25.
    var steps: Int = 25
    /// Learning rate. Default 1e-3.
    var lr: Double = 1e-3
    /// LoRA rank. Default 8.
    var rank: Int = 8
    /// LoRA alpha; effective scale = alpha/rank. Default 16.
    var alpha: Double = 16.0
    /// Max tokens per sample. Default 64.
    var seqLen: Int = 64
    /// Training sample cap. Default 3.
    var maxTrain: Int = 3
    /// Held-out validation samples (0 = disabled). Default 16.
    var maxValid: Int = 16
    /// Print NLL every N steps. Default 5.
    var logEvery: Int = 5
    /// When non-nil, the `--save <path>` flag is passed (Rust side in progress).
    var savePath: URL? = nil

    /// The exact argument array passed to the subprocess.
    var args: [String] {
        var a: [String] = [
            "--model-dir", modelDir.path,
            "--data-dir",  dataDir.path,
            "--first-layer", String(firstLayer),
            "--steps",       String(steps),
            "--lr",          String(lr),
            "--rank",        String(rank),
            "--alpha",       String(alpha),
            "--seq-len",     String(seqLen),
            "--max-train",   String(maxTrain),
            "--max-valid",   String(maxValid),
            "--log-every",   String(logEvery),
            "--json",         // future --json mode; older binaries ignore unknown flags
        ]
        if let save = savePath {
            a += ["--save", save.path]
        }
        return a
    }
}

// MARK: - QuantConfig

/// Quantization method selector.
enum QuantMethod {
    case q4        // `quantize_q4` — plain Q4_0, no rotation
    case quarot    // `quantize_quarot` — Hadamard-rotated Q4_0; requires a seed
}

/// Configuration for `quantize_q4` or `quantize_quarot`.
///
/// Note: both quantizers write all progress to **stderr**, which RunHandle already
/// merges into the event stream — no special handling needed in the driver.
struct QuantConfig {
    /// Full path to the BF16 input model directory.
    var modelDir: URL
    /// Output directory (created by the binary if absent).
    var outputDir: URL
    /// Which quantization method to run.
    var method: QuantMethod
    /// RNG seed for the Hadamard rotation (quarot only).
    /// If nil when method == .quarot, defaults to 0xC0FFEE.
    /// Passed as a decimal string; the binary also accepts `0x...` hex.
    var seed: UInt64? = nil
    /// Skip disk writes (full pipeline still runs).
    var dryRun: Bool = false

    /// The LatticeBinary enum case to use for this config.
    var binary: LatticeBinary {
        switch method {
        case .q4:     return .quantizeQ4
        case .quarot: return .quantizeQuaRot
        }
    }

    /// The exact argument array passed to the subprocess.
    var args: [String] {
        var a: [String] = [
            "--model-dir",  modelDir.path,
            "--output-dir", outputDir.path,
        ]
        if method == .quarot {
            let s = seed ?? 0xC0FFEE
            a += ["--seed", String(s)]
        }
        if dryRun { a.append("--dry-run") }
        return a
    }
}

// MARK: - GenConfig

/// Configuration for `generate_lora` (single-shot generation with optional adapter).
///
/// Either `modelDir` or `model` must be provided; `modelDir` takes priority when both
/// are set (the binary accepts both flags and uses whichever it sees).
/// generate_lora prints output atomically — there are no per-token events.
struct GenConfig {
    /// Full path to the model directory. Takes priority over `model` when set.
    var modelDir: URL? = nil
    /// Model name under `$LATTICE_MODEL_CACHE / $HOME/.lattice/models/`. Used when
    /// `modelDir` is nil.
    var model: String? = nil
    /// Path to a `.safetensors` LoRA adapter file. Nil = run base model only.
    var adapterPath: URL? = nil
    /// Text prompt fed to the model.
    var prompt: String
    /// Maximum tokens to generate. Default 64.
    var maxTokens: Int = 64
    /// RNG seed for deterministic sampling. Nil = non-deterministic.
    var seed: UInt64? = nil
    /// Sampling temperature. Default 0.7.
    var temperature: Double = 0.7

    /// The exact argument array passed to the subprocess.
    var args: [String] {
        var a: [String] = []
        if let dir = modelDir {
            a += ["--model-dir", dir.path]
        } else if let name = model {
            a += ["--model", name]
        }
        if let adapter = adapterPath {
            a += ["--lora", adapter.path]
        }
        a += [
            "--prompt",     prompt,
            "--max-tokens", String(maxTokens),
            "--temperature", String(temperature),
            "--json",         // streaming gen_token event protocol; older binaries ignore unknown flags
        ]
        if let s = seed {
            a += ["--seed", String(s)]
        }
        return a
    }
}

// MARK: - AppStore typed convenience methods

extension AppStore {

    /// Launch `train_grad_full` with the given config.
    ///
    /// The returned `LiveRun` is already wired into `self.liveRun` and will receive
    /// events via `consume(_:into:)`.
    @discardableResult
    @MainActor
    func startTrain(_ config: TrainConfig) -> LiveRun {
        let modelName = config.modelDir.lastPathComponent
        return launch(
            .trainGradFull,
            args: config.args,
            kind: .train,
            model: modelName,
            totalSteps: config.steps
        )
    }

    /// Launch `quantize_q4` or `quantize_quarot` depending on `config.method`.
    @discardableResult
    @MainActor
    func startQuantize(_ config: QuantConfig) -> LiveRun {
        resetQuantAccumulator()
        let modelName = config.modelDir.lastPathComponent
        let kind: RunKind = config.method == .q4 ? .quantizeQ4 : .quantizeQuaRot
        return launch(
            config.binary,
            args: config.args,
            kind: kind,
            model: modelName,
            totalSteps: nil
        )
    }

    /// Launch `generate_lora` (single-shot; output arrives as one `.status` burst).
    @discardableResult
    @MainActor
    func runGenerate(_ config: GenConfig) -> LiveRun {
        // Derive a display name: prefer the explicit dir/model name, fall back to prompt prefix.
        let modelName: String
        if let dir = config.modelDir {
            modelName = dir.lastPathComponent
        } else if let name = config.model {
            modelName = name
        } else {
            let prefix = config.prompt.prefix(24)
            modelName = prefix.isEmpty ? "generate_lora" : "\(prefix)…"
        }
        return launch(
            .generateLora,
            args: config.args,
            kind: .chat,     // closest RunKind for inference; no .generate case exists
            model: modelName,
            totalSteps: nil
        )
    }
}
