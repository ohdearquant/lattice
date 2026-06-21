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
/// to not passing the flag. `--json` is always appended to activate the structured
/// `@@lattice` event protocol (verified emitted by `train_grad_full --json`, 2026-06-21).
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
            "--json",         // structured @@lattice event protocol (verified train_grad_full)
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
/// generate_lora streams per-token `gen_token` events in `--json` mode (always enabled via args).
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

// MARK: - EvalConfig

/// Configuration for `eval_perplexity` (strided sliding-window perplexity, ADR-044).
///
/// Three measurement modes — pass the corresponding dir flag to select:
///   • CPU BF16:    set `modelDir`  (embeds tokenizer; `tokenizerDir` not needed)
///   • Metal Q4:   set `q4Dir`     + `tokenizerDir`
///   • QuaRot Q4:  set `quarotDir` + optionally `q4Dir` for the delta comparison
///
/// `--json` is always appended to activate the structured `@@lattice perplexity` event
/// protocol (shipped 2026-06-21). `--label` is included only when `label` is set
/// (disambiguates rows in multi-pass runs). Adapter perplexity is NOT supported: the CPU
/// forward path has no LoRA hook, so adapter A/B lives in Chat via generate_lora instead.
///
/// Flag names are taken verbatim from `crates/inference/src/bin/eval_perplexity.rs`
/// arg-parser (verified 2026-06-21).
struct EvalConfig {
    /// CPU mode: directory with `config.json` + safetensors + `tokenizer.json`.
    var modelDir: URL? = nil
    /// Metal Q4 mode: `quantize_q4` output directory (unrotated 4-bit weights).
    var q4Dir: URL? = nil
    /// Metal QuaRot Q4 directory; when combined with `q4Dir` triggers the delta comparison.
    var quarotDir: URL? = nil
    /// Tokenizer directory (Metal modes); holds `tokenizer.json` from the source checkpoint.
    var tokenizerDir: URL? = nil
    /// UTF-8 text corpus file to score (required).
    var corpusFile: URL
    /// Cap total tokens after tokenization (nil = no cap — binary default).
    var maxTokens: Int? = nil
    /// Human label for this measurement row (e.g. "bf16", "q4", "adapter").
    var label: String? = nil

    /// The exact argument array passed to the subprocess.
    var args: [String] {
        var a: [String] = []
        if let dir = modelDir    { a += ["--model-dir",      dir.path] }
        if let dir = q4Dir       { a += ["--q4-dir",         dir.path] }
        if let dir = quarotDir   { a += ["--quarot-q4-dir",  dir.path] }
        if let dir = tokenizerDir { a += ["--tokenizer-dir", dir.path] }
        a += ["--corpus-file", corpusFile.path]
        if let mt = maxTokens    { a += ["--max-tokens",     String(mt)] }
        if let lbl = label       { a += ["--label",          lbl] }
        a.append("--json")       // structured @@lattice perplexity event protocol
        return a
    }
}

// MARK: - EmbedConfig

/// Configuration for `embed` (batch text embedding with cosine similarity report).
///
/// The binary accepts a model identifier and one or more `--text` values.
/// `--json` is always appended to emit the structured `@@lattice embed_done` event.
///
/// `model` is the only positional parameter; texts are each preceded by `--text`.
struct EmbedConfig {
    /// Model identifier — a short name resolvable by the embed binary (e.g. "bge-small-en-v1.5").
    var model: String
    /// Texts to embed. Each becomes one `--text <value>` flag pair.
    var texts: [String]

    /// The exact argument array passed to the subprocess.
    var args: [String] {
        var a: [String] = ["--model", model]
        for text in texts {
            a += ["--text", text]
        }
        a.append("--json")       // structured @@lattice embed_done event protocol
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

    /// Launch `eval_perplexity` with the given config.
    ///
    /// The returned `LiveRun` accumulates `perplexity` events in `run.perplexities`.
    /// A single run may emit multiple rows (one per label). Use `run.onComplete` to
    /// chain a follow-up run in A/B sequences (e.g. base eval → adapter eval).
    @discardableResult
    @MainActor
    func runEval(_ config: EvalConfig) -> LiveRun {
        // Derive a display name: prefer explicit model/q4/quarot dir name, then label.
        let modelName: String
        if let dir = config.modelDir {
            modelName = dir.lastPathComponent
        } else if let dir = config.q4Dir {
            modelName = dir.lastPathComponent
        } else if let dir = config.quarotDir {
            modelName = dir.lastPathComponent
        } else {
            modelName = config.label ?? "eval"
        }
        return launch(
            .evalPerplexity,
            args: config.args,
            kind: .eval,
            model: modelName,
            totalSteps: nil
        )
    }

    /// Launch `embed` with the given config.
    ///
    /// The returned `LiveRun` receives a single `embed_done` event stored in `run.embed`.
    @discardableResult
    @MainActor
    func runEmbed(_ config: EmbedConfig) -> LiveRun {
        return launch(
            .embed,
            args: config.args,
            kind: .embed,
            model: config.model,
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
