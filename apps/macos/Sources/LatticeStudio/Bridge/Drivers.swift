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

/// Configuration for `generate_lora` (CPU BF16) or `chat_metal` (GPU Metal bf16/q4).
///
/// Either `modelDir` or `model` must be provided; `modelDir` takes priority when both
/// are set (the binary accepts both flags and uses whichever it sees).
///
/// Both binaries emit identical `@@lattice gen_token` streaming events, so the app
/// parser needs no changes when switching between CPU and GPU paths.
///
/// Honest-label contract: the *caller* (ChatScreen.send) sets `useGPU` to select the
/// binary and also embeds the label string in the bubble at send time. There is no path
/// where a CPU run appears labelled as GPU.
struct GenConfig {
    /// Full path to the model directory. Takes priority over `model` when set.
    var modelDir: URL? = nil
    /// Model name under `$LATTICE_MODEL_CACHE / $HOME/.lattice/models/`. Used when
    /// `modelDir` is nil.
    var model: String? = nil
    /// Tokenizer directory override. Required for Q4 models (no embedded tokenizer.json).
    /// When set, `--tokenizer-dir` is appended to the arg array.
    /// Ignored by `generate_lora` (unknown flag, silently dropped).
    var tokenizerDir: URL? = nil
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
    /// Top-k nucleus cutoff. Default 50. Passed as `--top-k` to both binaries.
    /// (verified: generate_lora + chat_metal both accept this flag, 2026-06-22).
    var topK: Int = 50
    /// Top-p nucleus cutoff. Default 0.9. Passed as `--top-p` to both binaries.
    /// (verified: generate_lora + chat_metal both accept this flag, 2026-06-22).
    var topP: Double = 0.9
    /// Repetition penalty. Default 1.1. Passed as `--repetition-penalty` to both binaries.
    /// (verified: generate_lora + chat_metal both accept this flag, 2026-06-22).
    var repetitionPenalty: Double = 1.1
    /// When true, dispatch to `chat_metal` (GPU Metal) instead of `generate_lora` (CPU).
    /// Setting this to true on a non-bf16 model is valid: chat_metal auto-detects Q4.
    var useGPU: Bool = false

    /// The exact argument array passed to the subprocess.
    var args: [String] {
        var a: [String] = []
        if let dir = modelDir {
            a += ["--model-dir", dir.path]
        } else if let name = model {
            a += ["--model", name]
        }
        if let tokDir = tokenizerDir {
            a += ["--tokenizer-dir", tokDir.path]
        }
        if let adapter = adapterPath {
            a += ["--lora", adapter.path]
        }
        a += [
            "--prompt",      prompt,
            "--max-tokens",  String(maxTokens),
            "--temperature", String(temperature),
            "--top-k",       String(topK),
            "--top-p",       String(topP),
            "--repetition-penalty", String(repetitionPenalty),
            "--json",         // streaming gen_token event protocol; both binaries honour this flag
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

// MARK: - DownloadConfig

/// Configuration for `embed --model <name> --download-only --json`.
///
/// Verified contract: the embed binary emits exactly one `@@lattice download_done` event
/// on stdout, then exits 0 (success) or 1 (failure). No `--text` is needed.
struct DownloadConfig {
    var canonicalName: String

    var args: [String] {
        ["--model", canonicalName, "--download-only", "--json"]
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

    /// Download an embedding model via `embed --model <name> --download-only --json`.
    ///
    /// Spawns a dedicated RunHandle (separate from `liveRun`) so downloads do not evict
    /// a running training/eval job. State changes (downloadingModels, downloadErrors) are
    /// always published on the main actor. On success, calls refreshModels() so the new
    /// model appears in the table and the "Get Models" sheet flips the row to INSTALLED.
    @MainActor
    func downloadModel(canonicalName: String) {
        downloadErrors.removeValue(forKey: canonicalName)
        downloadingModels.insert(canonicalName)

        guard let spec = LatticeBridge.launchSpec(.embed, args: DownloadConfig(canonicalName: canonicalName).args) else {
            downloadingModels.remove(canonicalName)
            downloadErrors[canonicalName] = "embed binary not found — run `make build` first"
            return
        }

        let h = RunHandle()
        let hid = ObjectIdentifier(h)
        var sawDone = false

        h.onEvent = { [weak self] ev in
            guard let self = self else { return }
            if case .downloadDone(let d) = ev {
                sawDone = true
                self.downloadingModels.remove(canonicalName)
                if d.ok {
                    self.refreshModels()
                } else {
                    self.downloadErrors[canonicalName] = d.error ?? "download failed (no error message)"
                }
            }
        }
        h.onExit = { [weak self] code in
            guard let self = self else { return }
            self._downloadHandles.removeValue(forKey: hid)
            // Guard against a binary that exits without emitting download_done.
            if !sawDone {
                self.downloadingModels.remove(canonicalName)
                if code != 0 {
                    self.downloadErrors[canonicalName] = "embed exited with code \(code)"
                } else {
                    self.refreshModels()
                }
            }
        }

        do {
            try h.start(spec)
            // Stash after successful start so the handle lives until onExit fires.
            _downloadHandles[hid] = h
        } catch {
            downloadingModels.remove(canonicalName)
            downloadErrors[canonicalName] = "launch failed: \(error.localizedDescription)"
        }
    }

    /// Import a model folder from disk into the model cache.
    ///
    /// Validates that the chosen folder contains `config.json` AND at least one
    /// `.safetensors` file before touching the cache. Copies off the main thread;
    /// publishes state on the main actor. Calls refreshModels() on success.
    @MainActor
    func importModel(from url: URL) {
        let folderName = url.lastPathComponent
        importError = nil
        importingModel = folderName

        Task.detached(priority: .userInitiated) {
            let fm = FileManager.default

            // Validate: must have config.json
            let configURL = url.appendingPathComponent("config.json")
            guard fm.fileExists(atPath: configURL.path) else {
                await MainActor.run {
                    self.importingModel = ""
                    self.importError = "'\(folderName)' has no config.json — not a model directory"
                }
                return
            }

            // Validate: must have at least one .safetensors file
            let children = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
            let hasSafetensors = children.contains { $0.pathExtension == "safetensors" }
            guard hasSafetensors else {
                await MainActor.run {
                    self.importingModel = ""
                    self.importError = "'\(folderName)' has no .safetensors files — not a model directory"
                }
                return
            }

            let dest = LatticeBridge.modelCacheDir.appendingPathComponent(folderName, isDirectory: true)

            // Refuse to overwrite an existing model to prevent silent data loss.
            if fm.fileExists(atPath: dest.path) {
                await MainActor.run {
                    self.importingModel = ""
                    self.importError = "'\(folderName)' already exists in the model cache — remove it first if you want to replace it"
                }
                return
            }

            // Ensure the model cache directory exists.
            do {
                try fm.createDirectory(at: LatticeBridge.modelCacheDir, withIntermediateDirectories: true)
            } catch {
                await MainActor.run {
                    self.importingModel = ""
                    self.importError = "could not create model cache directory: \(error.localizedDescription)"
                }
                return
            }

            // Copy to a hidden staging sibling on the same volume, then atomically rename
            // into place. A crash mid-copy leaves only `.importing-<name>` (skipped by
            // discoverModels via skipsHiddenFiles), never a half-written model at `dest`.
            let staging = LatticeBridge.modelCacheDir.appendingPathComponent(".importing-\(folderName)", isDirectory: true)
            try? fm.removeItem(at: staging)
            do {
                try fm.copyItem(at: url, to: staging)
                try fm.moveItem(at: staging, to: dest)
                await MainActor.run {
                    self.importingModel = ""
                    self.refreshModels()
                }
            } catch {
                try? fm.removeItem(at: staging)
                await MainActor.run {
                    self.importingModel = ""
                    self.importError = "copy failed: \(error.localizedDescription)"
                }
            }
        }
    }

    /// Launch `generate_lora` (CPU BF16) or `chat_metal` (GPU Metal), per `config.useGPU`.
    ///
    /// Both binaries emit identical `@@lattice gen_token` streaming events. The caller
    /// (ChatScreen.send) must embed the honest hardware label ("GPU Metal" / "CPU bf16") in
    /// the turn bubble AT SEND TIME — this method never inspects or fabricates that label.
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
        let binary: LatticeBinary = config.useGPU ? .chatMetal : .generateLora
        return launch(
            binary,
            args: config.args,
            kind: .chat,
            model: modelName,
            totalSteps: nil
        )
    }
}
