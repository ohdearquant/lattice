import SwiftUI

// MARK: - Navigation

/// The verb tabs in the main column. The app is model-centric: a model is selected in the
/// sidebar, then acted on through one of these verbs. Chat is home.
enum Screen: String, CaseIterable, Identifiable, Hashable {
    case chat, serve, quantize, train, inspect
    var id: String { rawValue }

    var index: String {
        switch self {
        case .chat: "01"; case .serve: "02"; case .quantize: "03"; case .train: "04"; case .inspect: "05"
        }
    }
    var title: String {
        switch self {
        case .chat: "Chat"; case .serve: "Serve"; case .quantize: "Quantize"
        case .train: "Train"; case .inspect: "Inspect"
        }
    }
    var symbol: String {
        switch self {
        case .chat:     "text.bubble"
        case .serve:    "dot.radiowaves.left.and.right"
        case .quantize: "rectangle.compress.vertical"
        case .train:    "dial.high"
        case .inspect:  "cube.transparent"
        }
    }
    var shortcut: KeyEquivalent {
        switch self {
        case .chat: "1"; case .serve: "2"; case .quantize: "3"; case .train: "4"; case .inspect: "5"
        }
    }

    /// Only Chat docks a settings inspector (target · backend · sampling). The other verbs
    /// carry their controls inline.
    var hasInspector: Bool { self == .chat }
}

// MARK: - Models on disk

enum ModelFormat: String, Equatable {
    case bf16 = "BF16"
    case q4 = "Q4"
    case quarot = "QuaRot Q4"
    case embedding = "Embed"
    case unknown = "—"

    var badge: String { self == .quarot ? "rotated" : rawValue }
    var isQuantized: Bool { self == .q4 || self == .quarot }
}

struct ModelInfo: Identifiable, Equatable {
    var id: String { path.path }
    var name: String
    var path: URL
    var format: ModelFormat
    var params: String?         // e.g. "0.8B" parsed from name/config
    var dtype: String           // BF16 / Q4_0 / mixed
    var sizeBytes: Int64
    var fileCount: Int
    var hasTokenizer: Bool
    var layerSummary: String?   // e.g. "18 GDN · 6 GQA"
    var hidden: Int?
    var vocab: Int?
    var contextLength: Int?     // max_position_embeddings; nil when config.json absent
    var attnHeads: Int?         // num_attention_heads (GQA query heads)
    var kvHeads: Int?           // num_key_value_heads (GQA KV heads)
    var headDim: Int?           // head_dim (explicit; NOT hidden/heads for qwen3.5)
    var gdnKeyHeads: Int?       // linear_num_key_heads (GatedDeltaNet)
    var gdnValueHeads: Int?     // linear_num_value_heads (GatedDeltaNet)
    var intermediateSize: Int?  // intermediate_size — FFN/MLP inner width (3584 for qwen3.5)
    var isEmbedding: Bool = false
    var adapters: [AdapterInfo] = []
}

struct AdapterInfo: Identifiable, Equatable {
    var id: String { path.path }
    var name: String
    var path: URL
    var rank: Int?
    var alpha: Double?
    var targetModules: String?
    var sizeBytes: Int64
    // MLX-format fields (nil when absent or config unreadable — honest)
    var baseModel: String? = nil
    var scale: Double? = nil
    var numLayers: Int? = nil
    var checkpointCount: Int = 0

    /// The `.safetensors` weight file to hand to `generate_lora --lora`, or `nil` when the
    /// package has no usable weight file.
    ///
    /// Resolved ONCE at discovery (`LatticeBridge.discoverAdapterPackages` /
    /// `discoverAdapters`) via `resolveWeightFile`, NOT recomputed per access — the Chat
    /// picker filters non-runnable adapters (`weightFile == nil`) on every re-render, so this
    /// must be a cheap stored read, never per-frame file I/O. `nil` is honest: the picker
    /// hides the adapter and callers omit `--lora` rather than pass a path the engine cannot
    /// read (it would otherwise silently run the base model under the adapter's label).
    var weightFile: URL? = nil

    /// Resolve the primary `.safetensors` weight file for an adapter package.
    ///
    /// `path` from `discoverAdapterPackages` is the package DIRECTORY, but the engine's
    /// loader (`load_peft_safetensors`) reads a single file via `std::fs::read` and errors on
    /// a directory. Resolve the file inside, by priority:
    ///   1. `adapters.safetensors`      — MLX/lattice final consolidated adapter
    ///   2. `adapter.safetensors`       — singular variant (lattice trainer output)
    ///   3. `adapter_model.safetensors` — PEFT
    ///   4. highest-numbered `*_adapters.safetensors` — MLX checkpoint (training stopped
    ///      before consolidation; pick the latest step)
    /// Returns `path` unchanged when it already points at a file (loose-file adapters), or
    /// `nil` when no weight file exists.
    static func resolveWeightFile(at path: URL) -> URL? {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: path.path, isDirectory: &isDir) else { return nil }
        if !isDir.boolValue { return path }

        func candidate(_ name: String) -> URL? {
            let u = path.appendingPathComponent(name)
            return fm.fileExists(atPath: u.path) ? u : nil
        }
        if let f = candidate("adapters.safetensors") { return f }
        if let f = candidate("adapter.safetensors") { return f }
        if let f = candidate("adapter_model.safetensors") { return f }

        guard let children = try? fm.contentsOfDirectory(
            at: path, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return nil }
        // The "<step>_adapters.safetensors" prefix is an integer step count, so pick the
        // highest step NUMERICALLY (10 > 9), not lexicographically (where "9" > "10" as
        // strings would load an older checkpoint), with the filename as a stable tie-breaker.
        func step(_ url: URL) -> Int {
            Int(url.lastPathComponent.prefix { $0.isNumber }) ?? -1
        }
        return children
            .filter { $0.lastPathComponent.hasSuffix("_adapters.safetensors") }
            .max { a, b in
                let sa = step(a), sb = step(b)
                return sa != sb ? sa < sb : a.lastPathComponent < b.lastPathComponent
            }
    }
}

// MARK: - Curated model catalog (Get Models sheet)

// UI treatment axis: .downloadable → one-click Download (engine has a checksum-verified
// downloader); .importOnly → "Copy HF URL" + manual Import (no in-app downloader). The
// Qwen embedding model is import-only too — its loader has no fetch path, only a local-dir
// lookup — so it belongs with the generative models, not in the Download section.
enum CuratedKind { case downloadable, importOnly }

struct CuratedModel: Identifiable {
    let id: String          // canonical name used by the embed binary
    let name: String
    let kind: CuratedKind
    let detail: String      // e.g. "384 dim · ~130 MB" or one-line description
    let approxSize: String? // downloadable models only
    let hfURL: String?      // import-only models — HuggingFace source link
}

let curatedCatalog: [CuratedModel] = [
    // Downloadable embeddings — `embed --model <id> --download-only --json` (checksum-verified)
    CuratedModel(id: "bge-small-en-v1.5",    name: "bge-small-en-v1.5",    kind: .downloadable,   detail: "384 dim · English",       approxSize: "~130 MB", hfURL: nil),
    CuratedModel(id: "bge-base-en-v1.5",     name: "bge-base-en-v1.5",     kind: .downloadable,   detail: "768 dim · English",       approxSize: "~440 MB", hfURL: nil),
    CuratedModel(id: "bge-large-en-v1.5",    name: "bge-large-en-v1.5",    kind: .downloadable,   detail: "1024 dim · English",      approxSize: "~1.3 GB", hfURL: nil),
    CuratedModel(id: "multilingual-e5-small", name: "multilingual-e5-small", kind: .downloadable,  detail: "384 dim · Multilingual",  approxSize: "~470 MB", hfURL: nil),
    CuratedModel(id: "multilingual-e5-base",  name: "multilingual-e5-base",  kind: .downloadable,  detail: "768 dim · Multilingual",  approxSize: "~1.1 GB", hfURL: nil),
    CuratedModel(id: "all-minilm-l6-v2",      name: "all-minilm-l6-v2",      kind: .downloadable,  detail: "384 dim · English",       approxSize: "~90 MB",  hfURL: nil),
    CuratedModel(id: "paraphrase-multilingual-minilm-l12-v2", name: "paraphrase-multilingual-minilm-l12-v2", kind: .downloadable,
                 detail: "384 dim · Multilingual", approxSize: "~470 MB", hfURL: nil),

    // Import-only — no in-app downloader; fetch from HuggingFace, then use Import from Disk
    CuratedModel(id: "qwen3.5-0.8b", name: "qwen3.5-0.8b", kind: .importOnly,
                 detail: "0.8B generative model — lightweight, fast",
                 approxSize: nil, hfURL: "https://huggingface.co/Qwen/Qwen3.5-0.8B"),
    CuratedModel(id: "qwen3.5-2b",   name: "qwen3.5-2b",   kind: .importOnly,
                 detail: "2B generative model — better quality",
                 approxSize: nil, hfURL: "https://huggingface.co/Qwen/Qwen3.5-2B"),
    CuratedModel(id: "qwen3-embedding-0.6b", name: "qwen3-embedding-0.6b", kind: .importOnly,
                 detail: "0.6B embedding model — import-only (no in-app download)",
                 approxSize: nil, hfURL: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"),
]

// MARK: - Chat domain models (owned by AppStore; moved here from ChatScreen so the store can persist them)

/// A single exchange in single-mode chat.
struct ChatTurn: Identifiable {
    enum TurnStatus { case running, done, failed }

    let id = UUID()
    let prompt: String
    /// The model that produced this turn, snapshotted at send time so the assistant label stays
    /// correct for historical turns even after the working model is switched. nil = unknown.
    var modelName: String? = nil
    var responseText: String = ""
    /// Reasoning trace parsed from the <think>…</think> block in the stream.
    /// Empty when the turn has no reasoning (thinking disabled, or model emitted none).
    var thinkingText: String = ""
    var status: TurnStatus = .running
    var tokensPerSecond: Double? = nil
    /// Per-turn token accounting, captured from the engine done event at resolution.
    /// `cachedInputTokens` is always 0 today: the serve loop resets the KV cache every turn,
    /// so there is no cross-turn prefix cache. Surfaced honestly rather than hidden.
    var promptTokens: Int? = nil
    var cachedInputTokens: Int = 0
    var reasoningTokens: Int? = nil
    var responseTokens: Int? = nil
    var totalMs: Double? = nil
    /// Error message from the run log — set when the turn finishes with an empty response.
    /// Shown instead of "(no output)" so Ocean knows WHY the engine produced nothing.
    var errorMessage: String? = nil
    /// The GenConfig used to produce this turn — stored so Retry can re-launch the identical run.
    var retryConfig: ChatGenConfig? = nil
    /// Honest hardware label, snapshotted at send time: "GPU Metal bf16", "CPU bf16", etc.
    /// Never updated after the run starts — what launched is what ran.
    var inferenceLabel: String? = nil
    /// Whether this turn's prompt PREFILLED an open `<think>` block (the enable_thinking=true
    /// path in renderChatML). When true the model's stream carries no opening tag — it begins
    /// mid-reasoning and only emits `</think>` before the answer — so a completed stream with
    /// no `</think>` boundary is unfinished reasoning, not the answer. Finalization keeps it as
    /// thinkingText and leaves responseText empty so the unfinished turn is skipped from history.
    /// Snapshotted at send time (Retry mutates the turn in place and keeps this flag).
    var prefilledOpenThink: Bool = false
}

/// The generation parameters needed to retry a failed/empty turn.
/// Mirrors fields of GenConfig but uses serialisation-friendly plain types so it can live on ChatTurn.
struct ChatGenConfig: Equatable {
    var modelDirPath: String?
    var model: String?
    var tokenizerDirPath: String?
    var adapterFilePath: String?
    var prompt: String
    var maxTokens: Int
    var seed: UInt64?
    var temperature: Double
    var topK: Int = 50
    var topP: Double = 0.9
    var repetitionPenalty: Double = 1.1
    /// Whether this config targets the GPU Metal path (chat_metal) or CPU (generate_lora).
    var useGPU: Bool = false
}

/// One A/B experiment pair (base column vs adapter column).
struct ComparePair: Identifiable {
    enum Side { case base, adapter }

    let id = UUID()
    let prompt: String           // raw user text (for display)
    let baseLabel: String        // base model name, snapshotted at creation
    let adapterLabel: String     // adapter name, snapshotted at creation
    var baseText: String = ""
    var baseTokS: Double? = nil
    var baseDone: Bool = false
    var adapterText: String = ""
    var adapterTokS: Double? = nil
    var adapterDone: Bool = false
    var failed: Bool = false
    var failureReason: String? = nil  // surface engine error when failed
}

/// Tracks which phase of an A/B run sequence is active.
enum ABPhase { case idle, base, adapter }

// MARK: - Eval compare domain models (owned by AppStore via evalComparePairs)

/// One column in an N-way compare pair — holds the result for a single model/adapter slot.
struct EvalColumn {
    /// Display label snapshotted at pair creation: "model-name · adapter-name · device-tag".
    /// Survives picker changes — labels never relabel a completed result.
    let label: String
    /// Accumulated token stream from gen_token events.
    var text: String = ""
    /// Final tok/s from the generation done event.
    var tokS: Double? = nil
    /// True once this column's run has completed (done or failed).
    var done: Bool = false
    /// True if the run failed AND produced no output (honest failure, not partial output).
    var failed: Bool = false
    /// Engine error message from the run log when failed == true. Honest-nil when unknown.
    var failureReason: String? = nil
}

/// One N-way compare experiment — shared prompt, one EvalColumn per model/adapter slot.
struct EvalComparePair: Identifiable {
    let id = UUID()
    /// Raw user prompt text (shown as the cell header in each column).
    let prompt: String
    /// One entry per slot at the time the pair was submitted. Immutable count; column content mutates.
    var columns: [EvalColumn]
}

// MARK: - PPL measurement cache

/// A single measured perplexity snapshot, keyed into AppStore.measuredPPL by modelID.
struct MeasuredPPL: Equatable, Codable {
    var modelID: String
    var bf16: Double?
    var quant: Double?
    var quantLabel: String?    // "Q4" or "QuaRot"
}

// MARK: - Runs (the lab notebook)

enum RunKind: String, Equatable, Codable {
    case train = "LoRA"
    case quantizeQ4 = "Q4"
    case quantizeQuaRot = "QuaRot"
    case chat = "Chat"
    case eval = "Eval"   // eval_perplexity — measures PPL for one or more model variants
    case embed = "Embed" // embed — produces embedding vectors and a pairwise cosine matrix
}
enum RunStatus: String, Equatable, Codable { case idle, running, paused, done, failed }

struct RunRecord: Identifiable, Equatable, Codable {
    var id: String
    var kind: RunKind
    var model: String
    var status: RunStatus
    var startedAt: Date
    var lastLoss: Double?
    var bestVal: Double?
    var durationS: Double?
    var configSummary: String?
    var adapterPath: String?
}

// MARK: - Live training series (drives the oscilloscope strip chart + scrub-to-freeze)

struct TrainPoint: Identifiable, Equatable {
    var id: Int { step }
    var step: Int
    var loss: Double
    var valLoss: Double?
    var gradNorm: Double?
    var lr: Double?
    var tokS: Double?
}

@Observable
final class LiveRun {
    var kind: RunKind
    var modelName: String
    var status: RunStatus = .running
    var startedAt: Date = Date()
    var totalSteps: Int?

    // Train
    var points: [TrainPoint] = []
    var bestVal: Double?
    var baseNLL: Double?
    var savedAdapterPath: String?

    // Quantize
    var quantBeforeMB: Double?
    var quantAfterMB: Double?
    var quantRatio: Double?
    var quantLayerIndex: Int = 0
    var quantLayerCount: Int = 0
    var verdict: String?
    var quantMaxAbs: Double?     // QuaRot forward-equivalence max abs error
    var quantScheme: String?     // dominant quantization scheme seen in quant_layer events

    // Generation streaming (chat / generate_lora --json mode)
    var genText: String = ""       // accumulated incremental token deltas from gen_token events
    var genTokS: Double? = nil     // final tokens/sec from the done event (tok_s field)
    var genDone: Bool = false      // set true when the gen_token done event arrives
    var genPromptTokens: Int? = nil   // input tokens, from the done event
    var genTokensTotal: Int? = nil    // total generated tokens, from the done event
    var genTotalMs: Double? = nil     // total prefill+decode wall time (ms), from the done event
    // Reasoning/response split, counted from gen_token events around the </think> boundary in
    // the generated text. sawThinkClose flips once </think> has streamed.
    var genReasoningTokens: Int = 0
    var sawThinkClose: Bool = false

    // Eval (eval_perplexity --json mode)
    // A single run may emit multiple rows — one per measurement label (bf16/q4/quarot/adapter).
    var perplexities: [LatticeEvent.Perplexity] = []

    // Embed (embed --json mode)
    // Exactly one embed_done event is expected per run (batch mode: all texts in one call).
    var embed: LatticeEvent.EmbedDone? = nil

    // Completion hook — fired by AppStore.finish() after status and RunRecord are set.
    // Used by screens to chain a follow-up run (e.g. base eval → adapter eval A/B sequence).
    // Closures cannot conform to Codable; never serialised to disk.
    var onComplete: ((LiveRun) -> Void)? = nil

    // Failure reason — the last meaningful error line from the subprocess (non-empty, non-banner).
    // Set by AppStore.finish() when the exit code is non-zero and the log contains an error.
    // Honest-nil when the process exited non-zero but produced no parseable error line.
    var failureReason: String? = nil

    // Shared
    var log: [String] = []

    var currentStep: Int { points.last?.step ?? 0 }
    var currentLoss: Double? { points.last?.loss }

    init(kind: RunKind, modelName: String) {
        self.kind = kind
        self.modelName = modelName
    }

    func appendLog(_ line: String) {
        log.append(line)
        if log.count > 4000 { log.removeFirst(log.count - 4000) }
    }
}
