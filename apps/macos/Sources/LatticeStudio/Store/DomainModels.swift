import SwiftUI

// MARK: - Navigation

enum Screen: String, CaseIterable, Identifiable, Hashable {
    case models, data, train, chat, embed
    var id: String { rawValue }

    var index: String {
        switch self {
        case .models: "01"; case .data: "02"; case .train: "03"; case .chat: "04"; case .embed: "05"
        }
    }
    var title: String {
        switch self {
        case .models: "MODELS"; case .data: "DATA"; case .train: "TRAIN"; case .chat: "CHAT"; case .embed: "EMBEDDINGS"
        }
    }
    var shortcut: KeyEquivalent {
        switch self {
        case .models: "1"; case .data: "2"; case .train: "3"; case .chat: "4"; case .embed: "5"
        }
    }

    /// Whether this screen has a right inspector (toggle sidebar). Drives the
    /// shared window-toolbar toggle visibility. The embed screen is a self-contained
    /// HSplitView tool and does not use the system inspector panel.
    var hasInspector: Bool { self != .embed }
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
