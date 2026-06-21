import SwiftUI

// MARK: - Navigation

enum Screen: String, CaseIterable, Identifiable, Hashable {
    case models, train, quantize, chat, data, runs
    var id: String { rawValue }

    var index: String {
        switch self {
        case .models: "01"; case .train: "02"; case .quantize: "03"
        case .chat: "04"; case .data: "05"; case .runs: "06"
        }
    }
    var title: String {
        switch self {
        case .models: "MODELS"; case .train: "TRAIN"; case .quantize: "QUANTIZE"
        case .chat: "CHAT"; case .data: "DATA"; case .runs: "RUNS"
        }
    }
    var shortcut: KeyEquivalent {
        switch self {
        case .models: "1"; case .train: "2"; case .quantize: "3"
        case .chat: "4"; case .data: "5"; case .runs: "r"
        }
    }
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
}

// MARK: - Runs (the lab notebook)

enum RunKind: String, Equatable { case train = "LoRA", quantizeQ4 = "Q4", quantizeQuaRot = "QuaRot", chat = "Chat" }
enum RunStatus: String, Equatable { case idle, running, paused, done, failed }

struct RunRecord: Identifiable, Equatable {
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

    // Generation streaming (chat / generate_lora --json mode)
    var genText: String = ""       // accumulated incremental token deltas from gen_token events
    var genTokS: Double? = nil     // final tokens/sec from the done event (tok_s field)
    var genDone: Bool = false      // set true when the gen_token done event arrives

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
