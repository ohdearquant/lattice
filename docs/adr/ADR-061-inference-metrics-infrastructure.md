# ADR-061: Inference Metrics & Experiment Runner Infrastructure

**Status**: Proposed
**Date**: 2026-05-27
**Crate**: lattice-inference (metrics collection), lattice-tune or new `experiment` module (runner)
**Research**: RQ-3 (`workspaces/20260527/03.md`)
**Depends on**: ADR-059 (ModelSpec), ADR-060 (Structured Pruning)

## Context

Lattice's current observability surface is PPL (strided sliding-window, ADR-044 step 4) and wall-clock throughput (bench scripts via `scripts/bench-compare.sh`). This is sufficient for "config A has PPL X, config B has PPL Y" but insufficient for _architectural insight_: which layers are doing useful work, which heads are redundant, where the memory bandwidth bottleneck is, and whether a given layer is a candidate for replacement, removal, or compression.

Specific shortcomings:

1. **No attention diagnostics.** We cannot determine whether a head's attention is sharply focused (low entropy, pruning or linear-attention candidate) or diffuse (global mixing, must keep). FlashAttention-2 deliberately avoids materializing the T x T attention matrix -- so these metrics require either modified kernels or a separate profiling pass.

2. **No layer importance signal.** Without per-layer activation norms, update ratios, or Block Influence (BI_l = 1 - cos(x_{l-1}, x_l), ShortGPT, Men et al. 2024), we cannot rank layers for pruning (ADR-060) without running a separate forward pass with custom hooks.

3. **No roofline context.** Per-kernel timing without bytes-moved and FLOPs-modeled data cannot distinguish memory-bound ops (where quantization helps) from compute-bound ops (where algorithmic improvements help). On Apple Silicon decode, nearly every op is deeply memory-bound (~1 FLOP/B for FP16 GEMV vs ridge points of 26-42 FLOP/B), but we have no automated classification.

4. **No experiment tracking.** Results live in terminal scrollback. No persistent store of (config, commit, hardware, corpus) -> (metrics). No paired comparison infrastructure beyond `bench-compare.sh` for Criterion benchmarks.

5. **No speculative decoding feedback.** Acceptance rate, the direct proxy for draft quality, is not logged.

These gaps block three planned workstreams: structured pruning (ADR-060), architecture search (ADR-059), and LoRA-heal validation loops.

## Decision

### D1: Three-mode metrics collection (`MetricsMode`)

```rust
pub enum MetricsMode {
    Off,
    CheapOnline,        // PPL, tok/s, layer latency, norms, update ratios, memory model
    AttentionProfile,   // + entropy, sparsity, KV page mass, pattern labels
    HeavyDiagnostics,   // + exact attention capture, randomized SVD/effective rank
}
```

Design principle: **production kernels never pay profiling cost by default.** When `MetricsMode::Off` or no `LayerMetrics` sink is attached, the hot path has zero branches and zero overhead.

### D2: Metrics catalog

Notation: layer `l`, head `h`, query position `t`, key position `s`, attention logits `a_{l,h,t,s}`, probabilities `p_{l,h,t,s} = softmax_s(a_{l,h,t,s})`, context length `T`, hidden size `d`, number of layers `L`.

#### CheapOnline tier (default for all experiments)

Collected at layer boundaries with negligible overhead relative to GEMMs.

| Metric | Mathematical definition | Cost | Architectural insight | Extraction |
|--------|------------------------|------|----------------------|------------|
| **Per-layer latency** | Wall-clock per `LayerOp::forward` | Negligible | Identifies hot layers; latency outliers suggest fusion opportunities | Timer wrap around layer call |
| **Activation norms** | `n_l(t) = \|\|x_l(t)\|\|_2 / sqrt(d)` | `O(L * T * d)` adds | Norm spikes indicate instability or quantization sensitivity | Reduction kernel at layer boundary |
| **Update ratio** | `u_l(t) = \|\|x_l(t) - x_{l-1}(t)\|\|_2 / \|\|x_{l-1}(t)\|\|_2` | Same as norms | Small ratio = near-identity block = pruning candidate. This IS the ShortGPT scoring signal for ADR-060 | Fuse with residual-write kernel |
| **Block Influence** | `BI_l = 1 - cos(x_{l-1}, x_l)` | Same as norms | Complementary to update ratio; cosine similarity vs magnitude change. ShortGPT (Men et al. 2024, arXiv:2403.03853) uses BI for layer pruning ranking. Caveat: cosine similarity alone may poorly predict PPL degradation -- always validate with paired PPL delta | Same reduction kernel |
| **Memory bandwidth model** | `BW_k = bytes_moved_k / time_k`; utilization `rho_k = BW_k / BW_peak` | Negligible | Separates memory-bound from compute-bound ops. FP16 GEMV at ~1 FLOP/B is deeply memory-bound on all M-series (ridge 26-42 FLOP/B) | Modeled byte counts + command-buffer timestamps |
| **Speculative acceptance** | `E[accepted_prefix] / draft_length` | Negligible | Direct proxy for draft quality; informs draft length, model, adaptive speculation | Counter bookkeeping in decode loop |
| **PPL / NLL** | `PPL = exp(-(1/N) sum_i log p(t_i \| t_{<i}))` | Part of evaluation | The ground truth quality metric | Token log-probs from forward pass |
| **Tokens/sec** | Output tokens / wall-clock seconds | Negligible | Throughput target | Timer around generation loop |

#### AttentionProfile tier (opt-in calibration pass)

Requires modified attention kernels or a slower profiling path. Run on a short calibration corpus (128-512 examples), not in production decode.

| Metric | Mathematical definition | Cost | Architectural insight | Extraction |
|--------|------------------------|------|----------------------|------------|
| **Attention entropy** | `H_{l,h,t} = -sum_s p_{l,h,t,s} log(p_{l,h,t,s})`. Normalize as `H / log(T_t)` for cross-length comparability. Aggregate by mean, p10/p50/p90, and token class. | If probabilities materialized: `O(H * T^2)` extra reductions in prefill, `O(H * T)` per decode token. If fused: a few scalar accumulators per row. | Low entropy = sharply focused head (local/sparse attention candidate, pruning candidate). High entropy = diffuse/global mixing (must keep or replace with linear attention). Entropy collapse is also a training-stability signal. | **Fused via Online Softmax Entropy** (see D3). No matrix materialization required. |
| **Attention sparsity** | `S_{l,h,t}(tau) = (1/T_t) sum_s 1[p_{l,h,t,s} < tau]`. Better: top-k mass `M_k = sum_{s in topk} p_s`, Gini coefficient, block sparsity by page/block. | `O(H * T^2)` prefill or `O(H * T)` decode for exact; approximate top-k/block mass cheaper if fused | Complements entropy: two heads can have similar entropy but different locality, sink-token dependence, or block patterns. Useful for sliding attention, page eviction, sparse kernel selection | Second row pass in attention kernel, or approximate top-k/block mass accumulation |
| **KV cache utilization** | Per-token: `U_l(s) = sum_{h,t>s} p_{l,h,t,s}`. Per-page: `U_l(b) = sum_{s in block_b} U_l(s)`. Also track recency curves, sink-token mass, protected-token mass. | Per-page cheap if accumulated inside attention; exact per-token has attention-probability cost | Directly informs KV eviction, compression, page promotion/demotion, sink-token protection, layer-specific cache policies | Accumulate attention mass per physical KV page in PagedKVCache, not per token |
| **Attention pattern classification** | Classify heads by entropy, sink mass, locality, top-k mass, recency decay, vertical/slash/sink patterns | Cheap if entropy/sparsity already instrumented | Decides KV eviction/compression strategy, local attention candidates, sink-token protection | Post-process entropy + sparsity vectors per head |

#### HeavyDiagnostics tier (rare, small windows)

Not compatible with FlashAttention (must materialize P). Run on short sequences (<512 tokens) only.

| Metric | Mathematical definition | Cost | Architectural insight | Extraction |
|--------|------------------------|------|----------------------|------------|
| **Effective rank of attention** | Threshold rank: `r_tau = #{i : sigma_i / sigma_1 > tau}`. Energy rank: `r_eps = min{k : sum_{i<=k} sigma_i^2 / sum_i sigma_i^2 >= 1-eps}`. Entropic rank: `exp(-sum_i q_i log q_i)`, `q_i = sigma_i / sum_j sigma_j`. | Exact SVD: `O(T^3)` per head. Randomized SVD/sketching: `O(T^2 * k)` if P materialized or streamable. | Low effective rank suggests low-rank/linear attention (Linformer-style) may preserve behavior. High rank suggests sparse/local patterns safer. Low-rank and sparse approximations work in different regimes. | Materialize full attention matrix, apply randomized SVD |
| **Activation outlier analysis** | Per-layer/channel: max/mean, p99.9, kurtosis, outlier fraction above `k * sigma`, channel concentration | Cheap reductions | Strong for quantization sensitivity prediction (LLM.int8, SmoothQuant). Choose quantization scheme, mixed precision, smoothing, per-channel scales | Reduction kernels at layer boundaries |

### D3: Online Softmax Entropy -- the fused implementation trick

This is the key insight enabling AttentionProfile metrics without materializing the attention matrix.

Flash-style attention already maintains online softmax accumulators per row: running max `m` and running denominator `l = sum exp(a_s - m)`. To compute entropy, add one additional scalar accumulator:

```
e = sum exp(a_s - m) * a_s    (weighted-logit accumulator)
```

Then at row completion:

```
H = log(l) + m - e / l
```

**Derivation:** By definition, `p_s = exp(a_s - m) / l`, so:
```
H = -sum_s p_s log(p_s)
  = -sum_s (exp(a_s - m) / l) * (a_s - m - log(l))
  = -(1/l) * [sum_s exp(a_s - m) * a_s - m * sum_s exp(a_s - m) - log(l) * sum_s exp(a_s - m)]
  = -(1/l) * [e - m * l - log(l) * l]
  = -e/l + m + log(l)
  = log(l) + m - e/l
```

Cost: three scalar accumulators (m, l, e) per row instead of two. Negligible overhead vs the baseline Flash kernel's memory traffic. Production kernels should NOT pay this cost by default -- gate behind `MetricsMode::AttentionProfile`.

This applies to both the Metal fused attention path (`crates/inference/src/attention/flash_causal.rs`) and the CPU attention path (`crates/inference/src/attention/standard.rs`).

### D4: `LayerMetrics` sink in `ForwardCtx`

```rust
pub struct ForwardCtx<'a> {
    pub layer_idx: usize,
    pub step: StepKind,
    pub shape: AttnShape,
    pub position_ids: &'a [u32],
    pub causal: bool,
    pub rotary: Option<&'a RotaryTables>,
    pub metrics: Option<&'a mut LayerMetrics>,  // None = zero overhead path
}

pub struct LayerMetrics {
    pub mode: MetricsMode,
    // CheapOnline
    pub latency_ns: u64,
    pub input_norm: f32,
    pub output_norm: f32,
    pub update_ratio: f32,
    pub block_influence: f32,
    // AttentionProfile (None when mode < AttentionProfile)
    pub entropy: Option<Vec<f32>>,          // per head
    pub sparsity: Option<Vec<f32>>,         // per head, top-k mass
    pub kv_page_mass: Option<Vec<f32>>,     // per KV page
    pub pattern_label: Option<Vec<String>>, // per head: "sink", "local", "global", "sparse"
}
```

When `metrics` is `None` or `MetricsMode::Off`, the compiler eliminates the branches entirely. The `Option<&mut LayerMetrics>` pattern guarantees zero overhead on the production path -- no vtable dispatch, no branch, no allocation.

### D5: Apple Silicon roofline model

The roofline separates memory-bound from compute-bound ops using operational intensity `I = FLOPs / bytes_moved` and the chip's ridge point `I* = PeakFLOP/s / PeakBW`.

#### Chip table (treat as initial ceilings -- replace with measured local microbenchmarks)

| Chip | Public/derived FP32 GPU peak | Apple-published memory BW | Ridge I* (FLOP/B) |
|------|-----------------------------:|--------------------------:|-------------------:|
| M1 Pro, 16-core GPU | ~5.3 TFLOP/s | 200 GB/s | ~26.5 |
| M1 Max, 32-core GPU | ~10.4 TFLOP/s | 400 GB/s | ~26 |
| M1 Ultra, 64-core GPU | ~21.2 TFLOP/s | 800 GB/s | ~26.5 |
| M2 Pro, 19-core GPU | ~6.79 TFLOP/s | 200 GB/s | ~34 |
| M2 Max, 38-core GPU | ~13.5 TFLOP/s | 400 GB/s | ~34 |
| M2 Ultra, 76-core GPU | ~27.2 TFLOP/s | 800 GB/s | ~34 |
| M3 Pro, 18-core GPU | ~6.35 TFLOP/s | 150 GB/s | ~42 |
| M3 Max, 40-core GPU | ~14.1 TFLOP/s | 400 GB/s | ~35 |
| M3 Ultra, 80-core GPU | ~28.2 TFLOP/s | 819 GB/s | ~34 |
| M4 Pro, 20-core GPU | ~8.0-9.2 TFLOP/s | 273 GB/s | ~29-34 |
| M4 Max, 40-core GPU | ~17.0 TFLOP/s | 546 GB/s | ~31 |

Sources: Apple support pages, Notebookcheck, Low End Mac, cpu-monkey for FP32 derived estimates. FP32 numbers are third-party derived, not from a single Apple "compute roofline" document. For production roofline work, run a local microbenchmark on the target machine, power mode, OS version, and thermal state.

#### Operation classification for Qwen3.5-0.8B decode (batch 1, FP16)

Architecture: hidden=1024, 24 layers, FFN intermediate=3584, 8 Q heads, 2 KV heads, head dim=256.

| Operation | FLOPs/bytes model | Typical I (FLOP/B) | Bound | Optimization lever |
|-----------|-------------------|--------------------:|-------|-------------------|
| Embedding lookup | Read d elements, ~0 arithmetic | ~0 | Memory/random-access | Cache locality, token batching |
| Q/K/V/O GEMV | 2mn FLOPs, mn * weight_bytes | ~1 (FP16) | **Strongly memory-bound** | Quantization, weight packing, fusion, prefetch |
| Attention QK^T | 2 * H_q * T * d_h FLOPs, reads K cache | ~1-4 (GQA reuse) | Memory-bound (long ctx) | KV layout, GQA, KV quantization |
| Softmax | O(H * T) reductions/exp | Low | Latency/reduction | Keep fused with QK/V |
| Attention * V | 2 * H_q * T * d_h FLOPs, reads V cache | ~1-4 (reuse) | Memory-bound | Same as QK |
| FFN gate/up GEMV | ~4 * d * d_ff FLOPs, two matrices | ~1 (FP16) | **Strongly memory-bound** | Quantize FFN weights first; fuse gate/up/SwiGLU |
| FFN down GEMV | 2 * d_ff * d FLOPs, one matrix | ~1 (FP16) | **Strongly memory-bound** | Quantization and layout |
| RMSNorm | Read + reduce + scale + write | <1 | Memory/reduction | Fuse with adjacent matmul |
| Residual add | Read 2 vectors, write 1, 1 add/elem | ~0.17 (FP16) | Memory-bound | Fuse with following norm |
| KV cache write | Write K,V per token per layer | 0 | Pure BW / cache pressure | Compact KV dtype/layout |
| KV cache read | Grows O(T) per decode token | Low | Increasingly memory-bound | Page/block hotness metrics |

Key insight: for decode on Apple GPUs (ridge 26-42 FLOP/B), a FP16 GEMV at ~1 FLOP/B is nowhere near compute-bound. Local LLM decode scales with memory bandwidth. This is why quantization is effective: it reduces bytes moved for weight- and KV-dominated kernels. Prefill is different -- large GEMMs reuse weights across many tokens and can become compute-bound or mixed.

#### Metal timing methodology

```rust
struct KernelTiming {
    op: OpKind,
    layer: Option<u16>,
    dispatch_id: u64,
    gpu_ns: u64,
    modeled_flops: u64,
    modeled_bytes: u64,
    achieved_gflops: f64,
    achieved_gbps: f64,
    bandwidth_utilization: f64,  // achieved_gbps / peak_gbps
}

enum TimingMode {
    None,
    CommandBufferAggregate,  // cheap: gpuEndTime - gpuStartTime per command buffer
    IsolatedDispatch,        // microbenchmark: one kernel per command buffer
    MetalCounters,           // MTLCounterSampleBuffer for per-encoder timing
}
```

Use `MTLCommandBuffer.gpuStartTime` / `gpuEndTime` for aggregate timing (Apple-documented GPU timestamps). For per-dispatch timing, use `MTLCounterSampleBuffer` or Xcode GPU capture -- per-dispatch in separate command buffers introduces scheduling distortion.

### D6: Experiment runner (microkernel architecture)

The runner is a trait-based microkernel that knows nothing about Transformer internals, pruning algorithms, LoRA, or specific metrics. It orchestrates components through narrow interfaces.

```rust
pub struct ExperimentSpec {
    pub experiment_id: String,
    pub base_config: ModelConfig,      // from ADR-059 ModelSpec
    pub weights: WeightSource,
    pub tokenizer: TokenizerSource,
    pub corpus: CorpusSpec,
    pub sweep: SweepSpec,
    pub eval: EvalSpec,
    pub instrumentation: InstrumentationSpec,
    pub recorder: RecorderSpec,
    pub seed: u64,
}

pub trait ConfigMutator: Send + Sync {
    fn expand(&self, base: &ModelConfig) -> anyhow::Result<Vec<ModelConfig>>;
}

pub trait Corpus {
    fn id(&self) -> &str;
    fn hash(&self) -> &str;
    fn examples(&self) -> Box<dyn Iterator<Item = EvalExample> + '_>;
}

pub trait MetricSink {
    fn on_run_start(&mut self, meta: &RunMeta) -> anyhow::Result<()> { Ok(()) }
    fn on_layer(&mut self, event: &LayerEvent) -> anyhow::Result<()> { Ok(()) }
    fn on_token(&mut self, event: &TokenEvent) -> anyhow::Result<()> { Ok(()) }
    fn on_kernel(&mut self, event: &KernelTiming) -> anyhow::Result<()> { Ok(()) }
    fn finish(&mut self) -> anyhow::Result<MetricBundle>;
}

pub trait Evaluator {
    fn evaluate(
        &mut self,
        model: &mut dyn InferenceModel,
        corpus: &dyn Corpus,
        sinks: &mut [&mut dyn MetricSink],
    ) -> anyhow::Result<TrialResult>;
}

pub trait Recorder {
    fn start_run(&mut self, meta: &RunMeta) -> anyhow::Result<RunId>;
    fn log_event(&mut self, run: RunId, event: &MetricEvent) -> anyhow::Result<()>;
    fn finish_run(&mut self, run: RunId, result: &TrialResult) -> anyhow::Result<()>;
}
```

High-level flow:

```rust
fn run_experiment(spec: ExperimentSpec) -> anyhow::Result<()> {
    let configs = spec.sweep.expand(&spec.base_config)?;
    let corpus = load_corpus(&spec.corpus)?;
    let weights = WeightStore::open(&spec.weights)?;

    for (trial_idx, cfg) in configs.into_iter().enumerate() {
        let run_meta = RunMeta::new(&spec, trial_idx, &cfg, &weights, &corpus)?;
        let mut recorder = make_recorder(&spec.recorder)?;
        let run_id = recorder.start_run(&run_meta)?;

        let mut model = load_model(&cfg, &weights)?;
        let mut sinks = make_metric_sinks(&spec.instrumentation, &mut recorder, run_id)?;
        let result = make_evaluator(&spec.eval)?
            .evaluate(&mut model, corpus.as_ref(), &mut sinks)?;

        recorder.finish_run(run_id, &result)?;
    }

    Ok(())
}
```

**Parallelism**: process-level isolation for GPU/Metal trials (one active heavy Metal trial per physical GPU). Thread-level (`rayon`) for CPU-only sweeps and corpus preprocessing. Memory-map weights and reuse OS page cache across processes.

### D7: Storage schema

#### SQLite (canonical structured store)

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    parent_run_id TEXT,
    status TEXT NOT NULL,
    timestamp_start TEXT NOT NULL,
    timestamp_end TEXT,
    git_commit TEXT NOT NULL,
    git_dirty INTEGER NOT NULL,
    binary_hash TEXT,
    model_id TEXT NOT NULL,
    weights_hash TEXT NOT NULL,
    tokenizer_hash TEXT NOT NULL,
    base_config_hash TEXT NOT NULL,
    trial_config_hash TEXT NOT NULL,
    corpus_id TEXT NOT NULL,
    corpus_hash TEXT NOT NULL,
    tokenized_corpus_hash TEXT,
    seed INTEGER NOT NULL,
    hardware_json TEXT NOT NULL,
    model_config_json TEXT NOT NULL,
    eval_config_json TEXT NOT NULL,
    instrumentation_json TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE metric_summaries (
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    aggregation TEXT,
    sample_count INTEGER,
    PRIMARY KEY (run_id, metric_name, aggregation),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE layer_metrics (
    run_id TEXT NOT NULL,
    layer INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    aggregation TEXT,
    sample_count INTEGER,
    PRIMARY KEY (run_id, layer, metric_name, aggregation)
);

CREATE TABLE head_metrics (
    run_id TEXT NOT NULL,
    layer INTEGER NOT NULL,
    head INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    aggregation TEXT,
    sample_count INTEGER,
    PRIMARY KEY (run_id, layer, head, metric_name, aggregation)
);

CREATE TABLE kernel_metrics (
    run_id TEXT NOT NULL,
    op TEXT NOT NULL,
    layer INTEGER,
    calls INTEGER NOT NULL,
    gpu_ns_total INTEGER NOT NULL,
    modeled_flops_total INTEGER,
    modeled_bytes_total INTEGER,
    achieved_gbps REAL,
    achieved_gflops REAL,
    bandwidth_utilization REAL,
    PRIMARY KEY (run_id, op, layer)
);

CREATE TABLE artifacts (
    run_id TEXT NOT NULL,
    artifact_name TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    media_type TEXT,
    PRIMARY KEY (run_id, artifact_name)
);
```

#### JSON-lines (append-only raw log)

Per-trial event streams stored as `runs/<run_id>/events.jsonl.zst` (zstd-compressed). Not queried directly -- use SQLite for structured access. Event examples:

```json
{"type":"layer","run_id":"...","example":17,"token":128,"layer":6,"norm":31.2,"update_ratio":0.014}
{"type":"head","run_id":"...","example":17,"token":128,"layer":6,"head":3,"entropy":0.42,"sink_mass":0.31}
{"type":"kernel","run_id":"...","op":"ffn_up","layer":6,"gpu_ns":82100,"bytes":7340032,"flops":7340032}
```

#### khive KG (distilled findings and provenance)

Store interpreted results, not telemetry:
- "For Qwen3.5-0.8B on M4 Max, FFN down GEMV is memory-bound at ~X GB/s"
- "Layer 17 removal increased PPL by +0.04 after 20 LoRA-heal steps"

Link experiment entity -> model entity -> finding entity with provenance edges.

### D8: Reproducibility contract

Every experiment run automatically records:

- Git commit, dirty flag, diff hash, binary hash, Cargo feature set
- Rust version, target triple
- Model config JSON and config hash, weight file hashes, tokenizer hash, quantization metadata
- Exact corpus text hash and tokenized corpus hash, prompt order, RNG seed, RNG algorithm
- Sampling parameters (temperature, top-p, seed)
- Hardware: Metal device name, registry ID, memory size, OS build
- Power mode, thermal state before/after
- Warmup and measurement iterations
- Batch size, context length, KV cache policy, page size
- Timing method (command-buffer aggregate, isolated dispatch, Metal counters)
- Number of repeats, confidence intervals

Comparison reports use paired evaluation: same examples, tokenization, prompt order, seeds per config. Report mean/median NLL, PPL and PPL delta vs baseline, tok/s and latency p50/p95, per-layer latency deltas, bootstrap CIs, paired permutation test on per-example NLL deltas, Holm correction for multi-config comparisons.

### D9: How metrics feed pruning (ADR-060 synergy)

**CheapOnline mode IS the pruning scoring signal.** No separate calibration pass needed.

| Metric | Pruning signal | ADR-060 use |
|--------|---------------|-------------|
| Update ratio / Block Influence | Rank layers by BI ascending -> lowest BI = most removable | `ShortGPTScorer` layer ranking |
| Activation norms | Detect norm spikes / quantization-sensitive layers | Mixed-precision decisions |
| Activation outlier analysis | Identify channels with extreme kurtosis | Per-channel quantization scaling (QuaRot, AWQ) |
| Attention entropy | Low entropy heads -> head pruning candidates (remove entire heads, not just layers) | Future head-pruning extension |
| Attention pattern classification | Sink-heavy heads: must keep sink tokens. Local-only heads: candidates for sliding-window replacement | KV cache policy per head |

The experiment runner's paired-PPL evaluation provides the mandatory validation gate: no irreversible pruning decision is made on proxy metrics alone. Pipeline:

1. CheapOnline forward -> rank candidates
2. Apply top-k candidates
3. Paired PPL eval -> accept/reject
4. Optional LoRA-heal (10-50 steps, rank-8, SGD) -> re-evaluate
5. Larger validation corpus for final gate

### D10: How metrics feed architecture search (ADR-059 synergy)

The gradient-free architecture search pipeline:

1. Run baseline CheapOnline + AttentionProfile on calibration corpus
2. Generate candidates:
   - Low update ratio -> layer prune/merge candidate
   - High activation outliers -> keep higher precision or apply smoothing
   - High entropy + broad support -> linear/global attention candidate (replace softmax with GatedDeltaNet or similar)
   - Local sparse pattern -> sliding/local attention candidate
   - Low KV page mass -> evict/compress page candidate
3. Run paired PPL eval for each candidate
4. Pareto frontier on: delta-NLL, latency, memory, bandwidth utilization
5. Optional LoRA-heal
6. Run larger validation corpus for top candidates

Most predictive and cheapest in practice (ranked):
1. **Small-corpus paired PPL delta**: strongest signal, costs a full eval
2. **Update ratio / Block Influence**: cheapest pruning heuristic
3. **Activation outliers**: best quantization predictor
4. **Attention pattern stats**: best for KV/attention-kernel choices
5. **Effective rank**: useful but heavy (HeavyDiagnostics)

Entropy + effective rank + update ratio across layer types enable automated layer-type selection: a layer with low entropy AND low effective rank is a strong candidate for linear attention; a layer with high BI is doing important work and should not be pruned.

## Implementation roadmap

### Phase 1: Cheap metrics and schema
- `MetricSink` trait
- SQLite recorder with schema above
- JSONL event writer
- PPL, tok/s, per-layer latency, residual norm, update ratio, Block Influence
- Modeled FLOPs/bytes per op
- Command-buffer aggregate timing
- **Immediately enables architecture sweeps**

### Phase 2: Attention profiling
- Instrumented attention path behind `--metrics-mode attention-profile`
- Online Softmax Entropy implementation in Flash causal and standard attention
- Top-k mass / sparsity thresholds
- Sink-token mass
- Per-page KV attention mass
- Pattern classification post-processor

### Phase 3: Roofline reports
- Per-op/layer output: modeled_flops, modeled_bytes, gpu_time_ns, achieved_gflops, achieved_gbps, %peak_bw, %peak_compute, roofline_bound
- Markdown table and CSV export
- Runtime chip detection via `MTLDevice` properties

### Phase 4: Minimal LoRA-heal (ADR-060 dependency)
- Start CPU-only: rank-r LoRA params, forward injection, manual backward for short sequence
- SGD optimizer, finite-difference gradient tests
- "modify -> 10 LoRA steps -> measure PPL" loop
- Move hot backward kernels to Metal

### Phase 5: Agent loop (ADR-059 dependency)
- Stable CLI contract:
  ```bash
  lattice-exp run --config model.toml --sweep sweep.toml \
      --corpus calib.jsonl --metrics attention-profile \
      --out lattice_experiments.sqlite
  ```
- Machine-readable summary output (JSON) with: run_id, config_hash, PPL, tok/s, memory_mb, best_layer_prune_candidates, memory_bound_ops, attention_findings
- This is the interface an RL agent or human search process needs

## Alternatives considered

1. **Always-on full metrics**: Rejected. Attention entropy in the hot path adds ~5-10% overhead. Three-mode design lets experiments opt in.
2. **External profiling only (Instruments, GPU counters)**: Rejected as primary path. External tools cannot correlate with model-level concepts (which layer, which head). But Metal `MTLCounterSampleBuffer` and Xcode GPU profiling are valuable supplementary signals.
3. **Criterion-only benchmarking**: Insufficient. Criterion measures kernel throughput but not architectural insight. Criterion remains valuable for micro-benchmarks (ADR-058); this ADR adds model-level insight on top.
4. **W&B / MLflow integration**: Deferred. The Rust-native experiment runner with SQLite/JSONL is sufficient for the single-developer workflow. W&B/MLflow integration can layer on top via the Recorder trait if needed.

## Consequences

- **Pruning synergy (ADR-060)**: Update ratio / Block Influence from CheapOnline mode feeds directly into ShortGPT scorer -- no separate calibration pass needed for layer importance.
- **Architecture search (ADR-059)**: Attention entropy + effective rank + pattern classification identify candidates for attention mechanism swapping.
- **Quantization guidance**: Activation outlier analysis identifies quantization-sensitive layers, informing QuaRot (ADR-044) mixed-precision decisions.
- **Experiment reproducibility**: "(config, commit, hardware, corpus) -> metrics" persistently recorded with full provenance.
- **Risk: MetricsMode complexity**: Mitigated by `Option<&mut LayerMetrics>` -- None path has zero overhead. The three-mode design explicitly bounds what each tier costs.
- **Risk: Storage growth**: Raw JSONL at per-token granularity can grow fast. Mitigated by zstd compression and storing raw events only per trial, not in SQLite.
- **Existing infrastructure preserved**: `bench-compare.sh` and Criterion benchmarks (ADR-058) continue to work for kernel-level A/B testing. This ADR adds the model-level instrumentation layer above that.

## References

- FlashAttention-2: Dao 2023 (arXiv:2205.14135) -- IO-aware exact attention with tiling
- ShortGPT: Men et al. 2024 (arXiv:2403.03853) -- Block Influence for layer pruning
- PagedAttention: Kwon et al. SOSP 2023 (arXiv:2309.06180) -- Paged KV cache
- Speculative Decoding: Leviathan et al. ICML 2023 (arXiv:2211.17192) -- Draft/verify with distribution preservation
- Roofline Model: Williams, Waterman, Patterson. Communications of the ACM 2009
- LLM Inference Roofline: Yuan et al. 2024 (arXiv:2402.16363) -- Survey and roofline insights
- Linformer: Wang et al. 2020 (arXiv:2006.04768) -- Low-rank attention approximation
- StreamingLLM: Xiao et al. 2023 (arXiv:2309.17453) -- Attention sink tokens
- LoRA: Hu et al. ICLR 2022 (arXiv:2106.09685) -- Low-rank adaptation
- LLM.int8: Dettmers et al. NeurIPS 2022 (arXiv:2208.07339) -- Activation outliers in quantization
- Apple Metal docs: MTLCommandBuffer.gpuStartTime, GPU counters, Metal tools
