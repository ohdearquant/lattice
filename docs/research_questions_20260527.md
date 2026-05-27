# Deep Research Questions — 2026-05-27

5 self-contained research prompts for ChatGPT Deep Research. Each is designed to be fed as a single session. They cover the complete frontier for making lattice a seed-round-ready composable architecture exploration platform.

---

## RQ-1: Composable Layer DSL and Architecture Search for Transformer Inference Engines

### Context

**lattice** is a pure-Rust transformer inference engine (57.5K LOC, 5 crates) that currently implements 10 attention mechanisms as separate modules: standard MHA, GQA, Flash (CPU-tiled), Flash-Causal, GDN (GatedDeltaNet — linear attention), GDN-Fused, Gated Attention (G1), Differential Attention, Native Sparse Attention (NSA), and a decode-specific path. The Qwen3.5 hybrid architecture uses a `LayerType` enum (`FullAttention` | `LinearAttention`) with a `Vec<LayerType>` to dispatch per-layer — e.g., `[GDN, GDN, GDN, GQA] × 6` for the 24-layer 0.8B model.

However, each attention module has a different Rust API (different struct signatures, different `forward` function parameters). There is **no common trait** — you can't swap GDN for NSA at layer 12 without editing the forward pass source code. Similarly, quantization (f32/f16/int8/int4/QuaRot) is hardcoded per-model, not composable per-layer.

The vision is to make lattice a **research platform for automated architecture exploration** — where an agent (human or RL) can describe a model configuration declaratively, run it, measure it, and iterate. This requires:

1. A common attention trait hierarchy
2. A layer composition DSL (declarative config → model)
3. Dynamic dispatch with minimal overhead
4. Composable quantization per-layer
5. Integration with pruning/sparsity tools (see RQ-2)

### Questions to research deeply

1. **Trait hierarchy design for heterogeneous attention**: GDN has recurrent state (`GatedDeltaNetState` — a matrix that persists across tokens). NSA has three parallel branches (compression, selection, sliding window). Standard MHA is stateless. Flash attention tiles differently than standard. What trait hierarchy accommodates all of these? Options: (a) single `trait Attention` with associated types for state/config/scratch, (b) separate `trait SoftmaxAttention` / `trait LinearAttention` / `trait SparseAttention` with a unifying `enum AttentionKind`, (c) a `trait Layer` at the transformer-block level rather than attention level. **For each option**: show the Rust trait signature, show how GDN and GQA would both implement it, and analyze the performance cost of dynamic dispatch (`dyn Trait` vs `enum_dispatch` vs compile-time generics).

2. **Layer composition DSL**: How should a model configuration be specified declaratively? Consider: YAML/JSON config → Rust model, or a Rust builder pattern, or a macro-based DSL. The config needs to express: (a) number of layers, (b) attention type per layer (from the 10 variants), (c) quantization level per layer (f32/f16/int8/int4), (d) FFN variant per layer (SwiGLU standard, MoE with top-k routing), (e) KV cache type per layer (flat/paged, f32/f16/int8). How do existing systems handle this? Survey: HuggingFace transformers `config.json` + `AutoModel`, llama.cpp's model-agnostic GGUF loader, Burn framework's module system, Candle's model definitions. What's the state of the art for declarative model composition in compiled languages (Rust, C++)?

3. **Zero-cost or near-zero-cost dispatch**: In an inference hot loop processing thousands of tokens, can per-layer dispatch be zero-cost? Techniques to evaluate: (a) `enum_dispatch` crate (generates match arms at compile time), (b) monomorphization via generics (compile N variants), (c) function pointers (`fn` not `dyn`), (d) arena-based vtable (all attention impls in a contiguous allocation). What's the measured overhead of `dyn Trait` dispatch on Apple Silicon M-series for a function called ~24 times per token (once per layer)?

4. **Architecture search without training**: Given a pre-trained model (e.g., Qwen3.5-0.8B), how can we explore architecture modifications without retraining from scratch? Methods: (a) swap a layer's attention type, transfer weights where dimensions match, measure PPL delta, (b) use LoRA adapters at modified layers to "heal" the mismatch with minimal training, (c) use gradient-free signals (attention entropy, activation norms) as quality proxies. Are there precedents for NAS-like exploration in inference-optimized engines (not training frameworks)? What does the search space look like for a 24-layer model with 10 attention types × 4 quantization levels = 40 options per layer?

5. **Composable quantization**: Similar to attention, quantization should be swappable per-layer or per-tensor. Design: `trait Quantizer { fn quantize(&self, tensor: &[f32]) -> QuantizedTensor; fn dequantize(&self, qt: &QuantizedTensor) -> Vec<f32>; }` with impls for f16, int8-symmetric, int8-asymmetric, int4-grouped, QuaRot-Q4. How do vLLM and TensorRT-LLM handle mixed-precision inference? How does this compose with the attention trait (a Flash attention kernel needs to know the KV precision for its inner loop)?

6. **Agent-driven architecture exploration**: For RL-based or LLM-agent-based architecture search, what's the right interface? The agent needs: (a) an action space (modify layer config), (b) an observation space (metrics from the modified model), (c) a reward signal (PPL improvement, speed improvement, memory reduction). How would this integrate with lattice's existing `compute_perplexity()` and Criterion benchmarks? What RL algorithms are suitable for this discrete combinatorial search space (PPO on discrete actions, evolutionary strategies, Bayesian optimization)?

### Desired output

Concrete Rust trait signatures for composable attention and quantization. A config schema (YAML or Rust struct) for declarative model construction. Performance analysis of dispatch mechanisms. Survey of architecture search methods applicable to inference engines. Design blueprint for an agent-driven exploration loop.

---

## RQ-2: Model Pruning, Sparsity, and Parameter Elimination — Finding and Removing What's Not Needed

### Context

Recent research demonstrates that large language models contain massive redundancy — researchers have been able to **halve or more** the parameter count without meaningful quality loss. This is one of the most impactful research directions for lattice, both as a compression technique (faster inference, smaller models) and as a research tool (understanding model structure).

**lattice** currently has NO pruning, sparsity analysis, or parameter importance tooling. The KG has zero entities for pruning concepts. The codebase supports weight quantization (Q4, Q8, QuaRot) but not weight removal. lattice-tune has LoRA adapter loading but no training, so retraining after pruning is not yet possible (though issue #88 tracks this).

The goal is to build a **pruning/sparsity toolbox** within lattice that enables:

1. Analyzing which parameters/heads/layers/neurons are important
2. Removing unimportant components (structured or unstructured)
3. Measuring quality impact (PPL delta, generation quality)
4. Iterating: prune → measure → prune more or stop
5. Exporting pruned models as new, smaller SafeTensors checkpoints

### Questions to research deeply

1. **Survey of LLM pruning methods (2023-2026)**: Provide a comprehensive survey of methods that achieve ≥30% parameter reduction on decoder-only transformers (Llama, Qwen, Mistral class) with ≤1 PPL point degradation. For each method, detail: (a) what it prunes (weights, heads, layers, neurons, channels), (b) whether it needs retraining/fine-tuning after pruning, (c) structured vs unstructured sparsity, (d) computational cost of the pruning decision, (e) reported results (model, sparsity %, PPL before/after). Key papers to cover:
   - **SparseGPT** (Frantar & Alistarh, 2023) — one-shot unstructured pruning using approximate inverse Hessian
   - **Wanda** (Sun et al., 2023) — pruning by weight magnitude × input activation norm, no retraining
   - **SliceGPT** (Ashkboos et al., 2024) — structured pruning via computational invariant rotation (same authors as QuaRot — directly relevant since lattice already has Hadamard rotation infrastructure)
   - **ShortGPT** (Men et al., 2024) — removing entire transformer layers based on Block Influence (BI) scores
   - **The Unreasonable Ineffectiveness of the Deeper Layers** (Gromov et al., 2024) — layer removal analysis
   - **LLM-Pruner** (Ma et al., 2023) — structured pruning preserving inter-layer dependencies
   - **FLAP** (An et al., 2024) — fluctuation-aware structured pruning
   - **Lottery Ticket Hypothesis** (Frankle & Carlin, 2018) — existence of sparse subnetworks matching full performance
   - Any 2025-2026 papers that build on these

2. **SliceGPT deep dive** (highest priority for lattice): SliceGPT uses orthogonal rotation (PCA-based or random) to identify and remove dimensions from the residual stream. This is architecturally identical to what QuaRot does for quantization — lattice already has the Hadamard rotation infrastructure (`quant/quarot/`). **Detailed questions**: (a) How does SliceGPT's rotation differ from QuaRot's? Can they share the same rotation matrix? (b) What's the exact algorithm: compute importance per dimension → rotate → slice → adjust downstream weights? (c) How does it interact with RoPE (position encoding depends on dimension)? (d) What are the reported results for Qwen-family models specifically? (e) Can SliceGPT + QuaRot be composed (rotate → slice dimensions → quantize remaining to int4)?

3. **Layer removal and depth pruning**: ShortGPT and Gromov et al. show that many transformer layers can be removed entirely. **Questions**: (a) How is Block Influence (BI) score computed? Is it cheap enough to run during inference? (b) What's the typical pattern — are deeper layers always less important, or is it model-dependent? (c) After removing layers, do we need to adjust layer normalization or residual connections? (d) How does layer removal interact with GDN vs GQA layers in a hybrid model like Qwen3.5 — are GDN (linear attention) layers more or less removable than GQA (full attention) layers? (e) Can we remove layers gradually (one at a time, measuring quality after each) as an interactive exploration tool?

4. **Attention head pruning**: Individual attention heads can be pruned based on importance scores. **Questions**: (a) What methods compute head importance — gradient-based, activation-based, or entropy-based? (b) For GQA (where K/V heads are shared across multiple Q heads), how does pruning interact with the grouping? Pruning a Q head is cheap; pruning a shared K/V head affects all Q heads in its group. (c) What's the typical sparsity achievable in attention heads for 0.5-1B parameter models? (d) Does head pruning compose with Flash attention (which assumes a fixed number of heads for tiling)?

5. **FFN neuron pruning and structured sparsity**: SwiGLU FFN layers in modern transformers are 2.67x the hidden dimension (e.g., hidden=512, FFN intermediate=1376 for Qwen3.5-0.8B). **Questions**: (a) What fraction of FFN neurons can typically be pruned? (b) How do activation-based importance scores (Wanda-style: weight × activation) compare to gradient-based for FFN pruning? (c) Can we identify "dead neurons" (near-zero activation across a calibration set) cheaply during inference? (d) For MoE models (which Qwen3.5 has via GDN's delta-rule experts), how does expert pruning work — can we eliminate entire experts?

6. **Pruning without retraining**: For lattice (which has no training loop yet), methods that work without retraining are critical. **Questions**: (a) What quality is achievable with zero-shot pruning (SparseGPT, Wanda) at 50% sparsity on Qwen-class models? (b) If we later add LoRA training (#88), can we do "prune then LoRA-heal" — prune aggressively, then train a small LoRA adapter to recover quality? What are the results for this approach? (c) Is there a pruning method that can be applied incrementally (prune 10%, measure, prune 10% more) rather than one-shot?

7. **Implementation considerations for Rust/Metal**: (a) For unstructured sparsity (zero weights scattered in matrices), what's the inference speedup? Dense GEMV with zeros isn't faster — does lattice need a sparse matrix format and sparse GEMV kernel? (b) For structured sparsity (entire heads/layers/neurons removed), the model is just smaller — no special kernel needed. Is structured pruning always preferable for inference engines? (c) How to represent a pruned model in SafeTensors — smaller tensors with a pruning mask, or physically smaller tensors with remapped indices? (d) For Metal GPU: does Apple's MPS framework support sparse matrix operations, or is dense-with-zeros the only option?

### Desired output

A pruning method comparison table (method, type, retraining needed, sparsity %, PPL delta, compute cost). Deep technical analysis of SliceGPT composability with QuaRot. Design for a pruning toolbox in Rust: importance scoring, structured removal, quality measurement loop. Specific results on Qwen-family models where available. Implementation blueprint for "prune → measure → iterate" workflow in lattice.

---

## RQ-3: Metrics, Profiling, and Automated Experimentation Infrastructure for Transformer Research

### Context

**lattice** has `compute_perplexity()` on both CPU and Metal paths, and 17 Criterion benchmarks covering individual kernels. But for architecture exploration (composable layers, pruning experiments, quantization sweeps), we need much richer instrumentation: attention entropy per layer, activation norm tracking, memory bandwidth utilization, per-layer latency profiling, and an automated experiment runner that takes a config and produces a structured metrics report.

The vision: an agent (human or RL) modifies a model config → lattice runs inference with instrumentation → metrics are recorded → the agent uses metrics to decide the next modification. This requires cheap-to-compute metrics that provide architectural insight during inference, plus an experiment tracking system.

lattice also needs a **training-as-exploration** capability. Currently the LoRA training pipeline is a placeholder (issue #88 — `train_step_cpu` returns synthetic loss). A minimal viable training step (even just SGD on LoRA parameters) would enable "modify architecture → LoRA-heal → measure quality" loops.

### Questions to research deeply

1. **Metrics catalog for architecture insight**: For each metric below, explain: (a) mathematical definition, (b) computational cost during inference, (c) what architectural insight it provides, (d) whether it can be extracted from optimized paths (Flash attention, fused kernels) or requires a separate profiling pass.
   - **Attention entropy**: `H = -sum(softmax_weights * log(softmax_weights))` per head. Low entropy = focused attention = important head. High entropy = diffuse = candidate for linear attention or pruning.
   - **Attention sparsity**: fraction of attention weights below a threshold. Related to entropy but distinct — a head can have low entropy (focused on few positions) but different sparsity patterns.
   - **Layer-wise activation norms**: L2 norm of the residual stream after each layer. Sudden drops indicate layers that don't contribute much (correlates with Block Influence score for layer pruning).
   - **Effective rank**: number of singular values of the attention weight matrix above a threshold. Low effective rank = layer could use lower-rank attention.
   - **KV cache utilization**: for PagedKVCache, which tokens' KV entries are actually attended to? If a layer consistently ignores certain positions, those KV entries could be evicted or compressed.
   - **Speculative decoding acceptance rate**: per-draft-length statistics. How does acceptance rate vary by prompt type, temperature, draft model?
   - **Memory bandwidth utilization**: what fraction of Apple Silicon's theoretical peak (400 GB/s on M2 Max) does each operation achieve? This is the roofline model — operations below the roofline ridge are memory-bound and benefit from quantization; above are compute-bound and benefit from algorithmic improvements.

2. **Roofline model for Apple Silicon inference**: What are the exact compute (TFLOPS) and memory bandwidth (GB/s) ceilings for M1/M2/M3/M4 Pro/Max/Ultra? For each lattice operation (embedding lookup, attention QK^T, attention softmax, attention V projection, FFN gate/up GEMV, FFN down GEMV, RMSNorm, residual add, KV cache read/write), classify as compute-bound or memory-bound at typical dimensions. What tool/methodology gives per-kernel roofline position? Can `MTLCommandBuffer.GPUStartTime/GPUEndTime` give accurate per-dispatch timing on Metal?

3. **Automated experiment runner design**: For "sweep config parameter X from A to B, measure metrics Y", what's the design pattern in Rust? Requirements: (a) takes a model config (from the DSL in RQ-1), (b) loads model weights, (c) runs a calibration/eval corpus through inference with metrics collection, (d) records results as structured data (JSON-lines, SQLite, or similar), (e) supports parallel experiments (different configs on different threads/processes), (f) produces comparison reports (tables, statistical significance tests). Survey how other ML tools handle this: Weights & Biases, MLflow, Optuna, Ray Tune. What's the minimal Rust-native equivalent?

4. **Minimal viable LoRA training for exploration**: lattice needs a training step to enable "modify → train → measure" loops. The full e2e pipeline (issue #88) is large scope. What's the **minimal viable** training step?
   - For LoRA-only gradients: given frozen transformer weights W and LoRA matrices A, B at linear projections, derive the analytical gradient ∂L/∂A and ∂L/∂B without a general autograd engine. The key insight: the transformer is frozen, so we only propagate gradients through the LoRA injection points.
   - Memory requirements: for Qwen3.5-0.8B with rank-8 LoRA on q_proj + v_proj across 24 layers, how much activation memory per training sample?
   - Can we use a first-order approximation (REINFORCE-style policy gradient) for even cheaper "training" that doesn't need a backward pass at all?
   - CPU vs Metal: is CPU sufficient for small LoRA training steps, or does the latency make exploration impractical?

5. **Gradient-free architecture search signals**: Without any training, what signals can guide architecture decisions? (a) Perplexity delta when swapping a layer type or removing a layer. (b) Mutual information between layer input and output (layers with high mutual information are near-identity and candidates for removal). (c) Activation outlier analysis (layers with extreme outliers are hard to quantize). (d) Attention pattern classification (which layers show "attention sinks" vs distributed attention). Which are most predictive of quality impact and cheapest to compute? Are there published correlations between these signals and actual quality (PPL/benchmark) deltas?

6. **Experiment tracking and reproducibility**: What should the experiment record schema look like? Proposal: `{ experiment_id, timestamp, model_config, git_commit, hardware_info, corpus_info, metrics: { ppl, tok_s, memory_mb, per_layer: [{entropy, norm, latency}] }, notes }`. How to ensure reproducibility — deterministic sampling seeds, fixed calibration corpora, hardware identification? Should results go in a local SQLite database, a JSON-lines file, or the khive KG (via `memory.remember`)?

### Desired output

Metrics catalog with cost/insight analysis. Roofline model for Apple Silicon. Experiment runner architecture in Rust. Minimal LoRA training derivation with memory budget. Gradient-free search signal analysis with published evidence of predictive power. Experiment schema design.

---

## RQ-4: Metal GPU Optimization + KV Cache Quantization Pipeline for Apple Silicon

### Context

**lattice** runs on Apple Silicon via Metal compute shaders (15,546 lines in `metal_qwen35.rs`). Current state:

- Decode attention: fused, online-softmax, one threadgroup per KV-head — works well for M=1 decode
- Prefill attention: processes all seq_len rows per KV head in one threadgroup — **NOT flash-tiled**, bandwidth-inefficient at long sequences
- KV cache: all f32 — no f16, int8, or int4 storage
- Metal throughput: ~118 tok/s under load (vs MLX 259 tok/s on same hardware), 18% vs 27% memory bandwidth utilization
- Metal shaders: all inline as Rust string constants in one file (no `.metal` files, no IDE support, no build-time validation)

The goal is two-fold: (1) close the MLX throughput gap via Metal Flash Attention 2 for prefill + kernel optimization, and (2) implement the full KV cache quantization chain (f16 → int8 → int4 with rotation).

9 GitHub issues cover this: #77 (GPU contention), #85 (MLX kernel study), #86 (shader extraction), #118 (quantized KV storage), #120 (fused quantized attention), #121 (f16 KV), #122 (pre-RoPE K quant), #123 (WHT/SRFT rotation for int4), #126 (Metal FA2 prefill).

### Questions to research deeply

1. **Metal Flash Attention 2 implementation**: FA2 tiles both Q and KV dimensions, keeping partial softmax accumulators in threadgroup memory. For Apple Silicon Metal compute:
   - What are the SRAM/threadgroup memory sizes on M1/M2/M3/M4? What's the optimal tile size (BQ, BKV) for each generation?
   - How does MLX's `scaled_dot_product_attention.metal` tile? Threadgroup sizes? Causal mask handling within tiles?
   - The Draw Things Metal FA2 (open source, MIT) claims 43-120% latency reduction. What's their tiling strategy vs Dao's CUDA FA2?
   - For GQA (G groups of H/G query heads sharing KV heads), what FA2 modifications are needed?
   - Can FA2 tiles read from paged KV buffers (non-contiguous physical pages), or does it require contiguous KV layout?
   - Metal-specific features: `simdgroup` operations, `imageblock`, tile shading — which do FA2 implementations exploit?

2. **KV cache f16 → int8 → int4 chain**:
   - **f16**: On Apple Silicon (which uses fp16 not bf16), what's the actual PPL delta for f16 vs f32 KV? Is it truly lossless?
   - **Pre-RoPE K quantization** (KVQuant Section 3.2): Quantizing K before RoPE preserves per-channel outlier structure. Implementation cost: RoPE applied per-attention-step vs once at write time. Quantify the extra compute on Apple Silicon.
   - **KIVI asymmetric K/V**: Per-channel K, per-token V. Why? How does this interact with GQA shared heads?
   - **WHT vs SRFT for KV rotation**: Apple's SRFT paper (arxiv:2605.05699) claims "negative latency" on M1. How does SRFT compare to WHT (which lattice already has for weight QuaRot)? Can they share the same rotation matrices?
   - **TurboQuant's "incremental FP16 decode buffer"**: 14x speedup reported. How does it work? Applicable to our Metal architecture?
   - **RateQuant optimal K/V bit budget**: Different bits for K vs V. Optimal allocation for Qwen?

3. **Fused quantized attention kernel for Metal**: A kernel that reads int8/int4 KV pages and dequantizes inline during QK^T and attn×V, avoiding the dequantize→f32→attention roundtrip.
   - What's the arithmetic intensity tradeoff — is inline dequant faster than separate dequant+f16-attention on Apple Silicon?
   - How does this compose with FA2 tiling (tiles need to dequant within threadgroup memory)?
   - Reference implementations: does llama.cpp's Metal backend have fused quantized attention?

4. **Closing the MLX throughput gap**: MLX achieves ~27% bandwidth utilization vs lattice's ~18% on M2 Max.
   - What techniques from MLX's `.metal` kernels are directly portable? (Tiling, threadgroup sizing, SIMD-group reductions)
   - The ~10% gap attributable to Apple's private AMX blocks (via MPS/MPSGraph) — can this be closed at all through public Metal API?
   - What's the expected speedup from Metal FA2 + f16 KV + fused quantized attention combined?

5. **Shader extraction and organization**: 15.5K lines of inline Metal shaders need extraction to `.metal` files. Design questions:
   - `include_str!` (compile-time inclusion, runtime compilation) vs `build.rs` + `xcrun metal` (compile-time Metal library)?
   - How does MLX organize their Metal shader library? How does llama.cpp?
   - Can we add `xcrun metal -c` validation to CI (macOS runner only)?

6. **Prefix cache wiring**: PrefixPageCache is fully built but never queried by `forward_prefill_impl()`. The wiring needs: hash prompt prefix → cache lookup → on hit, skip matched tokens in prefill → on miss, store computed KV pages after prefill. For Metal: CPU-side tree/hash → Metal buffer references. How does SGLang's RadixAttention handle the CPU-GPU boundary for prefix matching?

### Desired output

Metal FA2 implementation blueprint with tile sizes per M-series generation. KV quantization pipeline with implementation order and expected quality/speed numbers. Fused quantized attention kernel design. MLX kernel technique analysis. Shader organization recommendation. Prefix cache wiring design.

---

## RQ-5: Product Surface, Serving Architecture, and Seed-Round Competitive Positioning

### Context

**lattice** is a 57.5K LOC pure-Rust transformer inference engine with: 5 crates, 10 attention mechanisms, QuaRot Q4 quantization, speculative decoding (N-gram + MTP), continuous batching module, SIMD-accelerated embeddings (HF parity at cos ≥ 0.9998 on 4 model families), and hybrid GDN+GQA architecture support. But it has **zero product surface** — no unified CLI, no daemon, no HTTP API, no way for a non-Rust-developer to use it.

Every competitor has a `serve` command: llama.cpp (`llama-server`), MLX (`mlx_lm.server`), Ollama (`ollama serve`), vLLM (`vllm serve`). Without this, lattice is a library, not a product. We're preparing for seed-round fundraising.

4 GitHub issues cover the serving stack: #91 (unified CLI), #92 (daemon), #93 (OpenAI-compatible API), #94 (Anthropic Messages API).

### Questions to research deeply

1. **Rust HTTP server architecture for LLM inference**:
   - axum + tokio pattern for SSE streaming where inference is synchronous (Metal GPU blocks a thread). Architecture: async HTTP layer → channel → dedicated inference thread → Metal GPU. Show concrete axum handler code for `/v1/chat/completions` with SSE streaming.
   - Can two `MTLCommandQueue`s from the same `MTLDevice` execute concurrently on Apple Silicon? If sequential, what's the best pattern for interleaving prefill and decode?
   - How do llama.cpp-server and Ollama handle concurrent requests + model loading + GPU memory management?
   - Request cancellation: what happens when a client disconnects mid-stream? How to abort a decode batch?
   - Chat template application: parsing `tokenizer_config.json` Jinja templates in Rust. Is there a Jinja crate, or do we need hardcoded templates per model family?

2. **OpenAI-compatible API surface (minimal viable)**:
   - What subset of the OpenAI API do LangChain, LlamaIndex, Continue, Cursor, Aider actually call? What's the 80% subset we can implement in v1?
   - SSE streaming format: `data: {...}\n\n` terminated with `data: [DONE]\n\n`. Exact JSON schema for delta objects with token-by-token streaming.
   - The Anthropic Messages API (#94) uses typed SSE events: `message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`. Exact event format for SDK compatibility.
   - How do llama.cpp and Ollama handle `response_format: {"type": "json_object"}` for structured output? Does it require grammar-constrained decoding integration?

3. **Unified CLI design**:
   - Survey CLI designs of: `ollama` (pull/run/serve/list/show/rm), `llama-cli` (flags-based), `mlx_lm.generate` / `mlx_lm.server`, `vllm serve`.
   - What's the optimal UX for `lattice`? Proposed: `lattice pull/chat/complete/embed/serve/bench/info/quantize`. Which subcommands are MVP for seed-round demo?
   - Model management: downloading from HuggingFace Hub. Rust crate for HF Hub API? Or shell out to `huggingface-cli`?
   - Auto-pull on first use (download model if not local) — UX consideration. Progress bars, cache management, disk space checks.

4. **Seed-round competitive positioning**:
   - What companies have raised seed rounds for LLM inference/serving in 2024-2026? Their pitch, metrics, differentiation.
   - **lattice's unique angle**: pure Rust (no Python, no CUDA, no ONNX), composable architecture research platform (not just serving), Apple Silicon native, formal verification integration (styx project), rotation-aware quantization (QuaRot), 10 attention mechanisms. How does this position against llama.cpp (C++, broad hardware), MLX (Apple's own, Python), vLLM (Python, CUDA), Candle (Rust but limited features)?
   - What performance benchmarks do investors care about? Specific numbers that would be "impressive" for Qwen-class 0.8B model on M2 Max. (tok/s, TTFT, concurrent throughput)
   - Is embedding a differentiator or commodity? lattice-embed achieves HF parity with SIMD-accelerated quantized search.
   - Does on-device LoRA fine-tuning create meaningful differentiation?
   - What does the realistic 3-minute investor demo look like? Script it with `lattice` CLI commands.

5. **Cross-framework benchmark methodology**:
   - For an apples-to-apples comparison against llama.cpp, MLX, Ollama: what model, what quantization, what prompt, what sequence length, what hardware, what measurement methodology?
   - Statistical rigor: how many runs, warmup, trimming, confidence intervals. How does llama.cpp's `llama-bench` measure?
   - Beyond raw tok/s: TTFT (time to first token), p99 latency under concurrent load, peak memory, model load time. Which metrics matter most for different use cases (interactive chat vs batch embedding vs RAG pipeline)?

6. **SGLang RadixAttention for prefix sharing**:
   - How exactly does the radix tree work? Key structure, node sharing between sequences, memory overhead.
   - For integration with lattice's PagedKVCache (page_size=256): page-aligned vs token-aligned matching?
   - Cache hit rates for: multi-turn chat (system prompt reuse), RAG (shared document context), agent loops (tool-use prompt reuse).
   - LoRA namespace isolation in the radix tree (different adapters produce different KV for same prefix).
   - Eviction policy: LRU at leaf level, or reference-count-based?

### Desired output

Complete axum server architecture with code examples. OpenAI API compatibility matrix. CLI design spec. Competitive landscape table with feature comparison. Specific benchmark targets for seed-round. Cross-framework measurement methodology. RadixAttention data structure design. Investor demo script.

---

## How to use these questions

Feed each RQ as a **separate ChatGPT Deep Research session**. The questions are self-contained — each provides enough context to produce a useful answer without the others.

**Priority order**:

1. **RQ-2** (pruning/sparsity) — most novel for lattice, highest research density needed
2. **RQ-1** (composable DSL) — architectural foundation for everything else
3. **RQ-4** (Metal + KV quant) — performance-critical, needs specific numbers
4. **RQ-3** (metrics + experiments) — infrastructure for research loops
5. **RQ-5** (product surface) — most well-understood domain, can start before research returns

After responses come back, digest into:

1. KG entities (one per paper/concept/method discovered)
2. ADRs for the top-priority items (composable architecture ADR, pruning toolbox ADR, serving stack ADR)
3. Updated issue descriptions with research findings
4. Show plan with evidence-backed scope decisions
