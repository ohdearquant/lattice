# Design: lattice-inference

## Current Design

**Multi-model inference engine.** Unified crate serving 4 model families: Qwen3-Embedding (GQA decoder, last-token pooling), BERT/BGE (encoder, mean pooling), Cross-encoder (BERT + classifier head), Qwen3.5 (hybrid GDN+GQA, generation). Each has its own weight format, forward pass, and tokenizer. Entry point: `QwenModel::from_directory()` at `qwen.rs:404` -- detects single vs sharded safetensors, loads config from `config.json`, selects tokenizer, initializes Metal GPU if available. `ModelInferenceConfig` at `qwen.rs:244` provides runtime knobs (`eos_token_id`, `rope_table_max_seq_len`, `gpu_max_seq_len`) loaded from `inference_config.json`. Core types: `QwenConfig` (`qwen.rs:192`), `QwenWeights<'a>` (`f32_weights.rs:597`), `ForwardBuffers` (`qwen.rs:297`), `MetalForwardPass` (`metal.rs:1067`).

**Metal GPU: MSL template approach.** The Metal compute pipeline is a single 936-line MSL source template (`metal.rs:17-953`) compiled at runtime via `Device::new_library_with_source`. Model-specific constants are injected via string replacement before compilation: `msl_source_for(head_dim, gqa_groups)` at `metal.rs:1050-1058` substitutes 5 placeholders (see Invariants). Templates over compiled kernels: `head_dim` varies across models (64 for 0.6B, 80 for 4B, 128 for 8B+). Pre-compiling a binary per `head_dim` would require shipping N kernel binaries and a dispatch table. String-template substitution produces a single source of truth -- one MSL template, one validation function, one code path. Runtime compilation cost is ~50ms at model load, amortized over millions of forward calls. **Configurable head_dim shipped 2026-04-29** (commit `94f7d331e8`): previously hardcoded to `head_dim==128` and `gqa==2`. Now validation only requires `head_dim % 4 == 0` (float4 alignment) and valid GQA divisibility. This unblocked 0.6B (`head_dim=64`) and 4B (`head_dim=80`) on GPU. 17 MSL kernel functions compile into 7 pipeline states. GEMM variants selected dynamically by sequence length (r8 short, r16 medium, r32 long). Fused attention uses online softmax in registers. All buffers use `StorageModeShared` -- zero-copy unified memory on Apple Silicon.

**CPU fallback and NEON.** CPU activates when: (1) `LATTICE_NO_GPU` env var set, (2) `MetalForwardPass::new()` fails validation or device init, (3) GPU error at forward time logs warning and falls back. Selection at `qwen.rs:440-456`. Qwen3-Embedding CPU path: `cpu.rs` functions (`matmul_bt`, `rms_norm`, `silu_inplace`, `elementwise_mul`) use NEON/Accelerate BLAS. Qwen3.5 Q8 NEON path: `neon_forward.rs` (1926 LOC) -- full Q8_0 quantized forward pass. NEON kernel at `neon.rs:108-197`: 4-row parallel matvec via `vmull_s8 + vmlal_s8 -> vpadalq_s16 -> vaddvq_s32`. GPU is ~10-50x faster than CPU for embedding; CPU path exists for compatibility and testing, not production throughput.

**Weight loading: single vs sharded.** Single-file (`model.safetensors`): mmap via `SafetensorsFile::open()` at `f32_weights.rs:139`. Zero-copy -- `Tensor2D`/`Tensor1D` borrow directly from mmap on aligned LE platforms. F16/BF16 lazy-convert via `OnceLock<Box<[f32]>>`. Sharded (4B models): `model.safetensors.index.json` maps tensor names to shard filenames. `ShardedSafetensors::open_index()` at `f32_weights.rs:919` sets up lazy reader; shards opened on demand, cached. `load_qwen_weights_owned()` copies into `ShardedQwenBacking` (owned `Vec<f32>`) -> `QwenWeights<'static>`. RFC 1857 drop order ensures weights outlive backing via `SafetensorsStorage` held in `QwenModel._storage` (`qwen.rs:29`). Sharding required for 4B because single safetensors exceeds HuggingFace's 5GB shard limit.

**Embedding cache.** `Mutex<HashMap<u64, Vec<f32>>>` at `qwen.rs:391`. Key = `hash_token_ids(&ids) -> u64`. Cap: 10K entries (~40MB at 1024d). Eviction: flush-all at capacity (`qwen.rs:723-725`) -- simple but effective; embeddings are deterministic, so cache misses just recompute. LRU would add complexity for minimal benefit given mostly-unique-text workloads. Persistence: binary format `[hash:u64, dim:u32, floats:f32*dim]` per entry. Cache hit returns in <1us vs ~100ms forward pass.

## Alternatives Rejected

| Alternative                            | Rejected Because                                                                                                                                                                                    | Date       |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Pre-compiled MSL kernels per head_dim  | Would require N kernel binaries + dispatch table. Runtime compilation is ~50ms (once at load), negligible vs model lifetime. Single template = single source of truth, easier to audit.             | 2026-04-29 |
| LRU eviction for embedding cache       | Adds `lru` dependency and per-access bookkeeping. Retrieval workload is mostly unique texts, so eviction policy matters less than hit rate. Flush-all is O(1) and deterministic recompute is cheap. | --         |
| Copy-based weight loading (no mmap)    | Doubles memory for large models. Mmap enables zero-copy tensor access on aligned platforms. Only sharded path copies (unavoidable -- tensors span shard boundaries).                                | --         |
| Single-precision only (no F16/BF16/Q8) | Limits model support. F16/BF16 halves VRAM. Q8_0 enables Qwen3.5-2B on CPU within memory budget. Lazy conversion via `OnceLock` keeps API simple.                                                   | --         |
| Separate `.metal` asset files          | Cargo doesn't natively bundle GPU assets. `include_str!` loses the template substitution pattern. Inline const string keeps template + substitution + compilation in one file.                      | 2026-04-29 |

## Invariants

**MSL Placeholder <-> Rust Validation Correspondence:**

| MSL Placeholder        | Used In                                        | Rust Source                                        | Valid Values                              |
| ---------------------- | ---------------------------------------------- | -------------------------------------------------- | ----------------------------------------- |
| `__FA_HEAD_DIM__`      | `fused_attention` tile dim, threadgroup memory | `metal.rs:1054`, validated at `metal.rs:1021-1029` | nonzero, divisible by 4                   |
| `__FA_GQA_GROUPS__`    | `fused_attention` rows/TG, Q-head mapping      | `metal.rs:1055`, validated at `metal.rs:1030-1042` | positive int, `num_heads % kv_heads == 0` |
| `__FUSED_C_HEAD_DIM__` | `fused_qk_norm_rope` vector length             | `metal.rs:1056` (= head_dim)                       | same as `__FA_HEAD_DIM__`                 |
| `__FUSED_C_HALF_DIM__` | `fused_qk_norm_rope` RoPE pair count           | `metal.rs:1057` (= head_dim/2)                     | derived                                   |
| `__FUSED_C_THREADS__`  | `fused_qk_norm_rope` threadgroup size          | `metal.rs:1058` (= head_dim/2)                     | derived                                   |

**INV-1.** All 5 placeholders must be substituted before Metal compilation. If broken: MSL syntax error (unresolved identifiers).
**INV-2.** `head_dim % 4 == 0`. If broken: `validate_fused_kernel_shape` returns error, model falls back to CPU. Required for Metal `float4` vector operations.
**INV-3.** `num_attention_heads % num_key_value_heads == 0`. If broken: same fallback. GQA groups must be integer.
**INV-4.** On Metal validation failure, `self.metal = None` and all forward passes use CPU path (`qwen.rs:452-456`). If broken: panic or incorrect results on unsupported GPU configs.
**INV-5.** `ShardedQwenBacking` must outlive `QwenWeights`. If broken: use-after-free. Enforced by `SafetensorsStorage` held in `QwenModel._storage` with RFC 1857 drop order (`qwen.rs:29`).
**INV-6.** Cache key = hash of token ID sequence, not raw text. If broken: different whitespace/encoding produces different cache entries (benign but wasteful). Intentional -- token hash is cheaper.
**INV-7.** Fused QKV weight shape = `[q_dim + 2*kv_dim, hidden_size]`. If broken: GEMM produces garbage. Built at load time (`f32_weights.rs:607`).
**INV-8.** `ForwardBuffers` pre-allocated for `max_seq_len`. If broken: buffer overflow in forward pass. Allocated once at model load (`qwen.rs:297`).

**Model Compatibility Matrix (post configurable head_dim):**

| Model                | head_dim | GQA groups | Metal GPU | Notes                        |
| -------------------- | -------- | ---------- | --------- | ---------------------------- |
| Qwen3-Embedding-0.6B | 64       | 2          | YES       | 64 % 4 == 0                  |
| Qwen3-Embedding-4B   | 80       | 2          | YES       | 80 % 4 == 0, sharded weights |
| Qwen3-8B             | 128      | 4          | YES       | Original supported config    |
| Qwen3.5-14B          | 128      | 5          | YES       | Hybrid GDN+GQA architecture  |

## Known Concerns Acknowledged

**KC-1.** ~161 unsafe blocks across the crate (`lib.rs:4` declares 153 but is stale; tracking issue #1306). Categorized by TCB policy. Note: per-category counts below use line-match counts not block counts, so categories may overlap where a single block contains multiple unsafe operations.

| Category                               | Count | Location                      | Justification                                           |
| -------------------------------------- | ----- | ----------------------------- | ------------------------------------------------------- |
| Metal FFI                              | 73    | `metal.rs`, `metal_qwen35.rs` | Raw pointer access to GPU buffers, command encoding     |
| CPU SIMD                               | 49    | `cpu.rs`                      | NEON intrinsics for matmul/norm in compute-hot paths    |
| GDN fused SIMD                         | 46    | `gdn_fused.rs`                | NEON for fused GatedDeltaNet attention                  |
| Pooling SIMD                           | 9     | `pool.rs`                     | NEON for L2 norm, last-token/mean pooling               |
| Other (F16, tokenizer, mmap, sampling) | ~20   | Various                       | F16 byte reinterpretation, mmap pointer cast, Q8 matvec |

All are in compute-hot paths where safe abstractions would impose measurable overhead.

**KC-2.** `std::mem::transmute` for self-referential mmap lifetime (`qwen.rs:461`, `bert.rs` same pattern). Sound because `_storage` field drops after `weights` (RFC 1857 struct field drop order). Safe alternatives (`ouroboros`, `rental`) add dependency complexity for this single use case.

**KC-3.** Flush-all cache eviction at 10K cap instead of LRU. Embedding workload is mostly unique texts (retrieval indexing). LRU bookkeeping cost exceeds benefit. Deterministic recompute makes cache misses non-catastrophic.

**KC-4.** CONFIG.md lists Qwen3-Embedding-4B (`head_dim=80`) as Metal-incompatible. Stale after 2026-04-29 configurable head_dim change (commit `94f7d331e8`). The code is correct; CONFIG.md should be updated.

## Baseline Metrics

| Dimension | Metric                                 | Value   | Measured   | Threshold | Command                                                               |
| --------- | -------------------------------------- | ------- | ---------- | --------- | --------------------------------------------------------------------- |
| perf      | Forward latency (GPU, 0.6B, 512 tok)   | pending | --         | +20%      | `cargo bench -p lattice-inference --bench embed`                      |
| perf      | Forward latency (CPU, 0.6B, 512 tok)   | pending | --         | +20%      | `LATTICE_NO_GPU=1 cargo bench -p lattice-inference --bench embed`     |
| perf      | Model load time (0.6B, single file)    | pending | --         | +50%      | (part of bench)                                                       |
| perf      | Cache hit latency                      | <1us    | 2026-04-29 | +100%     | (part of bench)                                                       |
| security  | unsafe block count                     | 161     | 2026-04-29 | +5        | `grep -rw "unsafe {" crates/inference/src/ --include="*.rs" \| wc -l` |
| quality   | Model configs passing Metal validation | 4/4     | 2026-04-29 | 4/4       | see model compatibility matrix above                                  |

## Change Protocol

1. Read this DESIGN.md before modifying any file in `crates/inference/`.
2. Check Baseline Metrics -- run the measurement commands and compare against thresholds.
3. Check Known Concerns -- ensure your change doesn't re-introduce a concern already acknowledged.
4. Re-measure baselines after your change and update the table if values shift.
