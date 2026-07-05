# ADR-070: CPU Embedding Performance: Fused Batched Encode and the ONNX Parity Gate

**Status**: Proposed
**Date**: 2026-07-05
**Scope**: `crates/inference` BERT/embedding CPU forward path (`model/bert.rs`,
`attention/standard.rs`, `forward/cpu/*`, `pool.rs`); `crates/embed` `encode_batch`
consumers; embedding benchmarks
**Depends on**: ADR-058 (measure-first performance workflow)

---

## Context

lattice-embed is being positioned as a default embedding hot path for downstream vector
stores and retrieval systems, replacing ONNX Runtime based providers. That positioning
only holds if lattice is at least as fast as ONNX Runtime on the same model and hardware.

A same-model comparison (all-MiniLM-L6-v2, 384-dim, CPU, identical input corpus, output
cosine agreement 0.99993 so both sides compute equivalent work) measured on 2026-07-05
shows lattice losing across the board:

| Metric | lattice-embed | ONNX Runtime | Gap |
|---|---|---|---|
| Single-text latency p50 / p90 | 3.02 / 3.70 ms | 2.05 / 2.37 ms | ORT 1.4-1.6x faster |
| Batch throughput, bs=1 (texts/s) | 560 | 1074 | ORT 1.9x |
| Batch throughput, bs=8 | 525 | 1112 | ORT 2.1x |
| Batch throughput, bs=32 | 486 | 733 | ORT 1.5x |
| Batch throughput, bs=64 | 481 | 691 | ORT 1.4x |

The numbers are indicative (measured under background load, loadavg ~11 on 12 logical
cores); the ordering is robust. Two structural causes, both verified by reading the
source rather than inferred from the numbers:

1. **No fused batching.** `encode_batch` tokenizes inputs together but then loops
   sequentially per sequence through the forward pass, re-slicing each item to its real
   length. `AttentionBuffers` is sized by `max_seq_len` alone with no batch dimension,
   and the attention entry point asserts single-sequence input shape. The result is N
   repeated batch-1 forwards; lattice throughput is flat (~500 texts/s) at every batch
   size, which is exactly the signature of this structure. Notably, the WordPiece
   tokenizer already computes a correct padded batch and attention mask (`pad_batch` /
   `pad_ids`); the forward path discards them.
2. **Thread-starved orchestration.** The BERT forward orchestration and the non-macOS
   fallback kernels are single-threaded. On macOS, `matmul_bt` dispatches individual
   GEMM calls to Apple Accelerate (`forward/cpu/matmul.rs:84`), whose `cblas_sgemm`
   applies multi-threaded AMX tiling even at M=1 (`forward/cpu/blas.rs:74`), but the
   per-sequence loop hands it many small GEMMs instead of one large one, and the
   non-GEMM position-wise ops and the fallback path run on a single thread. ONNX
   Runtime both batches its GEMMs and runs an intra-op thread pool across the whole
   graph.

The position-wise kernels (`matmul_bt`, `layer_norm`, bias/GELU) are already
row-count-generic, so the bulk of the FLOPs (QKV projection and FFN) can be fused
across `batch * padded_seq_len` rows with no kernel changes. Only the O(seq^2)
attention score/context step is inherently per-sequence.

## Decision

1. **Fuse the batched encode path.** `encode_batch` keeps its public signature and
   becomes a genuinely batched forward: position-wise ops (embedding lookup, QKV projection,
   FFN, layer norm, bias/GELU) execute as single row-fused matmul calls across
   `batch * padded_seq_len` rows; the attention score/context step may loop per
   sequence. The tokenizer's existing padded batch and attention mask become the
   forward path's inputs instead of being discarded.
2. **Mask correctness is a hard invariant.** Padded positions are excluded via additive
   masking consistent with the existing attention masking conventions; mean pooling
   divides by real (mask-sum) length, never padded length; CLS pooling reads position 0
   per sequence; L2 normalization behavior is unchanged. All existing softmax
   fail-closed guards (NaN, +inf, all-masked) are preserved unweakened in the batched
   path.
3. **Add backend-aware intra-op thread parallelism.** Where a GEMM dispatches to a
   threaded BLAS backend (Accelerate on macOS), no additional row threading is layered
   on top: nesting a thread pool over BLAS worker threads oversubscribes cores. Bounded
   row-chunking (capped at min(available_parallelism, 8), existing threading idioms, no
   new dependencies) applies only to the non-BLAS fallback kernels and to non-GEMM
   position-wise ops (layer norm, bias/GELU, pooling) where an A/B measurement shows a
   win. Fusion first, threading only where measurement justifies it.
4. **Adopt an ONNX parity gate for the hot-path claim.** The "default embedding hot
   path" positioning, and any external contribution built on it, is blocked until
   lattice meets or beats ONNX Runtime CPU on the same model at single-text p50 and at
   batch throughput (bs 8 through 64), measured on an idle machine per the ADR-058
   workflow. Numbers measured under load are treated as directional only and are never
   quoted externally.

## Consequences

**Positive**
- Closes the measured 1.4-2.1x deficit with work that is structurally understood, not
  speculative: batching recovers the flat-throughput loss, threading recovers the
  single-text loss.
- The batched path also benefits every internal consumer of `encode_batch` (bulk
  ingestion, index builds), independent of any external positioning.
- The parity gate turns a marketing claim into a measurable acceptance criterion and
  prevents shipping performance claims that a competitor benchmark would contradict.

**Negative / risks**
- `attention/standard.rs` is correctness-critical; introducing a batch dimension and
  masking there is the riskiest edit in this change. Mitigation: a mutation-sensitive
  parity test encodes a mixed-length batch through the new fused path and through a
  reference loop of single-sequence encodes, asserting per-text cosine >= 0.99999 and
  max-abs-diff <= 1e-4, including boundary cases (one sequence at max length, one at
  length 1). The HF parity suite must stay green.
- Padding waste: fusing across `batch * padded_seq_len` rows spends FLOPs on pad
  positions. For typical mixed-length batches this is far cheaper than losing the GEMM
  fusion; length-bucketing is a possible later optimization and is out of scope here.
- Thread-pool interaction: embedding calls issued from an async runtime must not
  oversubscribe cores. The cap plus chunked work items bound this; the bench gate
  measures the end result rather than assuming it.

## Acceptance criteria

- Batched-vs-sequential parity test green (cosine >= 0.99999, max-abs-diff <= 1e-4,
  mixed and boundary lengths), and it fails when the masking or pooling logic is
  deliberately broken (mutation check).
- HF parity gate green; clippy `-D warnings`; fmt clean.
- A/B bench on the same machine, same session: batch throughput at bs 8, 32, 64
  materially above the pre-change baseline (expected roughly 1.5-2x from the fusion),
  and single-text p50 not regressed. This is the merge bar for the implementation PR;
  the binding external bar is the ONNX gate below.
- Idle-machine comparison vs ONNX Runtime (same model, same corpus): lattice >= ORT on
  single-text p50 and on batch throughput at bs 8, 32, 64. This is the gate for any
  external "hot path" claim.

## Alternatives considered

- **Ship the integration first, fix performance later.** Rejected: the integration's
  entire pitch is replacing the incumbent ONNX provider on its hot path; arriving
  measurably slower undermines the proposal and burns credibility that a later fix does
  not recover.
- **Threading only, no fused batching.** Rejected: threading N independent batch-1
  forwards recovers some throughput but keeps the memory-traffic and dispatch overhead
  of N separate forwards, and profiling of the flat throughput curve shows the batching
  structure, not raw compute, is the dominant loss at bs >= 8.
- **Offload embedding GEMMs to Metal.** Out of scope: the embedding path is
  deliberately pure-CPU and cross-platform (Linux ingest and a planned wasm32 target
  depend on that property); GPU acceleration is a separate, additive decision.
- **Length-bucketed batching.** Deferred: a refinement on top of fused batching, not a
  substitute for it; adds scheduling complexity before the simple fusion's headroom is
  exhausted.
