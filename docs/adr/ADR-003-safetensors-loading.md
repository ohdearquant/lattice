# ADR-003: SafeTensors Weight Loading

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`\
**Proposed amendment**: [ADR-088](ADR-088-sealed-native-embedding-attestation.md) would compose
the existing per-file mmap authority into a retained whole-snapshot embedding preparation
capability without creating a second mapping token.

---

## Context

Loading model weights is the dominant startup-latency operation for inference servers. For Qwen3-Embedding-0.6B (28 layers, 1024 hidden size), the checkpoint is distributed across multiple safetensors shards. The design must address:

1. **Parse cost**: The safetensors format uses a JSON header to describe tensor locations within the binary blob. Deserializing that full header into a generic JSON tree would add allocation and obscure duplicate-member checks; the crate already carries `serde_json`, so bounded string-token decoding can still reuse its standards-compliant Unicode handling.
2. **Copy cost**: F32 weights on a little-endian host can be used in-place from the memory-mapped file if alignment permits. Copying them defeats the point of safetensors.
3. **Type conversion**: F16 and BF16 tensors cannot be used in-place by the f32 compute kernels. They must be converted and cached.
4. **Sharding**: Multi-shard checkpoints must be served transparently, opening each shard lazily to avoid holding all file descriptors simultaneously.

Relevant implementation: `src/weights/f32_weights.rs`.

The file contains a hand-written structural `JsonParser` (with `serde_json` used only for one bounded string token at a time), `memmap2::Mmap` for the binary data, `TensorMeta` with an `OnceLock<Box<[f32]>>` for lazy conversion cache, and `ShardedSafetensors` for multi-file checkpoints.

Key code paths:

```rust
// Zero-copy F32 path
fn bytes_to_f32_slice(bytes: &[u8]) -> Option<&[f32]> {
    // Returns None if misaligned; caller falls back to copy
}

// F16 conversion (behind `f16` feature flag)
fn convert_f16_bytes_to_f32(bytes: &[u8]) -> Box<[f32]>

// Lazy conversion cache
struct TensorMeta {
    // ...
    cache: OnceLock<Box<[f32]>>,
}

// Sharded access — opens shard file on first access
struct ShardedSafetensors { /* lazy-open per-shard */ }
```

For Qwen3, weights are pre-fused at load time:

- `fused_qkv`: Q, K, V projection rows concatenated into a single `Vec<f32>` → one GEMM replaces three
- `fused_gate_up`: gate and up projection rows concatenated → one GEMM replaces two

The `ShardedQwenBacking` struct holds these owned `Vec<f32>` allocations alongside the `ShardedSafetensors`; the `QwenModel` holds both in a `SafetensorsStorage` enum, and Rust's RFC 1857 struct field drop order guarantees that tensor slices (which borrow from `ShardedQwenBacking`) are dropped before the backing allocation.

---

## Decision

Use **`memmap2::Mmap` with a hand-written JSON header parser** for zero-copy weight access. F32 weights on LE-aligned hardware bypass deserialization entirely. F16/BF16 conversions are deferred to first access and cached in a `OnceLock<Box<[f32]>>` per tensor. Shards are opened lazily on first tensor access. Qwen3 weights are fused (QKV, gate+up) eagerly at load time into owned `Vec<f32>` to collapse multiple GEMM calls.

### Ingress validation contract

`weights::ingress` is the single internal validation seam between untrusted external bytes and
trusted tensor data. A format-specific reader may parse framing and metadata before the seam, but a
borrowed view or owned tensor must not escape the reader until the representation has passed the
following checks, in order:

1. The source representation and dtype are supported by that reader.
2. Shape products and byte extents use checked arithmetic and agree exactly with the payload.
3. Decoded floating-point values are finite.
4. A quantizer verifies that its scale, bias, and encoded values satisfy the destination format's
   representability postconditions before publishing output.
5. Failures retain source and tensor attribution. Multi-tensor model construction is atomic at the
   public facade: it assembles into local values and returns `Err` without exposing a partially
   built model if any tensor fails.

The shipped safetensors route normalizes F32/F16/BF16 payloads to `f32` before calling
`ingress::validate_ingested_tensor` from `SafetensorsFile::get_f32_tensor`. Header parsing has
already checked shape products, byte extents, and exact contiguous coverage at that point. QuaRot,
native Q4/KHF1, and Q8 readers retain their existing validation until their format adapters are
routed through the seam; this ADR does not claim those migrations have landed.

Each safetensors `TensorMeta` carries a `OnceLock<Result<(), String>>` validation marker separate
from its optional F16/BF16 conversion cache, populated through `get_or_init` rather than a
separate check-then-set. `get_or_init` runs the validation closure at most once even under
concurrent first access — every caller blocks on the same in-flight initialization instead of each
independently observing an empty marker and redoing the O(n) finite-value scan before one `set`
wins. Repeated access to an aligned zero-copy F32 tensor therefore does not rescan already-validated
memory-mapped pages, and a failed validation is cached as its error message (`validate_ingested_tensor`
only ever returns `InferenceError::InvalidSafetensors`) and never as success.

### Amendment — strict headers and bound single-file visual loads (2026-08-09)

The bounded header parser enforces the SafeTensors framing and JSON contract before any tensor is
materialized: an eight-byte little-endian header length capped at 4 MiB; a UTF-8 JSON object that
begins with `{`; unique keys at the top level and in every parsed or skipped object; tensor objects
with unambiguous `dtype`, `shape`, and `data_offsets`; an optional `__metadata__` string-to-string
map; canonical JSON-string decoding (so raw and escaped-equivalent Unicode keys compare equal);
strict RFC 8259 number grammar (including no leading-zero integers); and only ASCII-space padding
after the one complete top-level object. Trailing commas, non-space trailing bytes, malformed
surrogate pairs, duplicate names or members, malformed numbers, and arbitrary metadata values
fail closed. Structural parsing remains hand-written and bounded; the already-present `serde_json`
dependency is used only to decode one bounded JSON string token canonically, not to deserialize the
header tree or collapse object members into maps.

For Qwen3.5 vision checkpoints, an entry named `model.safetensors.index.json` is authoritative.
Presence is checked with `symlink_metadata`, so a malformed, unreadable, or dangling index fails
rather than falling back. Without an index, discovery accepts exactly one `*.safetensors`
candidate whose resolved metadata is a file and requires its name to be `model.safetensors`;
symlinks are followed consistently with the checkpoint mmap trust policy. Zero, multiple, or
misnamed candidates fail closed. `quantize_index.json` retains precedence in the lower-level
vision loader, but constructors with only an f16 decoder path reject it explicitly.

`resolve_qwen35_single_decoder_safetensors` exposes this structural policy as an unstable public
path-preflight surface. `open_qwen35_single_decoder_safetensors` resolves and opens the selected
file once; for an indexed layout it also requires the authoritative `weight_map` to match the open
shard's complete tensor-name inventory exactly. `load_qwen35_vision_weights_from_safetensors` is
the corresponding unstable already-open reader surface. A pooled vision constructor materializes
both visual and decoder tensors through that one reader. This binds the two components to one open
file description even if the directory entry is atomically replaced during a multi-gigabyte load,
and prevents direct-reader loading from bypassing indexed membership. These surfaces do not compute
a content digest; identity-governing callers remain responsible for optional checkpoint attestation
after structural preflight.

---

## Key Design Choices

1. **Hand-written structural JSON parser**: The safetensors header is bounded and structurally
   constrained. The parser retains explicit duplicate-key, depth, metadata-shape, padding, and
   extent checks without allocating a generic JSON tree. The already-present `serde_json`
   dependency decodes only individual bounded string tokens so Unicode and escape equivalence are
   canonical before duplicate checks.
2. **`OnceLock` conversion cache**: On first access to an F16/BF16 tensor, `get_or_init` performs the conversion and stores it. Subsequent accesses return the cached `Box<[f32]>` with no synchronization overhead (post-init reads are lock-free).
3. **Zero-copy F32**: `bytes_to_f32_slice` returns `None` on misalignment and the caller falls back to a copy. In practice, safetensors aligns tensors to 8 bytes, which satisfies f32's 4-byte alignment requirement on all target platforms.
4. **Eager QKV fusion**: Fusing at load time rather than inference time means the fusion cost is paid once and inference hot paths see a single contiguous weight matrix. The cost is ~30% higher peak memory during the load window (both original and fused forms exist briefly).
5. **`unsafe` 'static lifetime extension**: `ShardedQwenBacking` is heap-allocated via `Box` and co-located with `QwenModel`. The tensor slices that borrow from it are given `'static` lifetime via `mem::transmute`. Safety is maintained by RFC 1857 drop ordering (fields drop in declaration order; the backing box is declared before the tensor slices). This is documented as a known unsafe pattern (57.5K LOC crate note in `src/lib.rs`).
6. **Validate before exposure**: Format readers converge on `weights::ingress` immediately before
   tensor data becomes trusted. Public model constructors only assemble `Self` after all required
   tensors have loaded successfully.
7. **Cache successful validation per tensor**: The validation marker prevents repeated scans of
   zero-copy pages without allowing a failed scan to become sticky success.

---

## Alternatives Considered

| Alternative                               | Pros                                | Cons                                                                                               | Why Not                                                               |
| ----------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Full-tree `serde_json` header parsing     | Less structural parser code         | Default maps collapse duplicate keys; a generic tree allocates more and obscures admission order   | Explicit bounded duplicate/metadata/layout checks stay reviewable     |
| `candle` weight loading primitives        | Battle-tested; type-safe tensor API | Pulls in the entire `candle` ecosystem; incompatible with the crate's no-ONNX/no-Python constraint | Defeats the purpose of a pure Rust inference engine                   |
| Copy all weights at load time             | Simpler memory model; no `unsafe`   | 1–4 GB of unnecessary allocations; slower startup; more GC pressure                                | Latency and memory budget                                             |
| Load shards eagerly                       | Simpler code                        | All shard file descriptors held open simultaneously; larger virtual memory footprint               | Resource usage                                                        |
| Lazy QKV fusion (fuse at first GEMM call) | Lower peak memory at load time      | Fusion must be re-done or cached; complicates hot path                                             | Inference is called millions of times; amortize the cost once at load |

---

## Consequences

**Positive**:

- F32 model startup is near-zero-copy: the OS maps pages on demand.
- No generic JSON-tree allocation on the checkpoint header path.
- F16/BF16 conversion cost is paid at most once per tensor per process lifetime.
- Finite-value validation cost is paid at most once per successfully accessed tensor.
- Single-GEMM QKV path reduces per-layer arithmetic from 5 matmuls to 3.

**Negative**:

- 153 `unsafe` blocks in the crate (documented in `src/lib.rs`); the 'static extension is one contributor.
- Hand-written JSON parser must be maintained if the safetensors spec evolves.
- Peak load-time memory is ~1.3× final steady-state memory (QKV fusion window).

**Risks**:

- Drop order dependency on RFC 1857 is an invariant that must be preserved. A future refactor that reorders struct fields or moves to `Arc` would invalidate the safety argument. Documented in `src/model/bert.rs`: "WARNING: Do NOT reorder the fields of BertModel."

---

## References

- `src/weights/f32_weights.rs` — `SafetensorsFile`, `JsonParser`, `TensorMeta`, `ShardedSafetensors`
- `src/weights/ingress.rs` — private external-bytes-to-tensor validation seam
- `src/model/{bert.rs,qwen.rs,qwen35/model.rs}` — atomic public construction facades
- `src/model/qwen.rs` — `SafetensorsStorage`, `ShardedQwenBacking`, field drop order comment
- `src/lib.rs` — stability note on 153 unsafe blocks
- SafeTensors format spec — https://github.com/huggingface/safetensors
- RFC 1857 — https://rust-lang.github.io/rfcs/1857-stabilize-drop-order.html
