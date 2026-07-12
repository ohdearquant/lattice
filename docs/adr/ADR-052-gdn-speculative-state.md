# ADR-052: GDN State Management for Speculative Rollback

**Status**: Accepted
**Date**: 2026-05-19
**Crate**: `lattice-inference`

---

## Context

Qwen3.5-2B's 24 transformer layers split into 18 GatedDeltaNet (GDN) linear-attention layers and
6 GQA full-attention layers (`[linear, linear, linear, full] × 6`, `full_attention_interval = 4`).
ADR-050 adds strict rejection sampling to the MTP speculative decoding path, which requires the
system to roll back model state when a draft token is rejected.

**KV cache rollback is already solved.** `FlatKVCache::truncate_to()` resets the GQA layers'
KV cache to any prior sequence length in O(1). `MtpTargetVerifier::rollback_cache_to()` calls
this on both the MTP verifier cache and the main model cache.

**GDN state rollback is not solved.** `GatedDeltaNetState` (`gdn.rs` line 64) holds two
mutable fields:

```rust
pub struct GatedDeltaNetState {
    pub s_matrices: Vec<f32>,   // num_heads * key_dim * value_dim floats
    pub conv_buffer: Vec<f32>,  // conv_dim * (kernel_size - 1) floats
    key_dim: usize,
    value_dim: usize,
}
```

For Qwen3.5-2B (`linear_num_key_heads = 16`, `linear_key_head_dim = 128`,
`linear_value_head_dim = 128`, `linear_conv_kernel_dim = 4`, `linear_qkv_dim = 6144`):

- `s_matrices`: `16 × 128 × 128 = 262,144` f32 values = **1 MB per GDN layer**
- `conv_buffer`: `6144 × 3 = 18,432` f32 values = **72 KB per GDN layer**
- Per-snapshot total (18 GDN layers): `18 × (1,048,576 + 73,728) bytes ≈ 20 MB`

Every GDN step (`gated_delta_net_step`) mutates both fields in place via the delta rule:

```
S = g * S + outer(k, (v - S @ k) * beta)
```

and via the causal conv1d rolling shift. Neither update is invertible. There is no
`truncate_to` analogue. The error bound under nonexpansive decay is `O(N * eps)` for N
spurious steps.

**Correctness gap.** When speculative decoding rejects a draft token at position `t`, the GDN
state has already been stepped through positions `t, t+1, ..., t+K-1` (all speculative). Without
a restore path, subsequent autoregressive generation runs from a corrupted state. This is silent:
generation continues with wrong recurrent memory. `rollback_cache_to` (speculative.rs line 559)
truncates only `FlatKVCache`; it does nothing for `gdn_states`.

ADR-050 scoped the GDN fix to `MtpTargetVerifier` trait extensions. This ADR specifies the
concrete design, memory layout, and integration contract that implements that scope.

---

## Decision

Add `snapshot()` and `restore_from()` to `GatedDeltaNetState`, extend `MtpTargetVerifier` with
GDN snapshot/restore methods, and integrate both into `mtp_verify_draft`.

The chosen strategy is **snapshot-before-draft**: take one snapshot of all GDN layers before the
speculative draft begins; restore from it on any rejection before running the correction token
forward pass. No per-token checkpoints. No recompute.

---

## Scope

Files changed:

- `crates/inference/src/attention/gdn.rs`: add `snapshot()`, `restore_from()` to
  `GatedDeltaNetState`; add `GdnSnapshot` type alias.
- `crates/inference/src/speculative.rs`: extend `MtpTargetVerifier` with
  `snapshot_gdn_states()` / `restore_gdn_states()`; add snapshot/restore calls to
  `mtp_verify_draft` around the draft generation and rejection branches.

No Metal kernels, no `NgramSpeculator`, no `verify_draft` (n-gram path), no generation.rs.

---

## Architecture

### New API on `GatedDeltaNetState`

```rust
/// Opaque snapshot of one GDN layer's recurrent state.
/// Layout: (s_matrices clone, conv_buffer clone).
pub type GdnLayerSnapshot = (Vec<f32>, Vec<f32>);

/// Snapshot of all GDN layers for a model instance.
pub type GdnSnapshot = Vec<GdnLayerSnapshot>;

impl GatedDeltaNetState {
    /// Clone s_matrices and conv_buffer into a new snapshot.
    /// Cost: O(num_heads * key_dim * value_dim + conv_dim * buf_len) per layer.
    pub fn snapshot(&self) -> GdnLayerSnapshot {
        (self.s_matrices.clone(), self.conv_buffer.clone())
    }

    /// Overwrite s_matrices and conv_buffer from a prior snapshot.
    /// Panics in debug if lengths do not match (invariant: snapshot taken from same config).
    pub fn restore_from(&mut self, snap: &GdnLayerSnapshot) {
        debug_assert_eq!(self.s_matrices.len(), snap.0.len());
        debug_assert_eq!(self.conv_buffer.len(), snap.1.len());
        self.s_matrices.copy_from_slice(&snap.0);
        self.conv_buffer.copy_from_slice(&snap.1);
    }
}
```

### `MtpTargetVerifier` trait extension

```rust
pub trait MtpTargetVerifier {
    fn cache_position(&self) -> usize;
    fn rollback_cache_to(&mut self, seq_len: usize) -> Result<(), InferenceError>;
    fn verify_tokens(&mut self, tokens: &[u32], start_pos: usize)
        -> Result<Vec<Vec<f32>>, InferenceError>;

    // New methods — added by this ADR:
    fn snapshot_gdn_states(&self) -> GdnSnapshot;
    fn restore_gdn_states(&mut self, snapshot: &GdnSnapshot);
}
```

Implementors hold a `Vec<GatedDeltaNetState>` (one per GDN layer). `snapshot_gdn_states`
calls `.snapshot()` on each, collects into a `Vec`. `restore_gdn_states` calls `.restore_from()`
on each with the corresponding element.

### Integration in `mtp_verify_draft`

```
before draft generation:
    let gdn_snap = target.snapshot_gdn_states();

acceptance loop (per draft token i):
    if reject:
        target.rollback_cache_to(target_start);
        target.restore_gdn_states(&gdn_snap);
        // then sample correction token from p'
        break

after full acceptance:
    drop(gdn_snap);  // not needed
```

The snapshot is heap-allocated once per speculative step. Because draft tokens are never
committed to target GDN state until accepted, restore is only needed on the rejection branch.
For the full-acceptance path, the snapshot is simply dropped.

### Memory budget (Qwen3.5-2B)

| Item                          | Size                               |
| ----------------------------- | ---------------------------------- |
| `s_matrices` per GDN layer    | 16 × 128 × 128 × 4 B = 1,048,576 B |
| `conv_buffer` per GDN layer   | 6,144 × 3 × 4 B = 73,728 B         |
| Per-layer snapshot            | ~1.07 MB                           |
| Full snapshot (18 GDN layers) | ~19.3 MB                           |

19 MB is ephemeral (held for one speculative step). On Apple Silicon unified memory
(16–192 GB), this is acceptable. Snapshot allocation is paid once per step, not per token.

---

## Alternatives Considered

**A. Per-token checkpoint (K snapshots for K draft tokens).** Enables partial rollback: if
token 3 of 5 is rejected, only restore to checkpoint 2. Memory cost: K × 19 MB = 96 MB for
K=5. The correctness benefit over snapshot-before-draft is nil: both paths reach the same
restored state. Snapshot-before-draft is simpler and uses 1/K the memory.

**B. Recompute-from-accepted.** After rejection at position t, rerun the GDN forward pass
from the last accepted position using only accepted tokens. Zero extra memory, but adds
O(accepted\_tokens × num\_gdn\_layers) compute per rejection event. For long prefixes this is
equivalent to full prefill latency on rejection. Memory is cheap on Apple Silicon; compute
latency is not free. Snapshot-before-draft trades 19 MB for zero recompute.

**C. Do not expose rollback; run target model forward-only during speculative steps.** This
avoids the problem entirely by running the target model in a stateless (prefill) mode across
the draft window. Correct, but requires passing the full token sequence each step rather than
incremental decode — O(context\_length) work per speculative step, eliminating speedup for
long contexts.

---

## Risks

**R1: Snapshot size scales with model.** Qwen3.6-27B has 48 active GDN layers and
`linear_num_value_heads = 48`, giving `48 × (48 × 128 × 128 + 6144 × 3) × 4 B ≈ 153 MB` per
snapshot. Still within unified memory budget but worth tracking. Mitigation: the snapshot is
ephemeral. Document the per-model cost formula in the struct docstring.

**R2: `restore_from` panics on length mismatch.** This is intentional: a length mismatch means
the snapshot was taken from a different config instance, which is a programming error.
`debug_assert` is sufficient; production builds skip the check.

**R3: Trait change requires all `MtpTargetVerifier` implementors to add two methods.** The
only extant implementor is the integration test mock at speculative.rs line 1771. One change
required; not a public API break (crate is `#[doc(hidden)]` for these types).

**R4: Draft does NOT advance target GDN state.** The integration design above depends on
`mtp_verify_draft` not mutating target GDN state during draft generation. This must be
verified before implementation: the MTP verifier (`MtpVerifier`) has its own separate state
and cache; it does not share `gdn_states` with the target model. Confirm at implementation
time.

---

## References

- `crates/inference/src/attention/gdn.rs` lines 64–106: `GatedDeltaNetState` struct and
  `reset()` method (existing cleanup pattern that `snapshot/restore` extends).
- `crates/inference/src/speculative.rs` line 555–561: `MtpVerifier::rollback_cache_to()`
  and the GQA-only `FlatKVCache::truncate_to()` call this ADR parallels for GDN.
- `crates/inference/src/speculative.rs` line 1075–1087: `MtpTargetVerifier` trait (the trait
  this ADR extends).
- ADR-006: Speculative Decoding — baseline speculative decode architecture.
- ADR-050: Rejection Sampling — strict speculative sampling; scoped the GDN gap that this
  ADR resolves.
- Error bound `O(N * eps)` under nonexpansive decay, motivating why skipping rollback is not
  a bounded-error approximation but unbounded state corruption.
