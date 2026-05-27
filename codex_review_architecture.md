Verdict: REQUEST CHANGES
Findings: 4 High, 3 Medium

## Summary

ADR-059 and ADR-060 are directionally sound: enum-dispatched layer composition is a reasonable fit for Lattice's closed set of kernels, ShortGPT's BI formula matches the paper, and the SliceGPT -> QuaRot order is the right order. The current text still has several contract defects that would mislead implementation, especially around MHA state, the "gated attention" taxonomy, SliceGPT slicing axes, and the Hadamard dimension constraints after slicing.

## Findings

### High: MHA is classified as KV-cache attention even though the code path is scratch-buffer based

Evidence: `docs/adr/ADR-059-composable-layer-architecture.md:103` says softmax attention includes MHA and `docs/adr/ADR-059-composable-layer-architecture.md:104` says state is "always a KV cache variant"; `docs/adr/ADR-059-composable-layer-architecture.md:167` separately says `None` is for "MHA with recomputed attention"; `docs/adr/ADR-059-composable-layer-architecture.md:235` then allocates `AttentionStateKind::Kv` for `Self::Mha`. The current MHA code exposes `AttentionBuffers` at `crates/inference/src/attention/standard.rs:5` and `multi_head_attention_in_place(..., buffers: &mut AttentionBuffers, ...)` at `crates/inference/src/attention/standard.rs:84`, with no KV-cache parameter. ADR-010 describes standard MHA as BERT encoder attention with pre-allocated `AttentionBuffers` at `docs/adr/ADR-010-attention-mechanisms.md:13`.

Why this matters: This is an internal contradiction and a code/spec mismatch. If implemented literally, BERT-style MHA either gets forced through a nonexistent KV-cache adapter or validation rejects valid encoder attention because `validate_model_spec` later says softmax attention without KV cache is rejected.

Suggested fix: Split the category contract into cached causal softmax attention and uncached/materialized encoder attention, or keep one softmax trait but allow `AttentionStateKind::None` plus `SoftmaxScratch` for MHA. Update `alloc_state`, validation, and the state table consistently.

### High: `GatedAttention` is counted as a standalone attention variant, but the code only has gating primitives

Evidence: ADR-059 counts `gated.rs` as `GatedAttention` in the 10-module inventory at `docs/adr/ADR-059-composable-layer-architecture.md:23`, includes `GatedAttention(GatedAttention)` in `AttentionKind` at `docs/adr/ADR-059-composable-layer-architecture.md:201`, and includes `GatedAttentionSpec` in `AttentionSpec` at `docs/adr/ADR-059-composable-layer-architecture.md:634`. The code file says it contains "Gated attention primitives" at `crates/inference/src/attention/gated.rs:1`, exports `deinterleave_q_gate` at `crates/inference/src/attention/gated.rs:30` and `apply_sigmoid_gate` at `crates/inference/src/attention/gated.rs:68`, and the actual Qwen3.5 full-attention path applies the gate after computing attention context at `crates/inference/src/model/qwen35/forward.rs:202` and `crates/inference/src/model/qwen35/forward.rs:214`.

Why this matters: The enum captures 10 files, but not 10 independently dispatchable attention algorithms. Treating the gate as an `AttentionKind` will either duplicate work already inside Qwen3.5 full attention or create a variant that cannot own projection, KV cache, RoPE, and output projection semantics.

Suggested fix: Make `gated.rs` a component/helper under a `Qwen35GatedGqa` or `FullGqaWithGate` wrapper, not its own `AttentionKind`. If the intended variant is a full gated SDPA implementation, specify that wrapper explicitly and distinguish it from the current primitive module.

### High: SliceGPT keeps the wrong side of the PCA-sorted residual dimensions

Evidence: ADR-060 sorts PCA eigenvectors by descending eigenvalue at `docs/adr/ADR-060-pruning-toolbox.md:270`, then says "Keep only the top d' residual dimensions" at `docs/adr/ADR-060-pruning-toolbox.md:281`, but immediately says input-reading tensors should "keep last d' columns" at `docs/adr/ADR-060-pruning-toolbox.md:283`. SliceGPT's paper computes `Q_l` as eigenvectors sorted by decreasing eigenvalues and then deletes minor components; see the SliceGPT ICLR PDF, Section 3.3-3.4: https://spcl.inf.ethz.ch/Publications/.pdf/2024_iclr_sliceGPT.pdf.

Why this matters: With eigenvectors ordered high-to-low, the top residual coordinates are the first coordinates after rotation. Keeping the last `d'` columns would retain the lowest-energy PCA directions and discard the signal SliceGPT is trying to preserve.

Suggested fix: Change Step 6 to keep the first `d'` residual dimensions after descending sort. If the implementation chooses to store low-energy dimensions first, say that explicitly and update Step 4 so the ordering contract is not contradictory.

### High: The SliceGPT -> QuaRot path advertises reduced widths that current QuaRot cannot rotate

Evidence: ADR-060 applies "QuaRot randomized Hadamard rotation in the reduced d' space" at `docs/adr/ADR-060-pruning-toolbox.md:364`, then claims `d'=896` and `d'=768` are "power-of-two-friendly" for Qwen3.5 at `docs/adr/ADR-060-pruning-toolbox.md:371`. Current QuaRot constructs `RandomizedHadamard` only when `n.is_power_of_two()` at `crates/inference/src/quant/quarot/hadamard.rs:139`, and errors otherwise at `crates/inference/src/quant/quarot/hadamard.rs:141`. ADR-044 also says non-power-of-two hidden dimensions are deferred and naive padding is not orthogonal at `docs/adr/ADR-044-quarot-rotated-quantization.md:77`.

Why this matters: The composition order is mathematically correct, but the advertised Qwen3.5 slice ratios cannot use the existing QuaRot Hadamard backend. `896` and `768` are multiples of powers of two, not powers of two themselves. Without a block-diagonal Hadamard implementation, the pruned+QuaRot stage fails at construction.

Suggested fix: Either restrict the first composed pipeline to widths supported by `RandomizedHadamard`, or make P9/P12 explicitly depend on a real `BlockHadamard` basis and update the examples to say they are block-Hadamard-compatible, not power-of-two-compatible.

### Medium: `OrthogonalBasis` does not line up with the existing QuaRot API contract

Evidence: ADR-060 proposes `OrthogonalBasis` with `fn apply_right(&self, x: &mut Tensor)` and no error return at `docs/adr/ADR-060-pruning-toolbox.md:325`, plus `Tensor`/`LinearWeight`-based fuse methods at `docs/adr/ADR-060-pruning-toolbox.md:331`. Existing QuaRot exposes `RandomizedHadamard::apply` and `apply_inverse` returning `Result` with dimension checks at `crates/inference/src/quant/quarot/hadamard.rs:156` and `crates/inference/src/quant/quarot/hadamard.rs:174`, while rotation absorption is shape-explicit over row-major slices via `absorb_input_rotation_f64(... rows, cols, rotation)` at `crates/inference/src/quant/quarot/rotation.rs:146` and `absorb_output_rotation_f64` at `crates/inference/src/quant/quarot/rotation.rs:168`.

Why this matters: The proposed trait hides the checks that make QuaRot safe to use and introduces orientation names (`apply_right`, `apply_left_t`) that do not directly match the existing row-major absorption helpers. It is therefore not yet a compatible refactor target for ADR-044's `RandomizedHadamard`.

Suggested fix: Define the trait around the existing primitives: `dim`, `supports_dim`, `apply`, `apply_inverse`, `absorb_input_rotation_{f32,f64}`, and `absorb_output_rotation_{f32,f64}` returning `Result`. Then add adapters for dense PCA and Hadamard bases rather than replacing the current checked API with unchecked `Tensor` methods.

### Medium: ADR-060 still uses the two-variant `LayerType` after depending on ADR-059's composable layer model

Evidence: ADR-060's `LayerStats` stores `pub layer_type: LayerType` at `docs/adr/ADR-060-pruning-toolbox.md:57`, and `CalibrationObserver::observe_block_input` takes `layer_type: LayerType` at `docs/adr/ADR-060-pruning-toolbox.md:115`. ADR-059 says the current `LayerType` enum is superseded by `AttentionKind` at `docs/adr/ADR-059-composable-layer-architecture.md:887` and puts that replacement in P1 at `docs/adr/ADR-059-composable-layer-architecture.md:891`.

Why this matters: The pruning toolbox is supposed to analyze composable architectures, but its core calibration records only `LinearAttention` versus `FullAttention`. That loses the distinction between GQA, Flash, Differential, NSA, Decode, and future wrappers before scoring or constraints can use it.

Suggested fix: Replace `LayerType` in ADR-060 with `AttentionTag`, `LayerSpec`, or a compact `LayerDescriptor` derived from ADR-059's normalized model spec. Keep a Qwen3.5-only convenience adapter if needed, but do not bake the legacy two-variant enum into the generic calibration API.

### Medium: `ArchitectureEnv` should be split out of ADR-059

Evidence: ADR-059 introduces `ArchitectureEnv` and `ArchAction` at `docs/adr/ADR-059-composable-layer-architecture.md:752`, then immediately moves into random search, evolutionary search, Bayesian optimization, PPO, and agent search at `docs/adr/ADR-059-composable-layer-architecture.md:773`. The migration plan already treats it as P6 and links it to ADR-061 at `docs/adr/ADR-059-composable-layer-architecture.md:896`, while the core dependency order says P5 and P6 are independent at `docs/adr/ADR-059-composable-layer-architecture.md:898`.

Why this matters: D6 is not needed to decide the composable layer runtime. Keeping it in ADR-059 broadens the ADR from "how layers are represented and dispatched" into "how architecture search is run", which makes the decision harder to review and gives implementers an unnecessary second contract to satisfy.

Suggested fix: Move D6 into ADR-061 or a dedicated architecture-search ADR. ADR-059 should keep only the hooks and serializable search-space fields needed by the layer DSL, plus a cross-reference that search environments consume the normalized model spec later.

## What I Checked

- Read ADR-059, ADR-060, ADR-010, and ADR-044.
- Compared ADR-059's attention taxonomy against `crates/inference/src/attention/mod.rs:1` through `crates/inference/src/attention/mod.rs:10` and the individual attention modules.
- Checked current Qwen3.5 forward usage of gated attention primitives in `crates/inference/src/model/qwen35/forward.rs`.
- Checked QuaRot Hadamard and rotation absorption APIs in `crates/inference/src/quant/quarot/hadamard.rs` and `crates/inference/src/quant/quarot/rotation.rs`.
- Verified ShortGPT BI against the ACL paper: https://aclanthology.org/2025.findings-acl.1035.pdf.
- Verified SliceGPT's PCA/slicing ordering against the ICLR paper and repository: https://spcl.inf.ethz.ch/Publications/.pdf/2024_iclr_sliceGPT.pdf and https://github.com/microsoft/TransformerCompression.

## False Positives Ruled Out

- ShortGPT formula in ADR-060 is correct. `docs/adr/ADR-060-pruning-toolbox.md:160` matches the ShortGPT paper's BI formula, and `docs/adr/ADR-060-pruning-toolbox.md:163` correctly states that lower BI means less transformation and a stronger removal candidate.
- The high-level composition order "SliceGPT PCA first -> slice -> QuaRot Hadamard second" is correct. The issue is not the order; it is the slicing-axis text and the reduced-dimension Hadamard backend constraint.
- ADR-059 does enumerate the 10 files currently under `crates/inference/src/attention/`; the problem is that one of those files is a primitive module rather than a standalone attention operation.

## What I Did Not Check

- I did not run `cargo test` or `make ci`; this was a static architecture/spec review.
- I did not validate paper result numbers in ADR-060's method comparison tables beyond the ShortGPT formula and SliceGPT mechanism requested here.
- I did not review ADR-061 or later research ADRs that may already carry some of the search-environment details.

## Recommended Next Steps

1. Fix the ADR-059 attention taxonomy and state model before implementing `AttentionKind`.
2. Correct ADR-060's SliceGPT slicing axis and explicitly gate SliceGPT+QuaRot on a supported orthogonal basis for `d'`.
3. Refactor ADR-060's calibration types off legacy `LayerType`.
4. Split `ArchitectureEnv` out of ADR-059 into the architecture-search ADR, leaving ADR-059 focused on layer representation and dispatch.

Re-review guidance: another focused pass is useful after these text changes because several findings touch closed taxonomies and implementation signatures.

Domain utility: SKIPPED - the requested `mcp__lore__suggest` / `compose` tools were not available in this session, so I used the local ADR/code review rubric and primary paper sources instead.
