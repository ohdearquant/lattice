# ADR-008: LoRA Injection via Trait Hook

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

---

## Context

Low-Rank Adaptation (LoRA) fine-tuning adapts a base model by adding pairs of low-rank matrices (A and B) to selected projection layers. At inference time, the adapter contribution `B @ A @ x` is added to the base layer output. The crate needs to support LoRA adapters without:

1. Modifying the base weight matrices (adapters must be separable from base weights).
2. Adding overhead to the common case (no adapter loaded).
3. Creating a circular dependency: the adapter implementation lives in `platform/tune`, which must not depend on `lattice-inference`. If the hook is defined in `lattice-inference`, `platform/tune` can implement it; if it were defined in `platform/tune`, `lattice-inference` would have to depend on `platform/tune`.

Relevant implementation: `src/lora_hook.rs`.

```rust
pub trait LoraHook: Send + Sync {
    fn apply(
        &self,
        layer_idx: usize,
        module: &str,
        x: &[f32],
        output: &mut [f32],
    );
}

pub struct NoopLoraHook;

impl LoraHook for NoopLoraHook {
    #[inline(always)]
    fn apply(&self, _: usize, _: &str, _: &[f32], _: &mut [f32]) {}
}
```

Module name strings used as identifiers:

- GQA attention: `"q_proj"`, `"k_proj"`, `"v_proj"`, `"o_proj"`
- GatedDeltaNet (Qwen3.5): `"in_proj_qkv"`, `"in_proj_z"`, `"in_proj_b"`, `"in_proj_a"`, `"out_proj"`
- FFN: `"gate_proj"`, `"up_proj"`, `"down_proj"`

The `apply()` call is placed in the forward pass after the base projection weight multiply and before the result is consumed downstream. The hook receives the input activations `x`, the output buffer `output` (already written by the base layer), and can add its delta in-place.

---

## Decision

Define `LoraHook` as a **`Send + Sync` trait in `lattice-inference`** with a `NoopLoraHook` zero-cost default implementation. The adapter implementation lives in `platform/tune`, which depends on `lattice-inference` (not the reverse). The forward pass calls `hook.apply(layer_idx, module, x, output)` for each projection site; the noop implementation is `#[inline(always)]` with an empty body, allowing the compiler to eliminate all calls when no adapter is loaded.

---

## Key Design Choices

1. **Trait in `lattice-inference`, not in `platform/tune`**: The dependency arrow must point from `platform/tune` → `lattice-inference`. If the trait were defined in `platform/tune`, the forward pass would need to import from `platform/tune`, creating an upward dependency. Defining the trait in the inference kernel preserves the correct layer separation.
2. **`NoopLoraHook` with `#[inline(always)]`**: LLVM sees the empty body at the call site when the concrete type is `NoopLoraHook`, inlines the empty function, and eliminates the call entirely. Zero runtime overhead when no adapter is loaded — confirmed by inspection of release builds (no call instruction generated for noop sites).
3. **String module names instead of enum**: An enum of module positions would require modifying `lora_hook.rs` whenever a new architecture adds a new projection. String names are open: `platform/tune` can handle any module name without a change to the kernel crate.
4. **Input activations passed to `apply()`**: Some adapter schemes require the input (e.g., `B @ A @ x` where the rank decomposition is applied to the input). Others require only the output. Passing both allows both schemes without an API change.
5. **`&mut [f32]` output**: The hook adds to the existing output buffer in-place. This avoids an allocation and allows the compiler to recognize it as an accumulation pattern.

---

## Alternatives Considered

| Alternative                                            | Pros                                          | Cons                                                                                         | Why Not                                                                        |
| ------------------------------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Merge LoRA weights into base weights at load time      | Zero inference overhead; one matmul per layer | Cannot hot-swap adapters; cannot serve multiple users with different adapters simultaneously | Adapter flexibility lost; multi-tenant serving impossible                      |
| Function pointer instead of trait                      | One indirection; slightly simpler             | Not `Send + Sync` by default; cannot carry state (closures with captures need Box<dyn Fn>)   | Trait objects are the idiomatic Rust pattern for stateful polymorphic dispatch |
| `Option<Box<dyn LoraHook>>` with `if let Some` guard   | Avoids vtable when `None`                     | Branch on hot path; harder to monomorphize; adds branch predictor pressure                   | `NoopLoraHook` inlining achieves zero overhead without a branch                |
| Define trait in a separate `lattice-lora-api` crate    | Clean dependency graph                        | Adds a third crate for a single trait                                                        | Single-trait crate is premature until a second consumer appears                |
| Pass adapters as weight tensors alongside base weights | Familiar to ML engineers                      | Requires forward pass to conditionally add rank decompositions; complicates weight struct    | Trait hook cleanly separates base model from adapter logic                     |

---

## Consequences

**Positive**:

- Zero overhead in production builds with `NoopLoraHook` (no branch, no call, no memory access).
- `platform/tune` can implement `LoraHook` with any rank decomposition without modifying the kernel.
- Multiple adapters can be composed by chaining hooks.
- The string-based module API is forward-compatible with new architectures.

**Negative**:

- String module names are stringly-typed: a typo in `platform/tune` (`"q_proj_"` instead of `"q_proj"`) will silently produce wrong results (the hook is called but matches no projection site, or matches the wrong one).
- No schema validation of which module names are valid for a given architecture.

**Risks**:

- If a future architecture renames a projection (e.g., `"qkv_proj"` instead of separate `"q_proj"`, `"k_proj"`, `"v_proj"`), existing adapter implementations targeting the old names will silently no-op rather than error.
- The `apply()` signature is not versioned. Adding a parameter (e.g., sequence position for position-dependent adapters) requires updating all implementors.

---

## References

- `src/lora_hook.rs` — `LoraHook` trait, `NoopLoraHook`, module name constants
- `src/forward/` — call sites for `hook.apply()` in GQA and FFN forward passes
- Hu et al. 2021 — "LoRA: Low-Rank Adaptation of Large Language Models" — https://arxiv.org/abs/2106.09685
