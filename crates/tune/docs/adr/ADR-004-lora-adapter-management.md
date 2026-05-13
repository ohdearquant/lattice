# ADR-004: LoRA Adapter Management

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-tune

## Context

`lattice-tune` must load LoRA adapters trained by external tools (Python PEFT, MLX) and apply them during inference to modify the output of specific linear projections in a transformer model. The workflow is: Python/PEFT trains the adapter and exports to safetensors → Rust loads the file → adapter is applied in the inference forward pass via a hook interface.

The adapter format is defined by PEFT (HuggingFace): tensor keys follow the pattern `base_model.model.model.layers.{i}.self_attn.{module}.lora_{A|B}.weight`. MLX uses a variant with lowercase `lora_{a|b}` and transposed matrix layouts. The loader must handle both formats.

LoRA math: `dW = B @ A` where `A: (rank, d_in)` and `B: (d_out, rank)`. Applied as `output += (alpha / rank) * B @ (A @ x)`.

## Decision

Three-layer design: `LoraConfig` (global config), `LoraLayer` (per-projection weights), `LoraAdapter` (collection of layers indexed by `(layer_idx, module_name)`). Loading from safetensors is behind `feature = "safetensors"`. Integration with `lattice-inference` is behind `feature = "inference-hook"`.

### Key Design Choices

#### LoraConfig and Scaling

`LoraConfig` holds `rank: usize`, `alpha: f32`, and `target_modules: Vec<String>`. The effective scaling factor is `alpha / rank`. When loading from PEFT safetensors, `alpha` defaults to `rank` (so `scale = 1.0`), matching PEFT's default behavior. Callers can override `alpha` post-load for explicit scaling.

`LoraConfig::scale()` guards against `rank == 0` by returning `0.0` rather than dividing by zero.

#### LoraLayer Layout

A/B matrices stored as row-major `Vec<f32>`:

- `a: Vec<f32>` — shape `(rank, d_in)`, row-major
- `b: Vec<f32>` — shape `(d_out, rank)`, row-major

This layout is what PEFT produces. MLX stores them transposed (`A: (d_in, rank)`, `B: (rank, d_out)`); the loader transposes on load.

#### LoraAdapter Indexing

`HashMap<(usize, String), LoraLayer>` where the key is `(layer_idx, module_name)`. Module names for Qwen3.5-2B and similar architectures: attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP (`gate_proj`, `up_proj`, `down_proj`). The `apply` method is a no-op for any `(layer_idx, module)` pair not present in the map.

#### PEFT Safetensors Loader

`load_peft_safetensors(path: &Path)` in `src/lora/safetensors.rs`:

1. File size check: rejects files > 10 GiB (`MAX_LORA_SIZE = 10 GB`) to prevent OOM from adversarial inputs
2. Reads entire file to `Vec<u8>`, deserializes with `safetensors::SafeTensors::deserialize`
3. Iterates all tensor names, calling `parse_peft_key` to extract `(layer_idx, module, A|B, is_transposed)`
4. Reads each tensor via `read_tensor_f32` (handles F32, F16, BF16 dtypes; non-finite value check)
5. Transposes MLX-format tensors in-place before storing
6. Pairs A and B matrices by `(layer_idx, module)` key; errors on orphaned A without B or B without A
7. Validates rank consistency across all layers (all layers must use the same rank)
8. Shape product overflow checked via `checked_mul` for each A and B size calculation
9. Data size verification: actual tensor element count must match `shape[0] * shape[1]`

`parse_peft_key` handles both PEFT (`base_model.model.model.layers.{i}.{block}.{module}.lora_{A|B}.weight`) and simple HuggingFace format (`model.layers.{i}.{block}.{module}.lora_{A|B}.weight`), and MLX lowercase (`lora_{a|b}` without `.weight` suffix). Returns `None` for non-LoRA keys (metadata tensors, etc.).

#### LoraHook Integration (feature `inference-hook`)

When compiled with `inference-hook`, `LoraAdapter` implements `lattice_inference::lora_hook::LoraHook`:

```rust
fn apply(&self, layer_idx: usize, module: &str, x: &[f32], output: &mut [f32])
```

This delegates to `LoraAdapter::apply`. The inference forward pass calls `hook.apply(layer_idx, module, x, output)` after the base projection, injecting the LoRA delta without modifying the base model weights.

### Alternatives Considered

| Alternative                                    | Pros                          | Cons                                                                 | Why Not                                                                                               |
| ---------------------------------------------- | ----------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Load adapter lazily on first use               | Avoids upfront file I/O       | Per-request disk access or caching complexity                        | Adapters are small enough to load entirely at startup; latency budget for inference cannot absorb I/O |
| Store A and B as `Vec<Vec<f32>>` (row-of-rows) | More readable indexing        | Extra indirection, harder to pass to BLAS/SIMD routines              | Flat row-major layout matches what SIMD and potential BLAS calls expect                               |
| `candle` tensor type for A/B storage           | Automatic GPU tensor support  | Heavy dependency, ties to candle's tensor lifecycle                  | ADR-001 mandates no external ML framework                                                             |
| Separate `alpha` per layer                     | Per-layer scaling flexibility | PEFT does not export per-layer alpha; would require format extension | PEFT convention: single alpha for the entire adapter                                                  |

## Consequences

### Positive

- `apply` is a no-op for missing layers — the forward pass does not need to check `has_adapter` before calling `apply`
- MLX format supported via transposition on load, eliminating a source of silent correctness bugs when porting adapters from Apple Silicon training setups
- Non-finite value check on load prevents silent propagation of NaN/Inf weights into inference
- `num_parameters()` enables logging/monitoring adapter size without manually summing A and B dimensions
- 10 GiB file size guard prevents memory exhaustion from adversarial safetensors files

### Negative

- Full file read into memory (`std::fs::read`) is required before parsing. For adapters larger than available RAM (unlikely but possible for full-model ranks), this will OOM rather than stream
- Rank consistency is enforced globally — a safetensors file with different ranks per layer will fail. Mixed-rank adapters (unusual but possible with custom PEFT configs) are not supported
- `LoraAdapter::apply` performs two sequential matrix-vector multiplications in scalar code (`apply_lora` in `src/lora/apply.rs`). No SIMD acceleration is currently present in the apply path.

### Risks

- The `inference-hook` feature creates a compile-time dependency on `lattice-inference`. If `lattice-inference`'s `LoraHook` trait changes signature, `lattice-tune` must update. The trait boundary should be stabilized before the feature is enabled in production builds.
- Rank validation enforces all layers use the same rank (extracted from the first A matrix encountered). If PEFT ever exports mixed-rank adapters, existing validation will reject them with an error that may be confusing ("inconsistent LoRA ranks").

## References

- `crates/tune/src/lora/mod.rs` — `LoraConfig`, `LoraLayer`, `LoraAdapter`, `apply`, `inference-hook` impl
- `crates/tune/src/lora/safetensors.rs` — `load_peft_safetensors`, `parse_peft_key`, `read_tensor_f32`, MLX transpose
- `crates/tune/src/lora/apply.rs` — `apply_lora` (A@x then B@(A@x) with scale)
- `crates/tune/src/bin/generate_lora.rs` — CLI for generating LoRA adapters
