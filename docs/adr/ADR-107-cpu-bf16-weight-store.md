# ADR-107: An opt-in bf16 weight store for the Qwen3.5 CPU path

**Status**: Proposed\
**Date**: 2026-11-26\
**Crates**: lattice-compute, lattice-model-qwen35, lattice-serve\
**Depends on**: ADR-098

## Context

The Qwen3.5 CPU path holds every weight as f32. When a checkpoint stores its matrices as BF16, the loader widens each
one at load, and the resident matrices take twice their on-disk bytes. The CPU host that serves `/v1/systemone`
(ADR-098) then holds twice the bytes those matrices take on disk.

A bf16 value widens to f32 exactly: its 16 bits become the top half of the f32 word. A matrix can therefore be held as
its on-disk bits and widened when it is used, without changing a single value the f32 path computes with. What it
costs is the widening work on every product, and that cost has to be measured before anyone is asked to pay it.

## Decision

1. **The store is a load option, and the default is f32.** `CpuWeightStore { F32, Bf16 }` is passed through
   `Qwen35LoadOptions::with_cpu_weight_store`. `Qwen35Model::from_safetensors` and every existing constructor load
   with `F32`, so no existing caller changes behaviour. The serving host exposes the choice as
   `--cpu-weight-store f32|bf16`.
2. **`Bf16` holds a matrix as bf16 only when the checkpoint stores it as BF16.** The covered matrices are the token
   embedding, the untied output head, the full-attention q/k/v/o projections, the GatedDeltaNet `in_proj_qkv`,
   `in_proj_z` and `out_proj` projections, and the dense FFN gate/up/down projections. A matrix stored as F32 or F16,
   and every other tensor (the norms, and the GatedDeltaNet `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias` and
   convolution tensors), is loaded as f32 as before. A checkpoint with no BF16 matrix loads
   entirely as f32 under `Bf16`, and the load report says so.
3. **A product with a bf16 matrix widens it in tiles of 1,024 rows into reused f32 scratch and runs the existing f32
   product on each tile.** The tile size is the one constant `BF16_TILE_ROWS`. The scratch is owned by the caller's
   forward scratch, so steady-state scoring allocates nothing for it. There is no separate bf16 arithmetic: every
   multiply-add is the f32 path's.
4. **Mixture-of-experts checkpoints refuse `Bf16`.** The expert path has not been measured with this store, so the
   load fails with an error that names the f32 store instead of loading something unmeasured.
5. **The load reports what it held.** `Qwen35Model::weight_load_report` counts the tensors read by on-disk dtype and
   by the form they are held in, as one line of the form
   `cpu weight store bf16: <n> BF16 held as f32, <m> BF16 held as bf16`. The serving host prints this line at
   startup, so a run's memory figure can be traced to the store that produced it.
6. **Training needs the f32 store.** The training accessors return `None` for a layer whose matrices are held as
   bf16, and the LoRA and full-depth drivers refuse such a layer with an error that names the f32 store.
7. **Acceptance is the ADR-098 decision 7 bound, not bit identity.** Before the serving flag merges, decider-0.8b is
   scored through `/v1/systemone` with `--cpu-weight-store bf16` on the benchmark host and must meet the same bounds
   the f32 store met. The unit tests assert a tolerance between the two stores on the same values and print whether
   the outputs were bit-identical as an observation; identity depends on the matrix library's blocking for a given
   shape and is not promised.
8. **The time cost is reported from a paired measurement.** `bench_systemone_scoring` runs with both stores on the
   same host in base, head, head, base order, and the figures are quoted beside the memory figure. A prototype on one
   Apple silicon host measured the bf16 store at 1.05 to 1.6 times the f32 matmul time for 512-row prefill products
   and about 2 times for the full-vocabulary output product; that prototype is not a paired run and is not the
   number this ADR rests on.

## Alternatives considered

- **Make bf16 the default.** Rejected until decision 7 has passed and decision 8 has been measured: the default
  should not trade time for memory on a user's behalf before the trade is known.
- **Hold the matrices as f16.** f16 cannot represent the bf16 exponent range, so values would change. bf16 bits
  widen exactly.
- **Widen the whole matrix for each product.** This keeps the f32 product but needs a full-size f32 buffer as large
  as the largest matrix, which gives that matrix's saving back on every call.
- **A native bf16 product kernel.** Possible later. A new kernel needs its own parity and bench work, while the tile
  widening keeps the existing f32 product as the only arithmetic.

## Consequences

- The covered matrices of a BF16 checkpoint take half their f32 memory under `Bf16`. Memory estimates for the CPU
  path must count a BF16 matrix at two bytes per value under `Bf16` and four under `F32`; the serving host's preflight
  estimate takes the store into account.
- Every product with a bf16 matrix pays the widening work. Decision 8 records how much.
- The other backends built from a model loaded under `Bf16` read each matrix through `to_f32()`: the Metal engine's
  packers and the Q8 and NEON quantizers. They keep working, widen each bf16 matrix once when they pack it, and hold the
  packed copy beside the model's bf16 matrices. The memory saving is the f32 CPU forward's alone.
- `WeightMatrix` replaces `Vec<f32>` as the field type of the matrices it covers. Callers that read a matrix as a
  slice use `as_f32()` (f32 store only) or `to_f32()` (either store).
