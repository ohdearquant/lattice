# Decode-step per-kernel GPU profile (#241)

Model: qwen3.5-0.8b (18 GDN linear-attention layers + 6 GQA full-attention layers).
Harness: `bench_decode_ab` warm decode, `LATTICE_PROFILE` (CPU-encode vs GPU) and
`LATTICE_PROFILE_GPU` (per-layer isolated GPU wall, added for #241). Apple Silicon, f16+metal-gpu.
Measured 2026-06-23. All numbers are warm steady-state (steps ~96-119).

## Gating result — decode is 99% GPU-bound

`LATTICE_PROFILE` decomposes the step into CPU-encode time per phase vs `other_us`
(the commit→`wait_until_completed` wall = GPU compute, since all 24 layers are one
command buffer with a single wait):

| | total/token | CPU-encode (embed+proj+gdn+gqa+mlp+final) | other_us (GPU) |
|---|---|---|---|
| f16 | ~6290 µs (159 tok/s) | ~37 µs (0.5%) | ~6250 µs (99.5%) |
| Q4  | ~4230 µs (236 tok/s) | ~37 µs (0.9%) | ~4190 µs (99.1%) |

**The lever is GPU kernel time, not CPU dispatch overhead.** (Consistent with the
prior prefill finding that GDN dispatch amortization is ~0%.) Q4 is +49% over f16
at single-token decode here.

## Per-layer GPU attribution (`LATTICE_PROFILE_GPU`)

Each layer is run in its own command buffer with a host-wall around commit+wait, so
every per-layer number carries ~1 barrier (~180 µs, derived from isolated-sum
~11.6 ms vs fused ~6.9 ms over 25 flushes). Relative attribution is robust; absolute
per-layer values are barrier-inflated.

| Phase | f16 µs/layer | Q4 µs/layer | Q4 effect | f16 share | Q4 share |
|---|---|---|---|---|---|
| GDN layer (×18) | 375 | 319 | −15% | 58% | ~63% |
| GQA layer (×6)  | 542 | 489 | −10% | 28% | ~32% |
| lm_head (×1)    | 1561 | 524 | **−3.0×** | 13% | ~5% |

## Findings (refines the prior decode-bandwidth note)

1. **No single kernel dominates.** f16: GDN-block ~50%, GQA-block ~31%, lm_head ~20%
   (after barrier correction). It is a distributed cost.
2. **Per-layer, a GQA layer is ~1.85× heavier than a GDN layer** (true ~362 vs ~195 µs).
   GDN dominates the step only by 3× layer count, NOT by per-layer cost. The naive
   "GDN scan is the expensive kernel" framing is wrong at the per-layer level.
3. **lm_head is pure weight-bandwidth** — Q4 collapses it 3× (1561→524 µs). In the
   shipping Q4 default it is already handled (~5%), a non-lever. In f16 it is the single
   largest kernel (~20%) and a clean Q4/Q8-lm_head lever if f16 decode matters.
4. **GDN and GQA layers are NOT weight-GEMV-bound** — Q4 only shaves 10-15%, vs the 3×
   it gives lm_head. So within those layers the dominant cost is quantization-immune:
   the GDN recurrence scan + conv1d (state ops), and GQA `decode_attention` over the KV
   cache (+ softmax). Their projection GEMVs are a minority of the layer.

## Conclusion — highest-value Q4-decode lever

For the shipping **Q4 path (236 tok/s)** the breakdown is ~GDN 53% / GQA 40% / head 7%.
The GDN recurrence path (18 layers, ~53%, quantization-immune) is the top target →
**#174 (GDN single-pass decode kernel)**. GQA `decode_attention` (~40%) is the second.

Open follow-up (gated first step of the attack task): a *direct* within-GDN-layer GPU
split (recurrence-scan vs conv1d vs in/out-projection) to quantify how much of the GDN
layer is the scan specifically. The Q4-immunity evidence above is strong but indirect;
this requires refactoring `encode_gdn_layer` to flush mid-function (it currently receives
only `&enc`, not the queue).
