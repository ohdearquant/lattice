# W3 3-bit MLP-Only Weight Path — Design & Review Surface (issue #420)

act_type: propose

**Status: REVIEW SURFACE — FOUNDER-GATED, DO NOT MERGE.**
This branch delivers a *tested CPU W3 file-format component* plus a *complete
design* for the Metal decode path. It does **not** deliver a runnable W3
inference path, and it carries an **unmeasured** silent-quality-loss risk. See
[§ SILENT QUALITY LOSS RISK](#silent-quality-loss-risk).

Branch: `feat/w3-mlp-420` (off `main @ 36d9bf68c`).
Upstream artifacts folded in: `../architect/design.md` (full design),
`../implementer/impl_report.md` (what was built), `../tester/test_report.md`
(test evidence), `../analyst/measurements.md` (measured PPL/decode).

---

## 1. Goal

W3 is the cheapest sub-4-bit decode byte-read reduction and the #1 confirmed
live decode lever: decode is weight-bandwidth-bound and dense MLP weights
(`gate_proj`, `up_proj`, `down_proj`) dominate the per-step bytes-read. Quantize
**only** those three dense MLP projections to 3-bit; keep attention, GDN,
embeddings, final norm, and `lm_head` at their existing Q4/f16. This cuts decode
bytes/step without touching the layers most sensitive to quantization error.

Non-goals: no CUDA, no ONNX, no MoE W3 (routed/shared experts stay Q4 — the W3
loader must return `UnsupportedModel` for `cfg.is_moe()`), no claim of quality
preservation without a measured PPL number.

---

## 2. 3-bit Format

### Block layout

- **Group size:** 32 weights/block (mirrors Q4, preserves the `K % 32 == 0`
  dispatch invariant).
- **Quantization:** unsigned 3-bit **asymmetric** min/max per block.
  - `scale = (max == min) ? 1.0 : (max - min) / 7.0`
  - `bias  = min`
  - `q[i]  = clamp(round((w[i] - bias) / scale), 0, 7)`
  - `w[i] ≈ q[i] * scale + bias`
- **`W3Block` — `#[repr(C)]`, exactly 16 bytes:**

  | field  | type      | bytes | offset |
  |--------|-----------|------:|-------:|
  | scale  | `u16` (f16 bit pattern) | 2 | 0 |
  | bias   | `u16` (f16 bit pattern) | 2 | 2 |
  | packed | `[u8; 12]`              | 12 | 4 |

  `const _: () = assert!(size_of::<W3Block>() == 16);`

- **Bit packing:** sequential, little-endian, LSB-first within the 12 payload
  bytes. `bit_offset = i*3`, `byte = bit_offset/8`, `shift = bit_offset%8`.
  Worked bytes (verified byte-for-byte in tests):
  - `byte0: q0[0:2], q1[0:2], q2[0:1]`
  - `byte1: q2[2], q3[0:2], q4[0:2], q5[0]`
  - `byte2: q5[1:2], q6[0:2], q7[0:2]`
  - `byte3: q8[0:2], q9[0:2], q10[0:1]`
  A lane-friendly view falls out naturally: lane `il ∈ 0..4` owns 8 weights and
  reads exactly 3 payload bytes at `packed + il*3`.

### Byte reduction (the lever)

| format | scale+bias | payload | block | payload Δ | block Δ |
|--------|-----------:|--------:|------:|----------:|--------:|
| Q4 (KHQ4 v2) | 4 B | 16 B | 20 B | — | — |
| **W3 (KHW3 v1)** | 4 B | **12 B** | **16 B** | **−25%** | **−20%** |

Scale+bias overhead is kept (not folded into a zero-point) specifically so the
W3 GEMV math matches the existing Q4 `bias·Σx` optimization and needs no new
zero-point convention.

### `.w3` file format (KHW3 v1)

```text
magic        b"KHW3"             4 B
version      1u32 LE             4 B
ndim         u32 LE              4 B
shape[i]     u64 LE × ndim
original_len u64 LE              8 B
blocks       [W3Block; n_blocks] n_blocks × 16 B
```

`payload_offset = 20 + ndim*8` (mirrors the KHQ4 header arithmetic). Header
parsing rejects: wrong magic, unsupported version, shape/product overflow,
`shape.product() != original_len`, and payload shorter than
`ceil(original_len/32) * 16`.

### Asymmetric vs symmetric

Asymmetric min/max chosen over symmetric signed 3-bit: with only 8 levels a
signed range is inherently imbalanced (`-4..3`) or wastes a code (`-3..3`), a bad
trade at 3-bit. Asymmetric minimizes per-block error on raw MLP weights and
mirrors Q4 v2 dequant. QuaRot-style rotation is **out of scope** here — no
confirmed W3 rotation quality bound exists for this task.

---

## 3. Layer Scope (MLP-only)

W3 applies **only** to dense MLP tensors:
`model.language_model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight`.

`is_w3_mlp_tensor_name` fails **closed**: it returns `false` for MoE routed
experts (`.mlp.experts.*`), shared experts (`.mlp.shared_expert*`), attention
`q/k/v/o`, GDN `in_proj_*`/`out_proj`, embeddings, `lm_head`, norms, biases, and
any non-`.weight` tensor. Everything not provably a dense MLP weight stays Q4/f16.

---

## 4. Metal Kernel Approach (DESIGNED — see §7 for build status)

Mixed-format path: base projections stay Q4 (`gemv_q4_decode`/`gemm_q4`); dense
MLP branches to new W3 kernels via `MetalDenseFfnWeights::{Q4,W3}` — the branch is
on the FFN weight enum, **never** on a global `QuantFormat`, so W3 can never leak
into attention/GDN/logits.

- **`gemv_w3_decode`** mirrors `gemv_q4_decode`: `NR=2`, `NSG=4`,
  `threads=(32,4,1)`, `dispatch=(ceil(N/2),1,1)`, `row_bytes=(K/32)*16`. Each lane
  reads **3** packed bytes (vs Q4's 4), unpacks 8 codes, and accumulates
  `Σ(q·yl)·scale + bias·Σyl` — the same `bias·Σx` optimization as Q4.
- **`gemm_w3`** is a naive batch GEMM mirror of `gemm_q4` for prefill /
  `verify_tokens_batch_gemm`. No tiled simdgroup-matrix path in v1 — the target
  lever is *decode* bandwidth; a correct naive GEMM keeps prefill/verify right.
- **Loader** `MetalQwen35State::from_w3_mlp_dir` mirrors `from_q4_dir`, loads
  `.w3` for dense MLP and `.q4`/`.f16` for everything else, and **rejects**:
  `cfg.is_moe()`, missing `.w3` MLP file, stray MLP `.q4` alongside `.w3` (no
  silent Q4 fallback), and any W3 K-dim that is zero or not divisible by 32
  (validated at load, not deferred to a dispatch-time panic).
- **Converter** `quantize_w3_mlp` streams safetensors → mixed `.w3`/`.q4`/`.f16`
  dir + `quantize_index.json`, routing tensors by
  `output_format(name)`.
- **Fused `gate||up`** batch dispatch uses
  `gate_byte_size = inter * (hidden/32) * W3_BLOCK_SIZE`.

Full type signatures and dispatch pseudocode are in `../architect/design.md`.

---

## 5. Measured PPL Delta + Decode A/B

Source: `../analyst/measurements.md`. Local model `qwen3.5-0.8b`, wiki corpus,
`window=128 stride=64`. **These are SMOKE measurements on a capped corpus, not a
publication-grade PPL run**, and the analyst flags them as such.

### 5.1 Perplexity

| Path | Runtime | Max tok | Scored | PPL | Δ vs f16 | Bound status |
|------|---------|--------:|-------:|-----:|---------:|--------------|
| f16/BF16 safetensors | CPU | 512 | 511 | 15.587897 | baseline | baseline |
| Q4 unrotated | Metal | 512 | 511 | 16.741339 | +1.153443 | outside known Q4 +0.1–0.3 |
| f16/BF16 safetensors | CPU | 1024 | 1023 | 17.299129 | baseline | baseline |
| Q4 unrotated | Metal | 1024 | 1023 | 18.709340 | +1.410211 | outside known Q4 +0.1–0.3 |
| **W3 MLP-only** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **INCOMPLETE — no runnable W3 eval path** |

W3 PPL attempt (recorded failure):

```text
$ ./target/release/eval_perplexity --w3-mlp-dir .../qwen3.5-0.8b-w3-mlp ...
ERROR: unknown argument: --w3-mlp-dir
```

The Q4 smoke deltas themselves land **outside** the known Q4 +0.1–0.3 full-eval
bound — a signal that the capped window/stride is not comparable to a full
WikiText-2 pass, so these numbers are plumbing/sanity checks, not acceptance
numbers, for either Q4 or W3.

### 5.2 Decode A/B (tok/s, slope method)

`decode_tok_per_s = (128-64) / (median_ms_128 − median_ms_64) * 1000`

| Path | Runtime format | Median 64 ms | Median 128 ms | tok/s | Status |
|------|----------------|-------------:|--------------:|------:|--------|
| Q4 directory | Metal Q4_0 | 427.587 | 831.570 | **158.42** | measured |
| safetensors-direct | Metal Q8_0 | 644.078 | 1192.490 | **116.70** | measured |
| **W3 MLP-only** | **N/A** | **N/A** | **N/A** | **N/A** | **INCOMPLETE — no W3 decode path** |

W3 decode attempt (recorded failure): `bench_decode_ab` on a W3 dir panics
`ModelNotFound(...)` — no `.safetensors`, no W3 directory detection.

The existing Q4-vs-Q8_0 gap (**+35.75%**) confirms that byte-read reduction
*does* translate to real decode speedup on this harness — which is the mechanism
W3 is designed to exploit further. **But W3's own speedup is not measured**, so
no W3 speed claim is made.

---

## SILENT QUALITY LOSS RISK

**This is the founder gate. The W3 quality cost is currently UNMEASURED, and an
unmeasured 3-bit quality cost is the single most dangerous failure mode for this
change.** A W3 model still decodes fluent-looking tokens even when quantization
has degraded it, so quality loss is *silent* — it will not surface as a crash or
an obviously-wrong output; it surfaces only as a higher perplexity / worse
downstream quality that a reader must go measure.

**Quantified cost basis (what we know):**

- Q4 (4-bit) MLP+everything costs a known **+0.1–0.3 PPL** vs f16 in a full eval.
  W3 uses **3 bits instead of 4** on the MLP — strictly fewer levels (8 vs 16),
  so W3's MLP quantization error is **strictly larger** than Q4's on those
  tensors. W3 PPL delta vs Q4 is therefore expected to be **positive and larger
  than the MLP's share of the +0.1–0.3 Q4 budget** — but the exact magnitude is
  **not measured on this branch**.
- The measured Q4 smoke delta on the capped corpus was already **+1.15 to +1.41
  PPL** (§5.1) — outside the known full-eval bound. This does **not** mean Q4 is
  that bad; it means the smoke harness is not comparability-grade. It is a
  concrete warning that **W3 must be evaluated on a full, comparable corpus**,
  not a capped smoke run, before any acceptance.

**Required to close this gate (none of these exist yet):**

1. A runnable W3 path: converter → `.w3` artifacts → `from_w3_mlp_dir` loader →
   `gemv_w3_decode` kernel (design §4; build status §7).
2. `eval_perplexity --w3-mlp-dir` wired for a full/comparable-corpus run.
3. The three measured deltas written back here:
   `delta_w3_vs_f16`, `delta_w3_vs_q4`, `delta_q4_vs_f16` — on the **same**
   corpus/window/token-cap.
4. A decode tok/s A/B confirming the byte reduction yields real speedup, so the
   speed **benefit** can be weighed against the quality **cost**.

**Additional silent-failure surfaces in the (designed) Metal path** — each
produces plausible tokens while being wrong, so each needs an explicit test:

- Missing `.w3` silently falling back to `.q4` (loader must reject stray MLP
  `.q4` in a W3 dir).
- Wrong `gate_byte_size` offset in fused `gate||up`, making `up` read from the
  middle of `gate`.
- Measuring only tok/s and calling quality "fine."
- Reporting a smoke PPL on too few tokens as a quality claim.

**Verdict: W3 quality is NOT verified. Per the task's Π_TBV rule, this path is
INCOMPLETE for quality until a full-corpus W3 PPL number exists. Do not merge on
the strength of the byte-reduction argument alone.**

---

## 6. Rust Module / Public API

Delivered public surface — `lattice_inference::weights::w3_weights`:

```rust
pub const W3_GROUP_SIZE: usize = 32;
pub const W3_PACKED_BYTES: usize = 12;
pub const W3_BLOCK_SIZE: usize = 16;

pub struct W3Block  { pub scale: u16, pub bias: u16, pub packed: [u8; 12] }
pub struct W3Tensor { pub blocks: Vec<W3Block>, pub shape: Vec<usize>, pub original_len: usize }
pub struct W3FileHeader { pub shape: Vec<usize>, pub original_len: usize, pub payload_offset: u64 }

pub fn quantize_row_w3(src: &[f32]) -> Result<Vec<u8>, InferenceError>;
pub fn dequantize_row_w3(data: &[u8], n_weights: usize) -> Vec<f32>;
pub fn quantize_tensor_w3(src: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>, InferenceError>;
pub fn quantize_bf16_to_w3(data: &[u16], shape: &[usize]) -> Result<W3Tensor, InferenceError>;
pub fn quantize_f32_to_w3(data: &[f32], shape: &[usize]) -> Result<W3Tensor, InferenceError>;
pub fn dequantize_w3_to_f32(tensor: &W3Tensor) -> Vec<f32>;
pub fn save_w3_file(path: &Path, tensor: &W3Tensor) -> Result<(), InferenceError>;
pub fn read_w3_header(file: &File) -> Result<W3FileHeader, InferenceError>;
pub fn load_w3_file(path: &Path) -> Result<W3Tensor, InferenceError>;
pub fn is_w3_mlp_tensor_name(name: &str) -> bool;
```

**Error handling:** all public shape validation returns
`InferenceError::{ShapeMismatch,InvalidInput}` via checked multiplication — no
`assert_eq!`, no `unwrap()`/`expect()` in library code (the two `try_into().expect`
calls narrow an already-length-checked `chunks_exact(16)` slice to `[u8;12]`,
provably infallible, identical to `q4_weights::load_q4_file`). Non-finite source
weights are rejected before min/max. Reuses `q4_weights::q4_{f16_to_f32,f32_to_f16}`
(same crate, zero churn) rather than duplicating f16 bit-twiddling.

---

## 7. DONE vs DESIGNED (explicit — no stubs presented as done)

### DONE — implemented + tested this pass

- `crates/inference/src/weights/w3_weights.rs` (~660 LOC + 35 inline tests): CPU
  W3 pack/dequant, `.w3` file I/O, MLP tensor-name classification.
- `crates/inference/tests/w3_weights_integration.rs` (11 independent integration
  tests): happy-path error bound, group-boundary isolation on non-multiple-of-32
  shapes, exact/one-past-multiple sizes, degenerate/zero-range blocks,
  empty/single-element edges, real temp-file roundtrip, quantization-level
  mutation guard. Mutation-sensitivity confirmed by fault injection (3/11 fail
  when bit-packing is broken, then reverted byte-identical).
- **46 W3 tests total, 100% pass, 0 workspace regressions** (1364 pre-existing).

### DESIGNED, NOT IMPLEMENTED (full spec in `../architect/design.md`)

1. `quantize_w3_mlp` converter binary.
2. Mixed W3/Q4 Metal loader (`from_w3_mlp_dir`, `W3WeightBuf`, `MlpQuantFormat`,
   `MetalDenseFfnWeights`) — **highest-risk untested surface**.
3. `gemv_w3_decode` / `gemm_w3` Metal kernels + dispatch helpers (needs a macOS
   Metal device to validate numerically against the CPU reference).
4. `new_w3_mlp` in-memory constructor + `new_with_mlp_quant_format`.
5. CLI/eval/bench wiring (`--w3-mlp-dir`, `bench_decode_ab` W3 detection).
6. **Quality + speed measurement** (W3 PPL delta, W3 decode A/B) — see
   [§ SILENT QUALITY LOSS RISK](#silent-quality-loss-risk).

**Recommended next increment:** converter (item 1, no Metal needed) → coupled
loader+kernels (items 2–3) → measurement (item 6) in the *same* pass, because a
W3 Metal path without a same-session PPL number would re-open this exact gate.

---

## 8. Gate Results

See `../coordinator/ship_report.md` for the pasted actual output of
`cargo fmt --check`, `cargo clippy --workspace -- -D warnings`,
`cargo clippy -p lattice-inference --features metal-gpu -- -D warnings`, and the
W3 test suites.
