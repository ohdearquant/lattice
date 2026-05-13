# Safety

## Unsafe Code Policy

Lattice's policy is: **unsafe is permitted only for SIMD intrinsic calls and memory-mapped
tensor views. All unsafe blocks must be justified in a comment at the call site.**

`unsafe` is never used for convenience. Any unsafe block that is not a SIMD intrinsic or
a direct mmap slice cast is a bug.

## Unsafe Block Count

Counts are recorded in each crate's top-level `lib.rs` comment and updated on each change:

| Crate               | `unsafe` blocks | Reason                                                                                   |
| ------------------- | --------------- | ---------------------------------------------------------------------------------------- |
| `lattice-inference` | 153             | SIMD matmul/attention kernels (AVX2, NEON, Metal FFI), mmap tensor slices, f16 bit-casts |
| `lattice-embed`     | 21              | SIMD intrinsic calls (AVX-512, AVX2, NEON) in distance and normalize kernels             |
| `lattice-fann`      | 0               | No unsafe — all operations are safe Rust                                                 |
| `lattice-transport` | 0               | No unsafe — pure safe Rust math                                                          |
| `lattice-tune`      | 0               | No unsafe                                                                                |

The `lattice-inference` count is high because each SIMD kernel for each operation
(matmul, softmax, norm, activation) contains multiple intrinsic call sites. The count
is tracked to prevent silent growth — see tracking issue #1306.

## Categories of Unsafe in lattice-inference

**SIMD intrinsics** — `_mm256_*` (AVX2), `vld1q_f32` / `vmlaq_f32` (NEON), `_mm512_*`
(AVX-512 nightly). These require unsafe because the Rust standard library does not expose
SIMD operations through a safe API. Each call site is preceded by a comment naming the
intrinsic family and asserting the pre-condition (e.g., "caller ensures slices are aligned
to 32 bytes").

**Memory-mapped tensor views** — The `SafetensorsFile` loader casts a byte slice from the
mmap into `&[f32]` or `&[u16]` via `bytemuck`. The unsafe is in the alignment and size
assertions. `bytemuck` is used precisely to reduce the raw pointer casting to a single
auditable location.

**Metal FFI** — The Metal GPU backend calls into `metal-rs` which itself uses `objc::msg_send!`.
These are isolated to `crates/inference/src/forward/metal.rs` and
`crates/inference/src/forward/metal_qwen35.rs`.

**f16 bit-casts** — Half-precision weights are loaded as `u16` and reinterpreted as `f16`.
The cast is done through `bytemuck::cast_slice` with the `Pod` bound, which is safe only for
types satisfying the `Pod` contract.

## Edition 2024 and `unsafe_op_in_unsafe_fn`

Workspace `Cargo.toml` sets:

```toml
[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "allow"
```

Rust 2024 edition makes `unsafe_op_in_unsafe_fn` a warning by default, meaning that
calling an unsafe function inside an `unsafe fn` still requires an explicit `unsafe {}`
block. This workspace allows the old behavior because the SIMD kernel functions are
`unsafe fn` with large bodies where re-wrapping every intrinsic call in a nested block
would reduce readability without adding safety guarantees (the entire function body is
already gated by the function's `unsafe` contract). This is a documented trade-off, not
a blanket exemption.

## Audit Approach

**Static**: `cargo clippy -D warnings` is run in CI. The `correctness` lint group is set
to `deny` at the workspace level (see `Cargo.toml`). Clippy does not catch all unsafe
misuse, but it catches use-after-free patterns in iterator chains and common misuses of
`std::slice::from_raw_parts`.

**Dynamic**: `cargo test` with `RUSTFLAGS="-Z sanitize=address"` under nightly can be run
locally to detect use-after-free and buffer overruns in unsafe blocks. This is not
currently in the required CI matrix due to nightly dependency, but is recommended before
any kernel change.

**Review**: PRs that touch any `unsafe` block require a second reviewer to confirm the
safety invariant is documented at the call site.

## Feature Flags That Reduce Unsafe Surface

| Flag                           | Effect                                                     |
| ------------------------------ | ---------------------------------------------------------- |
| Default (`native`, no GPU)     | 21 unsafe blocks in `embed` (SIMD only)                    |
| Disable `native` feature       | 0 unsafe blocks in `embed` (no inference, remote API only) |
| No `metal-gpu`                 | Removes Metal FFI unsafe blocks in `inference`             |
| No `f16`                       | Removes f16 bit-cast unsafe in `inference`                 |
| `lattice-fann` standalone      | 0 unsafe regardless of features                            |
| `lattice-transport` standalone | 0 unsafe regardless of features                            |

To build with the smallest possible unsafe surface for audit purposes:

```sh
# embed with no native inference — zero unsafe in embed crate
cargo build -p lattice-embed --no-default-features

# inference without Metal and without f16
cargo build -p lattice-inference --no-default-features --features std,download
```

## Known Debt

The `lattice-inference` crate has 22 `#[allow(dead_code)]` annotations on functions
retained for reference (superseded BLAS fallbacks, alternative attention paths). These
are tracked in `foundation/STABILITY.md §Tech Debt` under issue #1306. They will be
removed once the replacement paths are confirmed stable in production.
