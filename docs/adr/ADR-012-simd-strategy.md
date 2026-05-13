# ADR-012: Runtime SIMD Detection Strategy

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-embed

## Context

lattice-embed provides SIMD-accelerated vector operations (dot product, cosine similarity,
normalize, euclidean distance) for embedding computations. The crate must support
multiple platforms:

- **x86_64**: Intel/AMD processors with varying SIMD support (SSE4, AVX2, AVX-512)
- **aarch64**: ARM processors (Apple Silicon, AWS Graviton) with NEON
- **Other architectures**: Fallback for unsupported platforms

Performance requirements are significant. Benchmarks on 384-dimensional vectors show:

| Operation         | Scalar | SIMD  | Speedup |
| ----------------- | ------ | ----- | ------- |
| dot_product       | ~230ns | ~35ns | 6.5x    |
| cosine_similarity | ~650ns | ~90ns | 7x      |
| normalize         | ~400ns | ~60ns | 6.5x    |
| dot_product_i8    | ~300ns | ~25ns | 12x     |

The challenge: how to achieve these speedups while maintaining a single portable binary
that works across all target platforms.

## Decision

We use **runtime SIMD detection** with automatic scalar fallback via Rust's `std::arch`
module.

### Implementation

1. **Detection at startup**: Query CPU features once via `std::is_x86_feature_detected!`
   (x86_64) or compile-time detection (aarch64 NEON is mandatory)

2. **Cached configuration**: Store results in `SimdConfig` struct, queried once per
   process

3. **Dispatch priority**:

   - x86_64: AVX-512 VNNI > AVX2 + FMA > Scalar
   - aarch64: ARM NEON > Scalar

4. **Target feature functions**: Use `#[target_feature(enable = "...")]` with `unsafe`
   blocks for SIMD code paths

```rust
pub struct SimdConfig {
    pub avx2_enabled: bool,
    pub fma_enabled: bool,
    pub avx512vnni_enabled: bool,
    pub neon_enabled: bool,
}

pub fn simd_config() -> SimdConfig; // Detected once, cached
```

### Dispatch Logic

```rust
#[cfg(target_arch = "x86_64")]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let cfg = simd_config();
    if cfg.avx512vnni_enabled {
        unsafe { dot_product_avx512(a, b) }
    } else if cfg.avx2_enabled && cfg.fma_enabled {
        unsafe { dot_product_avx2_fma(a, b) }
    } else {
        dot_product_scalar(a, b)
    }
}
```

## Consequences

### Positive

- **Single portable binary**: Same artifact runs on all x86_64/aarch64 machines
- **Automatic optimization**: Best available SIMD used without user configuration
- **No runtime dependencies**: Pure Rust, no external SIMD libraries required
- **Graceful degradation**: Older CPUs get scalar fallback (correct, just slower)
- **Compile once, run anywhere**: Simplifies CI/CD and distribution

### Negative

- **Detection overhead**: One-time CPU feature detection at first use (~microseconds,
  negligible)
- **Code complexity**: Multiple implementations per operation (scalar + 2-3 SIMD
  variants)
- **Binary size increase**: All SIMD variants compiled into binary (~50KB additional)
- **Unsafe code**: SIMD intrinsics require `unsafe` blocks (contained within module)
- **Testing burden**: Must test all code paths (CI matrix for different CPU features)

### Neutral

- **Maintenance**: Adding new SIMD targets (e.g., AVX-10) requires new code paths
- **Debugging**: SIMD code harder to debug than scalar (mitigated by scalar fallback for
  comparison)

## Alternatives Considered

### Option A: Compile-Time Target Features

Build separate binaries for each target CPU (e.g., `target-cpu=native`,
`target-feature=+avx2`).

**Pros**:

- Zero runtime detection overhead
- Compiler can optimize entire codebase for specific CPU
- No dispatch branching

**Cons**:

- Multiple binaries to build, test, and distribute
- Users must select correct binary for their CPU
- CI/CD complexity (build matrix explosion)
- Incompatible binary causes illegal instruction crash

**Rejected because**: Distribution complexity outweighs marginal performance gains.
Detection overhead is negligible (once per process).

### Option B: No SIMD (Scalar Only)

Use standard Rust iterators and let LLVM auto-vectorize where possible.

**Pros**:

- Simplest implementation
- No unsafe code
- No platform-specific code

**Cons**:

- 6-12x slower vector operations
- LLVM auto-vectorization unreliable (may not trigger)
- Competitive disadvantage vs. other embedding libraries

**Rejected because**: Performance requirements demand explicit SIMD. 6-12x speedup is
too significant to forego.

### Option C: External SIMD Library (simdeez, packed_simd, portable-simd)

Use a SIMD abstraction crate instead of raw `std::arch`.

**Pros**:

- Cleaner API abstraction
- Cross-platform code sharing
- Community-maintained

**Cons**:

- Additional dependency
- May not expose all CPU features (AVX-512 VNNI for int8)
- Abstraction overhead
- Less control over exact instruction selection

**Rejected because**: We need precise control over int8 quantized dot product (AVX-512
VNNI), and direct `std::arch` provides better control with minimal overhead. The
complexity is contained within the `simd.rs` module.

## References

- [Rust std::arch documentation](https://doc.rust-lang.org/std/arch/index.html)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- TDS-embedding-services.md Section 6: SIMD Acceleration
