# ADR-002: SIMD Dispatch Strategy

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

---

## Context

Transformer inference on modern hardware requires vectorized arithmetic to achieve competitive throughput. The crate targets two ISA families with fundamentally different detection requirements:

- **aarch64 (Apple Silicon, Raspberry Pi)**: NEON is part of the ARMv8-A mandatory profile. Every compliant aarch64 CPU has it. Detection overhead is pure waste.
- **x86_64 (Intel, AMD)**: AVX2, FMA, and AVX-512 are optional extensions. Presence must be checked at runtime because a binary built on a Haswell host may run on a pre-Haswell target without them.

The codebase needs a single authoritative path through which all SIMD-dependent kernels discover the current machine's capabilities, without repeated runtime probing.

Relevant implementation: `src/forward/cpu/simd.rs`.

```rust
// Initialization via OnceLock — called once, stored globally.
static SIMD_CONFIG: OnceLock<SimdConfig> = OnceLock::new();

pub(crate) fn simd_config() -> SimdConfig {
    *SIMD_CONFIG.get_or_init(SimdConfig::detect)
}

impl SimdConfig {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: is_x86_feature_detected!("avx2"),
                fma:  is_x86_feature_detected!("fma"),
                avx512f: is_x86_feature_detected!("avx512f"),
                neon_enabled: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        { Self { avx2: false, fma: false, avx512f: false, neon_enabled: true } }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { Self { avx2: false, fma: false, avx512f: false, neon_enabled: false } }
    }
}
```

The NEON matrix-vector kernel in `src/forward/neon.rs` is gated by `#[cfg(target_arch = "aarch64")]` and `#[target_feature(enable = "neon")]`. The safe wrapper `matmul_q8_neon()` calls the inner directly on aarch64 (no runtime check) and falls back to scalar on all other targets.

---

## Decision

Use a **`OnceLock<SimdConfig>` singleton** initialized on first call to `simd_config()`. ISA branches are separated at compile time via `#[cfg]`; the aarch64 path sets `neon_enabled: true` unconditionally with no runtime check; the x86_64 path invokes `is_x86_feature_detected!` once and caches the result forever.

---

## Key Design Choices

1. **`OnceLock` instead of `Mutex<Option<_>>`**: zero-cost reads after initialization; the lock is only taken when the `OnceLock` is empty (exactly once per process).
2. **`Copy` on `SimdConfig`**: the struct is 4 bools. Returning by value from `simd_config()` avoids any reference lifetime concerns and lets the compiler inline the field checks.
3. **No compile-time feature flags for SIMD levels**: `#[cfg(target_feature = "avx2")]` would require `RUSTFLAGS="-C target-feature=+avx2"` at build time and would produce a binary that crashes on older CPUs. Runtime detection keeps the binary portable while still using the widest instruction set available at runtime.
4. **aarch64 skips runtime check entirely**: `is_arm_feature_detected!("neon")` exists in nightly Rust but is not stabilized and is unnecessary: the ARMv8-A spec mandates NEON. Checking it would be misleading (implies it could be absent) and adds a pointless CPUID-equivalent call.

---

## Alternatives Considered

| Alternative                                           | Pros                                                   | Cons                                                                          | Why Not                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `#[cfg(target_feature = "avx2")]` compile-time gating | Zero runtime overhead; enables autovectorization hints | Binary only runs on machines with that feature; separate build per micro-arch | Defeats the purpose of shipping a single binary                            |
| `std::sync::Mutex<Option<SimdConfig>>`                | Straightforward                                        | Adds lock overhead on every dispatch call; `Mutex` is not needed after init   | `OnceLock` was stabilized in Rust 1.70 and is the correct primitive        |
| Per-call `is_x86_feature_detected!`                   | No global state                                        | CPUID is not free; called thousands of times per forward pass                 | Unnecessary overhead; CPUID result is invariant for the process lifetime   |
| Runtime CPU dispatch via function pointer table       | Flexible; used by BLAS libraries                       | Significant complexity; not needed for the small set of kernels here          | Over-engineered for two ISA targets                                        |
| `portable-simd` nightly feature                       | Portable abstraction                                   | Nightly only; unstable API; performance depends on autovectorizer quality     | Stability requirement; explicit NEON intrinsics are faster and predictable |

---

## Consequences

**Positive**:

- Detection overhead is paid once per process, not once per matrix multiply.
- `simd_config()` is inlinable and branch-free on aarch64 (always returns `neon_enabled: true`).
- Adding a new ISA (e.g., SVE) requires adding one field to `SimdConfig` and one `#[cfg]` branch in `detect()`.

**Negative**:

- `OnceLock` global state makes unit tests that need to control SIMD behavior harder: the config is locked after the first call in a test binary.
- The aarch64 unconditional path is correct today but assumes no ARMv8-A implementation ships without NEON (a valid assumption at time of writing).

**Risks**:

- A future ARM embedded target (M-profile) without NEON would silently produce incorrect builds unless the `#[cfg(target_arch = "aarch64")]` branch is further qualified (e.g., `not(target_os = "none")`).

---

## References

- `src/forward/cpu/simd.rs` — `SimdConfig`, `simd_config()`, `OnceLock` initialization
- `src/forward/neon.rs` — `matmul_q8_neon()`, `#[target_feature(enable = "neon")]` inner
- Rust Reference: `is_x86_feature_detected!` — https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html
- ARMv8-A Architecture Reference Manual §A2.1 — NEON/Advanced SIMD mandatory
