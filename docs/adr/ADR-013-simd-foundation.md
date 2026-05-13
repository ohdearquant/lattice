# ADR-013: lattice-embed as SIMD Foundation Layer

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-embed

## Context

Vector operations (dot product, cosine similarity, euclidean distance, normalization)
may be duplicated across multiple crates in a codebase:

| Crate                | Implementation                 | Output               |
| -------------------- | ------------------------------ | -------------------- |
| lattice-embed        | SIMD (AVX2/NEON, 7x speedup)   | `f32`                |
| downstream-scorer    | Scalar                         | `DeterministicScore` |
| downstream-retrieval | Scalar (inline in distance.rs) | `f32`                |

This duplication causes:

1. Inconsistent performance (SIMD vs scalar)
2. Maintenance burden (multiple implementations)
3. Potential for divergence

lattice-embed already has:

- Proven SIMD implementations with runtime detection
- 7x speedup on 384-dim vectors (see benches/README.md)
- Int8 quantization support (12x speedup)
- Comprehensive benchmarks and tests

## Decision

**lattice-embed becomes the single source of truth for vector operations.**

### Architecture

```text
┌─────────────────────────────────────────────────┐
│ lattice-embed (SIMD Foundation)                 │
│                                                 │
│ simd.rs:                                        │
│   - cosine_similarity(a, b) -> f32              │
│   - dot_product(a, b) -> f32                    │
│   - euclidean_distance(a, b) -> f32             │
│   - normalize(v) -> ()                          │
│   - batch_* variants                            │
│                                                 │
│ service.rs:                                     │
│   - EmbeddingService trait                      │
│   - NativeEmbeddingService (local inference)    │
│                                                 │
│ cache.rs:                                       │
│   - LRU cache with Blake3 keys                  │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────────┐
│ downstream  │ │ downstream  │ │ downstream      │
│ scorer      │ │ retrieval   │ │ storage         │
│             │ │             │ │                 │
│ Wraps       │ │ Uses        │ │ (storage only)  │
│ SIMD ops    │ │ SIMD ops    │ │                 │
│ Returns     │ │ Formal      │ │                 │
│ typed       │ │ proofs      │ │                 │
│ score       │ │ remain      │ │                 │
└─────────────┘ └─────────────┘ └─────────────────┘
```

### API Changes

```rust
// Before (inline scalar in downstream scorer)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<DeterministicScore, ScoreError>

// After (downstream scorer wrapping lattice-embed)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<DeterministicScore, ScoreError> {
    if a.len() != b.len() { return Err(...); }
    let sim = lattice_embed::simd::cosine_similarity(a, b);
    Ok(DeterministicScore::from_f32(sim))
}
```

```rust
// Before (inline scalar in downstream retrieval)
let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();

// After (downstream retrieval using lattice-embed)
let dot = lattice_embed::simd::dot_product(a, b);
```

## Consequences

### Positive

- **Single implementation**: No duplication, one place to optimize
- **Consistent performance**: SIMD everywhere (7x speedup)
- **Clear layering**: lattice-embed = compute, downstream crates = typed wrappers
- **Easier maintenance**: Bug fixes propagate automatically

### Negative

- **New dependency**: Downstream scoring and retrieval crates depend on lattice-embed
- **Migration work**: Update existing call sites
- **Binary size**: lattice-embed SIMD variants (~50KB) in all dependents

### Neutral

- **Quantization**: Int8 ops stay in lattice-embed (not needed in all downstream crates)

## Migration Plan

1. **lattice-embed**: Export SIMD ops via stable public API
2. **Downstream scorer**: Add lattice-embed dependency, wrap SIMD ops
3. **Downstream retrieval**: Add lattice-embed dependency, use SIMD in distance computation
4. **Verification**: Ensure formal proofs still build, add SIMD golden tests

## References

- ADR-001-simd-strategy.md (runtime SIMD detection)
- benches/README.md (7x speedup measurements)
