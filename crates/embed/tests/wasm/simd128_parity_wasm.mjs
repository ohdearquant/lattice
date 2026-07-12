// wasm32 SIMD128 kernel parity gate.
//
// Compares the crate's `simd::{dot_product, squared_euclidean_distance,
// cosine_similarity, normalize}` wasm32 SIMD128 kernels (crates/embed/src/simd/
// {dot_product,distance,cosine,normalize}.rs) against the scalar reference,
// by loading TWO wasm-bindgen builds of the same crate side by side:
//
//   - baseline: plain `wasm32-unknown-unknown` build (no `simd128` target
//     feature) -- every `#[cfg(target_feature = "simd128")]` kernel in this
//     module is absent from the binary, so `SimdConfig::simd128_enabled` is
//     `false` and dispatch always falls through to the scalar kernel.
//   - simd128: same crate, same source, built with
//     `RUSTFLAGS="-C target-feature=+simd128"` -- dispatch picks the new
//     wasm32 SIMD128 kernels.
//
// Both builds expose the same four JS bindings (simdDotProduct,
// simdSquaredEuclideanDistance, simdCosineSimilarity, simdNormalize; see
// wasm.rs), so this is a same-process, same-language, wasm-vs-wasm
// comparison: no cross-framework or cross-model tolerance is needed, just a
// tight relative epsilon (see ASSERT below), which is the same discipline
// this crate's native tests apply to AVX-512/AVX2/NEON vs scalar (see
// crates/embed/src/simd/tests.rs).
//
// Run via `scripts/wasm-simd128-parity.sh` (builds both variants, then
// invokes this script) or directly with LATTICE_WASM_JS_BASELINE /
// LATTICE_WASM_JS_SIMD128 pointing at pre-built wasm-bindgen nodejs output.
//
// Mutation-sensitivity: this suite was manually verified to FAIL when a
// deliberate bug was introduced into `horizontal_sum_simd128`
// (crates/embed/src/simd/dot_product.rs) -- dropping lane 3 from the 4-lane
// reduction so only 3 of 4 SIMD128 lanes are summed -- then re-verified to
// PASS after reverting. See this PR's description for the before/after run
// transcript.

import { existsSync } from 'node:fs';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Deterministic PRNG (mulberry32) -- reproducible test vectors, no external
// dependency.
// ---------------------------------------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomVec(dim, seed) {
  const rng = mulberry32(seed);
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = rng() * 2 - 1; // [-1, 1)
  return v;
}

// f32-faithful scalar reference: `Math.fround` after every multiply/add
// rounds each intermediate to f32 the same way Rust's `f32 * f32 -> f32` /
// `f32 + f32 -> f32` do. Without this, JS's native double-precision math
// silently disagrees with Rust at the extremes (e.g. a f32 subnormal squared
// underflows to exactly 0.0 in f32, but not in f64) -- not a SIMD-vs-scalar
// difference, just this reference computing in the wrong precision. Every
// element read from a Float32Array is already an exact f32 value widened to
// a JS double, so only the arithmetic *results* need rounding, matching
// Rust's `a.iter().zip(b.iter())....sum()` left-to-right f32 fold exactly.
const f32 = Math.fround;

function scalarDot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s = f32(s + f32(a[i] * b[i]));
  return s;
}

function scalarSqL2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = f32(a[i] - b[i]);
    s = f32(s + f32(d * d));
  }
  return s;
}

function scalarCosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot = f32(dot + f32(a[i] * b[i]));
    na = f32(na + f32(a[i] * a[i]));
    nb = f32(nb + f32(b[i] * b[i]));
  }
  na = f32(Math.sqrt(na));
  nb = f32(Math.sqrt(nb));
  if (na === 0 || nb === 0) return 0;
  return f32(dot / f32(na * nb));
}

// ---------------------------------------------------------------------------
// Test vector cases
// ---------------------------------------------------------------------------

// Dims deliberately not all multiples of the wasm128 kernel's 16-float
// unrolled chunk size, so every case exercises the SIMD-main-body +
// remainder-vector + scalar-tail split (dot_product_simd128_unrolled's
// CHUNK_SIZE=16, SIMD_WIDTH=4).
const REMAINDER_DIMS = [1, 2, 3, 5, 7, 17, 385];
const TYPICAL_DIMS = [384, 768, 1024, 4096];

let failures = 0;
let checks = 0;

function assertClose(actual, expected, label) {
  checks++;
  const relTol = 1e-5;
  const absTol = 1e-6;
  const diff = Math.abs(actual - expected);
  const limit = Math.max(absTol, relTol * Math.max(Math.abs(expected), 1));
  if (!(diff <= limit)) {
    failures++;
    console.error(
      `FAIL [${label}] actual=${actual} expected=${expected} diff=${diff.toExponential(3)} limit=${limit.toExponential(3)}`,
    );
  }
}

function assertVecClose(actual, expected, label) {
  for (let i = 0; i < expected.length; i++) {
    assertClose(actual[i], expected[i], `${label}[${i}]`);
  }
}

function runPairCases(mod, label) {
  const cases = [];

  for (const dim of [...REMAINDER_DIMS, ...TYPICAL_DIMS]) {
    cases.push({ name: `random dim=${dim}`, a: randomVec(dim, 1000 + dim), b: randomVec(dim, 2000 + dim) });
  }

  // Zeros: exercises the norm==0 guards in cosine/normalize.
  cases.push({ name: 'zeros dim=64', a: new Float32Array(64), b: new Float32Array(64) });

  // Denormals: subnormal f32 range starts below ~1.1755e-38.
  {
    const dim = 64;
    const a = new Float32Array(dim).fill(1e-40);
    const b = new Float32Array(dim).fill(-1e-40);
    cases.push({ name: 'denormals dim=64', a, b });
  }

  // Adversarial: magnitude alternates by 12 orders every element -- a
  // reduction-order bug shows up as a large relative error here even though
  // it might hide in a uniformly-scaled random vector.
  for (const dim of [16, 17, 64, 385]) {
    const a = new Float32Array(dim);
    const b = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      a[i] = i % 2 === 0 ? 1e6 : 1e-6;
      b[i] = i % 2 === 0 ? 1.0 : -1.0;
    }
    cases.push({ name: `alternating-magnitude dim=${dim}`, a, b });
  }

  // Lane-position spikes: a single nonzero element at each of the 4 SIMD128
  // lane offsets (0,1,2,3 mod 4), isolated across otherwise-zero vectors, so
  // a lane-drop or lane-swap bug in the horizontal-sum/accumulator-combine
  // step is caught regardless of which lane it targets.
  for (const lane of [0, 1, 2, 3]) {
    const dim = 32;
    const a = new Float32Array(dim);
    const b = new Float32Array(dim);
    for (let i = lane; i < dim; i += 4) {
      a[i] = 3.0 + i * 0.1;
      b[i] = -2.0 + i * 0.05;
    }
    cases.push({ name: `lane-spike lane=${lane}`, a, b });
  }

  for (const c of cases) {
    const dot = mod.simdDotProduct(c.a, c.b);
    assertClose(dot, scalarDot(c.a, c.b), `${label} dot ${c.name}`);

    const sqL2 = mod.simdSquaredEuclideanDistance(c.a, c.b);
    assertClose(sqL2, scalarSqL2(c.a, c.b), `${label} sqL2 ${c.name}`);

    if (c.a.length > 0) {
      const cos = mod.simdCosineSimilarity(c.a, c.b);
      assertClose(cos, scalarCosine(c.a, c.b), `${label} cosine ${c.name}`);
    }
  }

  console.log(`[${label}] pair cases: ${cases.length}`);
}

function runNormalizeCases(mod, label) {
  const dims = [...REMAINDER_DIMS, ...TYPICAL_DIMS];
  for (const dim of dims) {
    const v = randomVec(dim, 5000 + dim);
    const expected = Float32Array.from(v);
    const norm = f32(Math.sqrt(scalarDot(expected, expected)));
    if (norm > 0) {
      for (let i = 0; i < dim; i++) expected[i] /= norm;
    }

    const actual = Float32Array.from(v);
    mod.simdNormalize(actual);
    assertVecClose(actual, expected, `${label} normalize dim=${dim}`);
  }

  // Zero vector: must stay unchanged (norm == 0 guard).
  {
    const zeros = new Float32Array(16);
    mod.simdNormalize(zeros);
    assertVecClose(zeros, new Float32Array(16), `${label} normalize zeros`);
  }

  // NaN element: the whole vector's L2 norm is NaN, so both the scalar and
  // every SIMD kernel in this module must leave the vector byte-unchanged
  // (see normalize.rs's `norm.is_nan() || norm <= 0.0` guard) rather than
  // scale by a NaN inverse norm. Dims deliberately not multiples of the
  // unrolled chunk size so the NaN can land in either the SIMD main body or
  // the scalar tail.
  for (const dim of [17, 129, 385]) {
    for (const nanAt of [3, dim - 1]) {
      const original = randomVec(dim, 6000 + dim + nanAt);
      original[nanAt] = NaN;
      const actual = Float32Array.from(original);
      mod.simdNormalize(actual);
      for (let i = 0; i < dim; i++) {
        checks++;
        const a = actual[i];
        const o = original[i];
        const same = Number.isNaN(a) && Number.isNaN(o) ? true : Object.is(a, o);
        if (!same) {
          failures++;
          console.error(`FAIL [${label} normalize NaN dim=${dim} nan_at=${nanAt}] index ${i}: ${a} vs original ${o}`);
        }
      }
    }
  }

  console.log(`[${label}] normalize cases: ${dims.length + 1 + 6}`);
}

// ---------------------------------------------------------------------------
// Load both wasm-bindgen builds and run
// ---------------------------------------------------------------------------

const baselinePath = process.env.LATTICE_WASM_JS_BASELINE;
const simd128Path = process.env.LATTICE_WASM_JS_SIMD128;

for (const [name, p] of [['LATTICE_WASM_JS_BASELINE', baselinePath], ['LATTICE_WASM_JS_SIMD128', simd128Path]]) {
  if (!p || !existsSync(p)) {
    console.error(`simd128_parity_wasm.mjs: ${name} not set or file missing (got ${p}). Run via scripts/wasm-simd128-parity.sh.`);
    process.exit(1);
  }
}

const baseline = await import(path.resolve(baselinePath));
const simd128 = await import(path.resolve(simd128Path));

// Sanity: the baseline build (no simd128 target feature) must be exercising
// the scalar kernel, i.e. it must itself agree with the JS scalar reference
// closely -- this is what makes the wasm-vs-wasm comparison below meaningful
// evidence about the SIMD128 kernel specifically, not just "two builds agree
// with each other by coincidence."
runPairCases(baseline, 'baseline-vs-scalar-reference');
runNormalizeCases(baseline, 'baseline-vs-scalar-reference');

// The check that matters: simd128-enabled build vs the same scalar reference.
runPairCases(simd128, 'simd128-vs-scalar-reference');
runNormalizeCases(simd128, 'simd128-vs-scalar-reference');

console.log(`\nwasm32 SIMD128 parity gate: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error('wasm32 SIMD128 parity gate FAILED');
  process.exit(1);
}
console.log('wasm32 SIMD128 parity gate PASSED');
process.exit(0);
