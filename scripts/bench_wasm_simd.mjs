#!/usr/bin/env node
// bench_wasm_simd.mjs: A/B benchmark of lattice-embed's wasm32 build with and
// without the `simd128` target feature.
//
// Builds lattice-embed for wasm32-unknown-unknown twice:
//   - baseline: `cargo build --target wasm32-unknown-unknown --features wasm`
//   - simd128:  same, with `RUSTFLAGS="-C target-feature=+simd128"`
// then wasm-bindgen's both to nodejs-target JS glue, loads the four vector-op
// bindings (simdDotProduct, simdSquaredEuclideanDistance,
// simdCosineSimilarity, simdNormalize -- see crates/embed/src/wasm.rs) from
// each, and times them over a range of embedding dimensions, reporting
// median and p95 latency in nanoseconds per call.
//
// Usage:
//   node scripts/bench_wasm_simd.mjs
//   node scripts/bench_wasm_simd.mjs --dims 384,768 --reps 5000
//
// Skip-graceful: exits 0 with a one-line reason if cargo, the wasm32 target,
// or wasm-bindgen are unavailable (mirrors scripts/wasm-parity.sh). Set
// LATTICE_BENCH_WASM_SIMD_ENFORCE=1 to turn a skip into a hard failure.
//
// Methodology notes (also printed in the output header):
//   - Each timed call passes fresh Float32Array inputs through wasm-bindgen's
//     normal marshalling path (copy into wasm linear memory), i.e. this
//     measures realistic JS-caller latency, not a hand-tuned zero-copy
//     microbenchmark. Both variants pay the identical marshalling cost, so
//     the A/B delta isolates the kernel change.
//   - `normalize` mutates its argument in place; the input buffer is reset
//     from a template array before each timed call via `timeCalls`' `prep`
//     callback, which runs before `t0` on every rep (including warmup), so
//     the reset copy is excluded from the timed region -- only the
//     wasm-bindgen call + kernel are measured.
//   - Timing uses `process.hrtime.bigint()` (nanosecond resolution). A
//     warmup phase runs before the timed reps to let V8 JIT the call site.

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const out = { dims: [384, 768, 1024, 4096], reps: 4000, warmup: 300, seed: 0xC0FFEE };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--dims') out.dims = argv[++i].split(',').map(Number);
    else if (a === '--reps') out.reps = Number(argv[++i]);
    else if (a === '--warmup') out.warmup = Number(argv[++i]);
    else if (a === '--seed') out.seed = Number(argv[++i]);
  }
  return out;
}

const ARGS = parseArgs(process.argv.slice(2));
const ENFORCE = process.env.LATTICE_BENCH_WASM_SIMD_ENFORCE;

function skipOrFail(reason) {
  if (ENFORCE) {
    console.error(`bench_wasm_simd: FAIL (LATTICE_BENCH_WASM_SIMD_ENFORCE=1): ${reason}`);
    process.exit(1);
  }
  console.log(`bench_wasm_simd: SKIPPED: ${reason}`);
  process.exit(0);
}

function hasCmd(cmd) {
  try {
    execFileSync(cmd, ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

if (!hasCmd('cargo')) skipOrFail('cargo not found on PATH');
if (!hasCmd('wasm-bindgen')) skipOrFail('wasm-bindgen (CLI) not found on PATH; install with: cargo install wasm-bindgen-cli --version 0.2.105');
try {
  const installed = execFileSync('rustup', ['target', 'list', '--installed']).toString();
  if (!installed.includes('wasm32-unknown-unknown')) {
    execFileSync('rustup', ['target', 'add', 'wasm32-unknown-unknown'], { stdio: 'inherit' });
  }
} catch {
  skipOrFail('wasm32-unknown-unknown target not installed and could not be added (rustup unavailable?)');
}

// ---------------------------------------------------------------------------
// Build both variants
// ---------------------------------------------------------------------------

function buildVariant(label, rustflags, outDir) {
  console.error(`=== bench_wasm_simd: building ${label} (RUSTFLAGS=${JSON.stringify(rustflags)}) ===`);
  execFileSync(
    'cargo',
    ['build', '--release', '--target', 'wasm32-unknown-unknown', '-p', 'lattice-embed', '--no-default-features', '--features', 'wasm'],
    { cwd: REPO, stdio: 'inherit', env: { ...process.env, RUSTFLAGS: rustflags } },
  );
  mkdirSync(outDir, { recursive: true });
  execFileSync(
    'wasm-bindgen',
    ['--target', 'nodejs', '--out-dir', outDir, path.join(REPO, 'target/wasm32-unknown-unknown/release/lattice_embed.wasm')],
    { stdio: 'inherit' },
  );
  return path.join(outDir, 'lattice_embed.js');
}

const baselineJs = buildVariant('baseline', '', path.join(REPO, 'target/bench-wasm-simd-baseline'));
const simd128Js = buildVariant('simd128', '-C target-feature=+simd128', path.join(REPO, 'target/bench-wasm-simd-simd128'));

const baseline = await import(path.resolve(baselineJs));
const simd128 = await import(path.resolve(simd128Js));

// ---------------------------------------------------------------------------
// Deterministic input generation (mulberry32, same generator as the parity
// test -- reproducible across runs given the same --seed).
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
  for (let i = 0; i < dim; i++) v[i] = rng() * 2 - 1;
  return v;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

function percentile(sortedNs, p) {
  const idx = Math.min(sortedNs.length - 1, Math.floor(sortedNs.length * p));
  return sortedNs[idx];
}

// `prep`, when given, runs once per rep (warmup reps included) immediately
// BEFORE `t0` -- for ops like `normalize` that mutate their input in place
// and need a fresh copy each call, this keeps the reset out of the timed
// region instead of folding it into the measured latency.
function timeCalls(fn, reps, warmup, prep) {
  for (let i = 0; i < warmup; i++) {
    if (prep) prep();
    fn();
  }
  const samples = new Float64Array(reps);
  for (let i = 0; i < reps; i++) {
    if (prep) prep();
    const t0 = process.hrtime.bigint();
    fn();
    const t1 = process.hrtime.bigint();
    samples[i] = Number(t1 - t0);
  }
  const sorted = Array.from(samples).sort((a, b) => a - b);
  return { median: percentile(sorted, 0.5), p95: percentile(sorted, 0.95) };
}

function benchPairOp(mod, opName, dim, seed, reps, warmup) {
  const a = randomVec(dim, seed);
  const b = randomVec(dim, seed + 1);
  const fn = () => mod[opName](a, b);
  return timeCalls(fn, reps, warmup);
}

function benchNormalize(mod, dim, seed, reps, warmup) {
  const template = randomVec(dim, seed);
  const scratch = new Float32Array(dim);
  const prep = () => scratch.set(template);
  const fn = () => mod.simdNormalize(scratch);
  return timeCalls(fn, reps, warmup, prep);
}

const OPS = [
  { name: 'dot_product', fn: (mod, dim, seed, reps, warmup) => benchPairOp(mod, 'simdDotProduct', dim, seed, reps, warmup) },
  { name: 'squared_l2', fn: (mod, dim, seed, reps, warmup) => benchPairOp(mod, 'simdSquaredEuclideanDistance', dim, seed, reps, warmup) },
  { name: 'cosine', fn: (mod, dim, seed, reps, warmup) => benchPairOp(mod, 'simdCosineSimilarity', dim, seed, reps, warmup) },
  { name: 'normalize', fn: (mod, dim, seed, reps, warmup) => benchNormalize(mod, dim, seed, reps, warmup) },
];

// ---------------------------------------------------------------------------
// Run + report
// ---------------------------------------------------------------------------

console.log('# wasm32 SIMD128 A/B bench: lattice-embed simd:: kernels');
console.log('#');
console.log(`# node: ${process.version}`);
console.log(`# platform: ${os.platform()} ${os.arch()} ${os.release()}`);
console.log(`# rustc: ${execFileSync('rustc', ['--version']).toString().trim()}`);
console.log(`# wasm-bindgen: ${execFileSync('wasm-bindgen', ['--version']).toString().trim()}`);
console.log('# baseline build: cargo build --release --target wasm32-unknown-unknown -p lattice-embed --no-default-features --features wasm');
console.log('# simd128 build:  RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown -p lattice-embed --no-default-features --features wasm');
console.log(`# dims: ${ARGS.dims.join(',')}`);
console.log(`# reps: ${ARGS.reps}, warmup: ${ARGS.warmup}, seed: ${ARGS.seed}`);
console.log('#');
console.log('| op | dim | baseline median (ns) | baseline p95 (ns) | simd128 median (ns) | simd128 p95 (ns) | median speedup |');
console.log('|---|---|---|---|---|---|---|');

for (const dim of ARGS.dims) {
  for (const op of OPS) {
    const b = op.fn(baseline, dim, ARGS.seed, ARGS.reps, ARGS.warmup);
    const s = op.fn(simd128, dim, ARGS.seed, ARGS.reps, ARGS.warmup);
    const speedup = b.median / s.median;
    console.log(
      `| ${op.name} | ${dim} | ${b.median.toFixed(1)} | ${b.p95.toFixed(1)} | ${s.median.toFixed(1)} | ${s.p95.toFixed(1)} | ${speedup.toFixed(2)}x |`,
    );
  }
}
