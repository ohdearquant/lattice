// wasm embedding parity gate.
//
// Loads the wasm-bindgen (`--target nodejs`) build of `lattice-embed`'s
// `LatticeEmbedder` binding and checks its output two ways:
//
//   1. wasm-vs-golden: the same committed HF-reference goldens used by the
//      native `embed_parity_vs_hf.rs` test (`bge_small_en_v15.json`,
//      `all_minilm_l6_v2.json`). Thresholds start from the native test's own
//      `COS_SIM_MIN_F32` / `MAX_ABS_DIFF_F32` constants.
//   2. wasm-vs-native: the same inputs run through `NativeEmbeddingService`
//      (dumped ahead of time by `dump_parity_embeddings` into a JSON file),
//      compared at a tighter tolerance. This is the check that isolates a
//      wasm-only regression (codegen, libm, tokenizer-construction
//      asymmetry) from an ordinary HF-vs-lattice numerical gap.
//
// A third input, `long_input_case.json`, has no HF golden at all: it is a
// natural-language paragraph that tokenizes to 429 WordPiece tokens for
// all-MiniLM-L6-v2 (above the model catalog's advisory 256-token cap, below
// the model's real 512-token position-table limit). It is checked only
// wasm-vs-native, at the same tight tolerance as check 2. This is the
// mutation-sensitive case: if the wasm and native code paths ever compute a
// different effective truncation length for a long input, this is where it
// would show up first, and nowhere in the short-input goldens.
//
// The golden JSON files remain the single source of truth for embedding
// vectors; this script never hardcodes a reference vector.
//
// Run via `scripts/wasm-parity.sh` (skip-graceful / fail-closed discipline
// lives there) or `make wasm-parity`. Exits non-zero on any check failure.

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..', '..', '..');
const FIXTURE_DIR = path.join(REPO_ROOT, 'crates', 'embed', 'tests', 'fixtures', 'embed_parity_v1');

// ---------------------------------------------------------------------------
// Tolerance constants: mirror crates/embed/tests/embed_parity_vs_hf.rs.
// ---------------------------------------------------------------------------

// wasm-vs-golden (HF reference): starting point taken directly from the
// native parity test's COS_SIM_MIN_F32 / MAX_ABS_DIFF_F32. Measured wasm
// numbers (2026-07, bge-small-en-v1.5 + all-MiniLM-L6-v2, 5 goldens each):
// min cosine 0.999809, max abs diff 3.546e-3; inside these thresholds with
// headroom, so no widening was needed.
const COS_SIM_MIN_F32 = 0.9990;
const MAX_ABS_DIFF_F32 = 5e-3;

// wasm-vs-native: both sides run the same lattice forward pass in f32, so
// this should be far tighter than the wasm-vs-HF gap (no cross-framework
// tokenizer/pooling differences, only wasm codegen / libm drift). Measured
// floor (2026-07): bge-small-en-v1.5 cosine 0.999956, all-MiniLM-L6-v2 short
// inputs cosine 0.999992, all-MiniLM-L6-v2 long-input (429 tokens) cosine
// 0.999985. Threshold is set with headroom below the measured floor.
const WASM_VS_NATIVE_COS_MIN = 0.9995;
const WASM_VS_NATIVE_MAX_ABS_DIFF = 3e-3;

const MODELS = [
  { label: 'bge_small_en_v15', modelDirName: 'bge-small-en-v1.5', useCls: true, goldenFile: 'bge_small_en_v15.json' },
  { label: 'all_minilm_l6_v2', modelDirName: 'all-minilm-l6-v2', useCls: false, goldenFile: 'all_minilm_l6_v2.json' },
];

// ---------------------------------------------------------------------------
// Helpers (mirror cosine_sim / max_abs_diff in embed_parity_vs_hf.rs exactly)
// ---------------------------------------------------------------------------

function cosineSim(a, b) {
  if (a.length !== b.length) {
    throw new Error(`dimension mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA < 1e-9 || normB < 1e-9) {
    return 0.0;
  }
  return dot / (normA * normB);
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    m = Math.max(m, Math.abs(a[i] - b[i]));
  }
  return m;
}

function readJson(p) {
  return JSON.parse(readFileSync(p, 'utf8'));
}

let failures = 0;
let checks = 0;

function check(kind, label, wasmVec, refVec, cosMin, maxDiffMax) {
  checks++;
  const cos = cosineSim(wasmVec, refVec);
  const diff = maxAbsDiff(wasmVec, refVec);
  const pass = cos >= cosMin && diff <= maxDiffMax;
  if (!pass) {
    failures++;
    console.error(
      `PARITY FAIL [${kind}] ${label}\n  cosine=${cos.toFixed(6)} (need >= ${cosMin})\n  max_abs_diff=${diff.toExponential(3)} (need <= ${maxDiffMax})`,
    );
  } else {
    console.log(`  [${kind}] ${label} cosine=${cos.toFixed(6)} max_abs_diff=${diff.toExponential(3)} PASS`);
  }
  return { cos, diff, pass };
}

// ---------------------------------------------------------------------------
// Resolve required inputs
// ---------------------------------------------------------------------------

const wasmJsPath = process.env.LATTICE_WASM_JS;
if (!wasmJsPath || !existsSync(wasmJsPath)) {
  console.error(
    `embed_parity_wasm.mjs: LATTICE_WASM_JS not set or file missing (got ${wasmJsPath}). ` +
      'This script expects the wasm-bindgen nodejs build to already exist; run it via scripts/wasm-parity.sh.',
  );
  process.exit(1);
}

const modelsDir = process.env.LATTICE_MODELS_DIR || path.join(process.env.HOME, '.lattice', 'models');

const nativeDumpPath = process.env.LATTICE_NATIVE_DUMP;
if (!nativeDumpPath || !existsSync(nativeDumpPath)) {
  console.error(
    `embed_parity_wasm.mjs: LATTICE_NATIVE_DUMP not set or file missing (got ${nativeDumpPath}). ` +
      'This script expects a native reference dump (dump_parity_embeddings) to already exist; run it via scripts/wasm-parity.sh.',
  );
  process.exit(1);
}
const nativeDump = readJson(nativeDumpPath);

const { LatticeEmbedder } = await import(path.resolve(wasmJsPath));

// ---------------------------------------------------------------------------
// Part 1 + 2: per-model goldens, wasm-vs-golden and wasm-vs-native
// ---------------------------------------------------------------------------

for (const modelSpec of MODELS) {
  const modelDir = path.join(modelsDir, modelSpec.modelDirName);
  const modelBytes = new Uint8Array(readFileSync(path.join(modelDir, 'model.safetensors')));
  const configBytes = new Uint8Array(readFileSync(path.join(modelDir, 'config.json')));
  const tokenizerBytes = new Uint8Array(readFileSync(path.join(modelDir, 'tokenizer.json')));

  const embedder = new LatticeEmbedder(modelBytes, configBytes, tokenizerBytes);
  if (modelSpec.useCls) {
    embedder.useClsPooling();
  }

  const goldens = readJson(path.join(FIXTURE_DIR, modelSpec.goldenFile));
  const nativeVecs = nativeDump[modelSpec.label];
  if (!nativeVecs || nativeVecs.length !== goldens.length) {
    throw new Error(
      `native dump missing or size mismatch for ${modelSpec.label}: expected ${goldens.length} vectors`,
    );
  }

  let minCosGolden = 1.0;
  let maxDiffGolden = 0.0;
  let minCosNative = 1.0;
  let maxDiffNative = 0.0;

  goldens.forEach((golden, i) => {
    const wasmVec = Array.from(embedder.embed(golden.input));
    if (wasmVec.length !== golden.embedding_dim) {
      failures++;
      console.error(
        `PARITY FAIL [dim] ${modelSpec.label} input=${JSON.stringify(golden.input)}: got ${wasmVec.length}, want ${golden.embedding_dim}`,
      );
      return;
    }

    const golderResult = check(
      'wasm-vs-golden',
      `${modelSpec.label} '${golden.input.slice(0, 40)}'`,
      wasmVec,
      golden.embedding,
      COS_SIM_MIN_F32,
      MAX_ABS_DIFF_F32,
    );
    minCosGolden = Math.min(minCosGolden, golderResult.cos);
    maxDiffGolden = Math.max(maxDiffGolden, golderResult.diff);

    const nativeResult = check(
      'wasm-vs-native',
      `${modelSpec.label} '${golden.input.slice(0, 40)}'`,
      wasmVec,
      nativeVecs[i],
      WASM_VS_NATIVE_COS_MIN,
      WASM_VS_NATIVE_MAX_ABS_DIFF,
    );
    minCosNative = Math.min(minCosNative, nativeResult.cos);
    maxDiffNative = Math.max(maxDiffNative, nativeResult.diff);
  });

  console.log(
    `[${modelSpec.label}] aggregate wasm-vs-golden: min_cosine=${minCosGolden.toFixed(6)} max_abs_diff=${maxDiffGolden.toExponential(3)}`,
  );
  console.log(
    `[${modelSpec.label}] aggregate wasm-vs-native: min_cosine=${minCosNative.toFixed(6)} max_abs_diff=${maxDiffNative.toExponential(3)}`,
  );
}

// ---------------------------------------------------------------------------
// Part 3: long-input MiniLM stress case (wasm-vs-native only, no HF golden).
// This is the mutation-sensitive check: a from_bytes/from_directory
// truncation-length divergence for a long input would surface here, not in
// the short-input goldens above.
// ---------------------------------------------------------------------------

const longInputPath = path.join(FIXTURE_DIR, 'long_input_case.json');
if (existsSync(longInputPath)) {
  const longCase = readJson(longInputPath);
  const nativeLongVecs = nativeDump['all_minilm_l6_v2_long_input'];
  if (!nativeLongVecs || nativeLongVecs.length !== 1) {
    throw new Error('native dump missing all_minilm_l6_v2_long_input entry');
  }

  const modelDir = path.join(modelsDir, 'all-minilm-l6-v2');
  const modelBytes = new Uint8Array(readFileSync(path.join(modelDir, 'model.safetensors')));
  const configBytes = new Uint8Array(readFileSync(path.join(modelDir, 'config.json')));
  const tokenizerBytes = new Uint8Array(readFileSync(path.join(modelDir, 'tokenizer.json')));
  const embedder = new LatticeEmbedder(modelBytes, configBytes, tokenizerBytes);

  const wasmVec = Array.from(embedder.embed(longCase.input));
  check('wasm-vs-native long-input', 'all_minilm_l6_v2_long_input', wasmVec, nativeLongVecs[0], WASM_VS_NATIVE_COS_MIN, WASM_VS_NATIVE_MAX_ABS_DIFF);
} else {
  console.error(`embed_parity_wasm.mjs: long_input_case.json not found at ${longInputPath}. This is a required check, not optional.`);
  failures++;
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

console.log(`\nwasm parity gate: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error('wasm parity gate FAILED');
  process.exit(1);
}
console.log('wasm parity gate PASSED');
process.exit(0);
