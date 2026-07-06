#!/usr/bin/env node
// Smoke test: simulates a caller's activation probe (embed(text, model) as
// the first thing tried) against the built package, using model weights
// already cached locally under ~/.lattice/models (no network fetch tier
// exercised here, see resolve.mjs for that path). Run with:
//   npm run build && npm run smoke
//
// This is the thing that tells a real embedder apart from a hash
// placeholder: a placeholder can return a 384-length vector with norm 1.0
// too, but it cannot make "a dog runs in the park" land closer to "a puppy
// runs in the park" than to "quarterly financial report".

import assert from 'node:assert/strict';

import { embed } from './index.mjs';
import { MODEL_REGISTRY } from './registry.mjs';

function l2Norm(vec) {
  let sum = 0;
  for (const x of vec) sum += x * x;
  return Math.sqrt(sum);
}

function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

const MODELS = ['minilm', 'bge', 'paraphrase-minilm'];
let allPass = true;

async function checkModel(model) {
  const pooling = MODEL_REGISTRY[model].pooling;
  console.log(`\n=== ${model} (pooling=${pooling}) ===`);

  const vec = await embed('the cat sat on the mat', model);
  assert.ok(vec !== null, `${model}: embed() returned null`);
  assert.ok(vec instanceof Float32Array, `${model}: result is not a Float32Array`);
  assert.equal(vec.length, 384, `${model}: expected 384 dims, got ${vec.length}`);
  const norm = l2Norm(vec);
  console.log(`  dims=${vec.length} norm=${norm.toFixed(6)}`);
  assert.ok(Math.abs(norm - 1.0) < 1e-3, `${model}: L2 norm not ~1.0 (got ${norm})`);

  const dog = await embed('a dog runs in the park', model);
  const puppy = await embed('a puppy runs in the park', model);
  const report = await embed('quarterly financial report', model);
  const cosNear = cosine(dog, puppy);
  const cosFar = cosine(dog, report);
  console.log(`  cosine(dog, puppy)=${cosNear.toFixed(6)}  cosine(dog, report)=${cosFar.toFixed(6)}`);
  assert.ok(
    cosNear > cosFar,
    `${model}: semantic sanity check failed (dog~puppy=${cosNear} <= dog~report=${cosFar})`,
  );
  console.log(`  PASS`);
}

for (const model of MODELS) {
  try {
    // eslint-disable-next-line no-await-in-loop
    await checkModel(model);
  } catch (err) {
    allPass = false;
    console.error(`FAIL [${model}]: ${err.message}`);
  }
}

console.log('\n=== degrade check: qwen3-0.6b (not a wasm-supported model) ===');
const qwenResult = await embed('x', 'qwen3-0.6b');
console.log(`  embed('x', 'qwen3-0.6b') = ${qwenResult}`);
if (qwenResult !== null) {
  allPass = false;
  console.error('FAIL: qwen3-0.6b did not degrade to null');
} else {
  console.log('  PASS');
}

console.log(`\nsmoke test: ${allPass ? 'PASS' : 'FAIL'}`);
process.exit(allPass ? 0 : 1);
