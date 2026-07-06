#!/usr/bin/env node
// Regression test for the manual-init race: a caller that starts the
// exported async `init()` with an argument whose resolution is delayed,
// then calls `embed()` before that `init()` settles, must never end up with
// two wasm instances (one bound to a stale cached embedder, one live). See
// index.mjs's comment above `ensureWasmInit` for why the raw wasm-bindgen
// async `init` cannot be used here: its finalize step runs after an await,
// so a second, unrelated synchronous init landing in between silently
// swaps the module-global instance out from under an already-constructed
// embedder, and the next `embed()` call on it throws a wasm
// "memory access out of bounds" trap. Run with:
//   npm run build && npm run test:init

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { init, embed } from './index.mjs';

function l2Norm(vec) {
  let sum = 0;
  for (const x of vec) sum += x * x;
  return Math.sqrt(sum);
}

const wasmBytes = readFileSync(new URL('./wasm/lattice_embed_bg.wasm', import.meta.url));

let releaseManualInit;
const manualInit = init({
  module_or_path: new Promise((resolve) => {
    releaseManualInit = () => resolve(wasmBytes);
  }),
});

const first = await embed('manual init race probe one', 'minilm');
assert.ok(first !== null, 'first embed() returned null');
assert.ok(first instanceof Float32Array, 'first result is not a Float32Array');
assert.equal(first.length, 384, `first: expected 384 dims, got ${first.length}`);
assert.ok(Math.abs(l2Norm(first) - 1.0) < 1e-3, `first: L2 norm not ~1.0 (got ${l2Norm(first)})`);

// Only now let the manual init's argument resolve, and wait for it to
// settle. Before the fix, this is the moment the raw async init's finalize
// step ran and swapped the module-global wasm instance out from under the
// embedder already constructed and cached above.
releaseManualInit();
await manualInit;

const second = await embed('manual init race probe two', 'minilm');
assert.ok(second !== null, 'second embed() returned null');
assert.ok(second instanceof Float32Array, 'second result is not a Float32Array');
assert.equal(second.length, 384, `second: expected 384 dims, got ${second.length}`);
assert.ok(Math.abs(l2Norm(second) - 1.0) < 1e-3, `second: L2 norm not ~1.0 (got ${l2Norm(second)})`);

// Same text through the same model should embed identically regardless of
// which side of the manual init it landed on.
const third = await embed('manual init race probe two', 'minilm');
assert.deepEqual(Array.from(second), Array.from(third), 'repeat embed of the same text diverged');

console.log('init race test: PASS');
