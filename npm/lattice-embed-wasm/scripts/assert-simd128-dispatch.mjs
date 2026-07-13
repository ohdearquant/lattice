#!/usr/bin/env node
// Post-build guard for build-wasm.sh: loads the wasm-bindgen bindings that
// script just generated and asserts the built artifact is actually
// dispatching to the wasm32 SIMD128 kernels, not silently falling back to
// scalar.
//
// This exists because a stale or misconfigured build can produce bindings
// that look entirely normal (same exports, same API surface, builds
// without error) while the underlying .wasm was never actually built with
// `-C target-feature=+simd128` in effect on the consumed artifact -- e.g. a
// shared CARGO_TARGET_DIR pointing wasm-bindgen at an old binary, or an
// inherited CARGO_ENCODED_RUSTFLAGS silently overriding the flag (see
// build-wasm.sh's own guard for that case). `simdSimd128Dispatch()`
// (crates/embed/src/wasm.rs, exported via wasm/lattice_embed.js) reports
// the crate's own dispatch decision (`SimdConfig::simd128_enabled()`), the
// same probe crates/embed/tests/wasm/simd128_parity_wasm.mjs checks before
// trusting any numeric comparison in that crate-level parity gate -- this
// script applies the identical check to the actual npm package artifact
// right after build-wasm.sh produces it.
//
// Usage: node assert-simd128-dispatch.mjs <out-dir>
//   <out-dir> must contain lattice_embed.js + lattice_embed_bg.wasm, i.e.
//   the --out-dir wasm-bindgen was just pointed at.

import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const outDir = process.argv[2];
if (!outDir) {
  console.error('assert-simd128-dispatch: usage: node assert-simd128-dispatch.mjs <out-dir>');
  process.exit(1);
}

const jsPath = path.join(outDir, 'lattice_embed.js');
const wasmPath = path.join(outDir, 'lattice_embed_bg.wasm');

const bindings = await import(pathToFileURL(jsPath).href);
const wasmBytes = readFileSync(wasmPath);
// Same init call shape index.mjs uses for the shipped package: synchronous,
// bundled-bytes init via the `{ module }` form (see `initSync` in
// lattice_embed.js) -- no fetch, no network, exactly one instance.
bindings.initSync({ module: wasmBytes });

const dispatch = bindings.simdSimd128Dispatch();
console.log(`assert-simd128-dispatch: simdSimd128Dispatch() = ${dispatch}`);
if (dispatch !== true) {
  console.error(
    'assert-simd128-dispatch: FAIL - the built artifact is not dispatching to the wasm32 SIMD128 kernels ' +
      '(simdSimd128Dispatch() returned false). This means the .wasm wasm-bindgen just consumed was not actually ' +
      'built with -C target-feature=+simd128 in effect -- a stale artifact from a mismatched target directory, ' +
      'or a RUSTFLAGS override, are the two known causes (see build-wasm.sh).',
  );
  process.exit(1);
}
console.log('assert-simd128-dispatch: PASS (simd128 dispatch confirmed)');
