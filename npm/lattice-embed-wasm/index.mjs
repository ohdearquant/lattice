// lattice-embed-wasm: a Node adapter around a pure-Rust BERT-family text
// embedding engine compiled to WebAssembly.
//
// The wasm core (./wasm/) is bytes-in, bytes-out and knows nothing about
// model names, file systems, or the network: it only knows how to load one
// model from raw bytes and embed strings with it (see wasm/lattice_embed.d.ts
// and the constructor doc comment there). This file is the thin layer on
// top: it resolves a model NAME to bytes (registry.mjs + resolve.mjs),
// memoizes one wasm instance per model, and applies the pooling strategy
// each model family expects.
//
// Two failure modes are handled deliberately differently:
//   - a model name that is unknown, or a model whose weights cannot be
//     resolved and verified (see resolve.mjs), returns `null`. This is a
//     degrade signal, not an error: a caller trying several embedding tiers
//     falls through to its next option.
//   - an error raised while actually running an already-loaded model
//     (a bug, a malformed model file that passed hash verification, ...) is
//     thrown, not swallowed, because it means something is broken rather
//     than "this option isn't available".

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import init, { initSync, initPanicHook, LatticeEmbedder } from './wasm/lattice_embed.js';
import { DEFAULT_MODEL, MODEL_REGISTRY, UNSUPPORTED_MODELS } from './registry.mjs';
import { resolveModelBytes } from './resolve.mjs';

export { initSync, init };

// wasm instantiation itself is synchronous (initSync loads local bytes; no
// fetch involved), so this needs no promise, just an idempotency guard.
let wasmInitialized = false;

function ensureWasmInit() {
  if (wasmInitialized) return;
  const wasmPath = fileURLToPath(new URL('./wasm/lattice_embed_bg.wasm', import.meta.url));
  const wasmBytes = readFileSync(wasmPath);
  initSync({ module: wasmBytes });
  initPanicHook();
  wasmInitialized = true;
}

// modelName -> Promise<LatticeEmbedder | null>, one entry per model actually
// requested. A failed resolution is evicted after settling so a transient
// failure (e.g. a network blip on the fetch tier) can be retried by a later
// call instead of being remembered as permanent.
const embedderCache = new Map();

async function buildEmbedder(modelName) {
  const entry = MODEL_REGISTRY[modelName];
  ensureWasmInit();
  const bytes = await resolveModelBytes(entry);
  if (!bytes) {
    console.error(`lattice-embed-wasm: could not resolve verified weights for model "${modelName}"`);
    return null;
  }
  const embedder = new LatticeEmbedder(bytes.model, bytes.config, bytes.tokenizer);
  if (entry.pooling === 'cls') {
    embedder.useClsPooling();
  }
  return embedder;
}

async function getEmbedder(modelName) {
  if (Object.prototype.hasOwnProperty.call(UNSUPPORTED_MODELS, modelName)) {
    console.error(
      `lattice-embed-wasm: model "${modelName}" is not supported over wasm: ${UNSUPPORTED_MODELS[modelName]}`,
    );
    return null;
  }
  if (!Object.prototype.hasOwnProperty.call(MODEL_REGISTRY, modelName)) {
    console.error(`lattice-embed-wasm: unknown model "${modelName}"`);
    return null;
  }

  if (!embedderCache.has(modelName)) {
    embedderCache.set(modelName, buildEmbedder(modelName));
  }
  const result = await embedderCache.get(modelName);
  if (result === null) {
    embedderCache.delete(modelName);
  }
  return result;
}

/**
 * Embed `text` with `model` (default `'minilm'`), returning an L2-normalized
 * `Float32Array`, or `null` if `model` is unknown, unsupported over wasm
 * (e.g. `'qwen3-0.6b'`), or its weights could not be resolved and verified.
 * Self-initializes the wasm core on first call.
 *
 * @param {string} text
 * @param {string} [model]
 * @returns {Promise<Float32Array | null>}
 */
export async function embed(text, model = DEFAULT_MODEL) {
  const embedder = await getEmbedder(model);
  if (!embedder) return null;
  return embedder.embed(text);
}

/**
 * Embed `text` with the default model. Equivalent to `embed(text)`.
 *
 * @param {string} text
 * @returns {Promise<Float32Array | null>}
 */
export async function embedText(text) {
  return embed(text, DEFAULT_MODEL);
}

/**
 * A small convenience wrapper that pins a model name so repeated calls don't
 * need to repeat it. Shares the same module-level memoized wasm instances
 * as `embed()`: constructing more than one `Embedder` for the same model
 * name does not load the model twice.
 */
export class Embedder {
  constructor(model = DEFAULT_MODEL) {
    this.model = model;
  }

  /**
   * @param {string} text
   * @returns {Promise<Float32Array | null>}
   */
  async embed(text) {
    return embed(text, this.model);
  }
}
