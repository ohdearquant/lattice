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

import { initSync as rawInitSync, initPanicHook, LatticeEmbedder } from './wasm/lattice_embed.js';
import { DEFAULT_MODEL, MODEL_REGISTRY, UNSUPPORTED_MODELS } from './registry.mjs';
import { resolveModelBytes } from './resolve.mjs';

// The wasm bytes are bundled with this package (wasm/lattice_embed_bg.wasm,
// see package.json "files") and never fetched over the network, so loading
// them is inherently synchronous and deterministic: there is exactly one
// wasm instance for this module's lifetime. The raw wasm-bindgen glue also
// exports an async `init` meant for fetch-based (browser) loading; this
// package deliberately never imports or calls it, because its finalize step
// runs after an internal await and can land after a second, unrelated
// initSync has already installed a different instance, silently swapping
// the module-global instance so a cached LatticeEmbedder's pointer no
// longer matches. Every init path below funnels through the single guard in
// `ensureWasmInit`, so that race cannot occur.
let wasmInitialized = false;

function ensureWasmInit() {
  if (wasmInitialized) return;
  const wasmPath = fileURLToPath(new URL('./wasm/lattice_embed_bg.wasm', import.meta.url));
  const wasmBytes = readFileSync(wasmPath);
  rawInitSync({ module: wasmBytes });
  initPanicHook();
  wasmInitialized = true;
}

/**
 * Initialize the wasm core synchronously from the bytes bundled with this
 * package. Idempotent: calling this more than once, or calling it after the
 * core has already auto-initialized on first `embed()`, is a no-op and
 * never creates a second wasm instance. Accepts and ignores an argument for
 * call-shape compatibility with the raw wasm-bindgen `initSync`; this
 * package always loads its own bundled bytes, never a caller-supplied
 * module.
 *
 * @param {unknown} [_input]
 */
export function initSync(_input) {
  ensureWasmInit();
}

/**
 * Initialize the wasm core. Exists for call-shape compatibility with the
 * raw wasm-bindgen async `init` (including the `{ module_or_path }`
 * convention some callers pass), but performs the same synchronous,
 * idempotent, bundled-bytes init as `initSync` underneath: this package's
 * wasm bytes ship locally, so there is never a fetch to await and never
 * more than one instance. Safe to call concurrently with `embed()` or with
 * itself.
 *
 * @param {unknown} [_input]
 * @returns {Promise<void>}
 */
export async function init(_input) {
  ensureWasmInit();
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
