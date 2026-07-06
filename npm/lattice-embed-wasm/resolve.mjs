// Resolves a model's three source files (model weights, config, tokenizer)
// to bytes, in priority order:
//
//   (a) explicit local-dir override: LATTICE_EMBED_MODELS_DIR (or
//       ~/.lattice/models by default), the directory layout the native
//       lattice-embed CLI already uses, so a machine that has run that CLI
//       has these files for free.
//   (b) this package's own on-disk cache (~/.cache/lattice-embed-wasm/models
//       by default, override with LATTICE_EMBED_WASM_CACHE_DIR).
//   (c) fetch from the pinned GitHub release asset URL in the registry and
//       verify sha256 before use; a verified download is written into the
//       cache directory from (b) so the next resolution is a cache hit.
//
// Every candidate is checked against the registry's pinned sha256 before its
// bytes are handed back. A candidate that fails verification is discarded
// (not used, not trusted) and resolution continues to the next tier; if
// every tier is exhausted without a verified candidate, resolution fails
// closed: the caller gets `null`; it never gets a wrong or partial vector.

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function localModelsDir() {
  return process.env.LATTICE_EMBED_MODELS_DIR || path.join(homedir(), '.lattice', 'models');
}

function cacheModelsDir() {
  return (
    process.env.LATTICE_EMBED_WASM_CACHE_DIR ||
    path.join(homedir(), '.cache', 'lattice-embed-wasm', 'models')
  );
}

// Reads {model, config, tokenizer} from `dir` using the registry entry's
// file names, verifies each against `entry.sha256`, and returns the bytes
// only if every file is present and every hash matches. Returns null
// otherwise (missing directory, missing file, or hash mismatch); the caller
// decides what to do next (try another tier, or give up).
function readAndVerify(dir, entry, sourceLabel) {
  const names = entry.files;
  const paths = {
    model: path.join(dir, names.model),
    config: path.join(dir, names.config),
    tokenizer: path.join(dir, names.tokenizer),
  };

  for (const key of Object.keys(paths)) {
    if (!existsSync(paths[key])) return null;
  }

  const bytes = {
    model: new Uint8Array(readFileSync(paths.model)),
    config: new Uint8Array(readFileSync(paths.config)),
    tokenizer: new Uint8Array(readFileSync(paths.tokenizer)),
  };

  for (const [key, fileName] of Object.entries(names)) {
    const expected = entry.sha256[fileName];
    const actual = sha256Hex(bytes[key]);
    if (expected && actual !== expected) {
      console.error(
        `lattice-embed-wasm: sha256 mismatch for ${fileName} from ${sourceLabel} ` +
          `(expected ${expected}, got ${actual}); discarding this candidate`,
      );
      return null;
    }
  }

  return bytes;
}

async function fetchAndVerify(entry) {
  const names = entry.files;
  const bytes = {};

  for (const [key, fileName] of Object.entries(names)) {
    const url = entry.releaseAssets[fileName];
    let response;
    try {
      response = await fetch(url);
    } catch (err) {
      console.error(`lattice-embed-wasm: fetch failed for ${url}: ${err.message}`);
      return null;
    }
    if (!response.ok) {
      console.error(`lattice-embed-wasm: fetch for ${url} returned HTTP ${response.status}`);
      return null;
    }
    const buf = new Uint8Array(await response.arrayBuffer());
    const expected = entry.sha256[fileName];
    const actual = sha256Hex(buf);
    if (expected && actual !== expected) {
      console.error(
        `lattice-embed-wasm: sha256 mismatch for downloaded ${fileName} ` +
          `(expected ${expected}, got ${actual}); discarding download`,
      );
      return null;
    }
    bytes[key] = buf;
  }

  return bytes;
}

function writeToCache(entry, bytes) {
  const dir = path.join(cacheModelsDir(), entry.localDir);
  mkdirSync(dir, { recursive: true });
  writeFileSync(path.join(dir, entry.files.model), bytes.model);
  writeFileSync(path.join(dir, entry.files.config), bytes.config);
  writeFileSync(path.join(dir, entry.files.tokenizer), bytes.tokenizer);
}

// Resolves `entry` (a MODEL_REGISTRY value) to verified {model, config,
// tokenizer} byte arrays, or null if no tier produced a verified result.
export async function resolveModelBytes(entry) {
  const localDir = path.join(localModelsDir(), entry.localDir);
  const fromLocal = readAndVerify(localDir, entry, `local override (${localDir})`);
  if (fromLocal) return fromLocal;

  const cacheDir = path.join(cacheModelsDir(), entry.localDir);
  const fromCache = readAndVerify(cacheDir, entry, `cache (${cacheDir})`);
  if (fromCache) return fromCache;

  const fromFetch = await fetchAndVerify(entry);
  if (fromFetch) {
    writeToCache(entry, fromFetch);
    return fromFetch;
  }

  return null;
}
