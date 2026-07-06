// Resolves a model's three source files (model weights, config, tokenizer)
// to bytes, in priority order:
//
//   (a) explicit local-dir override: LATTICE_EMBED_MODELS_DIR, or
//       LATTICE_MODEL_CACHE (the cache-directory env var the native
//       lattice-embed CLI reads, see crates/inference/src/lib.rs's
//       `default_cache_dir`), or ~/.lattice/models by default. This tier is
//       wasm-specific: it requires model.safetensors, config.json, AND
//       tokenizer.json in the model's subdirectory. A native download of a
//       WordPiece model (BGE/MiniLM) only fetches vocab.txt, not
//       tokenizer.json (see crates/inference/src/download.rs), so a
//       native-CLI cache populated that way is not directly usable here
//       unless a tokenizer.json is also present; this package does not
//       synthesize one from vocab.txt.
//   (b) this package's own on-disk cache (~/.cache/lattice-embed-wasm/models
//       by default, override with LATTICE_EMBED_WASM_CACHE_DIR).
//   (c) fetch from the pinned GitHub release asset URL in the registry and
//       verify sha256 before use; a verified download is written into the
//       cache directory from (b) so the next resolution is a cache hit.
//       Skipped entirely while `WEIGHTS_RELEASE_TAG` (registry.mjs) is
//       unset; see that file's comment for why.
//
// Every candidate is checked against the registry's pinned sha256 before its
// bytes are handed back. A MISSING pinned hash for any file an entry
// declares is itself a verification failure, never a skip; see
// registry.mjs, which refuses to register a model entry at module load if
// it lacks a pinned hash for one of its declared files, so `entry.sha256`
// is guaranteed complete by the time it reaches this module; this file's
// own per-file check is a second, independent guard against ever treating
// an unpinned file as verified. A candidate that fails verification is
// discarded (not used, not trusted) and resolution continues to the next
// tier; if every tier is exhausted without a verified candidate, resolution
// fails closed: the caller gets `null`; it never gets a wrong or partial
// vector.

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

// Checks `actual` (the sha256 hex of a candidate file's bytes) against the
// registry's pinned hash for `fileName`. A MISSING pinned hash is treated as
// a verification FAILURE, never a silent skip: an entry reaching this
// function is expected to have already passed the module-load registry
// validation (see registry.mjs) that guarantees a pinned hash for every
// declared file, so a missing hash here would mean that guarantee was
// bypassed somehow -- fail closed rather than trust unverified bytes on any
// path (local override, cache, or download). Shared by both verify sites
// below so the fail-closed rule can't drift between them.
function verifyHash(fileName, actual, entry, sourceLabel) {
  const expected = entry.sha256[fileName];
  if (!expected) {
    console.error(
      `lattice-embed-wasm: no pinned sha256 for ${fileName} in the registry; refusing to ` +
        `trust ${fileName} from ${sourceLabel} without a pinned hash`,
    );
    return false;
  }
  if (actual !== expected) {
    console.error(
      `lattice-embed-wasm: sha256 mismatch for ${fileName} from ${sourceLabel} ` +
        `(expected ${expected}, got ${actual}); discarding this candidate`,
    );
    return false;
  }
  return true;
}

// Local-override search directories, in priority order: an explicit wasm-
// specific override, then the native lattice-embed CLI's cache-directory
// env var (so a directory already populated by the native CLI is checked
// too, though see the module doc comment above about which files it needs
// to actually contain), then the shared default both env vars fall back to.
// Deduplicated so a shared default or identically-set env vars aren't
// checked twice.
function localModelsDirs() {
  const dirs = [];
  const fromEmbedVar = process.env.LATTICE_EMBED_MODELS_DIR;
  const fromNativeVar = process.env.LATTICE_MODEL_CACHE;
  if (fromEmbedVar && !dirs.includes(fromEmbedVar)) dirs.push(fromEmbedVar);
  if (fromNativeVar && !dirs.includes(fromNativeVar)) dirs.push(fromNativeVar);
  const fallback = path.join(homedir(), '.lattice', 'models');
  if (!dirs.includes(fallback)) dirs.push(fallback);
  return dirs;
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
    const actual = sha256Hex(bytes[key]);
    if (!verifyHash(fileName, actual, entry, sourceLabel)) return null;
  }

  return bytes;
}

// Module-level guard so the "remote-fetch tier skipped" notice is emitted
// once per process, not once per resolution attempt (a caller may try many
// models across a session; the reason is the same every time).
let fetchTierSkipWarned = false;

// True only if every declared file has a release-asset URL configured (see
// registry.mjs's `releaseUrl`, which returns null for every asset while
// `WEIGHTS_RELEASE_TAG` is unset). Used to skip the fetch tier entirely
// rather than build a request against a release that does not exist.
function fetchTierConfigured(entry) {
  return Object.values(entry.files).every((fileName) => Boolean(entry.releaseAssets[fileName]));
}

async function fetchAndVerify(entry) {
  if (!fetchTierConfigured(entry)) {
    if (!fetchTierSkipWarned) {
      fetchTierSkipWarned = true;
      console.warn(
        'lattice-embed-wasm: remote-fetch tier skipped -- WEIGHTS_RELEASE_TAG (registry.mjs) is ' +
          'unset because the model weight release assets have not been published yet; a model ' +
          'not found via the local override or cache tier resolves to null until that publish ' +
          'step lands and the tag is set',
      );
    }
    return null;
  }

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
    const actual = sha256Hex(buf);
    if (!verifyHash(fileName, actual, entry, `download (${url})`)) return null;
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
  for (const dir of localModelsDirs()) {
    const localDir = path.join(dir, entry.localDir);
    const fromLocal = readAndVerify(localDir, entry, `local override (${localDir})`);
    if (fromLocal) return fromLocal;
  }

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
