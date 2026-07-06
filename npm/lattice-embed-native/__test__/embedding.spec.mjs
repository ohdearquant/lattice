// Node test-runner suite against the REAL native binding (no mock). Loads
// MiniLM and BGE once each from local model directories under
// ~/.lattice/models and exercises sync/async single + batch embed, input
// validation error codes, and a concurrency-determinism check.
//
// One test in this file is explicitly MUTATION-SENSITIVE (see the test
// named "...mutation-sensitive" below): its assertions are only satisfiable
// because src/lib.rs's `encode_many` derives `EmbeddingBatch.dimensions`
// from the same value used to size the flat buffer. See the delivery
// report for the actual revert -> fail -> restore -> pass demonstration
// (requires rebuilding the native binary between steps, so it is not
// re-run automatically here).
import assert from 'node:assert/strict'
import test from 'node:test'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { loadModel, loadModelSync, splitBatch } from '../index.js'

const MODELS_DIR = process.env.LATTICE_MODEL_CACHE || join(homedir(), '.lattice', 'models')
const MINILM_DIR = join(MODELS_DIR, 'all-minilm-l6-v2')
const BGE_DIR = join(MODELS_DIR, 'bge-small-en-v1.5')

// Input-size guardrails mirror crates/embed/src/service/mod.rs's
// MAX_TEXT_CHARS (32768, measured as UTF-8 byte length -- see
// src/lib.rs's validate_text) and DEFAULT_MAX_BATCH_SIZE (1000), enforced
// at the native binding boundary in validate_text/validate_texts BEFORE
// tokenization or model execution.
const MAX_TEXT_CHARS = 32768
const DEFAULT_MAX_BATCH_SIZE = 1000

const minilm = loadModelSync({ modelPath: MINILM_DIR })
const bge = loadModelSync({ modelPath: BGE_DIR })

function l2Norm(vec) {
  let sum = 0
  for (const x of vec) sum += x * x
  return Math.sqrt(sum)
}

test('loadModelSync infers family + pooling from directory basename (minilm)', () => {
  assert.equal(minilm.dimension, 384)
  assert.equal(minilm.normalized, true)
})

test('loadModelSync infers family + pooling from directory basename (bge)', () => {
  assert.equal(bge.dimension, 384)
  assert.equal(bge.normalized, true)
})

test('embedSync returns a normalized 384-dim Float32Array (minilm)', () => {
  const vec = minilm.embedSync('the cat sat on the mat')
  assert.ok(vec instanceof Float32Array)
  assert.equal(vec.length, 384)
  const norm = l2Norm(vec)
  assert.ok(norm >= 0.999 && norm <= 1.001, `norm out of range: ${norm}`)
})

test('embedSync returns a normalized 384-dim Float32Array (bge, CLS pooling)', () => {
  const vec = bge.embedSync('the cat sat on the mat')
  assert.ok(vec instanceof Float32Array)
  assert.equal(vec.length, 384)
  const norm = l2Norm(vec)
  assert.ok(norm >= 0.999 && norm <= 1.001, `norm out of range: ${norm}`)
})

test('embed (async) matches embedSync closely (minilm)', async () => {
  const text = 'a dog runs in the park'
  const syncVec = minilm.embedSync(text)
  const asyncVec = await minilm.embed(text)
  assert.equal(asyncVec.length, syncVec.length)
  for (let i = 0; i < syncVec.length; i++) {
    assert.ok(Math.abs(syncVec[i] - asyncVec[i]) < 1e-6, `component ${i} differs`)
  }
})

test('embedBatchSync flat buffer matches per-row vector views and splitBatch', () => {
  const texts = ['a dog runs in the park', 'a puppy runs in the park', 'quarterly financial report']
  const batch = minilm.embedBatchSync(texts)
  assert.equal(batch.rows, 3)
  assert.equal(batch.dimensions, 384)

  const rowsFromVector = texts.map((_, i) => batch.vector(i))
  const rowsFromSplit = splitBatch(batch)
  assert.equal(rowsFromSplit.length, 3)
  for (let i = 0; i < 3; i++) {
    assert.deepEqual(Array.from(rowsFromVector[i]), Array.from(rowsFromSplit[i]))
    // Each row must independently match a direct embedSync of the same text
    // (batch path and single-item path must agree).
    const single = minilm.embedSync(texts[i])
    for (let d = 0; d < 384; d++) {
      assert.ok(Math.abs(rowsFromVector[i][d] - single[d]) < 1e-4, `row ${i} dim ${d} mismatch`)
    }
  }
})

test('embedBatch (async) agrees with embedBatchSync', async () => {
  const texts = ['a dog runs in the park', 'a puppy runs in the park']
  const syncBatch = minilm.embedBatchSync(texts)
  const asyncBatch = await minilm.embedBatch(texts)
  assert.equal(asyncBatch.rows, syncBatch.rows)
  assert.equal(asyncBatch.dimensions, syncBatch.dimensions)
  assert.deepEqual(Array.from(asyncBatch.data), Array.from(syncBatch.data))
})

// MUTATION-SENSITIVE: batch.dimensions is derived in src/lib.rs's
// `encode_many` from the same `dimension` value used to size the flat
// `data` buffer. If that field construction is corrupted (e.g.
// `dimensions: dimension + 1`), `batch.data.length` (still the real
// flattened row count * true per-row length) stops equaling
// `batch.rows * batch.dimensions`, and this assertion fails. See the
// delivery report for the actual revert/rebuild/fail, restore/rebuild/pass
// cycle.
test('embedBatchSync batch metadata matches real buffer length (mutation-sensitive)', () => {
  const texts = ['a dog runs in the park', 'a puppy runs in the park', 'quarterly financial report']
  const batch = minilm.embedBatchSync(texts)
  assert.equal(batch.dimensions, minilm.dimension)
  assert.equal(batch.data.length, batch.rows * batch.dimensions)
})

test('embedSync("") rejects with FL_EMBED_EMPTY_INPUT', () => {
  assert.throws(
    () => minilm.embedSync(''),
    (err) => err.code === 'FL_EMBED_EMPTY_INPUT',
  )
})

test('embed("") (async) rejects with FL_EMBED_EMPTY_INPUT', async () => {
  await assert.rejects(
    () => minilm.embed(''),
    (err) => err.code === 'FL_EMBED_EMPTY_INPUT',
  )
})

test('embedBatchSync([]) rejects with FL_EMBED_BAD_BATCH', () => {
  assert.throws(
    () => minilm.embedBatchSync([]),
    (err) => err.code === 'FL_EMBED_BAD_BATCH',
  )
})

// A non-empty array containing an empty item is rejected per-item, with the
// same FL_EMBED_EMPTY_INPUT code embedSync(text) uses -- FL_EMBED_BAD_BATCH
// is reserved for batch-shape problems (not an array, or zero items).
test('embedBatchSync(["ok", ""]) rejects with FL_EMBED_EMPTY_INPUT', () => {
  assert.throws(
    () => minilm.embedBatchSync(['ok', '']),
    (err) => err.code === 'FL_EMBED_EMPTY_INPUT',
  )
})

test('loadModelSync rejects a nonexistent modelPath with FL_EMBED_BAD_MODEL', () => {
  assert.throws(
    () => loadModelSync({ modelPath: join(MODELS_DIR, 'does-not-exist') }),
    (err) => err.code === 'FL_EMBED_BAD_MODEL',
  )
})

test('loadModelSync rejects normalize:false with FL_EMBED_BAD_OPTIONS', () => {
  assert.throws(
    () => loadModelSync({ modelPath: MINILM_DIR, normalize: false }),
    (err) => err.code === 'FL_EMBED_BAD_OPTIONS',
  )
})

test('loadModelSync rejects missing modelPath with FL_EMBED_BAD_OPTIONS', () => {
  assert.throws(
    () => loadModelSync({}),
    (err) => err.code === 'FL_EMBED_BAD_OPTIONS',
  )
})

// Malformed optional fields must fail with our own stable FL_EMBED_BAD_OPTIONS
// code, not a napi-native argument-conversion code (e.g. "StringExpected",
// "BooleanExpected"). This is only true because index.js's normalizeOptions()
// validates modelId/normalize BEFORE calling native code -- napi-rs converts
// the options object into Rust's typed LoadOptions struct as part of the
// native call itself, which would otherwise throw first with its own code.
test('loadModelSync rejects a non-string modelId with FL_EMBED_BAD_OPTIONS', () => {
  assert.throws(
    () => loadModelSync({ modelPath: MINILM_DIR, modelId: 123 }),
    (err) => err.code === 'FL_EMBED_BAD_OPTIONS',
  )
})

test('loadModel (async) rejects a non-string modelId with FL_EMBED_BAD_OPTIONS', async () => {
  await assert.rejects(
    () => loadModel({ modelPath: MINILM_DIR, modelId: 123 }),
    (err) => err.code === 'FL_EMBED_BAD_OPTIONS',
  )
})

test('loadModelSync rejects a non-boolean normalize with FL_EMBED_BAD_OPTIONS', () => {
  assert.throws(
    () => loadModelSync({ modelPath: MINILM_DIR, normalize: 'false' }),
    (err) => err.code === 'FL_EMBED_BAD_OPTIONS',
  )
})

test('loadModel (async) rejects a non-boolean normalize with FL_EMBED_BAD_OPTIONS', async () => {
  await assert.rejects(
    () => loadModel({ modelPath: MINILM_DIR, normalize: 'false' }),
    (err) => err.code === 'FL_EMBED_BAD_OPTIONS',
  )
})

test('embedSync rejects a text longer than MAX_TEXT_CHARS with FL_EMBED_INPUT_TOO_LARGE', () => {
  const oversized = 'a'.repeat(MAX_TEXT_CHARS + 1)
  assert.throws(
    () => minilm.embedSync(oversized),
    (err) => err.code === 'FL_EMBED_INPUT_TOO_LARGE',
  )
})

test('embed (async) rejects a text longer than MAX_TEXT_CHARS with FL_EMBED_INPUT_TOO_LARGE', async () => {
  const oversized = 'a'.repeat(MAX_TEXT_CHARS + 1)
  await assert.rejects(
    () => minilm.embed(oversized),
    (err) => err.code === 'FL_EMBED_INPUT_TOO_LARGE',
  )
})

test('embedBatchSync rejects a batch item longer than MAX_TEXT_CHARS with FL_EMBED_INPUT_TOO_LARGE', () => {
  const oversized = 'a'.repeat(MAX_TEXT_CHARS + 1)
  assert.throws(
    () => minilm.embedBatchSync(['ok', oversized]),
    (err) => err.code === 'FL_EMBED_INPUT_TOO_LARGE',
  )
})

test('embedBatch (async) rejects a batch item longer than MAX_TEXT_CHARS with FL_EMBED_INPUT_TOO_LARGE', async () => {
  const oversized = 'a'.repeat(MAX_TEXT_CHARS + 1)
  await assert.rejects(
    () => minilm.embedBatch(['ok', oversized]),
    (err) => err.code === 'FL_EMBED_INPUT_TOO_LARGE',
  )
})

test('embedBatchSync rejects a batch larger than DEFAULT_MAX_BATCH_SIZE with FL_EMBED_BAD_BATCH', () => {
  const tooManyTexts = Array.from({ length: DEFAULT_MAX_BATCH_SIZE + 1 }, () => 'x')
  assert.throws(
    () => minilm.embedBatchSync(tooManyTexts),
    (err) => err.code === 'FL_EMBED_BAD_BATCH',
  )
})

test('embedBatch (async) rejects a batch larger than DEFAULT_MAX_BATCH_SIZE with FL_EMBED_BAD_BATCH', async () => {
  const tooManyTexts = Array.from({ length: DEFAULT_MAX_BATCH_SIZE + 1 }, () => 'x')
  await assert.rejects(
    () => minilm.embedBatch(tooManyTexts),
    (err) => err.code === 'FL_EMBED_BAD_BATCH',
  )
})

test('unicode input is deterministic', () => {
  const text = '日本語のテキスト、emoji: 🚀, and mixed ASCII.'
  const first = minilm.embedSync(text)
  const second = minilm.embedSync(text)
  assert.deepEqual(Array.from(first), Array.from(second))
})

test('32-way concurrent async embed calls are deterministic (no data race)', async () => {
  const text = 'concurrency probe: a dog runs in the park'
  const expected = minilm.embedSync(text)
  const results = await Promise.all(Array.from({ length: 32 }, () => minilm.embed(text)))
  for (const [i, vec] of results.entries()) {
    assert.equal(vec.length, expected.length, `result ${i} length mismatch`)
    for (let d = 0; d < expected.length; d++) {
      assert.ok(Math.abs(vec[d] - expected[d]) < 1e-6, `result ${i} dim ${d} mismatch`)
    }
  }
})

test('semantic sanity: cosine(dog, puppy) > cosine(dog, report) (minilm)', () => {
  const dog = minilm.embedSync('a dog runs in the park')
  const puppy = minilm.embedSync('a puppy runs in the park')
  const report = minilm.embedSync('quarterly financial report')
  const cosine = (a, b) => {
    let dot = 0, na = 0, nb = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      na += a[i] * a[i]
      nb += b[i] * b[i]
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb))
  }
  assert.ok(cosine(dog, puppy) > cosine(dog, report))
})

test('semantic sanity: cosine(dog, puppy) > cosine(dog, report) (bge, CLS pooling)', () => {
  const dog = bge.embedSync('a dog runs in the park')
  const puppy = bge.embedSync('a puppy runs in the park')
  const report = bge.embedSync('quarterly financial report')
  const cosine = (a, b) => {
    let dot = 0, na = 0, nb = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      na += a[i] * a[i]
      nb += b[i] * b[i]
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb))
  }
  assert.ok(cosine(dog, puppy) > cosine(dog, report))
})

test('loadModel (async) agrees with loadModelSync', async () => {
  const asyncModel = await loadModel({ modelPath: MINILM_DIR })
  assert.equal(asyncModel.dimension, minilm.dimension)
  const vec = await asyncModel.embed('the cat sat on the mat')
  assert.equal(vec.length, 384)
})
