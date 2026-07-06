#!/usr/bin/env node
// Smoke test: loads MiniLM and BGE from local model directories under
// ~/.lattice/models (the lattice-inference default cache directory, see
// crates/inference/src/lib.rs's `default_cache_dir`) through the real
// native binding (no mock, no in-memory bytes) and asserts:
//   - the result is an instanceof Float32Array with the right dimension
//   - the L2 norm is within [0.999, 1.001] of 1.0
//   - every component is finite
//   - a semantic sanity check: cosine(dog, puppy) > cosine(dog, report)
// Exercises both the sync and async call paths. Run with:
//   npm run build && npm run smoke

import assert from 'node:assert/strict'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { loadModel, loadModelSync } from './index.js'

const MODELS_DIR = process.env.LATTICE_MODEL_CACHE || join(homedir(), '.lattice', 'models')

const MODELS = [
  { name: 'minilm', dir: join(MODELS_DIR, 'all-minilm-l6-v2') },
  { name: 'bge', dir: join(MODELS_DIR, 'bge-small-en-v1.5') },
]

function l2Norm(vec) {
  let sum = 0
  for (const x of vec) sum += x * x
  return Math.sqrt(sum)
}

function cosine(a, b) {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

function assertVector(vec, label) {
  assert.ok(vec instanceof Float32Array, `${label}: result is not a Float32Array`)
  assert.equal(vec.length, 384, `${label}: expected 384 dims, got ${vec.length}`)
  for (let i = 0; i < vec.length; i++) {
    assert.ok(Number.isFinite(vec[i]), `${label}: component ${i} is not finite (${vec[i]})`)
  }
  const norm = l2Norm(vec)
  assert.ok(
    norm >= 0.999 && norm <= 1.001,
    `${label}: L2 norm not in [0.999, 1.001] (got ${norm})`,
  )
  return norm
}

async function checkModel({ name, dir }) {
  console.log(`\n=== ${name} (${dir}) ===`)

  // Sync load + sync embed.
  const modelSync = loadModelSync({ modelPath: dir })
  console.log(`  dimension=${modelSync.dimension} normalized=${modelSync.normalized}`)
  assert.equal(modelSync.dimension, 384, `${name}: expected dimension 384`)
  assert.equal(modelSync.normalized, true, `${name}: expected normalized=true`)

  const vecSync = modelSync.embedSync('the cat sat on the mat')
  const normSync = assertVector(vecSync, `${name} (sync)`)
  console.log(`  [sync]  dims=${vecSync.length} norm=${normSync.toFixed(6)}`)

  const dogSync = modelSync.embedSync('a dog runs in the park')
  const puppySync = modelSync.embedSync('a puppy runs in the park')
  const reportSync = modelSync.embedSync('quarterly financial report')
  const cosNearSync = cosine(dogSync, puppySync)
  const cosFarSync = cosine(dogSync, reportSync)
  console.log(
    `  [sync]  cosine(dog, puppy)=${cosNearSync.toFixed(6)}  cosine(dog, report)=${cosFarSync.toFixed(6)}`,
  )
  assert.ok(
    cosNearSync > cosFarSync,
    `${name} (sync): semantic sanity check failed (dog~puppy=${cosNearSync} <= dog~report=${cosFarSync})`,
  )

  // Batch sync.
  const batchSync = modelSync.embedBatchSync([
    'a dog runs in the park',
    'a puppy runs in the park',
    'quarterly financial report',
  ])
  assert.equal(batchSync.rows, 3, `${name}: expected 3 batch rows`)
  assert.equal(batchSync.dimensions, 384, `${name}: expected batch dimensions 384`)
  assert.equal(
    batchSync.data.length,
    batchSync.rows * batchSync.dimensions,
    `${name}: flat batch buffer length mismatch`,
  )
  assertVector(batchSync.vector(0), `${name} (batch sync row 0)`)

  // Async load + async embed.
  const modelAsync = await loadModel({ modelPath: dir })
  const vecAsync = await modelAsync.embed('the cat sat on the mat')
  const normAsync = assertVector(vecAsync, `${name} (async)`)
  console.log(`  [async] dims=${vecAsync.length} norm=${normAsync.toFixed(6)}`)

  const batchAsync = await modelAsync.embedBatch([
    'a dog runs in the park',
    'a puppy runs in the park',
  ])
  assert.equal(batchAsync.rows, 2, `${name}: expected 2 async batch rows`)

  console.log(`  PASS`)
}

let allPass = true
for (const model of MODELS) {
  try {
    // eslint-disable-next-line no-await-in-loop
    await checkModel(model)
  } catch (err) {
    allPass = false
    console.error(`FAIL [${model.name}]: ${err.stack || err.message}`)
  }
}

console.log(`\nsmoke test: ${allPass ? 'PASS' : 'FAIL'}`)
process.exit(allPass ? 0 : 1)
