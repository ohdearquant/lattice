'use strict'

const native = require('./binding.js')

function attachErrorCode(error) {
  // napi-rs bridges a Rust `Error::new(Status::InvalidArg, "CODE: message")`
  // into a JS Error with `.code` ALREADY set to the raw napi Status name
  // (e.g. "InvalidArg"), not our own FL_EMBED_* code -- verified empirically
  // (a `!error.code` guard here silently keeps "InvalidArg" instead of the
  // real code, since a truthy string is already present). Always prefer our
  // own `CODE: message` prefix when the message has one, overwriting
  // whatever napi-rs set.
  if (error && typeof error === 'object' && typeof error.message === 'string') {
    const match = /^([A-Z0-9_]+):/.exec(error.message)
    if (match) error.code = match[1]
  }
  return error
}

function throwWithCode(error) {
  throw attachErrorCode(error)
}

function normalizeOptions(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    const error = new TypeError('FL_EMBED_BAD_OPTIONS: loadModel options must be an object')
    error.code = 'FL_EMBED_BAD_OPTIONS'
    throw error
  }

  if (typeof options.modelPath !== 'string' || options.modelPath.trim().length === 0) {
    const error = new TypeError('FL_EMBED_BAD_OPTIONS: options.modelPath must be a non-empty string')
    error.code = 'FL_EMBED_BAD_OPTIONS'
    throw error
  }

  return { ...options }
}

function assertText(text, index) {
  if (typeof text !== 'string') {
    const error = new TypeError(`FL_EMBED_EMPTY_INPUT: text at index ${index} must be a string`)
    error.code = 'FL_EMBED_EMPTY_INPUT'
    throw error
  }

  if (text.length === 0) {
    const error = new TypeError(`FL_EMBED_EMPTY_INPUT: text at index ${index} must not be empty`)
    error.code = 'FL_EMBED_EMPTY_INPUT'
    throw error
  }
}

function normalizeTexts(texts) {
  if (!Array.isArray(texts)) {
    const error = new TypeError('FL_EMBED_BAD_BATCH: texts must be an array')
    error.code = 'FL_EMBED_BAD_BATCH'
    throw error
  }

  if (texts.length === 0) {
    const error = new RangeError('FL_EMBED_BAD_BATCH: texts must contain at least one item')
    error.code = 'FL_EMBED_BAD_BATCH'
    throw error
  }

  texts.forEach(assertText)
  return texts.slice()
}

function attachBatchHelpers(batch) {
  if (!batch || !(batch.data instanceof Float32Array)) {
    const error = new TypeError('FL_EMBED_BAD_BATCH: native batch result is malformed')
    error.code = 'FL_EMBED_BAD_BATCH'
    throw error
  }

  Object.defineProperty(batch, 'vector', {
    enumerable: false,
    configurable: false,
    value(index) {
      if (!Number.isInteger(index) || index < 0 || index >= batch.rows) {
        throw new RangeError(`batch vector index out of range: ${index}`)
      }
      const start = index * batch.dimensions
      return batch.data.subarray(start, start + batch.dimensions)
    }
  })

  return batch
}

function splitBatch(batch) {
  return Array.from({ length: batch.rows }, (_, index) => batch.vector(index))
}

class EmbeddingModel {
  #nativeModel

  constructor(nativeModel) {
    this.#nativeModel = nativeModel
  }

  get dimension() {
    return this.#nativeModel.dimension
  }

  get normalized() {
    return this.#nativeModel.normalized
  }

  embedSync(text) {
    assertText(text, 0)
    try {
      return this.#nativeModel.embedSync(text)
    } catch (error) {
      throwWithCode(error)
    }
  }

  async embed(text) {
    assertText(text, 0)
    try {
      return await this.#nativeModel.embed(text)
    } catch (error) {
      throwWithCode(error)
    }
  }

  embedBatchSync(texts) {
    const normalized = normalizeTexts(texts)
    try {
      return attachBatchHelpers(this.#nativeModel.embedBatchSync(normalized))
    } catch (error) {
      throwWithCode(error)
    }
  }

  async embedBatch(texts) {
    const normalized = normalizeTexts(texts)
    try {
      return attachBatchHelpers(await this.#nativeModel.embedBatch(normalized))
    } catch (error) {
      throwWithCode(error)
    }
  }

  embedBatchListSync(texts) {
    return splitBatch(this.embedBatchSync(texts))
  }

  async embedBatchList(texts) {
    return splitBatch(await this.embedBatch(texts))
  }
}

function loadModelSync(options) {
  try {
    return new EmbeddingModel(native.loadModelSync(normalizeOptions(options)))
  } catch (error) {
    throwWithCode(error)
  }
}

async function loadModel(options) {
  try {
    return new EmbeddingModel(await native.loadModel(normalizeOptions(options)))
  } catch (error) {
    throwWithCode(error)
  }
}

module.exports = {
  EmbeddingModel,
  loadModel,
  loadModelSync,
  splitBatch
}
