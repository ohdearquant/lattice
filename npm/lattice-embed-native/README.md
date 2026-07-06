# @lattice-embed/native

Native (napi-rs) Node.js bindings for [`lattice-embed`](../../crates/embed):
local BERT-family text embeddings (MiniLM, BGE, E5) loaded directly from a
model directory on disk, with no WebAssembly overhead.

This is the native counterpart to [`lattice-embed-wasm`](../lattice-embed-wasm),
which loads model weights as in-memory byte buffers instead (no local
filesystem access, portable to any JS runtime including browsers). Use this
package when your Node process can read model files from disk directly and
you want the native (non-WASM) call path.

`v0` scope: local-directory model loading only. There is no remote model-id
resolution or download tier in this package (that is a wasm-package-style
concern); point `modelPath` at a directory you already have on disk.

## Status

This is a `v0` cut. Built and tested on darwin-arm64 only. The other six
napi-rs target triples (darwin-x64, linux-x64-gnu, linux-x64-musl,
linux-arm64-gnu, linux-arm64-musl, win32-x64-msvc) are scaffolded in
`package.json`'s `napi.targets` and `optionalDependencies` but not yet built
or tested. Package manager support tested: npm and pnpm.

## Install

```sh
npm install @lattice-embed/native
```

## Usage

```js
const { loadModelSync } = require('@lattice-embed/native')

const model = loadModelSync({ modelPath: '/path/to/all-minilm-l6-v2' })
console.log(model.dimension) // 384

const vec = model.embedSync('a dog runs in the park')
console.log(vec instanceof Float32Array, vec.length) // true 384

const batch = model.embedBatchSync([
  'a dog runs in the park',
  'a puppy runs in the park',
])
console.log(batch.rows, batch.dimensions) // 2 384
console.log(batch.vector(0)) // Float32Array view into batch.data

// Async variants (napi AsyncTask, run off the JS event loop thread):
const { loadModel } = require('@lattice-embed/native')
const asyncModel = await loadModel({ modelPath: '/path/to/bge-small-en-v1.5' })
const asyncVec = await asyncModel.embed('quarterly financial report')
```

`modelPath`'s directory name is used to infer the model family (pooling
strategy + expected dimension) unless an explicit `modelId` is given. This
works automatically for a directory named after its canonical slug, e.g.
`all-minilm-l6-v2` or `bge-small-en-v1.5`.

## Error codes

Errors thrown by this package carry a stable `.code`:

| Code | Meaning |
|---|---|
| `FL_EMBED_BAD_OPTIONS` | Malformed `loadModel`/`loadModelSync` options (missing `modelPath`, or an unsupported `normalize: false`). |
| `FL_EMBED_BAD_MODEL` | `modelPath` does not exist, the model family could not be determined, the family is not BERT-family (e.g. Qwen), or the engine failed to load/encode. |
| `FL_EMBED_EMPTY_INPUT` | `embed`/`embedSync` called with an empty string. |
| `FL_EMBED_BAD_BATCH` | `embedBatch`/`embedBatchSync` called with an empty array or an array containing an empty string. |
| `FL_EMBED_UNSUPPORTED_PLATFORM` | No native binary exists for the current `process.platform`/`process.arch`. |
| `FL_EMBED_NATIVE_LOAD_FAILED` | A native binary should exist for this platform but failed to load (missing optional dependency, ABI mismatch, etc). |

## Known v0 limitation: `normalize: false`

The underlying engine (`BertModel::encode`/`encode_batch` in
`crates/inference`) always L2-normalizes its output; there is no public
non-normalizing path. Rather than silently return a normalized vector when a
caller explicitly asks for `normalize: false`, this package rejects that
option with `FL_EMBED_BAD_OPTIONS`. `model.normalized` is always `true`.

## Development

```sh
npm run build        # napi build --release --platform --no-js
npm run build:debug   # faster iteration build
npm test              # node --test __test__/*.mjs
npm run smoke         # node smoke.mjs (requires local model directories)
npm run packlist      # pack-list guard: main package must not ship a .node binary
```

## License

Apache-2.0
