# lattice-embed-wasm

Text embeddings from a pure-Rust BERT-family encoder, compiled to
WebAssembly, wrapped in a small Node adapter that resolves a model name to
weight bytes, caches one loaded model instance per name, and applies the
right pooling strategy per model family.

The wasm core does one thing: given raw model bytes, a config, and a
tokenizer, it embeds a string. It has no filesystem access and no network
access; those are handled by this adapter, not the core. That split (bytes-in
compute core, thin resolving adapter on top) is why the package works the
same whether the weights are already on disk or need to be fetched first.

## Install

```
npm install lattice-embed-wasm
```

## Usage

```js
import { embed, embedText, Embedder } from 'lattice-embed-wasm';

// Primary form: pick a model explicitly.
const vec = await embed('a sentence to embed', 'minilm');

// Default model, no name needed.
const vec2 = await embedText('another sentence');

// Pin a model once, reuse it.
const e = new Embedder('bge');
const vec3 = await e.embed('a third sentence');
```

`embed` and `embedText` are async because the first call for a given model
may need to read or fetch its weights; the loaded model itself is cached in
memory after that, so later calls for the same model name are synchronous
compute with no extra I/O.

The wasm core self-initializes on first use. `initSync` and `init` are
re-exported for callers that want to control wasm instantiation timing
directly instead of relying on the adapter's automatic first-call init.

## Return value and failure modes

`embed()` and `embedText()` return a `Float32Array` (L2-normalized) on
success. They return `null`, not an error, when the requested model name is
unknown, is not supported in this wasm build, or its weights could not be
resolved and verified. This is a deliberate degrade signal for callers that
try several embedding options in order and want to fall through to the next
one rather than fail hard.

An error raised while actually running an already-loaded model is thrown,
not swallowed, since it means something is broken rather than "this
particular option is unavailable."

## Supported models

| name | pooling | dimensions |
| --- | --- | --- |
| `minilm` (default) | mean | 384 |
| `bge` | CLS | 384 |
| `paraphrase-minilm` | mean | 384 |

`qwen3-0.6b` is a known model name that is **not** supported over this wasm
channel: it is a decoder-style embedding model, and the wasm core here only
wraps a BERT encoder. `embed(text, 'qwen3-0.6b')` resolves to `null` rather
than throwing, so a caller can treat it as "use a different tier" instead of
handling a special case.

## Weight resolution

Model weights are not bundled in the package (they are much larger than
typical npm package sizes and are shared across every consumer, not just
this one). Each model's `model.safetensors` / `config.json` / `tokenizer.json`
are resolved in this order:

1. A local directory override: `LATTICE_EMBED_MODELS_DIR` (default
   `~/.lattice/models`), one subdirectory per model, e.g.
   `~/.lattice/models/all-minilm-l6-v2/`.
2. This package's own on-disk cache: `LATTICE_EMBED_WASM_CACHE_DIR` (default
   `~/.cache/lattice-embed-wasm/models`).
3. A pinned download: each model's three files are fetched from a specific
   GitHub release of this repository and checked against a pinned sha256
   hash before use. A verified download is written into the cache directory
   from step 2 so later calls are cache hits.

Every candidate, from any of the three sources, is checked against its
pinned hash before its bytes are used. A candidate that fails verification
is discarded, and resolution falls through to the next source; if none of
the three sources produce a verified result, the model is treated as
unavailable (`embed()` returns `null`) rather than risking a wrong or
tampered vector.

## Integration note

A downstream tool that wants to treat this package as one tier in a larger
embedding fallback chain typically points at it with two environment
variables it defines on its own side:

- one that names which npm package to load as the wasm-embedding provider
  (so it can be swapped without a code change), pointing at
  `lattice-embed-wasm`;
- one that selects which model name to pass into `embed(text, model)`.

Neither variable is read by this package itself; the model name is always
passed as an explicit argument to `embed()`.

## Development

```
npm run build   # compiles the wasm core and generates the JS bindings in wasm/
npm run smoke   # runs a local end-to-end check against cached model weights
```

Building requires `cargo`, the `wasm32-unknown-unknown` Rust target, and the
`wasm-bindgen` CLI at the version pinned in the workspace's `Cargo.toml`
(`cargo install wasm-bindgen-cli --version 0.2.105`). See
`scripts/build-wasm.sh`.

## License

Apache-2.0
