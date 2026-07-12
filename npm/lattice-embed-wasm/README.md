# @khive-ai/lattice-embed-wasm

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
npm install @khive-ai/lattice-embed-wasm
```

## Usage

```js
import { embed, embedText, Embedder } from '@khive-ai/lattice-embed-wasm';

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

## Performance (WebAssembly SIMD128)

The wasm core is built with `-C target-feature=+simd128`, which activates
SIMD128 kernels for the four vector ops on the crate's stable public
contract (`dot_product`, `squared_euclidean_distance`/L2, `cosine_similarity`,
`normalize`). wasm32 has no runtime CPU-feature detection, so this is a
compile-time choice: this package ships one artifact, built with the flag
on, and there is no separate non-SIMD128 fallback build published from this
flow.

Median per-call latency, baseline (scalar, no SIMD128) vs. SIMD128, measured
in Node v25.6.0 with `process.hrtime.bigint()`, 5000 timed reps + 500 warmup
reps per cell, both variants marshaling identical `Float32Array` inputs
through wasm-bindgen so the delta isolates the kernel, not call overhead:

| op | dim | baseline median (ns) | baseline p95 (ns) | simd128 median (ns) | simd128 p95 (ns) | median speedup |
|---|---|---|---|---|---|---|
| dot_product | 384 | 583.0 | 958.0 | 333.0 | 417.0 | 1.75x |
| squared_l2 | 384 | 584.0 | 1250.0 | 333.0 | 417.0 | 1.75x |
| cosine | 384 | 1208.0 | 1292.0 | 292.0 | 500.0 | 4.14x |
| normalize | 384 | 708.0 | 1375.0 | 334.0 | 916.0 | 2.12x |
| dot_product | 768 | 1042.0 | 1125.0 | 334.0 | 375.0 | 3.12x |
| squared_l2 | 768 | 1000.0 | 1084.0 | 333.0 | 375.0 | 3.00x |
| cosine | 768 | 2459.0 | 2584.0 | 375.0 | 416.0 | 6.56x |
| normalize | 768 | 1208.0 | 1292.0 | 375.0 | 459.0 | 3.22x |
| dot_product | 1024 | 1375.0 | 1458.0 | 375.0 | 417.0 | 3.67x |
| squared_l2 | 1024 | 1292.0 | 1375.0 | 375.0 | 417.0 | 3.45x |
| cosine | 1024 | 3333.0 | 3417.0 | 458.0 | 542.0 | 7.28x |
| normalize | 1024 | 1584.0 | 1708.0 | 500.0 | 542.0 | 3.17x |
| dot_product | 4096 | 4959.0 | 6875.0 | 959.0 | 1000.0 | 5.17x |
| squared_l2 | 4096 | 4916.0 | 4959.0 | 1000.0 | 1000.0 | 4.92x |
| cosine | 4096 | 13375.0 | 26708.0 | 1125.0 | 1250.0 | 11.89x |
| normalize | 4096 | 5541.0 | 5791.0 | 1250.0 | 2583.0 | 4.43x |

Speedup is consistent (1.75x-11.9x median) across all four kernels and
grows with dimension; `cosine` shows the largest win because it fuses three
reductions (dot, norm_a, norm_b) into one SIMD128 pass. These numbers cover
the four vector-op kernels directly; end-to-end `embed()` latency also
includes tokenization and the BERT forward pass, which are unaffected by
this flag.

**Runtime floor**: WebAssembly SIMD128 is a baseline feature in every
evergreen browser (shipped across Chrome, Firefox, Edge, and Safari since
2021-2023) and in Node.js since 16.4. This package's `engines.node` field
already requires `>=18`, comfortably above that floor, so no environment
that can load this package at all falls short of the SIMD128 requirement.

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

1. A local directory override: `LATTICE_EMBED_MODELS_DIR`, or
   `LATTICE_MODEL_CACHE` (the same cache-directory environment variable the
   native `lattice-embed` CLI reads), or `~/.lattice/models` by default; one
   subdirectory per model, e.g. `~/.lattice/models/all-minilm-l6-v2/`. This
   tier is **wasm-specific**: it requires `model.safetensors`, `config.json`,
   and `tokenizer.json` in that subdirectory. It is not automatic reuse of
   whatever the native CLI happens to have cached: a directory populated by
   a native download of a WordPiece model (BGE or MiniLM) contains
   `vocab.txt`, not `tokenizer.json`, and is not directly usable here until
   a `tokenizer.json` is also present. This package does not synthesize a
   `tokenizer.json` from `vocab.txt`.
2. This package's own on-disk cache: `LATTICE_EMBED_WASM_CACHE_DIR` (default
   `~/.cache/lattice-embed-wasm/models`).
3. A pinned download, once enabled: each model's three files are fetched
   from a specific GitHub release of this repository and checked against a
   pinned sha256 hash before use. This tier is a **publish step**, gated
   separately from this package's code: it activates only once the model
   weight assets have been uploaded as release assets and
   `WEIGHTS_RELEASE_TAG` (in `registry.mjs`) is set to that release's tag.
   Until then this tier is skipped (with a one-time warning explaining why,
   not spammed per call), and a model not found via steps 1-2 resolves to
   `null`. A verified download is written into the cache directory from
   step 2 so later calls are cache hits.

Every candidate, from any of the three tiers, is checked against its pinned
hash before its bytes are used. A **missing** pinned hash for any file a
model declares counts as a verification failure too, never a skip: the
registry itself enforces this at module load, refusing to register a model
entry that lacks a pinned hash for one of its declared files. A candidate
that fails verification is discarded, and resolution falls through to the
next source; if none of the tiers produce a verified result, the model is
treated as unavailable (`embed()` returns `null`) rather than risking a
wrong or tampered vector.

## Integration note

A downstream tool that wants to treat this package as one tier in a larger
embedding fallback chain typically points at it with two environment
variables it defines on its own side:

- one that names which npm package to load as the wasm-embedding provider
  (so it can be swapped without a code change), pointing at
  `@khive-ai/lattice-embed-wasm`;
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
(`cargo install wasm-bindgen-cli --version 0.2.105`). The build compiles
with `-C target-feature=+simd128` (see "Performance" above); see
`scripts/build-wasm.sh`.

## License

Apache-2.0
