# LoRA Safetensors and Governed Loading

The LoRA I/O path has two layers. `lora::safetensors` translates PEFT and MLX
weight files into `LoraAdapter` values and writes PEFT-compatible files.
`lora::loader` adds a manifest-controlled admission gate for serving. The
format parser establishes that the tensors form an adapter; the manifest loader
establishes that this particular file is approved, confined, intact, and
compatible with the requested serving context.

## LoRA factor layout

A LoRA update for one linear projection is:

```text
dW = B @ A
output += (alpha / rank) * B @ (A @ input)
```

The in-memory `LoraLayer` stores both matrices as row-major `f32` values:

| Factor | Shape | Purpose |
| --- | --- | --- |
| A | `(rank, d_in)` | Projects the base input down to the low-rank space |
| B | `(d_out, rank)` | Projects low-rank values up to the base output dimension |

Each pair is identified by `(layer_idx, module)`, where `layer_idx` is
zero-based and `module` is the final key segment such as `q_proj`, `v_proj`,
or `gate_proj`. An adapter's `LoraConfig` has one common rank, an `alpha`, and
the sorted set of target module names observed in its tensor keys. The runtime
scale is `alpha / rank`; invalid or non-finite scale data is rejected when a
`LoraAdapter` is constructed or saved.

## Accepted tensor keys

The parser recognizes PEFT uppercase suffixes and MLX lowercase suffixes. It
does not require one fixed leading prefix: it finds the `.layers.{index}.`
segment and uses the final preceding key component as the module name.

| Producer | A key | B key | Stored shape before normalization |
| --- | --- | --- | --- |
| PEFT | `...layers.{i}.self_attn.{module}.lora_A.weight` | `...layers.{i}.self_attn.{module}.lora_B.weight` | A `(rank, d_in)`, B `(d_out, rank)` |
| PEFT MLP | `...layers.{i}.mlp.{module}.lora_A.weight` | `...layers.{i}.mlp.{module}.lora_B.weight` | A `(rank, d_in)`, B `(d_out, rank)` |
| MLX | `...layers.{i}.{block}.{module}.lora_a` | `...layers.{i}.{block}.{module}.lora_b` | A `(d_in, rank)`, B `(rank, d_out)` |

The trailing `.weight` is optional. Non-LoRA tensors and keys that do not fit
this grammar are ignored; a file with no recognized `lora_A` or `lora_B`
tensors is rejected. MLX factors are transposed into the PEFT/in-memory layout
before pairing, but only after verifying that the source is two-dimensional
and that its stated element count does not overflow or disagree with its byte
data.

The writer emits the PEFT spelling with the full
`base_model.model.model.layers.{i}` prefix. It uses `self_attn` for
`q_proj`, `k_proj`, `v_proj`, and `o_proj`, and `mlp` for `gate_proj`,
`up_proj`, and `down_proj`. Unknown module names are written under
`self_attn`; the module name itself remains part of the key.

## Numeric representation and parser checks

Input tensors may be `F32`, `F16`, or `BF16`. Their data is decoded to
in-memory `f32` using little-endian elements; F16 and BF16 are converted
before any factor validation. Other safetensors dtypes are rejected. The
parser also rejects a byte length not divisible by the dtype width and rejects
any decoded NaN or infinity.

For every `(layer_idx, module)` key, loading is fail-closed:

1. Both an A and a B tensor must exist; an orphan of either kind is an error.
2. Both tensors must be two-dimensional.
3. Their rank axes must agree: `A.shape[0] == B.shape[1]` after any MLX
   transpose.
4. That rank must match the first factor pair's rank. Mixed-rank adapters are
   rejected.
5. `rank * d_in` and `d_out * rank` must not overflow `usize`, and each
   product must equal the decoded element count.
6. The completed configuration must pass `LoraAdapter::new`, including its
   finite-alpha and finite-effective-scale checks.

This means a successful parse represents complete, consistently shaped,
finite factor pairs. It does not by itself prove that the factors belong to a
particular base model; when the `inference-hook` feature is available, the
governed loader can additionally validate every layer index and projection
dimension against the active model configuration.

## Header metadata and export

The writer serializes F32 factor tensors and includes these normal adapter
metadata entries in the safetensors header:

| Key | Value |
| --- | --- |
| `rank` | `LoraConfig::rank` as text |
| `alpha` | `LoraConfig::alpha` as text |
| `target_modules` | Comma-joined target-module list |

The loader uses `alpha` if it is present. It must parse as a finite `f32`.
When absent, it uses `rank` as alpha, giving the neutral scale `1.0`. The
header's `rank` and `target_modules` are informational to this parser: the
authoritative rank and module set are derived from validated factors.

`save_peft_safetensors` may also receive `AdapterGovernance`, which writes all
six provenance fields together:

```text
gov_name
gov_owner
gov_base_model_rev
gov_tokenizer_rev
gov_dtype
gov_status
```

They describe an adapter but are not a substitute for the governed manifest.
In particular, the writer never embeds `integrity_sha256`: an integrity digest
is defined over the complete safetensors file, including the header. Writing
such a digest into that header changes the file and therefore changes the
digest. The manifest is the only authority for whole-file integrity.

An optional header `adapter_id` is read by the manifest loader. If it exists,
it must equal the manifest entry identifier. Its absence is allowed, which
keeps compatibility with PEFT exports that do not include it. Reading optional
governance metadata is advisory; missing or malformed optional header metadata
does not substitute for a manifest decision.

The exporter skips a layer whose A or B buffer is empty. Such a layer denotes a
module that was not trained; emitting a non-zero shaped tensor with no data
would create an invalid safetensors tensor view.

## Bounded file reads

All paths that materialize adapter bytes use a common 10 GiB maximum. The
limit protects the full-buffer parser from an oversized-file allocation attack
and has two checks:

1. Read file metadata first and reject a known-oversized file before opening
   it for content.
2. Open the file and read through `take(max_bytes + 1)`. The one-byte sentinel
   rejects a file that grew after the metadata check while allowing a file
   exactly at the limit.

The second check closes the metadata-to-open size race. Both the plain
path-based loader and the manifest loader use this guarded reader, so the
manifest path cannot bypass the memory bound.

## Manifest admission

`load_adapters_from_manifest` accepts a parsed `LoraManifest`, a base directory
for relative paths, an optional active-model configuration when
`inference-hook` is enabled, and optional `RunningRevisions`. It returns
`Ok(Vec<LoadedAdapter>)` only when every entry succeeds. Any anomaly returns
an error and discards the accumulated output; there is no partial result and
no silent skip.

Before opening any file, the loader scans the whole manifest. Every entry must
be `Approved`; a `Quarantined` or `Revoked` entry rejects the entire call. This
pre-scan prevents a later disallowed entry from allowing earlier files to be
read first. The per-entry status check is repeated as defense in depth.

For each approved entry, these checks run in order:

| Check | Requirement |
| ---: | --- |
| 1 | Status is `Approved`. |
| 2 | `uri` is relative, contains no `..` component, canonicalizes successfully, and its canonical target remains under the canonical base directory. This rejects absolute paths and symlink escapes. |
| 3–4 | The proven in-base file can be read under the 10 GiB cap, and the SHA-256 of those exact bytes equals `integrity_sha256`. Existence is folded into this guarded read to avoid a separate check/open race. |
| 5 | The bytes parse as a complete PEFT or normalized MLX LoRA adapter. |
| 6 | Parsed rank equals manifest `rank`. |
| 7 | Parsed alpha and manifest alpha are finite and differ by at most `1e-4`. If the file omits alpha, its synthesized `rank` value is still compared. |
| 8 | Every parsed target module is listed in manifest `target_modules`; the parser's set must be a subset of the manifest declaration. |
| 9 | When `inference-hook` is active and a model configuration is supplied, the adapter dimensions and layer indices validate against that model. |
| 10 | If a safetensors header has `adapter_id`, it equals the manifest `id`. |
| 11 | When running revisions are supplied, both base-model and tokenizer revisions exactly match unless the caller explicitly allows a mismatch. |

Canonicalization is a proof step, not a convenience. A target that cannot be
canonicalized is refused rather than read lexically, because the loader has not
proved it remains under the base directory. There remains a final-component
replacement window between canonicalization and open; whole-file SHA-256
verification rejects a substituted file whose bytes do not match the manifest.

## Revision enforcement

`RunningRevisions::strict(base_model_rev, tokenizer_rev)` is the normal
serving context. It compares both strings exactly with the manifest fields and
rejects either mismatch. The literal `"none"` is not a wildcard: legacy
entries using that sentinel require the running side to supply `"none"` too.

`RunningRevisions::permissive(...)` is an explicit migration escape hatch.
It permits mismatched revisions but marks the returned
`LoadedAdapter::rev_mismatch_overridden` as `true`. Callers should surface that
condition because the adapter can load and execute while producing degraded
results against a different base-model or tokenizer revision. Passing no
running revisions skips this check, which is appropriate only when there is no
live serving context, such as offline manifest validation.

## Serving pattern

```rust,no_run
use std::path::Path;
use lattice_tune::lora::{LoraManifest, RunningRevisions};
use lattice_tune::lora::loader::load_adapters_from_manifest;

let manifest = LoraManifest::load(Path::new("adapters/manifest.json"))?;
let revisions = RunningRevisions::strict("base-weights-revision", "tokenizer-revision");

#[cfg(not(feature = "inference-hook"))]
let loaded = load_adapters_from_manifest(
    &manifest,
    Path::new("adapters"),
    Some(&revisions),
)?;

#[cfg(feature = "inference-hook")]
let loaded = load_adapters_from_manifest(
    &manifest,
    Path::new("adapters"),
    Some(&model_config),
    Some(&revisions),
)?;

for adapter in &loaded {
    assert!(!adapter.rev_mismatch_overridden);
}
# Ok::<(), lattice_tune::TuneError>(())
```

For a simple local import without manifest governance, use
`load_peft_safetensors(path)` or `LoraAdapter::from_safetensors(path)`. That
path retains the bounded read and tensor-format validation, but it deliberately
does not establish status, approved location, whole-file integrity, manifest
claims, or serving-revision compatibility.
