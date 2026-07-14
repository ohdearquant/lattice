# LoRA Training Core

`lattice-tune` represents a LoRA adapter as a set of small, per-projection
weight updates. The same representation supports three different operations:

- applying an adapter during inference;
- fitting an adapter with the exact CPU backward tape or a one-event SGD refit;
- combining adapters or admitting them through a governed manifest.

This document describes the core behavior implemented by `lora/mod.rs`,
`train_core.rs`, `train.rs`, `optimizer.rs`, `blend.rs`, `online.rs`, and
`manifest.rs`. Safetensors parsing and manifest loading are separate modules,
but their externally visible validation rules are included where they define
the core contract.

## Adapter representation and inference

For a base linear projection with input width `d_in` and output width `d_out`,
a `LoraLayer` stores two row-major `f32` matrices:

| Matrix | Logical shape | Flat storage | Role |
| --- | --- | --- | --- |
| `A` | `(rank, d_in)` | `rank * d_in` values | Projects the activation into the low-rank space. |
| `B` | `(d_out, rank)` | `d_out * rank` values | Projects the low-rank activation back to output width. |

With adapter configuration `(rank, alpha)`, inference adds the following
update to the base projection output:

```text
scale = alpha / rank
h     = A @ x
output += scale * (B @ h)
```

The update is additive: the base projection remains responsible for producing
`output`, and LoRA only adds its correction. `LoraAdapter::apply` identifies a
layer by `(transformer_layer_index, module_name)`. A missing key is a no-op,
which allows a partial adapter to coexist with a model that has many more
projections.

`LoraConfig::scale()` deliberately returns `0.0` when `rank == 0`, and also
returns `0.0` for non-finite alpha or derived scale. Construction through
`LoraAdapter::new` validates that alpha and its effective scale are finite.
The micro trainer is stricter and rejects rank zero before allocating or
running the tape.

The adapter layer itself is intentionally model-neutral. Typical target names
are:

| Model family | Attention targets | Feed-forward targets |
| --- | --- | --- |
| Qwen-style transformer | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `gate_proj`, `up_proj`, `down_proj` |
| BERT or cross-encoder | `query`, `key`, `value`, `attn_output` | `ffn_intermediate`, `ffn_output` |

When the `inference-hook` feature is available, `validate_against` checks a
Qwen3.5 adapter before installation. It verifies every layer index and each
adapted projection's input and output width. This is the point to reject a
model/adapter shape mismatch before generation, rather than discovering it in
a projection call.

## Exact micro-LoRA training

The `train-backward` feature exposes `train_micro_lora`, a compact CPU trainer
for tokenized prompt/completion pairs. It is intended for a bounded suffix of a
Qwen3.5 model, not for the crate's general dataset and training facilities.

### Input contract

A `TrainingPair` contains the complete token sequence and the index where the
completion begins. For a pair of length `L`, supervised next-token positions
are:

```text
completion_start - 1 ..= L - 2
```

At position `t`, the model predicts `tokens[t + 1]`. This makes the prompt's
last token the context for the first completion token.

`MicroLoraConfig` has these defaults:

| Field | Default | Meaning |
| --- | ---: | --- |
| `rank` | `4` | Low-rank dimension for each trained projection. |
| `alpha` | `8.0` | LoRA alpha; execution scale is `alpha / rank`. |
| `first_layer` | `19` | First model layer included in the materialized suffix. |
| `steps` | `25` | Number of Adam updates. |
| `learning_rate` | `1e-3` | Adam learning rate. |
| `max_seq_len` | `64` | Per-pair sequence length ceiling. |

The trainer fails before model access when any caller-controlled condition
would make tape indexing or allocation unsafe:

- there must be at least one pair, and every pair needs at least two tokens;
- every pair must fit `max_seq_len`, and every token ID must be below the
  model vocabulary size;
- `completion_start` must be in `1..tokens.len()`;
- rank must be in `1..=512`;
- steps must be at most `100_000`, and `max_seq_len` at most `8_192`;
- `first_layer` must not exceed `TOP_LAYER` (`23`), and the model must contain
  at least 24 layers because the loop indexes the inclusive range through 23.

The rank, step, and sequence caps are allocation and runtime safety bounds,
not recommendations for model quality. The model's own context window remains
an independent limit.

### Materialized suffix and slot layout

The model runs the prefix before `first_layer` through its normal frozen
forward path. For each pair, the trainer captures the hidden state entering
`first_layer` and obtains the RoPE cosine and sine tables for that sequence.
It then materializes layers `first_layer..=23` as a tape.

Each materialized layer is classified as one of two mixer kinds:

- **GQA** layers receive a LoRA slot in the public micro-training path. The
  resulting adapter contains `q_proj` and `v_proj` layers for every such slot.
- **GatedDeltaNet (GDN)** layers are included in the forward and backward path
  so their derivatives reach earlier GQA layers, but `train_micro_lora` gives
  them no LoRA slots. They are frozen in this public helper.

The tape uses an explicit GQA-slot and GDN-slot layout rather than assuming a
slot number equals a model layer number. `TrainCtx::try_new` rejects duplicate
slot entries, indices outside the model, a GQA slot placed on a GDN layer, a
GDN slot placed on a GQA layer, and a layer claimed by both layouts. This keeps
the weight vectors, cache selection, and optimizer keys aligned.

### Initialization

Every GQA slot contains parameters for both `q_proj` and `v_proj`:

| Projection | A shape | B shape |
| --- | --- | --- |
| `q_proj` | `(rank, hidden)` | `(2 * q_dim, rank)` |
| `v_proj` | `(rank, hidden)` | `(kv_dim, rank)` |

The trainer initializes A with a deterministic uniform sample in
`[-1 / sqrt(hidden), +1 / sqrt(hidden)]` and initializes B to zero. Zero B
makes the initial adapter contribution exactly zero, preserving the frozen
model's forward behavior at the first step. Consequently the first update can
move B while A's gradient is initially zero; later steps update both factors.

All rank-by-dimension products are checked before the parameter vectors are
allocated. The same checked arithmetic is used by the tape's gradient buffers.

### Forward pass

For each materialized layer, the tape records the intermediates required by
reverse mode:

```text
h_layer_in = h
normed_pre, inv_pre = RMSNorm(h_layer_in, pre_shift)
mixer_out, mixer_cache = GQA-or-GDN mixer(normed_pre, optional LoRA)
h_mid = h_layer_in + mixer_out
normed_ffn, inv_ffn = RMSNorm(h_mid, post_shift)
ffn_out, gate_pre, up_pre = SwiGLU(normed_ffn)
h = h_mid + ffn_out
```

For GQA, the mixer receives A/B for Q and V together with the rank and
`alpha / rank` scale. For a GDN layer without a LoRA slot, the same saved
forward path is used with no LoRA matrices and a rank/scale of zero. Thus the
backward tape always has the mixer state it needs, irrespective of whether the
mixer itself is being adapted.

At every supervised position, the tape applies final RMSNorm, computes logits
with the language-model head, and stores the logits, target token, position,
and final normalization inverse. `eval_chain_nll` reports mean NLL across all
supervised positions in a cache set. `nll_and_grads` returns the summed NLL and
the count of supervised positions, while normalizing each logit derivative by
the number of positions in that sequence.

### Reverse pass

For one target token, the loss is negative log likelihood. The logit derivative
is the usual softmax probability minus the one-hot target, divided by the
number of supervised positions. The tape then propagates through the language
model head and final RMSNorm into the final hidden state.

Layers are processed in reverse order. At each layer, the derivative first
passes through the SwiGLU feed-forward branch and post-mixer RMSNorm, then
through the saved GQA or GDN mixer, and finally through pre-mixer RMSNorm. The
residual derivative is added at both residual joins. GQA mixer backpropagation
fills the slot's Q/V LoRA gradients; a GDN slot, when used by a lower-level
caller, yields gradients for all five of its LoRA projections.

The optional GDN parameter container has ten arrays per slot:

| Projection | A shape | B shape |
| --- | --- | --- |
| `in_proj_qkv` | `(rank, hidden)` | `(qkv_dim, rank)` |
| `in_proj_z` | `(rank, hidden)` | `(output_dim, rank)` |
| `in_proj_b` (beta) | `(rank, hidden)` | `(value_heads, rank)` |
| `in_proj_a` (alpha) | `(rank, hidden)` | `(value_heads, rank)` |
| `out_proj` | `(rank, output_dim)` | `(hidden, rank)` |

The beta and alpha B matrices are indexed by **value heads**, not key heads.
That shape must agree with the shipping GDN forward implementation and its
weight loader; using `num_kh` is incorrect for models whose key and value head
counts differ.

### Validated tape context

`TrainCtx` is the immutable agreement shared by forward, evaluation, reverse,
and optimizer entry points. It owns references to geometry and slot layout plus
the derived execution values and Adam policy; it does not own weights, caches,
gradients, LoRA vectors, or optimizer state.

Its constructor accepts `alpha` but stores only `rank` and the derived
`alpha / rank` scale. It rejects zero rank and non-finite alpha, learning rate,
beta values, epsilon, RMSNorm epsilon, or GDN attention scale. It also checks
that every `Dims` and `GdnDims` field was derived from the same model
configuration, including RMSNorm epsilon and the GDN scale. This avoids a
seemingly valid tape whose flattened dimensions describe a different model.

## Adam updates

`AdamState` stores first moments `m`, second moments `v`, and an update counter
`t` in maps keyed by parameter name. A key is expected to identify one stable
parameter tensor and shape for its lifetime. The training tape names GQA arrays
by model layer and factor (`l{layer}_a_q`, `l{layer}_b_q`, `l{layer}_a_v`, and
`l{layer}_b_v`); GDN arrays use analogous projection-specific names.

The update for an element uses:

```text
m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
theta -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

The timestep is per key, not global. One logical LoRA optimization round calls
`step` for several A and B tensors; a global counter would cause tensors later
in that round to use a larger bias-correction exponent than their actual number
of updates. With decoupled weight decay selected, the implementation first
applies `theta -= learning_rate * weight_decay * theta`, then takes the Adam
gradient step. Otherwise it performs standard Adam and ignores `weight_decay`.

The micro trainer uses Adam with beta values `0.9` and `0.999`, epsilon `1e-8`,
zero weight decay, and no decoupling. It cycles deterministically through the
provided pairs: training step `s` uses pair `(s - 1) % pairs.len()`.

## Single-event online refit

`adapt_step` is the smaller alternative to the full token-level tape. It fits
one existing `LoraLayer` to a desired output correction for a single input
activation. It is useful only when the caller already has an activation and a
target LoRA delta for one projection; it does not run the transformer.

With `scale = alpha / rank`, it minimizes the unnormalized squared error:

```text
h        = A @ input
delta    = scale * B @ h
residual = delta - target_delta
loss     = sum(residual^2)
```

The exact derivatives are:

```text
dL/dB[i, r] = 2 * scale * residual[i] * h[r]
dL/dA[r, j] = 2 * scale * (B^T @ residual)[r] * input[j]
```

`adapt_step` computes both gradient matrices, reports the loss and the
Euclidean norm of their concatenation, and applies in-place SGD updates to B
and A. `compute_lora_gradients` performs the same arithmetic but returns the
raw `LoraGradients` without mutating the adapter, which makes it suitable for
inspection or for a different optimizer.

Both entry points reject a missing `(layer_idx, module)` as `TuneError::Training`
and reject input or target slices whose lengths differ from `d_in` or `d_out`.
The local APIs do not impose a policy on learning rate or activation values;
callers that require finite-value or step-size checks must apply that policy
before calling them.

## Exact adapter blending

Some inference paths consume one adapter slot. `blend_lora_adapters` converts a
weighted collection of adapters into a standard single adapter on the CPU, so
the consumer need not dispatch one kernel per source adapter.

For source adapter `e`, let `s_e = alpha_e / rank_e` and let `w_e` be its
mixture weight. Its contribution is:

```text
Delta_e(x) = s_e * B_e @ (A_e @ x)
```

For a particular `(layer_idx, module)` group, the blend is exact, not a
low-rank approximation of the weighted sum:

```text
A_blend = vertical_concat(A_1, ..., A_N)
B_blend = horizontal_concat((w_1 * s_1) * B_1, ..., (w_N * s_N) * B_N)

B_blend @ (A_blend @ x) = sum_e w_e * Delta_e(x)
```

The result's A shape is `(sum_e rank_e, d_in)` and its B shape is
`(d_out, sum_e rank_e)`. The mixture coefficient and each source scale are
folded into B exactly once. The returned adapter uses `alpha == rank` and
therefore a configuration scale of one.

Grouping happens independently for every projection key in the union of source
keys. A source that lacks a key contributes nothing to that group. As a result,
different groups can have different layer ranks. The adapter-level `rank` is
the largest blended layer rank, but its matching alpha keeps the common adapter
scale at one for every group. A zero mixture weight is allowed; it still takes
part in the concatenation and consumes rank and allocation budget.

Before allocating, blending rejects:

- an empty adapter collection or a non-finite mixture weight;
- adapters that share a projection but disagree on `d_in` or `d_out`;
- a source A/B buffer that does not match its declared row-major shape;
- integer overflow in rank or size arithmetic;
- a summed rank above `MAX_BLEND_RANK_TOTAL` (`4096`) for one projection; or
- total planned elements above `MAX_BLEND_TOTAL_ELEMENTS` (`1 << 30`, about
  4 GiB of `f32` storage) across all projection groups.

The aggregate limit matters independently of the per-projection rank cap: a
large adapter can have many individually permitted projections whose combined
allocation is not acceptable.

## Manifest schema and governed loading

The serde-gated `LoraManifest` is a JSON document with a `version` and an
unordered list of adapter entries. `LoraManifest::new()` creates version 1.
Deserialization preserves the supplied `version` rather than enforcing it, so
callers that need a strict schema-version gate should check the field after
parsing.

Each `ManifestEntry` requires these fields:

| Field | Meaning | Enforced by governed load? |
| --- | --- | --- |
| `id` | Opaque adapter ID; a UUID is conventional. | Yes, if the safetensors header has `adapter_id`. |
| `name` | Human-readable identifier. | Provenance only. |
| `owner` | Responsible team or person. | Provenance only. |
| `uri` | Relative path to the adapter file. | Yes; must remain under the supplied base directory. |
| `integrity_sha256` | SHA-256 of the complete adapter file. | Yes. |
| `base_model_rev` | Base model revision used for training, or `"none"`. | Yes when running revisions are supplied. |
| `tokenizer_rev` | Tokenizer revision used for training, or `"none"`. | Yes when running revisions are supplied. |
| `rank` | Adapter's LoRA rank. | Yes; must equal the parsed adapter rank. |
| `alpha` | Adapter's LoRA alpha. | Yes; compared with `1e-4` tolerance. |
| `target_modules` | Declared module allow-list. | Yes; actual adapter modules must be a subset. |
| `dtype` | Training/save dtype label. | Provenance only; tensor parsing validates real data separately. |
| `status` | `approved`, `quarantined`, or `revoked`. | Yes; only `approved` is allowed. |

For example:

```json
{
  "version": 1,
  "adapters": [
    {
      "id": "support-v1",
      "name": "Support terminology",
      "owner": "search-team",
      "uri": "adapters/support-v1.safetensors",
      "integrity_sha256": "<sha256 of complete file>",
      "base_model_rev": "abc123",
      "tokenizer_rev": "def456",
      "rank": 8,
      "alpha": 16.0,
      "target_modules": ["q_proj", "v_proj"],
      "dtype": "f32",
      "status": "approved"
    }
  ]
}
```

`status` is intentionally the only approval field. `approved` is represented
by `status == "approved"`; a parallel boolean could disagree with the richer
`quarantined` and `revoked` states. Serde uses these lowercase snake-case
strings and rejects an unknown status value.

### Manifest I/O boundary

Manifest JSON is expected to be small. `LoraManifest::load` caps its on-disk
size at 64 MiB: it first rejects an oversized metadata length, then reads at
most 64 MiB plus one sentinel byte so a file that grows after the metadata
check also fails. It rejects non-UTF-8 content before JSON parsing. This avoids
materializing an attacker-controlled multi-gigabyte document merely to learn
that it is not a manifest.

`to_json` emits pretty JSON and `save` writes that UTF-8 representation. A
parse or serialization error is reported as a serialization error; filesystem
and bounded-read failures are I/O or validation errors as appropriate.

### Fail-closed admission sequence

The manifest-driven loader accepts a vector only if every entry passes every
applicable check. It has no partial-success or silent-skip path. Before opening
any adapter file, it pre-scans all entries: one `quarantined` or `revoked`
entry rejects the entire request and prevents file I/O for otherwise approved
entries.

For every approved entry, the loader then performs these ordered checks:

1. Recheck that the status is approved.
2. Resolve `uri` below the caller's base directory. Absolute paths and `..`
   components are rejected, and canonical base and target paths prove that a
   symlink cannot escape the base directory.
3. Require the target to exist and be readable as part of canonicalization and
   the bounded file read.
4. Read the bounded adapter bytes and require their SHA-256 to equal
   `integrity_sha256`.
5. Parse those bytes as a valid PEFT LoRA adapter.
6. Require the parsed rank to equal manifest `rank`.
7. Require parsed alpha to equal manifest `alpha` within `1e-4`; this also
   catches a missing file alpha that falls back to rank when the manifest
   declares a different value.
8. Require every actual target module to appear in the manifest's
   `target_modules` list.
9. When `inference-hook` is active and a model configuration is supplied,
   require adapter dimension validation against that model.
10. If the safetensors header contains `adapter_id`, require it to equal `id`.
11. When running revisions are supplied, require base-model and tokenizer
    revisions to match unless an explicit permissive override is in effect.

The whole-file integrity digest belongs in the manifest rather than in the
safetensors header: putting a file's complete digest inside that same file
would change the bytes being hashed. Header provenance fields are therefore
advisory metadata, while the manifest remains the authority for whole-file
integrity and admission status.

`RunningRevisions::strict` is the normal serving mode. It treats the literal
`"none"` as a real value, not a wildcard, so both sides must use `"none"` for
an untracked revision. `RunningRevisions::permissive` allows a mismatch only
for controlled migration work and marks the returned `LoadedAdapter` with
`rev_mismatch_overridden = true`; callers should make that override visible.
Passing no running revision context skips revision enforcement, and passing no
model configuration skips model-dimension validation. Those modes are useful
for offline inspection but are weaker admission checks than live serving.

