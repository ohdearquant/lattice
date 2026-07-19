# lattice-tune design

`lattice-tune` owns the training-side lifecycle around Lattice models. It
keeps data preparation, teacher labeling, training orchestration, adapter
work, and model registration as separate subsystems with explicit hand-offs
rather than one all-purpose pipeline.

This page is the map of that composition. Follow the linked topic guides for
formats, algorithms, and operational rules; follow the ADRs for the
alternatives that led to a boundary.

## Crate map

```text
                         raw text / external records
                                     │
                       ┌─────────────▼─────────────┐
                       │          distill          │
                       │ teacher policy + labels   │
                       └─────────────┬─────────────┘
                                     │ labels
                   embeddings supplied by application
                                     │
                       ┌─────────────▼─────────────┐
                       │            data           │
                       │ examples, datasets, batch │
                       └─────────────┬─────────────┘
                                     │ batches
                       ┌─────────────▼─────────────┐
                       │           train            │
                       │ loop, configuration, state │
                       └─────────────┬─────────────┘
                                     │ trained parameters / metrics
           ┌─────────────────────────┴──────────────────────────┐
           ▼                                                    ▼
┌──────────────────────┐                            ┌──────────────────────┐
│         lora         │                            │       registry       │
│ adapter train, I/O,  │                            │ versions, storage,   │
│ routing, inference   │                            │ shadow, rollback     │
└──────────┬───────────┘                            └──────────────────────┘
           │
           ▼
 lattice-inference (only through optional feature bridge)
```

The diagram has two related but independent paths:

- The **student-model path** turns examples into batches, trains a model, and
  records a versioned artifact and its provenance.
- The **adapter path** creates or loads a LoRA adapter, optionally trains its
  low-rank parameters, persists it through the LoRA I/O surface, and can
  attach it to `lattice-inference` when the appropriate feature is enabled.

The registry is not an implicit last step for either path: callers explicitly
register a model and its bytes. Likewise, data creation does not implicitly
invoke a teacher or an embedding service.

## Module ownership and hand-offs

| Module     | Owns                                                                                                           | Receives                                   | Produces                                                         | Guide                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `data`     | `TrainingExample`, labels, metadata, in-memory datasets, batches                                               | Embeddings and labels from a caller        | Filtered datasets and cloned batches                             | [data.md](data.md)                                                                       |
| `distill`  | Teacher configuration, endpoint policy, raw prompt handling, labeling result accounting                        | Raw conversational text                    | Results; training examples only after caller supplies embeddings | [distill.md](distill.md)                                                                 |
| `train`    | Generic training config, loop state, callbacks, checkpointing, and optional GPU surface                        | `Dataset` batches                          | Metrics, checkpoints, and trained model state                    | [train.md](train.md)                                                                     |
| `lora`     | Low-rank adapter representation, application, persistence, online/router work, and optional backward utilities | Base-projection context and adapter inputs | Adapter weights or inference-time deltas                         | [lora-core.md](lora-core.md), [lora-router.md](lora-router.md), [lora-io.md](lora-io.md) |
| `registry` | Registered model records, storage/query interfaces, live swaps, shadow evaluation, and rollback                | Model metadata and weight artifacts        | Versioned, deployable model records                              | [registry.md](registry.md)                                                               |
| `error`    | `TuneError` and `Result`                                                                                       | Failures from every subsystem              | A common, domain-specific error vocabulary                       | This page                                                                                |

Each subsystem has a deliberately small data boundary:

- `distill` does not generate embeddings or invoke training;
- `data` does not fetch or persist records;
- `train` consumes batches and does not choose a teacher or deployment target;
- `lora` is an adapter concern, not a dependency of the generic data model;
- `registry` records and serves artifacts but does not retrain them; and
- `error` standardizes failure reporting without becoming a control plane.

## Lifecycle 1: teacher-labeled classification data

```text
conversation history + current message
                  │
                  ▼
        RawExample::to_prompt()
                  │
                  ▼
    teacher result / LabelingResult
                  │
        caller aligns embeddings
                  │
                  ▼
        TrainingExample + metadata
                  │
                  ▼
     Dataset::with_config(...) ──► batches()
                  │
                  ▼
             TrainingLoop
                  │
                  ▼
       metrics + registered artifact
```

The important boundary in this route is between text and vectors. A
`RawExample` contains text; a `TrainingExample` contains vectors. The
distillation pipeline does not own an embedding model, so the caller must
produce context and message embeddings in the same order as its labeling
results. [distill.md](distill.md) defines result conversion; [data.md](data.md)
defines the vector and six-label contract.

At present `DistillationPipeline` is a placeholder teacher integration. It
fails closed when live teacher transport is unavailable; deterministic fixed
labels require the non-default `simulated-teacher` feature and explicitly named
simulation methods. Those methods do not format or bound a prompt — they never
inspect the example's content at all — and maintain statistics; the pipeline
makes no HTTP request and does not read an API-key environment variable. A
deployment should supply a real provider client before relying on this route
to generate labels.

## Lifecycle 2: train and register a model

The generic training route starts at `Dataset`:

1. Construct or receive valid `TrainingExample` values.
2. Choose eligibility and batch policy with `DatasetConfig`.
3. Iterate the dataset by epoch, allowing the training loop to consume
   `Batch` values.
4. Keep the training configuration, metrics, and checkpoint state with the
   trained artifact.
5. Explicitly create a registry record with its metadata and weights.
6. Use registry shadow or rollback facilities as an explicit deployment
   workflow when appropriate.

The training guide owns optimizer, callback, checkpoint, early-stopping, and
GPU details. The registry guide owns status transitions, storage behavior,
integrity, shadow traffic, and rollback details. This separation prevents
training mechanics from defining deployment policy.

The [model-registry ADR](ADR-002-model-registry.md) records why the registry
keeps versioning and lineage. The retired crate-local
[fine-tuning](adr/ADR-001-finetuning-pipeline.md),
[callbacks](adr/ADR-005-training-callbacks.md), and
[JIT adaptation](adr/ADR-006-jit-adaptation.md) ADR pointers lead to their
maintained repository-wide counterparts.

## Lifecycle 3: LoRA adapters

LoRA adapters are a second training output, distinct from a generic registered
student model:

```text
base model projections
        │
        ▼
LoraConfig + LoraLayer / LoraAdapter
        │
        ├──► apply a low-rank delta during an inference projection
        │
        ├──► serialize or load adapter artifacts
        │
        └──► optional backward and online/router update paths
```

The core adapter representation and projection application live in
[lora-core.md](lora-core.md). The router and online-update path is documented
in [lora-router.md](lora-router.md); the safetensors, manifests, and artifact
formats belong in [lora-io.md](lora-io.md). These documents should be read
together when adding a target module or adapter format.

The `inference-hook` feature is the narrow dependency bridge to
`lattice-inference`: it enables the implementation of
`lattice_inference::lora_hook::LoraHook` for `LoraAdapter`. Without that
feature, the adapter remains a training-side type and `lattice-tune` does not
need the inference crate for this path.

The LoRA representation decision is documented by the retired
[ADR-004 pointer](adr/ADR-004-lora-adapter-management.md), which directs
readers to the maintained root ADR.

## Features and dependency direction

`lattice-fann` is the required lower-level training dependency. The inference
crate is optional and only enters through feature-gated adapter injection or
backward training. That direction matters: training may consume inference
capabilities, while inference remains usable without `lattice-tune`.

| Feature          | Adds or enables                                           | Boundary it affects                 |
| ---------------- | --------------------------------------------------------- | ----------------------------------- |
| `std`            | Standard-library support; enabled by default              | Base crate                          |
| `serde`          | Serialization derives and JSON support                    | Data, config, checkpoints, metadata |
| `safetensors`    | Safe adapter/artifact serialization support               | LoRA I/O                            |
| `sqlite`         | SQLite-backed registry storage; enabled by default        | Registry                            |
| `gpu`            | WGPU-backed GPU training surface                          | Train                               |
| `gpu-tests`      | Hardware-dependent GPU tests                              | Train testing                       |
| `inference-hook` | `LoraHook` implementation for `LoraAdapter`               | LoRA → inference bridge             |
| `train-backward` | Inference backward/f16 support plus safetensors and serde | Exact-gradient LoRA binaries        |
| `mixture`        | Experimental online adapter-selector refit                | LoRA routing                        |

Feature gates select integrations; they do not erase the ownership
boundaries above. For example, enabling `safetensors` enables an artifact
format but does not automatically register a saved adapter, and enabling
`inference-hook` does not automatically attach an adapter to a running model.

## Error boundary

Every public subsystem can return the crate alias
`Result<T> = std::result::Result<T, TuneError>`. `TuneError` keeps failures
classified by the boundary that detected them:

| Family                       | Examples                                                        |
| ---------------------------- | --------------------------------------------------------------- |
| Data                         | Dataset errors, missing examples, invalid batches, dimensions   |
| Teacher                      | Provider failures and request timeouts                          |
| Training                     | Training errors and non-convergence                             |
| Configuration and validation | Invalid configuration or user/input validation                  |
| Registry and storage         | Missing/duplicate models and backing storage failures           |
| Artifact integrity           | Serialization, I/O, weight checksum, and memory-budget failures |

Callers should add context while preserving the variant whenever possible.
This lets a UI or orchestration layer distinguish an invalid local data record
from a teacher failure or a model-registry problem.

## Exact-gradient LoRA command-line workflow

The `train_grad`, `train_grad_layer23`, and `train_grad_full` executables are
private training tools compiled only with `train-backward`. They share a
minimal `train_common` module for JSONL loading, argument lookup, default
paths, and an explicit trust-but-verify (TBV) gate. Their typed option parsing
and usage strings remain intentionally local because their supported flags and
defaults differ.

### Shared JSONL samples

The loader accepts one JSON object per nonempty line with string `prompt` and
`completion` fields. It concatenates them, tokenizes the prompt and full text
separately, and records the number of prompt tokens as `completion_start`:

```text
Sample {
  tokens: tokenize(prompt + completion),
  completion_start: tokenize(prompt).real_length,
}
```

Rows with an empty or missing field, fewer than two full tokens, a full token
length above `--seq-len`, an empty tokenized prompt, or no completion tokens
after tokenization are skipped. Malformed nonempty JSON or I/O failures are
returned as errors. Collection stops once `--max-train` or `--max-valid`
accepted rows have been reached.

The default model directory is
`$HOME/.lattice/models/qwen3.5-0.8b` (or `./.lattice/models/qwen3.5-0.8b` if
`HOME` is absent). The default data directory is `data/lora-train`.

`ArgView` deliberately preserves the original permissive lookup behavior:
the first matching value flag wins; a dangling or missing value returns
`None`; presence flags are presence-only; and unknown flags are ignored. Each
binary decides how to parse or reject the returned text.

### Trust but verify is fail-closed

Before a gradient binary trusts an assembled or cached chain, `verify_tbv`
compares its masked negative log likelihood to the real model's
`compute_token_nlls` result. The absolute difference must be at most
`TBV_MAX_ABS_DIFF = 1e-2`. A non-finite reference, candidate, or difference,
or a larger difference, returns an error; it is not merely a diagnostic.

This gate is deliberately independent of local gradient checks. It catches a
wrong forward assembly—such as a missing residual, norm convention, or layer
handoff—before optimization can make its output appear plausible.

### `train_grad_full` tape

`train_grad_full` is the multi-layer exact reverse-mode LoRA trainer for
Qwen3.5. It materializes the inclusive layer range
`first_layer..=TOP_LAYER`, where `TOP_LAYER` is 23 and the default first layer
is 19. The default range therefore covers a GQA LoRA layer at 19, frozen-base
GatedDeltaNet layers at 20–22, and a GQA LoRA layer at 23. GDN base weights
remain frozen while the command can maintain GDN LoRA slots.

For each sample, it captures the frozen prefix output entering `first_layer`
and caches RoPE tables. The materialized tape then follows:

```text
h = frozen prefix output
for each materialized layer:
    pre = rms_norm(h, pre_norm)
    mixed = GQA(pre; LoRA) or GDN(pre; frozen base plus adapter slot)
    h_mid = h + mixed
    h = h_mid + swiglu(rms_norm(h_mid, post_norm))
logits = lm_head · rms_norm(h, final_norm)
loss = cross entropy over completion positions
```

The loss gradient propagates backward through the head, every materialized
FFN, mixer, residual, and norm. This is essential for a lower GQA layer: its
adapter gradient is only meaningful if the reverse path crosses the
intervening frozen GDN layers correctly.

Qwen3.5 layer and final RMSNorm are shifted:
`x * inverse_rms * (1 + gamma)`. The command precomputes `1 + gamma` for
those norms because the tape helpers accept plain gamma-like weights. GQA
q/k-normalization stays raw because the GQA forward path applies its own
shift internally.

The command performs a zero-adapter TBV check before it trains or checks
gradients. It also refuses a cache whose logits allocation would exceed 2 GiB,
and uses `valid.jsonl` only for held-out evaluation when requested.

### Gradient validation

`--gradcheck` uses nonzero random initialization for both A and B adapter
matrices so the A gradient and GDN gate paths cannot pass vacuously. It probes
both the largest-magnitude analytical gradients and deterministic strided
indices that a top-k selection could overlook. Each selected entry is compared
to central finite differences at `0.25×`, `0.5×`, `1×`, and `2×` of
`--fd-eps`; it keeps the best relative difference to avoid judging a deep
`f32` chain by a single roundoff- or truncation-dominated step. The default
center is `4e-3` because the chain's observed roundoff makes much smaller
central-difference steps unreliable.

Normal training initializes A with small random values and B with zeros, so
the initial delta reproduces the base model and B receives the first
nonzero update. The command's usage text is the authority for exact option
defaults, including model/data paths, layer range, steps, rank, alpha, sample
limits, validation, saving, probes, and finite-difference epsilon.

## Reading order

For a new implementation or integration:

1. Start with [data.md](data.md) for the cross-subsystem record contract.
2. Read [distill.md](distill.md) if labels come from a teacher.
3. Read [train.md](train.md) for generic training execution and checkpoints.
4. Choose the appropriate LoRA guide for adapters and inference integration.
5. Read [registry.md](registry.md) before persisting or deploying a trained
   model.
6. Consult the linked ADRs when changing a boundary or revisiting an accepted
   design decision.

## ArgView

`ArgView` is intentionally a lookup helper rather than a command-line parser.
It preserves the gradient binaries' legacy behavior: the first matching value
flag wins; an absent flag or a final flag with no following value returns
`None`; a presence flag is true whenever its token occurs; and unknown flags
are ignored. It does not validate a value's syntax or decide whether a missing
value is an error. Each binary owns those typed parsing decisions and its own
usage contract.

This permissiveness is a compatibility boundary. Do not add global validation
to `ArgView`, because it would silently change a binary's historical accepted
argument set. New flags should be parsed and checked by their owning binary.

## verify_tbv

`verify_tbv` is the shared trust-but-verify gate for cached or assembled
forwards. It compares a candidate masked NLL to the real model's masked
`compute_token_nlls` result and returns the two values plus their absolute
difference. The gate fails when either input or the computed difference is
non-finite, or when the difference is greater than
`TBV_MAX_ABS_DIFF = 1e-2`; it includes the supplied context in the error.

The boundary is intentionally inclusive: a difference exactly equal to the
tolerance passes. This is a correctness gate rather than a diagnostic, so a
caller must stop before optimization whenever it fails.

## train_grad

`train_grad` is the smallest exact reverse-mode LoRA trainer. It adapts only
the Qwen3.5 `lm_head`, whose frozen-transformer input is the final normalized
hidden vector `H_t` for each completion source position. The command runs the
real 24-layer forward once per sample, caching `(H_t, base_logits, target)`;
subsequent steps do not re-run the transformer.

For a rank-`r` adapter with A and B factors, each cached position computes:

```text
a_h      = A · H_t
logits_t = base_logits_t + scale · B · a_h
loss_t   = cross_entropy(logits_t, next_token)
```

It averages masked completion-token loss and gradients, then applies Adam to
the two adapter factors. Both gradient computation and reported NLL use the
LoRA-adjusted logits, so the curve measures the trained adapter rather than
the frozen base. Before optimization, a zero-B TBV check compares cached loss
to the real model's loss; no mismatch is treated as a harmless warning.

## train_grad_layer23

`train_grad_layer23` adapts `q_proj` and `v_proj` in Qwen3.5's top GQA layer,
layer 23. It captures the frozen prefix residual entering that layer and the
RoPE tables once per sample. Every lower layer and every base weight remains
frozen; only the four low-rank factors `(A_q, B_q, A_v, B_v)` are updated.

For each completion source position, the materialized chain is:

```text
normed   = rms_norm(h_in, pre_attn_norm)
attn_out = gated_GQA(normed; LoRA_q, LoRA_v)
h_mid    = h_in + attn_out
h_out    = h_mid + swiglu(rms_norm(h_mid, post_attn_norm))
logits   = lm_head · rms_norm(h_out, final_norm)
loss     = cross_entropy(logits, next_token)
```

The reverse pass crosses the head, final norm, FFN, residual, and GQA
attention, then deliberately discards the gradient below layer 23 because the
prefix is frozen. Layer and final RMSNorm weights are shifted as
`1 + gamma` before calling helpers that expect ordinary weights. GQA q/k norm
weights stay raw because its forward implementation applies that shift.

Training starts with random A and zero B, so the initial adapter reproduces
the base while B receives a nonzero first update. The zero-adapter TBV check
therefore validates the entire layer-23-plus-head reconstruction. When saved,
the adapter uses B as row-major `[d_out, rank]` and A as `[rank, d_in]`; gated
`q_proj` has `d_out = 2 * q_dim`, covering both Q and gate rows.

## train_grad_full details

`train_grad_full` extends the same exact-gradient approach across an inclusive
`first_layer..=23` materialized range. Each GQA layer gets a `q_proj`/`v_proj`
adapter slot; each GatedDeltaNet layer gets slots for its five supported
projections. The trainer borrows all frozen base weights, caches the frozen
prefix and RoPE tables, reconstructs the range, and fails closed if its
zero-adapter NLL does not match the real model.

GDN factor construction must go through `GdnLoraParams::shaped`, the sole
shape authority. In particular, the B factors for `in_proj_b` and `in_proj_a`
are sized by `value_heads`, not key heads. This matters for asymmetric GDN
geometries: deriving their sizes from `num_kh` can construct a superficially
plausible adapter whose beta and alpha projections have the wrong shape.

Gradcheck fills both A and B with nonzero noise so neither A gradients nor
gate paths pass vacuously. It probes both large analytical gradients and
deterministic strided positions, evaluating central differences at four
scales around `--fd-eps`. Taking the best relative error across those scales
is deliberate: a single finite-difference step can be dominated by roundoff
or truncation in this deep `f32` chain. The default center, `4e-3`, avoids the
too-small-step roundoff regime observed for the full model.

In normal training B begins at zero while A uses a
`1 / sqrt(input_dimension)` amplitude compatible with `mlx_lm`'s LoRA
initialization. The base forward is therefore reproduced at step zero and B
moves first. `valid.jsonl`, when available and enabled, is never trained on;
its NLL is the held-out counterpart to the training NLL when judging whether
optimization is learning rather than memorizing.
