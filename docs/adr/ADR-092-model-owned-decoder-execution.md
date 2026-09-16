# ADR-092: Model-owned decoder execution and a neutral serving boundary

**Status**: Proposed\
**Date**: 2026-09-16\
**Crate**: lattice-inference

## Context

Decoder generation needs one request-control implementation while retaining architecture-specific
execution. Qwen3.5 combines GDN recurrence and GQA cache state. Gemma 4 E2B uses distinct local
and global softmax attention, donor-cache relationships and per-layer inputs. Its CPU forward
stores unscaled QK dot products as attention scores and applies the configured final logit
softcap. A common numerical cache or layer implementation would erase meaningful differences.

The existing Metal worker constructs its model inside a dedicated thread. HTTP-facing handles
carry jobs and events; the state remains on its owner thread. The CPU server instead retains
shareable model weights and executes requests on blocking workers. A common serving interface
must preserve these different ownership and scheduling arrangements.

[ADR-090](ADR-090-shared-decoder-generation.md) specifies a private execution-session trait and
common generation policy. This record specifies how a loaded model supplies that session and
how frontends use it without concrete model types. It does not redefine token policy, acceptance
identity or numerical execution. The inspected source reference is
`7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c`; proposed interfaces below are design contracts, not
compiled API.

## Decision

Use the private, object-safe `DecoderSession` boundary from ADR-090 for model execution. Construct
sessions through private model adapters selected by a loader-local dispatch. Expose a neutral,
opaque runtime handle to the serving modules and binaries. The handle accepts neutral requests
and delivers existing output/events; it does not expose an enum of concrete models, concrete
tokenizers or mutable backend state.

Initial shared text execution covers canonical Qwen3.5 CPU, supported Qwen3.5 Metal profiles and
Gemma 4 E2B CPU. An explicit unavailable backend is rejected. Automatic backend selection records
the actual selected backend and entry profile in the private diagnostics specified by ADR-090.
Model family, checkpoint role, artifact format and backend availability are separately validated.
This decision does not add Gemma Metal, new Gemma variants, BERT generation or multimodal Gemma.

### Key Design Choices

#### One execution trait, with model-owned storage

The shared driver calls ADR-090's `prefill`, `select`, `metadata`, `decode` and `finish` operations.
Concrete sessions retain or safely borrow typed model weights and own their request scratch,
cache, prepared input and prediction identity. The driver owns policy and RNG progression.
There is no downcast, public tensor return type, common KV vector or generic cache-length setter.

CPU sessions select and score over borrowed logits. The Metal adapter preserves compact selection
and uses bounded metadata operations when admitted by the entry profile. It must not copy a dense
vocabulary vector merely to satisfy a common return type. Candidate selection, policy-final token,
metadata identity and subsequent consumption remain separate as ADR-090 requires.

The private loader may use an enum containing the supported concrete owners. Its match runs at
loading or request preparation, not in HTTP policy and not once per numerical layer. It supplies
a `DecoderSession` whose lifetime is bounded by the owner. A local session borrowing a model must
finish and drop before that model is replaced or an adapter revision changes. Avoid self-referential
storage and lifetime extension; a request-scope borrow or existing safe owned handle supplies the
lifetime relationship. Trait-object storage may allocate at request setup; the warm token loop
must not allocate merely to cross the interface.

#### Opaque handles preserve execution placement

Frontends use an opaque `PipelineClient` and neutral load/request descriptions. These are proposed
names, not new model extension traits. The runtime implementation may dispatch between CPU and
Metal internally. CPU requests retain the existing admission limits and blocking-executor
concurrency with shared immutable weights and separate mutable sessions. Do not serialize all CPU
requests onto one new FIFO just to resemble Metal.

For Metal, only a Send load description/factory and the job/event channels cross into the owner
thread. Construct and destroy the Metal owner and request sessions there. Do not require
`DecoderSession: Send + Sync`, add unsafe auto-trait implementations, or move a constructed Metal
session into a task. Preserve readiness-before-listen, admission permits, queued cancellation,
in-flight cancellation, adapter-control ordering and bounded shutdown from the existing worker.
No operation on an adapter may race with a request borrowing its mutable owner.

Model-specific constructor signatures and vision/adapter dispatch currently inside the worker
move behind the private runtime boundary as their routes migrate. The worker's transport and
lifecycle logic stay shared. Existing public compatibility paths can re-export supported worker
handles without exposing concrete model state in the frontend implementation.

#### Preparation remains model-aware below the frontend

Keep HTTP normalization separate from model prompt rendering. Preserve omitted versus explicit
options until the prompt adapter applies the selected model's defaults. Gemma uses its own
tokenizer, template and control tokens; Qwen thinking defaults are not copied onto it. Validate
prompt length without tokenizer truncation hiding an overflow, and preserve each entry profile's
error precedence, zero-budget behavior and unsupported-option refusals.

Gemma's adapter uses the existing typed forward/cache implementation, including the local/global
attention plan, donor-cache constraints, per-layer inputs, scaling and softcap. Add the supported
EOS-aware serving behavior through shared policy. Retain `generate_greedy` and
`generate_greedy_with_probe` as fixed-count diagnostics, including their existing final-token
evaluation and probe semantics. They are references for comparison, not the implementation of
the new serving loop.

#### Advanced routes are explicit, and compatibility is preserved

Keep supported public generation wrappers and old imports for neutral config/output types. A
wrapper may retain its entry-profile preparation while invoking the new common controller.
Neither a direct-entry wrapper nor a feature-disabled error stub is deleted simply because a
shared session exists. Preserve downstream `GenerateOutput` literals without adding fields or
changing exhaustiveness.

Prefix-cache reuse and speculative state repair remain typed extensions, with checked capability
negotiation and model/adapter/tokenizer/cache identity. Existing serving uses the prefix-cache
streaming route; ordinary generation alone is not full serving coverage. A stop inside an accepted
speculative span must restore or invalidate the unused suffix before reuse. Keep the batch-verifier
route explicitly outside shared-driver coverage until its separate disposition under ADR-090 D6.
Preserve existing multimodal and embedding routes without claiming that this text-only interface
has unified them. No unsupported capability becomes a silent no-op or CPU fallback.

#### Relationship to existing architecture decisions

ADR-009's Decision remains unchanged: “Each family has a distinct config struct, weight struct,
and forward pass implementation. No shared forward pass code between BERT and Qwen3.” Its
“Scope clarification: decoder generation control” remains unchanged. This execution/ownership
interface shares scheduling of model-specific operations, not the numerical implementation of
those operations. BERT and Qwen embedding/pooling remain independent. No part of ADR-009 is
superseded by this record.

ADR-080's existing worker, policy and normalization contracts remain binding. ADR-082 continues
to define the supported Gemma E2B profile and its numerical validation. ADR-090 continues to own
generation policy, profile refusal, prediction lifecycle, compatibility exclusions, measurement
budgets and rollout ordering. ADR-086's retired surface stays retired. A loader façade cannot
restore a retired API under a neutral spelling.

#### Completion requires both dependency and execution evidence

Add a dedicated `pipeline_boundary_contract` integration target. Proposed commands, **not run
and not yet implemented**, are:

```text
cargo test --locked -p lattice-inference --test pipeline_boundary_contract -- --nocapture
cargo test --locked -p lattice-inference --features metal-gpu,f16 --test pipeline_boundary_contract -- --nocapture
```

The first configuration runs on the CPU platform matrix; the second must run on a macOS Metal
host. These source-contract commands alone do not establish numerical or device execution.

The target must derive its roots from every `src/serve` module and every inference binary source,
including Cargo auto-discovered binaries and their nested modules. It reports the selected Cargo
targets, discovered modules, resolved dependencies and actually read files. Require nonempty
known-positive roots. Account for supported cfg combinations; a disabled local feature cannot
hide a forbidden dependency in another supported build.

Within those roots, reject concrete model/config/backend-state dependencies in imports, type
aliases, fields, generic parameters, constructors and calls, including locally inferred concrete
values. Follow bounded import/re-export chains into the same crate and public API index instead
of trusting a differently named symbol. Track module/path/include destinations outside the root
directories. Unknown relevant macro expansion, unresolved name or ambiguous classification must
fail completeness rather than be discarded. Supported macros require an explicit analyzed
expansion or contract. This is a scoped dependency resolver, not a claim that a substring scan
implements Rust name resolution.

The allowed crossing is a reviewed neutral operation with opaque storage. A helper under an
innocuous path that returns a concrete model, exposes its methods or leaks its config violates
the boundary. Generic model values cannot be smuggled through callbacks or blanket adapters in
the frontend. Compatibility imports for neutral generation DTOs remain valid when they resolve
to those neutral definitions. CLI model strings, descriptive comments and model-specific binary
names are not type dependencies and are not renamed by this decision.

Required controls include a renamed import alias, a re-export chain, an inferred constructor
result, a generic wrapper, a macro-hidden model construction, and a sibling module reached through
an include/import. Each must be rejected or explicitly unresolved. A comment/string containing a
model name must pass. An unreadable source and missing expected frontend root must fail. These
controls make a smaller discovered population distinguishable from successful migration.

Even that structural check has a semantic false negative: an opaque allowed operation can still
call the legacy generation loop internally. Separate real consumer controls must therefore prove
driver execution on both servers for Qwen CPU, Qwen Metal and Gemma E2B CPU. Require actual token
output and a private marker emitted inside the common controller with the selected profile and
backend. A fake session, a startup marker, or an adapter marker before an old-loop call does not
count. Deliberately bypass the controller and require failure; repeat for the prefix-cache serving
route. Preserve explicit batch-verifier exclusion in the result.

Keep golden/logit bounds, seeded token/event order, forced-final-token controls, cancellation at
queue/prefill/empty-text/decode boundaries, exactly-one terminal output, failed-session reuse
refusal, and feature-disabled/backend-unavailable tests. Compile a downstream-style output literal
through the old import and reject an attempted cross-thread Metal-session move. Source extraction
changes must also run guards which inspect the edited files, including the Metal lock contract.

Allocation and readback instrumentation must bracket actual worker execution on a common base/head
consumer. Preserve zero added compact-route dense readbacks, no interface-caused warm-loop
allocations, and ADR-090's per-diff timing dispositions. A pure-move label does not grant a waiver;
the ADR-087 proof or reachable paired measurement decides it. Numerical/lifetime fixes are
separate from routing changes. This record reports no measured cost or passing new gate.

### Alternatives Considered

| Alternative                                   | Advantage                             | Cost or failure                                                                          | Decision                                                                   |
| --------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Public model enum matched by frontends        | Explicit variants and static dispatch | Repeats family knowledge and exposes exhaustive variant churn                            | Keep enum dispatch private at loading/preparation only                     |
| Generic model type through HTTP state         | Static checking and specialization    | Couples frontend/runtime types to model/backend and spreads worker lifetime constraints  | Keep generics below the session boundary                                   |
| Trait wrapping whole old `generate` methods   | Easy initial routing                  | Preserves independent policy loops and cannot prove shared token control                 | Reject as the execution seam; retain only migration compatibility wrappers |
| Universal numerical forward/cache abstraction | Apparent kernel reuse                 | Conflates GDN, GQA, Gemma donor/sliding caches, attention and pooling                    | Reject; preserve typed model operations                                    |
| Split the large Metal file before routing     | Smaller files                         | Adds cfg/path/probe changes and removes a stable comparison target before the seam works | Characterize first; extract after live shared routes                       |
| Independent Gemma server first                | Fast isolated demonstration           | Duplicates HTTP policy and postpones the shared-route proof                              | Use Gemma to prove the common runtime                                      |
| One FIFO for every backend                    | Simple uniform ownership              | Can silently reduce existing CPU concurrency                                             | Preserve backend-specific execution placement behind neutral handles       |

## Consequences

### Positive

- Serving and CLI control no longer need concrete decoder state or family-specific token policy.
- Model mathematics and mutable execution resources remain under their typed owners.
- A new decoder must prove admission, session semantics and real shared execution before joining
  the common serving route.

### Negative

- Temporary compatibility wrappers and explicit advanced-route exclusions remain during migration.
- Loading, prompt preparation and execution placement still require model/backend-specific code.
- The full frontend dependency criterion also requires narrowly scoped embedding and diagnostic
  façades; decoder routing alone does not satisfy it.

### Risks

- Dynamic dispatch could impose allocations/readbacks or change request concurrency; controls must
  measure the real route and fail deliberate regressions.
- A source-only resolver can miss unsupported language constructs; unresolved inputs must refuse
  classification, with compile and execution coverage retained separately.
- A neutral façade can hide old control flow; only shared-driver bypass controls distinguish that
  from genuine migration.
- The source pin predates subsequent changes. Before implementation, recheck the selected owners,
  feature gates and compatibility calls at the new base. Implementation remains subject to
  acceptance of this Proposed decision.

## References

- [ADR-009: model architectures](ADR-009-model-architectures.md),
  [ADR-080: duplicated contracts](ADR-080-consolidation-duplicated-contracts.md),
  [ADR-082: Gemma E2B](ADR-082-gemma4-e2b-support.md),
  [ADR-086: retired decode API](ADR-086-retire-legacy-qwen-decode-api.md),
  [ADR-087: benchmark dispositions](ADR-087-bench-compare-gate-calibration-and-coverage.md),
  [ADR-090: shared generation](ADR-090-shared-decoder-generation.md).
- [Metal direct generation at the inspected revision](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/forward/metal_qwen35.rs#L9310).
- [Metal streaming wrapper and cancellation](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/forward/metal_qwen35.rs#L11138).
- [Worker construction and thread ownership](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/serve/metal_worker.rs#L1313).
- [Current serving prefix-cache route](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/serve/metal_worker.rs#L1458).
- [CPU frontend ownership](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/bin/lattice/serve.rs#L270).
- [Gemma attention scores](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/model/gemma4_model.rs#L371),
  [logit softcap](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/model/gemma4_model.rs#L491),
  [fixed-count diagnostic](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/model/gemma4_model.rs#L561).
- [Embedding-specific serving ownership](https://github.com/ohdearquant/lattice/blob/7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c/crates/inference/src/serve/embeddings.rs#L1).
