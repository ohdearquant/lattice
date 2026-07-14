# lattice-tune documentation

These guides hold the longer-form design and format references for
`lattice-tune`. Source rustdoc stays focused on public API contracts and
load-bearing invariants; start here when you need the wider subsystem context.

## Architecture and topic guides

| Guide | Scope |
| --- | --- |
| [design.md](design.md) | Crate architecture, subsystem boundaries, lifecycle hand-offs, features, error boundary, and exact-gradient CLI workflow |
| [data.md](data.md) | `TrainingExample` format, six-label order, metadata, filtering, batching, statistics, and splitting |
| [distill.md](distill.md) | Teacher configuration, endpoint policy, prompt construction, pipeline behavior, accounting, and label-to-example conversion |
| [lora-core.md](lora-core.md) | Core adapter representation, low-rank application, and training mechanics |
| [lora-router.md](lora-router.md) | Adapter selection, router behavior, and online updates |
| [lora-io.md](lora-io.md) | PEFT safetensors, manifests, loading, saving, and adapter artifact governance |
| [train.md](train.md) | Training configuration, loop execution, callbacks, checkpoints, and GPU behavior |
| [registry.md](registry.md) | Model records, storage, lineage, live deployment, shadow evaluation, and rollback |

### Suggested reading paths

- **Prepare teacher-labeled data:** [distill.md](distill.md) →
  [data.md](data.md) → [train.md](train.md).
- **Fine-tune or serve an adapter:** [lora-core.md](lora-core.md) →
  [lora-io.md](lora-io.md) → [lora-router.md](lora-router.md), when routing
  applies.
- **Promote a trained model:** [train.md](train.md) →
  [registry.md](registry.md).
- **Understand how the pieces fit:** start with [design.md](design.md), then
  follow the relevant topic links.

## Architecture decisions

The following records are part of this documentation set. Do not move or edit
the retired crate-local ADR pointers; each identifies the maintained
repository-wide decision record.

| Decision | Current location |
| --- | --- |
| Multi-provider teacher strategy | [ADR-001-teacher-providers.md](ADR-001-teacher-providers.md) |
| Model registry with lineage | [ADR-002-model-registry.md](ADR-002-model-registry.md) |
| Fine-tuning pipeline | [crate-local pointer](adr/ADR-001-finetuning-pipeline.md) → [maintained ADR-027](../../../docs/adr/ADR-027-finetuning-pipeline.md) |
| Knowledge distillation pipeline | [crate-local pointer](adr/ADR-003-knowledge-distillation.md) → [maintained ADR-030](../../../docs/adr/ADR-030-knowledge-distillation.md) |
| LoRA adapter management | [crate-local pointer](adr/ADR-004-lora-adapter-management.md) → [maintained ADR-031](../../../docs/adr/ADR-031-lora-adapter-management.md) |
| Training callbacks | [crate-local pointer](adr/ADR-005-training-callbacks.md) → [maintained ADR-032](../../../docs/adr/ADR-032-training-callbacks.md) |
| JIT adaptation | [crate-local pointer](adr/ADR-006-jit-adaptation.md) → [maintained ADR-033](../../../docs/adr/ADR-033-jit-adaptation.md) |
| Dataset pipeline | [crate-local pointer](adr/ADR-007-dataset-pipeline.md) → [maintained ADR-034](../../../docs/adr/ADR-034-dataset-pipeline.md) |

Use an ADR to understand an accepted design choice and rejected alternatives;
use the topic guides to implement against the current code.
