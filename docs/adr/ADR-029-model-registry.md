# ADR-029: Model Registry with Lineage

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-tune

## Context

lattice-tune trains compact student models via knowledge distillation. Production ML
systems require:

1. **Reproducibility** - Given a deployed model, trace back to exact training
   configuration, dataset, and metrics that produced it.

2. **Fine-tuning Lineage** - When a model is fine-tuned, record its parent model to
   understand evolutionary history and enable rollback.

3. **Version Coordination** - Multiple model versions may coexist (A/B testing, canary
   deployments). Clear versioning prevents deploying wrong models.

4. **Status Lifecycle** - Models progress through validation stages before production.
   Enforce gates to prevent deploying untested models.

5. **Integrity Verification** - Detect weight tampering or corruption via checksums
   before deployment.

## Decision

Implement an in-process `ModelRegistry` with `RegisteredModel` entries supporting semver
versioning and parent-child lineage:

```rust
pub struct RegisteredModel {
    pub id: Uuid,
    pub name: String,               // e.g., "intent_classifier"
    pub version: String,            // Semver e.g., "1.2.3"
    pub status: ModelStatus,
    pub metadata: ModelMetadata,
    pub registered_at: DateTime<Utc>,
    pub weights_path: Option<String>,
    pub weights_hash: Option<String>,  // SHA-256
    pub parent_id: Option<Uuid>,       // Lineage tracking
    pub children: Vec<Uuid>,           // Fine-tuned descendants
}
```

**Semver Versioning**: Use `MAJOR.MINOR.PATCH` format with comparison methods
(`is_newer_than`, `version_tuple`) for version ordering.

**ModelStatus Lifecycle**:

```text
Pending --> Validated --> Staged --> Production
                                         |
                                         v
                                   Archived/Deprecated
```

| Status     | Gate                    |
| ---------- | ----------------------- |
| Pending    | Registration complete   |
| Validated  | Test suite passed       |
| Staged     | Ready for canary/shadow |
| Production | Live traffic            |
| Archived   | Superseded              |
| Deprecated | Do not use              |

**Lineage Tracking**: Fine-tuned models reference `parent_id` pointing to base model.
Parent models accumulate `children` UUIDs for forward traceability.

## Consequences

### Positive

- **Full Traceability**: Any production model can be traced to training config, dataset,
  and parent lineage.
- **Rollback Capability**: If v1.2.0 degrades, promote v1.1.0 from Archived to
  Production.
- **Safe Promotion**: Status gates prevent deploying Pending models to production.
- **Integrity Checking**: SHA-256 weights hash detects corruption before inference.
- **Query by Version**: `find_latest(name)` returns highest semver for automated
  deployments.

### Negative

- **Storage Overhead**: Each version stores full weights (no delta compression).
- **In-Memory Limitation**: Default `ModelRegistry::in_memory()` does not persist across
  restarts.
- **Manual Lineage**: Parent-child relationships require explicit `with_parent()` calls;
  not auto-detected.
- **Orphan Risk**: Deleting a parent does not cascade to children, potentially leaving
  orphaned lineage references.

### Mitigations

- For persistent storage, integrate `ModelRegistry` with SQLite or filesystem backend
  (future work).
- Document lineage workflow in examples to ensure developers set `parent_id` on
  fine-tuned models.
- Validate `parent_id` exists before registration to prevent orphan references.

## Alternatives Considered

### Alternative 1: File-Based Versioning

**Approach**: Store models as files with naming convention `{name}_v{version}.bin`.

**Pros**:

- Simple implementation
- No registry code
- Works with standard filesystem tools

**Cons**:

- No metadata (training config, metrics) association
- No lineage tracking
- Manual version parsing prone to errors
- No status lifecycle enforcement

**Rejected because**: Lacks structured metadata and lineage, which are core requirements
for reproducibility in ML workflows.

### Alternative 2: No Versioning

**Approach**: Overwrite `{name}.bin` on each training run.

**Pros**:

- Minimal storage
- Simplest implementation

**Cons**:

- No rollback capability
- Cannot trace model to training parameters
- Breaks reproducibility requirement

**Rejected because**: Violates reproducibility principle. Cannot answer "what produced
this model?" after a training run.

### Alternative 3: External Registry (MLflow, W&B)

**Approach**: Integrate with MLflow Model Registry or Weights & Biases.

**Pros**:

- Battle-tested at scale
- Rich UI for exploration
- Built-in artifact storage
- Team collaboration features

**Cons**:

- External dependency
- Network latency for registration
- Licensing considerations
- Overkill for embedded/edge deployments

**Rejected because**: lattice-tune targets embedded and edge inference where external
service dependencies are undesirable. The `Custom(String)` variant in `TeacherProvider`
demonstrates preference for self-contained solutions. Future integration with external
registries can be added as an optional backend.

## References

- RegisteredModel: `crates/tune/src/registry/model.rs`
- ModelRegistry: `crates/tune/src/registry/storage.rs`
