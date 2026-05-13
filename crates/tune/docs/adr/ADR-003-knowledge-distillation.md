# ADR-003: Knowledge Distillation Pipeline

**Status**: Accepted
**Date**: 2026-05-13
**Crate**: lattice-tune

## Context

`lattice-tune` must generate training data for student neural networks by asking large teacher LLMs (Claude, GPT-4, Gemini, local) to classify conversation intents. This "knowledge distillation" replaces hand-labeling: the teacher generates soft probability distributions over intent classes, which become the training targets for a small student network trained via `lattice-fann`.

The pipeline must handle: (1) multiple teacher providers behind a common configuration interface, (2) confidence filtering (skip low-confidence labels), (3) input sanitization against prompt injection, (4) a clear data contract between labeled examples and training examples, and (5) statistics tracking for monitoring distillation quality.

## Decision

A `DistillationPipeline` struct orchestrating `TeacherConfig`, `DistillationConfig`, and batch/single labeling. Teacher API calls are described via prompt formatting; actual HTTP calls are marked as placeholder (the `reqwest` implementation is not yet wired). The data contract from labeled results to `TrainingExample` is fully defined and tested.

### Key Design Choices

#### Teacher Output Format and Intent Labels

The teacher system prompt instructs the LLM to output a JSON object with probability scores for exactly six intent classes:

| Class            | Meaning                                |
| ---------------- | -------------------------------------- |
| `continuation`   | Natural conversation continuation      |
| `topic_shift`    | User changing subject                  |
| `explicit_query` | Direct question or information request |
| `person_lookup`  | Looking up a contact or person         |
| `health_check`   | Health/wellness inquiry                |
| `task_status`    | Checking task or todo status           |

These map to `IntentLabels` — a flat struct with one `f32` field per class. Scores from the teacher are expected to sum to approximately 1.0 (soft labels). The `IntentLabels::softmax_normalize()` method is available for normalization when `DistillationConfig::normalize_labels = true`.

#### Input Sanitization

`RawExample::to_prompt()` applies two layers of sanitization before sending to the teacher:

1. Strip control characters (except `\n`, `\t`, `\r`) to prevent prompt injection via control sequences
2. Truncate each message to `MAX_MESSAGE_LENGTH = 10_000` characters and the total prompt to `MAX_PROMPT_LENGTH = 50_000` characters

Truncation appends `\n[truncated]` to signal the cut. Context messages are numbered and formatted with the current message labeled separately, matching the system prompt's expected structure.

#### Confidence Threshold

`DistillationConfig::min_confidence: Option<f32>`. When set, any labeling result below this threshold is marked as skipped (not included in training data). The `DistillationStats::skipped` counter tracks these. A confidence score is provided by the teacher response (currently `0.85` in the placeholder; real implementations parse it from the response).

#### Data Contract: LabelingResult to TrainingExample

The pipeline does not compute embeddings — that is the responsibility of the calling layer. `to_training_examples` takes:

- `results: &[LabelingResult]` — labeled outcomes from `label_batch`
- `context_embeddings: &[Vec<Vec<f32>>]` — one embedding vector per context message per example
- `message_embeddings: &[Vec<f32>]` — one embedding vector per example

The method pairs them by index (returns `TuneError::DimensionMismatch` on length mismatch), skipping failed results. Each `TrainingExample` gets an `ExampleMetadata` recording the teacher model name (`TeacherConfig::display_name()`), the labeling timestamp (`Utc::now()`), and the confidence score.

#### Statistics

`DistillationStats` tracks per-run: `total_processed`, `successful`, `failed`, `skipped`, `total_latency_ms`, `avg_latency_ms`, `avg_confidence`, and `label_distribution` (count of dominant class per successful example). Updated after each `label_single` call. `reset_stats()` clears for a fresh run.

#### Batch vs. Single

`label_batch(raws)` calls `label_single` for each example, collecting both successes and failures. Failures are represented as `LabelingResult::failure` (error field set, default labels, zero confidence) so the batch result is complete and the caller can audit failures without an early-return error path.

### Alternatives Considered

| Alternative                                         | Pros                                    | Cons                                                                                                                  | Why Not                                                                                                            |
| --------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Streaming async API (`Stream<Item=LabelingResult>`) | Better for large datasets, backpressure | Requires async runtime in `lattice-fann` consumer, complex error handling                                             | The current pipeline is synchronous from `lattice-fann`'s perspective; async is only in the teacher API call layer |
| Hard labels (argmax) instead of soft labels         | Simpler training loss                   | Loses teacher calibration signal; soft labels carry distribution information useful for training classification heads | Knowledge distillation's core value is the soft probability distribution                                           |
| Embedding inside pipeline                           | Single pipeline call                    | Embedding is provider-dependent and may be cached from the inference layer                                            | Separation of concerns: labeling ≠ embedding                                                                       |
| Per-class confidence scores                         | More granular quality signal            | Teachers don't natively provide per-class confidence; increases parsing complexity                                    | Overall response confidence is sufficient for filtering                                                            |

## Consequences

### Positive

- `to_training_examples` has a complete typed contract: every field in `TrainingExample` is populated, including `ExampleMetadata.teacher_model` for auditability
- Prompt sanitization prevents injection even if conversation data contains adversarial content
- `DistillationStats` enables monitoring distillation quality without external tooling
- `normalize_labels` option handles teachers that return unnormalized logits vs. already-normalized probabilities

### Negative

- The actual HTTP client (`reqwest`) is not wired — `label_single` is a placeholder that returns simulated labels. The interface contract is fully defined but the implementation awaits the async HTTP layer.
- Confidence score is a single `f32` — no per-class breakdown, no calibration information
- `label_batch` is sequential (no parallelism). Large batches against rate-limited teacher APIs will be slow.

### Risks

- `to_training_examples` uses index-aligned slices (results, context_embeddings, message_embeddings). If the caller filters results before passing them, the index alignment breaks. The `DimensionMismatch` error catches length mismatches but not index-misalignment bugs where lengths happen to match after filtering.
- `MAX_PROMPT_LENGTH = 50_000` is conservative for most teacher APIs but may need adjustment for context-heavy conversations with many messages.

## References

- `crates/tune/src/distill/pipeline/distill.rs` — `DistillationPipeline`, `label_single`, `label_batch`, `to_training_examples`
- `crates/tune/src/distill/pipeline/types.rs` — `LabelingResult`, `DistillationStats`, `RawExample`, prompt sanitization
- `crates/tune/src/distill/teacher/config.rs` — `TeacherConfig` presets (claude_sonnet, claude_haiku, gpt4, gemini_pro, local)
- `crates/tune/src/distill/teacher/mod.rs` — `TeacherProvider` enum
- `crates/tune/src/data/example.rs` — `IntentLabels`, `TrainingExample`, `ExampleMetadata`
