# ADR-088: Sealed Native Embedding Preparation and Attestation

**Status**: Proposed
**Date**: 2026-08-17
**Crates**: lattice-embed, lattice-inference
**Depends on**: ADR-003, ADR-005, ADR-014, ADR-015, ADR-016, ADR-017, ADR-087
**Amends on acceptance**: ADR-003, ADR-005, ADR-014, ADR-015, ADR-016, ADR-017
**Related constraint**: ADR-070 applies fused batching only to legacy/`Auto` paths unless a later
ADR-088 amendment changes the prepared identity and migration contract.

## Context

An embedding vector is meaningful only inside the exact vector space that produced it. A model
label and output dimension do not identify that space. Checkpoint bytes, tokenizer selection,
configuration, query and document transforms, pooling, normalization, truncation, provider code,
numeric execution policy, adapters, and caches can all change the output while retaining the same
label and dimension.

The existing native service is deliberately convenient rather than identity-bearing:

- `NativeEmbeddingService` loads lazily from ambient paths and environment variables;
- BERT's download check covers fewer files than the loader can consume, and checking a path does
  not bind later opens to the checked directory generation;
- tokenizer precedence means that adding `tokenizer.json` can change the selected tokenizer while
  a previously checked `vocab.txt` remains unchanged;
- optional configuration files and loader fallbacks can change effective semantics without being
  represented in `ModelConfig`;
- Qwen directory selection, Metal selection, CPU fallback, and persistent-cache import depend on
  ambient state at different times;
- Qwen's persistent cache is keyed by a model label and active dimension rather than a complete
  vector-space identity; and
- `ModelProvenance::hash` intentionally hashes lightweight metadata including load time. It is an
  audit hint, not a content digest.

The opened-file mmap boundary in ADR-003 prevents a path replacement from changing which bytes a
particular mapping reads. It does not establish that independently opened weights, tokenizer,
configuration, and shard files belong to one directory generation, nor does it identify the full
set of bytes and semantics used by an embedding service.

Downstream systems that persist vectors therefore cannot derive a complete immutable identity from
the current public API. Adding an identity method to `EmbeddingService` would be actively unsafe:
arbitrary implementations and legacy `NativeEmbeddingService` values could claim an identity
without proving that their service is backed by those bytes.

## Decision

### D1: add a separate prepared capability; do not bless the legacy trait

`lattice-embed` adds a concrete preparer and prepared native embedding type. The prepared type is
created only by the preparer and implements `EmbeddingService` itself. Its private state
inseparably owns:

1. the loaded model used for every request;
2. the sealed private snapshot from which that model was constructed;
3. Lattice's canonical pre/post attestation reports and any caller-supplied supplementary evidence
   when `prepare_with_attestor` is used; and
4. a bounded effective descriptor of every code-owned vector semantic.

The prepared value has no `into_parts`, raw-model getter, mutable snapshot getter, path getter,
file-descriptor getter, mmap token, or conversion into a legacy `NativeEmbeddingService`.
Attestation and descriptor accessors return borrows. The value is not `Clone`; callers share it as
an `Arc`, so the snapshot and model have one lifetime.

`EmbeddingService` gains no default identity or attestation method. `NativeEmbeddingService`,
`CachedEmbeddingService`, and out-of-tree implementations remain valid compatibility surfaces, but
they are not evidence suitable for a persisted vector-space identity.

The intended public shape is:

```rust,ignore
pub struct NativeEmbeddingPreparer { /* pinned snapshot parent + shared resource domain */ }
#[derive(Clone)]
pub struct NativeEmbeddingDrain { /* admission gate + jobs/objects/leases tracker */ }
pub struct PreparedModelDirectory { /* opaque pinned absolute source capability */ }
pub struct PreparedSnapshotDirectory { /* opaque pinned absolute directory capability */ }
pub struct NativePreparationLimits { /* private finite per-request ceilings */ }
pub struct NativeResourceBudget { /* private finite shared ceilings */ }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttestationAlgorithm { Sha256V1 }
pub struct CanonicalAttestationReport {
    algorithm: AttestationAlgorithm,
    digest: [u8; 32],
}
pub struct SupplementaryAttestationEvidence { /* private, bounded and digest-bound */ }

impl PreparedModelDirectory {
    pub fn open(path: &Path) -> Result<Self>;
}

impl PreparedSnapshotDirectory {
    pub fn open(path: &Path) -> Result<Self>;
}

impl ResolvedNativeEmbeddingSpec {
    pub fn try_new(
        source: PreparedModelDirectory,
        model_config: ModelConfig,
        backend: PreparedBackendPolicy,
        cache: PreparedCachePolicy,
    ) -> Result<Self>;
}

impl NativePreparationLimits {
    pub fn try_new(/* every finite file/path/snapshot/parse/retained ceiling */) -> Result<Self>;
}

impl NativeResourceBudget {
    pub fn try_new(
        max_concurrent_preparations: NonZeroUsize,
        max_concurrent_encodes: NonZeroUsize,
        max_retained_bytes: NonZeroU64,
        max_transient_work_bytes: NonZeroU64,
    ) -> Result<Self>;
}

impl NativeEmbeddingPreparer {
    pub fn open(
        snapshot_parent: PreparedSnapshotDirectory,
        budget: NativeResourceBudget,
    ) -> Result<(Self, NativeEmbeddingDrain)>;

    pub async fn prepare(
        &self,
        spec: ResolvedNativeEmbeddingSpec,
        limits: NativePreparationLimits,
    ) -> Result<PreparedNativeEmbedding>;

    pub async fn prepare_with_attestor<A, F>(
        &self,
        spec: ResolvedNativeEmbeddingSpec,
        limits: NativePreparationLimits,
        attestor: F,
    ) -> Result<PreparedNativeEmbedding>
    where
        A: CheckpointAttestor,
        F: FnMut() -> A + Send + 'static;
}

pub struct PreparedNativeEmbedding {
    // private: loaded model, sealed snapshot, report, effective descriptor
}

impl CanonicalAttestationReport {
    pub fn algorithm(&self) -> AttestationAlgorithm;
    pub fn digest(&self) -> &[u8; 32];
}

impl SupplementaryAttestationEvidence {
    pub fn try_new(
        algorithm: AttestationAlgorithm,
        digest: [u8; 32],
        payload: Vec<u8>,
    ) -> Result<Self>;
}

impl PreparedNativeEmbedding {
    pub fn canonical_attestation(&self) -> &CanonicalAttestationReport;
    pub fn supplementary_attestation(&self) -> Option<&SupplementaryAttestationEvidence>;
    pub fn effective_descriptor(&self) -> &EffectiveEmbeddingDescriptor;
}

impl NativeEmbeddingDrain {
    pub fn begin_drain(&self) -> Result<()>;
    pub async fn wait_released(&self) -> Result<()>;
}

impl EmbeddingService for PreparedNativeEmbedding { /* overrides every default entry point */ }
```

The exact module layout may change during implementation, but the capability and lifetime
boundaries above are normative. The directory constructors perform the platform-specific
absolute-component pinning described below; callers cannot construct either capability from a raw
`PathBuf`. The source and snapshot-parent capabilities must be distinct and non-nested. Limits and
budgets have no `Default`, unchecked, zero, or unbounded constructor: every finite ceiling is
explicit and validated with checked arithmetic before either capability performs preparation work.

### D2: v1 accepts only resolved local BERT input and has no ambient fallback

`ResolvedNativeEmbeddingSpec` contains an explicit absolute source root, validated `ModelConfig`,
backend policy, and cache policy. Preparation does not read `HOME`, `LATTICE_MODEL_CACHE`,
`LATTICE_QWEN_MODEL_DIR`, `LATTICE_EMBED_DIM`, `LATTICE_OFFLINE`, `LATTICE_NO_GPU`, or any other
environment variable. It performs no network access or automatic download. A convenience resolver
may read configuration before constructing a resolved spec, but the resolved value captures those
choices once and the preparation operation consumes only that value.

The first version supports only these four BERT-family local variants, using an explicit
`CpuPinned` backend: `BgeSmallEnV15`, `BgeBaseEnV15`, `BgeLargeEnV15`, and `AllMiniLmL6V2`. Their
prepared tokenizer layouts are WordPiece. The tokenizer
admission closure is limited to the WordPiece and BPE variants described in D4; no third tokenizer
family is admitted. BPE declarations are admitted by that closed profile, but no current BERT row
uses one and Qwen decoder variants are rejected by this D2's BERT-only model boundary. `CpuPinned`
disables Accelerate, AMX, and other vendor-library dispatch that cannot be completely identified,
captures one Lattice-owned
scalar/SIMD capability and dispatch profile at preparation, and uses that immutable profile for the
service lifetime. Every output-affecting operation—including matmul, attention softmax/exp,
GELU/tanh, layer normalization, pooling, and final normalization—uses a versioned Lattice-owned
kernel; prepared mode does not call a dynamically resolved system math routine whose implementation
is absent from the descriptor. It rejects Qwen, remote models, adapters, Metal, automatic backend
selection, vendor fallback, and per-call backend fallback with a typed unsupported-policy error.
This is a deliberate safe subset, not a claim that Qwen is permanently unsupported.

`MultilingualE5Small`, `MultilingualE5Base`, and `ParaphraseMultilingualMiniLmL12V2` are deferred to
a follow-up ADR. Their Unigram/SentencePiece tokenizer layouts are outside the v1 closure: a
tokenizer with `model.type` exactly `"Unigram"`, a `Metaspace` pre-tokenizer, or a
`sentencepiece.bpe.model`/`spiece.model` file selected by precedence is rejected during preparation
with a typed error naming the unsupported tokenizer model. The published `tokenizer.json` for
`ParaphraseMultilingualMiniLmL12V2` declares `model.type` `"Unigram"` over a 250,002-entry
vocabulary with a `Precompiled` normalizer and a `Sequence` pre-tokenizer, and the runtime loader
routes it to the SentencePiece tokenizer. The prepared closure rejects it with the same typed
unsupported-tokenizer error, and preparation never falls through to a lower-priority tokenizer
candidate after that rejection.

`lattice-inference` therefore gains an explicit BERT CPU-kernel-policy seam. Prepared services pass
their frozen `CpuPinned` profile; the legacy service passes `Auto` and retains today's behavior.
The direct pinned BERT constructor is the differential oracle for prepared output. Prepared mode
does not claim bit parity with an `Auto` legacy service that selects Accelerate or AMX.

A prepared value is a single-model capability. `supports_model` returns true for exactly the sealed
`EmbeddingModel`; `model_config` returns the frozen resolved configuration for that model. The
prepared implementation overrides every allocating or role-routing default on `EmbeddingService`:
`embed`, `embed_one`, `embed_with_role`, the hidden prevalidated hook, `embed_query`, and
`embed_passage`. Each entry point checks the caller's model against the sealed model before text
validation, prompt-prefix allocation, cloning, admission, or native work. Generic, Query, and
Passage calls use exactly the transforms recorded in the descriptor. A wrong-model call returns a
typed unsupported-model error with zero prefixing, cloning, admission, filesystem, or native work;
the non-fallible `model_config` trait method is not identity evidence for an unsupported model.

Prepared v1 also fixes batch execution to `CanonicalPerItemV1`, a descriptor field and provider-
semantics input. One admitted batch is processed as an ordered sequence of independent single-item
`CpuPinned` executions, with no packed-batch kernel and no neighboring-input-dependent padding or
accumulation order. The result order matches the request order. Throughput comes from independently
admitted concurrent calls, not a numerically distinct packed path. A later fused-batch mode requires
a new descriptor value, provider-semantics revision, identity golden, and exact invariance or
migration analysis.

Qwen support requires a later amendment that closes its complete shard/index/config inventory,
pins one realized backend without per-call fallback, measures and admits peak resident memory, and
removes or identity-binds both its in-memory and persistent embedding caches.

### D3: preparation creates one private, handle-copied snapshot

Preparation performs this sequence before returning a service:

1. Resolve and validate the vector-affecting source inventory without following an unbounded or
   escaping directory graph.
2. Open the source root and every selected entry with descriptor-relative, no-follow mechanics.
   The absolute source-root component walk rejects symlink/reparse components. Exactly one relative
   internal file-symlink hop is accepted when lexical normalization stays inside the pinned root and
   the descriptor-opened target is a regular file; its bytes are materialized into an ordinary
   snapshot file. Absolute links, link chains, directory links, escaping links, special files,
   multiply linked regular files, traversal names, and platforms without equivalent handle-pinning
   primitives fail closed.
3. Copy from those already-open source handles into a cryptographically random private snapshot.
   The preparer receives an already-resolved absolute snapshot-parent capability, pins it at
   construction, and never consults `TMPDIR` or another environment-dependent temporary path. The
   snapshot is created exclusively and owner-only outside the source root; logical relative names
   are normalized and unique.
4. Validate the complete snapshot inventory, all bounded metadata, tokenizer selection, model
   geometry, and the effective descriptor. No loader fallback is permitted in prepared mode.
5. Compute the first Lattice canonical attestation over the complete snapshot schedule, and,
   when `prepare_with_attestor` is used, obtain supplementary evidence from a fresh caller
   attestor. Then seal the snapshot read-only.
6. Load the model only from the sealed snapshot under the existing ADR-003 mmap trust boundary.
7. Re-read inventory and geometry from the same snapshot, reconstruct the effective descriptor,
   and compute a second canonical attestation using a fresh opened-handle read sequence. When
   `prepare_with_attestor` is used, obtain supplementary evidence from another fresh caller
   attestor as well.
8. Require the inventory, effective descriptor, and canonical reports to match exactly before
   publishing the prepared value. If `prepare_with_attestor` is used, its supplementary evidence
   digest binding and bounded payload must also validate on both passes and match; it is never a
   substitute for the canonical report.

The snapshot path is never returned. The prepared inner object retains the snapshot and every
resource needed by model mappings. Snapshot cleanup starts only after the loaded model and any
in-flight encoding job release their final reference. Under normal return, error, cancellation, or
panic unwinding, an unpublished partial snapshot is cleaned without exposing a partial service;
cleanup failure is observable. Process abort, power loss, and unlink failure can leave an
owner-only residue. `NativeEmbeddingPreparer::open` first acquires one cross-process exclusive
resource-domain lock for the pinned snapshot parent. The preparer, drain, every prepared inner, and
every job retain that domain lock transitively; another independently budgeted domain cannot open
the same parent until all of them are gone and receives a non-blocking typed busy error rather than
occupying a runtime worker. The fixed versioned domain-lock control is reusable and excluded from
generation census/deletion. Concurrent preparations inside one domain serialize census/recovery
through a supervisor-owned maintenance mutex while retaining their individual admission
reservations. Each snapshot additionally owns a cross-process generation lock retained by its
prepared inner object. Snapshot publication uses this crash-safe parent namespace protocol:

1. Choose a random bounded identifier. Exclusively create its parent-level lock file, acquire the
   OS lock on that same handle, fsync the file and parent, then atomically publish and fsync a
   versioned `creating` state record. The lock is held before a state record or data directory can be
   observed as live work; a lone lock file or state-replacement temporary is itself a recognized
   identifier-bound control state.
2. Create and fsync the identifier-bound data directory only after the lock and state record are
   durable. Populate it through descriptor-relative handles. No Lattice-created data directory can
   therefore be visible without an earlier durable control record.
3. After copy, validation, sealing, load, and the second attestation succeed, atomically replace and
   fsync the state record as `sealed`; the prepared inner retains the original lock handle for its
   lifetime.
4. Cleanup first atomically publishes and fsyncs `residue` while retaining the live lock, then
   removes data and finally its state/lock controls, fsyncing the parent at each committed boundary.
   If even the state replacement fails, the previous grammar-valid state remains and cleanup reports
   failure. A crash at any transition leaves either no generation artifact, a held live transition,
   or a recognized grammar-valid identifier-bound control/generation (including a lone lock,
   state-replacement temporary, or `creating`, `sealed`, or `residue` record); it never leaves an
   unmarked data directory.

After acquiring preparation admission, and before creating a new snapshot, the preparer performs a
bounded census of this closed namespace. It non-blockingly skips held locks; it may reclaim an
unlocked, old-enough `creating`, `sealed`, or `residue` generation and recognized orphan control
files through an idempotent state-specific cleanup. The disposition is closed:

| Census state                                                                                             | Required action before this preparation may publish                                                                                                                                                       |
| -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| grammar-valid generation or in-progress orphan control with its generation lock held                     | Skip that identifier; it is live work in this same exclusively locked/accounted domain.                                                                                                                   |
| grammar-valid unlocked `creating`, `sealed`, `residue`, lone lock, or state-temporary younger than grace | Return typed `SnapshotNamespaceRetry`; delete nothing and publish nothing.                                                                                                                                |
| grammar-valid unlocked state above at or older than grace                                                | Reclaim it completely and fsync the parent before continuing. Reclamation failure terminally closes preparation admission for this domain, returns typed `SnapshotNamespaceFault`, and publishes nothing. |
| unknown name/state, duplicate identifier/control, inconsistent triplet, or invalid control contents      | Return typed `SnapshotNamespaceCorrupt`, terminally close preparation admission, delete nothing, and publish nothing.                                                                                     |
| census entry/metadata-byte overflow or platform missing required lock/fsync/atomic-replace semantics     | Return the typed limit/unsupported error, terminally close preparation admission, delete nothing, and publish nothing.                                                                                    |

The fresh job never adopts pre-existing residue into its retained pool. Recovery deletes only bounded
recognized state without materializing its payload. If a stale generation—whether smaller or larger
than the new job's reservation—cannot be completely reclaimed, the parent is a terminal namespace
fault and no new bytes are created. Thus recovery needs no fictitious `ResidueCharge` source, and
“skip” is permitted only for a held generation already accounted by the same exclusive domain.

If normal cleanup cannot remove a snapshot, it leaves the last durable grammar-valid state as
reclaimable evidence and atomically transfers one `ResidueCharge` into a residue-debt record owned by
the supervisor before releasing any remainder. For an unpublished generation, the charge comes from
the preparation job's retained reservation; for a published generation, it comes from the prepared
object's retained lease. The charge covers every remaining snapshot data byte plus the measured
filesystem allocation or, where exact allocation is unavailable, the unreleased conservative ceiling
for state/lock/control files, normalized names, directories, and cleanup bookkeeping. It is never
smaller than the retained charge for objects that still exist. Model/job leases can then release,
but the resource domain does not make residue bytes available to new work. The first cleanup fault
permanently closes preparation admission for that resource domain; already-published services may
continue using the independent encode-work pool until the host begins drain, but no new retained
object can be published. The supervisor serializes the terminal transition and final prepared-inner
registration under one state mutex. That publication critical section includes the terminal check,
retained-lease transfer, registration as published, and commitment to the result channel. Exactly
two outcomes exist: publication linearizes first and the value is thereafter an already-published
service, or the fault linearizes first and the job transfers no published lease and cleans its
unpublished generation. A closed result receiver after the publication commit drops the registered
value through normal prepared cleanup; it does not roll back the linearization. Drain reports the
aggregate residue count, byte debt, and cleanup errors rather than
claiming success. The host closes and drops the faulted domain before opening a fresh preparer on
that snapshot parent; the fresh domain's bounded exclusive census reclaims the persisted residue, so
no cross-supervisor debt reconciliation is implied. Dropping the old closed domain may end its
in-memory accounting, but it does not erase the operational residue or make those disk bytes
reusable.

Read-only permissions and path privacy protect against accidental mutation and ambient path drift;
they are not a privilege or confidentiality boundary against hostile code running under the same
UID. Such code may discover, read, chmod, or race mutation of the private snapshot, and this ADR
does not guarantee detection of every hostile same-UID mutation. Deployments with that threat model
must add a distinct UID, sandbox, or platform-enforced immutable storage. Within the trusted-local-
filesystem model, the load and attestation contract rejects every observed add, delete, rename,
replacement, size change, or same-size content change in the vector-affecting snapshot closure
between the two passes.

### D4: the BERT inventory is explicit and loader precedence is evidence

Prepared BERT mode recognizes only the bounded model closure used by the BERT loader:

- `model.safetensors`;
- `config.json`;
- `tokenizer_config.json`;
- the supported tokenizer representation files (`tokenizer.json`, `vocab.txt`, `vocab.json`, and
  `merges.txt`) that are present and relevant to deterministic selection; and
- SentencePiece candidates (`tokenizer.model`, `sentencepiece.bpe.model`, and `spiece.model`) so
  their selection can be rejected explicitly rather than treated as an implicit fallback.

`model.safetensors`, `config.json`, and `tokenizer_config.json` are required in prepared v1. A
single-file BERT checkpoint is required; an index or shard layout is rejected until a later
inventory amendment specifies it.

Tokenizer selection freezes this exact precedence:

1. `tokenizer.json`;
2. `vocab.json` plus `merges.txt`;
3. `vocab.txt` plus `merges.txt`;
4. `vocab.txt`; and
5. `tokenizer.model`, `sentencepiece.bpe.model`, or `spiece.model`.

Every present recognized candidate that can participate in or shadow that selection is copied and
attested, even when it is lower in the precedence order. The effective descriptor names the
selected layout. A present malformed higher-priority candidate or a partial two-file layout fails;
prepared mode never falls through to a lower-priority candidate after such a failure. Adding a
higher-priority candidate therefore changes both inventory evidence and the selected-layout
descriptor instead of silently reusing an old identity. A selected SentencePiece candidate, or a
`tokenizer.json` whose `model.type` is `"Unigram"`, is rejected with a typed preparation error
naming the unsupported tokenizer model; it never falls through to a lower-priority WordPiece or
BPE candidate.

Malformed or missing required configuration fails; prepared mode does not derive silent defaults
from tensor shapes or a directory name. Files outside the recognized vector-affecting closure, such
as licenses and README files, are neither copied nor identity-bearing. If a future loader begins to
consume another file, its semantics revision and inventory contract must change together.

Prepared publication also closes the cross-artifact validation boundary that the legacy convenience
loader treats best-effort. Let `V`, `H`, `L`, `A`, `I`, `P`, and `T` be the required
`vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`,
`max_position_embeddings`, and `type_vocab_size` values from `config.json`. Every value is nonzero,
within its explicit preparation limit, and participates only in checked arithmetic; `A > 0`,
`H % A == 0`, and `layer_norm_eps` is finite and strictly positive. Before loading or publishing,
the prepared validator requires this exact tensor geometry:

| Tensor group                                                          | Required shape                  |
| --------------------------------------------------------------------- | ------------------------------- |
| word / position / token-type embeddings                               | `[V,H]` / `[P,H]` / `[T,H]`     |
| embedding LayerNorm weight and bias                                   | `[H]` each                      |
| each layer `0..L`: query, key, value, attention-output weights/biases | `[H,H]` / `[H]`                 |
| each layer `0..L`: both LayerNorm weight/bias pairs                   | `[H]` each                      |
| each layer `0..L`: intermediate weight/bias                           | `[I,H]` / `[I]`                 |
| each layer `0..L`: output weight/bias                                 | `[H,I]` / `[H]`                 |
| optional pooler pair, when either member is present                   | both present as `[H,H]` / `[H]` |

All required names exist exactly once, recognized encoder layer indices are exactly `0..L`, and no
configured layer is inferred from a partial tensor set. Every tokenizer vocabulary ID and configured
special-token ID that can be emitted is `< V`; every emitted token-type ID is `< T`; and tokenizer
vocabulary cardinality, word-embedding rows, and `config.json` agree. A mismatch is a typed
preparation error, never a warning, debug-only assertion, unchecked slice, or fallback.

Prepared BERT v1 also requires `ModelConfig::output_dim` to be absent and requires `H` to equal
both the sealed `EmbeddingModel::native_dimensions()` and the frozen `ModelConfig::dimensions()`.
Thus the vector length produced by the checkpoint, reported by the trait, and admitted by downstream
storage cannot diverge under a mislabeled same-family checkpoint. Output truncation remains a future
prepared-provider amendment rather than an inferred BERT behavior.

The realized sequence cap uses one closed precedence protocol. It scans
`tokenizer_config.json` before `config.json`, and within each file scans
`model_max_length`, `max_position_embeddings`, `n_positions`, `max_seq_len`, then
`truncation.max_length`. The first present key must be a positive exactly representable unsigned
integer; a malformed, zero, negative, fractional, or overflowing higher-priority value fails rather
than falling through. The selected raw value is capped at 2048 exactly as the legacy embedding
loader does, and that realized cap must be no greater than `P` or the actual position-embedding row
count. The descriptor records the selected source file, key, raw value, capped value, and truncation
protocol. This preserves accepted legacy truncation behavior for valid checkpoints while refusing a
configuration that could index beyond the sealed model.

Geometry alone is not sufficient. Prepared D4 also validates a closed set of output-affecting
`config.json` semantics before loading. `model_type` must be present and exactly `"bert"`.
`hidden_act` must be present and exactly `"gelu"`; the effective implementation value recorded in
the descriptor is `GeluTanhApproxV1`, the tanh-form GELU approximation
`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`, with Lattice's versioned Padé
`fast_tanh` kernel. It is not the erf-exact GELU and is not interchangeable with `relu`,
`gelu_new`, or another activation name. `position_embedding_type` must be present and exactly
`"absolute"`; absence is rejected rather than interpreted as an implicit default. The existing
numeric fields, including `layer_norm_eps`, remain required and validated as above.

The remaining BERT configuration allowlist is explicit. `is_decoder` and `add_cross_attention`
may be absent or `false`; `true` is rejected because this loader has no decoder or cross-attention
path. `pruned_heads` may be absent or an empty object; a non-empty value is rejected. The
`pad_token_id` field may be absent or must equal the selected vocabulary's `[PAD]` ID. The
inference-only or training-only fields `hidden_dropout_prob`, `attention_probs_dropout_prob`,
`classifier_dropout`, `use_cache`, `tie_word_embeddings`, `architectures`, `torch_dtype`,
`transformers_version`, `_name_or_path`, `id2label`, `label2id`, `problem_type`, `return_dict`,
`output_hidden_states`, and `output_attentions` are known output-irrelevant metadata for this
encoder and are ignored after bounded type validation, even when their declared values are
non-default.
No other `config.json` key is ignored: an unknown key, or a known BERT option that can change
embedding execution, is rejected with a typed preparation error. The descriptor records the
validated activation, position-embedding, decoder/cross-attention, and pooling-relevant semantic
profile, while actual tensor source dtypes and decode behavior are recorded from the sealed
SafeTensors inventory rather than trusted from `torch_dtype`.

### D4 tokenizer semantic closure

The tokenizer checks admit only WordPiece and BPE variants and cover declared behavior, not only
file precedence, vocabulary cardinality, and special-token IDs. Unigram/SentencePiece is outside
this closure and is rejected during preparation. `WordPieceTokenizer` in
`crates/inference/src/tokenizer/wordpiece.rs` always lowercases each Unicode scalar, maps its
supported accented forms and removes combining marks (therefore stripping accents from NFD input),
surrounds CJK characters with separators, surrounds punctuation with separators, converts Unicode
whitespace to ASCII-space separators, drops non-whitespace controls, and splits on the resulting
whitespace. It greedily longest-matches whole-word pieces first and `##` continuation pieces after
the first piece; an unmatched character emits the vocabulary `[UNK]` ID. It requires `[CLS]`,
`[SEP]`, `[PAD]`, `[UNK]`, and `[MASK]`; it emits `[CLS]` and `[SEP]` itself, pads with `[PAD]`,
and matches non-empty declared added/special tokens literally before normalizing surrounding text.
There is no cased, accent-preserving, alternate CJK, alternate punctuation, left-padding, or
decoder path in prepared WordPiece mode.

For a selected WordPiece `tokenizer.json`, the declared pipeline must be exactly this supported
shape, with no unlisted behavior-bearing members:

| Declaration      | Required value                                                                                                             |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `model.type`     | `"WordPiece"`; `model.vocab` is the selected vocabulary                                                                    |
| WordPiece model  | `unk_token` absent or `[UNK]`; `continuing_subword_prefix` absent or `##`; no unvalidated word-length cap                  |
| `normalizer`     | `BertNormalizer` with `clean_text=true`, `handle_chinese_chars=true`, `lowercase=true`, and `strip_accents=true` or `null` |
| `pre_tokenizer`  | `{"type":"BertPreTokenizer"}`                                                                                              |
| `post_processor` | `TemplateProcessing` with single `[CLS] $A [SEP]` and pair `[CLS] $A [SEP] $B:1 [SEP]:1`, with matching IDs                |

`strip_accents: null` is accepted only with `lowercase=true` and is canonicalized to effective
`strip_accents=true`, matching the Hugging Face BERT normalizer convention. Any other normalizer,
pre-tokenizer, post-processor template, or declared option is rejected with a typed preparation
error; the loader must not silently ignore a cased normalizer or a false `strip_accents` flag.

Added-token matching is also a closed declaration. The implementation consumes `content` and `id`
for literal matching before normalizing surrounding text (`WordPieceTokenizer`), while BPE also
consumes `special` to distinguish literal-rendered decode tokens from skipped control tokens
(`BpeTokenizer`). Every `added_tokens` entry must declare all five behavior-bearing flags. The
checked-in ADR-016 tokenizer fixtures contain these values:

| Fixture / `content` entries                                                                                              | (`single_word`, `lstrip`, `rstrip`, `normalized`, `special`) |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| BGE WordPiece: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`                                                              | (`false`, `false`, `false`, `false`, `true`)                 |
| Multilingual E5 Unigram: `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`                                                       | (`false`, `false`, `false`, `false`, `true`)                 |
| Qwen3 BPE control: `<\|endoftext\|>`, `<\|im_start\|>`, `<\|im_end\|>`, `<\|object_ref_start\|>`, `<\|object_ref_end\|>` | (`false`, `false`, `false`, `false`, `true`)                 |
| Qwen3 BPE control: `<\|box_start\|>`, `<\|box_end\|>`, `<\|quad_start\|>`, `<\|quad_end\|>`, `<\|vision_start\|>`        | (`false`, `false`, `false`, `false`, `true`)                 |
| Qwen3 BPE control: `<\|vision_end\|>`, `<\|vision_pad\|>`, `<\|image_pad\|>`, `<\|video_pad\|>`                          | (`false`, `false`, `false`, `false`, `true`)                 |
| Qwen3 BPE literal-rendered: `<tool_call>`, `</tool_call>`, `<\|fim_prefix\|>`, `<\|fim_middle\|>`                        | (`false`, `false`, `false`, `false`, `false`)                |
| Qwen3 BPE literal-rendered: `<\|fim_suffix\|>`, `<\|fim_pad\|>`, `<\|repo_name\|>`, `<\|file_sep\|>`                     | (`false`, `false`, `false`, `false`, `false`)                |
| Qwen3 BPE literal-rendered: `<tool_response>`, `</tool_response>`, `<think>`, `</think>`                                 | (`false`, `false`, `false`, `false`, `false`)                |

The admitted flag set is therefore `single_word=false`, `lstrip=false`, `rstrip=false`,
`normalized=false`, and `special=false` or `true`; missing flags are rejected too. These are the
only values whose behavior the literal matcher and BPE decode path reproduce. Any other value, or
any missing flag, is a typed preparation error naming the token and flag. In particular,
`normalized=true`, `lstrip=true`, or `rstrip=true` on an added token is not silently accepted. The
E5 fixture's `tokenizer_config.json` also contains an `AddedToken` object for `<mask>` with
`lstrip=true` and `normalized=true`; because that fixture is Unigram it is rejected under the
SentencePiece rule below, rather than treated as an admitted semantic.

When `tokenizer_config.json` contains these fields, they are cross-checked against the
`tokenizer.json` pipeline rather than used to override it: `do_lower_case=true`,
`strip_accents=true` or `null` under the same canonicalization rule, `tokenize_chinese_chars=true`,
`do_basic_tokenize=true`, and `padding_side`/`truncation_side` absent or `"right"`. Declared
`cls_token`, `sep_token`, `pad_token`, `unk_token`, and `mask_token` values, when present, must be
exactly `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, and `[MASK]` respectively. `never_split` must be absent,
`null`, or empty because only tokenizer.json added tokens are implemented. `clean_up_tokenization_spaces`
is a decode-only setting and is ignored after it is validated as a boolean. `tokenizer_class`, when
present, must be `BertTokenizer` or `BertTokenizerFast`. If a semantic field is
present in both files, the files must agree after canonicalization; omission does not override an
explicit value, and any contradiction is a typed error. The same rules apply to a legacy `vocab.txt`
layout using the canonical implicit WordPiece profile; a tokenizer_config declaration that differs
from that profile is rejected.

For the BPE layouts admitted by the inventory, the same closure applies. `BpeTokenizer` in
`crates/inference/src/tokenizer/bpe.rs` performs byte-level encoding and the fixed GPT-4-style
pre-tokenization implemented by `gpt4_regex_pretokenize`; it does not lowercase, strip accents,
split CJK specially, or apply a general normalizer. A BPE `tokenizer.json` therefore accepts only a
missing or `null` `normalizer`. Its `pre_tokenizer` must be absent, a bare `ByteLevel` with
`use_regex` absent or `true` and `add_prefix_space` absent or `false`, the exact fixed regex split
implemented by Lattice, or a `Sequence` containing only that regex split and the supported
`ByteLevel{use_regex:false}` tail. An arbitrary regex with the same JSON shape, `use_regex:false`
without the preceding supported split, `Metaspace`, `Whitespace`, `BertPreTokenizer`, or any other
declared pre-tokenizer is rejected; matching the outer type is not enough. BPE model flags that the
implementation does not honor (`dropout`, `continuing_subword_prefix`, `end_of_word_suffix`,
`fuse_unk`, `ignore_merges`, and `byte_fallback`) must be absent or their exact no-op value
(`null`/`false` as applicable), and the effective unknown-token choice and post-processing flags
must match the fields consumed by `BpeTokenizer`. Raw `vocab.json` plus `merges.txt` uses the
canonical fixed GPT-4-style profile because it has no declaration to override it.

For BPE `tokenizer_config.json`, `do_lower_case`, `strip_accents`, `tokenize_chinese_chars`, and
`add_prefix_space` must be absent or `false`; `padding_side`/`truncation_side` must be absent or
`"right"`; and `clean_up_tokenization_spaces` is decode-only and ignored after boolean validation.
Any declared normalizer, pre-tokenizer, or tokenizer-config field that could alter token IDs and is
not in this closed allowlist is rejected. The validated WordPiece or BPE semantic profile,
including all canonicalized normalization and pre-tokenization values, is included in D6 and is
reconstructed on the post-load pass.

### D5: Lattice owns the canonical digest; callers may add supplementary evidence

Lattice owns safe enumeration, normalized ordering, opened-handle reads, bounded streaming, and the
identity-bearing digest. It does not import a downstream registry protocol: the digest is the
evidence, while names of downstream tables, keys, and cache slots remain downstream. The caller may
also supply an attestor factory for supplementary evidence when `prepare_with_attestor` is used.
That method calls its factory exactly twice so pre-load and post-load passes never reuse mutable
hash state; `prepare` performs no caller-attestor callbacks.

Lattice always computes `CanonicalAttestationReport` itself. Its fixed algorithm identifier is
`AttestationAlgorithm::Sha256V1`, whose canonical domain is explicitly SHA-256-bound, and its
digest is the 32-byte SHA-256 result over this exact byte framing, with all integers big-endian:

```text
b"lattice.sealed-native-embedding.sha256.v1\0"
|| u64_be(file_count)
|| repeat in lexicographic normalized-path order:
     u64_be(path_byte_len) || path_bytes || u64_be(declared_len) || exactly declared_len content bytes
|| b"lattice.sealed-native-embedding.sha256.v1.end\0"
```

The domain string and terminal marker are fixed ASCII byte strings, not caller inputs. The file
count, path length, path bytes, declared length, and content are all part of the hash preimage;
content is fed as the canonical exact-fill 1 MiB chunk schedule described below. The terminal
marker is required even for an empty file set. `CanonicalAttestationReport` carries the
`AttestationAlgorithm` value and `[u8; 32]` digest as typed fields, so an algorithm change cannot
be confused with a digest produced by this version.

The conceptual trait is:

```rust,ignore
pub trait CheckpointAttestor: Send + 'static {
    fn begin(&mut self, file_count: u64) -> Result<()>;
    fn begin_file(&mut self, logical_path: &[u8], declared_len: u64) -> Result<()>;
    fn chunk(&mut self, bytes: &[u8]) -> Result<()>;
    fn end_file(&mut self) -> Result<()>;
    fn finish(self) -> Result<SupplementaryAttestationEvidence>;
}
```

When `prepare_with_attestor` is used, before any file callback Lattice supplies the exact file count
to the attestor. Files are emitted in strict lexicographic order by normalized logical relative-path
bytes. Each path is
emitted once, followed by its declared length and contiguous chunks covering exactly that many
bytes. The chunk schedule is canonical: every non-final chunk is exactly 1 MiB, the final chunk is
1..=1 MiB, and an empty file has no chunk calls. Lattice uses an exact-read loop rather than
exposing incidental short reads as chunk boundaries. No bytes from a later file are emitted before
`end_file` succeeds. Short reads, growth beyond the declared length, callback errors, count or
length disagreement, and byte-unequal second canonical digests fail publication.

The attestor receives no source or snapshot OS path, `File`, descriptor, mmap object, mutable
buffer, or model handle. `SupplementaryAttestationEvidence` is a Lattice-owned immutable,
bounded value containing the caller's payload and an explicit `AttestationAlgorithm::Sha256V1` plus
the canonical digest it claims to bind. External callers construct it only through the public
fallible `SupplementaryAttestationEvidence::try_new(algorithm, digest, payload)`, which records the
claimed algorithm and digest and rejects payloads outside 1..=4096 bytes; the fields remain private.
At publication Lattice requires the binding algorithm and digest to equal its own canonical report
on both passes; a malformed, missing, mismatched, or fixed/constant caller
result that does not bind the current digest is rejected. Supplementary bytes are recorded beside,
never instead of, the canonical report and cannot change or collide the identity material. They
count toward preparation and steady-state resource charges. The attestor remains trusted in-process
code for availability and confidentiality: it receives every checkpoint byte and may retain bytes,
allocate, block, or panic. Lattice bounds the bytes and chunk size it supplies, not callback-owned
memory or execution time, and supervisor leases remain held through every callback.

The canonical digest is the cryptographic content evidence. A caller may use supplementary evidence
for an external protocol, but a consumer must not treat that evidence as a replacement or use a
constant caller report to authorize identity reuse.

### D6: the effective descriptor exposes all non-file vector semantics

`EffectiveEmbeddingDescriptor` is immutable, bounded, and path/time/host-location independent. Its
borrowed typed accessors cover at least:

- a versioned Lattice provider-semantics revision;
- stable model variant and model family;
- native and active output dimensions, including an explicit absent/present truncation marker;
- selected tokenizer layout and tokenizer semantics revision;
- validated tokenizer normalization, pre-tokenization, special-token, padding, and truncation
  semantics, including the canonical WordPiece/BPE profile and cross-file agreement;
- validated BERT configuration semantics: `model_type`, `hidden_act`/`GeluTanhApproxV1`,
  `position_embedding_type`, decoder/cross-attention rejection state, and the closed ignored-field
  policy;
- catalog advisory token limit, realized tokenizer/model hard sequence cap, and truncation behavior;
- query and document instruction bytes with explicit absence markers;
- pooling strategy;
- output normalization strategy and constants;
- batch-execution protocol (`CanonicalPerItemV1` in prepared v1);
- complete validated tensor name/shape/source-dtype inventory digest and decode-to-f32 revision;
- weight, accumulator, and output dtypes relevant to outputs;
- adapter composition (`none` in v1);
- cache policy (`disabled` in v1);
- requested and realized execution backend; and
- the realized Lattice CPU numeric-kernel profile: target architecture, relevant build features,
  detected scalar/AVX2/FMA/AVX-512/NEON capability set, fixed matmul/pooling/normalization kernel
  families, versioned softmax/exp and GELU/tanh approximations, and versioned dispatch policy.
  Prepared v1 does not call Accelerate, cblas, AMX, system libm, or another OS/vendor numeric
  library; and
- a bounded `NumericBuildDescriptor`: exact `lattice-embed`/`lattice-inference` releases, canonical
  numeric-source manifest digest, sorted resolved numeric-dependency graph digest
  (package name/version/source identity/features), rustc release/commit and LLVM version, target
  triple/CPU, compile-time target features and Cargo features, optimization/LTO/codegen-unit/
  overflow-check/debug-assert/panic settings, and the complete classified codegen flags that can
  affect floating-point output.

The tensor-inventory digest is SHA-256 over the bytes
`lattice.embedding-tensor-inventory.v1\0`, followed by a big-endian `u64` tensor count. Lattice
sorts tensors by raw UTF-8 name bytes before hashing. Each tensor contributes a big-endian `u32`
name length and name, one source-dtype byte (`1 = F32`, `2 = F16`, `3 = BF16`), a big-endian `u32`
rank, and every dimension as a big-endian `u64`. Duplicate names, non-UTF-8 names, unsupported
dtypes, unrepresentable lengths/dimensions, and missing inventory fail before hashing. Iteration
order from a `HashMap` is never identity order.

It does not contain absolute paths, mtimes, inode numbers, load timestamps, process IDs, namespace,
or a downstream vector-space fingerprint. Lattice reports facts; the downstream owner decides how
to frame the canonical attestation material and descriptor into its own persisted identity.

The build descriptor is generated by a checked build-time manifest, not reconstructed from a crate
version at runtime. Its canonical digest framing is versioned and path/time independent. A build
with a missing source-kind identity, incomplete compiler identity, or unknown/unclassified codegen
flag marks prepared mode unavailable; it does not publish a partial descriptor. Dependency source
identity has one closed tagged encoding: registry packages use canonical registry identity plus the
published package checksum; git packages use canonical repository identity plus exact commit and
tree/content digest; first-party and other permitted path packages use a canonical package-relative
source-manifest content digest and never an absolute path. Path packages are not required to have a
Cargo.lock `source` or registry `checksum`; they are required to have the manifest digest. Any other
source kind is unsupported until amended. The manifest covers `Cargo.toml`, the resolved lock/
dependency input, toolchain/channel configuration, build-profile generation, and all vector-
affecting source paths. Runtime jobs also set and restore one recorded floating-point control policy
(round-to-nearest-even plus an explicit FTZ/DAZ choice) around native work; an unsupported or
uncontrollable floating-point environment fails before encoding. Ambient process FP state is never
silently inherited into an attested output.

Prepared v1 preserves the existing BERT input contract: `EmbeddingModel::max_input_tokens()` is an
advisory catalog value, while the tokenizer/config-derived realized sequence cap governs encoding.
The descriptor reports both values and their distinct meanings. It does not begin rejecting inputs
only because they exceed the advisory catalog value; the existing UTF-8 byte bound and realized
tokenizer/model truncation behavior remain authoritative.

The descriptor is constructed before load and independently reconstructed from the loaded model
and sealed snapshot after load. Exact mismatch fails publication. Every source change in a vector-
affecting surface must change the provider-semantics revision and its identity golden, even when an
individual descriptor field also changes. The implementation adds a checked, intentionally over-
inclusive path manifest covering the prepared facade, model configuration and role transforms,
BERT model, tokenizer, pooling/normalization, CPU forward, and weight-decode surfaces. CI fails when
a manifest-matched source or build input changes without both updates. The pull-request template
requires authors to classify any new vector-affecting path, dependency, compiler/codegen input, or
build-profile input and update the manifest. Field-mutation goldens prove that declared fields affect
identity; the manifest gate, not those goldens, enforces code-to-revision coupling.

### D7: prepared v1 disables every embedding-result cache

Prepared v1 neither wraps the service in `CachedEmbeddingService` nor imports or exports a Qwen
persistent cache. Any model-internal embedding-result cache is disabled. Presence of a legacy cache
file causes no read, validation, allocation, hit, or write.

Keying a future cache by the complete downstream vector-space identity is necessary but not enough
to authenticate cache contents. A future persistent-cache amendment must also define an owner-only
trusted cache root or cryptographic authenticity mechanism, bind dimensions and payload bytes,
retain exact collision-resistant request keys, reject non-finite vectors, and validate the complete
space identity before allocating entries.

Prepared v1 also disables tokenizer LRUs. The current entry-count cache can retain thousands of
owned maximum-length strings without a byte ceiling, so it is not admitted by the prepared
resource budget. A future tokenizer-cache amendment must use exact-input collision-resistant keys,
enforce a byte-weighted hard ceiling, include retained key/value bytes in the prepared object's
steady-state lease, and preserve the tokenizer semantics revision.

Wrapping or type-erasing a prepared service through `CachedEmbeddingService` or any other
decorator produces an identity-free compatibility service. A caller must not reuse the original
prepared report or descriptor to publish outputs from that wrapper.

### D8: admission is bounded before materialization and native work outlives waiters

`NativePreparationLimits` is required input rather than an ambient global. Construction validates
it before filesystem work. `NativeResourceBudget` bounds preparation concurrency, encode
concurrency, retained bytes, and transient work across every service produced by one preparer. It
uses independent retained and transient-work weighted pools. Their checked sum is the domain's
maximum simultaneous accounted bytes; neither pool may borrow from the other. A host that wants one
global resource ceiling must share that preparer/resource domain; opening multiple preparers
on distinct snapshot parents intentionally creates independent budgets and the host must account
for their sums. The same-parent domain lock prevents accidental double budgeting there. The
implementation uses checked arithmetic and enforces, before materializing the corresponding object:

- file-count and normalized-path byte limits;
- per-auxiliary-file and per-index byte limits;
- per-weight-file and aggregate snapshot byte limits;
- safetensors header bytes, tensor count, rank, and aggregate tensor-metadata limits before metadata
  materialization;
- bounded config/tokenizer parsing and bounded attestation chunks; specifically, the recursive
  `parse_value`/`parse_array`/`parse_object` parser in
  `crates/inference/src/tokenizer/common.rs` must enforce a fixed maximum nesting depth of 64
  before descending into any array or object, or use an explicitly iterative replacement. The
  checked-in tokenizer fixtures reach at most eight container levels, leaving substantial margin;
  over-depth input returns a typed preparation error rather than risking worker-stack exhaustion.
  The same bound applies to `config.json`, `tokenizer_config.json`, `special_tokens_map.json`, and
  safetensors headers wherever a nested-JSON parse exists;
- a conservative peak-resident-byte estimate for loaded weights, conversions, fused tensors,
  scratch, and output; and
- platform-representable file lengths, allocation sizes, and dimensions.

Before any filesystem call, preparation acquires one concurrency slot, a worst-case retained-pool
reservation computed from the caller's already-validated snapshot/model/retained maxima, and a
worst-case transient-pool reservation for preparation scratch. A request whose maximum charge
exceeds either pool fails immediately rather than queueing forever. Cancellation while queued starts
zero source or snapshot work. The supervisor, not the awaiting caller, owns the job guard and
permits before spawning blocking work.

After opening the complete bounded inventory, the preparer may reduce that reservation only from
handle metadata and bounded headers whose consequences it can enforce. Each copy reads exactly the
opened handle's censused `declared_len` into the snapshot, then performs one non-materialized byte
probe: short reads and growth both fail. It never copies toward a larger per-request ceiling after
releasing the corresponding global lease. False-small metadata therefore causes failure rather
than unaccounted global use.

On successful publication, the job transfers its retained-pool lease into the prepared inner
object. The retained charge explicitly covers conservative live model/mapping allocations, exact
snapshot bytes, tokenizer/vocabulary state, the descriptor, the fixed 32-byte canonical digest and,
when `prepare_with_attestor` is used, 1..=4096-byte supplementary attestation evidence, normalized
inventory/path bookkeeping,
live-lock/control state, and maximum retained internal-cache
bytes (zero in v1). Only the transient preparation scratch lease and preparation concurrency slot
release. The retained lease remains until the final prepared/job reference, model, and snapshot are
gone, or cleanup failure atomically transfers the complete `ResidueCharge` into the supervisor's
retained-pool residue debt.

Metadata is an early-refusal hint, not authority. Copy and attestation loops enforce the exact
declared length followed by one non-materialized growth probe and stop at the first disagreement.
No unbounded `read_to_string`, generic JSON tree, path list, or checkpoint buffer is permitted. The
snapshot is streamed with one fixed buffer and no second full checkpoint allocation.

Preparation and prepared encoding run blocking filesystem/native work outside Tokio workers.
Before an encode clones input or enters the blocking queue, it validates the borrowed request,
including the existing per-text and per-batch caps, computes checked aggregate input plus prompt,
conservative model scratch, and maximum output bytes, then acquires both an encode-concurrency slot
and a transient-work-pool lease. A single encode whose charge exceeds that pool fails immediately.
Retained leases never consume encode-work capacity, so a successfully published service cannot
strand itself behind its own model/snapshot lease. Queued cancellation performs no clone or native
call.

Once admitted, a dropped async waiter does not abort a running blocking load or encoding job and
does not destroy the model/snapshot while native code can still access them. The blocking job owns
its guard, lease, and prepared-inner `Arc` until native completion. Preparation/encode failure,
spawn failure, callback error, and panic unwinding each deregister exactly once after native work
and cleanup end.

Drain is deliberately two-phase. `NativeEmbeddingDrain::begin_drain` atomically, irreversibly, and
idempotently closes both preparation and encode admission before observing any count. It uses the
same supervisor state mutex as prepared publication: publication-first produces an already-published,
tracked object that the host must drop, while drain-first makes every admitted-but-unpublished
preparation clean without returning a service. The host then stops routing and drops every registry/
service `Arc` it owns.
`NativeEmbeddingDrain::wait_released` waits for every guard registered before spawn, every prepared
inner object, every model/snapshot cleanup attempt, and every releasable retained or transient lease
to reach zero. This prevents a new job from racing an idle observation and prevents a retained
prepared object from being reported as drained. Prepared objects retain the supervisor `Arc` even
if the preparer is dropped, and every host shutdown path retains a drain handle. Calling
`wait_released` before `begin_drain` returns a typed lifecycle error; retaining any prepared
reference keeps the wait pending. When all native access and cleanup attempts have ended, residue
debt is the sole exception to the zero-lease success condition: `wait_released` returns a typed
aggregate cleanup error carrying the still-charged residue count/bytes instead of hanging or
returning success. Process abort is outside this async drain guarantee.

### D9: publication is one atomic downstream capability

A downstream registry must publish the prepared service and identity material derived from its
borrowed `CanonicalAttestationReport` and `EffectiveEmbeddingDescriptor` as one atomic unit. The
identity material must include the report's `AttestationAlgorithm::Sha256V1` tag, its 32-byte
Lattice-computed digest, and a canonical encoding of the complete effective descriptor. Optional
`SupplementaryAttestationEvidence` is additional evidence only; it cannot replace, alter, or be
used instead of those two identity inputs. The downstream registry may wrap this material in its
own namespace and storage-specific framing, but it must not omit or substitute either Lattice
input. It must not expose a service first and fill in identity later, nor accept an independently
supplied identity alongside an arbitrary `EmbeddingService`. It must likewise reject outputs from a
wrapper or decorator that is not itself part of the prepared descriptor and attestation.

Lattice deliberately does not define Khive's table name, ANN key, lineage slot, cache key, daemon
configuration ID, or migration protocol. It provides the bound service, canonical digest, and
descriptor required for a consumer to define those safely. Existing vectors must be rebuilt from
source when that complete identity changes; relabeling vectors is not evidence of equivalence.

The descriptor reports the transformation for every `EmbeddingRole`, but it does not decide which
role a consumer assigns to a query or document call. A downstream identity must also bind that
role-routing policy. Calling `Generic` where another consumer calls `Query` is a downstream semantic
difference even though both use the same prepared Lattice service.

### D10: compatibility and rollout

This ADR is additive until a consumer elects to require the prepared capability:

- legacy constructors, automatic download, lazy loading, ambient environment resolution, and
  caches remain available through the legacy API;
- none of those legacy values may be promoted to a prepared value or attested retroactively;
- prepared and legacy services may coexist, but a persisted identity-governing registry must reject
  the legacy form; and
- the first implementation and release may claim only prepared BERT/CPU support. It must not claim
  that Qwen, Metal, remote providers, adapters, or persistent caches are attested.

Acceptance of this ADR authorizes the public API. It does not activate any downstream migration,
change a cache key, or prove a particular checkpoint. Consumers must exact-pin a release containing
the implementation and run their own identity goldens before serving persisted vectors from it.
The acceptance commit must atomically change this ADR and the index to `Accepted` and replace every
`Proposed amendment` backlink in ADR-003, ADR-005, and ADR-014 through ADR-017 with an accepted
`Amended by` relationship. It also changes ADR-070's `Proposed ADR-088` related-constraint wording
to the accepted status; partial status propagation is not acceptance.

## Required verification

The implementation PR must include deterministic tests for all of the following.

### Snapshot and inventory

- Once the source root and selected files are opened, replacing their paths cannot redirect copied
  bytes; the canonical digest identifies the exact private snapshot combination that is loaded.
- A one-hop internal relative file symlink is materialized; a root/ancestor symlink, link chain,
  directory link, escaping/absolute link, multiply linked file, and unsupported platform fail before
  copying target bytes.
- Non-regular files, path traversal, duplicate normalized paths, overlong UTF-8 paths, excess file
  count, and checked-arithmetic overflow fail closed.
- Hooks independently add, delete, rename, replace, truncate, grow, and same-size-mutate every
  recognized artifact between passes; no prepared value is returned.
- A higher-priority tokenizer candidate changes the selected layout and report; malformed or
  incomplete tokenizer layouts fail without fallback.
- Missing, malformed, oversized, or grow-after-metadata config/tokenizer files fail before an
  unbounded allocation.
- A tokenizer/config JSON regression fixture nests arrays and objects beyond the fixed depth-64
  bound while remaining below the byte limit; the `common.rs` value parser refuses it with a typed
  preparation error before descending into the over-depth container, with no stack overflow. The
  same test family covers `config.json`, `tokenizer_config.json`, `special_tokens_map.json`, and
  any safetensors header parsed as nested JSON.
- Snapshot bytes remain available while any service `Arc` or encode job exists and are cleaned only
  after model/mapping release.
- A live per-snapshot lock prevents scavenging; bounded-census overflow deletes nothing; an injected
  cleanup failure leaves an observable marked residue that a later exclusive stale census reclaims.
- Process-termination hooks cover every lock/state/directory creation, fsync, state-replace, seal,
  and cleanup boundary, including a lone lock control and state-replacement temporary. Each restart
  observes a held live transition or a grammar-valid reclaimable control/generation—never an unmarked
  data directory—and repeated crash recovery remains idempotent.
- Every row of the census disposition table has an exact test: held-live skips; unlocked-young
  retries without mutation; unlocked-old must reclaim; unknown/duplicate/inconsistent/overflow/
  unsupported states terminally fault without deletion or publication; and injected old-recovery
  failure faults even when the stale bytes exceed the new request's reservation.
- A second independently budgeted preparer on the same snapshot parent receives the typed busy error
  without blocking; the domain lock remains held by a surviving prepared object after its preparer
  is dropped and becomes reusable only after the complete domain lifetime ends.

### Attestation and descriptor

- Lattice computes identical `CanonicalAttestationReport` values on both passes from identical
  `begin(file_count)`, path, length, and canonical exact-fill 1 MiB chunk sequences on a stable
  fixture, including empty, boundary, and multi-chunk files. The test asserts the exact domain,
  big-endian framing, and terminal marker, including the zero-file case.
- The canonical report always has algorithm `Sha256V1` and a 32-byte digest. Evidence constructed
  through `SupplementaryAttestationEvidence::try_new` rejects payloads of 0 and 4097 bytes, while a
  value with a mismatched digest is rejected at publication. Evidence supplied when
  `prepare_with_attestor` is used also rejects a missing or mismatched canonical digest binding and
  a constant/fixed caller result that does not bind the current digest. Accepted evidence is
  immutable, borrowed from the service, and included in the retained charge. Multiple live services
  at the 4096-byte boundary consume the exact aggregate evidence charge and cannot exceed the
  retained pool.
- Path, timestamp, inode, source-root spelling, and load time do not change the report or descriptor.
- Two checkpoints with identical metadata but different tensor bytes produce different canonical
  digests and different downstream identities; mutating any artifact byte, path, declared length,
  model variant, dimension, tokenizer selection, query/document transform, pooling, normalization,
  dtype, tensor inventory, adapter marker, provider revision, backend, or numeric-kernel profile
  changes the downstream golden identity.
- The identity-material golden includes the `Sha256V1` algorithm tag, the canonical digest, and a
  canonical effective-descriptor encoding; replacing the digest with a caller report is rejected.
- Tensor metadata insertion permutations yield the same framed digest; every name/length/dtype/rank/
  dimension mutation changes it, and duplicate names or unsupported dtype tags fail.
- Every `NumericBuildDescriptor` field and runtime floating-point control field participates in the
  downstream identity golden. Builds that mutate toolchain/LLVM, dependency graph, target/features,
  profile/codegen flags, or FP policy either derive a distinct identity or fail prepared-mode
  construction; missing and unclassified inputs never reuse an identity.
- Architecture-specific MXCSR/FPCR sentinels prove that an instrumented prepared kernel observes the
  canonical FP policy and that the reused blocking worker observes its exact original state after
  success, typed failure, callback failure, and panic unwind; concurrent jobs cannot leak FP state
  into one another or a later legacy call.
- The report is borrowed from the service; no API can construct a prepared value from detached
  service and report parts.
- Pre-load and post-load descriptor mismatch or canonical-report mismatch fails publication;
  supplementary evidence supplied through `prepare_with_attestor` must also pass its digest-binding
  and equality checks.
- Every `BertConfig` zero/bound/divisibility/non-finite case, every required-tensor missing/duplicate/
  wrong-shape/layer-index case, tokenizer/config vocabulary disagreement, out-of-range ordinary or
  special token ID, token-type mismatch, and sequence-cap/position-row mismatch fails before
  publication. The sequence-cap file/key precedence and raw/capped descriptor fields have exact
  goldens, including malformed higher-priority values that do not fall through.
- `config.json` semantic goldens accept only `model_type="bert"`, `hidden_act="gelu"` with the
  `GeluTanhApproxV1` meaning, and `position_embedding_type="absolute"`; relative positions, ReLU,
  decoder mode, cross-attention, pruned heads, and unknown output-affecting fields fail with typed
  preparation errors. Known dropout, cache, classifier, label, and other output-irrelevant fields
  are covered as ignored metadata.
- The fixture matrix admits only WordPiece and BPE tokenizer variants. A `model.type="Unigram"`,
  `Metaspace` pre-tokenizer, or selected `tokenizer.model`, `sentencepiece.bpe.model`, or
  `spiece.model` fixture is rejected at preparation with a typed unsupported-tokenizer error;
  `MultilingualE5Small`, `MultilingualE5Base`, and `ParaphraseMultilingualMiniLmL12V2` remain deferred.
- WordPiece fixtures cover cased and accent-preserving normalizer declarations, altered CJK or
  punctuation behavior, non-right padding/truncation, non-empty `never_split`, changed special-token
  templates, and contradictions between `tokenizer.json` and `tokenizer_config.json`; each fails
  before publication. Accepted fixtures assert lowercasing, accent stripping, CJK/punctuation/
  whitespace/control handling, literal added-token handling, `[UNK]` fallback, and fixed special
  tokens. BPE fixtures cover null/non-null normalizers, every accepted pre-tokenizer shape, altered
  regexes and unsupported sequence children, unsupported model flags, and tokenizer-config
  contradictions; unsupported declarations fail and accepted normalization/pre-tokenization values
  enter the descriptor. Every `added_tokens` entry is checked against the table in D4: all five flags
  are present, only the listed values are accepted, and a negative fixture names the token and flag
  for an omitted or behavior-bearing value such as `normalized=true`, `lstrip=true`, or `rstrip=true`.

### Policy, resources, and cancellation

- Mutating every relevant environment variable after building `ResolvedNativeEmbeddingSpec` has no
  effect and causes zero environment reads inside preparation.
- Prepared v1 performs zero network calls and zero legacy cache reads/writes even when poisoned
  legacy files exist.
- Qwen, remote models, Metal, automatic backend selection, and fallback return typed unsupported
  errors before snapshot or model work.
- Every limit accepts its exact ceiling and rejects ceiling plus one before the associated large
  allocation or read; false-small metadata is caught by the streaming bound.
- A request larger than either applicable retained or transient-work pool fails without queueing;
  cancellation before admission starts zero backend work.
- Concurrent false-small sources never copy beyond their global leases; exact declared bytes plus
  the non-materialized growth probe fail rather than expanding a released reservation.
- Published services retain their steady-state model/snapshot lease. Destroying the preparer does
  not release it; only the final prepared/job reference and completed cleanup do.
- Retained and transient-work pools hit their exact independent ceilings. A service whose retained
  lease leaves zero retained headroom can still complete an admitted canonical one-item encode from
  the work pool; a preparation or encode that exceeds its own pool fails before queueing.
- Encode admission charges owned input, scratch, and output before cloning/spawning. Exact ceilings
  succeed, an over-budget single encode fails immediately, queued cancellation performs no clone,
  and a cancelled-call flood cannot exceed job/byte ceilings.
- Dropping the last waiter after blocking load or encode begins retains the weighted lease and
  model/snapshot ownership until native completion. `begin_drain` first closes both admission paths;
  after registries are dropped, `wait_released` waits for jobs, prepared objects, cleanup, and all
  leases. A concurrent start cannot race a false idle observation, and one deliberately retained
  prepared `Arc` keeps the wait pending.
- Injected pre-publication and post-publication state-fsync, unlink, and control-file cleanup failures
  transfer a complete data-plus-control `ResidueCharge` from the job reservation or prepared lease;
  preparation closes for the faulted domain, existing services can use only the independent encode
  pool until drain, and drain returns a typed aggregate cleanup error only after jobs/objects release,
  never success or an unbounded wait. After the host drops that domain, a fresh preparer's exclusive
  stale census reclaims the marked residue without cross-supervisor reconciliation.
- A deterministic race faults cleanup while another preparation is already admitted; the second job
  exercises both publication-gate interleavings. Publication-first is tracked as an already-published
  object; fault-first returns no service and retains every lease until its own cleanup or residue
  transfer completes. A closed result receiver after publication also runs tracked cleanup.
- A second barrier race exercises publication against `begin_drain`: publication-first keeps drain
  pending on the registered object, while drain-first returns no prepared service and waits through
  the admitted job's cleanup. Neither interleaving permits an untracked object or false idle result.
- Normal failure, callback error, cancellation, and panic unwinding publish nothing and clean or
  observably mark residue. Process abort remains an operational stale-snapshot case.
- Preparation uses only the explicit pinned snapshot parent, never `TMPDIR`; disk-full and partial-
  creation failures are cleaned or marked for bounded recovery.

### Existing behavior

- Legacy `NativeEmbeddingService` and `CachedEmbeddingService` public behavior remains unchanged.
- Prepared BERT output exactly matches a direct BERT oracle loaded from the same snapshot under the
  same `CpuPinned` profile. The new inference kernel-policy seam in `Auto` mode preserves legacy
  `NativeEmbeddingService` outputs and routing.
- For every public `EmbeddingService` entry point, the sealed model succeeds and a different model
  returns the typed unsupported-model error before prefixing, cloning, admission, or native work;
  `model_config` returns the frozen sealed configuration and `supports_model` is exact.
- A same-family checkpoint with `H` unequal to the sealed model's native dimension, or any BERT
  request carrying `output_dim`, fails before model load/publication and produces no identity.
- For each supported model/role, the same text has bit-identical output alone, at every position in
  a larger batch, beside different-length neighbors, after neighbor/order permutations, and across
  accepted batch sizes. This pins `CanonicalPerItemV1` and rules out a packed-kernel regression.
- Exact output, report, and descriptor goldens survive restart and source relocation.
- An implementation-owned closed fixture table enumerates exactly the four admitted BERT variants:
  `BgeSmallEnV15`, `BgeBaseEnV15`, `BgeLargeEnV15`, and `AllMiniLmL6V2`. Every row pins its
  production WordPiece tokenizer layout
  and pooling strategy and runs Generic, Query, and Passage for each F32/F16/BF16 source dtype that
  row accepts; an unsupported dtype is explicitly rejected and tested. Across the table, every
  accepted tokenizer layout is exercised, BGE CLS and MiniLM mean pooling are covered, and an input
  above the advisory catalog token count but within the realized tokenizer/model cap is retained.
  The `MultilingualE5Small` Unigram fixture is the required rejection fixture while
  `MultilingualE5Base` and `ParaphraseMultilingualMiniLmL12V2` are deferred with the Unigram family.
- Wrapping/type-erasing a prepared service cannot expose an attested capability or authorize reuse
  of the original report for wrapper outputs.

## Benchmark disposition

This ADR-only change is documentation and does not touch an executable benchmark target. The
implementation PR will touch `crates/embed/` and `crates/inference/`, so ADR-087 applies. Before
citing `make bench-compare`, that PR must enumerate all declared benchmark targets and prove whether
each changed function is reachable. A structurally unreachable preparation-only path may use the
ADR-087 proof instead of an unrelated A/B run. Any change reachable from model load or encode
benchmarks requires targeted measurement under the same sealed fixture and realized backend; the
full unrelated Criterion suite is not evidence for an unreachable diff.

## Alternatives considered

### Add `identity()` to `EmbeddingService`

Rejected. A trait method cannot prove that an arbitrary service uses the returned identity. A
compatibility default would bless unsealed implementations, while a required method would be a
large source break without closing the service/report pairing problem.

### Hash the model label, `ModelConfig`, or `ModelProvenance`

Rejected. These values omit loaded artifacts and effective semantics. `ModelProvenance` also
contains time-derived metadata and explicitly avoids reading weights.

### Hash the source directory and then load it in place

Rejected. Separate path opens can observe different generations, and same-user mutation can occur
between verification and mmap. A private opened-handle copy plus pre/post snapshot attestation is
the smallest coherent generation boundary.

### Trust existing per-file download checks

Rejected. They do not cover the full loader closure, tokenizer precedence, configuration defaults,
or one directory generation.

### Enable Qwen and cache support immediately

Rejected. Qwen adds sharded inventory, dynamic backend fallback, high peak memory, and a persistent
cache that is neither fully space-bound nor authenticated. Those are separable follow-up decisions;
including them would prevent the BERT prerequisite from landing safely.

### Let Lattice define the downstream fingerprint

Rejected as a downstream storage protocol. Lattice does define the canonical artifact digest and
its typed algorithm binding because those are the evidence of the bytes actually loaded. Downstream
systems still own the namespace, identity framing around the canonical digest plus descriptor, and
storage semantics; supplementary caller evidence preserves that boundary without becoming the
identity source.

### Let the caller own the canonical content digest

Rejected. A caller-controlled digest can be weak, constant, malformed, or computed over a different
file schedule. Lattice must compute the fixed SHA-256 digest over its own opened handles and exact
canonical schedule; caller output can only be optional, validated supplementary evidence.

## Consequences

### Positive

- A consumer can bind an embedding service to the exact artifact generation and effective vector
  semantics used by that service.
- Ambient path, environment, backend, tokenizer-precedence, and cache drift cannot silently reuse a
  prepared identity.
- Existing mmap authority remains singular; downstream code receives evidence rather than raw file
  capabilities.
- The first release has a small, reviewable trust surface: explicit local BERT, CPU-only, no cache.

### Negative

- Preparation copies the selected checkpoint once and reads the snapshot twice, adding startup I/O
  and temporary disk usage.
- Full-file collision-resistant attestation is linear in checkpoint bytes.
- Moving to a machine with a different realized numeric-kernel profile can produce a different
  downstream vector-space identity and require rebuilding vectors.
- Qwen, Metal, remote providers, adapters, and result caches remain unavailable to consumers that
  require attested preparation until follow-up amendments land.

### Risks

- Supplementary caller evidence may be deliberately weak or unavailable. Lattice never treats it as
  the content digest: its binding must match the canonical report, and identity-governing consumers
  must use the Lattice digest plus descriptor regardless of whether supplementary evidence is
  configured.
- Read-only filesystem permissions are not a hostile same-UID sandbox. The guarantee is generation
  coherence under the stated trust model, reinforced by opened handles, private path custody,
  pre/post attestation, and retained model resources.
- Provider semantics can drift if maintainers change vector-affecting code without bumping the
  semantics revision. The required checked path manifest, revision/identity golden gate, and
  new-path PR classification make that an explicit compatibility obligation.

## References

- [ADR-003: SafeTensors Weight Loading](ADR-003-safetensors-loading.md)
- [ADR-005: Tokenizer Architecture](ADR-005-tokenizer-architecture.md)
- [ADR-014: Embedding Service](ADR-014-embedding-service.md)
- [ADR-015: Sharded LRU Embedding Cache](ADR-015-embedding-cache.md)
- [ADR-016: Embedding Model Variants](ADR-016-model-variants.md)
- [ADR-017: NativeEmbeddingService Orchestration](ADR-017-native-embedding-service.md)
- [ADR-070: CPU embedding batched encode](ADR-070-cpu-embedding-batched-encode.md)
- [ADR-087: bench-compare gate calibration](ADR-087-bench-compare-gate-calibration-and-coverage.md)
