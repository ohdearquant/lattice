> file an issue for a lightweight vector-space identity later (content hash of weights+tokenizer+config)
>
> — the maintainer, [issue #2343](https://github.com/ohdearquant/lattice/issues/2343)

# ADR-104: Embedding Vector-Space Identity

**Status**: Proposed
**Date**: 2026-10-27
**Crates**: lattice-embed, lattice-arch, lattice-weights, lattice-serve
**Depends on**: ADR-003, ADR-005, ADR-014, ADR-015, ADR-016, ADR-017, ADR-103

## Context

Persisted embeddings need an identity that distinguishes the inputs and effective output policy
that produced them. A model name and dimension do not identify the consumed checkpoint,
tokenizer, configuration, pooling, normalization, truncation or role transformations.

`ModelProvenance` deliberately identifies a timestamped load event through lightweight metadata.
`EmbeddingKey` records model, revision, dimensions, metric, dtype and normalization. Their
existing APIs and serialization remain unchanged; neither value becomes a vector-space identity
by receiving a new label.

This proposal adds a declared identity across embedding producers and transports. It introduces
no sealed preparation, attestation or resource-budget lifecycle. Lower producer declarations,
effective-field schemas and successful-load reports remain OPEN as specified below. The proposal
does not claim an implementation or complete producer coverage.

## Decision

### D1: known equality describes a declared recipe and effective policy

Equal known IDs mean the same declared recipe and effective producer policy. They do not prove
semantic or numerical equivalence, vector quality, correct execution, or protection against
hostile or concurrent external mutation. Strict consumers refuse to mix unequal identities or
an Unavailable identity with any other result.

Unavailable is a separate availability state outside the known-ID namespace. Two Unavailable
results never establish a known match. A persisted `unknown` string, shared sentinel, model alias,
dimension, revision label or load timestamp cannot substitute for missing evidence.

Query and document transformations form one paired policy under one ID. Changing either
transformation changes that shared policy. Generic transformation and image/text compatibility
require an explicit family disposition; Generic cannot silently inherit a registry prefix.

### D2: Full is the default; Header requires explicit opt-in

| Mode                                | Declared inputs                                                                                                                                                                               | Limitation                                                                                                                                                                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full, DEFAULT, wire tag 1           | Exact consumed config and tokenizer-file bytes; consumed safetensors headers and whole-file sizes; complete consumed weight-file content digests; output revision and effective output policy | Detects content changes represented by the load-bound recipe. It does not attest execution or external mutation safety.                                                                      |
| Header, explicit opt-in, wire tag 0 | The same recipe and policy, except weight identity uses consumed safetensors headers and whole-file sizes without weight-data content digests                                                 | **Header-preserving weight-value edits are UNDETECTED. Equal Header IDs do not establish equal weight contents.** Public API and transport metadata descriptions must state this blind spot. |

Wire tag 0 does not select the default. Full and Header occupy distinct namespaces and never
compare equal, even if their digest bytes happen to coincide. There is no automatic cross-mode
compatibility or equality.

The stable forms are:

```text
lattice-embedding-space:v1:full:sha256:<64 lowercase hex digits>
lattice-embedding-space:v1:header:sha256:<64 lowercase hex digits>
```

Only known IDs have these forms. Parsing consumes exactly the supported namespace, version,
mode, algorithm and 64 lowercase hexadecimal digits, with no trailing bytes. Unsupported or
malformed values are not known IDs. Wire version, identity mode and Lattice output revision are
separate fields.

### D3: bind consumed inputs and returned vectors to the same load

The successful open/parse/load path captures the consumed-artifact report and retains it with
the loaded model. A later directory rescan, manifest opened for another load or free-form model
label cannot identify that model. A manifest may cache a recipe calculation only after its input
coverage and binding to that load are established.

Reports cover exact consumed config and tokenizer bytes and every consumed safetensors file.
The safetensors length prefix and complete consumed header span have explicit framing, including
tensor names, dtypes, shapes and offsets. Raw-header coverage deliberately permits conservative
invalidation from header reformatting. Full additionally covers complete consumed weight-file
contents. Missing config/tokenizer origins, unproven preloaded objects and opaque remote
producers yield Unavailable. A format outside the specified safetensors recipe remains an OPEN
producer gap; it cannot omit weight inputs and claim a known ID.

New identified results bind the vector batch and identity to one load/policy snapshot. A later
getter cannot label an older cached or concurrently produced vector. Native lazy initialization
binds the report and identity to the same successful `OnceLock` result and snapshots relevant
environment settings once. Load failures cannot produce a known identity for an unloaded model.

Nested cache wrappers forward both identity and effective model configuration. Disabled-cache,
partial-hit and all-hit paths preserve each returned vector's producer binding; full hits must
not attach an unrelated current identity merely because they bypass inner embedding. Persistent
caches need an identity-bound report/key/version or identified-path invalidation or bypass.
Unidentified cached vectors cannot acquire the current known ID by relabeling.

### D4: one tagged, length-framed v1 codec

The ID digest is SHA-256 of this single canonical stream, in order:

1. Domain bytes `LATTICE_EMBED_SPACE` followed by one NUL byte.
2. Wire version: u16 little-endian, value 1.
3. Mode: u8, value 0 for Header or 1 for Full; default selection is Full.
4. Framed output-revision UTF-8 string.
5. Framed producer-family UTF-8 string.
6. Artifact sequence.
7. Effective-policy sequence.

Strings and byte strings use a u32 little-endian byte length followed by the bytes. Sequence
counts use u32 little-endian. Reject lengths and counts that do not fit. Names are case-sensitive
UTF-8 bytes; Debug output, host-locale text, package version and incidental serializer order are
not canonical encodings.

Artifacts sort lexicographically by `(role, logical_identifier)` and reject duplicate pairs.
Role tags are config 0, tokenizer 1, safetensors 2, and safetensors index 3. Each artifact record contains, in order:

1. Role as u8.
2. Framed logical identifier.
3. Whole-file size as u64 little-endian.
4. A 32-byte SHA-256 recipe digest.

For config, tokenizer, and safetensors index records, the recipe digest covers the exact consumed
bytes. The index uses the stable logical identifier `model.safetensors.index.json`; its record
is required for a sharded Known ID, together with a Full record for every consumed shard. The
index must name every consumed tensor. No sharded Known ID was emitted before this addition,
so the new tag reverses no existing sharded ID. For safetensors, the recipe digest is over u64 little-endian consumed header length followed by the
exact consumed header bytes. Header has no weight-data digest; Full additionally appends a
32-byte digest over the complete consumed safetensors file to that recipe input. This preserves
raw-header coverage independently of parser object ordering. Logical identifiers describe stable
artifact roles rather than directory enumeration order or ambient paths.

Effective-policy entries sort lexicographically by unframed UTF-8 key bytes and reject duplicate
keys. Each entry encodes a framed key followed by one typed value:

| Value            | Tag | Payload                            |
| ---------------- | --- | ---------------------------------- |
| Boolean          | 0   | One byte, 0 or 1                   |
| Unsigned integer | 1   | u64 little-endian                  |
| Signed integer   | 2   | Two's-complement i64 little-endian |
| UTF-8 string     | 3   | Framed UTF-8 bytes                 |
| Opaque bytes     | 4   | Framed bytes                       |
| f32              | 5   | Little-endian IEEE bits            |
| f64              | 6   | Little-endian IEEE bits            |

Reject nonfinite policy values. Preserve finite signed-zero and other bit distinctions
conservatively unless the family resolver has already canonicalized them. Lists and compound
policies flatten into named indexed keys under an explicit family schema. Actual input text and
image content are not space-policy fields.

### D5: hash resolved output policy

The family schema must account for pooling, normalization, actual maximum length and truncation
side, final materialized or truncated dimension, Generic and paired Query/Passage transformations,
relevant special-token and input-program settings, and applicable vision, preprocessing,
precision and adapter selectors. Execution backend is result metadata, never an input to the ID;
a producer reports the actual backend per vector when known. Each family's inclusion and exclusion decisions remain
OPEN until its actual producer contract supplies a complete field table. This list and the
generic codec do not prove field completeness.

Hash effective values after defaults, overrides, caps and fallbacks resolve. Unsupported or
ignored request fields do not enter the recipe merely because they were requested. In
particular, BERT serving pooling may derive from a directory basename before a mean/cls override;
record the resolved pooling, rather than the locator. EmbeddingGemma2 configured output dimensions
must resolve to the actual materialized dimension. Per-call output policy changes require the ID
for that output, rather than a fixed service-wide ID.

Maintain an explicit Lattice output-revision token for output-affecting changes not already
distinguished by the recipe. Every such change needs a bump/no-bump disposition. Changing the
token changes known IDs; descriptor tests alone cannot ensure future changes update it.

### D6: separate canonical encoding from loading and family policy

Use one canonical descriptor/encoder in existing `lattice-arch`, below embed and serve. Keep
generic consumed-artifact reporting/digest support in `lattice-weights`, successful-load reports
with the loaders, and effective option resolution with each producer. Storage does not own
embedding policy, and no upward dependency is introduced.

The arch codec supplies canonical bytes and known-ID formatting/parsing. Embed and serve can
apply their existing SHA-256 support; this proposal assumes no new arch hashing dependency or
transitive hashing API. Weights can use its existing digest support for consumed-artifact reports.
Exact additive report/export declarations remain OPEN.

The generic embed trait exists without native support. Its unconditional surface must carry a
standard-library opaque string/byte value, without a new mandatory arch, core or inference
dependency. Native adapters use existing optional model/compute/core/arch/tokenizer/weights edges.
Serve already reaches inference and weights and can use the version-coupled facade route once an
additive export is established. Inference reaches arch; arch reaches weights/core/grammar.
Preserve existing feature boundaries and public constructors. If the helper/export cannot fit
these existing edges, resolve that architectural question before changing manifests; do not
duplicate canonical encoders or assume a new crate or dependency.

### D7: cover the entire producer and transport population

The population includes 11 registry variants: seven BERT variants, two Qwen3 embedding variants,
EmbeddingGemma2 and one remote variant. Native service coverage alone is insufficient. The
following 13 groups require explicit dispositions:

| Group | Surface                                                                  | Required behavior                                                                                                                                                                         |
| ----- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E1    | `crates/embed/src/service/mod.rs`                                        | Object-safe identity and identified-output exposure, with Unavailable default for external implementations. Preserve ordinary methods and public structs; bind metadata to actual output. |
| E2    | `service/native.rs`, `model.rs`                                          | Same successful lazy-load snapshot for report/identity; exhaustive registry/effective-policy mapping; no name/timestamp fallback.                                                         |
| E3    | `service/cached.rs`                                                      | Forward identity and effective model configuration through nesting; preserve binding for disabled, partial-hit and all-hit output.                                                        |
| E4    | `wasm.rs`                                                                | Identify supplied bytes and effective interpretation; pooling mutation changes the relevant identity. Do not invent pathname provenance.                                                  |
| E5    | `vision.rs`                                                              | Cover directory and preloaded construction, image/text policy, prompt, pooling, preprocessing and backend disposition. Unproven preloaded provenance is Unavailable.                      |
| E6    | `bin/embed.rs`                                                           | Expose identity for text/image structured output using resolved options while preserving ordinary output.                                                                                 |
| E7    | Pooled serving trait/handles in `crates/serve/src/serving_embeddings.rs` | Propagate the same identity through this abstraction independently of the embed trait. Loader format and dimensions are insufficient.                                                     |
| E8    | Text serving trait/handles in `crates/serve/src/serving_embeddings.rs`   | Propagate the same identity through the separate text abstraction and its handles.                                                                                                        |
| E9    | `serving_provider/providers/qwen/embedding.rs`                           | Cover directory/preloaded provenance, actual context/tokenizer cap and per-call pooling/fallback policy using the shared codec.                                                           |
| E10   | Pooled HTTP in `serve/embeddings.rs`, `bin/lattice/serve.rs`             | Resolve default `mean_visual` and supported overrides first; derive identity from the service/request policy rather than echoed labels. Exact additive transport shape remains OPEN.      |
| E11   | Text HTTP in `serve/embeddings.rs`, `bin/lattice_serve/main.rs`          | Cover distinct DTO, handler and construction arms. Unsupported dimensions stay rejected; ignored user/label metadata cannot forge identity.                                               |
| E12   | Direct family, forward and vision APIs                                   | OPEN declaration/configuration/report population. Account for all public embedding producers and any formats outside the safetensors recipe before claiming completion.                   |
| E13   | Remote and third-party producers                                         | Explicit Unavailable unless an authoritative recipe can be established. Aliases, dimensions and shared sentinels are insufficient.                                                        |

Paths abbreviated after E1 are relative to `crates/embed/src/`; serving paths after E8 are
relative to `crates/serve/src/`.

The following lower interfaces and coverage remain OPEN:

- Native BERT acquisition, `QwenModel::from_directory` and the EmbeddingGemma2
  `from_model_dir` boundary need successful-load reports of consumed config/tokenizer
  bytes, header spans, file sizes, selected mode and Full content digests, retained for the
  loaded object's lifetime.
- WASM `BertModel::from_bytes` needs a report from the supplied bytes and effective
  config/tokenizer interpretation, without requiring or inventing file origins.
- Vision and serving-Qwen preloaded constructors need additive provenance-aware paths or
  explicit Unavailable results for externally built components.
- BERT, Qwen, EmbeddingGemma2/Qwen35/vision and tokenizer state need complete public
  effective-setting reports and per-family field tables, including limits, truncation side,
  special tokens, preprocessing and applicable selectors. Upper DTO coverage does not prove
  these lower fields.
- Native Qwen persistent-cache identity validation and legacy-cache disposition need explicit
  coverage. Nested wrappers must forward effective model configuration as well as identity.
- Arch/weights declarations and the serving facade need an exact additive export path on
  existing feature edges. Available dependencies alone do not prove the needed API exists.

### D8: independent codec controls and successful-load controls

Implementation validation must distinguish these witnesses:

| Witness                                | Required control                                                                                                                                                                                                                                                                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Independent codec sensitivity          | Independently specified canonical inputs and expected IDs isolate every declared recipe component, including header digest, file size and Full content digest. Omitting each component fails its expected-ID assertion. This establishes codec coverage only.                                                                                            |
| Successful-load/report binding         | Valid input variants traverse the real successful load and report path. Independently check the reported recipe and returned default Full ID; exercise explicit Header separately. Isolate inputs where format invariants permit it. Document correlated size/offset changes and test their combined valid change without claiming impossible isolation. |
| Malformed rejection                    | Unsupported/malformed inputs are rejected separately. Unchanged parser/load rejection proves neither component sensitivity nor detection of a component-omission mutant.                                                                                                                                                                                 |
| Header blind spot and Full sensitivity | A tensor-data-only mutation preserving header/config/tokenizer/size keeps Header ID equal and changes default Full ID. Assert Full1 default, explicit Header0 opt-in and cross-mode inequality.                                                                                                                                                          |
| Digest-path control                    | Instrument identity digest requests separately from ordinary model reads. Full requests exact config/tokenizer and complete consumed weight-file content; Header requests config/tokenizer/header inputs and file sizes without weight-data digests. Equal IDs alone do not prove the selected digest path ran.                                          |
| Effective options and roles            | Perturb each specified family field; equivalent resolved defaults agree, including materialized dimensions. Changing either Query/Passage transformation changes their shared ID. Explicitly settle Generic and image/text compatibility.                                                                                                                |
| Lifecycle and caches                   | Exercise lazy, concurrent, failed and changed loads; nested wrappers; disabled, partial and all-hit caches; and persistent caches. No older vector receives a newer identity.                                                                                                                                                                            |
| Stable form and availability           | Independent golden encodings, process round trips, unsupported/malformed version/mode cases, output-revision changes and Unavailable-pair refusal.                                                                                                                                                                                                       |

Mutation controls also replace inputs with names, timestamps or unbound manifests; omit mode or
output revision; hash requested rather than effective options; drop role-policy/configuration
forwarding; and attach current identity to an older cached vector. Every component omission must
fail an independent expected-ID assertion. Pair codec sensitivity with successful-report binding;
helper-only controls do not establish producer coverage.

Validation must cover native-off/default, WASM, vision/image and applicable Metal/serving build
arms, all E1–E13 dispositions, compilation, tests, Clippy, rustdoc, formatting, repository parity
and the relevant bench disposition. Any strict comparison gate must enumerate newly allowed
comparisons and include must-deny controls for unequal and Unavailable identities. These are
required future witnesses, not results established by this proposal.

### D9: migration and completion

Build the shared report/codec and successful-load binding before native/configuration/wrapper/
cache propagation, then cover WASM, vision, direct producers, serving, HTTP, CLI and migration.
All stages belong to the same contract. Completion requires every embedding producer and
transport to expose its selected load-bound declared identity or explicit Unavailable, with
effective options and cache propagation accounted for. A helper or upper-only getter does not
complete the contract.

Existing vectors without authoritative stored identity stay unidentified. Do not backfill from
name, dimension or load-event hashes. Header-to-Full migration cannot upgrade old vectors by
relabeling. Re-embedding or retained authoritative input evidence is required. Preserve vector/ID
pairs during rollback. Unsupported schemas and modes remain unsupported.

### Alternatives Considered

| Alternative                                             | Benefit                                   | Reason not selected                                                                    |
| ------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------- |
| Model name, revision and dimensions                     | Small existing metadata                   | Does not bind consumed artifacts or effective output policy.                           |
| Reuse `ModelProvenance` or `EmbeddingKey` as the new ID | Avoids an additive identity value         | Changes established meanings and still omits authoritative consumed inputs.            |
| Header hashing by default                               | Avoids weight-data content hashing        | Misses header-preserving weight-value changes; available only through explicit opt-in. |
| Output-vector equality as identity                      | Compares observed samples                 | Samples cannot establish a stable declared input recipe or all possible outputs.       |
| Mandatory sealed preparation and attestation            | Stronger preparation and integrity claims | Introduces a separate lifecycle beyond this declared-identity contract.                |

## Consequences

### Positive

- Persisted vector/ID pairs support conservative, explicit mixing decisions across producers.
- Full binds declared identity to consumed weight, tokenizer and config content by default.
- One canonical codec and paired role policy avoid divergent producer encodings.
- Capability-aware additive surfaces preserve existing methods, constructors, provenance and keys.

### Negative

- Exact config/tokenizer/header bytes and output-revision changes can force re-embedding even
  when practical comparability is unchanged.
- Header opt-in trades away detection of header-preserving weight-value edits.
- Strict consumers refuse Unavailable results, including otherwise usable opaque producers.
- Full content hashing adds load cost. Measured for BGE-small-en-v1.5 on an x86-64 Linux host
  (Intel i9-13900HX): one SHA-256 over the 133,466,304-byte `model.safetensors` per load adds
  about 46 ms (median 126 ms to 172 ms for a load plus one short embed through the `embed` CLI,
  10 processes per arm in ABBA order, quiet host). It is paid once per model load, not per
  embedding, and grows with checkpoint size. Not yet measured: the Apple silicon macOS host,
  explicit Header, and other model families. No negligible-overhead claim is made.

### Risks

- Missing family fields or consumed-input reports can create false agreement; their OPEN
  coverage must be closed before known identity is exposed for those producers.
- File binding and external mutation assumptions remain explicit; a digest is not an
  attestation of execution or a new snapshot guarantee.
- Wrapper, per-call-policy and persistent-cache mistakes can mislabel old vectors unless
  vector/identity binding is enforced at return time.
- The output-revision token depends on an explicit disposition for future output changes.

## References

- [Issue #2343](https://github.com/ohdearquant/lattice/issues/2343)
- [ADR-003: SafeTensors Weight Loading](ADR-003-safetensors-loading.md)
- [ADR-005: Tokenizer Architecture](ADR-005-tokenizer-architecture.md)
- [ADR-014: Embedding Service](ADR-014-embedding-service.md)
- [ADR-015: Sharded LRU Embedding Cache](ADR-015-embedding-cache.md)
- [ADR-016: Embedding Model Variants](ADR-016-model-variants.md)
- [ADR-017: NativeEmbeddingService Orchestration](ADR-017-native-embedding-service.md)
- [ADR-103: Inference Crate Decomposition](ADR-103-inference-crate-decomposition.md)
