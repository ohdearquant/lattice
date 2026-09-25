# ADR-090: Shared decoder generation with model-owned execution sessions

**Status**: Accepted (2026-09-15)\
**Date**: 2026-09-15\
**Amended**: 2026-09-25, RNG ownership (see "Amendment, 2026-09-25" under D1's roles);
2026-09-25, requested chat options (see "Amendment, 2026-09-25" under D3);
2026-09-25, model prompt adapters (see "Amendment, 2026-09-25: model prompt adapters" under D3)\
**Crate**: lattice-inference

<!-- deno-fmt-ignore-start -->

## Context

This record serves an instruction from the maintainer, at the console, 2026-09-15T16:29:56Z:

> 先别qwen38吧，［…］好好的optimize/refactor一遍先把，之前让你们把forward 的那个pipeline给abstract出来的，一直没弄嘛，那个先弄好，确保一样的pipeline可以 serve qwen and gemma 4, 这样我才敢放心的去以后上更多的model。 然后 inference/model整理一下。， gemma4的东西单独放到一个folder里面，然后有些file也看着奇怪可以rename， reorg，然后有些file实在是长的离谱，看看怎么refactor

Rendered in English: hold off on Qwen3.8 for now; do a proper optimize/refactor pass first; the
forward-pipeline abstraction asked for earlier has still not been done, so do that first and make
sure the same pipeline can serve Qwen and Gemma 4, so that adding more models later is safe; then
tidy up `inference/model`, put the Gemma 4 material in its own folder, rename and reorganize the
files that look odd, and work out how to refactor the ones that are absurdly long.

［…］ marks one elided clause concerning internal work assignment. The complete instruction is held in
the maintainer's internal record.

<!-- deno-fmt-ignore-end -->

### Instruction-to-decision mapping

- D1 supplies one generation driver for Qwen and Gemma 4, with typed model execution beneath it.
- R14 groups the Gemma 4 modules into their own folder.
- R14 and R18 organize `inference/model` and define the bounded renames.
- D9 and R15–R17 address the exceptionally long files after the shared route is demonstrated.
- The Qwen3.8 deferral is an explicit scope boundary below.

Qwen3.5 and Gemma 4 E2B have separate numerical execution and generation APIs. Qwen's existing
`DecodePolicy` already coordinates generation policy across several entry profiles; Gemma owns a
different cache, layer schedule and per-layer inputs. The serving worker deliberately constructs and
retains non-Send Metal state on its own thread. Sharing the generation loop is useful only if those
properties survive the boundary. Renaming the Qwen engine or wrapping whole `generate` methods in a
trait would leave the duplicated control flow in place.

This decision introduces a **new shared generation-driver contract**. ADR-080 authorizes checked
helpers, including decode policy, while keeping numerical kernels separate; it does not authorize
this driver. ADR-009 prohibits sharing the numerical forward pass between BERT and Qwen3. This
decision preserves that prohibition and narrowly clarifies its scope through the companion
amendment. It does not supersede either accepted ADR in full.

The initial consumer set is Qwen3.5 CPU, Qwen3.5 Metal, and the implemented **Gemma 4 E2B text CPU
profile**. That does not mean all Gemma variants, all Qwen artifact formats on every backend, or
multimodal E2B support. ADR-082's accepted staged requirements remain authoritative for E2B.

The inspected source baseline is `292658628f49f16daa04673afe2649eb4f2a5e8d`. This is a design draft;
the signatures below are contract sketches, not compiled API. Runtime validation and public API
publication have not occurred as part of this decision record.

## Scope

No Qwen3.8 work is included in this record. The first live delivery is shared-driver text generation
for the named Qwen3.5 and Gemma 4 E2B CPU checkpoints in R03/R04; reorganization follows it.

## Decision

### D1. Share the driver; retain typed execution beneath it

Add one crate-private, object-safe `DecoderSession` boundary and one ordinary autoregressive driver.
A concrete session owns or safely borrows its immutable model and owns its mutable cache, scratch,
position, validated layer plan and backend resources. The driver never downcasts the session or
edits its cache cursor. No universal KV vector, numerical `Layer` trait, or tensor backend is added.
Currently supported public model APIs remain entry wrappers or explicitly named diagnostics; this
does not restore APIs retired by [ADR-086](ADR-086-retire-legacy-qwen-decode-api.md).

The driver owns output budget, sampler/RNG progression, grammar transitions, reasoning policy,
stop-string holdback, cancellation and observer ordering. Extend and relocate the existing
`DecodePolicy`; do not write a second independent policy engine. Model code owns logits, attention,
normalization, PLE, cache updates, prefill scheduling and speculative state restoration.

Illustrative interface, with all supporting types crate-private:

```rust,ignore
trait DecoderSession {
    fn capabilities(&self) -> &ExecutionCapabilities;
    fn prefill(&mut self, cancel: &dyn Cancellation) -> Result<StepStamp, InferenceError>;
    fn decode(&mut self, accepted: &AcceptedToken, cancel: &dyn Cancellation)
        -> Result<StepStamp, InferenceError>;
    fn select(&mut self, request: &SelectionRequest)
        -> Result<SelectionCandidate, InferenceError>;
    fn metadata(&mut self, prediction: PredictionId, final_token: u32,
                request: &MetadataRequest) -> Result<TokenMetadata, InferenceError>;
    fn finish(&mut self, disposition: FinishDisposition) -> Result<(), InferenceError>;
}
```

`StepStamp` identifies the evaluated input prefix and the current prediction, if one was produced.
The driver distinguishes four token roles. The distinction is required by the shipped static
reasoning-budget behavior documented in [ADR-076](ADR-076-adaptive-reasoning-priority.md), which
remains **Proposed**. At the inspected source ref, `GenerateConfig.reasoning_budget` and
`DecodePolicy::apply_override` can replace a sampled ID with `</think>`; the final ID then drives
grammar, scoring and subsequent consumption. Preserve the existing effective-budget and
disabled-path semantics, separate reasoning/answer budget accounting, and D3's unsupported-profile
refusals. This inherits implemented static behavior; it neither accepts ADR-076's proposed research
priorities nor adds its unbuilt adaptive entropy/confidence policy. D2 retains the existing
first-token exception and subsequent-step order.

The roles are:

1. **Candidate:** `SelectionCandidate` contains a sampled candidate ID and its `PredictionId`.
   `SelectionRequest` borrows sampling configuration, grammar mask and history, with randomness
   supplied under the legacy draw schedule. RNG ownership stays in the driver. Sampling alone
   authorizes neither publication nor consumption; preserve the sampling step and its exact RNG
   consumption even when a later reasoning-budget rule overrides its candidate.
2. **Policy-final token:** `DecodePolicy` resolves the candidate to the final ID, which may differ.
   Grammar and EOS decisions use that final ID. An invalid final ID cannot be replaced by its legal
   candidate merely to continue. Keep control ordering in the existing policy, not in adapters.
3. **Metadata identity:** when the entry profile requests logprobs or other token metadata, the
   private `metadata` operation scores the final ID against the same prediction using the legacy
   masking, temperature and scoring semantics. Preserve the prediction's pre-advance scoring view;
   advancing grammar must not remask that view for scoring. Its result identifies both prediction
   and final token. Candidate scores are not relabeled as final-token scores. Scoring performs no
   sampling, RNG advance, grammar transition, token publication or cache consumption. Call it at the
   existing policy's scoring point; refactor that checked operation into a bounded callback rather
   than introducing another transition engine. Unsupported metadata remains a preparation-time
   refusal under D3's entry profile, not a reason to widen capabilities or force dense readback
   universally.
4. **Consumption:** `AcceptedToken` binds the accepted final ID to the prediction that produced its
   candidate. It is the final token pending evaluation when generation continues. Decode must accept
   a policy-forced ID different from the sampled ID, reject stale prediction identity, and consume
   the accepted token at most once. A rejected or terminating token does not acquire a pending
   decode merely because it had a valid candidate.

CPU adapters may use common scalar selection/scoring helpers over their borrowed dense logits. Metal
may retain compact selection/readback for eligible profiles; when metadata is not requested, no
scoring operation or dense transfer is imposed. Do not materialize an owned vocabulary-sized vector
to cross this interface. Prediction eligibility survives candidate selection and final-token
metadata reads, and ends at consumption, cancellation, failure or finish. Another prefill/reset also
invalidates it. Data borrowed for an operation cannot escape a subsequent invalidating state
mutation; `PredictionId` identifies state, not a public borrow or a cache-position override.

#### Amendment, 2026-09-25: the session owns and draws the RNG

Role 1 above says "RNG ownership stays in the driver", and the driver paragraph earlier in this
section lists "sampler/RNG progression" among what the driver owns. Both are replaced by the rule
below. The rest of role 1 is unchanged: `select` still returns a sampled candidate, sampling still
authorizes neither publication nor consumption, and the sampling step and its exact RNG consumption
are still preserved when a reasoning-budget rule later overrides the candidate.

**Rule.** The concrete session owns the RNG state and performs every draw inside `select`. The
state is seeded once, when the session is constructed, through the same seed-to-state transform
the pre-migration entry uses. Draws follow that entry's legacy schedule for the selection mode in
use, including the modes that draw nothing: a degenerate temperature (non-finite, not positive, or
so small that `1/t` overflows) selects the argmax and consumes no draw. `SelectionRequest` carries
no RNG handle and no pre-drawn values, and the driver never draws.

**Why.** The number of draws per step is a property of the selection mode, and on Metal the mode is
fixed per request by the readback route: a dense row, a compact candidate set, or a fused argmax.
Randomness supplied by the driver would have to predict each session's per-mode schedule, which
moves model-specific sampling knowledge into the shared layer. The CPU sessions delivered under R03
and R04 already keep the state in the session (`decoder.rs`, the `SelectionRequest` documentation;
`decoder/qwen_cpu.rs`, the module's RNG-state note). Keeping the driver-owned wording would mean
reopening those sessions to move the state with no change in output. Their goldens are greedy, so
they draw nothing and cannot detect a change in draw schedule; the seeded-reproducibility
obligation below is what a sampled golden has to check.

**What the driver still owns.** Output budget, grammar transitions, reasoning policy, stop-string
holdback, cancellation and observer ordering, as stated above. Seeded reproducibility is a
per-session obligation: for a given seed and configuration, a session must produce the token
stream the pre-migration entry produced.

[ADR-092](ADR-092-model-owned-decoder-execution.md) states "The driver owns policy and RNG
progression" in its execution-trait section. This amendment governs that sentence too.

Published token IDs, stop-held text bytes, the accepted final token pending consumption and the
evaluated prefix remain distinct. Neither sampling nor recording an output token implies that its
KV/GDN state has already been evaluated. A cancelled or failed prediction cannot remain selectable
or scoreable. Preserve the current initial-token path separately as specified in D2.

The three concrete sketches are `QwenCpuSession<'model>` (typed Qwen cache and scratch),
`GemmaCpuSession<'model>` (typed Gemma donor/sliding/full cache and PLE scratch), and
`QwenMetalSession` (worker-local engine/session handles and existing GPU allocations). A
loader-local family match may select one; HTTP handlers and policy code must not repeat that match.
No `Send` or `Sync` bound is imposed on the execution trait. A Send factory moves into the worker
and constructs the session there. Compile-time controls must reject attempts to send the Metal
session across threads; no unsafe auto-trait implementation is permitted.

### D2. Define lifecycle and failure before sharing control flow

The lifecycle is prepared input → concrete session → prefill → candidate selection → policy
finalization and requested metadata → decode the accepted final token if continuing → repeat →
finish. Model preparation resolves the execution profile and checks inputs before state mutation.
The session retains the validated prepared input, so prefill can consume a complete span; the driver
does not require arbitrary chunking or single-token prefill. Current Qwen batched prefill and its
fallback-before-mutation rule stay intact; current Gemma text prefill may remain sequential.

There is one current prediction per evaluated prefix. Selection consumes no token. Subsequent decode
consumes exactly the policy-accepted final token pending evaluation, which need not equal the
candidate. Preserve `DecodePolicy::transition` ordering: reasoning override → grammar advance on the
final ID → emitted-token bookkeeping → EOS/stop-token check → internal output push → final-ID
logprob recording → reasoning-end capture → text/stop processing and observer emission → answer
budget. Grammar rejection or EOS terminates before push, logprob recording or next-token
consumption, as the existing entry requires. Requested metadata belongs to that final ID and current
prediction, without another RNG draw. A metadata failure follows the typed failure and exactly-one
terminal-result rules; it cannot substitute candidate metadata or leak a partially assembled result.

The prefill-derived first token retains its separate initialization, first-token metadata and
initial-stop behavior; do not route it through a later-token override/grammar/EOS sequence that the
old entry did not apply. Characterize this explicitly rather than inferring it from subsequent
steps. The driver checks cancellation, termination and observer ordering at the same logical points
as the compatibility entry. It must not add a final forward after the last requested output merely
to unify implementations. Gemma's fixed-count `generate_greedy` diagnostic, which does evaluate its
last sampled token, remains separate.

Every execution failure returns a typed error plus an internally known reuse disposition. A failure
before mutation preserves the prior valid session; a partial execution poisons the request session
unless the adapter proves complete rollback. Poisoned state is destroyed or reset through its typed
owner, never made reusable by changing a common sequence-length integer. Cancellation has explicit
observation points before prefill, between permitted chunks/steps and before publication. Queued
cancellation, in-flight cancellation and shutdown preserve the worker's existing exactly-one
terminal result. A failed `finish` invalidates reuse and must not emit a second terminal event.

Prefix-cache and speculative execution are typed extensions, not optional no-op methods on every
model. Prefix reuse binds model identity, weights/adapter revision, tokenizer/prompt semantics and
cache topology; the concrete owner validates append/reuse/invalidation. Speculation returns a
verified committed span and owns its typed snapshot/rollback operation. The common policy applies
each verified token once and never publishes a draft token before verification.

If budget, stop policy or cancellation ends inside a verified span, the extension must restore the
typed state to the retained prefix before advertising reuse, or invalidate the session. Unpublished
verified suffixes cannot remain silently cached as part of a shorter conversation. Controls must
exercise a stop inside the span, not only full acceptance or total rejection; published-token,
evaluated-prefix and pending-token accounting must remain distinct.

### D3. Preserve entry profiles and model admission

Capabilities are negotiated from the concrete model, backend and entry profile before execution. One
permissive family-level boolean must not widen a narrower public entry point. Preserve
preparation-error precedence as characterized in the first implementation issue, including
prefix-cache capability checks, empty/context inputs and zero-output-budget handling.

| Entry/profile                                             | Behavior to preserve or explicitly add                                                                                                                                                         |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Canonical Qwen CPU generation/streaming                   | Existing full-logit policy, supported grammar/logprobs/stops/reasoning, seed progression and event order. Do not substitute the narrower standalone profile.                                   |
| Standalone CPU F16, Q8 and NEON wrappers                  | Retain their existing unsupported-control refusals and preflight order independently of canonical CPU capabilities.                                                                            |
| Ordinary Qwen Metal direct/streaming                      | Preserve each current entry profile, compact selection where eligible, batched prefill, typed failures and feature-disabled stubs.                                                             |
| Qwen prefix cache                                         | Preserve fresh/reuse/invalidation, cancellation, early capability refusals and zero-budget semantics.                                                                                          |
| Qwen speculative routes                                   | Preserve verified-token policy and typed restoration; apply the explicit verifier disposition in D6.                                                                                           |
| Gemma E2B text CPU serving                                | Add EOS-aware serving through the shared driver with Gemma tokenizer/template/control tokens; retain the fixed-count diagnostic API unchanged. Unsupported controls fail explicitly.           |
| Explicit unavailable backend or unsupported artifact role | Reject; an explicit Metal request cannot silently become CPU. Auto selection records its actual backend in the private diagnostics defined below and honors the same supported-profile checks. |

Keep currently supported import paths for `GenerateConfig`, `GenerateOutput` and other relocated
public types, subject to ADR-086's retained removal boundary in D4. `GenerateConfig` is already
non-exhaustive at the inspected ref; `GenerateOutput` is not. Preserve default/constructor behavior
and downstream output struct literals. Changing fields or defaults is a compatibility change, not a
move. Missing HTTP options stay distinguishable from explicit values until the model's prompt
adapter applies defaults; do not apply Qwen thinking defaults to Gemma. This milestone adds no
fields to `GenerateOutput` and does not make it non-exhaustive. Backend, route and D6 exclusion
evidence belongs to crate-private diagnostic records and opt-in measurement markers; it does not
change the public output literal or HTTP response schema. The execution-profile decision must remain
inspectable there, including Auto's actual backend. R02 and R13 compile a downstream-style literal
through the old public import path. If a new public field later becomes necessary, stop that change
for a separately reviewed API/semver decision before implementation.

Family detection and artifact format detection are separate decisions. Tokenization must not hide an
over-limit prompt by truncating it before admission.

Gemma admission initially accepts only the supported E2B text configuration and target-decoder role.
A bounded prerequisite represents attention mode and artifact role explicitly, preserves documented
absent/null behavior for the pinned E2B fixture, and rejects unsupported non-null modes, zero-PLE
profiles and drafter roles before weight loading. Unknown semantic fields must not silently turn a
different attention profile into E2B. This prerequisite adds validation, not new model math.

At the inspected ref, the shipped E2B fixture contains `use_bidirectional_attention: null`, but the
raw config struct does not represent that field and does not deny unknown fields. The positive-PLE
check cannot serve as an independent attention-mode check just because the cited upstream variants
with a non-null mode also have zero PLE. R04a's mode-negative fixture therefore keeps valid E2B
geometry and positive PLE while changing only the mode. This is a source-derived admission gap; this
draft does not claim a real incompatible checkpoint was loaded or executed.

Within the typed Gemma plan, PLE dimensions/presence, own-versus-donor KV storage, head dimensions,
layer attention kind and mask policy are validated from supported configuration. Do not infer them
from the family label or give every layer independent writable KV. Future non-causal image spans
would require a layer- and phase-specific mask and complete-span input; the present milestone
rejects them. Upstream cross-family variation motivates this extensible boundary but does not
establish support for additional Lattice variants.

#### Amendment, 2026-09-25: requested chat options are a hidden, unstable type

The missing-versus-explicit distinction for HTTP options is carried by
`serve::contract::RequestedChatOptions`, a `pub` `#[doc(hidden)]` type produced by
`normalize_requested_options`. It holds each option as the request sent it and `None` for an omitted
one; request validation still makes every refusal that cannot depend on a default. Qwen's defaults
(the server's sampling defaults, the thinking switch and the `<|im_end|>` stop token) are applied in
one Qwen defaults step, which the existing normalization functions and both handlers' `GenerateConfig`
construction route through, so their observable behavior is unchanged. A hidden, unstable type was
chosen over a stable public one because the worker-local factory (R07) will move where defaults are
applied. `#[doc(hidden)]` is a convention, not a semver guarantee, and this type is not a
semver-covered surface. R07 owes a disposition: promote it to a stable type through the separate
API/semver review above, or fold it into the factory.

#### Amendment, 2026-09-25: model prompt adapters

Each model family's chat conventions live in one crate-private prompt adapter: rendering normalized
messages to the prompt, the family's stop token ids, and the defaults step over
`RequestedChatOptions`. The Qwen adapter wraps the Qwen defaults step above unchanged. The Gemma E2B
text adapter renders exactly what the checkpoint's `chat_template.jinja` renders for string-content
system, user and assistant turns, reads its stop ids and BOS spelling from the checkpoint, and applies
no thinking default: the template's opt-in `enable_thinking` mode has no request switch, and the Gemma
CPU session cannot enforce a reasoning budget, so a positive `reasoning_budget`, logprobs, `stop`
strings, images and typed content parts are refused with the contract's existing codes. The Gemma
preparation entry (`serve::prepare::prepare_gemma_chat_request`), its output type and the adapter
type are `pub` and `#[doc(hidden)]` only because the measurement example calls them; they carry the
same no-semver-guarantee status and the same R07 disposition as `RequestedChatOptions`. Routing Gemma
through the serving binaries remains R08/R09.

### D4. Reconcile existing decisions without weakening their tests

| Decision          | Remains binding                                                                                                                                 | Narrow delta / inherited acceptance                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ADR-009, Accepted | BERT and Qwen3 configurations, weights, numerical forward and pooling remain architecture-specific; no strategy-based shared numerical forward. | Clarify that this does not prohibit the decoder-only control loop in this ADR. Keep ADR-009 Accepted and add a reciprocal link; narrow its broad no-sharing consequence to numerical forward.                                                                                                                                                                                                                                                                                              |
| ADR-080, Accepted | Existing checked helpers and decode-policy invariants; separate CPU/SIMD/Metal/WGPU numerical kernels.                                          | Reuse its policy machinery. This ADR newly decides the driver and lifetime boundary; ADR-080 is not precedent for that boundary. Add a companion link, no status flip.                                                                                                                                                                                                                                                                                                                     |
| ADR-082, Accepted | Its entire E2B ladder, including CPU-before-Metal and goldens before dependent math.                                                            | Inherit Stage 4: full 35-layer local/global schedule, window-512 sliding plus full cache, 20 shared-KV layers and PLE; per-layer HF trace with both shared-KV families separately compared, then logits/first-three-token parity and execution past the sliding boundary. **Wrong-donor mutation must fail the per-layer trace.** Preserve Stage 2 shape/dtype and Stage 3 wrong-normalization negatives. No duplicate weaker gate and no claim that text-only passes vision/audio stages. |
| ADR-074, Proposed | Historical MTP evidence is a lead, not accepted current performance authority.                                                                  | D6 disposes of the batch route; keep this ADR Proposed and link the later validation/disposition record. Do not promote its historical numbers to current-family claims.                                                                                                                                                                                                                                                                                                                   |
| ADR-087, Accepted | Reachability, calibrated target/configuration distinctions, evidence limits and its permitted pure-move proof category.                         | Apply its pure intra-crate move proof and D8's structural-first disposition to eligible diffs; pair changes to timed behavior. All proof conditions, current operating guidance and per-diff target inventory remain binding; no threshold changes.                                                                                                                                                                                                                                        |

R04 must name the existing Stage-4 test/fixture identities and retained negative controls in its
acceptance manifest. If source tests have not implemented part of that accepted contract, record and
close that gap before claiming R04 complete; do not call a prescribed gate an executed one. Preserve
the distinction between per-layer numerical trace, final logits and task-level tokens. A matching
first token cannot substitute for the wrong-donor control.

The remaining generation-contract relationships are explicit below. A Proposed ADR can document
shipped behavior without its whole decision being accepted; preserve the source-verified behavior
and the document's actual status separately. None of these seven decisions is superseded here.

| Decision                                                                | Relationship and retained scope                                                                                                                                                                                                                                                                               |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ADR-086](ADR-086-retire-legacy-qwen-decode-api.md), Accepted           | Complements its canonical Qwen3.5 API choice; extends the maintained implementation beneath those entry points, with the retirement boundary below unchanged.                                                                                                                                                 |
| [ADR-046](ADR-046-structured-output.md), Accepted                       | Inherits optional `GenerateConfig.grammar`, pre-sampling masking and grammar-state ownership. Preserve final-token advancement and entry-specific capability/refusal behavior; sharing policy does not make every CPU, Metal or multimodal entry grammar-capable.                                             |
| [ADR-049](ADR-049-vision-encoder.md), Accepted with recorded amendments | Complements the common config contract of `generate_multimodal`. Preserve current multimodal signatures, input checks and grammar/logprob/reasoning-budget preflight refusals. The bounded text-driver milestone neither completes its vision stages nor establishes multimodal Qwen/Gemma parity.            |
| [ADR-055](ADR-055-online-drift-detection.md), Accepted                  | Preserves the planned optional, default-`None` drift callback boundary and application-owned transport wiring. At this ref the inference `DriftSampler` bridge and config field remain unimplemented; this migration does not invent that field, mark the integration complete or add a transport dependency. |
| [ADR-063](ADR-063-serving-architecture.md), Proposed                    | Complements preparation: retain the shipped server-profile normalization and budget clamps before config construction/driver entry, plus the worker's prompt-aware context check. Its broader serving architecture is not ratified by this extraction.                                                        |
| [ADR-068](ADR-068-grammar-wire-contract.md), Proposed                   | Complements the opt-in grammar field with a proposed wire contract. Preserve the currently shipped structured-request admission and refusal paths; this extraction adds no wire fields and does not treat all proposed grammar channels as already shipped or accepted.                                       |
| [ADR-076](ADR-076-adaptive-reasoning-priority.md), Proposed             | Inherits the shipped static forcing behavior described beside D1/D2, including candidate-to-final override and grammar interaction. Adaptive policies and their experimental graduation remain separate; the document stays Proposed.                                                                         |

ADR-086 rejected adapting the **retired generic Qwen free function** to Qwen3.5 because its model,
configuration, cache and output types were incompatible: keeping its symbol would silently change
semantics or retain duplicate public contracts. D1 instead extracts control beneath the
**supported** `Qwen35Model::generate` / `generate_streaming` entries, preserving canonical
`model::GenerateConfig` and `model::qwen35_config::GenerateOutput`, current defaults, output shapes
and per-entry behavior. It introduces no compatibility adapter for the removed module, function,
types or dead attention benchmark, and leaves `QwenModel::encode` independent. Keep the retired
surface absent and the canonical paths usable during R02/R03/R13; an incompatible conversion is a
separate API decision, not permission granted by the word "wrapper." The candidate/final identity
contract in D1/D2 addresses an internal policy boundary, rather than undoing that public retirement.
The current
[canonical presence guard](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/tests/legacy_generate_removed.rs#L76)
checks defaults and type availability but not these entry points, so R01 must add typed compile-time
references to both `Qwen35Model::generate` and `Qwen35Model::generate_streaming` with their
preserved signatures and require a renamed entry to fail compilation.

The source pin, rather than a historical implementation-status paragraph, determines the existing
behavior these relationships preserve. In particular, `serve::contract` now performs the daemon's
budget normalization before `build_cfg`, and `lattice_serve` admits a structured request and assigns
its compiled grammar after `build_cfg` initializes `grammar: None`. The older blanket statement that
serving cannot reach grammar must not be carried into this plan. Conversely, the absence of
`DriftSampler` in the pinned inference source is consistent with ADR-055's unimplemented-bridge
note, not evidence that its accepted design was retired. No reciprocal status changes are needed for
these seven inherited or complementary relationships; the earlier four companion amendments and
index changes remain part of this proposal.

### D5. Resolve existing Metal findings before moving the affected paths

Before R06 or R17, dispose of #1583 on the stable Qwen Metal path: measure the actual per-head norm
distribution, add a CPU/Metal differential fixture sensitive to epsilon, then land any agreed
alignment separately with a reverted-fix negative control. The inspected CPU normalization adds
epsilon and the eight inspected GDN Q/K shader scales omit it. Real output impact remains
unmeasured; Metal-against-Metal poisoned-input parity cannot establish cross-backend formula parity.

Before R06, R07 or R17, dispose of #1584 separately: measure long-lived Qwen decode/worker RSS with
and without correctly scoped autorelease pools, including cleanup/lifetime correctness, then either
land the justified fix and regression control or document an evidence-backed no-change closure. The
source asymmetry with ERNIE is verified, but a leak is not. Do not add pools solely to make source
text look alike. A pool must not drain while borrowed autoreleased objects still escape it.

CPU-only design, fixtures and adapters may proceed independently. These dispositions are explicit
dependencies of the affected Metal moves, not claims that fixes already exist. Re-pin the migration
base after their resolution and collect its measurements from that base. Do not combine a numerical
or lifetime fix with a relocation; otherwise an old/new comparison cannot isolate the move.

### D6. MTP has a bounded compatibility exclusion, not a universal driver capability

Keep ordinary generation as the default. The default MTP verifier and self-spec execution stay
distinct mechanisms; their existing opt-in and greedy restrictions are preserved, with shared policy
applied only to verified committed tokens. R12 does not expand draft depth, temperature support, or
MTP to Gemma.

**Do not migrate the batch-GEMM verifier into the shared execution contract in R12.** Retain it as a
named legacy experimental route pending a separate ADR-074 E1/E2 validation/disposition task. A
request selecting it stays on the explicitly identified legacy route; it must not silently fall back
to ordinary or sequential verification and count as shared-driver coverage. R13's milestone excludes
that route and reports the exclusion. This is a deliberate compatibility exception for review, not
completion of its migration. No new public cross-model batch-verifier method is added.

At the inspected ref, `var_os("LATTICE_MTP_BATCH").is_some()` selects that route: unset versus set
is the existing rule, including empty or `0`. Preserve that rule on the retained legacy path. Source
contains a cursor-rejection test for the batch method, so an absolute “never tested” claim is false.
The method still panics on MoE after earlier state work; this is not a safe unsupported-input
contract. The separate disposition must address pre-mutation MoE refusal and restoration on errors
before any new shared-runtime admission of this path. R12 must not convert that panic to a generic
error and rely on the current fallback without proving what state was committed.

The disposition task freezes checkpoint/weight-basis evidence and compares ordinary generation,
sequential MTP and batch MTP on declared workloads, with positive execution markers, acceptance and
rollback checks, actual token counts and paired timing. The existing `mtp_decode` target is a
starting point, not proof: it can skip unavailable model/device paths and must be externally checked
for the declared groups and actual exercised route. Cursor-rejection coverage does not validate
successful batch numerical execution. Its output must decide promotion or retirement through a
separate reviewed compatibility change. Until that happens the exclusion remains visible and no “all
speculative routes unified” claim is allowed.

ADR-074's old 175.7-to-63.0 tok/s comparison concerns its recorded Qwen 0.8B Q4 experiment, sourced
there to an issue comment rather than a committed benchmark artifact. That ADR itself warns of
possible weight-basis and verifier confounds. It motivates remeasurement; it does not establish
current throughput or justify a family-wide profitability verdict. This draft sets no throughput
target from that historical comparison.

### D7. Numerical and structural budgets

For unchanged algorithms, **no new numerical tolerance is granted**. Inherit each existing
per-operation/model gate's reference, precision, finite-value rule and declared tolerance. Compare
base/head on the same backend, dtype, artifact, prompt and seeded sampler. Preserve token/event
sequences where the legacy entry contract defines them. Do not require unqualified CPU/Metal bitwise
equality or relax a failed existing bound after observing the optimized result.

The interface permits **zero additional full-vocabulary host readbacks on paths that currently use
compact selection**, and **zero additional Rust allocation/reallocation calls or allocated bytes
attributable to the per-token interface** after session setup. These are proposed incremental gates,
not measurements of present code. The scope of the latter is Rust's instrumented allocator; it is
not a claim to count Objective-C, driver or GPU allocations.

R01 must extend the existing `inference_perf` bench's `CountingAlloc`, `AllocationSnapshot` and
allocation-report machinery to real ordinary-generation consumers before their migration. At the
pinned ref, `bench_q8_neon_forward_allocations` already measures and rejects nonzero warm-forward
allocation totals under `bench-internals`; it does not yet measure the proposed shared driver, Gemma
session or Metal serving loop. Reuse that mechanism rather than treating it as absent or claiming
its forward-only result certifies new callers. Keep the measurement support identical on the base
and head of each migration. This is instrument preparation, not authorization to run the bench
during this documentation round.

Count allocation calls, reallocation calls and requested bytes separately around the same warm
decode/select/policy region; report completed tokens, route, initial cache capacity and sampling
profile. Exclude setup, formatting and fixture allocation from that region. Isolate the process and
worker so unrelated concurrent Rust allocations cannot contaminate global counters; bracket actual
worker execution, not merely enqueueing. Reject missing groups, zero tokens and unstable repeated
counts. Declare which allocator entry points are counted, including zeroed allocation, and verify
that each is observed. A deliberately retained allocation or reallocation in the interface must fail
the incremental comparison, with an unchanged warmed control passing. Do not offset a new interface
allocation with an unrelated removal elsewhere; retain a bounded interface-only control as well as
the real consumer. Allocation-count and ordinary timing runs are separate, since the counting
allocator changes timing. Missing instrumentation keeps the row pending, never waived.

For the readback budget, R01 adds explicit counters at the actual full-logit host-read sites and
compact candidate-read sites, recording transfers and bytes around the selected request. Existing
`HiddenReadbackPathProofSnapshot` counts hidden-state reads, not full-vocabulary logits; it is not
this instrument. A deliberate dense-logit substitution on a compact-eligible request must fail the
counter assertion even if tokens match. Instrument every admitted site and refuse missing route or
counter evidence. Native/driver allocation behavior remains a design and lifetime-review obligation
with #1584's separate RSS experiment; Rust counts do not certify it. Preserve batched prefill and
cache capacity; a narrower or unexercised path cannot satisfy these controls.

For timing, this refactor authorizes no demonstrated regression as an automatic tradeoff. Use the
runner's calibrated configuration-specific classification without changing its thresholds. A
calibrated regression blocks this program until fixed or an explicit new tradeoff decision is
approved. An informational or inconclusive result is neither a pass nor a demonstrated regression:
record the reason, retain it as pending, and use the bounded measurement plan in D8. New target
thresholds require same-SHA calibration before they can make a gating claim. This draft does not
claim a statistically proven zero-overhead boundary or assign an unsupported percentage budget.

### D8. Decide the disposition before booking measurements

R03–R13 have **eleven separate performance dispositions**. Eleven paired jobs is a **ceiling, not a
schedule**: documentation-only, isolated test-only and provable pure intra-crate move rows take the
structural disposition by default, subject to the proof below. Only rows touching a timed path or
its performance-relevant build inputs book a paired job. R04a, neutral-type moves and later
extractions have their own dispositions in the rollout table. Source preparation may overlap; a
disposition is required for each completed row even when no measurement is booked.

Paired jobs run on **one designated benchmark host, not the development laptop**, with admission by
the maintainer. Serialize through `/tmp/lion-bench-window.lock` and `/tmp/lion-metal-gpu-test.lock`
using the repository runners that acquire those locks themselves. The maintainer identifies the
designated host in each execution handoff. A lock acquired for an earlier run does not establish
current admission. This is a queue mechanism, not a wall-clock estimate.

Before each issue is implemented, freeze its base/head procedure, target/group, model hashes,
feature/toolchain/profile, short/long prompt and output budget, selected backend, sampler and
warm/cold region. Separate load time, prefill/first-token latency, decode and peak request memory
when supported by the instrument. A lower-level forward benchmark does not measure shared-driver
overhead merely because both compile the same file. Add missing driver/Gemma/HTTP measurements in
the characterization/instrument issue, shared unchanged by base and head, before consuming its
numbers. A target added only at head cannot produce a like-for-like base measurement.

For each diff, enumerate every declared benchmark and measurement binary in the affected crates,
their `cfg`/required features and timed caller chains. Documentation-only or isolated test-source
proof names the absent timed call path and verifies identical performance-relevant build inputs.
Pure intra-crate moves must satisfy **all four ADR-087 D2 conditions**: the intra-crate boundary,
closed permitted-difference categories, exhaustive line enumeration, and direction of error. An
identical-looking body or the row's label alone proves nothing. Each structural disposition states
its residual risk. If the diff changes timed behavior or its manifests, features, profiles,
dependencies, generated code or measurement harness, use a reachable paired measurement; a failed
proof remains unresolved until measured or repaired. The inventory is never a reusable waiver.

Use the repository paired runner with matching reachable targets. It runs four ABBA arms and checks
in-phase load as well as boundaries. Preserve its locks, admission checks and exit status;
report-only zero exit alone is not a green verdict. For a reachable non-Criterion target use the
documented base/head `bench-command.sh --durable` comparison, recording that its quiet checks are
narrower. Do not imply it inherits the full paired runner's AC/thermal/HID/cooldown checks. Add
required route/marker and nonempty-output checks so a skipped target refuses certification.

Before reserving each window, record observed same-target A/A/previous valid job durations on the
benchmark host plus compilation and admission separately. If none exist, book a calibration/pilot,
not an invented production estimate. At most one predeclared confirmation job follows an ambiguous
initial pair; unresolved results remain pending for owner disposition, not repeated until green.
Current four-arm timings must not be estimated by blindly doubling the historical two-arm laptop
bound in the operating guide. Moving the base under the measured paths invalidates that pair.

### D9. Establish the shared route before reorganizing the tree

The rollout table makes R03 and R04 the first live milestone: each must generate a real token stream
through the shared driver on its named checkpoint before any R14–R18 work or the D9 population/probe
rebuild lands. R01 may prepare generation instruments before R03; it does not front-load this
rebuild. The derived-population checker is an acceptance condition for extraction rows R14–R17, not
a prerequisite for R03/R04. The later R13 serving milestone still requires real Qwen CPU, Qwen Metal
and Gemma E2B CPU execution through both serving surfaces, with supported
prefix/default-MTP/self-spec policy routed as declared and D6's experimental exclusion explicit.
Fake sessions are useful for policy tests but do not prove this milestone.

Afterward retain public compatibility façades while grouping Gemma modules and continuing the
existing Metal submodule decomposition. At the pinned ref,
`crates/inference/src/forward/metal_qwen35.rs` contains **41,331 lines**; it is the specific
long-file target for R16's test-family and R17's engine-construction extractions. Preserve ERNIE's
use of Gemma RoPE helpers. R15 moves portable tests first. R16 deliberately starts the inner-family
moves with `rms_reduce_854_parity` as a stress case, not as a portable fixture: its raw-string
`ORACLE_MSL` is compiled by `new_library_with_source` at runtime. Preserve the raw payload bytes,
compiler options, GPU lock, device enforcement and oracle comparison. A brace-depth scan that treats
MSL text as Rust structure cannot authorize the extraction; use Rust-aware item boundaries and prove
raw-string/comment and nested-module handling before the move. Later inner families remain separate
child issues.

After both live CPU milestones and before any R14–R17 extraction is accepted, replace the source
guards' hand-maintained searched population with a derived, checked population. Discover the parent
plus every Rust file recursively under `src/forward/metal_qwen35/` from `CARGO_MANIFEST_DIR`;
reconcile module declarations and explicit path/include destinations so a move outside that subtree
cannot silently leave the population. Classify production items through enclosing Rust cfg/module
structure, not directory name, text indentation or a match for `mod tests`. A file is excluded only
when its entire reachable contents are proven test-only; mixed production/test files remain included
with their production items. Use the union over supported production configurations:
`any(test, all(target_os = "macos",
feature = "metal-gpu"))` is not test-only. Unreadable paths,
unresolved includes/macros or ambiguous classification fail the completeness check instead of
shrinking the input.

The directory walk is a candidate source inventory, not the boundary of guarded ownership. Resolve
the before/after guarded owner and caller manifest against the supported crate module graph,
including sibling/shared modules declared by an ancestor and reached through imports or re-exports.
Those resolved destination items must join the actual read set even when the old Metal root has no
outgoing `mod`, `#[path]` or `include!` edge to them. Freeze the guard's owner/caller obligations
before discovery; do not regenerate them only from the files that the walk happened to find.
Unresolved or ambiguous external ownership refuses certification. This is bounded resolution of the
declared guarded owners/callers, not a general whole-program call-graph project.

The post-milestone D9 rebuild includes an external-owner relocation control: move a guarded sampling
helper or owner to a sibling module declared by its parent, retain a Metal-side import or re-export,
and leave no outgoing module/path/include edge from the old root. The resolved sibling must be read
or the checker must refuse. The unchanged relocated owner remains covered; a bypassed call or
retired declaration in that destination must fail its guard. Deliberately omit the destination
during discovery and require a completeness failure before any negative-symbol search, not perfect
coverage over a reduced set. Also require refusal for an unresolved external destination. Keep the
existing controls for omissions after discovery, nested files, cfg, test decoys, raw strings and
early test modules.

The current retired-declaration guard misses `mtp_weights.rs`, which is active for macOS Metal
production as well as tests. Its six-source list is not a completeness invariant. After R03/R04, the
D9 rebuild prepares a test-side discovery/classification checker; its coverage manifest must
distinguish discovered, classified and actually read items. Every R14–R17 acceptance carries that
check even if a row touches only Gemma and the Metal population should stay unchanged. Earlier
driver rows preserve their existing guards and reconcile affected owners/callers locally; the full
D9 checker does not gate R03/R04. In-memory fixtures must demonstrate that omitting a discovered
production file fails coverage, adding a nested production file is automatically covered, and
inserting a retired declaration in a formerly omitted file fails the negative assertion. A test-only
declaration must not self-trigger it. Do not prove completeness by comparing two copies of the same
manual list.

The parent-only `sample_decode_traced` count and constructor probes also use an exact four-space
`mod tests` delimiter. The count's current six matches mean one definition plus five named calls,
not a perpetual numeric invariant. R06 and R11 relocate ordinary/streaming and prefix callers; R12
audits advanced-route interactions, R13 removes only declared superseded loops, R15/R16 move test
modules, and R17 moves constructors. Each row reconciles the relevant probe's symbol/caller manifest
before the move. Preserve the multimodal callers that this text-only milestone does not migrate.
Replace the literal delimiter with Rust-aware production-item classification, including production
items after inline test modules. Raw strings containing the delimiter, changed test indentation and
an early test module must not truncate the checked population. Both positive call-site and negative
retired-symbol controls remain sensitive after relocation; adjusting the constant until the test
passes is not acceptance. The rollout companion names all seven current self-source probes and their
affected rows.

Every extraction that relocates a symbol cited by maintained documentation updates that
documentation in the same change. Record old and destination file, qualified symbol, relevant cfg
and immutable source ref. Current-use citations name the destination symbol and file, optionally
with a full-revision line link. Historical evidence keeps its original ref and is labeled historical
rather than retargeted. A valid line number below EOF does not establish claim support. The separate
C01 documentation lane handles pre-existing citations and does not change the 20-row driver scope.

Keep shader assembly bytes and compile options fixed. The production shader source is already
external; changing translation units, compiler concurrency or library routing is a separate measured
build/behavior change, not source-file housekeeping.

### D10. The serving surface stops naming concrete model types

Added at sign-off, 2026-09-15. The rest of this ADR proves the shared route by EXECUTION: R13
requires real token streams from both families through both serving surfaces. That is not the same
claim as the serving surface being model-agnostic. A serve worker can route every request through
the shared driver and still hold a concrete model type in its own struct, which is the shape at the
source ref.

Measured baseline at `7ba69b2f1c461b10b8c26eff458bc0afc1c3d07c`, stated so the acceptance has
something to move: `crates/inference/src/serve/` is five files, three of which name a concrete model
type, nineteen occurrences in total, sixteen of them `MetalQwen35State` in `serve/metal_worker.rs`,
two `Qwen35Model` in `serve/embeddings.rs`, one `QwenModel` in `serve/mod.rs`. The serving binaries
carry eleven, six and one. No second-family type occurs anywhere under `serve/`.

**The acceptance is an execution arm, and R13 does not complete without it.** A model family reaches
both serving surfaces through the worker factory with no edit to any file under
`crates/inference/src/serve/` and no edit to the serving binaries. It is proven by a real additional
family, or by a fixture family constructed the way a real one would be: loaded, admitted, and
routed through the same factory path. A fake session does not prove it, for the same reason D9 gives
for the live milestones.

**A name search rides beside that arm as a cheap tripwire, and is never promoted to the acceptance.**

```zsh
pat='MetalQwen35State|Qwen35Model|QwenModel|Gemma4Model|Ernie45[A-Za-z]*|PaddleOcr[A-Za-z]*'
serve=('crates/inference/src/serve/*.rs' 'crates/inference/src/bin/*.rs' 'crates/inference/src/bin/**/*.rs')

files=$(git grep -l '' -- $serve | wc -l | tr -d ' ')
ctl=$(git grep -w -E "$pat" -- 'crates/inference/src/forward/*.rs' | wc -l | tr -d ' ')
hits=$(git grep -nw -E "$pat" -- $serve | wc -l | tr -d ' ')

if   (( files == 0 )); then print -r -- "FAIL population empty: the search had nothing to read"
elif (( ctl   == 0 )); then print -r -- "FAIL control dead: the pattern matched nothing where it must match"
elif (( hits  > 0  )); then print -r -- "FAIL $hits concrete-type occurrences on the serving surface"
else                        print -r -- "PASS ($files files searched, control $ctl)"
fi
```

Three arms, one pass, and the order is the point: the population must be non-empty and the control
must match before an absence on the serving surface is allowed to mean anything. Run at
`79d422805bbde537e19bd17f2e89dd3fbf4780f6` it prints
`FAIL 134 concrete-type occurrences on the serving surface`, which is the expected reading today:
the execution arm above has not landed, so the tripwire should be red. A green reading before that
work exists would itself be the defect.

Its false negatives are written here rather than left to be rediscovered, because a tripwire that
looks like a gate is how the weaker check replaces the stronger one:

1. **A type alias.** `type Worker = MetalQwen35State;` in a neighbouring module satisfies the search
   at the serve site while the coupling is unchanged. Every lexical check is defeated by renaming.
2. **A family enum behind the factory.** Serve holds the boxed session; the factory matches on a
   family enum and constructs the concrete type one module away. The search passes, and adding a
   family still cannot be done without editing that match, which is the thing the acceptance is
   for.
3. **A cfg-selected type of the same name.** A platform-gated stub keeps the coupling portable and
   invisible to a search run on one platform. Measured at `79d422805bbde537e19bd17f2e89dd3fbf4780f6`,
   this one is not hypothetical: it is the shape two owners already have. `MetalQwen35State` is a
   real struct inside a `#[cfg(all(target_os = "macos", feature = "metal-gpu"))]` module and a unit
   struct under `#[cfg(not(...))]` with the same public generation surface, and
   `MetalErnie45State` is declared the same two ways. A run on Linux and a run on macOS are reading
   different type sets, so the platform the tripwire runs on is part of its result.

#### Amendment, 2026-09-17: the published command matched nothing, and the check is negated

The command first published with D10 used `\b` for its word boundaries. `git grep -E` is POSIX ERE
and has no `\b`, so the pattern matched nothing, silently. Measured at
`79d422805bbde537e19bd17f2e89dd3fbf4780f6` in one pass: the published form returns **0**, the same
alternation under `-w` returns **134**, and the POSIX bracket form `[[:<:]]…[[:>:]]` also returns
**134**. The two working forms agree; the published one is the outlier.

The direction is what makes this more than a typo. The check is negated, so a search that can never
match is a search that always passes, and as published it would have certified the serving surface
clean forever. That is why the replacement above asserts its own population and runs a must-match
control in the same invocation rather than beside it: an absence produced by a broken instrument and
an absence produced by a clean tree print the same thing. The tripwire was not wired into any
workflow, script or test, so nothing was falsely green in the interim; the hazard was prospective.
Tracked as [#1649](https://github.com/ohdearquant/lattice/issues/1649).

**The baseline re-derived at the same commit, beside the pin figures, so the spread is visible.**
Scope is `crates/inference/src/serve/*.rs`, the same scope the pin paragraph uses:

|                              | at `7ba69b2f1c` | at `79d422805b` |
| ---------------------------- | --------------- | --------------- |
| files in scope               | 5               | 7               |
| occurrences                  | 19              | 19              |
| files naming a concrete type | 3               | 4               |

The total held while the coupling spread, which is the reading an aggregate hides: two files were
added to the scope, a fourth file began naming a concrete type, and `MetalQwen35State` moved into
`serve/lora_registry.rs`, a file that did not exist at the pin. Per file and type at
`79d422805b`: `MetalQwen35State` 13 in `serve/metal_worker.rs`, 2 in `serve/lora_registry.rs`, 1 in
`serve/mod.rs`; `Qwen35Model` 1 in `serve/metal_worker.rs` and 1 in `serve/embeddings.rs`;
`QwenModel` 1 in `serve/embeddings.rs`.

One correction to the pin paragraph above while re-deriving it. Its per-type totals are right (16
`MetalQwen35State`, 2 `Qwen35Model`, 1 `QwenModel`, 19 in all) but three of the file attributions
are not: at the pin the 16 `MetalQwen35State` are 15 in `serve/metal_worker.rs` plus 1 in
`serve/mod.rs`, the 2 `Qwen35Model` are 1 in `serve/metal_worker.rs` plus 1 in
`serve/embeddings.rs`, and the single `QwenModel` is in `serve/embeddings.rs`, not in
`serve/mod.rs`. The counts a later reader would diff against are the per-file ones, so they are
restated here rather than left to be rediscovered.

A stronger enforcement exists and is deliberately not taken here: moving serving into a crate that
cannot depend on the concrete model implementations would make the separation structural rather than
lexical. That is a larger change than this ADR decides, it interacts with the crate-ownership rules
recorded elsewhere, and it is not a prerequisite for the execution arm above. Recording it keeps the
option visible for a later decision rather than leaving the grep looking like the ceiling.

## Alternatives considered

| Alternative                                      | Rejection / retained use                                                                                                                             |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shared numerical forward with strategy hooks     | Violates the retained ADR-009 boundary and hides Gemma/Qwen cache and math differences.                                                              |
| Public model enum everywhere                     | Repeats family matches and exposes exhaustive API churn; use a small private loader registry instead.                                                |
| Generic model parameters through HTTP/runtime    | Couples frontend types to kernels; retain concrete generics below the session boundary.                                                              |
| Trait wrapping whole existing `generate` methods | Shares dispatch only; keeps policy loops duplicated.                                                                                                 |
| Universal KV/reset-length protocol               | Cannot express typed donor caches, recurrence state and speculative repair safely.                                                                   |
| File reorganization before the driver            | Adds path/cfg/source-probe churn before the interface is proved; permit only a bounded demonstrated prerequisite.                                    |
| Batch verifier silently included in R12          | Preserves a stateful experimental path as a universal capability without successful-path proof. D6 keeps a visible bounded legacy exclusion instead. |

## Consequences and acceptance status

The design shares request control while leaving numerical specialization possible. It adds a private
dynamic boundary whose costs must be measured and a temporary, explicit batch-verifier exclusion.
Implementation starts with live generation on both named CPU checkpoints, then follows per-diff
dispositions and the existing Metal prerequisites. Keeping compatibility wrappers and multiple
diagnostic paths temporarily costs code; deleting them early would discard the comparison needed to
validate migration.

Before dependent implementation, review the companion ADR deltas together, compile-check the three
private interface sketches in the first implementation packet, and prove the new routing/admission
controls with hot/cold pairs and a must-reject control. This ADR proposes new gates, so absence of
an implementation does not count as a passing adversarial suite. Formal specification sign-off
completed 2026-09-15; execution of those controls remains pending, and a source review is not
their substitute.

## References

- [ADR-009](ADR-009-model-architectures.md),
  [ADR-080](ADR-080-consolidation-duplicated-contracts.md),
  [ADR-082](ADR-082-gemma4-e2b-support.md), [ADR-074](ADR-074-mtp-speculative-decoding-priority.md),
  [ADR-087](ADR-087-bench-compare-gate-calibration-and-coverage.md).
- [ADR-046](ADR-046-structured-output.md), [ADR-049](ADR-049-vision-encoder.md),
  [ADR-055](ADR-055-online-drift-detection.md), [ADR-063](ADR-063-serving-architecture.md),
  [ADR-068](ADR-068-grammar-wire-contract.md), [ADR-076](ADR-076-adaptive-reasoning-priority.md),
  [ADR-086](ADR-086-retire-legacy-qwen-decode-api.md).
- [Current generation config and static budget](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/model/qwen35_config.rs#L2340),
  [final-token transition](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/model/qwen35/generation.rs#L1727),
  [multimodal preflight](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/forward/metal_qwen35.rs#L38595),
  [request normalization](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/serve/contract.rs#L487),
  [config construction](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/bin/lattice_serve.rs#L1742),
  [structured grammar assignment](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/bin/lattice_serve.rs#L2079).
- [Qwen generation](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/model/qwen35/generation.rs),
  [entry preparation](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/model/qwen35/generation_setup.rs),
  [Gemma execution](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/model/gemma4_model.rs),
  [Gemma cache](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/model/gemma4_cache.rs).
- [Metal batch verifier](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/forward/metal_qwen35.rs#L4420),
  [selection](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/forward/metal_qwen35.rs#L8682),
  [cursor negative](https://github.com/ohdearquant/lattice/blob/292658628f49f16daa04673afe2649eb4f2a5e8d/crates/inference/src/forward/metal_qwen35.rs#L18836).
- [GDN normalization issue](https://github.com/ohdearquant/lattice/issues/1583),
  [autorelease lifetime issue](https://github.com/ohdearquant/lattice/issues/1584).
- [Upstream family variation](https://github.com/ggml-org/llama.cpp/pull/28335),
  [drafter artifact correction](https://github.com/ggml-org/llama.cpp/pull/28183).
