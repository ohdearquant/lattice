# ADR-102: Reorganising Metal execution around shared mechanisms and typed model sessions

**Status**: Proposed (2026-10-16)\
**Date**: 2026-10-16\
**Origin**: reorganisation request, 2026-10-15\
**Crate**: lattice-inference (`forward/`, `decoder/`); repository CI (file-size gate)\
**Related**: ADR-090 (extends D9 and its rows R15 to R17; keeps D6), ADR-074 (batch-GEMM MTP disposition), ADR-087
(measurement dispositions)

## Decision summary

The Qwen3.5 Metal implementation is split by responsibility into private modules until every one of its Rust files is at
most 3000 lines. After that, small Metal mechanisms that two models demonstrably share move into a crate-private
`forward::metal_runtime` module, and the remaining generation routes move onto the existing backend-independent
decoder. A CI gate caps every maintained file at 3000 lines; files already over the cap are frozen at their current size
and may only shrink. No public plugin interface, no universal layer trait and no new crate are introduced. Public paths
under `lattice_inference::forward::metal_qwen35` stay valid. The test and engine-construction moves are ADR-090's
rollout rows R15 to R17 and keep that ADR's order and acceptance; this ADR extends them to the rest of the file.

## Context

### What lattice has today (source-read at `main` a292df76)

| #  | Fact                                                                                                                                                                                                                                                                                                                                                                                                          | Pointer                                                                                                                        |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| C1 | `metal_qwen35.rs` is 47,220 lines: 15,879 lines of production code and 31,341 lines in 35 items gated on `cfg(test)`. One of those items, the inline `mod tests` inside `mod inner`, spans 27,760 lines. A separate pure-move change takes its topic sections into 21 files, leaving 22,550 lines: the same 15,879 production lines and 6,671 test lines, of which the remaining inline test parent is 3,090. | `crates/inference/src/forward/metal_qwen35.rs`, brace depth over each `cfg(test)`-gated item with comments and literals masked (a plain search for `cfg(test)` also matches three comments) |
| C2 | A backend-independent decoding boundary already exists: `DecoderSession` for ordinary token loops and `SpeculativeSession` for verified rounds. The driver takes `&mut dyn DecoderSession`.                                                                                                                                                                                                                   | `decoder.rs` (`trait DecoderSession`, `trait SpeculativeSession`), `decoder/driver.rs` (`fn run`)                              |
| C3 | Qwen3.5 direct, streaming and prefix-cache generation already run through that driver, as do sequential MTP and self-speculation.                                                                                                                                                                                                                                                                             | `decoder/qwen_metal.rs` (calls to `driver::run`, `run_speculative_direct`)                                                     |
| C4 | Four generation routes still run their own loop inside the model state: batch-GEMM MTP, text-plus-patch multimodal, vision generation, and the n-gram speculation entry. CI asserts the multimodal and vision bypasses explicitly.                                                                                                                                                                            | `metal_qwen35.rs` (`generate_greedy_mtp`, the multimodal and vision `generate` loops, `generate_with_speculation`)             |
| C5 | The ERNIE 4.5 and EmbeddingGemma 2 Metal paths contain near-identical allocation, binding and element-dispatch helpers, but their contracts differ in detail: ERNIE allocates exactly `bytes`, EmbeddingGemma 2 allocates `bytes.max(4)`, and the error prefixes differ. Qwen3.5's helpers return `Buffer` where the other two return `Result`.                                                               | `forward/metal_ernie45/state.rs` (`fn allocate`), `forward/metal_embeddinggemma2/state.rs` (`fn allocate`)                     |
| C6 | Hybrid-model state is not a plain KV cache. Speculative rollback restores GatedDeltaNet checkpoints, may replay a repair token and repairs the MTP cache position. M-RoPE separates the physical cache position from the rotary coordinate.                                                                                                                                                                   | `metal_qwen35.rs` (rollback and M-RoPE prefill paths)                                                                          |
| C7 | Declared Metal measurement targets, three bench targets and one measurement binary, execute Qwen3.5 forward and generation code. A production move inside `metal_qwen35` is therefore reachable from measured targets.                                                                                                                                                                                                                                                     | `crates/inference/Cargo.toml` (`[[bench]]` `metal_decode_bench`, `mtp_decode`, `cross_turn_prefix_cache_bench`; `[[bin]]` `bench_vision_prefill_ab`) |
| C8 | 39 tracked files exceed 3000 lines: 21 Rust or Metal source files, 4 Python files, 13 data files (tokenizers, manifests, recorded results) and `Cargo.lock`.                                                                                                                                                                                                                                                  | `git ls-tree -r` at `main`, counted with `wc -l`                                                                               |

### What ADR-090 already decides about this file

ADR-090 (Accepted) owns the first part of this work, and this ADR extends it rather than replacing it:

- **Order.** D9 establishes the shared route before the tree is reorganised. Its rollout ledger (`R00-ROLLOUT.md`)
  puts both live CPU results (R03 for Qwen, R04 for Gemma E2B) and the R13 serving milestone before any extraction
  row: Gemma folder grouping (R14), portable tests (R15), the inner test families one per change starting with
  `rms_reduce_854_parity` (R16), engine construction (R17), and then one documentation row (R18) for the remaining
  per-responsibility manifests. ADR-090's amendment of 2026-10-16 lets a pure test move land before R13 under
  stated conditions; production moves keep this order.
- **Acceptance of every extraction.** Before any R14 to R17 extraction is accepted, the source guards' hand-maintained
  searched population is replaced by a derived, checked population over the parent and every file under
  `forward/metal_qwen35/`. The seven parent self-source probes listed in the ledger each get a before/after population
  or an explicit no-impact disposition.
- **Documentation.** A change that relocates a symbol cited by maintained documentation updates that citation in the
  same change.
- **Batch-GEMM MTP.** D6 keeps the batch-GEMM verifier outside the shared execution contract until a separate ADR-074
  E1/E2 validation decides whether it is promoted or retired.

This ADR adds what ADR-090 does not cover: splitting the remaining production code, the 3000-line cap, the shared Metal
mechanisms, and moving the routes ADR-090 left as declared bypasses.

### The problem

A single file holding 15,879 lines of production code and over 30,000 lines of tests cannot be reviewed or navigated, and every new capability lands in it. The goal is
cheap model bring-up: a model that reuses existing kernels and state types should mostly supply configuration, weight
mapping and layer composition. A new architecture (a new recurrence, expert schedule or multimodal position scheme) will
still need its own kernels and state, and this ADR does not pretend otherwise.

### Roles this reorganisation keeps

lattice is both a library (the published crates and their public API) and a long-running inference service (the
serving binaries) that a host starts, stops and restarts. It is not an extension of a host application; that was
decided on request on 2026-10-16. So this ADR adds no public plugin or model interface: the reorganisation is internal
to `lattice-inference` and changes neither the library surface nor the service's lifecycle.

## Decision

### D1. Concrete modules first, shared mechanisms on evidence

Model implementations stay concrete, with typed engine and session state. A helper moves into the shared module only
when two real callers use it with documented compatible semantics, and the extraction preserves each caller's existing
contract (C5) instead of silently standardising it. The session-level dynamic dispatch of C2 stays as it is.

Not adopted now:

- a generic `MetalModel` trait (load, layer plan, encode), whose boundary would be guessed before reuse exists;
- the same trait as `dyn` objects, which would need an object-safe contract over associated state, specialised prefill
  and verification;
- an interpreted layer-plan enum as the universal contract, which GDN snapshots, MoE schedules, M-RoPE and MTP hidden
  taps would keep widening.

An enum or trait inside one model's implementation is fine. A layer-level interface is reconsidered when a second
architecture gives a concrete case where duplication is worse than the interface.

### D2. The existing decoder owns generation policy

Generation policy (stops, grammar, cancellation, finish handling, prefix disposition, tracing) stays in `decoder/`. The
four routes of C4 move onto it one route per change, each through a typed session that owns its own state: a multimodal
session owns injected embeddings and M-RoPE positions, and the driver never learns image grids. Ordinary and
verified-round loops stay separate drivers that share policy.

Each route migration preserves, and tests old against new:

| Concern                 | Preserved                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Stops and output        | Step-zero EOS, extra stop ids, stop strings spanning tokens, UTF-8 detokenizer tails, callback refusal, per-entry stop reasons |
| Grammar and metadata    | Mask, then selection, then grammar advance, then stop check, then push; the logits scored for metadata are the ones sampled    |
| Cancellation and finish | Before- and after-prefill and in-loop checkpoints; finish runs once; a finish error does not replace the primary error         |
| Prefix disposition      | Slot consumption, represented-token boundary, GDN and KV agreement, the optional final forward                                 |
| Tracing                 | Ordinary and speculative counters, route markers, Metal signpost boundaries; token equality alone does not prove the route ran |

A bypass assertion in CI is removed only in the change that replaces it with positive evidence that the driver ran the
route. The speculative driver's current refusals (grammar, logprobs, stop strings, reasoning budgets, non-greedy
requests) stay in force.

### D3. Order: split, then share, then migrate routes

1. **Split** `metal_qwen35` by responsibility into private modules (phases M1 to M8 below), each a move with no
   semantic change, until every `metal_qwen35` Rust file is at most 3000 lines.
2. **Share**: extract the ERNIE and EmbeddingGemma 2 helpers into `forward::metal_runtime` (S1 to S3), then adopt them
   in Qwen3.5 only where a small adapter keeps its contract (S4).
3. **Migrate** the remaining generation routes onto the decoder (D1 to D4 below).

Production moves start only after both of ADR-090's live CPU results (R03, R04) and its R13 milestone; a pure test
move may land earlier under that ADR's 2026-10-16 amendment. Every move phase carries ADR-090 D9's derived-population
check (row G1 below) and its documentation rule in its acceptance.

A move change never contains an extraction, and an extraction never contains a route migration, so a regression can be
attributed to one kind of change. No shared extraction and no new Metal model architecture merges while any `metal_qwen35`
Rust file keeps a manifest entry.

### D4. A 3000-line cap per file, with a ratchet for existing files

A CI script counts physical lines (the number of LF bytes, plus one for a final unterminated line) in every text file
git tracks at the change's head; a file with a NUL byte in its first 8 KiB is binary and is not counted. A file is
maintained when its extension is `.rs`, `.metal`, `.py`, `.sh`, `.bash` or `.md`, or when it is a workflow
definition directly under `.github/workflows/`. Tests and benches are included, and comments and blank lines count.
Formatting checks stay on, so packing code onto fewer lines is not a way to pass.

- **New files**: a maintained file without a manifest entry has at most 3000 lines.
- **Existing oversized files**: each has a ratchet entry whose ceiling is its count at the introducing commit. The
  head count may exceed neither the recorded ceiling nor the file's count at the change's base. A change that shrinks
  a listed file lowers its ceiling to the new count in the same change, and a ceiling left above the file's count
  fails the gate, so a file cannot shrink and later regrow to a stale ceiling. A file at or under 3000 lines leaves
  the manifest in the same change and cannot return. A rename carries its entry explicitly, naming the old path, and
  the ceiling rules apply against the old path's count at the base.
- **Data**: any other tracked text file over 3000 lines (tokenizer files, recorded fixtures and results,
  `Cargo.lock`) has a data entry naming its exact path. Globs and directories are refused, a maintained path cannot
  be listed as data, and an entry whose file is gone or no longer over 3000 lines fails as stale.
- **The manifest is not regenerated by CI.** A new ratchet entry or a raised ceiling fails the gate; a new data entry
  does not.

Each `metal_qwen35` Rust entry names the phase that retires it. The Qwen3.5 shader source
`forward/shaders/qwen35.metal` (3,311 lines) and the other oversized files are frozen by the same ratchet. Splitting
the shader changes the compiled shader source and is not part of this plan; retiring those entries is separate work,
scheduled outside this ADR.

A cohesive family of methods on one state can be split across private submodules with several `impl` blocks, and
shared test fixtures go to a test support module. No trait or crate is invented only to cross a file boundary.

### D5. LoRA, speculation and grammar

| Capability               | Shared                                                                                          | Per model                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| LoRA hot-swap            | Adapter selection in serving; upload and dispatch helpers once two consumers prove the contract | Projection names and shapes, rotation and basis validation, injection sites, supported routes       |
| MTP and self-speculation | `run_speculative` owns publication, length and stop policy over verified rounds                 | MTP head, hidden tap, GDN-first draft schedule, checkpoints, repair and replay, KV/MTP/GDN rollback |
| Grammar                  | The decoder owns grammar state and transitions                                                  | The session applies the mask to the prediction it samples and reports exhaustion                    |

"Hot-swap" means between requests. Loading or unloading an adapter keeps clearing cross-turn cache entries, because
equal shapes do not prove equal adapter contents. The batch MTP verifier stays disabled under an active adapter, and
grammar and logprobs keep requiring dense readback.

### D6. Layout and names

`forward/` stays. Shared helpers go in a new crate-private `forward::metal_runtime` (buffers, pipeline construction,
dispatch helpers, and later KV storage, GPU sampling routes and LoRA storage when two consumers justify them). The name
`forward::metal` already belongs to the Qwen3 path and is not reused. Qwen3.5 code moves under
`forward/metal_qwen35/inner/`, which already holds the dispatch helpers, GDN state, `layers/` and some test modules.
The remaining parts (engine, session, buffers, pipelines, `weights/`, forward step, `prefill/`, `speculative/`, LoRA,
prefix cache, `generation/`, diagnostics) are added as each phase lands.
`metal_qwen35.rs` becomes a facade that keeps the public items, non-Metal stubs and bench-gated exports.

`metal_runtime` must not import Qwen3.5 types or call generation wrappers. The public path
`lattice_inference::forward::metal_qwen35::*` stays valid; this ADR schedules no removal.

## Plan

Each row is one pull request. Every row passes the file-size gate. The 15,879 production lines need at least six
files at the cap; the plan lands eleven to fourteen production files, most well under it, plus the facade. The planned counts
below are estimates: each phase's pull request fixes its exact item list and file count from the code at its own base.
An acceptance cell names what is specific to its row. The measurement rule below applies to every row that moves or
changes production code, whether or not its cell repeats it.

| Phase | Change                                                                     | Files (planned)   | Acceptance                                                                                                                                    |
| ----- | -------------------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| P0    | File-size checker and manifest (D4)                                        | script + manifest | Gate cases: a new 3001-line file, growth of a listed file, shrink then regrow, a forged ceiling, removal, rename, CRLF, missing final newline |
| G1    | ADR-090 D9 derived-population checker for the source guards                | test + checker    | ADR-090 D9 controls: omitted, nested and relocated files, cfg and test decoys, raw strings; required by every M row                           |
| M1    | Portable helpers, stubs and portable tests into sibling modules (R15)      | 1-2               | Item map and body comparison; test lists equal under a path map; Metal and non-Metal public-surface builds; G1 coverage                       |
| M2    | Inline Metal test families into named test modules (R16 and its children)  | 3-4               | As M1; GPU-lock contract still finds every construction site; raw MSL payloads unchanged; probe dispositions; G1 coverage                     |
| M3    | Engine and session types, buffer and pipeline construction (R17)           | 2                 | Move evidence and paired Metal measurement; lifetime, allocation and command-completion behaviour unchanged; G1 coverage                      |
| M4    | Weight loaders and adapter operations                                      | 1-2               | Move evidence and measurement; Q3/Q4/MTP/MoE mapping and refusal tests, adapter rotation and invalidation unchanged                           |
| M5    | Forward-step orchestration and its preflight checks                        | 1                 | Move evidence and measurement; Q4/Q8 decode, LoRA path, dispatch markers, error-before-mutation checks                                        |
| M6    | Prefill orchestration (dense, full attention, GDN, chunked)                | 2                 | Move evidence and measurement; chunk-boundary state agreement, hidden and logit capture, MoE refusal                                          |
| M7    | Speculative execution, verifier and checkpoint mechanics                   | 2                 | Move evidence and measurement; sequential MTP, self-speculation and batch-GEMM routes unchanged and distinguishable                           |
| M8    | Generation wrappers, prefix cache, diagnostics, bench support              | 2-3               | Every `metal_qwen35` Rust file at most 3000 lines and its manifest entries removed; measurement; shared extraction may start |
| S1    | ERNIE and EmbeddingGemma 2 binding, scalar and element-dispatch helpers    | 1                 | Both callers on one implementation; same thread geometry and scalar layout; parity and measurement                                            |
| S2    | Their checked allocation and upload                                        | 1                 | Both zero-length policies, overflow and device-limit checks, error context preserved; parity and measurement                                  |
| S3    | Common pipeline construction                                               | 1                 | Same shader sources, function names, constants, compile options and failure behaviour; measurement                                            |
| S4    | Qwen3.5 adopts S1 to S3 where a small adapter keeps its contract           | 0                 | Command-buffer granularity, resource lifetime, readback volume and errors unchanged; measurement                                              |
| D1    | Text-plus-patch generation through a typed session and the decoder         | 1                 | Old against new output and state, first-token and stop-string flush; the bypass assertion replaced by an execution proof                      |
| D2    | Vision generation through a session owning its M-RoPE and injected prefill | 1                 | Image and text-only cases; physical against rotary position; cancellation; bypass assertion replaced                                          |
| D3    | Batch-GEMM MTP on the verified-round driver, if ADR-074 E1/E2 promotes it  | 0-1               | Every rejection boundary and repair replay; caches agree with the old route; positive speculative marker. If retired, this row removes it     |
| D4    | The n-gram speculation entry on the decoder                                | 0-1               | Its current cache, prefill and stop behaviour documented first; then parity, route proof and measurement                                      |
| V1    | A further model using the shared pieces without copying generation policy  | per model         | Driver proof, numerical parity against a reference, file cap, measurement                                                                     |

### Measurement for moves

C7 means a production move is reachable from declared Metal benches, so "pure move" is review evidence, not a
measurement waiver. Every M, S and D phase that touches production code carries a paired base-against-head measurement
of the reachable targets, scoped to the affected groups. Only test-only changes, after the full declared-target
population has been searched, may use the structural unreachability disposition.

## Alternatives considered

| Alternative                                                       | Why not                                                                                                                    |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Top-level `runtime/metal/` plus `models/*`                        | Adds a second model hierarchy and moves established owners, with public-path churn and no answer to the interface question |
| Shared runtime first, from the ERNIE and EmbeddingGemma 2 helpers | Leaves the largest file untouched, and treats purpose-similar helpers as equivalent (C5)                                   |
| Interleaving moves and abstractions in one change                 | Ownership, call structure and behaviour change together, so a regression cannot be attributed                              |
| A new Metal decode loop owner under `runtime/metal/decode`        | Duplicates the backend-independent decoder (C2) and gives CPU and Metal semantics another place to diverge                 |
| Byte-based cap, or excluding tests                                | No byte figure corresponds to the 3000-line review limit; excluding tests leaves the inline test parent oversized          |
| Per-file calendar deadlines                                       | Deadlines expire on untouched files and block unrelated work; phases tie removal to the work that does it                  |

## Consequences

- Qwen3.5 becomes navigable before any interface work, and the move evidence for each phase is mechanical.
- Some duplication remains until a second consumer exists. That is intended.
- Every production phase costs a Metal measurement run, which the measurement host's capacity bounds.
- A model on an existing architecture still needs its own module for configuration, weight mapping and composition.
  Bring-up is cheaper because policy and mechanisms are reused, not because a model fits in one small file.
- Hybrid architectures keep their own state and rollback. A shared cache type is adopted only where its semantics hold.

## Decided alongside

- **Old public paths stay.** Paths under `lattice_inference::forward::metal_qwen35` keep resolving through re-exports;
  removing any of them needs its own decision and a breaking release.
- **The V1 model is chosen when M8 lands**, by proposal and review at that time, from the models the project needs
  then. Reusing existing attention and quantisation tests the shared pieces cheaply; a new recurrent, expert or
  multimodal architecture tests them harder and needs new kernels.
