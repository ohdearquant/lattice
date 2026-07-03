# ADR-068: Pluggable Custom-Grammar Wire Contract for the OpenAI-Compatible Serving Surface

**Status**: Proposed
**Date**: 2026-07-03
**Crate**: lattice-inference (serving binaries: `lattice_serve`, `lattice`, `chat_metal`)

> Builds on ADR-046 (grammar engine) and ADR-063 (serving architecture). This ADR decides
> the request/response **wire contract** by which a caller reaches the already-shipped grammar
> engine with an arbitrary formal grammar, not just JSON. No engine internals change.

## Context

Lattice already has a grammar-constrained decoding engine (ADR-046). `GrammarSpec` is a closed
two-variant enum — `JsonSchema(serde_json::Value)` and `Gbnf(String)` — both compiling to one
internal byte-level PDA. The engine is opt-in through a single `GenerateConfig` field,
`grammar: Option<Arc<GrammarEngine>>`, and applies `mask_logits` before sampling. Constraining
decoding to an arbitrary context-free grammar is therefore a **solved, shipped capability**.

What does not exist is any way for a network or CLI caller to reach it. Every caller-facing
generation-config construction hardcodes `grammar: None`:

- `bin/lattice_serve.rs:202` (the shipped Metal/f16 HTTP daemon; `build_cfg`)
- `bin/chat_metal.rs:676` and `:779` (interactive chat, including its `--serve` warm mode)
- `bin/lattice.rs` (the unified-CLI binary from ADR-063) parses `response_format` only as a bare
  `{ "type": String }` discriminator (`ResponseFormat` at `:276`); a real `json_schema` payload's
  nested schema is silently dropped by serde, and `reject_unsupported` (`:525`) returns HTTP 400
  for any `response_format.type` other than `"text"`.

So the engine is reachable today only from Rust test code. Issue **#588** (OPEN) tracks one slice
of the gap: wiring `response_format.json_schema` into `lattice_serve`, with per-schema caching
(a `GrammarEngine` compile over Qwen3's 248,320-token vocabulary costs 50–200 ms per
`grammar/engine.rs:148-152`) and fail-closed 400s. #588 does not cover interactive chat, non-JSON
grammars, or any general "pass me a raw grammar" field.

The concrete driver for going beyond JSON is **LNDL** (a Lion-ecosystem symbolic DSL). Its
Layer-1 syntax is tag-structured prose with typed declaration/execution blocks and a terminal
`OUT{}` return. That output shape is deliberately **not JSON**, but Layer-1 is an ordinary
context-free grammar: free prose between tags is just "any bytes until the next delimiter," an
unremarkable character-class-plus-repetition construct that the existing GBNF front-end already
expresses. LNDL's typed and runtime layers are host-side concerns, not an inference-engine
responsibility.

The ecosystem offers **three incompatible conventions**, not one:

| Pattern | Representative servers | Shape |
| ------- | ---------------------- | ----- |
| A — sibling field outside `response_format` | vLLM, SGLang, llama.cpp | grammar is a parallel request field; `response_format` stays JSON-only |
| B — folded into the `response_format.type` enum | Fireworks | `{ "type": "grammar", "grammar": "<BNF>" }` as a peer of `json_object`/`text` |
| C — separate tool-scoped primitive | OpenAI Custom Tools | constrains one tool-call argument (Lark/regex), not the message |

One thing does converge across the field: the grammar **text format**. llama.cpp's GBNF is the
de-facto lineage; the widely-used masking libraries follow that EBNF spelling. **Lattice's engine
already speaks GBNF natively**, so aligning the wire on GBNF text costs zero grammar-format
translation. Adopting a Lark dialect (Pattern C) would require a new front-end compiler — out of
scope here and a direct contradiction of "engine internals are fixed."

## Decision

Expose the grammar engine through **two disjoint request channels with disjoint audiences**,
because there are two consumer classes with genuinely different needs:

1. **Generic OpenAI-SDK clients** (Aider, Continue, LangChain, LlamaIndex — the ecosystem-adoption
   wedge from ADR-063) will only ever send standard `response_format`. They must get OpenAI-identical
   behavior and must never be broken.
2. **First-party / power callers** (the Lion ecosystem, including an LNDL-to-GBNF compiler) control
   their own client and want to hand lattice an arbitrary formal grammar. They do not need
   OpenAI-SDK ergonomics; they need a clean, explicit, lattice-owned field.

Concretely:

- **JSON modes stay in `response_format`, OpenAI-faithful.** `response_format.type ∈
  {text, json_object, json_schema}`. `json_schema` carries the real nested schema and compiles to
  `GrammarSpec::JsonSchema`. This is exactly #588's scope and is the drop-in-compatible path.
- **Custom grammars go in a new sibling top-level field, `grammar`** (Pattern A), which a standard
  OpenAI client never emits and never sees. It compiles to `GrammarSpec::Gbnf`.
- The two are **mutually exclusive** on one request; sending both is a fail-closed 400.
- The **response shape is unchanged** — constrained output flows through the normal
  `choices[].message.content` and streaming deltas.

### The 6 forks — positions and reasoning

**Fork 1 — request-shape convention → (b) sibling top-level `grammar` field.**
Rejected (a) Fireworks-style `response_format.type = "grammar"`: it pollutes an enum that real
OpenAI client libraries validate against `{text, json_object, json_schema}`, and it conflates the
must-stay-clean passthrough path with the lattice-specific extension. Rejected (c) OpenAI
Custom-Tool-CFG: wrong semantic scope (it constrains a single tool-call argument, not the whole
message, whereas "the entire reply is LNDL" is a whole-message constraint), its Chat-Completions
endpoint support is unverified in practice (see Live experiments), and it mandates Lark, which the
engine cannot compile without new front-end work. Pattern A is also the dominant OSS convention
(three of four surveyed servers), so this choice converges with the ecosystem rather than inventing.

**Fork 2 — dialect surface → constrain the engine to GBNF + JSON-Schema, keep the wire `syntax`
field open.** The wire exposes `grammar.syntax` as a validated enum, but the only value accepted
today is `"gbnf"` (`"json_schema"` is also routable through the sibling field as a convenience for
callers who prefer one field, but the canonical JSON path remains `response_format`). Any other
value — `"lark"`, `"regex"`, anything — returns a fail-closed 400. The field exists so a future
dialect is an additive change (new enum value plus new compiler), never a wire break; but lattice
never advertises a dialect the engine cannot actually mask. Lattice does not take on chasing N
grammar front-ends: every other formal language compiles down to GBNF text on the caller's side.

**Fork 3 — LNDL integration locus → no LNDL-specific code in lattice.** LNDL-to-GBNF compilation
lives entirely host-side (the Python DSL layer); lattice only ever receives GBNF. LNDL's own status
is "parked, revisit when stabilized" — a moving target. Lattice ships on a public registry with a
semver contract; hard-coding pre-stable DSL syntax into the Rust engine would couple lattice's
release cadence to an unstable grammar and guarantee rework. The engine already speaks GBNF, so the
translation cost of keeping LNDL out is zero. This mirrors the existing layering exactly: a domain
schema (JSON Schema) compiles to the internal representation; LNDL is another domain grammar that
compiles to GBNF, which compiles to the PDA. Lattice stays domain-agnostic.

**Fork 4 — first-increment surface + #588 relationship → build on top of #588; HTTP first, chat
tracked explicitly.** #588 is **not superseded and not closed-and-absorbed**; it becomes the
foundational **Phase 1** of this ADR (standard `json_schema` via `response_format` in
`lattice_serve`, independently valuable because it is what the OpenAI-SDK clients actually emit,
and the lowest-risk increment). The custom-`grammar` field and the interactive/unified surfaces
layer on top in later phases. The new custom-grammar field ships first on the HTTP surface
(`lattice_serve`), because the motivating caller is a programmatic HTTP client, not a human at an
interactive prompt. `chat_metal` and the unified `lattice` binary are **not silently dropped** —
they are an explicit tracked phase, wired through one shared helper so the three entry points cannot
drift apart.

**Fork 5 — caching → process-global bounded LRU, keyed on grammar text + tokenizer identity,
serialized compiles, and this is a performance optimization, not a security control.** Cache key =
`hash(syntax ‖ definition ‖ tokenizer_hash)`. The tokenizer hash is mandatory: a `GrammarEngine` is
compiled against a specific vocabulary, so a cached engine is invalid if the served model/tokenizer
changes (cheap insurance today on single-model serving, correctness-critical the moment multi-model
lands). Scope is process-global (per-connection caching would defeat cross-request reuse, the whole
point). Eviction is LRU with a hard entry-count cap and a total-resident-bytes cap (each engine's
bitmask table can reach ~15 MB at high state counts per ADR-046 R1; the engine's existing
`MAX_GRAMMAR_STATES` bounds a single entry). Crucially — see Refutation R3 — caching only protects
the cooperative repeated-grammar case; a hostile or high-variety caller sending a distinct grammar
per request defeats it, so caching is paired with mandatory compile rate-limiting and pre-partition
complexity caps rather than shipped alone.

**Fork 6 — security posture → treat the new entry point as a fresh, more adversarial trust boundary;
re-verify, do not inherit.** #343 (closed) hardened `GrammarEngine::new` against three DoS paths for
untrusted grammar text, but that work exercised primarily the JSON-Schema front-end. Accepting raw
GBNF from arbitrary HTTP callers is a materially more adversarial surface and routes through a
**different** front-end (`parse_gbnf`, not the schema `compile` path), which may have its own
pathological-input behavior #343 never touched. This ADR requires (a) an explicit re-verification
pass of the three #343 paths reached specifically via the GBNF wire path, (b) complexity caps
(definition byte length, rule count, nesting/recursion depth, repetition bounds) enforced at
**parse** time in O(text), before the O(vocab x states) partition, (c) fail-closed 400 on any cap
breach with no partial mask, (d) serialized + rate-limited compilation, and (e) a hostile-input test
corpus as a ship gate. These are ship-blockers for the grammar field, not follow-ups.

### Wire contract (concrete)

Request — additive fields on the chat-completions body (unknown fields already ignored by serde, so
older/standard clients are unaffected):

```jsonc
{
  "model": "qwen3.5-0.8b-q4",
  "messages": [ { "role": "user", "content": "..." } ],

  // Channel 1 — OpenAI-standard, unchanged passthrough (Phase 1 / issue #588).
  "response_format": {
    "type": "json_schema",                 // "text" | "json_object" | "json_schema"
    "json_schema": {
      "name": "person",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": { "name": { "type": "string" }, "age": { "type": "integer" } },
        "required": ["name", "age"],
        "additionalProperties": false
      }
    }
  },

  // Channel 2 — lattice extension (Phase 2). MUTUALLY EXCLUSIVE with a constraining
  // response_format above. A standard OpenAI client never sends this field.
  "grammar": {
    "syntax": "gbnf",                      // accepted today: "gbnf" | "json_schema".
                                           // any other value -> 400 unsupported_grammar_syntax.
    "definition": "root ::= \"<lvar \" name \">\" body \"</lvar>\"\nname ::= [a-zA-Z_][a-zA-Z0-9_]*\nbody ::= [^<]*"
  }
}
```

Resolution rules:

- `response_format` absent or `{ "type": "text" }` and `grammar` absent → unconstrained decode
  (today's behavior).
- `response_format.type == "json_object"` → generic JSON grammar. `== "json_schema"` → compile the
  nested `schema` to `GrammarSpec::JsonSchema`. Unknown `type` → 400 `unsupported_response_format`.
- `grammar` present, `syntax == "gbnf"` → `GrammarSpec::Gbnf(definition)`. `syntax == "json_schema"`
  → `GrammarSpec::JsonSchema(definition-as-value)`. Any other `syntax` → 400
  `unsupported_grammar_syntax`.
- `grammar` present **and** `response_format` also constrains (`json_object`/`json_schema`) → 400
  `conflicting_constraints` (single logit mask; two constraints would fight — follows the llama.cpp
  "either json_schema or grammar, not both" precedent).

Response — **unchanged**. No new body fields. Grammar-valid tokens appear in the normal
`choices[].message.content` (non-streaming) and `delta.content` (streaming). Rationale: forking the
response shape would break OpenAI response parsers for zero benefit; the payoff is already that the
content is now syntactically valid. An optional non-body `X-Lattice-Grammar-Applied: true` response
header may signal that a constraint was active, without touching the JSON a client deserializes.

Error taxonomy (all fail-closed; `type: "invalid_request_error"` unless noted):

| Code | HTTP | When |
| ---- | ---- | ---- |
| `unsupported_response_format` | 400 | `response_format.type` not in the accepted set |
| `unsupported_grammar_syntax` | 400 | `grammar.syntax` not currently compilable |
| `conflicting_constraints` | 400 | both `grammar` and a constraining `response_format` present |
| `grammar_parse_error` | 400 | malformed GBNF / invalid JSON-Schema |
| `grammar_too_complex` | 400 | exceeds a parse-time complexity cap |
| `grammar_compile_rate_exceeded` | 429 | too many distinct-grammar compiles in the window |

## Refutation (the assumptions, attacked)

The task is to break these before recommending, not to pick what sounds standard. Work shown.

**R1 — Is OpenAI wire-compatibility even the right target?** Attacked hard. LNDL and future formal
grammars are not consumed by OpenAI-SDK clients at all, so "chase OpenAI parity" looked like the
wrong frame. What survived the attack is a **split of concerns, not a single frame**: the reason
lattice's OpenAI surface exists (ADR-063 D10) is drop-in adoption by Aider/Continue/LangChain, and
those clients emit standard `response_format` and nothing else — so for them, OpenAI-compat is
exactly right and must be strict. Custom grammars are a **different audience** who do not want
OpenAI shape at all. The refutation therefore did not overturn OpenAI-compat; it **dissolved the
false dichotomy**: keep `response_format` a narrow OpenAI passthrough (the refute's own preferred
posture) for the JSON case, and give custom grammars a lattice-owned sibling field that is not
pretending to be OpenAI. This is precisely the adopted design. The one thing that did **not** survive
is the premise that a single wire convention should serve both classes — that is why (a) and (c) are
rejected.

**R2 — Does "LNDL is GBNF-expressible" generalize?** Tried to find a realistic future grammar that
breaks "just compile to GBNF." Found one genuine theoretical breaker and showed why it does not
change the decision:

- *Context-sensitive constraints* — e.g. a closing tag that must echo the opening tag's name over an
  unbounded identifier alphabet (`<lvar foo>...</lvar foo>` where the closer must equal `foo`). This
  is the classic copy-language / matched-name case. A PDA can balance brackets but cannot copy an
  arbitrary-length name pulled from an open alphabet — that is context-sensitive, and **no
  context-free masking engine expresses it** (this is a limit of the whole technique, shared by every
  GBNF/xgrammar-class masker, not a lattice defect). But: (i) LNDL Layer-1 as specified uses fixed
  closers (`</lvar>`, `</lact>`), so it does **not** hit this; (ii) the universal industry practice —
  and the corroborating corpus guidance — is that grammar-constrained decoding enforces **syntax**,
  while cross-reference and semantic constraints (name-matching, type-checking, define-before-use) are
  enforced by the **host after generation**. So this breaker is real but out of scope for any masking
  engine, and is correctly handled host-side, not by widening lattice's wire.
- *Ambiguity requiring Earley/GLR* — irrelevant to masking. Constrained decoding never needs to fully
  parse; it only asks "does any legal continuation exist?", which the PDA-with-stack answers. There is
  a real engine limitation here (the backtracker does not rewind consumed bytes, causing some
  shared-prefix `anyOf` over-accept/over-reject), but that is an **engine correctness** matter tracked
  under #310/#322 and explicitly out of scope for this wire ADR.
- *Unbounded/lazy repetition terminals* — a documented failure class for grammar maskers generally
  (OpenAI's own CFG feature documents it). Lattice should document the same caveat; it is not a reason
  to change the wire.

Verdict: the assumption **survives for context-free syntax**, which covers every realistic
tag-structured DSL including LNDL Layer-1. The honest boundary — stated in the ADR rather than
hidden — is that context-sensitive/semantic constraints belong host-side and are not something
lattice (or any masking engine) should try to encode in GBNF, especially since encoding type
constraints into GBNF is known to blow up grammar size exponentially.

**R3 — Is per-grammar-text caching actually safe?** Attacked by modeling a hostile / high-variety
caller. The cache is keyed on grammar text, so it protects the case where the **same** grammar
recurs. Now send a **distinct** grammar per request: every request is a cache miss, forcing a fresh
50–200 ms compile plus up to ~15 MB of resident bitmask. Even at a modest distinct-grammar arrival
rate, compile CPU-time per wall-second exceeds one (a handful of requests per second each costing
~0.1–0.2 s of compile saturates a single worker), and the LRU thrashes — every entry evicted before
reuse. Result: sustained CPU saturation and latency collapse for **all** clients, cooperative ones
included. So the assumption **fails**: caching alone is not a DoS defense. This is a real vector the
ADR designs against up front, not a deferral. The design consequence: caching is declared a
**performance optimization for the cooperative case only**, and security against the distinct-flood
comes from three mandatory controls on the shared compile path — (1) parse-time complexity caps that
reject oversized grammars in O(text) before the O(vocab x states) partition, (2) **serialized
compilation** (at most one compile in flight, so a flood queues instead of fanning out CPU), and
(3) a **per-connection / per-key compile rate limit**. Note this exposure is identical for #588's
`json_schema` path — a hostile schema flood is the same attack — so these controls attach to the
**shared** compile helper and retroactively harden #588, whose acceptance criteria call for caching
but not rate-limiting.

**R4 — One ADR or two?** Pushed on whether "wire contract for custom grammars" (lattice-owned) and
"LNDL-to-GBNF compilation" (host-owned) are cleanly separable. They are, and the split is **not
forced** — it falls straight out of Fork 3. This ADR owns exactly the lattice boundary: accept a
grammar over the wire, cache it, harden it, mask correctly. The LNDL compiler is a different owner in
a different repo against a still-parked language; folding it in here would put a moving,
externally-owned target inside a semver-committed Rust crate's ADR. **Recommendation: split.** This
ADR (lattice) defines the GBNF wire field and says nothing about LNDL internals beyond naming it the
motivating consumer. A separate host-side design doc owns LNDL-to-GBNF and targets `grammar.definition`
as its output. The interface contract between the two is precisely the wire field defined here — a
thin, stable seam.

## Alternatives Considered

| Alternative | Pros | Cons | Why Not |
| ----------- | ---- | ---- | ------- |
| (a) `response_format.type = "grammar"` (Fireworks) | Minimal surface; one field | Pollutes an enum OpenAI clients validate; conflates clean-passthrough with lattice extension | Rejected — risks breaking strict OpenAI clients; muddies the two audiences |
| (c) OpenAI Custom-Tool-CFG (Lark, tool-scoped) | Forward-compat with OpenAI SDK tool call sites | Wrong semantic scope (one argument, not the message); Chat-Completions support unverified; needs a Lark front-end lattice lacks | Rejected — semantic mismatch + engine work explicitly out of scope |
| Put custom grammars inside `response_format.grammar` sub-object | Keeps one top-level field | Still couples the lattice extension to the OpenAI object; a client that echoes `response_format` back or validates it can choke | Rejected — sibling field keeps the passthrough truly pure |
| One ADR owning LNDL-to-GBNF too | Single narrative | Puts an unstable, externally-owned grammar inside a semver crate ADR | Rejected — see R4; split to a host-side doc |
| Hard-code LNDL syntax in the Rust engine | No host compiler needed | Couples lattice releases to a parked DSL; guaranteed rework | Rejected — see Fork 3 |
| Caching only (no rate limit), per #588 as written | Simple | Defeated by distinct-grammar flood (R3) | Rejected — pair cache with rate-limit + parse-time caps |
| Add response-body grammar metadata | Observability | Forks the response shape; breaks OpenAI response parsers | Rejected — use an optional response header instead |

## Relationship to issue #588

**Build on top of.** #588 is neither superseded nor closed-and-absorbed. It is the foundational
**Phase 1** of this ADR: `response_format.json_schema` wired into `lattice_serve` with caching and
fail-closed 400s, shipping independently because it is what OpenAI-SDK clients emit and is the
lowest-risk, highest-adoption increment. This ADR extends #588 in three ways: it routes #588's
compile through the shared hardened cache helper (adding the rate-limiting and parse-time caps that
#588's criteria omit), it adds the sibling `grammar` field for non-JSON grammars, and it extends
coverage to the interactive and unified surfaces. #588 stays open and ships first; the later phases
depend on it.

## Consequences

### Positive

- Standard OpenAI clients get `json_schema` structured output and are never broken; the extension
  field is invisible to them.
- First-party callers can plug in any context-free grammar as GBNF — including LNDL Layer-1 — with
  zero grammar-format translation, since GBNF is the engine's native format.
- The engine's correctness work (ADR-046) becomes reachable from real callers for the first time.
- One shared compile-cache-and-harden helper means the security posture is defined once and cannot
  drift across the three entry points.

### Negative

- Lattice carries a non-OpenAI extension field (`grammar`), a small permanent surface-area and
  documentation cost. Mitigated: it is optional, namespaced clearly as an extension, and matches the
  dominant OSS convention.
- The `syntax` enum advertises exactly two compilable dialects; callers wanting Lark/regex are turned
  away until a future compiler lands.

### Risks

- **DoS via distinct-grammar flood** (R3): mitigated by serialized + rate-limited compilation and
  parse-time complexity caps; the exact rate-limit default needs a live load test (see below).
- **Security-guarantee transfer** (Fork 6): #343's hardening must be re-verified against the GBNF
  wire path specifically, not assumed. Ship gate: hostile-input corpus.
- **Engine correctness caveats**: grammar output can still over-/under-accept on some shared-prefix
  schemas (#310/#322). The feature must be documented BETA with those caveats; this ADR does not
  claim engine correctness.

## Phased implementation plan (dependency-ordered)

1. **Phase 1 — `response_format.json_schema` in `lattice_serve` (issue #588, ships first).** Parse
   the nested schema, build `GrammarSpec::JsonSchema`, obtain vocab bytes from the loaded tokenizer,
   compile, cache by schema, fail-closed 400 for unsupported `response_format` modes. Independently
   valuable; no dependency on later phases.
2. **Phase 1.5 — shared `grammar_cache` + hardening module (new issue; security prerequisite).**
   Process-global bounded LRU keyed on `hash(syntax ‖ definition ‖ tokenizer_hash)`; at most one
   concurrent compile; parse-time complexity caps (byte length, rule count, depth, repetition) before
   partition; per-connection/per-key compile rate limit; hostile-input test corpus (deeply nested,
   huge repetition, left-recursive, adversarial-unicode grammars → bounded compile + clean 400/429,
   no hang/crash/OOM). Phase 1's compile path is refactored to route through this. Re-verifies the
   three #343 paths on the GBNF front-end.
3. **Phase 2 — sibling `grammar` field in `lattice_serve` (new issue; depends on 1.5).** Add
   `grammar: { syntax, definition }` to the request struct; route `gbnf` → `GrammarSpec::Gbnf` through
   the shared cache; enforce mutual exclusion with `response_format`; wire the new error codes.
4. **Phase 3 — parity across the remaining entry points (new issue; depends on 2).** Factor the
   request → `GrammarSpec` → cached-engine resolution into one shared helper; wire it into
   `chat_metal` (`:676`, `:779`) and the unified `lattice` binary's `ChatCompletionRequest` handler
   (replacing the bare `ResponseFormat { r#type }` with the full parse); retire caller-facing
   `grammar: None`. Closes the interactive-chat gap and prevents sibling-path drift.
5. **Phase 4 — LNDL-to-GBNF compiler (out of this repo; host-side design doc).** Owned by the DSL
   layer; emits `grammar.definition`. Referenced here only as the motivating consumer.
6. **Docs (new issue).** Public API reference for the `grammar` field, the `syntax` enum, the caps,
   the BETA status, and the #310/#322 correctness caveats.

## Live experiments required (not assertable from documentation)

1. **OpenAI Custom-Tool-CFG endpoint reality.** Whether OpenAI's `{ "type": "custom", "format": {
   "type": "grammar" } }` actually works on `/v1/chat/completions` versus only the Responses API is
   contradicted across OpenAI's own docs, and a live bug report shows 400s via
   `chat.completions.create`. This ADR sidesteps it by choosing Pattern A, so it does not block us —
   but if Pattern C parity is ever requested, it must be settled with a live-key test first, not from
   docs.
2. **Actual GBNF compile cost at 248K vocab.** The 50–200 ms figure is documented for the JSON-Schema
   path. `parse_gbnf` + partition cost for GBNF at Qwen3 vocab should be **measured** (per the repo's
   measure-first rule) before finalizing cache size and rate-limit thresholds.
3. **Rate-limit default.** The distinct-grammar arrival rate at which compiles saturate the worker
   needs a live load test; ship a measured default, not a guess.
4. **#353/#355 closure status.** The packet reports these CLOSED, but the live `grammar/mod.rs` BETA
   doc still lists them as open "since v0.3.0." Reconcile the doc against actual issue state and cite
   the live correctness trackers (#310/#322, which #588 itself references) rather than asserting the
   engine is fully correct.

## References

- ADR-046 — XGrammar Structured Output Engine (the grammar engine this ADR reaches; `GrammarSpec`,
  `GrammarEngine::new`, `mask_logits`, the 50–200 ms compile note)
- ADR-063 — Serving Architecture (the CLI/HTTP surface and OpenAI-compat rationale this touches)
- Issue #588 — `feat(serve): wire response_format.json_schema (grammar) into lattice_serve`
- Issues #310, #322 — open JSON-Schema-compiler/PDA correctness trackers (BETA caveats)
- Issue #343 (closed) — DoS hardening of `GrammarEngine::new`; re-verify against the GBNF wire path
- Source: `bin/lattice_serve.rs:181-206` (`build_cfg`, `grammar: None`); `bin/lattice.rs:244-278,
  525-556` (`ChatCompletionRequest`, `ResponseFormat`, `reject_unsupported`); `bin/chat_metal.rs:676,
  779`; `grammar/spec.rs:15-32`; `grammar/engine.rs:142-169`; `grammar/mod.rs:38-69`
- Ecosystem: vLLM structured outputs, SGLang EBNF, llama.cpp GBNF `grammar` field
  (`ggml-org/llama.cpp#11847`, "either json_schema or grammar, not both"), Fireworks grammar mode,
  OpenAI Custom Tools CFG (`openai-python#2667`)
- GBNF/EBNF shared lineage: the llama.cpp GBNF spelling the engine already accepts natively
