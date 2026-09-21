# ADR-095: The learning surface — feedback endpoint, router state, and what the API reports

**Status**: Accepted (2026-09-21)
**Date**: 2026-09-20
**Crate**: lattice-inference

## Context

ADR-093 fixed when routing runs; ADR-094 fixed where the routing code lives and what feeds it.
What remains is the part a user touches: how a learning signal arrives, where the learned state
lives across restarts, and what the HTTP surface reports about either.

Drafting this against the tree falsified two premises the design was resting on. Both are recorded
here rather than worked around, because each one decides a question that was assumed answered;
decisions 5 and 6 below are what they were replaced with.

**There is no authentication on this server.** `Authorization`, `Bearer`, `api-key` and `API_KEY`
appear zero times in either serving binary (`crates/inference/src/bin/lattice/serve.rs` and
`crates/inference/src/bin/lattice_serve.rs`), against 61 `axum` occurrences in the first as a
control that the search works. The only layer on the route table is a body-size limit
(`serve.rs:1758`). `/v1/lora/load` says so about itself in its own doc comment (`serve.rs:1651`):
"This unauthenticated route reads a caller-selected path on the server host." So a feedback
endpoint cannot be scoped as "the same admin surface and auth as `/v1/lora/load`" — that surface
has no auth to inherit, and a route that mutates learned router state is a strictly stronger
capability than one that loads a named file.

**There is no adapter directory.** Adapters arrive as a caller-supplied `path` string in the
request body (`serve/lora.rs:181-195`), one at a time; there is no `--lora-dir` or equivalent
anywhere in `crates/inference/src` (control: `--embedding-model` is findable by the same search,
13 hits in the same file). So "a router artifact on disk beside the adapters" has no "beside" —
there is no directory the adapters share.

**The route table is registered twice.** `/v1/lora`, `/v1/lora/load` and `/v1/lora/unload` are
each registered in two binaries: `bin/lattice/serve.rs:1755-1757` and `bin/lattice_serve.rs:3244`
onward. Any new route added to one and not the other is present on one binary and absent on the
other, and nothing currently fails when that happens.

## Decision

**1. Feedback is an explicit endpoint, never an implicit signal, and never synchronous.** A
`POST /v1/lora/feedback` accepts an explicit preference signal against a completion the server
issued. Implicit signals — continuation length, whether a stream was cancelled — are noise
attributed to the router. The endpoint appends to a bounded replay buffer and returns; the update
itself runs on a documented cadence (per N events or per interval, both configured and both
reported by `GET`), never inside the request that delivered the event and never inside a chat
request.

Where it does run, since "not in the request" leaves it unstated: the refit is CPU work on a
dedicated single worker thread in the serving process, bounded by a configured wall-time budget
that `GET` reports, and it never holds the residency registry lock across the refit — it takes the
gate bytes, releases, computes, and reacquires only to install the result. A refit therefore cannot
stall decode, and an overrunning refit is abandoned and counted rather than allowed to run long.

The buffer is bounded and its overflow policy is part of the contract rather than an
implementation detail: a full buffer drops the oldest event and increments a counter that `GET`
reports. A silent drop makes "learning is not converging" and "events are being discarded"
indistinguishable at every later reading.

**2. `GET /v1/lora` reports residency; the routed selection is reported per request.** `GET`
answers what is loaded, what the default mixture is, whether the router is enabled, and the router
version. It does not report a per-request selection, because there is no single current one once
routing is per request. The selection a given request actually used is reported in that request's
completion response metadata, where it is attributable to the request that caused it. The existing
`AdapterIndex` (`serve/lora.rs:120-126`) gains the router fields; `applied` keeps its present
meaning of the mixture materialised in the engine.

This is the same split ADR-091's floor forces and ADR-093 names: `applied` describes engine state,
not a routing decision, and per-request routing is what pulls the two apart.

**3. Router state is versioned, persisted, and pinnable.** Learned weights that do not survive a
restart are not learning, so the router's gate is written to a versioned artifact, loaded at
startup, and reported by `GET`. The version is what makes a rollback expressible: pinning a
version is the operation an operator needs when a refit degrades output, and ADR-091's collapse
guards reject a bad refit at reload rather than after it is serving.

**4. Both binaries or neither, enforced by one route-table constructor rather than by a test.**
Every route this ADR adds is registered in both serving binaries. The first draft of this decision
enforced that with a test enumerating each binary's route table and comparing the two sets; that
test cannot be built, because axum's `Router` does not expose its route table for enumeration.

The shape that works is structural rather than assertional: one constructor owns the route table
and both binaries call it. That deletes the duplicated registration, which is the thing that could
drift, so there is no longer a difference for a test to catch — the enforcement is that only one
copy exists. What remains for a test is at most an assertion that both binaries reference that
constructor, and if the constructor has no second caller the test has nothing left to say.

This retroactively covers the three `/v1/lora*` routes that already exist and are registered
separately in each binary today: they move into the same constructor, so the rule applies to the
routes this ADR adds and to the ones it inherits, rather than creating a second convention beside
the first. The predicted failure the original test was aimed at is not subtle, it is just
invisible — a feature works in every test that exercises one binary — and removing the copy
addresses it at the cause instead of detecting it afterwards.

_Amended 2026-09-21._ The mechanism above is not writable against this tree, and the aim survives
it. "One constructor owns the route table" assumed the duplicated registration was the duplication.
It is the thinnest layer of it: the two binaries do not share a state type — `lattice/serve.rs`'s
`AppState` carries a `ModelBackend` that is CPU-safetensors or Metal, with `max_tokens` defaults
and a request counter, while `lattice_serve.rs`'s is Metal-worker-only and carries the job client,
a Prometheus registry, the admission cap and byte-decoded vocab for the grammar engine — and the
handlers are per-binary functions over those types. A shared constructor would therefore be generic
over the state and take every handler as a parameter, replacing eight `.route()` lines with eight
handler arguments. That relocates the duplication into a longer call while the handlers and the
state, which are what actually drift, stay exactly where they are.

The aim — every route in both binaries or in neither — is kept, with a mechanism that exists today:

1. **A shared route list as data, not a constructor.** One `LORA_ROUTES` const in the serving
   crate, path and methods only, one copy. Data has no state type to be generic over.
2. **A presence test per binary.** Each binary builds its own `Router` with its own state and
   drives every entry in the list through tower's `oneshot`, asserting the response is not 404.
   A 405 or a 4xx from validation both count as registered — the test asks whether the route
   exists, and nothing else, because anything more would need the state the two binaries do not
   share. A route added to the list and forgotten in one binary reds that binary's test.
3. **A lint for the other direction.** A route registered in a binary and never added to the
   list is invisible to (2), so the set of `/v1/lora` **registrations** in each binary's source
   must equal the other's and equal the list. Registrations, not path literals: the subject is
   `.route("/v1/lora…", …)`, and a `/v1/lora` string that is not one — a doc comment, a request
   fixture, a log line — is deliberately outside it. A lint over literals answers a different
   question and answers it wrongly in both directions, going red on a test fixture and staying
   clean on a binary that mentions a path it never registers. It runs with a must-match control,
   the way the existing source-marker lint does, in the same CI step — a lint whose discovery can
   silently empty reports success while checking nothing. The two directions stay separate
   mechanisms on purpose: listed-but-not-registered is (2)'s arm, reached by driving a route that
   answers 404, and registered-but-not-listed is this one's.
4. **Validation parity is an audit, not code.** (1)–(3) establish that a route is registered, which
   is not the same as its behaviour being the same. The ADR carries a table per `/v1/lora*` route
   naming where each binary's handler performs each check, re-read whenever a handler changes. That
   is the failure this decision actually saw: both binaries rejected a non-finite adapter scale,
   but one rejected it at the HTTP boundary and the other only later inside `apply()`. Same route,
   same shared module, different reachable behaviour, and no route-table mechanism would have
   caught it, because both tables registered the route correctly.

The table, as of 2026-09-21. Positions are given as ordinals rather than line numbers, because a
line number in prose is never re-derived by the people who move the code and so can only decay. Read
`lattice serve` as `bin/lattice/serve.rs` and `lattice_serve` as `bin/lattice_serve.rs`.

The ordinals are not a transcript of what the handlers happen to do. They are consequences of one
rule, so that a reader who did not write the handlers can still say whether a row is right:

> A refusal that is a property of the **build** answers before the request is read, because it is
> true of every request. A refusal that is a property of the **runtime** answers after
> Content-Type, the body cap and the parse, so a malformed request gets the malformed-request
> answer whatever the runtime state happens to be. The same order on every route, on both
> binaries.

A row that does not follow from that is either a defect or an amendment to the rule, and saying
which is the point of writing the rule above the table. A table of ordinals alone cannot be wrong,
only outdated; a table with a predicate can be checked by someone who reads only the table.

| Route                  | Check                          | `lattice serve`                                                | `lattice_serve`                                                                 |
| ---------------------- | ------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `GET /v1/lora`         | backend resolution             | `adapter_client` in the handler; a build without Metal refuses | none — this binary's state is Metal-worker-only, so there is nothing to resolve |
| `GET /v1/lora`         | response body                  | shared `serve::lora::lora_list_body`                           | shared `serve::lora::lora_list_body`                                            |
| `POST /v1/lora/load`   | build refusal (no Metal built) | 0th — before anything is read                                  | n/a — this binary is Metal-only                                                 |
| `POST /v1/lora/load`   | Content-Type is JSON (415)     | 1st                                                            | 1st                                                                             |
| `POST /v1/lora/load`   | body size cap (413)            | 2nd                                                            | 2nd                                                                             |
| `POST /v1/lora/load`   | JSON parse and required fields | 3rd, `parse_lora_load`                                         | 3rd, `parse_lora_load`                                                          |
| `POST /v1/lora/load`   | backend resolution             | 4th, `adapter_client`                                          | n/a                                                                             |
| `POST /v1/lora/load`   | adapter path and name          | 5th, `prepare_adapter_load`                                    | 4th, `prepare_adapter_load`                                                     |
| `POST /v1/lora/load`   | telemetry on every outcome     | not emitted                                                    | `emit_serve_event`, including on each refusal                                   |
| `POST /v1/lora/unload` | build refusal (no Metal built) | 0th — before anything is read                                  | n/a — this binary is Metal-only                                                 |
| `POST /v1/lora/unload` | Content-Type is JSON (415)     | 1st                                                            | 1st                                                                             |
| `POST /v1/lora/unload` | body size cap (413)            | 2nd                                                            | 2nd                                                                             |
| `POST /v1/lora/unload` | JSON parse and required fields | 3rd, `parse_lora_unload`                                       | 3rd, `parse_lora_unload`                                                        |
| `POST /v1/lora/unload` | backend resolution             | 4th, `adapter_client`                                          | n/a                                                                             |
| `POST /v1/lora/unload` | telemetry on every outcome     | not emitted                                                    | `emit_serve_event`, including on each refusal                                   |

Two rows are divergences rather than descriptions, and writing the table is what surfaced them.

The `GET /v1/lora` body row was one. `lattice_serve` returned the bare residency snapshot after
`lattice serve` gained the `router` key, so one server reported which gate was serving and the other
did not. Both registered the route; the presence test and the parity lint both passed. Fixed by
moving the body into the shared module, which is the general form: a body assembled inside one
binary is a body the other can drift from.

The unload row was the other, and the rule above is what closed it. On `lattice serve`,
`lora_unload` used to resolve the adapter backend before Content-Type, the cap and the parse, while
`lora_load` on the same binary resolved it after, so one malformed request and an identical
malformed request answered with different statuses depending on which route received them. The
defence written in the code carried only for the build with no Metal compiled in — there the
refusal really is true of every request, bodyless or not — and it was applied to the Metal build,
where the same call is a worker lookup and says nothing about the binary. Splitting the two cases
is what produced the rule, and the rule then put the backend lookup after the parse on both routes.

The arms are worth naming, because a rule about ordering is invisible to any test that sends a
well-formed request. On the compiled-out build, a body that is malformed twice over — no JSON
content type and an unknown field — still receives the build's answer; parsing before that refusal
reds it. On the Metal build, an unknown field on either route receives the caller's answer on a
state whose backend cannot serve adapters at all; hoisting the backend lookup above the parse reds
it, and reds nothing else. Two of the pre-existing arms asserted the old order and were gated to
the build that can still express them, rather than deleted: a request-contract claim is a claim
about a build that serves the route.

No state-unification lane follows from this. It would be large and nothing here depends on it.

The retroactive half is unchanged in scope: the three existing `/v1/lora*` routes are covered on
these terms rather than moved into a constructor.

**5. The mutating routes are loopback-only by default; the read routes keep today's posture.**
The server's existing posture is local-first — `--host` defaults to `127.0.0.1`
(`bin/lattice_serve.rs:3436`) and startup already warns on a non-loopback bind (`:3442`). This
extends that posture rather than inventing a second one.

`POST /v1/lora/load`, `POST /v1/lora/unload`, `POST /v1/lora/feedback` and any router pin or
rollback route refuse with `403` on a non-loopback `--host`, with a body naming the flag that
would permit them. `--admin-token` permits them, and when it is set those routes require
`Authorization: Bearer <token>`. The token's value comes from an environment variable named by the
flag, never from argv, because argv is readable by every process on the host. `GET /v1/lora`,
completions and embeddings keep the posture they have today. Startup logs the resulting posture on
one line, so which routes are reachable is a fact in the log rather than a derivation from flags.

Shipping the feedback route unauthenticated alongside its neighbours is rejected: a mutation of
learned state on an open port is a different thing from a read, and the neighbours' posture is the
thing being narrowed here rather than a precedent to follow.

**6. `--router-state <path>` anchors the artifact, it is never caller-supplied, and there is no
derived default.** Enabling learning without `--router-state` refuses at startup, naming the flag.
Reads keep working without it: `GET /v1/lora` and serving against a pinned or absent gate do not
require the flag, because they do not write.

The first draft derived the path from the model directory as `<model_dir>/router/`, on the grounds
that `--model` always exists (`bin/lattice_serve.rs:3424`). That default is wrong-shaped, and the
reason generalises past this flag. Router state is a property of the triple (base model, adapter
set, gate), not of the base model alone. So two servers on one base model with different adapter
sets derive the _same_ directory, and each overwrites the other's learned state with the last
writer winning and nothing in either process able to notice. A model directory can also be
read-only or shared between users, in which case the derived path fails at the first refit rather
than at startup — the failure arrives a long way from the flag that caused it, after the server has
been accepting feedback and reporting that learning is on.

A refusal at startup naming the flag is the same shape ADR-094 decision 1 uses for a missing
embedding model, and for the same reason: it makes "learning is configured" and "learning can
persist" the same question, answered once, at the moment the operator can still act on the answer.

The artifact carries a version, a monotonic counter plus a content hash, and the previous version
is kept beside it, so a rollback is a rename rather than a refit. Pinning is `--router-pin
<version>` at startup. A pin route on the mutating surface is available under decision 5's
boundary and is deliberately not added here: startup pinning is what an operator needs during an
incident, and it is the arm that works when the process is the thing misbehaving.

### Amended 2026-09-21, re-reading decisions 5 and 6 as a threat model rather than a posture

Decisions 5 and 6 above describe where the routes listen and where the state lives. Neither says
what an attacker gets, and a review of the accepted set found four places where the answer is worse
than the text implies. They are recorded as amendments rather than rewrites because the decisions
hold; what was missing is the boundary each one assumes.

**Loopback is an assumption about the host, and it needs saying out loud.** Decision 5 leaves
`POST /v1/lora/feedback` unauthenticated on a loopback bind, and ADR-096 lets those events drive a
refit and a live gate activation. So on a shared or multi-tenant host, any local process can shape
the served policy with no credential at all. That is acceptable only under a single-user trust
boundary, which this project does assume elsewhere, and the cost of leaving it implicit is that the
assumption is invisible at exactly the deployment where it stops being true. So it is stated: the
loopback posture asserts that every local process is as trusted as the operator. Two cheap
narrowings follow from stating it, and both are in scope for the feedback PR rather than deferred.
Feedback carries the completion id it is about, and the server rejects an id it never issued, so a
process cannot vote on requests it did not make. And the route takes a fixed per-interval cap,
refusing over it, because an unbounded write path into learned state is a denial of service against
the policy itself even from a trusted caller with a bug.

**A bearer token over plain HTTP is a credential broadcast once per request.** Decision 5 permits
non-loopback mutation when `--admin-token` is set, and the serving binary speaks plain HTTP. An
on-path observer therefore captures the token from the first admin request and can load, unload,
pin or poison from then on. The token does not make the route safe on an open port; it makes it
auditable on a network that was already trusted. So the non-loopback mutating surface requires an
explicitly declared TLS-terminating proxy — a flag that says the operator has put one in front —
and refuses otherwise even when a token is set. Refusing is the right direction because the failure
it prevents is silent: a captured token produces no error anywhere, and the first symptom is a
policy that learned something nobody sent.

**The router-state directory needs the file protections the model path already has.** Decision 6
names an operator-selected path, head records, renames and persistence, and says nothing about what
happens when a local principal can write that directory. Replacing the artifact, or redirecting the
head record through a symlink, activates an untrusted gate on the next start with no signal. The
mechanism is not new work: `quant/q4_manifest.rs` already reads manifests with
`fs::symlink_metadata`, refuses to follow a symlink at the named path, and fails closed on a
dangling target or a permission error rather than reporting absence. Router state takes the same
treatment — no-follow open, refusal when the state directory is writable by a principal other than
the server's own user, and a durable atomic commit for the head record so a crash cannot leave a
half-written pointer that reads as valid. Fail closed on each, naming the path.

**A pin with no `--router-state` has no artifact to pin.** Decision 6 says reads do not require the
flag and that a pinned server serves the pinned gate, and those two sentences together describe an
unreachable configuration: `--router-pin <version>` names a version within a state directory, and
without the directory there is nothing to resolve the version against. The resolution is that a pin
is a read of router state, not an alternative to it. `--router-pin` requires `--router-state` and
refuses at startup naming the flag, in the same shape as every other refusal here. What does not
require the flag is serving with no gate at all, which is the case decision 6 meant: a server
without `--router-state` reports `"enabled": false` and routes nothing. Read-only differs from
absent, and conflating them is what produced a pin with no locator.

**Implicit preference variants are rejected at the wire, not translated.** This decision says the
feedback wire shape follows `PreferenceSignal`, whose current definition
(`crates/tune/src/lora/router_update.rs:34`, moving to `lattice-fann` under ADR-094 decision 3)
carries `ImplicitPositive` and `ImplicitNegative` alongside the explicit pair, at half reward
magnitude. ADR-096 admits only explicit feedback about a completed request. Following the enum
shape therefore accepts two variants the admission policy does not want, and nothing said which
of reject, ignore or translate applies — three behaviours that are indistinguishable to the sender
and produce three different training sets. They are rejected with a 400 naming the variant. The
sender learns its signal was not taken, which the ignore branch does not provide and which matters
precisely because implicit signals would otherwise be dropped silently for the whole life of a
deployment. Sharing a type with a trainer is not the same as sharing its admission policy, and the
wire shape is the place to say so.

### Ordering note: what this ADR describes and when it lands

Two rows in decision 4's table, and the `GET /v1/lora` body row, describe the tree after this
routing chain lands, not the tree this ADR merges into. At this ADR's own head, `lora_unload` in
`bin/lattice/serve.rs` still resolves the adapter backend before the content-type check, the body
cap and the parse, and `lora_list` still serializes the residency snapshot directly. Both changes
are made by the router-artifact PR later in this chain. The table is the contract the chain is
built to satisfy, and a reader checking it against this merge base will find the old order; that
is expected rather than a contradiction, and saying so here is cheaper than leaving the next
reader to discover it by grep.

### Two carriers this touches, found while writing it

`bin/lattice_serve.rs` documents `POST /v1/lora/load` — including the whole deployment-boundary
paragraph about caller-selected paths — in a doc comment attached to **`async fn lora_list`**
(`:3251-3265`), the GET route. The real `lora_load` is at `:3272` under a shorter comment that says
nothing about the boundary. The other binary has it correctly on `lora_load`
(`bin/lattice/serve.rs:1651`). So the rewrite decision 5 requires cannot be keyed on finding the
comment: done that way it edits the read route's documentation and leaves the mutating route
undocumented. It is also decision 4's failure mode already present in the tree, one binary having
drifted from the other with nothing failing.

The startup warning's text (`:3444-3448`) says "`/metrics` and every other route are
unauthenticated". Decision 5 makes that false for the mutating set, so the warning is rewritten in
the same change. A security notice that overstates exposure trains its reader to discount it.

## Alternatives considered

- **Learn from implicit signals.** Rejected: the signal is attributed to the router but produced by
  everything else in the request.
- **Offline-only updates from an exported buffer.** Not wrong, and it remains available, but it is
  not dynamic in the sense asked for; the endpoint plus a cadence is the smaller thing that is.
- **Report the routed selection in `GET`.** Rejected: with per-request routing there is no single
  current selection, so the field would report whichever request happened to be last.
- **Keep router state in memory only.** Rejected by decision 3's first sentence: the feedback
  endpoint would learn into a process that forgets on restart, which reads as learning that does
  not work rather than as state that was never saved.

## Consequences

- `docs/serve-http-api.md` changes in the same PR as the endpoint, because this is a public product
  surface and its shape is the contract.
- Decision 4's enforcement is the shared route-table constructor, not a test. This bullet used to
  call it a route-parity test, which is the design decision 4 rejects two paragraphs into itself,
  on the ground that axum's `Router` does not expose its table for enumeration. A superseded
  design left standing in the consequences list is a reader's shortest path to the wrong answer,
  since a consequences bullet is read as settled. The constructor retroactively covers the three
  existing `/v1/lora*` routes.
- Decision 1's cadence configuration and buffer counters are reported by `GET`, which means
  ADR-094's gate-free façade has to answer them in a build without the `mixture` feature too —
  as a disabled router with no version rather than as absent fields.

## Open, and deliberately not decided here

The numeric defaults for the cadence and the buffer bound, and the wire shape of the preference
signal, which should follow `lattice-fann`'s `PreferenceSignal` rather than inventing a second
vocabulary for the same thing.

---

Added at sign-off, 2026-09-21. Decisions 4 and 6 were rewritten as a condition of sign-off, and
decision 1 gained its "where it runs" paragraph. The original decision 4 specified a route-table
parity test that axum's `Router` cannot support, and the original decision 6 derived a default
state path from the model directory. Both earlier forms are described in place above rather than
deleted, because a decision that was reversed is evidence about the shape of the problem.
