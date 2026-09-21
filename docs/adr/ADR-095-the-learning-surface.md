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
- Decision 4's route-parity test is cheap and its absence is the only reason decision 4 needs
  stating; it also retroactively covers the three existing `/v1/lora*` routes.
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
