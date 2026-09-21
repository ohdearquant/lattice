# ADR-093: Mixture routing is decided per request, and a weight change does not rebuild the adapter

**Status**: Accepted (2026-09-21)
**Date**: 2026-09-20
**Crate**: lattice-inference

## Context

ADR-091 decided where mixture weights come from: a softmax over the gate scores the router already
computes, with a floor that drops rather than damps. It did not decide when the selection is fixed,
or what applying a changed selection costs. Its context assumes adapters "resident and selectable
per request" without ruling on either question, and both are now load-bearing, because the serving
path that applies a selection exists.

That path is `ResidencyRegistry::apply` (`crates/inference/src/serve/lora_registry.rs:175-230`). On
a selection change it blends the selected adapters on the CPU, unloads the GPU adapter slot,
uploads the blended result, and republishes the index. It performs those steps in that order
unconditionally, whether the adapter set changed or only the weights over an unchanged set did.

### What that costs, measured

Measured on an Apple M4, release build with `metal-gpu,f16`, exclusive bench window held for the
whole run, against `qwen3.5-0.8b-q4`. Decode runs at 135.5 tok/s, so one token is 7.38 ms. The
`SWAP_BENCH` output of `bench_lora_mixture` times the three phases separately:

| r | k | blend µs | unload µs | upload µs | total µs | share of one token |
| - | - | -------: | --------: | --------: | -------: | -----------------: |
| 1 | 1 |     14.1 |      12.6 |      29.8 |     56.5 |              0.77% |
| 1 | 4 |     55.4 |      18.5 |      64.8 |    138.7 |               1.9% |
| 1 | 8 |    131.8 |      20.8 |     106.5 |    259.1 |               3.5% |
| 2 | 1 |     25.6 |      13.0 |      36.2 |     74.8 |               1.0% |
| 2 | 4 |    109.7 |      20.1 |     101.7 |    231.5 |               3.1% |
| 2 | 8 |    261.9 |      23.8 |     161.4 |    447.1 |               6.1% |

The bound on that table, stated because the decision below rests on it: these are 6-entry adapters,
`q_proj` on the 6 full-attention layers of this configuration. A full adapter over every layer and
every projection is many times that. Blend cost scales close to linearly in entry count across the
two entry counts measured — 569.8 µs at 12 entries against 261.9 µs at 6, at `r=2 k=8` — so a
16x-larger adapter extrapolates to roughly one token's cost per selection change. That is a
derivation from two points, not a measurement, and no decision here depends on its precision, only
on its order of magnitude.

The measurement uses the same call shape the serving path uses, including `quarot_seed: None`
(`lora_registry.rs:20`), so it does not cover the QuaRot rotation that `eval_perplexity` requests
on its upload.

### What the blend actually does with the weights

`blend_lora_layer_data` (`crates/inference/src/forward/metal_qwen35.rs:1930-1950`) builds two
things per `(layer, module)` group:

- `a_blend` is a vertical stack of the A matrices. The mixture weights do not appear in it.
- `b_blend` is a horizontal concat of the B column blocks, each block scaled by its adapter's
  effective weight: `b_blend[row * rank_total + col_offset + c] = eff_weight * b[row * r_e + c]`.

So over an unchanged adapter set, a weight change leaves A byte-identical and changes B by a
per-column-block scalar. Nothing else in the blended result depends on the weights. The GPU side
carries a single adapter-wide `scale`, not per-block scales (`MetalLoraAdapter`,
`metal_qwen35.rs:1699-1706`), which is why the weights are folded into B during the blend rather
than applied at use time.

## Decision

**1. The selection is decided once per request, at prefill, and held for the whole generation.**
The router runs when the request's prompt is available and its output — the selected adapter set
and the weights over it — is fixed for every token that request generates. A request does not
re-route mid-generation.

**2. A weight change over an unchanged adapter set must not rebuild or re-upload A.** The contract
this fixes is on cost, not on mechanism: applying new weights to a resident, unchanged adapter set
must not pay the A stack or the A upload, because neither depends on the weights. The mechanism is
open below; what is decided is that the current behaviour — a full rebuild of both matrices on any
change — is not the contract, and a caller may rely on a weight-only change being cheaper than a
set change.

Until the mechanism PR lands, this is a contract on the design and not a property of the tree: the
code in `main` today rebuilds both matrices on every change, and nothing currently fails if it
keeps doing so.

**3. An adapter-set change keeps the existing apply path, at its measured cost, and the
per-request wiring ships on that path.** The wiring for decisions 1 and 2's caller-visible
behaviour lands against `ResidencyRegistry::apply` as it stands today and does not wait on
decision 2's mechanism. That is what the measurement buys: at the largest shape measured a full
rebuild is 6.1% of one token, once per request, so per-request routing is shippable before
anything is optimised. Changing which adapters participate changes A, so it pays the full blend,
unload and upload above. That path is not being optimised, and the table above is the evidence that
it does not need to be at per-request frequency.

**4. What is rejected, by name.** Per-token routing, in which the weights move as generation
proceeds. Under the current apply path each move costs a full rebuild, which the extrapolation
above puts at roughly one token per token at full-adapter shapes — a 2x decode slowdown. It is
rejected here as out of scope rather than as impossible; decisions 1 through 3 are what a
per-request router needs, and nothing in them forecloses revisiting this behind its own evidence.

_Amended 2026-09-21, while wiring the serving path._ Decision 1 says the selection is decided once
per request at prefill. Wiring it raised three questions it does not answer, and all three are
caller-visible, so they are decided here rather than left to the code.

**A request that names its own adapters keeps them.** The router runs only when `lora` is absent.
A caller who named adapters asked for those, and a learned guess overruling a stated intent would
make the request field advisory without saying so anywhere.

**Every trained column is routed, and the gate decides the weight over them.** Not a top-k. A `k`
would need a number nothing here has evidence for, and truncating the gate's own distribution is
the opposite of letting it choose — the distribution is the thing being learned. This is also why
no flag is added: a default of 1 would make a mixture router a single-adapter router for everyone
who did not read the flag list.

**A trained set that is not resident refuses the request, naming both lists.** The gate's columns
are labelled by the artifact's adapter names, so a resident set that does not match cannot be
routed at all, and there are only two ways to answer it. Refusing is loud and has a real cost: a
server started with `--router-state` and no adapters loaded refuses every chat request until they
are. Serving the base model instead is quiet and costs more, and it is the failure this ADR family
refuses at every other level — the operator configured routing, `GET /v1/lora` reports routing
enabled, and the only place the truth appears is in the answers, where it looks like a gate that
did not learn much. The cost of the loud answer is paid once, at the moment the operator can act;
the cost of the quiet one is paid by whoever later tries to explain the output.

A conversation with no user turn is deliberately not that case: it serves the base model, because
there is no text to route on under the recorded rule rather than a routable request being skipped.
Inventing a substitute — the system prompt, the rendered history — would route on text the gate
never saw and would look like it worked.

## Alternatives considered

- **A coefficient-only fast path over the existing rebuild.** Skip the blend when only the weights
  changed, keep the unload and upload. This was the shape considered before the measurement, and
  the measurement is what rules it out: blend is 25–58% of the swap, so the fast path removes at
  most that fraction while leaving the upload — including A's, which cannot have changed — fully
  paid. Decision 2 asks for the stronger property instead.
- **Per-block scales on the GPU adapter.** Carry each adapter's weight as a scalar the kernel
  applies, so a weight change touches no buffer at all. This is the most direct route to decision 2
  and is the leading candidate for its mechanism, but it changes the decode kernel's inner loop, so
  it takes its own before/after measurement and is not decided here.
- **Re-upload only B.** Weaker than the above and weaker than decision 2 warrants, but cheap: B is
  four times A's bytes at these dimensions (`d_out` 4096, `d_in` 1024), so this saves about a fifth
  of the upload and all of the A stack.
- **Leave the apply path alone.** Defensible on the numbers for per-request routing alone, and it
  is what happens if decision 2's mechanism is never built. It is rejected because it makes the
  cost of a weight change indistinguishable from the cost of a set change, which removes the
  caller's ability to rely on the difference — and a router that adjusts weights frequently over a
  stable set is the expected shape.

## Consequences

- The router's output becomes a per-request value with a defined lifetime, so it can be logged,
  reproduced and attributed to a request rather than to a moment.
- `ResidencyRegistry::apply` gains a distinction it does not have today, between a set change and a
  weight change, and its early return on an identical selection (`lora_registry.rs:197`) is not
  that distinction — it fires only when nothing changed at all.
- Decision 2 is not checkable until something can fail it, so the PR that builds its mechanism
  carries the arm that would: a test that reddens when a weight-only change over an unchanged
  adapter set re-uploads A, by counting uploads or byte-comparing the resident A across the change.
  Without that arm the decision is a sentence in a document, and a later change that quietly
  reinstates the full rebuild passes every suite.
- The ADR-091 floor interacts with decision 2: dropping an adapter below `epsilon` removes it from
  the set, so a weight change that crosses the floor is a set change and pays the full cost. That
  is correct and worth stating, because it means a weight-only path cannot be entered on the
  weights alone; it is entered on the surviving set being identical.

## Open, and deliberately not decided here

The mechanism for decision 2 — per-block GPU scales against a B-only re-upload — and the numeric
frequency at which a weight change is expected. Both want the same measurement, taken against a
full-size adapter rather than the 6-entry one above, which is the gap the evidence section names.
