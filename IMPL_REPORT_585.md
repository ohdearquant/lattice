# Implementation report: OpenAI-compatible `logprobs` / `top_logprobs` (#585)

## Summary

Adds OpenAI Chat Completions-compatible `logprobs` / `top_logprobs` support to the
`lattice` serve binary's non-streaming response path, on both the CPU and Metal
forward paths. When requested, each generated token's log-probability (and, if
`top_logprobs: N` is set, the top-N alternative tokens at that position) is
threaded through the generation pipeline into `GenerateOutput` and rendered into
the OpenAI `choices[].logprobs.content[]` shape. When not requested, the decode
loop does no extra work: no full-vocabulary softmax, no allocation, no retained
logit arrays.

## Request/response shape

Request fields (new, both optional):

```json
{ "logprobs": true, "top_logprobs": 3 }
```

- `top_logprobs` requires `logprobs: true`; the reverse is not required (`logprobs: true`
  alone is valid and returns per-token logprobs with an empty alternatives list).
- `top_logprobs` must be in `0..=20`, matching OpenAI's documented range.

Response shape (new, on `choices[].logprobs`, omitted from the JSON entirely when
`logprobs` was not requested):

```json
{
  "content": [
    {
      "token": "Hello",
      "logprob": -0.31,
      "bytes": [72, 101, 108, 108, 111],
      "top_logprobs": [
        { "token": "Hello", "logprob": -0.31, "bytes": [72, 101, 108, 108, 111] },
        { "token": "Hi", "logprob": -2.10, "bytes": [72, 105] }
      ]
    }
  ]
}
```

`bytes` is the raw UTF-8 byte decoding of the token via the tokenizer's existing
GPT-2 byte-to-unicode table; it is `null` when a token ID can't be resolved
against the vocabulary (fail-closed, see "Unresolved tokens" below).

## Where the logic lives

- `crates/inference/src/sampling.rs` — `compute_step_logprobs` (temperature-scaled
  log-softmax over the full vocabulary + top-k selection, reusing the existing
  descending-logit/NaN-last/lowest-token-id-wins total order used by the Metal-parity
  top-k path) and `record_logprob` (the opt-in entry point called from the decode loop).
- `crates/inference/src/model/qwen35_config.rs` — `TokenLogprob` / `TopLogprob` structs,
  `GenerateConfig.logprobs: Option<usize>` (the caller's `top_logprobs` count; `None`
  disables capture), `GenerateOutput.token_logprobs: Vec<TokenLogprob>`.
- `crates/inference/src/bin/lattice.rs` — the OpenAI wire format: `ChatCompletionRequest.logprobs`/`top_logprobs`,
  `validate_logprobs` (range + `top_logprobs`-without-`logprobs` checks), `render_token_logprob`/`build_choice_logprobs`
  (turn `Vec<TokenLogprob>` + the tokenizer into the response shape above), `Choice.logprobs: Option<ChoiceLogprobs>`.
- `crates/inference/src/tokenizer/bpe.rs` — extracted the existing lossless byte-decode
  logic into `byte_decode_token_bytes(token_str: &str) -> Vec<u8>` (the existing
  `byte_decode_token` now delegates to it) so `render_token_logprob` can produce the
  `bytes` field without duplicating the byte-to-unicode table.

## Forward paths

Both paths the `serve`/`chat` binary actually dispatches through were updated:

- **CPU** — `Qwen35Model::generate` / `generate_streaming` (`crates/inference/src/model/qwen35/generation.rs`):
  `record_logprob` is called at every decode-step call site (6 total, covering the
  plain/streaming/thinking-budget/LoRA variants) right after the sampled token ID
  is resolved, passing the step's full logit slice.
- **Metal** — `MetalQwen35State::generate_streaming` (`crates/inference/src/forward/metal_qwen35.rs`):
  the same `record_logprob` call, wired into the one function `lattice.rs`'s HTTP
  handler actually calls for Metal requests (`generate_streaming` is used for
  *both* streaming and non-streaming Metal HTTP responses, via a no-op callback
  for non-streaming requests). This is the only Metal entry point the serve
  binary reaches.

Other `MetalQwen35State` generation entry points (`generate_with_lora_mixture`,
plain `generate()` — used by the chat REPL binary and internal benchmarking/golden
harnesses, not `serve` — and `generate_streaming_with_prefix_cache*`) received only
the mechanical `token_logprobs: vec![]` field so `GenerateOutput` construction keeps
compiling; they do not compute logprobs and are not reachable from the HTTP surface,
so no rejection guard was added for them. `crates/inference/src/bin/lattice_serve.rs`
(a separate, minimal OpenAI-compatible HTTP binary) has no `logprobs` fields on its
own `ChatReq` at all — out of scope for this change, and received only the same
mechanical `logprobs: None` field addition to keep `build_cfg` compiling.

## Scope boundary: streaming + logprobs

`stream: true` combined with `logprobs: true` is rejected with HTTP 400 via
`reject_unsupported`, same as the other currently-unsupported combinations. Logprobs
are fully implemented on the non-streaming response path only. Streaming + logprobs
would require threading per-chunk logprob data through the SSE delta format, which
is a separate, larger design surface than this issue's scope.

`reject_unsupported` continues to reject `tools`, `tool_choice`, `n > 1`, and any
`response_format` other than `"text"` — unchanged from before this change.

## Hard constraint: zero cost when not requested

`record_logprob`'s first line is an early return:

```rust
pub(crate) fn record_logprob(
    token_logprobs: &mut Vec<TokenLogprob>,
    logits: &[f32],
    token_id: u32,
    temperature: f32,
    logprobs_requested: Option<usize>,
) {
    let Some(top_n) = logprobs_requested else {
        return;
    };
    let (logprob, top) = compute_step_logprobs(logits, token_id, temperature, top_n);
    token_logprobs.push(TokenLogprob { token_id, logprob, top });
}
```

Every call site in the decode loop calls this unconditionally (rather than guarding
the call itself), but when `logprobs_requested` is `None` — the default — it does
nothing: no call into `compute_step_logprobs` (the full-vocabulary softmax + top-k
pass), no allocation, no push. `logits` is passed as a slice into the already-computed
per-step logit buffer, not a copy. This is covered by
`test_record_logprob_noop_when_not_requested` (asserts the output vec stays empty)
and `test_record_logprob_appends_entry_when_requested` in `sampling.rs`.

## Correctness testing

**(a) No-`logprobs` request is behaviorally unchanged.** `Choice.logprobs` is
`#[serde(skip_serializing_if = "Option::is_none")]`; `choice_logprobs_omitted_from_json_when_none`
(new, `lattice.rs`) asserts the serialized JSON contains no `"logprobs"` substring at
all when the field is `None`. Combined with the no-op `record_logprob` path above,
the default response is unchanged both in computation and in wire shape.

**(b) `logprobs: true` matches a directly-computed softmax reference.**
`test_compute_step_logprobs_matches_hand_computed_softmax` (`sampling.rs`) computes
a hand-rolled log-softmax reference for a known 3-logit vector and compares
`compute_step_logprobs`'s output against it for every token index, within `1e-4`.
`build_choice_logprobs_shapes_content_and_alternatives` (new, `lattice.rs`) verifies
the same data is correctly carried through into the response-shape structs (token
string resolution, `logprob` value, `bytes`, nested `top_logprobs`) using a minimal
in-memory 2-token tokenizer built via `BpeTokenizer::from_vocab_and_merges`.

**(c) `top_logprobs: N` returns exactly N alternatives, sorted by probability.**
`test_compute_step_logprobs_top_n_sorted_descending_by_probability` and
`test_compute_step_logprobs_top_n_clamped_to_vocab_size` (`sampling.rs`) cover the
count and ordering; `validate_logprobs_top_logprobs_at_boundary_twenty_ok` /
`validate_logprobs_top_logprobs_over_twenty_rejected` (new, `lattice.rs`) cover the
`0..=20` boundary validation, and
`validate_logprobs_top_logprobs_without_logprobs_true_rejected` /
`validate_logprobs_top_logprobs_with_logprobs_false_rejected` cover the
`top_logprobs`-without-`logprobs` rejection.

13 new unit tests were added to `lattice.rs` this change (`validate_logprobs` x8,
`render_token_logprob`/`build_choice_logprobs` x3, `Choice.logprobs` JSON-shape x2),
alongside the pre-existing `sampling.rs` coverage of the computational core listed
above.

### Unresolved tokens

`render_token_logprob` fails closed on an out-of-vocabulary token ID: it emits a
placeholder token string (`<|unresolved_token_N|>`) and `bytes: null` rather than
panicking or fabricating a byte decoding. Covered by
`render_token_logprob_unresolved_id_fails_closed`.

### Test results

```
cargo test -p lattice-inference --bin lattice                          77 passed; 0 failed
cargo test -p lattice-inference --bin lattice --features "metal-gpu f16"  76 passed; 0 failed
```

(The one fewer test under `metal-gpu f16` is a pre-existing, unrelated test —
`metal_gpu_required_message_mentions_rebuild_flags` checks the "please rebuild with
metal-gpu" error message, which naturally doesn't apply once metal-gpu is already
compiled in. Not a regression from this change.)

```
cargo test -p lattice-inference     (default features, full crate)
  --lib: 1435 passed; 0 failed; 7 ignored (pre-existing)
  --bin lattice: 77 passed; 0 failed
  + all integration test binaries and doc-tests: passed
```

`cargo clippy --workspace --all-targets -- -D warnings`: clean (default features).
`cargo clippy -p lattice-inference --features "metal-gpu f16" --all-targets -- -D warnings`: clean.

## Performance evidence

`make bench-compare` (quick mode, `origin/main` @ `a00d7ebcd` vs this branch @ `4ad5d7cf3`):

- `lattice-inference` `elementwise_cpu_bench` (rms_norm, layer_norm, silu_inplace,
  gelu, add_bias_gelu, softmax_attention, elementwise_mul — the kernels the decode
  loop actually calls every step): **no regression**, every group within ±2.6%,
  well under the 3% silent-pass threshold. 0 FAIL-level (>7%) results anywhere in
  the run.
- `lattice-embed` `simd`: 20 groups landed in the 3-7% WARN band and 9 showed
  improvement. This crate has zero changes in this branch (the diff touches only
  `crates/inference/`), so these deltas are machine noise, not a real regression —
  confirmed by a concurrent, unrelated Metal GPU test process from another local
  worktree that was running throughout this measurement and independently produced
  40%+ run-to-run variance on an identical, unchanged CPU code path (see below).

**Supplementary CPU decode-loop A/B.** `record_logprob` is new code, so there is no
"before" version of it to diff, and no committed benchmark in this repo drives the
real CPU decode loop (`Qwen35Model::generate`) end-to-end — the existing benches
either use a synthetic reimplementation (`inference_perf.rs::bench_forward_with_cache`)
or only exercise `Sampler::sample` (`bench_sampler_allocation`), neither of which
calls `record_logprob`. To get a real before/after number on the actual changed
decode path, a temporary (not committed) harness ran `Qwen35Model::generate` on the
real `qwen3.5-0.8b` checkpoint at two token counts (32, 160) with `logprobs: None`,
computing the prefill-canceling decode slope this repo's own `bench_decode_ab`
harness uses, on both `origin/main` and this branch:

| revision | N=32 median (ms) | N=160 median (ms) | decode slope (tok/s) |
|---|---:|---:|---:|
| `origin/main` (a00d7ebcd) | 5234.6 | 13997.7 | 14.61 |
| this branch (4ad5d7cf3) | 4948.8 | 13646.4 | 14.72 |

Individual per-run times varied by over 40% between repeats at fixed N on both
revisions (e.g. HEAD N=32: 4718-6748ms across 5 runs) — the same unrelated background
Metal GPU test load affecting the `simd` numbers above. Within that noise band the
two slopes are indistinguishable; there is no evidence of a regression on the actual
decode path.

## Known gaps / follow-ups

- Streaming + `logprobs` is out of scope (rejected with HTTP 400, see above).
- `lattice_serve.rs`'s minimal HTTP server has no `logprobs` support at all (out of
  scope; its `ChatReq` has no such fields).
- The Metal `forward::metal_qwen35::` in-module test suite was not re-run under live
  execution this session (compile-checked clean under all four relevant
  feature/scope combinations, and the crate-wide default-feature test suite passed
  in full); a concurrent, unrelated Metal GPU test process from another local
  worktree was occupying the GPU for the full duration of this work, and re-running
  a second concurrent Metal test job against it would have produced unreliable
  results for both. Worth a follow-up run in isolation.
- No committed Criterion benchmark drives the real CPU or Metal decode loop
  end-to-end at real-model scale; the supplementary A/B above used a temporary,
  uncommitted harness. Adding one (that also gets a genuine before/after baseline,
  which requires the benchmarked function to already exist unchanged on `main`)
  would be useful follow-up infrastructure, independent of this issue.
