# ADR-098: A multi-slot decision read-out and a `/v1/systemone` endpoint

**Status**: Proposed (2026-10-10)\
**Date**: 2026-10-10\
**Extends**: ADR-097\
**Crate**: lattice-inference

<!-- deno-fmt-ignore-start -->

## Context

This record serves an instruction from the maintainer, at the console, 2026-10-10 15:09 and 15:27 (UTC-4):

> on mac, the metal prefill is still slow, we will need to do a lot of work over there, I do wanna support those inside lattice, dig into these, do some research, you keep on playing with these then, lmk when you have a proper bench of various 'system one' models from hugging face first

> plan stuff out, and make very detailed items, ［…］ and write down the plan, ［…］ maybe write some adr as well

［…］ marks two elided clauses concerning internal work assignment. The complete instruction is held in the
maintainer's internal record.

<!-- deno-fmt-ignore-end -->

This ADR serves the "support those inside lattice" part: how a decision model of this class is asked a question
through lattice. The benchmark of public checkpoints that the instruction asks for first is measurement work, not a
design decision, and is not recorded here.

A class of small models answers typed questions about a state (choose one of N options, yes/no, a score on an
ordered scale) with one forward pass and no decoding: the answer is a probability over the options read off the
model at a known position. Publicly: Mapika `decider-{0.8b,2b,4b}` (full fine-tunes of Qwen3.5-Base; the prompt
lists every question first, then one `Answer k: (` slot per question; the read-out at each slot is the `lm_head`
rows of the option-letter tokens, A-J then two-letter tokens up to 255 options, softmax at a fitted temperature),
autotrust `JEV-9B` / `JEV-27B` (frozen base + LoRA r=16 + a 24-slot fp32 head read at the last token of a bare
template, per-kind temperatures), and a public benchmark (the Decision Index: 110,201 requests over 37-42
benchmarks, chance-corrected skill, an HTTP wire format `POST /v1/systemone` with `{model, state, questions}` and
per-option probabilities in the answer). About 116 models have been submitted; the strongest open ones at 0.8B-2B
sit on Qwen3.5 backbones lattice already runs.

Lattice today (read at `80d45e0184f48f24b630f0e06471085cd5268529`):

- ADR-097 already provides the one-question case: `score_option_letters(prompt_ids, letter_ids)` on the CPU
  (`model/qwen35/eval.rs:161`) and on Metal (`forward/metal_qwen35.rs:7892`) resets the state, runs one prefill and
  reads the last-position logits at the given letter ids, returning `OptionScores { logits, probs, label_mass }`.
  Letters resolve through `option_scoring::resolve_option_letters`, which accepts `A..Z` and refuses any letter
  that is not exactly one token. Scores are uncalibrated by contract (ADR-097 decision 6), and ADR-097 leaves
  sharing one prefill across several questions out of scope.
- Metal also has a prefill that returns the final hidden states (`forward/metal_qwen35.rs:7941
  forward_prefill_with_hidden`) and an all-position logits prefill (`:8025 forward_prefill_all_logits`, which refuses
  a multi-token prompt while a LoRA adapter is active).
- Serving exposes `/v1/chat/completions` with `logprobs`/`top_logprobs` (at most 20, next-token only, through the chat
  template: `serve/contract.rs normalize_logprobs`, `serve/mod.rs format_normalized_chat_template`),
  `/v1/embeddings` and `/v1/lora*`. There is no template-free token-id entry on the serving path, no read-out at
  positions other than the last, no letter table past `Z`, no decision-head loader and no `/v1/systemone`.

So a decider checkpoint loads (it is a plain `Qwen3_5ForCausalLM`; to be verified) and can be asked one question per
prefill through a library call, but not the several questions its prompt layout carries, and not over HTTP. The cost
of this class is the prefill (the output is a fixed handful of numbers), so serving it well on Apple silicon is a
Metal prefill problem, and the public benchmark measures exactly that.

## Decision

1. **Generalize ADR-097's scorer from the last position to N slot positions.** One primitive below the serving
   layer, `prefill_readout(input: PrefillInput, slots: &[usize], rows: &[u32]) -> Vec<Vec<f32>>`: one prefill, the
   final (post-norm) hidden state gathered at each slot position, multiplied by the `lm_head` rows named in `rows`
   only, returning one logit vector per slot over `rows`. `PrefillInput` is raw token ids, and also the multimodal
   sequence the vision runtime already builds (image tokens and M-RoPE positions) when a state carries an image, so
   that image states are an input change later rather than a second primitive. Metal and CPU implement the same
   signature. `score_option_letters` stays as it is, with its callers and its bitwise tests; it is the case
   `slots = [last]`, `rows = letter_ids`, and those tests are the control that the general path reproduces it.
2. **`rows` is per request, not per question.** A request names one row list shared by every slot, at most 255
   distinct token ids, never the full vocabulary; the restricted projection is computed once as slots x rows. Each
   question reads its own subset of those logits (for the decider layout, the first n entries of the letter table for
   an n-option question), so a single question has at most 255 options and a request with many questions still
   projects onto at most 255 rows.
3. **The letter table extends ADR-097's resolution rule, it does not relax it.** A checkpoint's table (decider: `A`-`J`,
   then 229 two-letter tokens) is resolved through the loaded tokenizer, and every entry must encode to exactly one
   token with no duplicate ids, as in ADR-097 decision 3. A table that fails the check refuses at load, not at the
   first request.
4. **A decision head is a second source for the projection.** `rows` selects `lm_head` rows; a loaded
   `[n_slots, hidden]` fp32 head (`head.safetensors`) is the other way to build the same per-slot logits. LoRA applies
   as on the generation path; the read-out has no "unload the adapter" requirement, since the head class needs one.
5. **Calibration stays above the primitive.** The primitive returns raw logits, as ADR-097's scorer does. The serving
   layer applies the temperature the checkpoint publishes (one value, or one per question kind) before the softmax
   and reports the value it used. This is the checkpoint's own calibration, not a lattice-fitted one; ADR-097
   decision 6 (no content-free prior division) is unchanged.
6. **The wire layer is `/v1/systemone`, format-compatible with the public benchmark.** Request `{model, state,
   questions}` with kinds `choice` (2-255 options, each with criteria text), `noul` (yes/no), `score` (2-10 ordered
   levels); response `answers{key:{type, choice, probabilities}}` and `usage.input_tokens`. The prompt rendering for
   a checkpoint is a byte-exact port of that checkpoint's published prompt code, selected by a `readout` section in
   the checkpoint's lattice config. The chat template is never involved.
7. **Acceptance is differential, before any number is reported.** For a checkpoint with a public reference engine:
   first, on a single prompt, the hidden state at a slot agrees with the reference framework's last-layer post-norm
   hidden state to < 1e-3 in f32; then, on 300 benchmark requests, max |delta p| <= 1e-3 per option and identical
   argmax on 300 of 300 against the reference engine's answers; then a 2,000-request sample scores within 0.5 points
   of the reference engine on the same sample.
8. **Latency is reported per request beside the reference on the same host class**, binned by input tokens, and a
   serving bench target covers the path so later prefill work has a paired A/B. No throughput claim without it.
   ADR-097 decision 8 applies: the question and option text are part of the prefill and part of the quoted length.

## Alternatives considered

- **Amend ADR-097 instead of a new record.** Not chosen: ADR-097 is a library scorer with a deliberately narrow
  contract (one question, last position, uncalibrated, caller-rendered prompt). This record adds positions, a head,
  a calibration step and an HTTP surface; folding those into ADR-097 would change the contract its implementation
  and tests already pin.
- **Call `score_option_letters` once per question.** Rejected: the decider layout reads every question's slot from
  one prefill, and a per-question prefill multiplies the cost by the question count and changes what each slot sees.
- **Expose full-vocabulary logits at every position** (`forward_prefill_all_logits` exists). Rejected for serving:
  248,320 x N floats per request for a question with at most 255 options, and it refuses a multi-token prompt while an
  adapter is active.
- **Drive the read-out through `/v1/chat/completions` with `top_logprobs`.** Rejected: at most 20, next-token only,
  and the chat template changes the prompt bytes the checkpoints were trained on.
- **Decode one token greedily and treat the letter as the answer.** Rejected: loses the probability vector (the
  benchmark and the routers consume probabilities, and calibration is scored), and it costs a decode step.
- **Only the head, not the letter-slot read-out.** Rejected: the open 0.8B-4B checkpoints on the public board are
  letter-slot models; the head is a second projection source on the same primitive (decision 4), not a separate path.

## Consequences

- New surface: one primitive in `forward/` (Metal and CPU), one letter-table resolver beside
  `resolve_option_letters`, one head loader, one endpoint in the serving layer, one `readout` config section. No new
  crate, no new dependency.
- The benchmark kit's HTTP engine becomes a standing differential test for the serving path.
- Metal prefill throughput on 500-5,000-token prompts becomes a first-class measured number. The levers (batched
  prefill of N requests sharing a schema prefix, prefix reuse, KV precision, chunk size) each get their own ADR with
  an A/B, after the measurement exists.
- Out of scope here: 27B dense and MoE backbones, encoder-only decision models, adaptive thinking (a second, thinking
  pass on low-confidence answers). Image state is a follow-on over the same primitive (decision 1).

## Verification plan (pre-registered)

The checkpoint loads unchanged (perplexity on 20 lines finite and within 2x of the base checkpoint's); the general
path reproduces `score_option_letters` bitwise on its existing fixtures; the hidden-state differential < 1e-3; 300 of
300 argmax and max |delta p| <= 1e-3 against the reference engine; sample score within 0.5 points; latency table beside
the reference.
