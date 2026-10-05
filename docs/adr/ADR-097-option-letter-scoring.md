# ADR-097: Option-letter scoring: one prefill, logits read at the option letters

**Status**: Proposed
**Date**: 2026-10-05
**Crate**: lattice-inference

## Context

A classification-style decision ("which of these k options applies to this state?") is cheaper on a small causal model
if the options are listed in the prompt with letters and the answer is read from the next-token distribution at the
letter tokens, instead of generating text or running one continuation per option.

Two shapes were measured on Qwen3.5-0.8B QuaRot-Q4, Metal, Apple M4:

- Per-option continuation: prefill the state once, then score each option as its own continuation. A continuation
  (recurrent-state restore plus a few tokens) costs about 45 ms at a 512-token state, roughly the price of 51 prefill
  tokens, nearly independent of state length. At k = 14 that is about 635 ms on top of the prefill.
- Listed options, one prefill: each listed option costs a few prompt tokens at about 0.9 ms per token up to 512 tokens.
  Prefill (adapter off) measured 115.5 / 223.2 / 452.6 / 975.9 ms at 128 / 256 / 512 / 1024 tokens.

The second shape is the one this ADR adopts. It has a cost the first does not: the listed options interact. Changing
only the order of the options changes the argmax on 23.3% of permuted AG-NEWS item pairs in the measured zero-shot run
(k = 4), and substantially more on tasks the base model does not already solve. A content-free prompt (the same question and
options, with the state replaced by an empty string, `N/A` or `[MASK]`) also shows a letter prior on several option
lists: for example 0.80 on one letter of TREC's six.

This ADR adopts a mechanism, and it reports no result. On the decision tasks the scorer was built for (classifying a
knowledge-graph edge kind, a task's priority and a task's outcome from exported state), zero-shot accuracy with this
scorer on the 0.8B base was below the majority-class baseline on all three, so zero-shot scoring is not usable for
them. Whether an adapted model is usable is a separate question with its own measurement. Nothing in this ADR is
evidence that any task can be decided this way.

Today the only way to get this score is to call `try_forward_prefill` (or `forward_prefill`) and index the returned
vocabulary logits by hand, after resolving letter tokens by hand. Nothing checks that a letter is a single token, that
the prompt ends where a letter is expected, or that an active adapter takes the batched prefill path.

## Decision

1. **One entry point, one prefill.** `score_option_letters(prompt_ids, letter_ids) -> Result<OptionScores>` on the
   Metal Qwen3.5 state, with a CPU counterpart on the Qwen3.5 model for parity. It resets the recurrent and KV state,
   runs one prefill over `prompt_ids`, and reads the last-position logits at `letter_ids`. No sampling, no decode step.
2. **Output.** `OptionScores { logits: Vec<f32>, probs: Vec<f32>, label_mass: f32 }`: the k raw logits in the caller's
   order, their softmax over the k, and the full-vocabulary probability mass on the k letters. Every option's score is
   returned, never only the argmax, so calibration or permutation averaging can sit on top later without an API change. `label_mass` is a
   format-adherence signal (a prompt the model does not answer with a letter shows low mass); it is reported, never
   used to rescale `probs`.
3. **Letter resolution is checked, not assumed.** A helper resolves letters `A..` to token ids through the loaded
   tokenizer and refuses any letter that does not encode to exactly one token. For the Qwen3.5 tokenizer `A..N`
   resolve to single tokens (ids 32..45, no leading space) when the letter starts a line after the closed think block.
   The rendered prompt, including the chat template and the closed think block, is the caller's responsibility; the
   ADR fixes only what the scorer reads.
4. **Refusals.** Empty `prompt_ids`, `k = 0`, a letter id outside the vocabulary, duplicate letter ids, a prompt longer
   than the state's capacity, or a non-finite logit return an error. Nothing is clamped or guessed.
5. **Adapters.** With an adapter loaded, the prefill takes the batched adapter path (row-aware adapter kernels, after
   the fix for adapter-on prefill falling back to one step per token). The scorer never loads or unloads an adapter;
   the adapter in effect is whatever the state holds, and the scores carry that adapter's name when the caller asks.
6. **Uncalibrated by contract.** `probs` is the raw softmax over the k letters. Dividing by a content-free prior was
   measured and is not adopted: it improved two public sets but failed its registered control, so calibration is
   decided separately, against a control in which the answer lives only in the state.
7. **Option order is the caller's to control and to report.** The scorer does not permute or average. A consumer that
   trains an adapter for this scorer shuffles option order per training example, and every evaluation of an adapted
   scorer reports its order-sensitivity (the share of items whose argmax changes under a permutation of the options)
   after adaptation, beside accuracy.
8. **Latency is quoted with the options in the prompt.** Any time-to-first-token figure for a decision includes the
   listed options in the prompt length, since they are part of the prefill. A figure that leaves them out understates
   the cost by the option list's tokens.

## Consequences

- One prefill per decision; cost is linear in prompt length including the option list. Measured adapter-on prefill
  after the batched-path fix: 123.3 / 236.6 / 480.1 / 1028.4 ms at 128 / 256 / 512 / 1024 tokens, 1.048 to 1.055x
  adapter off, with a synthetic rank-8 adapter. Lengths past 1024 tokens and trained adapters of other ranks or module
  sets are unmeasured.
- Order sensitivity stays visible instead of averaged away. Permutation averaging (k passes) remains available to a
  caller and costs k prefills.
- Parity: the CPU and Metal scorers are compared on the same prompts with a maximum absolute difference in `probs`
  registered before the implementation runs, plus equal argmax.
- Out of scope: sharing one state prefill across several questions (snapshot and per-question continuation), MoE
  models, all-position logits with an adapter, and calibration.

## Alternatives considered

- **Per-option continuation** (rejected above on measured cost: about 45 ms per option).
- **Generate and parse** (rejected: decode steps cost more than one letter read and the parse can fail).
- **Expose raw logits only, no scorer** (rejected: every caller re-implements letter resolution, refusals and the
  adapter path check, and the first two are where the measured mistakes were).
