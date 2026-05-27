Verdict: REJECT
Findings: 1 Blocker, 4 High/Medium, 0 Low/Nit

Scope: performance and Metal GPU review of `docs/adr/ADR-062-metal-fa2-prefill.md`, `docs/adr/ADR-061-inference-metrics-infrastructure.md`, and the current Metal implementation at `crates/inference/src/forward/metal_qwen35.rs`.

## Findings

### [Blocker] ADR-062 does not provide a 32 KiB-safe FA2 tile for the actual Qwen3.5-0.8B head dimension

Evidence: `docs/adr/ADR-062-metal-fa2-prefill.md:173` says the starting tile table covers MLX-style instantiations, but `docs/adr/ADR-062-metal-fa2-prefill.md:175-181` only lists head dims `64`, `80`, and `128`. `docs/adr/ADR-062-metal-fa2-prefill.md:382-385` then makes Phase 2 deliverables only `D=64/80/128`. The target model is not in that set: `crates/inference/src/model/qwen35_config.rs:201-215` defines `qwen35_0_8b()` with `head_dim: 256`, and the existing Metal decode shader hard-codes `constexpr uint HEAD_DIM = 256` at `crates/inference/src/forward/metal_qwen35.rs:440-444`.

Why this matters: ADR-062 is explicitly about Qwen3.5 Metal prefill, but the proposed kernel variants cannot run the repository's current Qwen3.5-0.8B attention shape. This is not only a missing row: under the ADR's own 32 KiB TGM constraint, `D=256, BQ=32, BK=16` already needs 32.0 KiB for f16 Q/K/V tiles alone, before scores, row state, padding, or bank-conflict slack; `BQ=32, BK=32` needs 48.0 KiB for f16 Q/K/V alone.

Suggested fix: Add a D=256-specific design before accepting ADR-062. Either choose a smaller tile shape such as `BQ=16, BK=16` with explicit TGM/register accounting, or change the kernel staging so Q/V/scores are not all resident in threadgroup memory. Add D=256 to the implementation deliverables and acceptance tests.

### [High] The ADR-062 TGM table undercounts the buffers it says are resident

Evidence: `docs/adr/ADR-062-metal-fa2-prefill.md:159-161` says TGM holds Q `[BQ,D]`, K `[BK,D]`, V `[BK,D]`, softmax accumulators, and padding. But `docs/adr/ADR-062-metal-fa2-prefill.md:168` reports `D=128, BQ=32, BK=32` as 18.5 KiB f16 and 34.5 KiB f32. Q/K/V alone are `2 * 128 * (32 + 32 + 32) = 24,576 B = 24.0 KiB` for f16 and `48.0 KiB` for f32, before any scores or row state. The same inconsistency appears in the context summary at `docs/adr/ADR-062-metal-fa2-prefill.md:44`.

Why this matters: Tile feasibility is the load-bearing Metal design constraint. If the table is used as implementation guidance, engineers can select variants that exceed `maxTotalThreadgroupMemoryLength` or silently rely on a different staging strategy than the ADR describes.

Suggested fix: Replace the table with an explicit formula and one row per actual resident buffer. If scores are in registers or Q is streamed instead of staged, say so and update the pseudocode. Add a compile-time or pipeline-build assertion comparing `staticThreadgroupMemoryLength` against the device/pipeline limit.

### [Medium] ADR-061's entropy formula is correct only with an unstated online rescale, and the absolute-logit form is less stable than necessary

Evidence: `docs/adr/ADR-061-inference-metrics-infrastructure.md:85-95` defines `e = sum exp(a_s - m) * a_s` and `H = log(l) + m - e / l`. The derivation at `docs/adr/ADR-061-inference-metrics-infrastructure.md:97-104` is mathematically correct when `m`, `l`, and `e` are all expressed against the final running max.

Why this matters: In an online tiled kernel, when `m_new > m_old`, both `l` and `e` must be rescaled: `e_new = alpha * e_old + sum exp(a_s - m_new) * a_s`. The ADR states the final formula but not the update rule, which is the part implementers are most likely to get wrong. Also, using absolute logits makes entropy depend on subtracting two similarly shifted terms (`m - e/l`); a shifted accumulator `r = sum exp(a_s - m) * (a_s - m)` gives the equivalent `H = log(l) - r/l` and avoids that cancellation.

Suggested fix: Add the tiled recurrence for `e` explicitly, or switch the ADR to the shifted-logit accumulator. If ADR-062 uses the `exp2` path for metrics, also state whether entropy is accumulated in natural-log units or converted from log2 units.

### [Medium] The f16-to-int8-to-int4 chain is directionally sound, but one Qwen cost claim uses the wrong GQA group size

Evidence: `docs/adr/ADR-062-metal-fa2-prefill.md:267-315` describes an int8 path after f16 KV, and `docs/adr/ADR-062-metal-fa2-prefill.md:316-331` extends that to int4 with WHT rotation. The progression is plausible if the memory-reduction factors are relative to f32 and include metadata/residual-tail overhead. However, `docs/adr/ADR-062-metal-fa2-prefill.md:330` says pre-RoPE K quantization costs `~10-11% for Qwen's GQA group size 7`, while `docs/adr/ADR-062-metal-fa2-prefill.md:213` says Qwen3.5-0.8B has 8 Q heads, 2 KV heads, and `gqa_factor=4`.

Why this matters: The int4 path's overhead argument is numerically tied to group reuse. With `G=4`, the cited `3/(4G)` term is 18.75%, not 10-11%. That does not invalidate the chain, but it changes the performance tradeoff and should not be presented as Qwen-specific evidence.

Suggested fix: Clarify that `2x -> 4x -> 8x` is relative to f32 payload before metadata/tail overhead, and recompute the pre-RoPE K cost for Qwen3.5's `gqa_factor=4`. If another Qwen model has `G=7`, name that model separately.

### [Medium] ADR-062 overstates the current prefill synchronization overhead

Evidence: `docs/adr/ADR-062-metal-fa2-prefill.md:34` says current prefill has "`n` GPU command buffer synchronization points." The code creates one command buffer and one encoder for the whole prefill at `crates/inference/src/forward/metal_qwen35.rs:6311-6313`, then dispatches per-token work inside the same encoder loop at `crates/inference/src/forward/metal_qwen35.rs:6693-6826`.

Why this matters: The current path still has too many per-token dispatches and no query-block tiling, so the FA2 motivation remains valid. But command dispatch overhead and command-buffer synchronization are different Metal costs; diagnosing the wrong one can lead to the wrong benchmark design and mitigation.

Suggested fix: Change the ADR wording to "`n` per-token attention dispatches inside one prefill command buffer" unless a separate path actually commits/waits per token. Benchmark dispatch count, encoder duration, and GPU work separately.

## Checks That Passed

- Exp2 constant: `docs/adr/ADR-062-metal-fa2-prefill.md:138-155` uses `scale * log2(e)` before `exp2`, which is the correct conversion because `log2(e) = 1 / ln(2) ~= 1.4426950408889634`.
- Apple bandwidth spot-check: ADR-061's `200 GB/s` for M1 Pro at `docs/adr/ADR-061-inference-metrics-infrastructure.md:150` matches Apple's M1 Pro newsroom/spec text; `273 GB/s` for M4 Pro at `docs/adr/ADR-061-inference-metrics-infrastructure.md:159` matches Apple's M4 Pro newsroom/spec text.
- Apple TGM constraint: ADR-062's 32 KiB threadgroup-memory ceiling at `docs/adr/ADR-062-metal-fa2-prefill.md:40-44` matches Apple's Metal feature tables for Apple7-Apple10 families. The problem is the ADR's tile arithmetic, not the cited limit.
- Current decode shape: the existing decode shader is Qwen3.5-specific and requires `HEAD_DIM=256` at `crates/inference/src/forward/metal_qwen35.rs:440-444`, so the D=256 gap is grounded in both model config and current Metal code.

## Sources Consulted

- Apple M1 Pro/M1 Max newsroom: https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/
- Apple M4 Pro/M4 Max newsroom: https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
- Apple Metal Feature Set Tables: https://developer.apple.com/metal/capabilities/
- Qwen3.5-0.8B config: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/eb706f593d2d43c90a10271199c10b07ced7569a/config.json

## What I Checked

- Read ADR-061 and ADR-062 in full.
- Read the current Metal decode shader around `crates/inference/src/forward/metal_qwen35.rs:418-559`.
- Read current prefill code around `crates/inference/src/forward/metal_qwen35.rs:6238-6816`.
- Checked the Qwen3.5-0.8B local config preset and upstream config for `head_dim`, `num_attention_heads`, and `num_key_value_heads`.
- Recomputed TGM estimates for `D in {64,80,128,256}` and `BQ/BK` combinations with a small local script.

## What I Did Not Check

- I did not compile Metal shaders or run GPU benchmarks.
- I did not validate MLX Steel's exact register/TGM staging beyond comparing ADR claims to its stated resident-buffer model.
- I did not audit every future quantization paper citation in ADR-062; this pass focused on the requested Metal/performance issues.

## Recommended Next Steps

1. Revise ADR-062 around D=256 before implementation starts; this is the only blocker.
2. Recompute the tile table from an explicit resident-buffer formula and add D=256 candidate rows.
3. Amend ADR-061's online entropy section with the exact tiled recurrence.
4. Re-run a narrow review after those ADR changes; no broad architecture re-review is needed if the fixes stay local.

Domain utility: SKIPPED - no `mcp__lore__suggest` / `compose` tools were available in this session, so I used the repository review skills and primary sources instead.
