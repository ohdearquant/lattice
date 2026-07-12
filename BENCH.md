# bench-compare — #682 Stage 1 (dequant-on-demand MoE expert cache)

Command: `make bench-compare` (quick mode, defaults) from the worktree,
comparing `origin/main` vs the working tree (uncommitted #682 Stage 1
changes at the time of this run). `bench-compare.sh` labels both sides
`708f45e3f` because it resolves `HEAD` via `git rev-parse` (unaffected by
uncommitted changes) while actually running the benches in-place against the
dirty working tree for the "HEAD" side — the numbers below do reflect this
PR's code.

## Scope caveat (expected, not a gap)

The default `bench-compare` target (`elementwise_cpu_bench` for
`lattice-inference`, `simd` for `lattice-embed`) does **not** pass
`--features metal-gpu`. #682 Stage 1's changed code
(`crates/inference/src/forward/moe_expert_cache.rs`,
`crates/inference/src/forward/metal_qwen35.rs`'s `encode_moe_ffn` /
`load_moe_ffn_q4`) is entirely `#[cfg(all(target_os = "macos", feature =
"metal-gpu"))]`-gated and is not compiled into this bench binary at all, so
this run cannot regress from it by construction. There is no existing
Criterion bench target that exercises `encode_moe_ffn`'s MoE decode path
(recon in `PLAN.md` §1 confirms zero MoE bench_results ever landed) — a
throughput measurement against the checked-in synthetic MoE fixture would
need a new bench target, out of scope for Stage 1 per the task brief
("`make bench-compare` on the existing (non-35B, must already fit resident)
MoE-capable configs" — no such bench target exists yet to run). Numerical
correctness (identical logits with/without eviction pressure) is covered
instead by the mutation-sensitive tests in `metal_qwen35.rs` (see PR
description / final report).

## Result: `lattice-inference` groups — no regression

Every group Criterion actually ran for `lattice-inference`
(`rms_norm`, `layer_norm`, `silu_inplace`, `gelu`, `add_bias_gelu`,
`softmax_attention`, `elementwise_mul`) is **absent from the FAIL/WARN
table** — i.e. all within the ≤3.0% CI-lower noise band. Expected: this
PR's code isn't even linked into this bench binary (see scope caveat above).

## Result: `lattice-embed` groups — 6 FAIL / 19 WARN, unrelated crate

The gate did flag 6 FAIL (>7% regression) and 19 WARN (3-7%) entries, **all**
in `lattice-embed`'s `simd` bench target (`simd_batch_dot_product`,
`simd_euclidean_distance`, `simd_prepared_query_normalized_cosine`,
`int8_quantization`, `int8_raw_dot_product`, `binary_cosine_distance`, etc.).
This PR touches zero files under `crates/embed/` — `lattice-embed` does not
depend on the changed code paths. `bench-compare.sh`'s own header comment
documents `simd`-target quick-mode noise as a known issue (lattice#714:
"confirmed noise-dominated in --quick mode by a same-toolchain A/A
reproduction on identical refs"); this run's spread (up to +16.5% on one
group) is consistent with that pre-existing noise class on a shared machine,
not a code regression from an unrelated crate change. Full FAIL/WARN table
and all 263 measurements are in the raw `make bench-compare` output
(not committed — regenerate with the command above).

## Conclusion

No regression attributable to #682 Stage 1. The `lattice-embed` noise is
pre-existing and orthogonal (different crate, not on this PR's dependency
path); `lattice-inference`'s benched groups are clean.
