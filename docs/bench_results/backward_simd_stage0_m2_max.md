# Backward SIMD Stage 0 Baseline

- Issue: #849
- Source: `134a99d74b047acf823c45dc45dadc8b8581f2fc` plus the Stage 0 harness
- Date: 2026-07-29

## Run conditions

- Host: Apple M2 Max, 12 cores (8 performance, 4 efficiency), 32 GB RAM
- OS: macOS 27.0 (26A5388g)
- Toolchain: `rustc 1.94.1 (e408947bf 2026-03-25)`, `aarch64-apple-darwin`
- Criterion: 0.5.1; 20 samples; 1 second warm-up; 3 second measurement; 3% noise threshold
- Both machine-wide locks were acquired immediately and uncontended:
  `/tmp/lion-bench-window.lock` and `/tmp/lion-metal-gpu-test.lock`
- Command:

  ```text
  CARGO_TARGET_DIR="$HOME/build-target" \
    python3 scripts/lib/bench-locks.py \
      --label codex-849-stage0 \
      --status-file /tmp/codex-849-bench-locks.txt \
      -- cargo bench -p lattice-inference \
        --bench backward_stage0 --features train-backward
  ```

The table reports Criterion's bootstrap mean and 95% confidence interval. It is
a hardware-specific profiling baseline for deciding whether later ADR-083
stages merit implementation, not a cross-machine performance threshold.

## Shapes

| Primitive     | Tiny                                                               | Qwen3.5-0.8B production shape                                                      |
| ------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Cross entropy | sequence 4, completion 2, vocab 128                                | sequence 8, completion 4, vocab 248,320                                            |
| Linear VJP    | `d_in=64`, `d_out=192`                                             | `d_in=1024`, `d_out=3584`                                                          |
| LoRA VJP      | rank 4, `d_in=64`, `d_out=192`                                     | rank 16, `d_in=1024`, `d_out=3584`                                                 |
| RMSNorm       | dimension 64                                                       | dimension 1024                                                                     |
| RoPE          | head 64, rotary 32                                                 | head 256, rotary 64                                                                |
| SwiGLU        | hidden 64, intermediate 192                                        | hidden 1024, intermediate 3584                                                     |
| GQA           | sequence 4, hidden 64, 4 query / 2 KV heads, head 16, rank 4       | sequence 16, hidden 1024, 8 query / 2 KV heads, head 256, rotary 64, rank 16       |
| GDN           | sequence 4, hidden 64, 2 key / 4 value heads, key/value 16, rank 4 | sequence 8, hidden 1024, 16 key/value heads, key/value 128, convolution 4, rank 16 |

The GQA and GDN fixtures include the LoRA paths exercised by the training tape.
Their sequence lengths are bounded profiling windows while every layer
dimension matches the released 0.8B text decoder.

## Baseline distributions

| Primitive                | Shape        |     Mean |          95% CI |
| ------------------------ | ------------ | -------: | --------------: |
| `cross_entropy_backward` | tiny         | 1.300 µs |  1.257–1.369 µs |
| `cross_entropy_backward` | Qwen3.5-0.8B | 4.470 ms |  4.401–4.582 ms |
| `linear_vjp`             | tiny         | 587.4 ns |  554.2–623.2 ns |
| `linear_vjp`             | Qwen3.5-0.8B | 215.4 µs |  201.5–230.7 µs |
| `lora_vjp`               | tiny         | 1.151 µs |  1.116–1.195 µs |
| `lora_vjp`               | Qwen3.5-0.8B | 51.36 µs |  49.39–53.47 µs |
| `rmsnorm_backward`       | tiny         | 81.13 ns |  79.19–83.38 ns |
| `rmsnorm_backward`       | Qwen3.5-0.8B | 1.146 µs |  1.115–1.183 µs |
| `rope_backward`          | tiny         | 32.85 ns |  32.04–33.85 ns |
| `rope_backward`          | Qwen3.5-0.8B | 44.18 ns |  43.89–44.55 ns |
| `swiglu_backward`        | tiny         | 2.664 µs |  2.632–2.707 µs |
| `swiglu_backward`        | Qwen3.5-0.8B | 1.309 ms |  1.263–1.360 ms |
| `gqa_backward`           | tiny         | 13.11 µs |  12.84–13.45 µs |
| `gqa_backward`           | Qwen3.5-0.8B | 9.366 ms | 8.785–10.162 ms |
| `gdn_backward`           | tiny         | 46.62 µs |  46.31–47.13 µs |
| `gdn_backward`           | Qwen3.5-0.8B | 35.48 ms |  34.78–36.37 ms |

## `bench-compare` disposition

No A/B comparison applies to this Stage 0 measurement-only change. The only
library module added by the change is behind
`cfg(any(test, feature = "test-utils"))`; the benchmark target is built only
when selected explicitly with `train-backward`. Default production and
`make bench-compare` binaries therefore compile the same effective library
source as the base. The table above is the baseline that later SIMD stages
must compare against.
