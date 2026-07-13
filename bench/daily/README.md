# Daily bench rows

One JSONL row per day is appended to `YYYY-MM.jsonl` (schema `bench-row/v1`). Each row is
reproducible from the verbatim `command` field at the recorded `sha`, with host and run
configuration captured in the row and the native result stored under `bench/data/`.

## Registered suites

| Suite | Command | What it measures |
| --- | --- | --- |
| `simd_dot_product_micro` | `cargo bench -p lattice-embed --bench simd -- "simd_dot_product"` | Criterion mean latency (ns, 95% CI in the committed estimates) for the f32 dot-product kernel, scalar vs NEON SIMD, at embedding dims 384/768/1024/1536. The foundational primitive under every embedding similarity path. |

Additional suites (flagship end-to-end decode/prefill measurements) are registered here as
their runs begin landing rows.

## Row conventions

- `lane` is `perf` for metric rows from the suites above.
- Timing runs execute on an otherwise idle machine under an exclusive advisory lock so
  concurrent builds cannot contend the cores (see the repo docs on bench serialization).
- A blocked day is recorded as an explicit miss row: `{"schema": "bench-row/v1", "date": ...,
  "repo": "lattice", "miss": true, "reason": "<one line>"}`.
- `artifact` points at the committed native result under `bench/data/`; bulk Criterion state
  stays out of the repo — only the day's row and its directly referenced estimates land here.
