# Decision Index: public "system one" decision models on Apple silicon

A "system one" decision model reads a state (a prompt, a document, a list of candidates) and answers a set of
questions by returning a probability over each question's options from a single prefill, with no generated text.
The [Decision Index](https://github.com/apolinario/decision-index) is the public benchmark for this class: 110,201
requests in 43 benchmarks, 37 of which form the index across five areas.

This directory holds the scorer, the engine wrappers and the server launch scripts used to measure public decision
models on an Apple M4 with each model author's own inference code, and the resulting reference table. These are
**reference numbers, not lattice numbers**: no row below runs lattice. A lattice row is added through the same scorer
once lattice serves the multi-slot decision read-out described in
[ADR-098](../../docs/adr/ADR-098-decision-readout.md).

Status (2026-10-12, closed for this host): 6 of 8 reference arms scored; OneJev 0.8B and Intern-Decision-0.8B do
not fit the 16 GiB host as configured. A host with at least 32 GiB of unified memory is the next test for those two
arms and for the rows other arms left unanswered on memory here: OneJev was stopped at 17 GB with its peak unknown,
and Intern-Decision reached 12-15 GB at its default memory cap.

## Methodology

- **Suite.** Decision Index edition 0.3, rebuilt locally with the kit at commit `9eb2dbe2a358004c8782c66e40a83ac07b953fec`; every rebuilt row file is
  byte-identical (uncompressed) to the kit's published pins, and the kit's import check accepted each one. 110,201 requests in 43 benchmarks; 37 form the index (area weights
  0.258 / 0.258 / 0.200 / 0.183 / 0.100).
- **Sample.** 2,000 requests drawn by the kit's own sampler (`suite sample --n 2000`, seed 20260919, whole question
  groups, round-robin per benchmark); sha256 `9ba719e5ceb69e31c6434200fd7c4be882b09bbcffe188a61a82106bad4bd0af`.
  1,750 of the rows fall in the 37 indexed benchmarks.
- **Subset for slow arms.** A rule fixed before the first full arm: an arm whose median request on a 20-row smoke
  exceeds 2 s runs a 500-row subset of the sample instead of all 2,000 (decider-0.8b, the first arm, was exempted by
  the same rule and ran all 2,000). The subset is built by [`subset500.py`](subset500.py): round-robin over the 37
  indexed benchmarks in sample order, one whole question group per benchmark per turn, until at least 500 rows;
  deterministic, no seed. On this sample it yields 500 rows from all 37 benchmarks, sha256
  `9a22d80befd609fc30fd56ec67bdea830d1708b793d4f1acc615e89af939ed2e`. Such a row's standard error comes from its own
  bootstrap, so its agreement tolerance is wider.
- **Score.** [`score_sample.py`](score_sample.py) runs the kit's edition-0.3 scoring core (`benchmark_summary`,
  `added.report`, `index02.index_entry`) over the sample's rows only. On the full suite it reproduces the kit's own
  `scores.json` exactly (`--control-full`), and a one-value change to the kit's scoring file makes that check fail.
  The number reported is the kit's headline: balanced skill, chance-corrected, 0 = random, 100 = perfect, unanswered
  = wrong. That is the board's public score (`v03.pub`); the board's full score adds private tests that cannot be run
  here.
- **Uncertainty.** Bootstrap standard error: 1,000 resamples of whole question groups within each benchmark, seed
  20261010. The agreement check, fixed before any arm ran: |ours - board public| <= max(2, 2 x SE).
- **Engine.** Each model's own inference code, unmodified except for device plumbing stated per row. Models that ship
  a `/v1/systemone` server run under the kit's `http` engine against that server ([`servers/`](servers/)); the
  others run in process through a kit `Engine` subclass around the author's scoring call ([`engines/`](engines/),
  through [`run_engine.py`](run_engine.py)). Where the board documents lifting an author-set serving cap for an
  entry, the same lift is applied. One model at a time, the server restarted per model.
- **Latency.** Per-request wall time as the kit records it (`total_wall_ms`: the HTTP round trip including the
  server's prompt build), sequential requests, batch 1. Each chunk of requests ran under
  `scripts/bench-command.sh --durable` (machine bench-window and GPU locks, CPU idle floor checked before and after).
  Host: Apple M4 Mac mini, 16 GiB, macOS 27.0, Python 3.12.12, one virtual environment per model: torch 2.14.1
  (MPS) and transformers 5.17.0 for decider-0.8b, decider-2b (plus mlx 0.32.3), Jev-Style 0.8B and Eos; mlx 0.32.3,
  mlx-lm 0.31.3 and transformers 5.19.0 for Jev-Style 2B; mlx 0.32.2, mlx-lm 0.31.3, torch 2.8.0 and transformers
  5.17.0 for Kev.
- **Timeouts and restarts.** The kit's http client gives up on a request after 600 s, and the author servers keep
  computing an abandoned request, so later requests would queue behind it. Server arms therefore ran in chunks of at
  most 100 rows on a freshly started server, each chunk ending on its first error. Error rows were retried on a fresh
  server; a row that errored twice stays unanswered (scored wrong, the kit's rule). If the host reached critical
  memory pressure, the run stopped the client and then the server, and the row in flight stayed unanswered. After an
  arm finished, every unanswered row was retried alone on a fresh server; both the as-run and the retested scores are
  reported. In-process arms ran the same way, in chunks of 10 rows on a freshly loaded engine, with the MPS allocator
  capped at the device's recommended working set (11.84 GiB on this host; [`run_engine.py`](run_engine.py) sets
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`), so a request that does not fit fails alone instead of taking the host to
  critical memory pressure.

## Confounds

- **Not the board's hardware or kernels.** The board ran these models on CUDA (an RTX PRO 6000 for most entries) with
  the authors' fast paths (CUDA graphs, torch.compile, fp8 linears, flash-linear-attention, causal-conv1d). On MPS
  those are absent, and Qwen3.5's gated-delta and causal-conv layers fall back to the reference PyTorch
  implementations in transformers. The latency column is "author code on an M4" and is not comparable to the
  board's latency.
- **Numerics.** Where an author defaults to fp8 or fp16 and this run used bf16 (or the reverse), individual
  probabilities move; each row says which. The agreement check is what shows the score did not move beyond noise.
- **Sample, not suite.** About 47 scored requests per benchmark: per-benchmark scores are noisy, the index less so
  (the SE column).
- **Memory.** A 16 GiB host cannot hold some long retrieval requests (32 candidates) for some models. Those rows are
  unanswered and scored wrong, so the affected scores are lower bounds.

## Reference table

Board public = `v03.pub` from the board's `index.json` (generated 2026-10-10T07:54:16Z).

| Model (Hugging Face repo @ revision)                                        | Params | Board public     | Ours (2,000 sample unless stated)                                                                                                                                                                                                                     | SE   | Delta | Agreement       | Median ms (M4) | Input tokens / request |
| --------------------------------------------------------------------------- | ------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ----- | --------------- | -------------- | ---------------------- |
| Jev-Style 2B v3 (`chaoliangUNSW/Jev-Style-2B-Decision-v3-MLX` @ `11ce5d71`) | 2B     | 34.05            | 35.28 (2,000 of 2,000 answered)                                                                                                                                                                                                                       | 1.41 | +1.23 | pass (tol 2.82) | 481            | 475.5                  |
| decider-2b (`Mapika/decider-2b` @ `533964da`)                               | 2B     | 29.75            | 29.07 as run (6 unanswered); 29.18 after retest (5 unanswered)                                                                                                                                                                                        | 1.58 | -0.57 | pass (tol 3.15) | 808            | 465                    |
| decider-0.8b (`Mapika/decider-0.8b` @ `a0a01d6f`)                           | 0.8B   | not on the board | 22.13 (8 unanswered after retest)                                                                                                                                                                                                                     | 1.38 | n/a   | n/a             | 790            | 463                    |
| Jev-Style 0.8B v3 (`chaoliangUNSW/Jev-Style-0.8B-Decision-v3` @ `b023d1f9`) | 0.8B   | 17.00            | 16.50 (7 unanswered)                                                                                                                                                                                                                                  | 1.41 | -0.50 | pass (tol 2.82) | 817            | 539                    |
| Kev 0.8B r15 (`jaredpalmer/kev-0.8b` @ `bf75a6a8`)                          | 0.8B   | 16.78            | 16.72 (2,000 of 2,000 answered)                                                                                                                                                                                                                       | 1.20 | -0.06 | pass (tol 2.39) | 200            | 451                    |
| OneJev 0.8B (`OmniJev/OneJev-0.8B` @ `c3939d8b`)                            | 0.8B   | 22.22            | not measured: a text-only retrieval request (8,186 tokens, 32 candidates) drives the author server past 16 GiB                                                                                                                                        |      |       |                 |                |                        |
| Eos 0.8B (`vllm-sr/Decision-2.0-Eos-0.8B` @ `52cad37e`)                     | 0.8B   | 20.57            | 500-row subset: 18.37 as run (25 unanswered); 18.69 after retest (19 unanswered)                                                                                                                                                                      | 2.11 | -1.88 | pass (tol 4.21) | 1,021          | 556                    |
| Intern-Decision-0.8B (`internlm/Intern-Decision-0.8B` @ `85a0cc5a`)         | 0.8B   | 11.81            | not measured: with the first engine design the third smoke request exceeds 16 GiB; with the second (in-process, MPS memory capped), the default cap took the host to critical memory pressure and a 0.6 cap left 9 of 13 smoke rows out of MPS memory |      |       |                 |                |                        |

Input tokens are each server's own count (median over answered rows); the kit's own proxy count has a median of
338 to 343 per request.

Per-row notes:

- **Jev-Style 2B v3** ran the author's MLX bf16 build (the format its card recommends for Apple silicon), because the
  FP32 PyTorch path does not fit 16 GiB; backend and precision therefore differ from the board's run. p10 / p90
  latency: 134 ms / 7.1 s.
- **decider-2b** ran the author's own MPS fp16 path; the board figure came from FP8 and torch.compile on CUDA. The 5
  unanswered rows are 32-candidate retrieval rows that took the server to critical memory pressure on 16 GiB, as run
  and again alone; 29.18 is a lower bound. p10 / p90: 193 ms / 10.8 s.
- **decider-0.8b** ran its own repository's server through a device shim ([`servers/serve_mps.py`](servers/serve_mps.py)),
  since that revision ships the CUDA-only package. The 8 unanswered rows are 32-candidate retrieval rows that took a
  freshly started server past 16 GiB; 22.13 is a lower bound. Before the run it was expected between
  Intern-Decision-0.8B (11.81) and OneJev 0.8B (22.22), with any score within two standard errors of a bound counted
  as inconclusive at that bound. 22.13 is inside the range but 0.09 below the upper bound, so the check is
  inconclusive there. p10 / p90: 160 ms / 13.1 s.
- **Jev-Style 0.8B v3**: the 7 unanswered rows were refused by the author server with HTTP 422
  `input_budget_exceeded` (one question's own text is longer than the runtime's 2,048-token head budget, so it
  refuses rather than truncate). An offline replay of the server's request path predicted exactly these 7. p10 / p90:
  245 ms / 15.9 s.
- **Kev 0.8B r15** ran the author's default Apple-silicon path (MLX). p10 / p90: 48 ms / 2.2 s.
- **Eos 0.8B** ran in process through the author's `Decision2.system_one`
  ([`engines/eos_engine.py`](engines/eos_engine.py)) with the wrapper's defaults (bf16 weights resident) on MPS,
  allocator capped at 11.84 GiB. Its smoke median exceeded 2 s, so it ran the 500-row subset; its SE (2.11) is larger
  than the full-sample rows' (1.20 to 1.58). The 19 unanswered rows ran out of the capped allocator or took the host
  to critical memory pressure, as run and again alone; 18.69 is a lower bound. Two 10-row chunks failed the closing
  CPU-idle check, so their 11 records count for the score and are left out of the latency. p10 / p90: 267 ms /
  14.9 s.

## Reproducing a row

1. Install the kit at `9eb2dbe2a358004c8782c66e40a83ac07b953fec` in its own virtual environment (`pip install -e ".[rebuild]"`), then build, stage
   and sample edition 0.3 with the kit's own commands, as its README describes:
   `python -m decision_index suite rebuild --work work` (one source, `cais/hle`, needs a Hugging Face token with the
   dataset's terms accepted), `python -m decision_index suite import ...` with the three rebuilt row files (it refuses
   any file whose hash does not match and stages them in `suite-0.3/`), and
   `python -m decision_index suite sample --edition 0.3 --n 2000 --out sample-2000.jsonl`. Check the sha256 above.
   For a row run on the 500-row subset, build it in the same environment with
   `python -I subset500.py --sample sample-2000.jsonl --suite-dir suite-0.3 --out sample-subset500.jsonl` and check its
   sha256; steps 3 to 5 then take `sample-subset500.jsonl` in place of `sample-2000.jsonl`.
2. Download the model at the revision in the table and start its server, for example
   `bash servers/kev_serve.sh <model_dir> <python> 8000`. Each script states the author's flags it uses and why.
3. Run the kit against it:
   `python -I -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=<name> --rows sample-2000.jsonl --out results/<slug>`.
   In-process engines run through `python -I run_engine.py run --engine eos_engine:EosEngine --option model_dir=<dir> ...`
   with the same `--rows` and `--out`.
4. Score: `python -I score_sample.py --results results/<slug>/results.jsonl --sample sample-2000.jsonl --suite-dir <suite-0.3 dir> --boot 1000 --seed 20261010 --out <slug>-score.json`.
5. Latency and input size: `python -I latency_summary.py results/<slug>/results.jsonl sample-2000.jsonl`.

The table's rows ran step 3 in chunks (at most 100 rows per server, 10 rows per in-process engine) with the restart
and retest rules above; a single unchunked run can queue requests behind an abandoned one when a request times out.

Every file here is byte-identical to the copy that produced the table except six, which differ only in comments:
`run_engine.py`, `latency_summary.py`, `subset500.py`, `servers/jev_style_serve_mlx.sh`, `servers/onejev_serve.sh` and
`servers/decider08_serve.sh`; the last also locates `serve_mps.py` beside itself instead of through an absolute path,
and `subset500.py` ran under a different file name.
