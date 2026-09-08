#!/usr/bin/env python3
"""Size a W1 training run from a calibration log, instead of from an estimate.

The exact-gradient CPU trainer has three cost terms and only one of them is the training
itself. Reading them off a real run matters because the two overheads dominate:

  CACHE   building the frozen-prefix cache, once per run, linear in the number of TRAIN
          samples. It is not persisted, so every arm of a multi-seat experiment pays it again.
  SCORE   a scoring pass over EVERY train cache plus every held-out cache. It fires at the
          baseline, at each `--log-every` boundary, at the last step, and once more at the
          end, so a run pays for at least three of them however the flag is set.
  STEP    one forward+backward and one optimiser update on ONE sample. This is the only term
          that does the work the run exists for.

The script parses a calibration log for the first two and derives the third, then prints the
projected wall clock for candidate configurations. It refuses to print a projection if the log
does not contain the lines it needs, because a projection built from defaults would look
exactly like one built from measurement.

Usage:
    uv run python scripts/microlora/size_w1_run.py --log calib.log \
        --train-rows 216 --valid-rows 22 [--arms 6] [--budget-hours 12]
"""

from __future__ import annotations

import argparse
import re
import sys


def parse(log: str) -> dict:
    """Pull the measured quantities out of a trainer log. Missing any of them is fatal."""
    out: dict = {}
    m = re.search(r"(\d+) completion positions across (\d+) samples in ([\d.]+)s", log)
    if m:
        out["cache_samples"] = int(m.group(2))
        out["cache_secs"] = float(m.group(3))
    m = re.search(r"held-out: \d+ completion positions across (\d+) valid samples", log)
    if m:
        out["valid_samples"] = int(m.group(1))
    m = re.search(r"steps:\s+(\d+)", log)
    if m:
        out["steps"] = int(m.group(1))
    m = re.search(r"max-train:\s+(\d+)", log)
    if m:
        out["max_train"] = int(m.group(1))
    m = re.search(r"in ([\d.]+)s ===", log)
    if m:
        out["loop_secs"] = float(m.group(1))
    m = re.search(r"log-every[: ]+(\d+)", log)
    if m:
        out["log_every"] = int(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="a completed calibration log")
    ap.add_argument("--log-every", type=int, default=1, help="the calibration's --log-every (not always echoed in the header)")
    ap.add_argument("--train-rows", type=int, required=True, help="rows in the real train split")
    ap.add_argument("--valid-rows", type=int, required=True, help="rows in the real held-out split")
    ap.add_argument("--arms", type=int, default=6, help="total runs (seeds x label conditions)")
    ap.add_argument("--budget-hours", type=float, default=12.0)
    ap.add_argument("--score-rate", type=float, default=None,
                    help="MEASURED seconds per sample for one scoring pass, from a two-point "
                         "log-every run. Without it the cache rate is assumed, which this script "
                         "refuses to project from when the arithmetic disproves it.")
    ap.add_argument("--candidates", default="16:32,32:64,48:96,64:128,96:192,216:400",
                    help="comma-separated max_train:steps pairs to project")
    args = ap.parse_args()

    log = open(args.log, encoding="utf-8", errors="replace").read()
    got = parse(log)
    need = ("cache_samples", "cache_secs", "valid_samples", "steps", "loop_secs")
    missing = [k for k in need if k not in got]
    if missing:
        print(f"REFUSED: the log is missing {missing}. It is probably still running: the "
              f"per-step cost is only derivable from the final '=== done: ... in Ns ===' line.",
              file=sys.stderr)
        return 3

    n_cache, t_cache = got["cache_samples"], got["cache_secs"]
    n_valid, n_steps, t_loop = got["valid_samples"], got["steps"], got["loop_secs"]
    log_every = got.get("log_every", args.log_every)

    per_cache = t_cache / n_cache
    # The loop ran n_steps steps; a scoring pass fired every `log_every` steps and again on the
    # last step. Solve the two unknowns from the one measured total by assuming the scoring pass
    # costs the same per sample as the cache build does per sample, then reporting the residual
    # as the per-step cost. Stated as an assumption, not hidden: the cache build walks the frozen
    # prefix, a scoring pass walks the materialised layers and the output head, so this is an
    # ORDER-OF-MAGNITUDE split and the projection below is a planning number, never a claim.
    n_scores = n_steps // max(1, log_every)
    score_samples = n_scores * (n_cache + n_valid)
    per_score = args.score_rate if args.score_rate is not None else per_cache
    t_score_total = score_samples * per_score
    t_step = (t_loop - t_score_total) / n_steps

    print(f"measured, from {args.log}:")
    print(f"  cache build      {t_cache:8.1f}s over {n_cache} samples   -> {per_cache:6.2f}s/sample")
    print(f"  training loop    {t_loop:8.1f}s over {n_steps} steps with {n_scores} scoring passes")
    print(f"  scoring passes   {t_score_total:8.1f}s  ({score_samples} sample-forwards, priced at the cache rate)")
    print(f"  residual per step{t_step:8.1f}s  (forward+backward+update on ONE sample)")
    if t_step <= 0:
        # The model is not merely uncertain here, it is DISPROVED by its own arithmetic: pricing
        # the scoring passes at the cache rate consumes more than the whole measured loop. Printing
        # a projection table underneath that is how a broken instrument produces finding-shaped
        # output, so it refuses instead. The fix is a MEASURED scoring rate, and the two-point run
        # that yields one is named in the message rather than left to the reader to invent.
        print(f"\nREFUSED: pricing {n_scores} scoring passes at the cache rate ({per_cache:.2f}s/sample) "
              f"costs {t_score_total:.1f}s, which exceeds the entire measured loop of {t_loop:.1f}s. "
              f"The per-step residual comes out at {t_step:.1f}s, so the assumption is disproved, not "
              f"merely approximate: a scoring pass reuses the frozen prefix and walks only the "
              f"materialised layers, so it is CHEAPER per sample than a cache build, not equal.\n"
              f"\nNo projection is printed, because a planning number derived from a disproved model "
              f"is worse than no number: it would be quoted.\n"
              f"\nTo get a sound one, run the SAME config twice changing only --log-every, so the two "
              f"totals differ by a known number of scoring passes and nothing else. With log-every L1 "
              f"giving N1 passes and L2 giving N2, the scoring rate is "
              f"(T1 - T2) / ((N1 - N2) * (train + valid)), and the per-step cost then falls out of "
              f"either run. Pass the result back in with --score-rate.", file=sys.stderr)
        return 3
    print()

    budget = args.budget_hours * 3600
    print(f"projected wall clock per arm, and for {args.arms} arms (budget {args.budget_hours}h):")
    print(f"  {'max_train':>9} {'steps':>6} {'epochs':>7} {'cache':>8} {'score':>9} {'train':>9} {'per arm':>9} {'all arms':>9}  verdict")
    for pair in args.candidates.split(","):
        mt, st = (int(x) for x in pair.split(":"))
        if mt > args.train_rows:
            print(f"  {'':9} {'':6} candidate max_train={mt} exceeds the {args.train_rows} rows "
                  f"available; clamped (a silent clamp would misreport the row as measured)")
            mt = args.train_rows
        # At minimum three scoring passes fire: baseline, the last step, and the final report.
        c = mt * per_cache
        s = 3 * (mt + args.valid_rows) * per_cache
        tr = st * t_step
        arm = c + s + tr
        allarms = arm * args.arms
        verdict = "fits" if allarms <= budget else "OVER"
        print(f"  {mt:9} {st:6} {st/mt:7.2f} {c/3600:7.2f}h {s/3600:8.2f}h {tr/3600:8.2f}h "
              f"{arm/3600:8.2f}h {allarms/3600:8.2f}h  {verdict}")
    print()
    print("Note: the cache is rebuilt per arm because the trainer does not persist it. Arms that "
          "share a data split share a cache in principle but not in practice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
