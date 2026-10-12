"""Score a kit results.jsonl on the rows of a sample, with a group bootstrap SE.

The kit's `score` command scores against the whole suite, so unanswered suite rows count as wrong and a 2,000-row
sample reads as ~0. This scorer runs the kit's own edition-0.3 scoring core (benchmark_summary, added.report,
index02.index_entry) over the sample's rows only. `index_core` mirrors pipeline.score_run_v02 minus file writes;
`--control-full` runs it on the unrestricted suite and compares it with the kit's own scores.json for the same
results, which is the check that the mirror is faithful.

Bootstrap: within each benchmark, resample whole question groups with replacement (copies relabelled so the kit
treats them as distinct groups), re-score, SE = sample stdev of the index over resamples.

Usage (kit venv): python -I score_sample.py --results R --sample S --suite-dir D [--boot 1000 --seed 20261010]
                  [--control-full KIT_SCORES_JSON] --out OUT.json
"""
import argparse
import collections
import json
import random
import statistics
import sys


def load_kit():
    from decision_index.scoring import added, index02
    from decision_index.scoring.report import benchmark_summary, load_results
    from decision_index.suite.io import Suite
    return added, index02, benchmark_summary, load_results, Suite


class RowsSuite:
    """The two members the scoring core reads from a Suite: .edition and .rows()."""

    def __init__(self, edition, rows):
        self.edition = edition
        self._rows = rows

    def rows(self, apply_exclusions=False):
        return iter(self._rows)


def index_core(suite, results, kit):
    added, index02, benchmark_summary, _, _ = kit
    spec = index02.spec(suite.edition["id"])
    added_ids = {int(n) for n in spec["added"]}
    base, extra = [], collections.defaultdict(list)
    for r in suite.rows(apply_exclusions=True):
        n = r["_evaluation"]["catalog_id"]
        (extra[n] if n in added_ids else base).append(r)
    metrics = {int(n): (m["name"], m["key"]) for n, m in spec.get("metrics", {}).items()}
    summary = benchmark_summary(suite, results, "sample", None, rows=base, metrics=metrics)
    added_reports = {n: added.report(n, rows, results) for n, rows in sorted(extra.items())}
    return index02.index_entry(suite, results, summary, added_reports, spec)


def resample(rows, results, rng):
    by_bench = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        e = r["_evaluation"]
        by_bench[e["catalog_id"]][e["group_id"]].append(r)
    out_rows, out_res = [], {}
    for n in sorted(by_bench):
        gids = sorted(by_bench[n])
        for k in range(len(gids)):
            g = gids[rng.randrange(len(gids))]
            for r in by_bench[n][g]:
                e = dict(r["_evaluation"])
                rid = f"{e['run_id']}#b{k}"
                e["run_id"], e["group_id"] = rid, f"{e['group_id']}#b{k}"
                out_rows.append({**r, "_evaluation": e})
                if r["_evaluation"]["run_id"] in results:
                    out_res[rid] = results[r["_evaluation"]["run_id"]]
    return out_rows, out_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--edition", default="0.3")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--control-full", help="kit scores.json for the same results; compare the unrestricted core to it")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kit = load_kit()
    _, _, _, load_results, Suite = kit
    full = Suite(a.suite_dir, a.edition)
    results = load_results(a.results)
    out = {"results": a.results, "sample": a.sample, "edition": full.edition["id"], "answered": len(results)}

    if a.control_full:
        mine = index_core(full, results, kit)
        theirs = json.load(open(a.control_full))
        diffs = [k for k in theirs["index_benchmarks"] if theirs["index_benchmarks"][k] != mine["benchmarks"].get(k)]
        same = mine["index"] == theirs["decision_index"] and mine["areas"] == theirs["areas"] and not diffs
        out["control_full"] = {"mine_index": mine["index"], "kit_index": theirs["decision_index"], "benchmark_mismatches": diffs, "identical": same}
        print(json.dumps(out["control_full"]), flush=True)
        if not same:
            json.dump(out, open(a.out, "w"), indent=2)
            raise SystemExit("CONTROL FAILED: the scoring mirror disagrees with the kit on the full suite")

    excluded = full.excluded()
    sample_rows = [json.loads(l) for l in open(a.sample)]
    rows = [r for r in sample_rows if r["_evaluation"]["run_id"] not in excluded and full.in_edition(r["_evaluation"])]
    sample_ids = {r["_evaluation"]["run_id"] for r in rows}
    missing = sorted(sample_ids - set(results))
    out.update(sample_rows=len(sample_rows), scored_rows=len(rows), missing_results=len(missing), extra_results=len(set(results) - sample_ids))
    point = index_core(RowsSuite(full.edition, rows), results, kit)
    out["index"] = point["index"]
    out["raw_index"] = point["raw_index"]
    out["areas"] = [{k: x[k] for k in ("id", "skill", "raw", "coverage", "n")} for x in point["areas"]]
    out["benchmarks"] = point["benchmarks"]
    if a.boot:
        rng = random.Random(a.seed)
        vals = []
        for i in range(a.boot):
            br, bres = resample(rows, results, rng)
            vals.append(index_core(RowsSuite(full.edition, br), bres, kit)["index"])
            if (i + 1) % 100 == 0:
                print(f"boot {i + 1}/{a.boot}", file=sys.stderr, flush=True)
        out["boot"] = {"n": a.boot, "seed": a.seed, "se": statistics.stdev(vals), "mean": statistics.mean(vals),
                       "p2.5": sorted(vals)[int(0.025 * a.boot)], "p97.5": sorted(vals)[int(0.975 * a.boot) - 1]}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("index", "raw_index", "scored_rows", "missing_results") if k in out} | ({"se": out["boot"]["se"]} if a.boot else {})))


if __name__ == "__main__":
    main()
