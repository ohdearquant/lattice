"""Stratified subset of the sample for models too slow to run all 2,000 rows on the host.

Population: the sample rows in the 37 benchmarks the edition-0.3 index averages (index02.spec areas), minus the
suite's excluded runs. Round-robin over those benchmarks in sample order, taking one whole question group per
benchmark per turn (groups in sample order), until at least --n rows are picked; rows keep their sample order.
Deterministic, no seed. With the 2,000-row sample and --n 500 it yields 500 rows from all 37 benchmarks.

Usage (kit venv): python -I subset500.py --sample S --suite-dir D --out OUT [--n 500]
"""
import argparse
import collections
import json

from decision_index.scoring import index02
from decision_index.suite.io import Suite

ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True)
ap.add_argument("--suite-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()
full = Suite(a.suite_dir, "0.3")
excluded = full.excluded()
indexed = {int(n) for area in index02.spec("0.3")["areas"] for n in area["benchmarks"]}
assert len(indexed) == 37, len(indexed)
lines = open(a.sample).read().splitlines()
bench = collections.OrderedDict()
for line in lines:
    e = json.loads(line)["_evaluation"]
    if e["run_id"] in excluded or not full.in_edition(e) or e["catalog_id"] not in indexed:
        continue
    bench.setdefault(e["catalog_id"], collections.OrderedDict()).setdefault(e["group_id"], []).append(line)
queues = {b: list(g.values()) for b, g in bench.items()}
picked, turn = [], 0
while len(picked) < a.n and any(queues.values()):
    for b in bench:
        if queues[b] and len(picked) < a.n:
            picked.extend(queues[b].pop(0))
    turn += 1
keep = set(picked)
out = [line for line in lines if line in keep]
open(a.out, "w").write("\n".join(out) + "\n")
print(json.dumps({"population_rows": sum(len(r) for g in bench.values() for r in g.values()), "benchmarks": len(bench),
                  "rounds": turn, "rows": len(out), "groups_per_benchmark_min": min(sum(1 for r in g.values() if r[0] in keep) for g in bench.values())}))
