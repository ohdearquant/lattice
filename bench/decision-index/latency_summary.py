"""Per-request latency and input size of one reference arm, from the kit's results.jsonl.

Usage: python -I latency_summary.py <results.jsonl> [<rows.jsonl>]
Only the last record per run_id counts, as in the kit's own loader.
With a rows file, only results whose run_id is in it are counted, and a row missing from the results is reported.
Wall = total_wall_ms as the kit records it. Input tokens = the server's usage.input_tokens where the engine reports
it, beside the kit's proxy_tokens (the same count for every engine). Bins are by proxy_tokens.
"""
import json
import statistics
import sys

records = []
for line in open(sys.argv[1], encoding="utf-8"):  # complete lines only, as the kit's loader reads them
    if not line.endswith("\n"):
        break
    if line.strip():
        records.append(json.loads(line))
if not records:
    sys.exit("FAILED READ: no result rows in " + sys.argv[1])
last = {}
for r in records:  # the kit keeps the last record per run_id (report.py:16-22); a retried row's earlier error is history
    last[r["run_id"]] = r
results = list(last.values())
print(f"records={len(records)} distinct run_ids={len(results)}")
if len(sys.argv) > 2:
    want = {json.loads(line)["_evaluation"]["run_id"] for line in open(sys.argv[2]) if line.strip()}
    have = {r["run_id"] for r in results}
    print(f"rows file: {len(want)} rows, {len(want & have)} with a result, {len(want - have)} missing")
    results = [r for r in results if r["run_id"] in want]

status = {}
for r in results:
    status[r["status"]] = status.get(r["status"], 0) + 1
print("status:", status)
ok = [r for r in results if r["status"] == "ok"]
if not ok:
    sys.exit("no ok rows")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


wall = [r["total_wall_ms"] for r in ok]
proxy = [r["proxy_tokens"] for r in ok]
model_in = [r["raw_output"]["usage"]["input_tokens"] for r in ok
            if isinstance(r.get("raw_output"), dict) and isinstance(r["raw_output"].get("usage"), dict)
            and isinstance(r["raw_output"]["usage"].get("input_tokens"), int)]
print(f"ok n={len(ok)} wall_ms median={statistics.median(wall):.1f} p10={q(wall, .1):.1f} p90={q(wall, .9):.1f} "
      f"max={max(wall):.1f} sum_h={sum(wall) / 3.6e6:.2f}")
print(f"proxy_tokens median={statistics.median(proxy)}  server input_tokens: {len(model_in)} rows report it"
      + (f", median={statistics.median(model_in)}" if model_in else ""))
for lo, hi in [(0, 512), (512, 1024), (1024, 2048), (2048, 5120), (5120, 10**9)]:
    b = [r["total_wall_ms"] for r in ok if lo < r["proxy_tokens"] <= hi or (lo == 0 and r["proxy_tokens"] == 0)]
    if b:
        print(f"  proxy ({lo},{hi if hi < 10**9 else 'inf'}]: n={len(b)} median_ms={statistics.median(b):.1f}")
