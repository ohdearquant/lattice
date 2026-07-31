# perf_governor — macOS Bench Resource Guardrail

Hard gate that runs before (and during) any perf measurement on this machine.
Pure stdlib, no pip deps. macOS-only. No sudo required.

The module lives at `scripts/perf_governor.py` (tracked, CI-reachable, survives a
machine change). The runtime kill-switch sentinel is **decoupled** from the
module location and defaults to a stable repo-rooted path so the emergency-stop
path never moves when the script does (see KILL-SWITCH below).

## The 6 Guards

| # | Name        | Trigger                                                            | Action                                                                                                                                                                                                            |
| - | ----------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | AC-GATE     | Not on AC power                                                    | Refuse at preflight (fail-closed)                                                                                                                                                                                 |
| 2 | THERMAL     | `CPU_Speed_Limit < 100` or non-nominal `ProcessInfo.thermalState`  | Refuse at preflight; pause (SIGSTOP) + cooldown during run; hard abort after N cycles. Falls back from unavailable `pmset` output to Foundation's supported thermal state and fails closed if neither can be read |
| 3 | BOUNDED     | Wall-clock elapsed ≥ `max_window_s` (default 90 s)                 | Kill child process group; raise `GovernorAbort`                                                                                                                                                                   |
| 4 | COOLDOWN    | Between every run                                                  | Mandatory sleep (`cooldown_s`, default 30 s); kill-switch aborts it                                                                                                                                               |
| 5 | KILL-SWITCH | Sentinel file exists (default repo-rooted `.khive/loop/PERF_STOP`) | Immediate abort at any check point                                                                                                                                                                                |
| 6 | AFK-ONLY    | `HIDIdleTime < afk_threshold_s` (default 300 s)                    | Refuse at preflight (fail-closed); override with `afk_only=False` / `--no-afk`                                                                                                                                    |

## CLI

```sh
# Current system state (also prints the resolved sentinel_path)
python3 scripts/perf_governor.py --status

# Gate check only (exit 0 = clear, exit 2 = blocked)
python3 scripts/perf_governor.py --preflight

# Settle, collect the shared `lattice-machine-state-v1` JSON record, attach the
# gate verdict, and fail closed. `bench-compare` invokes this at all three phase
# boundaries with a 30 second settle/AFK floor.
python3 scripts/perf_governor.py --checkpoint --label before-base \
    --cooldown 30 --afk-threshold 30

# Demonstrate every guard tripping without a real bench (sanity demo)
python3 scripts/perf_governor.py --selftest

# Full gate: preflight → run → cooldown (replace 'cargo bench ...' with your cmd)
python3 scripts/perf_governor.py --run --label my-bench -- cargo bench -p lattice-inference

# Override options
python3 scripts/perf_governor.py --run --no-afk --max-window 60 --cooldown 15 \
    --label simd-dot -- cargo bench -p lattice-embed -- simd_dot_product
```

On an ordinary direct invocation, `--run` first re-execs through
`scripts/lib/bench-locks.py`, which retains both machine-wide descriptors for
the complete governor run. Invoke it directly; wrapping it in
`bench-command.sh` is redundant. The handoff assumes a cooperative caller; an
independent caller-side lock that does not forward the repository handoff
state would make the nested acquisition wait on its own ancestor.

## Kill-switch sentinel

The sentinel path resolves with precedence **`--sentinel` arg > `$PERF_GOVERNOR_SENTINEL` env > repo-rooted default**. The default is `<repo-root>/.khive/loop/PERF_STOP` regardless of where this script lives.

```sh
# Abort a running guarded session (default location)
mkdir -p .khive/loop && touch .khive/loop/PERF_STOP

# Use a custom sentinel location
python3 scripts/perf_governor.py --run --sentinel /tmp/my_stop -- cargo bench ...
PERF_GOVERNOR_SENTINEL=/tmp/my_stop python3 scripts/perf_governor.py --run -- cargo bench ...
```

## Programmatic API

```python
import sys
sys.path.insert(0, "scripts")
from perf_governor import PerfGovernor, GovernorAbort

gov = PerfGovernor(max_window_s=60, cooldown_s=15, afk_only=True)
gov.preflight()   # raises GovernorAbort if any gate blocks
rc = gov.run_guarded("my-bench", ["cargo", "bench", "-p", "lattice-inference"])
gov.cooldown()

# Custom kill-switch path (else: $PERF_GOVERNOR_SENTINEL, else repo-rooted default)
gov = PerfGovernor(sentinel_path="/tmp/my_stop")
```

For testing, inject fake readers:

```python
gov._thermal_reader = lambda: {"speed_limit": 70, "nominal": False}
gov._ac_reader = lambda: False
gov._idle_reader = lambda: 10.0
```
