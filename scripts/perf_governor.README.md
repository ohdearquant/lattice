# perf_governor — macOS Bench Resource Guardrail

Hard gate that runs before (and during) any perf measurement on this machine.
Pure stdlib, no pip deps. macOS-only. No sudo required.

The module lives at `scripts/perf_governor.py` (tracked, CI-reachable, survives a
machine change). The runtime kill-switch sentinel is **decoupled** from the
module location and defaults to a stable repo-rooted path so the emergency-stop
path never moves when the script does (see KILL-SWITCH below).

## The 6 Guards

| # | Name        | Trigger                                                            | Action                                                                                                                                                                                                     |
| - | ----------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | AC-GATE     | Not on AC power                                                    | Refuse at preflight (fail-closed)                                                                                                                                                                          |
| 2 | THERMAL     | `CPU_Speed_Limit < 100` or recorded warning                        | Refuse at preflight; pause (SIGSTOP) + cooldown during run; hard abort after N cycles. **Fail-OPEN** on a pmset read error (assumes nominal) — deliberate; BOUNDED + AFK + KILL-SWITCH still bound the run |
| 3 | BOUNDED     | Wall-clock elapsed ≥ `max_window_s` (default 90 s)                 | Kill child process group; raise `GovernorAbort`                                                                                                                                                            |
| 4 | COOLDOWN    | Between every run                                                  | Mandatory sleep (`cooldown_s`, default 30 s); kill-switch aborts it                                                                                                                                        |
| 5 | KILL-SWITCH | Sentinel file exists (default repo-rooted `.khive/loop/PERF_STOP`) | Immediate abort at any check point                                                                                                                                                                         |
| 6 | AFK-ONLY    | `HIDIdleTime < afk_threshold_s` (default 300 s)                    | Refuse at preflight (fail-closed); override with `afk_only=False` / `--no-afk`                                                                                                                             |

## CLI

```sh
# Current system state (also prints the resolved sentinel_path)
python3 scripts/perf_governor.py --status

# Gate check only (exit 0 = clear, exit 2 = blocked)
python3 scripts/perf_governor.py --preflight

# Demonstrate every guard tripping without a real bench (sanity demo)
python3 scripts/perf_governor.py --selftest

# Full gate: preflight → run → cooldown (replace 'cargo bench ...' with your cmd)
python3 scripts/perf_governor.py --run --label my-bench -- cargo bench -p lattice-inference

# Override options
python3 scripts/perf_governor.py --run --no-afk --max-window 60 --cooldown 15 \
    --label simd-dot -- cargo bench -p lattice-embed -- simd_dot_product
```

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
