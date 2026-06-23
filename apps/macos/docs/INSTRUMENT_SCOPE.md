# Lattice Instrument — Scope & Architecture

> Status as of 2026-06-21. Every claim cites a real file. `EXISTS` = code present.
> `PARTIAL` = code present, stated limitations apply. `PROPOSED` = not yet implemented.

---

## 1. Purpose & Scope

Lattice Instrument is a zero-dependency macOS 14 SwiftUI application (Swift 6.0 tools,
Package.swift line 7) that wraps the Rust `lattice-tune` and `lattice-inference` engine
binaries as subprocess instruments. Its purpose is to let Ocean run LoRA fine-tuning,
model quantization, chat testing, and training-data inspection without leaving a macOS
window — while the Rust binaries remain the single source of correctness.

The app is NOT a trainer or inference engine. It is an instrument panel: it spawns
processes, parses their output, and renders live readouts.

**Platform**: macOS 14+, Swift 6.0 tools, Swift language mode v5, zero external deps.
**Bundle ID**: `ai.khive.lattice.studio` (package-app.sh line 9).
**Min window**: 1080×720 (LatticeStudioApp.swift line 12).

---

## 2. Current State Inventory

### 2.1 Swift Source Files (28 files in 7 directories)

| Dir | File | Role | State |
|-----|------|------|-------|
| App/ | LatticeStudioApp.swift | @main entry, @NSApplicationDelegateAdaptor, @State AppStore | EXISTS |
| Bridge/ | LatticeBridge.swift | Process spawn, binary resolution, model/adapter discovery | EXISTS |
| Bridge/ | LatticeEvents.swift | `@@lattice` protocol decoder, HumanLineParser, QuantAccumulator | EXISTS |
| Bridge/ | Drivers.swift | TrainConfig, QuantConfig, GenConfig typed arg-builders; AppStore launch extensions | EXISTS |
| Store/ | AppStore.swift | @Observable @MainActor singleton, run lifecycle, event routing, run archive | EXISTS |
| Store/ | DomainModels.swift | Screen, ModelInfo, AdapterInfo, RunKind, LiveRun, TrainPoint, RunRecord | EXISTS |
| Shell/ | ContentView.swift | NavigationSplitView two-pane shell, CommandBar overlay, global ⌘1-6 shortcuts | EXISTS |
| Shell/ | LeftRail.swift | Wordmark, live RUN block (step+loss), nav rows, system memory bar | EXISTS |
| Screens/ | TrainScreen.swift | LoRA fine-tune config + live oscilloscope + control strip | EXISTS |
| Screens/ | QuantizeScreen.swift | Q4 / QuaRot config + layer progress + mass comparison | EXISTS |
| Screens/ | ModelsScreen.swift | Model DataTable + inspector + action row (Train/Quantize/Chat→) | EXISTS |
| Screens/ | ChatScreen.swift | Config strip + single-variant transcript + generate_lora subprocess | PARTIAL |
| Screens/ | DataScreen.swift | Source dir scan, .jsonl inspector, builder-script copy buttons | EXISTS |
| Screens/ | RunsScreen.swift | Run archive DataTable + live banner + inspector | EXISTS |
| Screens/ | ScreenScaffold.swift | Shared header chrome (index / title / subtitle / trailing slot) | EXISTS |
| Components/ | CommandBar.swift | ⌘K floating palette, fuzzy prefix match, 7 default commands | EXISTS |
| Components/ | StripChart.swift | Swift Charts oscilloscope: LineMark + AreaMark + scrub-to-freeze | EXISTS |
| Components/ | HeroNumber.swift | 56pt tabular-mono hero with .contentTransition(.numericText()) | EXISTS |
| Components/ | GatePill.swift | PASS/WARN/FAIL/RUN verdict pill, 6px radius, animated pulse on RUN | EXISTS |
| Components/ | FaderToggle.swift | Console spring-fader for binary mode choices (Q4↔QuaRot, BASE↔+ADAPTER) | EXISTS |
| Components/ | ReadoutWell.swift | 15pt tabular-mono well: label + value + unit + delta caret | EXISTS |
| Components/ | OpaquePanel.swift | Instrument panel surface (opaque, 1px hairline, 0px radius) + well surface | EXISTS |
| Components/ | ParamRow.swift | Config param row variants used in TrainScreen/QuantizeScreen | EXISTS |
| Components/ | DataTable.swift | Generic sortable DataTable used across Models/Data/Runs screens | EXISTS |
| Components/ | ContrastPair.swift | Before/after contrast pair (fold-wipe on completion) used in QuantizeScreen | EXISTS |
| Components/ | MassBars.swift | Dual mass bars (before/after MB) used in QuantizeScreen | EXISTS |
| Components/ | KeyCapChip.swift | ⌘K keyboard shortcut keycap chip used in CommandBar | EXISTS |
| Theme/ | Theme.swift | Adaptive palette, SF Mono fonts, motion constants — no asset catalog | EXISTS |

### 2.2 State Model

```
AppStore (@Observable @MainActor)
  ├── selection: Screen                    current nav screen
  ├── models: [ModelInfo]                  discovered on disk
  ├── runs: [RunRecord]                    JSON-persisted archive
  ├── liveRun: LiveRun?                    the one active subprocess run
  ├── workingModel: ModelInfo?             explicit cross-screen target
  ├── handle: RunHandle?                   the one live Process wrapper
  └── binariesReady: Bool                  prebuilt .lattice binary present
```

Only one subprocess runs at a time. `AppStore.launch()` calls `prior.stop()` before starting
a new one (AppStore.swift line 108).

### 2.3 What Works Today

- TrainScreen: full config → subprocess → live step/loss/eval/done parsing → StripChart,
  ReadoutWells, PAUSE/RESUME/STOP, adapter path on save. End-to-end verified in production
  (NLL 5.18→0.61 documented in MEMORY.md).
- QuantizeScreen: Q4 and QuaRot, layer progress, mass bars, ratio, verdict GatePill.
  Drives both `quantize_q4` and `quantize_quarot` binaries.
- ModelsScreen: model discovery (parses config.json), layer summary, adapter list, navigate-to
  shortcuts, Finder reveal.
- DataScreen: .jsonl scan, summary stats, 5-example preview, builder-script copy buttons.
- RunsScreen: persistent run archive (Application Support/LatticeStudio/runs.json), live banner.
- CommandBar: ⌘K palette with 7 commands, fuzzy match.
- ChatScreen config strip and transcript: functional for single-variant generation.

---

## 3. Engine-Wrapping Architecture

### 3.1 Process Spawn

`RunHandle` (LatticeBridge.swift) wraps `Foundation.Process` with two `Pipe`s:

```
AppStore.launch(bin, args)
    → LatticeBridge.launchSpec(bin, args)     resolves binary path
    → RunHandle.start(spec)                    process.launch()
        stdout → Pipe → readabilityHandler
            split on 0x0A
            LatticeEventParser.parse(line:)
            → .trainStep / .quantLayer / .genToken / .status / .unknown
        stderr → merged into same outPipe
        stdin ← RunHandle.send(line:)           (chat only)
        SIGSTOP / SIGCONT via RunHandle.pause() / resume()
    → AppStore.consume(event, into: liveRun)   main-queue dispatch
    → LiveRun @Observable → SwiftUI renders
```

`Source: LatticeBridge.swift (RunHandle class) + AppStore.swift lines 96-120`

### 3.2 Binary Path Resolution

Resolution order (LatticeBridge.swift `launchSpec`):

1. `.app/Contents/Resources/bin/<name>` (distribution build)
2. `$LATTICE_BIN_DIR/<name>` (env override)
3. `../target/release/<name>` (relative to bundle's Resources)
4. `cargo run -p <crate> --bin <name> --features <features>` (fallback)

`Source: LatticeBridge.swift + apps/macos/DISTRIBUTION.md`

### 3.3 Bundled Binaries (6)

Defined in `apps/macos/scripts/package-app.sh`:

| Binary | Crate | Features | Used For |
|--------|-------|----------|----------|
| `quantize_q4` | lattice-inference | (none) | Q4 quantize |
| `quantize_quarot` | lattice-inference | (none) | QuaRot quantize |
| `lattice` | lattice-inference | (none) | main inference binary |
| `qwen35_generate` | lattice-inference | (none) | Qwen3.5 generation |
| `train_grad_full` | lattice-tune | `train-backward` | LoRA fine-tune |
| `generate_lora` | lattice-tune | `safetensors,inference-hook` | LoRA chat generation |

### 3.4 Event Protocol

The bridge has two parse paths (LatticeEvents.swift):

**Path 1 — JSON sentinel (preferred, available when `--json` passed):**

```
@@lattice {"ev":"train_step","step":5,"loss":3.2341,"lr":0.001000}
@@lattice {"ev":"train_eval","step":5,"val_loss":3.1200,"best_val":3.1200}
@@lattice {"ev":"train_done","base_nll":5.18,"final_nll":0.61,"duration_s":120.0}
@@lattice {"ev":"gen_token","token":" hello","done":false}
@@lattice {"ev":"gen_token","token":"","done":true,"tok_s":28.4,"ttft_ms":312.0}
```

Prefix constant: `let kLatticeEventPrefix = "@@lattice "` (LatticeEvents.swift line 14).
Both `train_grad_full` and `generate_lora` emit these when `--json` is passed
(confirmed in train_grad_full.rs lines 1155/1239/1331; generate_lora.rs lines 166/183).

**Path 2 — HumanLineParser regex fallback (for quantize bins and older binaries):**

`HumanLineParser` parses current human-readable stdout of `quantize_q4` and `quantize_quarot`
via regex (LatticeEvents.swift). These bins have no `--json` flag — the human-readable format
is the only output. `QuantAccumulator` accumulates multiple summary lines to emit `quantDone`.

**Path 3 — pass-through:** Any line not matching Path 1 or 2 becomes `.status(line)` and
appends to `LiveRun.log`.

ANSI escape sequences are stripped before parsing via `stripANSI()` (LatticeEvents.swift).

### 3.5 Verified CLI Flags

**train_grad_full** (train_grad_full.rs lines 68-82):

```
--model-dir <PATH>      model directory
--data-dir  <PATH>      dataset dir with train.jsonl + valid.jsonl
--first-layer <N>       first trained layer (default 19)
--steps <N>             Adam steps (default 25)
--lr <F>                learning rate (default 1e-3)
--rank <N>              LoRA rank (default 8)
--alpha <F>             LoRA alpha (default 16.0)
--seq-len <N>           max tokens per sample (default 64)
--max-train <N>         training samples cap (default 3)
--max-valid <N>         held-out samples for eval (default 16)
--log-every <N>         print NLL every N steps (default 5)
--json                  emit @@lattice JSON events to stdout
--save <PATH>           write PEFT safetensors adapter after training
```

TrainConfig in Drivers.swift maps 1:1 to these flags. The `--json` flag comment in Drivers.swift
notes "future --json mode; older binaries ignore unknown flags" — but the flag IS implemented in
the current binary (train_grad_full.rs line 743).

**generate_lora** (generate_lora.rs lines 70-76):

```
--model-dir <PATH> | --model <PATH>
--lora <PATH>           adapter path
--prompt <STRING>
--max-tokens <N>
--temperature <F>
--json                  emit @@lattice gen_token events
--seed <U64>
```

**quantize_q4 / quantize_quarot** (confirmed in package-app.sh; no `--json`):

```
--model-dir <PATH>
--output-dir <PATH>
--dry-run               (Q4 and QuaRot)
--seed <U64>            (QuaRot only, default 0xC0FFEE)
```

### 3.6 ASCII Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                 LatticeStudio.app                   │
│                                                     │
│  ┌──────────┐    ┌───────────┐    ┌─────────────┐  │
│  │ AppStore │←─→ │ LiveRun   │←── │ SwiftUI     │  │
│  │@Observable    │@Observable│    │ Screens     │  │
│  └────┬─────┘    └───────────┘    └─────────────┘  │
│       │                                             │
│  ┌────▼──────────────────────────────────────────┐  │
│  │              RunHandle                        │  │
│  │  Process  │ outPipe (stdout+stderr merged)    │  │
│  │           │ inPipe  (stdin, chat only)        │  │
│  │           │ SIGSTOP/SIGCONT                   │  │
│  └────┬──────────────────────────────────────────┘  │
│       │ stdout lines                                │
│  ┌────▼──────────────────────────────────────────┐  │
│  │           LatticeEventParser                  │  │
│  │  Path 1: @@lattice prefix → JSON decode       │  │
│  │  Path 2: HumanLineParser (quantize bins)      │  │
│  │  Path 3: .status(line) passthrough            │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
           │ spawn
           ▼
┌──────────────────────────────────────────────────┐
│  Rust Binaries (.app/Contents/Resources/bin/)    │
│                                                  │
│  train_grad_full   (lattice-tune +train-backward)│
│  generate_lora     (lattice-tune +safetensors)   │
│  quantize_q4       (lattice-inference)           │
│  quantize_quarot   (lattice-inference)           │
│  lattice           (lattice-inference)           │
│  qwen35_generate   (lattice-inference)           │
└──────────────────────────────────────────────────┘
```

---

## 4. Surface-by-Surface Status

### Surface 1: LoRA Fine-Tune (TrainScreen.swift)

| Feature | State | Notes |
|---------|-------|-------|
| 12 config params (ParamRow) | EXISTS | model-dir, data-dir, first-layer, steps, lr, rank, alpha, seq-len, max-train, max-valid, log-every, save-path |
| NSOpenPanel directory choosers | EXISTS | model-dir + data-dir |
| TrainConfig → subprocess | EXISTS | Drivers.swift `startTrain(_:)` |
| `@@lattice` train_step events | EXISTS | real-time loss, lr, grad_norm, val_loss, tok_s |
| StripChart oscilloscope | EXISTS | 20Hz, scrub-to-freeze, val_loss overlay |
| ReadoutWells (6: STEP, TRAIN NLL, HELD-OUT, Δ FROM BASE, TOK/S, BEST VAL) | EXISTS | |
| HeroNumber (56pt best val) | EXISTS | .numericText() per-digit tick |
| PAUSE / RESUME / STOP | EXISTS | SIGSTOP/SIGCONT via RunHandle |
| Adapter path display on done | EXISTS | `@@lattice train_done.saved` field |
| `--json` mode actually wired | EXISTS | `--json` in TrainConfig.args; binary supports it |
| Grad-norm scheduling display | PARTIAL | TrainPoint has gradNorm field; no dedicated well yet |
| ETA display | PARTIAL | eta_s in TrainStep struct (Drivers.swift); no ReadoutWell shows it |
| Per-step LR schedule chart | PROPOSED | lr field present in TrainPoint; not charted separately |
| Multi-run overlay | PROPOSED | RunRecord has no point series; chart can only show the live run |

### Surface 2: Model Management (ModelsScreen.swift)

| Feature | State | Notes |
|---------|-------|-------|
| Model discovery (config.json) | EXISTS | LatticeBridge.discoverModels() |
| Layer summary (GDN/GQA counts) | EXISTS | parses `layer_types` array from config.json |
| Adapter discovery | EXISTS | LatticeBridge.discoverAdapters() |
| DataTable (8 columns) | EXISTS | NAME, FORMAT, PARAMS, LAYERS, SIZE, FILES, TOK, #ADAPTERS |
| Inspector panel | EXISTS | model wells + adapter list + Reveal buttons |
| Train→ / Quantize→ / Chat→ nav | EXISTS | AppStore.use(_:on:) |
| Finder reveal (NSWorkspace) | EXISTS | |
| Adapter rank/alpha metadata | MISSING | AdapterInfo.rank/alpha/targetModules always nil; discoverAdapters() does not parse adapter config |
| Model delete | PROPOSED | no delete action in ModelsScreen |
| Model download | PROPOSED | out of scope for v1; no CLI surface exists |

### Surface 3: Training Data Curation (DataScreen.swift)

| Feature | State | Notes |
|---------|-------|-------|
| Source dir field + Scan | EXISTS | immediate + 1 level deep |
| .jsonl file enumeration | EXISTS | |
| Summary strip (FILES, ≈TOKENS, AVG LEN, TRAIN, VALID) | EXISTS | |
| HeroNumber (total examples) | EXISTS | |
| Files DataTable | EXISTS | |
| 5-example preview panel | EXISTS | parses prompt/completion or raw line |
| Builder script copy buttons | EXISTS | `uv run scripts/build_claude_lora_dataset.py` + `uv run scripts/budget_lora_dataset.py` |
| Token count accuracy | PARTIAL | approximate chars/4; NOT lattice tokenizer |
| Train/valid split detection | PARTIAL | detected by filename (train.jsonl vs valid.jsonl); no visual split editor |
| In-app example editing | PROPOSED | read-only; editing out of scope for v1 |
| Builder script execution | PROPOSED | explicitly not runnable from UI (DataScreen.swift comment) |
| Lattice tokenizer integration | PROPOSED | would need FFI or subprocess to get exact counts |

### Surface 4: Sample-Testing Chat (ChatScreen.swift)

| Feature | State | Notes |
|---------|-------|-------|
| Config strip (model + adapter pickers) | EXISTS | |
| FaderToggle BASE↔+ADAPTER | EXISTS | changes adapterPath in next GenConfig only |
| Sampling params (temperature, max-tokens, seed) | EXISTS | |
| Single-variant transcript | EXISTS | ChatTurn model; streaming via genText accumulation |
| generate_lora subprocess + `--json` | EXISTS | GenConfig → AppStore.runGenerate() |
| Streaming gen_token events | EXISTS | onChange on store.liveRun?.genText accumulates deltas |
| Non-streaming fallback | EXISTS | filters log lines for non-"$ " prefix |
| True A/B lockstep streaming | MISSING | FaderToggle flip is manual; two parallel subprocesses never run; "0 ms reload" text is hardcoded UI label (FaderToggle.swift line 137) |
| Conversation history (multi-turn) | MISSING | each submission is a fresh subprocess invocation; no history passed |
| Adapter hot-swap mid-conversation | MISSING | DESIGN.md arc_swap/LiveModel references are aspirational; no engine API exists |

---

## 5. Gaps & Risks

### G1 — No true A/B side-by-side (MISSING, HIGH)

`ChatScreen.swift` comment: "We do NOT auto-run both variants — manual flip+resend is the v1
A/B story." The FaderToggle label "0 ms reload" is hardcoded text (FaderToggle.swift line 137),
not a live measurement. Running base and adapter in parallel requires two simultaneous `RunHandle`
instances and a side-by-side transcript view. AppStore currently enforces one active run at a
time (AppStore.swift line 108: `prior.stop()`).

### G2 — quantize bins have no `--json` (STRUCTURAL, HIGH)

`quantize_q4` and `quantize_quarot` emit only human-readable stdout. There is no `--json` flag
in these bins (confirmed by absence in package-app.sh and Rust source grep). All quantize
progress depends on `HumanLineParser` regex matching. `QuantDone.est_ppl_delta` is always nil
because no Rust producer emits that field. If the quantize binary output format changes, the
parser silently degrades to `.status` passthrough.

### G3 — Adapter metadata not parsed (MISSING, MEDIUM)

`AdapterInfo.rank`, `.alpha`, and `.targetModules` are always nil (DomainModels.swift lines
63-65; LatticeBridge.discoverAdapters() confirms no config parsing). The inspector shows
adapter file size and name only. Users cannot distinguish a rank-4 from a rank-64 adapter in
the UI without inspecting the file manually.

### G4 — Multi-turn conversation not supported (MISSING, MEDIUM)

Each chat generation is a fresh subprocess invocation with a single `--prompt` string. There is
no mechanism to pass prior turns to `generate_lora`. The binary interface has no `--history`
flag.

### G5 — Token count is approximate (MEDIUM)

DataScreen uses `chars / 4` for token estimation (DataScreen.swift comment). The actual lattice
tokenizer (a BPE tokenizer) is not called from Swift. For CJK or code content the estimate
degrades significantly.

### G6 — No historical chart replay (LOW)

`RunRecord` (DomainModels.swift lines 74-85) stores only scalar summary (lastLoss, bestVal,
durationS). The point series is discarded after the run ends. RunsScreen has no chart tab.

### G7 — JetBrains Mono deferred (LOW)

Theme.swift uses `.system(design: .monospaced)` (SF Mono). DESIGN.md specifies JetBrains Mono
for the number face. A bundled font requires adding the .ttf to Package.swift resources and
updating Theme.Fonts — no external dependency needed.

### R1 — HumanLineParser fragility

The regex-based fallback for quantize output is the only parse path for those bins. Format
changes in Rust (adding a unit suffix, changing a field name) silently degrade to `.status`
passthrough with no error. Mitigation: add `--json` to both quantize bins.

### R2 — Single-process constraint blocks parallel A/B

The current `AppStore.launch()` stop-prior-on-new-launch design is correct for the current
feature set but is an architectural blocker for true A/B. Lifting this requires either a
second `handle2: RunHandle?` field and a parallel `liveRun2`, or a session abstraction.

---

## 6. Proposed Next Slices

### P0 — Add `--json` to quantize bins (1 day, Rust + Swift)

Remove the HumanLineParser fragility for the quantize surface. In `quantize_q4` and
`quantize_quarot`, add `--json` flag parsing and emit `@@lattice quant_layer` and
`@@lattice quant_done` events, including `est_ppl_delta` (compute from before/after perplexity
probes or leave as nil until measured). In Swift, the HumanLineParser stays as fallback for
older binaries. Immediate benefit: `QuantDone.est_ppl_delta` becomes populated; parser is
no longer regex-fragile; protocol is consistent across all 4 event-emitting bins.

Files: `crates/inference/src/bin/quantize_q4.rs`, `quantize_quarot.rs`, `LatticeEvents.swift`
(minimal change — struct already has the field).

### P1 — Adapter metadata from config (0.5 day, Swift only)

In `LatticeBridge.discoverAdapters()`, after discovering an adapter directory, attempt to read
`adapter_config.json` (standard PEFT format) and parse `r` → `rank`, `lora_alpha` → `alpha`,
`target_modules` → `targetModules`. If the file is absent or malformed, values remain nil
(current behavior). This unblocks ModelsScreen inspector from showing useful adapter detail and
lets TrainScreen auto-populate rank/alpha when "use existing adapter" is a future feature.

Files: `LatticeBridge.swift` (`discoverAdapters` function only).

### P2 — True A/B side-by-side chat (3-5 days, Swift architecture change)

Lift the single-run constraint for the Chat surface. Add `handle2: RunHandle?` and
`liveRun2: LiveRun?` to AppStore (or introduce a `ChatSession` model that holds two
`(RunHandle, LiveRun)` pairs). ChatScreen renders an HSplitView with two transcript columns.
The FaderToggle becomes a "run both" trigger. The "0 ms reload" hardcoded label in
FaderToggle.swift line 137 becomes a real measurement from the time between the two
`gen_token done` events. This requires the binary to support a stable `--json` streaming
interface (already true for generate_lora).

Files: `AppStore.swift`, `ChatScreen.swift`, `FaderToggle.swift`, `DomainModels.swift`.

---

*Document generated from source reads. All file paths are absolute within the lattice repo at
`/Users/lion/projects/khive/lattice/`. No build was run. All claims cite specific file and
line numbers verified in the research phase.*
