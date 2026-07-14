# Lattice Studio — Design Decision

> **Status:** Decision doc. Two refined finalists, head-to-head, with a recommendation and a build sequence.
> **Target:** macOS 26, SwiftUI, `swift build`. Toolchain verified: Apple Swift 6.3.2 / `arm64-apple-macosx26.0`.
> **Backend:** the `lattice` pure-Rust engine, driven via CLI subprocesses (line-delimited JSON event stream).
> **Brand law (governs both directions):** *measure first, the number is the truth.* Bold is spent on the **numbers**, not on chrome. Adaptive light + dark from day one.
> **Current implementation status:** The shipped app forces `.preferredColorScheme(.dark)`; light mode remains deferred relative to this decision.

---

## TL;DR

Two monochrome, numeral-first directions survived refinement. They share one thesis (the app is the **instrument panel** of the engine, not a consumer wrapper) and diverge on **density vs. speed**.

| | **PRIMARY — Lattice Instrument** | **ALTERNATIVE — Lattice / Console** |
|---|---|---|
| Angle | Trading-desk × oscilloscope, data-forward pro tool | Linear/Raycast/Zed minimal, keyboard-first |
| Hero of the screen | A 56pt tabular-mono number on opaque matte | A 40pt tabular-mono number, Cmd-K runs everything |
| Density | High (readout wells, dense tables, 3 panes) | Lean (flat config lists, command-palette spine) |
| Accent | Signal Teal `#00E5C7` / `#00A892` | Voltage Blue `#3D7BFF` / `#2C5FE0` |
| Feasibility | High | Highest |
| Distinctiveness | Strong (full instrument metaphor) | Good (risks "another dark dev tool" if type is weak) |

**Recommendation: build Lattice Instrument**, grafting Console's Cmd-K command spine. Rationale in the [comparison](#head-to-head) and [recommendation](#recommendation) sections.

Both directions adopt three shared laws, grafted from the design exploration:
1. **Numbers never touch glass.** Glass (`.glassEffect` / `.regularMaterial`) is permitted on the navigation/overlay layer only — toolbar + Cmd-K palette. Every content surface holding a numeral is **opaque**. The truth does not shimmer.
2. **Before↔after ContrastPair** for the QuaRot quantization story — a decisive two-column reveal, never a buried claim.
3. **Verdict as text, not vibe** — every measurable action ends in a GATE PILL stating the result (`PASS` / `WARN` / `FAIL`).

---

# PRIMARY — Lattice Instrument

**Tagline:** *Measure first. The readout is the truth.*

## Philosophy

lattice owns hand-written Metal/AVX2/NEON kernels with zero framework — no Python, no ONNX, no CUDA, no runtime. The app should look like the **instrument panel of that engine**, not a wrapper around it. So every numeric quantity (loss, grad-norm, tok/s, PPL, MB, rank, lr) is a first-class citizen in tabular monospace that never reflows as digits change, on hairline-ruled opaque panels with almost no chrome — the way a DAW or a Bloomberg terminal trades whitespace for live truth.

One accent (signal teal) is spent only on **movement** — the loss trace, the active token stream, the now-cursor — so the eye always lands on the thing that is actually changing. It follows an editorial/high-contrast design direction by making **bold typography the hero** (a 56pt loss numeral, luminous data on ink) instead of illustration, and it beats a generic dashboard because it commits fully to the instrument metaphor: readouts, sweeps, gates — the literal visual translation of measure-first DNA.

The README is explicit that MLX is faster; lattice's edge is **capabilities + honesty**. So we sell the moat, not raw speed: QuaRot 4-bit, Q4+LoRA hot-swap with no reload, pure-Rust portability.

## Visual Identity

### Palette

This design specifies dark and light semantic color tokens. The shipped app currently locks
`preferredColorScheme(.dark)`, so dark is the only active appearance and light implementation is
deferred. The table below retains the intended paired values.

| Token | Dark hex | Light hex | Usage |
|---|---|---|---|
| Canvas / Panel face | `#0A0B0D` Vantablack | `#F7F8FA` Quartz | Primary background, the instrument face. **Opaque.** |
| Panel Raise | `#121419` | `#FFFFFF` | Elevated center panel + readout-well body (one step up). |
| Well Sink | `#070809` | `#ECEEF2` | Recessed fill **inside** a readout well (below the face). |
| Hairline | `#23262E` | `#DCDFE5` | 1px rules: dividers, grid, table separators, well borders. Depth via line. |
| Ink (text + numerals) | `#E8EAED` | `#14161A` | Primary text/numerals. ~14.5:1 dark / ~15:1 light. |
| Ink Dim | `#7C828D` | `#5C636E` | Labels, units, axis ticks, metadata. ~4.8:1 / ~5.2:1. |
| **Signal Teal** (accent) | `#00E5C7` | `#00A892` Teal Deep | Live trace, token stream, now-cursor, the **one** primary CTA/screen, focus ring. The only color that shifts value between modes (to hold ~4.5:1). |
| Teal Glow | `#00E5C7 @12%` | `#00A892 @12%` | Area-fill gradient under the live trace; under-glow behind newest segment. |
| Amber Caution | `#FFB020` | `#FFB020` (value-shifted) | Regression/warning: loss rising, grad-norm spike, PPL over budget, mem pressure. Replaces red for colorblind safety. |
| Crimson Halt | `#FF4D5E` | `#FF4D5E` (value-shifted) | Hard failure only: NaN loss, OOM, run crashed, parity FAIL. Rare by design. |

**WCAG AA is a hard floor:** hero Ink-on-panel ≥14:1, dim labels ≥4.5:1, teal-on-its-ground ≥4.5:1. Amber/Crimson keep hue and shift lightness for AA on each ground.

### Typography

Three axes, hard separation by role. **Mono is the signature** — the eye registers "this is an instrument" from the numerals alone.

- **DISPLAY / TITLES** — SF Pro Display (system), Bold/Heavy. Screen titles 17pt Semibold (-0.01em).
- **HERO NUMBER** — the current loss / compression ratio / PPL at **56pt Bold, -0.02em**, rendered in **JetBrains Mono tabular** (not the grotesque) so the headline number never jitters. This is the one place display scale and mono fuse.
- **NUMERALS + DATA** — JetBrains Mono (bundled, OFL), `.monospacedDigit()` always on. Sizes: 11pt table cells / 13pt dense readouts / 15pt readout-well value / 21pt secondary hero / 56pt hero. Units 11pt.
- **BODY / LABELS** — SF Pro Text 13pt Regular for prose; 11pt Medium ALL-CAPS +0.06em for instrument labels and units (`LOSS`, `TOK/S`, `GRAD-NORM`, `ΔPPL`). Max prose measure 640px.

Modular scale (pt): **11 · 13 · 15 · 21 · 34 · 56.**

### Spacing & Grid

8pt base grid; dense 4pt variant for table rows (28px row height, 32px "comfortable" toggle). Three-region shell, panels butt against 1px Hairline with **no gaps**: **LEFT RAIL 220px / fluid CENTER / RIGHT INSPECTOR 300px** (collapsible to 0 with `⌘\`). Internal panel padding 16px; section gutter 24px.

Readout wells: inset 1px Hairline border + Well-Sink fill + a 2px inner top-shadow (`rgba(0,0,0,0.5)` dark / `0.06` light) so they read machined-in. **Corner radius:** `0px` on panels and table cells (panels are ruled, not carded), `6px` on wells/controls/gate pills, `10px` on the floating command bar. Data tables go full-bleed. **One elevation step only — no card-on-card.**

### Materials & Motion

**Governing law: numbers never touch glass.** Glass is permitted on the navigation/overlay layer only (top toolbar + the floating ⌘K command bar use thin `.regularMaterial`). Every content surface — panels, wells, tables, the strip chart, transcripts — is opaque. This also dodges the Liquid-Glass perf caveats, because data is solid.

Motion is mechanical, never bouncy:
1. **Numerals tick per-digit** via `.contentTransition(.numericText())` at ~120ms — throttled to 8Hz on fast values (tok/s) so it never looks frantic.
2. **Loss curve draws as an oscilloscope sweep** (left→right) with a 1px teal now-cursor and a 12% teal under-glow; chart commits throttled to ~20Hz with rolling-window downsample.
3. **Panel focus** traces a 1px teal ring in 180ms ease-out.
4. **Spring (response 0.32, damping 0.85)** is reserved for exactly **one** element: the adapter hot-swap fader, so it feels like throwing a hardware switch.

Durations 120/180/240ms ease-out. No parallax, no decorative particles. Respects `accessibilityReduceMotion` (ticks→crossfade, sweep→instant) and `accessibilityReduceTransparency` (glass→solid material, same teal).

## Component Language

Instruments, not widgets. Eleven primitives, opaque unless noted.

1. **READOUT WELL** — the atomic unit: inset hairline block (Well-Sink fill, 6px radius, inner top-shadow) holding one 11pt ALL-CAPS dim label + one tabular-mono value (15pt) + unit + optional delta caret (▲/▼ teal=good/amber=bad). For loss, lr, grad-norm, tok/s, ETA, size, PPL.
2. **HERO NUMBER** — 56pt JetBrains-Mono-tabular value with a tiny mono unit and a 1px teal under-rule; ticks per-digit. The visual anchor of TRAIN and QUANTIZE.
3. **STRIP CHART (oscilloscope)** — Swift Charts `LineMark` (1.5px teal) + `AreaMark` (12% teal gradient) + `RuleMark` now-cursor; mono axis ticks, hairline grid, no legend. A draggable scrub line freezes every readout well to that step. On QUANTIZE it carries a **ghost base-PPL line** so the delta is a visible gap, not a claim.
4. **CONTRAST PAIR** — the before↔after comparator: two stacked READOUT WELL columns (size/bits/PPL) with a centered Δ chip; on completion the columns slide apart and a single hairline **"fold" wipe** (the QuaRot rotation metaphor) reveals the after-state.
5. **MASS BARS** — two horizontal bars to **true scale** (fp16 "before" Ink-dim / Q4 "after" teal) that animate to length while the compression ratio counts up into a HERO NUMBER. Lives directly under the CONTRAST PAIR on QUANTIZE.
6. **DATA TABLE** — 28px rows (32px comfortable), tabular-mono right-aligned numerals, hairline row rules, **no zebra**, sortable 11pt all-caps headers, selected row marked by a 2px teal left-border (no fill flood).
7. **PARAM ROW** — label (left, dim) + control (right) on one hairline-ruled line; sliders show their value as a live mono readout in the track. Forms are stacks of these, never boxed.
8. **GATE PILL** — status capsule encoding the verdict: `PASS` (teal) / `WARN` (amber) / `FAIL` (crimson) / `RUN` (animated teal pulse). e.g. `Q4 ok 405MB −74% · est.ΔPPL +0.09 PASS`.
9. **FADER TOGGLE** — the adapter A/B switch styled as a console fader (the one spring element).
10. **KEY-CAP CHIP** — a tiny 1px-outlined ⌘-cap on actionable elements so the keyboard map self-documents.
11. **COMMAND BAR** — a floating ⌘K mono palette (the one glass slab) where `train qwen3.5 r8` or `quantize quarot` parse into argument chips and fire a run without touching a form. *(Grafted from Console — see recommendation.)*

Buttons: rectangular, 6px radius, 1px-bordered hairline default; exactly **one** teal-filled primary per screen. No rounded cards, no drop shadows except the recessed-well inner shadow.

## Information Architecture

Three-pane instrument console (`NavigationSplitView`).

- **LEFT RAIL (220px, persistent):** wordmark `LATTICE` in mono + build hash; a pinned top **RUN block** (live job: model · step · loss · tok/s, or `idle`); vertical nav with mono index labels — `01 MODELS`, `02 TRAIN`, `03 QUANTIZE`, `04 CHAT`, `05 DATA`, `06 RUNS`, each with a ⌘1–⌘6 key-cap chip; pinned at the bottom a live **SYSTEM STRIP** (unified-memory bar, GPU%, active-run mini-readout) always reading out.
- **CENTER (fluid):** the active instrument screen.
- **RIGHT INSPECTOR (300px, contextual, collapsible `⌘\`):** details/config for the current selection.

### Screens

- **01 MODELS** — DATA TABLE of `~/.lattice/models` (name, params, dtype, format, size GB, files) with adapters indented under each base model (verified on-disk: `qwen3.5-0.8b` + `-q4` + `-q4-quarot`, plus `qwen3.5-2b`; embedding models — `all-minilm-l6-v2`, `bge-small-en-v1.5`, `qwen3-embedding-0.6b`, the e5/minilm multilingual pair — filtered to a sub-tab). Inspector shows the file manifest, `config.json` as readout wells (the **18-GDN / 6-GQA** layer split called out explicitly, vocab, ctx), and Download / Verify / Reveal-in-Finder. QuaRot weights get a `rotated` badge.
- **02 TRAIN** — left PARAM ROWS (model, target layers, rank r8, scale α, lr, steps, dataset). Center HERO NUMBER (live loss, ticking) over the oscilloscope STRIP CHART (train_loss + val_loss + grad-norm overlay, teal trace, now-cursor), with a row of READOUT WELLS (step, lr now, grad-norm, tok/s, ETA). Inspector: live stdout console + a GATE PILL on `best_val_loss` verdict. Scrub the curve to freeze all wells at any step.
- **03 QUANTIZE** — pick model + method (Q4 / QuaRot toggle, QuaRot badged `rotated`). Center the CONTRAST PAIR (size/bits/est.PPL with Δ) over MASS BARS (true-scale fp16→Q4) over a progress sweep, terminating in a VERDICT GATE PILL. Inspector: per-layer quant table (layer, scheme, error).
- **04 CHAT** — split A | B columns (base vs base+adapter) streaming the SAME prompt token-by-token in teal; the adapter selector is the HOT-SWAP FADER with a `0 ms reload` stamp; sampling READOUT WELLS (temp, top-p, top-k, seed) in the inspector; per-column tok/s + TTFT so testing is also measuring.
- **05 DATA** — left source files; center a `{prompt, completion}` pair DATA TABLE with a validate GATE PILL (token-count histogram, malformed-row count); inspector single-pair editor + export-to-JSONL that writes straight into the Train dataset field.
- **06 RUNS** — archive DATA TABLE (status pill, kind, model, last loss/best val, dur, when); selecting a row loads its config back into the relevant screen and shows its frozen strip chart — the lab notebook.

## Signature Interactions

1. **LIVE LOSS OSCILLOSCOPE + SCRUB-TO-FREEZE** — the TRAIN strip chart draws as a real-time left→right sweep with a 1px teal now-cursor and 12% under-glow; the 56pt hero loss numeral above it ticks digit-by-digit. Drag the cursor backward and **every** readout well rewinds to that step's exact lr / grad-norm / tok-s — inspect the precise state at the moment loss spiked. Only possible because wells are wired to the real per-step `EpochMetrics` / `AdaptStepResult` stream, not a redraw.
2. **HOT-SWAP FADER (the moat made physical)** — in CHAT the adapter selector is a console fader. Sliding it from BASE to BASE+LoRA r8 swaps the adapter with **no reload** — the B column visibly re-streams the same prompt under the new weights in real time and a teal `0 ms reload` stamp confirms it. The one place spring physics is allowed. Both columns share one prompt bar and stream in lockstep.
3. **THE VERDICT GATE + ⌘K SPINE** — every measurable action ends in a GATE PILL stating the truth as text (`Q4 ok 405MB −74% · est.ΔPPL +0.09 PASS`, flips amber if ΔPPL exceeds the honest budget); the CONTRAST PAIR completes with a single hairline "fold" wipe as the MASS BARS settle. ⌘K opens a glass mono palette where `quantize quarot qwen3.5` parses into argument chips and fires the run without a form.

## ASCII Mockups

```
MAIN SHELL — 02 TRAIN (dark) ---------------------------------------
+------------+--------------------------------------+---------------+
| LATTICE    | 02 TRAIN  qwen3.5-0.8b · lora r8  ⌘K | STDOUT     ⌘\ |
| ·a3f9c1    +--------------------------------------+ step 0420...  |
| ● RUN      |  L O S S                             | loss 0.612    |
|  step 420  |  0.6121  ▼0.004   step 420/1000      | lr 1.81e-4    |
| 01 MODELS ⌘1| +----------------------------------+ | grad 0.93     |
| 02 TRAIN  ◀ | |\__                            .· |◀| tok/s 1820    |
| 03 QUANTIZE| |   \\__               ..·         | |---------------|
| 04 CHAT   ⌘4| |      \\\___\__/\___·····  now►   | | GATE          |
| 05 DATA   ⌘5| +----------------------------------+ | [▣] eval PASS |
| 06 RUNS   ⌘R|  STEP    LR-NOW   GRAD   TOK/S  ETA | | best val 1.94 |
|············|  0420    1.81e-4  0.93   1820   4m  | +---------------+
| MEM ████░  | +- PARAMS ------------------------+ |               |
| 11.2/16 GB |  rank r8 · α16 · lr 1.8e-4 · 18 GDN |               |
| GPU 74%    |  dataset claude-lora.jsonl 4,812 rw | [ ■ STOP ]    |
+------------+--------------------------------------+---------------+

03 QUANTIZE -------------------------------------------------------
| 03 QUANTIZE  qwen3.5-0.8b   method:( Q4 )[ QuaRot ⟳ ]            |
|   BEFORE          ──fold──▶          AFTER       VERDICT         |
|   SIZE 1.61 GB                       405 MB      ▼ −74%          |
|   BITS 16                            4           rotated ok      |
|   PPL  15.86 (ghost)·····            15.95 est.  Δ+0.09 [▣]PASS  |
|   fp16 ████████████████████████  Q4 ███████   3.97× smaller     |
|   ███████████████░░░░ rotate→pack→verify  layer 18/24  sweep►    |
```

## Per-Feature Surfacing

- **1. LoRA Training → 02 TRAIN.** PARAM ROWS configure model / target-layers (18 GDN + 6 GQA selectable strip) / rank r8 / scale α / lr / steps / dataset, slider values shown as live mono readouts. Watched via a 56pt HERO loss numeral (ticking) over the oscilloscope STRIP CHART (train_loss + val_loss + grad-norm overlay) and READOUT WELLS for step / lr-now / grad-norm / tok-s / ETA — wired to the real `EpochMetrics{train_loss, val_loss, learning_rate, examples_seen, duration_secs}` plus per-step `AdaptStepResult{loss, grad_norm}` streamed as line-delimited JSON into an `@Observable` store. Inspector streams raw stdout and shows the eval GATE PILL on `best_val_loss`. Scrub the curve to freeze all wells at any step.
- **2. Quantization → 03 QUANTIZE.** Pick model + a Q4 / QuaRot method toggle (QuaRot badged `rotated ⟳`). A CONTRAST PAIR shows size / bits / est.PPL with the Δ in teal(improve)/amber(regress); below it MASS BARS animate to true scale (fp16 1.61GB → Q4 405MB) while the 3.97× ratio counts up; a progress sweep labels real phases (rotate→pack→verify); a final VERDICT GATE PILL states the result. The strip carries a ghost base-PPL line so quality cost is a visible gap. PPL is explicitly labeled `est.` unless a calibration set is supplied (measure-first honesty). Inspector: per-layer quant table.
- **3. Model Management → 01 MODELS.** Dense DATA TABLE of `~/.lattice/models` with adapters indented under their base model — reflecting the real on-disk layout. Tabular-mono right-aligned numerals, sortable headers, no zebra. Inspector reveals the full file manifest and `config.json` as readout wells (18-GDN / 6-GQA split explicit), with Download / Verify / Reveal-in-Finder. QuaRot weights get a `rotated` badge; the adapter row shows a `swappable · no reload` chip.
- **4. Chat / Sample Testing → 04 CHAT.** Two-column A|B layout streams the SAME prompt token-by-token (teal) through base vs base+adapter from ONE shared prompt bar. The adapter selector is the HOT-SWAP FADER (`0 ms reload` stamp); sliding it re-streams the B column live. Sampling controls (temp/top-p/top-k/seed) are inspector READOUT WELLS; each column shows live tok/s + TTFT so sample-testing doubles as measurement. Single-column mode collapses to one full-width transcript.
- **5. Dataset Prep → 05 DATA.** Source files on the left; center a `{prompt, completion}` pair DATA TABLE with a validate GATE PILL summarizing a token-count histogram and malformed-row count (tokenized with lattice's own tokenizer so counts match training exactly). Inspector is a single-pair editor; export writes JSONL straight into the Train screen's dataset field — raw to training-set, previewed and validated before it ever feeds a run.

---

# ALTERNATIVE — Lattice / Console

**Tagline:** *Measure first. Every number, one keystroke away.*

## Philosophy

lattice is an instrument, not an app. An oscilloscope doesn't decorate the waveform; it gets out of the way so the engineer trusts the trace. Console is near-monochrome so the one thing that earns color and movement — a live loss ticking down, a PPL delta, a parity PASS — is unmissable and pre-attentive. Everything is reachable from a single command bar (⌘K), because a tool that worships "measure first" should never make you hunt through menus for the measurement.

It follows the editorial/bold design direction by spending all of that boldness on the **numerals themselves**, not on chrome — the visual analog of pure-Rust with no Python, no ONNX, no runtime bloat. Same "numbers never touch glass" law; glass lives only on the navigation layer.

## Visual Identity

### Palette

Four elevation levels resolved per appearance: base(L0) → panel(L1) → raise(L2) → overlay. Dark is default + native habitat; light is a true warm-paper theme, not an inverted hack. Shadows appear **only** in light (single y2/blur8/6% step under cards); dark relies on borders alone.

| Token | Dark hex | Light hex | Usage |
|---|---|---|---|
| Base canvas (L0) | `#0B0C0E` Ink | `#FBFBFA` Paper (warm) | Window canvas, deepest surface. |
| Panel (L1) | `#131519` Slate-900 | `#FFFFFF` Surface | Sidebar, cards, metric rail, table body. |
| Raise (L2) | `#1B1E24` Slate-800 | `#F2F1EE` | Hover, selected-row base, input wells. |
| Hairline | `#262A31` | `#E6E4E0` | 1px inset borders + dividers + chart grid. |
| Text hi (+ hero numerals) | `#ECEEF1` | `#16181C` | Primary text/numerals. |
| Text mid | `#9AA1AC` | `#5E636B` | Secondary labels, units, ticks. |
| Text dim | `#5B626D` | (derived) | Tertiary / placeholder / disabled / key-cap outline. |
| **Voltage** (accent) | `#3D7BFF` | `#2C5FE0` Voltage-deep | Focus ring, the one primary CTA/screen, live-metric line, ⌘K selection, selected-row left-border. Rationed. ~4.6:1 on each base. |
| Signal-pass | `#2FB079` | `#2FB079` | Correctness only: parity PASS, quant OK, converging caret. Paired with a glyph (✓ / ▼). |
| Signal-warn | `#D9A227` | `#D9A227` | Drift / PPL regression / OOM-risk / loss-rising caret. Paired with ▲. |
| Signal-fail | `#E5523C` | `#E5523C` | Parity FAIL, NaN loss, crash. Rare; paired with ✕. |

Every signal color pairs with a redundant glyph for colorblind safety; signal hues shift lightness (not hue) to clear ~4.5:1 on both grounds.

### Typography

Three roles; the proportional-vs-monospace contrast **is** the identity (it signals "this region is a measurement").

- **DISPLAY = the numbers**, tabular-figure monospace (SF Mono on-system, JetBrains Mono bundled fallback for exact metrics; `.monospacedDigit` always). HERO 40pt/600 (loss, PPL, tok/s on Train + Quantize); SECTION 22pt/600 (metric tiles); INLINE 13pt/500 (table cells, config values). Hero gets **-0.01em tracking** with the unit as an 11pt mid-tone superscript-baseline label, so the number reads as authored even at rest (fixes the "console echo" risk).
- **BODY/UI = SF Pro Text** — 13pt regular for labels + prose, 11pt medium UPPERCASE micro-labels naming each metric (+0.06em: `LOSS · TOK/S · GRAD-NORM`).
- **MONO-SMALL = SF Mono 11.5pt** for config values, file paths, log console, JSONL preview cells (columns must align).

Modular scale (1.25): **11 / 13 / 16 / 22 / 28 / 40.** No serif anywhere.

### Spacing & Grid

8pt base, 4pt half-step for dense rows. Three-pane shell on a fixed rhythm: **sidebar 232pt · detail flexes · right rail 320pt** (collapsible `⌘/`). Row height 28pt dense / 32pt comfortable (Settings toggle — the named mitigation for 13in density). Card padding 16pt; gutters 24pt; outer margin 32pt. Density-without-clutter is bought with whitespace **between** blocks, never within. Corner radius: 6pt controls, 10pt cards, 14pt ⌘K palette/modals. Min window 1080×720.

### Materials & Motion

Restraint-first, governed by the same law: **glass only on the nav layer.** Liquid Glass on exactly two surfaces — the floating toolbar and the ⌘K palette slab over a dimmed canvas. All content is solid fill. Motion is functional: 120ms focus-ring + row-selection, 180ms pane collapse, 200ms ⌘K slab. The **one signature motion** — the live hero numeral uses `.contentTransition(.numericText())` so a dropping loss ticks digit-by-digit (~90ms); the delta caret crossfades color/direction in the SAME beat, and the loss curve extends one accent segment in the SAME beat — **three surfaces, one pulse.** No bounce, no parallax. Reduce-Motion → instant swaps + crossfade. Reduce-Transparency → glass → solid material (free via system).

## Component Language

Spare, line-based, table-centric — eight primitives + two control specials.

1. **COMMAND BAR** — the spine: a single ⌘K glass palette running everything (start training, quantize, switch model, open chat, jump to a run) with fuzzy match, recent actions, and inline **argument chips** (`train qwen3.5 r8` parses `r8` into an editable rank chip).
2. **METRIC TILE** — uppercase 11pt micro-label + 22–40pt tabular-mono number + delta caret (▼ pass / ▲ warn); the only place numbers go big.
3. **SPARK + FULL CHART** — Swift Charts `LineMark`, 1.5px accent stroke on transparent grid, hairline axes, no fills/legend chrome; hover scrubs a `RuleMark` crosshair with mono readout, freezing every metric tile to that step.
4. **DENSE DATA TABLE** — 28pt rows, mono value columns right-aligned, single accent left-border on selected row (no fill flood), sortable 11pt uppercase headers, zebra-free.
5. **CONFIG ROW** — label left, mono editable value right, inline validation dot; forms are flat lists, never boxed.
6. **KEY-CAP CHIP** — every actionable element shows its shortcut as a tiny outlined ⌘-cap (self-documenting keyboard map).
7. **STATUS PILL** — `PASS` / `RUNNING` / `FAILED` in semantic colors + glyph.
8. **CONTRAST PAIR** — left↔right split with a centered Δ chip for any before↔after; two values animate apart, Δ resolves pass/warn. Used for QuaRot and base↔adapter.

Control specials: the **HOT-SWAP FADER** (the Chat adapter A/B; the one place a spring is allowed) and the **TRUTH TOGGLE** (`⌘.`) — dims all nav glass so only opaque data remains. Buttons: mostly tertiary (text + key-cap); exactly **one** `.glassProminent` accent button per screen.

## Information Architecture

Three-pane `NavigationSplitView`.

- **LEFT SIDEBAR (232pt)** — flat, keyboard-navigable, no nested disclosure: a pinned top **RUN STATUS block** (live job: model · step · loss · tok/s + pause/stop, or `idle`), then `TRAIN ⌘1 · QUANTIZE ⌘2 · MODELS ⌘3 · CHAT ⌘4 · DATA ⌘5`, then a footer `RUNS ⌘R` + `Settings ⌘,`.
- **CENTER** — active screen.
- **RIGHT RAIL (320pt, `⌘/` collapse)** — context inspector.

### Screens

- **Runs / Home** — dense table of every run (status pill, model, kind LoRA/Q4/QuaRot, last-loss/PPL, duration, when) — the lab notebook; ⌘-click any row reopens its exact config.
- **Train ⌘1** — center = flat config list (model picker; an interactive **LAYER STRIP** rendering the 18 GatedDeltaNet + 6 GQA layers as selectable cells; rank, scale α, lr, steps, dataset); one accent `Start ⌘↵`. Right rail = live loss curve + metric tiles (loss/lr/grad-norm/tok-s/ETA) + scrolling log console.
- **Quantize ⌘2** — pick model + method segmented toggle Q4 / QuaRot (QuaRot framed as the premium moat); a CONTRAST PAIR before→after size (1.61GB → 405MB) animating apart with a "fold" wipe + Δ chip; progress strip with phase labels (rotate → pack → verify); a PPL/quality readout base→quant with a green/warn signal when a calib set is supplied.
- **Models ⌘3** — master table of `~/.lattice/models` + adapters (name, dtype, format, size, file count, status pill); right-rail inspector = full file manifest + actions Load / Hot-swap adapter / Reveal in Finder / Delete. Download/import is a ⌘K action.
- **Chat ⌘4** — streaming transcript; split-compare mode = base vs base+adapter in two lockstep-streaming columns; right rail = temp/top-p/max-tokens + the live HOT-SWAP FADER (`⌥A`).
- **Data ⌘5** — two-pane raw→`{prompt,completion}` builder: raw input left, JSONL preview table right + inline validator (row count, malformed flags, token-length histogram); export feeds the Train dataset picker.
- **Settings** — appearance (System/Light/Dark), row density, models dir, default hyperparams.

## Signature Interactions

1. **HOT-SWAP FADER, the moat made physical** — grounded in the real engine: `LiveModel` uses `arc_swap::ArcSwap` for lock-free atomic `swap()`. In Chat split-compare the right column carries a console fader. Slide BASE → BASE+LoRA-r8 and the engine swaps the adapter with no reload; the column flashes its accent left-border 120ms, a `0 ms reload` stamp confirms it, and the very next streamed token comes from the new adapter. The one place a spring (0.32) is allowed.
2. **THE NUMBER IS THE TRUTH, three surfaces one heartbeat** — on every `TrainingCallback.on_batch_end` the hero loss numeral ticks digit-by-digit (~90ms), the delta caret flips pass/warn in the SAME 90ms, and the loss curve extends one accent segment in the SAME beat. "Is it converging?" is answered pre-attentively without reading an axis. Scrub the crosshair and all three plus every metric tile freeze to that step.
3. **COMMAND BAR AS THE WHOLE APP** — ⌘K opens a glass palette where `train qwen3.5 r8` or `quantize quarot` parse into argument chips; you configure and launch a full run without touching a form. A persistent ⌘-cap legend means power users never reach for the mouse; every action also has a visible affordance. Bonus: `⌘.` Truth Toggle dims nav glass so only the opaque data canvas remains.

## ASCII Mockups

```
SHELL — Train (dark)
┌────────────┬───────────────────────────────┬─────────────────────────┐
│ ● RUN      │  Train · Qwen3.5-0.8B          │  LIVE METRICS      ⌘/   │
│ qwen3.5    │  ───────────────────────────  │  LOSS                   │
│ step 412   │  model    Qwen3.5-0.8B   ▾     │   0.612 ▼  step 412/1k  │
│ loss 0.612 │  layers  [GDN×18][GQA×6] ◧     │  ╭───────────────────╮  │
│ 184 tok/s  │  rank    8     scale α  16     │  │ ╲__               │  │
│  ⏸  ⏹      │  lr      2e-4  steps   1000    │  │    ╲────╲___       │  │
│────────────│  dataset claude_set.jsonl     │  ╰───────────────────╯  │
│ TRAIN    ⌘1│                                │  lr 2e-4   grad 0.91    │
│ QUANTIZE ⌘2│  ┌─────────────────────────┐   │  tok/s 184  eta 2m04s   │
│ MODELS   ⌘3│  │   Start run        ⌘↵   │   │  ── log ──────────────  │
│ CHAT     ⌘4│  └─────────────────────────┘   │  step 412 loss 0.612    │
│ DATA     ⌘5│                                │  step 411 loss 0.618    │
│────────────│                                │  ckpt saved @ 400       │
│ RUNS  ⌘R   │                                │                         │
└────────────┴───────────────────────────────┴─────────────────────────┘
 ⌘K command · ⌘/ rail · ⌘. truth · ↑↓ nav · ⏎ open

CHAT split-compare — hot-swap fader (base vs +adapter)
┌──────────────────────────────┬──────────────────────────────┐
│ BASE  Qwen3.5-0.8B           │ +ADAPTER  claude-r8  0ms swap │
│ The capital of France is     │ The capital of France is     │
│ Paris, a city known for▏     │ Paris — bonjour. As Claude▏  │
├──────────────────────────────┴──────────────────────────────┤
│ BASE ◁───────●────▷ +LoRA   temp 0.7 top-p 0.9  ⌘↵ send ⌥A │
└──────────────────────────────────────────────────────────────┘
```

## Per-Feature Surfacing

- **1. LoRA Training → Train ⌘1.** Flat config list (no nested cards): model picker, interactive LAYER STRIP (real 18 GatedDeltaNet + 6 GQA cells), rank, scale α, lr, steps, dataset. One rationed accent `Start ⌘↵`. Right rail streams the live loss curve + metric tiles (fed by `TrainingCallback.on_batch_end` / `on_epoch_end`) + a scrolling stdout log; `on_checkpoint` drops a "ckpt saved" line. Pause/stop in the pinned sidebar RUN block. Also launchable via ⌘K with argument chips.
- **2. Quantization → Quantize ⌘2.** Method as a segmented toggle Q4 vs QuaRot (QuaRot framed as the premium moat). A CONTRAST PAIR shows before→after size animating apart with a single "fold" wipe + Δ chip (−75%); a progress strip names the phase (rotate → pack → verify). When a calibration set is supplied a PPL/quality readout shows base→quant delta (15.86 → 15.92) with a pass/warn signal so quality is honest, never hidden. Mirrors the real `-q4` and `-q4-quarot` artifacts on disk.
- **3. Model Management → Models ⌘3.** Dense master table of `~/.lattice/models` AND adapters in one view: name, dtype, format, size, file count, status pill; tabular-mono columns keep sizes/shapes aligned. Right-rail inspector shows the full file manifest (18-GDN/6-GQA split from `config.json`) plus Load / Hot-swap adapter / Reveal in Finder / Delete. Download/import is a ⌘K action. QuaRot weights get a `rotated` badge.
- **4. Chat / Sample Testing → Chat ⌘4.** Signature split-compare: base vs base+adapter in two lockstep-streaming columns driven off two AsyncStreams throttled to a shared clock. Sampling controls (temp/top-p/max-tokens) in the right rail; the HOT-SWAP FADER (`⌥A`) does a live no-reload swap via the engine's `arc_swap` LiveModel with a `0 ms reload` stamp. Single-column mode for plain testing.
- **5. Dataset Prep → Data ⌘5.** Two-pane builder: raw input (paste / file) left, the derived `{prompt, completion}` set as a JSONL preview table right. Inline validator reports row count, malformed-row flags, and a token-length histogram (using lattice's own tokenizer so counts match training). Export writes straight into the Train dataset picker, closing the loop.

---

# Head-to-Head

| Criterion | Lattice Instrument | Lattice / Console | Edge |
|---|---|---|---|
| **Sleek / modern** | 9 — full instrument metaphor, recessed wells, oscilloscope | 9 — ruthless Linear-class restraint | tie |
| **Brand fit** | 10 — most literal "measure first" translation | 9 — same thesis, leans on a known aesthetic | Instrument |
| **Feature surfacing** | 9 — dedicated comparator/quant-table screens, denser readouts | 8 — slightly thinner; config lists + right-rail metrics | Instrument |
| **Feasibility** | 9 — all native APIs; density redraw is the only watch-item | 10 — least surface area, glass on nav only | Console |
| **Distinctiveness** | 8 — instrument/Bloomberg metaphor is on-brand but not category-novel | 7 — risks "another dark dev tool" without an exceptional type pass | Instrument |
| **Density posture** | High by default (with a comfortable toggle) | Lean by default (Cmd-K hides forms) | open design choice |

**When each wins.** Instrument wins when the user lives *inside* a run — watching loss, scrubbing the curve, reading six wells at once — and wants the screen to be a panel of live truth. It is the richer demo and the more defensible brand statement. Console wins on pure speed and minimal build risk: a power user who configures by keystroke and never wants to see a form, on the cheapest path to ship. Console's distinctiveness is its only soft spot — near-monochrome reads generic unless the hero numerals are genuinely authored.

The crucial point: **they are not opposites.** Console's Cmd-K command spine is the single best IA primitive in the whole exploration, and it slots cleanly into Instrument as primitive #11. The real fork is *density default*, not *which thesis*.

---

# Recommendation

**Build Lattice Instrument, with Console's ⌘K command spine grafted in from day one.**

Why:
1. **Brand fit is the tiebreaker and Instrument wins it 10 vs 9.** lattice's identity is literally "the number is the truth," and Instrument is the most complete translation of that into UI — readout wells, the oscilloscope strip chart with scrub-to-freeze, GATE PILL verdicts. It sells the *moat* (QuaRot ContrastPair, hot-swap fader) rather than raw speed, exactly as the README demands.
2. **The feasibility gap is small and closeable.** Instrument is a 9 to Console's 10. Its only real risk is 60fps redraw of the live curve under a fast step/token stream — mitigated by buffering points and throttling chart commits to ~20Hz, the same technique both directions already specify. Everything else maps to native APIs (`.monospacedDigit`, `.contentTransition(.numericText())`, Swift Charts `LineMark`+`RuleMark`+`chartOverlay`, `NavigationSplitView`). Verified buildable on the installed Swift 6.3.2 / macOS 26 toolchain.
3. **Grafting closes the only thing Console wins on.** Adding the ⌘K spine to Instrument gives power users the keyboard-first launch path without sacrificing the dense readout panels. The comfortable-row toggle (32px) handles the density-too-cold risk Instrument self-flags. We get Console's speed *and* Instrument's brand depth.
4. **Honors the editorial/bold design lean correctly.** Both reject the literal display-serif reading (a streaming loss in a 96pt serif visibly jitters — it contradicts "the truth does not shimmer"). Instrument spends all its boldness on a 56pt tabular-mono hero numeral: editorial confidence without the jitter risk.

The override on the "editorial/bold" lean is deliberate and stated: editorial is a *typographic posture*, not a literal magazine. We deliver it as **monospace-numeral boldness**, which is both more on-brand for an instrument and lower-risk to build. If the magazine-cover feel is specifically wanted, that is the genuine fork below.

---

# What Gets Built First

`apps/macos/` is greenfield. Sequence for a SwiftUI macOS app driven by lattice CLI subprocesses:

1. **SwiftPM target + shell.** Create `apps/macos/` SwiftPM executable target that `swift build` compiles. Stand up the `NavigationSplitView` three-pane shell, the left rail with static nav, and the asset-catalog semantic colors (both appearances, all elevation tokens). Goal: an empty but adaptive shell that launches.
2. **The opaque-panel + numeral kit.** Build the shared primitives that everything depends on: `OpaquePanel` container (enforces "numbers never touch glass"), `ReadoutWell`, `HeroNumber` (`.contentTransition(.numericText())`), `DataTable`, `GatePill`, `KeyCapChip`. Bundle JetBrains Mono. This is the visual identity made real.
3. **CLI bridge + event store.** A subprocess runner that spawns the lattice `tune` / `quantize` / `chat` bins and parses their line-delimited JSON `stdout` into an `@Observable` run store. Map JSON 1:1 onto the real serde structs (`EpochMetrics`, `AdaptStepResult`, `TrainingMetrics`). **Critical:** own the run `@State` *above* the view and key by run-id so a re-render never resets a live run.
4. **01 MODELS (read-only first).** The simplest real-data screen: list `~/.lattice/models`, parse `config.json` into readout wells. Proves the bridge and the table primitive against real on-disk artifacts before any long-running job.
5. **02 TRAIN (the flagship loop).** Wire PARAM ROWS → CLI launch → live HERO NUMBER + STRIP CHART + READOUT WELLS off the event store. Add scrub-to-freeze. This is the demo centerpiece; get the ~20Hz throttle right here.
6. **⌘K command spine.** The grafted Console primitive: a glass palette that parses `train qwen3.5 r8` into argument chips and fires step 5's launch path. Add the ⌘1–6 key-cap legend.
7. **03 QUANTIZE.** CONTRAST PAIR + MASS BARS + VERDICT GATE PILL over the quantize bin's progress stream. PPL hard-labeled `est.` unless a calib set is supplied.
8. **04 CHAT + HOT-SWAP FADER.** Dual-stream A|B columns; bind the fader to the engine's `arc_swap` `LiveModel.swap()` over the bridge. **Verify the FFI/CLI event surface exposes a structured no-reload swap before committing the fader demo** — otherwise it degrades to a reload and the wow is lost.
9. **05 DATA + 06 RUNS.** Dataset builder (export → Train dataset field) and the runs archive (reopen config, frozen strip chart). These close the loops and round out the five features.

**Gate at each step:** `swift build` green, and from step 4 onward, real data on screen (Π_TBV — no mock streams claimed as live).

---

## Risks Carried Forward (both directions)

- **Live-chart perf** — throttle chart commits ~20Hz, numeral tick ~8Hz, downsample history; decouple chart buffer from FFI callback rate.
- **SwiftUI state identity** — own run `@State` above the view, key by run-id, or a re-render wipes a live run.
- **Glass discipline erosion** — enforce via a single `OpaquePanel` container; glass banned below the chrome layer; honor `accessibilityReduceTransparency`.
- **One-accent discipline** — strict semantic token set (teal/voltage = live/CTA, amber = regress, crimson = fail only) so no stray colors creep in.
- **Estimated-PPL honesty** — hard-label `est.` and never draw the ghost base-PPL line as if measured.
- **Greenfield + bridge** — verify the engine exposes structured-progress quantize and a no-reload `set_lora`/`swap` over the bridge before committing the two signature WOWs.
