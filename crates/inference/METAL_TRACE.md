# lattice-inference — METAL_TRACE.md

`os_signpost` instrumentation of the Metal decode path, gated behind the `signpost` cargo feature (default OFF). It exists so a Metal System Trace captured in Instruments can show the per-decode-step launch and synchronization structure — command-buffer commits, interior `waitUntilCompleted` waits, host-side reads of GPU results — instead of a single opaque GPU busy span.

Source: `crates/inference/src/forward/signpost.rs` (interval guard + label statics) and its call sites in `crates/inference/src/forward/metal_qwen35.rs`.

## Capturing a trace

1. Build a decode binary (e.g. `chat_metal` or `lattice_serve`) with the feature on:
   ```bash
   cargo build --release -p lattice-inference --features f16,metal-gpu,signpost --bin chat_metal
   ```
2. Choose a capture path and category based on which tool you're using — the two differ (see "Runtime signpost mode" below):

   **Instruments.app (GUI)** — works with the default mode, no environment variable needed:
   - Open Instruments, choose the **Metal System Trace** template, and add the **os_signpost** instrument (or use the `os_signpost` points-of-interest track Metal System Trace already includes).
   - Launch the built binary as the trace target and run a decode workload.
   - In the trace timeline, filter the signpost track to subsystem `ai.lattice.inference`, category `DynamicTracing`. Each labeled interval below appears as its own row, aligned against the Metal command-buffer/GPU-schedule tracks Instruments already shows.

   **`xcrun xctrace` (CLI)** — requires `LATTICE_SIGNPOST_MODE=always` (see below), or the capture is silently empty:
   ```bash
   LATTICE_SIGNPOST_MODE=always xcrun xctrace record \
     --template 'Metal System Trace' --instrument os_signpost \
     --output trace.trace --launch -- ./target/release/chat_metal
   ```
   Filter the signpost track to subsystem `ai.lattice.inference`, category `decode` (not `DynamicTracing` — mode `always` uses the ordinary category, see below).

The feature is opt-in and adds no code to a default build: every call site is a zero-sized guard that only becomes a real `os_signpost` interval when built with `--features signpost` on macOS.

### Runtime signpost mode

The `signpost` feature's `os_log` category is chosen at runtime from the `LATTICE_SIGNPOST_MODE` environment variable, read once at first use:

| `LATTICE_SIGNPOST_MODE` | Category | `os_signpost_enabled` default | Use with |
|---|---|---|---|
| unset or `auto` (default) | `DynamicTracing` | `false` until an Instruments-style tool session attaches (idle-inert) | Instruments.app (GUI) |
| `always` | `decode` | `true` unconditionally, whether or not any tool is attached | `xcrun xctrace` (CLI) |

Both paths exist because the two tools drive `os_signpost_enabled` differently on macOS. `DynamicTracing` (`OS_LOG_CATEGORY_DYNAMIC_TRACING` in `<os/signpost.h>`) is disabled by default and only becomes enabled while a performance tool like Instruments.app is actively recording — that's what keeps the feature idle-inert (no signpost overhead) in a `--features signpost` build with nothing attached. Instruments.app's GUI recording session satisfies that contract. The `xcrun xctrace` CLI's `os_signpost` instrument, however, was observed on this project's macOS 26 development machine to **not** enable `DynamicTracing`-category signposts under `xctrace record --template 'Metal System Trace' --instrument os_signpost`: a direct `os_signpost_enabled` probe reported `DynamicTracing=0` while an ordinary category reported `decode=1` in the same process, under the same invocation. So the CLI path needs the ordinary `decode` category (`LATTICE_SIGNPOST_MODE=always`) to observe anything at all — trading idle-inertness for CLI-capture compatibility, which is fine for the duration of a deliberate trace-capture run.

## Label glossary

| Label | Scope | Meaning |
|---|---|---|
| `decode.step` | one call to the per-token forward pass | Wall-clock span of a single decode step: embedding lookup through logits/top-k readback. |
| `decode.cb_commit` | one `MTLCommandBuffer::commit()` call | Time to submit the step's command buffer to the GPU queue. |
| `decode.cb_wait` | one `waitUntilCompleted()` call | Interior wait for the GPU to finish the step's command buffer — the launch+sync cost the W1 speculation-lane promotion rule measures against. |
| `decode.host_scalar_read` | one host-side buffer read | CPU read of a `StorageModeShared` GPU buffer after the step's wait (logits, top-k candidates, or the pre-final hidden state used by MTP). |
| `decode.grammar_mask` | one grammar-engine logit mask | CPU-side grammar constraint applied to a step's logits before sampling. |
| `decode.sample` | one token-sampling call | CPU-side sampling (greedy, top-k/top-p, or compact-candidate) that picks the step's next token. |

## Scope

This instruments the existing autoregressive decode path only (`MetalQwen35State::forward_step` and its callers in `generate`/`generate_streaming*`). It adds no new engine behavior — no speculative decode, no verify step — that is future, separately-gated work.
