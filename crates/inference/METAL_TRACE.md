# lattice-inference — METAL_TRACE.md

`os_signpost` instrumentation of the Metal decode path, gated behind the `signpost` cargo feature (default OFF). It exists so a Metal System Trace captured in Instruments can show the per-decode-step launch and synchronization structure — command-buffer commits, interior `waitUntilCompleted` waits, host-side reads of GPU results — instead of a single opaque GPU busy span.

## Capturing a trace

1. Build a decode binary (e.g. `chat_metal` or `lattice_serve`) with the feature on:
   ```bash
   cargo build --release -p lattice-inference --features f16,metal-gpu,signpost --bin chat_metal
   ```
2. Open Instruments (`xcrun xctrace` or the Instruments app), choose the **Metal System Trace** template, and add the **os_signpost** instrument (or use the `os_signpost` points-of-interest track that Metal System Trace already includes).
3. Launch the built binary as the trace target and run a decode workload.
4. In the trace timeline, filter the signpost track to subsystem `ai.lattice.inference`, category `decode`. Each labeled interval below appears as its own row, aligned against the Metal command-buffer/GPU-schedule tracks Instruments already shows.

The feature is opt-in and adds no code to a default build: every call site is a zero-sized guard that only becomes a real `os_signpost` interval when built with `--features signpost` on macOS.

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
