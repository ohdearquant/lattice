# Gemma 4 E2B — Asset Manifest & Weight-Download Plan (ADR-082 Stage 0)

Companion doc to [ADR-082](adr/ADR-082-gemma4-e2b-support.md), which defines the ten-stage
goldens-first ladder for native Gemma 4 E2B support (text, vision, audio). This page covers only
Stage 0: what's committed to the repo as a fixture, how it was produced, and where the actual
multi-gigabyte checkpoint weights are expected to live once later stages need them. No lattice
engine code for this model family exists yet — see ADR-082 for the full ladder and evidence.

## Tensor manifest (committed fixture)

[`crates/inference/tests/fixtures/gemma4/e2b_tensor_manifest.json`](../crates/inference/tests/fixtures/gemma4/e2b_tensor_manifest.json)
is a full name → `{dtype, shape}` map for every tensor in the checkpoint's monolithic
`model.safetensors` file, plus a `bucket_counts` summary and a `metadata` block (source repo,
pinned revision, header byte length + SHA-256, extraction date).

Pinned checkpoint: `google/gemma-4-E2B-it` @ revision `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`.

| Bucket                 |    Count |
| ---------------------- | -------: |
| `model.language_model` |      600 |
| `model.vision_tower`   |      658 |
| `model.audio_tower`    |      751 |
| `model.embed_vision`   |        1 |
| `model.embed_audio`    |        1 |
| **Total**              | **2011** |

This was extracted **without downloading any weight bytes**, via two HTTP Range requests against
`model.safetensors`: bytes `0-7` give a little-endian `u64` header length `N`, then bytes `8..8+N`
are the safetensors JSON header itself (tensor names, dtypes, shapes, byte offsets — no tensor
data). Total transfer for this checkpoint is ~264 KB against a ~10.25 GB file.

**Generator / drift gate**: [`scripts/gemma4_tensor_manifest.py`](../scripts/gemma4_tensor_manifest.py).

```bash
# Verify: re-fetch the header at the pinned revision, diff against the committed fixture,
# fail closed on any drift (name/shape/dtype/count). This is the only script in this repo
# that touches the network for this model family.
uv run python3 scripts/gemma4_tensor_manifest.py

# Regenerate the committed fixture — deliberate, reviewable, never run by CI.
uv run python3 scripts/gemma4_tensor_manifest.py --write-fixture
```

The script hard-caps its own total fetch at 1 MB and refuses to proceed if the declared header
length would exceed that cap — a mistargeted URL or a non-safetensors response cannot silently
balloon into a multi-gigabyte download.

The Rust-side gate ([`crates/inference/tests/gemma4_tensor_manifest_test.rs`](../crates/inference/tests/gemma4_tensor_manifest_test.rs))
is offline-only: it loads and validates the already-committed fixture, and never touches the
network. It checks the total count, every bucket count, one load-bearing spot-check tensor per
subsystem (PLE embedding table, the local/global KV-shared-layer shape split, a vision
`Gemma4ClippableLinear` clip buffer, an audio Conv2d subsampler tensor), and two fail-closed
negative cases (corrupted/truncated manifest JSON, a manifest missing a required subsystem
bucket).

**Known discrepancy to track into Stage 2/4**: ADR-082's G5 states (source-read from
`configuration_gemma4.py`/`modeling_gemma4.py`) that the final 20 decoder layers "omit K/V
projection weights entirely." The header-extracted manifest does not show that: every one of the
35 `language_model` layers carries its own `self_attn.{k,v,q}_proj.weight` tensors in the
checkpoint. What the manifest does confirm is the real, load-bearing local/global attention split
those weights are keyed on (G3/G4) — the 7 global-attention layers get a 512-wide `k_proj`/
`v_proj`, the other 28 sliding-attention layers get a 256-wide one. Whatever "KV sharing" means at
runtime, it is not tensor omission at the checkpoint level; Stage 2 (config/loader preflight) and
Stage 4 (cache-topology implementation) need to reconcile this before building the shared-KV cache
path.

## Weight-download plan

No lattice code loads Gemma 4 weights yet (Stage 0 is manifest-only). This section documents where
downloads should land once a later stage needs the real checkpoint.

**Directory convention.** This repo already has an established model-cache convention for
HuggingFace downloads: `LATTICE_MODEL_CACHE` (default `~/.lattice/models`), used by
`ensure_model_files` in `crates/inference/src/download.rs` and documented in `AGENTS.md`'s
environment-variable table. Gemma 4 weights follow the same convention — there is no separate
`ASSET_DIR`/`LATTICE_ASSET_DIR` variable in this codebase (checked: no matches in `crates/` or
`docs/`), so Stage-4-and-later work should extend the existing `LATTICE_MODEL_CACHE` mechanism
rather than invent a new one. **Multi-gigabyte checkpoint downloads must target
`$LATTICE_MODEL_CACHE` (or its `~/.lattice/models` default) on external/attached storage — never
the internal disk of a CI runner or dev machine**, the same rule this repo already applies to
Qwen3.5/Qwen3.6 and the BERT embedding family.

**Variant sizes.** Only E2B has been header-audited by this ADR (Stage 0, zero weight bytes
fetched — see manifest `metadata`):

| Variant                | Checkpoint size (BF16, monolithic `model.safetensors`)                     | Source                         |
| ---------------------- | -------------------------------------------------------------------------- | ------------------------------ |
| E2B-it                 | ~10.25 GB (`x-linked-size` header, `10,246,621,918` bytes)                 | header-extracted, this Stage 0 |
| E4B, 12B, 26B-A4B, 31B | not audited — out of scope for this ADR (E2B-only, see ADR-082 "Deferred") | n/a                            |

**What each stage needs.**

| Stage(s) | Assets required                                                                                                                                                             |     Weight bytes touched      |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------: |
| 0        | Safetensors header only (this doc's manifest), plus `config.json`/`processor_config.json`/`tokenizer_config.json`/`tokenizer.json`/`chat_template.jinja` (small text files) |             none              |
| 1        | Tokenizer/chat-template fixtures (already small text files, no safetensors needed)                                                                                          |             none              |
| 2        | Manifest validator only — still no weight bytes                                                                                                                             |             none              |
| 3–4      | Full `model.language_model.*` weight slice (600 tensors) — the text decoder                                                                                                 | partial (~subset of 10.25 GB) |
| 5–6      | Full `model.vision_tower.*` + `model.embed_vision.*` slice (659 tensors)                                                                                                    |            partial            |
| 7–9      | Full `model.audio_tower.*` + `model.embed_audio.*` slice (752 tensors)                                                                                                      |            partial            |

Every stage from 3 onward downloads real weight bytes and must place them under
`$LATTICE_MODEL_CACHE`, consistent with `ensure_model_files`'s existing cache-then-download flow
and `LATTICE_OFFLINE`'s fail-closed behavior on a cache miss.
