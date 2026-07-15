#!/usr/bin/env python3
"""HF differential golden fixtures for the Gemma 4 E2B text math kernels
(ADR-082 Stage 3, G6-G10).

Unlike the Stage-0/1 scripts, this script never touches the network for the
model itself and never loads the ~10.25 GB E2B checkpoint. Stage 3 is pure
per-op math (RMSNorm, GeGLU MLP, scaled embedding, Q/K-norm-V-unscaled,
logit softcap, dual RoPE) -- it instantiates the pinned `transformers`
Gemma 4 modeling classes directly with small, deterministic, seeded
synthetic weights and inputs, and records each op's input/output tensors as
a committed fixture.

Fixture format follows `scripts/vision_goldens_qwen35.py`'s convention:
raw little-endian float32 `.bin` files (one per tensor) plus a small JSON
sidecar per op recording shapes/scalars/`.bin` refs (path, shape, sha256).

Provenance model (read this before touching tolerances or fixtures):
`crates/inference/tests/fixtures/gemma4/stage3/manifest.json` is a COMMITTED,
HAND-AUTHORED, IMMUTABLE spec -- not an output of this script. It records the
pinned reference source commit and per-module SHA-256 hashes, the expected
reference-derived RoPE parameters, per-op seeds, tolerances (with rationale
independent of any observed lattice result), and mutation-separation floors.
This script only ever *reads* that manifest and *verifies* against it:

  1. `verify_source_pins`: hashes the actually-imported `modeling_gemma4.py`,
     `modeling_rope_utils.py`, and `configuration_gemma4.py` and aborts
     (before writing anything) unless they match the manifest's
     `pinned_source_hashes`. This is why the invocations below use
     `uv run --with "transformers @ git+...@<pinned commit>"` instead of the
     project's released `transformers==5.12.1` pin (used by the other,
     unrelated Stage-0/1 vision golden scripts): the release tag and the
     ADR's pinned commit diverge, so the installed release must never be
     silently accepted as the reference.
  2. `verify_rope_reference`: loads the committed pinned E2B config fixture
     (`tests/fixtures/gemma4/e2b_config.json`) through the real
     `Gemma4TextConfig`, and asserts the resulting `rope_parameters` match
     the manifest's `expected_rope_parameters` before deriving the
     theta/partial-rotary-factor scalars the dual_rope golden's synthetic
     (reduced-head_dim) tensors use.
  3. `--write-fixture` only ever (re)writes the `<op>.json` + `<op>_*.bin`
     fixture files, never `manifest.json`. Changing a tolerance, seed, or
     rationale is a manifest edit, reviewed on its own, not a side effect of
     regenerating fixtures.

Shapes: real E2B ratios scaled down by 1/24 on `hidden_size` (1536 -> 64,
preserving `num_attention_heads=8`) and by 1/16 on head width (`head_dim`
256 -> 16, `global_head_dim` 512 -> 32, preserving the real 1:2 sliding:global
ratio and `partial_rotary_factor=0.25`). `rms_norm_eps` and
`final_logit_softcapping=30.0` are the real E2B config values, unscaled --
dimensionless/scale-sensitive constants, not shapes. RoPE thetas are
reference-derived (see `verify_rope_reference` above), not hardcoded. The
GeGLU MLP's `intermediate_size` uses a smaller-than-real ratio (1.5x hidden
vs the real 4x) purely to keep the committed `.bin` files small --
intermediate width doesn't change which formula/op-order bug this golden
catches, only its cost to store. `rms_norm_wide` additionally covers the
real, unscaled E2B `hidden_size=1536` reduction width (medium-4: the reused
lattice RMSNorm kernel is bounded f32 parity, not bit-for-bit, and that bound
should be demonstrated at the real width, not only at a 64-wide reduction).

Usage:
    # Regenerate the committed fixtures (deliberate, reviewable, never run by
    # CI) under the pinned reference commit:
    uv run --with "transformers @ git+https://github.com/huggingface/transformers@ab1771c9e42891d893189978a8009426d70b4688" \\
        python3 scripts/gemma4_stage3_goldens.py --write-fixture

    # Verify (default): regenerate in-memory and diff against the committed
    # JSON/bin. Fails closed on any drift, including source-pin drift.
    uv run --with "transformers @ git+https://github.com/huggingface/transformers@ab1771c9e42891d893189978a8009426d70b4688" \\
        python3 scripts/gemma4_stage3_goldens.py
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "inference"
    / "tests"
    / "fixtures"
    / "gemma4"
    / "stage3"
)
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
E2B_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "inference"
    / "tests"
    / "fixtures"
    / "gemma4"
    / "e2b_config.json"
)

# Synthetic shape reductions (see module docstring for the derivation). These
# are deliberately script-local, not manifest-declared: the manifest's
# provenance contract covers seeds/tolerances/rationale/source hashes/RoPE
# values, not test-shape reductions (ADR-082 review round 2, major-3).
HIDDEN = 64
NUM_HEADS = 8
HEAD_DIM_LOCAL = 16
HEAD_DIM_GLOBAL = 32
INTERMEDIATE = 96
VOCAB = 40
HIDDEN_WIDE = 1536  # real E2B hidden_size, for the rms_norm_wide golden

RMS_NORM_EPS = 1e-6
FINAL_LOGIT_SOFTCAPPING = 30.0

SOURCE_MODULE_IMPORT_PATHS = {
    "transformers/models/gemma4/modeling_gemma4.py": "transformers.models.gemma4.modeling_gemma4",
    "transformers/modeling_rope_utils.py": "transformers.modeling_rope_utils",
    "transformers/models/gemma4/configuration_gemma4.py": "transformers.models.gemma4.configuration_gemma4",
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    return sha256_bytes(Path(path).read_bytes())


# SHA-256 of the committed manifest.json, byte-exact. The manifest is the
# immutable provenance contract (seeds, shapes, tolerances, floors, rationale,
# source hashes); the generator refuses to run -- in EVERY mode, before any
# fixture write -- unless the manifest on disk hashes to exactly this value.
# Changing ANY declared manifest value therefore requires a paired edit to
# this constant, making a coordinated manifest+payload rewrite visible as two
# hunks in review instead of a silent regeneration.
MANIFEST_CONTRACT_SHA256 = "cc7088a3f7d0228abe7ab025c822bc81f69ed2a7a557ebb6da2ff01dfbdc02df"


def verify_manifest_contract() -> None:
    found = sha256_file(str(MANIFEST_PATH))
    if found != MANIFEST_CONTRACT_SHA256:
        print(
            "FATAL: manifest.json does not hash-match the reviewed contract "
            f"digest.\n  expected {MANIFEST_CONTRACT_SHA256}\n  found    "
            f"{found}\n"
            "The manifest is an immutable provenance contract: seeds, "
            "shapes, tolerances, mutation floors, rationale, and source "
            "hashes are all declared there and must not drift alongside a "
            "fixture regeneration. If a declared value legitimately needs "
            "to change, edit the manifest AND update "
            "MANIFEST_CONTRACT_SHA256 in this script in the same reviewed "
            "change.",
            file=sys.stderr,
        )
        sys.exit(1)


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        print(
            f"FATAL: {MANIFEST_PATH} does not exist -- manifest.json is a "
            "committed provenance spec authored independently of this "
            "generator; there is no bootstrap path that creates it "
            "automatically",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(MANIFEST_PATH.read_text())


def verify_source_pins(manifest: dict[str, Any]) -> None:
    """Major-1 fix: fail closed unless the transformers install actually
    executing this script hash-matches the manifest's pinned-source commit.
    Never falls back to a version-string check -- the released version tag
    and the ADR's pinned commit are known to diverge in these exact files.
    """
    import importlib

    expected = manifest["pinned_source_hashes"]
    mismatches: list[str] = []
    for key, dotted in SOURCE_MODULE_IMPORT_PATHS.items():
        want = expected.get(key)
        if want is None:
            mismatches.append(f"{key}: manifest has no expected hash recorded")
            continue
        try:
            mod = importlib.import_module(dotted)
            found = sha256_file(inspect.getfile(mod))
        except Exception as exc:  # noqa: BLE001 -- fail closed, report found=<error>
            mismatches.append(f"{key}: expected {want}, could not hash installed module ({exc})")
            continue
        if found != want:
            mismatches.append(f"{key}: expected {want}, found {found}")

    if mismatches:
        pinned_commit = manifest["pinned_source_commit"]
        print(
            "FATAL: the transformers install executing this script does not "
            f"hash-match the pinned reference commit {pinned_commit} -- "
            "goldens generated under it would not be reference-verified. "
            "Re-run under the pinned source, e.g.:\n"
            '  uv run --with "transformers @ git+https://github.com/'
            f'huggingface/transformers@{pinned_commit}" '
            "scripts/gemma4_stage3_goldens.py --write-fixture\n"
            "Mismatches (found vs expected):\n  " + "\n  ".join(mismatches),
            file=sys.stderr,
        )
        sys.exit(1)


def verify_runtime_versions(manifest: dict[str, Any]) -> None:
    import platform

    expected = manifest["runtime_versions"]
    actual = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {actual[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        print(
            "FATAL: runtime version drift from the manifest's declared "
            "runtime -- goldens would not be reproducible:\n  "
            + "\n  ".join(mismatches),
            file=sys.stderr,
        )
        sys.exit(1)


def verify_rope_reference(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Major-3 fix: derive the dual-RoPE theta/partial-rotary-factor records
    from the real `Gemma4TextConfig`, parsing the committed pinned E2B config
    fixture -- not from hardcoded literals or an inert stub config.
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    if not E2B_CONFIG_PATH.exists():
        print(f"FATAL: {E2B_CONFIG_PATH} does not exist", file=sys.stderr)
        sys.exit(1)
    full_config = json.loads(E2B_CONFIG_PATH.read_text())
    cfg = Gemma4TextConfig(**full_config["text_config"])
    actual = cfg.rope_parameters
    expected = manifest["expected_rope_parameters"]
    if actual != expected:
        print(
            "FATAL: rope_parameters derived from the pinned E2B config "
            "fixture via the reference Gemma4TextConfig do not match the "
            "manifest's expected_rope_parameters -- refusing to write "
            f"fixtures.\n  derived:  {actual}\n  expected: {expected}",
            file=sys.stderr,
        )
        sys.exit(1)
    return actual


class TensorRef:
    """Marks a raw torch.Tensor for `.bin`-file materialization instead of
    inline JSON serialization; wraps the tensor until `materialize_op`
    writes it out."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.detach().to(torch.float32).cpu().contiguous()


def materialize_op(op: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Walk `fields`, writing every `TensorRef` to `<op>_<key>.bin` and
    replacing it with a small JSON-safe ref (path/shape/dtype/sha256);
    non-tensor fields pass through unchanged."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, TensorRef):
            arr = value.tensor.numpy()
            data = np.ascontiguousarray(arr, dtype="<f4").tobytes()
            fname = f"{op}_{key}.bin"
            (FIXTURE_DIR / fname).write_bytes(data)
            out[key] = {
                "bin": fname,
                "shape": list(arr.shape),
                "dtype": "float32",
                "endianness": "little",
                "sha256": sha256_bytes(data),
                "num_elements": int(arr.size),
            }
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# G6: standard RMSNorm (modeling_gemma4.py: class Gemma4RMSNorm, `forward`).
#   normed = x.float() * pow(mean(x.float()**2, -1, keepdim=True) + eps, -0.5)
#   out = (normed * weight.float()).type_as(x)   [with_scale=True]
# Plain gamma scaling -- NOT lattice's Qwen3.5 shifted `(1 + gamma)` variant.
# ---------------------------------------------------------------------------
def gen_rms_norm(seed: int) -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    set_seed(seed)
    batch, seq = 2, 3
    x = torch.randn(batch, seq, HIDDEN, dtype=torch.float32)
    norm = Gemma4RMSNorm(dim=HIDDEN, eps=RMS_NORM_EPS, with_scale=True)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(HIDDEN, dtype=torch.float32) * 0.5 + 1.0)
    out = norm(x)
    return {
        "op": "rms_norm",
        "source": "transformers/models/gemma4/modeling_gemma4.py:197-215 (Gemma4RMSNorm)",
        "shape": [batch, seq, HIDDEN],
        "eps": RMS_NORM_EPS,
        "input": TensorRef(x),
        "weight": TensorRef(norm.weight),
        "output": TensorRef(out),
    }


# ---------------------------------------------------------------------------
# Medium-4: same op as G6, at the real E2B hidden_size=1536 reduction width
# (the other rms_norm golden above uses the 1/24-scaled synthetic width=64;
# this one demonstrates the bounded-f32-parity claim at the real width).
# ---------------------------------------------------------------------------
def gen_rms_norm_wide(seed: int) -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    set_seed(seed)
    batch, seq = 1, 2
    x = torch.randn(batch, seq, HIDDEN_WIDE, dtype=torch.float32)
    norm = Gemma4RMSNorm(dim=HIDDEN_WIDE, eps=RMS_NORM_EPS, with_scale=True)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(HIDDEN_WIDE, dtype=torch.float32) * 0.5 + 1.0)
    out = norm(x)
    return {
        "op": "rms_norm_wide",
        "source": "transformers/models/gemma4/modeling_gemma4.py:197-215 (Gemma4RMSNorm), hidden_size=1536 (real E2B width)",
        "shape": [batch, seq, HIDDEN_WIDE],
        "eps": RMS_NORM_EPS,
        "input": TensorRef(x),
        "weight": TensorRef(norm.weight),
        "output": TensorRef(out),
    }


# ---------------------------------------------------------------------------
# G7: GeGLU MLP (modeling_gemma4.py: class Gemma4TextMLP, `forward`).
#   down_proj(act_fn(gate_proj(x)) * up_proj(x)), act_fn = gelu_pytorch_tanh
# ---------------------------------------------------------------------------
def gen_geglu_mlp(seed: int) -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

    class MiniMlpConfig:
        hidden_size = HIDDEN
        intermediate_size = INTERMEDIATE
        hidden_activation = "gelu_pytorch_tanh"
        num_hidden_layers = 1
        num_kv_shared_layers = 0
        use_double_wide_mlp = False

    set_seed(seed)
    batch, seq = 2, 3
    x = torch.randn(batch, seq, HIDDEN, dtype=torch.float32)
    mlp = Gemma4TextMLP(MiniMlpConfig(), layer_idx=0)
    with torch.no_grad():
        gate_w = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.float32) * 0.1
        up_w = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.float32) * 0.1
        down_w = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.float32) * 0.1
        mlp.gate_proj.weight.copy_(gate_w)
        mlp.up_proj.weight.copy_(up_w)
        mlp.down_proj.weight.copy_(down_w)
    out = mlp(x)
    return {
        "op": "geglu_mlp",
        "source": "transformers/models/gemma4/modeling_gemma4.py:1074-1090 (Gemma4TextMLP)",
        "shape": [batch, seq, HIDDEN],
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "input": TensorRef(x),
        "gate_proj_weight": TensorRef(gate_w),
        "up_proj_weight": TensorRef(up_w),
        "down_proj_weight": TensorRef(down_w),
        "output": TensorRef(out),
    }


# ---------------------------------------------------------------------------
# G10a: scaled embedding (modeling_gemma4.py: class
# Gemma4TextScaledWordEmbedding, `forward`).
#   out = embedding_lookup(ids) * embed_scale, embed_scale = sqrt(hidden_size)
#   (a python float; here computed and applied in float32 -- see
#   Gemma4TextScaledWordEmbedding.forward: `.to(self.weight.dtype)`).
# ---------------------------------------------------------------------------
def gen_scaled_embedding(seed: int) -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextScaledWordEmbedding,
    )

    set_seed(seed)
    batch, seq = 2, 5
    embed_scale = float(HIDDEN**0.5)
    emb = Gemma4TextScaledWordEmbedding(
        VOCAB, HIDDEN, padding_idx=0, embed_scale=embed_scale
    )
    with torch.no_grad():
        weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.3
        emb.weight.copy_(weight)
    ids = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
    ids[0, 0] = 0  # exercise padding_idx=0 explicitly (forward is unaffected)
    out = emb(ids)
    return {
        "op": "scaled_embedding",
        "source": (
            "transformers/models/gemma4/modeling_gemma4.py:1465-1476 "
            "(Gemma4TextScaledWordEmbedding)"
        ),
        "vocab": VOCAB,
        "hidden": HIDDEN,
        "embed_scale": embed_scale,
        "embed_weight": TensorRef(weight),
        "input_ids": ids.tolist(),
        "output": TensorRef(out),
    }


# ---------------------------------------------------------------------------
# G4/G6: Q/K norm with V unscaled (modeling_gemma4.py: Gemma4TextAttention
# __init__ / forward: q_norm/k_norm (with_scale default True) and v_norm
# (with_scale=False, no weight); forward applies q_norm before RoPE, k_norm
# before RoPE, and v_norm with no RoPE at all -- V never rotates.
# This op golden covers the per-head-norm step in isolation; dual_rope below
# covers the RoPE step in isolation. Uses local head_dim=16, representative
# of a sliding-attention layer's per-head width.
# ---------------------------------------------------------------------------
def gen_qk_norm_v_unscaled(seed: int) -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    set_seed(seed)
    batch, seq, heads = 1, 3, 4
    q = torch.randn(batch, seq, heads, HEAD_DIM_LOCAL, dtype=torch.float32)
    k = torch.randn(batch, seq, heads, HEAD_DIM_LOCAL, dtype=torch.float32)
    v = torch.randn(batch, seq, heads, HEAD_DIM_LOCAL, dtype=torch.float32)

    q_norm = Gemma4RMSNorm(dim=HEAD_DIM_LOCAL, eps=RMS_NORM_EPS, with_scale=True)
    k_norm = Gemma4RMSNorm(dim=HEAD_DIM_LOCAL, eps=RMS_NORM_EPS, with_scale=True)
    v_norm = Gemma4RMSNorm(dim=HEAD_DIM_LOCAL, eps=RMS_NORM_EPS, with_scale=False)
    with torch.no_grad():
        q_norm.weight.copy_(torch.randn(HEAD_DIM_LOCAL, dtype=torch.float32) * 0.5 + 1.0)
        k_norm.weight.copy_(torch.randn(HEAD_DIM_LOCAL, dtype=torch.float32) * 0.5 + 1.0)

    q_out = q_norm(q)
    k_out = k_norm(k)
    v_out = v_norm(v)
    return {
        "op": "qk_norm_v_unscaled",
        "source": (
            "transformers/models/gemma4/modeling_gemma4.py (q_norm/k_norm/"
            "v_norm construction and application order in "
            "Gemma4TextAttention)"
        ),
        "shape": [batch, seq, heads, HEAD_DIM_LOCAL],
        "eps": RMS_NORM_EPS,
        "q_input": TensorRef(q),
        "k_input": TensorRef(k),
        "v_input": TensorRef(v),
        "q_norm_weight": TensorRef(q_norm.weight),
        "k_norm_weight": TensorRef(k_norm.weight),
        "q_output": TensorRef(q_out),
        "k_output": TensorRef(k_out),
        "v_output": TensorRef(v_out),
    }


# ---------------------------------------------------------------------------
# G10b: final logit softcapping (modeling_gemma4.py:
# Gemma4ForCausalLM.forward, after self.lm_head):
#   logits = logits / cap; logits = tanh(logits); logits = logits * cap
#
# Major-2 fix: this golden previously re-derived the three-op formula
# locally (three bare torch calls, no Gemma4 class involved at all) --
# exactly the same-implementation-on-both-sides failure mode goldens exist
# to prevent. This now builds a real `Gemma4ForCausalLM` from a real
# `Gemma4TextConfig` and captures logits from *its own* `forward`, so the
# softcap placement/condition/formula is whatever the pinned source actually
# does, not a hand-copy of it. The only stub is `model.model` (the decoder
# stack) -- swapped for a fixed hidden_states tensor so this golden stays a
# cheap, isolated per-op test rather than a full forward pass; `lm_head` and
# the softcap branch are the real `Gemma4ForCausalLM.forward` code path.
# ---------------------------------------------------------------------------
def gen_logit_softcap(seed: int) -> dict[str, Any]:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    set_seed(seed)
    batch, seq = 2, 5
    cfg = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=HEAD_DIM_LOCAL,
        global_head_dim=HEAD_DIM_GLOBAL,
        final_logit_softcapping=FINAL_LOGIT_SOFTCAPPING,
        tie_word_embeddings=False,
    )
    model = Gemma4ForCausalLM(cfg)
    model.eval()

    # Wide range (includes |logit| >> cap) so tanh saturation actually bites
    # -- otherwise a disabled-softcap negative test could stay within
    # tolerance by accident for small logits.
    hidden = (torch.rand(batch, seq, HIDDEN, dtype=torch.float32) - 0.5) * 8.0

    class _StubDecoderOutput:
        def __init__(self, last_hidden_state: torch.Tensor):
            self.last_hidden_state = last_hidden_state
            self.past_key_values = None
            self.hidden_states = None
            self.attentions = None
            self.shared_kv_states = None

    def _stub_decoder_forward(*_args: Any, **_kwargs: Any) -> _StubDecoderOutput:
        return _StubDecoderOutput(hidden)

    # Bypass only the decoder stack (self.model) -- lm_head + the softcap
    # branch below run as the real Gemma4ForCausalLM.forward code.
    model.model.forward = _stub_decoder_forward

    with torch.no_grad():
        lm_head_weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 2.0
        model.lm_head.weight.copy_(lm_head_weight)
        input_ids = torch.randint(0, VOCAB, (batch, seq), dtype=torch.long)
        result = model(input_ids=input_ids, use_cache=False)

    out = result.logits
    # The pre-softcap logits (lm_head(hidden) before the cap/tanh/uncap
    # branch) double as the uncapped-negative-test input: recompute them via
    # the same real lm_head weight, matching what Gemma4ForCausalLM.forward
    # feeds into its softcap branch.
    with torch.no_grad():
        pre_softcap = model.lm_head(hidden)

    return {
        "op": "logit_softcap",
        "source": (
            "transformers/models/gemma4/modeling_gemma4.py "
            "(Gemma4ForCausalLM.forward, lm_head then final_logit_softcapping "
            "branch), executed via a real Gemma4ForCausalLM/Gemma4TextConfig "
            "with only the decoder stack (self.model) stubbed"
        ),
        "shape": [batch, seq, VOCAB],
        "cap": FINAL_LOGIT_SOFTCAPPING,
        "input": TensorRef(pre_softcap),
        "output": TensorRef(out),
    }


# ---------------------------------------------------------------------------
# G8: dual RoPE (modeling_gemma4.py: Gemma4TextRotaryEmbedding.forward /
# compute_default_rope_parameters, transformers/modeling_rope_utils.py:
# _compute_proportional_rope_parameters; rotate_half + apply_rotary_pos_emb).
#
# rotate_half pairing convention (verified against the pinned source, this
# repo's worst historical bug class): x1 = x[..., :dim//2], x2 =
# x[..., dim//2:], rotate = cat(-x2, x1) -- i.e. dimension i is paired with
# dimension (dim//2 + i) ("stride-half"), NOT the interleaved (2i, 2i+1)
# convention. Local (sliding) layers use the "default" RoPE type: standard
# full-head_dim inv_freq. Global (full-attention) layers use the
# "proportional" RoPE type (_compute_proportional_rope_parameters): only the
# first `int(partial_rotary_factor * head_dim // 2)` of the head_dim//2
# frequency slots are non-zero; the rest are exactly zero, which under
# cos(0)=1/sin(0)=0 makes those dimension-pairs a no-op (pass-through)
# rather than truncating the tensor -- cos/sin are always head_dim-wide.
#
# Major-3 fix: `theta_local`/`theta_global`/`partial_rotary_factor` are no
# longer hardcoded literals -- they are the values `verify_rope_reference`
# derived from the real `Gemma4TextConfig` parsing the pinned E2B config
# fixture (asserted equal to the manifest's `expected_rope_parameters`
# before this function is even called). Only the head_dim/hidden_size/
# num_attention_heads inputs to the `MiniRopeConfig` stand-in below are
# synthetic reductions (per the module docstring); the `standardize_rope_params`
# no-op mirrors the base config method's role for this already-standardized
# `rope_parameters` dict shape.
# ---------------------------------------------------------------------------
def gen_dual_rope(
    seed: int, theta_local: float, theta_global: float, partial_rotary_factor: float
) -> dict[str, Any]:
    from transformers.modeling_rope_utils import _compute_proportional_rope_parameters
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    class MiniRopeConfig:
        def __init__(self, head_dim: int, rope_params: dict[str, Any]):
            self.head_dim = head_dim
            self.global_head_dim = head_dim
            self.hidden_size = HIDDEN
            self.num_attention_heads = NUM_HEADS
            self.rope_parameters = {"sliding_attention": None, "full_attention": None}
            self.rope_parameters.update(rope_params)

        def standardize_rope_params(self) -> None:
            return None

    set_seed(seed)
    batch, seq, heads = 1, 6, 2
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    # Local (sliding): "default" rope_type, full head_dim.
    local_cfg = MiniRopeConfig(
        HEAD_DIM_LOCAL,
        {"sliding_attention": {"rope_type": "default", "rope_theta": theta_local}},
    )
    local_inv_freq, local_scaling = Gemma4TextRotaryEmbedding.compute_default_rope_parameters(
        local_cfg, layer_type="sliding_attention"
    )

    # Global (full-attention): "proportional" rope_type, reference-derived
    # theta and partial_rotary_factor over global_head_dim.
    global_cfg = MiniRopeConfig(
        HEAD_DIM_GLOBAL,
        {
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": theta_global,
                "partial_rotary_factor": partial_rotary_factor,
            }
        },
    )
    global_inv_freq, global_scaling = _compute_proportional_rope_parameters(
        global_cfg, layer_type="full_attention", head_dim_key="global_head_dim"
    )

    def cos_sin(inv_freq: torch.Tensor, scaling: float) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors Gemma4TextRotaryEmbedding.forward.
        inv_freq_expanded = inv_freq[None, :, None].float().expand(batch, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * scaling, emb.sin() * scaling

    local_cos, local_sin = cos_sin(local_inv_freq, local_scaling)
    global_cos, global_sin = cos_sin(global_inv_freq, global_scaling)

    local_x = torch.randn(batch, seq, heads, HEAD_DIM_LOCAL, dtype=torch.float32)
    global_x = torch.randn(batch, seq, heads, HEAD_DIM_GLOBAL, dtype=torch.float32)
    local_out = apply_rotary_pos_emb(local_x, local_cos, local_sin, unsqueeze_dim=2)
    global_out = apply_rotary_pos_emb(global_x, global_cos, global_sin, unsqueeze_dim=2)

    return {
        "op": "dual_rope",
        "source": (
            "transformers/models/gemma4/modeling_gemma4.py "
            "(rotate_half, apply_rotary_pos_emb, Gemma4TextRotaryEmbedding); "
            "transformers/modeling_rope_utils.py "
            "(_compute_proportional_rope_parameters); theta/partial-factor "
            "values reference-derived from tests/fixtures/gemma4/"
            "e2b_config.json via Gemma4TextConfig (see verify_rope_reference)"
        ),
        "shape_local": [batch, seq, heads, HEAD_DIM_LOCAL],
        "shape_global": [batch, seq, heads, HEAD_DIM_GLOBAL],
        "theta_local": theta_local,
        "theta_global": theta_global,
        "partial_rotary_factor": partial_rotary_factor,
        "position_ids": position_ids.squeeze(0).tolist(),
        "local_input": TensorRef(local_x),
        "global_input": TensorRef(global_x),
        "local_inv_freq": TensorRef(local_inv_freq),
        "global_inv_freq": TensorRef(global_inv_freq),
        "local_output": TensorRef(local_out),
        "global_output": TensorRef(global_out),
    }


def build_all_ops(manifest: dict[str, Any], rope_params: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ops_spec = manifest["ops"]
    theta_local = rope_params["sliding_attention"]["rope_theta"]
    theta_global = rope_params["full_attention"]["rope_theta"]
    partial_rotary_factor = rope_params["full_attention"]["partial_rotary_factor"]
    return {
        "rms_norm": gen_rms_norm(ops_spec["rms_norm"]["seed"]),
        "rms_norm_wide": gen_rms_norm_wide(ops_spec["rms_norm_wide"]["seed"]),
        "geglu_mlp": gen_geglu_mlp(ops_spec["geglu_mlp"]["seed"]),
        "scaled_embedding": gen_scaled_embedding(ops_spec["scaled_embedding"]["seed"]),
        "qk_norm_v_unscaled": gen_qk_norm_v_unscaled(ops_spec["qk_norm_v_unscaled"]["seed"]),
        "logit_softcap": gen_logit_softcap(ops_spec["logit_softcap"]["seed"]),
        "dual_rope": gen_dual_rope(
            ops_spec["dual_rope"]["seed"], theta_local, theta_global, partial_rotary_factor
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    verify_manifest_contract()
    verify_source_pins(manifest)
    verify_runtime_versions(manifest)
    rope_params = verify_rope_reference(manifest)

    op_files = {op: spec["file"] for op, spec in manifest["ops"].items()}

    if args.write_fixture:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        all_ops = build_all_ops(manifest, rope_params)
        for op, data in all_ops.items():
            materialized = materialize_op(op, data)
            path = FIXTURE_DIR / op_files[op]
            path.write_text(json.dumps(materialized, indent=2) + "\n")
            print(f"wrote {path}")
        print(
            "NOTE: manifest.json was NOT written by this run -- it is a "
            "committed, hand-authored provenance spec; edit it directly if "
            "a seed/tolerance/rationale needs to change."
        )
        return 0

    # Verify mode: diff in-memory regeneration against committed fixtures.
    ok = True
    all_ops = build_all_ops(manifest, rope_params)
    for op, data in all_ops.items():
        materialized = materialize_op(f"_verify_{op}", data)
        path = FIXTURE_DIR / op_files[op]
        if not path.exists():
            print(f"FATAL: {path} does not exist", file=sys.stderr)
            ok = False
            continue
        committed = json.loads(path.read_text())
        # Compare sidecars structurally, but only after re-pathing the
        # freshly-regenerated (`_verify_<op>_*.bin`) refs to the committed
        # naming scheme, then diff bin bytes directly (not their paths).
        drift = False
        if set(committed.keys()) != set(materialized.keys()):
            drift = True
        else:
            for key, val in committed.items():
                new_val = materialized[key]
                if isinstance(val, dict) and "bin" in val:
                    committed_bytes = (FIXTURE_DIR / val["bin"]).read_bytes()
                    new_bytes = (FIXTURE_DIR / new_val["bin"]).read_bytes()
                    if committed_bytes != new_bytes or val["shape"] != new_val["shape"]:
                        drift = True
                    (FIXTURE_DIR / new_val["bin"]).unlink()
                elif val != new_val:
                    drift = True
        if drift:
            print(f"FATAL: {op} fixture drift between regenerated and committed", file=sys.stderr)
            ok = False

    if not ok:
        return 1
    print(
        "OK: all Stage-3 goldens match the committed fixtures byte-for-byte "
        "(source pins, runtime versions, and reference-derived RoPE "
        "parameters verified against the committed manifest)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
