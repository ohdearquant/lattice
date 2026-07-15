#!/usr/bin/env python3
"""HF differential golden fixtures for the Gemma 4 E2B text math kernels
(ADR-082 Stage 3, G6-G10).

Unlike the Stage-0/1 scripts, this script never touches the network and never
loads the ~10.25 GB E2B checkpoint. Stage 3 is pure per-op math (RMSNorm,
GeGLU MLP, scaled embedding, Q/K-norm-V-unscaled, logit softcap, dual RoPE) --
it instantiates the pinned `transformers` Gemma 4 modeling classes directly
with small, deterministic, seeded synthetic weights and inputs, and records
each op's input/output tensors as a committed fixture.

Fixture format follows `scripts/vision_goldens_qwen35.py`'s convention:
raw little-endian float32 `.bin` files (one per tensor) plus a small JSON
sidecar per op recording shapes/scalars/`.bin` refs (path, shape, sha256),
and a top-level `manifest.json` with provenance and predeclared per-op
tolerances. Embedding large arrays directly as nested JSON lists was tried
first and produced a 1.3 MB `geglu_mlp.json` (one line per float under
`json.dumps(indent=2)`) -- `.bin` is ~5x smaller and matches how every other
tensor-heavy golden in this repo is committed.

Reference module: `transformers.models.gemma4.modeling_gemma4` /
`transformers.modeling_rope_utils`, installed at pinned version
`transformers==5.12.1` (pyproject.toml `[tool.uv] dev-dependencies`, the same
runtime pin `scripts/vision_goldens_qwen35.py` uses). This is the first
Stage-3+ script to import the Gemma 4 *modeling* code (Stages 0/1 used only
safetensors-header extraction and the `tokenizers`/`jinja2` libraries) -- the
line numbers cited in this script's docstrings are read directly from the
installed package (`transformers/models/gemma4/modeling_gemma4.py`,
`transformers/modeling_rope_utils.py`) and cross-checked against ADR-082's
G6-G10 findings, which were themselves sourced from `transformers` commit
`ab1771c9e42891d893189978a8009426d70b4688` -- both describe the same
architecture and formulas (verified line-by-line below); no behavioral drift
was found between the two.

Shapes: real E2B ratios scaled down by 1/24 on `hidden_size` (1536 -> 64,
preserving `num_attention_heads=8`) and by 1/16 on head width (`head_dim`
256 -> 16, `global_head_dim` 512 -> 32, preserving the real 1:2 sliding:global
ratio and `partial_rotary_factor=0.25`). `rms_norm_eps`, RoPE thetas
(1e4 local / 1e6 global), and `final_logit_softcapping=30.0` are the real E2B
config values, unscaled -- these are dimensionless/scale-sensitive constants,
not shapes. The GeGLU MLP's `intermediate_size` uses a smaller-than-real
ratio (1.5x hidden vs the real 4x) purely to keep the committed `.bin` files
small -- intermediate width doesn't change which formula/op-order bug this
golden catches, only its cost to store.

Usage:
    # Regenerate the committed fixtures (deliberate, reviewable, never run by
    # CI):
    uv run python3 scripts/gemma4_stage3_goldens.py --write-fixture

    # Verify (default): regenerate in-memory and diff against the committed
    # JSON/bin. Fails closed on any drift. No network access.
    uv run python3 scripts/gemma4_stage3_goldens.py
"""

from __future__ import annotations

import argparse
import hashlib
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

SCHEMA_VERSION = 1
ADR = "ADR-082 Stage 3 (G6-G10)"

# Runtime versions this fixture set was generated with -- same pin as
# scripts/vision_goldens_qwen35.py (pyproject.toml `[tool.uv] dev-dependencies`).
# A version drift changes CPU forward-pass numerics silently enough that
# goldens generated under a different version are not trustworthy without
# re-review -- fail closed rather than silently regenerate.
REFERENCE_VERSIONS = {
    "torch": "2.13.0",
    "transformers": "5.12.1",
    "numpy": "2.4.6",
    "python": "3.11.12",
}

# Real E2B values (ADR-082 G2-G10), unscaled.
RMS_NORM_EPS = 1e-6
ROPE_THETA_LOCAL = 10_000.0
ROPE_THETA_GLOBAL = 1_000_000.0
PARTIAL_ROTARY_FACTOR = 0.25
FINAL_LOGIT_SOFTCAPPING = 30.0

# Scaled-down shapes (see module docstring for the derivation).
HIDDEN = 64
NUM_HEADS = 8
HEAD_DIM_LOCAL = 16
HEAD_DIM_GLOBAL = 32
INTERMEDIATE = 96  # 1.5x hidden -- kept small to bound committed .bin size
VOCAB = 40

# Per-op predeclared tolerances (max-abs-diff, f32 reference), justified in
# the manifest and read by the Rust tests -- never loosened post-hoc.
TOLERANCES: dict[str, float] = {
    "rms_norm": 1e-6,
    "geglu_mlp": 1e-5,
    "scaled_embedding": 1e-6,
    "qk_norm_v_unscaled": 1e-6,
    "logit_softcap": 3e-6,
    "dual_rope": 1e-6,
}
TOLERANCE_JUSTIFICATION: dict[str, str] = {
    "rms_norm": (
        "Elementwise reduction over hidden=64 in float32, single op (square, "
        "mean, +eps, pow(-0.5), multiply) with no matmul reassociation -- f32 "
        "round-trip precision, no accumulation-order sensitivity expected. "
        "Empirically measured max-abs-diff on the Rust implementation was "
        "~4.8e-7."
    ),
    "geglu_mlp": (
        "Two hidden x intermediate matmuls (gate_proj, up_proj) and one "
        "intermediate x hidden matmul (down_proj), each a 64/96-wide "
        "reduction. Reference uses torch's BLAS; the lattice kernel under "
        "test uses Accelerate/AMX or a scalar/SIMD fallback -- reduction "
        "order (and therefore rounding) can differ from torch's, per this "
        "repo's f32-reassociation-reorder bug class. 1e-5 accommodates that "
        "reordering (empirically measured max-abs-diff ~1.2e-7) while still "
        "catching a wrong-formula or wrong-activation bug (which would "
        "produce differences many orders of magnitude larger)."
    ),
    "scaled_embedding": (
        "A table lookup (exact copy, no arithmetic) followed by one scalar "
        "multiply (embed_scale = sqrt(hidden_size), computed once) -- no "
        "reduction, no reassociation risk. Empirically measured "
        "max-abs-diff was 0."
    ),
    "qk_norm_v_unscaled": (
        "Same elementwise-reduction shape as rms_norm, applied per-head over "
        "head_dim in {16, 32} (well within f32 exactness for this op). "
        "Empirically measured max-abs-diff was ~2.4e-7 (q/v), 0 (k)."
    ),
    "logit_softcap": (
        "Two scalar multiplies and one tanh, elementwise, no reduction -- "
        "but Rust's f32::tanh and PyTorch's libm tanh are two different "
        "correctly-rounded-to-a-few-ULPs implementations, not the same "
        "routine, so their outputs differ by a few ULPs rather than being "
        "bit-identical. Empirically measured max-abs-diff on the Rust "
        "implementation was ~1.9e-6 at cap=30 with logits up to +/-60; 3e-6 "
        "keeps headroom for that while still catching a wrong-formula or "
        "disabled-softcap bug (orders of magnitude larger, see the "
        "disabled-softcap negative test)."
    ),
    "dual_rope": (
        "cos/sin evaluated once per (position, frequency) pair then combined "
        "via two multiplies and one add per element (rotate_half) -- "
        "elementwise, no reduction; matches the reference's own float32 "
        "RoPE math exactly. Empirically measured max-abs-diff was ~2.4e-7 "
        "(local), ~1.2e-7 (global)."
    ),
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def validate_runtime_versions() -> None:
    import platform

    actual = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    mismatches = [
        f"{name}: expected {REFERENCE_VERSIONS[name]}, got {actual[name]}"
        for name in REFERENCE_VERSIONS
        if actual[name] != REFERENCE_VERSIONS[name]
    ]
    if mismatches:
        print(
            "FATAL: runtime version drift from the pinned reference "
            f"({ADR}) -- goldens would not be reproducible:\n  "
            + "\n  ".join(mismatches),
            file=sys.stderr,
        )
        sys.exit(1)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
def gen_rms_norm() -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    set_seed(1)
    batch, seq = 2, 3
    x = torch.randn(batch, seq, HIDDEN, dtype=torch.float32)
    norm = Gemma4RMSNorm(dim=HIDDEN, eps=RMS_NORM_EPS, with_scale=True)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(HIDDEN, dtype=torch.float32) * 0.5 + 1.0)
    out = norm(x)
    return {
        "op": "rms_norm",
        "source": "transformers/models/gemma4/modeling_gemma4.py:193-210 (Gemma4RMSNorm)",
        "shape": [batch, seq, HIDDEN],
        "eps": RMS_NORM_EPS,
        "input": TensorRef(x),
        "weight": TensorRef(norm.weight),
        "output": TensorRef(out),
    }


# ---------------------------------------------------------------------------
# G7: GeGLU MLP (modeling_gemma4.py: class Gemma4TextMLP, `forward`).
#   down_proj(act_fn(gate_proj(x)) * up_proj(x)), act_fn = gelu_pytorch_tanh
# ---------------------------------------------------------------------------
def gen_geglu_mlp() -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

    class MiniMlpConfig:
        hidden_size = HIDDEN
        intermediate_size = INTERMEDIATE
        hidden_activation = "gelu_pytorch_tanh"
        num_hidden_layers = 1
        num_kv_shared_layers = 0
        use_double_wide_mlp = False

    set_seed(2)
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
        "source": "transformers/models/gemma4/modeling_gemma4.py:1066-1082 (Gemma4TextMLP)",
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
def gen_scaled_embedding() -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextScaledWordEmbedding,
    )

    set_seed(3)
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
            "transformers/models/gemma4/modeling_gemma4.py:1458-1469 "
            "(Gemma4TextScaledWordEmbedding), instantiated at "
            "modeling_gemma4.py:1601-1602 with embed_scale=hidden_size**0.5"
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
# __init__ / forward, lines 1208-1213 construct q_norm/k_norm (with_scale
# default True) and v_norm (with_scale=False, no weight); forward applies
# q_norm before RoPE (line 1240-1241), k_norm before RoPE (line 1257), and
# v_norm with no RoPE at all (line 1260) -- V never rotates.
# This op golden covers the per-head-norm step in isolation; dual_rope below
# covers the RoPE step in isolation. Uses local head_dim=16, representative
# of a sliding-attention layer's per-head width.
# ---------------------------------------------------------------------------
def gen_qk_norm_v_unscaled() -> dict[str, Any]:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    set_seed(4)
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
            "transformers/models/gemma4/modeling_gemma4.py:1208-1213 "
            "(q_norm/k_norm/v_norm construction), 1240-1260 (application "
            "order: q_norm then RoPE, k_norm then RoPE, v_norm with no RoPE)"
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
# G10b: final logit softcapping (modeling_gemma4.py:1888-1892).
#   logits = logits / cap; logits = tanh(logits); logits = logits * cap
# ---------------------------------------------------------------------------
def gen_logit_softcap() -> dict[str, Any]:
    set_seed(5)
    batch, seq = 2, 5
    # Wide range (includes |logit| >> cap) so tanh saturation actually bites
    # -- otherwise a disabled-softcap negative test could stay within
    # tolerance by accident for small logits.
    logits = (torch.rand(batch, seq, VOCAB, dtype=torch.float32) - 0.5) * 120.0
    cap = FINAL_LOGIT_SOFTCAPPING
    out = logits / cap
    out = torch.tanh(out)
    out = out * cap
    return {
        "op": "logit_softcap",
        "source": "transformers/models/gemma4/modeling_gemma4.py:1888-1892",
        "shape": [batch, seq, VOCAB],
        "cap": cap,
        "input": TensorRef(logits),
        "output": TensorRef(out),
    }


# ---------------------------------------------------------------------------
# G8: dual RoPE (modeling_gemma4.py: Gemma4TextRotaryEmbedding.forward /
# compute_default_rope_parameters, transformers/modeling_rope_utils.py:
# _compute_proportional_rope_parameters; rotate_half + apply_rotary_pos_emb
# at modeling_gemma4.py:780-806).
#
# rotate_half pairing convention (verified against the pinned source, this
# repo's worst historical bug class): x1 = x[..., :dim//2], x2 =
# x[..., dim//2:], rotate = cat(-x2, x1) -- i.e. dimension i is paired with
# dimension (dim//2 + i) ("stride-half"), NOT the interleaved (2i, 2i+1)
# convention. Local (sliding) layers use the "default" RoPE type: standard
# full-head_dim inv_freq, theta=1e4. Global (full-attention) layers use the
# "proportional" RoPE type (_compute_proportional_rope_parameters): only the
# first `int(partial_rotary_factor * head_dim // 2)` of the head_dim//2
# frequency slots are non-zero (theta=1e6); the rest are exactly zero,
# which under cos(0)=1/sin(0)=0 makes those dimension-pairs a no-op
# (pass-through) rather than truncating the tensor -- cos/sin are always
# head_dim-wide.
# ---------------------------------------------------------------------------
def gen_dual_rope() -> dict[str, Any]:
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

    set_seed(6)
    batch, seq, heads = 1, 6, 2
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    # Local (sliding): "default" rope_type, full head_dim, theta=1e4.
    local_cfg = MiniRopeConfig(
        HEAD_DIM_LOCAL,
        {"sliding_attention": {"rope_type": "default", "rope_theta": ROPE_THETA_LOCAL}},
    )
    local_inv_freq, local_scaling = Gemma4TextRotaryEmbedding.compute_default_rope_parameters(
        local_cfg, layer_type="sliding_attention"
    )

    # Global (full-attention): "proportional" rope_type, theta=1e6,
    # partial_rotary_factor=0.25 over global_head_dim.
    global_cfg = MiniRopeConfig(
        HEAD_DIM_GLOBAL,
        {
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": ROPE_THETA_GLOBAL,
                "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
            }
        },
    )
    global_inv_freq, global_scaling = _compute_proportional_rope_parameters(
        global_cfg, layer_type="full_attention", head_dim_key="global_head_dim"
    )

    def cos_sin(inv_freq: torch.Tensor, scaling: float) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors Gemma4TextRotaryEmbedding.forward (modeling_gemma4.py:1160-1173).
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
            "transformers/models/gemma4/modeling_gemma4.py:780-806 "
            "(rotate_half, apply_rotary_pos_emb), 1093-1173 "
            "(Gemma4TextRotaryEmbedding); "
            "transformers/modeling_rope_utils.py:187-252 "
            "(_compute_proportional_rope_parameters)"
        ),
        "shape_local": [batch, seq, heads, HEAD_DIM_LOCAL],
        "shape_global": [batch, seq, heads, HEAD_DIM_GLOBAL],
        "theta_local": ROPE_THETA_LOCAL,
        "theta_global": ROPE_THETA_GLOBAL,
        "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
        "position_ids": position_ids.squeeze(0).tolist(),
        "local_input": TensorRef(local_x),
        "global_input": TensorRef(global_x),
        "local_inv_freq": TensorRef(local_inv_freq),
        "global_inv_freq": TensorRef(global_inv_freq),
        "local_output": TensorRef(local_out),
        "global_output": TensorRef(global_out),
    }


GENERATORS = {
    "rms_norm": gen_rms_norm,
    "geglu_mlp": gen_geglu_mlp,
    "scaled_embedding": gen_scaled_embedding,
    "qk_norm_v_unscaled": gen_qk_norm_v_unscaled,
    "logit_softcap": gen_logit_softcap,
    "dual_rope": gen_dual_rope,
}


def build_manifest(op_files: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "adr": ADR,
        "generator": "scripts/gemma4_stage3_goldens.py",
        "reference_module": "transformers.models.gemma4.modeling_gemma4",
        "adr_evidence_commit": "ab1771c9e42891d893189978a8009426d70b4688",
        "runtime_versions": REFERENCE_VERSIONS,
        "fixture_format": (
            "Each op has a <op>.json sidecar (shapes/scalars/tensor refs) "
            "plus one <op>_<field>.bin per tensor field (raw little-endian "
            "float32, shape/sha256 recorded in the sidecar's ref)."
        ),
        "ops": {
            op: {
                "file": fname,
                "tolerance_max_abs_diff": TOLERANCES[op],
                "tolerance_justification": TOLERANCE_JUSTIFICATION[op],
            }
            for op, fname in op_files.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()

    validate_runtime_versions()

    op_files = {op: f"{op}.json" for op in GENERATORS}
    manifest = build_manifest(op_files)

    if args.write_fixture:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        for op, fn in GENERATORS.items():
            data = materialize_op(op, fn())
            path = FIXTURE_DIR / op_files[op]
            path.write_text(json.dumps(data, indent=2) + "\n")
            print(f"wrote {path}")
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {MANIFEST_PATH}")
        return 0

    # Verify mode: diff in-memory regeneration against committed fixtures.
    ok = True
    if not MANIFEST_PATH.exists():
        print(f"FATAL: {MANIFEST_PATH} does not exist -- run with --write-fixture first", file=sys.stderr)
        return 1
    committed_manifest = json.loads(MANIFEST_PATH.read_text())
    if committed_manifest != manifest:
        print("FATAL: manifest drift between regenerated and committed", file=sys.stderr)
        ok = False

    for op, fn in GENERATORS.items():
        data = materialize_op(f"_verify_{op}", fn())
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
        if set(committed.keys()) != set(data.keys()):
            drift = True
        else:
            for key, val in committed.items():
                new_val = data[key]
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
    print("OK: all Stage-3 goldens match the committed fixtures byte-for-byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
