#!/usr/bin/env python3
"""HF differential golden fixtures for the Qwen3.5-0.8B vision path (ADR-069).

Pins the HF `transformers` reference output for a fixed procedural test image
and a fixed prompt, BEFORE any lattice engine vision forward-pass code exists.
Retires ADR-069 risk R1 (ViT correctness, consumed at S3) and R2 (M-RoPE
position-id correctness, consumed at S5) by giving those future stages a
committed target to diff against. This script performs no lattice engine
work — it only drives the HF reference implementation and writes fixtures.

Methodology (record exactly, every regeneration):
  - Checkpoint source: `--model-dir` (default `~/.lattice/models/qwen3.5-0.8b`),
    verified byte-identical (`config.json`, `model.safetensors.index.json`) to
    HF Hub `Qwen/Qwen3.5-0.8B` commit `2fc06364715b967f1860aea9cf38778875588b17`
    at the time this script was authored (2026-07-14). The checkpoint directory
    itself does not ship `tokenizer_config.json` / `chat_template.jinja` /
    `preprocessor_config.json` / `video_preprocessor_config.json` (only
    `config.json`, `model.safetensors*`, `tokenizer.json`); this script fetches
    those four small (~25 KB total) processor/tokenizer config files from that
    pinned Hub commit and merges them with the local weights in a scratch
    directory. Model weights are ALWAYS read from `--model-dir`, never from the
    network.
  - Runtime: `transformers==5.12.1` (pinned in this repo's `pyproject.toml`
    dev-dependencies alongside `torch`/`torchvision`/`pillow`), CPU, float32,
    single BLAS thread (`torch.set_num_threads(1)`) for reproducibility.
  - Image: a 256x256 deterministic procedural PNG (gradient background + a
    few fixed solid shapes), generated in-script from a fixed seed — see
    `make_golden_image`. 256x256 is the SMALLEST square image that clears the
    checkpoint's own `min_pixels=65536` processor floor while staying aligned
    to `patch_size(16) * spatial_merge_size(2) = 32`; it yields a 16x16 patch
    grid -> 64 post-merge visual tokens (dozens-to-~256 range, not thousands).
  - Prompt: "Describe this image." rendered through the checkpoint's chat
    template (default non-thinking mode).
  - Decode: greedy (`do_sample=False`), 8 new tokens, no repetition penalty.

Usage:
    uv run python scripts/vision_goldens_qwen35.py \\
        --model-dir ~/.lattice/models/qwen3.5-0.8b \\
        --out tests/fixtures/vision/

Re-running with unchanged inputs (checkpoint, this script, the pinned HF
processor revision, the pinned transformers version) reproduces byte-identical
fixtures — see `tests/fixtures/vision/README.md` for the determinism proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw

HF_MODEL_ID = "Qwen/Qwen3.5-0.8B"
# Pinned HF Hub commit for the four processor/tokenizer config files the local
# checkpoint directory does not ship. Verified byte-identical `config.json`
# against this same commit before use (see `verify_checkpoint_matches_hub`).
HF_PROCESSOR_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PROCESSOR_FILES = [
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]

IMAGE_SIZE = 256  # see module docstring: smallest size clearing min_pixels=65536
IMAGE_SEED = 20260714
PROMPT = "Describe this image."
MAX_NEW_TOKENS = 8
SCHEMA_VERSION = 1

# Runtime versions this fixture set was actually generated with (pinned exactly
# in pyproject.toml `[tool.uv] dev-dependencies`). `validate_runtime_versions`
# rejects any drift from these BEFORE the model is loaded, so a `uv run` that
# resolves a newer torch/torchvision/pillow/transformers fails closed instead
# of silently regenerating a fixture with different CPU numerics.
REFERENCE_VERSIONS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "pillow": "12.3.0",
    "transformers": "5.12.1",
}

# Files whose byte-identity to the pinned HF Hub revision is load-bearing: the
# checkpoint's own weight layout (`model.safetensors.index.json`) and config
# must both match, or `build_merged_dir` would silently symlink shards from a
# checkpoint that has drifted from the pinned revision.
PINNED_HUB_FILES = ("config.json", "model.safetensors.index.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_golden_image(path: Path) -> None:
    """Deterministic procedural test image: RGB gradient + fixed solid shapes.

    Fully seeded (no wall-clock or OS entropy); byte-identical PNG on every
    invocation for a fixed PIL/numpy version pair.
    """
    rng = np.random.default_rng(IMAGE_SEED)
    x = np.linspace(0, 255, IMAGE_SIZE, dtype=np.float32)
    y = np.linspace(0, 255, IMAGE_SIZE, dtype=np.float32)
    r = np.tile(x, (IMAGE_SIZE, 1))
    g = np.tile(y.reshape(-1, 1), (1, IMAGE_SIZE))
    b = np.full((IMAGE_SIZE, IMAGE_SIZE), 128.0, dtype=np.float32)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    shapes = [
        ("rectangle", (20, 20, 90, 90), (220, 30, 30)),
        ("ellipse", (140, 30, 230, 120), (30, 220, 60)),
        ("rectangle", (40, 150, 130, 230), (30, 60, 220)),
        ("ellipse", (150, 150, 220, 220), (250, 250, 40)),
    ]
    # rng is consulted (not hardcoded) so the shape set stays reproducibly
    # tied to IMAGE_SEED even though the coordinates above are fixed; a small
    # deterministic jitter keeps the content non-degenerate without breaking
    # byte-reproducibility across runs (same seed -> same jitter).
    jitter = rng.integers(-3, 4, size=(len(shapes), 4))
    for (kind, box, color), dj in zip(shapes, jitter):
        jbox = tuple(int(c + d) for c, d in zip(box, dj))
        if kind == "rectangle":
            draw.rectangle(jbox, fill=color)
        else:
            draw.ellipse(jbox, fill=color)

    img.save(path, format="PNG", optimize=False, compress_level=6)


def _fetch_hub_raw(name: str) -> bytes:
    url = f"https://huggingface.co/{HF_MODEL_ID}/raw/{HF_PROCESSOR_REVISION}/{name}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def _compare_pinned_files(model_dir: Path, fetch) -> None:
    """Byte-compare every file in `PINNED_HUB_FILES` against the pinned Hub
    revision. `fetch` is injected (name -> bytes) so this is unit-testable
    offline; production callers pass `_fetch_hub_raw`.

    Checks BOTH `config.json` and `model.safetensors.index.json` so a
    checkpoint that kept a matching config but drifted its shard layout
    (index.json) is rejected too, not just a config-only drift.
    """
    for name in PINNED_HUB_FILES:
        remote = fetch(name)
        local = (model_dir / name).read_bytes()
        if remote != local:
            raise RuntimeError(
                f"{model_dir}/{name} does not match HF Hub {HF_MODEL_ID}@{HF_PROCESSOR_REVISION}; "
                "the checkpoint has drifted from the pinned revision (or the pinned processor files "
                "fetched below would not correspond to this checkpoint). "
                "Re-pin HF_PROCESSOR_REVISION to a commit matching your local checkpoint, or restore "
                "the pinned checkpoint."
            )


def verify_checkpoint_matches_hub(model_dir: Path) -> None:
    _compare_pinned_files(model_dir, _fetch_hub_raw)


def validate_runtime_versions(actual: dict) -> None:
    """Reject a runtime whose torch/torchvision/pillow/transformers versions
    differ from `REFERENCE_VERSIONS` — called BEFORE the model is loaded.

    The manifest still records the actual versions observed (provenance), but
    that recording is not itself a drift signal: this check is the drift
    signal, and it fails closed before any fixture bytes are produced.
    """
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in REFERENCE_VERSIONS.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "runtime dependency versions differ from the pinned reference this fixture set was "
            f"generated with: {mismatches}. Install the exact pins from pyproject.toml, or if this "
            "is an intentional upgrade, update REFERENCE_VERSIONS and regenerate+re-review the "
            "fixtures (version drift can change CPU forward-pass numerics)."
        )


def fetch_processor_files(merged_dir: Path) -> None:
    for name in PROCESSOR_FILES:
        url = f"https://huggingface.co/{HF_MODEL_ID}/raw/{HF_PROCESSOR_REVISION}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear setup error
            raise RuntimeError(
                f"failed to fetch {name} from HF Hub ({HF_MODEL_ID}@{HF_PROCESSOR_REVISION}): {exc}. "
                f"If offline, manually place {PROCESSOR_FILES} (from that pinned commit) into "
                f"{merged_dir} and re-run."
            ) from exc
        (merged_dir / name).write_bytes(data)


def build_merged_dir(model_dir: Path, merged_dir: Path) -> None:
    """Symlink checkpoint weights + tokenizer into a scratch dir alongside the
    fetched processor config files, so AutoProcessor/AutoModel can load the
    real checkpoint without those files being written into `--model-dir`."""
    merged_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        target = merged_dir / name
        target.unlink(missing_ok=True)
        target.symlink_to((model_dir / name).resolve())
    for weight_file in model_dir.glob("model.safetensors*"):
        if weight_file.is_symlink() or weight_file.suffix == ".json":
            continue
        target = merged_dir / weight_file.name
        target.unlink(missing_ok=True)
        target.symlink_to(weight_file.resolve())
    fetch_processor_files(merged_dir)


def tensor_summary(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def write_f32_bin(path: Path, arr: np.ndarray) -> dict:
    flat = np.ascontiguousarray(arr, dtype="<f4")
    data = flat.tobytes()
    path.write_bytes(data)
    return {
        "path": path.name,
        "shape": list(arr.shape),
        "dtype": "float32",
        "endianness": "little",
        "sha256": sha256_bytes(data),
        "num_elements": int(flat.size),
    }


def write_json(path: Path, obj) -> dict:
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return {"path": path.name, "sha256": sha256_bytes(text.encode("utf-8"))}


def run(model_dir: Path, out_dir: Path) -> None:
    import torch
    import torchvision
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Fail closed on runtime drift BEFORE the model is loaded or any output is
    # created — a version mismatch can silently change CPU forward-pass numerics.
    validate_runtime_versions(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
            "transformers": transformers.__version__,
        }
    )

    torch.manual_seed(IMAGE_SEED)
    torch.set_num_threads(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    verify_checkpoint_matches_hub(model_dir)

    image_path = out_dir / "golden_image.png"
    make_golden_image(image_path)
    image_bytes = image_path.read_bytes()

    with tempfile.TemporaryDirectory(prefix="qwen35-vision-goldens-") as tmp:
        merged_dir = Path(tmp) / "model_merged"
        build_merged_dir(model_dir, merged_dir)

        processor = AutoProcessor.from_pretrained(str(merged_dir))
        model = AutoModelForImageTextToText.from_pretrained(str(merged_dir), dtype=torch.float32)
        model.eval()

        img = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[chat_text], images=[img], return_tensors="pt")

        with torch.no_grad():
            vision_out = model.model.visual(
                enc["pixel_values"].to(torch.float32),
                grid_thw=enc["image_grid_thw"],
                return_dict=True,
            )
            pos_ids, _mrope_deltas = model.model.get_rope_index(
                enc["input_ids"],
                enc["mm_token_type_ids"],
                image_grid_thw=enc["image_grid_thw"],
                attention_mask=enc["attention_mask"],
            )
            generated = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        input_ids = enc["input_ids"][0].tolist()
        prompt_len = enc["input_ids"].shape[1]
        greedy_new_tokens = generated[0, prompt_len:].tolist()
        image_grid_thw = enc["image_grid_thw"].tolist()
        num_placeholder_tokens = sum(1 for t in input_ids if t == model.config.image_token_id)

        pre_merger = vision_out.last_hidden_state.to(torch.float32).numpy()
        post_merger = vision_out.pooler_output.to(torch.float32).numpy()
        pixel_values_np = enc["pixel_values"].to(torch.float32).numpy()
        position_ids_np = pos_ids[:, 0, :].to(torch.int64).numpy()  # (3, seq_len)

        transformers_version = transformers.__version__
        torch_version = torch.__version__
        torchvision_version = torchvision.__version__
        pillow_version = PIL.__version__
        python_version = sys.version.split()[0]

    # --- sanity checks (also recorded in manifest.json / asserted by the caller) ---
    text_prefix_len = min(
        i for i, t in enumerate(input_ids) if t == model.config.vision_start_token_id
    )
    text_prefix_collapses_1d = bool(
        np.array_equal(position_ids_np[0, :text_prefix_len], position_ids_np[1, :text_prefix_len])
        and np.array_equal(position_ids_np[1, :text_prefix_len], position_ids_np[2, :text_prefix_len])
    )
    image_positions = [i for i, t in enumerate(input_ids) if t == model.config.image_token_id]
    image_span_diverges = bool(
        len(image_positions) > 0
        and not np.array_equal(
            position_ids_np[1, image_positions], position_ids_np[2, image_positions]
        )
    )
    post_merger_matches_placeholders = post_merger.shape[0] == num_placeholder_tokens

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "adr": "ADR-069",
        "source": {
            "hf_model_id": HF_MODEL_ID,
            "hf_processor_revision": HF_PROCESSOR_REVISION,
            "model_dir_config_sha256": sha256_file(model_dir / "config.json"),
            "model_dir_safetensors_index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
            "transformers_version": transformers_version,
            "torch_version": torch_version,
            "torchvision_version": torchvision_version,
            "pillow_version": pillow_version,
            "python_version": python_version,
        },
        "image": {
            "path": image_path.name,
            "size": [IMAGE_SIZE, IMAGE_SIZE],
            "seed": IMAGE_SEED,
            "sha256": sha256_bytes(image_bytes),
        },
        "prompt": PROMPT,
        "decode": {
            "strategy": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": "float32",
            "device": "cpu",
            "num_threads": 1,
        },
        "image_grid_thw": image_grid_thw,
        "num_image_placeholder_tokens": num_placeholder_tokens,
        "pixel_values_summary": tensor_summary(pixel_values_np),
        "sanity_checks": {
            "post_merger_count_matches_placeholder_count": post_merger_matches_placeholders,
            "text_prefix_positions_collapse_1d": text_prefix_collapses_1d,
            "image_span_positions_diverge": image_span_diverges,
        },
    }

    manifest["vit_pre_merger"] = write_f32_bin(out_dir / "vit_pre_merger_f32.bin", pre_merger)
    manifest["vit_post_merger"] = write_f32_bin(out_dir / "vit_post_merger_f32.bin", post_merger)
    manifest["input_ids"] = write_json(out_dir / "input_ids.json", input_ids)
    manifest["position_ids"] = write_json(
        out_dir / "position_ids.json",
        {"shape": [3, len(input_ids)], "axes": ["t", "h", "w"], "values": position_ids_np.tolist()},
    )
    manifest["greedy_tokens"] = write_json(
        out_dir / "greedy_tokens.json",
        {"max_new_tokens": MAX_NEW_TOKENS, "token_ids": greedy_new_tokens},
    )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not post_merger_matches_placeholders:
        raise RuntimeError(
            f"post-merger embedding count {post_merger.shape[0]} != "
            f"image placeholder token count {num_placeholder_tokens}"
        )
    if not text_prefix_collapses_1d:
        raise RuntimeError("text-prefix position ids did not collapse to 1-D (t == h == w)")
    if not image_span_diverges:
        raise RuntimeError("image-span position ids did not diverge across axes")

    print(f"wrote fixtures to {out_dir}")
    print(json.dumps(manifest["sanity_checks"], indent=2))


def self_test() -> None:
    """Offline unit tests for the fail-closed regeneration guards. No network
    access, no model load — exercises `_compare_pinned_files` and
    `validate_runtime_versions` directly with injected/synthetic data.

    Exits non-zero (via AssertionError) on the first guard that does not fail
    closed the way it is supposed to.
    """
    # --- _compare_pinned_files: matching files pass; a drifted index.json or
    # config.json must each fail BEFORE any output would be created. ---
    with tempfile.TemporaryDirectory(prefix="vision-goldens-selftest-") as tmp:
        model_dir = Path(tmp)
        (model_dir / "config.json").write_bytes(b'{"model_type": "qwen3_5_vl"}')
        (model_dir / "model.safetensors.index.json").write_bytes(b'{"weight_map": {}}')

        def fetch_matching(name: str) -> bytes:
            return (model_dir / name).read_bytes()

        _compare_pinned_files(model_dir, fetch_matching)  # must not raise

        def fetch_drifted_index(name: str) -> bytes:
            if name == "model.safetensors.index.json":
                return b'{"weight_map": {"drifted": "yes"}}'
            return (model_dir / name).read_bytes()

        try:
            _compare_pinned_files(model_dir, fetch_drifted_index)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "_compare_pinned_files did not fail closed on a drifted model.safetensors.index.json"
            )

        def fetch_drifted_config(name: str) -> bytes:
            if name == "config.json":
                return b'{"model_type": "something_else"}'
            return (model_dir / name).read_bytes()

        try:
            _compare_pinned_files(model_dir, fetch_drifted_config)
        except RuntimeError:
            pass
        else:
            raise AssertionError("_compare_pinned_files did not fail closed on a drifted config.json")

    # --- validate_runtime_versions: exact reference match passes; any single
    # dependency drifting from REFERENCE_VERSIONS must fail. ---
    validate_runtime_versions(dict(REFERENCE_VERSIONS))  # must not raise

    for name in REFERENCE_VERSIONS:
        drifted = dict(REFERENCE_VERSIONS)
        drifted[name] = "0.0.0-selftest-drift"
        try:
            validate_runtime_versions(drifted)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"validate_runtime_versions did not fail closed on {name!r} drift")

    print("self-test: OK (checkpoint-drift guard and version guard both fail closed as expected)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=Path("~/.lattice/models/qwen3.5-0.8b").expanduser())
    p.add_argument("--out", type=Path, default=Path("tests/fixtures/vision/"))
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline unit tests for the fail-closed regeneration guards and exit "
        "(no network, no model load, no --model-dir/--out required).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"--model-dir {model_dir} does not exist")
    run(model_dir, args.out)


if __name__ == "__main__":
    main()
