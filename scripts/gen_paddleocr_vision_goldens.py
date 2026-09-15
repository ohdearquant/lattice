#!/usr/bin/env python3
"""Generate the pinned PaddleOCR-VL vision and projector reference fixtures.

Requires the exact REFERENCE_VERSIONS below. Uses CPU float32, eager attention,
8 intra-op threads and 12 inter-op threads. These settings reproduced the
existing fixture; the original generator's historical runtime was not recorded.

Usage:
  python scripts/gen_paddleocr_vision_goldens.py \
    --model-dir /path/to/checkpoint --out-dir /path/to/fresh/output
  python scripts/gen_paddleocr_vision_goldens.py --self-test

Weights are read locally and verified by SHA-256. By default only two small
model-source files are downloaded from an immutable official HF revision and
hash-verified before import. For offline generation, --reference-source-dir
supplies those same two files without requiring an installed reference package.
The script installs nothing. --help and --self-test use only the standard library.

Outputs are vision_goldens.json and vision_goldens_manifest.json. The existing
fixture layout must reproduce its pinned digest before first_row_max_abs is
added. An incompatible runtime or changed reference never silently moves the
existing oracle. Cross-platform byte identity is checked, not assumed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import urllib.request


MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
REVISION = "c5630abae1d940eafe0697512a0325494b02ab42"
CASES = (("g4x4", 4, 4), ("g6x10", 6, 10), ("g12x8", 12, 8))
DTYPE = "weights bf16 upcast to f32, eager attention, use_rope=True, interpolate_pos_encoding=True"
PIXEL_FORMULA = "pixel[i,c,py,px] = ((i*7 + c*13 + py*3 + px*5) % 17) / 8 - 1; i = raster patch index"
OLD_FIELDS_SHA256 = "ea64c6e1cfad2a283ee2c878f2561631362bcff2b13aa3dbe83cdff1ff9d3783"
SOURCE_SHA256 = {
    "configuration_paddleocr_vl.py": "753dd93654c3a9c8c85a3eaee1e3092dd12591b0f2dce0305e1abfb7a41ff160",
    "modeling_paddleocr_vl.py": "c5013dff57ca8b87dc1de64d0fd839a44313de09d230a4fb2d08289d2cad5111",
}
CHECKPOINT_SHA256 = {
    "config.json": "ce7f4565f8b1db78532ad5d1b9ebe55c2139d49bd4cb04778b580a08a598f171",
    "model.safetensors": "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db",
}
REFERENCE_VERSIONS = {
    "python": "3.12.14", "numpy": "2.4.6", "torch": "2.13.0",
    "transformers": "5.12.1", "safetensors": "0.8.0", "einops": "0.8.2",
    "torchvision": "0.28.0", "huggingface-hub": "1.20.1",
    "tokenizers": "0.22.2", "pillow": "12.3.0",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_stamp(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def runtime_versions():
    actual = {"python": platform.python_version()}
    for name in REFERENCE_VERSIONS:
        if name != "python":
            try:
                actual[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual[name] = None
    return actual


def checkpoint_metadata(model_dir):
    result = {}
    for name in CHECKPOINT_SHA256:
        path = model_dir / name
        before = file_stamp(path)
        value = file_digest(path)
        require(file_stamp(path) == before, f"checkpoint changed while hashing: {name}")
        result[name] = {"sha256": value, "bytes": before[2], "stamp": before}
    return result


def source_url(name):
    return f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"


def reference_sources(source_dir):
    result = {}
    limit = 1024 * 1024
    for name in SOURCE_SHA256:
        if source_dir is not None:
            with (source_dir / name).open("rb") as stream:
                data = stream.read(limit + 1)
        else:
            request = urllib.request.Request(source_url(name), headers={"User-Agent": "lattice-vision-goldens"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(limit + 1)
        require(len(data) <= limit, f"reference source exceeds size limit: {name}")
        result[name] = data
    return result


def preflight(model_dir, source_dir, *, version_reader=runtime_versions,
              checkpoint_reader=checkpoint_metadata, source_reader=reference_sources):
    versions = version_reader()
    mismatches = {name: {"expected": expected, "actual": versions.get(name)}
                  for name, expected in REFERENCE_VERSIONS.items() if versions.get(name) != expected}
    require(not mismatches, f"reference runtime version mismatch: {mismatches}")
    checkpoint = checkpoint_reader(model_dir)
    require(set(checkpoint) == set(CHECKPOINT_SHA256), "checkpoint inventory mismatch")
    for name, expected in CHECKPOINT_SHA256.items():
        require(checkpoint[name]["sha256"] == expected, f"checkpoint digest mismatch: {name}")
    sources = source_reader(source_dir)
    require(set(sources) == set(SOURCE_SHA256), "reference source inventory mismatch")
    for name, expected in SOURCE_SHA256.items():
        require(digest(sources[name]) == expected, f"reference source digest mismatch: {name}")
    return versions, checkpoint, sources


@contextmanager
def load_reference(sources):
    package = "_lattice_paddleocr_reference"
    require(not any(name == package or name.startswith(package + ".") for name in sys.modules),
            "reference package already imported; use a fresh process")
    with tempfile.TemporaryDirectory(prefix="paddleocr-reference-") as temporary:
        directory = Path(temporary)
        (directory / "__init__.py").write_bytes(b"")
        for name, data in sources.items():
            (directory / name).write_bytes(data)
        try:
            modules = []
            for name in ("__init__", "configuration_paddleocr_vl", "modeling_paddleocr_vl"):
                qualified = package if name == "__init__" else f"{package}.{name}"
                path = directory / f"{name}.py"
                spec = importlib.util.spec_from_file_location(
                    qualified, path, submodule_search_locations=[str(directory)] if name == "__init__" else None
                )
                require(spec is not None and spec.loader is not None, f"cannot load reference module: {name}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = module
                spec.loader.exec_module(module)
                require(Path(module.__file__).resolve() == path.resolve(), "reference import escaped its temporary package")
                modules.append(module)
            yield modules[1], modules[2]
        finally:
            for name in list(sys.modules):
                if name == package or name.startswith(package + "."):
                    del sys.modules[name]


def formula_patches(np, h, w, patch):
    out = np.empty((h * w, 3, patch, patch), dtype=np.float32)
    for i in range(h * w):
        for c in range(3):
            for py in range(patch):
                for px in range(patch):
                    k = (i * 7 + c * 13 + py * 3 + px * 5) % 17
                    out[i, c, py, px] = np.float32(k) / np.float32(8.0) - np.float32(1.0)
    return out


def old_fixture_bytes(document, expected=OLD_FIELDS_SHA256):
    data = json.dumps(document, indent=1, allow_nan=False).encode("utf-8")
    actual = digest(data)
    require(actual == expected, f"existing fixture fields changed: sha256 {actual}, expected {expected}; no output published")
    return data


def measure(model_dir, config_module, modeling_module, np, torch):
    from safetensors.torch import load_file

    cfg = config_module.PaddleOCRVLConfig(**json.loads((model_dir / "config.json").read_text()))
    vcfg = cfg.vision_config
    require((vcfg.hidden_size, vcfg.num_hidden_layers, vcfg.patch_size, cfg.hidden_size) == (1152, 27, 14, 1024),
            "unexpected reference model dimensions")
    vcfg._attn_implementation = "eager"
    torch.manual_seed(0)
    # Normal CPU construction initializes the nonpersistent rotary-frequency buffer.
    visual = modeling_module.PaddleOCRVisionModel(vcfg)
    projector = modeling_module.Projector(cfg, vcfg)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    visual_state = {k[len("visual."):]: v for k, v in state.items() if k.startswith("visual.")}
    projector_state = {k[len("mlp_AR."):]: v for k, v in state.items() if k.startswith("mlp_AR.")}
    missing, unexpected = visual.load_state_dict(visual_state, strict=False)
    require(not unexpected and all(name.endswith("position_ids") for name in missing),
            f"vision state mismatch: missing={missing}, unexpected={unexpected}")
    projector.load_state_dict(projector_state, strict=True)
    inv = visual.vision_model.encoder.rotary_pos_emb.inv_freq
    rope_dim = (vcfg.hidden_size // vcfg.num_attention_heads) // 2
    require(inv.shape[0] == rope_dim // 2 and bool(torch.isfinite(inv).all()), "invalid rotary-frequency buffer")
    require(math.isclose(float(inv[1]), 1.0 / (10000.0 ** (2.0 / rope_dim)), rel_tol=1e-6),
            "rotary-frequency initialization mismatch")
    visual, projector = visual.float().eval(), projector.float().eval()
    for model in (visual, projector):
        for tensor in (*model.parameters(), *model.buffers()):
            require(tensor.device.type == "cpu", "reference tensor is not on CPU")
            require(not tensor.is_floating_point() or tensor.dtype == torch.float32, "reference tensor is not float32")
    captures, counts, handles = {}, {}, []

    def capture(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            require(tensor.device.type == "cpu" and tensor.dtype == torch.float32, f"{name}: wrong device/dtype")
            counts[name] = counts.get(name, 0) + 1
            captures[name] = tensor.detach().float().numpy().copy()
        return hook

    vm = visual.vision_model
    handles.append(vm.embeddings.register_forward_hook(capture("embed")))
    for index, layer in enumerate(vm.encoder.layers):
        handles.append(layer.register_forward_hook(capture(f"layer_{index}")))
    handles.append(vm.post_layernorm.register_forward_hook(capture("post_layernorm")))
    names = ["embed"] + [f"layer_{i}" for i in range(vcfg.num_hidden_layers)] + ["post_layernorm"]
    old = {"revision": REVISION, "dtype": DTYPE, "pixel_formula": PIXEL_FORMULA, "cases": []}
    maxima, cases = [], []
    try:
        for cid, h, w in CASES:
            n = h * w
            pv = torch.from_numpy(formula_patches(np, h, w, vcfg.patch_size)).unsqueeze(0)
            pos = torch.arange(n) % (h * w)
            cu = torch.tensor([0, n], dtype=torch.int32)
            samples = torch.zeros(n, dtype=torch.int64)
            captures.clear()
            counts.clear()
            with torch.no_grad():
                output = visual(pixel_values=pv, image_grid_thw=[(1, h, w)], position_ids=pos,
                                vision_return_embed_list=True, interpolate_pos_encoding=True,
                                sample_indices=samples, cu_seqlens=cu, return_pooler_output=False,
                                use_rope=True, window_size=-1)
                features = output.last_hidden_state
                require(isinstance(features, list) and len(features) == 1 and features[0].shape == (n, vcfg.hidden_size),
                        f"{cid}: unexpected vision output")
                merged = projector(features, torch.tensor([[1, h, w]]))
                require(isinstance(merged, list) and len(merged) == 1, f"{cid}: unexpected projector container")
                merged = merged[0].detach().float().numpy()
            require(set(counts) == set(names) and all(count == 1 for count in counts.values()), f"{cid}: checkpoint hook count mismatch")
            require(merged.shape == (n // 4, 1024) and np.isfinite(merged).all(), f"{cid}: invalid projector output")
            record = {"id": cid, "grid_h": h, "grid_w": w, "checkpoints": [], "projector": {}}
            for name in names:
                a = captures[name][0]
                require(a.shape == (n, 1152) and np.isfinite(a).all(), f"{cid}/{name}: invalid activation")
                record["checkpoints"].append({
                    "name": name, "last_tok_first8": [float(x) for x in a[-1, :8]],
                    "first_tok_first8": [float(x) for x in a[0, :8]], "mean_abs": float(np.abs(a).mean()),
                })
            record["projector"] = {
                "rows": int(merged.shape[0]), "first_row_first8": [float(x) for x in merged[0, :8]],
                "last_row_first8": [float(x) for x in merged[-1, :8]], "mean_abs": float(np.abs(merged).mean()),
                "row_mean_abs": [float(np.abs(row).mean()) for row in merged],
            }
            old["cases"].append(record)
            maxima.append(float(np.max(np.abs(merged[0, :]))))
            cases.append({"id": cid, "grid_h": h, "grid_w": w, "projector_shape": list(merged.shape),
                          "projector_sha256": digest(np.ascontiguousarray(merged, dtype="<f4").tobytes())})
    finally:
        for handle in handles:
            handle.remove()
    return old, maxima, cases


def generate(model_dir, out_dir, source_dir=None, *, version_reader=runtime_versions,
             checkpoint_reader=checkpoint_metadata, source_reader=reference_sources):
    require(not out_dir.exists() and not out_dir.is_symlink(), "output directory must not exist")
    versions, checkpoint, sources = preflight(
        model_dir, source_dir, version_reader=version_reader,
        checkpoint_reader=checkpoint_reader, source_reader=source_reader,
    )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.dont_write_bytecode = True
    import numpy as np
    import torch

    require(torch.get_default_device().type == "cpu" and torch.get_default_dtype() == torch.float32,
            "reference construction requires default CPU float32")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(12)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    require(torch.get_num_threads() == 8 and torch.get_num_interop_threads() == 12, "reference thread settings mismatch")
    with load_reference(sources) as (config_module, modeling_module):
        old, maxima, cases = measure(model_dir, config_module, modeling_module, np, torch)
    old_bytes = old_fixture_bytes(old)
    for case, value in zip(old["cases"], maxima):
        case["projector"]["first_row_max_abs"] = value
    fixture = json.dumps(old, indent=1, allow_nan=False).encode("utf-8")
    for name in CHECKPOINT_SHA256:
        require(file_stamp(model_dir / name) == checkpoint[name]["stamp"], f"checkpoint changed during generation: {name}")
    torch_config = torch.__config__.show()
    blas = re.search(r"\bBLAS_INFO=([^,\s]+)", torch_config)
    manifest = {
        "schema_version": 1,
        "generator": {"file": "gen_paddleocr_vision_goldens.py", "sha256": file_digest(Path(__file__))},
        "source": {"model_id": MODEL_ID, "revision": REVISION,
                   "reference_sources": {name: {"url": source_url(name), "sha256": digest(data), "bytes": len(data)}
                                         for name, data in sources.items()},
                   "checkpoint": {name: {"sha256": item["sha256"], "bytes": item["bytes"]}
                                  for name, item in checkpoint.items()}},
        "runtime": {"packages": versions, "device": "cpu", "dtype": "float32", "seed": 0,
                    "attention_implementation": "eager", "num_threads": torch.get_num_threads(),
                    "num_interop_threads": torch.get_num_interop_threads(),
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
                    "blas_info": blas.group(1) if blas else "unknown",
                    "torch_config_sha256": digest(torch_config.encode("utf-8"))},
        "pixel_formula": PIXEL_FORMULA,
        "projector_statistic": "first_row_max_abs = max(abs(projector[0, :])) over all 1024 columns",
        "cases": cases,
        "outputs": {"vision_goldens.json": {"sha256": digest(fixture), "bytes": len(fixture)},
                    "old_fields": {"sha256": digest(old_bytes), "byte_identical": True}},
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paddleocr-goldens-", dir=out_dir.parent) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        (staging / "vision_goldens.json").write_bytes(fixture)
        (staging / "vision_goldens_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        require(not out_dir.exists() and not out_dir.is_symlink(), "output directory appeared during generation")
        staging.rename(out_dir)
    print("[PADDLEOCR_VISION_GOLDENS] executed=true cases=3 old_fields_bytes_equal=true")


def self_test():
    from unittest.mock import patch

    require("torch" not in sys.modules and "numpy" not in sys.modules, "self-test requires a fresh process")
    sources = {name: f"synthetic {name}".encode() for name in SOURCE_SHA256}
    checkpoint = {name: {"sha256": value, "bytes": 1, "stamp": (1, 1, 1, 1, 1)}
                  for name, value in CHECKPOINT_SHA256.items()}
    controls = 0
    with tempfile.TemporaryDirectory(prefix="paddleocr-generator-self-test-") as temporary:
        root = Path(temporary)
        versions = lambda: dict(REFERENCE_VERSIONS)
        checkpoint_reader = lambda _path: {name: dict(value) for name, value in checkpoint.items()}
        source_reader = lambda _path: dict(sources)
        with patch.dict(SOURCE_SHA256, {name: digest(data) for name, data in sources.items()}):
            preflight(root, None, version_reader=versions, checkpoint_reader=checkpoint_reader, source_reader=source_reader)
            controls += 1
            def rejects(expected_message, **overrides):
                options = {"version_reader": versions, "checkpoint_reader": checkpoint_reader, "source_reader": source_reader}
                options.update(overrides)
                output = root / "must-remain-absent"
                try:
                    generate(root, output, **options)
                except (RuntimeError, FileNotFoundError) as error:
                    require(expected_message in str(error), f"wrong preflight rejection: {error}")
                else:
                    raise AssertionError("preflight accepted an invalid input")
                require(not output.exists(), "preflight failure created output")
                require("torch" not in sys.modules and "numpy" not in sys.modules, "preflight failure imported ML libraries")
            rejects("runtime version mismatch", version_reader=lambda: {**REFERENCE_VERSIONS, "torch": "wrong"})
            rejects("runtime version mismatch", version_reader=lambda: {k: v for k, v in REFERENCE_VERSIONS.items() if k != "einops"})
            for name in CHECKPOINT_SHA256:
                invalid = checkpoint_reader(root)
                invalid[name]["sha256"] = "0" * 64
                rejects("checkpoint digest mismatch", checkpoint_reader=lambda _path, data=invalid: data)
            for name in SOURCE_SHA256:
                rejects("reference source digest mismatch", source_reader=lambda _path, name=name: {**sources, name: b"changed"})
            rejects("reference source inventory mismatch", source_reader=lambda _path: {})
            def missing_source(_path):
                raise FileNotFoundError("missing reference source")
            rejects("missing reference source", source_reader=missing_source)
            controls += 8
        sample = {"revision": "synthetic", "cases": []}
        expected = digest(json.dumps(sample, indent=1).encode())
        old_fixture_bytes(sample, expected)
        try:
            old_fixture_bytes({**sample, "revision": "changed"}, expected)
        except RuntimeError as error:
            require("existing fixture fields changed" in str(error), "wrong old-field rejection")
        else:
            raise AssertionError("old-field digest guard accepted drift")
        controls += 2
    require("torch" not in sys.modules and "numpy" not in sys.modules, "self-test imported ML libraries")
    print(f"[PADDLEOCR_VISION_GENERATOR_SELF_TEST] passed={controls} ml_imports=false network=false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--reference-source-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(args.model_dir is None and args.out_dir is None and args.reference_source_dir is None,
                "--self-test does not accept generation paths")
        self_test()
        return
    if args.model_dir is None or args.out_dir is None:
        parser.error("generation requires --model-dir and --out-dir")
    generate(args.model_dir.expanduser().resolve(), args.out_dir.expanduser().absolute(),
             args.reference_source_dir.expanduser().resolve() if args.reference_source_dir else None)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        print(f"PaddleOCR vision generation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
