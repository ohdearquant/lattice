#!/usr/bin/env python3
"""Offline 224x224 synthetic Qwen3-VL differential against Transformers 4.57.1.

Pass the prebuilt qwen3_vl_synthetic_fixture executable. No checkpoint or network.
The HF processor is explicitly set to no resize; this is not default preprocessing.
"""
import argparse
import json
import subprocess

import numpy as np
import torch
import transformers
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    args = parser.parse_args()
    if transformers.__version__ != "4.57.1":
        raise RuntimeError(f"requires transformers 4.57.1, got {transformers.__version__}")
    torch.set_num_threads(1)
    rng = np.random.default_rng(5641)
    def random(shape, scale=0.2):
        return rng.uniform(-scale, scale, shape).astype(np.float32)
    y, x = np.indices((224, 224))
    image = np.stack([(((x * (5+c) + y * (11+c*3) + (x//16)*17 + (y//16)*29 + c*37) % 251) / 125.0 - 1.0)
                      for c in range(3)]).astype(np.float32)
    v = {"patch_weight": random((8,3,2,16,16), 0.003), "patch_bias": random(8),
         "positions": random((2304,8)), "qkv_weight": random((24,8)), "qkv_bias": random(24),
         "proj_weight": random((8,8)), "proj_bias": random(8),
         "fc1_weight": random((16,8)), "fc1_bias": random(16, 1.2),
         "fc2_weight": random((8,16)), "fc2_bias": random(8),
         "norm1_weight": random(8) + 1, "norm1_bias": random(8),
         "norm2_weight": random(8) + 1, "norm2_bias": random(8),
         "merger_norm_weight": random(8) + 1, "merger_norm_bias": random(8),
         "merger_fc1_weight": random((32,32)), "merger_fc1_bias": random(32, 1.2),
         "merger_fc2_weight": random((6,32)), "merger_fc2_bias": random(6)}
    request = {"image": image.flatten().tolist()}
    request.update({k: a.flatten().tolist() for k, a in v.items()})
    result = subprocess.run([args.fixture], input=json.dumps(request).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    observed = json.loads(result.stdout)
    processor = Qwen2VLImageProcessorFast(patch_size=16, temporal_patch_size=2, merge_size=2)
    processed = processor(images=image, do_resize=False, do_rescale=False, do_normalize=False,
                          do_convert_rgb=False, input_data_format="channels_first", return_tensors="pt")
    patches, grid = processed["pixel_values"], processed["image_grid_thw"]
    if tuple(patches.shape) != (196,1536) or grid.tolist() != [[1,14,14]]:
        raise RuntimeError(f"unexpected processor shape/grid: {patches.shape}, {grid}")
    cfg = Qwen3VLVisionConfig(depth=1, hidden_size=8, intermediate_size=16, num_heads=2,
        in_channels=3, patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
        out_hidden_size=6, num_position_embeddings=2304, deepstack_visual_indexes=[],
        hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = "eager"
    model = Qwen3VLVisionModel(cfg).eval()
    b, m = model.blocks[0], model.merger
    mapping = [(model.patch_embed.proj,"patch"), (b.attn.qkv,"qkv"), (b.attn.proj,"proj"),
               (b.mlp.linear_fc1,"fc1"), (b.mlp.linear_fc2,"fc2"), (b.norm1,"norm1"),
               (b.norm2,"norm2"), (m.norm,"merger_norm"), (m.linear_fc1,"merger_fc1"),
               (m.linear_fc2,"merger_fc2")]
    with torch.no_grad():
        for target, prefix in mapping:
            target.weight.copy_(torch.from_numpy(v[prefix + "_weight"]))
            target.bias.copy_(torch.from_numpy(v[prefix + "_bias"]))
        model.pos_embed.weight.copy_(torch.from_numpy(v["positions"]))
        captures = {}
        hooks = [model.patch_embed.register_forward_hook(
                    lambda _m,_i,out: captures.update(patch_embedding=out.detach().clone())),
                 b.register_forward_hook(lambda _m,_i,out: captures.update(block=out.detach().clone()))]
        positions = model.fast_pos_embed_interpolate(grid)
        merged, deepstack = model(patches, grid)
        for hook in hooks:
            hook.remove()
        if deepstack:
            raise RuntimeError("unexpected DeepStack output")
        reference = {"patches": patches, "patch_embedding": captures["patch_embedding"],
                     "positions": positions, "block": captures["block"], "merged": merged}
        if torch.max(torch.abs(positions)).item() < 1e-4:
            raise RuntimeError("position control is insensitive")
        if torch.max(torch.abs(captures["block"] - captures["patch_embedding"] - positions)).item() < 1e-4:
            raise RuntimeError("block control is insensitive")
        reordered = captures["block"].view(49,4,8).flip(1).reshape(196,8)
        if torch.max(torch.abs(model.merger(reordered) - merged)).item() < 1e-5:
            raise RuntimeError("merger-order control is insensitive")
    for name, tensor in reference.items():
        expected = tensor.numpy().flatten()
        actual = np.asarray(observed[name], dtype=np.float32)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise RuntimeError(f"{name}: shape mismatch or nonfinite")
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        maximum, l2 = float(np.max(np.abs(delta))), float(np.linalg.norm(delta))
        print(f"{name}: count={actual.size} max_abs={maximum:.9g} l2={l2:.9g}")
        if maximum > 2e-5 or l2 >= 1e-4:
            raise RuntimeError(f"{name}: differential tolerance failed")
    print(f"synthetic 224 no-resize differential passed: numpy={np.__version__} torch={torch.__version__} transformers={transformers.__version__}")


if __name__ == "__main__":
    main()
