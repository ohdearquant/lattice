#!/usr/bin/env python3
"""Compare a prebuilt synthetic CPU layer fixture against Transformers 4.45.2.

Run after building the fixture with feature qwen2-audio-stem:
  python3 scripts/compare_qwen2_audio_layer.py target/release/examples/qwen2_audio_layer_fixture
Uses deterministic synthetic f32 weights and inputs; no checkpoint or network.
"""
import argparse
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import Qwen2AudioEncoderConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoderLayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    args = parser.parse_args()
    if transformers.__version__ != "4.45.2":
        raise RuntimeError(f"requires pinned transformers 4.45.2, got {transformers.__version__}")
    torch.set_num_threads(1)
    rng = np.random.default_rng(5651)
    for time, hidden, heads, ffn, valid in [(1, 4, 1, 6, 1), (3, 4, 2, 7, 3), (4, 6, 2, 9, 2)]:
        def random(shape, scale=0.3):
            return rng.uniform(-scale, scale, shape).astype(np.float32)
        x = random((time, hidden), 2.0)
        values = {
            "norm1_weight": random(hidden) + 1, "norm1_bias": random(hidden),
            "q_weight": random((hidden, hidden)), "q_bias": random(hidden),
            "k_weight": random((hidden, hidden)),
            "v_weight": random((hidden, hidden)), "v_bias": random(hidden),
            "out_weight": random((hidden, hidden)), "out_bias": random(hidden),
            "norm2_weight": random(hidden) + 1, "norm2_bias": random(hidden),
            "fc1_weight": random((ffn, hidden)), "fc1_bias": random(ffn),
            "fc2_weight": random((hidden, ffn)), "fc2_bias": random(hidden),
        }
        request = {"time": time, "hidden": hidden, "heads": heads, "ffn": ffn,
                   "valid_keys": valid, "input": x.flatten().tolist()}
        request.update({k: a.flatten().tolist() for k, a in values.items()})
        result = subprocess.run([args.fixture], input=json.dumps(request).encode(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        observed = json.loads(result.stdout)
        config = Qwen2AudioEncoderConfig(d_model=hidden, encoder_attention_heads=heads,
                                         encoder_ffn_dim=ffn, activation_function="gelu")
        config._attn_implementation = "eager"
        layer = Qwen2AudioEncoderLayer(config).eval()
        modules = {"norm1": layer.self_attn_layer_norm, "query": layer.self_attn.q_proj,
                   "key": layer.self_attn.k_proj, "value": layer.self_attn.v_proj,
                   "attention": layer.self_attn.out_proj, "norm2": layer.final_layer_norm,
                   "fc1": layer.fc1}
        mapping = {"norm1": "norm1", "query": "q", "key": "k", "value": "v",
                   "attention": "out", "norm2": "norm2", "fc1": "fc1"}
        with torch.no_grad():
            for name, module in modules.items():
                prefix = mapping[name]
                module.weight.copy_(torch.from_numpy(values[prefix + "_weight"]))
                if module.bias is not None:
                    module.bias.copy_(torch.from_numpy(values[prefix + "_bias"]))
            if layer.self_attn.k_proj.bias is not None:
                raise RuntimeError("HF key projection unexpectedly has bias")
            layer.fc2.weight.copy_(torch.from_numpy(values["fc2_weight"]))
            layer.fc2.bias.copy_(torch.from_numpy(values["fc2_bias"]))
            captures = {}
            hooks = []
            for name, module in modules.items():
                def capture(_module, _input, output, name=name):
                    captures[name] = output.detach().clone()
                hooks.append(module.register_forward_hook(capture))
            tensor = torch.from_numpy(x[None])
            mask = torch.zeros((1, 1, time, time), dtype=torch.float32)
            mask[:, :, :, valid:] = float("-inf")
            output = layer(tensor, attention_mask=mask, layer_head_mask=None)[0]
            residual = tensor + captures["attention"]
            reference = {"norm1": captures["norm1"], "query": captures["query"],
                         "key": captures["key"], "value": captures["value"],
                         "attention": captures["attention"], "residual": residual,
                         "norm2": captures["norm2"],
                         "activated": F.gelu(captures["fc1"], approximate="none"), "output": output}
            for hook in hooks:
                hook.remove()
            if valid < time:
                unmasked = layer(tensor, attention_mask=None, layer_head_mask=None)[0]
                if torch.max(torch.abs(output - unmasked)).item() < 1e-4:
                    raise RuntimeError("mask control is insensitive")
        for name, tensor in reference.items():
            expected = tensor.numpy().flatten()
            actual = np.asarray(observed[name], dtype=np.float32)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise RuntimeError(f"{name}: shape mismatch or nonfinite")
            delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            maximum = float(delta.max())
            rms = float(np.sqrt(np.mean(delta * delta)))
            print(f"time={time} hidden={hidden} heads={heads} valid={valid} {name} max_abs={maximum:.9g} rms={rms:.9g}")
            if maximum > 2e-5:
                raise RuntimeError(f"{name}: max absolute error exceeds 2e-5")
    print(f"synthetic layer differential passed: numpy={np.__version__} torch={torch.__version__} transformers={transformers.__version__}")


if __name__ == "__main__":
    main()
