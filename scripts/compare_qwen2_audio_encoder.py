#!/usr/bin/env python3
"""Compare a synthetic encoder fixture to Transformers 4.45.2, offline.

Build with qwen2-audio-stem, then pass the path to qwen2_audio_encoder_fixture.
Uses no checkpoint; actual model support and full-size performance are out of scope.
"""
import argparse
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import Qwen2AudioEncoderConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioEncoder, Qwen2AudioMultiModalProjector,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    args = parser.parse_args()
    if transformers.__version__ != "4.45.2":
        raise RuntimeError(f"requires transformers 4.45.2, got {transformers.__version__}")
    torch.set_num_threads(1)
    rng = np.random.default_rng(5654)
    cases = [(2, 4, 0), (3, 3, 1), (4, 5, 2)]
    for time, valid_frames, count in cases:
        hidden, heads, ffn, channels, output = 4, 2, 7, 3, 5
        frames = 2 * time  # HF encoder requires exactly 2 * max_source_positions.
        valid_keys = (valid_frames + 1) // 2
        def random(shape, scale=0.3):
            return rng.uniform(-scale, scale, shape).astype(np.float32)
        x = random((channels, frames), 1.0)
        values = {"conv1_weight": random((hidden, channels, 3)), "conv1_bias": random(hidden),
                  "conv2_weight": random((hidden, hidden, 3)), "conv2_bias": random(hidden),
                  "positions": random((time, hidden), 1.0), "norm_weight": random(hidden) + 1,
                  "norm_bias": random(hidden), "projector_weight": random((output, hidden)),
                  "projector_bias": random(output)}
        layers = []
        for _ in range(count):
            layers.append({"norm1_weight": random(hidden) + 1, "norm1_bias": random(hidden),
                "q_weight": random((hidden, hidden)), "q_bias": random(hidden),
                "k_weight": random((hidden, hidden)), "v_weight": random((hidden, hidden)),
                "v_bias": random(hidden), "out_weight": random((hidden, hidden)), "out_bias": random(hidden),
                "norm2_weight": random(hidden) + 1, "norm2_bias": random(hidden),
                "fc1_weight": random((ffn, hidden)), "fc1_bias": random(ffn),
                "fc2_weight": random((hidden, ffn)), "fc2_bias": random(hidden)})
        request = {"input_channels": channels, "frames": frames, "valid_frames": valid_frames,
                   "hidden": hidden, "heads": heads, "ffn": ffn, "output": output,
                   "input": x.flatten().tolist(),
                   "layers": [{k: a.flatten().tolist() for k, a in d.items()} for d in layers]}
        request.update({k: a.flatten().tolist() for k, a in values.items()})
        result = subprocess.run([args.fixture], input=json.dumps(request).encode(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        observed = json.loads(result.stdout)
        cfg = Qwen2AudioEncoderConfig(d_model=hidden, num_mel_bins=channels,
            encoder_attention_heads=heads, encoder_ffn_dim=ffn, encoder_layers=count,
            max_source_positions=time, dropout=0.0, attention_dropout=0.0,
            activation_dropout=0.0, encoder_layerdrop=0.0, activation_function="gelu")
        cfg._attn_implementation = "eager"
        enc = Qwen2AudioEncoder(cfg).eval()
        projector = Qwen2AudioMultiModalProjector(SimpleNamespace(
            audio_config=cfg, text_config=SimpleNamespace(hidden_size=output))).eval()
        with torch.no_grad():
            for module, prefix in [(enc.conv1, "conv1"), (enc.conv2, "conv2"),
                                   (enc.layer_norm, "norm"), (projector.linear, "projector")]:
                module.weight.copy_(torch.from_numpy(values[prefix + "_weight"]))
                module.bias.copy_(torch.from_numpy(values[prefix + "_bias"]))
            enc.embed_positions.weight.copy_(torch.from_numpy(values["positions"]))
            for module, spec in zip(enc.layers, layers):
                mapping = [(module.self_attn_layer_norm, "norm1"), (module.self_attn.q_proj, "q"),
                    (module.self_attn.k_proj, "k"), (module.self_attn.v_proj, "v"),
                    (module.self_attn.out_proj, "out"), (module.final_layer_norm, "norm2"),
                    (module.fc1, "fc1"), (module.fc2, "fc2")]
                for target, prefix in mapping:
                    target.weight.copy_(torch.from_numpy(spec[prefix + "_weight"]))
                    if target.bias is not None:
                        target.bias.copy_(torch.from_numpy(spec[prefix + "_bias"]))
                if module.self_attn.k_proj.bias is not None:
                    raise RuntimeError("unexpected key projection bias")
            inp = torch.from_numpy(x[None])
            first = F.gelu(enc.conv1(inp), approximate="none")
            stem = F.gelu(enc.conv2(first), approximate="none")
            positioned = stem.transpose(1, 2) + enc.embed_positions.weight
            captured = {}
            hooks = []
            if count:
                hooks.append(enc.layers[-1].register_forward_hook(
                    lambda _m, _i, out: captured.update(layers=out[0].detach().clone())))
            hooks.append(enc.avg_pooler.register_forward_hook(
                lambda _m, _i, out: captured.update(pooled=out.detach().transpose(1, 2).clone())))
            mask = torch.zeros((1, 1, time, time), dtype=torch.float32)
            mask[:, :, :, valid_keys:] = float("-inf")
            normalized = enc(inp, attention_mask=mask).last_hidden_state
            projected = projector(normalized)
            for hook in hooks:
                hook.remove()
            last = captured.get("layers", positioned)
            reference = {"stem": stem, "positioned": positioned, "layers": last,
                         "pooled": captured["pooled"], "normalized": normalized, "projected": projected}
            if count and valid_keys < time:
                unmasked = enc(inp, attention_mask=None).last_hidden_state
                if torch.max(torch.abs(normalized - unmasked)).item() < 1e-6:
                    raise RuntimeError("valid-key mask control is insensitive")
        if observed["time"] != time or observed["valid_tokens"] != valid_keys // 2:
            raise RuntimeError("invalid physical or valid-token length")
        for name, tensor in reference.items():
            expected = tensor.numpy().flatten()
            actual = np.asarray(observed[name], dtype=np.float32)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise RuntimeError(f"{name}: shape mismatch or nonfinite")
            delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            maximum = float(delta.max())
            rms = float(np.sqrt(np.mean(delta * delta)))
            print(f"time={time} valid={valid_keys} layers={count} {name} max_abs={maximum:.9g} rms={rms:.9g}")
            if maximum > 2e-5:
                raise RuntimeError(f"{name}: max absolute error exceeds 2e-5")
    print(f"synthetic encoder differential passed: numpy={np.__version__} torch={torch.__version__} transformers={transformers.__version__}")


if __name__ == "__main__":
    main()
