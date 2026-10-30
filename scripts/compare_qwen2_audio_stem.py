#!/usr/bin/env python3
"""Compare a prebuilt synthetic CPU stem fixture with Hugging Face Qwen2AudioEncoder Conv1d + exact GELU.

Run on a host with numpy, torch and transformers:
  cargo build --release -p lattice-compute --example qwen2_audio_stem_fixture --features qwen2-audio-stem
  python3 scripts/compare_qwen2_audio_stem.py target/release/examples/qwen2_audio_stem_fixture
No checkpoint or network is used. Both intermediate stages are checked.
"""
import argparse
import json
import subprocess

import numpy as np
import torch
import transformers
import torch.nn.functional as F
from transformers import Qwen2AudioEncoderConfig
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    args = parser.parse_args()
    rng = np.random.default_rng(565)
    for frames in (1, 4, 7):
        channels, hidden = 2, 3
        x = rng.uniform(-2, 2, (channels, frames)).astype(np.float32)
        w1 = rng.uniform(-1, 1, (hidden, channels, 3)).astype(np.float32)
        w2 = rng.uniform(-1, 1, (hidden, hidden, 3)).astype(np.float32)
        b1 = rng.uniform(-0.3, 0.3, hidden).astype(np.float32)
        b2 = rng.uniform(-0.3, 0.3, hidden).astype(np.float32)
        x[0, 0] = -1.0
        x[-1, -1] = 1.0
        request = {"input_channels":channels, "frames":frames, "hidden":hidden,
                   "input":x.flatten().tolist(), "conv1":w1.flatten().tolist(), "bias1":b1.tolist(),
                   "conv2":w2.flatten().tolist(), "bias2":b2.tolist()}
        result = subprocess.run([args.fixture], input=json.dumps(request).encode(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        actual = json.loads(result.stdout)
        config = Qwen2AudioEncoderConfig(num_mel_bins=channels, d_model=hidden,
            encoder_layers=1, encoder_attention_heads=1, encoder_ffn_dim=6,
            max_source_positions=(frames + 1) // 2)
        encoder = Qwen2AudioEncoder(config).eval()
        with torch.no_grad():
            encoder.conv1.weight.copy_(torch.from_numpy(w1))
            encoder.conv1.bias.copy_(torch.from_numpy(b1))
            encoder.conv2.weight.copy_(torch.from_numpy(w2))
            encoder.conv2.bias.copy_(torch.from_numpy(b2))
            # HF encoder.forward uses these exact module calls and default GELU.
            first = F.gelu(encoder.conv1(torch.from_numpy(x[None])), approximate="none")
            second = F.gelu(encoder.conv2(first), approximate="none")
        for label, expected in (("first", first.numpy().flatten()), ("second", second.numpy().flatten())):
            observed = np.asarray(actual[label], dtype=np.float32)
            if observed.shape != expected.shape or not np.isfinite(observed).all():
                raise ValueError(f"{frames} {label}: wrong shape or nonfinite output")
            error = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
            maximum = float(error.max())
            print(f"frames={frames} stage={label} n={len(error)} max_abs={maximum:.9g}")
            if maximum > 2e-5:
                raise ValueError(f"{frames} {label}: PyTorch differential exceeded 2e-5")
        if actual["output_frames"] != (frames + 1) // 2:
            raise ValueError("stride-2 output frame count differs")
    print(f"synthetic stem differential passed: numpy={np.__version__} torch={torch.__version__} transformers={transformers.__version__}")


if __name__ == "__main__":
    main()
