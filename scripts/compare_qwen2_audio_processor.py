#!/usr/bin/env python3
"""Compare the Qwen2-Audio processor against a local pinned HF preprocessor.

Build the release fixture on the host, then run:
  python scripts/compare_qwen2_audio_processor.py MODEL_DIR FIXTURE_BINARY
MODEL_DIR must contain the pinned Qwen2-Audio preprocessor_config.json.
Requires numpy, torch and transformers; does not download a checkpoint.
"""
import argparse
import subprocess
import sys

import numpy as np
from transformers import WhisperFeatureExtractor

RATE = 16000
MAX_SAMPLES = 480000
FRAMES = 3000


def parse_output(data):
    if len(data) != 8 + FRAMES + 128 * FRAMES * 4:
        raise ValueError(f"unexpected output size {len(data)}")
    frames = int.from_bytes(data[:4], "little")
    tokens = int.from_bytes(data[4:8], "little")
    if frames != FRAMES:
        raise ValueError(f"expected {FRAMES} frames, got {frames}")
    mask = np.frombuffer(data, dtype=np.uint8, count=FRAMES, offset=8)
    features = np.frombuffer(data, dtype="<f4", offset=8 + FRAMES).reshape(128, FRAMES)
    if not np.isfinite(features).all():
        raise ValueError("non-finite output")
    return mask, tokens, features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("fixture")
    args = ap.parse_args()
    reference = WhisperFeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
    actual_profile = (reference.feature_size, reference.sampling_rate, reference.n_fft,
                      reference.hop_length, reference.n_samples, reference.nb_max_frames,
                      reference.padding_side, reference.padding_value)
    expected_profile = (128, RATE, 400, 160, MAX_SAMPLES, FRAMES, "right", 0.0)
    if actual_profile != expected_profile:
        raise ValueError(f"unexpected checkpoint preprocessor: {actual_profile}")
    tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(RATE) / RATE)).astype("<f4")
    short = np.zeros(321, dtype="<f4")
    short[0], short[-1] = 0.5, -0.25
    cases = [("short_boundary", short), ("tone_1s", tone), ("full_silence", np.zeros(MAX_SAMPLES, dtype="<f4"))]
    for label, pcm in cases:
        ref = reference(pcm, sampling_rate=RATE, return_attention_mask=True,
                        padding="max_length", truncation=True, return_tensors="np")
        ref_features = np.asarray(ref["input_features"][0], dtype=np.float32)
        ref_mask = np.asarray(ref["attention_mask"][0], dtype=np.uint8)
        output = subprocess.run([args.fixture], input=pcm.tobytes(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True)
        mask, tokens, features = parse_output(output.stdout)
        if ref_features.shape != (128, FRAMES) or ref_mask.shape != (FRAMES,):
            raise ValueError(f"{label}: invalid reference shapes")
        if not np.array_equal(mask, ref_mask):
            raise ValueError(f"{label}: feature attention mask differs")
        length = int(ref_mask.sum())
        expected_tokens = ((length - 1) // 2 + 1 - 2) // 2 + 1
        if tokens != expected_tokens:
            raise ValueError(f"{label}: token count {tokens} != {expected_tokens}")
        delta = features.astype(np.float64) - ref_features.astype(np.float64)
        max_abs = float(np.max(np.abs(delta)))
        rms = float(np.sqrt(np.mean(delta * delta)))
        print(f"{label}: frames={FRAMES} valid={length} tokens={tokens} max_abs={max_abs:.9g} rms={rms:.9g}")
        if not np.isfinite(ref_features).all() or max_abs > 2e-5 or rms > 1e-5:
            raise ValueError(f"{label}: feature mismatch")
    rejected = subprocess.run([args.fixture], input=np.zeros(MAX_SAMPLES + 1, dtype="<f4").tobytes(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if rejected.returncode == 0 or rejected.stdout:
        raise ValueError("oversize input was not refused before output")
    print("Qwen2-Audio processor differential passed (features and masks)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Qwen2-Audio processor differential failed: {error}", file=sys.stderr)
        sys.exit(1)
