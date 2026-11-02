#!/usr/bin/env python3
"""Differential Whisper 128-bin log-mel check; needs numpy and librosa.

From the repository root: python3 scripts/compare_whisper_log_mel.py
This builds/runs the CPU fixture and exits nonzero unless absolute L2 < 1e-5.
"""

import subprocess
import sys

import librosa
import numpy as np

RATE = 16000
samples = (0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(RATE) / RATE)).astype("<f4")
windowed = librosa.stft(samples, n_fft=400, hop_length=160, win_length=400,
                        window="hann", center=True, pad_mode="reflect")
power = np.abs(windowed[:, :-1]) ** 2
filters = librosa.filters.mel(sr=RATE, n_fft=400, n_mels=128,
                              fmin=0.0, fmax=8000.0, htk=False, norm="slaney")
mel = filters @ power
reference = np.log10(np.maximum(mel, np.float32(1e-10)))
reference = np.maximum(reference, reference.max() - np.float32(8.0))
reference = ((reference + np.float32(4.0)) / np.float32(4.0)).astype(np.float32)

proc = subprocess.run(
    ["cargo", "run", "--quiet", "--release", "-p", "lattice-compute", "--example", "whisper_mel_fixture"],
    input=samples.tobytes(), stdout=subprocess.PIPE, check=True,
)
if len(proc.stdout) < 4:
    raise RuntimeError("fixture returned no frame count")
frames = int.from_bytes(proc.stdout[:4], "little")
actual = np.frombuffer(proc.stdout[4:], dtype="<f4")
if frames != 100 or actual.size != 128 * frames or reference.shape != (128, frames):
    raise RuntimeError(f"shape mismatch: frames={frames}, values={actual.size}, reference={reference.shape}")
actual = actual.reshape(128, frames)
if not np.isfinite(actual).all() or not np.isfinite(reference).all():
    raise RuntimeError("non-finite features")
difference = actual.astype(np.float64) - reference.astype(np.float64)
l2 = float(np.linalg.norm(difference))
print(f"numpy={np.__version__} librosa={librosa.__version__} shape={reference.shape} "
      f"absolute_l2={l2:.12g} max_abs={np.max(np.abs(difference)):.12g} "
      f"rms={np.sqrt(np.mean(difference ** 2)):.12g}")
if not l2 < 1e-5:
    sys.exit("Whisper log-mel differential failed: absolute L2 must be < 1e-5")
