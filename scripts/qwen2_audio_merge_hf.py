#!/usr/bin/env python3
"""Generate synthetic Qwen2-Audio legacy-merge oracle data; no checkpoint needed.

Requires torch and exactly transformers 4.45.2 in the caller's host environment.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration

    if transformers.__version__ != "4.45.2":
        raise SystemExit(f"expected transformers 4.45.2, got {transformers.__version__}")
    model_stub = SimpleNamespace(
        config=SimpleNamespace(audio_token_index=9, ignore_index=-100),
        pad_token_id=0,
        padding_side="left",
    )
    cases = []
    for ids, count in [([9, 1, 2], 1), ([1, 9, 2, 3], 2), ([1, 2, 9], 4), ([1, 9, 2], 750)]:
        width = 2
        text = torch.arange(1, len(ids) * width + 1, dtype=torch.float32).reshape(1, len(ids), width)
        audio = torch.arange(101, 101 + count * width, dtype=torch.float32).reshape(1, count, width)
        mask = torch.ones((1, len(ids)), dtype=torch.long)
        input_attention_mask = mask.flatten().tolist()
        merged, attention, labels, positions, final_ids = (
            Qwen2AudioForConditionalGeneration._merge_input_ids_with_audio_features(
                model_stub, audio, torch.tensor([count]), text,
                torch.tensor([ids]), mask, None,
            )
        )
        if labels is not None:
            raise RuntimeError("unexpected HF labels")
        cases.append({
            "input_ids": ids,
            "input_attention_mask": input_attention_mask,
            "text_embeddings": text.flatten().tolist(),
            "audio_embeddings": audio.flatten().tolist(),
            "final_embeddings": merged.flatten().tolist(),
            "final_ids": final_ids.flatten().tolist(),
            "attention_mask": attention.flatten().tolist(),
            "position_ids": positions.flatten().tolist(),
        })
    args.output.write_text(json.dumps({"transformers_version": transformers.__version__, "cases": cases}), encoding="utf-8")


if __name__ == "__main__":
    main()
