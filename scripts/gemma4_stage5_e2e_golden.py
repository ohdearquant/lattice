#!/usr/bin/env python3
"""HF differential golden fixture for the Gemma 4 E2B text-only CPU forward +
greedy generate (ADR-082 Stage 5).

Loads the real `google/gemma-4-E2B-it` checkpoint from
`$LATTICE_MODEL_CACHE/gemma-4-e2b-it` (default `~/.lattice/models/gemma-4-e2b-it`,
downloaded via `hf download google/gemma-4-E2B-it --revision <pinned>`), runs
one fixed short prompt through the real `transformers` Gemma4 reference
(CPU, f32, 1 thread) and records:

  - `input_ids`: the exact token ids fed to both HF and lattice (BOS + prompt,
    no chat template -- add_special_tokens=False on the prompt text, BOS
    prepended explicitly so both sides agree byte-for-byte on tokenization).
  - `greedy_tokens`: the first 3 greedy-decoded token ids (do_sample=False).
  - `final_logits`: post-softcap logits at the prompt's last position
    (length vocab_size=262144 would be large; this fixture stores only the
    top-8 by absolute HF value plus their indices, sufficient to prove
    argmax/near-argmax parity without a 1MB+ fixture).
  - `hidden_states`: per-layer hidden state at the LAST prompt position,
    captured AFTER layers {0, 4, 15, 34} (before the final `model.norm`), via
    a forward hook directly on each `Gemma4TextDecoderLayer`, **not**
    `output_hidden_states=True`'s tuple: empirically verified that tuple's
    last entry is `last_hidden_state` (POST `model.norm`), not the raw
    pre-norm output of the final decoder layer -- entries 0..num_layers are
    "input to layer i" (entry 0 = embeddings), and the tuple never carries a
    separate raw pre-norm post-layer-(N-1) entry. A forward hook sidesteps
    that indexing entirely and is correct at every layer, including the
    last.

Run (uv, pinned reference transformers, per this repo's Stage-3 convention):

  uv run --with "transformers @ git+https://github.com/huggingface/transformers@ab1771c9e42891d893189978a8009426d70b4688" \
    --with torch --with accelerate python3 scripts/gemma4_stage5_e2e_golden.py
"""
import json
import os
import sys

import torch

MODEL_DIR = os.environ.get(
    "GEMMA4_MODEL_DIR",
    os.path.expanduser("~/.lattice/models/gemma-4-e2b-it"),
)
PROBE_LAYERS = [0, 4, 15, 34]
PROMPT = "The capital of France is"
OUT_PATH = "crates/inference/tests/fixtures/gemma4/stage5/e2e_golden.json"

# Sliding-window boundary coverage (ADR-082 stage 5 review finding 3):
# sliding_window=512 means a token at position p attends to keys
# [max(0, p - (window - 1)), p]. A prompt of BOUNDARY_LENGTHS tokens puts the
# LAST position at index (length - 1):
#   length=511 -> last position 510, window [0, 510]   (full history, window not yet binding)
#   length=512 -> last position 511, window [0, 511]   (exactly window-sized, boundary edge)
#   length=513 -> last position 512, window [1, 512]   (token 0 must be evicted)
# Reusing PROMPT's own token ids (cycled) keeps every id a real, previously
# tokenizer-verified id -- no need to hand-pick synthetic ids that might
# collide with reserved/special ranges.
BOUNDARY_LENGTHS = [511, 512, 513]
BOUNDARY_OUT_PATH = "crates/inference/tests/fixtures/gemma4/stage5/boundary_golden.json"


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)

    from transformers import AutoTokenizer
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    with open(f"{MODEL_DIR}/config.json") as f:
        cfg_dict = json.load(f)
    cfg = Gemma4Config(**cfg_dict)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    bos_id = cfg.text_config.bos_token_id
    prompt_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
    input_ids_list = [bos_id] + prompt_ids
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    print("loading model (CPU f32, text-only)...", file=sys.stderr)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_DIR, config=cfg, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()

    captured = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden_states[0, -1, :].detach().clone()

        return hook

    for layer_idx in PROBE_LAYERS:
        layer_module = model.model.language_model.layers[layer_idx]
        hooks.append(layer_module.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        text_out = model.model.language_model(
            input_ids=input_ids, output_hidden_states=False, use_cache=False
        )
        for h in hooks:
            h.remove()
        assert set(captured.keys()) == set(PROBE_LAYERS), (
            f"forward hooks did not fire for all probe layers: got {sorted(captured.keys())}"
        )
        probe = {layer: captured[layer].tolist() for layer in PROBE_LAYERS}

        logits = model.lm_head(text_out.last_hidden_state)
        cap = cfg.text_config.final_logit_softcapping
        logits = torch.tanh(logits / cap) * cap
        final_logits_last = logits[0, -1, :]
        top = torch.topk(final_logits_last, 8)

        gen = model.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )
        greedy_tokens = gen[0, input_ids.shape[1] :].tolist()

    fixture = {
        "metadata": {
            "source_repo": "google/gemma-4-E2B-it",
            "revision": "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf",
            "prompt": PROMPT,
            "reference_commit": "ab1771c9e42891d893189978a8009426d70b4688",
            "generator": "scripts/gemma4_stage5_e2e_golden.py",
        },
        "bos_token_id": bos_id,
        "prompt_token_ids": prompt_ids,
        "input_ids": input_ids_list,
        "greedy_tokens": greedy_tokens,
        "probe_layers": PROBE_LAYERS,
        "hidden_state_last_pos": {str(k): v for k, v in probe.items()},
        "final_logits_last_pos_top8": {
            "indices": top.indices.tolist(),
            "values": top.values.tolist(),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    print(json.dumps({k: v for k, v in fixture.items() if k != "hidden_state_last_pos"}, indent=2))

    cap = cfg.text_config.final_logit_softcapping
    boundary_cases = []
    with torch.no_grad():
        for length in BOUNDARY_LENGTHS:
            ids = [bos_id]
            i = 0
            while len(ids) < length:
                ids.append(prompt_ids[i % len(prompt_ids)])
                i += 1
            assert len(ids) == length
            ids_t = torch.tensor([ids], dtype=torch.long)

            out = model.model.language_model(
                input_ids=ids_t, output_hidden_states=False, use_cache=False
            )
            logits = model.lm_head(out.last_hidden_state)
            logits = torch.tanh(logits / cap) * cap
            last_logits = logits[0, -1, :]
            top = torch.topk(last_logits, 8)

            gen = model.generate(
                input_ids=ids_t, max_new_tokens=2, do_sample=False, use_cache=True
            )
            greedy_tokens = gen[0, ids_t.shape[1] :].tolist()

            boundary_cases.append(
                {
                    "length": length,
                    "input_ids": ids,
                    "final_logits_last_pos_top8": {
                        "indices": top.indices.tolist(),
                        "values": top.values.tolist(),
                    },
                    "greedy_tokens": greedy_tokens,
                }
            )
            print(f"boundary length={length} done", file=sys.stderr)

    boundary_fixture = {
        "metadata": fixture["metadata"],
        "cases": boundary_cases,
    }
    with open(BOUNDARY_OUT_PATH, "w") as f:
        json.dump(boundary_fixture, f, indent=2)
    print(f"wrote {BOUNDARY_OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
