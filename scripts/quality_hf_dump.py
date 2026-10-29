#!/usr/bin/env python3
"""Emit a comparator-v1 fixed-length greedy dump using local HF safetensors.

Usage: quality_hf_dump.py MODEL_DIR MANIFEST OUTPUT NAME
Requires torch and transformers with use_model_defaults support; no downloads.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

from quality_compare import DumpError, _invalid_constant, _unique_object, validate_dump


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def manifest_check(m):
    if not isinstance(m, dict) or set(m) != {"schema_version", "max_new_tokens", "cases"}:
        raise ValueError("invalid manifest keys")
    if type(m["schema_version"]) is not int or m["schema_version"] != 1:
        raise ValueError("unsupported manifest version")
    if type(m["max_new_tokens"]) is not int or m["max_new_tokens"] <= 0:
        raise ValueError("invalid max_new_tokens")
    if not isinstance(m["cases"], list) or not m["cases"]:
        raise ValueError("empty cases")
    seen = set()
    for c in m["cases"]:
        if not isinstance(c, dict) or set(c) != {"id", "prompt"} or any(not isinstance(c[k], str) or not c[k] for k in ("id", "prompt")) or c["id"] in seen:
            raise ValueError("invalid case or duplicate ID")
        seen.add(c["id"])


def protocol(maximum):
    p = load_json(Path(__file__).resolve().parent.parent / "crates/inference/quality_protocol_v1.json")
    expected = {"mode":"greedy", "stop_policy":"fixed-length-no-eos-or-stop", "temperature":0.0, "top_k":1, "top_p":1.0, "min_p":0.0, "repetition_penalty":1.0, "stop_token_ids":[], "enable_thinking":False, "enable_mtp":False, "reasoning_budget":None, "stop_strings":[], "grammar":None}
    if json.dumps(p, sort_keys=True, separators=(",", ":")) != json.dumps(expected, sort_keys=True, separators=(",", ":")):
        raise ValueError("unsupported quality protocol")
    p["max_new_tokens"] = maximum
    return p


def produce(m, name, tokenize, generate, eos_control):
    manifest_check(m)
    if not isinstance(name, str) or not name:
        raise ValueError("empty dump name")
    cfg = protocol(m["max_new_tokens"])
    if not eos_control():
        raise ValueError("early-EOS control failed")
    cases = []
    for c in m["cases"]:
        ids = tokenize(c["prompt"])
        if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("invalid prompt token ids")
        generated = generate(ids, m["max_new_tokens"])
        if not isinstance(generated, list) or len(generated) != m["max_new_tokens"] or any(type(x) is not int or x < 0 for x in generated):
            raise ValueError("incomplete generation")
        cases.append({"id":c["id"], "prompt":c["prompt"], "prompt_ids":ids, "generated_ids":generated, "stop_reason":"length", "complete":True})
    dump = {"schema_version":1, "name":name, "config":cfg, "cases":cases}
    validate_dump(dump, "output")
    return dump


def publish(path, dump):
    path = Path(path)
    data = json.dumps(dump, indent=2, allow_nan=False).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def main(argv):
    if len(argv) != 5:
        raise ValueError("usage: quality_hf_dump.py MODEL_DIR MANIFEST OUTPUT NAME")
    m = load_json(argv[2])
    manifest_check(m)
    if not argv[4]:
        raise ValueError("empty dump name")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList
    tokenizer = AutoTokenizer.from_pretrained(argv[1], local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(argv[1], local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32)
    model.eval()
    eos = model.generation_config.eos_token_id
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    vocab = model.get_output_embeddings().weight.shape[0]
    if type(eos) is not int or not 0 <= eos < vocab:
        raise ValueError("checkpoint has no valid EOS for the control")
    def cfg(maximum):
        return GenerationConfig(max_new_tokens=maximum, do_sample=False, num_beams=1,
            eos_token_id=None, forced_eos_token_id=None, forced_bos_token_id=None,
            pad_token_id=None, bos_token_id=None, min_length=0, min_new_tokens=None,
            max_time=None, stop_strings=None, repetition_penalty=1.0,
            no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
            encoder_repetition_penalty=1.0, diversity_penalty=0.0,
            num_beam_groups=1, num_return_sequences=1, bad_words_ids=None, suppress_tokens=None,
            begin_suppress_tokens=None, sequence_bias=None, renormalize_logits=False,
            constraints=None, force_words_ids=None, watermarking_config=None,
            guidance_scale=None, exponential_decay_length_penalty=None,
            remove_invalid_values=False, token_healing=False,
            temperature=None, top_k=None, top_p=None, min_p=None)
    def tokenize(prompt):
        return tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
    def run(ids, maximum, processors=None):
        inputs = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                generation_config=cfg(maximum), logits_processor=processors,
                use_model_defaults=False)
        all_ids = output[0].tolist()
        if all_ids[:len(ids)] != ids:
            raise ValueError("model changed prompt ids")
        return all_ids[len(ids):]
    def control():
        ids = tokenize(m["cases"][0]["prompt"])
        if not ids:
            return False
        class ForceEos(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if input_ids.shape[-1] == len(ids):
                    scores = torch.full_like(scores, float("-inf"))
                    scores[:, eos] = 0
                return scores
        result = run(ids, 2, LogitsProcessorList([ForceEos()]))
        ok = len(result) == 2 and result[0] == eos
        if ok:
            print(f"QUALITY_EOS_CONTROL forced_id={eos} output_count=2", file=sys.stderr)
        return ok
    dump = produce(m, argv[4], tokenize, lambda ids, n: run(ids, n), control)
    publish(argv[3], dump)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except Exception as error:
        print(f"quality HF dump refused: {error}", file=sys.stderr)
        sys.exit(2)
