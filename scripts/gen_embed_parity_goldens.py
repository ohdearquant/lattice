#!/usr/bin/env python3
"""Generate HF reference embeddings for embedding parity regression tests.

Writes JSON golden fixtures to crates/embed/tests/fixtures/embed_parity_v1/.
Each fixture file contains the HF-computed embedding vector, the tokenizer
input_ids, pooling strategy, and prompt prefix used.

Run once to generate fixtures, then commit them. The Rust test
crates/embed/tests/embed_parity_vs_hf.rs loads these fixtures and compares
lattice's computed embeddings against them.

Usage:
    uv run --with transformers --with torch --with numpy \
        scripts/gen_embed_parity_goldens.py

Requirements:
    - Model weights in ~/.lattice/models/{slug}/ (for E5, Qwen)
    - BAAI/bge-small-en-v1.5 loaded from HF hub (cached automatically)
    - transformers, torch, numpy via uv
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed. Run with:")
    print("  uv run --with transformers --with torch --with numpy scripts/gen_embed_parity_goldens.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fixture inputs — fixed set so goldens are reproducible.
# These are the same inputs used by the Rust parity test.
# ---------------------------------------------------------------------------
INPUTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pure Rust transformer inference engine.",
    "Café résumé naïve façade — Unicode test.",
    "   leading whitespace and    multiple    spaces   ",
    "短い日本語のテストです。",
]

HOME = os.environ["HOME"]
REPO_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = REPO_ROOT / "crates" / "embed" / "tests" / "fixtures" / "embed_parity_v1"


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


def generate_bge_small_goldens() -> list[dict]:
    """
    BAAI/bge-small-en-v1.5: CLS-token pooling + L2 normalize, no prompt prefix.

    Reference: https://huggingface.co/BAAI/bge-small-en-v1.5
    "Use the CLS token embedding and normalize to unit length."
    """
    model_id = "BAAI/bge-small-en-v1.5"
    print(f"Loading {model_id} from HF hub...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    print(f"  model type: {type(model).__name__}, hidden_size: {model.config.hidden_size}")

    goldens = []
    for text in INPUTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token = position 0
        cls_vec = outputs.last_hidden_state[0, 0].numpy()
        embedding = l2_normalize(cls_vec)

        goldens.append({
            "model_id": model_id,
            "pooling": "cls",
            "prompt_prefix": "",
            "input": text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "embedding": embedding.tolist(),
            "embedding_dim": len(embedding),
        })
        print(f"  [{len(goldens)}/5] '{text[:40]}...' → dim={len(embedding)}, norm={np.linalg.norm(embedding):.6f}")

    return goldens


def generate_e5_small_goldens() -> list[dict]:
    """
    intfloat/multilingual-e5-small: mean pooling + L2 normalize, 'passage: ' prefix.

    Reference: https://huggingface.co/intfloat/multilingual-e5-small
    "For documents, prepend 'passage: ' before embedding."
    The query side uses 'query: ', but for this golden set we generate passage
    embeddings (matching what embed_passage() produces in lattice).
    """
    model_path = Path(HOME) / ".lattice" / "models" / "multilingual-e5-small"
    if not (model_path / "model.safetensors").exists():
        print(f"ERROR: E5-small not found at {model_path}")
        print("Download it first: see scripts instructions")
        sys.exit(1)

    model_id = "intfloat/multilingual-e5-small"
    print(f"Loading {model_id} from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    model.eval()

    print(f"  model type: {type(model).__name__}, hidden_size: {model.config.hidden_size}")

    prompt_prefix = "passage: "
    goldens = []
    for text in INPUTS:
        prompted = prompt_prefix + text
        inputs = tokenizer(prompted, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Masked mean pool
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # [1, seq_len, 1]
        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        mean_vec = pooled[0].numpy()
        embedding = l2_normalize(mean_vec)

        goldens.append({
            "model_id": model_id,
            "pooling": "mean",
            "prompt_prefix": prompt_prefix,
            "input": text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "embedding": embedding.tolist(),
            "embedding_dim": len(embedding),
        })
        print(f"  [{len(goldens)}/5] '{text[:40]}...' → dim={len(embedding)}, norm={np.linalg.norm(embedding):.6f}")

    return goldens


def find_hf_cache_snapshot(model_id: str) -> Path | None:
    """Return the first existing snapshot dir for a HF model ID, or None."""
    # e.g. "sentence-transformers/all-MiniLM-L6-v2"
    # → "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>/"
    slug = "models--" + model_id.replace("/", "--")
    snapshots_dir = Path(HOME) / ".cache" / "huggingface" / "hub" / slug / "snapshots"
    if snapshots_dir.exists():
        children = sorted(snapshots_dir.iterdir())
        if children:
            return children[-1]  # latest snapshot
    return None


def generate_all_minilm_l6_v2_goldens() -> list[dict]:
    """
    sentence-transformers/all-MiniLM-L6-v2: mean pooling + L2 normalize, no prompt prefix.

    Reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    "Map sentences & paragraphs to a 384-dimensional dense vector space."
    WordPiece tokenizer (BERT-base-uncased style), mean pooling with attention mask.
    """
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    # Use the HF cache snapshot for full tokenizer config; weights are there too.
    model_path = find_hf_cache_snapshot(model_id)
    if model_path is None or not (model_path / "model.safetensors").exists():
        # Fallback: .lattice/models/ (weights only — will fail without tokenizer config)
        model_path = Path(HOME) / ".lattice" / "models" / "all-minilm-l6-v2"
        if not (model_path / "model.safetensors").exists():
            print(f"ERROR: all-MiniLM-L6-v2 not found in HF cache or .lattice/models/")
            sys.exit(1)

    print(f"Loading {model_id} from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    model.eval()

    print(f"  model type: {type(model).__name__}, hidden_size: {model.config.hidden_size}")

    prompt_prefix = ""
    goldens = []
    for text in INPUTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Masked mean pool (sentence-transformers convention)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # [1, seq_len, 1]
        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        mean_vec = pooled[0].numpy()
        embedding = l2_normalize(mean_vec)

        goldens.append({
            "model_id": model_id,
            "pooling": "mean",
            "prompt_prefix": prompt_prefix,
            "input": text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "embedding": embedding.tolist(),
            "embedding_dim": len(embedding),
        })
        print(f"  [{len(goldens)}/5] '{text[:40]}...' → dim={len(embedding)}, norm={np.linalg.norm(embedding):.6f}")

    return goldens


def generate_paraphrase_multilingual_minilm_l12_v2_goldens() -> list[dict]:
    """
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: mean pooling + L2 normalize.

    Reference: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    "Maps sentences & paragraphs to a 384 dimensional dense vector space."
    SentencePiece tokenizer (XLM-R style), mean pooling with attention mask, no prompt prefix.
    """
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Use the HF cache snapshot for full tokenizer config; weights are there too.
    model_path = find_hf_cache_snapshot(model_id)
    if model_path is None or not (model_path / "model.safetensors").exists():
        # Fallback: .lattice/models/
        model_path = Path(HOME) / ".lattice" / "models" / "paraphrase-multilingual-minilm-l12-v2"
        if not (model_path / "model.safetensors").exists():
            print(f"ERROR: paraphrase-multilingual-MiniLM-L12-v2 not found in HF cache or .lattice/models/")
            sys.exit(1)

    print(f"Loading {model_id} from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    model.eval()

    print(f"  model type: {type(model).__name__}, hidden_size: {model.config.hidden_size}")

    prompt_prefix = ""
    goldens = []
    for text in INPUTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Masked mean pool (sentence-transformers convention)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # [1, seq_len, 1]
        pooled = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        mean_vec = pooled[0].numpy()
        embedding = l2_normalize(mean_vec)

        goldens.append({
            "model_id": model_id,
            "pooling": "mean",
            "prompt_prefix": prompt_prefix,
            "input": text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "embedding": embedding.tolist(),
            "embedding_dim": len(embedding),
        })
        print(f"  [{len(goldens)}/5] '{text[:40]}...' → dim={len(embedding)}, norm={np.linalg.norm(embedding):.6f}")

    return goldens


def generate_qwen_goldens() -> list[dict]:
    """
    Qwen/Qwen3-Embedding-0.6B: last-token pooling + L2 normalize.

    Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    "Use last-token pooling." For queries, a task instruction is prepended.
    For documents (passage side), no prefix is used.

    We generate document embeddings (no prefix) here to match embed_passage()
    behavior (Qwen document_instruction() returns None).
    """
    model_path = Path(HOME) / ".lattice" / "models" / "qwen3-embedding-0.6b"
    if not (model_path / "model.safetensors").exists():
        print(f"ERROR: Qwen3-Embedding-0.6B not found at {model_path}")
        sys.exit(1)

    model_id = "Qwen/Qwen3-Embedding-0.6B"
    print(f"Loading {model_id} from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    model.eval()

    print(f"  model type: {type(model).__name__}")

    prompt_prefix = ""  # passage side for Qwen: no prefix (document_instruction() = None)
    goldens = []
    for text in INPUTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs)

        # Last token pool (position = seq_len - 1).
        # Qwen3 outputs BFloat16 on CPU; cast to f32 before converting to numpy.
        last_token_vec = outputs.last_hidden_state[0, seq_len - 1].float().numpy()
        embedding = l2_normalize(last_token_vec)

        goldens.append({
            "model_id": model_id,
            "pooling": "last_token",
            "prompt_prefix": prompt_prefix,
            "input": text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "embedding": embedding.tolist(),
            "embedding_dim": len(embedding),
        })
        print(f"  [{len(goldens)}/5] '{text[:40]}...' → dim={len(embedding)}, norm={np.linalg.norm(embedding):.6f}")

    return goldens


def write_fixture(filename: str, goldens: list[dict]) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURE_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(goldens, f, indent=2, ensure_ascii=False)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {len(goldens)} goldens to {out_path} ({size_kb:.1f} KB)")


def main() -> None:
    print("=== Generating HF parity goldens ===")
    print(f"Fixture output dir: {FIXTURE_DIR}")
    print()

    # BGE-small-en-v1.5
    print("--- BGE-small-en-v1.5 (CLS + L2 norm, no prefix) ---")
    bge_goldens = generate_bge_small_goldens()
    write_fixture("bge_small_en_v15.json", bge_goldens)
    print()

    # multilingual-e5-small
    print("--- multilingual-e5-small (mean + L2 norm, 'passage: ' prefix) ---")
    e5_goldens = generate_e5_small_goldens()
    write_fixture("multilingual_e5_small.json", e5_goldens)
    print()

    # all-MiniLM-L6-v2
    print("--- all-MiniLM-L6-v2 (mean + L2 norm, no prefix) ---")
    minilm_l6_goldens = generate_all_minilm_l6_v2_goldens()
    write_fixture("all_minilm_l6_v2.json", minilm_l6_goldens)
    print()

    # paraphrase-multilingual-MiniLM-L12-v2
    print("--- paraphrase-multilingual-MiniLM-L12-v2 (mean + L2 norm, no prefix) ---")
    paraphrase_goldens = generate_paraphrase_multilingual_minilm_l12_v2_goldens()
    write_fixture("paraphrase_multilingual_minilm_l12_v2.json", paraphrase_goldens)
    print()

    # Qwen3-Embedding-0.6B
    print("--- Qwen3-Embedding-0.6B (last-token + L2 norm, no prefix) ---")
    qwen_goldens = generate_qwen_goldens()
    write_fixture("qwen3_embedding_0_6b.json", qwen_goldens)
    print()

    print("=== Done. Fixture summary ===")
    for fname, goldens in [
        ("bge_small_en_v15.json", bge_goldens),
        ("multilingual_e5_small.json", e5_goldens),
        ("all_minilm_l6_v2.json", minilm_l6_goldens),
        ("paraphrase_multilingual_minilm_l12_v2.json", paraphrase_goldens),
        ("qwen3_embedding_0_6b.json", qwen_goldens),
    ]:
        path = FIXTURE_DIR / fname
        size_kb = path.stat().st_size / 1024
        print(f"  {fname}: {len(goldens)} entries, {size_kb:.1f} KB, dim={goldens[0]['embedding_dim']}")

    print()
    print("Run the Rust regression test:")
    print("  cargo test -p lattice-embed --test embed_parity_vs_hf")


if __name__ == "__main__":
    main()
