#!/usr/bin/env python3
"""Generate/verify Gemma 4 E2B tokenizer + chat-template parity fixtures
(ADR-082 Stage 1, G17).

Downloads `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
and `processor_config.json` from the checkpoint's pinned revision (the
`/resolve/<commit>/` URL form, so the pin is real, not advisory), records
their SHA-256 in a committed provenance manifest, and uses the HF
`tokenizers` library (NOT `transformers.AutoTokenizer`, which does not know
the `gemma4` architecture yet) plus `jinja2` to produce golden token IDs over
a declared corpus, a 2-turn chat-template rendering, exact byte-fallback
decode goldens (including invalid/incomplete UTF-8 byte runs), and the
Stage-1 marker-expansion arithmetic goldens (ADR-082 G11/G15/G17: image
placeholders expand to a fixed vision soft-token count, audio placeholders
to a duration-derived, capped count) sourced from the fetched
`processor_config.json`, not hardcoded.

Pinned checkpoint (ADR-082): `google/gemma-4-E2B-it` @ revision
`9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`.

Usage:
    # Verify (default): re-fetch the four files, diff their SHA-256 against
    # the committed manifest, regenerate goldens in-memory and diff against
    # the committed golden JSON. Fails closed on any drift. Touches the
    # network — not run by CI, a manual/periodic drift check.
    uv run python3 scripts/gemma4_tokenizer_goldens.py

    # Regenerate the committed fixtures (deliberate, reviewable, never run
    # by CI):
    uv run python3 scripts/gemma4_tokenizer_goldens.py --write-fixture
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO = "google/gemma-4-E2B-it"
REVISION = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}"

# Hard cap on total bytes fetched by this script across all four files.
# tokenizer.json is ~32.2 MB; 40 MB leaves headroom for the three small
# sidecar files without coming remotely close to the ~10.25 GB weight
# payload a mistargeted URL could otherwise pull in.
MAX_FETCH_BYTES = 40_000_000

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "inference"
    / "tests"
    / "fixtures"
    / "gemma4"
    / "tokenizer"
)
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
CORPUS_GOLDENS_PATH = FIXTURE_DIR / "corpus_goldens.json"
CHAT_TEMPLATE_GOLDEN_PATH = FIXTURE_DIR / "chat_template_golden.json"
DECODE_GOLDENS_PATH = FIXTURE_DIR / "decode_goldens.json"
EXPANSION_GOLDENS_PATH = FIXTURE_DIR / "expansion_goldens.json"
TOKENIZER_JSON_PATH = FIXTURE_DIR / "tokenizer.json"
TOKENIZER_CONFIG_PATH = FIXTURE_DIR / "tokenizer_config.json"
CHAT_TEMPLATE_JINJA_PATH = FIXTURE_DIR / "chat_template.jinja"
PROCESSOR_CONFIG_PATH = FIXTURE_DIR / "processor_config.json"

FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
]

# Byte-fallback decode goldens (ADR-082 Stage 1 major finding): `<0xE5>` (id
# 467) and `<0x8F>` (id 381) are the first two bytes of the 3-byte UTF-8
# encoding of "叫" (id 409 = `<0xAB>`, the third byte). `[467]` and
# `[467, 381]` are, respectively, a lone incomplete lead byte and a 2-byte
# run still missing its final continuation byte -- both invalid on their
# own. The ordinary-token id is resolved from the live vocab (`"a"`) rather
# than hardcoded, since only the byte-fallback ids are pinned by the
# required-goldens contract.
DECODE_CASES: list[tuple[str, list[int]]] = [
    ("decode_lone_incomplete_lead_byte", [467]),
    ("decode_two_byte_incomplete_run", [467, 381]),
    ("decode_valid_three_byte_fallback_run", [467, 381, 409]),
    ("decode_incomplete_run_then_ordinary_token", [467, 381, "a"]),
]

# Runtime versions this script was validated against (pinned in uv.lock as
# transitive deps of `transformers`). A version drift is not necessarily
# wrong, but it changes tokenization/rendering behavior silently enough
# that goldens generated under a different version are not trustworthy
# without re-review — fail closed rather than silently regenerate under an
# unvalidated toolchain.
EXPECTED_TOKENIZERS_VERSION = "0.22.2"
EXPECTED_JINJA2_VERSION = "3.1.6"

# Declared corpus (ADR-082 Stage 1 gate): ordinary text, Unicode (CJK +
# emoji + combining marks), whitespace runs, byte-fallback cases, the
# image/audio wrapper markers, and thought/tool markers.
CORPUS: list[tuple[str, str, str]] = [
    ("ascii_ordinary", "text", "The quick brown fox jumps over the lazy dog."),
    (
        "ascii_punctuation",
        "text",
        "Hello, world! How are you? (fine, thanks) -- 100% sure.",
    ),
    (
        "unicode_cjk",
        "unicode",
        "短い日本語のテストです。"
        "今日は良い天気ですね。",
    ),
    (
        "unicode_emoji",
        "unicode",
        "Great job! \U0001f600\U0001f680\U0001f389 family: "
        "\U0001f468‍\U0001f469‍\U0001f467",
    ),
    (
        "unicode_combining",
        "unicode",
        "combining: é vs é (decomposed e + U+0301 vs precomposed)",
    ),
    (
        "unicode_mixed",
        "unicode",
        "Café résumé naïve façade -- "
        "日本語 mixed with emoji \U0001f600",
    ),
    ("whitespace_multi_space", "whitespace", "a  b   c    d"),
    ("whitespace_tabs_newlines", "whitespace", "line1\tcol2\nline2\n\nline4"),
    (
        "whitespace_leading_trailing",
        "whitespace",
        "   leading and trailing spaces   ",
    ),
    ("whitespace_only", "whitespace", "   "),
    ("empty", "whitespace", ""),
    (
        "byte_fallback_rare_char",
        "byte_fallback",
        "rare: \U00010300 old italic letter",
    ),
    ("byte_fallback_private_use", "byte_fallback", "pua: \ue000\ue001 chars"),
    (
        "byte_fallback_replacement_char",
        "byte_fallback",
        "bad: � sequence",
    ),
    ("marker_image", "marker", "<|image|>"),
    ("marker_audio", "marker", "<|audio|>"),
    (
        "marker_image_audio_combo",
        "marker",
        "<|image|> describe this <|audio|> transcribe this",
    ),
    ("marker_boi_eoi", "marker", "<|image>content<image|>"),
    ("marker_boa_eoa", "marker", "<|audio>content<audio|>"),
    ("marker_video", "marker", "<|video|>"),
    (
        "thought_tool_markers",
        "marker",
        "<|think|><|channel>thought\nreasoning\n<channel|>"
        "<|turn>model\nanswer<turn|>"
        "<|tool>declaration<tool|>"
        "<|tool_call>call:foo{}<tool_call|>"
        "<|tool_response>response:foo{}<tool_response|>",
    ),
]

CHAT_MESSAGES = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, remaining_budget: int) -> tuple[bytes, int]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (pinned https HF URL)
        if resp.status != 200:
            raise RuntimeError(f"expected HTTP 200 fetching {url}, got {resp.status}")
        read_cap = remaining_budget + 1
        data = resp.read(read_cap)
        if len(data) > remaining_budget:
            raise RuntimeError(
                f"fetch of {url} exceeded the remaining budget "
                f"({remaining_budget} bytes) -- refusing to continue reading"
            )
        return data, remaining_budget - len(data)


def fetch_all() -> dict[str, bytes]:
    budget = MAX_FETCH_BYTES
    out: dict[str, bytes] = {}
    for name in FILES:
        data, budget = _fetch(f"{BASE_URL}/{name}", budget)
        out[name] = data
        print(f"  fetched {name}: {len(data)} bytes, sha256={_sha256_bytes(data)}")
    return out


def _validate_runtime_versions() -> None:
    import jinja2
    import tokenizers

    if tokenizers.__version__ != EXPECTED_TOKENIZERS_VERSION:
        raise RuntimeError(
            f"tokenizers=={tokenizers.__version__}, expected "
            f"{EXPECTED_TOKENIZERS_VERSION} -- goldens generated under a "
            "different tokenizers version are not trustworthy without "
            "re-review; pin via uv.lock or update EXPECTED_TOKENIZERS_VERSION "
            "deliberately"
        )
    if jinja2.__version__ != EXPECTED_JINJA2_VERSION:
        raise RuntimeError(
            f"jinja2=={jinja2.__version__}, expected {EXPECTED_JINJA2_VERSION} "
            "-- see tokenizers version check above for rationale"
        )
    print(
        f"  runtime versions OK: tokenizers=={tokenizers.__version__}, "
        f"jinja2=={jinja2.__version__}"
    )


def generate_goldens(
    files: dict[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from tokenizers import Tokenizer

    # `Tokenizer.from_str` loads directly from the fetched JSON text -- no
    # temporary file is written or needs cleaning up.
    tok = Tokenizer.from_str(files["tokenizer.json"].decode("utf-8"))

    corpus_goldens = []
    for case_id, category, text in CORPUS:
        enc = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(enc.ids)
        corpus_goldens.append(
            {
                "id": case_id,
                "category": category,
                "text": text,
                "ids": enc.ids,
                "tokens": enc.tokens,
                "decoded": decoded,
            }
        )

    import jinja2

    env = jinja2.Environment()
    template = env.from_string(files["chat_template.jinja"].decode("utf-8"))
    rendered = template.render(
        messages=CHAT_MESSAGES,
        bos_token="<bos>",
        eos_token="<eos>",
        add_generation_prompt=True,
        tools=None,
    )
    rendered_ids = tok.encode(rendered, add_special_tokens=False).ids
    chat_golden = {
        "description": "2-turn conversation (user, assistant), add_generation_prompt=True",
        "messages": CHAT_MESSAGES,
        "rendered_text": rendered,
        "ids": rendered_ids,
    }

    vocab = tok.get_vocab()
    decode_goldens = []
    for case_id, ids in DECODE_CASES:
        resolved_ids = [vocab[i] if isinstance(i, str) else i for i in ids]
        decode_goldens.append(
            {
                "id": case_id,
                "ids": resolved_ids,
                "decoded": tok.decode(resolved_ids),
            }
        )

    expansion_goldens = generate_expansion_goldens(files["processor_config.json"])

    return corpus_goldens, chat_golden, decode_goldens, expansion_goldens


# ADR-082 Stage-1 marker-expansion arithmetic (G11/G15/G17): every
# `<|image|>` marker expands to a fixed vision soft-token count; every
# `<|audio|>` marker expands to a duration-derived count capped at the
# processor's declared maximum. Mirrors `Gemma4Processor.replace_image_token`
# / `_compute_audio_num_tokens`'s documented `ceil(duration_ms /
# audio_ms_per_token)` cap in HF `transformers`
# (`processing_gemma4.py`) -- the full mel-framing + conv-subsampling detail
# behind that cap is Stage 6/9 (vision/audio end-to-end) scope, not Stage 1.
def generate_expansion_goldens(processor_config_bytes: bytes) -> dict[str, Any]:
    config = json.loads(processor_config_bytes)
    image_seq_length = config["image_seq_length"]
    audio_ms_per_token = config["audio_ms_per_token"]
    audio_seq_length = config["audio_seq_length"]

    def audio_tokens_for(duration_ms: int) -> int:
        return min(-(-duration_ms // audio_ms_per_token), audio_seq_length)

    image_cases = [
        {"marker_count": 0, "expected_tokens": 0},
        {"marker_count": 1, "expected_tokens": image_seq_length},
        {"marker_count": 3, "expected_tokens": 3 * image_seq_length},
    ]
    audio_cases = [
        {"durations_ms": [], "expected_tokens": []},
        {"durations_ms": [1000], "expected_tokens": [audio_tokens_for(1000)]},
        {
            "durations_ms": [1000, 40_000],
            "expected_tokens": [audio_tokens_for(1000), audio_tokens_for(40_000)],
        },
    ]
    return {
        "provenance": {
            "image_seq_length": image_seq_length,
            "audio_ms_per_token": audio_ms_per_token,
            "audio_seq_length": audio_seq_length,
        },
        "image_cases": image_cases,
        "audio_cases": audio_cases,
    }


def build_manifest(files: dict[str, bytes]) -> dict[str, Any]:
    import jinja2
    import tokenizers

    return {
        "source_repo": REPO,
        "revision": REVISION,
        "url_form": f"{BASE_URL}/<file>",
        "files": {
            name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
            for name, data in files.items()
        },
        "runtime_versions": {
            "tokenizers": tokenizers.__version__,
            "jinja2": jinja2.__version__,
        },
        "generator": "scripts/gemma4_tokenizer_goldens.py",
        "adr": "ADR-082 Stage 1 (G17)",
    }


def cmd_verify() -> int:
    print(f"Fetching from {BASE_URL} ...")
    files = fetch_all()
    _validate_runtime_versions()

    if not MANIFEST_PATH.exists():
        print("no committed manifest.json -- run --write-fixture first", file=sys.stderr)
        return 1
    committed_manifest = json.loads(MANIFEST_PATH.read_text())

    ok = True
    for name, data in files.items():
        expected_sha = committed_manifest["files"].get(name, {}).get("sha256")
        actual_sha = _sha256_bytes(data)
        if expected_sha != actual_sha:
            print(
                f"DRIFT: {name} sha256 mismatch: committed={expected_sha} "
                f"live={actual_sha}",
                file=sys.stderr,
            )
            ok = False

    corpus_goldens, chat_golden, decode_goldens, expansion_goldens = generate_goldens(files)
    committed_corpus = json.loads(CORPUS_GOLDENS_PATH.read_text())
    committed_chat = json.loads(CHAT_TEMPLATE_GOLDEN_PATH.read_text())
    committed_decode = json.loads(DECODE_GOLDENS_PATH.read_text())
    committed_expansion = json.loads(EXPANSION_GOLDENS_PATH.read_text())

    if corpus_goldens != committed_corpus:
        print("DRIFT: corpus_goldens.json does not match live generation", file=sys.stderr)
        ok = False
    if chat_golden != committed_chat:
        print("DRIFT: chat_template_golden.json does not match live generation", file=sys.stderr)
        ok = False
    if decode_goldens != committed_decode:
        print("DRIFT: decode_goldens.json does not match live generation", file=sys.stderr)
        ok = False
    if expansion_goldens != committed_expansion:
        print("DRIFT: expansion_goldens.json does not match live generation", file=sys.stderr)
        ok = False

    if not ok:
        return 1
    print("OK: fixtures match live fetch + regeneration, no drift.")
    return 0


def cmd_write_fixture() -> int:
    print(f"Fetching from {BASE_URL} ...")
    files = fetch_all()
    _validate_runtime_versions()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_JSON_PATH.write_bytes(files["tokenizer.json"])
    TOKENIZER_CONFIG_PATH.write_bytes(files["tokenizer_config.json"])
    CHAT_TEMPLATE_JINJA_PATH.write_bytes(files["chat_template.jinja"])
    PROCESSOR_CONFIG_PATH.write_bytes(files["processor_config.json"])

    corpus_goldens, chat_golden, decode_goldens, expansion_goldens = generate_goldens(files)
    CORPUS_GOLDENS_PATH.write_text(json.dumps(corpus_goldens, indent=2, ensure_ascii=False) + "\n")
    CHAT_TEMPLATE_GOLDEN_PATH.write_text(json.dumps(chat_golden, indent=2, ensure_ascii=False) + "\n")
    DECODE_GOLDENS_PATH.write_text(json.dumps(decode_goldens, indent=2, ensure_ascii=False) + "\n")
    EXPANSION_GOLDENS_PATH.write_text(json.dumps(expansion_goldens, indent=2, ensure_ascii=False) + "\n")

    manifest = build_manifest(files)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote fixtures to {FIXTURE_DIR}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="regenerate the committed fixtures (deliberate, reviewable, never run by CI)",
    )
    args = parser.parse_args()

    if args.write_fixture:
        return cmd_write_fixture()
    return cmd_verify()


if __name__ == "__main__":
    raise SystemExit(main())
