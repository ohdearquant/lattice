#!/usr/bin/env python3
"""Render a fixed chat-message matrix through the Gemma 4 E2B chat template.

Differential fixture for the Gemma prompt adapter in
`crates/inference/src/serve/prompt_adapter.rs`. The expected prompt strings
are produced here, by the checkpoint's own `chat_template.jinja` under the
same Jinja environment Hugging Face `transformers` uses for
`apply_chat_template` (`ImmutableSandboxedEnvironment(trim_blocks=True,
lstrip_blocks=True)` with the `loopcontrols` extension), never by the Rust
code. The Rust test requires the adapter to reproduce every string byte for
byte.

No network access and no `transformers` install: only `jinja2` and the
checkpoint's local files are read. The template must hash to the revision
pinned in `tests/fixtures/gemma4/tokenizer/manifest.json`.

Usage:
    uv run --no-project --with jinja2==3.1.6 python \
        scripts/gemma4_chat_template_matrix.py --model-dir <gemma-4-e2b-it dir>

    # Add --check to compare a fresh render with the committed fixture
    # instead of rewriting it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED_JINJA2_VERSION = "3.1.6"

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "crates" / "inference" / "tests" / "fixtures" / "gemma4"
MANIFEST_PATH = FIXTURE_DIR / "tokenizer" / "manifest.json"
OUTPUT_PATH = FIXTURE_DIR / "chat_template_matrix.json"

# Conversations rendered with add_generation_prompt=True. A case with
# `expect_refusal` must be refused by the serving contract before any
# rendering; the template's own output is still recorded for it, so the
# fixture shows what a silent acceptance would have produced.
CASES: list[dict[str, Any]] = [
    {
        "name": "single_user",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    },
    {
        "name": "system_user",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Hi"},
        ],
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Name a prime number between 10 and 20."},
            {"role": "assistant", "content": "13 is a prime number between 10 and 20."},
            {"role": "user", "content": "Name another one and explain why it is prime."},
        ],
    },
    {
        "name": "empty_system",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ],
    },
    {
        "name": "empty_user",
        "messages": [{"role": "user", "content": ""}],
    },
    {
        "name": "unicode",
        "messages": [
            {"role": "system", "content": "Réponds en français, s'il te plaît."},
            {"role": "user", "content": "Grüße, 世界! 🌍 — naïve café, ½ + ¼ ≠ 1"},
        ],
    },
    {
        "name": "assistant_last",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
    },
    {
        "name": "consecutive_assistant",
        "messages": [
            {"role": "user", "content": "Count to two."},
            {"role": "assistant", "content": "One."},
            {"role": "assistant", "content": "Two."},
            {"role": "user", "content": "Again."},
        ],
    },
    {
        "name": "assistant_after_system",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "assistant", "content": "Ready."},
            {"role": "user", "content": "Go."},
        ],
    },
    {
        "name": "mid_conversation_system",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "Colour of the sky?"},
        ],
    },
    {
        "name": "system_only",
        "messages": [{"role": "system", "content": "Only a system turn."}],
    },
    {
        "name": "whitespace_trimmed",
        "messages": [
            {"role": "system", "content": "  padded system \n"},
            {"role": "user", "content": "\t user with tabs\nand an inner newline \n\n"},
            {"role": "assistant", "content": "  spaced answer  \r\n"},
            {"role": "user", "content": " 　ideographic space "},
        ],
    },
    {
        "name": "python_only_whitespace",
        "messages": [
            {"role": "user", "content": "\x1c\x1dseparators\x1e\x1f"},
            {"role": "assistant", "content": "\x1f​zero width stays​\x1c"},
            {"role": "user", "content": "\x85next line "},
        ],
    },
    {
        "name": "assistant_thought_stripped",
        "messages": [
            {"role": "user", "content": "Is 7 prime?"},
            {
                "role": "assistant",
                "content": "<|channel>thought\nCheck divisors.<channel|>Yes, 7 is prime.",
            },
            {"role": "user", "content": "And 9?"},
        ],
    },
    {
        "name": "assistant_unclosed_thought",
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": " A <|channel>left open"},
            {"role": "user", "content": "Q2"},
        ],
    },
    {
        "name": "markup_is_not_escaped",
        "messages": [
            {"role": "user", "content": "<b>&amp; {{ not a template }} {% raw %} \"quoted\" 'single'"},
        ],
    },
    {
        "name": "tool_role",
        "messages": [
            {"role": "user", "content": "What was the result?"},
            {"role": "tool", "content": "{\"result\": 42}"},
            {"role": "user", "content": "Tell me."},
        ],
        "expect_refusal": "unsupported_feature",
    },
    {
        "name": "developer_role",
        "messages": [
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
        ],
        "expect_refusal": "unsupported_feature",
    },
    {
        "name": "unknown_role",
        "messages": [
            {"role": "function", "content": "x"},
            {"role": "user", "content": "Hi"},
        ],
        "expect_refusal": "invalid_role",
    },
    {
        "name": "text_content_parts",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": " Be brief. "}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": " first "},
                    {"type": "text", "text": " second "},
                ],
            },
        ],
        "expect_refusal": "unsupported_feature",
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hf_environment():
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )

    def strftime_now(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


def python_strip_whitespace() -> list[int]:
    """Code points `str.strip()` removes, which is what Jinja's `trim` filter calls."""
    return [cp for cp in range(sys.maxunicode + 1) if chr(cp).isspace()]


def build(model_dir: Path) -> dict[str, Any]:
    import jinja2

    if jinja2.__version__ != EXPECTED_JINJA2_VERSION:
        raise SystemExit(
            f"jinja2=={jinja2.__version__}, expected {EXPECTED_JINJA2_VERSION}; "
            "run with `uv run --no-project --with jinja2==3.1.6`"
        )

    template_bytes = (model_dir / "chat_template.jinja").read_bytes()
    tokenizer_config_bytes = (model_dir / "tokenizer_config.json").read_bytes()
    generation_config_bytes = (model_dir / "generation_config.json").read_bytes()
    config_bytes = (model_dir / "config.json").read_bytes()
    tokenizer_bytes = (model_dir / "tokenizer.json").read_bytes()

    manifest = json.loads(MANIFEST_PATH.read_text())
    pinned = manifest["files"]["chat_template.jinja"]["sha256"]
    template_sha = sha256_bytes(template_bytes)
    if template_sha != pinned:
        raise SystemExit(
            f"chat_template.jinja sha256 {template_sha} does not match the pinned {pinned}"
        )
    template = template_bytes.decode("utf-8")
    if "generation %}" in template:
        raise SystemExit(
            "template uses {% generation %}; the transformers AssistantTracker extension "
            "would be needed to render it"
        )

    tokenizer_config = json.loads(tokenizer_config_bytes)
    special = {
        key: tokenizer_config[key]
        for key in ("bos_token", "eos_token", "unk_token", "pad_token", "mask_token")
        if key in tokenizer_config
    }
    hf_template = hf_environment().from_string(template)
    plain_template = jinja2.Environment().from_string(template)

    def render(messages: list[dict[str, Any]], env_template, **extra) -> str:
        return env_template.render(
            messages=messages,
            tools=None,
            documents=None,
            add_generation_prompt=True,
            **special,
            **extra,
        )

    cases = []
    for case in CASES:
        rendered = render(case["messages"], hf_template)
        plain = render(case["messages"], plain_template)
        if plain != rendered:
            raise SystemExit(
                f"{case['name']}: default jinja2.Environment renders differently from the "
                "transformers environment; the fixture would depend on the environment"
            )
        entry: dict[str, Any] = {"name": case["name"], "messages": case["messages"]}
        if "expect_refusal" in case:
            entry["expect_refusal"] = case["expect_refusal"]
            entry["template_rendered"] = rendered
        else:
            entry["rendered"] = rendered
        cases.append(entry)

    thinking_messages = CASES[0]["messages"]
    thinking_rendered = render(thinking_messages, hf_template, enable_thinking=True)

    added = {
        token["content"]: token["id"]
        for token in json.loads(tokenizer_bytes)["added_tokens"]
    }
    control_token_ids = {
        name: added[tokenizer_config[key]]
        for key, name in (
            ("bos_token", "bos"),
            ("eos_token", "eos"),
            ("sot_token", "start_of_turn"),
            ("eot_token", "end_of_turn"),
            ("think_token", "think"),
            ("str_token", "start_of_tool_response"),
        )
    }
    generation_eos = json.loads(generation_config_bytes)["eos_token_id"]
    text_config_eos = json.loads(config_bytes)["text_config"]["eos_token_id"]
    generation_eos = generation_eos if isinstance(generation_eos, list) else [generation_eos]
    stop_token_ids = sorted({text_config_eos, *generation_eos})

    return {
        "description": (
            "Gemma 4 E2B chat_template.jinja rendered with add_generation_prompt=True over a "
            "fixed message matrix; the expected prompt strings for the Gemma prompt adapter"
        ),
        "generator": "scripts/gemma4_chat_template_matrix.py",
        "command": (
            "uv run --no-project --with jinja2==3.1.6 python "
            "scripts/gemma4_chat_template_matrix.py --model-dir <gemma-4-e2b-it>"
        ),
        "environment": (
            "jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, "
            "lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]); every case also "
            "renders identically under jinja2.Environment()"
        ),
        "jinja2": jinja2.__version__,
        "unicode_version": unicodedata.unidata_version,
        "source_repo": manifest["source_repo"],
        "revision": manifest["revision"],
        "files": {
            "chat_template.jinja": {"sha256": template_sha},
            "tokenizer_config.json": {"sha256": sha256_bytes(tokenizer_config_bytes)},
            "generation_config.json": {"sha256": sha256_bytes(generation_config_bytes)},
            "config.json": {"sha256": sha256_bytes(config_bytes)},
            "tokenizer.json": {"sha256": sha256_bytes(tokenizer_bytes)},
        },
        "render_kwargs": {
            "add_generation_prompt": True,
            "tools": None,
            "documents": None,
            **special,
        },
        "thinking_mode": {
            "template_variable": "enable_thinking",
            "messages": thinking_messages,
            "rendered_with_enable_thinking": thinking_rendered,
        },
        "generation_config_json": generation_config_bytes.decode("utf-8"),
        "text_config_eos_token_id": text_config_eos,
        "stop_token_ids": stop_token_ids,
        "control_token_ids": control_token_ids,
        "python_strip_whitespace": python_strip_whitespace(),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the committed fixture instead of writing it",
    )
    args = parser.parse_args()

    fixture = build(args.model_dir.expanduser())
    text = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if OUTPUT_PATH.read_text(encoding="utf-8") != text:
            print(f"DRIFT: {OUTPUT_PATH} differs from a fresh render", file=sys.stderr)
            return 1
        print(f"OK: {OUTPUT_PATH} matches ({len(fixture['cases'])} cases)")
        return 0
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(
        f"wrote {len(fixture['cases'])} cases to {OUTPUT_PATH.relative_to(REPO_ROOT)} "
        f"(chat_template.jinja sha256 {fixture['files']['chat_template.jinja']['sha256']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
