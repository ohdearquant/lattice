#!/usr/bin/env python3
"""Render a fixed chat-message matrix through the Qwen3.5 chat template.

Differential fixture for the Qwen3.5 chat renderer
(`format_chat_template` in `crates/models/qwen35/src/forward/metal_qwen35.rs`)
and the serving routes built on it. The expected prompt strings are produced
here, by the checkpoint's own `chat_template.jinja` under the same Jinja
environment Hugging Face `transformers` uses for `apply_chat_template`
(`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` with
the `loopcontrols` extension), never by the Rust code. The Rust tests require
the renderer to reproduce every string byte for byte.

No network access and no `transformers` install: only `jinja2` and the
checkpoint's `chat_template.jinja` and `tokenizer_config.json` are read. Both
must hash to the revision pinned below.

Usage:
    uv run --no-project --with jinja2==3.1.6 python \\
        scripts/qwen35_chat_template_matrix.py --model-dir <Qwen3.5-0.8B dir>

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

SOURCE_REPO = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PINNED_SHA256 = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = (
    REPO_ROOT / "crates" / "models" / "qwen35" / "tests" / "fixtures" / "chat_template_matrix.json"
)

IMAGE = {"type": "image"}

# Conversations rendered with add_generation_prompt=True, once with
# enable_thinking left undefined (the template's default, which closes an
# empty reasoning block) and once with enable_thinking=True.
#
# A case whose messages the template itself refuses records the template's
# error text under `template_error` instead of a rendering.
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
        "name": "trailing_consecutive_assistant",
        "messages": [
            {"role": "user", "content": "Count to two."},
            {"role": "assistant", "content": "<think>\nfirst\n</think>\n\nOne."},
            {"role": "assistant", "content": "Two."},
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
        "name": "whitespace_trimmed",
        "messages": [
            {"role": "system", "content": "  padded system \n"},
            {"role": "user", "content": "\t user with tabs\nand an inner newline \n\n"},
            {"role": "assistant", "content": "  spaced answer  \r\n"},
            {"role": "user", "content": " \u3000ideographic space "},
        ],
    },
    {
        "name": "python_only_whitespace",
        "messages": [
            {"role": "user", "content": "\x1c\x1dseparators\x1e\x1f"},
            {"role": "assistant", "content": "\x1f\u200bzero width stays\u200b\x1c"},
            {"role": "user", "content": "\x85next line "},
        ],
    },
    {
        "name": "history_reasoning_dropped",
        "messages": [
            {"role": "user", "content": "Is 7 prime?"},
            {
                "role": "assistant",
                "content": "<think>\nCheck divisors 2 to 6.\n</think>\n\nYes, 7 is prime.",
            },
            {"role": "user", "content": "And 9?"},
        ],
    },
    {
        "name": "empty_reasoning_block_in_history",
        "messages": [
            {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": "<think>\n\n</think>\n\nThe capital of France is Paris.",
            },
            {"role": "user", "content": "And its population?"},
        ],
    },
    {
        "name": "trailing_assistant_reasoning_kept",
        "messages": [
            {"role": "user", "content": "Is 7 prime?"},
            {
                "role": "assistant",
                "content": "<think>\n\n  Check divisors.  \n\n</think>\n\n\nYes. ",
            },
        ],
    },
    {
        "name": "close_tag_without_open_tag",
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "reasoning first</think>answer"},
        ],
    },
    {
        "name": "several_close_tags",
        "messages": [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "<think>a<think>b</think>middle</think>\n\nlast",
            },
        ],
    },
    {
        "name": "open_tag_without_close_tag",
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "<think>never closed"},
            {"role": "user", "content": "Q2"},
        ],
    },
    {
        "name": "reasoning_tags_in_user_content",
        "messages": [
            {"role": "user", "content": "<think>user text</think> stays as written"},
        ],
    },
    {
        "name": "tool_response_shaped_user",
        "messages": [
            {"role": "user", "content": "What is 6 times 7?"},
            {"role": "assistant", "content": "<think>\nmultiply\n</think>\n\nLet me check."},
            {"role": "user", "content": " <tool_response>\n42\n</tool_response>\n"},
        ],
    },
    {
        "name": "markup_is_not_escaped",
        "messages": [
            {
                "role": "user",
                "content": "<b>&amp; {{ not a template }} {% raw %} \"quoted\" 'single'",
            },
        ],
    },
    {
        "name": "image_trimmed",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "  Look at "},
                    IMAGE,
                    {"type": "text", "text": " and describe it. \n"},
                ],
            },
        ],
    },
    {
        "name": "image_first_with_history",
        "messages": [
            {"role": "system", "content": " Be brief. "},
            {"role": "user", "content": [IMAGE, {"type": "text", "text": " What is this? "}]},
            {"role": "assistant", "content": "<think>\nlooks\n</think>\n\nA cat."},
            {"role": "user", "content": "Colour? "},
        ],
    },
    {
        "name": "image_last_after_history",
        "messages": [
            {"role": "user", "content": "Hi "},
            {"role": "assistant", "content": "<think>\ngreet\n</think>\n\nHello."},
            {"role": "user", "content": [{"type": "text", "text": "This one: "}, IMAGE]},
        ],
    },
    {
        "name": "reasoning_content_field",
        "messages": [
            {"role": "user", "content": "Is 7 prime?"},
            {
                "role": "assistant",
                "content": "Yes.",
                "reasoning_content": "Check divisors.",
            },
        ],
        "without_reasoning_content": True,
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
        "name": "only_tool_response_shaped_user",
        "messages": [{"role": "user", "content": "<tool_response>42</tool_response>"}],
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

    file_bytes = {name: (model_dir / name).read_bytes() for name in PINNED_SHA256}
    for name, pinned in PINNED_SHA256.items():
        actual = sha256_bytes(file_bytes[name])
        if actual != pinned:
            raise SystemExit(f"{name} sha256 {actual} does not match the pinned {pinned}")
    template = file_bytes["chat_template.jinja"].decode("utf-8")
    if "generation %}" in template:
        raise SystemExit(
            "template uses {% generation %}; the transformers AssistantTracker extension "
            "would be needed to render it"
        )

    tokenizer_config = json.loads(file_bytes["tokenizer_config.json"])
    special = {
        key: tokenizer_config[key]
        for key in ("bos_token", "eos_token", "unk_token", "pad_token")
        if key in tokenizer_config
    }
    hf_template = hf_environment().from_string(template)
    plain_template = jinja2.Environment().from_string(template)

    def render(messages: list[dict[str, Any]], env_template, **extra) -> str:
        return env_template.render(
            messages=messages,
            tools=None,
            add_generation_prompt=True,
            **special,
            **extra,
        )

    def render_both(messages: list[dict[str, Any]], name: str, **extra) -> str:
        rendered = render(messages, hf_template, **extra)
        plain = render(messages, plain_template, **extra)
        if plain != rendered:
            raise SystemExit(
                f"{name}: default jinja2.Environment renders differently from the "
                "transformers environment; the fixture would depend on the environment"
            )
        return rendered

    cases = []
    for case in CASES:
        entry: dict[str, Any] = {"name": case["name"], "messages": case["messages"]}
        try:
            entry["rendered"] = render_both(case["messages"], case["name"])
            entry["rendered_enable_thinking"] = render_both(
                case["messages"], case["name"], enable_thinking=True
            )
        except jinja2.exceptions.TemplateError as error:
            if "rendered" in entry:
                raise SystemExit(f"{case['name']}: refused only with enable_thinking=True")
            entry["template_error"] = str(error)
        if case.get("without_reasoning_content"):
            stripped = [
                {key: value for key, value in message.items() if key != "reasoning_content"}
                for message in case["messages"]
            ]
            entry["rendered_without_reasoning_content"] = render_both(stripped, case["name"])
        cases.append(entry)

    return {
        "description": (
            "Qwen3.5 chat_template.jinja rendered with add_generation_prompt=True over a "
            "fixed message matrix; the expected prompt strings for the Qwen3.5 chat renderer"
        ),
        "generator": "scripts/qwen35_chat_template_matrix.py",
        "command": (
            "uv run --no-project --with jinja2==3.1.6 python "
            "scripts/qwen35_chat_template_matrix.py --model-dir <Qwen3.5-0.8B>"
        ),
        "environment": (
            "jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, "
            "lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]); every case also "
            "renders identically under jinja2.Environment()"
        ),
        "jinja2": jinja2.__version__,
        "unicode_version": unicodedata.unidata_version,
        "source_repo": SOURCE_REPO,
        "revision": REVISION,
        "files": {name: {"sha256": sha} for name, sha in PINNED_SHA256.items()},
        "render_kwargs": {
            "add_generation_prompt": True,
            "tools": None,
            **special,
        },
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(
        f"wrote {len(fixture['cases'])} cases to {OUTPUT_PATH.relative_to(REPO_ROOT)} "
        f"(chat_template.jinja sha256 {fixture['files']['chat_template.jinja']['sha256']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
