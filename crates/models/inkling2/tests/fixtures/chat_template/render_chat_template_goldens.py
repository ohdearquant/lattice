"""Render the published Inkling 2 chat template for a fixed set of conversations.

Usage: python render_chat_template_goldens.py <chat_template.jinja> <out dir>

Writes one file per case holding the exact rendered bytes (UTF-8, no added
newline) and a MANIFEST.json naming, per case, the inputs, the byte length and
the sha256 of the output, plus the sha256 of the template and the jinja2
version.

The environment is the one the transformers library builds for a chat
template: an ImmutableSandboxedEnvironment with trim_blocks and lstrip_blocks,
the loop-controls extension, a `raise_exception` global, a `tojson` filter that
does not escape non-ASCII or HTML characters, and a `generation` block tag.
The `generation` tag here emits its body unchanged; the library's own tag does
the same for the rendered text and additionally records where the block began
and ended. `add_generation_prompt` is not passed.
"""

import hashlib
import json
import sys
from pathlib import Path

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class GenerationBlock(Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_emit"), [], [], body).set_lineno(lineno)

    def _emit(self, caller):
        return caller()


def raise_exception(message):
    raise TemplateError(message)


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


SYSTEM = {"role": "system", "content": "You are terse."}
HELLO = {"role": "user", "content": "Hello"}
LONG = [
    SYSTEM,
    HELLO,
    {"role": "assistant", "content": "Hi."},
    {"role": "user", "content": "Line one\nLine two é"},
]

# name, messages, extra template variables (empty = reasoning_effort undefined)
CASES = [
    ("01-user", [HELLO], {}),
    ("02-user-effort0", [HELLO], {"reasoning_effort": 0}),
    ("03-system-user", [SYSTEM, HELLO], {}),
    ("04-system-only", [SYSTEM], {}),
    ("05-system-user-assistant-user", LONG, {}),
    ("06-system-user-assistant-user-effort0", LONG, {"reasoning_effort": 0}),
    (
        "07-two-systems-user",
        [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "C"},
        ],
        {},
    ),
]


def build_environment():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[GenerationBlock, jinja2.ext.loopcontrols],
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    template_path, out_dir = Path(argv[1]), Path(argv[2])
    source = template_path.read_bytes()
    template = build_environment().from_string(source.decode("utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "template_sha256": hashlib.sha256(source).hexdigest(),
        "jinja2": jinja2.__version__,
        "cases": [],
    }
    for name, messages, variables in CASES:
        rendered = template.render(messages=messages, **variables).encode("utf-8")
        (out_dir / f"{name}.txt").write_bytes(rendered)
        manifest["cases"].append(
            {
                "name": name,
                "file": f"{name}.txt",
                "messages": messages,
                "variables": variables,
                "bytes": len(rendered),
                "sha256": hashlib.sha256(rendered).hexdigest(),
            }
        )
        print(f"{name}\t{len(rendered)}\t{rendered.decode('utf-8')!r}")
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv)
