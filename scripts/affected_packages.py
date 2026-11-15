#!/usr/bin/env python3
"""Print the cargo package arguments a change between two commits needs checked.

Usage: scripts/affected_packages.py <base> <head> [--contract-tests]

Prints one line:
  "-p a -p b ..."  every workspace package that owns a changed file, plus every workspace
                   package that depends on one of those through a path dependency of any
                   kind (normal, dev or build), to closure;
  "--workspace"    when a changed file sits outside every package and is not on the
                   no-build list below (a manifest, the lockfile, the toolchain file,
                   root tests/ fixtures, ...);
  "NONE"           when every changed file is on the no-build list.

With --contract-tests it prints instead the `--test <name>` arguments for the
lattice-inference integration tests that read source outside their own crate (named
*contract*, or holding a "../" path themselves or through tests/support). Those tests
see every crate's source, so a scoped run includes them whatever the scope is.
"""

import json
import os
import subprocess
import sys

NO_BUILD_DIRS = ("docs/", ".github/", "asks/", "decisions/", "npm/", "scripts/")
NO_BUILD_FILES = {
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "REPORT.md",
    "_typos.toml",
    "deno.json",
    "pyproject.toml",
    "uv.lock",
    "LICENSE-APACHE",
    "LICENSE-MIT",
    "file-size-manifest.toml",
}


def changed_files(base: str, head: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", base, head], check=True, capture_output=True, text=True
    ).stdout
    return [line for line in out.splitlines() if line]


def workspace_graph() -> tuple[dict[str, str], dict[str, set[str]]]:
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    root = meta["workspace_root"]
    members = set(meta["workspace_members"])
    dirs = {
        p["name"]: os.path.relpath(os.path.dirname(p["manifest_path"]), root)
        for p in meta["packages"]
        if p["id"] in members
    }
    dependents: dict[str, set[str]] = {name: set() for name in dirs}
    for package in meta["packages"]:
        if package["name"] not in dirs:
            continue
        for dep in package["dependencies"]:
            if dep.get("path") and dep["name"] in dirs:
                dependents[dep["name"]].add(package["name"])
    return dirs, dependents


def scope(files: list[str], dirs: dict[str, str], dependents: dict[str, set[str]]) -> str:
    owners: set[str] = set()
    wide: list[str] = []
    for path in files:
        candidates = [n for n, d in dirs.items() if d != "." and (path + "/").startswith(d + "/")]
        if candidates:
            owners.add(max(candidates, key=lambda n: len(dirs[n])))
        elif not (path.startswith(NO_BUILD_DIRS) or path in NO_BUILD_FILES):
            wide.append(path)
    if wide:
        print("scope is the workspace because of: " + " ".join(wide[:5]), file=sys.stderr)
        return "--workspace"
    seen, todo = set(owners), list(owners)
    while todo:
        for dependent in dependents[todo.pop()]:
            if dependent not in seen:
                seen.add(dependent)
                todo.append(dependent)
    print(
        f"changed packages: {', '.join(sorted(owners)) or '(none)'}; "
        f"checked: {len(seen)} of {len(dirs)}",
        file=sys.stderr,
    )
    return " ".join(f"-p {n}" for n in sorted(seen)) if seen else "NONE"


def contract_tests() -> str:
    tests_dir = "crates/inference/tests"
    support_dir = os.path.join(tests_dir, "support")

    def reads_outside(path: str) -> bool:
        with open(path, encoding="utf-8") as handle:
            return '"../' in handle.read()

    support_reads = os.path.isdir(support_dir) and any(
        reads_outside(os.path.join(support_dir, f)) for f in os.listdir(support_dir) if f.endswith(".rs")
    )
    names = []
    for entry in sorted(os.listdir(tests_dir)):
        if not entry.endswith(".rs"):
            continue
        path = os.path.join(tests_dir, entry)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        uses_support = "mod support" in text or "#[path" in text
        if "contract" in entry or '"../' in text or (support_reads and uses_support):
            names.append(entry[:-3])
    return " ".join(f"--test {n}" for n in names)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--contract-tests" in sys.argv[1:]:
        print(contract_tests())
        return 0
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    files = changed_files(args[0], args[1])
    if not files:
        print("NONE")
        return 0
    dirs, dependents = workspace_graph()
    print(scope(files, dirs, dependents))
    return 0


if __name__ == "__main__":
    sys.exit(main())
