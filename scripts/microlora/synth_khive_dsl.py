#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Curate local-only NL -> request DSL examples, without dispatching any operation.

Requires verbatim per-verb help JSON and the matching `_verbs.json` registry capture.
The validator must implement the parse-only JSONL protocol described in CURATION.md.
Character estimates are a fallback budget, never a tokenizer acceptance claim.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import curation_guard
from curation_guard import GuardError, sensitive_reason

SPLITS = ("train", "valid", "test")
OUTPUT_FILES = frozenset({"CURATION.md"} | {split + suffix for split in SPLITS for suffix in (".jsonl", ".provenance.jsonl")})
SEED = "khive-dsl-curation-v1"
PARSER = "khive_request::parse_request"
DEFAULT_OUT = Path(".khive/inbound/lattice-microlora/khive-dsl")
EXCLUSIONS = {
    "brain.auto_feedback": "Result objects lack a complete nested shape and serve-attribution contract in ParamDef.",
    "moodboard.ingest": "Real image payloads and descriptor behavior are outside schema-only text synthesis.",
    "moodboard.preference": "Requires identity-bound descriptors and feature vectors not specified completely by ParamDef.",
    "moodboard.serve": "Requires a frozen feature schema and selection contract not specified completely by ParamDef.",
    "moodboard.train_preference": "Requires a matching installed descriptor identity unavailable from the schema capture.",
    "knowledge.adjudicate": "Requires an existing disputed section identity and its disambiguating content hash.",
    "knowledge.challenge": "Requires a section identity and disambiguation conditions not established by this fixture.",
    "query": "ParamDef does not supply the supported GQL/SPARQL grammar; parser acceptance alone cannot check its query language.",
    "propose": "Changeset domain discriminants and nested mutation semantics require a separate reviewed fixture.",
    # Packs registered after the reviewed template set was written. Each carries a contract the
    # schema capture does not specify (materialized trees, receipts, sandbox policy, repository and
    # platform state, stream ledgers, capability grants); rendering plausible arguments for them
    # would teach requests whose preconditions no fixture can establish.
    "exec.events": "Audit rows exist only after runs; a fixture cannot reference a materialized launch history.",
    "exec.identity": "Reports the host's resolved exec configuration; there is no argument-bearing intent to render.",
    "exec.receipt": "Requires an existing run receipt id that only a prior exec.run establishes.",
    "exec.run": "Executes a registered tool over a materialized tree under sandbox policy; side effects are outside schema-only synthesis.",
    "exec.runs": "Lists receipts for an actor identity that the capture does not establish.",
    "exec.tree": "Tree entries reference content by blob ref; the refs have no fixture without a prior blob.put.",
    "exec.tree_diff": "Requires two existing tree manifest references.",
    "exec.tree_get": "Requires an existing tree manifest reference.",
    "exec.tree_put": "Edits reference existing tree and blob refs and carry a mode vocabulary not specified by ParamDef.",
    "git.checkout": "Reads a resolved commit into a tree manifest; the commit and allowlisted repository are host state.",
    "git.diff": "input_kind discriminates commits from tree manifests with semantics ParamDef does not specify.",
    "git.gates": "Reads configured repository allowlist rows; nothing to render beyond a host path.",
    "git.ingest_cursor": "Cursor and checkpoint identities exist only after an ingest run.",
    "git.init": "Initializes an allowlisted directory; allowlist membership is host configuration, not schema.",
    "git.log": "Pathspec, decoration and paging contract is specified by the handler, not by ParamDef.",
    "git.pr_merge": "Platform merge with cross-account approval and expected_head compare; requires live platform state.",
    "git.pr_open": "Opens a pull request against a configured slug and visibility; requires live platform state.",
    "git.pr_review": "Submits a platform review bound to expected_head; requires live platform state.",
    "git.receipts": "Lists caller-owned durable receipts; caller identity is not part of the capture.",
    "git.reconcile": "Settles an existing unknown receipt using observed evidence that only a prior push produces.",
    "git.status": "Reads working-tree state of an allowlisted repository; host state, no reviewable intent.",
    "stream.append": "expected_seq and ledger density are checked against a live stream the capture does not hold.",
    "stream.batch": "Nested append and keyed-write members carry a member schema ParamDef does not specify.",
    "stream.read": "Requires an existing stream identity and sequence position.",
    "stream.stat": "Requires an existing stream identity.",
    "tool.check": "Policy decisions depend on grant and policy rows that the capture does not establish.",
    "tool.deny": "Requires an existing grant request id.",
    "tool.describe": "Requires an existing registry object; the decision field depends on the caller's grants.",
    "tool.grant": "Requires an existing grant request id; approval semantics need an identity context.",
    "tool.ingest": "Bulk registration from a live registry or an MCP tools/list payload not present in the capture.",
    "tool.list": "Lists registry objects; the kind vocabulary is specified by the handler, not by ParamDef.",
    "tool.policies": "Lists policy rows; nothing to render beyond an empty request.",
    "tool.policy": "Sets a policy row with actor and tool patterns whose grammar ParamDef does not specify.",
    "tool.register": "Registers a tool with capabilities, side-effect class and trust origin vocabularies outside ParamDef.",
    "tool.request": "Opens a grant request against a registry object the capture does not establish.",
    "tool.requests": "Lists grant requests and grants; nothing to render beyond an empty request.",
    "tool.revoke": "Requires an existing granted request id.",
    "tool.suggest": "Search over a registry plus capability graph; hits depend on registry state the capture does not hold.",
}


class CurationError(ValueError):
    """A fail-closed curation or validation error."""


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: str | bytes) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


@dataclass(frozen=True)
class Prev:
    path: str


def render_value(value):
    if isinstance(value, Prev):
        return "$prev." + value.path
    return compact(escape_literal_references(value))


def escape_literal_references(value):
    def reserved(text):
        return text == "$prev" or text.startswith(("$prev.", "$prev["))

    if isinstance(value, str):
        if value.startswith("\\") and reserved(value[1:]):
            raise CurationError(
                "A literal with exactly one backslash before $prev cannot be preserved in function-call DSL"
            )
        return "\\" + value if reserved(value) else value
    if isinstance(value, list):
        return [escape_literal_references(item) for item in value]
    if isinstance(value, dict):
        return {key: escape_literal_references(item) for key, item in value.items()}
    return value


def call(verb, args):
    return (
        verb + "(" + ",".join(k + "=" + render_value(v) for k, v in args.items()) + ")"
    )


def tagged(value):
    if isinstance(value, Prev):
        return {"kind": "prev_ref", "path": value.path}
    return {"kind": "value", "value": value}


def op_ast(verb, args):
    return {"tool": verb, "args": {key: tagged(value) for key, value in args.items()}}


@dataclass
class Example:
    prompt: str
    completion: str
    verbs: list[str]
    pack: str
    split: str
    template: str
    mode: str
    ops: list[dict]
    variant: int = 0
    origin: str = "synthetic"
    validation: dict = field(default_factory=dict)
    nested: list[dict] = field(default_factory=list)

    def row(self):
        return {"prompt": self.prompt, "completion": self.completion}

    def provenance(self):
        return {
            "pair_sha256": digest(compact(self.row())),
            "verbs": sorted(set(self.verbs)),
            "pack": self.pack,
            "split": self.split,
            "template": self.template,
            "variant": self.variant,
            "origin": self.origin,
            "validation": self.validation,
            "nested": self.nested,
        }


def read_schemas(directory):
    directory = Path(directory)
    registry_path = directory / "_verbs.json"
    try:
        registry = json.loads(registry_path.read_text())
        entries = registry["verbs"]
        registered = {entry["verb"]: entry["pack"] for entry in entries}
        if len(registered) != len(entries) or registry["total"] != len(entries):
            raise CurationError("Registry count or duplicate verb mismatch")
        schemas, hashes = {}, {registry_path.name: digest(registry_path.read_bytes())}
        for verb, pack in sorted(registered.items()):
            if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?", verb):
                raise CurationError("Invalid registry verb spelling")
            path = directory / (verb + ".json")
            data = path.read_bytes()
            schema = json.loads(data)
            if schema["verb"] != verb or schema["pack"] != pack:
                raise CurationError(f"Schema identity mismatch: {verb}")
            names = set()
            for param in schema["params"]:
                if (
                    param["name"] in names
                    or not isinstance(param["required"], bool)
                    or not isinstance(param["type"], str)
                ):
                    raise CurationError(f"Malformed parameter schema: {verb}")
                names.add(param["name"])
            schemas[verb] = schema
            hashes[path.name] = digest(data)
        return schemas, hashes
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise CurationError(f"Cannot load complete schema capture: {error}") from error


def split_verbs(schemas):
    packs = defaultdict(list)
    for verb, schema in schemas.items():
        packs[schema["pack"]].append(verb)
    result = {}
    for pack, verbs in sorted(packs.items()):
        verbs.sort(key=lambda verb: (digest(SEED + ":split:" + verb), verb))
        if pack == "schedule":
            result.update(dict.fromkeys(verbs, "test"))
            continue
        n = len(verbs)
        # Integer apportionment with an explicit small-pack coverage floor.
        held = max(1, int(n * 0.1 + 0.5)) if n >= 3 else 0
        boundaries = n - 2 * held, n - held
        for index, verb in enumerate(verbs):
            result[verb] = (
                "train"
                if index < boundaries[0]
                else "valid"
                if index < boundaries[1]
                else "test"
            )
    return result


AREAS = (
    "tokenizer",
    "cache",
    "parser",
    "scheduler",
    "search",
    "index",
    "storage",
    "router",
    "logging",
    "metrics",
    "embedding",
    "attention",
    "configuration",
    "serialization",
    "recovery",
    "pagination",
)
WORK = (
    "regression tests",
    "boundary cases",
    "error handling",
    "API documentation",
    "migration notes",
    "performance audit",
)
CHOICES = {
    "brain.feedback": (
        "useful",
        "not_useful",
        "wrong",
        "explicit_positive",
        "correction",
    ),
    "memory.feedback": (
        "useful",
        "not_useful",
        "wrong",
        "explicit_positive",
        "correction",
    ),
}


def fixture(index):
    area = AREAS[index % len(AREAS)]
    work = WORK[(index // len(AREAS)) % len(WORK)]
    topic = area + " " + work
    uid = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "https://example.invalid/fixture/" + topic)
    )
    other = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, "https://example.invalid/fixture/source/" + topic
        )
    )
    third = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "https://example.invalid/fixture/serve/" + topic)
    )
    return {
        "area": area,
        "work": work,
        "topic": topic,
        "id": uid,
        "other": other,
        "third": third,
        "slug": topic.replace(" ", "-"),
        "limit": (3, 5, 8, 10, 15, 20)[(index // 3) % 6],
        "actor": ("agent:curator", "agent:reviewer", "agent:builder")[(index // 7) % 3],
        "score": (0.3, 0.4, 0.5, 0.6, 0.7)[index % 5],
        "date": f"2027-03-{1 + index % 28:02d}T09:00:00Z",
        "note": f"Check {topic} before the next release.",
        "profile": area + "-recall-v1",
        "repo": "/srv/projects/" + area,
    }


def recipe(verb, index):
    """Reviewed intent templates; values are fixtures, not claims about live records."""
    f = fixture(index)
    t, uid, other, n = f["topic"], f["id"], f["other"], f["limit"]
    area, slug, score = f["area"], f["slug"], f["score"]
    # Every rendered argument has an explicit natural-language counterpart.
    if verb == "blob.put":
        text = f["note"]
        return (
            {"bytes": base64.b64encode(text.encode()).decode()},
            f"Store these UTF-8 bytes in the blob store (base64-encode them): {compact(text)}.",
        )
    if verb in ("blob.get", "blob.stat"):
        ref = digest("fixture-content:" + t)
        args = {"content_ref": ref}
        intent = (
            "Read the bytes" if verb == "blob.get" else "Check existence and byte size"
        ) + f" for existing blob {ref}."
        if verb == "blob.get" and index % 2:
            args["range"] = {"offset": index % 12, "length": n}
            intent += f" Return {n} bytes from byte offset {index % 12}."
        return args, intent
    if verb in (
        "brain.activate",
        "brain.archive",
        "brain.deactivate",
        "brain.profile",
        "brain.reset",
    ):
        intent = {
            "brain.activate": "Activate the existing profile",
            "brain.archive": "Archive the existing profile",
            "brain.deactivate": "Deactivate the existing profile",
            "brain.profile": "Show metadata and current state for profile",
            "brain.reset": "Reset posteriors to priors for active profile",
        }[verb]
        return {"profile_id": f["profile"]}, f"{intent} {compact(f['profile'])}."
    if verb == "brain.create_profile":
        return (
            {"name": slug + "-v1", "description": f["note"]},
            f"Create a recall profile named {compact(slug + '-v1')} with description {compact(f['note'])}.",
        )
    if verb in CHOICES:
        signal = CHOICES[verb][index % len(CHOICES[verb])]
        return (
            {"target_id": uid, "signal": signal},
            f"Record {compact(signal)} feedback on the recalled {'memory' if verb.startswith('memory') else 'brain result'} {uid} about {t}.",
        )
    if verb in ("brain.bind", "brain.bindings", "brain.unbind"):
        action = {
            "brain.bind": "Bind",
            "brain.bindings": "List bindings for",
            "brain.unbind": "Remove bindings for",
        }[verb]
        args = {"profile_id": f["profile"], "actor": f["actor"]}
        intent = (
            f"{action} profile {compact(f['profile'])} and actor {compact(f['actor'])}."
        )
        if verb == "brain.bind":
            args["priority"] = n
            intent += f" Give the binding priority {n}."
        return args, intent
    if verb == "brain.event_counts":
        return (
            {"since": f["date"], "actor": f["actor"], "exhaustive": True},
            f"Count all event kinds for actor {compact(f['actor'])} since {f['date']}, visiting every matching page.",
        )
    if verb == "brain.mark_turn":
        return {
            "label": t
        }, f"Mark the beginning of this unit of work with the label {compact(t)}."
    if verb == "brain.profiles":
        life = ("active", "inactive", "archived")[index % 3]
        return {"lifecycle": life}, f"List brain profiles whose lifecycle is {life}."
    if verb == "brain.resolve":
        return {
            "consumer_kind": "recall",
            "actor": f["actor"],
        }, f"Show which recall profile would serve actor {compact(f['actor'])}."
    if verb == "brain.register_adapter":
        content_hash = digest("fixture-weights:" + t)
        revision = "fixture-base-" + area
        return (
            {
                "adapter_id": slug,
                "content_hash": content_hash,
                "base_model_revision": revision,
            },
            f"Register adapter {compact(slug)} with supplied weights hash {content_hash} against active base revision {compact(revision)}.",
        )
    if verb == "code.ingest":
        lang = ("rust", "python", "typescript")[(index // 96) % 3]
        path = "/srv/projects/" + area + "/" + f["work"].replace(" ", "-")
        tiers = ["l1", "l1.5"] if (index // 288) % 2 else ["l1"]
        return (
            {"path": path, "languages": [lang], "tiers": tiers},
            f"Build a code map from folder {compact(path)} for {lang} only, ingesting tiers {compact(tiers)} into the default dedicated map database.",
        )
    if verb == "comm.delivered":
        return (
            {"id": uid},
            f"Check whether outbound message {uid} has its internal inbound sibling; this is the {t} update.",
        )
    if verb == "comm.inbox":
        status = ("unread", "read", "all")[index % 3]
        return (
            {"status": status, "subject_contains": area, "limit": n},
            f"Show up to {n} inbound messages with read-status {status} and {compact(area)} in the subject.",
        )
    if verb in ("comm.mark_read", "comm.read"):
        ids = [uid, other] if index % 2 else [uid]
        args = {"ids": ids}
        intent = f"Mark these inbound messages read: {compact(ids)}."
        if verb == "comm.mark_read":
            args["atomic"] = True
            intent += " Apply all updates together or none."
        else:
            intent += " Use the compatibility mark-read operation."
        return args, intent
    if verb == "comm.probe":
        return (
            {"actor": f["actor"], "stale_minutes": n},
            f"Probe inbound message metadata for actor {compact(f['actor'])}; count unread messages older than {n} minutes.",
        )
    if verb == "comm.reply":
        return {
            "id": uid,
            "content": f["note"],
        }, f"Reply to message {uid} with this body: {compact(f['note'])}."
    if verb == "comm.send":
        return (
            {"to": f["actor"], "subject": t, "content": f["note"]},
            f"Send actor {compact(f['actor'])} a message with subject {compact(t)} and body {compact(f['note'])}.",
        )
    if verb == "comm.thread":
        order = ("asc", "desc")[index % 2]
        return (
            {"id": uid, "limit": n, "order": order},
            f"Read thread rooted at {uid}, returning at most {n} messages {'oldest' if order == 'asc' else 'newest'} first.",
        )
    if verb == "git.branch":
        return (
            {"repo": f["repo"], "name": "topic/" + slug, "from": "main"},
            f"In repository {compact(f['repo'])}, create branch {compact('topic/' + slug)} from main.",
        )
    if verb == "git.commit":
        message = f"test({area}): cover {f['work']}"
        paths = ["tests/" + area + ".rs"]
        return (
            {"repo": f["repo"], "message": message, "paths": paths},
            f"Stage and commit only {compact(paths)} in repository {compact(f['repo'])} with message {compact(message)}.",
        )
    if verb == "git.digest":
        return (
            {"source": f["repo"], "max_items": n, "include": ["commits"]},
            f"Ingest at most {n} commits of provenance from local repository {compact(f['repo'])}; do not include issues or pull requests.",
        )
    if verb == "git.push":
        local = digest("fixture-local-sha:" + t)[:40]
        return (
            {
                "repo": f["repo"],
                "branch": "topic/" + slug,
                "expected_local": local,
                "expected_remote": None,
            },
            f"Push branch {compact('topic/' + slug)} from repository {compact(f['repo'])} at exactly local commit {local}, requiring that the branch does not yet exist on the remote.",
        )
    if verb == "gtd.assign":
        priority = ("p0", "p1", "p2", "p3")[index % 4]
        return (
            {"title": f["note"], "priority": priority, "assignee": f["actor"]},
            f"Create a {priority} task for {compact(f['actor'])} titled {compact(f['note'])}.",
        )
    if verb == "gtd.complete":
        return (
            {"id": uid, "result": f["note"]},
            f"Complete existing active task {uid} and record result {compact(f['note'])}.",
        )
    if verb == "gtd.transition":
        status = ("next", "active", "waiting", "someday")[index % 4]
        return (
            {"id": uid, "status": status, "note": f["note"]},
            f"Move task {uid} to {status}, with note {compact(f['note'])}; assume its current lifecycle permits that transition.",
        )
    if verb == "gtd.next":
        return {
            "limit": n,
            "assignee": f["actor"],
        }, f"List up to {n} actionable tasks assigned to {compact(f['actor'])}."
    if verb == "gtd.tasks":
        status = ("inbox", "next", "active", "waiting", "done")[index % 5]
        return {
            "status": status,
            "assignee": f["actor"],
            "limit": n,
        }, f"List up to {n} {status} tasks assigned to {compact(f['actor'])}."
    if verb == "context":
        return (
            {"query": t, "hops": 1, "limit": n},
            f"Build one-hop graph context about {compact(t)} with at most {n} query-selected anchors.",
        )
    if verb == "create":
        return (
            {
                "kind": "entity",
                "entity_kind": "concept",
                "name": t,
                "description": f["note"],
            },
            f"Create a concept entity named {compact(t)} with description {compact(f['note'])}.",
        )
    if verb == "delete":
        return {
            "id": uid,
            "hard": False,
        }, f"Soft-delete the obsolete record {uid} about {t}."
    if verb == "get":
        return {"id": uid}, f"Fetch the existing graph record {uid} about {t}."
    if verb == "link":
        return (
            {
                "source_id": uid,
                "target_id": other,
                "relation": "supports",
                "weight": score,
                "metadata": {"basis": "review"},
            },
            f"Link evidence document {uid} as supporting concept {other}, with weight {score} and metadata whose basis is review.",
        )
    if verb == "list":
        return (
            {
                "kind": "entity",
                "entity_kind": "concept",
                "limit": n,
                "offset": index % 30,
            },
            f"List concept entities, returning at most {n} after skipping the first {index % 30}.",
        )
    if verb == "merge":
        return (
            {"into_id": uid, "from_id": other, "dry_run": True},
            f"Preview merging duplicate {t} entity {other} into {uid}, keeping the latter; do not mutate either record.",
        )
    if verb == "neighbors":
        direction = ("incoming", "outgoing", "both")[index % 3]
        return (
            {"node_id": uid, "direction": direction, "min_weight": score},
            f"Show {direction} graph neighbors of {uid} with edge weights at least {score}.",
        )
    if verb == "resolve":
        return (
            {"refs": [t], "kind": "concept", "limit": n},
            f"Resolve {compact(t)} to a concept entity; return at most {n} candidates if ambiguous.",
        )
    if verb == "review":
        decision = ("approve", "reject", "comment", "request_changes")[index % 4]
        return (
            {"id": uid, "decision": decision, "comment": f["note"]},
            f"Review open proposal {uid} with decision {decision} and comment {compact(f['note'])}.",
        )
    if verb == "search":
        return {
            "kind": "entity",
            "query": t,
            "limit": n,
        }, f"Search graph entities for {compact(t)}, returning at most {n} matches."
    if verb == "traverse":
        depth = 1 + index % 3
        return (
            {"roots": [uid], "max_depth": depth, "limit": n},
            f"Traverse graph paths from {uid} to depth {depth}, allowing at most {n} non-root nodes.",
        )
    if verb == "update":
        return {
            "id": uid,
            "kind": "entity",
            "description": f["note"],
        }, f"Set entity {uid}'s description to {compact(f['note'])}."
    if verb == "verbs":
        pack = ("kg", "gtd", "memory", "brain", "comm", "schedule")[index % 6]
        return {
            "pack": pack
        }, f"List registered MCP-callable verbs belonging to the {pack} pack."
    if verb == "withdraw":
        return {
            "id": uid,
            "rationale": f["note"],
        }, f"Withdraw my open proposal {uid}, giving this reason: {compact(f['note'])}."
    if verb == "knowledge.cite":
        return (
            {"concept_id": uid, "source_id": other, "weight": score},
            f"Cite document {other} as the source introducing concept {uid}, with edge weight {score}.",
        )
    if verb == "knowledge.compose":
        budget = 500 + n * 100
        return (
            {"query": t, "max_tokens": budget},
            f"Compose a knowledge briefing about {compact(t)}, automatically selecting domains and budgeting {budget} output tokens.",
        )
    if verb == "knowledge.delete_atoms":
        return {
            "ids": [slug]
        }, f"Soft-delete the knowledge atom with exact slug {compact(slug)}."
    if verb == "knowledge.edit":
        content = f"For {t}, document the assumptions, check the boundary cases, and retain reproducible evidence for each conclusion."
        return (
            {
                "id": slug,
                "sections": [
                    {"section_type": "operational_guidance", "content": content}
                ],
            },
            f"Upsert the operational_guidance section of atom {compact(slug)} with this exact content: {compact(content)}.",
        )
    if verb == "knowledge.feedback":
        section = ("overview", "formalism", "examples")[index % 3]
        signal = ("useful", "not_useful", "wrong")[(index // 3) % 3]
        return {
            "section_signals": {section: signal},
            "target_id": uid,
        }, f"Rate the {section} section of knowledge atom {uid} as {signal}."
    if verb == "knowledge.fold":
        budget = n * 100
        return (
            {
                "candidates": [{"id": uid, "score": score, "size": 100}],
                "budget": budget,
            },
            f"Select knowledge candidates within a {budget}-token budget from this scored item: id {uid}, score {score}, size 100 tokens.",
        )
    if verb == "knowledge.get":
        return {
            "id": slug,
            "include_sections": True,
        }, f"Fetch knowledge atom {compact(slug)} including its sections."
    if verb == "knowledge.import":
        path = "/srv/notes/" + slug + ".md"
        strategy = ("section", "atom")[index % 2]
        return {
            "path": path,
            "format": "atlas_md",
            "chunk_strategy": strategy,
        }, f"Import atlas markdown file {compact(path)} using {strategy} chunking."
    if verb == "knowledge.index":
        return (
            {"ids": [slug], "rebuild_ann": True},
            f"Backfill embeddings for knowledge atom {compact(slug)} and rebuild the in-memory ANN index.",
        )
    if verb == "knowledge.learn":
        return (
            {"name": t, "description": f["note"]},
            f"Register a knowledge concept named {compact(t)} with description {compact(f['note'])}.",
        )
    if verb == "knowledge.list":
        return {
            "type": "atom",
            "limit": n,
            "offset": index % 30,
        }, f"List at most {n} knowledge atoms, skipping the first {index % 30}."
    if verb == "knowledge.search":
        return (
            {"query": t, "type": "atom", "limit": n},
            f"Search the teaching-atom corpus for {compact(t)}, returning at most {n} atoms.",
        )
    if verb == "knowledge.suggest":
        return {
            "query": t,
            "limit": n,
        }, f"Suggest at most {n} knowledge domains relevant to {compact(t)}."
    if verb == "knowledge.topic":
        return (
            {"query": t, "limit": n},
            f"Find up to {n} registered concepts whose name or description matches {compact(t)}.",
        )
    if verb == "knowledge.upsert_atoms":
        return (
            {"atoms": [{"slug": slug, "name": t, "content": f["note"]}]},
            f"Upsert a knowledge atom with slug {compact(slug)}, name {compact(t)}, and content {compact(f['note'])}. Preserve omitted fields on update.",
        )
    if verb == "knowledge.upsert_domains":
        return {
            "domains": [{"slug": slug, "name": t}]
        }, f"Upsert a knowledge domain with slug {compact(slug)} and name {compact(t)}."
    if verb == "memory.prune":
        return (
            {"min_salience": score, "before": 0, "dry_run": True},
            f"Preview pruning memories whose salience is below {score}; skip the expiry filter and do not delete anything.",
        )
    if verb == "memory.recall":
        return {"query": t, "limit": n}, f"Find up to {n} memories about {compact(t)}."
    if verb == "memory.remember":
        return {
            "content": f["note"],
            "salience": score,
            "memory_type": "semantic",
        }, f"Remember this semantic fact with salience {score}: {compact(f['note'])}."
    if verb == "moodboard.judge":
        choice = ("left", "right", "tie")[index % 3]
        return (
            {
                "serve_id": f["third"],
                "left_result_occurrence_id": uid,
                "right_result_occurrence_id": other,
                "choice": choice,
            },
            f"For existing visual serve {f['third']}, displayed left occurrence {uid} and right occurrence {other}, record my choice {choice}.",
        )
    if verb == "moodboard.search":
        return (
            {"asset_id": uid, "top_k": n},
            f"Find up to {n} visual neighbors of existing visual asset {uid}, excluding the asset itself.",
        )
    if verb == "schedule.agenda":
        return {
            "from": f["date"],
            "limit": n,
        }, f"Show at most {n} pending scheduled events starting at {f['date']}."
    if verb == "schedule.cancel":
        return {"id": uid}, f"Cancel existing scheduled event {uid} for {t}."
    if verb == "schedule.remind":
        return {
            "content": f["note"],
            "at": f["date"],
        }, f"Remind me at {f['date']} with message {compact(f['note'])}."
    if verb == "schedule.schedule":
        nested = call("schedule.remind", {"content": f["note"], "at": f["date"]})
        return (
            {"action": nested, "at": "2027-02-01T09:00:00Z"},
            f"At 2027-02-01T09:00:00Z, dispatch a call that schedules a reminder for {f['date']} with content {compact(f['note'])}.",
        )
    if verb == "session.export":
        fmt = ("json", "markdown")[index % 2]
        return {"id": uid, "format": fmt}, f"Export stored session {uid} as {fmt}."
    if verb == "session.resume":
        return {"id": uid}, f"Fetch the complete stored session {uid} concerning {t}."
    if verb == "session.list":
        return {
            "limit": n,
            "offset": index % 30,
            "provider": "codex",
        }, f"List at most {n} Codex sessions, skipping the first {index % 30}."
    if verb == "session.store":
        return (
            {"content": f["note"], "title": t, "provider": "codex"},
            f"Store a Codex session summary titled {compact(t)} with content {compact(f['note'])}.",
        )
    no_args = {
        "comm.health": "Show communication-channel health and polling freshness.",
        "comm.unread": "Count my unread inbound messages without retrieving message bodies.",
        "db_diagnostics": "Report database contention, graph integrity, and WAL diagnostics.",
        "stats": "Show aggregate counts of live graph entities, edges, and notes.",
        "whoami": "Show the runtime-resolved caller identity and visible namespaces.",
        "knowledge.stats": "Show teaching-corpus atom, domain, and coverage statistics.",
        "memory.vacuum": "Reclaim SQLite space freed by soft-deleted memory rows.",
        "moodboard.model": "Show the configured visual checkpoint identity and descriptor space without loading weights.",
    }
    if verb in no_args:
        return {}, no_args[verb]
    raise CurationError(
        f"No reviewed template for {verb}; explicitly exclude new verbs before generating"
    )


TRAPS = {
    "query": "q",
    "at": "due",
    "id": "thread_id",
    "ids": "slugs",
    "metadata": "properties",
}
CORRECTIONS = {wrong: right for right, wrong in TRAPS.items()}
PARAPHRASES = (
    "{intent}\nReturn only the request ops string.",
    "Write the request ops for this task: {intent}",
    "Translate this request to ops: {intent}",
)


def make_single(verb, index, schemas, splits):
    args, intent = recipe(verb, index)
    completion = call(verb, args)
    return Example(
        PARAPHRASES[index % 3].format(intent=intent),
        completion,
        [verb],
        schemas[verb]["pack"],
        splits[verb],
        "intent:" + verb,
        "single",
        [op_ast(verb, args)],
        index,
    )


def bounded(example):
    return (
        0 < len(example.prompt) <= 1200
        and 0 < len(example.completion) <= 300
        and len(example.prompt) + len(example.completion) <= 1500
    )


def add_split_guard(example, splits):
    if not example.verbs or any(verb not in splits for verb in example.verbs):
        raise CurationError("Unknown verb in split guard")
    partitions = {splits[verb] for verb in example.verbs}
    if partitions != {example.split}:
        raise CurationError("Composition crosses held-out verb partitions")


def untemplated_verbs(schemas):
    """Registered verbs with neither a reviewed template nor an exclusion, computed up front."""
    missing = []
    for verb in sorted(schemas):
        if verb in EXCLUSIONS:
            continue
        try:
            recipe(verb, 0)
        except CurationError:
            missing.append(verb)
    return missing


def candidates(schemas, splits):
    pools = defaultdict(list)
    drops = Counter()
    # Fail closed on the whole set at once: a refusal naming only the first verb makes every
    # registry change a sequence of one-verb discoveries.
    missing = untemplated_verbs(schemas)
    if missing:
        raise CurationError(
            f"No reviewed template for {len(missing)} registered verb(s): "
            + ", ".join(missing)
            + "; add a reviewed template or an EXCLUSIONS entry with a reason for each before generating"
        )
    active = [verb for verb in sorted(schemas) if verb not in EXCLUSIONS]
    # Round robin by verb prevents rich signatures from crowding out small ones.
    for index in range(384):
        for verb in active:
            item = make_single(verb, index, schemas, splits)
            if verb == "schedule.schedule":
                item.verbs.append("schedule.remind")
            if not bounded(item):
                drops["candidate_exceeds_character_budget"] += 1
                continue
            pools[item.pack].append(item)
            args, _ = recipe(verb, index)
            trap_keys = [
                key
                for key in args
                if key in TRAPS
                and TRAPS[key] not in args
                and TRAPS[key] not in {p["name"] for p in schemas[verb]["params"]}
            ]
            if trap_keys and index % 3 == 0:
                key = trap_keys[(index // 3) % len(trap_keys)]
                wrong = {
                    TRAPS[key] if k == key else k: value for k, value in args.items()
                }
                correction = Example(
                    f"Fix the parameter-name error in this attempted request, preserving its intent and all values:\n{call(verb, wrong)}\nReturn only corrected ops.",
                    item.completion,
                    item.verbs[:],
                    item.pack,
                    item.split,
                    "correct_parameter:" + TRAPS[key] + "->" + key,
                    "single",
                    item.ops,
                    index,
                )
                pools[item.pack].append(correction)
            # Independent, same-verb operations are always in one partition. Avoid
            # statusful operations that could race on one fixture record.
            if index % 3 == 1 and verb in {
                "memory.recall",
                "knowledge.search",
                "knowledge.suggest",
                "get",
                "comm.thread",
                "blob.stat",
                "session.resume",
                "gtd.tasks",
                "brain.profile",
                "moodboard.search",
                "schedule.agenda",
                "git.branch",
                "code.ingest",
            }:
                partner = {
                    "get": "neighbors",
                    "knowledge.search": "knowledge.suggest",
                }.get(verb, verb)
                if partner not in schemas or splits[partner] != splits[verb]:
                    partner = verb
                second = make_single(partner, index + 1, schemas, splits)
                _, intent1 = recipe(verb, index)
                _, intent2 = recipe(partner, index + 1)
                pair = Example(
                    f"Run these two independent requests as a parallel batch: {intent1} {intent2}",
                    "[" + item.completion + "," + second.completion + "]",
                    [verb, partner],
                    item.pack,
                    item.split,
                    "parallel:" + verb,
                    "parallel",
                    item.ops + second.ops,
                    index,
                )
                if bounded(pair):
                    pools[item.pack].append(pair)
            if index % 4 == 2 and verb in {"comm.probe", "knowledge.list"}:
                if verb == "comm.probe":
                    first = {
                        "actor": fixture(index)["actor"],
                        "stale_minutes": fixture(index)["limit"],
                    }
                    second = {**first, "since_us": Prev("cursor_us")}
                    intent = f"Probe actor {compact(first['actor'])} with stale threshold {first['stale_minutes']} minutes, then probe again with that threshold using the first response's cursor_us as since_us."
                else:
                    first = {
                        "type": "atom",
                        "after": "",
                        "limit": fixture(index)["limit"],
                    }
                    second = {**first, "after": Prev("next_after")}
                    intent = f"Fetch the first cursor page of knowledge atoms with limit {first['limit']}, then the next page using its next_after. Assume the first page has a non-null next_after."
                chained = Example(
                    intent + " Return chained ops.",
                    call(verb, first) + " | " + call(verb, second),
                    [verb],
                    item.pack,
                    item.split,
                    "chain_cursor:" + verb,
                    "chain",
                    [op_ast(verb, first), op_ast(verb, second)],
                    index,
                )
                if bounded(chained):
                    pools[item.pack].append(chained)
            if (
                verb == "knowledge.suggest"
                and "knowledge.fold" in schemas
                and splits[verb] == splits["knowledge.fold"]
            ):
                f = fixture(index)
                first = {"query": f["topic"], "limit": f["limit"]}
                second = {
                    "candidates": Prev("results"),
                    "budget": 500 + 100 * f["limit"],
                }
                chained = Example(
                    f"Suggest up to {first['limit']} knowledge domains about {compact(first['query'])}, then select from those results within a {second['budget']}-token budget. Return chained ops.",
                    call(verb, first) + " | " + call("knowledge.fold", second),
                    [verb, "knowledge.fold"],
                    item.pack,
                    item.split,
                    "chain_selection:knowledge.suggest",
                    "chain",
                    [op_ast(verb, first), op_ast("knowledge.fold", second)],
                    index,
                )
                if bounded(chained):
                    pools[item.pack].append(chained)
    return pools, drops


def select_candidates(pools, splits, per_pack=350):
    selected, pairs, completions, drops = [], set(), Counter(), Counter()
    for pack, pool in sorted(pools.items()):
        kept = 0
        for example in pool:
            add_split_guard(example, splits)
            key = (example.prompt, example.completion)
            if not bounded(example):
                drops["character_budget"] += 1
            elif key in pairs:
                drops["exact_duplicate"] += 1
            elif completions[example.completion] >= 3:
                drops["completion_frequency_cap"] += 1
            else:
                pairs.add(key)
                completions[example.completion] += 1
                selected.append(example)
                kept += 1
                if kept == per_pack:
                    break
        if kept < per_pack:
            drops[f"shortfall:{pack}"] = per_pack - kept
    return selected, drops


def value_matches(value, typename):
    typename = typename.strip()
    # Live captures spell alternatives three ways: "a | b", "a|b" and "a or b".
    if "|" in typename:
        return any(value_matches(value, arm) for arm in typename.split("|"))
    if " or " in typename:
        return any(value_matches(value, arm) for arm in typename.split(" or "))
    if typename == "null":
        return value is None
    if typename == "JSON value":
        return value is None or isinstance(value, (bool, int, float, str, list, dict))
    if typename.startswith("array<") and typename.endswith(">"):
        return isinstance(value, list) and all(
            value_matches(v, typename[6:-1]) for v in value
        )
    if typename.startswith("array of "):
        return isinstance(value, list) and all(
            value_matches(v, typename[9:]) for v in value
        )
    if typename == "array":
        return isinstance(value, list)
    if typename in ("bool", "boolean"):
        return isinstance(value, bool)
    if typename == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typename in ("number", "float"):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if typename == "string":
        return isinstance(value, str)
    if typename == "uuid":
        if not isinstance(value, str):
            return False
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            return bool(re.fullmatch(r"[0-9a-fA-F]{8,31}", value))
    if typename == "object":
        return isinstance(value, dict)
    raise CurationError("Unimplemented schema type: " + typename)


def validate_ast(parsed, schemas):
    mode, ops = parsed.get("mode"), parsed.get("ops")
    if (
        mode not in ("single", "parallel", "chain")
        or not isinstance(ops, list)
        or not ops
    ):
        raise CurationError("Invalid parser AST shape")
    if mode == "single" and len(ops) != 1:
        raise CurationError("Single mode contains multiple operations")
    verbs = []
    for index, op in enumerate(ops):
        if not isinstance(op, dict):
            raise CurationError("Invalid operation in parser AST")
        verb, args = op.get("tool"), op.get("args")
        if verb not in schemas:
            raise CurationError(f"Unknown or internal verb: {verb}")
        if not isinstance(args, dict):
            raise CurationError("Parser args are not an object")
        params = {param["name"]: param for param in schemas[verb]["params"]}
        if set(args) - set(params):
            raise CurationError(
                f"Unknown parameter for {verb}: {sorted(set(args) - set(params))}"
            )
        missing = [
            p["name"]
            for p in params.values()
            if p["required"] and p["name"] not in args
        ]
        if missing:
            raise CurationError(f"Required parameters missing for {verb}: {missing}")
        for key, argument in args.items():
            if not isinstance(argument, dict):
                raise CurationError("Missing parser argument tagging")
            if argument.get("kind") == "prev_ref":
                prior = ops[index - 1]["tool"] if index else None
                allowed = {
                    ("comm.probe", "comm.probe", "since_us", "cursor_us"),
                    ("knowledge.list", "knowledge.list", "after", "next_after"),
                    ("knowledge.suggest", "knowledge.fold", "candidates", "results"),
                }
                if (
                    mode != "chain"
                    or (prior, verb, key, argument.get("path")) not in allowed
                ):
                    raise CurationError("Unverified previous-result reference contract")
            elif argument.get("kind") != "value" or "value" not in argument:
                raise CurationError(
                    "Dynamic nested reference contracts are not in the reviewed fixture"
                )
            elif not value_matches(argument["value"], params[key]["type"]):
                raise CurationError(f"Wrong parameter type for {verb}.{key}")
        verbs.append(verb)
    return verbs


class Validator:
    def __init__(self, binary):
        self.binary = Path(binary).resolve()
        if not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise CurationError(
                "Required real-parser validator is missing or not executable"
            )
        self.binary_hash = digest(self.binary.read_bytes())
        self.source_hash = None

    def parse(self, completions, allow_invalid=False):
        if not completions:
            return []
        data = "".join(
            compact({"completion": completion}) + "\n" for completion in completions
        )
        try:
            process = subprocess.run(
                [str(self.binary)],
                input=data,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CurationError(f"Validator failed: {error}") from error
        if process.returncode not in (0, 1):
            raise CurationError(
                f"Validator exit {process.returncode}: {process.stderr[:500]}"
            )
        lines = process.stdout.splitlines()
        if len(lines) != len(completions):
            raise CurationError("Validator produced incomplete or extra output")
        results = []
        for index, (line, completion) in enumerate(
            zip(lines, completions, strict=True), 1
        ):
            try:
                result = json.loads(line)
            except json.JSONDecodeError as error:
                raise CurationError("Validator produced malformed JSON") from error
            if (
                not isinstance(result, dict)
                or result.get("line") != index
                or result.get("parser") != PARSER
                or result.get("executed") is not False
            ):
                raise CurationError(
                    "Validator marker, order, or no-execution assertion missing"
                )
            source = result.get("parser_source_sha256")
            if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{64}", source):
                raise CurationError("Validator source hash missing")
            if self.source_hash is None:
                self.source_hash = source
            if self.source_hash != source:
                raise CurationError("Validator source hash changed mid-run")
            if result.get("ok") is True:
                if (
                    result.get("completion") != completion
                    or result.get("completion_parsed_unchanged") is not True
                    or result.get("ast_json_roundtrip_equal") is not True
                    or result.get("parser_roundtrip_equal") is not True
                ):
                    raise CurationError(
                        "Validator changed completion or failed parser roundtrip"
                    )
            elif result.get("ok") is not False or not isinstance(
                result.get("error"), str
            ):
                raise CurationError("Invalid validator failure record")
            elif not allow_invalid:
                raise CurationError(f"Parser rejected row {index}: {result['error']}")
            results.append(result)
        expected_code = 1 if any(r["ok"] is False for r in results) else 0
        if process.returncode != expected_code:
            raise CurationError("Validator exit status disagrees with row statuses")
        return results

    def probe(self):
        text = 'quoted "line"\n雪 and café'
        results = self.parse(
            [call("memory.recall", {"query": text}), "memory.recall(query=)"], True
        )
        if (
            results[0].get("ops") != [op_ast("memory.recall", {"query": text})]
            or results[1]["ok"] is not False
        ):
            raise CurationError("Validator positive/negative parser probes failed")


def validate_examples(examples, validator, schemas, splits):
    results = validator.parse([example.completion for example in examples])
    for example, result in zip(examples, results, strict=True):
        verbs = validate_ast(result, schemas)
        if result["mode"] != example.mode or result["ops"] != example.ops:
            raise CurationError("Parser AST differs from intended operations")
        nested = []
        for op in result["ops"]:
            if op["tool"] == "schedule.schedule":
                action = op["args"]["action"]["value"]
                child = validator.parse([action])[0]
                # A single reminder is the only reviewed embedded dispatch template.
                if (
                    child.get("mode") != "single"
                    or len(child.get("ops", [])) != 1
                    or child["ops"][0]["tool"] != "schedule.remind"
                ):
                    raise CurationError("Unreviewed nested scheduled action")
                verbs += validate_ast(child, schemas)
                nested.append(
                    {
                        "completion": action,
                        "verbs": ["schedule.remind"],
                        "parsed_unchanged": True,
                        "registry_params_types": True,
                    }
                )
        if set(verbs) != set(example.verbs):
            raise CurationError(
                "Provenance verb inventory differs from parsed operations"
            )
        add_split_guard(example, splits)
        example.nested = nested
        example.validation = {
            "parser": PARSER,
            "parser_source_sha256": validator.source_hash,
            "validator_sha256": validator.binary_hash,
            "completion_parsed_unchanged": True,
            "parser_roundtrip_equal": True,
            "registry_params_types": True,
            "executed": False,
            "mode": result["mode"],
            "semantic_dispatch": "unverified",
            "tokenizer": "unverified_character_budget_only",
        }


def corrected_ast(result, schemas):
    """Only rename an unrecognized parameter when a single reviewed alias fits."""
    if result.get("mode") != "single" or len(result.get("ops", [])) != 1:
        return None
    op = result["ops"][0]
    verb, args = op.get("tool"), op.get("args")
    if verb not in schemas or not isinstance(args, dict):
        return None
    params = {p["name"] for p in schemas[verb]["params"]}
    bad = set(args) - params
    if len(bad) != 1:
        return None
    wrong = next(iter(bad))
    right = CORRECTIONS.get(wrong)
    if right not in params or right in args:
        return None
    if any(value.get("kind") != "value" for value in args.values()):
        return None
    fixed = {
        right if key == wrong else key: value["value"] for key, value in args.items()
    }
    return call(verb, fixed)


def merge_real(path, examples, validator, schemas, splits):
    skips = Counter()
    if path is None:
        return examples, skips
    data = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise CurationError(f"Malformed real JSONL at line {number}") from error
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("ops"), str)
                or not row["ops"]
                or not isinstance(row.get("ok"), bool)
            ):
                raise CurationError(f"Malformed real row at line {number}")
            if "corrected_ops" in row and (
                not isinstance(row["corrected_ops"], str) or not row["corrected_ops"]
            ):
                raise CurationError(f"Malformed corrected_ops at line {number}")
            data.append(row)
    originals = validator.parse([row["ops"] for row in data], allow_invalid=True)
    additions = []
    for number, (row, parsed) in enumerate(zip(data, originals, strict=True), 1):
        # Recorded text reaches both the prompt and the completion verbatim, so a row carrying
        # an address or a credential in any field is skipped whole; nothing is redacted.
        screened = sensitive_reason(
            "\n".join(
                value
                for value in (row["ops"], row.get("corrected_ops"), row.get("error"))
                if isinstance(value, str)
            )
        )
        if screened:
            skips["screened_" + screened] += 1
            continue
        if not parsed["ok"]:
            skips["unparseable_original_cannot_prove_verb_partition"] += 1
            continue
        original_verbs = [
            op.get("tool") for op in parsed.get("ops", []) if isinstance(op, dict)
        ]
        if not original_verbs or any(verb not in schemas for verb in original_verbs):
            skips["unknown_or_internal_original_verb"] += 1
            continue
        candidate = row["ops"] if row["ok"] else row.get("corrected_ops")
        if not row["ok"] and candidate is None and parsed["ok"]:
            candidate = corrected_ast(parsed, schemas)
        if candidate is None:
            skips["failed_operation_without_unique_correction"] += 1
            continue
        validated = validator.parse([candidate], allow_invalid=True)[0]
        if not validated["ok"]:
            skips["parser_rejected_real_or_correction"] += 1
            continue
        try:
            verbs = validate_ast(validated, schemas)
            if verbs != original_verbs:
                raise CurationError(
                    "Correction changes original verb inventory; manual review required"
                )
            if any(verb in EXCLUSIONS for verb in verbs):
                raise CurationError("Excluded semantic contract")
            # Nested payloads need the dedicated synthetic proof and are skipped.
            if "schedule.schedule" in verbs:
                raise CurationError("Real nested dispatch requires manual review")
            parts = {splits[verb] for verb in verbs}
            if len(parts) != 1:
                raise CurationError("Real composition crosses verb partitions")
            if row["ok"]:
                args_text = compact(
                    [
                        {
                            "operation": op["tool"],
                            "arguments": {
                                k: v["value"]
                                for k, v in op["args"].items()
                                if v["kind"] == "value"
                            },
                        }
                        for op in validated["ops"]
                    ]
                )
                # Reference-only rows have no supplied natural-language intent.
                prompt = f"Render this recorded request as {validated['mode']} ops, preserving every operation and argument: {args_text}"
                if any(
                    v["kind"] != "value"
                    for op in validated["ops"]
                    for v in op["args"].values()
                ):
                    raise CurationError(
                        "Real chain has no supplied natural-language reference intent"
                    )
            else:
                if candidate == row["ops"]:
                    raise CurationError("Failed operation has no changed correction")
                prompt = (
                    "Correct this failed request while preserving its intent:\n"
                    + row["ops"]
                )
                if isinstance(row.get("error"), str) and row["error"]:
                    prompt += "\nReported error: " + row["error"]
            item = Example(
                prompt,
                candidate,
                verbs,
                schemas[verbs[0]]["pack"],
                next(iter(parts)),
                "real_success_rendering" if row["ok"] else "real_corrected",
                validated["mode"],
                validated["ops"],
                number,
                "real_reference",
            )
            if not bounded(item):
                raise CurationError("Real row exceeds character budget")
            additions.append(item)
        except CurationError as error:
            skips[str(error)] += 1
    existing = {(e.prompt, e.completion) for e in examples}
    counts = Counter(e.completion for e in examples)
    for item in additions:
        if (item.prompt, item.completion) in existing or counts[item.completion] >= 3:
            skips["duplicate_or_completion_frequency_cap"] += 1
            continue
        existing.add((item.prompt, item.completion))
        counts[item.completion] += 1
        examples.append(item)
    return examples, skips


def safe_output(path):
    """Inside a repository, data must be ignored and entirely untracked (shared guard)."""
    try:
        return curation_guard.safe_output(path, OUTPUT_FILES)
    except GuardError as error:
        raise CurationError(str(error)) from error


def deciles(numbers):
    numbers = sorted(numbers)
    if not numbers:
        return []
    return [numbers[round((len(numbers) - 1) * p / 10)] for p in range(11)]


def curation_report(
    examples, schemas, hashes, splits, validator, drops, real_skips, merge_path
):
    counts = Counter((e.pack, e.split) for e in examples)
    kinds = Counter(e.template.split(":")[0] for e in examples)
    lines = [
        "# Khive DSL curation",
        "",
        f"Rows: {len(examples)}. Seed: `{SEED}`.",
        "",
        "Data is local-only. Inputs are the complete live registry plus verbatim ParamDef help captures.",
        "No generated operation was executed. The only subprocesses are local git path checks and the supplied parse-only validator.",
        "These are schema-based synthetic fixtures: IDs, blobs, profiles, paths, recipients, and record states are hypothetical.",
        "Parser acceptance and registry/parameter/type checks do not establish record existence, authorization, lifecycle transitions, nested domain semantics, task-to-argument equivalence, or successful dispatch.",
        "Templates use explicitly described parameter choices; they intentionally omit unmodeled optional parameters and do not override caller identity or namespace. Explicit actor fields denote requested routing/filter/binding contexts.",
        "",
        "## Validation",
        "",
        f"Validator binary SHA-256: `{validator.binary_hash}`.",
        f"Parser marker: `{PARSER}`; source SHA-256: `{validator.source_hash}`.",
        'The validator consumes JSONL `{"completion":"..."}` on stdin and emits one ordered JSONL result per input.',
        "Every accepted result asserts `executed=false`, exact unchanged completion, AST JSON roundtrip equality, and parser roundtrip equality; parsed verbs, parameter keys, required parameters, and ParamDef types are checked here.",
        "Synthetic parsed ASTs must equal the template's intended AST. Embedded schedule actions are parsed separately and inventoried for leakage checks. No help=true rewriting occurs.",
        "Positive and malformed-input probes test the validator protocol before use. This is not authentication of arbitrary executables: the operator must supply the reviewed parse-only harness; the recorded hashes identify it.",
        "Every staged JSONL and provenance row is read back, compared, and reparsed before publishing the output directory.",
        "",
        "## Budget and distribution",
        "",
        "Prompt <=1200 Unicode characters; completion <=300; total <=1500. No DSL string is truncated.",
        "Token estimates use ceil(total characters / 3.5). This is NOT tokenizer validation. Actual Qwen token lengths and seq-len=512 loader acceptance remain UNVERIFIED by this script.",
        "Deciles below list minimum, p10, p20, ..., p90, maximum.",
        "",
        "| Split | Rows | Total-character deciles | Estimated-token deciles |",
        "| --- | ---: | --- | --- |",
    ]
    for split in SPLITS:
        group = [e for e in examples if e.split == split]
        sizes = [len(e.prompt) + len(e.completion) for e in group]
        lines.append(
            f"| {split} | {len(group)} | {deciles(sizes)} | {deciles([math.ceil(n / 3.5) for n in sizes])} |"
        )
    lines += [
        "",
        "| Pack | Train | Valid | Test | Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pack in sorted({s["pack"] for s in schemas.values()}):
        ns = [counts[pack, split] for split in SPLITS]
        lines.append(f"| {pack} | {ns[0]} | {ns[1]} | {ns[2]} | {sum(ns)} |")
    lines += [
        "",
        "Template category counts: `" + compact(dict(sorted(kinds.items()))) + "`.",
        "Candidate filtering counts (before per-pack quota stops): `"
        + compact(dict(sorted(drops.items())))
        + "`.",
        "Exact (prompt, completion) pairs are unique; each completion occurs at most three times globally. No-argument verbs therefore contribute at most three standalone paraphrases.",
        "",
        "## Partitions",
        "",
        "Split identity is the whole registered verb, ordered by SHA-256 of the fixed seed plus verb name within its pack. The registry capture fixes this partition; reruns with the same registry are stable.",
        "Target proportions are 80/10/10. Heldout counts are round(10% of pack size), floored at one each for packs of three or more verbs; the remainder is train. Packs smaller than three are entirely train. Small packs cannot realize exact percentages; the complete assignments below state the actual result.",
        "The schedule pack is entirely test, including embedded actions. Every operation in a batch/chain/real row, including nested schedule actions, must belong to that row's one partition. Mentioning a pack name as a filter does not invoke its verbs.",
        "Splits are assigned over the entire captured registry before exclusions, so excluded verbs do not reshuffle other verbs.",
        'The comm.probe chain uses the documented since_us parameter contract naming the prior cursor_us. The knowledge.list chain starts after="" and consumes documented next_after; its prompt explicitly assumes a non-null continuation cursor. The DSL has no conditional guard; the fixture precondition is not a production pagination loop.',
        "",
    ]
    lines += [
        "The knowledge.suggest -> knowledge.fold chain consumes the documented results array of scored {id,name,score,size,members} items directly through candidates=$prev.results, as both verb help descriptions specify.",
        "",
    ]
    for split in SPLITS:
        lines += [f"### {split}", ""]
        for pack in sorted({s["pack"] for s in schemas.values()}):
            verbs = sorted(
                v
                for v, s in schemas.items()
                if s["pack"] == pack and splits[v] == split
            )
            lines.append(
                f"- {pack} ({len(verbs)} verbs): "
                + (", ".join(f"`{v}`" for v in verbs) or "none")
            )
    lines += [
        "",
        "## Coverage and exclusions",
        "",
        f"Captured schemas: {len(schemas)}. Verbs represented in accepted rows: {len({v for e in examples for v in e.verbs})}.",
        "",
    ]
    represented = Counter(v for e in examples for v in set(e.verbs))
    for verb in sorted(schemas):
        if verb in EXCLUSIONS:
            lines.append(f"- `{verb}`: EXCLUDED — {EXCLUSIONS[verb]}")
        elif not represented[verb]:
            lines.append(
                f"- `{verb}`: zero accepted rows after quota/dedup/budget; no coverage claimed."
            )
    lines += [
        "",
        "Per-verb accepted row counts: `"
        + compact(dict(sorted(represented.items())))
        + "`.",
        "",
        "## Real reference merge",
        "",
        "Source: "
        + (
            str(Path(merge_path).resolve())
            if merge_path
            else "none; schema synthesis only"
        )
        + ".",
        "Successful recorded ops have no supplied NL side, so optional real additions are labeled rendering-reference exercises, not natural-language observations. Failed requests require explicit corrected_ops or one uniquely recognized invalid parameter-name correction. Unknown/internal verbs, unproven references, unknown corrections, and cross-partition compositions are skipped with reasons.",
        "Real rows accepted: "
        + str(sum(e.origin == "real_reference" for e in examples))
        + ".",
        "Real skip reasons: `" + compact(dict(sorted(real_skips.items()))) + "`.",
        "",
        "## Source hashes",
        "",
    ]
    lines.extend(f"- `{path}`: `{sha}`" for path, sha in sorted(hashes.items()))
    lines += ["", "## Seeded examples", ""]
    rng = random.Random(SEED)
    for split in SPLITS:
        group = [e for e in examples if e.split == split]
        lines += [f"### {split}", ""]
        for example in rng.sample(group, min(3, len(group))):
            lines += ["```json", compact(example.row()), "```", ""]
    return "\n".join(lines)


def write_dataset(
    out, examples, validator, schemas, hashes, splits, drops, real_skips, merge_path
):
    out = safe_output(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".khive-dsl-curation-", dir=out.parent
    ) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        ordered = sorted(
            examples, key=lambda e: (SPLITS.index(e.split), digest(compact(e.row())))
        )
        for split in SPLITS:
            group = [example for example in ordered if example.split == split]
            with (
                (stage / (split + ".jsonl")).open("w") as rows,
                (stage / (split + ".provenance.jsonl")).open("w") as sidecar,
            ):
                for example in group:
                    rows.write(compact(example.row()) + "\n")
                    sidecar.write(compact(example.provenance()) + "\n")
            loaded = [
                json.loads(line)
                for line in (stage / (split + ".jsonl")).read_text().splitlines()
            ]
            provenance = [
                json.loads(line)
                for line in (stage / (split + ".provenance.jsonl"))
                .read_text()
                .splitlines()
            ]
            if len(loaded) != len(group) or len(provenance) != len(group):
                raise CurationError("Readback row count mismatch")
            for row, trace, original in zip(loaded, provenance, group, strict=True):
                if (
                    set(row) != {"prompt", "completion"}
                    or any(not isinstance(v, str) or not v for v in row.values())
                    or row != original.row()
                    or trace != original.provenance()
                ):
                    raise CurationError(
                        "Malformed or changed dataset/provenance row on readback"
                    )
            # Reparse the bytes read back from disk, never an in-memory substitute.
            reread = [
                Example(
                    row["prompt"],
                    row["completion"],
                    original.verbs,
                    original.pack,
                    original.split,
                    original.template,
                    original.mode,
                    original.ops,
                )
                for row, original in zip(loaded, group, strict=True)
            ]
            validate_examples(reread, validator, schemas, splits)
        report = curation_report(
            ordered, schemas, hashes, splits, validator, drops, real_skips, merge_path
        )
        (stage / "CURATION.md").write_text(report)
        if (stage / "CURATION.md").read_text() != report:
            raise CurationError("Curation report readback mismatch")
        # All files pass before an existing, dedicated data directory is replaced.
        backup = Path(temporary) / "previous"
        if out.exists():
            out.rename(backup)
        try:
            stage.rename(out)
        except OSError:
            if backup.exists():
                backup.rename(out)
            raise
    return Counter(example.split for example in ordered)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemas",
        type=Path,
        required=True,
        help="Complete local registry/help capture directory",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--validator",
        type=Path,
        required=True,
        help="Reviewed parse-only JSONL validator binary",
    )
    parser.add_argument(
        "--merge-real",
        type=Path,
        help="Optional local {ops,ok,error,corrected_ops?} reference JSONL",
    )
    args = parser.parse_args(argv)
    try:
        out = safe_output(args.out)
        schemas, hashes = read_schemas(args.schemas)
        splits = split_verbs(schemas)
        validator = Validator(args.validator)
        validator.probe()
        pools, drops = candidates(schemas, splits)
        examples, selection_drops = select_candidates(pools, splits)
        drops.update(selection_drops)
        examples, real_skips = merge_real(
            args.merge_real, examples, validator, schemas, splits
        )
        if not 3000 <= len(examples) <= 6000:
            raise CurationError(
                f"Target is 3000..6000 pairs; obtained {len(examples)}; no quota was silently relaxed"
            )
        validate_examples(examples, validator, schemas, splits)
        counts = write_dataset(
            out,
            examples,
            validator,
            schemas,
            hashes,
            splits,
            drops,
            real_skips,
            args.merge_real,
        )
        print(
            compact(
                {
                    "out": str(out),
                    "counts": {split: counts[split] for split in SPLITS},
                    "total": sum(counts.values()),
                    "tokenizer": "UNVERIFIED; character estimate only",
                    "executed": False,
                }
            )
        )
        return 0
    except (CurationError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
