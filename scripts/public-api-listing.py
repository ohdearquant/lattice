#!/usr/bin/env python3
"""List a crate's public items with re-exports resolved across crates.

Rustdoc JSON records a cross-crate re-export as a `use` item pointing at an item
in another crate. A listing that stops at `use` items therefore treats every
item moved behind a facade as absent. This script follows those targets through
each crate's own JSON until it reaches the defining item, then prints the path a
caller writes for the item at the facade.

Exit status:
  0 = listing written
  1 = malformed, missing, or unsupported input
  2 = command-line error

Usage:
  public-api-listing.py --crate lattice-inference --doc-dir target/doc
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_FORMAT_VERSIONS = frozenset({57})
UNRESOLVED_CRATES = frozenset({"std", "core", "alloc"})
ITEM_KINDS = frozenset(
    {
        "assoc_const",
        "assoc_type",
        "constant",
        "enum",
        "function",
        "macro",
        "module",
        "primitive",
        "static",
        "struct",
        "trait",
        "type_alias",
        "union",
        "variant",
        "impl",
        "struct_field",
        "use",
        "proc_attribute",
        "proc_derive",
        "proc_macro",
    }
)
TRAVERSED_KINDS = frozenset({"struct", "enum", "trait", "union"})
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DOC_HIDDEN_RE = re.compile(r"#\[doc\([^)]*\bhidden\b[^)]*\)\]")


class ListingError(Exception):
    """The requested listing cannot be produced without hiding a defect."""


@dataclass(frozen=True)
class CrateDocument:
    crate_name: str
    document: dict[str, Any]
    index: dict[str, Any]
    paths: dict[str, Any]
    external_crates: dict[str, Any]

    @property
    def root_id(self) -> int:
        root = self.document.get("root")
        if not isinstance(root, int):
            raise ListingError(f"{self.crate_name}: rustdoc JSON has no integer root")
        return root


class Listing:
    def __init__(self, crate_name: str, doc_dir: Path):
        self.crate_name = crate_name
        self.doc_dir = doc_dir
        self.documents: dict[str, CrateDocument] = {}
        self.listing: set[tuple[str, str]] = set()

    def run(self) -> list[str]:
        root = self.load_crate(self.crate_name)
        root_name = item_name(root, root.root_id)
        self.walk_module(root, root.root_id, root_name, ())
        if not self.listing:
            raise ListingError(f"{self.crate_name}: public listing is empty")
        return [f"{kind} {path}" for kind, path in sorted(self.listing)]

    def load_crate(self, crate_name: str) -> CrateDocument:
        cached = self.documents.get(crate_name)
        if cached is not None:
            return cached
        document_path = self.doc_dir / f"{crate_name}.json"
        if not document_path.is_file():
            identifier = crate_name.replace("-", "_")
            if identifier != crate_name:
                document_path = self.doc_dir / f"{identifier}.json"
        try:
            document = json.loads(document_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ListingError(f"missing rustdoc JSON for crate {crate_name}: {document_path}") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ListingError(f"cannot read rustdoc JSON for crate {crate_name}: {error}") from error
        if not isinstance(document, dict):
            raise ListingError(f"{document_path}: rustdoc JSON root is not an object")
        format_version = document.get("format_version")
        if not isinstance(format_version, int):
            raise ListingError(f"{document_path}: rustdoc JSON has no integer format_version")
        if format_version not in SUPPORTED_FORMAT_VERSIONS:
            versions = ", ".join(str(version) for version in sorted(SUPPORTED_FORMAT_VERSIONS))
            raise ListingError(
                f"{document_path}: unsupported rustdoc JSON format_version {format_version}; "
                f"this script supports {versions}"
            )
        index = required_mapping(document, "index", document_path)
        paths = required_mapping(document, "paths", document_path)
        external_crates = required_mapping(document, "external_crates", document_path)
        crate = CrateDocument(crate_name, document, index, paths, external_crates)
        self.documents[crate_name] = crate
        return crate

    def walk_module(
        self,
        crate: CrateDocument,
        module_id: int,
        module_path: str,
        stack: tuple[tuple[str, int], ...],
    ) -> None:
        inner = item_inner(crate, module_id)
        module = inner.get("module")
        if not isinstance(module, dict):
            raise ListingError(f"{crate.crate_name}: item {module_id} is not a module")
        children = module.get("items")
        if not isinstance(children, list):
            raise ListingError(f"{crate.crate_name}: module {module_id} has no item list")
        for child_id in children:
            if not isinstance(child_id, int):
                raise ListingError(f"{crate.crate_name}: module {module_id} has a non-integer item id")
            if not is_public(self.item(crate, child_id)):
                continue
            self.visit_item(crate, child_id, module_path, stack)

    def visit_item(
        self,
        crate: CrateDocument,
        item_id: int,
        parent_path: str,
        stack: tuple[tuple[str, int], ...],
    ) -> None:
        inner = item_inner(crate, item_id)
        if "use" in inner:
            self.visit_use(crate, item_id, parent_path, stack)
            return
        if "module" in inner:
            is_crate = bool(inner["module"].get("is_crate"))
            child_path = parent_path if is_crate else join_path(parent_path, item_name(crate, item_id))
            if not is_crate:
                self.emit(crate, item_id, "module", child_path)
            self.walk_module(crate, item_id, child_path, stack)
            return
        item_kind = item_kind_for(inner)
        if item_kind in {"impl", "struct_field", "variant"}:
            return
        if not is_public(self.item(crate, item_id)):
            return
        path = join_path(parent_path, item_name(crate, item_id))
        self.emit(crate, item_id, item_kind, path)
        if item_kind in TRAVERSED_KINDS:
            self.walk_type(crate, item_id, path, stack)

    def walk_type(
        self,
        crate: CrateDocument,
        item_id: int,
        path: str,
        stack: tuple[tuple[str, int], ...],
    ) -> None:
        marker = (crate.crate_name, item_id)
        if marker in stack:
            raise ListingError(f"{crate.crate_name}: recursive type traversal at {path}")
        nested = stack + (marker,)
        inner = item_inner(crate, item_id)
        if "struct" in inner:
            self.walk_impls(crate, inner["struct"].get("impls"), path)
            self.walk_struct_fields(crate, inner["struct"], path)
        elif "enum" in inner:
            variants = inner["enum"].get("variants")
            if not isinstance(variants, list):
                raise ListingError(f"{crate.crate_name}: enum {path} has no variant list")
            for variant_id in variants:
                self.walk_variant(crate, variant_id, path)
            self.walk_impls(crate, inner["enum"].get("impls"), path)
        elif "trait" in inner:
            items = inner["trait"].get("items")
            if not isinstance(items, list):
                raise ListingError(f"{crate.crate_name}: trait {path} has no item list")
            for trait_item_id in items:
                self.walk_trait_item(crate, trait_item_id, path)
        elif "union" in inner:
            fields = inner["union"].get("fields")
            if not isinstance(fields, list):
                raise ListingError(f"{crate.crate_name}: union {path} has no field list")
            for field_id in fields:
                self.walk_field(crate, field_id, path)
            self.walk_impls(crate, inner["union"].get("impls"), path)

    def walk_struct_fields(self, crate: CrateDocument, inner: dict[str, Any], path: str) -> None:
        kind = inner.get("kind")
        field_ids: list[Any] = []
        if isinstance(kind, dict):
            plain = kind.get("plain")
            if isinstance(plain, dict):
                field_ids.extend(plain.get("fields", []))
            tuple_fields = kind.get("tuple")
            if isinstance(tuple_fields, list):
                field_ids.extend(field_id for field_id in tuple_fields if field_id is not None)
        for field_id in field_ids:
            if not isinstance(field_id, int):
                raise ListingError(f"{crate.crate_name}: {path} has a non-integer field id")
            self.walk_field(crate, field_id, path)

    def walk_variant(
        self,
        crate: CrateDocument,
        variant_id: int,
        enum_path: str,
    ) -> None:
        variant = self.item(crate, variant_id)
        inner = item_inner(crate, variant_id)
        variant_inner = inner.get("variant")
        if not isinstance(variant_inner, dict):
            raise ListingError(f"{crate.crate_name}: item {variant_id} is not a variant")
        variant_path = join_path(enum_path, item_name(crate, variant_id))
        self.emit(crate, variant_id, "variant", variant_path)
        kind = variant_inner.get("kind")
        fields: list[Any] = []
        if isinstance(kind, dict):
            for shape in ("tuple", "struct"):
                shaped = kind.get(shape)
                if isinstance(shaped, list):
                    fields.extend(field_id for field_id in shaped if field_id is not None)
                elif isinstance(shaped, dict):
                    fields.extend(shaped.get("fields", []))
        for field_id in fields:
            if not isinstance(field_id, int):
                raise ListingError(f"{crate.crate_name}: {variant_path} has a non-integer field id")
            self.walk_field(crate, field_id, variant_path)

    def walk_field(self, crate: CrateDocument, field_id: int, parent_path: str) -> None:
        field = self.item(crate, field_id)
        if is_public(field):
            self.emit(
                crate,
                field_id,
                "field",
                join_path(parent_path, item_name(crate, field_id, allow_tuple_field=True)),
            )

    def walk_impls(self, crate: CrateDocument, impl_ids: Any, type_path: str) -> None:
        if not isinstance(impl_ids, list):
            raise ListingError(f"{crate.crate_name}: type {type_path} has no impl list")
        for impl_id in impl_ids:
            if not isinstance(impl_id, int):
                raise ListingError(f"{crate.crate_name}: {type_path} has a non-integer impl id")
            inner = item_inner(crate, impl_id).get("impl")
            if not isinstance(inner, dict):
                raise ListingError(f"{crate.crate_name}: item {impl_id} is not an impl")
            if inner.get("is_negative") or inner.get("blanket_impl") is not None:
                continue
            trait = inner.get("trait")
            trait_name = trait.get("path") if isinstance(trait, dict) else None
            items = inner.get("items")
            if not isinstance(items, list):
                raise ListingError(f"{crate.crate_name}: impl {impl_id} has no item list")
            for item_id in items:
                if not isinstance(item_id, int):
                    raise ListingError(f"{crate.crate_name}: impl {impl_id} has a non-integer item id")
                self.walk_impl_item(crate, item_id, type_path, trait_name)

    def walk_impl_item(
        self,
        crate: CrateDocument,
        item_id: int,
        type_path: str,
        trait_name: str | None,
    ) -> None:
        inner = item_inner(crate, item_id)
        if "function" in inner:
            kind = "trait_method" if trait_name else "method"
        elif "assoc_const" in inner:
            kind = "trait_assoc_const" if trait_name else "assoc_const"
        elif "assoc_type" in inner:
            kind = "trait_assoc_type" if trait_name else "assoc_type"
        elif "type_alias" in inner:
            kind = "type_alias"
        else:
            name = self.item(crate, item_id).get("name")
            raise ListingError(
                f"{crate.crate_name}: unknown impl item kind {sorted(inner)!r} at {type_path}::{name}"
            )
        self.emit(crate, item_id, kind, join_path(type_path, item_name(crate, item_id)))

    def walk_trait_item(self, crate: CrateDocument, item_id: int, trait_path: str) -> None:
        inner = item_inner(crate, item_id)
        if "function" in inner:
            kind = "trait_method"
        elif "assoc_const" in inner:
            kind = "trait_assoc_const"
        elif "assoc_type" in inner:
            kind = "trait_assoc_type"
        elif "type_alias" in inner:
            kind = "type_alias"
        else:
            name = self.item(crate, item_id).get("name")
            raise ListingError(
                f"{crate.crate_name}: unknown trait item kind {sorted(inner)!r} at {trait_path}::{name}"
            )
        self.emit(crate, item_id, kind, join_path(trait_path, item_name(crate, item_id)))

    def visit_use(
        self,
        crate: CrateDocument,
        use_id: int,
        parent_path: str,
        stack: tuple[tuple[str, int], ...],
    ) -> None:
        if not is_public(self.item(crate, use_id)):
            return
        inner = item_inner(crate, use_id).get("use")
        if not isinstance(inner, dict):
            raise ListingError(f"{crate.crate_name}: item {use_id} is not a use")
        if inner.get("is_glob"):
            target_id = inner.get("id")
            if not isinstance(target_id, int):
                raise ListingError(f"{crate.crate_name}: glob use item {use_id} has no target id")
            self.expand_glob(crate, use_id, target_id, parent_path, stack)
            return
        name = inner.get("name")
        if not isinstance(name, str):
            raise ListingError(f"{crate.crate_name}: use item {use_id} has no name")
        marker = (crate.crate_name, use_id)
        if marker in stack:
            raise ListingError(f"{crate.crate_name}: recursive re-export at {parent_path}::{name}")
        self.follow_use(crate, use_id, parent_path, join_path(parent_path, name), stack + (marker,))

    def expand_glob(
        self,
        crate: CrateDocument,
        use_id: int,
        target_id: int,
        parent_path: str,
        stack: tuple[tuple[str, int], ...],
    ) -> None:
        marker = (crate.crate_name, use_id)
        if marker in stack:
            raise ListingError(f"{crate.crate_name}: recursive glob re-export at {parent_path}")
        nested = stack + (marker,)
        record = crate.paths.get(str(target_id))
        if isinstance(record, dict):
            crate_id = record.get("crate_id")
            external = crate.external_crates.get(str(crate_id)) if isinstance(crate_id, int) else None
            if not isinstance(external, dict) or not isinstance(external.get("name"), str):
                raise ListingError(f"{crate.crate_name}: glob {use_id} targets an unknown crate")
            target_crate = self.load_crate(external["name"])
            parts = record.get("path")
            if not isinstance(parts, list) or not all(isinstance(part, str) for part in parts):
                raise ListingError(f"{crate.crate_name}: glob {use_id} has no usable target path")
            module_id = self.find_definition(target_crate, parts[1:]) if len(parts) > 1 else target_crate.root_id
            if module_id is None:
                module_id = self.find_reexported_definition(target_crate, parts[1:]) if len(parts) > 1 else target_crate.root_id
            if module_id is None:
                raise ListingError(f"{crate.crate_name}: glob {use_id} cannot resolve its target module")
            self.expand_module_glob(target_crate, module_id, parent_path, nested)
            return
        inner = item_inner(crate, target_id)
        if "module" in inner:
            self.expand_module_glob(crate, target_id, parent_path, nested)
            return
        if "use" in inner:
            nested_use = inner["use"]
            target_name = nested_use.get("name")
            if not isinstance(target_name, str):
                raise ListingError(f"{crate.crate_name}: glob {use_id} targets an unnamed use")
            self.follow_use(crate, use_id, parent_path, join_path(parent_path, target_name), nested)
            return
        raise ListingError(f"{crate.crate_name}: glob {use_id} targets a non-module item")

    def expand_module_glob(
        self,
        crate: CrateDocument,
        module_id: int,
        parent_path: str,
        stack: tuple[tuple[str, int], ...],
    ) -> None:
        module = item_inner(crate, module_id).get("module")
        if not isinstance(module, dict):
            raise ListingError(f"{crate.crate_name}: glob target {module_id} is not a module")
        children = module.get("items")
        if not isinstance(children, list):
            raise ListingError(f"{crate.crate_name}: module {module_id} has no item list")
        emitted = False
        for child_id in children:
            if not isinstance(child_id, int):
                raise ListingError(f"{crate.crate_name}: module {module_id} has a non-integer item id")
            node = self.item(crate, child_id)
            if not is_public(node):
                continue
            child_inner = item_inner(crate, child_id)
            if "use" in child_inner:
                child_use = child_inner["use"]
                if child_use.get("is_glob"):
                    self.expand_glob(crate, child_id, child_use.get("id"), parent_path, stack)
                else:
                    name = child_use.get("name")
                    if not isinstance(name, str):
                        raise ListingError(f"{crate.crate_name}: glob member {child_id} has no name")
                    self.follow_use(crate, child_id, parent_path, join_path(parent_path, name), stack)
                emitted = True
                continue
            if "module" in child_inner:
                name = item_name(crate, child_id)
                path = join_path(parent_path, name)
                self.emit(crate, child_id, "module", path)
                self.expand_module_glob(crate, child_id, path, stack)
                emitted = True
                continue
            kind = item_kind_for(child_inner)
            if kind in {"impl", "struct_field", "variant"}:
                continue
            path = join_path(parent_path, item_name(crate, child_id))
            self.emit_definition(crate, child_id, path, stack)
            emitted = True
        if not emitted:
            raise ListingError(f"{crate.crate_name}: glob target {module_id} exposed no public items")

    def follow_use(
        self,
        crate: CrateDocument,
        use_id: int,
        parent_path: str,
        target_path: str,
        stack: tuple[tuple[str, int], ...],
    ) -> None:
        inner = item_inner(crate, use_id).get("use")
        if not isinstance(inner, dict):
            raise ListingError(f"{crate.crate_name}: item {use_id} is not a use")
        target_id = inner.get("id")
        if not isinstance(target_id, int):
            raise ListingError(f"{crate.crate_name}: use item {use_id} has no target id")
        local = crate.index.get(str(target_id))
        if isinstance(local, dict):
            if "use" in item_inner(crate, target_id):
                nested = item_inner(crate, target_id)["use"]
                nested_name = nested.get("name")
                if not isinstance(nested_name, str):
                    raise ListingError(f"{crate.crate_name}: use item {target_id} has no name")
                marker = (crate.crate_name, target_id)
                if marker in stack:
                    raise ListingError(f"{crate.crate_name}: recursive re-export at {target_path}")
                self.follow_use(
                    crate,
                    target_id,
                    target_path.rsplit("::", 1)[0],
                    join_path(target_path.rsplit("::", 1)[0], nested_name),
                    stack + (marker,),
                )
                return
            self.emit_definition(crate, target_id, target_path, stack)
            return

        record = crate.paths.get(str(target_id))
        if not isinstance(record, dict):
            raise ListingError(f"{crate.crate_name}: use item {use_id} targets unknown item {target_id}")
        crate_id = record.get("crate_id")
        if not isinstance(crate_id, int):
            raise ListingError(f"{crate.crate_name}: item {target_id} has no integer crate_id")
        external = crate.external_crates.get(str(crate_id))
        if not isinstance(external, dict) or not isinstance(external.get("name"), str):
            raise ListingError(
                f"{crate.crate_name}: use item {use_id} targets crate {crate_id}, "
                "which is absent from external_crates"
            )
        crate_name = external["name"]
        if crate_name in UNRESOLVED_CRATES:
            raise ListingError(
                f"{crate.crate_name}: use item {use_id} re-exports from {crate_name}, "
                "whose rustdoc JSON is neither present nor part of the workspace"
            )
        parts = record.get("path")
        kind = record.get("kind")
        if not isinstance(parts, list) or not all(isinstance(part, str) for part in parts):
            raise ListingError(f"{crate.crate_name}: item {target_id} has no usable path")
        if not isinstance(kind, str):
            raise ListingError(f"{crate.crate_name}: item {target_id} has no kind")
        if len(parts) < 2:
            raise ListingError(
                f"{crate.crate_name}: use item {use_id} targets the root of {crate_name} "
                "rather than a named member"
            )
        target_crate = self.load_crate(crate_name)
        target_id_local = self.find_definition(target_crate, parts[1:])
        if target_id_local is None:
            target_id_local = self.find_reexported_definition(target_crate, parts[-1:])
        if target_id_local is None:
            member = "::".join(parts[1:])
            raise ListingError(
                f"{crate.crate_name}: use item {use_id} cannot resolve {crate_name}::{member}"
            )
        target = self.item(target_crate, target_id_local)
        target_inner = item_inner(target_crate, target_id_local)
        if kind not in target_inner and "use" not in target_inner:
            raise ListingError(
                f"{crate.crate_name}: use item {use_id} records kind {kind!r}, but "
                f"{crate_name} item {target_id_local} is {sorted(target_inner)!r}"
            )
        self.emit_definition(target_crate, target_id_local, target_path, stack)

    def find_definition(self, crate: CrateDocument, parts: list[str]) -> int | None:
        current_items = item_inner(crate, crate.root_id)["module"].get("items")
        if not isinstance(current_items, list):
            raise ListingError(f"{crate.crate_name}: crate root has no item list")
        current: int | None = None
        for index, component in enumerate(parts):
            matches = [
                item_id
                for item_id in current_items
                if isinstance(item_id, int) and self.item(crate, item_id).get("name") == component
            ]
            if len(matches) > 1:
                raise ListingError(
                    f"{crate.crate_name}: path {'::'.join(parts)!r} has {len(matches)} "
                    f"matches at {component!r}"
                )
            if not matches:
                return None
            current = matches[0]
            if index == len(parts) - 1:
                return current
            inner = item_inner(crate, current)
            if "use" in inner:
                current = self.resolve_local_use(crate, current, ())
                inner = item_inner(crate, current)
            if "module" not in inner:
                return None
            current_items = inner["module"].get("items")
            if not isinstance(current_items, list):
                raise ListingError(f"{crate.crate_name}: module {current} has no item list")
        return None

    def find_reexported_definition(self, crate: CrateDocument, parts: list[str]) -> int | None:
        if len(parts) < 1:
            return None
        wanted = parts[0]
        root = item_inner(crate, crate.root_id)["module"].get("items")
        if not isinstance(root, list):
            raise ListingError(f"{crate.crate_name}: crate root has no item list")
        candidates: list[int] = []
        for item_id in root:
            if not isinstance(item_id, int):
                raise ListingError(f"{crate.crate_name}: crate root has a non-integer item id")
            node = self.item(crate, item_id)
            inner = item_inner(crate, item_id)
            if "use" in inner and inner["use"].get("name") == wanted:
                candidates.append(item_id)
            elif "module" in inner and node.get("name") == wanted:
                candidates.append(item_id)
        if len(candidates) != 1:
            return None
        current = candidates[0]
        if "use" in item_inner(crate, current):
            current = self.resolve_local_use(crate, current, ())
        if len(parts) == 1:
            return current
        inner = item_inner(crate, current)
        if "module" not in inner:
            return None
        child_items = inner["module"].get("items")
        if not isinstance(child_items, list):
            raise ListingError(f"{crate.crate_name}: module {current} has no item list")
        return self.find_in_items(crate, child_items, parts[1:])

    def find_in_items(
        self, crate: CrateDocument, items: list[Any], parts: list[str]
    ) -> int | None:
        component = parts[0]
        matches = [
            item_id
            for item_id in items
            if isinstance(item_id, int) and self.item(crate, item_id).get("name") == component
        ]
        if len(matches) != 1:
            return None
        current = matches[0]
        if len(parts) == 1:
            return current
        if "use" in item_inner(crate, current):
            current = self.resolve_local_use(crate, current, ())
        inner = item_inner(crate, current)
        if "module" not in inner:
            return None
        children = inner["module"].get("items")
        if not isinstance(children, list):
            raise ListingError(f"{crate.crate_name}: module {current} has no item list")
        return self.find_in_items(crate, children, parts[1:])

    def resolve_local_use(
        self,
        crate: CrateDocument,
        use_id: int,
        stack: tuple[int, ...],
    ) -> int:
        if use_id in stack:
            raise ListingError(f"{crate.crate_name}: recursive local use at item {use_id}")
        inner = item_inner(crate, use_id).get("use")
        if not isinstance(inner, dict) or not isinstance(inner.get("id"), int):
            raise ListingError(f"{crate.crate_name}: use item {use_id} has no target id")
        target_id = inner["id"]
        if str(target_id) in crate.index and "use" in item_inner(crate, target_id):
            return self.resolve_local_use(crate, target_id, stack + (use_id,))
        return target_id

    def emit_definition(
        self,
        crate: CrateDocument,
        item_id: int,
        path: str,
        stack: tuple[tuple[str, int], ...],
    ) -> None:
        inner = item_inner(crate, item_id)
        if "use" in inner:
            target = inner["use"].get("id")
            if not isinstance(target, int):
                raise ListingError(f"{crate.crate_name}: use item {item_id} has no target id")
            self.follow_use(crate, item_id, path.rsplit("::", 1)[0], path, stack)
            return
        item_kind = item_kind_for(inner)
        if item_kind in {"impl", "struct_field", "variant"}:
            return
        if item_kind not in ITEM_KINDS:
            raise ListingError(f"{crate.crate_name}: unknown item kind {item_kind!r} at {path}")
        if item_kind == "module":
            self.emit(crate, item_id, "module", path)
            self.walk_module(crate, item_id, path, stack)
            return
        if item_kind == "use":
            target = inner["use"].get("id")
            if not isinstance(target, int):
                raise ListingError(f"{crate.crate_name}: use item {item_id} has no target id")
            self.follow_use(crate, item_id, path.rsplit("::", 1)[0], path, stack)
            return
        self.emit(crate, item_id, item_kind, path)
        if item_kind in TRAVERSED_KINDS:
            self.walk_type(crate, item_id, path, stack)

    def emit(self, crate: CrateDocument, item_id: int, kind: str, path: str) -> None:
        if is_hidden(self.item(crate, item_id)):
            self.listing.add(("hidden", path))
        else:
            self.listing.add((kind, path))

    def item(self, crate: CrateDocument, item_id: int) -> dict[str, Any]:
        node = crate.index.get(str(item_id))
        if not isinstance(node, dict):
            raise ListingError(f"{crate.crate_name}: item id {item_id} is missing from index")
        return node


def required_mapping(document: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ListingError(f"{path}: rustdoc JSON has no {key} object")
    return value


def item_inner(crate: CrateDocument, item_id: int) -> dict[str, Any]:
    inner = crate.index[str(item_id)].get("inner")
    if not isinstance(inner, dict) or len(inner) != 1:
        raise ListingError(f"{crate.crate_name}: item {item_id} has no single item kind")
    return inner


def item_name(crate: CrateDocument, item_id: int, *, allow_tuple_field: bool = False) -> str:
    name = crate.index[str(item_id)].get("name")
    if isinstance(name, str) and IDENTIFIER_RE.fullmatch(name) is not None:
        return name
    if allow_tuple_field and isinstance(name, str) and name.isdigit():
        return name
    raise ListingError(f"{crate.crate_name}: item {item_id} has an unusable name {name!r}")


def item_kind_for(inner: dict[str, Any]) -> str:
    if len(inner) != 1:
        raise ListingError(f"unknown item kind {sorted(inner)!r}")
    kind = next(iter(inner))
    if kind not in ITEM_KINDS:
        raise ListingError(f"unknown item kind {kind!r}")
    return kind


def is_public(node: dict[str, Any]) -> bool:
    visibility = node.get("visibility")
    if not isinstance(visibility, str):
        raise ListingError(f"item {node.get('id')} has no string visibility")
    return visibility == "public"


def is_hidden(node: dict[str, Any]) -> bool:
    attrs = node.get("attrs")
    if not isinstance(attrs, list):
        raise ListingError(f"item {node.get('id')} has no attribute list")
    for attribute in attrs:
        if isinstance(attribute, dict) and isinstance(attribute.get("other"), str):
            if DOC_HIDDEN_RE.search(attribute["other"]):
                return True
    return False


def join_path(parent: str, name: str) -> str:
    return f"{parent}::{name}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List a crate's public items with cross-crate re-exports resolved."
    )
    parser.add_argument(
        "--crate",
        required=True,
        help="package or crate name; hyphens resolve to the underscore form rustdoc writes",
    )
    parser.add_argument(
        "--doc-dir",
        required=True,
        type=Path,
        help="directory containing <crate>.json for the crate and its workspace dependencies",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lines = Listing(args.crate, args.doc_dir).run()
    except ListingError as error:
        print(f"public-api-listing: error: {error}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
