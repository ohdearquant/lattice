# Micro-adapter datasets

These tools prepare reproducible JSONL datasets with nonempty `prompt` and
`completion` strings. Generated data and source snapshots belong in local storage;
only the preparation tools are checked in.

## Commit messages

Set `LATTICE_REPO`, `RUNTIME_REPO`, and `LIONAGI_REPO` to local clones of
`ohdearquant/lattice`, `ohdearquant/khive`, and `ohdearquant/lionagi`. Set `DATA_DIR`
to your local dataset directory. Run from this checkout:

```sh
uv run scripts/microlora/mine_commit_msg.py \
  --repo "$LATTICE_REPO" \
  --repo "$RUNTIME_REPO" \
  --repo "$LIONAGI_REPO" \
  --out "$DATA_DIR/commit-msg"
```

Each repository's `main` is resolved to an immutable commit before mining. The
miner also requires each checkout's origin URL to name one of the three public
repositories. That is a wrong-checkout guard, not an authenticity check: whoever
controls the checkout's git configuration can satisfy it, so a replay is verified
by the recorded commit IDs, never by the URL match. The
miner walks non-merge first-parent history, filters conventional commit messages,
removes excluded diff paths and context lines, and includes the repository name in
each prompt. Oversized diffs receive an explicit omitted-lines marker. Duplicate
pairs and boilerplate completions are removed before output.

Git must support `--attr-source`. An isolated temporary bare repository reads the
source objects and the pinned tree's attributes, so local configuration, untracked
attributes, and the source index cannot change the rendered patches.

For a later replay, supply the full recorded commit ID for every repository with
repeated `--pin-main name=full-sha` arguments, using names `lattice`, `khive`, and
`lionagi`. Each revision must be a commit reachable from that repository's current
local `main`. Explicit pins keep a moving branch from changing the corpus; no
source branch or checkout is modified.

The author timestamp determines the split:

| Split | Author timestamp in UTC       |
| ----- | ----------------------------- |
| train | Before 2026-07-15             |
| valid | 2026-07-15 through 2026-08-14 |
| test  | 2026-08-15 onward             |

The output directory contains `train.jsonl`, `valid.jsonl`, `test.jsonl`, and
`CURATION.md`. Before writing, the miner refuses an output path that a repository
tracks or does not ignore, and an existing directory holding unrelated files.
Commits whose message or rendered diff carries an address, a credential shape, or
a control character are skipped whole; both lanes share one screen in
`curation_guard.py`. The report records source commit IDs, filtering counts, split
counts, length distributions, and reproducible sample rows. Readback validation
rejects malformed output. Source paths must exist; pass `--repo` explicitly when your
checkout layout differs from the defaults.

Prompt and completion character limits provide a conservative size estimate.
They do not prove that a row fits a tokenizer's sequence limit. Before training,
measure the complete prompt plus completion with the model's tokenizer, including
the completion boundary, and record any rows skipped by the training loader.
To reproduce exclusions established by that check, add
`--reject-pairs "$REJECTIONS_JSON"` to the mining command. This file is a JSON array
of SHA-256 hashes of rejected pairs. Hash the UTF-8 encoding of
`json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
The miner records the manifest hash and rejection counts and refuses unknown or
unused entries. It does not perform tokenization; recheck the final files with the
tokenizer after exclusions.

Run the miner's fixture tests with:

```sh
uv run --no-project python -m unittest discover \
  -s scripts/microlora -p 'test_mine_commit_msg.py'
```

## Natural-language requests to DSL

Capture the complete live registry using `verbs()`, then capture each listed verb
with `<verb>(help=true)`. Store the unchanged result objects in a local directory
named by `SCHEMA_DIR`: `_verbs.json` for the registry and `<verb>.json` for every
help result. Keep the original response envelopes alongside the capture when
available. The generator requires a help capture for every verb the captured
registry names and records the capture's input hashes in `CURATION.md`; it cannot
tell whether that registry was complete when captured.

Build the parser validator against a local runtime checkout. Set `VALIDATOR` to an
executable output path outside that runtime checkout:

```sh
uv run --no-project scripts/microlora/build_dsl_validator.py \
  --runtime-repo "$RUNTIME_REPO" \
  --out "$VALIDATOR"

uv run --no-project scripts/microlora/synth_khive_dsl.py \
  --schemas "$SCHEMA_DIR" \
  --validator "$VALIDATOR" \
  --out "$DATA_DIR/khive-dsl"
```

The builder uses an ephemeral Cargo project and cached dependencies with
`cargo build --offline`; it does not add a crate to either workspace. An optional
`--target-dir` retains a dedicated build cache. The default temporary build cache
is removed after the executable, lockfile, build log, and source-hash receipt are
saved. Rebuilding at the same output path reuses the saved lockfile with `--locked`.

The validator calls the runtime's real request parser and round-trips its AST.
It never dispatches operations. The generator checks registered verbs, required
parameters, parameter types, intended AST equality, and every file after readback.
These checks establish syntax and the stated schema contracts; they do not prove
record existence, authorization, or successful execution against a database.

Output includes the three JSONL splits, matching `.provenance.jsonl` sidecars, and
`CURATION.md`. It must be local storage outside tracked paths; repository locations
must be ignored and untracked, and both lanes refuse to write anywhere else.
Generated data is never part of the source commit.
The fixed partition assigns whole verbs to splits, with the entire `schedule` pack
held out for testing. Batches, chains, and embedded scheduled actions must stay
within one partition. A registered verb with neither a reviewed template nor a
listed exclusion stops generation before any row is written, and the refusal names
every such verb. The report lists actual small-pack proportions, excluded verbs
with their reasons, template counts, and validation limits. `check_split_integrity.py`
checks exact prompt and completion overlap and prompt-length balance between two
split files; it does not re-derive the verb partition, so a same-verb row in two
splits passes it.

To include recorded calls, add `--merge-real "$REAL_JSONL"`. Each input row must
contain string `ops`, boolean `ok`, and optionally `error` and `corrected_ops`.
Successful calls become rendering-reference exercises because this input format
contains no observed natural-language request. Failed calls need an explicit
correction or one uniquely recognized parameter-name correction; other failures
are skipped with reasons. Recorded calls receive the same parser and split checks,
and a row whose operations, correction, or error text carries an address or a
credential shape is skipped whole with a counted reason; nothing is redacted,
because the completion is the recorded text.

Run all DSL tests, including the actual parser and full dataset generation, with:

```sh
uv run --no-project scripts/microlora/test_synth_khive_dsl.py \
  --schemas "$SCHEMA_DIR" \
  --validator "$VALIDATOR"
```

The DSL generator also uses character budgets rather than tokenization. Check
the final prompt-plus-completion tokens with the target model before training.
