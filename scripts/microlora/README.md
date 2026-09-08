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
miner walks non-merge first-parent history, filters conventional commit messages,
removes excluded diff paths and context lines, and includes the repository name in
each prompt. Oversized diffs receive an explicit omitted-lines marker. Duplicate
pairs and boilerplate completions are removed before output.

Git must support `--attr-source`. An isolated temporary bare repository reads the
source objects and the pinned tree's attributes, so local configuration, untracked
attributes, and the source index cannot change the rendered patches.

The author timestamp determines the split:

| Split | Author timestamp in UTC       |
| ----- | ----------------------------- |
| train | Before 2026-07-15             |
| valid | 2026-07-15 through 2026-08-14 |
| test  | 2026-08-15 onward             |

The output directory contains `train.jsonl`, `valid.jsonl`, `test.jsonl`, and
`CURATION.md`. The report records source commit IDs, filtering counts, split counts,
length distributions, and reproducible sample rows. Readback validation rejects
malformed output. Source paths must exist; pass `--repo` explicitly when your
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
