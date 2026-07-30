# Frozen E2E parity reference regeneration

The ordinary pull-request, push, and merge-queue parity jobs compare lattice
against the committed Hugging Face reference in
`crates/inference/tests/fixtures/e2e_parity_reference_v1/reference.json`.
Regenerate that fixture only when deliberately advancing a reference dependency
or the pinned model snapshot.

## Pinned environment

Use Python 3.11 with the same reference packages installed by
`.github/workflows/e2e-parity.yml`:

```text
torch==2.12.1
transformers==5.12.1
tokenizers==0.22.2
huggingface_hub==1.20.1
```

The model is `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`. The revision and verified file
digests live in
`.github/actions/provision-qwen35-0-8b/action.yml`; the parity script repeats
the revision as `MODEL_REVISION`. Package pins live in the workflow, and the
fixture records the versions that produced it.

## Provision the local snapshot

Provision the exact, hash-verified snapshot described by
`.github/actions/provision-qwen35-0-8b/action.yml`. Make it available either at
the default `~/.lattice/models/qwen3.5-0.8b` path or set
`LATTICE_MODEL_DIR` to its flat snapshot directory. The directory must contain
the verified safetensors files and no pickle (`*.bin`) weights.

Build the CPU generator before regenerating:

```sh
cargo build --release --bin qwen35_generate -p lattice-inference --features f16
```

If `LATTICE_BIN` is set, it must name that verified local binary.

## Regenerate

From the repository root, with the pinned Python environment active, run:

```sh
E2E_MAX_TOKENS=15 python3 scripts/e2e_parity_check.py --regenerate
```

Regeneration runs isolated one-thread and four-thread Hugging Face workers. It
refuses to write the fixture if their token IDs disagree, a measured logit
margin is non-finite or below `0.1`, the requested token budget is less than
15, the model snapshot is unavailable or unsafe, or reference generation or
lattice parity fails. A successful run validates both thread-count outputs,
runs the lattice comparison, and rewrites `reference.json`.

## Review and pin updates

Review the fixture diff before committing it:

- Confirm `model.repo_id` and `model.revision` match the provision action and
  `MODEL_REVISION`.
- Confirm the four `package_versions` exactly match the workflow installation
  pins.
- Confirm every prompt still corresponds to `PROMPTS`, has exactly one
  4-token and one 15-token reference, and retains its intended `match_window`.
- Inspect every `generated_ids` and `logit_margins` change. The one-thread and
  four-thread agreement and the global minimum-margin summary must explain the
  new baseline; an unexplained prompt or token change is not a rebaseline.

In the same change, update all affected pins: the four package versions in
`.github/workflows/e2e-parity.yml`, or the model revision and verified hashes
in `.github/actions/provision-qwen35-0-8b/action.yml`, plus
`MODEL_REVISION` in `scripts/e2e_parity_check.py`. Run
`python3 -m unittest tests/test_e2e_parity_reference.py` and validate
`.github/workflows/e2e-parity.yml` with `actionlint`.
