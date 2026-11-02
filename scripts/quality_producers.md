# Fixed-length generation dumps

The paired producers accept the same local Qwen3.5 safetensors directory and UTF-8 manifest:

```json
{ "schema_version": 1, "max_new_tokens": 50, "cases": [{ "id": "example", "prompt": "The answer is" }] }
```

Each case id must be unique and each prompt must be nonempty. `max_new_tokens` must be positive. The CPU producer refuses tokenizer truncation and records its actual unpadded input ids. Both reject incomplete generations and publish only a complete JSON dump via an exclusive atomic link; choose a new output filename for each run.

```sh
cargo run --release -p lattice-inference --bin quality_dump -- /path/to/checkpoint manifest.json lattice.json lattice
python3 scripts/quality_hf_dump.py /path/to/checkpoint manifest.json hf.json hf
python3 scripts/quality_compare.py hf.json lattice.json
```

Both read `crates/inference/quality_protocol_v1.json` at runtime and reject unsupported effective settings; pass an optional fifth positional argument to either producer to select its path. The CPU default resolves relative to the current directory, while the HF default resolves relative to the script.

The HF producer requires local torch and transformers with `use_model_defaults` support; it requests CPU float32, local safetensors and no remote code. Before any output it forces the checkpoint's EOS as the first generated token and requires a second token, printing `QUALITY_EOS_CONTROL` to stderr when the control succeeds. It refuses if the checkpoint EOS or the control is unavailable. HF disables sampling processors because greedy sampling is selected; the shared `crates/inference/quality_protocol_v1.json` records the equivalent lattice greedy settings. Prompt-id mismatches make the comparator refuse, so this does not claim tokenizer parity. A real pinned checkpoint run and the numerical meaning of its comparison remain host validation work.
