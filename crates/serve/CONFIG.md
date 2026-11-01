# lattice-serve — CONFIG.md

Configuration reference for the lattice-serve HTTP server and CLI.

## Environment Variables

This table lists the `LATTICE_*` environment variables that only lattice-serve's non-test code under
`crates/serve/src/` reads. Variables that lattice-serve shares with other crates, such as `LATTICE_MODEL_CACHE`
and `LATTICE_KV_F16`, keep their single row in [`crates/inference/CONFIG.md`](../inference/CONFIG.md), whose
"How read" legend applies here too. `crates/inference/tests/env_var_contract.rs` reads both tables as one and
fails when a variable is listed in both.

| Variable                        | Crate | Read in                                | How read                                                                                                            | What it changes                   |
| ------------------------------- | ----- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `LATTICE_SERVE_EMBEDDING_MODEL` | serve | `bin/lattice_serve/main.rs` `imp::run` | value: path or model name, read only when `--embedding-model` is absent                                             | Embedding model the server loads. |
| `LATTICE_SERVE_MODEL`           | serve | `bin/lattice_serve/main.rs` `imp::run` | value: path or model name, read only when `--model` is absent                                                       | Chat model the server loads.      |
| `LATTICE_SERVE_PORT`            | serve | `bin/lattice_serve/main.rs` `imp::run` | value: `u16`, read only when `--port` is absent or unparsable; unparsable text falls through to the default `11435` | Port the server listens on.       |
