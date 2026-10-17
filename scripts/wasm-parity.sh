#!/usr/bin/env bash
# wasm-parity.sh: build the wasm32 embedding path and gate its output against
# the same HF-reference goldens and native-lattice reference used by the
# native parity test (crates/embed/tests/embed_parity_vs_hf.rs).
#
# Usage:
#   scripts/wasm-parity.sh
#   make wasm-parity
#
# Skip-graceful: if node, wasm-bindgen, the wasm32 target, or the bge-small /
# all-MiniLM-L6-v2 weights (model.safetensors under ~/.lattice/models/<name>/)
# are missing, this prints a
# one-line reason and exits 0 so a plain workspace build is unaffected.
#
# Fail-closed: set LATTICE_WASM_PARITY_ENFORCE=1 to turn every skip above into
# a hard failure (exit 1) instead; used by the CI gate, where a missing
# prerequisite means provisioning failed, not that the check is optional.
#
# Model files: the native embed CLI's --download-only fetches
# model.safetensors and (vocab.txt or tokenizer.json, model-dependent) but
# never config.json, and for WordPiece models (bge-small, MiniLM) never
# tokenizer.json either; the wasm LatticeEmbedder constructor needs both, so
# this script fetches them at the native downloader's pinned revisions and
# verifies both cached and downloaded bytes against the WASM model registry.
# Integrity and download errors always fail, including outside enforced CI.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${LATTICE_MODELS_DIR:-$HOME/.lattice/models}"
ENFORCE="${LATTICE_WASM_PARITY_ENFORCE:-}"

skip_or_fail() {
  reason="$1"
  if [ -n "$ENFORCE" ]; then
    echo "wasm-parity: FAIL (LATTICE_WASM_PARITY_ENFORCE=1): $reason" >&2
    exit 1
  fi
  echo "wasm-parity: SKIPPED: $reason"
  exit 0
}

# --- Tool prerequisites -----------------------------------------------------

command -v node >/dev/null 2>&1 || skip_or_fail "node not found on PATH"
command -v wasm-bindgen >/dev/null 2>&1 || skip_or_fail "wasm-bindgen (CLI) not found on PATH; install with: cargo install wasm-bindgen-cli --version 0.2.105"
command -v cargo >/dev/null 2>&1 || skip_or_fail "cargo not found on PATH"

if ! rustup target list --installed 2>/dev/null | grep -q '^wasm32-unknown-unknown$'; then
  rustup target add wasm32-unknown-unknown >/dev/null 2>&1 \
    || skip_or_fail "wasm32-unknown-unknown target not installed and could not be added"
fi

# --- Model file prerequisites ------------------------------------------------
for dir_name in bge-small-en-v1.5 all-minilm-l6-v2; do
  model_dir="$MODELS_DIR/$dir_name"
  if [ ! -f "$model_dir/model.safetensors" ]; then
    echo "wasm-parity: MISSING $model_dir/model.safetensors, provision with:" >&2
    echo "  cargo run --release -p lattice-embed --bin embed -- --model $dir_name --download-only" >&2
    skip_or_fail "model weights missing for $dir_name"
  fi
done

# Keep integrity failures outside skip_or_fail so optional runs cannot hide them.
node --input-type=module - "$REPO" "$MODELS_DIR" <<'JS'
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const [repo, modelsDir] = process.argv.slice(2);
const { MODEL_REGISTRY } = await import(
  pathToFileURL(join(repo, 'npm/lattice-embed-wasm/registry.mjs')).href
);
// These revisions match hf_source in crates/inference/src/download.rs.
const sources = [
  ['bge', 'BAAI/bge-small-en-v1.5', '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'],
  ['minilm', 'sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'],
];

function verify(bytes, expected, file) {
  const actual = createHash('sha256').update(bytes).digest('hex');
  if (actual !== expected) {
    throw new Error(`wasm-parity: SHA-256 mismatch for ${file}: expected ${expected}, got ${actual}`);
  }
}

for (const [name, hfId, revision] of sources) {
  const model = MODEL_REGISTRY[name];
  if (!model) throw new Error(`wasm-parity: missing model registry entry: ${name}`);
  const modelDir = join(modelsDir, model.localDir);
  for (const fileName of [model.files.config, model.files.tokenizer]) {
    const file = join(modelDir, fileName);
    let cached;
    try {
      cached = await readFile(file);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (cached !== undefined) {
      verify(cached, model.sha256[fileName], file);
      continue;
    }

    const url = `https://huggingface.co/${hfId}/resolve/${revision}/${fileName}`;
    console.log(`wasm-parity: fetching ${fileName} for ${hfId} at ${revision}`);
    const bytes = execFileSync('curl', ['-fsSL', '--max-time', '120', url], {
      maxBuffer: 16 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'inherit'],
    });
    verify(bytes, model.sha256[fileName], file);
    const temporaryDir = await mkdtemp(join(modelDir, '.wasm-parity-'));
    try {
      const temporaryFile = join(temporaryDir, fileName);
      await writeFile(temporaryFile, bytes);
      await rename(temporaryFile, file);
    } finally {
      await rm(temporaryDir, { recursive: true, force: true });
    }
  }
}
JS

for dir_name in bge-small-en-v1.5 all-minilm-l6-v2; do
  model_dir="$MODELS_DIR/$dir_name"
  for f in model.safetensors config.json tokenizer.json; do
    [ -f "$model_dir/$f" ] || skip_or_fail "$model_dir/$f still missing after provisioning attempt"
  done
done

# --- Build wasm --------------------------------------------------------------

echo "=== wasm-parity: building lattice-embed for wasm32 ==="
cargo build --release --target wasm32-unknown-unknown -p lattice-embed \
  --no-default-features --features wasm

WASM_OUT_DIR="$REPO/target/wasm-parity-out"
mkdir -p "$WASM_OUT_DIR"
wasm-bindgen --target nodejs --out-dir "$WASM_OUT_DIR" \
  "$REPO/target/wasm32-unknown-unknown/release/lattice_embed.wasm"

# --- Build native reference dump ---------------------------------------------

echo "=== wasm-parity: building native reference dump ==="
NATIVE_DUMP="$REPO/target/wasm-parity-native-dump.json"
DUMP_OUT="$NATIVE_DUMP" cargo run --release -p lattice-embed \
  --example dump_parity_embeddings

# --- Run the harness ---------------------------------------------------------

echo "=== wasm-parity: running Node harness ==="
LATTICE_WASM_JS="$WASM_OUT_DIR/lattice_embed.js" \
LATTICE_NATIVE_DUMP="$NATIVE_DUMP" \
LATTICE_MODELS_DIR="$MODELS_DIR" \
  node "$REPO/crates/embed/tests/wasm/embed_parity_wasm.mjs"
