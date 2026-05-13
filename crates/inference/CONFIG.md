# lattice-inference — CONFIG.md

Complete configuration reference for the lattice-inference pure-Rust transformer inference engine.

---

## ModelInferenceConfig

**Source**: `src/model/qwen.rs:244`

Per-model runtime configuration, loaded from `inference_config.json` in the model directory. Defaults preserve exact Qwen3-Embedding-0.6B behavior — a model with no `inference_config.json` works out of the box.

**Loading** (`qwen.rs:276`): `ModelInferenceConfig::load(model_dir)` reads `{model_dir}/inference_config.json`. Missing file → defaults. Malformed JSON → warning logged, defaults used. Non-Qwen models that lack `config.json` will error earlier at `parse_qwen_config`, so this struct only matters for Qwen variants.

### Fields

| Field                    | Type    | Default  | Source of Default                                   |
| ------------------------ | ------- | -------- | --------------------------------------------------- |
| `eos_token_id`           | `u32`   | `151643` | `default_eos_token_id()` at `qwen.rs:255`           |
| `rope_table_max_seq_len` | `usize` | `8192`   | `default_rope_table_max_seq_len()` at `qwen.rs:258` |
| `gpu_max_seq_len`        | `usize` | `2048`   | `default_gpu_max_seq_len()` at `qwen.rs:261`        |

**Default impl** at `qwen.rs:265` calls each `default_*` function.

### Field Details

#### `eos_token_id` — default `151643`

**What it does**: The end-of-sequence token ID appended during `tokenize_for_embedding()`. When this token is generated during text generation, decoding stops.

**Why `151643`**: This is Qwen3-Embedding-0.6B's EOS token as defined in the model's `tokenizer_config.json`. Different Qwen variants may use different values (check the model's tokenizer config). BERT/BGE models don't use this field — they have their own EOS handling via `[SEP]` token (ID 102).

**Where used**: Embedding tokenization (`qwen.rs` tokenize function), text generation EOS check (`generate.rs:49` `GenerateOutput.stopped_by_eos`).

#### `rope_table_max_seq_len` — default `8192`

**What it does**: Maximum sequence length for RoPE (Rotary Position Embedding) frequency table precomputation. Caps the cos/sin table allocation in both single-file and sharded weight loading paths.

**Why `8192`**: Qwen3 models natively support up to 32K positions (`max_position_embeddings` in `config.json`), but embedding workloads rarely exceed 8K tokens. Setting this to the full 32K wastes ~4× memory on frequency tables that are never indexed. For generation workloads targeting longer contexts, increase to `max_position_embeddings`.

**Where used**: RoPE table initialization (`rope.rs:8` `RopeTable`), model loading in `QwenModel::load_*()`.

#### `gpu_max_seq_len` — default `2048`

**What it does**: Maximum sequence length for Metal GPU buffer pre-allocation. Controls the size of `qkv_out`, `q_buf`, `k_buf`, `v_buf`, `attn_out`, and activation buffers allocated at model init time.

**Why `2048`**: Embedding inputs are typically <512 tokens; 2048 provides 4× headroom without excessive GPU memory usage. For the 4B model with wider hidden dimensions (2560 vs 1024), the per-token buffer cost is 2.5× higher — reduce to 1024 if GPU memory is tight. For generation workloads, increase up to `rope_table_max_seq_len`.

**Where used**: Metal forward pass buffer allocation (`forward/metal.rs` `MetalForwardPass`), KV cache sizing (`kv_cache.rs` `FlatKVCacheConfig`).

### Example: `inference_config.json`

```json
{
  "eos_token_id": 151643,
  "rope_table_max_seq_len": 8192,
  "gpu_max_seq_len": 2048
}
```

Omit any field to use the default. An empty `{}` is valid and equivalent to all defaults.

---

## GenerateConfig

**Source**: `src/generate.rs:25`

Text generation configuration for decoder-only models.

| Field            | Type             | Default   | Source                             | Why                                                                                           |
| ---------------- | ---------------- | --------- | ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `max_new_tokens` | `usize`          | `256`     | `Default` impl at `generate.rs:37` | Practical limit for chat-style responses. Increase for long-form generation                   |
| `sampling`       | `SamplingConfig` | see below | `SamplingConfig::default()`        | Balanced temperature+nucleus sampling                                                         |
| `eos_token_id`   | `Option<u32>`    | `None`    | `Default` impl                     | When `None`, generation runs to `max_new_tokens`. Set to model's EOS to enable early stopping |
| `include_prompt` | `bool`           | `false`   | `Default` impl                     | Whether output includes the original prompt text                                              |

---

## SamplingConfig

**Source**: `src/sampling.rs:9`

Token sampling parameters for text generation.

| Field                | Type    | Default | Source                             | Why                                                                                                            |
| -------------------- | ------- | ------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `temperature`        | `f32`   | `0.7`   | `Default` impl at `sampling.rs:21` | Standard "creative but coherent" temperature. 0.0 = greedy (deterministic), 1.0 = full entropy                 |
| `top_k`              | `usize` | `50`    | `Default` impl                     | Keep only 50 highest-probability tokens before sampling. 0 = disabled. Reduces noise from low-probability tail |
| `top_p`              | `f32`   | `0.9`   | `Default` impl                     | Nucleus sampling: keep tokens covering 90% cumulative probability. 1.0 = disabled                              |
| `repetition_penalty` | `f32`   | `1.1`   | `Default` impl                     | Penalize previously generated tokens. 1.0 = no penalty. Higher = more diverse but may degrade coherence        |

**Preset**: `SamplingConfig::greedy()` at `sampling.rs:33` — temperature=0.0, top_k=1, top_p=1.0, repetition_penalty=1.0.

---

## Metal GPU Shape Constraints

**Source**: `src/forward/metal.rs:1014-1043`

The Metal fused attention shader has **compile-time MSL constants** that cannot be changed at runtime. A Rust-side guard (`validate_fused_kernel_shape` at `metal.rs:1017`) validates the model config before any GPU buffer allocation.

### Constants

| MSL Constant       | Rust Constant            | Value | Purpose                                                                                   |
| ------------------ | ------------------------ | ----- | ----------------------------------------------------------------------------------------- |
| `FA_HEAD_DIM`      | `METAL_FUSED_HEAD_DIM`   | `128` | Fused attention tile dimension. Model requires `hidden_size / num_attention_heads == 128` |
| `FA_GQA_GROUPS`    | `METAL_FUSED_GQA_GROUPS` | `2`   | GQA group count. Model requires `num_attention_heads / num_key_value_heads == 2`          |
| `FUSED_C_HEAD_DIM` | (same as above)          | `128` | Used in fused QK-norm + RoPE kernel at `metal.rs:886`                                     |
| `FUSED_C_HALF_DIM` | (derived)                | `64`  | `HEAD_DIM / 2` for RoPE frequency pairs                                                   |

### Validation Logic

```
validate_fused_kernel_shape(config: &QwenConfig):
  1. config.head_dim != 128            → Err("head_dim mismatch")
  2. config.num_key_value_heads == 0    → Err("zero kv_heads")
  3. num_heads % num_kv_heads != 0      → Err("not divisible")
  4. num_heads / num_kv_heads != 2      → Err("GQA groups mismatch")
  All pass                              → Ok(())
```

**On mismatch**: `QwenModel` sets `self.metal = None`. All forward passes silently fall back to the **CPU NEON path** (`forward/neon_forward.rs`). No error is raised — the model still works, just slower.

### Model Compatibility Matrix

| Model                 | hidden | heads | kv_heads | head_dim | GQA   | Metal GPU? | Why               |
| --------------------- | ------ | ----- | -------- | -------- | ----- | ---------- | ----------------- |
| Qwen3-Embedding-0.6B  | 1024   | 16    | 8        | **64**   | 2     | **NO**     | head_dim=64 ≠ 128 |
| Qwen3-Embedding-4B    | 2560   | 32    | 16       | **80**   | 2     | **NO**     | head_dim=80 ≠ 128 |
| Qwen3-8B (generation) | 4096   | 32    | 8        | 128      | **4** | **NO**     | gqa=4 ≠ 2         |
| Qwen3.5-14B           | 5120   | 40    | 8        | 128      | **5** | **NO**     | gqa=5 ≠ 2         |

**Currently no Qwen3-Embedding model passes the Metal gate.** The GPU kernels were optimized for a Qwen3.5 generation configuration (head_dim=128, GQA=2) that no shipping model matches. All embedding models run on CPU/NEON.

**To add Metal support for other head_dim values**: Rewrite the MSL shader with parameterized tile dimensions. This is a shader rewrite, not a config change.

---

## Environment Variables

| Variable                      | Type           | Default                                  | Used In                                  | Source                                             |
| ----------------------------- | -------------- | ---------------------------------------- | ---------------------------------------- | -------------------------------------------------- |
| `LATTICE_NO_GPU`              | presence check | —                                        | Disable Metal GPU, force CPU path        | `qwen.rs:440,497`                                  |
| `LATTICE_INFERENCE_MODEL_DIR` | path           | `~/.lattice/models/`                     | Override model directory for all models  | `qwen.rs:1672,1695,1740`, `bert.rs:522`            |
| `LATTICE_MODEL_CACHE`         | path           | `~/.lattice/models/`                     | Model cache/download directory           | `lib.rs:49` (fallback to `$HOME/.lattice/models/`) |
| `LATTICE_QWEN_MODEL_DIR`      | path           | `~/.lattice/models/Qwen3-Embedding-0.6B` | Qwen backfill binary model dir           | `bin/backfill_qwen3.rs:13`                         |
| `LATTICE_QWEN36_MODEL_DIR`    | path           | `~/.lattice/models/Qwen3.6-3B-Instruct`  | Qwen3.6 generation model dir             | `model/qwen35.rs:2150`                             |
| `METAL_LAYERS`                | integer        | —                                        | Limit number of Metal GPU layers (debug) | `forward/metal.rs:1141`                            |
| `LATTICE_PROFILE`             | presence check | —                                        | Enable GPU profiling                     | `forward/metal_qwen35.rs:2499`                     |
| `LATTICE_COMPACT_TOPK`        | presence check | —                                        | Enable compact top-k GPU path            | `forward/metal_qwen35.rs:3821,5273`                |
| `LATTICE_COMPACT_TOPK_SELECT` | presence check | —                                        | Enable compact top-k select variant      | `forward/metal_qwen35.rs:3822,5274`                |
| `LATTICE_DECODE_ENFORCE_GATE` | presence check | —                                        | Enforce decode gate (profiling)          | `examples/profile_metal_decode.rs:696`             |

**Priority**: `LATTICE_INFERENCE_MODEL_DIR` overrides the default `~/.lattice/models/` path for both BERT and Qwen model loading. `LATTICE_MODEL_CACHE` is the download-path equivalent used by `default_cache_dir()`.

---

## Config File Locations

| File                    | Location                                                | Purpose                                                               |
| ----------------------- | ------------------------------------------------------- | --------------------------------------------------------------------- |
| `config.json`           | `{model_dir}/config.json`                               | HuggingFace model architecture config (hidden_size, num_layers, etc.) |
| `inference_config.json` | `{model_dir}/inference_config.json`                     | lattice-inference runtime config (EOS, RoPE, GPU buffer sizes)        |
| `engine.toml`           | `~/.lattice/engine.toml`                                | Active embedding model selection (consumed by platform/engine)        |
| Model weights           | `{model_dir}/*.safetensors`                             | Safetensors weight files (single or sharded)                          |
| Tokenizer               | `{model_dir}/tokenizer.json` or `{model_dir}/vocab.txt` | Tokenizer vocabulary and config                                       |

**Default model directory**: `~/.lattice/models/` — overridable via `LATTICE_INFERENCE_MODEL_DIR` or `LATTICE_MODEL_CACHE`.

---

## Adding a New Model — Checklist

1. **Add variant** to `EmbeddingModel` enum (`foundation/embed/src/model.rs:102`).
   Set `native_dimensions()`, `max_input_tokens()`, `query_instruction()`, `model_id()`, `supports_output_dim()`.
   Add to `is_local()` if it runs locally.

2. **Create `inference_config.json`** in the model directory.
   - `eos_token_id`: from the model's `tokenizer_config.json`
   - `rope_table_max_seq_len`: from `config.json` → `max_position_embeddings`
   - `gpu_max_seq_len`: based on expected workload and GPU memory budget

3. **Check Metal compatibility**: compute `head_dim = hidden_size / num_attention_heads`.
   If `head_dim != 128` or `gqa_groups != 2` → model runs CPU-only (automatic, no error).

4. **Test**: `cargo test -p lattice-inference`, manual embedding generation.
