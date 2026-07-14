# lattice-inference — CONFIG.md

Configuration reference for the lattice-inference pure-Rust transformer inference engine.

---

## ModelInferenceConfig

**Source**: `src/model/qwen.rs` (`ModelInferenceConfig`)

Per-model runtime configuration, loaded from `inference_config.json` in the model directory. Defaults preserve exact Qwen3-Embedding-0.6B behavior — a model with no `inference_config.json` works out of the box.

**Loading**: `ModelInferenceConfig::load(model_dir)` reads `{model_dir}/inference_config.json`. Missing file → defaults. Malformed JSON → warning logged, defaults used. Non-Qwen models that lack `config.json` will error earlier at `parse_qwen_config`, so this struct only matters for Qwen variants.

### Fields

| Field                    | Type     | Default                | Source of Default                                   |
| ------------------------ | -------- | ---------------------- | --------------------------------------------------- |
| `eos_token_id`           | `u32`    | `151643`               | `default_eos_token_id()` at `qwen.rs:283`           |
| `rope_table_max_seq_len` | `usize`  | `8192`                 | `default_rope_table_max_seq_len()` at `qwen.rs:286` |
| `gpu_max_seq_len`        | `usize`  | `2048`                 | `default_gpu_max_seq_len()` at `qwen.rs:289`        |
| `base_model_rev`         | `String` | `"none"`, then derived | `default_cache_compat_rev()` at `qwen.rs:292`       |
| `tokenizer_rev`          | `String` | `"none"`, then derived | `default_cache_compat_rev()` at `qwen.rs:292`       |

**Default impl** calls each `default_*` function.

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

**Where used**: Metal forward pass buffer allocation (`forward/metal.rs` `MetalForwardPass`), KV cache sizing (`kv_cache/flat.rs` `FlatKVCacheConfig`).

#### `base_model_rev` — default `"none"`, derived on load

**What it does**: Identifies the model revision recorded in embedding-cache manifests. `cache_load()` rejects a cache whose revision differs, before loading any entries.

**Precedence**: An explicit non-`"none"` value in `inference_config.json` wins. Otherwise `ModelInferenceConfig::load()` derives `sha256:<16 hex chars>` from `config.json` plus the weight-file names, lengths, and boundary samples. If `config.json` is missing or unreadable, the value remains `"none"`.

#### `tokenizer_rev` — default `"none"`, derived on load

**What it does**: Identifies the tokenizer revision recorded in embedding-cache manifests. `cache_load()` rejects a cache whose tokenizer revision differs.

**Precedence**: An explicit non-`"none"` value wins. Otherwise `ModelInferenceConfig::load()` derives `sha256:<16 hex chars>` from the complete `tokenizer.json` contents. If that file is missing or unreadable, the value remains `"none"`.

### Example: `inference_config.json`

```json
{
  "eos_token_id": 151643,
  "rope_table_max_seq_len": 8192,
  "gpu_max_seq_len": 2048,
  "base_model_rev": "qwen3-embedding-0.6b-rev1",
  "tokenizer_rev": "qwen3-tokenizer-rev1"
}
```

Omit any field to use the default. An empty `{}` is valid and equivalent to all defaults.

---

## GenerateConfig

**Source**: `src/generate.rs` (`GenerateConfig`)

Legacy text generation configuration for decoder-only models. This `crate::generate::GenerateConfig` type is deprecated since 0.5.1 but remains functional during its compatibility window; the canonical Qwen3.5 configuration is `crate::model::GenerateConfig` in `src/model/qwen35_config.rs`.

| Field               | Type                         | Default   | Source                              | Why                                                                                           |
| ------------------- | ---------------------------- | --------- | ----------------------------------- | --------------------------------------------------------------------------------------------- |
| `max_new_tokens`    | `usize`                      | `256`     | `Default` impl at `generate.rs:126` | Practical limit for chat-style responses. Increase for long-form generation                   |
| `sampling`          | `SamplingConfig`             | see below | `SamplingConfig::default()`         | Balanced temperature+nucleus sampling                                                         |
| `eos_token_id`      | `Option<u32>`                | `None`    | `Default` impl                      | When `None`, generation runs to `max_new_tokens`. Set to model's EOS to enable early stopping |
| `include_prompt`    | `bool`                       | `false`   | `Default` impl                      | Whether output includes the original prompt text                                              |
| `grammar`           | `Option<Arc<GrammarEngine>>` | `None`    | `Default` impl                      | Masks logits and advances grammar state at every generated token                              |
| `kv_cache_capacity` | `Option<usize>`              | `None`    | `Default` impl                      | Optional per-request cap on KV allocation and effective generation length                     |

When `grammar` masks every token, generation fails closed with `InferenceError::InvalidInput`; a token rejected while advancing the grammar stops with `StopReason::Grammar`. `kv_cache_capacity` is clamped to `[1, prompt_len + max_new_tokens]`; a cap smaller than the prompt is rejected, and generation stops with `StopReason::KvFull` when the cache reaches the effective cap.

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

**Source**: `src/forward/metal.rs` (`validate_fused_kernel_shape`, `msl_source_for`)

The Metal fused attention shader uses **model-specific compile-time MSL constants**. `MetalForwardPass::new` validates structural requirements, then `msl_source_for` injects the model's head dimension and GQA group count before compiling the shader library.

### Constants

| MSL Constant       | Injected value                              | Purpose                               |
| ------------------ | ------------------------------------------- | ------------------------------------- |
| `FA_HEAD_DIM`      | `config.head_dim`                           | Fused attention dimension             |
| `FA_GQA_GROUPS`    | `num_attention_heads / num_key_value_heads` | Query heads served by each KV head    |
| `FUSED_C_HEAD_DIM` | `config.head_dim`                           | Fused QK-norm + RoPE dimension        |
| `FUSED_C_HALF_DIM` | `config.head_dim / 2`                       | RoPE frequency-pair count             |
| `FUSED_C_THREADS`  | `config.head_dim / 2`                       | Fused QK-norm + RoPE threadgroup size |

### Validation Logic

```
validate_fused_kernel_shape(config: &QwenConfig):
  1. config.head_dim == 0                → Err("nonzero head_dim required")
  2. config.head_dim % 4 != 0            → Err("head_dim must be divisible by 4")
  3. config.num_key_value_heads == 0     → Err("nonzero kv_heads required")
  4. num_heads % num_kv_heads != 0       → Err("not divisible")
  All pass                               → Ok(num_heads / num_kv_heads)
```

**On mismatch**: `QwenModel` leaves `self.metal` unset. Forward passes use the CPU path instead; no initialization error is raised to the caller.

### Model Compatibility Matrix

| Model                | hidden | heads | kv_heads | head_dim | GQA | Structural gate |
| -------------------- | ------ | ----- | -------- | -------- | --- | --------------- |
| Qwen3-Embedding-0.6B | 1024   | 16    | 8        | 64       | 2   | Pass            |
| Qwen3-Embedding-4B   | 2560   | 32    | 16       | 80       | 2   | Pass            |

Both Qwen3 embedding shapes pass the structural gate. Actual GPU activation also requires a Metal device, successful shader compilation, and `LATTICE_NO_GPU` to be unset. Initialization failure leaves `QwenModel::metal` unset and falls back to CPU; a per-call Metal failure also logs a warning and retries that forward pass on CPU.

Qwen3.5 generation uses the separate `forward/metal_qwen35.rs` engine and is not governed by this embedding-path validator.

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

3. **Check Metal compatibility**: require nonzero `head_dim` divisible by 4, nonzero
   `num_key_value_heads`, and `num_attention_heads` divisible by `num_key_value_heads`.
   The resulting head dimension and GQA group count are injected into the MSL source.

4. **Test**: `cargo test -p lattice-inference`, manual embedding generation.
