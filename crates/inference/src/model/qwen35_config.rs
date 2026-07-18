//! Configuration for Qwen hybrid attention models (3.5-2B and 3.6-35B-A3B).
//!
//! Qwen3.5-2B uses a hybrid architecture with two attention mechanisms:
//! - **GatedDeltaNet** (linear attention): 18 layers with recurrent state
//! - **Full GQA** (standard attention): 6 layers with KV cache
//!
//! Qwen3.6-35B-A3B extends this with a Mixture-of-Experts (MoE) FFN:
//! 256 routed experts, top-8 selection, 1 shared expert, separate lm_head.

use crate::error::InferenceError;
use crate::grammar::GrammarEngine;
use crate::stop_reason::StopReason;
use std::path::Path;
use std::sync::Arc;

/// Chat turn end token for Qwen models.
pub const QWEN_CHAT_IM_END_TOKEN_ID: u32 = 248_046;

/// Token IDs for Qwen3.6 thinking mode (248K vocab tokenizer).
pub const QWEN3_THINK_OPEN_TOKEN_ID: u32 = 248_068;
pub const QWEN3_THINK_CLOSE_TOKEN_ID: u32 = 248_069;
pub const QWEN3_NEWLINE_TOKEN_ID: u32 = 198;

/// Upper bound on `num_hidden_layers` accepted from `config.json`.
///
/// `num_hidden_layers` comes from an untrusted checkpoint directory and, before this
/// bound existed, only had a nonzero check. `compute_layer_types`, `normalize_layer_mask`,
/// and the loader's `Vec::with_capacity` all allocate proportionally to it — an extreme
/// value (e.g. `usize::MAX`) reaches those allocations before any structural validation,
/// causing unbounded allocation / OOM / abort at model load. Real Qwen3.5 checkpoints are
/// well under 100 layers; 512 rejects any config a real checkpoint would never carry while
/// leaving generous headroom for future architectures.
pub(crate) const MAX_HIDDEN_LAYERS: usize = 512;

/// Upper bound on `vocab_size` accepted from `config.json`.
///
/// `vocab_size` comes from an untrusted checkpoint directory and drives the logits buffer
/// allocation (`resize(&mut self.logits, cfg.vocab_size)` in the decode cache) on every
/// forward pass. Real Qwen3.5/3.6 tokenizers are 151K-262K entries; 4,000,000 rejects any
/// config a real checkpoint would never carry (including a zero-byte-embedding attack that
/// pairs a huge declared vocab with an empty tensor) while leaving over an order of
/// magnitude of headroom for future tokenizers.
pub(crate) const MAX_VOCAB_SIZE: usize = 4_000_000;

/// Upper bound, in bytes, on the text embedding tensor (`vocab_size * hidden_size * 4`)
/// accepted from `config.json` (CLASS A3, materialization site 1).
///
/// `load_weights` (`model/qwen35/loading.rs:434`) materializes `embed_tokens` as an
/// owned `Vec<f32>` of exactly this product; [`MAX_VOCAB_SIZE`] and [`MAX_HIDDEN_SIZE`]
/// bound each factor individually, but not their product, which reaches multi-terabyte
/// scale well within both individual caps (e.g. `vocab_size = 4_000_000`, `hidden_size =
/// 1_048_576` passes both scalar caps while materializing ~16.8 TB). Real Qwen3.5/3.6
/// presets top out around `vocab_size = 248_320` * `hidden_size = 8192` (~8.1 GiB); 32
/// GiB (34_359_738_368) leaves roughly 4x headroom above that for future architectures
/// while rejecting the terabyte-scale hostile case by several orders of magnitude.
pub(crate) const MAX_EMBEDDING_BYTES: u128 = 34_359_738_368;

/// Upper bound on `head_dim` accepted from `config.json`.
///
/// `head_dim` comes from an untrusted checkpoint directory and sizes the RoPE table
/// (`RopeTable::new` allocates `max_seq_len * head_dim / 2` entries) independent of layer
/// mix — an all-linear-attention config never calls `checked_full_q_dim`, so this is the
/// only guard standing between an attacker-controlled `head_dim` and that allocation. Real
/// Qwen3.5/3.6 head dims are 64-256; 2048 rejects any config a real checkpoint would never
/// carry while leaving 8x headroom for future architectures.
pub(crate) const MAX_HEAD_DIM: usize = 2048;

/// Upper bound on `num_experts` accepted from `config.json` for MoE variants.
///
/// `num_experts` comes from an untrusted checkpoint directory and drives
/// `ForwardScratch::ensure_capacity`'s unconditional `resize(&mut self.router_logits,
/// cfg.num_experts.unwrap_or(0))` on every forward pass. A zero-sized checkpoint tensor
/// satisfies shape checks trivially (0 elements), so nothing else catches an inflated
/// value before that resize. Real Qwen3.5/3.6 MoE configs route across 8-256 experts;
/// 4096 rejects any config a real checkpoint would never carry while leaving well over an
/// order of magnitude of headroom for future architectures.
pub(crate) const MAX_NUM_EXPERTS: usize = 4096;

/// Upper bound on the full-attention `q_dim` / `kv_dim` products
/// (`num_attention_heads * head_dim`, `num_key_value_heads * head_dim`) accepted from
/// `config.json`.
///
/// `head_dim` is separately bounded by [`MAX_HEAD_DIM`], but `num_attention_heads` and
/// `num_key_value_heads` are not, so a non-overflowing product (e.g. `num_attention_heads =
/// 2^40`, `head_dim = 2048`) can still pass the `checked_mul` overflow guard in
/// `from_config_json_str` while driving `ForwardScratch::ensure_capacity`'s `q_buf` /
/// `context` allocations to exabyte scale. Real Qwen3.5/3.6 full-attention geometry tops out
/// around `num_attention_heads = 64` * `head_dim = 256` (`q_dim = 16384`); 1,048,576 (2^20)
/// leaves roughly 64x headroom above that for future architectures.
pub(crate) const MAX_FULL_ATTENTION_DIM: usize = 1_048_576;

/// Upper bound on `intermediate_size`, `moe_intermediate_size`, and
/// `shared_expert_intermediate_size` accepted from `config.json`.
///
/// These sizes drive MLP/MoE-expert scratch buffer (`Vec<f32>`) allocations at load and
/// generation time; a present-but-malformed config can pair a huge intermediate size with
/// zero-sized expert tensors (which pass shape checks trivially, being empty) and blow up
/// those allocations before any tensor is read. Real Qwen3.5/3.6 presets range up to 17,408
/// for `intermediate_size` and ~512 for the MoE/shared-expert variants; 1,048,576 (2^20)
/// leaves roughly 60x headroom above the largest observed real value.
pub(crate) const MAX_INTERMEDIATE_SIZE: usize = 1_048_576;

/// Upper bound on `VisionModelConfig::depth` accepted from `config.json`.
///
/// The vision checkpoint loader mints ~12 tensor-name `String`s per ViT block *before* any
/// tensor validation runs, so an unbounded `depth` (only checked nonzero) drives unbounded
/// `String` allocation ahead of any shape check. Real Qwen3.5-VL vision towers are ~24-48
/// blocks deep (Qwen3.5-VL vision ~32); 1024 leaves roughly 32x headroom above that.
pub(crate) const MAX_VISION_DEPTH: usize = 1024;

/// Upper bound on the length of small fixed-purpose config vectors accepted from
/// `config.json`: `RopeParams::mrope_section` and `VisionModelConfig::deepstack_visual_indexes`.
///
/// Both fields are deserialized as `Vec<usize>` before `from_config_json_str`'s validation
/// pass runs, so an unbounded declared length drives uncontrolled allocation at parse time --
/// even while neither field is yet consumed downstream (ADR-069 S1; wired in S3+). Real
/// `mrope_section` values carry 3-4 entries (one per M-RoPE axis) and real
/// `deepstack_visual_indexes` lists carry a handful of layer indexes; 1024 rejects any config a
/// real checkpoint would never carry while leaving generous headroom for future architectures.
pub(crate) const MAX_CONFIG_VECTOR_LEN: usize = 1024;

/// Upper bound on the GatedDeltaNet output dimension (`linear_num_value_heads *
/// linear_value_head_dim`) accepted from `config.json`.
///
/// `linear_output_dim()` is only checked-multiplied inside `load_linear_attention_weights`,
/// which an all-full-attention config never calls, yet `ForwardScratch::ensure_capacity`
/// unconditionally derives `max_q8_input` from it and resizes `x_q_scratch` regardless of
/// layer mix. A non-overflowing but attacker-inflated product (or one that wraps `usize`)
/// therefore reaches that allocation with no other guard in its path for such a config. Real
/// Qwen3.5/3.6 GDN configs top out around `linear_num_value_heads = 48` * `linear_value_head_dim
/// = 128` (`linear_output_dim = 6144`); 1,048,576 (2^20) leaves roughly 170x headroom above
/// that for future architectures.
pub(crate) const MAX_LINEAR_OUTPUT_DIM: usize = 1_048_576;

/// Upper bound on the GatedDeltaNet state-matrix three-factor product
/// (`linear_num_value_heads() * linear_key_head_dim * linear_value_head_dim`) accepted from
/// `config.json`.
///
/// `attention/gdn.rs` allocates `s_matrices = vec![0.0; value_heads * key_dim * value_dim]` at
/// first generation. [`MAX_LINEAR_OUTPUT_DIM`] only bounds the `value_heads * value_dim`
/// factor (`linear_output_dim()`); `linear_key_head_dim` is a free multiplier on top of that
/// budget, so e.g. `value_heads = 1`, `value_dim` near the `MAX_LINEAR_OUTPUT_DIM` cap, and
/// `linear_key_head_dim = 1_000_000` passes both existing guards while allocating on the order
/// of terabytes. Real Qwen3.5/3.6 GDN configs are around `linear_num_value_heads = 48` *
/// `linear_key_head_dim = 128` * `linear_value_head_dim = 128` (~786,432); 16,777,216 (2^24)
/// leaves roughly 21x headroom above that for future architectures.
pub(crate) const MAX_GDN_STATE_SIZE: usize = 16_777_216;

/// Upper bound on `max_position_embeddings` accepted from `config.json`.
///
/// Drives the Metal RoPE table allocation (`rope_max * rope_dim / 2` entries in
/// `build_rope_interleaved`) independent of `head_dim`, which is separately bounded by
/// [`MAX_HEAD_DIM`]. Real Qwen3.5/3.6 presets default to 262,144 and advertise context windows
/// up to roughly 1M; 4,194,304 (2^22) leaves roughly 4x headroom above that for future
/// architectures.
pub(crate) const MAX_POSITION_EMBEDDINGS: usize = 4_194_304;

/// Upper bound on `num_attention_heads` and `num_key_value_heads` accepted from
/// `config.json`.
///
/// `ForwardScratch::ensure_capacity` derives `scores = num_attention_heads * (max_kv_len + 1)`
/// independent of `head_dim` -- [`MAX_FULL_ATTENTION_DIM`] bounds the `num_attention_heads *
/// head_dim` product, but a config that pairs a small `head_dim` with a huge
/// `num_attention_heads` can still pass that product check while inflating `scores` on its
/// own. Real Qwen3.5/3.6 configs top out around 128 attention heads; 8,192 leaves roughly 64x
/// headroom above that for future architectures.
pub(crate) const MAX_ATTENTION_HEADS: usize = 8_192;

/// Upper bound on `hidden_size` accepted from `config.json`.
///
/// Drives the `hidden` / `residual` / `attn_out` / `ffn_out` / `expert_out` scratch buffer
/// allocations and the `max_q8_input` derivation in `ForwardScratch::ensure_capacity`, all
/// unconditional on every forward pass regardless of layer mix. Real Qwen3.5/3.6 presets top
/// out around 8,192; 1,048,576 (2^20) matches the scale of [`MAX_FULL_ATTENTION_DIM`] and
/// [`MAX_INTERMEDIATE_SIZE`] while leaving well over two orders of magnitude of headroom.
pub(crate) const MAX_HIDDEN_SIZE: usize = 1_048_576;

/// Upper bound on `linear_num_key_heads` accepted from `config.json`.
///
/// Unlike `linear_num_value_heads()`, which is implicitly bounded by [`MAX_LINEAR_OUTPUT_DIM`]
/// and [`MAX_GDN_STATE_SIZE`] (both product budgets that carry it as a factor),
/// `linear_num_key_heads` was previously checked only for zero -- it had no upper bound at
/// all. It is a free multiplier in `linear_qkv_dim()` (`2 * linear_num_key_heads *
/// linear_key_head_dim + linear_num_value_heads() * linear_value_head_dim`), which sizes the
/// GatedDeltaNet QKV projection weight, conv1d weight, and conv/activation scratch buffers
/// throughout `attention/gdn.rs`, `model/qwen35/loading.rs`, and `forward/metal_qwen35.rs` --
/// none of those call sites re-validate it. Real Qwen3.5/3.6 GDN configs use
/// `linear_num_key_heads = 16`; 8,192 matches the existing [`MAX_ATTENTION_HEADS`] scale while
/// leaving 512x headroom above that for future architectures.
pub(crate) const MAX_LINEAR_NUM_KEY_HEADS: usize = 8_192;

/// Upper bound on the *resolved* `linear_num_value_heads()` accepted from `config.json`.
///
/// Unlike `linear_num_key_heads`, `linear_num_value_heads` is a factor of two existing
/// product budgets ([`MAX_LINEAR_OUTPUT_DIM`] bounds `value_heads * linear_value_head_dim`;
/// [`MAX_GDN_STATE_SIZE`] bounds `value_heads * linear_key_head_dim * linear_value_head_dim`)
/// -- but both are products, and with the *other* factor pinned at 1 (a value real
/// checkpoints never carry, but a parseable config can), `value_heads` alone can reach
/// 1,048,576 while still passing both checks. `new_session_inner`
/// (`forward/metal_qwen35.rs`) uses `linear_num_value_heads()` directly as a free
/// multiplier in its GatedDeltaNet chunk-scratch geometry (`chunk_rows = num_chunks *
/// num_vh * GDN_CHUNK_SIZE`), independent of those two products, so this is the only
/// remaining guard bounding that multiplier on its own. Real Qwen3.5/3.6 GDN configs use
/// `linear_num_value_heads` up to 48; 8,192 matches [`MAX_LINEAR_NUM_KEY_HEADS`]'s scale
/// while leaving 170x headroom above that for future architectures. See
/// [`MAX_GDN_CHUNK_SCRATCH_BYTES`] for the accompanying product budget that this bound
/// alone does not close (a huge `linear_key_head_dim` / `linear_value_head_dim` with a
/// small `value_heads` still reaches multi-gigabyte chunk-scratch allocation).
pub(crate) const MAX_LINEAR_NUM_VALUE_HEADS: usize = 8_192;

/// Mirror of `forward/metal_qwen35.rs`'s `GDN_CHUNK_SIZE` constant (defined at
/// `metal_qwen35.rs:1426`). `new_session_inner`'s GatedDeltaNet chunk-scratch geometry
/// tiles the prefill window into chunks of this size; duplicated here (rather than
/// imported) because `forward/metal_qwen35.rs` is behind the `metal-gpu` feature and
/// `qwen35_config.rs` must validate on every build, including CPU-only ones. If the
/// engine's chunk size ever changes, this mirror must change with it -- see the
/// `test_gdn_chunk_scratch_mirror_matches_engine_constant` regression test.
const GDN_CHUNK_SIZE_MIRROR: usize = 32;

/// Mirror of `forward/metal_qwen35.rs`'s hardcoded `max_cache_len.min(512)` prefill cap
/// used in both `new_session_inner` call sites (`metal_qwen35.rs:3682`, `:15676`). The
/// number of GatedDeltaNet chunks (`num_chunks = bp.div_ceil(GDN_CHUNK_SIZE)`) is bounded
/// by this literal regardless of `max_position_embeddings`, since `bp` is always
/// `min(max_cache_len, 512)`. Duplicated here for the same reason as
/// [`GDN_CHUNK_SIZE_MIRROR`].
const GDN_MAX_PREFILL_MIRROR: usize = 512;

/// Upper bound, in bytes, on the largest single GatedDeltaNet chunk-scratch buffer
/// (`new_session_inner`'s `GdnChunkScratch`, `forward/metal_qwen35.rs`) accepted from
/// `config.json`.
///
/// [`MAX_LINEAR_NUM_VALUE_HEADS`] bounds `linear_num_value_heads()` alone, but the
/// chunk-scratch buffers multiply it by `linear_key_head_dim` or `linear_value_head_dim`
/// (the `q`/`k`/`w`/`k_right` buffers are `chunk_rows * key_head_dim`; `v`/`u`/`r` are
/// `chunk_rows * value_head_dim`; `kkt`/`qk_l` are `chunk_rows * GDN_CHUNK_SIZE`) -- a
/// config with `value_heads` pinned small and either head-dim factor huge still passes
/// [`MAX_GDN_STATE_SIZE`] (which bounds their three-factor product, so a huge head-dim
/// with `value_heads = 1` satisfies it trivially) while reaching multi-gigabyte
/// chunk-scratch allocation. `chunk_rows` itself is bounded independent of
/// `max_position_embeddings`, since the engine caps the prefill window at
/// [`GDN_MAX_PREFILL_MIRROR`] (`bp = max_cache_len.min(512)`) before deriving chunk count
/// -- so this budget only needs `value_heads` and the two head-dim fields, not
/// `max_position_embeddings`. Real Qwen3.5/3.6 GDN configs peak around `value_heads = 48`,
/// `key_head_dim = value_head_dim = 128`: `chunk_rows_upper = 16 * 48 * 32 = 24,576`,
/// worst single buffer `24,576 * 128 = 3,145,728` elements (~12.6 MiB f32). 1 GiB
/// (1,073,741,824 bytes) leaves roughly 85x headroom above that for future architectures,
/// matching [`MAX_ROPE_TABLE_BYTES`]'s scale and style.
pub(crate) const MAX_GDN_CHUNK_SCRATCH_BYTES: usize = 1_073_741_824;

/// Upper bound, in bytes, on the SUM of every per-session GatedDeltaNet scratch/qkv
/// buffer `new_session_inner` (`forward/metal_qwen35.rs`) allocates together (CLASS A2).
///
/// [`MAX_GDN_CHUNK_SCRATCH_BYTES`] only bounds the largest *single* `GdnChunkScratch`
/// buffer, but `new_session_inner` allocates several key-dimension buffers
/// (`gdn_qkv`, `gdn_z`, `gdn_qkvz`, `gdn_key_scratch`, `gdn_raw_out`) plus all twelve
/// `GdnChunkScratch` buffers *together* for one session -- a config can pass every
/// per-buffer guard while the aggregate still reaches several GiB. Mirrors every
/// allocation-site formula from `new_session_inner` (see the CLASS A2 table in the PR
/// body); computed in `u128` so the summation itself cannot overflow `usize`. Real
/// Qwen3.5/3.6 GDN configs (worst preset: `qwen36_27b`, value_heads=48,
/// key_dim=value_dim=128) sum to roughly 134 MiB; 2 GiB (2,147,483,648) leaves
/// roughly 16x headroom above that while rejecting the ~6 GiB hostile geometries
/// (e.g. `value_heads=4096, key_dim=128, value_dim=32`) that pass every per-buffer
/// guard individually.
pub(crate) const MAX_GDN_SESSION_BYTES: usize = 2_147_483_648;

/// Upper bound on `linear_conv_kernel_dim` accepted from `config.json`.
///
/// Previously checked only for zero (guarding the `linear_conv_kernel_dim - 1` unsigned
/// underflow). It sizes the GatedDeltaNet causal conv1d buffer (`conv_dim *
/// (linear_conv_kernel_dim - 1)` in `GatedDeltaNetState::new`, `attention/gdn.rs`) and the
/// conv1d weight tensor (`qkv_dim * kernel_size` in `model/qwen35/loading.rs`), both
/// unconditional at first generation / weight load regardless of layer mix. Real Qwen3.5/3.6
/// GDN configs use `linear_conv_kernel_dim = 4`; 512 rejects any config a real checkpoint
/// would never carry while leaving generous headroom for future architectures. See
/// [`MAX_GDN_CONV_BUFFER_SIZE`] for the accompanying product budget.
pub(crate) const MAX_CONV_KERNEL_DIM: usize = 512;

/// Upper bound on the GatedDeltaNet conv1d buffer element count
/// (`linear_qkv_dim() * (linear_conv_kernel_dim - 1)`) accepted from `config.json`.
///
/// [`MAX_CONV_KERNEL_DIM`] and [`MAX_LINEAR_NUM_KEY_HEADS`] bound their respective factors
/// individually, but their product still drives `GatedDeltaNetState::new`'s `conv_buffer =
/// vec![0.0; conv_dim * buf_len]` (mirrored in the Metal GDN conv buffer allocation in
/// `forward/metal_qwen35.rs`) -- two generously-headroomed per-factor caps can still compose
/// into a multi-gigabyte allocation. Real Qwen3.5/3.6 GDN configs have `linear_qkv_dim() ~
/// 6144` and `linear_conv_kernel_dim - 1 = 3` (~18,432 elements); 16,777,216 (2^24) matches the
/// [`MAX_GDN_STATE_SIZE`] scale while leaving roughly 900x headroom above that for future
/// architectures.
pub(crate) const MAX_GDN_CONV_BUFFER_SIZE: usize = 16_777_216;

/// Upper bound, in bytes, on the Metal RoPE cos/sin table pair accepted from `config.json`.
///
/// `build_rope_interleaved` (`forward/metal_qwen35.rs`) allocates two `Vec<f32>` of
/// `max_position_embeddings * (rope_dim / 2)` entries each (cos and sin tables) --
/// `4 * max_position_embeddings * rope_dim` bytes total. [`MAX_POSITION_EMBEDDINGS`] bounds
/// `max_position_embeddings` and [`MAX_HEAD_DIM`] bounds `rope_dim` transitively (via
/// `partial_rotary_factor <= 1.0`), but neither bounds the product, which reaches multiple GiB
/// well within both individual caps. Real Qwen3.5/3.6 presets are `max_position_embeddings =
/// 262,144` and `rope_dim` up to 128 (~134 MiB); 1 GiB (1,073,741,824) leaves roughly 8x
/// headroom above that for future architectures.
pub(crate) const MAX_ROPE_TABLE_BYTES: usize = 1_073_741_824;

/// Upper bound on `VisionModelConfig::hidden_size` accepted from `config.json`.
///
/// Only checked nonzero and overflow-guarded (via `checked_derived_sizes`'s `checked_mul`
/// chains) before this bound; a huge-but-non-overflowing value drives the `qkv_out` /
/// `mlp_intermediate` / patch-embed / position-embed tensor size derivations in
/// `checked_derived_sizes` to gigabyte-plus scale without tripping any overflow guard. Real
/// Qwen3.5-VL vision towers use ~1,280; 1,048,576 matches the [`MAX_HIDDEN_SIZE`] scale.
pub(crate) const MAX_VISION_HIDDEN_SIZE: usize = 1_048_576;

/// Upper bound on `VisionModelConfig::num_heads` accepted from `config.json`. Only checked
/// nonzero previously. Real Qwen3.5-VL vision towers use ~16; 8,192 matches the
/// [`MAX_ATTENTION_HEADS`] scale.
pub(crate) const MAX_VISION_NUM_HEADS: usize = 8_192;

/// Upper bound on `VisionModelConfig::patch_size` accepted from `config.json`. Only checked
/// nonzero previously; feeds the `patch_embed_weight` product in `checked_derived_sizes`. Real
/// Qwen3.5-VL presets use 14-32; 4,096 leaves generous headroom.
pub(crate) const MAX_VISION_PATCH_SIZE: usize = 4_096;

/// Upper bound on `VisionModelConfig::spatial_merge_size` accepted from `config.json`. Only
/// checked nonzero previously; squared in the `merger fc1` product in `checked_derived_sizes`.
/// Real Qwen3.5-VL presets use 2; 64 leaves generous headroom.
pub(crate) const MAX_VISION_SPATIAL_MERGE_SIZE: usize = 64;

/// Upper bound on `VisionModelConfig::out_hidden_size` accepted from `config.json`. Only
/// checked nonzero previously; feeds the `merger fc2` product in `checked_derived_sizes`. Real
/// Qwen3.5-VL presets match the text decoder `hidden_size`; 1,048,576 matches
/// [`MAX_HIDDEN_SIZE`].
pub(crate) const MAX_VISION_OUT_HIDDEN_SIZE: usize = 1_048_576;

/// Upper bound on `VisionModelConfig::temporal_patch_size` accepted from `config.json`. Only
/// checked nonzero previously; feeds the `patch_embed_weight` product in
/// `checked_derived_sizes`. Real Qwen3.5-VL presets use 2; 256 leaves generous headroom.
pub(crate) const MAX_VISION_TEMPORAL_PATCH_SIZE: usize = 256;

/// Upper bound on `VisionModelConfig::num_position_embeddings` accepted from `config.json`.
/// Only checked nonzero previously; feeds the `pos_embed` product (`num_position_embeddings *
/// hidden_size`) in `checked_derived_sizes`. Real Qwen3.5-VL presets use tens of thousands;
/// 16,777,216 (2^24) matches the [`MAX_GDN_STATE_SIZE`] scale.
pub(crate) const MAX_VISION_NUM_POSITION_EMBEDDINGS: usize = 16_777_216;

/// Upper bound on `VisionModelConfig::in_channels` accepted from `config.json`. Only checked
/// nonzero previously; feeds the `patch_embed_weight` product in `checked_derived_sizes`. Real
/// Qwen3.5-VL presets use 3 (RGB); 256 leaves generous headroom.
pub(crate) const MAX_VISION_IN_CHANNELS: usize = 256;

/// Upper bound, in bytes, on any single derived vision tensor
/// (`VisionModelConfig::checked_derived_sizes`) accepted from `config.json` (CLASS A3,
/// materialization site 2).
///
/// `checked_derived_sizes` was previously overflow-checked only (`checked_mul` chains
/// guarding `usize` wraparound); a huge-but-non-overflowing product (e.g. `hidden_size =
/// 1_048_576`, `spatial_merge_size = 1`, `out_hidden_size = 1_048_576`,
/// `num_position_embeddings = 16_777_216`, each individually in-budget per their own
/// per-field caps) still passes every overflow guard while requiring multi-terabyte
/// allocation for the merger / patch-embed / position-embed tensors in
/// `vision/checkpoint.rs`'s `assemble`. Real Qwen3.5-VL vision towers (hidden_size ~
/// 1,280, spatial_merge_size = 2) top out around 100 MiB per derived tensor (the merger
/// fc1/fc2 weights); 512 MiB (536,870,912) leaves roughly 5x headroom above that while
/// rejecting the multi-terabyte hostile case by several orders of magnitude.
pub(crate) const MAX_VISION_TENSOR_BYTES: u128 = 536_870_912;

/// Empty think block token sequence: `<think>\n\n</think>\n\n`.
/// Prefill this to disable chain-of-thought reasoning.
pub const QWEN3_NO_THINK_PREFIX: [u32; 6] = [
    QWEN3_THINK_OPEN_TOKEN_ID,
    QWEN3_NEWLINE_TOKEN_ID,
    QWEN3_NEWLINE_TOKEN_ID,
    QWEN3_THINK_CLOSE_TOKEN_ID,
    QWEN3_NEWLINE_TOKEN_ID,
    QWEN3_NEWLINE_TOKEN_ID,
];

/// **Unstable**: per-layer attention type selector for Qwen hybrid architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// GatedDeltaNet linear attention with recurrent state.
    LinearAttention,
    /// Standard grouped-query attention with KV cache.
    FullAttention,
}

/// **Unstable**: ViT geometry for vision-language checkpoint variants, parsed from the
/// top-level `vision_config` object in `config.json` (a sibling of `text_config`, not nested
/// inside it). `None` on [`Qwen35Config`] for text-only checkpoints.
///
/// Parsed but not yet consumed (ADR-069 stage S1) — weight loading (S2), the Metal ViT
/// forward pass, patch merger, and image-token injection land in later stages.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionModelConfig {
    /// Number of ViT transformer blocks.
    pub depth: usize,
    /// ViT hidden dimension.
    pub hidden_size: usize,
    /// ViT attention head count.
    pub num_heads: usize,
    /// Patch size in pixels (square patches).
    pub patch_size: usize,
    /// Spatial merge factor applied by the patch merger before projecting into decoder space.
    pub spatial_merge_size: usize,
    /// Merger output dimension (equals the text decoder `hidden_size` for this checkpoint).
    pub out_hidden_size: usize,
    /// Temporal patch size (frames folded per patch) for video input.
    pub temporal_patch_size: usize,
    /// Learned position-embedding table size.
    pub num_position_embeddings: usize,
    /// Input image channel count.
    pub in_channels: usize,
    /// DeepStack visual layer indexes; empty for checkpoints that don't use DeepStack fusion.
    #[serde(default, deserialize_with = "deserialize_deepstack_visual_indexes")]
    pub deepstack_visual_indexes: Vec<usize>,
}

impl VisionModelConfig {
    /// Validate structural invariants before this config is used to derive expected
    /// tensor shapes. A present-but-malformed `vision_config` (e.g. `depth: 0` or
    /// `num_heads: 0`) is syntactically valid JSON and would otherwise load a subset
    /// of `model.visual.*` tensors (or none) without any error — this must fail
    /// closed both here (parse boundary) and again at the `load_qwen35_vision_weights`
    /// public boundary, since callers can construct this struct directly.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.depth == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: depth must be > 0".to_string(),
            ));
        }
        // The vision checkpoint loader mints ~12 tensor-name Strings per ViT block *before*
        // any tensor validation runs, so an unbounded depth drives unbounded String
        // allocation ahead of any shape check. See `MAX_VISION_DEPTH` docs.
        if self.depth > MAX_VISION_DEPTH {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: depth ({}) exceeds MAX_VISION_DEPTH ({MAX_VISION_DEPTH})",
                self.depth
            )));
        }
        if self.hidden_size == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: hidden_size must be > 0".to_string(),
            ));
        }
        // Only overflow-guarded (via `checked_derived_sizes`'s `checked_mul` chains) before
        // this bound; a huge-but-non-overflowing value reaches those tensor-size derivations
        // unchecked. See `MAX_VISION_HIDDEN_SIZE` docs.
        if self.hidden_size > MAX_VISION_HIDDEN_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: hidden_size ({}) exceeds MAX_VISION_HIDDEN_SIZE \
                 ({MAX_VISION_HIDDEN_SIZE})",
                self.hidden_size
            )));
        }
        if self.num_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: num_heads must be > 0".to_string(),
            ));
        }
        if self.num_heads > MAX_VISION_NUM_HEADS {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: num_heads ({}) exceeds MAX_VISION_NUM_HEADS \
                 ({MAX_VISION_NUM_HEADS})",
                self.num_heads
            )));
        }
        if self.patch_size == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: patch_size must be > 0".to_string(),
            ));
        }
        if self.patch_size > MAX_VISION_PATCH_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: patch_size ({}) exceeds MAX_VISION_PATCH_SIZE \
                 ({MAX_VISION_PATCH_SIZE})",
                self.patch_size
            )));
        }
        if self.spatial_merge_size == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: spatial_merge_size must be > 0".to_string(),
            ));
        }
        if self.spatial_merge_size > MAX_VISION_SPATIAL_MERGE_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: spatial_merge_size ({}) exceeds \
                 MAX_VISION_SPATIAL_MERGE_SIZE ({MAX_VISION_SPATIAL_MERGE_SIZE})",
                self.spatial_merge_size
            )));
        }
        if self.out_hidden_size == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: out_hidden_size must be > 0".to_string(),
            ));
        }
        if self.out_hidden_size > MAX_VISION_OUT_HIDDEN_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: out_hidden_size ({}) exceeds \
                 MAX_VISION_OUT_HIDDEN_SIZE ({MAX_VISION_OUT_HIDDEN_SIZE})",
                self.out_hidden_size
            )));
        }
        if self.temporal_patch_size == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: temporal_patch_size must be > 0".to_string(),
            ));
        }
        if self.temporal_patch_size > MAX_VISION_TEMPORAL_PATCH_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: temporal_patch_size ({}) exceeds \
                 MAX_VISION_TEMPORAL_PATCH_SIZE ({MAX_VISION_TEMPORAL_PATCH_SIZE})",
                self.temporal_patch_size
            )));
        }
        if self.num_position_embeddings == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: num_position_embeddings must be > 0".to_string(),
            ));
        }
        if self.num_position_embeddings > MAX_VISION_NUM_POSITION_EMBEDDINGS {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: num_position_embeddings ({}) exceeds \
                 MAX_VISION_NUM_POSITION_EMBEDDINGS ({MAX_VISION_NUM_POSITION_EMBEDDINGS})",
                self.num_position_embeddings
            )));
        }
        if self.in_channels == 0 {
            return Err(InferenceError::Inference(
                "invalid vision_config: in_channels must be > 0".to_string(),
            ));
        }
        if self.in_channels > MAX_VISION_IN_CHANNELS {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: in_channels ({}) exceeds MAX_VISION_IN_CHANNELS \
                 ({MAX_VISION_IN_CHANNELS})",
                self.in_channels
            )));
        }
        // `build_pos_embed_and_rope_tables` (`vision/qwen35_vit.rs`) derives
        // `side = (num_position_embeddings as f64).sqrt().round() as usize` and then indexes
        // `pos_embed[(ch * side + cw) * hidden .. ]` for `ch, cw` up to `side - 1` -- i.e. it
        // assumes a `side * side` grid. A non-square `num_position_embeddings` (in-budget per
        // the `MAX_VISION_NUM_POSITION_EMBEDDINGS` check above, but geometrically invalid) makes
        // `side * side != num_position_embeddings`: when `side * side` exceeds the true table
        // length, `row_idx` can index past the `pos_embed` slice's end and panic; this is
        // dimension (B) (semantic validity), not (A) (allocation) -- the field's own product
        // budget passes trivially either way.
        let side = (self.num_position_embeddings as f64).sqrt().round() as usize;
        if side * side != self.num_position_embeddings {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: num_position_embeddings ({}) must be a perfect square \
                 (nearest side {side} squares to {})",
                self.num_position_embeddings,
                side * side
            )));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size, self.num_heads
            )));
        }
        for &idx in &self.deepstack_visual_indexes {
            if idx >= self.depth {
                return Err(InferenceError::Inference(format!(
                    "invalid vision_config: deepstack_visual_indexes entry {idx} is out of \
                     range for depth {}",
                    self.depth
                )));
            }
        }
        self.checked_derived_sizes()?;
        Ok(())
    }

    /// Checked arithmetic for the tensor-shape sizes the checkpoint loader derives from
    /// this config, so a pathological combination of large fields overflows into a typed
    /// error here rather than silently wrapping into an undersized allocation downstream.
    fn checked_derived_sizes(&self) -> Result<(), InferenceError> {
        let overflow = || {
            InferenceError::Inference(
                "invalid vision_config: a derived tensor size overflows usize".to_string(),
            )
        };
        // CLASS A3 (materialization site 2): each derived product below was previously
        // overflow-checked only; a huge-but-non-overflowing product still reaches
        // `vision/checkpoint.rs`'s `assemble` unbounded. Budget each product's byte size
        // (elements * 4) against `MAX_VISION_TENSOR_BYTES`, in addition to the existing
        // overflow guard, so the check runs before any tensor is materialized.
        let budget = |elems: usize, what: &str| -> Result<(), InferenceError> {
            let bytes = elems as u128 * 4;
            if bytes > MAX_VISION_TENSOR_BYTES {
                return Err(InferenceError::Inference(format!(
                    "invalid vision_config: derived tensor {what} ({bytes} bytes) exceeds \
                     MAX_VISION_TENSOR_BYTES ({MAX_VISION_TENSOR_BYTES})"
                )));
            }
            Ok(())
        };
        let qkv_out = self.hidden_size.checked_mul(3).ok_or_else(overflow)?;
        budget(qkv_out, "qkv_out")?;
        let mlp_intermediate = self.hidden_size.checked_mul(4).ok_or_else(overflow)?;
        budget(mlp_intermediate, "mlp_intermediate")?;
        let merge_in = self
            .spatial_merge_size
            .checked_mul(self.spatial_merge_size)
            .and_then(|sq| sq.checked_mul(self.hidden_size))
            .ok_or_else(overflow)?;
        let merger_fc1 = merge_in.checked_mul(merge_in).ok_or_else(overflow)?;
        budget(merger_fc1, "merger_fc1")?;
        let merger_fc2 = self
            .out_hidden_size
            .checked_mul(merge_in)
            .ok_or_else(overflow)?;
        budget(merger_fc2, "merger_fc2")?;
        let patch_embed_weight = self
            .hidden_size
            .checked_mul(self.in_channels)
            .and_then(|v| v.checked_mul(self.temporal_patch_size))
            .and_then(|v| v.checked_mul(self.patch_size))
            .and_then(|v| v.checked_mul(self.patch_size))
            .ok_or_else(overflow)?;
        budget(patch_embed_weight, "patch_embed_weight")?;
        let pos_embed = self
            .num_position_embeddings
            .checked_mul(self.hidden_size)
            .ok_or_else(overflow)?;
        budget(pos_embed, "pos_embed")?;
        Ok(())
    }
}

/// **Unstable**: Qwen hybrid attention model configuration; fields evolving with model variants.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct Qwen35Config {
    // --- Core dimensions ---
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,

    // --- Full attention config (GQA) ---
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    /// Fraction of head_dim that gets RoPE applied (0.25 = first 64 of 256 dims).
    pub partial_rotary_factor: f32,
    /// Nested RoPE config (27B+ models store rope_theta here instead of flat).
    #[serde(default)]
    pub rope_parameters: Option<RopeParams>,

    // --- Linear attention config (GatedDeltaNet) ---
    pub linear_num_key_heads: usize,
    /// None means use the method `linear_num_value_heads()` default.
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,

    // --- MoE config ---
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    pub output_router_logits: bool,
    #[serde(default)]
    pub router_aux_loss_coef: Option<f32>,

    // --- Embedding/output projection ---
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    // --- MTP (Multi-Token Prediction) config ---
    /// Number of MTP transformer layers (0 = no MTP).
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    /// Whether the MTP module uses dedicated embeddings separate from the main model.
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
    /// Rotation seed used during QuaRot conversion. `None` for non-QuaRot artifacts.
    /// Used at runtime to reconstruct `RandomizedHadamard` for MTP counter-rotation.
    #[serde(default)]
    pub quarot_rotation_seed: Option<u64>,

    // --- Layer pattern ---
    /// Every Nth layer is full attention (4 = [lin, lin, lin, full]).
    pub full_attention_interval: usize,
    /// Precomputed per-layer type, length = num_hidden_layers.
    #[serde(deserialize_with = "deserialize_bounded_vec")]
    pub layer_types: Vec<LayerType>,
    /// Per-layer active mask; `true` = active, `false` = pruned (identity skip).
    /// Length must equal `num_hidden_layers`. Defaults to all-true (no pruning).
    #[serde(default, deserialize_with = "deserialize_bounded_vec")]
    pub layer_mask: Vec<bool>,

    // --- Generation ---
    pub eos_token_id: u32,
    /// Model's native context window. Missing from a `config.json` is
    /// representable (defaults to 4096) so callers deriving the serve
    /// context clamp (#551) can distinguish "no real value present" from a
    /// present-but-small value rather than silently inheriting an unrelated
    /// preset's context length.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    // --- Vision-language extension (ADR-069 S1) ---
    /// ViT geometry from the top-level `vision_config` object; `None` for text-only
    /// checkpoints. Parsed but not yet consumed by the forward pass (S2 loads the weights;
    /// S3+ wires them into the decode path).
    #[serde(default)]
    pub vision_config: Option<VisionModelConfig>,
    /// Placeholder token id marking an image-patch span in the input sequence.
    #[serde(default)]
    pub image_token_id: Option<u32>,
    /// Placeholder token id marking a video-frame span in the input sequence.
    #[serde(default)]
    pub video_token_id: Option<u32>,
    /// Token id opening a vision content span (wraps image/video token runs).
    #[serde(default)]
    pub vision_start_token_id: Option<u32>,
    /// Token id closing a vision content span.
    #[serde(default)]
    pub vision_end_token_id: Option<u32>,
}

/// Deserializes a JSON array into `Vec<T>`, erroring once more than `MAX_HIDDEN_LAYERS`
/// elements have been pulled from the input. `layer_types` and `layer_mask` are per-layer
/// arrays that a valid config sizes to `num_hidden_layers <= MAX_HIDDEN_LAYERS`, but the
/// `MAX_HIDDEN_LAYERS` guard in `from_config_json_str` runs *after* serde has already
/// deserialized these fields — an attacker-controlled config.json carrying an
/// oversized array would otherwise allocate it in full before that guard ever runs. Erring
/// at `MAX_HIDDEN_LAYERS + 1` elements instead of allocating and checking `.len()` afterward
/// is what keeps the Vec itself from ever growing past the cap.
fn deserialize_bounded_vec<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    struct BoundedVecVisitor<T>(std::marker::PhantomData<T>);

    impl<'de, T> serde::de::Visitor<'de> for BoundedVecVisitor<T>
    where
        T: serde::Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "a sequence of at most {MAX_HIDDEN_LAYERS} elements")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while vec.len() < MAX_HIDDEN_LAYERS {
                match seq.next_element()? {
                    Some(elem) => vec.push(elem),
                    None => return Ok(vec),
                }
            }
            if seq.next_element::<T>()?.is_some() {
                return Err(serde::de::Error::custom(format!(
                    "sequence exceeds MAX_HIDDEN_LAYERS ({MAX_HIDDEN_LAYERS}) elements"
                )));
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_seq(BoundedVecVisitor(std::marker::PhantomData))
}

/// Deserializes a JSON array into `Vec<T>`, erroring once more than `MAX_CONFIG_VECTOR_LEN`
/// elements have been pulled from the input. Used for `RopeParams::mrope_section` and
/// `VisionModelConfig::deepstack_visual_indexes` -- both are small fixed-purpose vectors
/// deserialized before `from_config_json_str`'s validation pass runs, so a post-parse `.len()`
/// check does not prevent the allocation itself; erring during the `visit_seq` pull, like
/// [`deserialize_bounded_vec`] does for `MAX_HIDDEN_LAYERS`, is what keeps the Vec from ever
/// growing past the cap. `field_name` is threaded through so the resulting error names the
/// specific config field, since both callers of this helper share one generic implementation.
fn deserialize_config_vector<'de, D, T>(
    deserializer: D,
    field_name: &'static str,
) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    struct BoundedVecVisitor<T> {
        field_name: &'static str,
        marker: std::marker::PhantomData<T>,
    }

    impl<'de, T> serde::de::Visitor<'de> for BoundedVecVisitor<T>
    where
        T: serde::Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "a sequence of at most {MAX_CONFIG_VECTOR_LEN} elements for {}",
                self.field_name
            )
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while vec.len() < MAX_CONFIG_VECTOR_LEN {
                match seq.next_element()? {
                    Some(elem) => vec.push(elem),
                    None => return Ok(vec),
                }
            }
            if seq.next_element::<T>()?.is_some() {
                return Err(serde::de::Error::custom(format!(
                    "{} exceeds MAX_CONFIG_VECTOR_LEN ({MAX_CONFIG_VECTOR_LEN}) elements",
                    self.field_name
                )));
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_seq(BoundedVecVisitor {
        field_name,
        marker: std::marker::PhantomData,
    })
}

/// `serde(deserialize_with)` entry point for `VisionModelConfig::deepstack_visual_indexes`;
/// see [`deserialize_config_vector`].
fn deserialize_deepstack_visual_indexes<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserialize_config_vector(deserializer, "deepstack_visual_indexes")
}

/// `serde(deserialize_with)` entry point for `RopeParams::mrope_section`. Wraps
/// [`deserialize_config_vector`] to additionally handle the `Option` layer -- `mrope_section`
/// is absent from most checkpoints -- so the bound is enforced without disturbing the
/// existing `#[serde(default)]` behavior for a missing field.
fn deserialize_mrope_section<'de, D>(deserializer: D) -> Result<Option<Vec<usize>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct OptionVisitor;

    impl<'de> serde::de::Visitor<'de> for OptionVisitor {
        type Value = Option<Vec<usize>>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "an optional sequence of at most {MAX_CONFIG_VECTOR_LEN} elements for \
                 mrope_section"
            )
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D2>(self, deserializer: D2) -> Result<Self::Value, D2::Error>
        where
            D2: serde::Deserializer<'de>,
        {
            deserialize_config_vector(deserializer, "mrope_section").map(Some)
        }
    }

    deserializer.deserialize_option(OptionVisitor)
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_max_position_embeddings() -> usize {
    4096
}

// Nested rope_parameters in HF config.json (many models nest rope_theta and
// partial_rotary_factor here instead of at the top level of text_config).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct RopeParams {
    #[serde(default)]
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    /// Per-axis (temporal, height, width) M-RoPE section sizes, e.g. `[11, 11, 10]`.
    /// `None` when the checkpoint has no vision M-RoPE config (text-only decoders use plain
    /// 1-D partial RoPE). Parsed but not yet consumed by the forward pass (ADR-069 S1; wired
    /// in S3+).
    #[serde(default, deserialize_with = "deserialize_mrope_section")]
    pub mrope_section: Option<Vec<usize>>,
    /// Whether M-RoPE frequencies are interleaved `[T, H, W, T, H, W, ...]` (true for
    /// Qwen3.5) rather than block-concatenated. `None` when absent (text-only decoders).
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
}

// Private helper for HF config.json structure (outer wrapper with text_config). The
// vision-language fields are siblings of text_config at the top level of config.json, not
// nested inside it, so they are threaded onto the parsed Qwen35Config in from_config_json_str
// the same way tie_word_embeddings already is.
#[derive(Debug, serde::Deserialize)]
struct HfQwenConfigFile {
    #[serde(default)]
    text_config: Option<Qwen35Config>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    vision_config: Option<VisionModelConfig>,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default)]
    video_token_id: Option<u32>,
    #[serde(default)]
    vision_start_token_id: Option<u32>,
    #[serde(default)]
    vision_end_token_id: Option<u32>,
}

impl Default for Qwen35Config {
    fn default() -> Self {
        Self::qwen36_35b_a3b()
    }
}

/// A [`Qwen35Config`] that has passed [`Qwen35Config::validate`]'s full bounds/structural
/// validation (CLASS C, the ingress-enumeration fix). The only way to construct one is
/// [`Qwen35Config::validate`] (or the `TryFrom<Qwen35Config>` impl below, which wraps it) --
/// there is no public constructor that skips validation. Non-Metal loaders and session
/// constructors (`model/qwen35/loading.rs`, `model/qwen35/model.rs`) require this type
/// rather than a raw `Qwen35Config`, so a config built via direct `serde` deserialization
/// (which still lands on the raw, unvalidated `Qwen35Config` -- serde needs a type it can
/// construct field-by-field) cannot reach them without going through this checked
/// conversion first. `Deref` exposes read-only field access so call sites that only read
/// config fields don't need `.into_inner()` or `&*validated`. See the CLASS C ingress table
/// in the PR body for every entry point and its validation status.
#[derive(Debug, Clone)]
pub struct ValidatedQwen35Config(Qwen35Config);

impl std::ops::Deref for ValidatedQwen35Config {
    type Target = Qwen35Config;
    fn deref(&self) -> &Qwen35Config {
        &self.0
    }
}

impl ValidatedQwen35Config {
    /// Unwrap back into the raw `Qwen35Config`. Used by [`Qwen35Config::from_config_json_str`]
    /// (and its `from_config_json` / `from_model_dir` siblings) to keep their return type
    /// stable for existing callers while still running full validation underneath.
    pub fn into_inner(self) -> Qwen35Config {
        self.0
    }
}

impl TryFrom<Qwen35Config> for ValidatedQwen35Config {
    type Error = InferenceError;

    /// The checked conversion CLASS C requires: a raw `Qwen35Config` (e.g. one built via
    /// direct `serde_json::from_str::<Qwen35Config>`, which bypasses the `HfQwenConfigFile`
    /// wrapper and its `text_config` nesting) can only become a `ValidatedQwen35Config` by
    /// running the exact same [`Qwen35Config::validate`] bounds/structural checks that
    /// `from_config_json_str` runs.
    fn try_from(cfg: Qwen35Config) -> Result<Self, InferenceError> {
        cfg.validate()
    }
}

impl Qwen35Config {
    /// **Unstable**: default Qwen3.5-2B configuration; may change as model checkpoints update.
    pub fn qwen35_2b() -> Self {
        let num_hidden_layers = 24;
        let full_attention_interval = 4;
        let layer_types = compute_layer_types(num_hidden_layers, full_attention_interval);

        Self {
            hidden_size: 2048,
            num_hidden_layers,
            vocab_size: 248_320,
            intermediate_size: 6144,
            rms_norm_eps: 1e-6,
            // Full attention
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            // Linear attention (GatedDeltaNet)
            linear_num_key_heads: 16,
            linear_num_value_heads: Some(16),
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            // MoE (absent in Qwen3.5)
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            // Projection
            tie_word_embeddings: true,
            // Layer pattern
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            // Generation
            eos_token_id: 248_044,
            max_position_embeddings: 262_144,
            // MTP (absent in Qwen3.5)
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            // Vision-language extension (this preset is text-only, no vision tower)
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        }
    }

    /// **Unstable**: Qwen3.5-0.8B configuration. Same hybrid architecture as the
    /// 2B (24 layers, `[linear, linear, linear, full] x 6`), scaled down. The
    /// released checkpoint is a vision-language model; this is its text decoder.
    pub fn qwen35_0_8b() -> Self {
        let num_hidden_layers = 24;
        let full_attention_interval = 4;
        let layer_types = compute_layer_types(num_hidden_layers, full_attention_interval);

        Self {
            hidden_size: 1024,
            num_hidden_layers,
            vocab_size: 248_320,
            intermediate_size: 3584,
            rms_norm_eps: 1e-6,
            // Full attention
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            // Linear attention (GatedDeltaNet)
            linear_num_key_heads: 16,
            linear_num_value_heads: Some(16),
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            // MoE (absent — 0.8B is dense)
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            // Projection
            tie_word_embeddings: true,
            // Layer pattern
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            // Generation
            eos_token_id: 248_044,
            max_position_embeddings: 262_144,
            // MTP: Qwen3.5-0.8B ships 1 MTP layer
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            // Vision-language extension (this preset is text-only, no vision tower)
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        }
    }

    /// **Unstable**: Qwen3.6-35B-A3B text configuration defaults from HF `text_config`.
    pub fn qwen36_35b_a3b() -> Self {
        let num_hidden_layers = 40;
        let full_attention_interval = 4;
        let layer_types = compute_layer_types(num_hidden_layers, full_attention_interval);

        Self {
            hidden_size: 2048,
            num_hidden_layers,
            vocab_size: 248_320,
            intermediate_size: 6144,
            rms_norm_eps: 1e-6,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: 16,
            linear_num_value_heads: Some(32),
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            num_experts: Some(256),
            num_experts_per_tok: Some(8),
            moe_intermediate_size: Some(512),
            shared_expert_intermediate_size: Some(512),
            output_router_logits: false,
            router_aux_loss_coef: Some(0.001),
            tie_word_embeddings: false,
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            eos_token_id: 248_044,
            max_position_embeddings: 262_144,
            // MTP: Qwen3.6 has 1 MTP layer
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            // Vision-language extension (this preset is text-only, no vision tower)
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        }
    }

    /// **Unstable**: Qwen3.6-27B dense configuration; fields from HF `text_config`.
    pub fn qwen36_27b() -> Self {
        let num_hidden_layers = 64;
        let full_attention_interval = 4;
        let layer_types = compute_layer_types(num_hidden_layers, full_attention_interval);

        Self {
            hidden_size: 5120,
            num_hidden_layers,
            vocab_size: 248_320,
            intermediate_size: 17408,
            rms_norm_eps: 1e-6,
            // Full attention (GA)
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            // rope_theta is nested under rope_parameters.rope_theta in config.json and not
            // directly deserializable; hardcode the value from rope_parameters.rope_theta.
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            // Linear attention (GatedDeltaNet)
            linear_num_key_heads: 16,
            linear_num_value_heads: Some(48),
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            // MoE (absent — 27B is dense)
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            // Projection
            tie_word_embeddings: false,
            // Layer pattern
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            // Generation
            eos_token_id: 248_044,
            max_position_embeddings: 262_144,
            // MTP: Qwen3.6 has 1 MTP layer
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            // Vision-language extension (this preset is text-only, no vision tower)
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        }
    }

    /// Parse a HF config.json (which may wrap fields inside `text_config`).
    pub fn from_config_json(path: &Path) -> Result<Self, InferenceError> {
        let json = std::fs::read_to_string(path).map_err(InferenceError::Io)?;
        Self::from_config_json_str(&json)
    }

    /// Parse a HF config.json into a validated [`ValidatedQwen35Config`]. See
    /// [`Self::from_config_json`] for the raw-`Qwen35Config`-returning sibling.
    pub fn from_config_json_validated(
        path: &Path,
    ) -> Result<ValidatedQwen35Config, InferenceError> {
        let json = std::fs::read_to_string(path).map_err(InferenceError::Io)?;
        Self::from_config_json_str_validated(&json)
    }

    /// Resolve the architecture config for a model directory, requiring a
    /// real `config.json`.
    ///
    /// This is the single fallback policy for every loader in this crate
    /// (library and binaries alike; see issue #923). Every supported Qwen
    /// checkpoint ships a `config.json` describing its exact geometry, so a
    /// missing file is a hard, descriptive error naming the directory rather
    /// than a silently-substituted architecture preset. Before this helper
    /// existed, independent call sites each guessed a different preset
    /// (`qwen35_2b`, `qwen36_27b`, `qwen35_0_8b`) on a missing file — pointing
    /// the same config-less directory at different tools silently produced
    /// different model geometries, and the wrong-preset case failed later at
    /// tensor validation with an error that named the weights, not the
    /// missing config. Callers that need directory-not-found context in a
    /// different error type (e.g. the CLI binaries' `String` errors) should
    /// wrap this with `.map_err(...)`, not reimplement the existence check.
    pub fn from_model_dir(dir: &Path) -> Result<Self, InferenceError> {
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(InferenceError::ModelNotFound(format!(
                "missing config.json in {} -- every supported Qwen checkpoint ships one; \
                 no architecture preset is inferred from a config-less directory",
                dir.display()
            )));
        }
        Self::from_config_json(&config_path)
    }

    /// [`Self::from_model_dir`]'s validated sibling; see [`Self::from_config_json_validated`].
    pub fn from_model_dir_validated(dir: &Path) -> Result<ValidatedQwen35Config, InferenceError> {
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(InferenceError::ModelNotFound(format!(
                "missing config.json in {} -- every supported Qwen checkpoint ships one; \
                 no architecture preset is inferred from a config-less directory",
                dir.display()
            )));
        }
        Self::from_config_json_validated(&config_path)
    }

    /// Parse HF config.json text into a validated [`ValidatedQwen35Config`]. See
    /// [`Self::from_config_json_str`] for the raw-`Qwen35Config`-returning sibling kept
    /// stable for existing callers (CLASS C ingress table, PR body).
    pub fn from_config_json_str_validated(
        json: &str,
    ) -> Result<ValidatedQwen35Config, InferenceError> {
        let parsed: HfQwenConfigFile = serde_json::from_str(json)
            .map_err(|e| InferenceError::Inference(format!("invalid Qwen config.json: {e}")))?;
        let mut cfg = parsed
            .text_config
            .unwrap_or_else(Qwen35Config::qwen36_35b_a3b);

        if let Some(tie) = parsed.tie_word_embeddings {
            cfg.tie_word_embeddings = tie;
        }
        // Vision-language fields (ADR-069 S1) are siblings of text_config at the top level of
        // config.json, not nested inside it — thread them onto cfg the same way as
        // tie_word_embeddings above. Parsed but not yet consumed by the forward pass.
        cfg.vision_config = parsed.vision_config;
        cfg.image_token_id = parsed.image_token_id;
        cfg.video_token_id = parsed.video_token_id;
        cfg.vision_start_token_id = parsed.vision_start_token_id;
        cfg.vision_end_token_id = parsed.vision_end_token_id;
        // Many models nest rope_theta and partial_rotary_factor under rope_parameters
        // instead of at the text_config level — extract when the flat fields are unset.
        if let Some(rp) = &cfg.rope_parameters {
            if cfg.rope_theta == 0.0 && rp.rope_theta > 0.0 {
                cfg.rope_theta = rp.rope_theta;
            }
            if let Some(prf) = rp.partial_rotary_factor {
                cfg.partial_rotary_factor = prf;
            }
        }
        cfg.validate()
    }

    /// Parse HF config.json text into a `Qwen35Config`. Stable signature kept for existing
    /// callers (e.g. `forward/metal_qwen35.rs`'s `from_q4_dir`, deferred to the #1037
    /// rebase per the CLASS C ingress table in the PR body); runs the identical validation
    /// as [`Self::from_config_json_str_validated`], just unwrapping the newtype.
    pub fn from_config_json_str(json: &str) -> Result<Self, InferenceError> {
        Self::from_config_json_str_validated(json).map(ValidatedQwen35Config::into_inner)
    }

    /// Run this config's full bounds/structural validation, consuming it and returning the
    /// only way to construct a [`ValidatedQwen35Config`]. Applies the same checks whether
    /// this config came from `config.json` (via [`Self::from_config_json_str_validated`])
    /// or was constructed directly (e.g. via `serde_json::from_str::<Qwen35Config>` or a
    /// preset mutated in place) -- see the CLASS C ingress table in the PR body.
    pub fn validate(self) -> Result<ValidatedQwen35Config, InferenceError> {
        let mut cfg = self;
        // Bound `num_hidden_layers` before any layer-proportional allocation runs. This
        // must precede the `compute_layer_types` call and `normalize_layer_mask` below,
        // and the loader's `Vec::with_capacity(cfg.num_hidden_layers)` (which only ever
        // runs against a config that passed this check) -- an unbounded value would
        // otherwise reach those allocations first. See `MAX_HIDDEN_LAYERS` docs.
        if cfg.num_hidden_layers > MAX_HIDDEN_LAYERS {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: num_hidden_layers ({}) exceeds MAX_HIDDEN_LAYERS \
                 ({MAX_HIDDEN_LAYERS})",
                cfg.num_hidden_layers
            )));
        }
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            // compute_layer_types uses `(i + 1) % interval`; a zero interval (from an
            // explicit `"full_attention_interval": 0`, or the container `#[serde(default)]`
            // fallback when a preset's interval is zero) would panic with a remainder
            // divide-by-zero. Malformed input must surface as a typed error, not a panic.
            if cfg.full_attention_interval == 0 {
                return Err(InferenceError::Inference(
                    "invalid Qwen config.json: full_attention_interval must be > 0 when \
                     layer_types is absent or its length differs from num_hidden_layers"
                        .to_string(),
                ));
            }
            cfg.layer_types =
                compute_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
        }
        cfg.normalize_layer_mask();

        // Structural invariants. A parseable-but-malformed config.json can set these to zero
        // or inconsistent values that survive serde yet cause a downstream divide-by-zero,
        // out-of-bounds index, or unsigned underflow at model construction / forward (e.g.
        // `num_q_heads / num_kv_heads` in the GQA path, `head_vec[rope_dim / 2 + i]` in
        // partial RoPE, `linear_conv_kernel_dim - 1` in the GatedDeltaNet conv buffer).
        // Surface them as typed errors at this single load-time choke point rather than as a
        // panic deep in the forward pass. Presets satisfy all of these by construction.
        //
        // `hidden_size == 0` combined with an unbounded `vocab_size` lets a checkpoint declare
        // `embed_tokens` as `[huge_vocab, 0]` — a zero-byte tensor that passes shape checks
        // (0 columns, so no actual data to validate) and then drives an enormous `logits`
        // allocation (`resize(&mut self.logits, cfg.vocab_size)`) on first inference.
        if cfg.hidden_size == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: hidden_size must be > 0".to_string(),
            ));
        }
        // `hidden_size` drives the `hidden` / `residual` / `attn_out` / `ffn_out` /
        // `expert_out` scratch buffers and the `max_q8_input` derivation in
        // `ForwardScratch::ensure_capacity`, unconditionally on every forward pass. See
        // `MAX_HIDDEN_SIZE` docs.
        if cfg.hidden_size > MAX_HIDDEN_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: hidden_size ({}) exceeds MAX_HIDDEN_SIZE \
                 ({MAX_HIDDEN_SIZE})",
                cfg.hidden_size
            )));
        }
        if cfg.vocab_size == 0 || cfg.vocab_size > MAX_VOCAB_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: vocab_size ({}) must be > 0 and <= MAX_VOCAB_SIZE \
                 ({MAX_VOCAB_SIZE})",
                cfg.vocab_size
            )));
        }
        // CLASS A3 (materialization site 1): `load_weights` (`model/qwen35/loading.rs:434`)
        // materializes `embed_tokens` as an owned `Vec<f32>` of `vocab_size * hidden_size`
        // elements. `MAX_VOCAB_SIZE` and `MAX_HIDDEN_SIZE` bound each factor individually,
        // but not their product -- see `MAX_EMBEDDING_BYTES` docs.
        let embedding_bytes = cfg.vocab_size as u128 * cfg.hidden_size as u128 * 4;
        if embedding_bytes > MAX_EMBEDDING_BYTES {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: embedding tensor size ({embedding_bytes} bytes) \
                 exceeds MAX_EMBEDDING_BYTES ({MAX_EMBEDDING_BYTES}): vocab_size \
                 ({}) * hidden_size ({})",
                cfg.vocab_size, cfg.hidden_size
            )));
        }
        // `intermediate_size` is always present and drives MLP scratch buffer allocations
        // at every forward pass. See `MAX_INTERMEDIATE_SIZE` docs. Zero is a degenerate
        // zero-width dense FFN, not a valid no-op.
        if cfg.intermediate_size == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: intermediate_size must be > 0".to_string(),
            ));
        }
        if cfg.intermediate_size > MAX_INTERMEDIATE_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: intermediate_size ({}) exceeds \
                 MAX_INTERMEDIATE_SIZE ({MAX_INTERMEDIATE_SIZE})",
                cfg.intermediate_size
            )));
        }
        if cfg.num_attention_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: num_attention_heads must be > 0".to_string(),
            ));
        }
        // `ForwardScratch::ensure_capacity` derives `scores = num_attention_heads *
        // (max_kv_len + 1)` independent of `head_dim` -- `MAX_FULL_ATTENTION_DIM` bounds the
        // `num_attention_heads * head_dim` product below, but a small `head_dim` paired with a
        // huge `num_attention_heads` can pass that product check while still inflating
        // `scores`. See `MAX_ATTENTION_HEADS` docs.
        if cfg.num_attention_heads > MAX_ATTENTION_HEADS {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: num_attention_heads ({}) exceeds \
                 MAX_ATTENTION_HEADS ({MAX_ATTENTION_HEADS})",
                cfg.num_attention_heads
            )));
        }
        if cfg.num_key_value_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: num_key_value_heads must be > 0".to_string(),
            ));
        }
        // Sibling of the `num_attention_heads` bound above; same `MAX_ATTENTION_HEADS` budget.
        if cfg.num_key_value_heads > MAX_ATTENTION_HEADS {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: num_key_value_heads ({}) exceeds \
                 MAX_ATTENTION_HEADS ({MAX_ATTENTION_HEADS})",
                cfg.num_key_value_heads
            )));
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: num_attention_heads ({}) must be divisible by \
                 num_key_value_heads ({})",
                cfg.num_attention_heads, cfg.num_key_value_heads
            )));
        }
        if cfg.head_dim == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: head_dim must be > 0".to_string(),
            ));
        }
        // A config-level budget independent of layer mix: an all-linear-attention config
        // never calls `checked_full_q_dim` (full-attention Q projection sizing), so this is
        // the only guard bounding `RopeTable::new`'s `max_seq_len * head_dim / 2` allocation.
        if cfg.head_dim > MAX_HEAD_DIM {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: head_dim ({}) exceeds MAX_HEAD_DIM ({MAX_HEAD_DIM})",
                cfg.head_dim
            )));
        }
        // A global config-level budget, independent of layer mix: an all-linear-attention
        // config never loads a full-attention layer's tensors, so it never reaches the
        // loader's per-layer full-attention dimension checks -- yet
        // `ForwardScratch::ensure_capacity` unconditionally computes `full_q_dim()`,
        // `full_kv_dim()`, and `2 * q_dim` for every config during generation. Validate the
        // same products here, globally, so an all-linear config cannot smuggle overflowing
        // attention geometry past load.
        let full_q_dim = cfg
            .num_attention_heads
            .checked_mul(cfg.head_dim)
            .ok_or_else(|| {
                InferenceError::Inference(format!(
                    "invalid Qwen config.json: full-attention q_dim overflows usize: \
                 num_attention_heads ({}) * head_dim ({})",
                    cfg.num_attention_heads, cfg.head_dim
                ))
            })?;
        full_q_dim.checked_mul(2).ok_or_else(|| {
            InferenceError::Inference(format!(
                "invalid Qwen config.json: full-attention 2*q_dim scratch multiplication \
                 overflows usize: q_dim ({full_q_dim})"
            ))
        })?;
        let full_kv_dim = cfg
            .num_key_value_heads
            .checked_mul(cfg.head_dim)
            .ok_or_else(|| {
                InferenceError::Inference(format!(
                    "invalid Qwen config.json: full-attention kv_dim overflows usize: \
                 num_key_value_heads ({}) * head_dim ({})",
                    cfg.num_key_value_heads, cfg.head_dim
                ))
            })?;
        // The `checked_mul` calls above only reject wraparound; `head_dim` is separately
        // bounded (`MAX_HEAD_DIM`), but `num_attention_heads` / `num_key_value_heads` are
        // not, so a non-overflowing product can still be allocation-hostile. See
        // `MAX_FULL_ATTENTION_DIM` docs.
        if full_q_dim > MAX_FULL_ATTENTION_DIM {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: full-attention q_dim ({full_q_dim}) exceeds \
                 MAX_FULL_ATTENTION_DIM ({MAX_FULL_ATTENTION_DIM})"
            )));
        }
        if full_kv_dim > MAX_FULL_ATTENTION_DIM {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: full-attention kv_dim ({full_kv_dim}) exceeds \
                 MAX_FULL_ATTENTION_DIM ({MAX_FULL_ATTENTION_DIM})"
            )));
        }
        if cfg.num_hidden_layers == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: num_hidden_layers must be > 0".to_string(),
            ));
        }
        if cfg.linear_conv_kernel_dim == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: linear_conv_kernel_dim must be > 0".to_string(),
            ));
        }
        // Sizes the GatedDeltaNet causal conv1d buffer (`conv_dim * (linear_conv_kernel_dim -
        // 1)` in `GatedDeltaNetState::new`) and conv1d weight tensor unconditional on layer
        // mix. See `MAX_CONV_KERNEL_DIM` docs.
        if cfg.linear_conv_kernel_dim > MAX_CONV_KERNEL_DIM {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: linear_conv_kernel_dim ({}) exceeds \
                 MAX_CONV_KERNEL_DIM ({MAX_CONV_KERNEL_DIM})",
                cfg.linear_conv_kernel_dim
            )));
        }
        // `RopeTable::new` (`rope.rs`) computes `freq = 1.0 / theta.powf(...)` for every
        // table entry; `theta <= 0.0` (including `-0.0`) or non-finite drives every `freq`
        // to `inf`/`NaN`, which propagates through `angle.cos()`/`angle.sin()` into every
        // cos/sin table entry -- silently corrupting all attention output with no error
        // signal, since neither is a panic. `linear_output_dim`-style bounds only catch
        // allocation-hostile values, not correctness-hostile ones; this is dimension (B),
        // not (A) -- `rope_theta` drives no allocation. Real Qwen3.5/3.6 presets use
        // `10_000_000.0`.
        if !(cfg.rope_theta.is_finite() && cfg.rope_theta > 0.0) {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: rope_theta ({}) must be finite and > 0.0",
                cfg.rope_theta
            )));
        }
        // `rope_dim = (head_dim * partial_rotary_factor) as usize` is rotated in place over a
        // `head_dim`-length head slice; a factor > 1.0 makes `rope_dim` exceed `head_dim` and
        // indexes `head_vec[rope_dim / 2 + i]` out of bounds. Require a finite fraction.
        if !(cfg.partial_rotary_factor.is_finite()
            && cfg.partial_rotary_factor >= 0.0
            && cfg.partial_rotary_factor <= 1.0)
        {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: partial_rotary_factor ({}) must be in [0.0, 1.0]",
                cfg.partial_rotary_factor
            )));
        }
        // rope_dim = (head_dim * partial_rotary_factor) as usize.  apply_partial_rope pairs
        // dimensions as (i, half+i) for i in 0..half, where half = rope_dim / 2.  An odd
        // rope_dim silently truncates: only 2*(rope_dim/2) dims are rotated, leaving one dim
        // inside the documented "first rope_dim dimensions" range untouched — wrong output
        // with no error signal.  rope_dim == 0 makes RopeTable::max_positions() return 0,
        // which causes every non-empty-sequence call to the capacity-guarded APIs to fail
        // instead of the intended no-op.  rope_dim > head_dim indexes head_vec[half + i] past
        // the head_dim-length slice: the partial_rotary_factor <= 1.0 check above bounds this
        // in real arithmetic, but rope_dim() casts head_dim through f32, so a head_dim above
        // f32's exact-integer range (2^24) can round UP and derive rope_dim > head_dim even at
        // factor 1.0.  Reject all three fail-closed; no-RoPE variants that need rope_dim==0
        // require an explicit dispatch path (Refs #401).
        let rope_dim = cfg.rope_dim();
        if rope_dim < 2 || !rope_dim.is_multiple_of(2) || rope_dim > cfg.head_dim {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: derived rope_dim ({rope_dim}) must be even, >= 2, \
                 and <= head_dim ({hd}) (partial_rotary_factor={prf})",
                hd = cfg.head_dim,
                prf = cfg.partial_rotary_factor,
            )));
        }
        // `build_rope_interleaved` (`forward/metal_qwen35.rs`) allocates two `Vec<f32>` of
        // `max_position_embeddings * (rope_dim / 2)` entries each (cos and sin tables) --
        // `4 * max_position_embeddings * rope_dim` bytes total. `max_position_embeddings` and
        // `rope_dim` are each individually bounded above, but not their product. Mirror the
        // exact formula (cast through `u128` so the byte-count computation itself cannot
        // overflow `usize` on the way to the bound check) rather than inventing a new
        // rounding convention. See `MAX_ROPE_TABLE_BYTES` docs.
        let rope_table_bytes = 4u128 * cfg.max_position_embeddings as u128 * rope_dim as u128;
        if rope_table_bytes > MAX_ROPE_TABLE_BYTES as u128 {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: RoPE table size ({rope_table_bytes} bytes) exceeds \
                 MAX_ROPE_TABLE_BYTES ({MAX_ROPE_TABLE_BYTES}): max_position_embeddings \
                 ({}) * rope_dim ({rope_dim})",
                cfg.max_position_embeddings
            )));
        }
        // The GatedDeltaNet fused path divides by these head counts: `value_heads / key_heads`
        // (gdn_fused.rs ratio) and `h / ratio` per value head. A parseable config with
        // `linear_num_key_heads == 0`, `linear_num_value_heads == 0`, or value-heads not a
        // positive multiple of key-heads (ratio == 0) is an integer divide-by-zero panic deep in
        // the recurrence. Real GDN configs are key=16/value=32 (ratio 2); reject the rest here.
        let key_heads = cfg.linear_num_key_heads;
        let value_heads = cfg.linear_num_value_heads();
        if key_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: linear_num_key_heads must be > 0".to_string(),
            ));
        }
        // Unlike `linear_num_value_heads()`, `linear_num_key_heads` is not implicitly bounded
        // by `MAX_LINEAR_OUTPUT_DIM` / `MAX_GDN_STATE_SIZE` (neither product carries it as a
        // factor) -- it is a free multiplier in `linear_qkv_dim()`, which sizes the
        // GatedDeltaNet QKV projection weight, conv1d weight, and conv/activation scratch
        // buffers. See `MAX_LINEAR_NUM_KEY_HEADS` docs.
        if key_heads > MAX_LINEAR_NUM_KEY_HEADS {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: linear_num_key_heads ({key_heads}) exceeds \
                 MAX_LINEAR_NUM_KEY_HEADS ({MAX_LINEAR_NUM_KEY_HEADS})"
            )));
        }
        if value_heads == 0 || !value_heads.is_multiple_of(key_heads) {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: linear_num_value_heads ({value_heads}) must be a \
                 positive multiple of linear_num_key_heads ({key_heads})"
            )));
        }
        // Sibling of the `linear_num_key_heads` bound above; `value_heads` is only
        // implicitly bounded by `MAX_LINEAR_OUTPUT_DIM` / `MAX_GDN_STATE_SIZE` (both
        // product budgets that carry it as a factor alongside a head-dim field that can be
        // pinned at 1), so it needs its own direct cap the same way `linear_num_key_heads`
        // does. See `MAX_LINEAR_NUM_VALUE_HEADS` docs.
        if value_heads > MAX_LINEAR_NUM_VALUE_HEADS {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: linear_num_value_heads ({value_heads}) exceeds \
                 MAX_LINEAR_NUM_VALUE_HEADS ({MAX_LINEAR_NUM_VALUE_HEADS})"
            )));
        }
        // `linear_key_head_dim` / `linear_value_head_dim` size the GatedDeltaNet state matrix
        // (`s_matrices = vec![0.0; value_heads * key_dim * value_dim]` in `attention/gdn.rs`);
        // zero in either dimension is a degenerate zero-width recurrence rather than a valid
        // no-op. `linear_value_head_dim == 0` also collapses `linear_output_dim()` to zero,
        // which passes the `MAX_LINEAR_OUTPUT_DIM` check below trivially, so that guard alone
        // does not catch this case.
        if cfg.linear_key_head_dim == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: linear_key_head_dim must be > 0".to_string(),
            ));
        }
        if cfg.linear_value_head_dim == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: linear_value_head_dim must be > 0".to_string(),
            ));
        }
        // A global config-level budget, independent of layer mix: an all-full-attention config
        // never reaches `load_linear_attention_weights`'s per-layer `checked_linear_output_dim`
        // derivation, yet `ForwardScratch::ensure_capacity` unconditionally derives
        // `max_q8_input` from `linear_output_dim()` and resizes `x_q_scratch` for every config
        // regardless of whether any layer is actually linear-attention. Validate the same
        // checked product here, globally, so an all-full-attention config cannot smuggle an
        // overflowing or allocation-hostile linear-attention output dimension past load. See
        // `MAX_LINEAR_OUTPUT_DIM` docs.
        let linear_output_dim = value_heads
            .checked_mul(cfg.linear_value_head_dim)
            .ok_or_else(|| {
                InferenceError::Inference(format!(
                    "invalid Qwen config.json: linear attention output_dim overflows usize: \
                 linear_num_value_heads ({value_heads}) * linear_value_head_dim ({})",
                    cfg.linear_value_head_dim
                ))
            })?;
        if linear_output_dim > MAX_LINEAR_OUTPUT_DIM {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: linear attention output_dim ({linear_output_dim}) \
                 exceeds MAX_LINEAR_OUTPUT_DIM ({MAX_LINEAR_OUTPUT_DIM})"
            )));
        }
        // `MAX_LINEAR_OUTPUT_DIM` above only bounds the `value_heads * linear_value_head_dim`
        // factor; `linear_key_head_dim` is a free multiplier on top of it in the GatedDeltaNet
        // state matrix (`s_matrices = vec![0.0; value_heads * key_dim * value_dim]` in
        // `attention/gdn.rs`) -- a config with `value_heads = 1`, `linear_value_head_dim` near
        // the `MAX_LINEAR_OUTPUT_DIM` cap, and a huge `linear_key_head_dim` passes both guards
        // above while allocating on the order of terabytes at first generation. Validate the
        // full three-factor product globally, independent of layer mix, for the same reason as
        // the `linear_output_dim` budget above. See `MAX_GDN_STATE_SIZE` docs.
        let gdn_state_size = value_heads
            .checked_mul(cfg.linear_key_head_dim)
            .and_then(|v| v.checked_mul(cfg.linear_value_head_dim))
            .ok_or_else(|| {
                InferenceError::Inference(format!(
                    "invalid Qwen config.json: GatedDeltaNet state size overflows usize: \
                 linear_num_value_heads ({value_heads}) * linear_key_head_dim ({}) * \
                 linear_value_head_dim ({})",
                    cfg.linear_key_head_dim, cfg.linear_value_head_dim
                ))
            })?;
        if gdn_state_size > MAX_GDN_STATE_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: GatedDeltaNet state size ({gdn_state_size}) exceeds \
                 MAX_GDN_STATE_SIZE ({MAX_GDN_STATE_SIZE})"
            )));
        }
        // `MAX_LINEAR_NUM_VALUE_HEADS` bounds `value_heads` alone, but
        // `new_session_inner`'s GatedDeltaNet chunk-scratch buffers (`forward/metal_qwen35.rs`)
        // multiply it by `linear_key_head_dim` / `linear_value_head_dim`, which
        // `MAX_GDN_STATE_SIZE` only bounds as a three-factor product with `value_heads` --
        // a config with `value_heads` small and either head-dim field huge satisfies that
        // product trivially while still reaching a multi-gigabyte chunk-scratch allocation.
        // Mirror the exact engine formula (`chunk_rows = num_chunks * value_heads *
        // GDN_CHUNK_SIZE`, capped independent of `max_position_embeddings` because the
        // engine's prefill window is itself capped at `GDN_MAX_PREFILL_MIRROR`) via `u128`
        // so the bound computation itself cannot overflow `usize`. See
        // `MAX_GDN_CHUNK_SCRATCH_BYTES` docs.
        let num_chunks_upper =
            (GDN_MAX_PREFILL_MIRROR as u128).div_ceil(GDN_CHUNK_SIZE_MIRROR as u128);
        let chunk_rows_upper =
            num_chunks_upper * value_heads as u128 * GDN_CHUNK_SIZE_MIRROR as u128;
        let max_head_dim = cfg.linear_key_head_dim.max(cfg.linear_value_head_dim) as u128;
        let c2_upper = chunk_rows_upper * GDN_CHUNK_SIZE_MIRROR as u128;
        let worst_chunk_scratch_elems = (chunk_rows_upper * max_head_dim).max(c2_upper);
        let worst_chunk_scratch_bytes = worst_chunk_scratch_elems * 4;
        if worst_chunk_scratch_bytes > MAX_GDN_CHUNK_SCRATCH_BYTES as u128 {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: GatedDeltaNet chunk-scratch size \
                 ({worst_chunk_scratch_bytes} bytes) exceeds MAX_GDN_CHUNK_SCRATCH_BYTES \
                 ({MAX_GDN_CHUNK_SCRATCH_BYTES}): linear_num_value_heads ({value_heads}) * \
                 max(linear_key_head_dim, linear_value_head_dim) ({max_head_dim})"
            )));
        }
        // CLASS A2: `MAX_GDN_CHUNK_SCRATCH_BYTES` above only bounds the largest single
        // `GdnChunkScratch` buffer, but `new_session_inner` allocates several
        // key-dimension buffers (`gdn_qkv`, `gdn_z`, `gdn_qkvz`, `gdn_key_scratch`,
        // `gdn_raw_out`) plus all twelve `GdnChunkScratch` buffers together for one
        // session. Sum every allocation-site formula (mirroring `new_session_inner`
        // exactly; see the CLASS A2 table in the PR body) and budget the aggregate. `bp`
        // (the actual prefill window) is itself bounded by `GDN_MAX_PREFILL_MIRROR`
        // regardless of `max_position_embeddings`, so it is safe to use the same
        // worst-case upper bound here as the per-buffer check above. See
        // `MAX_GDN_SESSION_BYTES` docs.
        let bp_upper = GDN_MAX_PREFILL_MIRROR as u128;
        let output_dim_u = linear_output_dim as u128;
        let kd_u = cfg.linear_key_head_dim as u128;
        let vd_u = cfg.linear_value_head_dim as u128;
        let key_heads_u = key_heads as u128;
        let value_heads_u = value_heads as u128;
        // `linear_qkv_dim()`'s formula (`2 * linear_num_key_heads * linear_key_head_dim +
        // linear_output_dim()`), computed here rather than reusing the `conv_dim` local
        // (which is derived further below, after this check) to avoid reordering the
        // existing validation sequence.
        let qkv_dim_u = 2 * key_heads_u * kd_u + output_dim_u;
        let gdn_qkv_elems = bp_upper * qkv_dim_u;
        let gdn_z_elems = bp_upper * output_dim_u;
        let gdn_qkvz_elems = qkv_dim_u + output_dim_u;
        let gdn_key_scratch_elems = key_heads_u * (2 * kd_u + 1) + value_heads_u * 2;
        let gdn_raw_out_elems = output_dim_u;
        let chunk_raw_out_elems = bp_upper * output_dim_u;
        let qkw_kright_elems = 4 * chunk_rows_upper * kd_u;
        let vur_elems = 3 * chunk_rows_upper * vd_u;
        let bla_elems = chunk_rows_upper * 2;
        let gamma_elems = chunk_rows_upper;
        let gamma_end_elems = num_chunks_upper * value_heads_u;
        let kkt_qkl_elems = 2 * c2_upper;
        let total_session_elems = gdn_qkv_elems
            + gdn_z_elems
            + gdn_qkvz_elems
            + gdn_key_scratch_elems
            + gdn_raw_out_elems
            + chunk_raw_out_elems
            + qkw_kright_elems
            + vur_elems
            + bla_elems
            + gamma_elems
            + gamma_end_elems
            + kkt_qkl_elems;
        let total_session_bytes = total_session_elems * 4;
        if total_session_bytes > MAX_GDN_SESSION_BYTES as u128 {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: aggregate GatedDeltaNet per-session buffer size \
                 ({total_session_bytes} bytes) exceeds MAX_GDN_SESSION_BYTES \
                 ({MAX_GDN_SESSION_BYTES})"
            )));
        }
        // `MAX_LINEAR_NUM_KEY_HEADS` and `MAX_CONV_KERNEL_DIM` bound their respective factors
        // individually, but their product (via `linear_qkv_dim()`) still drives
        // `GatedDeltaNetState::new`'s `conv_buffer = vec![0.0; conv_dim * buf_len]`
        // (`attention/gdn.rs`, mirrored in the Metal GDN conv buffer allocation in
        // `forward/metal_qwen35.rs`), unconditional at first generation regardless of layer
        // mix. Validate the checked product globally for the same reason as the
        // `linear_output_dim` / `gdn_state_size` budgets above. See
        // `MAX_GDN_CONV_BUFFER_SIZE` docs.
        let conv_dim = key_heads
            .checked_mul(cfg.linear_key_head_dim)
            .and_then(|q| q.checked_mul(2))
            .and_then(|q| q.checked_add(linear_output_dim))
            .ok_or_else(|| {
                InferenceError::Inference(format!(
                    "invalid Qwen config.json: GatedDeltaNet qkv_dim overflows usize: \
                 2 * linear_num_key_heads ({key_heads}) * linear_key_head_dim ({}) + \
                 linear_output_dim ({linear_output_dim})",
                    cfg.linear_key_head_dim
                ))
            })?;
        let conv_buf_len = cfg.linear_conv_kernel_dim - 1;
        let gdn_conv_buffer_size = conv_dim.checked_mul(conv_buf_len).ok_or_else(|| {
            InferenceError::Inference(format!(
                "invalid Qwen config.json: GatedDeltaNet conv buffer size overflows usize: \
                 linear_qkv_dim ({conv_dim}) * (linear_conv_kernel_dim - 1) ({conv_buf_len})"
            ))
        })?;
        if gdn_conv_buffer_size > MAX_GDN_CONV_BUFFER_SIZE {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: GatedDeltaNet conv buffer size \
                 ({gdn_conv_buffer_size}) exceeds MAX_GDN_CONV_BUFFER_SIZE \
                 ({MAX_GDN_CONV_BUFFER_SIZE})"
            )));
        }
        // `RopeParams::mrope_section` is bounded at deserialize time
        // (`deserialize_mrope_section`, capped at `MAX_CONFIG_VECTOR_LEN`) rather than here --
        // a post-parse `.len()` check runs after serde has already materialized the Vec in
        // full, which does not prevent the allocation an oversized declared length would drive.
        // A present `vision_config` must be structurally valid (ADR-069 S1/S2 review
        // feedback): a present-but-malformed object (e.g. `depth: 0`) is syntactically
        // valid JSON and would otherwise silently load a truncated subset of
        // `model.visual.*` tensors instead of failing closed.
        if let Some(vision_cfg) = &cfg.vision_config {
            // `deepstack_visual_indexes` is likewise bounded at deserialize time
            // (`deserialize_deepstack_visual_indexes`) for the same reason -- see above.
            vision_cfg.validate()?;
        }
        // MoE routing dimensions come from an untrusted checkpoint directory and drive
        // `ForwardScratch::ensure_capacity`'s unconditional `resize(&mut self.router_logits,
        // cfg.num_experts.unwrap_or(0))` and `router_selected.resize(cfg.num_experts_per_tok
        // .unwrap_or(0), ..)` on every forward pass. A zero-element tensor satisfies shape
        // checks trivially, so nothing else catches an inflated `num_experts` /
        // `num_experts_per_tok` before those resizes run.
        //
        // `num_experts_per_tok` drives `router_selected.resize` independent of `num_experts`
        // presence -- a checkpoint can set `num_experts_per_tok` while leaving `num_experts`
        // unset, so this bound is unconditional rather than gated on `num_experts` like the
        // "must not exceed num_experts" check further down.
        if let Some(num_experts_per_tok) = cfg.num_experts_per_tok {
            if num_experts_per_tok == 0 {
                return Err(InferenceError::Inference(
                    "invalid Qwen config.json: num_experts_per_tok must be > 0".to_string(),
                ));
            }
            if num_experts_per_tok > MAX_NUM_EXPERTS {
                return Err(InferenceError::Inference(format!(
                    "invalid Qwen config.json: num_experts_per_tok ({num_experts_per_tok}) \
                     exceeds MAX_NUM_EXPERTS ({MAX_NUM_EXPERTS})"
                )));
            }
        }
        // Gate on `num_experts` presence so dense (non-MoE) configs, which leave it `None`,
        // are unaffected.
        if let Some(num_experts) = cfg.num_experts {
            if num_experts == 0 {
                return Err(InferenceError::Inference(
                    "invalid Qwen config.json: num_experts must be > 0".to_string(),
                ));
            }
            if num_experts > MAX_NUM_EXPERTS {
                return Err(InferenceError::Inference(format!(
                    "invalid Qwen config.json: num_experts ({num_experts}) exceeds \
                     MAX_NUM_EXPERTS ({MAX_NUM_EXPERTS})"
                )));
            }
            if let Some(num_experts_per_tok) = cfg.num_experts_per_tok
                && num_experts_per_tok > num_experts
            {
                return Err(InferenceError::Inference(format!(
                    "invalid Qwen config.json: num_experts_per_tok \
                     ({num_experts_per_tok}) must not exceed num_experts ({num_experts})"
                )));
            }
        }
        // MoE/shared-expert intermediate sizes drive their own scratch buffer allocations
        // independent of `num_experts` presence. Gate on each field's own presence (like the
        // `num_experts` if-let gate above) so dense configs, which leave both `None`, are
        // unaffected. Zero is a degenerate zero-width expert FFN, not a valid no-op, mirroring
        // the `intermediate_size == 0` check above. See `MAX_INTERMEDIATE_SIZE` docs.
        if let Some(moe_intermediate_size) = cfg.moe_intermediate_size {
            if moe_intermediate_size == 0 {
                return Err(InferenceError::Inference(
                    "invalid Qwen config.json: moe_intermediate_size must be > 0".to_string(),
                ));
            }
            if moe_intermediate_size > MAX_INTERMEDIATE_SIZE {
                return Err(InferenceError::Inference(format!(
                    "invalid Qwen config.json: moe_intermediate_size \
                     ({moe_intermediate_size}) exceeds MAX_INTERMEDIATE_SIZE \
                     ({MAX_INTERMEDIATE_SIZE})"
                )));
            }
        }
        if let Some(shared_expert_intermediate_size) = cfg.shared_expert_intermediate_size {
            if shared_expert_intermediate_size == 0 {
                return Err(InferenceError::Inference(
                    "invalid Qwen config.json: shared_expert_intermediate_size must be > 0"
                        .to_string(),
                ));
            }
            if shared_expert_intermediate_size > MAX_INTERMEDIATE_SIZE {
                return Err(InferenceError::Inference(format!(
                    "invalid Qwen config.json: shared_expert_intermediate_size \
                     ({shared_expert_intermediate_size}) exceeds MAX_INTERMEDIATE_SIZE \
                     ({MAX_INTERMEDIATE_SIZE})"
                )));
            }
        }
        // A zero `max_position_embeddings` is geometrically degenerate: the RoPE table has
        // zero rows (`RopeTable::max_positions() == 0`), so any non-empty forward pass
        // indexes `cos_at(0)` / `sin_at(0)` past the empty table -- a panic reachable via
        // the public debug forward (`qwen35_config.rs`'s own `cos_at`/`sin_at` callers).
        // The upper-bound check below only ever guarded the ceiling; this guards the floor.
        if cfg.max_position_embeddings == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: max_position_embeddings must be > 0".to_string(),
            ));
        }
        // `max_position_embeddings` drives the Metal RoPE table allocation (`rope_max *
        // rope_dim / 2` entries in `build_rope_interleaved`) independent of `head_dim`. See
        // `MAX_POSITION_EMBEDDINGS` docs.
        if cfg.max_position_embeddings > MAX_POSITION_EMBEDDINGS {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: max_position_embeddings ({}) exceeds \
                 MAX_POSITION_EMBEDDINGS ({MAX_POSITION_EMBEDDINGS})",
                cfg.max_position_embeddings
            )));
        }

        Ok(ValidatedQwen35Config(cfg))
    }

    /// Resolved linear value head count (falls back to 32 for Qwen3.6 if unset).
    pub fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads.unwrap_or(32)
    }

    /// Returns true when the model uses Mixture-of-Experts FFN layers.
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some()
            || self.num_experts_per_tok.is_some()
            || self.moe_intermediate_size.is_some()
            || self.shared_expert_intermediate_size.is_some()
    }

    /// Resolved MoE routed expert intermediate size (falls back to `intermediate_size`).
    pub fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// Resolved shared expert intermediate size.
    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.shared_expert_intermediate_size
            .unwrap_or_else(|| self.moe_intermediate_size())
    }

    /// **Unstable**: count of full-attention layers in the hybrid stack.
    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count()
    }

    /// **Unstable**: count of GatedDeltaNet linear-attention layers.
    pub fn num_linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count()
    }

    /// **Unstable**: Q projection dimension for full-attention layers.
    pub fn full_q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// **Unstable**: KV projection dimension for full-attention layers.
    pub fn full_kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// **Unstable**: number of layers that hold a KV cache.
    ///
    /// Only full-attention (GQA) layers carry a growing KV cache. The
    /// GatedDeltaNet linear-attention layers carry fixed-size recurrent state
    /// that does not grow with sequence length. Therefore KV memory scales with
    /// `num_full_attention_layers()`, not `num_hidden_layers`.
    ///
    /// For qwen3.5-0.8B this is 6 (not 24). A regression to all-24-layer KV
    /// allocation would 4× decode memory; this method makes the invariant
    /// testable.
    pub fn kv_cache_layer_count(&self) -> usize {
        self.num_full_attention_layers()
    }

    /// **Unstable**: total KV cache bytes consumed per input token.
    ///
    /// Formula: `num_full_attention_layers × 2 (K and V) × full_kv_dim × dtype_bytes`.
    ///
    /// Pass `dtype_bytes = 2` for f16, `dtype_bytes = 4` for f32.
    ///
    /// For qwen3.5-0.8B with f16:
    /// `6 × 2 × 512 × 2 = 12_288 B/token ≈ 48 MiB at a 4096-token context`.
    ///
    /// Numeric identity (preset-specific, NOT a general formula): at `dtype_bytes = 1`,
    /// `kv_bytes_per_token(1) = 6 × 2 × 512 = 6_144 = 24 × 256 = num_hidden_layers × head_dim`.
    /// Since `kv_bytes_per_token(1) = num_full × 2 × num_kv_heads × head_dim`, it equals
    /// `num_hidden_layers × head_dim` exactly when `num_full × 2 × num_kv_heads == num_hidden_layers`
    /// — true for 0.8B (`6 × 2 × 2 == 24`) but not in general. Do not rely on it across configs.
    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> usize {
        self.num_full_attention_layers() * 2 * self.full_kv_dim() * dtype_bytes
    }

    /// **Unstable**: number of RoPE dimensions (partial rotary factor applied).
    pub fn rope_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// **Unstable**: total QKV projection output size for GatedDeltaNet layers.
    pub fn linear_qkv_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim   // Q
        + self.linear_num_key_heads * self.linear_key_head_dim // K
        + self.linear_num_value_heads() * self.linear_value_head_dim // V
    }

    /// **Unstable**: output dimension for GatedDeltaNet layers.
    pub fn linear_output_dim(&self) -> usize {
        self.linear_num_value_heads() * self.linear_value_head_dim
    }

    /// **Unstable**: returns true when layer `i` uses full GQA attention.
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::LinearAttention)
            == LayerType::FullAttention
    }

    /// Normalizes `layer_mask` to `num_hidden_layers` all-true entries if length mismatches.
    fn normalize_layer_mask(&mut self) {
        if self.layer_mask.len() != self.num_hidden_layers {
            self.layer_mask = vec![true; self.num_hidden_layers];
        }
    }

    /// Returns true if layer `layer_idx` is active (not pruned).
    pub fn is_layer_active(&self, layer_idx: usize) -> bool {
        self.layer_mask.get(layer_idx).copied().unwrap_or(true)
    }

    /// Count of active (non-pruned) layers.
    pub fn num_active_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_layer_active(i))
            .count()
    }

    /// Count of active GatedDeltaNet linear-attention layers.
    pub fn num_active_linear_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_layer_active(i) && !self.is_full_attention(i))
            .count()
    }

    /// Count of active full-attention (GQA) layers.
    pub fn num_active_full_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_layer_active(i) && self.is_full_attention(i))
            .count()
    }

    /// Applies `mask` as the layer pruning mask. Panics if length or all-false.
    ///
    /// All built-in constructors (`qwen35_2b`, `qwen36_35b_a3b`, `qwen36_27b`) produce
    /// an all-true mask.  To enable pruning, call this method or [`Self::pruned_config`]
    /// after construction.
    pub fn apply_layer_mask(&mut self, mask: Vec<bool>) {
        assert_eq!(
            mask.len(),
            self.num_hidden_layers,
            "layer_mask length {} does not match num_hidden_layers {}",
            mask.len(),
            self.num_hidden_layers
        );
        assert!(
            mask.iter().any(|&active| active),
            "layer_mask must keep at least one active layer"
        );
        self.layer_mask = mask;
    }

    /// Returns a clone with `mask` applied as the pruning mask.
    pub fn pruned_config(&self, mask: Vec<bool>) -> Self {
        let mut cfg = self.clone();
        cfg.apply_layer_mask(mask);
        cfg
    }
}

/// **Unstable**: sampling configuration for text generation; temperature/top-k/top-p may expand.
#[derive(Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Random seed for sampling. `None` = seed from system time.
    pub seed: Option<u64>,
    /// Additional stop token IDs (beyond EOS). Generation stops on any of these.
    pub stop_token_ids: Vec<u32>,
    /// When false, caller prepends `QWEN3_NO_THINK_SYSTEM_MSG` to disable chain-of-thought.
    pub enable_thinking: bool,
    /// Enable multi-token prediction when the model has MTP weights loaded.
    /// Replaces the `LATTICE_MTP` env var for programmatic control.
    /// `None` = defer to `LATTICE_MTP` env var (backwards-compatible default).
    pub enable_mtp: Option<bool>,
    /// Optional grammar-constrained decoding engine (ADR-046).
    ///
    /// When set, `mask_logits` is called on CPU logits before sampling on every step.
    /// The Metal path copies logits to CPU before sampling — no additional GPU transfer needed.
    pub grammar: Option<Arc<GrammarEngine>>,
    /// Additional string-level stop sequences. When any appears in the output, generation
    /// halts and the matched text is excluded. Empty = disabled (default; parity-safe).
    pub stop_strings: Vec<String>,
    /// Reasoning-budget forcing (s1-style): after this many reasoning tokens are
    /// generated without a `</think>`, force-inject `</think>` to commit the model
    /// to an answer. `None` or `Some(0)` = disabled (no behaviour change).
    pub reasoning_budget: Option<usize>,
    /// Capture per-token log-probabilities (OpenAI `logprobs`/`top_logprobs`).
    /// `None` (default) disables capture entirely -- no extra allocation or
    /// computation is added to the decode loop. `Some(n)` captures the
    /// sampled token's log-probability plus its `n` highest-probability
    /// alternatives at every generated step (`n == 0` is valid: report only
    /// the sampled token's log-probability, no alternatives).
    pub logprobs: Option<usize>,
}

impl std::fmt::Debug for GenerateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerateConfig")
            .field("max_new_tokens", &self.max_new_tokens)
            .field("temperature", &self.temperature)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("repetition_penalty", &self.repetition_penalty)
            .field("seed", &self.seed)
            .field("stop_token_ids", &self.stop_token_ids)
            .field("enable_thinking", &self.enable_thinking)
            .field("enable_mtp", &self.enable_mtp)
            .field("grammar", &self.grammar.as_ref().map(|_| "<GrammarEngine>"))
            .field("stop_strings", &self.stop_strings)
            .field("reasoning_budget", &self.reasoning_budget)
            .field("logprobs", &self.logprobs)
            .finish()
    }
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: None,
            stop_token_ids: vec![QWEN_CHAT_IM_END_TOKEN_ID],
            enable_thinking: true,
            enable_mtp: None,
            grammar: None,
            stop_strings: vec![],
            reasoning_budget: None,
            logprobs: None,
        }
    }
}

/// Decide whether to force-close the thinking block this step (s1 budget forcing).
///
/// Returns `Some(close_id)` to override the sampled token with `</think>`, else `None`.
/// All conditions must hold: budget enabled and non-zero, thinking block is still open,
/// enough tokens have been generated. Returns `None` immediately if any guard fails so
/// the common disabled path costs a single `Option::None` check per step.
#[inline]
pub(crate) fn force_close_think(
    reasoning_budget: Option<usize>,
    enable_thinking: bool,
    thinking_closed: bool,
    generated_so_far: usize,
    close_id: Option<u32>,
) -> Option<u32> {
    let budget = reasoning_budget?;
    let close = close_id?;
    if enable_thinking && !thinking_closed && budget > 0 && generated_so_far >= budget {
        Some(close)
    } else {
        None
    }
}

/// Decode-loop iteration cap.
///
/// When a reasoning budget is active, reasoning tokens get their OWN budget (`rb`) ON TOP
/// of the answer budget (`max_new_tokens`), plus **1** for the forced `</think>` delimiter
/// itself: worst case `rb + max_new_tokens + 1` total tokens. The +1 is necessary because
/// the forced `</think>` is an extra token that is not part of either the reasoning content
/// or the answer — omitting it leaves the answer one token short (off-by-one).
///
/// Without a budget (`None` or `Some(0)`) the cap is unchanged (`max_new_tokens`), so the
/// disabled path is byte-identical to the pre-budget behaviour (parity-safe).
#[inline]
pub(crate) fn decode_cap(reasoning_budget: Option<usize>, max_new_tokens: usize) -> usize {
    match reasoning_budget {
        // rb reasoning-content tokens + 1 forced </think> delimiter + max_new_tokens answer tokens.
        Some(rb) if rb > 0 => rb.saturating_add(max_new_tokens).saturating_add(1),
        _ => max_new_tokens,
    }
}

/// One alternative token considered at a single generation step, paired with
/// its log-probability under the reporting distribution computed by
/// [`crate::sampling`]'s logprobs support. Ordered by descending probability
/// when produced via [`GenerateConfig::logprobs`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

/// Per-token log-probability data for one generated token. Only populated
/// when [`GenerateConfig::logprobs`] is `Some`; `GenerateOutput::token_logprobs`
/// stays empty otherwise (no extra allocation on the default path).
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    /// The token id that was actually sampled/generated at this step.
    pub token_id: u32,
    /// Natural-log probability of `token_id` under a temperature-scaled
    /// softmax over that step's raw logits.
    pub logprob: f32,
    /// The requested number of highest-probability alternatives at this step
    /// (may include `token_id` itself), sorted by descending probability.
    /// Empty when `logprobs` was requested with a `top_logprobs` count of 0.
    pub top: Vec<TopLogprob>,
}

/// **Unstable**: text generation output struct; fields may expand with streaming support.
///
/// # Stop-token contract (#613)
///
/// When generation ends because EOS or a configured `stop_token_ids` entry is
/// hit, that terminating token is **excluded** from `token_ids` and `text` —
/// it is never appended to the output. Every generation entry point across
/// this crate (CPU and Metal) honours this contract (see the
/// `stop_token_contract` test module for the cross-path regression sweep).
/// `generated_tokens` always equals `token_ids.len()`.
///
/// **`stop_strings` behave differently (#632).** A
/// `stop_strings` match truncates `text` to the point where the match begins,
/// but the token(s) whose decoded text completed the match are **not**
/// removed from `token_ids`/`generated_tokens` — the implementation cannot
/// "un-generate" a token once it has been decoded and appended (see
/// `decode_loop_with_stops` / `earliest_stop_match` in
/// `crate::model::qwen35::generation`). So for a `stop_strings` stop,
/// `token_ids.len()` (== `generated_tokens`) can exceed the number of tokens
/// whose text actually survived in the truncated `text`. The EOS /
/// `stop_token_ids` exclusion guarantee above does not extend to this case.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    /// Generated text (excluding prompt).
    pub text: String,
    /// Generated token IDs. Excludes the terminating token for EOS /
    /// `stop_token_ids` stops (see the stop-token contract above), but a
    /// `stop_strings` stop retains the token(s) that completed the match —
    /// see the `stop_strings` note above.
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Total tokens generated (excluding prompt).
    pub generated_tokens: usize,
    /// True when generation ended via a stop condition (EOS, a stop token, or a
    /// stop string); false when it ended by reaching `max_new_tokens`. Serve maps
    /// this to the OpenAI `finish_reason` ("stop" vs "length").
    pub stopped: bool,
    /// Why generation terminated. `Some` on every real generation exit; `None` only on
    /// non-generation returns that have no issue-listed cause.
    pub stop_reason: Option<StopReason>,
    /// Per-step log-probability data, one entry per generated token, in
    /// generation order. Empty unless `GenerateConfig::logprobs` was `Some`.
    pub token_logprobs: Vec<TokenLogprob>,
}

/// Compute the layer type pattern: every `interval`-th layer (1-indexed) is full attention.
/// For interval=4: layers 3, 7, 11, 15, 19, 23 are full (0-indexed).
pub(crate) fn compute_layer_types(num_layers: usize, interval: usize) -> Vec<LayerType> {
    (0..num_layers)
        .map(|i| {
            if (i + 1) % interval == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_construction_and_layer_types() {
        let cfg = Qwen35Config::qwen35_2b();

        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.layer_types.len(), 24);

        // Check the pattern: [lin, lin, lin, full] x 6
        assert_eq!(cfg.num_full_attention_layers(), 6);
        assert_eq!(cfg.num_linear_attention_layers(), 18);

        // Full attention at indices 3, 7, 11, 15, 19, 23
        let full_indices: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::FullAttention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23]);

        // Linear attention at all other indices
        for i in [
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
        ] {
            assert_eq!(cfg.layer_types[i], LayerType::LinearAttention);
        }
    }

    #[test]
    fn test_generate_config_defaults() {
        let gen_cfg = GenerateConfig::default();
        assert_eq!(gen_cfg.max_new_tokens, 256);
        assert!((gen_cfg.temperature - 0.7).abs() < 1e-6);
        assert_eq!(gen_cfg.top_k, 50);
        assert!((gen_cfg.top_p - 0.9).abs() < 1e-6);
        assert!((gen_cfg.repetition_penalty - 1.1).abs() < 1e-6);
        assert!(
            gen_cfg.stop_token_ids.contains(&QWEN_CHAT_IM_END_TOKEN_ID),
            "default stop tokens must include im_end"
        );
    }

    #[test]
    fn test_dimension_helpers() {
        let cfg = Qwen35Config::qwen35_2b();

        // Full attention dims
        assert_eq!(cfg.full_q_dim(), 8 * 256); // 2048
        assert_eq!(cfg.full_kv_dim(), 2 * 256); // 512
        assert_eq!(cfg.rope_dim(), 64); // 0.25 * 256

        // Linear attention dims (Qwen3.5-2B: 16 value heads)
        // Q: 16*128=2048, K: 16*128=2048, V: 16*128=2048 → total 6144
        assert_eq!(cfg.linear_qkv_dim(), 6144);
        assert_eq!(cfg.linear_output_dim(), 2048); // 16 * 128
    }

    #[test]
    fn test_is_full_attention() {
        let cfg = Qwen35Config::qwen35_2b();
        assert!(!cfg.is_full_attention(0));
        assert!(!cfg.is_full_attention(1));
        assert!(!cfg.is_full_attention(2));
        assert!(cfg.is_full_attention(3));
        assert!(!cfg.is_full_attention(4));
        assert!(cfg.is_full_attention(7));
        assert!(cfg.is_full_attention(23));
        // Out of bounds returns false (linear)
        assert!(!cfg.is_full_attention(100));
    }

    #[test]
    fn test_qwen36_hf_config_fixture_parse_fields() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen36_config.json"
        ));

        let cfg = Qwen35Config::from_config_json_str(json).expect("Qwen3.6 HF config parses");

        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.linear_num_value_heads(), 32);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_experts, Some(256));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.moe_intermediate_size, Some(512));
        assert_eq!(cfg.shared_expert_intermediate_size, Some(512));
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.is_moe());

        // 12 additional field assertions.
        assert_eq!(cfg.vocab_size, 248320);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.eos_token_id, 248044_u32);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert_eq!(cfg.rope_theta, 10_000_000.0_f64);
        assert_eq!(cfg.partial_rotary_factor, 0.25_f32);

        let full_indices: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::FullAttention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
    }

    #[test]
    fn test_qwen36_27b_preset_dimensions() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 64);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.intermediate_size, 17408);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert!((cfg.rope_theta - 10_000_000.0_f64).abs() < 1.0);
        assert!((cfg.partial_rotary_factor - 0.25_f32).abs() < 1e-6);
        assert_eq!(cfg.eos_token_id, 248_044_u32);
        assert_eq!(cfg.max_position_embeddings, 262_144);
    }

    #[test]
    fn test_qwen36_27b_preset_layer_types() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.layer_types.len(), 64);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(
            cfg.layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            48
        );
        assert_eq!(
            cfg.layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            16
        );
        // Every 4th layer (1-indexed) is full attention: indices 3, 7, 11, ..., 63
        for i in 0..64_usize {
            let expected = (i + 1) % 4 == 0;
            assert_eq!(
                cfg.layer_types[i] == LayerType::FullAttention,
                expected,
                "layer {i} type mismatch"
            );
        }
    }

    #[test]
    fn test_qwen36_27b_preset_not_moe() {
        let cfg = Qwen35Config::qwen36_27b();
        assert!(!cfg.is_moe());
        assert!(cfg.num_experts.is_none());
        assert!(cfg.num_experts_per_tok.is_none());
        assert!(cfg.moe_intermediate_size.is_none());
        assert!(cfg.shared_expert_intermediate_size.is_none());
        assert!(cfg.router_aux_loss_coef.is_none());
        assert!(!cfg.output_router_logits);
    }

    #[test]
    fn test_qwen36_27b_preset_gdn_fields() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads, Some(48));
        assert_eq!(cfg.linear_num_value_heads(), 48);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        // Untied embeddings — separate lm_head.weight
        assert!(!cfg.tie_word_embeddings);
        // MTP
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        assert!(!cfg.mtp_use_dedicated_embeddings);
    }

    #[test]
    fn test_qwen36_27b_from_config_json() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let path =
            std::path::PathBuf::from(format!("{home}/.lattice/models/qwen3.6-27b/config.json"));
        if !path.exists() {
            return; // model not downloaded; skip
        }
        let cfg = Qwen35Config::from_config_json(&path).expect("27B config.json parses");
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 64);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.intermediate_size, 17408);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.layer_types.len(), 64);
        assert!(!cfg.is_moe());
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        // rope_theta is nested under rope_parameters — from_config_json_str now extracts it.
        assert!(
            (cfg.rope_theta - 10_000_000.0_f64).abs() < 1.0,
            "rope_theta should be extracted from nested rope_parameters"
        );
    }

    // --- layer_mask tests ---

    #[test]
    fn test_layer_mask_default_all_true_27b() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.layer_mask.len(), 64);
        assert!(cfg.layer_mask.iter().all(|&active| active));
        assert_eq!(cfg.num_active_layers(), 64);
        assert_eq!(cfg.num_active_linear_attention_layers(), 48);
        assert_eq!(cfg.num_active_full_attention_layers(), 16);
    }

    #[test]
    fn test_num_active_layers_partial_mask() {
        let mut cfg = Qwen35Config::qwen36_27b();
        // Deactivate layer 0 (GDN), layer 3 (GQA), layer 4 (GDN).
        let mut mask = vec![true; 64];
        mask[0] = false;
        mask[3] = false;
        mask[4] = false;
        cfg.apply_layer_mask(mask);
        assert_eq!(cfg.num_active_layers(), 61);
        assert_eq!(cfg.num_active_linear_attention_layers(), 46);
        assert_eq!(cfg.num_active_full_attention_layers(), 15);
    }

    #[test]
    #[should_panic(expected = "layer_mask length")]
    fn test_apply_layer_mask_wrong_length_panics() {
        let mut cfg = Qwen35Config::qwen36_27b();
        cfg.apply_layer_mask(vec![true; 32]);
    }

    #[test]
    fn test_pruned_config_preserves_fields() {
        let cfg = Qwen35Config::qwen36_27b();
        let mut mask = vec![true; 64];
        mask[5] = false;
        mask[10] = false;
        let pruned = cfg.pruned_config(mask.clone());
        assert_eq!(pruned.hidden_size, cfg.hidden_size);
        assert_eq!(pruned.num_hidden_layers, cfg.num_hidden_layers);
        assert_eq!(pruned.vocab_size, cfg.vocab_size);
        assert_eq!(pruned.layer_types, cfg.layer_types);
        assert_eq!(pruned.layer_mask, mask);
        assert_eq!(pruned.num_active_layers(), 62);
    }

    #[test]
    #[should_panic(expected = "at least one active layer")]
    fn test_apply_layer_mask_all_false_panics() {
        let mut cfg = Qwen35Config::qwen36_27b();
        cfg.apply_layer_mask(vec![false; 64]);
    }

    #[test]
    fn test_layer_mask_normalizes_on_parse() {
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "full_attention_interval": 4,
                "eos_token_id": 1
            }
        }"#;
        let cfg = Qwen35Config::from_config_json_str(json).unwrap();
        assert_eq!(
            cfg.layer_mask.len(),
            4,
            "normalize_layer_mask must fill mask to num_hidden_layers"
        );
        assert!(
            cfg.layer_mask.iter().all(|&v| v),
            "normalized mask must be all-true"
        );
    }

    #[test]
    fn test_zero_full_attention_interval_errors_not_panics() {
        // A config.json with full_attention_interval: 0 must return a clean
        // InferenceError, never panic. layer_types is omitted, so the container
        // #[serde(default)] fills it from the preset (whose length differs from
        // num_hidden_layers: 4), forcing the recompute branch that calls
        // compute_layer_types(4, 0) -> (i + 1) % 0 -> divide-by-zero panic
        // without the guard.
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "full_attention_interval": 0,
                "eos_token_id": 1
            }
        }"#;
        let result = Qwen35Config::from_config_json_str(json);
        assert!(
            result.is_err(),
            "full_attention_interval: 0 must yield an InferenceError, not panic"
        );
    }

    #[test]
    fn test_zero_num_key_value_heads_errors_not_panics() {
        // An explicit num_key_value_heads: 0 survives serde but reaches a divide-by-zero
        // (`num_q_heads / num_kv_heads`) and a hard `assert!(num_kv_heads > 0)` in the GQA
        // forward path. Reject at parse time. Omitted fields fall back to the valid preset.
        // The substring assert proves THIS guard fired (not an unrelated default-zero field):
        // `#[serde(default)]` + `Default = qwen36_35b_a3b()` means every omitted field carries a
        // valid preset value, so the one explicit bad field is the only one that can trip.
        let json = r#"{"text_config": {"num_key_value_heads": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("num_key_value_heads: 0 must yield an InferenceError, not a panic")
            .to_string();
        assert!(
            err.contains("num_key_value_heads"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_indivisible_head_counts_error_not_panics() {
        // num_attention_heads not divisible by num_key_value_heads truncates the GQA group
        // count and over-runs the KV row (OOB read) on the unasserted release path.
        let json = r#"{"text_config": {"num_attention_heads": 3, "num_key_value_heads": 2}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("indivisible head counts must yield an InferenceError, not OOB/panic")
            .to_string();
        assert!(err.contains("divisible"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_zero_head_dim_errors() {
        let json = r#"{"text_config": {"head_dim": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("head_dim: 0 must yield an InferenceError")
            .to_string();
        assert!(err.contains("head_dim"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_zero_num_hidden_layers_errors() {
        let json = r#"{"text_config": {"num_hidden_layers": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("num_hidden_layers: 0 must yield an InferenceError")
            .to_string();
        assert!(
            err.contains("num_hidden_layers"),
            "wrong guard fired: {err}"
        );
    }

    /// A checkpoint-controlled `num_hidden_layers` far past any real Qwen3.5 depth must be
    /// rejected with a typed error at config-parse time, before `compute_layer_types` /
    /// `normalize_layer_mask` (both of which allocate a `Vec` proportional to
    /// `num_hidden_layers`) ever run. Uses `usize::MAX` -- the most extreme value a
    /// malicious `config.json` could carry -- to demonstrate the guard fires before any
    /// layer-proportional allocation is attempted.
    #[test]
    fn test_extreme_num_hidden_layers_rejected_before_allocation() {
        let json = format!(
            r#"{{"text_config": {{"num_hidden_layers": {}}}}}"#,
            usize::MAX
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "num_hidden_layers: usize::MAX must yield an InferenceError, not a panic/OOM",
            )
            .to_string();
        assert!(
            err.contains("num_hidden_layers") && err.contains("MAX_HIDDEN_LAYERS"),
            "wrong guard fired: {err}"
        );
    }

    /// Same DoS vector as above with a value (10_000_000) large enough to be an absurd
    /// layer count but small enough that, absent the `MAX_HIDDEN_LAYERS` guard, the
    /// resulting allocation would actually complete rather than aborting the process --
    /// this is the config used for the mutation revert-check (see fix report).
    #[test]
    fn test_ten_million_hidden_layers_rejected() {
        let json = r#"{"text_config": {"num_hidden_layers": 10000000}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("num_hidden_layers: 10_000_000 must yield an InferenceError")
            .to_string();
        assert!(
            err.contains("num_hidden_layers") && err.contains("MAX_HIDDEN_LAYERS"),
            "wrong guard fired: {err}"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Bounded layer_types / layer_mask deserialization
    // ──────────────────────────────────────────────────────────────────────

    /// `layer_types` sized past `MAX_HIDDEN_LAYERS` must be rejected during
    /// deserialization itself -- before the post-parse `num_hidden_layers` guard runs.
    /// `num_hidden_layers` is deliberately left at its default (40, well under the cap)
    /// so an `Err` here can only come from the bounded-seq deserializer, not the
    /// separate `num_hidden_layers > MAX_HIDDEN_LAYERS` check.
    #[test]
    fn test_layer_types_array_over_max_hidden_layers_rejected_at_deserialize() {
        let elems = vec!["\"linear_attention\""; MAX_HIDDEN_LAYERS + 1].join(",");
        let json = format!(r#"{{"text_config": {{"layer_types": [{elems}]}}}}"#);
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "a layer_types array of MAX_HIDDEN_LAYERS+1 elements must be rejected \
                 during deserialization, not allocated in full",
            )
            .to_string();
        assert!(
            err.contains("MAX_HIDDEN_LAYERS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_layer_types_in_bounds_array_accepted() {
        let json = r#"{"text_config": {"num_hidden_layers": 3, "full_attention_interval": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention"]}}"#;
        let cfg = Qwen35Config::from_config_json_str(json)
            .expect("an in-bounds layer_types array must be accepted");
        assert_eq!(cfg.layer_types.len(), 3);
    }

    /// Same DoS vector as above for `layer_mask`, the sibling per-layer array.
    #[test]
    fn test_layer_mask_array_over_max_hidden_layers_rejected_at_deserialize() {
        let elems = vec!["true"; MAX_HIDDEN_LAYERS + 1].join(",");
        let json = format!(r#"{{"text_config": {{"layer_mask": [{elems}]}}}}"#);
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "a layer_mask array of MAX_HIDDEN_LAYERS+1 elements must be rejected \
                 during deserialization, not allocated in full",
            )
            .to_string();
        assert!(
            err.contains("MAX_HIDDEN_LAYERS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_layer_mask_in_bounds_array_accepted() {
        let json = r#"{"text_config": {"num_hidden_layers": 3, "full_attention_interval": 4,
            "layer_mask": [true, false, true]}}"#;
        let cfg = Qwen35Config::from_config_json_str(json)
            .expect("an in-bounds layer_mask array must be accepted");
        assert_eq!(cfg.layer_mask, vec![true, false, true]);
    }

    // ──────────────────────────────────────────────────────────────────────
    // hidden_size / vocab_size bounds
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_hidden_size_zero_errors() {
        let json = r#"{"text_config": {"hidden_size": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("hidden_size: 0 must yield an InferenceError")
            .to_string();
        assert!(err.contains("hidden_size"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_vocab_size_zero_errors() {
        let json = r#"{"text_config": {"vocab_size": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("vocab_size: 0 must yield an InferenceError")
            .to_string();
        assert!(err.contains("vocab_size"), "wrong guard fired: {err}");
    }

    /// A checkpoint declaring `embed_tokens` as `[huge_vocab, 0]` (zero-byte tensor,
    /// passes shape checks) must be rejected at config-parse time via `MAX_VOCAB_SIZE`,
    /// before the `logits` buffer resize (`cache.rs`) sees this value.
    #[test]
    fn test_vocab_size_over_max_errors() {
        let json = format!(
            r#"{{"text_config": {{"vocab_size": {}}}}}"#,
            MAX_VOCAB_SIZE + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("vocab_size above MAX_VOCAB_SIZE must yield an InferenceError")
            .to_string();
        assert!(
            err.contains("vocab_size") && err.contains("MAX_VOCAB_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_vocab_size_at_max_accepted() {
        let json = format!(r#"{{"text_config": {{"vocab_size": {MAX_VOCAB_SIZE}}}}}"#);
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "vocab_size == MAX_VOCAB_SIZE must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // head_dim bound (config-level budget, independent of layer mix)
    // ──────────────────────────────────────────────────────────────────────

    /// An all-linear-attention layer mix never calls `checked_full_q_dim` (the
    /// full-attention Q projection sizing check), so `MAX_HEAD_DIM` must fire on its
    /// own to bound `RopeTable::new`'s allocation regardless of layer mix.
    #[test]
    fn test_head_dim_over_max_rejected_all_linear_config() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": {}, "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#,
            MAX_HEAD_DIM + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "head_dim above MAX_HEAD_DIM must yield an InferenceError even for an \
                 all-linear-attention config",
            )
            .to_string();
        assert!(
            err.contains("head_dim") && err.contains("MAX_HEAD_DIM"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_head_dim_at_max_accepted_all_linear_config() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": {MAX_HEAD_DIM}, "num_hidden_layers": 2,
                "partial_rotary_factor": 0.25,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "head_dim == MAX_HEAD_DIM must be accepted for an all-linear-attention config"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // full-attention geometry global overflow bound (config-level budget,
    // independent of layer mix)
    // ──────────────────────────────────────────────────────────────────────

    /// An all-linear-attention layer mix never loads full-attention tensors, so it never
    /// reaches the loader's per-layer full-attention `q_dim` checks -- yet
    /// `ForwardScratch::ensure_capacity` unconditionally computes `full_q_dim()` and
    /// `2 * q_dim` for every config during generation. `num_attention_heads * head_dim`
    /// must be rejected at parse time even when no full-attention layer is present.
    ///
    /// Originally exercised the `checked_mul` overflow guard directly (an `usize::MAX`
    /// `num_attention_heads` overflowing the `* head_dim` product). `MAX_ATTENTION_HEADS`
    /// (added alongside `MAX_HIDDEN_SIZE` / `MAX_POSITION_EMBEDDINGS` in the allocation-scalar
    /// sweep) now rejects any `num_attention_heads` above 8,192 before that multiplication
    /// runs -- and since `MAX_ATTENTION_HEADS * MAX_HEAD_DIM` (8,192 * 2,048 = 16,777,216)
    /// never overflows `usize`, the overflow branch is no longer reachable through
    /// `num_attention_heads` alone. The `checked_mul` call stays in the source as
    /// defense-in-depth (e.g. against a future `MAX_ATTENTION_HEADS` or `MAX_HEAD_DIM`
    /// increase); this test now covers `MAX_ATTENTION_HEADS` catching the same extreme input.
    #[test]
    fn test_full_q_dim_overflow_rejected_all_linear_config() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": 2, "num_attention_heads": {},
                "num_key_value_heads": 1, "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#,
            usize::MAX
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "an extreme num_attention_heads must yield an InferenceError even for an \
                 all-linear-attention config",
            )
            .to_string();
        assert!(
            err.contains("num_attention_heads") && err.contains("MAX_ATTENTION_HEADS"),
            "wrong guard fired: {err}"
        );
    }

    /// Same bypass as above, but driven through `num_key_value_heads` instead of
    /// `num_attention_heads`. `num_attention_heads` must remain a multiple of
    /// `num_key_value_heads` (an earlier guard), so both are set to `usize::MAX` here --
    /// this still proves the KV-dimension geometry is bound by the same global check, since
    /// `full_kv_dim` shares `head_dim` with `full_q_dim` and neither guard can be bypassed by
    /// routing the overflow through the KV head count instead of the Q head count.
    ///
    /// As with the Q-dim test above, `MAX_ATTENTION_HEADS` now catches this extreme
    /// `num_attention_heads` (checked before `num_key_value_heads`) ahead of the
    /// `checked_mul` overflow branch it originally targeted; see that test's doc comment.
    #[test]
    fn test_full_kv_dim_overflow_rejected_all_linear_config() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": 2, "num_attention_heads": {max},
                "num_key_value_heads": {max}, "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#,
            max = usize::MAX
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "an extreme num_attention_heads / num_key_value_heads must yield an \
                 InferenceError even for an all-linear-attention config",
            )
            .to_string();
        assert!(
            err.contains("MAX_ATTENTION_HEADS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_full_attention_geometry_accepted_full_attention_config() {
        // A real full-attention-carrying config (the default MoE preset) must still load.
        assert!(
            Qwen35Config::from_config_json_str("{}").is_ok(),
            "a valid full-attention-carrying config must be accepted"
        );
    }

    #[test]
    fn test_full_attention_geometry_accepted_all_linear_config() {
        let json = r#"{"text_config": {"num_hidden_layers": 2,
            "layer_types": ["linear_attention", "linear_attention"]}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "a valid all-linear-attention config must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // full-attention geometry allocation BUDGET (config-level, independent of
    // layer mix) -- distinct from the overflow-only checks above: a
    // non-overflowing-but-enormous num_attention_heads * head_dim must also be
    // rejected, since it drives ForwardScratch::ensure_capacity's q_buf /
    // context allocations to exabyte scale without ever wrapping usize.
    // ──────────────────────────────────────────────────────────────────────

    /// head_dim=2048 (== MAX_HEAD_DIM, itself legal) with num_attention_heads=513 gives
    /// full_q_dim = 1,050,624 -- one order of magnitude within usize (no overflow) but over
    /// MAX_FULL_ATTENTION_DIM. num_key_value_heads stays small so full_kv_dim is nowhere
    /// near the budget, isolating the q_dim branch specifically.
    #[test]
    fn test_full_q_dim_over_budget_rejected_all_linear_config() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": {MAX_HEAD_DIM}, "num_attention_heads": 513,
                "num_key_value_heads": 1, "num_hidden_layers": 2,
                "partial_rotary_factor": 0.25,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "a non-overflowing but budget-exceeding full_q_dim must yield an \
                 InferenceError even for an all-linear-attention config",
            )
            .to_string();
        assert!(
            err.contains("q_dim") && err.contains("MAX_FULL_ATTENTION_DIM"),
            "wrong guard fired: {err}"
        );
    }

    /// Same allocation-budget bypass, but with num_attention_heads == num_key_value_heads so
    /// full_kv_dim also exceeds the budget (kv_dim can never exceed q_dim given the
    /// num_attention_heads-is-a-multiple-of-num_key_value_heads invariant enforced earlier in
    /// this function, so this proves the KV-routed budget check is reachable via the same
    /// global check rather than dead code, mirroring the existing overflow test pair above).
    #[test]
    fn test_full_kv_dim_over_budget_rejected_all_linear_config() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": {MAX_HEAD_DIM}, "num_attention_heads": 513,
                "num_key_value_heads": 513, "num_hidden_layers": 2,
                "partial_rotary_factor": 0.25,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "a non-overflowing but budget-exceeding full_kv_dim must yield an \
                 InferenceError even for an all-linear-attention config",
            )
            .to_string();
        assert!(
            err.contains("MAX_FULL_ATTENTION_DIM"),
            "wrong guard fired: {err}"
        );
    }

    /// Boundary: head_dim=2048 (MAX_HEAD_DIM) * num_attention_heads=512 == exactly
    /// MAX_FULL_ATTENTION_DIM (1,048,576) must be accepted -- guards against an off-by-one
    /// in the new budget check.
    #[test]
    fn test_full_attention_dims_at_budget_accepted_all_linear_config() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": {MAX_HEAD_DIM}, "num_attention_heads": 512,
                "num_key_value_heads": 1, "num_hidden_layers": 2,
                "partial_rotary_factor": 0.25,
                "layer_types": ["linear_attention", "linear_attention"]}}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "full_q_dim == MAX_FULL_ATTENTION_DIM must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // MoE dimension bounds (num_experts / num_experts_per_tok)
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_moe_num_experts_zero_errors() {
        let json = r#"{"text_config": {"num_experts": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("num_experts: 0 must yield an InferenceError")
            .to_string();
        assert!(err.contains("num_experts"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_moe_num_experts_per_tok_zero_errors() {
        let json = r#"{"text_config": {"num_experts": 4, "num_experts_per_tok": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("num_experts_per_tok: 0 must yield an InferenceError")
            .to_string();
        assert!(
            err.contains("num_experts_per_tok"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_moe_num_experts_per_tok_over_num_experts_errors() {
        let json = r#"{"text_config": {"num_experts": 8, "num_experts_per_tok": 9}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("num_experts_per_tok > num_experts must yield an InferenceError")
            .to_string();
        assert!(
            err.contains("num_experts_per_tok") && err.contains("num_experts"),
            "wrong guard fired: {err}"
        );
    }

    /// A tiny zero-sized checkpoint can set `num_experts` to an extreme value; downstream,
    /// `ForwardScratch::ensure_capacity` unconditionally resizes `router_logits` to
    /// `num_experts`, so this must be bound at parse time.
    #[test]
    fn test_moe_num_experts_over_max_errors() {
        let json = format!(
            r#"{{"text_config": {{"num_experts": {}, "num_experts_per_tok": 1}}}}"#,
            MAX_NUM_EXPERTS + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("num_experts above MAX_NUM_EXPERTS must yield an InferenceError")
            .to_string();
        assert!(
            err.contains("num_experts") && err.contains("MAX_NUM_EXPERTS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_moe_valid_config_accepted() {
        let json = r#"{"text_config": {"num_experts": 256, "num_experts_per_tok": 8}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "a valid MoE config must be accepted"
        );
    }

    #[test]
    fn test_moe_dense_config_unaffected() {
        // A dense (non-MoE) checkpoint leaves both MoE fields absent from config.json;
        // represent that explicitly with `null` so the struct-level `#[serde(default)]`
        // fallback (which pulls from the MoE preset) does not mask the dense case.
        let json = r#"{"text_config": {"num_experts": null, "num_experts_per_tok": null,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "linear_attention"]}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "a dense config with no MoE fields must be unaffected by the MoE dimension checks"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // intermediate size bounds (intermediate_size / moe_intermediate_size /
    // shared_expert_intermediate_size)
    // ──────────────────────────────────────────────────────────────────────

    /// `intermediate_size` is always present and drives MLP scratch buffer allocations on
    /// every forward pass; a huge value paired with zero-sized tensors would otherwise pass
    /// shape checks trivially before blowing up allocation at generation time.
    #[test]
    fn test_intermediate_size_over_max_errors() {
        let json = format!(
            r#"{{"text_config": {{"intermediate_size": {}}}}}"#,
            MAX_INTERMEDIATE_SIZE + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "intermediate_size above MAX_INTERMEDIATE_SIZE must yield an InferenceError",
            )
            .to_string();
        assert!(
            err.contains("intermediate_size") && err.contains("MAX_INTERMEDIATE_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_intermediate_size_at_max_accepted() {
        let json =
            format!(r#"{{"text_config": {{"intermediate_size": {MAX_INTERMEDIATE_SIZE}}}}}"#);
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "intermediate_size == MAX_INTERMEDIATE_SIZE must be accepted"
        );
    }

    /// `moe_intermediate_size` is gated on its own `Option` presence, independent of
    /// `num_experts` -- explicitly set it (not just inherit the MoE preset default) so this
    /// test actually exercises the if-let gate rather than silently passing through an
    /// unrelated preset value.
    #[test]
    fn test_moe_intermediate_size_over_max_errors() {
        let json = format!(
            r#"{{"text_config": {{"moe_intermediate_size": {}}}}}"#,
            MAX_INTERMEDIATE_SIZE + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "moe_intermediate_size above MAX_INTERMEDIATE_SIZE must yield an InferenceError",
            )
            .to_string();
        assert!(
            err.contains("moe_intermediate_size") && err.contains("MAX_INTERMEDIATE_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    /// Same as above for `shared_expert_intermediate_size`, explicitly present so the if-let
    /// gate is genuinely exercised rather than trivially satisfied by the preset default.
    #[test]
    fn test_shared_expert_intermediate_size_over_max_errors() {
        let json = format!(
            r#"{{"text_config": {{"shared_expert_intermediate_size": {}}}}}"#,
            MAX_INTERMEDIATE_SIZE + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "shared_expert_intermediate_size above MAX_INTERMEDIATE_SIZE must yield an \
                 InferenceError",
            )
            .to_string();
        assert!(
            err.contains("shared_expert_intermediate_size")
                && err.contains("MAX_INTERMEDIATE_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_moe_and_shared_intermediate_size_at_max_accepted() {
        let json = format!(
            r#"{{"text_config": {{"moe_intermediate_size": {MAX_INTERMEDIATE_SIZE},
                "shared_expert_intermediate_size": {MAX_INTERMEDIATE_SIZE}}}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "moe_intermediate_size / shared_expert_intermediate_size == \
             MAX_INTERMEDIATE_SIZE must be accepted"
        );
    }

    #[test]
    fn test_dense_config_no_moe_intermediate_fields_unaffected() {
        // A dense (non-MoE) checkpoint leaves both fields absent; represent that explicitly
        // with `null` so the struct-level `#[serde(default)]` fallback (which pulls from the
        // MoE preset) does not mask the dense case.
        let json = r#"{"text_config": {"moe_intermediate_size": null,
            "shared_expert_intermediate_size": null,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "linear_attention"]}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "a dense config with no MoE intermediate fields must be unaffected by the \
             MAX_INTERMEDIATE_SIZE gated checks"
        );
    }

    #[test]
    fn test_zero_linear_conv_kernel_dim_errors() {
        // `linear_conv_kernel_dim - 1` underflows usize (panics in debug, wraps to a ~16 EiB
        // allocation in release) in the GatedDeltaNet conv-buffer sizing.
        let json = r#"{"text_config": {"linear_conv_kernel_dim": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("linear_conv_kernel_dim: 0 must yield an InferenceError, not underflow")
            .to_string();
        assert!(
            err.contains("linear_conv_kernel_dim"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_zero_linear_num_key_heads_errors() {
        // gdn_fused divides `value_heads / linear_num_key_heads`; a parseable 0 is a hard
        // integer divide-by-zero panic deep in the GatedDeltaNet recurrence (#342).
        let json = r#"{"text_config": {"linear_num_key_heads": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("linear_num_key_heads: 0 must yield an InferenceError, not divide-by-zero")
            .to_string();
        assert!(
            err.contains("linear_num_key_heads"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_value_heads_below_key_heads_errors() {
        // value_heads < key_heads makes the integer `ratio = value_heads / key_heads == 0`, then
        // `h / ratio` is a divide-by-zero panic. value=1/key=16 (preset) hits ratio 0.
        let json = r#"{"text_config": {"linear_num_value_heads": 1}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("value_heads < key_heads must yield an InferenceError, not divide-by-zero")
            .to_string();
        assert!(
            err.contains("linear_num_value_heads"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_value_heads_multiple_of_key_heads_accepted() {
        // Boundary: the real GDN shape (key 16, value 32 → ratio 2) must pass the divisibility
        // guard. Guards against the new check wrongly rejecting legitimate asymmetric heads.
        let json = r#"{"text_config": {"linear_num_key_heads": 16, "linear_num_value_heads": 32}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "key 16 / value 32 (ratio 2) is a real GDN config and must be accepted"
        );
    }

    #[test]
    fn test_partial_rotary_factor_above_one_errors() {
        // rope_dim = (head_dim * factor); factor > 1 makes rope_dim exceed head_dim and
        // indexes head_vec[rope_dim/2 + i] out of bounds in apply_partial_rope.
        let json = r#"{"text_config": {"partial_rotary_factor": 3.0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("partial_rotary_factor > 1.0 must yield an InferenceError, not OOB")
            .to_string();
        assert!(
            err.contains("partial_rotary_factor"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_partial_rotary_factor_one_accepted() {
        // Boundary: factor == 1.0 makes rope_dim == head_dim (full rotary), which is in range.
        // Guards against an off-by-one in the [0.0, 1.0] range check.
        let json = r#"{"text_config": {"partial_rotary_factor": 1.0}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "partial_rotary_factor == 1.0 (full rotary) must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // rope_dim invariant tests (issue #401)
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_odd_rope_dim_errors_not_panics() {
        // Mutation contract: removing the `rope_dim < 2 || rope_dim % 2 != 0` guard must
        // make this test FAIL (the call returns Ok instead of Err).
        //
        // head_dim=10, partial_rotary_factor=0.3 → rope_dim = (10 * 0.3) as usize = 3 (odd).
        // apply_partial_rope uses half = rope_dim / 2 = 1, rotating only pair (0,1) and
        // leaving dim 2 (inside the declared rotate range) silently unrotated — wrong output.
        let json = r#"{"text_config": {"head_dim": 10, "partial_rotary_factor": 0.3}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("odd rope_dim must yield an InferenceError, not silent wrong output")
            .to_string();
        assert!(err.contains("rope_dim"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_zero_rope_dim_errors_not_panics() {
        // Mutation contract: removing the `rope_dim < 2 || rope_dim % 2 != 0` guard must
        // make this test FAIL (the call returns Ok instead of Err).
        //
        // partial_rotary_factor=0.0 → rope_dim = 0.  RopeTable::new(0, ..) gives
        // max_positions()=0, so every non-empty-sequence call to the capacity-guarded APIs
        // rejects the input rather than applying a no-op — contrary to caller expectations.
        // Reject fail-closed until a dedicated no-RoPE dispatch path exists (Refs #401).
        let json = r#"{"text_config": {"partial_rotary_factor": 0.0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("zero rope_dim must yield an InferenceError, not capacity-zero surprise")
            .to_string();
        assert!(err.contains("rope_dim"), "wrong guard fired: {err}");
    }

    // `test_rope_dim_exceeds_head_dim_via_f32_rounding_errors` removed: it targeted
    // `rope_dim() casting head_dim through f32 and rounding up past head_dim`, only
    // observable at head_dim >= 2^24 (16,777,216). `MAX_HEAD_DIM` (2048) now rejects every
    // head_dim in that range before `rope_dim()` is ever computed, so the f32-rounding edge
    // is no longer reachable through any config this parser admits. `MAX_HEAD_DIM` itself is
    // covered by `test_head_dim_over_max_rejected_all_linear_config` /
    // `test_head_dim_at_max_accepted_all_linear_config`; the `rope_dim > cfg.head_dim` line
    // stays in the source as defense-in-depth against a future `MAX_HEAD_DIM` increase.

    #[test]
    fn test_generate_config_enable_thinking_default_and_toggle() {
        let default_cfg = GenerateConfig::default();
        assert!(
            default_cfg.enable_thinking,
            "default must have thinking enabled"
        );

        let no_think = GenerateConfig {
            enable_thinking: false,
            ..GenerateConfig::default()
        };
        assert!(!no_think.enable_thinking);
        // Other fields unaffected
        assert_eq!(no_think.max_new_tokens, 256);
        assert!((no_think.temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_qwen36_27b_layer_distribution_matches_config_json() {
        // Verify compute_layer_types(64, 4) produces the pattern from config.json:
        // [linear, linear, linear, full] × 16 times
        let types = compute_layer_types(64, 4);
        for chunk_start in (0..64).step_by(4) {
            assert_eq!(types[chunk_start], LayerType::LinearAttention);
            assert_eq!(types[chunk_start + 1], LayerType::LinearAttention);
            assert_eq!(types[chunk_start + 2], LayerType::LinearAttention);
            assert_eq!(types[chunk_start + 3], LayerType::FullAttention);
        }
    }

    #[test]
    fn test_qwen35_config_backward_compat() {
        // Qwen3.5 config (no MoE fields in JSON) must still deserialize correctly.
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "vocab_size": 248320,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 0.000001,
                "intermediate_size": 6144,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "eos_token_id": 248044,
                "max_position_embeddings": 262144,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            }
        }"#;

        let cfg = Qwen35Config::from_config_json_str(json).expect("backward-compat config parses");
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.linear_num_value_heads(), 16);
        assert!(!cfg.is_moe(), "Qwen3.5 must not be detected as MoE");
        assert!(
            cfg.tie_word_embeddings,
            "default tie_word_embeddings is true"
        );
        assert_eq!(cfg.num_experts, None);
        assert_eq!(cfg.num_experts_per_tok, None);
        assert_eq!(
            cfg.mtp_num_hidden_layers, 0,
            "Qwen3.5 mtp_num_hidden_layers must default to 0"
        );
        assert!(!cfg.mtp_use_dedicated_embeddings);
    }

    #[test]
    fn qwen35_config_missing_max_position_embeddings_defaults_4096() {
        // #551: a config.json without max_position_embeddings must not
        // silently inherit the container-level Default's (qwen36_35b_a3b)
        // 262_144 context — it must resolve to the documented 4096 fallback.
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "vocab_size": 248320,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 0.000001,
                "intermediate_size": 6144,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "eos_token_id": 248044,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            }
        }"#;

        let cfg = Qwen35Config::from_config_json_str(json)
            .expect("config without max_position_embeddings still parses");
        assert_eq!(cfg.max_position_embeddings, 4096);
    }

    #[test]
    fn test_qwen35_0_8b_preset_dimensions() {
        let cfg = Qwen35Config::qwen35_0_8b();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.intermediate_size, 3584);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.rope_dim(), 64); // 256 * 0.25
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads(), 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.eos_token_id, 248_044);
        assert_eq!(cfg.max_position_embeddings, 262_144);
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        assert!(cfg.tie_word_embeddings);
        assert!(!cfg.is_moe(), "Qwen3.5-0.8B is dense, not MoE");
        // Same hybrid pattern as the 2B: [linear, linear, linear, full] x 6.
        assert_eq!(cfg.layer_types.len(), 24);
        assert_eq!(cfg.num_full_attention_layers(), 6);
        assert_eq!(cfg.num_linear_attention_layers(), 18);
    }

    #[test]
    fn test_qwen35_0_8b_config_json_fixture_parses() {
        // Parse the real released config.json (downloaded verbatim) — proves
        // from_config_json handles the 0.8B checkpoint, not just my transcription.
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen35_0_8b_config.json"
        ));
        let cfg =
            Qwen35Config::from_config_json_str(json).expect("Qwen3.5-0.8B config.json parses");

        // Core dims must match the released checkpoint.
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.intermediate_size, 3584);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads(), 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.eos_token_id, 248_044);
        assert_eq!(cfg.max_position_embeddings, 262_144);
        assert_eq!(cfg.mtp_num_hidden_layers, 1);

        // Released checkpoint is a dense vision-language model: the MoE fields must still
        // resolve to None (0.8B has no MoE), while the vision_config / token-id / mrope
        // fields below are now parsed onto the config (ADR-069 S1) instead of being dropped.
        assert!(!cfg.is_moe(), "Qwen3.5-0.8B is dense, not MoE");

        // rope_theta and partial_rotary_factor are nested under rope_parameters
        // in this checkpoint; verify they resolve to the correct values.
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert!((cfg.partial_rotary_factor - 0.25).abs() < 1e-6);
        assert_eq!(cfg.rope_dim(), 64);

        // layer_types comes from the explicit JSON array: [lin, lin, lin, full] x 6.
        assert_eq!(cfg.layer_types.len(), 24);
        assert_eq!(cfg.num_full_attention_layers(), 6);
        assert_eq!(cfg.num_linear_attention_layers(), 18);
        let full_indices: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::FullAttention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23]);

        // tie_word_embeddings is taken from the outer wrapper.
        assert!(cfg.tie_word_embeddings);

        // ADR-069 S1: vision_config, the four vision token ids, and the M-RoPE section
        // fields are top-level / text_config.rope_parameters siblings, respectively, and
        // must now round-trip onto the config (parsed but not yet consumed by forward).
        let vision = cfg
            .vision_config
            .as_ref()
            .expect("released checkpoint has a vision_config");
        assert_eq!(vision.depth, 12);
        assert_eq!(vision.hidden_size, 768);
        assert_eq!(vision.num_heads, 12);
        assert_eq!(vision.patch_size, 16);
        assert_eq!(vision.spatial_merge_size, 2);
        assert_eq!(vision.out_hidden_size, 1024);
        assert_eq!(vision.temporal_patch_size, 2);
        assert_eq!(vision.num_position_embeddings, 2304);
        assert_eq!(vision.in_channels, 3);
        assert!(vision.deepstack_visual_indexes.is_empty());

        assert_eq!(cfg.image_token_id, Some(248_056));
        assert_eq!(cfg.video_token_id, Some(248_057));
        assert_eq!(cfg.vision_start_token_id, Some(248_053));
        assert_eq!(cfg.vision_end_token_id, Some(248_054));

        let rope_params = cfg
            .rope_parameters
            .as_ref()
            .expect("released checkpoint nests rope_parameters under text_config");
        assert_eq!(rope_params.mrope_section, Some(vec![11, 11, 10]));
        assert_eq!(rope_params.mrope_interleaved, Some(true));
    }

    #[test]
    fn test_text_only_config_has_no_vision_fields() {
        // A text-only config.json (no vision_config, no token ids, no mrope fields) must
        // still parse cleanly, with every vision-language field resolving to None — proving
        // the ADR-069 S1 additions don't regress the plain-text decoder path. Also asserts an
        // existing non-vision field to prove nothing else regressed in the same parse.
        let json = r#"{
            "text_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "vocab_size": 1000,
                "intermediate_size": 2048,
                "rms_norm_eps": 1e-6,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "eos_token_id": 999,
                "max_position_embeddings": 4096
            }
        }"#;
        let cfg =
            Qwen35Config::from_config_json_str(json).expect("text-only config.json still parses");

        assert_eq!(cfg.hidden_size, 1024, "non-vision field must still parse");
        assert!(cfg.vision_config.is_none());
        assert!(cfg.image_token_id.is_none());
        assert!(cfg.video_token_id.is_none());
        assert!(cfg.vision_start_token_id.is_none());
        assert!(cfg.vision_end_token_id.is_none());
        let rope_params = cfg.rope_parameters.clone().unwrap_or_default();
        assert!(rope_params.mrope_section.is_none());
        assert!(rope_params.mrope_interleaved.is_none());
    }

    /// Minimal text_config JSON (parses cleanly on its own, per
    /// `test_text_only_config_has_no_vision_fields`) plus a top-level `vision_config`
    /// object with every field valid except the ones the caller overrides.
    fn config_json_with_vision(vision_config_body: &str) -> String {
        format!(
            r#"{{
                "text_config": {{
                    "hidden_size": 1024,
                    "num_hidden_layers": 4,
                    "vocab_size": 1000,
                    "intermediate_size": 2048,
                    "rms_norm_eps": 1e-6,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 16,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "full_attention_interval": 4,
                    "eos_token_id": 999,
                    "max_position_embeddings": 4096
                }},
                "vision_config": {vision_config_body}
            }}"#
        )
    }

    #[test]
    fn parser_rejects_present_vision_config_with_depth_zero() {
        // A present-but-malformed vision_config (depth: 0) is syntactically valid JSON
        // and, before this fix, would silently load a truncated tensor set later. It must
        // be rejected here, at parse time.
        let json = config_json_with_vision(
            r#"{
                "depth": 0,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }"#,
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("depth: 0 vision_config must be rejected at parse time");
        assert!(
            err.to_string().contains("depth"),
            "error must name depth: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_num_heads_zero() {
        let json = config_json_with_vision(
            r#"{
                "depth": 12,
                "hidden_size": 768,
                "num_heads": 0,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }"#,
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("num_heads: 0 vision_config must be rejected at parse time");
        assert!(
            err.to_string().contains("num_heads"),
            "error must name num_heads: {err}"
        );
    }

    /// The vision checkpoint loader mints ~12 tensor-name Strings per ViT block *before* any
    /// tensor validation runs; an unbounded `depth` (previously only checked nonzero) drives
    /// unbounded String allocation ahead of any shape check.
    #[test]
    fn parser_rejects_present_vision_config_with_depth_over_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": {},
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
            MAX_VISION_DEPTH + 1
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("depth above MAX_VISION_DEPTH must be rejected at parse time")
            .to_string();
        assert!(
            err.contains("depth") && err.contains("MAX_VISION_DEPTH"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_present_vision_config_with_depth_at_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": {MAX_VISION_DEPTH},
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
        ));
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "depth == MAX_VISION_DEPTH must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // config vector length bounds (mrope_section / deepstack_visual_indexes)
    // ──────────────────────────────────────────────────────────────────────

    /// `RopeParams::mrope_section` is deserialized as a `Vec<usize>` before any validation
    /// runs; a huge declared length must be rejected at parse time rather than reaching an
    /// uncontrolled allocation, even though the field is not yet consumed by the forward pass.
    #[test]
    fn parser_rejects_present_mrope_section_over_max() {
        let huge: Vec<String> = (0..MAX_CONFIG_VECTOR_LEN + 1)
            .map(|_| "0".to_string())
            .collect();
        let json = format!(
            r#"{{"text_config": {{"rope_parameters": {{"mrope_section": [{}]}}}}}}"#,
            huge.join(",")
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("mrope_section length above MAX_CONFIG_VECTOR_LEN must be rejected")
            .to_string();
        assert!(
            err.contains("mrope_section") && err.contains("MAX_CONFIG_VECTOR_LEN"),
            "wrong guard fired: {err}"
        );
    }

    /// Accept-case control: a realistic `mrope_section` (one entry per M-RoPE axis) must
    /// still parse cleanly.
    #[test]
    fn parser_accepts_present_mrope_section_realistic_size() {
        let json = r#"{"text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}}}"#;
        let cfg = Qwen35Config::from_config_json_str(json)
            .expect("realistic mrope_section must be accepted");
        assert_eq!(
            cfg.rope_parameters
                .expect("rope_parameters present")
                .mrope_section,
            Some(vec![11, 11, 10])
        );
    }

    /// `VisionModelConfig::deepstack_visual_indexes` is deserialized as a `Vec<usize>` before
    /// any validation runs; a huge declared length must be rejected at parse time regardless
    /// of whether the indexes are ever consumed.
    #[test]
    fn parser_rejects_present_deepstack_visual_indexes_over_max() {
        let huge: Vec<String> = (0..MAX_CONFIG_VECTOR_LEN + 1)
            .map(|_| "0".to_string())
            .collect();
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 12,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3,
                "deepstack_visual_indexes": [{}]
            }}"#,
            huge.join(",")
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "deepstack_visual_indexes length above MAX_CONFIG_VECTOR_LEN must be rejected",
            )
            .to_string();
        assert!(
            err.contains("deepstack_visual_indexes") && err.contains("MAX_CONFIG_VECTOR_LEN"),
            "wrong guard fired: {err}"
        );
    }

    /// Accept-case control: a realistic `deepstack_visual_indexes` (a handful of in-range
    /// layer indexes) must still parse cleanly.
    #[test]
    fn parser_accepts_present_deepstack_visual_indexes_realistic_size() {
        let json = config_json_with_vision(
            r#"{
                "depth": 12,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3,
                "deepstack_visual_indexes": [2, 5, 8]
            }"#,
        );
        let cfg = Qwen35Config::from_config_json_str(&json)
            .expect("realistic deepstack_visual_indexes must be accepted");
        assert_eq!(
            cfg.vision_config
                .expect("vision_config present")
                .deepstack_visual_indexes,
            vec![2, 5, 8]
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // global linear-attention output-dim budget (independent of layer mix)
    // ──────────────────────────────────────────────────────────────────────

    /// An all-full-attention config never reaches `load_linear_attention_weights`'s per-layer
    /// `checked_linear_output_dim` derivation, yet `ForwardScratch::ensure_capacity`
    /// unconditionally derives `linear_output_dim()` for every config. A
    /// `linear_num_value_heads` of `usize::MAX` must overflow-reject here rather than
    /// panicking/wrapping downstream.
    #[test]
    fn parser_rejects_all_full_attention_config_with_overflowing_linear_output_dim() {
        let json = format!(
            r#"{{"text_config": {{
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 1,
            "linear_value_head_dim": {}
        }}}}"#,
            usize::MAX
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "overflowing linear_value_head_dim on an all-full-attention config must be \
                 rejected before any allocation",
            )
            .to_string();
        assert!(
            err.contains("output_dim") && err.contains("overflows"),
            "wrong guard fired: {err}"
        );
    }

    /// Same all-full-attention scenario, but with a huge, non-overflowing product that must
    /// still be rejected against `MAX_LINEAR_OUTPUT_DIM`.
    #[test]
    fn parser_rejects_all_full_attention_config_with_linear_output_dim_over_max() {
        // `linear_num_value_heads` sits at its own `MAX_LINEAR_NUM_VALUE_HEADS` cap (so that
        // guard does not fire first); `linear_value_head_dim` is the free factor that pushes
        // the `linear_output_dim` product over `MAX_LINEAR_OUTPUT_DIM` while each individual
        // factor stays within its own cap.
        let value_heads = MAX_LINEAR_NUM_VALUE_HEADS;
        let value_dim = (MAX_LINEAR_OUTPUT_DIM / value_heads) + 1;
        let json = format!(
            r#"{{"text_config": {{
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "full_attention"],
                "linear_num_key_heads": 1,
                "linear_num_value_heads": {value_heads},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": {value_dim}
            }}}}"#
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "linear_output_dim above MAX_LINEAR_OUTPUT_DIM on an all-full-attention config \
                 must be rejected",
            )
            .to_string();
        assert!(
            err.contains("output_dim") && err.contains("MAX_LINEAR_OUTPUT_DIM"),
            "wrong guard fired: {err}"
        );
    }

    /// Accept-case control: realistic linear-attention dims on an all-full-attention config
    /// must still parse cleanly -- the global budget must not reject real geometry.
    #[test]
    fn parser_accepts_all_full_attention_config_with_realistic_linear_dims() {
        let json = r#"{"text_config": {
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128
        }}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "realistic linear-attention dims on an all-full-attention config must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // KV-cache layer-count invariant tests (issue #170)
    // ──────────────────────────────────────────────────────────────────────

    /// KV cache layer count is the number of full-attention layers, NOT all layers.
    ///
    /// This is the primary regression guard: a silent switch to all-24-layer KV
    /// allocation would change `kv_cache_layer_count()` from 6 to 24 and cause
    /// this test to fail.
    #[test]
    fn kv_layer_count_excludes_linear_layers() {
        let cfg = Qwen35Config::qwen35_0_8b();
        assert_eq!(
            cfg.kv_cache_layer_count(),
            6,
            "must be full-attention count"
        );
        assert_ne!(
            cfg.kv_cache_layer_count(),
            cfg.num_hidden_layers,
            "kv_cache_layer_count must not equal num_hidden_layers (would 4× decode memory)"
        );
    }

    /// Full + linear layers must sum to total hidden layers for every preset.
    ///
    /// `compute_layer_types` produces exactly `num_hidden_layers` entries, each
    /// either FullAttention or LinearAttention, so this invariant must always hold.
    /// If any preset fails here it indicates a config bug.
    #[test]
    fn full_plus_linear_equals_total() {
        for (name, cfg) in [
            ("qwen35_0_8b", Qwen35Config::qwen35_0_8b()),
            ("qwen35_2b", Qwen35Config::qwen35_2b()),
            ("qwen36_35b_a3b", Qwen35Config::qwen36_35b_a3b()),
            ("qwen36_27b", Qwen35Config::qwen36_27b()),
        ] {
            assert_eq!(
                cfg.num_full_attention_layers() + cfg.num_linear_attention_layers(),
                cfg.num_hidden_layers,
                "{name}: full + linear must equal num_hidden_layers"
            );
        }
    }

    /// KV bytes per token for f16 matches the expected 12_288 B for qwen3.5-0.8B.
    ///
    /// Formula: `num_full(6) × 2(K+V) × full_kv_dim(512) × dtype_bytes(2) = 12_288`.
    /// At a 4096-token context this is 48 MiB.
    #[test]
    fn kv_bytes_per_token_f16() {
        let cfg = Qwen35Config::qwen35_0_8b();
        // 6 layers × 2 (K+V) × 512 (kv_dim) × 2 (f16) = 12_288 B/token = 48 MiB @ 4096 ctx
        assert_eq!(cfg.kv_bytes_per_token(2), 12_288);
    }

    /// Numeric identity: kv_bytes_per_token(1) == num_hidden_layers * head_dim for 0.8B.
    ///
    /// `6 × 2 × 512 × 1 = 6_144 = 24 × 256`. This is a coincidence specific to
    /// the 0.8B parameters (full_attention_interval=4, num_kv_heads=2); it does
    /// NOT generalise across configs and must not be used as a formula.
    #[test]
    fn kv_bytes_per_token_identity() {
        let cfg = Qwen35Config::qwen35_0_8b();
        assert_eq!(cfg.kv_bytes_per_token(1), 6_144);
        // Coincidence: matches num_hidden_layers × head_dim for this specific preset only.
        assert_eq!(
            cfg.kv_bytes_per_token(1),
            cfg.num_hidden_layers * cfg.head_dim
        );
    }

    // ── decode_cap unit tests ────────────────────────────────────────────────

    #[test]
    fn decode_cap_none_budget_returns_max() {
        // Disabled path must be byte-identical to the pre-budget behaviour.
        assert_eq!(decode_cap(None, 512), 512);
        assert_eq!(decode_cap(None, 0), 0);
    }

    #[test]
    fn decode_cap_zero_budget_returns_max() {
        // Some(0) is treated as disabled.
        assert_eq!(decode_cap(Some(0), 512), 512);
        assert_eq!(decode_cap(Some(0), 1), 1);
    }

    #[test]
    fn decode_cap_nonzero_budget_adds_budgets() {
        // Worst case = rb + max_new_tokens + 1 (the +1 is the forced </think> delimiter).
        // Mutation-sensitive: revert to rb+max and these assertions fail.
        assert_eq!(decode_cap(Some(2048), 512), 2561);
        assert_eq!(decode_cap(Some(1), 1), 3);
        assert_eq!(decode_cap(Some(100), 200), 301);
    }

    #[test]
    fn decode_cap_saturates_on_overflow() {
        // saturating_add must not wrap on usize::MAX inputs.
        assert_eq!(decode_cap(Some(usize::MAX), 1), usize::MAX);
        assert_eq!(decode_cap(Some(1), usize::MAX), usize::MAX);
    }

    // ── force_close_think unit tests ────────────────────────────────────────

    #[test]
    fn force_close_think_disabled_when_budget_none() {
        // budget=None → always returns None regardless of other args.
        assert_eq!(
            force_close_think(None, true, false, 100, Some(99)),
            None,
            "None budget must disable forcing"
        );
    }

    #[test]
    fn force_close_think_disabled_when_budget_zero() {
        // budget=Some(0) → budget > 0 guard fails → None.
        assert_eq!(
            force_close_think(Some(0), true, false, 100, Some(99)),
            None,
            "budget=0 must disable forcing"
        );
    }

    #[test]
    fn force_close_think_disabled_when_enable_thinking_false() {
        // enable_thinking=false → no reasoning block → forcing is a no-op.
        assert_eq!(
            force_close_think(Some(10), false, false, 20, Some(99)),
            None,
            "enable_thinking=false must disable forcing"
        );
    }

    #[test]
    fn force_close_think_disabled_when_already_closed() {
        // thinking_closed=true → block already closed → should not force again.
        assert_eq!(
            force_close_think(Some(10), true, true, 20, Some(99)),
            None,
            "already-closed thinking block must not force again"
        );
    }

    #[test]
    fn force_close_think_disabled_when_close_id_none() {
        // close_id=None → model has no </think> token → forcing is a no-op.
        assert_eq!(
            force_close_think(Some(10), true, false, 20, None),
            None,
            "close_id=None must disable forcing"
        );
    }

    #[test]
    fn force_close_think_fires_at_budget_boundary() {
        let close_id = 248_069_u32;
        // generated_so_far == budget → should force (mutation: >= not >).
        assert_eq!(
            force_close_think(Some(10), true, false, 10, Some(close_id)),
            Some(close_id),
            "must force when generated_so_far equals budget"
        );
        // generated_so_far > budget → also forces.
        assert_eq!(
            force_close_think(Some(10), true, false, 11, Some(close_id)),
            Some(close_id),
            "must force when generated_so_far exceeds budget"
        );
    }

    #[test]
    fn force_close_think_does_not_fire_before_budget() {
        let close_id = 248_069_u32;
        // generated_so_far < budget → must NOT force (mutation-sensitive boundary).
        assert_eq!(
            force_close_think(Some(10), true, false, 9, Some(close_id)),
            None,
            "must not force when generated_so_far is one below budget"
        );
        assert_eq!(
            force_close_think(Some(10), true, false, 0, Some(close_id)),
            None,
            "must not force when zero tokens generated"
        );
    }

    // -- from_model_dir: the single shared config-resolution policy (#923) --

    #[test]
    fn from_model_dir_errors_on_missing_config_json() {
        let tmp = tempfile::tempdir().unwrap();
        // Directory exists but has no config.json at all.
        let err = Qwen35Config::from_model_dir(tmp.path())
            .expect_err("a directory with no config.json must be a hard error");
        let msg = err.to_string();
        assert!(
            msg.contains("config.json"),
            "error must name the missing file: {msg}"
        );
        assert!(
            msg.contains(&tmp.path().display().to_string()),
            "error must name the offending directory: {msg}"
        );
    }

    #[test]
    fn from_model_dir_loads_a_real_config_json() {
        let tmp = tempfile::tempdir().unwrap();
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen35_0_8b_config.json"
        ));
        std::fs::write(tmp.path().join("config.json"), json).unwrap();

        let cfg = Qwen35Config::from_model_dir(tmp.path())
            .expect("a directory with a valid config.json must load");
        assert_eq!(
            cfg.hidden_size, 1024,
            "must parse the real 0.8B config, not a preset"
        );
    }

    #[test]
    fn from_model_dir_propagates_a_malformed_config_json_parse_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), "not valid json {{{").unwrap();

        let err = Qwen35Config::from_model_dir(tmp.path())
            .expect_err("malformed config.json must still be a parse error, not a preset");
        assert!(
            err.to_string().contains("config.json") || err.to_string().contains("invalid Qwen"),
            "error must reflect the parse failure: {err}"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // GatedDeltaNet state-matrix three-factor budget (MAX_GDN_STATE_SIZE)
    // ──────────────────────────────────────────────────────────────────────

    /// `MAX_LINEAR_OUTPUT_DIM` only bounds `value_heads * linear_value_head_dim`;
    /// `linear_key_head_dim` is a free multiplier on top of it in `attention/gdn.rs`'s
    /// `s_matrices` allocation. `value_heads = 1`, `linear_value_head_dim` just under the
    /// `MAX_LINEAR_OUTPUT_DIM` cap, and a huge `linear_key_head_dim` must pass
    /// `MAX_LINEAR_OUTPUT_DIM` (linear_output_dim = value_dim, well under the cap) while still
    /// being rejected by the three-factor `MAX_GDN_STATE_SIZE` budget.
    #[test]
    fn parser_rejects_gdn_state_size_over_max_with_small_output_dim() {
        let json = r#"{"text_config": {
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 1000000,
            "linear_value_head_dim": 128
        }}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err(
                "a huge linear_key_head_dim must be rejected by MAX_GDN_STATE_SIZE even though \
                 linear_output_dim (value_heads * linear_value_head_dim) is tiny",
            )
            .to_string();
        assert!(
            err.contains("GatedDeltaNet state size") && err.contains("MAX_GDN_STATE_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    /// Accept-case control: realistic GDN geometry (matching the preset) must still parse.
    #[test]
    fn parser_accepts_realistic_gdn_state_size() {
        let json = r#"{"text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128
        }}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "realistic GDN geometry must be accepted"
        );
    }

    /// Zero `linear_key_head_dim` is a degenerate zero-width GDN recurrence.
    #[test]
    fn parser_rejects_zero_linear_key_head_dim() {
        let json = r#"{"text_config": {"linear_key_head_dim": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("linear_key_head_dim: 0 must be rejected")
            .to_string();
        assert!(
            err.contains("linear_key_head_dim"),
            "wrong guard fired: {err}"
        );
    }

    /// Zero `linear_value_head_dim` is a degenerate zero-width GDN recurrence, and would
    /// otherwise collapse `linear_output_dim()` to zero -- passing `MAX_LINEAR_OUTPUT_DIM`
    /// trivially rather than being caught by it.
    #[test]
    fn parser_rejects_zero_linear_value_head_dim() {
        let json = r#"{"text_config": {"linear_value_head_dim": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("linear_value_head_dim: 0 must be rejected")
            .to_string();
        assert!(
            err.contains("linear_value_head_dim"),
            "wrong guard fired: {err}"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // hidden_size upper bound (MAX_HIDDEN_SIZE)
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn parser_rejects_hidden_size_over_max() {
        let json = format!(
            r#"{{"text_config": {{"hidden_size": {}}}}}"#,
            MAX_HIDDEN_SIZE + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("hidden_size above MAX_HIDDEN_SIZE must be rejected")
            .to_string();
        assert!(
            err.contains("hidden_size") && err.contains("MAX_HIDDEN_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_hidden_size_at_max() {
        // hidden_size alone (with vocab_size at its own realistic-preset value) does not
        // reach MAX_EMBEDDING_BYTES -- unlike the vision "_at_max" cases below, the text
        // embedding budget is generous enough (32 GiB) to keep this at-max-in-isolation case
        // accepted. See MAX_EMBEDDING_BYTES docs.
        let json =
            format!(r#"{{"text_config": {{"hidden_size": {MAX_HIDDEN_SIZE}, "vocab_size": 1}}}}"#);
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "hidden_size == MAX_HIDDEN_SIZE must be accepted when paired with a small vocab_size"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // attention head-count upper bound (MAX_ATTENTION_HEADS)
    // ──────────────────────────────────────────────────────────────────────

    /// `num_attention_heads` drives `scores = num_attention_heads * (max_kv_len + 1)`
    /// independent of `head_dim`; a small `head_dim` (1) paired with a huge
    /// `num_attention_heads` still passes `MAX_FULL_ATTENTION_DIM` (the product stays small)
    /// so `MAX_ATTENTION_HEADS` must fire on its own.
    #[test]
    fn parser_rejects_num_attention_heads_over_max() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": 1, "num_attention_heads": {}, "num_key_value_heads": 1}}}}"#,
            MAX_ATTENTION_HEADS + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("num_attention_heads above MAX_ATTENTION_HEADS must be rejected")
            .to_string();
        assert!(
            err.contains("num_attention_heads") && err.contains("MAX_ATTENTION_HEADS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_num_attention_heads_at_max() {
        // head_dim=8 with the default partial_rotary_factor (0.25) gives rope_dim=2, the
        // smallest value that clears the `rope_dim >= 2` guard -- head_dim=1 would derive
        // rope_dim=0 and fail that unrelated check instead of exercising this bound.
        let json = format!(
            r#"{{"text_config": {{"head_dim": 8, "num_attention_heads": {MAX_ATTENTION_HEADS}, "num_key_value_heads": 1}}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "num_attention_heads == MAX_ATTENTION_HEADS must be accepted"
        );
    }

    /// `num_attention_heads` must be divisible by `num_key_value_heads`, which forces
    /// `num_attention_heads >= num_key_value_heads` -- so isolating this bound requires
    /// `num_attention_heads` to stay within its own budget while `num_key_value_heads` alone
    /// exceeds it (the reverse combination is unreachable: an out-of-budget
    /// `num_key_value_heads` would always drag `num_attention_heads` out of budget too).
    #[test]
    fn parser_rejects_num_key_value_heads_over_max() {
        let json = format!(
            r#"{{"text_config": {{"head_dim": 1, "num_attention_heads": {MAX_ATTENTION_HEADS}, "num_key_value_heads": {}}}}}"#,
            MAX_ATTENTION_HEADS + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("num_key_value_heads above MAX_ATTENTION_HEADS must be rejected")
            .to_string();
        assert!(
            err.contains("num_key_value_heads") && err.contains("MAX_ATTENTION_HEADS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_num_key_value_heads_at_max() {
        // head_dim=8: see parser_accepts_num_attention_heads_at_max for why not 1.
        let json = format!(
            r#"{{"text_config": {{"head_dim": 8, "num_attention_heads": {MAX_ATTENTION_HEADS}, "num_key_value_heads": {MAX_ATTENTION_HEADS}}}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "num_key_value_heads == MAX_ATTENTION_HEADS must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // num_experts_per_tok upper bound independent of num_experts presence
    // ──────────────────────────────────────────────────────────────────────

    /// `router_selected.resize(cfg.num_experts_per_tok.unwrap_or(0), ..)` runs regardless of
    /// whether `num_experts` is present; a checkpoint that sets `num_experts_per_tok` while
    /// leaving `num_experts` unset must still be bounded.
    #[test]
    fn parser_rejects_num_experts_per_tok_over_max_without_num_experts() {
        let json = format!(
            r#"{{"text_config": {{"num_experts": null, "num_experts_per_tok": {}}}}}"#,
            MAX_NUM_EXPERTS + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "num_experts_per_tok above MAX_NUM_EXPERTS must be rejected even when \
                 num_experts is absent",
            )
            .to_string();
        assert!(
            err.contains("num_experts_per_tok") && err.contains("MAX_NUM_EXPERTS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_num_experts_per_tok_at_max_without_num_experts() {
        let json = format!(
            r#"{{"text_config": {{"num_experts": null, "num_experts_per_tok": {MAX_NUM_EXPERTS}}}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "num_experts_per_tok == MAX_NUM_EXPERTS must be accepted when num_experts is absent"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // nonzero lower bounds: intermediate_size / moe / shared-expert intermediate sizes
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn parser_rejects_zero_intermediate_size() {
        let json = r#"{"text_config": {"intermediate_size": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("intermediate_size: 0 must be rejected")
            .to_string();
        assert!(
            err.contains("intermediate_size"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_zero_moe_intermediate_size() {
        let json = r#"{"text_config": {"moe_intermediate_size": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("moe_intermediate_size: 0 must be rejected")
            .to_string();
        assert!(
            err.contains("moe_intermediate_size"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_zero_shared_expert_intermediate_size() {
        let json = r#"{"text_config": {"shared_expert_intermediate_size": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("shared_expert_intermediate_size: 0 must be rejected")
            .to_string();
        assert!(
            err.contains("shared_expert_intermediate_size"),
            "wrong guard fired: {err}"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // max_position_embeddings upper bound (MAX_POSITION_EMBEDDINGS)
    // ──────────────────────────────────────────────────────────────────────

    /// Drives the Metal RoPE table allocation (`rope_max * rope_dim / 2`), independent of
    /// `head_dim`.
    #[test]
    fn parser_rejects_max_position_embeddings_over_max() {
        // `head_dim`/`partial_rotary_factor` pinned to the smallest valid `rope_dim` (2) so
        // this trips only the `max_position_embeddings`-specific guard, not the
        // `MAX_ROPE_TABLE_BYTES` product budget (which the default preset's `rope_dim = 64`
        // would otherwise reach first at this `max_position_embeddings`).
        let json = format!(
            r#"{{"text_config": {{"max_position_embeddings": {}, "head_dim": 2, "partial_rotary_factor": 1.0}}}}"#,
            MAX_POSITION_EMBEDDINGS + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("max_position_embeddings above MAX_POSITION_EMBEDDINGS must be rejected")
            .to_string();
        assert!(
            err.contains("max_position_embeddings") && err.contains("MAX_POSITION_EMBEDDINGS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_max_position_embeddings_at_max() {
        let json = format!(
            r#"{{"text_config": {{"max_position_embeddings": {MAX_POSITION_EMBEDDINGS}}}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "max_position_embeddings == MAX_POSITION_EMBEDDINGS must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // linear_num_key_heads upper bound (MAX_LINEAR_NUM_KEY_HEADS)
    // ──────────────────────────────────────────────────────────────────────

    /// `linear_num_key_heads` was previously checked only for zero; it is a free multiplier
    /// in `linear_qkv_dim()` unconstrained by `MAX_LINEAR_OUTPUT_DIM` / `MAX_GDN_STATE_SIZE`
    /// (neither product carries it as a factor).
    #[test]
    fn parser_rejects_linear_num_key_heads_over_max() {
        let json = format!(
            r#"{{"text_config": {{
                "linear_num_key_heads": {over},
                "linear_num_value_heads": {over},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1
            }}}}"#,
            over = MAX_LINEAR_NUM_KEY_HEADS + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("linear_num_key_heads above MAX_LINEAR_NUM_KEY_HEADS must be rejected")
            .to_string();
        assert!(
            err.contains("linear_num_key_heads") && err.contains("MAX_LINEAR_NUM_KEY_HEADS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_linear_num_key_heads_at_max() {
        let json = format!(
            r#"{{"text_config": {{
                "linear_num_key_heads": {MAX_LINEAR_NUM_KEY_HEADS},
                "linear_num_value_heads": {MAX_LINEAR_NUM_KEY_HEADS},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1
            }}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "linear_num_key_heads == MAX_LINEAR_NUM_KEY_HEADS must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // linear_conv_kernel_dim upper bound (MAX_CONV_KERNEL_DIM)
    // ──────────────────────────────────────────────────────────────────────

    /// `linear_conv_kernel_dim` was previously checked only for zero (underflow guard on
    /// `linear_conv_kernel_dim - 1`); it directly scales the GatedDeltaNet conv1d buffer.
    #[test]
    fn parser_rejects_linear_conv_kernel_dim_over_max() {
        let json = format!(
            r#"{{"text_config": {{"linear_conv_kernel_dim": {}}}}}"#,
            MAX_CONV_KERNEL_DIM + 1
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("linear_conv_kernel_dim above MAX_CONV_KERNEL_DIM must be rejected")
            .to_string();
        assert!(
            err.contains("linear_conv_kernel_dim") && err.contains("MAX_CONV_KERNEL_DIM"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_linear_conv_kernel_dim_at_max() {
        let json =
            format!(r#"{{"text_config": {{"linear_conv_kernel_dim": {MAX_CONV_KERNEL_DIM}}}}}"#);
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "linear_conv_kernel_dim == MAX_CONV_KERNEL_DIM must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // GatedDeltaNet conv buffer two-factor product budget (MAX_GDN_CONV_BUFFER_SIZE)
    // ──────────────────────────────────────────────────────────────────────

    /// `MAX_LINEAR_NUM_KEY_HEADS`, `MAX_CONV_KERNEL_DIM`, and `MAX_GDN_STATE_SIZE` each bound
    /// their own factor and are individually satisfied here (`linear_num_key_heads` and
    /// `linear_conv_kernel_dim` sit exactly at their caps; `linear_num_value_heads *
    /// linear_key_head_dim * linear_value_head_dim` sits exactly at `MAX_GDN_STATE_SIZE`), yet
    /// the conv buffer element count (`linear_qkv_dim() * (linear_conv_kernel_dim - 1)`) still
    /// exceeds `MAX_GDN_CONV_BUFFER_SIZE` by roughly three orders of magnitude -- demonstrating
    /// the product-hostility the per-factor caps above do not close on their own.
    #[test]
    fn parser_rejects_gdn_conv_buffer_size_over_max_with_bounded_factors() {
        // `linear_num_key_heads` / `linear_num_value_heads` sit at their own caps and
        // `linear_key_head_dim` / `linear_value_head_dim` are kept small enough that neither
        // `MAX_GDN_STATE_SIZE` nor `MAX_GDN_CHUNK_SCRATCH_BYTES` fires first -- only the
        // `linear_conv_kernel_dim` product (`linear_qkv_dim() * (kernel_dim - 1)`) is over
        // `MAX_GDN_CONV_BUFFER_SIZE`.
        let json = format!(
            r#"{{"text_config": {{
                "linear_num_key_heads": {MAX_LINEAR_NUM_KEY_HEADS},
                "linear_num_value_heads": {MAX_LINEAR_NUM_VALUE_HEADS},
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": {MAX_CONV_KERNEL_DIM}
            }}}}"#
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "a conv buffer product over MAX_GDN_CONV_BUFFER_SIZE must be rejected even \
                 though every individual factor is at or under its own cap",
            )
            .to_string();
        // CLASS A2 (round p): the aggregate MAX_GDN_SESSION_BYTES check runs earlier in
        // `validate()` than the MAX_GDN_CONV_BUFFER_SIZE check this test originally targeted,
        // and this same hostile geometry now also exceeds the aggregate budget -- so the
        // aggregate guard fires first. Both are legitimate rejections of the same hostile
        // input; accept either.
        assert!(
            (err.contains("GatedDeltaNet conv buffer size")
                && err.contains("MAX_GDN_CONV_BUFFER_SIZE"))
                || err.contains("MAX_GDN_SESSION_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    /// Accept-case control: realistic GDN geometry (matching the preset) must still parse.
    #[test]
    fn parser_accepts_realistic_gdn_conv_buffer_size() {
        let json = r#"{"text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4
        }}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "realistic GDN conv buffer geometry must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // RoPE table two-factor byte budget (MAX_ROPE_TABLE_BYTES)
    // ──────────────────────────────────────────────────────────────────────

    /// `MAX_POSITION_EMBEDDINGS` bounds `max_position_embeddings` and `MAX_HEAD_DIM` bounds
    /// `rope_dim` transitively, both satisfied here (`max_position_embeddings` at its cap,
    /// `head_dim` at its cap with `partial_rotary_factor = 1.0`), yet the RoPE table byte
    /// count (`4 * max_position_embeddings * rope_dim`) exceeds `MAX_ROPE_TABLE_BYTES` by
    /// roughly 32x -- demonstrating the product-hostility neither per-factor cap closes alone.
    #[test]
    fn parser_rejects_rope_table_bytes_over_max_with_bounded_factors() {
        let json = format!(
            r#"{{"text_config": {{
                "max_position_embeddings": {MAX_POSITION_EMBEDDINGS},
                "head_dim": {MAX_HEAD_DIM},
                "partial_rotary_factor": 1.0
            }}}}"#
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "a RoPE table byte count over MAX_ROPE_TABLE_BYTES must be rejected even \
                 though max_position_embeddings and head_dim are each at or under their own cap",
            )
            .to_string();
        assert!(
            err.contains("RoPE table size") && err.contains("MAX_ROPE_TABLE_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    /// Accept-case control: realistic RoPE geometry (matching a real Qwen3.5/3.6 preset) must
    /// still parse.
    #[test]
    fn parser_accepts_realistic_rope_table_bytes() {
        let json = r#"{"text_config": {
            "max_position_embeddings": 262144,
            "head_dim": 256,
            "partial_rotary_factor": 0.25
        }}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "realistic RoPE table geometry must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // vision_config numeric field upper bounds
    // ──────────────────────────────────────────────────────────────────────

    /// Previously only overflow-guarded (via `checked_derived_sizes`); a huge-but-non-
    /// overflowing `hidden_size` reaches those tensor-size derivations unchecked.
    #[test]
    fn parser_rejects_present_vision_config_with_hidden_size_over_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": {},
                "num_heads": 1,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
            MAX_VISION_HIDDEN_SIZE + 1
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("hidden_size above MAX_VISION_HIDDEN_SIZE must be rejected")
            .to_string();
        assert!(
            err.contains("hidden_size") && err.contains("MAX_VISION_HIDDEN_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_hidden_size_at_max_over_byte_budget() {
        // CLASS A3 (round p): MAX_VISION_HIDDEN_SIZE alone is still generous (matching
        // MAX_HIDDEN_SIZE's scale), but composed into the merger fc1/fc2 and patch-embed
        // products, hidden_size at its own per-field cap now exceeds MAX_VISION_TENSOR_BYTES
        // even with every other field realistic -- the per-field cap is necessary but not
        // sufficient; the byte budget is the binding constraint. See MAX_VISION_TENSOR_BYTES
        // docs.
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": {MAX_VISION_HIDDEN_SIZE},
                "num_heads": 1,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("hidden_size == MAX_VISION_HIDDEN_SIZE must be rejected by the byte budget")
            .to_string();
        assert!(
            err.contains("MAX_VISION_TENSOR_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_num_heads_over_max() {
        let over = MAX_VISION_NUM_HEADS + 1;
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": {over},
                "num_heads": {over},
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("num_heads above MAX_VISION_NUM_HEADS must be rejected")
            .to_string();
        assert!(
            err.contains("num_heads") && err.contains("MAX_VISION_NUM_HEADS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_present_vision_config_with_num_heads_at_max() {
        // num_heads itself does not feed any `checked_derived_sizes` product (only
        // hidden_size % num_heads == 0 is checked), so it stays accepted at its own cap as
        // long as hidden_size (here set to a realistic divisible value, not also maxed) keeps
        // every derived tensor under MAX_VISION_TENSOR_BYTES.
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": {MAX_VISION_NUM_HEADS},
                "num_heads": {MAX_VISION_NUM_HEADS},
                "patch_size": 4,
                "spatial_merge_size": 1,
                "out_hidden_size": 1024,
                "temporal_patch_size": 1,
                "num_position_embeddings": 2304,
                "in_channels": 1
            }}"#
        ));
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "num_heads == MAX_VISION_NUM_HEADS must be accepted"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_patch_size_over_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": {},
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
            MAX_VISION_PATCH_SIZE + 1
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("patch_size above MAX_VISION_PATCH_SIZE must be rejected")
            .to_string();
        assert!(
            err.contains("patch_size") && err.contains("MAX_VISION_PATCH_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_patch_size_at_max_over_byte_budget() {
        // CLASS A3 (round p): patch_size is squared in patch_embed_weight; even at a
        // realistic hidden_size, patch_size at its own generous per-field cap overflows
        // MAX_VISION_TENSOR_BYTES. See MAX_VISION_TENSOR_BYTES docs.
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": {MAX_VISION_PATCH_SIZE},
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("patch_size == MAX_VISION_PATCH_SIZE must be rejected by the byte budget")
            .to_string();
        assert!(
            err.contains("MAX_VISION_TENSOR_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_spatial_merge_size_over_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": {},
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
            MAX_VISION_SPATIAL_MERGE_SIZE + 1
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("spatial_merge_size above MAX_VISION_SPATIAL_MERGE_SIZE must be rejected")
            .to_string();
        assert!(
            err.contains("spatial_merge_size") && err.contains("MAX_VISION_SPATIAL_MERGE_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_spatial_merge_size_at_max_over_byte_budget() {
        // CLASS A3 (round p): spatial_merge_size is squared into merge_in, which is then
        // squared again for merger_fc1 -- a quartic blowup that overflows
        // MAX_VISION_TENSOR_BYTES at the field's own per-field cap even with everything else
        // realistic. See MAX_VISION_TENSOR_BYTES docs.
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": {MAX_VISION_SPATIAL_MERGE_SIZE},
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "spatial_merge_size == MAX_VISION_SPATIAL_MERGE_SIZE must be rejected by the \
                 byte budget",
            )
            .to_string();
        assert!(
            err.contains("MAX_VISION_TENSOR_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_out_hidden_size_over_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": {},
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
            MAX_VISION_OUT_HIDDEN_SIZE + 1
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("out_hidden_size above MAX_VISION_OUT_HIDDEN_SIZE must be rejected")
            .to_string();
        assert!(
            err.contains("out_hidden_size") && err.contains("MAX_VISION_OUT_HIDDEN_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_out_hidden_size_at_max_over_byte_budget() {
        // CLASS A3 (round p): out_hidden_size is a direct factor of merger_fc2
        // (`out_hidden_size * merge_in`); at its own per-field cap it overflows
        // MAX_VISION_TENSOR_BYTES even with everything else realistic. See
        // MAX_VISION_TENSOR_BYTES docs.
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": {MAX_VISION_OUT_HIDDEN_SIZE},
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "out_hidden_size == MAX_VISION_OUT_HIDDEN_SIZE must be rejected by the byte \
                 budget",
            )
            .to_string();
        assert!(
            err.contains("MAX_VISION_TENSOR_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_temporal_patch_size_over_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": {},
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#,
            MAX_VISION_TEMPORAL_PATCH_SIZE + 1
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("temporal_patch_size above MAX_VISION_TEMPORAL_PATCH_SIZE must be rejected")
            .to_string();
        assert!(
            err.contains("temporal_patch_size") && err.contains("MAX_VISION_TEMPORAL_PATCH_SIZE"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_temporal_patch_size_at_max_over_byte_budget() {
        // CLASS A3 (round p): temporal_patch_size is a direct factor of patch_embed_weight;
        // at its own per-field cap it overflows MAX_VISION_TENSOR_BYTES even with everything
        // else realistic. See MAX_VISION_TENSOR_BYTES docs.
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": {MAX_VISION_TEMPORAL_PATCH_SIZE},
                "num_position_embeddings": 2304,
                "in_channels": 3
            }}"#
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "temporal_patch_size == MAX_VISION_TEMPORAL_PATCH_SIZE must be rejected by the \
                 byte budget",
            )
            .to_string();
        assert!(
            err.contains("MAX_VISION_TENSOR_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_num_position_embeddings_over_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": {},
                "in_channels": 3
            }}"#,
            MAX_VISION_NUM_POSITION_EMBEDDINGS + 1
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "num_position_embeddings above MAX_VISION_NUM_POSITION_EMBEDDINGS must be \
                 rejected",
            )
            .to_string();
        assert!(
            err.contains("num_position_embeddings")
                && err.contains("MAX_VISION_NUM_POSITION_EMBEDDINGS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_num_position_embeddings_at_max_over_byte_budget() {
        // CLASS A3 (round p): num_position_embeddings is a direct factor of pos_embed; at its
        // own per-field cap it overflows MAX_VISION_TENSOR_BYTES even with everything else
        // realistic. See MAX_VISION_TENSOR_BYTES docs.
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": {MAX_VISION_NUM_POSITION_EMBEDDINGS},
                "in_channels": 3
            }}"#
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "num_position_embeddings == MAX_VISION_NUM_POSITION_EMBEDDINGS must be \
                 rejected by the byte budget",
            )
            .to_string();
        assert!(
            err.contains("MAX_VISION_TENSOR_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_non_square_num_position_embeddings() {
        let json = config_json_with_vision(
            r#"{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2305,
                "in_channels": 3
            }"#,
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("non-square num_position_embeddings must be rejected")
            .to_string();
        assert!(
            err.contains("num_position_embeddings") && err.contains("perfect square"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_present_vision_config_with_square_num_position_embeddings() {
        let json = config_json_with_vision(
            r#"{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3
            }"#,
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "square num_position_embeddings (48^2 = 2304) must be accepted"
        );
    }

    #[test]
    fn parser_rejects_zero_max_position_embeddings() {
        let json = r#"{"text_config": {"max_position_embeddings": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("max_position_embeddings: 0 must yield an InferenceError, not panic")
            .to_string();
        assert!(
            err.contains("max_position_embeddings"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_max_position_embeddings_at_one() {
        let json = r#"{"text_config": {"max_position_embeddings": 1}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "max_position_embeddings: 1 must be accepted (minimal nonzero RoPE table)"
        );
    }

    #[test]
    fn parser_rejects_zero_rope_theta() {
        let json = r#"{"text_config": {"rope_theta": 0.0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("rope_theta: 0.0 must yield an InferenceError, not NaN/inf propagation")
            .to_string();
        assert!(err.contains("rope_theta"), "wrong guard fired: {err}");
    }

    #[test]
    fn parser_rejects_negative_rope_theta() {
        let json = r#"{"text_config": {"rope_theta": -1.0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("negative rope_theta must yield an InferenceError")
            .to_string();
        assert!(err.contains("rope_theta"), "wrong guard fired: {err}");
    }

    // Note: a dedicated "non-finite rope_theta via config.json" test is not constructible --
    // JSON has no `NaN`/`Infinity` literals, and serde_json itself rejects an out-of-range
    // numeric literal (e.g. `1e400`) as a parse error before `from_config_json_str`'s
    // validation ever runs, so that vector is already fail-closed one layer up. The
    // `is_finite()` half of the `rope_theta` guard below is defense-in-depth for callers
    // that construct `Qwen35Config` directly (bypassing JSON parsing); the zero/negative
    // tests above cover the vectors actually reachable through `config.json`.

    #[test]
    fn parser_accepts_realistic_rope_theta() {
        let json = r#"{"text_config": {"rope_theta": 10000000.0}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "realistic rope_theta must be accepted"
        );
    }

    #[test]
    fn parser_rejects_linear_num_value_heads_over_max() {
        let value_heads = MAX_LINEAR_NUM_VALUE_HEADS + 1;
        let json = format!(
            r#"{{"text_config": {{
                "linear_num_key_heads": 1,
                "linear_num_value_heads": {value_heads},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1
            }}}}"#
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("linear_num_value_heads above MAX_LINEAR_NUM_VALUE_HEADS must be rejected")
            .to_string();
        assert!(
            err.contains("linear_num_value_heads") && err.contains("MAX_LINEAR_NUM_VALUE_HEADS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_linear_num_value_heads_at_max() {
        let json = format!(
            r#"{{"text_config": {{
                "linear_num_key_heads": 1,
                "linear_num_value_heads": {MAX_LINEAR_NUM_VALUE_HEADS},
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1
            }}}}"#
        );
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "linear_num_value_heads == MAX_LINEAR_NUM_VALUE_HEADS must be accepted"
        );
    }

    /// The specific attack this guard closes: `linear_value_head_dim = 1`
    /// lets `value_heads` alone satisfy `MAX_LINEAR_OUTPUT_DIM` and `MAX_GDN_STATE_SIZE`
    /// (both product budgets with `value_head_dim` as a factor) at values far above what
    /// `MAX_LINEAR_NUM_VALUE_HEADS` alone would allow through those two checks -- but a
    /// large, non-degenerate `linear_key_head_dim` still drives the GatedDeltaNet
    /// chunk-scratch buffers (`new_session_inner`, `forward/metal_qwen35.rs`) to multi-GiB
    /// scale. This must be caught by `MAX_GDN_CHUNK_SCRATCH_BYTES`, not by the two
    /// pre-existing product budgets (which this config satisfies trivially).
    #[test]
    fn parser_rejects_gdn_chunk_scratch_over_max_with_value_heads_and_state_size_in_budget() {
        let huge_key_dim = 600_000;
        let json = format!(
            r#"{{"text_config": {{
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": {huge_key_dim},
            "linear_value_head_dim": 1
        }}}}"#
        );
        // Sanity: this config passes both pre-existing product budgets trivially
        // (value_heads = 1, value_dim = 1, so both products collapse to `huge_key_dim` / 1).
        assert!(
            huge_key_dim <= MAX_GDN_STATE_SIZE,
            "test fixture assumption: gdn_state_size must be within MAX_GDN_STATE_SIZE"
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err(
                "a GDN chunk-scratch product over MAX_GDN_CHUNK_SCRATCH_BYTES must be rejected \
                 even though linear_num_value_heads, MAX_LINEAR_OUTPUT_DIM, and \
                 MAX_GDN_STATE_SIZE all pass",
            )
            .to_string();
        assert!(
            err.contains("chunk-scratch") && err.contains("MAX_GDN_CHUNK_SCRATCH_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_realistic_gdn_chunk_scratch_geometry() {
        let json = r#"{"text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128
        }}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "realistic GDN chunk-scratch geometry must be accepted"
        );
    }

    #[test]
    fn parser_rejects_present_vision_config_with_in_channels_over_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": {}
            }}"#,
            MAX_VISION_IN_CHANNELS + 1
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("in_channels above MAX_VISION_IN_CHANNELS must be rejected")
            .to_string();
        assert!(
            err.contains("in_channels") && err.contains("MAX_VISION_IN_CHANNELS"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_present_vision_config_with_in_channels_at_max() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": 768,
                "num_heads": 12,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 1024,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": {MAX_VISION_IN_CHANNELS}
            }}"#
        ));
        assert!(
            Qwen35Config::from_config_json_str(&json).is_ok(),
            "in_channels == MAX_VISION_IN_CHANNELS must be accepted"
        );
    }

    // ── CLASS A2: aggregate GDN session-buffer budget ───────────────────────────────

    #[test]
    fn parser_rejects_hostile_gdn_aggregate_value_heads_geometry() {
        // Hostile geometry: passes the per-buffer MAX_GDN_CHUNK_SCRATCH_BYTES
        // guard and every per-field/product guard individually, but the AGGREGATE across all
        // GDN session buffers reaches multi-GiB.
        let json = r#"{"text_config": {
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 4096,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 32
        }}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("hostile aggregate GDN geometry (value_heads=4096) must be rejected")
            .to_string();
        assert!(
            err.contains("MAX_GDN_SESSION_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_rejects_hostile_gdn_aggregate_key_head_dim_geometry() {
        // Second hostile geometry: value_heads=1 keeps the three-factor
        // MAX_GDN_STATE_SIZE product and the per-buffer chunk-scratch guard both passing at
        // their boundary, but the free `linear_key_head_dim` multiplier still blows up the
        // aggregate session-buffer sum.
        let json = r#"{"text_config": {
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 524288,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 16
        }}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("hostile aggregate GDN geometry (key_head_dim=524288) must be rejected")
            .to_string();
        assert!(
            err.contains("MAX_GDN_SESSION_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_realistic_gdn_aggregate_geometry() {
        // qwen36_27b's real worst-case GDN geometry (value_heads=48, key/value head dims
        // 128) must stay well inside MAX_GDN_SESSION_BYTES.
        let cfg = Qwen35Config::qwen36_27b();
        assert!(
            cfg.validate().is_ok(),
            "realistic qwen36_27b GDN aggregate geometry must be accepted"
        );
    }

    // ── CLASS A3: materialization-site byte budgets ─────────────────────────────────

    #[test]
    fn parser_rejects_hostile_embedding_product() {
        let json = format!(
            r#"{{"text_config": {{"vocab_size": 4000000, "hidden_size": {MAX_HIDDEN_SIZE}}}}}"#
        );
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("hostile embedding product must be rejected before materialization")
            .to_string();
        assert!(
            err.contains("MAX_EMBEDDING_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn parser_accepts_realistic_embedding_product() {
        let cfg = Qwen35Config::qwen36_35b_a3b();
        assert!(
            cfg.validate().is_ok(),
            "realistic embedding product must be accepted"
        );
    }

    #[test]
    fn parser_rejects_hostile_vision_derived_tensor_product() {
        let json = config_json_with_vision(&format!(
            r#"{{
                "depth": 4,
                "hidden_size": {MAX_VISION_HIDDEN_SIZE},
                "num_heads": 1,
                "patch_size": 14,
                "spatial_merge_size": 1,
                "out_hidden_size": {MAX_VISION_OUT_HIDDEN_SIZE},
                "temporal_patch_size": 2,
                "num_position_embeddings": 16777216,
                "in_channels": 3
            }}"#
        ));
        let err = Qwen35Config::from_config_json_str(&json)
            .expect_err("hostile vision derived-tensor product must be rejected")
            .to_string();
        assert!(
            err.contains("MAX_VISION_TENSOR_BYTES"),
            "wrong guard fired: {err}"
        );
    }

    // ── CLASS C: ValidatedQwen35Config ingress enumeration ──────────────────────────

    #[test]
    fn validated_config_rejects_hostile_directly_constructed_config() {
        // A raw Qwen35Config built directly (bypassing from_config_json_str entirely, e.g.
        // via serde deserializing just the inner struct) must still be rejected by the
        // checked TryFrom conversion -- device-free, no loader/allocation involved.
        let mut cfg = Qwen35Config::qwen35_2b();
        cfg.linear_num_key_heads = 16;
        cfg.linear_num_value_heads = Some(4096);
        cfg.linear_key_head_dim = 128;
        cfg.linear_value_head_dim = 32;
        let result: Result<ValidatedQwen35Config, InferenceError> = cfg.try_into();
        assert!(
            result.is_err(),
            "hostile directly-constructed config must fail ValidatedQwen35Config::try_from"
        );
    }

    #[test]
    fn validated_config_accepts_and_derefs_realistic_config() {
        let validated = Qwen35Config::qwen35_2b()
            .validate()
            .expect("realistic preset must validate");
        // Deref exposes read-only field access without unwrapping.
        assert_eq!(validated.hidden_size, 2048);
        let raw = validated.into_inner();
        assert_eq!(raw.hidden_size, 2048);
    }

    #[test]
    fn from_config_json_str_validated_matches_from_config_json_str() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen36_config.json"
        ));
        let raw = Qwen35Config::from_config_json_str(json).expect("raw parse succeeds");
        let validated =
            Qwen35Config::from_config_json_str_validated(json).expect("validated parse succeeds");
        assert_eq!(raw.hidden_size, validated.hidden_size);
        assert_eq!(raw.num_hidden_layers, validated.num_hidden_layers);
    }
}
