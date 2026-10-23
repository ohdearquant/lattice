//! Configuration for Qwen hybrid attention models (3.5-2B and 3.6-35B-A3B).
//!
//! Qwen3.5-2B uses a hybrid architecture with two attention mechanisms:
//! - **GatedDeltaNet** (linear attention): 18 layers with recurrent state
//! - **Full GQA** (standard attention): 6 layers with KV cache
//!
//! Qwen3.6-35B-A3B extends this with a Mixture-of-Experts (MoE) FFN:
//! 256 routed experts, top-8 selection, 1 shared expert, separate lm_head.

use crate::error::InferenceError;
use std::path::Path;

use crate::model::config_file::read_config_json_bounded;

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
/// the bounded `layer_types`/`layer_mask` sequence deserializers, and the loader's
/// `Vec::with_capacity` all allocate proportionally to it — an extreme value (e.g.
/// `usize::MAX`) reaches those allocations before any structural validation, causing
/// unbounded allocation / OOM / abort at model load. Real Qwen3.5/3.6 checkpoints are
/// well under 100 layers; 4096 — this crate's existing ceiling for other untrusted
/// count fields, e.g. `grammar::json_schema::MAX_ARRAY_CARDINALITY` — rejects a
/// pathological value like `2^63` while leaving generous headroom for future
/// architectures. The bound's purpose is to make the allocation finite and small
/// relative to available memory, which 4096 layers satisfies with room to spare.
pub const MAX_HIDDEN_LAYERS: usize = 4096;

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

/// Upper bound, in bytes, on a single dense-FFN gate/up/down projection tensor
/// (`hidden_size * intermediate_size * 4`) accepted from `config.json` (FIX 8).
///
/// [`MAX_HIDDEN_SIZE`] and [`MAX_INTERMEDIATE_SIZE`] bound each factor independently at
/// 1,048,576; `load_dense_ffn_weights` (`model/qwen35/loading.rs`) materializes
/// `gate_proj`/`up_proj`/`down_proj` as owned `Vec<f32>` of exactly `hidden_size *
/// intermediate_size` elements each -- a config with both factors near their individual
/// caps (and `vocab_size = 1`, satisfying [`MAX_EMBEDDING_BYTES`] trivially) passes both
/// scalar checks while requiring ~4 TiB per tensor. Real Qwen3.5/3.6 dense presets top out
/// around `hidden_size = 8192` * `intermediate_size = 17408` (~570 MiB); 4 GiB
/// (4_294_967_296) leaves roughly 7x headroom above that while rejecting the
/// terabyte-scale hostile case by several orders of magnitude.
pub(crate) const MAX_DENSE_FFN_TENSOR_BYTES: u128 = 4_294_967_296;

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
/// engine's chunk size ever changes, this mirror must change with it. No automated
/// drift-guard test asserts this equality: `metal_qwen35.rs`'s `GDN_CHUNK_SIZE` is a
/// private `const` inside a module gated by the `metal-gpu` feature, and exposing it to a
/// cross-module equality test would require changing that file's visibility -- out of
/// scope for a round that must leave `forward/metal_qwen35.rs` untouched (see this PR's
/// scoping note). Change both constants together by hand if the engine's chunk size ever
/// changes; a `metal-gpu`-gated drift test is tracked as follow-up work.
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

/// Upper bound, in bytes, on the AGGREGATE of every linear-attention layer's persistent
/// GatedDeltaNet recurrent state (`s_matrices` + `conv_buffer`, `GatedDeltaNetState::new`,
/// `attention/gdn.rs`) across the whole model, accepted from `config.json`.
///
/// [`MAX_GDN_STATE_SIZE`] and [`MAX_GDN_CONV_BUFFER_SIZE`] bound one layer's state; neither
/// multiplies by [`Qwen35Config::num_linear_attention_layers`]. `generation.rs` (both
/// `generate` and `generate_streaming`) allocates one `GatedDeltaNetState` per linear layer
/// up front, before the first token: `(0..num_linear).map(|_| GatedDeltaNetState::new(cfg))`.
/// A config with `num_hidden_layers` near [`MAX_HIDDEN_LAYERS`], every layer typed
/// linear-attention, and per-layer state near the individual caps passes both of those
/// guards while the aggregate still reaches tens of gigabytes before any token is generated.
/// Real Qwen3.5/3.6 GDN configs (worst preset: `qwen36_27b`, ~40 linear layers, per-layer
/// state+conv ~3.2 MiB) sum to roughly 128 MiB; 4 GiB (4_294_967_296) leaves roughly 32x
/// headroom above that while rejecting the multi-hundred-gigabyte hostile case (512 layers *
/// per-layer caps) by more than an order of magnitude.
pub(crate) const MAX_GDN_CROSS_LAYER_STATE_BYTES: u128 = 4_294_967_296;

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
    /// ViT MLP intermediate (hidden) dimension. FIX 16: previously left unparsed (ADR-069
    /// S1) and the loader hardcoded `4 * hidden_size` instead -- official Qwen3.5-VL vision
    /// towers use `hidden_size = 1152`, `intermediate_size = 4304` (not `4 * 1152 =
    /// 4608`), so the hardcoded assumption rejects real checkpoints. `None` (field absent
    /// from `config.json`) falls back to `4 * hidden_size`, matching HF default semantics
    /// for architectures that omit an explicit vision MLP width.
    #[serde(default)]
    pub intermediate_size: Option<usize>,
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
        // The preprocessor (`vision::qwen35_vit::preprocess_qwen35_image`) is RGB-only by
        // construction: it decodes every image to a fixed 3-channel `RgbImage` and then
        // indexes `pixel[c]` for `c in 0..in_channels`. A shape-consistent checkpoint
        // declaring `in_channels >= 4` otherwise passes this validation and panics the sole
        // Metal worker on the first ordinary image request, because `pixel[3]` is out of
        // bounds for a 3-element `Rgb<u8>`. Fail closed here instead of silently truncating
        // or padding channels.
        //
        // This is a dimension (B) semantic-validity bound and it subsumes both the
        // `in_channels == 0` and the `in_channels > MAX_VISION_IN_CHANNELS` dimension (A)
        // allocation bounds on the parse path. The (A) bound below is kept deliberately: it
        // states the allocation limit independently, so relaxing this check for a future
        // non-RGB preprocessor cannot silently uncap the allocation.
        if self.in_channels != 3 {
            return Err(InferenceError::Inference(format!(
                "invalid vision_config: in_channels must be 3 (RGB); got {} -- the image \
                 preprocessor is RGB-only and cannot serve any other channel count",
                self.in_channels
            )));
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
        // FIX 16: `intermediate_size`, when present, replaces the `4 * hidden_size`
        // fallback as the vision MLP width -- bound it the same way as the text decoder's
        // `intermediate_size` (`MAX_INTERMEDIATE_SIZE`), since it drives the same class of
        // MLP-tensor allocation in `checked_derived_sizes` below.
        if let Some(vision_intermediate_size) = self.intermediate_size {
            if vision_intermediate_size == 0 {
                return Err(InferenceError::Inference(
                    "invalid vision_config: intermediate_size must be > 0 when present".to_string(),
                ));
            }
            if vision_intermediate_size > MAX_INTERMEDIATE_SIZE {
                return Err(InferenceError::Inference(format!(
                    "invalid vision_config: intermediate_size ({vision_intermediate_size}) \
                     exceeds MAX_INTERMEDIATE_SIZE ({MAX_INTERMEDIATE_SIZE})"
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
        // FIX 16: use the parsed `intermediate_size` when present rather than always
        // assuming `4 * hidden_size` -- see the field's docs.
        let mlp_intermediate = match self.intermediate_size {
            Some(v) => v,
            None => self.hidden_size.checked_mul(4).ok_or_else(overflow)?,
        };
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

// Whether text_config spells the flat RoPE fields at all. Qwen35Config is #[serde(default)], so
// its parsed values cannot tell an absent field from the preset default.
#[derive(Debug, Default, serde::Deserialize)]
struct HfFlatRopePresence {
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
}

#[derive(Debug, serde::Deserialize)]
struct HfFlatRopePresenceFile {
    #[serde(default)]
    text_config: Option<HfFlatRopePresence>,
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
        let json = read_config_json_bounded(path, "config.json")?;
        Self::from_config_json_str(&json)
    }

    /// Parse a HF config.json into a validated [`ValidatedQwen35Config`]. See
    /// [`Self::from_config_json`] for the raw-`Qwen35Config`-returning sibling.
    pub fn from_config_json_validated(
        path: &Path,
    ) -> Result<ValidatedQwen35Config, InferenceError> {
        let json = read_config_json_bounded(path, "config.json")?;
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

        // A text-only checkpoint saved as `Qwen3_5ForCausalLM` writes its text
        // configuration at the TOP LEVEL of config.json (`model_type: qwen3_5_text`), with
        // no `text_config` object. Read the geometry from there instead of substituting an
        // unrelated preset; a file carrying neither form is refused. The top-level object
        // deserializes into `Qwen35Config` exactly as `text_config` does -- the struct is
        // `#[serde(default)]` and does not deny unknown fields -- so unrelated keys such as
        // `architectures` and `transformers_version` are skipped. The vision-language fields
        // are NOT skipped: `Qwen35Config` declares them at its own top level, so this parse
        // reads them directly and the `parsed.*` copies below overwrite each with the
        // identical value read from the same key.
        let (mut cfg, flat_top_level) = match parsed.text_config {
            Some(cfg) => (cfg, false),
            None => {
                let root: serde_json::Value = serde_json::from_str(json).map_err(|e| {
                    InferenceError::Inference(format!("invalid Qwen config.json: {e}"))
                })?;
                if !root.as_object().is_some_and(|obj| {
                    obj.contains_key("num_hidden_layers") && obj.contains_key("hidden_size")
                }) {
                    return Err(InferenceError::Inference(
                        "invalid Qwen config.json: no text_config object and no top-level \
                         num_hidden_layers/hidden_size"
                            .to_string(),
                    ));
                }
                let cfg: Qwen35Config = serde_json::from_value(root).map_err(|e| {
                    InferenceError::Inference(format!("invalid Qwen config.json: {e}"))
                })?;
                (cfg, true)
            }
        };

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
        // instead of at the text_config level — use the nested value when the flat field is
        // absent from the JSON (or, for rope_theta, 0.0), and refuse when both disagree.
        // A wrapped checkpoint spells the flat fields inside `text_config`; a flat file
        // spells them at the top level. `field_prefix` keeps each conflict message naming
        // the field as it is written in the file.
        if let Some(rp) = &cfg.rope_parameters {
            let (flat, field_prefix) = if flat_top_level {
                let flat = serde_json::from_str::<HfFlatRopePresence>(json).map_err(|e| {
                    InferenceError::Inference(format!("invalid Qwen config.json: {e}"))
                })?;
                (flat, "")
            } else {
                let flat = serde_json::from_str::<HfFlatRopePresenceFile>(json)
                    .map_err(|e| {
                        InferenceError::Inference(format!("invalid Qwen config.json: {e}"))
                    })?
                    .text_config
                    .unwrap_or_default();
                (flat, "text_config.")
            };
            if rp.rope_theta != 0.0 {
                match flat.rope_theta.filter(|&t| t != 0.0) {
                    Some(t) if t != rp.rope_theta => {
                        return Err(InferenceError::Inference(format!(
                            "invalid Qwen config.json: {field_prefix}rope_theta ({t}) conflicts \
                             with {field_prefix}rope_parameters.rope_theta ({})",
                            rp.rope_theta
                        )));
                    }
                    _ => cfg.rope_theta = rp.rope_theta,
                }
            }
            if let Some(prf) = rp.partial_rotary_factor {
                match flat.partial_rotary_factor {
                    Some(f) if f != prf => {
                        return Err(InferenceError::Inference(format!(
                            "invalid Qwen config.json: {field_prefix}partial_rotary_factor ({f}) \
                             conflicts with {field_prefix}rope_parameters.partial_rotary_factor \
                             ({prf})"
                        )));
                    }
                    _ => cfg.partial_rotary_factor = prf,
                }
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
        // FIX 8: `MAX_HIDDEN_SIZE` and `MAX_INTERMEDIATE_SIZE` above bound each factor
        // independently; `load_dense_ffn_weights` (`model/qwen35/loading.rs`) materializes
        // gate/up/down projection tensors sized by their product, which neither per-factor
        // cap bounds. Gated on `!cfg.is_moe()` -- the loader only ever calls
        // `load_dense_ffn_weights` for a non-MoE config (`loading.rs`'s `is_moe()` branch),
        // so a MoE config's `intermediate_size` (which still exists on the struct but
        // drives a different, already-bounded, `MAX_INTERMEDIATE_SIZE`-capped expert path)
        // must not be double-bounded by a materialization site it never reaches. See
        // `MAX_DENSE_FFN_TENSOR_BYTES` docs.
        if !cfg.is_moe() {
            let dense_ffn_tensor_bytes =
                cfg.hidden_size as u128 * cfg.intermediate_size as u128 * 4;
            if dense_ffn_tensor_bytes > MAX_DENSE_FFN_TENSOR_BYTES {
                return Err(InferenceError::Inference(format!(
                    "invalid Qwen config.json: dense FFN tensor size \
                     ({dense_ffn_tensor_bytes} bytes) exceeds MAX_DENSE_FFN_TENSOR_BYTES \
                     ({MAX_DENSE_FFN_TENSOR_BYTES}): hidden_size ({}) * intermediate_size ({})",
                    cfg.hidden_size, cfg.intermediate_size
                )));
            }
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
        // FIX 15: RMSNorm computes `sqrt(mean(x^2) + eps)` and divides by it; a non-positive
        // (including `-0.0`) or non-finite `rms_norm_eps` drives that sqrt/division to
        // `NaN`, silently corrupting every layer's output with no error signal, since
        // neither is a panic -- the same correctness-not-allocation dimension (B) as
        // `rope_theta` below. Real Qwen3.5/3.6 presets use `1e-6`.
        if !(cfg.rms_norm_eps.is_finite() && cfg.rms_norm_eps > 0.0) {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: rms_norm_eps ({}) must be finite and > 0.0",
                cfg.rms_norm_eps
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
        // FIX 9+14: `MAX_GDN_STATE_SIZE` / `MAX_GDN_CONV_BUFFER_SIZE` above bound one
        // layer's persistent GatedDeltaNet state; neither multiplies by
        // `num_linear_attention_layers()`. `generation.rs` allocates one
        // `GatedDeltaNetState` (s_matrices + conv_buffer) per linear-attention layer
        // before the first token -- bound the aggregate across all layers. See
        // `MAX_GDN_CROSS_LAYER_STATE_BYTES` docs.
        let num_linear_layers = cfg.num_linear_attention_layers() as u128;
        let per_layer_state_bytes = (gdn_state_size as u128 + gdn_conv_buffer_size as u128) * 4;
        let cross_layer_state_bytes = per_layer_state_bytes * num_linear_layers;
        if cross_layer_state_bytes > MAX_GDN_CROSS_LAYER_STATE_BYTES {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: aggregate GatedDeltaNet cross-layer state size \
                 ({cross_layer_state_bytes} bytes) exceeds MAX_GDN_CROSS_LAYER_STATE_BYTES \
                 ({MAX_GDN_CROSS_LAYER_STATE_BYTES}): per-layer state ({per_layer_state_bytes} \
                 bytes) * num_linear_attention_layers ({num_linear_layers})"
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

    /// Checked variant of [`full_q_dim`](Self::full_q_dim). Config dimensions are parsed
    /// from an untrusted checkpoint's `config.json`; `num_attention_heads * head_dim` must
    /// reject overflow with a typed error instead of wrapping before Q8 geometry checks
    /// consume the result.
    pub fn checked_full_q_dim(&self) -> Result<usize, InferenceError> {
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "invalid Qwen config: num_attention_heads ({}) * head_dim ({}) overflows usize",
                    self.num_attention_heads, self.head_dim
                ))
            })
    }

    /// Checked variant of [`full_kv_dim`](Self::full_kv_dim). See [`checked_full_q_dim`](Self::checked_full_q_dim).
    pub fn checked_full_kv_dim(&self) -> Result<usize, InferenceError> {
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "invalid Qwen config: num_key_value_heads ({}) * head_dim ({}) overflows usize",
                    self.num_key_value_heads, self.head_dim
                ))
            })
    }

    /// Checked variant of [`linear_qkv_dim`](Self::linear_qkv_dim). See
    /// [`checked_full_q_dim`](Self::checked_full_q_dim).
    pub fn checked_linear_qkv_dim(&self) -> Result<usize, InferenceError> {
        let overflow = || {
            InferenceError::InvalidInput(format!(
                "invalid Qwen config: linear_num_key_heads ({}) * linear_key_head_dim ({}) + \
                 linear_num_value_heads ({}) * linear_value_head_dim ({}) overflows usize",
                self.linear_num_key_heads,
                self.linear_key_head_dim,
                self.linear_num_value_heads(),
                self.linear_value_head_dim,
            ))
        };
        let q = self
            .linear_num_key_heads
            .checked_mul(self.linear_key_head_dim)
            .ok_or_else(overflow)?;
        let k = self
            .linear_num_key_heads
            .checked_mul(self.linear_key_head_dim)
            .ok_or_else(overflow)?;
        let v = self
            .linear_num_value_heads()
            .checked_mul(self.linear_value_head_dim)
            .ok_or_else(overflow)?;
        q.checked_add(k)
            .and_then(|qk| qk.checked_add(v))
            .ok_or_else(overflow)
    }

    /// Checked variant of [`linear_output_dim`](Self::linear_output_dim). See
    /// [`checked_full_q_dim`](Self::checked_full_q_dim).
    pub fn checked_linear_output_dim(&self) -> Result<usize, InferenceError> {
        self.linear_num_value_heads()
            .checked_mul(self.linear_value_head_dim)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "invalid Qwen config: linear_num_value_heads ({}) * linear_value_head_dim ({}) \
                     overflows usize",
                    self.linear_num_value_heads(),
                    self.linear_value_head_dim
                ))
            })
    }

    /// Checked GatedDeltaNet conv1d weight element count: `linear_qkv_dim() * linear_conv_kernel_dim`.
    /// GDN-specific; net-new (no upstream free-function counterpart). See
    /// [`checked_full_q_dim`](Self::checked_full_q_dim).
    pub fn checked_linear_conv_len(&self) -> Result<usize, InferenceError> {
        let qkv_dim = self.checked_linear_qkv_dim()?;
        qkv_dim
            .checked_mul(self.linear_conv_kernel_dim)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "invalid Qwen config: linear_qkv_dim ({qkv_dim}) * linear_conv_kernel_dim ({}) \
                 overflows usize",
                    self.linear_conv_kernel_dim
                ))
            })
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

/// Checked doubling of a config-derived dimension (e.g. full-attention `q_rows = 2 * q_dim`).
/// `what` names the quantity in the error message on overflow.
pub(crate) fn checked_double(value: usize, what: &str) -> Result<usize, InferenceError> {
    value.checked_mul(2).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "invalid Qwen config: 2 * {what} ({value}) overflows usize"
        ))
    })
}

/// Neutral generation types, moved to [`crate::generation`] under ADR-092.
///
/// These aliases keep every `model::qwen35_config::` path resolving. They are deprecated
/// aliases rather than deprecated re-exports on purpose: `#[deprecated]` on a `pub use` is
/// accepted by the compiler and emits nothing, so it would read as a live deprecation and
/// warn no one. A deprecated type alias does fire.
#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::GenerateConfig`"
)]
pub type GenerateConfig = crate::generation::GenerateConfig;

#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::GenerateOutput`"
)]
pub type GenerateOutput = crate::generation::GenerateOutput;

#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::TokenLogprob`"
)]
pub type TokenLogprob = crate::generation::TokenLogprob;

#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::TopLogprob`"
)]
pub type TopLogprob = crate::generation::TopLogprob;

/// Number of additional tokens a forced reasoning close may wait for a line boundary.
pub(crate) const REASONING_CLOSE_GRACE_TOKENS: usize = 32;

/// Decide whether to force-close the thinking block this step (s1 budget forcing).
///
/// Returns `Some(close_id)` to override the sampled token with `</think>`, else `None`.
/// Once the budget is reached, the close waits for the previous emitted token to
/// contain a newline, and fires without one after
/// [`REASONING_CLOSE_GRACE_TOKENS`] more tokens. All of thinking enabled, a
/// non-zero budget, an open block, and the budget being reached must also hold.
/// Returns `None` immediately if any guard fails so the common disabled path
/// costs a single `Option::None` check per step.
///
/// `enable_thinking: false` makes `reasoning_budget` inert here regardless of its value --
/// no reasoning block is produced in that state (the caller is contracted to have primed
/// the prompt accordingly, see [`crate::generation::GenerateConfig::enable_thinking`]), so there is nothing to
/// force-close. `resolve_reasoning_close_token` (`qwen35/generation.rs`), which resolves
/// `close_id` for this function's `close_id` parameter, applies the identical
/// short-circuit before validating the tokenizer, so the two functions agree about what the
/// combination means instead of one rejecting it and the other silently no-opping.
#[inline]
pub(crate) fn force_close_think(
    reasoning_budget: Option<usize>,
    enable_thinking: bool,
    thinking_closed: bool,
    generated_so_far: usize,
    close_id: Option<u32>,
    previous_token_is_boundary: bool,
) -> Option<u32> {
    let budget = reasoning_budget?;
    let close = close_id?;
    if !enable_thinking || thinking_closed || budget == 0 || generated_so_far < budget {
        return None;
    }
    let grace_end = budget.saturating_add(REASONING_CLOSE_GRACE_TOKENS);
    (generated_so_far >= grace_end || previous_token_is_boundary).then_some(close)
}

/// Decode-loop iteration cap.
///
/// When a reasoning budget is active, reasoning tokens get their OWN budget (`rb`) ON TOP
/// of the answer budget (`max_new_tokens`), plus **1** for the forced `</think>` delimiter
/// itself: worst case `rb + max_new_tokens + 1` total tokens. The +1 is necessary because
/// the forced `</think>` is an extra token that is not part of either the reasoning content
/// or the answer — omitting it leaves the answer one token short (off-by-one).
///
/// The close may wait up to [`REASONING_CLOSE_GRACE_TOKENS`] past `rb` for a line
/// boundary. Those grace tokens are part of the same cap and come out of the answer
/// budget; they never extend it.
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
mod tests;

#[cfg(test)]
#[path = "qwen35_config_rope_tests.rs"]
mod nested_theta_tests;
