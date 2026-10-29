//! Metal GPU forward pass for Qwen3.5-2B hybrid attention model.
//!
//! Hybrid architecture: 18 GatedDeltaNet (linear) + 6 GQA (full) layers.
//! Pattern: [linear, linear, linear, full] x 6 = 24 layers.
//!
//! Design decisions:
//! - **All weights in unified memory** (StorageModeShared): zero-copy CPU/GPU access.
//! - **GPU dispatches for projections and FFN**: the big matmuls that dominate compute.
//! - **CPU-side GDN recurrence**: the S matrix update is sequential per-token and only
//!   ~17% of total compute. Runs on CPU reading from shared Metal buffers.
//! - **GPU dispatches for full attention**: Q/K/V projections, QK-norm, partial RoPE,
//!   fused attention, output gating, and O projection all on GPU.
//! - **Single command buffer per layer**: encode all dispatches, commit, wait.
//! - **Qwen3.5 RMS norm**: uses (1 + gamma) shifted normalization.
//! - **Interleaved partial RoPE**: only first rope_dim dimensions rotated.
//!
//! ## Half-precision weight storage (f16)
//!
//! Large projection weight matrices are stored as IEEE-754 half-precision (`f16`/`half`)
//! in Metal buffers to halve memory bandwidth. Activations, accumulators, norms, and
//! outputs remain `f32`. The MSL kernel `matmul_bt_half` loads weight tiles as `half`
//! and widens to `float` before accumulation, producing identical tiling and dispatch
//! geometry as the `f32` path.
//!
//! Buffers stored as f16: all projection weights (QKV, Z, B, A, out, gate, up, down,
//! q/k/v/o_proj). `embed_tokens` is f16 for the safetensors upload; a Q4 checkpoint maps
//! its payload and decodes rows on demand. Buffers kept as f32: norm weights, bias/log
//! scalars, conv1d weights (small, read by CPU), RoPE tables, activation scratch buffers.

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
mod mtp_weights;

#[cfg(any(test, feature = "metal-gpu"))]
mod q4_embed_lookup;

// -----------------------------------------------------------------------------
// MTP / self-spec route-around predicates (PR #657) --
// pure, `Device`-free, compiled unconditionally in test builds (same pattern
// as the `resolve_mtp_*` functions above: `mod inner` re-imports these via
// `use super::{...}` so the two real dispatch sites in `generate()` stay
// single-source-of-truth, but the boolean logic itself is unit-testable
// without a GPU or a Metal MTP/self-spec fixture).
// -----------------------------------------------------------------------------

/// Whether `generate()` should take the MTP greedy-draft branch instead of
/// plain per-token decode. Mirrors the `use_mtp` computation inline in
/// `MetalQwen35State::generate` exactly -- in particular
/// `gen_cfg.stop_strings.is_empty()` is load-bearing: MTP's draft/verify loop
/// has no string-stop awareness, so any non-empty `stop_strings` must route
/// around it to the plain loop (which does check stops after every token).
///
/// `gen_cfg.reasoning_budget.is_none()` is the same kind of load-bearing
/// route-around (ADR-080 C3, #783): MTP's draft/verify loop has no
/// budget-forcing awareness (no `decode_cap` / `force_close_think` wiring),
/// so any set `reasoning_budget` must route around it to the plain loop,
/// which either applies budget forcing or rejects the config via
/// `check_reasoning_budget_not_set`.
///
/// `gen_cfg.repetition_penalty == 1.0` is likewise load-bearing (PR #787):
/// both the MTP draft (`mtp_forward_one`) and verify
/// steps select tokens via raw `argmax_f32_first_wins` over logits, never
/// through `sample_token`, so a non-identity penalty configured by the caller
/// would be silently ignored -- a set penalty can change which token is
/// greedy-argmax-optimal, so this is an output-correctness issue, not only a
/// performance one. Any non-1.0 penalty must route around MTP to the plain
/// loop, which samples via `sample_token` and applies it.
///
/// `gen_cfg.logprobs.is_none()` is likewise load-bearing (PR #787):
/// `generate_greedy_mtp` unconditionally returns
/// `token_logprobs: vec![]`, so a set `logprobs` request must route around
/// MTP to the plain loop, which either captures logprobs or rejects the
/// config via `check_logprobs_not_set`.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn mtp_route_active(
    mtp_present: bool,
    mtp_enabled: bool,
    gen_cfg: &crate::generation::GenerateConfig,
    use_compact: bool,
) -> bool {
    mtp_present
        && mtp_enabled
        && gen_cfg.top_k <= 1
        && gen_cfg.temperature <= 0.0
        && !use_compact
        && gen_cfg.grammar.is_none()
        && gen_cfg.stop_strings.is_empty()
        && gen_cfg.reasoning_budget.is_none()
        && gen_cfg.repetition_penalty == 1.0
        && gen_cfg.logprobs.is_none()
}

/// Whether `generate()` should take the GDN-first self-speculative greedy
/// branch. Mirrors the `use_self_spec` computation inline in
/// `MetalQwen35State::generate` exactly -- same `stop_strings.is_empty()` /
/// `reasoning_budget.is_none()` / `repetition_penalty == 1.0` route-around
/// rationale as [`mtp_route_active`] (PR #787):
/// self-spec's verify step also selects via raw argmax, never `sample_token`.
/// `gen_cfg.logprobs.is_none()` is the same route-around
/// as [`mtp_route_active`]: `generate_greedy_self_spec` also unconditionally
/// returns `token_logprobs: vec![]`.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn self_spec_route_active(
    gdn_checkpoints_present: bool,
    self_spec_env_set: bool,
    gen_cfg: &crate::generation::GenerateConfig,
    use_compact: bool,
    num_active_linear_attention_layers: usize,
) -> bool {
    gdn_checkpoints_present
        && self_spec_env_set
        && gen_cfg.top_k <= 1
        && gen_cfg.temperature <= 0.0
        && !use_compact
        && gen_cfg.grammar.is_none()
        && gen_cfg.stop_strings.is_empty()
        && gen_cfg.reasoning_budget.is_none()
        && gen_cfg.repetition_penalty == 1.0
        && gen_cfg.logprobs.is_none()
        && num_active_linear_attention_layers > 0
}

/// Whether the MTP verify phase should use the batch-GEMM verifier instead of
/// the per-token verifier. The batch-GEMM verifier is opt-in through
/// `LATTICE_MTP_BATCH` (`batch_switch_on`), but its dispatch issues base
/// projections only and never applies a loaded LoRA adapter, while the
/// per-token verifier steps through the layer encoders that do. With an
/// adapter loaded (`adapter_loaded`) the switch is therefore ignored, so the
/// verified tokens and the state the verifier writes stay adapter-consistent
/// with plain decode.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn use_batch_gemm_verifier(batch_switch_on: bool, adapter_loaded: bool) -> bool {
    batch_switch_on && !adapter_loaded
}

#[cfg(test)]
mod route_predicate_tests;

// -----------------------------------------------------------------------------
// Chat Completion API -- CPU-available (#668)
// -----------------------------------------------------------------------------
//
// `ChatRole`, `ChatMessage`, and `format_chat_template` used to live inside
// `mod inner`, gated on `metal-gpu`, so a CPU-only build had no shared
// ChatML renderer to call and `lattice.rs`'s CPU serve path carried its own
// hand-rolled `render_prompt` instead (#661 follow-up). They are pure and
// `Device`-free -- same reasoning as `mtp_tensor_path` above -- so they move
// to module scope, unconditionally compiled, and `mod inner` re-imports them
// via `use super::{...}`.

/// **Unstable**: chat conversation role; variants may extend for tool/function roles.
///
/// Role in a chat conversation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

/// **Unstable**: single chat message; fields may expand with tool call support.
///
/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Inline image carried by a normalized chat content part.
    pub image: Option<ChatImage>,
}

/// **Unstable**: one inline image attached to a chat message.
#[derive(Debug, Clone)]
pub struct ChatImage {
    /// Base64-decoded PNG or JPEG file bytes.
    pub bytes: Vec<u8>,
    /// UTF-8 byte offset into [`ChatMessage::content`] where the image part
    /// appeared.
    pub text_offset: usize,
}

impl ChatMessage {
    /// **Unstable**: construct a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
            image: None,
        }
    }
    /// **Unstable**: construct a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            image: None,
        }
    }
    /// **Unstable**: construct an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            image: None,
        }
    }

    /// **Unstable**: construct a user message with one inline image.
    pub fn user_with_image(
        content: impl Into<String>,
        image_bytes: Vec<u8>,
        text_offset: usize,
    ) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            image: Some(ChatImage {
                bytes: image_bytes,
                text_offset,
            }),
        }
    }
}

/// **Unstable**: format messages into Qwen3.5 chat template; template format may change.
///
/// Format messages into Qwen3.5 chat template.
/// Template: <|im_start|>{role}\n{content}<|im_end|>\n
/// Final assistant turn left open for generation.
///
/// This formatter emits text turns only. Inline image payloads require the
/// multimodal vision generation path; the public text-chat generation entry
/// points reject such messages before calling this formatter.
pub fn format_chat_template(messages: &[ChatMessage]) -> String {
    format_chat_template_parts(
        messages
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str())),
    )
}

pub(crate) fn format_chat_template_parts<'a>(
    messages: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> String {
    let mut prompt = String::new();
    for (role, content) in messages {
        push_chat_turn_open(&mut prompt, role);
        prompt.push_str(content);
        push_chat_turn_close(&mut prompt);
    }
    push_chat_generation_open(&mut prompt);
    prompt
}

pub(crate) fn push_chat_turn_open(prompt: &mut String, role: &str) {
    prompt.push_str("<|im_start|>");
    prompt.push_str(role);
    prompt.push('\n');
}

pub(crate) fn push_chat_turn_close(prompt: &mut String) {
    prompt.push_str("<|im_end|>\n");
}

/// Open the trailing assistant turn left unclosed for generation — shared by
/// the canonical text-only renderer and the vision prompt builder so both
/// paths emit the identical generation-turn opener.
pub(crate) fn push_chat_generation_open(prompt: &mut String) {
    push_chat_turn_open(prompt, ChatRole::Assistant.as_str());
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod inner {
    mod buffers;
    mod diagnostics;
    mod dispatch;
    mod embedding_table;
    mod engine;
    mod gdn_state;
    mod generation;
    mod layers;
    mod lora;
    mod mtp;
    mod pipelines;
    mod prefill;
    mod prefill_chunk;
    mod prefix_cache;
    mod prefix_generation;
    mod sampling;
    mod spec_types;
    mod speculative;
    mod step;
    mod verify;
    mod weights;

    use super::GdnStateTrafficScope;
    use super::mtp_missing_weights_warning;
    use super::{MtpRoundOutcome, mtp_greedy_round, use_batch_gemm_verifier};
    use super::{mtp_route_active, multimodal_generate_preflight, self_spec_route_active};
    #[cfg(feature = "bench-internals")]
    use buffers::read_buffer_offset;
    use buffers::{
        GemmParams, GpuCandidate, MetalKvCache, MetalQwen35Activations, Q3WeightBuf,
        Q4_0_BLOCK_SIZE, Q4WeightBuf, Q8_0_BLOCK_SIZE, QK4_0, QK8_0, div_ceil, make_buffer,
        make_buffer_f16, make_buffer_q4_0, make_buffer_q8_0, make_zero_buffer,
        make_zero_byte_buffer, mmap_q4_weight, quantize_row_q8_0, read_buffer, write_buffer,
    };
    #[cfg(test)]
    use buffers::{convert_f16_row, f16_to_f32, f32_to_f16, mmap_q3_weight, read_buffer_f16};
    #[cfg(test)]
    use diagnostics::{
        FORCED_MOE_EXPERTS_FOR_TEST, MOE_PREFETCH_ORDERING_GATE_FOR_TEST,
        MOE_PREFETCH_PARALLEL_FOR_TEST, MOE_PREFETCH_STEP2_DONE_TX_FOR_TEST,
        MOE_ROUTER_RUNS_FOR_TEST, set_forced_moe_experts_for_test,
        set_moe_prefetch_ordering_gate_for_test, set_moe_prefetch_parallel_for_test,
        set_moe_prefetch_step2_done_tx_for_test, take_moe_router_runs_for_test,
    };
    pub use diagnostics::{
        HiddenReadbackPathProofSnapshot, LayerImportanceScore, LayerPruningPlan,
        LogitReadbackPathProofSnapshot, MoeRoutingTraceRecord, PathProofCounters,
        PathProofSnapshot, arm_moe_routing_trace, dump_moe_routing_trace_jsonl,
        take_moe_routing_trace,
    };
    use diagnostics::{
        MOE_DEVICE_ROUTE_PENDING, MOE_ROUTING_TRACE, StepProfile, publish_moe_device_routes,
    };
    use gdn_state::{MetalGdnState, MetalGdnStatePrecision, active_gdn_layer_indices};
    // Chat message types + the shared ChatML renderer (#668) live at the file
    // top level (module-scope, above `mod inner`) so CPU-only builds can
    // render through the same `format_chat_template` this Metal path uses,
    // same reasoning as the MTP resolution helpers below.
    use super::{ChatMessage, MtpHiddenTap, format_chat_template};
    // MTP Q4/F16 flavor + shape resolution (#630, #636) lives in a private
    // portable sibling module so it compiles and is unit-testable without the
    // `metal-gpu` feature at all.
    use super::mtp_weights::{
        MtpLoadErr, MtpTensorSource, resolve_mtp_norm, resolve_mtp_projection,
    };
    use super::q4_embed_lookup::Q4Embedding;
    #[cfg(feature = "gdn-state-counters")]
    use super::{GdnStateCopyKind, GdnStateTrafficCounters, GdnStateTrafficShape};
    use crate::attention::gdn::GatedDeltaNetState;
    use crate::attention::gdn_fused::GatedDeltaNetFusedScratch;
    use crate::generation::{GenerateConfig, GenerateOutput};
    use crate::model::qwen35::stop_strings::StopStringMatcher;
    use crate::model::qwen35::{
        AttentionWeights, GenerationEntryContract, GenerationPlan, GenerationPreparation,
        ModelWeights, REASONING_CLOSE_MARKER, prepare_generation,
    };
    use crate::stop_reason::StopReason;
    use crate::vision::multimodal::Qwen35VisionRequest;
    use embedding_table::EmbeddingTable;
    pub(crate) use engine::QuantFormat;
    pub use engine::{InferenceSession, MetalQwen35Engine};
    use engine::{build_rope_interleaved, derive_has_moe_layer};
    use lattice_arch::qwen35_config::Qwen35Config;
    use lattice_tokenizer::__internal::{self as tok, incremental_detokenizer_finish};
    use lattice_tokenizer::bpe::BpeTokenizer;
    use lattice_weights::q4_weights::quantize_row_q4_0;
    #[cfg(test)]
    use lora::MAX_BLEND_RANK_TOTAL;
    use lora::MetalLoraAdapter;
    pub use lora::{LoraLayerData, blend_lora_layer_data};
    use metal::*;
    use pipelines::build_pipelines;
    use prefix_cache::MetalCrossTurnPrefixCache;
    use prefix_cache::MetalGdnCheckpoint;
    pub use prefix_cache::{CachedChatCompletionOutput, CachedGenerateOutput, CrossTurnCacheStats};
    pub(crate) use sampling::{
        GpuTopkRoute, LM_HEAD_LOCAL_KS, LM_HEAD_ROWS_PER_TG, SamplingRouteEnvironment,
        SamplingRoutePlan, apply_sampling_route_plan, plan_sampling_route, sample_decode_traced,
        sample_from_candidates, sample_token,
    };
    #[cfg(test)]
    use sampling::{TEXT_CHAT_IMAGE_ERROR, choose_gpu_topk_route};
    use sampling::{decode_tokens, reject_inline_images_for_text_chat, resolve_block_local_k};
    pub(crate) use spec_types::{
        MetalGdnCheckpointPool, MetalMtpActivations, MetalMtpCache, MetalMtpDecodeMetrics,
        MetalMtpSession, MetalMtpWeights, SelfSpecMetrics, SpeculativeMetrics, SpeculativeRoute,
        SpeculativeRouteRecord,
    };
    use spec_types::{
        MetalMtpDenseMlpWeights, MetalMtpForwardOutput, MetalMtpLayerWeights, MetalStepOutput,
        MetalVerifyOutput, MtpQ4LoadResult,
    };
    use std::sync::atomic::Ordering;

    // ---------------------------------------------------------------------------
    // MSL Compute Shaders
    // ---------------------------------------------------------------------------

    // #854: the shared RMS-norm reduction helper is assembled as a prefix
    // fragment ahead of the kernel bodies that call it (Metal requires the
    // callee be visible textually before its call sites in the same
    // translation unit). The batched prefill kernels continue the same unit
    // from `qwen35_prefill.metal`, which uses `qwen35.metal` definitions
    // (`gdn_softplus`, `GDN_QK_NORM_EPS`) and so follows it.
    const MSL_SOURCE: &str = lattice_metal_runtime::concat_msl!(
        lattice_metal_runtime::msl::RMS_REDUCE,
        include_str!("shaders/qwen35.metal"),
        include_str!("shaders/qwen35_prefill.metal")
    );

    pub(super) const MSL_Q4_TILED_SOURCE: &str = include_str!("shaders/gemm_q4_tiled.metal");

    /// MSL source for the Q3 simdgroup-matrix tiled GEMM kernel (ADR-072 P1,
    /// W3 MLP-only weight format, #420 Stage 2). Mirrors [`MSL_Q4_TILED_SOURCE`]
    /// with the 16-byte plane-split 2+1 Q3 block payload substituted for Q4's
    /// 20-byte asymmetric block. See `weights::q3_weights` for the on-disk format.
    const MSL_Q3_TILED_SOURCE: &str = include_str!("shaders/gemm_q3_tiled.metal");

    /// MSL source for the Q8_0 simdgroup-matrix tiled GEMM kernel.
    ///
    /// Block layout: each Q8_0 block is 34 bytes — 2-byte f16 scale (little-endian)
    /// followed by 32 int8 values.  The kernel tiles BM=64 rows × BN=32 cols × BK=32
    /// depth and uses simdgroup_half8x8/float8x8 MMA accumulation, keeping accumulators
    /// in f32 while staging X and W as f16 in threadgroup memory.
    ///
    /// Threadgroup memory breakdown (~9.8 KB):
    ///   half  Xtg[64][32]  = 4096 bytes
    ///   char  Qraw[32][32] = 1024 bytes
    ///   half  Qscale[32]   =   64 bytes
    ///   half  Wtg[32][40]  = 2560 bytes  (BN_PAD=40 avoids bank conflicts)
    ///   float Ytg[32][16]  = 2048 bytes
    ///   float Zero[8][8]   =  256 bytes
    ///   Total              = 10048 bytes < 32 KB → 2× occupancy on M1
    const MSL_Q8_TILED_SOURCE: &str = include_str!("shaders/gemm_q8_tiled.metal");

    #[cfg(test)]
    std::thread_local! {
        static Q4_GEMM_FALLBACK_DISPATCHES_FOR_TEST: std::cell::Cell<u64> =
            const { std::cell::Cell::new(0) };
    }

    /// Weights for a GatedDeltaNet (linear attention) layer on GPU.
    pub(crate) struct MetalGdnLayerWeights {
        in_proj_qkv: Q4WeightBuf,  // [qkv_dim, hidden] — Q4/Q8
        in_proj_z: Q4WeightBuf,    // [output_dim, hidden] — Q4/Q8
        in_proj_qkvz: Q4WeightBuf, // [qkv_dim+output_dim, hidden] — Q4/Q8; concat of qkv||z rows for fused decode GEMV
        in_proj_b: Buffer,         // [num_value_heads, hidden] — f16 (CPU-read via unified mem)
        in_proj_a: Buffer,         // [num_value_heads, hidden] — f16 (CPU-read via unified mem)
        a_log: Buffer,             // [num_value_heads] — f32 (small, precision-sensitive)
        dt_bias: Buffer,           // [num_value_heads] — f32 (small, precision-sensitive)
        conv1d_weight: Buffer,     // [qkv_dim, kernel_size] — f32 (small, CPU-read)
        norm_weight: Buffer,       // [value_dim] — f32 (small, CPU-read)
        out_proj: Q4WeightBuf,     // [hidden, output_dim] — Q4/Q8
    }

    /// Weights for a full-attention (GQA) layer on GPU.
    pub(crate) struct MetalFullLayerWeights {
        q_proj: Q4WeightBuf, // [2*q_dim, hidden] (Q + gate_z interleaved) — Q4/Q8
        k_proj: Q4WeightBuf, // [kv_dim, hidden] — Q4/Q8
        v_proj: Q4WeightBuf, // [kv_dim, hidden] — Q4/Q8
        o_proj: Q4WeightBuf, // [hidden, q_dim] — Q4/Q8
        q_norm: Buffer,      // [head_dim] — f32 (small, precision-sensitive)
        k_norm: Buffer,      // [head_dim] — f32 (small, precision-sensitive)
    }

    /// Per-layer weights on GPU, discriminated by layer type.
    pub(crate) enum MetalLayerAttnWeights {
        Linear(MetalGdnLayerWeights),
        Full(MetalFullLayerWeights),
    }

    /// Where a MoE layer's routed-expert weights live: fully resident
    /// (safetensors-upload path, unchanged behavior) or a bounded LRU cache
    /// of per-expert slots dequantized on demand from a `.q4` file's mmap
    /// (#682 Stage 1, the `from_q4_dir` path only).
    enum RoutedExpertStorage {
        /// Routed expert gate+up / down projections fully resident, exactly
        /// as before #682: `gate_up` is `[num_experts * 2 * inter * hidden]`
        /// f16 (expert `e` gate at element `e * 2 * inter * hidden`, up at
        /// `e * 2 * inter * hidden + inter * hidden`), `down` is
        /// `[num_experts * hidden * inter]` f16 (expert `e` at
        /// `e * hidden * inter`). Used by `MetalQwen35Engine::new()`'s
        /// safetensors-upload constructor, which has no `.q4` file to mmap
        /// and lazily dequant from — that path's memory behavior is
        /// intentionally unchanged by #682 (see PLAN.md "KEEP behavior
        /// identical for the safetensors-upload MoE path").
        Eager { gate_up: Buffer, down: Buffer },
        /// Routed expert gate+up / down projections cached lazily (#682
        /// Stage 1): a bounded LRU pool of per-expert f16 slots, populated
        /// on demand from an mmap of the `.q4` file. Expert `e`'s resolved
        /// slot buffer holds gate at element offset `0` and up at element
        /// offset `inter * hidden` (the fused-tensor layout within one
        /// expert's slice is unchanged, just no longer addressed by a
        /// global `expert_id * ...` offset into a giant resident buffer).
        /// `RefCell` gives `encode_moe_ffn` (which only holds
        /// `&MoeMetalBuffers`) interior mutability for LRU bookkeeping —
        /// single-threaded decode, no concurrent access.
        Cached {
            gate_up: Box<std::cell::RefCell<crate::forward::moe_expert_cache::ExpertSlotCache>>,
            down: Box<std::cell::RefCell<crate::forward::moe_expert_cache::ExpertSlotCache>>,
        },
    }

    /// Pre-allocated Metal buffers for one MoE layer (ADR-053).
    ///
    /// Holds f16 expert weight buffers and f32 scratch buffers reused across tokens.
    /// All weights are resident in unified memory (StorageModeShared), except
    /// `routed` when it is `RoutedExpertStorage::Cached` (#682 Stage 1).
    struct MoeMetalBuffers {
        /// Routed expert gate+up / down projections — see [`RoutedExpertStorage`].
        routed: RoutedExpertStorage,
        /// Shared expert gate projection: [shared_inter * hidden] f16.
        shared_gate_proj: Buffer,
        /// Shared expert up projection: [shared_inter * hidden] f16.
        shared_up_proj: Buffer,
        /// Shared expert down projection: [hidden * shared_inter] f16.
        shared_down_proj: Buffer,
        /// Router gate weight: [num_experts * hidden] f32 (CPU-read for routing).
        router_gate: Buffer,
        /// Shared expert scalar gate weights: [hidden] f32 (CPU-read).
        shared_expert_gate: Buffer,
        /// Scratch: per-expert gate activation [inter] f32.
        scratch_gate: Buffer,
        /// Scratch: per-expert up activation [inter] f32 (reused across experts).
        scratch_up: Buffer,
        /// Scratch: single expert down output [hidden] f32 (before scale+accumulate).
        scratch_expert_out: Buffer,
        /// Accumulator: weighted sum of all expert outputs [hidden] f32.
        scratch_out: Buffer,
        /// Device route buffer (ADR-053 D3'), `5 * top_k + 1` u32 words written by
        /// `moe_router_topk` and `moe_shared_gate_scalar` (layout in `qwen35.metal`).
        /// `Some` only for Eager storage whose shape the device route supports;
        /// `None` keeps the synchronized CPU router.
        device_route: Option<Buffer>,
        /// Model dimensions, cached to avoid recomputing at dispatch time.
        num_experts: usize,
        inter: usize,
        shared_inter: usize,
        hidden: usize,
        top_k: usize,
    }

    /// Per-layer FFN weights on GPU: either dense SwiGLU or MoE.
    enum MetalFfnWeights {
        Dense {
            gate_up_proj: Q4WeightBuf, // [2 * intermediate, hidden] — gate || up, Q4/Q8
            down_proj: Q4WeightBuf,    // [hidden, intermediate] — Q4/Q8
        },
        Moe(Box<MoeMetalBuffers>),
    }

    /// Common per-layer weights (norms + MLP) on GPU.
    pub(crate) struct MetalCommonLayerWeights {
        input_layernorm: Buffer,          // [hidden] — f32 (norm)
        post_attention_layernorm: Buffer, // [hidden] — f32 (norm)
        ffn: MetalFfnWeights,
    }

    /// Maximum draft tokens verified in one MTP speculation round (first=target, second=MTP draft).
    const MTP_VERIFY_MAX_TOKENS: usize = 2;

    const GDN_CHUNK_SIZE: usize = 32;

    /// Parameters passed to all 5 GDN chunked-scan MSL kernels.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct GdnChunkParams {
        key_dim: u32,
        value_dim: u32,
        num_key_heads: u32,
        num_value_heads: u32,
        hidden_size: u32,
        q_total: u32,
        v_offset: u32,
        qkv_dim: u32,
        output_dim: u32,
        chunk_size: u32,
        n_tokens: u32,
        num_chunks: u32,
        active_chunk: u32,
        scale: f32,
        eps: f32,
    }

    /// Weight buffer references for one GDN layer's chunked-scan dispatch.
    #[derive(Clone, Copy)]
    struct GdnChunkedWeights<'a> {
        conv1d_weight: &'a Buffer,
        in_proj_b: &'a Buffer,
        in_proj_a: &'a Buffer,
        a_log: &'a Buffer,
        dt_bias: &'a Buffer,
        norm_weight: &'a Buffer,
    }

    /// Scratch buffers for the chunked GDN prefill scan (session-owned, reused per layer).
    struct GdnChunkScratch {
        q: Buffer,              // [chunks, heads, C, key_dim]
        k: Buffer,              // [chunks, heads, C, key_dim]
        v: Buffer,              // [chunks, heads, C, value_dim]
        beta_log_alpha: Buffer, // [chunks, heads, C, 2]
        gamma: Buffer,          // [chunks, heads, C] log-cumulative decay
        gamma_end: Buffer,      // [chunks, heads]
        kkt: Buffer,            // [chunks, heads, C, C]
        qk_l: Buffer,           // [chunks, heads, C, C]
        w: Buffer,              // [chunks, heads, C, key_dim]
        u: Buffer,              // [chunks, heads, C, value_dim]
        r: Buffer,              // [chunks, heads, C, value_dim]
        k_right: Buffer,        // [chunks, heads, C, key_dim]
        raw_out: Buffer,        // [n_tokens, output_dim]
    }

    /// Maximum GDN-only draft tokens per self-speculative round.
    const SELF_SPEC_MAX_DRAFT: usize = 4;

    /// Compiled MSL pipeline state objects.
    pub(crate) struct MetalQwen35Pipelines {
        gemv_decode: ComputePipelineState,
        gemv_decode_wide: ComputePipelineState,
        gemv_q8: ComputePipelineState,
        rms_norm: ComputePipelineState,
        partial_rope: ComputePipelineState,
        per_head_rms_norm: ComputePipelineState,
        decode_attention: ComputePipelineState,
        sigmoid_gate: ComputePipelineState,
        scatter_q_gate: ComputePipelineState,
        silu_mul: ComputePipelineState,
        silu_mul_fused: ComputePipelineState,
        copy: ComputePipelineState,
        copy_offset: ComputePipelineState,
        gemv_q4: ComputePipelineState,
        gemm_q4: ComputePipelineState,
        gemm_q4_tiled: Option<ComputePipelineState>,
        gemv_q3: ComputePipelineState,
        gemm_q3_tiled: Option<ComputePipelineState>,
        conv1d_silu: ComputePipelineState,
        gdn_recurrence: ComputePipelineState,
        gdn_chunk_materialize_c32: ComputePipelineState,
        gdn_chunk_solve_c32: ComputePipelineState,
        gdn_chunk_residual_output_c32: ComputePipelineState,
        gdn_chunk_state_update_c32: ComputePipelineState,
        gdn_chunk_norm_silu_c32: ComputePipelineState,
        gdn_chunk_conv_buf_update_c32: ComputePipelineState,
        fused_residual_add_norm: ComputePipelineState,
        copy_and_rms_norm: ComputePipelineState,
        add_and_copy: ComputePipelineState,
        copy_and_rms_norm_batch: ComputePipelineState,
        fused_residual_add_norm_batch: ComputePipelineState,
        gemm_q8: ComputePipelineState,
        gemm_q8_tiled: Option<ComputePipelineState>,
        topk_merge_pass: ComputePipelineState,
        argmax_first: ComputePipelineState,
        argmax_merge: ComputePipelineState,
        // Hierarchical k=50 SIMD-group tournament kernels (R2)
        topk_select50_first: ComputePipelineState,
        topk_select50_merge: ComputePipelineState,
        // Flash decode partitioned kernels (H3)
        decode_attn_partial: ComputePipelineState,
        decode_attn_reduce: ComputePipelineState,
        // H2+H4: Qwen3.6-27B specialized GDN kernel (optional, falls back to gdn_recurrence)
        gdn_recurrence_q36: Option<ComputePipelineState>,
        // H1+H3: Three-kernel sharded path (optional; used when all three are present)
        gdn_precompute_keys: Option<ComputePipelineState>,
        gdn_recurrence_sharded: Option<ComputePipelineState>,
        gdn_norm_silu: Option<ComputePipelineState>,
        // ADR-045: LoRA GEMV kernels (always compiled, zero cost when unused)
        lora_gemv_a: ComputePipelineState,
        lora_gemv_b_accum: ComputePipelineState,
        // Row-aware LoRA kernels for the batched prefill chunk (M token rows per dispatch)
        lora_gemm_a_rows: ComputePipelineState,
        lora_gemm_b_accum_rows: ComputePipelineState,
        // ADR-053: MoE Metal dispatch kernels
        moe_expert_gemv: ComputePipelineState,
        moe_scale_add: ComputePipelineState,
        moe_shared_gate_add: ComputePipelineState,
        moe_zero_buf: ComputePipelineState,
        // ADR-053 D3': MoE device route kernels
        moe_router_topk: ComputePipelineState,
        moe_shared_gate_scalar: ComputePipelineState,
        moe_expert_gemv_routed: ComputePipelineState,
        moe_scale_add_routed: ComputePipelineState,
        // Win 1 batch prefill prep kernels (ADR-126A)
        scatter_q_gate_batch: ComputePipelineState,
        per_head_rms_norm_batch: ComputePipelineState,
        partial_rope_batch: ComputePipelineState,
        copy_kv_cache_batch: ComputePipelineState,
        // Win 2 batched causal prefill attention (ADR-126B)
        prefill_attention_batched: ComputePipelineState,
        // f16 KV cache variants (#154): used when use_kv_f16 is true
        copy_offset_f16: ComputePipelineState,
        copy_kv_cache_batch_f16: ComputePipelineState,
        decode_attention_f16: ComputePipelineState,
        decode_attn_partial_f16: ComputePipelineState,
        prefill_attention_batched_f16: ComputePipelineState,
        // lm_head two-stage block top-k (issue #171): Stage 1 fused GEMV +
        // block-local argmax/top-k. Indexed by LM_HEAD_LOCAL_KS position.
        lm_head_block_topk_f16: [ComputePipelineState; 5],
        lm_head_block_topk_q4: [ComputePipelineState; 5],
    }

    /// Decide whether the issue #171 lm_head two-stage block-top-k path
    /// applies. Tried before `choose_gpu_topk_route`; falls back to
    /// `GpuTopkRoute::CpuFallback` (which the caller then re-resolves via the
    /// older function) when this round's compiled LOCAL_K set or top-p exactness
    /// requirement rule it out.
    ///
    /// `top_p < 1.0` requires `approx_topp_env` (`LATTICE_COMPACT_TOPP_APPROX`):
    /// nucleus sampling over the compact top-k candidates is NOT equivalent to
    /// full-vocab nucleus sampling once the tail is truncated, so exact mode is
    /// the default and the approximation is opt-in only.
    ///
    /// `top_k == 50` intentionally stays out of
    /// `LM_HEAD_LOCAL_KS` and therefore falls through to `CpuFallback` here, so
    /// the caller re-resolves it via `choose_gpu_topk_route`'s `HierarchicalK50`
    /// path, which still does a full-logit GEMV before its compact dispatch.
    /// This round's threadgroup-reduction claim (<=1,500 greedy / <=2,000
    /// top-k<=64) is scoped to the precompiled `LM_HEAD_LOCAL_KS` set
    /// `{1, 8, 16, 40, 64}`; adding a `local_k=50` Stage-1 pipeline variant
    /// (new MSL function constant, precompile, agreement/bench coverage) is
    /// out of scope for this PR and tracked as follow-up work rather than
    /// silently claimed here.
    fn choose_gpu_block_topk_route(
        top_k: usize,
        top_p: f32,
        compact_env: bool,
        approx_topp_env: bool,
    ) -> GpuTopkRoute {
        if !compact_env {
            return GpuTopkRoute::CpuFallback;
        }
        if top_p < 1.0 && !approx_topp_env {
            return GpuTopkRoute::CpuFallback;
        }
        match top_k {
            // top_k == 0 means top-k is DISABLED (full-distribution sampling,
            // see `SamplingConfig`), NOT greedy: routing it to BlockArgmax
            // would silently collapse sampling to top-1 on the opt-in path.
            // It must fall through to CpuFallback, whose legacy route keeps
            // the full logit distribution available to the sampler.
            1 => GpuTopkRoute::BlockArgmax,
            8 | 16 | 40 | 64 => GpuTopkRoute::BlockTopK {
                local_k: top_k as u32,
            },
            _ => GpuTopkRoute::CpuFallback,
        }
    }

    /// **Unstable**: Metal GPU forward pass for Qwen3.5/3.6; kernel set and weight layout evolving.
    ///
    /// Backward-compatible convenience wrapper bundling one `MetalQwen35Engine` (immutable
    /// model state) with one `InferenceSession` (mutable per-request state). Existing callers
    /// continue using this type unchanged. For concurrent multi-session serving, construct the
    /// engine and sessions separately once the public concurrent API ships.
    pub struct MetalQwen35State {
        pub(crate) engine: MetalQwen35Engine,
        pub(crate) session: InferenceSession,
        pub(crate) lora: Option<MetalLoraAdapter>,
        /// Per-instance flag: use the chunked-parallel GDN prefill scan (C=32).
        ///
        /// Defaults to `true` (default-ON) unless `LATTICE_GDN_CHUNKED=0` or
        /// `LATTICE_GDN_CHUNKED=false` is set at construction time. Toggle this
        /// field directly in tests to switch paths without touching the process
        /// environment (thread-safe under parallel test execution).
        pub(crate) use_gdn_chunked: bool,
        /// Per-instance flag: store KV cache in f16 (default OFF, opt-in).
        ///
        /// Halves KV memory and bandwidth for the GQA full-attention layers.
        /// f32 (default) is byte-identical to the pre-#154 path; greedy output and
        /// perplexity are unchanged when enabled (verified), but no decode speedup
        /// was measured at 1k context (the model is GatedDeltaNet-dominated, so KV
        /// reads are a small fraction of decode bandwidth). Set `LATTICE_KV_F16=1`
        /// or `LATTICE_KV_F16=true` at construction time to enable.
        pub(crate) use_kv_f16: bool,
        /// Cross-turn KV/GDN prefix cache (#462). Worker-local: never shared
        /// across threads or processes. Empty by default; only populated by
        /// the `*_with_prefix_cache` entry points.
        cross_turn_prefix_cache: MetalCrossTurnPrefixCache,
        /// Per-instance flag: whether the path-proof counters below are live.
        /// Read once from `LATTICE_METAL_PATH_PROOF` at construction time
        /// (same pattern as `use_kv_f16`) so every dispatch-site `fetch_add`
        /// is a no-op — not just relaxed-but-still-paid — on the default
        /// (disabled) path. See `PathProofCounters` for what this gates.
        pub(crate) path_proof_enabled: bool,
        /// Runtime path-proof counters (issue #239): prove which Metal attention/KV
        /// dispatch helpers actually ran, so CI can fail closed on a paravirtual
        /// runner that reports a Metal device but silently skips required kernels.
        /// Only recorded when `path_proof_enabled` is set (opt-in, zero cost
        /// otherwise); read via `path_proof_snapshot` and zeroed via
        /// `reset_path_proof_counters`.
        pub(crate) path_proof: PathProofCounters,
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    // Flash decode shape invariants: HEAD_DIM=256, GQA group ∈ [1,8].
    const METAL_FLASH_HEAD_DIM: usize = 256;
    const METAL_FLASH_MAX_GQA_GROUP: usize = 8;
    const METAL_FLASH_PARTITION_TOKENS: usize = 1024;

    fn validate_flash_decode_shape(
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<(), String> {
        if head_dim != METAL_FLASH_HEAD_DIM {
            return Err(format!(
                "Metal FlashAttention decode supports head_dim={METAL_FLASH_HEAD_DIM} only; got {head_dim}"
            ));
        }
        if num_q_heads == 0 || num_kv_heads == 0 {
            return Err(format!(
                "Metal FlashAttention decode requires nonzero heads; got num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"
            ));
        }
        if !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(format!(
                "Metal FlashAttention decode requires num_q_heads divisible by num_kv_heads; got {num_q_heads}/{num_kv_heads}"
            ));
        }
        let group_size = num_q_heads / num_kv_heads;
        if group_size == 0 || group_size > METAL_FLASH_MAX_GQA_GROUP {
            return Err(format!(
                "Metal FlashAttention decode supports GQA group size 1..={METAL_FLASH_MAX_GQA_GROUP}; got {group_size}"
            ));
        }
        let expected_q_dim = num_q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Metal FlashAttention q_dim overflow".to_string())?;
        let expected_kv_dim = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Metal FlashAttention kv_dim overflow".to_string())?;
        if q_dim != expected_q_dim {
            return Err(format!(
                "Metal FlashAttention q_dim mismatch: got {q_dim}, expected {expected_q_dim}"
            ));
        }
        if kv_dim != expected_kv_dim {
            return Err(format!(
                "Metal FlashAttention kv_dim mismatch: got {kv_dim}, expected {expected_kv_dim}"
            ));
        }
        Ok(())
    }

    /// Maps a model's GQA `group_size` (`num_attention_heads / num_key_value_heads`)
    /// to the smallest specialized `prefill_attention_batched_causal_{suffix}[_f16]`
    /// kernel variant whose compile-time `MAX_GRP` can hold it.
    ///
    /// Each variant sizes its threadgroup arrays (`q_s`/`score_s`/`m_s`/`l_s`/`alpha_s`)
    /// and per-thread `acc[MAX_GRP]` register array to a fixed `MAX_GRP` (see
    /// `DEFINE_PREFILL_ATTENTION_BATCHED_CAUSAL` in `MSL_SOURCE`), so smaller-group
    /// models (e.g. gs=4) get more threadgroup-memory occupancy by picking the g4
    /// variant instead of paying for MAX_GRP=8-sized arrays whose rows above
    /// group_size were always idle.
    ///
    /// SHIP-SAFETY GUARD: the kernel body itself is defense-in-depth
    /// (`if group_size == 0 || group_size > MAX_GRP { return; }`), but that guard
    /// makes the failure mode *silent* (zeroed/never-written attention output for
    /// the affected rows), not a compile error or panic. This mapping must always
    /// round UP to a variant whose `MAX_GRP >= group_size` — picking `g4` for a
    /// `group_size=6` model would silently drop attention output for 2 of every 6
    /// query heads in the group. `group_size` is validated to be in
    /// `1..=METAL_FLASH_MAX_GQA_GROUP` (8) by `validate_flash_decode_shape` before
    /// either constructor reaches the shared pipeline builder. This helper
    /// re-checks that invariant and returns `Err` for any
    /// out-of-range input rather than rounding it: there is no variant with
    /// `MAX_GRP > 8`, so a `group_size >= 9` could only be mapped to `g8`, whose
    /// kernel guard would then silently write no attention. Failing closed here
    /// keeps the "never select `MAX_GRP < group_size`" guarantee true even if a
    /// future caller reaches this without validating first.
    fn prefill_maxgrp_suffix(group_size: usize) -> Result<&'static str, String> {
        match group_size {
            1..=4 => Ok("g4"),
            5..=6 => Ok("g6"),
            7..=8 => Ok("g8"),
            _ => Err(format!(
                "prefill attention supports GQA group size 1..={METAL_FLASH_MAX_GQA_GROUP}; \
                 got {group_size} (no kernel variant has MAX_GRP > {METAL_FLASH_MAX_GQA_GROUP})"
            )),
        }
    }

    #[cfg(test)]
    mod min_p_compact_sampling_tests;

    /// **Unstable**: output from chat completion; fields may expand with streaming and usage stats.
    ///
    /// Output from chat completion.
    #[derive(Debug, Clone)]
    pub struct ChatCompletionOutput {
        pub message: ChatMessage,
        pub prompt_tokens: usize,
        pub completion_tokens: usize,
        /// The engine's actual stop cause (ADR-080 C2, #746): `true` when
        /// generation ended via an explicit stop condition (EOS,
        /// stop-token-id, or stop-string match, mirrors
        /// `GenerateOutput::stopped`); `false` when the token budget was
        /// exhausted or the caller cancelled. Previously this struct carried
        /// no stop-cause information at all, so every consumer that wanted
        /// an OpenAI `finish_reason` had no choice but to hardcode `"stop"`
        /// unconditionally -- exactly what `lattice_serve.rs`'s worker did.
        pub stopped: bool,
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests;
    // -----------------------------------------------------------------------
    // Blend correctness tests (inference-side LoraLayerData types)
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod blend_tests;

    /// Bench-only access to the real lm_head dispatch (issue #547). Kept
    /// behind `bench-internals` so `LmHeadBenchRoute`/`read_topk_candidates`
    /// visibility stays out of the normal public API; `run_once` calls the
    /// same `encode_final_head` the production decode path uses, so the
    /// group measures the shipped kernels rather than a bench-only
    /// reimplementation.
    #[cfg(feature = "bench-internals")]
    pub mod bench_support;
}

// Target-independent numerical support follows the macOS + Metal implementation.

// Stage-1 block-top-k and the full-logit lm_head both accumulate ~1K f32
// products for the real 0.8B checkpoint, but through different kernels/order.
// f32 unit roundoff is ~6e-8, so gamma_1024 is ~6e-5; for logits in the tens,
// boundary noise lands around 1e-4..1e-3. Existing lm_head GPU/full-logit
// parity checks in this file use 1e-3 absolute logit tolerance, so use that
// same scale only to accept rank-boundary ties, never broad ordering changes.
#[cfg(test)]
const LM_HEAD_TOPK_TIE_EPSILON: f32 = 1.0e-3;

// Q4 carries an additional noise floor on top of the f32-accumulation-order
// noise the F16 epsilon above already accounts for: the block-topk Stage-1
// kernel and the full-logit GEMV each dequantize the Q4 4-bit blocks before
// accumulating, and the two paths' dequantize+accumulate orders round
// differently. That makes Q4 boundary noise measurably wider than F16's, so
// reusing LM_HEAD_TOPK_TIE_EPSILON here under-tolerates and produces the
// exact flaky failures this constant exists to fix (issue #623).
//
// Value derived empirically (2026-07-03), not guessed, in TWO phases — the
// second phase revised the first upward, and that revision (not just the
// final number) is worth keeping on record:
//
// Phase 1 (initial 5x survey): 5 repeated runs each of
// lm_head_q4_real_checkpoint_greedy_agreement (2000 positions) and
// lm_head_q4_real_checkpoint_topk_set_agreement (100 positions x k in
// {8,16,40,64}) against the real (non-stale) Qwen3.5-0.8B Q4 checkpoint,
// logging the boundary margin of every mismatch via temp instrumentation.
// Observed margins: greedy {7.91e-3, 6.90e-4, 1.25e-2} (2 runs clean); SET,
// all at k=40 in one run, {1.18e-3, 1.69e-3, 6.17e-4, 1.14e-4} (4 runs
// clean). Phase-1 max 1.2513e-2, which an earlier revision of this constant
// set to ~1.6x (2.0e-2).
//
// Phase 2 (post-fix 3x confirmation, same day): rerunning
// lm_head_q4_real_checkpoint_greedy_agreement 3 more times against the final
// comparator-based code (i.e. no longer count-and-continue; each run panics
// on its first non-tie mismatch) surfaced margins of 3.61e-2, 2.01e-2, and
// 5.23e-2 — all 3 runs mismatched, and the new max (5.23e-2) is 4.2x
// phase-1's. This run was isolated (no concurrent Metal GPU process on the
// machine, verified via the machine-wide GPU test lock), so it is not
// explained by the cross-process GPU contention noted below — it is the
// more likely true noise ceiling, phase-1's 5 samples having under-covered
// it (consistent with issue #623: this kernel family's reduction order is
// not run-to-run deterministic, so small samples under-cover the tail).
//
// A separate, later confirmation pass on lm_head_q4_real_checkpoint_topk_set_agreement
// (4 more runs) produced a much larger outlier margin (0.167, k=8, step 3) that
// this constant deliberately does NOT chase: that run was contended by a
// concurrent Metal GPU test process on the same machine (see the
// machine-wide GPU test lock this repo now enforces for exactly this
// reason). Concurrent GPU load corrupting numerics (not just timing) for
// these exact tests is an already-documented risk in this repo -- exactly
// what the machine-wide `gpu_test_lock()` advisory flock serializes against
// -- so that batch's data is excluded from this constant's derivation.
//
// This constant is set to 1.0e-1: ~1.9x the phase-1+phase-2 combined greedy
// max (5.23e-2, from data with no identified contention), inside the
// analysis's suggested 1.5-2x safety band, and a clean round number that is
// exactly 100x LM_HEAD_TOPK_TIE_EPSILON. Follow-up: a clean (verified
// uncontended) re-confirmation of both Q4 tests would still be worthwhile,
// since phase-2 shows a single 5x survey can under-cover this kernel
// family's tail even before contention is considered.
//
// Unlike LM_HEAD_TOPK_TIE_EPSILON above, this constant's only consumers are
// the two Q4 real-checkpoint GPU tests inside `mod inner::tests`, which
// require the metal-gpu feature (there is no Q4-flavored pure-logic
// comparator unit test below). A bare `#[cfg(test)]` would leave it unused
// — and clippy-`-D warnings`-flagged — under `--tests --features f16`
// without `metal-gpu`, so gate it on the same `all(...)` its consumers live
// behind.
#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
const LM_HEAD_TOPK_TIE_EPSILON_Q4: f32 = 1.0e-1;

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
struct TopkSetDiffToken {
    token_id: u32,
    baseline_logit: f32,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopkSetMismatchReason {
    InvalidInput,
    SetCardinalityMismatch,
    BoundaryMarginTooWide,
    DifferingTokenOutsideBoundary,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
struct TopkSetMismatch {
    k: usize,
    margin: f32,
    epsilon: f32,
    boundary_logit: f32,
    differing_tokens: Vec<TopkSetDiffToken>,
    reason: TopkSetMismatchReason,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
enum TopkSetAgreement {
    ExactSet,
    AcceptedBoundaryTie {
        margin: f32,
        epsilon: f32,
        boundary_logit: f32,
        differing_tokens: Vec<TopkSetDiffToken>,
    },
}

#[cfg(test)]
fn topk_set_mismatch(
    k: usize,
    margin: f32,
    epsilon: f32,
    boundary_logit: f32,
    differing_tokens: &[TopkSetDiffToken],
    reason: TopkSetMismatchReason,
) -> TopkSetMismatch {
    TopkSetMismatch {
        k,
        margin,
        epsilon,
        boundary_logit,
        differing_tokens: differing_tokens.to_vec(),
        reason,
    }
}

#[cfg(test)]
fn topk_set_agreement_or_boundary_tie(
    baseline_logits: &[f32],
    baseline_set: &std::collections::HashSet<u32>,
    candidate_set: &std::collections::HashSet<u32>,
    k: usize,
    epsilon: f32,
) -> Result<TopkSetAgreement, TopkSetMismatch> {
    if baseline_set == candidate_set {
        return Ok(TopkSetAgreement::ExactSet);
    }

    let mut differing_tokens: Vec<TopkSetDiffToken> = baseline_set
        .symmetric_difference(candidate_set)
        .map(|&token_id| TopkSetDiffToken {
            token_id,
            baseline_logit: baseline_logits
                .get(token_id as usize)
                .copied()
                .unwrap_or(f32::NAN),
        })
        .collect();
    differing_tokens.sort_by_key(|diff| diff.token_id);

    if k == 0 || k >= baseline_logits.len() || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(topk_set_mismatch(
            k,
            f32::NAN,
            epsilon,
            f32::NAN,
            &differing_tokens,
            TopkSetMismatchReason::InvalidInput,
        ));
    }

    let mut sorted_logits = baseline_logits.to_vec();
    sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let boundary_logit = sorted_logits[k - 1];
    let first_excluded_logit = sorted_logits[k];
    let margin = boundary_logit - first_excluded_logit;

    if baseline_set.len() != k || candidate_set.len() != k {
        return Err(topk_set_mismatch(
            k,
            margin,
            epsilon,
            boundary_logit,
            &differing_tokens,
            TopkSetMismatchReason::SetCardinalityMismatch,
        ));
    }

    if margin >= epsilon {
        return Err(topk_set_mismatch(
            k,
            margin,
            epsilon,
            boundary_logit,
            &differing_tokens,
            TopkSetMismatchReason::BoundaryMarginTooWide,
        ));
    }

    if differing_tokens
        .iter()
        .all(|diff| (diff.baseline_logit - boundary_logit).abs() <= epsilon)
    {
        Ok(TopkSetAgreement::AcceptedBoundaryTie {
            margin,
            epsilon,
            boundary_logit,
            differing_tokens,
        })
    } else {
        Err(topk_set_mismatch(
            k,
            margin,
            epsilon,
            boundary_logit,
            &differing_tokens,
            TopkSetMismatchReason::DifferingTokenOutsideBoundary,
        ))
    }
}

#[cfg(test)]
mod topk_boundary_tie_tests;

// ---------------------------------------------------------------------------
// GPU-free preflight helpers for the multimodal generation path.
//
// These functions live at module level (no cfg gate) so they compile and are
// testable on every platform without Metal GPU hardware.
// ---------------------------------------------------------------------------

/// Validates `gen_cfg` for the multimodal generation path before any Metal
/// device or GPU work is initiated.
///
/// Grammar-constrained decoding is not wired into the multimodal forward
/// pass; this function converts a silent correctness failure (unconstrained
/// output despite `gen_cfg.grammar` being set) into a typed `InvalidInput`
/// error that callers can act on. Per-token logprob capture (#585) is not
/// wired into this path either, so `gen_cfg.logprobs` is rejected the same
/// way rather than silently returning an empty `token_logprobs`.
///
/// Reasoning-budget forcing (ADR-080 C3, #783) is not wired into this path
/// either (no `decode_cap` / `force_close_think`), so `gen_cfg.reasoning_budget`
/// is rejected the same way. `gen_cfg.stop_strings` does NOT need an
/// equivalent guard here: `generate_multimodal` already applies it via its own
/// `stop_text_state` / `StopStringMatcher` wiring further down.
///
/// Compiled when Metal-GPU is enabled (the production caller lives inside
/// `mod inner`) or during test builds so that the module-level test can
/// exercise the guard logic on any platform without GPU hardware.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn multimodal_generate_preflight(
    gen_cfg: &crate::generation::GenerateConfig,
) -> Result<(), lattice_inference_core::InferenceError> {
    crate::model::qwen35::check_grammar_not_set(gen_cfg)?;
    crate::model::qwen35::check_logprobs_not_set(gen_cfg)?;
    crate::model::qwen35::check_reasoning_budget_not_set(gen_cfg)
}

#[cfg(test)]
mod multimodal_preflight_tests;

// ---------------------------------------------------------------------------
// GPU-free grammar fail-closed guard (mirrors CPU `has_finite_logit`, #611).
//
// `generate`, `generate_streaming_with_cancel`, and
// `generate_streaming_with_prefix_cache_inner` all apply grammar masking via
// `GrammarEngine::mask_logits`, which sets every disallowed logit position to
// `f32::NEG_INFINITY`. `crate::model::qwen35::generation` (the canonical CPU path)
// guards every `mask_logits` call with a `has_finite_logit`
// check and fail closed with `InferenceError::GrammarConstraintBlocked` when
// the grammar blocks every token — otherwise the sampler's non-finite-max short-circuit
// would silently return the first candidate's token id (numerically safe, but
// a silent violation of the grammar contract). This is that same guard,
// mirrored for the Metal paths.
//
// Lives at module level (no cfg gate beyond test/macos+metal-gpu) so it
// compiles and is unit-testable on every platform without Metal GPU hardware,
// the same way `multimodal_generate_preflight` above is.
// ---------------------------------------------------------------------------

/// Returns `true` when at least one logit is strictly greater than
/// `f32::NEG_INFINITY` — i.e. the grammar mask leaves at least one legal token.
#[cfg(test)]
pub(crate) fn has_finite_logit(logits: &[f32]) -> bool {
    logits.iter().any(|&l| l > f32::NEG_INFINITY)
}

#[cfg(test)]
mod has_finite_logit_tests;

/// Pure decision for whether to warn that MTP was requested but its weights
/// did not load, and with which message.
///
/// Lives at module level (no cfg gate) so it compiles and is unit-testable on
/// every platform without Metal GPU hardware.
pub fn mtp_missing_weights_warning(
    mtp_requested: bool,
    mtp_loaded: bool,
    missing_file: Option<&std::path::Path>,
) -> Option<String> {
    if !mtp_requested || mtp_loaded {
        return None;
    }

    let missing_file = missing_file?;
    Some(format!(
        "LATTICE_MTP=1 requested MTP, but first required weight file is missing: {}; MTP is DISABLED, decode falls back to the plain path.",
        missing_file.display()
    ))
}

// ---------------------------------------------------------------------------
// Pure, GPU-free decision logic for the greedy MTP loop.
//
// Lives at module level (no cfg gate) so it compiles and is unit-testable on
// every platform without Metal GPU hardware.
// ---------------------------------------------------------------------------

/// Which target hidden state the MTP draft head receives as its hidden-state input.
///
/// The reference Qwen3.5 MTP implementations feed the head the target's hidden
/// state after the model's final RMSNorm, the same vector the language-model head
/// reads. The target forward captures the hidden state before that norm, so the
/// head applies the final norm itself unless the checkpoint cannot supply the
/// norm's scale. The tap is decided once when the MTP weights load and does not
/// change afterwards; query it with `MetalQwen35State::mtp_hidden_tap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpHiddenTap {
    /// The head receives the target's hidden state after the final RMSNorm,
    /// matching the reference implementations.
    PostFinalNorm,
    /// The head receives the target's hidden state before the final RMSNorm.
    /// Used for rotated (QuaRot) checkpoints: conversion folds the final-norm
    /// scale into the language-model head and stores a neutral scale in its
    /// place, so the original scale is not available at runtime.
    PreFinalNorm,
}

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
impl MtpHiddenTap {
    /// The tap for a checkpoint: rotated checkpoints keep the pre-norm hidden.
    pub(crate) fn for_checkpoint(rotated: bool) -> Self {
        if rotated {
            Self::PreFinalNorm
        } else {
            Self::PostFinalNorm
        }
    }

    /// Stable lowercase name used in load and statistics log lines.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::PostFinalNorm => "post-final-norm",
            Self::PreFinalNorm => "pre-final-norm",
        }
    }
}

#[cfg(test)]
mod mtp_hidden_tap_tests;

/// Outcome of one greedy-MTP round after verification against the target model.
#[derive(Debug, PartialEq, Eq)]
pub enum MtpRoundOutcome {
    /// Emit these tokens (in order) and then stop generation.
    EmitAndStop(Vec<u32>),
    /// Emit these tokens and continue with the given next pending token.
    EmitAndContinue { emit: Vec<u32>, next_pending: u32 },
}

/// Pure per-round decision for the greedy-MTP speculative loop.
///
/// The caller (Metal loop) handles all GPU state mutations (KV-cache advance,
/// rollback, GDN checkpoint repair). This function only decides *what tokens
/// to emit* and *whether to stop or continue*, making it fully testable
/// without GPU hardware.
///
/// # Arguments
///
/// * `pending_token` – The target model's already-decided greedy token for the
///   current position (came from prefill argmax or a previous verify argmax).
/// * `draft_token` – The cheap draft model's proposal for position + 1.
/// * `accepted` – `true` iff `draft_token == argmax(target_logits_0)`.
/// * `bonus_token` – On accept: `argmax(target_logits_1)` (prediction for
///   position + 2). On reject: `argmax(target_logits_0)` (target's own
///   choice for the draft position, replaces the rejected draft).
/// * `is_stop` – Returns `true` for EOS / stop-token IDs.
///
/// # Stop-token contract (#613, matches canonical plain greedy generation)
///
/// Plain greedy never appends the terminating stop token to the output
/// (it checks the stop condition before appending the token). This function
/// honours the same invariant for every terminal case: a token that is itself
/// a stop token is never included in the emitted vector.
///
/// **Accept** (`accepted == true`):
/// - If `draft_token` is a stop token: `EmitAndStop([pending])` — the accepted
///   draft is excluded. The bonus (pos+2 prediction) is irrelevant once
///   generation ends. (This still resolves #237 concern 2: the draft's EOS
///   guess is verified against the target before any stop decision is made,
///   rather than short-circuited before verify.)
/// - If `bonus_token` is a stop token: `EmitAndStop([pending, draft_token])` —
///   both committed tokens are emitted; the bonus that would have followed is
///   excluded.
/// - Otherwise: `EmitAndContinue { emit: [pending, draft_token], next_pending: bonus }`.
///
/// **Reject** (`accepted == false`):
/// - `bonus_token` is the target's own greedy choice at the draft position.
/// - If `bonus_token` is a stop token: `EmitAndStop([pending])` — the
///   replacement stop token is excluded; only the already-committed `pending`
///   is emitted.
/// - Otherwise: `EmitAndContinue { emit: [pending], next_pending: bonus }`.
pub fn mtp_greedy_round(
    pending_token: u32,
    draft_token: u32,
    accepted: bool,
    bonus_token: u32,
    is_stop: impl Fn(u32) -> bool,
) -> MtpRoundOutcome {
    if accepted {
        // Draft accepted: target agrees draft_token belongs at pos+1.
        if is_stop(draft_token) {
            // The accepted token is itself a stop — exclude it, emit only
            // the already-committed pending token, and halt.
            return MtpRoundOutcome::EmitAndStop(vec![pending_token]);
        }
        if is_stop(bonus_token) {
            // Accepted non-stop draft, but target's next greedy token is a
            // stop. Emit the committed [pending, draft_token]; the bonus stop
            // token itself is excluded.
            return MtpRoundOutcome::EmitAndStop(vec![pending_token, draft_token]);
        }
        MtpRoundOutcome::EmitAndContinue {
            emit: vec![pending_token, draft_token],
            next_pending: bonus_token,
        }
    } else {
        // Draft rejected.  Emit pending only; caller rolls back GPU state.
        // bonus_token is the target's replacement greedy token at the draft position.
        if is_stop(bonus_token) {
            // Target's own replacement is a stop — exclude it, emit only the
            // already-committed pending token.
            return MtpRoundOutcome::EmitAndStop(vec![pending_token]);
        }
        MtpRoundOutcome::EmitAndContinue {
            emit: vec![pending_token],
            next_pending: bonus_token,
        }
    }
}

// ---------------------------------------------------------------------------
// GDN state-traffic counters (issue #422): byte-level accounting of GDN
// recurrent-state read/write/copy traffic per decode step and per MTP
// verify, the measure-first prerequisite for ADR-058.
//
// The scope/copy-kind marker enums stay unconditional (zero-cost, no
// dependencies) so `forward_step_inner_impl` can carry an explicit scope
// parameter through the shared hot loop without a second, duplicated
// implementation gated on the feature. All byte-accounting types and their
// arithmetic are gated behind `gdn-state-counters` and live at module level
// (no macOS/metal-gpu cfg) so the formulas are unit-testable on every
// platform. Default builds compile out every accounting hook and stay
// byte-identical to the pre-#422 decode path (ADR-422-1).
// ---------------------------------------------------------------------------

/// Which logical traffic bucket a GDN state access belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GdnStateTrafficScope {
    /// Normal single-token decode (`forward_step`, self-spec draft/fallback).
    Decode,
    /// MTP target verification, including checkpoint/restore traffic.
    MtpVerify,
}

/// Which direction a GDN state checkpoint-pool copy moved data.
#[cfg(feature = "gdn-state-counters")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnStateCopyKind {
    /// Live GDN buffers copied into a checkpoint slot.
    Snapshot,
    /// A checkpoint slot copied back into the live GDN buffers.
    Restore,
}

/// Per-layer and aggregate GDN recurrent-state byte shape, derived from config.
///
/// Mirrors the buffer sizes `new_session_inner` allocates: each typed Metal GDN
/// layer owns `linear_qkv_dim * (linear_conv_kernel_dim - 1)` conv-history f32
/// values and `linear_num_value_heads * linear_value_head_dim *
/// linear_key_head_dim` recurrent-matrix f32 values.
#[cfg(feature = "gdn-state-counters")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnStateTrafficShape {
    pub active_gdn_layers: usize,
    pub allocated_gdn_layers: usize,
    pub conv_bytes_per_layer: u64,
    pub s_bytes_per_layer: u64,
    pub per_layer_state_bytes: u64,
    pub active_state_bytes: u64,
    pub allocated_state_bytes: u64,
}

#[cfg(feature = "gdn-state-counters")]
impl GdnStateTrafficShape {
    /// Derives the GDN state byte shape from a resolved `Qwen35Config`.
    pub fn try_from_config(
        cfg: &lattice_arch::qwen35_config::Qwen35Config,
    ) -> Result<Self, lattice_inference_core::InferenceError> {
        fn overflow(what: &str) -> lattice_inference_core::InferenceError {
            lattice_inference_core::InferenceError::Inference(format!(
                "GDN state shape: {what} overflow"
            ))
        }

        let active_gdn_layers = cfg.num_active_linear_attention_layers();
        let allocated_gdn_layers = cfg.num_linear_attention_layers();

        let conv_floats = cfg
            .linear_qkv_dim()
            .checked_mul(cfg.linear_conv_kernel_dim.saturating_sub(1))
            .ok_or_else(|| overflow("conv_floats"))?;
        let s_floats = cfg
            .linear_num_value_heads()
            .checked_mul(cfg.linear_value_head_dim)
            .and_then(|v| v.checked_mul(cfg.linear_key_head_dim))
            .ok_or_else(|| overflow("s_floats"))?;

        let conv_bytes_per_layer = (conv_floats as u64)
            .checked_mul(4)
            .ok_or_else(|| overflow("conv_bytes_per_layer"))?;
        let s_bytes_per_layer = (s_floats as u64)
            .checked_mul(4)
            .ok_or_else(|| overflow("s_bytes_per_layer"))?;
        let per_layer_state_bytes = conv_bytes_per_layer
            .checked_add(s_bytes_per_layer)
            .ok_or_else(|| overflow("per_layer_state_bytes"))?;
        let active_state_bytes = per_layer_state_bytes
            .checked_mul(active_gdn_layers as u64)
            .ok_or_else(|| overflow("active_state_bytes"))?;
        let allocated_state_bytes = per_layer_state_bytes
            .checked_mul(allocated_gdn_layers as u64)
            .ok_or_else(|| overflow("allocated_state_bytes"))?;

        Ok(Self {
            active_gdn_layers,
            allocated_gdn_layers,
            conv_bytes_per_layer,
            s_bytes_per_layer,
            per_layer_state_bytes,
            active_state_bytes,
            allocated_state_bytes,
        })
    }

    /// Re-points `allocated_gdn_layers` / `allocated_state_bytes` at the GDN
    /// state buffers actually allocated for this session. `try_from_config`
    /// defaults `allocated` to the architectural `num_linear_attention_layers()`
    /// (what the f16 constructor allocates), but the Q4 constructor allocates
    /// only `num_active_linear_attention_layers()` buffers. The checkpoint /
    /// restore blit copies exactly the allocated buffers, so `record_copy` must
    /// charge the real count to avoid overcounting a pruned Q4 session.
    pub fn with_allocated_layers(mut self, allocated_gdn_layers: usize) -> Self {
        self.allocated_gdn_layers = allocated_gdn_layers;
        self.allocated_state_bytes = self
            .per_layer_state_bytes
            .saturating_mul(allocated_gdn_layers as u64);
        self
    }
}

/// Read/write/copy byte totals for one traffic scope (decode or MTP verify).
#[cfg(feature = "gdn-state-counters")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnStateTrafficBucket {
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub snapshot_count: u64,
    pub restore_count: u64,
}

#[cfg(feature = "gdn-state-counters")]
impl GdnStateTrafficBucket {
    pub fn copy_count(&self) -> u64 {
        self.snapshot_count.saturating_add(self.restore_count)
    }
}

/// Full GDN state-traffic report for one session snapshot window.
#[cfg(feature = "gdn-state-counters")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnStateTrafficReport {
    pub shape: GdnStateTrafficShape,
    pub decode: GdnStateTrafficBucket,
    pub mtp_verify: GdnStateTrafficBucket,
}

#[cfg(feature = "gdn-state-counters")]
impl GdnStateTrafficReport {
    pub fn reset(&mut self) {
        self.decode = GdnStateTrafficBucket::default();
        self.mtp_verify = GdnStateTrafficBucket::default();
    }
}

/// Mutable per-session GDN state-traffic accumulator.
///
/// Not `Clone`-cheap by design intent (it is session-owned mutable state);
/// derives `Clone` only because `report()` snapshots are taken by value.
///
/// Only constructed in production from `mod inner` (macOS + `metal-gpu`); the
/// extra `any(test, ...)` arm compiles it for the module-level unit tests on
/// every platform without making it dead code in a `gdn-state-counters`-only,
/// non-metal-gpu lib build (no runtime behavior implied by either arm).
#[cfg(feature = "gdn-state-counters")]
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GdnStateTrafficCounters {
    report: GdnStateTrafficReport,
}

#[cfg(feature = "gdn-state-counters")]
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
impl GdnStateTrafficCounters {
    pub(crate) fn new(shape: GdnStateTrafficShape) -> Self {
        Self {
            report: GdnStateTrafficReport {
                shape,
                decode: GdnStateTrafficBucket::default(),
                mtp_verify: GdnStateTrafficBucket::default(),
            },
        }
    }

    pub(crate) fn report(&self) -> GdnStateTrafficReport {
        self.report
    }

    pub(crate) fn reset(&mut self) {
        self.report.reset();
    }

    fn bucket_mut(&mut self, scope: GdnStateTrafficScope) -> &mut GdnStateTrafficBucket {
        match scope {
            GdnStateTrafficScope::Decode => &mut self.report.decode,
            GdnStateTrafficScope::MtpVerify => &mut self.report.mtp_verify,
        }
    }

    /// Records one logical full-state read + write for `layer_count` GDN layer
    /// dispatches (one `encode_gdn_layer` call = one active layer accessed).
    pub(crate) fn record_layer_access(&mut self, scope: GdnStateTrafficScope, layer_count: usize) {
        let bytes = self
            .report
            .shape
            .per_layer_state_bytes
            .saturating_mul(layer_count as u64);
        let bucket = self.bucket_mut(scope);
        bucket.read_bytes = bucket.read_bytes.saturating_add(bytes);
        bucket.write_bytes = bucket.write_bytes.saturating_add(bytes);
    }

    /// Records one checkpoint-pool copy: a full logical read of the source
    /// side plus a full logical write of the destination side, sized to the
    /// *allocated* (not just active) GDN layer count, since the checkpoint
    /// blit copies every allocated layer's buffers regardless of pruning.
    pub(crate) fn record_copy(&mut self, scope: GdnStateTrafficScope, kind: GdnStateCopyKind) {
        let bytes = self.report.shape.allocated_state_bytes;
        let bucket = self.bucket_mut(scope);
        bucket.read_bytes = bucket.read_bytes.saturating_add(bytes);
        bucket.write_bytes = bucket.write_bytes.saturating_add(bytes);
        match kind {
            GdnStateCopyKind::Snapshot => {
                bucket.snapshot_count = bucket.snapshot_count.saturating_add(1)
            }
            GdnStateCopyKind::Restore => {
                bucket.restore_count = bucket.restore_count.saturating_add(1)
            }
        }
    }
}

#[cfg(all(test, feature = "gdn-state-counters"))]
mod gdn_state_traffic_tests;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use inner::{
    ChatCompletionOutput, HiddenReadbackPathProofSnapshot, LayerImportanceScore, LayerPruningPlan,
    LogitReadbackPathProofSnapshot, LoraLayerData, MetalQwen35State, MoeRoutingTraceRecord,
    PathProofSnapshot, arm_moe_routing_trace, blend_lora_layer_data, dump_moe_routing_trace_jsonl,
    take_moe_routing_trace,
};

/// Existing prefix-cache result used by the version-coupled serving operations.
#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "serving-support"
))]
#[doc(hidden)]
pub use inner::CachedGenerateOutput;

/// The ordinary-generation route planner and samplers, shared with
/// `crate::decoder::qwen_metal` so the decoder session selects exactly as
/// `generate` and `generate_streaming_with_cancel` do.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use inner::{
    GpuTopkRoute, SamplingRouteEnvironment, SamplingRoutePlan, SpeculativeMetrics,
    apply_sampling_route_plan, plan_sampling_route, sample_decode_traced, sample_from_candidates,
    sample_token,
};

#[cfg(all(
    target_os = "macos",
    feature = "metal-gpu",
    feature = "bench-internals"
))]
pub use inner::bench_support;

// Stub for non-macOS or non-metal builds.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub use lattice_lora::LoraLayerData;

/// Stub: blend_lora_layer_data is only meaningful on macOS + metal-gpu.
///
/// Always returns `Err` on other platforms so callers fail at runtime rather
/// than silently producing wrong results.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub fn blend_lora_layer_data(
    _inputs: &[(&[LoraLayerData], f32)],
) -> Result<Vec<LoraLayerData>, lattice_inference_core::InferenceError> {
    Err(lattice_inference_core::InferenceError::Inference(
        "blend_lora_layer_data requires macOS + metal-gpu feature".into(),
    ))
}

/// **Unstable**: Metal Qwen3.5 forward state; stub on non-macOS; full impl behind metal-gpu feature.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub struct MetalQwen35State;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
impl MetalQwen35State {
    /// **Unstable**: construct Metal Qwen3.5 state; always fails without metal-gpu feature.
    pub fn new(
        _weights: &crate::model::qwen35::ModelWeights,
        _cfg: &lattice_arch::qwen35_config::Qwen35Config,
        _max_cache_len: usize,
    ) -> Result<Self, String> {
        Err("Metal GPU not available (requires macOS + metal-gpu feature)".into())
    }

    /// **Unstable**: load Q4/F16 directly stub; always fails without metal-gpu feature.
    pub fn from_q4_dir(
        _q4_dir: &std::path::Path,
        _tokenizer_path: &std::path::Path,
        _cfg: &lattice_arch::qwen35_config::Qwen35Config,
        _max_cache_len: usize,
    ) -> Result<Self, String> {
        Err("Metal GPU not available (requires macOS + metal-gpu feature)".into())
    }

    /// Stub: always returns false without metal-gpu feature.
    pub fn has_mtp(&self) -> bool {
        false
    }

    /// Stub: always returns `None` without metal-gpu feature.
    pub fn mtp_hidden_tap(&self) -> Option<MtpHiddenTap> {
        None
    }

    /// **Unstable**: Metal single-token forward step stub.
    pub fn forward_step(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        match self.try_forward_step(token_id, position) {
            Ok(logits) => logits,
            Err(error) => panic!("{error}"),
        }
    }

    /// **Unstable**: fallible Metal single-token forward step stub.
    pub fn try_forward_step(
        &mut self,
        _token_id: u32,
        _position: usize,
    ) -> Result<Vec<f32>, lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: fallible Metal hidden-returning step stub.
    pub fn forward_step_with_hidden(
        &mut self,
        _token_id: u32,
        _position: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal prefill stub; panics without macOS + metal-gpu.
    ///
    /// Use [`Self::try_forward_prefill`] for a typed capability error.
    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
        match self.try_forward_prefill(token_ids) {
            Ok(logits) => logits,
            Err(error) => panic!("{error}"),
        }
    }

    /// **Unstable**: fallible Metal prefill stub.
    ///
    /// # Errors
    ///
    /// Always returns [`lattice_inference_core::InferenceError::Inference`] without
    /// macOS + `metal-gpu` support.
    pub fn try_forward_prefill(
        &mut self,
        _token_ids: &[u32],
    ) -> Result<Vec<f32>, lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: fallible Metal hidden-returning prefill stub.
    pub fn forward_prefill_with_hidden(
        &mut self,
        _token_ids: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>), lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal option-letter scoring stub.
    ///
    /// # Errors
    ///
    /// Always returns [`lattice_inference_core::InferenceError::Inference`] without
    /// macOS + `metal-gpu` support.
    pub fn score_option_letters(
        &mut self,
        _prompt_ids: &[u32],
        _letter_ids: &[u32],
    ) -> Result<lattice_compute::option_scoring::OptionScores, lattice_inference_core::InferenceError>
    {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal n-gram speculative generation stub.
    ///
    /// Always returns a typed capability error without macOS + `metal-gpu`.
    pub fn generate_with_speculation(
        &mut self,
        _prompt_tokens: &[u32],
        _max_new_tokens: usize,
        _eos_token: u32,
        _max_ngram: usize,
        _max_draft: usize,
    ) -> Result<Vec<u32>, lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal generate stub; always errors without metal-gpu feature.
    ///
    /// #856 follow-up: this used to
    /// return an unconditional `Ok(GenerateOutput { .. })` with empty
    /// text/tokens for *every* prompt, including non-empty ones. Because
    /// `MetalQwen35State` is a public unit struct on this cfg (directly
    /// constructible without going through the fallible
    /// `new`/`from_q4_dir`), that made this the one reachable fail-open
    /// hole in the #856 unified empty-prompt contract: an external caller
    /// building without the `metal-gpu` feature could construct the stub
    /// directly and get a *successful* empty completion back from an
    /// empty (or any) prompt.
    ///
    /// Changing this method's return type
    /// from `-> GenerateOutput` to `-> Result<GenerateOutput,
    /// InferenceError>` directly is itself a semver-breaking signature
    /// change -- this stub is what every **default** build gets
    /// (`metal-gpu` is not a default feature), so any existing downstream
    /// default-build caller consuming a bare `GenerateOutput` would stop
    /// compiling, and `cargo-semver-checks`' inherent-method lint would
    /// not catch a non-unit-to-non-unit return type change. As an interim
    /// step (v0.6.1), the signature was kept as `-> GenerateOutput` and
    /// the stub panicked instead: fail-loud rather than fail-open, but
    /// still a rougher contract than a typed `Err`.
    ///
    /// #917 (this signature, targeting the 0.7.0 semver-major release
    /// alongside the other already-known 0.7.0 breaking changes): the
    /// return type is now `Result<GenerateOutput, InferenceError>`,
    /// matching the real metal-gpu-enabled implementation's signature
    /// exactly. This is a deliberate breaking change gated to 0.7.0; it
    /// replaces the v0.6.1 panic with a typed capability error, following
    /// the same "fail with `Err(InferenceError::Inference(..))`" pattern
    /// used by every other stub method on this type
    /// (`blend_lora_layer_data`, `load_lora_adapter`,
    /// `compute_token_nlls`, `compute_perplexity`). See
    /// <https://github.com/ohdearquant/lattice/issues/917> (sibling to
    /// #916's `generate_with_speculation` tracking).
    pub fn generate(
        &mut self,
        _prompt: &str,
        _tokenizer: &lattice_tokenizer::bpe::BpeTokenizer,
        _cfg: &crate::generation::GenerateConfig,
    ) -> Result<crate::generation::GenerateOutput, lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: load LoRA adapter stub; always errors without metal-gpu feature.
    pub fn load_lora_adapter(
        &mut self,
        _layers: Vec<LoraLayerData>,
        _scale: f32,
        _quarot_seed: Option<u64>,
    ) -> Result<(), lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: reset recurrent KV state stub.
    pub fn reset_state(&mut self) {}

    /// **Unstable**: max context stub; always 0 without metal-gpu feature.
    pub fn max_context(&self) -> usize {
        0
    }

    /// **Unstable**: hidden size stub; always 0 without metal-gpu feature.
    pub fn hidden_size(&self) -> usize {
        0
    }

    /// **Stable**: pooled-embedding stub; always errors without metal-gpu feature.
    pub fn embed_tokens(
        &mut self,
        _tokens: &[u32],
        _pooling: crate::model::qwen35::HiddenPooling,
    ) -> Result<Vec<f32>, lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal Q4 per-token NLL stub; always errors without metal-gpu feature.
    pub fn compute_token_nlls(
        &mut self,
        _tokens: &[u32],
    ) -> Result<Vec<f32>, lattice_inference_core::InferenceError> {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal Q4 perplexity stub; always errors without metal-gpu feature.
    pub fn compute_perplexity(
        &mut self,
        _tokens: &[u32],
        _cfg: &crate::model::qwen35::PerplexityConfig,
    ) -> Result<crate::model::qwen35::PerplexityReport, lattice_inference_core::InferenceError>
    {
        Err(lattice_inference_core::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }
}

/// Regression coverage for the non-metal-gpu `MetalQwen35State` stub (#856
/// follow-up; #917 follow-up). This runs in the
/// DEFAULT build (no `metal-gpu` feature, hence no `#[cfg(metal-gpu)]` on
/// the test itself) -- exactly the build a downstream crate gets by
/// default, and exactly the build where `MetalQwen35State` is directly
/// constructible as a public unit struct without going through the
/// fallible `new`/`from_q4_dir`.
///
/// #917: the stub's `generate` now returns
/// `Result<GenerateOutput, InferenceError>`, matching the metal-gpu-enabled
/// implementation's signature. These tests assert the typed `Err`, not a
/// panic (the v0.6.1 interim behavior).
///
/// Mutation sensitivity: reverting the stub's `generate` to the v0.6.1
/// panic (or to the original unconditional `Ok(GenerateOutput { .. })`)
/// makes both tests below fail -- neither a panic nor a bare `Ok` matches
/// the `Err(InferenceError::Inference(..))` assertion.
#[cfg(all(test, not(all(target_os = "macos", feature = "metal-gpu"))))]
mod non_metal_stub_tests {
    use super::MetalQwen35State;
    use crate::generation::GenerateConfig;
    use lattice_inference_core::InferenceError;
    use lattice_tokenizer::bpe::BpeTokenizer;
    use std::collections::HashMap;

    fn tiny_tokenizer() -> BpeTokenizer {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
            vocab.insert((*c).to_string(), i as u32);
        }
        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
        ];
        BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap()
    }

    #[test]
    fn non_metal_stub_try_forward_prefill_returns_capability_error() {
        let mut state = MetalQwen35State;
        let result = state.try_forward_prefill(&[1, 2]);
        match result {
            Err(InferenceError::Inference(message)) => {
                assert!(message.contains("metal-gpu"), "{message}");
            }
            other => panic!("expected Metal capability error, got {other:?}"),
        }
    }

    #[test]
    fn non_metal_stub_score_option_letters_returns_capability_error() {
        let mut state = MetalQwen35State;
        match state.score_option_letters(&[1, 2], &[3, 4]) {
            Err(InferenceError::Inference(message)) => {
                assert!(message.contains("metal-gpu"), "{message}");
            }
            other => panic!("expected Metal capability error, got {other:?}"),
        }
    }

    #[test]
    fn non_metal_stub_generate_rejects_empty_prompt() {
        let tokenizer = tiny_tokenizer();
        let gen_cfg = GenerateConfig::default();
        let mut state = MetalQwen35State;

        // Must return a typed Err (fail loud, not fail open) rather than a
        // successful-looking empty GenerateOutput for an empty prompt.
        let result = state.generate("", &tokenizer, &gen_cfg);
        match result {
            Err(InferenceError::Inference(msg)) => {
                assert!(
                    msg.contains("metal-gpu"),
                    "expected capability error mentioning metal-gpu, got: {msg}"
                );
            }
            other => panic!("expected Err(InferenceError::Inference(..)), got: {other:?}"),
        }
    }

    /// The stub errors for every prompt, not only the empty one --
    /// directly-constructing it (bypassing the fallible `new`/`from_q4_dir`)
    /// must never yield a successful (if vacuous) completion.
    #[test]
    fn non_metal_stub_generate_rejects_nonempty_prompt() {
        let tokenizer = tiny_tokenizer();
        let gen_cfg = GenerateConfig::default();
        let mut state = MetalQwen35State;

        let result = state.generate("hello", &tokenizer, &gen_cfg);
        match result {
            Err(InferenceError::Inference(msg)) => {
                assert!(
                    msg.contains("metal-gpu"),
                    "expected capability error mentioning metal-gpu, got: {msg}"
                );
            }
            other => panic!("expected Err(InferenceError::Inference(..)), got: {other:?}"),
        }
    }

    #[test]
    fn non_metal_stub_speculative_generation_returns_capability_error() {
        let mut state = MetalQwen35State;

        let result = state.generate_with_speculation(&[0], 1, u32::MAX, 5, 4);
        assert!(matches!(
            result,
            Err(InferenceError::Inference(ref message))
                if message
                    == "Metal GPU not available (requires macOS + metal-gpu feature)"
        ));
    }
}

// ---------------------------------------------------------------------------
// Differential unit tests for `mtp_greedy_round`.
//
// These tests run on every platform without Metal GPU hardware (no cfg gate).
// They compare the output of a tiny plain-greedy oracle against the sequence
// produced by driving `mtp_greedy_round` with the same synthetic target
// logits, across hand-constructed cases and a seeded randomised sweep.
//
// EOS-faithful contract (all three terminal cases A/B/C must match oracle):
//   A: prefill argmax is EOS → oracle emits [EOS]; simulation must also emit [EOS].
//   B: accept + bonus is stop → oracle emits [..., pending, draft, bonus];
//      simulation must emit the same (bonus emitted as trailing stop token).
//   C: reject + replacement is stop → oracle emits [..., pending, replacement];
//      simulation must emit the same.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod mtp_load_warning_tests;

#[cfg(test)]
mod mtp_greedy_round_tests;

// ---------------------------------------------------------------------------
// Differential unit tests for `generate_greedy_self_spec`'s stop-token contract.
//
// Strategy: mirror `mtp_greedy_round_tests` — write an independent plain-greedy
// oracle (check is_stop, then push, bounded by max_new_tokens) and a pure
// simulation of the self-spec round logic. The simulation encodes exactly the
// post-fix decision logic for each termination path WITHOUT re-deriving it from
// the buggy pre-fix code, so it acts as an independent reference.
//
// Stop-token contract (#613, matches canonical GenerateOutput docs): the
// terminating token is EXCLUDED from token_ids/text at every termination path:
//   A:  prefill argmax is stop → emit [], generated_tokens = 0.
//   R:  rejection: target replacement is stop → emit [pending] (replacement excluded).
//   FA: full-accept: next_pending is stop → emit [..] without next_pending.
//   FB: fallback (no-checkpoint, checkpoint-failure, verify-failure): target's next
//       is stop → emit up to (not including) that stop, budget-clipped.
//
// These tests run on every platform (no cfg gate — no GPU required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod self_spec_eos_tests;

/// Shader source shared with the repository's measured matrix-shape probe.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub const Q4_TILED_GEMM_SHADER_SOURCE: &str = inner::MSL_Q4_TILED_SOURCE;
