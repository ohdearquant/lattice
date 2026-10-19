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
//! q/k/v/o_proj) and embed_tokens. Buffers kept as f32: norm weights, bias/log scalars,
//! conv1d weights (small, read by CPU), RoPE tables, activation scratch buffers.

#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
mod mtp_weights;

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
    mod dispatch;
    mod gdn_state;
    mod layers;

    use super::GdnStateTrafficScope;
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
    #[cfg(feature = "gdn-state-counters")]
    use super::{
        GdnStateCopyKind, GdnStateTrafficCounters, GdnStateTrafficReport, GdnStateTrafficShape,
    };
    use crate::attention::gdn::GatedDeltaNetState;
    use crate::attention::gdn_fused::GatedDeltaNetFusedScratch;
    use crate::generation::{GenerateConfig, GenerateOutput};
    use crate::model::qwen35::stop_strings::StopStringMatcher;
    use crate::model::qwen35::{
        AttentionWeights, GenerationEntryContract, GenerationPlan, GenerationPreparation,
        ModelWeights, prepare_generation,
    };
    use crate::model::qwen35_config::Qwen35Config;
    use crate::stop_reason::StopReason;
    use crate::tokenizer::bpe::BpeTokenizer;
    use crate::vision::multimodal::Qwen35VisionRequest;
    use crate::weights::q4_weights::quantize_row_q4_0;
    use lattice_tokenizer::__internal::{
        incremental_detokenizer_finish, incremental_detokenizer_new, incremental_detokenizer_push,
    };
    use metal::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    // ---------------------------------------------------------------------------
    // MSL Compute Shaders
    // ---------------------------------------------------------------------------

    // #854: the shared RMS-norm reduction helper is assembled as a prefix
    // fragment ahead of the kernel bodies that call it (Metal requires the
    // callee be visible textually before its call sites in the same
    // translation unit).
    const MSL_SOURCE: &str = concat!(
        include_str!("shaders/rms_reduce.metal"),
        include_str!("shaders/qwen35.metal")
    );

    const MSL_Q4_TILED_SOURCE: &str = include_str!("shaders/gemm_q4_tiled.metal");

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

    // ---------------------------------------------------------------------------
    // GPU Buffer Structures
    // ---------------------------------------------------------------------------

    /// A quantized weight buffer, paired with the byte offset within that buffer where
    /// the actual Q4 payload starts.
    ///
    /// For whole-file mmap buffers (H1 optimisation): `buffer` spans the entire
    /// mmapped file and `payload_offset` skips the file header.  For regular
    /// `new_buffer_with_data` buffers (Q8_0 path and concatenated Q4 tensors):
    /// `payload_offset` is 0.  Dispatch helpers always bind via
    /// `enc.set_buffer(slot, Some(&qw.buffer), qw.payload_offset)` so both cases
    /// are handled uniformly.
    pub(crate) struct Q4WeightBuf {
        buffer: Buffer,
        /// Byte offset of Q4 payload within `buffer`.
        payload_offset: u64,
        // Keeps the mmap alive as long as the Metal buffer needs the memory.
        _mmap: Option<memmap2::Mmap>,
    }

    impl Q4WeightBuf {
        /// Wrap a normally-allocated Metal buffer (payload at offset 0).
        fn from_buffer(buffer: Buffer) -> Self {
            Q4WeightBuf {
                buffer,
                payload_offset: 0,
                _mmap: None,
            }
        }

        /// Wrap a no-copy Metal buffer backed by `mmap` with a non-zero payload offset.
        ///
        /// The GPU is the only thing that will ever read these bytes, so the
        /// caller must present the [`Q4BlocksChecked`] witness that
        /// `open_and_mmap_q4_file` issues for an eagerly checked mapping.
        /// A caller that deferred the block check has no witness to pass and
        /// cannot reach this constructor without visibly unwrapping a `None`.
        fn from_mmap(
            buffer: Buffer,
            payload_offset: u64,
            mmap: memmap2::Mmap,
            _blocks_checked: crate::weights::q4_weights::Q4BlocksChecked,
        ) -> Self {
            Q4WeightBuf {
                buffer,
                payload_offset,
                _mmap: Some(mmap),
            }
        }
    }

    /// Open `path`, mmap the whole file, and register it as a Metal no-copy
    /// `StorageModeShared` buffer.  Stores the mmap owner inside the returned
    /// [`Q4WeightBuf`] so the pages remain valid for the lifetime of the buffer.
    ///
    /// This is the one Q4 load path whose bytes are never decoded on the CPU:
    /// the packed nibbles stay resident in the no-copy GPU buffer and the
    /// per-block scale/bias is dequantized by the Metal kernel at dispatch
    /// time. There is therefore no traversal here to fold a scale/bias check
    /// into, and a file with a correct header but a NaN or infinite scale
    /// would otherwise load cleanly and propagate the NaN into the logits at
    /// the first GEMV. The load requests [`Q4BlockCheck::Now`] so the check
    /// happens before the buffer exists.
    ///
    /// That costs one sequential pass over the payload, faulting in pages
    /// this no-copy mapping was otherwise designed not to touch — a real cost
    /// paid once per weight at load, deliberately, because the alternative is
    /// admitting non-finite weights to the GPU.
    ///
    /// # Safety invariant
    /// The mmap is read-only (`MAP_PRIVATE`) and the model files must not be
    /// modified while the process is running.
    #[cfg(feature = "metal-gpu")]
    fn mmap_q4_weight(
        device: &Device,
        path: &std::path::Path,
        label: &str,
        expected_shape: &[usize],
    ) -> Result<Q4WeightBuf, String> {
        use crate::weights::q4_weights::{Q4BlockCheck, open_and_mmap_q4_file};

        let tensor_name = path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("native Q4 tensor");
        let (header, mmap, blocks_checked) = open_and_mmap_q4_file(
            path,
            Some(expected_shape),
            Q4BlockCheck::Now { tensor_name },
        )?;
        let blocks_checked = blocks_checked.ok_or_else(|| {
            format!(
                "{}: block metadata was not checked before no-copy GPU publication",
                path.display()
            )
        })?;

        // The mmap pointer is page-aligned (guaranteed by the OS).  We create the Metal
        // buffer over the *whole* file so the pointer alignment requirement of
        // new_buffer_with_bytes_no_copy is satisfied.  The Q4 payload is then addressed
        // via enc.set_buffer(slot, ..., header.payload_offset) at dispatch time.
        let buf = device.new_buffer_with_bytes_no_copy(
            mmap.as_ptr().cast(),
            mmap.len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        buf.set_label(label);

        Ok(Q4WeightBuf::from_mmap(
            buf,
            header.payload_offset,
            mmap,
            blocks_checked,
        ))
    }

    /// A Q3-quantized weight buffer (ADR-072 P1, #420 Stage 2), paired with the
    /// byte offset within `buffer` where the Q3 block payload starts.
    ///
    /// Restricted to MLP gate/up/down projections — see
    /// [`crate::weights::q3_weights::validate_q3_mlp_role`]. Mirrors
    /// [`Q4WeightBuf`]'s buffer/offset/mmap-lifetime structure exactly; kept as
    /// a separate type (not a third `Q4WeightBuf` variant) so the MLP-only
    /// role restriction is enforced at construction, not by convention.
    pub(crate) struct Q3WeightBuf {
        buffer: Buffer,
        /// Byte offset of the Q3 payload within `buffer`.
        payload_offset: u64,
        // Keeps the mmap alive as long as the Metal buffer needs the memory.
        _mmap: Option<memmap2::Mmap>,
    }

    impl Q3WeightBuf {
        /// Wrap a normally-allocated Metal buffer (payload at offset 0).
        ///
        /// Mirrors `Q4WeightBuf::from_buffer` for API-shape parity; unused
        /// until a non-mmap Q3 construction path (e.g. concatenated MoE-style
        /// upload) exists.
        #[allow(dead_code)]
        fn from_buffer(buffer: Buffer) -> Self {
            Q3WeightBuf {
                buffer,
                payload_offset: 0,
                _mmap: None,
            }
        }

        /// Wrap a no-copy Metal buffer backed by `mmap` with a non-zero payload offset.
        ///
        /// Exercised today only by [`mmap_q3_weight`] and its tests; live
        /// checkpoint loading does not yet route MLP tensors through it.
        fn from_mmap(buffer: Buffer, payload_offset: u64, mmap: memmap2::Mmap) -> Self {
            Q3WeightBuf {
                buffer,
                payload_offset,
                _mmap: Some(mmap),
            }
        }
    }

    /// Open `path`, mmap the whole file, and register it as a Metal no-copy
    /// `StorageModeShared` buffer holding a Q3 tensor — mirrors [`mmap_q4_weight`].
    ///
    /// Fails closed on two independent conditions before the mmap ever reaches
    /// the GPU: `tensor_name` must name an MLP gate/up/down projection (see
    /// [`crate::weights::q3_weights::validate_q3_mlp_role`]), and the file's
    /// declared header must not claim a payload larger than the file actually
    /// holds (see `validate_q3_header_payload_bounds`) — the same truncation
    /// hazard `mmap_q4_weight` guards against, since this path never reads the
    /// payload bytes itself.
    ///
    /// # Safety invariant
    /// The mmap is read-only (`MAP_PRIVATE`) and the model files must not be
    /// modified while the process is running. [`crate::weights::mmap_trust::open_trusted_mmap_file`]
    /// enforces the write-boundary half of that invariant (group/other-writable,
    /// foreign ownership, a writable ancestor directory, or a permission-granting
    /// extended ACL) the same way [`mmap_q4_weight`] does; same-uid mutation
    /// through a second fd remains the documented residual, as for every mmap
    /// site in this crate.
    ///
    /// Not yet called from live checkpoint loading (proven end-to-end by this
    /// module's `mmap_q3_weight_*` tests instead).
    #[cfg(feature = "metal-gpu")]
    #[allow(dead_code)]
    fn mmap_q3_weight(
        device: &Device,
        path: &std::path::Path,
        tensor_name: &str,
        label: &str,
    ) -> Result<Q3WeightBuf, String> {
        use crate::weights::q3_weights::validate_q3_mlp_role;

        validate_q3_mlp_role(tensor_name).map_err(|e| e.to_string())?;

        let mut handle = crate::weights::mmap_trust::open_trusted_mmap_file(path)?;
        let header = crate::weights::q3_weights::read_q3_header(&mut handle)
            .map_err(|e| format!("failed to parse Q3 header {}: {e}", path.display()))?;
        let file_len = handle
            .len()
            .map_err(|e| format!("failed to stat {}: {e}", path.display()))?;
        crate::weights::q3_weights::validate_q3_header_payload_bounds(&header, file_len, path)
            .map_err(|e| format!("failed to validate Q3 payload {}: {e}", path.display()))?;
        let mmap = crate::weights::mmap_trust::map_and_verify_trusted(&handle)?;

        let buf = device.new_buffer_with_bytes_no_copy(
            mmap.as_ptr().cast(),
            mmap.len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        buf.set_label(label);

        Ok(Q3WeightBuf::from_mmap(buf, header.payload_offset, mmap))
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

    // ADR-051 §"MTP tensors still safety-skipped in quantize_quarot": Phase 1 keeps
    // every MTP projection as f16 (unquantized) so the counter-rotation operates on
    // unquantized weights. Buffers hold raw f16 (u16-on-disk converted via
    // `make_buffer_f16`); the GEMV path uses `dispatch_matmul_half`. Phase 2 will
    // rotate and quantize these tensors.
    struct MetalMtpDenseMlpWeights {
        gate_proj: Buffer, // f16, [intermediate, hidden]
        up_proj: Buffer,   // f16, [intermediate, hidden]
        down_proj: Buffer, // f16, [hidden, intermediate]
    }

    struct MetalMtpLayerWeights {
        input_layernorm: Buffer,          // [hidden]
        post_attention_layernorm: Buffer, // [hidden]
        q_proj: Buffer,                   // f16, [2*q_dim, hidden]
        k_proj: Buffer,                   // f16, [kv_dim, hidden]
        v_proj: Buffer,                   // f16, [kv_dim, hidden]
        o_proj: Buffer,                   // f16, [hidden, q_dim]
        q_norm: Buffer,                   // [head_dim]
        k_norm: Buffer,                   // [head_dim]
        mlp: MetalMtpDenseMlpWeights,
    }

    pub(crate) struct MetalMtpWeights {
        fc: Buffer,                    // f16, [hidden, 2*hidden]
        pre_fc_norm_embedding: Buffer, // [hidden]
        pre_fc_norm_hidden: Buffer,    // [hidden]
        layers: Vec<MetalMtpLayerWeights>,
        norm: Buffer, // [hidden]
        /// Which target hidden the head receives; decided once at load.
        hidden_tap: MtpHiddenTap,
    }

    struct MtpQ4LoadResult {
        weights: Option<MetalMtpWeights>,
        first_missing_file: Option<std::path::PathBuf>,
    }

    pub(crate) struct MetalMtpCache {
        k_buf: Buffer,
        v_buf: Buffer,
        seq_len: usize,
        #[allow(dead_code)] // capacity bound stored for future MTP bounds checking
        max_cache_len: usize,
    }

    impl MetalMtpCache {
        fn reset(&mut self) {
            self.seq_len = 0;
        }
    }

    pub(crate) struct MetalMtpActivations {
        fused: Buffer,       // [2*hidden] — normed embedding concat normed hidden
        hidden: Buffer,      // [hidden]
        residual: Buffer,    // [hidden]
        q: Buffer,           // [2*q_dim] interleaved Q+gate from q_proj
        q_separated: Buffer, // [q_dim] Q after scatter
        gate_z: Buffer,      // [q_dim] gate after scatter
        k: Buffer,           // [kv_dim]
        v: Buffer,           // [kv_dim]
        attn_out: Buffer,    // [q_dim.max(hidden)]
        gate: Buffer,        // [intermediate]
        up: Buffer,          // [intermediate]
        ffn_out: Buffer,     // [hidden]
        logits: Buffer,      // [vocab]
    }

    /// Per-session MTP mutable state (cache + activations only; weights live on the engine).
    pub(crate) struct MetalMtpSession {
        pub(crate) cache: MetalMtpCache,
        pub(crate) activations: MetalMtpActivations,
    }

    /// Per-speculation-round GDN state checkpoint pool.
    ///
    /// Stores up to `max_tokens + 1` snapshots (slot 0 = base, slot k = after k tokens).
    pub(crate) struct MetalGdnCheckpointPool {
        max_tokens: usize,
        conv_slots: Vec<Vec<Buffer>>, // [slot][linear_layer]
        s_slots: Vec<Vec<Buffer>>,    // [slot][linear_layer]
        active_base_seq_len: Option<usize>,
        mtp_base_seq_len: Option<usize>,
        /// Batch-verifier repair: (token_id, position) to replay on rejection for GDN correctness.
        batch_repair_token: Option<(u32, usize)>,
    }

    impl MetalGdnCheckpointPool {
        fn new(device: &Device, max_tokens: usize, state: &MetalGdnState) -> Self {
            let num_layers = state.len();
            let geometry = state.geometry();
            debug_assert_eq!(geometry.allocated_layers, num_layers);
            debug_assert!(geometry.active_layers <= geometry.architectural_layers);
            debug_assert_eq!(state.precision(), MetalGdnStatePrecision::F32);
            let slots = max_tokens + 1;
            let conv_slots = (0..slots)
                .map(|slot| {
                    (0..num_layers)
                        .map(|i| {
                            let buf = device.new_buffer(
                                state.layer(i).conv_buffer().length(),
                                MTLResourceOptions::StorageModeShared,
                            );
                            buf.set_label(&format!("gdn_ckpt_conv_s{slot}_l{i}"));
                            buf
                        })
                        .collect()
                })
                .collect();
            let s_slots = (0..slots)
                .map(|slot| {
                    (0..num_layers)
                        .map(|i| {
                            let buf = device.new_buffer(
                                state.layer(i).s_matrix().length(),
                                MTLResourceOptions::StorageModeShared,
                            );
                            buf.set_label(&format!("gdn_ckpt_s_s{slot}_l{i}"));
                            buf
                        })
                        .collect()
                })
                .collect();
            Self {
                max_tokens,
                conv_slots,
                s_slots,
                active_base_seq_len: None,
                mtp_base_seq_len: None,
                batch_repair_token: None,
            }
        }
    }

    struct MetalVerifyOutput {
        logits: Vec<Vec<f32>>, // per-token full-vocab logits
        #[allow(dead_code)]
        // pre-final hidden retained for MTP chaining; not yet consumed by callers
        final_hidden: Vec<f32>, // pre-final hidden of last verified row
        // Raw (pre-final-RMSNorm) hidden state of the FIRST verified row (the
        // pending token's own position), i.e. the state that predicted
        // `tokens[1]`. Two consumers:
        // - On a full MTP accept the second verified row commits as a real
        //   token, and its MTP-cache entry pairs that token's embedding with
        //   this hidden state, matching the pairing `mtp_prefill_append`'s doc
        //   comment establishes (lattice#1396).
        // - On a K=1 draft rejection on the sequential verifier, the caller
        //   restores `self.session.last_pre_final_hidden` to this value rather
        //   than leaving it at whatever `forward_step_inner`'s capture-hidden
        //   side effect last wrote for `tokens[1]` -- the *draft* token's
        //   hidden, from a verify step whose KV/GDN mutations get rolled back
        //   but whose `last_pre_final_hidden` write does not. The batch-GEMM
        //   verifier's own repair-replay already restores it, so that caller
        //   does not read this field on reject.
        // The batch-GEMM path leaves it empty when the session carries no MTP
        // head.
        first_pre_final_hidden: Vec<f32>,
        // Raw (pre-final-RMSNorm) hidden state of every verified row, in token
        // order. Filled only by `verify_tokens_batched_impl(.., keep_all_rows =
        // true)` and empty everywhere else, so the greedy MTP loop's verify
        // pays no extra clone. The consumer is
        // `MtpTargetVerifier::verify_tokens_with_hidden`.
        pre_final_hidden_rows: Vec<Vec<f32>>,
    }

    struct MetalStepOutput {
        logits: Vec<f32>,
        pre_final_hidden: Vec<f32>,
        // Kept alive by binding the returned struct (not discarding it),
        // rather than dropping at the end of `forward_step_inner_impl`'s
        // body — the zero-copy greedy path needs `decode.step` to span
        // through its post-return logits scan (see
        // `forward_step_greedy_argmax`); every other caller either drops
        // this immediately (`.logits`, same timing as before) or holds a
        // `Scope::NotDecode` no-op guard, which is a no-op to drop late.
        _signpost_step: crate::forward::signpost::Interval,
    }

    struct MetalMtpForwardOutput {
        token_id: u32,
        logits: Vec<f32>,
    }

    /// Decoding metrics emitted when LATTICE_MTP=1.
    #[derive(Default)]
    pub(crate) struct MetalMtpDecodeMetrics {
        rounds: usize,
        mtp_forwards: usize,
        verify_calls: usize,
        accepted_extra_tokens: usize,
        fallback_tokens: usize,
        mtp_ms: f64,
        verify_ms: f64,
        rollback_ms: f64,
    }

    /// Decoding metrics emitted when LATTICE_SELF_SPEC_VERBOSE=1.
    #[derive(Default)]
    pub(crate) struct SelfSpecMetrics {
        rounds: usize,
        draft_forwards: usize,
        verify_calls: usize,
        accepted_extra_tokens: usize,
        fallback_tokens: usize,
        draft_ms: f64,
        verify_ms: f64,
        rollback_ms: f64,
    }

    /// The speculative decode route one request ran on.
    ///
    /// The default MTP verifier and the GDN-first self-speculative route are distinct
    /// mechanisms that both apply the shared decode policy to the tokens a verification
    /// pass committed. The batch-GEMM verifier is a legacy experimental route that does
    /// not: it is named here so a request on it can never be mistaken for shared-route
    /// coverage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum SpeculativeRoute {
        /// MTP draft with the sequential verifier, under the shared decode policy.
        MtpShared,
        /// GDN-first self-speculation, under the shared decode policy.
        SelfSpecShared,
        /// MTP draft with the batch-GEMM verifier (`LATTICE_MTP_BATCH`, no adapter): the
        /// legacy loop, excluded from the shared decode policy.
        MtpBatchGemmLegacy,
    }

    impl SpeculativeRoute {
        /// Whether this route applies the shared decode policy to committed tokens.
        pub(crate) fn uses_shared_policy(self) -> bool {
            !matches!(self, Self::MtpBatchGemmLegacy)
        }

        /// The stable name printed by the verbose route marker.
        pub(crate) fn label(self) -> &'static str {
            match self {
                Self::MtpShared => "shared-mtp",
                Self::SelfSpecShared => "shared-self-spec",
                Self::MtpBatchGemmLegacy => "legacy-batch-gemm-excluded",
            }
        }
    }

    /// What a request left behind about its speculative route. `trace` counts the rounds
    /// the shared driver ran and the committed tokens it offered to the policy; both are
    /// zero on the legacy batch-GEMM route, which has no shared driver.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct SpeculativeRouteRecord {
        pub(crate) route: SpeculativeRoute,
        pub(crate) trace: crate::decoder::SpeculativeTrace,
    }

    /// The per-request counters of whichever speculative route is running.
    pub(crate) enum SpeculativeMetrics {
        Mtp(MetalMtpDecodeMetrics),
        SelfSpec(SelfSpecMetrics),
    }

    impl SpeculativeMetrics {
        pub(crate) fn mtp() -> Self {
            Self::Mtp(MetalMtpDecodeMetrics::default())
        }

        pub(crate) fn self_spec() -> Self {
            Self::SelfSpec(SelfSpecMetrics::default())
        }

        pub(crate) fn route(&self) -> SpeculativeRoute {
            match self {
                Self::Mtp(_) => SpeculativeRoute::MtpShared,
                Self::SelfSpec(_) => SpeculativeRoute::SelfSpecShared,
            }
        }
    }

    /// Reusable activation buffers on GPU (pre-allocated for single-token decode).
    pub(crate) struct MetalQwen35Activations {
        hidden: Buffer,      // [hidden_size]
        residual: Buffer,    // [hidden_size]
        attn_out: Buffer,    // [max(q_dim, hidden_size)] for attention output
        q: Buffer,           // [2 * q_dim] for Q+gate interleaved from q_proj
        q_separated: Buffer, // [q_dim] Q after scatter
        gate_z: Buffer,      // [q_dim] gate after scatter
        k: Buffer,           // [kv_dim]
        v: Buffer,           // [kv_dim]
        gate: Buffer, // [2 * intermediate_size] for Dense decode gate||up; batch uses first bp rows
        up: Buffer,   // [intermediate_size] kept for batch separate-buffer paths
        ffn_out: Buffer, // [hidden_size]
        // GDN projection buffers
        gdn_qkv: Buffer,         // [qkv_dim]  — used by batch/prefill and CPU paths
        gdn_z: Buffer,           // [output_dim] — used by batch/prefill and CPU paths
        gdn_qkvz: Buffer,        // [qkv_dim+output_dim] — decode-only fused projection scratch
        gdn_key_scratch: Buffer, // [num_key_heads*(2*key_dim+1) + num_value_heads*2] — H3 split key/value scratch
        gdn_raw_out: Buffer,     // [output_dim] — H1 raw recurrence output before norm+SiLU
        gdn_chunk: GdnChunkScratch, // chunked-scan scratch (default path; serial if LATTICE_GDN_CHUNKED=0)
        logits: Buffer,             // [vocab_size]
        // Top-k compact readback buffers (zero-initialized; sized for MAX_TOP_K=256).
        topk_scratch_a: Buffer, // [first_pass_groups * MAX_TOP_K] TopKCandidate pairs
        topk_scratch_b: Buffer, // same — ping-pong merge target
        // Flash decode partitioned path (H3): [max_partitions * num_q_heads * (head_dim+2)] f32
        attn_partials: Buffer,
        // MTP: pre-final hidden state (before final RMSNorm) for the last processed token.
        // The draft head applies the final norm to it on the CPU on unrotated
        // checkpoints (`MtpHiddenTap`), so this buffer stays pre-norm.
        pre_final_hidden: Buffer, // [hidden_size]
        // Pooled-embedding capture: the last processed token's hidden state
        // AFTER the final RMSNorm. Deliberately a separate buffer from
        // `pre_final_hidden` above rather than a reuse: the two hold the same
        // token at different points in the network, they differ by the norm's
        // rescale, and a name that reads plausibly for both is how a caller
        // ends up with numbers from the wrong layer. See
        // `encode_capture_final_hidden`.
        final_hidden: Buffer, // [hidden_size]
        // MTP: logits for all K verified tokens (K <= MTP_VERIFY_MAX_TOKENS).
        verify_logits: Buffer, // [MTP_VERIFY_MAX_TOKENS * vocab_size]
    }

    /// KV cache for full-attention layers stored in Metal unified memory.
    pub(crate) struct MetalKvCache {
        /// k_cache[full_layer_idx]: flat [max_cache_len * kv_dim]
        k_bufs: Vec<Buffer>,
        /// v_cache[full_layer_idx]: flat [max_cache_len * kv_dim]
        v_bufs: Vec<Buffer>,
        /// Current number of tokens cached.
        seq_len: usize,
        kv_dim: usize,
        max_cache_len: usize,
    }

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

    /// Vocab rows owned by one Stage-1 lm_head block-top-k threadgroup.
    /// Fixed at 256 because the kernel reuses `bitonic_sort_desc_1024`, whose
    /// inner loop stride is hardcoded to a 256-thread launch.
    pub(crate) const LM_HEAD_ROWS_PER_TG: u32 = 256;
    /// LOCAL_K values with precompiled Stage-1 pipeline variants. Index 0 is
    /// the greedy (argmax) variant; the rest back `GpuTopkRoute::BlockTopK`.
    pub(crate) const LM_HEAD_LOCAL_KS: [u32; 5] = [1, 8, 16, 40, 64];

    fn lm_head_local_k_slot(k: u32) -> Option<usize> {
        LM_HEAD_LOCAL_KS
            .iter()
            .position(|&candidate| candidate == k)
    }

    // -----------------------------------------------------------------------
    // GPU Top-K routing
    // -----------------------------------------------------------------------

    /// Which GPU kernel path handles top-k for the current generate call.
    ///
    /// Computed once by `choose_gpu_topk_route`; stored in `compact_route` so
    /// the dispatch path doesn't re-evaluate env vars per token.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum GpuTopkRoute {
        /// CPU fallback — no GPU top-k dispatch; `compact_topk` stays 0.
        CpuFallback,
        /// k=1 two-stage simdgroup argmax — reserved for a future faster kernel.
        /// R1 bench (2026-04-26, M2 Max): CPU NEON 89 µs vs GPU 240 µs (2.70×); dead path.
        #[allow(dead_code)]
        Argmax,
        /// k=2..=64 simdgroup selection — reserved for a future non-bitonic kernel.
        /// Bench prototype failed gate by 2.4-2.8×; not reachable from production.
        #[allow(dead_code)]
        Select64,
        /// k=65..=256 experimental — reserved for a future non-bitonic kernel.
        #[allow(dead_code)]
        Select256Experimental,
        /// k=50 hierarchical SIMD-group tournament (two-stage, no bitonic sort).
        /// Requires LATTICE_COMPACT_TOPK and LATTICE_COMPACT_TOPK_SELECT env vars.
        HierarchicalK50,
        /// Issue #171: Stage-1 fused GEMV + block-local argmax, Stage-2 global
        /// reduce via the existing `argmax_merge` kernel. Greedy (k<=1) only.
        BlockArgmax,
        /// Issue #171: Stage-1 fused GEMV + block-local top-k, Stage-2 global
        /// reduce via the existing `topk_merge_pass` kernel. `local_k` must be
        /// one of `LM_HEAD_LOCAL_KS` (a precompiled Stage-1 pipeline variant).
        BlockTopK { local_k: u32 },
    }

    /// Decide which GPU top-k route to take for a given `top_k` and env flags.
    ///
    /// `compact_env`: `LATTICE_COMPACT_TOPK` is enabled (by value, not presence — see
    /// `crate::env_switch_enabled`).
    /// `selection_env`: `LATTICE_COMPACT_TOPK_SELECT` is enabled (unlocks k>1 routes).
    fn choose_gpu_topk_route(top_k: usize, compact_env: bool, selection_env: bool) -> GpuTopkRoute {
        // R1 benchmark (2026-04-26, M2 Max, vocab=248,320):
        //   CPU NEON argmax k=1 = 89 µs; GPU argmax k=1 = 240 µs (2.70×, t=137, p<<0.0001).
        //   GPU k>1 (bitonic/select64) = 811–940 µs (2.5-3.0× vs CPU 314–366 µs).
        //   Hierarchical k=50 tournament under evaluation (R2, gated by both env flags).
        if compact_env && selection_env && top_k == 50 {
            return GpuTopkRoute::HierarchicalK50;
        }
        GpuTopkRoute::CpuFallback
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct SamplingRouteEnvironment {
        pub(crate) compact: bool,
        pub(crate) selection: bool,
        pub(crate) approximate_top_p: bool,
    }

    impl SamplingRouteEnvironment {
        pub(crate) fn current() -> Self {
            Self {
                compact: crate::env_switch_enabled("LATTICE_COMPACT_TOPK"),
                selection: crate::env_switch_enabled("LATTICE_COMPACT_TOPK_SELECT"),
                approximate_top_p: crate::env_switch_enabled("LATTICE_COMPACT_TOPP_APPROX"),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct SamplingRoutePlan {
        pub(crate) use_compact: bool,
        pub(crate) compact_route: GpuTopkRoute,
        pub(crate) compact_topk: usize,
    }

    pub(crate) fn plan_sampling_route(
        gen_cfg: &GenerateConfig,
        history_is_empty: bool,
        environment: SamplingRouteEnvironment,
    ) -> SamplingRoutePlan {
        let block_route = choose_gpu_block_topk_route(
            gen_cfg.top_k,
            gen_cfg.top_p,
            environment.compact,
            environment.approximate_top_p,
        );
        let route = if block_route != GpuTopkRoute::CpuFallback {
            block_route
        } else {
            choose_gpu_topk_route(gen_cfg.top_k, environment.compact, environment.selection)
        };
        // This uniform `is_none()` check is not one policy for all three
        // callers: direct and prefix-cache generation already rejected
        // `logprobs: Some(_)` in their own `check_logprobs_not_set` preflight
        // before ever reaching this function, so for them the check below is
        // defense-in-depth and always sees `None`. Streaming has no such
        // preflight -- it intentionally supports `logprobs: Some(_)` to
        // capture per-token log-probabilities -- so for streaming this is the
        // real, load-bearing full-logit gate: compact sampling cannot
        // provide full-logit logprob semantics, so a logprobs capture
        // request must force the full-logit (non-compact) path here.
        let logprobs_eligible = gen_cfg.logprobs.is_none();
        let use_compact = route != GpuTopkRoute::CpuFallback
            && (gen_cfg.repetition_penalty == 1.0 || history_is_empty)
            && gen_cfg.grammar.is_none()
            && logprobs_eligible;

        if use_compact {
            SamplingRoutePlan {
                use_compact,
                compact_route: route,
                compact_topk: match route {
                    GpuTopkRoute::BlockArgmax => 1,
                    GpuTopkRoute::BlockTopK { local_k } => local_k as usize,
                    _ => gen_cfg.top_k,
                },
            }
        } else {
            SamplingRoutePlan {
                use_compact,
                compact_route: GpuTopkRoute::CpuFallback,
                compact_topk: 0,
            }
        }
    }

    pub(crate) fn apply_sampling_route_plan(
        plan: SamplingRoutePlan,
        compact_route: &mut GpuTopkRoute,
        compact_topk: &mut usize,
        compact_result: &mut Vec<crate::sampling::Candidate>,
    ) -> bool {
        *compact_route = plan.compact_route;
        *compact_topk = plan.compact_topk;
        if !plan.use_compact {
            compact_result.clear();
        }
        plan.use_compact
    }

    /// Resolves a `compact_route` to `(local_k, precompiled Stage-1 pipeline
    /// slot)`, or `None` if the route isn't a block-top-k route at all
    /// (`Argmax`/`Select64`/`HierarchicalK50`/`CpuFallback`) **or** if
    /// `local_k` has no precompiled Stage-1 pipeline variant.
    ///
    /// `choose_gpu_block_topk_route` only ever constructs `BlockArgmax` /
    /// `BlockTopK { local_k }` with `local_k` in `LM_HEAD_LOCAL_KS`, so the
    /// "route says Block* but local_k is unsupported" branch below should be
    /// unreachable in production. It is guarded rather than assumed: both
    /// call sites (`forward_prefill_impl`, `encode_final_head`) route their
    /// decision through this function, so an invariant violation here falls
    /// through to those call sites' existing full-logit exact-path code
    /// instead of panicking — the exact path is always correct, so this
    /// fail-closed fallback can only cost the compact-path speedup, never
    /// produce a wrong token. `debug_assert!` surfaces the violation loudly
    /// in tests/dev builds without making library code panic in release.
    fn resolve_block_local_k(route: GpuTopkRoute) -> Option<(u32, usize)> {
        let local_k = match route {
            GpuTopkRoute::BlockArgmax => 1u32,
            GpuTopkRoute::BlockTopK { local_k } => local_k,
            _ => return None,
        };
        match lm_head_local_k_slot(local_k) {
            Some(slot) => Some((local_k, slot)),
            None => {
                debug_assert!(
                    false,
                    "compact_route requested unsupported LOCAL_K {local_k} (not in \
                     LM_HEAD_LOCAL_KS); falling back to the exact full-logit path"
                );
                None
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub(crate) enum QuantFormat {
        #[default]
        Q8_0,
        Q4_0,
    }

    impl QuantFormat {
        fn from_env() -> Option<Self> {
            match std::env::var("LATTICE_QUANT_FORMAT")
                .ok()?
                .to_uppercase()
                .as_str()
            {
                "Q8_0" | "Q8" => Some(QuantFormat::Q8_0),
                "Q4_0" | "Q4" => Some(QuantFormat::Q4_0),
                _ => None,
            }
        }

        fn from_model_size(cfg: &Qwen35Config) -> Self {
            let h = cfg.hidden_size;
            let i = cfg.intermediate_size;
            let n = cfg.num_hidden_layers;
            let v = cfg.vocab_size;
            let est_params = n * (4 * h * h + 3 * h * i) + v * h;
            if est_params > 2_000_000_000 {
                QuantFormat::Q4_0
            } else {
                QuantFormat::Q8_0
            }
        }

        fn resolve(cfg: &Qwen35Config) -> Self {
            if let Some(env_fmt) = Self::from_env() {
                return env_fmt;
            }
            Self::from_model_size(cfg)
        }
    }

    /// Shared immutable model state for Qwen3.5/3.6 Metal GPU inference.
    ///
    /// Holds all state that does not change between requests: compiled Metal pipelines,
    /// layer weights, embedding tables, RoPE cosine/sine tables, and model config.
    /// Designed to be wrapped in `Arc<MetalQwen35Engine>` for concurrent multi-session
    /// serving — one engine shared across many `InferenceSession` instances.
    ///
    /// **Deferred**: The public concurrent API
    /// (`Engine::forward_step(&self, session: &mut InferenceSession, ...)`) is deferred
    /// to a follow-up PR; this PR delivers the struct-level separation as prerequisite
    /// infrastructure. For now, use `MetalQwen35State` as the backward-compatible
    /// single-session entry point.
    pub struct MetalQwen35Engine {
        // Device handle kept alive for GPU lifetime management (the MTLDevice must
        // outlive the command queue/pipelines it created).
        pub(crate) device: Device,
        pub(crate) queue: CommandQueue,
        pub(crate) pipelines: MetalQwen35Pipelines,
        pub(crate) layer_weights: Vec<(MetalLayerAttnWeights, MetalCommonLayerWeights)>,
        pub(crate) embed_tokens: Buffer, // [vocab_size, hidden] — f16
        pub(crate) embed_tokens_q8: Q4WeightBuf, // [vocab_size, hidden] — Q8_0/Q4 for logits GEMV
        pub(crate) final_norm: Buffer,   // [hidden] — f32
        pub(crate) rope_cos: Buffer,     // [max_pos * half_rope_dim] — f32
        pub(crate) rope_sin: Buffer,     // [max_pos * half_rope_dim] — f32
        pub(crate) config: Qwen35Config,
        pub(crate) quant_format: QuantFormat,
        // Immutable MTP weights only; cache/activations live on InferenceSession.
        pub(crate) mtp_weights: Option<MetalMtpWeights>,
        /// Rotation for QuaRot counter-rotate path. `Some` iff model was loaded from
        /// a QuaRot q4 artifact (from_q4_dir) and the config carries a rotation seed.
        /// Used by mtp_forward_one to apply R^T to embed and the captured target
        /// hidden before the O-space MTP forward, and R to mtp_h_out before the
        /// logits GEMV. Its presence also selects `MtpHiddenTap::PreFinalNorm`: the
        /// head keeps the pre-final-norm hidden because the final-norm scale is
        /// folded into lm_head in a rotated artifact.
        pub(crate) quarot_rotation: Option<crate::quant::quarot::hadamard::RandomizedHadamard>,
        /// The base model's own QuaRot rotation seed, independent of whether MTP
        /// weights are present (unlike `quarot_rotation`, which is only built when
        /// `mtp_weights` is `Some`). Set once at construction from the same
        /// `quarot_seed_opt` `from_q4_dir` computes for `quarot_rotation`; `None` for
        /// every non-QuaRot construction path. `load_lora_adapter` resolves an
        /// adapter's own (optional) seed against this value instead of silently
        /// skipping rotation when the adapter carries no seed of its own.
        pub(crate) base_quarot_seed: Option<u64>,
        /// Whether any active layer's FFN is `MetalFfnWeights::Moe`. Computed once at
        /// construction — layer FFN kinds never change after load (LoRA attaches to
        /// attention projections, MTP weights are separate) — so the batched-prefill
        /// MoE guards read this instead of rescanning `layer_weights` per call.
        pub(crate) has_moe_layer: bool,
    }

    /// Per-request mutable inference state.
    ///
    /// Holds all state that evolves during a single inference request: KV cache,
    /// GDN recurrent state, activation scratch buffers, MTP session cache, and
    /// sampling state. One instance is required per concurrent inference request.
    /// Created via `MetalQwen35Engine::new_session(max_cache_len)`.
    pub struct InferenceSession {
        pub(crate) activations: MetalQwen35Activations,
        pub(crate) gdn_states: Vec<GatedDeltaNetState>,
        #[allow(dead_code)]
        // GDN GPU scratch buffer, to be wired in a future GPU-GDN pass. Not yet
        // tracked by a filed issue.
        pub(crate) gdn_scratch: GatedDeltaNetFusedScratch,
        pub(crate) gdn_gpu_state: MetalGdnState,
        pub(crate) gdn_gpu_conv_out: Buffer, // [qkv_dim] temporary conv1d output
        pub(crate) kv_cache: MetalKvCache,
        /// Maximum batch prefill length (activation buffers sized for this).
        pub(crate) max_prefill: usize,
        /// When >0, forward_step dispatches GPU top-k; set to gen_cfg.top_k before generate loops.
        pub(crate) compact_topk: usize,
        /// Routing decision for the current generate call.
        pub(crate) compact_route: GpuTopkRoute,
        /// Populated by forward_step / forward_prefill when compact_topk > 0.
        pub(crate) compact_result: Vec<crate::sampling::Candidate>,
        // MTP per-session cache and activations (None if model has no MTP).
        pub(crate) mtp: Option<MetalMtpSession>,
        /// True for the remainder of the current request once `generate()`'s
        /// `use_mtp` (see [`super::mtp_route_active`]) commits to the MTP
        /// greedy/verify path. Distinct from `mtp.is_some()`, which only
        /// reports whether the checkpoint carries MTP weights: a checkpoint
        /// can ship MTP weights while a specific request runs with MTP
        /// disabled (`enable_mtp: Some(false)`), and pre-final-hidden-capture
        /// gates must not treat that request as MTP-active just because the
        /// weights happen to be resident (#1336 round 2). Reset to `false` by
        /// `reset_state()`, which every generate entry point calls before
        /// this flag is next read or set.
        pub(crate) mtp_active: bool,
        /// Which speculative decode route the current request ran on, and what the shared
        /// policy was offered on it. `None` until a request takes a speculative route and
        /// after `reset_state()`. A request on the batch-GEMM verifier is recorded as
        /// [`SpeculativeRoute::MtpBatchGemmLegacy`], which is how it is told apart from
        /// shared-route coverage.
        pub(crate) speculative_route: Option<SpeculativeRouteRecord>,
        pub(crate) gdn_checkpoints: Option<MetalGdnCheckpointPool>,
        pub(crate) last_pre_final_hidden: Vec<f32>,
        /// Cursor value (`kv_cache.seq_len`) at which `last_pre_final_hidden` was last
        /// written: the buffer holds the hidden state of the token at position
        /// `marker - 1` of the history that was live then. `None` means the buffer
        /// cannot be tied to any live token. A cursor that moved without a write
        /// leaves the marker behind; a rewind below it, a reset and a cross-turn
        /// restore clear it. Only `MtpTargetVerifier::last_hidden` reads it.
        pub(crate) last_hidden_cursor: Option<usize>,
        /// Raw (pre-final-RMSNorm) target hidden states captured by
        /// `forward_prefill_batched_chunk` for every prompt position in the most
        /// recent `forward_prefill`/`forward_prefill_all_logits` call, laid out as
        /// `n * hidden_size` f32 values in prompt order (#1340). Cleared at the top
        /// of `forward_prefill_impl` and drained by `mtp_prefill`, which rebuilds
        /// the MTP draft head's KV cache from it so the first draft attends to the
        /// prompt prefix instead of a single row.
        pub(crate) mtp_prefill_hidden: Vec<f32>,
        /// Opt-in: when set, every forward path that applies the final RMSNorm
        /// also copies the last token's normed hidden state into
        /// `activations.final_hidden`. Off by default so the decode loop pays
        /// nothing for a capture it never reads.
        pub(crate) capture_final_hidden: bool,
        /// Set at encode time by [`MetalQwen35State::encode_capture_final_hidden`],
        /// cleared by the embedding API before it dispatches. It answers "did
        /// THIS call write the capture buffer", which is the difference between
        /// an embedding of the caller's tokens and a stale vector from whatever
        /// ran last. A stale vector is well-formed, correctly sized, and wrong.
        pub(crate) final_hidden_captured: std::sync::atomic::AtomicBool,
        /// GDN state-traffic byte counters (issue #422). `None` unless both the
        /// `gdn-state-counters` feature is on and `LATTICE_GDN_STATE_COUNTERS` is enabled.
        #[cfg(feature = "gdn-state-counters")]
        pub(crate) gdn_state_traffic: Option<GdnStateTrafficCounters>,
    }

    impl InferenceSession {
        /// Authoritative decode cursor. Derived from `kv_cache.seq_len` rather
        /// than stored separately, so the two can never drift (#1708): every
        /// forward/prefill/rollback/verify path already advances or rewinds
        /// `kv_cache.seq_len` as the single source of truth, and a duplicate
        /// field only tracked a subset of those sites.
        #[allow(dead_code)] // read by tests only; no production path branches on it
        pub(crate) fn position(&self) -> usize {
            self.kv_cache.seq_len
        }

        /// Jump the live cache cursor directly to `position` (cross-turn cache
        /// restore, batched-prefill chunk advance), rather than the per-token
        /// `+= 1` the forward-step paths use.
        ///
        /// A jump replaces the history the buffered hidden state belonged to, so it
        /// clears the freshness marker; a caller that also writes the buffer marks it
        /// again afterwards.
        pub(crate) fn set_position(&mut self, position: usize) {
            self.kv_cache.seq_len = position;
            self.last_hidden_cursor = None;
        }

        /// Move the cursor to `position` for a rollback (or a re-run from an earlier
        /// slot). A rewind below the marker discards the history the buffered hidden
        /// state came from, so it clears the marker even if replaying later brings the
        /// cursor back to the same value. A rewind to exactly the marker keeps the
        /// first `marker` tokens, so the buffer still belongs to the last of them.
        pub(crate) fn rewind_position(&mut self, position: usize) {
            self.kv_cache.seq_len = position;
            if self.last_hidden_cursor.is_some_and(|m| position < m) {
                self.last_hidden_cursor = None;
            }
        }

        /// Record that `last_pre_final_hidden` now holds the hidden state of the token at
        /// `cursor - 1`.
        pub(crate) fn mark_pre_final_hidden(&mut self, cursor: usize) {
            self.last_hidden_cursor = Some(cursor);
        }
    }

    // ---------------------------------------------------------------------------
    // LoRA adapter state (ADR-045 step 3)
    // ---------------------------------------------------------------------------

    /// Metal buffers for a single LoRA-adapted projection.
    struct MetalLoraProjection {
        a_buf: Buffer,
        b_buf: Buffer,
        rank: u32,
        d_in: u32,
        d_out: u32,
    }

    /// Loaded LoRA adapter state on Metal GPU.
    pub(crate) struct MetalLoraAdapter {
        /// Nested lookup: layer_idx → module_name → projection. No allocation on lookup.
        projections: Vec<std::collections::HashMap<String, MetalLoraProjection>>,
        scale: f32,
        /// Shared rank scratch: `max_rows * max_rank` floats. A decode step uses the first
        /// row; a batched prefill chunk uses one `rank`-wide row per token.
        intermediate: Buffer,
        // max rank used to size `intermediate` buffer at load time; stored for adapter introspection
        max_rank: u32,
        /// Token rows `intermediate` can hold: the state's maximum prefill chunk.
        max_rows: usize,
    }

    impl MetalLoraAdapter {
        fn get_projection(&self, layer_idx: usize, module: &str) -> Option<&MetalLoraProjection> {
            self.projections.get(layer_idx)?.get(module)
        }
    }

    /// Input data for a single LoRA layer to be loaded onto GPU.
    pub struct LoraLayerData {
        /// Transformer layer index (0-based).
        pub layer_idx: usize,
        /// Projection module name (e.g., `"q_proj"`, `"o_proj"`).
        pub module: String,
        /// A matrix, row-major `(rank × d_in)`.
        pub a: Vec<f32>,
        /// B matrix, row-major `(d_out × rank)`.
        pub b: Vec<f32>,
        /// LoRA rank.
        pub rank: usize,
        /// Input dimension.
        pub d_in: usize,
        /// Output dimension.
        pub d_out: usize,
    }

    /// Maximum total rank allowed across all adapters in a single blend.
    ///
    /// The Metal GEMV kernels assume a modest rank budget (≤ ~64 per adapter in a
    /// typical mixture).  This cap bounds allocations and rejects adversarially large
    /// adapter pools before `Vec::with_capacity`.
    ///
    /// Re-exported from [`lattice_fann::lora::MAX_BLEND_RANK_TOTAL`], the shared
    /// cap this crate and `lattice-tune`'s CPU LoRA blend both enforce. Only
    /// referenced by tests below — non-test code calls `lattice_fann::lora::*`
    /// directly so the cap itself lives in exactly one place.
    #[cfg(test)]
    pub(crate) const MAX_BLEND_RANK_TOTAL: usize = lattice_fann::lora::MAX_BLEND_RANK_TOTAL;

    /// Reject any element of `inputs` whose inner layer slice is empty.
    ///
    /// An empty inner slice passes an outer `inputs.is_empty()` check (the
    /// outer slice itself is non-empty) but contributes nothing to a
    /// group-by-`(layer_idx, module)` pass, silently producing an empty
    /// result instead of an admission error. Split out from
    /// [`blend_lora_layer_data`] so callers that mutate state before
    /// blending (e.g. [`MetalQwen35State::generate_with_lora_mixture`], which
    /// unloads the currently loaded adapter first) can run this check first
    /// and reject the request before that mutation, not after.
    fn reject_empty_inner_layers(
        inputs: &[(&[LoraLayerData], f32)],
    ) -> Result<(), crate::error::InferenceError> {
        for (idx, (layers, _)) in inputs.iter().enumerate() {
            if layers.is_empty() {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "blend_lora_layer_data: inputs[{idx}] has an empty layer slice"
                )));
            }
        }
        Ok(())
    }

    /// Blend multiple sets of [`LoraLayerData`] into one rank-Σr set for use with
    /// the single-slot [`MetalQwen35State::load_lora_adapter`] API.
    ///
    /// # Blend math
    ///
    /// For adapter `e` with layer data `L_e` and mixture weight `w_e`, the
    /// effective delta is `w_e · B_e @ (A_e @ x)` (the `alpha/rank` scale is
    /// NOT present in `LoraLayerData` — it is passed separately as `scale` to
    /// `load_lora_adapter`; here `w_e` IS that effective scale).
    ///
    /// The blended matrices are:
    /// - `A_blend = vconcat[A_1; …; A_N]`       shape `(Σrank_e) × d_in`
    /// - `B_blend = hconcat[w_1·B_1 | …]`       shape `d_out × (Σrank_e)`
    ///
    /// The resulting layer set should be loaded with `scale = 1.0` because
    /// the per-adapter weights are already folded into the B blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `inputs` is empty
    /// - Any element of `inputs` has an empty inner layer slice
    /// - Any weight is not finite
    /// - Two adapters have conflicting `d_in` / `d_out` for the same `(layer_idx, module)`
    /// - The summed rank for a single projection exceeds `MAX_BLEND_RANK_TOTAL`
    /// - The aggregate blend size across all projections exceeds `MAX_BLEND_TOTAL_ELEMENTS`
    /// - A layer's A or B buffer length does not match its declared `rank`, `d_in`, or `d_out`
    /// - Rank accumulation or a size product overflows `usize`
    pub fn blend_lora_layer_data(
        inputs: &[(&[LoraLayerData], f32)],
    ) -> Result<Vec<LoraLayerData>, crate::error::InferenceError> {
        use crate::error::InferenceError;
        use std::collections::HashMap;

        if inputs.is_empty() {
            return Err(InferenceError::InvalidInput(
                "blend_lora_layer_data: inputs must not be empty".into(),
            ));
        }

        // An empty inner layer slice passes the outer `inputs.is_empty()`
        // guard above (the outer slice itself is non-empty) but contributes
        // nothing to `grouped` below, silently producing `Ok(Vec::new())`
        // instead of an admission error. Callers that unload the currently
        // loaded adapter before blending (e.g. `generate_with_lora_mixture`)
        // must reject this before that mutation, not after — see
        // `reject_empty_inner_layers` and its call site there.
        reject_empty_inner_layers(inputs)?;

        for (idx, (_, w)) in inputs.iter().enumerate() {
            lattice_fann::lora::check_finite_weight("blend_lora_layer_data", idx, *w)
                .map_err(InferenceError::InvalidInput)?;
        }

        // Group by (layer_idx, module): collect refs to layer data and effective weights.
        // Keyed by `&str` borrowed from each layer's own `module: String` (which
        // outlives this function body via `inputs`), not a per-layer clone.
        let mut grouped: HashMap<(usize, &str), Vec<(&LoraLayerData, f32)>> = HashMap::new();
        for (layers, weight) in inputs {
            for layer in *layers {
                grouped
                    .entry((layer.layer_idx, layer.module.as_str()))
                    .or_default()
                    .push((layer, *weight));
            }
        }

        // Pre-allocation planning -- the per-group MAX_BLEND_RANK_TOTAL cap,
        // the aggregate MAX_BLEND_TOTAL_ELEMENTS cap, and cross-adapter
        // dimension agreement -- is the ONE shared check the adapter
        // residency registry's state publication also runs (issue #1735), so
        // a `GET /v1/lora` report of whether the resident set can be blended
        // and this function's own refusal can never disagree. It sees only
        // each entry's shape, never an A/B buffer.
        //
        // `grouped` above is already the grouping `plan_blend` would build
        // internally from a flat iterator, so this calls `plan_grouped`
        // directly over `grouped`'s own entries instead of `plan_blend`,
        // skipping a second `HashMap` and per-group `Vec` that would
        // otherwise re-derive the same grouping from scratch on every blend.
        let groups = grouped.iter().map(|(&(layer_idx, module), entries)| {
            (
                layer_idx,
                module,
                entries
                    .iter()
                    .map(|(layer, _)| lattice_fann::lora::BlendProjection {
                        layer_idx: layer.layer_idx,
                        module: layer.module.as_str(),
                        rank: layer.rank,
                        d_in: layer.d_in,
                        d_out: layer.d_out,
                    }),
            )
        });
        let planned = lattice_fann::lora::plan_grouped("blend_lora_layer_data", groups)
            .map_err(InferenceError::InvalidInput)?;

        let mut result: Vec<LoraLayerData> = Vec::with_capacity(planned.len());
        for plan in planned {
            let entries = grouped.get(&(plan.layer_idx, plan.module)).ok_or_else(|| {
                InferenceError::Inference(format!(
                    "blend_lora_layer_data: planned group (layer {}, module {}) is absent \
                         from the grouped adapter set",
                    plan.layer_idx, plan.module
                ))
            })?;
            let d_in = plan.d_in;
            let d_out = plan.d_out;
            let rank_total = plan.rank_total;

            // Validate source slice lengths before any allocation: a malformed adapter
            // whose A or B buffer is the wrong size would cause out-of-bounds copies.
            // `plan_grouped` never sees these buffers, so this check stays here.
            for (idx, (entry, _)) in entries.iter().enumerate() {
                lattice_fann::lora::check_buffer_lengths(
                    "blend_lora_layer_data",
                    idx,
                    entry.rank,
                    d_in,
                    d_out,
                    entry.a.len(),
                    entry.b.len(),
                )
                .map_err(InferenceError::InvalidInput)?;
            }

            let a_buf_len = rank_total.checked_mul(d_in).ok_or_else(|| {
                InferenceError::InvalidInput(
                    "blend_lora_layer_data: rank_total * d_in overflowed usize".into(),
                )
            })?;
            let b_buf_len = d_out.checked_mul(rank_total).ok_or_else(|| {
                InferenceError::InvalidInput(
                    "blend_lora_layer_data: d_out * rank_total overflowed usize".into(),
                )
            })?;

            // A_blend: vertical stack of A matrices.
            let mut a_blend: Vec<f32> = Vec::with_capacity(a_buf_len);
            for (layer, _) in entries.iter() {
                a_blend.extend_from_slice(&layer.a);
            }

            // B_blend: horizontal concat of weight-scaled B column blocks.
            // B is row-major (d_out × rank); row r: b[r*rank..(r+1)*rank].
            let mut b_blend = vec![0.0f32; b_buf_len];
            let mut col_offset = 0usize;
            for (layer, eff_weight) in entries.iter() {
                let r_e = layer.rank;
                for row in 0..d_out {
                    let dst = row * rank_total + col_offset;
                    let src = row * r_e;
                    for c in 0..r_e {
                        b_blend[dst + c] = eff_weight * layer.b[src + c];
                    }
                }
                col_offset += r_e;
            }

            result.push(LoraLayerData {
                layer_idx: plan.layer_idx,
                module: plan.module.to_string(),
                a: a_blend,
                b: b_blend,
                rank: rank_total,
                d_in,
                d_out,
            });
        }

        Ok(result)
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
    // Cross-turn KV prefix cache (#462)
    // ---------------------------------------------------------------------------

    /// Per-call statistics for a cache-aware generate/chat call.
    #[derive(Debug, Clone)]
    pub struct CrossTurnCacheStats {
        pub slot_id: crate::kv_cache::CrossTurnSlotId,
        pub prompt_tokens: usize,
        pub reused_tokens: usize,
        pub prefetched_tokens: usize,
        pub mode: crate::kv_cache::PrefixReuseMode,
    }

    /// [`GenerateOutput`] plus cross-turn cache stats for the call that produced it.
    #[derive(Debug, Clone)]
    pub struct CachedGenerateOutput {
        pub output: GenerateOutput,
        pub cache: CrossTurnCacheStats,
    }

    /// [`ChatCompletionOutput`] plus cross-turn cache stats for the call that produced it.
    #[derive(Debug, Clone)]
    pub struct CachedChatCompletionOutput {
        pub output: ChatCompletionOutput,
        pub cache: CrossTurnCacheStats,
    }

    /// One sparse GDN checkpoint retained at an earlier turn boundary
    /// (#590): the token length it represents plus the exact GDN
    /// recurrent-state snapshot taken at that boundary. Valid only while
    /// the live KV rows below `len` still hold the prefix the snapshot was
    /// taken against — the same lifetime as the owning entry, enforced by
    /// storing checkpoints inside `MetalCrossTurnPrefixEntry` and pruning
    /// on every save via `checkpoint_survives_save`.
    #[derive(Debug, Clone)]
    struct MetalGdnCheckpoint {
        len: usize,
        snapshot: crate::attention::gdn::GdnSnapshot,
    }

    /// Maximum sparse checkpoints retained per entry (#590). Each GDN
    /// snapshot is ~19MiB on Qwen3.5-0.8B (#516's measurement), so the
    /// worst-case cross-turn memory is `CAP + 1` snapshots (~76MiB): the
    /// ring plus the entry's own boundary snapshot. Eviction keeps the
    /// DEEPEST boundaries — in append-only chat flow boundaries grow
    /// monotonically, so this equals keeping the most recent turns; a
    /// deeper checkpoint also saves the most re-prefill work when a late
    /// turn is edited or regenerated. The cost of an evicted checkpoint is
    /// only a longer replay prefill, never wrong output.
    const CROSS_TURN_GDN_CHECKPOINT_CAP: usize = 3;

    /// One retained cross-turn entry: the generic (Metal-agnostic) planning
    /// record plus the exact GDN recurrent-state snapshot taken at the same
    /// token boundary. The live full-attention KV buffers are NOT copied here
    /// — they stay in `self.session.kv_cache` and are only logically
    /// truncated/advanced (see `restore_cross_turn_prefix`).
    #[derive(Debug, Clone)]
    struct MetalCrossTurnPrefixEntry {
        generic: crate::kv_cache::CrossTurnPrefixEntry,
        gdn_snapshot: crate::attention::gdn::GdnSnapshot,
        /// Sparse GDN checkpoints at earlier turn boundaries, ascending by
        /// `len`, at most [`CROSS_TURN_GDN_CHECKPOINT_CAP`] (#590). These
        /// enable `PrefixReuseMode::ReplayFromCheckpoint` when a new prompt
        /// diverges mid-history instead of falling all the way back to
        /// `FullRefill`.
        checkpoints: Vec<MetalGdnCheckpoint>,
    }

    /// Worker-local, single-live-entry cross-turn prefix cache (#516).
    ///
    /// Ownership invariant: at most ONE `MetalCrossTurnPrefixEntry` is ever
    /// retained, bounded by design (never an unbounded per-slot map). The
    /// entry is valid **iff** the live `self.session.kv_cache` rows and
    /// `self.session.position` were left by exactly the generation call that
    /// saved it. Any other mutation of live KV/GDN state — a different
    /// slot's generation, the plain (non-cache-aware) generate path, a
    /// `reset_state()` call, or a LoRA adapter load/unload — must invalidate
    /// the entry before that mutation is allowed to proceed, because the
    /// entry does not own private copies of the full-attention KV buffers
    /// (see `MetalCrossTurnPrefixEntry` above): it can only ever describe
    /// the ONE live buffer this process currently has. `get`/`take` both
    /// enforce the slot match themselves so a caller can never accidentally
    /// restore a different slot's entry.
    #[derive(Debug, Default)]
    struct MetalCrossTurnPrefixCache {
        entry: Option<MetalCrossTurnPrefixEntry>,
    }

    impl MetalCrossTurnPrefixCache {
        /// Returns the retained entry only if it belongs to `slot_id`. A
        /// different (or absent) slot is always a miss — never a partial or
        /// stale match — so the planner falls back to `FullRefill`.
        fn get(
            &self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
        ) -> Option<&MetalCrossTurnPrefixEntry> {
            self.entry.as_ref().filter(|e| e.generic.slot_id == slot_id)
        }

        /// Replace whatever is currently retained (if anything, for any
        /// slot) with `entry`. At most one GDN snapshot is ever alive.
        fn insert(&mut self, entry: MetalCrossTurnPrefixEntry) {
            self.entry = Some(entry);
        }

        /// Remove and return the retained entry iff it belongs to
        /// `slot_id`, leaving the cache empty either way it existed for
        /// that slot. Used by `restore_cross_turn_prefix` so an
        /// `ExactAppend` restore consumes its GDN snapshot instead of
        /// cloning it, and so the cache is provably empty for the
        /// remainder of the call until a fresh save succeeds.
        fn take(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
        ) -> Option<MetalCrossTurnPrefixEntry> {
            if self
                .entry
                .as_ref()
                .is_some_and(|e| e.generic.slot_id == slot_id)
            {
                self.entry.take()
            } else {
                None
            }
        }

        fn remove(&mut self, slot_id: crate::kv_cache::CrossTurnSlotId) {
            if self
                .entry
                .as_ref()
                .is_some_and(|e| e.generic.slot_id == slot_id)
            {
                self.entry = None;
            }
        }

        fn clear(&mut self) {
            self.entry = None;
        }
    }

    /// Dispatch/readback-site counters for the Metal path-proof probe.
    ///
    /// `&self`-compatible (interior mutability) because the dispatch helpers that
    /// increment these run inside `&self` methods sharing one command encoder.
    #[derive(Default)]
    pub struct PathProofCounters {
        pub prefill_kv_batch: AtomicU64,
        pub prefill_attn_batched: AtomicU64,
        pub prefill_hidden_readback: AtomicU64,
        pub prefill_full_vocab_logit_readback: AtomicU64,
        pub prefill_compact_candidate_logit_readback: AtomicU64,
        pub decode_kv_copy: AtomicU64,
        pub decode_attn_direct: AtomicU64,
        pub decode_attn_split_partial: AtomicU64,
        pub decode_attn_split_reduce: AtomicU64,
        pub decode_hidden_readback: AtomicU64,
        pub decode_full_vocab_logit_readback: AtomicU64,
        pub decode_compact_candidate_logit_readback: AtomicU64,
        /// Device-to-host bytes moved by the logit readbacks counted above.
        ///
        /// These are not redundant with the counts, for two independent reasons. A
        /// compact-eligible request that takes the full-vocab path moves
        /// `vocab_size * 4` bytes instead of `k * 8`, a ratio near 1187 at a 151936
        /// vocabulary and `k = 64`, while both spellings increment some counter by
        /// exactly 1. And `prefill_full_vocab_logit_readback` is incremented by two
        /// sites that move different amounts: the perplexity path reads
        /// `n * vocab_size` values, the ordinary path reads `vocab_size`. The event
        /// count reports 1 for both and destroys the factor of `n`, with no
        /// substitution and no adversary involved.
        pub prefill_full_vocab_logit_readback_bytes: AtomicU64,
        pub prefill_compact_candidate_logit_readback_bytes: AtomicU64,
        pub decode_full_vocab_logit_readback_bytes: AtomicU64,
        pub decode_compact_candidate_logit_readback_bytes: AtomicU64,
    }

    impl PathProofCounters {
        fn reset(&self) {
            self.prefill_kv_batch.store(0, Ordering::Relaxed);
            self.prefill_attn_batched.store(0, Ordering::Relaxed);
            self.prefill_hidden_readback.store(0, Ordering::Relaxed);
            self.prefill_full_vocab_logit_readback
                .store(0, Ordering::Relaxed);
            self.prefill_compact_candidate_logit_readback
                .store(0, Ordering::Relaxed);
            self.decode_kv_copy.store(0, Ordering::Relaxed);
            self.decode_attn_direct.store(0, Ordering::Relaxed);
            self.decode_attn_split_partial.store(0, Ordering::Relaxed);
            self.decode_attn_split_reduce.store(0, Ordering::Relaxed);
            self.decode_hidden_readback.store(0, Ordering::Relaxed);
            self.decode_full_vocab_logit_readback
                .store(0, Ordering::Relaxed);
            self.decode_compact_candidate_logit_readback
                .store(0, Ordering::Relaxed);
            self.prefill_full_vocab_logit_readback_bytes
                .store(0, Ordering::Relaxed);
            self.prefill_compact_candidate_logit_readback_bytes
                .store(0, Ordering::Relaxed);
            self.decode_full_vocab_logit_readback_bytes
                .store(0, Ordering::Relaxed);
            self.decode_compact_candidate_logit_readback_bytes
                .store(0, Ordering::Relaxed);
        }
    }

    /// Snapshot of [`PathProofCounters`] plus the KV-cache precision mode in effect,
    /// suitable for formatting into the `[METAL_PATH_PROOF]` stderr marker.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct PathProofSnapshot {
        pub prefill_kv_batch: u64,
        pub prefill_attn_batched: u64,
        pub decode_kv_copy: u64,
        pub decode_attn_direct: u64,
        pub decode_attn_split_partial: u64,
        pub decode_attn_split_reduce: u64,
        pub kv_f16: bool,
    }

    /// Explicit hidden-readback counts recorded by the Metal path-proof probe.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct HiddenReadbackPathProofSnapshot {
        pub decode: u64,
        pub prefill: u64,
    }

    /// Explicit vocabulary-logit-readback counts recorded by the Metal path-proof
    /// probe, split by shape: a full-vocab readback (`vocab_size` f32 values) versus
    /// a compact-candidate readback (the top-k shortlist only). These are separate
    /// dispatch sites in [`MetalQwen35State`] from the hidden-state readback counted
    /// by [`HiddenReadbackPathProofSnapshot`] — a request eligible for the compact
    /// path that instead takes the full-vocab path changes this snapshot's shape
    /// without changing that one's.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct LogitReadbackPathProofSnapshot {
        pub decode_full_vocab: u64,
        pub decode_compact_candidate: u64,
        pub prefill_full_vocab: u64,
        pub prefill_compact_candidate: u64,
        /// Device-to-host bytes for each of the four readback shapes above.
        ///
        /// ADR-090 D7 asks for transfers *and* bytes around the selected request (the
        /// bare label `D7` is reused elsewhere in this file for an unrelated item, so
        /// the ADR is named here rather than assumed); the counts
        /// above are the transfers. A shape check can be satisfied by a substitution
        /// that moves three orders of magnitude more data under a different field
        /// name, which is what these make assertable.
        pub decode_full_vocab_bytes: u64,
        pub decode_compact_candidate_bytes: u64,
        pub prefill_full_vocab_bytes: u64,
        pub prefill_compact_candidate_bytes: u64,
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

    /// Create a Metal buffer holding f32 data in shared (unified) memory.
    fn make_buffer(device: &Device, data: &[f32], label: &str) -> Buffer {
        let byte_len = std::mem::size_of_val(data) as u64;
        let buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        buf
    }

    /// Create a Metal buffer holding f16 (half-precision) data converted from f32.
    ///
    /// Converts each f32 element to IEEE-754 half precision at load time. The resulting
    /// buffer is half the size of the f32 equivalent and is consumed by `matmul_bt_half`.
    fn make_buffer_f16(device: &Device, data: &[f32], label: &str) -> Buffer {
        let f16_data: Vec<u16> = data.iter().map(|&x| f32_to_f16(x)).collect();
        let byte_len = (f16_data.len() * std::mem::size_of::<u16>()) as u64;
        let buf = device.new_buffer_with_data(
            f16_data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        buf
    }

    /// Q8_0 block: f16 scale (2 bytes) + 32 int8 values = 34 bytes.
    const QK8_0: usize = 32;
    const Q8_0_BLOCK_SIZE: usize = 2 + QK8_0;
    const QK4_0: usize = 32;
    const Q4_0_BLOCK_SIZE: usize = 20;

    /// Quantize a row of f32 weights to Q8_0 blocks.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any element in `src` is non-finite (NaN or ±inf).
    /// IEEE 754: `NaN > x` is always false, so a fold-max over NaN leaves
    /// `amax` unchanged and the NaN is silently cast to 0 by the `clamp` below.
    fn quantize_row_q8_0(src: &[f32]) -> Result<Vec<u8>, String> {
        assert_eq!(src.len() % QK8_0, 0);
        let nblocks = src.len() / QK8_0;
        let mut packed = Vec::with_capacity(nblocks * Q8_0_BLOCK_SIZE);
        for b in 0..nblocks {
            let block = &src[b * QK8_0..(b + 1) * QK8_0];
            for &v in block {
                if !v.is_finite() {
                    return Err(format!(
                        "Q8_0 weight block {b} contains a non-finite value ({v}); \
                         source weights must be finite"
                    ));
                }
            }
            let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            packed.extend_from_slice(&f32_to_f16(d).to_ne_bytes());
            for &v in block {
                packed.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
        Ok(packed)
    }

    /// Create a Metal buffer with Q8_0 quantized weights from f32 data.
    fn make_buffer_q8_0(device: &Device, data: &[f32], label: &str) -> Result<Buffer, String> {
        assert_eq!(data.len() % QK8_0, 0);
        let packed = quantize_row_q8_0(data)?;
        let byte_len = packed.len() as u64;
        let buf = device.new_buffer_with_data(
            packed.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        Ok(buf)
    }

    fn make_buffer_q4_0(device: &Device, data: &[f32], label: &str) -> Result<Buffer, String> {
        assert_eq!(
            data.len() % QK4_0,
            0,
            "make_buffer_q4_0: data length {} not divisible by {QK4_0}",
            data.len()
        );
        let packed = quantize_row_q4_0(data).map_err(|e| e.to_string())?;
        let byte_len = packed.len() as u64;
        let buf = device.new_buffer_with_data(
            packed.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        Ok(buf)
    }

    fn make_zero_buffer(device: &Device, num_floats: usize, label: &str) -> Buffer {
        let byte_len = (num_floats * std::mem::size_of::<f32>()) as u64;
        let buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        buf.set_label(label);
        buf
    }

    /// Allocate a zero-initialised f16 buffer holding `num_halves` half-precision elements.
    fn make_zero_buffer_f16(device: &Device, num_halves: usize, label: &str) -> Buffer {
        let byte_len = (num_halves * std::mem::size_of::<u16>()) as u64;
        let buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        buf.set_label(label);
        buf
    }

    /// Allocate a zero-initialised byte buffer (not float-sized).
    /// Used for top-k candidate scratch buffers whose element size is 8 bytes.
    fn make_zero_byte_buffer(device: &Device, byte_len: usize, label: &str) -> Buffer {
        let buf = device.new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        buf.set_label(label);
        buf
    }

    /// GemmParams layout matching the MSL GemmParams struct (6 × u32 = 24 bytes).
    #[repr(C)]
    struct GemmParams {
        m: u32,
        n: u32,
        k: u32,
        lda: u32,
        ldb: u32,
        ldc: u32,
    }

    fn div_ceil(a: u64, b: u64) -> u64 {
        a.div_ceil(b)
    }

    /// Convert f32 to IEEE-754 half-precision (f16) stored as u16 bits.
    ///
    /// Thin wrapper over the single always-compiled scalar encoder in
    /// [`crate::weights::half_bits`] (lattice#799) — this module used to
    /// carry its own independent copy of the bit-twiddling arithmetic.
    #[inline]
    fn f32_to_f16(x: f32) -> u16 {
        crate::weights::half_bits::f32_to_f16_bits(x)
    }

    /// Convert f16 bits (u16) back to f32.
    ///
    /// Used for CPU-side reads of f16 weight buffers when needed (e.g., small
    /// GDN projections dispatched on CPU). Thin wrapper over
    /// [`crate::weights::half_bits::f16_bits_to_f32`] (lattice#799).
    #[inline]
    fn f16_to_f32(bits: u16) -> f32 {
        crate::weights::half_bits::f16_bits_to_f32(bits)
    }

    /// Convert a contiguous slice of IEEE-754 half-precision values (stored as u16 bits)
    /// to f32, writing into a pre-allocated destination buffer.
    ///
    /// Scalar `f16_to_f32` throughout. A 4-lane NEON fast path (`vcvt_f32_f16`) was
    /// tried on aarch64 but requires the nightly-only `stdarch_neon_f16` feature
    /// (rust-lang/rust#136306) and broke stable-toolchain builds (#568) — this runs
    /// off the GPU decode hot path (one row per token embedding lookup / readback),
    /// so the scalar cost is not worth the unstable dependency.
    ///
    /// # Safety
    /// - `src` must point to at least `n` contiguous, initialized u16 values.
    /// - `dst` must point to at least `n` contiguous, writable f32 values.
    /// - `n` may be zero (no-op).
    unsafe fn convert_f16_row(src: *const u16, dst: *mut f32, n: usize) {
        for i in 0..n {
            *dst.add(i) = f16_to_f32(*src.add(i));
        }
    }

    /// Read f32 slice from a Metal shared-mode buffer.
    ///
    /// SAFETY: Buffer must be StorageModeShared with sufficient length and no
    /// concurrent GPU writes (command buffer completed or not yet submitted).
    unsafe fn read_buffer(buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        let mut out = vec![0.0f32; len];
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
        out
    }

    /// Device-side layout of one top-k candidate: `struct TopKCandidate { float logit;
    /// uint token_id; }` in the shader.
    ///
    /// Module-scoped rather than local to [`MetalQwen35State::read_topk_candidates`] so the
    /// readback sites can size their transfers with `size_of::<GpuCandidate>()`. The shader
    /// owns this layout, so a literal byte count at those sites would rot silently the first
    /// time the shader struct changed and nothing would fail.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub(crate) struct GpuCandidate {
        logit: f32,
        token_id: u32,
    }

    /// Read f32 slice from a Metal shared-mode buffer starting at a given element offset.
    ///
    /// SAFETY: Same requirements as `read_buffer`; `offset_f32 + len` must be within bounds.
    #[allow(dead_code)] // debug/calibration helper used for CPU-side readback at non-zero offsets
    unsafe fn read_buffer_offset(buf: &Buffer, offset_f32: usize, len: usize) -> Vec<f32> {
        let ptr = (buf.contents() as *const f32).add(offset_f32);
        std::slice::from_raw_parts(ptr, len).to_vec()
    }

    /// Read f16 data from a Metal shared-mode buffer, converting to f32.
    ///
    /// Used for CPU-side reads of f16 weight buffers (GDN small projections).
    ///
    /// SAFETY: Buffer must be StorageModeShared with sufficient length and no
    /// concurrent GPU writes (command buffer completed or not yet submitted).
    #[allow(dead_code)] // CPU-side f16 readback helper; used in debugging and future calibration paths
    unsafe fn read_buffer_f16(buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const u16;
        let mut out = vec![0.0f32; len];
        // SAFETY: ptr has len u16 values; out.as_mut_ptr() has len f32 values.
        convert_f16_row(ptr, out.as_mut_ptr(), len);
        out
    }

    /// Write f32 slice into a Metal shared-mode buffer.
    ///
    /// SAFETY: Buffer must be StorageModeShared with sufficient capacity and no
    /// concurrent GPU reads (command buffer completed or not yet submitted).
    unsafe fn write_buffer(buf: &Buffer, data: &[f32]) {
        let dst = buf.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }

    /// Single derivation point for the engine's `has_moe_layer` flag. Both
    /// state constructors (f16 and Q4) build the flag from this same layer
    /// scan, so a change to layer classification cannot update one
    /// constructor and silently leave the other inconsistent.
    fn derive_has_moe_layer(
        layer_weights: &[(MetalLayerAttnWeights, MetalCommonLayerWeights)],
    ) -> bool {
        layer_weights
            .iter()
            .any(|(_, common_w)| matches!(common_w.ffn, MetalFfnWeights::Moe(_)))
    }

    /// Build interleaved partial RoPE cos/sin tables.
    /// For Qwen3.5: partial_rotary_factor=0.25, head_dim=256 => rope_dim=64, half=32.
    /// Tables: [max_pos, half_rope_dim] each.
    ///
    /// Qwen3.5 uses `theta^(-2i/rope_dim)` in the denominator (verified
    /// empirically by PPL regression when using head_dim). The partial-RoPE
    /// frequency spectrum is compressed into the first `rope_dim` dimensions
    /// rather than sliced from a wider head_dim spectrum.
    fn build_rope_interleaved(rope_dim: usize, max_pos: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
        let half = rope_dim / 2;
        let mut cos_data = Vec::with_capacity(max_pos * half);
        let mut sin_data = Vec::with_capacity(max_pos * half);

        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64);
                let angle = pos as f64 * freq;
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
        }

        (cos_data, sin_data)
    }

    impl MetalKvCache {
        fn new(
            device: &Device,
            num_full_layers: usize,
            kv_dim: usize,
            max_cache_len: usize,
            use_kv_f16: bool,
        ) -> Self {
            let num_elems = max_cache_len * kv_dim;
            let mut k_bufs = Vec::with_capacity(num_full_layers);
            let mut v_bufs = Vec::with_capacity(num_full_layers);
            for i in 0..num_full_layers {
                if use_kv_f16 {
                    k_bufs.push(make_zero_buffer_f16(
                        device,
                        num_elems,
                        &format!("kv_k_{i}"),
                    ));
                    v_bufs.push(make_zero_buffer_f16(
                        device,
                        num_elems,
                        &format!("kv_v_{i}"),
                    ));
                } else {
                    k_bufs.push(make_zero_buffer(device, num_elems, &format!("kv_k_{i}")));
                    v_bufs.push(make_zero_buffer(device, num_elems, &format!("kv_v_{i}")));
                }
            }
            let cache = Self {
                k_bufs,
                v_bufs,
                seq_len: 0,
                kv_dim,
                max_cache_len,
            };
            // ADR-064: runtime KV layout assertion (debug builds only).
            // Layout is f16 (2 bytes/elem) when use_kv_f16, f32 (4 bytes/elem) otherwise.
            #[cfg(debug_assertions)]
            if num_full_layers > 0 {
                let element_size = if use_kv_f16 { 2 } else { 4 };
                let expected_bytes = max_cache_len * kv_dim * element_size;
                let layout_name = if use_kv_f16 { "f16" } else { "f32" };
                for (i, (k, v)) in cache.k_bufs.iter().zip(cache.v_bufs.iter()).enumerate() {
                    debug_assert_eq!(
                        k.length() as usize,
                        expected_bytes,
                        "KV cache K[{i}] size mismatch: expected {layout_name} layout ({expected_bytes} bytes)"
                    );
                    debug_assert_eq!(
                        v.length() as usize,
                        expected_bytes,
                        "KV cache V[{i}] size mismatch: expected {layout_name} layout ({expected_bytes} bytes)"
                    );
                }
            }
            cache
        }

        fn reset(&mut self) {
            self.seq_len = 0;
            // No need to zero the buffers — seq_len controls valid range.
        }
    }

    // -----------------------------------------------------------------------
    // Layer importance scoring — calibration harness
    // -----------------------------------------------------------------------

    /// Per-layer hidden-state snapshot for calibration scoring.
    struct LayerTrace {
        layer_idx: usize,
        input: Vec<f32>,
        output: Vec<f32>,
    }

    /// Layer importance score derived from hidden-state cosine similarity.
    #[derive(Debug, Clone)]
    pub struct LayerImportanceScore {
        pub layer_idx: usize,
        pub layer_type: crate::model::qwen35_config::LayerType,
        /// Mean cosine similarity of input vs output hidden state across calibration prompts.
        /// High cosine → small change → safer prune candidate.
        pub mean_cosine: f32,
        /// Importance proxy: `1.0 - mean_cosine`.
        pub importance: f32,
    }

    /// Pruning plan returned by [`MetalQwen35State::score_layer_importance`].
    #[derive(Debug, Clone)]
    pub struct LayerPruningPlan {
        /// Layers sorted by `mean_cosine` descending (highest cosine first = best prune candidate).
        pub scores: Vec<LayerImportanceScore>,
        /// Recommended per-layer mask (`false` = prune, `true` = keep).
        pub recommended_mask: Vec<bool>,
        /// Set when `prune_layers` exceeds 20 % of the total layer count.
        pub warning: Option<String>,
    }

    /// Numerically stable cosine similarity using f64 accumulators.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut aa = 0.0f64;
        let mut bb = 0.0f64;
        for (&x, &y) in a.iter().zip(b) {
            let x = x as f64;
            let y = y as f64;
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        if aa == 0.0 || bb == 0.0 {
            return 0.0;
        }
        (dot / (aa.sqrt() * bb.sqrt())) as f32
    }

    /// Compile one Stage-1 lm_head block-top-k pipeline variant with the
    /// HIDDEN / ROWS_PER_TG / LOCAL_K function constants baked in.
    fn make_lm_head_block_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
        hidden: u32,
        rows_per_tg: u32,
        local_k: u32,
    ) -> Result<ComputePipelineState, String> {
        let constants = FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            &hidden as *const u32 as *const std::ffi::c_void,
            MTLDataType::UInt,
            0,
        );
        constants.set_constant_value_at_index(
            &rows_per_tg as *const u32 as *const std::ffi::c_void,
            MTLDataType::UInt,
            1,
        );
        constants.set_constant_value_at_index(
            &local_k as *const u32 as *const std::ffi::c_void,
            MTLDataType::UInt,
            2,
        );
        let func = library.get_function(name, Some(constants)).map_err(|e| {
            format!(
                "function '{name}' (HIDDEN={hidden}, ROWS_PER_TG={rows_per_tg}, LOCAL_K={local_k}) not found: {e}"
            )
        })?;
        device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("pipeline for '{name}' LOCAL_K={local_k} failed: {e}"))
    }

    /// Compile all five LOCAL_K variants of a Stage-1 lm_head block-top-k kernel.
    fn make_lm_head_block_pipelines(
        device: &Device,
        library: &Library,
        name: &str,
        hidden: u32,
    ) -> Result<[ComputePipelineState; 5], String> {
        let mut out: Vec<ComputePipelineState> = Vec::with_capacity(LM_HEAD_LOCAL_KS.len());
        for &local_k in LM_HEAD_LOCAL_KS.iter() {
            out.push(make_lm_head_block_pipeline(
                device,
                library,
                name,
                hidden,
                LM_HEAD_ROWS_PER_TG,
                local_k,
            )?);
        }
        out.try_into().map_err(|_| {
            format!(
                "expected {} pipeline variants for '{name}'",
                LM_HEAD_LOCAL_KS.len()
            )
        })
    }

    fn build_pipelines(
        device: &Device,
        cfg: &Qwen35Config,
    ) -> Result<MetalQwen35Pipelines, String> {
        let prefill_grp_suffix =
            prefill_maxgrp_suffix(cfg.num_attention_heads / cfg.num_key_value_heads)?;

        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(MSL_SOURCE, &opts)
            .map_err(|e| format!("Metal shader compilation failed: {e}"))?;

        let make_pipeline = |name: &str| -> Result<ComputePipelineState, String> {
            let func = library
                .get_function(name, None)
                .map_err(|e| format!("function '{name}' not found: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("pipeline for '{name}' failed: {e}"))
        };

        let make_optional_gemm_q4_tiled = || -> Option<ComputePipelineState> {
            // simdgroup_float8x8 matrix ops are available since Apple7 (M1).
            // The compile/pipeline `.ok()?` chain below falls back to the
            // naive gemm_q4 on any device where the V3.0 source fails.
            if !device.supports_family(MTLGPUFamily::Apple7) {
                return None;
            }
            let tiled_opts = CompileOptions::new();
            tiled_opts.set_language_version(MTLLanguageVersion::V3_0);
            let lib = device
                .new_library_with_source(MSL_Q4_TILED_SOURCE, &tiled_opts)
                .ok()?;
            let func = lib.get_function("gemm_q4_tiled", None).ok()?;
            device.new_compute_pipeline_state_with_function(&func).ok()
        };

        let make_optional_gemm_q3_tiled = || -> Option<ComputePipelineState> {
            // Same Apple7 simdgroup-matrix gate as Q4/Q8; falls back to the
            // gemv_q3_decode-only (M=1) path on any device/compile failure.
            if !device.supports_family(MTLGPUFamily::Apple7) {
                return None;
            }
            let tiled_opts = CompileOptions::new();
            tiled_opts.set_language_version(MTLLanguageVersion::V3_0);
            let lib = device
                .new_library_with_source(MSL_Q3_TILED_SOURCE, &tiled_opts)
                .ok()?;
            let func = lib.get_function("gemm_q3_tiled", None).ok()?;
            device.new_compute_pipeline_state_with_function(&func).ok()
        };

        let make_optional_gemm_q8_tiled = || -> Option<ComputePipelineState> {
            // Q8_0 simdgroup-matrix tiled GEMM: same Apple7 gate as Q4.
            if !device.supports_family(MTLGPUFamily::Apple7) {
                return None;
            }
            let tiled_opts = CompileOptions::new();
            tiled_opts.set_language_version(MTLLanguageVersion::V3_0);
            let lib = device
                .new_library_with_source(MSL_Q8_TILED_SOURCE, &tiled_opts)
                .ok()?;
            let func = lib.get_function("gemm_q8_tiled", None).ok()?;
            device.new_compute_pipeline_state_with_function(&func).ok()
        };

        Ok(MetalQwen35Pipelines {
            gemv_decode: make_pipeline("gemv_decode_m1")?,
            gemv_decode_wide: make_pipeline("gemv_decode_wide_f16")?,
            gemv_q8: make_pipeline("gemv_q8_decode")?,
            gemv_q4: make_pipeline("gemv_q4_decode")?,
            gemm_q4: make_pipeline("gemm_q4")?,
            gemm_q4_tiled: make_optional_gemm_q4_tiled(),
            gemv_q3: make_pipeline("gemv_q3_decode")?,
            gemm_q3_tiled: make_optional_gemm_q3_tiled(),
            rms_norm: make_pipeline("rms_norm_qwen35")?,
            partial_rope: make_pipeline("partial_rope_interleaved")?,
            per_head_rms_norm: make_pipeline("per_head_rms_norm")?,
            decode_attention: make_pipeline("decode_attention")?,
            sigmoid_gate: make_pipeline("sigmoid_gate")?,
            scatter_q_gate: make_pipeline("scatter_q_gate")?,
            silu_mul: make_pipeline("silu_mul")?,
            silu_mul_fused: make_pipeline("silu_mul_fused")?,
            copy: make_pipeline("copy_buf")?,
            copy_offset: make_pipeline("copy_buf_offset")?,
            conv1d_silu: make_pipeline("conv1d_depthwise_silu")?,
            gdn_recurrence: make_pipeline("gdn_recurrence_fused")?,
            gdn_chunk_materialize_c32: make_pipeline("gdn_chunk_materialize_c32")?,
            gdn_chunk_solve_c32: make_pipeline("gdn_chunk_solve_c32")?,
            gdn_chunk_residual_output_c32: make_pipeline("gdn_chunk_residual_output_c32")?,
            gdn_chunk_state_update_c32: make_pipeline("gdn_chunk_state_update_c32")?,
            gdn_chunk_norm_silu_c32: make_pipeline("gdn_chunk_norm_silu_c32")?,
            gdn_chunk_conv_buf_update_c32: make_pipeline("gdn_chunk_conv_buf_update_c32")?,
            gdn_recurrence_q36: library
                .get_function("gdn_recurrence_fused_q36", None)
                .ok()
                .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
            gdn_precompute_keys: library
                .get_function("gdn_precompute_keys", None)
                .ok()
                .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
            gdn_recurrence_sharded: library
                .get_function("gdn_recurrence_sharded", None)
                .ok()
                .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
            gdn_norm_silu: library
                .get_function("gdn_norm_silu", None)
                .ok()
                .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
            fused_residual_add_norm: make_pipeline("fused_residual_add_norm")?,
            copy_and_rms_norm: make_pipeline("copy_and_rms_norm")?,
            add_and_copy: make_pipeline("add_and_copy")?,
            copy_and_rms_norm_batch: make_pipeline("copy_and_rms_norm_batch")?,
            fused_residual_add_norm_batch: make_pipeline("fused_residual_add_norm_batch")?,
            gemm_q8: make_pipeline("gemm_q8")?,
            gemm_q8_tiled: make_optional_gemm_q8_tiled(),
            topk_merge_pass: make_pipeline("logits_topk_merge_pass")?,
            argmax_first: make_pipeline("logits_argmax_first")?,
            argmax_merge: make_pipeline("logits_argmax_merge")?,
            topk_select50_first: make_pipeline("logits_topk_select50_first")?,
            topk_select50_merge: make_pipeline("logits_topk_select50_merge")?,
            decode_attn_partial: make_pipeline("decode_attention_flash_partial")?,
            decode_attn_reduce: make_pipeline("decode_attention_flash_reduce")?,
            lora_gemv_a: make_pipeline("lora_gemv_a")?,
            lora_gemv_b_accum: make_pipeline("lora_gemv_b_accum")?,
            lora_gemm_a_rows: make_pipeline("lora_gemm_a_rows")?,
            lora_gemm_b_accum_rows: make_pipeline("lora_gemm_b_accum_rows")?,
            // ADR-053: MoE Metal dispatch kernels
            moe_expert_gemv: make_pipeline("moe_expert_gemv")?,
            moe_scale_add: make_pipeline("moe_scale_add")?,
            moe_shared_gate_add: make_pipeline("moe_shared_gate_add")?,
            moe_zero_buf: make_pipeline("zero_buf")?,
            scatter_q_gate_batch: make_pipeline("scatter_q_gate_batch")?,
            per_head_rms_norm_batch: make_pipeline("per_head_rms_norm_batch")?,
            partial_rope_batch: make_pipeline("partial_rope_batch")?,
            copy_kv_cache_batch: make_pipeline("copy_kv_cache_batch")?,
            prefill_attention_batched: make_pipeline(&format!(
                "prefill_attention_batched_causal_{prefill_grp_suffix}"
            ))?,
            copy_offset_f16: make_pipeline("copy_buf_offset_f16")?,
            copy_kv_cache_batch_f16: make_pipeline("copy_kv_cache_batch_f16")?,
            decode_attention_f16: make_pipeline("decode_attention_f16")?,
            decode_attn_partial_f16: make_pipeline("decode_attention_flash_partial_f16")?,
            prefill_attention_batched_f16: make_pipeline(&format!(
                "prefill_attention_batched_causal_{prefill_grp_suffix}_f16"
            ))?,
            lm_head_block_topk_f16: make_lm_head_block_pipelines(
                device,
                &library,
                "lm_head_block_topk_f16",
                cfg.hidden_size as u32,
            )?,
            lm_head_block_topk_q4: make_lm_head_block_pipelines(
                device,
                &library,
                "lm_head_block_topk_q4",
                cfg.hidden_size as u32,
            )?,
        })
    }

    impl MetalQwen35Engine {
        /// Load immutable model resources from CPU weights. Does NOT allocate any session state.
        ///
        /// `max_cache_len` is NOT accepted here; pass it to `new_session` instead.
        pub fn new(weights: &ModelWeights, cfg: &Qwen35Config) -> Result<Self, String> {
            let device =
                Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;

            tracing::info!(
                name = device.name(),
                "Metal GPU initialized for Qwen3.5-2B forward pass (f16 weights)"
            );

            let queue = device.new_command_queue();

            let quant_format = QuantFormat::resolve(cfg);
            tracing::info!(
                ?quant_format,
                "Weight quantization format for Metal forward pass"
            );

            validate_flash_decode_shape(
                cfg.head_dim,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.num_attention_heads * cfg.head_dim,
                cfg.num_key_value_heads * cfg.head_dim,
            )?;

            let pipelines = build_pipelines(&device, cfg)?;

            // Upload per-layer weights
            let quant_tag = match quant_format {
                QuantFormat::Q8_0 => "q8",
                QuantFormat::Q4_0 => "q4",
            };
            let block_bytes = match quant_format {
                QuantFormat::Q8_0 => Q8_0_BLOCK_SIZE,
                QuantFormat::Q4_0 => Q4_0_BLOCK_SIZE,
            };
            let pack_quant = |data: &[f32]| -> Result<Vec<u8>, String> {
                match quant_format {
                    QuantFormat::Q8_0 => {
                        assert_eq!(data.len() % QK8_0, 0);
                        quantize_row_q8_0(data)
                    }
                    QuantFormat::Q4_0 => {
                        assert_eq!(
                            data.len() % QK4_0,
                            0,
                            "pack_quant: data length {} not divisible by {QK4_0}",
                            data.len()
                        );
                        quantize_row_q4_0(data).map_err(|e| e.to_string())
                    }
                }
            };
            let make_fused_quant_buffer =
                |device: &Device, gate_bytes: &[u8], up_bytes: &[u8], label: &str| -> Buffer {
                    let combined_size = gate_bytes.len() + up_bytes.len();
                    let buf = device
                        .new_buffer(combined_size as u64, MTLResourceOptions::StorageModeShared);
                    unsafe {
                        let dst = buf.contents() as *mut u8;
                        std::ptr::copy_nonoverlapping(gate_bytes.as_ptr(), dst, gate_bytes.len());
                        std::ptr::copy_nonoverlapping(
                            up_bytes.as_ptr(),
                            dst.add(gate_bytes.len()),
                            up_bytes.len(),
                        );
                    }
                    buf.set_label(label);
                    buf
                };
            let make_buffer_quant =
                |device: &Device, data: &[f32], label: &str| -> Result<Buffer, String> {
                    match quant_format {
                        QuantFormat::Q8_0 => make_buffer_q8_0(device, data, label),
                        QuantFormat::Q4_0 => make_buffer_q4_0(device, data, label),
                    }
                };

            let mut layer_weights = Vec::with_capacity(cfg.num_hidden_layers);
            for (i, (attn_w, common_w)) in weights.layers.iter().enumerate() {
                // Build the FFN weight variant for this layer (Dense or MoE).
                let metal_ffn = match &common_w.ffn {
                    crate::model::qwen35::FeedForwardWeights::Dense(d) => {
                        let gate_bytes = pack_quant(&d.gate_proj)?;
                        let up_bytes = pack_quant(&d.up_proj)?;
                        debug_assert_eq!(
                            d.gate_proj.len(),
                            cfg.intermediate_size * cfg.hidden_size
                        );
                        debug_assert_eq!(d.up_proj.len(), cfg.intermediate_size * cfg.hidden_size);
                        debug_assert_eq!(
                            gate_bytes.len(),
                            cfg.intermediate_size * (cfg.hidden_size / 32) * block_bytes
                        );
                        debug_assert_eq!(up_bytes.len(), gate_bytes.len());
                        MetalFfnWeights::Dense {
                            gate_up_proj: Q4WeightBuf::from_buffer(make_fused_quant_buffer(
                                &device,
                                &gate_bytes,
                                &up_bytes,
                                &format!("L{i}.gate_up.{quant_tag}"),
                            )),
                            down_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &d.down_proj,
                                &format!("L{i}.down.{quant_tag}"),
                            )?),
                        }
                    }
                    crate::model::qwen35::FeedForwardWeights::Moe(m) => {
                        let num_exp = m.experts.num_experts;
                        let inter = m.experts.intermediate_size;
                        let s_inter = m.shared_expert.intermediate_size;
                        let hid = m.experts.hidden_size;
                        let top_k = m.router.num_experts_per_tok;

                        // Validate memory headroom: routed experts alone must fit comfortably.
                        // Each f16 element is 2 bytes. Use 0.85 × recommendedMaxWorkingSetSize.
                        let routed_bytes = num_exp * (2 * inter * hid + hid * inter) * 2;
                        let max_working = device.recommended_max_working_set_size();
                        let threshold = (max_working as f64 * 0.85) as u64;
                        if (routed_bytes as u64) > threshold {
                            return Err(format!(
                                "MoE layer {i}: routed expert weights ({} MiB) exceed \
                                 0.85 × recommendedMaxWorkingSetSize ({} MiB). \
                                 Use a 64 GB+ configuration.",
                                routed_bytes / (1024 * 1024),
                                threshold / (1024 * 1024)
                            ));
                        }

                        // gate_up_proj: [num_experts, 2*inter, hidden] stored as f32 in CPU memory.
                        // Convert to f16 for GPU — halves bandwidth.
                        let routed_gate_up = make_buffer_f16(
                            &device,
                            &m.experts.gate_up_proj,
                            &format!("L{i}.moe.gate_up.f16"),
                        );
                        let routed_down = make_buffer_f16(
                            &device,
                            &m.experts.down_proj,
                            &format!("L{i}.moe.down.f16"),
                        );
                        let shared_gate_proj = make_buffer_f16(
                            &device,
                            &m.shared_expert.gate_proj,
                            &format!("L{i}.moe.sh_gate.f16"),
                        );
                        let shared_up_proj = make_buffer_f16(
                            &device,
                            &m.shared_expert.up_proj,
                            &format!("L{i}.moe.sh_up.f16"),
                        );
                        let shared_down_proj = make_buffer_f16(
                            &device,
                            &m.shared_expert.down_proj,
                            &format!("L{i}.moe.sh_down.f16"),
                        );
                        // Router gate kept as f32 — CPU reads this for routing.
                        let router_gate =
                            make_buffer(&device, &m.router.gate, &format!("L{i}.moe.router.f32"));
                        let shared_expert_gate = make_buffer(
                            &device,
                            &m.shared_expert.shared_expert_gate,
                            &format!("L{i}.moe.sh_gate_scalar.f32"),
                        );

                        // Scratch buffers reused per token.
                        let scratch_gate = make_zero_buffer(
                            &device,
                            s_inter.max(inter),
                            &format!("L{i}.moe.scr_gate"),
                        );
                        let scratch_up = make_zero_buffer(
                            &device,
                            s_inter.max(inter),
                            &format!("L{i}.moe.scr_up"),
                        );
                        let scratch_expert_out =
                            make_zero_buffer(&device, hid, &format!("L{i}.moe.scr_exp_out"));
                        let scratch_out =
                            make_zero_buffer(&device, hid, &format!("L{i}.moe.scr_out"));

                        MetalFfnWeights::Moe(Box::new(MoeMetalBuffers {
                            routed: RoutedExpertStorage::Eager {
                                gate_up: routed_gate_up,
                                down: routed_down,
                            },
                            shared_gate_proj,
                            shared_up_proj,
                            shared_down_proj,
                            router_gate,
                            shared_expert_gate,
                            scratch_gate,
                            scratch_up,
                            scratch_expert_out,
                            scratch_out,
                            num_experts: num_exp,
                            inter,
                            shared_inter: s_inter,
                            hidden: hid,
                            top_k,
                        }))
                    }
                };

                let common = MetalCommonLayerWeights {
                    // Norms stay f32 (small, precision-sensitive)
                    input_layernorm: make_buffer(
                        &device,
                        &common_w.input_layernorm,
                        &format!("L{i}.in_norm"),
                    ),
                    post_attention_layernorm: make_buffer(
                        &device,
                        &common_w.post_attention_layernorm,
                        &format!("L{i}.post_norm"),
                    ),
                    ffn: metal_ffn,
                };

                let attn = match attn_w {
                    AttentionWeights::Linear(gdn_w) => {
                        MetalLayerAttnWeights::Linear(MetalGdnLayerWeights {
                            // Large projections to f16
                            in_proj_qkv: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &gdn_w.in_proj_qkv,
                                &format!("L{i}.gdn.qkv.{quant_tag}"),
                            )?),
                            in_proj_z: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &gdn_w.in_proj_z,
                                &format!("L{i}.gdn.z.{quant_tag}"),
                            )?),
                            in_proj_qkvz: {
                                let mut combined = gdn_w.in_proj_qkv.clone();
                                combined.extend_from_slice(&gdn_w.in_proj_z);
                                Q4WeightBuf::from_buffer(make_buffer_quant(
                                    &device,
                                    &combined,
                                    &format!("L{i}.gdn.qkvz.{quant_tag}"),
                                )?)
                            },
                            // in_proj_b/a: keep f16 — CPU reads these for GDN recurrence
                            in_proj_b: make_buffer_f16(
                                &device,
                                &gdn_w.in_proj_b,
                                &format!("L{i}.gdn.b.f16"),
                            ),
                            in_proj_a: make_buffer_f16(
                                &device,
                                &gdn_w.in_proj_a,
                                &format!("L{i}.gdn.a.f16"),
                            ),
                            // Small scalars stay f32 (precision-sensitive, CPU-read)
                            a_log: make_buffer(&device, &gdn_w.a_log, &format!("L{i}.gdn.a_log")),
                            dt_bias: make_buffer(
                                &device,
                                &gdn_w.dt_bias,
                                &format!("L{i}.gdn.dt_bias"),
                            ),
                            // Conv1d and norm: f32 (small, read by CPU directly)
                            conv1d_weight: make_buffer(
                                &device,
                                &gdn_w.conv1d_weight,
                                &format!("L{i}.gdn.conv1d"),
                            ),
                            norm_weight: make_buffer(
                                &device,
                                &gdn_w.norm_weight,
                                &format!("L{i}.gdn.norm"),
                            ),
                            // Output projection: f16 (large)
                            out_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &gdn_w.out_proj,
                                &format!("L{i}.gdn.out.{quant_tag}"),
                            )?),
                        })
                    }
                    AttentionWeights::Full(full_w) => {
                        MetalLayerAttnWeights::Full(MetalFullLayerWeights {
                            // Large projections to f16
                            q_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &full_w.q_proj,
                                &format!("L{i}.full.q.{quant_tag}"),
                            )?),
                            k_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &full_w.k_proj,
                                &format!("L{i}.full.k.{quant_tag}"),
                            )?),
                            v_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &full_w.v_proj,
                                &format!("L{i}.full.v.{quant_tag}"),
                            )?),
                            o_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &full_w.o_proj,
                                &format!("L{i}.full.o.{quant_tag}"),
                            )?),
                            // Norm weights stay f32 (small, precision-sensitive)
                            q_norm: make_buffer(
                                &device,
                                &full_w.q_norm,
                                &format!("L{i}.full.q_norm"),
                            ),
                            k_norm: make_buffer(
                                &device,
                                &full_w.k_norm,
                                &format!("L{i}.full.k_norm"),
                            ),
                        })
                    }
                };

                layer_weights.push((attn, common));
            }

            // Embedding matrix: f16 (large, used for both embed and tied lm_head)
            // embed_tokens: keep f16 for embedding lookup (CPU reads it).
            // Add a Q8_0 version for the logits projection (GPU matmul).
            let embed_tokens = make_buffer_f16(&device, &weights.embed_tokens, "embed_tokens.f16");
            let embed_tokens_q8 = Q4WeightBuf::from_buffer(make_buffer_quant(
                &device,
                &weights.embed_tokens,
                &format!("embed_tokens.{quant_tag}"),
            )?);
            // Final norm: f32 (small, precision-sensitive)
            let final_norm = make_buffer(&device, &weights.final_norm, "final_norm");

            // Build RoPE tables for partial interleaved rotation (f32).
            // Engine does not know the session cache length; size to max_position_embeddings.
            let rope_dim = cfg.rope_dim();
            let rope_max = cfg.max_position_embeddings;
            let (cos_data, sin_data) = build_rope_interleaved(rope_dim, rope_max, cfg.rope_theta);
            let rope_cos = make_buffer(&device, &cos_data, "rope_cos");
            let rope_sin = make_buffer(&device, &sin_data, "rope_sin");

            let has_moe_layer = derive_has_moe_layer(&layer_weights);

            Ok(Self {
                device,
                queue,
                pipelines,
                layer_weights,
                embed_tokens,
                embed_tokens_q8,
                final_norm,
                rope_cos,
                rope_sin,
                config: cfg.clone(),
                quant_format,
                mtp_weights: None,
                quarot_rotation: None,
                base_quarot_seed: None,
                has_moe_layer,
            })
        }

        /// Allocate per-session mutable state for a single concurrent request.
        pub fn new_session(&self, max_cache_len: usize) -> Result<InferenceSession, String> {
            let use_kv_f16 = matches!(
                std::env::var("LATTICE_KV_F16").as_deref(),
                Ok("1") | Ok("true")
            );
            self.new_session_inner(max_cache_len, use_kv_f16)
        }

        fn new_session_inner(
            &self,
            max_cache_len: usize,
            use_kv_f16: bool,
        ) -> Result<InferenceSession, String> {
            if max_cache_len == 0 {
                return Err("new_session: max_cache_len must be > 0".into());
            }
            if max_cache_len > self.config.max_position_embeddings {
                return Err(format!(
                    "new_session: max_cache_len {max_cache_len} exceeds max_position_embeddings {}",
                    self.config.max_position_embeddings
                ));
            }
            let cfg = &self.config;
            let device = &self.device;
            let hidden = cfg.hidden_size;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let inter = cfg.intermediate_size;
            let max_prefill = max_cache_len.min(512);
            let bp = max_prefill;

            let activations = MetalQwen35Activations {
                hidden: make_zero_buffer(device, bp * hidden, "act_hidden"),
                residual: make_zero_buffer(device, bp * hidden, "act_residual"),
                attn_out: make_zero_buffer(device, bp * q_dim.max(hidden), "act_attn_out"),
                q: make_zero_buffer(device, bp * 2 * q_dim, "act_q_interleaved"),
                q_separated: make_zero_buffer(device, bp * q_dim, "act_q"),
                gate_z: make_zero_buffer(device, bp * q_dim, "act_gate_z"),
                k: make_zero_buffer(device, bp * kv_dim, "act_k"),
                v: make_zero_buffer(device, bp * kv_dim, "act_v"),
                gate: make_zero_buffer(device, bp * 2 * inter, "act_gate"),
                up: make_zero_buffer(device, bp * inter, "act_up"),
                ffn_out: make_zero_buffer(device, bp * hidden, "act_ffn_out"),
                gdn_qkv: make_zero_buffer(device, bp * qkv_dim, "act_gdn_qkv"),
                gdn_z: make_zero_buffer(device, bp * output_dim, "act_gdn_z"),
                gdn_qkvz: make_zero_buffer(device, qkv_dim + output_dim, "act_gdn_qkvz"),
                gdn_key_scratch: make_zero_buffer(
                    device,
                    cfg.linear_num_key_heads * (2 * cfg.linear_key_head_dim + 1)
                        + cfg.linear_num_value_heads() * 2,
                    "act_gdn_key_scratch",
                ),
                gdn_raw_out: make_zero_buffer(device, output_dim, "act_gdn_raw_out"),
                gdn_chunk: {
                    let num_chunks = bp.div_ceil(GDN_CHUNK_SIZE);
                    let num_vh = cfg.linear_num_value_heads();
                    let kd = cfg.linear_key_head_dim;
                    let vd = cfg.linear_value_head_dim;
                    let chunk_rows = num_chunks * num_vh * GDN_CHUNK_SIZE;
                    let c2 = num_chunks * num_vh * GDN_CHUNK_SIZE * GDN_CHUNK_SIZE;
                    GdnChunkScratch {
                        q: make_zero_buffer(device, chunk_rows * kd, "gdn_chunk_q"),
                        k: make_zero_buffer(device, chunk_rows * kd, "gdn_chunk_k"),
                        v: make_zero_buffer(device, chunk_rows * vd, "gdn_chunk_v"),
                        beta_log_alpha: make_zero_buffer(device, chunk_rows * 2, "gdn_chunk_bla"),
                        gamma: make_zero_buffer(device, chunk_rows, "gdn_chunk_gamma"),
                        gamma_end: make_zero_buffer(
                            device,
                            num_chunks * num_vh,
                            "gdn_chunk_gamma_end",
                        ),
                        kkt: make_zero_buffer(device, c2, "gdn_chunk_kkt"),
                        qk_l: make_zero_buffer(device, c2, "gdn_chunk_qkl"),
                        w: make_zero_buffer(device, chunk_rows * kd, "gdn_chunk_w"),
                        u: make_zero_buffer(device, chunk_rows * vd, "gdn_chunk_u"),
                        r: make_zero_buffer(device, chunk_rows * vd, "gdn_chunk_r"),
                        k_right: make_zero_buffer(device, chunk_rows * kd, "gdn_chunk_k_right"),
                        raw_out: make_zero_buffer(device, bp * output_dim, "gdn_chunk_raw_out"),
                    }
                },
                // This `[vocab_size]` buffer is allocated
                // once per session and reused for both the compact and exact
                // routes, so a compact-only generation still carries it. Issue
                // #171's threadgroup-reduction claim is about *dispatch*, not
                // this allocation: the compact greedy/top-k path (see
                // `use_compact` in `generate_streaming`/`generate_cache_aware`)
                // never issues the full-logit GEMV or reads/writes this buffer
                // per token, so no per-token full-logit materialization or
                // readback happens on that path. Making the allocation itself
                // lazy/optional is a separate, larger session-layout change
                // (this buffer backs the exact fallback used by grammar,
                // repetition penalty, and non-compact sampling) and is left as
                // follow-up rather than attempted under this round's scope.
                logits: make_zero_buffer(device, cfg.vocab_size, "act_logits"),
                topk_scratch_a: {
                    let groups = cfg.vocab_size.div_ceil(1024);
                    make_zero_byte_buffer(device, groups * 256 * 8, "topk_scratch_a")
                },
                topk_scratch_b: {
                    let groups = cfg.vocab_size.div_ceil(1024);
                    make_zero_byte_buffer(device, groups * 256 * 8, "topk_scratch_b")
                },
                attn_partials: {
                    let max_partitions = max_cache_len.div_ceil(1024);
                    let stride = cfg.head_dim + 2;
                    make_zero_buffer(
                        device,
                        max_partitions * cfg.num_attention_heads * stride,
                        "attn_partials",
                    )
                },
                pre_final_hidden: make_zero_buffer(device, hidden, "act_pre_final_hidden"),
                final_hidden: make_zero_buffer(device, hidden, "act_final_hidden"),
                verify_logits: make_zero_buffer(
                    device,
                    MTP_VERIFY_MAX_TOKENS * cfg.vocab_size,
                    "act_verify_logits",
                ),
            };

            let num_linear = cfg.num_linear_attention_layers();
            let num_full = cfg.num_full_attention_layers();
            let gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
                .map(|_| GatedDeltaNetState::new(cfg))
                .collect();
            let gdn_scratch = GatedDeltaNetFusedScratch::default();

            let qkv_dim2 = cfg.linear_qkv_dim();
            let gdn_gpu_state = MetalGdnState::new(device, cfg, num_linear);
            let gdn_gpu_conv_out = make_zero_buffer(device, qkv_dim2, "gdn_conv_out");
            let kv_cache = MetalKvCache::new(device, num_full, kv_dim, max_cache_len, use_kv_f16);

            // Allocate MTP session state if engine has MTP weights loaded.
            let mtp = self.mtp_weights.as_ref().map(|_| {
                let q_dim_mtp = cfg.full_q_dim();
                MetalMtpSession {
                    cache: MetalMtpCache {
                        k_buf: make_zero_buffer(device, max_cache_len * kv_dim, "mtp.kv_k"),
                        v_buf: make_zero_buffer(device, max_cache_len * kv_dim, "mtp.kv_v"),
                        seq_len: 0,
                        max_cache_len,
                    },
                    activations: MetalMtpActivations {
                        fused: make_zero_buffer(device, 2 * hidden, "mtp.act_fused"),
                        hidden: make_zero_buffer(device, hidden, "mtp.act_hidden"),
                        residual: make_zero_buffer(device, hidden, "mtp.act_residual"),
                        q: make_zero_buffer(device, 2 * q_dim_mtp, "mtp.act_q"),
                        q_separated: make_zero_buffer(device, q_dim_mtp, "mtp.act_q_sep"),
                        gate_z: make_zero_buffer(device, q_dim_mtp, "mtp.act_gate_z"),
                        k: make_zero_buffer(device, kv_dim, "mtp.act_k"),
                        v: make_zero_buffer(device, kv_dim, "mtp.act_v"),
                        attn_out: make_zero_buffer(
                            device,
                            q_dim_mtp.max(hidden),
                            "mtp.act_attn_out",
                        ),
                        gate: make_zero_buffer(device, inter, "mtp.act_gate"),
                        up: make_zero_buffer(device, inter, "mtp.act_up"),
                        ffn_out: make_zero_buffer(device, hidden, "mtp.act_ffn_out"),
                        logits: make_zero_buffer(device, cfg.vocab_size, "mtp.act_logits"),
                    },
                }
            });

            let need_checkpoints = mtp.is_some() || crate::env_switch_enabled("LATTICE_SELF_SPEC");
            // Self-spec verifies `[pending_token] ++ draft_tokens` = `1 + SELF_SPEC_MAX_DRAFT`
            // tokens through `verify_tokens_batched`, which checkpoints slot 0 (pre-verify)
            // plus one slot per processed token. Pool size = max_tokens + 1, so we need
            // `max_tokens >= 1 + SELF_SPEC_MAX_DRAFT` to avoid out-of-bounds slot indexing.
            let checkpoint_max_tokens = if mtp.is_some() {
                MTP_VERIFY_MAX_TOKENS.max(1 + SELF_SPEC_MAX_DRAFT)
            } else {
                1 + SELF_SPEC_MAX_DRAFT
            };
            let gdn_checkpoints = if need_checkpoints {
                Some(MetalGdnCheckpointPool::new(
                    device,
                    checkpoint_max_tokens,
                    &gdn_gpu_state,
                ))
            } else {
                None
            };

            Ok(InferenceSession {
                activations,
                gdn_states,
                gdn_scratch,
                gdn_gpu_state,
                gdn_gpu_conv_out,
                kv_cache,
                max_prefill,
                compact_topk: 0,
                compact_route: GpuTopkRoute::CpuFallback,
                compact_result: Vec::new(),
                mtp,
                mtp_active: false,
                speculative_route: None,
                gdn_checkpoints,
                last_pre_final_hidden: vec![0.0f32; hidden],
                last_hidden_cursor: None,
                mtp_prefill_hidden: Vec::new(),
                capture_final_hidden: false,
                final_hidden_captured: std::sync::atomic::AtomicBool::new(false),
                #[cfg(feature = "gdn-state-counters")]
                gdn_state_traffic: if crate::env_switch_enabled("LATTICE_GDN_STATE_COUNTERS") {
                    Some(GdnStateTrafficCounters::new(
                        GdnStateTrafficShape::try_from_config(cfg)
                            .map_err(|e| e.to_string())?
                            .with_allocated_layers(num_linear),
                    ))
                } else {
                    None
                },
            })
        }
    }

    /// Per-phase timing accumulator for `forward_step_inner`.
    #[derive(Default)]
    struct StepProfile {
        embedding_us: u128,
        projection_us: u128,
        gdn_recurrence_us: u128,
        gqa_attention_us: u128,
        mlp_us: u128,
        final_us: u128,
        // GPU-time fields populated by LATTICE_DECODE_PROFILE per-group command-buffer split.
        // True GPU execution time measured as Instant::now() around commit()+wait_until_completed()
        // for serialized per-group command buffers. Zero when LATTICE_DECODE_PROFILE is not set.
        gpu_gdn_us: u128,
        gpu_gqa_us: u128,
        gpu_lm_head_us: u128,
        // 4-category de-confounded fields: separates shared MLP from layer-specific mixer/attn.
        gpu_gdn_mixer_us: u128,
        gpu_gqa_attn_us: u128,
        gpu_mlp_us: u128,
    }

    #[cfg(test)]
    thread_local! {
        /// Test-only override for `MetalQwen35State::encode_moe_ffn`'s
        /// CPU-routed top-k expert selection. When set, the real
        /// logits/softmax computation still runs (harmless extra CPU work,
        /// compiled out entirely in non-test builds) but its result is
        /// discarded in favor of this list, so a test can drive a specific,
        /// deterministic sequence of `ExpertSlotCache::resolve` calls
        /// through the SAME real encoded command buffer `forward_step` uses
        /// in production — unlike a helper that calls `resolve()` directly
        /// outside any encoder, this exercises the actual within-token GPU
        /// dispatch ordering the eviction invariant depends on. Cleared
        /// with `None` between tests/tokens by the caller.
        static FORCED_MOE_EXPERTS_FOR_TEST: std::cell::RefCell<Option<Vec<usize>>> =
            const { std::cell::RefCell::new(None) };
    }

    #[cfg(test)]
    pub(crate) fn set_forced_moe_experts_for_test(experts: Option<Vec<usize>>) {
        FORCED_MOE_EXPERTS_FOR_TEST.with(|c| *c.borrow_mut() = experts);
    }

    #[cfg(test)]
    thread_local! {
        /// Test-only override for `MetalQwen35State::encode_moe_ffn`'s
        /// routed-expert prefetch fan-out (#682 Stage 2): `None` (the
        /// production default) prefetches via rayon; `Some(false)` forces
        /// the identical pre-assigned plan through a plain serial
        /// iterator instead, so a test can assert the two fan-out
        /// strategies produce byte-identical decode output from the same
        /// checkpoint and token sequence. `Some(true)` forces rayon
        /// explicitly (same as unset, provided for symmetry). Cleared
        /// with `None` between tests by the caller.
        static MOE_PREFETCH_PARALLEL_FOR_TEST: std::cell::RefCell<Option<bool>> =
            const { std::cell::RefCell::new(None) };
    }

    #[cfg(test)]
    pub(crate) fn set_moe_prefetch_parallel_for_test(parallel: Option<bool>) {
        MOE_PREFETCH_PARALLEL_FOR_TEST.with(|c| *c.borrow_mut() = parallel);
    }

    #[cfg(test)]
    thread_local! {
        /// Test-only ordering-proof hook, paired with
        /// `MOE_PREFETCH_STEP2_DONE_TX_FOR_TEST` below: when set, applies
        /// ONLY to the gate_up cache's dequant spawn for the next
        /// `encode_moe_ffn` call (down's is unaffected — proving overlap
        /// for one cache is sufficient evidence of the mechanism). Moved
        /// (via `.take()`) into `ExpertSlotCache::spawn_dequant` on the
        /// calling thread, same reasoning as the panic-injection hook
        /// above. Cleared to `None` after being taken.
        static MOE_PREFETCH_ORDERING_GATE_FOR_TEST:
            std::cell::RefCell<Option<crate::forward::moe_expert_cache::PrefetchOrderingGate>> =
            const { std::cell::RefCell::new(None) };
    }

    #[cfg(test)]
    pub(crate) fn set_moe_prefetch_ordering_gate_for_test(
        gate: Option<crate::forward::moe_expert_cache::PrefetchOrderingGate>,
    ) {
        MOE_PREFETCH_ORDERING_GATE_FOR_TEST.with(|c| *c.borrow_mut() = gate);
    }

    #[cfg(test)]
    thread_local! {
        /// Test-only: when set, `encode_moe_ffn` sends on it immediately
        /// after Step 2's shared-expert GEMVs are fully encoded (before
        /// joining the routed-expert dequant spawn(s)), so a test can
        /// prove Step 2 encoding completed while a still-gated dequant
        /// spawn is deliberately blocked on
        /// `MOE_PREFETCH_ORDERING_GATE_FOR_TEST`'s `release_rx`. Cleared
        /// to `None` after being taken.
        static MOE_PREFETCH_STEP2_DONE_TX_FOR_TEST:
            std::cell::RefCell<Option<std::sync::mpsc::Sender<()>>> =
            const { std::cell::RefCell::new(None) };
    }

    #[cfg(test)]
    pub(crate) fn set_moe_prefetch_step2_done_tx_for_test(tx: Option<std::sync::mpsc::Sender<()>>) {
        MOE_PREFETCH_STEP2_DONE_TX_FOR_TEST.with(|c| *c.borrow_mut() = tx);
    }

    /// One recorded top-k routing decision: which experts `encode_moe_ffn`'s
    /// CPU router selected for one token at one MoE layer, and their
    /// (post-renormalization) gate weights, in `selected_ids[i]` /
    /// `gate_weights[i]` correspondence.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct MoeRoutingTraceRecord {
        pub layer_idx: usize,
        pub token_idx: usize,
        pub selected_ids: Vec<usize>,
        pub gate_weights: Vec<f32>,
    }

    thread_local! {
        /// Optional per-token routing-divergence trace collector for
        /// `encode_moe_ffn`'s CPU top-k selection (#682 Stage 4 instrumentation
        /// seam). Same arm/drain-via-free-function shape as
        /// `FORCED_MOE_EXPERTS_FOR_TEST` above, but NOT `#[cfg(test)]`-gated:
        /// production harnesses (e.g. `eval_perplexity`) arm it outside of
        /// tests to record per-token top-k expert-id sets and gate weights,
        /// for computing set-agreement (Jaccard) between a quantized arm and
        /// a full-precision reference. `None` (the default) makes every
        /// `encode_moe_ffn` call a single branch-on-None with no allocation.
        static MOE_ROUTING_TRACE: std::cell::RefCell<Option<Vec<MoeRoutingTraceRecord>>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Arm the routing-trace collector on this thread: every subsequent
    /// `encode_moe_ffn` call appends one [`MoeRoutingTraceRecord`], until
    /// [`take_moe_routing_trace`] disarms it. Re-arming an already-armed
    /// collector discards any records buffered since the previous arm.
    pub fn arm_moe_routing_trace() {
        MOE_ROUTING_TRACE.with(|c| *c.borrow_mut() = Some(Vec::new()));
    }

    /// Disarm the collector and return every record buffered since the last
    /// [`arm_moe_routing_trace`] call (empty if it was never armed).
    pub fn take_moe_routing_trace() -> Vec<MoeRoutingTraceRecord> {
        MOE_ROUTING_TRACE.with(|c| c.borrow_mut().take().unwrap_or_default())
    }

    /// Write `records` as newline-delimited JSON, one [`MoeRoutingTraceRecord`]
    /// per line, so an external harness can diff a trace against a reference run.
    pub fn dump_moe_routing_trace_jsonl(
        records: &[MoeRoutingTraceRecord],
        path: &std::path::Path,
    ) -> std::io::Result<()> {
        use std::io::Write;
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        for record in records {
            serde_json::to_writer(&mut out, record).map_err(std::io::Error::other)?;
            out.write_all(b"\n")?;
        }
        out.flush()
    }

    impl MetalQwen35State {
        /// **Unstable**: create from CPU weights; weight layout and f16 conversion may change.
        pub fn new(
            weights: &ModelWeights,
            cfg: &Qwen35Config,
            max_cache_len: usize,
        ) -> Result<Self, String> {
            let engine = MetalQwen35Engine::new(weights, cfg)?;
            let use_gdn_chunked = !matches!(
                std::env::var("LATTICE_GDN_CHUNKED").as_deref(),
                Ok("0") | Ok("false")
            );
            let use_kv_f16 = matches!(
                std::env::var("LATTICE_KV_F16").as_deref(),
                Ok("1") | Ok("true")
            );
            let path_proof_enabled = matches!(
                std::env::var("LATTICE_METAL_PATH_PROOF").as_deref(),
                Ok("1") | Ok("true")
            );
            // new_session reads LATTICE_KV_F16 identically, so session.kv_cache
            // and the use_kv_f16 field below agree by construction.
            let session = engine.new_session(max_cache_len)?;
            Ok(Self {
                engine,
                session,
                lora: None,
                use_gdn_chunked,
                use_kv_f16,
                cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
                path_proof_enabled,
                path_proof: PathProofCounters::default(),
            })
        }

        /// Returns `true` if MTP weights were loaded and the session has an active MTP state.
        pub fn has_mtp(&self) -> bool {
            self.session.mtp.is_some()
        }

        /// Which target hidden state the MTP draft head receives, decided when the
        /// MTP weights loaded. `None` when no MTP weights are loaded.
        pub fn mtp_hidden_tap(&self) -> Option<MtpHiddenTap> {
            self.engine.mtp_weights.as_ref().map(|w| w.hidden_tap)
        }

        /// Whether the loaded checkpoint is rotated (QuaRot): the property
        /// `MtpHiddenTap::for_checkpoint` is keyed on. Unlike `mtp_hidden_tap`, which is
        /// `None` without MTP weights, and `engine.quarot_rotation`, which is only built when
        /// MTP weights are present, `base_quarot_seed` is set for every rotated checkpoint.
        /// `quarot_rotation` is consulted as well so a state that carries a rotation without
        /// a seed is never treated as unrotated.
        fn checkpoint_is_rotated(&self) -> bool {
            self.engine.base_quarot_seed.is_some() || self.engine.quarot_rotation.is_some()
        }

        /// Zeroes the Metal path-proof counters.
        ///
        /// Call before a one-shot `generate`/`generate_streaming` run so
        /// [`Self::path_proof_snapshot`], [`Self::hidden_readback_path_proof_snapshot`],
        /// and [`Self::logit_readback_path_proof_snapshot`] afterward reflect only
        /// that run's dispatches, explicit hidden readbacks, and logit readbacks.
        pub fn reset_path_proof_counters(&self) {
            self.path_proof.reset();
        }

        /// Snapshots the Metal attention/KV-cache path-proof counters (issue #239)
        /// without resetting them, alongside the KV-cache precision mode in effect.
        pub fn path_proof_snapshot(&self) -> PathProofSnapshot {
            PathProofSnapshot {
                prefill_kv_batch: self.path_proof.prefill_kv_batch.load(Ordering::Relaxed),
                prefill_attn_batched: self.path_proof.prefill_attn_batched.load(Ordering::Relaxed),
                decode_kv_copy: self.path_proof.decode_kv_copy.load(Ordering::Relaxed),
                decode_attn_direct: self.path_proof.decode_attn_direct.load(Ordering::Relaxed),
                decode_attn_split_partial: self
                    .path_proof
                    .decode_attn_split_partial
                    .load(Ordering::Relaxed),
                decode_attn_split_reduce: self
                    .path_proof
                    .decode_attn_split_reduce
                    .load(Ordering::Relaxed),
                kv_f16: self.use_kv_f16,
            }
        }

        /// Snapshots explicit hidden-readback path-proof counters without resetting them.
        pub fn hidden_readback_path_proof_snapshot(&self) -> HiddenReadbackPathProofSnapshot {
            HiddenReadbackPathProofSnapshot {
                decode: self
                    .path_proof
                    .decode_hidden_readback
                    .load(Ordering::Relaxed),
                prefill: self
                    .path_proof
                    .prefill_hidden_readback
                    .load(Ordering::Relaxed),
            }
        }

        /// Snapshots explicit vocabulary-logit-readback path-proof counters (full-vocab
        /// vs compact-candidate, decode and prefill) without resetting them. See
        /// [`LogitReadbackPathProofSnapshot`] for why this is a separate instrument from
        /// [`Self::hidden_readback_path_proof_snapshot`].
        pub fn logit_readback_path_proof_snapshot(&self) -> LogitReadbackPathProofSnapshot {
            LogitReadbackPathProofSnapshot {
                decode_full_vocab: self
                    .path_proof
                    .decode_full_vocab_logit_readback
                    .load(Ordering::Relaxed),
                decode_compact_candidate: self
                    .path_proof
                    .decode_compact_candidate_logit_readback
                    .load(Ordering::Relaxed),
                prefill_full_vocab: self
                    .path_proof
                    .prefill_full_vocab_logit_readback
                    .load(Ordering::Relaxed),
                prefill_compact_candidate: self
                    .path_proof
                    .prefill_compact_candidate_logit_readback
                    .load(Ordering::Relaxed),
                decode_full_vocab_bytes: self
                    .path_proof
                    .decode_full_vocab_logit_readback_bytes
                    .load(Ordering::Relaxed),
                decode_compact_candidate_bytes: self
                    .path_proof
                    .decode_compact_candidate_logit_readback_bytes
                    .load(Ordering::Relaxed),
                prefill_full_vocab_bytes: self
                    .path_proof
                    .prefill_full_vocab_logit_readback_bytes
                    .load(Ordering::Relaxed),
                prefill_compact_candidate_bytes: self
                    .path_proof
                    .prefill_compact_candidate_logit_readback_bytes
                    .load(Ordering::Relaxed),
            }
        }

        /// Resets GDN state-traffic counters to zero without changing the shape.
        ///
        /// No-op when counters are inactive (feature off, or
        /// `LATTICE_GDN_STATE_COUNTERS` unset at construction time).
        #[cfg(feature = "gdn-state-counters")]
        pub fn reset_gdn_state_traffic(&mut self) {
            if let Some(ref mut counters) = self.session.gdn_state_traffic {
                counters.reset();
            }
        }

        /// Returns the current GDN state-traffic report without resetting it.
        ///
        /// Returns `None` when counters are inactive.
        #[cfg(feature = "gdn-state-counters")]
        pub fn gdn_state_traffic_report(&self) -> Option<GdnStateTrafficReport> {
            self.session
                .gdn_state_traffic
                .as_ref()
                .map(GdnStateTrafficCounters::report)
        }

        /// Returns the current GDN state-traffic report and resets the counters,
        /// so callers can print one row per decode step without subtracting totals.
        ///
        /// Returns `None` when counters are inactive.
        #[cfg(feature = "gdn-state-counters")]
        pub fn take_gdn_state_traffic_report(&mut self) -> Option<GdnStateTrafficReport> {
            let report = self
                .session
                .gdn_state_traffic
                .as_ref()
                .map(GdnStateTrafficCounters::report);
            if let Some(ref mut counters) = self.session.gdn_state_traffic {
                counters.reset();
            }
            report
        }

        #[cfg(feature = "gdn-state-counters")]
        fn record_gdn_layer_access(&mut self, scope: GdnStateTrafficScope, layer_count: usize) {
            if let Some(ref mut counters) = self.session.gdn_state_traffic {
                counters.record_layer_access(scope, layer_count);
            }
        }

        #[cfg(feature = "gdn-state-counters")]
        fn record_gdn_copy(&mut self, scope: GdnStateTrafficScope, kind: GdnStateCopyKind) {
            if let Some(ref mut counters) = self.session.gdn_state_traffic {
                counters.record_copy(scope, kind);
            }
        }

        /// Returns the expected `(d_in, d_out)` for a LoRA projection given the layer type.
        ///
        /// Returns an error if `module` is not valid for the layer type at `layer_idx`,
        /// or if the module name is not recognised.
        fn expected_lora_shape(
            cfg: &crate::model::qwen35_config::Qwen35Config,
            layer_idx: usize,
            module: &str,
        ) -> Result<(usize, usize), crate::error::InferenceError> {
            use crate::error::InferenceError;

            // The refusal set is DERIVED from the servable set rather than spelled
            // again here. Written out, this list was a second copy of a fact the
            // trainers also hold, and two independently maintained copies of
            // "which GDN modules can be served" drift silently: the adapter
            // trains, the loss is minimised for a delta this kernel will never
            // apply, and nothing says so until load. `crates/inference/src/
            // lora_hook.rs` is the one copy; both ends read it.
            if crate::lora_hook::GDN_LORA_MODULES.contains(&module)
                && !crate::lora_hook::gdn_lora_module_is_servable(module)
            {
                if cfg.is_full_attention(layer_idx) {
                    return Err(InferenceError::Inference(format!(
                        "unknown LoRA module '{module}'"
                    )));
                }
                return Err(InferenceError::Inference(format!(
                    "module '{module}' is not yet supported in Metal forward (consumed inside \
                     fused GDN recurrence kernel); target other GDN modules instead"
                )));
            }

            let shape = crate::lora_hook::qwen35_projection_shape(cfg, layer_idx, module)
                .map_err(InferenceError::Inference)?;
            Ok((shape.d_in, shape.d_out))
        }

        /// Load a LoRA adapter onto the Metal GPU for inference.
        ///
        /// Resolves the QuaRot rotation seed to use before uploading anything: when
        /// `quarot_seed` is `None`, the base model's own rotation seed is used (so an
        /// adapter loaded onto a QuaRot-rotated base is rotated by default, rather than
        /// silently skipped); when `quarot_seed` is `Some`, it must agree with the
        /// base's seed if the base has one, and the base must actually be rotated if
        /// it does not. See `# Errors` below. Once resolved, rotation corrections are
        /// applied for a `Some` seed, then all A/B matrices are uploaded to Metal
        /// shared-memory buffers.
        ///
        /// Each `LoraLayerData` must name a valid Qwen3.5 projection module for its
        /// layer type, and its `(d_in, d_out)` must match the model's actual projection
        /// shape. Full-attention modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
        /// GDN modules: `in_proj_qkv`, `in_proj_z`, `out_proj`.
        /// (`in_proj_a`, `in_proj_b` not yet supported — consumed inside fused kernels).
        /// MLP modules (both layer types): `gate_proj`, `up_proj`, `down_proj`.
        ///
        /// This is the deliberate low-level entry point: `scale` is applied
        /// exactly as given, with no cross-check against any adapter
        /// identity (no descriptor, no per-layer rank comparison, nothing
        /// derived from `layers` itself). The caller owns the invariant that
        /// `scale` is the correct `alpha / rank` for the adapter these
        /// `layers` actually represent — this function has no way to verify
        /// that on its own. Prefer [`Self::load_lora_adapter_with_descriptor`]
        /// when `scale` should instead be derived from, and checked against,
        /// a [`lattice_fann::lora::LoraDescriptor`]'s declared rank and
        /// target modules.
        ///
        /// # Errors
        ///
        /// Returns an error if:
        /// - `layers` is empty
        /// - `scale` is not finite (NaN/inf)
        /// - An adapter is already loaded (call `unload_lora_adapter` first)
        /// - A module name is unknown or invalid for the layer type
        /// - `(d_in, d_out)` does not match the model projection shape
        /// - A/B vector lengths are inconsistent with rank × dimensions
        /// - `layer_idx` is out of range
        /// - Duplicate `(layer_idx, module)` entries exist
        /// - Any module is not in the rotation plan (when a seed resolves, from
        ///   `quarot_seed` or the base model)
        /// - Rank or dimensions overflow `u32`
        /// - `quarot_seed` is `Some` and disagrees with the base model's own QuaRot
        ///   rotation seed
        /// - `quarot_seed` is `Some` but the base model is not QuaRot-rotated
        pub fn load_lora_adapter(
            &mut self,
            mut layers: Vec<LoraLayerData>,
            scale: f32,
            quarot_seed: Option<u64>,
        ) -> Result<(), crate::error::InferenceError> {
            use crate::error::InferenceError;
            use crate::quant::quarot::lora::{LoraLayerMut, rotate_adapter_for_quarot};
            use crate::quant::quarot::plan::RotationPlan;

            if self.lora.is_some() {
                return Err(InferenceError::Inference(
                    "LoRA adapter already loaded; call unload_lora_adapter first".into(),
                ));
            }
            if layers.is_empty() {
                return Err(InferenceError::Inference(
                    "load_lora_adapter: layers must not be empty".into(),
                ));
            }
            if !scale.is_finite() {
                return Err(InferenceError::Inference(format!(
                    "load_lora_adapter: scale must be finite, got {scale}"
                )));
            }

            let hidden_dim = self.engine.config.hidden_size;

            // Resolve the seed to rotate with, rather than trusting `quarot_seed`
            // alone: an adapter that names no seed of its own is rotated with the
            // base's own seed when the base is QuaRot-rotated, instead of silently
            // loading unrotated onto a rotated base (see this function's doc comment).
            let resolved_seed = match (quarot_seed, self.engine.base_quarot_seed) {
                (None, base_seed) => base_seed,
                (Some(adapter_seed), None) => {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: adapter specifies QuaRot seed {adapter_seed}, but \
                         the base model has no QuaRot rotation seed (base is not rotated)"
                    )));
                }
                (Some(adapter_seed), Some(base_seed)) if adapter_seed != base_seed => {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: adapter QuaRot seed {adapter_seed} does not match \
                         base model QuaRot rotation seed {base_seed}"
                    )));
                }
                (Some(adapter_seed), Some(_)) => Some(adapter_seed),
            };

            if let Some(seed) = resolved_seed {
                let plan = RotationPlan::qwen35_residual_stream_linear_layers();
                let lora_muts: Vec<LoraLayerMut<'_>> = layers
                    .iter_mut()
                    .map(|l| LoraLayerMut {
                        layer_idx: l.layer_idx,
                        module: &l.module,
                        a: &mut l.a,
                        b: &mut l.b,
                        rank: l.rank,
                        d_in: l.d_in,
                        d_out: l.d_out,
                    })
                    .collect();
                rotate_adapter_for_quarot(lora_muts, seed, hidden_dim, &plan)?;
            }

            let device = &self.engine.device;
            let num_layers = self.engine.config.num_hidden_layers;
            let mut max_rank = 0u32;
            let mut projections: Vec<std::collections::HashMap<String, MetalLoraProjection>> = (0
                ..num_layers)
                .map(|_| std::collections::HashMap::new())
                .collect();

            for layer in &layers {
                if layer.rank == 0 {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' has rank=0",
                        layer.layer_idx, layer.module
                    )));
                }
                if layer.d_in == 0 || layer.d_out == 0 {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' has zero dimension \
                         (d_in={}, d_out={})",
                        layer.layer_idx, layer.module, layer.d_in, layer.d_out
                    )));
                }
                if layer.layer_idx >= num_layers {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer_idx {} >= num_hidden_layers {}",
                        layer.layer_idx, num_layers
                    )));
                }
                let (exp_d_in, exp_d_out) =
                    Self::expected_lora_shape(&self.engine.config, layer.layer_idx, &layer.module)?;
                if layer.d_in != exp_d_in || layer.d_out != exp_d_out {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' has (d_in={}, d_out={}) \
                         but model expects (d_in={}, d_out={})",
                        layer.layer_idx, layer.module, layer.d_in, layer.d_out, exp_d_in, exp_d_out
                    )));
                }
                let expected_a = layer.rank.checked_mul(layer.d_in).ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' rank*d_in overflow",
                        layer.layer_idx, layer.module
                    ))
                })?;
                let expected_b = layer.d_out.checked_mul(layer.rank).ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' d_out*rank overflow",
                        layer.layer_idx, layer.module
                    ))
                })?;
                if layer.a.len() != expected_a {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' A length {} != rank*d_in {}",
                        layer.layer_idx,
                        layer.module,
                        layer.a.len(),
                        expected_a
                    )));
                }
                if layer.b.len() != expected_b {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' B length {} != d_out*rank {}",
                        layer.layer_idx,
                        layer.module,
                        layer.b.len(),
                        expected_b
                    )));
                }
                let rank = u32::try_from(layer.rank).map_err(|_| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: rank {} exceeds u32",
                        layer.rank
                    ))
                })?;
                let d_in = u32::try_from(layer.d_in).map_err(|_| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: d_in {} exceeds u32",
                        layer.d_in
                    ))
                })?;
                let d_out = u32::try_from(layer.d_out).map_err(|_| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: d_out {} exceeds u32",
                        layer.d_out
                    ))
                })?;
                if rank > max_rank {
                    max_rank = rank;
                }
                let a_buf = device.new_buffer_with_data(
                    layer.a.as_ptr() as *const _,
                    (layer.a.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let b_buf = device.new_buffer_with_data(
                    layer.b.as_ptr() as *const _,
                    (layer.b.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                if projections[layer.layer_idx].contains_key(&layer.module) {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: duplicate projection for layer {} module '{}'",
                        layer.layer_idx, layer.module
                    )));
                }
                projections[layer.layer_idx].insert(
                    layer.module.clone(),
                    MetalLoraProjection {
                        a_buf,
                        b_buf,
                        rank,
                        d_in,
                        d_out,
                    },
                );
            }

            let max_rows = self.session.max_prefill.max(1);
            let intermediate_bytes = (max_rows as u64)
                .checked_mul(max_rank as u64)
                .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>() as u64))
                .ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: rank scratch size overflow \
                         (max_prefill={max_rows}, max_rank={max_rank})"
                    ))
                })?;
            let intermediate =
                device.new_buffer(intermediate_bytes, MTLResourceOptions::StorageModeShared);

            self.lora = Some(MetalLoraAdapter {
                projections,
                scale,
                intermediate,
                max_rank,
                max_rows,
            });
            // The cross-turn cache's adapter
            // identity (`cross_turn_metadata`'s `adapter_id`) is a shape-based
            // hash (max_rank/scale/projection count), not a content hash, so
            // two different adapters with the same shape can collide on the
            // same `AdapterId`. The metadata hash is
            // defense-in-depth; the unconditional clear here is the actual
            // correctness mechanism.
            self.cross_turn_prefix_cache.clear();

            Ok(())
        }

        /// Load a LoRA adapter using the shared cross-crate
        /// [`lattice_fann::lora::LoraDescriptor`] as the source of the adapter's
        /// scale and declared target modules, instead of a bare `scale: f32`
        /// the caller must have derived correctly on its own.
        ///
        /// Validates `descriptor.alpha`/effective scale (finite),
        /// `descriptor.target_modules` (recognized names), that every
        /// nonempty `layers` entry's rank matches `descriptor.rank`, and that
        /// the set of modules present in `layers` matches
        /// `descriptor.target_modules` exactly — before delegating to
        /// [`Self::load_lora_adapter`] with `descriptor.scale()`. Every
        /// per-layer, per-architecture shape check `load_lora_adapter` already
        /// performs still applies on top of this.
        ///
        /// `descriptor.rank` and `descriptor.target_modules` are the inputs
        /// `descriptor.scale()` is computed from; without this check a caller
        /// could pass `layers` built at one rank alongside a `descriptor` built
        /// at another (or covering a different module set) and silently get
        /// the wrong scale applied to correctly-shaped buffers.
        ///
        /// # Errors
        ///
        /// Returns an error if the descriptor's alpha/scale is not finite, if
        /// `target_modules` contains an unrecognized name, if any layer's
        /// rank disagrees with `descriptor.rank`, if the loaded module set
        /// disagrees with `descriptor.target_modules`, or for any reason
        /// [`Self::load_lora_adapter`] itself would reject the call.
        pub fn load_lora_adapter_with_descriptor(
            &mut self,
            layers: Vec<LoraLayerData>,
            descriptor: &lattice_fann::lora::LoraDescriptor,
            quarot_seed: Option<u64>,
        ) -> Result<(), crate::error::InferenceError> {
            use crate::error::InferenceError;

            descriptor
                .validate()
                .map_err(InferenceError::InvalidInput)?;
            lattice_fann::lora::validate_target_modules(
                &descriptor.target_modules,
                lattice_fann::lora::KNOWN_LORA_TARGET_MODULES,
            )
            .map_err(InferenceError::InvalidInput)?;

            for layer in &layers {
                if layer.rank != descriptor.rank {
                    return Err(InferenceError::InvalidInput(format!(
                        "load_lora_adapter_with_descriptor: layer {} module '{}' has \
                         rank={} but descriptor declares rank={}",
                        layer.layer_idx, layer.module, layer.rank, descriptor.rank
                    )));
                }
            }

            let mut layer_modules: Vec<&str> = layers.iter().map(|l| l.module.as_str()).collect();
            layer_modules.sort_unstable();
            layer_modules.dedup();
            let mut declared_modules: Vec<&str> = descriptor
                .target_modules
                .iter()
                .map(String::as_str)
                .collect();
            declared_modules.sort_unstable();
            declared_modules.dedup();
            if layer_modules != declared_modules {
                return Err(InferenceError::InvalidInput(format!(
                    "load_lora_adapter_with_descriptor: loaded layer modules {layer_modules:?} \
                     do not match descriptor.target_modules {declared_modules:?}"
                )));
            }

            self.load_lora_adapter(layers, descriptor.scale(), quarot_seed)
        }

        /// Unload the currently loaded LoRA adapter, freeing GPU buffers.
        pub fn unload_lora_adapter(&mut self) {
            self.lora = None;
            // See the matching comment in
            // `load_lora_adapter` — any adapter identity change must
            // invalidate the retained cross-turn entry.
            self.cross_turn_prefix_cache.clear();
        }

        /// Returns true if a LoRA adapter is currently loaded.
        pub fn has_lora_adapter(&self) -> bool {
            self.lora.is_some()
        }

        /// Blend multiple LoRA adapter layer sets into one rank-Σr layer set, then
        /// generate text through the existing single-adapter path.
        ///
        /// # Why blend instead of dispatching N adapters
        ///
        /// The Metal forward path holds exactly one `Option<MetalLoraAdapter>` slot.
        /// Rather than changing that kernel contract, we pre-blend on the CPU once
        /// at request start: the result is a valid single adapter loaded through the
        /// unchanged `load_lora_adapter` path.
        ///
        /// # Arguments
        ///
        /// * `adapter_weights` — slices of `(Vec<LoraLayerData>, mixture_weight)`.
        ///   Each element is one adapter's layer data paired with its effective weight
        ///   `w_e`.  `mixture_weight` IS the fully pre-scaled effective weight: the
        ///   caller must fold any `alpha / rank` scaling (and the mixture fraction)
        ///   into it before calling this function.  `blend_lora_layer_data` folds
        ///   `w_e` directly into the B column blocks and loads the result with
        ///   `scale = 1.0`; passing a raw mixture fraction without the `alpha/rank`
        ///   factor will produce under-scaled output.
        /// * `prompt` — prompt text, tokenised internally.
        /// * `tokenizer` — BPE tokeniser.
        /// * `gen_cfg` — generation configuration.
        ///
        /// The adapter is unloaded after generation so the slot is free for the next
        /// request.
        ///
        /// # Decode cost model
        ///
        /// Per-token LoRA GEMV cost grows linearly with `rank_total = Σ rank_e`
        /// across the mixture.  The Metal kernel assumes modest rank (≤ ~64);
        /// a large `k × r` budget will linearly increase decode latency.
        ///
        /// # Errors
        ///
        /// Returns an error if blending fails (e.g. dimension mismatch across
        /// adapters for the same projection), if adapter loading fails, or if the
        /// request fails the direct generator's preflight. Admission errors are
        /// returned before the existing adapter or prefix cache is changed.
        pub fn generate_with_lora_mixture(
            &mut self,
            adapter_weights: &[(&[LoraLayerData], f32)],
            prompt: &str,
            tokenizer: &crate::tokenizer::BpeTokenizer,
            gen_cfg: &crate::generation::GenerateConfig,
        ) -> Result<crate::generation::GenerateOutput, crate::error::InferenceError> {
            if adapter_weights.is_empty() {
                return Err(crate::error::InferenceError::InvalidInput(
                    "generate_with_lora_mixture: adapter_weights must not be empty".into(),
                ));
            }

            match self.prepare_direct_generation(prompt, tokenizer, gen_cfg)? {
                GenerationPreparation::Complete(output) => return Ok(output),
                GenerationPreparation::Ready(_) => {}
            }

            // Reject an empty inner layer slice here, before the adapter unload
            // below — `blend_lora_layer_data` also runs this check, but only
            // after the unload has already destroyed the previously loaded
            // adapter. Admission errors must be returned before any existing
            // adapter or prefix cache is changed (see this method's doc comment).
            reject_empty_inner_layers(adapter_weights)?;

            // Unload any previously loaded adapter so the slot is free.
            self.unload_lora_adapter();

            // Blend on the CPU. The per-adapter scale (alpha/rank) and mixture weight
            // are folded into the B column blocks; the resulting blended adapter uses
            // an effective scale of 1.0.
            let blended = blend_lora_layer_data(adapter_weights)?;

            // The blended adapter has weights-already-scaled; pass scale=1.0.
            self.load_lora_adapter(blended, 1.0, None)?;

            let output = self.generate(prompt, tokenizer, gen_cfg);

            // Always unload after the request so the slot does not leak into the
            // next call on this state object — regardless of whether generate()
            // succeeded or failed closed (e.g. an exhausted grammar mask, #611).
            self.unload_lora_adapter();

            output
        }

        // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
        // dispatch creates command buffers on the same long-lived worker thread.
        // Unmeasured individually; fixed on the structural argument, not on numbers.
        fn checkpoint_gdn_to_slot(
            &mut self,
            slot: usize,
            _traffic_scope: GdnStateTrafficScope,
        ) -> Result<(), crate::error::InferenceError> {
            objc::rc::autoreleasepool(|| self.checkpoint_gdn_to_slot_dispatch(slot, _traffic_scope))
        }

        /// Copy live GDN conv/S buffers into the given checkpoint slot (blocking GPU blit).
        fn checkpoint_gdn_to_slot_dispatch(
            &mut self,
            slot: usize,
            _traffic_scope: GdnStateTrafficScope,
        ) -> Result<(), crate::error::InferenceError> {
            let num_layers = self.session.gdn_gpu_state.len();
            let conv_sizes: Vec<u64> = self
                .session
                .gdn_gpu_state
                .layers()
                .map(|layer| layer.conv_buffer().length())
                .collect();
            let s_sizes: Vec<u64> = self
                .session
                .gdn_gpu_state
                .layers()
                .map(|layer| layer.s_matrix().length())
                .collect();
            // Resolve the checkpoint pool BEFORE allocating Metal command objects
            // so an uninitialized pool returns Err without leaking an un-ended
            // blit encoder or uncommitted command buffer on the error path.
            let pool = self.session.gdn_checkpoints.as_ref().ok_or_else(|| {
                crate::error::InferenceError::Inference(
                    "GDN checkpoint pool not initialized".into(),
                )
            })?;
            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_blit_command_encoder();
            for i in 0..num_layers {
                let layer = self.session.gdn_gpu_state.layer(i);
                enc.copy_from_buffer(
                    layer.conv_buffer(),
                    0,
                    &pool.conv_slots[slot][i],
                    0,
                    conv_sizes[i],
                );
                enc.copy_from_buffer(layer.s_matrix(), 0, &pool.s_slots[slot][i], 0, s_sizes[i]);
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            #[cfg(feature = "gdn-state-counters")]
            self.record_gdn_copy(_traffic_scope, GdnStateCopyKind::Snapshot);
            Ok(())
        }

        // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
        // dispatch creates command buffers on the same long-lived worker thread.
        // Unmeasured individually; fixed on the structural argument, not on numbers.
        fn restore_gdn_slot_blocking(
            &mut self,
            slot: usize,
            _traffic_scope: GdnStateTrafficScope,
        ) -> Result<(), crate::error::InferenceError> {
            objc::rc::autoreleasepool(|| {
                self.restore_gdn_slot_blocking_dispatch(slot, _traffic_scope)
            })
        }

        /// Restore live GDN conv/S buffers from the given checkpoint slot (blocking GPU blit).
        fn restore_gdn_slot_blocking_dispatch(
            &mut self,
            slot: usize,
            _traffic_scope: GdnStateTrafficScope,
        ) -> Result<(), crate::error::InferenceError> {
            let num_layers = self.session.gdn_gpu_state.len();
            let conv_sizes: Vec<u64> = self
                .session
                .gdn_gpu_state
                .layers()
                .map(|layer| layer.conv_buffer().length())
                .collect();
            let s_sizes: Vec<u64> = self
                .session
                .gdn_gpu_state
                .layers()
                .map(|layer| layer.s_matrix().length())
                .collect();
            // Resolve the checkpoint pool BEFORE allocating Metal command objects
            // so an uninitialized pool returns Err without leaking an un-ended
            // blit encoder or uncommitted command buffer on the error path.
            let pool = self.session.gdn_checkpoints.as_ref().ok_or_else(|| {
                crate::error::InferenceError::Inference(
                    "GDN checkpoint pool not initialized".into(),
                )
            })?;
            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_blit_command_encoder();
            for i in 0..num_layers {
                let layer = self.session.gdn_gpu_state.layer(i);
                enc.copy_from_buffer(
                    &pool.conv_slots[slot][i],
                    0,
                    layer.conv_buffer(),
                    0,
                    conv_sizes[i],
                );
                enc.copy_from_buffer(&pool.s_slots[slot][i], 0, layer.s_matrix(), 0, s_sizes[i]);
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            #[cfg(feature = "gdn-state-counters")]
            self.record_gdn_copy(_traffic_scope, GdnStateCopyKind::Restore);
            Ok(())
        }

        /// Roll back KV cache and GDN state to `seq_len` after a speculative rejection.
        ///
        /// Restores from slot `seq_len - active_base_seq_len` in the checkpoint pool.
        fn rollback_speculative_state_to(
            &mut self,
            seq_len: usize,
        ) -> Result<(), crate::error::InferenceError> {
            let base = self
                .session
                .gdn_checkpoints
                .as_ref()
                .and_then(|p| p.active_base_seq_len)
                .ok_or_else(|| {
                    crate::error::InferenceError::Inference(
                        "No active speculation checkpoint".into(),
                    )
                })?;
            let slot = seq_len.checked_sub(base).ok_or_else(|| {
                crate::error::InferenceError::Inference(format!(
                    "rollback seq_len {seq_len} < base {base}"
                ))
            })?;

            // Extract batch-repair info and MTP base before any mutable borrows.
            let batch_repair = self
                .session
                .gdn_checkpoints
                .as_mut()
                .and_then(|p| p.batch_repair_token.take());
            let mtp_base = self
                .session
                .gdn_checkpoints
                .as_ref()
                .and_then(|p| p.mtp_base_seq_len);

            if let Some((repair_token, repair_pos)) = batch_repair {
                // Batch verifier path: restore GDN base, then replay the first verify token so
                // the GDN state lands at S_{repair_pos} (= state after pending_token was processed).
                self.restore_gdn_slot_blocking(0, GdnStateTrafficScope::MtpVerify)?;
                // Set seq_len so forward_step_inner writes KV to the correct slot.
                self.session.rewind_position(repair_pos);
                // Repair step: reprocesses pending_token, advancing GDN and capturing hidden.
                #[cfg(feature = "gdn-state-counters")]
                let repair_out = self.forward_step_inner_with_traffic_scope(
                    repair_token,
                    repair_pos,
                    true,
                    GdnStateTrafficScope::MtpVerify,
                );
                #[cfg(not(feature = "gdn-state-counters"))]
                let repair_out = self.forward_step_inner(
                    repair_token,
                    repair_pos,
                    true,
                    crate::forward::signpost::Scope::NotDecode,
                );
                self.session.last_pre_final_hidden = repair_out.pre_final_hidden;
                self.session.mark_pre_final_hidden(repair_pos + 1);
                self.session.rewind_position(seq_len);
            } else {
                // Sequential verifier path: slot-based GDN restore.
                self.restore_gdn_slot_blocking(slot, GdnStateTrafficScope::MtpVerify)?;
                self.session.rewind_position(seq_len);
            }

            // Bug 3 fix: restore MTP KV cache position to prevent ghost entries.
            if let Some(base_mtp) = mtp_base
                && let Some(ref mut mtp) = self.session.mtp
            {
                mtp.cache.seq_len = base_mtp + slot;
            }
            Ok(())
        }

        /// Run the target model sequentially on `tokens` starting at `start_pos`,
        /// checkpointing GDN state after each token for speculative rollback.
        ///
        /// Returns per-token logits and the pre-final hidden state of the last token.
        /// This is the H1 sequential path (V_2≈2.0); a batch-GEMM verifier is a follow-up.
        fn verify_tokens_batched(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
            self.verify_tokens_batched_impl(tokens, start_pos, false)
        }

        /// `verify_tokens_batched`, optionally retaining every verified row's raw
        /// pre-final hidden in `MetalVerifyOutput::pre_final_hidden_rows`. The
        /// rows come from the same per-token forward that produced the logits.
        fn verify_tokens_batched_impl(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
            keep_all_rows: bool,
        ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
            self.check_forward_range_capacity(start_pos, tokens.len(), true)?;
            self.check_live_cursor("verify_tokens_batched", start_pos)?;
            if self.session.gdn_checkpoints.is_none() {
                return Err(crate::error::InferenceError::Inference(
                    "GDN checkpoint pool required for verify_tokens_batched".into(),
                ));
            }
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.active_base_seq_len = Some(start_pos);
                p.mtp_base_seq_len = self.session.mtp.as_ref().map(|m| m.cache.seq_len);
            }
            self.checkpoint_gdn_to_slot(0, GdnStateTrafficScope::MtpVerify)?;
            let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
            let mut final_hidden = Vec::new();
            let mut first_pre_final_hidden = Vec::new();
            let mut pre_final_hidden_rows: Vec<Vec<f32>> = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                #[cfg(feature = "gdn-state-counters")]
                let out = self.forward_step_inner_with_traffic_scope(
                    token,
                    start_pos + i,
                    true,
                    GdnStateTrafficScope::MtpVerify,
                );
                #[cfg(not(feature = "gdn-state-counters"))]
                let out = self.forward_step_inner(
                    token,
                    start_pos + i,
                    true,
                    crate::forward::signpost::Scope::NotDecode,
                );
                self.checkpoint_gdn_to_slot(i + 1, GdnStateTrafficScope::MtpVerify)?;
                all_logits.push(out.logits);
                if keep_all_rows {
                    pre_final_hidden_rows.push(out.pre_final_hidden.clone());
                }
                if i == 0 {
                    // lattice#1396: the pending token's own pre-final hidden,
                    // needed to append the accepted draft's MTP-cache row and
                    // to restore `last_pre_final_hidden` on a K=1 reject —
                    // captured here for free (this step already computes it)
                    // before the next iteration's `final_hidden` overwrite.
                    first_pre_final_hidden = out.pre_final_hidden.clone();
                }
                final_hidden = out.pre_final_hidden;
            }
            Ok(MetalVerifyOutput {
                logits: all_logits,
                final_hidden,
                first_pre_final_hidden,
                pre_final_hidden_rows,
            })
        }

        // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
        // dispatch creates command buffers on the same long-lived worker thread.
        // Unmeasured individually; fixed on the structural argument, not on numbers.
        fn verify_tokens_batch_gemm(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
            objc::rc::autoreleasepool(|| self.verify_tokens_batch_gemm_dispatch(tokens, start_pos))
        }

        /// Batch-GEMM verifier (activate with `LATTICE_MTP_BATCH=1`).
        ///
        /// Processes K verify tokens layer-by-layer using batch GEMM for all weight projections,
        /// with sequential GDN recurrence within each layer. Reduces weight-load cost vs the
        /// sequential verifier: each weight matrix is loaded once for K outputs instead of K times.
        ///
        /// **GDN rollback strategy**: Checkpoints GDN base (slot 0) before the batch. On rejection,
        /// `rollback_speculative_state_to` restores slot 0 and runs one repair forward step for the
        /// pending token to re-establish the correct intermediate GDN state.
        ///
        /// Expected V_2: ~1.35 at α=100%, ~1.60 at α=75%, ~2.30 at α=5%.
        fn verify_tokens_batch_gemm_dispatch(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
            self.check_live_cursor("verify_tokens_batch_gemm", start_pos)?;
            let n = tokens.len();
            if n == 0 || n > MTP_VERIFY_MAX_TOKENS {
                return Err(crate::error::InferenceError::Inference(format!(
                    "verify_batch: bad token count {n} (max {MTP_VERIFY_MAX_TOKENS})"
                )));
            }
            self.check_forward_range_capacity(start_pos, n, false)?;
            if self.session.gdn_checkpoints.is_none() {
                return Err(crate::error::InferenceError::Inference(
                    "GDN checkpoint pool required for verify_tokens_batch_gemm".into(),
                ));
            }

            let cfg = self.engine.config.clone();
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let kv_dim = cfg.full_kv_dim();
            let q_dim = cfg.full_q_dim();
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let head_dim = cfg.head_dim;
            let half_rope_dim = (cfg.rope_dim() / 2) as u32;
            let m = n as u32;

            // Checkpoint GDN base + record repair token for rollback.
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.active_base_seq_len = Some(start_pos);
                p.mtp_base_seq_len = self.session.mtp.as_ref().map(|mtp| mtp.cache.seq_len);
                p.batch_repair_token = if n > 0 {
                    Some((tokens[0], start_pos))
                } else {
                    None
                };
            }
            self.checkpoint_gdn_to_slot(0, GdnStateTrafficScope::MtpVerify)?;

            // Batch embedding: f16 → f32 for all K tokens into hidden[0..K*hidden].
            // SAFETY: embed_tokens is StorageModeShared f16, no GPU in flight;
            // each id is validated < vocab_size before computing offset.
            unsafe {
                let src_base = self.engine.embed_tokens.contents() as *const u16;
                let dst_base = self.session.activations.hidden.contents() as *mut f32;
                for (t, &id) in tokens.iter().enumerate() {
                    assert!(
                        (id as usize) < cfg.vocab_size,
                        "verify_batch: token {id} >= vocab_size {}",
                        cfg.vocab_size
                    );
                    let src = src_base.add(id as usize * hidden);
                    let dst = dst_base.add(t * hidden);
                    // SAFETY: embed_tokens row has hidden u16 values; dst row has hidden f32 values.
                    convert_f16_row(src, dst, hidden);
                }
            }

            let mut active_layer_idx = 0usize;
            let mut linear_idx = 0usize;
            let mut full_idx = 0usize;
            #[cfg(feature = "gdn-state-counters")]
            let mut gdn_verify_layer_dispatches = 0usize;

            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            for layer_i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(layer_i) {
                    continue;
                }
                let compact_idx = active_layer_idx;
                active_layer_idx += 1;
                let is_linear = matches!(
                    &self.engine.layer_weights[compact_idx].0,
                    MetalLayerAttnWeights::Linear(_)
                );
                let (_, common_w) = &self.engine.layer_weights[compact_idx];
                let w_in_norm = &common_w.input_layernorm as *const Buffer;
                let w_post_norm = &common_w.post_attention_layernorm as *const Buffer;
                let (w_gate_up, w_down, gate_byte_size) = match &common_w.ffn {
                    MetalFfnWeights::Dense {
                        gate_up_proj,
                        down_proj,
                    } => {
                        let byte_size: u64 = match self.engine.quant_format {
                            QuantFormat::Q8_0 => {
                                (inter * (hidden / QK8_0) * Q8_0_BLOCK_SIZE) as u64
                            }
                            QuantFormat::Q4_0 => {
                                (inter * (hidden / QK4_0) * Q4_0_BLOCK_SIZE) as u64
                            }
                        };
                        (
                            gate_up_proj as *const Q4WeightBuf,
                            down_proj as *const Q4WeightBuf,
                            byte_size,
                        )
                    }
                    MetalFfnWeights::Moe(_) => {
                        panic!(
                            "verify_tokens_batch_gemm: MoE layers are not supported in batch \
                                prefill (M>1). ADR-053 v1 targets decode-mode only."
                        );
                    }
                };

                if is_linear {
                    // GDN layer: batch projections + sequential recurrence (causal dependency).
                    let (w_qkv, w_z, w_b, w_a, w_alog, w_dtb, w_conv, w_norm, w_out) = {
                        let MetalLayerAttnWeights::Linear(gdn_w) =
                            &self.engine.layer_weights[compact_idx].0
                        else {
                            unreachable!()
                        };
                        (
                            &gdn_w.in_proj_qkv as *const Q4WeightBuf,
                            &gdn_w.in_proj_z as *const Q4WeightBuf,
                            &gdn_w.in_proj_b as *const Buffer,
                            &gdn_w.in_proj_a as *const Buffer,
                            &gdn_w.a_log as *const Buffer,
                            &gdn_w.dt_bias as *const Buffer,
                            &gdn_w.conv1d_weight as *const Buffer,
                            &gdn_w.norm_weight as *const Buffer,
                            &gdn_w.out_proj as *const Q4WeightBuf,
                        )
                    };
                    let qkv_d = cfg.linear_qkv_dim();
                    let out_d = cfg.linear_output_dim();
                    let num_h = cfg.linear_num_key_heads;
                    let key_d = cfg.linear_key_head_dim;
                    let val_d = cfg.linear_value_head_dim;
                    let ks = cfg.linear_conv_kernel_dim as u32;

                    // Batch: fused copy hidden → residual + norm hidden in-place.
                    // SAFETY: w_in_norm is a live layer-owned buffer pointer (dropped only when
                    // the model is destroyed). Activation buffers hidden/residual are sized for
                    // m * hidden f32 elements. The Metal command buffer enc is active.
                    unsafe {
                        self.dispatch_copy_and_rms_norm_batch(
                            enc,
                            &self.session.activations.hidden,
                            &self.session.activations.residual,
                            &*w_in_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                    }

                    // Batch: QKV + Z projections (GEMM, M=K — one weight load for K outputs).
                    // SAFETY: w_qkv, w_z are live layer-owned buffers.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_qkv,
                            &self.session.activations.gdn_qkv,
                            0,
                            m,
                            qkv_d as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_z,
                            &self.session.activations.gdn_z,
                            0,
                            m,
                            out_d as u32,
                            hidden as u32,
                        );
                    }

                    // Sequential: conv1d + recurrence per token (inherently causal).
                    // SAFETY: Token offsets are within m-sized activation buffers;
                    // conv/state/weight buffers are live for the command buffer.
                    for t in 0..n {
                        let qkv_off = (t * qkv_d) as u64 * 4;
                        let z_off = (t * out_d) as u64 * 4;
                        let h_off = (t * hidden) as u64 * 4;
                        unsafe {
                            enc.set_compute_pipeline_state(&self.engine.pipelines.conv1d_silu);
                            enc.set_buffer(
                                0,
                                Some(self.session.gdn_gpu_state.layer(linear_idx).conv_buffer()),
                                0,
                            );
                            enc.set_buffer(1, Some(&self.session.activations.gdn_qkv), qkv_off);
                            enc.set_buffer(2, Some(&*w_conv), 0);
                            enc.set_buffer(3, Some(&self.session.gdn_gpu_conv_out), 0);
                            let qd = qkv_d as u32;
                            enc.set_bytes(4, 4, &qd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &ks as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(qkv_d as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );

                            #[repr(C)]
                            struct GdnRecurParams {
                                key_dim: u32,
                                value_dim: u32,
                                num_key_heads: u32,
                                num_value_heads: u32,
                                hidden_size: u32,
                                q_total: u32,
                                v_offset: u32,
                                scale: f32,
                                eps: f32,
                            }
                            let num_vh = cfg.linear_num_value_heads();
                            let q_total = (num_h * key_d) as u32;
                            let params = GdnRecurParams {
                                key_dim: key_d as u32,
                                value_dim: val_d as u32,
                                num_key_heads: num_h as u32,
                                num_value_heads: num_vh as u32,
                                hidden_size: hidden as u32,
                                q_total,
                                v_offset: q_total * 2,
                                scale: 1.0 / (key_d as f32).sqrt(),
                                eps: cfg.rms_norm_eps,
                            };
                            enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_recurrence);
                            enc.set_buffer(
                                0,
                                Some(self.session.gdn_gpu_state.layer(linear_idx).s_matrix()),
                                0,
                            );
                            enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                            enc.set_buffer(2, Some(&self.session.activations.gdn_z), z_off);
                            enc.set_buffer(3, Some(&self.session.activations.hidden), h_off);
                            enc.set_buffer(4, Some(&*w_b), 0);
                            enc.set_buffer(5, Some(&*w_a), 0);
                            enc.set_buffer(6, Some(&*w_alog), 0);
                            enc.set_buffer(7, Some(&*w_dtb), 0);
                            enc.set_buffer(8, Some(&*w_norm), 0);
                            enc.set_buffer(9, Some(&self.session.activations.gdn_z), z_off);
                            enc.set_bytes(
                                10,
                                std::mem::size_of::<GdnRecurParams>() as u64,
                                &params as *const GdnRecurParams as *const _,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(num_vh as u64, 1, 1),
                                MTLSize::new(32, 4, 1),
                            );
                        }
                        #[cfg(feature = "gdn-state-counters")]
                        {
                            gdn_verify_layer_dispatches += 1;
                        }
                    }

                    // Batch: out_proj + residual + post-norm + MLP.
                    // SAFETY: w_out, w_gate_up, w_down are live layer-owned buffers.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gdn_z,
                            0,
                            &*w_out,
                            &self.session.activations.attn_out,
                            0,
                            m,
                            hidden as u32,
                            out_d as u32,
                        );
                    }
                    // Batch: fused residual-add + norm for MLP.
                    // SAFETY: w_post_norm is a live layer-owned buffer pointer.
                    unsafe {
                        self.dispatch_fused_residual_add_norm_batch(
                            enc,
                            &self.session.activations.residual,
                            &self.session.activations.attn_out,
                            &self.session.activations.residual,
                            &self.session.activations.hidden,
                            &*w_post_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                        self.dispatch_gemm_at(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_gate_up,
                            0,
                            &self.session.activations.gate,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm_at(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_gate_up,
                            gate_byte_size,
                            &self.session.activations.up,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                    }
                    self.dispatch_silu_mul(enc, m * inter as u32);
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gate,
                            0,
                            &*w_down,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            inter as u32,
                        );
                    }
                    // Batch: fused end-of-layer residual add + copy.
                    self.dispatch_add_and_copy(
                        enc,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );
                    linear_idx += 1;
                } else {
                    // Full-attention layer: batch QKV/O/MLP projections, sequential per-token ops.
                    let (w_q, w_k, w_v, w_o, w_qn, w_kn) = {
                        let MetalLayerAttnWeights::Full(full_w) =
                            &self.engine.layer_weights[compact_idx].0
                        else {
                            unreachable!()
                        };
                        (
                            &full_w.q_proj as *const Q4WeightBuf,
                            &full_w.k_proj as *const Q4WeightBuf,
                            &full_w.v_proj as *const Q4WeightBuf,
                            &full_w.o_proj as *const Q4WeightBuf,
                            &full_w.q_norm as *const Buffer,
                            &full_w.k_norm as *const Buffer,
                        )
                    };
                    let scale = 1.0f32 / (head_dim as f32).sqrt();

                    // Batch: fused copy hidden → residual + norm hidden in-place.
                    // SAFETY: w_in_norm is a live layer-owned buffer pointer.
                    unsafe {
                        self.dispatch_copy_and_rms_norm_batch(
                            enc,
                            &self.session.activations.hidden,
                            &self.session.activations.residual,
                            &*w_in_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                    }

                    // Batch: Q/K/V projections.
                    // SAFETY: w_q, w_k, w_v are live layer-owned buffers.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_q,
                            &self.session.activations.q,
                            0,
                            m,
                            (2 * q_dim) as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_k,
                            &self.session.activations.k,
                            0,
                            m,
                            kv_dim as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_v,
                            &self.session.activations.v,
                            0,
                            m,
                            kv_dim as u32,
                            hidden as u32,
                        );
                    }

                    // Sequential: scatter, norm, RoPE, KV store, attention, gate (causal).
                    // KV cache writes at absolute positions start_pos+t; attention sees cache[0..start_pos+t+1].
                    // SAFETY: Per-token offsets are within m-sized activation buffers;
                    // KV cache buffers are live StorageModeShared, sized for max_cache_len.
                    for t in 0..n {
                        let q_off = (t * 2 * q_dim) as u64 * 4;
                        let qs_off = (t * q_dim) as u64 * 4;
                        let gz_off = (t * q_dim) as u64 * 4;
                        let k_off = (t * kv_dim) as u64 * 4;
                        let v_off = (t * kv_dim) as u64 * 4;
                        let ao_off = (t * q_dim) as u64 * 4;
                        let abs_pos = start_pos + t;
                        let cache_len = (abs_pos + 1) as u32;
                        let kv_dst_off = (abs_pos * kv_dim) as u32;

                        unsafe {
                            // Scatter Q + gate from interleaved q_proj.
                            enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
                            enc.set_buffer(0, Some(&self.session.activations.q), q_off);
                            enc.set_buffer(1, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(2, Some(&self.session.activations.gate_z), gz_off);
                            let nh = num_q_heads as u32;
                            let hd = head_dim as u32;
                            enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );

                            // Per-head RMS norm Q and K.
                            enc.set_compute_pipeline_state(
                                &self.engine.pipelines.per_head_rms_norm,
                            );
                            enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(1, Some(&*w_qn), 0);
                            enc.set_bytes(2, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(3, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(4, 4, &cfg.rms_norm_eps as *const f32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(nh as u64, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                            enc.set_buffer(1, Some(&*w_kn), 0);
                            let nkh = num_kv_heads as u32;
                            enc.set_bytes(2, 4, &nkh as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(nkh as u64, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );

                            // Partial RoPE for Q and K at absolute position (start_pos + t).
                            let pos = abs_pos as u32;
                            enc.set_compute_pipeline_state(&self.engine.pipelines.partial_rope);
                            enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(1, Some(&self.engine.rope_cos), 0);
                            enc.set_buffer(2, Some(&self.engine.rope_sin), 0);
                            enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &half_rope_dim as *const u32 as *const _);
                            enc.set_bytes(6, 4, &pos as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                            enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                            let nkh = num_kv_heads as u32;
                            enc.set_bytes(3, 4, &nkh as *const u32 as *const _);
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                        }

                        // KV store at absolute position (start_pos + t).
                        if self.use_kv_f16 {
                            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset_f16);
                        } else {
                            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
                        }
                        enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                        enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                        let cnt = kv_dim as u32;
                        enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                        enc.set_bytes(3, 4, &kv_dst_off as *const u32 as *const _);
                        let wg = 256u64;
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );
                        enc.set_buffer(0, Some(&self.session.activations.v), v_off);
                        enc.set_buffer(1, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );

                        // Causal attention: Q[t] against full cache[0..start_pos+t+1].
                        if self.use_kv_f16 {
                            enc.set_compute_pipeline_state(
                                &self.engine.pipelines.decode_attention_f16,
                            );
                        } else {
                            enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
                        }
                        enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                        enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                        enc.set_buffer(2, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                        enc.set_buffer(3, Some(&self.session.activations.attn_out), ao_off);
                        enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                        let hd = head_dim as u32;
                        let nqh = num_q_heads as u32;
                        let nkh = num_kv_heads as u32;
                        let qd = q_dim as u32;
                        let kvd = kv_dim as u32;
                        enc.set_bytes(5, 4, &hd as *const u32 as *const _);
                        enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                        enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                        enc.set_bytes(8, 4, &qd as *const u32 as *const _);
                        enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
                        enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                        // Grid axis = num_kv_heads: the kernel maps gid → kvh and handles
                        // each GQA group of Q heads internally per threadgroup. Dispatching
                        // nqh wastes (nqh - nkh) threadgroups that the kernel's early-return
                        // guard discards; output is bit-identical but scheduling is nkh-sized.
                        enc.dispatch_thread_groups(
                            MTLSize::new(nkh as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );

                        // Sigmoid gate.
                        let cnt = q_dim as u32;
                        enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
                        enc.set_buffer(0, Some(&self.session.activations.attn_out), ao_off);
                        enc.set_buffer(1, Some(&self.session.activations.gate_z), gz_off);
                        enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                        let wg = 256u64;
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );
                    }

                    // Batch: O projection + residual + post-norm + MLP.
                    // SAFETY: w_o, w_gate_up, w_down are live layer-owned buffers.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.attn_out,
                            0,
                            &*w_o,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            q_dim as u32,
                        );
                    }
                    // Batch: fused residual-add + norm for MLP.
                    // SAFETY: w_post_norm is a live layer-owned buffer pointer.
                    unsafe {
                        self.dispatch_fused_residual_add_norm_batch(
                            enc,
                            &self.session.activations.residual,
                            &self.session.activations.ffn_out,
                            &self.session.activations.residual,
                            &self.session.activations.hidden,
                            &*w_post_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                        self.dispatch_gemm_at(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_gate_up,
                            0,
                            &self.session.activations.gate,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm_at(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_gate_up,
                            gate_byte_size,
                            &self.session.activations.up,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                    }
                    self.dispatch_silu_mul(enc, m * inter as u32);
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gate,
                            0,
                            &*w_down,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            inter as u32,
                        );
                    }
                    // Batch: fused end-of-layer residual add + copy.
                    self.dispatch_add_and_copy(
                        enc,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );
                    full_idx += 1;
                }
            } // end layer loop

            // Capture pre-final hidden for last token (for MTP next round).
            let last_off = ((n - 1) * hidden) as u64 * 4;
            if self.session.mtp.is_some() {
                enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
                enc.set_buffer(0, Some(&self.session.activations.hidden), last_off);
                enc.set_buffer(1, Some(&self.session.activations.pre_final_hidden), 0);
                let cnt = hidden as u32;
                let dst_off = 0u32;
                enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                enc.set_bytes(3, 4, &dst_off as *const u32 as *const _);
                let wg = 256u64;
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );
            }

            // Final RMSNorm for all K tokens (batch).
            self.dispatch_rms_norm(
                enc,
                &self.session.activations.hidden,
                &self.engine.final_norm,
                hidden as u32,
                m,
                cfg.rms_norm_eps,
            );
            // Site 1 of 3. `last_off` is the last of the K verified tokens.
            self.encode_capture_final_hidden(enc, last_off, hidden);

            // Logits GEMM for all K tokens → verify_logits[K * vocab_size].
            self.dispatch_gemm(
                enc,
                &self.session.activations.hidden,
                0,
                &self.engine.embed_tokens_q8,
                &self.session.activations.verify_logits,
                0,
                m,
                cfg.vocab_size as u32,
                hidden as u32,
            );

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            #[cfg(feature = "gdn-state-counters")]
            self.record_gdn_layer_access(
                GdnStateTrafficScope::MtpVerify,
                gdn_verify_layer_dispatches,
            );

            // Advance KV cache position.
            self.session.kv_cache.seq_len = start_pos + n;

            // Read back per-token logits from verify_logits buffer.
            let vocab = cfg.vocab_size;
            let flat = unsafe { read_buffer(&self.session.activations.verify_logits, n * vocab) };
            let all_logits: Vec<Vec<f32>> = (0..n)
                .map(|t| flat[t * vocab..(t + 1) * vocab].to_vec())
                .collect();

            // Read back and update pre-final hidden (on accept, used by next mtp_forward_one).
            let final_hidden = if self.session.mtp.is_some() {
                let h = unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
                self.session.last_pre_final_hidden = h.clone();
                self.session.mark_pre_final_hidden(start_pos + n);
                h
            } else {
                // No write: the cursor moved past the marker (or, with the verified range
                // starting below it, rewrote the history the marker described).
                if start_pos < self.session.last_hidden_cursor.unwrap_or(0) {
                    self.session.last_hidden_cursor = None;
                }
                Vec::new()
            };

            // lattice#1396: raw pre-final hidden of the FIRST verified row (the
            // pending token's own position), needed to append the accepted
            // draft's MTP-cache row. `activations.residual` holds this value
            // untouched: the layer loop's own end-of-layer fused add+copy
            // (`dispatch_add_and_copy`) leaves `residual[i] == hidden[i]` for
            // every one of the `n` verified rows right after the last layer,
            // and unlike `activations.hidden`, `residual` is never written
            // again by the batch RMSNorm dispatched above — that dispatch
            // only mutates `hidden` in place. No extra GPU dispatch is needed;
            // this is a second CPU-side read of an already-computed buffer,
            // the same pattern the MTP prefill capture uses on `hidden` itself.
            let first_pre_final_hidden = if self.session.mtp.is_some() {
                unsafe { read_buffer(&self.session.activations.residual, hidden) }
            } else {
                Vec::new()
            };

            Ok(MetalVerifyOutput {
                logits: all_logits,
                final_hidden,
                first_pre_final_hidden,
                pre_final_hidden_rows: Vec::new(),
            })
        }

        // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
        // dispatch creates command buffers on the same long-lived worker thread.
        // Unmeasured individually; fixed on the structural argument, not on numbers.
        fn mtp_forward_one(
            &mut self,
            pending_token: u32,
            position: usize,
        ) -> MetalMtpForwardOutput {
            objc::rc::autoreleasepool(|| self.mtp_forward_one_dispatch(pending_token, position))
        }

        /// RMSNorm for the MTP head's two CPU-computed pre-fc norms
        /// (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`). Applies this
        /// architecture's shifted convention, `x * inv_rms * (1.0 + gamma)`,
        /// matching every GPU-kernel RMSNorm in this forward pass and the
        /// model's own `final_norm`. Plain `gamma` (no `+1`) is a different,
        /// incompatible convention and silently inverts the sign of most
        /// channels on a checkpoint whose norm weights center negative.
        fn mtp_pre_fc_rmsnorm(x: &mut [f32], gamma: &[f32], eps: f32) {
            debug_assert_eq!(x.len(), gamma.len());
            let mut sum_sq = 0.0f32;
            for &v in x.iter() {
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
            for (v, &g) in x.iter_mut().zip(gamma.iter()) {
                *v = *v * inv_rms * (1.0 + g);
            }
        }

        /// Brings a captured (pre-final-norm) target hidden to the form the MTP
        /// head's hidden-state input expects, per [`MtpHiddenTap`]: under
        /// `PostFinalNorm` the target's final RMSNorm is applied here (CPU, same
        /// shifted convention and scale buffer the GPU final norm uses); under
        /// `PreFinalNorm` the hidden is left untouched. Both MTP entry points call
        /// this after the QuaRot counter-rotation and before `pre_fc_norm_hidden`.
        fn mtp_apply_hidden_tap(&self, hidden: &mut [f32]) {
            if self.mtp_hidden_tap() != Some(MtpHiddenTap::PostFinalNorm) {
                return;
            }
            self.apply_final_norm_cpu(hidden);
        }

        /// The target's final RMSNorm on the CPU: the shifted `x / rms(x) * (1 + gamma)`
        /// convention with the engine's final-norm scale, the same one the GPU final norm
        /// uses. It is only meaningful on a hidden in the original (unrotated) basis of a
        /// checkpoint whose final-norm scale is stored, i.e. not a rotated checkpoint. Shared
        /// by [`Self::mtp_apply_hidden_tap`] and the `MtpTargetVerifier` hidden-state methods.
        fn apply_final_norm_cpu(&self, hidden: &mut [f32]) {
            debug_assert!(self.engine.final_norm.length() as usize >= hidden.len() * 4);
            // SAFETY: `final_norm` is a StorageModeShared f32 buffer of `hidden_size`
            // values written at load; `hidden` has `hidden_size` values.
            let gamma = unsafe {
                std::slice::from_raw_parts(
                    self.engine.final_norm.contents() as *const f32,
                    hidden.len(),
                )
            };
            Self::mtp_pre_fc_rmsnorm(hidden, gamma, self.engine.config.rms_norm_eps);
        }

        /// Draft one extra token using the MTP module.
        ///
        /// Runs the single MTP attention+MLP layer on the target model's hidden
        /// state to predict `pending_token + 1`. The hidden state is captured before
        /// the target's final norm (`self.session.last_pre_final_hidden`); the head
        /// receives it after the final norm on unrotated checkpoints and before it
        /// on rotated ones (see [`MtpHiddenTap`], [`Self::mtp_apply_hidden_tap`]).
        /// Returns the draft token id and logits.
        ///
        /// Panics if `self.session.mtp` is None.
        fn mtp_forward_one_dispatch(
            &mut self,
            pending_token: u32,
            position: usize,
        ) -> MetalMtpForwardOutput {
            let cfg = self.engine.config.clone();
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let head_dim = cfg.head_dim;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let half_rope_dim = (cfg.rope_dim() / 2) as u32;
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            // ---- Phase 1: CPU ----

            // 1. Embedding lookup (f16 → f32), same token as was just generated.
            let mut normed_embed = vec![0.0f32; hidden];
            assert!((pending_token as usize) < cfg.vocab_size);
            unsafe {
                let src = (self.engine.embed_tokens.contents() as *const u16)
                    .add(pending_token as usize * hidden);
                // SAFETY: embed_tokens row has hidden u16 values; normed_embed has hidden f32 values.
                convert_f16_row(src, normed_embed.as_mut_ptr(), hidden);
            }

            if let Some(ref rot) = self.engine.quarot_rotation {
                // Rotation dim is built from `cfg.hidden_size` at session construction
                // (see `from_q4_dir` line ~10112), and `normed_embed` is `vec![0.0f32; hidden]`,
                // so dimensions cannot mismatch here. The `Result` API is preserved for
                // type compatibility; `debug_assert` catches programming errors in tests.
                debug_assert_eq!(rot.dim(), normed_embed.len());
                let _ = rot.apply_inverse(&mut normed_embed);
            }

            // 2. CPU RMSNorm of embedding using pre_fc_norm_embedding weights.
            {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let gamma = unsafe {
                    std::slice::from_raw_parts(
                        mtp_weights.pre_fc_norm_embedding.contents() as *const f32,
                        hidden,
                    )
                };
                Self::mtp_pre_fc_rmsnorm(&mut normed_embed, gamma, cfg.rms_norm_eps);
            }

            // 3. CPU RMSNorm of the target hidden using pre_fc_norm_hidden weights. The
            // captured hidden is pre-final-norm; `mtp_apply_hidden_tap` applies the
            // target's final norm first on unrotated checkpoints.
            let mut normed_hidden = self.session.last_pre_final_hidden.clone();
            if let Some(ref rot) = self.engine.quarot_rotation {
                // See note above: rotation dim == hidden_size by construction.
                debug_assert_eq!(rot.dim(), normed_hidden.len());
                let _ = rot.apply_inverse(&mut normed_hidden);
            }
            self.mtp_apply_hidden_tap(&mut normed_hidden);
            {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let gamma = unsafe {
                    std::slice::from_raw_parts(
                        mtp_weights.pre_fc_norm_hidden.contents() as *const f32,
                        hidden,
                    )
                };
                Self::mtp_pre_fc_rmsnorm(&mut normed_hidden, gamma, cfg.rms_norm_eps);
            }

            // 4. CPU concat [normed_embed || normed_hidden] into fused buffer (2*hidden).
            {
                let mtp = self.session.mtp.as_ref().unwrap();
                let dst = mtp.activations.fused.contents() as *mut f32;
                unsafe {
                    for (i, &v) in normed_embed.iter().enumerate() {
                        *dst.add(i) = v;
                    }
                    for (i, &v) in normed_hidden.iter().enumerate() {
                        *dst.add(hidden + i) = v;
                    }
                }
            }

            // ---- Phase 2: GPU ----
            // Collect raw pointers before encoder to avoid simultaneous borrow conflicts.
            let (
                w_fc,
                w_input_ln,
                w_post_attn_ln,
                w_q_proj,
                w_k_proj,
                w_v_proj,
                w_o_proj,
                w_q_norm,
                w_k_norm,
                w_gate_proj,
                w_up_proj,
                w_down_proj,
                w_norm,
            ) = {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let lw = &mtp_weights.layers[0];
                (
                    &mtp_weights.fc as *const Buffer,
                    &lw.input_layernorm as *const Buffer,
                    &lw.post_attention_layernorm as *const Buffer,
                    &lw.q_proj as *const Buffer,
                    &lw.k_proj as *const Buffer,
                    &lw.v_proj as *const Buffer,
                    &lw.o_proj as *const Buffer,
                    &lw.q_norm as *const Buffer,
                    &lw.k_norm as *const Buffer,
                    &lw.mlp.gate_proj as *const Buffer,
                    &lw.mlp.up_proj as *const Buffer,
                    &lw.mlp.down_proj as *const Buffer,
                    &mtp_weights.norm as *const Buffer,
                )
            };
            let (
                buf_fused,
                buf_hidden,
                buf_residual,
                buf_q,
                buf_q_sep,
                buf_gate_z,
                buf_k,
                buf_v,
                buf_attn_out,
                buf_gate,
                buf_up,
                buf_ffn_out,
                buf_logits,
            ) = {
                let a = &self.session.mtp.as_ref().unwrap().activations;
                (
                    &a.fused as *const Buffer,
                    &a.hidden as *const Buffer,
                    &a.residual as *const Buffer,
                    &a.q as *const Buffer,
                    &a.q_separated as *const Buffer,
                    &a.gate_z as *const Buffer,
                    &a.k as *const Buffer,
                    &a.v as *const Buffer,
                    &a.attn_out as *const Buffer,
                    &a.gate as *const Buffer,
                    &a.up as *const Buffer,
                    &a.ffn_out as *const Buffer,
                    &a.logits as *const Buffer,
                )
            };
            let (buf_k_cache, buf_v_cache, mtp_seq_len) = {
                let c = &self.session.mtp.as_ref().unwrap().cache;
                (
                    &c.k_buf as *const Buffer,
                    &c.v_buf as *const Buffer,
                    c.seq_len,
                )
            };

            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let wg = 256u64;

            unsafe {
                // fc: fused (2*hidden) → hidden
                // ADR-051 Phase 1: MTP projections are f16; `dispatch_matmul_half`
                // expects (a: f32 act, b: f16 weight, c: f32 out) with the M=1
                // decode GEMV pipeline (`gemv_decode_m1`).
                self.dispatch_matmul_half(
                    enc,
                    &*buf_fused,
                    &*w_fc,
                    &*buf_hidden,
                    1,
                    hidden as u32,
                    (2 * hidden) as u32,
                );

                // Pre-attention: copy hidden → residual, RMSNorm hidden in-place.
                self.dispatch_copy_and_rms_norm(
                    enc,
                    &*buf_hidden,
                    &*buf_residual,
                    &*w_input_ln,
                    hidden as u32,
                    cfg.rms_norm_eps,
                );

                // Q/K/V projections (f16 weights, Phase 1).
                self.dispatch_matmul_half(
                    enc,
                    &*buf_hidden,
                    &*w_q_proj,
                    &*buf_q,
                    1,
                    (2 * q_dim) as u32,
                    hidden as u32,
                );
                self.dispatch_matmul_half(
                    enc,
                    &*buf_hidden,
                    &*w_k_proj,
                    &*buf_k,
                    1,
                    kv_dim as u32,
                    hidden as u32,
                );
                self.dispatch_matmul_half(
                    enc,
                    &*buf_hidden,
                    &*w_v_proj,
                    &*buf_v,
                    1,
                    kv_dim as u32,
                    hidden as u32,
                );

                // Scatter Q+gate from q → q_separated, gate_z (inline).
                enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
                enc.set_buffer(0, Some(&*buf_q), 0);
                enc.set_buffer(1, Some(&*buf_q_sep), 0);
                enc.set_buffer(2, Some(&*buf_gate_z), 0);
                let nh = num_q_heads as u32;
                let hd = head_dim as u32;
                enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );

                // Per-head RMS norm on Q and K.
                self.dispatch_per_head_rms_norm(
                    enc,
                    &*buf_q_sep,
                    &*w_q_norm,
                    num_q_heads as u32,
                    hd,
                    cfg.rms_norm_eps,
                );
                self.dispatch_per_head_rms_norm(
                    enc,
                    &*buf_k,
                    &*w_k_norm,
                    num_kv_heads as u32,
                    hd,
                    cfg.rms_norm_eps,
                );

                // Partial RoPE. MTP speculative decoding is text-only in this
                // stage (ADR-069 RECON §5 open question); no M-RoPE override.
                self.dispatch_partial_rope(
                    enc,
                    &*buf_q_sep,
                    num_q_heads as u32,
                    hd,
                    half_rope_dim,
                    position as u32,
                    None,
                );
                self.dispatch_partial_rope(
                    enc,
                    &*buf_k,
                    num_kv_heads as u32,
                    hd,
                    half_rope_dim,
                    position as u32,
                    None,
                );

                // Append K,V to MTP KV cache.
                let kv_cache_off = (mtp_seq_len * kv_dim) as u32;
                self.dispatch_copy_offset(enc, &*buf_k, &*buf_k_cache, kv_dim as u32, kv_cache_off);
                self.dispatch_copy_offset(enc, &*buf_v, &*buf_v_cache, kv_dim as u32, kv_cache_off);

                // Flash decode attention (direct path — MTP cache is always short).
                let cur_mtp_seq = (mtp_seq_len + 1) as u32;
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
                enc.set_buffer(0, Some(&*buf_q_sep), 0);
                enc.set_buffer(1, Some(&*buf_k_cache), 0);
                enc.set_buffer(2, Some(&*buf_v_cache), 0);
                enc.set_buffer(3, Some(&*buf_attn_out), 0);
                enc.set_bytes(4, 4, &cur_mtp_seq as *const u32 as *const _);
                enc.set_bytes(5, 4, &hd as *const u32 as *const _);
                let nq = num_q_heads as u32;
                let nkv = num_kv_heads as u32;
                let qd = q_dim as u32;
                let kvd = kv_dim as u32;
                enc.set_bytes(6, 4, &nq as *const u32 as *const _);
                enc.set_bytes(7, 4, &nkv as *const u32 as *const _);
                enc.set_bytes(8, 4, &qd as *const u32 as *const _);
                enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
                enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_kv_heads as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );

                // Sigmoid gate: attn_out *= sigmoid(gate_z) (inline).
                enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
                enc.set_buffer(0, Some(&*buf_attn_out), 0);
                enc.set_buffer(1, Some(&*buf_gate_z), 0);
                let qd_u32 = q_dim as u32;
                enc.set_bytes(2, 4, &qd_u32 as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );

                // O projection: attn_out → ffn_out (f16 weights, Phase 1).
                self.dispatch_matmul_half(
                    enc,
                    &*buf_attn_out,
                    &*w_o_proj,
                    &*buf_ffn_out,
                    1,
                    hidden as u32,
                    q_dim as u32,
                );
                // Copy ffn_out → attn_out for use as delta in fused residual+norm.
                self.dispatch_copy(enc, &*buf_ffn_out, &*buf_attn_out, hidden as u32);

                // Post-attention residual + norm (MLP input).
                self.dispatch_fused_residual_add_norm(
                    enc,
                    &*buf_residual,
                    &*buf_attn_out,
                    &*buf_residual,
                    &*buf_hidden,
                    &*w_post_attn_ln,
                    hidden as u32,
                    cfg.rms_norm_eps,
                );

                // Dense MLP (f16 weights, Phase 1).
                self.dispatch_matmul_half(
                    enc,
                    &*buf_hidden,
                    &*w_gate_proj,
                    &*buf_gate,
                    1,
                    inter as u32,
                    hidden as u32,
                );
                self.dispatch_matmul_half(
                    enc,
                    &*buf_hidden,
                    &*w_up_proj,
                    &*buf_up,
                    1,
                    inter as u32,
                    hidden as u32,
                );
                // SiLU-mul (inline).
                enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul);
                enc.set_buffer(0, Some(&*buf_gate), 0);
                enc.set_buffer(1, Some(&*buf_up), 0);
                let inter_u32 = inter as u32;
                enc.set_bytes(2, 4, &inter_u32 as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(inter as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );
                self.dispatch_matmul_half(
                    enc,
                    &*buf_gate,
                    &*w_down_proj,
                    &*buf_ffn_out,
                    1,
                    hidden as u32,
                    inter as u32,
                );

                // MLP residual add.
                self.dispatch_add_and_copy(
                    enc,
                    &*buf_ffn_out,
                    &*buf_residual,
                    &*buf_hidden,
                    hidden as u32,
                );

                // Final RMSNorm.
                self.dispatch_rms_norm(
                    enc,
                    &*buf_hidden,
                    &*w_norm,
                    hidden as u32,
                    1,
                    cfg.rms_norm_eps,
                );
            }

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            if let Some(ref rot) = self.engine.quarot_rotation {
                // SAFETY: `buf_hidden` is the MTP `hidden` activation buffer, allocated
                // with capacity `hidden` f32 elements at session construction. The
                // command buffer that wrote it (`cmd`) has been awaited above via
                // `wait_until_completed()`, so the contents are visible to the host.
                let mut h_out = unsafe { read_buffer(&*buf_hidden, hidden) };
                debug_assert_eq!(rot.dim(), h_out.len());
                let _ = rot.apply(&mut h_out);
                // SAFETY: same buffer, no other command buffer in flight. We write
                // back the rotated values before the next command buffer (`cmd2`) is
                // submitted on the next line.
                unsafe { write_buffer(&*buf_hidden, &h_out) };
            }

            let cmd2 = self.engine.queue.new_command_buffer();
            let enc2 = cmd2.new_compute_command_encoder();
            unsafe {
                // Logits GEMV using shared lm_head (embed_tokens_q8).
                self.dispatch_gemm(
                    enc2,
                    &*buf_hidden,
                    0,
                    &self.engine.embed_tokens_q8,
                    &*buf_logits,
                    0,
                    1,
                    cfg.vocab_size as u32,
                    hidden as u32,
                );
            }
            enc2.end_encoding();
            cmd2.commit();
            cmd2.wait_until_completed();

            // ---- Phase 3: CPU ----
            // PR #787: this was the one surviving
            // production last-wins `max_by` argmax after the #280 fix -- the
            // MTP draft-token pick, called from `generate_greedy_mtp`. Swapped
            // for the shared first-wins `argmax_f32_first_wins` helper so a
            // tie at the draft boundary resolves the same way as every other
            // greedy pick in the crate (a differing draft tie can only reduce
            // speculative acceptance, never change final target output, but
            // the duplicate last-wins contract itself is the defect).
            let logits = unsafe { read_buffer(&*buf_logits, cfg.vocab_size) };
            let draft_token = crate::sampling::argmax_f32_first_wins(&logits);

            // Advance MTP KV cache.
            if let Some(ref mut mtp) = self.session.mtp {
                mtp.cache.seq_len += 1;
            }

            MetalMtpForwardOutput {
                token_id: draft_token,
                logits,
            }
        }

        /// Rebuilds the MTP draft head's KV cache from the prompt just processed by
        /// `forward_prefill`/`forward_prefill_all_logits` (#1340).
        ///
        /// Without this, `reset_state` zeroes the MTP cache and the only prefill on
        /// the Metal path fills the *target* model's cache, so the first
        /// `mtp_forward_one` call of a generation dispatches attention with
        /// `cache_len = 1` — the draft can only ever attend to the row it just
        /// wrote, independent of the prompt.
        ///
        /// `mtp_forward_one(pending_token, pos)` pairs `pending_token`'s embedding
        /// with `last_pre_final_hidden` — the target's hidden state, captured before
        /// the final norm, from the position *before* `pending_token`, i.e. the
        /// hidden state that predicted it. This mirrors that pairing across the whole
        /// prompt: for each position `p` in `1..prompt_ids.len()`, it pairs
        /// `prompt_ids[p]` with the target's captured hidden state at position
        /// `p - 1`, stored in `self.session.mtp_prefill_hidden` by
        /// `forward_prefill_batched_chunk`. Each append then brings the captured
        /// hidden to the head's input per [`MtpHiddenTap`], exactly as the live
        /// draft does.
        /// Position 0 has no predecessor hidden state (nothing came before the first
        /// prompt token) and is intentionally not represented in the MTP cache, so a
        /// prompt of length `L` yields `L - 1` prefilled entries; the live call for
        /// the first generated token then appends the `L`-th, giving that call's
        /// attention dispatch `cache_len == L` instead of `1`.
        ///
        /// No-op when the session has no MTP head, when the prompt is too short to
        /// yield a pair, or when `mtp_prefill_hidden` was not populated for this
        /// exact prompt (e.g. the LoRA-active prefill path, which forwards token by
        /// token via `forward_step` and does not feed the accumulator) — captured
        /// length is asserted against `prompt_ids.len() * hidden_size` and anything
        /// else is treated as "nothing to prefill" rather than misinterpreted.
        fn mtp_prefill(&mut self, prompt_ids: &[u32]) {
            let captured = std::mem::take(&mut self.session.mtp_prefill_hidden);
            if self.session.mtp.is_none() {
                return;
            }
            let hidden = self.engine.config.hidden_size;
            let n = prompt_ids.len();
            if n < 2 || captured.len() != n * hidden {
                return;
            }
            for p in 1..n {
                let hidden_in = &captured[(p - 1) * hidden..p * hidden];
                self.mtp_prefill_append(prompt_ids[p], hidden_in, p);
            }
        }

        // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
        // dispatch creates command buffers on the same long-lived worker thread.
        // Unmeasured individually; fixed on the structural argument, not on numbers.
        fn mtp_prefill_append(&mut self, token_id: u32, hidden_in: &[f32], position: usize) {
            objc::rc::autoreleasepool(|| {
                self.mtp_prefill_append_dispatch(token_id, hidden_in, position)
            })
        }

        /// Appends one MTP cache entry (K/V only — no Q, attention, MLP, or logits)
        /// for `token_id` at absolute position `position`, using `hidden_in` (the
        /// target's hidden state captured before the final norm) in place of
        /// `self.session.last_pre_final_hidden`; the head's hidden-state input is
        /// derived from it per [`MtpHiddenTap`]. Mirrors the CPU phase and GPU
        /// dispatch sequence of `mtp_forward_one` up through the KV cache append —
        /// everything after that point (attention, gating, MLP, logits) only matters
        /// for a position acting as a *query*, and a prefilled position is only ever
        /// a future *key*.
        fn mtp_prefill_append_dispatch(
            &mut self,
            token_id: u32,
            hidden_in: &[f32],
            position: usize,
        ) {
            let cfg = self.engine.config.clone();
            let hidden = cfg.hidden_size;
            let kv_dim = cfg.full_kv_dim();
            let num_kv_heads = cfg.num_key_value_heads;
            let head_dim = cfg.head_dim as u32;
            let half_rope_dim = (cfg.rope_dim() / 2) as u32;

            // ---- Phase 1: CPU (mirrors mtp_forward_one's embedding + hidden norm + fuse) ----
            let mut normed_embed = vec![0.0f32; hidden];
            assert!((token_id as usize) < cfg.vocab_size);
            unsafe {
                let src = (self.engine.embed_tokens.contents() as *const u16)
                    .add(token_id as usize * hidden);
                // SAFETY: embed_tokens row has hidden u16 values; normed_embed has hidden f32 values.
                convert_f16_row(src, normed_embed.as_mut_ptr(), hidden);
            }
            if let Some(ref rot) = self.engine.quarot_rotation {
                debug_assert_eq!(rot.dim(), normed_embed.len());
                let _ = rot.apply_inverse(&mut normed_embed);
            }
            {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let gamma = unsafe {
                    std::slice::from_raw_parts(
                        mtp_weights.pre_fc_norm_embedding.contents() as *const f32,
                        hidden,
                    )
                };
                Self::mtp_pre_fc_rmsnorm(&mut normed_embed, gamma, cfg.rms_norm_eps);
            }

            let mut normed_hidden = hidden_in.to_vec();
            if let Some(ref rot) = self.engine.quarot_rotation {
                debug_assert_eq!(rot.dim(), normed_hidden.len());
                let _ = rot.apply_inverse(&mut normed_hidden);
            }
            self.mtp_apply_hidden_tap(&mut normed_hidden);
            {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let gamma = unsafe {
                    std::slice::from_raw_parts(
                        mtp_weights.pre_fc_norm_hidden.contents() as *const f32,
                        hidden,
                    )
                };
                Self::mtp_pre_fc_rmsnorm(&mut normed_hidden, gamma, cfg.rms_norm_eps);
            }

            {
                let mtp = self.session.mtp.as_ref().unwrap();
                let dst = mtp.activations.fused.contents() as *mut f32;
                unsafe {
                    for (i, &v) in normed_embed.iter().enumerate() {
                        *dst.add(i) = v;
                    }
                    for (i, &v) in normed_hidden.iter().enumerate() {
                        *dst.add(hidden + i) = v;
                    }
                }
            }

            // ---- Phase 2: GPU (K/V-only: fc, input-norm, k/v proj, k-norm, rope, cache append) ----
            let (w_fc, w_input_ln, w_k_proj, w_v_proj, w_k_norm) = {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let lw = &mtp_weights.layers[0];
                (
                    &mtp_weights.fc as *const Buffer,
                    &lw.input_layernorm as *const Buffer,
                    &lw.k_proj as *const Buffer,
                    &lw.v_proj as *const Buffer,
                    &lw.k_norm as *const Buffer,
                )
            };
            let (buf_fused, buf_hidden, buf_residual, buf_k, buf_v) = {
                let a = &self.session.mtp.as_ref().unwrap().activations;
                (
                    &a.fused as *const Buffer,
                    &a.hidden as *const Buffer,
                    &a.residual as *const Buffer,
                    &a.k as *const Buffer,
                    &a.v as *const Buffer,
                )
            };
            let (buf_k_cache, buf_v_cache, mtp_seq_len) = {
                let c = &self.session.mtp.as_ref().unwrap().cache;
                (
                    &c.k_buf as *const Buffer,
                    &c.v_buf as *const Buffer,
                    c.seq_len,
                )
            };

            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            unsafe {
                self.dispatch_matmul_half(
                    enc,
                    &*buf_fused,
                    &*w_fc,
                    &*buf_hidden,
                    1,
                    hidden as u32,
                    (2 * hidden) as u32,
                );
                self.dispatch_copy_and_rms_norm(
                    enc,
                    &*buf_hidden,
                    &*buf_residual,
                    &*w_input_ln,
                    hidden as u32,
                    cfg.rms_norm_eps,
                );
                self.dispatch_matmul_half(
                    enc,
                    &*buf_hidden,
                    &*w_k_proj,
                    &*buf_k,
                    1,
                    kv_dim as u32,
                    hidden as u32,
                );
                self.dispatch_matmul_half(
                    enc,
                    &*buf_hidden,
                    &*w_v_proj,
                    &*buf_v,
                    1,
                    kv_dim as u32,
                    hidden as u32,
                );
                self.dispatch_per_head_rms_norm(
                    enc,
                    &*buf_k,
                    &*w_k_norm,
                    num_kv_heads as u32,
                    head_dim,
                    cfg.rms_norm_eps,
                );
                self.dispatch_partial_rope(
                    enc,
                    &*buf_k,
                    num_kv_heads as u32,
                    head_dim,
                    half_rope_dim,
                    position as u32,
                    None,
                );
                let kv_cache_off = (mtp_seq_len * kv_dim) as u32;
                self.dispatch_copy_offset(enc, &*buf_k, &*buf_k_cache, kv_dim as u32, kv_cache_off);
                self.dispatch_copy_offset(enc, &*buf_v, &*buf_v_cache, kv_dim as u32, kv_cache_off);
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            if let Some(ref mut mtp) = self.session.mtp {
                mtp.cache.seq_len += 1;
            }
        }

        /// Internal single-token forward step.
        ///
        /// When `capture_hidden` is true (or when `self.session.mtp` is loaded), copies the
        /// pre-final hidden state into `self.session.activations.pre_final_hidden` before the
        /// final RMSNorm, then reads it back to `self.session.last_pre_final_hidden`.
        fn forward_step_inner(
            &mut self,
            token_id: u32,
            position: usize,
            capture_hidden: bool,
            signpost_scope: crate::forward::signpost::Scope,
        ) -> MetalStepOutput {
            self.forward_step_inner_impl(
                token_id,
                position,
                capture_hidden,
                false,
                true,
                GdnStateTrafficScope::Decode,
                None,
                None,
                signpost_scope,
            )
        }

        /// Same as `forward_step_inner`, but attributes GDN state traffic to the
        /// given scope instead of the `Decode` default. Used by the MTP verify and
        /// speculative-repair paths so their state traffic is not misattributed to
        /// (and does not double count against) normal decode.
        #[cfg(feature = "gdn-state-counters")]
        fn forward_step_inner_with_traffic_scope(
            &mut self,
            token_id: u32,
            position: usize,
            capture_hidden: bool,
            scope: GdnStateTrafficScope,
        ) -> MetalStepOutput {
            // Both callers (repair replay, batched verifier) are MTP-verify
            // work, never autoregressive decode — always `NotDecode`.
            self.forward_step_inner_impl(
                token_id,
                position,
                capture_hidden,
                false,
                true,
                scope,
                None,
                None,
                crate::forward::signpost::Scope::NotDecode,
            )
        }

        /// Qwen3.5 vision (ADR-069 Metal S5 / MP2): run a single decoder step with the
        /// token-embedding lookup REPLACED by a caller-supplied row (HF `masked_scatter`
        /// semantics — mirrors `cpu_f16::forward_step_f16`'s `injected_embedding` contract).
        /// `token_id` is unused when injecting (the embedding lookup itself is skipped) but
        /// is still required to size-check against `vocab_size` in the non-injected caller
        /// paths that share `forward_step_inner_impl`; callers here pass `0` (always valid).
        ///
        /// The caller (`generate_multimodal_vision`) has already validated `embedding`'s
        /// length and finiteness before this is reached — this is a private, non-`pub`
        /// entry point, so that validation is not repeated per call.
        ///
        /// `generate_multimodal_vision` itself now calls
        /// [`Self::forward_step_injected_mrope`] (ADR-069 MP3 needs a per-token
        /// cos/sin row at every call site); this plain-injection entry point is
        /// retained because unit tests exercise the MP2 injection contract in
        /// isolation, without the M-RoPE table machinery.
        #[allow(dead_code)] // exercised by injection-only unit tests, see doc comment
        fn forward_step_injected(&mut self, embedding: &[f32], position: usize) -> Vec<f32> {
            self.cross_turn_prefix_cache.clear();
            // Never reached on a production path (see doc comment) — always `NotDecode`.
            self.forward_step_inner_impl(
                0,
                position,
                false,
                false,
                true,
                GdnStateTrafficScope::Decode,
                Some(embedding),
                None,
                crate::forward::signpost::Scope::NotDecode,
            )
            .logits
        }

        /// Qwen3.5 vision (ADR-069 Metal MP3): same REPLACE-injection contract as
        /// [`Self::forward_step_injected`], plus a per-token interleaved-axis
        /// M-RoPE cos/sin row for the six GQA layers (see
        /// `forward_step_inner_impl`'s `mrope_cos_sin` doc). Used for image-pad
        /// prefill positions, where `injected_embedding` and `mrope_cos_sin` are
        /// both `Some`.
        ///
        /// `emit_head` gates the terminal RMSNorm + lm_head dispatch, same
        /// contract as `forward_step_inner_impl`'s parameter of the same name —
        /// pass `false` for a multimodal prefill position whose logits will
        /// never be sampled (KV/GDN state still advances); pass `true` for every
        /// decode step and for the prompt's final prefill position.
        fn forward_step_injected_mrope(
            &mut self,
            embedding: &[f32],
            position: usize,
            mrope_cos_sin: Option<(&[f32], &[f32])>,
            emit_head: bool,
            signpost_scope: crate::forward::signpost::Scope,
        ) -> Vec<f32> {
            self.cross_turn_prefix_cache.clear();
            self.forward_step_inner_impl(
                0,
                position,
                false,
                false,
                emit_head,
                GdnStateTrafficScope::Decode,
                Some(embedding),
                mrope_cos_sin,
                signpost_scope,
            )
            .logits
        }

        /// Qwen3.5 vision (ADR-069 Metal MP3): ordinary token-id embedding lookup
        /// (no injection), plus a per-token M-RoPE cos/sin row — used for
        /// non-image-pad positions in a multimodal request (text tokens and
        /// vision delimiters keep the normal embedding lookup, but still rotate
        /// with the 3-axis table once an image has shifted the position
        /// coordinate) and for every decode step after a multimodal prefill.
        ///
        /// `emit_head`: see [`Self::forward_step_injected_mrope`]'s doc.
        fn forward_step_mrope(
            &mut self,
            token_id: u32,
            position: usize,
            mrope_cos_sin: Option<(&[f32], &[f32])>,
            emit_head: bool,
            signpost_scope: crate::forward::signpost::Scope,
        ) -> Vec<f32> {
            self.cross_turn_prefix_cache.clear();
            self.forward_step_inner_impl(
                token_id,
                position,
                false,
                false,
                emit_head,
                GdnStateTrafficScope::Decode,
                None,
                mrope_cos_sin,
                signpost_scope,
            )
            .logits
        }

        // lattice#1584: `new_command_buffer` hands back a borrowed reference to an
        // autoreleased, unretained object, so every command buffer and encoder this
        // dispatch creates stays alive until some pool drains. The serving worker thread
        // is long-lived by construction, so its outermost pool never drains during a
        // session. Measured on this funnel at 3000 decode steps: 1.3-1.5 KB per step
        // accumulates without this pool and nothing accumulates with it, at identical
        // throughput.
        #[allow(clippy::too_many_arguments)]
        fn forward_step_inner_impl(
            &mut self,
            token_id: u32,
            position: usize,
            capture_hidden: bool,
            skip_logits_readback: bool,
            emit_head: bool,
            _traffic_scope: GdnStateTrafficScope,
            injected_embedding: Option<&[f32]>,
            mrope_cos_sin: Option<(&[f32], &[f32])>,
            signpost_scope: crate::forward::signpost::Scope,
        ) -> MetalStepOutput {
            objc::rc::autoreleasepool(|| {
                self.forward_step_inner_impl_dispatch(
                    token_id,
                    position,
                    capture_hidden,
                    skip_logits_readback,
                    emit_head,
                    _traffic_scope,
                    injected_embedding,
                    mrope_cos_sin,
                    signpost_scope,
                )
            })
        }

        /// `mrope_cos_sin`, when supplied (Qwen3.5 vision M-RoPE, ADR-069 MP3),
        /// replaces the six GQA layers' 1-D `engine.rope_cos`/`rope_sin` table
        /// lookup (keyed by `position`) with this caller-supplied interleaved-axis
        /// row for every full-attention layer in this step — mirrors
        /// `cpu_f16::full_attention_step_f16`'s `mrope_cos_sin` parameter. GDN
        /// layers never consume RoPE and are unaffected either way.
        ///
        /// `emit_head`, when `false`, skips the terminal RMSNorm + lm_head dispatch
        /// (`encode_final_head`) and its GPU readback entirely — every layer still
        /// runs, so the KV cache and GDN recurrent state are fully advanced, but
        /// `logits` comes back empty (issue #1336: a multimodal prefill position
        /// whose logits are never sampled has no reason to pay for the vocabulary
        /// projection). Forced back to `true` whenever `capture_hidden` or an
        /// active MTP session needs the pre-final hidden state this step, so a
        /// caller cannot accidentally starve MTP capture by passing `false`.
        #[allow(clippy::too_many_arguments)]
        fn forward_step_inner_impl_dispatch(
            &mut self,
            token_id: u32,
            position: usize,
            capture_hidden: bool,
            skip_logits_readback: bool,
            emit_head: bool,
            _traffic_scope: GdnStateTrafficScope,
            injected_embedding: Option<&[f32]>,
            mrope_cos_sin: Option<(&[f32], &[f32])>,
            signpost_scope: crate::forward::signpost::Scope,
        ) -> MetalStepOutput {
            let _signpost_step = crate::forward::signpost::interval_in(
                signpost_scope,
                crate::forward::signpost::Label::DecodeStep,
            );
            let cfg = self.engine.config.clone();
            let hidden = cfg.hidden_size;

            // A caller that wants pre-final hidden capture (MTP) still needs the
            // terminal RMSNorm dispatched even if it passed `emit_head=false`
            // (only the multimodal prefill loop does today, and never with MTP
            // active) — this keeps the flag safe by construction rather than by
            // caller discipline. `mtp_active` is this request's own commitment
            // to the MTP path (set by `generate()`'s `use_mtp`, cleared by
            // `reset_state()`), not `self.session.mtp.is_some()`: the latter
            // reports only that the checkpoint carries MTP weights, which stays
            // true even when this request runs with `enable_mtp: Some(false)`
            // (#1336 round 2 — that combination must still honor
            // `emit_head=false` for the multimodal prefill loop).
            let run_head = emit_head || capture_hidden || self.session.mtp_active;

            // Build the single-token M-RoPE cos/sin buffers once for this step (all
            // six GQA layers rotate with the same per-token row); `None` keeps every
            // `dispatch_partial_rope` call on the unchanged 1-D `engine.rope_cos`/
            // `rope_sin` path, so text-only decode is structurally untouched.
            let mrope_row_bufs: Option<(Buffer, Buffer)> =
                mrope_cos_sin.map(|(cos_row, sin_row)| {
                    (
                        make_buffer(&self.engine.device, cos_row, "mrope_cos_row"),
                        make_buffer(&self.engine.device, sin_row, "mrope_sin_row"),
                    )
                });
            let mrope_override: Option<(&Buffer, &Buffer)> =
                mrope_row_bufs.as_ref().map(|(c, s)| (c, s));

            // --- Per-phase timing (enabled by env LATTICE_PROFILE=1) ---
            let profiling = crate::env_switch_enabled("LATTICE_PROFILE");
            let decode_profiling = crate::env_switch_enabled("LATTICE_DECODE_PROFILE");

            if crate::env_switch_enabled("LATTICE_GDN_CPU") {
                #[cfg(not(debug_assertions))]
                panic!("LATTICE_GDN_CPU=1 is debug-only and cannot be used in release decode");
            }

            let mut prof = StepProfile::default();
            let mut gdn_gpu_dispatches = 0usize;
            let gdn_cpu_dispatches = 0usize;
            let expected_gdn_dispatches = cfg.num_active_linear_attention_layers();
            let step_start = std::time::Instant::now();

            // --- Embedding lookup (CPU, read f16 embed_tokens, write f32 hidden) ---
            // Qwen3.5 vision (ADR-069 Metal S5 / MP2): `injected_embedding` REPLACES this
            // lookup with a caller-supplied row (HF `masked_scatter`; mirrors
            // `cpu_f16::forward_step_f16`'s contract) instead of reading `embed_tokens`.
            let t0 = std::time::Instant::now();
            match injected_embedding {
                Some(row) => {
                    debug_assert_eq!(
                        row.len(),
                        hidden,
                        "forward_step_injected: caller must pre-validate row length"
                    );
                    // SAFETY: hidden buffer is StorageModeShared f32, no GPU in flight;
                    // caller (forward_step_injected) guarantees row.len() == hidden.
                    unsafe {
                        let dst = self.session.activations.hidden.contents() as *mut f32;
                        std::ptr::copy_nonoverlapping(row.as_ptr(), dst, hidden);
                    }
                }
                None => {
                    let embed_offset = token_id as usize * hidden;
                    assert!(
                        (token_id as usize) < cfg.vocab_size,
                        "forward_step: token_id {token_id} >= vocab_size {}",
                        cfg.vocab_size
                    );
                    // SAFETY: embed_tokens is StorageModeShared f16, no GPU in flight;
                    // token_id validated < vocab_size above so embed_offset is in bounds.
                    unsafe {
                        let src =
                            (self.engine.embed_tokens.contents() as *const u16).add(embed_offset);
                        let dst = self.session.activations.hidden.contents() as *mut f32;
                        // SAFETY: embed_tokens row has hidden u16 values; hidden buffer has hidden f32 values.
                        convert_f16_row(src, dst, hidden);
                    }
                }
            }

            if profiling {
                prof.embedding_us += t0.elapsed().as_micros();
            }

            let mut active_layer_idx = 0usize;
            let mut linear_idx = 0usize;
            let mut full_idx = 0usize;

            let kv_dim = cfg.full_kv_dim();

            if decode_profiling {
                // Per-group command-buffer split path (LATTICE_DECODE_PROFILE=1).
                // Each layer gets its own command buffer committed and waited before the next.
                // GPU time = Instant::now() wraps commit()+wait_until_completed() per layer.
                // This serializes cross-kernel GPU pipelining; absolute throughput is lower
                // than the fused path.
                for layer_i in 0..cfg.num_hidden_layers {
                    if !cfg.is_layer_active(layer_i) {
                        continue;
                    }
                    let compact_idx = active_layer_idx;
                    active_layer_idx += 1;
                    let is_linear = matches!(
                        &self.engine.layer_weights[compact_idx].0,
                        MetalLayerAttnWeights::Linear(_)
                    );
                    // SAFETY: same raw-pointer idiom as the fused path below.
                    // cmd is ref-counted by Metal and outlives the queue borrow.
                    // encode_* methods need &mut self but don't touch engine.queue.
                    let layer_cmd = unsafe {
                        &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
                    };
                    let layer_enc = layer_cmd.new_compute_command_encoder();
                    if is_linear {
                        // Mixer-only (no MLP): measures pure GDN recurrence cost.
                        let gdn_dispatches_this_layer = self.encode_gdn_layer(
                            layer_enc,
                            compact_idx,
                            linear_idx,
                            position,
                            layer_i,
                            &cfg,
                            &mut prof,
                            profiling,
                            false,
                        );
                        gdn_gpu_dispatches += gdn_dispatches_this_layer;
                        #[cfg(feature = "gdn-state-counters")]
                        self.record_gdn_layer_access(_traffic_scope, gdn_dispatches_this_layer);
                        linear_idx += 1;
                        layer_enc.end_encoding();
                        let t_mixer = std::time::Instant::now();
                        {
                            let _signpost_commit = crate::forward::signpost::interval_in(
                                signpost_scope,
                                crate::forward::signpost::Label::DecodeCbCommit,
                            );
                            layer_cmd.commit();
                        }
                        {
                            let _signpost_wait = crate::forward::signpost::interval_in(
                                signpost_scope,
                                crate::forward::signpost::Label::DecodeCbWait,
                            );
                            layer_cmd.wait_until_completed();
                        }
                        prof.gpu_gdn_mixer_us += t_mixer.elapsed().as_micros();
                        // MLP-only command buffer for this GDN layer.
                        let mlp_cmd = unsafe {
                            &*(self.engine.queue.new_command_buffer()
                                as *const metal::CommandBufferRef)
                        };
                        let mlp_enc = mlp_cmd.new_compute_command_encoder();
                        self.encode_mlp_block(
                            mlp_enc,
                            compact_idx,
                            layer_i,
                            position,
                            &cfg,
                            &mut prof,
                            false,
                        );
                        mlp_enc.end_encoding();
                        let t_mlp = std::time::Instant::now();
                        {
                            let _signpost_commit = crate::forward::signpost::interval_in(
                                signpost_scope,
                                crate::forward::signpost::Label::DecodeCbCommit,
                            );
                            mlp_cmd.commit();
                        }
                        {
                            let _signpost_wait = crate::forward::signpost::interval_in(
                                signpost_scope,
                                crate::forward::signpost::Label::DecodeCbWait,
                            );
                            mlp_cmd.wait_until_completed();
                        }
                        prof.gpu_mlp_us += t_mlp.elapsed().as_micros();
                    } else {
                        // Attn-only (no MLP): measures pure GQA attention cost.
                        self.encode_gqa_layer(
                            layer_enc,
                            compact_idx,
                            full_idx,
                            position,
                            kv_dim,
                            layer_i,
                            &cfg,
                            &mut prof,
                            profiling,
                            false,
                            mrope_override,
                        );
                        full_idx += 1;
                        layer_enc.end_encoding();
                        let t_attn = std::time::Instant::now();
                        {
                            let _signpost_commit = crate::forward::signpost::interval_in(
                                signpost_scope,
                                crate::forward::signpost::Label::DecodeCbCommit,
                            );
                            layer_cmd.commit();
                        }
                        {
                            let _signpost_wait = crate::forward::signpost::interval_in(
                                signpost_scope,
                                crate::forward::signpost::Label::DecodeCbWait,
                            );
                            layer_cmd.wait_until_completed();
                        }
                        prof.gpu_gqa_attn_us += t_attn.elapsed().as_micros();
                        // MLP-only command buffer for this GQA layer.
                        let mlp_cmd = unsafe {
                            &*(self.engine.queue.new_command_buffer()
                                as *const metal::CommandBufferRef)
                        };
                        let mlp_enc = mlp_cmd.new_compute_command_encoder();
                        self.encode_mlp_block(
                            mlp_enc,
                            compact_idx,
                            layer_i,
                            position,
                            &cfg,
                            &mut prof,
                            false,
                        );
                        mlp_enc.end_encoding();
                        let t_mlp = std::time::Instant::now();
                        {
                            let _signpost_commit = crate::forward::signpost::interval_in(
                                signpost_scope,
                                crate::forward::signpost::Label::DecodeCbCommit,
                            );
                            mlp_cmd.commit();
                        }
                        {
                            let _signpost_wait = crate::forward::signpost::interval_in(
                                signpost_scope,
                                crate::forward::signpost::Label::DecodeCbWait,
                            );
                            mlp_cmd.wait_until_completed();
                        }
                        prof.gpu_mlp_us += t_mlp.elapsed().as_micros();
                    }
                    // Keep legacy aggregates for backward compat with existing tooling.
                    prof.gpu_gdn_us = prof.gpu_gdn_mixer_us;
                    prof.gpu_gqa_us = prof.gpu_gqa_attn_us;
                }

                let head_cmd = unsafe {
                    &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
                };
                let head_enc = head_cmd.new_compute_command_encoder();
                let topk_which_inner = if run_head {
                    self.encode_final_head(head_enc, &cfg, capture_hidden, &mut prof, profiling)
                } else {
                    None
                };
                head_enc.end_encoding();
                let t_gpu = std::time::Instant::now();
                {
                    let _signpost_commit = crate::forward::signpost::interval_in(
                        signpost_scope,
                        crate::forward::signpost::Label::DecodeCbCommit,
                    );
                    head_cmd.commit();
                }
                {
                    let _signpost_wait = crate::forward::signpost::interval_in(
                        signpost_scope,
                        crate::forward::signpost::Label::DecodeCbWait,
                    );
                    head_cmd.wait_until_completed();
                }
                prof.gpu_lm_head_us += t_gpu.elapsed().as_micros();

                let total_gpu_us = prof.gpu_gdn_mixer_us
                    + prof.gpu_gqa_attn_us
                    + prof.gpu_mlp_us
                    + prof.gpu_lm_head_us;
                let pct = |v: u128| -> f64 {
                    if total_gpu_us == 0 {
                        0.0
                    } else {
                        v as f64 / total_gpu_us as f64 * 100.0
                    }
                };
                eprintln!("[DECODE_PROFILE] step {position}: total_gpu_us={total_gpu_us}");
                eprintln!(
                    "  gdn_mixer:   {:>8} µs  ({:5.1}%)  [18 GDN layers: norm+in_proj+conv1d+recurrence+out_proj (no MLP)]",
                    prof.gpu_gdn_mixer_us,
                    pct(prof.gpu_gdn_mixer_us)
                );
                eprintln!(
                    "  gqa_attn:    {:>8} µs  ({:5.1}%)  [6 GQA layers: norm+qkv_proj+attention+o_proj (no MLP)]",
                    prof.gpu_gqa_attn_us,
                    pct(prof.gpu_gqa_attn_us)
                );
                eprintln!(
                    "  mlp:         {:>8} µs  ({:5.1}%)  [all 24 layers: post_attn_norm+gate_up_proj+silu_mul+down_proj]",
                    prof.gpu_mlp_us,
                    pct(prof.gpu_mlp_us)
                );
                eprintln!(
                    "  lm_head:     {:>8} µs  ({:5.1}%)  [final norm + lm_head GEMV]",
                    prof.gpu_lm_head_us,
                    pct(prof.gpu_lm_head_us)
                );
                eprintln!(
                    "  NOTE: GPU time under serialized per-group command buffers (2 cmd bufs/layer + 1 for lm_head). Absolute throughput is lower than the fused path."
                );

                debug_assert_eq!(
                    gdn_gpu_dispatches, expected_gdn_dispatches,
                    "GDN GPU recurrence dispatch count mismatch"
                );

                if profiling {
                    let total_us = step_start.elapsed().as_micros();
                    let accounted_us = prof.embedding_us
                        + prof.projection_us
                        + prof.gdn_recurrence_us
                        + prof.gqa_attention_us
                        + prof.mlp_us
                        + prof.final_us;
                    let other_us = total_us.saturating_sub(accounted_us);
                    eprintln!(
                        "[PROFILE] step {position}: total_us={total_us} \
                         embedding_us={} projection_us={} gdn_recurrence_us={} \
                         gqa_attention_us={} mlp_us={} final_us={} other_us={} \
                         gdn_gpu_dispatches={} expected_gdn_dispatches={} gdn_cpu_dispatches={}",
                        prof.embedding_us,
                        prof.projection_us,
                        prof.gdn_recurrence_us,
                        prof.gqa_attention_us,
                        prof.mlp_us,
                        prof.final_us,
                        other_us,
                        gdn_gpu_dispatches,
                        expected_gdn_dispatches,
                        gdn_cpu_dispatches,
                    );
                }

                // SAFETY: GPU completed (wait_until_completed called above for head_cmd).
                let pre_final_hidden = if capture_hidden || self.session.mtp.is_some() {
                    if capture_hidden && self.path_proof_enabled {
                        self.path_proof
                            .decode_hidden_readback
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    let _signpost_host_read = crate::forward::signpost::interval_in(
                        signpost_scope,
                        crate::forward::signpost::Label::DecodeHostScalarRead,
                    );
                    let h =
                        unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
                    self.session.last_pre_final_hidden = h.clone();
                    self.session
                        .mark_pre_final_hidden(self.session.kv_cache.seq_len + 1);
                    h
                } else {
                    Vec::new()
                };

                let logits = if !run_head || skip_logits_readback {
                    vec![]
                } else if let Some(which) = topk_which_inner {
                    if self.path_proof_enabled {
                        self.path_proof
                            .decode_compact_candidate_logit_readback
                            .fetch_add(1, Ordering::Relaxed);
                        self.path_proof
                            .decode_compact_candidate_logit_readback_bytes
                            .fetch_add(
                                (self.session.compact_topk * std::mem::size_of::<GpuCandidate>())
                                    as u64,
                                Ordering::Relaxed,
                            );
                    }
                    let _signpost_host_read = crate::forward::signpost::interval_in(
                        signpost_scope,
                        crate::forward::signpost::Label::DecodeHostScalarRead,
                    );
                    let k = self.session.compact_topk;
                    let candidates = unsafe { self.read_topk_candidates(which, k) };
                    self.session.compact_result = candidates;
                    vec![]
                } else {
                    if self.path_proof_enabled {
                        self.path_proof
                            .decode_full_vocab_logit_readback
                            .fetch_add(1, Ordering::Relaxed);
                        self.path_proof
                            .decode_full_vocab_logit_readback_bytes
                            .fetch_add(
                                (cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
                                Ordering::Relaxed,
                            );
                    }
                    let _signpost_host_read = crate::forward::signpost::interval_in(
                        signpost_scope,
                        crate::forward::signpost::Label::DecodeHostScalarRead,
                    );
                    unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
                };
                self.session.kv_cache.seq_len += 1;

                return MetalStepOutput {
                    logits,
                    pre_final_hidden,
                    _signpost_step,
                };
            }

            // Fused single-command-buffer path (default, zero-cost when LATTICE_DECODE_PROFILE is off).
            // Single command buffer for ALL 24 layers — no intermediate waits.
            // Raw pointer breaks the borrow chain: cmd is ref-counted by Metal
            // and outlives the queue borrow. The encode_* methods need &mut self
            // (for session state) but don't touch engine.queue.
            let cmd = unsafe {
                &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
            };
            let enc = cmd.new_compute_command_encoder();

            for layer_i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(layer_i) {
                    continue;
                }
                let compact_idx = active_layer_idx;
                active_layer_idx += 1;
                let is_linear = matches!(
                    &self.engine.layer_weights[compact_idx].0,
                    MetalLayerAttnWeights::Linear(_)
                );
                if is_linear {
                    let gdn_dispatches_this_layer = self.encode_gdn_layer(
                        enc,
                        compact_idx,
                        linear_idx,
                        position,
                        layer_i,
                        &cfg,
                        &mut prof,
                        profiling,
                        true,
                    );
                    gdn_gpu_dispatches += gdn_dispatches_this_layer;
                    #[cfg(feature = "gdn-state-counters")]
                    self.record_gdn_layer_access(_traffic_scope, gdn_dispatches_this_layer);
                    linear_idx += 1;
                } else {
                    self.encode_gqa_layer(
                        enc,
                        compact_idx,
                        full_idx,
                        position,
                        kv_dim,
                        layer_i,
                        &cfg,
                        &mut prof,
                        profiling,
                        true,
                        mrope_override,
                    );
                    full_idx += 1;
                }
            } // end layer loop

            let topk_which = if run_head {
                self.encode_final_head(enc, &cfg, capture_hidden, &mut prof, profiling)
            } else {
                None
            };

            // Single submit for entire forward pass + optional top-k.
            enc.end_encoding();
            {
                let _signpost_commit = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeCbCommit,
                );
                cmd.commit();
            }
            {
                let _signpost_wait = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeCbWait,
                );
                cmd.wait_until_completed();
            }

            // Diagnostic: dump post-final-norm hidden state magnitudes for
            // divergence analysis vs MLX. Set LATTICE_HIDDEN_DUMP=path to enable.
            if let Ok(dump_path) = std::env::var("LATTICE_HIDDEN_DUMP") {
                // SAFETY: hidden buffer is StorageModeShared, GPU completed (just waited).
                // Reads `cfg.hidden_size` f32 values starting at offset 0 (post-RMSNorm hidden
                // for position 0 — only valid for decode/single-token forward_step).
                let hidden_vec: Vec<f32> = unsafe {
                    let ptr = self.session.activations.hidden.contents() as *const f32;
                    std::slice::from_raw_parts(ptr, cfg.hidden_size).to_vec()
                };
                let mag = hidden_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
                let max = hidden_vec.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let mean = hidden_vec.iter().sum::<f32>() / hidden_vec.len() as f32;
                eprintln!(
                    "[hidden-dump] token={token_id} pos={position} L2_norm={mag:.4} max_abs={max:.4} mean={mean:.4}"
                );
                let bytes: Vec<u8> = hidden_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(&dump_path, &bytes).expect("failed to dump hidden");
            }

            debug_assert_eq!(
                gdn_gpu_dispatches, expected_gdn_dispatches,
                "GDN GPU recurrence dispatch count mismatch"
            );

            if profiling {
                let total_us = step_start.elapsed().as_micros();
                let accounted_us = prof.embedding_us
                    + prof.projection_us
                    + prof.gdn_recurrence_us
                    + prof.gqa_attention_us
                    + prof.mlp_us
                    + prof.final_us;
                let other_us = total_us.saturating_sub(accounted_us);
                eprintln!(
                    "[PROFILE] step {position}: total_us={total_us} \
                     embedding_us={} projection_us={} gdn_recurrence_us={} \
                     gqa_attention_us={} mlp_us={} final_us={} other_us={} \
                     gdn_gpu_dispatches={} expected_gdn_dispatches={} gdn_cpu_dispatches={}",
                    prof.embedding_us,
                    prof.projection_us,
                    prof.gdn_recurrence_us,
                    prof.gqa_attention_us,
                    prof.mlp_us,
                    prof.final_us,
                    other_us,
                    gdn_gpu_dispatches,
                    expected_gdn_dispatches,
                    gdn_cpu_dispatches,
                );
            }

            // Read back pre-final hidden when requested (for MTP input).
            // SAFETY: GPU completed, pre_final_hidden is StorageModeShared.
            let pre_final_hidden = if capture_hidden || self.session.mtp.is_some() {
                if capture_hidden && self.path_proof_enabled {
                    self.path_proof
                        .decode_hidden_readback
                        .fetch_add(1, Ordering::Relaxed);
                }
                let _signpost_host_read = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeHostScalarRead,
                );
                let h = unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
                self.session.last_pre_final_hidden = h.clone();
                self.session
                    .mark_pre_final_hidden(self.session.kv_cache.seq_len + 1);
                h
            } else {
                Vec::new()
            };

            let logits = if !run_head || skip_logits_readback {
                vec![]
            } else if let Some(which) = topk_which {
                // Compact path: read k*(f32+u32)=k*8 bytes instead of vocab*4 bytes.
                // SAFETY: GPU completed, buffers are StorageModeShared.
                if self.path_proof_enabled {
                    self.path_proof
                        .decode_compact_candidate_logit_readback
                        .fetch_add(1, Ordering::Relaxed);
                    self.path_proof
                        .decode_compact_candidate_logit_readback_bytes
                        .fetch_add(
                            (self.session.compact_topk * std::mem::size_of::<GpuCandidate>())
                                as u64,
                            Ordering::Relaxed,
                        );
                }
                let _signpost_host_read = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeHostScalarRead,
                );
                let k = self.session.compact_topk;
                let candidates = unsafe { self.read_topk_candidates(which, k) };
                self.session.compact_result = candidates;
                vec![] // generate loop reads compact_result, not these logits
            } else {
                // Full path: read vocab_size f32 logits back to host.
                // SAFETY: GPU completed, buffer is StorageModeShared.
                if self.path_proof_enabled {
                    self.path_proof
                        .decode_full_vocab_logit_readback
                        .fetch_add(1, Ordering::Relaxed);
                    self.path_proof
                        .decode_full_vocab_logit_readback_bytes
                        .fetch_add(
                            (cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
                            Ordering::Relaxed,
                        );
                }
                let _signpost_host_read = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeHostScalarRead,
                );
                unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
            };
            self.session.kv_cache.seq_len += 1;

            MetalStepOutput {
                logits,
                pre_final_hidden,
                _signpost_step,
            }
        }

        fn check_forward_step_capacity(
            &self,
            position: usize,
        ) -> Result<(), crate::error::InferenceError> {
            use crate::error::InferenceError;

            let max_cache_len = self.session.kv_cache.max_cache_len;
            if position >= max_cache_len {
                return Err(InferenceError::InvalidInput(format!(
                    "forward_step: position {position} exceeds max position {}",
                    max_cache_len - 1
                )));
            }
            if self.session.kv_cache.seq_len >= max_cache_len {
                return Err(InferenceError::InvalidInput(format!(
                    "forward_step: KV cache is full at {} tokens (max_cache_len {max_cache_len})",
                    self.session.kv_cache.seq_len
                )));
            }
            Ok(())
        }

        fn check_forward_token_id(
            &self,
            entry_point: &str,
            token_id: u32,
        ) -> Result<(), crate::error::InferenceError> {
            if token_id as usize >= self.engine.config.vocab_size {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "{entry_point}: token_id {token_id} out of range: vocab_size is {}",
                    self.engine.config.vocab_size
                )));
            }
            Ok(())
        }

        fn check_forward_token_ids(
            &self,
            entry_point: &str,
            token_ids: &[u32],
        ) -> Result<(), crate::error::InferenceError> {
            for (index, &token_id) in token_ids.iter().enumerate() {
                if token_id as usize >= self.engine.config.vocab_size {
                    return Err(crate::error::InferenceError::InvalidInput(format!(
                        "{entry_point}: token_ids[{index}]={token_id} out of range: vocab_size \
                         is {}",
                        self.engine.config.vocab_size
                    )));
                }
            }
            Ok(())
        }

        fn validate_live_cursor(
            entry_point: &str,
            supplied_position: usize,
            live_cursor: usize,
        ) -> Result<(), crate::error::InferenceError> {
            if supplied_position != live_cursor {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "{entry_point}: supplied position {supplied_position} does not match the \
                     live cache cursor {live_cursor}; KV placement and attention length derive \
                     from kv_cache.seq_len while RoPE uses the supplied position"
                )));
            }
            Ok(())
        }

        fn check_live_cursor(
            &self,
            entry_point: &str,
            supplied_position: usize,
        ) -> Result<(), crate::error::InferenceError> {
            Self::validate_live_cursor(
                entry_point,
                supplied_position,
                self.session.kv_cache.seq_len,
            )
        }

        fn validate_hidden_prefill_fresh_session(
            seq_len: usize,
            gdn_state_is_initial: bool,
        ) -> Result<(), crate::error::InferenceError> {
            if seq_len != 0 || !gdn_state_is_initial {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "forward_prefill_with_hidden: requires a fresh session (kv_cache.seq_len \
                     == 0 and GDN recurrent state at its initial condition), found \
                     kv_cache.seq_len={seq_len}; this call always dispatches from position 0, so \
                     calling it against a session with live state would overwrite the \
                     existing KV prefix while leaving GDN state conditioned on the tokens it \
                     overwrote. Call reset_state() first to start a new prompt."
                )));
            }
            Ok(())
        }

        /// The fresh-session predicate: `kv_cache.seq_len == 0 && gdn_state_is_initial()`,
        /// with `gdn_state_is_initial()` treated as vacuously satisfied when the session
        /// has no GDN layers at all — there is no recurrent state for such a session to
        /// hold live, so `seq_len == 0` is already the complete freshness criterion.
        /// The ONLY place this predicate is computed — every caller that needs it goes
        /// through here (or through [`Self::check_hidden_prefill_fresh_session`], which
        /// wraps it) so a caller can never check `seq_len` without also checking GDN state.
        fn prefill_session_freshness(&self) -> (usize, bool) {
            let seq_len = self.session.kv_cache.seq_len;
            let gdn_fresh = !self.has_gdn_layers() || self.gdn_state_is_initial();
            (seq_len, seq_len == 0 && gdn_fresh)
        }

        fn check_hidden_prefill_fresh_session(&self) -> Result<(), crate::error::InferenceError> {
            let (seq_len, fresh) = self.prefill_session_freshness();
            Self::validate_hidden_prefill_fresh_session(seq_len, fresh)
        }

        fn check_raw_prefill_fresh_session(
            &self,
            entry_point: &str,
        ) -> Result<(), crate::error::InferenceError> {
            let (seq_len, fresh) = self.prefill_session_freshness();
            if !fresh {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "{entry_point}: requires a fresh session (kv_cache.seq_len == 0 and GDN \
                     recurrent state at its initial condition), found \
                     kv_cache.seq_len={seq_len}; this call always dispatches from position \
                     0. Call reset_state() first to start a new prompt."
                )));
            }
            Ok(())
        }

        /// Batched prefill has no MoE schedule: `assert_batched_prefill_dense_only`
        /// panics on it. Reported here as a typed error so a caller sees a rejection
        /// instead of the process aborting; the assert stays as the internal
        /// invariant for direct callers of the batched chunk schedulers.
        ///
        /// This is the unconditional form: it rejects whenever the model has an MoE
        /// layer, for call sites every remaining branch of which is batched.
        /// [`Self::forward_prefill_from`] calls it after its single-token and
        /// LoRA-on-MoE fast paths have already returned via per-token forward steps
        /// (which support MoE). Entry points whose per-token routing is expressible as a
        /// predicate on the input use
        /// [`Self::check_prefill_moe_batched_unsupported`], which applies the
        /// token-count/LoRA gate before delegating here.
        fn reject_moe_batched(
            &self,
            entry_point: &str,
        ) -> Result<(), crate::error::InferenceError> {
            if self.engine.has_moe_layer {
                return Err(crate::error::InferenceError::UnsupportedModel(format!(
                    "{entry_point}: batched prefill (GEMM path) does not support MoE \
                     layers. Decode one token at a time via forward_step for MoE \
                     models."
                )));
            }
            Ok(())
        }

        /// Gated form of [`Self::reject_moe_batched`] for entry points that route a
        /// single-token or LoRA-active input to the per-token path, where MoE is
        /// supported (a LoRA-active dense model takes the batched schedulers, which
        /// apply the adapter; this gate only matters when an MoE layer exists): only
        /// a multi-token, no-LoRA input reaches the batched
        /// schedulers, so only that shape is rejected.
        ///
        /// Wired into the public entry points that can reach the batched schedulers
        /// through `forward_prefill_impl` or the with-hidden chunk loop:
        /// [`Self::try_forward_prefill`], [`Self::forward_prefill_with_hidden`], and
        /// [`Self::forward_prefill_all_logits`]. The suffix primitive
        /// [`Self::forward_prefill_from`] — reached by the prefix-cache generation
        /// path — uses the unconditional form above. Every request-facing caller
        /// prefills through one of the four: `generate`,
        /// `generate_with_speculation`, `generate_streaming`,
        /// `generate_streaming_with_cancel`, and `embed_tokens` through the
        /// first, `compute_token_nlls` through the third,
        /// `generate_streaming_with_prefix_cache*` through the fourth — so an
        /// MoE model reports this input instead of aborting the process.
        ///
        /// [`Self::forward_prefill`] is the exception, by construction: it unwraps
        /// [`Self::try_forward_prefill`]'s result, so it still aborts. Its remaining
        /// callers are this module's tests, the bench and profiling examples, and any
        /// external consumer that wants the panicking form.
        fn check_prefill_moe_batched_unsupported(
            &self,
            entry_point: &str,
            token_count: usize,
        ) -> Result<(), crate::error::InferenceError> {
            if token_count <= 1 || self.lora.is_some() {
                return Ok(());
            }
            self.reject_moe_batched(entry_point)
        }

        /// Pure capacity precondition for a Metal dispatch spanning `token_count` positions
        /// starting at `start_pos` — every position `start_pos..start_pos + token_count` must
        /// land inside the RoPE table / KV cache, and (when `gdn_pool_capacity` is `Some`)
        /// `token_count` must not exceed the fixed-size GDN checkpoint pool. Checked before any
        /// of it reaches Metal dispatch or GDN checkpoint mutation, so a caller-supplied token
        /// count can never leave session state partially advanced on rejection.
        ///
        /// `gdn_pool_capacity` bounds callers that checkpoint one pool slot per processed token
        /// (slot 0 = base, slot k = after k tokens, so valid slots are `0..=capacity` and at
        /// most `capacity` tokens may be processed after the base checkpoint).
        ///
        /// Free function (no `&self`) so it is unit-testable on CPU without a Metal device.
        fn validate_dispatch_capacity(
            start_pos: usize,
            token_count: usize,
            kv_capacity: usize,
            gdn_pool_capacity: Option<usize>,
        ) -> Result<(), crate::error::InferenceError> {
            use crate::error::InferenceError;

            if token_count == 0 {
                return Ok(());
            }
            let last_position = start_pos.saturating_add(token_count - 1);
            if last_position >= kv_capacity {
                return Err(InferenceError::InvalidInput(format!(
                    "start_pos {start_pos} + {token_count} tokens reaches position \
                     {last_position}, exceeding max position {}",
                    kv_capacity - 1
                )));
            }
            if let Some(pool_capacity) = gdn_pool_capacity
                && token_count > pool_capacity
            {
                return Err(InferenceError::InvalidInput(format!(
                    "start_pos {start_pos} + {token_count} tokens exceeds GDN checkpoint pool \
                     capacity {pool_capacity} (checkpoint slots 0..={pool_capacity})"
                )));
            }
            Ok(())
        }

        /// Method form of [`Self::validate_dispatch_capacity`], resolving the KV/RoPE bound
        /// from live session state.
        ///
        /// Scope, stated precisely because the obvious paraphrase of it is wrong: this is
        /// shared by every caller that dispatches a caller-supplied *range*,
        /// `start_pos..start_pos + token_count`. It is not on the single-position decode
        /// path. `forward_step_decode` and `forward_step_greedy_argmax` take one position
        /// and are bounded by their caller instead: each decode loop tests the pending
        /// position against `kv_cache.max_cache_len` and breaks with `StopReason::KvFull`
        /// before dispatching, with the headroom that loop needs. There are three distinct
        /// headroom requirements, not three loops — the loops are more numerous than the
        /// shapes, and reading the list below as a list of loops is the mistake this
        /// paragraph exists to prevent. The MTP loop reserves two slots
        /// (`max_cache_len.saturating_sub(2)`); the self-speculative loop reserves
        /// `SELF_SPEC_MAX_DRAFT + 1`; every other decode loop, including the multimodal and
        /// vision ones, needs a single free slot and tests `seq_len >= max_cache_len`
        /// directly. The streaming entry `generate_streaming_with_cancel` decodes through
        /// the decoder session instead, whose `decode` refuses a full cache with an error;
        /// its admission bound keeps every request clear of that refusal.
        /// Read as "every Metal dispatch goes through here" this comment is false, and a
        /// reader who checks it against the decode path will correctly conclude the guard is
        /// bypassed.
        ///
        /// Pass `check_gdn_pool = true` for callers that checkpoint one GDN pool slot per
        /// token (`verify_tokens_batched`); other callers pass `false` since they never
        /// advance the GDN checkpoint past the base slot.
        fn check_forward_range_capacity(
            &self,
            start_pos: usize,
            token_count: usize,
            check_gdn_pool: bool,
        ) -> Result<(), crate::error::InferenceError> {
            let gdn_pool_capacity = if check_gdn_pool {
                self.session.gdn_checkpoints.as_ref().map(|p| p.max_tokens)
            } else {
                None
            };
            Self::validate_dispatch_capacity(
                start_pos,
                token_count,
                self.session.kv_cache.max_cache_len,
                gdn_pool_capacity,
            )
        }

        /// True when the session has at least one GatedDeltaNet linear-attention
        /// layer, i.e. `gdn_state_is_initial()` has a population to answer about.
        /// A session with zero GDN layers holds no recurrent state at all — that
        /// is a distinct fact from "the recurrent state is clean" and callers must
        /// check this before asking [`Self::gdn_state_is_initial`], never conflate
        /// the two by treating an empty population as if it were a clean one.
        fn has_gdn_layers(&self) -> bool {
            self.session.gdn_gpu_state.len() != 0
        }

        /// True when every GDN recurrent-state buffer (S matrices and conv1d
        /// rolling buffers, for every linear-attention layer) holds its
        /// post-`reset_state()` zero value. Reads the live GPU buffers directly
        /// (same buffers and safety invariant as `snapshot_gdn_states`), so this
        /// reflects the state Metal dispatch will actually see, not the CPU
        /// `gdn_states` mirror.
        ///
        /// # Panics
        ///
        /// Panics when the session has no GDN layers (`has_gdn_layers()` is
        /// false). A vacuous "true" there would silently conflate "no buffers to
        /// check" with "buffers checked and clean" — two different facts a caller
        /// may need to tell apart. Callers that consider both cases equally fresh
        /// (there is nothing to protect when there is no GDN state) must say so
        /// explicitly at the call site, e.g. `!self.has_gdn_layers() ||
        /// self.gdn_state_is_initial()`, as [`Self::prefill_session_freshness`]
        /// does.
        fn gdn_state_is_initial(&self) -> bool {
            let num_layers = self.session.gdn_gpu_state.len();
            assert!(
                num_layers > 0,
                "gdn_state_is_initial: session has no GDN layers; check has_gdn_layers() \
                 first — an empty population and a clean population are not the same fact"
            );
            for i in 0..num_layers {
                let layer = self.session.gdn_gpu_state.layer(i);
                let conv_buf = layer.conv_buffer();
                let s_buf = layer.s_matrix();
                let conv_floats = (conv_buf.length() / 4) as usize;
                let s_floats = (s_buf.length() / 4) as usize;
                // SAFETY: GPU buffers are StorageModeShared (allocated with
                // MTLResourceOptions::StorageModeShared in `new`/`from_q4_dir`), so
                // `contents()` points to host-readable memory. `length()` is the
                // exact allocated byte length and is divisible by 4 (we always
                // allocate f32 buffers). Callers invoke this before their own GPU
                // dispatch, with no command buffer in flight, so no GPU write can
                // race with this read — same invariant as `snapshot_gdn_states`.
                let conv_nonzero = unsafe {
                    let ptr = conv_buf.contents() as *const f32;
                    std::slice::from_raw_parts(ptr, conv_floats)
                        .iter()
                        .any(|&v| v != 0.0)
                };
                if conv_nonzero {
                    return false;
                }
                let s_nonzero = unsafe {
                    let ptr = s_buf.contents() as *const f32;
                    std::slice::from_raw_parts(ptr, s_floats)
                        .iter()
                        .any(|&v| v != 0.0)
                };
                if s_nonzero {
                    return false;
                }
            }
            true
        }

        /// **Unstable**: fallible single-token forward step; kernel dispatch strategy evolving.
        ///
        /// Run a single token through the full model. Returns logits [vocab_size].
        ///
        /// `position` must equal the live cache cursor (`kv_cache.seq_len`). KV
        /// placement and attention length derive from the live cursor while RoPE
        /// uses the supplied `position`; sparse or out-of-order positions are not
        /// supported.
        ///
        /// # Errors
        ///
        /// Returns [`crate::error::InferenceError::InvalidInput`] before GPU dispatch
        /// when `token_id >= vocab_size`, when `position` does not equal
        /// `kv_cache.seq_len`, or when `position` or the next KV-cache row is
        /// outside the session capacity. On rejection, session state is left
        /// unchanged.
        pub fn try_forward_step(
            &mut self,
            token_id: u32,
            position: usize,
        ) -> Result<Vec<f32>, crate::error::InferenceError> {
            self.check_forward_token_id("try_forward_step", token_id)?;
            self.check_forward_step_capacity(position)?;
            self.check_live_cursor("try_forward_step", position)?;
            self.cross_turn_prefix_cache.clear();
            let output = self.forward_step_inner(
                token_id,
                position,
                false,
                crate::forward::signpost::Scope::NotDecode,
            );
            Ok(output.logits)
        }

        /// **Unstable**: fallible single-token forward with explicit hidden readback.
        ///
        /// Returns `(logits, pre_final_hidden)`. The hidden row is captured before
        /// the final RMSNorm overwrites the activation buffer. Existing
        /// [`Self::try_forward_step`] and generation paths retain their current
        /// readback behavior.
        ///
        /// `position` must equal the live cache cursor (`kv_cache.seq_len`): KV
        /// placement and attention length derive from `kv_cache.seq_len` while
        /// RoPE rotates by the supplied `position`, so a mismatched `position`
        /// would rotate the token for a position other than the KV row it is
        /// actually written to. Callers driving an out-of-order or sparse
        /// position sequence are not supported by this entry point.
        ///
        /// # Errors
        ///
        /// Returns [`crate::error::InferenceError::InvalidInput`] before GPU
        /// dispatch when `token_id`, `position`, or the next KV-cache row is
        /// outside the session capacity, or when `position` does not match
        /// `kv_cache.seq_len`. On rejection, session state (KV cache, GDN
        /// state, hidden-readback counters) is left unchanged.
        pub fn forward_step_with_hidden(
            &mut self,
            token_id: u32,
            position: usize,
        ) -> Result<(Vec<f32>, Vec<f32>), crate::error::InferenceError> {
            self.check_forward_token_id("forward_step_with_hidden", token_id)?;
            self.check_forward_step_capacity(position)?;
            self.check_live_cursor("forward_step_with_hidden", position)?;
            self.cross_turn_prefix_cache.clear();
            let output = self.forward_step_inner(
                token_id,
                position,
                true,
                crate::forward::signpost::Scope::NotDecode,
            );
            Ok((output.logits, output.pre_final_hidden))
        }

        /// **Unstable**: single-token forward step; kernel dispatch strategy evolving.
        ///
        /// Run a single token through the full model. Returns logits [vocab_size].
        ///
        /// # Panics
        ///
        /// Panics if `token_id >= vocab_size`, if `position` does not equal the
        /// live cache cursor (`kv_cache.seq_len`), or if `position` or the next
        /// KV-cache row exceeds the session capacity. Use [`Self::try_forward_step`]
        /// to receive a typed input error. These checks run before GPU dispatch.
        ///
        /// # Cross-turn cache invalidation (#516)
        ///
        /// This is a public raw-forward entry point that advances live KV/GDN
        /// state outside the cache-aware generation family's ownership, so it
        /// must clear any retained [`MetalCrossTurnPrefixCache`] entry before
        /// mutating — otherwise a later `generate_streaming_with_prefix_cache`
        /// call could restore a GDN snapshot for KV rows this call already
        /// overwrote. Safe to call from inside the cache-aware decode loop
        /// itself: by the time that loop reaches its own `forward_step` call,
        /// `restore_cross_turn_prefix` has already `take()`-n (ExactAppend) or
        /// `reset_state()` has already cleared (FullRefill) the slot's entry,
        /// so this clear is a no-op there — the cache-aware path never has a
        /// live entry saved while it is still generating.
        pub fn forward_step(&mut self, token_id: u32, position: usize) -> Vec<f32> {
            // General-purpose raw entry point: reused for prompt prefill (a
            // token-at-a-time loop, e.g. `forward_prefill_impl`,
            // `forward_prefill_from`, multimodal prefill) as well as by
            // external library consumers calling it directly, so it cannot
            // assume decode scope. The production autoregressive decode
            // loops call `forward_step_decode` instead (below), which is
            // `Scope::Decode`. Delegates to `try_forward_step` so this path
            // gets the same before-dispatch capacity validation as external
            // callers of the fallible API.
            match self.try_forward_step(token_id, position) {
                Ok(logits) => logits,
                Err(error) => panic!("{error}"),
            }
        }

        /// Same contract as [`Self::forward_step`], for the production
        /// autoregressive decode loops (`generate`, `generate_streaming*`,
        /// `generate_streaming_with_prefix_cache*`, and
        /// `crate::decoder::qwen_metal::QwenMetalSession`) only — marks the
        /// `decode.*` signpost interval `Scope::Decode` so it is not silent.
        /// Not `pub`: external consumers get the general-purpose
        /// `forward_step`, which stays `Scope::NotDecode` since it cannot
        /// know whether the caller is decoding or prefilling.
        pub(crate) fn forward_step_decode(&mut self, token_id: u32, position: usize) -> Vec<f32> {
            self.cross_turn_prefix_cache.clear();
            self.forward_step_inner(
                token_id,
                position,
                false,
                crate::forward::signpost::Scope::Decode,
            )
            .logits
        }

        /// Zero-copy greedy argmax: run forward pass then scan GPU shared buffer
        /// directly for the argmax token ID, avoiding 993KB allocation+copy.
        pub(crate) fn forward_step_greedy_argmax(&mut self, token_id: u32, position: usize) -> u32 {
            let cfg = self.engine.config.clone();
            // Binding (not discarding) the returned guard keeps `decode.step`
            // alive through the host-read/argmax scan below, matching its
            // documented span (through logits/top-k readback) — the scan is
            // logically part of this step, not a separate one.
            let _step_guard = self.forward_step_inner_impl(
                token_id,
                position,
                false,
                true,
                true,
                GdnStateTrafficScope::Decode,
                None,
                None,
                crate::forward::signpost::Scope::Decode,
            );

            // Fused host read + greedy sample: this scan both reads the
            // shared logits buffer off the GPU and picks the argmax token in
            // the same pass (the whole point of the zero-copy path), so it
            // carries both signpost labels for the span it covers.
            let _signpost_host_read = crate::forward::signpost::interval(
                crate::forward::signpost::Label::DecodeHostScalarRead,
            );
            let _signpost_sample =
                crate::forward::signpost::interval(crate::forward::signpost::Label::DecodeSample);
            // SAFETY: GPU completed (wait_until_completed called in forward_step_inner).
            // logits buffer is StorageModeShared — CPU can read it directly.
            unsafe {
                let ptr = self.session.activations.logits.contents() as *const f32;
                let vocab_size = cfg.vocab_size;
                let mut best_id = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for i in 0..vocab_size {
                    let v = *ptr.add(i);
                    if v > best_val {
                        best_val = v;
                        best_id = i as u32;
                    }
                }
                best_id
            }
        }

        // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
        // dispatch creates command buffers on the same long-lived worker thread.
        // Unmeasured individually; fixed on the structural argument, not on numbers.
        fn forward_step_gdn_only(&mut self, token_id: u32, position: usize) -> Vec<f32> {
            objc::rc::autoreleasepool(|| self.forward_step_gdn_only_dispatch(token_id, position))
        }

        /// Run a token through GDN layers only, skipping all GQA layers.
        ///
        /// GQA layers are bypassed: the residual stream passes through unmodified,
        /// the KV cache is NOT advanced, and `session.kv_cache.seq_len` is NOT
        /// incremented. The function returns full-vocab logits computed from the
        /// GDN-only hidden state via final-norm + lm_head.
        ///
        /// Caller is responsible for checkpointing GDN state before calling.
        fn forward_step_gdn_only_dispatch(&mut self, token_id: u32, position: usize) -> Vec<f32> {
            let cfg = self.engine.config.clone();
            let hidden = cfg.hidden_size;

            // Embedding lookup — identical to forward_step_inner.
            // SAFETY: embed_tokens is StorageModeShared f16, no GPU in flight;
            // token_id validated < vocab_size by the generate loop above.
            unsafe {
                let src = (self.engine.embed_tokens.contents() as *const u16)
                    .add(token_id as usize * hidden);
                let dst = self.session.activations.hidden.contents() as *mut f32;
                // SAFETY: embed_tokens row has hidden u16 values; hidden buffer has hidden f32 values.
                convert_f16_row(src, dst, hidden);
            }

            let mut active_layer_idx = 0usize;
            let mut linear_idx = 0usize;

            // Single command buffer for all GDN layers (GQA layers produce no encoder work).
            // Raw pointer breaks the borrow chain; identical idiom to forward_step_inner.
            let cmd = unsafe {
                &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
            };
            let enc = cmd.new_compute_command_encoder();

            let mut prof = StepProfile::default();
            for layer_i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(layer_i) {
                    continue;
                }
                let compact_idx = active_layer_idx;
                active_layer_idx += 1;
                let is_linear = matches!(
                    &self.engine.layer_weights[compact_idx].0,
                    MetalLayerAttnWeights::Linear(_)
                );
                if is_linear {
                    self.encode_gdn_layer(
                        enc,
                        compact_idx,
                        linear_idx,
                        position,
                        layer_i,
                        &cfg,
                        &mut prof,
                        false,
                        true,
                    );
                    #[cfg(feature = "gdn-state-counters")]
                    self.record_gdn_layer_access(GdnStateTrafficScope::Decode, 1);
                    linear_idx += 1;
                }
                // GQA layers: residual stream passes through without modification.
            }

            // Final head (RMSNorm + lm_head) — capture_hidden=false, no MTP side-effect.
            self.encode_final_head(enc, &cfg, false, &mut prof, false);

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Read logits — do NOT increment kv_cache.seq_len.
            // SAFETY: GPU completed, buffer is StorageModeShared.
            unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
        }

        /// **Unstable**: fallible batch prompt prefill; kernel strategy may change.
        ///
        /// Returns logits for the last token while advancing KV and GDN state. This
        /// is the non-panicking raw-prefill route; existing callers of
        /// [`Self::forward_prefill`] can migrate by handling the returned
        /// [`Result`]. Empty input retains `forward_prefill`'s existing zero-logit
        /// result.
        ///
        /// # Fresh-prompt semantics
        ///
        /// This call always dispatches from position 0. It requires a fresh
        /// session (`kv_cache.seq_len == 0` and GDN recurrent state at its initial
        /// condition) and does not reset on the caller's behalf. Call
        /// [`Self::reset_state`] before starting a new prompt on a reused session.
        ///
        /// # Errors
        ///
        /// Returns [`crate::error::InferenceError::InvalidInput`] before GPU
        /// dispatch for a non-fresh session, an out-of-vocabulary token, or a token
        /// range beyond the session capacity, and
        /// [`crate::error::InferenceError::UnsupportedModel`] for a multi-token
        /// prompt on an MoE model with no active LoRA adapter. Rejection leaves
        /// session state unchanged.
        pub fn try_forward_prefill(
            &mut self,
            token_ids: &[u32],
        ) -> Result<Vec<f32>, crate::error::InferenceError> {
            self.check_raw_prefill_fresh_session("try_forward_prefill")?;
            self.check_forward_token_ids("try_forward_prefill", token_ids)?;
            self.check_forward_range_capacity(0, token_ids.len(), false)?;
            self.check_prefill_moe_batched_unsupported("try_forward_prefill", token_ids.len())?;
            self.cross_turn_prefix_cache.clear();
            Ok(self.forward_prefill_impl(token_ids, false))
        }

        /// **Unstable**: batch prompt prefill; prefill kernel and fallback threshold may change.
        ///
        /// Batch prefill: process all prompt tokens at once using GEMM.
        ///
        /// Returns logits for the LAST token. Updates KV cache and GDN state.
        /// Uses batch GEMM (M=prompt_len) for projections instead of per-token GEMV,
        /// giving ~10-20x speedup on prefill for typical prompt lengths.
        ///
        /// Prompts longer than max_prefill are processed in max_prefill-sized batched
        /// chunks. A loaded LoRA adapter is applied inside the batched chunks; only a
        /// LoRA-active model with MoE layers stays on the per-token forward_step
        /// fallback.
        ///
        /// # Fresh-prompt semantics
        ///
        /// This call always dispatches from position 0. It requires a fresh
        /// session (`kv_cache.seq_len == 0` and GDN recurrent state at its
        /// initial condition), does not reset state on the caller's behalf,
        /// and does not append to an existing prefix. Call [`Self::reset_state`]
        /// before reusing a session for a new prompt.
        ///
        /// # Panics
        ///
        /// Panics when [`Self::try_forward_prefill`] returns an error: the session
        /// is not fresh, a token is outside the vocabulary, or the token range
        /// exceeds capacity. New code should use the fallible route and handle its
        /// typed input error. This compatibility wrapper preserves the existing
        /// return type for callers that treat those conditions as contract
        /// violations.
        ///
        /// # Cross-turn cache invalidation (#516)
        ///
        /// Public raw-forward entry point — see [`Self::forward_step`]'s doc
        /// comment for the invariant this enforces. The clear happens inside
        /// [`Self::try_forward_prefill`], which this wraps. No production caller in
        /// this crate remains — the generation entry points prefill through the
        /// fallible route directly — so this wrapper's callers are this module's own
        /// tests, the bench and profiling examples, and any external consumer that
        /// wants the panicking form.
        pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
            match self.try_forward_prefill(token_ids) {
                Ok(logits) => logits,
                Err(error) => panic!("{error}"),
            }
        }

        /// **Unstable**: score a list of option letters with one prompt prefill (ADR-097).
        ///
        /// Resets the recurrent and KV state, runs exactly one
        /// [`Self::try_forward_prefill`] over `prompt_ids` and reads the last-position logits at
        /// `letter_ids`. There is no sampling and no decode step. The returned
        /// [`OptionScores`](crate::option_scoring::OptionScores) holds the k raw logits in the
        /// caller's order, their softmax over the k, and the full-vocabulary mass on the k letters.
        ///
        /// The scorer never loads or unloads an adapter: whichever adapter the state holds is the
        /// one in effect. On a dense model an active adapter is applied inside the batched prefill;
        /// on a model with MoE layers it keeps the per-token path. Because the call
        /// resets the state first, any session that was in progress is discarded, and the state is
        /// left advanced by the prompt, so the next call resets again.
        ///
        /// # Errors
        ///
        /// Returns [`crate::error::InferenceError::InvalidInput`] before any state reset or GPU
        /// dispatch for an empty `prompt_ids`, a prompt token outside the vocabulary, a prompt
        /// longer than the session capacity, an empty `letter_ids`, a letter id outside the
        /// vocabulary or a repeated letter id; a rejected call leaves the session untouched.
        /// Returns [`crate::error::InferenceError::UnsupportedModel`] for a multi-token prompt on
        /// an MoE model with no active adapter, and [`crate::error::InferenceError::Inference`]
        /// when the logits contain a non-finite value.
        pub fn score_option_letters(
            &mut self,
            prompt_ids: &[u32],
            letter_ids: &[u32],
        ) -> Result<crate::option_scoring::OptionScores, crate::error::InferenceError> {
            if prompt_ids.is_empty() {
                return Err(crate::error::InferenceError::InvalidInput(
                    "score_option_letters: prompt_ids must not be empty".into(),
                ));
            }
            crate::option_scoring::check_letter_ids(self.engine.config.vocab_size, letter_ids)?;
            self.check_forward_token_ids("score_option_letters", prompt_ids)?;
            self.check_forward_range_capacity(0, prompt_ids.len(), false)?;
            self.check_prefill_moe_batched_unsupported("score_option_letters", prompt_ids.len())?;
            self.reset_state();
            let logits = self.try_forward_prefill(prompt_ids)?;
            crate::option_scoring::option_scores_from_logits(&logits, letter_ids)
        }

        /// **Unstable**: fallible prompt prefill with explicit hidden readback.
        ///
        /// Returns `(last_token_logits, pre_final_hidden)`. The hidden row is
        /// captured for the prompt's last token before final RMSNorm. Batched,
        /// chunked, one-token, and MoE-with-LoRA fallback paths advance KV/GDN state by
        /// exactly the same token range as [`Self::forward_prefill`].
        ///
        /// # Fresh-prompt semantics
        ///
        /// This call always dispatches from position 0. It requires a fresh
        /// session (`kv_cache.seq_len == 0` and GDN recurrent state at its
        /// initial condition) and rejects otherwise — it does not reset the
        /// session on the caller's behalf, and it does not implement append
        /// semantics against an existing prefix. Calling it against a session
        /// with live state would overwrite the existing KV prefix rows while
        /// leaving GDN recurrent state conditioned on the tokens it overwrote,
        /// an unreconstructible mix. Call [`Self::reset_state`] first to start a
        /// new prompt on a session that has already generated.
        ///
        /// # Errors
        ///
        /// Returns [`crate::error::InferenceError::InvalidInput`] before GPU
        /// dispatch for an empty prompt, an out-of-vocabulary token, a prompt
        /// whose token range exceeds the session capacity, or a session that is
        /// not fresh (see above). Returns
        /// [`crate::error::InferenceError::UnsupportedModel`] for a multi-token
        /// prompt against a model with MoE layers and no active LoRA adapter: that
        /// input takes the batched schedulers, which have no MoE schedule. On
        /// rejection, session state (KV cache, GDN state, hidden-readback counters)
        /// is left unchanged.
        pub fn forward_prefill_with_hidden(
            &mut self,
            token_ids: &[u32],
        ) -> Result<(Vec<f32>, Vec<f32>), crate::error::InferenceError> {
            use crate::error::InferenceError;

            if token_ids.is_empty() {
                return Err(InferenceError::InvalidInput(
                    "forward_prefill_with_hidden: token_ids must not be empty".into(),
                ));
            }
            self.check_forward_token_ids("forward_prefill_with_hidden", token_ids)?;
            self.check_hidden_prefill_fresh_session()?;
            self.check_forward_range_capacity(0, token_ids.len(), false)?;
            self.check_prefill_moe_batched_unsupported(
                "forward_prefill_with_hidden",
                token_ids.len(),
            )?;
            self.cross_turn_prefix_cache.clear();

            if token_ids.len() == 1 {
                return self.forward_step_with_hidden(token_ids[0], 0);
            }
            if self.lora.is_some() && self.engine.has_moe_layer {
                // The batched schedulers have no MoE schedule; per-token steps do.
                let last_index = token_ids.len() - 1;
                for (position, &token_id) in token_ids[..last_index].iter().enumerate() {
                    self.try_forward_step(token_id, position)?;
                }
                return self.forward_step_with_hidden(token_ids[last_index], last_index);
            }

            let max_prefill = self.session.max_prefill;
            if token_ids.len() <= max_prefill {
                let logits = self.forward_prefill_batched_chunk(token_ids, 0, false, true, true);
                return Ok((logits, self.session.last_pre_final_hidden.clone()));
            }

            let mut start_pos = 0usize;
            let mut last_logits = Vec::new();
            for chunk in token_ids.chunks(max_prefill) {
                let is_last = start_pos + chunk.len() == token_ids.len();
                last_logits =
                    self.forward_prefill_batched_chunk(chunk, start_pos, false, is_last, is_last);
                start_pos += chunk.len();
            }
            Ok((last_logits, self.session.last_pre_final_hidden.clone()))
        }

        /// **Unstable**: prefill that returns logits for ALL `n` positions, not
        /// just the last. Output is `n * vocab_size` f32 values, row-major:
        /// `logits[i * vocab_size .. (i+1) * vocab_size]` predicts `tokens[i+1]`.
        ///
        /// Used by perplexity evaluation. Allocates a per-call buffer of size
        /// `n * vocab_size * 4` bytes — for `vocab_size=248K` and `n=128` that's
        /// ~127MB. Callers should cap `n` accordingly (typical PPL window: 128).
        ///
        /// # Fresh-prompt semantics
        ///
        /// This call always dispatches from position 0. It requires a fresh
        /// session (`kv_cache.seq_len == 0` and GDN recurrent state at its
        /// initial condition), does not reset state on the caller's behalf,
        /// and does not append to an existing prefix. Call [`Self::reset_state`]
        /// before reusing a session for a new prompt.
        ///
        /// # Panics
        ///
        /// Panics if any `token_ids[i] >= vocab_size`. See [`forward_prefill`] for details.
        ///
        /// # Errors
        ///
        /// Returns [`crate::error::InferenceError::InvalidInput`] before GPU
        /// dispatch when the session is not fresh (see above). Returns an error
        /// when more than one position is requested while a LoRA adapter is active,
        /// because the batched all-position path does not apply LoRA projections.
        /// Returns [`crate::error::InferenceError::UnsupportedModel`] for a
        /// multi-token prompt against a model with MoE layers: the all-position
        /// path is batched by construction and the batched schedulers have no MoE
        /// schedule.
        ///
        /// # Cross-turn cache invalidation (#516)
        ///
        /// Public raw-forward entry point — see [`Self::forward_step`]'s doc
        /// comment for the invariant this enforces.
        pub fn forward_prefill_all_logits(
            &mut self,
            token_ids: &[u32],
        ) -> Result<Vec<f32>, crate::error::InferenceError> {
            self.check_raw_prefill_fresh_session("forward_prefill_all_logits")?;
            if token_ids.len() > 1 && self.lora.is_some() {
                return Err(crate::error::InferenceError::Inference(
                    "forward_prefill_all_logits: all-position prefill does not apply LoRA \
                     adapters; unload the adapter before requesting all-position logits"
                        .into(),
                ));
            }
            self.check_prefill_moe_batched_unsupported(
                "forward_prefill_all_logits",
                token_ids.len(),
            )?;
            self.cross_turn_prefix_cache.clear();
            Ok(self.forward_prefill_impl(token_ids, true))
        }

        fn forward_prefill_impl(&mut self, token_ids: &[u32], all_positions: bool) -> Vec<f32> {
            // Start of a fresh prompt pass: drop any hidden states captured by a
            // previous prefill so `mtp_prefill` never pairs this prompt's tokens
            // with a stale prior prompt's hidden states (#1340).
            self.session.mtp_prefill_hidden.clear();
            let n = token_ids.len();
            let vocab = self.engine.config.vocab_size;
            // Validate all ids once at the entry point (O(seq_len)) before any GPU work.
            // Do NOT duplicate this inside the hot embedding loop; the inner chunk code
            // relies on this pre-scan and omits the per-token guard.
            for (t, &id) in token_ids.iter().enumerate() {
                assert!(
                    (id as usize) < vocab,
                    "forward_prefill: token_ids[{t}]={id} out of range: vocab_size is {vocab}",
                );
            }
            if n == 0 {
                return if all_positions {
                    vec![]
                } else {
                    vec![0.0; vocab]
                };
            }
            if n == 1 {
                return self.forward_step(token_ids[0], 0);
            }
            if self.lora.is_some() && self.engine.has_moe_layer {
                // The batched schedulers have no MoE schedule; per-token steps do.
                let mut last_logits = Vec::new();
                for (pos, &id) in token_ids.iter().enumerate() {
                    last_logits = self.forward_step(id, pos);
                }
                return last_logits;
            }
            assert!(
                n <= self.session.kv_cache.max_cache_len,
                "prefill length {} exceeds max_cache_len {}",
                n,
                self.session.kv_cache.max_cache_len
            );
            let max_prefill = self.session.max_prefill;
            if n <= max_prefill {
                // Only/last chunk — its logits are always the caller's answer.
                return self.forward_prefill_batched_chunk(
                    token_ids,
                    0,
                    all_positions,
                    true,
                    false,
                );
            }
            // Chunked batched prefill: each chunk is one command buffer (preserving the
            // n≤512 fast path within each chunk). GDN recurrent state threads across
            // boundaries automatically — session GPU buffers are mutated in place and
            // never reset between chunks within a single request.
            if all_positions {
                let mut all_logits = Vec::with_capacity(n * vocab);
                let mut start_pos = 0usize;
                for chunk in token_ids.chunks(max_prefill) {
                    // Perplexity needs every chunk's per-position logits.
                    all_logits.extend(
                        self.forward_prefill_batched_chunk(chunk, start_pos, true, true, false),
                    );
                    start_pos += chunk.len();
                }
                all_logits
            } else {
                let mut last_logits = Vec::new();
                let mut start_pos = 0usize;
                for chunk in token_ids.chunks(max_prefill) {
                    // Only the LAST chunk's tail (RMSNorm+lm_head+MTP+top-k) is ever kept —
                    // intermediate chunks still commit their layer work and advance
                    // set_position, they just skip the terminal tail (Experiment B).
                    let is_last = start_pos + chunk.len() == n;
                    last_logits =
                        self.forward_prefill_batched_chunk(chunk, start_pos, false, is_last, false);
                    start_pos += chunk.len();
                }
                last_logits
            }
        }

        fn copy_prefill_embeddings(&self, token_ids: &[u32], hidden: usize) {
            // SAFETY: embed_tokens is StorageModeShared f16 with no GPU work in flight;
            // callers validate every token id before reaching the prefill schedulers.
            unsafe {
                let src_base = self.engine.embed_tokens.contents() as *const u16;
                let dst_base = self.session.activations.hidden.contents() as *mut f32;
                for (t, &id) in token_ids.iter().enumerate() {
                    let src = src_base.add(id as usize * hidden);
                    let dst = dst_base.add(t * hidden);
                    convert_f16_row(src, dst, hidden);
                }
            }
        }

        fn encode_prefill_dense_mlp(
            &self,
            enc: &ComputeCommandEncoderRef,
            common_w: &MetalCommonLayerWeights,
            attention_delta: &Buffer,
            cfg: &Qwen35Config,
            layer_idx: usize,
            m: u32,
        ) {
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let (w_gate_up, w_down, gate_byte_size) = match &common_w.ffn {
                MetalFfnWeights::Dense {
                    gate_up_proj,
                    down_proj,
                } => {
                    let byte_size = match self.engine.quant_format {
                        QuantFormat::Q8_0 => (inter * (hidden / QK8_0) * Q8_0_BLOCK_SIZE) as u64,
                        QuantFormat::Q4_0 => (inter * (hidden / QK4_0) * Q4_0_BLOCK_SIZE) as u64,
                    };
                    (gate_up_proj, down_proj, byte_size)
                }
                MetalFfnWeights::Moe(_) => {
                    panic!("batch prefill does not support MoE layers when M>1")
                }
            };

            self.dispatch_fused_residual_add_norm_batch(
                enc,
                &self.session.activations.residual,
                attention_delta,
                &self.session.activations.residual,
                &self.session.activations.hidden,
                &common_w.post_attention_layernorm,
                hidden as u32,
                m,
                cfg.rms_norm_eps,
            );
            self.dispatch_gemm_at(
                enc,
                &self.session.activations.hidden,
                0,
                w_gate_up,
                0,
                &self.session.activations.gate,
                0,
                m,
                inter as u32,
                hidden as u32,
            );
            self.dispatch_gemm_at(
                enc,
                &self.session.activations.hidden,
                0,
                w_gate_up,
                gate_byte_size,
                &self.session.activations.up,
                0,
                m,
                inter as u32,
                hidden as u32,
            );
            // Adapter deltas land on the separate gate / up matrices before silu_mul.
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.hidden,
                0,
                hidden as u32,
                &self.session.activations.gate,
                0,
                inter as u32,
                m,
                layer_idx,
                "gate_proj",
            );
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.hidden,
                0,
                hidden as u32,
                &self.session.activations.up,
                0,
                inter as u32,
                m,
                layer_idx,
                "up_proj",
            );
            self.dispatch_silu_mul(enc, m * inter as u32);
            self.dispatch_gemm(
                enc,
                &self.session.activations.gate,
                0,
                w_down,
                &self.session.activations.ffn_out,
                0,
                m,
                hidden as u32,
                inter as u32,
            );
            // down_proj reads the post-silu_mul gate matrix and adds into ffn_out before
            // the residual add.
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.gate,
                0,
                inter as u32,
                &self.session.activations.ffn_out,
                0,
                hidden as u32,
                m,
                layer_idx,
                "down_proj",
            );
            self.dispatch_add_and_copy(
                enc,
                &self.session.activations.ffn_out,
                &self.session.activations.residual,
                &self.session.activations.hidden,
                m * hidden as u32,
            );
        }

        fn encode_gdn_prefill_before_recurrence(
            &self,
            enc: &ComputeCommandEncoderRef,
            compact_idx: usize,
            layer_idx: usize,
            cfg: &Qwen35Config,
            m: u32,
        ) {
            let (attn_w, common_w) = &self.engine.layer_weights[compact_idx];
            let MetalLayerAttnWeights::Linear(gdn_w) = attn_w else {
                unreachable!()
            };
            self.dispatch_copy_and_rms_norm_batch(
                enc,
                &self.session.activations.hidden,
                &self.session.activations.residual,
                &common_w.input_layernorm,
                cfg.hidden_size as u32,
                m,
                cfg.rms_norm_eps,
            );
            self.dispatch_gemm(
                enc,
                &self.session.activations.hidden,
                0,
                &gdn_w.in_proj_qkv,
                &self.session.activations.gdn_qkv,
                0,
                m,
                cfg.linear_qkv_dim() as u32,
                cfg.hidden_size as u32,
            );
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.hidden,
                0,
                cfg.hidden_size as u32,
                &self.session.activations.gdn_qkv,
                0,
                cfg.linear_qkv_dim() as u32,
                m,
                layer_idx,
                "in_proj_qkv",
            );
            self.dispatch_gemm(
                enc,
                &self.session.activations.hidden,
                0,
                &gdn_w.in_proj_z,
                &self.session.activations.gdn_z,
                0,
                m,
                cfg.linear_output_dim() as u32,
                cfg.hidden_size as u32,
            );
            // The recurrence reads and then overwrites gdn_z, so the in_proj_z delta must
            // be in place before it runs.
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.hidden,
                0,
                cfg.hidden_size as u32,
                &self.session.activations.gdn_z,
                0,
                cfg.linear_output_dim() as u32,
                m,
                layer_idx,
                "in_proj_z",
            );
        }

        fn encode_gdn_prefill_recurrence(
            &self,
            enc: &ComputeCommandEncoderRef,
            compact_idx: usize,
            linear_idx: usize,
            cfg: &Qwen35Config,
            n: usize,
            chunked_enabled: bool,
        ) {
            let MetalLayerAttnWeights::Linear(gdn_w) = &self.engine.layer_weights[compact_idx].0
            else {
                unreachable!()
            };
            let chunk_params = if chunked_enabled {
                Self::make_gdn_chunk_params(cfg, n)
            } else {
                None
            };
            if let Some(params) = chunk_params {
                let weights = GdnChunkedWeights {
                    conv1d_weight: &gdn_w.conv1d_weight,
                    in_proj_b: &gdn_w.in_proj_b,
                    in_proj_a: &gdn_w.in_proj_a,
                    a_log: &gdn_w.a_log,
                    dt_bias: &gdn_w.dt_bias,
                    norm_weight: &gdn_w.norm_weight,
                };
                self.dispatch_gdn_chunked_prefill_layer(enc, linear_idx, weights, params);
                return;
            }

            let qkv_d = cfg.linear_qkv_dim();
            let out_d = cfg.linear_output_dim();
            let num_h = cfg.linear_num_key_heads;
            let key_d = cfg.linear_key_head_dim;
            let val_d = cfg.linear_value_head_dim;
            let num_vh = cfg.linear_num_value_heads();
            let q_total = (num_h * key_d) as u32;
            #[repr(C)]
            struct GdnRecurParams {
                key_dim: u32,
                value_dim: u32,
                num_key_heads: u32,
                num_value_heads: u32,
                hidden_size: u32,
                q_total: u32,
                v_offset: u32,
                scale: f32,
                eps: f32,
            }
            let params = GdnRecurParams {
                key_dim: key_d as u32,
                value_dim: val_d as u32,
                num_key_heads: num_h as u32,
                num_value_heads: num_vh as u32,
                hidden_size: cfg.hidden_size as u32,
                q_total,
                v_offset: q_total * 2,
                scale: 1.0 / (key_d as f32).sqrt(),
                eps: cfg.rms_norm_eps,
            };
            for t in 0..n {
                let qkv_off = (t * qkv_d) as u64 * 4;
                let z_off = (t * out_d) as u64 * 4;
                let h_off = (t * cfg.hidden_size) as u64 * 4;

                enc.set_compute_pipeline_state(&self.engine.pipelines.conv1d_silu);
                enc.set_buffer(
                    0,
                    Some(self.session.gdn_gpu_state.layer(linear_idx).conv_buffer()),
                    0,
                );
                enc.set_buffer(1, Some(&self.session.activations.gdn_qkv), qkv_off);
                enc.set_buffer(2, Some(&gdn_w.conv1d_weight), 0);
                enc.set_buffer(3, Some(&self.session.gdn_gpu_conv_out), 0);
                let qd = qkv_d as u32;
                let ks = cfg.linear_conv_kernel_dim as u32;
                enc.set_bytes(4, 4, &qd as *const u32 as *const _);
                enc.set_bytes(5, 4, &ks as *const u32 as *const _);
                let wg = 256u64;
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(qkv_d as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );

                enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_recurrence);
                enc.set_buffer(
                    0,
                    Some(self.session.gdn_gpu_state.layer(linear_idx).s_matrix()),
                    0,
                );
                enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                enc.set_buffer(2, Some(&self.session.activations.gdn_z), z_off);
                enc.set_buffer(3, Some(&self.session.activations.hidden), h_off);
                enc.set_buffer(4, Some(&gdn_w.in_proj_b), 0);
                enc.set_buffer(5, Some(&gdn_w.in_proj_a), 0);
                enc.set_buffer(6, Some(&gdn_w.a_log), 0);
                enc.set_buffer(7, Some(&gdn_w.dt_bias), 0);
                enc.set_buffer(8, Some(&gdn_w.norm_weight), 0);
                enc.set_buffer(9, Some(&self.session.activations.gdn_z), z_off);
                enc.set_bytes(
                    10,
                    std::mem::size_of::<GdnRecurParams>() as u64,
                    &params as *const GdnRecurParams as *const _,
                );
                enc.dispatch_thread_groups(
                    MTLSize::new(num_vh as u64, 1, 1),
                    MTLSize::new(32, 4, 1),
                );
            }
        }

        fn encode_gdn_prefill_after_recurrence(
            &self,
            enc: &ComputeCommandEncoderRef,
            compact_idx: usize,
            layer_idx: usize,
            cfg: &Qwen35Config,
            m: u32,
        ) {
            let (attn_w, common_w) = &self.engine.layer_weights[compact_idx];
            let MetalLayerAttnWeights::Linear(gdn_w) = attn_w else {
                unreachable!()
            };
            self.dispatch_gemm(
                enc,
                &self.session.activations.gdn_z,
                0,
                &gdn_w.out_proj,
                &self.session.activations.attn_out,
                0,
                m,
                cfg.hidden_size as u32,
                cfg.linear_output_dim() as u32,
            );
            // out_proj reads the post-recurrence gdn_z rows and adds into attn_out before
            // the MLP's residual-add consumes it.
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.gdn_z,
                0,
                cfg.linear_output_dim() as u32,
                &self.session.activations.attn_out,
                0,
                cfg.hidden_size as u32,
                m,
                layer_idx,
                "out_proj",
            );
            self.encode_prefill_dense_mlp(
                enc,
                common_w,
                &self.session.activations.attn_out,
                cfg,
                layer_idx,
                m,
            );
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_full_attention_prefill_layer(
            &self,
            enc: &ComputeCommandEncoderRef,
            compact_idx: usize,
            layer_idx: usize,
            full_idx: usize,
            cfg: &Qwen35Config,
            n: usize,
            m: u32,
            start_pos: usize,
        ) {
            let hidden = cfg.hidden_size;
            let kv_dim = cfg.full_kv_dim();
            let q_dim = cfg.full_q_dim();
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let head_dim = cfg.head_dim;
            let half_rope_dim = (cfg.rope_dim() / 2) as u32;
            let (attn_w, common_w) = &self.engine.layer_weights[compact_idx];
            let MetalLayerAttnWeights::Full(full_w) = attn_w else {
                unreachable!()
            };

            self.dispatch_copy_and_rms_norm_batch(
                enc,
                &self.session.activations.hidden,
                &self.session.activations.residual,
                &common_w.input_layernorm,
                hidden as u32,
                m,
                cfg.rms_norm_eps,
            );
            self.dispatch_gemm(
                enc,
                &self.session.activations.hidden,
                0,
                &full_w.q_proj,
                &self.session.activations.q,
                0,
                m,
                (2 * q_dim) as u32,
                hidden as u32,
            );
            // Q rows carry both the query and gate lanes (width 2 * q_dim); the delta must
            // land before the scatter, per-head norm, RoPE and KV write consume them.
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.hidden,
                0,
                hidden as u32,
                &self.session.activations.q,
                0,
                (2 * q_dim) as u32,
                m,
                layer_idx,
                "q_proj",
            );
            self.dispatch_gemm(
                enc,
                &self.session.activations.hidden,
                0,
                &full_w.k_proj,
                &self.session.activations.k,
                0,
                m,
                kv_dim as u32,
                hidden as u32,
            );
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.hidden,
                0,
                hidden as u32,
                &self.session.activations.k,
                0,
                kv_dim as u32,
                m,
                layer_idx,
                "k_proj",
            );
            self.dispatch_gemm(
                enc,
                &self.session.activations.hidden,
                0,
                &full_w.v_proj,
                &self.session.activations.v,
                0,
                m,
                kv_dim as u32,
                hidden as u32,
            );
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.hidden,
                0,
                hidden as u32,
                &self.session.activations.v,
                0,
                kv_dim as u32,
                m,
                layer_idx,
                "v_proj",
            );

            let base_pos = start_pos as u32;
            let nqh = num_q_heads as u32;
            let nkh = num_kv_heads as u32;
            let hd = head_dim as u32;
            self.dispatch_scatter_q_gate_batch(enc, m, nqh, hd);
            self.dispatch_per_head_rms_norm_batch(
                enc,
                &self.session.activations.q_separated,
                &full_w.q_norm,
                m,
                nqh,
                hd,
                cfg.rms_norm_eps,
            );
            self.dispatch_per_head_rms_norm_batch(
                enc,
                &self.session.activations.k,
                &full_w.k_norm,
                m,
                nkh,
                hd,
                cfg.rms_norm_eps,
            );
            self.dispatch_partial_rope_batch(
                enc,
                &self.session.activations.q_separated,
                m,
                nqh,
                hd,
                half_rope_dim,
                base_pos,
            );
            self.dispatch_partial_rope_batch(
                enc,
                &self.session.activations.k,
                m,
                nkh,
                hd,
                half_rope_dim,
                base_pos,
            );
            self.dispatch_copy_kv_cache_batch(
                enc,
                &self.session.activations.k,
                &self.session.activations.v,
                &self.session.kv_cache.k_bufs[full_idx],
                &self.session.kv_cache.v_bufs[full_idx],
                m,
                kv_dim as u32,
                base_pos,
            );

            let scale = 1.0f32 / (head_dim as f32).sqrt();
            if self
                .dispatch_prefill_attention_batched(
                    enc,
                    &self.session.kv_cache.k_bufs[full_idx],
                    &self.session.kv_cache.v_bufs[full_idx],
                    base_pos,
                    m,
                    hd,
                    nqh,
                    nkh,
                    q_dim as u32,
                    kv_dim as u32,
                    scale,
                )
                .is_err()
            {
                for t in 0..n {
                    let abs_pos = start_pos + t;
                    let qs_off = (t * q_dim) as u64 * 4;
                    let ao_off = (t * q_dim) as u64 * 4;
                    let cache_len = (abs_pos + 1) as u32;
                    if self.use_kv_f16 {
                        enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention_f16);
                    } else {
                        enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
                    }
                    enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                    enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                    enc.set_buffer(2, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                    enc.set_buffer(3, Some(&self.session.activations.attn_out), ao_off);
                    enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                    enc.set_bytes(5, 4, &hd as *const u32 as *const _);
                    enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                    enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                    let qd = q_dim as u32;
                    let kvd = kv_dim as u32;
                    enc.set_bytes(8, 4, &qd as *const u32 as *const _);
                    enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
                    enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(nkh as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
            }

            self.dispatch_sigmoid_gate(enc, m * q_dim as u32);
            self.dispatch_gemm(
                enc,
                &self.session.activations.attn_out,
                0,
                &full_w.o_proj,
                &self.session.activations.ffn_out,
                0,
                m,
                hidden as u32,
                q_dim as u32,
            );
            // o_proj reads the sigmoid-gated attention rows and adds into ffn_out before
            // the MLP's residual-add consumes it.
            self.dispatch_lora_rows_if_active(
                enc,
                &self.session.activations.attn_out,
                0,
                q_dim as u32,
                &self.session.activations.ffn_out,
                0,
                hidden as u32,
                m,
                layer_idx,
                "o_proj",
            );
            self.encode_prefill_dense_mlp(
                enc,
                common_w,
                &self.session.activations.ffn_out,
                cfg,
                layer_idx,
                m,
            );
        }

        /// Every active layer's FFN must be dense (`MetalFfnWeights::Dense`) for the
        /// batched (M>1 GEMM) prefill schedulers — `encode_prefill_dense_mlp` panics
        /// on `MetalFfnWeights::Moe`. Called before any embedding copy, command
        /// buffer, or per-layer encode so an unsupported config is rejected with zero
        /// state mutation instead of panicking mid-encode after earlier layers'
        /// GDN/KV work has already been recorded (or, for schedulers that split
        /// encoding across multiple command buffers, already committed to the GPU).
        fn assert_batched_prefill_dense_only(&self) {
            assert!(
                !self.engine.has_moe_layer,
                "batched prefill does not support MoE layers (M>1 GEMM path); \
                 use decode-mode forward_step for MoE models instead"
            );
        }

        // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
        // dispatch creates command buffers on the same long-lived worker thread.
        // Unmeasured individually; fixed on the structural argument, not on numbers.
        fn forward_prefill_batched_chunk(
            &mut self,
            token_ids: &[u32],
            start_pos: usize,
            all_positions: bool,
            emit_logits: bool,
            capture_hidden: bool,
        ) -> Vec<f32> {
            objc::rc::autoreleasepool(|| {
                self.forward_prefill_batched_chunk_dispatch(
                    token_ids,
                    start_pos,
                    all_positions,
                    emit_logits,
                    capture_hidden,
                )
            })
        }

        /// Batched single-command-buffer prefill for a contiguous token slice starting
        /// at absolute position `start_pos`.
        ///
        /// Writes full-attention K/V rows `start_pos..start_pos+n` and advances GDN
        /// recurrent state in the session GPU buffers. Called by `forward_prefill_impl`
        /// once per chunk; for `n ≤ max_prefill` it IS the entire prefill (one command
        /// buffer, all 24 layers, final logits).
        ///
        /// `emit_logits` gates ONLY the terminal RMSNorm, lm_head, hidden capture/readback,
        /// and top-k tail, which is purely terminal (it does not feed the residual
        /// stream, KV cache, or GDN recurrent state — those are fully advanced by the
        /// per-layer loop above it). When `false`, the layer command buffer for this
        /// chunk still commits and `set_position` still advances; only the wasted tail
        /// work (and its GPU readback) is skipped. Callers with an intermediate chunk of
        /// a chunked, last-token-only (`!all_positions`) prefill — whose tail output is
        /// always discarded — pass `false`; every other caller passes `true`.
        fn forward_prefill_batched_chunk_dispatch(
            &mut self,
            token_ids: &[u32],
            start_pos: usize,
            all_positions: bool,
            emit_logits: bool,
            capture_hidden: bool,
        ) -> Vec<f32> {
            let n = token_ids.len();
            debug_assert!(
                n <= self.session.max_prefill,
                "forward_prefill_batched_chunk: n={n} exceeds max_prefill={}",
                self.session.max_prefill
            );
            assert!(
                start_pos + n <= self.session.kv_cache.max_cache_len,
                "prefill chunk start_pos={start_pos} + n={n} exceeds max_cache_len {}",
                self.session.kv_cache.max_cache_len
            );
            self.assert_batched_prefill_dense_only();
            let cfg = self.engine.config.clone();
            let m = n as u32;
            let hidden = cfg.hidden_size;

            self.copy_prefill_embeddings(token_ids, hidden);

            let mut active_layer_idx = 0usize;
            let mut linear_idx = 0usize;
            let mut full_idx = 0usize;

            let chunked_requested = self.use_gdn_chunked;
            let chunked_supported = Self::supports_gdn_chunked_prefill(&cfg);
            if chunked_requested && !chunked_supported {
                tracing::warn!(
                    "chunked GDN prefill unsupported for this shape (0.8B only); \
                     falling back to serial scan (set LATTICE_GDN_CHUNKED=0 to suppress)"
                );
            }
            let chunked_enabled = chunked_requested && chunked_supported;

            // Single command buffer for entire prefill (all 24 layers + final logits)
            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            for layer_i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(layer_i) {
                    continue;
                }
                let compact_idx = active_layer_idx;
                active_layer_idx += 1;
                let is_linear = matches!(
                    &self.engine.layer_weights[compact_idx].0,
                    MetalLayerAttnWeights::Linear(_)
                );

                if is_linear {
                    self.encode_gdn_prefill_before_recurrence(enc, compact_idx, layer_i, &cfg, m);
                    self.encode_gdn_prefill_recurrence(
                        enc,
                        compact_idx,
                        linear_idx,
                        &cfg,
                        n,
                        chunked_enabled,
                    );
                    self.encode_gdn_prefill_after_recurrence(enc, compact_idx, layer_i, &cfg, m);
                    linear_idx += 1;
                } else {
                    self.encode_full_attention_prefill_layer(
                        enc,
                        compact_idx,
                        layer_i,
                        full_idx,
                        &cfg,
                        n,
                        m,
                        start_pos,
                    );
                    full_idx += 1;
                }
            }

            // MTP prefill capture (#1340): `activations.hidden` currently holds the
            // raw, pre-final-RMSNorm target hidden state for every one of this
            // chunk's `n` positions. The tail below overwrites row `n-1` (default)
            // or every row (`all_positions`) with its post-RMSNorm value in place,
            // so this read must land before that dispatch runs — hence ending and
            // committing the layer-loop's command buffer here instead of after the
            // tail. `mtp_prefill` (called from `generate()`) drains the accumulated
            // rows to rebuild the MTP draft head's KV cache from the prompt. Gated
            // on `mtp.is_some()` so a session without an MTP head pays nothing extra
            // — the pre-existing single-command-buffer path below is untouched.
            let (cmd, enc) = if self.session.mtp.is_some() {
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                // SAFETY: the command buffer just completed; `hidden` (the
                // activations buffer, not the local `hidden: usize` row-length) is
                // StorageModeShared and the layer loop above wrote exactly `n`
                // valid rows into it.
                let raw_hidden =
                    unsafe { read_buffer(&self.session.activations.hidden, n * hidden) };
                self.session
                    .mtp_prefill_hidden
                    .extend_from_slice(&raw_hidden);

                if !emit_logits {
                    self.session.set_position(start_pos + n);
                    return vec![];
                }

                let cmd = self.engine.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                (cmd, enc)
            } else {
                (cmd, enc)
            };

            if !emit_logits {
                // Intermediate chunk of a chunked, `!all_positions` prefill: the caller
                // only ever keeps the LAST chunk's logits, so the terminal tail below
                // (MTP capture, RMSNorm, lm_head, top-k, and their GPU readback) is pure
                // waste here — it does not feed the residual stream, KV cache, or GDN
                // recurrent state, which the per-layer loop above has already fully
                // advanced. Still commit this chunk's layer command buffer and advance
                // the session cursor exactly as the emit_logits==true path does below.
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                self.session.set_position(start_pos + n);
                return vec![];
            }

            // ---------- Final: RMSNorm + lm_head ----------
            // Two modes:
            //   - all_positions=false: RMSNorm + lm_head GEMV for last token only
            //     (default — used by chat/decode prefill, returns vocab_size f32)
            //   - all_positions=true:  RMSNorm + lm_head GEMM for all n tokens
            //     (used by perplexity eval, returns n*vocab_size f32, allocates a
            //     per-call buffer of n*vocab*4 bytes)
            let last_off = ((n - 1) * hidden) as u64 * 4;

            // Per-call output buffer for all-positions mode (n * vocab f32).
            // Allocated lazily to keep the chat path zero-overhead.
            let ppl_buf = if all_positions {
                let bytes = (n * cfg.vocab_size * 4) as u64;
                Some(
                    self.engine
                        .device
                        .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
                )
            } else {
                None
            };

            // Pre-final-hidden capture (last-token only; not needed for ppl).
            if !all_positions && (capture_hidden || self.session.mtp.is_some()) {
                enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
                enc.set_buffer(0, Some(&self.session.activations.hidden), last_off);
                enc.set_buffer(1, Some(&self.session.activations.pre_final_hidden), 0);
                let cnt = hidden as u32;
                let dst_off = 0u32;
                enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                enc.set_bytes(3, 4, &dst_off as *const u32 as *const _);
                let wg = 256u64;
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );
            }

            // RMSNorm — single-token (last) or all n tokens.
            {
                enc.set_compute_pipeline_state(&self.engine.pipelines.rms_norm);
                let (off, nr) = if all_positions {
                    (0u64, n as u32)
                } else {
                    (last_off, 1u32)
                };
                enc.set_buffer(0, Some(&self.session.activations.hidden), off);
                enc.set_buffer(1, Some(&self.engine.final_norm), 0);
                let rl = hidden as u32;
                enc.set_bytes(2, 4, &rl as *const u32 as *const _);
                enc.set_bytes(3, 4, &nr as *const u32 as *const _);
                enc.set_bytes(4, 4, &cfg.rms_norm_eps as *const f32 as *const _);
                // One threadgroup per row in batched mode, one for single-row mode.
                enc.dispatch_thread_groups(MTLSize::new(nr as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            // Site 2 of 3. `last_off` is normed under both branches above:
            // `all_positions` norms every row including the last, and the
            // single-token branch norms exactly that row.
            self.encode_capture_final_hidden(enc, last_off, hidden);

            // lm_head — for Q8 format (auto-quantized from F16 source), use the
            // FP16 embed_tokens buffer to avoid the ~0.79 PPL gap from per-row
            // symmetric Q8 quantization (which can't represent outlier rows
            // cleanly, corrupting hidden·embed dot products vs MLX).
            // For Q4 format (from .q4 dirs), keep using the Q4 path because
            // the FP16 buffer was dequantized from Q4 and the original Q4
            // matmul kernel is faster + handles QuaRot rotations correctly.
            // Issue #171: when the last-token route only needs argmax/top-k (never
            // the full logit vector), skip the full [vocab_size] GEMV entirely —
            // Stage 1 fuses the GEMV with a block-local exact reduction, Stage 2
            // (dispatch_block_topk_merge_enc) reduces those compact candidates to
            // the global result. all_positions (perplexity) always takes the
            // full-logit path below; it never sets compact_route to a Block route
            // in the generate loop, but we gate here too as defense in depth.
            let block_local_k = if all_positions {
                None
            } else {
                resolve_block_local_k(self.session.compact_route)
            };

            let topk_which = if let Some((local_k, slot)) = block_local_k {
                let candidate_groups =
                    self.dispatch_lm_head_block_stage1_enc(enc, &cfg, last_off, local_k, slot);
                Some(self.dispatch_block_topk_merge_enc(enc, candidate_groups, local_k))
            } else {
                let use_fp16_lm_head = matches!(self.engine.quant_format, QuantFormat::Q8_0);
                if use_fp16_lm_head {
                    let vocab_n = cfg.vocab_size as u32;
                    let hidden_k = hidden as u32;
                    if let Some(ref pb) = ppl_buf {
                        for p in 0..n {
                            let hidden_off = (p * hidden) as u64 * 4;
                            let logits_off = (p * cfg.vocab_size) as u64 * 4;
                            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode_wide);
                            enc.set_buffer(0, Some(&self.session.activations.hidden), hidden_off);
                            enc.set_buffer(1, Some(&self.engine.embed_tokens), 0);
                            enc.set_buffer(2, Some(pb), logits_off);
                            enc.set_bytes(3, 4, &vocab_n as *const u32 as *const _);
                            enc.set_bytes(4, 4, &hidden_k as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(vocab_n.div_ceil(4) as u64, 1, 1),
                                MTLSize::new(32, 4, 1),
                            );
                        }
                    } else {
                        enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode_wide);
                        enc.set_buffer(0, Some(&self.session.activations.hidden), last_off);
                        enc.set_buffer(1, Some(&self.engine.embed_tokens), 0);
                        enc.set_buffer(2, Some(&self.session.activations.logits), 0);
                        enc.set_bytes(3, 4, &vocab_n as *const u32 as *const _);
                        enc.set_bytes(4, 4, &hidden_k as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(vocab_n.div_ceil(4) as u64, 1, 1),
                            MTLSize::new(32, 4, 1),
                        );
                    }
                } else if let Some(ref pb) = ppl_buf {
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &self.engine.embed_tokens_q8,
                        pb,
                        0,
                        n as u32,
                        cfg.vocab_size as u32,
                        hidden as u32,
                    );
                } else {
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.hidden,
                        last_off,
                        &self.engine.embed_tokens_q8,
                        &self.session.activations.logits,
                        0,
                        1,
                        cfg.vocab_size as u32,
                        hidden as u32,
                    );
                }

                // Append top-k in the same encoder when compact mode is active (last-token only).
                if !all_positions && self.session.compact_topk > 0 {
                    let k = self.session.compact_topk as u32;
                    Some(self.dispatch_topk_enc(enc, cfg.vocab_size as u32, k))
                } else {
                    None
                }
            };

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // `set_position` clears the freshness marker, so it runs before the capture
            // below, which marks the new cursor.
            self.session.set_position(start_pos + n);
            if !all_positions && (capture_hidden || self.session.mtp.is_some()) {
                if capture_hidden && self.path_proof_enabled {
                    self.path_proof
                        .prefill_hidden_readback
                        .fetch_add(1, Ordering::Relaxed);
                }
                // SAFETY: GPU completed, pre_final_hidden is StorageModeShared.
                self.session.last_pre_final_hidden =
                    unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
                self.session.mark_pre_final_hidden(start_pos + n);
            }

            if let Some(pb) = ppl_buf {
                // SAFETY: GPU completed, ppl_buf is StorageModeShared and sized n*vocab.
                if self.path_proof_enabled {
                    self.path_proof
                        .prefill_full_vocab_logit_readback
                        .fetch_add(1, Ordering::Relaxed);
                    // `n *` is the whole reason the byte counter exists: this site and
                    // the ordinary one below share a counter and move different amounts.
                    self.path_proof
                        .prefill_full_vocab_logit_readback_bytes
                        .fetch_add(
                            (n * cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
                            Ordering::Relaxed,
                        );
                }
                unsafe { read_buffer(&pb, n * cfg.vocab_size) }
            } else if let Some(which) = topk_which {
                // SAFETY: GPU completed, buffers are StorageModeShared.
                if self.path_proof_enabled {
                    self.path_proof
                        .prefill_compact_candidate_logit_readback
                        .fetch_add(1, Ordering::Relaxed);
                    self.path_proof
                        .prefill_compact_candidate_logit_readback_bytes
                        .fetch_add(
                            (self.session.compact_topk * std::mem::size_of::<GpuCandidate>())
                                as u64,
                            Ordering::Relaxed,
                        );
                }
                let k = self.session.compact_topk;
                let candidates = unsafe { self.read_topk_candidates(which, k) };
                self.session.compact_result = candidates;
                vec![]
            } else {
                // SAFETY: The command buffer has completed and logits is a
                // StorageModeShared buffer sized for vocab_size f32 values.
                if self.path_proof_enabled {
                    self.path_proof
                        .prefill_full_vocab_logit_readback
                        .fetch_add(1, Ordering::Relaxed);
                    self.path_proof
                        .prefill_full_vocab_logit_readback_bytes
                        .fetch_add(
                            (cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
                            Ordering::Relaxed,
                        );
                }
                unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
            }
        }

        // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
        // dispatch creates command buffers on the same long-lived worker thread.
        // Unmeasured individually; fixed on the structural argument, not on numbers.
        #[cfg(feature = "bench-internals")]
        fn forward_prefill_chunk_gdn_isolated(
            &mut self,
            token_ids: &[u32],
            start_pos: usize,
            capture: &[bench_support::GdnCaptureRequest],
        ) -> (
            bench_support::GdnIsolatedChunkTiming,
            Vec<bench_support::GdnLayerCapture>,
        ) {
            objc::rc::autoreleasepool(|| {
                self.forward_prefill_chunk_gdn_isolated_dispatch(token_ids, start_pos, capture)
            })
        }

        /// GDN-recurrence-isolating variant of [`Self::forward_prefill_batched_chunk`],
        /// built for the #175 bench harness (`bin/bench_gdn_prefill_ab.rs`).
        ///
        /// Structurally identical to `forward_prefill_batched_chunk`: same op sequence,
        /// same buffers, same dispatch calls, same per-layer branch (GDN vs full
        /// attention). The ONLY difference is *where command-buffer boundaries fall*:
        /// immediately before and after each GDN layer's recurrence dispatch (either
        /// `dispatch_gdn_chunked_prefill_layer`'s C32 family, or the serial
        /// conv1d+recurrence per-token loop) the current encoder is ended and its
        /// command buffer committed + `wait_until_completed`, a fresh command buffer
        /// is opened for the GDN dispatch alone, then another fresh one for the rest of
        /// the layer. This isolates the GDN segment's CPU wall-clock time from the
        /// surrounding GEMMs/norms so the two scans' recurrence cost can be compared
        /// without the O(n^2) attention layers (or the GDN layers' own non-recurrence
        /// GEMMs) swamping the signal.
        ///
        /// No GDN kernel, dispatch argument, threadgroup geometry, or buffer differs
        /// from the production path — only the encoder/command-buffer boundaries
        /// change. This cannot introduce a new GDN-state race: within one encoder,
        /// Metal already serializes same-buffer read/write hazards between
        /// consecutive dispatches (the production path's correctness already depends
        /// on this), and an explicit `commit` + `wait_until_completed` between two
        /// *separate* command buffers on the same StorageModeShared buffers is a
        /// strictly stronger ordering guarantee than that. Splitting into more,
        /// smaller command buffers can only add synchronization stalls, never change
        /// what a kernel reads or writes.
        ///
        /// Always skips the terminal RMSNorm/lm_head/top-k tail (mirrors
        /// `forward_prefill_batched_chunk`'s `emit_logits = false` path) — logits are
        /// never needed for a dispatch-timing measurement, skipping the tail removes a
        /// large amount of code unrelated to GDN isolation, and the tail's cost is
        /// identical between the serial and chunked paths regardless (it depends on
        /// neither), so omitting it does not bias the serial-vs-chunked comparison.
        ///
        /// Handles exactly one `max_prefill`-sized chunk starting at `start_pos`
        /// (`n = token_ids.len() <= self.session.max_prefill`, asserted below); the
        /// caller drives the `max_prefill`-sized chunking loop for prompts longer than
        /// `max_prefill`, exactly as `forward_prefill_impl` does for the production
        /// path, threading GDN recurrent state across chunk boundaries via the
        /// session's GPU buffers (unchanged by this method).
        ///
        /// Gated on `bench-internals` (like `bench_support` itself, which owns this
        /// method's return type) — this is bench-only isolation plumbing, not part of
        /// the production dispatch path, so it stays out of the default `metal-gpu`
        /// build the way `bench_support`'s other helpers do.
        ///
        /// `capture`: optional read-only revalidation hooks (gdn175 S1b). For every
        /// entry whose `linear_idx` matches the GDN layer currently being processed,
        /// immediately after that layer's GDN command buffer has completed (and
        /// before the next GDN layer's materialize kernel can overwrite the shared
        /// scratch buffers), this reads back that entry's `value_head`'s per-token
        /// q/k/v/beta/log_alpha straight off the session's `gdn_chunk` scratch —
        /// see `capture_gdn_chunk_scratch`. Pass `&[]` for the normal timing-only
        /// path (unchanged behavior/dispatch either way; this never perturbs what
        /// gets dispatched, only what gets read back afterward).
        #[cfg(feature = "bench-internals")]
        fn forward_prefill_chunk_gdn_isolated_dispatch(
            &mut self,
            token_ids: &[u32],
            start_pos: usize,
            capture: &[bench_support::GdnCaptureRequest],
        ) -> (
            bench_support::GdnIsolatedChunkTiming,
            Vec<bench_support::GdnLayerCapture>,
        ) {
            let n = token_ids.len();
            assert!(
                n <= self.session.max_prefill,
                "forward_prefill_chunk_gdn_isolated: n={n} exceeds max_prefill={}",
                self.session.max_prefill
            );
            assert!(
                start_pos + n <= self.session.kv_cache.max_cache_len,
                "prefill chunk start_pos={start_pos} + n={n} exceeds max_cache_len {}",
                self.session.kv_cache.max_cache_len
            );
            self.assert_batched_prefill_dense_only();
            let cfg = self.engine.config.clone();
            let m = n as u32;
            let hidden = cfg.hidden_size;

            self.copy_prefill_embeddings(token_ids, hidden);

            let mut active_layer_idx = 0usize;
            let mut linear_idx = 0usize;
            let mut full_idx = 0usize;

            let chunked_requested = self.use_gdn_chunked;
            let chunked_supported = Self::supports_gdn_chunked_prefill(&cfg);
            // Bench-internals path: fail LOUD instead of falling back. The
            // production path degrades gracefully to the serial recurrence, but a
            // bench that did the same would measure the serial path while the
            // harness labels the arm "chunked" — a silent instrument corruption.
            assert!(
                !chunked_requested || chunked_supported,
                "GDN chunked prefill requested (set_gdn_chunked(true)) but this config \
                 does not support the chunked path — refusing to silently measure the \
                 serial recurrence under a 'chunked' label"
            );
            let chunked_enabled = chunked_requested && chunked_supported;

            let mut timing = bench_support::GdnIsolatedChunkTiming::default();
            let mut captured: Vec<bench_support::GdnLayerCapture> = Vec::new();

            let mut cmd = self.engine.queue.new_command_buffer();
            let mut enc = cmd.new_compute_command_encoder();
            let mut seg_start = std::time::Instant::now();

            for layer_i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(layer_i) {
                    continue;
                }
                let compact_idx = active_layer_idx;
                active_layer_idx += 1;
                let is_linear = matches!(
                    &self.engine.layer_weights[compact_idx].0,
                    MetalLayerAttnWeights::Linear(_)
                );

                if is_linear {
                    self.encode_gdn_prefill_before_recurrence(enc, compact_idx, layer_i, &cfg, m);

                    // ---- GDN isolation boundary: close the pre-GDN (non-GDN) segment ----
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    timing.non_gdn_ms += seg_start.elapsed().as_secs_f64() * 1000.0;

                    let gdn_cmd = self.engine.queue.new_command_buffer();
                    let gdn_enc = gdn_cmd.new_compute_command_encoder();
                    let gdn_start = std::time::Instant::now();

                    self.encode_gdn_prefill_recurrence(
                        gdn_enc,
                        compact_idx,
                        linear_idx,
                        &cfg,
                        n,
                        chunked_enabled,
                    );

                    gdn_enc.end_encoding();
                    gdn_cmd.commit();
                    gdn_cmd.wait_until_completed();
                    timing.gdn_ms += gdn_start.elapsed().as_secs_f64() * 1000.0;
                    timing.num_gdn_layers += 1;

                    // gdn175 S1b: read-only capture, right after this layer's GDN
                    // command buffer completed and before the next GDN layer's
                    // materialize kernel can overwrite the shared gdn_chunk scratch
                    // (see capture_gdn_chunk_scratch's doc comment for the safety
                    // argument). `linear_idx` here is still this layer's index —
                    // it only increments at the bottom of this branch.
                    for req in capture {
                        if req.linear_idx == linear_idx {
                            captured.push(self.capture_gdn_chunk_scratch(
                                linear_idx,
                                req.value_head,
                                n,
                            ));
                        }
                    }

                    // ---- reopen a fresh segment for the rest of this layer + subsequent layers ----
                    cmd = self.engine.queue.new_command_buffer();
                    enc = cmd.new_compute_command_encoder();
                    seg_start = std::time::Instant::now();

                    self.encode_gdn_prefill_after_recurrence(enc, compact_idx, layer_i, &cfg, m);
                    linear_idx += 1;
                } else {
                    self.encode_full_attention_prefill_layer(
                        enc,
                        compact_idx,
                        layer_i,
                        full_idx,
                        &cfg,
                        n,
                        m,
                        start_pos,
                    );
                    full_idx += 1;
                }
            }

            // Close the trailing non-GDN segment (mirrors the emit_logits=false early
            // return in forward_prefill_batched_chunk: still commit + advance the
            // session cursor, just never compute logits).
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            timing.non_gdn_ms += seg_start.elapsed().as_secs_f64() * 1000.0;

            self.session.set_position(start_pos + n);
            (timing, captured)
        }

        /// gdn175 S1b: read-only capture of one (GDN layer, value head)'s
        /// materialized per-token chunkwise-scan inputs — q/k/v/beta/log_alpha —
        /// off the session's shared `gdn_chunk` scratch buffers, de-chunked from
        /// `[chunks, heads, C, dim]` back to `[n_tokens, dim]` token order.
        ///
        /// SAFETY / correctness argument (verified against
        /// `forward/shaders/qwen35.metal`): `sc.q`/`sc.k`/`sc.v`/`sc.beta_log_alpha`
        /// are written ONLY by `gdn_chunk_materialize_c32` and read (`device const`,
        /// never written) by `gdn_chunk_solve_c32` and
        /// `gdn_chunk_residual_output_c32`; `gdn_chunk_conv_buf_update_c32`,
        /// `gdn_chunk_state_update_c32`, and `gdn_chunk_norm_silu_c32` never touch
        /// them at all. So once this layer's `gdn_cmd.wait_until_completed()` has
        /// returned (the call site, immediately before this method runs), these
        /// four buffers hold their final values for this layer and stay untouched
        /// until the NEXT GDN layer's own materialize dispatch reuses the same
        /// session-owned scratch. Reading them here is a pure CPU-side copy off a
        /// StorageModeShared buffer with no GPU work in flight — it cannot perturb
        /// the dispatch this call is capturing.
        ///
        /// De-chunking: `dispatch_gdn_chunk_materialize_c32` writes row
        /// `((chunk*num_value_heads + head)*GDN_CHUNK_SIZE + j)*dim + d` for token
        /// `chunk*GDN_CHUNK_SIZE + j` (see `qwen35.metal`'s `head_row` computation);
        /// only the LAST active chunk is short (materialize's `ci = min(chunk_size,
        /// n_tokens - chunk_base)` loop bound leaves the tail rows at their
        /// zero-initialized default). Reading exactly `rows_in_chunk * dim`
        /// contiguous elements per chunk therefore already skips the zero-padded
        /// tail rows and yields the valid tokens in order — no separate reshuffle
        /// needed after concatenating the per-chunk reads.
        ///
        /// `log_alpha` is returned RAW (natural-log, per-token, NOT cumulative) —
        /// the S1b spec's `alpha[t] = exp(log_alpha[t])` transform is left to the
        /// caller so the oracle-feeding step stays visibly separate from capture.
        #[cfg(feature = "bench-internals")]
        fn capture_gdn_chunk_scratch(
            &self,
            linear_idx: usize,
            value_head: usize,
            n_tokens: usize,
        ) -> bench_support::GdnLayerCapture {
            let cfg = &self.engine.config;
            let num_vh = cfg.linear_num_value_heads();
            let kd = cfg.linear_key_head_dim;
            let vd = cfg.linear_value_head_dim;
            assert!(
                value_head < num_vh,
                "capture_gdn_chunk_scratch: value_head={value_head} out of range \
                 (num_value_heads={num_vh})"
            );
            assert!(
                n_tokens > 0,
                "capture_gdn_chunk_scratch: n_tokens must be > 0"
            );

            let num_chunks_active = n_tokens.div_ceil(GDN_CHUNK_SIZE);
            let tail = n_tokens - (num_chunks_active - 1) * GDN_CHUNK_SIZE;

            let sc = &self.session.activations.gdn_chunk;
            let mut q = Vec::with_capacity(n_tokens * kd);
            let mut k = Vec::with_capacity(n_tokens * kd);
            let mut v = Vec::with_capacity(n_tokens * vd);
            let mut beta = Vec::with_capacity(n_tokens);
            let mut log_alpha = Vec::with_capacity(n_tokens);

            for chunk in 0..num_chunks_active {
                let rows = if chunk + 1 == num_chunks_active {
                    tail
                } else {
                    GDN_CHUNK_SIZE
                };
                let base_row = (chunk * num_vh + value_head) * GDN_CHUNK_SIZE;
                // SAFETY: see this method's doc comment — GPU completed, read-only,
                // no concurrent write to sc.q/k/v/beta_log_alpha at this point.
                unsafe {
                    q.extend_from_slice(&read_buffer_offset(&sc.q, base_row * kd, rows * kd));
                    k.extend_from_slice(&read_buffer_offset(&sc.k, base_row * kd, rows * kd));
                    v.extend_from_slice(&read_buffer_offset(&sc.v, base_row * vd, rows * vd));
                    let bla = read_buffer_offset(&sc.beta_log_alpha, base_row * 2, rows * 2);
                    for pair in bla.chunks_exact(2) {
                        beta.push(pair[0]);
                        log_alpha.push(pair[1]);
                    }
                }
            }

            debug_assert_eq!(q.len(), n_tokens * kd);
            debug_assert_eq!(k.len(), n_tokens * kd);
            debug_assert_eq!(v.len(), n_tokens * vd);
            debug_assert_eq!(beta.len(), n_tokens);
            debug_assert_eq!(log_alpha.len(), n_tokens);

            bench_support::GdnLayerCapture {
                linear_idx,
                value_head,
                length: n_tokens,
                dk: kd,
                dv: vd,
                q,
                k,
                v,
                beta,
                log_alpha,
            }
        }

        /// MTP greedy decode (LATTICE_MTP=1, greedy only).
        ///
        /// Each round drafts one extra token via MTP, verifies two tokens with the target
        /// model, accepts the draft if the target agrees, and rolls back and uses the
        /// target's token otherwise.
        ///
        /// The default (sequential) verifier runs through the shared speculative driver,
        /// which applies the shared decode policy to the verified, committed tokens only.
        /// The batch-GEMM verifier (`LATTICE_MTP_BATCH` on, no adapter loaded) is a legacy
        /// experimental route that keeps its own loop: it is selected exactly as before,
        /// never falls back to the sequential verifier, and records
        /// [`SpeculativeRoute::MtpBatchGemmLegacy`] so a request on it is distinguishable
        /// from shared-route coverage.
        fn generate_greedy_mtp(
            &mut self,
            prefill_logits: &[f32],
            prompt_len: usize,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> Result<GenerateOutput, crate::error::InferenceError> {
            let use_batch = super::use_batch_gemm_verifier(
                crate::env_switch_enabled("LATTICE_MTP_BATCH"),
                self.lora.is_some(),
            );
            if use_batch {
                return Ok(self.generate_greedy_mtp_batch_gemm_legacy(
                    prefill_logits,
                    prompt_len,
                    tokenizer,
                    gen_cfg,
                ));
            }
            self.generate_speculative_shared(
                SpeculativeMetrics::mtp(),
                prefill_logits,
                prompt_len,
                tokenizer,
                gen_cfg,
            )
        }

        /// GDN-first self-speculative decode.
        ///
        /// Drafts K tokens via GDN-only forwards (cheap O(n) recurrence, no KV cache
        /// writes), then verifies with the full model (GDN+GQA) in batch. Accepts the
        /// longest agreeing prefix and rolls back GDN state on partial rejection. The
        /// committed tokens go through the shared speculative driver.
        ///
        /// Activated by `LATTICE_SELF_SPEC=1`. Requires greedy decode (temperature <= 0,
        /// top_k <= 1). Requires `gdn_checkpoints` to be allocated in the session.
        fn generate_greedy_self_spec(
            &mut self,
            prefill_logits: &[f32],
            prompt_len: usize,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> Result<GenerateOutput, crate::error::InferenceError> {
            self.generate_speculative_shared(
                SpeculativeMetrics::self_spec(),
                prefill_logits,
                prompt_len,
                tokenizer,
                gen_cfg,
            )
        }

        /// Runs a request on a shared speculative route: the decoder session hands the
        /// driver verified rounds and the driver applies the decode policy to them.
        fn generate_speculative_shared(
            &mut self,
            metrics: SpeculativeMetrics,
            prefill_logits: &[f32],
            prompt_len: usize,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> Result<GenerateOutput, crate::error::InferenceError> {
            let eos_token_id = self.engine.config.eos_token_id;
            let mut session = crate::decoder::qwen_metal::QwenMetalSpeculativeSession::new(
                self,
                metrics,
                prefill_logits,
            );
            crate::decoder::qwen_metal::run_speculative_direct(
                &mut session,
                gen_cfg,
                prompt_len,
                eos_token_id,
                tokenizer,
            )
        }

        /// Records which speculative route the current request took and, under the
        /// route's verbose switch, prints the route marker line.
        pub(crate) fn record_speculative_route(
            &mut self,
            route: SpeculativeRoute,
            trace: crate::decoder::SpeculativeTrace,
        ) {
            self.session.speculative_route = Some(SpeculativeRouteRecord { route, trace });
            let (switch, tag) = match route {
                SpeculativeRoute::SelfSpecShared => ("LATTICE_SELF_SPEC_VERBOSE", "SELF_SPEC"),
                SpeculativeRoute::MtpShared | SpeculativeRoute::MtpBatchGemmLegacy => {
                    ("LATTICE_MTP_VERBOSE", "MTP")
                }
            };
            if crate::env_switch_enabled(switch) {
                eprintln!(
                    "[{tag}] route={} shared_policy={} driver_rounds={} offered={}",
                    route.label(),
                    route.uses_shared_policy(),
                    trace.rounds,
                    trace.offered,
                );
            }
        }

        /// One verification round of a shared speculative route. `pending` is the token
        /// the target already chose for the next position; `room` is how many more tokens
        /// the request may commit. The returned round holds only tokens the target model
        /// evaluated and agreed with, plus the target's prediction that follows them.
        pub(crate) fn speculative_round(
            &mut self,
            metrics: &mut SpeculativeMetrics,
            pending: u32,
            room: usize,
            is_stop: &dyn Fn(u32) -> bool,
        ) -> crate::decoder::VerifiedRound {
            match metrics {
                SpeculativeMetrics::Mtp(m) => self.mtp_sequential_round(m, pending, room, is_stop),
                SpeculativeMetrics::SelfSpec(m) => self.self_spec_round(m, pending, room, is_stop),
            }
        }

        /// What the loops this driver replaced did once their last round ended: the MTP
        /// route refreshes `last_pre_final_hidden` and both routes print their verbose
        /// counters.
        pub(crate) fn finish_speculative(&mut self, metrics: &SpeculativeMetrics) {
            match metrics {
                SpeculativeMetrics::Mtp(metrics) => {
                    self.session.last_pre_final_hidden = {
                        let pos = self.session.kv_cache.seq_len;
                        if pos > 0 {
                            // Re-run one step to get fresh hidden for any downstream caller.
                            // (verify_tokens_batched already left last_pre_final_hidden current.)
                            self.session.last_pre_final_hidden.clone()
                        } else {
                            self.session.last_hidden_cursor = None;
                            vec![0.0f32; self.engine.config.hidden_size]
                        }
                    };

                    if crate::env_switch_enabled("LATTICE_MTP_VERBOSE") {
                        eprintln!(
                            "[MTP] rounds={} mtp_fwd={} verify={} accepted_extra={} rollbacks={} fallbacks={} mtp_ms={:.1} verify_ms={:.1} rb_ms={:.1} tap={}",
                            metrics.rounds,
                            metrics.mtp_forwards,
                            metrics.verify_calls,
                            metrics.accepted_extra_tokens,
                            metrics.verify_calls - metrics.accepted_extra_tokens,
                            metrics.fallback_tokens,
                            metrics.mtp_ms,
                            metrics.verify_ms,
                            metrics.rollback_ms,
                            self.mtp_hidden_tap().map_or("none", MtpHiddenTap::label),
                        );
                    }
                }
                SpeculativeMetrics::SelfSpec(metrics) => {
                    if crate::env_switch_enabled("LATTICE_SELF_SPEC_VERBOSE") {
                        eprintln!(
                            "[SELF_SPEC] rounds={} draft_fwd={} verify={} accepted_extra={} fallbacks={} draft_ms={:.1} verify_ms={:.1} rb_ms={:.1}",
                            metrics.rounds,
                            metrics.draft_forwards,
                            metrics.verify_calls,
                            metrics.accepted_extra_tokens,
                            metrics.fallback_tokens,
                            metrics.draft_ms,
                            metrics.verify_ms,
                            metrics.rollback_ms,
                        );
                    }
                }
            }
        }

        /// One MTP round under the sequential verifier.
        ///
        /// Drafts one token for the position after `pending_token`, verifies
        /// `[pending_token, draft]` with the target, and returns the tokens the target
        /// committed. A rejected draft is rolled back here and never appears in the
        /// returned round. The stop-token contract is `mtp_greedy_round`'s: a stop token
        /// is excluded from `committed` and reported as `next`.
        fn mtp_sequential_round(
            &mut self,
            metrics: &mut MetalMtpDecodeMetrics,
            pending_token: u32,
            room: usize,
            is_stop: &dyn Fn(u32) -> bool,
        ) -> crate::decoder::VerifiedRound {
            use crate::decoder::VerifiedRound;
            let pos = self.session.kv_cache.seq_len;
            if pos >= self.session.kv_cache.max_cache_len.saturating_sub(2) {
                // Not enough room for 2 more tokens: commit pending and stop.
                return VerifiedRound {
                    committed: vec![pending_token],
                    next: None,
                    cache_full: true,
                };
            }

            // --- MTP draft phase ---
            // `mtp_forward_one` writes exactly one MTP KV cache row (for
            // `pending_token`) and advances `mtp.cache.seq_len` past it before this
            // round's verify/rollback machinery runs. Snapshot the pre-draft cursor
            // so a rejection restores the MTP cache to its state as of the top of
            // this round, not as of after the draft's own write: `verify_tokens_batched`
            // captures `mtp_base_seq_len` from whatever the cursor reads when *it* runs,
            // which is already past that row (#1341).
            let mtp_seq_len_pre_draft = self.session.mtp.as_ref().map(|m| m.cache.seq_len);
            let t_mtp = std::time::Instant::now();
            let draft = self.mtp_forward_one(pending_token, pos);
            metrics.mtp_ms += t_mtp.elapsed().as_secs_f64() * 1000.0;
            metrics.mtp_forwards += 1;

            // NOTE: We intentionally do NOT short-circuit on draft EOS here.
            // The draft's EOS guess must be verified against the target model.
            // If the target agrees, the round commits pending and reports the stop.
            // If the target disagrees, the reject path replaces the draft as usual.
            // Short-circuiting before verify caused concern #2 of #237: a wrong
            // draft-EOS would silently truncate generation early.

            // --- Verify phase (sequential verifier) ---
            let t_verify = std::time::Instant::now();
            let verify_result = self.verify_tokens_batched(&[pending_token, draft.token_id], pos);
            let Ok(verify_out) = verify_result else {
                // Fallback: commit pending only; the draft stays unverified and is only
                // a candidate, so the driver checks it before it becomes pending.
                self.session.rewind_position(pos + 1);
                metrics.fallback_tokens += 1;
                return VerifiedRound {
                    committed: vec![pending_token],
                    next: Some(draft.token_id),
                    cache_full: false,
                };
            };
            metrics.verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
            metrics.verify_calls += 1;
            // Correct the base the verifier just captured: it read the MTP cursor
            // after the draft phase already advanced it by one row, so restoring
            // `base_mtp + slot` on rejection would double-count that row and leave
            // the cursor one slot past the last one `mtp_forward_one` actually wrote.
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.mtp_base_seq_len = mtp_seq_len_pre_draft;
            }

            // Route the accept/reject decision through `rejection_sample_draft` so
            // the live MTP loop and the trait-level `mtp_verify_draft` share the
            // same verifier implementation (ADR-050). This is the GREEDY MTP route
            // (entered only when top_k<=1 && temperature<=0.0), so verification uses
            // `greedy=true`: argmax-only acceptance that is deterministic and
            // token-for-token equivalent to plain greedy `generate()`. Passing
            // `greedy=false` here drove a clock-seeded RNG, making greedy MTP output
            // non-deterministic and not argmax-equivalent (#237). In greedy mode the
            // verifier never reads `draft_logits` (it may even be `&[]` per the
            // contract), but we still hand it the draft's own logits: the value is
            // ignored on the greedy path and this keeps `draft.logits` a live read.
            //   draft_tokens          = [draft.token_id]
            //   draft_logits          = [draft.logits]        (ignored in greedy mode)
            //   initial_target_logits = verify_out.logits[0]  (predicts draft position)
            //   target_logits         = [verify_out.logits[1]] (bonus on full accept)
            // temperature is ignored on the greedy=true path (argmax is invariant to
            // positive scaling), so any finite positive placeholder is safe here:
            // this route is entered only when gen_cfg.temperature <= 0.0, so the real
            // config value cannot satisfy `> 0` validation anyway.
            let Ok(rs) = crate::speculative::rejection_sample_draft(
                &[draft.token_id],
                std::slice::from_ref(&draft.logits),
                &verify_out.logits[0],
                std::slice::from_ref(&verify_out.logits[1]),
                true,
                1.0,
                None,
            ) else {
                // Fallback: commit pending only, as for a failed verify.
                self.session.rewind_position(pos + 1);
                metrics.fallback_tokens += 1;
                return VerifiedRound {
                    committed: vec![pending_token],
                    next: Some(draft.token_id),
                    cache_full: false,
                };
            };

            // GPU state mutations must happen before the pure decision function:
            // - Accept: clear batch_repair_token (rollback won't run this round),
            //   and append the MTP-cache row `mtp_forward_one` never writes for
            //   the accepted draft token itself.
            // - Reject: roll back KV cache and GDN state to pos+1.
            let accepted = rs.accepted_count == 1;
            if accepted {
                if let Some(ref mut p) = self.session.gdn_checkpoints {
                    p.batch_repair_token = None;
                }
                // lattice#1396: `mtp_forward_one` above wrote only the row for
                // `pending_token` at `pos`; a full accept commits `draft.token_id`
                // at `pos + 1` without ever giving it its own MTP-cache row, so
                // the next round's `mtp_forward_one(bonus_token, pos + 2)` lands
                // one physical slot behind its RoPE position (a positional hole
                // per accepted transition). `mtp_prefill_append` is the same
                // K/V-only append primitive `mtp_prefill` uses to backfill
                // historical prompt positions: a prefilled position is only
                // ever a future key, exactly this case. Pair the accepted
                // token's embedding with `first_pre_final_hidden`, the verify
                // pass's pre-final hidden for `pending_token`'s own position
                // (the hidden state that predicted the accepted draft), matching
                // the pairing `mtp_prefill_append`'s own doc comment establishes.
                self.mtp_prefill_append(
                    draft.token_id,
                    &verify_out.first_pre_final_hidden,
                    pos + 1,
                );
                metrics.accepted_extra_tokens += 1;
            } else {
                let t_rb = std::time::Instant::now();
                let _ = self.rollback_speculative_state_to(pos + 1);
                metrics.rollback_ms += t_rb.elapsed().as_secs_f64() * 1000.0;
                // The sequential verifier's rollback restores GDN state and the KV
                // cache cursor but not `last_pre_final_hidden`, which
                // `verify_tokens_batched` last wrote for the rejected draft token.
                // Restore it to the hidden state that predicted the target's real
                // replacement, or the next MTP draft is fed the wrong input.
                self.session.last_pre_final_hidden = verify_out.first_pre_final_hidden.clone();
                self.session.mark_pre_final_hidden(pos + 1);
            }

            // Delegate the pure emit/stop/continue decision to `mtp_greedy_round`.
            // `bonus_token` meaning depends on the accept/reject outcome:
            //   accept -> argmax(verify_out.logits[1])  (target pred for pos+2)
            //   reject -> argmax(verify_out.logits[0])  (target's replacement at pos+1)
            // Both are encoded in rs.bonus_token by `rejection_sample_draft`.
            let bonus_token = rs.bonus_token.unwrap_or(0);
            // A round can commit up to two tokens (pending and an accepted draft), and
            // the request may have room for fewer: clip to `room` so greedy MTP never
            // exceeds `max_new_tokens`. Plain greedy `generate()` enters only
            // `max_new_tokens - 1` decode steps, so it stops at the cap even when the
            // next token would be EOS.
            match super::mtp_greedy_round(
                pending_token,
                draft.token_id,
                accepted,
                bonus_token,
                is_stop,
            ) {
                super::MtpRoundOutcome::EmitAndStop(mut tokens) => {
                    // Stop-token contract (#613): `tokens` holds only the committed
                    // (non-stop) tokens for this round. The stop token that ended it is
                    // reported as `next`, and the driver counts it as a stop only when the
                    // request still had room for it (#632).
                    let stop = if accepted && is_stop(draft.token_id) {
                        draft.token_id
                    } else {
                        bonus_token
                    };
                    tokens.truncate(room);
                    VerifiedRound {
                        committed: tokens,
                        next: Some(stop),
                        cache_full: false,
                    }
                }
                super::MtpRoundOutcome::EmitAndContinue {
                    mut emit,
                    next_pending,
                } => {
                    if emit.len() < room {
                        metrics.rounds += 1;
                    }
                    emit.truncate(room);
                    VerifiedRound {
                        committed: emit,
                        next: Some(next_pending),
                        cache_full: false,
                    }
                }
            }
        }

        /// One GDN-first self-speculative round: drafts up to `SELF_SPEC_MAX_DRAFT`
        /// tokens with GDN-only forwards, verifies `[pending] ++ drafts` with the full
        /// model, and returns `pending` plus the longest draft prefix the target agreed
        /// with. A rejected draft is rolled back here and never appears in the returned
        /// round.
        fn self_spec_round(
            &mut self,
            metrics: &mut SelfSpecMetrics,
            pending_token: u32,
            room: usize,
            is_stop: &dyn Fn(u32) -> bool,
        ) -> crate::decoder::VerifiedRound {
            use crate::decoder::VerifiedRound;
            // Shared first-wins argmax helper (ADR-080 C3, #783) rather than a
            // hand-rolled `max_by` closure, which kept the LAST tied maximum
            // instead of the engine-wide-contract first tied maximum.
            let argmax_logits = crate::sampling::argmax_f32_first_wins;

            let pos = self.session.kv_cache.seq_len;
            if pos + SELF_SPEC_MAX_DRAFT + 1 >= self.session.kv_cache.max_cache_len {
                return VerifiedRound {
                    committed: vec![pending_token],
                    next: None,
                    cache_full: true,
                };
            }

            // --- Draft phase: K GDN-only forwards ---
            let t_draft = std::time::Instant::now();
            // Self-spec uses slot-based rollback: `verify_tokens_batched` populates one
            // slot per verify token (slot k = state after k tokens processed by the full
            // model from the *pre-draft* base). We do NOT take a `batch_repair_token`
            // shortcut here: that path replays only the pending token on rejection,
            // leaving GDN state inconsistent with the KV cache after partial acceptance.
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.active_base_seq_len = Some(pos);
                p.mtp_base_seq_len = None;
                p.batch_repair_token = None;
            }
            // Slot 0 reserves the pre-draft GDN state. `forward_step_gdn_only` below
            // mutates the live GDN buffers; if we did not capture pre-draft state here,
            // the slot-based rollback (which expects verify_tokens_batched's slots
            // 1..=K to be full-model forwards from this pre-draft base) would be
            // computed against a contaminated reference. With no checkpoint pool, or a
            // failed capture, fall through to a single-token decode.
            if self.session.gdn_checkpoints.is_none()
                || self
                    .checkpoint_gdn_to_slot(0, GdnStateTrafficScope::Decode)
                    .is_err()
            {
                let logits = self
                    .forward_step_inner(
                        pending_token,
                        pos,
                        false,
                        crate::forward::signpost::Scope::Decode,
                    )
                    .logits;
                let next = argmax_logits(&logits);
                if !is_stop(next) && 1 < room {
                    metrics.fallback_tokens += 1;
                }
                return VerifiedRound {
                    committed: vec![pending_token],
                    next: Some(next),
                    cache_full: false,
                };
            }

            let mut draft_tokens: Vec<u32> = Vec::with_capacity(SELF_SPEC_MAX_DRAFT);
            let first_draft_logits = self.forward_step_gdn_only(pending_token, pos);
            let first_draft = argmax_logits(&first_draft_logits);
            metrics.draft_forwards += 1;
            if !is_stop(first_draft) {
                draft_tokens.push(first_draft);
                let mut cur_draft = first_draft;
                for _ in 1..SELF_SPEC_MAX_DRAFT {
                    if is_stop(cur_draft) {
                        break;
                    }
                    let d_pos = pos + draft_tokens.len();
                    let logits = self.forward_step_gdn_only(cur_draft, d_pos);
                    let next = argmax_logits(&logits);
                    metrics.draft_forwards += 1;
                    draft_tokens.push(next);
                    cur_draft = next;
                }
            }
            metrics.draft_ms += t_draft.elapsed().as_secs_f64() * 1000.0;

            // --- Verify phase: full model over [pending_token] ++ draft_tokens ---
            let mut verify_input: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
            verify_input.push(pending_token);
            verify_input.extend_from_slice(&draft_tokens);

            // Restore the pre-draft GDN state from slot 0 before verification. The
            // GDN-only draft forwards above mutated the live GDN buffers; without
            // this restore, `verify_tokens_batched` would compose full-model forwards
            // on top of the GDN-only draft mutations, producing slots that do NOT
            // match the canonical "pre-draft + N full-model tokens" semantics that
            // slot-based rollback relies on.
            let restore_ok = self
                .restore_gdn_slot_blocking(0, GdnStateTrafficScope::Decode)
                .is_ok();

            let t_verify = std::time::Instant::now();
            let Ok(verify_out) = (if restore_ok {
                self.verify_tokens_batched(&verify_input, pos)
            } else {
                Err(crate::error::InferenceError::Inference(
                    "GDN pre-verify restore failed".into(),
                ))
            }) else {
                // Verification failed: commit pending; the first draft stays unverified
                // and is only a candidate for the driver to check.
                self.session.rewind_position(pos + 1);
                metrics.fallback_tokens += 1;
                return VerifiedRound {
                    committed: vec![pending_token],
                    next: Some(argmax_logits(&first_draft_logits)),
                    cache_full: false,
                };
            };
            metrics.verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
            metrics.verify_calls += 1;

            // verify_out.logits[i] = full-model output after processing verify_input[i].
            // Logits[0] predicts what comes after pending_token.
            // Compare logits[i] argmax with draft_tokens[i] for i in 0..draft_len.

            // pending_token is always accepted (full model processed it above).
            let mut committed: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
            committed.push(pending_token);

            let mut accepted_drafts = 0usize;
            let mut rejection_next: Option<u32> = None;

            for (i, &draft) in draft_tokens.iter().enumerate() {
                if committed.len() >= room {
                    break;
                }
                let target = argmax_logits(&verify_out.logits[i]);
                if target == draft {
                    if is_stop(draft) {
                        // Stop-token contract (#613): the accepted draft is itself a
                        // stop and is never committed. Room was available (checked
                        // above), so the driver reports a genuine stop.
                        return VerifiedRound {
                            committed,
                            next: Some(draft),
                            cache_full: false,
                        };
                    }
                    committed.push(draft);
                    accepted_drafts += 1;
                    metrics.accepted_extra_tokens += 1;
                } else {
                    rejection_next = Some(target);
                    break;
                }
            }

            if let Some(next_token) = rejection_next {
                // Partial or full rejection: roll back GDN/KV to accepted prefix boundary.
                let t_rb = std::time::Instant::now();
                let _ = self.rollback_speculative_state_to(pos + accepted_drafts + 1);
                metrics.rollback_ms += t_rb.elapsed().as_secs_f64() * 1000.0;
                if !is_stop(next_token) && committed.len() < room {
                    metrics.rounds += 1;
                }
                return VerifiedRound {
                    committed,
                    next: Some(next_token),
                    cache_full: false,
                };
            }

            // Full acceptance: next pending = full-model prediction after last verified token.
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.batch_repair_token = None;
            }
            let last_idx = draft_tokens
                .len()
                .min(verify_out.logits.len().saturating_sub(1));
            let next_pending = argmax_logits(&verify_out.logits[last_idx]);
            metrics.rounds += 1;
            VerifiedRound {
                committed,
                next: Some(next_pending),
                cache_full: false,
            }
        }

        /// The legacy batch-GEMM MTP loop (`LATTICE_MTP_BATCH` on, no adapter loaded).
        ///
        /// Kept as its own loop: the batch-GEMM verifier is an experimental route that
        /// is not part of the shared decode policy, and it never falls back to the
        /// sequential verifier. Its MoE panic is the verifier's own and is left alone.
        fn generate_greedy_mtp_batch_gemm_legacy(
            &mut self,
            prefill_logits: &[f32],
            prompt_len: usize,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> GenerateOutput {
            self.record_speculative_route(
                SpeculativeRoute::MtpBatchGemmLegacy,
                crate::decoder::SpeculativeTrace::default(),
            );
            let cfg = self.engine.config.clone();

            // Mirror canonical plain greedy generation: max_new_tokens == 0
            // means "generate nothing".  Return before sampling so we never emit a
            // token the caller did not ask for — including the case-A prefill-EOS path
            // below, which would otherwise emit one stop token for a zero budget.
            if gen_cfg.max_new_tokens == 0 {
                return GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: Some(StopReason::Length),
                    token_logprobs: vec![],
                };
            }

            // Greedy argmax from prefill logits. Shared first-wins helper
            // (ADR-080 C3, #783) rather than a hand-rolled `max_by`, which
            // keeps the LAST tied maximum instead of the engine-wide-contract
            // first tied maximum.
            let pending_first = crate::sampling::argmax_f32_first_wins(prefill_logits);

            let is_stop = |id: u32| id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id);

            if is_stop(pending_first) {
                // Stop-token contract (#613): the terminating token is excluded
                // from token_ids/text. max_new_tokens == 0 already returned above,
                // so budget was available for this token — this is a genuine
                // stop, not one clipped by the cap.
                return GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: true,
                    stop_reason: Some(StopReason::Eos),
                    token_logprobs: vec![],
                };
            }

            let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
            let mut pending_token = pending_first;
            let mut metrics = MetalMtpDecodeMetrics::default();
            let mut stopped = false;
            let mut stop_reason = StopReason::Length;

            while generated_ids.len() < gen_cfg.max_new_tokens {
                let pos = self.session.kv_cache.seq_len;
                if pos >= self.session.kv_cache.max_cache_len.saturating_sub(2) {
                    // Not enough room for 2 more tokens — flush pending and stop.
                    generated_ids.push(pending_token);
                    stop_reason = StopReason::KvFull;
                    break;
                }

                // --- MTP draft phase ---
                // `mtp_forward_one` writes exactly one MTP KV cache row (for
                // `pending_token`) and advances `mtp.cache.seq_len` past it before this
                // round's verify/rollback machinery runs. Snapshot the pre-draft cursor
                // so a rejection restores the MTP cache to its state as of the top of
                // this round, not as of after the draft's own write — `verify_tokens_batched`/
                // `verify_tokens_batch_gemm` capture `mtp_base_seq_len` from whatever the
                // cursor reads when *they* run, which is already past that row (#1341).
                let mtp_seq_len_pre_draft = self.session.mtp.as_ref().map(|m| m.cache.seq_len);
                let t_mtp = std::time::Instant::now();
                let draft = self.mtp_forward_one(pending_token, pos);
                metrics.mtp_ms += t_mtp.elapsed().as_secs_f64() * 1000.0;
                metrics.mtp_forwards += 1;

                // NOTE: We intentionally do NOT short-circuit on draft EOS here.
                // The draft's EOS guess must be verified against the target model.
                // If the target agrees → emit pending + EOS, stop.
                // If the target disagrees → reject path replaces the draft as usual.
                // Short-circuiting before verify caused concern #2 of #237: a wrong
                // draft-EOS would silently truncate generation early.

                // --- Verify phase (LATTICE_MTP_BATCH=1 selects batch-GEMM verifier,
                // unless an adapter is loaded: that verifier applies none) ---
                let t_verify = std::time::Instant::now();
                let verify_result =
                    self.verify_tokens_batch_gemm(&[pending_token, draft.token_id], pos);
                let Ok(verify_out) = verify_result else {
                    // Fallback: accept pending, advance normally.
                    self.session.rewind_position(pos + 1);
                    generated_ids.push(pending_token);
                    metrics.fallback_tokens += 1;
                    pending_token = draft.token_id;
                    continue;
                };
                metrics.verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
                metrics.verify_calls += 1;
                // Correct the base the verifier just captured: it read the MTP cursor
                // after the draft phase already advanced it by one row, so restoring
                // `base_mtp + slot` on rejection would double-count that row and leave
                // the cursor one slot past the last one `mtp_forward_one` actually wrote.
                if let Some(ref mut p) = self.session.gdn_checkpoints {
                    p.mtp_base_seq_len = mtp_seq_len_pre_draft;
                }

                // Route the accept/reject decision through `rejection_sample_draft` so
                // the live MTP loop and the trait-level `mtp_verify_draft` share the
                // same verifier implementation (ADR-050). This is the GREEDY MTP loop
                // (entered only when top_k<=1 && temperature<=0.0), so verification uses
                // `greedy=true`: argmax-only acceptance that is deterministic and
                // token-for-token equivalent to plain greedy `generate()`. Passing
                // `greedy=false` here drove a clock-seeded RNG, making greedy MTP output
                // non-deterministic and not argmax-equivalent (#237). In greedy mode the
                // verifier never reads `draft_logits` (it may even be `&[]` per the
                // contract), but we still hand it the draft's own logits: the value is
                // ignored on the greedy path and this keeps `draft.logits` a live read.
                //   draft_tokens          = [draft.token_id]
                //   draft_logits          = [draft.logits]        (ignored in greedy mode)
                //   initial_target_logits = verify_out.logits[0]  (predicts draft position)
                //   target_logits         = [verify_out.logits[1]] (bonus on full accept)
                // temperature is ignored on the greedy=true path (argmax is invariant to
                // positive scaling), so any finite positive placeholder is safe here —
                // this loop is entered only when gen_cfg.temperature <= 0.0 (see comment
                // above), so the real config value cannot satisfy `> 0` validation anyway.
                let Ok(rs) = crate::speculative::rejection_sample_draft(
                    &[draft.token_id],
                    std::slice::from_ref(&draft.logits),
                    &verify_out.logits[0],
                    std::slice::from_ref(&verify_out.logits[1]),
                    true,
                    1.0,
                    None,
                ) else {
                    // Fallback: accept pending, advance normally.
                    self.session.rewind_position(pos + 1);
                    generated_ids.push(pending_token);
                    metrics.fallback_tokens += 1;
                    pending_token = draft.token_id;
                    continue;
                };

                // GPU state mutations must happen before the pure decision function:
                // - Accept: clear batch_repair_token (rollback won't run this round),
                //   and append the MTP-cache row `mtp_forward_one` never writes for
                //   the accepted draft token itself.
                // - Reject: roll back KV cache and GDN state to pos+1.
                let accepted = rs.accepted_count == 1;
                if accepted {
                    if let Some(ref mut p) = self.session.gdn_checkpoints {
                        p.batch_repair_token = None;
                    }
                    // lattice#1396: `mtp_forward_one` above wrote only the row for
                    // `pending_token` at `pos`; a full accept commits `draft.token_id`
                    // at `pos + 1` without ever giving it its own MTP-cache row, so
                    // the next round's `mtp_forward_one(bonus_token, pos + 2)` lands
                    // one physical slot behind its RoPE position (a positional hole
                    // per accepted transition). `mtp_prefill_append` is the same
                    // K/V-only append primitive `mtp_prefill` uses to backfill
                    // historical prompt positions — a prefilled position is only
                    // ever a future key, exactly this case. Pair the accepted
                    // token's embedding with `first_pre_final_hidden`, the verify
                    // pass's pre-final hidden for `pending_token`'s own position
                    // (the hidden state that predicted the accepted draft), matching
                    // the pairing `mtp_prefill_append`'s own doc comment establishes.
                    self.mtp_prefill_append(
                        draft.token_id,
                        &verify_out.first_pre_final_hidden,
                        pos + 1,
                    );
                    metrics.accepted_extra_tokens += 1;
                } else {
                    let t_rb = std::time::Instant::now();
                    let _ = self.rollback_speculative_state_to(pos + 1);
                    metrics.rollback_ms += t_rb.elapsed().as_secs_f64() * 1000.0;
                    // The batch-GEMM verifier's own repair-replay restores
                    // `last_pre_final_hidden`, so nothing is restored here.
                }

                // Delegate the pure emit/stop/continue decision to `mtp_greedy_round`.
                // `bonus_token` meaning depends on the accept/reject outcome:
                //   accept → argmax(verify_out.logits[1])  (target pred for pos+2)
                //   reject → argmax(verify_out.logits[0])  (target's replacement at pos+1)
                // Both are encoded in rs.bonus_token by `rejection_sample_draft`.
                let bonus_token = rs.bonus_token.unwrap_or(0);
                // The loop guard (`generated_ids.len() < max_new_tokens`) guarantees at
                // least one slot remains here, but a round can emit up to three tokens
                // (case B: [pending, draft, bonus]).  Clip every emission to the
                // remaining budget so greedy MTP never exceeds `max_new_tokens` — plain
                // greedy `generate()` enters only `max_new_tokens - 1` decode steps, so
                // it stops at the cap even when the next token would be EOS.
                match super::mtp_greedy_round(
                    pending_token,
                    draft.token_id,
                    accepted,
                    bonus_token,
                    is_stop,
                ) {
                    super::MtpRoundOutcome::EmitAndStop(tokens) => {
                        let remaining = gen_cfg.max_new_tokens - generated_ids.len();
                        let emit_len = tokens.len().min(remaining);
                        generated_ids.extend_from_slice(&tokens[..emit_len]);
                        // Stop-token contract (#613): `tokens` holds only the committed
                        // (non-stop) tokens for this round — the stop token itself is
                        // never in it. #632: `emit_len == tokens.len()`
                        // alone is not sufficient — if the committed payload exactly
                        // fills `remaining`, the excluded stop token would have landed
                        // one position *beyond* `max_new_tokens`, so plain greedy would
                        // never have reached it either and reports Length, not Eos.
                        // `stopped` is therefore true only when every committed token
                        // was emitted AND at least one more slot remained for the
                        // (excluded) stop token itself — i.e. the stop was genuinely
                        // reached within budget, not merely coincident with the cap.
                        stopped = emit_len == tokens.len() && remaining > tokens.len();
                        if stopped {
                            stop_reason = StopReason::Eos;
                        }
                        break;
                    }
                    super::MtpRoundOutcome::EmitAndContinue { emit, next_pending } => {
                        let remaining = gen_cfg.max_new_tokens - generated_ids.len();
                        generated_ids.extend_from_slice(&emit[..emit.len().min(remaining)]);
                        if generated_ids.len() >= gen_cfg.max_new_tokens {
                            break;
                        }
                        pending_token = next_pending;
                    }
                }
                metrics.rounds += 1;
            }

            // Last update to pre_final_hidden from the verify output.
            self.session.last_pre_final_hidden = {
                let pos = self.session.kv_cache.seq_len;
                if pos > 0 {
                    // Re-run one step to get fresh hidden for any downstream caller.
                    // (verify_tokens_batched already left last_pre_final_hidden current.)
                    self.session.last_pre_final_hidden.clone()
                } else {
                    self.session.last_hidden_cursor = None;
                    vec![0.0f32; cfg.hidden_size]
                }
            };

            if crate::env_switch_enabled("LATTICE_MTP_VERBOSE") {
                eprintln!(
                    "[MTP] rounds={} mtp_fwd={} verify={} accepted_extra={} rollbacks={} fallbacks={} mtp_ms={:.1} verify_ms={:.1} rb_ms={:.1} tap={}",
                    metrics.rounds,
                    metrics.mtp_forwards,
                    metrics.verify_calls,
                    metrics.accepted_extra_tokens,
                    metrics.verify_calls - metrics.accepted_extra_tokens,
                    metrics.fallback_tokens,
                    metrics.mtp_ms,
                    metrics.verify_ms,
                    metrics.rollback_ms,
                    self.mtp_hidden_tap().map_or("none", MtpHiddenTap::label),
                );
            }

            let text = decode_tokens(tokenizer, &generated_ids);
            GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
                stop_reason: Some(stop_reason),
                token_logprobs: vec![],
            }
        }

        /// **Unstable**: greedy n-gram speculative generation over raw token ids.
        ///
        /// Owns the complete Metal lifecycle: validates the request before
        /// mutation, resets recurrent state, prefills the prompt once, then
        /// forwards every emitted non-EOS token at its absolute position. On
        /// success with a non-zero generation budget, the live KV/GDN state
        /// therefore represents `prompt_tokens.len() + generated.len()` tokens.
        /// For a valid prompt, a zero generation budget returns an empty vector
        /// without resetting existing state.
        ///
        /// # Errors
        ///
        /// Returns [`crate::error::InferenceError::Inference`] for an empty
        /// prompt or, for a non-zero generation budget, when the prompt plus
        /// budget exceeds [`Self::max_context`]. Returns
        /// [`crate::error::InferenceError::InvalidInput`] when a prompt token
        /// cannot index the configured embedding table. Admission errors occur
        /// before reset, prefill, allocation of generation history, or GPU work.
        ///
        /// `max_ngram` is clamped to
        /// [`crate::speculative::MAX_NGRAM_LIMIT`]; the n-gram search scans the
        /// prompt once per attempted length, so an unbounded value would let a
        /// caller spend arbitrary CPU before the first token.
        ///
        /// Returns [`crate::error::InferenceError::UnsupportedModel`] for a
        /// multi-token prompt on a model with MoE layers and no active LoRA
        /// adapter: that prompt would take the Metal batched prefill path, which
        /// has no MoE schedule. The rejection happens before GPU dispatch, in
        /// place of the process abort this entry point produced previously.
        /// Dense models, with or without a LoRA adapter, and LoRA-active MoE models
        /// (which prefill per token) are unaffected.
        ///
        /// [`Self::generate`], [`Self::generate_streaming`],
        /// [`Self::generate_streaming_with_cancel`], and
        /// [`Self::generate_with_speculation`] reach the same check through
        /// [`Self::try_forward_prefill`] and return the rejection as `Err`.
        /// [`Self::forward_prefill`] is the one remaining unwrapping wrapper,
        /// so callers using it still abort on this case.
        pub fn generate_with_speculation(
            &mut self,
            prompt_tokens: &[u32],
            max_new_tokens: usize,
            eos_token: u32,
            max_ngram: usize,
            max_draft: usize,
        ) -> Result<Vec<u32>, crate::error::InferenceError> {
            crate::model::qwen35::check_prompt_not_empty(prompt_tokens.len())?;
            crate::model::qwen35::check_prompt_ids_in_vocab(
                prompt_tokens,
                self.engine.config.vocab_size,
            )?;
            if max_new_tokens == 0 {
                return Ok(Vec::new());
            }
            crate::model::qwen35::check_context_budget(
                prompt_tokens.len(),
                None,
                max_new_tokens,
                self.max_context(),
            )?;

            self.reset_state();
            // Prefill through the fallible entry point, not the unwrapping
            // wrapper: the batched path has no MoE schedule, so an ordinary
            // multi-token prompt on an MoE model must come back as
            // `UnsupportedModel` rather than aborting this request-facing call.
            let prefill_logits = self.try_forward_prefill(prompt_tokens)?;
            crate::speculative::generate_with_speculation_from_prefill(
                prompt_tokens,
                max_new_tokens,
                eos_token,
                prefill_logits,
                |token, position| self.try_forward_step(token, position),
                max_ngram,
                max_draft,
            )
        }

        fn prepare_direct_generation(
            &self,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> Result<GenerationPreparation, crate::error::InferenceError> {
            prepare_generation(
                tokenizer,
                prompt,
                gen_cfg,
                self.engine.config.vocab_size,
                self.max_context(),
                GenerationEntryContract::MetalDirect,
            )
        }

        /// The teardown for a compact sampling route engaged by
        /// [`crate::decoder::qwen_metal::QwenMetalSession`], whether generation
        /// completed, was cancelled, or failed at prefill.
        pub(crate) fn disengage_compact_route(&mut self) {
            self.session.compact_topk = 0;
            self.session.compact_route = GpuTopkRoute::CpuFallback;
        }

        /// **Unstable**: generate text from a prompt; sampling parameters and output format may change.
        ///
        /// Generate text from a prompt.
        ///
        /// # Errors
        ///
        /// Returns `InferenceError::InvalidInput` if grammar-constrained decoding
        /// blocks every token (prefill or any decode step) — mirrors the CPU
        /// `generate` contract (#611).
        /// Returns `InferenceError::Inference` when the prompt plus requested decode
        /// cap exceeds [`Self::max_context`].
        /// Returns `InferenceError::UnsupportedModel` for a multi-token prompt on a
        /// model with MoE layers and no active LoRA adapter: prefill for that input
        /// is batched and the batched schedulers have no MoE schedule.
        pub fn generate(
            &mut self,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> Result<GenerateOutput, crate::error::InferenceError> {
            let plan = match self.prepare_direct_generation(prompt, tokenizer, gen_cfg)? {
                GenerationPreparation::Complete(output) => return Ok(output),
                GenerationPreparation::Ready(plan) => plan,
            };

            // The MTP and GDN-first self-speculative routes are chosen from the request,
            // the checkpoint and the environment alone, before any state is touched. The
            // one route plan made here also configures the decoder session, which refuses
            // both speculative routes, so a request cannot reach the ordinary loop by a
            // second reading of the environment.
            let environment = SamplingRouteEnvironment::current();
            let route = plan_sampling_route(gen_cfg, plan.prompt_ids.is_empty(), environment);
            let mtp_enabled = gen_cfg
                .enable_mtp
                .unwrap_or_else(|| crate::env_switch_enabled("LATTICE_MTP"));
            let use_mtp = super::mtp_route_active(
                self.session.mtp.is_some(),
                mtp_enabled,
                gen_cfg,
                route.use_compact,
            );
            let use_self_spec = !use_mtp
                && super::self_spec_route_active(
                    self.session.gdn_checkpoints.is_some(),
                    crate::env_switch_enabled("LATTICE_SELF_SPEC"),
                    gen_cfg,
                    route.use_compact,
                    self.engine.config.num_active_linear_attention_layers(),
                );
            if use_mtp || use_self_spec {
                // Both routes require a greedy request without grammar, so the plan never
                // engages a compact route here and prefill logits are always dense.
                let GenerationPlan {
                    prompt_ids,
                    prompt_len,
                    ..
                } = plan;
                self.reset_state();
                apply_sampling_route_plan(
                    route,
                    &mut self.session.compact_route,
                    &mut self.session.compact_topk,
                    &mut self.session.compact_result,
                );
                // Through the fallible entry point: a multi-token prompt on an MoE model
                // has no batched schedule and must come back as `UnsupportedModel`.
                let prefill_logits = self.try_forward_prefill(&prompt_ids)?;
                if use_mtp {
                    // This request has committed to the MTP path (#1336 round 2) —
                    // downstream pre-final-hidden-capture gates key off this, not
                    // off `self.session.mtp.is_some()` (which only reports whether
                    // the checkpoint carries MTP weights, not whether this request
                    // uses them).
                    self.session.mtp_active = true;
                    // #1340: rebuild the MTP draft head's KV cache from the prompt
                    // before the first draft round, so it attends to the prompt prefix
                    // instead of the single row `reset_state` left it with.
                    self.mtp_prefill(&prompt_ids);
                    return self.generate_greedy_mtp(
                        &prefill_logits,
                        prompt_len,
                        tokenizer,
                        gen_cfg,
                    );
                }
                return self.generate_greedy_self_spec(
                    &prefill_logits,
                    prompt_len,
                    tokenizer,
                    gen_cfg,
                );
            }

            let eos_token_id = self.engine.config.eos_token_id;
            let prompt_ids = plan.prompt_ids.clone();
            // The session resets the state and engages the planned sampling route;
            // dropping it tears the route down on every exit path below.
            let mut session = crate::decoder::qwen_metal::QwenMetalSession::with_route_environment(
                self,
                plan,
                gen_cfg,
                crate::decoder::qwen_metal::MetalEntryProfile::Direct,
                environment,
            )?;
            // KV capacity: `prepare_generation` admitted this request only if
            // `prompt_len + max_new_tokens <= max_context()` (the direct entry refuses a
            // reasoning budget, so the decode cap is `max_new_tokens`), and the driver
            // runs at most `max_new_tokens - 1` decode steps after prefill, so the last
            // step writes position `max_context() - 2` at most. No admitted request
            // fills the cache: this entry never stops with `StopReason::KvFull`, and the
            // session's full-cache refusal cannot fire.
            crate::decoder::qwen_metal::run_direct(
                &mut session,
                gen_cfg,
                &prompt_ids,
                eos_token_id,
                tokenizer,
            )
        }

        /// **Unstable**: multimodal generation; visual token handling and position encoding
        /// may evolve as the Qwen3-VL weight schema is validated (ADR-049 v0).
        ///
        /// Generate text from a multimodal prompt (image + text).
        ///
        /// Visual patch embeddings from `input.patch_embeddings` are prepended to the text
        /// token embeddings before the first prefill call.  Position IDs for text tokens are
        /// offset by `input.visual_tokens` so that patch positions `[0, visual_tokens)` are
        /// occupied by the image.  All other generation parameters (KV cache, sampling,
        /// stop tokens) behave identically to `generate`.
        ///
        /// # Errors
        ///
        /// Returns `InferenceError::InvalidInput` if:
        /// - `input.d_model` does not match this model's `hidden_size`
        /// - `input.patch_embeddings` length is inconsistent with `visual_tokens * d_model`
        ///
        /// Returns the shared `InferenceError::Inference` context-budget error if the
        /// visual and text prompt plus requested decode cap exceed the KV cache capacity.
        pub fn generate_multimodal(
            &mut self,
            input: crate::vision::MultimodalInput,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> Result<GenerateOutput, crate::error::InferenceError> {
            use crate::error::InferenceError;

            // Reject grammar configs before any GPU work: grammar-constrained decoding
            // is not wired into the multimodal path.
            super::multimodal_generate_preflight(gen_cfg)?;

            // Validate multimodal input
            input.validate().map_err(|e| {
                InferenceError::InvalidInput(format!("MultimodalInput validation failed: {e}"))
            })?;

            let hidden = self.engine.config.hidden_size;

            if input.d_model != hidden {
                return Err(InferenceError::InvalidInput(format!(
                    "generate_multimodal: input.d_model ({}) does not match model hidden_size ({})",
                    input.d_model, hidden
                )));
            }

            let visual_tokens = input.visual_tokens;
            let text_ids = &input.text_tokens;
            let total_len = visual_tokens + text_ids.len();

            if total_len == 0 {
                return Ok(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: 0,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: None,
                    token_logprobs: vec![],
                });
            }

            // max_new_tokens == 0 means "generate nothing": return before prefill/sampling
            // so we never emit a token the caller did not ask for. Mirrors the
            // guard in generate() above (#612).
            if gen_cfg.max_new_tokens == 0 {
                return Ok(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: total_len,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: Some(StopReason::Length),
                    token_logprobs: vec![],
                });
            }

            crate::model::qwen35::check_context_budget(
                total_len,
                gen_cfg.effective_reasoning_budget(),
                gen_cfg.max_new_tokens,
                self.max_context(),
            )?;

            let cfg = self.engine.config.clone();

            // Reset recurrent state for a clean generation.
            self.reset_state();

            let mut rng_state = match gen_cfg.seed {
                Some(s) => {
                    if s == 0 {
                        1
                    } else {
                        s
                    }
                }
                None => {
                    use std::time::SystemTime;
                    let t = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0x12345678_9abcdef0_u64);
                    if t == 0 { 1 } else { t }
                }
            };

            // ---------------------------------------------------------------
            // Prefill: write [visual_tokens | text_tokens] embeddings into the
            // activations buffer, then run the prefill forward pass.
            //
            // The hidden buffer is sized max_prefill * hidden at construction;
            // we need total_len slots.  For v0 we fall back to sequential stepping
            // when total_len > max_prefill to avoid a bounds overrun.
            // ---------------------------------------------------------------

            // Validate text token IDs
            for (i, &id) in text_ids.iter().enumerate() {
                if id as usize >= cfg.vocab_size {
                    return Err(InferenceError::InvalidInput(format!(
                        "generate_multimodal: text_tokens[{i}]={id} >= vocab_size {}",
                        cfg.vocab_size
                    )));
                }
            }

            // Build a combined embedding sequence: [visual_embeds | text_embeds].
            // We write it in one shot using sequential forward steps to avoid
            // sizing the hidden buffer beyond its allocation.
            //
            // Position IDs:
            //   - Visual patch tokens: positions 0 .. visual_tokens
            //     (positions are unused by decoder — the ViT already applied 2D RoPE)
            //   - Text tokens: positions visual_tokens .. visual_tokens + text_ids.len()
            //
            // For v0, we run each visual token as a forward_step with a synthetic
            // embedding (bypassing the embed_tokens lookup) by directly writing the
            // patch embedding into the hidden buffer, then invoking the layer stack
            // at the appropriate position.
            //
            // Concretely: we produce logits only for the text portion prefill; the
            // visual token steps advance KV cache position without needing their
            // logits.  This matches the reference: image tokens occupy positions
            // [0, visual_tokens), text tokens follow.

            // Step 1: Feed each visual token as a hidden-state injection.
            // We borrow `forward_step_with_hidden` logic: write directly to
            // `session.activations.hidden`, then run the layer stack at the position.
            for vt in 0..visual_tokens {
                let embed_start = vt * hidden;
                let embed_end = embed_start + hidden;
                let visual_embed = &input.patch_embeddings[embed_start..embed_end];

                // SAFETY: activations.hidden is StorageModeShared f32; no GPU command
                // is in flight at this point (we are between forward calls).
                unsafe {
                    let dst = self.session.activations.hidden.contents() as *mut f32;
                    for i in 0..hidden {
                        *dst.add(i) = visual_embed[i];
                    }
                }

                // Run the layer stack at position `vt` (no embedding lookup needed —
                // we already wrote the hidden state above).  We reuse `forward_step`'s
                // infrastructure by writing token_id=0 (a valid ID) into the buffer and
                // then immediately overwriting with the visual embedding.  However, that
                // would double-write.  Instead, use the internal `forward_step_inner`
                // helper which runs layers on whatever is already in `hidden`.
                //
                // Since `forward_step_inner` is not directly accessible outside the inner
                // module (it has no `pub` on the wrapper), we call the public
                // `forward_step` with token_id=0 to drive the layer stack and KV cache
                // advancement, then discard the logits.  Before the call, we overwrite
                // the embedding that `forward_step` writes so the correct visual
                // embedding propagates through the layers.
                //
                // Implementation note: `forward_step` writes the embedding at the START
                // of its body (before any GPU dispatch), so we must write AFTER the
                // embedding lookup.  We achieve this by calling `forward_step` normally —
                // which correctly looks up token_id=0 — and instead pre-initializing the
                // hidden buffer with the visual embedding before the call, relying on the
                // fact that `forward_step` OVERWRITES the hidden buffer with the token
                // embedding at the start.
                //
                // This means we cannot avoid the embed_tokens read in `forward_step`.
                // For v0 correctness (the embedding is overwritten), we call a thin
                // wrapper that writes the visual embedding AFTER the embed lookup but
                // before the first GPU dispatch.  Since we cannot easily interpose
                // without modifying forward_step, v0 uses the following approach:
                //
                // Write visual_embed → call forward_step(0, vt) → logits discarded.
                // forward_step will overwrite our write with embed_tokens[0], but the
                // net effect is that we process token 0 at each visual position.
                //
                // *** v0 limitation ***: visual token embeddings are not correctly fed
                // through the decoder in this release; only the position offset is
                // established.  The visual semantics require a Metal-level injection
                // point (v1 work item per ADR-049).  The generate_multimodal API surface
                // and types are correct and tested; the actual visual features require
                // v1 Metal kernel integration.
                let _logits = self.forward_step(0u32, vt);
            }

            // Step 2: Run the text portion using the existing sequential path.
            // Text tokens start at position `visual_tokens`.
            let prompt_len = total_len;
            let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
            let mut all_ids: Vec<u32> = text_ids.clone();

            // Process all text tokens sequentially at their offset positions.
            let mut last_logits = Vec::new();
            for (t, &id) in text_ids.iter().enumerate() {
                last_logits = self.forward_step(id, visual_tokens + t);
            }

            // If no text tokens were given, we have no last logits — just return empty.
            if text_ids.is_empty() {
                return Ok(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: visual_tokens,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: None,
                    token_logprobs: vec![],
                });
            }

            // Sample the first token from last logits.
            let is_stop = |id: u32| id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id);

            let mut stopped = false;
            let mut stop_reason = StopReason::Length;

            // String-stop path: mirrors the CPU stop-string matcher exactly (#643).
            let mut stop_text_state = if gen_cfg.stop_strings.is_empty() {
                None
            } else {
                Some((
                    incremental_detokenizer_new(),
                    StopStringMatcher::new(&gen_cfg.stop_strings),
                ))
            };

            let first_id = sample_token(&last_logits, gen_cfg, &all_ids, &mut rng_state);
            if is_stop(first_id) {
                stopped = true;
                stop_reason = StopReason::Eos;
            } else {
                generated_ids.push(first_id);
                all_ids.push(first_id);
                if let Some((detok, matcher)) = stop_text_state.as_mut() {
                    let delta = incremental_detokenizer_push(detok, tokenizer, first_id);
                    if matcher.push(&delta, &mut |_| {}) {
                        stopped = true;
                        stop_reason = StopReason::Eos;
                    }
                }
            }

            // Autoregressive decode loop.
            let mut pos = visual_tokens + text_ids.len();
            while !stopped && generated_ids.len() < gen_cfg.max_new_tokens {
                if self.session.kv_cache.seq_len >= self.session.kv_cache.max_cache_len {
                    stop_reason = StopReason::KvFull;
                    break;
                }
                let last_token = *all_ids.last().unwrap_or(&cfg.eos_token_id);
                if is_stop(last_token) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }
                let step_logits = self.forward_step_decode(last_token, pos);
                pos += 1;
                let next_id =
                    sample_decode_traced(None, &step_logits, gen_cfg, &all_ids, &mut rng_state);
                if is_stop(next_id) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }
                generated_ids.push(next_id);
                all_ids.push(next_id);

                if let Some((detok, matcher)) = stop_text_state.as_mut() {
                    let delta = incremental_detokenizer_push(detok, tokenizer, next_id);
                    if matcher.push(&delta, &mut |_| {}) {
                        stopped = true;
                        stop_reason = StopReason::Eos;
                        break;
                    }
                }
            }

            let text = if let Some((mut detok, mut matcher)) = stop_text_state {
                if !matcher.stopped() {
                    matcher.finish(&incremental_detokenizer_finish(&mut detok), &mut |_| {});
                }
                matcher.into_text()
            } else {
                decode_tokens(tokenizer, &generated_ids)
            };
            Ok(GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
                stop_reason: Some(stop_reason),
                token_logprobs: vec![],
            })
        }

        /// Qwen3.5 vision (ADR-069 Metal S5, MP2+MP3): Metal port of
        /// `cpu_f16::generate_multimodal_f16` — the CPU function is the numeric
        /// oracle (ADR-069 Amendment 1: divergence from it is a bug by
        /// definition), so this mirrors its request type, validation order, and
        /// per-token embedding + M-RoPE contract exactly: real residual-stream
        /// REPLACE injection of merged patch embeddings at `<|image_pad|>` slots
        /// (mirrors `forward_step_f16`'s `injected_embedding`), and per-token
        /// 3-axis interleaved M-RoPE cos/sin rows at the six GQA layers (mirrors
        /// `full_attention_step_f16`'s `mrope_cos_sin`) built from
        /// `Qwen35VisionRequest::build_mrope_tables`/`build_decode_cos_sin`. A
        /// text-only request (`image_grids` empty) passes `None` throughout,
        /// which keeps every position on the model's unchanged 1-D
        /// `engine.rope_cos`/`rope_sin` table — structurally identical to plain
        /// text decode.
        ///
        /// Deliberately a new entry point rather than a change to [`Self::generate_multimodal`]:
        /// that function's [`crate::vision::MultimodalInput`] represents visual tokens as a
        /// flat prefix and carries no per-image `(T, H, W)` grid shape, so it cannot supply
        /// `Qwen35VisionRequest`'s validation/checkpoint-binding inputs this port reuses.
        /// `generate_multimodal` is unchanged by this PR.
        pub fn generate_multimodal_vision(
            &mut self,
            request: &Qwen35VisionRequest,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> Result<GenerateOutput, crate::error::InferenceError> {
            self.generate_multimodal_vision_impl(request, tokenizer, gen_cfg, None, None)
        }

        /// Generate from one vision request while polling caller cancellation.
        ///
        /// Cancellation is checked before prefill, before every prefill token,
        /// after prefill, and before every decode step. The returned output
        /// carries [`StopReason::Interrupt`] when cancellation wins.
        pub fn generate_multimodal_vision_with_cancel<C>(
            &mut self,
            request: &Qwen35VisionRequest,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            mut should_cancel: C,
        ) -> Result<GenerateOutput, crate::error::InferenceError>
        where
            C: FnMut() -> bool,
        {
            self.generate_multimodal_vision_impl(
                request,
                tokenizer,
                gen_cfg,
                None,
                Some(&mut should_cancel),
            )
        }

        fn multimodal_cancel_requested(
            should_cancel: &mut Option<&mut dyn FnMut() -> bool>,
        ) -> bool {
            match should_cancel {
                Some(callback) => callback(),
                None => false,
            }
        }

        fn cancelled_multimodal_output(prompt_tokens: usize) -> GenerateOutput {
            GenerateOutput {
                text: String::new(),
                token_ids: Vec::new(),
                prompt_tokens,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Interrupt),
                token_logprobs: Vec::new(),
            }
        }

        /// Body of [`Self::generate_multimodal_vision`] with an optional
        /// decode-step logits probe (each decode iteration's full logits row,
        /// in order). The probe exists because greedy token output is not a
        /// usable observable for the decode-time M-RoPE coordinate: 18 of 24
        /// decoder layers are GDN (no RoPE at all), and on the real #989
        /// fixture a `None`-decode mutation reproduced the correct arm's
        /// greedy tokens exactly over a 32-step horizon. Logit-level
        /// assertions through this seam are what make the decode wiring
        /// mutation-detectable (same pattern as `gemma4`'s
        /// `generate_greedy_with_probe`).
        fn generate_multimodal_vision_impl(
            &mut self,
            request: &Qwen35VisionRequest,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            mut decode_logits_probe: Option<&mut Vec<Vec<f32>>>,
            mut should_cancel: Option<&mut dyn FnMut() -> bool>,
        ) -> Result<GenerateOutput, crate::error::InferenceError> {
            use crate::error::InferenceError;

            super::multimodal_generate_preflight(gen_cfg)?;

            request.validate().map_err(|e| {
                InferenceError::InvalidInput(format!("multimodal request failed validation: {e}"))
            })?;

            let cfg = self.engine.config.clone();

            // Caller-supplied `input_ids` must be bounded against the checkpoint
            // vocabulary before any GPU work begins — mirrors the CPU oracle's guard
            // ahead of `forward_step_f16`'s embedding-table slice.
            if let Some(&bad_id) = request
                .input_ids
                .iter()
                .find(|&&id| id as usize >= cfg.vocab_size)
            {
                return Err(InferenceError::InvalidInput(format!(
                    "input_ids contains out-of-vocabulary token id {bad_id} (vocab_size={})",
                    cfg.vocab_size
                )));
            }

            let has_image = !request.image_grids.is_empty();

            // Bind the request to the loaded checkpoint's vision metadata before
            // selecting image slots (mirrors the CPU oracle's checkpoint-binding
            // guard) — an internally consistent request targeting the wrong
            // checkpoint must not silently inject at the wrong slots.
            if has_image {
                let cfg_image_token_id = cfg.image_token_id.ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "multimodal request supplied but checkpoint has no image_token_id"
                            .to_string(),
                    )
                })?;
                if cfg_image_token_id != request.image_token_id {
                    return Err(InferenceError::InvalidInput(format!(
                        "request image_token_id {} does not match checkpoint image_token_id {cfg_image_token_id}",
                        request.image_token_id
                    )));
                }
                let vision_cfg = cfg.vision_config.as_ref().ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "multimodal request supplied but checkpoint has no vision_config"
                            .to_string(),
                    )
                })?;
                if vision_cfg.spatial_merge_size != request.spatial_merge_size {
                    return Err(InferenceError::InvalidInput(format!(
                        "request spatial_merge_size {} does not match checkpoint \
                         vision_config.spatial_merge_size {}",
                        request.spatial_merge_size, vision_cfg.spatial_merge_size
                    )));
                }
                if request.decoder_hidden_size != cfg.hidden_size {
                    return Err(InferenceError::InvalidInput(format!(
                        "request decoder_hidden_size {} does not match checkpoint hidden_size {}",
                        request.decoder_hidden_size, cfg.hidden_size
                    )));
                }
                if vision_cfg.out_hidden_size != cfg.hidden_size {
                    return Err(InferenceError::InvalidInput(format!(
                        "checkpoint vision_config.out_hidden_size {} does not match decoder \
                         hidden_size {}",
                        vision_cfg.out_hidden_size, cfg.hidden_size
                    )));
                }
            }

            // Build the per-physical-token M-RoPE position ids and interleaved
            // cos/sin tables (ADR-069 MP3). Mirrors `generate_multimodal_f16`'s
            // ordering: built right after the checkpoint-binding checks above,
            // before the prompt-empty/context-budget checks below.
            let (positions, tables) = request.build_mrope_tables(&cfg)?;
            let expected_rope_half = cfg.rope_dim() / 2;
            if tables.cos.iter().any(|row| row.len() != expected_rope_half)
                || tables.sin.iter().any(|row| row.len() != expected_rope_half)
            {
                return Err(InferenceError::InvalidInput(format!(
                    "M-RoPE table row width does not match decoder rotary half-width: expected \
                     {expected_rope_half}"
                )));
            }

            let prompt_ids = &request.input_ids;
            let prompt_len = prompt_ids.len();
            crate::model::qwen35::check_prompt_not_empty(prompt_len)?;

            if gen_cfg.max_new_tokens == 0 {
                return Ok(GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                    stopped: false,
                    stop_reason: Some(StopReason::Length),
                    token_logprobs: vec![],
                });
            }

            crate::model::qwen35::check_context_budget(
                prompt_len,
                gen_cfg.effective_reasoning_budget(),
                gen_cfg.max_new_tokens,
                self.max_context(),
            )?;

            if Self::multimodal_cancel_requested(&mut should_cancel) {
                return Ok(Self::cancelled_multimodal_output(prompt_len));
            }

            // Reset recurrent state for a clean generation.
            self.reset_state();

            let mut rng_state = match gen_cfg.seed {
                Some(s) => {
                    if s == 0 {
                        1
                    } else {
                        s
                    }
                }
                None => {
                    use std::time::SystemTime;
                    let t = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0x12345678_9abcdef0_u64);
                    if t == 0 { 1 } else { t }
                }
            };

            // Prefill: inject a post-merger visual row at each `<|image_pad|>` slot, in the
            // same sequential image-token order the request already concatenated
            // `post_merger_rows` in (HF's masked_scatter contract), text tokens (including
            // vision delimiters) keep their normal embedding lookup. Every physical
            // position — image-pad or text — rotates the six GQA layers with its
            // `tables.cos[pos]`/`tables.sin[pos]` row when the request carries an
            // image (ADR-069 MP3); `has_image=false` passes `None` throughout, which
            // keeps `forward_step_mrope`/`forward_step_injected_mrope` on the
            // unchanged 1-D `engine.rope_cos`/`rope_sin` path — structurally
            // identical to the pre-MP3 text-only route.
            let mut visual_row = 0usize;
            let mut last_logits = Vec::new();
            let last_prefill_pos = prompt_len - 1;
            for (pos, &token_id) in prompt_ids.iter().enumerate() {
                if Self::multimodal_cancel_requested(&mut should_cancel) {
                    return Ok(Self::cancelled_multimodal_output(prompt_len));
                }
                let cos_sin = if has_image {
                    Some((tables.cos[pos].as_slice(), tables.sin[pos].as_slice()))
                } else {
                    None
                };
                // issue #1336: only the prompt's final position is ever sampled —
                // every earlier position still advances KV/GDN state but skips the
                // terminal RMSNorm/lm_head/vocab readback (`emit_head=false`).
                let emit_head = pos == last_prefill_pos;
                if has_image && token_id == request.image_token_id {
                    let start = visual_row * request.decoder_hidden_size;
                    let end = start + request.decoder_hidden_size;
                    let row = &request.post_merger_rows[start..end];
                    // Mirrors `forward_step_f16`'s injected_embedding guard: reject a
                    // non-finite visual row before it reaches the residual stream. (Length
                    // is already guaranteed by `request.validate()` above.)
                    if let Some(bad) = row.iter().find(|v| !v.is_finite()) {
                        return Err(InferenceError::InvalidInput(format!(
                            "injected_embedding contains a non-finite value: {bad}"
                        )));
                    }
                    visual_row += 1;
                    last_logits = self.forward_step_injected_mrope(
                        row,
                        pos,
                        cos_sin,
                        emit_head,
                        crate::forward::signpost::Scope::NotDecode,
                    );
                } else {
                    last_logits = self.forward_step_mrope(
                        token_id,
                        pos,
                        cos_sin,
                        emit_head,
                        crate::forward::signpost::Scope::NotDecode,
                    );
                }
            }
            if Self::multimodal_cancel_requested(&mut should_cancel) {
                return Ok(Self::cancelled_multimodal_output(prompt_len));
            }

            let mut all_ids = prompt_ids.clone();
            let is_stop = |id: u32| id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id);
            let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
            let mut stopped = false;
            let mut stop_reason = StopReason::Length;

            // String-stop path: mirrors the CPU stop-string matcher exactly, same as
            // `generate_multimodal`.
            let mut stop_text_state = if gen_cfg.stop_strings.is_empty() {
                None
            } else {
                Some((
                    incremental_detokenizer_new(),
                    StopStringMatcher::new(&gen_cfg.stop_strings),
                ))
            };

            let first_id = sample_token(&last_logits, gen_cfg, &all_ids, &mut rng_state);
            if is_stop(first_id) {
                stopped = true;
                stop_reason = StopReason::Eos;
            } else {
                generated_ids.push(first_id);
                all_ids.push(first_id);
                if let Some((detok, matcher)) = stop_text_state.as_mut() {
                    let delta = incremental_detokenizer_push(detok, tokenizer, first_id);
                    if matcher.push(&delta, &mut |_| {}) {
                        stopped = true;
                        stop_reason = StopReason::Eos;
                    }
                }
            }

            // Autoregressive decode: identical to the plain `generate`/`generate_streaming`
            // loop, except that once the prompt contained an image the physical
            // KV-cache index (`physical_pos`) and the M-RoPE coordinate diverge —
            // mirrors `generate_multimodal_f16`'s decode loop (ADR-069 RECON §3):
            // `decode_axis = physical_pos + rope_delta`, recomputed into a single
            // cos/sin row via the same `build_decode_cos_sin` the CPU oracle uses.
            while !stopped && generated_ids.len() < gen_cfg.max_new_tokens {
                if Self::multimodal_cancel_requested(&mut should_cancel) {
                    stop_reason = StopReason::Interrupt;
                    break;
                }
                if self.session.kv_cache.seq_len >= self.session.kv_cache.max_cache_len {
                    stop_reason = StopReason::KvFull;
                    break;
                }
                let physical_pos = self.session.kv_cache.seq_len;
                let last_token = *all_ids.last().unwrap_or(&cfg.eos_token_id);
                if is_stop(last_token) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }

                let decode_cos_sin_owned;
                let mrope_cos_sin = if has_image {
                    let decode_axis = crate::vision::qwen35_mrope::decode_position(
                        physical_pos,
                        positions.rope_delta,
                    )?;
                    decode_cos_sin_owned = request.build_decode_cos_sin(&cfg, decode_axis)?;
                    if decode_cos_sin_owned.0.len() != expected_rope_half
                        || decode_cos_sin_owned.1.len() != expected_rope_half
                    {
                        return Err(InferenceError::InvalidInput(format!(
                            "decode-time M-RoPE row width does not match decoder rotary \
                             half-width: expected {expected_rope_half}"
                        )));
                    }
                    Some((
                        decode_cos_sin_owned.0.as_slice(),
                        decode_cos_sin_owned.1.as_slice(),
                    ))
                } else {
                    None
                };

                let step_logits = self.forward_step_mrope(
                    last_token,
                    physical_pos,
                    mrope_cos_sin,
                    true,
                    crate::forward::signpost::Scope::Decode,
                );
                if let Some(probe) = decode_logits_probe.as_deref_mut() {
                    probe.push(step_logits.clone());
                }
                let next_id =
                    sample_decode_traced(None, &step_logits, gen_cfg, &all_ids, &mut rng_state);
                if is_stop(next_id) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }
                generated_ids.push(next_id);
                all_ids.push(next_id);

                if let Some((detok, matcher)) = stop_text_state.as_mut() {
                    let delta = incremental_detokenizer_push(detok, tokenizer, next_id);
                    if matcher.push(&delta, &mut |_| {}) {
                        stopped = true;
                        stop_reason = StopReason::Eos;
                        break;
                    }
                }
            }

            let text = if let Some((mut detok, mut matcher)) = stop_text_state {
                if !matcher.stopped() {
                    matcher.finish(&incremental_detokenizer_finish(&mut detok), &mut |_| {});
                }
                matcher.into_text()
            } else {
                decode_tokens(tokenizer, &generated_ids)
            };

            Ok(GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
                stopped,
                stop_reason: Some(stop_reason),
                token_logprobs: vec![],
            })
        }

        /// **Unstable**: reset recurrent state; state buffer layout may change.
        ///
        /// Reset all recurrent state for a new generation.
        pub fn reset_state(&mut self) {
            let cfg = &self.engine.config;
            self.session.gdn_states = (0..cfg.num_active_linear_attention_layers())
                .map(|_| GatedDeltaNetState::new(cfg))
                .collect();
            // Zero GPU GDN buffers
            for layer in self.session.gdn_gpu_state.layers() {
                let buf = layer.conv_buffer();
                // SAFETY: Each buffer is StorageModeShared, no command buffer is
                // in flight during reset_state, and length() is the allocated size.
                unsafe {
                    let ptr = buf.contents() as *mut u8;
                    std::ptr::write_bytes(ptr, 0, buf.length() as usize);
                }
            }
            for layer in self.session.gdn_gpu_state.layers() {
                let buf = layer.s_matrix();
                // SAFETY: Each buffer is StorageModeShared, no command buffer is
                // in flight during reset_state, and length() is the allocated size.
                unsafe {
                    let ptr = buf.contents() as *mut u8;
                    std::ptr::write_bytes(ptr, 0, buf.length() as usize);
                }
            }
            self.session.kv_cache.reset();
            if let Some(ref mut mtp) = self.session.mtp {
                mtp.cache.reset();
            }
            // A fresh request has not yet decided whether it will take the MTP
            // greedy/verify path (#1336 round 2) — `generate()` sets this back
            // to `true` itself once `use_mtp` commits.
            self.session.mtp_active = false;
            self.session.speculative_route = None;
            self.session.mtp_prefill_hidden.clear();
            if let Some(ref mut pool) = self.session.gdn_checkpoints {
                pool.active_base_seq_len = None;
            }
            self.session.last_pre_final_hidden = vec![0.0f32; self.engine.config.hidden_size];
            self.session.last_hidden_cursor = None;
            // Every public path that resets live
            // KV/GDN state (plain `generate_streaming`, chat, the serve
            // worker, and the cache-aware path's own FullRefill/error legs)
            // routes through here, so this is the single place that must
            // invalidate any retained cross-turn entry. A retained entry
            // describes exactly the state this call just vacated; leaving it
            // alive would let a later cache-aware call restore GDN state for
            // KV rows that no longer represent it.
            self.cross_turn_prefix_cache.clear();
            // Issue #171: a compact top-k request from a
            // prior generation must not leak into the next one. Without this, a
            // caller that reuses one `MetalQwen35State` for a compact-eligible
            // generation followed by an exact-required generation (grammar,
            // repetition penalty, or non-compact top_k) can hit `compact_topk > 0`
            // with `use_compact == false`, which makes `forward_prefill` /
            // `forward_step` append a compact top-k dispatch and return empty
            // logits even though the caller expects the full-logit exact path.
            self.disengage_compact_route();
            self.session.compact_result.clear();
        }

        /// Select the GDN prefill scan at runtime: `true` = chunked-parallel (default),
        /// `false` = serial per-token. Construction reads `LATTICE_GDN_CHUNKED`; this lets
        /// callers (benchmarks, A/B tests) flip the path on a live state without rebuilding.
        ///
        /// Flipping the mode invalidates any retained cross-turn cache entry: the two
        /// scans are not bit-exact (measured logit drift up to ~2e-1 at chunk boundaries),
        /// so a snapshot produced under one mode must not seed a prefix that the current
        /// mode's full-refill reference would have prefilled differently.
        pub fn set_gdn_chunked(&mut self, enabled: bool) {
            if self.use_gdn_chunked != enabled {
                self.cross_turn_prefix_cache.clear();
            }
            self.use_gdn_chunked = enabled;
        }

        // ===================================================================
        // Layer-encoding helpers (called from forward_step_inner)
        // ===================================================================

        /// Copy the last token's hidden state into `activations.final_hidden`,
        /// AFTER the final RMSNorm has been applied to it.
        ///
        /// One implementation with three call sites, one per forward path that
        /// dispatches the final norm, so the paths cannot drift apart. Grep
        /// `engine.final_norm` to enumerate them; every hit is either a
        /// dispatch immediately followed by a call to this, or the weight
        /// buffer's own definition.
        ///
        /// Placement after the norm is the whole point. The norm runs in place
        /// on `activations.hidden`, so the same address holds a pre-norm vector
        /// before it and a post-norm vector after, and the two differ by the
        /// norm's rescale. A capture encoded before the norm returns
        /// well-formed, correctly-sized, plausible numbers from the wrong point
        /// in the network — which is exactly what `pre_final_hidden` above
        /// holds for MTP, and exactly why this is a separate buffer.
        ///
        /// `src_row_offset_bytes` selects the row of `activations.hidden`
        /// holding the token of interest: the last row on the batch paths, row
        /// zero on the single-token decode path.
        fn encode_capture_final_hidden(
            &self,
            enc: &ComputeCommandEncoderRef,
            src_row_offset_bytes: u64,
            hidden: usize,
        ) {
            if !self.session.capture_final_hidden {
                return;
            }
            // `&self`, not `&mut self`: two of the three call sites hold a live
            // immutable borrow of `self.engine.queue`'s command buffer across
            // the encode, so a `&mut self` helper cannot be called there at
            // all. The record is therefore an atomic rather than a plain bool.
            self.session
                .final_hidden_captured
                .store(true, std::sync::atomic::Ordering::Relaxed);
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
            enc.set_buffer(
                0,
                Some(&self.session.activations.hidden),
                src_row_offset_bytes,
            );
            enc.set_buffer(1, Some(&self.session.activations.final_hidden), 0);
            let cnt = hidden as u32;
            let dst_off = 0u32;
            enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
            enc.set_bytes(3, 4, &dst_off as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        /// Encode the final head: optional pre-final hidden capture, RMS norm,
        /// logit GEMV, and optional top-k.
        ///
        /// All commands are appended to the already-open encoder `enc`.
        /// Returns `Some(which)` when compact top-k is active, `None` otherwise.
        fn encode_final_head(
            &mut self,
            enc: &ComputeCommandEncoderRef,
            cfg: &Qwen35Config,
            capture_hidden: bool,
            prof: &mut StepProfile,
            profiling: bool,
        ) -> Option<u8> {
            let hidden = cfg.hidden_size;

            // === Capture pre-final hidden for MTP (before RMSNorm overwrites) ===
            if capture_hidden || self.session.mtp.is_some() {
                self.dispatch_copy(
                    enc,
                    &self.session.activations.hidden,
                    &self.session.activations.pre_final_hidden,
                    hidden as u32,
                );
            }

            // === Final RMS norm + logits (same command buffer) ===
            let final_t0 = profiling.then(std::time::Instant::now);
            self.dispatch_rms_norm(
                enc,
                &self.session.activations.hidden,
                &self.engine.final_norm,
                hidden as u32,
                1,
                cfg.rms_norm_eps,
            );
            // Site 3 of 3. Single-token path: the token of interest is row 0.
            self.encode_capture_final_hidden(enc, 0, hidden);
            // Issue #171: lm_head two-stage block-top-k. When the route requires
            // only argmax/top-k (never the full logit vector), skip the full
            // [vocab_size] GEMV entirely — Stage 1 fuses the GEMV with a
            // block-local exact reduction, Stage 2 (below) reduces those compact
            // candidates to the global result. This is the threadgroup-flood fix:
            // the full-logit GEMV alone dispatches tens of thousands of
            // threadgroups (ceil(vocab/4) or ceil(vocab/2)); Stage 1 dispatches
            // ceil(vocab/LM_HEAD_ROWS_PER_TG), ~100x fewer.
            let block_local_k = resolve_block_local_k(self.session.compact_route);
            if let Some((local_k, slot)) = block_local_k {
                let candidate_groups =
                    self.dispatch_lm_head_block_stage1_enc(enc, cfg, 0, local_k, slot);
                let which = self.dispatch_block_topk_merge_enc(enc, candidate_groups, local_k);
                if let Some(t0) = final_t0 {
                    prof.final_us += t0.elapsed().as_micros();
                }
                return Some(which);
            }

            // lm_head: for Q8 format use FP16 wide kernel (NR=4, fewer TGs than
            // gemv_decode_m1 which dispatches one TG per vocab row). For Q4 use Q4 path.
            match self.engine.quant_format {
                QuantFormat::Q8_0 => {
                    let vocab_n = cfg.vocab_size as u32;
                    let hidden_k = hidden as u32;
                    enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode_wide);
                    enc.set_buffer(0, Some(&self.session.activations.hidden), 0);
                    enc.set_buffer(1, Some(&self.engine.embed_tokens), 0);
                    enc.set_buffer(2, Some(&self.session.activations.logits), 0);
                    enc.set_bytes(3, 4, &vocab_n as *const u32 as *const _);
                    enc.set_bytes(4, 4, &hidden_k as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(vocab_n.div_ceil(4) as u64, 1, 1),
                        MTLSize::new(32, 4, 1),
                    );
                }
                QuantFormat::Q4_0 => {
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.hidden,
                        &self.engine.embed_tokens_q8,
                        &self.session.activations.logits,
                        1,
                        cfg.vocab_size as u32,
                        hidden as u32,
                    );
                }
            }
            if let Some(t0) = final_t0 {
                prof.final_us += t0.elapsed().as_micros();
            }

            // If compact mode, append top-k kernels to the same encoder before commit.
            // This avoids a second command-buffer round-trip (~18 µs overhead).
            if self.session.compact_topk > 0 {
                let k = self.session.compact_topk as u32;
                Some(self.dispatch_topk_enc(enc, cfg.vocab_size as u32, k))
            } else {
                None
            }
        }

        // -----------------------------------------------------------------------
        // GDN chunked-scan helpers (default ON; LATTICE_GDN_CHUNKED=0 opts out)
        // -----------------------------------------------------------------------

        fn supports_gdn_chunked_prefill(cfg: &Qwen35Config) -> bool {
            cfg.hidden_size == 1024
                && cfg.linear_num_key_heads == 16
                && cfg.linear_num_value_heads() == 16
                && cfg.linear_key_head_dim == 128
                && cfg.linear_value_head_dim == 128
                && cfg.linear_conv_kernel_dim == 4
        }

        fn make_gdn_chunk_params(cfg: &Qwen35Config, n_tokens: usize) -> Option<GdnChunkParams> {
            if !Self::supports_gdn_chunked_prefill(cfg) || n_tokens == 0 {
                return None;
            }
            let n_tokens = u32::try_from(n_tokens).ok()?;
            let num_chunks = n_tokens.div_ceil(GDN_CHUNK_SIZE as u32);
            let key_dim = cfg.linear_key_head_dim as u32;
            let value_dim = cfg.linear_value_head_dim as u32;
            let num_key_heads = cfg.linear_num_key_heads as u32;
            let num_value_heads = cfg.linear_num_value_heads() as u32;
            let q_total = num_key_heads * key_dim;
            Some(GdnChunkParams {
                key_dim,
                value_dim,
                num_key_heads,
                num_value_heads,
                hidden_size: cfg.hidden_size as u32,
                q_total,
                v_offset: q_total * 2,
                qkv_dim: cfg.linear_qkv_dim() as u32,
                output_dim: cfg.linear_output_dim() as u32,
                chunk_size: GDN_CHUNK_SIZE as u32,
                n_tokens,
                num_chunks,
                active_chunk: 0,
                scale: 1.0 / (cfg.linear_key_head_dim as f32).sqrt(),
                eps: cfg.rms_norm_eps,
            })
        }

        /// Read top-k compact candidates from the appropriate scratch buffer.
        ///
        /// SAFETY: GPU command buffer must have completed before calling.
        unsafe fn read_topk_candidates(
            &self,
            which: u8,
            k: usize,
        ) -> Vec<crate::sampling::Candidate> {
            let buf = if which == 0 {
                &self.session.activations.topk_scratch_a
            } else {
                &self.session.activations.topk_scratch_b
            };
            let ptr = buf.contents() as *const GpuCandidate;
            let mut out = Vec::with_capacity(k);
            for i in 0..k {
                let gc = *ptr.add(i);
                out.push(crate::sampling::Candidate {
                    token_id: gc.token_id,
                    logit: gc.logit,
                });
            }
            out
        }

        // ===================================================================
        // Layer importance scoring
        // ===================================================================

        /// Run a prefix-masked forward pass capturing the last-token hidden state
        /// before and after each active layer for importance scoring.
        ///
        /// For N active layers, performs N sequential model passes where pass `i`
        /// activates only layers `[active_0 .. active_i]`.  Each pass processes all
        /// tokens in `token_ids` via `forward_step_inner`.  Because preceding layers
        /// are identical across passes, `output[i]` is the exact post-layer-i hidden
        /// state from the full-model run.
        ///
        /// Intended for offline calibration only.
        /// Cost: O(N²) in active layers — performs N passes of increasing length
        /// (pass i activates layers 0..=i, total layer-token work ≈ N²/2).
        /// Acceptable for calibration-only usage where N ≤ 64.
        fn forward_prefill_layer_traces_last_token(
            &mut self,
            token_ids: &[u32],
        ) -> Result<Vec<LayerTrace>, crate::error::InferenceError> {
            if token_ids.is_empty() {
                return Err(crate::error::InferenceError::Inference(
                    "forward_prefill_layer_traces_last_token: empty token sequence".to_string(),
                ));
            }
            // `reset_state` below zeroes `kv_cache.seq_len`, so this call's own
            // `forward_step_inner` loop drives `position` from 0..token_ids.len() — a
            // caller-supplied length reaching this public entry point via
            // `score_layer_importance`'s `calibration_prompts`. Validate against the
            // session's RoPE/KV capacity before any pass dispatches, same as
            // `check_forward_step_capacity` does for `forward_step`.
            self.check_forward_range_capacity(0, token_ids.len(), false)?;
            let orig_cfg = self.engine.config.clone();
            let hidden = orig_cfg.hidden_size;
            let n = token_ids.len();

            let active_indices: Vec<usize> = (0..orig_cfg.num_hidden_layers)
                .filter(|&i| orig_cfg.is_layer_active(i))
                .collect();

            // Input to the first active layer = embedding of the last token.
            let last_id = *token_ids.last().unwrap();
            assert!(
                (last_id as usize) < orig_cfg.vocab_size,
                "token_id {last_id} >= vocab_size {}",
                orig_cfg.vocab_size
            );
            // SAFETY: embed_tokens is StorageModeShared f16, no GPU command in flight.
            let first_input: Vec<f32> = unsafe {
                let src = (self.engine.embed_tokens.contents() as *const u16)
                    .add(last_id as usize * hidden);
                let mut out = vec![0.0f32; hidden];
                // SAFETY: embed_tokens row has hidden u16 values; out has hidden f32 values.
                convert_f16_row(src, out.as_mut_ptr(), hidden);
                out
            };

            let mut traces: Vec<LayerTrace> = Vec::with_capacity(active_indices.len());
            let mut prev_output: Vec<f32> = first_input;

            for (active_pos, &layer_i) in active_indices.iter().enumerate() {
                // Mask: activate only layers active_indices[0..=active_pos].
                let mut trace_mask = vec![false; orig_cfg.num_hidden_layers];
                for j in 0..=active_pos {
                    trace_mask[active_indices[j]] = true;
                }
                self.engine.config.layer_mask = trace_mask;
                self.reset_state();

                // Process all tokens; capture pre-final hidden for the last one.
                for (pos, &id) in token_ids.iter().enumerate() {
                    let is_last = pos == n - 1;
                    let _ = self.forward_step_inner(
                        id,
                        pos,
                        is_last,
                        crate::forward::signpost::Scope::NotDecode,
                    );
                }

                // last_pre_final_hidden = post-last-active-layer hidden for the last token.
                let output_hidden = self.session.last_pre_final_hidden.clone();
                let input_hidden = prev_output.clone();
                prev_output = output_hidden.clone();

                traces.push(LayerTrace {
                    layer_idx: layer_i,
                    input: input_hidden,
                    output: output_hidden,
                });
            }

            // Restore original config and leave model in a clean state.
            self.engine.config.layer_mask = orig_cfg.layer_mask;
            self.reset_state();

            Ok(traces)
        }

        /// Score each active layer's importance by measuring how much it changes the
        /// last-token hidden state across `calibration_prompts`.
        ///
        /// Layers with high mean cosine similarity change the hidden state least and
        /// are the safest prune candidates.  Returns a [`LayerPruningPlan`] containing
        /// per-layer scores (sorted highest cosine first), a type-balanced recommended
        /// mask for `prune_layers` removals, and an optional Gromov warning when
        /// `prune_layers` exceeds 20 % of the total layer count.
        pub fn score_layer_importance(
            &mut self,
            calibration_prompts: &[Vec<u32>],
            prune_layers: usize,
        ) -> Result<LayerPruningPlan, crate::error::InferenceError> {
            use crate::model::qwen35_config::LayerType;
            let cfg = self.engine.config.clone();

            let mut sums = vec![0.0f64; cfg.num_hidden_layers];
            let mut counts = vec![0usize; cfg.num_hidden_layers];

            for prompt in calibration_prompts {
                let traces = self.forward_prefill_layer_traces_last_token(prompt)?;
                for trace in &traces {
                    let cos = cosine_similarity(&trace.input, &trace.output);
                    sums[trace.layer_idx] += cos as f64;
                    counts[trace.layer_idx] += 1;
                }
            }

            let mut scores: Vec<LayerImportanceScore> = (0..cfg.num_hidden_layers)
                .filter(|&i| cfg.is_layer_active(i))
                .map(|i| {
                    let mean_cosine = if counts[i] == 0 {
                        0.0
                    } else {
                        (sums[i] / counts[i] as f64) as f32
                    };
                    LayerImportanceScore {
                        layer_idx: i,
                        layer_type: cfg.layer_types[i],
                        mean_cosine,
                        importance: 1.0 - mean_cosine,
                    }
                })
                .collect();

            // Highest cosine first = lowest importance = best prune candidate.
            scores.sort_by(|a, b| b.mean_cosine.total_cmp(&a.mean_cosine));

            // Type-balanced quota: ~1 GQA removed per 3 GDN.
            let gqa_quota = (prune_layers + 2) / 4;
            let gdn_quota = prune_layers.saturating_sub(gqa_quota);
            let mut gdn_left = gdn_quota;
            let mut gqa_left = gqa_quota;
            let mut recommended_mask = cfg.layer_mask.clone();

            for score in &scores {
                match score.layer_type {
                    LayerType::LinearAttention if gdn_left > 0 => {
                        recommended_mask[score.layer_idx] = false;
                        gdn_left -= 1;
                    }
                    LayerType::FullAttention if gqa_left > 0 => {
                        recommended_mask[score.layer_idx] = false;
                        gqa_left -= 1;
                    }
                    _ => {}
                }
                if gdn_left == 0 && gqa_left == 0 {
                    break;
                }
            }

            let warning = if prune_layers * 5 > cfg.num_hidden_layers {
                Some(format!(
                    "Gromov warning: requested prune_layers={prune_layers} exceeds 20% of {} layers",
                    cfg.num_hidden_layers
                ))
            } else {
                None
            };

            Ok(LayerPruningPlan {
                scores,
                recommended_mask,
                warning,
            })
        }
    }

    // -----------------------------------------------------------------------
    // Compact sampling helper (replaces private sample_token)
    // -----------------------------------------------------------------------

    /// Sample from a compact candidate set returned by GPU top-k.
    ///
    /// The candidates have already been selected by the GPU (no full-vocab
    /// scan needed).  Repetition penalty is applied CPU-side to the compact
    /// set as a round-0 approximation (accurate when recently-repeated tokens
    /// are not ranked far below the top-k boundary).
    pub(crate) fn sample_from_candidates(
        candidates: &[crate::sampling::Candidate],
        cfg: &GenerateConfig,
        previous_ids: &[u32],
        rng_state: &mut u64,
    ) -> u32 {
        // `min_p` is read from `GenerateConfig` (0.0 = disabled, its default,
        // when no caller sets it).
        sample_from_candidates_impl(candidates, cfg, previous_ids, rng_state, cfg.min_p)
    }

    fn sample_from_candidates_impl(
        candidates: &[crate::sampling::Candidate],
        cfg: &GenerateConfig,
        previous_ids: &[u32],
        rng_state: &mut u64,
        min_p: f32,
    ) -> u32 {
        use crate::sampling::CandidateSet;
        let mut cs = CandidateSet::from_candidates(candidates.to_vec());

        if cfg.repetition_penalty != 1.0 {
            cs.apply_repetition_penalty(previous_ids, cfg.repetition_penalty);
        }
        // A degenerate temperature (non-finite, <= 0, or finite-but-tiny so 1/t
        // overflows) has no valid scaling; the t -> 0+ limit is argmax.
        if crate::sampling::temperature_degenerate(cfg.temperature) {
            return cs.argmax();
        }
        cs.apply_temperature(cfg.temperature);

        let raw = xorshift64(rng_state);
        // #351: `(raw as f64 / u64::MAX as f64) as f32` rounds up to exactly 1.0
        // when raw == u64::MAX, and sample_top_p with r == 1.0 can select the
        // worst-probability token. Route through the canonical [0, 1) helper
        // (provably < 1.0), matching the CPU sampling path.
        let r = crate::sampling::uniform_f32_from_u64(raw);
        cs.sample_min_p_top_p(min_p, cfg.top_p, r)
    }

    #[cfg(test)]
    mod min_p_compact_sampling_tests;

    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Full-logit sampling fallback (used when top_k == 0 or top_k > MAX_TOP_K).
    ///
    /// Delegates to `crate::sampling::sample_full_logits`, the shared engine
    /// also used by the Qwen CPU decode loops
    /// (`model/qwen35/sampling.rs::sample_token`). See that function's doc
    /// comment for the three optimization levers applied here relative to the
    /// old per-call `CandidateSet::from_full_logits` implementation: repetition
    /// penalty over the unique-id set from `previous_ids` (not a per-candidate
    /// `.contains()` scan over the full vocabulary), a streaming min-heap
    /// partial top-k select feeding the min-p/top-p softmax draw (so softmax
    /// only ever runs over the surviving k candidates), and thread-local scratch
    /// buffers reused across decode steps instead of a fresh vocab-sized
    /// `CandidateSet` per token.
    pub(crate) fn sample_token(
        logits: &[f32],
        cfg: &GenerateConfig,
        previous_ids: &[u32],
        rng_state: &mut u64,
    ) -> u32 {
        // `min_p` is read from `GenerateConfig` (0.0 = disabled, its default,
        // when no caller sets it). `top_n_sigma` has no `GenerateConfig` field
        // yet and no production entry point sets it, so it stays disabled.
        crate::sampling::sample_full_logits(logits, cfg, previous_ids, rng_state, cfg.min_p, 0.0)
    }

    /// Signpost-traced sampling shared by every autoregressive decode loop's
    /// per-step token pick: opens the `decode.sample` interval, then routes
    /// to the compact-candidate or full-logits sampler.
    ///
    /// Centralizes what was five independent copy-pasted call sites: two of
    /// the five -- the multimodal decode loops --
    /// had already drifted to call `sample_token` directly, with neither the
    /// interval nor the compact-route policy the other three loops apply).
    /// `compact` is `Some(candidates)` for the GPU-top-k route (mirrors the
    /// `use_compact` branch every call site used to inline) and `None` for
    /// callers with no compact route, which is the multimodal decode loops'
    /// only case -- they gain `decode.sample` instrumentation as a side
    /// effect of routing through here instead of `sample_token` directly.
    pub(crate) fn sample_decode_traced(
        compact: Option<&[crate::sampling::Candidate]>,
        step_logits: &[f32],
        cfg: &GenerateConfig,
        previous_ids: &[u32],
        rng_state: &mut u64,
    ) -> u32 {
        let _signpost_sample =
            crate::forward::signpost::interval(crate::forward::signpost::Label::DecodeSample);
        match compact {
            Some(candidates) => sample_from_candidates(candidates, cfg, previous_ids, rng_state),
            None => sample_token(step_logits, cfg, previous_ids, rng_state),
        }
    }

    fn decode_tokens(tokenizer: &BpeTokenizer, ids: &[u32]) -> String {
        lattice_tokenizer::__internal::decode_tokens(tokenizer, ids)
    }

    // -----------------------------------------------------------------------
    // Chat Completion API
    // -----------------------------------------------------------------------

    const TEXT_CHAT_IMAGE_ERROR: &str =
        "inline image messages require the multimodal vision generation path";

    fn reject_inline_images_for_text_chat(
        messages: &[ChatMessage],
    ) -> Result<(), crate::error::InferenceError> {
        if messages.iter().any(|message| message.image.is_some()) {
            return Err(crate::error::InferenceError::InvalidInput(
                TEXT_CHAT_IMAGE_ERROR.to_string(),
            ));
        }
        Ok(())
    }

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

    impl MetalQwen35State {
        /// **Unstable**: chat completion; stop token handling and output format may change.
        ///
        /// Chat completion: format messages with Qwen3.5 template, generate response.
        /// Stops on <|im_end|> or EOS or max tokens.
        ///
        /// # Errors
        ///
        /// Returns `InferenceError::InvalidInput` if `messages` contains an
        /// inline image, which requires the multimodal vision entry point, or
        /// if grammar-constrained decoding blocks every token — propagated
        /// from [`Self::generate`] (#611).
        pub fn chat_completion(
            &mut self,
            messages: &[ChatMessage],
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> Result<ChatCompletionOutput, crate::error::InferenceError> {
            reject_inline_images_for_text_chat(messages)?;
            let prompt = format_chat_template(messages);
            // Add <|im_end|> as stop token
            let mut cfg = gen_cfg.clone();
            if let Some(im_end_id) = tokenizer.special_token_id("<|im_end|>")
                && !cfg.stop_token_ids.contains(&im_end_id)
            {
                cfg.stop_token_ids.push(im_end_id);
            }
            let result = self.generate(&prompt, tokenizer, &cfg)?;
            let text = result.text.trim_end().to_string();
            Ok(ChatCompletionOutput {
                message: ChatMessage::assistant(text),
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.generated_tokens,
                stopped: result.stopped,
            })
        }

        /// **Unstable**: streaming generation; callback signature may change.
        ///
        /// Streaming generation: calls `on_token` for each generated token.
        /// Returns total generated token count.
        /// The callback receives (token_text, token_id) and returns true to continue.
        ///
        /// Cancellable only through `on_token`'s return value. Callers that need to
        /// observe cancellation independently of text emission -- so an
        /// empty-delta decode iteration, or the uninterruptible prefill call
        /// itself, can't silently swallow a disconnect -- should call
        /// [`Self::generate_streaming_with_cancel`] instead.
        pub fn generate_streaming<F>(
            &mut self,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
        ) -> Result<GenerateOutput, crate::error::InferenceError>
        where
            F: FnMut(&str, u32) -> bool,
        {
            self.generate_streaming_with_cancel(prompt, tokenizer, gen_cfg, on_token, || false)
        }

        /// Cancellation-aware sibling of [`Self::generate_streaming`].
        ///
        /// `should_cancel` is polled independently of `on_token`: before starting
        /// the prefill pass, immediately after it returns, and at the top of every
        /// decode iteration -- all before any further expensive GPU work runs for
        /// that step. This closes two gaps `on_token`-only cancellation has:
        /// prefill itself has no callback point at all (a long prompt can run to
        /// completion after the client is already gone), and the decode loop only
        /// calls `on_token` for non-empty text deltas, so a run of tokens that
        /// decode to an incomplete UTF-8 tail (a multi-token codepoint mid-stream)
        /// would otherwise skip the cancellation check for those iterations too.
        /// SSE emission itself stays gated on non-empty deltas, unchanged.
        ///
        /// `generate_streaming` is the `should_cancel = || false` convenience form
        /// and is otherwise identical; both share this one implementation.
        ///
        /// # Errors
        ///
        /// Returns `InferenceError::InvalidInput` if grammar-constrained decoding
        /// blocks every token (prefill or any decode step) — mirrors the CPU
        /// `generate_streaming` contract (#611).
        /// Returns `InferenceError::Inference` when the prompt plus effective decode
        /// cap exceeds [`Self::max_context`].
        /// Returns `InferenceError::UnsupportedModel` for a multi-token prompt on a
        /// model with MoE layers and no active LoRA adapter, for the reason given on
        /// [`Self::generate`]. `generate_streaming` shares this implementation and so
        /// reports the same error.
        pub fn generate_streaming_with_cancel<F, C>(
            &mut self,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
            should_cancel: C,
        ) -> Result<GenerateOutput, crate::error::InferenceError>
        where
            F: FnMut(&str, u32) -> bool,
            C: FnMut() -> bool,
        {
            let plan = match prepare_generation(
                tokenizer,
                prompt,
                gen_cfg,
                self.engine.config.vocab_size,
                self.max_context(),
                GenerationEntryContract::MetalStreaming,
            )? {
                GenerationPreparation::Complete(output) => return Ok(output),
                GenerationPreparation::Ready(plan) => plan,
            };
            let vocab_size = self.engine.config.vocab_size;
            let eos_token_id = self.engine.config.eos_token_id;
            let prompt_ids = plan.prompt_ids.clone();

            // The session resets the state and engages the planned sampling route;
            // dropping it tears the route down on every exit path below.
            let mut session = crate::decoder::qwen_metal::QwenMetalSession::new(
                self,
                plan,
                gen_cfg,
                crate::decoder::qwen_metal::MetalEntryProfile::Streaming,
            )?;

            // Budget forcing: resolve the </think> token id once before decoding.
            // Only paid when reasoning_budget is Some; None path is a single branch.
            let think_close_id = crate::model::qwen35::resolve_reasoning_close_token(
                tokenizer,
                gen_cfg.reasoning_budget,
                gen_cfg.enable_thinking,
                vocab_size,
            )?;

            // KV capacity: `prepare_generation` admitted this request only if
            // `prompt_len + decode_cap(..) <= max_context()` (`check_context_budget`),
            // and the driver runs at most `decode_cap(..) - 1` decode steps after
            // prefill, so the last step writes position `max_context() - 2` at most.
            // No admitted request fills the cache: this entry never stops with
            // `StopReason::KvFull`, and the session's full-cache refusal cannot fire.
            crate::decoder::qwen_metal::run_streaming(
                &mut session,
                gen_cfg,
                think_close_id,
                &prompt_ids,
                eos_token_id,
                tokenizer,
                on_token,
                should_cancel,
            )
        }

        // ---------------------------------------------------------------------------
        // Q4 direct-load helpers
        // ---------------------------------------------------------------------------

        /// Build the sanitized file-stem from a tensor name (dots/slashes → underscores).
        fn q4_tensor_path(
            dir: &std::path::Path,
            tensor_name: &str,
            ext: &str,
        ) -> std::path::PathBuf {
            let sanitized: String = tensor_name
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            dir.join(format!("{sanitized}.{ext}"))
        }

        /// Create a Metal buffer from pre-quantized Q4 block bytes (raw `[u8]`).
        ///
        /// The bytes are the serialized `Q4Block` array as written by `save_q4_file`
        /// (after the file header), i.e. `n_blocks × 20` raw bytes.
        fn make_buffer_from_q4_raw(device: &Device, raw: &[u8], label: &str) -> Buffer {
            let buf = device.new_buffer_with_data(
                raw.as_ptr() as *const _,
                raw.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            buf.set_label(label);
            buf
        }

        /// Dequantize a [`Q4Tensor`] to f16 (as `u16` bit patterns) and create a Metal buffer.
        ///
        /// Used for `embed_tokens`: the CPU embedding lookup reads f16 values from this buffer,
        /// so we need an f16 Metal buffer even when the on-disk format is Q4.
        fn make_buffer_f16_from_q4(
            device: &Device,
            tensor: &crate::weights::q4_weights::Q4Tensor,
            label: &str,
        ) -> Buffer {
            use crate::weights::q4_weights::{q4_f16_to_f32, q4_f32_to_f16};
            let mut f16_data: Vec<u16> = Vec::with_capacity(tensor.original_len);
            for block in &tensor.blocks {
                let scale = q4_f16_to_f32(block.scale);
                let bias = q4_f16_to_f32(block.bias);
                for b in 0..16 {
                    let byte_val = block.packed[b];
                    let w0 = (byte_val & 0x0f) as f32 * scale + bias;
                    let w1 = (byte_val >> 4) as f32 * scale + bias;
                    f16_data.push(q4_f32_to_f16(w0));
                    f16_data.push(q4_f32_to_f16(w1));
                }
            }
            f16_data.truncate(tensor.original_len);
            let byte_len = (f16_data.len() * std::mem::size_of::<u16>()) as u64;
            let buf = device.new_buffer_with_data(
                f16_data.as_ptr() as *const _,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            );
            buf.set_label(label);
            buf
        }

        /// Load raw Q4 block bytes from a `.q4` file (strips the file header).
        ///
        /// Returns the `Q4Block` bytes and the `original_len` (for validation).
        fn load_q4_raw_bytes(
            path: &std::path::Path,
            expected_shape: &[usize],
        ) -> Result<(Vec<u8>, usize), String> {
            use crate::weights::q4_weights::load_q4_file_checked;
            let tensor = load_q4_file_checked(path, expected_shape)
                .map_err(|e| format!("failed to load Q4 file {}: {e}", path.display()))?;
            let n_blocks = tensor.blocks.len();
            // Re-serialise the blocks into raw bytes (20 bytes each, Q4Block is
            // #[repr(C)] with scale + bias + 16 packed nibbles).
            // SAFETY: Q4Block is #[repr(C)] size 20, alignment 2 (from leading
            // `scale: u16`); byte-cast is valid because target element type is u8.
            let raw: Vec<u8> = unsafe {
                std::slice::from_raw_parts(tensor.blocks.as_ptr().cast::<u8>(), n_blocks * 20)
                    .to_vec()
            };
            Ok((raw, tensor.original_len))
        }

        /// Mmap a `.q4` file and dequantize its payload directly into an f16
        /// Metal buffer, without ever materializing the raw Q4 bytes as an
        /// owned `Vec` first (unlike `load_q4_raw_bytes`/`load_q4_file`, which
        /// `read_exact` the whole file into a fresh heap allocation before any
        /// dequant happens). MoE routed-expert `.experts.gate_up_proj` /
        /// `.experts.down_proj` files hold every expert for a layer fused into
        /// one array and can be tens of GB; mmap keeps the source pages backed
        /// by the file itself (evictable under memory pressure, like every
        /// other Q4WeightBuf tensor `from_q4_dir` loads) instead of pinning
        /// them in anonymous process memory for the duration of the read. The
        /// output is f16 because the MoE decode kernels (`moe_expert_gemv`,
        /// `gemv_decode`) read `half`, not packed Q4 — ADR-053's MoE Metal
        /// kernels were built against the safetensors-upload path, which
        /// already materializes experts as f16 (see `MetalQwen35Engine::new()`
        /// around the `FeedForwardWeights::Moe` arm); no Q4-native MoE GEMV
        /// kernel exists yet, so this mirrors that convention rather than
        /// introducing a second on-GPU numeric format for MoE only.
        ///
        /// `expected_shape` is validated against the on-disk header BEFORE the
        /// mmap or any allocation happens: `validate_q4_header_payload_bounds`
        /// only proves the header is *self*-consistent (`product(shape) ==
        /// original_len` and the file is physically long enough), never that
        /// the shape matches what `encode_moe_ffn` requires. A structurally
        /// valid but short or same-numel-transposed expert file (e.g. a `[1,
        /// H]` router file under a 4-expert config, or `[E, H, 2I]` in place
        /// of `[E, 2I, H]`) would otherwise pass, hand `encode_moe_ffn` a
        /// buffer smaller than — or laid out differently than — the
        /// `num_experts * ...` slice it forms from the raw pointer, and
        /// violate that unsafe function's buffer-size invariant (mirrors the
        /// shape check `resolve_mtp_projection`/
        /// `resolve_mtp_norm` already apply to MTP tensors above).
        fn load_q4_mmap_dequant_f16(
            device: &Device,
            path: &std::path::Path,
            label: &str,
            expected_shape: &[usize],
        ) -> Result<Buffer, String> {
            use crate::weights::q4_weights::{
                Q4BlockCheck, open_and_mmap_q4_file, q4_f16_to_f32, q4_f32_to_f16,
                validate_q4_block_metadata,
            };
            // Per-block scale/bias is validated inside the dequant loop below,
            // which already reads every block, instead of by a separate pass
            // over the same bytes.
            let (header, mmap, _) = open_and_mmap_q4_file(
                path,
                Some(expected_shape),
                Q4BlockCheck::InCallerTraversal {
                    traversal: "load_q4_mmap_dequant_f16 dequant loop",
                },
            )?;
            let payload = mmap.get(header.payload_offset as usize..).ok_or_else(|| {
                format!(
                    "{}: payload_offset {} beyond mapped length {}",
                    path.display(),
                    header.payload_offset,
                    mmap.len()
                )
            })?;
            let source = path.display().to_string();
            let tensor_name = path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or("native Q4 tensor");
            let mut f16_data: Vec<u16> = Vec::with_capacity(header.original_len);
            for (index, chunk) in payload.chunks_exact(20).enumerate() {
                let scale_bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
                let bias_bits = u16::from_ne_bytes([chunk[2], chunk[3]]);
                validate_q4_block_metadata(&source, tensor_name, index, scale_bits, bias_bits)
                    .map_err(|e| e.to_string())?;
                let scale = q4_f16_to_f32(scale_bits);
                let bias = q4_f16_to_f32(bias_bits);
                for b in 0..16 {
                    let byte_val = chunk[4 + b];
                    f16_data.push(q4_f32_to_f16((byte_val & 0x0f) as f32 * scale + bias));
                    f16_data.push(q4_f32_to_f16((byte_val >> 4) as f32 * scale + bias));
                }
            }
            f16_data.truncate(header.original_len);
            let byte_len = (f16_data.len() * std::mem::size_of::<u16>()) as u64;
            let buf = device.new_buffer_with_data(
                f16_data.as_ptr() as *const _,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            );
            buf.set_label(label);
            Ok(buf)
        }

        /// Same mmap route as [`Self::load_q4_mmap_dequant_f16`], but dequantizes
        /// to f32 for the two small MoE tensors the CPU reads directly every
        /// forward step: the router gate (`mlp.gate.weight`, routing logits) and
        /// the shared-expert scalar gate (`mlp.shared_expert_gate.weight`,
        /// sigmoid gate value) — both are read via `Buffer::contents() as *const
        /// f32` in `encode_moe_ffn`, so they must land as f32, not f16.
        ///
        /// `expected_shape` is validated the same way and at the same point as
        /// in [`Self::load_q4_mmap_dequant_f16`]: before
        /// mmap/allocation, against the raw header only.
        fn load_q4_mmap_dequant_f32(
            device: &Device,
            path: &std::path::Path,
            label: &str,
            expected_shape: &[usize],
        ) -> Result<Buffer, String> {
            use crate::weights::q4_weights::{
                Q4BlockCheck, dequantize_row_q4_0, open_and_mmap_q4_file,
                validate_q4_block_metadata,
            };
            let (header, mmap, _) = open_and_mmap_q4_file(
                path,
                Some(expected_shape),
                Q4BlockCheck::InCallerTraversal {
                    traversal: "load_q4_mmap_dequant_f32 scale/bias loop",
                },
            )?;
            let payload = mmap.get(header.payload_offset as usize..).ok_or_else(|| {
                format!(
                    "{}: payload_offset {} beyond mapped length {}",
                    path.display(),
                    header.payload_offset,
                    mmap.len()
                )
            })?;
            // Only the two small CPU-read MoE scalar gates route through this
            // f32 path (router gate, shared-expert gate — see doc comment
            // above); `dequantize_row_q4_0` is shared with non-ingress call
            // sites and cannot itself validate, so scale/bias metadata is
            // checked here against the same already-mapped bytes it dequantizes.
            let source = path.display().to_string();
            let tensor_name = path
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or("native Q4 tensor");
            for (index, chunk) in payload.chunks_exact(20).enumerate() {
                let scale_bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
                let bias_bits = u16::from_ne_bytes([chunk[2], chunk[3]]);
                validate_q4_block_metadata(&source, tensor_name, index, scale_bits, bias_bits)
                    .map_err(|e| e.to_string())?;
            }
            let values = dequantize_row_q4_0(payload, header.original_len);
            Ok(make_buffer(device, &values, label))
        }

        /// Build MoE FFN weights for layer `prefix` from Q4-dir tensors.
        ///
        /// Mirrors `MetalQwen35Engine::new()`'s MoE branch (ADR-053, the
        /// safetensors-upload constructor around the `FeedForwardWeights::Moe`
        /// arm): same `MoeMetalBuffers` layout, same headroom guard, same f16
        /// GPU representation for routed/shared expert weights and f32 for the
        /// two CPU-read router/gate scalars. The only difference is the source:
        /// every tensor here comes from a `.q4` file in `q4_dir` via
        /// [`Self::load_q4_mmap_dequant_f16`] / [`Self::load_q4_mmap_dequant_f32`]
        /// (mmap, not a full-file read) instead of an in-memory `ModelWeights`
        /// produced by the CPU safetensors reader.
        fn load_moe_ffn_q4(
            device: &Device,
            q4_dir: &std::path::Path,
            cfg: &Qwen35Config,
            prefix: &str,
            layer_idx: usize,
            moe_layers_already_sized: usize,
        ) -> Result<MetalFfnWeights, String> {
            use crate::forward::moe_expert_cache::{
                ExpertSlotCache, MoeExpertCacheConfig, moe_expert_cache_num_slots,
            };

            let num_experts = cfg
                .num_experts
                .ok_or_else(|| "from_q4_dir: MoE config missing num_experts".to_string())?;
            let top_k = cfg
                .num_experts_per_tok
                .ok_or_else(|| "from_q4_dir: MoE config missing num_experts_per_tok".to_string())?;
            let inter = cfg.moe_intermediate_size();
            let shared_inter = cfg.shared_expert_intermediate_size();
            let hidden = cfg.hidden_size;

            // #682 Stage 1: routed experts are no longer eagerly dequantized
            // to full-size resident buffers (that path needed ~61 GiB
            // f16-resident for Qwen3.5-35B-A3B, independent of how many
            // experts a token actually activates — see PLAN.md §1). Instead,
            // size a bounded LRU cache of per-expert slots against this
            // device's *remaining* memory budget: `num_experts` slots (the
            // "zero-eviction fast path", functionally the old eager behavior
            // but lazily populated and evictable) when that fits under 0.85
            // × recommendedMaxWorkingSetSize minus what this device already
            // has allocated (fixed model buffers, KV/GDN/prefix state, and
            // any earlier MoE layer's own cache slots), split evenly across
            // the MoE layers not yet sized (this layer included), else
            // auto-shrunk (floored at `top_k`, below which the cache cannot
            // serve even one token's routed-expert set). `moe_layers_
            // already_sized` must be this layer's live ordinal among MoE
            // layers processed so far on this device — every earlier one
            // already built its own `ExpertSlotCache` buffers, which are
            // part of `current_allocated_size()` by the time this call
            // runs, so dividing by the *total* MoE layer count on every
            // call (instead of by the layers not yet sized) would subtract
            // each already-sized layer's bytes from the numerator while
            // still dividing by a constant denominator that includes them,
            // geometrically starving later layers. See
            // `moe_expert_cache_num_slots`'s doc comment for exactly which
            // residency classes `current_allocated_size()` covers, which it
            // doesn't (CPU-side allocations and the OS/process reserve),
            // and the known limitation this does not close (later layers'
            // not-yet-loaded dense weights are also invisible to it).
            let gate_up_bytes_per_expert = (2 * inter * hidden * 2) as u64; // f16
            let down_bytes_per_expert = (hidden * inter * 2) as u64; // f16
            let per_expert_bytes_total = gate_up_bytes_per_expert + down_bytes_per_expert;
            let num_moe_layers = cfg.num_active_layers();
            let max_working = device.recommended_max_working_set_size();
            let already_allocated = device.current_allocated_size();
            let cache_cfg = MoeExpertCacheConfig::from_env()
                .map_err(|e| format!("from_q4_dir: MoE layer {layer_idx}: {e}"))?;
            let num_slots = moe_expert_cache_num_slots(
                &cache_cfg,
                num_experts,
                top_k,
                per_expert_bytes_total,
                num_moe_layers,
                max_working,
                already_allocated,
                moe_layers_already_sized,
            )
            .map_err(|e| format!("from_q4_dir: MoE layer {layer_idx}: {e}"))?;

            let routed_gate_up_cache = std::cell::RefCell::new(ExpertSlotCache::new(
                device,
                &Self::q4_tensor_path(q4_dir, &format!("{prefix}.mlp.experts.gate_up_proj"), "q4"),
                &[num_experts, 2 * inter, hidden],
                num_slots,
                &format!("L{layer_idx}.moe.gate_up.f16.cache"),
            )?);
            let routed_down_cache = std::cell::RefCell::new(ExpertSlotCache::new(
                device,
                &Self::q4_tensor_path(q4_dir, &format!("{prefix}.mlp.experts.down_proj"), "q4"),
                &[num_experts, hidden, inter],
                num_slots,
                &format!("L{layer_idx}.moe.down.f16.cache"),
            )?);
            let shared_gate_proj = Self::load_q4_mmap_dequant_f16(
                device,
                &Self::q4_tensor_path(
                    q4_dir,
                    &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                    "q4",
                ),
                &format!("L{layer_idx}.moe.sh_gate.f16"),
                &[shared_inter, hidden],
            )?;
            let shared_up_proj = Self::load_q4_mmap_dequant_f16(
                device,
                &Self::q4_tensor_path(
                    q4_dir,
                    &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                    "q4",
                ),
                &format!("L{layer_idx}.moe.sh_up.f16"),
                &[shared_inter, hidden],
            )?;
            let shared_down_proj = Self::load_q4_mmap_dequant_f16(
                device,
                &Self::q4_tensor_path(
                    q4_dir,
                    &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                    "q4",
                ),
                &format!("L{layer_idx}.moe.sh_down.f16"),
                &[hidden, shared_inter],
            )?;
            let router_gate = Self::load_q4_mmap_dequant_f32(
                device,
                &Self::q4_tensor_path(q4_dir, &format!("{prefix}.mlp.gate.weight"), "q4"),
                &format!("L{layer_idx}.moe.router.f32"),
                &[num_experts, hidden],
            )?;
            let shared_expert_gate = Self::load_q4_mmap_dequant_f32(
                device,
                &Self::q4_tensor_path(
                    q4_dir,
                    &format!("{prefix}.mlp.shared_expert_gate.weight"),
                    "q4",
                ),
                &format!("L{layer_idx}.moe.sh_gate_scalar.f32"),
                &[1, hidden],
            )?;

            let scratch_gate = make_zero_buffer(
                device,
                shared_inter.max(inter),
                &format!("L{layer_idx}.moe.scr_gate"),
            );
            let scratch_up = make_zero_buffer(
                device,
                shared_inter.max(inter),
                &format!("L{layer_idx}.moe.scr_up"),
            );
            let scratch_expert_out =
                make_zero_buffer(device, hidden, &format!("L{layer_idx}.moe.scr_exp_out"));
            let scratch_out =
                make_zero_buffer(device, hidden, &format!("L{layer_idx}.moe.scr_out"));

            Ok(MetalFfnWeights::Moe(Box::new(MoeMetalBuffers {
                routed: RoutedExpertStorage::Cached {
                    gate_up: Box::new(routed_gate_up_cache),
                    down: Box::new(routed_down_cache),
                },
                shared_gate_proj,
                shared_up_proj,
                shared_down_proj,
                router_gate,
                shared_expert_gate,
                scratch_gate,
                scratch_up,
                scratch_expert_out,
                scratch_out,
                num_experts,
                inter,
                shared_inter,
                hidden,
                top_k,
            })))
        }

        /// Load MTP weights from a Q4 directory and build `MetalMtpRuntime`.
        ///
        /// Returns no weights if the model has no MTP layers or if any weight file is
        /// missing; in the latter case the first missing file path is carried in the
        /// result so the caller can warn instead of silently falling back.
        ///
        /// `is_quarot` must reflect whether `q4_dir` is a QuaRot (rotated-space)
        /// checkpoint — determined by the caller from `quarot_seed` presence
        /// (`quantize_index.json` object manifest or the legacy `config.json`
        /// field) BEFORE calling this function, so flavor detection never
        /// depends on which files happen to exist per-tensor (#636).
        /// Returns `Err` (hard load failure, not a gracefully-disabled
        /// warning) when an MTP tensor is present but untrustworthy: an
        /// incompatible stale `.q4` sibling under `is_quarot`, or a
        /// shape/transpose mismatch against the tensor's expected dimensions.
        fn load_mtp_q4_weights(
            q4_dir: &std::path::Path,
            cfg: &Qwen35Config,
            device: &Device,
            is_quarot: bool,
        ) -> Result<MtpQ4LoadResult, String> {
            if cfg.mtp_num_hidden_layers == 0 {
                return Ok(MtpQ4LoadResult {
                    weights: None,
                    first_missing_file: None,
                });
            }

            let hidden = cfg.hidden_size;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let inter = cfg.intermediate_size;
            let head_dim = cfg.head_dim;
            // Projections may load .q4 only on a plain (non-QuaRot) checkpoint;
            // QuaRot dirs must keep every MTP tensor unrotated .f16.
            let allow_q4 = !is_quarot;

            // Helper: resolve + widen an MTP norm to an f32 Metal buffer (for
            // RMSNorm kernels that read f32 input).
            let load_norm_as_f32 =
                |name: &str, expected_shape: &[usize], label: &str| -> Result<Buffer, MtpLoadErr> {
                    let (values, _) = resolve_mtp_norm(q4_dir, name, expected_shape)?;
                    Ok(make_buffer(device, &values, label))
                };

            // Helper: resolve one of the 8 MTP projections and materialize it
            // as an f16 Metal buffer for `dispatch_matmul_half` / `gemv_decode_m1`
            // (ADR-051 Phase 1: both flavors store MTP projections unrotated).
            let load_proj_as_half =
                |name: &str, expected_shape: &[usize], label: &str| -> Result<Buffer, MtpLoadErr> {
                    match resolve_mtp_projection(q4_dir, name, expected_shape, allow_q4)? {
                        MtpTensorSource::Q4(tensor) => Ok(
                            MetalQwen35State::make_buffer_f16_from_q4(device, &tensor, label),
                        ),
                        MtpTensorSource::F16 { values, .. } => {
                            Ok(make_buffer_f16(device, &values, label))
                        }
                    }
                };

            let weights = (|| -> Result<MetalMtpWeights, MtpLoadErr> {
                // --- Layer 0 weights (ADR-051: 8 projections as f16, 7 norms as f32) ---
                let input_layernorm = load_norm_as_f32(
                    "mtp.layers.0.input_layernorm.weight",
                    &[hidden],
                    "mtp.l0.input_layernorm",
                )?;
                let post_attention_layernorm = load_norm_as_f32(
                    "mtp.layers.0.post_attention_layernorm.weight",
                    &[hidden],
                    "mtp.l0.post_attn_layernorm",
                )?;
                let q_proj = load_proj_as_half(
                    "mtp.layers.0.self_attn.q_proj.weight",
                    &[2 * q_dim, hidden],
                    "mtp.l0.q_proj",
                )?;
                let k_proj = load_proj_as_half(
                    "mtp.layers.0.self_attn.k_proj.weight",
                    &[kv_dim, hidden],
                    "mtp.l0.k_proj",
                )?;
                let v_proj = load_proj_as_half(
                    "mtp.layers.0.self_attn.v_proj.weight",
                    &[kv_dim, hidden],
                    "mtp.l0.v_proj",
                )?;
                let o_proj = load_proj_as_half(
                    "mtp.layers.0.self_attn.o_proj.weight",
                    &[hidden, q_dim],
                    "mtp.l0.o_proj",
                )?;
                let q_norm = load_norm_as_f32(
                    "mtp.layers.0.self_attn.q_norm.weight",
                    &[head_dim],
                    "mtp.l0.q_norm",
                )?;
                let k_norm = load_norm_as_f32(
                    "mtp.layers.0.self_attn.k_norm.weight",
                    &[head_dim],
                    "mtp.l0.k_norm",
                )?;
                let gate_proj = load_proj_as_half(
                    "mtp.layers.0.mlp.gate_proj.weight",
                    &[inter, hidden],
                    "mtp.l0.gate_proj",
                )?;
                let up_proj = load_proj_as_half(
                    "mtp.layers.0.mlp.up_proj.weight",
                    &[inter, hidden],
                    "mtp.l0.up_proj",
                )?;
                let down_proj = load_proj_as_half(
                    "mtp.layers.0.mlp.down_proj.weight",
                    &[hidden, inter],
                    "mtp.l0.down_proj",
                )?;

                // --- Top-level MTP weights ---
                let fc = load_proj_as_half("mtp.fc.weight", &[hidden, 2 * hidden], "mtp.fc")?;
                let pre_fc_norm_embedding = load_norm_as_f32(
                    "mtp.pre_fc_norm_embedding.weight",
                    &[hidden],
                    "mtp.pre_fc_norm_embedding",
                )?;
                let pre_fc_norm_hidden = load_norm_as_f32(
                    "mtp.pre_fc_norm_hidden.weight",
                    &[hidden],
                    "mtp.pre_fc_norm_hidden",
                )?;
                let norm = load_norm_as_f32("mtp.norm.weight", &[hidden], "mtp.norm")?;

                eprintln!("[mtp] Loaded MTP layer 0 weights from {}", q4_dir.display());
                // A rotated checkpoint carries a neutral final-norm scale (the real
                // scale is folded into lm_head), so only an unrotated one can supply
                // the post-final-norm hidden the reference implementations feed the head.
                let hidden_tap = MtpHiddenTap::for_checkpoint(is_quarot);
                let hidden_tap_note = match hidden_tap {
                    MtpHiddenTap::PostFinalNorm => "",
                    MtpHiddenTap::PreFinalNorm => {
                        " (rotated checkpoint: the final-norm scale is folded into lm_head)"
                    }
                };
                eprintln!("[mtp] hidden tap: {}{hidden_tap_note}", hidden_tap.label());
                Ok(MetalMtpWeights {
                    fc,
                    pre_fc_norm_embedding,
                    pre_fc_norm_hidden,
                    layers: vec![MetalMtpLayerWeights {
                        input_layernorm,
                        post_attention_layernorm,
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        q_norm,
                        k_norm,
                        mlp: MetalMtpDenseMlpWeights {
                            gate_proj,
                            up_proj,
                            down_proj,
                        },
                    }],
                    norm,
                    hidden_tap,
                })
            })();

            match weights {
                Ok(weights) => Ok(MtpQ4LoadResult {
                    weights: Some(weights),
                    first_missing_file: None,
                }),
                Err(MtpLoadErr::Missing(first_missing_file)) => Ok(MtpQ4LoadResult {
                    weights: None,
                    first_missing_file: Some(first_missing_file),
                }),
                Err(MtpLoadErr::Incompatible(message)) => Err(message),
            }
        }

        /// Create a new [`MetalQwen35State`] by loading pre-quantized Q4/F16 files directly
        /// from a directory, without going through the full f32 weight representation.
        ///
        /// This avoids the ~51 GB peak RAM usage of the `new()` path (which loads all weights
        /// as `Vec<f32>` and then quantizes on the Metal side) and instead reads Q4/f16 files
        /// written by `foundation/inference/src/bin/quantize_q4.rs`.
        ///
        /// # File layout
        ///
        /// `q4_dir` must contain files produced by `quantize_q4`:
        /// - Large projections (`*_proj*.weight`, `gate_proj`, `up_proj`, `down_proj`,
        ///   `embed_tokens`, `lm_head`) → `<sanitized_name>.q4`
        /// - Small tensors (norms, biases, A_log, dt_bias, conv1d) → `<sanitized_name>.f16`
        ///
        /// # Errors
        ///
        /// Returns `Err(String)` on missing files, I/O failures, or Metal initialisation
        /// failures.
        pub fn from_q4_dir(
            q4_dir: &std::path::Path,
            tokenizer_path: &std::path::Path,
            cfg: &Qwen35Config,
            max_cache_len: usize,
        ) -> Result<Self, String> {
            let device =
                Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;

            tracing::info!(
                name = device.name(),
                "Metal GPU initialized for Q4 direct-load forward pass"
            );

            let queue = device.new_command_queue();

            // Q4 dir always uses Q4_0 format.
            let quant_format = QuantFormat::Q4_0;

            validate_flash_decode_shape(
                cfg.head_dim,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.num_attention_heads * cfg.head_dim,
                cfg.num_key_value_heads * cfg.head_dim,
            )?;

            // Cache-capacity validation, matching `new_session` so a
            // `from_q4_dir` call cannot construct a runtime whose KV cap
            // outruns the RoPE table (`partial_rope_interleaved` indexes
            // `cos_tab` / `sin_tab` by `position`; without this guard a
            // `--max-cache-len > max_position_embeddings` argument would
            // race past the precomputed table at decode time).
            if max_cache_len == 0 {
                return Err("from_q4_dir: max_cache_len must be > 0".to_string());
            }
            if max_cache_len > cfg.max_position_embeddings {
                return Err(format!(
                    "from_q4_dir: max_cache_len {max_cache_len} exceeds max_position_embeddings {} \
                     (the RoPE table is sized to max_position_embeddings; running past it would \
                     produce out-of-bounds GPU reads or incorrect rotary positions)",
                    cfg.max_position_embeddings
                ));
            }

            // Two-way config/artifact contract for the logits head:
            //
            // - `tie_word_embeddings=false` (e.g., QuaRot output where
            //   step 3c flips the flag and materializes a fused
            //   `(1 + γ_final)` + rotation `lm_head.weight`) MUST ship
            //   the rotated head as its own `lm_head.weight.q4`. Falling
            //   back to `embed_tokens` here would silently feed the wrong
            //   matrix into the final logits GEMV.
            // - `tie_word_embeddings=true` (the legitimate "embed_tokens
            //   IS the head" case) MUST NOT ship `lm_head.weight.q4` —
            //   if it does, the on-disk config-flip was likely skipped
            //   (HF Qwen3.5 stores the flag in both `text_config` and
            //   top level; `Qwen35Config::from_config_json_str` takes
            //   the top-level value, so a partial update can leave the
            //   parsed flag stale). The materialized head would then be
            //   dead weight and the runtime would silently fall back to
            //   `embed_tokens` — exactly the contamination ADR-044
            //   §"Step 3c contract" warns against and `quant::quarot::lm_head`
            //   flags at the converter boundary.
            //
            // Both mismatches let `eval_perplexity --quarot-q4-dir`
            // print a PPL delta and PASS/FAIL verdict on the wrong
            // logits head, so the loader refuses both directions before
            // any tensor I/O.
            let lm_head_path = Self::q4_tensor_path(q4_dir, "lm_head.weight", "q4");
            let lm_head_present = lm_head_path.exists();
            if !cfg.tie_word_embeddings && !lm_head_present {
                return Err(format!(
                    "from_q4_dir: tie_word_embeddings=false but lm_head.q4 is missing at {}; \
                     the runtime requires the materialized lm_head matrix (ADR-044 step 3c) \
                     and must not fall back to embed_tokens, which would yield wrong logits \
                     and a misleading perplexity report",
                    lm_head_path.display()
                ));
            }
            if cfg.tie_word_embeddings && lm_head_present {
                return Err(format!(
                    "from_q4_dir: tie_word_embeddings=true but lm_head.q4 is present at {}; \
                     the on-disk config likely missed the step-3c `tie_word_embeddings=false` \
                     flip (HF Qwen3.5 carries the flag in both nested `text_config` and the \
                     top level — see `Qwen35Config::from_config_json_str`). Loading would \
                     silently ignore the materialized head and fall back to embed_tokens, \
                     yielding wrong logits and a misleading perplexity report",
                    lm_head_path.display()
                ));
            }

            let pipelines = build_pipelines(&device, cfg)?;

            let hidden = cfg.hidden_size;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let value_heads = cfg.linear_num_value_heads();
            let inter = cfg.intermediate_size;

            // ----------------------------------------------------------------
            // Phase timers — measures I/O and Metal buffer creation costs.
            // ----------------------------------------------------------------
            let t_total = std::time::Instant::now();
            let mut dur_a = std::time::Duration::ZERO; // Phase A: qkvz I/O + Metal copies
            let dur_b_cell = std::cell::Cell::new(std::time::Duration::ZERO); // Phase B: F16 loads
            let dur_c_cell = std::cell::Cell::new(std::time::Duration::ZERO); // Phase C: mmap Q4

            // ----------------------------------------------------------------
            // Helper: mmap a Q4 file and create a Metal no-copy buffer.
            // Zero CPU copies — GPU pages fault lazily from mmap'd file.
            // ----------------------------------------------------------------
            let load_q4_buf = |name: &str,
                               label: &str,
                               expected_shape: &[usize]|
             -> Result<Q4WeightBuf, String> {
                let t = std::time::Instant::now();
                let path = Self::q4_tensor_path(q4_dir, name, "q4");
                let result = mmap_q4_weight(&device, &path, label, expected_shape);
                dur_c_cell.set(dur_c_cell.get() + t.elapsed());
                result
            };
            let load_q4_raw_timed =
                |name: &str, expected_shape: &[usize]| -> Result<(Vec<u8>, usize), String> {
                    let t = std::time::Instant::now();
                    let path = Self::q4_tensor_path(q4_dir, name, "q4");
                    let result = Self::load_q4_raw_bytes(&path, expected_shape);
                    dur_c_cell.set(dur_c_cell.get() + t.elapsed());
                    result
                };
            let make_fused_q4_buf = |gate_raw: &[u8], up_raw: &[u8], label: &str| -> Q4WeightBuf {
                let combined_size = gate_raw.len() + up_raw.len();
                let buf =
                    device.new_buffer(combined_size as u64, MTLResourceOptions::StorageModeShared);
                unsafe {
                    let dst = buf.contents() as *mut u8;
                    std::ptr::copy_nonoverlapping(gate_raw.as_ptr(), dst, gate_raw.len());
                    std::ptr::copy_nonoverlapping(
                        up_raw.as_ptr(),
                        dst.add(gate_raw.len()),
                        up_raw.len(),
                    );
                }
                buf.set_label(label);
                Q4WeightBuf::from_buffer(buf)
            };

            // ----------------------------------------------------------------
            // Helper: load an F16 file, convert to f32, create a Metal f32 buffer.
            // ----------------------------------------------------------------
            let load_f16_buf_f32 = |name: &str,
                                    label: &str,
                                    expected_shape: &[usize]|
             -> Result<Buffer, String> {
                let t = std::time::Instant::now();
                let path = Self::q4_tensor_path(q4_dir, name, "f16");
                let (values, _shape) =
                    crate::weights::q4_weights::load_f16_tensor_file_checked(&path, expected_shape)
                        .map_err(|e| format!("failed to load f16 file {}: {e}", path.display()))?;
                let buf = make_buffer(&device, &values, label);
                dur_b_cell.set(dur_b_cell.get() + t.elapsed());
                Ok(buf)
            };

            // ----------------------------------------------------------------
            // H3: Merge-on-first-load for in_proj_qkvz.
            // On first load, concatenate qkv+z payloads into a single merged .q4 file
            // so that subsequent loads can use zero-copy mmap (like all other Q4 weights).
            // Merged filename encodes source sizes → stale cache auto-invalidated on model update.
            // Falls back to CPU concat if model dir is read-only.
            // Metal buffer creation stays on the main thread (Metal is not thread-safe).
            // ----------------------------------------------------------------
            let t_a_io = std::time::Instant::now();
            let mut qkvz_merge_map: std::collections::HashMap<usize, Option<std::path::PathBuf>> = {
                use rayon::prelude::*;
                let pairs: Vec<Result<(usize, Option<std::path::PathBuf>), String>> =
                    (0..cfg.num_hidden_layers)
                        .into_par_iter()
                        .filter(|&i| cfg.is_layer_active(i) && !cfg.is_full_attention(i))
                        .map(|i| {
                            let prefix = format!("model.language_model.layers.{i}");
                            let qkv_p = Self::q4_tensor_path(
                                q4_dir,
                                &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                                "q4",
                            );
                            let z_p = Self::q4_tensor_path(
                                q4_dir,
                                &format!("{prefix}.linear_attn.in_proj_z.weight"),
                                "q4",
                            );
                            let qkv_size = std::fs::metadata(&qkv_p)
                                .map_err(|e| format!("metadata {}: {e}", qkv_p.display()))?.len();
                            let z_size = std::fs::metadata(&z_p)
                                .map_err(|e| format!("metadata {}: {e}", z_p.display()))?.len();
                            // Filename encodes source sizes for cache-invalidation.
                            let merged_path = q4_dir.join(
                                format!("merged_qkvz_{i}_{qkv_size}_{z_size}.q4")
                            );
                            // Expected size: 36-byte header (ndim=2, payload_offset=36)
                            // + qkv payload + z payload.  All source weight files are 2D
                            // so payload_offset=36 for each.
                            let expected_size =
                                crate::weights::q4_weights::merged_qkvz_expected_size(qkv_size, z_size)
                                    .map_err(|e| format!("layer {i} merge size: {e}"))?;
                            // Content-integrity check (#504 remaining slice 1): size alone
                            // cannot catch a same-size stale/corrupted merged artifact, so
                            // this also verifies the merged file's payload hash against a
                            // hash freshly derived from the *current* qkv/z source files —
                            // fail closed to a rebuild on any mismatch or read error.
                            let is_valid = crate::weights::q4_weights::merged_qkvz_cache_is_valid(
                                &merged_path,
                                expected_size,
                                &qkv_p,
                                &z_p,
                            );
                            if !is_valid {
                                let _ = std::fs::remove_file(&merged_path);
                                if let Err(e) = crate::weights::q4_weights::write_merged_qkvz(&qkv_p, &z_p, &merged_path) {
                                    eprintln!("[load] merge-on-first-load failed layer {i}: {e}; using CPU fallback");
                                    return Ok((i, None));
                                }
                            }
                            Ok((i, Some(merged_path)))
                        })
                        .collect();
                pairs.into_iter().collect::<Result<_, _>>()?
            };
            dur_a += t_a_io.elapsed();

            // ----------------------------------------------------------------
            // Per-layer weights
            // ----------------------------------------------------------------
            let mut layer_weights: Vec<(MetalLayerAttnWeights, MetalCommonLayerWeights)> =
                Vec::with_capacity(cfg.num_active_layers());

            for i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(i) {
                    tracing::debug!(layer = i, "skipping pruned layer weight loading");
                    continue;
                }
                let prefix = format!("model.language_model.layers.{i}");

                // Common layer weights
                let common = MetalCommonLayerWeights {
                    input_layernorm: load_f16_buf_f32(
                        &format!("{prefix}.input_layernorm.weight"),
                        &format!("L{i}.in_norm"),
                        &[hidden],
                    )?,
                    post_attention_layernorm: load_f16_buf_f32(
                        &format!("{prefix}.post_attention_layernorm.weight"),
                        &format!("L{i}.post_norm"),
                        &[hidden],
                    )?,
                    ffn: if cfg.is_moe() {
                        // MoE checkpoint: routed + shared expert tensors live under
                        // `{prefix}.mlp.{experts,shared_expert,shared_expert_gate,gate}.*`
                        // (see `should_quantize` in quantize_q4.rs — the `.experts.gate_up_proj`
                        // / `.experts.down_proj` fused per-layer arrays, plus every other MoE
                        // tensor here, are Q4-quantized). `MetalFfnWeights::Dense`'s
                        // `mlp.{gate,up,down}_proj.weight` files do not exist for this layer.
                        //
                        // `layer_weights.len()` is this layer's live ordinal among MoE
                        // layers already sized on this device: `is_moe()` is a whole-
                        // checkpoint flag (every active layer takes this branch when
                        // true, per `Qwen35Config::is_moe`), and every prior iteration
                        // of this loop already pushed its `(attn, common)` pair before
                        // this one runs — so `layer_weights.len()` is exactly the count
                        // of MoE layers whose `ExpertSlotCache` buffers are already
                        // resident, never a guess.
                        Self::load_moe_ffn_q4(
                            &device,
                            q4_dir,
                            cfg,
                            &prefix,
                            i,
                            layer_weights.len(),
                        )?
                    } else {
                        let (gate_raw, _) = load_q4_raw_timed(
                            &format!("{prefix}.mlp.gate_proj.weight"),
                            &[inter, hidden],
                        )?;
                        let (up_raw, _) = load_q4_raw_timed(
                            &format!("{prefix}.mlp.up_proj.weight"),
                            &[inter, hidden],
                        )?;
                        MetalFfnWeights::Dense {
                            gate_up_proj: make_fused_q4_buf(
                                &gate_raw,
                                &up_raw,
                                &format!("L{i}.gate_up.q4"),
                            ),
                            down_proj: load_q4_buf(
                                &format!("{prefix}.mlp.down_proj.weight"),
                                &format!("L{i}.down.q4"),
                                &[hidden, inter],
                            )?,
                        }
                    },
                };

                let attn = if cfg.is_full_attention(i) {
                    MetalLayerAttnWeights::Full(MetalFullLayerWeights {
                        q_proj: load_q4_buf(
                            &format!("{prefix}.self_attn.q_proj.weight"),
                            &format!("L{i}.full.q.q4"),
                            &[2 * q_dim, hidden],
                        )?,
                        k_proj: load_q4_buf(
                            &format!("{prefix}.self_attn.k_proj.weight"),
                            &format!("L{i}.full.k.q4"),
                            &[kv_dim, hidden],
                        )?,
                        v_proj: load_q4_buf(
                            &format!("{prefix}.self_attn.v_proj.weight"),
                            &format!("L{i}.full.v.q4"),
                            &[kv_dim, hidden],
                        )?,
                        o_proj: load_q4_buf(
                            &format!("{prefix}.self_attn.o_proj.weight"),
                            &format!("L{i}.full.o.q4"),
                            &[hidden, q_dim],
                        )?,
                        // q_norm and k_norm are norm.weight → stored as .f16
                        q_norm: load_f16_buf_f32(
                            &format!("{prefix}.self_attn.q_norm.weight"),
                            &format!("L{i}.full.q_norm"),
                            &[cfg.head_dim],
                        )?,
                        k_norm: load_f16_buf_f32(
                            &format!("{prefix}.self_attn.k_norm.weight"),
                            &format!("L{i}.full.k_norm"),
                            &[cfg.head_dim],
                        )?,
                    })
                } else {
                    // in_proj_b and in_proj_a are quantized on disk (ends_with _proj_b.weight /
                    // _proj_a.weight → Q4), but in new() they become f16 Metal buffers for the
                    // CPU GDN recurrence path.  Load Q4 → dequantize → f16 Metal buffer.
                    let load_q4_as_f16_buf = |name: &str,
                                              label: &str,
                                              expected_shape: &[usize]|
                     -> Result<Buffer, String> {
                        let t = std::time::Instant::now();
                        let path = Self::q4_tensor_path(q4_dir, name, "q4");
                        let tensor =
                            crate::weights::q4_weights::load_q4_file_checked(&path, expected_shape)
                                .map_err(|e| {
                                    format!("failed to load Q4 file {}: {e}", path.display())
                                })?;
                        let buf = Self::make_buffer_f16_from_q4(&device, &tensor, label);
                        dur_b_cell.set(dur_b_cell.get() + t.elapsed());
                        Ok(buf)
                    };

                    MetalLayerAttnWeights::Linear(MetalGdnLayerWeights {
                        in_proj_qkv: load_q4_buf(
                            &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                            &format!("L{i}.gdn.qkv.q4"),
                            &[qkv_dim, hidden],
                        )?,
                        in_proj_z: load_q4_buf(
                            &format!("{prefix}.linear_attn.in_proj_z.weight"),
                            &format!("L{i}.gdn.z.q4"),
                            &[output_dim, hidden],
                        )?,
                        in_proj_qkvz: {
                            let t_qkvz = std::time::Instant::now();
                            let r = match qkvz_merge_map.remove(&i) {
                                Some(Some(merged_path)) => mmap_q4_weight(
                                    &device,
                                    &merged_path,
                                    &format!("L{i}.gdn.qkvz.q4"),
                                    &[qkv_dim + output_dim, hidden],
                                )?,
                                _ => {
                                    // Fallback: model dir is read-only — CPU concat path
                                    let pfx = format!("model.language_model.layers.{i}");
                                    let qkv_p = Self::q4_tensor_path(
                                        q4_dir,
                                        &format!("{pfx}.linear_attn.in_proj_qkv.weight"),
                                        "q4",
                                    );
                                    let z_p = Self::q4_tensor_path(
                                        q4_dir,
                                        &format!("{pfx}.linear_attn.in_proj_z.weight"),
                                        "q4",
                                    );
                                    let (mut raw, _) =
                                        Self::load_q4_raw_bytes(&qkv_p, &[qkv_dim, hidden])?;
                                    let (z_raw, _) =
                                        Self::load_q4_raw_bytes(&z_p, &[output_dim, hidden])?;
                                    raw.extend_from_slice(&z_raw);
                                    Q4WeightBuf::from_buffer(Self::make_buffer_from_q4_raw(
                                        &device,
                                        &raw,
                                        &format!("L{i}.gdn.qkvz.q4"),
                                    ))
                                }
                            };
                            dur_a += t_qkvz.elapsed();
                            r
                        },
                        // in_proj_b and in_proj_a: Q4 on disk, but stored as f16 Metal buffers
                        in_proj_b: load_q4_as_f16_buf(
                            &format!("{prefix}.linear_attn.in_proj_b.weight"),
                            &format!("L{i}.gdn.b.f16"),
                            &[value_heads, hidden],
                        )?,
                        in_proj_a: load_q4_as_f16_buf(
                            &format!("{prefix}.linear_attn.in_proj_a.weight"),
                            &format!("L{i}.gdn.a.f16"),
                            &[value_heads, hidden],
                        )?,
                        // Small scalars: f32 Metal buffers (CPU-read in GDN recurrence)
                        a_log: load_f16_buf_f32(
                            &format!("{prefix}.linear_attn.A_log"),
                            &format!("L{i}.gdn.a_log"),
                            &[value_heads],
                        )?,
                        dt_bias: load_f16_buf_f32(
                            &format!("{prefix}.linear_attn.dt_bias"),
                            &format!("L{i}.gdn.dt_bias"),
                            &[value_heads],
                        )?,
                        conv1d_weight: load_f16_buf_f32(
                            &format!("{prefix}.linear_attn.conv1d.weight"),
                            &format!("L{i}.gdn.conv1d"),
                            &[qkv_dim, 1, cfg.linear_conv_kernel_dim],
                        )?,
                        norm_weight: load_f16_buf_f32(
                            &format!("{prefix}.linear_attn.norm.weight"),
                            &format!("L{i}.gdn.norm"),
                            // Per-HEAD RMSNorm: one gain per value-head element,
                            // not one per output element. `linear_output_dim()` is
                            // `linear_num_value_heads() * linear_value_head_dim`, so
                            // expecting it here demands the full GDN output width and
                            // rejects the checkpoint's own [linear_value_head_dim]
                            // tensor. The safetensors loader has always expected
                            // `cfg.linear_value_head_dim` for this same tensor.
                            &[cfg.linear_value_head_dim],
                        )?,
                        out_proj: load_q4_buf(
                            &format!("{prefix}.linear_attn.out_proj.weight"),
                            &format!("L{i}.gdn.out.q4"),
                            &[hidden, output_dim],
                        )?,
                    })
                };

                layer_weights.push((attn, common));
            }

            let t_d = std::time::Instant::now();

            // ----------------------------------------------------------------
            // embed_tokens: Q4 on disk.
            //   - embed_tokens (f16 buf): for CPU embedding lookup
            //   - embed_tokens_q8 (Q4 buf): for logits GEMV on GPU
            // ----------------------------------------------------------------
            let embed_name = "model.language_model.embed_tokens.weight";
            let embed_path = Self::q4_tensor_path(q4_dir, embed_name, "q4");
            let embed_tensor = crate::weights::q4_weights::load_q4_file_checked(
                &embed_path,
                &[cfg.vocab_size, hidden],
            )
            .map_err(|e| format!("failed to load embed_tokens Q4 file: {e}"))?;
            // f16 buffer for CPU embedding lookup
            let embed_tokens =
                Self::make_buffer_f16_from_q4(&device, &embed_tensor, "embed_tokens.f16");
            // Q4 buffer for GPU logits GEMV — mmap no-copy
            let embed_q4_path = Self::q4_tensor_path(q4_dir, embed_name, "q4");
            let embed_tokens_q8 = mmap_q4_weight(
                &device,
                &embed_q4_path,
                "embed_tokens.q4",
                &[cfg.vocab_size, hidden],
            )?;

            // ----------------------------------------------------------------
            // final_norm: stored as .f16 (it's a norm.weight tensor)
            // ----------------------------------------------------------------
            let final_norm =
                load_f16_buf_f32("model.language_model.norm.weight", "final_norm", &[hidden])?;

            // ----------------------------------------------------------------
            // lm_head: only present when tie_word_embeddings = false.
            //
            // Step-4b loader contract: when `!cfg.tie_word_embeddings` we
            // require `lm_head.weight.q4` on disk — the early-return guard
            // at the top of `from_q4_dir` already rejected the missing-file
            // case, so this branch only sees a path that exists. We load
            // it into the `embed_tokens_q8` slot because that buffer is
            // what the runtime's logits GEMV indexes; the field is shared
            // between tied (embed_tokens IS the head) and untied
            // (separate fused-and-rotated head) artifacts.
            // ----------------------------------------------------------------
            let embed_tokens_q8 = if !cfg.tie_word_embeddings {
                let lm_head_path = Self::q4_tensor_path(q4_dir, "lm_head.weight", "q4");
                mmap_q4_weight(
                    &device,
                    &lm_head_path,
                    "lm_head.q4",
                    &[cfg.vocab_size, hidden],
                )?
            } else {
                embed_tokens_q8
            };

            // ----------------------------------------------------------------
            // RoPE tables (same as new())
            // ----------------------------------------------------------------
            let rope_dim = cfg.rope_dim();
            let rope_max = cfg.max_position_embeddings.min(max_cache_len + 64);
            let (cos_data, sin_data) = build_rope_interleaved(rope_dim, rope_max, cfg.rope_theta);
            let rope_cos = make_buffer(&device, &cos_data, "rope_cos");
            let rope_sin = make_buffer(&device, &sin_data, "rope_sin");

            // ----------------------------------------------------------------
            // Activation buffers (same as new())
            // ----------------------------------------------------------------
            let max_prefill = max_cache_len.min(512);
            let bp = max_prefill;
            let activations = MetalQwen35Activations {
                hidden: make_zero_buffer(&device, bp * hidden, "act_hidden"),
                residual: make_zero_buffer(&device, bp * hidden, "act_residual"),
                attn_out: make_zero_buffer(&device, bp * q_dim.max(hidden), "act_attn_out"),
                q: make_zero_buffer(&device, bp * 2 * q_dim, "act_q_interleaved"),
                q_separated: make_zero_buffer(&device, bp * q_dim, "act_q"),
                gate_z: make_zero_buffer(&device, bp * q_dim, "act_gate_z"),
                k: make_zero_buffer(&device, bp * kv_dim, "act_k"),
                v: make_zero_buffer(&device, bp * kv_dim, "act_v"),
                gate: make_zero_buffer(&device, bp * 2 * inter, "act_gate"),
                up: make_zero_buffer(&device, bp * inter, "act_up"),
                ffn_out: make_zero_buffer(&device, bp * hidden, "act_ffn_out"),
                gdn_qkv: make_zero_buffer(&device, bp * qkv_dim, "act_gdn_qkv"),
                gdn_z: make_zero_buffer(&device, bp * output_dim, "act_gdn_z"),
                gdn_qkvz: make_zero_buffer(&device, qkv_dim + output_dim, "act_gdn_qkvz"),
                gdn_key_scratch: make_zero_buffer(
                    &device,
                    cfg.linear_num_key_heads * (2 * cfg.linear_key_head_dim + 1)
                        + cfg.linear_num_value_heads() * 2,
                    "act_gdn_key_scratch",
                ),
                gdn_raw_out: make_zero_buffer(&device, output_dim, "act_gdn_raw_out"),
                gdn_chunk: {
                    let num_chunks = bp.div_ceil(GDN_CHUNK_SIZE);
                    let num_vh = cfg.linear_num_value_heads();
                    let kd = cfg.linear_key_head_dim;
                    let vd = cfg.linear_value_head_dim;
                    let chunk_rows = num_chunks * num_vh * GDN_CHUNK_SIZE;
                    let c2 = num_chunks * num_vh * GDN_CHUNK_SIZE * GDN_CHUNK_SIZE;
                    GdnChunkScratch {
                        q: make_zero_buffer(&device, chunk_rows * kd, "gdn_chunk_q"),
                        k: make_zero_buffer(&device, chunk_rows * kd, "gdn_chunk_k"),
                        v: make_zero_buffer(&device, chunk_rows * vd, "gdn_chunk_v"),
                        beta_log_alpha: make_zero_buffer(&device, chunk_rows * 2, "gdn_chunk_bla"),
                        gamma: make_zero_buffer(&device, chunk_rows, "gdn_chunk_gamma"),
                        gamma_end: make_zero_buffer(
                            &device,
                            num_chunks * num_vh,
                            "gdn_chunk_gamma_end",
                        ),
                        kkt: make_zero_buffer(&device, c2, "gdn_chunk_kkt"),
                        qk_l: make_zero_buffer(&device, c2, "gdn_chunk_qkl"),
                        w: make_zero_buffer(&device, chunk_rows * kd, "gdn_chunk_w"),
                        u: make_zero_buffer(&device, chunk_rows * vd, "gdn_chunk_u"),
                        r: make_zero_buffer(&device, chunk_rows * vd, "gdn_chunk_r"),
                        k_right: make_zero_buffer(&device, chunk_rows * kd, "gdn_chunk_k_right"),
                        raw_out: make_zero_buffer(&device, bp * output_dim, "gdn_chunk_raw_out"),
                    }
                },
                logits: make_zero_buffer(&device, cfg.vocab_size, "act_logits"),
                topk_scratch_a: {
                    let groups = cfg.vocab_size.div_ceil(1024);
                    make_zero_byte_buffer(&device, groups * 256 * 8, "topk_scratch_a")
                },
                topk_scratch_b: {
                    let groups = cfg.vocab_size.div_ceil(1024);
                    make_zero_byte_buffer(&device, groups * 256 * 8, "topk_scratch_b")
                },
                attn_partials: {
                    let max_partitions = max_cache_len.div_ceil(1024);
                    let stride = cfg.head_dim + 2;
                    make_zero_buffer(
                        &device,
                        max_partitions * cfg.num_attention_heads * stride,
                        "attn_partials",
                    )
                },
                pre_final_hidden: make_zero_buffer(&device, hidden, "act_pre_final_hidden"),
                final_hidden: make_zero_buffer(&device, hidden, "act_final_hidden"),
                verify_logits: make_zero_buffer(
                    &device,
                    MTP_VERIFY_MAX_TOKENS * cfg.vocab_size,
                    "act_verify_logits",
                ),
            };

            // ----------------------------------------------------------------
            // GDN recurrent states (same as new())
            // ----------------------------------------------------------------
            let num_linear = active_gdn_layer_indices(cfg).len();
            let num_full = cfg.num_active_full_attention_layers();
            let gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
                .map(|_| GatedDeltaNetState::new(cfg))
                .collect();
            let gdn_scratch = GatedDeltaNetFusedScratch::default();

            let qkv_dim = cfg.linear_qkv_dim();
            let gdn_gpu_state = MetalGdnState::new(&device, cfg, num_linear);
            let gdn_gpu_conv_out = make_zero_buffer(&device, qkv_dim, "gdn_conv_out");

            let use_kv_f16 = matches!(
                std::env::var("LATTICE_KV_F16").as_deref(),
                Ok("1") | Ok("true")
            );
            let path_proof_enabled = matches!(
                std::env::var("LATTICE_METAL_PATH_PROOF").as_deref(),
                Ok("1") | Ok("true")
            );
            let kv_cache = MetalKvCache::new(&device, num_full, kv_dim, max_cache_len, use_kv_f16);

            let dur_d = t_d.elapsed();
            eprintln!(
                "[load-timer] Phase A (qkvz parallel I/O + Metal copy): {:.3}s",
                dur_a.as_secs_f64()
            );
            eprintln!(
                "[load-timer] Phase B (F16 buffer loads): {:.3}s",
                dur_b_cell.get().as_secs_f64()
            );
            eprintln!(
                "[load-timer] Phase C (mmap Q4 weight loads): {:.3}s",
                dur_c_cell.get().as_secs_f64()
            );
            eprintln!(
                "[load-timer] Phase D (embed/rope/activations/GDN): {:.3}s",
                dur_d.as_secs_f64()
            );
            eprintln!(
                "[load-timer] Total: {:.3}s",
                t_total.elapsed().as_secs_f64()
            );

            // tokenizer_path is accepted but not stored — tokenizer is loaded by the caller.
            let _ = tokenizer_path;

            // Determine the QuaRot flavor BEFORE loading any MTP projection, so a
            // stale MTP `.q4` sibling in a rotated-space checkpoint fails closed
            // instead of silently loading rotated weights under counter-rotation
            // (#636). ADR-051 contract: `quarot_seed` lives in
            // `quantize_index.json`. Fall back to the legacy `config.json` field
            // (`quarot_rotation_seed`) for backwards compatibility with artifacts
            // produced before the contract was formalized. #504 remaining slice 2:
            // a *present but malformed* index is a hard error (fail-closed) — only
            // a genuinely absent index falls back. A well-formed `V1Online` index
            // is *also* rejected here (this is the only load path into
            // `MetalQwen35State`, Metal being the only forward implementation) — see
            // `read_quarot_seed_from_index`'s doc comment.
            let index_seed =
                crate::quant::quarot::convert::read_quarot_seed_from_index(q4_dir, cfg)
                    .map_err(|e| format!("from_q4_dir: {e}"))?;
            let quarot_seed_opt = index_seed.or(cfg.quarot_rotation_seed);
            let is_quarot = quarot_seed_opt.is_some();

            // Load MTP weights (cache+activations go into session, not engine).
            let mtp_requested = crate::env_switch_enabled("LATTICE_MTP");
            let MtpQ4LoadResult {
                weights: mtp_weights_opt,
                first_missing_file: mtp_missing_file,
            } = Self::load_mtp_q4_weights(q4_dir, cfg, &device, is_quarot)
                .map_err(|e| format!("from_q4_dir: {e}"))?;
            if let Some(message) = super::mtp_missing_weights_warning(
                mtp_requested,
                mtp_weights_opt.is_some(),
                mtp_missing_file.as_deref(),
            ) {
                tracing::warn!("{}", message);
            }
            let mtp_session = mtp_weights_opt.as_ref().map(|_| {
                let cache = MetalMtpCache {
                    k_buf: make_zero_buffer(&device, max_cache_len * kv_dim, "mtp.kv_k"),
                    v_buf: make_zero_buffer(&device, max_cache_len * kv_dim, "mtp.kv_v"),
                    seq_len: 0,
                    max_cache_len,
                };
                let mtp_activations = MetalMtpActivations {
                    fused: make_zero_buffer(&device, 2 * hidden, "mtp.act_fused"),
                    hidden: make_zero_buffer(&device, hidden, "mtp.act_hidden"),
                    residual: make_zero_buffer(&device, hidden, "mtp.act_residual"),
                    q: make_zero_buffer(&device, 2 * q_dim, "mtp.act_q"),
                    q_separated: make_zero_buffer(&device, q_dim, "mtp.act_q_sep"),
                    gate_z: make_zero_buffer(&device, q_dim, "mtp.act_gate_z"),
                    k: make_zero_buffer(&device, kv_dim, "mtp.act_k"),
                    v: make_zero_buffer(&device, kv_dim, "mtp.act_v"),
                    attn_out: make_zero_buffer(&device, q_dim.max(hidden), "mtp.act_attn_out"),
                    gate: make_zero_buffer(&device, inter, "mtp.act_gate"),
                    up: make_zero_buffer(&device, inter, "mtp.act_up"),
                    ffn_out: make_zero_buffer(&device, hidden, "mtp.act_ffn_out"),
                    logits: make_zero_buffer(&device, cfg.vocab_size, "mtp.act_logits"),
                };
                MetalMtpSession {
                    cache,
                    activations: mtp_activations,
                }
            });
            let need_checkpoints =
                mtp_weights_opt.is_some() || crate::env_switch_enabled("LATTICE_SELF_SPEC");
            // Self-spec verifies `[pending_token] ++ draft_tokens` = `1 + SELF_SPEC_MAX_DRAFT`
            // tokens through `verify_tokens_batched`; pool size must cover one slot per token
            // plus the pre-verify base slot.
            let checkpoint_max_tokens = if mtp_weights_opt.is_some() {
                MTP_VERIFY_MAX_TOKENS.max(1 + SELF_SPEC_MAX_DRAFT)
            } else {
                1 + SELF_SPEC_MAX_DRAFT
            };
            let gdn_checkpoints = if need_checkpoints {
                Some(MetalGdnCheckpointPool::new(
                    &device,
                    checkpoint_max_tokens,
                    &gdn_gpu_state,
                ))
            } else {
                None
            };

            let quarot_rotation = if mtp_weights_opt.is_some() {
                match quarot_seed_opt {
                    Some(seed) => Some(
                        crate::quant::quarot::hadamard::RandomizedHadamard::new(seed, hidden)
                            .map_err(|e| {
                                format!(
                                    "from_q4_dir: failed to build QuaRot rotation \
                                         (seed={seed}, hidden={hidden}): {e}"
                                )
                            })?,
                    ),
                    None => None,
                }
            } else {
                None
            };

            let has_moe_layer = derive_has_moe_layer(&layer_weights);
            Ok(Self {
                engine: MetalQwen35Engine {
                    device,
                    queue,
                    pipelines,
                    layer_weights,
                    embed_tokens,
                    embed_tokens_q8,
                    final_norm,
                    rope_cos,
                    rope_sin,
                    config: cfg.clone(),
                    quant_format,
                    mtp_weights: mtp_weights_opt,
                    quarot_rotation,
                    base_quarot_seed: quarot_seed_opt,
                    has_moe_layer,
                },
                session: InferenceSession {
                    activations,
                    gdn_states,
                    gdn_scratch,
                    gdn_gpu_state,
                    gdn_gpu_conv_out,
                    kv_cache,
                    max_prefill,
                    compact_topk: 0,
                    compact_route: GpuTopkRoute::CpuFallback,
                    compact_result: Vec::new(),
                    mtp: mtp_session,
                    mtp_active: false,
                    speculative_route: None,
                    gdn_checkpoints,
                    last_pre_final_hidden: vec![0.0f32; hidden],
                    last_hidden_cursor: None,
                    mtp_prefill_hidden: Vec::new(),
                    capture_final_hidden: false,
                    final_hidden_captured: std::sync::atomic::AtomicBool::new(false),
                    #[cfg(feature = "gdn-state-counters")]
                    gdn_state_traffic: if crate::env_switch_enabled("LATTICE_GDN_STATE_COUNTERS") {
                        Some(GdnStateTrafficCounters::new(
                            GdnStateTrafficShape::try_from_config(cfg)
                                .map_err(|e| e.to_string())?
                                .with_allocated_layers(num_linear),
                        ))
                    } else {
                        None
                    },
                },
                lora: None,
                use_gdn_chunked: !matches!(
                    std::env::var("LATTICE_GDN_CHUNKED").as_deref(),
                    Ok("0") | Ok("false")
                ),
                use_kv_f16,
                cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
                path_proof_enabled,
                path_proof: PathProofCounters::default(),
            })
        }

        /// **Unstable**: streaming chat completion; callback signature may change.
        ///
        /// Streaming chat completion with token-by-token callback.
        ///
        /// Cancellable only through `on_token`'s return value; see
        /// [`Self::chat_completion_streaming_with_cancel`] for a variant that also
        /// observes cancellation during prefill and on empty-delta decode steps.
        pub fn chat_completion_streaming<F>(
            &mut self,
            messages: &[ChatMessage],
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
        ) -> Result<ChatCompletionOutput, crate::error::InferenceError>
        where
            F: FnMut(&str, u32) -> bool,
        {
            self.chat_completion_streaming_with_cancel(
                messages,
                tokenizer,
                gen_cfg,
                on_token,
                || false,
            )
        }

        /// Cancellation-aware sibling of [`Self::chat_completion_streaming`]; see
        /// [`Self::generate_streaming_with_cancel`] for exactly what `should_cancel`
        /// observes that `on_token` alone cannot.
        ///
        /// # Errors
        ///
        /// Returns `InferenceError::InvalidInput` if `messages` contains an
        /// inline image, which requires the multimodal vision entry point, or
        /// if grammar-constrained decoding blocks every token — propagated
        /// from [`Self::generate_streaming_with_cancel`] (#611).
        pub fn chat_completion_streaming_with_cancel<F, C>(
            &mut self,
            messages: &[ChatMessage],
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
            should_cancel: C,
        ) -> Result<ChatCompletionOutput, crate::error::InferenceError>
        where
            F: FnMut(&str, u32) -> bool,
            C: FnMut() -> bool,
        {
            reject_inline_images_for_text_chat(messages)?;
            let prompt = format_chat_template(messages);
            let mut cfg = gen_cfg.clone();
            if let Some(im_end_id) = tokenizer.special_token_id("<|im_end|>")
                && !cfg.stop_token_ids.contains(&im_end_id)
            {
                cfg.stop_token_ids.push(im_end_id);
            }
            let result = self.generate_streaming_with_cancel(
                &prompt,
                tokenizer,
                &cfg,
                on_token,
                should_cancel,
            )?;
            let text = result.text.trim_end().to_string();
            Ok(ChatCompletionOutput {
                message: ChatMessage::assistant(text),
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.generated_tokens,
                stopped: result.stopped,
            })
        }
    }

    // -----------------------------------------------------------------------
    // Perplexity harness (ADR-044 step 4b)
    // -----------------------------------------------------------------------

    impl MetalQwen35State {
        /// **Unstable**: largest window length the Metal forward path can serve
        /// in a single call. Bounded by the KV-cache capacity chosen at session
        /// construction (`new_session(max_cache_len)` or
        /// `MetalQwen35State::new(_, _, max_cache_len)`), NOT by the RoPE table
        /// — `new_session` already rejects `max_cache_len >
        /// max_position_embeddings`, so the KV cap is always the tighter bound.
        /// Callers driving the perplexity harness must keep their effective
        /// window strictly at or below this value; `forward_step` panics
        /// otherwise (KV-cache full assertion).
        pub fn max_context(&self) -> usize {
            self.session.kv_cache.max_cache_len
        }

        /// The model's hidden size, which is the length of every vector
        /// [`Self::embed_tokens`] returns.
        pub fn hidden_size(&self) -> usize {
            self.engine.config.hidden_size
        }

        /// **Stable**: Metal sibling of
        /// [`crate::model::qwen35::Qwen35Model::embed_tokens`] — embed `tokens`
        /// as one `[hidden_size]` vector by pooling the model's final hidden
        /// states.
        ///
        /// Returns the same quantity as the CPU method, from the same point in
        /// the network: the last position's hidden state after the final
        /// RMSNorm, before the language-model head. The two are held together
        /// by an equivalence test rather than by inspection.
        ///
        /// The returned vector is **not** L2-normalized, matching the CPU
        /// method and HuggingFace's `AutoModel`-plus-pooling convention.
        ///
        /// # Session state
        ///
        /// This resets the session before running. GDN recurrent state carries
        /// across calls, so embedding without a reset would fold whatever the
        /// session generated previously into the vector. Do not interleave this
        /// with an in-progress generation and expect that generation to
        /// continue.
        ///
        /// # Errors
        ///
        /// - `tokens` empty, longer than the session's context, or containing
        ///   an id at or above `vocab_size`.
        /// - More than one token against a model with MoE layers: the prefill this
        ///   runs to produce the hidden state is batched, and the batched schedulers
        ///   have no MoE schedule.
        /// - [`HiddenPooling::Mean`], which this path does not implement. Mean
        ///   pooling needs every position's *post-norm* hidden state, and the
        ///   Metal prefill normalizes only the rows it is about to project to
        ///   logits — the last row on the default path, and nothing at all on a
        ///   non-final chunk of a long prompt, which returns before the norm.
        ///   Producing a number here anyway would mean silently returning
        ///   last-token pooling under a mean-pooling request, so this refuses
        ///   instead. Use the CPU path for mean pooling.
        pub fn embed_tokens(
            &mut self,
            tokens: &[u32],
            pooling: crate::model::qwen35::HiddenPooling,
        ) -> Result<Vec<f32>, crate::error::InferenceError> {
            use crate::error::InferenceError;
            use crate::model::qwen35::HiddenPooling;

            if pooling != HiddenPooling::LastToken {
                return Err(InferenceError::Inference(format!(
                    "embed_tokens: the Metal path implements {:?} only; {pooling:?} \
                     needs post-norm hidden states at every position, which this \
                     forward path does not produce",
                    HiddenPooling::LastToken
                )));
            }
            if tokens.is_empty() {
                return Err(InferenceError::Inference(
                    "embed_tokens: need at least 1 token, got 0".to_string(),
                ));
            }
            let max_context = self.max_context();
            if tokens.len() > max_context {
                return Err(InferenceError::Inference(format!(
                    "embed_tokens: {} tokens exceeds session context {max_context}",
                    tokens.len()
                )));
            }
            let vocab = self.engine.config.vocab_size;
            if let Some((i, &id)) = tokens
                .iter()
                .enumerate()
                .find(|&(_, &id)| id as usize >= vocab)
            {
                // Checked here rather than left to `forward_prefill`, which
                // asserts and panics. A library consumer passing raw ids gets a
                // typed error, matching the CPU method.
                return Err(InferenceError::Inference(format!(
                    "embed_tokens: tokens[{i}]={id} is out of range: vocab_size is {vocab}"
                )));
            }

            let hidden = self.engine.config.hidden_size;
            self.reset_state();

            let previous = self.session.capture_final_hidden;
            self.session.capture_final_hidden = true;
            self.session
                .final_hidden_captured
                .store(false, std::sync::atomic::Ordering::Relaxed);
            // Fallible route: a multi-token prompt on an MoE model has no batched
            // prefill schedule. Restore the capture flag before propagating, so a
            // rejected call leaves the session exactly as it found it.
            let prefill = self.try_forward_prefill(tokens);
            self.session.capture_final_hidden = previous;
            prefill?;

            if !self
                .session
                .final_hidden_captured
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                // Fail closed. The capture buffer persists across calls, so
                // returning its contents when this call did not write them
                // would hand back a well-formed vector for somebody else's
                // input — indistinguishable from a correct answer at the call
                // site. A forward path that norms without capturing is a bug in
                // this file, not a condition the caller can fix, so say so.
                return Err(InferenceError::Inference(
                    "embed_tokens: the forward path did not capture a final hidden state; \
                     a final-RMSNorm site is missing its capture call"
                        .to_string(),
                ));
            }

            // SAFETY: `forward_prefill` waits on its command buffers before
            // returning, and `final_hidden` is StorageModeShared and sized for
            // `hidden` f32 values.
            let v = unsafe { read_buffer(&self.session.activations.final_hidden, hidden) };
            Ok(v)
        }

        /// **Unstable**: Metal Q4 sibling of [`crate::model::qwen35::Qwen35Model::compute_token_nlls`].
        ///
        /// Computes per-position cross-entropy NLLs for an autoregressive
        /// forward pass over `tokens` on the Metal forward path. Resets all
        /// recurrent state ([`MetalQwen35State::reset_state`]) before stepping
        /// so a single call does not depend on the caller's previous KV cache
        /// / GDN state, and subsequent calls do not contaminate each other —
        /// this is what makes the function safe to use as the per-window NLL
        /// kernel inside [`crate::model::qwen35::run_strided_perplexity`].
        ///
        /// Returns a `Vec<f32>` of length `tokens.len() - 1`. The value at
        /// index `i` is `-log p(tokens[i + 1] | tokens[0..=i])`, where the
        /// softmax runs over the full vocabulary at decode step `i`.
        ///
        /// Errors if `tokens.len() < 2`, if `tokens.len() > self.max_context()`
        /// (the KV cache would overflow during the walk), if any
        /// `token_id >= cfg.vocab_size`, or if a LoRA adapter is active.
        pub fn compute_token_nlls(
            &mut self,
            tokens: &[u32],
        ) -> Result<Vec<f32>, crate::error::InferenceError> {
            if tokens.len() < 2 {
                return Err(crate::error::InferenceError::Inference(format!(
                    "compute_token_nlls: need at least 2 tokens, got {}",
                    tokens.len()
                )));
            }
            let max_context = self.max_context();
            if tokens.len() > max_context {
                return Err(crate::error::InferenceError::Inference(format!(
                    "compute_token_nlls: tokens.len() ({}) exceeds Metal KV-cache capacity ({}); \
                     use a shorter window or rebuild the session with a larger max_cache_len",
                    tokens.len(),
                    max_context
                )));
            }
            let vocab_size = self.engine.config.vocab_size;
            if let Some((bad_idx, &bad)) = tokens
                .iter()
                .enumerate()
                .find(|&(_, &t)| (t as usize) >= vocab_size)
            {
                return Err(crate::error::InferenceError::Inference(format!(
                    "compute_token_nlls: tokens[{bad_idx}]={bad} >= vocab_size {vocab_size}"
                )));
            }

            // Single batch prefill producing logits at all n positions, then
            // pick out the n-1 next-token NLLs. Same code path as production
            // decode prefill (forward_prefill) — extended to emit per-position
            // lm_head outputs via dispatch_gemm with m=n instead of m=1.
            // Sequential forward_step from pos=0 with reset_state does NOT
            // produce usable next-token distributions and is not used here.
            self.reset_state();
            let flat = self.forward_prefill_all_logits(tokens)?;
            debug_assert_eq!(flat.len(), tokens.len() * vocab_size);
            let mut nlls = Vec::with_capacity(tokens.len() - 1);
            for i in 0..tokens.len() - 1 {
                let logits = &flat[i * vocab_size..(i + 1) * vocab_size];
                let target = tokens[i + 1] as usize;
                nlls.push(crate::model::qwen35::log_softmax_nll(logits, target));
            }
            Ok(nlls)
        }

        /// **Unstable**: Metal Q4 sibling of [`crate::model::qwen35::Qwen35Model::compute_perplexity`].
        ///
        /// Thin wrapper around [`crate::model::qwen35::run_strided_perplexity`]
        /// driving the shared aggregator with this state's
        /// [`Self::compute_token_nlls`] as the per-window NLL kernel. Each
        /// window resets recurrent state, so context never crosses a window
        /// boundary — identical semantics to the CPU path.
        ///
        /// Use this on a [`MetalQwen35State`] loaded via [`Self::from_q4_dir`]
        /// to measure unrotated-Q4 PPL or rotated-Q4 PPL (the output of
        /// `bin/quantize_quarot`); the rotated-vs-unrotated delta is the
        /// acceptance number for ADR-044 step 4.
        pub fn compute_perplexity(
            &mut self,
            tokens: &[u32],
            cfg: &crate::model::qwen35::PerplexityConfig,
        ) -> Result<crate::model::qwen35::PerplexityReport, crate::error::InferenceError> {
            let max_ctx = self.max_context();
            crate::model::qwen35::run_strided_perplexity(tokens, cfg, max_ctx, |slice| {
                self.compute_token_nlls(slice)
            })
        }
    }

    // `MtpTargetVerifier` is a public trait and this impl's
    // mutating methods (`rollback_cache_to`, `verify_tokens`,
    // `verify_tokens_with_hidden`) advance live KV/GDN state exactly like
    // `forward_step`/`forward_prefill`. They are not wired into any live Metal
    // generate loop (Metal MTP decode uses `mtp_greedy_round`, a self-contained
    // mechanism) and only tests call `mtp_verify_draft` in this repository, but
    // a consumer holding a `&mut MetalQwen35State` and `use`-ing this trait can
    // call them directly, so they are part of the public raw-forward boundary
    // and must clear the retained cross-turn entry before mutating, same as the
    // inherent methods.
    impl crate::speculative::MtpTargetVerifier for MetalQwen35State {
        fn cache_position(&self) -> usize {
            self.session.kv_cache.seq_len
        }

        fn rollback_cache_to(
            &mut self,
            seq_len: usize,
        ) -> Result<(), crate::error::InferenceError> {
            self.cross_turn_prefix_cache.clear();
            self.rollback_speculative_state_to(seq_len)
        }

        fn verify_tokens(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
            self.check_live_cursor("MtpTargetVerifier::verify_tokens", start_pos)?;
            self.cross_turn_prefix_cache.clear();
            let out = self.verify_tokens_batched(tokens, start_pos)?;
            Ok(out.logits)
        }

        // Some only when the buffered hidden state is provably the one the target's final
        // norm read for the token at `cursor - 1` of the live history: the freshness marker
        // is set at every write of the buffer, left behind by a cursor move that did not
        // write it, and cleared by a rewind below it, a reset and a cross-turn restore. A
        // rotated checkpoint keeps its hidden states in the rotated basis, so it never
        // supplies one.
        fn last_hidden(&self) -> Option<Vec<f32>> {
            if self.checkpoint_is_rotated()
                || self.session.last_hidden_cursor != Some(self.session.kv_cache.seq_len)
                || self.session.last_pre_final_hidden.len() != self.engine.config.hidden_size
            {
                return None;
            }
            let mut hidden = self.session.last_pre_final_hidden.clone();
            self.apply_final_norm_cpu(&mut hidden);
            Some(hidden)
        }

        #[allow(clippy::type_complexity)]
        fn verify_tokens_with_hidden(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>), crate::error::InferenceError> {
            // A rotated checkpoint keeps its hidden states in the rotated basis and stores a
            // neutral final-norm scale, so it cannot supply post-final-norm rows in the
            // original basis. Behave exactly as `verify_tokens`: same forward, same state.
            if self.checkpoint_is_rotated() {
                return Ok((self.verify_tokens(tokens, start_pos)?, None));
            }
            self.check_live_cursor("MtpTargetVerifier::verify_tokens_with_hidden", start_pos)?;
            self.cross_turn_prefix_cache.clear();
            let out = self.verify_tokens_batched_impl(tokens, start_pos, true)?;
            let hidden = self.engine.config.hidden_size;
            // Every row or none: a missing row is recoverable, a wrong one is not.
            if out.pre_final_hidden_rows.len() != tokens.len()
                || out.pre_final_hidden_rows.iter().any(|r| r.len() != hidden)
            {
                return Ok((out.logits, None));
            }
            let mut rows = out.pre_final_hidden_rows;
            for row in &mut rows {
                self.apply_final_norm_cpu(row);
            }
            Ok((out.logits, Some(rows)))
        }

        fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
            let num_layers = self.session.gdn_gpu_state.len();
            let mut snap = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let layer = self.session.gdn_gpu_state.layer(i);
                let conv_buf = layer.conv_buffer();
                let s_buf = layer.s_matrix();
                let conv_floats = (conv_buf.length() / 4) as usize;
                let s_floats = (s_buf.length() / 4) as usize;
                // SAFETY: GPU buffers are StorageModeShared (allocated with
                // MTLResourceOptions::StorageModeShared in `new`/`from_q4_dir`), so
                // `contents()` points to host-readable memory. `length()` is the
                // exact allocated byte length and is divisible by 4 (we always
                // allocate f32 buffers). Callers of `snapshot_gdn_states` invoke it
                // outside any in-flight command buffer (in `mtp_verify_draft` before
                // `target.verify_tokens`), so no GPU write can race with this read.
                let conv = unsafe {
                    let ptr = conv_buf.contents() as *const f32;
                    std::slice::from_raw_parts(ptr, conv_floats).to_vec()
                };
                let s = unsafe {
                    let ptr = s_buf.contents() as *const f32;
                    std::slice::from_raw_parts(ptr, s_floats).to_vec()
                };
                snap.push((s, conv));
            }
            snap
        }

        fn restore_gdn_states(&mut self, snapshot: &crate::attention::gdn::GdnSnapshot) {
            if let Err(error) = self.validate_gdn_restore_snapshot(snapshot) {
                tracing::warn!(
                    error = %error,
                    "refusing malformed GDN state restore"
                );
                return;
            }
            self.restore_gdn_states_validated(snapshot);
        }
    }

    impl MetalQwen35State {
        fn validate_gdn_restore_snapshot(
            &self,
            snapshot: &crate::attention::gdn::GdnSnapshot,
        ) -> Result<(), crate::error::InferenceError> {
            use crate::error::InferenceError;

            let num_layers = self.session.gdn_gpu_state.len();
            if snapshot.len() != num_layers {
                return Err(InferenceError::PrefixCache(format!(
                    "restore_gdn_states: snapshot has {} layers, expected {num_layers}",
                    snapshot.len()
                )));
            }
            for (i, (s_snap, conv_snap)) in snapshot.iter().enumerate() {
                let layer = self.session.gdn_gpu_state.layer(i);
                let conv_buf = layer.conv_buffer();
                let s_buf = layer.s_matrix();
                let conv_floats = (conv_buf.length() / 4) as usize;
                let s_floats = (s_buf.length() / 4) as usize;
                if conv_snap.len() != conv_floats || s_snap.len() != s_floats {
                    return Err(InferenceError::PrefixCache(format!(
                        "restore_gdn_states: layer {i} shape mismatch: conv {} != \
                         {conv_floats} or s {} != {s_floats}",
                        conv_snap.len(),
                        s_snap.len()
                    )));
                }
            }
            Ok(())
        }

        fn restore_gdn_states_validated(&mut self, snapshot: &crate::attention::gdn::GdnSnapshot) {
            for (i, (s_snap, conv_snap)) in snapshot.iter().enumerate() {
                let layer = self.session.gdn_gpu_state.layer(i);
                let conv_buf = layer.conv_buffer();
                let s_buf = layer.s_matrix();
                // SAFETY: see snapshot_gdn_states. StorageModeShared lets the CPU write
                // the buffer directly; validation above proves each snapshot exactly
                // matches its destination, and callers invoke this outside any in-flight
                // command buffer (rejection branch in `mtp_verify_draft`).
                unsafe {
                    let dst = conv_buf.contents() as *mut f32;
                    std::ptr::copy_nonoverlapping(conv_snap.as_ptr(), dst, conv_snap.len());
                }
                unsafe {
                    let dst = s_buf.contents() as *mut f32;
                    std::ptr::copy_nonoverlapping(s_snap.as_ptr(), dst, s_snap.len());
                }
                if i < self.session.gdn_states.len() {
                    self.session.gdn_states[i].restore_from(&(s_snap.clone(), conv_snap.clone()));
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Cross-turn KV prefix cache: Metal-specific runtime (#462)
    // -----------------------------------------------------------------------

    impl MetalQwen35State {
        /// Drop all retained cross-turn cache slots. Call this after any
        /// error surfaces from a `*_with_prefix_cache` entry point (or from
        /// the caller's own error handling) so a later turn cannot restore
        /// from state that may no longer match the live KV/GDN buffers.
        pub fn clear_cross_turn_prefix_cache(&mut self) {
            self.cross_turn_prefix_cache.clear();
        }

        /// Everything that invalidates cross-turn reuse if it changes
        /// between calls: model identity, tokenizer, adapter, RoPE/KV
        /// layout, and chat template. Coarse-grained on purpose — a false
        /// invalidation only costs a full re-prefill, never wrong output.
        fn cross_turn_metadata(
            &self,
            tokenizer: &BpeTokenizer,
        ) -> crate::kv_cache::CrossTurnPrefixMetadata {
            use rustc_hash::FxHasher;
            use std::hash::{Hash, Hasher};

            let cfg = &self.engine.config;

            let mut layer_hasher = FxHasher::default();
            for lt in &cfg.layer_types {
                (*lt as u8).hash(&mut layer_hasher);
            }
            let layer_pattern_hash = layer_hasher.finish();

            let mut model_hasher = FxHasher::default();
            cfg.hidden_size.hash(&mut model_hasher);
            cfg.num_hidden_layers.hash(&mut model_hasher);
            cfg.vocab_size.hash(&mut model_hasher);
            cfg.rope_theta.to_bits().hash(&mut model_hasher);
            cfg.mtp_num_hidden_layers.hash(&mut model_hasher);
            layer_pattern_hash.hash(&mut model_hasher);
            let model_fingerprint = model_hasher.finish();

            let mut tok_hasher = FxHasher::default();
            tokenizer.vocab_size().hash(&mut tok_hasher);
            tokenizer
                .special_token_id("<|im_end|>")
                .hash(&mut tok_hasher);
            // Tokenizer identity only -- resolved unconditionally regardless
            // of whether *this* request has an active reasoning budget.
            // reasoning_budget is a generation-time policy choice, not a
            // prefix-defining input: two requests with identical prompts and
            // tokenizer state must hash to the same fingerprint whether or
            // not either one sets a budget, so a turn that happens to omit
            // (or vary) reasoning_budget can still reuse the other's cached
            // prefix instead of forcing FullRefill. The close-marker
            // VALIDATION that a budget is actually enforceable still runs
            // separately, per-request, in `resolve_reasoning_close_token`.
            // Hash EVERY id rendering the marker, not just the first tier hit.
            // Two tokenizers that agree on the winning id but differ in their
            // remaining aliases are not the same tokenizer, and folding only
            // one id in would hash them identically. Sorted because the added
            // -token walk is over a HashMap, and a cache key should not depend
            // on iteration order.
            let mut close_marker_ids = lattice_tokenizer::__internal::token_ids_for_content(
                tokenizer,
                crate::model::qwen35::REASONING_CLOSE_MARKER,
            );
            close_marker_ids.sort_unstable();
            close_marker_ids.hash(&mut tok_hasher);
            let tokenizer_fingerprint = tok_hasher.finish();

            let adapter_id = match &self.lora {
                None => crate::kv_cache::AdapterId::BASE,
                Some(adapter) => {
                    let mut h = FxHasher::default();
                    adapter.max_rank.hash(&mut h);
                    adapter.scale.to_bits().hash(&mut h);
                    adapter.projections.len().hash(&mut h);
                    crate::kv_cache::AdapterId::new(h.finish())
                }
            };

            crate::kv_cache::CrossTurnPrefixMetadata {
                model_fingerprint,
                tokenizer_fingerprint,
                adapter_id,
                vocab_size: cfg.vocab_size,
                max_cache_len: self.session.kv_cache.max_cache_len,
                kv_f16: self.use_kv_f16,
                rope_theta_bits: cfg.rope_theta.to_bits(),
                partial_rotary_factor_bits: Some(cfg.partial_rotary_factor.to_bits()),
                layer_pattern_hash,
                chat_template_version: 1,
            }
        }

        /// Pure planning step: delegates to [`crate::kv_cache::plan_prefix_reuse`]
        /// against whatever is currently cached for `slot_id`, supplying the
        /// entry's retained sparse GDN checkpoint boundaries (#590) so the
        /// planner can produce `PrefixReuseMode::ReplayFromCheckpoint` for
        /// mid-history divergence instead of always falling back to
        /// `FullRefill`.
        fn plan_cross_turn_reuse(
            &self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            metadata: &crate::kv_cache::CrossTurnPrefixMetadata,
            prompt_ids: &[u32],
        ) -> crate::kv_cache::PrefixRestorePlan {
            let entry = self.cross_turn_prefix_cache.get(slot_id);
            let checkpoint_lens: Vec<usize> = entry
                .map(|e| e.checkpoints.iter().map(|c| c.len).collect())
                .unwrap_or_default();
            crate::kv_cache::plan_prefix_reuse(
                entry.map(|e| &e.generic),
                metadata,
                prompt_ids,
                &checkpoint_lens,
            )
        }

        /// Pure preflight over the suffix a [`crate::kv_cache::PrefixRestorePlan`]
        /// selects — every suffix token id within `vocab_size`, the suffix
        /// non-empty, and `plan.suffix_start + suffix.len()` within
        /// `max_cache_len`. Must run BEFORE `restore_cross_turn_prefix`/
        /// `reset_state` in `generate_streaming_with_prefix_cache_and_cancel_inner`
        /// (#835): those calls `take()` the cache slot's entry (ExactAppend /
        /// ReplayFromCheckpoint) or clear live state (FullRefill), and the
        /// public wrapper's error-recovery arm additionally evicts the slot's
        /// cache entry outright on ANY `Err` from `_inner`. Rejecting an
        /// invalid suffix here, before any of that runs, leaves a warmed slot
        /// untouched instead of destroying it for a request that never
        /// touched live KV/GDN state.
        ///
        /// Mirrors `forward_prefill_from`'s own checks byte-for-byte — same
        /// `InferenceError` variant and message shape — so this is
        /// observationally a pure timing fix, not a new error taxonomy.
        /// `forward_prefill_from`'s checks are left in place unchanged
        /// (defense in depth for its other, non-prefix-cache callers).
        fn plan_prefix_request(
            &self,
            prompt_ids: &[u32],
            plan: &crate::kv_cache::PrefixRestorePlan,
        ) -> Result<(), crate::error::InferenceError> {
            use crate::error::InferenceError;

            let vocab = self.engine.config.vocab_size;
            let suffix = &prompt_ids[plan.suffix_start..];
            for (t, &id) in suffix.iter().enumerate() {
                if (id as usize) >= vocab {
                    return Err(InferenceError::InvalidInput(format!(
                        "forward_prefill_from: token_ids[{t}]={id} out of range: vocab_size is {vocab}"
                    )));
                }
            }
            if suffix.is_empty() {
                return Err(InferenceError::PrefixCache(
                    "forward_prefill_from: cannot prefill an empty suffix without a saved \
                     logits source"
                        .into(),
                ));
            }
            if plan.suffix_start + suffix.len() > self.session.kv_cache.max_cache_len {
                return Err(InferenceError::InvalidInput(format!(
                    "forward_prefill_from: start_pos {} + len {} exceeds max_cache_len {}",
                    plan.suffix_start,
                    suffix.len(),
                    self.session.kv_cache.max_cache_len
                )));
            }
            Ok(())
        }

        /// Restore live state to the plan's reusable boundary.
        ///
        /// `ExactAppend` and `ReplayFromCheckpoint` (#590) both consume the
        /// cached entry and return it, so the caller can carry its
        /// checkpoint ring (and its own boundary snapshot) into the
        /// end-of-generation save. `FullRefill` is handled by the caller
        /// (`self.reset_state()`) per design.md step 5 and returns `None`.
        fn restore_cross_turn_prefix(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            plan: &crate::kv_cache::PrefixRestorePlan,
        ) -> Result<Option<MetalCrossTurnPrefixEntry>, crate::error::InferenceError> {
            use crate::error::InferenceError;
            use crate::kv_cache::PrefixReuseMode;

            match plan.mode {
                PrefixReuseMode::ExactAppend => {
                    // Take (not clone): from this point on the cache is
                    // EMPTY for this slot until a fresh save succeeds at
                    // the end of generation. Any early return between here
                    // and that save (error `?`, caller-interrupt) leaves
                    // the cache empty rather than retaining a stale entry
                    // — fail-closed by construction (#516). This also
                    // removes a ~19MiB GDN
                    // snapshot clone per restore.
                    let entry = self.cross_turn_prefix_cache.take(slot_id).ok_or_else(|| {
                        InferenceError::PrefixCache(
                            "restore_cross_turn_prefix: ExactAppend plan but no cached entry \
                             for this slot"
                                .into(),
                        )
                    })?;
                    self.restore_gdn_states_checked(&entry.gdn_snapshot)?;
                    self.session.set_position(plan.reusable_len);
                    Ok(Some(entry))
                }
                PrefixReuseMode::ReplayFromCheckpoint { checkpoint_len } => {
                    // Same take-not-clone fail-closed discipline as
                    // ExactAppend. The GDN state is restored from the ring
                    // checkpoint at exactly `checkpoint_len`; the live KV
                    // rows below that boundary are reused in place (they
                    // still hold the checkpoint's prefix — the planner only
                    // selects checkpoints at or below the shared token
                    // prefix), and the caller prefills the divergent suffix
                    // forward from `checkpoint_len` via
                    // `forward_prefill_from`, overwriting rows at and above
                    // the divergence point.
                    let entry = self.cross_turn_prefix_cache.take(slot_id).ok_or_else(|| {
                        InferenceError::PrefixCache(
                            "restore_cross_turn_prefix: ReplayFromCheckpoint plan but no cached \
                             entry for this slot"
                                .into(),
                        )
                    })?;
                    let checkpoint = entry
                        .checkpoints
                        .iter()
                        .find(|c| c.len == checkpoint_len)
                        .ok_or_else(|| {
                            InferenceError::PrefixCache(format!(
                                "restore_cross_turn_prefix: plan selected checkpoint_len \
                                 {checkpoint_len} but no retained snapshot exists at that \
                                 boundary"
                            ))
                        })?;
                    self.restore_gdn_states_checked(&checkpoint.snapshot)?;
                    self.session.set_position(checkpoint_len);
                    Ok(Some(entry))
                }
                PrefixReuseMode::FullRefill => {
                    self.reset_state();
                    Ok(None)
                }
            }
        }

        /// Prefill `token_ids` starting at absolute position `start_pos`,
        /// writing KV/RoPE rows and advancing GDN state from that position —
        /// the suffix-prefill primitive the design requires. Unlike
        /// `forward_prefill`/`forward_prefill_impl` (which always start at
        /// position 0, including their `len == 1` fast path), every branch
        /// here honors `start_pos`, so a single reused-prefix token forwards
        /// at its true absolute position.
        pub(crate) fn forward_prefill_from(
            &mut self,
            token_ids: &[u32],
            start_pos: usize,
            all_positions: bool,
        ) -> Result<Vec<f32>, crate::error::InferenceError> {
            use crate::error::InferenceError;

            let vocab = self.engine.config.vocab_size;
            for (t, &id) in token_ids.iter().enumerate() {
                if (id as usize) >= vocab {
                    return Err(InferenceError::InvalidInput(format!(
                        "forward_prefill_from: token_ids[{t}]={id} out of range: vocab_size is {vocab}"
                    )));
                }
            }
            if token_ids.is_empty() {
                return Err(InferenceError::PrefixCache(
                    "forward_prefill_from: cannot prefill an empty suffix without a saved \
                     logits source"
                        .into(),
                ));
            }
            if token_ids.len() == 1 {
                return self.try_forward_step(token_ids[0], start_pos);
            }
            self.check_forward_range_capacity(start_pos, token_ids.len(), false)?;
            if self.lora.is_some() {
                if all_positions {
                    return Err(InferenceError::PrefixCache(
                        "forward_prefill_from: all-position output is not supported with an \
                         active LoRA adapter"
                            .into(),
                    ));
                }
                if self.engine.has_moe_layer {
                    // The batched schedulers have no MoE schedule; per-token steps do.
                    let mut last_logits = Vec::new();
                    for (i, &id) in token_ids.iter().enumerate() {
                        last_logits = self.forward_step(id, start_pos + i);
                    }
                    return Ok(last_logits);
                }
            }
            // Every branch below is batched (the single-token and LoRA-on-MoE fast paths
            // returned above via per-token forward steps, which support MoE), so the
            // MoE rejection applies unconditionally from here (#1448).
            self.reject_moe_batched("forward_prefill_from")?;
            let max_prefill = self.session.max_prefill;
            if token_ids.len() <= max_prefill {
                // Only/last chunk — its logits are always the caller's answer.
                return Ok(self.forward_prefill_batched_chunk(
                    token_ids,
                    start_pos,
                    all_positions,
                    true,
                    false,
                ));
            }
            if all_positions {
                let mut all_logits = Vec::with_capacity(token_ids.len() * vocab);
                let mut pos = start_pos;
                for chunk in token_ids.chunks(max_prefill) {
                    // Perplexity needs every chunk's per-position logits.
                    all_logits
                        .extend(self.forward_prefill_batched_chunk(chunk, pos, true, true, false));
                    pos += chunk.len();
                }
                Ok(all_logits)
            } else {
                let total = token_ids.len();
                let mut last_logits = Vec::new();
                let mut pos = start_pos;
                for chunk in token_ids.chunks(max_prefill) {
                    // Only the LAST chunk's tail is ever kept — see
                    // forward_prefill_batched_chunk's doc comment (Experiment B).
                    let is_last = pos + chunk.len() == start_pos + total;
                    last_logits =
                        self.forward_prefill_batched_chunk(chunk, pos, false, is_last, false);
                    pos += chunk.len();
                }
                Ok(last_logits)
            }
        }

        /// Save `represented_token_ids` as the new cross-turn entry for
        /// `slot_id`. Requires that live KV state already represents exactly
        /// those tokens (`kv_cache.seq_len == represented_token_ids.len()`) —
        /// callers must prefill/restore before saving, never after a partial
        /// or speculative step.
        fn save_cross_turn_prefix(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            metadata: crate::kv_cache::CrossTurnPrefixMetadata,
            represented_token_ids: Vec<u32>,
            consumed_entry: Option<MetalCrossTurnPrefixEntry>,
        ) -> Result<(), crate::error::InferenceError> {
            use crate::error::InferenceError;

            let represented_len = represented_token_ids.len();
            if represented_len != self.session.kv_cache.seq_len {
                return Err(InferenceError::PrefixCache(format!(
                    "save_cross_turn_prefix: represented_len {represented_len} != kv seq_len {}",
                    self.session.kv_cache.seq_len
                )));
            }
            let gdn_snapshot = self.snapshot_gdn_states_checked()?;
            let kv = crate::kv_cache::KvPrefixHandle {
                represented_len,
                num_full_attention_layers: self.session.kv_cache.k_bufs.len(),
                kv_dim: self.session.kv_cache.kv_dim,
                max_cache_len: self.session.kv_cache.max_cache_len,
                kv_f16: self.use_kv_f16,
            };

            // #590: carry surviving sparse checkpoints from the entry this
            // generation consumed (ExactAppend / ReplayFromCheckpoint), and
            // promote that entry's own boundary snapshot into the ring. A
            // checkpoint survives only while the live KV rows below its
            // boundary still hold the tokens it was taken against — the
            // pure predicate `checkpoint_survives_save` encodes that rule
            // against the common token prefix. `FullRefill` saves pass
            // `None`: `reset_state` rebuilt every KV row from scratch, so
            // no prior snapshot is byte-consistent with the live buffers
            // (chunk boundaries differ between the original and the
            // re-prefill), and the ring starts empty by design.
            let mut checkpoints: Vec<MetalGdnCheckpoint> = Vec::new();
            if let Some(old) = consumed_entry {
                let common = crate::kv_cache::longest_common_token_prefix(
                    &old.generic.token_ids,
                    &represented_token_ids,
                );
                for cp in old.checkpoints {
                    if crate::kv_cache::checkpoint_survives_save(cp.len, common, represented_len) {
                        checkpoints.push(cp);
                    }
                }
                if old.generic.gdn_snapshot_len == old.generic.represented_len
                    && crate::kv_cache::checkpoint_survives_save(
                        old.generic.represented_len,
                        common,
                        represented_len,
                    )
                {
                    checkpoints.push(MetalGdnCheckpoint {
                        len: old.generic.represented_len,
                        snapshot: old.gdn_snapshot,
                    });
                }
                checkpoints.sort_by_key(|c| c.len);
                checkpoints.dedup_by_key(|c| c.len);
                // Evict the shallowest boundaries first — see
                // CROSS_TURN_GDN_CHECKPOINT_CAP for the tradeoff.
                if checkpoints.len() > CROSS_TURN_GDN_CHECKPOINT_CAP {
                    let excess = checkpoints.len() - CROSS_TURN_GDN_CHECKPOINT_CAP;
                    checkpoints.drain(..excess);
                }
            }

            let generic = crate::kv_cache::CrossTurnPrefixEntry {
                slot_id,
                metadata,
                token_ids: represented_token_ids,
                represented_len,
                kv,
                gdn_snapshot_len: represented_len,
            };
            self.cross_turn_prefix_cache
                .insert(MetalCrossTurnPrefixEntry {
                    generic,
                    gdn_snapshot,
                    checkpoints,
                });
            Ok(())
        }

        /// Best-effort cache save: a bookkeeping failure must not fail an
        /// otherwise-successful generation. On failure the slot is cleared
        /// rather than left holding a possibly-stale entry (fail-closed).
        fn save_cross_turn_prefix_or_clear(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            metadata: crate::kv_cache::CrossTurnPrefixMetadata,
            represented_token_ids: Vec<u32>,
            consumed_entry: Option<MetalCrossTurnPrefixEntry>,
        ) {
            if self
                .save_cross_turn_prefix(slot_id, metadata, represented_token_ids, consumed_entry)
                .is_err()
            {
                self.cross_turn_prefix_cache.remove(slot_id);
            }
        }

        /// [`Self::snapshot_gdn_states`] has no failure mode today, but is
        /// wrapped in `Result` so cache callers have one fallible surface to
        /// widen behind if that ever changes.
        fn snapshot_gdn_states_checked(
            &self,
        ) -> Result<crate::attention::gdn::GdnSnapshot, crate::error::InferenceError> {
            use crate::speculative::MtpTargetVerifier;
            Ok(self.snapshot_gdn_states())
        }

        /// Fallible sibling of [`Self::restore_gdn_states`] for callers that
        /// can propagate a malformed-snapshot error instead of using the
        /// infallible trait entry point's logged, no-write rejection.
        fn restore_gdn_states_checked(
            &mut self,
            snapshot: &crate::attention::gdn::GdnSnapshot,
        ) -> Result<(), crate::error::InferenceError> {
            self.validate_gdn_restore_snapshot(snapshot)?;
            self.restore_gdn_states_validated(snapshot);
            Ok(())
        }

        /// Cache-aware sibling of [`Self::generate_streaming`].
        ///
        /// Detects the longest shared token prefix between `slot_id`'s
        /// cached entry and this prompt; when it is an exact, fully
        /// represented append-only prefix (`PrefixReuseMode::ExactAppend`),
        /// restores the GDN snapshot, sets the logical position, and
        /// prefills only the divergent suffix via `forward_prefill_from`.
        /// Otherwise falls back to `reset_state` + full prefill — byte-for-
        /// byte the same path `generate_streaming` takes, so the decode loop
        /// below is identical in both cases and produces the same generated
        /// token IDs as the non-cached path for the same conversation
        /// (token-identity invariant; see design.md).
        ///
        /// `generate_streaming` and `chat_completion_streaming` are left
        /// untouched and never use this cache, so their behavior stays
        /// deterministic and unaffected by any cache state.
        ///
        /// # Errors
        ///
        /// Returns the shared context-budget error before cache restore or eviction
        /// when the prompt plus effective decode cap exceeds [`Self::max_context`].
        /// Returns `InferenceError::UnsupportedModel` when a model with MoE
        /// layers reaches a batched prefill — a multi-token suffix (or full
        /// fallback prefill) with no active LoRA adapter. Single-token
        /// suffixes and LoRA-active prefills of an MoE model run per-token
        /// forward steps, which support MoE (see [`Self::reject_moe_batched`] and the gated
        /// [`Self::check_prefill_moe_batched_unsupported`]).
        pub fn generate_streaming_with_prefix_cache<F>(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
        ) -> Result<CachedGenerateOutput, crate::error::InferenceError>
        where
            F: FnMut(&str, u32) -> bool,
        {
            self.generate_streaming_with_prefix_cache_and_cancel(
                slot_id,
                prompt,
                tokenizer,
                gen_cfg,
                on_token,
                || false,
            )
        }

        /// Cancellation-aware sibling of [`Self::generate_streaming_with_prefix_cache`];
        /// see [`Self::generate_streaming_with_cancel`] for exactly what
        /// `should_cancel` observes that `on_token` alone cannot. Fail-closed
        /// invariant: every path that can observe a cancellation runs strictly
        /// after `restore_cross_turn_prefix` has already `take`n the slot's
        /// entry (for `ExactAppend`) or `reset_state` has cleared the whole
        /// cache (for `FullRefill`), so an early cancel-return never re-saves
        /// and the slot is left empty rather than holding a stale/partial
        /// entry — the same guarantee the existing error path documents
        /// (#516).
        pub fn generate_streaming_with_prefix_cache_and_cancel<F, C>(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
            should_cancel: C,
        ) -> Result<CachedGenerateOutput, crate::error::InferenceError>
        where
            F: FnMut(&str, u32) -> bool,
            C: FnMut() -> bool,
        {
            self.generate_streaming_with_prefix_cache_with_trace(
                slot_id,
                prompt,
                tokenizer,
                gen_cfg,
                on_token,
                should_cancel,
            )
            .map(|(output, _trace)| output)
        }

        /// The prefix-cache entry's one implementation: the output
        /// [`Self::generate_streaming_with_prefix_cache_and_cancel`] returns, with
        /// the driver trace of the request that produced it. The trace is all
        /// zeroes when the request returned before reaching the driver.
        pub(crate) fn generate_streaming_with_prefix_cache_with_trace<F, C>(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
            should_cancel: C,
        ) -> Result<
            (CachedGenerateOutput, crate::decoder::driver::DriverTrace),
            crate::error::InferenceError,
        >
        where
            F: FnMut(&str, u32) -> bool,
            C: FnMut() -> bool,
        {
            // #856/#922/#827/#1354: prompt tokenization, the empty-prompt
            // guard, and the prompt-plus-decode-budget context bound are
            // delegated to the same `prepare_generation` shared preparation
            // `generate_streaming_with_cancel` already routes through,
            // closing the last unmigrated generation entry point (issue
            // #827). This uses `GenerationEntryContract::MetalPrefixCacheStreaming`
            // (#1354), not the plain-streaming `MetalStreaming` variant this
            // call used to reuse: `MetalStreaming`'s permissive
            // `validate_capabilities` exists because plain streaming
            // supports both `logprobs` and `enable_mtp`, but this path does
            // not, so folding those checks into that variant would have
            // wrongly restricted plain streaming too. `MetalPrefixCacheStreaming`
            // carries its own `check_logprobs_not_set` /
            // `check_mtp_not_requested` guards in `validate_before_tokenization`
            // (run ahead of tokenization inside `prepare_generation`, same
            // position these two guards ran in as an explicit preflight
            // here before this change), so the error text and precedence
            // for a request that violates more than one guard at once are
            // unchanged: this call must still return here, before any
            // cache/session state mutation and before `_inner`'s
            // destructive-on-`Err` recovery arm is reachable, so a rejected
            // call never evicts a valid pre-existing cross-turn entry it
            // never touched (PR #787).
            let prompt_prep = crate::model::qwen35::prepare_generation(
                tokenizer,
                prompt,
                gen_cfg,
                self.engine.config.vocab_size,
                self.max_context(),
                crate::model::qwen35::GenerationEntryContract::MetalPrefixCacheStreaming,
            )?;
            let gen_plan = match prompt_prep {
                crate::model::qwen35::GenerationPreparation::Complete(output) => {
                    // Zero-budget requests return before any state mutation,
                    // leaving an existing cache entry exactly as-is.
                    return Ok((
                        CachedGenerateOutput {
                            cache: CrossTurnCacheStats {
                                slot_id,
                                prompt_tokens: output.prompt_tokens,
                                reused_tokens: 0,
                                prefetched_tokens: 0,
                                mode: crate::kv_cache::PrefixReuseMode::FullRefill,
                            },
                            output,
                        },
                        crate::decoder::driver::DriverTrace::default(),
                    ));
                }
                crate::model::qwen35::GenerationPreparation::Ready(plan) => plan,
            };

            // #835: validate the suffix a reuse plan would select for this
            // slot BEFORE calling `_inner` at all -- same reasoning as the
            // `check_logprobs_not_set`/`check_mtp_not_requested` guards
            // `prepare_generation` already ran above, ahead of tokenization
            // (PR #787, #1354): the `match` below unconditionally calls
            // `reset_state()` and evicts the cache slot on ANY `Err` from
            // `_inner`, so a suffix-validation-only rejection (out-of-vocab
            // token, empty suffix, or a suffix overflowing `max_cache_len`)
            // must return here, before `_inner` runs `restore_cross_turn_prefix`
            // (or `_inner`'s own destructive-on-`Err` recovery is even
            // reachable), or a valid pre-existing cross-turn entry this call
            // never touched would be destroyed alongside it.
            //
            // `prepare_generation`'s zero-budget short-circuit above already
            // returned for `max_new_tokens == 0`, so `max_new_tokens > 0` is
            // guaranteed from here on.
            let metadata = self.cross_turn_metadata(tokenizer);
            let plan = self.plan_cross_turn_reuse(slot_id, &metadata, &gen_plan.prompt_ids);
            self.plan_prefix_request(&gen_plan.prompt_ids, &plan)?;

            // Budget forcing: validate the </think> token id here too,
            // before `_inner` is even called -- same reasoning as the
            // suffix-plan preflight immediately above (#835) and the
            // `check_logprobs_not_set`/`check_mtp_not_requested` preflights
            // above that (PR #787). `_inner` resolves this again (defense
            // in depth, mirroring the suffix-plan double-check), but only
            // THIS early return -- before the destructive `match` below
            // runs `restore_cross_turn_prefix` or `reset_state()` --
            // guarantees a missing/out-of-range </think> token never
            // destroys a valid pre-existing cross-turn entry the request
            // never touched.
            //
            // Gated on `max_new_tokens > 0`, same as the suffix-plan
            // preflight above: `_inner`'s own zero-budget guard (before its
            // `resolve_reasoning_close_token` call) is
            // unreachable-before-return for a zero-token request, so an
            // unconditional resolve here would disagree with every other
            // generation path (plain CPU `generate()`, non-cached-prefix
            // Metal `generate_streaming()`) -- both short-circuit
            // `max_new_tokens == 0` to a zero-token `StopReason::Length`
            // BEFORE resolving the close marker, so a `reasoning_budget`
            // set alongside `max_new_tokens == 0` succeeds there even with
            // a missing/invalid `</think>` marker.
            if gen_cfg.max_new_tokens > 0 {
                crate::model::qwen35::resolve_reasoning_close_token(
                    tokenizer,
                    gen_cfg.reasoning_budget,
                    gen_cfg.enable_thinking,
                    self.engine.config.vocab_size,
                )?;
            }

            match self.generate_streaming_with_prefix_cache_and_cancel_inner(
                slot_id,
                gen_plan,
                tokenizer,
                gen_cfg,
                on_token,
                should_cancel,
            ) {
                Ok(out) => Ok(out),
                Err(e) => {
                    // Fail-closed: state and cache may disagree after a
                    // restore/prefill/save error, so drop both rather than
                    // risk a later turn reusing an inconsistent boundary.
                    self.reset_state();
                    self.cross_turn_prefix_cache.remove(slot_id);
                    Err(e)
                }
            }
        }

        /// Restores the plan's reusable boundary (or resets the state for a full
        /// refill), runs the suffix through the shared decoder driver over a
        /// [`crate::decoder::qwen_metal::QwenMetalSession`], then commits the
        /// cross-turn slot as the request's exit dictates (see
        /// `crate::decoder::qwen_metal::run_prefix_cache_streaming`).
        ///
        /// The preflights the public entry ran (suffix content, `</think>` id) run
        /// again here, before any state or slot is touched, so this function's own
        /// ordering invariant does not rest on its one caller.
        fn generate_streaming_with_prefix_cache_and_cancel_inner<F, C>(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            gen_plan: crate::model::qwen35::GenerationPlan,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
            should_cancel: C,
        ) -> Result<
            (CachedGenerateOutput, crate::decoder::driver::DriverTrace),
            crate::error::InferenceError,
        >
        where
            F: FnMut(&str, u32) -> bool,
            C: FnMut() -> bool,
        {
            use crate::decoder::qwen_metal::{
                PrefixCommit, QwenMetalSession, run_prefix_cache_streaming,
            };
            use crate::kv_cache::PrefixReuseMode;

            let prompt_ids = gen_plan.prompt_ids.clone();
            let prompt_len = prompt_ids.len();
            debug_assert!(
                prompt_len > 0,
                "the wrapper's check_prompt_not_empty (#856) must reject an \
                 empty prompt before calling _inner"
            );
            debug_assert!(
                gen_cfg.max_new_tokens > 0,
                "the wrapper's prepare_generation zero-budget short-circuit \
                 (#827) must return before calling _inner"
            );

            let metadata = self.cross_turn_metadata(tokenizer);
            let plan = self.plan_cross_turn_reuse(slot_id, &metadata, &prompt_ids);
            self.plan_prefix_request(&prompt_ids, &plan)?;

            let think_close_id = crate::model::qwen35::resolve_reasoning_close_token(
                tokenizer,
                gen_cfg.reasoning_budget,
                gen_cfg.enable_thinking,
                self.engine.config.vocab_size,
            )?;

            // The consumed entry (ExactAppend / ReplayFromCheckpoint) is
            // carried to the end-of-generation save so its checkpoint ring
            // and boundary snapshot survive across turns (#590).
            let mut consumed_entry = match plan.mode {
                PrefixReuseMode::ExactAppend | PrefixReuseMode::ReplayFromCheckpoint { .. } => {
                    self.restore_cross_turn_prefix(slot_id, &plan)?
                }
                PrefixReuseMode::FullRefill => {
                    self.reset_state();
                    None
                }
            };

            // The session borrows the state for the whole run and tears the
            // sampling route down when it drops, before the commit below.
            let eos_token_id = self.engine.config.eos_token_id;
            let run = {
                let mut session = QwenMetalSession::over_restored_state(
                    self,
                    gen_plan,
                    plan.suffix_start,
                    gen_cfg,
                    SamplingRouteEnvironment::current(),
                )?;
                run_prefix_cache_streaming(
                    &mut session,
                    gen_cfg,
                    think_close_id,
                    &prompt_ids,
                    eos_token_id,
                    tokenizer,
                    on_token,
                    should_cancel,
                )?
            };

            match run.commit {
                PrefixCommit::Leave => self.cross_turn_prefix_cache.remove(slot_id),
                PrefixCommit::Save { silent_step } => {
                    // The silent step forwards the last pushed token so the next
                    // turn can reuse through the full assistant output. It
                    // samples and emits nothing. A full KV cache leaves the
                    // boundary one token short, which stays consistent: KV, GDN
                    // and `represented_len` all agree there.
                    if silent_step && let Some(&last_pushed_id) = run.output.token_ids.last() {
                        let seq_len = self.session.kv_cache.seq_len;
                        if seq_len < self.session.kv_cache.max_cache_len {
                            let _ = self.forward_step_decode(last_pushed_id, seq_len);
                        }
                    }
                    let represented_len = self.session.kv_cache.seq_len;
                    debug_assert!(represented_len >= prompt_len);
                    let mut represented_token_ids = prompt_ids;
                    represented_token_ids
                        .extend_from_slice(&run.output.token_ids[..represented_len - prompt_len]);
                    self.save_cross_turn_prefix_or_clear(
                        slot_id,
                        metadata,
                        represented_token_ids,
                        consumed_entry.take(),
                    );
                }
            }

            Ok((
                CachedGenerateOutput {
                    output: run.output,
                    cache: CrossTurnCacheStats {
                        slot_id,
                        prompt_tokens: prompt_len,
                        reused_tokens: if run.prefill_ran {
                            plan.reusable_len
                        } else {
                            0
                        },
                        prefetched_tokens: if run.prefill_ran { plan.suffix_len } else { 0 },
                        mode: plan.mode,
                    },
                },
                run.trace,
            ))
        }

        /// Cache-aware sibling of [`Self::chat_completion_streaming`]: same
        /// ChatML formatting and `<|im_end|>` stop-token handling, routed
        /// through [`Self::generate_streaming_with_prefix_cache`] instead of
        /// `generate_streaming`.
        pub fn chat_completion_streaming_with_prefix_cache<F>(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            messages: &[ChatMessage],
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
        ) -> Result<CachedChatCompletionOutput, crate::error::InferenceError>
        where
            F: FnMut(&str, u32) -> bool,
        {
            self.chat_completion_streaming_with_prefix_cache_and_cancel(
                slot_id,
                messages,
                tokenizer,
                gen_cfg,
                on_token,
                || false,
            )
        }

        /// Cancellation-aware sibling of
        /// [`Self::chat_completion_streaming_with_prefix_cache`]: same ChatML
        /// formatting and `<|im_end|>` stop-token handling, routed through
        /// [`Self::generate_streaming_with_prefix_cache_and_cancel`] instead
        /// of `generate_streaming_with_prefix_cache`. See
        /// [`Self::chat_completion_streaming_with_cancel`] for the analogous
        /// non-cache entry point this mirrors.
        ///
        /// # Errors
        ///
        /// Returns `InferenceError::InvalidInput` if `messages` contains an
        /// inline image, which requires the multimodal vision entry point, or
        /// if the delegated generation path rejects the input.
        pub fn chat_completion_streaming_with_prefix_cache_and_cancel<F, C>(
            &mut self,
            slot_id: crate::kv_cache::CrossTurnSlotId,
            messages: &[ChatMessage],
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
            should_cancel: C,
        ) -> Result<CachedChatCompletionOutput, crate::error::InferenceError>
        where
            F: FnMut(&str, u32) -> bool,
            C: FnMut() -> bool,
        {
            reject_inline_images_for_text_chat(messages)?;
            let prompt = format_chat_template(messages);
            let mut cfg = gen_cfg.clone();
            if let Some(im_end_id) = tokenizer.special_token_id("<|im_end|>")
                && !cfg.stop_token_ids.contains(&im_end_id)
            {
                cfg.stop_token_ids.push(im_end_id);
            }
            let cached = self.generate_streaming_with_prefix_cache_and_cancel(
                slot_id,
                &prompt,
                tokenizer,
                &cfg,
                on_token,
                should_cancel,
            )?;
            let text = cached.output.text.trim_end().to_string();
            Ok(CachedChatCompletionOutput {
                output: ChatCompletionOutput {
                    message: ChatMessage::assistant(text),
                    prompt_tokens: cached.output.prompt_tokens,
                    completion_tokens: cached.output.generated_tokens,
                    stopped: cached.output.stopped,
                },
                cache: cached.cache,
            })
        }
    }

    #[cfg(all(test, feature = "router-learning", feature = "serve", feature = "f16"))]
    pub(crate) fn reference_scoring_fixture() -> (Qwen35Config, ModelWeights) {
        tests::option_scoring_fixture()
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
    pub mod bench_support {
        use super::*;

        #[derive(Clone, Copy, Debug)]
        pub enum LmHeadBenchRoute {
            Full,
            BlockArgmax,
            BlockTopK { local_k: u32 },
        }

        #[derive(Clone, Copy, Debug)]
        pub struct LmHeadBenchOutput {
            pub marker: u32,
        }

        pub struct LmHeadBenchFixture {
            state: MetalQwen35State,
        }

        impl LmHeadBenchFixture {
            pub fn new(state: MetalQwen35State) -> Self {
                Self { state }
            }

            /// Runs one real forward step so `session.activations.hidden`
            /// holds a live hidden state before timing starts — `run_once`
            /// then measures only the head dispatch, not
            /// embedding/attention/MLP.
            pub fn prepare_hidden_for_bench(&mut self, token_id: u32, position: usize) {
                let _ = self.state.forward_step(token_id, position);
            }

            /// Dispatches the real head route once and reads back a minimal
            /// marker to force GPU completion — mirrors what
            /// `LATTICE_DECODE_PROFILE`'s head timing observes in
            /// production.
            pub fn run_once(&mut self, route: LmHeadBenchRoute) -> LmHeadBenchOutput {
                let (gpu_route, topk) = match route {
                    LmHeadBenchRoute::Full => (GpuTopkRoute::CpuFallback, 0usize),
                    LmHeadBenchRoute::BlockArgmax => (GpuTopkRoute::BlockArgmax, 1usize),
                    LmHeadBenchRoute::BlockTopK { local_k } => {
                        (GpuTopkRoute::BlockTopK { local_k }, local_k as usize)
                    }
                };
                self.state.session.compact_route = gpu_route;
                self.state.session.compact_topk = topk;

                let cfg = self.state.engine.config.clone();
                let mut prof = StepProfile::default();
                // Raw pointer breaks the borrow chain, matching the
                // production fused-command-buffer path above: `cmd` is
                // ref-counted by Metal and outlives the `engine.queue`
                // borrow, and `encode_final_head` needs `&mut self` for
                // session state but never touches `engine.queue`.
                let cmd = unsafe {
                    &*(self.state.engine.queue.new_command_buffer()
                        as *const metal::CommandBufferRef)
                };
                let enc = cmd.new_compute_command_encoder();
                let which = self
                    .state
                    .encode_final_head(enc, &cfg, false, &mut prof, false);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();

                let marker = match which {
                    Some(slot) => {
                        // SAFETY: command buffer completed above via
                        // `wait_until_completed`; `topk` candidates were
                        // written into the ping-pong scratch buffer this
                        // dispatch just selected via `slot`.
                        let candidates = unsafe { self.state.read_topk_candidates(slot, topk) };
                        candidates.first().map(|c| c.token_id).unwrap_or(0)
                    }
                    None => {
                        // SAFETY: command buffer completed above via
                        // `wait_until_completed`; `logits[0]` was written by
                        // the full-route GEMV this dispatch just issued.
                        let logit =
                            unsafe { read_buffer(&self.state.session.activations.logits, 1) }[0];
                        logit.to_bits()
                    }
                };
                LmHeadBenchOutput { marker }
            }
        }

        /// Per-chunk GDN/non-GDN wall-clock dispatch time, in milliseconds, for the
        /// #175 harness (`bin/bench_gdn_prefill_ab.rs`). `gdn_ms` covers only the
        /// isolated recurrence dispatch (chunked C32 family or serial
        /// conv1d+recurrence); `non_gdn_ms` covers everything else in the same
        /// `max_prefill`-sized chunk (embedding excluded — see
        /// [`forward_prefill_gdn_isolated_chunk`]'s doc comment).
        #[derive(Clone, Copy, Debug, Default)]
        pub struct GdnIsolatedChunkTiming {
            pub gdn_ms: f64,
            pub non_gdn_ms: f64,
            pub num_gdn_layers: usize,
        }

        impl GdnIsolatedChunkTiming {
            pub fn total_ms(&self) -> f64 {
                self.gdn_ms + self.non_gdn_ms
            }

            pub fn accumulate(&mut self, other: &GdnIsolatedChunkTiming) {
                self.gdn_ms += other.gdn_ms;
                self.non_gdn_ms += other.non_gdn_ms;
                self.num_gdn_layers += other.num_gdn_layers;
            }
        }

        /// #175 harness entry point. Runs ONE `max_prefill`-sized prefill chunk
        /// (`token_ids.len() <= max_prefill(state)`) with GDN recurrence dispatch
        /// isolated into its own command buffer(s) (see
        /// `MetalQwen35State::forward_prefill_chunk_gdn_isolated`'s doc comment for
        /// exactly what changed vs the production dispatch path — nothing GDN-kernel
        /// -related). Returns wall-clock GDN vs non-GDN dispatch time for this chunk.
        ///
        /// For prompts longer than `max_prefill`, the caller must drive the chunking
        /// loop itself (mirrors `forward_prefill_impl`'s production chunking):
        ///
        /// ```ignore
        /// state.set_gdn_chunked(chunked);
        /// state.reset_state();
        /// let mut total = GdnIsolatedChunkTiming::default();
        /// let mp = max_prefill(&state);
        /// let mut start = 0usize;
        /// for chunk in token_ids.chunks(mp) {
        ///     let timing = forward_prefill_gdn_isolated_chunk(&mut state, chunk, start)?;
        ///     total.accumulate(&timing);
        ///     start += chunk.len();
        /// }
        /// ```
        ///
        /// # Errors
        ///
        /// Returns [`crate::error::InferenceError::InvalidInput`] before cache
        /// invalidation or GPU dispatch when `start_pos` is not the live cursor,
        /// a token is out of range, the chunk is empty or too large, or its range
        /// exceeds session capacity.
        pub fn forward_prefill_gdn_isolated_chunk(
            state: &mut MetalQwen35State,
            token_ids: &[u32],
            start_pos: usize,
        ) -> Result<GdnIsolatedChunkTiming, crate::error::InferenceError> {
            preflight_bench_prefill(
                state,
                "bench_support::forward_prefill_gdn_isolated_chunk",
                token_ids,
                start_pos,
            )?;
            Ok(state
                .forward_prefill_chunk_gdn_isolated(token_ids, start_pos, &[])
                .0)
        }

        /// One (GDN layer, value head) target for a [`GdnLayerCapture`] read —
        /// gdn175 S1b real-activation revalidation of the chunkwise WY/DPLR oracle
        /// (`gdn_chunk_ref.rs`) against a real Qwen3.5-0.8B prefill. `linear_idx` is
        /// the 0-based index among GDN ("linear_attention") layers ONLY — the same
        /// counter `forward_prefill_chunk_gdn_isolated` increments once per GDN
        /// layer it processes, NOT the absolute index into `layer_types`.
        #[derive(Clone, Copy, Debug)]
        pub struct GdnCaptureRequest {
            pub linear_idx: usize,
            pub value_head: usize,
        }

        /// Captured per-token chunkwise-scan inputs for one (GDN layer, value
        /// head), de-chunked to `[length, dim]` token order (see
        /// `MetalQwen35State::capture_gdn_chunk_scratch`'s doc comment for the
        /// read-site safety argument and the de-chunking layout proof). `log_alpha`
        /// is RAW (per-token, natural-log, NOT cumulative) — the S1b oracle feed
        /// transform `alpha[t] = exp(log_alpha[t])` is the caller's job, never
        /// `gamma`/cumulative log-alpha (see gdn175 S1b spec: feeding either would
        /// double-apply the decay the oracle's own `chunkwise_gdn` recomputes
        /// internally).
        #[derive(Clone, Debug)]
        pub struct GdnLayerCapture {
            pub linear_idx: usize,
            pub value_head: usize,
            pub length: usize,
            pub dk: usize,
            pub dv: usize,
            pub q: Vec<f32>,         // [length, dk]
            pub k: Vec<f32>,         // [length, dk]
            pub v: Vec<f32>,         // [length, dv]
            pub beta: Vec<f32>,      // [length]
            pub log_alpha: Vec<f32>, // [length], raw natural-log per-token decay
        }

        /// S1b variant of [`forward_prefill_gdn_isolated_chunk`]: identical
        /// dispatch, plus a read-only capture of real per-token GDN-layer
        /// recurrence inputs for each requested (layer, head) target. Capture
        /// cannot perturb the measured/production dispatch — see
        /// `MetalQwen35State::capture_gdn_chunk_scratch`'s doc comment.
        ///
        /// # Errors
        ///
        /// Returns the same pre-dispatch input errors as
        /// [`forward_prefill_gdn_isolated_chunk`].
        pub fn forward_prefill_gdn_isolated_chunk_capture(
            state: &mut MetalQwen35State,
            token_ids: &[u32],
            start_pos: usize,
            capture: &[GdnCaptureRequest],
        ) -> Result<(GdnIsolatedChunkTiming, Vec<GdnLayerCapture>), crate::error::InferenceError>
        {
            preflight_bench_prefill(
                state,
                "bench_support::forward_prefill_gdn_isolated_chunk_capture",
                token_ids,
                start_pos,
            )?;
            Ok(state.forward_prefill_chunk_gdn_isolated(token_ids, start_pos, capture))
        }

        /// The model's vocab size, exposed so harness bins can validate token ids
        /// before entering a measured region. Each forward helper also validates
        /// token ids in its preflight because the embedding copy is an unsafe read
        /// indexed by token id.
        pub fn vocab_size(state: &MetalQwen35State) -> usize {
            state.engine.config.vocab_size
        }

        /// First out-of-vocab token id, if any: `Some((index, id))` where
        /// `id as usize >= vocab`. Pure scan, factored out for testability.
        pub fn first_out_of_vocab(token_ids: &[u32], vocab: usize) -> Option<(usize, u32)> {
            token_ids
                .iter()
                .enumerate()
                .find(|&(_, &id)| id as usize >= vocab)
                .map(|(i, &id)| (i, id))
        }

        fn preflight_bench_prefill(
            state: &mut MetalQwen35State,
            entry_point: &str,
            token_ids: &[u32],
            start_pos: usize,
        ) -> Result<(), crate::error::InferenceError> {
            if token_ids.is_empty() {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "{entry_point}: token_ids must not be empty"
                )));
            }
            if token_ids.len() > state.session.max_prefill {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "{entry_point}: chunk length {} exceeds max_prefill {}",
                    token_ids.len(),
                    state.session.max_prefill
                )));
            }
            state.check_forward_token_ids(entry_point, token_ids)?;
            state.check_forward_range_capacity(start_pos, token_ids.len(), false)?;
            state.check_live_cursor(entry_point, start_pos)?;
            state.cross_turn_prefix_cache.clear();
            Ok(())
        }

        /// True iff this state's config supports the chunked GDN prefill path.
        /// Harness bins must hard-fail when this is false rather than proceed: a
        /// state with `set_gdn_chunked(true)` but no support would otherwise run
        /// the serial recurrence while the harness labels the arm "chunked".
        pub fn gdn_chunked_prefill_supported(state: &MetalQwen35State) -> bool {
            MetalQwen35State::supports_gdn_chunked_prefill(&state.engine.config)
        }

        /// The session's `max_prefill` (single-command-buffer chunk cap; 512 for all
        /// current configs, `max_cache_len.min(512)`) — the caller-side chunking loop
        /// for [`forward_prefill_gdn_isolated_chunk`] must split prompts on this
        /// boundary exactly as the production `forward_prefill_impl` does.
        pub fn max_prefill(state: &MetalQwen35State) -> usize {
            state.session.max_prefill
        }

        /// Production-path calibration entry point for the #175 harness. Runs ONE
        /// `max_prefill`-sized prefill chunk through the real, unmodified
        /// [`MetalQwen35State::forward_prefill_batched_chunk`] dispatch — single
        /// command buffer, no per-layer command-buffer splits — the same path
        /// `forward_prefill_impl` uses in production. This exists so the isolated
        /// harness's per-layer-split measurement can be anchored against an
        /// unmodified-dispatch total, rather than against a stale prior number: the
        /// isolated method's own command-buffer splits add CPU<->GPU sync overhead
        /// that a total-vs-total comparison must not include.
        ///
        /// `all_positions=true` (writes full-attention K/V rows and advances GDN
        /// state for every token in `token_ids`, matching
        /// `forward_prefill_chunk_gdn_isolated`'s contract) and `emit_logits=false`
        /// (skips the terminal RMSNorm/lm_head/top-k tail, again matching the
        /// isolated path, so the two measurements cover comparable work — see
        /// `forward_prefill_chunk_gdn_isolated`'s doc comment for why omitting the
        /// tail does not bias a serial-vs-chunked ratio).
        ///
        /// Returns nothing: unlike the isolated path, production dispatch is a
        /// single command buffer per chunk (one commit + `wait_until_completed`
        /// already inside `forward_prefill_batched_chunk`), so the caller times the
        /// call from outside with a plain CPU `Instant` — no internal
        /// instrumentation is needed or added.
        ///
        /// # Errors
        ///
        /// Returns the same pre-dispatch input errors as
        /// [`forward_prefill_gdn_isolated_chunk`].
        pub fn forward_prefill_production_chunk(
            state: &mut MetalQwen35State,
            token_ids: &[u32],
            start_pos: usize,
        ) -> Result<(), crate::error::InferenceError> {
            preflight_bench_prefill(
                state,
                "bench_support::forward_prefill_production_chunk",
                token_ids,
                start_pos,
            )?;
            let _ = state.forward_prefill_batched_chunk(token_ids, start_pos, true, false, false);
            Ok(())
        }

        /// Issue #1336 harness support (`bin/bench_vision_prefill_ab.rs`): a
        /// checkpoint-shaped multimodal-prefill fixture sized to a
        /// caller-chosen visual-row count, generalizing the
        /// `#[cfg(test)]`-only `vision_gate_fixture` (fixed at 4 rows,
        /// adequate for a correctness gate but too short for the
        /// terminal-head-skip saving to be a measurable fraction of total
        /// prefill time). `t=1`, a square merged grid with
        /// `side_merged = ceil(sqrt(num_visual_rows_target))`, so the actual
        /// row count returned may exceed the target slightly (rounded up to
        /// the next square); the harness reports the actual count.
        ///
        /// Per-lane-varying (not per-row-constant) synthetic content, same
        /// reasoning as `vision_gate_fixture`: a constant vector is
        /// invariant under Qwen3.5's RMSNorm one layer in. Real pixel
        /// content is not required here — the emit_head optimization gates
        /// the decoder's terminal RMSNorm + lm_head dispatch only, which is
        /// insensitive to what values the injected embedding carries, only
        /// to its shape (see this bin's module doc for the corollary: this
        /// harness does not include ViT/merger cost).
        pub fn vision_prefill_fixture(
            cfg: &Qwen35Config,
            num_visual_rows_target: usize,
            seed: f32,
        ) -> Result<crate::vision::multimodal::Qwen35VisionRequest, String> {
            let vision_cfg = cfg
                .vision_config
                .as_ref()
                .ok_or_else(|| "checkpoint carries no vision_config".to_string())?;
            let image_token_id = cfg
                .image_token_id
                .ok_or_else(|| "checkpoint carries no image_token_id".to_string())?;
            let merge = vision_cfg.spatial_merge_size;
            if merge == 0 {
                return Err("checkpoint vision_config.spatial_merge_size is 0".to_string());
            }
            let rows_target = num_visual_rows_target.max(1);
            let side_merged = (rows_target as f64).sqrt().ceil() as usize;
            let side_merged = side_merged.max(1);
            let grid = crate::vision::qwen35_vit::GridThw {
                t: 1,
                h: side_merged * merge,
                w: side_merged * merge,
            };
            let num_rows = (grid.t * grid.h * grid.w) / (merge * merge);

            let mut input_ids = vec![10u32, 11];
            input_ids.extend(std::iter::repeat_n(image_token_id, num_rows));
            input_ids.push(12);

            let mut post_merger_rows = vec![0.0f32; num_rows * cfg.hidden_size];
            for (i, v) in post_merger_rows.iter_mut().enumerate() {
                *v = seed + 0.01 * ((i % 97) as f32 - 48.0);
            }

            Ok(crate::vision::multimodal::Qwen35VisionRequest {
                input_ids,
                image_grids: vec![grid],
                post_merger_rows,
                image_token_id,
                spatial_merge_size: merge,
                decoder_hidden_size: cfg.hidden_size,
            })
        }

        /// Issue #1336 harness support: runs one multimodal prefill over
        /// `request`, selecting `emit_head` per position exactly the way
        /// `generate_multimodal_vision_impl` does in production
        /// (`pos == last_prefill_pos`) when `always_emit_head=false`, or
        /// forcing it `true` at every position (the pre-#1336 baseline
        /// shape) when `always_emit_head=true`. Both arms call the same
        /// `forward_step_mrope`/`forward_step_injected_mrope` functions the
        /// production path calls — same code, same binary, differing only
        /// in this flag — mirroring
        /// `generate_multimodal_vision_emit_head_skip_matches_always_emit_reference`'s
        /// `run_prefill` closure.
        ///
        /// Returns nothing: like [`forward_prefill_production_chunk`], every
        /// `forward_step_mrope`/`forward_step_injected_mrope` call already
        /// waits for its own command buffer before returning, so the caller
        /// times the call from outside with a plain CPU `Instant` — no
        /// internal instrumentation is needed or added. `state` must already
        /// be freshly reset (`state.reset_state()`) by the caller.
        pub fn run_multimodal_prefill(
            state: &mut MetalQwen35State,
            request: &crate::vision::multimodal::Qwen35VisionRequest,
            cfg: &Qwen35Config,
            always_emit_head: bool,
        ) -> Result<Vec<f32>, String> {
            let (_positions, tables) = request
                .build_mrope_tables(cfg)
                .map_err(|e| format!("build_mrope_tables: {e}"))?;
            let prompt_len = request.input_ids.len();

            let mut visual_row = 0usize;
            let mut last_logits = Vec::new();
            for (pos, &token_id) in request.input_ids.iter().enumerate() {
                let cos_sin = Some((tables.cos[pos].as_slice(), tables.sin[pos].as_slice()));
                let emit_head = always_emit_head || pos + 1 == prompt_len;
                if token_id == request.image_token_id {
                    let row_start = visual_row * request.decoder_hidden_size;
                    let row_end = row_start + request.decoder_hidden_size;
                    let row = &request.post_merger_rows[row_start..row_end];
                    visual_row += 1;
                    last_logits = state.forward_step_injected_mrope(
                        row,
                        pos,
                        cos_sin,
                        emit_head,
                        crate::forward::signpost::Scope::NotDecode,
                    );
                } else {
                    last_logits = state.forward_step_mrope(
                        token_id,
                        pos,
                        cos_sin,
                        emit_head,
                        crate::forward::signpost::Scope::NotDecode,
                    );
                }
            }
            Ok(last_logits)
        }

        #[cfg(test)]
        mod tests {
            use super::first_out_of_vocab;

            #[test]
            fn first_out_of_vocab_accepts_in_range() {
                assert_eq!(first_out_of_vocab(&[0, 1, 99], 100), None);
                assert_eq!(first_out_of_vocab(&[], 100), None);
            }

            #[test]
            fn first_out_of_vocab_finds_first_offender() {
                // Two offenders; the FIRST (index 1) must be reported.
                assert_eq!(first_out_of_vocab(&[5, 200, 300], 100), Some((1, 200)));
            }

            #[test]
            fn first_out_of_vocab_rejects_id_equal_to_vocab() {
                // vocab_size itself is out of range (valid ids are 0..vocab_size).
                assert_eq!(first_out_of_vocab(&[100], 100), Some((0, 100)));
            }
        }
    }
}

// Target-independent numerical support follows the macOS + Metal implementation.
#[cfg(test)]
mod public_scheduling_entry_point_tests;

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
) -> Result<(), crate::error::InferenceError> {
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
        cfg: &crate::model::qwen35_config::Qwen35Config,
    ) -> Result<Self, crate::error::InferenceError> {
        fn overflow(what: &str) -> crate::error::InferenceError {
            crate::error::InferenceError::Inference(format!("GDN state shape: {what} overflow"))
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

#[cfg(all(
    test,
    target_os = "macos",
    feature = "metal-gpu",
    feature = "router-learning",
    feature = "serve",
    feature = "f16"
))]
pub(crate) use inner::reference_scoring_fixture;

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
/// **Unstable**: LoRA layer data stub; real impl behind metal-gpu feature.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub struct LoraLayerData {
    pub layer_idx: usize,
    pub module: String,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub d_in: usize,
    pub d_out: usize,
}

/// Stub: blend_lora_layer_data is only meaningful on macOS + metal-gpu.
///
/// Always returns `Err` on other platforms so callers fail at runtime rather
/// than silently producing wrong results.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub fn blend_lora_layer_data(
    _inputs: &[(&[LoraLayerData], f32)],
) -> Result<Vec<LoraLayerData>, crate::error::InferenceError> {
    Err(crate::error::InferenceError::Inference(
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
        _cfg: &crate::model::qwen35_config::Qwen35Config,
        _max_cache_len: usize,
    ) -> Result<Self, String> {
        Err("Metal GPU not available (requires macOS + metal-gpu feature)".into())
    }

    /// **Unstable**: load Q4/F16 directly stub; always fails without metal-gpu feature.
    pub fn from_q4_dir(
        _q4_dir: &std::path::Path,
        _tokenizer_path: &std::path::Path,
        _cfg: &crate::model::qwen35_config::Qwen35Config,
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
    ) -> Result<Vec<f32>, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: fallible Metal hidden-returning step stub.
    pub fn forward_step_with_hidden(
        &mut self,
        _token_id: u32,
        _position: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
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
    /// Always returns [`crate::error::InferenceError::Inference`] without
    /// macOS + `metal-gpu` support.
    pub fn try_forward_prefill(
        &mut self,
        _token_ids: &[u32],
    ) -> Result<Vec<f32>, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: fallible Metal hidden-returning prefill stub.
    pub fn forward_prefill_with_hidden(
        &mut self,
        _token_ids: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>), crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal option-letter scoring stub.
    ///
    /// # Errors
    ///
    /// Always returns [`crate::error::InferenceError::Inference`] without
    /// macOS + `metal-gpu` support.
    pub fn score_option_letters(
        &mut self,
        _prompt_ids: &[u32],
        _letter_ids: &[u32],
    ) -> Result<crate::option_scoring::OptionScores, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
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
    ) -> Result<Vec<u32>, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
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
        _tokenizer: &crate::tokenizer::bpe::BpeTokenizer,
        _cfg: &crate::generation::GenerateConfig,
    ) -> Result<crate::generation::GenerateOutput, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: load LoRA adapter stub; always errors without metal-gpu feature.
    pub fn load_lora_adapter(
        &mut self,
        _layers: Vec<LoraLayerData>,
        _scale: f32,
        _quarot_seed: Option<u64>,
    ) -> Result<(), crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
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
    ) -> Result<Vec<f32>, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal Q4 per-token NLL stub; always errors without metal-gpu feature.
    pub fn compute_token_nlls(
        &mut self,
        _tokens: &[u32],
    ) -> Result<Vec<f32>, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal Q4 perplexity stub; always errors without metal-gpu feature.
    pub fn compute_perplexity(
        &mut self,
        _tokens: &[u32],
        _cfg: &crate::model::qwen35::PerplexityConfig,
    ) -> Result<crate::model::qwen35::PerplexityReport, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
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
    use crate::error::InferenceError;
    use crate::generation::GenerateConfig;
    use crate::tokenizer::bpe::BpeTokenizer;
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

/// lattice#1630 (widening lattice#1584): no production Metal dispatch anywhere under
/// `crates/inference/src` may create a command buffer outside an autorelease pool.
///
/// `metal-rs`'s `new_command_buffer` returns a borrowed reference to an autoreleased,
/// unretained object, so a command buffer created with no enclosing pool survives until
/// the thread's outermost pool drains, which on the long-lived serving worker never
/// happens during a session. The measured cost on the Qwen3.5 decode funnel was 1.3-1.5
/// KB per step (lattice#1584).
///
/// lattice#1584's guard scanned this file's population alone, so the GEMM dispatch in
/// `metal_gemm.rs` and the Qwen3-Embedding forward in `metal.rs` were never swept: same
/// unpooled `new_command_buffer` call, different file. Rather than hand-listing those two
/// files as a second known-sites population (the exact gap that let them go unswept the
/// first time), this guard walks `crates/inference/src` itself and scans every file it
/// finds, so a future production module that dispatches Metal work is discovered
/// automatically. Every function in a discovered file's production region that creates a
/// command buffer must be a `*_dispatch` body reached through a pooling wrapper of the
/// same base name, or appear in `EXEMPT` with its reason.
#[cfg(test)]
mod metal_command_buffer_pool_tests;
