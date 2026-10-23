use super::*;

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
    pub(crate) embed_tokens: EmbeddingTable, // [vocab_size, hidden] — f16 or Q4
    pub(crate) embed_tokens_q8: Q4WeightBuf, // [vocab_size, hidden] — Q8_0/Q4 for logits GEMV
    pub(crate) final_norm: Buffer,           // [hidden] — f32
    pub(crate) rope_cos: Buffer,             // [max_pos * half_rope_dim] — f32
    pub(crate) rope_sin: Buffer,             // [max_pos * half_rope_dim] — f32
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
    /// `use_mtp` (see [`super::super::mtp_route_active`]) commits to the MTP
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

/// Single derivation point for the engine's `has_moe_layer` flag. Both
/// state constructors (f16 and Q4) build the flag from this same layer
/// scan, so a change to layer classification cannot update one
/// constructor and silently leave the other inconsistent.
pub(super) fn derive_has_moe_layer(
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
pub(super) fn build_rope_interleaved(
    rope_dim: usize,
    max_pos: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
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

impl MetalQwen35Engine {
    /// Load immutable model resources from CPU weights. Does NOT allocate any session state.
    ///
    /// `max_cache_len` is NOT accepted here; pass it to `new_session` instead.
    pub fn new(weights: &ModelWeights, cfg: &Qwen35Config) -> Result<Self, String> {
        let device = Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;

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
                let buf =
                    device.new_buffer(combined_size as u64, MTLResourceOptions::StorageModeShared);
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
                    debug_assert_eq!(d.gate_proj.len(), cfg.intermediate_size * cfg.hidden_size);
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
                    let scratch_up =
                        make_zero_buffer(&device, s_inter.max(inter), &format!("L{i}.moe.scr_up"));
                    let scratch_expert_out =
                        make_zero_buffer(&device, hid, &format!("L{i}.moe.scr_exp_out"));
                    let scratch_out = make_zero_buffer(&device, hid, &format!("L{i}.moe.scr_out"));

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
                        dt_bias: make_buffer(&device, &gdn_w.dt_bias, &format!("L{i}.gdn.dt_bias")),
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
                        q_norm: make_buffer(&device, &full_w.q_norm, &format!("L{i}.full.q_norm")),
                        k_norm: make_buffer(&device, &full_w.k_norm, &format!("L{i}.full.k_norm")),
                    })
                }
            };

            layer_weights.push((attn, common));
        }

        // Embedding matrix: f16, shared by the CPU row lookup and the Q8_0 logits kernels.
        let embed_tokens = EmbeddingTable::F16(make_buffer_f16(
            &device,
            &weights.embed_tokens,
            "embed_tokens.f16",
        ));
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
                    gamma_end: make_zero_buffer(device, num_chunks * num_vh, "gdn_chunk_gamma_end"),
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
                    attn_out: make_zero_buffer(device, q_dim_mtp.max(hidden), "mtp.act_attn_out"),
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
    pub(super) fn checkpoint_is_rotated(&self) -> bool {
        self.engine.base_quarot_seed.is_some() || self.engine.quarot_rotation.is_some()
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
}
