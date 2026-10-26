use super::{
    EmbeddingTable, GDN_CHUNK_SIZE, GatedDeltaNetFusedScratch, GatedDeltaNetState, GdnChunkScratch,
    GpuTopkRoute, InferenceSession, MTP_VERIFY_MAX_TOKENS, MetalCommonLayerWeights,
    MetalCrossTurnPrefixCache, MetalFfnWeights, MetalFullLayerWeights, MetalGdnCheckpointPool,
    MetalGdnLayerWeights, MetalGdnState, MetalKvCache, MetalLayerAttnWeights, MetalMtpActivations,
    MetalMtpCache, MetalMtpDenseMlpWeights, MetalMtpLayerWeights, MetalMtpSession, MetalMtpWeights,
    MetalQwen35Activations, MetalQwen35Engine, MetalQwen35State, MoeMetalBuffers, MtpHiddenTap,
    MtpLoadErr, MtpQ4LoadResult, MtpTensorSource, PathProofCounters, Q4Embedding, Q4WeightBuf,
    QuantFormat, Qwen35Config, RoutedExpertStorage, SELF_SPEC_MAX_DRAFT, active_gdn_layer_indices,
    build_pipelines, build_rope_interleaved, derive_has_moe_layer, make_buffer, make_buffer_f16,
    make_zero_buffer, make_zero_byte_buffer, mmap_q4_weight, resolve_mtp_norm,
    resolve_mtp_projection, validate_flash_decode_shape,
};
#[cfg(feature = "gdn-state-counters")]
use super::{GdnStateTrafficCounters, GdnStateTrafficShape};
use metal::{Buffer, Device, MTLResourceOptions};

impl MetalQwen35State {
    // ---------------------------------------------------------------------------
    // Q4 direct-load helpers
    // ---------------------------------------------------------------------------

    /// Build the sanitized file-stem from a tensor name (dots/slashes → underscores).
    fn q4_tensor_path(dir: &std::path::Path, tensor_name: &str, ext: &str) -> std::path::PathBuf {
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
    /// Used for MTP projections and GDN input projections, which the GPU consumes as f16.
    pub(super) fn make_buffer_f16_from_q4(
        device: &Device,
        tensor: &lattice_weights::q4_weights::Q4Tensor,
        label: &str,
    ) -> Buffer {
        use lattice_weights::q4_weights::{q4_f16_to_f32, q4_f32_to_f16};
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
        use lattice_weights::q4_weights::load_q4_file_checked;
        let tensor = load_q4_file_checked(path, expected_shape)
            .map_err(|e| format!("failed to load Q4 file {}: {e}", path.display()))?;
        let n_blocks = tensor.blocks.len();
        // Re-serialise the blocks into raw bytes (20 bytes each, Q4Block is
        // #[repr(C)] with scale + bias + 16 packed nibbles).
        // SAFETY: Q4Block is #[repr(C)] size 20, alignment 2 (from leading
        // `scale: u16`); byte-cast is valid because target element type is u8.
        let raw: Vec<u8> = unsafe {
            std::slice::from_raw_parts(tensor.blocks.as_ptr().cast::<u8>(), n_blocks * 20).to_vec()
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
    pub(super) fn load_q4_mmap_dequant_f16(
        device: &Device,
        path: &std::path::Path,
        label: &str,
        expected_shape: &[usize],
    ) -> Result<Buffer, String> {
        use lattice_weights::q4_weights::{
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
        use lattice_weights::q4_weights::dequantize_row_q4_0;
        use lattice_weights::q4_weights::{
            Q4BlockCheck, open_and_mmap_q4_file, validate_q4_block_metadata,
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
        let scratch_out = make_zero_buffer(device, hidden, &format!("L{layer_idx}.moe.scr_out"));

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
            // ADR-053 D3' covers Eager storage only: Cached admits experts on the
            // host when they are selected, so it keeps the synchronized CPU router.
            device_route: None,
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
    pub(super) fn load_mtp_q4_weights(
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
        let load_proj_as_half = |name: &str,
                                 expected_shape: &[usize],
                                 label: &str|
         -> Result<Buffer, MtpLoadErr> {
            match resolve_mtp_projection(q4_dir, name, expected_shape, allow_q4)? {
                MtpTensorSource::Q4(tensor) => Ok(MetalQwen35State::make_buffer_f16_from_q4(
                    device, &tensor, label,
                )),
                MtpTensorSource::F16 { values, .. } => Ok(make_buffer_f16(device, &values, label)),
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
        let device = Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;

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
        let load_q4_buf =
            |name: &str, label: &str, expected_shape: &[usize]| -> Result<Q4WeightBuf, String> {
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
        let load_f16_buf_f32 =
            |name: &str, label: &str, expected: &[usize]| -> Result<Buffer, String> {
                let t = std::time::Instant::now();
                let path = Self::q4_tensor_path(q4_dir, name, "f16");
                let (values, _shape) =
                    lattice_weights::q4_weights::load_f16_tensor_file_checked(&path, expected)
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
                            lattice_weights::q4_weights::merged_qkvz_expected_size(qkv_size, z_size)
                                .map_err(|e| format!("layer {i} merge size: {e}"))?;
                        // Content-integrity check (#504 remaining slice 1): size alone
                        // cannot catch a same-size stale/corrupted merged artifact, so
                        // this also verifies the merged file's payload hash against a
                        // hash freshly derived from the *current* qkv/z source files —
                        // fail closed to a rebuild on any mismatch or read error.
                        let is_valid = lattice_weights::q4_weights::merged_qkvz_cache_is_valid(
                            &merged_path,
                            expected_size,
                            &qkv_p,
                            &z_p,
                        );
                        if !is_valid {
                            let _ = std::fs::remove_file(&merged_path);
                            if let Err(e) = lattice_weights::q4_weights::write_merged_qkvz(&qkv_p, &z_p, &merged_path) {
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
                    Self::load_moe_ffn_q4(&device, q4_dir, cfg, &prefix, i, layer_weights.len())?
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
                let load_q4_as_f16_buf =
                    |name: &str, label: &str, expected_shape: &[usize]| -> Result<Buffer, String> {
                        let t = std::time::Instant::now();
                        let path = Self::q4_tensor_path(q4_dir, name, "q4");
                        let tensor = lattice_weights::q4_weights::load_q4_file_checked(
                            &path,
                            expected_shape,
                        )
                        .map_err(|e| format!("failed to load Q4 file {}: {e}", path.display()))?;
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
        //   - embed_tokens / embed_tokens_q8: the Q4 payload for CPU lookup and GPU GEMV
        // ----------------------------------------------------------------
        let embed_name = "model.language_model.embed_tokens.weight";
        let embed_path = Self::q4_tensor_path(q4_dir, embed_name, "q4");
        let embed_tokens = EmbeddingTable::Q4(Q4Embedding::open(
            &embed_path,
            cfg.vocab_size,
            hidden,
            embed_name,
        )?);
        // Q4 buffer for GPU logits GEMV — mmap no-copy
        let embed_tokens_q8 = mmap_q4_weight(
            &device,
            &embed_path,
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
        let gdn_geometry =
            lattice_compute::attention::gdn::qwen35_gdn_geometry(cfg).map_err(|e| e.to_string())?;
        let gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::from_geometry(&gdn_geometry))
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
        let rotation_target = lattice_quant::quarot::plan::qwen35_rotation_target(cfg);
        let index_seed = lattice_quant::__internal::read_quarot_seed_from_index(
            q4_dir,
            &rotation_target,
            cfg.quarot_rotation_seed,
        )
        .map_err(|e| format!("from_q4_dir: {e}"))?;
        let quarot_seed_opt = index_seed.or(cfg.quarot_rotation_seed);
        let is_quarot = quarot_seed_opt.is_some();

        // Load MTP weights (cache+activations go into session, not engine).
        let mtp_requested = lattice_model_io::env_switch_enabled("LATTICE_MTP");
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
            mtp_weights_opt.is_some() || lattice_model_io::env_switch_enabled("LATTICE_SELF_SPEC");
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
                    lattice_quant::quarot::hadamard::RandomizedHadamard::new(seed, hidden)
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
                gdn_state_traffic: if lattice_model_io::env_switch_enabled(
                    "LATTICE_GDN_STATE_COUNTERS",
                ) {
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
}
