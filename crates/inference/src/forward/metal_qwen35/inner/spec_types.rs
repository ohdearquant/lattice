use super::{MetalGdnState, MetalGdnStatePrecision, MtpHiddenTap};
use metal::{Buffer, Device, MTLResourceOptions};

// ADR-051 §"MTP tensors still safety-skipped in quantize_quarot": Phase 1 keeps
// every MTP projection as f16 (unquantized) so the counter-rotation operates on
// unquantized weights. Buffers hold raw f16 (u16-on-disk converted via
// `make_buffer_f16`); the GEMV path uses `dispatch_matmul_half`. Phase 2 will
// rotate and quantize these tensors.
pub(super) struct MetalMtpDenseMlpWeights {
    pub(super) gate_proj: Buffer, // f16, [intermediate, hidden]
    pub(super) up_proj: Buffer,   // f16, [intermediate, hidden]
    pub(super) down_proj: Buffer, // f16, [hidden, intermediate]
}

pub(super) struct MetalMtpLayerWeights {
    pub(super) input_layernorm: Buffer,          // [hidden]
    pub(super) post_attention_layernorm: Buffer, // [hidden]
    pub(super) q_proj: Buffer,                   // f16, [2*q_dim, hidden]
    pub(super) k_proj: Buffer,                   // f16, [kv_dim, hidden]
    pub(super) v_proj: Buffer,                   // f16, [kv_dim, hidden]
    pub(super) o_proj: Buffer,                   // f16, [hidden, q_dim]
    pub(super) q_norm: Buffer,                   // [head_dim]
    pub(super) k_norm: Buffer,                   // [head_dim]
    pub(super) mlp: MetalMtpDenseMlpWeights,
}

pub(crate) struct MetalMtpWeights {
    pub(super) fc: Buffer,                    // f16, [hidden, 2*hidden]
    pub(super) pre_fc_norm_embedding: Buffer, // [hidden]
    pub(super) pre_fc_norm_hidden: Buffer,    // [hidden]
    pub(super) layers: Vec<MetalMtpLayerWeights>,
    pub(super) norm: Buffer, // [hidden]
    /// Which target hidden the head receives; decided once at load.
    pub(super) hidden_tap: MtpHiddenTap,
}

pub(super) struct MtpQ4LoadResult {
    pub(super) weights: Option<MetalMtpWeights>,
    pub(super) first_missing_file: Option<std::path::PathBuf>,
}

pub(crate) struct MetalMtpCache {
    pub(super) k_buf: Buffer,
    pub(super) v_buf: Buffer,
    pub(super) seq_len: usize,
    #[allow(dead_code)] // capacity bound stored for future MTP bounds checking
    pub(super) max_cache_len: usize,
}

impl MetalMtpCache {
    pub(super) fn reset(&mut self) {
        self.seq_len = 0;
    }
}

pub(crate) struct MetalMtpActivations {
    pub(super) fused: Buffer, // [2*hidden] — normed embedding concat normed hidden
    pub(super) hidden: Buffer, // [hidden]
    pub(super) residual: Buffer, // [hidden]
    pub(super) q: Buffer,     // [2*q_dim] interleaved Q+gate from q_proj
    pub(super) q_separated: Buffer, // [q_dim] Q after scatter
    pub(super) gate_z: Buffer, // [q_dim] gate after scatter
    pub(super) k: Buffer,     // [kv_dim]
    pub(super) v: Buffer,     // [kv_dim]
    pub(super) attn_out: Buffer, // [q_dim.max(hidden)]
    pub(super) gate: Buffer,  // [intermediate]
    pub(super) up: Buffer,    // [intermediate]
    pub(super) ffn_out: Buffer, // [hidden]
    pub(super) logits: Buffer, // [vocab]
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
    pub(super) max_tokens: usize,
    pub(super) conv_slots: Vec<Vec<Buffer>>, // [slot][linear_layer]
    pub(super) s_slots: Vec<Vec<Buffer>>,    // [slot][linear_layer]
    pub(super) active_base_seq_len: Option<usize>,
    pub(super) mtp_base_seq_len: Option<usize>,
    /// Batch-verifier repair: (token_id, position) to replay on rejection for GDN correctness.
    pub(super) batch_repair_token: Option<(u32, usize)>,
}

impl MetalGdnCheckpointPool {
    pub(super) fn new(device: &Device, max_tokens: usize, state: &MetalGdnState) -> Self {
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

pub(super) struct MetalVerifyOutput {
    pub(super) logits: Vec<Vec<f32>>, // per-token full-vocab logits
    #[allow(dead_code)]
    // pre-final hidden retained for MTP chaining; not yet consumed by callers
    pub(super) final_hidden: Vec<f32>, // pre-final hidden of last verified row
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
    pub(super) first_pre_final_hidden: Vec<f32>,
    // Raw (pre-final-RMSNorm) hidden state of every verified row, in token
    // order. Filled only by `verify_tokens_batched_impl(.., keep_all_rows =
    // true)` and empty everywhere else, so the greedy MTP loop's verify
    // pays no extra clone. The consumer is
    // `MtpTargetVerifier::verify_tokens_with_hidden`.
    pub(super) pre_final_hidden_rows: Vec<Vec<f32>>,
}

pub(super) struct MetalStepOutput {
    pub(super) logits: Vec<f32>,
    pub(super) pre_final_hidden: Vec<f32>,
    // Kept alive by binding the returned struct (not discarding it),
    // rather than dropping at the end of `forward_step_inner_impl`'s
    // body — the zero-copy greedy path needs `decode.step` to span
    // through its post-return logits scan (see
    // `forward_step_greedy_argmax`); every other caller either drops
    // this immediately (`.logits`, same timing as before) or holds a
    // `Scope::NotDecode` no-op guard, which is a no-op to drop late.
    pub(super) _signpost_step: crate::forward::signpost::Interval,
}

pub(super) struct MetalMtpForwardOutput {
    pub(super) token_id: u32,
    pub(super) logits: Vec<f32>,
}

/// Decoding metrics emitted when LATTICE_MTP=1.
#[derive(Default)]
pub(crate) struct MetalMtpDecodeMetrics {
    pub(super) rounds: usize,
    pub(super) mtp_forwards: usize,
    pub(super) verify_calls: usize,
    pub(super) accepted_extra_tokens: usize,
    pub(super) fallback_tokens: usize,
    pub(super) mtp_ms: f64,
    pub(super) verify_ms: f64,
    pub(super) rollback_ms: f64,
}

/// Decoding metrics emitted when LATTICE_SELF_SPEC_VERBOSE=1.
#[derive(Default)]
pub(crate) struct SelfSpecMetrics {
    pub(super) rounds: usize,
    pub(super) draft_forwards: usize,
    pub(super) verify_calls: usize,
    pub(super) accepted_extra_tokens: usize,
    pub(super) fallback_tokens: usize,
    pub(super) draft_ms: f64,
    pub(super) verify_ms: f64,
    pub(super) rollback_ms: f64,
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
