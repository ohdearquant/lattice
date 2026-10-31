use super::{ChatMessage, GpuCandidate, MetalQwen35State, choose_gpu_block_topk_route};
use crate::generation::GenerateConfig;
use lattice_tokenizer::__internal as tok;
use lattice_tokenizer::bpe::BpeTokenizer;

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
/// `lattice_model_io::env_switch_enabled`).
/// `selection_env`: `LATTICE_COMPACT_TOPK_SELECT` is enabled (unlocks k>1 routes).
pub(super) fn choose_gpu_topk_route(
    top_k: usize,
    compact_env: bool,
    selection_env: bool,
) -> GpuTopkRoute {
    // R1 benchmark (2026-04-26, M2 Max, vocab=248,320):
    //   CPU NEON argmax k=1 = 89 µs; GPU argmax k=1 = 240 µs (2.70×, t=137, p<<0.0001).
    //   GPU k>1 (bitonic/select64) = 811–940 µs (2.5-3.0× vs CPU 314–366 µs).
    //   Hierarchical k=50 tournament under evaluation (R2, gated by both env flags).
    if compact_env && selection_env && top_k == 50 {
        return GpuTopkRoute::HierarchicalK50;
    }
    GpuTopkRoute::CpuFallback
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
            compact: lattice_model_io::env_switch_enabled("LATTICE_COMPACT_TOPK"),
            selection: lattice_model_io::env_switch_enabled("LATTICE_COMPACT_TOPK_SELECT"),
            approximate_top_p: lattice_model_io::env_switch_enabled("LATTICE_COMPACT_TOPP_APPROX"),
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
pub(super) fn resolve_block_local_k(route: GpuTopkRoute) -> Option<(u32, usize)> {
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

impl MetalQwen35State {
    /// Read top-k compact candidates from the appropriate scratch buffer.
    ///
    /// SAFETY: GPU command buffer must have completed before calling.
    pub(super) unsafe fn read_topk_candidates(
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

pub(super) fn decode_tokens(tokenizer: &BpeTokenizer, ids: &[u32]) -> String {
    tok::decode_tokens(tokenizer, ids)
}

// -----------------------------------------------------------------------
// Chat Completion API
// -----------------------------------------------------------------------

pub(super) const TEXT_CHAT_IMAGE_ERROR: &str =
    "inline image messages require the multimodal vision generation path";

pub(super) fn reject_inline_images_for_text_chat(
    messages: &[ChatMessage],
) -> Result<(), lattice_inference_core::InferenceError> {
    if messages.iter().any(|message| message.image.is_some()) {
        return Err(lattice_inference_core::InferenceError::InvalidInput(
            TEXT_CHAT_IMAGE_ERROR.to_string(),
        ));
    }
    Ok(())
}
