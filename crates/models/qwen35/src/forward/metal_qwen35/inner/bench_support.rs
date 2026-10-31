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
            &*(self.state.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
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
                let logit = unsafe { read_buffer(&self.state.session.activations.logits, 1) }[0];
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
/// Returns [`lattice_inference_core::InferenceError::InvalidInput`] before cache
/// invalidation or GPU dispatch when `start_pos` is not the live cursor,
/// a token is out of range, the chunk is empty or too large, or its range
/// exceeds session capacity.
pub fn forward_prefill_gdn_isolated_chunk(
    state: &mut MetalQwen35State,
    token_ids: &[u32],
    start_pos: usize,
) -> Result<GdnIsolatedChunkTiming, lattice_inference_core::InferenceError> {
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
) -> Result<(GdnIsolatedChunkTiming, Vec<GdnLayerCapture>), lattice_inference_core::InferenceError>
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
) -> Result<(), lattice_inference_core::InferenceError> {
    if token_ids.is_empty() {
        return Err(lattice_inference_core::InferenceError::InvalidInput(
            format!("{entry_point}: token_ids must not be empty"),
        ));
    }
    if token_ids.len() > state.session.max_prefill {
        return Err(lattice_inference_core::InferenceError::InvalidInput(
            format!(
                "{entry_point}: chunk length {} exceeds max_prefill {}",
                token_ids.len(),
                state.session.max_prefill
            ),
        ));
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
) -> Result<(), lattice_inference_core::InferenceError> {
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
