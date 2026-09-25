//! Qwen Metal state and adapter residency confined to one serving worker.

use super::ServingRuntime;
use crate::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::kv_cache::CrossTurnSlotId;
use crate::serve::lora::{
    AdapterControlError, AdapterControlResult, AdapterIndex, LoraSelection, ResidencyLimits,
};
use crate::serve::lora_registry::ResidencyRegistry;
use crate::serve::metal_worker::{
    AdapterCommand, JobRoute, VisionRequestBuild, VisionRuntime, WorkerFailure, WorkerMetadata,
    build_vision_request, cancelled_output, check_prompt_fits_window, classify_job,
    render_text_prompt_within_window,
};
use crate::tokenizer::bpe::BpeTokenizer;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

pub(crate) struct QwenMetalRuntime {
    state: MetalQwen35State,
    tokenizer: Arc<BpeTokenizer>,
    vision: VisionRuntime,
    registry: ResidencyRegistry,
    metadata: WorkerMetadata,
}

impl QwenMetalRuntime {
    pub(crate) fn new(
        state: MetalQwen35State,
        tokenizer: Arc<BpeTokenizer>,
        vision: VisionRuntime,
        metadata: WorkerMetadata,
        index: Arc<RwLock<AdapterIndex>>,
        limits: ResidencyLimits,
    ) -> Self {
        Self {
            state,
            tokenizer,
            vision,
            registry: ResidencyRegistry::new(index, limits),
            metadata,
        }
    }
}

impl ServingRuntime for QwenMetalRuntime {
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        cfg: &GenerateConfig,
        lora: &[LoraSelection],
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        let state = &mut self.state;
        let tokenizer = self.tokenizer.as_ref();
        let vision_runtime = &mut self.vision;
        let registry = &mut self.registry;
        let meta = &self.metadata;
        if let JobRoute::Vision {
            message_index: image_message_index,
        } = classify_job(messages)?
        {
            if should_cancel() {
                return Ok(cancelled_output());
            }
            let config = state.engine.config.clone();
            let (request, metal_dispatches, gemm_calls) = match build_vision_request(
                vision_runtime,
                &config,
                tokenizer,
                messages,
                image_message_index,
                should_cancel,
                |prompt_len| {
                    check_prompt_fits_window(
                        meta.context_window_policy,
                        meta.model_max_context,
                        prompt_len,
                        cfg,
                    )
                },
            )? {
                VisionRequestBuild::Ready {
                    request,
                    metal_dispatches,
                    gemm_calls,
                } => (request, metal_dispatches, gemm_calls),
                VisionRequestBuild::Cancelled => return Ok(cancelled_output()),
            };
            if should_cancel() {
                return Ok(cancelled_output());
            }
            eprintln!(
                "[metal-worker] route=vision dispatch=multimodal \
                 metal_gemm_dispatches={metal_dispatches} \
                 metal_gemm_calls={gemm_calls}"
            );
            registry
                .apply(lora, state)
                .map_err(WorkerFailure::Rejected)?;
            let output = state
                .generate_multimodal_vision_with_cancel(&request, tokenizer, cfg, should_cancel)
                .map_err(WorkerFailure::from)?;
            if !output.text.is_empty() {
                let _ = on_token(&output.text, 0);
            }
            return Ok(output);
        }

        // Render the ChatML prompt exactly once (#828/#832: the prior
        // `lattice_serve.rs` path rendered it a second time inside its own
        // window preflight); reused for both the window check and generation.
        let (prompt, _prompt_len) = render_text_prompt_within_window(
            tokenizer,
            messages,
            meta.context_window_policy,
            meta.model_max_context,
            cfg,
        )
        .map_err(WorkerFailure::Rejected)?;

        // Cache-aware + cancellation-aware call (#462/#744):
        // reuses the previous turn's shared token prefix
        // instead of a full re-prefill on every request, and
        // observes client disconnect before prefill,
        // immediately after prefill, and at the top of every
        // decode iteration. This worker thread owns one
        // `MetalQwen35State` for the whole process lifetime, so
        // `CrossTurnSlotId::DEFAULT` is the only slot that
        // exists; the planner re-verifies the retained prefix
        // against this request's prompt on every call and
        // falls back to `PrefixReuseMode::FullRefill` whenever
        // they diverge, so correctness never depends on
        // distinguishing clients.
        //
        // DEPLOYMENT ASSUMPTION, stated because it is currently
        // true only by the accident that no multi-tenant consumer
        // exists: this path assumes a single tenant, or clients
        // that mutually trust one another. Reuse-versus-refill is
        // externally visible as latency, so while no request can
        // read another's content, a client CAN observe that some
        // other request recently shared a prefix with its own.
        // A shared inference endpoint serving mutually distrusting
        // clients must key the slot per tenant via
        // `CrossTurnSlotId::new`, not inherit `DEFAULT`.
        if should_cancel() {
            return Ok(cancelled_output());
        }
        registry
            .apply(lora, state)
            .map_err(WorkerFailure::Rejected)?;
        let cached = state.generate_streaming_with_prefix_cache_and_cancel(
            CrossTurnSlotId::DEFAULT,
            &prompt,
            tokenizer,
            cfg,
            on_token,
            should_cancel,
        );
        if let Ok(c) = &cached {
            eprintln!(
                "[metal-worker] cross-turn cache: mode={:?} reused={} \
                 prefetched={} prompt={}",
                c.cache.mode,
                c.cache.reused_tokens,
                c.cache.prefetched_tokens,
                c.cache.prompt_tokens,
            );
        }
        cached.map(|c| c.output).map_err(WorkerFailure::from)
    }

    fn control(
        &mut self,
        command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError> {
        match command {
            AdapterCommand::Load {
                name,
                path,
                layers,
                descriptor,
            } => {
                let id = self.registry.load(name, path, layers, *descriptor)?;
                self.registry.metadata(id).map(AdapterControlResult::Loaded)
            }
            AdapterCommand::Unload { id } => self
                .registry
                .unload(id, &mut self.state)
                .map(AdapterControlResult::Unloaded),
        }
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        self.vision.shared_capability()
    }
}
