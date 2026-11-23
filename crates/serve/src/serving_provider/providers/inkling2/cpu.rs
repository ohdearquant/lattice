//! Inkling 2 text generation on the CPU for `lattice serve`.

use super::preparation::{
    Inkling2PromptAdapter, Marker, PromptPiece, normalize_standalone_inkling2,
    prepare_inkling2_chat_request, prompt_pieces,
};
use crate::serve::ApiError;
use crate::serve::contract::{ChatRequest, GenerationDefaults, ValidatedChatRequest};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::lora::{AdapterControlError, AdapterControlResult};
use crate::serve::prepare::{PreparedChatRequest, lora_unsupported_backend};
use crate::serve::route::{DriverEvidence, ServedRoute};
use crate::serving_cpu_host::{CpuRuntimeSource, SharedCpuHandle};
use crate::serving_host::ServedOutput;
use crate::serving_preparation::{PreparationHandle, PreparedCpuChat, RequestPreparation};
use crate::serving_provider::{CheckpointEvidence, resolve_serve_context};
use crate::serving_runtime::{ServingRuntime, WorkerFailure};
use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_worker::AdapterCommand;
use lattice_inference::error::InferenceError;
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_model_inkling2::{Inkling2CpuDecoder, Inkling2PromptPart};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// A loaded Inkling 2 checkpoint, ready to serve chat requests on the CPU.
pub(crate) struct Inkling2CpuServing {
    decoder: Inkling2CpuDecoder,
    adapter: Inkling2PromptAdapter,
    max_context: usize,
    /// The id of each template marker, indexed by [`Marker`].
    marker_ids: [u32; Marker::ALL.len()],
}

/// The parts of a framed prompt, each marker as its id in `marker_ids`, which is indexed by
/// [`Marker`].
pub(super) fn prompt_parts<'a>(
    framed: &'a str,
    marker_ids: &[u32; Marker::ALL.len()],
) -> Vec<Inkling2PromptPart<'a>> {
    prompt_pieces(framed)
        .into_iter()
        .map(|piece| match piece {
            PromptPiece::Marker(marker) => Inkling2PromptPart::Token(marker_ids[marker as usize]),
            PromptPiece::Text(text) => Inkling2PromptPart::Text(text),
        })
        .collect()
}

impl Inkling2CpuServing {
    /// Load the checkpoint in `dir` and resolve the served context window against
    /// `requested`.
    ///
    /// The stop id, read before the weights, is the `eos_token_id` of `generation_config.json`,
    /// which must equal the `config.json` end-of-sequence id that the decoder stops on.
    ///
    /// # Errors
    /// Fails when the checkpoint has no single stop id, see
    /// [`Inkling2PromptAdapter::checkpoint_stop_token_id`], when the model cannot be loaded, or
    /// when the tokenizer lacks a template marker or spells one with several ids.
    fn load(dir: &Path, requested: Option<usize>) -> Result<Self, InferenceError> {
        let eos_token_id = Inkling2PromptAdapter::checkpoint_stop_token_id(dir)?;
        let decoder = Inkling2CpuDecoder::open(dir)?;
        let (max_context, source) =
            resolve_serve_context(decoder.model().config().max_position_embeddings, requested);
        let serving = Self::new(decoder, eos_token_id, max_context)?;
        eprintln!(
            "[lattice serve] CPU context window {max_context} ({})",
            source.describe()
        );
        Ok(serving)
    }

    /// Serve `decoder`, stopping on `eos_token_id`, with a context window of `max_context`
    /// tokens.
    ///
    /// # Errors
    /// Fails when the tokenizer lacks a template marker or spells one with several ids.
    pub(super) fn new(
        decoder: Inkling2CpuDecoder,
        eos_token_id: u32,
        max_context: usize,
    ) -> Result<Self, InferenceError> {
        let mut marker_ids = [0; Marker::ALL.len()];
        for marker in Marker::ALL {
            marker_ids[marker as usize] = decoder.marker_id(marker.spelling())?;
        }
        Ok(Self {
            decoder,
            adapter: Inkling2PromptAdapter::new(eos_token_id),
            max_context,
            marker_ids,
        })
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    /// The token ids of a framed prompt: each marker as its id, the text between with no
    /// added-token matching. Admission and generation both count with this.
    pub(super) fn tokenize_prompt(&self, framed: &str) -> Vec<u32> {
        self.decoder
            .tokenize_parts(&prompt_parts(framed, &self.marker_ids))
    }

    fn tokenize_len(&self, framed: &str) -> usize {
        self.tokenize_prompt(framed).len()
    }

    fn prepare(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<(PreparedChatRequest, GenerateConfig), ApiError> {
        let prepared = prepare_inkling2_chat_request(
            &self.adapter,
            req,
            model_id,
            default_max_tokens,
            max_tokens_cap,
            |prompt| self.tokenize_len(prompt),
            || self.max_context,
        )?;
        let gen_cfg = prepared.gen_cfg;
        Ok((
            PreparedChatRequest {
                messages: Vec::new(),
                max_tokens: gen_cfg.max_new_tokens,
                total_token_cap: gen_cfg.total_token_cap,
                temperature: gen_cfg.temperature,
                top_p: gen_cfg.top_p,
                logprobs: gen_cfg.logprobs,
                prompt: prepared.prompt.into_framed(),
                stop_strings: gen_cfg.stop_strings.clone(),
                reasoning_budget: gen_cfg.reasoning_budget,
                seed: gen_cfg.seed,
                stream: prepared.stream,
            },
            gen_cfg,
        ))
    }

    /// Generate the reply to a prepared `prompt`. The output's text, and what `on_content`
    /// receives, is the reply's content field; `on_reasoning` receives its reasoning field,
    /// which the returned string holds whole.
    fn generate<F, R, C>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        on_content: F,
        on_reasoning: R,
        should_cancel: C,
    ) -> Result<(GenerateOutput, String, DriverEvidence), InferenceError>
    where
        F: FnMut(&str) -> bool,
        R: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        let ids = self.tokenize_prompt(prompt);
        let reply =
            self.decoder
                .generate_reply(&ids, gen_cfg, on_content, on_reasoning, should_cancel)?;
        Ok((
            reply.output,
            reply.reasoning,
            DriverEvidence {
                opened: reply.trace.opened,
                consumed: reply.trace.consumed,
            },
        ))
    }
}

struct Inkling2CpuSource {
    serving: Arc<Inkling2CpuServing>,
}

impl CpuRuntimeSource for Inkling2CpuSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(Inkling2PreparedCpuRuntime {
            serving: &self.serving,
            vision: Arc::new(AtomicBool::new(false)),
        })
    }
}

/// Runs a prepared Inkling 2 prompt on a request's blocking task.
struct Inkling2PreparedCpuRuntime<'a> {
    serving: &'a Inkling2CpuServing,
    vision: Arc<AtomicBool>,
}

impl ServingRuntime for Inkling2PreparedCpuRuntime<'_> {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        self.execute_with_reasoning(
            input,
            cfg,
            http_stream,
            on_token,
            &mut |_| true,
            should_cancel,
        )
        .map(|served| served.output)
    }

    fn execute_with_reasoning(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        on_reasoning: &mut dyn FnMut(&str) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<ServedOutput, WorkerFailure> {
        let RuntimeInput::PreparedText { prompt, entry } = input else {
            return Err(WorkerFailure::Failed(
                "chat messages are not supported by the prepared CPU runtime".to_owned(),
            ));
        };
        let (output, reasoning, evidence) = match entry {
            TextGenerationEntry::Complete => self
                .serving
                .generate(prompt, cfg, |_| true, |_| true, || false)
                .map_err(|error| WorkerFailure::Failed(error.to_string()))?,
            TextGenerationEntry::StreamingWithCancel => self
                .serving
                .generate(
                    prompt,
                    cfg,
                    |delta| on_token(delta, 0),
                    |piece| on_reasoning(piece),
                    should_cancel,
                )
                .map_err(|error| WorkerFailure::Failed(error.to_string()))?,
        };
        eprintln!(
            "{}",
            ServedRoute::INKLING2_CPU.request_marker(http_stream, evidence)
        );
        Ok(ServedOutput::new(output, reasoning))
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError> {
        Err(AdapterControlError::InvalidAdapter(
            "runtime LoRA adapters are not supported for this model".to_string(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.vision)
    }
}

struct Inkling2Preparation {
    serving: Arc<Inkling2CpuServing>,
}

impl RequestPreparation for Inkling2Preparation {
    fn tokenize_len(&self, prompt: &str) -> usize {
        self.serving.tokenize_len(prompt)
    }

    fn max_context(&self) -> usize {
        self.serving.max_context()
    }

    fn tokenizer(&self) -> Option<&BpeTokenizer> {
        None
    }

    fn prepare_cpu(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<PreparedCpuChat, ApiError> {
        let (prepared, config) =
            self.serving
                .prepare(req, model_id, default_max_tokens, max_tokens_cap)?;
        Ok(PreparedCpuChat { prepared, config })
    }

    fn prepare_lattice_with_thinking(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        _vision_supported: bool,
    ) -> Result<(PreparedChatRequest, bool), ApiError> {
        self.serving
            .prepare(req, model_id, default_max_tokens, max_tokens_cap)
            .map(|(prepared, config)| (prepared, config.enable_thinking))
    }

    fn normalize_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        _vision_supported: bool,
    ) -> Result<ValidatedChatRequest, ApiError> {
        normalize_standalone_inkling2(
            &self.serving.adapter,
            req,
            defaults,
            model_id,
            self.serving.max_context,
        )
    }

    fn standalone_generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        self.serving.adapter.generate_config(req)
    }

    fn lattice_generate_config(
        &self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: Option<u64>,
        stop_strings: Vec<String>,
        reasoning_budget: Option<usize>,
        logprobs: Option<usize>,
        _enable_thinking: bool,
    ) -> GenerateConfig {
        let mut c = GenerateConfig::default();
        c.max_new_tokens = max_tokens;
        c.temperature = temperature;
        c.top_p = top_p;
        c.seed = seed;
        c.stop_token_ids = self.serving.adapter.stop_token_ids().to_vec();
        c.enable_thinking = false;
        c.stop_strings = stop_strings;
        c.reasoning_budget = reasoning_budget;
        c.logprobs = logprobs;
        c
    }

    fn refuse_standalone_unsupported(&self, req: &ChatRequest) -> Result<(), ApiError> {
        if req
            .lora
            .as_ref()
            .is_some_and(|selection| !selection.is_empty())
        {
            return Err(lora_unsupported_backend());
        }
        if req
            .response_format
            .as_ref()
            .is_some_and(|format| format.r#type == "json_schema")
        {
            return Err(ApiError::BadRequest {
                message: "response_format.type 'json_schema' is not supported for this model; \
                          use 'text'"
                    .to_string(),
                code: "unsupported_feature",
            });
        }
        Ok(())
    }

    fn supports_adapters(&self) -> bool {
        false
    }
}

/// Load the claimed checkpoint for the `lattice serve` CPU route.
pub(crate) fn load(
    evidence: &CheckpointEvidence,
    requested_context: Option<usize>,
) -> Result<SharedCpuHandle, String> {
    let serving = Inkling2CpuServing::load(&evidence.directory, requested_context)
        .map_err(|error| format!("failed to load Inkling 2 model: {error}"))?;
    Ok(handle(serving))
}

/// The CPU host of `serving`, with its request preparation.
pub(super) fn handle(serving: Inkling2CpuServing) -> SharedCpuHandle {
    let serving = Arc::new(serving);
    let preparation = PreparationHandle::new(Arc::new(Inkling2Preparation {
        serving: Arc::clone(&serving),
    }));
    let source: Arc<dyn CpuRuntimeSource> = Arc::new(Inkling2CpuSource { serving });
    SharedCpuHandle::new(source, preparation)
}

#[cfg(test)]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Inkling2CpuSource>();
};
