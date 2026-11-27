//! CPU serving runtimes for `lattice serve`.
//!
//! Lives beside `serving_factory`, outside `serve/`, so the serving surface
//! does not name a concrete model type: the binaries hold a
//! [`GemmaCpuServing`] and the traced Qwen entries below, not the models.
//!
//! [`GemmaCpuServing`] is Gemma 4 E2B text serving on the CPU backend. It owns
//! what `lattice serve` needs to answer chat requests for a Gemma 4
//! safetensors checkpoint: the model, a tokenizer bounded by the checkpoint's
//! context window, and the Gemma prompt adapter that reads the chat
//! conventions (BOS spelling, stop ids) from the checkpoint. Preparation goes
//! through [`prepare_gemma_chat_request`], and generation runs the Gemma CPU
//! session under the shared decoder driver, returning the driver's ledger
//! counters beside the output so the caller can record that it did.
//!
//! The standalone `lattice_serve` binary reaches the same type through its
//! worker: the preparation entries below (`normalize_standalone`,
//! `standalone_generate_config`, `render_within_window`) are the
//! per-request steps the worker-local Gemma runtime and the standalone
//! handler take, and the checkpoint is loaded by the same [`GemmaCpuServing::load`].
//!
//! [`qwen_generate_traced`] and [`qwen_generate_streaming_traced`] run the
//! Qwen3.5 CPU entries and return the same counters.
//!
//! Not a stable API: `#[doc(hidden)]` is a convention, not a semver guarantee.
//!
//! [`GemmaCpuServing`]: crate::serving_cpu::GemmaCpuServing
//! [`prepare_gemma_chat_request`]: crate::serve::prepare::prepare_gemma_chat_request
//! [`GemmaCpuServing::load`]: crate::serving_cpu::GemmaCpuServing::load
//! [`qwen_generate_traced`]: crate::serving_cpu::qwen_generate_traced
//! [`qwen_generate_streaming_traced`]: crate::serving_cpu::qwen_generate_streaming_traced

use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, MessageContent, NormalizedChatMessage, NormalizedChatRole,
    ServeProfile, ValidatedChatRequest, normalize_requested_options,
    validate_context_window_with_budget,
};
use crate::serve::prepare::{GemmaPromptAdapter, PreparedChatRequest, prepare_gemma_chat_request};
use crate::serve::prompt_adapter::PromptAdapter as _;
use crate::serve::route::DriverEvidence;
use lattice_inference::error::InferenceError;
use lattice_inference::forward::metal_qwen35::{ChatMessage, ChatRole};
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use lattice_inference::model::gemma4_config::Gemma4Config;
use lattice_inference::model::gemma4_model::Gemma4Model;
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use std::path::Path;

/// A loaded Gemma 4 E2B text checkpoint, ready to serve on the CPU.
#[doc(hidden)]
pub struct GemmaCpuServing {
    model: Gemma4Model,
    tokenizer: GemmaBpeTokenizer,
    adapter: GemmaPromptAdapter,
    max_context: usize,
}

impl std::fmt::Debug for GemmaCpuServing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GemmaCpuServing")
            .field("max_context", &self.max_context)
            .finish_non_exhaustive()
    }
}

impl GemmaCpuServing {
    /// Load the checkpoint in `dir`.
    ///
    /// The configuration, tokenizer and chat conventions are read before the
    /// weights, so a directory whose prompt files are wrong fails without
    /// first loading the model.
    ///
    /// # Errors
    /// Fails when the directory is not a supported Gemma 4 E2B text
    /// checkpoint, or when its tokenizer, chat conventions or weights cannot
    /// be loaded.
    pub fn load(dir: &Path) -> Result<Self, InferenceError> {
        let max_context = Gemma4Config::from_model_dir(dir)?.max_position_embeddings;
        // Truncation happens at the context window and never below it, so the
        // context check sees the whole prompt.
        let tokenizer = GemmaBpeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json"))?
            .with_max_seq_len(max_context);
        let adapter = GemmaPromptAdapter::from_model_dir(dir, &tokenizer)?;
        let model = Gemma4Model::from_safetensors(dir)?;
        Ok(Self {
            model,
            tokenizer,
            adapter,
            max_context,
        })
    }

    /// The checkpoint's context window in tokens.
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Token count of a rendered prompt, counted as generation tokenizes it:
    /// each template marker as its id and the text between with no added-token
    /// matching. Nothing is truncated.
    pub fn tokenize_len(&self, prompt: &str) -> usize {
        self.adapter.prompt_token_ids(&self.tokenizer, prompt).len()
    }

    /// Validate `req` under the `lattice serve` profile, refuse every control
    /// Gemma cannot serve, render the prompt with the checkpoint's chat
    /// template and check it against the context window.
    ///
    /// Returns the request in the shape the serving handler consumes, plus the
    /// `GenerateConfig` the Gemma adapter built for it.
    ///
    /// # Errors
    /// The contract's refusal for a malformed request, an unsupported control
    /// or a prompt that does not fit.
    pub fn prepare(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<(PreparedChatRequest, GenerateConfig), ApiError> {
        let prepared = prepare_gemma_chat_request(
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
                prompt: prepared.prompt,
                stop_strings: gen_cfg.stop_strings.clone(),
                reasoning_budget: gen_cfg.reasoning_budget,
                seed: gen_cfg.seed,
                stream: prepared.stream,
            },
            gen_cfg,
        ))
    }

    /// Validate `req` under the standalone `lattice_serve` profile and apply
    /// the Gemma defaults step, refusing every control Gemma cannot serve.
    ///
    /// The refusals and their order are those of [`Self::prepare`]: the
    /// profile's own validation first (image content included), then typed
    /// content parts, then the adapter's refusals of `logprobs`, stop
    /// strings and a reasoning budget. The context window is checked by the
    /// worker-local runtime on the rendered prompt, as it is for Qwen.
    ///
    /// # Errors
    /// The contract's refusal for a malformed request or an unsupported
    /// control.
    pub fn normalize_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
    ) -> Result<ValidatedChatRequest, ApiError> {
        let options = normalize_requested_options(
            req,
            ServeProfile::lattice_serve(model_id, self.max_context).with_vision_support(false),
        )?;
        if req
            .messages
            .iter()
            .any(|message| matches!(message.content, MessageContent::Parts(_)))
        {
            return Err(ApiError::BadRequest {
                message: "typed content parts are not supported for this model; send message \
                          content as a string"
                    .to_string(),
                code: "unsupported_feature",
            });
        }
        crate::serving_provider::providers::framed_prompt::refuse_reserved_characters(
            &options.messages,
        )?;
        self.adapter.apply_defaults(defaults, options)
    }

    /// The `GenerateConfig` for a request [`Self::normalize_standalone`]
    /// admitted.
    pub fn standalone_generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        self.adapter.generate_config(req)
    }

    /// The `GenerateConfig` for sampling fields a `lattice serve` request
    /// has already prepared, with the checkpoint's stop ids and no thinking
    /// switch, as the Gemma adapter builds it.
    pub fn lattice_generate_config(
        &self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: Option<u64>,
        stop_strings: Vec<String>,
        reasoning_budget: Option<usize>,
        logprobs: Option<usize>,
    ) -> GenerateConfig {
        let mut c = GenerateConfig::default();
        c.max_new_tokens = max_tokens;
        c.temperature = temperature;
        c.top_p = top_p;
        c.seed = seed;
        c.stop_token_ids = self.adapter.stop_token_ids().to_vec();
        c.enable_thinking = false;
        c.stop_strings = stop_strings;
        c.reasoning_budget = reasoning_budget;
        c.logprobs = logprobs;
        c
    }

    /// Render engine chat `messages` with the checkpoint's chat template and
    /// admit the prompt against the context window, returning the prompt and
    /// its token count before any truncation.
    ///
    /// # Errors
    /// `context_length_exceeded` when the prompt plus the generation budget
    /// does not fit, `vision_unsupported` for a message that carries an
    /// image, and `invalid_messages` for content holding U+FDD0 or U+FDD1.
    pub fn render_within_window(
        &self,
        messages: &[ChatMessage],
        cfg: &GenerateConfig,
    ) -> Result<(String, usize), ApiError> {
        let normalized = messages
            .iter()
            .map(|message| {
                if message.image.is_some() {
                    return Err(ApiError::BadRequest {
                        message: "image input requires a vision-capable model".to_owned(),
                        code: "vision_unsupported",
                    });
                }
                let role = match message.role {
                    ChatRole::System => NormalizedChatRole::System,
                    ChatRole::User => NormalizedChatRole::User,
                    ChatRole::Assistant => NormalizedChatRole::Assistant,
                    _ => {
                        return Err(ApiError::BadRequest {
                            message: "unsupported chat role".to_owned(),
                            code: "invalid_role",
                        });
                    }
                };
                Ok(NormalizedChatMessage {
                    role,
                    content: message.content.clone(),
                    image: None,
                })
            })
            .collect::<Result<Vec<_>, ApiError>>()?;
        let prompt = self.adapter.render(&normalized)?;
        let prompt_len = self.tokenize_len(&prompt);
        validate_context_window_with_budget(
            prompt_len,
            cfg.max_new_tokens,
            cfg.reasoning_budget,
            self.max_context,
        )?;
        Ok((prompt, prompt_len))
    }

    /// Prompt token ids for a rendered prompt, refusing one longer than the
    /// context window.
    fn prompt_ids(&self, prompt: &str) -> Result<Vec<u32>, InferenceError> {
        let ids = self.adapter.prompt_token_ids(&self.tokenizer, prompt);
        if ids.len() > self.max_context {
            return Err(InferenceError::Inference(format!(
                "prompt ({} tokens) exceeds the tokenizer window ({})",
                ids.len(),
                self.max_context
            )));
        }
        Ok(ids)
    }

    /// Generate to completion for a prepared `prompt`.
    ///
    /// # Errors
    /// Propagates tokenizer and generation failures.
    pub fn generate(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
    ) -> Result<(GenerateOutput, DriverEvidence), InferenceError> {
        let ids = self.prompt_ids(prompt)?;
        self.generate_ids(&ids, gen_cfg)
    }

    /// Generate for already tokenized `prompt_ids`.
    pub(crate) fn generate_ids(
        &self,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
    ) -> Result<(GenerateOutput, DriverEvidence), InferenceError> {
        lattice_inference::serving_support::gemma_generate_traced(&self.model, prompt_ids, gen_cfg)
            .map(|(output, trace)| (output, DriverEvidence::from_trace(trace)))
    }

    /// Generate for a prepared `prompt`, delivering text deltas to `on_token`
    /// and polling `should_cancel`, as the Qwen CPU streaming entry does.
    ///
    /// # Errors
    /// Propagates tokenizer and generation failures.
    pub fn generate_streaming<F, C>(
        &self,
        prompt: &str,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<(GenerateOutput, DriverEvidence), InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        let ids = self.prompt_ids(prompt)?;
        lattice_inference::serving_support::gemma_generate_streaming_traced(
            &self.model,
            &ids,
            gen_cfg,
            on_token,
            should_cancel,
        )
        .map(|(output, trace)| (output, DriverEvidence::from_trace(trace)))
    }
}

/// [`Qwen35Model::generate`] with the driver's counters kept. It runs the same
/// dispatch `generate` does; only the discarded trace is returned.
#[doc(hidden)]
pub fn qwen_generate_traced(
    model: &Qwen35Model,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> Result<(GenerateOutput, DriverEvidence), InferenceError> {
    lattice_inference::serving_support::qwen_generate_traced(model, prompt, gen_cfg)
        .map(|(output, trace)| (output, DriverEvidence::from_trace(trace)))
}

/// [`Qwen35Model::generate_streaming_with_cancel`] with the driver's counters
/// kept. It runs the same dispatch with a no-op raw-event observer, as that
/// entry does; only the discarded trace is returned.
#[doc(hidden)]
pub fn qwen_generate_streaming_traced<F, C>(
    model: &Qwen35Model,
    prompt: &str,
    gen_cfg: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<(GenerateOutput, DriverEvidence), InferenceError>
where
    F: FnMut(&str) -> bool,
    C: FnMut() -> bool,
{
    lattice_inference::serving_support::qwen_generate_streaming_traced(
        model,
        prompt,
        gen_cfg,
        on_token,
        should_cancel,
    )
    .map(|(output, trace)| (output, DriverEvidence::from_trace(trace)))
}

/// [`qwen_generate_traced`] for a prompt the Qwen route rendered, each template marker framed:
/// the prompt is tokenized with each marker as its token and nothing truncated.
pub(crate) fn qwen_generate_framed_traced(
    model: &Qwen35Model,
    framed: &str,
    gen_cfg: &GenerateConfig,
) -> Result<(GenerateOutput, DriverEvidence), InferenceError> {
    let ids = crate::serving_provider::providers::qwen::framed::framed_prompt_ids(
        model.tokenizer(),
        framed,
    );
    lattice_inference::serving_support::qwen_generate_ids_traced(model, ids, gen_cfg)
        .map(|(output, trace)| (output, DriverEvidence::from_trace(trace)))
}

/// [`qwen_generate_streaming_traced`] for a prompt the Qwen route rendered: each template
/// marker is tokenized as its token and nothing is truncated. The `lattice serve` CPU
/// handler generates through this entry.
pub fn qwen_generate_streaming_framed_traced<F, C>(
    model: &Qwen35Model,
    framed: &str,
    gen_cfg: &GenerateConfig,
    on_token: F,
    should_cancel: C,
) -> Result<(GenerateOutput, DriverEvidence), InferenceError>
where
    F: FnMut(&str) -> bool,
    C: FnMut() -> bool,
{
    let ids = crate::serving_provider::providers::qwen::framed::framed_prompt_ids(
        model.tokenizer(),
        framed,
    );
    lattice_inference::serving_support::qwen_generate_streaming_ids_traced(
        model,
        ids,
        gen_cfg,
        on_token,
        should_cancel,
    )
    .map(|(output, trace)| (output, DriverEvidence::from_trace(trace)))
}

/// The number of ids [`qwen_generate_streaming_framed_traced`] generates from for a prompt
/// the Qwen route rendered. The `lattice serve` CPU handler admits a prompt against the
/// context window by this count.
pub fn qwen_framed_prompt_len(
    tokenizer: &lattice_inference::tokenizer::bpe::BpeTokenizer,
    framed: &str,
) -> usize {
    crate::serving_provider::providers::qwen::framed::framed_prompt_ids(tokenizer, framed).len()
}

#[cfg(test)]
fn fixtures() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../inference/tests/fixtures/gemma4")
}

/// The committed tokenizer and chat-template fixtures around a tiny
/// zero-weight model: real prompt rendering, a model small enough to run
/// without a checkpoint.
#[cfg(test)]
pub(crate) fn tiny_zero_serving() -> GemmaCpuServing {
    let matrix: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("chat_template_matrix.json"))
            .expect("committed chat template matrix"),
    )
    .expect("matrix is JSON");
    let dir = tempfile::tempdir().expect("temp checkpoint dir");
    std::fs::copy(
        fixtures().join("e2b_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config.json");
    std::fs::copy(
        fixtures().join("tokenizer").join("tokenizer_config.json"),
        dir.path().join("tokenizer_config.json"),
    )
    .expect("copy tokenizer_config.json");
    std::fs::write(
        dir.path().join("generation_config.json"),
        matrix["generation_config_json"]
            .as_str()
            .expect("recorded generation_config.json"),
    )
    .expect("write generation_config.json");

    // The vocabulary of the committed tokenizer, so a rendered prompt's ids
    // are all in range.
    let model = lattice_inference::model::gemma4_model::tiny_zero_model_with_vocab(262_144);
    let max_context = model.config().max_position_embeddings;
    let tokenizer = GemmaBpeTokenizer::from_tokenizer_json(
        &fixtures().join("tokenizer").join("tokenizer.json"),
    )
    .expect("committed Gemma tokenizer")
    .with_max_seq_len(max_context);
    let adapter =
        GemmaPromptAdapter::from_model_dir(dir.path(), &tokenizer).expect("Gemma adapter");
    GemmaCpuServing {
        model,
        tokenizer,
        adapter,
        max_context,
    }
}

#[cfg(test)]
mod tests;
