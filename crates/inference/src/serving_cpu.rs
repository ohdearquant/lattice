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

use crate::error::InferenceError;
use crate::forward::metal_qwen35::{ChatMessage, ChatRole};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::model::gemma4_config::Gemma4Config;
use crate::model::gemma4_model::Gemma4Model;
use crate::model::qwen35::Qwen35Model;
use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, MessageContent, NormalizedChatMessage, NormalizedChatRole,
    ServeProfile, ValidatedChatRequest, normalize_requested_options,
    validate_context_window_with_budget,
};
use crate::serve::prepare::{GemmaPromptAdapter, PreparedChatRequest, prepare_gemma_chat_request};
use crate::serve::prompt_adapter::PromptAdapter as _;
use crate::serve::route::DriverEvidence;
use crate::tokenizer::Tokenizer as _;
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
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

    /// Token count of `text` before any truncation.
    ///
    /// Batch tokenization pads to the longest input rather than to the
    /// tokenizer's maximum length, so a single text carries no padding.
    pub fn tokenize_len(&self, text: &str) -> usize {
        self.tokenizer
            .tokenize_batch(&[text])
            .first()
            .map_or(0, |tokenized| tokenized.pre_truncation_len)
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
                temperature: gen_cfg.temperature,
                top_p: gen_cfg.top_p,
                logprobs: gen_cfg.logprobs,
                prompt: prepared.prompt,
                stop_strings: gen_cfg.stop_strings.clone(),
                reasoning_budget: gen_cfg.reasoning_budget,
                enable_thinking: gen_cfg.enable_thinking,
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
        GenerateConfig {
            max_new_tokens: max_tokens,
            temperature,
            top_p,
            seed,
            stop_token_ids: self.adapter.stop_token_ids().to_vec(),
            enable_thinking: false,
            stop_strings,
            reasoning_budget,
            logprobs,
            ..GenerateConfig::default()
        }
    }

    /// Render engine chat `messages` with the checkpoint's chat template and
    /// admit the prompt against the context window, returning the prompt and
    /// its token count before any truncation.
    ///
    /// # Errors
    /// `context_length_exceeded` when the prompt plus the generation budget
    /// does not fit, and `vision_unsupported` for a message that carries an
    /// image.
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
                Ok(NormalizedChatMessage {
                    role: match message.role {
                        ChatRole::System => NormalizedChatRole::System,
                        ChatRole::User => NormalizedChatRole::User,
                        ChatRole::Assistant => NormalizedChatRole::Assistant,
                    },
                    content: message.content.clone(),
                    image: None,
                })
            })
            .collect::<Result<Vec<_>, ApiError>>()?;
        let prompt = self.adapter.render(&normalized);
        let prompt_len = self.tokenize_len(&prompt);
        validate_context_window_with_budget(
            prompt_len,
            cfg.max_new_tokens,
            cfg.reasoning_budget,
            self.max_context,
        )?;
        Ok((prompt, prompt_len))
    }

    /// Prompt token ids for a rendered prompt, refusing one the tokenizer
    /// would truncate.
    fn prompt_ids(&self, prompt: &str) -> Result<Vec<u32>, InferenceError> {
        let tokenized = self
            .tokenizer
            .tokenize_batch(&[prompt])
            .into_iter()
            .next()
            .ok_or_else(|| InferenceError::Inference("prompt was not tokenized".into()))?;
        if tokenized.real_length != tokenized.pre_truncation_len {
            return Err(InferenceError::Inference(format!(
                "prompt ({} tokens) exceeds the tokenizer window ({})",
                tokenized.pre_truncation_len, self.max_context
            )));
        }
        tokenized
            .input_ids
            .get(..tokenized.real_length)
            .map(<[u32]>::to_vec)
            .ok_or_else(|| {
                InferenceError::Inference("tokenized prompt is shorter than reported".into())
            })
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
        self.model
            .generate_with_trace(prompt_ids, gen_cfg)
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
        self.model
            .generate_streaming_via_driver(&ids, gen_cfg, on_token, should_cancel)
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
    model
        .generate_with_trace(prompt, gen_cfg)
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
    model
        .generate_streaming_with_trace(prompt, gen_cfg, on_token, should_cancel, |_| {})
        .map(|(output, trace)| (output, DriverEvidence::from_trace(trace)))
}

#[cfg(test)]
fn fixtures() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
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
    let model = crate::model::gemma4_model::tiny_zero_model_with_vocab(262_144);
    let max_context = model.config.max_position_embeddings;
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
mod tests {
    use super::*;
    use crate::serve::route::ServedRoute;
    use serde_json::json;

    fn serving() -> GemmaCpuServing {
        tiny_zero_serving()
    }

    fn request(extra: serde_json::Value) -> ChatRequest {
        let mut body = json!({
            "model": "served-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        });
        if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            body.extend(extra.clone());
        }
        serde_json::from_value(body).expect("chat request body")
    }

    #[test]
    fn prepare_renders_the_gemma_template_and_builds_the_gemma_config() {
        let serving = serving();
        let (prepared, gen_cfg) = serving
            .prepare(&request(json!({"stream": true})), "served-model", 64, 4096)
            .expect("a plain chat request is admitted");
        assert!(
            prepared
                .prompt
                .starts_with("<bos><|turn>user\nhello<turn|>\n")
        );
        assert!(prepared.prompt.ends_with("<|turn>model\n"));
        assert!(prepared.stream);
        assert!(prepared.messages.is_empty());
        assert_eq!(prepared.max_tokens, 5);
        assert_eq!(gen_cfg.max_new_tokens, 5);
        assert!(
            gen_cfg.stop_token_ids.contains(&106),
            "the checkpoint's end-of-turn id stops the turn: {:?}",
            gen_cfg.stop_token_ids
        );
        assert!(!gen_cfg.enable_thinking);
    }

    #[test]
    fn prepare_refuses_the_controls_gemma_cannot_serve_with_the_contract_codes() {
        let serving = serving();
        for (extra, code) in [
            (json!({"stop": ["x"]}), "unsupported_feature"),
            (json!({"logprobs": true}), "unsupported_feature"),
            (json!({"reasoning_budget": 8}), "unsupported_feature"),
        ] {
            match serving.prepare(&request(extra.clone()), "served-model", 64, 4096) {
                Err(ApiError::BadRequest { code: got, .. }) => {
                    assert_eq!(got, code, "{extra}");
                }
                other => panic!("{extra}: expected a BadRequest refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn prepare_checks_the_prompt_against_the_checkpoint_context_window() {
        let serving = serving();
        let long = "word ".repeat(serving.max_context() + 8);
        let req: ChatRequest = serde_json::from_value(json!({
            "model": "served-model",
            "messages": [{"role": "user", "content": long}],
            "max_tokens": 5,
        }))
        .expect("chat request body");
        match serving.prepare(&req, "served-model", 64, 4096) {
            Err(ApiError::BadRequest { code, .. }) => assert_eq!(code, "context_length_exceeded"),
            other => panic!("expected context_length_exceeded, got {other:?}"),
        }
    }

    #[test]
    fn generation_runs_under_the_shared_driver_and_says_so() {
        let serving = serving();
        let cfg = GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        };
        let (output, evidence) = serving
            .generate_ids(&[2, 3], &cfg)
            .expect("generation over the tiny model must succeed");
        assert_eq!(output.token_ids.len(), 3);
        assert_eq!(
            evidence,
            DriverEvidence {
                opened: 3,
                consumed: 2
            }
        );
        assert!(evidence.went_through_driver());
    }

    fn greedy_cfg() -> GenerateConfig {
        GenerateConfig {
            max_new_tokens: 3,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        }
    }

    #[test]
    fn qwen_traced_generation_reports_the_drivers_own_counters() {
        let model = crate::model::qwen35::test_support::tiny_zero_model();
        let cfg = greedy_cfg();
        let (reference, trace) = model
            .generate_with_trace("a", &cfg)
            .expect("tiny model generates");
        let (output, evidence) =
            qwen_generate_traced(&model, "a", &cfg).expect("traced generation succeeds");
        assert_eq!(output.token_ids, reference.token_ids);
        assert_eq!(evidence, DriverEvidence::from_trace(trace));
        assert!(evidence.went_through_driver(), "{evidence:?}");
        assert_eq!(evidence.consumed + 1, evidence.opened, "{evidence:?}");
    }

    #[test]
    fn qwen_traced_streaming_reports_the_drivers_own_counters() {
        let model = crate::model::qwen35::test_support::tiny_zero_model();
        let cfg = greedy_cfg();
        let reference = model
            .generate("a", &cfg)
            .expect("tiny model generates")
            .token_ids;
        let mut deltas = 0usize;
        let (output, evidence) = qwen_generate_streaming_traced(
            &model,
            "a",
            &cfg,
            |_| {
                deltas += 1;
                true
            },
            || false,
        )
        .expect("traced streaming succeeds");
        assert_eq!(output.token_ids, reference);
        assert!(evidence.went_through_driver(), "{evidence:?}");
        assert_eq!(evidence.consumed + 1, evidence.opened, "{evidence:?}");
        assert!(deltas <= output.generated_tokens);
    }

    fn shared_route_assertion(marker: &str) -> Result<(), &'static str> {
        if marker.contains(" driver=shared ") {
            Ok(())
        } else if marker.contains(" driver=bypassed ") {
            Err("driver=bypassed")
        } else {
            Err("missing driver disposition")
        }
    }

    #[test]
    fn supported_route_marker_rejects_a_test_bypassing_adapter() {
        let model = crate::model::qwen35::test_support::tiny_zero_model();
        let cfg = crate::generation::GenerateConfig {
            max_new_tokens: 2,
            temperature: 0.0,
            ..Default::default()
        };
        let (output, evidence) =
            qwen_generate_traced(&model, "a", &cfg).expect("the supported CPU route generates");
        assert!(output.generated_tokens > 0);
        let supported = ServedRoute::QWEN35_CPU.request_marker(false, evidence);
        assert_eq!(shared_route_assertion(&supported), Ok(()), "{supported}");
        eprintln!("R00-CONTROL supported marker accepted: {supported}");

        struct BypassingAdapter;

        impl BypassingAdapter {
            fn request_marker(&self) -> String {
                ServedRoute::QWEN35_CPU.request_marker(false, DriverEvidence::default())
            }
        }

        let bypassed = BypassingAdapter.request_marker();
        eprintln!("R00-CONTROL bypass adapter rejected as driver=bypassed: {bypassed}");
        assert_eq!(
            shared_route_assertion(&bypassed),
            Err("driver=bypassed"),
            "{bypassed}"
        );
    }
}
