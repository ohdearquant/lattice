use std::sync::Arc;

use crate::serve::ApiError;
use crate::serve::contract::{ChatRequest, GenerationDefaults, ServeProfile, ValidatedChatRequest};
use crate::serve::lora::LoraSelection;
use crate::serve::prepare::{PreparationHandle, PreparedChatRequest};
use crate::serving_cpu_host::SharedCpuHandle;
use crate::serving_worker::{ScoreReply, ScoreRow, WorkerClient, WorkerEvent};
use lattice_inference::forward::metal_qwen35::ChatMessage;
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use tokio::sync::Semaphore;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serve::lora::{AdapterControlError, AdapterControlResult, AdapterIndex};
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::serving_worker::AdapterCommand;
#[cfg(feature = "metal-gpu")]
use lattice_inference::tokenizer::Tokenizer;

/// Request data passed from the unified HTTP handler to its execution host.
#[doc(hidden)]
pub struct LatticeExecutionRequest {
    /// Rendered prompt used by the shared CPU host.
    pub prompt: String,
    /// Sampling and generation options.
    pub config: GenerateConfig,
    /// Normalized messages used by the affine worker.
    pub messages: Option<Vec<ChatMessage>>,
    /// Resolved adapter selection.
    pub selection: Vec<LoraSelection>,
    /// Whether thinking was explicitly enabled in the request.
    pub explicitly_thinking: bool,
}

/// One generation's output, with the text of the reply's reasoning field when the model
/// produced one.
#[doc(hidden)]
#[derive(Debug)]
#[non_exhaustive]
pub struct ServedOutput {
    /// The reply's content text, its token ids and its counts. `generated_tokens` counts every
    /// generated token, the reasoning's included.
    pub output: GenerateOutput,
    /// The reasoning text, `None` when the reply has none; never an empty string.
    pub reasoning: Option<String>,
}

impl ServedOutput {
    /// An output whose reasoning is `reasoning`, where an empty string is no reasoning.
    pub fn new(output: GenerateOutput, reasoning: String) -> Self {
        Self {
            output,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
        }
    }

    /// An output with no reasoning.
    pub fn without_reasoning(output: GenerateOutput) -> Self {
        Self {
            output,
            reasoning: None,
        }
    }
}

#[derive(Clone)]
enum ExecutionHost {
    SharedCpu {
        cpu: SharedCpuHandle,
        generation_admission: Arc<Semaphore>,
    },
    Affine {
        client: WorkerClient,
    },
}

/// Shared request facade for the HTTP serving binaries.
#[doc(hidden)]
#[derive(Clone)]
pub struct ServingHandle {
    host: ExecutionHost,
    #[cfg(feature = "metal-gpu")]
    cli_fallback: Option<CliPreparationFallbackRef>,
}

#[cfg(feature = "metal-gpu")]
#[derive(Clone)]
struct CliPreparationFallbackRef {
    tokenizer: Arc<BpeTokenizer>,
    max_context: usize,
}

impl ServingHandle {
    /// Wrap the shared CPU host and its existing HTTP admission semaphore.
    pub fn shared_cpu(cpu: SharedCpuHandle, generation_admission: Arc<Semaphore>) -> Self {
        Self {
            host: ExecutionHost::SharedCpu {
                cpu,
                generation_admission,
            },
            #[cfg(feature = "metal-gpu")]
            cli_fallback: None,
        }
    }

    /// Run the shared CPU host's generation and scoring on a dedicated pool of
    /// `threads` threads instead of the global rayon pool. A handle on the
    /// affine worker is returned unchanged.
    ///
    /// # Errors
    ///
    /// When the pool's threads cannot be started.
    pub fn with_cpu_threads(
        mut self,
        threads: std::num::NonZeroUsize,
    ) -> Result<Self, rayon::ThreadPoolBuildError> {
        if let ExecutionHost::SharedCpu { cpu, .. } = &mut self.host {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads.get())
                .thread_name(|index| format!("lattice-cpu-{index}"))
                .build()?;
            cpu.set_pool(Arc::new(pool));
        }
        Ok(self)
    }

    #[cfg(test)]
    pub(crate) fn shared_cpu_for_test(&self) -> Option<&SharedCpuHandle> {
        match &self.host {
            ExecutionHost::SharedCpu { cpu, .. } => Some(cpu),
            ExecutionHost::Affine { .. } => None,
        }
    }

    /// Wrap the affine worker for the unified command line server.
    #[cfg(feature = "metal-gpu")]
    pub fn affine_cli(
        client: WorkerClient,
        tokenizer: Arc<BpeTokenizer>,
        max_context: usize,
    ) -> Self {
        Self {
            host: ExecutionHost::Affine { client },
            cli_fallback: Some(CliPreparationFallbackRef {
                tokenizer,
                max_context,
            }),
        }
    }

    /// Wrap the affine worker for the standalone server.
    pub fn affine_standalone(client: WorkerClient) -> Self {
        Self {
            host: ExecutionHost::Affine { client },
            #[cfg(feature = "metal-gpu")]
            cli_fallback: None,
        }
    }

    /// Prepare one request using the unified server's existing preparation path.
    pub fn prepare_lattice_chat_request(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<(PreparedChatRequest, GenerateConfig), ApiError> {
        match &self.host {
            ExecutionHost::SharedCpu { cpu, .. } => {
                let prepared = cpu.prepare(req, model_id, default_max_tokens, max_tokens_cap)?;
                Ok((prepared.prepared, prepared.config))
            }
            ExecutionHost::Affine { client } => {
                let vision_supported = client.supports_vision();
                if let Some(preparation) = client.preparation() {
                    return preparation
                        .prepare_lattice_with_config(
                            req,
                            model_id,
                            default_max_tokens,
                            max_tokens_cap,
                            vision_supported,
                        )
                        .map(|prepared| (prepared.prepared, prepared.config));
                }

                #[cfg(feature = "metal-gpu")]
                {
                    let fallback =
                        self.cli_fallback
                            .as_ref()
                            .ok_or_else(|| ApiError::Internal {
                                message: "inference worker unavailable".to_string(),
                            })?;
                    crate::serve::prepare::prepare_chat_request_with_config(
                        req,
                        model_id,
                        default_max_tokens,
                        max_tokens_cap,
                        vision_supported,
                        |prompt| fallback.tokenizer.tokenize(prompt).pre_truncation_len,
                        || fallback.max_context,
                    )
                    .map(|prepared| (prepared.prepared, prepared.config))
                }
                #[cfg(not(feature = "metal-gpu"))]
                {
                    Err(ApiError::Internal {
                        message: "inference worker unavailable".to_string(),
                    })
                }
            }
        }
    }

    /// Prepare one standalone request using its existing profile and context fallback.
    pub fn prepare_standalone_chat_request(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        max_context: usize,
    ) -> Result<(ValidatedChatRequest, GenerateConfig), ApiError> {
        let vision_supported = self.supports_vision();
        if let Some(preparation) = self.preparation() {
            return preparation.prepare_standalone(req, defaults, model_id, vision_supported);
        }
        crate::serve::prepare::prepare_standalone_chat_request(
            req,
            defaults,
            ServeProfile::lattice_serve(model_id, max_context)
                .with_vision_support(vision_supported),
        )
    }

    /// Refuse standalone controls unsupported by the loaded provider.
    pub fn refuse_standalone_unsupported(&self, req: &ChatRequest) -> Result<(), ApiError> {
        if let Some(preparation) = self.preparation() {
            preparation.refuse_standalone_unsupported(req)?;
        }
        Ok(())
    }

    /// Whether the loaded provider accepts runtime adapters.
    pub fn supports_adapters(&self) -> bool {
        self.preparation()
            .is_none_or(PreparationHandle::supports_adapters)
    }

    /// Tokenizer used to render token log probabilities in the unified server.
    pub fn tokenizer(&self) -> Option<&BpeTokenizer> {
        match &self.host {
            ExecutionHost::SharedCpu { cpu, .. } => cpu.preparation().tokenizer(),
            ExecutionHost::Affine { client } => {
                #[cfg(feature = "metal-gpu")]
                let fallback = self
                    .cli_fallback
                    .as_ref()
                    .map(|fallback| fallback.tokenizer.as_ref());
                #[cfg(not(feature = "metal-gpu"))]
                let fallback = None;
                fallback.or_else(|| client.preparation().and_then(PreparationHandle::tokenizer))
            }
        }
    }

    /// Provider preparation attached to the loaded host, when present.
    pub fn preparation(&self) -> Option<&PreparationHandle> {
        match &self.host {
            ExecutionHost::SharedCpu { cpu, .. } => Some(cpu.preparation()),
            ExecutionHost::Affine { client } => client.preparation(),
        }
    }

    /// Read live image support for the current request.
    pub fn supports_vision(&self) -> bool {
        match &self.host {
            ExecutionHost::SharedCpu { .. } => false,
            ExecutionHost::Affine { client } => client.supports_vision(),
        }
    }

    /// Validate a resolved adapter selection against the loaded worker.
    pub fn validate_lora(&self, selection: &[LoraSelection]) -> Result<(), ApiError> {
        match &self.host {
            ExecutionHost::SharedCpu { .. } => {
                Err(crate::serve::prepare::lora_unsupported_backend())
            }
            ExecutionHost::Affine { client } => client.validate_lora(selection),
        }
    }

    /// Current number of outstanding worker or CPU generation permits.
    pub fn available_permits(&self) -> usize {
        match &self.host {
            ExecutionHost::SharedCpu {
                generation_admission,
                ..
            } => generation_admission.available_permits(),
            ExecutionHost::Affine { client } => client.available_permits(),
        }
    }

    /// Submit one standalone request and return the raw worker event stream.
    pub fn submit_standalone(
        &self,
        messages: Vec<ChatMessage>,
        config: GenerateConfig,
        cancel: tokio::sync::watch::Receiver<bool>,
        selection: Vec<LoraSelection>,
        stream: bool,
        explicitly_thinking: bool,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<WorkerEvent>, ApiError> {
        match &self.host {
            ExecutionHost::SharedCpu { .. } => Err(ApiError::Internal {
                message: "inference worker unavailable".to_string(),
            }),
            ExecutionHost::Affine { client } => client.submit_with_lora_mode_and_thinking(
                messages,
                config,
                cancel,
                selection,
                stream,
                explicitly_thinking,
            ),
        }
    }

    /// Run one non-streaming unified-server request.
    pub async fn complete_lattice(
        &self,
        request: LatticeExecutionRequest,
    ) -> Result<ServedOutput, ApiError> {
        match &self.host {
            ExecutionHost::SharedCpu {
                cpu,
                generation_admission,
            } => {
                let permit = acquire_cpu_generation_slot(generation_admission)?;
                let handle = cpu.spawn_completion(request.prompt, request.config);
                tokio::spawn(async move {
                    let _permit = permit;
                    handle.await
                })
                .await
                .map_err(|error| {
                    eprintln!("task join error: {error}");
                    ApiError::Internal {
                        message: "inference failed".to_string(),
                    }
                })?
                .map_err(|error| {
                    eprintln!("task join error: {error}");
                    ApiError::Internal {
                        message: "inference failed".to_string(),
                    }
                })?
                .map_err(|error| {
                    eprintln!("generation error: {error}");
                    ApiError::Internal {
                        message: "inference failed".to_string(),
                    }
                })
            }
            ExecutionHost::Affine { client } => {
                let (_guard, cancel) = crate::serve::cancel_pair();
                let messages = request.messages.ok_or_else(|| ApiError::Internal {
                    message: "inference worker unavailable".to_string(),
                })?;
                let mut receiver = client.submit_with_lora_mode_and_thinking(
                    messages,
                    request.config,
                    cancel,
                    request.selection,
                    false,
                    request.explicitly_thinking,
                )?;
                drain_worker_events(&mut receiver, None, |_delta| true)
                    .await
                    .map(ServedOutput::without_reasoning)
                    .map_err(map_worker_generation_error)
            }
        }
    }

    /// Start one streaming unified-server request and preserve its admission boundary.
    ///
    /// `on_delta` receives the reply's content and `on_reasoning` its reasoning, both from the
    /// generation thread, in generated order. Either returning false stops generation. The
    /// affine worker produces no reasoning field.
    pub async fn stream_lattice<F, R, G>(
        &self,
        request: LatticeExecutionRequest,
        cancel: tokio::sync::watch::Receiver<bool>,
        on_delta: F,
        on_reasoning: R,
        finish: G,
    ) -> Result<(), ApiError>
    where
        F: FnMut(&str) -> bool + Send + 'static,
        R: FnMut(&str) -> bool + Send + 'static,
        G: FnOnce(Result<GenerateOutput, ApiError>) + Send + 'static,
    {
        match &self.host {
            ExecutionHost::SharedCpu {
                cpu,
                generation_admission,
            } => {
                let permit = acquire_cpu_generation_slot(generation_admission)?;
                cpu.spawn_streaming(
                    request.prompt,
                    request.config,
                    cancel,
                    on_delta,
                    on_reasoning,
                    move |result| {
                        let _permit = permit;
                        match result {
                            Ok(output) => finish(Ok(output)),
                            Err(error) => {
                                eprintln!("generation error (streaming): {error}");
                                finish(Err(ApiError::Internal { message: error }));
                            }
                        }
                    },
                );
                Ok(())
            }
            ExecutionHost::Affine { client } => {
                drop(on_reasoning);
                let has_selection = !request.selection.is_empty();
                let messages = request.messages.ok_or_else(|| ApiError::Internal {
                    message: "inference worker unavailable".to_string(),
                })?;
                let mut receiver = client.submit_with_lora_mode_and_thinking(
                    messages,
                    request.config,
                    cancel,
                    request.selection,
                    true,
                    request.explicitly_thinking,
                )?;
                let first = if !has_selection {
                    None
                } else {
                    match receiver.recv().await {
                        Some(WorkerEvent::Rejected(error)) => return Err(error),
                        Some(
                            WorkerEvent::Failed(message) | WorkerEvent::ConstraintBlocked(message),
                        ) => return Err(ApiError::Internal { message }),
                        Some(event) => Some(event),
                        None => {
                            return Err(ApiError::Internal {
                                message: "inference worker unavailable".into(),
                            });
                        }
                    }
                };
                tokio::spawn(async move {
                    let result = drain_worker_events(&mut receiver, first, on_delta).await;
                    if let Err(error) = &result {
                        eprintln!("generation error (streaming, metal): {error:?}");
                    }
                    finish(result);
                });
                Ok(())
            }
        }
    }
}

/// Preserve the unified server's synchronous CPU generation admission error.
fn acquire_cpu_generation_slot(
    admission: &Arc<Semaphore>,
) -> Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
    admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::ServiceUnavailable {
            message:
                "too many concurrent CPU chat generations on /v1/chat/completions; retry shortly"
                    .to_string(),
        })
}

/// Preserve the unified server's worker-generation error translation.
#[doc(hidden)]
pub fn map_worker_generation_error(error: ApiError) -> ApiError {
    match error {
        error @ (ApiError::ServiceUnavailable { .. } | ApiError::BadRequest { .. }) => error,
        other => {
            eprintln!("generation error (metal): {other:?}");
            ApiError::Internal {
                message: "inference failed".to_string(),
            }
        }
    }
}

async fn drain_worker_events(
    receiver: &mut tokio::sync::mpsc::UnboundedReceiver<WorkerEvent>,
    mut first: Option<WorkerEvent>,
    mut on_delta: impl FnMut(&str) -> bool + Send + 'static,
) -> Result<GenerateOutput, ApiError> {
    let mut deliver_deltas = true;
    loop {
        let event = match first.take() {
            Some(event) => Some(event),
            None => receiver.recv().await,
        };
        let Some(event) = event else {
            return Err(ApiError::Internal {
                message: "inference worker unavailable".to_string(),
            });
        };
        match normalize_cancelled(event) {
            WorkerEvent::Delta(delta) => {
                if deliver_deltas && !on_delta(&delta) {
                    deliver_deltas = false;
                }
            }
            WorkerEvent::Complete(output) => return Ok(output),
            WorkerEvent::Rejected(error) => return Err(error),
            WorkerEvent::Failed(message) | WorkerEvent::ConstraintBlocked(message) => {
                return Err(ApiError::Internal {
                    message: format!("generation failed: {message}"),
                });
            }
            WorkerEvent::Cancelled => {
                unreachable!("normalize_cancelled already rewrote Cancelled into Complete")
            }
        }
    }
}

fn normalize_cancelled(event: WorkerEvent) -> WorkerEvent {
    match event {
        WorkerEvent::Cancelled => WorkerEvent::Complete(GenerateOutput {
            text: String::new(),
            token_ids: vec![],
            prompt_tokens: 0,
            generated_tokens: 0,
            stopped: false,
            stop_reason: Some(lattice_inference::StopReason::Interrupt),
            token_logprobs: vec![],
        }),
        other => other,
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
impl ServingHandle {
    /// Return the worker client or the refusal selected by the caller surface.
    pub fn adapter_client_or(&self, refusal: ApiError) -> Result<&WorkerClient, ApiError> {
        match &self.host {
            ExecutionHost::SharedCpu { .. } => Err(refusal),
            ExecutionHost::Affine { client } => Ok(client),
        }
    }

    /// Submit one adapter control command through the affine worker.
    pub fn submit_adapter_command(
        &self,
        command: AdapterCommand,
    ) -> Result<
        tokio::sync::oneshot::Receiver<Result<AdapterControlResult, AdapterControlError>>,
        ApiError,
    > {
        self.adapter_client_or(crate::serve::prepare::lora_unsupported_backend())?
            .submit_adapter_command(command)
    }

    /// Snapshot adapter residency through the affine worker.
    pub fn adapter_index(&self) -> Result<AdapterIndex, ApiError> {
        Ok(self
            .adapter_client_or(crate::serve::prepare::lora_unsupported_backend())?
            .adapter_index())
    }
}

impl ServingHandle {
    /// Submit option-scoring rows: through the affine worker (see
    /// [`WorkerClient::submit_score`]), or on a blocking task for the shared
    /// CPU host, which admits a scoring request against the same cap as its
    /// generations and answers 503 when the cap is full. `time_limit` counts
    /// from when the backend starts scoring the rows; `None` is no limit. Must
    /// be called from within a Tokio runtime.
    pub fn submit_score(
        &self,
        rows: Vec<ScoreRow>,
        time_limit: Option<std::time::Duration>,
    ) -> Result<tokio::sync::oneshot::Receiver<ScoreReply>, ApiError> {
        match &self.host {
            ExecutionHost::SharedCpu {
                cpu,
                generation_admission,
            } => {
                let permit = Arc::clone(generation_admission)
                    .try_acquire_owned()
                    .map_err(|_| ApiError::ServiceUnavailable {
                        message: "too many outstanding CPU requests; retry shortly".to_string(),
                    })?;
                Ok(cpu.spawn_score(rows, time_limit, permit))
            }
            ExecutionHost::Affine { client } => client.submit_score(rows, time_limit),
        }
    }
}
