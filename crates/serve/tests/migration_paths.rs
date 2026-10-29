//! Compile-time coverage of the public serving paths after extraction.

#![allow(unused_imports)]

use lattice_serve::serve as _;
use lattice_serve::serve::ApiError as _;
use lattice_serve::serve::BASELINE_CANNED_COMPLETION_TOKENS as _;
use lattice_serve::serve::BASELINE_CANNED_PROMPT_TOKENS as _;
use lattice_serve::serve::BASELINE_CANNED_TEXT as _;
use lattice_serve::serve::BASELINE_SSE_EVENTS as _;
use lattice_serve::serve::Binary as _;
use lattice_serve::serve::CHAT_COMPLETIONS_PARITY_CASES as _;
use lattice_serve::serve::CancelOnDrop as _;
use lattice_serve::serve::CaseBody as _;
use lattice_serve::serve::EventExpectation as _;
use lattice_serve::serve::ExpectedObservation as _;
use lattice_serve::serve::ExpectedResponse as _;
use lattice_serve::serve::FieldExpectation as _;
use lattice_serve::serve::GenerateConfigSnapshot as _;
use lattice_serve::serve::OBSERVATION_GOLDEN_USER_HI_THERE_CHATML as _;
use lattice_serve::serve::OVERFLOW_PARITY_CONTEXT_WINDOW as _;
use lattice_serve::serve::OVERFLOW_PARITY_MAX_TOKENS as _;
use lattice_serve::serve::OVERFLOW_PARITY_MAX_TOKENS_CAP as _;
use lattice_serve::serve::OVERFLOW_PARITY_REQUEST_BODY as _;
use lattice_serve::serve::ParityCase as _;
use lattice_serve::serve::ProductionAdapterObservation as _;
use lattice_serve::serve::REQUEST_BODY_LIMIT_BYTES as _;
use lattice_serve::serve::Scalar as _;
use lattice_serve::serve::assert_observation_matches as _;
use lattice_serve::serve::cancel_pair as _;
use lattice_serve::serve::check_sse_events as _;
use lattice_serve::serve::contract as _;
use lattice_serve::serve::contract::ChatRequest as _;
use lattice_serve::serve::contract::ContentPart as _;
use lattice_serve::serve::contract::GenerationDefaults as _;
use lattice_serve::serve::contract::ImageUrl as _;
use lattice_serve::serve::contract::JsonSchemaFormat as _;
use lattice_serve::serve::contract::MAX_CUMULATIVE_CONTENT_BYTES as _;
use lattice_serve::serve::contract::MAX_CUMULATIVE_STOP_BYTES as _;
use lattice_serve::serve::contract::MAX_DECODED_IMAGE_BYTES as _;
use lattice_serve::serve::contract::MAX_MESSAGE_COUNT as _;
use lattice_serve::serve::contract::MAX_STOP_STRING_BYTES as _;
use lattice_serve::serve::contract::Message as _;
use lattice_serve::serve::contract::MessageContent as _;
use lattice_serve::serve::contract::NormalizedChatImage as _;
use lattice_serve::serve::contract::NormalizedChatMessage as _;
use lattice_serve::serve::contract::NormalizedChatRole as _;
use lattice_serve::serve::contract::RequestedChatOptions as _;
use lattice_serve::serve::contract::ResponseFormat as _;
use lattice_serve::serve::contract::ServeProfile as _;
use lattice_serve::serve::contract::ValidatedChatRequest as _;
use lattice_serve::serve::contract::decode_inline_image as _;
use lattice_serve::serve::contract::is_message_flood_error as _;
use lattice_serve::serve::contract::message_flood_text as _;
use lattice_serve::serve::contract::normalize_messages as _;
use lattice_serve::serve::contract::normalize_request as _;
use lattice_serve::serve::contract::normalize_request_with_context as _;
use lattice_serve::serve::contract::normalize_request_with_context_and_budget as _;
use lattice_serve::serve::contract::normalize_requested_options as _;
use lattice_serve::serve::contract::parse_stop_strings as _;
use lattice_serve::serve::contract::validate_context_window as _;
use lattice_serve::serve::contract::validate_context_window_with_budget as _;
use lattice_serve::serve::contract::validate_temperature as _;
use lattice_serve::serve::contract::validate_top_p as _;
use lattice_serve::serve::embeddings as _;
use lattice_serve::serve::embeddings::BERT_ENCODER_LOADER as _;
use lattice_serve::serve::embeddings::EmbeddingDatum as _;
use lattice_serve::serve::embeddings::EmbeddingInputItem as _;
use lattice_serve::serve::embeddings::EmbeddingModel as _;
use lattice_serve::serve::embeddings::EmbeddingsInput as _;
use lattice_serve::serve::embeddings::EmbeddingsRequest as _;
use lattice_serve::serve::embeddings::EmbeddingsResponse as _;
use lattice_serve::serve::embeddings::EmbeddingsUsage as _;
use lattice_serve::serve::embeddings::MAX_EMBEDDING_INPUT_COUNT as _;
use lattice_serve::serve::embeddings::MAX_EMBEDDING_TOTAL_TOKENS as _;
use lattice_serve::serve::embeddings::MAX_EMBEDDINGS_BATCH_SIZE as _;
use lattice_serve::serve::embeddings::MAX_EMBEDDINGS_TOTAL_TOKENS as _;
use lattice_serve::serve::embeddings::NormalizedEmbeddingItem as _;
use lattice_serve::serve::embeddings::QWEN35_F16_DECODER_LOADER as _;
use lattice_serve::serve::embeddings::TextEmbeddingsInput as _;
use lattice_serve::serve::embeddings::TextEmbeddingsRequest as _;
use lattice_serve::serve::embeddings::build_embeddings_response as _;
use lattice_serve::serve::embeddings::check_embeddings_item_fits_window as _;
use lattice_serve::serve::embeddings::check_embeddings_model as _;
use lattice_serve::serve::embeddings::check_embeddings_total_tokens as _;
use lattice_serve::serve::embeddings::check_embeddings_unsupported_fields as _;
use lattice_serve::serve::embeddings::check_pooled_embedding_total_tokens as _;
use lattice_serve::serve::embeddings::check_pooled_service_total_tokens as _;
use lattice_serve::serve::embeddings::embed_items as _;
use lattice_serve::serve::embeddings::embed_pooled_items as _;
use lattice_serve::serve::embeddings::map_embedding_error as _;
use lattice_serve::serve::embeddings::normalize_embedding_items as _;
use lattice_serve::serve::embeddings::parse_embeddings_input as _;
use lattice_serve::serve::embeddings::parse_pooling as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serve::embeddings::test_support as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serve::embeddings::test_support::tiny_embedding_handle as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serve::embeddings::test_support::tiny_embedding_model as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serve::embeddings::test_support::tiny_png_data_uri as _;
use lattice_serve::serve::finish_reason as _;
use lattice_serve::serve::format_normalized_chat_template as _;
use lattice_serve::serve::into_engine_chat_messages as _;
use lattice_serve::serve::lora as _;
use lattice_serve::serve::lora::AdapterControlError as _;
use lattice_serve::serve::lora::AdapterControlReceiver as _;
use lattice_serve::serve::lora::AdapterControlResult as _;
use lattice_serve::serve::lora::AdapterIndex as _;
use lattice_serve::serve::lora::AdapterMetadata as _;
use lattice_serve::serve::lora::DEFAULT_MAX_RESIDENT_ADAPTER_BYTES as _;
use lattice_serve::serve::lora::DEFAULT_MAX_RESIDENT_ADAPTERS as _;
use lattice_serve::serve::lora::LORA_ROUTES as _;
use lattice_serve::serve::lora::LoraSelection as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::lora::PreparedAdapter as _;
use lattice_serve::serve::lora::ROUTER_FEEDBACK_ROUTES as _;
use lattice_serve::serve::lora::RequestedAdapters as _;
use lattice_serve::serve::lora::ResidencyLimits as _;
use lattice_serve::serve::lora::adapter_failure_code as _;
use lattice_serve::serve::lora::list_request as _;
use lattice_serve::serve::lora::load_request as _;
use lattice_serve::serve::lora::load_success_body as _;
use lattice_serve::serve::lora::lora_list_body as _;
use lattice_serve::serve::lora::parse_lora_load as _;
use lattice_serve::serve::lora::parse_lora_load_path as _;
use lattice_serve::serve::lora::parse_lora_unload as _;
use lattice_serve::serve::lora::parse_resident_limit as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::lora::prepare_adapter_load as _;
use lattice_serve::serve::lora::requested_adapters as _;
use lattice_serve::serve::lora::unknown_adapter as _;
use lattice_serve::serve::lora::unload_request as _;
use lattice_serve::serve::lora::unload_success_body as _;
use lattice_serve::serve::lora::validate_scales as _;
use lattice_serve::serve::lora::worker_unavailable as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::AdapterCommand as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::ContextWindowPolicy as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::ControlRequest as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::DEFAULT_MAX_PENDING_JOBS as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::MetalWorker as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::MetalWorkerClient as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::MetalWorkerOwner as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::StartupError as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::VisionRuntime as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::WorkerEvent as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::WorkerJob as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::WorkerMessage as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serve::metal_worker::WorkerMetadata as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "bench-internals")]
use lattice_serve::serve::metal_worker::bench_support as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "bench-internals")]
use lattice_serve::serve::metal_worker::bench_support::check_prompt_fits_window as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "bench-internals")]
use lattice_serve::serve::metal_worker::bench_support::spawn_with_measurement_lock as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "test-utils")]
use lattice_serve::serve::metal_worker::spawn_fake as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "test-utils")]
use lattice_serve::serve::metal_worker::spawn_fake_with_cap as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "test-utils")]
use lattice_serve::serve::metal_worker::spawn_fake_with_vision as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "test-utils")]
use lattice_serve::serve::metal_worker::test_client_and_jobs as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "test-utils")]
use lattice_serve::serve::metal_worker::test_client_and_jobs_with_adapters as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[cfg(feature = "test-utils")]
use lattice_serve::serve::metal_worker::test_client_and_jobs_with_cap as _;
use lattice_serve::serve::metrics as _;
use lattice_serve::serve::metrics::ServeMetrics as _;
use lattice_serve::serve::models_list_body as _;
use lattice_serve::serve::prepare as _;
use lattice_serve::serve::prepare::GemmaPromptAdapter as _;
use lattice_serve::serve::prepare::PreparationHandle as _;
use lattice_serve::serve::prepare::PreparedChatRequest as _;
use lattice_serve::serve::prepare::PreparedGemmaChatRequest as _;
use lattice_serve::serve::prepare::build_cfg as _;
use lattice_serve::serve::prepare::explicitly_enabled_thinking as _;
use lattice_serve::serve::prepare::lattice_gen_cfg as _;
use lattice_serve::serve::prepare::lattice_gen_cfg_with_thinking as _;
use lattice_serve::serve::prepare::lora_unsupported_backend as _;
use lattice_serve::serve::prepare::prepare_chat_request as _;
use lattice_serve::serve::prepare::prepare_chat_request_with_config as _;
use lattice_serve::serve::prepare::prepare_gemma_chat_request as _;
use lattice_serve::serve::prepare::prepare_standalone_chat_request as _;
use lattice_serve::serve::reject_zero_max_tokens as _;
use lattice_serve::serve::require_json_content_type as _;
use lattice_serve::serve::root_body as _;
use lattice_serve::serve::route as _;
use lattice_serve::serve::route::DriverEvidence as _;
use lattice_serve::serve::route::GEMMA_METAL_UNSUPPORTED_CODE as _;
use lattice_serve::serve::route::RouteRefusal as _;
use lattice_serve::serve::route::ServedBackend as _;
use lattice_serve::serve::route::ServedRoute as _;
use lattice_serve::serve::route::select_route as _;
use lattice_serve::serve::route::select_standalone_route as _;
use lattice_serve::serve::routing as _;
use lattice_serve::serve::routing::EmbedderIdentity as _;
use lattice_serve::serve::routing::PromptSource as _;
use lattice_serve::serve::routing::Routability as _;
use lattice_serve::serve::routing::RoutedRequest as _;
use lattice_serve::serve::routing::ServedRouter as _;
#[cfg(feature = "mixture")]
use lattice_serve::serve::routing::ServingRouter as _;
#[cfg(not(feature = "mixture"))]
use lattice_serve::serve::routing::ServingRouter as _;
use lattice_serve::serve::routing::check_representation as _;
use lattice_serve::serve::routing::check_service_representation as _;
use lattice_serve::serve::routing::context_text as _;
use lattice_serve::serve::routing::routability as _;
use lattice_serve::serve::routing::trained_pooling as _;
use lattice_serve::serve::routing::trained_prompt_source as _;
use lattice_serve::serve::serve_until_shutdown as _;
use lattice_serve::serve::systemone as _;
use lattice_serve::serve::systemone::AnswerValue as _;
use lattice_serve::serve::systemone::AssembledAnswers as _;
use lattice_serve::serve::systemone::AssembledPrompt as _;
use lattice_serve::serve::systemone::DeciderConfig as _;
use lattice_serve::serve::systemone::IndexEntry as _;
use lattice_serve::serve::systemone::PlannedRow as _;
use lattice_serve::serve::systemone::PreparedSystemOne as _;
use lattice_serve::serve::systemone::PyValue as _;
use lattice_serve::serve::systemone::QuestionKind as _;
use lattice_serve::serve::systemone::RenderedQuestion as _;
use lattice_serve::serve::systemone::RowKind as _;
use lattice_serve::serve::systemone::RowPrompt as _;
use lattice_serve::serve::systemone::SystemOneError as _;
use lattice_serve::serve::systemone::SystemOneRequest as _;
use lattice_serve::serve::systemone::SystemOneResponse as _;
use lattice_serve::serve::systemone::answer as _;
use lattice_serve::serve::systemone::answer::AnswerValue as _;
use lattice_serve::serve::systemone::answer::AssembledAnswers as _;
use lattice_serve::serve::systemone::answer::assemble as _;
use lattice_serve::serve::systemone::answer::certainty as _;
use lattice_serve::serve::systemone::answer::combine_isolated as _;
use lattice_serve::serve::systemone::answer::format_answer as _;
use lattice_serve::serve::systemone::assemble as _;
use lattice_serve::serve::systemone::assemble_prompt as _;
use lattice_serve::serve::systemone::config as _;
use lattice_serve::serve::systemone::config::DeciderConfig as _;
use lattice_serve::serve::systemone::config::parse_decider_config as _;
use lattice_serve::serve::systemone::json as _;
use lattice_serve::serve::systemone::json::PyValue as _;
use lattice_serve::serve::systemone::json::parse as _;
use lattice_serve::serve::systemone::json::python_float_repr as _;
use lattice_serve::serve::systemone::json::render as _;
use lattice_serve::serve::systemone::load_decider_config as _;
use lattice_serve::serve::systemone::parse_decider_config as _;
use lattice_serve::serve::systemone::parse_request as _;
use lattice_serve::serve::systemone::plan_rows as _;
use lattice_serve::serve::systemone::prepare_request as _;
use lattice_serve::serve::systemone::prompt as _;
use lattice_serve::serve::systemone::prompt::AssembledPrompt as _;
use lattice_serve::serve::systemone::prompt::RowPrompt as _;
use lattice_serve::serve::systemone::prompt::assemble_prompt as _;
use lattice_serve::serve::systemone::prompt::encode as _;
use lattice_serve::serve::systemone::render_question as _;
use lattice_serve::serve::systemone::render_state as _;
use lattice_serve::serve::systemone::request as _;
use lattice_serve::serve::systemone::request::ANNOTATE_MIN as _;
use lattice_serve::serve::systemone::request::IndexEntry as _;
use lattice_serve::serve::systemone::request::LETTER_LABELS as _;
use lattice_serve::serve::systemone::request::MAX_CHOICE as _;
use lattice_serve::serve::systemone::request::MAX_LEVELS as _;
use lattice_serve::serve::systemone::request::PlannedRow as _;
use lattice_serve::serve::systemone::request::QuestionKind as _;
use lattice_serve::serve::systemone::request::RenderedQuestion as _;
use lattice_serve::serve::systemone::request::RowKind as _;
use lattice_serve::serve::systemone::request::SystemOneRequest as _;
use lattice_serve::serve::systemone::request::annotate_indices as _;
use lattice_serve::serve::systemone::request::parse_request as _;
use lattice_serve::serve::systemone::request::plan_rows as _;
use lattice_serve::serve::systemone::request::render_question as _;
use lattice_serve::serve::systemone::request::render_state as _;
use lattice_serve::serve::systemone::request::txt as _;
use lattice_serve::serve::systemone::resolve_labels as _;
use lattice_serve::serve::systemone::text as _;
use lattice_serve::serve::systemone::text::is_zero as _;
use lattice_serve::serve::systemone::text::python_is_digit as _;
use lattice_serve::serve::systemone::text::python_is_space as _;
use lattice_serve::serve::systemone::text::python_repr_of as _;
use lattice_serve::serve::systemone::text::python_repr_str as _;
use lattice_serve::serve::systemone::text::python_round as _;
use lattice_serve::serve::systemone::text::python_str_of_list_criterion as _;
use lattice_serve::serve::systemone::text::strip_level_number as _;
use lattice_serve::serving_cpu as _;
use lattice_serve::serving_cpu::GemmaCpuServing as _;
use lattice_serve::serving_cpu::qwen_generate_streaming_traced as _;
use lattice_serve::serving_cpu::qwen_generate_traced as _;
use lattice_serve::serving_cpu_host as _;
use lattice_serve::serving_cpu_host::SharedCpuHandle as _;
use lattice_serve::serving_embeddings as _;
use lattice_serve::serving_embeddings::EmbeddingRepresentation as _;
use lattice_serve::serving_embeddings::PooledEmbeddingHandle as _;
use lattice_serve::serving_embeddings::TextEmbeddingHandle as _;
use lattice_serve::serving_embeddings::TokenCounts as _;
use lattice_serve::serving_factory as _;
use lattice_serve::serving_factory::ServingFactory as _;
use lattice_serve::serving_host as _;
use lattice_serve::serving_host::LatticeExecutionRequest as _;
use lattice_serve::serving_host::ServingHandle as _;
use lattice_serve::serving_host::map_worker_generation_error as _;
use lattice_serve::serving_provider as _;
use lattice_serve::serving_provider::CheckpointEvidence as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::DEFAULT_SERVE_MAX_CONTEXT as _;
use lattice_serve::serving_provider::EmbeddingSelectionError as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::FALLBACK_MODEL_MAX_CONTEXT as _;
use lattice_serve::serving_provider::ImplicitPooledEmbeddingPlan as _;
use lattice_serve::serving_provider::LatticeOpenOptions as _;
use lattice_serve::serving_provider::PooledEmbeddingPlan as _;
use lattice_serve::serving_provider::SelectedProvider as _;
use lattice_serve::serving_provider::SelectionError as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::ServeContextSource as _;
use lattice_serve::serving_provider::ServingEntry as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::StandaloneLoadOptions as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::StandaloneLoadPlan as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::StandaloneOptionPresence as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::StandalonePostReady as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::StandaloneQwenLoader as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::StandaloneStartupDescription as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::StartedStandalone as _;
use lattice_serve::serving_provider::TextEmbeddingPlan as _;
use lattice_serve::serving_provider::inspect_checkpoint as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::model_context_from_config as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::parse_max_context as _;
use lattice_serve::serving_provider::plan_pooled_embeddings as _;
use lattice_serve::serving_provider::plan_text_embeddings as _;
use lattice_serve::serving_provider::providers as _;
use lattice_serve::serving_provider::providers::bert as _;
use lattice_serve::serving_provider::providers::gemma as _;
use lattice_serve::serving_provider::providers::gemma::load_cpu as _;
use lattice_serve::serving_provider::providers::qwen as _;
use lattice_serve::serving_provider::providers::qwen::load_cpu as _;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use lattice_serve::serving_provider::resolve_serve_context as _;
use lattice_serve::serving_provider::select as _;
use lattice_serve::serving_worker as _;
use lattice_serve::serving_worker::AdapterCommand as _;
use lattice_serve::serving_worker::AffineWorker as _;
use lattice_serve::serving_worker::ContextWindowPolicy as _;
use lattice_serve::serving_worker::ControlRequest as _;
use lattice_serve::serving_worker::DEFAULT_MAX_PENDING_JOBS as _;
use lattice_serve::serving_worker::StartupError as _;
use lattice_serve::serving_worker::WorkerClient as _;
use lattice_serve::serving_worker::WorkerEvent as _;
use lattice_serve::serving_worker::WorkerJob as _;
use lattice_serve::serving_worker::WorkerMessage as _;
use lattice_serve::serving_worker::WorkerMetadata as _;
use lattice_serve::serving_worker::WorkerOwner as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serving_worker::spawn_fake as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serving_worker::spawn_fake_with_cap as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serving_worker::spawn_fake_with_vision as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serving_worker::test_client_and_jobs as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serving_worker::test_client_and_jobs_with_adapters as _;
#[cfg(feature = "test-utils")]
use lattice_serve::serving_worker::test_client_and_jobs_with_cap as _;
