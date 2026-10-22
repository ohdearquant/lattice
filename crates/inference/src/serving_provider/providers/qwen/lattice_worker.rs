use crate::forward::metal_qwen35::MetalQwen35State;
use crate::model::qwen35_config::Qwen35Config;
use crate::serve::lora::ResidencyLimits;
use crate::serve::metal_worker::{ContextWindowPolicy, StartupError};
use crate::serving_factory::ServingFactory;
use crate::serving_host::ServingHandle;
use crate::serving_provider::LatticeOpenOptions;
use crate::serving_provider::providers::qwen::vision::VisionRuntime;
use crate::serving_runtime_contract::WorkerMetadata;
use crate::serving_worker::AffineWorker;
use crate::tokenizer::bpe::BpeTokenizer;
use std::path::PathBuf;
use std::sync::Arc;

pub(super) fn open(
    model_dir: PathBuf,
    options: LatticeOpenOptions,
) -> Result<ServingHandle, String> {
    let tokenizer_path = options
        .tokenizer_dir
        .as_deref()
        .unwrap_or(&model_dir)
        .join("tokenizer.json");
    let tokenizer = Arc::new(
        BpeTokenizer::from_tokenizer_json(&tokenizer_path)
            .map_err(|e| format!("tokenizer load failed ({}): {e}", tokenizer_path.display()))?,
    );
    // The shared worker loader needs its own owned tokenizer (it renders and
    // tokenizes the prompt once per request for the KV-window check,
    // `check_prompt_fits_window`). Clone the `Arc` above rather than re-read
    // from disk, so `tokenizer.json` is parsed exactly once. This is a
    // one-time, startup-only in-memory clone, not a per-request cost.
    let tokenizer_for_worker = (*tokenizer).clone();
    // Preserve this binary's existing behavior: the context window is this
    // fixed cap, not re-derived from `state.max_context()` after loading
    // (unlike `providers::qwen::worker::load_model`). Pass it to the shared
    // worker's `WorkerMetadata` too, so its internal
    // `check_prompt_fits_window` invariant agrees with the HTTP-layer
    // `check_context_window` preflight that already runs first in
    // `prepare_chat_request`.
    let context_window = options.context_window;
    let vision_config = Qwen35Config::from_model_dir(&model_dir)
        .map_err(|e| format!("config.json load failed: {e}"))?;
    let mut vision_runtime = VisionRuntime::from_model_config(model_dir.clone(), &vision_config);
    if options.preload_vision {
        // Eager-load now, on this startup thread, before the Metal worker
        // thread spawns. Lazy stays the default (see the `--preload-vision`
        // help text); a failure here must not abort startup. Warn and fall
        // back to the normal lazy load on the first image request, exactly
        // as if this flag were absent.
        if let Err(err) = vision_runtime.preload() {
            eprintln!(
                "Warning: --preload-vision failed, falling back to lazy vision loading: {err}"
            );
        }
    }
    let model_dir_for_loader = model_dir.clone();
    let tokenizer_path_for_loader = tokenizer_path.clone();
    let (owner, client, _meta, _preparation) = AffineWorker::spawn_with_vision(
        ServingFactory::qwen_metal(
            move || {
                let cfg = Qwen35Config::from_model_dir(&model_dir_for_loader)
                    .map_err(|e| format!("config.json load failed: {e}"))?;
                let state = MetalQwen35State::from_q4_dir(
                    &model_dir_for_loader,
                    &tokenizer_path_for_loader,
                    &cfg,
                    context_window,
                )
                .map_err(|e| format!("Q4 model load failed: {e}"))?;
                Ok((
                    state,
                    tokenizer_for_worker,
                    WorkerMetadata {
                        format: "q4".to_string(),
                        model_max_context: context_window,
                        context_window_policy: ContextWindowPolicy::PromptAndMaxTokens,
                    },
                ))
            },
            vision_runtime,
        ),
        options.max_pending,
        ResidencyLimits {
            max_adapters: options.max_resident_adapters,
            max_bytes: options.max_resident_adapter_bytes,
        },
    )
    .map_err(|error| match error {
        StartupError::Load(message) => message,
        // Keep the unified CLI's prior wording, distinct from the standalone
        // worker's generic `StartupError::ThreadExited` display text, for the
        // same condition: the worker exited before sending readiness.
        StartupError::ThreadExited => {
            "Metal worker thread exited before loading finished".to_string()
        }
        // The CLI parser rejects an out-of-range `--max-pending` before this
        // call, so this arm is unreachable through normal CLI startup. Keep
        // the display mapping for defense in depth.
        err @ StartupError::InvalidMaxPending { .. } => err.to_string(),
        err @ StartupError::InvalidResidencyLimits { .. } => err.to_string(),
    })?;
    // The explicit owner is not needed by this binary: every production
    // `MetalWorkerClient` retains an owner clone. The clone held by `client`
    // keeps the worker alive through the router's lifetime; on a normal
    // return, dropping the last client closes the queue before the last owner
    // performs its bounded join. As with the standalone server, that
    // guarantee does not reach this file's fatal `std::process::exit` calls:
    // process exit never runs destructors, so any owner clone still alive at
    // that point is dropped without performing the bounded join.
    drop(owner);
    Ok(ServingHandle::affine_cli(client, tokenizer, context_window))
}
