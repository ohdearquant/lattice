use crate::forward::metal_qwen35::{
    ChatMessage, push_chat_generation_open, push_chat_turn_close, push_chat_turn_open,
};
use crate::model::qwen35_config::{Qwen35Config, VisionModelConfig};
use crate::serve::ApiError;
use crate::serve::prompt_adapter::apply_qwen_thinking_mode;
use crate::serving_runtime_contract::WorkerFailure;
#[cfg(test)]
use crate::tokenizer::Tokenizer as _;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::vision::VisionError;
use crate::vision::checkpoint::{
    Qwen35VisionWeights, load_qwen35_vision_weights_with_cancel,
    validate_qwen35_vision_weight_inventory,
};
use crate::vision::multimodal::Qwen35VisionRequest;
use crate::vision::qwen35_merger::qwen35_merger_forward_with_cancel;
use crate::vision::qwen35_vit::preprocess_qwen35_image_for_serve;
use crate::vision::qwen35_vit_metal::qwen35_vit_forward_metal_with_cancel;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

enum VisionState {
    Unsupported,
    Pending {
        model_dir: PathBuf,
        config: VisionModelConfig,
    },
    Loaded(Qwen35VisionWeights),
}

#[derive(Debug)]
pub(crate) enum VisionRuntimeLoad<'a> {
    Ready(&'a Qwen35VisionWeights),
    Unsupported,
    Cancelled,
}

/// Worker-local vision capability and lazily loaded vision weights.
///
/// Loading stays on the same dedicated thread that owns the Metal decoder
/// and happens only for the first admitted image request. Text-only servers
/// and text-only traffic retain the pre-vision startup and memory profile.
pub struct VisionRuntime {
    state: VisionState,
    vision_supported: Arc<AtomicBool>,
}

impl VisionRuntime {
    /// Resolve a runtime from concrete checkpoint metadata.
    pub fn from_model_config(model_dir: PathBuf, config: &Qwen35Config) -> Self {
        let token_metadata_present = match (
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
        ) {
            (Some(image), Some(start), Some(end)) => {
                [image, start, end]
                    .into_iter()
                    .all(|token| (token as usize) < config.vocab_size)
                    && image != start
                    && image != end
                    && start != end
            }
            _ => false,
        };
        let supported_weight_source = token_metadata_present
            && config.vision_config.as_ref().is_some_and(|vision_config| {
                validate_qwen35_vision_weight_inventory(&model_dir, vision_config).is_ok()
            });
        let state = match (
            &config.vision_config,
            token_metadata_present,
            supported_weight_source,
        ) {
            (Some(vision_config), true, true) => VisionState::Pending {
                model_dir,
                config: vision_config.clone(),
            },
            _ => VisionState::Unsupported,
        };
        let vision_supported =
            matches!(&state, VisionState::Pending { .. } | VisionState::Loaded(_));
        Self {
            state,
            vision_supported: Arc::new(AtomicBool::new(vision_supported)),
        }
    }

    /// Text-only runtime used by the compatibility
    /// [`crate::serving_worker::AffineWorker::spawn`] entry point and GPU-free worker tests.
    pub fn unsupported() -> Self {
        Self {
            state: VisionState::Unsupported,
            vision_supported: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Whether request normalization may admit image content parts.
    pub fn is_supported(&self) -> bool {
        self.vision_supported.load(Ordering::Acquire)
    }

    pub(crate) fn shared_capability(&self) -> Arc<AtomicBool> {
        self.vision_supported.clone()
    }

    /// Eagerly load vision weights now, ahead of the first image request
    /// (`--preload-vision`). A no-op when there is nothing pending (already
    /// loaded, or the checkpoint never advertised vision capability).
    ///
    /// On failure, deliberately leaves `state` at `Pending` rather than
    /// `Failed`, matching [`Self::get_or_load`]'s retry policy: a startup
    /// preload attempt is a best-effort optimization, so the caller is
    /// expected to warn and continue, and the FIRST image request still
    /// gets its own (possibly successful) lazy-load attempt via
    /// `get_or_load`, exactly as if `--preload-vision` had never been
    /// passed. This is what "falls back to lazy loading" means for this
    /// flag: preload failing must never disable vision for the session.
    pub fn preload(&mut self) -> Result<(), String> {
        let (model_dir, config) = match &self.state {
            VisionState::Pending { model_dir, config } => (model_dir.clone(), config.clone()),
            VisionState::Unsupported | VisionState::Loaded(_) => {
                return Ok(());
            }
        };
        let mut never_cancel = || false;
        match load_qwen35_vision_weights_with_cancel(&model_dir, &config, &mut never_cancel) {
            Ok(Some(weights)) => {
                self.state = VisionState::Loaded(weights);
                Ok(())
            }
            // `never_cancel` always returns `false`, so `should_cancel` can never
            // observe a cancellation request; `None` is unreachable in practice.
            Ok(None) => Ok(()),
            Err(err) => Err(format!("vision preload failed: {err}")),
        }
    }

    /// Attempt the lazy vision-weight load, retrying on every call while it
    /// keeps failing: a failed attempt leaves `self.state` at `Pending`
    /// rather than caching a permanent failure, and leaves
    /// `vision_supported` untouched (still advertised), so the next
    /// admitted image request tries the load again instead of being
    /// rejected forever from one transient failure. There is no
    /// permanent-failure policy elsewhere in the runtime for this codebase
    /// to honor instead.
    pub(crate) fn get_or_load(
        &mut self,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<VisionRuntimeLoad<'_>, String> {
        if let VisionState::Pending { model_dir, config } = &self.state {
            let model_dir = model_dir.clone();
            let config = config.clone();
            match load_qwen35_vision_weights_with_cancel(&model_dir, &config, should_cancel) {
                Ok(Some(weights)) => self.state = VisionState::Loaded(weights),
                Ok(None) => return Ok(VisionRuntimeLoad::Cancelled),
                Err(err) => return Err(format!("vision weights failed to load: {err}")),
            }
        }
        match &self.state {
            VisionState::Unsupported => Ok(VisionRuntimeLoad::Unsupported),
            VisionState::Loaded(weights) => Ok(VisionRuntimeLoad::Ready(weights)),
            VisionState::Pending { .. } => {
                self.vision_supported.store(false, Ordering::Release);
                Err("vision weights remained pending after a load attempt".to_string())
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn tokenize_text(tokenizer: &BpeTokenizer, text: &str) -> Vec<u32> {
    let encoded = tokenizer.tokenize(text);
    encoded.input_ids[..encoded.real_length].to_vec()
}

/// Build the vision prompt's text fragments around the image splice: `before`
/// ends immediately after the image message's leading text (image tokens are
/// inserted after this point) and `after` carries the trailing text plus every
/// later turn and the generation opener. Kept separate from token insertion
/// so it shares the same turn-sequencing primitives as the canonical
/// text-only renderer (`format_chat_template`) and so `before + after` is
/// directly comparable to that renderer's output around the splice point.
pub(crate) fn build_vision_prompt_text(
    messages: &[ChatMessage],
    image_message_index: usize,
    enable_thinking: bool,
    _explicitly_thinking: bool,
) -> Result<(String, String), WorkerFailure> {
    let image_message = &messages[image_message_index];
    let image = image_message
        .image
        .as_ref()
        .ok_or_else(|| WorkerFailure::Failed("vision dispatch lost its image payload".into()))?;
    if image.text_offset > image_message.content.len()
        || !image_message.content.is_char_boundary(image.text_offset)
    {
        return Err(WorkerFailure::Failed(
            "normalized image text offset is not a UTF-8 boundary".into(),
        ));
    }

    let mut before = String::new();
    for message in &messages[..image_message_index] {
        push_chat_turn_open(&mut before, message.role.as_str());
        before.push_str(&message.content);
        push_chat_turn_close(&mut before);
    }
    push_chat_turn_open(&mut before, image_message.role.as_str());
    before.push_str(&image_message.content[..image.text_offset]);

    let mut after = String::new();
    after.push_str(&image_message.content[image.text_offset..]);
    push_chat_turn_close(&mut after);
    for message in &messages[image_message_index + 1..] {
        push_chat_turn_open(&mut after, message.role.as_str());
        after.push_str(&message.content);
        push_chat_turn_close(&mut after);
    }
    push_chat_generation_open(&mut after);
    apply_qwen_thinking_mode(&mut after, enable_thinking);

    Ok((before, after))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_vision_prompt_ids(
    messages: &[ChatMessage],
    image_message_index: usize,
    tokenizer: &BpeTokenizer,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    image_token_id: u32,
    image_pad_count: usize,
    enable_thinking: bool,
    explicitly_thinking: bool,
) -> Result<Vec<u32>, WorkerFailure> {
    let (before, after) = build_vision_prompt_text(
        messages,
        image_message_index,
        enable_thinking,
        explicitly_thinking,
    )?;

    let mut inserted_ids = Vec::with_capacity(image_pad_count.saturating_add(2));
    inserted_ids.push(vision_start_token_id);
    inserted_ids.extend(std::iter::repeat_n(image_token_id, image_pad_count));
    inserted_ids.push(vision_end_token_id);
    let ids = tokenizer.tokenize_fragments_with_inserted_ids(&before, &inserted_ids, &after);
    if ids.iter().filter(|&&id| id == image_token_id).count() != image_pad_count {
        return Err(WorkerFailure::Rejected(ApiError::BadRequest {
            message: "message text must not contain the checkpoint's reserved image token"
                .to_string(),
            code: "invalid_messages",
        }));
    }
    Ok(ids)
}

#[derive(Debug)]
pub(crate) enum VisionRequestBuild {
    Ready {
        request: Qwen35VisionRequest,
        metal_dispatches: usize,
        gemm_calls: usize,
    },
    Cancelled,
}

pub(crate) fn build_vision_request(
    runtime: &mut VisionRuntime,
    config: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    messages: &[ChatMessage],
    image_message_index: usize,
    enable_thinking: bool,
    explicitly_thinking: bool,
    should_cancel: &mut dyn FnMut() -> bool,
    window_preflight: impl FnOnce(usize) -> Result<(), ApiError>,
) -> Result<VisionRequestBuild, WorkerFailure> {
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }
    let vision_config = config.vision_config.as_ref().ok_or_else(|| {
        WorkerFailure::Rejected(ApiError::BadRequest {
            message: "image input requires a vision-capable model".to_string(),
            code: "vision_unsupported",
        })
    })?;
    let image_token_id = config.image_token_id.ok_or_else(|| {
        WorkerFailure::Failed("vision checkpoint has no image_token_id".to_string())
    })?;
    let vision_start_token_id = config.vision_start_token_id.ok_or_else(|| {
        WorkerFailure::Failed("vision checkpoint has no vision_start_token_id".to_string())
    })?;
    let vision_end_token_id = config.vision_end_token_id.ok_or_else(|| {
        WorkerFailure::Failed("vision checkpoint has no vision_end_token_id".to_string())
    })?;
    let image = messages[image_message_index]
        .image
        .as_ref()
        .ok_or_else(|| WorkerFailure::Failed("vision dispatch lost its image payload".into()))?;

    let (pixel_values, grid) = preprocess_qwen35_image_for_serve(&image.bytes, vision_config)
        .map_err(|err| {
            // `DimensionsExceeded` is a well-formed image that is simply
            // larger than this server will process (a client can resize and
            // retry); every other `VisionError` here means the bytes
            // themselves are not a decodable image, which stays on the
            // generic `invalid_image` code.
            let code = match &err {
                VisionError::DimensionsExceeded(_) => "image_dimensions_exceeded",
                _ => "invalid_image",
            };
            WorkerFailure::Rejected(ApiError::BadRequest {
                message: format!("image preprocessing failed: {err}"),
                code,
            })
        })?;
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }
    let merge_area = vision_config
        .spatial_merge_size
        .checked_mul(vision_config.spatial_merge_size)
        .filter(|&area| area > 0)
        .ok_or_else(|| WorkerFailure::Failed("vision spatial_merge_size is invalid".to_string()))?;
    if !grid.num_patches().is_multiple_of(merge_area) {
        return Err(WorkerFailure::Rejected(ApiError::BadRequest {
            message: "image patch grid is incompatible with the checkpoint merge size".to_string(),
            code: "invalid_image",
        }));
    }
    let image_pad_count = grid.num_patches() / merge_area;
    let input_ids = build_vision_prompt_ids(
        messages,
        image_message_index,
        tokenizer,
        vision_start_token_id,
        vision_end_token_id,
        image_token_id,
        image_pad_count,
        enable_thinking,
        explicitly_thinking,
    )?;
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }
    window_preflight(input_ids.len()).map_err(WorkerFailure::Rejected)?;
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }

    let weights = match runtime.get_or_load(should_cancel).map_err(|detail| {
        // `detail` (from `load_qwen35_vision_weights_with_cancel`) routinely
        // names the checkpoint directory (e.g. a missing-manifest message
        // naming `model_dir`); log the full detail server-side only and
        // send the client a fixed, path-free message. The load is
        // retryable (see `VisionRuntime::get_or_load`), so this is a client
        // rejection (400), not a permanent server failure (500).
        eprintln!("[metal-worker] {detail}");
        WorkerFailure::Rejected(ApiError::BadRequest {
            message: "vision weights failed to load; the request can be retried".to_string(),
            code: "vision_load_failed",
        })
    })? {
        VisionRuntimeLoad::Ready(weights) => weights,
        VisionRuntimeLoad::Unsupported => {
            return Err(WorkerFailure::Rejected(ApiError::BadRequest {
                message: "image input requires a vision-capable model".to_string(),
                code: "vision_unsupported",
            }));
        }
        VisionRuntimeLoad::Cancelled => return Ok(VisionRequestBuild::Cancelled),
    };
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }
    let Some(vit_output) = qwen35_vit_forward_metal_with_cancel(
        weights,
        vision_config,
        &pixel_values,
        grid,
        should_cancel,
    )
    .map_err(|err| WorkerFailure::Failed(format!("vision forward failed: {err}")))?
    else {
        return Ok(VisionRequestBuild::Cancelled);
    };
    let Some(post_merger) = qwen35_merger_forward_with_cancel(
        &weights.merger,
        vision_config,
        &vit_output.hidden_states,
        should_cancel,
    )
    .map_err(|err| WorkerFailure::Failed(format!("vision merger failed: {err}")))?
    else {
        return Ok(VisionRequestBuild::Cancelled);
    };
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }

    Ok(VisionRequestBuild::Ready {
        request: Qwen35VisionRequest {
            input_ids,
            image_grids: vec![grid],
            post_merger_rows: post_merger,
            image_token_id,
            spatial_merge_size: vision_config.spatial_merge_size,
            decoder_hidden_size: config.hidden_size,
        },
        metal_dispatches: vit_output.metal_dispatches,
        gemm_calls: vit_output.gemm_calls,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobRoute {
    Text,
    Vision { message_index: usize },
}

pub(crate) fn classify_job(messages: &[ChatMessage]) -> Result<JobRoute, WorkerFailure> {
    let mut image_positions = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.image.is_some());
    let first = image_positions.next();
    if image_positions.next().is_some() {
        return Err(WorkerFailure::Rejected(ApiError::BadRequest {
            message: "only one image is supported per request".to_string(),
            code: "multiple_images_unsupported",
        }));
    }
    Ok(match first {
        Some((_message_index, message))
            if message.role != crate::forward::metal_qwen35::ChatRole::User =>
        {
            return Err(WorkerFailure::Rejected(ApiError::BadRequest {
                message: "image content is supported only on user messages".to_string(),
                code: "invalid_image_role",
            }));
        }
        Some((message_index, _)) => JobRoute::Vision { message_index },
        None => JobRoute::Text,
    })
}
