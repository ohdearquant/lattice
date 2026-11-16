//! The `/v1/systemone` HTTP route.
//!
//! The route is served only when the loaded checkpoint directory carries a
//! `decider_config.json`; [`DeciderServing::load`] reads that file, the
//! checkpoint's `tokenizer.json` and its label table at startup, so a
//! checkpoint whose config or labels are refused stops the server before it
//! listens. A request takes a preparation slot, is parsed, rendered and
//! assembled on a blocking thread, and its rows are submitted to the serving
//! host as one scoring job: a worker-backed host scores them in one queued
//! request, the shared CPU host on a blocking task that stops between rows
//! once the caller hangs up. The answers are built by [`score_prepared`].
//!
//! Refusals follow the reference server: HTTP 422 with a `detail` message. A
//! scorer failure is a server error; a full preparation cap or worker queue
//! is a 503.

use std::path::Path;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use lattice_inference::option_scoring::DecisionLabelTable;
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_tokenizer::Tokenizer;

use super::score::{ScoreFailure, score_prepared};
use super::{DeciderConfig, SystemOneError, load_decider_config, prepare_request, resolve_labels};
use crate::serve::ApiError;
use crate::serving_host::ServingHandle;
use crate::serving_worker::ScoreRow;

/// The checkpoint-bound pieces the route needs: its tokenizer, its resolved
/// label table and its config.
pub struct DeciderServing {
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    labels: DecisionLabelTable,
    config: DeciderConfig,
}

impl DeciderServing {
    /// Build from parts. The tokenizer must encode without truncation: the
    /// prompt layer slices the state itself and refuses a truncated encode.
    pub fn from_parts(
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        labels: DecisionLabelTable,
        config: DeciderConfig,
    ) -> Self {
        Self {
            tokenizer,
            labels,
            config,
        }
    }

    /// Load the decider pieces of a checkpoint directory, or `None` when it
    /// has no `decider_config.json`.
    ///
    /// # Errors
    ///
    /// Refuses a config the port does not accept, an unreadable
    /// `tokenizer.json`, and a label table that does not resolve to single,
    /// distinct tokens.
    pub fn load(checkpoint_dir: &Path) -> Result<Option<Self>, SystemOneError> {
        if !checkpoint_dir.join("decider_config.json").is_file() {
            return Ok(None);
        }
        let config = load_decider_config(checkpoint_dir)?;
        let path = checkpoint_dir.join("tokenizer.json");
        // The window only bounds truncation: batch encodes pad to their own
        // length, and the request body limit bounds the input.
        let tokenizer = BpeTokenizer::from_tokenizer_json(&path)
            .map_err(|error| {
                SystemOneError::Refused(format!("could not load {}: {error}", path.display()))
            })?
            .with_max_seq_len(usize::MAX);
        let labels = resolve_labels(&tokenizer, config.max_options)?;
        Ok(Some(Self::from_parts(Arc::new(tokenizer), labels, config)))
    }

    /// The response `model` string, `decider-<version>`.
    pub fn model_name(&self) -> String {
        super::SystemOneResponse::keyword_for(&self.config)
    }
}

#[derive(Clone)]
/// The most `/v1/systemone` requests preparing at once. Preparation
/// tokenizes the state and assembles every row before the scoring admission
/// check, so a request takes one of these slots first and is refused with
/// 503 when none is free.
pub const MAX_CONCURRENT_PREPARATIONS: usize = 4;

#[derive(Clone)]
struct RouteState {
    model: ServingHandle,
    decider: Arc<DeciderServing>,
    preparation: Arc<tokio::sync::Semaphore>,
}

/// The `/v1/systemone` router, to merge into a server's router.
pub fn router(model: ServingHandle, decider: Arc<DeciderServing>) -> Router {
    router_with_preparation(
        model,
        decider,
        Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_PREPARATIONS)),
    )
}

fn router_with_preparation(
    model: ServingHandle,
    decider: Arc<DeciderServing>,
    preparation: Arc<tokio::sync::Semaphore>,
) -> Router {
    Router::new()
        .route("/v1/systemone", post(systemone))
        .with_state(RouteState {
            model,
            decider,
            preparation,
        })
}

fn refused(error: SystemOneError) -> Response {
    let SystemOneError::Refused(detail) = error;
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(serde_json::json!({ "detail": detail })),
    )
        .into_response()
}

fn internal(message: impl Into<String>) -> Response {
    ApiError::Internal {
        message: message.into(),
    }
    .into_response()
}

async fn systemone(State(state): State<RouteState>, body: Bytes) -> Response {
    let Ok(permit) = Arc::clone(&state.preparation).try_acquire_owned() else {
        return ApiError::ServiceUnavailable {
            message: "too many /v1/systemone requests are being prepared; retry shortly"
                .to_string(),
        }
        .into_response();
    };
    let Ok(text) = String::from_utf8(body.to_vec()) else {
        return refused(SystemOneError::Refused(
            "the request body is not UTF-8".to_string(),
        ));
    };
    let decider = Arc::clone(&state.decider);
    let mut prepared = match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        prepare_request(
            decider.tokenizer.as_ref(),
            &decider.labels,
            &decider.config,
            &text,
        )
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(error)) => return refused(error),
        Err(error) => return internal(format!("request preparation failed: {error}")),
    };

    let mut rows = Vec::with_capacity(prepared.prompt.rows.len());
    for (index, row) in prepared.prompt.rows.iter_mut().enumerate() {
        let Some(row_labels) = prepared.label_ids.get(..row.nopts) else {
            return refused(SystemOneError::Refused(format!(
                "row {index} has {} options, the label table holds {}",
                row.nopts,
                prepared.label_ids.len()
            )));
        };
        // The ids move to the worker: scoring below reads only the logits it
        // returned, and the response's token count was fixed at preparation.
        rows.push(ScoreRow {
            ids: std::mem::take(&mut row.ids),
            labels: row_labels.to_vec(),
        });
    }

    let receiver = match state.model.submit_score(rows) {
        Ok(receiver) => receiver,
        Err(error) => return error.into_response(),
    };
    let logits = match receiver.await {
        Ok(Ok(logits)) => logits,
        Ok(Err(message)) => return internal(format!("scoring failed: {message}")),
        Err(_) => return internal("the inference worker stopped before replying"),
    };

    let mut rows = logits.into_iter();
    let scored = score_prepared(&prepared, |_, _| {
        rows.next()
            .ok_or_else(|| "the worker returned fewer rows than were sent".to_string())
    });
    match scored {
        Ok(response) => Json(response.to_serde_json()).into_response(),
        Err(ScoreFailure::Refused(error)) => refused(error),
        Err(ScoreFailure::Scorer { row, error }) => internal(format!("row {row}: {error}")),
    }
}

#[cfg(test)]
mod tests;
