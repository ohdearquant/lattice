//! Shared OpenAI `/v1/embeddings` wire DTO, input normalization, and
//! response building (issue #584).
//!
//! Deliberately independent of `lattice-embed`'s `EmbeddingService` trait:
//! `lattice-embed` already depends on `lattice-inference` (for `BertModel`
//! and `QwenModel`), so a reverse dependency edge from this crate back onto
//! `lattice-embed` is a cyclic package dependency Cargo refuses to build
//! (verified with `cargo tree -p lattice-inference` after adding the edge:
//! "cyclic package dependency ... lattice-embed ... depends on itself").
//! This module normalizes the wire request and builds the wire response;
//! the actual embedding compute is `crate::model::bert::BertModel`, already
//! part of this crate, called directly by each binary's own handler.

use serde::Deserialize;
use serde_json::{Value, json};

use super::ApiError;

/// Maximum number of input texts accepted in a single `/v1/embeddings`
/// request. Mirrors OpenAI's own documented cap ("any array must be 2048
/// dimensions or less") so a client that already respects the real API's
/// limit never trips this one; it also bounds the CPU time and memory a
/// single request can force onto this server's `encode_batch` call before
/// any inference happens.
pub const MAX_EMBEDDINGS_BATCH_SIZE: usize = 2048;

/// OpenAI `input` field: a single string, or an array of strings.
///
/// `#[serde(untagged)]` tries each variant in order, matching how OpenAI
/// clients actually serialize this field (a bare JSON string or a bare JSON
/// array, never a wrapper object).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

/// Wire request body for `POST /v1/embeddings`.
#[derive(Debug, Default, Deserialize)]
pub struct EmbeddingsRequest {
    /// Text(s) to embed. `None` when the field is omitted or explicitly
    /// `null` — validated by [`parse_embeddings_input`].
    #[serde(default)]
    pub input: Option<EmbeddingsInput>,
    /// Requested model identifier. `None` when omitted; `Some("")` when the
    /// client sends an explicit empty string — both are validated by
    /// [`check_embeddings_model`], matching `contract::validate_model_name`'s
    /// `OptionalExact` policy (an explicit empty string is NOT treated as
    /// absent).
    #[serde(default)]
    pub model: Option<String>,
}

/// Extracts and validates the input texts from a parsed [`EmbeddingsRequest`].
///
/// Order is preserved exactly: index `i` of the returned `Vec` is index `i`
/// of the caller's array (or the sole element for a single string).
///
/// # Errors
///
/// - `input` missing or `null` — [`ApiError::BadRequest`] `invalid_request`.
/// - `input` is an empty string, or an empty array — `invalid_input`.
/// - any element is an empty string (including the single-string case) —
///   `invalid_input`, naming the offending index.
/// - the array has more than [`MAX_EMBEDDINGS_BATCH_SIZE`] elements —
///   `batch_size_exceeds_limit`.
pub fn parse_embeddings_input(input: &Option<EmbeddingsInput>) -> Result<Vec<String>, ApiError> {
    let texts = match input {
        None => {
            return Err(ApiError::BadRequest {
                message: "input is required".to_string(),
                code: "invalid_request",
            });
        }
        Some(EmbeddingsInput::One(text)) => vec![text.clone()],
        Some(EmbeddingsInput::Many(texts)) => texts.clone(),
    };
    if texts.is_empty() {
        return Err(ApiError::BadRequest {
            message: "input must not be empty".to_string(),
            code: "invalid_input",
        });
    }
    if texts.len() > MAX_EMBEDDINGS_BATCH_SIZE {
        return Err(ApiError::BadRequest {
            message: format!(
                "input has {} elements; maximum is {MAX_EMBEDDINGS_BATCH_SIZE}",
                texts.len()
            ),
            code: "batch_size_exceeds_limit",
        });
    }
    if let Some(index) = texts.iter().position(String::is_empty) {
        return Err(ApiError::BadRequest {
            message: format!("input[{index}] must not be empty"),
            code: "invalid_input",
        });
    }
    Ok(texts)
}

/// Validates a requested `model` against the single model this server has
/// loaded, using the same `OptionalExact` policy
/// `contract::ServeProfile::lattice_serve` applies to chat completions: an
/// omitted `model` is accepted; a present `model` (including an explicit
/// empty string) must equal `served`.
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] `model_not_found` on a mismatch.
pub fn check_embeddings_model(requested: Option<&str>, served: &str) -> Result<(), ApiError> {
    match requested {
        None => Ok(()),
        Some(requested) if requested == served => Ok(()),
        Some(requested) => Err(ApiError::BadRequest {
            message: format!("model '{requested}' is not loaded; this server serves '{served}'"),
            code: "model_not_found",
        }),
    }
}

/// Builds the `POST /v1/embeddings` success response body, OpenAI's
/// `CreateEmbeddingResponse` shape (verified against the `openai-python` SDK's
/// `src/openai/types/create_embedding_response.py` and
/// `src/openai/types/embedding.py`, since `platform.openai.com`'s own API
/// reference is not fetchable from this environment):
///
/// ```text
/// { object: "list", data: [{ object: "embedding", embedding: [f32], index }],
///   model, usage: { prompt_tokens, total_tokens } }
/// ```
///
/// `index` is assigned from each embedding's position in `embeddings`, which
/// must already be in the caller's original input order — this function
/// does not reorder anything.
///
/// `prompt_tokens` and `total_tokens` are equal: an embeddings request has no
/// completion tokens, matching OpenAI's own `Usage` shape for this endpoint.
pub fn build_embeddings_response(
    model: &str,
    embeddings: &[Vec<f32>],
    prompt_tokens: u64,
) -> Value {
    let data: Vec<Value> = embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({
                "object": "embedding",
                "embedding": embedding,
                "index": index,
            })
        })
        .collect();
    json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: &str) -> Option<EmbeddingsInput> {
        Some(EmbeddingsInput::One(s.to_string()))
    }

    fn many(items: &[&str]) -> Option<EmbeddingsInput> {
        Some(EmbeddingsInput::Many(
            items.iter().map(ToString::to_string).collect(),
        ))
    }

    #[test]
    fn parse_embeddings_input_missing_is_invalid_request() {
        let err = parse_embeddings_input(&None).unwrap_err();
        assert_eq!(err.code(), "invalid_request");
    }

    #[test]
    fn parse_embeddings_input_null_is_invalid_request() {
        // `#[serde(default)]` maps an explicit JSON `null` to `None` before
        // this function ever runs, same as an omitted field; this pins that
        // the two are indistinguishable by the time validation sees them.
        let req: EmbeddingsRequest = serde_json::from_str(r#"{"input":null}"#).unwrap();
        let err = parse_embeddings_input(&req.input).unwrap_err();
        assert_eq!(err.code(), "invalid_request");
    }

    #[test]
    fn parse_embeddings_input_single_string_is_one_element_at_index_zero() {
        let texts = parse_embeddings_input(&one("hello")).unwrap();
        assert_eq!(texts, vec!["hello".to_string()]);
    }

    #[test]
    fn parse_embeddings_input_array_preserves_order() {
        let texts = parse_embeddings_input(&many(&["a", "b", "c"])).unwrap();
        assert_eq!(
            texts,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn parse_embeddings_input_empty_string_is_invalid_input() {
        let err = parse_embeddings_input(&one("")).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
    }

    #[test]
    fn parse_embeddings_input_empty_array_is_invalid_input() {
        let err = parse_embeddings_input(&many(&[])).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
    }

    #[test]
    fn parse_embeddings_input_empty_string_inside_array_names_its_index() {
        let err = parse_embeddings_input(&many(&["a", "", "c"])).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        assert!(
            err.message().contains("input[1]"),
            "message must name the offending index: {}",
            err.message()
        );
    }

    #[test]
    fn parse_embeddings_input_over_batch_limit_is_rejected() {
        let items: Vec<String> = (0..MAX_EMBEDDINGS_BATCH_SIZE + 1)
            .map(|i| i.to_string())
            .collect();
        let input = Some(EmbeddingsInput::Many(items));
        let err = parse_embeddings_input(&input).unwrap_err();
        assert_eq!(err.code(), "batch_size_exceeds_limit");
    }

    #[test]
    fn parse_embeddings_input_at_batch_limit_is_accepted() {
        let items: Vec<String> = (0..MAX_EMBEDDINGS_BATCH_SIZE)
            .map(|i| i.to_string())
            .collect();
        let input = Some(EmbeddingsInput::Many(items));
        let texts = parse_embeddings_input(&input).unwrap();
        assert_eq!(texts.len(), MAX_EMBEDDINGS_BATCH_SIZE);
    }

    #[test]
    fn check_embeddings_model_accepts_omitted() {
        check_embeddings_model(None, "served-model").unwrap();
    }

    #[test]
    fn check_embeddings_model_accepts_exact_match() {
        check_embeddings_model(Some("served-model"), "served-model").unwrap();
    }

    #[test]
    fn check_embeddings_model_rejects_mismatch() {
        let err = check_embeddings_model(Some("other-model"), "served-model").unwrap_err();
        assert_eq!(err.code(), "model_not_found");
    }

    #[test]
    fn check_embeddings_model_rejects_explicit_empty_string() {
        // Matches contract::validate_model_name's OptionalExact policy: an
        // explicit empty string is a present-but-wrong model, not absence.
        let err = check_embeddings_model(Some(""), "served-model").unwrap_err();
        assert_eq!(err.code(), "model_not_found");
    }

    #[test]
    fn build_embeddings_response_shape() {
        // 0.5/0.25 (not 0.1/0.2/0.3): exactly representable in both f32 and
        // f64, so the f32 -> serde_json::Value (f64) widening this function
        // performs can't introduce a false mismatch against the literal
        // JSON array below.
        let body = build_embeddings_response("served-model", &[vec![0.5, 0.25, 1.0]], 4);
        assert_eq!(body["object"], "list");
        assert_eq!(body["model"], "served-model");
        assert_eq!(body["usage"]["prompt_tokens"], 4);
        assert_eq!(body["usage"]["total_tokens"], 4);
        assert_eq!(body["data"][0]["object"], "embedding");
        assert_eq!(body["data"][0]["index"], 0);
        assert_eq!(
            body["data"][0]["embedding"],
            serde_json::json!([0.5, 0.25, 1.0])
        );
    }

    #[test]
    fn build_embeddings_response_single_string_case_gets_index_zero() {
        let body = build_embeddings_response("served-model", &[vec![1.0]], 1);
        assert_eq!(body["data"].as_array().unwrap().len(), 1);
        assert_eq!(body["data"][0]["index"], 0);
    }

    #[test]
    fn build_embeddings_response_indices_reflect_input_order() {
        // Mutation-proven (see REPORT.md): hardcoding `"index": 0` for every
        // element in `build_embeddings_response` leaves this test red while
        // `build_embeddings_response_single_string_case_gets_index_zero`
        // above stays green (its only correct index already is 0) --
        // exactly the reason this test uses three distinct vectors instead
        // of one.
        let body = build_embeddings_response("served-model", &[vec![0.0], vec![1.0], vec![2.0]], 3);
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 3);
        for (expected_index, item) in data.iter().enumerate() {
            assert_eq!(
                item["index"], expected_index,
                "data[{expected_index}] must carry index {expected_index}"
            );
            assert_eq!(
                item["embedding"],
                serde_json::json!([expected_index as f32]),
                "data[{expected_index}] must carry the embedding at that input position"
            );
        }
    }
}
