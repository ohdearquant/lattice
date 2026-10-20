use super::*;

#[test]
fn recorded_router_loader_identifier_is_pinned() {
    assert_eq!(
        crate::serve::embeddings::QWEN35_F16_DECODER_LOADER,
        "qwen35-f16-decoder"
    );
}

#[test]
fn text_unsupported_parameters_precede_each_other_and_missing_input() {
    let cases = [
        (
            r#"{"dimensions":3,"encoding_format":"base64"}"#,
            "dimensions",
        ),
        (r#"{"dimensions":3}"#, "dimensions"),
        (r#"{"encoding_format":"base64"}"#, "encoding_format"),
    ];

    for (body, first_field) in cases {
        let request: TextEmbeddingsRequest =
            serde_json::from_str(body).expect("request fixture must parse");
        let error = check_embeddings_unsupported_fields(&request)
            .and_then(|()| parse_embeddings_input(&request.input).map(|_| ()))
            .expect_err("unsupported parameter must precede missing input");
        assert_eq!(error.code(), "unsupported_parameter");
        assert!(
            error.message().contains(first_field),
            "expected {first_field} to win, got {}",
            error.message()
        );
    }
}

#[test]
fn text_dimensions_null_is_accepted_but_a_number_is_rejected() {
    let null_request: TextEmbeddingsRequest =
        serde_json::from_str(r#"{"input":"a","dimensions":null}"#)
            .expect("null dimensions request must parse");
    check_embeddings_unsupported_fields(&null_request)
        .expect("null dimensions must be treated as absent");

    let numeric_request: TextEmbeddingsRequest =
        serde_json::from_str(r#"{"input":"a","dimensions":3}"#)
            .expect("numeric dimensions request must parse");
    let error = check_embeddings_unsupported_fields(&numeric_request)
        .expect_err("numeric dimensions must be rejected");
    assert_eq!(error.code(), "unsupported_parameter");
    assert_eq!(
        error.message(),
        "dimensions is not supported by this server; the loaded model's full output width is always returned"
    );
}

#[test]
fn text_batch_limit_precedes_an_empty_item() {
    let mut texts = vec!["a".to_string(); MAX_EMBEDDINGS_BATCH_SIZE + 1];
    texts[MAX_EMBEDDINGS_BATCH_SIZE] = String::new();
    let input = Some(TextEmbeddingsInput::Many(texts));

    let error =
        parse_embeddings_input(&input).expect_err("item count must be checked before empty items");
    assert_eq!(error.code(), "batch_size_exceeds_limit");
    assert_eq!(
        error.message(),
        format!(
            "input has {} elements; maximum is {}",
            MAX_EMBEDDINGS_BATCH_SIZE + 1,
            MAX_EMBEDDINGS_BATCH_SIZE
        )
    );
}
