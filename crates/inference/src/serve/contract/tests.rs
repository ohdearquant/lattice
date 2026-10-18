use super::*;

fn request(body: &str) -> ChatRequest {
    serde_json::from_str(body).unwrap()
}

fn defaults() -> GenerationDefaults {
    GenerationDefaults::standard(16)
}

fn inline_image_request(role: &str, url: &str) -> ChatRequest {
    request(&format!(
        r#"{{"model":"model","messages":[{{"role":"{role}","content":[
                {{"type":"text","text":"before"}},
                {{"type":"image_url","image_url":{{"url":"{url}"}}}},
                {{"type":"text","text":"after"}}
            ]}}]}}"#
    ))
}

fn valid_png_data_uri() -> String {
    let image = image::RgbImage::new(1, 1);
    let mut bytes = Vec::new();
    image
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("PNG fixture must encode");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn vision_profile() -> ServeProfile<'static> {
    ServeProfile::lattice("model", 32).with_vision_support(true)
}
#[test]
fn standard_generation_defaults_snapshot() {
    let defaults = GenerationDefaults::standard(123);
    assert_eq!(defaults.max_tokens, 123);
    assert_eq!(defaults.temperature, 0.7);
    assert_eq!(defaults.top_k, 50);
    assert_eq!(defaults.top_p, 0.9);
    assert_eq!(defaults.repetition_penalty, 1.1);
    assert_eq!(defaults.reasoning_budget, None);
}

#[test]
fn contract_source_has_no_backend_message_dependency() {
    let source = include_str!("../contract.rs");
    assert!(!source.contains(concat!("crate::", "forward")));
}

#[test]
fn vision_profile_preserves_inline_image_and_text_position() {
    let req = inline_image_request("user", &valid_png_data_uri());
    let validated = normalize_request(&req, defaults(), vision_profile()).unwrap();
    let message = &validated.messages[0];
    assert_eq!(message.content, "beforeafter");
    let image = message
        .image
        .as_ref()
        .expect("image must survive normalization");
    assert!(image.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert_eq!(image.text_offset, "before".len());
}

#[test]
fn text_only_profile_rejects_image_by_loaded_capability() {
    let req = inline_image_request("user", &valid_png_data_uri());
    let err = normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap_err();
    assert_eq!(err.code(), "vision_unsupported");
    assert_eq!(err.message(), "image input requires a vision-capable model");
}

#[test]
fn vision_profile_rejects_remote_image_urls() {
    let req = inline_image_request("user", "https://example.com/image.png");
    let err = normalize_request(&req, defaults(), vision_profile()).unwrap_err();
    assert_eq!(err.code(), "unsupported_image_url_scheme");
}

#[test]
fn vision_profile_rejects_malformed_or_empty_data_uris() {
    for url in [
        "data:image/png;base64",
        "data:image/png,AAAA",
        "data:text/plain;base64,cG5n",
        "data:image/png;base64,",
        "data:image/png;base64,***",
        "data:image/png;base64,cG5n",
    ] {
        let req = inline_image_request("user", url);
        let err = normalize_request(&req, defaults(), vision_profile()).unwrap_err();
        assert_eq!(err.code(), "invalid_image", "url={url}");
    }
}

#[test]
fn vision_profile_rejects_media_type_payload_mismatch() {
    let uri = valid_png_data_uri().replacen("image/png", "image/jpeg", 1);
    let req = inline_image_request("user", &uri);
    let err = normalize_request(&req, defaults(), vision_profile()).unwrap_err();
    assert_eq!(err.code(), "invalid_image");
    assert!(err.message().contains("declares image/jpeg"));
}

#[test]
fn vision_profile_rejects_oversized_decoded_image_before_allocation() {
    let payload = "A".repeat(MAX_ENCODED_IMAGE_BYTES + 4);
    let req = inline_image_request("user", &format!("data:image/png;base64,{payload}"));
    let err = normalize_request(&req, defaults(), vision_profile()).unwrap_err();
    assert_eq!(err.code(), "invalid_image");
    assert!(err.message().contains("base64 bytes; maximum"));
}

#[test]
fn vision_profile_rejects_multiple_images_across_messages() {
    let uri = valid_png_data_uri();
    let req = request(&format!(
        r#"{{"model":"model","messages":[
                {{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"{uri}"}}}}]}},
                {{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"{uri}"}}}}]}}
            ]}}"#
    ));
    let err = normalize_request(&req, defaults(), vision_profile()).unwrap_err();
    assert_eq!(err.code(), "multiple_images_unsupported");
}

#[test]
fn vision_profile_rejects_image_on_non_user_role() {
    let uri = valid_png_data_uri();
    let req = request(&format!(
        r#"{{"model":"model","messages":[
                {{"role":"system","content":[{{"type":"image_url","image_url":{{"url":"{uri}"}}}}]}},
                {{"role":"user","content":"question"}}
            ]}}"#
    ));
    let err = normalize_request(&req, defaults(), vision_profile()).unwrap_err();
    assert_eq!(err.code(), "invalid_image_role");
}

#[test]
fn image_requests_reject_unwired_generation_extensions() {
    let uri = valid_png_data_uri();
    let mut req = inline_image_request("user", &uri);
    req.stream = Some(true);
    assert_eq!(
        normalize_request(&req, defaults(), vision_profile())
            .unwrap_err()
            .code(),
        "unsupported_feature"
    );

    let mut req = inline_image_request("user", &uri);
    req.logprobs = Some(true);
    assert_eq!(
        normalize_request(&req, defaults(), vision_profile())
            .unwrap_err()
            .code(),
        "unsupported_feature"
    );

    // Pinned to the exact `BadRequest` variant (not just `.code()`), since
    // that variant is what `ApiError`'s `IntoResponse` impl maps to HTTP
    // 400 (`serve/mod.rs`, exercised there and in each binary's own
    // envelope tests) -- this is the proof that `image_unsupported_combination`
    // is a 400, not merely that some error carries that code string.
    let mut req = inline_image_request("user", &uri);
    req.reasoning_budget = Some(serde_json::value::RawValue::from_string("4".to_string()).unwrap());
    assert!(matches!(
        normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("model", 32).with_vision_support(true),
        )
        .unwrap_err(),
        ApiError::BadRequest {
            code: "image_unsupported_combination",
            ..
        }
    ));

    let mut req = inline_image_request("user", &uri);
    req.response_format = Some(ResponseFormat {
        r#type: "json_schema".to_string(),
        json_schema: None,
    });
    assert!(matches!(
        normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("model", 32).with_vision_support(true),
        )
        .unwrap_err(),
        ApiError::BadRequest {
            code: "image_unsupported_combination",
            ..
        }
    ));
}

#[test]
fn serving_adapters_do_not_restate_standard_defaults() {
    // `lattice` is now a module tree (#834), not a single file; concatenate
    // its split source files so this still scans the same source text.
    let lattice_source = concat!(
        include_str!("../../bin/lattice/main.rs"),
        include_str!("../../bin/lattice/chat.rs"),
        include_str!("../../bin/lattice/doctor.rs"),
        include_str!("../../bin/lattice/serve.rs"),
    );
    for (name, source) in [
        ("lattice", lattice_source),
        (
            "lattice_serve",
            include_str!("../../bin/lattice_serve/main.rs"),
        ),
    ] {
        for literal in [
            "temperature: 0.7",
            "top_k: 50",
            "top_p: 0.9",
            "repetition_penalty: 1.1",
            "unwrap_or(512)",
            "unwrap_or(0.7)",
            "unwrap_or(50)",
            "unwrap_or(0.9)",
            "unwrap_or(1.1)",
        ] {
            assert!(
                !source.contains(literal),
                "{name} restates canonical generation default {literal}"
            );
        }
    }
}

#[test]
fn shared_sampling_bounds_reject_invalid_values() {
    assert_eq!(
        validate_temperature(-0.1).unwrap_err().code(),
        "invalid_temperature"
    );
    assert_eq!(
        validate_temperature(2.1).unwrap_err().code(),
        "invalid_temperature"
    );
    assert_eq!(validate_top_p(0.0).unwrap_err().code(), "invalid_top_p");
    assert_eq!(validate_top_p(1.1).unwrap_err().code(), "invalid_top_p");
}

#[test]
fn both_profiles_use_shared_sampling_bounds() {
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"temperature":2.5}"#,
    );
    for profile in [
        ServeProfile::lattice("model", 32),
        ServeProfile::lattice_serve("model", 32),
    ] {
        assert_eq!(
            normalize_request(&req, defaults(), profile)
                .unwrap_err()
                .code(),
            "invalid_temperature"
        );
    }
}

#[test]
fn profiles_preserve_stop_and_max_token_policy() {
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"max_tokens":31,"stop":"done"}"#,
    );
    let lattice = normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap();
    assert_eq!(lattice.max_tokens, 31);
    assert_eq!(lattice.stop_strings, ["done"]);

    // #831: lattice_serve now also accepts and propagates `stop`,
    // aligned with the `lattice` profile (was previously rejected
    // outright with unsupported_feature).
    let daemon_with_stop =
        normalize_request(&req, defaults(), ServeProfile::lattice_serve("model", 100)).unwrap();
    assert_eq!(daemon_with_stop.stop_strings, ["done"]);

    let req =
        request(r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"max_tokens":31}"#);
    let daemon =
        normalize_request(&req, defaults(), ServeProfile::lattice_serve("model", 16)).unwrap();
    assert_eq!(daemon.max_tokens, 15);
}

#[test]
fn profiles_preserve_sampling_extension_policy() {
    // lattice serve never carried top_k/repetition_penalty in its
    // pre-shared-contract DTO and silently ignored them; the shared
    // contract must keep accepting-and-ignoring these fields for that
    // profile rather than rejecting the request (regression guard).
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":20,"repetition_penalty":1.2}"#,
    );
    let lattice = normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap();
    assert_eq!(lattice.top_k, defaults().top_k);
    assert_eq!(lattice.repetition_penalty, defaults().repetition_penalty);

    let daemon =
        normalize_request(&req, defaults(), ServeProfile::lattice_serve("model", 32)).unwrap();
    assert_eq!(daemon.top_k, 20);
    assert_eq!(daemon.repetition_penalty, 1.2);
}

#[test]
fn both_profiles_apply_reasoning_budget() {
    // #831 (maintainer-binding decision): `lattice serve` must actually
    // apply `reasoning_budget`, matching `lattice_serve`'s pre-existing
    // behavior, instead of accepting-and-ignoring it. Mutation-sensitive:
    // dropping the reasoning_budget field assignment or zeroing it turns
    // either assertion `None`.
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"reasoning_budget":40}"#,
    );

    // Context must leave enough decode room (>= reasoning_budget) so the
    // context-window clamp in normalize_request doesn't also shrink the
    // value under test; that clamp is exercised separately.
    let lattice = normalize_request(&req, defaults(), ServeProfile::lattice("model", 100)).unwrap();
    assert_eq!(lattice.reasoning_budget, Some(40));

    let daemon =
        normalize_request(&req, defaults(), ServeProfile::lattice_serve("model", 100)).unwrap();
    assert_eq!(daemon.reasoning_budget, Some(40));
}

#[test]
fn lattice_profile_tolerates_malformed_ignored_sampling_fields() {
    // Regression: pre-refactor, lattice serve's DTO didn't model
    // top_k/repetition_penalty at all, so a type-mismatched value (e.g. a
    // string where a number is expected) was silently ignored along with
    // everything else serde didn't recognize. The shared typed DTO must
    // preserve that tolerance rather than hard-failing the whole request
    // with invalid_request_body before profile normalization runs.
    // `reasoning_budget` is excluded here (unlike pre-#831): the lattice
    // profile now honors it, so a malformed value is strictly rejected —
    // see `honoring_profile_still_rejects_malformed_sampling_fields`.
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":"ignored","repetition_penalty":"x"}"#,
    );
    let lattice = normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap();
    assert_eq!(lattice.top_k, defaults().top_k);
    assert_eq!(lattice.repetition_penalty, defaults().repetition_penalty);
}

#[test]
fn honoring_profile_still_rejects_malformed_sampling_fields() {
    // Control: a profile that actually USES top_k / repetition_penalty
    // must keep strict validation — tolerance is scoped to profiles that
    // ignore the field, not a blanket type-laxness relaxation. Neither
    // profile ignores `reasoning_budget` as of #831, so both reject a
    // malformed value.
    for (field, bad_value) in [("top_k", r#""ignored""#), ("repetition_penalty", r#""x""#)] {
        let body = format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"{field}":{bad_value}}}"#
        );
        let req = request(&body);
        assert_eq!(
            normalize_request(&req, defaults(), ServeProfile::lattice_serve("model", 32),)
                .unwrap_err()
                .code(),
            "invalid_request_body",
            "field {field} should have been rejected on the honoring profile"
        );
    }

    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"reasoning_budget":"y"}"#,
    );
    for profile in [
        ServeProfile::lattice("model", 32),
        ServeProfile::lattice_serve("model", 32),
    ] {
        assert_eq!(
            normalize_request(&req, defaults(), profile)
                .unwrap_err()
                .code(),
            "invalid_request_body",
            "malformed reasoning_budget should have been rejected on both profiles"
        );
    }
}

/// ADR-094 amendment: the `lora` field carries THREE states and they must
/// stay pairwise distinguishable after deserialization.
///
/// The arms are the point, not the classification helper. Under the previous
/// `#[serde(default)] pub lora: Vec<LoraSelection>`, an absent field and an
/// explicit `[]` produced the identical value, so "route for me" could not be
/// spelled at all. This test fails if that collapse returns — which is what a
/// later `#[serde(default)]` on a non-`Option` type, or a `default` attribute
/// added to the `Option`, would silently reintroduce.
#[test]
fn the_lora_field_keeps_absent_empty_and_explicit_pairwise_distinct() {
    use super::super::lora::{LoraSelection, RequestedAdapters, requested_adapters};

    let body = |lora: &str| {
        format!(r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}]{lora}}}"#)
    };
    let parse = |lora: &str| {
        serde_json::from_str::<ChatRequest>(&body(lora))
            .expect("body should deserialize")
            .lora
    };

    let absent = parse("");
    let empty = parse(r#","lora":[]"#);
    let explicit = parse(r#","lora":[{"id":3,"scale":0.5}]"#);

    // The raw field first. Asserting only the classification would let a
    // regression that collapses absent and empty pass if the classifier were
    // changed to compensate, and these are two separate claims.
    assert_eq!(absent, None, "an omitted field must stay None");
    assert_eq!(empty, Some(vec![]), "an explicit [] must stay Some(empty)");
    assert_ne!(absent, empty, "absent and [] must not deserialize alike");
    assert_eq!(
        explicit,
        Some(vec![LoraSelection { id: 3, scale: 0.5 }]),
        "an explicit list must survive intact"
    );

    // Then the three states the rest of the serving path branches on.
    assert_eq!(requested_adapters(absent), RequestedAdapters::Unspecified);
    assert_eq!(requested_adapters(empty), RequestedAdapters::PinnedBase);
    assert_eq!(
        requested_adapters(explicit),
        RequestedAdapters::Explicit(vec![LoraSelection { id: 3, scale: 0.5 }])
    );
}

/// `Unspecified` and `PinnedBase` both apply nothing today, and that is what
/// makes the field change behaviour preserving while no router exists. They
/// are still not interchangeable: routing replaces the `Unspecified` arm and
/// must leave `PinnedBase` alone, so the emptiness of the two selections must
/// never become the thing the serving path tests.
#[test]
fn unspecified_and_pinned_base_apply_nothing_today_but_only_one_is_routable() {
    use super::super::lora::RequestedAdapters;

    assert!(RequestedAdapters::Unspecified.selection().is_empty());
    assert!(RequestedAdapters::PinnedBase.selection().is_empty());
    assert!(RequestedAdapters::Unspecified.is_routable());
    assert!(
        !RequestedAdapters::PinnedBase.is_routable(),
        "a client pinning the base model with [] must never be routed"
    );
}

#[test]
fn message_count_over_limit_is_rejected_during_deserialization() {
    // Regression: the message-count bound is enforced inline while
    // `messages` deserializes ([`deserialize_bounded_messages`]), so an
    // over-limit request never reaches a constructed `ChatRequest`,
    // `normalize_request`, or `check_context` at all -- there is no
    // separate preflight pass and no post-deserialization gap to close.
    let messages: Vec<String> = (0..MAX_MESSAGE_COUNT + 1)
        .map(|_| r#"{"role":"user","content":""}"#.to_string())
        .collect();
    let body = format!(r#"{{"model":"model","messages":[{}]}}"#, messages.join(","));
    let err = serde_json::from_str::<ChatRequest>(&body).unwrap_err();
    assert!(is_message_flood_error(&err));
}

#[test]
fn message_flood_error_short_circuits_at_max_plus_one() {
    // Single-pass, short-circuiting: an array far larger than
    // `MAX_MESSAGE_COUNT` still rejects promptly, because
    // `deserialize_bounded_messages`'s visitor bails the instant its
    // running count exceeds the limit -- it never visits the remaining
    // elements. A large-but-finite array here proves the bail actually
    // fires past the boundary, not just that a walk-to-completion would
    // eventually reject it too.
    let messages: Vec<String> = (0..MAX_MESSAGE_COUNT * 4)
        .map(|_| r#"{"role":"user","content":""}"#.to_string())
        .collect();
    let body = format!(r#"{{"model":"model","messages":[{}]}}"#, messages.join(","));
    let err = serde_json::from_str::<ChatRequest>(&body).unwrap_err();
    assert!(is_message_flood_error(&err));
}

#[test]
fn non_flood_deserialize_errors_are_not_misclassified_as_message_flood() {
    assert!(!is_message_flood_error(
        &serde_json::from_str::<ChatRequest>("not json").unwrap_err()
    ));
    assert!(!is_message_flood_error(
        &serde_json::from_str::<ChatRequest>(
            r#"{"model":"model","messages":[{"role":123,"content":"hi"}]}"#
        )
        .unwrap_err()
    ));
}

#[test]
fn cumulative_content_bytes_over_limit_is_rejected() {
    let big_content = "x".repeat(MAX_CUMULATIVE_CONTENT_BYTES + 1);
    let body =
        format!(r#"{{"model":"model","messages":[{{"role":"user","content":"{big_content}"}}]}}"#);
    let req = request(&body);
    assert_eq!(
        normalize_request(&req, defaults(), ServeProfile::lattice("model", 32),)
            .unwrap_err()
            .code(),
        "invalid_request_body"
    );
}

#[test]
fn message_count_and_content_bytes_at_the_limit_are_accepted() {
    let messages: Vec<String> = (0..MAX_MESSAGE_COUNT)
        .map(|i| {
            if i + 1 == MAX_MESSAGE_COUNT {
                r#"{"role":"user","content":"hi"}"#.to_string()
            } else {
                r#"{"role":"user","content":""}"#.to_string()
            }
        })
        .collect();
    let body = format!(r#"{{"model":"model","messages":[{}]}}"#, messages.join(","));
    let req = request(&body);
    normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap();
}

#[test]
fn stop_string_over_byte_limit_is_rejected_before_matcher_construction() {
    // Regression (B1): a near-1MiB stop string must be rejected as
    // invalid_request_body before StopStringMatcher ever sees it -- not
    // parsed into `stop_strings` and handed to the matcher/decode loop.
    let big_stop = "x".repeat(MAX_STOP_STRING_BYTES + 1);
    let body = format!(
        r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"stop":"{big_stop}"}}"#
    );
    let req = request(&body);
    let err = normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap_err();
    assert_eq!(err.code(), "invalid_request_body");

    // Same bound applies to each element of a stop array.
    let body = format!(
        r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"stop":["{big_stop}"]}}"#
    );
    let req = request(&body);
    assert_eq!(
        normalize_request(&req, defaults(), ServeProfile::lattice("model", 32),)
            .unwrap_err()
            .code(),
        "invalid_request_body"
    );
}

#[test]
fn stop_strings_over_cumulative_byte_limit_are_rejected() {
    // Regression (B1): four individually-under-limit stop strings that
    // together exceed MAX_CUMULATIVE_STOP_BYTES must still be rejected
    // -- the per-string cap alone is not the only bound enforced. Each
    // string is well under MAX_STOP_STRING_BYTES; only the sum crosses
    // MAX_CUMULATIVE_STOP_BYTES.
    let each = "x".repeat(MAX_CUMULATIVE_STOP_BYTES / 4 + 1);
    let stops = format!(r#""{each}","{each}","{each}","{each}""#);
    let body = format!(
        r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"stop":[{stops}]}}"#
    );
    let req = request(&body);
    assert_eq!(
        normalize_request(&req, defaults(), ServeProfile::lattice("model", 32),)
            .unwrap_err()
            .code(),
        "invalid_request_body"
    );
}

#[test]
fn stop_string_at_the_byte_limit_is_accepted() {
    let ok_stop = "x".repeat(MAX_STOP_STRING_BYTES);
    let body = format!(
        r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"stop":"{ok_stop}"}}"#
    );
    let req = request(&body);
    let validated =
        normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap();
    assert_eq!(validated.stop_strings, vec![ok_stop]);
}

#[test]
fn non_scalar_extension_field_rejected_without_cloning_large_payload() {
    // Regression (B2): a large non-scalar (array) value for a
    // sampling-extension field must be rejected on the honoring profile
    // via the cheap `is_array`/`is_object` shape check in
    // `parse_ignorable_field`, before any `Value::clone()` or
    // `from_value` deserialization attempt is made on the whole payload.
    // Asserting the exact "must be a scalar value" message (not just the
    // shared `invalid_request_body` code, which `from_value`'s own
    // type-mismatch error also produces) pins that the shape-check path
    // fired, not the deserialize-then-fail path the pre-B2 code took.
    let big_array = format!("[{}]", vec!["1"; 100_000].join(","));
    let body = format!(
        r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"top_k":{big_array}}}"#
    );
    let req = request(&body);
    let err =
        normalize_request(&req, defaults(), ServeProfile::lattice_serve("model", 32)).unwrap_err();
    assert_eq!(err.code(), "invalid_request_body");
    assert_eq!(err.message(), "top_k must be a scalar value");
}

#[test]
fn wrong_model_with_conflicting_stream_and_logprobs_rejects_as_unsupported_feature() {
    // Regression (B3): restores the pre-shared-contract first-error
    // precedence -- `stream: true` + `logprobs: true` must be rejected
    // as unsupported_feature even when the request ALSO targets the
    // wrong model, matching both binaries' original `reject_unsupported`
    // ordering (ran before the model-id check).
    let req = request(
        r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}],"stream":true,"logprobs":true}"#,
    );
    assert_eq!(
        normalize_request(&req, defaults(), ServeProfile::lattice("served-model", 32),)
            .unwrap_err()
            .code(),
        "unsupported_feature"
    );
}

#[test]
fn wrong_model_with_unsupported_logprobs_on_daemon_profile_rejects_as_unsupported_feature() {
    // Regression (B3): the standalone daemon profile never supports
    // logprobs at all; that rejection must still win over model_not_found.
    let req = request(
        r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}],"logprobs":true}"#,
    );
    assert_eq!(
        normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("served-model", 32),
        )
        .unwrap_err()
        .code(),
        "unsupported_feature"
    );
}

#[test]
fn wrong_model_with_conflicting_max_tokens_alias_rejects_as_invalid_request() {
    // Regression (B3): a conflicting max_tokens/max_completion_tokens
    // pair must be rejected as invalid_request even when the request
    // ALSO targets the wrong model, matching the standalone daemon's
    // original `reject_unsupported` ordering (ran before the model-id
    // check).
    let req = request(
        r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
    );
    assert_eq!(
        normalize_request(
            &req,
            defaults(),
            ServeProfile::lattice_serve("served-model", 32),
        )
        .unwrap_err()
        .code(),
        "invalid_request"
    );
}

#[test]
fn wrong_model_with_conflicting_max_tokens_alias_rejects_as_model_not_found_on_lattice_profile() {
    // Regression: the `lattice` profile's original (pre-shared-contract)
    // precedence checked the max_tokens/max_completion_tokens conflict
    // inside `validate_max_tokens`, which ran AFTER the model-id check
    // -- unlike the daemon profile above. A wrong model combined with a
    // conflicting alias pair must still surface `model_not_found` on
    // this profile.
    let req = request(
        r#"{"model":"wrong-model","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
    );
    assert_eq!(
        normalize_request(&req, defaults(), ServeProfile::lattice("served-model", 32),)
            .unwrap_err()
            .code(),
        "model_not_found"
    );
}

#[test]
fn lattice_profile_still_rejects_genuinely_unsupported_fields() {
    // Control: tools/tool_choice remain a hard 400 on both profiles —
    // the restored tolerance is scoped to exactly the three named
    // fields, not a blanket "accept everything" relaxation.
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}"#,
    );
    assert_eq!(
        normalize_request(&req, defaults(), ServeProfile::lattice("model", 32),)
            .unwrap_err()
            .code(),
        "unsupported_feature"
    );
}

#[test]
fn unmodeled_openai_top_level_fields_are_ignored_not_rejected() {
    let req = serde_json::from_str::<ChatRequest>(
        r#"{
                "model": "model",
                "messages": [{"role": "user", "content": "hi"}],
                "presence_penalty": 1.0,
                "frequency_penalty": 0.5,
                "logit_bias": {"123": -100},
                "user": "end-user-id"
            }"#,
    )
    .expect("unmodeled top-level fields must be ignored, not rejected");
    assert_eq!(req.model.as_deref(), Some("model"));
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn context_window_accepts_boundary_and_rejects_overflow() {
    // #831: the shared formula reserves one delimiter slot
    // (`prompt + max_tokens + reasoning_budget + 1 <= max_context`), so
    // the exact boundary now needs one fewer prompt/decode token than
    // `max_context` to leave room for that reserved slot.
    validate_context_window(7, 8, 16).unwrap();
    assert_eq!(
        validate_context_window(8, 8, 16).unwrap_err().code(),
        "context_length_exceeded"
    );
}

#[test]
fn context_window_accounts_for_reasoning_budget() {
    // Mutation-sensitive: dropping the `+ reasoning_budget` term from
    // `validate_context_window_with_budget`'s formula would accept this
    // request (7 + 8 + 1 == 16), rather than rejecting it for the 4
    // extra reasoning tokens (7 + 8 + 4 + 1 == 20 > 16).
    validate_context_window_with_budget(7, 8, Some(0), 16).unwrap();
    assert_eq!(
        validate_context_window_with_budget(7, 8, Some(4), 16)
            .unwrap_err()
            .code(),
        "context_length_exceeded"
    );
    validate_context_window_with_budget(3, 8, Some(4), 16).unwrap();
}

#[test]
fn context_window_error_message_names_reasoning_budget_on_reasoning_only_overflow() {
    // prompt + max_tokens alone fit (3 + 4 + 1 == 8 <= 16), so this
    // request is only rejected because of the reasoning budget -- the
    // error text must say so, not just blame prompt/max_tokens (#1324
    // review finding 4).
    validate_context_window_with_budget(3, 4, None, 16).unwrap();
    let err = validate_context_window_with_budget(3, 4, Some(10), 16).unwrap_err();
    assert_eq!(err.code(), "context_length_exceeded");
    let message = err.message();
    assert!(
        message.contains("reasoning_budget (10)"),
        "error message must name the reasoning budget: {message}"
    );
    assert!(
        message.contains("18 tokens required"),
        "error message must name the total required tokens: {message}"
    );
}

#[test]
fn optional_exact_rejects_explicit_empty_model() {
    let err = validate_model_name(Some(""), ModelNamePolicy::OptionalExact("served")).unwrap_err();
    assert_eq!(err.code(), "model_not_found");
}

#[test]
fn optional_exact_accepts_omitted_model() {
    validate_model_name(None, ModelNamePolicy::OptionalExact("served")).unwrap();
}

#[test]
fn optional_exact_matches_and_rejects_by_name() {
    validate_model_name(Some("served"), ModelNamePolicy::OptionalExact("served")).unwrap();
    assert_eq!(
        validate_model_name(Some("other"), ModelNamePolicy::OptionalExact("served"))
            .unwrap_err()
            .code(),
        "model_not_found"
    );
}

#[test]
fn required_exact_rejects_missing_or_empty_and_accepts_match() {
    assert_eq!(
        validate_model_name(None, ModelNamePolicy::RequiredExact("served"))
            .unwrap_err()
            .code(),
        "invalid_request"
    );
    assert_eq!(
        validate_model_name(Some(""), ModelNamePolicy::RequiredExact("served"))
            .unwrap_err()
            .code(),
        "invalid_request"
    );
    validate_model_name(Some("served"), ModelNamePolicy::RequiredExact("served")).unwrap();
}

fn message(body: &str) -> Message {
    serde_json::from_str(body).unwrap()
}

// Production traffic on both servers normalizes messages through this
// contract-owned representation before one shared adapter crosses into
// the engine type.
#[test]
fn normalize_messages_accepts_system_user_assistant_roles() {
    let messages = [
        message(r#"{"role":"system","content":"sys"}"#),
        message(r#"{"role":"user","content":"usr"}"#),
        message(r#"{"role":"assistant","content":"asst"}"#),
    ];
    let normalized = normalize_messages(&messages).unwrap();
    assert_eq!(normalized[0].role, NormalizedChatRole::System);
    assert_eq!(normalized[0].content, "sys");
    assert_eq!(normalized[1].role, NormalizedChatRole::User);
    assert_eq!(normalized[1].content, "usr");
    assert_eq!(normalized[2].role, NormalizedChatRole::Assistant);
    assert_eq!(normalized[2].content, "asst");
}

#[test]
fn normalize_messages_rejects_unrecognized_role() {
    let messages = [message(r#"{"role":"moderator","content":"hi"}"#)];
    assert_eq!(
        normalize_messages(&messages).unwrap_err().code(),
        "invalid_role"
    );
}

#[test]
fn normalize_messages_rejects_known_but_unsupported_roles() {
    for role in ["tool", "developer"] {
        let messages = [message(&format!(r#"{{"role":"{role}","content":"hi"}}"#))];
        assert_eq!(
            normalize_messages(&messages).unwrap_err().code(),
            "unsupported_feature"
        );
    }
}

#[test]
fn normalize_messages_checks_content_before_role() {
    // Contract choice, deliberately kept rather than restored:
    // `normalize_messages` validates content before role, matching
    // `lattice.rs`'s original pre-shared-contract order.
    // `lattice_serve.rs`'s original order was the reverse (role before
    // content); the shared normalizer now applies content-before-role
    // uniformly on both server profiles. A message with both an invalid
    // role and unsupported (image) content surfaces the content error.
    let messages = [message(
        r#"{"role":"moderator","content":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}"#,
    )];
    let err = normalize_messages(&messages).unwrap_err();
    assert_eq!(err.message(), "image input requires a vision-capable model");
}

#[test]
fn honoring_profile_treats_explicit_null_sampling_fields_as_absent() {
    // Regression (null-handling): an explicit JSON `null` for an honored
    // extension field must default, same as an omitted field, not
    // reject. Also pins that the `Option<Value>` -> `Option<Box<RawValue>>`
    // field-type change preserves serde's existing
    // null-maps-to-`None` `Option<T>` handling -- `parse_ignorable_field`
    // never even runs for these three fields on a `"field": null`
    // request, since the value is `None` by the time it reaches there.
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":null,"repetition_penalty":null,"reasoning_budget":null}"#,
    );
    let daemon =
        normalize_request(&req, defaults(), ServeProfile::lattice_serve("model", 100)).unwrap();
    assert_eq!(daemon.top_k, defaults().top_k);
    assert_eq!(daemon.repetition_penalty, defaults().repetition_penalty);
    assert_eq!(daemon.reasoning_budget, None);
}

#[test]
fn honoring_profile_rejects_near_body_cap_nested_extension_field() {
    // Regression: a near-1MiB nested array submitted for an
    // honored extension field must be rejected via the cheap first-byte
    // span check in `parse_ignorable_field`, never via `from_str` on the
    // whole span and never via building a `Value` tree -- `RawValue`'s
    // `Deserialize` impl only records the span during the outer
    // `ChatRequest` parse, it does not recurse into it.
    let big_array = format!("[{}]", vec!["1"; REQUEST_BODY_LIMIT_BYTES / 2].join(","));
    let body = format!(
        r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"top_k":{big_array}}}"#
    );
    let req = request(&body);
    let err =
        normalize_request(&req, defaults(), ServeProfile::lattice_serve("model", 32)).unwrap_err();
    assert_eq!(err.code(), "invalid_request_body");
    assert_eq!(err.message(), "top_k must be a scalar value");
}

#[test]
fn non_honoring_profile_ignores_near_body_cap_nested_extension_field() {
    // Regression: the same near-1MiB nested array on a
    // profile that does not honor the field must be silently ignored,
    // not rejected -- `parse_ignorable_field` is never even called for
    // it (see the `else { defaults.top_k }` branch in
    // `normalize_request`), so the span is never inspected at all.
    let big_array = format!("[{}]", vec!["1"; REQUEST_BODY_LIMIT_BYTES / 2].join(","));
    let body = format!(
        r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"top_k":{big_array}}}"#
    );
    let req = request(&body);
    let lattice = normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap();
    assert_eq!(lattice.top_k, defaults().top_k);
}

#[test]
fn lattice_profile_rejects_json_schema_without_materializing_the_schema() {
    // Regression: `JsonSchemaFormat::schema` must stay a `RawValue` span
    // rather than an eagerly-parsed `Value` tree, because the `lattice`
    // profile (`structured_output_supported: false`) always rejects
    // `response_format.type: "json_schema"` via `reject_unsupported`
    // without ever reading `schema`. A schema whose nesting exceeds
    // `serde_json`'s ~128-level recursion limit proves the point
    // structurally: parsing it into `Value` during the outer
    // `ChatRequest` deserialization would itself fail with a recursion
    // error before `normalize_request` ever runs, but `RawValue`
    // (which only records the byte span, never recurses into it) lets
    // deserialization succeed, so the request reaches
    // `reject_unsupported` and is rejected for the right reason:
    // unsupported response-format type, not an accidental parse error
    // one level of the stack never asked to pay for.
    let mut nested = String::new();
    for _ in 0..200 {
        nested.push('[');
    }
    nested.push('1');
    for _ in 0..200 {
        nested.push(']');
    }
    assert!(
        serde_json::from_str::<Value>(&nested).is_err(),
        "fixture must exceed serde_json's Value recursion limit"
    );
    let body = format!(
        r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],
            "response_format":{{"type":"json_schema","json_schema":{{"name":"n","strict":true,"schema":{nested}}}}}}}"#
    );
    let req = request(&body);
    let err = normalize_request(&req, defaults(), ServeProfile::lattice("model", 32)).unwrap_err();
    assert_eq!(err.code(), "unsupported_feature");
    assert_eq!(
        err.message(),
        "response_format.type 'json_schema' is not supported; use 'text'"
    );
}

#[test]
fn valid_request_body_is_parsed_by_a_single_deserialize_call() {
    // Regression: there is no separate raw-bytes preflight pass over
    // `messages` anymore -- `serde_json::from_str::<ChatRequest>` is the
    // only parse a caller needs to run, and it both counts and
    // materializes the array in that one call.
    let messages: Vec<String> = (0..MAX_MESSAGE_COUNT)
        .map(|_| r#"{"role":"user","content":""}"#.to_string())
        .collect();
    let body = format!(r#"{{"model":"model","messages":[{}]}}"#, messages.join(","));
    let req = serde_json::from_str::<ChatRequest>(&body).unwrap();
    assert_eq!(req.messages.len(), MAX_MESSAGE_COUNT);
}

#[test]
fn normalize_messages_passes_through_empty_and_whitespace_content() {
    // Documents the current production contract: normalize_messages
    // does not trim or reject empty/whitespace content — it is passed
    // through unchanged, same as any other text content.
    let messages = [
        message(r#"{"role":"user","content":""}"#),
        message(r#"{"role":"user","content":"   "}"#),
    ];
    let normalized = normalize_messages(&messages).unwrap();
    assert_eq!(normalized[0].content, "");
    assert_eq!(normalized[1].content, "   ");
}

fn requested(body: &str, profile: ServeProfile<'_>) -> RequestedChatOptions {
    normalize_requested_options(&request(body), profile).unwrap()
}

fn effective(body: &str, defaults: GenerationDefaults) -> ValidatedChatRequest {
    normalize_request(
        &request(body),
        defaults,
        ServeProfile::lattice_serve("model", 4096),
    )
    .unwrap()
}

#[test]
fn requested_options_distinguish_absent_from_explicit_default() {
    let defaults = GenerationDefaults::standard(512);
    let profile = ServeProfile::lattice_serve("model", 4096);
    let absent = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let absent_options = requested(absent, profile);
    assert_eq!(absent_options.max_tokens, None);
    assert_eq!(absent_options.temperature, None);
    assert_eq!(absent_options.top_k, None);
    assert_eq!(absent_options.top_p, None);
    assert_eq!(absent_options.repetition_penalty, None);
    assert_eq!(absent_options.seed, None);
    assert_eq!(absent_options.stream, None);
    assert_eq!(absent_options.reasoning_budget, None);
    assert_eq!(absent_options.enable_thinking, None);
    assert_eq!(absent_options.logprobs, None);
    assert_eq!(absent_options.top_logprobs, None);

    // Each option sent with the value its default would have supplied:
    // the requested options keep it, and the defaults step resolves the
    // explicit and the omitted form to the same effective request.
    type IsExplicit = fn(&RequestedChatOptions) -> bool;
    let cases: [(&str, IsExplicit); 10] = [
        (r#""max_tokens":512"#, |o| o.max_tokens == Some(512)),
        (r#""max_completion_tokens":512"#, |o| {
            o.max_tokens == Some(512)
        }),
        (r#""temperature":0.7"#, |o| o.temperature == Some(0.7)),
        (r#""top_k":50"#, |o| o.top_k == Some(50)),
        (r#""top_p":0.9"#, |o| o.top_p == Some(0.9)),
        (r#""repetition_penalty":1.1"#, |o| {
            o.repetition_penalty == Some(1.1)
        }),
        (r#""stream":false"#, |o| o.stream == Some(false)),
        (r#""reasoning_budget":0"#, |o| o.reasoning_budget == Some(0)),
        (r#""chat_template_kwargs":{"enable_thinking":true}"#, |o| {
            o.enable_thinking == Some(true)
        }),
        (r#""logprobs":false"#, |o| o.logprobs == Some(false)),
    ];
    let absent_effective = effective(absent, defaults);
    for (field, is_explicit) in cases {
        let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{field}}}"#);
        let options = requested(&body, profile);
        assert!(is_explicit(&options), "{field} must be kept as sent");
        let explicit_effective = effective(&body, defaults);
        assert_eq!(
            format!("{explicit_effective:?}"),
            format!("{absent_effective:?}"),
            "{field} must resolve to the default's effective value"
        );
    }
}

#[test]
fn requested_options_distinguish_absent_from_explicit_default_top_logprobs() {
    let profile = ServeProfile::lattice("model", 4096);
    let absent = requested(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"logprobs":true}"#,
        profile,
    );
    let explicit = requested(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"logprobs":true,"top_logprobs":0}"#,
        profile,
    );
    assert_eq!(absent.top_logprobs, None);
    assert_eq!(explicit.top_logprobs, Some(0));
    let defaults = QwenChatDefaults::new(GenerationDefaults::standard(16));
    assert_eq!(defaults.apply(absent).unwrap().logprobs, Some(0));
    assert_eq!(defaults.apply(explicit).unwrap().logprobs, Some(0));
}

#[test]
fn requested_options_keep_values_the_defaults_step_rewrites() {
    // A context-clamped max_tokens and a reasoning budget are kept as sent;
    // the defaults step applies the clamp.
    let profile = ServeProfile::lattice_serve("model", 64);
    let body =
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":500,"reasoning_budget":40}"#;
    let options = requested(body, profile);
    assert_eq!(options.max_tokens, Some(500));
    assert_eq!(options.reasoning_budget, Some(40));
    let validated = QwenChatDefaults::new(GenerationDefaults::standard(16))
        .apply(options)
        .unwrap();
    assert_eq!(validated.max_tokens, 63);
    assert_eq!(validated.reasoning_budget, None);

    // Options a profile ignores are not reported as requested.
    let ignored = requested(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"top_k":7,"repetition_penalty":1.5}"#,
        ServeProfile::lattice("model", 4096),
    );
    assert_eq!(ignored.top_k, None);
    assert_eq!(ignored.repetition_penalty, None);
}

#[test]
fn requested_options_leave_default_dependent_refusals_to_the_defaults_step() {
    // An out-of-range default refuses only once the defaults step runs.
    let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let profile = ServeProfile::lattice_serve("model", 4096);
    let options = requested(body, profile);
    assert_eq!(options.temperature, None);
    let invalid = GenerationDefaults {
        temperature: 3.0,
        ..GenerationDefaults::standard(16)
    };
    let err = QwenChatDefaults::new(invalid).apply(options).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_temperature",
            ..
        }
    ));
    // An explicit out-of-range value is refused without any defaults.
    let err = normalize_requested_options(
        &request(r#"{"messages":[{"role":"user","content":"hi"}],"temperature":3.0}"#),
        profile,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_temperature",
            ..
        }
    ));
}

#[test]
fn chat_template_thinking_options_preserve_absence_true_false() {
    for profile in [
        ServeProfile::lattice("model", 100),
        ServeProfile::lattice_serve("model", 100),
    ] {
        for (field, expected) in [
            ("", None),
            (r#", "chat_template_kwargs":{}"#, None),
            (
                r#", "chat_template_kwargs":{"enable_thinking":true}"#,
                Some(true),
            ),
            (
                r#", "chat_template_kwargs":{"enable_thinking":false}"#,
                Some(false),
            ),
        ] {
            let body = format!(
                r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}]{field}}}"#
            );
            let options = requested(&body, profile);
            assert_eq!(options.enable_thinking, expected, "field={field}");
        }
    }
}

#[test]
fn chat_template_thinking_defaults_preserve_qwen_default() {
    for profile in [
        ServeProfile::lattice("model", 100),
        ServeProfile::lattice_serve("model", 100),
    ] {
        for (field, expected) in [
            ("", true),
            (r#", "chat_template_kwargs":{}"#, true),
            (r#", "chat_template_kwargs":{"enable_thinking":true}"#, true),
            (
                r#", "chat_template_kwargs":{"enable_thinking":false}"#,
                false,
            ),
        ] {
            let body = format!(
                r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}]{field}}}"#
            );
            let (_, _, enable_thinking) = normalize_request_with_context_and_thinking(
                &request(&body),
                defaults(),
                profile,
                |_, _, _, _| Ok(String::new()),
            )
            .unwrap();
            assert_eq!(enable_thinking, expected, "field={field}");
        }
    }
}

#[test]
fn chat_template_kwargs_reject_unknown_keys() {
    for kwargs in [
        r#"{"unknown":true}"#,
        r#"{"enable_thinking":false,"unknown":[1,2,3]}"#,
    ] {
        let req = request(&format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"chat_template_kwargs":{kwargs}}}"#
        ));
        for profile in [
            ServeProfile::lattice("model", 100),
            ServeProfile::lattice_serve("model", 100),
        ] {
            let error = normalize_requested_options(&req, profile).unwrap_err();
            assert_eq!(error.code(), "invalid_request_body");
            assert!(error.message().contains("unknown field"));
            assert!(error.message().contains("unknown"));
        }
    }
}

#[test]
fn chat_template_kwargs_reject_non_object_values() {
    for kwargs in ["null", "true", "false", "0", r#""false""#, "[]", "[false]"] {
        let req = request(&format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"chat_template_kwargs":{kwargs}}}"#
        ));
        for profile in [
            ServeProfile::lattice("model", 100),
            ServeProfile::lattice_serve("model", 100),
        ] {
            let error = normalize_requested_options(&req, profile).unwrap_err();
            assert_eq!(error.code(), "invalid_request_body", "kwargs={kwargs}");
            assert!(error.message().contains("chat_template_kwargs"));
        }
    }
}

#[test]
fn chat_template_thinking_reject_non_boolean_values() {
    for value in ["null", "0", "1", r#""false""#, "[]", "{}", "[false]"] {
        let req = request(&format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"chat_template_kwargs":{{"enable_thinking":{value}}}}}"#
        ));
        for profile in [
            ServeProfile::lattice("model", 100),
            ServeProfile::lattice_serve("model", 100),
        ] {
            let error = normalize_requested_options(&req, profile).unwrap_err();
            assert_eq!(error.code(), "invalid_request_body", "value={value}");
            assert!(error.message().contains("expected a boolean"));
        }
    }
}

#[test]
fn chat_template_thinking_reject_duplicate_keys() {
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false,"enable_thinking":true}}"#,
    );
    for profile in [
        ServeProfile::lattice("model", 100),
        ServeProfile::lattice_serve("model", 100),
    ] {
        let error = normalize_requested_options(&req, profile).unwrap_err();
        assert_eq!(error.code(), "invalid_request_body");
        assert!(error.message().contains("duplicate field"));
    }
}

#[test]
fn requested_thinking_option_is_model_family_neutral() {
    for model in ["qwen", "gemma"] {
        for enabled in [true, false] {
            let body = format!(
                r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"chat_template_kwargs":{{"enable_thinking":{enabled}}}}}"#
            );
            let options = requested(&body, ServeProfile::lattice(model, 100));
            assert_eq!(options.enable_thinking, Some(enabled));
        }
    }
}

#[test]
fn thinking_off_makes_reasoning_budget_inert_in_context_callback() {
    for budget in [
        "",
        r#", "reasoning_budget":0"#,
        r#", "reasoning_budget":40"#,
    ] {
        let req = request(&format!(
            r#"{{"model":"model","messages":[{{"role":"user","content":"hi"}}],"chat_template_kwargs":{{"enable_thinking":false}}{budget}}}"#
        ));
        let generation = GenerationDefaults {
            reasoning_budget: Some(20),
            ..defaults()
        };
        let (validated, prompt, enable_thinking) = normalize_request_with_context_and_thinking(
            &req,
            generation,
            ServeProfile::lattice("model", 100),
            |_, max_tokens, effective_budget, enabled| {
                assert!(!enabled);
                assert_eq!(effective_budget, None);
                validate_context_window_with_budget(1, max_tokens, effective_budget, 18)?;
                Ok("prepared".to_owned())
            },
        )
        .unwrap();
        assert!(!enable_thinking);
        assert_eq!(validated.reasoning_budget, None);
        assert_eq!(prompt, "prepared");
    }
}

#[test]
fn legacy_context_callbacks_remain_compatible_when_thinking_is_off() {
    let req = request(
        r#"{"model":"model","messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false},"reasoning_budget":40}"#,
    );
    let profile = ServeProfile::lattice("model", 100);
    let (validated, _) =
        normalize_request_with_context_and_budget(&req, defaults(), profile, |_, _, budget| {
            assert_eq!(budget, None);
            Ok("three arguments".to_owned())
        })
        .unwrap();
    assert_eq!(validated.reasoning_budget, None);
    let (validated, _) = normalize_request_with_context(&req, defaults(), profile, |_, _| {
        Ok("two arguments".to_owned())
    })
    .unwrap();
    assert_eq!(validated.reasoning_budget, None);
}
