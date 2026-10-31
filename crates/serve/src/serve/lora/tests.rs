use super::*;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn write_adapter_with_unknown_modules(path: &std::path::Path, unknown_modules: usize) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut modules = vec!["q_proj".to_owned()];
    modules.extend((0..unknown_modules).map(|index| format!("unknown_{index:02}")));
    for module in modules {
        for matrix in ["A", "B"] {
            let start = payload.len();
            payload.extend_from_slice(&1.0f32.to_le_bytes());
            header.insert(
                format!("base_model.model.model.layers.0.self_attn.{module}.lora_{matrix}.weight"),
                serde_json::json!({
                    "dtype": "F32", "shape": [1, 1], "data_offsets": [start, payload.len()]
                }),
            );
        }
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload);
    std::fs::write(path, bytes).unwrap();
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn prepare_adapter_rejects_skipped_factor_tensors() {
    use axum::response::IntoResponse;

    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("adapter.safetensors");
    write_adapter_with_unknown_modules(&file, 1);
    let error = match prepare_adapter_load(file.to_str().unwrap(), "adapter") {
        Ok(_) => panic!("adapter with skipped factors must be refused"),
        Err(error) => error,
    };
    let ApiError::BadRequest { ref message, code } = error else {
        panic!("expected a bad request");
    };
    assert_eq!(code, "lora_load_failed");
    assert!(message.contains("self_attn.unknown_00.lora_A.weight"));
    assert!(message.contains("self_attn.unknown_00.lora_B.weight"));
    assert_eq!(
        error.into_response().status(),
        axum::http::StatusCode::BAD_REQUEST
    );

    write_adapter_with_unknown_modules(&file, 0);
    let prepared = prepare_adapter_load(file.to_str().unwrap(), "adapter").unwrap();
    assert_eq!(prepared.rank, 1);
    assert_eq!(prepared.layers, 1);
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn prepare_adapter_reports_sorted_first_eight_skipped_tensors() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("adapter.safetensors");
    write_adapter_with_unknown_modules(&file, 5);
    let error = match prepare_adapter_load(file.to_str().unwrap(), "adapter") {
        Ok(_) => panic!("adapter with skipped factors must be refused"),
        Err(error) => error,
    };
    let ApiError::BadRequest { message, code } = error else {
        panic!("expected a bad request");
    };
    assert_eq!(code, "lora_load_failed");
    let names: Vec<_> = (0..4)
        .flat_map(|index| {
            ["A", "B"].map(move |matrix| {
                format!("base_model.model.model.layers.0.self_attn.unknown_{index:02}.lora_{matrix}.weight")
            })
        })
        .collect();
    assert_eq!(
        message,
        format!(
            "adapter has factor tensors for modules this model does not support: {}, and 2 more",
            names.join(", ")
        )
    );
}

/// A bootstrap, a learned gate and a legacy one are three different
/// answers on this surface, and the responses they produce are identical.
/// The legacy arm is the one that matters: reporting it as `"bootstrap"`
/// would describe a possibly-trained gate as a deliberate baseline, which
/// is wrong in the reassuring direction.
#[test]
fn the_lora_index_reports_how_the_serving_gate_was_produced() {
    use lattice_inference::router_state::{
        RouterArtifact, RouterLineage, RouterPolicy, RouterProvenance, TrainedRepresentation,
    };

    let index = AdapterIndex {
        adapters: Vec::new(),
        applied: Vec::new(),
        blend_refusal: None,
    };
    let representation = TrainedRepresentation {
        embedding_model: "qwen3.5-0.8b".into(),
        pooling: "mean_visual".into(),
        prompt_source: "last_user_message".into(),
        loader_format: "qwen35-f16-decoder".into(),
        input_width: 8,
    };
    let policy = |training_state: &str| RouterPolicy {
        kind: "uniform".into(),
        tau: None,
        epsilon: 0.0,
        tau_floor: 0.0,
        entropy_floor: 0.0,
        max_consecutive_collapsed: 0,
        training_state: training_state.into(),
        objective_version: None,
        exploration_scale: None,
    };
    let lineage = RouterLineage {
        schema_id: "schema-1".into(),
        base_model: "qwen3.5-0.8b".into(),
        parent: None,
        admission_record: None,
    };
    let artifact = |provenance| RouterArtifact {
        version: 1,
        adapter_names: vec!["technical".into()],
        representation: representation.clone(),
        gate_bytes: vec![1, 2, 3],
        provenance,
    };

    for (provenance, expected) in [
        (
            RouterProvenance::recorded(policy("bootstrap"), lineage.clone()),
            "bootstrap",
        ),
        (
            RouterProvenance::recorded(policy("learned"), lineage.clone()),
            "learned",
        ),
        (RouterProvenance::Legacy, "unrecorded"),
    ] {
        let served = artifact(provenance);
        let resolved = lattice_inference::router_state::ResolvedRouter {
            artifact: served,
            pinned: false,
        };
        let body = lora_list_body(
            &index,
            Some(resolved.report(
                "qwen3.5-0.8b",
                lattice_inference::router_state::AppliedPolicy {
                    kind: "uniform".to_string(),
                    tau: None,
                    epsilon: 0.0,
                    exploration_scale: None,
                    learning_enabled: false,
                },
            )),
        );
        assert_eq!(
            body["router"]["training_state"], expected,
            "the three provenances must not report the same thing"
        );
    }
}

/// The body says what requests are served under, not what the file records.
/// A server that declined to enable a recorded learned policy reports the
/// uniform one it is running, and this is the only surface an operator has
/// for telling the two apart during a qualification run.
#[test]
fn the_body_reports_the_policy_in_force() {
    let index = AdapterIndex {
        adapters: Vec::new(),
        applied: Vec::new(),
        blend_refusal: None,
    };
    use lattice_inference::router_state::{
        RouterArtifact, RouterLineage, RouterPolicy, RouterProvenance, TrainedRepresentation,
    };

    let resolved = lattice_inference::router_state::ResolvedRouter {
        artifact: RouterArtifact {
            version: 1,
            adapter_names: vec!["technical".into()],
            representation: TrainedRepresentation {
                embedding_model: "qwen3.5-0.8b".into(),
                pooling: "mean_visual".into(),
                prompt_source: "last_user_message".into(),
                loader_format: "qwen35-f16-decoder".into(),
                input_width: 8,
            },
            gate_bytes: vec![1, 2, 3],
            provenance: RouterProvenance::recorded(
                RouterPolicy {
                    kind: "softmax".into(),
                    tau: Some(0.7),
                    epsilon: 0.05,
                    tau_floor: 0.1,
                    entropy_floor: 0.35,
                    max_consecutive_collapsed: 3,
                    training_state: "learned".into(),
                    objective_version: Some("whole-mixture-v1".into()),
                    exploration_scale: Some(0.5),
                },
                RouterLineage {
                    schema_id: "schema-1".into(),
                    base_model: "qwen3.5-0.8b".into(),
                    parent: None,
                    admission_record: None,
                },
            ),
        },
        pinned: false,
    };
    let body = lora_list_body(
        &index,
        Some(resolved.report(
            "qwen3.5-0.8b",
            lattice_inference::router_state::AppliedPolicy {
                kind: "softmax".to_string(),
                tau: Some(0.7),
                epsilon: 0.05,
                exploration_scale: Some(0.5),
                learning_enabled: true,
            },
        )),
    );
    assert_eq!(body["router"]["policy"]["kind"], "softmax");
    assert_eq!(body["router"]["policy"]["tau"], 0.7);
    assert_eq!(body["router"]["policy"]["exploration_scale"], 0.5);
    assert_eq!(body["router"]["policy"]["learning_enabled"], true);
    // The control that makes the four above about the policy and not about
    // the fixture: a deterministic server reports its own values under the
    // same keys, and they differ.
    let deterministic = lora_list_body(
        &index,
        Some(resolved.report(
            "qwen3.5-0.8b",
            lattice_inference::router_state::AppliedPolicy {
                kind: "uniform".to_string(),
                tau: None,
                epsilon: 0.0,
                exploration_scale: None,
                learning_enabled: false,
            },
        )),
    );
    assert_eq!(deterministic["router"]["policy"]["kind"], "uniform");
    assert_eq!(deterministic["router"]["policy"]["learning_enabled"], false);
}

#[test]
fn duplicate_ids_are_rejected_before_admission() {
    let index = AdapterIndex {
        adapters: [7, 11]
            .into_iter()
            .map(|id| AdapterMetadata {
                id,
                name: format!("adapter-{id}"),
                path: format!("{id}.safetensors"),
                rank: 1,
                layers: 1,
            })
            .collect(),
        applied: Vec::new(),
        ..Default::default()
    };
    let unique = [
        LoraSelection { id: 7, scale: 0.0 },
        LoraSelection {
            id: 11,
            scale: -0.5,
        },
    ];
    assert!(index.validate(&unique).is_ok());
    assert!(index.validate(&[]).is_ok());
    for scale in [0.0, 1.0, -0.25] {
        let selection = [unique[0], unique[1], LoraSelection { id: 7, scale }];
        let error = index.validate(&selection).unwrap_err();
        assert!(matches!(error, ApiError::BadRequest { .. }));
        assert_eq!(error.code(), "lora_duplicate_adapter_id");
        assert_eq!(error.message(), "duplicate LoRA adapter id 7");
    }
}

#[test]
fn selections_reject_nonfinite_scales_and_unknown_ids() {
    let index = AdapterIndex::default();
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = index
            .validate(&[LoraSelection { id: 42, scale }])
            .unwrap_err();
        assert!(error.message().contains("finite"));
    }
    let error = index
        .validate(&[LoraSelection { id: 42, scale: 1.0 }])
        .unwrap_err();
    assert!(matches!(error, ApiError::BadRequest { .. }));
    assert!(error.message().contains("42"));
    assert!(index.validate(&[]).is_ok());
}

#[test]
fn load_name_and_unload_id_are_strictly_parsed() {
    assert_eq!(
        parse_lora_load(br#"{"path":"a","name":"display"}"#).unwrap(),
        ("a".into(), "display".into())
    );
    assert_eq!(
        parse_lora_load(br#"{"path":"a"}"#).unwrap(),
        ("a".into(), "a".into())
    );
    assert!(parse_lora_load(br#"{"path":"a","name":" "}"#).is_err());
    assert_eq!(
        parse_lora_unload(br#"{"id":4294967295}"#).unwrap(),
        u32::MAX
    );
    for invalid in [
        br#"{}"#.as_slice(),
        br#"{"id":-1}"#,
        br#"{"id":4294967296}"#,
        br#"{"id":0,"extra":1}"#,
    ] {
        assert!(parse_lora_unload(invalid).is_err());
    }
}

fn code_of(err: ApiError) -> &'static str {
    match err {
        ApiError::BadRequest { code, .. } => code,
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn a_well_formed_body_yields_the_path() {
    assert_eq!(
        parse_lora_load_path(br#"{"path":"/tmp/a.safetensors"}"#).unwrap(),
        "/tmp/a.safetensors"
    );
}

#[test]
fn a_missing_path_is_refused() {
    assert_eq!(
        code_of(parse_lora_load_path(br#"{}"#).unwrap_err()),
        "invalid_request"
    );
}

/// The point of the rejection is that the value is not silently dropped: a
/// caller asking for a scale this route does not apply must be told so, not
/// served an adapter at the file's own scale as if the field had been honored.
#[test]
fn an_unknown_field_is_refused_rather_than_ignored() {
    let err = parse_lora_load_path(br#"{"path":"/tmp/a.safetensors","scale":2.0}"#).unwrap_err();
    let ApiError::BadRequest { message, code } = err else {
        panic!("expected BadRequest");
    };
    assert_eq!(code, "invalid_request");
    assert!(
        message.contains("scale"),
        "message must name the field: {message}"
    );
}

#[test]
fn a_whitespace_path_is_refused() {
    assert_eq!(
        code_of(parse_lora_load_path(br#"{"path":"   "}"#).unwrap_err()),
        "invalid_request"
    );
}

#[test]
fn a_non_object_body_is_refused() {
    assert_eq!(
        code_of(parse_lora_load_path(br#"["/tmp/a.safetensors"]"#).unwrap_err()),
        "invalid_request"
    );
}

#[test]
fn a_worker_failure_separates_unknown_id_from_a_bad_adapter() {
    assert_eq!(
        adapter_failure_code(&AdapterControlError::NotFound(7)),
        "lora_adapter_not_found"
    );
    assert_eq!(
        adapter_failure_code(&AdapterControlError::InvalidAdapter(
            "unknown LoRA adapter id 7".into()
        )),
        "lora_load_failed"
    );
}

#[test]
fn the_success_bodies_name_the_object_and_the_state() {
    let loaded = load_success_body(0, "a", "/tmp/a.safetensors", 8, 24);
    assert_eq!(loaded["object"], "lora.adapter");
    assert_eq!(loaded["status"], "loaded");
    assert_eq!(loaded["rank"], 8);
    assert_eq!(loaded["layers"], 24);
    assert_eq!(unload_success_body(0)["status"], "unloaded");
}
