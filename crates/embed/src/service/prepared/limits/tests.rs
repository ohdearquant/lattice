use super::*;
use std::num::{NonZeroU64, NonZeroUsize};

type CeilingGroups = (
    InventoryCeilings,
    TensorCeilings,
    BertCeilings,
    ParseCeilings,
    ResourceCeilings,
);

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn nz64(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn build(groups: CeilingGroups) -> LimitResult<PreparationCeilings> {
    PreparationCeilings::try_new(groups.0, groups.1, groups.2, groups.3, groups.4)
}

fn build_with_platform_max(
    groups: CeilingGroups,
    platform_max: u64,
) -> LimitResult<PreparationCeilings> {
    PreparationCeilings::try_new_with_platform_max(
        groups.0,
        groups.1,
        groups.2,
        groups.3,
        groups.4,
        platform_max,
    )
}

fn distinct_groups() -> CeilingGroups {
    (
        InventoryCeilings::new(
            nz(2),
            nz(3),
            nz64(5),
            nz64(43),
            nz64(47),
            nz64(59),
            nz64(61),
            nz64(67),
            nz(7),
        ),
        TensorCeilings::new(nz64(41), nz(11), nz(13), nz(17), nz(19), nz64(71), nz64(73)),
        BertCeilings {
            max_vocab_size: nz(79),
            max_hidden_size: nz(83),
            max_hidden_layers: nz(89),
            max_attention_heads: nz(97),
            max_intermediate_size: nz(101),
            max_position_embeddings: nz(103),
            max_type_vocab_size: nz(107),
        },
        ParseCeilings {
            max_config_bytes: nz64(23),
            max_tokenizer_file_bytes: nz64(29),
            max_config_parse_work_bytes: nz64(109),
            max_tokenizer_parse_work_bytes: nz64(113),
            max_safetensors_parse_work_bytes: nz64(127),
        },
        ResourceCeilings {
            max_model_retained_bytes: nz64(131),
            max_tokenizer_retained_bytes: nz64(137),
            max_descriptor_control_bytes: nz64(139),
            max_model_load_work_bytes: nz64(149),
        },
    )
}

fn geometry_ceilings(
    max_tensors: usize,
    max_elements: u64,
    bert: BertCeilings,
) -> PreparationCeilings {
    build((
        InventoryCeilings::new(
            nz(32),
            nz(32),
            nz64(256),
            nz64(256),
            nz64(256),
            nz64(512),
            nz64(512),
            nz64(256),
            nz(32),
        ),
        TensorCeilings::new(
            nz64(64),
            nz(max_tensors),
            nz(128),
            nz(8),
            nz(8_192),
            nz64(max_elements),
            nz64(1_024),
        ),
        bert,
        ParseCeilings::all(nz64(32)),
        ResourceCeilings::all(nz64(32)),
    ))
    .unwrap()
}

fn baseline_facts() -> CandidateBertFacts {
    CandidateBertFacts {
        vocab_size: 3,
        hidden_size: 2,
        hidden_layers: 1,
        attention_heads: 1,
        intermediate_size: 4,
        position_embeddings: 2,
        type_vocab_size: 1,
        layer_norm_eps: 1e-12,
        has_pooler: false,
    }
}

fn with_bert_axis(
    mut facts: CandidateBertFacts,
    axis: LimitAxis,
    value: u64,
) -> CandidateBertFacts {
    match axis {
        LimitAxis::VocabSize => facts.vocab_size = value,
        LimitAxis::HiddenSize => facts.hidden_size = value,
        LimitAxis::HiddenLayers => facts.hidden_layers = value,
        LimitAxis::AttentionHeads => facts.attention_heads = value,
        LimitAxis::IntermediateSize => facts.intermediate_size = value,
        LimitAxis::PositionEmbeddings => facts.position_embeddings = value,
        LimitAxis::TypeVocabSize => facts.type_vocab_size = value,
        _ => panic!("not a BERT geometry axis"),
    }
    facts
}

fn census_ceilings(
    max_tensors: usize,
    max_name: usize,
    max_rank: usize,
    max_dimension: usize,
    max_elements: u64,
    max_metadata: u64,
) -> PreparationCeilings {
    build((
        InventoryCeilings::new(
            nz(8),
            nz(8),
            nz64(16),
            nz64(16),
            nz64(16),
            nz64(32),
            // See the matching comment in `test_tensor_inventory_ceilings`:
            // `TensorCensus::push` bounds the running declared-payload
            // total against this axis, and this helper's tensor tests
            // push several tensors whose summed bytes can exceed the
            // placeholder `32` the sibling file-size axes use.
            nz64(1_000_000),
            nz64(16),
            nz(8),
        ),
        TensorCeilings::new(
            nz64(8),
            nz(max_tensors),
            nz(max_name),
            nz(max_rank),
            nz(max_dimension),
            nz64(max_elements),
            nz64(max_metadata),
        ),
        BertCeilings::all(nz(8)),
        ParseCeilings::all(nz64(8)),
        ResourceCeilings::all(nz64(8)),
    ))
    .unwrap()
}

fn charge_groups() -> CeilingGroups {
    (
        InventoryCeilings::new(
            nz(1),
            nz(1),
            nz64(1),
            nz64(2),
            nz64(4),
            nz64(9),
            nz64(16),
            nz64(128),
            nz(1),
        ),
        TensorCeilings::new(nz64(1), nz(1), nz(1), nz(1), nz(1), nz64(1), nz64(256)),
        BertCeilings::all(nz(1)),
        ParseCeilings {
            max_config_bytes: nz64(1),
            max_tokenizer_file_bytes: nz64(1),
            max_config_parse_work_bytes: nz64(1),
            max_tokenizer_parse_work_bytes: nz64(2),
            max_safetensors_parse_work_bytes: nz64(4),
        },
        ResourceCeilings {
            max_model_retained_bytes: nz64(32),
            max_tokenizer_retained_bytes: nz64(64),
            max_descriptor_control_bytes: nz64(512),
            max_model_load_work_bytes: nz64(8),
        },
    )
}

fn unit_ceilings_unchecked() -> PreparationCeilings {
    PreparationCeilings {
        inventory: InventoryCeilings::new(
            nz(1),
            nz(1),
            nz64(1),
            nz64(1),
            nz64(1),
            nz64(1),
            nz64(1),
            nz64(1),
            nz(1),
        ),
        tensors: TensorCeilings::new(nz64(1), nz(1), nz(1), nz(1), nz(1), nz64(1), nz64(1)),
        bert: BertCeilings::all(nz(1)),
        parsing: ParseCeilings::all(nz64(1)),
        resources: ResourceCeilings::all(nz64(1)),
    }
}

fn set_platform_axis(ceilings: &mut PreparationCeilings, axis: LimitAxis, value: u64) {
    let value = nz64(value);
    match axis {
        LimitAxis::TotalPathBytes => ceilings.inventory.max_total_path_bytes = value,
        LimitAxis::AuxFileBytes => ceilings.inventory.max_aux_file_bytes = value,
        LimitAxis::IndexFileBytes => ceilings.inventory.max_index_file_bytes = value,
        LimitAxis::WeightFileBytes => ceilings.inventory.max_weight_file_bytes = value,
        LimitAxis::SnapshotBytes => ceilings.inventory.max_snapshot_bytes = value,
        LimitAxis::CensusMetadataBytes => {
            ceilings.inventory.max_census_metadata_bytes = value;
        }
        LimitAxis::HeaderBytes => ceilings.tensors.max_header_bytes = value,
        LimitAxis::AggregateElements => ceilings.tensors.max_aggregate_elements = value,
        LimitAxis::TensorMetadataBytes => ceilings.tensors.max_metadata_bytes = value,
        LimitAxis::ConfigBytes => ceilings.parsing.max_config_bytes = value,
        LimitAxis::TokenizerFileBytes => ceilings.parsing.max_tokenizer_file_bytes = value,
        LimitAxis::ConfigParseWorkBytes => {
            ceilings.parsing.max_config_parse_work_bytes = value;
        }
        LimitAxis::TokenizerParseWorkBytes => {
            ceilings.parsing.max_tokenizer_parse_work_bytes = value;
        }
        LimitAxis::SafetensorsParseWorkBytes => {
            ceilings.parsing.max_safetensors_parse_work_bytes = value;
        }
        LimitAxis::ModelRetainedBytes => ceilings.resources.max_model_retained_bytes = value,
        LimitAxis::TokenizerRetainedBytes => {
            ceilings.resources.max_tokenizer_retained_bytes = value;
        }
        LimitAxis::DescriptorControlBytes => {
            ceilings.resources.max_descriptor_control_bytes = value;
        }
        LimitAxis::ModelLoadWorkBytes => ceilings.resources.max_model_load_work_bytes = value,
        _ => panic!("not a platform-sized u64 ceiling"),
    }
}

fn assert_invalid_relation(groups: CeilingGroups, lower: LimitAxis, upper: LimitAxis) {
    assert_eq!(
        build(groups).unwrap_err(),
        PreparationLimitError::InvalidRelation { lower, upper }
    );
}

fn assert_push_error_without_mutation(
    mut census: TensorCensus,
    fact: TensorFact<'_>,
    ceilings: &PreparationCeilings,
    expected: PreparationLimitError,
) {
    let before = census;
    assert_eq!(census.push(fact, ceilings).unwrap_err(), expected);
    assert_eq!(census, before);
}

fn assert_charge_delta(
    mutate: impl FnOnce(&mut CeilingGroups),
    retained_delta: u64,
    work_delta: u64,
) {
    let base = PreparationCharge::worst_case(&build(charge_groups()).unwrap()).unwrap();
    let mut groups = charge_groups();
    mutate(&mut groups);
    let changed = PreparationCharge::worst_case(&build(groups).unwrap()).unwrap();
    assert_eq!(
        changed.retained_bytes(),
        base.retained_bytes() + retained_delta
    );
    assert_eq!(changed.work_bytes(), base.work_bytes() + work_delta);
}

#[test]
fn every_limit_axis_accepts_exact_and_rejects_plus_one() {
    let ceilings = build(distinct_groups()).unwrap();
    let cases: [(LimitAxis, u64); 32] = [
        (LimitAxis::Files, 2),
        (LimitAxis::PathBytes, 3),
        (LimitAxis::TotalPathBytes, 5),
        (LimitAxis::AuxFileBytes, 43),
        (LimitAxis::IndexFileBytes, 47),
        (LimitAxis::WeightFileBytes, 59),
        (LimitAxis::SnapshotBytes, 61),
        (LimitAxis::CensusEntries, 7),
        (LimitAxis::CensusMetadataBytes, 67),
        (LimitAxis::HeaderBytes, 41),
        (LimitAxis::Tensors, 11),
        (LimitAxis::TensorNameBytes, 13),
        (LimitAxis::Rank, 17),
        (LimitAxis::Dimension, 19),
        (LimitAxis::AggregateElements, 71),
        (LimitAxis::TensorMetadataBytes, 73),
        (LimitAxis::VocabSize, 79),
        (LimitAxis::HiddenSize, 83),
        (LimitAxis::HiddenLayers, 89),
        (LimitAxis::AttentionHeads, 97),
        (LimitAxis::IntermediateSize, 101),
        (LimitAxis::PositionEmbeddings, 103),
        (LimitAxis::TypeVocabSize, 107),
        (LimitAxis::ConfigBytes, 23),
        (LimitAxis::TokenizerFileBytes, 29),
        (LimitAxis::ConfigParseWorkBytes, 109),
        (LimitAxis::TokenizerParseWorkBytes, 113),
        (LimitAxis::SafetensorsParseWorkBytes, 127),
        (LimitAxis::ModelRetainedBytes, 131),
        (LimitAxis::TokenizerRetainedBytes, 137),
        (LimitAxis::DescriptorControlBytes, 139),
        (LimitAxis::ModelLoadWorkBytes, 149),
    ];

    for (axis, limit) in cases {
        assert_eq!(ceilings.limit(axis).unwrap(), limit);
        assert_eq!(ceilings.validate(axis, limit), Ok(()));
        assert_eq!(
            ceilings.validate(axis, limit + 1).unwrap_err(),
            PreparationLimitError::Exceeded {
                axis,
                actual: limit + 1,
                limit,
            }
        );
    }
}

#[test]
fn constructor_accepts_relation_equality_and_rejects_plus_one() {
    let mut groups = distinct_groups();
    groups.0.max_path_bytes = nz(5);
    assert!(build(groups).is_ok());
    groups.0.max_path_bytes = nz(6);
    assert_invalid_relation(groups, LimitAxis::PathBytes, LimitAxis::TotalPathBytes);

    let mut groups = distinct_groups();
    groups.0.max_aux_file_bytes = nz64(61);
    assert!(build(groups).is_ok());
    groups.0.max_aux_file_bytes = nz64(62);
    assert_invalid_relation(groups, LimitAxis::AuxFileBytes, LimitAxis::SnapshotBytes);

    let mut groups = distinct_groups();
    groups.0.max_index_file_bytes = nz64(61);
    assert!(build(groups).is_ok());
    groups.0.max_index_file_bytes = nz64(62);
    assert_invalid_relation(groups, LimitAxis::IndexFileBytes, LimitAxis::SnapshotBytes);

    let mut groups = distinct_groups();
    groups.0.max_weight_file_bytes = nz64(61);
    assert!(build(groups).is_ok());
    groups.0.max_weight_file_bytes = nz64(62);
    assert_invalid_relation(groups, LimitAxis::WeightFileBytes, LimitAxis::SnapshotBytes);

    let mut groups = distinct_groups();
    groups.3.max_config_bytes = nz64(43);
    assert!(build(groups).is_ok());
    groups.3.max_config_bytes = nz64(44);
    assert_invalid_relation(groups, LimitAxis::ConfigBytes, LimitAxis::AuxFileBytes);

    let mut groups = distinct_groups();
    groups.3.max_tokenizer_file_bytes = nz64(43);
    assert!(build(groups).is_ok());
    groups.3.max_tokenizer_file_bytes = nz64(44);
    assert_invalid_relation(
        groups,
        LimitAxis::TokenizerFileBytes,
        LimitAxis::AuxFileBytes,
    );

    let mut groups = distinct_groups();
    groups.1.max_header_bytes = nz64(51);
    assert!(build(groups).is_ok());
    groups.1.max_header_bytes = nz64(52);
    assert_invalid_relation(groups, LimitAxis::HeaderBytes, LimitAxis::WeightFileBytes);
}

#[test]
fn constructor_rejects_header_frame_overflow() {
    let mut groups = distinct_groups();
    groups.0.max_weight_file_bytes = nz64(u64::MAX);
    groups.0.max_snapshot_bytes = nz64(u64::MAX);
    groups.1.max_header_bytes = nz64(u64::MAX);
    assert_eq!(
        build(groups).unwrap_err(),
        overflow(ChargeExpression::HeaderFrame)
    );
}

#[test]
fn synthetic_platform_limit_accepts_exact_and_rejects_plus_one() {
    let artificial_max = u64::from(u32::MAX);
    assert_eq!(
        platform_usize_with_max(LimitAxis::SnapshotBytes, artificial_max, artificial_max).unwrap(),
        usize::try_from(artificial_max).unwrap()
    );
    assert_eq!(
        platform_usize_with_max(LimitAxis::SnapshotBytes, artificial_max + 1, artificial_max,)
            .unwrap_err(),
        PreparationLimitError::PlatformUnrepresentable {
            axis: LimitAxis::SnapshotBytes,
            value: artificial_max + 1,
        }
    );
}

#[test]
fn constructor_invokes_the_injected_platform_gate() {
    let mut groups = distinct_groups();
    groups.4.max_model_load_work_bytes = nz64(150);
    assert!(build_with_platform_max(groups, 150).is_ok());
    assert_eq!(
        build_with_platform_max(groups, 149).unwrap_err(),
        PreparationLimitError::PlatformUnrepresentable {
            axis: LimitAxis::ModelLoadWorkBytes,
            value: 150,
        }
    );
}

#[test]
fn constructor_routes_every_u64_allocation_cap_through_platform_check() {
    let axes = [
        LimitAxis::TotalPathBytes,
        LimitAxis::AuxFileBytes,
        LimitAxis::IndexFileBytes,
        LimitAxis::WeightFileBytes,
        LimitAxis::SnapshotBytes,
        LimitAxis::CensusMetadataBytes,
        LimitAxis::HeaderBytes,
        LimitAxis::AggregateElements,
        LimitAxis::TensorMetadataBytes,
        LimitAxis::ConfigBytes,
        LimitAxis::TokenizerFileBytes,
        LimitAxis::ConfigParseWorkBytes,
        LimitAxis::TokenizerParseWorkBytes,
        LimitAxis::SafetensorsParseWorkBytes,
        LimitAxis::ModelRetainedBytes,
        LimitAxis::TokenizerRetainedBytes,
        LimitAxis::DescriptorControlBytes,
        LimitAxis::ModelLoadWorkBytes,
    ];
    assert_eq!(
        unit_ceilings_unchecked().validate_platform_caps_with_max(1),
        Ok(())
    );

    for axis in axes {
        let mut ceilings = unit_ceilings_unchecked();
        set_platform_axis(&mut ceilings, axis, 2);
        assert_eq!(ceilings.validate_platform_caps_with_max(2), Ok(()));
        assert_eq!(
            ceilings.validate_platform_caps_with_max(1).unwrap_err(),
            PreparationLimitError::PlatformUnrepresentable { axis, value: 2 }
        );
    }
}

#[test]
fn candidate_bert_geometry_uses_each_normative_coefficient() {
    let ceilings = geometry_ceilings(1_000, 1_000_000, BertCeilings::all(nz(128)));
    let facts = CandidateBertFacts {
        vocab_size: 3,
        hidden_size: 5,
        hidden_layers: 7,
        attention_heads: 1,
        intermediate_size: 11,
        position_embeddings: 13,
        type_vocab_size: 17,
        layer_norm_eps: 1e-12,
        has_pooler: false,
    };
    let geometry = CandidateBertGeometry::try_new(facts, &ceilings).unwrap();
    assert_eq!(geometry.required_tensors(), 117);
    assert_eq!(geometry.required_elements(), 2_037);
    assert_eq!(geometry.fused_qkv_elements(), 630);

    let pooled = CandidateBertGeometry::try_new(
        CandidateBertFacts {
            has_pooler: true,
            ..facts
        },
        &ceilings,
    )
    .unwrap();
    assert_eq!(pooled.required_tensors(), 119);
    assert_eq!(pooled.required_elements(), 2_067);
    assert_eq!(pooled.fused_qkv_elements(), 630);
}

#[test]
fn candidate_bert_geometry_checks_every_input_axis() {
    let axes = [
        LimitAxis::VocabSize,
        LimitAxis::HiddenSize,
        LimitAxis::HiddenLayers,
        LimitAxis::AttentionHeads,
        LimitAxis::IntermediateSize,
        LimitAxis::PositionEmbeddings,
        LimitAxis::TypeVocabSize,
    ];
    let ceilings = geometry_ceilings(1_000, 1_000_000, BertCeilings::all(nz(16)));

    for axis in axes {
        let zero = with_bert_axis(baseline_facts(), axis, 0);
        assert_eq!(
            CandidateBertGeometry::try_new(zero, &ceilings).unwrap_err(),
            PreparationLimitError::InvalidGeometry {
                fault: BertGeometryFault::Zero(axis),
            }
        );

        let mut exact = with_bert_axis(baseline_facts(), axis, 16);
        if axis == LimitAxis::AttentionHeads {
            exact.hidden_size = 16;
        }
        assert!(CandidateBertGeometry::try_new(exact, &ceilings).is_ok());

        let too_large = with_bert_axis(baseline_facts(), axis, 17);
        assert_eq!(
            CandidateBertGeometry::try_new(too_large, &ceilings).unwrap_err(),
            PreparationLimitError::Exceeded {
                axis,
                actual: 17,
                limit: 16,
            }
        );
    }
}

#[test]
fn candidate_bert_geometry_rejects_head_remainder_and_invalid_epsilon() {
    let ceilings = geometry_ceilings(1_000, 1_000_000, BertCeilings::all(nz(16)));
    let remainder = CandidateBertFacts {
        hidden_size: 3,
        attention_heads: 2,
        ..baseline_facts()
    };
    assert_eq!(
        CandidateBertGeometry::try_new(remainder, &ceilings).unwrap_err(),
        PreparationLimitError::InvalidGeometry {
            fault: BertGeometryFault::AttentionHeadRemainder,
        }
    );

    for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let facts = CandidateBertFacts {
            layer_norm_eps: epsilon,
            ..baseline_facts()
        };
        assert_eq!(
            CandidateBertGeometry::try_new(facts, &ceilings).unwrap_err(),
            PreparationLimitError::InvalidGeometry {
                fault: BertGeometryFault::InvalidLayerNormEpsilon,
            }
        );
    }
}

#[test]
fn position_embeddings_are_not_capped_by_the_later_sequence_limit() {
    let ceilings = geometry_ceilings(1_000, 1_000_000, BertCeilings::all(nz(4_096)));
    let facts = CandidateBertFacts {
        position_embeddings: 4_096,
        ..baseline_facts()
    };
    assert!(CandidateBertGeometry::try_new(facts, &ceilings).is_ok());
}

#[test]
fn candidate_bert_geometry_enforces_derived_census_ceilings() {
    let facts = baseline_facts();
    assert!(
        CandidateBertGeometry::try_new(
            facts,
            &geometry_ceilings(21, 70, BertCeilings::all(nz(16))),
        )
        .is_ok()
    );
    assert_eq!(
        CandidateBertGeometry::try_new(
            facts,
            &geometry_ceilings(20, 70, BertCeilings::all(nz(16))),
        )
        .unwrap_err(),
        PreparationLimitError::Exceeded {
            axis: LimitAxis::Tensors,
            actual: 21,
            limit: 20,
        }
    );
    assert_eq!(
        CandidateBertGeometry::try_new(
            facts,
            &geometry_ceilings(21, 69, BertCeilings::all(nz(16))),
        )
        .unwrap_err(),
        PreparationLimitError::Exceeded {
            axis: LimitAxis::AggregateElements,
            actual: 70,
            limit: 69,
        }
    );

    let pooled = CandidateBertFacts {
        has_pooler: true,
        ..facts
    };
    let geometry = CandidateBertGeometry::try_new(
        pooled,
        &geometry_ceilings(23, 76, BertCeilings::all(nz(16))),
    )
    .unwrap();
    assert_eq!(geometry.required_tensors(), 23);
    assert_eq!(geometry.required_elements(), 76);
}

#[test]
fn geometry_formulas_fail_closed_at_integer_boundaries() {
    let last_layer = (usize::MAX - 7) / 16;
    assert_eq!(
        required_tensor_count(last_layer, true).unwrap(),
        last_layer * 16 + 7
    );
    assert_eq!(
        required_tensor_count(last_layer + 1, true).unwrap_err(),
        overflow(ChargeExpression::RequiredTensorCount)
    );

    assert_eq!(
        required_elements(u64::MAX - 4, 1, 0, 1, 1, 1, false).unwrap(),
        u64::MAX
    );
    assert_eq!(
        required_elements(u64::MAX - 3, 1, 0, 1, 1, 1, false).unwrap_err(),
        overflow(ChargeExpression::RequiredTensorElements)
    );
    assert_eq!(
        required_elements(1, u64::MAX, 1, 1, 1, 1, false).unwrap_err(),
        overflow(ChargeExpression::RequiredTensorElements)
    );

    let last_qkv_layer = u64::MAX / 6;
    assert_eq!(
        fused_qkv_elements(1, last_qkv_layer).unwrap(),
        last_qkv_layer * 6
    );
    assert_eq!(
        fused_qkv_elements(1, last_qkv_layer + 1).unwrap_err(),
        overflow(ChargeExpression::FusedQkvElements)
    );
    assert_eq!(
        fused_qkv_elements(u64::MAX, 1).unwrap_err(),
        overflow(ChargeExpression::FusedQkvElements)
    );
}

#[test]
fn tensor_census_enforces_each_bound_and_is_failure_atomic() {
    let exact = census_ceilings(1, 4, 2, 3, 6, 9);
    let fact = TensorFact {
        name_bytes: 4,
        metadata_bytes: 5,
        dimensions: &[2, 3],
        dtype: TensorDtype::F16,
    };
    let mut census = TensorCensus::new();
    census.push(fact, &exact).unwrap();
    assert_eq!(census.tensor_count(), 1);
    assert_eq!(census.total_name_bytes(), 4);
    assert_eq!(census.total_metadata_bytes(), 5);
    assert_eq!(census.total_elements(), 6);
    assert_eq!(census.declared_tensor_bytes(), 12);

    assert_push_error_without_mutation(
        TensorCensus::new(),
        TensorFact {
            name_bytes: 5,
            ..fact
        },
        &exact,
        PreparationLimitError::Exceeded {
            axis: LimitAxis::TensorNameBytes,
            actual: 5,
            limit: 4,
        },
    );
    assert_push_error_without_mutation(
        TensorCensus::new(),
        TensorFact {
            dimensions: &[1, 1, 1],
            ..fact
        },
        &exact,
        PreparationLimitError::Exceeded {
            axis: LimitAxis::Rank,
            actual: 3,
            limit: 2,
        },
    );
    assert_push_error_without_mutation(
        TensorCensus::new(),
        TensorFact {
            dimensions: &[4],
            ..fact
        },
        &exact,
        PreparationLimitError::Exceeded {
            axis: LimitAxis::Dimension,
            actual: 4,
            limit: 3,
        },
    );
    assert_push_error_without_mutation(
        census,
        TensorFact {
            name_bytes: 1,
            metadata_bytes: 0,
            dimensions: &[1],
            dtype: TensorDtype::F16,
        },
        &exact,
        PreparationLimitError::Exceeded {
            axis: LimitAxis::Tensors,
            actual: 2,
            limit: 1,
        },
    );
    assert_push_error_without_mutation(
        TensorCensus::new(),
        TensorFact {
            metadata_bytes: 6,
            ..fact
        },
        &exact,
        PreparationLimitError::Exceeded {
            axis: LimitAxis::TensorMetadataBytes,
            actual: 10,
            limit: 9,
        },
    );

    let element_cap = census_ceilings(1, 4, 2, 4, 6, 9);
    assert_push_error_without_mutation(
        TensorCensus::new(),
        TensorFact {
            dimensions: &[2, 4],
            ..fact
        },
        &element_cap,
        PreparationLimitError::Exceeded {
            axis: LimitAxis::AggregateElements,
            actual: 8,
            limit: 6,
        },
    );
}

#[test]
fn tensor_census_accounts_all_supported_source_dtypes() {
    let ceilings = census_ceilings(3, 4, 2, 3, 18, 27);
    let mut census = TensorCensus::new();
    for dtype in [TensorDtype::F32, TensorDtype::F16, TensorDtype::Bf16] {
        census
            .push(
                TensorFact {
                    name_bytes: 4,
                    metadata_bytes: 5,
                    dimensions: &[2, 3],
                    dtype,
                },
                &ceilings,
            )
            .unwrap();
    }
    assert_eq!(census.total_elements(), 18);
    assert_eq!(census.declared_tensor_bytes(), 48);
}

/// Ceilings identical to `census_ceilings`'s shape but with an explicit,
/// tight `max_snapshot_bytes` — the axis under test — so the boundary is
/// exercised in isolation from every other axis. `max_weight_file_bytes`
/// is pinned to the same value (a single-shard snapshot) and
/// `max_header_bytes` stays low enough to satisfy the framed-header
/// relation against it.
fn ceilings_with_snapshot_bytes(max_snapshot_bytes: u64) -> PreparationCeilings {
    build((
        InventoryCeilings::new(
            nz(8),
            nz(8),
            nz64(16),
            nz64(16),
            nz64(16),
            nz64(max_snapshot_bytes),
            nz64(max_snapshot_bytes),
            nz64(64),
            nz(8),
        ),
        TensorCeilings::new(nz64(8), nz(8), nz(64), nz(8), nz(64), nz64(64), nz64(64)),
        BertCeilings::all(nz(8)),
        ParseCeilings::all(nz64(8)),
        ResourceCeilings::all(nz64(64)),
    ))
    .unwrap()
}

#[test]
fn tensor_census_bounds_the_declared_payload_total_by_snapshot_bytes() {
    // Two individually-legal 12-byte tensors (dims [2, 3], F16) sum to
    // 24 declared bytes. Every other axis (name, rank, dimension,
    // element, tensor count) is satisfied by each tensor on its own, so
    // only a ceiling on the RUNNING TOTAL can catch this — which is
    // exactly issue #1470's gap: `declared_tensor_bytes` accumulated
    // with no ceiling of its own.
    let fact = TensorFact {
        name_bytes: 1,
        metadata_bytes: 0,
        dimensions: &[2, 3],
        dtype: TensorDtype::F16,
    };

    let exact = ceilings_with_snapshot_bytes(24);
    let mut census = TensorCensus::new();
    census.push(fact, &exact).unwrap();
    census.push(fact, &exact).unwrap();
    assert_eq!(census.declared_tensor_bytes(), 24);

    let tight = ceilings_with_snapshot_bytes(23);
    let mut census = TensorCensus::new();
    census.push(fact, &tight).unwrap();
    assert_push_error_without_mutation(
        census,
        fact,
        &tight,
        PreparationLimitError::Exceeded {
            axis: LimitAxis::SnapshotBytes,
            actual: 24,
            limit: 23,
        },
    );
}

#[test]
fn tensor_arithmetic_accepts_exact_boundaries_and_rejects_overflow() {
    assert_eq!(checked_tensor_elements(&[u64::MAX, 1]).unwrap(), u64::MAX);
    assert_eq!(
        checked_tensor_elements(&[u64::MAX, 2]).unwrap_err(),
        overflow(ChargeExpression::TensorElements)
    );

    let exact_f32_elements = u64::MAX / 4;
    assert_eq!(
        checked_tensor_bytes(exact_f32_elements, TensorDtype::F32).unwrap(),
        exact_f32_elements * 4
    );
    assert_eq!(
        checked_tensor_bytes(exact_f32_elements + 1, TensorDtype::F32).unwrap_err(),
        overflow(ChargeExpression::TensorBytes)
    );
}

#[test]
fn tensor_census_accumulator_overflow_is_failure_atomic() {
    let ceilings = census_ceilings(8, 8, 2, 8, 64, 64);
    let fact = TensorFact {
        name_bytes: 1,
        metadata_bytes: 1,
        dimensions: &[1],
        dtype: TensorDtype::F16,
    };

    assert_push_error_without_mutation(
        TensorCensus {
            tensor_count: usize::MAX,
            ..TensorCensus::new()
        },
        fact,
        &ceilings,
        overflow(ChargeExpression::TensorCensus),
    );
    assert_push_error_without_mutation(
        TensorCensus {
            total_name_bytes: u64::MAX,
            ..TensorCensus::new()
        },
        fact,
        &ceilings,
        overflow(ChargeExpression::TensorCensus),
    );
    assert_push_error_without_mutation(
        TensorCensus {
            total_metadata_bytes: u64::MAX,
            ..TensorCensus::new()
        },
        fact,
        &ceilings,
        overflow(ChargeExpression::TensorCensus),
    );
    assert_push_error_without_mutation(
        TensorCensus {
            total_elements: u64::MAX,
            ..TensorCensus::new()
        },
        fact,
        &ceilings,
        overflow(ChargeExpression::TensorCensus),
    );
    assert_push_error_without_mutation(
        TensorCensus {
            declared_tensor_bytes: u64::MAX,
            ..TensorCensus::new()
        },
        fact,
        &ceilings,
        overflow(ChargeExpression::TensorBytes),
    );
}

#[test]
fn preparation_charge_contains_distinct_retained_and_work_terms() {
    let ceilings = build(charge_groups()).unwrap();
    let charge = PreparationCharge::worst_case(&ceilings).unwrap();
    assert_eq!(charge.retained_bytes(), 5_105);
    assert_eq!(charge.work_bytes(), 1_052_687);

    let retained_without_report = 16 + 32 + 64 + 1 + 128 + 256 + 512;
    assert_eq!(
        charge.retained_bytes() - retained_without_report,
        u64::try_from(super::super::MAX_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES).unwrap()
    );
    let work_without_second_report = ATTESTATION_CHUNK_BYTES + 1 + 2 + 4 + 8;
    assert_eq!(
        charge.work_bytes() - work_without_second_report,
        u64::try_from(super::super::MAX_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES).unwrap()
    );
}

#[test]
fn every_charge_input_changes_only_its_pool() {
    assert_charge_delta(|groups| groups.0.max_snapshot_bytes = nz64(17), 1, 0);
    assert_charge_delta(|groups| groups.4.max_model_retained_bytes = nz64(33), 1, 0);
    assert_charge_delta(
        |groups| groups.4.max_tokenizer_retained_bytes = nz64(65),
        1,
        0,
    );
    assert_charge_delta(|groups| groups.0.max_total_path_bytes = nz64(2), 1, 0);
    assert_charge_delta(
        |groups| groups.0.max_census_metadata_bytes = nz64(129),
        1,
        0,
    );
    assert_charge_delta(|groups| groups.1.max_metadata_bytes = nz64(257), 1, 0);
    assert_charge_delta(
        |groups| groups.4.max_descriptor_control_bytes = nz64(513),
        1,
        0,
    );
    assert_charge_delta(
        |groups| groups.3.max_config_parse_work_bytes = nz64(2),
        0,
        1,
    );
    assert_charge_delta(
        |groups| groups.3.max_tokenizer_parse_work_bytes = nz64(3),
        0,
        1,
    );
    assert_charge_delta(
        |groups| groups.3.max_safetensors_parse_work_bytes = nz64(5),
        0,
        1,
    );
    assert_charge_delta(|groups| groups.4.max_model_load_work_bytes = nz64(9), 0, 1);
}

#[test]
fn unrelated_ceilings_do_not_change_preparation_charge() {
    assert_charge_delta(|groups| groups.2.max_vocab_size = nz(2), 0, 0);
    assert_charge_delta(|groups| groups.0.max_files = nz(2), 0, 0);
    assert_charge_delta(|groups| groups.1.max_tensors = nz(2), 0, 0);
}

#[test]
fn checked_charge_sums_fail_closed_on_overflow() {
    assert_eq!(
        checked_sum(&[u64::MAX - 1, 1], ChargeExpression::RetainedCharge,).unwrap(),
        u64::MAX
    );
    assert_eq!(
        checked_sum(&[u64::MAX, 1], ChargeExpression::RetainedCharge).unwrap_err(),
        overflow(ChargeExpression::RetainedCharge)
    );
    assert_eq!(
        checked_sum(&[u64::MAX, 1], ChargeExpression::WorkCharge).unwrap_err(),
        overflow(ChargeExpression::WorkCharge)
    );
}
