use super::*;
use crate::weights::{Tensor1D, Tensor2D, TransformerLayerWeights};
use std::num::NonZeroU64;

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn raw() -> RawBertConfigFacts {
    RawBertConfigFacts::new(11, 12, 3, 3, 17, 19, 5, 1e-12)
}

fn limits() -> BertGeometryLimits {
    BertGeometryLimits::new(
        nz(11),
        nz(12),
        nz(3),
        nz(3),
        nz(17),
        nz(4_096),
        nz(5),
        nz(55),
        nz(3_927),
    )
}

fn no_pooler() -> BertPoolerMembers {
    BertPoolerMembers::new(false, false)
}

fn geometry() -> PreparedBertGeometry {
    analyze_prepared_geometry(raw(), &limits(), no_pooler()).unwrap()
}

fn raw_with_axis(
    mut facts: RawBertConfigFacts,
    axis: BertFactAxis,
    value: u64,
) -> RawBertConfigFacts {
    match axis {
        BertFactAxis::VocabSize => facts.vocab_size = value,
        BertFactAxis::HiddenSize => facts.hidden_size = value,
        BertFactAxis::HiddenLayers => facts.num_hidden_layers = value,
        BertFactAxis::AttentionHeads => facts.num_attention_heads = value,
        BertFactAxis::IntermediateSize => facts.intermediate_size = value,
        BertFactAxis::PositionEmbeddings => facts.max_position_embeddings = value,
        BertFactAxis::TypeVocabSize => facts.type_vocab_size = value,
        BertFactAxis::RequiredTensors | BertFactAxis::RequiredElements => {
            panic!("derived axes are not raw config fields")
        }
    }
    facts
}

#[test]
fn raw_axes_reject_zero_accept_exact_ceiling_and_reject_ceiling_plus_one() {
    let exact = raw();
    let ceilings = limits();
    assert!(analyze_prepared_geometry(exact, &ceilings, no_pooler()).is_ok());

    let axes = [
        (BertFactAxis::VocabSize, 11),
        (BertFactAxis::HiddenSize, 12),
        (BertFactAxis::HiddenLayers, 3),
        (BertFactAxis::AttentionHeads, 3),
        (BertFactAxis::IntermediateSize, 17),
        (BertFactAxis::PositionEmbeddings, 4_096),
        (BertFactAxis::TypeVocabSize, 5),
    ];

    for (axis, limit) in axes {
        assert_eq!(
            analyze_prepared_geometry(raw_with_axis(exact, axis, 0), &ceilings, no_pooler(),)
                .unwrap_err(),
            PreparedBertFactError::Zero(axis),
        );
        assert_eq!(
            analyze_prepared_geometry(
                raw_with_axis(exact, axis, limit + 1),
                &ceilings,
                no_pooler(),
            )
            .unwrap_err(),
            PreparedBertFactError::Exceeded {
                axis,
                actual: limit + 1,
                limit,
            },
        );
    }
}

#[test]
fn every_raw_axis_is_checked_for_platform_representability_before_conversion() {
    let platform_max = 64;
    let broad = BertGeometryLimits::all(nz(1_000), nz(10_000), nz(10_000_000));
    let base = RawBertConfigFacts::new(1, 64, 1, 1, 1, 1, 1, 1e-12);
    let axes = [
        BertFactAxis::VocabSize,
        BertFactAxis::HiddenSize,
        BertFactAxis::HiddenLayers,
        BertFactAxis::AttentionHeads,
        BertFactAxis::IntermediateSize,
        BertFactAxis::PositionEmbeddings,
        BertFactAxis::TypeVocabSize,
    ];

    for axis in axes {
        assert!(
            analyze_prepared_geometry_with_platform_max(
                raw_with_axis(base, axis, platform_max),
                &broad,
                no_pooler(),
                platform_max,
            )
            .is_ok(),
            "{axis:?} exact platform maximum must be accepted",
        );
        assert_eq!(
            analyze_prepared_geometry_with_platform_max(
                raw_with_axis(base, axis, platform_max + 1),
                &broad,
                no_pooler(),
                platform_max,
            )
            .unwrap_err(),
            PreparedBertFactError::PlatformUnrepresentable {
                axis,
                value: platform_max + 1,
            },
        );
    }
}

#[test]
fn attention_geometry_requires_an_exact_nonzero_partition() {
    let mut facts = raw();
    facts.hidden_size = 14;
    let broad = BertGeometryLimits::all(nz(100), nz(1_000), nz(1_000_000));
    assert_eq!(
        analyze_prepared_geometry(facts, &broad, no_pooler()).unwrap_err(),
        PreparedBertFactError::HiddenHeadRemainder {
            hidden_size: 14,
            attention_heads: 3,
        },
    );

    facts.hidden_size = 15;
    let geometry = analyze_prepared_geometry(facts, &broad, no_pooler()).unwrap();
    assert_eq!(geometry.head_dim(), 5);
}

#[test]
fn epsilon_must_survive_f64_to_f32_realization_as_positive_and_finite() {
    let rejected = [
        0.0,
        -0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(1),
        f64::MAX,
    ];
    for layer_norm_eps in rejected {
        let mut facts = raw();
        facts.layer_norm_eps = layer_norm_eps;
        assert_eq!(
            analyze_prepared_geometry(facts, &limits(), no_pooler()).unwrap_err(),
            PreparedBertFactError::InvalidLayerNormEpsilon,
            "epsilon {layer_norm_eps:?} must fail",
        );
    }

    let validated = analyze_prepared_geometry(raw(), &limits(), no_pooler()).unwrap();
    assert_eq!(validated.layer_norm_eps(), 1e-12_f32);

    for layer_norm_eps in [f64::from(f32::from_bits(1)), f64::from(f32::MAX)] {
        let mut facts = raw();
        facts.layer_norm_eps = layer_norm_eps;
        assert!(
            analyze_prepared_geometry(facts, &limits(), no_pooler()).is_ok(),
            "realizable endpoint {layer_norm_eps:?} must be accepted",
        );
    }
}

#[test]
fn pooler_members_are_an_exact_pair() {
    for (weight, bias) in [(false, false), (true, true)] {
        assert!(
            analyze_prepared_geometry(raw(), &limits(), BertPoolerMembers::new(weight, bias),)
                .is_ok()
        );
    }
    for (weight, bias) in [(true, false), (false, true)] {
        assert_eq!(
            analyze_prepared_geometry(raw(), &limits(), BertPoolerMembers::new(weight, bias),)
                .unwrap_err(),
            PreparedBertFactError::IncompletePooler { weight, bias },
        );
    }
}

#[test]
fn distinct_geometry_fixture_pins_every_formula_coefficient() {
    let plain = geometry();
    assert!(!plain.has_pooler());
    assert_eq!(plain.required_tensor_count(), 53);
    assert_eq!(plain.required_elements(), 3_771);
    assert_eq!(plain.fused_qkv_elements(), 1_404);

    let pooled =
        analyze_prepared_geometry(raw(), &limits(), BertPoolerMembers::new(true, true)).unwrap();
    assert!(pooled.has_pooler());
    assert_eq!(pooled.required_tensor_count(), 55);
    assert_eq!(pooled.required_elements(), 3_927);
    assert_eq!(
        pooled.required_elements() - plain.required_elements(),
        12 * 12 + 12
    );
    assert_eq!(pooled.fused_qkv_elements(), plain.fused_qkv_elements());
}

#[test]
fn hidden_layers_and_attention_heads_are_not_interchangeable() {
    let facts = RawBertConfigFacts::new(11, 12, 2, 3, 17, 19, 5, 1e-12);
    let exact = BertGeometryLimits::new(
        nz(11),
        nz(12),
        nz(2),
        nz(3),
        nz(17),
        nz(19),
        nz(5),
        nz(39),
        nz(2_818),
    );
    let plain = analyze_prepared_geometry(facts, &exact, no_pooler()).unwrap();
    assert_eq!(plain.hidden_layers(), 2);
    assert_eq!(plain.attention_heads(), 3);
    let config = plain.bert_config();
    assert_eq!(config.num_hidden_layers, 2);
    assert_eq!(config.num_attention_heads, 3);
    assert_eq!(plain.required_tensor_count(), 37);
    assert_eq!(plain.required_elements(), 2_662);
    assert_eq!(plain.fused_qkv_elements(), 936);

    let pooled =
        analyze_prepared_geometry(facts, &exact, BertPoolerMembers::new(true, true)).unwrap();
    assert_eq!(pooled.required_tensor_count(), 39);
    assert_eq!(pooled.required_elements(), 2_818);

    let too_many_layers = RawBertConfigFacts::new(11, 12, 3, 3, 17, 19, 5, 1e-12);
    assert_eq!(
        analyze_prepared_geometry(too_many_layers, &exact, no_pooler()).unwrap_err(),
        PreparedBertFactError::Exceeded {
            axis: BertFactAxis::HiddenLayers,
            actual: 3,
            limit: 2,
        },
    );
    let too_many_heads = RawBertConfigFacts::new(11, 12, 2, 4, 17, 19, 5, 1e-12);
    assert_eq!(
        analyze_prepared_geometry(too_many_heads, &exact, no_pooler()).unwrap_err(),
        PreparedBertFactError::Exceeded {
            axis: BertFactAxis::AttentionHeads,
            actual: 4,
            limit: 3,
        },
    );
}

#[test]
fn position_embeddings_above_sequence_cap_are_valid_geometry() {
    let mut facts = raw();
    facts.max_position_embeddings = 4_096;
    let broad = BertGeometryLimits::all(nz(4_096), nz(53), nz(60_000));
    let validated = analyze_prepared_geometry(facts, &broad, no_pooler()).unwrap();
    assert_eq!(validated.position_embeddings(), 4_096);
}

#[test]
fn derived_tensor_and_element_ceilings_are_enforced_after_checked_math() {
    let facts = raw();
    let tensor_limited = BertGeometryLimits::all(nz(4_096), nz(52), nz(3_771));
    assert_eq!(
        analyze_prepared_geometry(facts, &tensor_limited, no_pooler()).unwrap_err(),
        PreparedBertFactError::Exceeded {
            axis: BertFactAxis::RequiredTensors,
            actual: 53,
            limit: 52,
        },
    );

    let element_limited = BertGeometryLimits::all(nz(4_096), nz(53), nz(3_770));
    assert_eq!(
        analyze_prepared_geometry(facts, &element_limited, no_pooler()).unwrap_err(),
        PreparedBertFactError::Exceeded {
            axis: BertFactAxis::RequiredElements,
            actual: 3_771,
            limit: 3_770,
        },
    );
}

#[test]
fn checked_geometry_helpers_reject_expression_overflow() {
    assert_eq!(
        checked_required_tensor_count(u64::MAX, false).unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::RequiredTensorCount,),
    );
    for (v, h, l, i, p, t, pooler) in [
        (u64::MAX, 1, 1, 1, 1, 1, false),
        (u64::MAX - 1, 1, 1, 1, 1, 1, false),
        (u64::MAX - 3, 1, 1, 1, 1, 1, false),
        (u64::MAX / 2, 2, 1, 1, 1, 1, false),
        (1, u64::MAX, 1, 1, 1, 1, false),
        (1, u64::from(u32::MAX), 1, 1, 1, 1, false),
        (1, 2, 1, u64::MAX, 1, 1, false),
        (1, 1, 1, u64::MAX / 2 + 1, 1, 1, false),
        (1, 1, 1, (u64::MAX - 3) / 2, 1, 1, false),
        (1, 1, 1, (u64::MAX - 7) / 2, 1, 1, false),
        (1, 1, 1, u64::MAX / 3, 1, 1, false),
        (1, 2, u64::MAX, 2, 1, 1, false),
        (12, 1, u64::MAX / 16, 1, 1, 1, false),
        (11, 1, u64::MAX / 16, 1, 1, 1, true),
    ] {
        assert_eq!(
            checked_required_tensor_elements(v, h, l, i, p, t, pooler).unwrap_err(),
            PreparedBertFactError::ArithmeticOverflow(BertFactExpression::RequiredTensorElements,),
        );
    }
    assert_eq!(
        checked_fused_qkv_elements(u64::MAX, 1).unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::FusedQkvElements),
    );
    assert_eq!(
        checked_fused_qkv_elements(u64::from(u32::MAX), 1).unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::FusedQkvElements),
    );
    assert_eq!(
        checked_fused_qkv_elements(2, u64::MAX).unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::FusedQkvElements),
    );
}

#[test]
fn realized_geometry_is_the_only_path_to_legacy_bert_config() {
    let validated = geometry();
    let config = validated.bert_config();
    assert_eq!(config.vocab_size, 11);
    assert_eq!(config.hidden_size, 12);
    assert_eq!(config.num_hidden_layers, 3);
    assert_eq!(config.num_attention_heads, 3);
    assert_eq!(config.intermediate_size, 17);
    assert_eq!(config.max_position_embeddings, 19);
    assert_eq!(config.type_vocab_size, 5);
    assert_eq!(config.layer_norm_eps.to_bits(), 1e-12_f32.to_bits());
}

#[test]
fn required_subset_count_and_elements_must_match_exactly() {
    let geometry = geometry();
    let exact = RequiredBertTensorCensus::new(53, 3_771, 0, 0, 0);
    assert!(checked_logical_weight_payload_footprint(&geometry, exact, 15_084).is_ok());

    for actual in [52, 54] {
        let census = RequiredBertTensorCensus::new(actual, 3_771, 0, 0, 0);
        assert_eq!(
            checked_logical_weight_payload_footprint(&geometry, census, 15_084).unwrap_err(),
            PreparedBertFactError::RequiredTensorCountMismatch {
                expected: 53,
                actual,
            },
        );
    }
    for actual in [3_770, 3_772] {
        let census = RequiredBertTensorCensus::new(53, actual, 0, 0, 0);
        assert_eq!(
            checked_logical_weight_payload_footprint(&geometry, census, u64::MAX).unwrap_err(),
            PreparedBertFactError::RequiredElementCountMismatch {
                expected: 3_771,
                actual,
            },
        );
    }
}

#[test]
fn dtype_widths_and_zero_copy_disposition_are_accounted_independently() {
    let geometry = geometry();
    let mixed = RequiredBertTensorCensus::new(53, 1, 2, 3, 3_765);
    let footprint = checked_logical_weight_payload_footprint(&geometry, mixed, 7_548).unwrap();
    assert_eq!(footprint.required_source_payload_bytes(), 7_548);
    assert_eq!(footprint.materialized_f32_bytes(), 15_080);
    assert_eq!(footprint.fused_qkv_bytes(), 5_616);
    assert_eq!(footprint.logical_weight_payload_bytes(), 28_244);

    let zero_copy = RequiredBertTensorCensus::new(53, 3_771, 0, 0, 0);
    let copied = RequiredBertTensorCensus::new(53, 3_770, 1, 0, 0);
    let zero_copy = checked_logical_weight_payload_footprint(&geometry, zero_copy, 15_084).unwrap();
    let copied = checked_logical_weight_payload_footprint(&geometry, copied, 15_084).unwrap();
    assert_eq!(
        copied.required_source_payload_bytes(),
        zero_copy.required_source_payload_bytes(),
    );
    assert_eq!(copied.materialized_f32_bytes(), 4);
    assert_eq!(zero_copy.materialized_f32_bytes(), 0);
    assert_eq!(
        copied.logical_weight_payload_bytes() - zero_copy.logical_weight_payload_bytes(),
        4,
    );
}

#[test]
fn mapped_file_covers_required_payload_but_may_include_extra_inventory() {
    let geometry = geometry();
    let census = RequiredBertTensorCensus::new(53, 3_771, 0, 0, 0);
    let required = 15_084;
    assert!(checked_logical_weight_payload_footprint(&geometry, census, required).is_ok());
    assert_eq!(
        checked_logical_weight_payload_footprint(&geometry, census, required - 1).unwrap_err(),
        PreparedBertFactError::MappedWeightFileTooSmall {
            mapped: required - 1,
            required_source_payload: required,
        },
    );

    let with_extra_inventory =
        checked_logical_weight_payload_footprint(&geometry, census, required + 123).unwrap();
    assert_eq!(
        with_extra_inventory.mapped_weight_file_bytes(),
        required + 123
    );
    assert_eq!(
        with_extra_inventory.logical_weight_payload_bytes(),
        required + 123 + 5_616,
    );
}

#[test]
fn footprint_arithmetic_is_checked_at_every_stage() {
    let geometry = PreparedBertGeometry::for_test(u64::MAX, u64::MAX);
    assert_eq!(
        checked_required_census_elements(&RequiredBertTensorCensus::new(1, u64::MAX, 1, 0, 0,))
            .unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::CensusElements),
    );
    for census in [
        RequiredBertTensorCensus::new(1, u64::MAX - 1, 1, 1, 0),
        RequiredBertTensorCensus::new(1, u64::MAX - 2, 1, 1, 1),
    ] {
        assert_eq!(
            checked_required_census_elements(&census).unwrap_err(),
            PreparedBertFactError::ArithmeticOverflow(BertFactExpression::CensusElements),
        );
    }
    assert_eq!(
        checked_required_source_payload_bytes(
            &RequiredBertTensorCensus::new(1, u64::MAX, 0, 0, 0,)
        )
        .unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::RequiredSourcePayloadBytes,),
    );
    for census in [
        RequiredBertTensorCensus::new(1, 0, 0, u64::MAX, 0),
        RequiredBertTensorCensus::new(1, 0, 0, 0, u64::MAX),
        RequiredBertTensorCensus::new(1, u64::MAX / 4, 0, 2, 0),
        RequiredBertTensorCensus::new(1, u64::MAX / 4, 0, 1, 1),
    ] {
        assert_eq!(
            checked_required_source_payload_bytes(&census).unwrap_err(),
            PreparedBertFactError::ArithmeticOverflow(
                BertFactExpression::RequiredSourcePayloadBytes,
            ),
        );
    }
    assert_eq!(
        checked_materialized_f32_bytes(&RequiredBertTensorCensus::new(1, 0, u64::MAX, 0, 0,))
            .unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::MaterializedF32Bytes,),
    );
    for census in [
        RequiredBertTensorCensus::new(1, 0, u64::MAX, 1, 0),
        RequiredBertTensorCensus::new(1, 0, u64::MAX - 1, 1, 1),
    ] {
        assert_eq!(
            checked_materialized_f32_bytes(&census).unwrap_err(),
            PreparedBertFactError::ArithmeticOverflow(BertFactExpression::MaterializedF32Bytes,),
        );
    }
    assert_eq!(
        checked_fused_qkv_bytes(&geometry).unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::FusedQkvBytes),
    );

    let geometry = PreparedBertGeometry::for_test(1, 0);
    let census = RequiredBertTensorCensus::new(1, 0, 1, 0, 0);
    assert_eq!(
        checked_logical_weight_payload_footprint(&geometry, census, u64::MAX).unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::LogicalWeightPayloadBytes,),
    );

    let geometry = PreparedBertGeometry::for_test(1, 1);
    let census = RequiredBertTensorCensus::new(1, 1, 0, 0, 0);
    assert_eq!(
        checked_logical_weight_payload_footprint(&geometry, census, u64::MAX - 3).unwrap_err(),
        PreparedBertFactError::ArithmeticOverflow(BertFactExpression::LogicalWeightPayloadBytes,),
    );
}

#[test]
fn fused_qkv_estimate_matches_the_actual_builder_layout() {
    let matrix = [1.0_f32, 2.0, 3.0, 4.0];
    let vector = [1.0_f32, 2.0];
    let matrix_view = Tensor2D {
        data: &matrix,
        rows: 2,
        cols: 2,
    };
    let vector_view = Tensor1D {
        data: &vector,
        len: 2,
    };
    let layer = TransformerLayerWeights {
        query_weight: matrix_view,
        query_bias: vector_view,
        key_weight: matrix_view,
        key_bias: vector_view,
        value_weight: matrix_view,
        value_bias: vector_view,
        attn_output_weight: matrix_view,
        attn_output_bias: vector_view,
        attn_layer_norm_weight: vector_view,
        attn_layer_norm_bias: vector_view,
        ffn_intermediate_weight: matrix_view,
        ffn_intermediate_bias: vector_view,
        ffn_output_weight: matrix_view,
        ffn_output_bias: vector_view,
        ffn_layer_norm_weight: vector_view,
        ffn_layer_norm_bias: vector_view,
    };
    let fused = super::super::LayerFusedQkv::build(&layer);
    let estimated = checked_fused_qkv_elements(2, 1).unwrap();
    assert_eq!(
        u64::try_from(fused.weight.len() + fused.bias.len()).unwrap(),
        estimated,
    );
}
