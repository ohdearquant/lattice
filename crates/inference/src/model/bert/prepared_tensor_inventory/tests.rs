use std::num::NonZeroU64;

use super::super::prepared_facts::{
    BertFactAxis, BertGeometryLimits, PreparedBertFactError, RawBertConfigFacts,
    RequiredBertTensorCensus, checked_logical_weight_payload_footprint,
};
use super::*;
use crate::weights::prepared_safetensors::{
    PreparedSafetensorsFramingLimits, PreparedSafetensorsHeaderPlan,
    plan_prepared_safetensors_header,
};
use crate::weights::prepared_safetensors_inventory::{
    F32FileOffsetAlignment, PreparedSafetensorsInventoryFacts, PreparedSafetensorsInventoryLimits,
    parse_prepared_safetensors_header_inventory,
};

const V: u64 = 7;
const H: u64 = 4;
const L: u64 = 2;
const A: u64 = 2;
const I: u64 = 6;
const P: u64 = 5;
const T: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureDtype {
    F32,
    F16,
    Bf16,
}

impl FixtureDtype {
    fn tag(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
        }
    }

    fn width(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

#[derive(Clone, Debug)]
struct FixtureTensor {
    name: String,
    dimensions: Vec<u64>,
    dtype: FixtureDtype,
}

impl FixtureTensor {
    fn new(name: impl Into<String>, dimensions: &[u64]) -> Self {
        Self {
            name: name.into(),
            dimensions: dimensions.to_vec(),
            dtype: FixtureDtype::F32,
        }
    }

    fn elements(&self) -> u64 {
        self.dimensions.iter().copied().product()
    }
}

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn raw() -> RawBertConfigFacts {
    RawBertConfigFacts::new(V, H, L, A, I, P, T, 1e-5)
}

fn geometry_limits() -> BertGeometryLimits {
    BertGeometryLimits::new(
        nz(1_000),
        nz(1_000),
        nz(64),
        nz(64),
        nz(4_000),
        nz(4_000),
        nz(64),
        nz(2_000),
        nz(1_000_000_000),
    )
}

fn inventory_limits() -> PreparedSafetensorsInventoryLimits {
    PreparedSafetensorsInventoryLimits::try_new(
        nz(2_000),
        nz(512),
        nz(8),
        nz(1_000_000),
        nz(1_000_000_000),
        nz(4_000_000),
        nz(8_000_000),
    )
    .unwrap()
}

fn required_tensors(pooler: bool) -> Vec<FixtureTensor> {
    let mut tensors = vec![
        FixtureTensor::new("embeddings.word_embeddings.weight", &[V, H]),
        FixtureTensor::new("embeddings.position_embeddings.weight", &[P, H]),
        FixtureTensor::new("embeddings.token_type_embeddings.weight", &[T, H]),
        FixtureTensor::new("embeddings.LayerNorm.weight", &[H]),
        FixtureTensor::new("embeddings.LayerNorm.bias", &[H]),
    ];
    for layer in 0..L {
        let prefix = format!("encoder.layer.{layer}");
        for (suffix, dimensions) in [
            ("attention.self.query.weight", vec![H, H]),
            ("attention.self.query.bias", vec![H]),
            ("attention.self.key.weight", vec![H, H]),
            ("attention.self.key.bias", vec![H]),
            ("attention.self.value.weight", vec![H, H]),
            ("attention.self.value.bias", vec![H]),
            ("attention.output.dense.weight", vec![H, H]),
            ("attention.output.dense.bias", vec![H]),
            ("attention.output.LayerNorm.weight", vec![H]),
            ("attention.output.LayerNorm.bias", vec![H]),
            ("intermediate.dense.weight", vec![I, H]),
            ("intermediate.dense.bias", vec![I]),
            ("output.dense.weight", vec![H, I]),
            ("output.dense.bias", vec![H]),
            ("output.LayerNorm.weight", vec![H]),
            ("output.LayerNorm.bias", vec![H]),
        ] {
            tensors.push(FixtureTensor::new(
                format!("{prefix}.{suffix}"),
                &dimensions,
            ));
        }
    }
    if pooler {
        tensors.push(FixtureTensor::new("pooler.dense.weight", &[H, H]));
        tensors.push(FixtureTensor::new("pooler.dense.bias", &[H]));
    }
    tensors
}

fn expected_tensor(name: &str) -> PreparedBertTensor {
    match name {
        "embeddings.word_embeddings.weight" => PreparedBertTensor::WordEmbeddingsWeight,
        "embeddings.position_embeddings.weight" => PreparedBertTensor::PositionEmbeddingsWeight,
        "embeddings.token_type_embeddings.weight" => PreparedBertTensor::TokenTypeEmbeddingsWeight,
        "embeddings.LayerNorm.weight" => PreparedBertTensor::EmbeddingsLayerNormWeight,
        "embeddings.LayerNorm.bias" => PreparedBertTensor::EmbeddingsLayerNormBias,
        "pooler.dense.weight" => PreparedBertTensor::PoolerWeight,
        "pooler.dense.bias" => PreparedBertTensor::PoolerBias,
        _ => {
            let rest = name.strip_prefix("encoder.layer.").unwrap();
            let (layer, suffix) = rest.split_once('.').unwrap();
            let member = match suffix {
                "attention.self.query.weight" => PreparedBertLayerTensor::QueryWeight,
                "attention.self.query.bias" => PreparedBertLayerTensor::QueryBias,
                "attention.self.key.weight" => PreparedBertLayerTensor::KeyWeight,
                "attention.self.key.bias" => PreparedBertLayerTensor::KeyBias,
                "attention.self.value.weight" => PreparedBertLayerTensor::ValueWeight,
                "attention.self.value.bias" => PreparedBertLayerTensor::ValueBias,
                "attention.output.dense.weight" => PreparedBertLayerTensor::AttentionOutputWeight,
                "attention.output.dense.bias" => PreparedBertLayerTensor::AttentionOutputBias,
                "attention.output.LayerNorm.weight" => {
                    PreparedBertLayerTensor::AttentionLayerNormWeight
                }
                "attention.output.LayerNorm.bias" => {
                    PreparedBertLayerTensor::AttentionLayerNormBias
                }
                "intermediate.dense.weight" => PreparedBertLayerTensor::IntermediateWeight,
                "intermediate.dense.bias" => PreparedBertLayerTensor::IntermediateBias,
                "output.dense.weight" => PreparedBertLayerTensor::OutputWeight,
                "output.dense.bias" => PreparedBertLayerTensor::OutputBias,
                "output.LayerNorm.weight" => PreparedBertLayerTensor::OutputLayerNormWeight,
                "output.LayerNorm.bias" => PreparedBertLayerTensor::OutputLayerNormBias,
                _ => panic!("unexpected required suffix {suffix}"),
            };
            PreparedBertTensor::Encoder {
                layer: layer.parse().unwrap(),
                member,
            }
        }
    }
}

fn encode_header(tensors: &[FixtureTensor], padding: usize) -> (String, u64) {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (index, tensor) in tensors.iter().enumerate() {
        if index != 0 {
            header.push(',');
        }
        let bytes = tensor.elements().checked_mul(tensor.dtype.width()).unwrap();
        let end = offset.checked_add(bytes).unwrap();
        let dimensions = tensor
            .dimensions
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            tensor.name,
            tensor.dtype.tag(),
            dimensions,
            offset,
            end
        ));
        offset = end;
    }
    header.push('}');
    header.extend(std::iter::repeat_n(' ', padding));
    (header, offset)
}

fn parse_inventory(
    tensors: &[FixtureTensor],
    padding: usize,
) -> (
    PreparedSafetensorsInventoryFacts,
    PreparedSafetensorsHeaderPlan,
) {
    let (header, data_len) = encode_header(tensors, padding);
    let header_len = u64::try_from(header.len()).unwrap();
    let declared = 8_u64
        .checked_add(header_len)
        .and_then(|value| value.checked_add(data_len))
        .unwrap();
    let framing = PreparedSafetensorsFramingLimits::try_new(nz(declared), nz(header_len)).unwrap();
    let plan =
        plan_prepared_safetensors_header(header_len.to_le_bytes(), declared, &framing).unwrap();
    let facts =
        parse_prepared_safetensors_header_inventory(header.as_bytes(), &plan, &inventory_limits())
            .unwrap();
    (facts, plan)
}

fn match_tensors(
    tensors: &[FixtureTensor],
    padding: usize,
) -> Result<MatchedBertInventoryFacts, PreparedBertTensorMatchError> {
    let (inventory, _) = parse_inventory(tensors, padding);
    match_prepared_bert_inventory(raw(), &geometry_limits(), &inventory)
}

fn required_elements(pooler: bool) -> u64 {
    H * (V + P + T + 2) + L * (4 * H * H + 2 * H * I + 9 * H + I) + u64::from(pooler) * (H * H + H)
}

#[test]
fn complete_required_closure_derives_geometry_and_conservative_census() {
    for pooler in [false, true] {
        let matched = match_tensors(&required_tensors(pooler), 0).unwrap();
        assert_eq!(matched.geometry().hidden_layers(), L as usize);
        assert_eq!(matched.geometry().has_pooler(), pooler);
        assert_eq!(
            matched.required_tensor_census(),
            RequiredBertTensorCensus::new(
                5 + 16 * L + 2 * u64::from(pooler),
                0,
                required_elements(pooler),
                0,
                0,
            )
        );
    }
}

#[test]
fn pooler_presence_is_derived_from_inventory_before_geometry_analysis() {
    for (missing, weight, bias) in [
        ("pooler.dense.weight", false, true),
        ("pooler.dense.bias", true, false),
    ] {
        let mut tensors = required_tensors(true);
        tensors.retain(|tensor| tensor.name != missing);
        assert_eq!(
            match_tensors(&tensors, 0).unwrap_err(),
            PreparedBertTensorMatchError::Geometry(PreparedBertFactError::IncompletePooler {
                weight,
                bias
            })
        );
    }
}

#[test]
fn unrelated_and_unknown_encoder_extras_do_not_enter_required_census() {
    let baseline = match_tensors(&required_tensors(false), 0).unwrap();
    let mut with_extras = required_tensors(false);
    let mut classifier = FixtureTensor::new("classifier.weight", &[11, H]);
    classifier.dtype = FixtureDtype::Bf16;
    with_extras.push(classifier);
    with_extras.push(FixtureTensor::new(
        "encoder.layer.bad.auxiliary.weight",
        &[999, 999],
    ));
    with_extras.push(FixtureTensor::new(
        "encoder.layer.0.adapter.attention.self.query.weight",
        &[999, 999],
    ));
    assert_eq!(
        match_tensors(&with_extras, 0)
            .unwrap()
            .required_tensor_census(),
        baseline.required_tensor_census()
    );
}

#[test]
fn every_required_slot_is_counted_without_total_inventory_equality() {
    let complete = required_tensors(false);
    for missing in complete.iter().map(|tensor| tensor.name.clone()) {
        let mut tensors = complete.clone();
        tensors.retain(|tensor| tensor.name != missing);
        assert_eq!(
            match_tensors(&tensors, 0).unwrap_err(),
            PreparedBertTensorMatchError::RequiredTensorCountMismatch {
                expected: 5 + 16 * L,
                actual: 5 + 16 * L - 1,
            },
            "missing {missing}"
        );
    }
}

#[test]
fn every_required_name_maps_to_its_exact_typed_role() {
    let complete = required_tensors(true);
    for (index, original) in complete.iter().enumerate() {
        let mut tensors = complete.clone();
        tensors[index].dimensions.clear();
        assert_eq!(
            match_tensors(&tensors, 0).unwrap_err(),
            PreparedBertTensorMatchError::InvalidShape {
                tensor: expected_tensor(&original.name),
                fault: BertShapeFault::Rank {
                    expected: original.dimensions.len(),
                    actual: 0,
                },
            },
            "required role {}",
            original.name
        );
    }
}

#[test]
fn closed_layer_suffixes_require_canonical_in_range_ascii_indices() {
    let query = "encoder.layer.0.attention.self.query.weight";
    for (replacement, expected) in [
        (
            "encoder.layer..attention.self.query.weight",
            PreparedBertTensorMatchError::InvalidLayerIndex {
                member: PreparedBertLayerTensor::QueryWeight,
                fault: BertLayerIndexFault::Empty,
            },
        ),
        (
            "encoder.layer.01.attention.self.query.weight",
            PreparedBertTensorMatchError::InvalidLayerIndex {
                member: PreparedBertLayerTensor::QueryWeight,
                fault: BertLayerIndexFault::LeadingZero,
            },
        ),
        (
            "encoder.layer.x.attention.self.query.weight",
            PreparedBertTensorMatchError::InvalidLayerIndex {
                member: PreparedBertLayerTensor::QueryWeight,
                fault: BertLayerIndexFault::NonAsciiDigit { at: 0 },
            },
        ),
        (
            "encoder.layer.18446744073709551616.attention.self.query.weight",
            PreparedBertTensorMatchError::InvalidLayerIndex {
                member: PreparedBertLayerTensor::QueryWeight,
                fault: BertLayerIndexFault::Overflow,
            },
        ),
        (
            "encoder.layer.2.attention.self.query.weight",
            PreparedBertTensorMatchError::LayerIndexOutOfRange {
                index: L,
                hidden_layers: L as usize,
            },
        ),
    ] {
        let mut tensors = required_tensors(false);
        tensors
            .iter_mut()
            .find(|tensor| tensor.name == query)
            .unwrap()
            .name = replacement.to_string();
        assert_eq!(match_tensors(&tensors, 0).unwrap_err(), expected);
    }
}

#[test]
fn layer_index_parser_accepts_canonical_multidigit_and_pins_fault_positions() {
    assert_eq!(parse_layer_index(b"10"), Ok(10));
    assert_eq!(
        parse_layer_index(u64::MAX.to_string().as_bytes()),
        Ok(u64::MAX)
    );
    assert_eq!(
        parse_layer_index(b"1x"),
        Err(BertLayerIndexFault::NonAsciiDigit { at: 1 })
    );
    assert_eq!(
        classify_required_tensor(b"encoder.layer.10.attention.self.query.weight", 11),
        Ok(Some(PreparedBertTensor::Encoder {
            layer: 10,
            member: PreparedBertLayerTensor::QueryWeight,
        }))
    );
}

#[test]
fn exact_shape_families_reject_rank_axis_and_transpose_mutations() {
    for (name, dimensions, tensor, fault) in [
        (
            "embeddings.word_embeddings.weight",
            vec![V * H],
            PreparedBertTensor::WordEmbeddingsWeight,
            BertShapeFault::Rank {
                expected: 2,
                actual: 1,
            },
        ),
        (
            "embeddings.word_embeddings.weight",
            vec![V, H, 1],
            PreparedBertTensor::WordEmbeddingsWeight,
            BertShapeFault::Rank {
                expected: 2,
                actual: 3,
            },
        ),
        (
            "encoder.layer.0.intermediate.dense.weight",
            vec![H, I],
            PreparedBertTensor::Encoder {
                layer: 0,
                member: PreparedBertLayerTensor::IntermediateWeight,
            },
            BertShapeFault::Dimension {
                axis: 0,
                expected: I as usize,
                actual: H,
            },
        ),
        (
            "encoder.layer.0.output.dense.weight",
            vec![I, H],
            PreparedBertTensor::Encoder {
                layer: 0,
                member: PreparedBertLayerTensor::OutputWeight,
            },
            BertShapeFault::Dimension {
                axis: 0,
                expected: H as usize,
                actual: I,
            },
        ),
        (
            "encoder.layer.0.intermediate.dense.weight",
            vec![I, H + 1],
            PreparedBertTensor::Encoder {
                layer: 0,
                member: PreparedBertLayerTensor::IntermediateWeight,
            },
            BertShapeFault::Dimension {
                axis: 1,
                expected: H as usize,
                actual: H + 1,
            },
        ),
    ] {
        let mut tensors = required_tensors(false);
        tensors
            .iter_mut()
            .find(|candidate| candidate.name == name)
            .unwrap()
            .dimensions = dimensions;
        assert_eq!(
            match_tensors(&tensors, 0).unwrap_err(),
            PreparedBertTensorMatchError::InvalidShape { tensor, fault }
        );
    }
}

#[test]
fn dtype_buckets_are_required_only_and_f32_alignment_never_grants_zero_copy() {
    let mut tensors = required_tensors(false);
    tensors[0].dtype = FixtureDtype::F16;
    tensors[1].dtype = FixtureDtype::Bf16;
    let f16 = tensors[0].elements();
    let bf16 = tensors[1].elements();
    let f32 = required_elements(false) - f16 - bf16;

    let expected = RequiredBertTensorCensus::new(5 + 16 * L, 0, f32, f16, bf16);
    let mut saw_aligned = false;
    let mut saw_unaligned = false;
    for padding in 0..4 {
        let (inventory, _) = parse_inventory(&tensors, padding);
        let alignment = inventory
            .tensors()
            .find(|tensor| tensor.name_bytes() == b"embeddings.token_type_embeddings.weight")
            .unwrap()
            .f32_file_offset_alignment()
            .unwrap();
        saw_aligned |= alignment == F32FileOffsetAlignment::FourByteAligned;
        saw_unaligned |= alignment == F32FileOffsetAlignment::Unaligned;
        let matched = match_prepared_bert_inventory(raw(), &geometry_limits(), &inventory).unwrap();
        assert_eq!(matched.required_tensor_census(), expected);
    }
    assert!(saw_aligned);
    assert!(saw_unaligned);
}

#[test]
fn footprint_integration_uses_the_whole_declared_file_not_payload_only() {
    let tensors = required_tensors(false);
    let (inventory, plan) = parse_inventory(&tensors, 3);
    let matched = match_prepared_bert_inventory(raw(), &geometry_limits(), &inventory).unwrap();
    let mapped = u64::try_from(plan.declared_file_len()).unwrap();
    let footprint = checked_logical_weight_payload_footprint(
        &matched.geometry(),
        matched.required_tensor_census(),
        mapped,
    )
    .unwrap();
    assert_eq!(footprint.mapped_weight_file_bytes(), mapped);
    assert!(mapped > inventory.source_payload_bytes());
}

#[test]
fn geometry_errors_stay_typed_before_tensor_matching() {
    let mut tensors = required_tensors(false);
    tensors.remove(0);
    tensors[0].name = "encoder.layer.bad.attention.self.query.weight".to_string();
    let (inventory, _) = parse_inventory(&tensors, 0);
    let invalid = RawBertConfigFacts::new(V, H, L, A, I, P, 0, 1e-5);
    assert_eq!(
        match_prepared_bert_inventory(invalid, &geometry_limits(), &inventory).unwrap_err(),
        PreparedBertTensorMatchError::Geometry(PreparedBertFactError::Zero(
            BertFactAxis::TypeVocabSize
        ))
    );
}

#[test]
fn subset_count_mismatch_precedes_shape_validation() {
    let mut tensors = required_tensors(false);
    tensors.remove(0);
    tensors
        .iter_mut()
        .find(|tensor| tensor.name == "encoder.layer.0.intermediate.dense.weight")
        .unwrap()
        .dimensions = vec![H, I];
    assert_eq!(
        match_tensors(&tensors, 0).unwrap_err(),
        PreparedBertTensorMatchError::RequiredTensorCountMismatch {
            expected: 5 + 16 * L,
            actual: 5 + 16 * L - 1,
        }
    );
}

#[test]
fn checked_accumulators_pin_exact_and_overflow_boundaries() {
    for expression in [
        BertMatchExpression::RequiredTensorCount,
        BertMatchExpression::F32CopiedElements,
        BertMatchExpression::F16Elements,
        BertMatchExpression::Bf16Elements,
        BertMatchExpression::RequiredElements,
    ] {
        assert_eq!(checked_add(u64::MAX - 1, 1, expression), Ok(u64::MAX));
        assert_eq!(
            checked_add(u64::MAX, 1, expression),
            Err(PreparedBertTensorMatchError::ArithmeticOverflow(expression))
        );
    }
}
