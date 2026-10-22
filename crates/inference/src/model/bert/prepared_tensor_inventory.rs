//! Dormant required-tensor matching for sealed BERT preparation.
//!
//! This module consumes an already bounded, duplicate-free SafeTensors inventory.
//! It derives pooler presence, validates checked BERT geometry, and matches the
//! exact required tensor-name and shape closure without allocating. Unrelated
//! tensors remain permitted and identity-bearing in the complete inventory.
//!
//! The returned census conservatively treats every required F32 tensor as copied.
//! A file-offset alignment fact alone does not authorize a mapping or zero-copy
//! disposition. This module performs no filesystem access, payload reads,
//! mapping, model loading, publication, or serving work and has no live caller.

use super::prepared_facts::{
    BertGeometryLimits, BertPoolerMembers, PreparedBertFactError, PreparedBertGeometry,
    RawBertConfigFacts, RequiredBertTensorCensus, analyze_prepared_geometry,
};
use crate::weights::prepared_safetensors_inventory::{
    PreparedSafetensorsDtype, PreparedSafetensorsInventoryFacts, PreparedSafetensorsTensorView,
};

const LAYER_PREFIX: &[u8] = b"encoder.layer.";

const LAYER_SUFFIXES: [(&[u8], PreparedBertLayerTensor); 16] = [
    (
        b".attention.self.query.weight",
        PreparedBertLayerTensor::QueryWeight,
    ),
    (
        b".attention.self.query.bias",
        PreparedBertLayerTensor::QueryBias,
    ),
    (
        b".attention.self.key.weight",
        PreparedBertLayerTensor::KeyWeight,
    ),
    (
        b".attention.self.key.bias",
        PreparedBertLayerTensor::KeyBias,
    ),
    (
        b".attention.self.value.weight",
        PreparedBertLayerTensor::ValueWeight,
    ),
    (
        b".attention.self.value.bias",
        PreparedBertLayerTensor::ValueBias,
    ),
    (
        b".attention.output.dense.weight",
        PreparedBertLayerTensor::AttentionOutputWeight,
    ),
    (
        b".attention.output.dense.bias",
        PreparedBertLayerTensor::AttentionOutputBias,
    ),
    (
        b".attention.output.LayerNorm.weight",
        PreparedBertLayerTensor::AttentionLayerNormWeight,
    ),
    (
        b".attention.output.LayerNorm.bias",
        PreparedBertLayerTensor::AttentionLayerNormBias,
    ),
    (
        b".intermediate.dense.weight",
        PreparedBertLayerTensor::IntermediateWeight,
    ),
    (
        b".intermediate.dense.bias",
        PreparedBertLayerTensor::IntermediateBias,
    ),
    (
        b".output.dense.weight",
        PreparedBertLayerTensor::OutputWeight,
    ),
    (b".output.dense.bias", PreparedBertLayerTensor::OutputBias),
    (
        b".output.LayerNorm.weight",
        PreparedBertLayerTensor::OutputLayerNormWeight,
    ),
    (
        b".output.LayerNorm.bias",
        PreparedBertLayerTensor::OutputLayerNormBias,
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertLayerTensor {
    QueryWeight,
    QueryBias,
    KeyWeight,
    KeyBias,
    ValueWeight,
    ValueBias,
    AttentionOutputWeight,
    AttentionOutputBias,
    AttentionLayerNormWeight,
    AttentionLayerNormBias,
    IntermediateWeight,
    IntermediateBias,
    OutputWeight,
    OutputBias,
    OutputLayerNormWeight,
    OutputLayerNormBias,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTensor {
    WordEmbeddingsWeight,
    PositionEmbeddingsWeight,
    TokenTypeEmbeddingsWeight,
    EmbeddingsLayerNormWeight,
    EmbeddingsLayerNormBias,
    Encoder {
        layer: u64,
        member: PreparedBertLayerTensor,
    },
    PoolerWeight,
    PoolerBias,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BertLayerIndexFault {
    Empty,
    LeadingZero,
    NonAsciiDigit { at: usize },
    Overflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BertShapeFault {
    Rank {
        expected: usize,
        actual: usize,
    },
    Dimension {
        axis: usize,
        expected: usize,
        actual: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BertMatchExpression {
    RequiredTensorCount,
    F32CopiedElements,
    F16Elements,
    Bf16Elements,
    RequiredElements,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertTensorMatchError {
    Geometry(PreparedBertFactError),
    InvalidLayerIndex {
        member: PreparedBertLayerTensor,
        fault: BertLayerIndexFault,
    },
    LayerIndexOutOfRange {
        index: u64,
        hidden_layers: usize,
    },
    RequiredTensorCountMismatch {
        expected: u64,
        actual: u64,
    },
    InvalidShape {
        tensor: PreparedBertTensor,
        fault: BertShapeFault,
    },
    ArithmeticOverflow(BertMatchExpression),
    RequiredElementCountMismatch {
        expected: u64,
        actual: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MatchedBertInventoryFacts {
    geometry: PreparedBertGeometry,
    required_tensor_census: RequiredBertTensorCensus,
}

impl MatchedBertInventoryFacts {
    pub(super) fn geometry(self) -> PreparedBertGeometry {
        self.geometry
    }

    pub(super) fn required_tensor_census(self) -> RequiredBertTensorCensus {
        self.required_tensor_census
    }
}

pub(super) fn match_prepared_bert_inventory(
    raw: RawBertConfigFacts,
    limits: &BertGeometryLimits,
    inventory: &PreparedSafetensorsInventoryFacts,
) -> Result<MatchedBertInventoryFacts, PreparedBertTensorMatchError> {
    let pooler = inspect_pooler_members(inventory);
    let geometry = analyze_prepared_geometry(raw, limits, pooler)
        .map_err(PreparedBertTensorMatchError::Geometry)?;

    let mut required_tensor_count = 0_u64;
    for tensor in inventory.tensors() {
        if classify_required_tensor(tensor.name_bytes(), geometry.hidden_layers())?.is_some() {
            required_tensor_count = checked_add(
                required_tensor_count,
                1,
                BertMatchExpression::RequiredTensorCount,
            )?;
        }
    }
    if required_tensor_count != geometry.required_tensor_count() {
        return Err(PreparedBertTensorMatchError::RequiredTensorCountMismatch {
            expected: geometry.required_tensor_count(),
            actual: required_tensor_count,
        });
    }

    let mut f32_copied_elements = 0_u64;
    let mut f16_elements = 0_u64;
    let mut bf16_elements = 0_u64;
    let mut required_elements = 0_u64;
    for tensor in inventory.tensors() {
        let Some(required) =
            classify_required_tensor(tensor.name_bytes(), geometry.hidden_layers())?
        else {
            continue;
        };
        validate_required_shape(tensor, required, geometry)?;
        required_elements = checked_add(
            required_elements,
            tensor.elements(),
            BertMatchExpression::RequiredElements,
        )?;
        match tensor.dtype() {
            PreparedSafetensorsDtype::F32 => {
                f32_copied_elements = checked_add(
                    f32_copied_elements,
                    tensor.elements(),
                    BertMatchExpression::F32CopiedElements,
                )?;
            }
            PreparedSafetensorsDtype::F16 => {
                f16_elements = checked_add(
                    f16_elements,
                    tensor.elements(),
                    BertMatchExpression::F16Elements,
                )?;
            }
            PreparedSafetensorsDtype::Bf16 => {
                bf16_elements = checked_add(
                    bf16_elements,
                    tensor.elements(),
                    BertMatchExpression::Bf16Elements,
                )?;
            }
        }
    }
    if required_elements != geometry.required_elements() {
        return Err(PreparedBertTensorMatchError::RequiredElementCountMismatch {
            expected: geometry.required_elements(),
            actual: required_elements,
        });
    }

    Ok(MatchedBertInventoryFacts {
        geometry,
        required_tensor_census: RequiredBertTensorCensus::new(
            required_tensor_count,
            0,
            f32_copied_elements,
            f16_elements,
            bf16_elements,
        ),
    })
}

fn inspect_pooler_members(inventory: &PreparedSafetensorsInventoryFacts) -> BertPoolerMembers {
    let mut weight = false;
    let mut bias = false;
    for tensor in inventory.tensors() {
        match tensor.name_bytes() {
            b"pooler.dense.weight" => weight = true,
            b"pooler.dense.bias" => bias = true,
            _ => {}
        }
    }
    BertPoolerMembers::new(weight, bias)
}

fn classify_required_tensor(
    name: &[u8],
    hidden_layers: usize,
) -> Result<Option<PreparedBertTensor>, PreparedBertTensorMatchError> {
    let fixed = match name {
        b"embeddings.word_embeddings.weight" => Some(PreparedBertTensor::WordEmbeddingsWeight),
        b"embeddings.position_embeddings.weight" => {
            Some(PreparedBertTensor::PositionEmbeddingsWeight)
        }
        b"embeddings.token_type_embeddings.weight" => {
            Some(PreparedBertTensor::TokenTypeEmbeddingsWeight)
        }
        b"embeddings.LayerNorm.weight" => Some(PreparedBertTensor::EmbeddingsLayerNormWeight),
        b"embeddings.LayerNorm.bias" => Some(PreparedBertTensor::EmbeddingsLayerNormBias),
        b"pooler.dense.weight" => Some(PreparedBertTensor::PoolerWeight),
        b"pooler.dense.bias" => Some(PreparedBertTensor::PoolerBias),
        _ => None,
    };
    if fixed.is_some() {
        return Ok(fixed);
    }

    let Some(rest) = name.strip_prefix(LAYER_PREFIX) else {
        return Ok(None);
    };
    let Some(separator) = rest.iter().position(|byte| *byte == b'.') else {
        return Ok(None);
    };
    let (index_bytes, suffix) = rest.split_at(separator);
    let Some((_, member)) = LAYER_SUFFIXES.iter().find(|(known, _)| *known == suffix) else {
        return Ok(None);
    };
    let member = *member;
    let index = parse_layer_index(index_bytes)
        .map_err(|fault| PreparedBertTensorMatchError::InvalidLayerIndex { member, fault })?;
    if index >= hidden_layers as u64 {
        return Err(PreparedBertTensorMatchError::LayerIndexOutOfRange {
            index,
            hidden_layers,
        });
    }
    Ok(Some(PreparedBertTensor::Encoder {
        layer: index,
        member,
    }))
}

fn parse_layer_index(index: &[u8]) -> Result<u64, BertLayerIndexFault> {
    if index.is_empty() {
        return Err(BertLayerIndexFault::Empty);
    }
    let mut value = 0_u64;
    for (at, byte) in index.iter().copied().enumerate() {
        if !byte.is_ascii_digit() {
            return Err(BertLayerIndexFault::NonAsciiDigit { at });
        }
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add(u64::from(byte - b'0')))
            .ok_or(BertLayerIndexFault::Overflow)?;
    }
    if index.len() > 1 && index[0] == b'0' {
        return Err(BertLayerIndexFault::LeadingZero);
    }
    Ok(value)
}

fn validate_required_shape(
    tensor: PreparedSafetensorsTensorView<'_>,
    required: PreparedBertTensor,
    geometry: PreparedBertGeometry,
) -> Result<(), PreparedBertTensorMatchError> {
    let h = geometry.hidden_size();
    let expected = match required {
        PreparedBertTensor::WordEmbeddingsWeight => (geometry.vocab_size(), Some(h)),
        PreparedBertTensor::PositionEmbeddingsWeight => (geometry.position_embeddings(), Some(h)),
        PreparedBertTensor::TokenTypeEmbeddingsWeight => (geometry.type_vocab_size(), Some(h)),
        PreparedBertTensor::EmbeddingsLayerNormWeight
        | PreparedBertTensor::EmbeddingsLayerNormBias
        | PreparedBertTensor::PoolerBias => (h, None),
        PreparedBertTensor::PoolerWeight => (h, Some(h)),
        PreparedBertTensor::Encoder { member, .. } => match member {
            PreparedBertLayerTensor::QueryWeight
            | PreparedBertLayerTensor::KeyWeight
            | PreparedBertLayerTensor::ValueWeight
            | PreparedBertLayerTensor::AttentionOutputWeight => (h, Some(h)),
            PreparedBertLayerTensor::QueryBias
            | PreparedBertLayerTensor::KeyBias
            | PreparedBertLayerTensor::ValueBias
            | PreparedBertLayerTensor::AttentionOutputBias
            | PreparedBertLayerTensor::AttentionLayerNormWeight
            | PreparedBertLayerTensor::AttentionLayerNormBias
            | PreparedBertLayerTensor::OutputBias
            | PreparedBertLayerTensor::OutputLayerNormWeight
            | PreparedBertLayerTensor::OutputLayerNormBias => (h, None),
            PreparedBertLayerTensor::IntermediateWeight => (geometry.intermediate_size(), Some(h)),
            PreparedBertLayerTensor::IntermediateBias => (geometry.intermediate_size(), None),
            PreparedBertLayerTensor::OutputWeight => (h, Some(geometry.intermediate_size())),
        },
    };
    let expected_rank = usize::from(expected.1.is_some()) + 1;
    let actual = tensor.dimensions();
    if actual.len() != expected_rank {
        return Err(PreparedBertTensorMatchError::InvalidShape {
            tensor: required,
            fault: BertShapeFault::Rank {
                expected: expected_rank,
                actual: actual.len(),
            },
        });
    }
    validate_dimension(required, actual, 0, expected.0)?;
    if let Some(second) = expected.1 {
        validate_dimension(required, actual, 1, second)?;
    }
    Ok(())
}

fn validate_dimension(
    tensor: PreparedBertTensor,
    actual: &[u64],
    axis: usize,
    expected: usize,
) -> Result<(), PreparedBertTensorMatchError> {
    let actual = actual[axis];
    if actual != expected as u64 {
        return Err(PreparedBertTensorMatchError::InvalidShape {
            tensor,
            fault: BertShapeFault::Dimension {
                axis,
                expected,
                actual,
            },
        });
    }
    Ok(())
}

fn checked_add(
    left: u64,
    right: u64,
    expression: BertMatchExpression,
) -> Result<u64, PreparedBertTensorMatchError> {
    left.checked_add(right)
        .ok_or(PreparedBertTensorMatchError::ArithmeticOverflow(expression))
}

#[cfg(test)]
mod tests;
