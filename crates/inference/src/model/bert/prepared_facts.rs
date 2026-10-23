//! Dormant checked BERT facts for sealed native preparation.
//!
//! This module deliberately performs no parsing, filesystem, model loading,
//! allocation, or serving work. It validates raw numeric facts and computes a
//! narrowly scoped logical weight-footprint estimate for a later prepared
//! loader.

use super::BertConfig;
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BertFactAxis {
    VocabSize,
    HiddenSize,
    HiddenLayers,
    AttentionHeads,
    IntermediateSize,
    PositionEmbeddings,
    TypeVocabSize,
    RequiredTensors,
    RequiredElements,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BertFactExpression {
    RequiredTensorCount,
    RequiredTensorElements,
    FusedQkvElements,
    CensusElements,
    RequiredSourcePayloadBytes,
    MaterializedF32Bytes,
    FusedQkvBytes,
    LogicalWeightPayloadBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertFactError {
    Zero(BertFactAxis),
    Exceeded {
        axis: BertFactAxis,
        actual: u64,
        limit: u64,
    },
    PlatformUnrepresentable {
        axis: BertFactAxis,
        value: u64,
    },
    InvalidLayerNormEpsilon,
    HiddenHeadRemainder {
        hidden_size: u64,
        attention_heads: u64,
    },
    IncompletePooler {
        weight: bool,
        bias: bool,
    },
    ArithmeticOverflow(BertFactExpression),
    RequiredTensorCountMismatch {
        expected: u64,
        actual: u64,
    },
    RequiredElementCountMismatch {
        expected: u64,
        actual: u64,
    },
    MappedWeightFileTooSmall {
        mapped: u64,
        required_source_payload: u64,
    },
}

type FactResult<T> = Result<T, PreparedBertFactError>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct RawBertConfigFacts {
    vocab_size: u64,
    hidden_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    intermediate_size: u64,
    max_position_embeddings: u64,
    type_vocab_size: u64,
    layer_norm_eps: f64,
}

impl RawBertConfigFacts {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        vocab_size: u64,
        hidden_size: u64,
        num_hidden_layers: u64,
        num_attention_heads: u64,
        intermediate_size: u64,
        max_position_embeddings: u64,
        type_vocab_size: u64,
        layer_norm_eps: f64,
    ) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BertGeometryLimits {
    max_vocab_size: NonZeroU64,
    max_hidden_size: NonZeroU64,
    max_hidden_layers: NonZeroU64,
    max_attention_heads: NonZeroU64,
    max_intermediate_size: NonZeroU64,
    max_position_embeddings: NonZeroU64,
    max_type_vocab_size: NonZeroU64,
    max_required_tensors: NonZeroU64,
    max_required_elements: NonZeroU64,
}

impl BertGeometryLimits {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        max_vocab_size: NonZeroU64,
        max_hidden_size: NonZeroU64,
        max_hidden_layers: NonZeroU64,
        max_attention_heads: NonZeroU64,
        max_intermediate_size: NonZeroU64,
        max_position_embeddings: NonZeroU64,
        max_type_vocab_size: NonZeroU64,
        max_required_tensors: NonZeroU64,
        max_required_elements: NonZeroU64,
    ) -> Self {
        Self {
            max_vocab_size,
            max_hidden_size,
            max_hidden_layers,
            max_attention_heads,
            max_intermediate_size,
            max_position_embeddings,
            max_type_vocab_size,
            max_required_tensors,
            max_required_elements,
        }
    }

    #[cfg(test)]
    fn all(
        raw_axis_limit: NonZeroU64,
        max_required_tensors: NonZeroU64,
        max_required_elements: NonZeroU64,
    ) -> Self {
        Self::new(
            raw_axis_limit,
            raw_axis_limit,
            raw_axis_limit,
            raw_axis_limit,
            raw_axis_limit,
            raw_axis_limit,
            raw_axis_limit,
            max_required_tensors,
            max_required_elements,
        )
    }

    fn limit(self, axis: BertFactAxis) -> u64 {
        match axis {
            BertFactAxis::VocabSize => self.max_vocab_size.get(),
            BertFactAxis::HiddenSize => self.max_hidden_size.get(),
            BertFactAxis::HiddenLayers => self.max_hidden_layers.get(),
            BertFactAxis::AttentionHeads => self.max_attention_heads.get(),
            BertFactAxis::IntermediateSize => self.max_intermediate_size.get(),
            BertFactAxis::PositionEmbeddings => self.max_position_embeddings.get(),
            BertFactAxis::TypeVocabSize => self.max_type_vocab_size.get(),
            BertFactAxis::RequiredTensors => self.max_required_tensors.get(),
            BertFactAxis::RequiredElements => self.max_required_elements.get(),
        }
    }

    fn validate(self, axis: BertFactAxis, actual: u64) -> FactResult<()> {
        let limit = self.limit(axis);
        if actual > limit {
            return Err(PreparedBertFactError::Exceeded {
                axis,
                actual,
                limit,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BertPoolerMembers {
    weight: bool,
    bias: bool,
}

impl BertPoolerMembers {
    pub(super) fn new(weight: bool, bias: bool) -> Self {
        Self { weight, bias }
    }

    fn validate(self) -> FactResult<bool> {
        if self.weight != self.bias {
            return Err(PreparedBertFactError::IncompletePooler {
                weight: self.weight,
                bias: self.bias,
            });
        }
        Ok(self.weight)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertGeometry {
    vocab_size: NonZeroUsize,
    hidden_size: NonZeroUsize,
    hidden_layers: NonZeroUsize,
    attention_heads: NonZeroUsize,
    head_dim: NonZeroUsize,
    intermediate_size: NonZeroUsize,
    position_embeddings: NonZeroUsize,
    type_vocab_size: NonZeroUsize,
    layer_norm_eps_bits: u32,
    has_pooler: bool,
    required_tensor_count: u64,
    required_elements: u64,
    fused_qkv_elements: u64,
}

impl PreparedBertGeometry {
    pub(super) fn vocab_size(self) -> usize {
        self.vocab_size.get()
    }

    pub(super) fn hidden_size(self) -> usize {
        self.hidden_size.get()
    }

    pub(super) fn hidden_layers(self) -> usize {
        self.hidden_layers.get()
    }

    pub(super) fn attention_heads(self) -> usize {
        self.attention_heads.get()
    }

    pub(super) fn head_dim(self) -> usize {
        self.head_dim.get()
    }

    pub(super) fn intermediate_size(self) -> usize {
        self.intermediate_size.get()
    }

    pub(super) fn position_embeddings(self) -> usize {
        self.position_embeddings.get()
    }

    pub(super) fn type_vocab_size(self) -> usize {
        self.type_vocab_size.get()
    }

    pub(super) fn layer_norm_eps(self) -> f32 {
        f32::from_bits(self.layer_norm_eps_bits)
    }

    pub(super) fn has_pooler(self) -> bool {
        self.has_pooler
    }

    pub(super) fn required_tensor_count(self) -> u64 {
        self.required_tensor_count
    }

    pub(super) fn required_elements(self) -> u64 {
        self.required_elements
    }

    pub(super) fn fused_qkv_elements(self) -> u64 {
        self.fused_qkv_elements
    }

    pub(super) fn bert_config(self) -> BertConfig {
        BertConfig {
            vocab_size: self.vocab_size(),
            hidden_size: self.hidden_size(),
            num_hidden_layers: self.hidden_layers(),
            num_attention_heads: self.attention_heads(),
            intermediate_size: self.intermediate_size(),
            max_position_embeddings: self.position_embeddings(),
            type_vocab_size: self.type_vocab_size(),
            layer_norm_eps: self.layer_norm_eps(),
        }
    }

    #[cfg(test)]
    fn for_test(required_elements: u64, fused_qkv_elements: u64) -> Self {
        let one = NonZeroUsize::MIN;
        Self {
            vocab_size: one,
            hidden_size: one,
            hidden_layers: one,
            attention_heads: one,
            head_dim: one,
            intermediate_size: one,
            position_embeddings: one,
            type_vocab_size: one,
            layer_norm_eps_bits: 1.0_f32.to_bits(),
            has_pooler: false,
            required_tensor_count: 1,
            required_elements,
            fused_qkv_elements,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RequiredBertTensorCensus {
    tensor_count: u64,
    f32_zero_copy_elements: u64,
    f32_copied_elements: u64,
    f16_elements: u64,
    bf16_elements: u64,
}

impl RequiredBertTensorCensus {
    pub(super) fn new(
        tensor_count: u64,
        f32_zero_copy_elements: u64,
        f32_copied_elements: u64,
        f16_elements: u64,
        bf16_elements: u64,
    ) -> Self {
        Self {
            tensor_count,
            f32_zero_copy_elements,
            f32_copied_elements,
            f16_elements,
            bf16_elements,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BertWeightPayloadFootprint {
    mapped_weight_file_bytes: u64,
    required_source_payload_bytes: u64,
    materialized_f32_bytes: u64,
    fused_qkv_bytes: u64,
    logical_weight_payload_bytes: u64,
}

impl BertWeightPayloadFootprint {
    pub(super) fn mapped_weight_file_bytes(self) -> u64 {
        self.mapped_weight_file_bytes
    }

    pub(super) fn required_source_payload_bytes(self) -> u64 {
        self.required_source_payload_bytes
    }

    pub(super) fn materialized_f32_bytes(self) -> u64 {
        self.materialized_f32_bytes
    }

    pub(super) fn fused_qkv_bytes(self) -> u64 {
        self.fused_qkv_bytes
    }

    /// Logical mapped/decoded/fused weight-payload accounting only.
    ///
    /// This is not RSS, peak memory, or a complete retained-resource bound.
    pub(super) fn logical_weight_payload_bytes(self) -> u64 {
        self.logical_weight_payload_bytes
    }
}

pub(super) fn analyze_prepared_geometry(
    raw: RawBertConfigFacts,
    limits: &BertGeometryLimits,
    pooler: BertPoolerMembers,
) -> FactResult<PreparedBertGeometry> {
    let platform_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
    analyze_prepared_geometry_with_platform_max(raw, limits, pooler, platform_max)
}

fn analyze_prepared_geometry_with_platform_max(
    raw: RawBertConfigFacts,
    limits: &BertGeometryLimits,
    pooler: BertPoolerMembers,
    platform_max: u64,
) -> FactResult<PreparedBertGeometry> {
    let fields = [
        (BertFactAxis::VocabSize, raw.vocab_size),
        (BertFactAxis::HiddenSize, raw.hidden_size),
        (BertFactAxis::HiddenLayers, raw.num_hidden_layers),
        (BertFactAxis::AttentionHeads, raw.num_attention_heads),
        (BertFactAxis::IntermediateSize, raw.intermediate_size),
        (
            BertFactAxis::PositionEmbeddings,
            raw.max_position_embeddings,
        ),
        (BertFactAxis::TypeVocabSize, raw.type_vocab_size),
    ];
    for (axis, value) in fields {
        if value == 0 {
            return Err(PreparedBertFactError::Zero(axis));
        }
        limits.validate(axis, value)?;
        validate_platform(axis, value, platform_max)?;
    }
    if !raw.layer_norm_eps.is_finite() || raw.layer_norm_eps <= 0.0 {
        return Err(PreparedBertFactError::InvalidLayerNormEpsilon);
    }
    let has_pooler = pooler.validate()?;
    if !raw.hidden_size.is_multiple_of(raw.num_attention_heads) {
        return Err(PreparedBertFactError::HiddenHeadRemainder {
            hidden_size: raw.hidden_size,
            attention_heads: raw.num_attention_heads,
        });
    }

    let required_tensor_count = checked_required_tensor_count(raw.num_hidden_layers, has_pooler)?;
    let required_elements = checked_required_tensor_elements(
        raw.vocab_size,
        raw.hidden_size,
        raw.num_hidden_layers,
        raw.intermediate_size,
        raw.max_position_embeddings,
        raw.type_vocab_size,
        has_pooler,
    )?;
    let fused_qkv_elements = checked_fused_qkv_elements(raw.hidden_size, raw.num_hidden_layers)?;
    limits.validate(BertFactAxis::RequiredTensors, required_tensor_count)?;
    limits.validate(BertFactAxis::RequiredElements, required_elements)?;

    let vocab_size = to_nonzero_usize(BertFactAxis::VocabSize, raw.vocab_size)?;
    let hidden_size = to_nonzero_usize(BertFactAxis::HiddenSize, raw.hidden_size)?;
    let hidden_layers = to_nonzero_usize(BertFactAxis::HiddenLayers, raw.num_hidden_layers)?;
    let attention_heads = to_nonzero_usize(BertFactAxis::AttentionHeads, raw.num_attention_heads)?;
    let intermediate_size =
        to_nonzero_usize(BertFactAxis::IntermediateSize, raw.intermediate_size)?;
    let position_embeddings = to_nonzero_usize(
        BertFactAxis::PositionEmbeddings,
        raw.max_position_embeddings,
    )?;
    let type_vocab_size = to_nonzero_usize(BertFactAxis::TypeVocabSize, raw.type_vocab_size)?;
    let head_dim = NonZeroUsize::new(hidden_size.get() / attention_heads.get()).ok_or(
        PreparedBertFactError::HiddenHeadRemainder {
            hidden_size: raw.hidden_size,
            attention_heads: raw.num_attention_heads,
        },
    )?;
    #[allow(clippy::cast_possible_truncation)]
    let layer_norm_eps = raw.layer_norm_eps as f32;
    if !layer_norm_eps.is_finite() || layer_norm_eps <= 0.0 {
        return Err(PreparedBertFactError::InvalidLayerNormEpsilon);
    }

    Ok(PreparedBertGeometry {
        vocab_size,
        hidden_size,
        hidden_layers,
        attention_heads,
        head_dim,
        intermediate_size,
        position_embeddings,
        type_vocab_size,
        layer_norm_eps_bits: layer_norm_eps.to_bits(),
        has_pooler,
        required_tensor_count,
        required_elements,
        fused_qkv_elements,
    })
}

fn validate_platform(axis: BertFactAxis, value: u64, platform_max: u64) -> FactResult<()> {
    if value > platform_max {
        return Err(PreparedBertFactError::PlatformUnrepresentable { axis, value });
    }
    Ok(())
}

fn to_nonzero_usize(axis: BertFactAxis, value: u64) -> FactResult<NonZeroUsize> {
    let value = usize::try_from(value)
        .map_err(|_| PreparedBertFactError::PlatformUnrepresentable { axis, value })?;
    NonZeroUsize::new(value).ok_or(PreparedBertFactError::Zero(axis))
}

fn overflow(expression: BertFactExpression) -> PreparedBertFactError {
    PreparedBertFactError::ArithmeticOverflow(expression)
}

fn checked_required_tensor_count(hidden_layers: u64, has_pooler: bool) -> FactResult<u64> {
    let expression = BertFactExpression::RequiredTensorCount;
    hidden_layers
        .checked_mul(16)
        .and_then(|value| value.checked_add(5))
        .and_then(|value| value.checked_add(if has_pooler { 2 } else { 0 }))
        .ok_or_else(|| overflow(expression))
}

#[allow(clippy::too_many_arguments)]
fn checked_required_tensor_elements(
    vocab_size: u64,
    hidden_size: u64,
    hidden_layers: u64,
    intermediate_size: u64,
    position_embeddings: u64,
    type_vocab_size: u64,
    has_pooler: bool,
) -> FactResult<u64> {
    let expression = BertFactExpression::RequiredTensorElements;
    let hidden_squared = hidden_size
        .checked_mul(hidden_size)
        .ok_or_else(|| overflow(expression))?;
    let embedding_terms = vocab_size
        .checked_add(position_embeddings)
        .and_then(|value| value.checked_add(type_vocab_size))
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| overflow(expression))?;
    let embedding_elements = hidden_size
        .checked_mul(embedding_terms)
        .ok_or_else(|| overflow(expression))?;
    let layer_elements = hidden_squared
        .checked_mul(4)
        .and_then(|value| {
            hidden_size
                .checked_mul(intermediate_size)
                .and_then(|term| term.checked_mul(2))
                .and_then(|term| value.checked_add(term))
        })
        .and_then(|value| {
            hidden_size
                .checked_mul(9)
                .and_then(|term| value.checked_add(term))
        })
        .and_then(|value| value.checked_add(intermediate_size))
        .ok_or_else(|| overflow(expression))?;
    let all_layer_elements = hidden_layers
        .checked_mul(layer_elements)
        .ok_or_else(|| overflow(expression))?;
    let pooler_elements = if has_pooler {
        hidden_squared
            .checked_add(hidden_size)
            .ok_or_else(|| overflow(expression))?
    } else {
        0
    };
    embedding_elements
        .checked_add(all_layer_elements)
        .and_then(|value| value.checked_add(pooler_elements))
        .ok_or_else(|| overflow(expression))
}

fn checked_fused_qkv_elements(hidden_size: u64, hidden_layers: u64) -> FactResult<u64> {
    let expression = BertFactExpression::FusedQkvElements;
    let per_layer = hidden_size
        .checked_mul(hidden_size)
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| {
            hidden_size
                .checked_mul(3)
                .and_then(|bias| value.checked_add(bias))
        })
        .ok_or_else(|| overflow(expression))?;
    hidden_layers
        .checked_mul(per_layer)
        .ok_or_else(|| overflow(expression))
}

fn checked_required_census_elements(census: &RequiredBertTensorCensus) -> FactResult<u64> {
    census
        .f32_zero_copy_elements
        .checked_add(census.f32_copied_elements)
        .and_then(|value| value.checked_add(census.f16_elements))
        .and_then(|value| value.checked_add(census.bf16_elements))
        .ok_or_else(|| overflow(BertFactExpression::CensusElements))
}

fn checked_required_source_payload_bytes(census: &RequiredBertTensorCensus) -> FactResult<u64> {
    let expression = BertFactExpression::RequiredSourcePayloadBytes;
    let f32_bytes = census
        .f32_zero_copy_elements
        .checked_add(census.f32_copied_elements)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| overflow(expression))?;
    let f16_bytes = census
        .f16_elements
        .checked_mul(2)
        .ok_or_else(|| overflow(expression))?;
    let bf16_bytes = census
        .bf16_elements
        .checked_mul(2)
        .ok_or_else(|| overflow(expression))?;
    f32_bytes
        .checked_add(f16_bytes)
        .and_then(|value| value.checked_add(bf16_bytes))
        .ok_or_else(|| overflow(expression))
}

fn checked_materialized_f32_bytes(census: &RequiredBertTensorCensus) -> FactResult<u64> {
    let expression = BertFactExpression::MaterializedF32Bytes;
    census
        .f32_copied_elements
        .checked_add(census.f16_elements)
        .and_then(|value| value.checked_add(census.bf16_elements))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| overflow(expression))
}

fn checked_fused_qkv_bytes(geometry: &PreparedBertGeometry) -> FactResult<u64> {
    geometry
        .fused_qkv_elements
        .checked_mul(4)
        .ok_or_else(|| overflow(BertFactExpression::FusedQkvBytes))
}

pub(super) fn checked_logical_weight_payload_footprint(
    geometry: &PreparedBertGeometry,
    required: RequiredBertTensorCensus,
    mapped_weight_file_bytes: u64,
) -> FactResult<BertWeightPayloadFootprint> {
    if required.tensor_count != geometry.required_tensor_count {
        return Err(PreparedBertFactError::RequiredTensorCountMismatch {
            expected: geometry.required_tensor_count,
            actual: required.tensor_count,
        });
    }
    let actual_elements = checked_required_census_elements(&required)?;
    if actual_elements != geometry.required_elements {
        return Err(PreparedBertFactError::RequiredElementCountMismatch {
            expected: geometry.required_elements,
            actual: actual_elements,
        });
    }
    let required_source_payload_bytes = checked_required_source_payload_bytes(&required)?;
    if mapped_weight_file_bytes < required_source_payload_bytes {
        return Err(PreparedBertFactError::MappedWeightFileTooSmall {
            mapped: mapped_weight_file_bytes,
            required_source_payload: required_source_payload_bytes,
        });
    }
    let materialized_f32_bytes = checked_materialized_f32_bytes(&required)?;
    let fused_qkv_bytes = checked_fused_qkv_bytes(geometry)?;
    let logical_weight_payload_bytes = mapped_weight_file_bytes
        .checked_add(materialized_f32_bytes)
        .and_then(|value| value.checked_add(fused_qkv_bytes))
        .ok_or_else(|| overflow(BertFactExpression::LogicalWeightPayloadBytes))?;

    Ok(BertWeightPayloadFootprint {
        mapped_weight_file_bytes,
        required_source_payload_bytes,
        materialized_f32_bytes,
        fused_qkv_bytes,
        logical_weight_payload_bytes,
    })
}

#[cfg(test)]
mod tests;
