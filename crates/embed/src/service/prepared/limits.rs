//! Dormant checked limits and accounting for sealed native preparation.
//!
//! This module deliberately performs no filesystem, parser, tokenizer, or
//! model work. It is a private arithmetic substrate until ADR-088 is accepted
//! and the inference crate exposes bounded preparation facts.

use std::num::{NonZeroU64, NonZeroUsize};

const ATTESTATION_CHUNK_BYTES: u64 = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LimitAxis {
    Files,
    PathBytes,
    TotalPathBytes,
    AuxFileBytes,
    IndexFileBytes,
    WeightFileBytes,
    SnapshotBytes,
    CensusEntries,
    CensusMetadataBytes,
    HeaderBytes,
    Tensors,
    TensorNameBytes,
    Rank,
    Dimension,
    AggregateElements,
    TensorMetadataBytes,
    VocabSize,
    HiddenSize,
    HiddenLayers,
    AttentionHeads,
    IntermediateSize,
    PositionEmbeddings,
    TypeVocabSize,
    ConfigBytes,
    TokenizerFileBytes,
    ConfigParseWorkBytes,
    TokenizerParseWorkBytes,
    SafetensorsParseWorkBytes,
    ModelRetainedBytes,
    TokenizerRetainedBytes,
    DescriptorControlBytes,
    ModelLoadWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ChargeExpression {
    AttestationReportBytes,
    HeaderFrame,
    RequiredTensorCount,
    RequiredTensorElements,
    FusedQkvElements,
    TensorElements,
    TensorBytes,
    TensorCensus,
    RetainedCharge,
    WorkCharge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BertGeometryFault {
    Zero(LimitAxis),
    AttentionHeadRemainder,
    InvalidLayerNormEpsilon,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparationLimitError {
    Exceeded {
        axis: LimitAxis,
        actual: u64,
        limit: u64,
    },
    ArithmeticOverflow {
        expression: ChargeExpression,
    },
    PlatformUnrepresentable {
        axis: LimitAxis,
        value: u64,
    },
    InvalidRelation {
        lower: LimitAxis,
        upper: LimitAxis,
    },
    InvalidGeometry {
        fault: BertGeometryFault,
    },
}

pub(super) type LimitResult<T> = Result<T, PreparationLimitError>;

#[derive(Clone, Copy, Debug)]
pub(super) struct InventoryCeilings {
    max_files: NonZeroUsize,
    max_path_bytes: NonZeroUsize,
    max_total_path_bytes: NonZeroU64,
    max_aux_file_bytes: NonZeroU64,
    max_index_file_bytes: NonZeroU64,
    max_weight_file_bytes: NonZeroU64,
    max_snapshot_bytes: NonZeroU64,
    max_census_metadata_bytes: NonZeroU64,
    max_census_entries: NonZeroUsize,
}

impl InventoryCeilings {
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_files: NonZeroUsize,
        max_path_bytes: NonZeroUsize,
        max_total_path_bytes: NonZeroU64,
        max_aux_file_bytes: NonZeroU64,
        max_index_file_bytes: NonZeroU64,
        max_weight_file_bytes: NonZeroU64,
        max_snapshot_bytes: NonZeroU64,
        max_census_metadata_bytes: NonZeroU64,
        max_census_entries: NonZeroUsize,
    ) -> Self {
        Self {
            max_files,
            max_path_bytes,
            max_total_path_bytes,
            max_aux_file_bytes,
            max_index_file_bytes,
            max_weight_file_bytes,
            max_snapshot_bytes,
            max_census_metadata_bytes,
            max_census_entries,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TensorCeilings {
    max_header_bytes: NonZeroU64,
    max_tensors: NonZeroUsize,
    max_tensor_name_bytes: NonZeroUsize,
    max_rank: NonZeroUsize,
    max_dimension: NonZeroUsize,
    max_aggregate_elements: NonZeroU64,
    max_metadata_bytes: NonZeroU64,
}

impl TensorCeilings {
    fn new(
        max_header_bytes: NonZeroU64,
        max_tensors: NonZeroUsize,
        max_tensor_name_bytes: NonZeroUsize,
        max_rank: NonZeroUsize,
        max_dimension: NonZeroUsize,
        max_aggregate_elements: NonZeroU64,
        max_metadata_bytes: NonZeroU64,
    ) -> Self {
        Self {
            max_header_bytes,
            max_tensors,
            max_tensor_name_bytes,
            max_rank,
            max_dimension,
            max_aggregate_elements,
            max_metadata_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BertCeilings {
    max_vocab_size: NonZeroUsize,
    max_hidden_size: NonZeroUsize,
    max_hidden_layers: NonZeroUsize,
    max_attention_heads: NonZeroUsize,
    max_intermediate_size: NonZeroUsize,
    max_position_embeddings: NonZeroUsize,
    max_type_vocab_size: NonZeroUsize,
}

impl BertCeilings {
    #[cfg(test)]
    fn all(value: NonZeroUsize) -> Self {
        Self {
            max_vocab_size: value,
            max_hidden_size: value,
            max_hidden_layers: value,
            max_attention_heads: value,
            max_intermediate_size: value,
            max_position_embeddings: value,
            max_type_vocab_size: value,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ParseCeilings {
    max_config_bytes: NonZeroU64,
    max_tokenizer_file_bytes: NonZeroU64,
    max_config_parse_work_bytes: NonZeroU64,
    max_tokenizer_parse_work_bytes: NonZeroU64,
    max_safetensors_parse_work_bytes: NonZeroU64,
}

impl ParseCeilings {
    #[cfg(test)]
    fn all(value: NonZeroU64) -> Self {
        Self {
            max_config_bytes: value,
            max_tokenizer_file_bytes: value,
            max_config_parse_work_bytes: value,
            max_tokenizer_parse_work_bytes: value,
            max_safetensors_parse_work_bytes: value,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ResourceCeilings {
    max_model_retained_bytes: NonZeroU64,
    max_tokenizer_retained_bytes: NonZeroU64,
    max_descriptor_control_bytes: NonZeroU64,
    max_model_load_work_bytes: NonZeroU64,
}

impl ResourceCeilings {
    #[cfg(test)]
    fn all(value: NonZeroU64) -> Self {
        Self {
            max_model_retained_bytes: value,
            max_tokenizer_retained_bytes: value,
            max_descriptor_control_bytes: value,
            max_model_load_work_bytes: value,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PreparationCeilings {
    inventory: InventoryCeilings,
    tensors: TensorCeilings,
    bert: BertCeilings,
    parsing: ParseCeilings,
    resources: ResourceCeilings,
}

impl PreparationCeilings {
    fn try_new(
        inventory: InventoryCeilings,
        tensors: TensorCeilings,
        bert: BertCeilings,
        parsing: ParseCeilings,
        resources: ResourceCeilings,
    ) -> LimitResult<Self> {
        let platform_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
        Self::try_new_with_platform_max(inventory, tensors, bert, parsing, resources, platform_max)
    }

    fn try_new_with_platform_max(
        inventory: InventoryCeilings,
        tensors: TensorCeilings,
        bert: BertCeilings,
        parsing: ParseCeilings,
        resources: ResourceCeilings,
        platform_max: u64,
    ) -> LimitResult<Self> {
        let ceilings = Self {
            inventory,
            tensors,
            bert,
            parsing,
            resources,
        };
        ceilings.validate_relations()?;
        ceilings.validate_platform_caps_with_max(platform_max)?;
        Ok(ceilings)
    }

    fn validate(&self, axis: LimitAxis, actual: u64) -> LimitResult<()> {
        let limit = self.limit(axis)?;
        if actual > limit {
            return Err(PreparationLimitError::Exceeded {
                axis,
                actual,
                limit,
            });
        }
        Ok(())
    }

    fn limit(&self, axis: LimitAxis) -> LimitResult<u64> {
        let usize_limit = |value: NonZeroUsize| {
            u64::try_from(value.get()).map_err(|_| PreparationLimitError::ArithmeticOverflow {
                expression: ChargeExpression::TensorCensus,
            })
        };
        match axis {
            LimitAxis::Files => usize_limit(self.inventory.max_files),
            LimitAxis::PathBytes => usize_limit(self.inventory.max_path_bytes),
            LimitAxis::TotalPathBytes => Ok(self.inventory.max_total_path_bytes.get()),
            LimitAxis::AuxFileBytes => Ok(self.inventory.max_aux_file_bytes.get()),
            LimitAxis::IndexFileBytes => Ok(self.inventory.max_index_file_bytes.get()),
            LimitAxis::WeightFileBytes => Ok(self.inventory.max_weight_file_bytes.get()),
            LimitAxis::SnapshotBytes => Ok(self.inventory.max_snapshot_bytes.get()),
            LimitAxis::CensusEntries => usize_limit(self.inventory.max_census_entries),
            LimitAxis::CensusMetadataBytes => Ok(self.inventory.max_census_metadata_bytes.get()),
            LimitAxis::HeaderBytes => Ok(self.tensors.max_header_bytes.get()),
            LimitAxis::Tensors => usize_limit(self.tensors.max_tensors),
            LimitAxis::TensorNameBytes => usize_limit(self.tensors.max_tensor_name_bytes),
            LimitAxis::Rank => usize_limit(self.tensors.max_rank),
            LimitAxis::Dimension => usize_limit(self.tensors.max_dimension),
            LimitAxis::AggregateElements => Ok(self.tensors.max_aggregate_elements.get()),
            LimitAxis::TensorMetadataBytes => Ok(self.tensors.max_metadata_bytes.get()),
            LimitAxis::VocabSize => usize_limit(self.bert.max_vocab_size),
            LimitAxis::HiddenSize => usize_limit(self.bert.max_hidden_size),
            LimitAxis::HiddenLayers => usize_limit(self.bert.max_hidden_layers),
            LimitAxis::AttentionHeads => usize_limit(self.bert.max_attention_heads),
            LimitAxis::IntermediateSize => usize_limit(self.bert.max_intermediate_size),
            LimitAxis::PositionEmbeddings => usize_limit(self.bert.max_position_embeddings),
            LimitAxis::TypeVocabSize => usize_limit(self.bert.max_type_vocab_size),
            LimitAxis::ConfigBytes => Ok(self.parsing.max_config_bytes.get()),
            LimitAxis::TokenizerFileBytes => Ok(self.parsing.max_tokenizer_file_bytes.get()),
            LimitAxis::ConfigParseWorkBytes => Ok(self.parsing.max_config_parse_work_bytes.get()),
            LimitAxis::TokenizerParseWorkBytes => {
                Ok(self.parsing.max_tokenizer_parse_work_bytes.get())
            }
            LimitAxis::SafetensorsParseWorkBytes => {
                Ok(self.parsing.max_safetensors_parse_work_bytes.get())
            }
            LimitAxis::ModelRetainedBytes => Ok(self.resources.max_model_retained_bytes.get()),
            LimitAxis::TokenizerRetainedBytes => {
                Ok(self.resources.max_tokenizer_retained_bytes.get())
            }
            LimitAxis::DescriptorControlBytes => {
                Ok(self.resources.max_descriptor_control_bytes.get())
            }
            LimitAxis::ModelLoadWorkBytes => Ok(self.resources.max_model_load_work_bytes.get()),
        }
    }

    fn validate_relations(&self) -> LimitResult<()> {
        self.relation(
            LimitAxis::PathBytes,
            self.limit(LimitAxis::PathBytes)?,
            LimitAxis::TotalPathBytes,
            self.limit(LimitAxis::TotalPathBytes)?,
        )?;
        for (axis, value) in [
            (
                LimitAxis::AuxFileBytes,
                self.inventory.max_aux_file_bytes.get(),
            ),
            (
                LimitAxis::IndexFileBytes,
                self.inventory.max_index_file_bytes.get(),
            ),
            (
                LimitAxis::WeightFileBytes,
                self.inventory.max_weight_file_bytes.get(),
            ),
        ] {
            self.relation(
                axis,
                value,
                LimitAxis::SnapshotBytes,
                self.inventory.max_snapshot_bytes.get(),
            )?;
        }
        self.relation(
            LimitAxis::ConfigBytes,
            self.parsing.max_config_bytes.get(),
            LimitAxis::AuxFileBytes,
            self.inventory.max_aux_file_bytes.get(),
        )?;
        self.relation(
            LimitAxis::TokenizerFileBytes,
            self.parsing.max_tokenizer_file_bytes.get(),
            LimitAxis::AuxFileBytes,
            self.inventory.max_aux_file_bytes.get(),
        )?;
        let framed_header = 8_u64
            .checked_add(self.tensors.max_header_bytes.get())
            .ok_or(PreparationLimitError::ArithmeticOverflow {
                expression: ChargeExpression::HeaderFrame,
            })?;
        self.relation(
            LimitAxis::HeaderBytes,
            framed_header,
            LimitAxis::WeightFileBytes,
            self.inventory.max_weight_file_bytes.get(),
        )
    }

    fn relation(
        &self,
        lower: LimitAxis,
        lower_value: u64,
        upper: LimitAxis,
        upper_value: u64,
    ) -> LimitResult<()> {
        if lower_value > upper_value {
            return Err(PreparationLimitError::InvalidRelation { lower, upper });
        }
        Ok(())
    }

    fn validate_platform_caps_with_max(&self, platform_max: u64) -> LimitResult<()> {
        for (axis, value) in [
            (
                LimitAxis::TotalPathBytes,
                self.inventory.max_total_path_bytes.get(),
            ),
            (
                LimitAxis::AuxFileBytes,
                self.inventory.max_aux_file_bytes.get(),
            ),
            (
                LimitAxis::IndexFileBytes,
                self.inventory.max_index_file_bytes.get(),
            ),
            (
                LimitAxis::WeightFileBytes,
                self.inventory.max_weight_file_bytes.get(),
            ),
            (
                LimitAxis::SnapshotBytes,
                self.inventory.max_snapshot_bytes.get(),
            ),
            (
                LimitAxis::CensusMetadataBytes,
                self.inventory.max_census_metadata_bytes.get(),
            ),
            (LimitAxis::HeaderBytes, self.tensors.max_header_bytes.get()),
            (
                LimitAxis::AggregateElements,
                self.tensors.max_aggregate_elements.get(),
            ),
            (
                LimitAxis::TensorMetadataBytes,
                self.tensors.max_metadata_bytes.get(),
            ),
            (LimitAxis::ConfigBytes, self.parsing.max_config_bytes.get()),
            (
                LimitAxis::TokenizerFileBytes,
                self.parsing.max_tokenizer_file_bytes.get(),
            ),
            (
                LimitAxis::ConfigParseWorkBytes,
                self.parsing.max_config_parse_work_bytes.get(),
            ),
            (
                LimitAxis::TokenizerParseWorkBytes,
                self.parsing.max_tokenizer_parse_work_bytes.get(),
            ),
            (
                LimitAxis::SafetensorsParseWorkBytes,
                self.parsing.max_safetensors_parse_work_bytes.get(),
            ),
            (
                LimitAxis::ModelRetainedBytes,
                self.resources.max_model_retained_bytes.get(),
            ),
            (
                LimitAxis::TokenizerRetainedBytes,
                self.resources.max_tokenizer_retained_bytes.get(),
            ),
            (
                LimitAxis::DescriptorControlBytes,
                self.resources.max_descriptor_control_bytes.get(),
            ),
            (
                LimitAxis::ModelLoadWorkBytes,
                self.resources.max_model_load_work_bytes.get(),
            ),
        ] {
            platform_usize_with_max(axis, value, platform_max)?;
        }
        Ok(())
    }
}

fn platform_usize(axis: LimitAxis, value: u64) -> LimitResult<usize> {
    let platform_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
    platform_usize_with_max(axis, value, platform_max)
}

fn platform_usize_with_max(axis: LimitAxis, value: u64, platform_max: u64) -> LimitResult<usize> {
    if value > platform_max {
        return Err(PreparationLimitError::PlatformUnrepresentable { axis, value });
    }
    usize::try_from(value)
        .map_err(|_| PreparationLimitError::PlatformUnrepresentable { axis, value })
}

#[derive(Clone, Copy, Debug)]
struct CandidateBertFacts {
    vocab_size: u64,
    hidden_size: u64,
    hidden_layers: u64,
    attention_heads: u64,
    intermediate_size: u64,
    position_embeddings: u64,
    type_vocab_size: u64,
    layer_norm_eps: f64,
    has_pooler: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CandidateBertGeometry {
    required_tensors: usize,
    required_elements: u64,
    fused_qkv_elements: u64,
}

impl CandidateBertGeometry {
    fn try_new(facts: CandidateBertFacts, ceilings: &PreparationCeilings) -> LimitResult<Self> {
        let fields = [
            (LimitAxis::VocabSize, facts.vocab_size),
            (LimitAxis::HiddenSize, facts.hidden_size),
            (LimitAxis::HiddenLayers, facts.hidden_layers),
            (LimitAxis::AttentionHeads, facts.attention_heads),
            (LimitAxis::IntermediateSize, facts.intermediate_size),
            (LimitAxis::PositionEmbeddings, facts.position_embeddings),
            (LimitAxis::TypeVocabSize, facts.type_vocab_size),
        ];
        for (axis, value) in fields {
            if value == 0 {
                return Err(PreparationLimitError::InvalidGeometry {
                    fault: BertGeometryFault::Zero(axis),
                });
            }
            ceilings.validate(axis, value)?;
        }
        if !facts.layer_norm_eps.is_finite() || facts.layer_norm_eps <= 0.0 {
            return Err(PreparationLimitError::InvalidGeometry {
                fault: BertGeometryFault::InvalidLayerNormEpsilon,
            });
        }

        let vocab_size = platform_usize(LimitAxis::VocabSize, facts.vocab_size)?;
        let hidden_size = platform_usize(LimitAxis::HiddenSize, facts.hidden_size)?;
        let hidden_layers = platform_usize(LimitAxis::HiddenLayers, facts.hidden_layers)?;
        let attention_heads = platform_usize(LimitAxis::AttentionHeads, facts.attention_heads)?;
        let intermediate_size =
            platform_usize(LimitAxis::IntermediateSize, facts.intermediate_size)?;
        let position_embeddings =
            platform_usize(LimitAxis::PositionEmbeddings, facts.position_embeddings)?;
        let type_vocab_size = platform_usize(LimitAxis::TypeVocabSize, facts.type_vocab_size)?;
        if hidden_size % attention_heads != 0 {
            return Err(PreparationLimitError::InvalidGeometry {
                fault: BertGeometryFault::AttentionHeadRemainder,
            });
        }

        let required_tensors = required_tensor_count(hidden_layers, facts.has_pooler)?;

        let v = u64::try_from(vocab_size)
            .map_err(|_| overflow(ChargeExpression::RequiredTensorElements))?;
        let h = u64::try_from(hidden_size)
            .map_err(|_| overflow(ChargeExpression::RequiredTensorElements))?;
        let l = u64::try_from(hidden_layers)
            .map_err(|_| overflow(ChargeExpression::RequiredTensorElements))?;
        let i = u64::try_from(intermediate_size)
            .map_err(|_| overflow(ChargeExpression::RequiredTensorElements))?;
        let p = u64::try_from(position_embeddings)
            .map_err(|_| overflow(ChargeExpression::RequiredTensorElements))?;
        let t = u64::try_from(type_vocab_size)
            .map_err(|_| overflow(ChargeExpression::RequiredTensorElements))?;
        let required_elements = required_elements(v, h, l, i, p, t, facts.has_pooler)?;
        let fused_qkv_elements = fused_qkv_elements(h, l)?;
        ceilings.validate(
            LimitAxis::Tensors,
            u64::try_from(required_tensors)
                .map_err(|_| overflow(ChargeExpression::RequiredTensorCount))?,
        )?;
        ceilings.validate(LimitAxis::AggregateElements, required_elements)?;

        Ok(Self {
            required_tensors,
            required_elements,
            fused_qkv_elements,
        })
    }

    fn required_tensors(self) -> usize {
        self.required_tensors
    }

    fn required_elements(self) -> u64 {
        self.required_elements
    }

    fn fused_qkv_elements(self) -> u64 {
        self.fused_qkv_elements
    }
}

fn overflow(expression: ChargeExpression) -> PreparationLimitError {
    PreparationLimitError::ArithmeticOverflow { expression }
}

fn required_tensor_count(hidden_layers: usize, has_pooler: bool) -> LimitResult<usize> {
    hidden_layers
        .checked_mul(16)
        .and_then(|value| value.checked_add(5))
        .and_then(|value| value.checked_add(usize::from(has_pooler) * 2))
        .ok_or_else(|| overflow(ChargeExpression::RequiredTensorCount))
}

fn required_elements(
    v: u64,
    h: u64,
    l: u64,
    i: u64,
    p: u64,
    t: u64,
    has_pooler: bool,
) -> LimitResult<u64> {
    let expression = ChargeExpression::RequiredTensorElements;
    let h2 = h.checked_mul(h).ok_or_else(|| overflow(expression))?;
    let embedding_terms = v
        .checked_add(p)
        .and_then(|value| value.checked_add(t))
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| overflow(expression))?;
    let embeddings = h
        .checked_mul(embedding_terms)
        .ok_or_else(|| overflow(expression))?;
    let per_layer = h2
        .checked_mul(4)
        .and_then(|value| {
            h.checked_mul(i)
                .and_then(|hi| hi.checked_mul(2))
                .and_then(|term| value.checked_add(term))
        })
        .and_then(|value| h.checked_mul(9).and_then(|term| value.checked_add(term)))
        .and_then(|value| value.checked_add(i))
        .ok_or_else(|| overflow(expression))?;
    let layers = l
        .checked_mul(per_layer)
        .ok_or_else(|| overflow(expression))?;
    let pooler = if has_pooler {
        h2.checked_add(h).ok_or_else(|| overflow(expression))?
    } else {
        0
    };
    embeddings
        .checked_add(layers)
        .and_then(|value| value.checked_add(pooler))
        .ok_or_else(|| overflow(expression))
}

fn fused_qkv_elements(h: u64, l: u64) -> LimitResult<u64> {
    let expression = ChargeExpression::FusedQkvElements;
    let per_layer = h
        .checked_mul(h)
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| h.checked_mul(3).and_then(|term| value.checked_add(term)))
        .ok_or_else(|| overflow(expression))?;
    l.checked_mul(per_layer).ok_or_else(|| overflow(expression))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TensorDtype {
    F32,
    F16,
    Bf16,
}

impl TensorDtype {
    fn width(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    pub(super) fn digest_tag(self) -> u8 {
        match self {
            Self::F32 => 1,
            Self::F16 => 2,
            Self::Bf16 => 3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TensorFact<'a> {
    name_bytes: u64,
    metadata_bytes: u64,
    dimensions: &'a [u64],
    dtype: TensorDtype,
}

impl<'a> TensorFact<'a> {
    pub(super) fn new(
        name_bytes: u64,
        metadata_bytes: u64,
        dimensions: &'a [u64],
        dtype: TensorDtype,
    ) -> Self {
        Self {
            name_bytes,
            metadata_bytes,
            dimensions,
            dtype,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TensorCensus {
    tensor_count: usize,
    total_name_bytes: u64,
    total_metadata_bytes: u64,
    total_elements: u64,
    declared_tensor_bytes: u64,
}

impl TensorCensus {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn push(
        &mut self,
        fact: TensorFact<'_>,
        ceilings: &PreparationCeilings,
    ) -> LimitResult<()> {
        ceilings.validate(LimitAxis::TensorNameBytes, fact.name_bytes)?;
        ceilings.validate(
            LimitAxis::Rank,
            u64::try_from(fact.dimensions.len())
                .map_err(|_| overflow(ChargeExpression::TensorCensus))?,
        )?;

        for &dimension in fact.dimensions {
            platform_usize(LimitAxis::Dimension, dimension)?;
            ceilings.validate(LimitAxis::Dimension, dimension)?;
        }
        let tensor_elements = checked_tensor_elements(fact.dimensions)?;
        let tensor_bytes = checked_tensor_bytes(tensor_elements, fact.dtype)?;
        let tensor_count = self
            .tensor_count
            .checked_add(1)
            .ok_or_else(|| overflow(ChargeExpression::TensorCensus))?;
        ceilings.validate(
            LimitAxis::Tensors,
            u64::try_from(tensor_count).map_err(|_| overflow(ChargeExpression::TensorCensus))?,
        )?;
        let total_name_bytes = self
            .total_name_bytes
            .checked_add(fact.name_bytes)
            .ok_or_else(|| overflow(ChargeExpression::TensorCensus))?;
        let total_metadata_bytes = self
            .total_metadata_bytes
            .checked_add(fact.metadata_bytes)
            .ok_or_else(|| overflow(ChargeExpression::TensorCensus))?;
        let bounded_metadata = total_name_bytes
            .checked_add(total_metadata_bytes)
            .ok_or_else(|| overflow(ChargeExpression::TensorCensus))?;
        ceilings.validate(LimitAxis::TensorMetadataBytes, bounded_metadata)?;
        let total_elements = self
            .total_elements
            .checked_add(tensor_elements)
            .ok_or_else(|| overflow(ChargeExpression::TensorCensus))?;
        ceilings.validate(LimitAxis::AggregateElements, total_elements)?;
        // This counts declared tensor-payload bytes (shape × source-dtype
        // width) only. It does not validate offsets/ranges, and the later
        // inference-owned estimator must separately bound decoded/mapped
        // allocations before any allocation or model publication.
        let declared_tensor_bytes = self
            .declared_tensor_bytes
            .checked_add(tensor_bytes)
            .ok_or_else(|| overflow(ChargeExpression::TensorBytes))?;
        // Every declared tensor byte has to live somewhere inside the
        // prepared snapshot, so the running total is bounded by the same
        // aggregate ceiling that already bounds each individual weight
        // file's on-disk size (`validate_relations`'s
        // `WeightFileBytes <= SnapshotBytes` relation) — reusing it here
        // means many individually-legal tensors can no longer sum to an
        // unbounded payload without a fresh, hand-picked constant.
        ceilings.validate(LimitAxis::SnapshotBytes, declared_tensor_bytes)?;

        self.tensor_count = tensor_count;
        self.total_name_bytes = total_name_bytes;
        self.total_metadata_bytes = total_metadata_bytes;
        self.total_elements = total_elements;
        self.declared_tensor_bytes = declared_tensor_bytes;
        Ok(())
    }

    fn tensor_count(self) -> usize {
        self.tensor_count
    }

    fn total_name_bytes(self) -> u64 {
        self.total_name_bytes
    }

    fn total_metadata_bytes(self) -> u64 {
        self.total_metadata_bytes
    }

    fn total_elements(self) -> u64 {
        self.total_elements
    }

    fn declared_tensor_bytes(self) -> u64 {
        self.declared_tensor_bytes
    }
}

fn checked_tensor_elements(dimensions: &[u64]) -> LimitResult<u64> {
    dimensions.iter().try_fold(1_u64, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| overflow(ChargeExpression::TensorElements))
    })
}

fn checked_tensor_bytes(elements: u64, dtype: TensorDtype) -> LimitResult<u64> {
    elements
        .checked_mul(dtype.width())
        .ok_or_else(|| overflow(ChargeExpression::TensorBytes))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparationCharge {
    retained_bytes: u64,
    work_bytes: u64,
}

impl PreparationCharge {
    fn worst_case(ceilings: &PreparationCeilings) -> LimitResult<Self> {
        let max_report_bytes = u64::try_from(super::MAX_SUPPLEMENTARY_ATTESTATION_EVIDENCE_BYTES)
            .map_err(|_| overflow(ChargeExpression::AttestationReportBytes))?;
        let retained_bytes = [
            ceilings.inventory.max_snapshot_bytes.get(),
            ceilings.resources.max_model_retained_bytes.get(),
            ceilings.resources.max_tokenizer_retained_bytes.get(),
            ceilings.inventory.max_total_path_bytes.get(),
            ceilings.inventory.max_census_metadata_bytes.get(),
            ceilings.tensors.max_metadata_bytes.get(),
            ceilings.resources.max_descriptor_control_bytes.get(),
            max_report_bytes,
        ];
        let retained_bytes = checked_sum(&retained_bytes, ChargeExpression::RetainedCharge)?;
        // The first report becomes the prepared object's retained report. A
        // second fresh report must coexist transiently while the two complete
        // attestations are compared before publication.
        let work_bytes = [
            ATTESTATION_CHUNK_BYTES,
            max_report_bytes,
            ceilings.parsing.max_config_parse_work_bytes.get(),
            ceilings.parsing.max_tokenizer_parse_work_bytes.get(),
            ceilings.parsing.max_safetensors_parse_work_bytes.get(),
            ceilings.resources.max_model_load_work_bytes.get(),
        ];
        let work_bytes = checked_sum(&work_bytes, ChargeExpression::WorkCharge)?;
        Ok(Self {
            retained_bytes,
            work_bytes,
        })
    }

    fn retained_bytes(self) -> u64 {
        self.retained_bytes
    }

    fn work_bytes(self) -> u64 {
        self.work_bytes
    }
}

fn checked_sum(values: &[u64], expression: ChargeExpression) -> LimitResult<u64> {
    values.iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(*value).ok_or_else(|| overflow(expression))
    })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn test_tensor_inventory_ceilings(
    max_tensors: usize,
    max_name_bytes: usize,
    max_rank: usize,
    max_dimension: usize,
    max_elements: u64,
    max_metadata: u64,
) -> PreparationCeilings {
    let nz = |value| NonZeroUsize::new(value).expect("test ceiling must be non-zero");
    let nz64 = |value| NonZeroU64::new(value).expect("test ceiling must be non-zero");
    PreparationCeilings::try_new(
        InventoryCeilings::new(
            nz(1),
            nz(1),
            nz64(1),
            nz64(9),
            nz64(9),
            nz64(9),
            // `TensorCensus::push` bounds the running declared-payload total
            // against this axis (see `push`'s `SnapshotBytes` check). This
            // helper is shared by tests that deliberately push tensors sized
            // off `max_elements`/`max_dimension` up to `u64::MAX`/
            // `usize::MAX` to exercise arithmetic overflow, so a fixed
            // placeholder here (matching the `9` the sibling file-size axes
            // use) would trip this axis before that overflow ever occurs.
            // Pinning it to the same platform ceiling `PreparationCeilings`
            // itself enforces elsewhere makes the check a no-op for every
            // caller that doesn't care about it, while a test that wants to
            // probe it directly still builds its own `PreparationCeilings`.
            nz64(u64::try_from(usize::MAX).unwrap_or(u64::MAX)),
            nz64(1),
            nz(1),
        ),
        TensorCeilings::new(
            nz64(1),
            nz(max_tensors),
            nz(max_name_bytes),
            nz(max_rank),
            nz(max_dimension),
            nz64(max_elements),
            nz64(max_metadata),
        ),
        BertCeilings::all(nz(1)),
        ParseCeilings::all(nz64(1)),
        ResourceCeilings::all(nz64(1)),
    )
    .expect("test tensor inventory ceilings must be internally valid")
}

#[cfg(test)]
mod tests;
