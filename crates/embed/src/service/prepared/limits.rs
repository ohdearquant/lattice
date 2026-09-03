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
        let max_report_bytes = u64::try_from(super::MAX_ATTESTATION_REPORT_BYTES)
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
            nz64(9),
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
mod tests {
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
                nz64(32),
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
            platform_usize_with_max(LimitAxis::SnapshotBytes, artificial_max, artificial_max)
                .unwrap(),
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
            u64::try_from(super::super::MAX_ATTESTATION_REPORT_BYTES).unwrap()
        );
        let work_without_second_report = ATTESTATION_CHUNK_BYTES + 1 + 2 + 4 + 8;
        assert_eq!(
            charge.work_bytes() - work_without_second_report,
            u64::try_from(super::super::MAX_ATTESTATION_REPORT_BYTES).unwrap()
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
}
