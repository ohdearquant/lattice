//! Dormant, bounded SafeTensors inventory facts for sealed native preparation.
//!
//! This module parses an already-framed header slice. It performs no filesystem
//! access, mapping, payload reads, model loading, or serving work and has no live
//! production caller. Pass one validates grammar and computes a zero-allocation
//! census. Only then does pass two fallibly reserve bounded arenas, materialize
//! decoded names and metadata keys, reject duplicates, and validate exact global
//! payload coverage.
//!
//! A successful value is an inspected header inventory, not a trusted checkpoint.
//! It does not bind bytes to an opened handle, close TOCTOU, authorize a mapping,
//! or prove that an F32 tensor is actually eligible for zero-copy loading.
//! Retained-inventory and parse-work byte facts are disjoint logical accounting
//! terms. They exclude `Vec` headers, capacity slack, allocator overhead, and fixed
//! stack state; a future live resource domain must reserve those conservatively.

use std::num::NonZeroU64;
use std::ops::Range;

use super::prepared_safetensors::PreparedSafetensorsHeaderPlan;

const CANONICAL_INVENTORY_U32_FIELD_MAX: u64 = u32::MAX as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InventoryAxis {
    Tensors,
    TensorNameBytes,
    Rank,
    Dimension,
    AggregateElements,
    MetadataBytes,
    SafetensorsParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InventoryExpression {
    TensorCount,
    DecodedNameBytes,
    DimensionCount,
    TensorElements,
    TensorSourceBytes,
    AggregateElements,
    MetadataBytes,
    SafetensorsParseWorkBytes,
    AbsoluteDataOffset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TensorMember {
    Dtype,
    Shape,
    DataOffsets,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AllocationArena {
    TensorRecords,
    TensorNameBytes,
    Dimensions,
    MetadataKeyRecords,
    MetadataKeyBytes,
}

impl AllocationArena {
    #[cfg(test)]
    const ALL: [Self; 5] = [
        Self::TensorRecords,
        Self::TensorNameBytes,
        Self::Dimensions,
        Self::MetadataKeyRecords,
        Self::MetadataKeyBytes,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeaderSyntaxFault {
    ExpectedObjectStart,
    ExpectedString,
    ExpectedColon,
    ExpectedCommaOrEnd,
    TrailingComma,
    TrailingNonSpace,
    InvalidStringEscape,
    InvalidUnicodeEscape,
    InvalidUnsignedInteger,
    UnexpectedEof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreparedSafetensorsInventoryError {
    HeaderLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidUtf8 {
        valid_up_to: usize,
    },
    MalformedHeader {
        at: usize,
        fault: HeaderSyntaxFault,
    },
    Exceeded {
        axis: InventoryAxis,
        actual: u64,
        limit: u64,
    },
    PlatformUnrepresentable {
        axis: InventoryAxis,
        value: u64,
    },
    ArithmeticOverflow {
        expression: InventoryExpression,
    },
    MissingInventory,
    DuplicateMetadataMember,
    InvalidMetadata,
    DuplicateMetadataKey,
    MissingTensorMember {
        entry: u64,
        member: TensorMember,
    },
    DuplicateTensorMember {
        entry: u64,
        member: TensorMember,
    },
    UnknownTensorMember {
        entry: u64,
    },
    InvalidTensorMember {
        entry: u64,
        member: TensorMember,
    },
    UnsupportedDtype {
        entry: u64,
    },
    InvalidDataOffsetsArity {
        entry: u64,
        actual: u64,
    },
    ReversedDataRange {
        entry: u64,
        start: u64,
        end: u64,
    },
    DataRangePastEnd {
        entry: u64,
        end: u64,
        data_len: u64,
    },
    TensorByteLengthMismatch {
        entry: u64,
        expected: u64,
        actual: u64,
    },
    DuplicateTensorName,
    NonContiguousDataRange {
        expected_start: u64,
        actual_start: u64,
    },
    TrailingPayload {
        covered: u64,
        data_len: u64,
    },
    AllocationFailed {
        arena: AllocationArena,
        requested_bytes: usize,
    },
    SecondPassCensusMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreparedSafetensorsInventoryLimits {
    max_tensors: u64,
    max_tensor_name_bytes: u64,
    max_rank: u64,
    max_dimension: u64,
    max_aggregate_elements: u64,
    max_metadata_bytes: u64,
    max_parse_work_bytes: u64,
}

impl PreparedSafetensorsInventoryLimits {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        max_tensors: NonZeroU64,
        max_tensor_name_bytes: NonZeroU64,
        max_rank: NonZeroU64,
        max_dimension: NonZeroU64,
        max_aggregate_elements: NonZeroU64,
        max_metadata_bytes: NonZeroU64,
        max_parse_work_bytes: NonZeroU64,
    ) -> Result<Self, PreparedSafetensorsInventoryError> {
        let usize_max = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
        let isize_max = u64::try_from(isize::MAX).unwrap_or(u64::MAX);
        Self::try_new_with_platform_max(
            max_tensors,
            max_tensor_name_bytes,
            max_rank,
            max_dimension,
            max_aggregate_elements,
            max_metadata_bytes,
            max_parse_work_bytes,
            usize_max,
            usize_max.min(isize_max),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_new_with_platform_max(
        max_tensors: NonZeroU64,
        max_tensor_name_bytes: NonZeroU64,
        max_rank: NonZeroU64,
        max_dimension: NonZeroU64,
        max_aggregate_elements: NonZeroU64,
        max_metadata_bytes: NonZeroU64,
        max_parse_work_bytes: NonZeroU64,
        platform_usize_max: u64,
        platform_span_max: u64,
    ) -> Result<Self, PreparedSafetensorsInventoryError> {
        let values = [
            (InventoryAxis::Tensors, max_tensors.get()),
            (InventoryAxis::TensorNameBytes, max_tensor_name_bytes.get()),
            (InventoryAxis::Rank, max_rank.get()),
            (InventoryAxis::Dimension, max_dimension.get()),
        ];
        for (axis, value) in values {
            validate_platform(axis, value, platform_usize_max)?;
        }
        // Canonical downstream inventory framing uses u32 lengths for decoded
        // tensor names and ranks. Refuse an unframeable policy at construction.
        validate_limit(
            InventoryAxis::TensorNameBytes,
            max_tensor_name_bytes.get(),
            CANONICAL_INVENTORY_U32_FIELD_MAX,
        )?;
        validate_limit(
            InventoryAxis::Rank,
            max_rank.get(),
            CANONICAL_INVENTORY_U32_FIELD_MAX,
        )?;
        validate_platform(
            InventoryAxis::MetadataBytes,
            max_metadata_bytes.get(),
            platform_span_max,
        )?;

        Ok(Self {
            max_tensors: max_tensors.get(),
            max_tensor_name_bytes: max_tensor_name_bytes.get(),
            max_rank: max_rank.get(),
            max_dimension: max_dimension.get(),
            max_aggregate_elements: max_aggregate_elements.get(),
            max_metadata_bytes: max_metadata_bytes.get(),
            max_parse_work_bytes: max_parse_work_bytes.get(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreparedSafetensorsDtype {
    F32,
    F16,
    Bf16,
}

impl PreparedSafetensorsDtype {
    fn width(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum F32FileOffsetAlignment {
    FourByteAligned,
    Unaligned,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DecodedSpan {
    start: usize,
    len: usize,
}

impl DecodedSpan {
    fn end(self) -> Option<usize> {
        self.start.checked_add(self.len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedSafetensorsTensorRecord {
    name: DecodedSpan,
    dimensions: DecodedSpan,
    dtype: PreparedSafetensorsDtype,
    data_start: u64,
    data_end: u64,
    elements: u64,
    source_bytes: u64,
    f32_file_offset_alignment: Option<F32FileOffsetAlignment>,
}

#[derive(Debug)]
pub(crate) struct PreparedSafetensorsInventoryFacts {
    records: Vec<PreparedSafetensorsTensorRecord>,
    name_bytes: Vec<u8>,
    dimensions: Vec<u64>,
    aggregate_elements: u64,
    source_payload_bytes: u64,
    logical_retained_inventory_metadata_bytes: usize,
    logical_safetensors_parse_work_bytes: usize,
    decoded_metadata_value_bytes: usize,
}

impl PartialEq for PreparedSafetensorsInventoryFacts {
    fn eq(&self, other: &Self) -> bool {
        if self.aggregate_elements != other.aggregate_elements
            || self.source_payload_bytes != other.source_payload_bytes
            || self.logical_retained_inventory_metadata_bytes
                != other.logical_retained_inventory_metadata_bytes
            || self.logical_safetensors_parse_work_bytes
                != other.logical_safetensors_parse_work_bytes
            || self.decoded_metadata_value_bytes != other.decoded_metadata_value_bytes
        {
            return false;
        }
        let mut left = self.tensors();
        let mut right = other.tensors();
        loop {
            match (left.next(), right.next()) {
                (None, None) => return true,
                (Some(left), Some(right)) => {
                    if left.name_bytes() != right.name_bytes()
                        || left.dimensions() != right.dimensions()
                        || left.dtype() != right.dtype()
                        || left.data_range() != right.data_range()
                        || left.elements() != right.elements()
                        || left.source_bytes() != right.source_bytes()
                        || left.f32_file_offset_alignment() != right.f32_file_offset_alignment()
                    {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
}

impl Eq for PreparedSafetensorsInventoryFacts {}

impl PreparedSafetensorsInventoryFacts {
    pub(crate) fn tensors(&self) -> impl Iterator<Item = PreparedSafetensorsTensorView<'_>> {
        self.records
            .iter()
            .map(|record| PreparedSafetensorsTensorView {
                record,
                name_bytes: &self.name_bytes,
                dimensions: &self.dimensions,
            })
    }

    pub(crate) fn tensor_count(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn aggregate_elements(&self) -> u64 {
        self.aggregate_elements
    }

    pub(crate) fn source_payload_bytes(&self) -> u64 {
        self.source_payload_bytes
    }

    pub(crate) fn logical_retained_inventory_metadata_bytes(&self) -> usize {
        self.logical_retained_inventory_metadata_bytes
    }

    pub(crate) fn logical_safetensors_parse_work_bytes(&self) -> usize {
        self.logical_safetensors_parse_work_bytes
    }

    pub(crate) fn decoded_metadata_value_bytes(&self) -> usize {
        self.decoded_metadata_value_bytes
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedSafetensorsTensorView<'a> {
    record: &'a PreparedSafetensorsTensorRecord,
    name_bytes: &'a [u8],
    dimensions: &'a [u64],
}

impl PreparedSafetensorsTensorView<'_> {
    pub(crate) fn name_bytes(&self) -> &[u8] {
        span_bytes(self.name_bytes, self.record.name)
    }

    pub(crate) fn dimensions(&self) -> &[u64] {
        span_dimensions(self.dimensions, self.record.dimensions)
    }

    pub(crate) fn dtype(&self) -> PreparedSafetensorsDtype {
        self.record.dtype
    }

    pub(crate) fn data_range(&self) -> Range<u64> {
        self.record.data_start..self.record.data_end
    }

    pub(crate) fn elements(&self) -> u64 {
        self.record.elements
    }

    pub(crate) fn source_bytes(&self) -> u64 {
        self.record.source_bytes
    }

    pub(crate) fn f32_file_offset_alignment(&self) -> Option<F32FileOffsetAlignment> {
        self.record.f32_file_offset_alignment
    }
}

pub(crate) fn parse_prepared_safetensors_header_inventory(
    header: &[u8],
    plan: &PreparedSafetensorsHeaderPlan,
    limits: &PreparedSafetensorsInventoryLimits,
) -> Result<PreparedSafetensorsInventoryFacts, PreparedSafetensorsInventoryError> {
    parse_inventory_with_probe(header, plan, limits, &mut |_, _| Ok(()))
}

#[cfg(test)]
fn parse_prepared_safetensors_header_inventory_with_reserve_probe<F>(
    header: &[u8],
    plan: &PreparedSafetensorsHeaderPlan,
    limits: &PreparedSafetensorsInventoryLimits,
    probe: &mut F,
) -> Result<PreparedSafetensorsInventoryFacts, PreparedSafetensorsInventoryError>
where
    F: FnMut(AllocationArena, usize) -> Result<(), PreparedSafetensorsInventoryError>,
{
    parse_inventory_with_probe(header, plan, limits, probe)
}

fn parse_inventory_with_probe<F>(
    header: &[u8],
    plan: &PreparedSafetensorsHeaderPlan,
    limits: &PreparedSafetensorsInventoryLimits,
    probe: &mut F,
) -> Result<PreparedSafetensorsInventoryFacts, PreparedSafetensorsInventoryError>
where
    F: FnMut(AllocationArena, usize) -> Result<(), PreparedSafetensorsInventoryError>,
{
    if header.len() != plan.header_len() {
        return Err(PreparedSafetensorsInventoryError::HeaderLengthMismatch {
            expected: plan.header_len(),
            actual: header.len(),
        });
    }
    if let Err(error) = std::str::from_utf8(header) {
        return Err(PreparedSafetensorsInventoryError::InvalidUtf8 {
            valid_up_to: error.valid_up_to(),
        });
    }

    let mut census_collector = CensusCollector;
    let census = walk_header(header, plan, limits, &mut census_collector)?;
    if census.tensor_count == 0 {
        return Err(PreparedSafetensorsInventoryError::MissingInventory);
    }

    let mut builder = BuildCollector::new(census, probe)?;
    let second_census = walk_header(header, plan, limits, &mut builder)?;
    validate_second_pass(census, second_census, builder.matches(census))?;
    builder.finish(census, plan)
}

fn validate_second_pass(
    expected: InventoryCensus,
    actual: InventoryCensus,
    builder_matches: bool,
) -> Result<(), PreparedSafetensorsInventoryError> {
    if expected != actual || !builder_matches {
        return Err(PreparedSafetensorsInventoryError::SecondPassCensusMismatch);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InventoryCensus {
    tensor_count: u64,
    decoded_name_bytes: u64,
    dimension_count: u64,
    aggregate_elements: u64,
    source_payload_bytes: u64,
    metadata_key_count: u64,
    decoded_metadata_key_bytes: u64,
    decoded_metadata_value_bytes: u64,
    retained_metadata_bytes: u64,
    parse_work_bytes: u64,
}

trait InventoryCollector {
    fn tensor_name(
        &mut self,
        token: JsonStringToken<'_>,
    ) -> Result<DecodedSpan, PreparedSafetensorsInventoryError>;
    fn dimension(&mut self, value: u64) -> Result<(), PreparedSafetensorsInventoryError>;
    fn tensor_record(
        &mut self,
        record: PreparedSafetensorsTensorRecord,
    ) -> Result<(), PreparedSafetensorsInventoryError>;
    fn metadata_key(
        &mut self,
        token: JsonStringToken<'_>,
    ) -> Result<(), PreparedSafetensorsInventoryError>;
    fn dimension_start(&self) -> usize;
}

struct CensusCollector;

impl InventoryCollector for CensusCollector {
    fn tensor_name(
        &mut self,
        token: JsonStringToken<'_>,
    ) -> Result<DecodedSpan, PreparedSafetensorsInventoryError> {
        Ok(DecodedSpan {
            start: 0,
            len: to_usize(InventoryAxis::TensorNameBytes, token.decoded_len)?,
        })
    }

    fn dimension(&mut self, _value: u64) -> Result<(), PreparedSafetensorsInventoryError> {
        Ok(())
    }

    fn tensor_record(
        &mut self,
        _record: PreparedSafetensorsTensorRecord,
    ) -> Result<(), PreparedSafetensorsInventoryError> {
        Ok(())
    }

    fn metadata_key(
        &mut self,
        _token: JsonStringToken<'_>,
    ) -> Result<(), PreparedSafetensorsInventoryError> {
        Ok(())
    }

    fn dimension_start(&self) -> usize {
        0
    }
}

struct BuildCollector {
    records: Vec<PreparedSafetensorsTensorRecord>,
    name_bytes: Vec<u8>,
    dimensions: Vec<u64>,
    metadata_keys: Vec<DecodedSpan>,
    metadata_key_bytes: Vec<u8>,
}

impl BuildCollector {
    fn new<F>(
        census: InventoryCensus,
        probe: &mut F,
    ) -> Result<Self, PreparedSafetensorsInventoryError>
    where
        F: FnMut(AllocationArena, usize) -> Result<(), PreparedSafetensorsInventoryError>,
    {
        let mut records = Vec::new();
        reserve_exact(
            &mut records,
            to_usize(InventoryAxis::Tensors, census.tensor_count)?,
            AllocationArena::TensorRecords,
            probe,
        )?;
        let mut name_bytes = Vec::new();
        reserve_exact(
            &mut name_bytes,
            to_usize(InventoryAxis::MetadataBytes, census.decoded_name_bytes)?,
            AllocationArena::TensorNameBytes,
            probe,
        )?;
        let mut dimensions = Vec::new();
        reserve_exact(
            &mut dimensions,
            to_usize(InventoryAxis::MetadataBytes, census.dimension_count)?,
            AllocationArena::Dimensions,
            probe,
        )?;
        let mut metadata_keys = Vec::new();
        reserve_exact(
            &mut metadata_keys,
            to_usize(
                InventoryAxis::SafetensorsParseWorkBytes,
                census.metadata_key_count,
            )?,
            AllocationArena::MetadataKeyRecords,
            probe,
        )?;
        let mut metadata_key_bytes = Vec::new();
        reserve_exact(
            &mut metadata_key_bytes,
            to_usize(
                InventoryAxis::SafetensorsParseWorkBytes,
                census.decoded_metadata_key_bytes,
            )?,
            AllocationArena::MetadataKeyBytes,
            probe,
        )?;
        Ok(Self {
            records,
            name_bytes,
            dimensions,
            metadata_keys,
            metadata_key_bytes,
        })
    }

    fn matches(&self, census: InventoryCensus) -> bool {
        u64::try_from(self.records.len()) == Ok(census.tensor_count)
            && u64::try_from(self.name_bytes.len()) == Ok(census.decoded_name_bytes)
            && u64::try_from(self.dimensions.len()) == Ok(census.dimension_count)
            && u64::try_from(self.metadata_keys.len()) == Ok(census.metadata_key_count)
            && u64::try_from(self.metadata_key_bytes.len()) == Ok(census.decoded_metadata_key_bytes)
    }

    fn finish(
        mut self,
        census: InventoryCensus,
        plan: &PreparedSafetensorsHeaderPlan,
    ) -> Result<PreparedSafetensorsInventoryFacts, PreparedSafetensorsInventoryError> {
        let names = &self.name_bytes;
        self.records.sort_unstable_by(|left, right| {
            record_name(names, left).cmp(record_name(names, right))
        });
        if self
            .records
            .windows(2)
            .any(|pair| record_name(names, &pair[0]) == record_name(names, &pair[1]))
        {
            return Err(PreparedSafetensorsInventoryError::DuplicateTensorName);
        }

        let metadata_bytes = &self.metadata_key_bytes;
        self.metadata_keys.sort_unstable_by(|left, right| {
            span_bytes(metadata_bytes, *left).cmp(span_bytes(metadata_bytes, *right))
        });
        if self
            .metadata_keys
            .windows(2)
            .any(|pair| span_bytes(metadata_bytes, pair[0]) == span_bytes(metadata_bytes, pair[1]))
        {
            return Err(PreparedSafetensorsInventoryError::DuplicateMetadataKey);
        }

        self.records.sort_unstable_by(|left, right| {
            left.data_start
                .cmp(&right.data_start)
                .then(left.data_end.cmp(&right.data_end))
                .then_with(|| record_name(names, left).cmp(record_name(names, right)))
        });
        let mut frontier = 0_u64;
        for record in &self.records {
            if record.data_start != frontier {
                return Err(PreparedSafetensorsInventoryError::NonContiguousDataRange {
                    expected_start: frontier,
                    actual_start: record.data_start,
                });
            }
            frontier = record.data_end;
        }
        let data_len = u64::try_from(plan.data_len())
            .map_err(|_| overflow(InventoryExpression::TensorSourceBytes))?;
        if frontier != data_len {
            return Err(PreparedSafetensorsInventoryError::TrailingPayload {
                covered: frontier,
                data_len,
            });
        }

        self.records.sort_unstable_by(|left, right| {
            record_name(names, left).cmp(record_name(names, right))
        });
        Ok(PreparedSafetensorsInventoryFacts {
            records: self.records,
            name_bytes: self.name_bytes,
            dimensions: self.dimensions,
            aggregate_elements: census.aggregate_elements,
            source_payload_bytes: census.source_payload_bytes,
            logical_retained_inventory_metadata_bytes: to_usize(
                InventoryAxis::MetadataBytes,
                census.retained_metadata_bytes,
            )?,
            logical_safetensors_parse_work_bytes: to_usize(
                InventoryAxis::SafetensorsParseWorkBytes,
                census.parse_work_bytes,
            )?,
            decoded_metadata_value_bytes: to_usize(
                InventoryAxis::SafetensorsParseWorkBytes,
                census.decoded_metadata_value_bytes,
            )?,
        })
    }
}

impl InventoryCollector for BuildCollector {
    fn tensor_name(
        &mut self,
        token: JsonStringToken<'_>,
    ) -> Result<DecodedSpan, PreparedSafetensorsInventoryError> {
        append_token(
            &mut self.name_bytes,
            token,
            AllocationArena::TensorNameBytes,
        )
    }

    fn dimension(&mut self, value: u64) -> Result<(), PreparedSafetensorsInventoryError> {
        if self.dimensions.len() == self.dimensions.capacity() {
            return Err(PreparedSafetensorsInventoryError::SecondPassCensusMismatch);
        }
        self.dimensions.push(value);
        Ok(())
    }

    fn tensor_record(
        &mut self,
        record: PreparedSafetensorsTensorRecord,
    ) -> Result<(), PreparedSafetensorsInventoryError> {
        if self.records.len() == self.records.capacity() {
            return Err(PreparedSafetensorsInventoryError::SecondPassCensusMismatch);
        }
        self.records.push(record);
        Ok(())
    }

    fn metadata_key(
        &mut self,
        token: JsonStringToken<'_>,
    ) -> Result<(), PreparedSafetensorsInventoryError> {
        if self.metadata_keys.len() == self.metadata_keys.capacity() {
            return Err(PreparedSafetensorsInventoryError::SecondPassCensusMismatch);
        }
        let span = append_token(
            &mut self.metadata_key_bytes,
            token,
            AllocationArena::MetadataKeyBytes,
        )?;
        self.metadata_keys.push(span);
        Ok(())
    }

    fn dimension_start(&self) -> usize {
        self.dimensions.len()
    }
}

#[derive(Clone, Copy)]
struct ShapeFact {
    span: DecodedSpan,
    elements: u64,
}

fn walk_header<C: InventoryCollector>(
    header: &[u8],
    plan: &PreparedSafetensorsHeaderPlan,
    limits: &PreparedSafetensorsInventoryLimits,
    collector: &mut C,
) -> Result<InventoryCensus, PreparedSafetensorsInventoryError> {
    let mut scanner = HeaderScanner::new(header);
    scanner.expect_byte(b'{', HeaderSyntaxFault::ExpectedObjectStart)?;
    scanner.skip_ws();
    let mut census = InventoryCensus::default();
    let mut saw_metadata = false;

    if scanner.peek() == Some(b'}') {
        scanner.bump();
    } else {
        loop {
            let key = scanner.parse_string()?;
            scanner.skip_ws();
            scanner.expect_byte(b':', HeaderSyntaxFault::ExpectedColon)?;
            scanner.skip_ws();

            if key.matches(b"__metadata__") {
                if saw_metadata {
                    return Err(PreparedSafetensorsInventoryError::DuplicateMetadataMember);
                }
                saw_metadata = true;
                parse_metadata(&mut scanner, &mut census, collector)?;
            } else {
                census.tensor_count = census
                    .tensor_count
                    .checked_add(1)
                    .ok_or_else(|| overflow(InventoryExpression::TensorCount))?;
                validate_limit(
                    InventoryAxis::Tensors,
                    census.tensor_count,
                    limits.max_tensors,
                )?;
                validate_limit(
                    InventoryAxis::TensorNameBytes,
                    key.decoded_len,
                    limits.max_tensor_name_bytes,
                )?;
                census.decoded_name_bytes = census
                    .decoded_name_bytes
                    .checked_add(key.decoded_len)
                    .ok_or_else(|| overflow(InventoryExpression::DecodedNameBytes))?;
                let name = collector.tensor_name(key)?;
                let entry = census.tensor_count - 1;
                parse_tensor(
                    &mut scanner,
                    plan,
                    limits,
                    &mut census,
                    collector,
                    name,
                    entry,
                )?;
            }

            scanner.skip_ws();
            match scanner.peek() {
                Some(b',') => {
                    scanner.bump();
                    scanner.skip_ws();
                    if scanner.peek() == Some(b'}') {
                        return Err(scanner.error(HeaderSyntaxFault::TrailingComma));
                    }
                }
                Some(b'}') => {
                    scanner.bump();
                    break;
                }
                Some(_) => {
                    return Err(scanner.error(HeaderSyntaxFault::ExpectedCommaOrEnd));
                }
                None => return Err(scanner.error(HeaderSyntaxFault::UnexpectedEof)),
            }
        }
    }

    while scanner.peek() == Some(b' ') {
        scanner.bump();
    }
    if scanner.peek().is_some() {
        return Err(scanner.error(HeaderSyntaxFault::TrailingNonSpace));
    }

    finish_census(header.len(), limits, &mut census)?;
    Ok(census)
}

#[allow(clippy::too_many_arguments)]
fn parse_tensor<C: InventoryCollector>(
    scanner: &mut HeaderScanner<'_>,
    plan: &PreparedSafetensorsHeaderPlan,
    limits: &PreparedSafetensorsInventoryLimits,
    census: &mut InventoryCensus,
    collector: &mut C,
    name: DecodedSpan,
    entry: u64,
) -> Result<(), PreparedSafetensorsInventoryError> {
    if scanner.peek() != Some(b'{') {
        return Err(PreparedSafetensorsInventoryError::InvalidTensorMember {
            entry,
            member: TensorMember::Dtype,
        });
    }
    scanner.bump();
    scanner.skip_ws();

    let mut dtype = None;
    let mut shape = None;
    let mut offsets = None;
    if scanner.peek() != Some(b'}') {
        loop {
            let key = scanner.parse_string()?;
            let member = if key.matches(b"dtype") {
                TensorMember::Dtype
            } else if key.matches(b"shape") {
                TensorMember::Shape
            } else if key.matches(b"data_offsets") {
                TensorMember::DataOffsets
            } else {
                return Err(PreparedSafetensorsInventoryError::UnknownTensorMember { entry });
            };
            let duplicate = match member {
                TensorMember::Dtype => dtype.is_some(),
                TensorMember::Shape => shape.is_some(),
                TensorMember::DataOffsets => offsets.is_some(),
            };
            if duplicate {
                return Err(PreparedSafetensorsInventoryError::DuplicateTensorMember {
                    entry,
                    member,
                });
            }
            scanner.skip_ws();
            scanner.expect_byte(b':', HeaderSyntaxFault::ExpectedColon)?;
            scanner.skip_ws();

            match member {
                TensorMember::Dtype => {
                    if scanner.peek() != Some(b'"') {
                        return Err(PreparedSafetensorsInventoryError::InvalidTensorMember {
                            entry,
                            member,
                        });
                    }
                    let token = scanner.parse_string()?;
                    dtype = Some(parse_dtype(token, entry)?);
                }
                TensorMember::Shape => {
                    shape = Some(parse_shape(scanner, limits, collector, entry)?);
                }
                TensorMember::DataOffsets => {
                    offsets = Some(parse_offsets(scanner, entry)?);
                }
            }

            scanner.skip_ws();
            match scanner.peek() {
                Some(b',') => {
                    scanner.bump();
                    scanner.skip_ws();
                    if scanner.peek() == Some(b'}') {
                        return Err(scanner.error(HeaderSyntaxFault::TrailingComma));
                    }
                }
                Some(b'}') => break,
                Some(_) => {
                    return Err(scanner.error(HeaderSyntaxFault::ExpectedCommaOrEnd));
                }
                None => return Err(scanner.error(HeaderSyntaxFault::UnexpectedEof)),
            }
        }
    }
    scanner.expect_byte(b'}', HeaderSyntaxFault::ExpectedCommaOrEnd)?;

    let dtype = dtype.ok_or(PreparedSafetensorsInventoryError::MissingTensorMember {
        entry,
        member: TensorMember::Dtype,
    })?;
    let shape = shape.ok_or(PreparedSafetensorsInventoryError::MissingTensorMember {
        entry,
        member: TensorMember::Shape,
    })?;
    census.dimension_count = census
        .dimension_count
        .checked_add(
            u64::try_from(shape.span.len)
                .map_err(|_| overflow(InventoryExpression::DimensionCount))?,
        )
        .ok_or_else(|| overflow(InventoryExpression::DimensionCount))?;
    let (start, end) = offsets.ok_or(PreparedSafetensorsInventoryError::MissingTensorMember {
        entry,
        member: TensorMember::DataOffsets,
    })?;
    if start > end {
        return Err(PreparedSafetensorsInventoryError::ReversedDataRange { entry, start, end });
    }
    let data_len = u64::try_from(plan.data_len())
        .map_err(|_| overflow(InventoryExpression::TensorSourceBytes))?;
    if end > data_len {
        return Err(PreparedSafetensorsInventoryError::DataRangePastEnd {
            entry,
            end,
            data_len,
        });
    }
    let source_bytes = checked_product(
        shape.elements,
        dtype.width(),
        InventoryExpression::TensorSourceBytes,
    )?;
    let actual = end - start;
    if actual != source_bytes {
        return Err(
            PreparedSafetensorsInventoryError::TensorByteLengthMismatch {
                entry,
                expected: source_bytes,
                actual,
            },
        );
    }
    census.aggregate_elements = census
        .aggregate_elements
        .checked_add(shape.elements)
        .ok_or_else(|| overflow(InventoryExpression::AggregateElements))?;
    validate_limit(
        InventoryAxis::AggregateElements,
        census.aggregate_elements,
        limits.max_aggregate_elements,
    )?;
    census.source_payload_bytes = census
        .source_payload_bytes
        .checked_add(source_bytes)
        .ok_or_else(|| overflow(InventoryExpression::TensorSourceBytes))?;

    let absolute_start = u64::try_from(plan.header_end())
        .map_err(|_| overflow(InventoryExpression::AbsoluteDataOffset))?
        .checked_add(start)
        .ok_or_else(|| overflow(InventoryExpression::AbsoluteDataOffset))?;
    let alignment = if dtype == PreparedSafetensorsDtype::F32 {
        Some(if absolute_start % 4 == 0 {
            F32FileOffsetAlignment::FourByteAligned
        } else {
            F32FileOffsetAlignment::Unaligned
        })
    } else {
        None
    };
    collector.tensor_record(PreparedSafetensorsTensorRecord {
        name,
        dimensions: shape.span,
        dtype,
        data_start: start,
        data_end: end,
        elements: shape.elements,
        source_bytes,
        f32_file_offset_alignment: alignment,
    })
}

fn parse_dtype(
    token: JsonStringToken<'_>,
    entry: u64,
) -> Result<PreparedSafetensorsDtype, PreparedSafetensorsInventoryError> {
    if token.matches(b"F32") {
        Ok(PreparedSafetensorsDtype::F32)
    } else if token.matches(b"F16") {
        Ok(PreparedSafetensorsDtype::F16)
    } else if token.matches(b"BF16") {
        Ok(PreparedSafetensorsDtype::Bf16)
    } else {
        Err(PreparedSafetensorsInventoryError::UnsupportedDtype { entry })
    }
}

fn parse_shape<C: InventoryCollector>(
    scanner: &mut HeaderScanner<'_>,
    limits: &PreparedSafetensorsInventoryLimits,
    collector: &mut C,
    entry: u64,
) -> Result<ShapeFact, PreparedSafetensorsInventoryError> {
    if scanner.peek() != Some(b'[') {
        return Err(PreparedSafetensorsInventoryError::InvalidTensorMember {
            entry,
            member: TensorMember::Shape,
        });
    }
    scanner.bump();
    scanner.skip_ws();
    let start = collector.dimension_start();
    let mut rank = 0_u64;
    let mut elements = 1_u64;
    if scanner.peek() != Some(b']') {
        loop {
            let dimension = scanner.parse_u64()?;
            rank = rank
                .checked_add(1)
                .ok_or_else(|| overflow(InventoryExpression::DimensionCount))?;
            validate_limit(InventoryAxis::Rank, rank, limits.max_rank)?;
            validate_limit(InventoryAxis::Dimension, dimension, limits.max_dimension)?;
            validate_platform(
                InventoryAxis::Dimension,
                dimension,
                u64::try_from(usize::MAX).unwrap_or(u64::MAX),
            )?;
            elements = checked_product(elements, dimension, InventoryExpression::TensorElements)?;
            collector.dimension(dimension)?;
            scanner.skip_ws();
            match scanner.peek() {
                Some(b',') => {
                    scanner.bump();
                    scanner.skip_ws();
                    if scanner.peek() == Some(b']') {
                        return Err(scanner.error(HeaderSyntaxFault::TrailingComma));
                    }
                }
                Some(b']') => break,
                Some(_) => {
                    return Err(scanner.error(HeaderSyntaxFault::ExpectedCommaOrEnd));
                }
                None => return Err(scanner.error(HeaderSyntaxFault::UnexpectedEof)),
            }
        }
    }
    scanner.expect_byte(b']', HeaderSyntaxFault::ExpectedCommaOrEnd)?;
    let len = to_usize(InventoryAxis::Rank, rank)?;
    Ok(ShapeFact {
        span: DecodedSpan { start, len },
        elements,
    })
}

fn parse_offsets(
    scanner: &mut HeaderScanner<'_>,
    entry: u64,
) -> Result<(u64, u64), PreparedSafetensorsInventoryError> {
    if scanner.peek() != Some(b'[') {
        return Err(PreparedSafetensorsInventoryError::InvalidTensorMember {
            entry,
            member: TensorMember::DataOffsets,
        });
    }
    scanner.bump();
    scanner.skip_ws();
    let mut values = [0_u64; 2];
    let mut count = 0_u64;
    if scanner.peek() != Some(b']') {
        loop {
            let value = scanner.parse_u64()?;
            if let Ok(index) = usize::try_from(count)
                && index < values.len()
            {
                values[index] = value;
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| overflow(InventoryExpression::TensorCount))?;
            scanner.skip_ws();
            match scanner.peek() {
                Some(b',') => {
                    scanner.bump();
                    scanner.skip_ws();
                    if scanner.peek() == Some(b']') {
                        return Err(scanner.error(HeaderSyntaxFault::TrailingComma));
                    }
                }
                Some(b']') => break,
                Some(_) => {
                    return Err(scanner.error(HeaderSyntaxFault::ExpectedCommaOrEnd));
                }
                None => return Err(scanner.error(HeaderSyntaxFault::UnexpectedEof)),
            }
        }
    }
    scanner.expect_byte(b']', HeaderSyntaxFault::ExpectedCommaOrEnd)?;
    if count != 2 {
        return Err(PreparedSafetensorsInventoryError::InvalidDataOffsetsArity {
            entry,
            actual: count,
        });
    }
    Ok((values[0], values[1]))
}

fn parse_metadata<C: InventoryCollector>(
    scanner: &mut HeaderScanner<'_>,
    census: &mut InventoryCensus,
    collector: &mut C,
) -> Result<(), PreparedSafetensorsInventoryError> {
    if scanner.peek() != Some(b'{') {
        return Err(PreparedSafetensorsInventoryError::InvalidMetadata);
    }
    scanner.bump();
    scanner.skip_ws();
    if scanner.peek() == Some(b'}') {
        scanner.bump();
        return Ok(());
    }
    loop {
        if scanner.peek() != Some(b'"') {
            return Err(PreparedSafetensorsInventoryError::InvalidMetadata);
        }
        let key = scanner.parse_string()?;
        census.metadata_key_count = census
            .metadata_key_count
            .checked_add(1)
            .ok_or_else(|| overflow(InventoryExpression::MetadataBytes))?;
        census.decoded_metadata_key_bytes = census
            .decoded_metadata_key_bytes
            .checked_add(key.decoded_len)
            .ok_or_else(|| overflow(InventoryExpression::MetadataBytes))?;
        collector.metadata_key(key)?;
        scanner.skip_ws();
        scanner.expect_byte(b':', HeaderSyntaxFault::ExpectedColon)?;
        scanner.skip_ws();
        if scanner.peek() != Some(b'"') {
            return Err(PreparedSafetensorsInventoryError::InvalidMetadata);
        }
        let value = scanner.parse_string()?;
        census.decoded_metadata_value_bytes = census
            .decoded_metadata_value_bytes
            .checked_add(value.decoded_len)
            .ok_or_else(|| overflow(InventoryExpression::MetadataBytes))?;
        scanner.skip_ws();
        match scanner.peek() {
            Some(b',') => {
                scanner.bump();
                scanner.skip_ws();
                if scanner.peek() == Some(b'}') {
                    return Err(scanner.error(HeaderSyntaxFault::TrailingComma));
                }
            }
            Some(b'}') => {
                scanner.bump();
                break;
            }
            Some(_) => return Err(scanner.error(HeaderSyntaxFault::ExpectedCommaOrEnd)),
            None => return Err(scanner.error(HeaderSyntaxFault::UnexpectedEof)),
        }
    }
    Ok(())
}

fn finish_census(
    header_len: usize,
    limits: &PreparedSafetensorsInventoryLimits,
    census: &mut InventoryCensus,
) -> Result<(), PreparedSafetensorsInventoryError> {
    let record_width = usize_to_u64(
        InventoryExpression::MetadataBytes,
        std::mem::size_of::<PreparedSafetensorsTensorRecord>(),
    )?;
    let record_bytes = census
        .tensor_count
        .checked_mul(record_width)
        .ok_or_else(|| overflow(InventoryExpression::MetadataBytes))?;
    let dimension_width = usize_to_u64(
        InventoryExpression::MetadataBytes,
        std::mem::size_of::<u64>(),
    )?;
    let dimension_bytes = census
        .dimension_count
        .checked_mul(dimension_width)
        .ok_or_else(|| overflow(InventoryExpression::MetadataBytes))?;
    census.retained_metadata_bytes = record_bytes
        .checked_add(census.decoded_name_bytes)
        .and_then(|value| value.checked_add(dimension_bytes))
        .ok_or_else(|| overflow(InventoryExpression::MetadataBytes))?;
    validate_limit(
        InventoryAxis::MetadataBytes,
        census.retained_metadata_bytes,
        limits.max_metadata_bytes,
    )?;

    let metadata_key_record_width = usize_to_u64(
        InventoryExpression::SafetensorsParseWorkBytes,
        std::mem::size_of::<DecodedSpan>(),
    )?;
    let metadata_key_record_bytes = census
        .metadata_key_count
        .checked_mul(metadata_key_record_width)
        .ok_or_else(|| overflow(InventoryExpression::SafetensorsParseWorkBytes))?;
    census.parse_work_bytes = u64::try_from(header_len)
        .map_err(|_| overflow(InventoryExpression::SafetensorsParseWorkBytes))?
        .checked_add(metadata_key_record_bytes)
        .and_then(|value| value.checked_add(census.decoded_metadata_key_bytes))
        .ok_or_else(|| overflow(InventoryExpression::SafetensorsParseWorkBytes))?;
    validate_limit(
        InventoryAxis::SafetensorsParseWorkBytes,
        census.parse_work_bytes,
        limits.max_parse_work_bytes,
    )?;
    if census.decoded_metadata_value_bytes > u64::try_from(header_len).unwrap_or(u64::MAX) {
        return Err(overflow(InventoryExpression::MetadataBytes));
    }
    Ok(())
}

struct HeaderScanner<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> HeaderScanner<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.position += 1;
        Some(byte)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    fn error(&self, fault: HeaderSyntaxFault) -> PreparedSafetensorsInventoryError {
        PreparedSafetensorsInventoryError::MalformedHeader {
            at: self.position,
            fault,
        }
    }

    fn expect_byte(
        &mut self,
        expected: u8,
        fault: HeaderSyntaxFault,
    ) -> Result<(), PreparedSafetensorsInventoryError> {
        match self.peek() {
            Some(actual) if actual == expected => {
                self.position += 1;
                Ok(())
            }
            _ => Err(self.error(fault)),
        }
    }

    fn parse_string(&mut self) -> Result<JsonStringToken<'a>, PreparedSafetensorsInventoryError> {
        self.expect_byte(b'"', HeaderSyntaxFault::ExpectedString)?;
        let content_start = self.position;
        loop {
            match self.bump() {
                Some(b'"') => {
                    let content_end = self.position - 1;
                    let raw = &self.bytes[content_start..content_end];
                    let decoded_len = decode_json_content(raw, content_start, &mut |_| Ok(()))?;
                    return Ok(JsonStringToken {
                        raw,
                        raw_start: content_start,
                        decoded_len,
                    });
                }
                Some(b'\\') => {
                    if self.bump().is_none() {
                        return Err(self.error(HeaderSyntaxFault::UnexpectedEof));
                    }
                }
                Some(0..=0x1f) => {
                    return Err(self.error(HeaderSyntaxFault::InvalidStringEscape));
                }
                Some(_) => {}
                None => return Err(self.error(HeaderSyntaxFault::UnexpectedEof)),
            }
        }
    }

    fn parse_u64(&mut self) -> Result<u64, PreparedSafetensorsInventoryError> {
        let start = self.position;
        let mut value = 0_u64;
        match self.peek() {
            Some(b'0') => {
                self.bump();
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error(HeaderSyntaxFault::InvalidUnsignedInteger));
                }
            }
            Some(b'1'..=b'9') => {
                while let Some(digit @ b'0'..=b'9') = self.peek() {
                    self.bump();
                    value = value
                        .checked_mul(10)
                        .and_then(|current| current.checked_add(u64::from(digit - b'0')))
                        .ok_or_else(|| self.error(HeaderSyntaxFault::InvalidUnsignedInteger))?;
                }
            }
            _ => return Err(self.error(HeaderSyntaxFault::InvalidUnsignedInteger)),
        }
        if start == self.position || matches!(self.peek(), Some(b'.' | b'e' | b'E' | b'+' | b'-')) {
            return Err(self.error(HeaderSyntaxFault::InvalidUnsignedInteger));
        }
        Ok(value)
    }
}

#[derive(Clone, Copy)]
struct JsonStringToken<'a> {
    raw: &'a [u8],
    raw_start: usize,
    decoded_len: u64,
}

impl JsonStringToken<'_> {
    fn matches(self, expected: &[u8]) -> bool {
        let mut offset = 0_usize;
        let mut equal = true;
        let decoded = decode_json_content(self.raw, self.raw_start, &mut |chunk| {
            let Some(end) = offset.checked_add(chunk.len()) else {
                equal = false;
                return Ok(());
            };
            if expected.get(offset..end) != Some(chunk) {
                equal = false;
            }
            offset = end;
            Ok(())
        });
        decoded.is_ok() && equal && offset == expected.len()
    }
}

fn decode_json_content<F>(
    raw: &[u8],
    raw_start: usize,
    emit: &mut F,
) -> Result<u64, PreparedSafetensorsInventoryError>
where
    F: FnMut(&[u8]) -> Result<(), PreparedSafetensorsInventoryError>,
{
    let mut position = 0_usize;
    let mut decoded_len = 0_u64;
    while position < raw.len() {
        if raw[position] != b'\\' {
            let start = position;
            while position < raw.len() && raw[position] != b'\\' {
                if raw[position] <= 0x1f {
                    return Err(PreparedSafetensorsInventoryError::MalformedHeader {
                        at: raw_start + position,
                        fault: HeaderSyntaxFault::InvalidStringEscape,
                    });
                }
                position += 1;
            }
            let chunk = &raw[start..position];
            emit(chunk)?;
            decoded_len = decoded_len
                .checked_add(usize_to_u64(
                    InventoryExpression::DecodedNameBytes,
                    chunk.len(),
                )?)
                .ok_or_else(|| overflow(InventoryExpression::DecodedNameBytes))?;
            continue;
        }

        let escape_at = position;
        position += 1;
        let escaped = raw.get(position).copied().ok_or(
            PreparedSafetensorsInventoryError::MalformedHeader {
                at: raw_start + escape_at,
                fault: HeaderSyntaxFault::InvalidStringEscape,
            },
        )?;
        position += 1;
        let mut single = [0_u8; 4];
        let output: &[u8] = match escaped {
            b'"' | b'\\' | b'/' => {
                single[0] = escaped;
                &single[..1]
            }
            b'b' => {
                single[0] = 0x08;
                &single[..1]
            }
            b'f' => {
                single[0] = 0x0c;
                &single[..1]
            }
            b'n' => {
                single[0] = b'\n';
                &single[..1]
            }
            b'r' => {
                single[0] = b'\r';
                &single[..1]
            }
            b't' => {
                single[0] = b'\t';
                &single[..1]
            }
            b'u' => {
                let (first, next) = parse_hex_quad(raw, position, raw_start)?;
                position = next;
                let scalar = if (0xd800..=0xdbff).contains(&first) {
                    if raw.get(position..position + 2) != Some(b"\\u") {
                        return Err(PreparedSafetensorsInventoryError::MalformedHeader {
                            at: raw_start + position,
                            fault: HeaderSyntaxFault::InvalidUnicodeEscape,
                        });
                    }
                    let (second, after) = parse_hex_quad(raw, position + 2, raw_start)?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return Err(PreparedSafetensorsInventoryError::MalformedHeader {
                            at: raw_start + position,
                            fault: HeaderSyntaxFault::InvalidUnicodeEscape,
                        });
                    }
                    position = after;
                    0x1_0000 + ((u32::from(first) - 0xd800) << 10) + (u32::from(second) - 0xdc00)
                } else if (0xdc00..=0xdfff).contains(&first) {
                    return Err(PreparedSafetensorsInventoryError::MalformedHeader {
                        at: raw_start + escape_at,
                        fault: HeaderSyntaxFault::InvalidUnicodeEscape,
                    });
                } else {
                    u32::from(first)
                };
                let character = char::from_u32(scalar).ok_or(
                    PreparedSafetensorsInventoryError::MalformedHeader {
                        at: raw_start + escape_at,
                        fault: HeaderSyntaxFault::InvalidUnicodeEscape,
                    },
                )?;
                character.encode_utf8(&mut single).as_bytes()
            }
            _ => {
                return Err(PreparedSafetensorsInventoryError::MalformedHeader {
                    at: raw_start + escape_at,
                    fault: HeaderSyntaxFault::InvalidStringEscape,
                });
            }
        };
        emit(output)?;
        decoded_len = decoded_len
            .checked_add(usize_to_u64(
                InventoryExpression::DecodedNameBytes,
                output.len(),
            )?)
            .ok_or_else(|| overflow(InventoryExpression::DecodedNameBytes))?;
    }
    Ok(decoded_len)
}

fn parse_hex_quad(
    raw: &[u8],
    start: usize,
    raw_start: usize,
) -> Result<(u16, usize), PreparedSafetensorsInventoryError> {
    let end = start
        .checked_add(4)
        .ok_or(PreparedSafetensorsInventoryError::MalformedHeader {
            at: raw_start + start,
            fault: HeaderSyntaxFault::InvalidUnicodeEscape,
        })?;
    let digits = raw
        .get(start..end)
        .ok_or(PreparedSafetensorsInventoryError::MalformedHeader {
            at: raw_start + start,
            fault: HeaderSyntaxFault::InvalidUnicodeEscape,
        })?;
    let mut value = 0_u16;
    for &digit in digits {
        let nibble = match digit {
            b'0'..=b'9' => u16::from(digit - b'0'),
            b'a'..=b'f' => u16::from(digit - b'a') + 10,
            b'A'..=b'F' => u16::from(digit - b'A') + 10,
            _ => {
                return Err(PreparedSafetensorsInventoryError::MalformedHeader {
                    at: raw_start + start,
                    fault: HeaderSyntaxFault::InvalidUnicodeEscape,
                });
            }
        };
        value = (value << 4) | nibble;
    }
    Ok((value, end))
}

fn append_token(
    arena: &mut Vec<u8>,
    token: JsonStringToken<'_>,
    _allocation_arena: AllocationArena,
) -> Result<DecodedSpan, PreparedSafetensorsInventoryError> {
    let len = to_usize(InventoryAxis::MetadataBytes, token.decoded_len)?;
    let start = arena.len();
    let end = start
        .checked_add(len)
        .ok_or_else(|| overflow(InventoryExpression::MetadataBytes))?;
    if end > arena.capacity() {
        return Err(PreparedSafetensorsInventoryError::SecondPassCensusMismatch);
    }
    let decoded = decode_json_content(token.raw, token.raw_start, &mut |chunk| {
        arena.extend_from_slice(chunk);
        Ok(())
    })?;
    if decoded != token.decoded_len || arena.len() != end {
        return Err(PreparedSafetensorsInventoryError::SecondPassCensusMismatch);
    }
    Ok(DecodedSpan { start, len })
}

fn reserve_exact<T, F>(
    arena: &mut Vec<T>,
    elements: usize,
    kind: AllocationArena,
    probe: &mut F,
) -> Result<(), PreparedSafetensorsInventoryError>
where
    F: FnMut(AllocationArena, usize) -> Result<(), PreparedSafetensorsInventoryError>,
{
    let requested_bytes = elements.checked_mul(std::mem::size_of::<T>()).ok_or(
        PreparedSafetensorsInventoryError::AllocationFailed {
            arena: kind,
            requested_bytes: usize::MAX,
        },
    )?;
    probe(kind, requested_bytes)?;
    arena.try_reserve_exact(elements).map_err(|_| {
        PreparedSafetensorsInventoryError::AllocationFailed {
            arena: kind,
            requested_bytes,
        }
    })
}

fn record_name<'a>(names: &'a [u8], record: &PreparedSafetensorsTensorRecord) -> &'a [u8] {
    span_bytes(names, record.name)
}

fn span_bytes(bytes: &[u8], span: DecodedSpan) -> &[u8] {
    match span.end().and_then(|end| bytes.get(span.start..end)) {
        Some(value) => value,
        None => &[],
    }
}

fn span_dimensions(dimensions: &[u64], span: DecodedSpan) -> &[u64] {
    match span.end().and_then(|end| dimensions.get(span.start..end)) {
        Some(value) => value,
        None => &[],
    }
}

fn validate_limit(
    axis: InventoryAxis,
    actual: u64,
    limit: u64,
) -> Result<(), PreparedSafetensorsInventoryError> {
    if actual > limit {
        return Err(PreparedSafetensorsInventoryError::Exceeded {
            axis,
            actual,
            limit,
        });
    }
    Ok(())
}

fn validate_platform(
    axis: InventoryAxis,
    value: u64,
    platform_max: u64,
) -> Result<(), PreparedSafetensorsInventoryError> {
    if value > platform_max {
        return Err(PreparedSafetensorsInventoryError::PlatformUnrepresentable { axis, value });
    }
    Ok(())
}

fn to_usize(axis: InventoryAxis, value: u64) -> Result<usize, PreparedSafetensorsInventoryError> {
    usize::try_from(value)
        .map_err(|_| PreparedSafetensorsInventoryError::PlatformUnrepresentable { axis, value })
}

fn usize_to_u64(
    expression: InventoryExpression,
    value: usize,
) -> Result<u64, PreparedSafetensorsInventoryError> {
    u64::try_from(value).map_err(|_| overflow(expression))
}

fn overflow(expression: InventoryExpression) -> PreparedSafetensorsInventoryError {
    PreparedSafetensorsInventoryError::ArithmeticOverflow { expression }
}

fn checked_product(
    left: u64,
    right: u64,
    expression: InventoryExpression,
) -> Result<u64, PreparedSafetensorsInventoryError> {
    left.checked_mul(right).ok_or_else(|| overflow(expression))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::weights::prepared_safetensors::{
        PreparedSafetensorsFramingLimits, PreparedSafetensorsHeaderPlan,
        plan_prepared_safetensors_header,
    };

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn plan(header_len: usize, data_len: usize) -> PreparedSafetensorsHeaderPlan {
        let declared = 8_u64
            .checked_add(u64::try_from(header_len).unwrap())
            .and_then(|value| value.checked_add(u64::try_from(data_len).unwrap()))
            .unwrap();
        let limits = PreparedSafetensorsFramingLimits::try_new(
            nz(declared),
            nz(u64::try_from(header_len).unwrap()),
        )
        .unwrap();
        plan_prepared_safetensors_header(
            u64::try_from(header_len).unwrap().to_le_bytes(),
            declared,
            &limits,
        )
        .unwrap()
    }

    fn wide_limits() -> PreparedSafetensorsInventoryLimits {
        PreparedSafetensorsInventoryLimits::try_new(
            nz(128),
            nz(256),
            nz(32),
            nz(1_000_000),
            nz(10_000_000),
            nz(1_000_000),
            nz(2_000_000),
        )
        .unwrap()
    }

    fn parse(
        header: &[u8],
        data_len: usize,
    ) -> Result<PreparedSafetensorsInventoryFacts, PreparedSafetensorsInventoryError> {
        parse_prepared_safetensors_header_inventory(
            header,
            &plan(header.len(), data_len),
            &wide_limits(),
        )
    }

    fn tensor<'a>(
        facts: &'a PreparedSafetensorsInventoryFacts,
        name: &[u8],
    ) -> PreparedSafetensorsTensorView<'a> {
        facts
            .tensors()
            .find(|tensor| tensor.name_bytes() == name)
            .unwrap()
    }

    fn assert_malformed(header: &[u8], data_len: usize, expected: HeaderSyntaxFault) {
        let error = parse(header, data_len).unwrap_err();
        assert!(
            matches!(
                error,
                PreparedSafetensorsInventoryError::MalformedHeader { fault, .. }
                    if fault == expected
            ),
            "expected {expected:?}, got {error:?}"
        );
    }

    const CANONICAL: &str = r#"{"zz":{"shape":[1,2],"data_offsets":[4,8],"dtype":"F16"},"__metadata__":{"format":"pt","rocket":"\ud83d\ude80"},"aa":{"data_offsets":[0,4],"dtype":"F32","shape":[1]},"mm":{"dtype":"BF16","shape":[],"data_offsets":[8,10]}}"#;

    #[test]
    fn canonical_inventory_is_sorted_deterministic_and_exactly_accounted() {
        let first = parse(CANONICAL.as_bytes(), 10).unwrap();
        let permuted = r#"{"mm":{"shape":[],"data_offsets":[8,10],"dtype":"BF16"},"aa":{"shape":[1],"dtype":"F32","data_offsets":[0,4]},"__metadata__":{"rocket":"\ud83d\ude80","format":"pt"},"zz":{"dtype":"F16","data_offsets":[4,8],"shape":[1,2]}}"#;
        let second = parse(permuted.as_bytes(), 10).unwrap();

        assert_ne!(first.name_bytes, second.name_bytes);
        assert_ne!(first.dimensions, second.dimensions);
        assert_eq!(first, second);
        let renamed = parse(CANONICAL.replacen("\"aa\"", "\"ab\"", 1).as_bytes(), 10).unwrap();
        assert_ne!(first, renamed);
        let reshaped = parse(CANONICAL.replacen("[1,2]", "[2,1]", 1).as_bytes(), 10).unwrap();
        assert_ne!(first, reshaped);
        let names: Vec<_> = first
            .tensors()
            .map(|tensor| tensor.name_bytes().to_vec())
            .collect();
        assert_eq!(names, [b"aa".to_vec(), b"mm".to_vec(), b"zz".to_vec()]);

        let a = tensor(&first, b"aa");
        assert_eq!(a.dtype(), PreparedSafetensorsDtype::F32);
        assert_eq!(a.dimensions(), &[1]);
        assert_eq!(a.data_range(), 0..4);
        assert_eq!(a.elements(), 1);
        assert_eq!(a.source_bytes(), 4);

        let m = tensor(&first, b"mm");
        assert_eq!(m.dtype(), PreparedSafetensorsDtype::Bf16);
        assert!(m.dimensions().is_empty());
        assert_eq!(m.elements(), 1);
        assert_eq!(m.source_bytes(), 2);

        let z = tensor(&first, b"zz");
        assert_eq!(z.dtype(), PreparedSafetensorsDtype::F16);
        assert_eq!(z.dimensions(), &[1, 2]);
        assert_eq!(z.elements(), 2);
        assert_eq!(z.source_bytes(), 4);

        let retained = 3 * std::mem::size_of::<PreparedSafetensorsTensorRecord>() + 6 + 3 * 8;
        let parse_work = CANONICAL.len() + 2 * std::mem::size_of::<DecodedSpan>() + 12;
        assert_eq!(first.tensor_count(), 3);
        assert_eq!(first.aggregate_elements(), 4);
        assert_eq!(first.source_payload_bytes(), 10);
        assert_eq!(first.logical_retained_inventory_metadata_bytes(), retained);
        assert_eq!(first.logical_safetensors_parse_work_bytes(), parse_work);
        assert_eq!(first.decoded_metadata_value_bytes(), 6);
    }

    #[test]
    fn exact_header_envelope_and_closed_outer_grammar_are_pinned() {
        let one = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        assert!(parse(one, 4).is_ok());
        assert!(parse(
            b"{ \n \t\"a\" : { \"dtype\" : \"F32\" , \"shape\" : [ 1 ] , \"data_offsets\" : [ 0 , 4 ] } }",
            4,
        )
        .is_ok());
        assert_eq!(
            parse_prepared_safetensors_header_inventory(
                &one[..one.len() - 1],
                &plan(one.len(), 4),
                &wide_limits(),
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::HeaderLengthMismatch {
                expected: one.len(),
                actual: one.len() - 1,
            }
        );
        let mut longer = one.to_vec();
        longer.push(b' ');
        assert_eq!(
            parse_prepared_safetensors_header_inventory(
                &longer,
                &plan(one.len(), 4),
                &wide_limits(),
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::HeaderLengthMismatch {
                expected: one.len(),
                actual: one.len() + 1,
            }
        );

        let invalid_utf8 = b"{\"\xff\":{}}";
        assert!(matches!(
            parse(invalid_utf8, 0),
            Err(PreparedSafetensorsInventoryError::InvalidUtf8 { .. })
        ));
        assert_malformed(
            br#" {"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            4,
            HeaderSyntaxFault::ExpectedObjectStart,
        );
        assert_malformed(
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},}"#,
            4,
            HeaderSyntaxFault::TrailingComma,
        );
        assert_malformed(
            b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}\n",
            4,
            HeaderSyntaxFault::TrailingNonSpace,
        );
        let mut padded = one.to_vec();
        padded.extend_from_slice(b"   ");
        assert!(parse(&padded, 4).is_ok());
        for empty in [
            br#"{}"#.as_slice(),
            br#"{"__metadata__":{"format":"pt"}}"#.as_slice(),
        ] {
            assert_eq!(
                parse(empty, 0).unwrap_err(),
                PreparedSafetensorsInventoryError::MissingInventory
            );
        }
    }

    #[test]
    fn every_limit_accepts_exact_and_rejects_plus_one_before_reserve() {
        let facts = parse(CANONICAL.as_bytes(), 10).unwrap();
        let exact = PreparedSafetensorsInventoryLimits::try_new(
            nz(3),
            nz(2),
            nz(2),
            nz(2),
            nz(4),
            nz(u64::try_from(facts.logical_retained_inventory_metadata_bytes()).unwrap()),
            nz(u64::try_from(facts.logical_safetensors_parse_work_bytes()).unwrap()),
        )
        .unwrap();
        assert!(
            parse_prepared_safetensors_header_inventory(
                CANONICAL.as_bytes(),
                &plan(CANONICAL.len(), 10),
                &exact,
            )
            .is_ok()
        );

        let cases = [
            (
                InventoryAxis::Tensors,
                3,
                2,
                PreparedSafetensorsInventoryLimits::try_new(
                    nz(2),
                    nz(256),
                    nz(32),
                    nz(1_000_000),
                    nz(10_000_000),
                    nz(1_000_000),
                    nz(2_000_000),
                )
                .unwrap(),
            ),
            (
                InventoryAxis::TensorNameBytes,
                2,
                1,
                PreparedSafetensorsInventoryLimits::try_new(
                    nz(128),
                    nz(1),
                    nz(32),
                    nz(1_000_000),
                    nz(10_000_000),
                    nz(1_000_000),
                    nz(2_000_000),
                )
                .unwrap(),
            ),
            (
                InventoryAxis::Rank,
                2,
                1,
                PreparedSafetensorsInventoryLimits::try_new(
                    nz(128),
                    nz(256),
                    nz(1),
                    nz(1_000_000),
                    nz(10_000_000),
                    nz(1_000_000),
                    nz(2_000_000),
                )
                .unwrap(),
            ),
            (
                InventoryAxis::Dimension,
                2,
                1,
                PreparedSafetensorsInventoryLimits::try_new(
                    nz(128),
                    nz(256),
                    nz(32),
                    nz(1),
                    nz(10_000_000),
                    nz(1_000_000),
                    nz(2_000_000),
                )
                .unwrap(),
            ),
            (
                InventoryAxis::AggregateElements,
                4,
                3,
                PreparedSafetensorsInventoryLimits::try_new(
                    nz(128),
                    nz(256),
                    nz(32),
                    nz(1_000_000),
                    nz(3),
                    nz(1_000_000),
                    nz(2_000_000),
                )
                .unwrap(),
            ),
            (
                InventoryAxis::MetadataBytes,
                u64::try_from(facts.logical_retained_inventory_metadata_bytes()).unwrap(),
                u64::try_from(facts.logical_retained_inventory_metadata_bytes() - 1).unwrap(),
                PreparedSafetensorsInventoryLimits::try_new(
                    nz(128),
                    nz(256),
                    nz(32),
                    nz(1_000_000),
                    nz(10_000_000),
                    nz(
                        u64::try_from(facts.logical_retained_inventory_metadata_bytes() - 1)
                            .unwrap(),
                    ),
                    nz(2_000_000),
                )
                .unwrap(),
            ),
            (
                InventoryAxis::SafetensorsParseWorkBytes,
                u64::try_from(facts.logical_safetensors_parse_work_bytes()).unwrap(),
                u64::try_from(facts.logical_safetensors_parse_work_bytes() - 1).unwrap(),
                PreparedSafetensorsInventoryLimits::try_new(
                    nz(128),
                    nz(256),
                    nz(32),
                    nz(1_000_000),
                    nz(10_000_000),
                    nz(1_000_000),
                    nz(u64::try_from(facts.logical_safetensors_parse_work_bytes() - 1).unwrap()),
                )
                .unwrap(),
            ),
        ];

        for (axis, actual, limit, limits) in cases {
            let mut attempts = Vec::new();
            let error = parse_prepared_safetensors_header_inventory_with_reserve_probe(
                CANONICAL.as_bytes(),
                &plan(CANONICAL.len(), 10),
                &limits,
                &mut |arena, _| {
                    attempts.push(arena);
                    Ok(())
                },
            )
            .unwrap_err();
            assert_eq!(
                error,
                PreparedSafetensorsInventoryError::Exceeded {
                    axis,
                    actual,
                    limit,
                }
            );
            assert!(attempts.is_empty(), "{axis:?} failed after reserve began");
        }
    }

    #[test]
    fn limit_policy_pins_platform_and_canonical_u32_framing() {
        assert!(
            PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                nz(8),
                nz(8),
                nz(8),
                nz(8),
                nz(1),
                nz(8),
                nz(1),
                8,
                8,
            )
            .is_ok()
        );
        let platform_cases = [
            (
                InventoryAxis::Tensors,
                PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                    nz(9),
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(1),
                    8,
                    8,
                ),
            ),
            (
                InventoryAxis::TensorNameBytes,
                PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                    nz(1),
                    nz(9),
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(1),
                    8,
                    8,
                ),
            ),
            (
                InventoryAxis::Rank,
                PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                    nz(1),
                    nz(1),
                    nz(9),
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(1),
                    8,
                    8,
                ),
            ),
            (
                InventoryAxis::Dimension,
                PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(9),
                    nz(1),
                    nz(1),
                    nz(1),
                    8,
                    8,
                ),
            ),
            (
                InventoryAxis::MetadataBytes,
                PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(1),
                    nz(9),
                    nz(1),
                    8,
                    8,
                ),
            ),
        ];
        for (axis, result) in platform_cases {
            assert_eq!(
                result.unwrap_err(),
                PreparedSafetensorsInventoryError::PlatformUnrepresentable { axis, value: 9 }
            );
        }

        for (axis, name, rank) in [
            (
                InventoryAxis::TensorNameBytes,
                CANONICAL_INVENTORY_U32_FIELD_MAX + 1,
                1,
            ),
            (
                InventoryAxis::Rank,
                1,
                CANONICAL_INVENTORY_U32_FIELD_MAX + 1,
            ),
        ] {
            let result = PreparedSafetensorsInventoryLimits::try_new_with_platform_max(
                nz(1),
                nz(name),
                nz(rank),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                u64::MAX,
                u64::MAX,
            );
            assert_eq!(
                result.unwrap_err(),
                PreparedSafetensorsInventoryError::Exceeded {
                    axis,
                    actual: CANONICAL_INVENTORY_U32_FIELD_MAX + 1,
                    limit: CANONICAL_INVENTORY_U32_FIELD_MAX,
                }
            );
        }

        let isize_max = u64::try_from(isize::MAX).unwrap();
        assert!(
            PreparedSafetensorsInventoryLimits::try_new(
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(isize_max),
                nz(1),
            )
            .is_ok()
        );
        assert_eq!(
            PreparedSafetensorsInventoryLimits::try_new(
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(1),
                nz(isize_max + 1),
                nz(1),
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::PlatformUnrepresentable {
                axis: InventoryAxis::MetadataBytes,
                value: isize_max + 1,
            }
        );
    }

    #[test]
    fn semantic_equality_compares_each_independent_materialized_fact() {
        let f16 = parse(
            br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}} "#,
            2,
        )
        .unwrap();
        let bf16 = parse(
            br#"{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
            2,
        )
        .unwrap();
        assert_eq!(
            f16.logical_safetensors_parse_work_bytes(),
            bf16.logical_safetensors_parse_work_bytes()
        );
        assert_ne!(f16, bf16);

        let original_ranges = parse(
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
            8,
        )
        .unwrap();
        let swapped_ranges = parse(
            br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            8,
        )
        .unwrap();
        assert_ne!(original_ranges, swapped_ranges);

        let compact = parse(
            br#"{"a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#,
            0,
        )
        .unwrap();
        let padded = parse(
            br#"{"a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}} "#,
            0,
        )
        .unwrap();
        assert_ne!(compact, padded);

        let one_decoded_value_byte = parse(
            br#"{"__metadata__":{"k":"\n"},"a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#,
            0,
        )
        .unwrap();
        let two_decoded_value_bytes = parse(
            br#"{"__metadata__":{"k":"ab"},"a":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#,
            0,
        )
        .unwrap();
        assert_eq!(
            one_decoded_value_byte.logical_safetensors_parse_work_bytes(),
            two_decoded_value_bytes.logical_safetensors_parse_work_bytes()
        );
        assert_ne!(one_decoded_value_byte, two_decoded_value_bytes);
    }

    #[test]
    fn decoded_names_and_member_aliases_use_json_semantics_without_normalization() {
        let duplicate = r#"{"é":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"middle":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"\u00e9":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        assert_eq!(
            parse(duplicate.as_bytes(), 0).unwrap_err(),
            PreparedSafetensorsInventoryError::DuplicateTensorName
        );

        let distinct = r#"{"é":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"e\u0301":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"\u0000":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"a/b":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"../x":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        let facts = parse(distinct.as_bytes(), 0).unwrap();
        let names: Vec<_> = facts
            .tensors()
            .map(|tensor| tensor.name_bytes().to_vec())
            .collect();
        assert!(names.contains(&Vec::new()));
        assert!(names.contains(&vec![0]));
        assert!(names.contains(&b"a/b".to_vec()));
        assert!(names.contains(&b"../x".to_vec()));
        assert!(names.contains(&"é".as_bytes().to_vec()));
        assert!(names.contains(&"e\u{301}".as_bytes().to_vec()));

        let surrogate = parse(
            br#"{"\ud83d\ude80":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
            0,
        )
        .unwrap();
        assert_eq!(
            tensor(&surrogate, "🚀".as_bytes()).name_bytes(),
            "🚀".as_bytes()
        );

        let duplicate_member =
            br#"{"a":{"dtype":"F32","d\u0074ype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        assert!(matches!(
            parse(duplicate_member, 0),
            Err(PreparedSafetensorsInventoryError::DuplicateTensorMember {
                member: TensorMember::Dtype,
                ..
            })
        ));
    }

    #[test]
    fn metadata_is_bounded_string_map_and_values_never_enter_an_arena() {
        let valid = r#"{"__metadata__":{"raw":"é","escaped":"\n","bmp":"\u03bb","pair":"\ud83d\ude80"},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        let facts = parse(valid.as_bytes(), 0).unwrap();
        assert_eq!(facts.decoded_metadata_value_bytes(), 2 + 1 + 2 + 4);
        assert!(facts.decoded_metadata_value_bytes() <= valid.len());

        let repeated = br#"{"__metadata__":{},"__metadata__":{},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        assert_eq!(
            parse(repeated, 0).unwrap_err(),
            PreparedSafetensorsInventoryError::DuplicateMetadataMember
        );
        let duplicate_key = r#"{"__metadata__":{"é":"x","middle":"z","\u00e9":"y"},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        assert_eq!(
            parse(duplicate_key.as_bytes(), 0).unwrap_err(),
            PreparedSafetensorsInventoryError::DuplicateMetadataKey
        );
        for bad in [
            br#"{"__metadata__":[],"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#
                .as_slice(),
            br#"{"__metadata__":{"x":1},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#
                .as_slice(),
        ] {
            assert_eq!(
                parse(bad, 0).unwrap_err(),
                PreparedSafetensorsInventoryError::InvalidMetadata
            );
        }
        let missing_colon =
            br#"{"__metadata__":{"x" "y"},"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#;
        let colon_at = missing_colon
            .windows(b" \"y\"".len())
            .position(|window| window == b" \"y\"")
            .unwrap()
            + 1;
        assert_eq!(
            parse(missing_colon, 0).unwrap_err(),
            PreparedSafetensorsInventoryError::MalformedHeader {
                at: colon_at,
                fault: HeaderSyntaxFault::ExpectedColon,
            }
        );
        let unknown =
            br#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0],"bomb":{"deep":[[[[]]]]}}}"#;
        assert!(matches!(
            parse(unknown, 0),
            Err(PreparedSafetensorsInventoryError::UnknownTensorMember { .. })
        ));
    }

    #[test]
    fn dtype_members_and_unsigned_number_grammar_are_closed() {
        for dtype in ["I64", "F64", "f32", "F8_E4M3", "FUTURE"] {
            let header =
                format!(r#"{{"a":{{"dtype":"{dtype}","shape":[0],"data_offsets":[0,0]}}}}"#);
            assert!(matches!(
                parse(header.as_bytes(), 0),
                Err(PreparedSafetensorsInventoryError::UnsupportedDtype { .. })
            ));
        }

        let scalar = parse(
            br#"{"scalar":{"dtype":"F16","shape":[],"data_offsets":[0,2]},"zero":{"dtype":"BF16","shape":[7,0,9],"data_offsets":[2,2]}}"#,
            2,
        )
        .unwrap();
        assert_eq!(tensor(&scalar, b"scalar").elements(), 1);
        assert_eq!(tensor(&scalar, b"zero").elements(), 0);

        assert_eq!(
            parse(br#"{"a":{"shape":[0],"data_offsets":[0,0]}}"#, 0).unwrap_err(),
            PreparedSafetensorsInventoryError::MissingTensorMember {
                entry: 0,
                member: TensorMember::Dtype,
            }
        );
        assert_eq!(
            parse(
                br#"{"a":{"dtype":"F32","dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
                0,
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::DuplicateTensorMember {
                entry: 0,
                member: TensorMember::Dtype,
            }
        );
        assert_eq!(
            parse(br#"{"a":{"dtype":1,"shape":[0],"data_offsets":[0,0]}}"#, 0,).unwrap_err(),
            PreparedSafetensorsInventoryError::InvalidTensorMember {
                entry: 0,
                member: TensorMember::Dtype,
            }
        );
        assert_malformed(
            br#"{"a":{"dtype":"\q","shape":[0],"data_offsets":[0,0]}}"#,
            0,
            HeaderSyntaxFault::InvalidStringEscape,
        );

        for header in [
            br#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}"#.as_slice(),
            br#"{"a":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}"#.as_slice(),
            br#"{"a":{"dtype":"F32","shape":[1e0],"data_offsets":[0,4]}}"#.as_slice(),
            br#"{"a":{"dtype":"F32","shape":[01],"data_offsets":[0,4]}}"#.as_slice(),
            br#"{"a":{"dtype":"F32","shape":[18446744073709551616],"data_offsets":[0,0]}}"#
                .as_slice(),
        ] {
            assert_malformed(header, 4, HeaderSyntaxFault::InvalidUnsignedInteger);
        }
        for (header, actual) in [
            (
                br#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0]}}"#.as_slice(),
                1,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0,0]}}"#.as_slice(),
                3,
            ),
        ] {
            assert_eq!(
                parse(header, 0).unwrap_err(),
                PreparedSafetensorsInventoryError::InvalidDataOffsetsArity { entry: 0, actual }
            );
        }
    }

    #[test]
    fn local_ranges_require_order_bounds_and_exact_dtype_extent() {
        assert_eq!(
            parse(
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
                4,
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::ReversedDataRange {
                entry: 0,
                start: 4,
                end: 0,
            }
        );
        assert_eq!(
            parse(
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,5]}}"#,
                4,
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::DataRangePastEnd {
                entry: 0,
                end: 5,
                data_len: 4,
            }
        );
        for (header, data_len, expected, actual) in [
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}}"#.as_slice(),
                3,
                4,
                3,
            ),
            (
                br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,3]}}"#.as_slice(),
                3,
                2,
                3,
            ),
        ] {
            assert_eq!(
                parse(header, data_len).unwrap_err(),
                PreparedSafetensorsInventoryError::TensorByteLengthMismatch {
                    entry: 0,
                    expected,
                    actual,
                }
            );
        }
    }

    #[test]
    fn global_layout_is_exact_and_zero_ranges_only_live_at_the_frontier() {
        let valid = br#"{"z0":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"zm":{"dtype":"F16","shape":[0],"data_offsets":[4,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"ze":{"dtype":"BF16","shape":[0],"data_offsets":[8,8]}}"#;
        assert!(parse(valid, 8).is_ok());

        for (header, expected_start, actual_start) in [
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#.as_slice(),
                0,
                1,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[5,9]}}"#.as_slice(),
                4,
                5,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[3,7]}}"#.as_slice(),
                4,
                3,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.as_slice(),
                4,
                0,
            ),
            (
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"z":{"dtype":"F32","shape":[0],"data_offsets":[2,2]}}"#.as_slice(),
                4,
                2,
            ),
        ] {
            assert_eq!(
                parse(header, 9).unwrap_err(),
                PreparedSafetensorsInventoryError::NonContiguousDataRange {
                    expected_start,
                    actual_start,
                }
            );
        }
        assert_eq!(
            parse(
                br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                5,
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::TrailingPayload {
                covered: 4,
                data_len: 5,
            }
        );

        let empty = br#"{"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"b":{"dtype":"F16","shape":[0],"data_offsets":[0,0]}}"#;
        assert!(parse(empty, 0).is_ok());
    }

    #[test]
    fn f32_alignment_is_only_an_absolute_file_offset_fact() {
        let base = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        for spaces in 0..4 {
            let mut header = base.as_bytes().to_vec();
            header.extend(std::iter::repeat_n(b' ', spaces));
            let facts = parse(&header, 4).unwrap();
            let actual = tensor(&facts, b"a").f32_file_offset_alignment();
            let expected = if (8 + header.len()).is_multiple_of(4) {
                Some(F32FileOffsetAlignment::FourByteAligned)
            } else {
                Some(F32FileOffsetAlignment::Unaligned)
            };
            assert_eq!(actual, expected);
        }

        let half = parse(
            br#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#,
            4,
        )
        .unwrap();
        assert_eq!(tensor(&half, b"a").f32_file_offset_alignment(), None);
        assert_eq!(tensor(&half, b"b").f32_file_offset_alignment(), None);

        let mut shifted = br#"{"half":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"float":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}"#.to_vec();
        while !(8 + shifted.len()).is_multiple_of(4) {
            shifted.push(b' ');
        }
        let shifted = parse(&shifted, 6).unwrap();
        assert_eq!(
            tensor(&shifted, b"float").f32_file_offset_alignment(),
            Some(F32FileOffsetAlignment::Unaligned)
        );
    }

    #[test]
    fn every_arena_reservation_is_fallible_and_ordered() {
        let requested = [
            3 * std::mem::size_of::<PreparedSafetensorsTensorRecord>(),
            6,
            3 * std::mem::size_of::<u64>(),
            2 * std::mem::size_of::<DecodedSpan>(),
            12,
        ];
        for (index, arena) in AllocationArena::ALL.into_iter().enumerate() {
            let mut attempts = Vec::new();
            let error = parse_prepared_safetensors_header_inventory_with_reserve_probe(
                CANONICAL.as_bytes(),
                &plan(CANONICAL.len(), 10),
                &wide_limits(),
                &mut |actual, requested_bytes| {
                    attempts.push(actual);
                    if actual == arena {
                        Err(PreparedSafetensorsInventoryError::AllocationFailed {
                            arena: actual,
                            requested_bytes,
                        })
                    } else {
                        Ok(())
                    }
                },
            )
            .unwrap_err();
            assert_eq!(
                error,
                PreparedSafetensorsInventoryError::AllocationFailed {
                    arena,
                    requested_bytes: requested[index],
                }
            );
            assert_eq!(attempts, AllocationArena::ALL[..=index]);
        }
    }

    #[test]
    fn arithmetic_and_second_pass_invariants_fail_closed_with_typed_errors() {
        let unlimited = PreparedSafetensorsInventoryLimits {
            max_tensors: u64::MAX,
            max_tensor_name_bytes: u64::MAX,
            max_rank: u64::MAX,
            max_dimension: u64::MAX,
            max_aggregate_elements: u64::MAX,
            max_metadata_bytes: u64::MAX,
            max_parse_work_bytes: u64::MAX,
        };
        let mut record_overflow = InventoryCensus {
            tensor_count: u64::MAX,
            ..InventoryCensus::default()
        };
        assert_eq!(
            finish_census(0, &unlimited, &mut record_overflow).unwrap_err(),
            PreparedSafetensorsInventoryError::ArithmeticOverflow {
                expression: InventoryExpression::MetadataBytes,
            }
        );
        let mut parse_work_overflow = InventoryCensus {
            metadata_key_count: u64::MAX,
            ..InventoryCensus::default()
        };
        assert_eq!(
            finish_census(0, &unlimited, &mut parse_work_overflow).unwrap_err(),
            PreparedSafetensorsInventoryError::ArithmeticOverflow {
                expression: InventoryExpression::SafetensorsParseWorkBytes,
            }
        );
        assert_eq!(
            checked_product(u64::MAX, 1, InventoryExpression::TensorElements).unwrap(),
            u64::MAX
        );
        assert_eq!(
            checked_product(u64::MAX, 2, InventoryExpression::TensorElements).unwrap_err(),
            PreparedSafetensorsInventoryError::ArithmeticOverflow {
                expression: InventoryExpression::TensorElements,
            }
        );
        assert_eq!(
            checked_product(
                u64::MAX,
                PreparedSafetensorsDtype::F32.width(),
                InventoryExpression::TensorSourceBytes,
            )
            .unwrap_err(),
            PreparedSafetensorsInventoryError::ArithmeticOverflow {
                expression: InventoryExpression::TensorSourceBytes,
            }
        );

        #[cfg(target_pointer_width = "64")]
        {
            let shape_overflow =
                br#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#;
            assert_eq!(
                parse_prepared_safetensors_header_inventory(
                    shape_overflow,
                    &plan(shape_overflow.len(), 0),
                    &unlimited,
                )
                .unwrap_err(),
                PreparedSafetensorsInventoryError::ArithmeticOverflow {
                    expression: InventoryExpression::TensorElements,
                }
            );
            let source_overflow =
                br#"{"a":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,0]}}"#;
            assert_eq!(
                parse_prepared_safetensors_header_inventory(
                    source_overflow,
                    &plan(source_overflow.len(), 0),
                    &unlimited,
                )
                .unwrap_err(),
                PreparedSafetensorsInventoryError::ArithmeticOverflow {
                    expression: InventoryExpression::TensorSourceBytes,
                }
            );
        }

        let expected = InventoryCensus {
            tensor_count: 1,
            ..InventoryCensus::default()
        };
        assert_eq!(
            validate_second_pass(expected, InventoryCensus::default(), true).unwrap_err(),
            PreparedSafetensorsInventoryError::SecondPassCensusMismatch
        );
        assert_eq!(
            validate_second_pass(expected, expected, false).unwrap_err(),
            PreparedSafetensorsInventoryError::SecondPassCensusMismatch
        );
    }
}
