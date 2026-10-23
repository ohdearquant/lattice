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
    TrailingNonWhitespace,
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

    // The reference reader parses this buffer as JSON (the `safetensors`
    // crate calls `serde_json::from_str`), whose own end-of-input check
    // (`Deserializer::end`) skips exactly the space/tab/CR/LF class after the
    // final `}` before requiring EOF — the same class `skip_ws` already uses
    // for every interior gap in this grammar. Scanning only space here was an
    // inconsistency with the rest of this parser, not a narrower contract:
    // a tab or newline after a valid header was rejected here while the
    // reference, and this parser's own interior grammar, both accept it.
    scanner.skip_ws();
    if scanner.peek().is_some() {
        return Err(scanner.error(HeaderSyntaxFault::TrailingNonWhitespace));
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
mod tests;
