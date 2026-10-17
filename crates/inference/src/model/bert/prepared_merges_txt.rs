//! Dormant prepared-BERT BPE `merges.txt` lexical census.
//!
//! This module mirrors the legacy text-level acceptance and occurrence-rank
//! rules without constructing an effective merge table, with one deliberate
//! divergence: the legacy loader (`tokenizer::bpe::parse_merges_txt`) takes
//! the first two whitespace-separated tokens of a line and silently drops any
//! remainder, so `"a b c"` becomes the pair `("a", "b")` with `c` discarded.
//! This module refuses that line instead, naming it in the error, because a
//! prepared-checkpoint validator's job is to reject a malformed input the
//! legacy path would have silently misread rather than reproduce the
//! misreading. The lexical census is allocation-free; a separate transient
//! capability can fallibly retain only borrowed operand spans under explicit
//! scratch and two-pass-work limits. The module performs no filesystem access
//! and has no live caller. Unique-pair proof, first-wins effective ranks,
//! vocabulary membership, and tokenizer construction remain deliberately out
//! of scope.

use std::mem::size_of;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertBpeMergesTxtLimitAxis {
    MergesTxtBytes,
    MergeEntries,
    ParseWorkBytes,
    RetainedPairSpanBytes,
    TotalParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertBpeMergesTxtExpression {
    BaseParseWorkBytes,
    MergeEntryCount,
    MergeLineNumber,
    RetainedPairSpanBytes,
    TotalParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertBpeMergesTxtAllocationArena {
    MergePairSpans,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertBpeMergesTxtError {
    PlatformUnrepresentable {
        axis: PreparedBertBpeMergesTxtLimitAxis,
        value: u64,
    },
    Exceeded {
        axis: PreparedBertBpeMergesTxtLimitAxis,
        actual: u64,
        limit: u64,
    },
    ArithmeticOverflow(PreparedBertBpeMergesTxtExpression),
    InvalidUtf8 {
        valid_up_to: usize,
    },
    /// A line carries more than two whitespace-separated tokens. The legacy
    /// loader takes the first two and silently drops the rest; this module
    /// refuses the line instead, naming its 1-based line number.
    MalformedMergeLine {
        line: u64,
    },
    AllocationFailed {
        arena: PreparedBertBpeMergesTxtAllocationArena,
        requested_bytes: u64,
    },
    SecondPassCensusMismatch {
        expected: u64,
        actual: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertBpeMergesTxtLimits {
    max_merges_txt_bytes: NonZeroU64,
    max_merge_entries: NonZeroU64,
    max_parse_work_bytes: NonZeroU64,
}

impl PreparedBertBpeMergesTxtLimits {
    pub(super) fn try_new(
        max_merges_txt_bytes: NonZeroU64,
        max_merge_entries: NonZeroU64,
        max_parse_work_bytes: NonZeroU64,
    ) -> Result<Self, PreparedBertBpeMergesTxtError> {
        let platform_span_max = (usize::MAX as u64).min(isize::MAX as u64);
        Self::try_new_with_platform_max(
            max_merges_txt_bytes,
            max_merge_entries,
            max_parse_work_bytes,
            platform_span_max,
        )
    }

    fn try_new_with_platform_max(
        max_merges_txt_bytes: NonZeroU64,
        max_merge_entries: NonZeroU64,
        max_parse_work_bytes: NonZeroU64,
        platform_span_max: u64,
    ) -> Result<Self, PreparedBertBpeMergesTxtError> {
        for (axis, value) in [
            (
                PreparedBertBpeMergesTxtLimitAxis::MergesTxtBytes,
                max_merges_txt_bytes.get(),
            ),
            (
                PreparedBertBpeMergesTxtLimitAxis::MergeEntries,
                max_merge_entries.get(),
            ),
            (
                PreparedBertBpeMergesTxtLimitAxis::ParseWorkBytes,
                max_parse_work_bytes.get(),
            ),
        ] {
            if value > platform_span_max {
                return Err(PreparedBertBpeMergesTxtError::PlatformUnrepresentable { axis, value });
            }
        }
        Ok(Self {
            max_merges_txt_bytes,
            max_merge_entries,
            max_parse_work_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertBpeRetainedMergePairLimits {
    lexical: PreparedBertBpeMergesTxtLimits,
    max_retained_pair_span_bytes: NonZeroU64,
    max_total_parse_work_bytes: NonZeroU64,
}

impl PreparedBertBpeRetainedMergePairLimits {
    pub(super) fn try_new(
        lexical: PreparedBertBpeMergesTxtLimits,
        max_retained_pair_span_bytes: NonZeroU64,
        max_total_parse_work_bytes: NonZeroU64,
    ) -> Result<Self, PreparedBertBpeMergesTxtError> {
        let platform_span_max = (usize::MAX as u64).min(isize::MAX as u64);
        Self::try_new_with_platform_max(
            lexical,
            max_retained_pair_span_bytes,
            max_total_parse_work_bytes,
            platform_span_max,
        )
    }

    fn try_new_with_platform_max(
        lexical: PreparedBertBpeMergesTxtLimits,
        max_retained_pair_span_bytes: NonZeroU64,
        max_total_parse_work_bytes: NonZeroU64,
        platform_span_max: u64,
    ) -> Result<Self, PreparedBertBpeMergesTxtError> {
        for (axis, value) in [
            (
                PreparedBertBpeMergesTxtLimitAxis::RetainedPairSpanBytes,
                max_retained_pair_span_bytes.get(),
            ),
            (
                PreparedBertBpeMergesTxtLimitAxis::TotalParseWorkBytes,
                max_total_parse_work_bytes.get(),
            ),
        ] {
            if value > platform_span_max {
                return Err(PreparedBertBpeMergesTxtError::PlatformUnrepresentable { axis, value });
            }
        }
        Ok(Self {
            lexical,
            max_retained_pair_span_bytes,
            max_total_parse_work_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertBpeMergesTxtLexicalFacts {
    merges_txt_bytes: u64,
    merge_entry_count: u64,
    max_merge_rank: Option<u64>,
    logical_parse_work_bytes: u64,
}

impl PreparedBertBpeMergesTxtLexicalFacts {
    pub(super) fn merges_txt_bytes(self) -> u64 {
        self.merges_txt_bytes
    }

    pub(super) fn merge_entry_count(self) -> u64 {
        self.merge_entry_count
    }

    pub(super) fn max_merge_rank(self) -> Option<u64> {
        self.max_merge_rank
    }

    pub(super) fn logical_parse_work_bytes(self) -> u64 {
        self.logical_parse_work_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertBpeMergePairSpan<'a> {
    left: &'a str,
    right: &'a str,
    source_rank: usize,
}

impl<'a> PreparedBertBpeMergePairSpan<'a> {
    pub(super) fn left(self) -> &'a str {
        self.left
    }

    pub(super) fn right(self) -> &'a str {
        self.right
    }

    pub(super) fn source_rank(self) -> usize {
        self.source_rank
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct PreparedBertBpeRetainedMergePairFacts<'a> {
    lexical: PreparedBertBpeMergesTxtLexicalFacts,
    pairs: Vec<PreparedBertBpeMergePairSpan<'a>>,
    retained_pair_span_bytes: u64,
    total_logical_parse_work_bytes: u64,
}

impl<'a> PreparedBertBpeRetainedMergePairFacts<'a> {
    pub(super) fn lexical(&self) -> PreparedBertBpeMergesTxtLexicalFacts {
        self.lexical
    }

    pub(super) fn pairs(&self) -> &[PreparedBertBpeMergePairSpan<'a>] {
        &self.pairs
    }

    pub(super) fn retained_pair_span_bytes(&self) -> u64 {
        self.retained_pair_span_bytes
    }

    pub(super) fn total_logical_parse_work_bytes(&self) -> u64 {
        self.total_logical_parse_work_bytes
    }
}

#[derive(Clone, Copy)]
enum MergePairReserveMode {
    Actual,
    #[cfg(test)]
    Fail,
}

pub(super) fn census_prepared_bert_bpe_merges_txt(
    bytes: &[u8],
    limits: &PreparedBertBpeMergesTxtLimits,
) -> Result<PreparedBertBpeMergesTxtLexicalFacts, PreparedBertBpeMergesTxtError> {
    let preflight = preflight_merges_txt(bytes, limits)?;
    census_prepared_bert_bpe_merges_txt_after_preflight(bytes, limits, preflight)
}

pub(super) fn retain_prepared_bert_bpe_merges_txt_pairs<'a>(
    bytes: &'a [u8],
    limits: &PreparedBertBpeRetainedMergePairLimits,
) -> Result<PreparedBertBpeRetainedMergePairFacts<'a>, PreparedBertBpeMergesTxtError> {
    retain_prepared_bert_bpe_merges_txt_pairs_with_reserve(
        bytes,
        limits,
        MergePairReserveMode::Actual,
    )
}

fn retain_prepared_bert_bpe_merges_txt_pairs_with_reserve<'a>(
    bytes: &'a [u8],
    limits: &PreparedBertBpeRetainedMergePairLimits,
    reserve_mode: MergePairReserveMode,
) -> Result<PreparedBertBpeRetainedMergePairFacts<'a>, PreparedBertBpeMergesTxtError> {
    let preflight = preflight_merges_txt(bytes, &limits.lexical)?;
    let total_logical_parse_work_bytes = preflight.logical_parse_work_bytes.checked_mul(2).ok_or(
        PreparedBertBpeMergesTxtError::ArithmeticOverflow(
            PreparedBertBpeMergesTxtExpression::TotalParseWorkBytes,
        ),
    )?;
    enforce_limit(
        PreparedBertBpeMergesTxtLimitAxis::TotalParseWorkBytes,
        total_logical_parse_work_bytes,
        limits.max_total_parse_work_bytes.get(),
    )?;

    let lexical =
        census_prepared_bert_bpe_merges_txt_after_preflight(bytes, &limits.lexical, preflight)?;
    let expected_count = usize::try_from(lexical.merge_entry_count).map_err(|_| {
        PreparedBertBpeMergesTxtError::PlatformUnrepresentable {
            axis: PreparedBertBpeMergesTxtLimitAxis::MergeEntries,
            value: lexical.merge_entry_count,
        }
    })?;
    let span_size = u64::try_from(size_of::<PreparedBertBpeMergePairSpan<'_>>()).map_err(|_| {
        PreparedBertBpeMergesTxtError::ArithmeticOverflow(
            PreparedBertBpeMergesTxtExpression::RetainedPairSpanBytes,
        )
    })?;
    let requested_bytes = lexical.merge_entry_count.checked_mul(span_size).ok_or(
        PreparedBertBpeMergesTxtError::ArithmeticOverflow(
            PreparedBertBpeMergesTxtExpression::RetainedPairSpanBytes,
        ),
    )?;
    enforce_limit(
        PreparedBertBpeMergesTxtLimitAxis::RetainedPairSpanBytes,
        requested_bytes,
        limits.max_retained_pair_span_bytes.get(),
    )?;

    let mut pairs = Vec::new();
    match reserve_mode {
        MergePairReserveMode::Actual => pairs.try_reserve_exact(expected_count).map_err(|_| {
            PreparedBertBpeMergesTxtError::AllocationFailed {
                arena: PreparedBertBpeMergesTxtAllocationArena::MergePairSpans,
                requested_bytes,
            }
        })?,
        #[cfg(test)]
        MergePairReserveMode::Fail => {
            return Err(PreparedBertBpeMergesTxtError::AllocationFailed {
                arena: PreparedBertBpeMergesTxtAllocationArena::MergePairSpans,
                requested_bytes,
            });
        }
    }
    let retained_pair_span_bytes = u64::try_from(pairs.capacity())
        .ok()
        .and_then(|capacity| capacity.checked_mul(span_size))
        .ok_or(PreparedBertBpeMergesTxtError::ArithmeticOverflow(
            PreparedBertBpeMergesTxtExpression::RetainedPairSpanBytes,
        ))?;
    enforce_limit(
        PreparedBertBpeMergesTxtLimitAxis::RetainedPairSpanBytes,
        retained_pair_span_bytes,
        limits.max_retained_pair_span_bytes.get(),
    )?;

    let text =
        std::str::from_utf8(bytes).map_err(|error| PreparedBertBpeMergesTxtError::InvalidUtf8 {
            valid_up_to: error.valid_up_to(),
        })?;
    for (index, line) in text.lines().enumerate() {
        let (left, right) = match legacy_merge_pair(line) {
            LegacyMergeLine::Skip => continue,
            LegacyMergeLine::Malformed => {
                return Err(PreparedBertBpeMergesTxtError::MalformedMergeLine {
                    line: legacy_merge_line_number(index)?,
                });
            }
            LegacyMergeLine::Pair(left, right) => (left, right),
        };
        let actual = u64::try_from(pairs.len())
            .ok()
            .and_then(|count| count.checked_add(1))
            .ok_or(PreparedBertBpeMergesTxtError::ArithmeticOverflow(
                PreparedBertBpeMergesTxtExpression::MergeEntryCount,
            ))?;
        if actual > lexical.merge_entry_count || pairs.len() >= pairs.capacity() {
            return Err(PreparedBertBpeMergesTxtError::SecondPassCensusMismatch {
                expected: lexical.merge_entry_count,
                actual,
            });
        }
        let source_rank = pairs.len();
        pairs.push(PreparedBertBpeMergePairSpan {
            left,
            right,
            source_rank,
        });
    }
    let actual = u64::try_from(pairs.len()).map_err(|_| {
        PreparedBertBpeMergesTxtError::ArithmeticOverflow(
            PreparedBertBpeMergesTxtExpression::MergeEntryCount,
        )
    })?;
    if actual != lexical.merge_entry_count {
        return Err(PreparedBertBpeMergesTxtError::SecondPassCensusMismatch {
            expected: lexical.merge_entry_count,
            actual,
        });
    }

    Ok(PreparedBertBpeRetainedMergePairFacts {
        lexical,
        pairs,
        retained_pair_span_bytes,
        total_logical_parse_work_bytes,
    })
}

#[derive(Clone, Copy)]
struct MergesTxtPreflight {
    merges_txt_bytes: u64,
    logical_parse_work_bytes: u64,
}

fn preflight_merges_txt(
    bytes: &[u8],
    limits: &PreparedBertBpeMergesTxtLimits,
) -> Result<MergesTxtPreflight, PreparedBertBpeMergesTxtError> {
    let merges_txt_bytes = u64::try_from(bytes.len()).map_err(|_| {
        PreparedBertBpeMergesTxtError::PlatformUnrepresentable {
            axis: PreparedBertBpeMergesTxtLimitAxis::MergesTxtBytes,
            value: u64::MAX,
        }
    })?;
    enforce_limit(
        PreparedBertBpeMergesTxtLimitAxis::MergesTxtBytes,
        merges_txt_bytes,
        limits.max_merges_txt_bytes.get(),
    )?;

    // Reserve a fixed conservative eight complete input spans before parsing:
    // UTF-8 validation, line scanning, Unicode trim from both ends, comment
    // classification, two token scans, and per-line accounting.
    let logical_parse_work_bytes = checked_base_parse_work(merges_txt_bytes)?;
    enforce_limit(
        PreparedBertBpeMergesTxtLimitAxis::ParseWorkBytes,
        logical_parse_work_bytes,
        limits.max_parse_work_bytes.get(),
    )?;

    Ok(MergesTxtPreflight {
        merges_txt_bytes,
        logical_parse_work_bytes,
    })
}

fn census_prepared_bert_bpe_merges_txt_after_preflight(
    bytes: &[u8],
    limits: &PreparedBertBpeMergesTxtLimits,
    preflight: MergesTxtPreflight,
) -> Result<PreparedBertBpeMergesTxtLexicalFacts, PreparedBertBpeMergesTxtError> {
    let text =
        std::str::from_utf8(bytes).map_err(|error| PreparedBertBpeMergesTxtError::InvalidUtf8 {
            valid_up_to: error.valid_up_to(),
        })?;

    let mut merge_entry_count = 0_u64;
    for (index, line) in text.lines().enumerate() {
        match legacy_merge_pair(line) {
            LegacyMergeLine::Skip => continue,
            LegacyMergeLine::Malformed => {
                return Err(PreparedBertBpeMergesTxtError::MalformedMergeLine {
                    line: legacy_merge_line_number(index)?,
                });
            }
            LegacyMergeLine::Pair(_left, _right) => {}
        }
        // Duplicate pairs deliberately remain separate occurrences. The live
        // constructor assigns their raw ranks before applying first-wins map
        // semantics, which belongs to a later retained-table slice.
        merge_entry_count = merge_entry_count.checked_add(1).ok_or(
            PreparedBertBpeMergesTxtError::ArithmeticOverflow(
                PreparedBertBpeMergesTxtExpression::MergeEntryCount,
            ),
        )?;
        enforce_limit(
            PreparedBertBpeMergesTxtLimitAxis::MergeEntries,
            merge_entry_count,
            limits.max_merge_entries.get(),
        )?;
    }

    Ok(PreparedBertBpeMergesTxtLexicalFacts {
        merges_txt_bytes: preflight.merges_txt_bytes,
        merge_entry_count,
        max_merge_rank: merge_entry_count.checked_sub(1),
        logical_parse_work_bytes: preflight.logical_parse_work_bytes,
    })
}

enum LegacyMergeLine<'a> {
    /// Blank or `#`-comment line, or a line with fewer than two operands. The
    /// legacy loader forms no pair from it either, so nothing is discarded.
    Skip,
    Pair(&'a str, &'a str),
    /// More than two whitespace-separated tokens: the legacy loader would
    /// silently drop everything past the second one.
    Malformed,
}

fn legacy_merge_pair(line: &str) -> LegacyMergeLine<'_> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return LegacyMergeLine::Skip;
    }
    let mut operands = line.split_whitespace();
    let (Some(left), Some(right)) = (operands.next(), operands.next()) else {
        return LegacyMergeLine::Skip;
    };
    if operands.next().is_some() {
        return LegacyMergeLine::Malformed;
    }
    LegacyMergeLine::Pair(left, right)
}

/// 1-based line number for a `str::lines()` index, for the error a caller
/// sees; the merges-text byte count is already bounded well under `u64::MAX`
/// by `preflight_merges_txt`, so this only fails on a platform where `usize`
/// itself cannot fit in `u64`.
fn legacy_merge_line_number(index: usize) -> Result<u64, PreparedBertBpeMergesTxtError> {
    u64::try_from(index)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or(PreparedBertBpeMergesTxtError::ArithmeticOverflow(
            PreparedBertBpeMergesTxtExpression::MergeLineNumber,
        ))
}

fn checked_base_parse_work(merges_txt_bytes: u64) -> Result<u64, PreparedBertBpeMergesTxtError> {
    merges_txt_bytes
        .checked_mul(8)
        .ok_or(PreparedBertBpeMergesTxtError::ArithmeticOverflow(
            PreparedBertBpeMergesTxtExpression::BaseParseWorkBytes,
        ))
}

fn enforce_limit(
    axis: PreparedBertBpeMergesTxtLimitAxis,
    actual: u64,
    limit: u64,
) -> Result<(), PreparedBertBpeMergesTxtError> {
    if actual > limit {
        return Err(PreparedBertBpeMergesTxtError::Exceeded {
            axis,
            actual,
            limit,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
