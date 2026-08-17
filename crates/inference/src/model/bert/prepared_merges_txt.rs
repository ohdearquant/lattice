//! Dormant prepared-BERT BPE `merges.txt` lexical census.
//!
//! This module mirrors the legacy text-level acceptance and occurrence-rank
//! rules without constructing an effective merge table. The lexical census is
//! allocation-free; a separate transient capability can fallibly retain only
//! borrowed operand spans under explicit scratch and two-pass-work limits. The
//! module performs no filesystem access and has no live caller. Unique-pair
//! proof, first-wins effective ranks, vocabulary membership, and tokenizer
//! construction remain deliberately out of scope.

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
    for line in text.lines() {
        let Some((left, right)) = legacy_merge_pair(line) else {
            continue;
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
    for line in text.lines() {
        let Some((_left, _right)) = legacy_merge_pair(line) else {
            continue;
        };
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

fn legacy_merge_pair(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let mut operands = line.split_whitespace();
    Some((operands.next()?, operands.next()?))
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
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn limits(bytes: u64, entries: u64, work: u64) -> PreparedBertBpeMergesTxtLimits {
        PreparedBertBpeMergesTxtLimits::try_new(nz(bytes), nz(entries), nz(work)).unwrap()
    }

    #[test]
    fn census_matches_legacy_unicode_line_and_rank_semantics() {
        let source = concat!(
            "\u{2003}#version: 0.2\r\n",
            "a\u{00a0}b ignored\n",
            "single\n",
            "a b\n",
            "c\u{2028}d\n",
        );
        let bytes = u64::try_from(source.len()).unwrap();
        let facts =
            census_prepared_bert_bpe_merges_txt(source.as_bytes(), &limits(bytes, 3, bytes * 8))
                .unwrap();

        assert_eq!(facts.merges_txt_bytes(), bytes);
        assert_eq!(facts.merge_entry_count(), 3);
        assert_eq!(facts.max_merge_rank(), Some(2));
        assert_eq!(facts.logical_parse_work_bytes(), bytes * 8);

        // Bare CR is Unicode whitespace, not a `str::lines()` delimiter.
        let bare_cr = census_prepared_bert_bpe_merges_txt(b"a\rb", &limits(3, 1, 24)).unwrap();
        assert_eq!(bare_cr.merge_entry_count(), 1);

        // U+200B ZERO WIDTH SPACE is not Rust Unicode whitespace.
        let zero_width = "a\u{200b}b";
        let zero_width_bytes = u64::try_from(zero_width.len()).unwrap();
        let zero_width_facts = census_prepared_bert_bpe_merges_txt(
            zero_width.as_bytes(),
            &limits(zero_width_bytes, 1, zero_width_bytes * 8),
        )
        .unwrap();
        assert_eq!(zero_width_facts.merge_entry_count(), 0);
        assert_eq!(zero_width_facts.max_merge_rank(), None);
    }

    #[test]
    fn duplicates_consume_ranks_and_empty_or_incomplete_input_is_valid() {
        let duplicate = b"a b\na b\n";
        let facts = census_prepared_bert_bpe_merges_txt(duplicate, &limits(10, 2, 80)).unwrap();
        assert_eq!(facts.merge_entry_count(), 2);
        assert_eq!(facts.max_merge_rank(), Some(1));

        for source in [b"".as_slice(), b"\n# comment\nsingle\n".as_slice()] {
            let byte_count = u64::try_from(source.len()).unwrap();
            let facts = census_prepared_bert_bpe_merges_txt(
                source,
                &limits(byte_count.max(1), 1, byte_count.saturating_mul(8).max(1)),
            )
            .unwrap();
            assert_eq!(facts.merge_entry_count(), 0);
            assert_eq!(facts.max_merge_rank(), None);
        }
    }

    #[test]
    fn byte_entry_work_and_platform_limits_are_typed() {
        let source = b"a b";
        let exact = census_prepared_bert_bpe_merges_txt(source, &limits(3, 1, 24)).unwrap();
        assert_eq!(exact.logical_parse_work_bytes(), 24);

        assert_eq!(
            census_prepared_bert_bpe_merges_txt(source, &limits(2, 1, 24)),
            Err(PreparedBertBpeMergesTxtError::Exceeded {
                axis: PreparedBertBpeMergesTxtLimitAxis::MergesTxtBytes,
                actual: 3,
                limit: 2,
            })
        );
        assert_eq!(
            census_prepared_bert_bpe_merges_txt(b"a b\nc d", &limits(7, 1, 56)),
            Err(PreparedBertBpeMergesTxtError::Exceeded {
                axis: PreparedBertBpeMergesTxtLimitAxis::MergeEntries,
                actual: 2,
                limit: 1,
            })
        );
        assert_eq!(
            census_prepared_bert_bpe_merges_txt(source, &limits(3, 1, 23)),
            Err(PreparedBertBpeMergesTxtError::Exceeded {
                axis: PreparedBertBpeMergesTxtLimitAxis::ParseWorkBytes,
                actual: 24,
                limit: 23,
            })
        );

        assert!(
            PreparedBertBpeMergesTxtLimits::try_new_with_platform_max(nz(8), nz(8), nz(8), 8,)
                .is_ok()
        );
        assert_eq!(
            PreparedBertBpeMergesTxtLimits::try_new_with_platform_max(nz(9), nz(8), nz(8), 8,),
            Err(PreparedBertBpeMergesTxtError::PlatformUnrepresentable {
                axis: PreparedBertBpeMergesTxtLimitAxis::MergesTxtBytes,
                value: 9,
            })
        );
    }

    #[test]
    fn invalid_utf8_and_work_overflow_are_typed() {
        assert_eq!(
            census_prepared_bert_bpe_merges_txt(&[0xff], &limits(1, 1, 8)),
            Err(PreparedBertBpeMergesTxtError::InvalidUtf8 { valid_up_to: 0 })
        );
        assert_eq!(
            checked_base_parse_work(u64::MAX),
            Err(PreparedBertBpeMergesTxtError::ArithmeticOverflow(
                PreparedBertBpeMergesTxtExpression::BaseParseWorkBytes,
            ))
        );
    }

    fn retained_limits(
        bytes: u64,
        entries: u64,
        span_bytes: u64,
        total_work: u64,
    ) -> PreparedBertBpeRetainedMergePairLimits {
        let lexical = limits(bytes.max(1), entries.max(1), bytes.saturating_mul(8).max(1));
        PreparedBertBpeRetainedMergePairLimits::try_new(
            lexical,
            nz(span_bytes.max(1)),
            nz(total_work.max(1)),
        )
        .unwrap()
    }

    #[test]
    fn retained_pairs_preserve_exact_operands_occurrence_ranks_and_source_borrows() {
        let source = concat!(
            "\u{2003}# header\r\n",
            "a\u{00a0}b ignored\n",
            "single\n",
            "a b\n",
            "c\u{2028}d\n",
        );
        let bytes = u64::try_from(source.len()).unwrap();
        let retained = retain_prepared_bert_bpe_merges_txt_pairs(
            source.as_bytes(),
            &retained_limits(bytes, 3, 1024, bytes * 16),
        )
        .unwrap();
        let observed: Vec<_> = retained
            .pairs()
            .iter()
            .copied()
            .map(|pair| (pair.left(), pair.right(), pair.source_rank()))
            .collect();
        assert_eq!(observed, [("a", "b", 0), ("a", "b", 1), ("c", "d", 2)]);
        assert_eq!(retained.lexical().merge_entry_count(), 3);
        assert_eq!(retained.total_logical_parse_work_bytes(), bytes * 16);

        let source_start = source.as_ptr() as usize;
        let source_end = source_start + source.len();
        for pair in retained.pairs() {
            for operand in [pair.left(), pair.right()] {
                let operand_start = operand.as_ptr() as usize;
                assert!(operand_start >= source_start);
                assert!(operand_start + operand.len() <= source_end);
            }
        }
    }

    #[test]
    fn duplicate_pairs_remain_distinct_for_later_first_wins_materialization() {
        let source = b"a b\nc d\na b";
        let bytes = u64::try_from(source.len()).unwrap();
        let retained = retain_prepared_bert_bpe_merges_txt_pairs(
            source,
            &retained_limits(bytes, 3, 1024, bytes * 16),
        )
        .unwrap();
        assert_eq!(retained.pairs().len(), 3);
        assert_eq!(retained.pairs()[0].source_rank(), 0);
        assert_eq!(retained.pairs()[2].source_rank(), 2);
        assert_eq!(retained.pairs()[0].left(), retained.pairs()[2].left());
        assert_eq!(retained.pairs()[0].right(), retained.pairs()[2].right());
    }

    #[test]
    fn total_work_actual_span_capacity_empty_and_reserve_failures_are_bounded() {
        let source = b"a b\nc d";
        let bytes = u64::try_from(source.len()).unwrap();
        let broad = retain_prepared_bert_bpe_merges_txt_pairs(
            source,
            &retained_limits(bytes, 2, 1024, bytes * 16),
        )
        .unwrap();
        let actual_span_bytes = broad.retained_pair_span_bytes();
        assert!(actual_span_bytes > 0);

        let exact = retained_limits(bytes, 2, actual_span_bytes, bytes * 16);
        assert!(retain_prepared_bert_bpe_merges_txt_pairs(source, &exact).is_ok());
        assert_eq!(
            retain_prepared_bert_bpe_merges_txt_pairs(
                source,
                &retained_limits(bytes, 2, actual_span_bytes, bytes * 16 - 1),
            ),
            Err(PreparedBertBpeMergesTxtError::Exceeded {
                axis: PreparedBertBpeMergesTxtLimitAxis::TotalParseWorkBytes,
                actual: bytes * 16,
                limit: bytes * 16 - 1,
            })
        );
        assert_eq!(
            retain_prepared_bert_bpe_merges_txt_pairs(
                source,
                &retained_limits(bytes, 2, actual_span_bytes - 1, bytes * 16),
            ),
            Err(PreparedBertBpeMergesTxtError::Exceeded {
                axis: PreparedBertBpeMergesTxtLimitAxis::RetainedPairSpanBytes,
                actual: actual_span_bytes,
                limit: actual_span_bytes - 1,
            })
        );

        let requested_bytes =
            2 * u64::try_from(std::mem::size_of::<PreparedBertBpeMergePairSpan<'_>>()).unwrap();
        assert_eq!(
            retain_prepared_bert_bpe_merges_txt_pairs_with_reserve(
                source,
                &exact,
                MergePairReserveMode::Fail,
            ),
            Err(PreparedBertBpeMergesTxtError::AllocationFailed {
                arena: PreparedBertBpeMergesTxtAllocationArena::MergePairSpans,
                requested_bytes,
            })
        );

        let empty =
            retain_prepared_bert_bpe_merges_txt_pairs(b"", &retained_limits(1, 1, 1, 1)).unwrap();
        assert!(empty.pairs().is_empty());
        assert_eq!(empty.retained_pair_span_bytes(), 0);
        assert_eq!(empty.total_logical_parse_work_bytes(), 0);
    }
}
