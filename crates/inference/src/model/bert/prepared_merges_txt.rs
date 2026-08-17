//! Dormant prepared-BERT BPE `merges.txt` lexical census.
//!
//! This module mirrors the legacy text-level acceptance and occurrence-rank
//! rules without retaining operands or constructing an effective merge table.
//! It performs no filesystem access or allocation and has no live caller.
//! Unique-pair proof, first-wins effective ranks, vocabulary membership, and
//! tokenizer construction remain deliberately out of scope.

use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertBpeMergesTxtLimitAxis {
    MergesTxtBytes,
    MergeEntries,
    ParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertBpeMergesTxtExpression {
    BaseParseWorkBytes,
    MergeEntryCount,
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

pub(super) fn census_prepared_bert_bpe_merges_txt(
    bytes: &[u8],
    limits: &PreparedBertBpeMergesTxtLimits,
) -> Result<PreparedBertBpeMergesTxtLexicalFacts, PreparedBertBpeMergesTxtError> {
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

    let text =
        std::str::from_utf8(bytes).map_err(|error| PreparedBertBpeMergesTxtError::InvalidUtf8 {
            valid_up_to: error.valid_up_to(),
        })?;

    let mut merge_entry_count = 0_u64;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut operands = line.split_whitespace();
        let Some(_left) = operands.next() else {
            continue;
        };
        let Some(_right) = operands.next() else {
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
        merges_txt_bytes,
        merge_entry_count,
        max_merge_rank: merge_entry_count.checked_sub(1),
        logical_parse_work_bytes,
    })
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
}
