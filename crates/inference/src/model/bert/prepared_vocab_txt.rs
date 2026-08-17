//! Dormant, layout-neutral prepared-BERT `vocab.txt` facts.
//!
//! This module mirrors only the line-to-ID substrate shared by the legacy
//! WordPiece `vocab.txt` and BPE `vocab.txt` + `merges.txt` loaders. It does
//! not validate special tokens, merges, emitted IDs, or any live file.

use std::cmp::Ordering;
use std::mem::size_of;
use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertVocabTxtLimitAxis {
    VocabTxtBytes,
    VocabEntries,
    SpanScratchBytes,
    ParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertVocabTxtExpression {
    BaseParseWorkBytes,
    SpanScratchBytes,
    EntryCount,
    HeapIndex,
    ParseWorkBytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertVocabTxtAllocationArena {
    TokenSpans,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedBertVocabTxtError {
    PlatformUnrepresentable {
        axis: PreparedBertVocabTxtLimitAxis,
        value: u64,
    },
    Exceeded {
        axis: PreparedBertVocabTxtLimitAxis,
        actual: u64,
        limit: u64,
    },
    ArithmeticOverflow(PreparedBertVocabTxtExpression),
    InvalidUtf8 {
        valid_up_to: usize,
    },
    EmptyVocabulary,
    TokenIdUnrepresentable {
        cardinality: u64,
    },
    DuplicateToken {
        first_id: u32,
        duplicate_id: u32,
    },
    AllocationFailed {
        arena: PreparedBertVocabTxtAllocationArena,
        requested_bytes: u64,
    },
    SecondPassCensusMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertVocabTxtLimits {
    max_vocab_txt_bytes: NonZeroU64,
    max_vocab_entries: NonZeroU64,
    max_span_scratch_bytes: NonZeroU64,
    max_parse_work_bytes: NonZeroU64,
}

impl PreparedBertVocabTxtLimits {
    pub(super) fn try_new(
        max_vocab_txt_bytes: NonZeroU64,
        max_vocab_entries: NonZeroU64,
        max_span_scratch_bytes: NonZeroU64,
        max_parse_work_bytes: NonZeroU64,
    ) -> Result<Self, PreparedBertVocabTxtError> {
        let platform_span_max = (usize::MAX as u64).min(isize::MAX as u64);
        Self::try_new_with_platform_max(
            max_vocab_txt_bytes,
            max_vocab_entries,
            max_span_scratch_bytes,
            max_parse_work_bytes,
            platform_span_max,
        )
    }

    fn try_new_with_platform_max(
        max_vocab_txt_bytes: NonZeroU64,
        max_vocab_entries: NonZeroU64,
        max_span_scratch_bytes: NonZeroU64,
        max_parse_work_bytes: NonZeroU64,
        platform_span_max: u64,
    ) -> Result<Self, PreparedBertVocabTxtError> {
        for (axis, value) in [
            (
                PreparedBertVocabTxtLimitAxis::VocabTxtBytes,
                max_vocab_txt_bytes.get(),
            ),
            (
                PreparedBertVocabTxtLimitAxis::VocabEntries,
                max_vocab_entries.get(),
            ),
            (
                PreparedBertVocabTxtLimitAxis::SpanScratchBytes,
                max_span_scratch_bytes.get(),
            ),
            (
                PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
                max_parse_work_bytes.get(),
            ),
        ] {
            if value > platform_span_max {
                return Err(PreparedBertVocabTxtError::PlatformUnrepresentable { axis, value });
            }
        }
        let max_dense_entries = u64::from(u32::MAX) + 1;
        if max_vocab_entries.get() > max_dense_entries {
            return Err(PreparedBertVocabTxtError::TokenIdUnrepresentable {
                cardinality: max_vocab_entries.get(),
            });
        }
        Ok(Self {
            max_vocab_txt_bytes,
            max_vocab_entries,
            max_span_scratch_bytes,
            max_parse_work_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PreparedBertVocabTxtFacts {
    vocab_txt_bytes: u64,
    vocabulary_cardinality: NonZeroU64,
    max_token_id: u32,
    span_scratch_bytes: u64,
    logical_parse_work_bytes: u64,
}

impl PreparedBertVocabTxtFacts {
    pub(super) fn vocab_txt_bytes(self) -> u64 {
        self.vocab_txt_bytes
    }

    pub(super) fn vocabulary_cardinality(self) -> NonZeroU64 {
        self.vocabulary_cardinality
    }

    pub(super) fn max_token_id(self) -> u32 {
        self.max_token_id
    }

    pub(super) fn span_scratch_bytes(self) -> u64 {
        self.span_scratch_bytes
    }

    pub(super) fn logical_parse_work_bytes(self) -> u64 {
        self.logical_parse_work_bytes
    }
}

#[derive(Clone, Copy)]
struct TokenSpan<'a> {
    token: &'a [u8],
    id: u32,
}

struct WorkMeter {
    used: u64,
    limit: u64,
}

impl WorkMeter {
    fn with_base(base: u64, limit: u64) -> Result<Self, PreparedBertVocabTxtError> {
        if base > limit {
            return Err(PreparedBertVocabTxtError::Exceeded {
                axis: PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
                actual: base,
                limit,
            });
        }
        Ok(Self { used: base, limit })
    }

    fn charge(&mut self, amount: u64) -> Result<(), PreparedBertVocabTxtError> {
        let next =
            self.used
                .checked_add(amount)
                .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                    PreparedBertVocabTxtExpression::ParseWorkBytes,
                ))?;
        if next > self.limit {
            return Err(PreparedBertVocabTxtError::Exceeded {
                axis: PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
                actual: next,
                limit: self.limit,
            });
        }
        self.used = next;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ReserveMode {
    Actual,
    #[cfg(test)]
    Fail,
}

pub(super) fn parse_prepared_bert_vocab_txt(
    bytes: &[u8],
    limits: &PreparedBertVocabTxtLimits,
) -> Result<PreparedBertVocabTxtFacts, PreparedBertVocabTxtError> {
    parse_prepared_bert_vocab_txt_with_reserve(bytes, limits, ReserveMode::Actual)
}

fn parse_prepared_bert_vocab_txt_with_reserve(
    bytes: &[u8],
    limits: &PreparedBertVocabTxtLimits,
    reserve_mode: ReserveMode,
) -> Result<PreparedBertVocabTxtFacts, PreparedBertVocabTxtError> {
    let vocab_txt_bytes = u64::try_from(bytes.len()).map_err(|_| {
        PreparedBertVocabTxtError::PlatformUnrepresentable {
            axis: PreparedBertVocabTxtLimitAxis::VocabTxtBytes,
            value: u64::MAX,
        }
    })?;
    enforce_limit(
        PreparedBertVocabTxtLimitAxis::VocabTxtBytes,
        vocab_txt_bytes,
        limits.max_vocab_txt_bytes.get(),
    )?;
    // Charge one complete input span for UTF-8 validation, one for each of the
    // two line scans, and one for worst-case trailing-CR normalization. The
    // data-dependent duplicate proof adds its own charge below.
    let base_work =
        vocab_txt_bytes
            .checked_mul(4)
            .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                PreparedBertVocabTxtExpression::BaseParseWorkBytes,
            ))?;
    let mut work = WorkMeter::with_base(base_work, limits.max_parse_work_bytes.get())?;
    let text =
        std::str::from_utf8(bytes).map_err(|error| PreparedBertVocabTxtError::InvalidUtf8 {
            valid_up_to: error.valid_up_to(),
        })?;

    let mut entry_count = 0_u64;
    for _ in text.lines() {
        entry_count =
            entry_count
                .checked_add(1)
                .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                    PreparedBertVocabTxtExpression::EntryCount,
                ))?;
        enforce_limit(
            PreparedBertVocabTxtLimitAxis::VocabEntries,
            entry_count,
            limits.max_vocab_entries.get(),
        )?;
    }
    let vocabulary_cardinality =
        NonZeroU64::new(entry_count).ok_or(PreparedBertVocabTxtError::EmptyVocabulary)?;
    let max_token_id = checked_max_token_id(entry_count)?;
    let requested_span_bytes = entry_count
        .checked_mul(u64::try_from(size_of::<TokenSpan<'_>>()).map_err(|_| {
            PreparedBertVocabTxtError::ArithmeticOverflow(
                PreparedBertVocabTxtExpression::SpanScratchBytes,
            )
        })?)
        .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
            PreparedBertVocabTxtExpression::SpanScratchBytes,
        ))?;
    enforce_limit(
        PreparedBertVocabTxtLimitAxis::SpanScratchBytes,
        requested_span_bytes,
        limits.max_span_scratch_bytes.get(),
    )?;
    let entry_capacity = usize::try_from(entry_count).map_err(|_| {
        PreparedBertVocabTxtError::PlatformUnrepresentable {
            axis: PreparedBertVocabTxtLimitAxis::VocabEntries,
            value: entry_count,
        }
    })?;

    let mut spans = Vec::new();
    match reserve_mode {
        ReserveMode::Actual => spans.try_reserve_exact(entry_capacity).map_err(|_| {
            PreparedBertVocabTxtError::AllocationFailed {
                arena: PreparedBertVocabTxtAllocationArena::TokenSpans,
                requested_bytes: requested_span_bytes,
            }
        })?,
        #[cfg(test)]
        ReserveMode::Fail => {
            return Err(PreparedBertVocabTxtError::AllocationFailed {
                arena: PreparedBertVocabTxtAllocationArena::TokenSpans,
                requested_bytes: requested_span_bytes,
            });
        }
    }
    let span_scratch_bytes = u64::try_from(spans.capacity())
        .ok()
        .and_then(|capacity| capacity.checked_mul(u64::try_from(size_of::<TokenSpan<'_>>()).ok()?))
        .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
            PreparedBertVocabTxtExpression::SpanScratchBytes,
        ))?;
    enforce_limit(
        PreparedBertVocabTxtLimitAxis::SpanScratchBytes,
        span_scratch_bytes,
        limits.max_span_scratch_bytes.get(),
    )?;

    for (id, line) in text.lines().enumerate() {
        if spans.len() >= spans.capacity() {
            return Err(PreparedBertVocabTxtError::SecondPassCensusMismatch);
        }
        let id =
            u32::try_from(id).map_err(|_| PreparedBertVocabTxtError::TokenIdUnrepresentable {
                cardinality: entry_count,
            })?;
        spans.push(TokenSpan {
            token: line.trim_end_matches('\r').as_bytes(),
            id,
        });
    }
    if spans.len() != entry_capacity {
        return Err(PreparedBertVocabTxtError::SecondPassCensusMismatch);
    }

    heap_sort_token_spans(&mut spans, &mut work)?;
    for pair in spans.windows(2) {
        if compare_token_bytes(pair[0].token, pair[1].token, &mut work)? == Ordering::Equal {
            return Err(PreparedBertVocabTxtError::DuplicateToken {
                first_id: pair[0].id.min(pair[1].id),
                duplicate_id: pair[0].id.max(pair[1].id),
            });
        }
    }

    Ok(PreparedBertVocabTxtFacts {
        vocab_txt_bytes,
        vocabulary_cardinality,
        max_token_id,
        span_scratch_bytes,
        logical_parse_work_bytes: work.used,
    })
}

fn enforce_limit(
    axis: PreparedBertVocabTxtLimitAxis,
    actual: u64,
    limit: u64,
) -> Result<(), PreparedBertVocabTxtError> {
    if actual > limit {
        return Err(PreparedBertVocabTxtError::Exceeded {
            axis,
            actual,
            limit,
        });
    }
    Ok(())
}

fn checked_max_token_id(cardinality: u64) -> Result<u32, PreparedBertVocabTxtError> {
    let last = cardinality
        .checked_sub(1)
        .ok_or(PreparedBertVocabTxtError::EmptyVocabulary)?;
    u32::try_from(last)
        .map_err(|_| PreparedBertVocabTxtError::TokenIdUnrepresentable { cardinality })
}

fn compare_token_bytes(
    left: &[u8],
    right: &[u8],
    work: &mut WorkMeter,
) -> Result<Ordering, PreparedBertVocabTxtError> {
    work.charge(1)?;
    for (&left_byte, &right_byte) in left.iter().zip(right) {
        work.charge(2)?;
        match left_byte.cmp(&right_byte) {
            Ordering::Equal => {}
            ordering => return Ok(ordering),
        }
    }
    Ok(left.len().cmp(&right.len()))
}

fn compare_spans(
    left: TokenSpan<'_>,
    right: TokenSpan<'_>,
    work: &mut WorkMeter,
) -> Result<Ordering, PreparedBertVocabTxtError> {
    let token_order = compare_token_bytes(left.token, right.token, work)?;
    Ok(token_order.then_with(|| left.id.cmp(&right.id)))
}

fn heap_sort_token_spans(
    spans: &mut [TokenSpan<'_>],
    work: &mut WorkMeter,
) -> Result<(), PreparedBertVocabTxtError> {
    if spans.len() < 2 {
        return Ok(());
    }
    for root in (0..(spans.len() / 2)).rev() {
        work.charge(1)?;
        sift_down(spans, root, spans.len(), work)?;
    }
    for end in (1..spans.len()).rev() {
        work.charge(1)?;
        spans.swap(0, end);
        sift_down(spans, 0, end, work)?;
    }
    Ok(())
}

fn sift_down(
    spans: &mut [TokenSpan<'_>],
    mut root: usize,
    end: usize,
    work: &mut WorkMeter,
) -> Result<(), PreparedBertVocabTxtError> {
    loop {
        work.charge(1)?;
        let child = root
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                PreparedBertVocabTxtExpression::HeapIndex,
            ))?;
        if child >= end {
            return Ok(());
        }
        let mut largest = root;
        if compare_spans(spans[largest], spans[child], work)? == Ordering::Less {
            largest = child;
        }
        let right = child
            .checked_add(1)
            .ok_or(PreparedBertVocabTxtError::ArithmeticOverflow(
                PreparedBertVocabTxtExpression::HeapIndex,
            ))?;
        if right < end && compare_spans(spans[largest], spans[right], work)? == Ordering::Less {
            largest = right;
        }
        if largest == root {
            return Ok(());
        }
        spans.swap(root, largest);
        root = largest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: u64) -> NonZeroU64 {
        match NonZeroU64::new(value) {
            Some(value) => value,
            None => panic!("test limits must be nonzero"),
        }
    }

    fn limits(bytes: u64, entries: u64, scratch: u64, work: u64) -> PreparedBertVocabTxtLimits {
        match PreparedBertVocabTxtLimits::try_new(nz(bytes), nz(entries), nz(scratch), nz(work)) {
            Ok(limits) => limits,
            Err(error) => panic!("valid test limits: {error:?}"),
        }
    }

    #[test]
    fn exact_line_semantics_produce_dense_owned_facts() {
        for (input, expected_entries) in [
            (b"alpha\nbeta\n".as_slice(), 2),
            (b"alpha\r\nbeta\r\n".as_slice(), 2),
            (b" alpha \n\n".as_slice(), 2),
        ] {
            let facts = match parse_prepared_bert_vocab_txt(input, &limits(64, 8, 512, 4096)) {
                Ok(facts) => facts,
                Err(error) => panic!("valid vocab: {error:?}"),
            };
            assert_eq!(facts.vocab_txt_bytes(), input.len() as u64);
            assert_eq!(facts.vocabulary_cardinality().get(), expected_entries);
            assert_eq!(facts.max_token_id(), expected_entries as u32 - 1);
            assert!(facts.span_scratch_bytes() >= expected_entries * 16);
            assert!(facts.logical_parse_work_bytes() >= input.len() as u64 * 4);
        }
        assert_eq!(
            parse_prepared_bert_vocab_txt(b"", &limits(1, 1, 64, 64)),
            Err(PreparedBertVocabTxtError::EmptyVocabulary)
        );
    }

    #[test]
    fn normalized_duplicates_fail_but_prefixes_and_spaces_remain_distinct() {
        for (input, first_id, duplicate_id) in [
            (b"same\nother\nsame".as_slice(), 0, 2),
            (b"same\r\nother\nsame\r".as_slice(), 0, 2),
            (b"\nother\n\n".as_slice(), 0, 2),
        ] {
            assert_eq!(
                parse_prepared_bert_vocab_txt(input, &limits(64, 8, 512, 4096)),
                Err(PreparedBertVocabTxtError::DuplicateToken {
                    first_id,
                    duplicate_id,
                })
            );
        }
        for input in [b"a\naa".as_slice(), b"a\na ".as_slice()] {
            assert!(parse_prepared_bert_vocab_txt(input, &limits(64, 8, 512, 4096)).is_ok());
        }
    }

    #[test]
    fn resource_boundaries_fail_before_unbounded_work_or_growth() {
        let input = b"a\nb";
        assert_eq!(
            parse_prepared_bert_vocab_txt(input, &limits(2, 2, 128, 1024)),
            Err(PreparedBertVocabTxtError::Exceeded {
                axis: PreparedBertVocabTxtLimitAxis::VocabTxtBytes,
                actual: 3,
                limit: 2,
            })
        );
        assert_eq!(
            parse_prepared_bert_vocab_txt(input, &limits(3, 1, 128, 1024)),
            Err(PreparedBertVocabTxtError::Exceeded {
                axis: PreparedBertVocabTxtLimitAxis::VocabEntries,
                actual: 2,
                limit: 1,
            })
        );
        assert!(matches!(
            parse_prepared_bert_vocab_txt(input, &limits(3, 2, 1, 1024)),
            Err(PreparedBertVocabTxtError::Exceeded {
                axis: PreparedBertVocabTxtLimitAxis::SpanScratchBytes,
                ..
            })
        ));
        assert_eq!(
            parse_prepared_bert_vocab_txt(input, &limits(3, 2, 128, 11)),
            Err(PreparedBertVocabTxtError::Exceeded {
                axis: PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
                actual: 12,
                limit: 11,
            })
        );
        assert!(matches!(
            parse_prepared_bert_vocab_txt_with_reserve(
                input,
                &limits(3, 2, 128, 1024),
                ReserveMode::Fail,
            ),
            Err(PreparedBertVocabTxtError::AllocationFailed {
                arena: PreparedBertVocabTxtAllocationArena::TokenSpans,
                ..
            })
        ));
    }

    #[test]
    fn utf8_platform_and_dense_id_boundaries_are_typed() {
        assert_eq!(
            parse_prepared_bert_vocab_txt(b"\xff", &limits(1, 1, 64, 64)),
            Err(PreparedBertVocabTxtError::InvalidUtf8 { valid_up_to: 0 })
        );
        assert_eq!(checked_max_token_id(1), Ok(0));
        assert_eq!(checked_max_token_id(u64::from(u32::MAX) + 1), Ok(u32::MAX));
        assert_eq!(
            checked_max_token_id(u64::from(u32::MAX) + 2),
            Err(PreparedBertVocabTxtError::TokenIdUnrepresentable {
                cardinality: u64::from(u32::MAX) + 2,
            })
        );
        assert_eq!(
            PreparedBertVocabTxtLimits::try_new_with_platform_max(nz(8), nz(8), nz(9), nz(8), 8,),
            Err(PreparedBertVocabTxtError::PlatformUnrepresentable {
                axis: PreparedBertVocabTxtLimitAxis::SpanScratchBytes,
                value: 9,
            })
        );
    }
}
