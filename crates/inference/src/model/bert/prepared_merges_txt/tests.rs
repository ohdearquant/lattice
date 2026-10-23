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
        "a\u{00a0}b\n",
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
fn a_line_with_a_third_token_is_refused_not_truncated() {
    // Regression for the truncation defect: `"a b c"` must be REJECTED
    // and name its 1-based line number, never silently narrowed to the
    // pair ("a", "b") with "c" dropped.
    let source = b"a b\nc d e\n";
    let bytes = u64::try_from(source.len()).unwrap();
    assert_eq!(
        census_prepared_bert_bpe_merges_txt(source, &limits(bytes, 10, bytes * 8)),
        Err(PreparedBertBpeMergesTxtError::MalformedMergeLine { line: 2 })
    );
    assert_eq!(
        retain_prepared_bert_bpe_merges_txt_pairs(
            source,
            &retained_limits(bytes, 10, 1024, bytes * 16),
        ),
        Err(PreparedBertBpeMergesTxtError::MalformedMergeLine { line: 2 })
    );

    // A three-token line is malformed regardless of which whitespace
    // class separates the tokens (the NBSP case that motivated the
    // fixture change above), and the line number counts blank/comment
    // lines too.
    let unicode_source = "# header\na\u{00a0}b c\n";
    let unicode_bytes = u64::try_from(unicode_source.len()).unwrap();
    assert_eq!(
        census_prepared_bert_bpe_merges_txt(
            unicode_source.as_bytes(),
            &limits(unicode_bytes, 10, unicode_bytes * 8),
        ),
        Err(PreparedBertBpeMergesTxtError::MalformedMergeLine { line: 2 })
    );

    // A single incomplete token is not "malformed" in this sense: there
    // is no extra data to discard, so the line is skipped exactly as the
    // legacy loader skips it.
    let incomplete = b"single\na b\n";
    let incomplete_bytes = u64::try_from(incomplete.len()).unwrap();
    let facts = census_prepared_bert_bpe_merges_txt(
        incomplete,
        &limits(incomplete_bytes, 10, incomplete_bytes * 8),
    )
    .unwrap();
    assert_eq!(facts.merge_entry_count(), 1);
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
        PreparedBertBpeMergesTxtLimits::try_new_with_platform_max(nz(8), nz(8), nz(8), 8,).is_ok()
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
        "a\u{00a0}b\n",
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
