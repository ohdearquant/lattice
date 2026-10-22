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

#[test]
fn exact_known_token_ids_are_observed_without_constraining_generic_vocabularies() {
    let input = b"ordinary\n[MASK]\r\n[CLS]\n[UNK]\n[SEP]\n[PAD]";
    let broad = limits(128, 16, 1024, 8192);
    let facts = match parse_prepared_bert_vocab_txt(input, &broad) {
        Ok(facts) => facts,
        Err(error) => panic!("valid WordPiece vocab: {error:?}"),
    };
    assert_eq!(facts.known_token_ids().cls(), Some(2));
    assert_eq!(facts.known_token_ids().sep(), Some(4));
    assert_eq!(facts.known_token_ids().pad(), Some(5));
    assert_eq!(facts.known_token_ids().unk(), Some(3));
    assert_eq!(facts.known_token_ids().mask(), Some(1));

    let exact_work = facts.logical_parse_work_bytes();
    assert!(parse_prepared_bert_vocab_txt(input, &limits(128, 16, 1024, exact_work)).is_ok());
    assert!(matches!(
        parse_prepared_bert_vocab_txt(input, &limits(128, 16, 1024, exact_work - 1),),
        Err(PreparedBertVocabTxtError::Exceeded {
            axis: PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
            ..
        })
    ));

    let generic = match parse_prepared_bert_vocab_txt(b"a\nb", &broad) {
        Ok(facts) => facts,
        Err(error) => panic!("valid generic BPE vocab: {error:?}"),
    };
    assert_eq!(
        generic.known_token_ids(),
        PreparedBertVocabTxtKnownTokenIds {
            cls: None,
            sep: None,
            pad: None,
            unk: None,
            mask: None,
        }
    );

    let near_miss = match parse_prepared_bert_vocab_txt(
        b"[CLS]\n[SEP]\n[PAD]\n[UNK] \n[unk]\n[MASK]",
        &broad,
    ) {
        Ok(facts) => facts,
        Err(error) => panic!("valid generic near-miss vocab: {error:?}"),
    };
    assert_eq!(near_miss.known_token_ids().unk(), None);
    assert_eq!(
        validate_prepared_bert_wordpiece_vocab_txt(near_miss),
        Err(PreparedBertWordPieceVocabTxtError::MissingSpecialToken(
            PreparedBertWordPieceSpecialToken::Unk,
        ))
    );
}

#[test]
fn wordpiece_validation_requires_specials_in_legacy_order() {
    let names = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];
    let tokens = [
        PreparedBertWordPieceSpecialToken::Cls,
        PreparedBertWordPieceSpecialToken::Sep,
        PreparedBertWordPieceSpecialToken::Pad,
        PreparedBertWordPieceSpecialToken::Unk,
        PreparedBertWordPieceSpecialToken::Mask,
    ];
    let complete = match parse_prepared_bert_vocab_txt(
        names.join("\n").as_bytes(),
        &limits(128, 8, 512, 8192),
    ) {
        Ok(facts) => facts,
        Err(error) => panic!("valid complete vocab: {error:?}"),
    };
    let validated = match validate_prepared_bert_wordpiece_vocab_txt(complete) {
        Ok(facts) => facts,
        Err(error) => panic!("complete specials: {error:?}"),
    };
    assert_eq!(validated.cls_id(), 0);
    assert_eq!(validated.sep_id(), 1);
    assert_eq!(validated.pad_id(), 2);
    assert_eq!(validated.unk_id(), 3);
    assert_eq!(validated.mask_id(), 4);
    assert_eq!(validated.vocab_txt(), complete);

    for (missing_index, expected) in tokens.into_iter().enumerate() {
        let input = names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| (index != missing_index).then_some(*name))
            .collect::<Vec<_>>()
            .join("\n");
        let facts =
            match parse_prepared_bert_vocab_txt(input.as_bytes(), &limits(128, 8, 512, 8192)) {
                Ok(facts) => facts,
                Err(error) => panic!("valid incomplete vocab: {error:?}"),
            };
        assert_eq!(
            validate_prepared_bert_wordpiece_vocab_txt(facts),
            Err(PreparedBertWordPieceVocabTxtError::MissingSpecialToken(
                expected,
            ))
        );
    }
}

#[test]
fn bpe_control_ids_follow_raw_vocab_txt_constructor_precedence() {
    let input = b"ordinary\r\n<|pad|>\n<pad>\r\n[PAD]\n<|endoftext|>\r\n</s>\n<unk>\r\n[UNK]\n<bos>\r\n<s>\n<eos>";
    let generic = match parse_prepared_bert_vocab_txt(input, &limits(256, 16, 2048, 32_768)) {
        Ok(facts) => facts,
        Err(error) => panic!("valid BPE vocab: {error:?}"),
    };
    let known = generic.bpe_known_token_ids();
    assert_eq!(known.pipe_pad(), Some(1));
    assert_eq!(known.pad(), Some(2));
    assert_eq!(known.endoftext(), Some(4));
    assert_eq!(known.slash_s(), Some(5));
    assert_eq!(known.unk(), Some(6));
    assert_eq!(known.bos(), Some(8));
    assert_eq!(known.s(), Some(9));
    assert_eq!(known.eos(), Some(10));
    assert_eq!(generic.known_token_ids().pad(), Some(3));
    assert_eq!(generic.known_token_ids().unk(), Some(7));
    let resolved = resolve_prepared_bert_bpe_vocab_txt(generic);
    assert_eq!(resolved.pad_id(), 1);
    assert_eq!(resolved.unk_id(), Some(6));
    assert_eq!(resolved.bos_id(), Some(8));
    assert_eq!(resolved.eos_id(), Some(10));
    assert_eq!(resolved.vocab_txt(), generic);

    let exact_work = generic.logical_parse_work_bytes();
    assert!(parse_prepared_bert_vocab_txt(input, &limits(256, 16, 2048, exact_work),).is_ok());
    assert!(matches!(
        parse_prepared_bert_vocab_txt(input, &limits(256, 16, 2048, exact_work - 1),),
        Err(PreparedBertVocabTxtError::Exceeded {
            axis: PreparedBertVocabTxtLimitAxis::ParseWorkBytes,
            ..
        })
    ));
}

#[test]
fn bpe_control_fallbacks_and_near_misses_are_exact() {
    for pair in KNOWN_TOKEN_TARGETS.windows(2) {
        assert!(pair[0].0 < pair[1].0, "known-token table must be sorted");
    }
    for (input, pad, unk, bos, eos) in [
        (
            "<pad>\n[PAD]\n<|endoftext|>\n</s>\n[UNK]\n<s>",
            0,
            Some(4),
            Some(5),
            Some(3),
        ),
        ("[PAD]\n<|endoftext|>\n</s>", 0, None, None, Some(2)),
        ("<|endoftext|>\n</s>", 0, None, None, Some(1)),
        ("</s>", 0, None, None, Some(0)),
        ("ordinary", 0, None, None, None),
        ("<PAD>\n<unk> \n<BOS>\n<eos> ", 0, None, None, None),
    ] {
        let generic =
            match parse_prepared_bert_vocab_txt(input.as_bytes(), &limits(256, 16, 2048, 32_768)) {
                Ok(facts) => facts,
                Err(error) => panic!("valid fallback vocab: {error:?}"),
            };
        let resolved = resolve_prepared_bert_bpe_vocab_txt(generic);
        assert_eq!(resolved.pad_id(), pad, "input={input:?}");
        assert_eq!(resolved.unk_id(), unk, "input={input:?}");
        assert_eq!(resolved.bos_id(), bos, "input={input:?}");
        assert_eq!(resolved.eos_id(), eos, "input={input:?}");
    }
}
