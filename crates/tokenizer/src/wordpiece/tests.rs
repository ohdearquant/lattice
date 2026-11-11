use super::*;

fn synthetic_spec_vocab() -> String {
    let max_id = 16_515usize;
    let mut tokens = (0..=max_id)
        .map(|i| format!("[unused{i}]"))
        .collect::<Vec<_>>();

    let entries = [
        (0usize, "[PAD]"),
        (100, "[UNK]"),
        (101, "[CLS]"),
        (102, "[SEP]"),
        (103, "[MASK]"),
        (1037, "a"),
        (1996, "the"),
        (2003, "is"),
        (2088, "world"),
        (2829, "brown"),
        (2890, "##re"),
        (4248, "quick"),
        (4419, "fox"),
        (4667, "##ding"),
        (4895, "un"),
        (7592, "hello"),
        (7861, "em"),
        (8270, "##bed"),
        (9932, "ai"),
        (10_880, "##zable"),
        (13_970, "transforming"),
        (16_515, "##cogni"),
    ];

    for (idx, token) in entries {
        tokens[idx] = token.to_string();
    }

    tokens.join("\n")
}

#[test]
fn test_trie_longest_match() {
    let mut builder = PlainTrieBuilder::new();
    builder.insert("em", 1);
    builder.insert("embed", 2);
    builder.insert("embedding", 3);
    let trie = builder.build();
    let chars: Vec<char> = "embedding".chars().collect();
    assert_eq!(trie.longest_match_chars(&chars, 0), Some((3, 9)));
}

#[test]
fn test_empty_content_added_token_does_not_hang() {
    // Regression (mirrors the BPE guard): a malformed tokenizer.json with a
    // zero-length added-token `content` injects "" into added_tokens.
    // `"".starts_with(_)` is always true, so match_added_token returns the
    // same `pos` and tokenize_to_ids_into never advances — an infinite loop
    // with unbounded output on the first tokenize() call. The constructor
    // must drop zero-length added tokens. Run on a worker thread with a 5s
    // timeout so a re-introduced hang fails fast instead of stalling the
    // test binary.
    let mut vocab = HashMap::new();
    for (s, i) in [
        ("[PAD]", 0u32),
        ("[UNK]", 1),
        ("[CLS]", 2),
        ("[SEP]", 3),
        ("[MASK]", 4),
        ("a", 5),
        ("b", 6),
        ("c", 7),
    ] {
        vocab.insert(s.to_string(), i);
    }
    let mut added = HashMap::new();
    added.insert(String::new(), 1u32);
    let tokenizer = WordPieceTokenizer::from_vocab_map_with_added(
        vocab,
        added,
        DEFAULT_WORDPIECE_CACHE_CAPACITY,
        DEFAULT_MAX_SEQ_LEN,
    )
    .expect("construct wordpiece tokenizer with empty added token");

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(tokenizer.tokenize("abc").real_length);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(len) => assert!(
            len < 1000,
            "empty added token produced runaway output ({len}); zero-length tokens must be dropped"
        ),
        Err(_) => {
            panic!("tokenize() hung on a zero-length added token (infinite-loop regression)")
        }
    }
}

#[test]
fn test_empty_content_vocab_token_does_not_hang() {
    // Regression (companion to the added-token guard): a malformed
    // tokenizer.json `model.vocab` can carry an empty key (`{"": id}`) or a
    // "##"-only key (which strip_prefix("##") reduces to ""). Either inserts
    // the empty string into a trie, which sets the trie *root* value;
    // longest_match_chars then returns a zero-length match (idx == start) at
    // every position and wordpiece_tokenize_word_into's `start = end` never
    // advances — an infinite loop with unbounded output. Construction must drop
    // empty subwords and the loop must reject non-advancing matches. Run on a
    // worker thread with a 5s timeout so a re-introduced hang fails fast.
    let mut vocab = HashMap::new();
    for (s, i) in [
        ("[PAD]", 0u32),
        ("[UNK]", 1),
        ("[CLS]", 2),
        ("[SEP]", 3),
        ("[MASK]", 4),
        ("a", 5),
        ("b", 6),
        ("c", 7),
        ("", 8),   // raw empty vocab key
        ("##", 9), // "##"-only key strips to ""
    ] {
        vocab.insert(s.to_string(), i);
    }
    let tokenizer = WordPieceTokenizer::from_vocab_map_with_added(
        vocab,
        HashMap::new(),
        DEFAULT_WORDPIECE_CACHE_CAPACITY,
        DEFAULT_MAX_SEQ_LEN,
    )
    .expect("construct wordpiece tokenizer with empty vocab token");

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(tokenizer.tokenize("abc").real_length);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(len) => assert!(
            len < 1000,
            "empty vocab token produced runaway output ({len}); empty subwords must be dropped"
        ),
        Err(_) => {
            panic!("tokenize() hung on a zero-length vocab token (infinite-loop regression)")
        }
    }
}

#[test]
fn test_tokenize_hello_world() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    let input = tokenizer.tokenize("hello world");
    assert_eq!(input.input_ids[0], 101);
    assert_eq!(input.input_ids[1], 7592);
    assert_eq!(input.input_ids[2], 2088);
    assert_eq!(input.input_ids[3], 102);
    assert_eq!(input.real_length, 4);
}

#[test]
fn test_tokenize_subwords() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    let input = tokenizer.tokenize("embedding");
    assert_eq!(input.input_ids[1], 7861);
    assert_eq!(input.input_ids[2], 8270);
    assert_eq!(input.input_ids[3], 4667);
    assert_eq!(input.real_length, 5);
}

fn whole_word_unknown_fixture(text: &str) -> Vec<u32> {
    let tokenizer =
        WordPieceTokenizer::from_str("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n##z").unwrap();
    let input = tokenizer.tokenize(text);
    input.input_ids[..input.real_length].to_vec()
}

#[test]
fn wordpiece_unknown_discards_matched_prefix_and_suffix() {
    assert_eq!(whole_word_unknown_fixture("helloxz"), vec![2, 1, 3]);
}

#[test]
fn wordpiece_unknown_at_first_character_discards_following_pieces() {
    assert_eq!(whole_word_unknown_fixture("xz"), vec![2, 1, 3]);
}

#[test]
fn wordpiece_multiple_unknown_characters_are_one_unknown_word() {
    assert_eq!(whole_word_unknown_fixture("xx"), vec![2, 1, 3]);
}

#[test]
fn wordpiece_successful_greedy_segmentation_remains_intact() {
    assert_eq!(whole_word_unknown_fixture("helloz"), vec![2, 5, 6, 3]);
    assert_eq!(whole_word_unknown_fixture("hello"), vec![2, 5, 3]);
}

fn punctuation_category_fixture(text: &str) -> Vec<u32> {
    let tokenizer = WordPieceTokenizer::from_str(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na€b\na∑b\na😀b\naाb\na\n—\nb\na⁄b\n$",
    )
    .unwrap();
    let input = tokenizer.tokenize(text);
    input.input_ids[..input.real_length].to_vec()
}

#[test]
fn wordpiece_keeps_currency_symbol_in_word() {
    assert_eq!(punctuation_category_fixture("a€b"), vec![2, 5, 3]);
}

#[test]
fn wordpiece_keeps_math_symbol_in_word() {
    assert_eq!(punctuation_category_fixture("a∑b"), vec![2, 6, 3]);
}

#[test]
fn wordpiece_keeps_other_symbol_in_word() {
    assert_eq!(punctuation_category_fixture("a😀b"), vec![2, 7, 3]);
}

#[test]
fn wordpiece_keeps_spacing_combining_mark_in_word() {
    assert_eq!(punctuation_category_fixture("aाb"), vec![2, 8, 3]);
}

#[test]
fn wordpiece_splits_unicode_punctuation() {
    assert_eq!(punctuation_category_fixture("a—b"), vec![2, 9, 10, 11, 3]);
}

#[test]
fn wordpiece_keeps_symbol_inside_punctuation_block() {
    assert_eq!(punctuation_category_fixture("a⁄b"), vec![2, 12, 3]);
}

#[test]
fn wordpiece_splits_ascii_symbol_as_bert_punctuation() {
    assert_eq!(punctuation_category_fixture("a$b"), vec![2, 9, 13, 11, 3]);
}

fn nfd_symbol_fixture(text: &str) -> Vec<u32> {
    let tokenizer =
        WordPieceTokenizer::from_str("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\n`\n=\n<\n>\nb")
            .unwrap();
    let input = tokenizer.tokenize(text);
    input.input_ids[..input.real_length].to_vec()
}

#[test]
fn wordpiece_nfd_greek_varia_matches_hf() {
    assert_eq!(nfd_symbol_fixture("a\u{1fef}b"), vec![2, 5, 6, 10, 3]);
}

#[test]
fn wordpiece_nfd_not_equal_matches_hf() {
    assert_eq!(nfd_symbol_fixture("a\u{2260}b"), vec![2, 5, 7, 10, 3]);
}

#[test]
fn wordpiece_nfd_not_less_than_matches_hf() {
    assert_eq!(nfd_symbol_fixture("a\u{226e}b"), vec![2, 5, 8, 10, 3]);
}

#[test]
fn wordpiece_nfd_not_greater_than_matches_hf() {
    assert_eq!(nfd_symbol_fixture("a\u{226f}b"), vec![2, 5, 9, 10, 3]);
}

#[test]
fn wordpiece_default_word_length_limit_is_one_hundred_characters() {
    let tokenizer =
        WordPieceTokenizer::from_str("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\n##a").unwrap();
    let at_limit = tokenizer.tokenize(&"a".repeat(100));
    assert_eq!(at_limit.real_length, 102);
    assert_eq!(at_limit.input_ids[1], 5);
    assert!(at_limit.input_ids[2..101].iter().all(|&id| id == 6));
    let over_limit = tokenizer.tokenize(&"a".repeat(101));
    assert_eq!(&over_limit.input_ids[..over_limit.real_length], &[2, 1, 3]);
}

#[test]
fn wordpiece_word_length_limit_counts_unicode_characters() {
    let tokenizer =
        WordPieceTokenizer::from_str("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nя\n##я").unwrap();
    let at_limit = tokenizer.tokenize(&"я".repeat(100));
    assert_eq!(at_limit.real_length, 102);
    let over_limit = tokenizer.tokenize(&"я".repeat(101));
    assert_eq!(&over_limit.input_ids[..over_limit.real_length], &[2, 1, 3]);
}

#[test]
fn wordpiece_json_uses_configured_word_length_limit() {
    let json = r###"{"model":{"type":"WordPiece","max_input_chars_per_word":3,"vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"##a":6}}}"###;
    let tokenizer = WordPieceTokenizer::from_tokenizer_json_str(json).unwrap();
    let at_limit = tokenizer.tokenize("aaa");
    assert_eq!(
        &at_limit.input_ids[..at_limit.real_length],
        &[2, 5, 6, 6, 3]
    );
    let over_limit = tokenizer.tokenize("aaaa");
    assert_eq!(&over_limit.input_ids[..over_limit.real_length], &[2, 1, 3]);
}

#[test]
fn wordpiece_json_uses_configured_unknown_token() {
    let json = r#"{"model":{"type":"WordPiece","unk_token":"<missing>","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"<missing>":5,"a":6}}}"#;
    let tokenizer = WordPieceTokenizer::from_tokenizer_json_str(json).unwrap();
    let input = tokenizer.tokenize("x");
    assert_eq!(&input.input_ids[..input.real_length], &[2, 5, 3]);
}

#[test]
fn wordpiece_json_custom_unknown_does_not_require_default_unknown() {
    let json = r#"{"model":{"type":"WordPiece","unk_token":"<missing>","vocab":{"[PAD]":0,"<missing>":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5}}}"#;
    let tokenizer = WordPieceTokenizer::from_tokenizer_json_str(json).unwrap();
    let input = tokenizer.tokenize("x");
    assert_eq!(&input.input_ids[..input.real_length], &[2, 1, 3]);
}

#[test]
fn wordpiece_effective_settings_bind_word_length_limit_after_clone() {
    let default_json = r###"{"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"##a":6}}}"###;
    let limited_json = r###"{"model":{"type":"WordPiece","max_input_chars_per_word":3,"vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"##a":6}}}"###;
    let default = WordPieceTokenizer::from_tokenizer_json_str(default_json).unwrap();
    let limited = WordPieceTokenizer::from_tokenizer_json_str(limited_json).unwrap();
    assert_ne!(default.embedding_settings(), limited.embedding_settings());
    assert_eq!(
        limited.embedding_settings(),
        limited.with_max_seq_len(512).embedding_settings()
    );
}

#[test]
fn test_spec_tokenizer_vectors() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();

    let cases: &[(&str, &[u32])] = &[
        ("hello world", &[7592, 2088]),
        ("The quick brown fox", &[1996, 4248, 2829, 4419]),
        ("embedding", &[7861, 8270, 4667]),
        ("unrecognizable", &[4895, 2890, 16515, 10880]),
        ("", &[]),
        ("a", &[1037]),
        (
            "AI is transforming the world",
            &[9932, 2003, 13970, 1996, 2088],
        ),
    ];

    for (text, expected) in cases {
        let input = tokenizer.tokenize(text);
        let actual = &input.input_ids[1..input.real_length - 1];
        assert_eq!(actual, *expected, "text={text:?}");
    }
}

#[test]
fn test_pre_tokenize_lowercase_punctuation_and_accents() {
    let tokens = pre_tokenize("He\u{0301}llo, WORLD!");
    assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
}

#[test]
fn test_tokenize_batch_pads_to_batch_max() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    let batch = tokenizer.tokenize_batch(&["hello", "hello world"]);
    assert_eq!(batch[0].input_ids.len(), 4);
    assert_eq!(batch[1].input_ids.len(), 4);
    assert_eq!(batch[0].attention_mask, vec![1, 1, 1, 0]);
    assert_eq!(batch[1].attention_mask, vec![1, 1, 1, 1]);
}

/// A deterministic, varied batch: ASCII, accented, CJK, empty, whitespace,
/// punctuation, a sub-`max_seq_len` long text and a text past
/// `DEFAULT_MAX_SEQ_LEN` (512) so the truncation path is exercised too.
fn varied_batch_texts() -> Vec<String> {
    let mut texts: Vec<String> = vec![
        String::new(),
        " ".to_string(),
        "a".to_string(),
        "hello".to_string(),
        "hello world".to_string(),
        "HELLO, WORLD!".to_string(),
        "héllo wörld".to_string(),
        "He\u{0301}llo".to_string(),
        "中文测试".to_string(),
        "embedding plugin".to_string(),
        "unrecognizable transforming ai tokenizer throughput".to_string(),
        "the quick brown fox".to_string(),
    ];
    texts.push("hello world ".repeat(80));
    texts.push("hello ".repeat(600));

    let seeds = [
        "hello",
        "world",
        "quick",
        "brown",
        "fox",
        "embedding",
        "ai",
        "tokenizer",
    ];
    let mut i = 0usize;
    while texts.len() < 120 {
        let a = seeds[i % seeds.len()];
        let b = seeds[(i * 7 + 3) % seeds.len()];
        texts.push(format!("{a} {b} 中文 {i}"));
        i += 1;
    }
    texts
}

#[test]
fn test_tokenize_batch_matches_per_text_tokenize_on_rayon_pools() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    let texts = varied_batch_texts();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    // `tokenize` pads every text to `max_seq_len` while `tokenize_batch`
    // pads to the batch maximum, so the two structs differ in trailing pad
    // slots by design. Compare the tokens that are real, the real length
    // and the pre-truncation count; those are the batch-independent
    // outputs of the tokenizer itself.
    fn real(input: &TokenizedInput) -> (Vec<u32>, usize, usize) {
        (
            input.input_ids[..input.real_length].to_vec(),
            input.real_length,
            input.pre_truncation_len,
        )
    }
    let expected: Vec<(Vec<u32>, usize, usize)> = refs
        .iter()
        .map(|text| real(&tokenizer.tokenize(text)))
        .collect();

    for threads in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let batch = pool.install(|| tokenizer.tokenize_batch(&refs));
        let actual: Vec<(Vec<u32>, usize, usize)> = batch.iter().map(real).collect();
        assert_eq!(
            actual, expected,
            "a {threads}-thread pool diverged from per-text tokenize"
        );
    }

    // With a `max_seq_len` small enough to bind on every text, both paths
    // pad to the same length, so the full structs must compare equal.
    let clipped = tokenizer.with_max_seq_len(4);
    let expected_full: Vec<TokenizedInput> =
        refs.iter().map(|text| clipped.tokenize(text)).collect();
    for threads in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let batch = pool.install(|| clipped.tokenize_batch(&refs));
        assert_eq!(
            batch, expected_full,
            "a {threads}-thread pool diverged from per-text tokenize at max_seq_len 4"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn test_tokenize_batch_chunks_stay_on_current_pool_threads() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    let texts = varied_batch_texts();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    assert!(
        refs.len() >= 100,
        "the bound below assumes a 100-plus batch"
    );

    let prefix = "wp-batch-bound-";
    batch_chunk_probe::discard_matching(prefix);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .thread_name(move |index| format!("{prefix}{index}"))
        .build()
        .unwrap();
    let batch = pool.install(|| tokenizer.tokenize_batch(&refs));
    assert_eq!(batch.len(), refs.len());

    let recorded = batch_chunk_probe::take_matching(prefix);
    assert!(
        !recorded.is_empty(),
        "the chunk hook observed no chunk; the batch did not take the parallel path"
    );
    // `par_chunks` splits the batch into exactly `min(pool_threads, texts)`
    // chunks up front, so a 2-thread pool over 100-plus texts produces two
    // chunk executions. Sizing the chunks from a wider thread count (the
    // host's parallelism rather than the pool's) would raise this count.
    // The count is summed over executing threads so it does not depend on
    // how the pool schedules the two chunks onto its two threads.
    let chunk_executions: usize = recorded.iter().map(|(_, lengths)| lengths.len()).sum();
    assert_eq!(
        chunk_executions, 2,
        "expected two chunks from a 2-thread pool, saw {recorded:?}"
    );
    let threads: std::collections::BTreeSet<&str> =
        recorded.iter().map(|(name, _)| name.as_str()).collect();
    assert!(
        threads.len() <= 2,
        "chunks ran on {} threads ({threads:?}); a 2-thread pool must bound the batch",
        threads.len()
    );
    for name in &threads {
        assert!(
            name.starts_with(prefix),
            "chunk ran on {name}, outside the test pool"
        );
    }
    let texts_seen: usize = recorded
        .iter()
        .flat_map(|(_, lengths)| lengths.iter())
        .sum();
    assert_eq!(
        texts_seen,
        refs.len(),
        "the hook did not observe every text exactly once"
    );
}

#[test]
fn test_wordpiece_cache_hits() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    let first = tokenizer.tokenize("embedding embedding");
    let second = tokenizer.tokenize("embedding embedding");
    assert_eq!(first, second);
}

// --- cross-encoder pair tokenization tests ---

#[test]
fn test_tokenize_pair_layout() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    let input = tokenizer.tokenize_pair("hello", "world");
    // Expected: [CLS](101) hello(7592) [SEP](102) world(2088) [SEP](102)
    assert_eq!(input.real_length, 5);
    assert_eq!(input.input_ids[0], 101); // [CLS]
    assert_eq!(input.input_ids[1], 7592); // hello
    assert_eq!(input.input_ids[2], 102); // first [SEP]
    assert_eq!(input.input_ids[3], 2088); // world
    assert_eq!(input.input_ids[4], 102); // final [SEP]
}

#[test]
fn test_tokenize_pair_token_type_ids() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    let input = tokenizer.tokenize_pair("hello", "world");
    // Segment 0: [CLS], hello, first [SEP] → type 0
    // Segment 1: world, final [SEP] → type 1
    assert_eq!(
        &input.token_type_ids[..input.real_length],
        &[0u32, 0, 0, 1, 1]
    );
    // Padding positions must also be type 0
    for i in input.real_length..input.input_ids.len() {
        assert_eq!(
            input.token_type_ids[i], 0,
            "padding at position {i} must have token_type_id 0"
        );
    }
}

#[test]
fn test_tokenize_pair_doc_truncated_before_query() {
    // max_seq_len=6: payload_budget=3, query "hello world"=2 tokens → doc_budget=1
    // doc "is the world" → 3 tokens, truncated to 1 → [is(2003)]
    // Expected: [CLS(101), hello(7592), world(2088), SEP(102), is(2003), SEP(102)]
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab())
        .unwrap()
        .with_max_seq_len(6);
    let input = tokenizer.tokenize_pair("hello world", "is the world");
    assert_eq!(input.real_length, 6);
    assert_eq!(input.input_ids[0], 101); // [CLS]
    assert_eq!(input.input_ids[1], 7592); // hello
    assert_eq!(input.input_ids[2], 2088); // world
    assert_eq!(input.input_ids[3], 102); // first [SEP]
    assert_eq!(input.input_ids[4], 2003); // is (doc, truncated to 1 token)
    assert_eq!(input.input_ids[5], 102); // final [SEP]
    assert_eq!(&input.token_type_ids[..6], &[0u32, 0, 0, 0, 1, 1]);
}

#[test]
fn test_tokenize_pair_query_truncated_drops_doc() {
    // max_seq_len=4: payload_budget=1, query "hello world"=2 > 1
    // → query truncated to [hello(7592)], doc cleared
    // Expected: [CLS(101), hello(7592), SEP(102), SEP(102)]
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab())
        .unwrap()
        .with_max_seq_len(4);
    let input = tokenizer.tokenize_pair("hello world", "is");
    assert_eq!(input.real_length, 4);
    assert_eq!(input.input_ids[0], 101); // [CLS]
    assert_eq!(input.input_ids[1], 7592); // hello (only query token kept)
    assert_eq!(input.input_ids[2], 102); // first [SEP]
    assert_eq!(input.input_ids[3], 102); // final [SEP]
    // token_type_ids: CLS=0, hello=0, first SEP=0, final SEP=1
    assert_eq!(&input.token_type_ids[..4], &[0u32, 0, 0, 1]);
}

#[test]
fn test_tokenize_pair_supports_pair_tokenization() {
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
    assert!(tokenizer.supports_pair_tokenization());
}

#[test]
fn test_tokenize_pair_padding_attention_mask() {
    // With large max_seq_len, positions past real_length must have attention_mask=0
    let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab())
        .unwrap()
        .with_max_seq_len(16);
    let input = tokenizer.tokenize_pair("hello", "world");
    assert_eq!(input.real_length, 5);
    for i in 0..5 {
        assert_eq!(
            input.attention_mask[i], 1,
            "real token at {i} must be unmasked"
        );
    }
    for i in 5..16 {
        assert_eq!(input.attention_mask[i], 0, "padding at {i} must be masked");
        assert_eq!(input.token_type_ids[i], 0, "padding type at {i} must be 0");
    }
}

const CONTINUATION_PREFIX_JSON: &str = r###"{"model":{"type":"WordPiece","continuing_subword_prefix":"@@","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"@@b":6,"##b":7,"b":8}}}"###;
const EMPTY_PREFIX_JSON: &str = r#"{"model":{"type":"WordPiece","continuing_subword_prefix":"","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"b":6}}}"#;

fn continuation_prefix_ids(tokenizer: &WordPieceTokenizer, text: &str) -> Vec<u32> {
    let input = tokenizer.tokenize(text);
    input.input_ids[..input.real_length].to_vec()
}

#[test]
fn wordpiece_json_uses_continuing_subword_prefix() {
    let tokenizer = WordPieceTokenizer::from_tokenizer_json_str(CONTINUATION_PREFIX_JSON).unwrap();
    assert_eq!(
        continuation_prefix_ids(&tokenizer, "ab"),
        [2, 5, 6, 3],
        "configured continuation prefix selected wrong token IDs"
    );
    assert_eq!(tokenizer.token_for_id(6), Some("@@b"));
}

#[test]
fn wordpiece_default_prefix_constructors_agree() {
    let missing = r###"{"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"##b":6,"@@b":7}}}"###;
    let explicit = r###"{"model":{"type":"WordPiece","continuing_subword_prefix":"##","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"##b":6,"@@b":7}}}"###;
    let tokenizers = [
        WordPieceTokenizer::from_tokenizer_json_str(missing).unwrap(),
        WordPieceTokenizer::from_tokenizer_json_str(explicit).unwrap(),
        WordPieceTokenizer::from_str("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\n##b\n@@b").unwrap(),
    ];
    for tokenizer in &tokenizers {
        assert_eq!(continuation_prefix_ids(tokenizer, "ab"), [2, 5, 6, 3]);
        assert_eq!(
            tokenizer.embedding_settings(),
            tokenizers[0].embedding_settings()
        );
    }
}

#[test]
fn wordpiece_custom_prefix_edge_cases() {
    let alphabetic = r#"{"model":{"type":"WordPiece","continuing_subword_prefix":"zz","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"zzb":6,"zz":7}}}"#;
    let unicode = r#"{"model":{"type":"WordPiece","continuing_subword_prefix":"я","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"яb":6}}}"#;
    let missing_continuation = r#"{"model":{"type":"WordPiece","continuing_subword_prefix":"zz","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"b":6}}}"#;
    let cases: [(&str, &str, &str, &[u32]); 8] = [
        ("prefix-looking initial key", alphabetic, "zzb", &[2, 6, 3]),
        ("prefix-only initial key", alphabetic, "zz", &[2, 7, 3]),
        ("alphabetic continuation", alphabetic, "ab", &[2, 5, 6, 3]),
        (
            "explicit empty prefix",
            EMPTY_PREFIX_JSON,
            "ab",
            &[2, 5, 6, 3],
        ),
        ("Unicode continuation", unicode, "ab", &[2, 5, 6, 3]),
        ("Unicode initial key", unicode, "яb", &[2, 6, 3]),
        (
            "raw key is not continuation",
            missing_continuation,
            "ab",
            &[2, 1, 3],
        ),
        ("raw initial key", missing_continuation, "b", &[2, 6, 3]),
    ];
    for (name, json, text, expected) in cases {
        let tokenizer = WordPieceTokenizer::from_tokenizer_json_str(json).unwrap();
        assert_eq!(
            continuation_prefix_ids(&tokenizer, text),
            expected,
            "continuation prefix edge case: {name}"
        );
    }
}

#[test]
fn wordpiece_custom_prefix_routes_agree() {
    let tokenizer = WordPieceTokenizer::from_tokenizer_json_str(CONTINUATION_PREFIX_JSON).unwrap();
    for _ in 0..2 {
        assert_eq!(continuation_prefix_ids(&tokenizer, "ab"), [2, 5, 6, 3]);
    }
    let shorter = tokenizer.with_max_seq_len(16);
    assert_eq!(continuation_prefix_ids(&shorter, "ab"), [2, 5, 6, 3]);
    let batch = shorter.tokenize_batch(&["ab", "b"]);
    assert_eq!(&batch[0].input_ids[..batch[0].real_length], &[2, 5, 6, 3]);
    assert_eq!(&batch[1].input_ids[..batch[1].real_length], &[2, 8, 3]);
    let pair = shorter.tokenize_pair("ab", "b");
    assert_eq!(&pair.input_ids[..pair.real_length], &[2, 5, 6, 3, 8, 3]);
    assert_eq!(
        &pair.token_type_ids[..pair.real_length],
        &[0, 0, 0, 0, 1, 1]
    );
}

#[test]
fn wordpiece_continuation_prefix_effective_settings() {
    use crate::common::TokenizerPolicyValue as V;

    let prefix_setting = |tokenizer: &WordPieceTokenizer| {
        tokenizer
            .embedding_settings()
            .unwrap()
            .into_iter()
            .find(|(key, _)| key == "continuing_subword_prefix")
            .map(|(_, value)| value)
    };
    let custom = WordPieceTokenizer::from_tokenizer_json_str(CONTINUATION_PREFIX_JSON).unwrap();
    let empty = WordPieceTokenizer::from_tokenizer_json_str(EMPTY_PREFIX_JSON).unwrap();
    let default =
        WordPieceTokenizer::from_str("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\n##b").unwrap();
    assert_eq!(prefix_setting(&default), None);
    assert_eq!(prefix_setting(&custom), Some(V::String("@@".into())));
    assert_eq!(prefix_setting(&empty), Some(V::String(String::new())));
    for tokenizer in [&custom, &empty] {
        assert_eq!(
            prefix_setting(tokenizer),
            prefix_setting(&tokenizer.with_max_seq_len(16))
        );
        assert_eq!(
            tokenizer.embedding_settings(),
            tokenizer.with_max_seq_len(512).embedding_settings()
        );
        assert_ne!(
            tokenizer.embedding_settings(),
            tokenizer.with_max_seq_len(16).embedding_settings()
        );
    }
}

#[test]
fn wordpiece_continuation_prefix_requires_string_when_present() {
    for value in ["null", "7"] {
        let json = CONTINUATION_PREFIX_JSON.replacen(r#""@@""#, value, 1);
        let error = WordPieceTokenizer::from_tokenizer_json_str(&json).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("WordPiece continuing_subword_prefix must be a string")
        );
    }
}
