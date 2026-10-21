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
