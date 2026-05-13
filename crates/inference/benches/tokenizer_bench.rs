//! Criterion benchmarks for tokenizer throughput.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use lattice_inference::{BpeTokenizer, Tokenizer, WordPieceTokenizer};
use std::collections::HashMap;

#[derive(Clone)]
struct BaselineWordPiece {
    vocab: HashMap<String, u32>,
    unk_id: u32,
    cls_id: u32,
    sep_id: u32,
}

impl BaselineWordPiece {
    fn from_vocab_text(vocab_text: &str) -> Self {
        let mut vocab = HashMap::new();
        for (idx, line) in vocab_text.lines().enumerate() {
            vocab.insert(line.trim().to_string(), idx as u32);
        }
        Self {
            unk_id: *vocab.get("[UNK]").unwrap(),
            cls_id: *vocab.get("[CLS]").unwrap(),
            sep_id: *vocab.get("[SEP]").unwrap(),
            vocab,
        }
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.cls_id];
        for word in text.to_lowercase().split_whitespace() {
            let chars: Vec<char> = word.chars().collect();
            let mut start = 0usize;
            while start < chars.len() {
                let mut end = chars.len();
                let mut found = None;
                while start < end {
                    let piece = if start == 0 {
                        chars[start..end].iter().collect::<String>()
                    } else {
                        let mut value = String::from("##");
                        value.extend(chars[start..end].iter());
                        value
                    };
                    if let Some(&id) = self.vocab.get(piece.as_str()) {
                        found = Some((id, end));
                        break;
                    }
                    end -= 1;
                }
                if let Some((id, end)) = found {
                    ids.push(id);
                    start = end;
                } else {
                    ids.push(self.unk_id);
                    start += 1;
                }
            }
        }
        ids.push(self.sep_id);
        ids
    }
}

fn synthetic_wordpiece_vocab() -> String {
    let max_id = 32_767usize;
    let mut tokens = (0..=max_id)
        .map(|i| format!("[unused{}]", i))
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
        (16_516, "token"),
        (16_517, "##izer"),
        (16_518, "throughput"),
        (16_519, "cache"),
        (16_520, "##able"),
    ];

    for (idx, token) in entries {
        tokens[idx] = token.to_string();
    }

    tokens.join("\n")
}

fn synthetic_bpe() -> BpeTokenizer {
    let mut vocab = HashMap::new();
    let base = [
        "h", "e", "l", "o", "Ġ", "w", "r", "d", "t", "q", "u", "i", "c", "k", "b", "n", "f", "x",
        "a", "m", "s", "p", "g",
    ];
    for (idx, token) in base.into_iter().enumerate() {
        vocab.insert(token.to_string(), idx as u32);
    }

    let mut next_id = vocab.len() as u32;
    for token in [
        "he",
        "hel",
        "hell",
        "hello",
        "Ġw",
        "Ġwo",
        "Ġwor",
        "Ġworl",
        "Ġworld",
        "Ġq",
        "Ġqu",
        "Ġqui",
        "Ġquic",
        "Ġquick",
        "Ġb",
        "Ġbr",
        "Ġbro",
        "Ġbrow",
        "Ġbrown",
        "Ġf",
        "Ġfo",
        "Ġfox",
        "<|endoftext|>",
    ] {
        vocab.insert(token.to_string(), next_id);
        next_id += 1;
    }

    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
        ("hel".to_string(), "l".to_string()),
        ("hell".to_string(), "o".to_string()),
        ("Ġ".to_string(), "w".to_string()),
        ("Ġw".to_string(), "o".to_string()),
        ("Ġwo".to_string(), "r".to_string()),
        ("Ġwor".to_string(), "l".to_string()),
        ("Ġworl".to_string(), "d".to_string()),
        ("Ġ".to_string(), "q".to_string()),
        ("Ġq".to_string(), "u".to_string()),
        ("Ġqu".to_string(), "i".to_string()),
        ("Ġqui".to_string(), "c".to_string()),
        ("Ġquic".to_string(), "k".to_string()),
        ("Ġ".to_string(), "b".to_string()),
        ("Ġb".to_string(), "r".to_string()),
        ("Ġbr".to_string(), "o".to_string()),
        ("Ġbro".to_string(), "w".to_string()),
        ("Ġbrow".to_string(), "n".to_string()),
        ("Ġ".to_string(), "f".to_string()),
        ("Ġf".to_string(), "o".to_string()),
        ("Ġfo".to_string(), "x".to_string()),
    ];

    BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap()
}

fn repeated_text(tokens: usize) -> String {
    let base = [
        "hello",
        "world",
        "the",
        "quick",
        "brown",
        "fox",
        "embedding",
        "unrecognizable",
        "transforming",
        "ai",
        "tokenizer",
        "throughput",
        "cacheable",
    ];
    let mut out = String::new();
    for idx in 0..tokens {
        if idx > 0 {
            out.push(' ');
        }
        out.push_str(base[idx % base.len()]);
    }
    out
}

fn tokenizer_bench(c: &mut Criterion) {
    let vocab_text = synthetic_wordpiece_vocab();
    let baseline = BaselineWordPiece::from_vocab_text(&vocab_text);
    let trie = WordPieceTokenizer::from_str(&vocab_text).unwrap();
    let bpe = synthetic_bpe();

    let mut wp_group = c.benchmark_group("wordpiece_single");
    for len in [16usize, 64, 256, 1024] {
        let text = repeated_text(len);
        wp_group.bench_with_input(
            BenchmarkId::new("baseline_hashmap", len),
            &text,
            |b, text| {
                b.iter(|| black_box(baseline.tokenize(black_box(text))));
            },
        );
        wp_group.bench_with_input(
            BenchmarkId::new("double_array_trie", len),
            &text,
            |b, text| {
                b.iter(|| black_box(trie.tokenize(black_box(text))));
            },
        );
    }
    wp_group.finish();

    let mut bpe_group = c.benchmark_group("bpe_single");
    for len in [16usize, 64, 256, 1024] {
        let text = repeated_text(len);
        bpe_group.bench_with_input(BenchmarkId::new("byte_level_bpe", len), &text, |b, text| {
            b.iter(|| black_box(bpe.tokenize(black_box(text))));
        });
    }
    bpe_group.finish();

    let mut batch_group = c.benchmark_group("wordpiece_batch");
    for batch in [1usize, 8, 32, 128] {
        let texts = (0..batch).map(|_| repeated_text(64)).collect::<Vec<_>>();
        let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        batch_group.bench_with_input(
            BenchmarkId::new("double_array_trie", batch),
            &refs,
            |b, refs| {
                b.iter(|| black_box(trie.tokenize_batch(black_box(refs))));
            },
        );
    }
    batch_group.finish();
}

criterion_group!(benches, tokenizer_bench);
criterion_main!(benches);
