/// Tokenizer parity audit: compares lattice token IDs against HF reference IDs
/// generated with `from tokenizers import Tokenizer; t = Tokenizer.from_pretrained(MODEL)`.
///
/// Run: cargo test -p lattice-inference --test audit_tokenizer_parity
///
/// Tokenizer fixtures are committed under crates/inference/tests/fixtures/tokenizers/
/// so these tests run on a clean checkout without downloading model weights.
use std::path::PathBuf;

use lattice_inference::{Tokenizer, load_tokenizer};

fn tokenizer_fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tokenizers")
}

fn bge_dir() -> PathBuf {
    tokenizer_fixture_root().join("bge-small-en-v1.5")
}

fn e5_dir() -> PathBuf {
    tokenizer_fixture_root().join("multilingual-e5-small")
}

fn qwen_dir() -> PathBuf {
    tokenizer_fixture_root().join("qwen3-embedding-0.6b")
}

fn gpt2_raw_dir() -> PathBuf {
    tokenizer_fixture_root().join("gpt2-raw-bytelevel")
}

struct Case {
    input: &'static str,
    expected: &'static [u32],
}

fn check_parity(label: &str, tok: &dyn Tokenizer, cases: &[Case]) {
    let mut failures = 0;
    for (i, c) in cases.iter().enumerate() {
        let out = tok.tokenize(c.input);
        let actual = &out.input_ids[..out.real_length];
        if actual != c.expected {
            failures += 1;
            eprintln!(
                "PARITY FAIL [{label}] case {}: input={:?}\n  hf_expected: {:?}\n  lattice_got: {:?}",
                i + 1,
                c.input,
                c.expected,
                actual,
            );
        }
    }
    assert_eq!(
        failures,
        0,
        "[{label}] {failures}/{} parity cases failed — see stderr for details",
        cases.len()
    );
}

// HF reference IDs collected with tokenizers==0.23.1 on 2026-05-25.
// Command: uv run python -c "from tokenizers import Tokenizer; t = Tokenizer.from_pretrained('<MODEL>'); print(t.encode('<INPUT>').ids)"

#[test]
fn bge_small_en_v15_wordpiece_parity() {
    let dir = bge_dir();
    let tok = load_tokenizer(&dir)
        .unwrap_or_else(|e| panic!("load bge tokenizer fixture from {}: {e}", dir.display()));
    check_parity(
        "bge-small-en-v1.5 (WordPiece)",
        tok.as_ref(),
        &[
            Case {
                input: "Hello, world!",
                expected: &[101, 7592, 1010, 2088, 999, 102],
            },
            Case {
                input: " leading space",
                expected: &[101, 2877, 2686, 102],
            },
            Case {
                input: "cafe resume naive",
                expected: &[101, 7668, 13746, 15743, 102],
            },
            Case {
                input: "",
                expected: &[101, 102],
            },
            Case {
                input: "This is a longer sentence with multiple tokens to verify sequential encoding.",
                expected: &[
                    101, 2023, 2003, 1037, 2936, 6251, 2007, 3674, 19204, 2015, 2000, 20410, 25582,
                    17181, 1012, 102,
                ],
            },
            Case {
                input: "[CLS] special tokens [SEP]",
                expected: &[101, 101, 2569, 19204, 2015, 102, 102],
            },
            Case {
                input: "123.456",
                expected: &[101, 13138, 1012, 3429, 2575, 102],
            },
            Case {
                input: "multi\nline\ntext",
                expected: &[101, 4800, 2240, 3793, 102],
            },
            Case {
                input: "https://example.com/path?query=value",
                expected: &[
                    101, 16770, 1024, 1013, 1013, 2742, 1012, 4012, 1013, 4130, 1029, 23032, 1027,
                    3643, 102,
                ],
            },
            // CJK + Hiragana/Katakana: voiced chars (で→て, etc.) must be
            // stripped of their combining dakuten (U+3099) to match HF
            // BertNormalizer strip_accents behaviour on NFD input.
            Case {
                input: "短い日本語のテストです。",
                expected: &[
                    101, 100, 1647, 1864, 1876, 1950, 1671, 30239, 30233, 30240, 30191, 30184,
                    1636, 102,
                ],
            },
            // Voiced hiragana only — every character has a dakuten: が→か, ぎ→き, etc.
            Case {
                input: "がぎぐげご",
                expected: &[101, 1651, 30178, 30179, 30180, 30181, 102],
            },
        ],
    );
}

#[test]
fn multilingual_e5_small_sentencepiece_parity() {
    let dir = e5_dir();
    let tok = load_tokenizer(&dir)
        .unwrap_or_else(|e| panic!("load e5 tokenizer fixture from {}: {e}", dir.display()));
    check_parity(
        "multilingual-e5-small (SentencePiece/Unigram)",
        tok.as_ref(),
        &[
            Case {
                input: "Hello, world!",
                expected: &[0, 35378, 4, 8999, 38, 2],
            },
            Case {
                input: " leading space",
                expected: &[0, 105207, 32628, 2],
            },
            Case {
                input: "cafe resume naive",
                expected: &[0, 43185, 138755, 24, 5844, 2],
            },
            Case {
                input: "",
                expected: &[0, 2],
            },
            Case {
                input: "This is a longer sentence with multiple tokens to verify sequential encoding.",
                expected: &[
                    0, 3293, 83, 10, 51713, 149357, 678, 48716, 47, 84694, 47, 493, 40383, 243228,
                    289, 22, 587, 6238, 5, 2,
                ],
            },
            Case {
                input: "[CLS] special tokens [SEP]",
                expected: &[
                    0, 378, 441, 19759, 268, 5361, 47, 84694, 378, 294, 21290, 268, 2,
                ],
            },
            Case {
                input: "123.456",
                expected: &[0, 37638, 5, 121317, 2],
            },
            Case {
                input: "multi\nline\ntext",
                expected: &[0, 6024, 13315, 7986, 2],
            },
            Case {
                input: "https://example.com/path?query=value",
                expected: &[
                    0, 3975, 696, 3355, 11, 33209, 5, 277, 64, 128405, 32, 944, 1294, 1369, 27494,
                    13, 2,
                ],
            },
            // Whitespace regression cases — HF ref collected with AutoTokenizer
            // (transformers==4.x) on 2026-05-25.  These cover the trailing-space bug
            // where lattice used to emit an extra ▁ (token 6) before EOS.
            Case {
                input: "   leading whitespace and    multiple    spaces   ",
                expected: &[0, 105207, 35011, 65421, 136, 48716, 32628, 7, 2],
            },
            Case {
                input: "trailing space ",
                expected: &[0, 141037, 214, 32628, 2],
            },
        ],
    );
}

#[test]
fn qwen3_embedding_0_6b_bpe_parity() {
    let dir = qwen_dir();
    let tok = load_tokenizer(&dir)
        .unwrap_or_else(|e| panic!("load qwen tokenizer fixture from {}: {e}", dir.display()));
    check_parity(
        "Qwen3-Embedding-0.6B (BPE)",
        tok.as_ref(),
        &[
            Case {
                input: "Hello, world!",
                expected: &[9707, 11, 1879, 0, 151643],
            },
            Case {
                input: " leading space",
                expected: &[6388, 3550, 151643],
            },
            Case {
                input: "cafe resume naive",
                expected: &[924, 1859, 15688, 49665, 151643],
            },
            Case {
                input: "",
                expected: &[151643],
            },
            Case {
                input: "This is a longer sentence with multiple tokens to verify sequential encoding.",
                expected: &[
                    1986, 374, 264, 5021, 11652, 448, 5248, 11211, 311, 10146, 51000, 11170, 13,
                    151643,
                ],
            },
            Case {
                input: "[CLS] special tokens [SEP]",
                expected: &[58, 87716, 60, 3281, 11211, 508, 81376, 60, 151643],
            },
            Case {
                input: "123.456",
                expected: &[16, 17, 18, 13, 19, 20, 21, 151643],
            },
            Case {
                input: "multi\nline\ntext",
                expected: &[26268, 198, 1056, 198, 1318, 151643],
            },
            Case {
                input: "https://example.com/path?query=value",
                expected: &[2428, 1110, 8687, 905, 50976, 30, 1631, 46538, 151643],
            },
            Case {
                input: "你好世界 مرحبا Привет",
                expected: &[
                    108386, 99489, 23364, 126860, 124671, 79484, 26991, 8178, 151643,
                ],
            },
        ],
    );
}

// (#330) A raw GPT-2-format vocab.json + merges.txt directory has no
// tokenizer.json, so `load_tokenizer` routes it through
// `BpeTokenizer::from_files`, which has no pre_tokenizer metadata to inspect
// and previously defaulted to the hand-rolled `byte_level_pretokenize`
// fallback. That fallback diverges from GPT-2's actual regex-based word
// boundaries, which is the real, unconditional default for every GPT-2-format
// tokenizer HF ships (verified against the real `gpt2` and `roberta-base`
// tokenizer.json files on the Hub — both specify literally
// `"pre_tokenizer": {"type": "ByteLevel", ...}` with no `use_regex` override,
// which defaults to `true` — and against the "slow" `GPT2Tokenizer` Python
// implementation, which applies the identical regex directly).
//
// HF reference IDs collected with tokenizers==0.22.2 on 2026-07-01, loading
// the same vocab.json/merges.txt committed in fixtures/gpt2-raw-bytelevel/:
//   from tokenizers import Tokenizer
//   from tokenizers.models import BPE
//   from tokenizers.pre_tokenizers import ByteLevel
//   tok = Tokenizer(BPE(vocab=<vocab.json>, merges=<merges.txt>, unk_token=None))
//   tok.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
//   tok.encode(<INPUT>, add_special_tokens=True).ids
#[test]
fn gpt2_raw_vocab_bytelevel_fallback_parity() {
    let dir = gpt2_raw_dir();
    let tok = load_tokenizer(&dir)
        .unwrap_or_else(|e| panic!("load gpt2 raw-vocab fixture from {}: {e}", dir.display()));
    check_parity(
        "gpt2-raw-bytelevel (BPE, no tokenizer.json)",
        tok.as_ref(),
        &[
            Case {
                input: "Hello, world!",
                expected: &[15496, 11, 995, 0],
            },
            Case {
                input: "hello world",
                expected: &[31373, 995],
            },
            // Whitespace-run boundary (#330 finding 1). Real GPT-2's merge
            // table has no rule pairing the byte-mapped newline/tab char with
            // a following letter, so the wrong 2-piece fallback segmentation
            // ["hello", "\nworld"] happens to BPE-decompose to the same ids
            // as the correct 3-piece ["hello", "\n", "world"] for THIS vocab
            // — these cases exercise the code path but do not by themselves
            // discriminate the bug at the id level (see bpe.rs unit tests
            // `gpt4_regex_pretokenize_matches_hf_on_whitespace_runs` for a
            // piece-level (pre-BPE-merge) golden that does).
            Case {
                input: "hello\nworld",
                expected: &[31373, 198, 6894],
            },
            Case {
                input: "hello\tworld",
                expected: &[31373, 197, 6894],
            },
            Case {
                input: "hello  world",
                expected: &[31373, 220, 995],
            },
            // Contraction boundary (#330 finding 2): the fixed HF contraction
            // set is `'s 't 're 've 'm 'll 'd`. The prior fallback primed on
            // `'` and swallowed the whole trailing letter run, fusing "'st"
            // into one piece instead of splitting "'s" + "t". This DOES
            // discriminate at the id level: pre-fix lattice returned
            // [64, 6, 301] ("a", "'", "st") instead of HF's [64, 338, 83]
            // ("a", "'s", "t") — verified empirically against this exact
            // fixture before implementing the fix.
            Case {
                input: "a'st",
                expected: &[64, 338, 83],
            },
            Case {
                input: "wasn't",
                expected: &[9776, 77, 470],
            },
            Case {
                input: "don't wasn't I'm",
                expected: &[9099, 470, 2492, 470, 314, 1101],
            },
        ],
    );
}
