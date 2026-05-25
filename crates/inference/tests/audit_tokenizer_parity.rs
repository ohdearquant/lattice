/// Tokenizer parity audit: compares lattice token IDs against HF reference IDs
/// generated with `from tokenizers import Tokenizer; t = Tokenizer.from_pretrained(MODEL)`.
///
/// Run: cargo test -p lattice-inference --test audit_tokenizer_parity
///
/// Requires the three HF model snapshots present in ~/.cache/huggingface/hub/.
/// If a model directory is absent, its tests are skipped (not failed) so the
/// test file can live in CI without downloading models.
use std::path::PathBuf;

use lattice_inference::{Tokenizer, load_tokenizer};

fn hf_cache() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home).join(".cache/huggingface/hub")
}

fn bge_dir() -> PathBuf {
    hf_cache()
        .join("models--BAAI--bge-small-en-v1.5")
        .join("snapshots")
        .join("5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
}

fn e5_dir() -> PathBuf {
    hf_cache()
        .join("models--intfloat--multilingual-e5-small")
        .join("snapshots")
        .join("614241f622f53c4eeff9890bdc4f31cfecc418b3")
}

fn qwen_dir() -> PathBuf {
    hf_cache()
        .join("models--Qwen--Qwen3-Embedding-0.6B")
        .join("snapshots")
        .join("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
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
    if !dir.exists() {
        eprintln!("SKIP bge-small-en-v1.5: {}", dir.display());
        return;
    }
    let tok = load_tokenizer(&dir).expect("load bge tokenizer");
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
        ],
    );
}

#[test]
fn multilingual_e5_small_sentencepiece_parity() {
    let dir = e5_dir();
    if !dir.exists() {
        eprintln!("SKIP multilingual-e5-small: {}", dir.display());
        return;
    }
    let tok = load_tokenizer(&dir).expect("load e5 tokenizer");
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
    if !dir.exists() {
        eprintln!("SKIP Qwen3-Embedding-0.6B: {}", dir.display());
        return;
    }
    let tok = load_tokenizer(&dir).expect("load qwen tokenizer");
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
