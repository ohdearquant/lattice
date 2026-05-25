//! End-to-end tokenizer parity test for the embedding service path (P0-E1).
//!
//! Confirms that the tokenizer used by the embedding service produces the same
//! token IDs as the HF reference tokenizer.  This is the "tokenizer fixes flow
//! through to the embed layer" gate: if this passes, the fixes shipped in
//! impl-tokenizer-fixes (SentencePiece BOS/EOS, Qwen BPE EOS, AddedToken
//! longest-match) are correctly wired through the `BertModel` load path that
//! `NativeEmbeddingService` uses internally.
//!
//! Tests operate at the **tokenized-ID level** so no model weight files are
//! required.  Each test loads the tokenizer from the HF snapshot directory
//! (the same path `BertModel::from_pretrained` resolves to), calls
//! `tokenize()`, and asserts the IDs match the golden HF reference values
//! from `audit_tokenizer_parity.rs`.
//!
//! Tests are skipped (not failed) when the HF snapshot is absent, matching
//! the convention used by `audit_tokenizer_parity.rs`.
//!
//! Run:
//!   cargo test -p lattice-embed --test tokenizer_parity_e2e

use lattice_inference::{Tokenizer, load_tokenizer};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// HF snapshot path helpers — match audit_tokenizer_parity.rs
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Test helper
// ---------------------------------------------------------------------------

struct Case {
    input: &'static str,
    expected: &'static [u32],
}

/// Load a tokenizer from `dir` via the same code path `BertModel::from_pretrained`
/// uses, then assert token IDs match HF reference values.
fn check_embed_service_tokenizer_parity(label: &str, tok: &dyn Tokenizer, cases: &[Case]) {
    let mut failures = 0;
    for (i, c) in cases.iter().enumerate() {
        let out = tok.tokenize(c.input);
        let actual = &out.input_ids[..out.real_length];
        if actual != c.expected {
            failures += 1;
            eprintln!(
                "PARITY FAIL [{label}] case {}: input={:?}\n  hf_expected: {:?}\n  lattice_got:  {:?}",
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
        "[{label}] {failures}/{} tokenizer-parity cases failed — see stderr",
        cases.len()
    );
}

// ---------------------------------------------------------------------------
// BGE WordPiece — confirms AddedToken longest-match fix (P1-T3) flows through
// ---------------------------------------------------------------------------

/// P0-E1: BGE WordPiece tokenizer used by NativeEmbeddingService produces
/// correct IDs, including AddedToken special tokens as whole units.
///
/// Reference IDs match `bge_small_en_v15_wordpiece_parity` in
/// `audit_tokenizer_parity.rs`.  Both tests must agree; if either fails the
/// embed path and the raw tokenizer path have diverged.
#[test]
fn embed_service_bge_tokenizer_parity() {
    let dir = bge_dir();
    if !dir.exists() {
        eprintln!(
            "SKIP embed_service_bge_tokenizer_parity: {} not found",
            dir.display()
        );
        return;
    }

    let tok = load_tokenizer(&dir).expect("load bge tokenizer");
    check_embed_service_tokenizer_parity(
        "embed-path / bge-small-en-v1.5 (WordPiece)",
        tok.as_ref(),
        &[
            Case {
                input: "Hello, world!",
                expected: &[101, 7592, 1010, 2088, 999, 102],
            },
            Case {
                input: "",
                expected: &[101, 102],
            },
            // P1-T3 fix: [CLS] and [SEP] in input text matched as AddedToken units.
            Case {
                input: "[CLS] special tokens [SEP]",
                expected: &[101, 101, 2569, 19204, 2015, 102, 102],
            },
            Case {
                input: "This is a longer sentence with multiple tokens to verify sequential encoding.",
                expected: &[
                    101, 2023, 2003, 1037, 2936, 6251, 2007, 3674, 19204, 2015, 2000, 20410, 25582,
                    17181, 1012, 102,
                ],
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

// ---------------------------------------------------------------------------
// E5 SentencePiece — confirms BOS/EOS injection (P0-T1) flows through
// ---------------------------------------------------------------------------

/// P0-E1: E5 SentencePiece tokenizer used by NativeEmbeddingService wraps every
/// sequence with BOS id=0 and EOS id=2 (P0-T1 fix).
///
/// Before the fix: 0/9 cases passed (BOS/EOS absent).
/// After the fix: 9/9 cases pass.  These cases are a subset of those in
/// `multilingual_e5_small_sentencepiece_parity` in `audit_tokenizer_parity.rs`.
#[test]
fn embed_service_e5_tokenizer_parity() {
    let dir = e5_dir();
    if !dir.exists() {
        eprintln!(
            "SKIP embed_service_e5_tokenizer_parity: {} not found",
            dir.display()
        );
        return;
    }

    let tok = load_tokenizer(&dir).expect("load e5 tokenizer");
    check_embed_service_tokenizer_parity(
        "embed-path / multilingual-e5-small (SentencePiece)",
        tok.as_ref(),
        &[
            // BOS (0) + EOS (2) framing on normal text.
            Case {
                input: "Hello, world!",
                expected: &[0, 35378, 4, 8999, 38, 2],
            },
            // Empty input: BOS + EOS only.
            Case {
                input: "",
                expected: &[0, 2],
            },
            Case {
                input: " leading space",
                expected: &[0, 105207, 32628, 2],
            },
            Case {
                input: "This is a longer sentence with multiple tokens to verify sequential encoding.",
                expected: &[
                    0, 3293, 83, 10, 51713, 149357, 678, 48716, 47, 84694, 47, 493, 40383, 243228,
                    289, 22, 587, 6238, 5, 2,
                ],
            },
            Case {
                input: "https://example.com/path?query=value",
                expected: &[
                    0, 3975, 696, 3355, 11, 33209, 5, 277, 64, 128405, 32, 944, 1294, 1369, 27494,
                    13, 2,
                ],
            },
        ],
    );
}

// ---------------------------------------------------------------------------
// Qwen BPE — confirms EOS injection (P0-T2) flows through
// ---------------------------------------------------------------------------

/// P0-E1: Qwen BPE tokenizer used by NativeEmbeddingService appends EOS
/// id=151643 (P0-T2 fix).
///
/// Before the fix: 0/10 cases passed (EOS absent).
/// After the fix: 10/10 cases pass.  These cases are a subset of those in
/// `qwen3_embedding_0_6b_bpe_parity` in `audit_tokenizer_parity.rs`.
#[test]
fn embed_service_qwen_tokenizer_parity() {
    let dir = qwen_dir();
    if !dir.exists() {
        eprintln!(
            "SKIP embed_service_qwen_tokenizer_parity: {} not found",
            dir.display()
        );
        return;
    }

    let tok = load_tokenizer(&dir).expect("load qwen tokenizer");
    check_embed_service_tokenizer_parity(
        "embed-path / qwen3-embedding-0.6b (BPE)",
        tok.as_ref(),
        &[
            // EOS (151643) appended.
            Case {
                input: "Hello, world!",
                expected: &[9707, 11, 1879, 0, 151643],
            },
            // Empty input: just EOS.
            Case {
                input: "",
                expected: &[151643],
            },
            Case {
                input: " leading space",
                expected: &[6388, 3550, 151643],
            },
            Case {
                input: "This is a longer sentence with multiple tokens to verify sequential encoding.",
                expected: &[
                    1986, 374, 264, 5021, 11652, 448, 5248, 11211, 311, 10146, 51000, 11170, 13,
                    151643,
                ],
            },
            Case {
                input: "https://example.com/path?query=value",
                expected: &[2428, 1110, 8687, 905, 50976, 30, 1631, 46538, 151643],
            },
        ],
    );
}
