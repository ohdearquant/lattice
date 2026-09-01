//! PaddleOCR-VL (ERNIE-family) tokenizer goldens.
//!
//! Two structural claims, mirroring `gemma4_tokenizer_goldens_test.rs`:
//!
//! 1. `GemmaBpeTokenizer::from_ernie_tokenizer_json_str` reproduces the exact
//!    HF `tokenizers` fast-path ids and decode output for the pinned
//!    `PaddlePaddle/PaddleOCR-VL-1.6` `tokenizer.json` (goldens generated
//!    with `tokenizers.Tokenizer.from_file` against this fixture).
//! 2. The path is additive and explicitly selected: the Gemma constructor
//!    rejects the ERNIE fixture (Sequence-wrapped normalizer, null
//!    pre-tokenizer) and the ERNIE constructor rejects the Gemma fixture
//!    (bare Replace normalizer, literal-Split pre-tokenizer), so neither
//!    shape can silently load through the other's validator.
//!
//! Corpus emphasis, driven by the OCR workload: single digits '0'-'9' are
//! non-special ADDED tokens (ids 3-12), so digit-bearing text must split
//! digits out before BPE and render them on decode; `<|LOC_n|>`
//! bbox-coordinate tokens (ids 100297-101300) are likewise non-special added
//! tokens that must survive decode, while the 22 `special: true` markers
//! (image/audio/LOC structure) are dropped, matching
//! `skip_special_tokens=True`.

use lattice_inference::tokenizer::common::Tokenizer;
use lattice_inference::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::LazyLock;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/paddleocr_vl/tokenizer")
}

fn read_fixture(name: &str) -> String {
    let path = fixture_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()))
}

static TOKENIZER_JSON: LazyLock<String> = LazyLock::new(|| read_fixture("tokenizer.json"));
static TOKENIZER: LazyLock<GemmaBpeTokenizer> = LazyLock::new(|| {
    GemmaBpeTokenizer::from_ernie_tokenizer_json_str(&TOKENIZER_JSON)
        .expect("pinned PaddleOCR-VL tokenizer.json must load through the ERNIE path")
});

#[derive(Deserialize)]
struct CorpusCase {
    id: String,
    #[allow(dead_code)]
    category: String,
    text: String,
    ids: Vec<u32>,
    #[allow(dead_code)]
    tokens: Vec<String>,
    decoded: String,
}

fn load_corpus() -> Vec<CorpusCase> {
    serde_json::from_str(&read_fixture("corpus_goldens.json")).expect("valid corpus_goldens.json")
}

/// Raw (unpadded) tokenize-to-ids for golden comparison, mirroring the
/// helper in `gemma4_tokenizer_goldens_test.rs`.
fn tokenize_ids(text: &str) -> Vec<u32> {
    let padded = TOKENIZER.tokenize(text);
    padded.input_ids[..padded.real_length].to_vec()
}

#[test]
fn corpus_goldens_encode_matches_hf_ids() {
    let corpus = load_corpus();
    assert!(
        corpus.len() >= 17,
        "corpus shrank to {} cases — goldens file truncated?",
        corpus.len()
    );
    for case in &corpus {
        let ids = tokenize_ids(&case.text);
        assert_eq!(
            ids, case.ids,
            "case {:?} (text {:?}): got {:?}, HF golden {:?}",
            case.id, case.text, ids, case.ids
        );
    }
}

#[test]
fn corpus_goldens_decode_matches_hf_skip_specials() {
    for case in load_corpus() {
        let decoded = TOKENIZER
            .decode(&case.ids)
            .expect("decode always returns Some");
        assert_eq!(
            decoded, case.decoded,
            "case {:?} (ids {:?}): decoded {:?}, HF golden {:?}",
            case.id, case.ids, decoded, case.decoded
        );
    }
}

#[test]
fn round_trip_cases_round_trip_byte_for_byte() {
    // A case round-trips exactly when the HF golden itself does (its text
    // contains no `special: true` markers, which drop on decode). Deriving
    // the set from the golden instead of a hand-kept category list keeps the
    // predicate self-describing; the count assertion keeps it from silently
    // becoming vacuous if the corpus changes.
    let round_trip_cases: Vec<CorpusCase> = load_corpus()
        .into_iter()
        .filter(|case| case.decoded == case.text)
        .collect();
    assert!(
        round_trip_cases.len() >= 12,
        "only {} round-trip cases — corpus or decode goldens changed shape",
        round_trip_cases.len()
    );
    for case in &round_trip_cases {
        let decoded = TOKENIZER
            .decode(&case.ids)
            .expect("decode always returns Some");
        assert_eq!(
            decoded, case.text,
            "case {:?} did not round-trip byte-for-byte",
            case.id
        );
    }
}

/// Digits are non-special added tokens at ids 3-12 ('0' = 3 .. '9' = 12);
/// OCR output is digit-dense, so pin the mapping itself rather than only
/// exercising it through corpus cases.
#[test]
fn digits_map_to_added_token_ids_three_through_twelve() {
    for (digit, expected_id) in ('0'..='9').zip(3u32..=12) {
        let ids = tokenize_ids(&digit.to_string());
        assert_eq!(
            ids,
            vec![expected_id],
            "digit {digit:?} must tokenize to the single added-token id {expected_id}"
        );
    }
}

#[test]
fn image_markers_tokenize_to_single_ids_and_drop_on_decode() {
    let start = tokenize_ids("<|IMAGE_START|>");
    let placeholder = tokenize_ids("<|IMAGE_PLACEHOLDER|>");
    let end = tokenize_ids("<|IMAGE_END|>");
    for (name, ids) in [
        ("<|IMAGE_START|>", &start),
        ("<|IMAGE_PLACEHOLDER|>", &placeholder),
        ("<|IMAGE_END|>", &end),
    ] {
        assert_eq!(ids.len(), 1, "{name} must be a single token, got {ids:?}");
    }
    let all = [start[0], placeholder[0], end[0]];
    assert_eq!(
        TOKENIZER.decode(&all).expect("decode always returns Some"),
        "",
        "special image markers must be dropped on decode (skip_special_tokens)"
    );
}

#[derive(Deserialize)]
struct ManifestFile {
    bytes: u64,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    source_repo: String,
    revision: String,
    files: std::collections::HashMap<String, ManifestFile>,
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

#[test]
fn manifest_matches_committed_fixture_bytes() {
    let manifest: Manifest =
        serde_json::from_str(&read_fixture("manifest.json")).expect("valid manifest.json");
    assert_eq!(manifest.source_repo, "PaddlePaddle/PaddleOCR-VL-1.6");
    assert_eq!(
        manifest.revision,
        "c5630abae1d940eafe0697512a0325494b02ab42"
    );

    for (name, entry) in &manifest.files {
        let bytes = std::fs::read(fixture_dir().join(name))
            .unwrap_or_else(|e| panic!("failed to read committed {name}: {e}"));
        assert_eq!(bytes.len() as u64, entry.bytes, "{name} byte count drifted");
        assert_eq!(sha256_hex(&bytes), entry.sha256, "{name} sha256 drifted");
    }
}

// Additive-path negative tests: each validator rejects the other's shape.

#[test]
fn gemma_constructor_rejects_ernie_tokenizer_json() {
    let err = GemmaBpeTokenizer::from_tokenizer_json_str(&TOKENIZER_JSON)
        .expect_err("Gemma validator must reject the ERNIE shape");
    let message = err.to_string();
    assert!(
        message.contains("normalizer"),
        "rejection must name the diverging stage; got {message:?}"
    );
}

#[test]
fn ernie_constructor_rejects_gemma_tokenizer_json() {
    let gemma_json = {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gemma4/tokenizer/tokenizer.json");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read gemma fixture {}: {e}", path.display()))
    };
    let err = GemmaBpeTokenizer::from_ernie_tokenizer_json_str(&gemma_json)
        .expect_err("ERNIE validator must reject the Gemma shape");
    let message = err.to_string();
    assert!(
        message.contains("normalizer") || message.contains("pre_tokenizer"),
        "rejection must name the diverging stage; got {message:?}"
    );
}
