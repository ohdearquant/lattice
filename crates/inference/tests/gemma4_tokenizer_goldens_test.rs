//! Fixture-only gate for the Gemma 4 BPE tokenizer (ADR-082 Stage 1, G17).
//!
//! **Offline-only.** Unlike `scripts/gemma4_tokenizer_goldens.py` (which
//! fetches `tokenizer.json`/`tokenizer_config.json`/`chat_template.jinja`
//! over HTTPS), this test never touches the network — it loads the fixtures
//! already committed at `tests/fixtures/gemma4/tokenizer/`, produced by a
//! single real run of that script against the pinned checkpoint revision
//! (`google/gemma-4-E2B-it` @ `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`,
//! HF `tokenizers==0.22.2` / `jinja2==3.1.6` — see `manifest.json`).
//!
//! Three things are asserted:
//! 1. `GemmaBpeTokenizer` (the new, additive path) reproduces the exact HF
//!    token IDs for every corpus case and the 2-turn chat-template
//!    rendering, plus a plain-text decode round trip.
//! 2. The **existing** Qwen-oriented `BpeTokenizer::from_tokenizer_json_str`
//!    still rejects this same `tokenizer.json` — proving the new path is
//!    additive, not a silent widening of the general loader (ADR-082's
//!    mandated negative test).

use lattice_inference::{BpeTokenizer, GemmaBpeTokenizer, Tokenizer};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::LazyLock;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4/tokenizer")
}

fn read_fixture(name: &str) -> String {
    let path = fixture_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()))
}

static TOKENIZER_JSON: LazyLock<String> = LazyLock::new(|| read_fixture("tokenizer.json"));
static TOKENIZER: LazyLock<GemmaBpeTokenizer> =
    LazyLock::new(|| GemmaBpeTokenizer::from_tokenizer_json_str(&TOKENIZER_JSON).unwrap());

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

#[derive(Deserialize)]
struct ChatTemplateGolden {
    #[allow(dead_code)]
    description: String,
    #[allow(dead_code)]
    messages: serde_json::Value,
    rendered_text: String,
    ids: Vec<u32>,
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

fn load_corpus() -> Vec<CorpusCase> {
    serde_json::from_str(&read_fixture("corpus_goldens.json")).expect("valid corpus_goldens.json")
}

fn load_chat_golden() -> ChatTemplateGolden {
    serde_json::from_str(&read_fixture("chat_template_golden.json"))
        .expect("valid chat_template_golden.json")
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------

#[test]
fn manifest_matches_committed_tokenizer_json_bytes() {
    let manifest: Manifest =
        serde_json::from_str(&read_fixture("manifest.json")).expect("valid manifest.json");
    assert_eq!(manifest.source_repo, "google/gemma-4-E2B-it");
    assert_eq!(
        manifest.revision,
        "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"
    );

    for (name, entry) in &manifest.files {
        let bytes = std::fs::read(fixture_dir().join(name))
            .unwrap_or_else(|e| panic!("failed to read committed {name}: {e}"));
        assert_eq!(bytes.len() as u64, entry.bytes, "{name} byte count drifted");
        assert_eq!(sha256_hex(&bytes), entry.sha256, "{name} sha256 drifted");
    }
}

// ---------------------------------------------------------------------
// Corpus goldens (exact HF token-ID equality)
// ---------------------------------------------------------------------

#[test]
fn corpus_goldens_match_exact_hf_token_ids() {
    let cases = load_corpus();
    assert!(cases.len() >= 20, "expected a substantial declared corpus");

    let mut categories = std::collections::HashSet::new();
    for case in &cases {
        categories.insert(case.category.clone());
        let ids = TOKENIZER.tokenize_ids_for_test(&case.text);
        assert_eq!(
            ids, case.ids,
            "case {:?} (text={:?}) produced {:?}, expected {:?}",
            case.id, case.text, ids, case.ids
        );
    }

    for required in ["text", "unicode", "whitespace", "byte_fallback", "marker"] {
        assert!(
            categories.contains(required),
            "corpus is missing required category {required:?}"
        );
    }
}

#[test]
fn corpus_goldens_decode_round_trips_plain_text() {
    for case in load_corpus() {
        if case.category == "marker" {
            // Special-token markers are dropped on decode by design
            // (skip_special_tokens semantics) — covered separately below,
            // not a round-trip case.
            continue;
        }
        let decoded = TOKENIZER
            .decode(&case.ids)
            .expect("decode always returns Some");
        assert_eq!(decoded, case.decoded, "case {:?} decode mismatch", case.id);
        assert_eq!(
            decoded, case.text,
            "case {:?} did not round-trip byte-for-byte",
            case.id
        );
    }
}

#[test]
fn image_and_audio_markers_tokenize_to_single_ids() {
    let cases = load_corpus();
    let image = cases
        .iter()
        .find(|c| c.id == "marker_image")
        .expect("marker_image case present");
    let audio = cases
        .iter()
        .find(|c| c.id == "marker_audio")
        .expect("marker_audio case present");
    assert_eq!(image.ids.len(), 1, "<|image|> must be a single token");
    assert_eq!(audio.ids.len(), 1, "<|audio|> must be a single token");
    assert_ne!(image.ids[0], audio.ids[0]);
}

// ---------------------------------------------------------------------
// Chat template (rendered string + token IDs)
// ---------------------------------------------------------------------

#[test]
fn chat_template_2turn_rendering_matches_hf() {
    let golden = load_chat_golden();
    assert!(golden.rendered_text.starts_with("<bos>"));
    assert!(golden.rendered_text.contains("<|turn>user\n"));
    assert!(golden.rendered_text.contains("<|turn>model\n"));

    let ids = TOKENIZER.tokenize_ids_for_test(&golden.rendered_text);
    assert_eq!(
        ids, golden.ids,
        "chat-template rendering did not tokenize to the golden IDs"
    );
}

// ---------------------------------------------------------------------
// Additive-path negative test (ADR-082 Stage 1 mandated gate)
// ---------------------------------------------------------------------

#[test]
fn existing_qwen_bpe_loader_rejects_gemma_tokenizer_json() {
    let result = BpeTokenizer::from_tokenizer_json_str(&TOKENIZER_JSON);
    assert!(
        result.is_err(),
        "existing BpeTokenizer loader must reject Gemma's literal-space \
         Split + \u{2581} normalizer tokenizer.json -- if this now succeeds, \
         the general loader has been silently widened to accept the Gemma \
         shape, which ADR-082 Stage 1 explicitly forbids"
    );
}

#[test]
fn gemma_tokenizer_vocab_size_matches_checkpoint() {
    assert_eq!(TOKENIZER.vocab_size(), 262_144);
}

// Small helper trait to expose raw (unpadded) tokenize-to-ids for golden
// comparison without pulling in the padded `TokenizedInput` shape.
trait TestTokenizeIds {
    fn tokenize_ids_for_test(&self, text: &str) -> Vec<u32>;
}

impl TestTokenizeIds for GemmaBpeTokenizer {
    fn tokenize_ids_for_test(&self, text: &str) -> Vec<u32> {
        let padded = self.tokenize(text);
        padded.input_ids[..padded.real_length].to_vec()
    }
}
