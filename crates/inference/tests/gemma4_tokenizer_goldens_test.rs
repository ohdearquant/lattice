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
//! Four things are asserted:
//! 1. `GemmaBpeTokenizer` (the new, additive path) reproduces the exact HF
//!    token IDs for every corpus case and the 2-turn chat-template
//!    rendering, plus exact byte-fallback decode goldens (including
//!    invalid/incomplete UTF-8 byte runs).
//! 2. The **existing** Qwen-oriented `BpeTokenizer::from_tokenizer_json_str`
//!    still rejects this same `tokenizer.json` — proving the new path is
//!    additive, not a silent widening of the general loader (ADR-082's
//!    mandated negative test).
//! 3. The constructor fails closed on a mutated pre-tokenizer, decoder, or
//!    model field it depends on.
//! 4. The Stage-1 marker-expansion arithmetic (ADR-082 G11/G15/G17) matches
//!    the pinned checkpoint's `processor_config.json`-derived goldens.

use lattice_inference::{
    BpeTokenizer, GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES, GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
    GEMMA4_AUDIO_MAX_SOFT_TOKENS, GEMMA4_AUDIO_MS_PER_SOFT_TOKEN, GEMMA4_AUDIO_SAMPLING_RATE_HZ,
    GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE, GemmaBpeTokenizer, Tokenizer,
    audio_marker_expansion_tokens, image_marker_expansion_tokens,
    total_audio_marker_expansion_tokens,
};
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
struct DecodeCase {
    id: String,
    ids: Vec<u32>,
    decoded: String,
}

#[derive(Deserialize)]
struct ExpansionProvenance {
    image_seq_length: u32,
    audio_ms_per_token: u32,
    audio_seq_length: u32,
    audio_sampling_rate_hz: u32,
    audio_frame_length_samples: u32,
    audio_hop_length_samples: u32,
}

#[derive(Deserialize)]
struct ImageExpansionCase {
    marker_count: usize,
    expected_tokens: usize,
}

#[derive(Deserialize)]
struct AudioExpansionCase {
    durations_ms: Vec<u32>,
    expected_tokens: Vec<u32>,
}

#[derive(Deserialize)]
struct ExpansionGoldens {
    provenance: ExpansionProvenance,
    image_cases: Vec<ImageExpansionCase>,
    audio_cases: Vec<AudioExpansionCase>,
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

fn load_decode_goldens() -> Vec<DecodeCase> {
    serde_json::from_str(&read_fixture("decode_goldens.json")).expect("valid decode_goldens.json")
}

fn load_expansion_goldens() -> ExpansionGoldens {
    serde_json::from_str(&read_fixture("expansion_goldens.json"))
        .expect("valid expansion_goldens.json")
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

// ---------------------------------------------------------------------
// Byte-fallback decode goldens (exact HF decode parity for invalid/
// incomplete UTF-8 byte-fallback runs; ADR-082 Stage 1 major finding)
// ---------------------------------------------------------------------

#[test]
fn byte_fallback_decode_matches_exact_hf_output() {
    let cases = load_decode_goldens();
    assert!(
        cases.len() >= 4,
        "expected the required lone/incomplete/valid/mixed decode cases"
    );
    for case in &cases {
        let decoded = TOKENIZER
            .decode(&case.ids)
            .expect("decode always returns Some");
        assert_eq!(
            decoded, case.decoded,
            "case {:?} (ids={:?}) decoded {:?}, expected {:?}",
            case.id, case.ids, decoded, case.decoded
        );
    }

    // Pin the shape of the required cases directly (not just via the fixture),
    // so a fixture regeneration that silently drops a case still fails loudly.
    let by_id = |id: &str| {
        cases
            .iter()
            .find(|c| c.id == id)
            .unwrap_or_else(|| panic!("missing required decode case {id:?}"))
    };
    assert_eq!(
        by_id("decode_lone_incomplete_lead_byte").decoded,
        "\u{FFFD}"
    );
    assert_eq!(
        by_id("decode_two_byte_incomplete_run").decoded,
        "\u{FFFD}\u{FFFD}"
    );
    assert_eq!(by_id("decode_valid_three_byte_fallback_run").decoded, "叫");
    assert_eq!(
        by_id("decode_incomplete_run_then_ordinary_token").decoded,
        "\u{FFFD}\u{FFFD}a"
    );
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

// ---------------------------------------------------------------------
// Stage-1 marker-expansion arithmetic (ADR-082 G11/G15/G17)
// ---------------------------------------------------------------------

#[test]
fn marker_expansion_arithmetic_matches_processor_config_goldens() {
    let goldens = load_expansion_goldens();

    // The committed Rust constants must not silently drift from the
    // checkpoint's own `processor_config.json` the goldens were derived from.
    assert_eq!(
        GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE, goldens.provenance.image_seq_length,
        "GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE drifted from processor_config.json"
    );
    assert_eq!(
        GEMMA4_AUDIO_MS_PER_SOFT_TOKEN, goldens.provenance.audio_ms_per_token,
        "GEMMA4_AUDIO_MS_PER_SOFT_TOKEN drifted from processor_config.json"
    );
    assert_eq!(
        GEMMA4_AUDIO_MAX_SOFT_TOKENS, goldens.provenance.audio_seq_length,
        "GEMMA4_AUDIO_MAX_SOFT_TOKENS drifted from processor_config.json"
    );
    assert_eq!(
        GEMMA4_AUDIO_SAMPLING_RATE_HZ, goldens.provenance.audio_sampling_rate_hz,
        "GEMMA4_AUDIO_SAMPLING_RATE_HZ drifted from processor_config.json"
    );
    assert_eq!(
        GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES, goldens.provenance.audio_frame_length_samples,
        "GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES drifted from processor_config.json"
    );
    assert_eq!(
        GEMMA4_AUDIO_HOP_LENGTH_SAMPLES, goldens.provenance.audio_hop_length_samples,
        "GEMMA4_AUDIO_HOP_LENGTH_SAMPLES drifted from processor_config.json"
    );

    assert!(goldens.image_cases.iter().any(|c| c.marker_count == 0));
    assert!(goldens.image_cases.iter().any(|c| c.marker_count == 1));
    assert!(goldens.image_cases.iter().any(|c| c.marker_count > 1));
    for case in &goldens.image_cases {
        assert_eq!(
            image_marker_expansion_tokens(case.marker_count),
            case.expected_tokens,
            "image marker_count={} expansion mismatch",
            case.marker_count
        );
    }

    assert!(
        goldens
            .audio_cases
            .iter()
            .any(|c| c.durations_ms.is_empty())
    );
    assert!(
        goldens
            .audio_cases
            .iter()
            .any(|c| c.durations_ms.len() == 1)
    );
    assert!(goldens.audio_cases.iter().any(|c| c.durations_ms.len() > 1));
    for case in &goldens.audio_cases {
        assert_eq!(case.durations_ms.len(), case.expected_tokens.len());
        for (&duration_ms, &expected) in case.durations_ms.iter().zip(&case.expected_tokens) {
            assert_eq!(
                audio_marker_expansion_tokens(duration_ms),
                expected,
                "audio duration_ms={duration_ms} expansion mismatch"
            );
        }
        // total_audio_marker_expansion_tokens must agree with the per-marker
        // sum for zero, one, and multiple markers alike (each `case` here
        // covers exactly one of those shapes -- see the `any()` asserts
        // above).
        assert_eq!(
            total_audio_marker_expansion_tokens(&case.durations_ms) as u64,
            case.expected_tokens.iter().map(|&t| t as u64).sum::<u64>(),
            "total_audio_marker_expansion_tokens mismatch for durations_ms={:?}",
            case.durations_ms
        );
    }

    // Exact post-subsampling boundary arithmetic (ADR-082 G15, transformers @
    // ab1771c9, processing_gemma4.py:260-298): 16 kHz post-subsampling token
    // counts do NOT follow `ceil(duration_ms / 40)` at these durations --
    // 1-10 ms clips have zero mel frames (0 tokens, not 1), and 40-41 ms
    // clips reduce to exactly one soft token after the two stride-2 Conv2d
    // layers (not 2). These are asserted independently of the fixture above
    // so a fixture regeneration bug can't silently launder a wrong formula.
    let exact_boundary_cases: &[(u32, u32)] = &[(0, 0), (1, 0), (10, 0), (11, 1), (40, 1), (41, 1)];
    for &(duration_ms, expected) in exact_boundary_cases {
        assert_eq!(
            audio_marker_expansion_tokens(duration_ms),
            expected,
            "audio duration_ms={duration_ms} exact post-subsampling expansion mismatch"
        );
    }

    // Explicit boundary check: a duration well past the 30s card limit must
    // clamp at the cap rather than grow unbounded.
    assert_eq!(
        audio_marker_expansion_tokens(1_000_000),
        GEMMA4_AUDIO_MAX_SOFT_TOKENS
    );
    assert_eq!(image_marker_expansion_tokens(0), 0);

    // total_audio_marker_expansion_tokens over zero/one/multiple markers,
    // independent of the fixture-driven loop above.
    assert_eq!(total_audio_marker_expansion_tokens(&[]), 0);
    assert_eq!(
        total_audio_marker_expansion_tokens(&[1_000]),
        audio_marker_expansion_tokens(1_000) as usize
    );
    assert_eq!(
        total_audio_marker_expansion_tokens(&[41, 1_000, 40_000]),
        audio_marker_expansion_tokens(41) as usize
            + audio_marker_expansion_tokens(1_000) as usize
            + audio_marker_expansion_tokens(40_000) as usize
    );
}

// ---------------------------------------------------------------------
// Fail-closed constructor validation (ADR-082 Stage 1 medium finding):
// mutating any field `validate_gemma_bpe_shape` checks must reject
// construction, proving the constructor's fail-closed claim isn't
// decorative.
// ---------------------------------------------------------------------

/// A minimal but complete Gemma-shaped `tokenizer.json`, built directly
/// (not derived from the 32 MB committed fixture) so each mutation test
/// stays fast and each mutated field is isolated from the others.
fn minimal_valid_tokenizer_json() -> serde_json::Value {
    serde_json::json!({
        "normalizer": {
            "type": "Replace",
            "pattern": {"String": " "},
            "content": "\u{2581}",
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": false,
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        },
        "added_tokens": [],
        "model": {
            "type": "BPE",
            "dropout": serde_json::Value::Null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": serde_json::Value::Null,
            "end_of_word_suffix": serde_json::Value::Null,
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": {"<unk>": 0, "a": 1, "<0x61>": 2, "\u{2581}": 3},
            "merges": [],
        },
    })
}

fn assert_constructs(value: &serde_json::Value, context: &str) {
    let text = value.to_string();
    assert!(
        GemmaBpeTokenizer::from_tokenizer_json_str(&text).is_ok(),
        "{context}: expected construction to succeed"
    );
}

fn assert_rejects(value: &serde_json::Value, context: &str) {
    let text = value.to_string();
    assert!(
        GemmaBpeTokenizer::from_tokenizer_json_str(&text).is_err(),
        "{context}: expected construction to fail closed, but it succeeded"
    );
}

#[test]
fn minimal_valid_tokenizer_json_constructs() {
    assert_constructs(&minimal_valid_tokenizer_json(), "unmutated baseline");
}

#[test]
fn constructor_rejects_wrong_split_pattern() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["pre_tokenizer"]["pattern"]["String"] = serde_json::json!("\t");
    assert_rejects(&mutated, "pre_tokenizer.pattern.String == \"\\t\"");
}

#[test]
fn constructor_rejects_wrong_split_behavior() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["pre_tokenizer"]["behavior"] = serde_json::json!("Isolated");
    assert_rejects(&mutated, "pre_tokenizer.behavior == \"Isolated\"");
}

#[test]
fn constructor_rejects_inverted_split() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["pre_tokenizer"]["invert"] = serde_json::json!(true);
    assert_rejects(&mutated, "pre_tokenizer.invert == true");
}

#[test]
fn constructor_rejects_reordered_decoder_sequence() {
    let mut mutated = minimal_valid_tokenizer_json();
    let decoders = mutated["decoder"]["decoders"].as_array().unwrap().clone();
    mutated["decoder"]["decoders"] = serde_json::json!([
        decoders[1].clone(),
        decoders[0].clone(),
        decoders[2].clone()
    ]);
    assert_rejects(&mutated, "decoder order ByteFallback,Replace,Fuse");
}

#[test]
fn constructor_rejects_missing_fuse_stage() {
    let mut mutated = minimal_valid_tokenizer_json();
    let decoders = mutated["decoder"]["decoders"].as_array().unwrap().clone();
    mutated["decoder"]["decoders"] = serde_json::json!([decoders[0].clone(), decoders[1].clone()]);
    assert_rejects(&mutated, "decoder missing trailing Fuse");
}

#[test]
fn constructor_rejects_ignore_merges_true() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["model"]["ignore_merges"] = serde_json::json!(true);
    assert_rejects(&mutated, "model.ignore_merges == true");
}

#[test]
fn constructor_rejects_fuse_unk_false() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["model"]["fuse_unk"] = serde_json::json!(false);
    assert_rejects(&mutated, "model.fuse_unk == false");
}

#[test]
fn constructor_rejects_non_null_dropout() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["model"]["dropout"] = serde_json::json!(0.1);
    assert_rejects(&mutated, "model.dropout == 0.1");
}

#[test]
fn constructor_rejects_non_null_continuing_subword_prefix() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["model"]["continuing_subword_prefix"] = serde_json::json!("##");
    assert_rejects(&mutated, "model.continuing_subword_prefix == \"##\"");
}

#[test]
fn constructor_rejects_byte_fallback_false() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["model"]["byte_fallback"] = serde_json::json!(false);
    assert_rejects(&mutated, "model.byte_fallback == false");
}

#[test]
fn constructor_rejects_non_null_end_of_word_suffix() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["model"]["end_of_word_suffix"] = serde_json::json!("</w>");
    assert_rejects(&mutated, "model.end_of_word_suffix == \"</w>\"");
}

#[test]
fn constructor_rejects_decoder_replace_wrong_content() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["decoder"]["decoders"][0]["content"] = serde_json::json!("_");
    assert_rejects(&mutated, "decoder[0].content == \"_\" (not \" \")");
}

#[test]
fn constructor_rejects_decoder_replace_wrong_pattern() {
    let mut mutated = minimal_valid_tokenizer_json();
    mutated["decoder"]["decoders"][0]["pattern"]["String"] = serde_json::json!("_");
    assert_rejects(
        &mutated,
        "decoder[0].pattern.String == \"_\" (not \"\u{2581}\")",
    );
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
