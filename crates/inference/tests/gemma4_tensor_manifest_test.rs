//! Fixture-only gate for the Gemma 4 E2B tensor manifest (ADR-082 Stage 0).
//!
//! **Offline-only.** Unlike `scripts/gemma4_tensor_manifest.py` (which fetches
//! the safetensors header over HTTP Range requests), this test never touches
//! the network — it loads and validates the fixture already committed at
//! `tests/fixtures/gemma4/e2b_tensor_manifest.json`, produced by a single
//! real run of that script against the pinned checkpoint revision
//! (`google/gemma-4-E2B-it` @ `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`).
//!
//! **KV-shared-layer marker note.** ADR-082's G5 states (source-read from
//! `configuration_gemma4.py`/`modeling_gemma4.py`) that "the final 20
//! decoder layers omit K/V projection weights entirely." The header-extracted
//! manifest this test loads does **not** bear that out: all 35
//! `language_model` layers carry their own `self_attn.{k,v,q}_proj.weight`
//! tensors — the checkpoint always materializes per-layer projections. What
//! the manifest *does* confirm, directly and load-bearingly (G3/G4), is the
//! local/global attention-geometry split that KV sharing is keyed on: the 7
//! global-attention layers (indices 4, 9, 14, 19, 24, 29, 34) carry a
//! 512-wide `k_proj`/`v_proj`, and the other 28 sliding-attention layers
//! carry a 256-wide one. `kv_shared_layer_type_marker_matches_schedule`
//! below checks that real, header-extracted split rather than asserting a
//! tensor absence the checkpoint does not exhibit — a Stage-2/4 open item is
//! to reconcile G5's "omit" wording against this.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Deserialize, Clone)]
struct TensorEntry {
    dtype: String,
    shape: Vec<u64>,
}

/// Immutable provenance fields — everything except the generation
/// timestamp, which is expected to change on every fetch.
#[derive(Deserialize, Clone, PartialEq, Debug)]
struct ManifestMetadata {
    source_repo: String,
    revision: String,
    source_url: String,
    header_length_bytes: u64,
    header_sha256: String,
    total_bytes_fetched: u64,
    #[allow(dead_code)]
    extraction_date: String,
}

#[derive(Deserialize)]
struct Manifest {
    metadata: ManifestMetadata,
    bucket_counts: HashMap<String, u64>,
    total_tensors: u64,
    tensors: HashMap<String, TensorEntry>,
}

const EXPECTED_TOTAL: u64 = 2011;
const EXPECTED_BUCKETS: &[(&str, u64)] = &[
    ("model.audio_tower", 751),
    ("model.vision_tower", 658),
    ("model.language_model", 600),
    ("model.embed_audio", 1),
    ("model.embed_vision", 1),
];

/// Ground truth confirmed by header extraction on 2026-07-14 (see G16 in
/// ADR-082 and `scripts/gemma4_tensor_manifest.py`). If a re-fetch
/// disagrees, that is drift — investigate, do not adjust these to match.
const EXPECTED_SOURCE_REPO: &str = "google/gemma-4-E2B-it";
const EXPECTED_REVISION: &str = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf";
const EXPECTED_SOURCE_URL: &str = "https://huggingface.co/google/gemma-4-E2B-it/resolve/9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf/model.safetensors";
const EXPECTED_HEADER_LENGTH_BYTES: u64 = 263_952;
const EXPECTED_HEADER_SHA256: &str =
    "12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e";
const EXPECTED_TOTAL_BYTES_FETCHED: u64 = 263_960;

/// The 7-member global-attention schedule (indices 4, 9, 14, 19, 24, 29, 34
/// — G3). All other layers (0..35 minus these) are sliding-attention.
const GLOBAL_LAYER_INDICES: &[u64] = &[4, 9, 14, 19, 24, 29, 34];
/// Total decoder layer count (G3/G4).
const TOTAL_LAYERS: u64 = 35;
const GLOBAL_KV_WIDTH: u64 = 512;
const SLIDING_KV_WIDTH: u64 = 256;
const KV_HIDDEN_DIM: u64 = 1536;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gemma4")
        .join("e2b_tensor_manifest.json")
}

fn load_manifest_str(data: &str) -> Result<Manifest, String> {
    serde_json::from_str(data).map_err(|e| format!("bad manifest JSON: {e}"))
}

fn load_committed_manifest() -> Manifest {
    let path = fixture_path();
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    load_manifest_str(&data).unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display()))
}

/// Validation helper mirroring what a Stage-2 config/loader preflight would
/// run against a fetched manifest: total count, every declared bucket
/// present at its expected count, and every immutable provenance field
/// (everything but the generation timestamp). Returns the first failure
/// reason, or `None` if the manifest is well-formed.
fn validate_manifest(m: &Manifest) -> Option<String> {
    if m.total_tensors != EXPECTED_TOTAL {
        return Some(format!(
            "total_tensors {} != expected {EXPECTED_TOTAL}",
            m.total_tensors
        ));
    }
    if m.tensors.len() as u64 != m.total_tensors {
        return Some(format!(
            "tensors map has {} entries but total_tensors says {}",
            m.tensors.len(),
            m.total_tensors
        ));
    }
    for (bucket, expected_count) in EXPECTED_BUCKETS {
        let actual = m.bucket_counts.get(*bucket).copied();
        if actual != Some(*expected_count) {
            return Some(format!(
                "bucket {bucket} count {actual:?} != expected {expected_count}"
            ));
        }
    }
    if m.metadata.source_repo != EXPECTED_SOURCE_REPO {
        return Some(format!(
            "metadata.source_repo {:?} != expected {EXPECTED_SOURCE_REPO:?}",
            m.metadata.source_repo
        ));
    }
    if m.metadata.revision != EXPECTED_REVISION {
        return Some(format!(
            "metadata.revision {:?} != expected {EXPECTED_REVISION:?}",
            m.metadata.revision
        ));
    }
    if m.metadata.source_url != EXPECTED_SOURCE_URL {
        return Some(format!(
            "metadata.source_url {:?} != expected {EXPECTED_SOURCE_URL:?}",
            m.metadata.source_url
        ));
    }
    if m.metadata.header_length_bytes != EXPECTED_HEADER_LENGTH_BYTES {
        return Some(format!(
            "metadata.header_length_bytes {} != expected {EXPECTED_HEADER_LENGTH_BYTES}",
            m.metadata.header_length_bytes
        ));
    }
    if m.metadata.header_sha256 != EXPECTED_HEADER_SHA256 {
        return Some(format!(
            "metadata.header_sha256 {:?} != expected {EXPECTED_HEADER_SHA256:?}",
            m.metadata.header_sha256
        ));
    }
    if m.metadata.total_bytes_fetched != EXPECTED_TOTAL_BYTES_FETCHED {
        return Some(format!(
            "metadata.total_bytes_fetched {} != expected {EXPECTED_TOTAL_BYTES_FETCHED}",
            m.metadata.total_bytes_fetched
        ));
    }
    None
}

#[test]
fn gemma4_manifest_total_tensor_count_is_2011() {
    let m = load_committed_manifest();
    assert_eq!(m.total_tensors, EXPECTED_TOTAL);
    assert_eq!(m.tensors.len() as u64, EXPECTED_TOTAL);
}

#[test]
fn gemma4_manifest_bucket_counts_match_adr_082_g16() {
    let m = load_committed_manifest();
    for (bucket, expected_count) in EXPECTED_BUCKETS {
        assert_eq!(
            m.bucket_counts.get(*bucket).copied(),
            Some(*expected_count),
            "bucket {bucket} count mismatch"
        );
    }
    let declared_total: u64 = EXPECTED_BUCKETS.iter().map(|(_, c)| c).sum();
    assert_eq!(
        declared_total, EXPECTED_TOTAL,
        "declared bucket counts must sum to the total tensor count"
    );
}

#[test]
fn gemma4_spot_check_ple_embedding_table() {
    let m = load_committed_manifest();
    let name = "model.language_model.embed_tokens_per_layer.weight";
    let t = m
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing PLE tensor {name}"));
    // [vocab=262144, num_layers=35 * per_layer_dim=256] per G9.
    assert_eq!(t.shape, vec![262144, 8960]);
    assert_eq!(t.dtype, "BF16");
}

/// All 35 decoder layers must carry their own Q/K/V projections (see module
/// doc — this contradicts a literal reading of G5's "final 20 layers omit
/// K/V projection weights"). Kept separate from the geometry check below so
/// a layer silently losing a projection tensor entirely is distinguished
/// from one merely having the wrong K/V width.
#[test]
fn gemma4_all_layers_have_qkv_projections() {
    let m = load_committed_manifest();
    for layer in 0..TOTAL_LAYERS {
        for proj in ["q_proj", "k_proj", "v_proj"] {
            let name = format!("model.language_model.layers.{layer}.self_attn.{proj}.weight");
            assert!(
                m.tensors.contains_key(&name),
                "layer {layer} is missing {proj}: expected tensor {name}"
            );
        }
    }
}

/// The real, header-extracted marker for the local/global attention split
/// (G3/G4): global-attention layers carry a 512-wide k_proj/v_proj, all
/// other (sliding-attention) layers carry a 256-wide one. Iterates the full
/// 35-layer schedule — not just one global/sliding sample pair — so a
/// mid-schedule layer with the wrong width cannot slip through.
#[test]
fn gemma4_kv_shared_layer_type_marker_matches_schedule() {
    let m = load_committed_manifest();

    for layer in 0..TOTAL_LAYERS {
        let is_global = GLOBAL_LAYER_INDICES.contains(&layer);
        let expected_width = if is_global {
            GLOBAL_KV_WIDTH
        } else {
            SLIDING_KV_WIDTH
        };
        let schedule_kind = if is_global { "global" } else { "sliding" };

        for proj in ["k_proj", "v_proj"] {
            let name = format!("model.language_model.layers.{layer}.self_attn.{proj}.weight");
            let t = m
                .tensors
                .get(&name)
                .unwrap_or_else(|| panic!("expected {name} present in manifest"));
            assert_eq!(
                t.shape,
                vec![expected_width, KV_HIDDEN_DIM],
                "layer {layer} is {schedule_kind}-attention — expected a {expected_width}-wide {proj}"
            );
        }
    }
}

#[test]
fn gemma4_spot_check_vision_clippable_linear_clip_buffer() {
    let m = load_committed_manifest();
    let name = "model.vision_tower.encoder.layers.0.self_attn.q_proj.input_max";
    let t = m
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing vision clip buffer {name}"));
    // Gemma4ClippableLinear persists scalar min/max clip buffers per G11 —
    // shape [] (0-d scalar), not a disposable quantization side-channel.
    assert_eq!(t.shape, Vec::<u64>::new());
    assert_eq!(t.dtype, "BF16");
}

#[test]
fn gemma4_spot_check_audio_conv_subsampler_tensor() {
    let m = load_committed_manifest();
    let name = "model.audio_tower.subsample_conv_projection.layer0.conv.weight";
    let t = m
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("missing audio subsampler tensor {name}"));
    // First 3x3 stride-2 Conv2d block, 1 -> 128 channels per G14.
    assert_eq!(t.shape, vec![128, 1, 3, 3]);
    assert_eq!(t.dtype, "BF16");
}

/// Fail-closed negative test: a corrupted/truncated manifest JSON must be
/// rejected by the parser, not silently accepted with missing fields.
#[test]
fn gemma4_corrupt_truncated_manifest_json_is_rejected() {
    let good = std::fs::read_to_string(fixture_path()).expect("read committed fixture");
    // Truncate mid-document — this is not valid JSON.
    let truncated = &good[..good.len() / 2];
    let result = load_manifest_str(truncated);
    assert!(
        result.is_err(),
        "truncated manifest JSON must fail to parse, not silently load partial data"
    );

    // Also check structurally-invalid-but-parseable JSON (missing required
    // fields) is rejected.
    let malformed = r#"{"metadata": {}, "bucket_counts": {}}"#;
    let result2 = load_manifest_str(malformed);
    assert!(
        result2.is_err(),
        "manifest JSON missing required fields (total_tensors, tensors) must fail to parse"
    );
}

/// Fail-closed negative test: a manifest missing a required subsystem
/// bucket must fail `validate_manifest`, not silently pass.
#[test]
fn gemma4_manifest_missing_required_bucket_fails_validation() {
    let mut m = load_committed_manifest();
    assert!(
        validate_manifest(&m).is_none(),
        "sanity: committed manifest should validate cleanly before mutation"
    );

    m.bucket_counts.remove("model.audio_tower");
    let failure = validate_manifest(&m);
    assert!(
        failure.is_some(),
        "removing the model.audio_tower bucket must fail validate_manifest"
    );
    assert!(failure.unwrap().contains("model.audio_tower"));
}

/// Fail-closed negative test: a mutated tensor count must fail validation
/// even when every bucket individually still looks present.
#[test]
fn gemma4_manifest_wrong_total_tensor_count_fails_validation() {
    let mut m = load_committed_manifest();
    m.total_tensors += 1;
    let failure = validate_manifest(&m);
    assert!(
        failure.is_some(),
        "a total_tensors off-by-one must fail validate_manifest"
    );
}

/// Fail-closed negative test: a wrong (but still present) bucket count must
/// fail validation, not just a fully missing bucket.
#[test]
fn gemma4_manifest_wrong_bucket_count_fails_validation() {
    let mut m = load_committed_manifest();
    m.bucket_counts
        .insert("model.language_model".to_string(), 599);
    let failure = validate_manifest(&m);
    assert!(
        failure.is_some(),
        "a wrong model.language_model bucket count must fail validate_manifest"
    );
    assert!(failure.unwrap().contains("model.language_model"));
}

/// Fail-closed negative test: a changed pinned revision must fail
/// validation — the whole point of the pin is that it does not drift.
#[test]
fn gemma4_manifest_changed_revision_fails_validation() {
    let mut m = load_committed_manifest();
    m.metadata.revision = "0000000000000000000000000000000000000000".to_string();
    let failure = validate_manifest(&m);
    assert!(
        failure.is_some(),
        "a changed metadata.revision must fail validate_manifest"
    );
    assert!(failure.unwrap().contains("revision"));
}

/// Fail-closed negative test: a changed header SHA-256 must fail
/// validation — this is the provenance hash tying the manifest to the
/// exact bytes that were fetched.
#[test]
fn gemma4_manifest_changed_header_sha256_fails_validation() {
    let mut m = load_committed_manifest();
    m.metadata.header_sha256 = "0".repeat(64);
    let failure = validate_manifest(&m);
    assert!(
        failure.is_some(),
        "a changed metadata.header_sha256 must fail validate_manifest"
    );
    assert!(failure.unwrap().contains("header_sha256"));
}

/// Fail-closed negative test: a manifest missing a required metadata field
/// entirely must fail to parse (typed struct, not a permissive
/// `serde_json::Value`).
#[test]
fn gemma4_manifest_missing_required_metadata_field_is_rejected() {
    let malformed = r#"{
        "metadata": {
            "source_repo": "google/gemma-4-E2B-it",
            "revision": "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf",
            "source_url": "https://huggingface.co/google/gemma-4-E2B-it/resolve/9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf/model.safetensors",
            "header_length_bytes": 263952,
            "total_bytes_fetched": 263960,
            "extraction_date": "2026-07-15T00:40:25Z"
        },
        "bucket_counts": {},
        "total_tensors": 0,
        "tensors": {}
    }"#;
    let result = load_manifest_str(malformed);
    assert!(
        result.is_err(),
        "manifest metadata missing header_sha256 must fail to parse"
    );
}
