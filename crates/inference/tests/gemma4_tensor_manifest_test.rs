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

#[derive(Deserialize)]
struct Manifest {
    #[allow(dead_code)]
    metadata: serde_json::Value,
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

/// Late-layer index inside the 7-member global-attention schedule
/// (indices 4, 9, 14, 19, 24, 29, 34 — G3).
const GLOBAL_LAYER_IDX: u64 = 34;
/// A sliding-attention (non-global) layer, for contrast.
const SLIDING_LAYER_IDX: u64 = 33;

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
/// present at its expected count. Returns the first failure reason, or
/// `None` if the manifest is well-formed.
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

#[test]
fn gemma4_kv_shared_layer_type_marker_matches_schedule() {
    let m = load_committed_manifest();

    let global_k =
        format!("model.language_model.layers.{GLOBAL_LAYER_IDX}.self_attn.k_proj.weight");
    let sliding_k =
        format!("model.language_model.layers.{SLIDING_LAYER_IDX}.self_attn.k_proj.weight");

    // Both are present in the checkpoint (see module doc — this contradicts
    // a literal reading of G5's "final 20 layers omit K/V projection
    // weights"). What IS a real, header-extracted marker is the shape
    // split: global layers get a 512-wide K/V head, sliding layers a
    // 256-wide one (G3/G4).
    let global_t = m
        .tensors
        .get(&global_k)
        .unwrap_or_else(|| panic!("expected {global_k} present in manifest"));
    let sliding_t = m
        .tensors
        .get(&sliding_k)
        .unwrap_or_else(|| panic!("expected {sliding_k} present in manifest"));

    assert_eq!(
        global_t.shape,
        vec![512, 1536],
        "layer {GLOBAL_LAYER_IDX} is in the global-attention schedule (G3) — expected a 512-wide k_proj"
    );
    assert_eq!(
        sliding_t.shape,
        vec![256, 1536],
        "layer {SLIDING_LAYER_IDX} is a sliding-attention layer — expected a 256-wide k_proj"
    );
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
