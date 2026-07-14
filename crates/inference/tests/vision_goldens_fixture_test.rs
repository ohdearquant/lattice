//! Fixture-loader round-trip for the HF differential vision goldens (ADR-069).
//!
//! `scripts/vision_goldens_qwen35.py` pins HF `transformers` reference output
//! (ViT patch features, position ids, greedy tokens) for a fixed image +
//! prompt under `tests/fixtures/vision/`, so future ADR-069 stages (S3-S5)
//! have a committed cross-framework target before any lattice vision
//! forward-pass code exists. This test exercises the fixture-reading side of
//! that contract — `manifest.json` parses, the two raw little-endian f32
//! `.bin` tensors round-trip to their recorded shape/sha256/finiteness, and
//! the JSON side-fixtures cross-check against the manifest — WITHOUT running
//! any engine forward pass. It exists so fixture-loading code is written and
//! exercised in CI before S3 needs it, not after.

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct TensorManifest {
    path: String,
    shape: Vec<usize>,
    dtype: String,
    endianness: String,
    sha256: String,
    num_elements: usize,
}

#[derive(Deserialize)]
struct JsonFileManifest {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    adr: String,
    image_grid_thw: Vec<[u64; 3]>,
    num_image_placeholder_tokens: usize,
    vit_pre_merger: TensorManifest,
    vit_post_merger: TensorManifest,
    input_ids: JsonFileManifest,
    position_ids: JsonFileManifest,
    greedy_tokens: JsonFileManifest,
    sanity_checks: Value,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("vision")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Load a raw little-endian f32 tensor fixture, verifying its byte length and
/// sha256 against `manifest.json` before decoding.
fn load_f32_tensor(dir: &Path, tm: &TensorManifest) -> Vec<f32> {
    assert_eq!(
        tm.dtype, "float32",
        "unexpected dtype in manifest: {}",
        tm.dtype
    );
    assert_eq!(
        tm.endianness, "little",
        "unexpected endianness in manifest: {}",
        tm.endianness
    );

    let path = dir.join(&tm.path);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));

    assert_eq!(
        bytes.len(),
        tm.num_elements * 4,
        "{}: byte length does not match manifest num_elements * 4",
        tm.path
    );
    assert_eq!(
        sha256_hex(&bytes),
        tm.sha256,
        "{}: sha256 mismatch vs manifest.json (fixture corrupted or regenerated without updating manifest)",
        tm.path
    );

    let shape_product: usize = tm.shape.iter().product();
    assert_eq!(
        shape_product, tm.num_elements,
        "{}: manifest shape {:?} does not multiply out to num_elements {}",
        tm.path, tm.shape, tm.num_elements
    );

    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(values.len(), tm.num_elements);
    values
}

fn load_json_checked(dir: &Path, jm: &JsonFileManifest) -> Value {
    let path = dir.join(&jm.path);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
    assert_eq!(
        sha256_hex(&bytes),
        jm.sha256,
        "{}: sha256 mismatch vs manifest.json",
        jm.path
    );
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parsing {path:?}: {e}"))
}

#[test]
fn manifest_parses_and_matches_schema() {
    let dir = fixture_dir();
    let manifest_path = dir.join("manifest.json");
    let manifest_bytes =
        std::fs::read(&manifest_path).unwrap_or_else(|e| panic!("reading {manifest_path:?}: {e}"));
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .unwrap_or_else(|e| panic!("parsing {manifest_path:?}: {e}"));

    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.adr, "ADR-069");
    assert_eq!(manifest.image_grid_thw.len(), 1, "fixture is single-image");

    let sanity = manifest
        .sanity_checks
        .as_object()
        .expect("sanity_checks is an object");
    for (key, value) in sanity {
        assert_eq!(
            value.as_bool(),
            Some(true),
            "recorded sanity check {key:?} was not true when the fixture was generated"
        );
    }
}

#[test]
fn vit_pre_and_post_merger_tensors_round_trip_finite() {
    let dir = fixture_dir();
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();

    let pre = load_f32_tensor(&dir, &manifest.vit_pre_merger);
    let post = load_f32_tensor(&dir, &manifest.vit_post_merger);

    assert!(!pre.is_empty());
    assert!(!post.is_empty());
    assert!(
        pre.iter().all(|v| v.is_finite()),
        "vit_pre_merger contains non-finite values"
    );
    assert!(
        post.iter().all(|v| v.is_finite()),
        "vit_post_merger contains non-finite values"
    );

    // Pre-merger tensor shape is (num_patches, vision hidden_size); post-merger
    // is (num_visual_tokens, out_hidden_size). num_patches for a single image
    // is grid_t * grid_h * grid_w from image_grid_thw.
    let [grid_t, grid_h, grid_w] = manifest.image_grid_thw[0];
    let num_patches = (grid_t * grid_h * grid_w) as usize;
    assert_eq!(manifest.vit_pre_merger.shape[0], num_patches);

    // Post-merger row count must equal the number of expanded image
    // placeholder tokens the decoder will see (S4's token-stream contract).
    assert_eq!(
        manifest.vit_post_merger.shape[0],
        manifest.num_image_placeholder_tokens
    );
}

#[test]
fn token_and_position_fixtures_are_internally_consistent() {
    let dir = fixture_dir();
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();

    let input_ids = load_json_checked(&dir, &manifest.input_ids);
    let input_ids = input_ids
        .as_array()
        .expect("input_ids.json is a JSON array");
    assert!(!input_ids.is_empty());

    let position_ids = load_json_checked(&dir, &manifest.position_ids);
    let shape = position_ids["shape"]
        .as_array()
        .expect("position_ids.shape");
    assert_eq!(
        shape[0].as_u64(),
        Some(3),
        "position ids must be 3-axis (t, h, w)"
    );
    assert_eq!(shape[1].as_u64(), Some(input_ids.len() as u64));

    let values = position_ids["values"]
        .as_array()
        .expect("position_ids.values");
    assert_eq!(values.len(), 3);
    for axis in values {
        let axis = axis.as_array().expect("position_ids axis row");
        assert_eq!(axis.len(), input_ids.len());
    }

    // Text-prefix collapse: axis 0 (t) and axis 1 (h) must agree with axis 2
    // (w) at sequence position 0 (a plain text token, before any image span).
    let t0 = values[0][0].as_i64().unwrap();
    let h0 = values[1][0].as_i64().unwrap();
    let w0 = values[2][0].as_i64().unwrap();
    assert_eq!(
        (t0, h0),
        (w0, w0),
        "text-prefix position ids must collapse to 1-D (t == h == w)"
    );

    let greedy = load_json_checked(&dir, &manifest.greedy_tokens);
    let token_ids = greedy["token_ids"]
        .as_array()
        .expect("greedy_tokens.token_ids");
    assert_eq!(
        greedy["max_new_tokens"].as_u64(),
        Some(token_ids.len() as u64),
        "greedy_tokens.json max_new_tokens must match the recorded token count"
    );
    assert!(!token_ids.is_empty(), "expected at least one greedy token");
}
