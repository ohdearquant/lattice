//! Fixture-loader round-trip for the HF differential vision goldens (ADR-069).
//!
//! `scripts/vision_goldens_qwen35.py` pins HF `transformers` reference output
//! (ViT patch features, position ids, greedy tokens) for a fixed image +
//! prompt under `tests/fixtures/vision/`, so future ADR-069 stages (S3-S5)
//! have a committed cross-framework target before any lattice vision
//! forward-pass code exists. This test exercises the fixture-reading side of
//! that contract — `manifest.json` parses, the two raw little-endian f32
//! `.bin` tensors round-trip to their recorded shape/sha256/finiteness, the
//! golden PNG hash/dimension-verifies, the image-token id is counted end to
//! end, the documented M-RoPE position-id properties are checked across their
//! FULL ranges (not a single prefix position), and a block of hardcoded
//! anchor constants (`anchors` module below) cross-checks the fixtures
//! independently of anything `manifest.json` itself claims — WITHOUT running
//! any engine forward pass. It exists so fixture-loading code is written and
//! exercised in CI before S3 needs it, not after.

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Reviewed anchor constants for the pinned HF source + generation
/// methodology. Deliberately NOT read from `manifest.json`: a corrupted or
/// maliciously "self-consistent" regeneration that mutates both a fixture
/// file and the manifest entry describing it must still be caught here,
/// because these values are fixed at review time and only change as part of
/// an explicitly reviewed fixture refresh (bump them by hand, in this file).
mod anchors {
    // Fixed filenames, not read from the manifest's `path` fields, so an
    // edited manifest can't redirect these checks onto a different file.
    pub const GOLDEN_IMAGE_FILE: &str = "golden_image.png";
    pub const VIT_PRE_MERGER_FILE: &str = "vit_pre_merger_f32.bin";
    pub const VIT_POST_MERGER_FILE: &str = "vit_post_merger_f32.bin";
    pub const INPUT_IDS_FILE: &str = "input_ids.json";
    pub const POSITION_IDS_FILE: &str = "position_ids.json";

    pub const HF_PROCESSOR_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";
    pub const PROMPT: &str = "Describe this image.";

    // Every load-bearing `source` field, fixed at review time. A
    // self-consistent edit that changes `hf_model_id`, either checkpoint
    // digest, or any runtime/Python version — while leaving the payload
    // files and the manifest's own digests mutually consistent — must still
    // be caught here, because `Source` (below) is deserialized and checked
    // field-by-field against these constants, not against anything
    // `manifest.json` claims about itself.
    pub const HF_MODEL_ID: &str = "Qwen/Qwen3.5-0.8B";
    pub const MODEL_DIR_CONFIG_SHA256: &str =
        "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204";
    pub const MODEL_DIR_SAFETENSORS_INDEX_SHA256: &str =
        "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e";
    pub const TORCH_VERSION: &str = "2.13.0";
    pub const TORCHVISION_VERSION: &str = "0.28.0";
    pub const PILLOW_VERSION: &str = "12.3.0";
    pub const TRANSFORMERS_VERSION: &str = "5.12.1";
    pub const NUMPY_VERSION: &str = "2.4.6";
    pub const PYTHON_VERSION: &str = "3.11.12";

    /// `<|image_pad|>` token id (see `tests/fixtures/vision/README.md`).
    pub const IMAGE_TOKEN_ID: i64 = 248056;
    pub const NUM_IMAGE_TOKENS: usize = 64;

    pub const VIT_PRE_MERGER_SHA256: &str =
        "1a82b0bb6c873fb6aa62b35e33ad7c6c0bc50f028b119e3e4862f6bc0bdf62d3";
    pub const VIT_POST_MERGER_SHA256: &str =
        "6b5ef28b198a370449bbbcd7bdfcf5cbe82b5d03fdd750e91336fb38fa4cacf9";
    pub const GOLDEN_IMAGE_SHA256: &str =
        "d5083740a73e5d90cce9f75c7e7eac7efcb965ae9ad0f173ded2e370e6d7924b";
    pub const GOLDEN_IMAGE_WIDTH: u32 = 256;
    pub const GOLDEN_IMAGE_HEIGHT: u32 = 256;

    // Anchored directly, NOT via `manifest.json`'s `input_ids`/`position_ids`
    // `sha256` fields: a self-consistent tamper (edit the fixture, then
    // update the manifest's own digest to match) would otherwise still pass
    // `load_json_checked`. These constants are fixed at review time from the
    // committed files and only change as part of an explicitly reviewed
    // fixture refresh.
    pub const INPUT_IDS_SHA256: &str =
        "9beda884e9b718c0ce58c657c7ac5535936582f20d2141a8b03f75b7ab0adfd5";
    pub const POSITION_IDS_SHA256: &str =
        "f1407311a976e23e559e32edb8eafe47bcd4213c3b4dad5561b0ee8b5404fe3d";

    pub const GREEDY_TOKENS: [i64; 8] = [1919, 2099, 369, 264, 2972, 12896, 518, 19556];
}

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
struct ImageManifest {
    path: String,
    size: [u32; 2],
    sha256: String,
}

/// The full `source` provenance object, deserialized as a typed struct (not
/// read field-by-field out of `Value`) so that adding a new load-bearing
/// field to the generator without also anchoring it here is a compile error
/// at the call site below, not a silent gap.
#[derive(Deserialize)]
struct Source {
    hf_model_id: String,
    hf_processor_revision: String,
    model_dir_config_sha256: String,
    model_dir_safetensors_index_sha256: String,
    transformers_version: String,
    torch_version: String,
    torchvision_version: String,
    pillow_version: String,
    numpy_version: String,
    python_version: String,
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    adr: String,
    source: Source,
    image_grid_thw: Vec<[u64; 3]>,
    num_image_placeholder_tokens: usize,
    image: ImageManifest,
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

/// Parse a PNG's IHDR chunk directly (no image-decoding dependency needed for
/// a plain, uncompressed-metadata dimension check): 8-byte signature, then a
/// 4-byte chunk length, `b"IHDR"`, 4-byte width, 4-byte height (both
/// big-endian), bit depth, color type.
fn png_dimensions(bytes: &[u8]) -> (u32, u32, u8, u8) {
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    assert!(
        bytes.len() >= 26,
        "file too short to contain a PNG IHDR chunk"
    );
    assert_eq!(&bytes[0..8], &SIGNATURE, "missing PNG signature");
    assert_eq!(&bytes[12..16], b"IHDR", "first PNG chunk is not IHDR");
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    let bit_depth = bytes[24];
    let color_type = bytes[25];
    (width, height, bit_depth, color_type)
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

#[test]
fn image_fixture_hash_and_dimensions_verified() {
    let dir = fixture_dir();
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();

    let path = dir.join(&manifest.image.path);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));

    assert_eq!(
        sha256_hex(&bytes),
        manifest.image.sha256,
        "golden_image.png sha256 mismatch vs manifest.json (fixture replaced or deleted without updating the manifest)"
    );

    let (width, height, bit_depth, color_type) = png_dimensions(&bytes);
    assert_eq!(
        [width, height],
        manifest.image.size,
        "PNG dimensions mismatch vs manifest.json"
    );
    assert_eq!(width, 256, "expected a 256x256 golden image");
    assert_eq!(height, 256, "expected a 256x256 golden image");
    assert_eq!(bit_depth, 8, "expected an 8-bit-depth PNG");
    assert_eq!(
        color_type, 2,
        "expected PNG color type 2 (RGB truecolor, no alpha)"
    );
}

#[test]
fn image_token_id_count_and_position_id_full_range_properties() {
    let dir = fixture_dir();
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();

    let input_ids_json = load_json_checked(&dir, &manifest.input_ids);
    let input_ids: Vec<i64> = input_ids_json
        .as_array()
        .expect("input_ids.json is a JSON array")
        .iter()
        .map(|v| v.as_i64().expect("input_ids entries are integers"))
        .collect();

    let image_positions: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|&(_, &tok)| tok == anchors::IMAGE_TOKEN_ID)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        image_positions.len(),
        anchors::NUM_IMAGE_TOKENS,
        "expected exactly {} occurrences of image token id {} in input_ids",
        anchors::NUM_IMAGE_TOKENS,
        anchors::IMAGE_TOKEN_ID
    );
    assert_eq!(
        image_positions.len(),
        manifest.num_image_placeholder_tokens,
        "image token count must match manifest.num_image_placeholder_tokens"
    );

    let position_ids_json = load_json_checked(&dir, &manifest.position_ids);
    let axis_values = |i: usize| -> Vec<i64> {
        position_ids_json["values"][i]
            .as_array()
            .unwrap_or_else(|| panic!("position_ids.values[{i}] missing"))
            .iter()
            .map(|v| v.as_i64().expect("position id entries are integers"))
            .collect()
    };
    let t = axis_values(0);
    let h = axis_values(1);
    let w = axis_values(2);
    assert_eq!(t.len(), input_ids.len());
    assert_eq!(h.len(), input_ids.len());
    assert_eq!(w.len(), input_ids.len());

    // Image span must be contiguous, since the M-RoPE grid-sweep check below
    // assumes an unbroken run of merged-grid positions.
    let first_image_pos = *image_positions.first().unwrap();
    let last_image_pos = *image_positions.last().unwrap();
    assert_eq!(
        image_positions,
        (first_image_pos..=last_image_pos).collect::<Vec<_>>(),
        "image token positions must be contiguous"
    );

    // FULL-range prefix collapse: every position strictly before the image
    // span must have t == h == w (not just position 0).
    for i in 0..first_image_pos {
        assert_eq!(
            (t[i], h[i]),
            (w[i], w[i]),
            "text-prefix position {i} did not collapse to 1-D (t == h == w)"
        );
    }

    // FULL-range image-span divergence + merged-grid sweep pattern: t stays
    // constant across the whole image span while h/w sweep the merged
    // grid_h x grid_w grid (spatial_merge_size = 2), as documented in
    // tests/fixtures/vision/README.md.
    let [_grid_t, grid_h, grid_w] = manifest.image_grid_thw[0];
    let merge = 2u64;
    let merged_h = grid_h / merge;
    let merged_w = grid_w / merge;
    assert_eq!(
        (merged_h * merged_w) as usize,
        anchors::NUM_IMAGE_TOKENS,
        "merged grid size must equal the image token count"
    );
    let t_image = t[first_image_pos];
    let h0 = h[first_image_pos];
    let w0 = w[first_image_pos];
    for (k, &pos) in image_positions.iter().enumerate() {
        let k = k as u64;
        assert_eq!(
            t[pos], t_image,
            "image-span position {pos} t axis must stay constant"
        );
        assert_eq!(
            h[pos] as u64,
            h0 as u64 + k / merged_w,
            "image-span position {pos} h axis does not match the merged-grid sweep"
        );
        assert_eq!(
            w[pos] as u64,
            w0 as u64 + k % merged_w,
            "image-span position {pos} w axis does not match the merged-grid sweep"
        );
    }
    // Divergence across the whole span (not necessarily at every single
    // position — the grid sweep above pins the exact per-position values,
    // which is strictly stronger; this just also checks the h/w axes are not
    // *identical arrays* across the span, matching the generator's own
    // `image_span_positions_diverge` sanity check).
    assert_ne!(
        image_positions.iter().map(|&i| h[i]).collect::<Vec<_>>(),
        image_positions.iter().map(|&i| w[i]).collect::<Vec<_>>(),
        "image-span h and w axes must diverge somewhere across the span"
    );

    // FULL-range trailing-text reconvergence: every position after the image
    // span collapses back to 1-D and continues incrementing by 1.
    for i in (last_image_pos + 1)..t.len() {
        assert_eq!(
            (t[i], h[i]),
            (w[i], w[i]),
            "trailing-text position {i} did not reconverge to 1-D (t == h == w)"
        );
    }
    for i in (last_image_pos + 2)..t.len() {
        assert_eq!(
            t[i],
            t[i - 1] + 1,
            "trailing-text position ids did not increment by 1 at position {i}"
        );
    }
}

#[test]
fn anchored_reference_values_match_reviewed_constants() {
    let dir = fixture_dir();
    let manifest_bytes = std::fs::read(dir.join("manifest.json")).unwrap();

    // Deliberately re-parsed as raw JSON (not the `Manifest` struct) for the
    // one field (`prompt`) that isn't part of `Source`, and cross-checked
    // against `anchors`, so a mutation to `manifest.json` itself cannot make
    // this test pass — these constants are fixed here, independent of
    // anything the manifest claims about itself.
    let manifest_raw: Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(
        manifest_raw["prompt"].as_str(),
        Some(anchors::PROMPT),
        "fixture prompt drifted from the reviewed anchor"
    );

    // Every load-bearing `source` field, deserialized as the full typed
    // `Source` struct (not partial `Value` indexing) and checked against a
    // reviewed anchor: model ID, processor revision, both checkpoint
    // digests, and all six runtime/Python versions. A self-consistent edit
    // that changes any of these together (e.g. a different `hf_model_id`
    // plus a matching runtime version bump) must fail here even though it
    // wouldn't touch any payload file's bytes.
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let source = &manifest.source;
    assert_eq!(
        source.hf_model_id,
        anchors::HF_MODEL_ID,
        "hf_model_id drifted from the reviewed anchor"
    );
    assert_eq!(
        source.hf_processor_revision,
        anchors::HF_PROCESSOR_REVISION,
        "pinned HF source revision drifted from the reviewed anchor"
    );
    assert_eq!(
        source.model_dir_config_sha256,
        anchors::MODEL_DIR_CONFIG_SHA256,
        "checkpoint config.json digest drifted from the reviewed anchor"
    );
    assert_eq!(
        source.model_dir_safetensors_index_sha256,
        anchors::MODEL_DIR_SAFETENSORS_INDEX_SHA256,
        "checkpoint model.safetensors.index.json digest drifted from the reviewed anchor"
    );
    assert_eq!(
        source.torch_version,
        anchors::TORCH_VERSION,
        "torch runtime version drifted from the reviewed anchor"
    );
    assert_eq!(
        source.torchvision_version,
        anchors::TORCHVISION_VERSION,
        "torchvision runtime version drifted from the reviewed anchor"
    );
    assert_eq!(
        source.pillow_version,
        anchors::PILLOW_VERSION,
        "pillow runtime version drifted from the reviewed anchor"
    );
    assert_eq!(
        source.transformers_version,
        anchors::TRANSFORMERS_VERSION,
        "transformers runtime version drifted from the reviewed anchor"
    );
    assert_eq!(
        source.numpy_version,
        anchors::NUMPY_VERSION,
        "numpy runtime version drifted from the reviewed anchor"
    );
    assert_eq!(
        source.python_version,
        anchors::PYTHON_VERSION,
        "python runtime version drifted from the reviewed anchor"
    );

    let image_bytes = std::fs::read(dir.join(anchors::GOLDEN_IMAGE_FILE))
        .unwrap_or_else(|e| panic!("reading {}: {e}", anchors::GOLDEN_IMAGE_FILE));
    assert_eq!(
        sha256_hex(&image_bytes),
        anchors::GOLDEN_IMAGE_SHA256,
        "golden_image.png content drifted from the reviewed anchor digest"
    );
    let (width, height, _bit_depth, _color_type) = png_dimensions(&image_bytes);
    assert_eq!(width, anchors::GOLDEN_IMAGE_WIDTH);
    assert_eq!(height, anchors::GOLDEN_IMAGE_HEIGHT);

    let pre_bytes = std::fs::read(dir.join(anchors::VIT_PRE_MERGER_FILE))
        .unwrap_or_else(|e| panic!("reading {}: {e}", anchors::VIT_PRE_MERGER_FILE));
    assert_eq!(
        sha256_hex(&pre_bytes),
        anchors::VIT_PRE_MERGER_SHA256,
        "vit_pre_merger_f32.bin content drifted from the reviewed anchor digest"
    );

    let post_bytes = std::fs::read(dir.join(anchors::VIT_POST_MERGER_FILE))
        .unwrap_or_else(|e| panic!("reading {}: {e}", anchors::VIT_POST_MERGER_FILE));
    assert_eq!(
        sha256_hex(&post_bytes),
        anchors::VIT_POST_MERGER_SHA256,
        "vit_post_merger_f32.bin content drifted from the reviewed anchor digest"
    );

    // input_ids.json / position_ids.json: fixed filenames + reviewed digests,
    // independent of `manifest.json`'s own (mutable) `sha256` fields for
    // these files — closes the self-consistent-tamper gap where an edited
    // token or position id plus an updated manifest digest would otherwise
    // still pass `load_json_checked` in the other tests.
    let input_ids_bytes = std::fs::read(dir.join(anchors::INPUT_IDS_FILE))
        .unwrap_or_else(|e| panic!("reading {}: {e}", anchors::INPUT_IDS_FILE));
    assert_eq!(
        sha256_hex(&input_ids_bytes),
        anchors::INPUT_IDS_SHA256,
        "input_ids.json content drifted from the reviewed anchor digest"
    );

    let position_ids_bytes = std::fs::read(dir.join(anchors::POSITION_IDS_FILE))
        .unwrap_or_else(|e| panic!("reading {}: {e}", anchors::POSITION_IDS_FILE));
    assert_eq!(
        sha256_hex(&position_ids_bytes),
        anchors::POSITION_IDS_SHA256,
        "position_ids.json content drifted from the reviewed anchor digest"
    );

    let greedy = load_json_checked(&dir, &manifest.greedy_tokens);
    let token_ids: Vec<i64> = greedy["token_ids"]
        .as_array()
        .expect("greedy_tokens.token_ids")
        .iter()
        .map(|v| v.as_i64().expect("greedy token ids are integers"))
        .collect();
    assert_eq!(
        token_ids,
        anchors::GREEDY_TOKENS.to_vec(),
        "greedy decode tokens drifted from the reviewed anchor"
    );
}
