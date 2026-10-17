use super::*;
use crate::model::qwen35_config::Qwen35Config;

fn real_vision_cfg() -> VisionModelConfig {
    VisionModelConfig {
        depth: 12,
        hidden_size: 768,
        num_heads: 12,
        patch_size: 16,
        spatial_merge_size: 2,
        out_hidden_size: 1024,
        temporal_patch_size: 2,
        num_position_embeddings: 2304,
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: None,
    }
}

#[test]
fn missing_manifest_is_a_descriptive_error_not_a_panic() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = real_vision_cfg();
    let err = load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("a directory with neither manifest must be a hard error");
    let msg = err.to_string();
    assert!(
        msg.contains("model.safetensors.index.json") || msg.contains("quantize_index.json"),
        "error must name the missing manifests: {msg}"
    );
}

#[test]
fn tensor_names_count_matches_153_for_depth_12() {
    let cfg = real_vision_cfg();
    // 2 (patch_embed) + 1 (pos_embed) + 12 * 12 (blocks) + 6 (merger) = 153.
    assert_eq!(tensor_names(&cfg).len(), 153);
}

#[test]
fn depth_zero_vision_config_rejected_at_loader_boundary() {
    // `load_qwen35_vision_weights` is public, so a caller can hand it a directly
    // constructed `VisionModelConfig` that never went through
    // `Qwen35Config::from_config_json_str`'s parse-time validation. depth: 0 must
    // still fail closed here, before the (nonexistent) directory is even touched.
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = real_vision_cfg();
    cfg.depth = 0;
    let err = load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("depth: 0 must be rejected at the public loader boundary");
    assert!(
        err.to_string().contains("depth"),
        "error must name depth: {err}"
    );
}

#[test]
fn num_heads_zero_vision_config_rejected_at_loader_boundary() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = real_vision_cfg();
    cfg.num_heads = 0;
    let err = load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("num_heads: 0 must be rejected at the public loader boundary");
    assert!(
        err.to_string().contains("num_heads"),
        "error must name num_heads: {err}"
    );
}

#[test]
fn q4_full_inventory_with_depth_zero_is_rejected() {
    // A checkpoint that genuinely carries the full 153-tensor real inventory, paired
    // with a (malformed) vision_config claiming depth: 0, must error rather than
    // silently returning a nine-tensor `Qwen35VisionWeights`.
    let tmp = tempfile::tempdir().unwrap();
    let full_cfg = real_vision_cfg();
    let entries: Vec<String> = tensor_names(&full_cfg)
        .into_iter()
        .map(|name| format!(r#"{{"name":"{name}","file":"missing.f16","quantized":false}}"#))
        .collect();
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        format!("[{}]", entries.join(",")),
    )
    .expect("test setup: write manifest");

    let mut depth_zero_cfg = full_cfg;
    depth_zero_cfg.depth = 0;
    let err = load_qwen35_vision_weights(tmp.path(), &depth_zero_cfg)
        .expect_err("depth: 0 with a full 153-entry inventory present must still be rejected");
    assert!(
        err.to_string().contains("depth"),
        "error must name depth: {err}"
    );
}

fn tiny_vision_cfg() -> VisionModelConfig {
    VisionModelConfig {
        depth: 1,
        hidden_size: 4,
        num_heads: 2,
        patch_size: 2,
        spatial_merge_size: 1,
        out_hidden_size: 4,
        temporal_patch_size: 1,
        num_position_embeddings: 4,
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: None,
    }
}

/// The 21 expected (name, shape) pairs for [`tiny_vision_cfg`], computed
/// independently of `assemble`'s derivation so a fixture bug can't cancel out a
/// production bug.
fn tiny_expected_shapes() -> Vec<(String, Vec<usize>)> {
    let hidden = 4;
    let qkv_out = 3 * hidden;
    let mlp_intermediate = 4 * hidden;
    let merge_in = hidden; // spatial_merge_size^2 (1) * hidden
    let out_hidden = 4;
    let mut v = vec![
        (
            // [hidden, in_channels, temporal_patch_size, patch_size, patch_size]
            "model.visual.patch_embed.proj.weight".to_string(),
            vec![hidden, 3, 1, 2, 2],
        ),
        (
            "model.visual.patch_embed.proj.bias".to_string(),
            vec![hidden],
        ),
        ("model.visual.pos_embed.weight".to_string(), vec![4, hidden]),
        (
            "model.visual.merger.linear_fc1.weight".to_string(),
            vec![merge_in, merge_in],
        ),
        (
            "model.visual.merger.linear_fc1.bias".to_string(),
            vec![merge_in],
        ),
        (
            "model.visual.merger.linear_fc2.weight".to_string(),
            vec![out_hidden, merge_in],
        ),
        (
            "model.visual.merger.linear_fc2.bias".to_string(),
            vec![out_hidden],
        ),
        ("model.visual.merger.norm.weight".to_string(), vec![hidden]),
        ("model.visual.merger.norm.bias".to_string(), vec![hidden]),
    ];
    for (suffix, shape) in [
        ("attn.qkv.weight", vec![qkv_out, hidden]),
        ("attn.qkv.bias", vec![qkv_out]),
        ("attn.proj.weight", vec![hidden, hidden]),
        ("attn.proj.bias", vec![hidden]),
        ("mlp.linear_fc1.weight", vec![mlp_intermediate, hidden]),
        ("mlp.linear_fc1.bias", vec![mlp_intermediate]),
        ("mlp.linear_fc2.weight", vec![hidden, mlp_intermediate]),
        ("mlp.linear_fc2.bias", vec![hidden]),
        ("norm1.weight", vec![hidden]),
        ("norm1.bias", vec![hidden]),
        ("norm2.weight", vec![hidden]),
        ("norm2.bias", vec![hidden]),
    ] {
        v.push((format!("model.visual.blocks.0.{suffix}"), shape));
    }
    v
}

/// A [`TensorSource`] that records every name passed to `get_f32_tensor_owned`, so a
/// test can assert a specific tensor's shape mismatch is caught from the
/// header-declared shape without that tensor's data ever being copied (other,
/// correctly-shaped tensors ahead of it in iteration order are still fetched
/// normally).
struct CountingSource {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    materialized: std::cell::RefCell<std::collections::HashSet<String>>,
}

impl TensorSource for CountingSource {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
        Ok(self.tensors.contains_key(name))
    }
    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        Ok(self.tensors.get(name).map(|(_, s)| s.clone()))
    }
    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        self.materialized.borrow_mut().insert(name.to_string());
        self.tensors
            .get(name)
            .map(|(d, s)| (d.clone(), s.clone()))
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }
}

#[test]
fn cancellable_loader_stops_before_touching_the_model_directory() {
    let result = load_qwen35_vision_weights_with_cancel(
        Path::new("/path/that/must/not/be/read"),
        &tiny_vision_cfg(),
        &mut || true,
    )
    .expect("cancellation is not a loader failure");
    assert!(result.is_none());
}

#[test]
fn tensor_fetch_polls_cancellation_between_materializations() {
    let cfg = tiny_vision_cfg();
    let names = tensor_names(&cfg);
    let first_name = names[0].clone();
    let tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = tiny_expected_shapes()
        .into_iter()
        .map(|(name, shape)| {
            let numel: usize = shape.iter().product();
            (name, (vec![0.5f32; numel], shape))
        })
        .collect();
    let mut source = CountingSource {
        tensors,
        materialized: std::cell::RefCell::new(std::collections::HashSet::new()),
    };
    let mut polls = 0;
    let result = fetch_expected_tensors_with_cancel(&mut source, names, &cfg, &mut || {
        polls += 1;
        polls == 3
    })
    .expect("cancellation is not a tensor-fetch failure");
    assert!(result.is_none());
    assert_eq!(polls, 3);
    assert_eq!(source.materialized.borrow().len(), 1);
    assert!(source.materialized.borrow().contains(&first_name));
}

#[test]
fn tensor_fetch_accepts_configured_non_four_x_mlp_width() {
    let mut cfg = tiny_vision_cfg();
    cfg.hidden_size = 1152;
    cfg.intermediate_size = Some(4304);
    let expected = [
        (
            "model.visual.blocks.0.mlp.linear_fc1.weight",
            vec![4304, 1152],
        ),
        ("model.visual.blocks.0.mlp.linear_fc1.bias", vec![4304]),
        (
            "model.visual.blocks.0.mlp.linear_fc2.weight",
            vec![1152, 4304],
        ),
        ("model.visual.blocks.0.mlp.linear_fc2.bias", vec![1152]),
    ];
    let names = expected
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect::<Vec<_>>();
    let tensors = expected
        .iter()
        .map(|(name, shape)| ((*name).to_string(), (Vec::new(), shape.clone())))
        .collect();
    let mut source = CountingSource {
        tensors,
        materialized: std::cell::RefCell::new(std::collections::HashSet::new()),
    };

    let fetched = fetch_expected_tensors(&mut source, names.clone(), &cfg)
        .expect("configured 1152/4304 MLP shapes must pass loader preflight");
    assert_eq!(fetched.len(), names.len());
    for name in names {
        assert!(
            source.materialized.borrow().contains(&name),
            "{name} must reach materialization after passing preflight"
        );
    }
}

/// An undersized (but declared-shape-visible) block tensor must be rejected from
/// its header-declared shape, before `get_f32_tensor_owned` ever copies its data.
/// `assemble`'s own post-materialization check would also catch this shape
/// mismatch, so `expect_err` alone would not be mutation-sensitive to the
/// preflight; the "mutated name was never materialized" assertion is what pins the
/// preflight-before-materialize ordering specifically.
#[test]
fn fetch_expected_tensors_rejects_undersized_tensor_before_materialization() {
    let cfg = tiny_vision_cfg();
    let names = tensor_names(&cfg);
    let mutated_name = "model.visual.blocks.0.mlp.linear_fc1.weight".to_string();
    let hidden = cfg.hidden_size;

    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = tiny_expected_shapes()
        .into_iter()
        .map(|(name, shape)| {
            let numel: usize = shape.iter().product();
            (name, (vec![0.5f32; numel], shape))
        })
        .collect();
    // Declared shape undersized relative to [4*hidden, hidden], data consistent
    // with the (wrong) declared shape.
    tensors.insert(
        mutated_name.clone(),
        (
            vec![0.5f32; (4 * hidden - 1) * hidden],
            vec![4 * hidden - 1, hidden],
        ),
    );
    let mut source = CountingSource {
        tensors,
        materialized: std::cell::RefCell::new(std::collections::HashSet::new()),
    };

    let err = fetch_expected_tensors(&mut source, names, &cfg)
        .expect_err("undersized fc1 weight must be rejected");
    match err {
        InferenceError::ShapeMismatch { name, .. } => {
            assert_eq!(name, mutated_name, "error must name the mutated tensor");
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
    assert!(
        !source.materialized.borrow().contains(&mutated_name),
        "the mismatched tensor's data must never be copied"
    );
}

/// Corrupt the FC2 weight entry in `shapes` to a same-numel transposition
/// (`[hidden, mlp_intermediate]` -> `[mlp_intermediate, hidden]`; both have 64
/// elements for [`tiny_vision_cfg`]).
fn transpose_fc2_weight(shapes: &mut [(String, Vec<usize>)]) {
    for (name, shape) in shapes.iter_mut() {
        if name == "model.visual.blocks.0.mlp.linear_fc2.weight" {
            assert_eq!(
                *shape,
                vec![4, 16],
                "fixture assumption for tiny_vision_cfg"
            );
            *shape = vec![16, 4];
            return;
        }
    }
    panic!("fc2 weight entry not found in fixture");
}

fn write_multi_f32_tensor_shard(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header_parts = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        let shape_str = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header_parts.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[{start},{end}]}}"#
        ));
    }
    let header = format!("{{{}}}", header_parts.join(","));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(path, &bytes).expect("test setup: write shard");
}

fn tiny_f32_tensors(value: f32) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    tiny_expected_shapes()
        .into_iter()
        .map(|(name, shape)| {
            let numel: usize = shape.iter().product();
            (name, shape, vec![value; numel])
        })
        .collect()
}

fn assert_vision_weights_eq(actual: &Qwen35VisionWeights, expected: &Qwen35VisionWeights) {
    assert_eq!(actual.patch_embed_weight, expected.patch_embed_weight);
    assert_eq!(
        actual.patch_embed_weight_shape,
        expected.patch_embed_weight_shape
    );
    assert_eq!(actual.patch_embed_bias, expected.patch_embed_bias);
    assert_eq!(actual.pos_embed, expected.pos_embed);
    assert_eq!(actual.blocks.len(), expected.blocks.len());
    for (actual, expected) in actual.blocks.iter().zip(&expected.blocks) {
        assert_eq!(actual.qkv_weight, expected.qkv_weight);
        assert_eq!(actual.qkv_bias, expected.qkv_bias);
        assert_eq!(actual.proj_weight, expected.proj_weight);
        assert_eq!(actual.proj_bias, expected.proj_bias);
        assert_eq!(actual.fc1_weight, expected.fc1_weight);
        assert_eq!(actual.fc1_bias, expected.fc1_bias);
        assert_eq!(actual.fc2_weight, expected.fc2_weight);
        assert_eq!(actual.fc2_bias, expected.fc2_bias);
        assert_eq!(actual.norm1_weight, expected.norm1_weight);
        assert_eq!(actual.norm1_bias, expected.norm1_bias);
        assert_eq!(actual.norm2_weight, expected.norm2_weight);
        assert_eq!(actual.norm2_bias, expected.norm2_bias);
    }
    assert_eq!(actual.merger.fc1_weight, expected.merger.fc1_weight);
    assert_eq!(actual.merger.fc1_bias, expected.merger.fc1_bias);
    assert_eq!(actual.merger.fc2_weight, expected.merger.fc2_weight);
    assert_eq!(actual.merger.fc2_bias, expected.merger.fc2_bias);
    assert_eq!(actual.merger.norm_weight, expected.merger.norm_weight);
    assert_eq!(actual.merger.norm_bias, expected.merger.norm_bias);
}

#[test]
fn single_file_checkpoint_loads_without_an_index() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let tensors = tiny_f32_tensors(0.5);
    write_multi_f32_tensor_shard(&tmp.path().join("model.safetensors"), &tensors);

    let weights = load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect("a valid single model.safetensors must not require a synthetic index");
    assert_eq!(weights.tensor_count(), tensors.len());
}

#[cfg(unix)]
#[test]
fn unindexed_model_safetensors_symlink_to_file_is_a_valid_candidate() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("checkpoint-blob"), b"fixture")
        .expect("test setup: write symlink target");
    symlink("checkpoint-blob", tmp.path().join("model.safetensors"))
        .expect("test setup: create model.safetensors symlink");

    let resolved = resolve_qwen35_single_decoder_safetensors(tmp.path())
        .expect("a safetensors symlink resolving to a file is a supported candidate");
    assert_eq!(resolved, tmp.path().join("model.safetensors"));
}

#[test]
fn single_file_inventory_preflight_accepts_valid_header() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let tensors = tiny_f32_tensors(0.5);
    write_multi_f32_tensor_shard(&tmp.path().join("model.safetensors"), &tensors);

    validate_qwen35_vision_weight_inventory(tmp.path(), &cfg)
        .expect("single-file structural preflight must accept the exact visual inventory");
}

#[test]
fn single_file_checkpoint_rejects_ambiguous_unindexed_candidates() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let tensors = tiny_f32_tensors(0.5);
    write_multi_f32_tensor_shard(&tmp.path().join("model.safetensors"), &tensors);
    std::fs::write(tmp.path().join("leftover.safetensors"), b"not a checkpoint")
        .expect("test setup: write second candidate");

    let err = load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("multiple unindexed safetensors files must be ambiguous");
    let msg = err.to_string();
    assert!(msg.contains("ambiguous unindexed checkpoint"), "got: {msg}");
    assert!(msg.contains("2 safetensors files"), "got: {msg}");
}

#[test]
fn single_file_inventory_preflight_rejects_missing_visual_tensor() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let omitted = "model.visual.merger.norm.bias";
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = tiny_expected_shapes()
        .into_iter()
        .filter(|(name, _)| name != omitted)
        .map(|(name, shape)| {
            let numel: usize = shape.iter().product();
            (name, shape, vec![0.5f32; numel])
        })
        .collect();
    write_multi_f32_tensor_shard(&tmp.path().join("model.safetensors"), &tensors);

    let err = validate_qwen35_vision_weight_inventory(tmp.path(), &cfg)
        .expect_err("an incomplete visual inventory must fail structural preflight");
    assert!(err.to_string().contains("inventory mismatch"), "got: {err}");
}

#[test]
fn single_file_checkpoint_rejects_corrupt_header() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    std::fs::write(tmp.path().join("model.safetensors"), u64::MAX.to_le_bytes())
        .expect("test setup: write corrupt checkpoint");

    let err = load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("a corrupt safetensors header must fail before tensor loading");
    assert!(
        matches!(err, InferenceError::InvalidSafetensors(_)),
        "got: {err:?}"
    );
}

#[test]
fn indexed_checkpoint_path_remains_preferred_over_plain_file() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let tensors = tiny_f32_tensors(0.5);
    write_multi_f32_tensor_shard(&tmp.path().join("model.safetensors"), &tensors);
    std::fs::write(
        tmp.path().join("model.safetensors.index.json"),
        b"not valid json",
    )
    .expect("test setup: write invalid index");

    load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("an existing index must not silently fall back to model.safetensors");
}

#[cfg(unix)]
#[test]
fn dangling_index_sentinel_does_not_fall_back_to_plain_file() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let tensors = tiny_f32_tensors(0.5);
    write_multi_f32_tensor_shard(&tmp.path().join("model.safetensors"), &tensors);
    symlink(
        "missing-index-target.json",
        tmp.path().join("model.safetensors.index.json"),
    )
    .expect("test setup: create dangling index sentinel");

    resolve_qwen35_single_decoder_safetensors(tmp.path())
        .expect_err("a dangling authoritative index must not resolve the plain file");
    load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("a dangling authoritative index must not load the plain file");
}

#[cfg(unix)]
#[test]
fn dangling_quantize_sentinel_does_not_fall_back_to_plain_file() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let tensors = tiny_f32_tensors(0.5);
    write_multi_f32_tensor_shard(&tmp.path().join("model.safetensors"), &tensors);
    symlink(
        "missing-quantize-target.json",
        tmp.path().join("quantize_index.json"),
    )
    .expect("test setup: create dangling quantized sentinel");

    load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("a dangling authoritative quantized manifest must not load the plain file");
}

#[test]
fn indexed_layout_binds_visual_and_decoder_resolution_to_the_same_shard() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let shapes = tiny_expected_shapes();
    let plain_tensors = tiny_f32_tensors(0.1);
    write_multi_f32_tensor_shard(&tmp.path().join("model.safetensors"), &plain_tensors);

    let shard_name = "model-00001-of-00001.safetensors";
    let indexed_tensors = tiny_f32_tensors(0.9);
    write_multi_f32_tensor_shard(&tmp.path().join(shard_name), &indexed_tensors);
    let weight_map = shapes
        .iter()
        .map(|(name, _)| format!(r#""{name}":"{shard_name}""#))
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(
        tmp.path().join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{{weight_map}}}}}"#),
    )
    .expect("test setup: write index");

    let resolved =
        resolve_qwen35_single_decoder_safetensors(tmp.path()).expect("one-shard index resolves");
    assert_eq!(resolved, tmp.path().join(shard_name));
    let weights =
        load_qwen35_vision_weights(tmp.path(), &cfg).expect("indexed visual checkpoint loads");
    assert_eq!(weights.patch_embed_weight[0], 0.9f32);
}

#[test]
fn single_file_and_one_shard_index_load_identical_vision_weights() {
    let single = tempfile::tempdir().unwrap();
    let indexed = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = tiny_expected_shapes()
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let numel: usize = shape.iter().product();
            (name, shape, vec![i as f32 / 100.0; numel])
        })
        .collect();
    write_multi_f32_tensor_shard(&single.path().join("model.safetensors"), &tensors);

    let shard_name = "model-00001-of-00001.safetensors";
    write_multi_f32_tensor_shard(&indexed.path().join(shard_name), &tensors);
    let weight_map = tensors
        .iter()
        .map(|(name, _, _)| format!(r#""{name}":"{shard_name}""#))
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(
        indexed.path().join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{{weight_map}}}}}"#),
    )
    .expect("test setup: write index");

    let from_single =
        load_qwen35_vision_weights(single.path(), &cfg).expect("single-file checkpoint loads");
    let from_index = load_qwen35_vision_weights(indexed.path(), &cfg)
        .expect("one-shard indexed checkpoint loads");
    assert_vision_weights_eq(&from_single, &from_index);
}

fn assert_fc2_shape_mismatch(result: Result<Qwen35VisionWeights, InferenceError>) {
    let err = result.expect_err("transposed FC2 (same numel, wrong shape) must be rejected");
    match err {
        InferenceError::ShapeMismatch { name, .. } => {
            assert!(
                name.contains("fc2"),
                "expected FC2 shape mismatch, got {name}"
            )
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn fp16_same_numel_transposed_fc2_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let mut shapes = tiny_expected_shapes();
    transpose_fc2_weight(&mut shapes);

    let shard = tmp.path().join("model-00001-of-00001.safetensors");
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = shapes
        .iter()
        .map(|(name, shape)| {
            let numel: usize = shape.iter().product();
            (name.clone(), shape.clone(), vec![0.5f32; numel])
        })
        .collect();
    write_multi_f32_tensor_shard(&shard, &tensors);

    let weight_map = shapes
        .iter()
        .map(|(name, _)| format!(r#""{name}":"model-00001-of-00001.safetensors""#))
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(
        tmp.path().join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{{weight_map}}}}}"#),
    )
    .expect("test setup: write index");

    assert_fc2_shape_mismatch(load_qwen35_vision_weights(tmp.path(), &cfg));
}

#[test]
fn q4_same_numel_transposed_fc2_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let mut shapes = tiny_expected_shapes();
    transpose_fc2_weight(&mut shapes);

    let mut manifest_entries = Vec::new();
    for (i, (name, shape)) in shapes.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = vec![0.25_f64; numel];
        let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&data, shape)
            .expect("quantize succeeds");
        let file_name = format!("t{i}.q4");
        crate::weights::q4_weights::save_q4_file(&tmp.path().join(&file_name), &q4)
            .expect("test setup: write q4 file");
        manifest_entries.push(format!(
            r#"{{"name":"{name}","file":"{file_name}","quantized":true}}"#
        ));
    }
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        format!("[{}]", manifest_entries.join(",")),
    )
    .expect("test setup: write manifest");

    assert_fc2_shape_mismatch(load_qwen35_vision_weights(tmp.path(), &cfg));
}

/// #1069: quantize_index.json file entries are untrusted checkpoint
/// content; an entry escaping the model directory must be rejected
/// before its file is read.
#[test]
fn q4_manifest_entry_escaping_model_dir_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    std::fs::create_dir_all(&model_dir).expect("test setup");
    // A structurally valid q4 file OUTSIDE the model dir — containment,
    // not file validity, must be what rejects the entry.
    let data = vec![0.25_f64; 4];
    let q4 =
        crate::weights::q4_weights::quantize_f64_to_q4(&data, &[2, 2]).expect("quantize succeeds");
    crate::weights::q4_weights::save_q4_file(&tmp.path().join("evil.q4"), &q4)
        .expect("test setup: write q4 file");
    std::fs::write(
        model_dir.join("quantize_index.json"),
        r#"[{"name":"model.visual.patch_embed.proj.weight","file":"../evil.q4","quantized":true}]"#,
    )
    .expect("test setup: write manifest");

    let err = load_qwen35_vision_weights(&model_dir, &tiny_vision_cfg())
        .expect_err("escaping manifest entry must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("must stay within the model directory") || msg.contains("escapes model root"),
        "unexpected error: {err}"
    );
}

/// The manifest's declared shape must be checked against `vision_cfg` BEFORE the
/// tensor's file is opened, matching what `fetch_expected_tensors` does on the fp16
/// path.
///
/// Pointing the entry at a file that does not exist is what makes this
/// mutation-sensitive. With the preflight the loader never gets that far and returns
/// ShapeMismatch; without it the loader tries to read `missing.q4` and reports a read
/// failure instead. Asserting merely that *some* error came back would pass either
/// way and guard nothing.
#[test]
fn q4_manifest_shape_disagreeing_with_config_is_rejected_before_the_file_is_read() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
            tmp.path().join("quantize_index.json"),
            r#"[{"name":"model.visual.patch_embed.proj.bias","file":"missing.q4","quantized":true,"shape":[9999]}]"#,
        )
        .expect("test setup: write manifest");
    assert!(
        !tmp.path().join("missing.q4").exists(),
        "test setup: the tensor file must NOT exist, that absence is the assertion"
    );

    let err = load_qwen35_vision_weights(tmp.path(), &tiny_vision_cfg())
        .expect_err("a manifest shape contradicting vision_cfg must be rejected");
    match err {
        InferenceError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "model.visual.patch_embed.proj.bias");
            assert_eq!(expected, vec![4]);
            assert_eq!(actual, vec![9999]);
        }
        other => panic!("expected ShapeMismatch before any file read, got: {other}"),
    }
}

/// When the manifest omits `shape`, the `.q4` header carries the only declared shape
/// there is and the manifest-level preflight has nothing to compare. The header must
/// then be checked against `vision_cfg` before `dequantize_q4_to_f32` allocates the
/// full f32 buffer.
///
/// Mutation-sensitivity needs care here, and it is exactly the trap the fp16-path
/// test author already called out: `assemble` rejects a contradictory shape too, so
/// `expect_err` alone still passes with the fix reverted. The discriminator is WHICH
/// error comes back. This manifest carries exactly one tensor and it is deliberately
/// not the one `assemble` takes first, so with the header check removed the loader
/// dequantizes happily and then dies on MissingTensor for `patch_embed.proj.weight`,
/// a different error about a different tensor.
#[test]
fn q4_header_shape_disagreeing_with_config_is_rejected_before_dequantization() {
    let tmp = tempfile::tempdir().unwrap();
    let data = vec![0.25_f64; 64];
    let q4 =
        crate::weights::q4_weights::quantize_f64_to_q4(&data, &[64]).expect("quantize succeeds");
    crate::weights::q4_weights::save_q4_file(&tmp.path().join("t0.q4"), &q4)
        .expect("test setup: write q4 file");
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        r#"[{"name":"model.visual.patch_embed.proj.bias","file":"t0.q4","quantized":true}]"#,
    )
    .expect("test setup: write manifest");

    let err = load_qwen35_vision_weights(tmp.path(), &tiny_vision_cfg())
        .expect_err("a q4 header shape contradicting vision_cfg must be rejected");
    match err {
        InferenceError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "model.visual.patch_embed.proj.bias");
            assert_eq!(expected, vec![4]);
            assert_eq!(actual, vec![64]);
        }
        other => panic!("expected ShapeMismatch before dequantization, got: {other}"),
    }
}

fn write_khf1_f16_file(path: &Path, shape: &[usize], values: &[f32]) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHF1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for d in shape {
        buf.extend_from_slice(&(*d as u64).to_le_bytes());
    }
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        buf.extend_from_slice(&crate::weights::half_bits::f32_to_f16_bits(*v).to_le_bytes());
    }
    std::fs::write(path, &buf).expect("test setup: write f16 file");
}

#[test]
fn f16_companion_same_numel_transposed_fc2_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let mut shapes = tiny_expected_shapes();
    transpose_fc2_weight(&mut shapes);

    let mut manifest_entries = Vec::new();
    for (i, (name, shape)) in shapes.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let values = vec![0.5f32; numel];
        let file_name = format!("t{i}.f16");
        write_khf1_f16_file(&tmp.path().join(&file_name), shape, &values);
        manifest_entries.push(format!(
            r#"{{"name":"{name}","file":"{file_name}","quantized":false}}"#
        ));
    }
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        format!("[{}]", manifest_entries.join(",")),
    )
    .expect("test setup: write manifest");

    assert_fc2_shape_mismatch(load_qwen35_vision_weights(tmp.path(), &cfg));
}

/// The `.f16` companion arm needs its own header check, and this is the case that
/// proves it: a non-quantized entry whose manifest omits `shape` skips the preflight
/// before `contained_shard_path` (there is no declared shape to compare) and does not
/// reach the `.q4` header check (that arm is not taken). Without a header check here
/// the tensor is fully read and converted before `assemble` notices.
///
/// The single manifest entry is the bias rather than `patch_embed.proj.weight`, so
/// without the guard the loader gets far enough for `assemble` to report the absent
/// weight as `MissingTensor`. That makes the assertion below discriminating: it can
/// only pass if the header was checked before materialization, not merely because
/// loading failed somewhere.
#[test]
fn f16_header_shape_disagreeing_with_config_is_rejected_before_materialization() {
    let tmp = tempfile::tempdir().unwrap();
    write_khf1_f16_file(&tmp.path().join("t0.f16"), &[64], &vec![0.5f32; 64]);
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        r#"[{"name":"model.visual.patch_embed.proj.bias","file":"t0.f16","quantized":false}]"#,
    )
    .expect("test setup: write manifest");

    let err = load_qwen35_vision_weights(tmp.path(), &tiny_vision_cfg())
        .expect_err("an f16 header shape contradicting vision_cfg must be rejected");
    match err {
        InferenceError::ShapeMismatch {
            name,
            expected,
            actual,
        } => {
            assert_eq!(name, "model.visual.patch_embed.proj.bias");
            assert_eq!(expected, vec![4]);
            assert_eq!(actual, vec![64]);
        }
        other => panic!("expected ShapeMismatch before materialization, got: {other}"),
    }
}

/// Proves the shape comparison happens BEFORE the payload is read, rather than merely
/// before `assemble`.
///
/// The fixture's header declares 64 elements but the file carries no payload bytes at
/// all. If the loader read the payload first it would fail on the truncated read and
/// surface `InvalidSafetensors`; only a loader that compares the header against the
/// expected shape first can return `ShapeMismatch` for this input. The two outcomes
/// are therefore distinguishable, which is the whole point of the fixture.
///
/// What this does not prove is the same-handle property. That the header and payload
/// come from one open cannot be shown without a filesystem seam to swap the file
/// mid-load; it is structural, held by `load_f16_tensor_file_expecting` performing a
/// single open, and this test would still pass if that were split back into two.
#[test]
fn f16_shape_is_compared_before_the_payload_is_read() {
    let tmp = tempfile::tempdir().unwrap();
    let mut buf = Vec::new();
    buf.extend_from_slice(b"KHF1");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&64u64.to_le_bytes());
    buf.extend_from_slice(&64u64.to_le_bytes());
    // Deliberately no payload: 64 declared elements, zero bytes of data.
    std::fs::write(tmp.path().join("t0.f16"), &buf).expect("test setup: write f16 header");
    std::fs::write(
        tmp.path().join("quantize_index.json"),
        r#"[{"name":"model.visual.patch_embed.proj.bias","file":"t0.f16","quantized":false}]"#,
    )
    .expect("test setup: write manifest");

    let err = load_qwen35_vision_weights(tmp.path(), &tiny_vision_cfg())
        .expect_err("a header disagreeing with vision_cfg must be rejected");
    match err {
        InferenceError::ShapeMismatch { actual, .. } => assert_eq!(actual, vec![64]),
        other => panic!(
            "expected ShapeMismatch from the header check, which proves the payload \
                 read was never attempted; got: {other}"
        ),
    }
}

#[test]
fn q4_manifest_duplicate_visual_tensor_name_is_rejected() {
    // A tiny valid q4 inventory (all 21 entries `tiny_vision_cfg` expects, each with
    // a correct shape) plus one repeated `model.visual.*` name must be rejected
    // before `assemble`'s inventory-exactness check ever runs -- otherwise
    // `HashMap::insert` would silently let the duplicate's file overwrite the
    // first's, and the loader would return `Ok` depending on manifest entry order.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let shapes = tiny_expected_shapes();

    let mut manifest_entries = Vec::new();
    for (i, (name, shape)) in shapes.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = vec![0.25_f64; numel];
        let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&data, shape)
            .expect("quantize succeeds");
        let file_name = format!("t{i}.q4");
        crate::weights::q4_weights::save_q4_file(&tmp.path().join(&file_name), &q4)
            .expect("test setup: write q4 file");
        manifest_entries.push(format!(
            r#"{{"name":"{name}","file":"{file_name}","quantized":true}}"#
        ));
    }
    // Duplicate: same name and shape as an existing entry, written to a distinct
    // file so a naive "does the file exist" check would not catch it.
    let (dup_name, dup_shape) = &shapes[0];
    let dup_numel: usize = dup_shape.iter().product();
    let dup_q4 =
        crate::weights::q4_weights::quantize_f64_to_q4(&vec![0.5_f64; dup_numel], dup_shape)
            .expect("quantize succeeds");
    let dup_file_name = "dup.q4";
    crate::weights::q4_weights::save_q4_file(&tmp.path().join(dup_file_name), &dup_q4)
        .expect("test setup: write duplicate q4 file");
    manifest_entries.push(format!(
        r#"{{"name":"{dup_name}","file":"{dup_file_name}","quantized":true}}"#
    ));

    std::fs::write(
        tmp.path().join("quantize_index.json"),
        format!("[{}]", manifest_entries.join(",")),
    )
    .expect("test setup: write manifest");

    let err = load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("a duplicate model.visual.* manifest entry must be rejected");
    match err {
        InferenceError::Inference(msg) => {
            assert!(
                msg.contains("duplicate") && msg.contains(dup_name),
                "expected a duplicate-entry error naming {dup_name}, got: {msg}"
            );
        }
        other => panic!("expected InferenceError::Inference, got {other:?}"),
    }
}

#[test]
fn q4_manifest_unexpected_visual_tensor_rejected_before_dequantizing() {
    // FIX 4 regression: a manifest entry whose name is NOT in the set
    // `vision_cfg` expects must be rejected before its file is even opened -- not
    // decoded and only caught afterward by `assemble`'s "unconsumed leftover" check.
    // A valid tiny inventory plus one hostile extra entry proves the new pre-decode
    // name check fires; the assertion on the error text ("unexpected" + naming the
    // rejected tensor, not "unconsumed") distinguishes this from the old ordering.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let shapes = tiny_expected_shapes();

    let mut manifest_entries = Vec::new();
    for (i, (name, shape)) in shapes.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let data: Vec<f64> = vec![0.25_f64; numel];
        let q4 = crate::weights::q4_weights::quantize_f64_to_q4(&data, shape)
            .expect("quantize succeeds");
        let file_name = format!("t{i}.q4");
        crate::weights::q4_weights::save_q4_file(&tmp.path().join(&file_name), &q4)
            .expect("test setup: write q4 file");
        manifest_entries.push(format!(
            r#"{{"name":"{name}","file":"{file_name}","quantized":true}}"#
        ));
    }
    // An extra, valid, individually in-budget q4 tensor whose name `vision_cfg`
    // (depth=1) never asks for. Deliberately does NOT contain the substring
    // "unexpected" -- the post-decode "unconsumed leftover" fallback check (see
    // `assemble`) also names the rejected tensor in its error text, so asserting on
    // "unexpected" + the raw name alone would pass via either code path and fail to
    // discriminate a pre-decode rejection from a post-decode one.
    let hostile_name = "model.visual.blocks.0.rogue_injected_tensor";
    let hostile_q4 = crate::weights::q4_weights::quantize_f64_to_q4(&[0.5_f64; 4], &[4])
        .expect("quantize succeeds");
    let hostile_file_name = "hostile.q4";
    crate::weights::q4_weights::save_q4_file(&tmp.path().join(hostile_file_name), &hostile_q4)
        .expect("test setup: write hostile q4 file");
    manifest_entries.push(format!(
        r#"{{"name":"{hostile_name}","file":"{hostile_file_name}","quantized":true}}"#
    ));

    std::fs::write(
        tmp.path().join("quantize_index.json"),
        format!("[{}]", manifest_entries.join(",")),
    )
    .expect("test setup: write manifest");

    let err = load_qwen35_vision_weights(tmp.path(), &cfg)
        .expect_err("an unexpected model.visual.* manifest entry must be rejected");
    match err {
        InferenceError::Inference(msg) => {
            assert!(
                msg.contains("rejected before dequantizing/decoding") && msg.contains(hostile_name),
                "expected a pre-decode unexpected-tensor error naming {hostile_name}, \
                     got: {msg}"
            );
        }
        other => panic!("expected InferenceError::Inference, got {other:?}"),
    }
}

#[test]
fn fp16_sharded_index_duplicate_raw_weight_map_key_is_rejected() {
    // A raw `weight_map` JSON object can name the same tensor twice, mapped to two
    // different shards. Ordinary map deserialization collapses that to one
    // `HashMap` entry (last member wins) *before* the inventory-count check ever
    // runs, so the count stays exact and the loader silently resolves the tensor
    // from whichever shard happened to be the last raw JSON member -- raw-member-
    // order-dependent, not a deterministic error. All 21 tiny names resolve to
    // shard A; one name (`dup_name`) is additionally repeated, resolving to shard B
    // with a different (but shape-correct) value, so a silent last-write-wins
    // collapse would succeed with B's value instead of failing.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = tiny_vision_cfg();
    let shapes = tiny_expected_shapes();

    let shard_a = tmp.path().join("model-00001-of-00002.safetensors");
    let tensors_a: Vec<(String, Vec<usize>, Vec<f32>)> = shapes
        .iter()
        .map(|(name, shape)| {
            let numel: usize = shape.iter().product();
            (name.clone(), shape.clone(), vec![0.5f32; numel])
        })
        .collect();
    write_multi_f32_tensor_shard(&shard_a, &tensors_a);

    let (dup_name, dup_shape) = shapes[0].clone();
    let dup_numel: usize = dup_shape.iter().product();
    let shard_b = tmp.path().join("model-00002-of-00002.safetensors");
    write_multi_f32_tensor_shard(
        &shard_b,
        &[(dup_name.clone(), dup_shape, vec![9.0f32; dup_numel])],
    );

    let mut weight_map_members: Vec<String> = shapes
        .iter()
        .map(|(name, _)| format!(r#""{name}":"model-00001-of-00002.safetensors""#))
        .collect();
    // Duplicate raw member for `dup_name`, pointing at the second shard.
    weight_map_members.push(format!(
        r#""{dup_name}":"model-00002-of-00002.safetensors""#
    ));
    std::fs::write(
        tmp.path().join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{{}}}}}"#, weight_map_members.join(",")),
    )
    .expect("test setup: write index");

    let err = load_qwen35_vision_weights(tmp.path(), &cfg).expect_err(
        "a duplicate raw weight_map key routed to a second shard must be rejected, \
             not silently resolved by raw member order",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate") && msg.contains(&dup_name),
        "expected a duplicate-key error naming {dup_name}, got: {msg}"
    );
}

#[test]
fn assemble_rejects_unconsumed_leftover_tensors_when_depth_understates_inventory() {
    // The q4 path loads every `model.visual.*` manifest entry up front, not just the
    // names `vision_cfg.depth` implies. If the checkpoint has more real block tensors
    // than `depth` accounts for, the leftovers must be a hard error, not silently
    // dropped. The 21 tensors `small_cfg` (depth=1) does expect are given correct
    // shapes so this test exercises the leftover check specifically, not the
    // per-tensor shape check above it.
    let full_cfg = real_vision_cfg(); // depth 12, 153 real tensors
    let mut small_cfg = full_cfg.clone();
    small_cfg.depth = 1;

    let hidden = full_cfg.hidden_size;
    let qkv_out = 3 * hidden;
    let mlp_intermediate = 4 * hidden;
    let merge_in = full_cfg.spatial_merge_size * full_cfg.spatial_merge_size * hidden;
    let out_hidden = full_cfg.out_hidden_size;
    let shape_for = |name: &str| -> Vec<usize> {
        match name {
            "model.visual.patch_embed.proj.weight" => vec![
                hidden,
                full_cfg.in_channels,
                full_cfg.temporal_patch_size,
                full_cfg.patch_size,
                full_cfg.patch_size,
            ],
            "model.visual.pos_embed.weight" => vec![full_cfg.num_position_embeddings, hidden],
            "model.visual.merger.linear_fc1.weight" => vec![merge_in, merge_in],
            "model.visual.merger.linear_fc1.bias" => vec![merge_in],
            "model.visual.merger.linear_fc2.weight" => vec![out_hidden, merge_in],
            "model.visual.merger.linear_fc2.bias" => vec![out_hidden],
            n if n.ends_with("attn.qkv.weight") => vec![qkv_out, hidden],
            n if n.ends_with("attn.qkv.bias") => vec![qkv_out],
            n if n.ends_with("attn.proj.weight") => vec![hidden, hidden],
            n if n.ends_with("mlp.linear_fc1.weight") => vec![mlp_intermediate, hidden],
            n if n.ends_with("mlp.linear_fc1.bias") => vec![mlp_intermediate],
            n if n.ends_with("mlp.linear_fc2.weight") => vec![hidden, mlp_intermediate],
            _ => vec![hidden], // *.bias, norm1/2.*, merger.norm.*
        }
    };

    let expected_names: std::collections::HashSet<String> =
        tensor_names(&small_cfg).into_iter().collect();
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for name in tensor_names(&full_cfg) {
        if expected_names.contains(&name) {
            let shape = shape_for(&name);
            let numel: usize = shape.iter().product();
            tensors.insert(name, (vec![0.0_f32; numel], shape));
        } else {
            // A leftover tensor beyond small_cfg's depth=1 — its shape is irrelevant
            // because it must never be consumed by `take`.
            tensors.insert(name, (vec![0.0_f32], vec![1]));
        }
    }

    let err = assemble(tensors, &small_cfg)
        .expect_err("leftover model.visual.* tensors beyond depth=1 must be rejected");
    match err {
        InferenceError::Inference(msg) => {
            assert!(
                msg.contains("unconsumed"),
                "expected an unconsumed-tensor inventory error, got: {msg}"
            );
        }
        other => panic!("expected InferenceError::Inference, got {other:?}"),
    }
}

#[test]
fn assemble_accepts_official_qwen35_vl_vision_mlp_dims() {
    // FIX 16 regression: official Qwen3.5-VL vision tower dims are hidden_size=1152,
    // intermediate_size=4304 (NOT 4*1152=4608). Both loader preflight and `assemble`
    // must derive the expected MLP shapes from that explicit checkpoint geometry.
    let mut cfg = tiny_vision_cfg();
    cfg.hidden_size = 1152;
    cfg.num_heads = 1; // must divide hidden_size; head geometry is irrelevant here
    cfg.intermediate_size = Some(4304);

    let hidden = cfg.hidden_size;
    let mlp_intermediate = cfg.intermediate_size.unwrap();
    let qkv_out = 3 * hidden;
    let merge_in = cfg.spatial_merge_size * cfg.spatial_merge_size * hidden;
    let out_hidden = cfg.out_hidden_size;
    let shape_for = |name: &str| -> Vec<usize> {
        match name {
            "model.visual.patch_embed.proj.weight" => vec![
                hidden,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ],
            "model.visual.pos_embed.weight" => vec![cfg.num_position_embeddings, hidden],
            "model.visual.merger.linear_fc1.weight" => vec![merge_in, merge_in],
            "model.visual.merger.linear_fc1.bias" => vec![merge_in],
            "model.visual.merger.linear_fc2.weight" => vec![out_hidden, merge_in],
            "model.visual.merger.linear_fc2.bias" => vec![out_hidden],
            n if n.ends_with("attn.qkv.weight") => vec![qkv_out, hidden],
            n if n.ends_with("attn.qkv.bias") => vec![qkv_out],
            n if n.ends_with("attn.proj.weight") => vec![hidden, hidden],
            n if n.ends_with("mlp.linear_fc1.weight") => vec![mlp_intermediate, hidden],
            n if n.ends_with("mlp.linear_fc1.bias") => vec![mlp_intermediate],
            n if n.ends_with("mlp.linear_fc2.weight") => vec![hidden, mlp_intermediate],
            _ => vec![hidden], // *.bias, norm1/2.*, merger.norm.*
        }
    };
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for name in tensor_names(&cfg) {
        let shape = shape_for(&name);
        let numel: usize = shape.iter().product();
        tensors.insert(name, (vec![0.0_f32; numel], shape));
    }

    assemble(tensors, &cfg).expect("official Qwen3.5-VL vision dims (1152/4304) must be accepted");
}

// BF16/F16 safetensors tensors decode to f32 unconditionally (no feature
// gate); this runs in every feature set and self-skips when the checkpoint
// is absent.
#[test]
fn loads_real_fp16_checkpoint_with_correct_shapes() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let model_dir = std::path::PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b"));
    if !model_dir.join("config.json").exists() {
        return; // model not downloaded; skip
    }
    let cfg = Qwen35Config::from_model_dir(&model_dir).expect("0.8b config.json parses");
    let vision_cfg = cfg
        .vision_config
        .expect("released 0.8b checkpoint has a vision_config");

    let weights = load_qwen35_vision_weights(&model_dir, &vision_cfg)
        .expect("fp16 vision weights must load without error");

    assert_eq!(weights.tensor_count(), 153);
    assert_eq!(weights.blocks.len(), 12);
    assert_eq!(weights.patch_embed_weight.len(), 768 * 3 * 2 * 16 * 16);
    assert_eq!(weights.patch_embed_bias.len(), 768);
    assert_eq!(weights.pos_embed.len(), 2304 * 768);

    let block0 = &weights.blocks[0];
    assert_eq!(block0.qkv_weight.len(), 2304 * 768);
    assert_eq!(block0.qkv_bias.len(), 2304);
    assert_eq!(block0.proj_weight.len(), 768 * 768);
    assert_eq!(block0.proj_bias.len(), 768);
    assert_eq!(block0.fc1_weight.len(), 3072 * 768);
    assert_eq!(block0.fc1_bias.len(), 3072);
    assert_eq!(block0.fc2_weight.len(), 768 * 3072);
    assert_eq!(block0.fc2_bias.len(), 768);
    assert_eq!(block0.norm1_weight.len(), 768);
    assert_eq!(block0.norm2_weight.len(), 768);

    assert_eq!(weights.merger.fc1_weight.len(), 3072 * 3072);
    assert_eq!(weights.merger.fc2_weight.len(), 1024 * 3072);
    assert_eq!(weights.merger.norm_weight.len(), 768);
}

#[test]
fn loads_real_q4_checkpoint_with_correct_shapes() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let model_dir = std::path::PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b-q4"));
    if !model_dir.join("config.json").exists() {
        return; // model not downloaded; skip
    }
    let cfg = Qwen35Config::from_model_dir(&model_dir).expect("q4 config.json parses");
    let vision_cfg = cfg
        .vision_config
        .expect("released q4 checkpoint has a vision_config");

    let weights = load_qwen35_vision_weights(&model_dir, &vision_cfg)
        .expect("q4 vision weights must load without error");

    assert_eq!(weights.tensor_count(), 153);
    assert_eq!(weights.blocks.len(), 12);
    assert_eq!(weights.patch_embed_weight.len(), 768 * 3 * 2 * 16 * 16);
    assert_eq!(weights.pos_embed.len(), 2304 * 768);

    let block0 = &weights.blocks[0];
    assert_eq!(block0.qkv_weight.len(), 2304 * 768);
    assert_eq!(block0.proj_weight.len(), 768 * 768);
    assert_eq!(block0.fc1_weight.len(), 3072 * 768);
    assert_eq!(block0.fc2_weight.len(), 768 * 3072);

    assert_eq!(weights.merger.fc1_weight.len(), 3072 * 3072);
    assert_eq!(weights.merger.fc2_weight.len(), 1024 * 3072);

    // Dequantized values must be finite (no NaN/Inf leaking through the q4 path).
    assert!(block0.qkv_weight.iter().all(|v| v.is_finite()));
    assert!(weights.merger.fc2_weight.iter().all(|v| v.is_finite()));
}
