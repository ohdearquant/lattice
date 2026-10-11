use super::*;
use crate::model::embeddinggemma2_image::{PreparedImage, ProcessorIdentity};
use std::collections::BTreeSet;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};

const CONFIG: &str = r#"{
  "vision_config": {
    "model_type": "gemma4_vision", "dtype": "bfloat16",
    "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 16,
    "num_attention_heads": 12, "num_key_value_heads": 12,
    "head_dim": 64, "global_head_dim": 64, "patch_size": 16,
    "pooling_kernel_size": 3, "position_embedding_size": 10240,
    "default_output_length": 280, "rms_norm_eps": 0.000001,
    "hidden_activation": "gelu_pytorch_tanh", "attention_bias": false,
    "standardize": false, "use_clipped_linears": false,
    "rope_parameters": {"rope_type": "axial", "rope_theta": 100.0}
  }
}"#;

fn production_config() -> VisionConfig {
    VisionConfig::from_config_json_str(CONFIG, 512).unwrap()
}

fn tiny_config() -> VisionConfig {
    // This bypass is private to loader tests; production parsing rejects these dimensions.
    VisionConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 16,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        patch_size: 16,
        position_embedding_size: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 100.0,
        text_hidden_size: 8,
    }
}

fn independent_header(cfg: &VisionConfig) -> String {
    let h = cfg.hidden_size;
    let d = cfg.head_dim;
    let q = cfg.num_attention_heads * d;
    let kv = cfg.num_key_value_heads * d;
    let f = cfg.intermediate_size;
    let mut shapes = BTreeMap::new();
    for layer in 0..16 {
        let prefix = format!("vision_tower.encoder.layers.{layer}.");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            shapes.insert(format!("{prefix}{suffix}"), vec![h]);
        }
        for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            shapes.insert(format!("{prefix}{suffix}"), vec![d]);
        }
        for (suffix, shape) in [
            ("self_attn.q_proj.linear.weight", vec![q, h]),
            ("self_attn.k_proj.linear.weight", vec![kv, h]),
            ("self_attn.v_proj.linear.weight", vec![kv, h]),
            ("self_attn.o_proj.linear.weight", vec![h, q]),
            ("mlp.gate_proj.linear.weight", vec![f, h]),
            ("mlp.up_proj.linear.weight", vec![f, h]),
            ("mlp.down_proj.linear.weight", vec![h, f]),
        ] {
            shapes.insert(format!("{prefix}{suffix}"), shape);
        }
    }
    shapes.insert(
        "vision_tower.patch_embedder.input_proj.weight".into(),
        vec![h, 3 * cfg.patch_size * cfg.patch_size],
    );
    shapes.insert(
        "vision_tower.patch_embedder.position_embedding_table".into(),
        vec![2, cfg.position_embedding_size, h],
    );
    shapes.insert(
        "embed_vision.embedding_projection.weight".into(),
        vec![cfg.text_hidden_size, h],
    );
    let mut offset = 0usize;
    let mut header = serde_json::Map::new();
    for (name, shape) in shapes {
        let end = offset + shape.iter().product::<usize>() * 2;
        header.insert(
            name,
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    serde_json::to_string(&header).unwrap()
}

fn records_from_header(header: &str) -> Vec<TensorMetadata> {
    let parsed: serde_json::Value = serde_json::from_str(header).unwrap();
    parsed
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, tensor)| TensorMetadata {
            name: name.clone(),
            shape: tensor["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|size| size.as_u64().unwrap() as usize)
                .collect(),
            dtype: tensor["dtype"].as_str().map(str::to_string),
        })
        .collect()
}

#[test]
fn vision_config_rejects_missing_and_unsupported_fields() {
    let root: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    for name in root["vision_config"].as_object().unwrap().keys() {
        let mut bad = root.clone();
        bad["vision_config"].as_object_mut().unwrap().remove(name);
        let error = VisionConfig::from_config_json_str(&bad.to_string(), 512).unwrap_err();
        assert!(error.to_string().contains(name), "{name}: {error}");
    }
    assert!(
        VisionConfig::from_config_json_str("{}", 512)
            .unwrap_err()
            .to_string()
            .contains("vision_config")
    );
    for (field, value) in [
        ("hidden_size", serde_json::json!(4)),
        ("num_hidden_layers", serde_json::json!(15)),
        ("num_attention_heads", serde_json::json!(0)),
        ("num_key_value_heads", serde_json::json!(1)),
        ("head_dim", serde_json::json!(32)),
        ("global_head_dim", serde_json::json!(32)),
        ("intermediate_size", serde_json::json!(0)),
        ("patch_size", serde_json::json!(8)),
        ("pooling_kernel_size", serde_json::json!(2)),
        ("position_embedding_size", serde_json::json!(0)),
        ("default_output_length", serde_json::json!(256)),
        ("model_type", serde_json::json!("other")),
        ("dtype", serde_json::json!("float32")),
        ("hidden_activation", serde_json::json!("gelu")),
        ("attention_bias", serde_json::json!(true)),
        ("standardize", serde_json::json!(true)),
        ("use_clipped_linears", serde_json::json!(true)),
        ("rms_norm_eps", serde_json::json!(0.0)),
    ] {
        let mut bad = root.clone();
        bad["vision_config"][field] = value;
        let error = VisionConfig::from_config_json_str(&bad.to_string(), 512).unwrap_err();
        assert!(error.to_string().contains(field), "{field}: {error}");
    }
    for (field, value) in [
        ("rope_type", serde_json::json!("default")),
        ("rope_theta", serde_json::json!(0.0)),
    ] {
        let mut bad = root.clone();
        bad["vision_config"]["rope_parameters"][field] = value;
        assert!(
            VisionConfig::from_config_json_str(&bad.to_string(), 512)
                .unwrap_err()
                .to_string()
                .contains(field)
        );
    }
    assert!(
        VisionConfig::from_config_json_str(CONFIG, 8)
            .unwrap_err()
            .to_string()
            .contains("text_config.hidden_size")
    );
}

#[test]
fn vision_header_matches_all_211_names_and_shapes_without_payload() {
    let cfg = production_config();
    let header = independent_header(&cfg);
    let records = records_from_header(&header);
    assert_eq!(records.len(), 211);
    assert_eq!(
        records
            .iter()
            .map(|record| record.shape.iter().product::<usize>())
            .sum::<usize>(),
        167_757_824
    );
    assert!(validate_metadata(&records, &cfg).is_ok());
    let generated: BTreeMap<_, _> = inventory(&cfg)
        .unwrap()
        .into_iter()
        .map(|spec| (spec.name, spec.shape))
        .collect();
    let expected: BTreeMap<_, _> = records
        .into_iter()
        .map(|record| (record.name, record.shape))
        .collect();
    assert_eq!(generated, expected);
}

#[test]
fn vision_header_rejects_each_missing_tensor_by_name() {
    let cfg = production_config();
    let records = records_from_header(&independent_header(&cfg));
    for missing in 0..records.len() {
        let mut bad = records.clone();
        let removed = bad.remove(missing);
        match validate_metadata(&bad, &cfg) {
            Err(InferenceError::MissingTensor(name)) => assert_eq!(name, removed.name),
            other => panic!("missing {}: {other:?}", removed.name),
        }
    }
}

#[test]
fn vision_header_rejects_each_wrong_shape_by_name() {
    let cfg = production_config();
    let records = records_from_header(&independent_header(&cfg));
    for changed in 0..records.len() {
        let mut bad = records.clone();
        bad[changed].shape[0] += 1;
        match validate_metadata(&bad, &cfg) {
            Err(InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            }) => {
                assert_eq!(name, records[changed].name);
                assert_eq!(expected, records[changed].shape);
                assert_eq!(actual, bad[changed].shape);
            }
            other => panic!("shape {}: {other:?}", records[changed].name),
        }
    }
}

#[test]
fn vision_header_rejects_extras_duplicates_and_unknown_dtype() {
    let cfg = production_config();
    let records = records_from_header(&independent_header(&cfg));
    for name in ["vision_tower.extra.weight", "embed_vision.extra.weight"] {
        let mut bad = records.clone();
        bad.push(TensorMetadata {
            name: name.into(),
            shape: vec![1],
            dtype: Some("BF16".into()),
        });
        assert!(
            matches!(validate_metadata(&bad, &cfg), Err(InferenceError::InvalidSafetensors(message)) if message.contains(name))
        );
    }
    let mut duplicate = records.clone();
    duplicate.push(records[0].clone());
    assert!(
        matches!(validate_metadata(&duplicate, &cfg), Err(InferenceError::InvalidSafetensors(message)) if message.contains("duplicate") && message.contains(&records[0].name))
    );
    for dtype in [None, Some("F32".to_string())] {
        let mut bad = records.clone();
        bad[10].dtype = dtype;
        assert!(
            matches!(validate_metadata(&bad, &cfg), Err(InferenceError::InvalidSafetensors(message)) if message.contains(&records[10].name) && message.contains("BF16"))
        );
    }
    let mut unrelated = records;
    for name in [
        "audio_tower.extra.weight",
        "language_model.extra.weight",
        "embed_audio.extra.weight",
    ] {
        unrelated.push(TensorMetadata {
            name: name.into(),
            shape: vec![1],
            dtype: Some("F32".into()),
        });
    }
    assert!(validate_metadata(&unrelated, &cfg).is_ok());
}

struct NoPayload;

impl TensorSource for NoPayload {
    fn has_tensor(&mut self, _name: &str) -> Result<bool, InferenceError> {
        panic!("unexpected source access before metadata rejection")
    }
    fn tensor_shape(&mut self, _name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        panic!("unexpected source access before metadata rejection")
    }
    fn tensor_dtype(&mut self, _name: &str) -> Result<Option<String>, InferenceError> {
        panic!("unexpected source access before metadata rejection")
    }
    fn get_f32_tensor_owned(
        &mut self,
        _name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        panic!("unexpected payload access before complete metadata validation")
    }
}

#[test]
fn vision_metadata_validation_precedes_all_payload_reads() {
    let cfg = tiny_config();
    let mut records = records_from_header(&independent_header(&cfg));
    let last = records.last_mut().unwrap();
    let invalid_name = last.name.clone();
    last.shape[0] += 1;
    assert!(
        matches!(load_weights(&mut NoPayload, cfg, &records), Err(InferenceError::ShapeMismatch { name, .. }) if name == invalid_name)
    );
}

fn fixture_tensor_tag(name: &str) -> usize {
    match name {
        "vision_tower.patch_embedder.input_proj.weight" => 208,
        "vision_tower.patch_embedder.position_embedding_table" => 209,
        "embed_vision.embedding_projection.weight" => 210,
        _ => {
            let (layer, suffix) = name
                .strip_prefix("vision_tower.encoder.layers.")
                .unwrap()
                .split_once('.')
                .unwrap();
            let layer = layer.parse::<usize>().unwrap();
            assert!(layer < 16, "fixture layer for {name}");
            let role = match suffix {
                "input_layernorm.weight" => 0,
                "post_attention_layernorm.weight" => 1,
                "pre_feedforward_layernorm.weight" => 2,
                "post_feedforward_layernorm.weight" => 3,
                "self_attn.q_norm.weight" => 4,
                "self_attn.k_norm.weight" => 5,
                "self_attn.q_proj.linear.weight" => 6,
                "self_attn.k_proj.linear.weight" => 7,
                "self_attn.v_proj.linear.weight" => 8,
                "self_attn.o_proj.linear.weight" => 9,
                "mlp.gate_proj.linear.weight" => 10,
                "mlp.up_proj.linear.weight" => 11,
                "mlp.down_proj.linear.weight" => 12,
                _ => panic!("unknown fixture tensor {name}"),
            };
            layer * 13 + role
        }
    }
}

fn fixture_bf16_bits(name: &str, index: usize) -> u16 {
    let tag = fixture_tensor_tag(name);
    // Distinct names and indices must expose routing and ordering mistakes.
    let magnitude = 0x2000_u16 + ((31 * tag + 7 * index) % 7936) as u16;
    let sign = if (tag + index).is_multiple_of(2) {
        0
    } else {
        0x8000
    };
    magnitude | sign
}

fn small_checkpoint(cfg: &VisionConfig, poison: Option<&str>) -> Vec<u8> {
    let header = independent_header(cfg);
    let records = records_from_header(&header);
    let mut output = Vec::new();
    output.extend_from_slice(&(header.len() as u64).to_le_bytes());
    output.extend_from_slice(header.as_bytes());
    for record in records {
        for index in 0..record.shape.iter().product::<usize>() {
            let bits = if poison == Some(record.name.as_str()) && index == 0 {
                0x7fc0u16
            } else {
                fixture_bf16_bits(&record.name, index)
            };
            output.extend_from_slice(&bits.to_le_bytes());
        }
    }
    output
}

struct ExpectedLoadedField<'a> {
    name: String,
    shape: Vec<usize>,
    values: &'a [f32],
}

fn independently_named_fields<'a>(
    weights: &'a VisionWeights,
    cfg: &VisionConfig,
) -> Vec<ExpectedLoadedField<'a>> {
    let h = cfg.hidden_size;
    let d = cfg.head_dim;
    let q = cfg.num_attention_heads * d;
    let kv = cfg.num_key_value_heads * d;
    let f = cfg.intermediate_size;
    let mut fields = vec![
        ExpectedLoadedField {
            name: "vision_tower.patch_embedder.input_proj.weight".into(),
            shape: vec![h, 3 * cfg.patch_size * cfg.patch_size],
            values: &weights.input_proj,
        },
        ExpectedLoadedField {
            name: "vision_tower.patch_embedder.position_embedding_table".into(),
            shape: vec![2, cfg.position_embedding_size, h],
            values: &weights.position_embedding_table,
        },
        ExpectedLoadedField {
            name: "embed_vision.embedding_projection.weight".into(),
            shape: vec![cfg.text_hidden_size, h],
            values: &weights.embedding_projection,
        },
    ];
    for (index, layer) in weights.layers.iter().enumerate() {
        let prefix = format!("vision_tower.encoder.layers.{index}.");
        for (suffix, shape, values) in [
            (
                "input_layernorm.weight",
                vec![h],
                layer.input_layernorm.as_slice(),
            ),
            (
                "post_attention_layernorm.weight",
                vec![h],
                layer.post_attention_layernorm.as_slice(),
            ),
            (
                "pre_feedforward_layernorm.weight",
                vec![h],
                layer.pre_feedforward_layernorm.as_slice(),
            ),
            (
                "post_feedforward_layernorm.weight",
                vec![h],
                layer.post_feedforward_layernorm.as_slice(),
            ),
            ("self_attn.q_norm.weight", vec![d], layer.q_norm.as_slice()),
            ("self_attn.k_norm.weight", vec![d], layer.k_norm.as_slice()),
            (
                "self_attn.q_proj.linear.weight",
                vec![q, h],
                layer.q_proj.as_slice(),
            ),
            (
                "self_attn.k_proj.linear.weight",
                vec![kv, h],
                layer.k_proj.as_slice(),
            ),
            (
                "self_attn.v_proj.linear.weight",
                vec![kv, h],
                layer.v_proj.as_slice(),
            ),
            (
                "self_attn.o_proj.linear.weight",
                vec![h, q],
                layer.o_proj.as_slice(),
            ),
            (
                "mlp.gate_proj.linear.weight",
                vec![f, h],
                layer.gate_proj.as_slice(),
            ),
            (
                "mlp.up_proj.linear.weight",
                vec![f, h],
                layer.up_proj.as_slice(),
            ),
            (
                "mlp.down_proj.linear.weight",
                vec![h, f],
                layer.down_proj.as_slice(),
            ),
        ] {
            fields.push(ExpectedLoadedField {
                name: format!("{prefix}{suffix}"),
                shape,
                values,
            });
        }
    }
    fields
}

fn assert_named_weights(weights: &VisionWeights, cfg: &VisionConfig) {
    assert_eq!(weights.layers.len(), 16);
    assert_eq!(
        (
            weights.config.hidden_size,
            weights.config.intermediate_size,
            weights.config.num_hidden_layers,
            weights.config.num_attention_heads,
            weights.config.num_key_value_heads,
            weights.config.head_dim,
            weights.config.patch_size,
            weights.config.position_embedding_size,
            weights.config.text_hidden_size,
        ),
        (
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.patch_size,
            cfg.position_embedding_size,
            cfg.text_hidden_size,
        )
    );
    assert_eq!(
        weights.config.rms_norm_eps.to_bits(),
        cfg.rms_norm_eps.to_bits()
    );
    assert_eq!(
        weights.config.rope_theta.to_bits(),
        cfg.rope_theta.to_bits()
    );
    let fields = independently_named_fields(weights, cfg);
    assert_eq!(fields.len(), 211);
    let named_shapes: BTreeMap<_, _> = fields
        .iter()
        .map(|field| (field.name.clone(), field.shape.clone()))
        .collect();
    assert_eq!(named_shapes.len(), 211, "duplicate expected field name");
    let header_shapes: BTreeMap<_, _> = records_from_header(&independent_header(cfg))
        .into_iter()
        .map(|record| (record.name, record.shape))
        .collect();
    assert_eq!(named_shapes, header_shapes, "independent field shapes");
    for field in fields {
        assert_eq!(
            field.values.len(),
            field.shape.iter().product::<usize>(),
            "{}: loaded field length for shape {:?}",
            field.name,
            field.shape
        );
        for (index, value) in field.values.iter().enumerate() {
            let expected = u32::from(fixture_bf16_bits(&field.name, index)) << 16;
            assert_eq!(
                value.to_bits(),
                expected,
                "{}: exact BF16 widening at element {index}",
                field.name
            );
        }
    }
}

#[test]
fn vision_checkpoint_patterns_distinguish_all_names_and_indices() {
    let cfg = tiny_config();
    let records = records_from_header(&independent_header(&cfg));
    assert_eq!(records.len(), 211);
    let mut first_words = BTreeSet::new();
    for record in records {
        assert!(
            first_words.insert(fixture_bf16_bits(&record.name, 0)),
            "{}: tensor pattern aliases another name",
            record.name
        );
        let count = record.shape.iter().product::<usize>();
        assert!(count < 7936, "{}: index pattern period", record.name);
        let mut elements = BTreeSet::new();
        for index in 0..count {
            let bits = fixture_bf16_bits(&record.name, index);
            assert!(f32::from_bits(u32::from(bits) << 16).is_finite());
            assert!(
                elements.insert(bits),
                "{}: index {index} aliases an earlier element",
                record.name
            );
        }
    }
    assert_eq!(first_words.len(), 211);
}

#[test]
fn vision_loader_widens_all_211_bf16_tensors_exactly() {
    let cfg = tiny_config();
    let bytes = small_checkpoint(&cfg, None);
    assert!(bytes.len() < 100_000);
    let mut file = SafetensorsFile::from_bytes(bytes).unwrap();
    let records = metadata(&file).unwrap();
    assert_eq!(records.len(), 211);
    let weights = load_weights(&mut file, cfg.clone(), &records).unwrap();
    assert_named_weights(&weights, &cfg);
}

#[derive(Clone, Copy)]
enum Corruption {
    ReturnedShape,
    ReturnedLength,
    Nonfinite(f32),
}

struct CorruptSource {
    records: Vec<TensorMetadata>,
    corruption: Corruption,
}

impl TensorSource for CorruptSource {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
        Ok(self.records.iter().any(|record| record.name == name))
    }
    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        Ok(self
            .records
            .iter()
            .find(|record| record.name == name)
            .map(|record| record.shape.clone()))
    }
    fn tensor_dtype(&mut self, name: &str) -> Result<Option<String>, InferenceError> {
        Ok(self
            .records
            .iter()
            .find(|record| record.name == name)
            .and_then(|record| record.dtype.clone()))
    }
    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        let mut shape = self.tensor_shape(name)?.unwrap();
        let mut values = vec![1.5; shape.iter().product()];
        if name == "vision_tower.patch_embedder.input_proj.weight" {
            match self.corruption {
                Corruption::ReturnedShape => shape[0] += 1,
                Corruption::ReturnedLength => {
                    let _ = values.pop();
                }
                Corruption::Nonfinite(value) => values[0] = value,
            }
        }
        Ok((values, shape))
    }
}

#[test]
fn vision_loader_checks_returned_shape_length_and_finiteness() {
    let cfg = tiny_config();
    let records = records_from_header(&independent_header(&cfg));
    let target = "vision_tower.patch_embedder.input_proj.weight";
    let mut source = CorruptSource {
        records: records.clone(),
        corruption: Corruption::ReturnedShape,
    };
    assert!(
        matches!(load_weights(&mut source, cfg.clone(), &records), Err(InferenceError::ShapeMismatch { name, .. }) if name == target)
    );
    for corruption in [
        Corruption::ReturnedLength,
        Corruption::Nonfinite(f32::NAN),
        Corruption::Nonfinite(f32::INFINITY),
    ] {
        let mut source = CorruptSource {
            records: records.clone(),
            corruption,
        };
        assert!(
            matches!(load_weights(&mut source, cfg.clone(), &records), Err(InferenceError::InvalidSafetensors(message)) if message.contains(target))
        );
    }
}

#[test]
fn vision_inventory_reports_checked_shape_overflow() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = usize::MAX;
    assert!(
        matches!(inventory(&cfg), Err(InferenceError::InvalidSafetensors(message)) if message.contains("vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight") && message.contains("overflow"))
    );
}

#[test]
fn vision_loader_rejects_nonfinite_and_cache_never_publishes_partial_weights() {
    let cfg = tiny_config();
    let poison = "embed_vision.embedding_projection.weight";
    let cache = SuccessCache::default();
    let calls = AtomicUsize::new(0);
    let failed = cache.get_or_try_init(|| {
        calls.fetch_add(1, Ordering::SeqCst);
        let mut file = SafetensorsFile::from_bytes(small_checkpoint(&cfg, Some(poison)))?;
        let records = metadata(&file)?;
        load_weights(&mut file, cfg.clone(), &records)
    });
    assert!(
        matches!(failed, Err(InferenceError::InvalidSafetensors(message)) if message.contains(poison))
    );
    assert!(cache.value.lock().unwrap().is_none());
    let first = cache
        .get_or_try_init(|| {
            calls.fetch_add(1, Ordering::SeqCst);
            let mut file = SafetensorsFile::from_bytes(small_checkpoint(&cfg, None))?;
            let records = metadata(&file)?;
            load_weights(&mut file, cfg.clone(), &records)
        })
        .unwrap();
    let second = cache
        .get_or_try_init(|| -> Result<VisionWeights, InferenceError> {
            panic!("success was not cached")
        })
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_named_weights(&first, &cfg);
    assert_named_weights(&second, &cfg);
}

#[test]
fn vision_cache_concurrent_first_use_initializes_once() {
    let cache = Arc::new(SuccessCache::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let ready = Arc::new(Barrier::new(5));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let cache = Arc::clone(&cache);
            let calls = Arc::clone(&calls);
            let ready = Arc::clone(&ready);
            std::thread::spawn(move || {
                ready.wait();
                cache
                    .get_or_try_init(|| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        std::thread::yield_now();
                        Ok(vec![1u32, 2, 3])
                    })
                    .unwrap()
            })
        })
        .collect();
    ready.wait();
    let values: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(values.iter().all(|value| Arc::ptr_eq(&values[0], value)));
}

#[test]
fn vision_cache_poison_returns_a_typed_error() {
    let cache = Arc::new(SuccessCache::<u8>::default());
    let copy = Arc::clone(&cache);
    assert!(
        std::thread::spawn(move || {
            let _guard = copy.value.lock().unwrap();
            panic!("poison fixture");
        })
        .join()
        .is_err()
    );
    assert!(
        matches!(cache.get_or_try_init(|| Ok(1)), Err(InferenceError::Inference(message)) if message.contains("cache lock poisoned"))
    );
}

fn empty_checkpoint() -> Vec<u8> {
    let mut bytes = 2u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    bytes
}

#[cfg(unix)]
#[test]
fn vision_lazy_holder_retains_original_descriptor_and_config() {
    use std::os::unix::fs::MetadataExt;

    let directory = tempfile::tempdir().unwrap();
    let weights_path = directory.path().join("model.safetensors");
    let config_path = directory.path().join("config.json");
    std::fs::write(&weights_path, empty_checkpoint()).unwrap();
    std::fs::write(&config_path, CONFIG).unwrap();
    let config_json = std::fs::read_to_string(&config_path).unwrap();
    let file = crate::weights::mmap_trust::open_regular_file_no_hang(&weights_path).unwrap();
    let original = file.metadata().unwrap();
    let holder = LazyVision::from_checkpoint(config_json, file, weights_path.clone(), 512);
    std::fs::rename(&weights_path, directory.path().join("original.safetensors")).unwrap();
    std::fs::write(&weights_path, b"invalid replacement").unwrap();
    std::fs::write(&config_path, b"{}").unwrap();
    let replacement = std::fs::metadata(&weights_path).unwrap();
    assert_ne!(
        (original.dev(), original.ino()),
        (replacement.dev(), replacement.ino())
    );
    let held = holder
        .checkpoint
        .as_ref()
        .unwrap()
        .file
        .lock()
        .unwrap()
        .metadata()
        .unwrap();
    assert_eq!((held.dev(), held.ino()), (original.dev(), original.ino()));
    assert!(
        matches!(holder.get(), Err(InferenceError::MissingTensor(name)) if name == "vision_tower.encoder.layers.0.input_layernorm.weight")
    );
    assert!(!holder.is_loaded());
}

#[test]
fn vision_checkpoint_defers_parsing_and_retries_after_failure() {
    let directory = tempfile::tempdir().unwrap();
    let weights_path = directory.path().join("model.safetensors");
    std::fs::write(&weights_path, b"invalid").unwrap();
    let file = crate::weights::mmap_trust::open_regular_file_no_hang(&weights_path).unwrap();
    let holder = LazyVision::from_checkpoint(CONFIG.into(), file, weights_path.clone(), 512);
    assert!(!holder.is_loaded());
    assert!(matches!(
        holder.get(),
        Err(InferenceError::InvalidSafetensors(_))
    ));
    assert!(!holder.is_loaded());
    std::fs::write(&weights_path, empty_checkpoint()).unwrap();
    assert!(
        matches!(holder.get(), Err(InferenceError::MissingTensor(name)) if name == "vision_tower.encoder.layers.0.input_layernorm.weight")
    );
    assert!(!holder.is_loaded());
}

#[cfg(unix)]
#[test]
fn vision_checkpoint_rechecks_original_descriptor_trust_on_retry() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().unwrap();
    let weights_path = directory.path().join("model.safetensors");
    let original_path = directory.path().join("original.safetensors");
    std::fs::write(&weights_path, empty_checkpoint()).unwrap();
    std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let file = crate::weights::mmap_trust::open_regular_file_no_hang(&weights_path).unwrap();
    let holder = LazyVision::from_checkpoint(CONFIG.into(), file, weights_path.clone(), 512);
    std::fs::rename(&weights_path, &original_path).unwrap();
    std::fs::write(&weights_path, empty_checkpoint()).unwrap();
    std::fs::set_permissions(&weights_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::set_permissions(&original_path, std::fs::Permissions::from_mode(0o666)).unwrap();
    assert!(
        matches!(holder.get(), Err(InferenceError::InvalidSafetensors(message)) if message.contains("checkpoint file is outside") && message.contains("writable by group/other"))
    );
    assert!(!holder.is_loaded());
    std::fs::set_permissions(&original_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(
        matches!(holder.get(), Err(InferenceError::MissingTensor(name)) if name == "vision_tower.encoder.layers.0.input_layernorm.weight")
    );
    assert!(!holder.is_loaded());
}

#[test]
fn vision_checkpoint_lock_poison_returns_a_typed_error() {
    let directory = tempfile::tempdir().unwrap();
    let weights_path = directory.path().join("model.safetensors");
    std::fs::write(&weights_path, empty_checkpoint()).unwrap();
    let file = crate::weights::mmap_trust::open_regular_file_no_hang(&weights_path).unwrap();
    let holder = Arc::new(LazyVision::from_checkpoint(
        CONFIG.into(),
        file,
        weights_path,
        512,
    ));
    let copy = Arc::clone(&holder);
    assert!(
        std::thread::spawn(move || {
            let checkpoint = copy.checkpoint.as_ref().unwrap();
            let _guard = checkpoint.file.lock().unwrap();
            panic!("checkpoint lock poison fixture");
        })
        .join()
        .is_err()
    );
    assert!(
        matches!(holder.get(), Err(InferenceError::Inference(message)) if message.contains("checkpoint lock poisoned"))
    );
    assert!(!holder.is_loaded());
}

// ---------------------------------------------------------------------------
// Forward tests: RoPE lane pairing, pooling buckets, cancellation.
// ---------------------------------------------------------------------------

/// Applies the reference's `apply_multidimensional_rope` for two axes to one head,
/// written out independently of the implementation under test.
fn reference_axial_rotate(head: &[f32], x: f64, y: f64, theta: f64) -> Vec<f32> {
    let head_dim = head.len();
    let per_axis = head_dim / 4;
    let part = head_dim / 2;
    let inv: Vec<f64> = (0..per_axis)
        .map(|j| 1.0 / theta.powf(2.0 * j as f64 / (2 * per_axis) as f64))
        .collect();
    let mut out = head.to_vec();
    for axis in 0..2 {
        let coordinate = if axis == 0 { x } else { y };
        let offset = axis * part;
        for j in 0..per_axis {
            let angle = coordinate * inv[j];
            let (sine, cosine) = angle.sin_cos();
            let low = offset + j;
            let high = offset + per_axis + j;
            let u = out[low] as f64;
            let v = out[high] as f64;
            out[low] = (u * cosine - v * sine) as f32;
            out[high] = (v * cosine + u * sine) as f32;
        }
    }
    out
}

#[test]
fn axial_rope_pairs_lane_j_with_j_plus_half_within_each_axis() {
    use crate::model::gemma4_ops::{vision_axial_rope_apply, vision_axial_rope_inv_freq};
    let head_dim = 64usize;
    let heads = 1usize;
    let inv_freq = vision_axial_rope_inv_freq(head_dim, 100.0);

    // Lane 0 and lane 16 both live in the axis-0 half, so a y-only change must leave a
    // j<->j+32 pairing invariant while a j<->j+16 pairing moves. This is the discriminating
    // case the mutation control flips.
    let base: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.1) - 3.0).collect();
    let positions = [[3usize, 0usize]];
    let mut got = base.clone();
    vision_axial_rope_apply(&mut got, &positions, &inv_freq, heads, head_dim);
    let expected = reference_axial_rotate(&base, 3.0, 0.0, 100.0);
    for (index, (&a, &b)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "axial rotation lane {index}: {a} vs {b}"
        );
    }

    // A pure axis-0 rotation must leave the axis-1 half untouched.
    for index in 32..64 {
        assert!(
            (got[index] - base[index]).abs() < 1e-6,
            "axis-1 lane {index} moved on an x-only position"
        );
    }
    // And it must pair (j, j+16), not (j, j+32): lane 32 belongs to axis 1.
    assert!(
        (got[32] - base[32]).abs() < 1e-6,
        "lane 32 is axis 1 and must not pair with lane 0"
    );
    assert!(
        (got[16] - base[16]).abs() > 1e-4,
        "lane 16 must rotate with lane 0"
    );
}

#[test]
fn axial_rope_with_y_position_uses_the_second_axis_block() {
    use crate::model::gemma4_ops::{vision_axial_rope_apply, vision_axial_rope_inv_freq};
    let head_dim = 64usize;
    let inv_freq = vision_axial_rope_inv_freq(head_dim, 100.0);
    let base: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.07) + 0.5).collect();
    let positions = [[0usize, 5usize]];
    let mut got = base.clone();
    vision_axial_rope_apply(&mut got, &positions, &inv_freq, 1, head_dim);
    let expected = reference_axial_rotate(&base, 0.0, 5.0, 100.0);
    for (index, (&a, &b)) in got.iter().zip(&expected).enumerate() {
        assert!((a - b).abs() < 1e-5, "y rotation lane {index}: {a} vs {b}");
    }
    for index in 0..32 {
        assert!(
            (got[index] - base[index]).abs() < 1e-6,
            "axis-0 lane {index} moved on a y-only position"
        );
    }

    let mut distinct_inv_freq = inv_freq.clone();
    for frequency in &mut distinct_inv_freq[head_dim / 4..] {
        *frequency *= 2.0;
    }
    let mut distinct = base.clone();
    vision_axial_rope_apply(&mut distinct, &positions, &distinct_inv_freq, 1, head_dim);
    let expected_distinct = reference_axial_rotate(&base, 0.0, 10.0, 100.0);
    for (index, (&a, &b)) in distinct.iter().zip(&expected_distinct).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "distinct y-frequency lane {index}: {a} vs {b}"
        );
    }
}

#[test]
fn pooling_buckets_and_occupancy_on_a_six_by_nine_grid() {
    let weights = tiny_test_weights();
    let hidden = weights.config.hidden_size;
    let grid_w = 9usize;
    let grid_h = 6usize;
    let mut positions = Vec::new();
    for y in 0..grid_h {
        for x in 0..grid_w {
            positions.push([x, y]);
        }
    }
    let tokens = positions.len();
    let encoder: Vec<f32> = (0..tokens * hidden).map(|i| i as f32).collect();
    let (values, occupancy) = weights.pool(&encoder, &positions).unwrap();
    assert_eq!(values.len(), (grid_w / 3) * (grid_h / 3) * hidden);
    assert_eq!(occupancy, vec![true; (grid_w / 3) * (grid_h / 3)]);

    let scale = (hidden as f32).sqrt();
    for y in 0..grid_h {
        for x in 0..grid_w {
            let bucket = (x / 3) + (grid_w / 3) * (y / 3);
            for channel in 0..hidden {
                let sum: f32 = positions
                    .iter()
                    .enumerate()
                    .filter(|&(_, &[px, py])| px / 3 == x / 3 && py / 3 == y / 3)
                    .map(|(row, _)| encoder[row * hidden + channel])
                    .sum();
                let expected = sum / 9.0 * scale;
                let got = values[bucket * hidden + channel];
                assert!(
                    (got - expected).abs() < 1e-3,
                    "bucket {bucket} channel {channel}: {got} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn pooling_rejects_a_grid_that_is_not_a_multiple_of_three() {
    let weights = tiny_test_weights();
    let hidden = weights.config.hidden_size;
    // A 3-wide, 2-tall grid: the width is a multiple of 3 but the height is not, so the
    // height guard is what must fire (a width-only guard would accept this and then read
    // a bucket count the grid cannot fill).
    let mut positions = Vec::new();
    for y in 0..2 {
        for x in 0..3 {
            positions.push([x, y]);
        }
    }
    let encoder = vec![1.0f32; positions.len() * hidden];
    assert!(
        matches!(weights.pool(&encoder, &positions), Err(InferenceError::Inference(message)) if message.contains("multiples of 3"))
    );
}

#[test]
fn tower_forward_cancels_before_and_between_layers() {
    let weights = tiny_test_weights();
    let hidden = weights.config.hidden_size;
    let tokens = 9usize;
    let positions: Vec<[usize; 2]> = (0..tokens).map(|i| [i % 3, i / 3]).collect();
    let patch = vec![0.25f32; tokens * hidden];
    assert!(matches!(
        weights.tower_forward(&patch, &positions, &mut || true),
        Err(InferenceError::Inference(message)) if message.contains("cancelled")
    ));
    let mut calls = 0usize;
    let result = weights.tower_forward(&patch, &positions, &mut || {
        calls += 1;
        calls > 1
    });
    assert!(
        matches!(result, Err(InferenceError::Inference(message)) if message.contains("cancelled"))
    );
    assert!(calls > 1, "cancellation is polled between layers");
}

#[test]
fn bridge_matches_an_independent_norm_then_projection() {
    let weights = tiny_test_weights();
    let hidden = weights.config.hidden_size;
    let text_hidden = weights.config.text_hidden_size;
    let rows = 3usize;
    let values: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32) * 0.05 - 0.4)
        .collect();
    let projected = weights.bridge(&values, rows).unwrap();
    assert_eq!(projected.len(), rows * text_hidden);

    let eps = weights.config.rms_norm_eps;
    for row in 0..rows {
        let source = &values[row * hidden..(row + 1) * hidden];
        let mean_square = source.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean_square + eps).sqrt();
        let normed: Vec<f32> = source.iter().map(|v| v * inv).collect();
        for out in 0..text_hidden {
            let weight_row = &weights.embedding_projection[out * hidden..(out + 1) * hidden];
            let expected: f32 = weight_row.iter().zip(&normed).map(|(w, n)| w * n).sum();
            let got = projected[row * text_hidden + out];
            assert!(
                (got - expected).abs() < 1e-4,
                "bridge row {row} out {out}: {got} vs {expected}"
            );
        }
    }
}

#[test]
fn bridge_applies_scale_free_norm_before_projection() {
    let mut weights = tiny_test_weights();
    let hidden = weights.config.hidden_size;
    let text_hidden = weights.config.text_hidden_size;
    assert_eq!(hidden, text_hidden);
    weights.embedding_projection.fill(0.0);
    for channel in 0..hidden {
        weights.embedding_projection[channel * hidden + channel] = 1.0;
    }
    let values = vec![2.0f32; hidden];
    let projected = weights.bridge(&values, 1).unwrap();
    let expected = 2.0 / (4.0 + weights.config.rms_norm_eps).sqrt();
    assert_eq!(projected.len(), hidden);
    for (channel, &value) in projected.iter().enumerate() {
        assert!(
            (value - expected).abs() < 1e-6,
            "bridge normalization channel {channel}: {value} vs {expected}"
        );
    }
}

#[test]
fn patch_embed_centers_pixels_and_adds_learned_positions() {
    let weights = tiny_test_weights();
    let cfg = &weights.config;
    let hidden = cfg.hidden_size;
    let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
    let size = cfg.position_embedding_size;
    let positions: Vec<[usize; 2]> = (0..9).map(|row| [row % 3, row / 3]).collect();
    let mut patch_values = vec![0f32; positions.len() * patch_dim];
    for (index, value) in patch_values.iter_mut().enumerate() {
        *value = (index as f32 % 17.0) / 16.0;
    }
    let prepared = PreparedImage::new(
        patch_values.clone(),
        positions.clone(),
        3,
        3,
        9,
        1,
        2520,
        ProcessorIdentity::already_resized_rgb8(),
    )
    .unwrap();
    let got = weights.patch_embed(&prepared).unwrap();
    assert_eq!(got.len(), positions.len() * hidden);

    let table = &weights.position_embedding_table;
    for (row, &[x, y]) in positions.iter().enumerate() {
        for out in 0..hidden {
            let projected: f32 = (0..patch_dim)
                .map(|k| {
                    2.0 * (patch_values[row * patch_dim + k] - 0.5)
                        * weights.input_proj[out * patch_dim + k]
                })
                .sum();
            let expected =
                projected + table[x * hidden + out] + table[size * hidden + y * hidden + out];
            let value = got[row * hidden + out];
            assert!(
                (value - expected).abs() < 1e-4,
                "row {row} out {out}: {value} vs {expected}"
            );
        }
    }

    // Centering maps 0.5 to exactly zero, so a flat mid-gray patch has no projection term.
    assert_eq!(2.0 * (0.5 - 0.5), 0.0);
}
