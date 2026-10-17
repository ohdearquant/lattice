use super::*;

const CFG: &str = r#"{"hidden_size": 1024, "vision_config": {"hidden_size": 1152,
        "intermediate_size": 4304, "num_hidden_layers": 27, "num_attention_heads": 16,
        "patch_size": 14, "image_size": 384, "layer_norm_eps": 1e-06, "spatial_merge_size": 2}}"#;

fn valid_cfg() -> PaddleOcrVisionConfig {
    PaddleOcrVisionConfig::from_config_json_str(CFG).unwrap()
}

#[test]
fn config_parses_pinned_shape() {
    let cfg = PaddleOcrVisionConfig::from_config_json_str(CFG).unwrap();
    assert_eq!(cfg.head_dim(), 72);
    assert_eq!(cfg.pos_table_side(), 27);
    assert_eq!(cfg.patch_len(), 588);
    assert_eq!(cfg.merged_width(), 4608);
    assert_eq!(cfg.text_hidden_size, 1024);
}

#[test]
fn config_rejects_unsupported_merge_and_heads() {
    let bad_merge = CFG.replace("\"spatial_merge_size\": 2", "\"spatial_merge_size\": 3");
    assert!(PaddleOcrVisionConfig::from_config_json_str(&bad_merge).is_err());
    let bad_heads = CFG.replace("\"num_attention_heads\": 16", "\"num_attention_heads\": 7");
    assert!(PaddleOcrVisionConfig::from_config_json_str(&bad_heads).is_err());
}

#[test]
fn config_rejects_layer_count_above_cap() {
    let mut cfg = valid_cfg();
    for layers in [0, MAX_VISION_LAYERS + 1, usize::MAX] {
        cfg.num_hidden_layers = layers;
        let error = cfg.validate().unwrap_err().to_string();
        assert!(error.contains("MAX_VISION_LAYERS"), "{error}");
    }

    cfg.num_hidden_layers = MAX_VISION_LAYERS;
    cfg.validate().unwrap();
    cfg.num_hidden_layers = 27;
    cfg.validate().unwrap();
}

#[test]
fn config_enforces_dimension_caps_and_lower_bounds() {
    let mut cfg = valid_cfg();

    cfg.hidden_size = 0;
    assert!(cfg.validate().is_err());
    cfg = valid_cfg();
    cfg.hidden_size = MAX_VISION_HIDDEN_SIZE;
    cfg.num_attention_heads = MAX_VISION_HEADS;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 1;
    cfg.validate().unwrap();
    cfg.hidden_size = MAX_VISION_HIDDEN_SIZE + 1;
    assert!(cfg.validate().is_err());

    cfg = valid_cfg();
    cfg.intermediate_size = 1;
    cfg.validate().unwrap();
    cfg.intermediate_size = MAX_VISION_INTERMEDIATE_SIZE;
    cfg.validate().unwrap();
    cfg.intermediate_size = MAX_VISION_INTERMEDIATE_SIZE + 1;
    assert!(cfg.validate().is_err());

    cfg = valid_cfg();
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 1;
    cfg.validate().unwrap();
    cfg.num_attention_heads = MAX_VISION_HEADS;
    cfg.hidden_size = MAX_VISION_HIDDEN_SIZE;
    cfg.num_hidden_layers = 1;
    cfg.intermediate_size = 1;
    cfg.validate().unwrap();
    cfg.num_attention_heads = MAX_VISION_HEADS + 1;
    assert!(cfg.validate().is_err());

    cfg = valid_cfg();
    cfg.image_size = cfg.patch_size;
    cfg.validate().unwrap();
    cfg.image_size = MAX_VISION_POSITION_SIDE * cfg.patch_size;
    cfg.validate().unwrap();
    cfg.image_size = (MAX_VISION_POSITION_SIDE + 1) * cfg.patch_size;
    assert!(cfg.validate().is_err());

    cfg = valid_cfg();
    cfg.text_hidden_size = 1;
    cfg.validate().unwrap();
    cfg.text_hidden_size = MAX_VISION_TEXT_HIDDEN_SIZE;
    cfg.validate().unwrap();
    cfg.text_hidden_size = MAX_VISION_TEXT_HIDDEN_SIZE + 1;
    assert!(cfg.validate().is_err());

    cfg = valid_cfg();
    cfg.num_channels = 3;
    cfg.validate().unwrap();
    cfg.num_channels = 4;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_checkpoint_over_aggregate_budget() {
    let mut cfg = valid_cfg();
    cfg.num_hidden_layers = MAX_VISION_LAYERS;
    cfg.hidden_size = MAX_VISION_HIDDEN_SIZE;
    cfg.intermediate_size = MAX_VISION_INTERMEDIATE_SIZE;
    cfg.num_attention_heads = MAX_VISION_HEADS;
    let error = cfg.validate().unwrap_err().to_string();
    assert!(error.contains("MAX_VISION_CHECKPOINT_BYTES"), "{error}");

    valid_cfg().validate().unwrap();
}

#[test]
fn validate_bounds_patch_size() {
    let mut cfg = valid_cfg();
    cfg.patch_size = MAX_VISION_PATCH_SIZE + 1;
    cfg.image_size = cfg.patch_size;
    let error = cfg.validate().unwrap_err().to_string();
    assert!(error.contains("MAX_VISION_PATCH_SIZE"), "{error}");

    cfg.patch_size = MAX_VISION_PATCH_SIZE;
    cfg.validate().unwrap();
}

struct PanickingTensorSource;

impl TensorSource for PanickingTensorSource {
    fn has_tensor(&mut self, _name: &str) -> Result<bool, InferenceError> {
        panic!("invalid configuration reached tensor source");
    }

    fn tensor_shape(&mut self, _name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        panic!("invalid configuration reached tensor source");
    }

    fn get_f32_tensor_owned(
        &mut self,
        _name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        panic!("invalid configuration reached tensor source");
    }
}

struct CountingTensorSource {
    declared_shape: Option<Vec<usize>>,
    materializations: usize,
}

impl TensorSource for CountingTensorSource {
    fn has_tensor(&mut self, _name: &str) -> Result<bool, InferenceError> {
        Ok(self.declared_shape.is_some())
    }

    fn tensor_shape(&mut self, _name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        Ok(self.declared_shape.clone())
    }

    fn get_f32_tensor_owned(
        &mut self,
        _name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        self.materializations += 1;
        Ok((Vec::new(), Vec::new()))
    }
}

#[test]
fn load_tensor_preflights_wrong_shape_without_materialization() {
    let mut source = CountingTensorSource {
        declared_shape: Some(vec![4]),
        materializations: 0,
    };
    let error = load_tensor(&mut source, "tensor", &[2, 2]).unwrap_err();
    assert!(matches!(error, InferenceError::ShapeMismatch { .. }));
    assert_eq!(source.materializations, 0);
}

#[test]
fn load_tensor_preflights_missing_tensor_without_materialization() {
    let mut source = CountingTensorSource {
        declared_shape: None,
        materializations: 0,
    };
    let error = load_tensor(&mut source, "tensor", &[2, 2]).unwrap_err();
    assert!(matches!(error, InferenceError::MissingTensor(name) if name == "tensor"));
    assert_eq!(source.materializations, 0);
}

#[test]
fn load_tensor_rejects_oversized_declared_tensor_before_materialization() {
    let oversized = crate::model::qwen35_config::MAX_VISION_TENSOR_BYTES as usize / 4 + 1;
    let mut source = CountingTensorSource {
        declared_shape: Some(vec![oversized]),
        materializations: 0,
    };
    let error = load_tensor(&mut source, "tensor", &[oversized])
        .unwrap_err()
        .to_string();
    assert!(error.contains("MAX_VISION_TENSOR_BYTES"), "{error}");
    assert_eq!(source.materializations, 0);
}

#[test]
fn loader_validates_before_reading_tensors() {
    let mut cfg = valid_cfg();
    cfg.num_hidden_layers = MAX_VISION_LAYERS + 1;
    let mut source = PanickingTensorSource;
    let error = PaddleOcrVisionWeights::load(&mut source, &cfg)
        .unwrap_err()
        .to_string();
    assert!(error.contains("MAX_VISION_LAYERS"), "{error}");
}

#[test]
fn check_grid_rejects_quadratic_attention_blowup() {
    let cfg = valid_cfg();
    let error = check_grid(&cfg, 256, 256).unwrap_err().to_string();
    assert!(error.contains("MAX_VISION_TOKENS"));
}

#[test]
fn check_grid_accepts_largest_legal_grid() {
    let cfg = valid_cfg();
    assert_eq!(
        check_grid(&cfg, 2, MAX_VISION_TOKENS / 2).unwrap(),
        MAX_VISION_TOKENS
    );
}

#[test]
fn check_grid_caps_tokens_at_shipped_budget() {
    let cfg = valid_cfg();
    assert_eq!(check_grid(&cfg, 64, 80).unwrap(), 5120);
    let error = check_grid(&cfg, 74, 70).unwrap_err().to_string();
    assert!(error.contains("MAX_VISION_TOKENS"), "{error}");
}

#[test]
fn check_grid_rejects_overflow_and_unmerged_shapes() {
    let cfg = valid_cfg();
    assert!(check_grid(&cfg, usize::MAX, 2).is_err());
    assert!(check_grid(&cfg, 1, 2).is_err());
}

#[test]
fn half_pixel_bilinear_identity_when_grid_matches_table() {
    // Sampling a 3x3 table onto a 3x3 grid must return the table rows exactly.
    let side = 3;
    let hidden = 2;
    let table: Vec<f32> = (0..side * side * hidden).map(|v| v as f32).collect();
    for h in 0..side {
        for w in 0..side {
            let mut out = vec![0.0f32; hidden];
            bilinear_pos_embed_half_pixel(&table, hidden, side, side, side, h, w, &mut out);
            let row = &table[(h * side + w) * hidden..(h * side + w + 1) * hidden];
            assert_eq!(out, row);
        }
    }
}

#[test]
fn half_pixel_bilinear_upsample_matches_torch_convention() {
    // 2-wide table [0, 1] upsampled to 4: torch align_corners=False gives
    // [0, 0.25, 0.75, 1].
    let table = [0.0f32, 0.0, 1.0, 1.0]; // side 2, hidden 1, rows: (0,0)=0 (0,1)=0 (1,0)=1 (1,1)=1
    let expect = [0.0f32, 0.25, 0.75, 1.0];
    for (h, e) in expect.iter().enumerate() {
        let mut out = [0.0f32];
        bilinear_pos_embed_half_pixel(&table, 1, 2, 4, 1, h, 0, &mut out);
        assert!((out[0] - e).abs() < 1e-6, "row {h}: {} vs {e}", out[0]);
    }
}
