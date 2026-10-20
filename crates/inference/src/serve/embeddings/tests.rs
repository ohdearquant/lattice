use super::*;
use std::path::Path;
use test_support::{tiny_embedding_model, tiny_png_data_uri};

mod characterization;

/// Tensor inventory a tiny f16-decoder + vision checkpoint must carry
/// for [`open_qwen35_single_decoder_safetensors`],
/// [`load_qwen35_vision_weights_from_safetensors`], and
/// [`load_f16_weights`] to all succeed. Mirrors
/// `lattice-embed::vision::tests::tiny_vlm_checkpoint_shapes` exactly:
/// both crates load through the same checkpoint-reading functions, so
/// the same on-disk shape works for either loader.
fn tiny_vlm_checkpoint_shapes() -> Vec<(String, Vec<usize>)> {
    let hidden = 8usize;
    let mut shapes = vec![
        (
            "model.language_model.embed_tokens.weight".to_string(),
            vec![16, hidden],
        ),
        ("model.language_model.norm.weight".to_string(), vec![hidden]),
        (
            "model.language_model.layers.0.input_layernorm.weight".to_string(),
            vec![hidden],
        ),
        (
            "model.language_model.layers.0.post_attention_layernorm.weight".to_string(),
            vec![hidden],
        ),
        (
            "model.language_model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![4, hidden],
        ),
        (
            "model.language_model.layers.0.mlp.up_proj.weight".to_string(),
            vec![4, hidden],
        ),
        (
            "model.language_model.layers.0.mlp.down_proj.weight".to_string(),
            vec![hidden, 4],
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![16, hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.k_proj.weight".to_string(),
            vec![hidden, hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.v_proj.weight".to_string(),
            vec![hidden, hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.o_proj.weight".to_string(),
            vec![hidden, hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.q_norm.weight".to_string(),
            vec![hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.k_norm.weight".to_string(),
            vec![hidden],
        ),
        (
            "model.visual.patch_embed.proj.weight".to_string(),
            vec![hidden, 3, 1, 2, 2],
        ),
        (
            "model.visual.patch_embed.proj.bias".to_string(),
            vec![hidden],
        ),
        (
            "model.visual.pos_embed.weight".to_string(),
            vec![16, hidden],
        ),
        (
            "model.visual.merger.linear_fc1.weight".to_string(),
            vec![32, 32],
        ),
        ("model.visual.merger.linear_fc1.bias".to_string(), vec![32]),
        (
            "model.visual.merger.linear_fc2.weight".to_string(),
            vec![hidden, 32],
        ),
        (
            "model.visual.merger.linear_fc2.bias".to_string(),
            vec![hidden],
        ),
        ("model.visual.merger.norm.weight".to_string(), vec![hidden]),
        ("model.visual.merger.norm.bias".to_string(), vec![hidden]),
    ];
    for (suffix, shape) in [
        ("attn.qkv.weight", vec![24, hidden]),
        ("attn.qkv.bias", vec![24]),
        ("attn.proj.weight", vec![hidden, hidden]),
        ("attn.proj.bias", vec![hidden]),
        ("mlp.linear_fc1.weight", vec![32, hidden]),
        ("mlp.linear_fc1.bias", vec![32]),
        ("mlp.linear_fc2.weight", vec![hidden, 32]),
        ("mlp.linear_fc2.bias", vec![hidden]),
        ("norm1.weight", vec![hidden]),
        ("norm1.bias", vec![hidden]),
        ("norm2.weight", vec![hidden]),
        ("norm2.bias", vec![hidden]),
    ] {
        shapes.push((format!("model.visual.blocks.0.{suffix}"), shape));
    }
    shapes
}

fn write_f32_safetensors(path: &Path, shapes: &[(String, Vec<usize>)]) {
    let mut header_parts = Vec::with_capacity(shapes.len());
    let mut data = Vec::new();
    for (i, (name, shape)) in shapes.iter().enumerate() {
        let start = data.len();
        let numel: usize = shape.iter().product();
        for _ in 0..numel {
            data.extend_from_slice(&((i + 1) as f32 / 100.0).to_le_bytes());
        }
        let end = data.len();
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header_parts.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[{start},{end}]}}"#
        ));
    }
    let header = format!("{{{}}}", header_parts.join(","));
    let mut bytes = Vec::with_capacity(8 + header.len() + data.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(path, bytes).expect("write safetensors fixture");
}

fn write_tiny_tokenizer_json(dir: &Path) {
    let tokenizer = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "a": 0, "b": 1, "c": 2, "d": 3,
                    "e": 4, "f": 5, "g": 6, "h": 7,
                    "i": 8, "j": 9, "k": 10, "l": 11,
                    "m": 12, "n": 13, "o": 14, "p": 15
                },
                "merges": []
            }
        }"#;
    std::fs::write(dir.join("tokenizer.json"), tokenizer).expect("write tokenizer.json");
}

/// Writes a tiny on-disk VLM checkpoint (`config.json`,
/// `tokenizer.json`, single-file `model.safetensors`) with a
/// caller-chosen `max_position_embeddings`, so a test can pin it above
/// `BpeTokenizer`'s 4096 constructor default and exercise
/// [`EmbeddingModel::from_directory`]'s tokenizer-capping behavior end
/// to end.
fn write_tiny_vlm_checkpoint(dir: &Path, max_position_embeddings: usize) {
    let config = format!(
        r#"{{
            "text_config": {{
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "vocab_size": 16,
                "intermediate_size": 4,
                "rms_norm_eps": 0.000001,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 1.0,
                "rope_parameters": {{
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [2, 1, 1],
                    "mrope_interleaved": true
                }},
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "tie_word_embeddings": true,
                "full_attention_interval": 1,
                "layer_types": ["full_attention"],
                "layer_mask": [true],
                "eos_token_id": 15,
                "max_position_embeddings": {max_position_embeddings}
            }},
            "vision_config": {{
                "depth": 1,
                "hidden_size": 8,
                "num_heads": 2,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 8,
                "temporal_patch_size": 1,
                "num_position_embeddings": 16,
                "in_channels": 3,
                "deepstack_visual_indexes": []
            }},
            "image_token_id": 9,
            "vision_start_token_id": 10,
            "vision_end_token_id": 11,
            "tie_word_embeddings": true
        }}"#
    );
    std::fs::write(dir.join("config.json"), &config).expect("write config.json");
    write_tiny_tokenizer_json(dir);
    write_f32_safetensors(
        &dir.join("model.safetensors"),
        &tiny_vlm_checkpoint_shapes(),
    );
}

/// Issue #1408, loader half: a `tokenizer.json` with no explicit cap
/// parses through `BpeTokenizer::from_tokenizer_json` at its
/// constructor default (4096). A checkpoint whose
/// `max_position_embeddings` sits above that default (here 5000, still
/// under the 8192 ceiling `capped_tokenizer` applies) must come out of
/// `from_directory` with its tokenizer cap raised to match. This is a
/// different property from
/// `embedding_model_new_raises_a_tokenizer_cap_below_max_context` below:
/// that test calls `EmbeddingModel::new` directly and would still pass
/// if `from_directory` stopped routing through `new` (e.g. reverted to
/// building `Self { .. }` directly); this test exercises
/// `from_directory`'s own path end to end and reddens on that revert.
#[test]
fn from_directory_raises_tokenizer_cap_to_context_window_above_tokenizer_default() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_tiny_vlm_checkpoint(tmp.path(), 5000);

    let model = EmbeddingModel::from_directory(tmp.path())
        .expect("VLM checkpoint with a wide context window must load");
    assert_eq!(
        model.tokenizer.max_seq_len(),
        5000,
        "tokenizer cap must be raised to max_position_embeddings, not left at the \
             tokenizer's own 4096 default"
    );
}

/// Issue #1408, constructor half: `EmbeddingModel::new` itself -- not
/// just the `from_directory` loader -- must raise a caller-supplied
/// tokenizer's cap to at least `max_context()`. Without this, a caller
/// passing a default `BpeTokenizer` (cap 4096) alongside a config whose
/// `max_position_embeddings.min(8192)` exceeds 4096 reproduces the
/// #1408 bug through this constructor instead of the loader.
#[test]
fn embedding_model_new_raises_a_tokenizer_cap_below_max_context() {
    let base = tiny_embedding_model();
    let mut config = base.config.clone();
    config.max_position_embeddings = 5000;
    // `BpeTokenizer::from_vocab_and_merges` defaults to
    // `DEFAULT_BPE_MAX_SEQ_LEN` (4096), strictly below the 5000-token
    // window just configured -- exactly the mismatch `new` must close.
    let mut vocab_map = std::collections::HashMap::new();
    for (i, c) in ["a", "b", "c"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    let tokenizer =
        BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs");
    assert!(tokenizer.max_seq_len() < 5000, "fixture precondition");

    let model = EmbeddingModel::new(
        base.weights.clone(),
        config,
        base.vision_weights.clone(),
        tokenizer,
    );
    assert_eq!(model.max_context(), 5000);
    assert_eq!(
        model.tokenizer.max_seq_len(),
        5000,
        "new() must raise a below-window tokenizer cap to max_context()"
    );
}

#[test]
fn embed_items_happy_path_text_only() {
    let model = tiny_embedding_model();
    let items = normalize_embedding_items(vec![EmbeddingInputItem::Text("a".to_string())]).unwrap();
    let (data, usage) = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0].index, 0);
    assert_eq!(data[0].embedding.len(), model.dimensions());
    assert!(usage.prompt_tokens > 0);
}

#[test]
fn embed_items_rejects_text_item_over_context_window() {
    let model = tiny_embedding_model();
    assert_eq!(model.max_context(), 512);
    // Single-character vocab, no merges: each "a" is its own token, so
    // this tokenizes to 600 tokens, above the tiny fixture's 512-token
    // context window.
    let over_limit_text = "a".repeat(600);
    let items = normalize_embedding_items(vec![EmbeddingInputItem::Text(over_limit_text)]).unwrap();
    let err = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "context_length_exceeded");
            assert!(message.contains("input item 0"), "message: {message}");
            assert!(message.contains("600"), "message: {message}");
            assert!(message.contains("512"), "message: {message}");
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn embed_items_accepts_text_item_within_context_window() {
    let model = tiny_embedding_model();
    let items = normalize_embedding_items(vec![EmbeddingInputItem::Text("a".repeat(10))]).unwrap();
    let (data, usage) = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(usage.prompt_tokens, 10);
}

/// Builds a fixture with the OPPOSITE ordering of
/// `embed_items_rejects_text_item_over_context_window` above: that
/// test's `max_context() == 512` sits BELOW the tokenizer's default
/// (4096) truncation cap, so a long input survives tokenization intact
/// and trips the guard whether the guard reads the real or the
/// pre-truncation count -- it cannot distinguish the fix from a no-op.
/// This fixture pins a tokenizer cap (5) BELOW the context window
/// (1000), which is the ordering issue #1408 actually needs: a
/// comparison against `tokenize_len`'s real (post-truncation) count
/// would clamp to 5 tokens for any input and could never exceed the
/// 1000-token window, making the rejection unreachable.
///
/// Built via a direct struct literal, not `EmbeddingModel::new`:
/// `new` now enforces the #1408 constructor invariant (tokenizer cap at
/// or above `max_context()`, see `capped_tokenizer`) and would silently
/// raise this fixture's deliberately-below-window cap back to 1000,
/// destroying the exact mismatch this fixture exists to construct.
/// This is a different property from the constructor invariant test
/// (`embedding_model_new_raises_a_tokenizer_cap_below_max_context`
/// below) -- that test asserts `new` closes this gap for real callers;
/// this fixture must keep it open to exercise the admission-count
/// semantics on their own. Same-crate private-field access from this
/// child `mod tests` is what makes the bypass possible.
fn context_window_above_tokenizer_cap_model() -> EmbeddingModel {
    let base = tiny_embedding_model();
    let mut config = base.config.clone();
    config.max_position_embeddings = 1000;
    let tokenizer = base.tokenizer.clone().with_max_seq_len(5);
    EmbeddingModel {
        weights: base.weights.clone(),
        config,
        vision_weights: base.vision_weights.clone(),
        tokenizer,
    }
}

#[test]
fn embed_items_rejects_text_item_whose_pre_truncation_length_exceeds_a_window_above_the_tokenizer_cap()
 {
    let model = context_window_above_tokenizer_cap_model();
    assert_eq!(model.max_context(), 1000);
    assert_eq!(model.tokenizer.max_seq_len(), 5);

    // Pre-truncation length 1200 (above the 1000-token window), which
    // the tokenizer would truncate to 5 real tokens before the forward
    // pass. Under the pre-#1408 comparison (real length 5 vs. window
    // 1000) this would be silently ACCEPTED and embedded from a
    // 5-token truncated prefix. The admission guard must reject it.
    let over_limit_text = "a".repeat(1200);
    let items = normalize_embedding_items(vec![EmbeddingInputItem::Text(over_limit_text)]).unwrap();
    let err = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "context_length_exceeded");
            assert!(message.contains("input item 0"), "message: {message}");
            assert!(message.contains("1200"), "message: {message}");
            assert!(message.contains("1000"), "message: {message}");
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn embed_items_accepted_prompt_tokens_matches_the_count_actually_embedded_when_truncated() {
    // Same inverted fixture, but with an input whose pre-truncation
    // length (12) still fits the window (1000) -- ACCEPTED -- while the
    // tokenizer truncates it to 5 tokens before the forward pass.
    // `usage.prompt_tokens` must report 5 (what was actually embedded),
    // not 12: the naive one-liner fix (swap `tokenize_len` to always
    // return the pre-truncation count) would report 12 here, billing
    // tokens that were never embedded -- see `usage.prompt_tokens`'s own
    // doc comment on `EmbeddingsUsage`.
    let model = context_window_above_tokenizer_cap_model();
    let text = "a".repeat(12);
    let items = normalize_embedding_items(vec![EmbeddingInputItem::Text(text)]).unwrap();
    let (data, usage) = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(
        usage.prompt_tokens, 5,
        "usage must report the real embedded count, not the pre-truncation count"
    );
}

#[test]
fn embed_items_happy_path_image() {
    let model = tiny_embedding_model();
    let items = normalize_embedding_items(vec![EmbeddingInputItem::Image {
        url: tiny_png_data_uri(0),
    }])
    .unwrap();
    let (data, _usage) = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap();
    assert_eq!(data.len(), 1);
    let norm: f32 = data[0].embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
}

#[test]
fn embed_items_image_usage_matches_scaffold_token_formula() {
    let model = tiny_embedding_model();
    let data_uri = tiny_png_data_uri(2);
    let image_bytes = decode_inline_image(&data_uri).unwrap();
    let vision_cfg = model.config.vision_config.as_ref().unwrap();
    let (_pixels, grid) = preprocess_qwen35_image(&image_bytes, vision_cfg).unwrap();
    let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
    let expected_tokens = 2 + grid.num_patches() / merge_sq;

    let items =
        normalize_embedding_items(vec![EmbeddingInputItem::Image { url: data_uri }]).unwrap();
    let (_data, usage) = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap();
    assert_eq!(usage.prompt_tokens, expected_tokens);
}

#[test]
fn embed_items_rejects_image_item_over_context_window() {
    let base = tiny_embedding_model();
    let mut config = base.config.clone();
    config.max_position_embeddings = 4;
    let model = EmbeddingModel::new(
        base.weights.clone(),
        config,
        base.vision_weights.clone(),
        base.tokenizer.clone(),
    );
    assert_eq!(model.max_context(), 4);

    let data_uri = tiny_png_data_uri(3);
    let image_bytes = decode_inline_image(&data_uri).unwrap();
    let vision_cfg = model.config.vision_config.as_ref().unwrap();
    let (_pixels, grid) = preprocess_qwen35_image(&image_bytes, vision_cfg).unwrap();
    let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
    let expected_tokens = 2 + grid.num_patches() / merge_sq;
    assert!(
        expected_tokens > model.max_context(),
        "fixture must exceed the tiny context window: {expected_tokens} tokens vs {} \
             context",
        model.max_context()
    );

    let items =
        normalize_embedding_items(vec![EmbeddingInputItem::Image { url: data_uri }]).unwrap();
    let err = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "context_length_exceeded");
            assert!(message.contains("input item 0"), "message: {message}");
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn embed_items_mixed_batch_preserves_input_order() {
    let model = tiny_embedding_model();
    let items = normalize_embedding_items(vec![
        EmbeddingInputItem::Text("a".to_string()),
        EmbeddingInputItem::Image {
            url: tiny_png_data_uri(1),
        },
        EmbeddingInputItem::Text("b".to_string()),
    ])
    .unwrap();
    let (data, _usage) = embed_items(&model, items, PoolingStrategy::MeanVisualTokens).unwrap();
    assert_eq!(data.len(), 3);
    assert_eq!(data[0].index, 0);
    assert_eq!(data[1].index, 1);
    assert_eq!(data[2].index, 2);
}

#[test]
fn check_pooled_embedding_total_tokens_accepts_within_budget() {
    let model = tiny_embedding_model();
    let items = normalize_embedding_items(vec![EmbeddingInputItem::Text("a".to_string())]).unwrap();
    check_pooled_embedding_total_tokens(&model, &items).unwrap();
}

#[test]
fn check_pooled_embedding_total_tokens_rejects_over_budget() {
    let model = tiny_embedding_model();
    // Every item individually fits `model.max_context()` (512, see
    // `tiny_embedding_model`'s config); the aggregate across all of them
    // does not. Repeated single-character words tokenize to one token
    // each for this vocab's no-merge tokenizer (see
    // `tiny_embedding_model`'s doc comment), so 80 items of 480 "a"
    // characters is 38,400 aggregate tokens against a 32,768 budget.
    let item_len = 480;
    assert!(item_len <= model.max_context());
    let item_count = 80;
    assert!(
        item_count * item_len > MAX_EMBEDDING_TOTAL_TOKENS,
        "fixture must actually exceed the budget it tests"
    );
    let items = normalize_embedding_items(
        std::iter::repeat_with(|| EmbeddingInputItem::Text("a".repeat(item_len)))
            .take(item_count)
            .collect(),
    )
    .unwrap();
    let err = check_pooled_embedding_total_tokens(&model, &items).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "batch_token_budget_exceeded");
            assert!(
                message.contains(&MAX_EMBEDDING_TOTAL_TOKENS.to_string()),
                "message: {message}"
            );
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn check_pooled_embedding_total_tokens_reports_the_oversized_item_first() {
    // A single item whose OWN scaffold token count already exceeds
    // `max_context` must be reported as `context_length_exceeded`
    // (naming that item), not the coarser aggregate
    // `batch_token_budget_exceeded` -- mirroring `embed_items`'s own
    // per-item-first check order.
    let base = tiny_embedding_model();
    let mut config = base.config.clone();
    config.max_position_embeddings = 4;
    let model = EmbeddingModel::new(
        base.weights.clone(),
        config,
        base.vision_weights.clone(),
        base.tokenizer.clone(),
    );
    assert_eq!(model.max_context(), 4);

    let items = normalize_embedding_items(vec![EmbeddingInputItem::Text("a".repeat(10))]).unwrap();
    let err = check_pooled_embedding_total_tokens(&model, &items).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "context_length_exceeded");
            assert!(message.contains("input item 0"), "message: {message}");
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn parse_pooling_defaults_to_mean_visual() {
    assert_eq!(
        parse_pooling(None).unwrap(),
        PoolingStrategy::MeanVisualTokens
    );
    assert_eq!(
        parse_pooling(Some("mean_visual")).unwrap(),
        PoolingStrategy::MeanVisualTokens
    );
}

#[test]
fn parse_pooling_accepts_last_token() {
    assert_eq!(
        parse_pooling(Some("last_token")).unwrap(),
        PoolingStrategy::LastToken
    );
}

#[test]
fn parse_pooling_rejects_unknown_value() {
    let err = parse_pooling(Some("max")).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_pooling",
            ..
        }
    ));
}

#[test]
fn embeddings_input_single_item_becomes_one_element_vec() {
    let input: EmbeddingsInput = serde_json::from_str(r#""hello""#).unwrap();
    assert_eq!(input.into_items().len(), 1);
}

#[test]
fn embeddings_input_array_preserves_order() {
    let input: EmbeddingsInput = serde_json::from_str(r#"["a", "b", "c"]"#).unwrap();
    let items = input.into_items();
    assert_eq!(items.len(), 3);
    assert!(matches!(&items[0], EmbeddingInputItem::Text(s) if s == "a"));
    assert!(matches!(&items[2], EmbeddingInputItem::Text(s) if s == "c"));
}

#[test]
fn embedding_input_item_parses_image_url_object() {
    let item: EmbeddingInputItem = serde_json::from_str(
        r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}"#,
    )
    .unwrap();
    assert!(
        matches!(item, EmbeddingInputItem::Image { url } if url == "data:image/png;base64,AA==")
    );
}

#[test]
fn embedding_input_item_rejects_unknown_type() {
    let err = serde_json::from_str::<EmbeddingInputItem>(r#"{"type":"video_url"}"#).unwrap_err();
    assert!(err.to_string().contains("video_url"));
}

#[test]
fn normalize_embedding_items_rejects_empty_input() {
    let err = normalize_embedding_items(vec![]).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_input",
            ..
        }
    ));
}

#[test]
fn normalize_embedding_items_rejects_remote_url() {
    let err = normalize_embedding_items(vec![EmbeddingInputItem::Image {
        url: "https://example.com/cat.png".to_string(),
    }])
    .unwrap_err();
    assert_eq!(err.code(), "unsupported_image_url_scheme");
}

#[test]
fn normalize_embedding_items_rejects_malformed_data_uri() {
    let err = normalize_embedding_items(vec![EmbeddingInputItem::Image {
        url: "data:image/png,not-base64-marked".to_string(),
    }])
    .unwrap_err();
    assert!(matches!(err, ApiError::BadRequest { .. }));
}

/// Issue: a bare `"data:..."` string arriving as a plain-text `input`
/// item was silently tokenized as text instead of being rejected. A
/// well-formed image data URI passed as text must be caught exactly
/// like a malformed one below -- the defect is the *shape* (text vs.
/// `image_url` object), not the payload's validity.
#[test]
fn normalize_embedding_items_rejects_well_formed_data_uri_as_text() {
    let err = normalize_embedding_items(vec![EmbeddingInputItem::Text(tiny_png_data_uri(0))])
        .unwrap_err();
    match err {
        ApiError::BadRequest { code, message } => {
            assert_eq!(code, "data_uri_as_text_item");
            assert!(message.contains("data:"), "message: {message}");
            assert!(message.contains("image_url"), "message: {message}");
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

/// Same shape defect, malformed payload -- must be caught before any
/// attempt to parse it as an image, so the error names the shape
/// problem rather than a base64/format problem.
#[test]
fn normalize_embedding_items_rejects_malformed_data_uri_as_text() {
    let err = normalize_embedding_items(vec![EmbeddingInputItem::Text(
        "data:image/png;base64,@@not-base64@@".to_string(),
    )])
    .unwrap_err();
    assert_eq!(err.code(), "data_uri_as_text_item");
}

/// The minimal data URI `data:,` (empty text/plain payload) is still a
/// data URI by shape, so it is refused as a text item like any other.
#[test]
fn normalize_embedding_items_rejects_empty_data_uri_as_text() {
    let err = normalize_embedding_items(vec![EmbeddingInputItem::Text("data:,".to_string())])
        .unwrap_err();
    assert_eq!(err.code(), "data_uri_as_text_item");
}

/// Prose that opens with "data:" is not a data URI: whitespace before the
/// first comma, or no comma at all, keeps it an ordinary text item.
#[test]
fn normalize_embedding_items_accepts_prose_beginning_with_data_colon() {
    for prose in [
        "data: results show 3 rows, all valid",
        "data:",
        "data: none",
    ] {
        let items =
            normalize_embedding_items(vec![EmbeddingInputItem::Text(prose.to_string())]).unwrap();
        assert!(matches!(&items[0], NormalizedEmbeddingItem::Text(s) if s == prose));
    }
}

/// Negative control: ordinary text is never affected by the new guard,
/// including text that merely mentions "data:" mid-string rather than
/// beginning with it.
#[test]
fn normalize_embedding_items_accepts_plain_text_mentioning_data_colon() {
    let items = normalize_embedding_items(vec![EmbeddingInputItem::Text(
        "see data: below for details".to_string(),
    )])
    .unwrap();
    assert!(
        matches!(&items[0], NormalizedEmbeddingItem::Text(s) if s == "see data: below for details")
    );
}

#[test]
fn normalize_embedding_items_preserves_order_of_mixed_items() {
    let items = normalize_embedding_items(vec![
        EmbeddingInputItem::Text("first".to_string()),
        EmbeddingInputItem::Text("second".to_string()),
    ])
    .unwrap();
    assert!(matches!(&items[0], NormalizedEmbeddingItem::Text(s) if s == "first"));
    assert!(matches!(&items[1], NormalizedEmbeddingItem::Text(s) if s == "second"));
}

#[test]
fn normalize_embedding_items_rejects_over_cap_input() {
    let items = (0..MAX_EMBEDDING_INPUT_COUNT + 1)
        .map(|_| EmbeddingInputItem::Text("x".to_string()))
        .collect();
    let err = normalize_embedding_items(items).unwrap_err();
    assert!(matches!(
        err,
        ApiError::BadRequest {
            code: "invalid_input",
            ..
        }
    ));
}

fn one(s: &str) -> Option<TextEmbeddingsInput> {
    Some(TextEmbeddingsInput::One(s.to_string()))
}

fn many(items: &[&str]) -> Option<TextEmbeddingsInput> {
    Some(TextEmbeddingsInput::Many(
        items.iter().map(ToString::to_string).collect(),
    ))
}

#[test]
fn parse_embeddings_input_missing_is_invalid_request() {
    let err = parse_embeddings_input(&None).unwrap_err();
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn parse_embeddings_input_null_is_invalid_request() {
    // `#[serde(default)]` maps an explicit JSON `null` to `None` before
    // this function ever runs, same as an omitted field; this pins that
    // the two are indistinguishable by the time validation sees them.
    let req: TextEmbeddingsRequest = serde_json::from_str(r#"{"input":null}"#).unwrap();
    let err = parse_embeddings_input(&req.input).unwrap_err();
    assert_eq!(err.code(), "invalid_request");
}

#[test]
fn parse_embeddings_input_single_string_is_one_element_at_index_zero() {
    let texts = parse_embeddings_input(&one("hello")).unwrap();
    assert_eq!(texts, vec!["hello".to_string()]);
}

#[test]
fn parse_embeddings_input_array_preserves_order() {
    let texts = parse_embeddings_input(&many(&["a", "b", "c"])).unwrap();
    assert_eq!(
        texts,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[test]
fn parse_embeddings_input_empty_string_is_invalid_input() {
    let err = parse_embeddings_input(&one("")).unwrap_err();
    assert_eq!(err.code(), "invalid_input");
}

#[test]
fn parse_embeddings_input_empty_array_is_invalid_input() {
    let err = parse_embeddings_input(&many(&[])).unwrap_err();
    assert_eq!(err.code(), "invalid_input");
}

#[test]
fn parse_embeddings_input_empty_string_inside_array_names_its_index() {
    let err = parse_embeddings_input(&many(&["a", "", "c"])).unwrap_err();
    assert_eq!(err.code(), "invalid_input");
    assert!(
        err.message().contains("input[1]"),
        "message must name the offending index: {}",
        err.message()
    );
}

#[test]
fn parse_embeddings_input_over_batch_limit_is_rejected() {
    let items: Vec<String> = (0..MAX_EMBEDDINGS_BATCH_SIZE + 1)
        .map(|i| i.to_string())
        .collect();
    let input = Some(TextEmbeddingsInput::Many(items));
    let err = parse_embeddings_input(&input).unwrap_err();
    assert_eq!(err.code(), "batch_size_exceeds_limit");
}

#[test]
fn parse_embeddings_input_at_batch_limit_is_accepted() {
    let items: Vec<String> = (0..MAX_EMBEDDINGS_BATCH_SIZE)
        .map(|i| i.to_string())
        .collect();
    let input = Some(TextEmbeddingsInput::Many(items));
    let texts = parse_embeddings_input(&input).unwrap();
    assert_eq!(texts.len(), MAX_EMBEDDINGS_BATCH_SIZE);
}

#[test]
fn check_embeddings_total_tokens_accepts_at_limit() {
    check_embeddings_total_tokens(MAX_EMBEDDINGS_TOTAL_TOKENS).unwrap();
}

#[test]
fn check_embeddings_total_tokens_rejects_over_limit() {
    let err = check_embeddings_total_tokens(MAX_EMBEDDINGS_TOTAL_TOKENS + 1).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "batch_token_budget_exceeded");
            assert!(
                message.contains(&(MAX_EMBEDDINGS_TOTAL_TOKENS + 1).to_string()),
                "message: {message}"
            );
            assert!(
                message.contains(&MAX_EMBEDDINGS_TOTAL_TOKENS.to_string()),
                "message: {message}"
            );
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn check_embeddings_item_fits_window_accepts_at_limit() {
    check_embeddings_item_fits_window(0, 512, 512).unwrap();
}

#[test]
fn check_embeddings_item_fits_window_rejects_over_limit() {
    let err = check_embeddings_item_fits_window(2, 513, 512).unwrap_err();
    match err {
        ApiError::BadRequest { message, code } => {
            assert_eq!(code, "context_length_exceeded");
            assert!(message.contains("item 2"), "message: {message}");
            assert!(message.contains("513"), "message: {message}");
            assert!(message.contains("512"), "message: {message}");
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[test]
fn check_embeddings_unsupported_fields_accepts_absent() {
    let req: TextEmbeddingsRequest = serde_json::from_str(r#"{"input":"hi"}"#).unwrap();
    check_embeddings_unsupported_fields(&req).unwrap();
}

#[test]
fn check_embeddings_unsupported_fields_accepts_user_and_float_encoding() {
    let req: TextEmbeddingsRequest =
        serde_json::from_str(r#"{"input":"hi","user":"trace-123","encoding_format":"float"}"#)
            .unwrap();
    check_embeddings_unsupported_fields(&req).unwrap();
}

#[test]
fn check_embeddings_unsupported_fields_rejects_dimensions() {
    let req: TextEmbeddingsRequest =
        serde_json::from_str(r#"{"input":"hi","dimensions":3}"#).unwrap();
    let err = check_embeddings_unsupported_fields(&req).unwrap_err();
    assert_eq!(err.code(), "unsupported_parameter");
    assert!(err.message().contains("dimensions"), "{}", err.message());
}

#[test]
fn check_embeddings_unsupported_fields_rejects_non_float_encoding_format() {
    let req: TextEmbeddingsRequest =
        serde_json::from_str(r#"{"input":"hi","encoding_format":"base64"}"#).unwrap();
    let err = check_embeddings_unsupported_fields(&req).unwrap_err();
    assert_eq!(err.code(), "unsupported_parameter");
    assert!(
        err.message().contains("encoding_format"),
        "{}",
        err.message()
    );
}

#[test]
fn text_embeddings_request_rejects_unknown_field() {
    // `pooling` is not a field of this wire type (unlike the sibling
    // vision-language `/v1/embeddings` route's `EmbeddingsRequest`,
    // which does accept one) -- `deny_unknown_fields` must reject it
    // rather than silently drop it.
    let err = serde_json::from_str::<TextEmbeddingsRequest>(r#"{"input":"hi","pooling":"cls"}"#)
        .unwrap_err();
    let _ = err; // presence of an Err is the assertion; message shape is serde's own.
}

#[test]
fn check_embeddings_model_accepts_omitted() {
    check_embeddings_model(None, "served-model").unwrap();
}

#[test]
fn check_embeddings_model_accepts_exact_match() {
    check_embeddings_model(Some("served-model"), "served-model").unwrap();
}

#[test]
fn check_embeddings_model_rejects_mismatch() {
    let err = check_embeddings_model(Some("other-model"), "served-model").unwrap_err();
    assert_eq!(err.code(), "model_not_found");
}

#[test]
fn check_embeddings_model_rejects_explicit_empty_string() {
    // Matches contract::validate_model_name's OptionalExact policy: an
    // explicit empty string is a present-but-wrong model, not absence.
    let err = check_embeddings_model(Some(""), "served-model").unwrap_err();
    assert_eq!(err.code(), "model_not_found");
}

#[test]
fn build_embeddings_response_shape() {
    // 0.5/0.25 (not 0.1/0.2/0.3): exactly representable in both f32 and
    // f64, so the f32 -> serde_json::Value (f64) widening this function
    // performs can't introduce a false mismatch against the literal
    // JSON array below.
    let body = build_embeddings_response("served-model", &[vec![0.5, 0.25, 1.0]], 4);
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "served-model");
    assert_eq!(body["usage"]["prompt_tokens"], 4);
    assert_eq!(body["usage"]["total_tokens"], 4);
    assert_eq!(body["data"][0]["object"], "embedding");
    assert_eq!(body["data"][0]["index"], 0);
    assert_eq!(
        body["data"][0]["embedding"],
        serde_json::json!([0.5, 0.25, 1.0])
    );
}

#[test]
fn build_embeddings_response_single_string_case_gets_index_zero() {
    let body = build_embeddings_response("served-model", &[vec![1.0]], 1);
    assert_eq!(body["data"].as_array().unwrap().len(), 1);
    assert_eq!(body["data"][0]["index"], 0);
}

#[test]
fn build_embeddings_response_indices_reflect_input_order() {
    // Mutation-proven: hardcoding `"index": 0` for every
    // element in `build_embeddings_response` leaves this test red while
    // `build_embeddings_response_single_string_case_gets_index_zero`
    // above stays green (its only correct index already is 0) --
    // exactly the reason this test uses three distinct vectors instead
    // of one.
    let body = build_embeddings_response("served-model", &[vec![0.0], vec![1.0], vec![2.0]], 3);
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    for (expected_index, item) in data.iter().enumerate() {
        assert_eq!(
            item["index"], expected_index,
            "data[{expected_index}] must carry index {expected_index}"
        );
        assert_eq!(
            item["embedding"],
            serde_json::json!([expected_index as f32]),
            "data[{expected_index}] must carry the embedding at that input position"
        );
    }
}
