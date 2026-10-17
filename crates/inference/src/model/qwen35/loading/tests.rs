use super::*;
use crate::model::qwen35_config::{LayerType, Qwen35Config};

struct NullTensorSource;

/// A mock tensor source backed by a HashMap. Returns MissingTensor for unknown names.
struct MockTensorSource {
    tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl TensorSource for MockTensorSource {
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
        self.tensors
            .get(name)
            .map(|(d, s)| (d.clone(), s.clone()))
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }
}

/// A mock tensor source whose declared metadata shape (`tensor_shape`) can differ
/// from the actual stored tensor shape, and which counts calls to
/// `get_f32_tensor_owned`. Used to prove the shape preflight in
/// `load_owned_tensor_checked` rejects a mismatch from metadata alone, without
/// ever materializing the owned buffer.
struct CallCountingMockTensorSource {
    /// name -> (data, declared shape returned by `tensor_shape`)
    tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    owned_get_calls: std::cell::Cell<usize>,
}

impl TensorSource for CallCountingMockTensorSource {
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
        self.owned_get_calls.set(self.owned_get_calls.get() + 1);
        self.tensors
            .get(name)
            .map(|(d, s)| (d.clone(), s.clone()))
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }
}

/// A mock tensor source that counts calls to `get_f32_tensor_owned`, so a test can
/// assert a shape mismatch is caught from the header-declared shape without ever
/// copying the tensor's data.
struct CountingTensorSource {
    tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    materialize_calls: std::cell::Cell<usize>,
}

impl TensorSource for CountingTensorSource {
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
        self.materialize_calls.set(self.materialize_calls.get() + 1);
        self.tensors
            .get(name)
            .map(|(d, s)| (d.clone(), s.clone()))
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }
}

impl TensorSource for NullTensorSource {
    fn has_tensor(&mut self, _name: &str) -> Result<bool, InferenceError> {
        Ok(false)
    }
    fn tensor_shape(&mut self, _name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        Ok(None)
    }
    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        Err(InferenceError::MissingTensor(name.to_string()))
    }
}

/// Verify that a key-head-shaped in_proj_b tensor in an asymmetric GDN config is rejected
/// at load time (returns Err, not Ok or panic).
#[test]
fn gdn_decay_tensor_key_head_shape_rejected() {
    // qwen36_35b_a3b has linear_num_key_heads=16, linear_num_value_heads=Some(32) —
    // an asymmetric config so the shape check actually does something.
    let cfg = Qwen35Config::qwen36_35b_a3b();
    let hidden = cfg.hidden_size; // 2048
    let key_heads = cfg.linear_num_key_heads; // 16
    let value_heads = cfg.linear_num_value_heads(); // 32
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();

    assert_ne!(
        key_heads, value_heads,
        "test requires asymmetric key/value heads"
    );

    let prefix = "layers.0";
    let mut src = MockTensorSource {
        tensors: [
            // qkv and z are correctly-shaped so the in_proj_b mismatch below is the
            // first (and only) error load_linear_attention_weights should surface.
            (
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                (vec![0.0f32; qkv_dim * hidden], vec![qkv_dim, hidden]),
            ),
            (
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                (vec![0.0f32; output_dim * hidden], vec![output_dim, hidden]),
            ),
            // in_proj_b is key-head-shaped ([16, hidden]) instead of value-head-shaped
            // ([32, hidden]) — this should be caught by load_owned_tensor_checked
            (
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                (vec![0.0f32; key_heads * hidden], vec![key_heads, hidden]),
            ),
        ]
        .into_iter()
        .collect(),
    };

    let result = load_linear_attention_weights(&mut src, &cfg, prefix);

    match result {
        Err(InferenceError::ShapeMismatch {
            name,
            expected,
            actual,
        }) => {
            assert!(
                name.contains("in_proj_b"),
                "mismatch should name the in_proj_b tensor, got: {name}"
            );
            assert_eq!(
                expected,
                vec![value_heads, hidden],
                "expected shape should be value-head-sized"
            );
            assert_eq!(
                actual,
                vec![key_heads, hidden],
                "actual shape should reflect the key-head-shaped data we supplied"
            );
        }
        Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
        Ok(_) => panic!("expected Err for key-head-shaped decay tensor, got Ok"),
    }
}

/// A finite but undersized dense-FFN `gate_proj` tensor must be rejected during
/// loading (`Err(ShapeMismatch)`), not accepted and left to panic downstream in
/// the GEMM bounds assertion during the forward pass.
#[test]
fn dense_ffn_undersized_gate_proj_shape_rejected() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let prefix = "layers.0";
    let undersized_rows = intermediate / 2;

    let mut src = MockTensorSource {
        tensors: [(
            format!("{prefix}.mlp.gate_proj.weight"),
            (
                vec![0.0f32; undersized_rows * hidden],
                vec![undersized_rows, hidden],
            ),
        )]
        .into_iter()
        .collect(),
    };

    match load_dense_ffn_weights(&mut src, &cfg, prefix) {
        Err(InferenceError::ShapeMismatch {
            name,
            expected,
            actual,
        }) => {
            assert!(
                name.contains("gate_proj"),
                "mismatch should name the gate_proj tensor, got: {name}"
            );
            assert_eq!(expected, vec![intermediate, hidden]);
            assert_eq!(actual, vec![undersized_rows, hidden]);
        }
        Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
        Ok(_) => panic!("expected Err for undersized gate_proj, got Ok"),
    }
}

/// A `gate_proj` tensor whose safetensors-declared axis shape is transposed
/// (`[hidden, intermediate]` instead of `[intermediate, hidden]`) but has the same
/// total element count must be rejected, not silently reinterpreted as row-major.
/// Before the loader validated declared axis shapes, only the flattened element
/// count was checked, so a transposed tensor with matching element count passed
/// ingress and was misparsed, producing silently wrong outputs downstream.
#[test]
fn dense_ffn_rejects_transposed_gate_proj_with_matching_element_count() {
    let cfg = Qwen35Config {
        hidden_size: 4,
        intermediate_size: 8,
        ..Qwen35Config::qwen35_2b()
    };
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let prefix = "layers.0";

    let mut src = MockTensorSource {
        tensors: [
            // Declared as [hidden, intermediate] — the transpose of the expected
            // [intermediate, hidden] — with the same total element count (32).
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                (vec![0.0f32; hidden * inter], vec![hidden, inter]),
            ),
            (
                format!("{prefix}.mlp.up_proj.weight"),
                (vec![0.0f32; inter * hidden], vec![inter, hidden]),
            ),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                (vec![0.0f32; hidden * inter], vec![hidden, inter]),
            ),
        ]
        .into_iter()
        .collect(),
    };

    match load_dense_ffn_weights(&mut src, &cfg, prefix) {
        Err(InferenceError::ShapeMismatch {
            name,
            expected,
            actual,
        }) => {
            assert!(
                name.contains("gate_proj"),
                "mismatch should name the gate_proj tensor, got: {name}"
            );
            assert_eq!(expected, vec![inter, hidden]);
            assert_eq!(actual, vec![hidden, inter]);
        }
        Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
        Ok(_) => panic!("expected Err for transposed gate_proj, got Ok"),
    }
}

/// A finite but undersized full-attention `q_proj` tensor (half the required
/// `2*q_dim` rows for the attn_output_gate=true fused Q+gate projection) must be
/// rejected during loading, not accepted and left to panic downstream.
#[test]
fn full_attention_undersized_q_proj_shape_rejected() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let hidden = cfg.hidden_size;
    let q_dim = cfg.full_q_dim();
    let prefix = "layers.0";
    let undersized_rows = q_dim; // expected 2*q_dim; supply only q_dim rows

    let mut src = MockTensorSource {
        tensors: [(
            format!("{prefix}.self_attn.q_proj.weight"),
            (
                vec![0.0f32; undersized_rows * hidden],
                vec![undersized_rows, hidden],
            ),
        )]
        .into_iter()
        .collect(),
    };

    let result = load_full_attention_weights(&mut src, &cfg, prefix);

    match result {
        Err(InferenceError::ShapeMismatch {
            name,
            expected,
            actual,
        }) => {
            assert!(
                name.contains("q_proj"),
                "mismatch should name the q_proj tensor, got: {name}"
            );
            assert_eq!(expected, vec![2 * q_dim, hidden]);
            assert_eq!(actual, vec![undersized_rows, hidden]);
        }
        Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
        Ok(_) => panic!("expected Err for undersized q_proj, got Ok"),
    }
}

/// A finite but undersized GDN `in_proj_qkv` tensor must be rejected during
/// loading, not accepted and left to panic downstream.
#[test]
fn gdn_undersized_in_proj_qkv_shape_rejected() {
    let cfg = Qwen35Config::qwen35_0_8b();
    let hidden = cfg.hidden_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let prefix = "layers.0";
    let undersized_rows = qkv_dim / 2;

    let mut src = MockTensorSource {
        tensors: [(
            format!("{prefix}.linear_attn.in_proj_qkv.weight"),
            (
                vec![0.0f32; undersized_rows * hidden],
                vec![undersized_rows, hidden],
            ),
        )]
        .into_iter()
        .collect(),
    };

    let result = load_linear_attention_weights(&mut src, &cfg, prefix);

    match result {
        Err(InferenceError::ShapeMismatch {
            name,
            expected,
            actual,
        }) => {
            assert!(
                name.contains("in_proj_qkv"),
                "mismatch should name the in_proj_qkv tensor, got: {name}"
            );
            assert_eq!(expected, vec![qkv_dim, hidden]);
            assert_eq!(actual, vec![undersized_rows, hidden]);
        }
        Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
        Ok(_) => panic!("expected Err for undersized in_proj_qkv, got Ok"),
    }
}

#[test]
fn shape_mismatch_rejected_before_tensor_data_is_materialized() {
    let mut src = CountingTensorSource {
        tensors: [("t".to_string(), (vec![0.0f32; 4], vec![4, 1]))]
            .into_iter()
            .collect(),
        materialize_calls: std::cell::Cell::new(0),
    };

    let result = load_owned_tensor_checked(&mut src, "t", &[1, 4]);

    assert!(
        matches!(result, Err(InferenceError::ShapeMismatch { .. })),
        "expected ShapeMismatch, got {result:?}"
    );
    assert_eq!(
        src.materialize_calls.get(),
        0,
        "a declared-shape mismatch must be rejected before the tensor data is copied"
    );
}

#[test]
fn moe_missing_num_experts_returns_err_not_panic() {
    let mut cfg = Qwen35Config::qwen36_35b_a3b();
    cfg.num_experts = None;
    cfg.num_experts_per_tok = Some(8);
    let mut src = NullTensorSource;
    let result = load_moe_ffn_weights(&mut src, &cfg, "layers.0", cfg.hidden_size);
    match result {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("num_experts"),
                "message should name the field: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
        Ok(_) => panic!("expected Err, got Ok"),
    }
}

#[test]
fn moe_missing_num_experts_per_tok_returns_err_not_panic() {
    let mut cfg = Qwen35Config::qwen36_35b_a3b();
    cfg.num_experts = Some(256);
    cfg.num_experts_per_tok = None;
    let mut src = NullTensorSource;
    let result = load_moe_ffn_weights(&mut src, &cfg, "layers.0", cfg.hidden_size);
    match result {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("num_experts_per_tok"),
                "message should name the field: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
        Ok(_) => panic!("expected Err, got Ok"),
    }
}

// ── checked_* overflow-guard boundary tests ─────────────────────────────────────
//
// These dimension products are derived from config.json fields (an untrusted
// checkpoint directory). A config with maliciously large but individually
// in-range `usize` fields (e.g. num_attention_heads and head_dim each valid
// on their own) can make the *product* overflow usize, which would silently
// wrap to a small value in release builds and load a shape-incompatible
// (undersized) tensor buffer, or panic in debug builds. `checked_*` must
// reject the overflowing case with a typed `InvalidInput` error and accept a
// large-but-non-overflowing product.

#[test]
fn checked_double_accepts_large_valid_and_rejects_overflow() {
    assert_eq!(checked_double(usize::MAX / 2, "x").unwrap(), usize::MAX - 1);
    assert!(checked_double(usize::MAX, "x").is_err());
}

#[test]
fn checked_full_q_dim_accepts_large_valid_and_rejects_overflow() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_attention_heads = 1 << 40;
    cfg.head_dim = 1 << 20;
    assert_eq!(checked_full_q_dim(&cfg).unwrap(), 1 << 60);

    cfg.num_attention_heads = usize::MAX;
    cfg.head_dim = 2;
    match checked_full_q_dim(&cfg) {
        Err(InferenceError::InvalidInput(msg)) => assert!(msg.contains("q_dim")),
        other => panic!("expected InvalidInput overflow error, got {other:?}"),
    }
}

#[test]
fn checked_full_kv_dim_accepts_large_valid_and_rejects_overflow() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.num_key_value_heads = 1 << 40;
    cfg.head_dim = 1 << 20;
    assert_eq!(checked_full_kv_dim(&cfg).unwrap(), 1 << 60);

    cfg.num_key_value_heads = usize::MAX;
    cfg.head_dim = 2;
    match checked_full_kv_dim(&cfg) {
        Err(InferenceError::InvalidInput(msg)) => assert!(msg.contains("kv_dim")),
        other => panic!("expected InvalidInput overflow error, got {other:?}"),
    }
}

#[test]
fn checked_linear_qkv_dim_accepts_large_valid_and_rejects_overflow() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.linear_num_key_heads = 1 << 20;
    cfg.linear_key_head_dim = 1 << 20;
    cfg.linear_num_value_heads = Some(1 << 20);
    cfg.linear_value_head_dim = 1 << 20;
    // k = v = 1<<40, qkv = k + k + v = 3<<40 -- large but well within usize range.
    assert_eq!(checked_linear_qkv_dim(&cfg).unwrap(), 3 * (1usize << 40));

    cfg.linear_num_key_heads = usize::MAX;
    cfg.linear_key_head_dim = 2;
    match checked_linear_qkv_dim(&cfg) {
        Err(InferenceError::InvalidInput(msg)) => assert!(msg.contains("overflows usize")),
        other => panic!("expected InvalidInput overflow error, got {other:?}"),
    }
}

#[test]
fn checked_linear_output_dim_accepts_large_valid_and_rejects_overflow() {
    let mut cfg = Qwen35Config::qwen35_0_8b();
    cfg.linear_num_value_heads = Some(1 << 40);
    cfg.linear_value_head_dim = 1 << 20;
    assert_eq!(checked_linear_output_dim(&cfg).unwrap(), 1 << 60);

    cfg.linear_num_value_heads = Some(usize::MAX);
    cfg.linear_value_head_dim = 2;
    match checked_linear_output_dim(&cfg) {
        Err(InferenceError::InvalidInput(msg)) => assert!(msg.contains("output_dim")),
        other => panic!("expected InvalidInput overflow error, got {other:?}"),
    }
}

// ── shape-preflight-before-materialization tests ───────────────────────────────
//
// `load_owned_tensor_checked` must reject a shape mismatch using only the
// metadata `tensor_shape` returns, before ever calling `get_f32_tensor_owned`.
// A checkpoint tensor whose declared shape is oversized should not pay the cost
// of decoding/copying the owned buffer just to be rejected afterward.

#[test]
fn mismatched_shape_rejected_without_materializing_owned_buffer() {
    let name = "layers.0.mlp.gate_proj.weight";
    // Declared (metadata) shape is huge and mismatched; the actual stored data is
    // tiny, so if the preflight were skipped and get_f32_tensor_owned were called,
    // it would return this tiny buffer paired with the huge declared shape below.
    let declared_mismatched_shape = vec![1 << 20, 1 << 20];
    let mut src = CallCountingMockTensorSource {
        tensors: [(
            name.to_string(),
            (vec![0.0f32; 4], declared_mismatched_shape.clone()),
        )]
        .into_iter()
        .collect(),
        owned_get_calls: std::cell::Cell::new(0),
    };

    let result = load_owned_tensor_checked(&mut src, name, &[4, 4]);

    match result {
        Err(InferenceError::ShapeMismatch {
            name: got_name,
            expected,
            actual,
        }) => {
            assert_eq!(got_name, name);
            assert_eq!(expected, vec![4, 4]);
            assert_eq!(actual, declared_mismatched_shape);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
    assert_eq!(
        src.owned_get_calls.get(),
        0,
        "a mismatched tensor must be rejected via the metadata preflight \
             without ever materializing the owned buffer"
    );
}

#[test]
fn correctly_shaped_tensor_still_loads() {
    let name = "layers.0.mlp.gate_proj.weight";
    let mut src = CallCountingMockTensorSource {
        tensors: [(name.to_string(), (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))]
            .into_iter()
            .collect(),
        owned_get_calls: std::cell::Cell::new(0),
    };

    let result = load_owned_tensor_checked(&mut src, name, &[2, 2]);

    assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        src.owned_get_calls.get(),
        1,
        "a correctly-shaped tensor must still be materialized exactly once"
    );
}

#[test]
fn moe_gate_up_shape_overflow_returns_err_not_panic() {
    let mut cfg = Qwen35Config::qwen36_35b_a3b();
    cfg.num_experts = Some(256);
    cfg.num_experts_per_tok = Some(8);
    cfg.moe_intermediate_size = Some(usize::MAX);
    let hidden = cfg.hidden_size;
    // The router's gate.weight is present and correctly shaped so the loader
    // actually reaches the overflowing gate_up_proj shape computation instead of
    // failing earlier on a missing tensor.
    let mut src = MockTensorSource {
        tensors: [(
            "layers.0.mlp.gate.weight".to_string(),
            (vec![0.0f32; 256 * hidden], vec![256, hidden]),
        )]
        .into_iter()
        .collect(),
    };
    let result = load_moe_ffn_weights(&mut src, &cfg, "layers.0", hidden);
    match result {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("moe_intermediate_size"),
                "message should name the field: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
        Ok(_) => panic!("expected Err for overflowing moe_intermediate_size, got Ok"),
    }
}

/// A tiny, single-layer, dense-FFN, linear-attention-only config used to exercise
/// `load_weights` end-to-end without a real checkpoint.
fn tiny_linear_layer_cfg() -> Qwen35Config {
    Qwen35Config {
        hidden_size: 4,
        num_hidden_layers: 1,
        vocab_size: 4,
        intermediate_size: 8,
        linear_num_key_heads: 1,
        linear_key_head_dim: 2,
        linear_num_value_heads: Some(1),
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 2,
        layer_types: vec![LayerType::LinearAttention],
        layer_mask: vec![true],
        tie_word_embeddings: true,
        ..Qwen35Config::qwen35_2b()
    }
}

/// Build a complete, cfg-consistent tensor set for `tiny_linear_layer_cfg()`, with
/// `input_layernorm`/`post_attention_layernorm` shaped `[hidden]` (or overridden via
/// `iln_len`/`paln_len` to a malformed length).
fn tiny_linear_layer_tensors(
    cfg: &Qwen35Config,
    iln_len: usize,
    paln_len: usize,
) -> MockTensorSource {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();
    let kernel_size = cfg.linear_conv_kernel_dim;
    let prefix = "model.language_model.layers.0";

    MockTensorSource {
        tensors: [
            (
                "model.language_model.embed_tokens.weight".to_string(),
                (vec![0.0f32; vocab * hidden], vec![vocab, hidden]),
            ),
            (
                "model.language_model.norm.weight".to_string(),
                (vec![1.0f32; hidden], vec![hidden]),
            ),
            (
                format!("{prefix}.input_layernorm.weight"),
                (vec![1.0f32; iln_len], vec![iln_len]),
            ),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                (vec![1.0f32; paln_len], vec![paln_len]),
            ),
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                (vec![0.0f32; inter * hidden], vec![inter, hidden]),
            ),
            (
                format!("{prefix}.mlp.up_proj.weight"),
                (vec![0.0f32; inter * hidden], vec![inter, hidden]),
            ),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                (vec![0.0f32; hidden * inter], vec![hidden, inter]),
            ),
            (
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                (vec![0.0f32; qkv_dim * hidden], vec![qkv_dim, hidden]),
            ),
            (
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                (vec![0.0f32; output_dim * hidden], vec![output_dim, hidden]),
            ),
            (
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                (
                    vec![0.0f32; value_heads * hidden],
                    vec![value_heads, hidden],
                ),
            ),
            (
                format!("{prefix}.linear_attn.in_proj_a.weight"),
                (
                    vec![0.0f32; value_heads * hidden],
                    vec![value_heads, hidden],
                ),
            ),
            (
                format!("{prefix}.linear_attn.A_log"),
                (vec![0.0f32; value_heads], vec![value_heads]),
            ),
            (
                format!("{prefix}.linear_attn.dt_bias"),
                (vec![0.0f32; value_heads], vec![value_heads]),
            ),
            (
                format!("{prefix}.linear_attn.conv1d.weight"),
                (
                    vec![0.0f32; qkv_dim * kernel_size],
                    vec![qkv_dim, 1, kernel_size],
                ),
            ),
            (
                format!("{prefix}.linear_attn.norm.weight"),
                (
                    vec![1.0f32; cfg.linear_value_head_dim],
                    vec![cfg.linear_value_head_dim],
                ),
            ),
            (
                format!("{prefix}.linear_attn.out_proj.weight"),
                (vec![0.0f32; hidden * output_dim], vec![hidden, output_dim]),
            ),
        ]
        .into_iter()
        .collect(),
    }
}

/// A malformed per-layer `input_layernorm` (length `hidden - 1`, e.g. from a
/// checkpoint's `config.json` disagreeing with its own tensor) must be rejected by
/// `load_weights`, not silently truncate: before this guard, the unchecked
/// `load_owned_tensor` accepted any length and `qwen35_rms_norm` zipped only the
/// shared prefix, producing wrong logits with no error signal.
#[test]
fn load_weights_rejects_malformed_input_layernorm() {
    let cfg = tiny_linear_layer_cfg();
    let hidden = cfg.hidden_size;
    let mut src = tiny_linear_layer_tensors(&cfg, hidden - 1, hidden);
    let validated = cfg.clone().validate().expect("tiny cfg must validate");
    match load_weights(&mut src, &validated) {
        Err(InferenceError::ShapeMismatch { name, .. }) => {
            assert!(
                name.contains("input_layernorm"),
                "error must name input_layernorm, got: {name}"
            );
        }
        Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
        Ok(_) => panic!("expected Err for malformed input_layernorm, got Ok"),
    }
}

/// Same as above for `post_attention_layernorm`.
#[test]
fn load_weights_rejects_malformed_post_attention_layernorm() {
    let cfg = tiny_linear_layer_cfg();
    let hidden = cfg.hidden_size;
    let mut src = tiny_linear_layer_tensors(&cfg, hidden, hidden - 1);
    let validated = cfg.clone().validate().expect("tiny cfg must validate");
    match load_weights(&mut src, &validated) {
        Err(InferenceError::ShapeMismatch { name, .. }) => {
            assert!(
                name.contains("post_attention_layernorm"),
                "error must name post_attention_layernorm, got: {name}"
            );
        }
        Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
        Ok(_) => panic!("expected Err for malformed post_attention_layernorm, got Ok"),
    }
}

#[test]
fn load_weights_accepts_correctly_shaped_layernorms() {
    let cfg = tiny_linear_layer_cfg();
    let hidden = cfg.hidden_size;
    let mut src = tiny_linear_layer_tensors(&cfg, hidden, hidden);
    let validated = cfg.clone().validate().expect("tiny cfg must validate");
    load_weights(&mut src, &validated).expect("correctly shaped layernorms must load");
}

/// Regression for #1035: a finite, undersized dense-FFN tensor must cause
/// `load_weights` — the routine `Qwen35Model::from_safetensors` uses to assemble a
/// model — to return `Err`, not reach the forward pass and panic in the release-active
/// GEMM bounds guard.
#[test]
fn load_weights_rejects_undersized_dense_ffn_tensor() {
    let cfg = tiny_linear_layer_cfg();
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let mut src = tiny_linear_layer_tensors(&cfg, hidden, hidden);
    // down_proj declared as [hidden, inter - 1] instead of [hidden, inter] --
    // finite, just smaller than the config requires.
    src.tensors.insert(
        "model.language_model.layers.0.mlp.down_proj.weight".to_string(),
        (vec![0.0f32; hidden * (inter - 1)], vec![hidden, inter - 1]),
    );

    let validated = cfg.clone().validate().expect("tiny cfg must validate");
    match load_weights(&mut src, &validated) {
        Err(InferenceError::ShapeMismatch { name, .. }) => {
            assert!(
                name.contains("down_proj"),
                "error must name down_proj, got: {name}"
            );
        }
        Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
        Ok(_) => panic!(
            "expected Err for undersized down_proj, got Ok (an undersized tensor would \
                 reach the forward pass and panic in the GEMM bounds guard)"
        ),
    }
}

/// A config with `num_attention_heads = 2^63` and `head_dim = 2` overflows
/// `full_q_dim()`/`full_kv_dim()` in release builds. `load_full_attention_weights`
/// must reject this via the checked accessors before deriving a wrapped, undersized
/// expected tensor shape for `q_proj`/`k_proj`/`v_proj`/`o_proj` — a checkpoint that
/// happens to match the wrapped shape would otherwise load successfully and panic
/// (or corrupt state) once the forward pass indexes by the original, unwrapped head
/// count.
#[test]
fn load_full_attention_weights_rejects_overflowing_config() {
    let cfg = Qwen35Config {
        num_attention_heads: 1 << 63,
        num_key_value_heads: 1 << 63,
        head_dim: 2,
        ..Qwen35Config::qwen35_2b()
    };
    let mut src = NullTensorSource;

    match load_full_attention_weights(&mut src, &cfg, "layers.0") {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("overflow"),
                "error must describe the overflow, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
        Ok(_) => panic!(
            "expected Err for overflowing full-attention config, got Ok (would derive a \
                 wrapped tensor shape and panic downstream)"
        ),
    }
}

/// Same overflow-rejection contract for the GatedDeltaNet dimension helpers used by
/// `load_weights`: a pathological `linear_num_key_heads` / `linear_key_head_dim` /
/// `linear_value_head_dim` combination must surface as a typed error at ingress
/// instead of wrapping into an undersized `qkv_dim`/`output_dim`.
#[test]
fn load_weights_rejects_overflowing_linear_attention_config() {
    let cfg = Qwen35Config {
        linear_num_key_heads: 1 << 63,
        linear_num_value_heads: Some(1 << 63),
        linear_key_head_dim: 2,
        linear_value_head_dim: 2,
        ..Qwen35Config::qwen35_2b()
    };
    // `load_weights` only accepts a validated config, so the pathological combination
    // is rejected before it can reach the loader at all -- the contract this test
    // guards (no wrapped `qkv_dim`/`output_dim` ever reaches tensor assembly) now
    // holds one layer earlier. The per-helper loader guard is covered directly by
    // `load_linear_attention_weights_rejects_overflowing_config`.
    let err = cfg
        .validate()
        .expect_err("overflowing linear-attention config must not validate")
        .to_string();
    assert!(
        !err.is_empty(),
        "rejection must carry a diagnostic, got an empty message"
    );
}
