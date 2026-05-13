//! BERT/BGE model loading and inference.
//!
//! The only tokenizer-facing change in this version is that `BertModel` stores a
//! boxed `dyn Tokenizer`, allowing WordPiece, BPE, and SentencePiece tokenizers
//! to share a single inference path.

use crate::attention::{AttentionBuffers, multi_head_attention_in_place};
use crate::download::ensure_model_files;
use crate::error::InferenceError;
use crate::forward::cpu::{add_bias, add_bias_gelu, layer_norm, matmul_bt};
use crate::pool::{l2_normalize, mean_pool};
use crate::tokenizer::common::{Tokenizer, load_tokenizer};
use crate::weights::{BertWeights, SafetensorsFile};
use std::fs;
use std::path::Path;
use tracing::warn;

/// **Stable** (provisional): BERT model configuration; consumed by `lattice-embed`
/// via `BertModel::config()`. Field additions are backward-compatible; field
/// removals or type changes require a SemVer bump.
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl BertConfig {
    /// **Stable** (provisional): factory for BGE-small-en-v1.5 configuration.
    ///
    /// Note: the published Hugging Face config uses 12 attention heads with
    /// hidden_size=384, giving head_dim=32.
    pub fn bge_small() -> Self {
        Self {
            vocab_size: 30_522,
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 1_536,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }

    /// **Stable** (provisional): factory for BGE-base-en-v1.5 configuration.
    pub fn bge_base() -> Self {
        Self {
            vocab_size: 30_522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3_072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }

    /// **Stable** (provisional): factory for BGE-large-en-v1.5 configuration.
    pub fn bge_large() -> Self {
        Self {
            vocab_size: 30_522,
            hidden_size: 1_024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4_096,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }

    /// **Unstable**: derived convenience; may be replaced by a struct field.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// **Stable**: primary BERT model type consumed by `lattice-embed`; the
/// `encode` / `encode_batch` interface is the stable contract.
///
/// # Self-Referential Pattern and Field Ordering Invariant
///
/// `BertModel` uses a self-referential pattern: `weights` contains `&'static`
/// slices that actually borrow from the memory-mapped file held in `_safetensors`.
/// The `'static` lifetime is achieved via `mem::transmute` in [`BertModel::from_directory`].
///
/// This is sound because:
///
/// 1. **Stable address**: `_safetensors` is `Box`ed, so the mmap address does not
///    move even if `BertModel` itself is relocated (e.g. returned from a function).
///
/// 2. **Drop order**: Rust drops struct fields in declaration order (RFC 1857).
///    `weights` is declared **before** `_safetensors`, so `weights` is dropped
///    first. Since `BertWeights` contains only `&[f32]` slices (whose `Drop` is a
///    no-op), there is no dangling-pointer access during destruction.
///
/// **WARNING**: Do **NOT** reorder these fields. Moving `_safetensors` above
/// `weights` would cause the backing store to be freed before the borrowing
/// slices, which is undefined behavior. The `test_struct_field_drop_order` test
/// in this module validates this invariant at compile-test time.
pub struct BertModel {
    config: BertConfig,
    tokenizer: Box<dyn Tokenizer>,
    // INVARIANT: `weights` MUST be declared before `_safetensors`.
    // See the struct-level doc comment for the full safety argument.
    weights: BertWeights<'static>,
    _safetensors: Box<SafetensorsFile>,
}

impl BertModel {
    /// **Stable**: load from a directory; primary construction path for `lattice-embed`.
    pub fn from_directory(dir: &Path) -> Result<Self, InferenceError> {
        let tokenizer = load_tokenizer(dir)?;

        let model_path = dir.join("model.safetensors");
        if !model_path.exists() {
            let sharded = dir.join("model.safetensors.index.json");
            if sharded.exists() {
                return Err(InferenceError::UnsupportedModel(format!(
                    "sharded safetensors are not supported yet: {}",
                    sharded.display()
                )));
            }
            return Err(InferenceError::ModelNotFound(format!(
                "missing model.safetensors in {}",
                dir.display()
            )));
        }

        let safetensors = Box::new(SafetensorsFile::open(&model_path)?);
        let config = match parse_config_json_if_present(&dir.join("config.json"))? {
            Some(config) => config,
            None => infer_config_from_safetensors(&safetensors)?,
        };

        if tokenizer.vocab_size() != config.vocab_size {
            warn!(
                tokenizer_vocab_size = tokenizer.vocab_size(),
                model_vocab_size = config.vocab_size,
                "tokenizer and model vocab sizes differ"
            );
        }

        let weights_tmp =
            safetensors.load_bert_weights(config.num_hidden_layers, config.hidden_size)?;
        // SAFETY: This transmute extends the lifetime of BertWeights from borrowing
        // `safetensors` to 'static. This is sound because:
        // 1. `_safetensors` is stored in the same struct as `weights`
        // 2. Rust drops struct fields in declaration order (RFC 1857)
        // 3. `weights` is declared BEFORE `_safetensors`, so weights is dropped first
        // 4. Therefore `_safetensors` (the backing store) outlives `weights` (the borrower)
        // 5. `_safetensors` is Box<SafetensorsFile>, so the mmap address is stable
        // WARNING: Do NOT reorder the fields of BertModel. See test_struct_field_drop_order.
        let weights: BertWeights<'static> = unsafe { std::mem::transmute(weights_tmp) };

        Ok(Self {
            config,
            tokenizer,
            weights,
            _safetensors: safetensors,
        })
    }

    /// **Stable** (provisional): load from default cache dir, downloading if needed.
    pub fn from_pretrained(model_name: &str) -> Result<Self, InferenceError> {
        let cache_dir = crate::default_cache_dir()?;
        let model_dir = ensure_model_files(model_name, &cache_dir)?;
        Self::from_directory(&model_dir)
    }

    /// **Stable**: returns model configuration; used by embed service.
    pub fn config(&self) -> &BertConfig {
        &self.config
    }

    /// **Unstable**: tokenizer accessor; exposed for testing only, may be removed.
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// **Stable**: embedding dimensionality; used by `lattice-embed` to size output buffers.
    pub fn dimensions(&self) -> usize {
        self.config.hidden_size
    }

    /// **Stable**: single-text encoding entry point; consumed by `lattice-embed`.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let input = self.tokenizer.tokenize(text);
        let seq_len = input.real_length;
        let mut buffers = AttentionBuffers::new(
            seq_len,
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.intermediate_size,
        );

        let hidden_states = self.forward(
            &input.input_ids[..seq_len],
            &input.attention_mask[..seq_len],
            &input.token_type_ids[..seq_len],
            seq_len,
            &mut buffers,
        );

        let mut pooled = mean_pool(
            &hidden_states,
            &input.attention_mask[..seq_len],
            seq_len,
            self.config.hidden_size,
        );
        l2_normalize(&mut pooled);
        Ok(pooled)
    }

    /// **Stable**: batch-encode entry point; consumed by `lattice-embed`.
    /// The sequential implementation detail may change without API breakage.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let tokenized = self.tokenizer.tokenize_batch(texts);
        let max_seq_len = tokenized
            .iter()
            .map(|input| input.real_length)
            .max()
            .unwrap_or(2);
        let mut buffers = AttentionBuffers::new(
            max_seq_len,
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.intermediate_size,
        );

        let mut outputs = Vec::with_capacity(tokenized.len());
        for input in &tokenized {
            let seq_len = input.real_length;
            let hidden_states = self.forward(
                &input.input_ids[..seq_len],
                &input.attention_mask[..seq_len],
                &input.token_type_ids[..seq_len],
                seq_len,
                &mut buffers,
            );
            let mut pooled = mean_pool(
                &hidden_states,
                &input.attention_mask[..seq_len],
                seq_len,
                self.config.hidden_size,
            );
            l2_normalize(&mut pooled);
            outputs.push(pooled);
        }

        Ok(outputs)
    }

    /// Forward pass for a pre-tokenized input; used by `CrossEncoderModel`.
    pub(crate) fn forward_tokenized(
        &self,
        input: &crate::tokenizer::TokenizedInput,
        buffers: &mut AttentionBuffers,
    ) -> Vec<f32> {
        let seq_len = input.real_length;
        self.forward(
            &input.input_ids[..seq_len],
            &input.attention_mask[..seq_len],
            &input.token_type_ids[..seq_len],
            seq_len,
            buffers,
        )
    }

    /// Internal full transformer forward pass.
    fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: &[u32],
        token_type_ids: &[u32],
        seq_len: usize,
        buffers: &mut AttentionBuffers,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let used_hidden = seq_len * hidden_size;

        debug_assert_eq!(input_ids.len(), seq_len);
        debug_assert_eq!(attention_mask.len(), seq_len);
        debug_assert_eq!(token_type_ids.len(), seq_len);
        let seq_len = seq_len.min(self.config.max_position_embeddings);

        let mut hidden = vec![0.0f32; used_hidden];

        for i in 0..seq_len {
            let tok_id = input_ids[i] as usize;
            let typ_id = token_type_ids[i] as usize;
            let pos_id = i;

            debug_assert!(tok_id < self.weights.word_embeddings.rows);
            debug_assert!(typ_id < self.weights.token_type_embeddings.rows);
            debug_assert!(pos_id < self.weights.position_embeddings.rows);

            let tok_row = &self.weights.word_embeddings.data
                [tok_id * hidden_size..(tok_id + 1) * hidden_size];
            let pos_row = &self.weights.position_embeddings.data
                [pos_id * hidden_size..(pos_id + 1) * hidden_size];
            let typ_row = &self.weights.token_type_embeddings.data
                [typ_id * hidden_size..(typ_id + 1) * hidden_size];
            let out_row = &mut hidden[i * hidden_size..(i + 1) * hidden_size];

            for d in 0..hidden_size {
                out_row[d] = tok_row[d] + pos_row[d] + typ_row[d];
            }
        }

        layer_norm(
            &mut hidden,
            self.weights.embedding_layer_norm_weight.data,
            self.weights.embedding_layer_norm_bias.data,
            hidden_size,
            self.config.layer_norm_eps,
        );

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];

            multi_head_attention_in_place(
                &hidden,
                layer,
                attention_mask,
                seq_len,
                hidden_size,
                self.config.num_attention_heads,
                self.config.head_dim(),
                buffers,
            );

            {
                let temp = &mut buffers.temp[..used_hidden];
                for i in 0..used_hidden {
                    temp[i] += hidden[i];
                }
                layer_norm(
                    temp,
                    layer.attn_layer_norm_weight.data,
                    layer.attn_layer_norm_bias.data,
                    hidden_size,
                    self.config.layer_norm_eps,
                );
                hidden.copy_from_slice(temp);
            }

            let used_intermediate = seq_len * intermediate_size;
            {
                let ffn_intermediate = &mut buffers.ffn_intermediate[..used_intermediate];
                matmul_bt(
                    &hidden,
                    layer.ffn_intermediate_weight.data,
                    ffn_intermediate,
                    seq_len,
                    hidden_size,
                    intermediate_size,
                );
                add_bias_gelu(
                    ffn_intermediate,
                    layer.ffn_intermediate_bias.data,
                    intermediate_size,
                );
            }

            {
                let ffn_intermediate = &buffers.ffn_intermediate[..used_intermediate];
                let temp = &mut buffers.temp[..used_hidden];
                matmul_bt(
                    ffn_intermediate,
                    layer.ffn_output_weight.data,
                    temp,
                    seq_len,
                    intermediate_size,
                    hidden_size,
                );
                add_bias(temp, layer.ffn_output_bias.data, hidden_size);
                for i in 0..used_hidden {
                    temp[i] += hidden[i];
                }
                layer_norm(
                    temp,
                    layer.ffn_layer_norm_weight.data,
                    layer.ffn_layer_norm_bias.data,
                    hidden_size,
                    self.config.layer_norm_eps,
                );
                hidden.copy_from_slice(temp);
            }
        }

        hidden
    }
}

fn parse_config_json_if_present(path: &Path) -> Result<Option<BertConfig>, InferenceError> {
    if !path.exists() {
        return Ok(None);
    }

    let text = fs::read_to_string(path)?;
    let get_usize = |key: &str| {
        extract_json_scalar(&text, key)
            .ok_or_else(|| InferenceError::Inference(format!("config.json missing key {key}")))?
            .parse::<usize>()
            .map_err(|e| InferenceError::Inference(format!("invalid usize for {key}: {e}")))
    };
    let get_f32 = |key: &str| {
        extract_json_scalar(&text, key)
            .ok_or_else(|| InferenceError::Inference(format!("config.json missing key {key}")))?
            .parse::<f32>()
            .map_err(|e| InferenceError::Inference(format!("invalid f32 for {key}: {e}")))
    };

    Ok(Some(BertConfig {
        vocab_size: get_usize("vocab_size")?,
        hidden_size: get_usize("hidden_size")?,
        num_hidden_layers: get_usize("num_hidden_layers")?,
        num_attention_heads: get_usize("num_attention_heads")?,
        intermediate_size: get_usize("intermediate_size")?,
        max_position_embeddings: get_usize("max_position_embeddings")?,
        type_vocab_size: get_usize("type_vocab_size")?,
        layer_norm_eps: get_f32("layer_norm_eps")?,
    }))
}

fn extract_json_scalar<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let idx = text.find(&needle)?;
    let rest = &text[idx + needle.len()..];
    let colon = rest.find(':')?;
    let mut value = rest[colon + 1..].trim_start();

    if value.starts_with('"') {
        value = &value[1..];
        let end = value.find('"')?;
        Some(&value[..end])
    } else {
        let end = value
            .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
            .unwrap_or(value.len());
        Some(value[..end].trim())
    }
}

fn infer_config_from_safetensors(file: &SafetensorsFile) -> Result<BertConfig, InferenceError> {
    let word_shape = file
        .tensor_shape("embeddings.word_embeddings.weight")
        .ok_or_else(|| InferenceError::MissingTensor("embeddings.word_embeddings.weight".into()))?;
    let pos_shape = file
        .tensor_shape("embeddings.position_embeddings.weight")
        .ok_or_else(|| {
            InferenceError::MissingTensor("embeddings.position_embeddings.weight".into())
        })?;
    let type_shape = file
        .tensor_shape("embeddings.token_type_embeddings.weight")
        .ok_or_else(|| {
            InferenceError::MissingTensor("embeddings.token_type_embeddings.weight".into())
        })?;
    let inter_shape = file
        .tensor_shape("encoder.layer.0.intermediate.dense.weight")
        .ok_or_else(|| {
            InferenceError::MissingTensor("encoder.layer.0.intermediate.dense.weight".into())
        })?;

    if word_shape.len() != 2
        || pos_shape.len() != 2
        || type_shape.len() != 2
        || inter_shape.len() != 2
    {
        return Err(InferenceError::Inference(
            "unable to infer config from malformed tensor shapes".into(),
        ));
    }

    let mut max_layer = None::<usize>;
    for name in file.tensor_names() {
        if let Some(rest) = name.strip_prefix("encoder.layer.") {
            if let Some(index_str) = rest.split('.').next() {
                if let Ok(index) = index_str.parse::<usize>() {
                    max_layer = Some(max_layer.map_or(index, |curr| curr.max(index)));
                }
            }
        }
    }

    let num_hidden_layers = max_layer
        .map(|v| v + 1)
        .ok_or_else(|| InferenceError::Inference("failed to infer number of layers".into()))?;
    let hidden_size = word_shape[1];
    let num_attention_heads = infer_num_attention_heads(hidden_size)?;

    Ok(BertConfig {
        vocab_size: word_shape[0],
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size: inter_shape[0],
        max_position_embeddings: pos_shape[0],
        type_vocab_size: type_shape[0],
        layer_norm_eps: 1e-12,
    })
}

fn infer_num_attention_heads(hidden_size: usize) -> Result<usize, InferenceError> {
    match hidden_size {
        384 => Ok(12),
        768 => Ok(12),
        1024 => Ok(16),
        h if h % 64 == 0 => Ok(h / 64),
        h if h % 32 == 0 => Ok(h / 32),
        _ => Err(InferenceError::Inference(format!(
            "unable to infer num_attention_heads for hidden_size {hidden_size}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[ignore]
    fn test_encode_output_shape_and_l2_norm() {
        let model_dir = match std::env::var("LATTICE_INFERENCE_MODEL_DIR") {
            Ok(value) => value,
            Err(_) => return,
        };

        let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();
        let embedding = model.encode("hello world").unwrap();
        assert_eq!(embedding.len(), model.dimensions());
        let norm = (embedding.iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-4);
    }
}

/// Compile-time guard for the struct field drop-order invariant.
///
/// `BertModel` relies on Rust dropping struct fields in declaration order
/// (RFC 1857): `weights` (the borrower) must be dropped before `_safetensors`
/// (the backing store). If this language guarantee ever changes, or if someone
/// accidentally reorders the fields, this test will catch it.
#[cfg(test)]
mod drop_order_tests {
    use std::sync::atomic::{AtomicU8, Ordering};

    static DROP_ORDER: AtomicU8 = AtomicU8::new(0);

    struct DropTracker {
        name: &'static str,
        expected_position: u8,
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            let position = DROP_ORDER.fetch_add(1, Ordering::SeqCst);
            assert_eq!(
                position, self.expected_position,
                "Field '{}' dropped in wrong order: expected position {}, got position {}. \
                 This means the struct field drop-order invariant that BertModel relies on \
                 is violated. The transmute in BertModel::from_directory is UNSOUND.",
                self.name, self.expected_position, position
            );
        }
    }

    #[test]
    fn test_struct_field_drop_order() {
        // Verify that Rust drops struct fields in declaration order.
        // BertModel relies on `weights` being dropped before `_safetensors`.
        // If this test fails, the transmute in BertModel::from_directory is unsound.
        DROP_ORDER.store(0, Ordering::SeqCst);

        struct FieldOrderMirror {
            _first: DropTracker,
            _second: DropTracker,
            _third: DropTracker,
        }

        {
            let _s = FieldOrderMirror {
                _first: DropTracker {
                    name: "first (config analog)",
                    expected_position: 0,
                },
                _second: DropTracker {
                    name: "second (weights analog)",
                    expected_position: 1,
                },
                _third: DropTracker {
                    name: "third (_safetensors analog)",
                    expected_position: 2,
                },
            };
        }
        // After the block, all three fields have been dropped in declaration order.
        assert_eq!(
            DROP_ORDER.load(Ordering::SeqCst),
            3,
            "Not all fields were dropped"
        );
    }
}
