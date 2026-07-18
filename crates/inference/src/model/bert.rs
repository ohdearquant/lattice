//! BERT/BGE model loading and inference.
//!
//! The only tokenizer-facing change in this version is that `BertModel` stores a
//! boxed `dyn Tokenizer`, allowing WordPiece, BPE, and SentencePiece tokenizers
//! to share a single inference path.

use crate::attention::{
    AttentionBuffers, multi_head_attention_batched, multi_head_attention_in_place,
};
use crate::download::ensure_model_files;
use crate::error::InferenceError;
use crate::forward::cpu::{
    add_bias, add_bias_gelu, layer_norm, matmul_bt, residual_add_layer_norm,
};
use crate::lora_hook::{LoraHook, NoopLoraHook, apply_lora_rows};
use crate::pool::{BertPooling, cls_pool, l2_normalize, mean_pool};
use crate::tokenizer::common::{Tokenizer, load_tokenizer};
use crate::weights::{BertWeights, SafetensorsFile, TransformerLayerWeights};
use std::fs;
use std::path::Path;
use tracing::warn;

/// Per-layer fused Q/K/V weight+bias, built once at model-load time from a
/// `TransformerLayerWeights` layer's separate `query`/`key`/`value` tensors.
///
/// Kept as its own private structure rather than a field on
/// `TransformerLayerWeights` itself: that struct is an all-public,
/// externally constructible API surface (`crate::weights::TransformerLayerWeights`),
/// so adding a field there is a breaking change under SemVer (adding a public
/// field to an all-public struct forces every downstream struct literal to
/// change). See `crate::attention::standard::multi_head_attention_in_place`'s
/// doc comment and PR #678.
struct LayerFusedQkv {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerFusedQkv {
    /// Concatenate `query`/`key`/`value` weight rows (and biases) vertically
    /// into one `[3*hidden_size, hidden_size]` weight (`[3*hidden_size]`
    /// bias), enabling one `matmul_bt` call per layer instead of three
    /// (#674). Building this once per layer at load time, rather than per
    /// forward call, keeps the perf win: the concat cost is amortized over
    /// every subsequent `encode`/`encode_batch` call for this model.
    fn build(layer: &TransformerLayerWeights<'_>) -> Self {
        let mut weight = Vec::with_capacity(
            layer.query_weight.data.len()
                + layer.key_weight.data.len()
                + layer.value_weight.data.len(),
        );
        weight.extend_from_slice(layer.query_weight.data);
        weight.extend_from_slice(layer.key_weight.data);
        weight.extend_from_slice(layer.value_weight.data);

        let mut bias = Vec::with_capacity(
            layer.query_bias.data.len() + layer.key_bias.data.len() + layer.value_bias.data.len(),
        );
        bias.extend_from_slice(layer.query_bias.data);
        bias.extend_from_slice(layer.key_bias.data);
        bias.extend_from_slice(layer.value_bias.data);

        Self { weight, bias }
    }
}

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

    /// **Unstable**: parse a `BertConfig` from `config.json` text, without
    /// touching the filesystem; signature may change as the from-bytes model
    /// loading API matures.
    ///
    /// This is the string counterpart to the file-based `config.json` parsing
    /// `from_directory` performs internally. Used by
    /// [`BertModel::from_bytes`]; unlike `from_directory`, there is no
    /// fallback to inferring the config from safetensors tensor shapes when a
    /// required key is missing.
    pub fn from_json_str(config_json: &str) -> Result<Self, InferenceError> {
        parse_config_json_str(config_json)
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
    /// Pooling strategy used to reduce hidden states to a single embedding vector.
    /// Defaults to `BertPooling::Mean` for backwards compatibility.
    pooling: BertPooling,
    /// Per-layer fused Q/K/V weight+bias (#674/#678), built once here at load
    /// time from `weights.layers`. Not part of the self-referential
    /// `weights`/`_safetensors` pair above: this is owned, copied data, so it
    /// carries no lifetime/drop-order constraint.
    fused_qkv: Vec<LayerFusedQkv>,
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

        Self::assemble(config, tokenizer, safetensors)
    }

    /// **Stable** (provisional): load from default cache dir, downloading if needed.
    pub fn from_pretrained(model_name: &str) -> Result<Self, InferenceError> {
        let cache_dir = crate::default_cache_dir()?;
        let model_dir = ensure_model_files(model_name, &cache_dir)?;
        Self::from_directory(&model_dir)
    }

    /// **Unstable**: construct a model from in-memory bytes, without touching the
    /// filesystem; signature may change as this from-bytes loading API matures.
    ///
    /// This is the in-memory counterpart to [`from_directory`](Self::from_directory):
    /// it takes the same three logical inputs a model directory provides
    /// (safetensors weights, `config.json`, `tokenizer.json`) as owned/borrowed
    /// bytes instead of a path. Intended for hosts with no real filesystem:
    /// wasm32-unknown-unknown has no working `mmap` (see
    /// [`SafetensorsFile::from_bytes`]), and for callers that already hold the
    /// model bytes in memory (e.g. bytes handed in from JavaScript).
    ///
    /// `config_json` and `tokenizer_json` must be the UTF-8 text of
    /// `config.json` and `tokenizer.json` respectively. Unlike
    /// `from_directory`, there is no config-from-safetensors-shape inference
    /// fallback and no support for the legacy tokenizer formats
    /// (`vocab.json`+`merges.txt`, `vocab.txt`, `tokenizer.model`), both
    /// require a config and a `tokenizer.json` to be supplied explicitly.
    pub fn from_bytes(
        weights_bytes: Vec<u8>,
        config_json: &str,
        tokenizer_json: &str,
    ) -> Result<Self, InferenceError> {
        let safetensors = Box::new(SafetensorsFile::from_bytes(weights_bytes)?);
        let config = BertConfig::from_json_str(config_json)?;
        // The tokenizer's own sequence-length cap has no `tokenizer_config.json`
        // to read here (see `tokenizer_from_json_str`'s doc comment), so derive
        // it from the config we already have, mirroring `infer_model_max_seq_len`'s
        // 2048 embedding-workload cap.
        let max_seq_len = config.max_position_embeddings.min(2048);
        let tokenizer = crate::tokenizer::tokenizer_from_json_str(tokenizer_json, max_seq_len)?;

        Self::assemble(config, tokenizer, safetensors)
    }

    /// Shared tail of [`from_directory`](Self::from_directory) and
    /// [`from_bytes`](Self::from_bytes): validate, load weights, and assemble `Self`.
    fn assemble(
        config: BertConfig,
        tokenizer: Box<dyn Tokenizer>,
        safetensors: Box<SafetensorsFile>,
    ) -> Result<Self, InferenceError> {
        if tokenizer.vocab_size() != config.vocab_size {
            warn!(
                tokenizer_vocab_size = tokenizer.vocab_size(),
                model_vocab_size = config.vocab_size,
                "tokenizer and model vocab sizes differ"
            );
        }

        let weights_tmp =
            safetensors.load_bert_weights(config.num_hidden_layers, config.hidden_size)?;
        let fused_qkv: Vec<LayerFusedQkv> = weights_tmp
            .layers
            .iter()
            .map(LayerFusedQkv::build)
            .collect();
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
            pooling: BertPooling::default(),
            fused_qkv,
        })
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

    /// **Stable** (provisional): set the pooling strategy.
    ///
    /// Must be called before any encoding.  The `NativeEmbeddingService` uses this to
    /// route BGE models through CLS pooling and E5/MiniLM through mean pooling.
    pub fn set_pooling(&mut self, pooling: BertPooling) {
        self.pooling = pooling;
    }

    /// **Unstable**: pooling strategy accessor for testing.
    pub fn pooling(&self) -> BertPooling {
        self.pooling
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

        let mut pooled = self.pool(&hidden_states, &input.attention_mask[..seq_len], seq_len);
        l2_normalize(&mut pooled);
        Ok(pooled)
    }

    /// **Stable**: batch-encode entry point; consumed by `lattice-embed`.
    ///
    /// Runs a single fused batched forward pass over all texts, using a packed
    /// (padding-free) row layout (#677): every sequence contributes only its real
    /// tokens, concatenated back to back, tracked by a `cu_seqlens` cumulative-offset
    /// index instead of a shared `batch * seq_len` padded stride. The position-wise
    /// stages (embedding lookup, Q/K/V projection, FFN, output projection) are each
    /// one `matmul_bt`/`layer_norm` call over every real row -- no wasted rows for
    /// padding. Only the O(seq_len^2) attention score/context step loops per
    /// sequence (see [`crate::attention::multi_head_attention_batched`]) are run
    /// serially, one sequence at a time, over disjoint output slices. The
    /// implementation detail may change without API breakage.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() == 1 {
            // Single item: the batched path's flatten/scratch-allocation overhead
            // buys nothing over the plain single-sequence path.
            return Ok(vec![self.encode(texts[0])?]);
        }

        self.encode_batch_packed(texts)
    }

    /// Packed batched-encode pipeline (#677): always runs the fused batched
    /// forward pass over `texts`, regardless of count.
    ///
    /// Split out from [`Self::encode_batch`] so the parity test can exercise this
    /// path directly at batch size 1 too -- `encode_batch` itself shortcuts a
    /// single text to [`Self::encode`], which never touches [`Self::forward_batch`].
    fn encode_batch_packed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        let tokenized = self.tokenizer.tokenize_batch(texts);
        let batch = tokenized.len();
        let hidden_size = self.config.hidden_size;

        // Packed layout: concatenate each sequence's REAL tokens only (no padding),
        // tracked via `cu_seqlens` (cumulative offsets; `cu_seqlens[b]..cu_seqlens[b
        // + 1]` is sequence b's row range). `TokenizedInput::real_length` already
        // records each sequence's true length independently of how the tokenizer
        // padded it for its own batch-local `input_ids`/`token_type_ids`, so packing
        // is a matter of slicing off the real prefix, not re-deriving it. The
        // `max_position_embeddings` clamp mirrors `forward_with_hook`'s single-item
        // truncation of an over-long sequence, applied per sequence rather than
        // uniformly across the whole batch.
        let mut cu_seqlens = Vec::with_capacity(batch + 1);
        cu_seqlens.push(0usize);
        let mut input_ids = Vec::new();
        let mut token_type_ids = Vec::new();
        for input in &tokenized {
            let real = input.real_length.min(self.config.max_position_embeddings);
            input_ids.extend_from_slice(&input.input_ids[..real]);
            token_type_ids.extend_from_slice(&input.token_type_ids[..real]);
            let end = cu_seqlens.last().copied().unwrap_or(0) + real;
            cu_seqlens.push(end);
        }

        let hidden = self.forward_batch(&input_ids, &token_type_ids, &cu_seqlens, &NoopLoraHook);

        // Every packed row is real, so pooling needs an all-ones mask rather than
        // the padding-aware mask the old padded path built. One shared buffer sized
        // to the batch's longest real sequence, sliced per item below, instead of
        // allocating a fresh mask per sequence.
        let max_real_len = (0..batch)
            .map(|b| cu_seqlens[b + 1] - cu_seqlens[b])
            .max()
            .unwrap_or(0);
        let ones_mask = vec![1u32; max_real_len];

        let mut outputs = Vec::with_capacity(batch);
        for b in 0..batch {
            let start = cu_seqlens[b];
            let end = cu_seqlens[b + 1];
            let real_len = end - start;
            let hidden_b = &hidden[start * hidden_size..end * hidden_size];
            let mask_b = &ones_mask[..real_len];
            let mut pooled = self.pool(hidden_b, mask_b, real_len);
            l2_normalize(&mut pooled);
            outputs.push(pooled);
        }

        Ok(outputs)
    }

    /// Apply the configured pooling strategy to `hidden_states`.
    ///
    /// Both `encode` and `encode_batch` delegate here so the pooling branch
    /// is in one place.  L2 normalization is applied by the caller.
    fn pool(&self, hidden_states: &[f32], attention_mask: &[u32], seq_len: usize) -> Vec<f32> {
        match self.pooling {
            BertPooling::Mean => mean_pool(
                hidden_states,
                attention_mask,
                seq_len,
                self.config.hidden_size,
            ),
            BertPooling::CLS => cls_pool(hidden_states, seq_len, self.config.hidden_size),
        }
    }

    /// Forward pass for a pre-tokenized input; used by `CrossEncoderModel`.
    pub(crate) fn forward_tokenized(
        &self,
        input: &crate::tokenizer::TokenizedInput,
        buffers: &mut AttentionBuffers,
    ) -> Vec<f32> {
        self.forward_tokenized_with_hook(input, buffers, &NoopLoraHook)
    }

    /// Hook-aware forward pass for a pre-tokenized input; used by `CrossEncoderModel`.
    pub(crate) fn forward_tokenized_with_hook(
        &self,
        input: &crate::tokenizer::TokenizedInput,
        buffers: &mut AttentionBuffers,
        lora: &dyn LoraHook,
    ) -> Vec<f32> {
        let seq_len = input.real_length;
        // `lora` here is caller-supplied and may be a real adapter (see
        // `CrossEncoderModel::score_with_hook`). The FFN fast path (#675) is
        // still taken here: `LoraHook::apply` only ever adds a delta that
        // depends on the pre-FFN hidden state, not on the bias-add's output,
        // so doing the adapter add-in before the fused bias+GELU pass gives
        // the same value as adding it in between, up to floating-point
        // summation-order rounding (f32 addition is not associative, so this
        // is a reassociation, not an exact-value guarantee). See
        // `forward_with_hook`'s doc comment for why that reassociation is
        // accepted here.
        self.forward_with_hook(
            &input.input_ids[..seq_len],
            &input.attention_mask[..seq_len],
            &input.token_type_ids[..seq_len],
            seq_len,
            buffers,
            lora,
        )
    }

    /// Internal full transformer forward pass (no-op hook).
    fn forward(
        &self,
        input_ids: &[u32],
        attention_mask: &[u32],
        token_type_ids: &[u32],
        seq_len: usize,
        buffers: &mut AttentionBuffers,
    ) -> Vec<f32> {
        self.forward_with_hook(
            input_ids,
            attention_mask,
            token_type_ids,
            seq_len,
            buffers,
            &NoopLoraHook,
        )
    }

    /// Hook-aware internal full transformer forward pass.
    ///
    /// The FFN intermediate stage always uses the fused `add_bias_gelu`
    /// kernel (#675) instead of separate `add_bias`/`gelu` passes, including
    /// when a real `lora` adapter is active: `LoraHook::apply` only adds
    /// `scale * B @ (A @ x)` into the buffer, a delta that depends on the
    /// pre-FFN hidden state (`x`) rather than on the bias-add's output, so it
    /// can be applied before the fused bias+GELU pass instead of between the
    /// bias-add and GELU as before.
    ///
    /// This reorder changes floating-point summation order. Adding the bias
    /// then the delta, versus adding the delta then the bias, is not
    /// guaranteed to produce a bit-identical result, because f32 addition is
    /// not associative (this can matter under extreme cancellation, e.g.
    /// terms differing by many orders of magnitude). We accept this as a
    /// reassociation, not a bit-exact equivalence, for these reasons:
    ///
    /// - Real BERT FFN pre-activations are O(10-100) with O(1) biases and
    ///   small LoRA deltas, well outside any cancellation regime.
    /// - There is no exact-output consumer on this adapter path.
    /// - The `embed_parity_vs_hf` test empirically confirms parity against
    ///   the reference implementation (cosine similarity at least 0.9998) on
    ///   every shipping BERT-family model.
    /// - lattice's SIMD kernels already reassociate f32 sums routinely
    ///   elsewhere in this crate.
    fn forward_with_hook(
        &self,
        input_ids: &[u32],
        attention_mask: &[u32],
        token_type_ids: &[u32],
        seq_len: usize,
        buffers: &mut AttentionBuffers,
        lora: &dyn LoraHook,
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
            let fused = &self.fused_qkv[layer_idx];

            multi_head_attention_in_place(
                &hidden,
                layer,
                &fused.weight,
                &fused.bias,
                attention_mask,
                seq_len,
                hidden_size,
                self.config.num_attention_heads,
                self.config.head_dim(),
                buffers,
                lora,
                layer_idx,
            );

            {
                let temp = &buffers.temp[..used_hidden];
                residual_add_layer_norm(
                    &mut hidden,
                    temp,
                    layer.attn_layer_norm_weight.data,
                    layer.attn_layer_norm_bias.data,
                    hidden_size,
                    self.config.layer_norm_eps,
                );
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
                // #675: adapter add-in before the fused bias+GELU pass (see
                // `forward_with_hook`'s doc comment for the accepted
                // f32-reassociation argument versus the previous
                // add_bias-then-adapter-then-gelu ordering). Extracted into
                // `apply_ffn_intermediate_lora_and_activation` so this
                // ordering can be unit-tested directly, without a model
                // (see that function's doc comment).
                apply_ffn_intermediate_lora_and_activation(
                    lora,
                    layer_idx,
                    &hidden,
                    ffn_intermediate,
                    layer.ffn_intermediate_bias.data,
                    hidden_size,
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
                apply_lora_rows(
                    lora,
                    layer_idx,
                    "ffn_output",
                    ffn_intermediate,
                    temp,
                    intermediate_size,
                    hidden_size,
                );
                let temp = &buffers.temp[..used_hidden];
                residual_add_layer_norm(
                    &mut hidden,
                    temp,
                    layer.ffn_layer_norm_weight.data,
                    layer.ffn_layer_norm_bias.data,
                    hidden_size,
                    self.config.layer_norm_eps,
                );
            }
        }

        hidden
    }

    /// Fused batched forward pass over a packed (padding-free) row layout (#677).
    ///
    /// `input_ids`/`token_type_ids` are the concatenation of every sequence's REAL
    /// tokens, with no padding rows anywhere; `cu_seqlens` (length `batch + 1`,
    /// `cu_seqlens[0] == 0`) gives the cumulative row offsets, so sequence `b`
    /// occupies rows `cu_seqlens[b]..cu_seqlens[b + 1]`. Returns a flat
    /// `[total, hidden_size]` hidden-state tensor (`total == cu_seqlens[batch]`);
    /// callers slice out each sequence's `cu_seqlens[b]..cu_seqlens[b + 1]` region
    /// and pool it individually.
    ///
    /// This is a new function, not a modification of [`Self::forward`]/
    /// [`Self::forward_with_hook`] -- `encode()` and `forward_tokenized` (used by
    /// `CrossEncoderModel`) are untouched and keep their existing single-sequence
    /// behavior and performance.
    fn forward_batch(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
        cu_seqlens: &[usize],
        lora: &dyn LoraHook,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim();

        debug_assert!(!cu_seqlens.is_empty());
        debug_assert_eq!(cu_seqlens[0], 0);
        let batch = cu_seqlens.len() - 1;
        let total = cu_seqlens[batch];

        debug_assert_eq!(input_ids.len(), total);
        debug_assert_eq!(token_type_ids.len(), total);

        let used_hidden = total * hidden_size;
        let used_intermediate = total * intermediate_size;

        let mut hidden = vec![0.0f32; used_hidden];

        // Embedding lookup: position id resets to `i` (the in-sequence index) for
        // every sequence `b`, derived from `cu_seqlens` rather than a uniform
        // stride -- the one correctness-critical detail carried over from the
        // pre-#677 padded path's flattening. Callers (`encode_batch_packed`) are
        // responsible for capping each sequence's real length at
        // `max_position_embeddings` before building `cu_seqlens`/`input_ids`, the
        // same truncation `forward_with_hook` applies per single sequence; this
        // function trusts `cu_seqlens` to already reflect that.
        for b in 0..batch {
            let start = cu_seqlens[b];
            let end = cu_seqlens[b + 1];
            for (i, row) in (start..end).enumerate() {
                let tok_id = input_ids[row] as usize;
                let typ_id = token_type_ids[row] as usize;
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
                let out_row = &mut hidden[row * hidden_size..(row + 1) * hidden_size];

                for d in 0..hidden_size {
                    out_row[d] = tok_row[d] + pos_row[d] + typ_row[d];
                }
            }
        }

        // Embedding LayerNorm: one call over every row in the batch -- unchanged,
        // row-count-generic kernel (crate::forward::cpu::layer_norm).
        layer_norm(
            &mut hidden,
            self.weights.embedding_layer_norm_weight.data,
            self.weights.embedding_layer_norm_bias.data,
            hidden_size,
            self.config.layer_norm_eps,
        );

        let mut q = vec![0.0f32; used_hidden];
        let mut k = vec![0.0f32; used_hidden];
        let mut v = vec![0.0f32; used_hidden];
        let mut qkv = vec![0.0f32; used_hidden * 3];
        let mut concat = vec![0.0f32; used_hidden];
        let mut attn_out = vec![0.0f32; used_hidden];
        let mut ffn_intermediate = vec![0.0f32; used_intermediate];
        let mut temp = vec![0.0f32; used_hidden];

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];
            let fused = &self.fused_qkv[layer_idx];

            multi_head_attention_batched(
                &hidden,
                layer,
                &fused.weight,
                &fused.bias,
                cu_seqlens,
                hidden_size,
                num_heads,
                head_dim,
                &mut q,
                &mut k,
                &mut v,
                &mut qkv,
                &mut concat,
                &mut attn_out,
                lora,
                layer_idx,
            );

            residual_add_layer_norm(
                &mut hidden,
                &attn_out,
                layer.attn_layer_norm_weight.data,
                layer.attn_layer_norm_bias.data,
                hidden_size,
                self.config.layer_norm_eps,
            );

            matmul_bt(
                &hidden,
                layer.ffn_intermediate_weight.data,
                &mut ffn_intermediate,
                total,
                hidden_size,
                intermediate_size,
            );
            // #675: route through the shared `apply_ffn_intermediate_lora_and_activation`
            // helper so this batched path and `forward_with_hook` cannot drift.
            // `encode_batch` only ever supplies `NoopLoraHook`, so the adapter
            // add-in is a no-op here and this stays a direct swap for the old
            // add_bias + gelu pair.
            apply_ffn_intermediate_lora_and_activation(
                lora,
                layer_idx,
                &hidden,
                &mut ffn_intermediate,
                layer.ffn_intermediate_bias.data,
                hidden_size,
                intermediate_size,
            );

            matmul_bt(
                &ffn_intermediate,
                layer.ffn_output_weight.data,
                &mut temp,
                total,
                intermediate_size,
                hidden_size,
            );
            add_bias(&mut temp, layer.ffn_output_bias.data, hidden_size);
            apply_lora_rows(
                lora,
                layer_idx,
                "ffn_output",
                &ffn_intermediate,
                &mut temp,
                intermediate_size,
                hidden_size,
            );
            residual_add_layer_norm(
                &mut hidden,
                &temp,
                layer.ffn_layer_norm_weight.data,
                layer.ffn_layer_norm_bias.data,
                hidden_size,
                self.config.layer_norm_eps,
            );
        }

        hidden
    }

    /// Pre-#677 padded batched-encode pipeline, preserved as a test-only reference.
    ///
    /// Rebuilds the `batch * seq_len` padded tensor `encode_batch` used to build
    /// before the packed/varlen rewrite, and runs it through
    /// [`forward_batch_padded_reference`](Self::forward_batch_padded_reference).
    /// Exists solely so the parity test has an independently-computed ground truth
    /// that does not share code with the packed production path.
    #[cfg(test)]
    fn encode_batch_padded_reference(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        let tokenized = self.tokenizer.tokenize_batch(texts);
        let batch = tokenized.len();
        let hidden_size = self.config.hidden_size;

        let seq_len = tokenized
            .iter()
            .map(|input| input.input_ids.len())
            .max()
            .unwrap_or(2)
            .max(1);

        let rows = batch * seq_len;
        let mut input_ids = Vec::with_capacity(rows);
        let mut attention_mask = Vec::with_capacity(rows);
        let mut token_type_ids = Vec::with_capacity(rows);
        for input in &tokenized {
            let real = input.input_ids.len();
            input_ids.extend_from_slice(&input.input_ids);
            input_ids.extend(std::iter::repeat_n(0u32, seq_len - real));
            attention_mask.extend_from_slice(&input.attention_mask);
            attention_mask.extend(std::iter::repeat_n(0u32, seq_len - real));
            token_type_ids.extend_from_slice(&input.token_type_ids);
            token_type_ids.extend(std::iter::repeat_n(0u32, seq_len - real));
        }

        let hidden = self.forward_batch_padded_reference(
            &input_ids,
            &attention_mask,
            &token_type_ids,
            batch,
            seq_len,
            &NoopLoraHook,
        );

        let mut outputs = Vec::with_capacity(batch);
        for b in 0..batch {
            let off = b * seq_len * hidden_size;
            let hidden_b = &hidden[off..off + seq_len * hidden_size];
            let mask_b = &attention_mask[b * seq_len..(b + 1) * seq_len];
            let mut pooled = self.pool(hidden_b, mask_b, seq_len);
            l2_normalize(&mut pooled);
            outputs.push(pooled);
        }

        outputs
    }

    /// Pre-#677 padded batched forward pass, preserved as a test-only reference.
    ///
    /// This is [`Self::forward_batch`] as it existed before the packed/varlen
    /// rewrite: `input_ids`/`attention_mask`/`token_type_ids` are flat
    /// `batch * seq_len` arrays (row `b * seq_len + i` is token `i` of sequence
    /// `b`), every sequence padded to the same `seq_len`. It exists solely to give
    /// the parity test an independent ground truth for the packed production path,
    /// computed through a genuinely different (not just relabeled) code path.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_padded_reference(
        &self,
        input_ids: &[u32],
        attention_mask: &[u32],
        token_type_ids: &[u32],
        batch: usize,
        padded_seq_len: usize,
        lora: &dyn LoraHook,
    ) -> Vec<f32> {
        use crate::attention::multi_head_attention_batched_padded_reference;

        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim();

        debug_assert_eq!(input_ids.len(), batch * padded_seq_len);
        debug_assert_eq!(attention_mask.len(), batch * padded_seq_len);
        debug_assert_eq!(token_type_ids.len(), batch * padded_seq_len);

        let seq_len = padded_seq_len.min(self.config.max_position_embeddings);
        let rows = batch * seq_len;

        let used_hidden = rows * hidden_size;
        let used_intermediate = rows * intermediate_size;

        let mut hidden = vec![0.0f32; used_hidden];

        for b in 0..batch {
            let src_offset = b * padded_seq_len;
            let dst_offset = b * seq_len;
            for i in 0..seq_len {
                let src_row = src_offset + i;
                let dst_row = dst_offset + i;
                let tok_id = input_ids[src_row] as usize;
                let typ_id = token_type_ids[src_row] as usize;
                let pos_id = i;

                let tok_row = &self.weights.word_embeddings.data
                    [tok_id * hidden_size..(tok_id + 1) * hidden_size];
                let pos_row = &self.weights.position_embeddings.data
                    [pos_id * hidden_size..(pos_id + 1) * hidden_size];
                let typ_row = &self.weights.token_type_embeddings.data
                    [typ_id * hidden_size..(typ_id + 1) * hidden_size];
                let out_row = &mut hidden[dst_row * hidden_size..(dst_row + 1) * hidden_size];

                for d in 0..hidden_size {
                    out_row[d] = tok_row[d] + pos_row[d] + typ_row[d];
                }
            }
        }

        layer_norm(
            &mut hidden,
            self.weights.embedding_layer_norm_weight.data,
            self.weights.embedding_layer_norm_bias.data,
            hidden_size,
            self.config.layer_norm_eps,
        );

        let attention_mask_clamped: Vec<u32>;
        let attention_mask: &[u32] = if seq_len == padded_seq_len {
            attention_mask
        } else {
            attention_mask_clamped = (0..batch)
                .flat_map(|b| {
                    let off = b * padded_seq_len;
                    attention_mask[off..off + seq_len].iter().copied()
                })
                .collect();
            &attention_mask_clamped
        };

        let mut q = vec![0.0f32; used_hidden];
        let mut k = vec![0.0f32; used_hidden];
        let mut v = vec![0.0f32; used_hidden];
        let mut qkv = vec![0.0f32; used_hidden * 3];
        let mut concat = vec![0.0f32; used_hidden];
        let mut attn_out = vec![0.0f32; used_hidden];
        let mut ffn_intermediate = vec![0.0f32; used_intermediate];
        let mut temp = vec![0.0f32; used_hidden];

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];
            let fused = &self.fused_qkv[layer_idx];

            multi_head_attention_batched_padded_reference(
                &hidden,
                layer,
                &fused.weight,
                &fused.bias,
                attention_mask,
                batch,
                seq_len,
                hidden_size,
                num_heads,
                head_dim,
                &mut q,
                &mut k,
                &mut v,
                &mut qkv,
                &mut concat,
                &mut attn_out,
                lora,
                layer_idx,
            );

            residual_add_layer_norm(
                &mut hidden,
                &attn_out,
                layer.attn_layer_norm_weight.data,
                layer.attn_layer_norm_bias.data,
                hidden_size,
                self.config.layer_norm_eps,
            );

            matmul_bt(
                &hidden,
                layer.ffn_intermediate_weight.data,
                &mut ffn_intermediate,
                rows,
                hidden_size,
                intermediate_size,
            );
            apply_ffn_intermediate_lora_and_activation(
                lora,
                layer_idx,
                &hidden,
                &mut ffn_intermediate,
                layer.ffn_intermediate_bias.data,
                hidden_size,
                intermediate_size,
            );

            matmul_bt(
                &ffn_intermediate,
                layer.ffn_output_weight.data,
                &mut temp,
                rows,
                intermediate_size,
                hidden_size,
            );
            add_bias(&mut temp, layer.ffn_output_bias.data, hidden_size);
            apply_lora_rows(
                lora,
                layer_idx,
                "ffn_output",
                &ffn_intermediate,
                &mut temp,
                intermediate_size,
                hidden_size,
            );
            residual_add_layer_norm(
                &mut hidden,
                &temp,
                layer.ffn_layer_norm_weight.data,
                layer.ffn_layer_norm_bias.data,
                hidden_size,
                self.config.layer_norm_eps,
            );
        }

        hidden
    }
}

/// Apply the `"ffn_intermediate"` LoRA adapter delta, then the fused
/// bias+GELU pass, in that order (#675).
///
/// Extracted out of `BertModel::forward_with_hook` as a free function so the
/// ordering invariant it encodes (the adapter delta must land before GELU,
/// not after) can be unit-tested directly against small synthetic buffers,
/// without needing a full model
/// (`test_ffn_intermediate_stage_lora_delta_lands_before_gelu`). The
/// model-backed `test_ffn_intermediate_lora_delta_lands_before_gelu` test
/// covers the same invariant end to end through a real `BertModel`, gated
/// behind `LATTICE_INFERENCE_MODEL_DIR` since it needs a downloaded model.
fn apply_ffn_intermediate_lora_and_activation(
    lora: &dyn LoraHook,
    layer_idx: usize,
    hidden: &[f32],
    ffn_intermediate: &mut [f32],
    bias: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
) {
    apply_lora_rows(
        lora,
        layer_idx,
        "ffn_intermediate",
        hidden,
        ffn_intermediate,
        hidden_size,
        intermediate_size,
    );
    add_bias_gelu(ffn_intermediate, bias, intermediate_size);
}

fn parse_config_json_if_present(path: &Path) -> Result<Option<BertConfig>, InferenceError> {
    if !path.exists() {
        return Ok(None);
    }

    let text = fs::read_to_string(path)?;
    parse_config_json_str(&text).map(Some)
}

/// Parse `config.json` text into a `BertConfig`. Shared by the file-based
/// `parse_config_json_if_present` and the public `BertConfig::from_json_str`
/// (used by [`BertModel::from_bytes`]).
fn parse_config_json_str(text: &str) -> Result<BertConfig, InferenceError> {
    let get_usize = |key: &str| {
        extract_json_scalar(text, key)
            .ok_or_else(|| InferenceError::Inference(format!("config.json missing key {key}")))?
            .parse::<usize>()
            .map_err(|e| InferenceError::Inference(format!("invalid usize for {key}: {e}")))
    };
    let get_f32 = |key: &str| {
        extract_json_scalar(text, key)
            .ok_or_else(|| InferenceError::Inference(format!("config.json missing key {key}")))?
            .parse::<f32>()
            .map_err(|e| InferenceError::Inference(format!("invalid f32 for {key}: {e}")))
    };

    Ok(BertConfig {
        vocab_size: get_usize("vocab_size")?,
        hidden_size: get_usize("hidden_size")?,
        num_hidden_layers: get_usize("num_hidden_layers")?,
        num_attention_heads: get_usize("num_attention_heads")?,
        intermediate_size: get_usize("intermediate_size")?,
        max_position_embeddings: get_usize("max_position_embeddings")?,
        type_vocab_size: get_usize("type_vocab_size")?,
        layer_norm_eps: get_f32("layer_norm_eps")?,
    })
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
        if let Some(rest) = name.strip_prefix("encoder.layer.")
            && let Some(index_str) = rest.split('.').next()
            && let Ok(index) = index_str.parse::<usize>()
        {
            max_layer = Some(max_layer.map_or(index, |curr| curr.max(index)));
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
    use crate::forward::cpu::gelu;
    use crate::pool::{cls_pool, l2_normalize, mean_pool};
    use approx::assert_relative_eq;
    use std::sync::Mutex;

    #[test]
    #[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
    fn test_encode_output_shape_and_l2_norm() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };

        let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();
        let embedding = model.encode("hello world").unwrap();
        assert_eq!(embedding.len(), model.dimensions());
        let norm = (embedding.iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-4);
    }

    /// Guards `BertModel::from_bytes` against real model files: a directory
    /// with `model.safetensors` + `config.json` + `tokenizer.json` (unlike
    /// `LATTICE_INFERENCE_MODEL_DIR` above, `from_bytes` needs both JSON files
    /// explicitly since it has no config-from-shape inference or vocab.txt
    /// fallback). Asserts the from-bytes embedding matches `from_directory`'s
    /// output for the same model bit-for-bit within float tolerance.
    #[test]
    #[ignore = "requires LATTICE_INFERENCE_BYTES_MODEL_DIR"]
    fn test_from_bytes_matches_from_directory() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_BYTES_MODEL_DIR") else {
            return;
        };
        let dir = Path::new(&model_dir);

        let weights_bytes = fs::read(dir.join("model.safetensors")).unwrap();
        let config_json = fs::read_to_string(dir.join("config.json")).unwrap();
        let tokenizer_json = fs::read_to_string(dir.join("tokenizer.json")).unwrap();

        let mut model = BertModel::from_bytes(weights_bytes, &config_json, &tokenizer_json)
            .expect("from_bytes should load a real BGE-family model directory");
        // BGE v1.5 uses CLS pooling per its model card; from_bytes has no
        // per-model pooling table (that lives in lattice-embed's
        // `EmbeddingModel::bert_pooling`), so the test sets it explicitly.
        model.set_pooling(BertPooling::CLS);

        let embedding = model.encode("hello world").unwrap();
        assert_eq!(embedding.len(), model.dimensions());
        let norm = (embedding.iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-4);

        let mut reference = BertModel::from_directory(dir)
            .expect("from_directory should also load the same model directory");
        reference.set_pooling(BertPooling::CLS);
        let reference_embedding = reference.encode("hello world").unwrap();

        assert_eq!(embedding.len(), reference_embedding.len());
        for (a, b) in embedding.iter().zip(reference_embedding.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "from_bytes and from_directory embeddings diverge: {a} vs {b}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Fused batched forward parity (mutation-sensitive)
    //
    // Guards the `encode_batch` rewrite: the fused batched forward path
    // (`forward_batch` + `multi_head_attention_batched`) must produce the same
    // per-text embeddings as looping `encode()` one text at a time. This catches
    // the two correctness traps a naive batch-flattening introduces: cross-sequence
    // attention leakage through the padding mask, and position-id drift for any
    // sequence after the first (see forward_batch's embedding-lookup comment).
    //
    // Reverting `encode_batch` to loop per-item (or breaking the position-id /
    // mask plumbing in `forward_batch`/`multi_head_attention_batched`) must make
    // these tests fail, not just the CI bench numbers regress.
    // -------------------------------------------------------------------------

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let nb = (b.iter().map(|x| x * x).sum::<f32>()).sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    #[test]
    #[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
    fn test_encode_batch_matches_per_item_encode_mixed_lengths() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };
        let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();

        // Mixed lengths (short to long) force real padding across the batch.
        let short = "hi";
        let medium = "the quick brown fox jumps over the lazy dog";
        let long_text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod \
            tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis \
            nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis \
            aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat \
            nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui \
            officia deserunt mollit anim id est laborum";
        let texts: Vec<&str> = vec![
            short,
            medium,
            long_text,
            "another sentence",
            short,
            medium,
            "one more",
            long_text,
        ];

        let batched = model.encode_batch(&texts).unwrap();
        assert_eq!(batched.len(), texts.len());

        for (i, text) in texts.iter().enumerate() {
            let solo = model.encode(text).unwrap();
            let cos = cosine(&batched[i], &solo);
            let max_abs_diff = batched[i]
                .iter()
                .zip(solo.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                cos >= 0.99999,
                "text[{i}] ({text:?}...): batched vs solo cosine {cos} < 0.99999"
            );
            assert!(
                max_abs_diff <= 1e-4,
                "text[{i}] ({text:?}...): max-abs-diff {max_abs_diff} > 1e-4"
            );
        }
    }

    #[test]
    #[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
    fn test_encode_batch_boundary_max_and_min_length() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };
        let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();

        // One sequence at exactly the model's max position length, one at length 1
        // (single real token beyond [CLS]/[SEP] framing) -- the widest padding gap
        // this path can produce.
        let max_len_text = "word ".repeat(model.config().max_position_embeddings);
        let one_token_text = "a";
        let texts: Vec<&str> = vec![&max_len_text, one_token_text];

        let batched = model.encode_batch(&texts).unwrap();
        assert_eq!(batched.len(), 2);

        for (i, text) in texts.iter().enumerate() {
            let solo = model.encode(text).unwrap();
            let cos = cosine(&batched[i], &solo);
            let max_abs_diff = batched[i]
                .iter()
                .zip(solo.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(cos >= 0.99999, "boundary text[{i}]: cosine {cos} < 0.99999");
            assert!(
                max_abs_diff <= 1e-4,
                "boundary text[{i}]: max-abs-diff {max_abs_diff} > 1e-4"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Packed vs padded batched-encode parity (#677, mutation-sensitive)
    //
    // Guards the packed/varlen batched-encode rewrite: `encode_batch_packed`'s
    // packed `cu_seqlens`-indexed forward pass must match `encode_batch_padded_reference`'s
    // independently computed pre-#677 padded forward pass (preserved verbatim as a
    // `#[cfg(test)]`-only reference; see `forward_batch_padded_reference` and
    // `crate::attention::multi_head_attention_batched_padded_reference`).
    //
    // Both sides call `encode_batch_packed`/`encode_batch_padded_reference`
    // directly, bypassing the public `encode_batch`'s single-item shortcut
    // (which calls `encode()` and never touches either batched kernel), so a
    // batch of size 1 genuinely exercises the packed kernel's degenerate
    // one-segment `cu_seqlens = [0, total]` case instead of trivially matching
    // by construction.
    //
    // Parametrized over four shapes; the max-variance shape ([1, 2, 128]-ish
    // real tokens) is the one most likely to expose an off-by-one in the
    // per-sequence `cu_seqlens[b]`/`cu_seqlens[b + 1]` window that
    // `multi_head_attention_batched` indexes with -- a wrong offset there
    // shows up as a large divergence on the longest sequence, not a rounding
    // difference, since it would pull in (or drop) real rows from a
    // neighboring sequence's attention window.
    //
    // NOT literal bit-for-bit equality. Measured directly on this model: any
    // sequence that is the batch's longest (so the padded reference never
    // actually padded it) comes back exactly bit-identical (`max_abs_diff ==
    // 0.0`), but a sequence that WAS padded in the reference lands within
    // ~2e-7 absolute of the packed output, not `0`. Isolated with a standalone
    // harness varying only attention's total row count (real rows identical,
    // extra masked padding rows appended): `multi_head_attention`'s output for
    // the same real rows shifts by 1 ULP once the masked-out row count
    // changes, even though `exp(-inf) == 0` zeroes every padding contribution
    // algebraically. Apple's Accelerate `cblas_sgemm` (the `matmul_bt`
    // backend on this platform) and the SIMD softmax reduction are not
    // proven shape-invariant at the bit level for a fixed set of real
    // rows plus a varying number of masked rows; only the row COUNT changes,
    // but that changes GEMM tiling/blocking and reduction order, which
    // reassociates the sum. This is the same class of accepted f32
    // reassociation already documented elsewhere in this crate (e.g. the
    // LoRA adapter reorder in `forward_with_hook`), just measured here to a
    // much tighter bound than that convention's cosine/max-abs-diff check
    // because this parity claim is far stronger (identical kernel, only the
    // row count and per-sequence real length differ). `TOLERANCE` below is
    // ~45x the largest diff measured across all four shapes on this model,
    // and still 10x tighter than the crate's existing batched-vs-solo parity
    // tolerance (1e-4) -- a genuine `cu_seqlens` indexing bug pulls in a
    // neighboring sequence's rows and blows through it by several orders of
    // magnitude, not a rounding-scale amount; that class stays caught.
    // -------------------------------------------------------------------------

    #[test]
    #[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
    fn encode_batch_packed_matches_padded() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };
        let model = BertModel::from_directory(Path::new(&model_dir)).unwrap();

        let short = "hi";
        let medium = "the quick brown fox jumps over the lazy dog";
        let long_text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod \
            tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis \
            nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis \
            aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat \
            nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui \
            officia deserunt mollit anim id est laborum";
        let uniform_64 = "word ".repeat(64);
        let one_word = "a";
        let two_words = "hi there";
        let long_128 = "word ".repeat(128);

        let shapes: Vec<(&str, Vec<&str>)> = vec![
            ("single_row", vec![medium]),
            (
                "uniform_three_same_length",
                vec![
                    uniform_64.as_str(),
                    uniform_64.as_str(),
                    uniform_64.as_str(),
                ],
            ),
            (
                "max_variance_1_2_128",
                vec![one_word, two_words, long_128.as_str()],
            ),
            ("mixed_5_30_100", vec![short, medium, long_text]),
        ];

        // ~45x the largest max-abs-diff measured across all four shapes on this
        // model (2.2e-7), and 10x tighter than the crate's existing batched-vs-solo
        // parity tolerance (1e-4, see `test_encode_batch_matches_per_item_encode_mixed_lengths`).
        const TOLERANCE: f32 = 1e-5;

        for (name, texts) in shapes {
            let packed = model.encode_batch_packed(&texts).unwrap();
            let padded = model.encode_batch_padded_reference(&texts);

            assert_eq!(packed.len(), padded.len(), "{name}: output count mismatch");
            for (i, (p, q)) in packed.iter().zip(padded.iter()).enumerate() {
                assert_eq!(
                    p.len(),
                    q.len(),
                    "{name}: text[{i}] embedding length mismatch"
                );
                let max_abs_diff = p
                    .iter()
                    .zip(q.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs_diff <= TOLERANCE,
                    "{name}: text[{i}] packed vs padded max_abs_diff {max_abs_diff} exceeds {TOLERANCE}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // FFN-intermediate LoRA-adapter reorder correctness (#675, mutation-sensitive)
    //
    // `forward_with_hook` calls `apply_ffn_intermediate_lora_and_activation`,
    // which runs `lora.apply("ffn_intermediate", ...)` before the fused
    // `add_bias_gelu` kernel, on the argument that the bias-add and the
    // adapter delta are both additions into the same buffer ahead of a single
    // GELU, so their relative order does not matter as long as both land
    // before GELU (see that function's and `forward_with_hook`'s doc comments
    // for the accepted-reassociation argument).
    //
    // Two tests cover this, both of which must fail if the ordering regresses
    // to add_bias_gelu-then-adapter (delta landing after GELU instead of
    // before):
    //   - `test_ffn_intermediate_stage_lora_delta_lands_before_gelu` runs by
    //     default (no `#[ignore]`): it calls
    //     `apply_ffn_intermediate_lora_and_activation` directly against small
    //     synthetic buffers, so it needs no downloaded model and always runs
    //     in CI.
    //   - `test_ffn_intermediate_lora_delta_lands_before_gelu` is the
    //     end-to-end version through a real `BertModel` and a real adapter
    //     hook reachable the way `CrossEncoderModel::score_with_hook` reaches
    //     it; it is `#[ignore]`d and gated on `LATTICE_INFERENCE_MODEL_DIR`
    //     because building a real BERT-family model in-process needs a
    //     downloaded checkpoint (safetensors weights, config, and tokenizer),
    //     which this crate does not synthesize for tests.
    // -------------------------------------------------------------------------

    /// Default (non-`#[ignore]`) unit test: exercises
    /// `apply_ffn_intermediate_lora_and_activation` directly against small,
    /// fixed synthetic buffers, with no model required, so this runs in CI
    /// by default.
    #[test]
    fn test_ffn_intermediate_stage_lora_delta_lands_before_gelu() {
        // Small, fixed sizes and values -- deterministic, no RNG dependency.
        let seq_len = 2;
        let hidden_size = 3;
        let intermediate_size = 4;
        let used_hidden = seq_len * hidden_size;
        let used_intermediate = seq_len * intermediate_size;

        let hidden: Vec<f32> = (0..used_hidden).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let weight: Vec<f32> = (0..intermediate_size * hidden_size)
            .map(|i| 0.05 * (i as f32 + 1.0))
            .collect();
        let bias: Vec<f32> = (0..intermediate_size)
            .map(|i| 0.01 * (i as f32 + 1.0))
            .collect();
        let delta = 0.05_f32;

        let mut raw = vec![0.0f32; used_intermediate];
        matmul_bt(
            &hidden,
            &weight,
            &mut raw,
            seq_len,
            hidden_size,
            intermediate_size,
        );

        // Production composition under test: the same function
        // `forward_with_hook` calls.
        let hook = CapturingDeltaHook {
            target_module: "ffn_intermediate",
            delta,
            captured_x: Mutex::new(Vec::new()),
        };
        let mut produced = raw.clone();
        apply_ffn_intermediate_lora_and_activation(
            &hook,
            0,
            &hidden,
            &mut produced,
            &bias,
            hidden_size,
            intermediate_size,
        );

        // (a) The adapter must actually reach the result.
        let mut without_delta = raw.clone();
        add_bias(&mut without_delta, &bias, intermediate_size);
        gelu(&mut without_delta);
        let max_abs_diff_vs_no_delta = produced
            .iter()
            .zip(without_delta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff_vs_no_delta > 1e-3,
            "an active ffn_intermediate adapter must move the output; \
             max-abs-diff vs no-delta was only {max_abs_diff_vs_no_delta}"
        );

        // (b) Independent reference that hardcodes "delta lands before gelu"
        // via separate, unfused matmul_bt/add_bias/gelu calls.
        let mut reference = raw;
        add_bias(&mut reference, &bias, intermediate_size);
        for v in reference.iter_mut() {
            *v += delta;
        }
        gelu(&mut reference); // separate, unfused gelu -- independent of add_bias_gelu

        let max_abs_diff_vs_reference = produced
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff_vs_reference < 1e-5,
            "apply_ffn_intermediate_lora_and_activation does not match the \
             delta-lands-before-gelu reference: max-abs-diff {max_abs_diff_vs_reference}"
        );
    }

    /// `LoraHook` that captures the FFN block's input activation (`x`) for one
    /// target module and adds a constant delta to that projection's output.
    ///
    /// Capturing `x` rather than `output` keeps the reference computation in
    /// the test below independent of where the hook actually fires relative
    /// to bias/GELU: `x` is produced before the FFN block starts, so its
    /// value does not depend on the ordering under test.
    struct CapturingDeltaHook {
        target_module: &'static str,
        delta: f32,
        captured_x: Mutex<Vec<f32>>,
    }

    impl LoraHook for CapturingDeltaHook {
        fn apply(&self, _layer_idx: usize, module: &str, x: &[f32], output: &mut [f32]) {
            if module != self.target_module {
                return;
            }
            self.captured_x.lock().unwrap().extend_from_slice(x);
            for o in output.iter_mut() {
                *o += self.delta;
            }
        }
    }

    /// Build a `TokenizedInput` + fresh `AttentionBuffers` for one text, the
    /// same way `BertModel::encode` does internally.
    fn tokenize_and_buffers(
        model: &BertModel,
        text: &str,
    ) -> (crate::tokenizer::TokenizedInput, AttentionBuffers) {
        let input = model.tokenizer.tokenize(text);
        let seq_len = input.real_length;
        let buffers = AttentionBuffers::new(
            seq_len,
            model.config.hidden_size,
            model.config.num_attention_heads,
            model.config.intermediate_size,
        );
        (input, buffers)
    }

    #[test]
    #[ignore = "requires LATTICE_INFERENCE_MODEL_DIR"]
    fn test_ffn_intermediate_lora_delta_lands_before_gelu() {
        let Ok(model_dir) = std::env::var("LATTICE_INFERENCE_MODEL_DIR") else {
            return;
        };
        let mut model = BertModel::from_directory(Path::new(&model_dir)).unwrap();

        // Truncate to a single transformer layer so the adapter's effect does
        // not cascade through further layers: the reference below is then a
        // direct, checkable reconstruction of exactly one FFN block.
        model.weights.layers.truncate(1);
        model.fused_qkv.truncate(1);
        model.config.num_hidden_layers = 1;

        let text = "the quick brown fox jumps over the lazy dog";
        let delta = 0.05_f32;

        // Baseline: no adapter.
        let (input, mut buffers) = tokenize_and_buffers(&model, text);
        let baseline = model.forward_tokenized_with_hook(&input, &mut buffers, &NoopLoraHook);

        // With adapter: a constant delta added to the ffn_intermediate
        // projection, capturing its input activation as it flows through.
        let hook = CapturingDeltaHook {
            target_module: "ffn_intermediate",
            delta,
            captured_x: Mutex::new(Vec::new()),
        };
        let (input2, mut buffers2) = tokenize_and_buffers(&model, text);
        let with_delta = model.forward_tokenized_with_hook(&input2, &mut buffers2, &hook);

        // (a) The adapter must actually reach the result -- otherwise this
        // whole path is dead and the rest of the test proves nothing.
        let max_abs_diff_vs_baseline = with_delta
            .iter()
            .zip(baseline.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff_vs_baseline > 1e-3,
            "an active ffn_intermediate adapter must move the output; \
             max-abs-diff vs no-op baseline was only {max_abs_diff_vs_baseline}"
        );

        // (b) Build a reference that hardcodes "delta lands before GELU"
        // using separate, unfused kernel calls (matmul_bt / add_bias / gelu
        // one at a time), so it does not just replay whatever internal order
        // forward_with_hook happens to use.
        let seq_len = input2.real_length;
        let hidden_size = model.config.hidden_size;
        let intermediate_size = model.config.intermediate_size;
        let used_hidden = seq_len * hidden_size;
        let used_intermediate = seq_len * intermediate_size;
        let layer = &model.weights.layers[0];

        let captured_x = hook.captured_x.lock().unwrap().clone();
        assert_eq!(captured_x.len(), used_hidden);

        let mut raw = vec![0.0f32; used_intermediate];
        matmul_bt(
            &captured_x,
            layer.ffn_intermediate_weight.data,
            &mut raw,
            seq_len,
            hidden_size,
            intermediate_size,
        );
        add_bias(
            &mut raw,
            layer.ffn_intermediate_bias.data,
            intermediate_size,
        );
        for v in raw.iter_mut() {
            *v += delta;
        }
        gelu(&mut raw); // separate, unfused gelu -- independent of add_bias_gelu

        let mut reference = vec![0.0f32; used_hidden];
        matmul_bt(
            &raw,
            layer.ffn_output_weight.data,
            &mut reference,
            seq_len,
            intermediate_size,
            hidden_size,
        );
        add_bias(&mut reference, layer.ffn_output_bias.data, hidden_size);
        for i in 0..used_hidden {
            reference[i] += captured_x[i];
        }
        layer_norm(
            &mut reference,
            layer.ffn_layer_norm_weight.data,
            layer.ffn_layer_norm_bias.data,
            hidden_size,
            model.config.layer_norm_eps,
        );

        assert_eq!(reference.len(), with_delta.len());
        let max_abs_diff_vs_reference = reference
            .iter()
            .zip(with_delta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff_vs_reference < 1e-4,
            "forward_with_hook's ffn_intermediate output does not match the \
             delta-lands-before-gelu reference: max-abs-diff {max_abs_diff_vs_reference}"
        );
    }

    // -------------------------------------------------------------------------
    // Deterministic pooling tests (P1-E3)
    //
    // These tests use fixed hidden-state tensors — no model weights needed.
    // They validate the pooling routing at the kernel level: CLS extracts
    // position 0, mean computes an attention-mask-weighted average, and L2
    // normalisation produces a unit vector in both cases.
    // -------------------------------------------------------------------------

    /// Fixed 2-token, 4-dim hidden-state tensor.
    ///
    /// Token 0 (CLS):  [1.0, 0.0, 0.0, 0.0]
    /// Token 1 (word): [0.0, 1.0, 0.0, 0.0]
    /// Both tokens are real (attention_mask = [1, 1]).
    fn hidden_2x4() -> (Vec<f32>, Vec<u32>) {
        let hidden = vec![
            1.0_f32, 0.0, 0.0, 0.0, // token 0 (CLS)
            0.0_f32, 1.0, 0.0, 0.0, // token 1 (word)
        ];
        let mask = vec![1_u32, 1];
        (hidden, mask)
    }

    /// CLS pooling returns the first-token hidden state ([1,0,0,0]), then L2 normalises.
    ///
    /// The CLS token is already unit-length here, so after L2 it stays [1,0,0,0].
    /// This matches the BGE model-card recipe: `model_output[0][:, 0]` + L2.
    #[test]
    fn test_cls_pool_extracts_first_token_and_l2_unit_norm() {
        let (hidden, _mask) = hidden_2x4();
        let seq_len = 2;
        let hidden_size = 4;

        let mut pooled = cls_pool(&hidden, seq_len, hidden_size);

        // Before L2: should be the CLS row [1,0,0,0].
        assert_eq!(
            pooled,
            vec![1.0, 0.0, 0.0, 0.0],
            "CLS row mismatch before L2"
        );

        l2_normalize(&mut pooled);

        // CLS row is already unit-length → unchanged.
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
        assert_relative_eq!(pooled[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(pooled[1], 0.0, epsilon = 1e-6);
    }

    /// Mean pooling with uniform mask averages all tokens, then L2 normalises.
    ///
    /// With hidden = [[1,0,0,0],[0,1,0,0]] and mask [1,1],
    /// mean = [0.5, 0.5, 0, 0].  After L2: [1/√2, 1/√2, 0, 0] ≈ [0.7071, 0.7071, 0, 0].
    ///
    /// This matches the E5/MiniLM model-card recipe: masked mean pooling + L2.
    #[test]
    fn test_mean_pool_averages_masked_tokens_and_l2_unit_norm() {
        let (hidden, mask) = hidden_2x4();
        let seq_len = 2;
        let hidden_size = 4;

        let mut pooled = mean_pool(&hidden, &mask, seq_len, hidden_size);

        // Before L2: mean of [1,0,0,0] and [0,1,0,0] = [0.5, 0.5, 0, 0].
        assert_relative_eq!(pooled[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(pooled[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(pooled[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(pooled[3], 0.0, epsilon = 1e-6);

        l2_normalize(&mut pooled);

        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);

        // L2 of [0.5, 0.5, 0, 0]: magnitude = √0.5, so normalised = [1/√2, 1/√2, 0, 0].
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(pooled[0], inv_sqrt2, epsilon = 1e-5);
        assert_relative_eq!(pooled[1], inv_sqrt2, epsilon = 1e-5);
    }

    /// CLS and mean pooling of the same hidden states produce DIFFERENT vectors.
    ///
    /// This is the key correctness guarantee for P1-E3: using the wrong pooling
    /// strategy for a model produces a meaningfully different embedding.
    #[test]
    fn test_cls_and_mean_produce_different_embeddings() {
        let (hidden, mask) = hidden_2x4();
        let seq_len = 2;
        let hidden_size = 4;

        let mut cls = cls_pool(&hidden, seq_len, hidden_size);
        let mut mean = mean_pool(&hidden, &mask, seq_len, hidden_size);

        l2_normalize(&mut cls);
        l2_normalize(&mut mean);

        // CLS = [1, 0, 0, 0],  mean = [1/√2, 1/√2, 0, 0] — these differ.
        assert_ne!(
            cls, mean,
            "CLS and mean pooling must produce different unit vectors"
        );
    }

    /// Mean pooling with a padding mask ignores masked positions.
    ///
    /// With hidden = [[1,0,0,0],[0,1,0,0]] and mask [1, 0],
    /// only token 0 contributes: mean = [1, 0, 0, 0].
    #[test]
    fn test_mean_pool_respects_padding_mask() {
        let hidden = vec![
            1.0_f32, 0.0, 0.0, 0.0, // token 0 (real)
            0.0_f32, 1.0, 0.0, 0.0, // token 1 (pad, mask=0)
        ];
        let mask = vec![1_u32, 0]; // second token is padding
        let seq_len = 2;
        let hidden_size = 4;

        let pooled = mean_pool(&hidden, &mask, seq_len, hidden_size);

        // Only token 0 is unmasked → mean = [1,0,0,0].
        assert_relative_eq!(pooled[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(pooled[1], 0.0, epsilon = 1e-6);
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
