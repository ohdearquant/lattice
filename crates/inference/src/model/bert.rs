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
use crate::tokenizer::common::{EmbeddingTokenCounts, Tokenizer, load_tokenizer};
use crate::weights::{BertWeights, SafetensorsFile, TransformerLayerWeights};
use std::fs;
use std::path::Path;
use tracing::warn;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_facts;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_config;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_tensor_inventory;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_tokenizer_selection;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_file_presence;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_tokenizer_config;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_sequence_cap;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_sequence_cap_adapter;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_sequence_cap_finalizer;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_tokenizer_json_model;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_vocab_txt;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_wordpiece_vocab_match;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_merges_txt;

#[cfg_attr(not(test), allow(dead_code))]
mod prepared_bpe_vocab_match;

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
    ///
    /// Every required weight passes the shared ingress validation seam before
    /// `Self` is assembled. A validation failure returns an error and drops all
    /// local loading state; callers cannot observe a partially built model.
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
    /// As with [`from_directory`](Self::from_directory), all required weights
    /// are validated before `Self` is assembled, so an error never exposes a
    /// partially built model.
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

    // Classification borrows the existing mmap-backed pooler without changing
    // the independent mean/CLS pooling policy used by embedding callers.
    pub(crate) fn pooler_parameters(&self) -> (&[f32], &[f32]) {
        (
            self.weights.pooler_weight.data,
            self.weights.pooler_bias.data,
        )
    }

    /// **Unstable**: tokenizer accessor; exposed for testing only, may be removed.
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// **Unstable**: token counts of `text` as `encode` and `encode_batch` consume it.
    ///
    /// `before_truncation` is the tokenizer's own pre-truncation length, special tokens
    /// included, so one tokenization yields both figures. `embedded` is the part of it the
    /// model keeps: the tokenizer's sequence limit and the position-embedding table both cap it.
    pub fn embedding_token_counts(&self, text: &str) -> EmbeddingTokenCounts {
        let input = self.tokenizer.tokenize(text);
        EmbeddingTokenCounts {
            before_truncation: input.pre_truncation_len,
            embedded: input.real_length.min(self.config.max_position_embeddings),
        }
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
    /// sequence (see `crate::attention::multi_head_attention_batched`) are run
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
mod tests;

/// Compile-time guard for the struct field drop-order invariant.
///
/// `BertModel` relies on Rust dropping struct fields in declaration order
/// (RFC 1857): `weights` (the borrower) must be dropped before `_safetensors`
/// (the backing store). If this language guarantee ever changes, or if someone
/// accidentally reorders the fields, this test will catch it.
#[cfg(test)]
mod drop_order_tests;
