//! Qwen3-Embedding model: decoder-only transformer inference.
//!
//! Architecture: causal attention with GQA, RoPE, RMSNorm, SwiGLU FFN,
//! last-token pooling. Supports Matryoshka (output dimension truncation).

use crate::attention::gqa::{GqaConfig, GqaScratch, apply_gqa_attention};
use crate::download::ensure_model_files;
use crate::error::InferenceError;
use crate::forward::cpu::{elementwise_mul, matmul_bt, rms_norm, silu_inplace};
use crate::forward::metal::MetalForwardPass;
use crate::pool::{l2_normalize, last_token_pool};
use crate::rope::RopeTable;
use crate::tokenizer::common::{Tokenizer, load_tokenizer};
use crate::weights::{QwenWeights, SafetensorsFile, ShardedQwenBacking, ShardedSafetensors};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

/// Backing storage for Qwen model weights — either a single mmap'd file or
/// an owned heap allocation for sharded checkpoints.
///
/// Drop order within `QwenModel` guarantees that `weights` is dropped before
/// this enum (RFC 1857 struct field ordering), keeping all tensor slice
/// references valid for the model's lifetime.
///
/// The inner fields are kept solely for their `Drop` effect (RAII backing store).
#[allow(dead_code)]
enum SafetensorsStorage {
    /// Single `model.safetensors` file — tensors borrow from the mmap.
    Single(Box<SafetensorsFile>),
    /// Sharded checkpoint — tensors borrow from the owned `Vec<f32>` fields.
    Sharded(Box<ShardedQwenBacking>),
}

/// **Unstable**: per-component timing for profiling forward passes; fields may grow.
#[derive(Debug, Clone, Default)]
pub struct ProfileTimings {
    pub tokenize_us: u64,
    pub embed_lookup_us: u64,
    /// Per-layer breakdown (indexed by layer).
    pub layers: Vec<LayerTimings>,
    pub final_norm_us: u64,
    pub pool_us: u64,
    pub normalize_us: u64,
    pub total_us: u64,
}

/// **Unstable**: per-layer timing breakdown; field set may change as attention paths evolve.
#[derive(Debug, Clone, Default)]
pub struct LayerTimings {
    pub pre_norm_us: u64,
    pub qkv_proj_us: u64,
    pub qk_norm_us: u64,
    pub rope_us: u64,
    pub attention_us: u64,
    pub o_proj_us: u64,
    pub post_norm_us: u64,
    pub ffn_us: u64,
}

impl ProfileTimings {
    /// **Unstable**: human-readable timing report for debugging; format may change.
    pub fn print_report(&self, seq_len: usize) {
        println!("=== Qwen3 Forward Pass Profile (seq_len={seq_len}) ===\n");
        println!("{:>20}  {:>10}", "Component", "Time (us)");
        println!("{:->20}  {:->10}", "", "");
        println!("{:>20}  {:>10}", "Tokenize", self.tokenize_us);
        println!("{:>20}  {:>10}", "Embed lookup", self.embed_lookup_us);

        // Aggregate layer stats.
        let n = self.layers.len();
        let mut sum = LayerTimings::default();
        for l in &self.layers {
            sum.pre_norm_us += l.pre_norm_us;
            sum.qkv_proj_us += l.qkv_proj_us;
            sum.qk_norm_us += l.qk_norm_us;
            sum.rope_us += l.rope_us;
            sum.attention_us += l.attention_us;
            sum.o_proj_us += l.o_proj_us;
            sum.post_norm_us += l.post_norm_us;
            sum.ffn_us += l.ffn_us;
        }
        let layer_total = sum.pre_norm_us
            + sum.qkv_proj_us
            + sum.qk_norm_us
            + sum.rope_us
            + sum.attention_us
            + sum.o_proj_us
            + sum.post_norm_us
            + sum.ffn_us;

        println!("{:>20}  {:>10}  ({n} layers)", "Layers total", layer_total);
        println!(
            "{:>20}  {:>10}  ({:.0}/layer)",
            "  Pre-norm",
            sum.pre_norm_us,
            sum.pre_norm_us as f64 / n as f64
        );
        println!(
            "{:>20}  {:>10}  ({:.0}/layer)",
            "  QKV proj",
            sum.qkv_proj_us,
            sum.qkv_proj_us as f64 / n as f64
        );
        println!(
            "{:>20}  {:>10}  ({:.0}/layer)",
            "  QK-norm",
            sum.qk_norm_us,
            sum.qk_norm_us as f64 / n as f64
        );
        println!(
            "{:>20}  {:>10}  ({:.0}/layer)",
            "  RoPE",
            sum.rope_us,
            sum.rope_us as f64 / n as f64
        );
        println!(
            "{:>20}  {:>10}  ({:.0}/layer)",
            "  Attention",
            sum.attention_us,
            sum.attention_us as f64 / n as f64
        );
        println!(
            "{:>20}  {:>10}  ({:.0}/layer)",
            "  O proj",
            sum.o_proj_us,
            sum.o_proj_us as f64 / n as f64
        );
        println!(
            "{:>20}  {:>10}  ({:.0}/layer)",
            "  Post-norm",
            sum.post_norm_us,
            sum.post_norm_us as f64 / n as f64
        );
        println!(
            "{:>20}  {:>10}  ({:.0}/layer)",
            "  FFN",
            sum.ffn_us,
            sum.ffn_us as f64 / n as f64
        );
        println!("{:>20}  {:>10}", "Final norm", self.final_norm_us);
        println!("{:>20}  {:>10}", "Pool", self.pool_us);
        println!("{:>20}  {:>10}", "L2 normalize", self.normalize_us);
        println!("{:->20}  {:->10}", "", "");
        println!(
            "{:>20}  {:>10}  ({:.1}ms)",
            "TOTAL",
            self.total_us,
            self.total_us as f64 / 1000.0
        );

        // Percentage breakdown.
        let t = self.total_us.max(1) as f64;
        println!("\n--- Percentage Breakdown ---");
        println!(
            "  Tokenize:     {:>5.1}%",
            self.tokenize_us as f64 / t * 100.0
        );
        println!(
            "  Embed lookup: {:>5.1}%",
            self.embed_lookup_us as f64 / t * 100.0
        );
        println!(
            "  QKV proj:     {:>5.1}%",
            sum.qkv_proj_us as f64 / t * 100.0
        );
        println!(
            "  QK-norm:      {:>5.1}%",
            sum.qk_norm_us as f64 / t * 100.0
        );
        println!("  RoPE:         {:>5.1}%", sum.rope_us as f64 / t * 100.0);
        println!(
            "  Attention:    {:>5.1}%",
            sum.attention_us as f64 / t * 100.0
        );
        println!("  O proj:       {:>5.1}%", sum.o_proj_us as f64 / t * 100.0);
        println!("  FFN:          {:>5.1}%", sum.ffn_us as f64 / t * 100.0);
        println!(
            "  Norms:        {:>5.1}%",
            (sum.pre_norm_us + sum.post_norm_us + self.final_norm_us) as f64 / t * 100.0
        );
        println!(
            "  Pool+Norm:    {:>5.1}%",
            (self.pool_us + self.normalize_us) as f64 / t * 100.0
        );
    }
}

/// **Unstable**: Qwen3 embedding model configuration; consumed by `QwenModel` loader.
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
}

impl QwenConfig {
    /// **Unstable**: hardcoded Qwen3-Embedding-0.6B defaults.
    pub fn qwen3_embedding_0_6b() -> Self {
        Self {
            vocab_size: 151_669,
            hidden_size: 1024,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 3072,
            max_position_embeddings: 32_768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    /// **Unstable**: total Q projection output dimension.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// **Unstable**: total KV projection output dimension.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// **Unstable**: GQA group size (Q heads per KV head).
    pub fn num_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

/// Per-model inference configuration.
///
/// Loaded from `inference_config.json` in the model directory.
/// Defaults preserve exact Qwen3-Embedding-0.6B behavior.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct ModelInferenceConfig {
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,

    #[serde(default = "default_rope_table_max_seq_len")]
    pub rope_table_max_seq_len: usize,

    #[serde(default = "default_gpu_max_seq_len")]
    pub gpu_max_seq_len: usize,
}

fn default_eos_token_id() -> u32 {
    151_643
}
fn default_rope_table_max_seq_len() -> usize {
    8192
}
fn default_gpu_max_seq_len() -> usize {
    2048
}

impl Default for ModelInferenceConfig {
    fn default() -> Self {
        Self {
            eos_token_id: default_eos_token_id(),
            rope_table_max_seq_len: default_rope_table_max_seq_len(),
            gpu_max_seq_len: default_gpu_max_seq_len(),
        }
    }
}

impl ModelInferenceConfig {
    pub fn load(model_dir: &Path) -> Self {
        let path = model_dir.join("inference_config.json");
        match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<Self>(&text) {
                Ok(cfg) => cfg,
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "failed to parse inference_config.json, using defaults"
                    );
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }
}

/// Pre-allocated buffers for the forward pass, reused across calls.
/// Buffers grow monotonically — allocate larger if needed, never shrink.
struct ForwardBuffers {
    hidden: Vec<f32>,
    residual: Vec<f32>,
    /// Fused QKV GEMM output: [seq_len, q_dim + 2*kv_dim].
    qkv_buf: Vec<f32>,
    /// Scattered contiguous Q: [seq_len, q_dim].
    q_buf: Vec<f32>,
    /// Scattered contiguous K: [seq_len, kv_dim].
    k_buf: Vec<f32>,
    /// Scattered contiguous V: [seq_len, kv_dim].
    v_buf: Vec<f32>,
    attn_out: Vec<f32>,
    /// GQA scratch: batched Q, scores, context, K/V per KV-head.
    gqa_scratch: GqaScratch,
    /// Fused gate+up GEMM output: [seq_len, 2*intermediate_size].
    gate_up_buf: Vec<f32>,
    /// Scattered contiguous gate: [seq_len, intermediate_size].
    gate_buf: Vec<f32>,
    /// Scattered contiguous up: [seq_len, intermediate_size].
    up_buf: Vec<f32>,
    ffn_out: Vec<f32>,
}

impl ForwardBuffers {
    fn new() -> Self {
        Self {
            hidden: Vec::new(),
            residual: Vec::new(),
            qkv_buf: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            attn_out: Vec::new(),
            gqa_scratch: GqaScratch::default(),
            gate_up_buf: Vec::new(),
            gate_buf: Vec::new(),
            up_buf: Vec::new(),
            ffn_out: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, seq_len: usize, cfg: &QwenConfig) {
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.q_dim() + 2 * cfg.kv_dim();
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let gate_up_dim = 2 * inter;

        let resize = |buf: &mut Vec<f32>, needed: usize| {
            if buf.len() < needed {
                buf.resize(needed, 0.0);
            }
        };

        resize(&mut self.hidden, seq_len * hidden);
        resize(&mut self.residual, seq_len * hidden);
        resize(&mut self.qkv_buf, seq_len * qkv_dim);
        resize(&mut self.q_buf, seq_len * q_dim);
        resize(&mut self.k_buf, seq_len * kv_dim);
        resize(&mut self.v_buf, seq_len * kv_dim);
        resize(&mut self.attn_out, seq_len * q_dim);
        self.gqa_scratch.reserve_for(
            seq_len,
            GqaConfig {
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            },
        );
        resize(&mut self.gate_up_buf, seq_len * gate_up_dim);
        resize(&mut self.gate_buf, seq_len * inter);
        resize(&mut self.up_buf, seq_len * inter);
        resize(&mut self.ffn_out, seq_len * hidden);
    }
}

/// A Qwen3 model loaded and ready for inference.
///
/// Uses the same self-referential mmap pattern as BertModel — see BertModel
/// docs for the safety argument. Field order is critical: weights before backing.
/// Maximum number of cached embeddings. 10K × 1024d × 4B = ~40MB.
const EMBEDDING_CACHE_CAP: usize = 10_000;

/// **Unstable**: Qwen3-Embedding model loaded for inference; internal architecture under active development.
pub struct QwenModel {
    config: QwenConfig,
    inference_config: ModelInferenceConfig,
    tokenizer: Box<dyn Tokenizer>,
    rope: RopeTable,
    output_dim: Option<usize>,
    /// Reusable forward-pass buffers. Mutex for Sync (uncontended lock ~25ns).
    buffers: Mutex<ForwardBuffers>,
    /// Embedding cache: hash(token_ids) → embedding vector. Skips forward pass on hit.
    cache: Mutex<HashMap<u64, Vec<f32>>>,
    /// Optional Metal GPU forward pass. When available, transformer layers run on GPU
    /// while embedding lookup and pooling stay on CPU.
    metal: Option<Mutex<MetalForwardPass>>,
    // INVARIANT: `weights` MUST be declared before `_storage`.
    // RFC 1857 guarantees struct fields are dropped in declaration order, so
    // `weights` (which holds slices into `_storage`) is dropped first.
    weights: QwenWeights<'static>,
    _storage: SafetensorsStorage,
}

impl QwenModel {
    /// **Unstable**: load Qwen3 model from a local directory; path convention may change.
    pub fn from_directory(dir: &Path) -> Result<Self, InferenceError> {
        let tokenizer = load_tokenizer(dir)?;
        let inference_config = ModelInferenceConfig::load(dir);

        let model_path = dir.join("model.safetensors");
        let index_path = dir.join("model.safetensors.index.json");

        let config_path = dir.join("config.json");
        let config = if config_path.exists() {
            parse_qwen_config(&config_path)?
        } else if is_qwen3_embedding_0_6b_dir(dir) {
            QwenConfig::qwen3_embedding_0_6b()
        } else {
            return Err(InferenceError::Inference(format!(
                "config.json not found in {} — required for non-0.6B models",
                dir.display()
            )));
        };

        let gpu_max_seq_len = inference_config.gpu_max_seq_len;

        if model_path.exists() {
            // --- Single-file path (original behaviour) ---
            let safetensors = Box::new(SafetensorsFile::open(&model_path)?);

            let weights_tmp = safetensors.load_qwen_weights(
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.intermediate_size,
            )?;

            // Try to create Metal GPU forward pass before transmuting weights.
            // MetalForwardPass copies weights into GPU buffers, so it only borrows temporarily.
            let metal = if std::env::var("LATTICE_NO_GPU").is_ok() {
                tracing::info!("Metal GPU disabled by LATTICE_NO_GPU env var");
                None
            } else {
                match MetalForwardPass::new(&config, &weights_tmp, gpu_max_seq_len) {
                    Ok(m) => {
                        tracing::info!(
                            max_seq_len = gpu_max_seq_len,
                            "Metal GPU forward pass enabled"
                        );
                        Some(Mutex::new(m))
                    }
                    Err(e) => {
                        tracing::debug!("Metal GPU not available, using CPU: {e}");
                        None
                    }
                }
            };

            // SAFETY: The backing store (_storage / safetensors) outlives weights
            // because it is dropped after weights (RFC 1857 struct field drop order).
            let weights: QwenWeights<'static> = unsafe { std::mem::transmute(weights_tmp) };

            let rope_max = config
                .max_position_embeddings
                .min(inference_config.rope_table_max_seq_len);
            let rope = RopeTable::new(config.head_dim, rope_max, config.rope_theta);

            Ok(Self {
                config,
                inference_config,
                tokenizer,
                rope,
                output_dim: None,
                buffers: Mutex::new(ForwardBuffers::new()),
                cache: Mutex::new(HashMap::with_capacity(1024)),
                metal,
                weights,
                _storage: SafetensorsStorage::Single(safetensors),
            })
        } else if index_path.exists() {
            // --- Sharded path ---
            tracing::info!(
                index = %index_path.display(),
                "Loading sharded safetensors checkpoint"
            );
            let mut sharded = ShardedSafetensors::open_index(&index_path)?;

            let (backing, weights_tmp) = sharded.load_qwen_weights_owned(
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.intermediate_size,
            )?;

            // Try Metal GPU. MetalForwardPass copies weights — it only borrows
            // weights_tmp briefly here, before we hand ownership to the model.
            let metal = if std::env::var("LATTICE_NO_GPU").is_ok() {
                tracing::info!("Metal GPU disabled by LATTICE_NO_GPU env var");
                None
            } else {
                match MetalForwardPass::new(&config, &weights_tmp, gpu_max_seq_len) {
                    Ok(m) => {
                        tracing::info!(
                            max_seq_len = gpu_max_seq_len,
                            "Metal GPU forward pass enabled"
                        );
                        Some(Mutex::new(m))
                    }
                    Err(e) => {
                        tracing::debug!("Metal GPU not available, using CPU: {e}");
                        None
                    }
                }
            };

            // SAFETY: `weights_tmp` already has 'static lifetime from
            // `load_qwen_weights_owned` — the slices point into `backing` which
            // is a heap-allocated Box.  We store `backing` in `_storage` and
            // declare `weights` before `_storage` so drop order is correct.
            let weights: QwenWeights<'static> = weights_tmp;

            let rope_max = config
                .max_position_embeddings
                .min(inference_config.rope_table_max_seq_len);
            let rope = RopeTable::new(config.head_dim, rope_max, config.rope_theta);

            Ok(Self {
                config,
                inference_config,
                tokenizer,
                rope,
                output_dim: None,
                buffers: Mutex::new(ForwardBuffers::new()),
                cache: Mutex::new(HashMap::with_capacity(1024)),
                metal,
                weights,
                _storage: SafetensorsStorage::Sharded(backing),
            })
        } else {
            Err(InferenceError::ModelNotFound(format!(
                "missing model.safetensors or model.safetensors.index.json in {}",
                dir.display()
            )))
        }
    }

    /// **Unstable**: load from the default cache, downloading if absent.
    pub fn from_pretrained(model_name: &str) -> Result<Self, InferenceError> {
        let cache_dir = crate::default_cache_dir()?;
        let model_dir = ensure_model_files(model_name, &cache_dir)?;
        Self::from_directory(&model_dir)
    }

    /// **Unstable**: set Matryoshka output truncation dimension.
    pub fn set_output_dim(&mut self, dim: Option<usize>) {
        self.output_dim = dim;
    }

    /// **Unstable**: access the model configuration.
    pub fn config(&self) -> &QwenConfig {
        &self.config
    }

    /// **Unstable**: access the tokenizer.
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// **Unstable**: access model weights for lm_head weight tying.
    pub fn weights(&self) -> &QwenWeights<'_> {
        &self.weights
    }

    /// **Unstable**: access the RoPE frequency table.
    pub fn rope(&self) -> &RopeTable {
        &self.rope
    }

    /// **Unstable**: active output dimension (after Matryoshka truncation, if set).
    pub fn dimensions(&self) -> usize {
        self.output_dim.unwrap_or(self.config.hidden_size)
    }

    /// Whether Metal GPU acceleration is available for the forward pass.
    pub fn has_gpu(&self) -> bool {
        self.metal.is_some()
    }

    /// Number of cached embeddings.
    pub fn cache_size(&self) -> usize {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Clear the embedding cache.
    pub fn cache_clear(&self) {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// Load embedding cache from a binary file. Format: repeated [hash:u64, dim:u32, floats:f32*dim].
    pub fn cache_load(&self, path: &Path) -> Result<usize, InferenceError> {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(InferenceError::Inference(format!("cache load: {e}"))),
        };
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut pos = 0;
        let mut count = 0;
        while pos + 12 <= data.len() {
            let hash = u64::from_le_bytes(
                data[pos..pos + 8]
                    .try_into()
                    .expect("invariant: cache record has an 8-byte hash"),
            );
            let dim = u32::from_le_bytes(
                data[pos + 8..pos + 12]
                    .try_into()
                    .expect("invariant: cache record has a 4-byte dimension"),
            ) as usize;
            pos += 12;
            let Some(float_bytes) = dim.checked_mul(4) else {
                break;
            };
            let Some(end) = pos.checked_add(float_bytes) else {
                break;
            };
            if end > data.len() {
                break;
            }
            let floats: Vec<f32> = data[pos..end]
                .chunks_exact(4)
                .map(|c| {
                    f32::from_le_bytes(
                        c.try_into()
                            .expect("invariant: chunks_exact(4) yields four-byte chunks"),
                    )
                })
                .collect();
            pos = end;
            cache.insert(hash, floats);
            count += 1;
        }
        Ok(count)
    }

    /// Save embedding cache to a binary file.
    pub fn cache_save(&self, path: &Path) -> Result<usize, InferenceError> {
        let cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let count = cache.len();
        let mut buf = Vec::with_capacity(count * (12 + self.dimensions() * 4));
        for (&hash, embedding) in cache.iter() {
            buf.extend_from_slice(&hash.to_le_bytes());
            buf.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
            for &f in embedding {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, &buf)
            .map_err(|e| InferenceError::Inference(format!("cache save: {e}")))?;
        Ok(count)
    }

    fn tokenize_for_embedding(&self, text: &str) -> (Vec<u32>, usize) {
        let input = self.tokenizer.tokenize(text);
        let max_len = self.tokenizer.max_seq_len();
        let mut ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();

        let eos = self.inference_config.eos_token_id;
        if ids.len() < max_len {
            ids.push(eos);
        } else {
            if let Some(last) = ids.last_mut() {
                *last = eos;
            }
        }

        let seq_len = ids.len();
        (ids, seq_len)
    }

    /// Encode a single text into an embedding vector.
    ///
    /// Uses an internal LRU-style cache keyed by token IDs. Repeated texts
    /// return the cached embedding in <1us instead of running the ~100ms
    /// forward pass.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let (ids, seq_len) = self.tokenize_for_embedding(text);

        // Cache lookup by hash of token IDs.
        let key = hash_token_ids(&ids);
        if let Some(cached) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
        {
            return Ok(cached.clone());
        }

        let hidden_states = self.forward(&ids, seq_len);

        // Build attention mask for last_token_pool (all 1s since we have no padding).
        let attention_mask = vec![1u32; seq_len];
        let pooled = last_token_pool(
            &hidden_states,
            &attention_mask,
            seq_len,
            self.config.hidden_size,
        );

        // Matryoshka truncation + L2 normalize.
        let dim = self.dimensions();
        let mut output = if dim < pooled.len() {
            pooled[..dim].to_vec()
        } else {
            pooled
        };
        l2_normalize(&mut output);

        // Cache the result. Evict all if at capacity (simple flush strategy).
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if cache.len() >= EMBEDDING_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, output.clone());

        Ok(output)
    }

    /// Encode text and return embeddings at EVERY layer exit point (0..num_layers).
    /// Used for early-exit analysis: find the shallowest layer that produces good embeddings.
    /// Returns Vec of (layer_index, embedding) — layer 0 is after the first transformer layer.
    pub fn encode_all_layers(&self, text: &str) -> Result<Vec<Vec<f32>>, InferenceError> {
        let (ids, seq_len) = self.tokenize_for_embedding(text);
        let hidden_size = self.config.hidden_size;
        let dim = self.dimensions();

        // Run forward pass, capturing hidden states after each layer.
        let all_hidden = self.forward_all_layers(&ids, seq_len);
        let attention_mask = vec![1u32; seq_len];

        let mut results = Vec::with_capacity(all_hidden.len());
        for layer_hidden in &all_hidden {
            // Apply final RMSNorm to a copy.
            let mut h = layer_hidden.clone();
            rms_norm(
                &mut h,
                self.weights.norm_weight.data,
                hidden_size,
                self.config.rms_norm_eps,
            );

            let pooled = last_token_pool(&h, &attention_mask, seq_len, hidden_size);
            let mut output = if dim < pooled.len() {
                pooled[..dim].to_vec()
            } else {
                pooled
            };
            l2_normalize(&mut output);
            results.push(output);
        }
        Ok(results)
    }

    /// Forward pass that returns hidden states after EVERY layer.
    fn forward_all_layers(&self, input_ids: &[u32], seq_len: usize) -> Vec<Vec<f32>> {
        // Reuse the standard forward logic but capture hidden states at each layer.
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = self.config.q_dim();
        let kv_dim = self.config.kv_dim();
        let groups = self.config.num_groups();
        let eps = self.config.rms_norm_eps;
        let qkv_dim = q_dim + 2 * kv_dim;
        let gate_up_dim = 2 * self.config.intermediate_size;
        let inter = self.config.intermediate_size;

        let mut hidden = vec![0.0f32; seq_len * hidden_size];
        for i in 0..seq_len {
            let tok_id = input_ids[i] as usize;
            let emb_row =
                &self.weights.embed_tokens.data[tok_id * hidden_size..(tok_id + 1) * hidden_size];
            hidden[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(emb_row);
        }

        let mut residual = vec![0.0f32; seq_len * hidden_size];
        let mut qkv_buf = vec![0.0f32; seq_len * qkv_dim];
        let mut q_buf = vec![0.0f32; seq_len * q_dim];
        let mut k_buf = vec![0.0f32; seq_len * kv_dim];
        let mut v_buf = vec![0.0f32; seq_len * kv_dim];
        let mut attn_out = vec![0.0f32; seq_len * q_dim];
        let mut scores_head = vec![0.0f32; seq_len * seq_len];
        let mut q_head = vec![0.0f32; seq_len * head_dim];
        let mut k_head = vec![0.0f32; seq_len * head_dim];
        let mut v_head_t = vec![0.0f32; head_dim * seq_len];
        let mut context_head = vec![0.0f32; seq_len * head_dim];
        let mut gate_up_buf = vec![0.0f32; seq_len * gate_up_dim];
        let mut gate_buf = vec![0.0f32; seq_len * inter];
        let mut up_buf = vec![0.0f32; seq_len * inter];
        let mut ffn_out = vec![0.0f32; seq_len * hidden_size];

        let mut all_hidden = Vec::with_capacity(self.config.num_hidden_layers);

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];

            residual.copy_from_slice(&hidden);
            rms_norm(
                &mut hidden,
                layer.input_layernorm_weight.data,
                hidden_size,
                eps,
            );

            matmul_bt(
                &hidden,
                &layer.fused_qkv,
                &mut qkv_buf,
                seq_len,
                hidden_size,
                qkv_dim,
            );
            for i in 0..seq_len {
                let r = i * qkv_dim;
                q_buf[i * q_dim..(i + 1) * q_dim].copy_from_slice(&qkv_buf[r..r + q_dim]);
                k_buf[i * kv_dim..(i + 1) * kv_dim]
                    .copy_from_slice(&qkv_buf[r + q_dim..r + q_dim + kv_dim]);
                v_buf[i * kv_dim..(i + 1) * kv_dim]
                    .copy_from_slice(&qkv_buf[r + q_dim + kv_dim..r + qkv_dim]);
            }

            for pos in 0..seq_len {
                for h in 0..num_heads {
                    let s = pos * q_dim + h * head_dim;
                    rms_norm(
                        &mut q_buf[s..s + head_dim],
                        layer.q_norm_weight.data,
                        head_dim,
                        eps,
                    );
                }
                for h in 0..num_kv_heads {
                    let s = pos * kv_dim + h * head_dim;
                    rms_norm(
                        &mut k_buf[s..s + head_dim],
                        layer.k_norm_weight.data,
                        head_dim,
                        eps,
                    );
                }
            }
            for pos in 0..seq_len {
                for h in 0..num_heads {
                    let s = pos * q_dim + h * head_dim;
                    self.rope.apply(&mut q_buf[s..s + head_dim], pos);
                }
                for h in 0..num_kv_heads {
                    let s = pos * kv_dim + h * head_dim;
                    self.rope.apply(&mut k_buf[s..s + head_dim], pos);
                }
            }

            let scale = 1.0 / (head_dim as f32).sqrt();
            for h in 0..num_heads {
                let kv_h = h / groups;
                for i in 0..seq_len {
                    q_head[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                        &q_buf[i * q_dim + h * head_dim..i * q_dim + (h + 1) * head_dim],
                    );
                }
                for i in 0..seq_len {
                    k_head[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                        &k_buf[i * kv_dim + kv_h * head_dim..i * kv_dim + (kv_h + 1) * head_dim],
                    );
                }
                matmul_bt(
                    &q_head[..seq_len * head_dim],
                    &k_head[..seq_len * head_dim],
                    &mut scores_head[..seq_len * seq_len],
                    seq_len,
                    head_dim,
                    seq_len,
                );
                for qi in 0..seq_len {
                    let row = &mut scores_head[qi * seq_len..(qi + 1) * seq_len];
                    for ki in 0..seq_len {
                        if ki > qi {
                            row[ki] = f32::NEG_INFINITY;
                        } else {
                            row[ki] *= scale;
                        }
                    }
                    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for v in row.iter_mut() {
                            *v *= inv;
                        }
                    }
                }
                for i in 0..seq_len {
                    let v_off = i * kv_dim + kv_h * head_dim;
                    for d in 0..head_dim {
                        v_head_t[d * seq_len + i] = v_buf[v_off + d];
                    }
                }
                matmul_bt(
                    &scores_head[..seq_len * seq_len],
                    &v_head_t[..head_dim * seq_len],
                    &mut context_head[..seq_len * head_dim],
                    seq_len,
                    seq_len,
                    head_dim,
                );
                for i in 0..seq_len {
                    attn_out[i * q_dim + h * head_dim..i * q_dim + (h + 1) * head_dim]
                        .copy_from_slice(&context_head[i * head_dim..(i + 1) * head_dim]);
                }
            }

            matmul_bt(
                &attn_out,
                layer.o_proj_weight.data,
                &mut hidden,
                seq_len,
                q_dim,
                hidden_size,
            );
            for i in 0..seq_len * hidden_size {
                hidden[i] += residual[i];
            }

            residual.copy_from_slice(&hidden);
            rms_norm(
                &mut hidden,
                layer.post_attention_layernorm_weight.data,
                hidden_size,
                eps,
            );

            matmul_bt(
                &hidden,
                &layer.fused_gate_up,
                &mut gate_up_buf,
                seq_len,
                hidden_size,
                gate_up_dim,
            );
            for i in 0..seq_len {
                let r = i * gate_up_dim;
                gate_buf[i * inter..(i + 1) * inter].copy_from_slice(&gate_up_buf[r..r + inter]);
                up_buf[i * inter..(i + 1) * inter]
                    .copy_from_slice(&gate_up_buf[r + inter..r + gate_up_dim]);
            }
            silu_inplace(&mut gate_buf[..seq_len * inter]);
            elementwise_mul(&mut gate_buf[..seq_len * inter], &up_buf[..seq_len * inter]);
            matmul_bt(
                &gate_buf,
                layer.down_proj_weight.data,
                &mut ffn_out,
                seq_len,
                inter,
                hidden_size,
            );
            for i in 0..seq_len * hidden_size {
                hidden[i] = residual[i] + ffn_out[i];
            }

            all_hidden.push(hidden.clone());
        }

        all_hidden
    }

    /// Batch-encode multiple texts, reusing buffers.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, InferenceError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut outputs = Vec::with_capacity(texts.len());

        for text in texts {
            let (ids, seq_len) = self.tokenize_for_embedding(text);
            let hidden_states = self.forward(&ids, seq_len);

            let attention_mask = vec![1u32; seq_len];
            let pooled = last_token_pool(
                &hidden_states,
                &attention_mask,
                seq_len,
                self.config.hidden_size,
            );

            let dim = self.dimensions();
            let mut output = if dim < pooled.len() {
                pooled[..dim].to_vec()
            } else {
                pooled
            };
            l2_normalize(&mut output);
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Encode with per-component timing breakdown.
    pub fn encode_profiled(
        &self,
        text: &str,
    ) -> Result<(Vec<f32>, ProfileTimings), InferenceError> {
        let total_start = Instant::now();
        let mut timings = ProfileTimings::default();

        let t = Instant::now();
        let (ids, seq_len) = self.tokenize_for_embedding(text);
        timings.tokenize_us = t.elapsed().as_micros() as u64;

        let (hidden_states, layer_timings) = self.forward_profiled(&ids, seq_len, &mut timings);
        timings.layers = layer_timings;

        let t = Instant::now();
        let attention_mask = vec![1u32; seq_len];
        let pooled = last_token_pool(
            &hidden_states,
            &attention_mask,
            seq_len,
            self.config.hidden_size,
        );
        timings.pool_us = t.elapsed().as_micros() as u64;

        let dim = self.dimensions();
        let mut output = if dim < pooled.len() {
            pooled[..dim].to_vec()
        } else {
            pooled
        };

        let t = Instant::now();
        l2_normalize(&mut output);
        timings.normalize_us = t.elapsed().as_micros() as u64;

        timings.total_us = total_start.elapsed().as_micros() as u64;
        Ok((output, timings))
    }

    /// Profiled forward pass — same scatter+contiguous logic as forward() but collects per-component timings.
    fn forward_profiled(
        &self,
        input_ids: &[u32],
        seq_len: usize,
        timings: &mut ProfileTimings,
    ) -> (Vec<f32>, Vec<LayerTimings>) {
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = self.config.q_dim();
        let kv_dim = self.config.kv_dim();
        let eps = self.config.rms_norm_eps;
        let qkv_dim = q_dim + 2 * kv_dim;
        let gate_up_dim = 2 * self.config.intermediate_size;
        let inter = self.config.intermediate_size;

        let t = Instant::now();
        let mut hidden = vec![0.0f32; seq_len * hidden_size];
        for i in 0..seq_len {
            let tok_id = input_ids[i] as usize;
            let emb_row =
                &self.weights.embed_tokens.data[tok_id * hidden_size..(tok_id + 1) * hidden_size];
            hidden[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(emb_row);
        }
        timings.embed_lookup_us = t.elapsed().as_micros() as u64;

        let mut residual = vec![0.0f32; seq_len * hidden_size];
        let mut qkv_buf = vec![0.0f32; seq_len * qkv_dim];
        let mut q_buf = vec![0.0f32; seq_len * q_dim];
        let mut k_buf = vec![0.0f32; seq_len * kv_dim];
        let mut v_buf = vec![0.0f32; seq_len * kv_dim];
        let mut attn_out = vec![0.0f32; seq_len * q_dim];
        let mut gate_up_buf = vec![0.0f32; seq_len * gate_up_dim];
        let mut gate_buf = vec![0.0f32; seq_len * inter];
        let mut up_buf = vec![0.0f32; seq_len * inter];
        let mut ffn_out = vec![0.0f32; seq_len * hidden_size];

        let mut layer_timings = Vec::with_capacity(self.config.num_hidden_layers);

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];
            let mut lt = LayerTimings::default();

            let t = Instant::now();
            residual.copy_from_slice(&hidden);
            rms_norm(
                &mut hidden,
                layer.input_layernorm_weight.data,
                hidden_size,
                eps,
            );
            lt.pre_norm_us = t.elapsed().as_micros() as u64;

            // Fused QKV projection: 1 GEMM instead of 3.
            let t = Instant::now();
            matmul_bt(
                &hidden,
                &layer.fused_qkv,
                &mut qkv_buf,
                seq_len,
                hidden_size,
                qkv_dim,
            );
            // Scatter fused QKV to contiguous Q, K, V buffers.
            for i in 0..seq_len {
                let qkv_row = i * qkv_dim;
                q_buf[i * q_dim..(i + 1) * q_dim]
                    .copy_from_slice(&qkv_buf[qkv_row..qkv_row + q_dim]);
                k_buf[i * kv_dim..(i + 1) * kv_dim]
                    .copy_from_slice(&qkv_buf[qkv_row + q_dim..qkv_row + q_dim + kv_dim]);
                v_buf[i * kv_dim..(i + 1) * kv_dim]
                    .copy_from_slice(&qkv_buf[qkv_row + q_dim + kv_dim..qkv_row + qkv_dim]);
            }
            lt.qkv_proj_us = t.elapsed().as_micros() as u64;

            // QK-norm on contiguous Q and K buffers.
            let t = Instant::now();
            for pos in 0..seq_len {
                for h in 0..num_heads {
                    let q_start = pos * q_dim + h * head_dim;
                    rms_norm(
                        &mut q_buf[q_start..q_start + head_dim],
                        layer.q_norm_weight.data,
                        head_dim,
                        eps,
                    );
                }
                for h in 0..num_kv_heads {
                    let k_start = pos * kv_dim + h * head_dim;
                    rms_norm(
                        &mut k_buf[k_start..k_start + head_dim],
                        layer.k_norm_weight.data,
                        head_dim,
                        eps,
                    );
                }
            }
            lt.qk_norm_us = t.elapsed().as_micros() as u64;

            // RoPE on contiguous Q and K buffers.
            let t = Instant::now();
            for pos in 0..seq_len {
                for h in 0..num_heads {
                    let q_start = pos * q_dim + h * head_dim;
                    self.rope
                        .apply(&mut q_buf[q_start..q_start + head_dim], pos);
                }
                for h in 0..num_kv_heads {
                    let k_start = pos * kv_dim + h * head_dim;
                    self.rope
                        .apply(&mut k_buf[k_start..k_start + head_dim], pos);
                }
            }
            lt.rope_us = t.elapsed().as_micros() as u64;

            // Grouped-Query Attention (BLAS per-head).
            let t = Instant::now();
            let mut scores_head = vec![0.0f32; seq_len * seq_len];
            let mut q_head = vec![0.0f32; seq_len * head_dim];
            let mut k_head = vec![0.0f32; seq_len * head_dim];
            let mut v_head_t = vec![0.0f32; head_dim * seq_len];
            let mut context_head = vec![0.0f32; seq_len * head_dim];
            let scale = 1.0 / (head_dim as f32).sqrt();
            let groups = num_heads / num_kv_heads;

            for h in 0..num_heads {
                let kv_h = h / groups;

                // Extract Q_head [seq_len, head_dim].
                for i in 0..seq_len {
                    q_head[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                        &q_buf[i * q_dim + h * head_dim..i * q_dim + (h + 1) * head_dim],
                    );
                }
                // Extract K_head [seq_len, head_dim].
                for i in 0..seq_len {
                    k_head[i * head_dim..(i + 1) * head_dim].copy_from_slice(
                        &k_buf[i * kv_dim + kv_h * head_dim..i * kv_dim + (kv_h + 1) * head_dim],
                    );
                }

                // scores = Q_head @ K_head^T  [seq_len, seq_len] via BLAS.
                matmul_bt(
                    &q_head[..seq_len * head_dim],
                    &k_head[..seq_len * head_dim],
                    &mut scores_head[..seq_len * seq_len],
                    seq_len,
                    head_dim,
                    seq_len,
                );

                // Scale + causal mask + softmax.
                for qi in 0..seq_len {
                    let row = &mut scores_head[qi * seq_len..(qi + 1) * seq_len];
                    for ki in 0..seq_len {
                        if ki > qi {
                            row[ki] = f32::NEG_INFINITY;
                        } else {
                            row[ki] *= scale;
                        }
                    }
                    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for v in row.iter_mut() {
                            *v *= inv;
                        }
                    }
                }

                // Transpose V_head for matmul_bt trick: V_T[head_dim, seq_len]
                for i in 0..seq_len {
                    let v_off = i * kv_dim + kv_h * head_dim;
                    for d in 0..head_dim {
                        v_head_t[d * seq_len + i] = v_buf[v_off + d];
                    }
                }

                // context = scores @ V = scores @ V_T^T via matmul_bt [seq_len, head_dim]
                matmul_bt(
                    &scores_head[..seq_len * seq_len],
                    &v_head_t[..head_dim * seq_len],
                    &mut context_head[..seq_len * head_dim],
                    seq_len,
                    seq_len,
                    head_dim,
                );

                // Write back.
                for i in 0..seq_len {
                    attn_out[i * q_dim + h * head_dim..i * q_dim + (h + 1) * head_dim]
                        .copy_from_slice(&context_head[i * head_dim..(i + 1) * head_dim]);
                }
            }
            lt.attention_us = t.elapsed().as_micros() as u64;

            let t = Instant::now();
            matmul_bt(
                &attn_out,
                layer.o_proj_weight.data,
                &mut hidden,
                seq_len,
                q_dim,
                hidden_size,
            );
            for i in 0..seq_len * hidden_size {
                hidden[i] += residual[i];
            }
            lt.o_proj_us = t.elapsed().as_micros() as u64;

            let t = Instant::now();
            residual.copy_from_slice(&hidden);
            rms_norm(
                &mut hidden,
                layer.post_attention_layernorm_weight.data,
                hidden_size,
                eps,
            );
            lt.post_norm_us = t.elapsed().as_micros() as u64;

            // Fused gate+up FFN: 1 GEMM instead of 2.
            let t = Instant::now();
            matmul_bt(
                &hidden,
                &layer.fused_gate_up,
                &mut gate_up_buf,
                seq_len,
                hidden_size,
                gate_up_dim,
            );

            // Scatter fused gate+up to separate contiguous buffers.
            for i in 0..seq_len {
                let gu_row = i * gate_up_dim;
                gate_buf[i * inter..(i + 1) * inter]
                    .copy_from_slice(&gate_up_buf[gu_row..gu_row + inter]);
                up_buf[i * inter..(i + 1) * inter]
                    .copy_from_slice(&gate_up_buf[gu_row + inter..gu_row + gate_up_dim]);
            }

            // SiLU + elementwise mul on contiguous buffers.
            silu_inplace(&mut gate_buf[..seq_len * inter]);
            elementwise_mul(&mut gate_buf[..seq_len * inter], &up_buf[..seq_len * inter]);

            // Down projection on contiguous gate_buf.
            matmul_bt(
                &gate_buf,
                layer.down_proj_weight.data,
                &mut ffn_out,
                seq_len,
                inter,
                hidden_size,
            );
            for i in 0..seq_len * hidden_size {
                hidden[i] = residual[i] + ffn_out[i];
            }
            lt.ffn_us = t.elapsed().as_micros() as u64;

            layer_timings.push(lt);
        }

        let t = Instant::now();
        rms_norm(&mut hidden, self.weights.norm_weight.data, hidden_size, eps);
        timings.final_norm_us = t.elapsed().as_micros() as u64;

        (hidden, layer_timings)
    }

    /// Full decoder forward pass. Returns hidden states [seq_len * hidden_size].
    ///
    /// When a Metal GPU is available, runs the transformer layers on GPU (5x faster).
    /// Embedding lookup and final pooling always run on CPU.
    fn forward(&self, input_ids: &[u32], seq_len: usize) -> Vec<f32> {
        // GPU path: embedding lookup on CPU, transformer layers on Metal GPU.
        // MetalForwardPass validates seq_len <= its max_seq_len internally.
        if let Some(ref metal_mutex) = self.metal {
            {
                let hidden_size = self.config.hidden_size;
                let mut hidden_input = vec![0.0f32; seq_len * hidden_size];
                for i in 0..seq_len {
                    let tok_id = input_ids[i] as usize;
                    let src = &self.weights.embed_tokens.data
                        [tok_id * hidden_size..(tok_id + 1) * hidden_size];
                    hidden_input[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(src);
                }
                let mut metal = metal_mutex
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match metal.forward(&hidden_input, seq_len) {
                    Ok(output) => return output,
                    Err(e) => {
                        tracing::warn!(
                            "Metal forward failed (seq_len={seq_len}), falling back to CPU: {e}"
                        );
                    }
                }
            }
        }

        // CPU path: full forward pass.
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = self.config.q_dim();
        let kv_dim = self.config.kv_dim();
        let eps = self.config.rms_norm_eps;
        let qkv_dim = q_dim + 2 * kv_dim;
        let gate_up_dim = 2 * self.config.intermediate_size;
        let inter = self.config.intermediate_size;

        let mut bufs = self
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        bufs.ensure_capacity(seq_len, &self.config);

        // Destructure to satisfy the borrow checker — each field is independent.
        let ForwardBuffers {
            hidden,
            residual,
            qkv_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out,
            gqa_scratch,
            gate_up_buf,
            gate_buf,
            up_buf,
            ffn_out,
        } = &mut *bufs;

        let hidden = &mut hidden[..seq_len * hidden_size];
        let residual = &mut residual[..seq_len * hidden_size];

        // Token embeddings (no position embeddings — RoPE is applied in attention).
        for i in 0..seq_len {
            let tok_id = input_ids[i] as usize;
            let emb_row =
                &self.weights.embed_tokens.data[tok_id * hidden_size..(tok_id + 1) * hidden_size];
            hidden[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(emb_row);
        }

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.weights.layers[layer_idx];

            // --- Pre-attention RMSNorm ---
            residual.copy_from_slice(hidden);
            rms_norm(hidden, layer.input_layernorm_weight.data, hidden_size, eps);

            // --- Fused QKV projection: 1 GEMM instead of 3 ---
            let qkv_buf = &mut qkv_buf[..seq_len * qkv_dim];
            matmul_bt(
                hidden,
                &layer.fused_qkv,
                qkv_buf,
                seq_len,
                hidden_size,
                qkv_dim,
            );

            // --- Scatter fused QKV to contiguous Q, K, V buffers ---
            let q_buf = &mut q_buf[..seq_len * q_dim];
            let k_buf = &mut k_buf[..seq_len * kv_dim];
            let v_buf = &mut v_buf[..seq_len * kv_dim];
            for i in 0..seq_len {
                let qkv_row = i * qkv_dim;
                q_buf[i * q_dim..(i + 1) * q_dim]
                    .copy_from_slice(&qkv_buf[qkv_row..qkv_row + q_dim]);
                k_buf[i * kv_dim..(i + 1) * kv_dim]
                    .copy_from_slice(&qkv_buf[qkv_row + q_dim..qkv_row + q_dim + kv_dim]);
                v_buf[i * kv_dim..(i + 1) * kv_dim]
                    .copy_from_slice(&qkv_buf[qkv_row + q_dim + kv_dim..qkv_row + qkv_dim]);
            }

            // --- QK-norm on contiguous Q and K buffers ---
            for pos in 0..seq_len {
                for h in 0..num_heads {
                    let q_start = pos * q_dim + h * head_dim;
                    rms_norm(
                        &mut q_buf[q_start..q_start + head_dim],
                        layer.q_norm_weight.data,
                        head_dim,
                        eps,
                    );
                }
                for h in 0..num_kv_heads {
                    let k_start = pos * kv_dim + h * head_dim;
                    rms_norm(
                        &mut k_buf[k_start..k_start + head_dim],
                        layer.k_norm_weight.data,
                        head_dim,
                        eps,
                    );
                }
            }

            // --- Apply RoPE on contiguous Q and K buffers ---
            for pos in 0..seq_len {
                for h in 0..num_heads {
                    let q_start = pos * q_dim + h * head_dim;
                    self.rope
                        .apply(&mut q_buf[q_start..q_start + head_dim], pos);
                }
                for h in 0..num_kv_heads {
                    let k_start = pos * kv_dim + h * head_dim;
                    self.rope
                        .apply(&mut k_buf[k_start..k_start + head_dim], pos);
                }
            }

            // --- Grouped-Query Attention (batched per KV-head) ---
            let attn_out = &mut attn_out[..seq_len * q_dim];
            let gqa_cfg = GqaConfig {
                num_heads,
                num_kv_heads,
                head_dim,
            };
            apply_gqa_attention(
                &q_buf[..seq_len * q_dim],
                &k_buf[..seq_len * kv_dim],
                &v_buf[..seq_len * kv_dim],
                attn_out,
                seq_len,
                gqa_cfg,
                gqa_scratch,
            );

            // Output projection: maps q_dim back to hidden_size.
            matmul_bt(
                attn_out,
                layer.o_proj_weight.data,
                hidden,
                seq_len,
                q_dim,
                hidden_size,
            );

            // Residual connection.
            for i in 0..seq_len * hidden_size {
                hidden[i] += residual[i];
            }

            // --- Pre-FFN RMSNorm ---
            residual.copy_from_slice(hidden);
            rms_norm(
                hidden,
                layer.post_attention_layernorm_weight.data,
                hidden_size,
                eps,
            );

            // --- Fused SwiGLU FFN: 1 GEMM instead of 2 for gate+up ---
            let gate_up_buf = &mut gate_up_buf[..seq_len * gate_up_dim];
            matmul_bt(
                hidden,
                &layer.fused_gate_up,
                gate_up_buf,
                seq_len,
                hidden_size,
                gate_up_dim,
            );

            // Scatter fused gate+up to separate contiguous buffers.
            let gate_buf = &mut gate_buf[..seq_len * inter];
            let up_buf = &mut up_buf[..seq_len * inter];
            for i in 0..seq_len {
                let gu_row = i * gate_up_dim;
                gate_buf[i * inter..(i + 1) * inter]
                    .copy_from_slice(&gate_up_buf[gu_row..gu_row + inter]);
                up_buf[i * inter..(i + 1) * inter]
                    .copy_from_slice(&gate_up_buf[gu_row + inter..gu_row + gate_up_dim]);
            }

            // SiLU + elementwise mul on contiguous buffers.
            silu_inplace(&mut gate_buf[..seq_len * inter]);
            elementwise_mul(&mut gate_buf[..seq_len * inter], &up_buf[..seq_len * inter]);

            // Down projection on contiguous gate_buf.
            let ffn_out = &mut ffn_out[..seq_len * hidden_size];
            matmul_bt(
                gate_buf,
                layer.down_proj_weight.data,
                ffn_out,
                seq_len,
                inter,
                hidden_size,
            );

            // Residual connection.
            for i in 0..seq_len * hidden_size {
                hidden[i] = residual[i] + ffn_out[i];
            }
        }

        // Final RMSNorm.
        rms_norm(hidden, self.weights.norm_weight.data, hidden_size, eps);

        hidden.to_vec()
    }
}

/// FNV-1a hash of token IDs — fast, deterministic, good distribution.
fn hash_token_ids(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &id in ids {
        let bytes = id.to_le_bytes();
        for &b in &bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn is_qwen3_embedding_0_6b_dir(dir: &Path) -> bool {
    dir.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.contains("0.6b") || n.contains("0_6b"))
        .unwrap_or(false)
}

fn parse_qwen_config(path: &Path) -> Result<QwenConfig, InferenceError> {
    let text = std::fs::read_to_string(path)?;
    let get = |key: &str| -> Result<usize, InferenceError> {
        extract_json_usize(&text, key)
            .ok_or_else(|| InferenceError::Inference(format!("config.json missing key {key}")))
    };

    let rope_theta = extract_json_f64(&text, "rope_theta").unwrap_or(1_000_000.0);
    let rms_norm_eps = extract_json_f64(&text, "rms_norm_eps")
        .map(|v| v as f32)
        .unwrap_or(1e-6);

    let num_attention_heads = get("num_attention_heads")?;
    let hidden_size = get("hidden_size")?;
    // head_dim may be explicit in config, or inferred from hidden_size / num_heads.
    let head_dim =
        extract_json_usize(&text, "head_dim").unwrap_or(hidden_size / num_attention_heads);

    Ok(QwenConfig {
        vocab_size: get("vocab_size")?,
        hidden_size,
        num_hidden_layers: get("num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads: get("num_key_value_heads")?,
        head_dim,
        intermediate_size: get("intermediate_size")?,
        max_position_embeddings: get("max_position_embeddings")?,
        rms_norm_eps,
        rope_theta,
    })
}

fn extract_json_usize(text: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let idx = text.find(&needle)?;
    let rest = &text[idx + needle.len()..];
    let colon = rest.find(':')?;
    let value = rest[colon + 1..].trim_start();
    let end = value
        .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        .unwrap_or(value.len());
    value[..end].trim().parse().ok()
}

fn extract_json_f64(text: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let idx = text.find(&needle)?;
    let rest = &text[idx + needle.len()..];
    let colon = rest.find(':')?;
    let value = rest[colon + 1..].trim_start();
    let end = value
        .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        .unwrap_or(value.len());
    value[..end].trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_config_dimensions() {
        let cfg = QwenConfig::qwen3_embedding_0_6b();
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.q_dim(), 2048); // 16 * 128
        assert_eq!(cfg.kv_dim(), 1024); // 8 * 128
        assert_eq!(cfg.num_groups(), 2); // 16 / 8
    }

    #[test]
    fn test_parse_config_json() {
        let json = r#"{
            "vocab_size": 151669,
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 3072,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0
        }"#;
        assert_eq!(extract_json_usize(json, "hidden_size"), Some(1024));
        assert_eq!(extract_json_usize(json, "num_key_value_heads"), Some(8));
        assert_eq!(extract_json_usize(json, "head_dim"), None); // not in this test JSON
        assert_eq!(extract_json_f64(json, "rope_theta"), Some(1_000_000.0));
    }

    #[test]
    #[ignore] // Requires model files: set LATTICE_INFERENCE_MODEL_DIR
    fn test_qwen_long_text_bench() {
        let model_dir = match std::env::var("LATTICE_INFERENCE_MODEL_DIR") {
            Ok(v) => v,
            Err(_) => return,
        };
        let model = QwenModel::from_directory(std::path::Path::new(&model_dir)).unwrap();
        // Warmup
        let _ = model.encode("warmup").unwrap();

        let long_text = "The Qwen3 embedding model uses a decoder-only transformer architecture with grouped query attention and rotary position embeddings for efficient multilingual text representation. ".repeat(25);
        println!("Long text: {} chars", long_text.len());

        let t = std::time::Instant::now();
        let n = 3;
        for _ in 0..n {
            let _ = model.encode(&long_text).unwrap();
        }
        let ms = t.elapsed().as_millis() as f64 / n as f64;
        println!("Long encode: {ms:.0}ms avg ({n} runs)");
    }

    #[test]
    #[ignore] // Requires model files: set LATTICE_INFERENCE_MODEL_DIR
    fn test_qwen_multilingual() {
        let model_dir = match std::env::var("LATTICE_INFERENCE_MODEL_DIR") {
            Ok(v) => v,
            Err(_) => return,
        };
        let model = QwenModel::from_directory(std::path::Path::new(&model_dir)).unwrap();

        let pairs = [
            ("记得吃药", "remember to take medicine"),
            ("今天天气不错", "the weather is nice today"),
            ("量子物理", "quantum physics"),
            ("我想吃火锅", "I want to eat hotpot"),
        ];

        println!("\n--- Multilingual cosine similarity ---");
        for (zh, en) in &pairs {
            let e_zh = model.encode(zh).unwrap();
            let e_en = model.encode(en).unwrap();
            let cos: f32 = e_zh.iter().zip(e_en.iter()).map(|(a, b)| a * b).sum();
            println!("{zh:12} <-> {en:35} = {cos:.4}");
        }

        // Cross-pair: unrelated should be low.
        let pill_zh = model.encode("记得吃药").unwrap();
        let physics_en = model.encode("quantum physics").unwrap();
        let cross: f32 = pill_zh
            .iter()
            .zip(physics_en.iter())
            .map(|(a, b)| a * b)
            .sum();
        println!(
            "记得吃药       <-> quantum physics                      = {cross:.4} (unrelated)"
        );

        // Related pair should be higher than unrelated.
        let pill_en = model.encode("remember to take medicine").unwrap();
        let related: f32 = pill_zh.iter().zip(pill_en.iter()).map(|(a, b)| a * b).sum();
        assert!(
            related > cross + 0.1,
            "related pair ({related:.4}) should score higher than unrelated ({cross:.4})"
        );
    }

    #[test]
    #[ignore] // Requires model files: set LATTICE_INFERENCE_MODEL_DIR
    fn test_qwen_encode_real_model() {
        let model_dir = match std::env::var("LATTICE_INFERENCE_MODEL_DIR") {
            Ok(v) => v,
            Err(_) => return,
        };

        let model = QwenModel::from_directory(std::path::Path::new(&model_dir)).unwrap();
        assert_eq!(model.config().hidden_size, 1024);
        assert_eq!(model.dimensions(), 1024);

        let embedding = model.encode("hello world").unwrap();
        assert_eq!(embedding.len(), 1024);

        // Verify L2-normalized.
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "embedding should be L2-normalized, got norm={norm}"
        );

        // Different texts should produce different embeddings.
        let emb2 = model.encode("quantum physics").unwrap();
        let dot: f32 = embedding.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot < 0.99,
            "different texts should have different embeddings, cosine={dot}"
        );

        println!(
            "embed dim={}, norm={norm:.4}, cosine_sim={dot:.4}",
            embedding.len()
        );
    }

    #[test]
    fn test_default_inference_config_matches_06b() {
        let cfg = ModelInferenceConfig::default();
        assert_eq!(cfg.eos_token_id, 151_643);
        assert_eq!(cfg.rope_table_max_seq_len, 8192);
        assert_eq!(cfg.gpu_max_seq_len, 2048);
    }

    #[test]
    fn test_load_inference_config_from_json() {
        let json =
            r#"{"eos_token_id": 151645, "rope_table_max_seq_len": 32768, "gpu_max_seq_len": 4096}"#;
        let cfg: ModelInferenceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.eos_token_id, 151_645);
        assert_eq!(cfg.rope_table_max_seq_len, 32_768);
        assert_eq!(cfg.gpu_max_seq_len, 4096);
    }

    #[test]
    fn test_partial_config_uses_serde_defaults() {
        let json = r#"{"gpu_max_seq_len": 4096}"#;
        let cfg: ModelInferenceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.eos_token_id, 151_643,
            "unset eos_token_id should use 0.6B default"
        );
        assert_eq!(
            cfg.rope_table_max_seq_len, 8192,
            "unset rope_table_max_seq_len should use 0.6B default"
        );
        assert_eq!(
            cfg.gpu_max_seq_len, 4096,
            "explicitly set field should override default"
        );

        let json2 = r#"{"eos_token_id": 151645}"#;
        let cfg2: ModelInferenceConfig = serde_json::from_str(json2).unwrap();
        assert_eq!(cfg2.eos_token_id, 151_645, "set field should override");
        assert_eq!(
            cfg2.rope_table_max_seq_len, 8192,
            "unset field should use default"
        );
        assert_eq!(cfg2.gpu_max_seq_len, 2048, "unset field should use default");

        let empty: ModelInferenceConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.eos_token_id, 151_643);
        assert_eq!(empty.rope_table_max_seq_len, 8192);
        assert_eq!(empty.gpu_max_seq_len, 2048);
    }

    #[test]
    fn test_missing_config_non_06b_errors() {
        let tmp = std::env::temp_dir().join("lattice_test_non06b_err");
        let model_dir = tmp.join("qwen3-embedding-4b");
        std::fs::create_dir_all(&model_dir).unwrap();

        assert!(
            !is_qwen3_embedding_0_6b_dir(&model_dir),
            "4b dir must NOT be detected as 0.6B"
        );

        let config_path = model_dir.join("config.json");
        let result = parse_qwen_config(&config_path);
        assert!(
            result.is_err(),
            "parse_qwen_config must error when config.json does not exist"
        );

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("No such file")
                || err_msg.contains("not found")
                || err_msg.contains("os error"),
            "error should indicate missing file, got: {err_msg}"
        );

        let dir_06b = tmp.join("qwen3-embedding-0.6b");
        std::fs::create_dir_all(&dir_06b).unwrap();
        assert!(
            is_qwen3_embedding_0_6b_dir(&dir_06b),
            "0.6b dir must be detected as 0.6B"
        );

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_is_qwen3_embedding_06b_dir_detection() {
        use std::path::Path;
        assert!(is_qwen3_embedding_0_6b_dir(Path::new(
            "/models/qwen3-embedding-0.6b"
        )));
        assert!(is_qwen3_embedding_0_6b_dir(Path::new(
            "/models/qwen3_embedding_0_6b"
        )));
        assert!(
            !is_qwen3_embedding_0_6b_dir(Path::new("/models/Qwen3-Embedding-0.6B")),
            "case-sensitive: uppercase 0.6B does not match lowercase check"
        );
        assert!(!is_qwen3_embedding_0_6b_dir(Path::new(
            "/models/qwen3-embedding-4b"
        )));
        assert!(!is_qwen3_embedding_0_6b_dir(Path::new(
            "/models/some-other-model"
        )));
        assert!(!is_qwen3_embedding_0_6b_dir(Path::new("/")));
    }

    #[test]
    fn test_eos_token_id_wired_correctly() {
        let tmp = std::env::temp_dir().join("lattice_test_eos_wired");
        std::fs::create_dir_all(&tmp).unwrap();

        let custom_eos: u32 = 151_645;
        let json = format!(r#"{{"eos_token_id": {custom_eos}}}"#);
        std::fs::write(tmp.join("inference_config.json"), &json).unwrap();

        let cfg = ModelInferenceConfig::load(&tmp);
        assert_eq!(
            cfg.eos_token_id, custom_eos,
            "load() must read eos_token_id from inference_config.json"
        );
        assert_eq!(
            cfg.rope_table_max_seq_len, 8192,
            "unset rope field must get 0.6B default"
        );
        assert_eq!(
            cfg.gpu_max_seq_len, 2048,
            "unset gpu field must get 0.6B default"
        );

        let full_json =
            r#"{"eos_token_id": 99999, "rope_table_max_seq_len": 16384, "gpu_max_seq_len": 1024}"#;
        std::fs::write(tmp.join("inference_config.json"), full_json).unwrap();
        let cfg2 = ModelInferenceConfig::load(&tmp);
        assert_eq!(cfg2.eos_token_id, 99_999);
        assert_eq!(cfg2.rope_table_max_seq_len, 16_384);
        assert_eq!(cfg2.gpu_max_seq_len, 1024);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_model_inference_config_load_missing_file() {
        let dir = std::env::temp_dir().join("lattice_test_missing_cfg");
        std::fs::create_dir_all(&dir).ok();
        let cfg = ModelInferenceConfig::load(&dir);
        assert_eq!(cfg.eos_token_id, 151_643);
        assert_eq!(cfg.rope_table_max_seq_len, 8192);
        assert_eq!(cfg.gpu_max_seq_len, 2048);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_model_inference_config_load_malformed_json() {
        let tmp = std::env::temp_dir().join("lattice_test_malformed_cfg");
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("inference_config.json"), "not valid json {{{").unwrap();
        let cfg = ModelInferenceConfig::load(&tmp);
        assert_eq!(
            cfg.eos_token_id, 151_643,
            "malformed JSON must fall back to defaults"
        );
        assert_eq!(cfg.rope_table_max_seq_len, 8192);
        assert_eq!(cfg.gpu_max_seq_len, 2048);
        std::fs::remove_dir_all(&tmp).ok();
    }
}
