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
use crate::tokenizer::common::{EmbeddingTokenCounts, TokenizedInput, Tokenizer, load_tokenizer};
use crate::weights::{QwenWeights, SafetensorsFile, ShardedQwenBacking, ShardedSafetensors};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

/// Backing storage for Qwen model weights — either a single mmap'd file or
/// an owned heap allocation for sharded checkpoints.
///
/// `QwenModel` wraps both `weights` and `_storage` in `ManuallyDrop` and
/// implements `Drop` to drop `weights` before `_storage`, keeping tensor
/// slice references valid regardless of field declaration order.
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

    /// Base model revision the embedding cache was produced against. Checked
    /// at `cache_load` time so a cache built for a different model revision
    /// fails closed instead of poisoning lookups with stale embeddings.
    ///
    /// If left unset (or explicitly `"none"`) in `inference_config.json`,
    /// `ModelInferenceConfig::load` derives it via `derive_base_model_rev`:
    /// a boundary-sampled fingerprint over `config.json` plus the weight
    /// shard filenames/lengths/boundary bytes, so two checkpoints that share
    /// `config.json` but differ in weights (e.g. a HF re-upload with fixed
    /// weights and unchanged config) derive distinct revisions instead of a
    /// vacuously-matching one. An explicit non-`"none"` value in the JSON
    /// always wins over derivation.
    #[serde(default = "default_cache_compat_rev")]
    pub base_model_rev: String,

    /// Tokenizer revision the embedding cache was produced against, checked
    /// the same way as `base_model_rev`.
    ///
    /// If left unset (or explicitly `"none"`) in `inference_config.json`,
    /// `ModelInferenceConfig::load` derives it as `sha256:<first 16 hex
    /// chars of the SHA-256 of tokenizer.json>` so the compat check is
    /// non-vacuous by default. An explicit non-`"none"` value in the JSON
    /// always wins over derivation.
    #[serde(default = "default_cache_compat_rev")]
    pub tokenizer_rev: String,
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
fn default_cache_compat_rev() -> String {
    "none".to_string()
}

impl Default for ModelInferenceConfig {
    fn default() -> Self {
        Self {
            eos_token_id: default_eos_token_id(),
            rope_table_max_seq_len: default_rope_table_max_seq_len(),
            gpu_max_seq_len: default_gpu_max_seq_len(),
            base_model_rev: default_cache_compat_rev(),
            tokenizer_rev: default_cache_compat_rev(),
        }
    }
}

impl ModelInferenceConfig {
    pub fn load(model_dir: &Path) -> Self {
        let path = model_dir.join("inference_config.json");
        let mut cfg = match std::fs::read_to_string(&path) {
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
        };

        // An explicit value in inference_config.json always wins; only derive
        // a real revision when the field is still at the "unset" sentinel, so
        // the manifest compat check in `cache_load` is non-vacuous by default
        // instead of comparing "none" == "none" on every real deployment.
        if cfg.base_model_rev == default_cache_compat_rev()
            && let Some(rev) = derive_base_model_rev(model_dir)
        {
            cfg.base_model_rev = rev;
        }
        if cfg.tokenizer_rev == default_cache_compat_rev()
            && let Some(rev) = derive_cache_compat_rev(&model_dir.join("tokenizer.json"))
        {
            cfg.tokenizer_rev = rev;
        }

        cfg
    }
}

/// Derive a short, stable revision tag from a file's contents:
/// `sha256:<first 16 hex chars of the file's SHA-256>`. Used to give
/// `base_model_rev`/`tokenizer_rev` a real, content-addressed value when
/// `inference_config.json` does not set one explicitly. Returns `None` if the
/// file is missing or unreadable — callers must leave the field at its
/// existing sentinel in that case, since a missing derivation source is not a
/// construction failure.
fn derive_cache_compat_rev(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let full_hex = embedding_cache_sha256_hex(&bytes);
    Some(format!("sha256:{}", &full_hex[..16]))
}

/// Number of bytes sampled from the head and tail of a weight shard larger
/// than [`WEIGHT_FINGERPRINT_INLINE_CAP`] when folding it into
/// `derive_base_model_rev`.
const WEIGHT_FINGERPRINT_SAMPLE_LEN: u64 = 1024 * 1024; // 1 MiB
/// Weight shards at or under this size are hashed in full; larger shards are
/// hashed via [`WEIGHT_FINGERPRINT_SAMPLE_LEN`]-byte head/tail samples only,
/// so `derive_base_model_rev` never reads a whole multi-GB checkpoint shard.
const WEIGHT_FINGERPRINT_INLINE_CAP: u64 = 2 * 1024 * 1024; // 2 MiB

/// Resolve the model directory's weight shard filenames for
/// `derive_base_model_rev`, in the same precedence order the weight loader
/// itself uses: a single `model.safetensors` if present — matching
/// `QwenModel::from_directory`, which takes the single-file path first when
/// both a single file and a sharded index exist — else a sharded
/// `model.safetensors.index.json` if present (its `weight_map` values,
/// deduplicated and lexically sorted so the fingerprint is stable regardless
/// of key iteration order), else no weight files (config-only derivation).
fn resolve_weight_fingerprint_files(dir: &Path) -> Vec<String> {
    if dir.join("model.safetensors").is_file() {
        return vec!["model.safetensors".to_string()];
    }
    let index_path = dir.join("model.safetensors.index.json");
    if let Ok(index_bytes) = std::fs::read(&index_path)
        && let Ok(index_json) = serde_json::from_slice::<serde_json::Value>(&index_bytes)
        && let Some(weight_map) = index_json.get("weight_map").and_then(|v| v.as_object())
    {
        let files: std::collections::BTreeSet<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        if !files.is_empty() {
            return files.into_iter().collect();
        }
    }
    Vec::new()
}

/// Fold one weight shard's filename, byte length, and boundary-sampled
/// content into `hasher`, as part of `derive_base_model_rev`. Shards at or
/// under [`WEIGHT_FINGERPRINT_INLINE_CAP`] are hashed whole; larger shards
/// are hashed via their first and last [`WEIGHT_FINGERPRINT_SAMPLE_LEN`]
/// bytes only. Uses buffered/seeked reads throughout — never loads a whole
/// multi-GB shard into memory.
fn fold_weight_file_into_hasher(
    hasher: &mut Sha256,
    dir: &Path,
    file_name: &str,
) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    // Index-declared shard names are untrusted checkpoint content (#1069);
    // a containment failure degrades to the config-only derivation via this
    // function's existing error path, same as any other unreadable shard.
    let path = crate::weights::contained_shard_path(dir, file_name)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let opened = crate::weights::mmap_trust::open_regular_file_no_hang(&path)
        .map_err(std::io::Error::other)?;
    let mut file = std::io::BufReader::new(opened);
    let len = file.get_ref().metadata()?.len();

    hasher.update(file_name.as_bytes());
    hasher.update(len.to_le_bytes());

    if len <= WEIGHT_FINGERPRINT_INLINE_CAP {
        let mut buf = Vec::with_capacity(len as usize);
        file.read_to_end(&mut buf)?;
        hasher.update(&buf);
        return Ok(());
    }

    let sample_len = WEIGHT_FINGERPRINT_SAMPLE_LEN as usize;
    let mut head = vec![0u8; sample_len];
    file.read_exact(&mut head)?;
    hasher.update(&head);

    file.seek(SeekFrom::Start(len - WEIGHT_FINGERPRINT_SAMPLE_LEN))?;
    let mut tail = vec![0u8; sample_len];
    file.read_exact(&mut tail)?;
    hasher.update(&tail);

    Ok(())
}

/// Derive `base_model_rev` from `config.json` plus a boundary-sampled
/// fingerprint of the model directory's weight files, so two checkpoints
/// that share `config.json` (and `tokenizer.json`) but differ in weights —
/// e.g. a HF re-upload that fixes weights but leaves config unchanged —
/// derive distinct revisions instead of an identical one that would let a
/// stale embedding cache pass the `cache_load` compat check.
///
/// This is a **boundary-sampled revision fingerprint, not an
/// adversarial-collision-resistant content hash**: for each resolved weight
/// shard (see [`resolve_weight_fingerprint_files`]) it folds in the shard's
/// filename, byte length, and first/last 1 MiB of content, not every byte
/// of every shard. It is designed to detect checkpoint revision drift,
/// including weight-only updates, cheaply — it is not designed to resist a
/// file crafted to preserve this fingerprint while corrupting unsampled
/// bytes. The integrity boundary for the cache payload itself remains
/// `payload_sha256` in the embedding cache manifest, which does hash every
/// payload byte.
///
/// Returns `None` only when `config.json` itself is missing or unreadable
/// (mirroring `derive_cache_compat_rev`), in which case the caller leaves
/// `base_model_rev` at its existing sentinel. A read failure partway
/// through a weight file degrades gracefully to the config-only derivation
/// (`derive_cache_compat_rev`) rather than failing `ModelInferenceConfig`
/// construction.
fn derive_base_model_rev(dir: &Path) -> Option<String> {
    let config_path = dir.join("config.json");
    let config_bytes = std::fs::read(&config_path).ok()?;

    let weight_files = resolve_weight_fingerprint_files(dir);

    let mut hasher = Sha256::new();
    hasher.update(&config_bytes);

    for file_name in &weight_files {
        if fold_weight_file_into_hasher(&mut hasher, dir, file_name).is_err() {
            return derive_cache_compat_rev(&config_path);
        }
    }

    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest.as_slice() {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    Some(format!("sha256:{}", &hex[..16]))
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

/// Embedding cache file magic, identifying the versioned manifest format
/// below. Distinguishes governed cache files from the pre-manifest raw
/// record stream, which is now rejected rather than silently parsed.
const EMBEDDING_CACHE_MAGIC: &[u8; 8] = b"LQECACHE";
/// Current embedding cache manifest format version.
const EMBEDDING_CACHE_VERSION: u32 = 1;
/// Upper bound on the manifest header length, rejected before JSON parsing
/// so a corrupt or hostile `manifest_len` cannot drive an oversized read.
const EMBEDDING_CACHE_MANIFEST_CAP: u32 = 64 * 1024;

/// Process-global counter folded into `cache_save`'s temp file name so two
/// threads in the same process (e.g. two `NativeEmbeddingService` instances
/// sharing a global cache path) never race on the same `<name>.tmp.<pid>`
/// path — the previous PID-only name was unique across processes but not
/// within one, letting concurrent saves overwrite or rename over each
/// other's temp file.
static EMBEDDING_CACHE_TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bound on retries when a `<name>.tmp.<pid>.<seq>` path unexpectedly
/// already exists (e.g. a leftover from a killed process that reused a PID
/// and hit the same sequence number). `cache_save` fails closed rather than
/// looping forever once this many attempts collide.
const EMBEDDING_CACHE_TMP_MAX_ATTEMPTS: u32 = 16;

/// Integrity and compatibility manifest embedded in the embedding cache
/// file header, mirroring the LoRA adapter manifest pattern
/// (`crates/tune/src/lora/manifest.rs`): a content hash plus compatibility
/// fields checked before any cache record is trusted.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct EmbeddingCacheManifest {
    version: u32,
    payload_sha256: String,
    base_model_rev: String,
    tokenizer_rev: String,
    embedding_dim: usize,
    entry_count: usize,
}

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
    // Both fields are `ManuallyDrop` so that `Drop for QwenModel` can enforce
    // explicit drop order: `weights` before `_storage`. This is independent of
    // field declaration order — a reorder cannot cause use-after-free.
    weights: ManuallyDrop<QwenWeights<'static>>,
    _storage: ManuallyDrop<SafetensorsStorage>,
}

impl Drop for QwenModel {
    fn drop(&mut self) {
        // SAFETY: `weights` holds slice references into the data owned by `_storage`.
        // We drop `weights` first to release those references, then drop `_storage`.
        // Using `ManuallyDrop` here makes this ordering explicit and independent of
        // field declaration order — a future field reorder cannot cause use-after-free.
        unsafe {
            ManuallyDrop::drop(&mut self.weights);
            ManuallyDrop::drop(&mut self._storage);
        }
    }
}

/// ADR-080 cluster C1: apply the causal mask then the fail-closed softmax
/// normalization to a single `[seq_len]` attention-score row in place, where
/// `qi` is this row's query position and `scale` is the pre-softmax score
/// scale (`1 / sqrt(head_dim)`).
///
/// This is the shared body of what used to be two byte-identical inline
/// softmax duplicates in `forward_all_layers` and `forward_profiled` (#780);
/// extracting it here both routes the fail-closed decision through the
/// shared `attention::softmax_row` contract and removes the duplication
/// between the two call sites in this file.
fn causal_softmax_row(row: &mut [f32], qi: usize, scale: f32) {
    for (ki, v) in row.iter_mut().enumerate() {
        if ki > qi {
            *v = f32::NEG_INFINITY;
        } else {
            *v *= scale;
        }
    }
    let (max_val, any_nan) = crate::attention::softmax_row::row_max_and_any_nan(row);
    if crate::attention::softmax_row::row_fails_closed_pre_exp(max_val, any_nan) {
        row.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    crate::attention::softmax_row::finalize_row(row, sum);
}

/// The id sequence `QwenModel` embeds for one tokenizer result, with its token counts.
///
/// A tokenizer that already ended the sequence with `eos` is used as is; otherwise the model
/// appends `eos`, or overwrites the last token with it when the sequence already fills
/// `max_len`. `embedded` is the length of the ids built here, and `before_truncation` is the
/// tokenizer's pre-truncation length plus the end token when the model had to supply it.
fn embedding_sequence(
    input: &TokenizedInput,
    max_len: usize,
    eos: u32,
) -> (Vec<u32>, EmbeddingTokenCounts) {
    let mut ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();

    // Only append EOS if the tokenizer has not already done so. Qwen3's
    // tokenizer.json post_processor (TemplateProcessing) appends <|endoftext|>
    // automatically; a second append would pool from a spurious extra EOS
    // token instead of the real last-content token, causing embedding divergence.
    let tokenizer_ended = ids.last() == Some(&eos);
    if !tokenizer_ended {
        if ids.len() < max_len {
            ids.push(eos);
        } else if let Some(last) = ids.last_mut() {
            *last = eos;
        }
    }

    let counts = EmbeddingTokenCounts {
        before_truncation: input.pre_truncation_len + usize::from(!tokenizer_ended),
        embedded: ids.len(),
    };
    (ids, counts)
}

fn effective_context_len(
    config: &QwenConfig,
    inference_config: &ModelInferenceConfig,
    tokenizer_max_seq_len: usize,
) -> Result<usize, InferenceError> {
    let max_context = config
        .max_position_embeddings
        .min(inference_config.rope_table_max_seq_len)
        .min(tokenizer_max_seq_len);
    if max_context == 0 {
        return Err(InferenceError::InvalidInput(format!(
            "Qwen effective context must be non-zero: max_position_embeddings={}, \
             rope_table_max_seq_len={}, tokenizer_max_seq_len={tokenizer_max_seq_len}",
            config.max_position_embeddings, inference_config.rope_table_max_seq_len
        )));
    }
    Ok(max_context)
}

impl QwenModel {
    /// **Unstable**: load Qwen3 model from a local directory; path convention may change.
    ///
    /// Single-file and sharded checkpoints validate every accessed tensor
    /// before `Self` is assembled. If loading fails partway through the
    /// checkpoint, local backing and fused-weight state are dropped and no
    /// partially built model is returned.
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

        let max_context =
            effective_context_len(&config, &inference_config, tokenizer.max_seq_len())?;
        let gpu_max_seq_len = inference_config.gpu_max_seq_len.min(max_context);

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
            let metal = if crate::env_switch_enabled("LATTICE_NO_GPU") {
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

            // SAFETY: The safetensors backing is stored in `_storage`; `QwenModel::drop`
            // explicitly drops `weights` before `_storage`, independent of field order.
            let weights: QwenWeights<'static> = unsafe { std::mem::transmute(weights_tmp) };

            let rope = RopeTable::new(config.head_dim, max_context, config.rope_theta);

            Ok(Self {
                config,
                inference_config,
                tokenizer,
                rope,
                output_dim: None,
                buffers: Mutex::new(ForwardBuffers::new()),
                cache: Mutex::new(HashMap::with_capacity(1024)),
                metal,
                weights: ManuallyDrop::new(weights),
                _storage: ManuallyDrop::new(SafetensorsStorage::Single(safetensors)),
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
            let metal = if crate::env_switch_enabled("LATTICE_NO_GPU") {
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

            // SAFETY: `weights_tmp` contains slices into `backing`. `backing` is moved
            // into `_storage`, and `QwenModel::drop` explicitly drops `weights` before
            // `_storage`, independent of field declaration order.
            let weights: QwenWeights<'static> = weights_tmp;

            let rope = RopeTable::new(config.head_dim, max_context, config.rope_theta);

            Ok(Self {
                config,
                inference_config,
                tokenizer,
                rope,
                output_dim: None,
                buffers: Mutex::new(ForwardBuffers::new()),
                cache: Mutex::new(HashMap::with_capacity(1024)),
                metal,
                weights: ManuallyDrop::new(weights),
                _storage: ManuallyDrop::new(SafetensorsStorage::Sharded(backing)),
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

    /// Load embedding cache from a versioned binary file with an integrity
    /// manifest. Fails closed on any magic/version/hash/compatibility
    /// mismatch instead of loading a partial or stale cache; a missing file
    /// is not an error since no artifact was expected to be loaded.
    pub fn cache_load(&self, path: &Path) -> Result<usize, InferenceError> {
        let data = match read_embedding_cache_file(path, self.dimensions()) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(InferenceError::Inference(format!("cache load: {e}"))),
        };
        let entries = parse_embedding_cache_file(
            &data,
            self.dimensions(),
            &self.inference_config.base_model_rev,
            &self.inference_config.tokenizer_rev,
        )?;

        let count = entries.len();
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (hash, floats) in entries {
            cache.insert(hash, floats);
        }
        Ok(count)
    }

    /// Save embedding cache to a binary file, prefixed with an integrity and
    /// compatibility manifest (magic, version, payload SHA-256, base model
    /// and tokenizer revisions) so `cache_load` can verify it before trust.
    ///
    /// Writes to a temp file in the same directory and renames it over the
    /// target, so a crash or kill mid-write cannot leave a truncated file at
    /// `path` — the fail-closed loader would otherwise reject that truncated
    /// file forever. Mirrors the temp-file + rename pattern in
    /// `forward::metal_qwen35::write_merged_qkvz`.
    pub fn cache_save(&self, path: &Path) -> Result<usize, InferenceError> {
        let cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let count = cache.len();
        let payload = serialize_embedding_cache_payload(&cache);
        let file_bytes = wrap_embedding_cache_payload(
            payload,
            &self.inference_config.base_model_rev,
            &self.inference_config.tokenizer_rev,
            self.dimensions(),
            count,
        )?;
        write_embedding_cache_file_atomic(path, &file_bytes)?;
        Ok(count)
    }

    fn tokenize_for_embedding(&self, text: &str) -> Result<(Vec<u32>, usize), InferenceError> {
        let input = self.tokenizer.tokenize(text);
        let (ids, _) = embedding_sequence(
            &input,
            self.tokenizer.max_seq_len(),
            self.inference_config.eos_token_id,
        );

        let seq_len = ids.len();
        let rope_capacity = self.rope.max_positions();
        if seq_len > rope_capacity {
            return Err(InferenceError::InvalidInput(format!(
                "Qwen input length {seq_len} exceeds RoPE capacity {rope_capacity}"
            )));
        }
        Ok((ids, seq_len))
    }

    /// **Unstable**: token counts of `text` as `encode` and `encode_batch` consume it.
    ///
    /// `before_truncation` comes from the tokenizer's own pre-truncation length, so one
    /// tokenization yields both figures. Both count the end-of-sequence token every
    /// embedded sequence carries, whether the tokenizer appends it or the model does.
    /// The limit is the tokenizer's sequence limit.
    pub fn embedding_token_counts(&self, text: &str) -> EmbeddingTokenCounts {
        embedding_sequence(
            &self.tokenizer.tokenize(text),
            self.tokenizer.max_seq_len(),
            self.inference_config.eos_token_id,
        )
        .1
    }

    /// Encode a single text into an embedding vector.
    ///
    /// Uses an internal LRU-style cache keyed by token IDs. Repeated texts
    /// return the cached embedding in <1us instead of running the ~100ms
    /// forward pass.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let (ids, seq_len) = self.tokenize_for_embedding(text)?;

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
        let (ids, seq_len) = self.tokenize_for_embedding(text)?;
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
                // ADR-080 C1: shared `causal_softmax_row` helper (defined above
                // `impl QwenModel`) replaces the bare `sum > 0.0` check (which
                // had no `else` and left a NaN/+inf-poisoned row un-zeroed,
                // #780) and deduplicates what used to be two byte-identical
                // secondary-prefill softmax copies in this file.
                for qi in 0..seq_len {
                    let row = &mut scores_head[qi * seq_len..(qi + 1) * seq_len];
                    causal_softmax_row(row, qi, scale);
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
            let (ids, seq_len) = self.tokenize_for_embedding(text)?;
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
        let (ids, seq_len) = self.tokenize_for_embedding(text)?;
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
                // ADR-080 C1: shared `causal_softmax_row` helper (defined above
                // `impl QwenModel`) replaces the bare `sum > 0.0` check (which
                // had no `else` and left a NaN/+inf-poisoned row un-zeroed,
                // #780) and deduplicates what used to be two byte-identical
                // secondary-prefill softmax copies in this file.
                for qi in 0..seq_len {
                    let row = &mut scores_head[qi * seq_len..(qi + 1) * seq_len];
                    causal_softmax_row(row, qi, scale);
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

/// SHA-256 of `bytes`, formatted as lowercase hex — used for the embedding
/// cache payload integrity field, mirroring `download::sha256_hex`.
fn embedding_cache_sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest.as_slice() {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{byte:02x}");
    }
    hex
}

/// Upper bound on a well-formed embedding cache file for the given output
/// dimension: capacity-bound payload plus the bounded manifest header.
/// Stat-checked before reading so a corrupt or hostile file cannot drive an
/// unbounded allocation.
fn max_embedding_cache_file_len(dimensions: usize) -> u64 {
    let header = 8u64 + 4; // magic + manifest_len
    let manifest_cap = u64::from(EMBEDDING_CACHE_MANIFEST_CAP);
    // Saturating throughout: a hostile or absurd `dimensions` must saturate
    // to u64::MAX rather than wrap, so the cap this feeds into
    // `read_embedding_cache_file_bounded` stays a genuine upper bound instead
    // of wrapping around to a small value that would under-reject.
    let record_len = (dimensions as u64).saturating_mul(4).saturating_add(12);
    let max_payload = (EMBEDDING_CACHE_CAP as u64).saturating_mul(record_len);
    header
        .saturating_add(manifest_cap)
        .saturating_add(max_payload)
}

/// Read an embedding cache file from disk, rejecting anything larger than
/// `max_embedding_cache_file_len`. The stat is only a fast-path; the read is
/// itself bounded so a file that grows or is swapped between the stat and the
/// read cannot drive an allocation past the cap.
fn read_embedding_cache_file(path: &Path, dimensions: usize) -> std::io::Result<Vec<u8>> {
    read_embedding_cache_file_bounded(path, max_embedding_cache_file_len(dimensions))
}

/// Read a file, enforcing `max_len` as a hard cap. The stat is only a
/// fast-path; the read itself is bounded via `take(max_len + 1)` so a file
/// that grows or is swapped between the stat and the read cannot drive an
/// allocation past the cap.
fn read_embedding_cache_file_bounded(path: &Path, max_len: u64) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    // Fast-path: stat before open. A known-oversized file is refused without
    // allocating for its contents (defence-in-depth — the read below is also
    // bounded, so this stat is an optimisation, not the enforcing check).
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > max_len {
        return Err(std::io::Error::other(format!(
            "embedding cache file too large: {} bytes exceeds cap of {max_len} bytes",
            metadata.len()
        )));
    }
    // Bounded read: take max_len + 1 bytes so a file that grew after the stat
    // is still caught. If buf.len() exceeds max_len at read time, reject even
    // though the stat passed. Saturating so a max_len already at u64::MAX
    // (from a saturated `max_embedding_cache_file_len`) cannot wrap to 0 and
    // turn the bound into an unbounded read.
    let f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.take(max_len.saturating_add(1)).read_to_end(&mut buf)?;
    if buf.len() as u64 > max_len {
        return Err(std::io::Error::other(format!(
            "embedding cache file too large: read exceeds cap of {max_len} bytes"
        )));
    }
    Ok(buf)
}

/// Encode the embedding cache into the raw payload record stream:
/// repeated `[hash: u64 LE, dim: u32 LE, floats: f32 LE * dim]`.
fn serialize_embedding_cache_payload(cache: &HashMap<u64, Vec<f32>>) -> Vec<u8> {
    let payload_cap: usize = cache.values().map(|v| 12 + v.len() * 4).sum();
    let mut buf = Vec::with_capacity(payload_cap);
    for (&hash, embedding) in cache.iter() {
        buf.extend_from_slice(&hash.to_le_bytes());
        buf.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
        for &f in embedding {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
    buf
}

/// Wrap a raw payload record stream with the embedding cache manifest
/// header: `[magic: 8][manifest_len: u32 LE][manifest JSON][payload]`.
fn wrap_embedding_cache_payload(
    payload: Vec<u8>,
    base_model_rev: &str,
    tokenizer_rev: &str,
    dimensions: usize,
    entry_count: usize,
) -> Result<Vec<u8>, InferenceError> {
    let manifest = EmbeddingCacheManifest {
        version: EMBEDDING_CACHE_VERSION,
        payload_sha256: embedding_cache_sha256_hex(&payload),
        base_model_rev: base_model_rev.to_string(),
        tokenizer_rev: tokenizer_rev.to_string(),
        embedding_dim: dimensions,
        entry_count,
    };
    let manifest_bytes = serde_json::to_vec(&manifest).map_err(|e| {
        InferenceError::Inference(format!("embedding cache manifest serialize: {e}"))
    })?;
    if manifest_bytes.len() > EMBEDDING_CACHE_MANIFEST_CAP as usize {
        return Err(InferenceError::Inference(format!(
            "embedding cache manifest length {} exceeds cap of {EMBEDDING_CACHE_MANIFEST_CAP} bytes",
            manifest_bytes.len()
        )));
    }
    let mut file_bytes = Vec::with_capacity(8 + 4 + manifest_bytes.len() + payload.len());
    file_bytes.extend_from_slice(EMBEDDING_CACHE_MAGIC);
    file_bytes.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
    file_bytes.extend_from_slice(&manifest_bytes);
    file_bytes.extend_from_slice(&payload);
    Ok(file_bytes)
}

/// Open a temp file next to `path` under a name guaranteed unique for this
/// call: `<path's file name>.tmp.<pid>.<seq>`, where `seq` comes from the
/// process-global `EMBEDDING_CACHE_TMP_SEQ` counter. Opened with
/// `File::create_new` so the open itself fails closed instead of silently
/// truncating an existing file if the name is somehow already taken (e.g. a
/// leftover from a killed process that reused both the PID and the
/// in-process sequence number); on that collision, retries with the next
/// sequence number up to `EMBEDDING_CACHE_TMP_MAX_ATTEMPTS` times.
///
/// This is what makes concurrent `cache_save` calls in the same process
/// safe: the previous `<name>.tmp.<pid>` scheme was unique across processes
/// but not within one, so two threads saving to the same shared cache path
/// could open/write/rename over each other's temp file.
fn open_unique_embedding_cache_tmp_file(
    path: &Path,
) -> Result<(std::path::PathBuf, std::fs::File), InferenceError> {
    let base_tmp_name = path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_else(|| std::ffi::OsString::from("embedding_cache"));
    let pid = std::process::id();

    for _ in 0..EMBEDDING_CACHE_TMP_MAX_ATTEMPTS {
        let seq = EMBEDDING_CACHE_TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut name = base_tmp_name.clone();
        name.push(format!(".tmp.{pid}.{seq}"));
        let candidate = path.with_file_name(name);
        match std::fs::File::create_new(&candidate) {
            Ok(file) => return Ok((candidate, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(InferenceError::Inference(format!("cache save: {e}"))),
        }
    }

    Err(InferenceError::Inference(format!(
        "cache save: exhausted {EMBEDDING_CACHE_TMP_MAX_ATTEMPTS} unique temp file name attempts for {}",
        path.display()
    )))
}

/// Atomically write `file_bytes` to `path`: create the parent directory if
/// needed, write to a call-unique temp file in the same directory (see
/// [`open_unique_embedding_cache_tmp_file`]), then `fs::rename` it over
/// `path` so a crash or kill mid-write cannot leave a truncated file at
/// `path` — the fail-closed loader would otherwise reject that truncated
/// file forever. Mirrors the temp-file + rename pattern in
/// `forward::metal_qwen35::write_merged_qkvz`.
///
/// Shared by `QwenModel::cache_save` and its concurrency test so both
/// exercise the exact same collision-safe temp-naming and write logic.
fn write_embedding_cache_file_atomic(path: &Path, file_bytes: &[u8]) -> Result<(), InferenceError> {
    use std::io::Write as _;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| InferenceError::Inference(format!("cache save: {e}")))?;
    }

    let (tmp_path, mut tmp_file) = open_unique_embedding_cache_tmp_file(path)?;

    if let Err(e) = tmp_file.write_all(file_bytes) {
        drop(tmp_file);
        let _ = std::fs::remove_file(&tmp_path);
        return Err(InferenceError::Inference(format!("cache save: {e}")));
    }
    if let Err(e) = tmp_file.flush() {
        drop(tmp_file);
        let _ = std::fs::remove_file(&tmp_path);
        return Err(InferenceError::Inference(format!("cache save: {e}")));
    }
    drop(tmp_file);

    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(InferenceError::Inference(format!("cache save: {e}")));
    }
    Ok(())
}

/// Parse and fully validate an embedding cache file: magic, version,
/// manifest length cap, payload SHA-256, base-model/tokenizer compatibility,
/// dimension, finiteness, and exact payload consumption. Returns `Err`
/// before any record is trusted on the first mismatch found — the cache
/// (`self.cache`) is only mutated by the caller after this returns `Ok`.
fn parse_embedding_cache_file(
    data: &[u8],
    dimensions: usize,
    expected_base_model_rev: &str,
    expected_tokenizer_rev: &str,
) -> Result<Vec<(u64, Vec<f32>)>, InferenceError> {
    if data.len() < 8 || &data[..8] != EMBEDDING_CACHE_MAGIC {
        return Err(InferenceError::Inference(
            "embedding cache magic mismatch".to_string(),
        ));
    }
    if data.len() < 12 {
        return Err(InferenceError::Inference(
            "embedding cache manifest header truncated".to_string(),
        ));
    }
    // Bounds already ensured `data.len() >= 12`; copy the fixed-width field
    // without `try_into().expect()` (AGENTS.md bars `expect` in library code).
    let mut manifest_len_bytes = [0u8; 4];
    manifest_len_bytes.copy_from_slice(&data[8..12]);
    let manifest_len = u32::from_le_bytes(manifest_len_bytes);
    if manifest_len > EMBEDDING_CACHE_MANIFEST_CAP {
        return Err(InferenceError::Inference(format!(
            "embedding cache manifest length {manifest_len} exceeds cap of {EMBEDDING_CACHE_MANIFEST_CAP} bytes"
        )));
    }
    let manifest_start = 12usize;
    let manifest_end = manifest_start + manifest_len as usize;
    if data.len() < manifest_end {
        return Err(InferenceError::Inference(format!(
            "embedding cache manifest truncated: expected {manifest_len} bytes, found {}",
            data.len() - manifest_start
        )));
    }
    let manifest: EmbeddingCacheManifest =
        serde_json::from_slice(&data[manifest_start..manifest_end]).map_err(|e| {
            InferenceError::Inference(format!("embedding cache manifest parse error: {e}"))
        })?;
    if manifest.version != EMBEDDING_CACHE_VERSION {
        return Err(InferenceError::Inference(format!(
            "embedding cache version mismatch: expected {EMBEDDING_CACHE_VERSION}, found {}",
            manifest.version
        )));
    }

    let payload = &data[manifest_end..];
    let computed_sha256 = embedding_cache_sha256_hex(payload);
    if computed_sha256 != manifest.payload_sha256 {
        return Err(InferenceError::Inference(format!(
            "embedding cache payload_sha256 mismatch: expected {}, computed {computed_sha256}",
            manifest.payload_sha256
        )));
    }
    if manifest.base_model_rev != expected_base_model_rev {
        return Err(InferenceError::Inference(format!(
            "embedding cache base_model_rev mismatch: expected {expected_base_model_rev}, found {}",
            manifest.base_model_rev
        )));
    }
    if manifest.tokenizer_rev != expected_tokenizer_rev {
        return Err(InferenceError::Inference(format!(
            "embedding cache tokenizer_rev mismatch: expected {expected_tokenizer_rev}, found {}",
            manifest.tokenizer_rev
        )));
    }
    if manifest.embedding_dim != dimensions {
        return Err(InferenceError::Inference(format!(
            "embedding cache embedding_dim mismatch: expected {dimensions}, found {}",
            manifest.embedding_dim
        )));
    }

    if manifest.entry_count > EMBEDDING_CACHE_CAP {
        return Err(InferenceError::Inference(format!(
            "embedding cache entry_count {} exceeds cap of {EMBEDDING_CACHE_CAP}",
            manifest.entry_count
        )));
    }
    let mut entries = Vec::with_capacity(manifest.entry_count);
    let mut pos = 0usize;
    // Bounded by the manifest-declared entry count, not by payload length:
    // a payload with fewer usable bytes than declared fails as a truncated
    // record, and a payload with bytes left over after the declared entries
    // are consumed fails as trailing bytes below — a lone `entry_count`
    // mutation can no longer smuggle extra or missing records past the hash
    // check, since payload_sha256 covers the payload bytes but not this
    // count field.
    for record_index in 0..manifest.entry_count {
        if pos + 12 > payload.len() {
            return Err(InferenceError::Inference(
                "embedding cache truncated record".to_string(),
            ));
        }
        // The `pos + 12 > payload.len()` guard above ensures these fixed-width
        // slices exist; copy them without `try_into().expect()`.
        let mut hash_bytes = [0u8; 8];
        hash_bytes.copy_from_slice(&payload[pos..pos + 8]);
        let hash = u64::from_le_bytes(hash_bytes);
        let mut dim_bytes = [0u8; 4];
        dim_bytes.copy_from_slice(&payload[pos + 8..pos + 12]);
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        pos += 12;
        if dim != dimensions {
            return Err(InferenceError::Inference(format!(
                "embedding cache record dimension mismatch at record {record_index}: expected {dimensions}, found {dim}"
            )));
        }
        let Some(float_bytes) = dim.checked_mul(4) else {
            return Err(InferenceError::Inference(
                "embedding cache truncated record".to_string(),
            ));
        };
        let Some(end) = pos.checked_add(float_bytes) else {
            return Err(InferenceError::Inference(
                "embedding cache truncated record".to_string(),
            ));
        };
        if end > payload.len() {
            return Err(InferenceError::Inference(
                "embedding cache truncated record".to_string(),
            ));
        }
        let mut floats = Vec::with_capacity(dim);
        for (float_index, chunk) in payload[pos..end].chunks_exact(4).enumerate() {
            // `chunks_exact(4)` yields four-byte chunks; copy without `expect`.
            let mut float_bytes = [0u8; 4];
            float_bytes.copy_from_slice(chunk);
            let f = f32::from_le_bytes(float_bytes);
            if !f.is_finite() {
                return Err(InferenceError::Inference(format!(
                    "embedding cache non-finite float at record {record_index}, index {float_index}"
                )));
            }
            floats.push(f);
        }
        pos = end;
        entries.push((hash, floats));
    }
    if pos != payload.len() {
        return Err(InferenceError::Inference(
            "embedding cache trailing bytes".to_string(),
        ));
    }

    Ok(entries)
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
    let num_key_value_heads = get("num_key_value_heads")?;
    // Both head counts are used as divisors (head_dim inference below,
    // QwenConfig::num_groups, GQA grouping). A zero from a caller-supplied
    // config.json would otherwise be an integer divide-by-zero panic, so reject
    // it as InvalidInput here rather than crashing deep in the forward pass.
    if num_attention_heads == 0 || num_key_value_heads == 0 {
        return Err(InferenceError::InvalidInput(format!(
            "config.json: num_attention_heads ({num_attention_heads}) and \
             num_key_value_heads ({num_key_value_heads}) must both be non-zero"
        )));
    }
    // GQA grouping requires num_attention_heads divisible by num_key_value_heads.
    // `GqaConfig::groups()` only `debug_assert`s this (stripped in release), and
    // `apply_gqa_attention` hard-asserts it; a non-divisible caller-supplied
    // config.json would otherwise panic deep in the forward pass. Reject it here
    // at load time, mirroring `Qwen35Config::from_json_str`.
    if num_attention_heads % num_key_value_heads != 0 {
        return Err(InferenceError::InvalidInput(format!(
            "config.json: num_attention_heads ({num_attention_heads}) must be \
             divisible by num_key_value_heads ({num_key_value_heads})"
        )));
    }
    let hidden_size = get("hidden_size")?;
    // head_dim may be explicit in config, or inferred from hidden_size / num_heads.
    let head_dim =
        extract_json_usize(&text, "head_dim").unwrap_or(hidden_size / num_attention_heads);
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(InferenceError::InvalidInput(format!(
            "config.json: head_dim ({head_dim}) must be non-zero and even for RoPE"
        )));
    }

    Ok(QwenConfig {
        vocab_size: get("vocab_size")?,
        hidden_size,
        num_hidden_layers: get("num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads,
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
mod tests;
