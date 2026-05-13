//! Criterion baselines for the top 5 inference hot paths (Round 0 measurements).
//!
//! Covers:
//!   OPT-002: Sampler allocation (vocab-scale clone + sort per token)
//!   OPT-003: NEON Q8_0 GEMV (decode matvec allocations)
//!   OPT-004: Paged KV cache append and gather
//!   OPT-005: BPE tokenizer allocation churn
//!
//! OPT-001 (Metal forward_step logits redundant readback) requires the
//! `metal-gpu` feature and real model files. Run that baseline separately:
//!   cargo run --example bench_metal --features f16,metal-gpu --release
//!
//! Run all baselines here:
//!   cargo bench -p lattice-inference --bench inference_perf 2>&1

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Counting allocator — wraps the system allocator to track per-call statistics.
// Only active in this bench binary; does not affect production code.
// ---------------------------------------------------------------------------

struct CountingAlloc;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES_ALLOCATED: AtomicU64 = AtomicU64::new(0);
static BYTES_DEALLOCATED: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES_ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES_DEALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES_ALLOCATED.fetch_add(new_size as u64, Ordering::Relaxed);
        BYTES_DEALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

struct AllocationSnapshot {
    alloc_calls: u64,
    realloc_calls: u64,
    bytes_allocated: u64,
}

impl AllocationSnapshot {
    fn capture() -> Self {
        Self {
            alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
            realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
            bytes_allocated: BYTES_ALLOCATED.load(Ordering::Relaxed),
        }
    }

    fn delta_since(start: &AllocationSnapshot) -> AllocationDelta {
        AllocationDelta {
            alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed) - start.alloc_calls,
            realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed) - start.realloc_calls,
            bytes_allocated: BYTES_ALLOCATED.load(Ordering::Relaxed) - start.bytes_allocated,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct AllocationDelta {
    alloc_calls: u64,
    realloc_calls: u64,
    bytes_allocated: u64,
}

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use lattice_inference::forward::cpu::matmul_bt;
#[cfg(feature = "bench-internals")]
use lattice_inference::forward::cpu::{elementwise_mul, rms_norm, silu_inplace};
use lattice_inference::forward::neon::{
    matmul_q8_neon, matvec_q8_scalar, pack_weights_q8, quantize_vec_q8,
};
use lattice_inference::kv_cache::{
    EvictionPolicy, FlatKVCache, FlatKVCacheConfig, PagedKVCache, PagedKVCacheConfig,
};
#[cfg(feature = "bench-internals")]
use lattice_inference::rope::RopeTable;
use lattice_inference::sampling::{Sampler, SamplingConfig};
use lattice_inference::{BpeTokenizer, Tokenizer};

// ---------------------------------------------------------------------------
// Shared PRNG — xorshift32, no external dep, deterministic.
// ---------------------------------------------------------------------------

fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn rand_f32_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed ^ (len as u32).wrapping_mul(0x9E37_79B9);
    if state == 0 {
        state = 0xDEAD_BEEF;
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let bits = xorshift32(&mut state);
        // Small activations typical of model logits (-2 to +2)
        out.push((bits as f32 / u32::MAX as f32) * 4.0 - 2.0);
    }
    out
}

// ---------------------------------------------------------------------------
// OPT-002: Sampler allocation
//
// The Metal `sample_token` function (private to metal_qwen35.rs) and the
// public `Sampler::sample` share the same allocation pattern:
//   1. `.to_vec()` clone of the full logit vector  (vocab_size * 4 bytes)
//   2. Full-vocab indexed pair allocation for top-k sort
//   3. Softmax probability vector allocation
//
// Qwen3.5 vocab size = 151,936. Default sampling: temperature=0.7, top_k=50,
// top_p=0.9, rep_penalty=1.1.  This runs for every generated token.
// ---------------------------------------------------------------------------

const QWEN_VOCAB_SIZE: usize = 151_936;

fn bench_sampler_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampler_allocation");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let logits = rand_f32_vec(QWEN_VOCAB_SIZE, 0xA1B2_C3D4);

    // Default Qwen sampling: exercises full vocab clone + sort + softmax + top-p.
    group.throughput(Throughput::Elements(QWEN_VOCAB_SIZE as u64));
    group.bench_function("default_topk50_topp0.9", |b| {
        b.iter_batched(
            || Sampler::new(SamplingConfig::default()).with_seed(0xDEAD_BEEF),
            |mut sampler| black_box(sampler.sample(black_box(&logits))),
            BatchSize::SmallInput,
        );
    });

    // Greedy baseline: argmax only, no allocations beyond logits clone.
    group.bench_function("greedy_argmax_baseline", |b| {
        b.iter_batched(
            || Sampler::new(SamplingConfig::greedy()),
            |mut sampler| black_box(sampler.sample(black_box(&logits))),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// OPT-003: NEON Q8_0 GEMV (decode matvec)
//
// Production shapes from Qwen3.5-0.5B (hidden=2048):
//   Q/K/V projection: k=2048, n=2048  (or n=256 for KV heads)
//   FFN gate/up:      k=2048, n=8192
//   FFN down:         k=8192, n=2048
//
// The current `matmul_q8_neon` wrapper: quantizes x into a new Vec<i8>,
// allocates a new output Vec<f32>, then runs the NEON kernel. OPT-003 proposes
// `_into` variants that write into pre-allocated scratch buffers.
// ---------------------------------------------------------------------------

fn bench_simd_q8_neon_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_q8_neon_matvec");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    // (label, n=output_rows, k=input_cols)
    let cases: &[(&str, usize, usize)] = &[
        ("q_proj_k2048_n2048", 2048, 2048),
        ("kv_proj_k2048_n256", 256, 2048),
        ("ffn_gate_k2048_n8192", 8192, 2048),
        ("ffn_down_k8192_n2048", 2048, 8192),
    ];

    for &(label, n, k) in cases {
        let x = rand_f32_vec(k, 0x1111_0000 ^ n as u32);
        let w_f32 = rand_f32_vec(n * k, 0x2222_0000 ^ n as u32 ^ (k as u32).wrapping_shl(8));
        let packed = pack_weights_q8(&w_f32, n, k);

        // Pre-quantize x once: used for the scalar path (no per-call alloc).
        let (x_q, x_scale) = quantize_vec_q8(&x);
        let mut output = vec![0.0f32; n];

        // Throughput expressed as 2*N*K GEMV FLOPs for comparability.
        group.throughput(Throughput::Elements(2u64 * n as u64 * k as u64));

        // Full wrapper: quantize x (alloc Vec<i8>) + NEON dispatch + alloc Vec<f32> output.
        group.bench_function(
            BenchmarkId::new("matmul_q8_neon_full_wrapper", label),
            |b| {
                b.iter(|| {
                    let y = matmul_q8_neon(black_box(&x), black_box(&packed), n, k);
                    black_box(y);
                });
            },
        );

        // Scalar path with pre-quantized x writing into an existing buffer.
        // Represents the _into API target for OPT-003.
        group.bench_function(
            BenchmarkId::new("matvec_q8_scalar_prequant_into", label),
            |b| {
                b.iter(|| {
                    matvec_q8_scalar(
                        black_box(&x_q),
                        black_box(x_scale),
                        black_box(&packed),
                        n,
                        k,
                        black_box(output.as_mut_slice()),
                    );
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// OPT-004: Paged KV cache append and gather
//
// The two hot sub-operations per decode step:
//   append_kv_layer: one token written per layer (includes O(num_pages) LRU scan)
//   gather_k / gather_v: full-sequence read for attention (one copy per token)
//
// Synthetic config: num_kv_heads=1, head_dim=128 → kv_dim=128.
// Real Qwen3.5-0.5B: 24 layers, 8 KV heads, head_dim=64, kv_dim=512.
// Algorithmic patterns (LRU linear scan, per-token table resolve) are identical;
// absolute numbers scale linearly with kv_dim.
// ---------------------------------------------------------------------------

const KV_DIM: usize = 128;
const PAGE_SIZE: usize = 128;

fn paged_config(seq_capacity: usize, num_layers: usize) -> PagedKVCacheConfig {
    let max_pages = seq_capacity / PAGE_SIZE + 4;
    PagedKVCacheConfig {
        page_size: PAGE_SIZE,
        max_pages,
        num_layers,
        num_kv_heads: 1,
        head_dim: KV_DIM,
        eviction: EvictionPolicy::None,
    }
}

fn prefilled_cache_1layer(seq_len: usize) -> PagedKVCache {
    let k_tok = rand_f32_vec(KV_DIM, 0xBBBB_0001);
    let v_tok = rand_f32_vec(KV_DIM, 0xBBBB_0002);
    let mut cache = PagedKVCache::new(paged_config(seq_len, 1));
    for _ in 0..seq_len {
        cache.append_kv_layer(0, &k_tok, &v_tok);
        cache.advance();
    }
    cache
}

fn bench_kv_cache_paged(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_paged");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let k_tok = rand_f32_vec(KV_DIM, 0xAAAA_0001);
    let v_tok = rand_f32_vec(KV_DIM, 0xAAAA_0002);

    // --- kv_append_f32: cost of appending one token to a pre-seeded f32 cache ---
    // Throughput unit: 1 token per iteration (tok/s via Throughput::Elements(1)).
    // Setup builds a warm cache at each context length outside the timing loop.
    // Also prints static memory accounting to stderr before the first run.
    for &seq_len in &[1024usize, 4096, 16384] {
        {
            let warm = prefilled_cache_1layer(seq_len);
            eprintln!(
                "\nkv_cache_paged/f32/seq{seq_len}: total_memory_bytes={} used_memory_bytes={} logical_pages={}",
                warm.total_memory_bytes(),
                warm.used_memory_bytes(),
                warm.num_pages(),
            );
        }

        group.throughput(Throughput::Elements(1));
        group.bench_function(
            BenchmarkId::new("kv_append_f32", format!("seq{seq_len}")),
            |b| {
                b.iter_batched(
                    || prefilled_cache_1layer(seq_len),
                    |mut cache| {
                        cache.append_kv_layer(0, black_box(&k_tok), black_box(&v_tok));
                        cache.advance();
                        black_box(cache);
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    // TODO(i2): kv_append_q8/seq{N} — identical harness with cache_type_k=CacheType::Q8
    //   and cache_type_v=CacheType::Q8.  Uncomment once paged.rs has CacheType + Q8 PagePool.
    //
    // for &seq_len in &[1024usize, 4096, 16384] {
    //     group.throughput(Throughput::Elements(1));
    //     group.bench_function(
    //         BenchmarkId::new("kv_append_q8", format!("seq{seq_len}")),
    //         |b| {
    //             b.iter_batched(
    //                 || prefilled_cache_1layer_q8(seq_len),
    //                 |mut cache| {
    //                     cache.append_kv_layer(0, black_box(&k_tok), black_box(&v_tok));
    //                     cache.advance();
    //                     black_box(cache);
    //                 },
    //                 BatchSize::LargeInput,
    //             );
    //         },
    //     );
    // }

    // --- kv_gather_f32: full-sequence K+V gather from a pre-filled f32 cache ---
    // Throughput unit: bytes transferred (K output + V output) per iteration.
    // Measures the cost of the existing token-at-a-time copy loop in gather_k/gather_v.
    for &seq_len in &[1024usize, 4096, 16384] {
        let cache = prefilled_cache_1layer(seq_len);
        let mut k_dst = vec![0.0f32; seq_len * KV_DIM];
        let mut v_dst = vec![0.0f32; seq_len * KV_DIM];
        // K bytes + V bytes written to caller buffers per iteration.
        let bytes = (seq_len * KV_DIM * std::mem::size_of::<f32>() * 2) as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_function(
            BenchmarkId::new("kv_gather_f32", format!("seq{seq_len}")),
            |b| {
                b.iter(|| {
                    cache.gather_k(0, black_box(k_dst.as_mut_slice()));
                    cache.gather_v(0, black_box(v_dst.as_mut_slice()));
                    black_box(&k_dst);
                    black_box(&v_dst);
                });
            },
        );
    }

    // TODO(i2): kv_gather_q8/seq{N} — identical harness with Q8 cache; dst is f32
    //   (dequantized on gather). Throughput bytes = same K+V f32 output size.
    //
    // for &seq_len in &[1024usize, 4096, 16384] {
    //     let cache = prefilled_cache_1layer_q8(seq_len);
    //     let mut k_dst = vec![0.0f32; seq_len * KV_DIM];
    //     let mut v_dst = vec![0.0f32; seq_len * KV_DIM];
    //     let bytes = (seq_len * KV_DIM * std::mem::size_of::<f32>() * 2) as u64;
    //     group.throughput(Throughput::Bytes(bytes));
    //     group.bench_function(
    //         BenchmarkId::new("kv_gather_q8", format!("seq{seq_len}")),
    //         |b| {
    //             b.iter(|| {
    //                 cache.gather_k(0, black_box(k_dst.as_mut_slice()));
    //                 cache.gather_v(0, black_box(v_dst.as_mut_slice()));
    //                 black_box(&k_dst);
    //                 black_box(&v_dst);
    //             });
    //         },
    //     );
    // }

    group.finish();
}

// ---------------------------------------------------------------------------
// OPT-005: BPE tokenizer
//
// The current path allocates per-word: byte-encoded String, merge node Strings,
// heap candidates, and cached ID Vecs. OPT-005 proposes scratch reuse.
//
// Two measurement scenarios:
//   cache_hit:  repeated identical text — LRU hits for all words
//   cache_miss: fresh text each call   — full BPE merge path for each word
//
// Real tokenizer: set LATTICE_INFERENCE_MODEL_DIR or place tokenizer.json at
//   ~/.lattice/models/qwen3.5-0.5b/tokenizer.json (or Qwen3.5-0.5B/).
// Falls back to a synthetic GPT-2-style BPE if no real tokenizer found.
// ---------------------------------------------------------------------------

fn qwen_tokenizer_path() -> Option<std::path::PathBuf> {
    let from_env = std::env::var("LATTICE_INFERENCE_MODEL_DIR")
        .ok()
        .map(|d| std::path::PathBuf::from(d).join("tokenizer.json"));

    let home = std::env::var("HOME").ok();
    let from_home_lower = home
        .as_deref()
        .map(|h| std::path::Path::new(h).join(".lattice/models/qwen3.5-0.5b/tokenizer.json"));
    let from_home_upper = home
        .as_deref()
        .map(|h| std::path::Path::new(h).join(".lattice/models/Qwen3.5-0.5B/tokenizer.json"));

    [from_env, from_home_lower, from_home_upper]
        .into_iter()
        .flatten()
        .find(|p| p.exists())
}

fn build_synthetic_bpe() -> BpeTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut id = 0u32;

    // Printable ASCII characters and the GPT-2 space token (Ġ = U+0120).
    // These become the base vocabulary from which merges build larger tokens.
    let single_chars = [
        "Ġ", "T", "h", "e", "q", "u", "i", "c", "k", "b", "r", "o", "w", "n", "f", "x", "j", "m",
        "p", "s", "v", "l", "a", "z", "d", "g", "y", "t", "A", "I", "N", "L", "P", "B", ".", ",",
        "!", "'", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", "\n",
    ];
    for &ch in &single_chars {
        vocab.insert(ch.to_string(), id);
        id += 1;
    }

    // Common English subword tokens produced by BPE merging.
    let merged = [
        "th",
        "the",
        "Ġthe",
        "Ġa",
        "Ġin",
        "Ġof",
        "Ġto",
        "Ġand",
        "Ġis",
        "Ġfor",
        "Ġon",
        "Ġwith",
        "Ġit",
        "Ġat",
        "he",
        "er",
        "ing",
        "ed",
        "re",
        "en",
        "on",
        "at",
        "an",
        "Ġqu",
        "Ġqui",
        "Ġquic",
        "Ġquick",
        "Ġbr",
        "Ġbro",
        "Ġbrow",
        "Ġbrown",
        "Ġfo",
        "Ġfox",
        "Ġwor",
        "Ġworl",
        "Ġworld",
        "mo",
        "od",
        "mod",
        "model",
        "tok",
        "toke",
        "token",
        "inf",
        "infe",
        "infer",
        "att",
        "atte",
        "atten",
        "attent",
        "attenti",
        "attentio",
        "attention",
        "tra",
        "tran",
        "trans",
        "transf",
        "transfo",
        "transfor",
        "transform",
        "pro",
        "proc",
        "process",
        "la",
        "ng",
        "lan",
        "lang",
        "langu",
        "langua",
        "language",
        "ar",
        "art",
        "arti",
        "artif",
        "artifi",
        "artic",
        "artific",
        "artificia",
        "artificial",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
    ];
    for tok in merged {
        vocab.entry(tok.to_string()).or_insert_with(|| {
            let v = id;
            id += 1;
            v
        });
    }

    // BPE merge rules in priority order (lower index = higher priority).
    let merges = vec![
        ("t".to_string(), "h".to_string()),
        ("th".to_string(), "e".to_string()),
        ("Ġ".to_string(), "t".to_string()),
        ("Ġt".to_string(), "h".to_string()),
        ("Ġth".to_string(), "e".to_string()),
        ("Ġ".to_string(), "a".to_string()),
        ("Ġ".to_string(), "i".to_string()),
        ("Ġi".to_string(), "n".to_string()),
        ("Ġ".to_string(), "o".to_string()),
        ("Ġo".to_string(), "f".to_string()),
        ("Ġ".to_string(), "w".to_string()),
        ("Ġw".to_string(), "o".to_string()),
        ("Ġwo".to_string(), "r".to_string()),
        ("Ġwor".to_string(), "l".to_string()),
        ("Ġworl".to_string(), "d".to_string()),
        ("Ġ".to_string(), "q".to_string()),
        ("Ġq".to_string(), "u".to_string()),
        ("Ġqu".to_string(), "i".to_string()),
        ("Ġqui".to_string(), "c".to_string()),
        ("Ġquic".to_string(), "k".to_string()),
        ("Ġ".to_string(), "b".to_string()),
        ("Ġb".to_string(), "r".to_string()),
        ("Ġbr".to_string(), "o".to_string()),
        ("Ġbro".to_string(), "w".to_string()),
        ("Ġbrow".to_string(), "n".to_string()),
        ("Ġ".to_string(), "f".to_string()),
        ("Ġf".to_string(), "o".to_string()),
        ("Ġfo".to_string(), "x".to_string()),
        ("e".to_string(), "r".to_string()),
        ("i".to_string(), "n".to_string()),
        ("e".to_string(), "n".to_string()),
        ("o".to_string(), "n".to_string()),
        ("r".to_string(), "e".to_string()),
        ("m".to_string(), "o".to_string()),
        ("mo".to_string(), "d".to_string()),
        ("mod".to_string(), "e".to_string()),
        ("mode".to_string(), "l".to_string()),
        ("t".to_string(), "o".to_string()),
        ("to".to_string(), "k".to_string()),
        ("tok".to_string(), "e".to_string()),
        ("toke".to_string(), "n".to_string()),
        ("i".to_string(), "ng".to_string()),
        ("a".to_string(), "t".to_string()),
        ("a".to_string(), "n".to_string()),
        ("l".to_string(), "a".to_string()),
        ("la".to_string(), "n".to_string()),
        ("lan".to_string(), "g".to_string()),
        ("lang".to_string(), "u".to_string()),
        ("langu".to_string(), "a".to_string()),
        ("langua".to_string(), "g".to_string()),
        ("language".to_string(), "s".to_string()),
    ];

    BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap()
}

// Repeating-sentence corpus of the requested character count.
fn corpus_text(char_count: usize, offset: usize) -> String {
    let sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "Artificial intelligence and machine learning transform modern inference systems. ",
        "Natural language processing enables computers to understand human text efficiently. ",
        "Transformer architectures with attention mechanisms process sequential data. ",
        "Token embedding and BPE merging are fundamental operations in language models. ",
        "The inference engine processes tokenized inputs through transformer layers. ",
        "Attention heads compute query key value projections for each sequence token. ",
        "Quantized weights reduce memory bandwidth during matrix vector multiply operations. ",
    ];
    let mut text = String::with_capacity(char_count + 64);
    let mut idx = offset % sentences.len();
    while text.len() < char_count {
        text.push_str(sentences[idx % sentences.len()]);
        idx += 1;
    }
    text.truncate(char_count);
    text
}

fn bench_tokenizer_bpe(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenizer_bpe");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    let (bpe, source_label) = match qwen_tokenizer_path() {
        Some(path) => match BpeTokenizer::from_tokenizer_json(&path) {
            Ok(tok) => (tok, "real_qwen"),
            Err(e) => {
                eprintln!("[inference_perf] real tokenizer load failed: {e}; using synthetic");
                (build_synthetic_bpe(), "synthetic")
            }
        },
        None => {
            eprintln!(
                "[inference_perf] no real tokenizer found; using synthetic BPE.\n\
                 Set LATTICE_INFERENCE_MODEL_DIR or place tokenizer.json at\n\
                 ~/.lattice/models/qwen3.5-0.5b/tokenizer.json for real-model baselines."
            );
            (build_synthetic_bpe(), "synthetic")
        }
    };

    for char_count in [128usize, 512, 1024, 4096] {
        // Warm text: identical on every call — LRU cache absorbs all per-word cost.
        let warm_text = corpus_text(char_count, 0);
        // Prime the cache once before measuring.
        let _ = bpe.tokenize(&warm_text);
        let tok_count = bpe.tokenize(&warm_text).real_length as u64;

        group.throughput(Throughput::Elements(tok_count.max(1)));

        // Cache-hit path: same text repeated. Exercises cache lookup + ID copy.
        group.bench_function(
            BenchmarkId::new(
                format!("cache_hit_{source_label}"),
                format!("chars{char_count}"),
            ),
            |b| {
                b.iter(|| {
                    let result = bpe.tokenize(black_box(&warm_text));
                    black_box(result);
                });
            },
        );

        // Cache-miss path: fresh text each sample — forces full BPE merge per word.
        group.bench_function(
            BenchmarkId::new(
                format!("cache_miss_{source_label}"),
                format!("chars{char_count}"),
            ),
            |b| {
                let mut counter = 0usize;
                b.iter_batched(
                    || {
                        counter = counter.wrapping_add(1);
                        corpus_text(char_count, counter * 3)
                    },
                    |text| {
                        let result = bpe.tokenize(black_box(&text));
                        black_box(result);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// OPT-003 caller-level benchmark: Q8 NEON forward step (R1-002)
//
// Exercises `forward_step_q8_neon` directly — the production hot path that
// dispatches to all Q8 projections, GDN recurrence, and GQA attention.
// The low-level `matmul_q8_neon` wrapper benchmarks above cannot prove
// caller-level allocation removal; this group can.
//
// Two cases:
//   pos0       — first token, seq_len=0, cold KV cache
//   warm_seq128 — one additional step after 128 tokens of warmup
//
// Compile with: --features bench-internals
// Run with:
//   RUSTC_WRAPPER="" cargo bench -p lattice-inference \
//     --features bench-internals --bench inference_perf -- q8_neon_forward
// ---------------------------------------------------------------------------

fn bench_q8_neon_forward(c: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        use lattice_inference::forward::neon_forward::bench_support::Q8ForwardBenchFixture;

        let fixture = Q8ForwardBenchFixture::synthetic_2layer();

        let mut group = c.benchmark_group("q8_neon_forward");
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));

        // pos0: cold start, seq_len=0
        group.bench_function("forward_step_synthetic_2layer_pos0", |b| {
            b.iter_batched(
                || fixture.state_with_capacity(0, 1),
                |mut state| black_box(fixture.step(&mut state, 42)),
                BatchSize::LargeInput,
            );
        });

        // warm_seq128: pre-warmed with 128 tokens, measure one additional step
        group.bench_function("forward_step_synthetic_2layer_warm_seq128", |b| {
            b.iter_batched(
                || fixture.state_with_capacity(128, 1),
                |mut state| black_box(fixture.step(&mut state, 42)),
                BatchSize::LargeInput,
            );
        });

        group.finish();
    }
    #[cfg(not(feature = "bench-internals"))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// Q8 NEON allocation-count benchmark (before/after zero-alloc migration)
//
// Measures allocations-per-token for two model shapes:
//   synthetic_2layer  — 2 layers (1 GDN + 1 full), hidden=256, vocab=8192
//   qwen35_24layer_shape — 24 layers (18 GDN + 6 full), Qwen35-2B dims, vocab=256
//
// Run with:
//   RUSTC_WRAPPER="" cargo bench -p lattice-inference \
//     --features bench-internals --bench inference_perf -- q8_neon_forward_allocations
//
// Reports output to stderr in the format:
//   q8_neon_forward_allocations/<config>/<phase>
//   tokens=N alloc_calls_total=A realloc_calls_total=R bytes_allocated_total=B
//   allocations_per_token=A/N reallocations_per_token=R/N bytes_allocated_per_token=B/N
//
// Determinism gate: runs 3 consecutive samples after fixture warmup; panics if
// alloc_calls, realloc_calls, or bytes_allocated differ across runs.
// ---------------------------------------------------------------------------

// Print allocation counts without asserting zero — use for "before" snapshots.
fn allocation_count_print(
    label: &str,
    phase: &str,
    tokens: usize,
    run_fn: &mut impl FnMut() -> AllocationDelta,
) {
    let samples: Vec<AllocationDelta> = (0..3).map(|_| run_fn()).collect();
    let s0 = samples[0];
    for (i, &s) in samples.iter().enumerate().skip(1) {
        if s != s0 {
            panic!(
                "allocation count is non-deterministic at run {i} for {label}/{phase}: \
                 run0=({},{},{}) run{i}=({},{},{})",
                s0.alloc_calls,
                s0.realloc_calls,
                s0.bytes_allocated,
                s.alloc_calls,
                s.realloc_calls,
                s.bytes_allocated,
            );
        }
    }
    let n = tokens as f64;
    eprintln!(
        "\nq8_neon_forward_allocations/{label}/{phase}\n\
         tokens={tokens}\n\
         alloc_calls_total={}\n\
         realloc_calls_total={}\n\
         bytes_allocated_total={}\n\
         allocations_per_token={:.2}\n\
         reallocations_per_token={:.2}\n\
         bytes_allocated_per_token={:.0}",
        s0.alloc_calls,
        s0.realloc_calls,
        s0.bytes_allocated,
        s0.alloc_calls as f64 / n,
        s0.realloc_calls as f64 / n,
        s0.bytes_allocated as f64 / n,
    );
}

fn allocation_count_report(
    label: &str,
    phase: &str,
    tokens: usize,
    run_fn: &mut impl FnMut() -> AllocationDelta,
) {
    let samples: Vec<AllocationDelta> = (0..3).map(|_| run_fn()).collect();

    let s0 = samples[0];
    for (i, &s) in samples.iter().enumerate().skip(1) {
        if s != s0 {
            panic!(
                "allocation count is non-deterministic at run {i} for {label}/{phase}: \
                 run0=({},{},{}) run{i}=({},{},{})",
                s0.alloc_calls,
                s0.realloc_calls,
                s0.bytes_allocated,
                s.alloc_calls,
                s.realloc_calls,
                s.bytes_allocated,
            );
        }
    }

    if s0.alloc_calls != 0 || s0.realloc_calls != 0 || s0.bytes_allocated != 0 {
        panic!(
            "allocation gate failed for {label}/{phase}: \
             alloc_calls_total={} realloc_calls_total={} bytes_allocated_total={}",
            s0.alloc_calls, s0.realloc_calls, s0.bytes_allocated,
        );
    }

    let n = tokens as f64;
    eprintln!(
        "\nq8_neon_forward_allocations/{label}/{phase}\n\
         tokens={tokens}\n\
         alloc_calls_total={}\n\
         realloc_calls_total={}\n\
         bytes_allocated_total={}\n\
         allocations_per_token={:.2}\n\
         reallocations_per_token={:.2}\n\
         bytes_allocated_per_token={:.0}",
        s0.alloc_calls,
        s0.realloc_calls,
        s0.bytes_allocated,
        s0.alloc_calls as f64 / n,
        s0.realloc_calls as f64 / n,
        s0.bytes_allocated as f64 / n,
    );
}

fn bench_q8_neon_forward_allocations(c: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        use lattice_inference::forward::neon_forward::bench_support::Q8ForwardBenchFixture;

        let mut group = c.benchmark_group("q8_neon_forward_allocations");
        group.sample_size(10);

        // --- synthetic_2layer ---
        {
            let fixture = Q8ForwardBenchFixture::synthetic_2layer();
            let measured_tokens = 16usize;
            let warm_len = 128usize;

            allocation_count_report("synthetic_2layer", "after", measured_tokens, &mut || {
                let mut state = fixture.state_with_capacity(warm_len, measured_tokens);
                let start = AllocationSnapshot::capture();
                for t in 0..measured_tokens {
                    let _ = black_box(fixture.step(&mut state, t as u32 + 42));
                }
                AllocationSnapshot::delta_since(&start)
            });

            // Criterion latency measurement (structural — keeps group alive).
            group.bench_function("synthetic_2layer_allocation_gate", |b| {
                b.iter_batched(
                    || fixture.state_with_capacity(warm_len, measured_tokens),
                    |mut state| {
                        let start = AllocationSnapshot::capture();
                        for t in 0..measured_tokens {
                            let _ = black_box(fixture.step(&mut state, t as u32 + 42));
                        }
                        let delta = AllocationSnapshot::delta_since(&start);
                        black_box(delta.alloc_calls)
                    },
                    criterion::BatchSize::LargeInput,
                );
            });
        }

        // --- qwen35_24layer_shape ---
        {
            let fixture = Q8ForwardBenchFixture::qwen35_24layer_shape();
            let measured_tokens = 1usize;
            let warm_len = 128usize;

            allocation_count_report(
                "qwen35_24layer_shape",
                "after",
                measured_tokens,
                &mut || {
                    let mut state = fixture.state_with_capacity(warm_len, measured_tokens);
                    let start = AllocationSnapshot::capture();
                    for t in 0..measured_tokens {
                        let _ = black_box(fixture.step(&mut state, t as u32 + 42));
                    }
                    AllocationSnapshot::delta_since(&start)
                },
            );

            group.bench_function("qwen35_24layer_shape_allocation_gate", |b| {
                b.iter_batched(
                    || fixture.state_with_capacity(warm_len, measured_tokens),
                    |mut state| {
                        let start = AllocationSnapshot::capture();
                        for t in 0..measured_tokens {
                            let _ = black_box(fixture.step(&mut state, t as u32 + 42));
                        }
                        let delta = AllocationSnapshot::delta_since(&start);
                        black_box(delta.alloc_calls)
                    },
                    criterion::BatchSize::LargeInput,
                );
            });
        }

        group.finish();
    }
    #[cfg(not(feature = "bench-internals"))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// OPT-LOGIT: final logits projection benchmark (vocab=248320, hidden=2048)
//
// The generate.rs scalar loop (lines 400-407) does one dot product per vocab
// row against the final hidden state. This benchmark compares:
//   scalar             — exact loop from generate.rs
//   matmul_bt_existing — existing CPU dispatch (Accelerate on macOS, NEON tiled
//                        on aarch64 non-macOS). Same as the Qwen3.5 qwen35.rs path.
//
// Run:
//   cargo bench -p lattice-inference --bench inference_perf -- logits_projection
// ---------------------------------------------------------------------------

const LOGITS_VOCAB: usize = 248_320;
const LOGITS_HIDDEN: usize = 2_048;

fn try_rand_f32_vec(len: usize, seed: u32) -> Option<Vec<f32>> {
    let mut v: Vec<f32> = Vec::new();
    v.try_reserve_exact(len).ok()?;
    let mut state = seed ^ (len as u32).wrapping_mul(0x9E37_79B9);
    if state == 0 {
        state = 0xDEAD_BEEF;
    }
    for _ in 0..len {
        let bits = xorshift32(&mut state);
        v.push((bits as f32 / u32::MAX as f32) * 4.0 - 2.0);
    }
    Some(v)
}

fn bench_logits_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("logits_projection");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    // Throughput in elements/s = (VOCAB * HIDDEN) / seconds_per_iter
    group.throughput(Throughput::Elements(
        (LOGITS_VOCAB as u64) * (LOGITS_HIDDEN as u64),
    ));

    let hidden = rand_f32_vec(LOGITS_HIDDEN, 0xC0DE_2048);

    // ~1.94 GiB weight matrix; skip gracefully if OOM
    let weight = match try_rand_f32_vec(LOGITS_VOCAB * LOGITS_HIDDEN, 0xC0DE_0001) {
        Some(w) => w,
        None => {
            eprintln!(
                "[bench_logits_projection] Cannot allocate {:.1} GiB weight matrix — skip",
                (LOGITS_VOCAB * LOGITS_HIDDEN * 4) as f64 / (1u64 << 30) as f64
            );
            group.finish();
            return;
        }
    };

    let mut out = vec![0.0f32; LOGITS_VOCAB];

    // scalar: identical to the generate.rs:400-407 loop that we will optimize
    group.bench_function("scalar", |b| {
        b.iter(|| {
            let h = black_box(&hidden[..]);
            let w = black_box(&weight[..]);
            let o = black_box(out.as_mut_slice());
            for v in 0..LOGITS_VOCAB {
                let row = &w[v * LOGITS_HIDDEN..(v + 1) * LOGITS_HIDDEN];
                let mut dot = 0.0f32;
                for j in 0..LOGITS_HIDDEN {
                    dot += h[j] * row[j];
                }
                o[v] = dot;
            }
            black_box(&out);
        });
    });

    // matmul_bt: existing CPU dispatch path
    //   macOS      → Accelerate cblas_sgemm (AMX, multi-threaded even for M=1)
    //   aarch64    → NEON tiled path
    //   x86_64     → AVX2/AVX-512 tiled path
    // matmul_bt(A, B, C, m, k, n):  A=[m,k]  B=[n,k](rows)  C=[m,n]
    // For logits: A=[1,HIDDEN], B=[VOCAB,HIDDEN], C=[1,VOCAB]
    group.bench_function("matmul_bt_existing", |b| {
        b.iter(|| {
            matmul_bt(
                black_box(&hidden[..]),
                black_box(&weight[..]),
                black_box(out.as_mut_slice()),
                1,
                LOGITS_HIDDEN,
                LOGITS_VOCAB,
            );
            black_box(&out);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// OPT-FWD: forward_with_cache allocation gate (requires --features bench-internals)
//
// Measures allocations-per-token and latency for a single-token decode at
// Qwen3.5-target dimensions. The "allocating_current" variant mirrors
// generate.rs:forward_with_cache exactly: 11 top-level buffer allocs +
// 1 logits alloc + 24 layers × 8 heads × 1 query = 192 score allocs
// = 204 allocs/token total (before ForwardScratch).
//
// Run:
//   cargo bench -p lattice-inference --features bench-internals \
//     --bench inference_perf -- generate_forward_with_cache
// ---------------------------------------------------------------------------

const FWD_VOCAB: usize = 248_320;
const FWD_HIDDEN: usize = 2_048;
const FWD_LAYERS: usize = 24;
const FWD_N_HEADS: usize = 8;
const FWD_N_KV_HEADS: usize = 2;
const FWD_HEAD_DIM: usize = 256;
const FWD_INTER: usize = 6_144;
const FWD_WARM_KV: usize = 128;
const FWD_Q_DIM: usize = FWD_N_HEADS * FWD_HEAD_DIM; // 2048
const FWD_KV_DIM: usize = FWD_N_KV_HEADS * FWD_HEAD_DIM; // 512
const FWD_QKV_DIM: usize = FWD_Q_DIM + 2 * FWD_KV_DIM; // 3072
const FWD_RMS_EPS: f32 = 1e-6;
const FWD_ROPE_THETA: f64 = 10_000_000.0;

// Synthetic single-layer weights; one set shared across all FWD_LAYERS.
#[cfg(feature = "bench-internals")]
struct SyntheticLayerWeights {
    input_layernorm: Vec<f32>,     // [FWD_HIDDEN]
    qkv_proj: Vec<f32>, // [FWD_QKV_DIM * FWD_HIDDEN]  →  B in matmul_bt(…, m=1, k=HIDDEN, n=QKV_DIM)
    qkv_bias: Vec<f32>, // [FWD_QKV_DIM]
    q_norm: Vec<f32>,   // [FWD_HEAD_DIM]
    k_norm: Vec<f32>,   // [FWD_HEAD_DIM]
    o_proj: Vec<f32>,   // [FWD_HIDDEN * FWD_Q_DIM]     →  B in matmul_bt(…, m=1, k=Q_DIM, n=HIDDEN)
    post_attn_layernorm: Vec<f32>, // [FWD_HIDDEN]
    gate_up_proj: Vec<f32>, // [2*FWD_INTER * FWD_HIDDEN]   →  B in matmul_bt(…, m=1, k=HIDDEN, n=2*INTER)
    down_proj: Vec<f32>, // [FWD_HIDDEN * FWD_INTER]      →  B in matmul_bt(…, m=1, k=INTER, n=HIDDEN)
}

#[cfg(feature = "bench-internals")]
impl SyntheticLayerWeights {
    fn new(seed: u32) -> Self {
        Self {
            input_layernorm: rand_f32_vec(FWD_HIDDEN, seed ^ 0x01),
            qkv_proj: rand_f32_vec(FWD_QKV_DIM * FWD_HIDDEN, seed ^ 0x02),
            qkv_bias: rand_f32_vec(FWD_QKV_DIM, seed ^ 0x03),
            q_norm: rand_f32_vec(FWD_HEAD_DIM, seed ^ 0x04),
            k_norm: rand_f32_vec(FWD_HEAD_DIM, seed ^ 0x05),
            o_proj: rand_f32_vec(FWD_HIDDEN * FWD_Q_DIM, seed ^ 0x06),
            post_attn_layernorm: rand_f32_vec(FWD_HIDDEN, seed ^ 0x07),
            gate_up_proj: rand_f32_vec(2 * FWD_INTER * FWD_HIDDEN, seed ^ 0x08),
            down_proj: rand_f32_vec(FWD_HIDDEN * FWD_INTER, seed ^ 0x09),
        }
    }
}

// Allocating attention: one Vec<f32> per head per query position.
// Matches the generate.rs:441 allocation pattern that ForwardScratch eliminates.
#[cfg(feature = "bench-internals")]
fn compute_attention_alloc(
    output: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq_len: usize,
    kv_seq_len: usize,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_h = h / groups;
        for qi in 0..q_seq_len {
            let q_off = qi * (num_heads * head_dim) + h * head_dim;
            // Per-head score allocation — this is what ForwardScratch replaces
            let mut scores = vec![0.0f32; kv_seq_len];
            for ki in 0..kv_seq_len {
                let k_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                scores[ki] = dot * scale;
            }
            // Causal mask
            let max_attend = start_pos + qi;
            for ki in (max_attend + 1)..kv_seq_len {
                scores[ki] = f32::NEG_INFINITY;
            }
            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= sum;
                }
            }
            // Weighted sum over V
            let out_off = qi * (num_heads * head_dim) + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..kv_seq_len {
                    let v_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                    val += scores[ki] * v[v_off + d];
                }
                output[out_off + d] = val;
            }
        }
    }
}

// Single-token decode with all per-call allocations intact (pre-optimization baseline).
// Mirrors generate.rs:forward_with_cache for seq_len=1 using synthetic weights.
// Returns the logits Vec (alloc #12).
#[cfg(feature = "bench-internals")]
#[allow(clippy::too_many_arguments)]
fn forward_allocating_decode(
    token_id: u32,
    start_pos: usize,
    cache: &mut FlatKVCache,
    embed_tokens: &[f32], // [FWD_VOCAB, FWD_HIDDEN]
    lw: &SyntheticLayerWeights,
    norm_weight: &[f32], // [FWD_HIDDEN]
    rope: &RopeTable,
) -> Vec<f32> {
    let tok = (token_id as usize) % FWD_VOCAB;

    // Alloc #1
    let mut hidden = vec![0.0f32; FWD_HIDDEN];
    hidden.copy_from_slice(&embed_tokens[tok * FWD_HIDDEN..(tok + 1) * FWD_HIDDEN]);

    // Allocs #2-11
    let mut residual = vec![0.0f32; FWD_HIDDEN];
    let mut qkv_buf = vec![0.0f32; FWD_QKV_DIM];
    let mut q_buf = vec![0.0f32; FWD_Q_DIM];
    let mut k_buf = vec![0.0f32; FWD_KV_DIM];
    let mut v_buf = vec![0.0f32; FWD_KV_DIM];
    let mut attn_out = vec![0.0f32; FWD_Q_DIM];
    let mut gate_up_buf = vec![0.0f32; 2 * FWD_INTER];
    let mut gate_buf = vec![0.0f32; FWD_INTER];
    let mut up_buf = vec![0.0f32; FWD_INTER];
    let mut ffn_out = vec![0.0f32; FWD_HIDDEN];

    for layer in 0..FWD_LAYERS {
        // Pre-attention RMS norm
        residual.copy_from_slice(&hidden);
        rms_norm(&mut hidden, &lw.input_layernorm, FWD_HIDDEN, FWD_RMS_EPS);

        // QKV projection: A=[1,HIDDEN], B=[QKV_DIM,HIDDEN], C=[1,QKV_DIM]
        matmul_bt(
            &hidden,
            &lw.qkv_proj,
            &mut qkv_buf,
            1,
            FWD_HIDDEN,
            FWD_QKV_DIM,
        );
        for j in 0..FWD_QKV_DIM {
            qkv_buf[j] += lw.qkv_bias[j];
        }

        // Scatter QKV
        q_buf.copy_from_slice(&qkv_buf[..FWD_Q_DIM]);
        k_buf.copy_from_slice(&qkv_buf[FWD_Q_DIM..FWD_Q_DIM + FWD_KV_DIM]);
        v_buf.copy_from_slice(&qkv_buf[FWD_Q_DIM + FWD_KV_DIM..]);

        // QK norm (per head)
        for h in 0..FWD_N_HEADS {
            let off = h * FWD_HEAD_DIM;
            rms_norm(
                &mut q_buf[off..off + FWD_HEAD_DIM],
                &lw.q_norm,
                FWD_HEAD_DIM,
                FWD_RMS_EPS,
            );
        }
        for h in 0..FWD_N_KV_HEADS {
            let off = h * FWD_HEAD_DIM;
            rms_norm(
                &mut k_buf[off..off + FWD_HEAD_DIM],
                &lw.k_norm,
                FWD_HEAD_DIM,
                FWD_RMS_EPS,
            );
        }

        // RoPE (single token at position start_pos)
        for h in 0..FWD_N_HEADS {
            rope.apply(
                &mut q_buf[h * FWD_HEAD_DIM..(h + 1) * FWD_HEAD_DIM],
                start_pos,
            );
        }
        for h in 0..FWD_N_KV_HEADS {
            rope.apply(
                &mut k_buf[h * FWD_HEAD_DIM..(h + 1) * FWD_HEAD_DIM],
                start_pos,
            );
        }

        // Write K/V to cache at start_pos (mutable borrow ends before shared borrows below)
        {
            let dst_off = start_pos * FWD_KV_DIM;
            cache.k_buffer_mut(layer)[dst_off..dst_off + FWD_KV_DIM].copy_from_slice(&k_buf);
            cache.v_buffer_mut(layer)[dst_off..dst_off + FWD_KV_DIM].copy_from_slice(&v_buf);
        }

        // Attention over full cached K/V (start_pos prior tokens + 1 current)
        let cached_seq_len = start_pos + 1;
        let k_end = cached_seq_len * FWD_KV_DIM;
        // Allocs #13..+8 per layer (8 heads × 1 query): the 192 score allocs
        compute_attention_alloc(
            &mut attn_out,
            &q_buf,
            &cache.k_buffer(layer)[..k_end],
            &cache.v_buffer(layer)[..k_end],
            1,
            cached_seq_len,
            start_pos,
            FWD_N_HEADS,
            FWD_N_KV_HEADS,
            FWD_HEAD_DIM,
        );

        // O projection: A=[1,Q_DIM], B=[HIDDEN,Q_DIM], C=[1,HIDDEN]
        matmul_bt(&attn_out, &lw.o_proj, &mut hidden, 1, FWD_Q_DIM, FWD_HIDDEN);
        for i in 0..FWD_HIDDEN {
            hidden[i] += residual[i];
        }

        // Post-attention RMS norm
        residual.copy_from_slice(&hidden);
        rms_norm(
            &mut hidden,
            &lw.post_attn_layernorm,
            FWD_HIDDEN,
            FWD_RMS_EPS,
        );

        // Gate+Up projection: A=[1,HIDDEN], B=[2*INTER,HIDDEN], C=[1,2*INTER]
        matmul_bt(
            &hidden,
            &lw.gate_up_proj,
            &mut gate_up_buf,
            1,
            FWD_HIDDEN,
            2 * FWD_INTER,
        );
        gate_buf.copy_from_slice(&gate_up_buf[..FWD_INTER]);
        up_buf.copy_from_slice(&gate_up_buf[FWD_INTER..]);

        // SwiGLU
        silu_inplace(&mut gate_buf);
        elementwise_mul(&mut gate_buf, &up_buf);

        // Down projection: A=[1,INTER], B=[HIDDEN,INTER], C=[1,HIDDEN]
        matmul_bt(
            &gate_buf,
            &lw.down_proj,
            &mut ffn_out,
            1,
            FWD_INTER,
            FWD_HIDDEN,
        );
        for i in 0..FWD_HIDDEN {
            hidden[i] = residual[i] + ffn_out[i];
        }
    }

    // Final RMS norm
    rms_norm(&mut hidden, norm_weight, FWD_HIDDEN, FWD_RMS_EPS);

    // Alloc #12: logits
    let mut logits = vec![0.0f32; FWD_VOCAB];
    matmul_bt(&hidden, embed_tokens, &mut logits, 1, FWD_HIDDEN, FWD_VOCAB);
    logits
}

// ---------------------------------------------------------------------------
// Opt 1+2+3: ForwardScratch + score-slice attention + matmul_bt logits
// ---------------------------------------------------------------------------

// Pre-allocated scratch buffers — reused across tokens, zero allocs on warm path.
#[cfg(feature = "bench-internals")]
struct ForwardBenchScratch {
    hidden: Vec<f32>,      // [FWD_HIDDEN]
    residual: Vec<f32>,    // [FWD_HIDDEN]
    qkv_buf: Vec<f32>,     // [FWD_QKV_DIM]
    q_buf: Vec<f32>,       // [FWD_Q_DIM]
    k_buf: Vec<f32>,       // [FWD_KV_DIM]
    v_buf: Vec<f32>,       // [FWD_KV_DIM]
    attn_out: Vec<f32>,    // [FWD_Q_DIM]
    gate_up_buf: Vec<f32>, // [2*FWD_INTER]
    gate_buf: Vec<f32>,    // [FWD_INTER]
    up_buf: Vec<f32>,      // [FWD_INTER]
    ffn_out: Vec<f32>,     // [FWD_HIDDEN]
    scores: Vec<f32>,      // [FWD_N_HEADS * q_seq_len * kv_seq_len]
    logits: Vec<f32>,      // [FWD_VOCAB]
}

#[cfg(feature = "bench-internals")]
impl ForwardBenchScratch {
    fn new(max_kv_seq_len: usize) -> Self {
        Self {
            hidden: vec![0.0; FWD_HIDDEN],
            residual: vec![0.0; FWD_HIDDEN],
            qkv_buf: vec![0.0; FWD_QKV_DIM],
            q_buf: vec![0.0; FWD_Q_DIM],
            k_buf: vec![0.0; FWD_KV_DIM],
            v_buf: vec![0.0; FWD_KV_DIM],
            attn_out: vec![0.0; FWD_Q_DIM],
            gate_up_buf: vec![0.0; 2 * FWD_INTER],
            gate_buf: vec![0.0; FWD_INTER],
            up_buf: vec![0.0; FWD_INTER],
            ffn_out: vec![0.0; FWD_HIDDEN],
            scores: vec![0.0; FWD_N_HEADS * max_kv_seq_len],
            logits: vec![0.0; FWD_VOCAB],
        }
    }
}

// Opt 2: score-scratch attention — same math as compute_attention_alloc but
// indexes into a pre-allocated scratch slice instead of allocating per head.
#[cfg(feature = "bench-internals")]
fn compute_attention_scratch(
    output: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq_len: usize,
    kv_seq_len: usize,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scores_scratch: &mut [f32],
) {
    let groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let kv_h = h / groups;
        for qi in 0..q_seq_len {
            let q_off = qi * (num_heads * head_dim) + h * head_dim;
            let score_offset = (h * q_seq_len + qi) * kv_seq_len;
            let scores = &mut scores_scratch[score_offset..score_offset + kv_seq_len];
            scores.fill(0.0);
            for ki in 0..kv_seq_len {
                let k_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                scores[ki] = dot * scale;
            }
            let max_attend = start_pos + qi;
            for ki in (max_attend + 1)..kv_seq_len {
                scores[ki] = f32::NEG_INFINITY;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= sum;
                }
            }
            let out_off = qi * (num_heads * head_dim) + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..kv_seq_len {
                    let v_off = ki * (num_kv_heads * head_dim) + kv_h * head_dim;
                    val += scores[ki] * v[v_off + d];
                }
                output[out_off + d] = val;
            }
        }
    }
}

// Opt 1+2+3: scratch-based decode — 0 allocs on warm path.
// Opt 1: all activation buffers reused from scratch.
// Opt 2: score scratch slice replaces 192 per-head Vec allocs.
// Opt 3: logits via matmul_bt (Accelerate on macOS) replaces scalar loop.
#[cfg(feature = "bench-internals")]
#[allow(clippy::too_many_arguments)]
fn forward_scratch_decode(
    token_id: u32,
    start_pos: usize,
    cache: &mut FlatKVCache,
    embed_tokens: &[f32],
    lw: &SyntheticLayerWeights,
    norm_weight: &[f32],
    rope: &RopeTable,
    scratch: &mut ForwardBenchScratch,
) {
    let tok = (token_id as usize) % FWD_VOCAB;
    scratch
        .hidden
        .copy_from_slice(&embed_tokens[tok * FWD_HIDDEN..(tok + 1) * FWD_HIDDEN]);

    for layer in 0..FWD_LAYERS {
        scratch.residual.copy_from_slice(&scratch.hidden);
        rms_norm(
            &mut scratch.hidden,
            &lw.input_layernorm,
            FWD_HIDDEN,
            FWD_RMS_EPS,
        );

        matmul_bt(
            &scratch.hidden,
            &lw.qkv_proj,
            &mut scratch.qkv_buf,
            1,
            FWD_HIDDEN,
            FWD_QKV_DIM,
        );
        for j in 0..FWD_QKV_DIM {
            scratch.qkv_buf[j] += lw.qkv_bias[j];
        }

        scratch.q_buf.copy_from_slice(&scratch.qkv_buf[..FWD_Q_DIM]);
        scratch
            .k_buf
            .copy_from_slice(&scratch.qkv_buf[FWD_Q_DIM..FWD_Q_DIM + FWD_KV_DIM]);
        scratch
            .v_buf
            .copy_from_slice(&scratch.qkv_buf[FWD_Q_DIM + FWD_KV_DIM..]);

        for h in 0..FWD_N_HEADS {
            let off = h * FWD_HEAD_DIM;
            rms_norm(
                &mut scratch.q_buf[off..off + FWD_HEAD_DIM],
                &lw.q_norm,
                FWD_HEAD_DIM,
                FWD_RMS_EPS,
            );
        }
        for h in 0..FWD_N_KV_HEADS {
            let off = h * FWD_HEAD_DIM;
            rms_norm(
                &mut scratch.k_buf[off..off + FWD_HEAD_DIM],
                &lw.k_norm,
                FWD_HEAD_DIM,
                FWD_RMS_EPS,
            );
        }

        for h in 0..FWD_N_HEADS {
            rope.apply(
                &mut scratch.q_buf[h * FWD_HEAD_DIM..(h + 1) * FWD_HEAD_DIM],
                start_pos,
            );
        }
        for h in 0..FWD_N_KV_HEADS {
            rope.apply(
                &mut scratch.k_buf[h * FWD_HEAD_DIM..(h + 1) * FWD_HEAD_DIM],
                start_pos,
            );
        }

        {
            let dst_off = start_pos * FWD_KV_DIM;
            cache.k_buffer_mut(layer)[dst_off..dst_off + FWD_KV_DIM]
                .copy_from_slice(&scratch.k_buf);
            cache.v_buffer_mut(layer)[dst_off..dst_off + FWD_KV_DIM]
                .copy_from_slice(&scratch.v_buf);
        }

        let cached_seq_len = start_pos + 1;
        let k_end = cached_seq_len * FWD_KV_DIM;
        let score_len = FWD_N_HEADS * cached_seq_len;

        // Opt 2: pass score scratch slice — no Vec alloc per head
        // Need to split scratch borrows: attn_out + scores mut, q_buf imm
        let (attn_slice, score_slice) = {
            let attn = &mut scratch.attn_out as *mut Vec<f32>;
            let scores = &mut scratch.scores as *mut Vec<f32>;
            // SAFETY: attn_out and scores are distinct fields; no aliasing.
            unsafe { (&mut *attn, &mut *scores) }
        };
        compute_attention_scratch(
            attn_slice,
            &scratch.q_buf,
            &cache.k_buffer(layer)[..k_end],
            &cache.v_buffer(layer)[..k_end],
            1,
            cached_seq_len,
            start_pos,
            FWD_N_HEADS,
            FWD_N_KV_HEADS,
            FWD_HEAD_DIM,
            &mut score_slice[..score_len],
        );

        matmul_bt(
            &scratch.attn_out,
            &lw.o_proj,
            &mut scratch.hidden,
            1,
            FWD_Q_DIM,
            FWD_HIDDEN,
        );
        for i in 0..FWD_HIDDEN {
            scratch.hidden[i] += scratch.residual[i];
        }

        scratch.residual.copy_from_slice(&scratch.hidden);
        rms_norm(
            &mut scratch.hidden,
            &lw.post_attn_layernorm,
            FWD_HIDDEN,
            FWD_RMS_EPS,
        );

        matmul_bt(
            &scratch.hidden,
            &lw.gate_up_proj,
            &mut scratch.gate_up_buf,
            1,
            FWD_HIDDEN,
            2 * FWD_INTER,
        );
        scratch
            .gate_buf
            .copy_from_slice(&scratch.gate_up_buf[..FWD_INTER]);
        scratch
            .up_buf
            .copy_from_slice(&scratch.gate_up_buf[FWD_INTER..]);
        silu_inplace(&mut scratch.gate_buf);
        elementwise_mul(&mut scratch.gate_buf, &scratch.up_buf);

        matmul_bt(
            &scratch.gate_buf,
            &lw.down_proj,
            &mut scratch.ffn_out,
            1,
            FWD_INTER,
            FWD_HIDDEN,
        );
        for i in 0..FWD_HIDDEN {
            scratch.hidden[i] = scratch.residual[i] + scratch.ffn_out[i];
        }
    }

    rms_norm(&mut scratch.hidden, norm_weight, FWD_HIDDEN, FWD_RMS_EPS);
    // Opt 3: matmul_bt for logits (Accelerate on macOS, no scalar loop, no alloc)
    matmul_bt(
        &scratch.hidden,
        embed_tokens,
        &mut scratch.logits,
        1,
        FWD_HIDDEN,
        FWD_VOCAB,
    );
}

fn bench_forward_with_cache(c: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        // Build fixture (done once outside timing)
        let embed_tokens = match try_rand_f32_vec(FWD_VOCAB * FWD_HIDDEN, 0xFEED_0001) {
            Some(v) => v,
            None => {
                eprintln!(
                    "[bench_forward_with_cache] Cannot allocate {:.1} GiB embed_tokens — skip",
                    (FWD_VOCAB * FWD_HIDDEN * 4) as f64 / (1u64 << 30) as f64
                );
                return;
            }
        };
        let lw = SyntheticLayerWeights::new(0xFEED_0002);
        let norm_weight = rand_f32_vec(FWD_HIDDEN, 0xFEED_0003);
        let rope = RopeTable::new(FWD_HEAD_DIM, FWD_WARM_KV + 2, FWD_ROPE_THETA);

        let cache_cfg =
            FlatKVCacheConfig::for_qwen3(FWD_LAYERS, FWD_N_KV_HEADS, FWD_HEAD_DIM, FWD_WARM_KV + 2);

        // Precomputed warm K/V sequence (same token repeated — deterministic)
        let warm_kv_seq: Vec<f32> = vec![0.01f32; FWD_WARM_KV * FWD_KV_DIM];

        let make_warmed_cache = || {
            let mut cache = FlatKVCache::new(cache_cfg.clone());
            for layer in 0..FWD_LAYERS {
                cache.prefill_layer(layer, &warm_kv_seq, &warm_kv_seq, FWD_WARM_KV);
            }
            cache.advance_by(FWD_WARM_KV);
            cache
        };

        // Allocation-count report (before scratch — just prints, does not assert zero)
        {
            const MEASURED: usize = 1;
            allocation_count_print(
                "generate_forward_with_cache",
                "allocating_current",
                MEASURED,
                &mut || {
                    // Cache setup is NOT in the measured section
                    let mut cache = make_warmed_cache();
                    let snap = AllocationSnapshot::capture();
                    for _ in 0..MEASURED {
                        let _ = black_box(forward_allocating_decode(
                            42u32,
                            FWD_WARM_KV,
                            &mut cache,
                            &embed_tokens,
                            &lw,
                            &norm_weight,
                            &rope,
                        ));
                    }
                    AllocationSnapshot::delta_since(&snap)
                },
            );
        }

        // Criterion latency benchmark
        let mut group = c.benchmark_group("generate_forward_with_cache");
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
        group.throughput(Throughput::Elements(1)); // 1 token/iter → elements/s = tok/s

        group.bench_function("allocating_current", |b| {
            b.iter_batched(
                || make_warmed_cache(),
                |mut cache| {
                    black_box(forward_allocating_decode(
                        42u32,
                        FWD_WARM_KV,
                        &mut cache,
                        black_box(&embed_tokens),
                        &lw,
                        &norm_weight,
                        &rope,
                    ));
                },
                BatchSize::LargeInput,
            );
        });

        // --- Opt 1+2+3: scratch_dispatch ---
        // Allocation gate: should report 0 allocs on warm path
        {
            const MEASURED: usize = 1;
            let mut scratch = ForwardBenchScratch::new(FWD_WARM_KV + 1);
            allocation_count_report(
                "generate_forward_with_cache",
                "scratch_dispatch",
                MEASURED,
                &mut || {
                    let mut cache = make_warmed_cache();
                    let snap = AllocationSnapshot::capture();
                    for _ in 0..MEASURED {
                        black_box(forward_scratch_decode(
                            42u32,
                            FWD_WARM_KV,
                            &mut cache,
                            &embed_tokens,
                            &lw,
                            &norm_weight,
                            &rope,
                            &mut scratch,
                        ));
                    }
                    AllocationSnapshot::delta_since(&snap)
                },
            );
        }

        group.bench_function("scratch_dispatch", |b| {
            let mut scratch = ForwardBenchScratch::new(FWD_WARM_KV + 1);
            b.iter_batched(
                || make_warmed_cache(),
                |mut cache| {
                    black_box(forward_scratch_decode(
                        42u32,
                        FWD_WARM_KV,
                        &mut cache,
                        black_box(&embed_tokens),
                        &lw,
                        &norm_weight,
                        &rope,
                        &mut scratch,
                    ));
                },
                BatchSize::LargeInput,
            );
        });

        // --- decode_f32: scratch-dispatch decode step at varying context lengths ---
        // Throughput unit: 1 token per iteration (tok/s via Throughput::Elements(1)).
        // Uses the same forward_scratch_decode (0-alloc warm path) as scratch_dispatch.
        // Context lengths: seq1024, seq4096, seq16384.
        //
        // TODO(i2): decode_q8/seq{N} — same structure with PagedKVCache + CacheType::Q8.
        //   Requires: q8 page storage in paged.rs, gather dequant kernels, and
        //   apply_gqa_attention_paged in gqa.rs wired through forward_scratch_decode.
        let decode_contexts: &[(&str, usize)] =
            &[("seq1024", 1024), ("seq4096", 4096), ("seq16384", 16384)];

        for &(label, warm_kv) in decode_contexts {
            let warm_kv_seq_ctx =
                match try_rand_f32_vec(warm_kv * FWD_KV_DIM, 0xDEAD_0000u32 ^ warm_kv as u32) {
                    Some(v) => v,
                    None => {
                        eprintln!(
                            "[decode_f32/{label}] Cannot allocate {:.1} MiB warm_kv_seq — skip",
                            (warm_kv * FWD_KV_DIM * 4) as f64 / (1u64 << 20) as f64
                        );
                        continue;
                    }
                };

            let cache_cfg_ctx =
                FlatKVCacheConfig::for_qwen3(FWD_LAYERS, FWD_N_KV_HEADS, FWD_HEAD_DIM, warm_kv + 2);
            let rope_ctx = RopeTable::new(FWD_HEAD_DIM, warm_kv + 2, FWD_ROPE_THETA);

            let make_ctx_cache = || {
                let mut cache = FlatKVCache::new(cache_cfg_ctx.clone());
                for layer in 0..FWD_LAYERS {
                    cache.prefill_layer(layer, &warm_kv_seq_ctx, &warm_kv_seq_ctx, warm_kv);
                }
                cache.advance_by(warm_kv);
                cache
            };

            group.bench_function(BenchmarkId::new("decode_f32", label), |b| {
                let mut scratch = ForwardBenchScratch::new(warm_kv + 1);
                b.iter_batched(
                    make_ctx_cache,
                    |mut cache| {
                        black_box(forward_scratch_decode(
                            42u32,
                            warm_kv,
                            &mut cache,
                            black_box(&embed_tokens),
                            &lw,
                            &norm_weight,
                            &rope_ctx,
                            &mut scratch,
                        ));
                    },
                    BatchSize::LargeInput,
                );
            });
        }

        group.finish();
    }
    #[cfg(not(feature = "bench-internals"))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// MTP speculative decode mock benchmark
//
// NOTE: These are MOCK baselines only. Real numbers require model weights and
//   the MTP implementation that i2 adds to speculative.rs
//   (MtpTargetVerifier + mtp_verify_draft).
//
// Mock forward: deterministic random logits via xorshift32 (no model weights).
// Acceptance: xorshift32(draft_seq_pos) % 100 < accept_pct.
//
// Run (mock only, <30 s):
//   cargo bench -p lattice-inference --bench inference_perf \
//     --features bench-internals -- mtp_speculative_decode
//
// TODO(i2): When speculative::{MtpTargetVerifier, mtp_verify_draft} land,
//   replace BenchMtpTarget and mock_mtp_decode_loop with the real types.
// ---------------------------------------------------------------------------

const MOCK_MTP_VOCAB: usize = 8_192; // smaller than real (248 320) for <30 s
const MOCK_MTP_PROMPT_LEN: usize = 16;
const MOCK_MTP_GEN_TOKENS: usize = 64;

// Deterministic forward: generates MOCK_MTP_VOCAB f32 logits.
// seed_hi selects "model" (target=0xA000_0000, draft=0xD000_0000).
#[cfg(feature = "bench-internals")]
fn mtp_mock_forward(seed_hi: u32, step: usize) -> Vec<f32> {
    rand_f32_vec(
        MOCK_MTP_VOCAB,
        seed_hi ^ (step as u32).wrapping_mul(0x1000_0007),
    )
}

// Deterministic per-position acceptance: xorshift32 maps position to bool.
// accept_pct=100 → always, 75 → ~75%, 50 → ~50%.
#[cfg(feature = "bench-internals")]
fn mtp_mock_accept(accept_pct: u32, draft_seq_pos: u32) -> bool {
    let mut x = draft_seq_pos.wrapping_add(0x9E37_79B9);
    if x == 0 {
        x = 1;
    }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (x % 100) < accept_pct
}

// Local mirror of the planned speculative::MtpTargetVerifier trait.
// TODO(i2): Replace with `use lattice_inference::speculative::MtpTargetVerifier`
//   once the trait is added to speculative.rs.
#[cfg(feature = "bench-internals")]
trait BenchMtpTarget {
    fn cache_position(&self) -> usize;
    fn rollback_cache_to(&mut self, seq_len: usize);
    /// Returns one logit vec per token (exercises Vec<Vec<f32>> allocation).
    fn verify_tokens(&mut self, tokens: &[u32], start_pos: usize) -> Vec<Vec<f32>>;
}

#[cfg(feature = "bench-internals")]
struct MockBenchTarget {
    cache_pos: usize,
}

#[cfg(feature = "bench-internals")]
impl BenchMtpTarget for MockBenchTarget {
    fn cache_position(&self) -> usize {
        self.cache_pos
    }
    fn rollback_cache_to(&mut self, seq_len: usize) {
        self.cache_pos = seq_len;
    }
    fn verify_tokens(&mut self, tokens: &[u32], start_pos: usize) -> Vec<Vec<f32>> {
        let result = tokens
            .iter()
            .enumerate()
            .map(|(i, _)| black_box(mtp_mock_forward(0xE000_0000, start_pos + i)))
            .collect();
        self.cache_pos += tokens.len();
        result
    }
}

#[cfg(feature = "bench-internals")]
fn new_mock_target() -> MockBenchTarget {
    MockBenchTarget {
        cache_pos: MOCK_MTP_PROMPT_LEN,
    }
}

#[cfg(feature = "bench-internals")]
#[derive(Default, Clone, Copy)]
struct MockDecodeStats {
    generated_tokens: usize,
    target_forwards: usize,
    mtp_forwards: usize,
    accepted_tokens: usize,
}

#[cfg(feature = "bench-internals")]
impl MockDecodeStats {
    fn accepted_tokens_per_forward(self) -> f64 {
        if self.target_forwards == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.target_forwards as f64
        }
    }
    fn acceptance_rate(self) -> f64 {
        if self.mtp_forwards == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.mtp_forwards as f64
        }
    }
}

// Simulates MTP speculative decode loop with deterministic acceptance.
// Each "forward" call generates random logits to simulate allocation cost.
// target.verify_tokens exercises Vec<Vec<f32>> allocation matching the real trait.
//
// TODO(i2): When mtp_verify_draft lands, the round body should call:
//   speculative::mtp_verify_draft(verifier, current_token, pos, main_hidden,
//     &initial_logits, eos_token, &mut target)
#[cfg(feature = "bench-internals")]
fn mock_mtp_decode_loop(
    gen_tokens: usize,
    draft_len: usize,
    accept_pct: u32,
    target: &mut impl BenchMtpTarget,
) -> MockDecodeStats {
    let mut stats = MockDecodeStats::default();
    let mut draft_seq_pos: u32 = 0;
    // scratch token slice reused per round (avoid per-round allocation)
    let mut round_draft_tokens = Vec::with_capacity(draft_len);

    while stats.generated_tokens < gen_tokens {
        // Draft phase: up to draft_len MTP forwards; stop at first rejection.
        let mut round_accepted = 0usize;
        round_draft_tokens.clear();
        for _d in 0..draft_len {
            let _draft = black_box(mtp_mock_forward(0xD000_0000, stats.mtp_forwards));
            stats.mtp_forwards += 1;

            if mtp_mock_accept(accept_pct, draft_seq_pos) {
                draft_seq_pos = draft_seq_pos.wrapping_add(1);
                round_draft_tokens.push(draft_seq_pos); // mock token id
                round_accepted += 1;
                stats.accepted_tokens += 1;
                stats.generated_tokens += 1;
                if stats.generated_tokens >= gen_tokens {
                    break;
                }
            } else {
                draft_seq_pos = draft_seq_pos.wrapping_add(1);
                break;
            }
        }

        // Target verify: calls target.verify_tokens to exercise Vec<Vec<f32>> alloc.
        let start_pos = target.cache_position();
        let verify_tokens = if round_accepted > 0 {
            &round_draft_tokens[..round_accepted]
        } else {
            &round_draft_tokens[..0]
        };
        let _logits_vec = if !verify_tokens.is_empty() {
            black_box(target.verify_tokens(verify_tokens, start_pos))
        } else {
            // Full rejection: simulate one target forward for fallback token.
            let fallback = [0u32];
            black_box(target.verify_tokens(&fallback, start_pos))
        };
        target.rollback_cache_to(start_pos + round_accepted);
        stats.target_forwards += 1;

        // Full rejection emits one fallback token from target logits.
        if round_accepted == 0 && stats.generated_tokens < gen_tokens {
            stats.generated_tokens += 1;
        }
    }

    stats
}

fn bench_mtp_mock_speculative_decode(c: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        let mut group = c.benchmark_group("mtp_speculative_decode");
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
        group.throughput(Throughput::Elements(MOCK_MTP_GEN_TOKENS as u64));

        // Greedy baseline: MOCK_MTP_GEN_TOKENS target forwards, no draft phase.
        group.bench_function("greedy_64tok", |b| {
            b.iter(|| {
                let mut n = 0u32;
                for step in 0..MOCK_MTP_GEN_TOKENS {
                    let _logits = black_box(mtp_mock_forward(0xA000_0000, step));
                    n += 1;
                }
                black_box(n)
            });
        });

        // MTP mock at draft_length × acceptance_rate.
        // TODO(i2): Replace mock_mtp_decode_loop with real mtp_verify_draft.
        for draft_len in [2usize, 4, 8] {
            for (label_suffix, accept_pct) in [("50pct", 50u32), ("75pct", 75), ("100pct", 100)] {
                let label = format!("mtp_draft{draft_len}_{label_suffix}");
                group.bench_function(&label, |b| {
                    b.iter(|| {
                        let mut target = new_mock_target();
                        black_box(mock_mtp_decode_loop(
                            MOCK_MTP_GEN_TOKENS,
                            black_box(draft_len),
                            black_box(accept_pct),
                            &mut target,
                        ))
                    });
                });
            }
        }

        group.finish();

        // Summary table (outside Criterion timing).
        // NOTE: tok/s from Criterion above are mock baselines only.
        //   Real numbers require weights + speculative::mtp_verify_draft from i2.
        println!(
            "\n-- MTP Mock Baseline Summary \
             (mock_vocab={MOCK_MTP_VOCAB}, gen_tokens={MOCK_MTP_GEN_TOKENS}) --"
        );
        println!(
            "scenario,draft_length,acceptance_rate,accepted_tokens_per_forward,target_forwards,mtp_forwards,generated_tokens"
        );
        println!(
            "greedy,0,1.00,1.00,{},{},{}",
            MOCK_MTP_GEN_TOKENS, 0, MOCK_MTP_GEN_TOKENS
        );
        for draft_len in [2usize, 4, 8] {
            for (_, accept_pct) in [("50pct", 50u32), ("75pct", 75), ("100pct", 100)] {
                let s = mock_mtp_decode_loop(
                    MOCK_MTP_GEN_TOKENS,
                    draft_len,
                    accept_pct,
                    &mut new_mock_target(),
                );
                println!(
                    "mtp_mock,{draft_len},{:.2},{:.3},{},{},{}",
                    s.acceptance_rate(),
                    s.accepted_tokens_per_forward(),
                    s.target_forwards,
                    s.mtp_forwards,
                    s.generated_tokens,
                );
            }
        }
        println!("NOTE: Mock baselines. Real numbers require model weights + MTP implementation.");
    }
    #[cfg(not(feature = "bench-internals"))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// H3 RoPE profiling — piggyback on inference_perf
// ---------------------------------------------------------------------------

fn bench_rope_apply_decode(c: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        const ROPE_N_HEADS: usize = 28;
        const ROPE_N_KV_HEADS: usize = 4;
        const ROPE_HEAD_DIM: usize = 128;
        const ROPE_POSITION: usize = 2047;
        const ROPE_THETA: f64 = 1_000_000.0;

        let rope = RopeTable::new(ROPE_HEAD_DIM, ROPE_POSITION + 1, ROPE_THETA);
        let mut q = vec![0.0f32; ROPE_N_HEADS * ROPE_HEAD_DIM];
        let mut k = vec![0.0f32; ROPE_N_KV_HEADS * ROPE_HEAD_DIM];

        let mut group = c.benchmark_group("rope_apply_decode");
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_millis(500));
        group.measurement_time(std::time::Duration::from_secs(1));

        group.bench_function("qwen3_h28_kv4_pos2047", |b| {
            b.iter(|| {
                for h in 0..ROPE_N_HEADS {
                    rope.apply(
                        &mut q[h * ROPE_HEAD_DIM..(h + 1) * ROPE_HEAD_DIM],
                        ROPE_POSITION,
                    );
                }
                for h in 0..ROPE_N_KV_HEADS {
                    rope.apply(
                        &mut k[h * ROPE_HEAD_DIM..(h + 1) * ROPE_HEAD_DIM],
                        ROPE_POSITION,
                    );
                }
                black_box((&q, &k));
            });
        });

        group.finish();
    }
    #[cfg(not(feature = "bench-internals"))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// Criterion main
// ---------------------------------------------------------------------------

criterion_group!(
    name = perf_benches;
    config = Criterion::default();
    targets =
        bench_sampler_allocation,
        bench_simd_q8_neon_matvec,
        bench_kv_cache_paged,
        bench_tokenizer_bpe,
        bench_q8_neon_forward,
        bench_q8_neon_forward_allocations,
        bench_logits_projection,
        bench_forward_with_cache,
        bench_mtp_mock_speculative_decode,
        bench_rope_apply_decode,
);
criterion_main!(perf_benches);
