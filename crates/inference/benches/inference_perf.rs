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

#[allow(dead_code)]
struct AllocationSnapshot {
    alloc_calls: u64,
    realloc_calls: u64,
    bytes_allocated: u64,
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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
use lattice_inference::kv_cache::{EvictionPolicy, PagedKVCache, PagedKVCacheConfig};
#[cfg(feature = "bench-internals")]
use lattice_inference::kv_cache::{FlatKVCache, FlatKVCacheConfig};
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

    // Min-p on top of the default filters (#1394): exercises the min-p
    // cutoff over the top-k survivors.
    group.bench_function("default_topk50_topp0.9_minp0.05", |b| {
        b.iter_batched(
            || {
                Sampler::new(SamplingConfig::default())
                    .with_seed(0xDEAD_BEEF)
                    .with_min_p(0.05)
            },
            |mut sampler| black_box(sampler.sample(black_box(&logits))),
            BatchSize::SmallInput,
        );
    });

    // Min-p with top-k disabled (#1394): the cutoff runs over the full
    // vocabulary, the case where rejecting the tail matters most.
    group.bench_function("topk0_topp0.9_minp0.05_full_vocab", |b| {
        b.iter_batched(
            || {
                let mut cfg = SamplingConfig::default();
                cfg.top_k = 0;
                Sampler::new(cfg).with_seed(0xDEAD_BEEF).with_min_p(0.05)
            },
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
        let packed = pack_weights_q8(&w_f32, n, k).expect("valid benchmark matrix");

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
    let mut cache = PagedKVCache::try_new(paged_config(seq_len, 1))
        .expect("valid paged bench config must succeed");
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

    // TODO(#808): kv_append_q8/seq{N} — identical harness with cache_type_k=CacheType::Q8
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

    // TODO(#808): kv_gather_q8/seq{N} — identical harness with Q8 cache; dst is f32
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
//   ~/.lattice/models/qwen3.5-0.8b/tokenizer.json (or Qwen3.5-0.8B/).
// Falls back to a synthetic GPT-2-style BPE if no real tokenizer found.
// ---------------------------------------------------------------------------

fn qwen_tokenizer_path() -> Option<std::path::PathBuf> {
    let from_env = std::env::var("LATTICE_INFERENCE_MODEL_DIR")
        .ok()
        .map(|d| std::path::PathBuf::from(d).join("tokenizer.json"));

    let home = std::env::var("HOME").ok();
    let from_home_lower = home
        .as_deref()
        .map(|h| std::path::Path::new(h).join(".lattice/models/qwen3.5-0.8b/tokenizer.json"));
    let from_home_upper = home
        .as_deref()
        .map(|h| std::path::Path::new(h).join(".lattice/models/Qwen3.5-0.8B/tokenizer.json"));

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
                 ~/.lattice/models/qwen3.5-0.8b/tokenizer.json for real-model baselines."
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

// Builds the stderr heading for one allocation report. The suite is the caller's
// Criterion group, so a report can never be filed under another bench's name; an
// empty `label` collapses to `<suite>/<phase>` for benches whose only axis is the
// phase.
fn allocation_report_heading(suite: &str, label: &str, phase: &str) -> String {
    if label.is_empty() {
        format!("{suite}/{phase}")
    } else {
        format!("{suite}/{label}/{phase}")
    }
}

// Asserts that a zeroing allocation is visible to `CountingAlloc`. `GlobalAlloc`'s
// `alloc_zeroed` is not overridden above, so zeroed allocations are only counted
// while its default implementation routes through `alloc`. Overriding it later to
// recover the platform calloc path would stop counting a whole category, and every
// total would drop -- which reads as an improvement. This runs on the bench's own
// path rather than as a `#[test]`, because this target is declared `harness = false`
// and libtest never runs here.
fn assert_zeroed_allocation_is_counted() {
    let before = AllocationSnapshot::capture();
    let zeroed: Vec<u8> = vec![0; 4096];
    black_box(&zeroed);
    let delta = AllocationSnapshot::delta_since(&before);
    assert!(
        delta.alloc_calls > 0,
        "zeroed allocation was not observed by CountingAlloc: a 4 KiB vec![0; _] moved \
         alloc_calls by {} and bytes_allocated by {}; allocation totals in this bench \
         are undercounting every calloc-shaped allocation",
        delta.alloc_calls,
        delta.bytes_allocated,
    );
}

// Print allocation counts without asserting zero — use for "before" snapshots.
#[allow(dead_code)]
fn allocation_count_print(
    suite: &str,
    label: &str,
    phase: &str,
    tokens: usize,
    run_fn: &mut impl FnMut() -> AllocationDelta,
) {
    assert_zeroed_allocation_is_counted();
    let heading = allocation_report_heading(suite, label, phase);
    let samples: Vec<AllocationDelta> = (0..3).map(|_| run_fn()).collect();
    let s0 = samples[0];
    for (i, &s) in samples.iter().enumerate().skip(1) {
        if s != s0 {
            panic!(
                "allocation count is non-deterministic at run {i} for {heading}: \
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
        "\n{heading}\n\
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

#[allow(dead_code)]
fn allocation_count_report(
    suite: &str,
    label: &str,
    phase: &str,
    tokens: usize,
    run_fn: &mut impl FnMut() -> AllocationDelta,
) {
    assert_zeroed_allocation_is_counted();
    let heading = allocation_report_heading(suite, label, phase);
    let samples: Vec<AllocationDelta> = (0..3).map(|_| run_fn()).collect();

    let s0 = samples[0];
    for (i, &s) in samples.iter().enumerate().skip(1) {
        if s != s0 {
            panic!(
                "allocation count is non-deterministic at run {i} for {heading}: \
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
            "allocation gate failed for {heading}: \
             alloc_calls_total={} realloc_calls_total={} bytes_allocated_total={}",
            s0.alloc_calls, s0.realloc_calls, s0.bytes_allocated,
        );
    }

    let n = tokens as f64;
    eprintln!(
        "\n{heading}\n\
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

            allocation_count_report(
                "q8_neon_forward_allocations",
                "synthetic_2layer",
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
                "q8_neon_forward_allocations",
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
// ADR-090 D7 / row R01: per-token allocation instrument on the Qwen CPU
// shared-driver route.
//
// Extends the machinery above (`CountingAlloc`, `AllocationSnapshot`,
// `AllocationDelta`, `allocation_report_heading`) to the real ordinary-generation
// consumer. Nothing above is modified, so every other group in this binary is
// untouched; this group does no timing at all.
//
// Counted allocator entry points (declared, and each one proven to move by a
// positive control before any measurement is trusted):
//   alloc          counted in alloc_calls and bytes_allocated
//   alloc_zeroed   not overridden: the default method forwards to `alloc`, so a
//                  zeroed allocation lands in alloc_calls and bytes_allocated
//   realloc        counted in realloc_calls, and its new size in bytes_allocated
//   dealloc        tallied separately and not part of this gate
// Native, driver and GPU allocations are outside a Rust allocator and are not
// counted. bytes_allocated is requested bytes: a realloc contributes its whole
// new size, not the growth.
//
// Region: the warm decode/select/policy region of one streaming request, from the
// push of token WARM_IN_TOKENS to the push of the last completed token. Setup,
// tokenization, prefill, the first iterations and everything after the last push
// are outside it. The bracket is taken by the raw-token events the driver fires at
// the push, inside the worker that executes the request, so it measures execution
// and not enqueueing.
//
// Isolation: counters are process-wide because the CPU forward runs matmul on
// rayon threads whose allocations belong to the request. The request therefore
// runs alone in this process: one dedicated worker thread executes the warm-up and
// every repeat, the calling thread is parked in `join`, and a quiet-process probe
// precedes every arm. Unrelated concurrent allocation is rejected by the stability
// gate, and a control with a deliberate noise thread proves the gate does so.
//
// Run (the group only runs when selected by name):
//   RUSTC_WRAPPER="" cargo bench -p lattice-inference \
//     --features bench-internals,test-utils --bench inference_perf -- \
//     qwen_cpu_driver_allocations
// The real-checkpoint arm runs when a Qwen3.5-0.8B directory is found
// (LATTICE_CPU_GREEDY_MODEL_DIR, LATTICE_MODEL_DIR, LATTICE_INFERENCE_MODEL_DIR,
// then ~/.lattice/models/qwen3.5-0.8b) and prints a SKIPPED line otherwise. A skip
// is not a pass.
// ---------------------------------------------------------------------------

const QWEN_CPU_DRIVER_ALLOCATIONS: &str = "qwen_cpu_driver_allocations";

fn qwen_cpu_driver_allocations_selected() -> bool {
    std::env::args().any(|a| a.contains(QWEN_CPU_DRIVER_ALLOCATIONS))
}

#[cfg(all(feature = "bench-internals", feature = "test-utils"))]
mod qwen_cpu_driver_allocations {
    use super::{
        AllocationDelta, AllocationSnapshot, QWEN_CPU_DRIVER_ALLOCATIONS,
        allocation_report_heading, assert_zeroed_allocation_is_counted,
    };
    use lattice_inference::GenerateConfig;
    use lattice_inference::decoder_bench_support::{
        InterfaceProbe, RouteRun, run_interface_only, run_qwen_cpu_probed, run_qwen_cpu_streaming,
    };
    use lattice_inference::model::qwen35::test_support::tiny_zero_model;
    use lattice_inference::model::qwen35::{Qwen35Model, RawGenEvent};
    use std::hint::black_box;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Measured repeats after one warm-up run. The repeats must agree exactly.
    const REPEATS: usize = 3;
    /// The region opens at the push of this token. Tokens before it are warm-in.
    const WARM_IN_TOKENS: usize = 3;
    const SYNTHETIC_TOKENS: usize = 48;
    const REAL_TOKENS: usize = 16;
    const INTERFACE_ONLY_PROMPT_TOKENS: usize = 8;
    const SYNTHETIC_PROMPT: &str = "abc abc abc abc";
    const REAL_PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Route {
        Streaming,
        Probed,
        InterfaceOnly,
    }

    impl Route {
        fn name(self) -> &'static str {
            match self {
                Route::Streaming => "qwen_cpu_streaming",
                Route::Probed => "qwen_cpu_probed",
                Route::InterfaceOnly => "interface_only",
            }
        }
    }

    /// What a probe does inside the per-token interface. Only the bench ever holds
    /// the retaining code; no library build contains it.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Injection {
        None,
        AllocOnSelect,
        AllocOnDecode,
        ReallocOnDecode,
    }

    impl Injection {
        fn name(self) -> &'static str {
            match self {
                Injection::None => "unchanged",
                Injection::AllocOnSelect => "retained_alloc_in_select",
                Injection::AllocOnDecode => "retained_alloc_in_decode",
                Injection::ReallocOnDecode => "retained_realloc_in_decode",
            }
        }
    }

    struct Profile {
        name: &'static str,
        cfg: GenerateConfig,
    }

    fn greedy_profile(max_new_tokens: usize, stop_token_ids: Vec<u32>) -> Profile {
        let mut cfg = GenerateConfig::default();
        cfg.max_new_tokens = max_new_tokens;
        cfg.temperature = 0.0;
        cfg.repetition_penalty = 1.0;
        cfg.seed = Some(7);
        cfg.enable_thinking = false;
        cfg.stop_token_ids = stop_token_ids;
        Profile { name: GREEDY, cfg }
    }

    fn sampled_profile(max_new_tokens: usize, stop_token_ids: Vec<u32>) -> Profile {
        let mut cfg = GenerateConfig::default();
        cfg.max_new_tokens = max_new_tokens;
        cfg.temperature = 0.7;
        cfg.top_k = 50;
        cfg.top_p = 0.9;
        cfg.repetition_penalty = 1.1;
        cfg.seed = Some(7);
        cfg.enable_thinking = false;
        cfg.stop_token_ids = stop_token_ids;
        Profile { name: SAMPLED, cfg }
    }

    /// Probe that counts its own calls and, per `injection`, retains an allocation or
    /// a reallocation inside the interface. Everything it will touch is sized before
    /// the request starts, so an unchanged probe allocates nothing.
    struct BenchProbe {
        injection: Injection,
        selects: usize,
        decodes: usize,
        // One heap allocation per retained element is the point of the injection.
        #[allow(clippy::vec_box)]
        sink: Vec<Box<[u8; 64]>>,
        grow: Vec<u8>,
    }

    impl BenchProbe {
        fn new(injection: Injection, capacity: usize) -> Self {
            Self {
                injection,
                selects: 0,
                decodes: 0,
                sink: match injection {
                    Injection::AllocOnSelect | Injection::AllocOnDecode => {
                        Vec::with_capacity(capacity)
                    }
                    _ => Vec::new(),
                },
                grow: match injection {
                    Injection::ReallocOnDecode => Vec::with_capacity(1),
                    _ => Vec::new(),
                },
            }
        }
    }

    impl InterfaceProbe for BenchProbe {
        fn after_select(&mut self) {
            self.selects += 1;
            if self.injection == Injection::AllocOnSelect {
                self.sink.push(Box::new(black_box([0u8; 64])));
            }
        }

        fn after_decode(&mut self) {
            self.decodes += 1;
            match self.injection {
                Injection::AllocOnDecode => self.sink.push(Box::new(black_box([0u8; 64]))),
                Injection::ReallocOnDecode => {
                    // Exact growth by one byte: every call after the first reallocates.
                    self.grow.reserve_exact(1);
                    self.grow.push(0);
                }
                _ => {}
            }
        }
    }

    struct ArmSpec<'a> {
        model_label: &'a str,
        route: Route,
        injection: Injection,
        model: Option<&'a Qwen35Model>,
        prompt: &'a str,
        profile: &'a Profile,
    }

    /// One request's region measurement.
    #[derive(Clone, Debug)]
    struct Sample {
        delta: AllocationDelta,
        region_tokens: usize,
        completed_tokens: usize,
        prompt_tokens: usize,
        opened: usize,
        consumed: usize,
        selects: usize,
        decodes: usize,
        initial_kv_capacity_floats: Option<usize>,
        token_ids: Vec<u32>,
        /// Per-iteration (alloc_calls, realloc_calls, bytes) inside the region.
        series: Vec<(u64, u64, u64)>,
    }

    /// Collects one snapshot per pushed token into storage sized before the request.
    struct Recorder {
        points: Vec<(usize, AllocationSnapshot)>,
    }

    impl Recorder {
        fn with_capacity(max_new_tokens: usize) -> Self {
            Self {
                points: Vec::with_capacity(max_new_tokens + 4),
            }
        }

        fn on_event(&mut self, event: RawGenEvent) {
            if let RawGenEvent::RawToken { index } = event {
                self.points.push((index, AllocationSnapshot::capture()));
            }
        }
    }

    fn sample_from(rec: Recorder, run: RouteRun, probe: &BenchProbe) -> Sample {
        let mut delta = AllocationDelta {
            alloc_calls: 0,
            realloc_calls: 0,
            bytes_allocated: 0,
        };
        let mut region_tokens = 0usize;
        let mut series = Vec::new();
        let start = rec.points.iter().position(|(i, _)| *i == WARM_IN_TOKENS);
        if let (Some(s), Some(last)) = (start, rec.points.last()) {
            let first = &rec.points[s];
            if last.0 > first.0 {
                region_tokens = last.0 - first.0;
                delta = AllocationDelta {
                    alloc_calls: last.1.alloc_calls - first.1.alloc_calls,
                    realloc_calls: last.1.realloc_calls - first.1.realloc_calls,
                    bytes_allocated: last.1.bytes_allocated - first.1.bytes_allocated,
                };
                for w in rec.points[s..].windows(2) {
                    series.push((
                        w[1].1.alloc_calls - w[0].1.alloc_calls,
                        w[1].1.realloc_calls - w[0].1.realloc_calls,
                        w[1].1.bytes_allocated - w[0].1.bytes_allocated,
                    ));
                }
            }
        }
        Sample {
            delta,
            region_tokens,
            completed_tokens: run.token_ids.len(),
            prompt_tokens: run.prompt_tokens,
            opened: run.opened,
            consumed: run.consumed,
            selects: probe.selects,
            decodes: probe.decodes,
            initial_kv_capacity_floats: run.initial_kv_capacity_floats,
            token_ids: run.token_ids,
            series,
        }
    }

    fn run_once(spec: &ArmSpec<'_>) -> Sample {
        let cfg = &spec.profile.cfg;
        let mut probe = BenchProbe::new(spec.injection, cfg.max_new_tokens + 4);
        let mut rec = Recorder::with_capacity(cfg.max_new_tokens);
        let run = {
            let mut on_event = |e: RawGenEvent| rec.on_event(e);
            match (spec.route, spec.model) {
                (Route::Streaming, Some(model)) => {
                    run_qwen_cpu_streaming(model, spec.prompt, cfg, &mut on_event)
                }
                (Route::Probed, Some(model)) => {
                    run_qwen_cpu_probed(model, spec.prompt, cfg, &mut probe, &mut on_event)
                }
                (Route::InterfaceOnly, _) => {
                    run_interface_only(INTERFACE_ONLY_PROMPT_TOKENS, cfg, &mut probe, &mut on_event)
                }
                _ => panic!("route {} needs a model", spec.route.name()),
            }
        }
        .unwrap_or_else(|e| panic!("{} run failed: {e}", spec.route.name()));
        sample_from(rec, run, &probe)
    }

    /// The process must be quiet before an arm starts: no allocation by anything else.
    fn assert_process_quiet() {
        let before = AllocationSnapshot::capture();
        std::thread::sleep(Duration::from_millis(25));
        let idle = AllocationSnapshot::delta_since(&before);
        assert_eq!(
            idle,
            AllocationDelta {
                alloc_calls: 0,
                realloc_calls: 0,
                bytes_allocated: 0
            },
            "the process is not quiet: {idle:?} allocated in 25 ms with no request running, so a \
             measurement now would be contaminated"
        );
    }

    /// One worker executes the warm-up and every repeat; the caller is parked in join.
    fn measure_unchecked(spec: &ArmSpec<'_>) -> Vec<Sample> {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .name("alloc-instrument-worker".into())
                .spawn_scoped(scope, || {
                    let _warm_up = run_once(spec);
                    (0..REPEATS).map(|_| run_once(spec)).collect::<Vec<_>>()
                })
                .expect("spawning the measurement worker")
                .join()
                .expect("the measurement worker panicked")
        })
    }

    fn measure(spec: &ArmSpec<'_>) -> Vec<Sample> {
        assert_process_quiet();
        measure_unchecked(spec)
    }

    /// Accepts a set of repeats as one certified sample, or names why it refuses.
    fn certify(samples: &[Sample], need_hooks: bool) -> Result<Sample, String> {
        let first = samples
            .first()
            .ok_or_else(|| "no samples: the group did not run".to_string())?;
        if samples.len() != REPEATS {
            return Err(format!("expected {REPEATS} repeats, got {}", samples.len()));
        }
        if first.region_tokens == 0 || first.completed_tokens == 0 {
            return Err(format!(
                "zero tokens in the region (completed {}, region {})",
                first.completed_tokens, first.region_tokens
            ));
        }
        if first.consumed + 1 != first.opened {
            return Err(format!(
                "the run did not go through the shared driver (opened {}, consumed {})",
                first.opened, first.consumed
            ));
        }
        if need_hooks
            && (first.selects < first.region_tokens || first.decodes < first.region_tokens)
        {
            return Err(format!(
                "interface hooks did not fire for the region (selects {}, decodes {}, region {})",
                first.selects, first.decodes, first.region_tokens
            ));
        }
        for (i, s) in samples.iter().enumerate().skip(1) {
            if s.delta != first.delta
                || s.region_tokens != first.region_tokens
                || s.completed_tokens != first.completed_tokens
                || s.token_ids != first.token_ids
                || s.series != first.series
            {
                return Err(format!(
                    "unstable repeats at run {i}: run0=({},{},{}) tokens={} run{i}=({},{},{}) tokens={}",
                    first.delta.alloc_calls,
                    first.delta.realloc_calls,
                    first.delta.bytes_allocated,
                    first.region_tokens,
                    s.delta.alloc_calls,
                    s.delta.realloc_calls,
                    s.delta.bytes_allocated,
                    s.region_tokens,
                ));
            }
        }
        Ok(first.clone())
    }

    fn certify_or_panic(label: &str, samples: &[Sample], need_hooks: bool) -> Sample {
        certify(samples, need_hooks).unwrap_or_else(|why| panic!("REFUSE {label}: {why}"))
    }

    #[derive(Clone, Debug)]
    enum Verdict {
        Pass,
        Fail(String),
        Refuse(String),
    }

    impl Verdict {
        fn kind(&self) -> &'static str {
            match self {
                Verdict::Pass => "PASS",
                Verdict::Fail(_) => "FAIL",
                Verdict::Refuse(_) => "REFUSE",
            }
        }

        fn text(&self) -> String {
            match self {
                Verdict::Pass => "PASS".to_string(),
                Verdict::Fail(why) => format!("FAIL({why})"),
                Verdict::Refuse(why) => format!("REFUSE({why})"),
            }
        }
    }

    /// The incremental comparison: the candidate may not add one allocation call, one
    /// reallocation call or one requested byte over the base, each counter on its own.
    fn compare_incremental(base: &Sample, cand: &Sample) -> Verdict {
        if base.region_tokens == 0 || cand.region_tokens == 0 {
            return Verdict::Refuse("zero tokens".into());
        }
        if base.region_tokens != cand.region_tokens
            || base.completed_tokens != cand.completed_tokens
        {
            return Verdict::Refuse(format!(
                "token counts differ: base {}/{} candidate {}/{}",
                base.region_tokens,
                base.completed_tokens,
                cand.region_tokens,
                cand.completed_tokens
            ));
        }
        let mut added = Vec::new();
        let pairs = [
            (
                "alloc_calls",
                base.delta.alloc_calls,
                cand.delta.alloc_calls,
            ),
            (
                "realloc_calls",
                base.delta.realloc_calls,
                cand.delta.realloc_calls,
            ),
            (
                "bytes_allocated",
                base.delta.bytes_allocated,
                cand.delta.bytes_allocated,
            ),
        ];
        for (name, b, c) in pairs {
            if c > b {
                added.push(format!("{name} +{}", c - b));
            }
        }
        if added.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail(added.join(", "))
        }
    }

    /// The real consumer and the interface-only run must both stay within base. A
    /// removal elsewhere in a model can cancel an interface addition in the first;
    /// it cannot in the second.
    fn combine(real: &Verdict, interface: &Verdict) -> Verdict {
        match (real, interface) {
            (Verdict::Refuse(a), _) | (_, Verdict::Refuse(a)) => Verdict::Refuse(a.clone()),
            (Verdict::Fail(a), Verdict::Fail(b)) => {
                Verdict::Fail(format!("real: {a}; interface-only: {b}"))
            }
            (Verdict::Fail(a), _) => Verdict::Fail(format!("real: {a}")),
            (_, Verdict::Fail(b)) => Verdict::Fail(format!("interface-only: {b}")),
            _ => Verdict::Pass,
        }
    }

    fn counts(s: &Sample) -> String {
        format!(
            "({},{},{})",
            s.delta.alloc_calls, s.delta.realloc_calls, s.delta.bytes_allocated
        )
    }

    fn control(name: &str, expected: &str, base: &Sample, cand: &Sample, verdict: &Verdict) {
        eprintln!(
            "CONTROL {name}: expected={expected} observed={} base={} candidate={} tokens={}",
            verdict.text(),
            counts(base),
            counts(cand),
            base.region_tokens
        );
        assert_eq!(
            verdict.kind(),
            expected,
            "control failed: {name} expected {expected} and observed {}",
            verdict.text()
        );
    }

    fn print_record(
        model_label: &str,
        route: Route,
        injection: Injection,
        profile: &str,
        s: &Sample,
    ) {
        let n = s.region_tokens as f64;
        let heading =
            allocation_report_heading(QWEN_CPU_DRIVER_ALLOCATIONS, model_label, route.name());
        let iterations_with_alloc = s.series.iter().filter(|p| p.0 > 0).count();
        let iterations_with_realloc = s.series.iter().filter(|p| p.1 > 0).count();
        // An iteration ending at the push of token k is series[k - WARM_IN_TOKENS - 1].
        // Each entry is (push index, alloc calls, realloc calls, requested bytes).
        let realloc_iterations: Vec<(usize, u64, u64, u64)> = s
            .series
            .iter()
            .enumerate()
            .filter(|(_, p)| p.1 > 0)
            .map(|(i, p)| (WARM_IN_TOKENS + 1 + i, p.0, p.1, p.2))
            .collect();
        let quiet_bytes = s.series.iter().filter(|p| p.1 == 0).map(|p| p.2);
        let (quiet_min, quiet_max) =
            quiet_bytes.fold((u64::MAX, 0u64), |(lo, hi), b| (lo.min(b), hi.max(b)));
        let capacity = s
            .initial_kv_capacity_floats
            .map_or("n/a".to_string(), |c| c.to_string());
        eprintln!(
            "\n{heading}\n\
             variant={} profile={profile}\n\
             prompt_tokens={} completed_tokens={} tokens={} region_from_push={WARM_IN_TOKENS} \
             initial_kv_capacity_floats={capacity} driver_opened={} driver_consumed={}\n\
             alloc_calls_total={}\n\
             realloc_calls_total={}\n\
             bytes_allocated_total={}\n\
             allocations_per_token={:.2}\n\
             reallocations_per_token={:.2}\n\
             bytes_allocated_per_token={:.0}\n\
             iterations_with_alloc={iterations_with_alloc} iterations_with_realloc={iterations_with_realloc} \
             realloc_iterations={realloc_iterations:?}\n\
             bytes_in_iterations_without_realloc_min={quiet_min} max={quiet_max} repeats={REPEATS}",
            injection.name(),
            s.prompt_tokens,
            s.completed_tokens,
            s.region_tokens,
            s.opened,
            s.consumed,
            s.delta.alloc_calls,
            s.delta.realloc_calls,
            s.delta.bytes_allocated,
            s.delta.alloc_calls as f64 / n,
            s.delta.realloc_calls as f64 / n,
            s.delta.bytes_allocated as f64 / n,
        );
        eprintln!(
            "record model={model_label} route={} variant={} profile={profile} prompt_tokens={} \
             completed_tokens={} tokens={} initial_kv_capacity_floats={capacity} alloc_calls={} \
             realloc_calls={} bytes_requested={}",
            route.name(),
            injection.name(),
            s.prompt_tokens,
            s.completed_tokens,
            s.region_tokens,
            s.delta.alloc_calls,
            s.delta.realloc_calls,
            s.delta.bytes_allocated,
        );
    }

    fn arm<'a>(
        model_label: &'a str,
        route: Route,
        injection: Injection,
        model: Option<&'a Qwen35Model>,
        prompt: &'a str,
        profile: &'a Profile,
    ) -> ArmSpec<'a> {
        ArmSpec {
            model_label,
            route,
            injection,
            model,
            prompt,
            profile,
        }
    }

    // Every group the suite is designed to produce is listed by `required_groups`. A group
    // that silently stops running leaves the ledger short and the instrument refuses to
    // report.
    static CERTIFIED_GROUPS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    /// Every group the suite is designed to produce. A group that silently stops running
    /// leaves the ledger short and the instrument refuses to report.
    const GREEDY: &str = "greedy_t0_rp1.0_seed7";
    const SAMPLED: &str = "sampled_t0.7_k50_p0.9_rp1.1_seed7";

    fn required_groups(
        model: &str,
        with_interface_only: bool,
        with_offset: bool,
        sampled_full: bool,
    ) -> Vec<String> {
        let mut keys = Vec::new();
        if with_interface_only {
            for variant in [
                "unchanged",
                "retained_alloc_in_select",
                "retained_realloc_in_decode",
            ] {
                keys.push(format!("interface_only/interface_only/{variant}/{GREEDY}"));
            }
        }
        let mut arms = vec![
            ("qwen_cpu_streaming", "unchanged"),
            ("qwen_cpu_probed", "unchanged"),
            ("qwen_cpu_probed", "retained_alloc_in_select"),
            ("qwen_cpu_probed", "retained_realloc_in_decode"),
        ];
        if with_offset {
            arms.push(("qwen_cpu_probed", "retained_alloc_in_decode"));
        }
        for (route, variant) in arms {
            keys.push(format!("{model}/{route}/{variant}/{GREEDY}"));
        }
        if sampled_full {
            for (route, variant) in [
                ("qwen_cpu_streaming", "unchanged"),
                ("qwen_cpu_probed", "unchanged"),
                ("qwen_cpu_probed", "retained_alloc_in_select"),
                ("qwen_cpu_probed", "retained_realloc_in_decode"),
            ] {
                keys.push(format!("{model}/{route}/{variant}/{SAMPLED}"));
            }
        } else {
            keys.push(format!("{model}/qwen_cpu_streaming/unchanged/{SAMPLED}"));
        }
        keys
    }

    fn require_groups(required: &[String]) {
        let ledger = CERTIFIED_GROUPS.lock().expect("group ledger poisoned");
        let missing: Vec<&String> = required
            .iter()
            .filter(|key| !ledger.iter().any(|seen| seen == *key))
            .collect();
        assert!(
            missing.is_empty(),
            "REFUSE: groups the suite is designed to run produced no certified sample: {missing:?}"
        );
    }

    fn measure_certified(spec: &ArmSpec<'_>) -> Sample {
        let need_hooks = spec.route != Route::Streaming;
        let samples = measure(spec);
        let sample = certify_or_panic(
            &format!(
                "{}/{}/{}",
                spec.model_label,
                spec.route.name(),
                spec.injection.name()
            ),
            &samples,
            need_hooks,
        );
        print_record(
            spec.model_label,
            spec.route,
            spec.injection,
            spec.profile.name,
            &sample,
        );
        CERTIFIED_GROUPS
            .lock()
            .expect("group ledger poisoned")
            .push(format!(
                "{}/{}/{}/{}",
                spec.model_label,
                spec.route.name(),
                spec.injection.name(),
                spec.profile.name
            ));
        sample
    }

    /// Each declared entry point moves its own counter, and an idle region moves none.
    fn entry_point_controls() {
        assert_zeroed_allocation_is_counted();

        let before = AllocationSnapshot::capture();
        let boxed = Box::new(black_box(0xA5A5_A5A5_A5A5_A5A5u64));
        let d = AllocationSnapshot::delta_since(&before);
        black_box(&boxed);
        assert!(
            d.alloc_calls >= 1 && d.bytes_allocated >= 8 && d.realloc_calls == 0,
            "control failed: alloc entry point did not move alloc_calls alone: {d:?}"
        );
        eprintln!(
            "CONTROL entry_point_alloc: expected=alloc_calls>=1,realloc_calls=0 observed={d:?}"
        );

        let before = AllocationSnapshot::capture();
        let zeroed = vec![0u8; 65_537];
        let d = AllocationSnapshot::delta_since(&before);
        black_box(&zeroed);
        assert!(
            d.alloc_calls >= 1 && d.bytes_allocated >= 65_537 && d.realloc_calls == 0,
            "control failed: a zeroed allocation was not observed: {d:?}"
        );
        eprintln!(
            "CONTROL entry_point_alloc_zeroed: expected=alloc_calls>=1,bytes>=65537,realloc_calls=0 observed={d:?}"
        );

        let mut grown: Vec<u8> = Vec::with_capacity(16);
        grown.push(1);
        let before = AllocationSnapshot::capture();
        grown.reserve_exact(4096);
        let d = AllocationSnapshot::delta_since(&before);
        black_box(&grown);
        assert!(
            d.realloc_calls == 1 && d.alloc_calls == 0 && d.bytes_allocated >= 4097,
            "control failed: a reallocation was not observed as a realloc: {d:?}"
        );
        eprintln!(
            "CONTROL entry_point_realloc: expected=realloc_calls=1,alloc_calls=0 observed={d:?}"
        );

        let before = AllocationSnapshot::capture();
        let idle = AllocationSnapshot::delta_since(&before);
        assert_eq!(
            idle,
            AllocationDelta {
                alloc_calls: 0,
                realloc_calls: 0,
                bytes_allocated: 0
            },
            "control failed: an empty region moved a counter"
        );
        eprintln!("CONTROL entry_point_idle_region: expected=0,0,0 observed={idle:?}");
    }

    struct InterfaceEvidence {
        clean: Sample,
        clean_again: Sample,
        alloc_in_select: Sample,
        realloc_in_decode: Sample,
    }

    /// The bounded interface-only run: the shared driver over a session that does no
    /// model work, unchanged and with each retained injection.
    fn interface_only_evidence(profile: &Profile) -> InterfaceEvidence {
        eprintln!(
            "\n# interface-only control: the shared driver over a session with no model work"
        );
        let label = "interface_only";
        let clean = measure_certified(&arm(
            label,
            Route::InterfaceOnly,
            Injection::None,
            None,
            "",
            profile,
        ));
        let clean_again = measure_certified(&arm(
            label,
            Route::InterfaceOnly,
            Injection::None,
            None,
            "",
            profile,
        ));
        let alloc_in_select = measure_certified(&arm(
            label,
            Route::InterfaceOnly,
            Injection::AllocOnSelect,
            None,
            "",
            profile,
        ));
        let realloc_in_decode = measure_certified(&arm(
            label,
            Route::InterfaceOnly,
            Injection::ReallocOnDecode,
            None,
            "",
            profile,
        ));

        let v = compare_incremental(&clean, &clean_again);
        control(
            "interface_only_unchanged_passes",
            "PASS",
            &clean,
            &clean_again,
            &v,
        );

        let v = compare_incremental(&clean, &alloc_in_select);
        control(
            "interface_only_retained_alloc_fails",
            "FAIL",
            &clean,
            &alloc_in_select,
            &v,
        );
        let t = clean.region_tokens as u64;
        assert_eq!(
            alloc_in_select.delta.alloc_calls - clean.delta.alloc_calls,
            t,
            "control failed: the injected allocation is not attributed one per token"
        );
        assert_eq!(
            alloc_in_select.delta.bytes_allocated - clean.delta.bytes_allocated,
            64 * t,
            "control failed: the injected bytes are not 64 per token"
        );

        let v = compare_incremental(&clean, &realloc_in_decode);
        control(
            "interface_only_retained_realloc_fails",
            "FAIL",
            &clean,
            &realloc_in_decode,
            &v,
        );
        assert_eq!(
            realloc_in_decode.delta.realloc_calls - clean.delta.realloc_calls,
            t,
            "control failed: the injected reallocation is not attributed one per token"
        );
        assert_eq!(
            realloc_in_decode.delta.alloc_calls, clean.delta.alloc_calls,
            "control failed: a retained reallocation moved alloc_calls"
        );

        InterfaceEvidence {
            clean,
            clean_again,
            alloc_in_select,
            realloc_in_decode,
        }
    }

    /// The route suite on one model and profile: the clean base, the unchanged probe,
    /// and the two retained injections, each judged together with the interface-only run.
    fn route_suite(
        model_label: &str,
        model: &Qwen35Model,
        prompt: &str,
        profile: &Profile,
        iface: &InterfaceEvidence,
    ) -> Sample {
        eprintln!(
            "\n# route suite: model={model_label} profile={}",
            profile.name
        );
        let base = measure_certified(&arm(
            model_label,
            Route::Streaming,
            Injection::None,
            Some(model),
            prompt,
            profile,
        ));

        // The probed route must be the production route plus a decorator, nothing else.
        let unchanged = measure_certified(&arm(
            model_label,
            Route::Probed,
            Injection::None,
            Some(model),
            prompt,
            profile,
        ));
        assert_eq!(
            unchanged.token_ids, base.token_ids,
            "control failed: the probed route generated different tokens from the production route"
        );
        let v = combine(
            &compare_incremental(&base, &unchanged),
            &compare_incremental(&iface.clean, &iface.clean_again),
        );
        control(
            "unchanged_warmed_path_passes",
            "PASS",
            &base,
            &unchanged,
            &v,
        );
        assert_eq!(
            counts(&base),
            counts(&unchanged),
            "control failed: the probed route does not reproduce the production route's counts"
        );

        let t = base.region_tokens as u64;

        let alloc = measure_certified(&arm(
            model_label,
            Route::Probed,
            Injection::AllocOnSelect,
            Some(model),
            prompt,
            profile,
        ));
        let v = combine(
            &compare_incremental(&base, &alloc),
            &compare_incremental(&iface.clean, &iface.alloc_in_select),
        );
        control(
            "retained_alloc_in_interface_fails",
            "FAIL",
            &base,
            &alloc,
            &v,
        );
        assert_eq!(
            alloc.delta.alloc_calls - base.delta.alloc_calls,
            t,
            "control failed: the injected allocation is not attributed one per token"
        );
        assert_eq!(
            alloc.delta.bytes_allocated - base.delta.bytes_allocated,
            64 * t,
            "control failed: the injected bytes are not 64 per token"
        );

        let realloc = measure_certified(&arm(
            model_label,
            Route::Probed,
            Injection::ReallocOnDecode,
            Some(model),
            prompt,
            profile,
        ));
        let v = combine(
            &compare_incremental(&base, &realloc),
            &compare_incremental(&iface.clean, &iface.realloc_in_decode),
        );
        control(
            "retained_realloc_in_interface_fails",
            "FAIL",
            &base,
            &realloc,
            &v,
        );
        assert_eq!(
            realloc.delta.realloc_calls - base.delta.realloc_calls,
            t,
            "control failed: the injected reallocation is not attributed one per token"
        );
        assert_eq!(
            realloc.delta.alloc_calls, base.delta.alloc_calls,
            "control failed: a retained reallocation moved alloc_calls"
        );

        base
    }

    /// An addition in the interface cancelled by an unrelated removal on the real
    /// consumer: the real comparison alone is blind to it, the combined one is not.
    fn offset_control(
        model_label: &str,
        model: &Qwen35Model,
        prompt: &str,
        profile: &Profile,
        iface: &InterfaceEvidence,
    ) {
        let base_with_unrelated = measure_certified(&arm(
            model_label,
            Route::Probed,
            Injection::AllocOnDecode,
            Some(model),
            prompt,
            profile,
        ));
        let head = measure_certified(&arm(
            model_label,
            Route::Probed,
            Injection::AllocOnSelect,
            Some(model),
            prompt,
            profile,
        ));
        let real_only = compare_incremental(&base_with_unrelated, &head);
        control(
            "offset_real_consumer_alone_is_blind",
            "PASS",
            &base_with_unrelated,
            &head,
            &real_only,
        );
        let combined = combine(
            &real_only,
            &compare_incremental(&iface.clean, &iface.alloc_in_select),
        );
        control(
            "offset_caught_by_interface_only_control",
            "FAIL",
            &base_with_unrelated,
            &head,
            &combined,
        );
    }

    /// Contamination, zero tokens, a missing group and a missing route marker are refused.
    fn refusal_controls(model_label: &str, model: &Qwen35Model, prompt: &str, profile: &Profile) {
        let spec = arm(
            model_label,
            Route::Streaming,
            Injection::None,
            Some(model),
            prompt,
            profile,
        );

        let stop = AtomicBool::new(false);
        let noisy = std::thread::scope(|scope| {
            let noise = scope.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    black_box(Box::new(black_box([0u8; 32])));
                }
            });
            let samples = measure_unchecked(&spec);
            stop.store(true, Ordering::Relaxed);
            noise.join().expect("the noise thread panicked");
            samples
        });
        let refusal = certify(&noisy, false);
        eprintln!(
            "CONTROL concurrent_noise_is_refused: expected=REFUSE(unstable) observed={}",
            match &refusal {
                Ok(_) => "ACCEPTED".to_string(),
                Err(why) => format!("REFUSE({why})"),
            }
        );
        assert!(
            matches!(&refusal, Err(why) if why.contains("unstable")),
            "control failed: contaminated repeats were certified"
        );

        let mut short = Profile {
            name: profile.name,
            cfg: profile.cfg.clone(),
        };
        short.cfg.max_new_tokens = WARM_IN_TOKENS;
        let zero = measure(&arm(
            model_label,
            Route::Streaming,
            Injection::None,
            Some(model),
            prompt,
            &short,
        ));
        let refusal = certify(&zero, false);
        eprintln!(
            "CONTROL zero_tokens_is_refused: expected=REFUSE(zero tokens) observed={}",
            match &refusal {
                Ok(_) => "ACCEPTED".to_string(),
                Err(why) => format!("REFUSE({why})"),
            }
        );
        assert!(
            matches!(&refusal, Err(why) if why.contains("zero tokens")),
            "control failed: a zero-token region was certified"
        );

        let refusal = certify(&[], false);
        eprintln!(
            "CONTROL missing_group_is_refused: expected=REFUSE(no samples) observed={}",
            match &refusal {
                Ok(_) => "ACCEPTED".to_string(),
                Err(why) => format!("REFUSE({why})"),
            }
        );
        assert!(
            matches!(&refusal, Err(why) if why.contains("no samples")),
            "control failed: an empty group was certified"
        );

        let streaming = measure(&spec);
        let refusal = certify(&streaming, true);
        eprintln!(
            "CONTROL missing_interface_hooks_are_refused: expected=REFUSE(hooks) observed={}",
            match &refusal {
                Ok(_) => "ACCEPTED".to_string(),
                Err(why) => format!("REFUSE({why})"),
            }
        );
        assert!(
            matches!(&refusal, Err(why) if why.contains("hooks did not fire")),
            "control failed: a route with no interface hooks was certified as a probed route"
        );
    }

    fn locate_checkpoint() -> Option<std::path::PathBuf> {
        for var in [
            "LATTICE_CPU_GREEDY_MODEL_DIR",
            "LATTICE_MODEL_DIR",
            "LATTICE_INFERENCE_MODEL_DIR",
        ] {
            if let Ok(dir) = std::env::var(var) {
                let dir = std::path::PathBuf::from(dir);
                assert!(
                    dir.is_dir(),
                    "{var} names {dir:?}, which is not a directory; refusing to fall back to a skip"
                );
                return Some(dir);
            }
        }
        let home = std::env::var("HOME").ok()?;
        let dir = std::path::PathBuf::from(home).join(".lattice/models/qwen3.5-0.8b");
        dir.is_dir().then_some(dir)
    }

    pub(super) fn run() {
        eprintln!(
            "\n# {QWEN_CPU_DRIVER_ALLOCATIONS}: counted entry points: alloc, alloc_zeroed (default \
             method, forwards to alloc), realloc; dealloc tallied separately, not gated"
        );
        entry_point_controls();

        let model = tiny_zero_model();
        let synthetic = "synthetic_tiny_zero";
        let greedy = greedy_profile(SYNTHETIC_TOKENS, Vec::new());
        let sampled = sampled_profile(SYNTHETIC_TOKENS, Vec::new());

        let iface = interface_only_evidence(&greedy);

        let greedy_base = route_suite(synthetic, &model, SYNTHETIC_PROMPT, &greedy, &iface);
        route_suite(synthetic, &model, SYNTHETIC_PROMPT, &sampled, &iface);
        offset_control(synthetic, &model, SYNTHETIC_PROMPT, &greedy, &iface);
        refusal_controls(synthetic, &model, SYNTHETIC_PROMPT, &greedy);
        require_groups(&required_groups(synthetic, true, true, true));
        eprintln!(
            "\nR01 SYNTHETIC: INSTRUMENT VALID. warm-region base (greedy) = {} over {} tokens",
            counts(&greedy_base),
            greedy_base.region_tokens
        );

        match locate_checkpoint() {
            None => eprintln!(
                "\nR01 REAL-CHECKPOINT: SKIPPED. No Qwen3.5-0.8B directory found (set \
                 LATTICE_CPU_GREEDY_MODEL_DIR, or place it under ~/.lattice/models/qwen3.5-0.8b). \
                 A skip is not a pass."
            ),
            Some(dir) => {
                let real = Qwen35Model::from_safetensors(&dir)
                    .unwrap_or_else(|e| panic!("loading {dir:?} failed: {e}"));
                let label = "real_qwen3.5-0.8b";
                let stops = vec![151_645];
                let greedy = greedy_profile(REAL_TOKENS, stops.clone());
                let sampled = sampled_profile(REAL_TOKENS, stops);
                let base = route_suite(label, &real, REAL_PROMPT, &greedy, &iface);
                measure_certified(&arm(
                    label,
                    Route::Streaming,
                    Injection::None,
                    Some(&real),
                    REAL_PROMPT,
                    &sampled,
                ));
                require_groups(&required_groups(label, false, false, false));
                eprintln!(
                    "\nR01 REAL-CHECKPOINT: INSTRUMENT VALID on {dir:?}. warm-region base (greedy) = {} over {} tokens",
                    counts(&base),
                    base.region_tokens
                );
            }
        }
    }
}

fn bench_qwen_cpu_driver_allocations(c: &mut Criterion) {
    let _ = c;
    if !qwen_cpu_driver_allocations_selected() {
        return;
    }
    #[cfg(all(feature = "bench-internals", feature = "test-utils"))]
    qwen_cpu_driver_allocations::run();
    #[cfg(not(all(feature = "bench-internals", feature = "test-utils")))]
    panic!(
        "REFUSE: {QWEN_CPU_DRIVER_ALLOCATIONS} was selected but this build lacks the \
         bench-internals and test-utils features; the instrument did not run"
    );
}

// ---------------------------------------------------------------------------
// OPT-LOGIT: final logits projection benchmark (vocab=248320, hidden=2048)
//
// The former generic scalar loop did one dot product per vocabulary row
// against the final hidden state. This benchmark compares:
//   scalar             — direct per-output dot-product loop
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
    let Some(weight) = try_rand_f32_vec(LOGITS_VOCAB * LOGITS_HIDDEN, 0xC0DE_0001) else {
        eprintln!(
            "[bench_logits_projection] Cannot allocate {:.1} GiB weight matrix — skip",
            (LOGITS_VOCAB * LOGITS_HIDDEN * 4) as f64 / (1u64 << 30) as f64
        );
        group.finish();
        return;
    };

    let mut out = vec![0.0f32; LOGITS_VOCAB];

    // scalar: direct per-output dot-product reference
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
// Qwen3.5-target dimensions. The "allocating_current" variant models the
// retired flat-cache decode shape: 11 top-level buffer allocations plus
// 1 logits alloc + 24 layers × 8 heads × 1 query = 192 score allocs
// = 204 allocs/token total (before ForwardScratch).
//
// Run:
//   cargo bench -p lattice-inference --features bench-internals \
//     --bench inference_perf -- generate_forward_with_cache
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const FWD_VOCAB: usize = 248_320;
#[allow(dead_code)]
const FWD_HIDDEN: usize = 2_048;
#[allow(dead_code)]
const FWD_LAYERS: usize = 24;
#[allow(dead_code)]
const FWD_N_HEADS: usize = 8;
#[allow(dead_code)]
const FWD_N_KV_HEADS: usize = 2;
#[allow(dead_code)]
const FWD_HEAD_DIM: usize = 256;
#[allow(dead_code)]
const FWD_INTER: usize = 6_144;
#[allow(dead_code)]
const FWD_WARM_KV: usize = 128;
#[allow(dead_code)]
const FWD_Q_DIM: usize = FWD_N_HEADS * FWD_HEAD_DIM; // 2048
#[allow(dead_code)]
const FWD_KV_DIM: usize = FWD_N_KV_HEADS * FWD_HEAD_DIM; // 512
#[allow(dead_code)]
const FWD_QKV_DIM: usize = FWD_Q_DIM + 2 * FWD_KV_DIM; // 3072
#[allow(dead_code)]
const FWD_RMS_EPS: f32 = 1e-6;
#[allow(dead_code)]
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
// Matches the retired generic decode allocation pattern that ForwardScratch eliminates.
#[cfg(feature = "bench-internals")]
#[allow(clippy::too_many_arguments)]
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
// Models a single-token flat-cache decode using synthetic weights.
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

    let mut hidden = vec![0.0f32; FWD_HIDDEN];
    hidden.copy_from_slice(&embed_tokens[tok * FWD_HIDDEN..(tok + 1) * FWD_HIDDEN]);

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

        cache
            .append_kv(layer, &k_buf, &v_buf)
            .expect("benchmark cache has capacity");

        // Attention over full cached K/V (start_pos prior tokens + 1 current)
        let cached_seq_len = start_pos + 1;
        let k_end = cached_seq_len * FWD_KV_DIM;
        let cache_k: Vec<f32> = cache.k_buffer(layer)[..k_end]
            .iter()
            .map(|value| value.to_f32())
            .collect();
        let cache_v: Vec<f32> = cache.v_buffer(layer)[..k_end]
            .iter()
            .map(|value| value.to_f32())
            .collect();
        compute_attention_alloc(
            &mut attn_out,
            &q_buf,
            &cache_k,
            &cache_v,
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
    cache_k: Vec<f32>,     // [max_kv_seq_len * FWD_KV_DIM]
    cache_v: Vec<f32>,     // [max_kv_seq_len * FWD_KV_DIM]
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
            cache_k: vec![0.0; FWD_KV_DIM * max_kv_seq_len],
            cache_v: vec![0.0; FWD_KV_DIM * max_kv_seq_len],
            logits: vec![0.0; FWD_VOCAB],
        }
    }
}

// Opt 2: score-scratch attention — same math as compute_attention_alloc but
// indexes into a pre-allocated scratch slice instead of allocating per head.
#[cfg(feature = "bench-internals")]
#[allow(clippy::too_many_arguments)]
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

        cache
            .append_kv(layer, &scratch.k_buf, &scratch.v_buf)
            .expect("benchmark cache has capacity");

        let cached_seq_len = start_pos + 1;
        let k_end = cached_seq_len * FWD_KV_DIM;
        let score_len = FWD_N_HEADS * cached_seq_len;
        for (dst, src) in scratch.cache_k[..k_end]
            .iter_mut()
            .zip(&cache.k_buffer(layer)[..k_end])
        {
            *dst = src.to_f32();
        }
        for (dst, src) in scratch.cache_v[..k_end]
            .iter_mut()
            .zip(&cache.v_buffer(layer)[..k_end])
        {
            *dst = src.to_f32();
        }

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
            &scratch.cache_k[..k_end],
            &scratch.cache_v[..k_end],
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
        let Some(embed_tokens) = try_rand_f32_vec(FWD_VOCAB * FWD_HIDDEN, 0xFEED_0001) else {
            eprintln!(
                "[bench_forward_with_cache] Cannot allocate {:.1} GiB embed_tokens — skip",
                (FWD_VOCAB * FWD_HIDDEN * 4) as f64 / (1u64 << 30) as f64
            );
            return;
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
                cache
                    .prefill_layer(layer, &warm_kv_seq, &warm_kv_seq, FWD_WARM_KV)
                    .unwrap();
            }
            cache.advance_by(FWD_WARM_KV).unwrap();
            cache
        };

        // Allocation-count report (before scratch — just prints, does not assert zero)
        {
            const MEASURED: usize = 1;
            allocation_count_print(
                "generate_forward_with_cache",
                "",
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
                &make_warmed_cache,
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
                "",
                "scratch_dispatch",
                MEASURED,
                &mut || {
                    let mut cache = make_warmed_cache();
                    let snap = AllocationSnapshot::capture();
                    for _ in 0..MEASURED {
                        forward_scratch_decode(
                            42u32,
                            FWD_WARM_KV,
                            &mut cache,
                            &embed_tokens,
                            &lw,
                            &norm_weight,
                            &rope,
                            &mut scratch,
                        );
                    }
                    AllocationSnapshot::delta_since(&snap)
                },
            );
        }

        group.bench_function("scratch_dispatch", |b| {
            let mut scratch = ForwardBenchScratch::new(FWD_WARM_KV + 1);
            b.iter_batched(
                &make_warmed_cache,
                |mut cache| {
                    forward_scratch_decode(
                        42u32,
                        FWD_WARM_KV,
                        &mut cache,
                        black_box(&embed_tokens),
                        &lw,
                        &norm_weight,
                        &rope,
                        &mut scratch,
                    );
                },
                BatchSize::LargeInput,
            );
        });

        // --- decode_f32: scratch-dispatch decode step at varying context lengths ---
        // Throughput unit: 1 token per iteration (tok/s via Throughput::Elements(1)).
        // Uses the same forward_scratch_decode (0-alloc warm path) as scratch_dispatch.
        // Context lengths: seq1024, seq4096, seq16384.
        //
        // TODO(#808): decode_q8/seq{N} — same structure with PagedKVCache + CacheType::Q8.
        //   Requires: q8 page storage in paged.rs, gather dequant kernels, and
        //   apply_gqa_attention_paged in gqa.rs wired through forward_scratch_decode.
        let decode_contexts: &[(&str, usize)] =
            &[("seq1024", 1024), ("seq4096", 4096), ("seq16384", 16384)];

        for &(label, warm_kv) in decode_contexts {
            let Some(warm_kv_seq_ctx) =
                try_rand_f32_vec(warm_kv * FWD_KV_DIM, 0xDEAD_0000u32 ^ warm_kv as u32)
            else {
                eprintln!(
                    "[decode_f32/{label}] Cannot allocate {:.1} MiB warm_kv_seq — skip",
                    (warm_kv * FWD_KV_DIM * 4) as f64 / (1u64 << 20) as f64
                );
                continue;
            };

            let cache_cfg_ctx =
                FlatKVCacheConfig::for_qwen3(FWD_LAYERS, FWD_N_KV_HEADS, FWD_HEAD_DIM, warm_kv + 2);
            let rope_ctx = RopeTable::new(FWD_HEAD_DIM, warm_kv + 2, FWD_ROPE_THETA);

            let make_ctx_cache = || {
                let mut cache = FlatKVCache::new(cache_cfg_ctx.clone());
                for layer in 0..FWD_LAYERS {
                    cache
                        .prefill_layer(layer, &warm_kv_seq_ctx, &warm_kv_seq_ctx, warm_kv)
                        .unwrap();
                }
                cache.advance_by(warm_kv).unwrap();
                cache
            };

            group.bench_function(BenchmarkId::new("decode_f32", label), |b| {
                let mut scratch = ForwardBenchScratch::new(warm_kv + 1);
                b.iter_batched(
                    make_ctx_cache,
                    |mut cache| {
                        forward_scratch_decode(
                            42u32,
                            warm_kv,
                            &mut cache,
                            black_box(&embed_tokens),
                            &lw,
                            &norm_weight,
                            &rope_ctx,
                            &mut scratch,
                        );
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
        bench_qwen_cpu_driver_allocations,
        bench_logits_projection,
        bench_forward_with_cache,
        bench_rope_apply_decode,
);
criterion_main!(perf_benches);
