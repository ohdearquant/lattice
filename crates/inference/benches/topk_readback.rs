//! Criterion baselines for logit readback and sampling hot paths (Round 0,
//! pre-optimization).
//!
//! Benchmarks captured here:
//!   1. full_logit_readback   — memcpy of 248_320 f32 values (current Metal path)
//!   2. compact_readback      — memcpy of k*8 bytes for k=1,50,256 (future compact path)
//!   3. sampling_pipeline     — Sampler::sample end-to-end at Qwen3.5 vocab=248_320
//!   4. topk_selection        — isolated select_nth_unstable_by + sort for k=1,50,256
//!
//! Metal-gated (macOS + --features metal-gpu) additionally benchmarks:
//!   5. full_logit_readback_metal  — MTLBuffer StorageModeShared copy of 248_320 f32
//!   6. compact_readback_metal     — MTLBuffer copy of k*8 bytes for k=1,50,256
//!   7. noop_command_buffer        — Metal command-buffer commit+wait overhead floor
//!
//! CPU-only run:
//!   cargo bench -p lattice-inference --bench topk_readback
//!
//! Metal run:
//!   cargo bench -p lattice-inference --features "f16,metal-gpu" --bench topk_readback

use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use lattice_inference::sampling::{Candidate, CandidateSet, Sampler, SamplingConfig};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Qwen3.5-0.6B vocabulary size. The existing inference_perf.rs uses 151_936
/// (Qwen3-Embedding vocab), which underestimates readback cost by ~63%.
const QWEN35_VOCAB_SIZE: usize = 248_320;

/// Smaller vocab used for fast parity checks — tests first-pass + one merge pass.
const PARITY_VOCAB_SIZE: usize = 4096;

/// Inline MSL source for the current (bitonic-sort) top-k kernels used in
/// benchmarks 8 and 11.  Kept here so both bench functions share one copy.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const TOPK_BENCH_MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct TopKCandidate { float logit; uint token_id; };
inline TopKCandidate topk_sentinel() { return TopKCandidate{-INFINITY, UINT_MAX}; }
inline bool topk_better(TopKCandidate a, TopKCandidate b) {
    if (isnan(a.logit)) return false; if (isnan(b.logit)) return true;
    if (a.logit != b.logit) return a.logit > b.logit; return a.token_id < b.token_id;
}
inline void bitonic_sort_desc_2048(threadgroup TopKCandidate* tg, uint tid) {
    for (uint step=2u; step<=2048u; step<<=1u) {
        for (uint dist=step>>1u; dist>0u; dist>>=1u) {
            for (uint idx=tid; idx<2048u; idx+=256u) {
                uint ixj=idx^dist;
                if (ixj>idx) {
                    bool sc=((idx&step)!=0u)?topk_better(tg[idx],tg[ixj]):topk_better(tg[ixj],tg[idx]);
                    if(sc){TopKCandidate tmp=tg[idx];tg[idx]=tg[ixj];tg[ixj]=tmp;}
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
kernel void logits_topk_first_pass(
    device const float* logits [[buffer(0)]],
    device TopKCandidate* partial_out [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    constant uint& top_k [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup TopKCandidate tg[2048];
    uint base=tgid*1280u;
    for(uint j=0u;j<5u;++j){uint slot=tid+j*256u;uint gidx=base+slot;
        if(gidx<vocab_size){float v=logits[gidx];tg[slot]=isnan(v)?topk_sentinel():TopKCandidate{v,gidx};}
        else tg[slot]=topk_sentinel();}
    for(uint slot=1280u+tid;slot<2048u;slot+=256u)tg[slot]=topk_sentinel();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bitonic_sort_desc_2048(tg,tid);
    for(uint out=tid;out<top_k;out+=256u)partial_out[tgid*top_k+out]=tg[out];
}
kernel void logits_topk_merge_pass(
    device const TopKCandidate* in_buf [[buffer(0)]],
    device TopKCandidate* out_buf [[buffer(1)]],
    constant uint& input_groups [[buffer(2)]],
    constant uint& top_k [[buffer(3)]],
    constant uint& fan_in [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup TopKCandidate tg[2048];
    uint start_group=tgid*fan_in;
    uint max_items=min(fan_in*top_k,2048u);
    for(uint slot=tid;slot<2048u;slot+=256u){
        if(slot<max_items){uint g=slot/top_k;uint ci=slot%top_k;uint gg=start_group+g;
            tg[slot]=(gg<input_groups)?in_buf[gg*top_k+ci]:topk_sentinel();}
        else tg[slot]=topk_sentinel();}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bitonic_sort_desc_2048(tg,tid);
    for(uint out=tid;out<top_k;out+=256u)out_buf[tgid*top_k+out]=tg[out];
}
inline TopKCandidate simd_argmax(TopKCandidate v, uint simd_width) {
    for (uint off=simd_width>>1u; off>0u; off>>=1u) {
        TopKCandidate other;
        other.logit=simd_shuffle_down(v.logit,off);
        other.token_id=simd_shuffle_down(v.token_id,off);
        if (topk_better(other,v)) v=other;
    }
    return v;
}
kernel void logits_argmax_first(
    device const float* logits [[buffer(0)]],
    device TopKCandidate* group_out [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    threadgroup TopKCandidate sg_winners[32];
    TopKCandidate best=topk_sentinel();
    uint idx=tgid*1024u+tid;
    if (idx<vocab_size){float v=logits[idx];best=isnan(v)?topk_sentinel():TopKCandidate{v,idx};}
    best=simd_argmax(best,simd_width);
    if (lane==0u) sg_winners[sgid]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TopKCandidate tg_best=(tid<32u)?sg_winners[tid]:topk_sentinel();
    if (sgid==0u){tg_best=simd_argmax(tg_best,simd_width);if(lane==0u)group_out[tgid]=tg_best;}
}
kernel void logits_argmax_merge(
    device const TopKCandidate* group_in [[buffer(0)]],
    device TopKCandidate* out [[buffer(1)]],
    constant uint& group_count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    threadgroup TopKCandidate sg_winners[32];
    TopKCandidate best=topk_sentinel();
    for (uint i=tid;i<group_count;i+=1024u){TopKCandidate c=group_in[i];if(topk_better(c,best))best=c;}
    best=simd_argmax(best,simd_width);
    if (lane==0u) sg_winners[sgid]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TopKCandidate final_best=(tid<32u)?sg_winners[tid]:topk_sentinel();
    if (sgid==0u){final_best=simd_argmax(final_best,simd_width);if(lane==0u)out[0]=final_best;}
}
inline void bitonic_sort_desc_1024(threadgroup TopKCandidate* tg, uint tid) {
    for (uint step=2u;step<=1024u;step<<=1u) {
        for (uint dist=step>>1u;dist>0u;dist>>=1u) {
            for (uint idx=tid;idx<1024u;idx+=256u) {
                uint ixj=idx^dist;
                if (ixj>idx){
                    bool sc=((idx&step)!=0u)?topk_better(tg[idx],tg[ixj]):topk_better(tg[ixj],tg[idx]);
                    if(sc){TopKCandidate tmp=tg[idx];tg[idx]=tg[ixj];tg[ixj]=tmp;}
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
kernel void logits_topk_fast_first(
    device const float* logits [[buffer(0)]],
    device TopKCandidate* partial_out [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    constant uint& top_k [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup TopKCandidate tg[1024];
    uint base=tgid*1024u;
    for(uint j=0u;j<4u;++j){uint slot=tid+j*256u;uint gidx=base+slot;
        if(gidx<vocab_size){float v=logits[gidx];tg[slot]=isnan(v)?topk_sentinel():TopKCandidate{v,gidx};}
        else tg[slot]=topk_sentinel();}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bitonic_sort_desc_1024(tg,tid);
    for(uint out=tid;out<top_k;out+=256u)partial_out[tgid*top_k+out]=tg[out];
}

// ---------------------------------------------------------------------------
// select64 kernels — non-sort repeated-selection (negative-bitonic baseline)
// ---------------------------------------------------------------------------
inline TopKCandidate sg_reduce_best(TopKCandidate v, uint simd_width) {
    for (uint off=simd_width>>1u; off>0u; off>>=1u) {
        TopKCandidate other;
        other.logit=simd_shuffle_down(v.logit,off);
        other.token_id=simd_shuffle_down(v.token_id,off);
        if (topk_better(other,v)) v=other;
    }
    return v;
}
kernel void logits_topk_select64_first(
    device const float*   logits      [[buffer(0)]],
    device TopKCandidate* partial_out [[buffer(1)]],
    constant uint&        vocab_size  [[buffer(2)]],
    constant uint&        top_k       [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lane       [[thread_index_in_simdgroup]],
    uint sgid       [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    const uint TILE=8192u; const uint SG_TILE=1024u;
    uint base=tgid*TILE+sgid*SG_TILE;
    thread TopKCandidate local[32];
    uint selected_mask=0u;
    for (uint j=0u;j<32u;++j){
        uint idx=base+lane+j*simd_width;
        if (idx<vocab_size){float v=logits[idx];local[j]=isnan(v)?topk_sentinel():TopKCandidate{v,idx};}
        else local[j]=topk_sentinel();
    }
    for (uint rank=0u;rank<64u;++rank){
        TopKCandidate lane_best=topk_sentinel(); uint lane_best_j=UINT_MAX;
        for (uint j=0u;j<32u;++j){
            if (((selected_mask>>j)&1u)==0u&&topk_better(local[j],lane_best)){lane_best=local[j];lane_best_j=j;}
        }
        TopKCandidate winner=sg_reduce_best(lane_best,simd_width);
        winner.logit=simd_broadcast(winner.logit,0u);
        winner.token_id=simd_broadcast(winner.token_id,0u);
        if (lane==0u) partial_out[(tgid*8u+sgid)*64u+rank]=winner;
        if (lane_best_j!=UINT_MAX&&local[lane_best_j].token_id==winner.token_id) selected_mask|=(1u<<lane_best_j);
    }
}
kernel void logits_topk_select64_merge(
    device const TopKCandidate* in_buf       [[buffer(0)]],
    device       TopKCandidate* out_buf      [[buffer(1)]],
    constant uint&              input_groups [[buffer(2)]],
    constant uint&              top_k        [[buffer(3)]],
    constant uint&              fan_in       [[buffer(4)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lane       [[thread_index_in_simdgroup]],
    uint sgid       [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    threadgroup TopKCandidate tg[2048];
    threadgroup uchar selected[2048];
    threadgroup TopKCandidate sg_winners[8];
    threadgroup TopKCandidate rank_winner;
    uint start_group=tgid*fan_in;
    uint max_items=min(fan_in*64u,2048u);
    for (uint slot=tid;slot<2048u;slot+=256u){
        if (slot<max_items){uint g=slot/64u;uint ci=slot%64u;uint gg=start_group+g;
            tg[slot]=(gg<input_groups)?in_buf[gg*64u+ci]:topk_sentinel();}
        else tg[slot]=topk_sentinel();
        selected[slot]=0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rank=0u;rank<64u;++rank){
        TopKCandidate local_best=topk_sentinel();
        for (uint slot=tid;slot<max_items;slot+=256u){
            if (selected[slot]==0u&&topk_better(tg[slot],local_best)) local_best=tg[slot];
        }
        TopKCandidate sg_best=sg_reduce_best(local_best,simd_width);
        if (lane==0u) sg_winners[sgid]=sg_best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        TopKCandidate final_best=(tid<8u)?sg_winners[tid]:topk_sentinel();
        if (sgid==0u){final_best=sg_reduce_best(final_best,simd_width);
            if (lane==0u){rank_winner=final_best;out_buf[tgid*64u+rank]=final_best;}}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint slot=tid;slot<max_items;slot+=256u){
            if (tg[slot].token_id==rank_winner.token_id) selected[slot]=1u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// Hierarchical k=50 SIMD-group tournament kernels (R2 benchmark baseline)
// Same logic as logits_topk_select50_first / logits_topk_select50_merge
// in metal_qwen35.rs — kept here so benchmarks can self-compile.
// ---------------------------------------------------------------------------
constant uint TOPK50 = 50u;
constant uint SELECT50_THREADS = 256u;
constant uint SELECT50_ITEMS_PER_THREAD = 4u;
constant uint SELECT50_TILE = 1024u;
constant uint SELECT50_MAX_SIMDGROUPS = 16u;
constant uint SELECT50_MERGE_LOCAL_MAX = 50u;

kernel void logits_topk_select50_first(
    device const float*   logits      [[buffer(0)]],
    device TopKCandidate* partial_out [[buffer(1)]],
    constant uint&        vocab_size  [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lane       [[thread_index_in_simdgroup]],
    uint sgid       [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    threadgroup TopKCandidate sg_top[SELECT50_MAX_SIMDGROUPS * TOPK50];
    if (simd_width < 16u) return;
    uint num_sg  = SELECT50_THREADS / simd_width;
    uint sg_tile = simd_width * SELECT50_ITEMS_PER_THREAD;
    uint base    = tgid * SELECT50_TILE + sgid * sg_tile;
    thread TopKCandidate local[SELECT50_ITEMS_PER_THREAD];
    uint selected_mask = 0u;
    for (uint j=0u;j<SELECT50_ITEMS_PER_THREAD;++j){
        uint idx=base+lane+j*simd_width;
        if (idx<vocab_size){float v=logits[idx];local[j]=isnan(v)?topk_sentinel():TopKCandidate{v,idx};}
        else local[j]=topk_sentinel();
    }
    for (uint rank=0u;rank<TOPK50;++rank){
        TopKCandidate lane_best=topk_sentinel(); uint lane_best_j=UINT_MAX;
        for (uint j=0u;j<SELECT50_ITEMS_PER_THREAD;++j){
            if(((selected_mask>>j)&1u)==0u&&topk_better(local[j],lane_best)){lane_best=local[j];lane_best_j=j;}
        }
        TopKCandidate winner=simd_argmax(lane_best,simd_width);
        winner.logit=simd_broadcast(winner.logit,0u);
        winner.token_id=simd_broadcast(winner.token_id,0u);
        if (lane==0u) sg_top[sgid*TOPK50+rank]=winner;
        if (lane_best_j!=UINT_MAX&&local[lane_best_j].token_id==winner.token_id) selected_mask|=(1u<<lane_best_j);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid==0u){
        thread TopKCandidate merge_vals[SELECT50_MERGE_LOCAL_MAX];
        ulong merge_selected=0ul;
        uint candidate_count=num_sg*TOPK50;
        for (uint j=0u;j<SELECT50_MERGE_LOCAL_MAX;++j){
            uint slot=lane+j*simd_width;
            merge_vals[j]=(slot<candidate_count)?sg_top[slot]:topk_sentinel();
        }
        for (uint rank=0u;rank<TOPK50;++rank){
            TopKCandidate lane_best=topk_sentinel(); uint lane_best_j=UINT_MAX;
            for (uint j=0u;j<SELECT50_MERGE_LOCAL_MAX;++j){
                if(((merge_selected>>j)&1ul)==0ul&&topk_better(merge_vals[j],lane_best)){lane_best=merge_vals[j];lane_best_j=j;}
            }
            TopKCandidate winner=simd_argmax(lane_best,simd_width);
            winner.logit=simd_broadcast(winner.logit,0u);
            winner.token_id=simd_broadcast(winner.token_id,0u);
            if (lane==0u) partial_out[tgid*TOPK50+rank]=winner;
            if (lane_best_j!=UINT_MAX&&merge_vals[lane_best_j].token_id==winner.token_id) merge_selected|=(1ul<<lane_best_j);
        }
    }
}

kernel void logits_topk_select50_merge(
    device const TopKCandidate* in_buf       [[buffer(0)]],
    device       TopKCandidate* out_buf      [[buffer(1)]],
    constant uint&              input_groups [[buffer(2)]],
    constant uint&              fan_in       [[buffer(3)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lane       [[thread_index_in_simdgroup]],
    uint sgid       [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    if (simd_width<16u) return;
    if (sgid!=0u) return;
    thread TopKCandidate vals[SELECT50_MERGE_LOCAL_MAX];
    ulong selected=0ul;
    uint start_group=tgid*fan_in;
    uint remaining_groups=(start_group<input_groups)?(input_groups-start_group):0u;
    uint groups_here=min(fan_in,remaining_groups);
    uint candidate_count=groups_here*TOPK50;
    for (uint j=0u;j<SELECT50_MERGE_LOCAL_MAX;++j){
        uint slot=lane+j*simd_width;
        if (slot<candidate_count){uint g=slot/TOPK50;uint ci=slot-g*TOPK50;vals[j]=in_buf[(start_group+g)*TOPK50+ci];}
        else vals[j]=topk_sentinel();
    }
    for (uint rank=0u;rank<TOPK50;++rank){
        TopKCandidate lane_best=topk_sentinel(); uint lane_best_j=UINT_MAX;
        for (uint j=0u;j<SELECT50_MERGE_LOCAL_MAX;++j){
            if(((selected>>j)&1ul)==0ul&&topk_better(vals[j],lane_best)){lane_best=vals[j];lane_best_j=j;}
        }
        TopKCandidate winner=simd_argmax(lane_best,simd_width);
        winner.logit=simd_broadcast(winner.logit,0u);
        winner.token_id=simd_broadcast(winner.token_id,0u);
        if (lane==0u) out_buf[tgid*TOPK50+rank]=winner;
        if (lane_best_j!=UINT_MAX&&vals[lane_best_j].token_id==winner.token_id) selected|=(1ul<<lane_best_j);
    }
}
"#;

/// Full readback byte count: 248_320 * 4 = 993_280 bytes (~970 KiB per token).
const FULL_READBACK_BYTES: u64 = (QWEN35_VOCAB_SIZE * 4) as u64;

// ---------------------------------------------------------------------------
// Shared PRNG (xorshift32, same as inference_perf.rs)
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
        // Small activations typical of model logits (-2 to +2).
        out.push((bits as f32 / u32::MAX as f32) * 4.0 - 2.0);
    }
    out
}

// ---------------------------------------------------------------------------
// 1. full_logit_readback — CPU memcpy simulating current Metal readback
//
// After wait_until_completed(), the Metal decode loop calls:
//     read_buffer(buf, cfg.vocab_size)
// which is literally:
//     let ptr = buf.contents() as *const f32;
//     let mut out = vec![0.0f32; len];
//     std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
//
// On Apple Silicon unified memory, buf.contents() is a CPU-accessible pointer
// into the shared physical memory that the GPU wrote to. The copy is a raw
// memcpy of FULL_READBACK_BYTES per generated token.
//
// This benchmark measures ONLY the memory-copy overhead — no GPU computation,
// no command-buffer wait. The Metal-gated variant (bench 5) measures the same
// copy from an actual MTLBuffer to confirm the overhead is identical.
// ---------------------------------------------------------------------------

fn bench_full_logit_readback(c: &mut Criterion) {
    let src = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xCAFE_BABE);

    let mut group = c.benchmark_group("full_logit_readback");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes(FULL_READBACK_BYTES));

    group.bench_function("cpu_memcpy_248320_f32", |b| {
        b.iter(|| {
            let mut dst = vec![0.0f32; QWEN35_VOCAB_SIZE];
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), QWEN35_VOCAB_SIZE);
            }
            black_box(dst)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. compact_readback — CPU memcpy for future compact path (k*8 bytes)
//
// If GPU-side top-k is added, readback shrinks from 993,280 B to k*8 bytes:
//   k=1:   8 B  (124,160× smaller)
//   k=50:  400 B  (2,483× smaller)
//   k=256: 2,048 B  (485× smaller)
//
// This bench captures the memcpy floor for the compact path so i2 can report
// actual before/after numbers for the readback component alone.
// ---------------------------------------------------------------------------

fn bench_compact_readback(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_readback");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &k in &[1usize, 10, 50, 256] {
        let byte_count = (k * 8) as u64;
        // Source represents a compact GPU candidate buffer: (f32 logit, u32 token_id) pairs.
        let src: Vec<u8> = (0..k * 8).map(|i| (i ^ 0xAB) as u8).collect();

        group.throughput(Throughput::Bytes(byte_count));
        group.bench_function(
            BenchmarkId::new("cpu_memcpy_candidates", format!("k{k}")),
            |b| {
                b.iter(|| {
                    let mut dst = vec![0u8; k * 8];
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), k * 8);
                    }
                    black_box(dst)
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. sampling_pipeline — Sampler::sample end-to-end at Qwen3.5 vocab
//
// Current per-token hot path (all allocations included):
//   a. logits.to_vec()                  — 993,280 B clone
//   b. Apply repetition penalty         — O(recent_tokens) writes
//   c. Collect indexed pairs            — 1,986,560 B alloc + O(n) map
//   d. select_nth_unstable_by(k-1)      — O(n) partial select
//   e. truncate(k) + sort k candidates  — O(k log k)
//   f. Softmax + top-p filtering        — O(k)
//   g. Weighted random sample           — O(k)
//
// k=1 (greedy) skips steps c-g and runs argmax only. Included as floor.
// ---------------------------------------------------------------------------

fn bench_sampling_pipeline(c: &mut Criterion) {
    let logits = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xA1B2_C3D4);

    let mut group = c.benchmark_group("sampling_pipeline");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(QWEN35_VOCAB_SIZE as u64));

    // Greedy (k=1): argmax only — no select_nth, no probability allocation.
    group.bench_function("greedy_k1", |b| {
        b.iter_batched(
            || Sampler::new(SamplingConfig::greedy()),
            |mut sampler| black_box(sampler.sample(black_box(&logits))),
            BatchSize::SmallInput,
        );
    });

    // Default (k=50, top_p=0.9): the production sampling configuration.
    group.bench_function("default_k50_topp0.9", |b| {
        b.iter_batched(
            || Sampler::new(SamplingConfig::default()).with_seed(0xDEAD_BEEF),
            |mut sampler| black_box(sampler.sample(black_box(&logits))),
            BatchSize::SmallInput,
        );
    });

    // Wide (k=256, top_p=0.9): larger candidate set for nucleus sampling.
    group.bench_function("wide_k256_topp0.9", |b| {
        b.iter_batched(
            || {
                Sampler::new(SamplingConfig {
                    temperature: 0.7,
                    top_k: 256,
                    top_p: 0.9,
                    repetition_penalty: 1.1,
                })
                .with_seed(0xDEAD_BEEF)
            },
            |mut sampler| black_box(sampler.sample(black_box(&logits))),
            BatchSize::SmallInput,
        );
    });

    // New: repeated calls on a warm Sampler — measures scratch reuse benefit.
    // Unlike default_k50_topp0.9 (fresh Sampler per batch), this keeps the same
    // Sampler alive so candidate_scratch and prob_scratch stay allocated across
    // all iterations. Comparing with default_k50_topp0.9 quantifies the cost of
    // the initial Vec<Candidate> capacity growth in the first few calls.
    group.bench_function("repeated_scratch_reuse/k50_topp0.9", |b| {
        let mut sampler = Sampler::new(SamplingConfig::default()).with_seed(0xDEAD_BEEF);
        // One warm-up call to grow candidate_scratch to full-vocab capacity.
        let _ = sampler.sample(&logits);
        b.iter(|| black_box(sampler.sample(black_box(&logits))));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. topk_selection — isolated CPU top-k selection (current method)
//
// Reproduces the exact selection code path inside Sampler::sample, with all
// per-call allocations included (matching production cost):
//   - indexed pair Vec:       248_320 * 8 B = ~1.9 MB per token
//   - select_nth_unstable_by: O(n) partial select on 248_320 elements
//   - sort of k retained:     O(k log k)
//
// For k=1, Sampler::sample uses fast argmax — benchmarked separately here.
// ---------------------------------------------------------------------------

fn bench_topk_selection(c: &mut Criterion) {
    let logits = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xFEED_FACE);

    let mut group = c.benchmark_group("topk_selection");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(QWEN35_VOCAB_SIZE as u64));

    // k=1: argmax scan — the greedy path in Sampler::sample.
    group.bench_function("argmax_k1", |b| {
        b.iter(|| {
            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in black_box(&logits).iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            black_box(best_idx)
        });
    });

    // k=10, k=50 and k=256: collect indexed pairs + select_nth + sort.
    // Allocations are included to match the production cost.
    for &k in &[10usize, 50, 256] {
        group.bench_function(
            BenchmarkId::new("collect_select_sort", format!("k{k}")),
            |b| {
                b.iter(|| {
                    // Same as Sampler::sample steps c-e.
                    let mut indexed: Vec<(u32, f32)> = black_box(&logits)
                        .iter()
                        .enumerate()
                        .map(|(i, &l)| (i as u32, l))
                        .collect();
                    indexed.select_nth_unstable_by(k - 1, |a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    indexed.truncate(k);
                    indexed
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    black_box(indexed)
                });
            },
        );
    }

    // New: retain_top_k without the full-vocab Vec<Candidate> build in timed path.
    // Setup (not timed): CandidateSet::from_full_logits builds the full Vec<Candidate>.
    // Body (timed): CandidateSet::retain_top_k — select_nth_unstable_by + truncate only.
    // Comparing with collect_select_sort/k50 isolates the alloc+collect cost from
    // the selection cost.
    for &k in &[10usize, 50, 256] {
        group.bench_function(
            BenchmarkId::new("retain_top_k_no_alloc", format!("k{k}")),
            |b| {
                b.iter_batched(
                    || CandidateSet::from_full_logits(&logits),
                    |mut cs| {
                        cs.retain_top_k(k);
                        black_box(cs)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    // New: comparator cost — partial_cmp (current) vs total_cmp (H3 candidate).
    // Setup (not timed): build full-vocab Vec<Candidate>.
    // Body (timed): select_nth_unstable_by + truncate with different comparators.
    // Both produce descending order for finite logits; difference is NaN handling
    // and the Option unwrap in partial_cmp. Our benchmark data has no NaNs so
    // this isolates pure comparator branch cost.
    {
        let k = 50usize;
        group.bench_function(
            BenchmarkId::new("select_nth_partial_cmp", format!("k{k}")),
            |b| {
                b.iter_batched(
                    || {
                        logits
                            .iter()
                            .enumerate()
                            .map(|(i, &l)| Candidate {
                                token_id: i as u32,
                                logit: l,
                            })
                            .collect::<Vec<Candidate>>()
                    },
                    |mut candidates| {
                        candidates.select_nth_unstable_by(k - 1, |a, b| {
                            b.logit
                                .partial_cmp(&a.logit)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        candidates.truncate(k);
                        black_box(candidates)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        group.bench_function(
            BenchmarkId::new("select_nth_total_cmp", format!("k{k}")),
            |b| {
                b.iter_batched(
                    || {
                        logits
                            .iter()
                            .enumerate()
                            .map(|(i, &l)| Candidate {
                                token_id: i as u32,
                                logit: l,
                            })
                            .collect::<Vec<Candidate>>()
                    },
                    |mut candidates| {
                        // total_cmp descending: b before a when b.logit > a.logit.
                        // NaN sorts after all finite values in total_cmp (IEEE 754 total order),
                        // which differs from partial_cmp unwrap_or(Equal).
                        candidates
                            .select_nth_unstable_by(k - 1, |a, b| b.logit.total_cmp(&a.logit));
                        candidates.truncate(k);
                        black_box(candidates)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. full_logit_readback_metal — actual MTLBuffer copy (Metal-gated)
//
// Allocates a StorageModeShared MTLBuffer containing 248_320 synthetic f32
// logits and benchmarks the same ptr::copy_nonoverlapping used by read_buffer()
// in metal_qwen35.rs. Confirms that the CPU memcpy benchmark above is a valid
// proxy for unified-memory readback on the current device.
// ---------------------------------------------------------------------------

fn bench_full_logit_readback_metal(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use metal::{Device, MTLResourceOptions};

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("[topk_readback] No Metal device — skipping Metal readback bench");
                return;
            }
        };

        let src_data = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xCAFE_BABE);
        let buf = device.new_buffer(FULL_READBACK_BYTES, MTLResourceOptions::StorageModeShared);

        // Populate with the same deterministic logits as the CPU benchmark.
        unsafe {
            let ptr = buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(src_data.as_ptr(), ptr, QWEN35_VOCAB_SIZE);
        }

        let mut group = c.benchmark_group("full_logit_readback_metal");
        group.sample_size(30);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
        group.throughput(Throughput::Bytes(FULL_READBACK_BYTES));

        // Same pattern as read_buffer() in metal_qwen35.rs.
        group.bench_function("metal_shared_buf_248320_f32", |b| {
            b.iter(|| {
                let mut dst = vec![0.0f32; QWEN35_VOCAB_SIZE];
                unsafe {
                    let ptr = buf.contents() as *const f32;
                    std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), QWEN35_VOCAB_SIZE);
                }
                black_box(dst)
            });
        });

        group.finish();
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// 6. compact_readback_metal — actual MTLBuffer compact copy (Metal-gated)
//
// Mirrors bench_compact_readback but uses actual MTLBuffer allocations to
// confirm the readback ceiling for future compact top-k candidates.
// ---------------------------------------------------------------------------

fn bench_compact_readback_metal(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use metal::{Device, MTLResourceOptions};

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!(
                    "[topk_readback] No Metal device — skipping compact Metal readback bench"
                );
                return;
            }
        };

        let mut group = c.benchmark_group("compact_readback_metal");
        group.sample_size(30);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));

        for &k in &[1usize, 10, 50, 256] {
            let byte_count = (k * 8) as u64;
            let buf = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);

            // Fill with sentinel data representing (f32 logit, u32 token_id) pairs.
            unsafe {
                let ptr = buf.contents() as *mut u8;
                for i in 0..(k * 8) {
                    *ptr.add(i) = (i ^ 0xAB) as u8;
                }
            }

            group.throughput(Throughput::Bytes(byte_count));
            group.bench_function(
                BenchmarkId::new("metal_shared_buf_candidates", format!("k{k}")),
                |b| {
                    b.iter(|| {
                        let mut dst = vec![0u8; k * 8];
                        unsafe {
                            let ptr = buf.contents() as *const u8;
                            std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), k * 8);
                        }
                        black_box(dst)
                    });
                },
            );
        }

        group.finish();
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// 7. noop_command_buffer — Metal command-buffer overhead floor (Metal-gated)
//
// Measures the fixed latency of commit() + wait_until_completed() on an empty
// command buffer. This is the Metal synchronization floor: any top-k kernel
// dispatch overhead is measured relative to this baseline.
//
// Based on the pattern from foundation/inference/examples/bench_dispatch2.rs:153.
// ---------------------------------------------------------------------------

fn bench_noop_command_buffer(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use metal::Device;

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("[topk_readback] No Metal device — skipping noop command buffer bench");
                return;
            }
        };

        let queue = device.new_command_queue();

        let mut group = c.benchmark_group("noop_command_buffer");
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));

        // Baseline: commit + wait for an empty command buffer.
        // Any top-k kernel bench should subtract this floor.
        group.bench_function("commit_and_wait", |b| {
            b.iter(|| {
                let cmd = queue.new_command_buffer();
                cmd.commit();
                cmd.wait_until_completed();
                black_box(())
            });
        });

        group.finish();
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// 8. metal_topk_dispatch_only — GPU top-k kernel dispatch + wait (Metal-gated)
//
// Measures: dispatch first_pass + merge passes + wait_until_completed for
// k=1, 50, 256 over a 248,320-element logits buffer.
// Does NOT include the compact readback (see bench 9 for that).
// ---------------------------------------------------------------------------

fn bench_metal_topk_dispatch_only(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use metal::{CompileOptions, Device, MTLResourceOptions};

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("[topk_readback] No Metal device — skipping top-k dispatch bench");
                return;
            }
        };

        let opts = CompileOptions::new();
        let lib = match device.new_library_with_source(TOPK_BENCH_MSL_SRC, &opts) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[topk_readback] MSL compile failed: {e}");
                return;
            }
        };
        let make_pipe = |name: &str| {
            lib.get_function(name, None)
                .and_then(|f| device.new_compute_pipeline_state_with_function(&f))
                .ok()
        };
        let argmax_first_pipe = match make_pipe("logits_argmax_first") {
            Some(p) => p,
            None => {
                eprintln!("[topk_readback] argmax_first pipeline failed");
                return;
            }
        };
        let argmax_merge_pipe = match make_pipe("logits_argmax_merge") {
            Some(p) => p,
            None => {
                eprintln!("[topk_readback] argmax_merge pipeline failed");
                return;
            }
        };
        let fast_first_pipe = match make_pipe("logits_topk_fast_first") {
            Some(p) => p,
            None => {
                eprintln!("[topk_readback] topk_fast_first pipeline failed");
                return;
            }
        };
        let merge_pass_pipe = match make_pipe("logits_topk_merge_pass") {
            Some(p) => p,
            None => {
                eprintln!("[topk_readback] topk_merge_pass pipeline failed");
                return;
            }
        };
        let sel64_first_pipe = match make_pipe("logits_topk_select64_first") {
            Some(p) => p,
            None => {
                eprintln!("[topk_readback] select64_first pipeline failed");
                return;
            }
        };
        let sel64_merge_pipe = match make_pipe("logits_topk_select64_merge") {
            Some(p) => p,
            None => {
                eprintln!("[topk_readback] select64_merge pipeline failed");
                return;
            }
        };
        let sel50_first_pipe = match make_pipe("logits_topk_select50_first") {
            Some(p) => p,
            None => {
                eprintln!("[topk_readback] select50_first pipeline failed");
                return;
            }
        };
        let sel50_merge_pipe = match make_pipe("logits_topk_select50_merge") {
            Some(p) => p,
            None => {
                eprintln!("[topk_readback] select50_merge pipeline failed");
                return;
            }
        };

        let queue = device.new_command_queue();

        let src_data = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xCAFE_BABE);
        let logits_buf =
            device.new_buffer(FULL_READBACK_BYTES, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = logits_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(src_data.as_ptr(), ptr, QWEN35_VOCAB_SIZE);
        }

        // Scratch: 243 groups × 256 × 8 = 499,584 B — covers bitonic, select64, and hierarchical k=50.
        let scratch_bytes = 243usize * 256 * 8;
        let scratch_a =
            device.new_buffer(scratch_bytes as u64, MTLResourceOptions::StorageModeShared);
        let scratch_b =
            device.new_buffer(scratch_bytes as u64, MTLResourceOptions::StorageModeShared);

        let mut group = c.benchmark_group("metal_topk_dispatch_only");
        group.sample_size(100);
        group.warm_up_time(Duration::from_secs(2));
        group.measurement_time(Duration::from_secs(10));

        // Bitonic negative baseline: k=1 argmax, k=10/50/256 bitonic sort (REJECTED path).
        // These are NOT production recommendations — shown for comparison only.
        for &k in &[1u32, 10, 50, 256] {
            group.bench_function(
                BenchmarkId::new("bitonic_negative_baseline", format!("k{k}")),
                |b| {
                    // 10 untimed warm-up dispatches before Criterion timing.
                    for _ in 0..10 {
                        let cmd = queue.new_command_buffer();
                        let enc = cmd.new_compute_command_encoder();
                        let vocab_u32 = QWEN35_VOCAB_SIZE as u32;
                        if k == 1 {
                            let groups = vocab_u32.div_ceil(1024);
                            enc.set_compute_pipeline_state(&argmax_first_pipe);
                            enc.set_buffer(0, Some(&logits_buf), 0);
                            enc.set_buffer(1, Some(&scratch_a), 0);
                            enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                metal::MTLSize::new(groups as u64, 1, 1),
                                metal::MTLSize::new(1024, 1, 1),
                            );
                            enc.set_compute_pipeline_state(&argmax_merge_pipe);
                            enc.set_buffer(0, Some(&scratch_a), 0);
                            enc.set_buffer(1, Some(&scratch_b), 0);
                            enc.set_bytes(2, 4, &groups as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                metal::MTLSize::new(1, 1, 1),
                                metal::MTLSize::new(1024, 1, 1),
                            );
                        } else {
                            let tile = 1024u32;
                            let first_groups = vocab_u32.div_ceil(tile);
                            enc.set_compute_pipeline_state(&fast_first_pipe);
                            enc.set_buffer(0, Some(&logits_buf), 0);
                            enc.set_buffer(1, Some(&scratch_a), 0);
                            enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                            enc.set_bytes(3, 4, &k as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                metal::MTLSize::new(first_groups as u64, 1, 1),
                                metal::MTLSize::new(256, 1, 1),
                            );
                            let mut cg = first_groups;
                            let mut w = 0u8;
                            while cg > 1 {
                                let fan_in = (2048u32 / k).max(2).min(cg);
                                let og = cg.div_ceil(fan_in);
                                let (ib, ob) = if w == 0 {
                                    (&scratch_a, &scratch_b)
                                } else {
                                    (&scratch_b, &scratch_a)
                                };
                                enc.set_compute_pipeline_state(&merge_pass_pipe);
                                enc.set_buffer(0, Some(ib), 0);
                                enc.set_buffer(1, Some(ob), 0);
                                enc.set_bytes(2, 4, &cg as *const u32 as *const _);
                                enc.set_bytes(3, 4, &k as *const u32 as *const _);
                                enc.set_bytes(4, 4, &fan_in as *const u32 as *const _);
                                enc.dispatch_thread_groups(
                                    metal::MTLSize::new(og as u64, 1, 1),
                                    metal::MTLSize::new(256, 1, 1),
                                );
                                cg = og;
                                w = 1 - w;
                            }
                        }
                        enc.end_encoding();
                        cmd.commit();
                        cmd.wait_until_completed();
                    }
                    b.iter(|| {
                        let cmd = queue.new_command_buffer();
                        let enc = cmd.new_compute_command_encoder();
                        let vocab_u32 = QWEN35_VOCAB_SIZE as u32;
                        let which;
                        if k == 1 {
                            let groups = vocab_u32.div_ceil(1024);
                            enc.set_compute_pipeline_state(&argmax_first_pipe);
                            enc.set_buffer(0, Some(&logits_buf), 0);
                            enc.set_buffer(1, Some(&scratch_a), 0);
                            enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                metal::MTLSize::new(groups as u64, 1, 1),
                                metal::MTLSize::new(1024, 1, 1),
                            );
                            enc.set_compute_pipeline_state(&argmax_merge_pipe);
                            enc.set_buffer(0, Some(&scratch_a), 0);
                            enc.set_buffer(1, Some(&scratch_b), 0);
                            enc.set_bytes(2, 4, &groups as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                metal::MTLSize::new(1, 1, 1),
                                metal::MTLSize::new(1024, 1, 1),
                            );
                            which = 1u8;
                        } else {
                            let tile = 1024u32;
                            let first_groups = vocab_u32.div_ceil(tile);
                            enc.set_compute_pipeline_state(&fast_first_pipe);
                            enc.set_buffer(0, Some(&logits_buf), 0);
                            enc.set_buffer(1, Some(&scratch_a), 0);
                            enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                            enc.set_bytes(3, 4, &k as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                metal::MTLSize::new(first_groups as u64, 1, 1),
                                metal::MTLSize::new(256, 1, 1),
                            );
                            let mut current_groups = first_groups;
                            let mut w = 0u8;
                            while current_groups > 1 {
                                let fan_in = (2048u32 / k).max(2).min(current_groups);
                                let out_groups = current_groups.div_ceil(fan_in);
                                let (in_b, out_b) = if w == 0 {
                                    (&scratch_a, &scratch_b)
                                } else {
                                    (&scratch_b, &scratch_a)
                                };
                                enc.set_compute_pipeline_state(&merge_pass_pipe);
                                enc.set_buffer(0, Some(in_b), 0);
                                enc.set_buffer(1, Some(out_b), 0);
                                enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
                                enc.set_bytes(3, 4, &k as *const u32 as *const _);
                                enc.set_bytes(4, 4, &fan_in as *const u32 as *const _);
                                enc.dispatch_thread_groups(
                                    metal::MTLSize::new(out_groups as u64, 1, 1),
                                    metal::MTLSize::new(256, 1, 1),
                                );
                                current_groups = out_groups;
                                w = 1 - w;
                            }
                            which = w;
                        }
                        enc.end_encoding();
                        cmd.commit();
                        cmd.wait_until_completed();
                        black_box(which)
                    });
                },
            );
        }

        // select64: repeated-selection kernels for k=10 and k=50 (candidate kernels to beat CPU).
        // tile=8192, threadgroup=256 (8 simdgroups × SG_TILE=1024).
        // partial groups = ceil(248320/8192)*8 = 31*8 = 248; each group emits 64 candidates.
        for &k in &[10u32, 50] {
            let k32 = k;
            group.bench_function(BenchmarkId::new("select64", format!("k{k}")), |b| {
                let vocab_u32 = QWEN35_VOCAB_SIZE as u32;
                let tile = 8192u32;
                let tgs_per_vocab = vocab_u32.div_ceil(tile); // 31
                let partial_groups = tgs_per_vocab * 8; // 248 simdgroup slots
                let fan_in = partial_groups.min(32);
                let merge_out_groups = partial_groups.div_ceil(fan_in); // 1 or a few

                // Warm-up
                for _ in 0..10 {
                    let cmd = queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&sel64_first_pipe);
                    enc.set_buffer(0, Some(&logits_buf), 0);
                    enc.set_buffer(1, Some(&scratch_a), 0);
                    enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                    enc.set_bytes(3, 4, &k32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(tgs_per_vocab as u64, 1, 1),
                        metal::MTLSize::new(256, 1, 1),
                    );
                    enc.set_compute_pipeline_state(&sel64_merge_pipe);
                    enc.set_buffer(0, Some(&scratch_a), 0);
                    enc.set_buffer(1, Some(&scratch_b), 0);
                    enc.set_bytes(2, 4, &partial_groups as *const u32 as *const _);
                    enc.set_bytes(3, 4, &k32 as *const u32 as *const _);
                    enc.set_bytes(4, 4, &fan_in as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(merge_out_groups as u64, 1, 1),
                        metal::MTLSize::new(256, 1, 1),
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                }

                b.iter(|| {
                    let cmd = queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&sel64_first_pipe);
                    enc.set_buffer(0, Some(&logits_buf), 0);
                    enc.set_buffer(1, Some(&scratch_a), 0);
                    enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                    enc.set_bytes(3, 4, &k32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(tgs_per_vocab as u64, 1, 1),
                        metal::MTLSize::new(256, 1, 1),
                    );
                    enc.set_compute_pipeline_state(&sel64_merge_pipe);
                    enc.set_buffer(0, Some(&scratch_a), 0);
                    enc.set_buffer(1, Some(&scratch_b), 0);
                    enc.set_bytes(2, 4, &partial_groups as *const u32 as *const _);
                    enc.set_bytes(3, 4, &k32 as *const u32 as *const _);
                    enc.set_bytes(4, 4, &fan_in as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(merge_out_groups as u64, 1, 1),
                        metal::MTLSize::new(256, 1, 1),
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    black_box(())
                });
            });
        }

        // hierarchical k=50: two-stage SIMD-group tournament (no bitonic sort).
        // tile=1024, 256 threads, fan_in=16.
        // first_pass_groups = ceil(248320/1024) = 243; two merge passes: 243→16→1.
        {
            let vocab_u32 = QWEN35_VOCAB_SIZE as u32;
            let tile = 1024u32;
            let first_groups = vocab_u32.div_ceil(tile); // 243

            // Warm-up
            for _ in 0..10 {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&sel50_first_pipe);
                enc.set_buffer(0, Some(&logits_buf), 0);
                enc.set_buffer(1, Some(&scratch_a), 0);
                enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    metal::MTLSize::new(first_groups as u64, 1, 1),
                    metal::MTLSize::new(256, 1, 1),
                );
                let mut cg = first_groups;
                let mut w = 0u8;
                while cg > 1 {
                    let fan_in: u32 = 16u32.min(cg);
                    let og = cg.div_ceil(fan_in);
                    let (ib, ob) = if w == 0 {
                        (&scratch_a, &scratch_b)
                    } else {
                        (&scratch_b, &scratch_a)
                    };
                    enc.set_compute_pipeline_state(&sel50_merge_pipe);
                    enc.set_buffer(0, Some(ib), 0);
                    enc.set_buffer(1, Some(ob), 0);
                    enc.set_bytes(2, 4, &cg as *const u32 as *const _);
                    enc.set_bytes(3, 4, &fan_in as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(og as u64, 1, 1),
                        metal::MTLSize::new(256, 1, 1),
                    );
                    cg = og;
                    w = 1 - w;
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }

            group.bench_function("hierarchical_dispatch_only/k50", |b| {
                b.iter(|| {
                    let cmd = queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&sel50_first_pipe);
                    enc.set_buffer(0, Some(&logits_buf), 0);
                    enc.set_buffer(1, Some(&scratch_a), 0);
                    enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(first_groups as u64, 1, 1),
                        metal::MTLSize::new(256, 1, 1),
                    );
                    let mut current_groups = first_groups;
                    let mut w = 0u8;
                    while current_groups > 1 {
                        let fan_in: u32 = 16u32.min(current_groups);
                        let out_groups = current_groups.div_ceil(fan_in);
                        let (in_b, out_b) = if w == 0 {
                            (&scratch_a, &scratch_b)
                        } else {
                            (&scratch_b, &scratch_a)
                        };
                        enc.set_compute_pipeline_state(&sel50_merge_pipe);
                        enc.set_buffer(0, Some(in_b), 0);
                        enc.set_buffer(1, Some(out_b), 0);
                        enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
                        enc.set_bytes(3, 4, &fan_in as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            metal::MTLSize::new(out_groups as u64, 1, 1),
                            metal::MTLSize::new(256, 1, 1),
                        );
                        current_groups = out_groups;
                        w = 1 - w;
                    }
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    black_box(w)
                });
            });
        }

        group.finish();
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// 9. metal_topk_plus_readback — combined dispatch + compact candidate readback
//
// Measures the full GPU top-k path as seen by the inference loop:
//   dispatch kernels → commit → wait → copy k*8 bytes from shared MTLBuffer.
// Rows: k=1 argmax, k=10/50 select64, k=10/50/256 bitonic (negative baseline).
// ---------------------------------------------------------------------------

fn bench_metal_topk_plus_readback(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use metal::{CompileOptions, Device, MTLResourceOptions};

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("[topk_readback] No Metal device — skipping plus_readback bench");
                return;
            }
        };
        let opts = CompileOptions::new();
        let lib = match device.new_library_with_source(TOPK_BENCH_MSL_SRC, &opts) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[topk_readback] MSL compile failed: {e}");
                return;
            }
        };
        let make_pipe = |name: &str| {
            lib.get_function(name, None)
                .and_then(|f| device.new_compute_pipeline_state_with_function(&f))
                .ok()
        };
        let argmax_first_pipe = match make_pipe("logits_argmax_first") {
            Some(p) => p,
            None => return,
        };
        let argmax_merge_pipe = match make_pipe("logits_argmax_merge") {
            Some(p) => p,
            None => return,
        };
        let fast_first_pipe = match make_pipe("logits_topk_fast_first") {
            Some(p) => p,
            None => return,
        };
        let merge_pass_pipe = match make_pipe("logits_topk_merge_pass") {
            Some(p) => p,
            None => return,
        };
        let sel64_first_pipe = match make_pipe("logits_topk_select64_first") {
            Some(p) => p,
            None => return,
        };
        let sel64_merge_pipe = match make_pipe("logits_topk_select64_merge") {
            Some(p) => p,
            None => return,
        };
        let sel50_first_pipe = match make_pipe("logits_topk_select50_first") {
            Some(p) => p,
            None => return,
        };
        let sel50_merge_pipe = match make_pipe("logits_topk_select50_merge") {
            Some(p) => p,
            None => return,
        };

        let queue = device.new_command_queue();
        let src_data = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xCAFE_BABE);
        let logits_buf =
            device.new_buffer(FULL_READBACK_BYTES, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = logits_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(src_data.as_ptr(), ptr, QWEN35_VOCAB_SIZE);
        }
        let scratch_bytes = 243usize * 256 * 8;
        let scratch_a =
            device.new_buffer(scratch_bytes as u64, MTLResourceOptions::StorageModeShared);
        let scratch_b =
            device.new_buffer(scratch_bytes as u64, MTLResourceOptions::StorageModeShared);

        // Readback buffer sized for max k=256 candidates (256*8=2048 bytes).
        let readback_buf: Vec<u8> = vec![0u8; 256 * 8];

        let mut group = c.benchmark_group("metal_topk_plus_readback");
        group.sample_size(100);
        group.warm_up_time(Duration::from_secs(2));
        group.measurement_time(Duration::from_secs(10));

        // k=1 argmax dispatch + 8-byte readback.
        {
            let k: u32 = 1;
            let vocab_u32 = QWEN35_VOCAB_SIZE as u32;
            let groups = vocab_u32.div_ceil(1024);
            group.bench_function("argmax_plus_readback/k1", |b| {
                b.iter(|| {
                    let cmd = queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&argmax_first_pipe);
                    enc.set_buffer(0, Some(&logits_buf), 0);
                    enc.set_buffer(1, Some(&scratch_a), 0);
                    enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(groups as u64, 1, 1),
                        metal::MTLSize::new(1024, 1, 1),
                    );
                    enc.set_compute_pipeline_state(&argmax_merge_pipe);
                    enc.set_buffer(0, Some(&scratch_a), 0);
                    enc.set_buffer(1, Some(&scratch_b), 0);
                    enc.set_bytes(2, 4, &groups as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(1, 1, 1),
                        metal::MTLSize::new(1024, 1, 1),
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    let mut dst = [0u8; 8];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            scratch_b.contents() as *const u8,
                            dst.as_mut_ptr(),
                            8,
                        );
                    }
                    black_box(dst)
                });
            });
            let _ = k;
        }

        // select64 dispatch + k*8-byte readback for k=10 and k=50.
        for &k in &[10u32, 50] {
            let vocab_u32 = QWEN35_VOCAB_SIZE as u32;
            let tile = 8192u32;
            let tgs_per_vocab = vocab_u32.div_ceil(tile);
            let partial_groups = tgs_per_vocab * 8;
            let fan_in = partial_groups.min(32);
            let merge_out_groups = partial_groups.div_ceil(fan_in);
            let readback_bytes = (k * 8) as usize;
            group.bench_function(
                BenchmarkId::new("select64_plus_readback", format!("k{k}")),
                |b| {
                    b.iter(|| {
                        let cmd = queue.new_command_buffer();
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(&sel64_first_pipe);
                        enc.set_buffer(0, Some(&logits_buf), 0);
                        enc.set_buffer(1, Some(&scratch_a), 0);
                        enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                        enc.set_bytes(3, 4, &k as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            metal::MTLSize::new(tgs_per_vocab as u64, 1, 1),
                            metal::MTLSize::new(256, 1, 1),
                        );
                        enc.set_compute_pipeline_state(&sel64_merge_pipe);
                        enc.set_buffer(0, Some(&scratch_a), 0);
                        enc.set_buffer(1, Some(&scratch_b), 0);
                        enc.set_bytes(2, 4, &partial_groups as *const u32 as *const _);
                        enc.set_bytes(3, 4, &k as *const u32 as *const _);
                        enc.set_bytes(4, 4, &fan_in as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            metal::MTLSize::new(merge_out_groups as u64, 1, 1),
                            metal::MTLSize::new(256, 1, 1),
                        );
                        enc.end_encoding();
                        cmd.commit();
                        cmd.wait_until_completed();
                        let mut dst = vec![0u8; readback_bytes];
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                scratch_b.contents() as *const u8,
                                dst.as_mut_ptr(),
                                readback_bytes,
                            );
                        }
                        black_box(dst)
                    });
                },
            );
        }

        // bitonic negative baseline: dispatch + k*8-byte readback for k=10, 50, 256.
        for &k in &[10u32, 50, 256] {
            let vocab_u32 = QWEN35_VOCAB_SIZE as u32;
            let readback_bytes = (k * 8) as usize;
            group.bench_function(
                BenchmarkId::new("bitonic_plus_readback", format!("k{k}")),
                |b| {
                    b.iter(|| {
                        let cmd = queue.new_command_buffer();
                        let enc = cmd.new_compute_command_encoder();
                        let tile = 1024u32;
                        let first_groups = vocab_u32.div_ceil(tile);
                        enc.set_compute_pipeline_state(&fast_first_pipe);
                        enc.set_buffer(0, Some(&logits_buf), 0);
                        enc.set_buffer(1, Some(&scratch_a), 0);
                        enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                        enc.set_bytes(3, 4, &k as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            metal::MTLSize::new(first_groups as u64, 1, 1),
                            metal::MTLSize::new(256, 1, 1),
                        );
                        let mut current_groups = first_groups;
                        let mut w = 0u8;
                        while current_groups > 1 {
                            let fan_in = (2048u32 / k).max(2).min(current_groups);
                            let out_groups = current_groups.div_ceil(fan_in);
                            let (in_b, out_b) = if w == 0 {
                                (&scratch_a, &scratch_b)
                            } else {
                                (&scratch_b, &scratch_a)
                            };
                            enc.set_compute_pipeline_state(&merge_pass_pipe);
                            enc.set_buffer(0, Some(in_b), 0);
                            enc.set_buffer(1, Some(out_b), 0);
                            enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
                            enc.set_bytes(3, 4, &k as *const u32 as *const _);
                            enc.set_bytes(4, 4, &fan_in as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                metal::MTLSize::new(out_groups as u64, 1, 1),
                                metal::MTLSize::new(256, 1, 1),
                            );
                            current_groups = out_groups;
                            w = 1 - w;
                        }
                        enc.end_encoding();
                        cmd.commit();
                        cmd.wait_until_completed();
                        let final_buf = if w == 0 { &scratch_a } else { &scratch_b };
                        let mut dst = vec![0u8; readback_bytes];
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                final_buf.contents() as *const u8,
                                dst.as_mut_ptr(),
                                readback_bytes,
                            );
                        }
                        black_box(dst)
                    });
                },
            );
        }

        // hierarchical k=50 dispatch + 400-byte readback.
        {
            let vocab_u32 = QWEN35_VOCAB_SIZE as u32;
            let tile = 1024u32;
            let first_groups = vocab_u32.div_ceil(tile); // 243
            let readback_bytes = 50usize * 8; // 400 bytes
            group.bench_function("hierarchical_plus_readback/k50", |b| {
                b.iter(|| {
                    let cmd = queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&sel50_first_pipe);
                    enc.set_buffer(0, Some(&logits_buf), 0);
                    enc.set_buffer(1, Some(&scratch_a), 0);
                    enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(first_groups as u64, 1, 1),
                        metal::MTLSize::new(256, 1, 1),
                    );
                    let mut current_groups = first_groups;
                    let mut w = 0u8;
                    while current_groups > 1 {
                        let fan_in: u32 = 16u32.min(current_groups);
                        let out_groups = current_groups.div_ceil(fan_in);
                        let (in_b, out_b) = if w == 0 {
                            (&scratch_a, &scratch_b)
                        } else {
                            (&scratch_b, &scratch_a)
                        };
                        enc.set_compute_pipeline_state(&sel50_merge_pipe);
                        enc.set_buffer(0, Some(in_b), 0);
                        enc.set_buffer(1, Some(out_b), 0);
                        enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
                        enc.set_bytes(3, 4, &fan_in as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            metal::MTLSize::new(out_groups as u64, 1, 1),
                            metal::MTLSize::new(256, 1, 1),
                        );
                        current_groups = out_groups;
                        w = 1 - w;
                    }
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    let final_buf = if w == 0 { &scratch_a } else { &scratch_b };
                    let mut dst = vec![0u8; readback_bytes];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            final_buf.contents() as *const u8,
                            dst.as_mut_ptr(),
                            readback_bytes,
                        );
                    }
                    black_box(dst)
                });
            });
        }

        let _ = readback_buf;
        group.finish();
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// 10. candidate_sampler — sample from compact CandidateSet (CPU, post-GPU)
//
// Measures the CPU sampling cost AFTER GPU top-k: temperature + top-p + sample
// from k candidates.  This is the "after" measurement for the CPU sampling component.
// ---------------------------------------------------------------------------

fn bench_candidate_sampler(c: &mut Criterion) {
    use lattice_inference::sampling::{Candidate, CandidateSet};

    let logits = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xDEAD_BEEF);

    let mut group = c.benchmark_group("candidate_sampler");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &k in &[1usize, 10, 50, 256] {
        // Pre-build candidate set (simulates GPU top-k output).
        let candidates: Vec<Candidate> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| Candidate {
                token_id: i as u32,
                logit: l,
            })
            .collect::<Vec<_>>()
            .into_iter()
            .take(k)
            .collect();

        group.throughput(criterion::Throughput::Elements(k as u64));
        group.bench_function(
            BenchmarkId::new("sample_top_p_from_k_candidates", format!("k{k}")),
            |b| {
                b.iter_batched(
                    || candidates.clone(),
                    |cands| {
                        let mut cs = CandidateSet::from_candidates(cands);
                        black_box(cs.sample_top_p(0.9, 0.5))
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Helpers for parity and NaN/tie tests
// ---------------------------------------------------------------------------

/// CPU top-k oracle: returns up to k (token_id, logit) pairs sorted descending
/// by logit.  NaN logits are excluded.  Ties broken by ascending token_id,
/// matching the GPU `topk_better` comparator.
fn topk_cpu_oracle(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    indexed.truncate(k);
    indexed
}

/// Dispatch the optimized top-k kernels (Phase 3b) and return (token_id, logit) pairs.
///
/// Routes k=1 through argmax kernels, k>1 through fast_first + merge.
/// Tile size 1024; no padding waste.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn gpu_topk_dispatch(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    argmax_first_pipe: &metal::ComputePipelineState,
    argmax_merge_pipe: &metal::ComputePipelineState,
    fast_first_pipe: &metal::ComputePipelineState,
    merge_pipe: &metal::ComputePipelineState,
    logits: &[f32],
    k: u32,
) -> Vec<(u32, f32)> {
    use metal::MTLResourceOptions;
    let n = logits.len();
    let vocab_u32 = n as u32;

    let logits_buf = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(logits.as_ptr(), logits_buf.contents() as *mut f32, n);
    }

    // Size scratch for tile=1024 groups
    let tile_groups = vocab_u32.div_ceil(1024) as usize;
    let scratch_bytes = (tile_groups * 256 * 8).max(k as usize * 8) as u64;
    let scratch_a = device.new_buffer(scratch_bytes, MTLResourceOptions::StorageModeShared);
    let scratch_b = device.new_buffer(scratch_bytes, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    let which: u8;
    if k == 1 {
        let groups = vocab_u32.div_ceil(1024);
        enc.set_compute_pipeline_state(argmax_first_pipe);
        enc.set_buffer(0, Some(&logits_buf), 0);
        enc.set_buffer(1, Some(&scratch_a), 0);
        enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(groups as u64, 1, 1),
            metal::MTLSize::new(1024, 1, 1),
        );
        enc.set_compute_pipeline_state(argmax_merge_pipe);
        enc.set_buffer(0, Some(&scratch_a), 0);
        enc.set_buffer(1, Some(&scratch_b), 0);
        enc.set_bytes(2, 4, &groups as *const u32 as *const _);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(1024, 1, 1),
        );
        which = 1;
    } else {
        let first_groups = vocab_u32.div_ceil(1024);
        enc.set_compute_pipeline_state(fast_first_pipe);
        enc.set_buffer(0, Some(&logits_buf), 0);
        enc.set_buffer(1, Some(&scratch_a), 0);
        enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
        enc.set_bytes(3, 4, &k as *const u32 as *const _);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(first_groups as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        let mut current_groups = first_groups;
        let mut w: u8 = 0;
        while current_groups > 1 {
            let fan_in = (2048u32 / k).max(2).min(current_groups);
            let out_groups = current_groups.div_ceil(fan_in);
            enc.set_compute_pipeline_state(merge_pipe);
            let (in_b, out_b) = if w == 0 {
                (&scratch_a, &scratch_b)
            } else {
                (&scratch_b, &scratch_a)
            };
            enc.set_buffer(0, Some(in_b), 0);
            enc.set_buffer(1, Some(out_b), 0);
            enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
            enc.set_bytes(3, 4, &k as *const u32 as *const _);
            enc.set_bytes(4, 4, &fan_in as *const u32 as *const _);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(out_groups as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            current_groups = out_groups;
            w = 1 - w;
        }
        which = w;
    }

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let k_usize = k as usize;
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct GpuCandidate {
        logit: f32,
        token_id: u32,
    }
    let final_buf = if which == 0 { &scratch_a } else { &scratch_b };
    unsafe {
        let ptr = final_buf.contents() as *const GpuCandidate;
        (0..k_usize)
            .map(|i| {
                let gc = *ptr.add(i);
                (gc.token_id, gc.logit)
            })
            .collect()
    }
}

/// Dispatch the hierarchical k=50 SIMD-group tournament and return candidates.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn gpu_topk_select50_dispatch(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    sel50_first_pipe: &metal::ComputePipelineState,
    sel50_merge_pipe: &metal::ComputePipelineState,
    logits: &[f32],
) -> Vec<(u32, f32)> {
    use metal::MTLResourceOptions;
    let n = logits.len();
    let vocab_u32 = n as u32;
    let logits_buf = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(logits.as_ptr(), logits_buf.contents() as *mut f32, n);
    }
    let tile_groups = vocab_u32.div_ceil(1024) as usize;
    let scratch_bytes = (tile_groups * 50 * 8).max(50 * 8) as u64;
    let scratch_a = device.new_buffer(scratch_bytes, MTLResourceOptions::StorageModeShared);
    let scratch_b = device.new_buffer(scratch_bytes, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    let first_groups = vocab_u32.div_ceil(1024);
    enc.set_compute_pipeline_state(sel50_first_pipe);
    enc.set_buffer(0, Some(&logits_buf), 0);
    enc.set_buffer(1, Some(&scratch_a), 0);
    enc.set_bytes(2, 4, &vocab_u32 as *const u32 as *const _);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(first_groups as u64, 1, 1),
        metal::MTLSize::new(256, 1, 1),
    );

    let mut current_groups = first_groups;
    let mut w: u8 = 0;
    while current_groups > 1 {
        let fan_in: u32 = 16u32.min(current_groups);
        let out_groups = current_groups.div_ceil(fan_in);
        let (in_b, out_b) = if w == 0 {
            (&scratch_a, &scratch_b)
        } else {
            (&scratch_b, &scratch_a)
        };
        enc.set_compute_pipeline_state(sel50_merge_pipe);
        enc.set_buffer(0, Some(in_b), 0);
        enc.set_buffer(1, Some(out_b), 0);
        enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
        enc.set_bytes(3, 4, &fan_in as *const u32 as *const _);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(out_groups as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        current_groups = out_groups;
        w = 1 - w;
    }

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct GpuCandidate {
        logit: f32,
        token_id: u32,
    }
    let final_buf = if w == 0 { &scratch_a } else { &scratch_b };
    unsafe {
        let ptr = final_buf.contents() as *const GpuCandidate;
        (0..50usize)
            .map(|i| {
                let gc = *ptr.add(i);
                (gc.token_id, gc.logit)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 11. topk_parity — GPU vs CPU correctness gate (Metal-gated)
//
// Phase 3a scaffolding: validates the current bitonic top-k returns the same
// candidates as the CPU oracle for k=1, 50, 256 over PARITY_VOCAB_SIZE=4096
// logits.  4096 logits at tile=1280 → 4 first-pass groups → 1 merge pass.
//
// Correctness gates run ONCE at the START of this benchmark group (outside
// the timing loop).  Any mismatch panics immediately so failures are visible
// in `cargo bench` output.  The penalty-ordering section documents the current
// bug (Target 1) by proving that penalty MUST be applied before top-k.
//
// Run:
//   cargo bench -p lattice-inference --features "f16,metal-gpu" \
//     --bench topk_readback -- topk_parity
// ---------------------------------------------------------------------------

fn bench_topk_parity(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use metal::{CompileOptions, Device};

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("[topk_parity] No Metal device — skipping");
                return;
            }
        };
        let opts = CompileOptions::new();
        let lib = match device.new_library_with_source(TOPK_BENCH_MSL_SRC, &opts) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[topk_parity] MSL compile failed: {e}");
                return;
            }
        };
        let make_pipe = |name: &str| {
            lib.get_function(name, None)
                .and_then(|f| device.new_compute_pipeline_state_with_function(&f))
                .ok()
        };
        let argmax_first_pipe = match make_pipe("logits_argmax_first") {
            Some(p) => p,
            None => {
                eprintln!("[topk_parity] argmax_first pipeline failed");
                return;
            }
        };
        let argmax_merge_pipe = match make_pipe("logits_argmax_merge") {
            Some(p) => p,
            None => {
                eprintln!("[topk_parity] argmax_merge pipeline failed");
                return;
            }
        };
        let fast_first_pipe = match make_pipe("logits_topk_fast_first") {
            Some(p) => p,
            None => {
                eprintln!("[topk_parity] topk_fast_first pipeline failed");
                return;
            }
        };
        let merge_pipe = match make_pipe("logits_topk_merge_pass") {
            Some(p) => p,
            None => {
                eprintln!("[topk_parity] merge_pass pipeline failed");
                return;
            }
        };
        let sel50_first_pipe = match make_pipe("logits_topk_select50_first") {
            Some(p) => p,
            None => {
                eprintln!("[topk_parity] select50_first pipeline failed");
                return;
            }
        };
        let sel50_merge_pipe = match make_pipe("logits_topk_select50_merge") {
            Some(p) => p,
            None => {
                eprintln!("[topk_parity] select50_merge pipeline failed");
                return;
            }
        };
        let queue = device.new_command_queue();

        let parity_logits = rand_f32_vec(PARITY_VOCAB_SIZE, 0xFEED_C0DE);

        let dispatch = |logits: &[f32], k: u32| {
            gpu_topk_dispatch(
                &device,
                &queue,
                &argmax_first_pipe,
                &argmax_merge_pipe,
                &fast_first_pipe,
                &merge_pipe,
                logits,
                k,
            )
        };

        // --- parity: GPU top-k matches CPU oracle for k=1, 10, 50, 256 at 4096 vocab ---
        for &k in &[1u32, 10, 50, 256] {
            let cpu = topk_cpu_oracle(&parity_logits, k as usize);
            let gpu = dispatch(&parity_logits, k);
            assert_eq!(
                gpu.len(),
                k as usize,
                "parity k={k}: GPU returned {} candidates, expected {k}",
                gpu.len()
            );
            assert_eq!(
                gpu[0].0, cpu[0].0,
                "parity k={k}: GPU top-1 token_id={} != CPU top-1 token_id={}",
                gpu[0].0, cpu[0].0
            );
            assert!(
                (gpu[0].1 - cpu[0].1).abs() < 1e-5,
                "parity k={k}: GPU logit={:.6} != CPU logit={:.6}",
                gpu[0].1,
                cpu[0].1
            );
            let cpu_ids: std::collections::HashSet<u32> = cpu.iter().map(|&(id, _)| id).collect();
            for (i, &(gpu_id, _)) in gpu.iter().enumerate() {
                assert!(
                    cpu_ids.contains(&gpu_id),
                    "parity k={k} vocab4096: GPU candidate[{i}] token_id={gpu_id} not in CPU top-k"
                );
            }
        }

        // --- full-vocab parity: bitonic GPU matches CPU oracle at QWEN35_VOCAB_SIZE=248320 ---
        let full_vocab_logits = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xABCD_1234);
        for &k in &[1u32, 10, 50] {
            let cpu = topk_cpu_oracle(&full_vocab_logits, k as usize);
            let gpu = dispatch(&full_vocab_logits, k);
            assert_eq!(
                gpu.len(),
                k as usize,
                "full_vocab parity k={k}: length mismatch"
            );
            assert_eq!(
                gpu[0].0, cpu[0].0,
                "full_vocab parity k={k}: GPU top-1={} != CPU top-1={}",
                gpu[0].0, cpu[0].0
            );
            let cpu_ids: std::collections::HashSet<u32> = cpu.iter().map(|&(id, _)| id).collect();
            for (i, &(gpu_id, _)) in gpu.iter().enumerate() {
                assert!(
                    cpu_ids.contains(&gpu_id),
                    "full_vocab parity k={k} vocab248320: GPU candidate[{i}] token_id={gpu_id} not in CPU top-k"
                );
            }
        }

        // --- NaN exclusion: NaN at token 0 must not outrank finite logits ---
        {
            let mut nan_logits = vec![0.0f32; 1280];
            nan_logits[0] = f32::NAN;
            nan_logits[5] = 2.0;
            nan_logits[10] = 1.5;
            let gpu_nan = dispatch(&nan_logits, 1);
            assert_eq!(
                gpu_nan[0].0, 5,
                "nan_exclusion: GPU top-1 must be token 5, got {}",
                gpu_nan[0].0
            );
        }

        // --- tie-breaking: equal logits → lower token_id wins (topk_better) ---
        {
            let mut tie_logits = vec![0.0f32; 1280];
            tie_logits[100] = 1.0;
            tie_logits[7] = 1.0;
            tie_logits[0] = 0.5;
            let gpu_tie = dispatch(&tie_logits, 1);
            assert_eq!(
                gpu_tie[0].0, 7,
                "tie_breaking: lower token_id=7 must win over 100, got {}",
                gpu_tie[0].0
            );
        }

        // --- penalty ordering (guard confirmed by Target 1 fix) ---
        // GPU returns raw logit winner (token 0). CPU with penalty=1.2 prefers token 1.
        // The repetition_penalty guard (now added to use_compact) prevents misuse.
        {
            let mut pen_logits = vec![0.0f32; 1280];
            pen_logits[0] = 2.0;
            pen_logits[1] = 1.85;

            let cpu_no_pen = topk_cpu_oracle(&pen_logits, 1);
            assert_eq!(
                cpu_no_pen[0].0, 0,
                "penalty_doc: no-penalty oracle must return token 0"
            );

            let mut pen_applied = pen_logits.clone();
            pen_applied[0] /= 1.2_f32;
            let cpu_pen = topk_cpu_oracle(&pen_applied, 1);
            assert_eq!(
                cpu_pen[0].0, 1,
                "penalty_doc: penalty-applied oracle must return token 1, got {}",
                cpu_pen[0].0
            );

            let gpu_raw = dispatch(&pen_logits, 1);
            assert_eq!(gpu_raw[0].0, 0, "penalty_doc: GPU raw must return token 0");
        }

        // --- hierarchical k=50 parity: correct top-50 at 4096 vocab ---
        {
            let cpu50 = topk_cpu_oracle(&parity_logits, 50);
            let hier = gpu_topk_select50_dispatch(
                &device,
                &queue,
                &sel50_first_pipe,
                &sel50_merge_pipe,
                &parity_logits,
            );
            assert_eq!(
                hier.len(),
                50,
                "hierarchical k=50: returned {} candidates",
                hier.len()
            );
            assert_eq!(
                hier[0].0, cpu50[0].0,
                "hierarchical k=50: top-1 token_id={} != CPU top-1 token_id={}",
                hier[0].0, cpu50[0].0
            );
            assert!(
                (hier[0].1 - cpu50[0].1).abs() < 1e-5,
                "hierarchical k=50: top-1 logit={:.6} != CPU logit={:.6}",
                hier[0].1,
                cpu50[0].1
            );
            let cpu_ids: std::collections::HashSet<u32> = cpu50.iter().map(|&(id, _)| id).collect();
            for (i, &(gpu_id, _)) in hier.iter().enumerate() {
                assert!(
                    cpu_ids.contains(&gpu_id),
                    "hierarchical k=50 parity: candidate[{i}] token_id={gpu_id} not in CPU top-50"
                );
            }
        }

        // --- hierarchical k=50 full-vocab parity: correct top-50 at 248320 vocab ---
        {
            let full_vocab_logits_h = rand_f32_vec(QWEN35_VOCAB_SIZE, 0xABCD_5678);
            let cpu50 = topk_cpu_oracle(&full_vocab_logits_h, 50);
            let hier = gpu_topk_select50_dispatch(
                &device,
                &queue,
                &sel50_first_pipe,
                &sel50_merge_pipe,
                &full_vocab_logits_h,
            );
            assert_eq!(
                hier.len(),
                50,
                "hierarchical full_vocab k=50: length mismatch"
            );
            assert_eq!(
                hier[0].0, cpu50[0].0,
                "hierarchical full_vocab k=50: top-1 GPU={} != CPU={}",
                hier[0].0, cpu50[0].0
            );
            let cpu_ids: std::collections::HashSet<u32> = cpu50.iter().map(|&(id, _)| id).collect();
            for (i, &(gpu_id, _)) in hier.iter().enumerate() {
                assert!(
                    cpu_ids.contains(&gpu_id),
                    "hierarchical full_vocab k=50: candidate[{i}] token_id={gpu_id} not in CPU top-50"
                );
            }
        }

        // Timing: dispatch roundtrip at PARITY_VOCAB_SIZE for Phase 3b comparison.
        let mut group = c.benchmark_group("topk_parity");
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));

        for &k in &[1u32, 10, 50, 256] {
            group.bench_function(
                BenchmarkId::new("dispatch_and_wait_4096", format!("k{k}")),
                |b| {
                    b.iter(|| black_box(dispatch(&parity_logits, k)));
                },
            );
        }

        // Hierarchical k=50 timing at PARITY_VOCAB_SIZE.
        group.bench_function("hierarchical_dispatch_and_wait_4096/k50", |b| {
            b.iter(|| {
                black_box(gpu_topk_select50_dispatch(
                    &device,
                    &queue,
                    &sel50_first_pipe,
                    &sel50_merge_pipe,
                    &parity_logits,
                ))
            });
        });

        group.finish();
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _ = c;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    name = topk_readback_benches;
    config = Criterion::default();
    targets =
        bench_full_logit_readback,
        bench_compact_readback,
        bench_sampling_pipeline,
        bench_topk_selection,
        bench_full_logit_readback_metal,
        bench_compact_readback_metal,
        bench_noop_command_buffer,
        bench_metal_topk_dispatch_only,
        bench_metal_topk_plus_readback,
        bench_candidate_sampler,
        bench_topk_parity,
);
criterion_main!(topk_readback_benches);
