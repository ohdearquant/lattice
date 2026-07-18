//! Bounded LRU cache of dequantized MoE routed-expert weight buffers,
//! populated lazily from mmap'd `.q4` files on cache miss (issue #682, Stage
//! 1: "dequant-on-demand routed experts with a bounded LRU cache").
//!
//! ## Why this exists
//!
//! `MetalQwen35State::load_moe_ffn_q4` used to eagerly dequantize every
//! routed expert, in every MoE layer, into one giant f16-resident Metal
//! buffer per (layer, gate_up|down) tensor at load time — for
//! `Qwen3.5-35B-A3B` (256 experts × 40 layers) that is ~61 GiB, regardless
//! of how many experts a token actually activates (top-8 of 256 = 3.1%).
//! This module replaces that with a fixed-size pool of `N` pre-allocated
//! per-expert buffer slots, evicted least-recently-used, backed by an mmap
//! of the on-disk `.q4` file (same container format, no header/layout
//! change — see `PLAN.md` §0/§3 for the block-alignment proof this relies
//! on: every expert's Q4 bytes are already a contiguous, 32-value-block-
//! aligned byte range because `num_experts` is the *outer* dimension of the
//! fused per-layer tensor).
//!
//! ## GPU buffer lifetime invariant (read before touching eviction logic)
//!
//! `MetalQwen35State::forward_step_inner_impl`'s default (fused) decode path
//! encodes **the entire forward pass for one token — every layer, including
//! every routed-expert GEMV dispatch — into a single Metal command buffer**,
//! then calls `cmd.commit(); cmd.wait_until_completed();` before
//! `forward_step` returns. Decode is otherwise fully synchronous/serial (one
//! token at a time, no prefetch/pipelining across tokens in Stage 1). Two
//! consequences pin down exactly when a cache slot's Metal buffer may be
//! safely overwritten:
//!
//! 1. **Across tokens**: by the time the NEXT token's `encode_moe_ffn` call
//!    (and thus the next round of `ExpertSlotCache::resolve` calls) begins,
//!    `wait_until_completed()` has already returned for the previous
//!    token's command buffer — every GPU command that could reference any
//!    slot has finished executing. So any slot may be evicted at a token
//!    boundary, unconditionally.
//! 2. **Within one token**: a slot must not be overwritten while a command
//!    appended earlier in the *same, still-open* command buffer still reads
//!    it. The write (a CPU `memcpy` into a `StorageModeShared` buffer's
//!    `contents()`) and the read (a GPU dispatch encoded before the buffer
//!    is committed) are not ordered by anything except which one the CPU
//!    performs first — encoding a GEMV against a slot, then later in the
//!    same token overwriting that slot before the buffer is submitted, would
//!    make the already-encoded GEMV race the new dequant write once the GPU
//!    actually executes it.
//!
//! Stage 1's chosen scheme is the "evict-only-after-token-boundary" option:
//! [`ExpertSlotCache::begin_token`] clears a per-slot "touched this token"
//! flag, [`ExpertSlotCache::resolve`] sets it on every slot it touches (hit
//! or miss), and eviction ([`ExpertSlotCache::pick_eviction_slot`]) only
//! ever selects an *untouched* slot. This is sufficient — not just
//! plausible — because [`moe_expert_cache_num_slots`] is required to return
//! `num_slots >= top_k` (validated at construction, see
//! [`ExpertSlotCache::new`]), and a single token's routed-expert loop
//! (`encode_moe_ffn` Step 3) never resolves more than `top_k` distinct
//! experts. So there are always enough untouched slots left to satisfy every
//! miss within one token, and no same-token slot ever needs to be evicted
//! twice.
//!
//! No fence, semaphore, or double-buffering is needed beyond this
//! touched-this-token bookkeeping *because* decode never overlaps two
//! tokens' GPU work (prefetching a future token's experts while the
//! current token's command buffer is still in flight would need to
//! revisit this invariant — out of scope; see the note below, which
//! covers a narrower, already-safe form of prefetch).
//!
//! ## Stage 2: within-token prefetch overlap
//!
//! `encode_moe_ffn` issues every routed-expert load for the CURRENT token
//! right after CPU routing decides `selected`, splits into three phases
//! ([`ExpertSlotCache::plan_prefetch`] → [`ExpertSlotCache::spawn_dequant`]
//! → [`ExpertSlotCache::apply_prefetch_results`]) so the actual I/O +
//! dequant work overlaps with encoding Step 2's shared-expert GEMVs
//! instead of blocking in front of them:
//!
//! 1. **Plan** (single-threaded, cheap): classify every selected expert as
//!    a hit or miss and commit every slot's ownership bookkeeping (owner,
//!    `expert_to_slot`, `slot_touched`, LRU position) — see
//!    `plan_prefetch`'s doc comment. A miss also clears the assigned
//!    slot's [`ExpertSlotCache::slot_ready`] flag to `false` *before* any
//!    dequant work starts.
//! 2. **Spawn**: `spawn_dequant` hands the plan's misses to a
//!    `std::thread::scope`-spawned thread (optionally rayon-parallel
//!    across misses within that one thread), borrowing only the cache's
//!    read-only mmap'd byte table — never its Metal buffers or
//!    bookkeeping — and returns immediately with a join handle.
//! 3. `encode_moe_ffn` encodes Step 2 on the calling thread while the
//!    spawned dequant work runs concurrently, THEN joins the handle and
//!    calls `apply_prefetch_results`, which copies each result into its
//!    slot and only THEN flips `slot_ready` back to `true`. Step 3's
//!    `get_prefetched` lookups (after the join) never race the copy.
//!
//! This stays inside the "within one token" case the invariant above
//! already covers (not the cross-token case the paragraph above calls out
//! of scope): every task from one `plan_prefetch` call owns a slot no
//! other task or bookkeeping mutation touches concurrently, and Step 2's
//! GEMV dispatches never reference a slot the same token's dequant phase
//! is still writing (they read the shared-expert buffers, which
//! `spawn_dequant`'s closure never touches).
//!
//! **Failure isolation** (`slot_ready`): if a dequant task never
//! completes (panics, most likely), its slot's ownership is already
//! committed but `slot_ready` stays `false` — `get_prefetched` refuses to
//! serve it, and the NEXT `plan_prefetch` call for that expert sees an
//! unready mapping and reloads into the same slot (no eviction, since
//! this expert already owns it) rather than treating it as a hit. A
//! sibling cache's (e.g. gate_up succeeding while down panics, or vice
//! versa) already-joined results are still applied — only the failed
//! side's slots stay unready — before the failure is propagated.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use rayon::prelude::*;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use std::collections::VecDeque;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use std::path::Path;

/// Environment variable overriding the number of resident expert-cache slots
/// per (layer, gate_up|down) tensor. Primarily for tests and manual tuning;
/// production sizing should rely on [`moe_expert_cache_num_slots`]'s
/// device-budget derivation.
pub const MOE_EXPERT_CACHE_SLOTS_ENV: &str = "LATTICE_MOE_EXPERT_CACHE_SLOTS";

/// Cache-pool sizing knobs, mirroring the shape of `PagedKVCacheConfig`
/// (`kv_cache/paged.rs`) for the analogous MoE-expert paging problem.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoeExpertCacheConfig {
    /// Explicit slot-count override. `None` defers to
    /// [`moe_expert_cache_num_slots`]'s device-budget-derived default.
    pub num_slots: Option<usize>,
}

impl MoeExpertCacheConfig {
    /// Read [`MOE_EXPERT_CACHE_SLOTS_ENV`]. Unset → `Ok(Self { num_slots: None
    /// })`, which falls back to the device-budget default. Set but not a
    /// positive integer → `Err`: a garbage override (typo, empty string,
    /// `0`, negative) must fail loudly rather than being silently treated
    /// as "unset" and falling back to a default the caller did not ask for
    /// — see the working-set-budget guard this configures downstream in
    /// [`moe_expert_cache_num_slots`].
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(MOE_EXPERT_CACHE_SLOTS_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self { num_slots: None }),
            Err(std::env::VarError::NotUnicode(raw)) => Err(format!(
                "{MOE_EXPERT_CACHE_SLOTS_ENV}={raw:?} is not valid UTF-8 — refusing to fall \
                 back to the device-budget default silently; unset the variable to use the \
                 default"
            )),
            Ok(raw) => {
                let trimmed = raw.trim();
                match trimmed.parse::<usize>() {
                    Ok(0) => Err(format!(
                        "{MOE_EXPERT_CACHE_SLOTS_ENV}=\"{raw}\" is 0 — an expert-cache needs at \
                         least 1 slot; unset the variable to use the device-budget default"
                    )),
                    Ok(n) => Ok(Self { num_slots: Some(n) }),
                    Err(e) => Err(format!(
                        "{MOE_EXPERT_CACHE_SLOTS_ENV}=\"{raw}\" is not a valid positive integer \
                         ({e}) — refusing to fall back to the device-budget default silently; \
                         unset the variable to use the default"
                    )),
                }
            }
        }
    }
}

/// Derive how many per-expert buffer slots one (layer, gate_up|down) cache
/// should hold.
///
/// Pure function (no `Device`/`Buffer` dependency) so the sizing policy is
/// unit-testable without a GPU. Policy, per PLAN.md's Stage 1 spec:
///
/// - An explicit `cfg.num_slots` always wins, clamped to
///   `[top_k, num_experts]` (below `top_k` the cache cannot serve even one
///   token's routed-expert set; above `num_experts` is meaningless — there
///   is nothing more to cache).
/// - Otherwise, default to `num_experts` (the "zero-eviction fast path": all
///   experts fit, laziness only defers *when* each is dequantized, not
///   whether it ultimately stays resident) **if** that fits under
///   `0.85 * recommended_max_working_set_size` split evenly across every
///   MoE layer.
/// - If it doesn't fit, auto-shrink to the largest `num_slots` that does,
///   floored at `top_k`.
/// - If even `top_k` slots' worth of bytes don't fit the per-layer budget,
///   return `Err` — this checkpoint's expert shape cannot run a functioning
///   cache on this device, full stop (no silent degradation to a
///   non-functional cache size).
pub fn moe_expert_cache_num_slots(
    cfg: &MoeExpertCacheConfig,
    num_experts: usize,
    top_k: usize,
    per_expert_bytes: u64,
    num_moe_layers: usize,
    recommended_max_working_set_size: u64,
) -> Result<usize, String> {
    if num_experts == 0 {
        return Err("moe_expert_cache_num_slots: num_experts must be > 0".to_string());
    }
    if top_k == 0 {
        return Err("moe_expert_cache_num_slots: top_k must be > 0".to_string());
    }
    if top_k > num_experts {
        return Err(format!(
            "moe_expert_cache_num_slots: top_k ({top_k}) exceeds num_experts ({num_experts})"
        ));
    }

    // The working-set budget is derived and enforced unconditionally — an
    // explicit `cfg.num_slots` override changes how many slots we'd *like*,
    // never whether the device can actually hold them. Skipping this for
    // the override case is exactly the bug this function used to have: a
    // garbage-large `LATTICE_MOE_EXPERT_CACHE_SLOTS` could clamp up to
    // `num_experts` and reconstruct the ~61 GiB fully-resident allocation
    // Stage 1 exists to avoid.
    let num_moe_layers = num_moe_layers.max(1) as u64;
    let threshold = (recommended_max_working_set_size as f64 * 0.85) as u64;
    let per_layer_budget = threshold / num_moe_layers;

    let min_required_bytes = per_expert_bytes.saturating_mul(top_k as u64);
    if min_required_bytes > per_layer_budget {
        return Err(format!(
            "moe_expert_cache_num_slots: even the minimum top_k={top_k} concurrently-resident \
             expert slots need {min_required_bytes} bytes/layer, which exceeds this device's \
             per-layer MoE budget of {per_layer_budget} bytes (0.85 × \
             recommendedMaxWorkingSetSize={recommended_max_working_set_size} / \
             {num_moe_layers} MoE layers). This checkpoint's expert shape cannot fit even the \
             lazy dequant-on-demand cache on this device."
        ));
    }

    let affordable = if per_expert_bytes == 0 {
        num_experts
    } else {
        (per_layer_budget / per_expert_bytes) as usize
    };
    let budget_max_slots = affordable.clamp(top_k, num_experts);

    if let Some(n) = cfg.num_slots {
        // An override may ask for fewer slots than the budget-derived
        // maximum (that's just a smaller cache, always safe) but must never
        // be granted more than `budget_max_slots` — silently capping here
        // mirrors the no-override path's own auto-shrink-to-fit behavior
        // (this function already silently shrinks the `num_experts`
        // default when it doesn't fit; an override gets the same courtesy,
        // never a memory-budget exemption).
        return Ok(n.clamp(top_k, budget_max_slots));
    }

    Ok(budget_max_slots)
}

// ---------------------------------------------------------------------------
// Metal-backed slot cache (macOS + metal-gpu only: needs `Device`/`Buffer`).
// ---------------------------------------------------------------------------

/// Read-only description of where one MoE tensor's (gate_up or down)
/// per-expert Q4 byte ranges live inside an mmap'd `.q4` file, plus the
/// dequant routine scoped to a single expert's slice.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
struct ExpertByteTable {
    mmap: memmap2::Mmap,
    payload_offset: u64,
    /// Element count of ONE expert's slice (i.e. `original_len / num_experts`).
    per_expert_elems: usize,
    per_expert_bytes: usize,
    num_experts: usize,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
impl ExpertByteTable {
    /// Open `path`, validate its header against `expected_shape` (same
    /// mismatched/transposed-layout guard as
    /// `MetalQwen35State::load_q4_mmap_dequant_f16`), mmap it, and derive
    /// the per-expert byte-offset table. `expected_shape[0]` must be
    /// `num_experts` (outer dimension) — this is the structural invariant
    /// PLAN.md §3 proves holds for every Q4 MoE checkpoint this loader
    /// writes; it is re-verified here, not just assumed, via the
    /// block-alignment check below.
    fn open(path: &Path, expected_shape: &[usize]) -> Result<Self, String> {
        use crate::weights::q4_weights::validate_q4_file;

        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let header = validate_q4_file(&mut file, path, Some(expected_shape))
            .map_err(|e| format!("failed to validate Q4 payload {}: {e}", path.display()))?;

        let num_experts = expected_shape[0];
        if num_experts == 0 {
            return Err(format!(
                "{}: expert-major shape {expected_shape:?} has zero experts",
                path.display()
            ));
        }
        if !header.original_len.is_multiple_of(num_experts) {
            return Err(format!(
                "{}: original_len {} is not evenly divisible by num_experts {} — the \
                 expert-major outer-dimension slicing invariant this cache relies on does not \
                 hold for this file",
                path.display(),
                header.original_len,
                num_experts
            ));
        }
        let per_expert_elems = header.original_len / num_experts;
        // Block-alignment invariant (PLAN.md §0/§3): every expert's slice
        // must start and end on a 32-element Q4 block boundary so it can be
        // sliced out of the fused per-layer tensor without touching a
        // neighboring expert's blocks. Structurally true for every
        // `moe_intermediate_size`/`hidden_size` combination `Qwen35Config`
        // currently supports (`2*inter*hidden` and `hidden*inter` are both
        // multiples of 32), but asserted here — not just claimed in a
        // comment — because a violation would silently corrupt or overrun
        // neighboring experts' bytes instead of failing to load.
        if !per_expert_elems.is_multiple_of(32) {
            return Err(format!(
                "{}: per-expert element count {per_expert_elems} is not a multiple of the Q4 \
                 block size (32) — an expert's byte range would not be block-aligned within \
                 the fused per-layer tensor, which this cache's contiguous-slice addressing \
                 requires. This model's moe_intermediate_size/hidden_size combination is \
                 incompatible with the dequant-on-demand expert cache without a container \
                 format change (out of scope for Stage 1 — see PLAN.md §0).",
                path.display()
            ));
        }
        let per_expert_blocks = per_expert_elems / 32;
        let per_expert_bytes = per_expert_blocks * 20;

        // SAFETY: read-only mmap of a file this process does not mutate
        // while running (same invariant as `mmap_q4_weight` /
        // `load_q4_mmap_dequant_f16`).
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| format!("failed to mmap {}: {e}", path.display()))?;

        Ok(Self {
            mmap,
            payload_offset: header.payload_offset,
            per_expert_elems,
            per_expert_bytes,
            num_experts,
        })
    }

    fn expert_bytes(&self, expert_id: usize) -> Result<&[u8], String> {
        if expert_id >= self.num_experts {
            return Err(format!(
                "expert_id {expert_id} out of range (num_experts={})",
                self.num_experts
            ));
        }
        let start = self.payload_offset as usize + expert_id * self.per_expert_bytes;
        let end = start + self.per_expert_bytes;
        self.mmap.get(start..end).ok_or_else(|| {
            format!(
                "expert {expert_id} byte range {start}..{end} beyond mapped length {}",
                self.mmap.len()
            )
        })
    }

    /// Dequantize exactly one expert's Q4 blocks to f16 (same per-block
    /// scale/bias/nibble math as `load_q4_mmap_dequant_f16`, scoped to one
    /// expert's byte range instead of the whole tensor).
    fn dequant_expert_f16(&self, expert_id: usize) -> Result<Vec<u16>, String> {
        use crate::weights::q4_weights::{q4_f16_to_f32, q4_f32_to_f16};

        let bytes = self.expert_bytes(expert_id)?;
        let mut out: Vec<u16> = Vec::with_capacity(self.per_expert_elems);
        for chunk in bytes.chunks_exact(20) {
            let scale = q4_f16_to_f32(u16::from_ne_bytes([chunk[0], chunk[1]]));
            let bias = q4_f16_to_f32(u16::from_ne_bytes([chunk[2], chunk[3]]));
            for b in 0..16 {
                let byte_val = chunk[4 + b];
                out.push(q4_f32_to_f16((byte_val & 0x0f) as f32 * scale + bias));
                out.push(q4_f32_to_f16((byte_val >> 4) as f32 * scale + bias));
            }
        }
        out.truncate(self.per_expert_elems);
        Ok(out)
    }
}

/// One cache-miss dequant-and-copy task produced by
/// [`ExpertSlotCache::plan_prefetch`]: which expert to dequantize, and
/// which pre-assigned buffer slot to copy the result into. Every task in
/// one `plan_prefetch` call owns a distinct `slot` (planning already
/// removed it from every other task's consideration), so a batch of
/// `PrefetchTask`s is safe to execute in any order, or concurrently, with
/// zero coordination beyond "each task only touches its own `slot`".
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) struct PrefetchTask {
    pub(crate) slot: usize,
    pub(crate) expert_id: usize,
}

/// One dequantized expert's f16 data, tagged with the slot
/// [`ExpertSlotCache::plan_prefetch`] pre-assigned it to — the output of
/// [`ExpertSlotCache::spawn_dequant`]'s background phase and the input to
/// [`ExpertSlotCache::apply_prefetch_results`]'s single-threaded finish
/// phase.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
type PrefetchDequantResult = (usize, Vec<u16>);

/// A `spawn_dequant` call's full batch of [`PrefetchDequantResult`]s.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
type PrefetchDequantResults = Vec<PrefetchDequantResult>;

/// Test-only synchronization handles letting a test observe and control
/// [`ExpertSlotCache::spawn_dequant`]'s background dequant work: the
/// spawned closure sends on `started_tx` and then blocks on `release_rx`
/// before doing any real dequant work, so a test can deterministically
/// prove other code ran while the dequant phase was in flight (no sleeps —
/// pure channel handshake). Owned handles, not a `Clone` type: a test
/// builds one pair per token it wants to instrument and moves it in.
#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
pub(crate) struct PrefetchOrderingGate {
    pub(crate) started_tx: std::sync::mpsc::Sender<()>,
    pub(crate) release_rx: std::sync::mpsc::Receiver<()>,
}

/// See [`ExpertSlotCache::debug_snapshot`].
#[cfg(all(test, target_os = "macos", feature = "metal-gpu"))]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ExpertSlotCacheSnapshot {
    pub(crate) slot_owner: Vec<Option<usize>>,
    pub(crate) expert_to_slot: Vec<(usize, usize)>,
    pub(crate) slot_touched: Vec<bool>,
    pub(crate) slot_ready: Vec<bool>,
    pub(crate) lru: Vec<usize>,
    pub(crate) hit_miss_eviction: (usize, usize, usize),
}

/// Bounded pool of `N` pre-allocated per-expert f16 Metal buffer slots for
/// ONE (layer, gate_up|down) MoE tensor, LRU-evicted, backed by an mmap of
/// the tensor's `.q4` file. See the module doc comment for the GPU buffer
/// lifetime invariant this cache's eviction policy relies on.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) struct ExpertSlotCache {
    table: ExpertByteTable,
    slot_elems: usize,
    slots: Vec<metal::Buffer>,
    slot_owner: Vec<Option<usize>>,
    /// Set by `resolve()` for every slot it touches (hit or miss) since the
    /// last `begin_token()`; `pick_eviction_slot` only evicts `false` slots.
    slot_touched: Vec<bool>,
    /// `true` once a slot's Metal buffer actually holds the bytes for its
    /// current `slot_owner` — i.e. its dequant task ran to completion and
    /// [`Self::apply_prefetch_results`] (or `load_into`) copied the result
    /// in. `plan_prefetch`/`load_into` set a newly-assigned slot's flag to
    /// `false` the instant ownership is committed (before any dequant I/O
    /// runs); the apply step is the ONLY thing that flips it back to
    /// `true`. This closes the failure mode where a dequant task panics
    /// (or is otherwise never completed) after ownership bookkeeping is
    /// already committed: `get_prefetched` refuses to hand back an unready
    /// slot, and the next `plan_prefetch` call treats a still-unready
    /// mapping as a fresh miss (reusing the same slot, no eviction) rather
    /// than a hit — so a later token can never observe the stale/never-
    /// written bytes left behind by the failed task.
    slot_ready: Vec<bool>,
    /// Slot indices in LRU order, front = least-recently-used.
    lru: VecDeque<usize>,
    expert_to_slot: std::collections::HashMap<usize, usize>,
    label: String,
    /// Test-only hit/miss/eviction counters (zero-cost in non-test builds:
    /// the fields don't exist and every increment site is `#[cfg(test)]`).
    /// Lets a GPU test assert the *effective* number of misses/evictions a
    /// same-command-buffer `resolve()` sequence produced, rather than only
    /// inferring it indirectly from decode output.
    #[cfg(test)]
    hit_count: usize,
    #[cfg(test)]
    miss_count: usize,
    #[cfg(test)]
    eviction_count: usize,
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
impl ExpertSlotCache {
    /// Build a cache with `num_slots` pre-allocated (but not yet populated —
    /// this constructor does zero dequant work) slots for the tensor at
    /// `path`, validated against `expected_shape` (`[num_experts, ...]`).
    pub(crate) fn new(
        device: &metal::Device,
        path: &Path,
        expected_shape: &[usize],
        num_slots: usize,
        label: &str,
    ) -> Result<Self, String> {
        if num_slots == 0 {
            return Err(format!(
                "{label}: ExpertSlotCache requires at least 1 slot (got 0)"
            ));
        }
        let table = ExpertByteTable::open(path, expected_shape)?;
        let slot_elems = table.per_expert_elems;
        let byte_len = (slot_elems * std::mem::size_of::<u16>()) as u64;
        let slots: Vec<metal::Buffer> = (0..num_slots)
            .map(|i| {
                let buf = device.new_buffer(byte_len, metal::MTLResourceOptions::StorageModeShared);
                buf.set_label(&format!("{label}.slot{i}"));
                buf
            })
            .collect();
        Ok(Self {
            table,
            slot_elems,
            slots,
            slot_owner: vec![None; num_slots],
            slot_touched: vec![false; num_slots],
            slot_ready: vec![false; num_slots],
            lru: (0..num_slots).collect(),
            expert_to_slot: std::collections::HashMap::with_capacity(num_slots),
            label: label.to_string(),
            #[cfg(test)]
            hit_count: 0,
            #[cfg(test)]
            miss_count: 0,
            #[cfg(test)]
            eviction_count: 0,
        })
    }

    /// Effective number of resident slots this cache was actually built
    /// with — lets a test assert the real slot count a run exercised
    /// (rather than trusting that an env override wasn't silently ignored
    /// or reclamped elsewhere). Test-only: compiled out entirely in
    /// non-test builds.
    #[cfg(test)]
    pub(crate) fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Full bookkeeping snapshot for equivalence testing: does the
    /// `plan_prefetch`/`spawn_dequant`/`apply_prefetch_results` path
    /// produce EXACTLY the same ownership/touched/ready/LRU/counter state
    /// as an equivalent sequence of `resolve()` calls? `expert_to_slot` is
    /// captured as a sorted `Vec` (not the `HashMap` itself) so two
    /// snapshots compare equal regardless of hashing/iteration order.
    /// Test-only: compiled out entirely in non-test builds.
    #[cfg(test)]
    pub(crate) fn debug_snapshot(&self) -> ExpertSlotCacheSnapshot {
        let mut expert_to_slot: Vec<(usize, usize)> =
            self.expert_to_slot.iter().map(|(&e, &s)| (e, s)).collect();
        expert_to_slot.sort_unstable();
        ExpertSlotCacheSnapshot {
            slot_owner: self.slot_owner.clone(),
            expert_to_slot,
            slot_touched: self.slot_touched.clone(),
            slot_ready: self.slot_ready.clone(),
            lru: self.lru.iter().copied().collect(),
            hit_miss_eviction: (self.hit_count, self.miss_count, self.eviction_count),
        }
    }

    /// Raw f16 bit-pattern contents of `slot`, read directly back off the
    /// Metal `StorageModeShared` buffer — lets a test prove a recovered
    /// slot's bytes genuinely came from a fresh dequant rather than
    /// stale/uninitialized memory, by comparing against another cache's
    /// (or a direct `ExpertByteTable` dequant's) output for the same
    /// expert. Test-only: compiled out entirely in non-test builds.
    #[cfg(test)]
    pub(crate) fn slot_bits(&self, slot: usize) -> Vec<u16> {
        // SAFETY: `slots[slot]` is a StorageModeShared buffer sized
        // exactly `slot_elems * 2` bytes at construction, CPU-readable at
        // any time via `contents()` (no separate GPU-side synchronization
        // needed for StorageModeShared) — same buffer this cache's own
        // `apply_prefetch_results`/`load_into` read/write through.
        unsafe {
            let ptr = self.slots[slot].contents() as *const u16;
            std::slice::from_raw_parts(ptr, self.slot_elems).to_vec()
        }
    }

    /// Cumulative `(hits, misses, evictions)` since construction. Test-only:
    /// compiled out entirely in non-test builds.
    #[cfg(test)]
    pub(crate) fn hit_miss_eviction_counts(&self) -> (usize, usize, usize) {
        (self.hit_count, self.miss_count, self.eviction_count)
    }

    /// Clear the "touched this token" bookkeeping. Must be called once per
    /// decoded token, before the first `resolve()` call for that token — see
    /// the module doc comment for why this makes eviction safe.
    pub(crate) fn begin_token(&mut self) {
        self.slot_touched.iter_mut().for_each(|t| *t = false);
    }

    /// Resolve `expert_id` to its resident buffer slot, dequantizing on miss.
    /// Panics only on cache misconfiguration that construction-time
    /// validation (`moe_expert_cache_num_slots` enforcing `num_slots >=
    /// top_k`) should have already prevented — see `pick_eviction_slot`.
    ///
    /// Test-only as of #682 Stage 2: `encode_moe_ffn`'s production decode
    /// path now prefetches every selected expert up front via
    /// [`Self::prefetch_experts`] and looks slots up with
    /// [`Self::get_prefetched`], so this one-at-a-time resolve-on-miss
    /// entry point is exercised only by test helpers that drive the cache
    /// directly (e.g. forcing eviction pressure outside a real
    /// `encode_moe_ffn` call). Kept for that test surface, not dead code.
    #[cfg(test)]
    pub(crate) fn resolve(&mut self, expert_id: usize) -> &metal::Buffer {
        if let Some(&slot) = self.expert_to_slot.get(&expert_id) {
            if self.slot_ready[slot] {
                #[cfg(test)]
                {
                    self.hit_count += 1;
                }
                self.touch(slot);
                return &self.slots[slot];
            }
            // Same "committed but never populated" case `plan_prefetch`
            // handles — reload into the slot this expert already owns
            // rather than evicting anything.
            #[cfg(test)]
            {
                self.miss_count += 1;
            }
            self.load_into(slot, expert_id);
            return &self.slots[slot];
        }
        #[cfg(test)]
        {
            self.miss_count += 1;
        }
        let slot = self.pick_eviction_slot();
        self.load_into(slot, expert_id);
        &self.slots[slot]
    }

    /// Fetch the buffer for `expert_id`, which MUST already be resident —
    /// i.e. [`Self::prefetch_experts`] (or [`Self::plan_prefetch`] +
    /// applying its tasks) already ran for this token and covered
    /// `expert_id`. Unlike `resolve()`, this performs no hit/miss
    /// accounting, eviction, or LRU touch — that bookkeeping already
    /// happened during planning — so a caller that prefetches every
    /// expert it needs this token before dispatching (`encode_moe_ffn`'s
    /// Step 3) can look the buffer up without double-counting Stage 1's
    /// hit/miss/eviction counters or re-touching an already-touched slot.
    pub(crate) fn get_prefetched(&self, expert_id: usize) -> &metal::Buffer {
        let slot = self.expert_to_slot.get(&expert_id).unwrap_or_else(|| {
            panic!(
                "{}: get_prefetched({expert_id}) called without a prior prefetch_experts() (or \
                 plan_prefetch()) covering this expert this token — caller bug, not a data \
                 problem",
                self.label
            )
        });
        assert!(
            self.slot_ready[*slot],
            "{}: get_prefetched({expert_id}) found slot {slot} assigned but not yet populated \
             — its dequant task never completed (e.g. panicked) after plan_prefetch committed \
             ownership. This must never be reached in production: a caller that prefetched \
             every selected expert (and correctly propagated any dequant failure instead of \
             swallowing it) always applies results, or fails the whole token, before reaching \
             Step 3's lookups — a data/caller bug, not routine cache behavior",
            self.label
        );
        &self.slots[*slot]
    }

    /// Single-threaded planning pass: classify every id in `expert_ids` as
    /// a cache hit or miss (in order), touching hits' slots immediately
    /// (protecting them from eviction by a later miss in this same call)
    /// and, for each miss, committing its eviction-target slot's full
    /// bookkeeping (old-owner eviction, `expert_to_slot` insertion,
    /// touched flag, LRU position) right away — identical side effects to
    /// what calling [`Self::resolve`] once per id, in the same order,
    /// would produce. The only thing NOT done here is the actual
    /// I/O + dequant + buffer copy for misses: that is deferred to the
    /// returned [`PrefetchTask`]s, which [`Self::prefetch_experts`] may
    /// then run in parallel — see the module doc comment's Stage 2
    /// section for why deferring only that part is safe.
    ///
    /// `expert_ids` may contain the same id more than once (defensive,
    /// though `encode_moe_ffn`'s top-k selection never repeats one within
    /// a token) — a repeat is simply a hit against the slot the first
    /// occurrence just claimed — INCLUDING a repeat that was itself cold
    /// (a miss the first time this same call assigned it a fresh
    /// `PrefetchTask`): `planned_this_call` tracks every expert this call
    /// has already assigned a slot to (ready or not), so a second
    /// occurrence never pushes a second task for the same slot, matching
    /// what a second `resolve()` call for the same id would do (by the
    /// time a second `resolve()` runs, the first's `load_into` already
    /// completed synchronously and left the slot ready — a hit).
    pub(crate) fn plan_prefetch(&mut self, expert_ids: &[usize]) -> Vec<PrefetchTask> {
        let mut tasks = Vec::new();
        let mut planned_this_call: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for &expert_id in expert_ids {
            if let Some(&slot) = planned_this_call.get(&expert_id) {
                // Repeat of an id this SAME call already assigned a task
                // to (cold or not) — the task already covers it, so this
                // is a hit against that pending assignment: no second
                // task, no second slot mutation (the distinct-slot
                // contract every task in one `plan_prefetch` call relies
                // on — see `PrefetchTask`'s doc comment).
                #[cfg(test)]
                {
                    self.hit_count += 1;
                }
                self.touch(slot);
                continue;
            }
            if let Some(&slot) = self.expert_to_slot.get(&expert_id) {
                if self.slot_ready[slot] {
                    #[cfg(test)]
                    {
                        self.hit_count += 1;
                    }
                    self.touch(slot);
                    continue;
                }
                // Ownership is committed but the slot's previous dequant
                // task never completed (see `slot_ready`'s doc comment —
                // most likely it panicked). Treat this exactly like a
                // miss EXCEPT no eviction is needed: this expert already
                // owns the slot, so reload into the same slot rather than
                // picking a new one. Still counts as a miss (the data
                // genuinely is not resident) and still needs a fresh
                // `PrefetchTask`.
                #[cfg(test)]
                {
                    self.miss_count += 1;
                }
                self.slot_ready[slot] = false;
                self.touch(slot);
                planned_this_call.insert(expert_id, slot);
                tasks.push(PrefetchTask { slot, expert_id });
                continue;
            }
            #[cfg(test)]
            {
                self.miss_count += 1;
            }
            let slot = self.pick_eviction_slot();
            if let Some(old_owner) = self.slot_owner[slot].take() {
                self.expert_to_slot.remove(&old_owner);
                #[cfg(test)]
                {
                    self.eviction_count += 1;
                }
            }
            self.slot_owner[slot] = Some(expert_id);
            self.expert_to_slot.insert(expert_id, slot);
            self.slot_ready[slot] = false;
            self.touch(slot);
            planned_this_call.insert(expert_id, slot);
            tasks.push(PrefetchTask { slot, expert_id });
        }
        tasks
    }

    /// Spawn the dequant phase for an already-planned `tasks` list onto
    /// `scope`, so the caller can encode other GPU work on the calling
    /// thread while it runs and only join it later — see the module doc
    /// comment's Stage 2 section. Borrows nothing from `self` except its
    /// (immutable, plain-data) byte table and label: no `slots`
    /// (`metal::Buffer`s are not known to be safely shareable across
    /// threads) and no bookkeeping field crosses the thread boundary, so
    /// the returned handle's closure only ever performs read-only mmap
    /// access plus CPU dequant math. Returns `None` when `tasks` is empty
    /// (nothing to spawn, nothing to join).
    ///
    /// `parallel` controls only whether multiple tasks within THIS spawned
    /// closure fan out across rayon's pool or run on a plain serial
    /// iterator — it does not affect whether spawning itself happens,
    /// which this method always does (real cross-thread overlap with
    /// whatever the caller encodes next does not depend on how many
    /// threads the dequant phase itself uses).
    ///
    /// `panic_on_expert`, when `Some(id)`, deliberately panics before
    /// dequantizing expert `id` instead of doing the real work — test-only
    /// fault-injection hook for exercising the `slot_ready` recovery path;
    /// always `None` outside tests.
    pub(crate) fn spawn_dequant<'scope, 'env>(
        &'env self,
        tasks: Vec<PrefetchTask>,
        parallel: bool,
        panic_on_expert: Option<usize>,
        #[cfg(test)] ordering_gate: Option<PrefetchOrderingGate>,
        scope: &'scope std::thread::Scope<'scope, 'env>,
    ) -> Option<std::thread::ScopedJoinHandle<'scope, PrefetchDequantResults>> {
        if tasks.is_empty() {
            return None;
        }
        let table = &self.table;
        let label = self.label.as_str();
        Some(scope.spawn(move || {
            #[cfg(test)]
            if let Some(gate) = ordering_gate {
                let _ = gate.started_tx.send(());
                let _ = gate.release_rx.recv();
            }
            let dequant_one = |expert_id: usize| -> Vec<u16> {
                if panic_on_expert == Some(expert_id) {
                    panic!(
                        "{label}: test-injected dequant panic for expert {expert_id} — this \
                         message should never appear outside the readiness-recovery test that \
                         deliberately triggers it"
                    );
                }
                table.dequant_expert_f16(expert_id).unwrap_or_else(|e| {
                    panic!(
                        "{label}: failed to dequantize expert {expert_id} during prefetch \
                         (should be unreachable — expert_id is always < num_experts and the \
                         byte table was validated at construction): {e}"
                    )
                })
            };
            if parallel && tasks.len() > 1 {
                tasks
                    .par_iter()
                    .map(|t| (t.slot, dequant_one(t.expert_id)))
                    .collect()
            } else {
                tasks
                    .iter()
                    .map(|t| (t.slot, dequant_one(t.expert_id)))
                    .collect()
            }
        }))
    }

    /// Copy every `(slot, data)` dequant result into its pre-assigned
    /// slot's Metal buffer and mark that slot ready — the single-threaded
    /// "finish" half of the spawn/finish split `spawn_dequant` begins. Must
    /// be called with the results of a `spawn_dequant` join (or an
    /// equivalent synchronous dequant) before any `get_prefetched` lookup
    /// for the experts it covers.
    pub(crate) fn apply_prefetch_results(&mut self, results: &[PrefetchDequantResult]) {
        for (slot, data) in results {
            debug_assert_eq!(data.len(), self.slot_elems);
            // SAFETY: `plan_prefetch` already committed this slot's
            // ownership and touched it (protecting it from eviction by any
            // other task in this same plan) before `spawn_dequant` started,
            // and no GPU command referencing this slot's OLD contents can
            // still be in flight (module doc comment's cross-/within-token
            // invariant). Each `slot` value here is unique within
            // `results` (one per distinct `PrefetchTask`), so these writes
            // never alias each other either.
            unsafe {
                let dst = self.slots[*slot].contents() as *mut u16;
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
            self.slot_ready[*slot] = true;
        }
    }

    /// Convenience wrapper composing plan + spawn + immediate join + apply
    /// for callers that don't need real cross-phase overlap (test helpers,
    /// and anything driving the cache outside a real `encode_moe_ffn` call).
    /// `encode_moe_ffn` itself calls `plan_prefetch` + `spawn_dequant` +
    /// `apply_prefetch_results` directly so it can encode other GPU work
    /// between spawning and joining — see the module doc comment.
    #[cfg(test)]
    pub(crate) fn prefetch_experts(&mut self, expert_ids: &[usize], parallel: bool) {
        let tasks = self.plan_prefetch(expert_ids);
        if tasks.is_empty() {
            return;
        }
        let results = std::thread::scope(|scope| {
            let handle = self.spawn_dequant(tasks, parallel, None, None, scope);
            handle
                .expect("tasks is non-empty, spawn_dequant only returns None for empty tasks")
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e))
        });
        self.apply_prefetch_results(&results);
    }

    fn pick_eviction_slot(&mut self) -> usize {
        let pos = self
            .lru
            .iter()
            .position(|&s| !self.slot_touched[s])
            .unwrap_or_else(|| {
                panic!(
                    "{}: no untouched expert-cache slot available for eviction — this means \
                     more than num_slots={} distinct experts were resolved within a single \
                     token, which `moe_expert_cache_num_slots` should have prevented by \
                     enforcing num_slots >= top_k at construction time (cache misconfigured, \
                     not a data problem)",
                    self.label,
                    self.slots.len()
                )
            });
        self.lru.remove(pos).unwrap_or_else(|| {
            panic!(
                "{}: lru.remove({pos}) found nothing — `pos` was just returned by \
                 `lru.iter().position(..)` on this same `self.lru` with no mutation in \
                 between, so this is unreachable unless that invariant breaks (cache \
                 internally corrupted, not a data problem)",
                self.label
            )
        })
    }

    /// Test-only (see `resolve`'s doc comment): the production path's
    /// equivalent bookkeeping+copy is inlined across `plan_prefetch` +
    /// `prefetch_experts` instead, so their dequant work can be deferred
    /// and fanned out.
    #[cfg(test)]
    fn load_into(&mut self, slot: usize, expert_id: usize) {
        if let Some(old_owner) = self.slot_owner[slot].take() {
            // A same-owner reload (the "committed but never populated"
            // case `resolve`'s caller already detected) is not an
            // eviction — nothing outside this expert's own data is being
            // discarded, so only count it when the slot is genuinely
            // changing hands.
            if old_owner != expert_id {
                self.expert_to_slot.remove(&old_owner);
                #[cfg(test)]
                {
                    self.eviction_count += 1;
                }
            }
        }
        let f16_data = self
            .table
            .dequant_expert_f16(expert_id)
            .unwrap_or_else(|e| {
                panic!(
                    "{}: failed to dequantize expert {expert_id} (should be unreachable — \
                 expert_id is always < num_experts and the byte table was validated at \
                 construction): {e}",
                    self.label
                )
            });
        debug_assert_eq!(f16_data.len(), self.slot_elems);
        // SAFETY: `slots[slot]` is a StorageModeShared buffer sized exactly
        // `slot_elems * 2` bytes at construction (`ExpertSlotCache::new`);
        // no GPU command currently in flight can read it — see the module
        // doc comment's GPU buffer lifetime invariant (cross-token safety
        // via `wait_until_completed()` before the next token starts,
        // within-token safety via `pick_eviction_slot` refusing touched
        // slots).
        unsafe {
            let dst = self.slots[slot].contents() as *mut u16;
            std::ptr::copy_nonoverlapping(f16_data.as_ptr(), dst, f16_data.len());
        }
        self.slot_owner[slot] = Some(expert_id);
        self.expert_to_slot.insert(expert_id, slot);
        self.slot_ready[slot] = true;
        self.touch(slot);
    }

    fn touch(&mut self, slot: usize) {
        self.slot_touched[slot] = true;
        if let Some(pos) = self.lru.iter().position(|&s| s == slot) {
            self.lru.remove(pos);
        }
        self.lru.push_back(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(num_slots: Option<usize>) -> MoeExpertCacheConfig {
        MoeExpertCacheConfig { num_slots }
    }

    #[test]
    fn env_override_clamped_to_top_k_and_num_experts_when_budget_is_not_binding() {
        // Generous working set (400 GB) so the budget-derived maximum for
        // this shape is >= num_experts (256) and num_experts is the actual
        // binding upper bound — the case the original clamp-only-to-
        // num_experts logic was designed for.
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(1)), 256, 8, 6_291_456, 40, 400_000_000_000)
                .unwrap(),
            8,
            "below top_k clamps up"
        );
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(9999)), 256, 8, 6_291_456, 40, 400_000_000_000)
                .unwrap(),
            256,
            "above num_experts clamps down to num_experts when the budget has headroom to spare"
        );
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(64)), 256, 8, 6_291_456, 40, 400_000_000_000)
                .unwrap(),
            64,
            "in-range passes through unchanged"
        );
    }

    #[test]
    fn env_override_capped_at_working_set_budget_even_when_below_num_experts() {
        // Same 256-expert / top_k=8 / 40-layer / 6 MiB-per-expert shape as
        // above, but on a tight 40 GB device budget: the budget-derived
        // maximum here is 135 slots — well below num_experts (256). This is
        // the exact scenario the Major finding covers: a large override
        // (or, unbounded, a fully-resident cache) must NOT be able to
        // exceed what the device can actually hold, even though 135 and
        // 9999 both clamp to *something* under the old `[top_k,
        // num_experts]`-only bound.
        let budget_max = 135;
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(9999)), 256, 8, 6_291_456, 40, 40_000_000_000)
                .unwrap(),
            budget_max,
            "an override far above num_experts must cap at the working-set budget, not num_experts"
        );
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(200)), 256, 8, 6_291_456, 40, 40_000_000_000)
                .unwrap(),
            budget_max,
            "an override below num_experts but above the budget must still be capped at the budget"
        );
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(100)), 256, 8, 6_291_456, 40, 40_000_000_000)
                .unwrap(),
            100,
            "an override under the budget passes through unchanged"
        );
    }

    #[test]
    fn default_zero_eviction_fast_path_when_everything_fits() {
        // Small synthetic-test-sized shapes: everything should fit easily,
        // defaulting to num_experts (zero-eviction fast path).
        let n = moe_expert_cache_num_slots(&cfg(None), 4, 1, 4096, 1, 8_000_000_000).unwrap();
        assert_eq!(n, 4);
    }

    #[test]
    fn auto_shrinks_under_device_budget() {
        // Qwen3.5-35B-A3B-shaped numbers: 256 experts, top_k=8, 40 MoE
        // layers, ~6 MiB/expert (gate_up+down combined, f16), on a 32 GiB
        // device (`recommended_max_working_set_size` conservatively modeled
        // as ~28 GiB here, mirroring real macOS headroom below total RAM).
        let per_expert_bytes = 2 * 512 * 2048 * 2 + 2048 * 512 * 2; // gate_up + down, f16
        let n = moe_expert_cache_num_slots(
            &cfg(None),
            256,
            8,
            per_expert_bytes as u64,
            40,
            28 * 1024 * 1024 * 1024,
        )
        .unwrap();
        assert!(
            n < 256,
            "expected auto-shrink below num_experts=256 on a 32 GiB-class device, got {n}"
        );
        assert!(n >= 8, "must never shrink below top_k=8, got {n}");
    }

    #[test]
    fn errors_when_even_top_k_does_not_fit() {
        // Absurdly tiny device budget: even 1 slot (top_k=1) can't fit.
        let err =
            moe_expert_cache_num_slots(&cfg(None), 4, 1, 10_000_000_000, 1, 1_000_000).unwrap_err();
        assert!(
            err.contains("cannot fit even the lazy dequant-on-demand cache"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn rejects_zero_num_experts_or_top_k() {
        assert!(moe_expert_cache_num_slots(&cfg(None), 0, 1, 100, 1, 1_000_000_000).is_err());
        assert!(moe_expert_cache_num_slots(&cfg(None), 4, 0, 100, 1, 1_000_000_000).is_err());
    }

    #[test]
    fn rejects_top_k_exceeding_num_experts() {
        assert!(moe_expert_cache_num_slots(&cfg(None), 4, 5, 100, 1, 1_000_000_000).is_err());
    }

    /// Serializes every test below that mutates the real process
    /// environment (`LATTICE_MOE_EXPERT_CACHE_SLOTS` is process-global —
    /// `cargo test` runs this file's tests on multiple threads within one
    /// process, so unguarded concurrent `set_var`/`remove_var` calls would
    /// race and flake).
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Sets (or clears) `LATTICE_MOE_EXPERT_CACHE_SLOTS` for the duration of
    /// `f`, holding `ENV_TEST_LOCK` and restoring the prior value
    /// (including "was unset") on the way out, even if `f` panics.
    fn with_env_var<R>(value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior = std::env::var(MOE_EXPERT_CACHE_SLOTS_ENV).ok();
        // SAFETY: serialized by `ENV_TEST_LOCK` above — no concurrent
        // reader/writer of this process-global variable.
        unsafe {
            match value {
                Some(v) => std::env::set_var(MOE_EXPERT_CACHE_SLOTS_ENV, v),
                None => std::env::remove_var(MOE_EXPERT_CACHE_SLOTS_ENV),
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        // SAFETY: see above.
        unsafe {
            match &prior {
                Some(v) => std::env::set_var(MOE_EXPERT_CACHE_SLOTS_ENV, v),
                None => std::env::remove_var(MOE_EXPERT_CACHE_SLOTS_ENV),
            }
        }
        match result {
            Ok(r) => r,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[test]
    fn from_env_unset_is_none() {
        with_env_var(None, || {
            assert_eq!(MoeExpertCacheConfig::from_env().unwrap().num_slots, None);
        });
    }

    #[test]
    fn from_env_valid_positive_integer_is_some() {
        with_env_var(Some("42"), || {
            assert_eq!(
                MoeExpertCacheConfig::from_env().unwrap().num_slots,
                Some(42)
            );
        });
        with_env_var(Some("  7  "), || {
            assert_eq!(MoeExpertCacheConfig::from_env().unwrap().num_slots, Some(7));
        });
    }

    #[test]
    fn from_env_garbage_errors_loudly_instead_of_silently_falling_back() {
        with_env_var(Some("not-a-number"), || {
            let err = MoeExpertCacheConfig::from_env().unwrap_err();
            assert!(
                err.contains("not a valid positive integer"),
                "unexpected error message: {err}"
            );
        });
        with_env_var(Some("-5"), || {
            assert!(MoeExpertCacheConfig::from_env().is_err());
        });
        with_env_var(Some(""), || {
            assert!(MoeExpertCacheConfig::from_env().is_err());
        });
    }

    #[test]
    fn from_env_zero_errors_loudly() {
        with_env_var(Some("0"), || {
            let err = MoeExpertCacheConfig::from_env().unwrap_err();
            assert!(err.contains("is 0"), "unexpected error message: {err}");
        });
    }
}
