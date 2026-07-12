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
//! tokens' GPU work (Stage 2 — prefetching a future token's experts while
//! the current token's command buffer is still in flight — would need to
//! revisit this invariant; it is explicitly out of scope here).

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
    /// Read [`MOE_EXPERT_CACHE_SLOTS_ENV`] (unset/unparseable → `None`,
    /// which falls back to the device-budget default).
    pub fn from_env() -> Self {
        let num_slots = std::env::var(MOE_EXPERT_CACHE_SLOTS_ENV)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok());
        Self { num_slots }
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

    if let Some(n) = cfg.num_slots {
        return Ok(n.clamp(top_k, num_experts));
    }

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
    Ok(affordable.clamp(top_k, num_experts))
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
        use crate::weights::q4_weights::{read_q4_header, validate_q4_header_payload_bounds};

        let file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let header = read_q4_header(&file)
            .map_err(|e| format!("failed to parse Q4 header {}: {e}", path.display()))?;
        if header.shape != expected_shape {
            return Err(format!(
                "{}: MoE expert-cache tensor has shape {:?}, expected {expected_shape:?} — \
                 refusing to build a lazy per-expert byte table over a mismatched/transposed \
                 layout (same hazard the eager `load_q4_mmap_dequant_f16` path guards against)",
                path.display(),
                header.shape
            ));
        }
        let file_len = file
            .metadata()
            .map_err(|e| format!("failed to stat {}: {e}", path.display()))?
            .len();
        validate_q4_header_payload_bounds(&header, file_len, path)
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
    /// Slot indices in LRU order, front = least-recently-used.
    lru: VecDeque<usize>,
    expert_to_slot: std::collections::HashMap<usize, usize>,
    label: String,
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
            lru: (0..num_slots).collect(),
            expert_to_slot: std::collections::HashMap::with_capacity(num_slots),
            label: label.to_string(),
        })
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
    pub(crate) fn resolve(&mut self, expert_id: usize) -> &metal::Buffer {
        if let Some(&slot) = self.expert_to_slot.get(&expert_id) {
            self.touch(slot);
            return &self.slots[slot];
        }
        let slot = self.pick_eviction_slot();
        self.load_into(slot, expert_id);
        &self.slots[slot]
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
        self.lru.remove(pos).expect("position() found it")
    }

    fn load_into(&mut self, slot: usize, expert_id: usize) {
        if let Some(old_owner) = self.slot_owner[slot].take() {
            self.expert_to_slot.remove(&old_owner);
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
    fn env_override_clamped_to_top_k_and_num_experts() {
        // Below top_k clamps up.
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(1)), 256, 8, 6_291_456, 40, 40_000_000_000)
                .unwrap(),
            8
        );
        // Above num_experts clamps down.
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(9999)), 256, 8, 6_291_456, 40, 40_000_000_000)
                .unwrap(),
            256
        );
        // In-range passes through unchanged.
        assert_eq!(
            moe_expert_cache_num_slots(&cfg(Some(64)), 256, 8, 6_291_456, 40, 40_000_000_000)
                .unwrap(),
            64
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
}
