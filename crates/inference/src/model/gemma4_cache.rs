//! Gemma 4 shared-KV cache: donor-slot indirection (ADR-082 Amendment 1, stage 4).
//!
//! Gemma 4's decoder functionally shares K/V state across its trailing
//! `num_kv_shared_layers` layers: each such layer never runs its own
//! `k_proj`/`v_proj`, and instead consumes the current-pass K/V published by
//! the *donor* -- the last non-shared layer of the same attention type
//! (sliding or global). This cache allocates a storage slot only for the
//! non-shared layers and resolves every layer (shared or not) to a slot via
//! a build-time indirection map, so KV memory scales with the non-shared
//! layer count rather than the full layer count.
//!
//! Donor slots are ordinary slots, sized per their own attention type.
//! Global-type slots retain the full context. Sliding-type slots implement
//! the *publish* rule that a differential probe against the pinned reference
//! established (a naive "read view == window" model does not match the
//! reference during prefill): on each append, the slot first compacts its
//! existing content down to the last `sliding_window - 1` positions, then
//! appends the entire incoming chunk; the read view is the whole resulting
//! buffer. Truncation to `sliding_window - 1` therefore happens lazily, at
//! the *next* append, not at read time -- a prefill chunk publishes its full
//! length, and only a subsequent append trims the prior tail. Repeated
//! single-token decode steps converge on exactly `sliding_window` published
//! positions (`sliding_window - 1` retained + 1 new). Shared layers read
//! their donor's slot directly -- there is no side buffer and no copy, and
//! the public API has no writer for a shared layer's "own" state, since it
//! has none.
//!
//! Storage is f32 (not the f16 half-precision used by
//! [`crate::kv_cache::flat::FlatKVCache`]): this stage targets CPU-only
//! correctness and the read path returns borrowed slices directly to
//! callers, which a quantized backing store would require dequantizing into
//! a scratch buffer on every read. f16 packing can be layered on later
//! without changing this module's public shape.
use crate::error::InferenceError;
use crate::model::gemma4_config::Gemma4Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotKind {
    /// Lazy window-minus-one retention published in full per chunk -- see
    /// the module docs for the exact rule.
    Sliding,
    /// Full-context retention up to `capacity` positions.
    Global,
}

#[derive(Debug)]
struct Slot {
    kind: SlotKind,
    /// Per-position width: `num_key_value_heads * attn_head_dim(owner_layer)`.
    kv_dim: usize,
    /// Sliding: `sliding_window` (the retention bound used in the append
    /// rule, not a hard cap on buffer size). Global: the cache's
    /// `max_seq_len` (a hard cap -- `append` rejects an overflow).
    capacity: usize,
    /// Global: fixed size `[capacity * kv_dim]`. Sliding: grows on demand to
    /// hold the retained tail plus the current chunk (transient, per-chunk
    /// -- see [`Slot::append`]). Valid data always occupies
    /// `[0, seq_len * kv_dim)`.
    k: Vec<f32>,
    v: Vec<f32>,
    seq_len: usize,
}

impl Slot {
    fn try_new(kind: SlotKind, kv_dim: usize, capacity: usize) -> Result<Self, InferenceError> {
        if kv_dim == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: slot kv_dim must be > 0".to_string(),
            ));
        }
        if capacity == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: slot capacity must be > 0".to_string(),
            ));
        }
        let elems = capacity.checked_mul(kv_dim).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "gemma4 kv cache: capacity ({capacity}) * kv_dim ({kv_dim}) overflows usize"
            ))
        })?;
        Ok(Self {
            kind,
            kv_dim,
            capacity,
            k: vec![0.0; elems],
            v: vec![0.0; elems],
            seq_len: 0,
        })
    }

    /// Append a chunk of `q_len = k.len() / kv_dim` tokens (`q_len >= 1`).
    ///
    /// Global slots reject an append that would exceed `capacity`.
    ///
    /// Sliding slots implement the measured publish rule (differentially
    /// probed against the pinned reference, see the module docs): first
    /// compact any existing content down to its last `capacity - 1`
    /// positions, then append the entire incoming chunk. The read view
    /// after this call is the whole resulting buffer -- a prefill chunk
    /// therefore publishes its full length, and window retention to
    /// `capacity - 1` is applied lazily, at the *next* append. The backing
    /// buffer grows as needed to hold `capacity - 1 + q_len` positions.
    fn append(&mut self, k: &[f32], v: &[f32]) -> Result<(), InferenceError> {
        if k.len() != v.len() {
            return Err(InferenceError::InvalidInput(format!(
                "gemma4 kv cache: k length {} != v length {}",
                k.len(),
                v.len()
            )));
        }
        if !k.len().is_multiple_of(self.kv_dim) {
            return Err(InferenceError::InvalidInput(format!(
                "gemma4 kv cache: k/v length {} is not a multiple of kv_dim {}",
                k.len(),
                self.kv_dim
            )));
        }
        let q_len = k.len() / self.kv_dim;
        if q_len == 0 {
            return Ok(());
        }

        match self.kind {
            SlotKind::Global => {
                let new_len = self.seq_len.checked_add(q_len).ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "gemma4 kv cache: global slot seq_len overflow".to_string(),
                    )
                })?;
                if new_len > self.capacity {
                    return Err(InferenceError::InvalidInput(format!(
                        "gemma4 kv cache: global slot capacity {} exceeded (seq_len={}, \
                         appending {q_len} tokens)",
                        self.capacity, self.seq_len
                    )));
                }
                let off = self.seq_len * self.kv_dim;
                self.k[off..off + k.len()].copy_from_slice(k);
                self.v[off..off + v.len()].copy_from_slice(v);
                self.seq_len = new_len;
            }
            SlotKind::Sliding => {
                // Compact existing content to its last `capacity - 1`
                // positions before publishing the incoming chunk.
                let retain = self.seq_len.min(self.capacity.saturating_sub(1));
                if retain < self.seq_len {
                    let drop = self.seq_len - retain;
                    let valid_end = self.seq_len * self.kv_dim;
                    self.k.copy_within((drop * self.kv_dim)..valid_end, 0);
                    self.v.copy_within((drop * self.kv_dim)..valid_end, 0);
                }
                self.seq_len = retain;

                let new_len = self.seq_len.checked_add(q_len).ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "gemma4 kv cache: sliding slot seq_len overflow".to_string(),
                    )
                })?;
                let needed_elems = new_len.checked_mul(self.kv_dim).ok_or_else(|| {
                    InferenceError::InvalidInput(
                        "gemma4 kv cache: sliding slot buffer size overflow".to_string(),
                    )
                })?;
                if self.k.len() < needed_elems {
                    self.k.resize(needed_elems, 0.0);
                    self.v.resize(needed_elems, 0.0);
                }

                let off = self.seq_len * self.kv_dim;
                self.k[off..off + k.len()].copy_from_slice(k);
                self.v[off..off + v.len()].copy_from_slice(v);
                self.seq_len = new_len;
            }
        }
        Ok(())
    }
}

/// **Unstable**: Gemma 4 shared-KV cache (ADR-082 Amendment 1, stage 4).
///
/// See the module docs for the donor-slot design. Construct via [`Self::new`]
/// against a validated (or test-constructed) [`Gemma4Config`]; write only
/// non-shared layers via [`Self::append_kv`]; read any layer -- shared or
/// not -- via [`Self::k_view`] / [`Self::v_view`].
#[derive(Debug)]
pub struct Gemma4KvCache {
    /// `layer_kv_slot[layer]` = slot index that layer reads from. For a
    /// non-shared layer this is its own (unique) slot; for a shared layer
    /// this is its donor's slot. Length = `num_hidden_layers`.
    layer_kv_slot: Vec<usize>,
    /// `is_shared[layer]` = whether `layer` is KV-shared (has no writable
    /// slot of its own). Length = `num_hidden_layers`.
    is_shared: Vec<bool>,
    slots: Vec<Slot>,
}

impl Gemma4KvCache {
    /// Build the donor-slot indirection map from `cfg` and allocate one
    /// storage slot per non-shared layer, sized `max_seq_len` for global
    /// slots and `cfg.sliding_window` for sliding slots.
    ///
    /// # Errors
    /// Returns `InferenceError` (never panics) if:
    /// - `cfg.num_hidden_layers` is `0`;
    /// - `cfg.layer_types.len()` does not equal `cfg.num_hidden_layers` (a
    ///   short `layer_types` would otherwise silently classify the missing
    ///   trailing entries as sliding);
    /// - `cfg.num_key_value_heads`, `cfg.head_dim`, or `cfg.global_head_dim`
    ///   is `0`;
    /// - `cfg.sliding_window` is `0` while at least one layer is a sliding
    ///   layer, or `max_seq_len` is `0`;
    /// - `cfg.num_key_value_heads * cfg.attn_head_dim(layer)` overflows
    ///   `usize` for any non-shared layer;
    /// - any shared layer has no non-shared donor layer of the same
    ///   attention type among the non-shared layers (a config/checkpoint
    ///   structural disagreement -- there is no valid donor to fall back
    ///   to, so this is rejected rather than silently paired cross-type);
    /// - any layer's resolved donor disagrees on attention type, or the
    ///   completed slot count disagrees with the independently-filtered
    ///   non-shared layer count (both are internal build-time asserts:
    ///   they should be unreachable given the construction above, but are
    ///   checked explicitly per the "hard error, not debug_assert"
    ///   contract);
    /// - a slot's `capacity * kv_dim` product overflows `usize`.
    pub fn new(cfg: &Gemma4Config, max_seq_len: usize) -> Result<Self, InferenceError> {
        let n = cfg.num_hidden_layers;
        if n == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: num_hidden_layers must be > 0".to_string(),
            ));
        }
        // Structural validation happens before any allocation: a
        // directly-constructed (not `Gemma4Config::validate`-checked) config
        // must fail closed rather than silently misclassify or overflow.
        if cfg.layer_types.len() != n {
            return Err(InferenceError::InvalidInput(format!(
                "gemma4 kv cache: layer_types has {} entries but num_hidden_layers is {n}",
                cfg.layer_types.len()
            )));
        }
        if cfg.num_key_value_heads == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: num_key_value_heads must be > 0".to_string(),
            ));
        }
        if cfg.head_dim == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: head_dim must be > 0".to_string(),
            ));
        }
        if cfg.global_head_dim == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: global_head_dim must be > 0".to_string(),
            ));
        }
        if max_seq_len == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: max_seq_len must be > 0".to_string(),
            ));
        }
        let any_sliding_layer = (0..n).any(|l| !cfg.is_global_layer(l));
        if any_sliding_layer && cfg.sliding_window == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: sliding_window must be > 0 when any layer is a sliding layer"
                    .to_string(),
            ));
        }

        let first_shared = cfg.first_kv_shared_layer_idx();

        // Non-shared layers 0..first_shared each get their own slot, in
        // order, so slot index == layer index for every non-shared layer.
        let mut layer_kv_slot = vec![0usize; n];
        for (layer, slot) in layer_kv_slot.iter_mut().enumerate().take(first_shared) {
            *slot = layer;
        }

        // Independently-filtered owner list, used below for the
        // post-allocation slot-count assert (not a restatement of the
        // formula that produced `first_shared`).
        let owner_layers: Vec<usize> = (0..n).filter(|&l| !cfg.is_kv_shared_layer(l)).collect();

        // Last non-shared donor layer seen per attention type, scanning the
        // non-shared prefix in layer order.
        let mut last_sliding_donor: Option<usize> = None;
        let mut last_global_donor: Option<usize> = None;
        for layer in 0..first_shared {
            if cfg.is_global_layer(layer) {
                last_global_donor = Some(layer);
            } else {
                last_sliding_donor = Some(layer);
            }
        }

        for layer in first_shared..n {
            let donor = if cfg.is_global_layer(layer) {
                last_global_donor.ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "gemma4 kv cache: shared global layer {layer} has no non-shared \
                         global-attention donor layer among layers 0..{first_shared}"
                    ))
                })?
            } else {
                last_sliding_donor.ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "gemma4 kv cache: shared sliding layer {layer} has no non-shared \
                         sliding-attention donor layer among layers 0..{first_shared}"
                    ))
                })?
            };
            layer_kv_slot[layer] = donor;
        }

        // Build-time assert: every layer's resolved donor agrees on
        // attention type (non-shared layers trivially agree with
        // themselves; this is the substantive check for shared layers).
        for layer in 0..n {
            let donor = layer_kv_slot[layer];
            if cfg.is_global_layer(layer) != cfg.is_global_layer(donor) {
                return Err(InferenceError::Inference(format!(
                    "gemma4 kv cache: build-time same-type assert failed: layer {layer} \
                     (global={}) maps to a slot owned by donor layer {donor} (global={})",
                    cfg.is_global_layer(layer),
                    cfg.is_global_layer(donor)
                )));
            }
        }

        let is_shared: Vec<bool> = (0..n).map(|l| cfg.is_kv_shared_layer(l)).collect();

        let mut slots = Vec::with_capacity(first_shared);
        for owner_layer in 0..first_shared {
            let kv_dim = cfg
                .num_key_value_heads
                .checked_mul(cfg.attn_head_dim(owner_layer))
                .ok_or_else(|| {
                    InferenceError::InvalidInput(format!(
                        "gemma4 kv cache: num_key_value_heads ({}) * attn_head_dim ({}) \
                         overflows usize for layer {owner_layer}",
                        cfg.num_key_value_heads,
                        cfg.attn_head_dim(owner_layer)
                    ))
                })?;
            let (kind, capacity) = if cfg.is_global_layer(owner_layer) {
                (SlotKind::Global, max_seq_len)
            } else {
                (SlotKind::Sliding, cfg.sliding_window)
            };
            slots.push(Slot::try_new(kind, kv_dim, capacity)?);
        }

        // Build-time assert: the actually-completed slot count matches the
        // independently-filtered owner list, not a restatement of the
        // formula that produced the `0..first_shared` allocation loop bound.
        if slots.len() != owner_layers.len() {
            return Err(InferenceError::Inference(format!(
                "gemma4 kv cache: build-time slot-count assert failed: allocated {} slots but \
                 the independently-filtered non-shared layer count is {} (num_hidden_layers={n})",
                slots.len(),
                owner_layers.len()
            )));
        }

        Ok(Self {
            layer_kv_slot,
            is_shared,
            slots,
        })
    }

    /// Number of storage slots (== number of non-shared layers).
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Slot index that `layer` resolves to (its own slot if non-shared,
    /// otherwise its donor's slot).
    pub fn layer_slot(&self, layer: usize) -> Result<usize, InferenceError> {
        self.layer_kv_slot.get(layer).copied().ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "gemma4 kv cache: layer index {layer} out of bounds (num_hidden_layers={})",
                self.layer_kv_slot.len()
            ))
        })
    }

    /// Whether `layer` is a KV-shared layer (read-only: no `append_kv` path).
    pub fn is_kv_shared_layer(&self, layer: usize) -> Result<bool, InferenceError> {
        self.is_shared.get(layer).copied().ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "gemma4 kv cache: layer index {layer} out of bounds (num_hidden_layers={})",
                self.is_shared.len()
            ))
        })
    }

    /// Append a chunk of `k.len() / kv_dim` tokens to `layer`'s own slot.
    ///
    /// Sequential-layer-order contract: within one chunked-prefill call
    /// sequence, callers must invoke this in increasing layer order so that
    /// a shared layer's subsequent [`Self::k_view`] / [`Self::v_view`] read
    /// (in the same chunk) observes its donor's just-appended state -- see
    /// the module docs and the `chunk_boundary_shared_layer_sees_current_chunk`
    /// test.
    ///
    /// # Errors
    /// Returns `InferenceError::InvalidInput` if `layer` is out of bounds,
    /// `layer` is KV-shared (shared layers have no writable slot -- they
    /// have no `k_proj`/`v_proj` of their own to write), `k`/`v` lengths
    /// disagree, their common length is not a multiple of the slot's
    /// `kv_dim`, or the append would exceed a global slot's capacity.
    pub fn append_kv(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<(), InferenceError> {
        if layer >= self.layer_kv_slot.len() {
            return Err(InferenceError::InvalidInput(format!(
                "gemma4 kv cache: layer index {layer} out of bounds (num_hidden_layers={})",
                self.layer_kv_slot.len()
            )));
        }
        if self.is_shared[layer] {
            return Err(InferenceError::InvalidInput(format!(
                "gemma4 kv cache: layer {layer} is a KV-shared layer and has no writable slot -- \
                 shared layers read their donor's slot via k_view/v_view and never append"
            )));
        }
        let slot_idx = self.layer_kv_slot[layer];
        self.slots[slot_idx].append(k, v)
    }

    /// Current valid sequence length of the slot `layer` resolves to (its
    /// own slot if non-shared, its donor's if shared).
    pub fn seq_len(&self, layer: usize) -> Result<usize, InferenceError> {
        Ok(self.slot_for_layer(layer)?.seq_len)
    }

    /// Read-only view of `layer`'s resolved K slot, `[seq_len * kv_dim]`,
    /// in logical (oldest-to-newest) order. For a shared layer this is
    /// exactly its donor's slot content.
    pub fn k_view(&self, layer: usize) -> Result<&[f32], InferenceError> {
        let slot = self.slot_for_layer(layer)?;
        Ok(&slot.k[..slot.seq_len * slot.kv_dim])
    }

    /// Read-only view of `layer`'s resolved V slot; see [`Self::k_view`].
    pub fn v_view(&self, layer: usize) -> Result<&[f32], InferenceError> {
        let slot = self.slot_for_layer(layer)?;
        Ok(&slot.v[..slot.seq_len * slot.kv_dim])
    }

    fn slot_for_layer(&self, layer: usize) -> Result<&Slot, InferenceError> {
        let slot_idx = self.layer_slot(layer)?;
        Ok(&self.slots[slot_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::gemma4_config::Gemma4LayerType;
    use std::path::PathBuf;

    fn pinned_e2b_config() -> Gemma4Config {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("gemma4")
            .join("e2b_config.json");
        let json = std::fs::read_to_string(path).expect("read committed e2b_config.json fixture");
        Gemma4Config::from_config_json_str(&json).expect("pinned e2b fixture must parse")
    }

    /// A small config for cheap unit tests. 6 layers, types
    /// [S, F, S, S, F, S], num_kv_shared_layers=2 ->
    /// first_kv_shared_layer_idx=4. Non-shared prefix layers 0..4 =
    /// [S, F, S, S]: sliding donor = layer 3 (last S), global donor =
    /// layer 1 (last F). Shared layers: 4 (F) -> donor 1; 5 (S) -> donor 3.
    fn tiny_config(sliding_window: usize) -> Gemma4Config {
        let layer_types = vec![
            Gemma4LayerType::SlidingAttention, // 0
            Gemma4LayerType::FullAttention,    // 1 (global donor)
            Gemma4LayerType::SlidingAttention, // 2
            Gemma4LayerType::SlidingAttention, // 3 (sliding donor)
            Gemma4LayerType::FullAttention,    // 4 (shared, global)
            Gemma4LayerType::SlidingAttention, // 5 (shared, sliding)
        ];
        Gemma4Config {
            hidden_size: 8,
            num_hidden_layers: 6,
            vocab_size: 32,
            intermediate_size: 16,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            global_head_dim: 8,
            sliding_window,
            attention_k_eq_v: false,
            attention_bias: false,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.5,
            layer_types,
            num_kv_shared_layers: 2,
            use_double_wide_mlp_raw: true,
            hidden_size_per_layer_input: 4,
            hidden_activation: "gelu_pytorch_tanh".to_string(),
            final_logit_softcapping: 30.0,
            tie_word_embeddings: true,
            eos_token_id: 1,
            max_position_embeddings: 4096,
        }
    }

    // -- Test 1: slot-map correctness on the real e2b fixture ---------------

    #[test]
    fn e2b_slot_map_matches_amendment_1() {
        let cfg = pinned_e2b_config();
        let cache = Gemma4KvCache::new(&cfg, 4096).expect("e2b config must build a valid cache");

        assert_eq!(cache.num_slots(), 15);
        assert_eq!(cfg.first_kv_shared_layer_idx(), 15);

        for layer in 0..15 {
            assert_eq!(cache.layer_slot(layer).unwrap(), layer);
            assert!(!cache.is_kv_shared_layer(layer).unwrap());
        }
        for layer in 15..35 {
            let slot = cache.layer_slot(layer).unwrap();
            let expected = if cfg.is_global_layer(layer) { 14 } else { 13 };
            assert_eq!(
                slot,
                expected,
                "layer {layer} (global={}) should map to donor slot {expected}",
                cfg.is_global_layer(layer)
            );
            assert_eq!(
                cfg.is_global_layer(layer),
                cfg.is_global_layer(slot),
                "layer {layer} and its donor slot {slot} must agree on attention type"
            );
            assert!(cache.is_kv_shared_layer(layer).unwrap());
        }
    }

    // -- Test 2: shared-layer write rejected ---------------------------------

    #[test]
    fn shared_layer_write_rejected_with_error() {
        let cfg = tiny_config(8);
        let mut cache = Gemma4KvCache::new(&cfg, 64).unwrap();
        let kv_dim_global = cfg.num_key_value_heads * cfg.global_head_dim;
        let k = vec![1.0f32; kv_dim_global];
        let v = vec![2.0f32; kv_dim_global];

        let err = cache
            .append_kv(4, &k, &v)
            .expect_err("layer 4 is KV-shared and must reject a write");
        assert!(
            err.to_string().contains("KV-shared") || err.to_string().contains("shared"),
            "error should describe the shared-layer rejection: {err}"
        );

        // Non-shared layer 1 (the donor layer 4 maps to) accepts the write.
        cache
            .append_kv(1, &k, &v)
            .expect("donor layer write must succeed");
    }

    // -- Test 3: sliding-window publish rule ---------------------------------

    #[test]
    fn sliding_prefill_publishes_full_chunk_then_retains_window_minus_one() {
        let cfg = tiny_config(8);
        let mut cache = Gemma4KvCache::new(&cfg, 64).unwrap();
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim; // sliding width
        const PROMPT_LEN: usize = 24;

        let k: Vec<f32> = (0..PROMPT_LEN * kv_dim).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..PROMPT_LEN * kv_dim)
            .map(|i| (i as f32) + 10_000.0)
            .collect();

        // Layer 3 is a non-shared sliding layer (the sliding donor). A
        // single append -- even one far larger than the window -- publishes
        // the whole chunk: measured against the pinned reference, retention
        // to window-1 happens lazily at the *next* append, not at read time.
        cache.append_kv(3, &k, &v).unwrap();

        assert_eq!(cache.seq_len(3).unwrap(), PROMPT_LEN);
        assert_eq!(cache.k_view(3).unwrap(), k.as_slice());
        assert_eq!(cache.v_view(3).unwrap(), v.as_slice());

        // The next append (one decode token) first compacts the retained
        // tail to window-1=7 positions, then appends the new token: view =
        // last 7 of the prefill + the new token, 8 total.
        let k_next = vec![9_999.0f32; kv_dim];
        let v_next = vec![8_888.0f32; kv_dim];
        cache.append_kv(3, &k_next, &v_next).unwrap();

        assert_eq!(cache.seq_len(3).unwrap(), 8);
        let mut expected_k = k[(PROMPT_LEN - 7) * kv_dim..].to_vec();
        expected_k.extend_from_slice(&k_next);
        let mut expected_v = v[(PROMPT_LEN - 7) * kv_dim..].to_vec();
        expected_v.extend_from_slice(&v_next);
        assert_eq!(cache.k_view(3).unwrap(), expected_k.as_slice());
        assert_eq!(cache.v_view(3).unwrap(), expected_v.as_slice());
    }

    // -- Test 4: donor/shared read identity -----------------------------------

    #[test]
    fn shared_layer_view_matches_donor_after_every_append() {
        let cfg = tiny_config(8);
        let mut cache = Gemma4KvCache::new(&cfg, 64).unwrap();
        let kv_dim_sliding = cfg.num_key_value_heads * cfg.head_dim;
        let kv_dim_global = cfg.num_key_value_heads * cfg.global_head_dim;

        for step in 0..5 {
            let k_s = vec![step as f32; kv_dim_sliding];
            let v_s = vec![(step as f32) + 0.5; kv_dim_sliding];
            cache.append_kv(3, &k_s, &v_s).unwrap(); // sliding donor
            assert_eq!(cache.k_view(5).unwrap(), cache.k_view(3).unwrap());
            assert_eq!(cache.v_view(5).unwrap(), cache.v_view(3).unwrap());

            let k_g = vec![(step as f32) * 2.0; kv_dim_global];
            let v_g = vec![(step as f32) * 2.0 + 0.5; kv_dim_global];
            cache.append_kv(1, &k_g, &v_g).unwrap(); // global donor
            assert_eq!(cache.k_view(4).unwrap(), cache.k_view(1).unwrap());
            assert_eq!(cache.v_view(4).unwrap(), cache.v_view(1).unwrap());
        }
    }

    // -- Test 5: chunk-boundary -------------------------------------------

    #[test]
    fn chunk_boundary_publishes_retained_tail_plus_whole_chunk() {
        // window=4 (retention bound = window-1=3). chunk1=5 tokens: the
        // donor's first-ever append always publishes the whole chunk
        // regardless of window size. chunk2=3 tokens then genuinely crosses
        // the window boundary on the *next* append.
        let cfg = tiny_config(4);
        let mut cache = Gemma4KvCache::new(&cfg, 64).unwrap();
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let k1: Vec<f32> = (0..5 * kv_dim).map(|i| i as f32).collect();
        let v1: Vec<f32> = (0..5 * kv_dim).map(|i| (i as f32) + 500.0).collect();
        cache.append_kv(3, &k1, &v1).unwrap();
        assert_eq!(cache.seq_len(3).unwrap(), 5);
        assert_eq!(cache.k_view(3).unwrap(), k1.as_slice());
        assert_eq!(cache.v_view(3).unwrap(), v1.as_slice());
        // Shared layer 5 already sees the whole chunk 1 via its donor.
        assert_eq!(cache.k_view(5).unwrap(), k1.as_slice());
        assert_eq!(cache.v_view(5).unwrap(), v1.as_slice());

        let k2: Vec<f32> = (1000..1000 + 3 * kv_dim).map(|i| i as f32).collect();
        let v2: Vec<f32> = (2000..2000 + 3 * kv_dim).map(|i| i as f32).collect();
        cache.append_kv(3, &k2, &v2).unwrap();

        // Retain the last (window-1)=3 positions of chunk1, then publish the
        // whole 3-token chunk2: 3 + 3 = 6 positions total.
        let mut expected_k = k1[2 * kv_dim..].to_vec();
        expected_k.extend_from_slice(&k2);
        let mut expected_v = v1[2 * kv_dim..].to_vec();
        expected_v.extend_from_slice(&v2);
        assert_eq!(expected_k.len(), 6 * kv_dim);

        assert_eq!(cache.seq_len(3).unwrap(), 6);
        assert_eq!(cache.k_view(3).unwrap(), expected_k.as_slice());
        assert_eq!(cache.v_view(3).unwrap(), expected_v.as_slice());
        // Shared layer's view after chunk 2 includes chunk 2's data.
        assert_eq!(
            cache.k_view(5).unwrap(),
            expected_k.as_slice(),
            "shared layer must see donor's chunk-2 append, not a stale chunk-1-only view"
        );
        assert_eq!(cache.v_view(5).unwrap(), expected_v.as_slice());

        // Repeated one-token decodes settle at exactly window (4) positions.
        for step in 0..3usize {
            let kd = vec![(9_000 + step) as f32; kv_dim];
            let vd = vec![(9_500 + step) as f32; kv_dim];
            cache.append_kv(3, &kd, &vd).unwrap();
            assert_eq!(cache.seq_len(3).unwrap(), 4);
            assert_eq!(cache.k_view(3).unwrap().len(), 4 * kv_dim);
            assert_eq!(cache.v_view(3).unwrap().len(), 4 * kv_dim);
        }
    }

    // -- Test 6: build-time assert negatives ---------------------------------

    #[test]
    fn shared_layer_with_no_same_type_donor_errors() {
        // Non-shared prefix has only sliding layers; a shared layer is
        // global -> no valid same-type donor exists.
        let mut cfg = tiny_config(8);
        cfg.layer_types = vec![
            Gemma4LayerType::SlidingAttention, // 0
            Gemma4LayerType::SlidingAttention, // 1
            Gemma4LayerType::SlidingAttention, // 2
            Gemma4LayerType::SlidingAttention, // 3
            Gemma4LayerType::FullAttention,    // 4 (shared, global: no donor)
            Gemma4LayerType::SlidingAttention, // 5 (shared, sliding: fine)
        ];
        let err = Gemma4KvCache::new(&cfg, 64)
            .expect_err("a shared global layer with no non-shared global donor must error");
        assert!(
            err.to_string().contains("global") && err.to_string().contains("donor"),
            "error must describe the missing global donor: {err}"
        );
    }

    #[test]
    fn all_layers_shared_errors_no_donor_available() {
        // num_kv_shared_layers == num_hidden_layers -> zero non-shared
        // layers -> zero slots, but every layer still needs a donor of its
        // type, which cannot exist. This is the degenerate slot-count-zero
        // case: it must fail closed rather than silently produce a cache
        // with unreadable shared layers.
        let mut cfg = tiny_config(8);
        cfg.num_kv_shared_layers = cfg.num_hidden_layers;
        let err = Gemma4KvCache::new(&cfg, 64)
            .expect_err("an all-shared config (zero non-shared donors) must error");
        assert!(
            err.to_string().contains("donor"),
            "error must describe the missing donor: {err}"
        );
    }

    // -- Test 6b: fail-closed structural validation --------------------------

    #[test]
    fn undersized_layer_types_errors() {
        // 5 entries for num_hidden_layers=6: without an explicit length
        // check, `is_global_layer` treats the missing trailing entry as
        // sliding rather than rejecting the malformed config.
        let mut cfg = tiny_config(8);
        cfg.layer_types.pop();
        let err = Gemma4KvCache::new(&cfg, 64)
            .expect_err("layer_types shorter than num_hidden_layers must be a hard error");
        assert!(
            err.to_string().contains("layer_types"),
            "error must name layer_types: {err}"
        );
    }

    #[test]
    fn overflowing_kv_dim_errors() {
        let mut cfg = tiny_config(8);
        cfg.num_key_value_heads = usize::MAX;
        let err = Gemma4KvCache::new(&cfg, 64).expect_err(
            "num_key_value_heads * attn_head_dim overflowing usize must be a hard error",
        );
        assert!(
            err.to_string().contains("overflow"),
            "error must describe the overflow: {err}"
        );
    }

    fn assert_zeroed_field_errors(cfg: Gemma4Config, field: &str) {
        let err = Gemma4KvCache::new(&cfg, 64)
            .err()
            .unwrap_or_else(|| panic!("zeroed {field} must be a hard error"));
        assert!(
            err.to_string().contains(field),
            "error for zeroed {field} must mention it: {err}"
        );
    }

    #[test]
    fn zero_dimension_fields_error_before_allocation() {
        let mut cfg = tiny_config(8);
        cfg.num_key_value_heads = 0;
        assert_zeroed_field_errors(cfg, "num_key_value_heads");

        let mut cfg = tiny_config(8);
        cfg.head_dim = 0;
        assert_zeroed_field_errors(cfg, "head_dim");

        let mut cfg = tiny_config(8);
        cfg.global_head_dim = 0;
        assert_zeroed_field_errors(cfg, "global_head_dim");

        let mut cfg = tiny_config(8);
        cfg.sliding_window = 0;
        assert_zeroed_field_errors(cfg, "sliding_window");
    }

    #[test]
    fn zero_max_seq_len_errors() {
        let cfg = tiny_config(8);
        let err = Gemma4KvCache::new(&cfg, 0).expect_err("max_seq_len=0 must be a hard error");
        assert!(
            err.to_string().contains("max_seq_len"),
            "error must name max_seq_len: {err}"
        );
    }

    // -- Test 7: heterogeneous width -----------------------------------------

    #[test]
    fn heterogeneous_slot_width_uses_global_head_dim_for_global_slots() {
        let cfg = tiny_config(8);
        assert_ne!(cfg.head_dim, cfg.global_head_dim);
        let mut cache = Gemma4KvCache::new(&cfg, 64).unwrap();

        let kv_dim_sliding = cfg.num_key_value_heads * cfg.head_dim; // 1*4=4
        let kv_dim_global = cfg.num_key_value_heads * cfg.global_head_dim; // 1*8=8
        assert_ne!(kv_dim_sliding, kv_dim_global);

        cache
            .append_kv(3, &vec![1.0; kv_dim_sliding], &vec![1.0; kv_dim_sliding])
            .unwrap();
        cache
            .append_kv(1, &vec![1.0; kv_dim_global], &vec![1.0; kv_dim_global])
            .unwrap();

        assert_eq!(cache.k_view(3).unwrap().len(), kv_dim_sliding);
        assert_eq!(cache.k_view(1).unwrap().len(), kv_dim_global);
        // Shared layers inherit their donor's width.
        assert_eq!(cache.k_view(5).unwrap().len(), kv_dim_sliding);
        assert_eq!(cache.k_view(4).unwrap().len(), kv_dim_global);
    }

    // -- Misc boundary coverage ----------------------------------------------

    #[test]
    fn zero_num_hidden_layers_errors() {
        let mut cfg = tiny_config(8);
        cfg.num_hidden_layers = 0;
        cfg.layer_types.clear();
        cfg.num_kv_shared_layers = 0;
        let err =
            Gemma4KvCache::new(&cfg, 64).expect_err("zero num_hidden_layers must be a hard error");
        assert!(err.to_string().contains("num_hidden_layers"));
    }

    #[test]
    fn global_slot_capacity_exceeded_errors() {
        let cfg = tiny_config(8);
        let mut cache = Gemma4KvCache::new(&cfg, 4).unwrap(); // max_seq_len=4
        let kv_dim_global = cfg.num_key_value_heads * cfg.global_head_dim;
        let over = vec![1.0f32; 5 * kv_dim_global];
        let err = cache
            .append_kv(1, &over, &over)
            .expect_err("appending beyond a global slot's max_seq_len must error");
        assert!(err.to_string().contains("capacity"));
    }

    #[test]
    fn out_of_bounds_layer_index_errors_on_every_accessor() {
        let cfg = tiny_config(8);
        let mut cache = Gemma4KvCache::new(&cfg, 64).unwrap();
        assert!(cache.layer_slot(6).is_err());
        assert!(cache.is_kv_shared_layer(6).is_err());
        assert!(cache.seq_len(6).is_err());
        assert!(cache.k_view(6).is_err());
        assert!(cache.v_view(6).is_err());
        assert!(cache.append_kv(6, &[1.0], &[1.0]).is_err());
    }
}
