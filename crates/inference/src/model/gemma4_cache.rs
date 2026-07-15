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
//! Donor slots are ordinary slots, sized per their own attention type:
//! sliding-type slots retain only the last `sliding_window` positions
//! (window-truncated, matching the reference `Cache` a sliding donor reads
//! through); global-type slots retain the full context. Shared layers read
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
    /// Window-truncated retention: only the last `capacity` positions survive.
    Sliding,
    /// Full-context retention up to `capacity` positions.
    Global,
}

#[derive(Debug)]
struct Slot {
    kind: SlotKind,
    /// Per-position width: `num_key_value_heads * attn_head_dim(owner_layer)`.
    kv_dim: usize,
    /// Sliding: `sliding_window`. Global: the cache's `max_seq_len`.
    capacity: usize,
    /// `[capacity * kv_dim]`; valid data occupies `[0, seq_len * kv_dim)`.
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
    /// Global slots reject an append that would exceed `capacity`. Sliding
    /// slots implement window-truncated retention: once the slot would hold
    /// more than `capacity` positions, the oldest positions are shifted out
    /// (or, when the incoming chunk alone is >= `capacity`, only its own
    /// tail survives) so that a read always sees exactly the most recent
    /// `min(total_appended, capacity)` positions in order.
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
                if q_len >= self.capacity {
                    // The incoming chunk alone fills (or overflows) the
                    // window: only its own last `capacity` tokens survive,
                    // discarding all prior state.
                    let start_tok = q_len - self.capacity;
                    let k_tail = &k[start_tok * self.kv_dim..];
                    let v_tail = &v[start_tok * self.kv_dim..];
                    self.k[..k_tail.len()].copy_from_slice(k_tail);
                    self.v[..v_tail.len()].copy_from_slice(v_tail);
                    self.seq_len = self.capacity;
                } else {
                    let total_new = self.seq_len + q_len;
                    if total_new > self.capacity {
                        let drop = total_new - self.capacity;
                        let valid_end = self.seq_len * self.kv_dim;
                        self.k.copy_within((drop * self.kv_dim)..valid_end, 0);
                        self.v.copy_within((drop * self.kv_dim)..valid_end, 0);
                        self.seq_len -= drop;
                    }
                    let off = self.seq_len * self.kv_dim;
                    self.k[off..off + k.len()].copy_from_slice(k);
                    self.v[off..off + v.len()].copy_from_slice(v);
                    self.seq_len += q_len;
                }
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
    /// - any shared layer has no non-shared donor layer of the same
    ///   attention type among the non-shared layers (a config/checkpoint
    ///   structural disagreement -- there is no valid donor to fall back
    ///   to, so this is rejected rather than silently paired cross-type);
    /// - the derived slot count does not equal the number of non-shared
    ///   layers, or any layer's resolved donor disagrees on attention type
    ///   (both are internal build-time asserts: they should be unreachable
    ///   given the construction above, but are checked explicitly per the
    ///   "hard error, not debug_assert" contract);
    /// - `cfg.sliding_window` is `0`, `max_seq_len` is `0`, or a slot's
    ///   `capacity * kv_dim` product overflows `usize`.
    pub fn new(cfg: &Gemma4Config, max_seq_len: usize) -> Result<Self, InferenceError> {
        let n = cfg.num_hidden_layers;
        if n == 0 {
            return Err(InferenceError::InvalidInput(
                "gemma4 kv cache: num_hidden_layers must be > 0".to_string(),
            ));
        }
        let first_shared = cfg.first_kv_shared_layer_idx();

        // Non-shared layers 0..first_shared each get their own slot, in
        // order, so slot index == layer index for every non-shared layer.
        let mut layer_kv_slot = vec![0usize; n];
        for (layer, slot) in layer_kv_slot.iter_mut().enumerate().take(first_shared) {
            *slot = layer;
        }

        // Build-time assert: slot count == non-shared layer count. Derived
        // independently from `is_kv_shared_layer` (rather than reusing
        // `first_shared` on both sides) so this assert is not tautological.
        let actual_non_shared = (0..n).filter(|&l| !cfg.is_kv_shared_layer(l)).count();
        if first_shared != actual_non_shared {
            return Err(InferenceError::Inference(format!(
                "gemma4 kv cache: build-time slot-count assert failed: derived slot count \
                 {first_shared} != non-shared layer count {actual_non_shared} \
                 (num_hidden_layers={n})"
            )));
        }

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
            let kv_dim = cfg.num_key_value_heads * cfg.attn_head_dim(owner_layer);
            let (kind, capacity) = if cfg.is_global_layer(owner_layer) {
                (SlotKind::Global, max_seq_len)
            } else {
                (SlotKind::Sliding, cfg.sliding_window)
            };
            slots.push(Slot::try_new(kind, kv_dim, capacity)?);
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

    // -- Test 3: sliding window truncation -----------------------------------

    #[test]
    fn sliding_window_truncation_keeps_last_window_in_order() {
        let cfg = tiny_config(8);
        let mut cache = Gemma4KvCache::new(&cfg, 64).unwrap();
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim; // sliding width
        const PROMPT_LEN: usize = 24;

        let k: Vec<f32> = (0..PROMPT_LEN * kv_dim).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..PROMPT_LEN * kv_dim)
            .map(|i| (i as f32) + 1000.0)
            .collect();

        // Layer 3 is a non-shared sliding layer (the sliding donor).
        cache.append_kv(3, &k, &v).unwrap();

        assert_eq!(cache.seq_len(3).unwrap(), 8);
        let k_view = cache.k_view(3).unwrap();
        assert_eq!(k_view.len(), 8 * kv_dim);

        let expected: Vec<f32> = ((PROMPT_LEN - 8) * kv_dim..PROMPT_LEN * kv_dim)
            .map(|i| i as f32)
            .collect();
        assert_eq!(k_view, expected.as_slice());
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
    fn chunk_boundary_shared_layer_sees_current_chunk() {
        // window=6, chunks of 5 then 3 tokens: total=8 > 6, genuinely crosses
        // the window boundary (drops 2 of the first chunk's tokens).
        let cfg = tiny_config(6);
        let mut cache = Gemma4KvCache::new(&cfg, 64).unwrap();
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let chunk1: Vec<f32> = (0..5 * kv_dim).map(|i| i as f32).collect();
        cache.append_kv(3, &chunk1, &chunk1).unwrap();
        assert_eq!(cache.seq_len(3).unwrap(), 5);
        // Shared layer 5 already sees chunk 1 via its donor (layer 3).
        assert_eq!(cache.k_view(5).unwrap(), chunk1.as_slice());

        let chunk2: Vec<f32> = (100..100 + 3 * kv_dim).map(|i| i as f32).collect();
        cache.append_kv(3, &chunk2, &chunk2).unwrap();
        assert_eq!(cache.seq_len(3).unwrap(), 6);

        // total appended = 8 tokens, capacity = 6 -> drop the oldest 2 of
        // chunk1's 5 tokens, keeping its last 3 + all 3 of chunk2.
        let mut expected = chunk1[2 * kv_dim..].to_vec();
        expected.extend_from_slice(&chunk2);
        assert_eq!(expected.len(), 6 * kv_dim);

        assert_eq!(cache.k_view(3).unwrap(), expected.as_slice());
        // Shared layer's view after chunk 2 includes chunk 2's data.
        assert_eq!(
            cache.k_view(5).unwrap(),
            expected.as_slice(),
            "shared layer must see donor's chunk-2 append, not a stale chunk-1-only view"
        );
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
